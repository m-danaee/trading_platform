
from __future__ import annotations

import json
import logging
import math
import os
import shutil
from itertools import combinations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.rb_evaluator_v5 import EvaluatorV5BacktestEngine as CPUBacktestEngine, _build_rule_signal_mask
from gpu_fuzzy_trader.backtest.df_slim import downcast_numeric_df
from gpu_fuzzy_trader.data.loader import Data_Loader
from gpu_fuzzy_trader.scoring import return_to_drawdown, profit_factor_term

logger = logging.getLogger(__name__)


@dataclass
class CandidateRecord:
    rule: dict
    train_metrics: dict
    valid_metrics: dict
    score: float
    mask: np.ndarray | None = None


def _f(metrics: dict | None, key: str, default: float = 0.0) -> float:
    try:
        value = float((metrics or {}).get(key, default))
    except Exception:
        value = default
    return value if math.isfinite(value) else default


def _i(metrics: dict | None, key: str, default: int = 0) -> int:
    try:
        return int((metrics or {}).get(key, default))
    except Exception:
        return default


def _train_valid_shape(train_ret: float, valid_ret: float) -> tuple[bool, float, float]:
    """Return (ok, bonus, penalty) for the desired train≈slightly-above-test shape.

    In rb/governor mode the user wants train return to be a little above
    test return, not below it and not massively higher.  This helper makes that
    preference explicit in both filtering and scoring.
    """
    if not bool(getattr(_cfg, "RB_REQUIRE_TRAIN_SLIGHTLY_ABOVE_TEST", False)):
        return True, 0.0, 0.0
    if valid_ret <= 0.0 or train_ret <= 0.0:
        return False, 0.0, 0.0

    min_ratio = float(getattr(_cfg, "RB_TRAIN_VALID_MIN_RATIO", 1.03))
    max_ratio = float(getattr(_cfg, "RB_TRAIN_VALID_MAX_RATIO", 1.35))
    min_abs_gap = float(getattr(_cfg, "RB_TRAIN_VALID_MIN_ABS_GAP", 0.20))
    max_abs_gap = float(getattr(_cfg, "RB_TRAIN_VALID_MAX_ABS_GAP", 12.0))
    below_weight = float(getattr(_cfg, "RB_TRAIN_BELOW_TEST_PENALTY", 900.0))
    high_weight = float(getattr(_cfg, "RB_TRAIN_TOO_HIGH_PENALTY", 220.0))
    bonus_weight = float(getattr(_cfg, "RB_TRAIN_VALID_SHAPE_BONUS", 160.0))

    ratio = train_ret / max(valid_ret, 1e-9)
    gap = train_ret - valid_ret

    lower_ratio_target = valid_ret * min_ratio
    lower_abs_target = valid_ret + min_abs_gap
    min_train_target = max(lower_ratio_target, lower_abs_target)

    upper_ratio_target = valid_ret * max_ratio
    upper_abs_target = valid_ret + max_abs_gap
    max_train_target = min(upper_ratio_target, upper_abs_target)
    if max_train_target < min_train_target:
        max_train_target = min_train_target + max(0.5, min_abs_gap)

    penalty = 0.0
    if train_ret < min_train_target:
        miss = min_train_target - train_ret
        penalty += below_weight * (miss / max(abs(valid_ret), 1.0))
    if train_ret > max_train_target:
        excess = train_ret - max_train_target
        penalty += high_weight * (excess / max(abs(valid_ret), 1.0))

    ok = train_ret >= min_train_target and train_ret <= max_train_target
    if ok:
        mid = 0.5 * (min_train_target + max_train_target)
        half = max(1e-9, 0.5 * (max_train_target - min_train_target))
        closeness = max(0.0, 1.0 - abs(train_ret - mid) / half)
        bonus = bonus_weight * closeness
    else:
        bonus = 0.0
    return ok, float(bonus), float(penalty)


def _evaluator_health_penalty(metrics: dict, *, role: str = "valid") -> float:
    """Penalize evaluator_v5 execution problems: too many skipped signals, low executed/raw ratio, too many open positions."""
    raw = max(0, _i(metrics, "raw_signal_count", 0))
    executed = max(0, _i(metrics, "executed_trades", 0))
    skipped = max(0, _i(metrics, "skipped_min_notional_count", 0))
    max_pos = max(0, _i(metrics, "max_simultaneous_positions", 0))

    max_skip = float(getattr(_cfg, "RB_MAX_SKIPPED_SIGNAL_RATIO", 0.20))
    min_exec = float(getattr(_cfg, "RB_MIN_EXECUTED_RAW_RATIO", 0.60))
    skip_weight = float(getattr(_cfg, "RB_SKIPPED_RATIO_PENALTY", 3500.0))
    exec_weight = float(getattr(_cfg, "RB_EXECUTED_RATIO_PENALTY", 2500.0))
    pos_limit = int(getattr(_cfg, "RB_MAX_SIMULTANEOUS_POSITIONS", 10))
    pos_weight = float(getattr(_cfg, "RB_MAX_POSITIONS_PENALTY", 120.0))

    penalty = 0.0
    if raw > 0:
        skip_ratio = skipped / raw
        exec_ratio = executed / raw
        if skip_ratio > max_skip:
            penalty += (skip_ratio - max_skip) * skip_weight * (1.5 if role == "test" else 1.0)
        if exec_ratio < min_exec:
            penalty += (min_exec - exec_ratio) * exec_weight * (1.5 if role == "test" else 1.0)
    if max_pos > pos_limit:
        penalty += (max_pos - pos_limit) * pos_weight * (1.2 if role == "test" else 1.0)
    return float(penalty)


def _rule_risk_ok(rule: dict) -> bool:
    if not bool(getattr(_cfg, "RB_REQUIRE_TP_SL_ABOVE_ONE", True)):
        return True
    tp = float(rule.get("tp", getattr(_cfg, "RB_DEFAULT_TP", 2.0)))
    sl = float(rule.get("sl", getattr(_cfg, "RB_DEFAULT_SL", 1.2)))
    return tp >= float(getattr(_cfg, "RB_MIN_TP", 1.0)) and sl >= float(getattr(_cfg, "RB_MIN_SL", 1.0))


def _rule_to_engine(rule: dict) -> dict:
    tp = float(rule.get("tp", getattr(_cfg, "RB_DEFAULT_TP", 2.0)))
    sl = float(rule.get("sl", getattr(_cfg, "RB_DEFAULT_SL", 1.2)))
    if bool(getattr(_cfg, "RB_REQUIRE_TP_SL_ABOVE_ONE", True)):
        tp = max(tp, float(getattr(_cfg, "RB_MIN_TP", 1.01)))
        sl = max(sl, float(getattr(_cfg, "RB_MIN_SL", 1.01)))
    return {
        "conditions": list(rule.get("conditions", [])),
        "tp": tp,
        "sl": sl,
        "capital_pct": float(rule.get("capital_pct", getattr(_cfg, "RB_DEFAULT_CAPITAL_PCT", 12.5))),
    }


def _strategy(direction: str, rules: list[dict], *, risk_optimized: bool = False, extra: dict | None = None) -> dict:
    clean_rules = [_rule_to_engine(r) for r in rules]
    out = {
        "direction": direction,
        "rules_set": clean_rules,
        "risk_optimized": bool(risk_optimized),
    }
    if extra:
        out.update(extra)
    return out


def _score_metrics(train_m: dict, valid_m: dict, *, min_train_trades: int | None = None, min_valid_trades: int | None = None) -> float:
    """Dominant objective: return/DD with train return slightly above test return."""
    min_train_trades = int(min_train_trades if min_train_trades is not None else getattr(_cfg, "RB_MIN_TRAIN_TRADES", 25))
    min_valid_trades = int(min_valid_trades if min_valid_trades is not None else getattr(_cfg, "RB_MIN_VALID_TRADES", 15))
    dd_floor = float(getattr(_cfg, "RB_RETURN_DD_FLOOR", 0.50))

    train_ret = _f(train_m, "total_return_pct")
    train_dd = _f(train_m, "max_drawdown_pct", 100.0)
    train_pf = _f(train_m, "profit_factor", 0.0)
    train_wr = _f(train_m, "win_rate", 0.0)
    train_trades = _i(train_m, "executed_trades", 0)

    valid_ret = _f(valid_m, "total_return_pct")
    valid_dd = _f(valid_m, "max_drawdown_pct", 100.0)
    valid_pf = _f(valid_m, "profit_factor", 0.0)
    valid_wr = _f(valid_m, "win_rate", 0.0)
    valid_trades = _i(valid_m, "executed_trades", 0)

    train_ratio = return_to_drawdown(train_ret, train_dd, dd_floor)
    valid_ratio = return_to_drawdown(valid_ret, valid_dd, dd_floor)
    shape_ok, shape_bonus, shape_penalty = _train_valid_shape(train_ret, valid_ret)

    score = (
        120.0 * valid_ratio
        + 45.0 * train_ratio
        + 4.5 * valid_ret
        + 1.2 * train_ret
        + 14.0 * profit_factor_term(valid_pf, 5.0)
        + 5.0 * profit_factor_term(train_pf, 5.0)
        + 0.06 * valid_wr
        + 0.025 * train_wr
        - 0.25 * valid_dd
        - 0.08 * train_dd
        + shape_bonus
        - shape_penalty
    )

    score -= _evaluator_health_penalty(train_m, role="train")
    score -= _evaluator_health_penalty(valid_m, role="valid")

    if train_ret <= float(getattr(_cfg, "RB_MIN_TRAIN_RETURN", 0.0)):
        score -= 500.0 + abs(train_ret) * 20.0
    if valid_ret <= float(getattr(_cfg, "RB_MIN_VALID_RETURN", 0.0)):
        score -= 1000.0 + abs(valid_ret) * 30.0
    if train_pf < float(getattr(_cfg, "RB_MIN_TRAIN_PF", 1.0)):
        score -= (float(getattr(_cfg, "RB_MIN_TRAIN_PF", 1.0)) - train_pf) * 60.0
    if valid_pf < float(getattr(_cfg, "RB_MIN_VALID_PF", 1.0)):
        score -= (float(getattr(_cfg, "RB_MIN_VALID_PF", 1.0)) - valid_pf) * 120.0
    if train_trades < min_train_trades:
        score -= (min_train_trades - train_trades) * float(getattr(_cfg, "RB_TRADE_PENALTY", 0.80))
    if valid_trades < min_valid_trades:
        score -= (min_valid_trades - valid_trades) * float(getattr(_cfg, "RB_TRADE_PENALTY", 0.80)) * 1.5
    if bool(getattr(_cfg, "RB_REQUIRE_TRAIN_SLIGHTLY_ABOVE_TEST", False)) and not shape_ok:
        score -= 250.0

    score -= max(0.0, train_ratio - valid_ratio) * float(getattr(_cfg, "RB_TRAIN_VALID_RATIO_GAP_WEIGHT", 12.0))
    score -= max(0.0, train_ret - valid_ret) * float(getattr(_cfg, "RB_TRAIN_VALID_RETURN_GAP_WEIGHT", 0.55))
    return float(score)




def _combined_return_score(train_m: dict, valid_m: dict) -> float:
    """Profit objective for lenient rule addition, but now evaluator_v5 aware.

    A new rule can still be added mainly when it increases profit, but it must
    not create the  failure mode where most raw signals are skipped by the
    evaluator.  Therefore the score subtracts execution-health penalties.
    """
    train_ret = _f(train_m, "total_return_pct")
    valid_ret = _f(valid_m, "total_return_pct")
    score = train_ret + valid_ret
    score -= _evaluator_health_penalty(train_m, role="train") / 35.0
    score -= _evaluator_health_penalty(valid_m, role="valid") / 35.0
    return float(score)


def _positive_returns(train_m: dict, valid_m: dict) -> bool:
    if not bool(getattr(_cfg, "RB_GLOBAL_REQUIRE_POSITIVE_TRAIN_TEST", True)):
        return True
    return _f(train_m, "total_return_pct") > 0.0 and _f(valid_m, "total_return_pct") > 0.0


def _rule_key(rule: dict) -> tuple[str, ...]:
    return tuple(sorted(str(c) for c in rule.get("conditions", [])))


def _is_symbol_condition(condition: str) -> bool:
    text = str(condition).strip().lower()
    if text.startswith("symbol is "):
        return True
    if text.startswith("[symbol] is "):
        return True
    return False


def _strip_symbol_conditions(conditions: list[str]) -> list[str]:
    return [str(c) for c in conditions if not _is_symbol_condition(str(c))]


def _has_symbol_condition(rule: dict) -> bool:
    return any(_is_symbol_condition(str(c)) for c in rule.get("conditions", []))


def _symbol_condition(sym: object) -> str:
    text = str(sym).strip()
    return f"symbol is {text}"


def _available_symbols(*dfs: pd.DataFrame) -> list[str]:
    vals: list[str] = []
    seen: set[str] = set()
    for df in dfs:
        if df is None or "symbol" not in df.columns:
            continue
        for v in pd.Series(df["symbol"]).dropna().unique().tolist():
            text = str(v).strip()
            if text and text not in seen:
                seen.add(text)
                vals.append(text)
    def key(x: str):
        try:
            return (0, float(x))
        except Exception:
            return (1, x)
    return sorted(vals, key=key)


def _ensure_symbol_filtered_rule(rule: dict, symbols: list[str]) -> dict:
    """Return rule with an explicit symbol filter when required.

    This is a safety net for output files.  Scoring normally specializes rules
    before evaluation.
    """
    out = dict(rule)
    conditions = list(out.get("conditions", []))
    if not bool(getattr(_cfg, "RB_REQUIRE_SYMBOL_FILTERS", False)):
        out["conditions"] = conditions
        return out
    if any(_is_symbol_condition(str(c)) for c in conditions):
        out["conditions"] = conditions
        return out
    max_symbols = int(getattr(_cfg, "RB_SYMBOL_MAX_SYMBOLS_PER_RULE", 3))
    use_symbols = symbols[:max(1, max_symbols)] if symbols else []
    out["conditions"] = conditions + [_symbol_condition(s) for s in use_symbols]
    return out


def _symbol_specialized_variants(
    rule: dict,
    train_engine: CPUBacktestEngine,
    valid_engine: CPUBacktestEngine,
    symbols: list[str],
) -> list[dict]:
    """Build symbol-filtered variants and rank them using evaluator_v5 scoring.

    Important: this runs *before* candidate scoring, so the reported metrics are
    for the exact JSON that later goes into evaluator_v5.
    """
    base = _rule_to_engine(rule)
    if not bool(getattr(_cfg, "RB_REQUIRE_SYMBOL_FILTERS", False)):
        return [base]
    if _has_symbol_condition(base):
        return [base]
    if not symbols:
        return [base]

    base_conditions = _strip_symbol_conditions(list(base.get("conditions", [])))
    max_symbols = max(1, int(getattr(_cfg, "RB_SYMBOL_MAX_SYMBOLS_PER_RULE", 3)))
    top_single = max(1, int(getattr(_cfg, "RB_SYMBOL_TOP_SINGLE_SYMBOLS", 5)))
    max_variants = max(1, int(getattr(_cfg, "RB_SYMBOL_MAX_VARIANTS_PER_RULE", 10)))
    min_train_trades = int(getattr(_cfg, "RB_SYMBOL_MIN_TRAIN_TRADES", 10))
    min_valid_trades = int(getattr(_cfg, "RB_SYMBOL_MIN_TEST_TRADES", 6))

    scored_singles: list[tuple[float, str, dict, dict]] = []
    for sym in symbols:
        variant = dict(base)
        variant["conditions"] = base_conditions + [_symbol_condition(sym)]
        try:
            tr = train_engine.simulate_rule_set([variant])
            te = valid_engine.simulate_rule_set([variant])
        except Exception:
            continue
        if _i(tr, "executed_trades") < min_train_trades or _i(te, "executed_trades") < min_valid_trades:
            continue
        score = _score_metrics(tr, te, min_train_trades=min_train_trades, min_valid_trades=min_valid_trades)
        scored_singles.append((score, sym, tr, te))

    scored_singles.sort(key=lambda x: x[0], reverse=True)
    candidate_symbol_sets: list[tuple[str, ...]] = [(sym,) for _score, sym, _tr, _te in scored_singles[:top_single]]

    if bool(getattr(_cfg, "RB_SYMBOL_USE_COMBINATIONS", True)):
        top_syms = [sym for _score, sym, _tr, _te in scored_singles[:top_single]]
        for k in range(2, min(max_symbols, len(top_syms)) + 1):
            for combo in combinations(top_syms, k):
                candidate_symbol_sets.append(tuple(combo))

    if not candidate_symbol_sets:
        candidate_symbol_sets = [tuple(symbols[:max_symbols])]

    scored_variants: list[tuple[float, dict]] = []
    seen_sets: set[tuple[str, ...]] = set()
    for sym_set in candidate_symbol_sets:
        sym_set = tuple(dict.fromkeys(str(s) for s in sym_set))
        if not sym_set or sym_set in seen_sets:
            continue
        seen_sets.add(sym_set)
        variant = dict(base)
        variant["conditions"] = base_conditions + [_symbol_condition(s) for s in sym_set]
        try:
            tr = train_engine.simulate_rule_set([variant])
            te = valid_engine.simulate_rule_set([variant])
            score = _score_metrics(tr, te, min_train_trades=min_train_trades, min_valid_trades=min_valid_trades)
        except Exception:
            score = -1e18
        scored_variants.append((score, variant))

    scored_variants.sort(key=lambda x: x[0], reverse=True)
    return [v for _score, v in scored_variants[:max_variants]] or [_ensure_symbol_filtered_rule(base, symbols)]

def _is_positive_good(train_m: dict, valid_m: dict, *, min_train_trades: int | None = None, min_valid_trades: int | None = None) -> bool:
    min_train_trades = int(min_train_trades if min_train_trades is not None else getattr(_cfg, "RB_MIN_TRAIN_TRADES", 25))
    min_valid_trades = int(min_valid_trades if min_valid_trades is not None else getattr(_cfg, "RB_MIN_VALID_TRADES", 15))
    train_ret = _f(train_m, "total_return_pct")
    valid_ret = _f(valid_m, "total_return_pct")
    shape_ok, _, _ = _train_valid_shape(train_ret, valid_ret)
    if bool(getattr(_cfg, "RB_REQUIRE_TRAIN_SLIGHTLY_ABOVE_TEST", False)) and not shape_ok:
        return False

    def execution_ok(m: dict) -> bool:
        raw = max(0, _i(m, "raw_signal_count", 0))
        if raw <= 0:
            return False
        skipped = max(0, _i(m, "skipped_min_notional_count", 0))
        executed = max(0, _i(m, "executed_trades", 0))
        max_skip = float(getattr(_cfg, "RB_MAX_SKIPPED_SIGNAL_RATIO", 0.20))
        min_exec = float(getattr(_cfg, "RB_MIN_EXECUTED_RAW_RATIO", 0.60))
        return (skipped / raw) <= max_skip and (executed / raw) >= min_exec

    return (
        train_ret > float(getattr(_cfg, "RB_MIN_TRAIN_RETURN", 0.0))
        and valid_ret > float(getattr(_cfg, "RB_MIN_VALID_RETURN", 0.0))
        and _f(train_m, "profit_factor") >= float(getattr(_cfg, "RB_MIN_TRAIN_PF", 1.0))
        and _f(valid_m, "profit_factor") >= float(getattr(_cfg, "RB_MIN_VALID_PF", 1.0))
        and _i(train_m, "executed_trades") >= min_train_trades
        and _i(valid_m, "executed_trades") >= min_valid_trades
        and execution_ok(train_m)
        and execution_ok(valid_m)
    )


def _combined_train_df(train_df: pd.DataFrame, val_df: pd.DataFrame | None) -> pd.DataFrame:
    if val_df is None or len(val_df) == 0:
        return train_df
    return downcast_numeric_df(pd.concat([train_df, val_df], ignore_index=True, sort=False))


def _prepare_scoring_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a scoring frame with evaluator ordering."""
    out = df.copy()
    if "datetime" in out.columns:
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce", utc=True)
        out["datetime"] = out["datetime"].dt.tz_localize(None)
    if "datetime" in out.columns and "symbol" in out.columns:
        out = out.sort_values(["datetime", "symbol"]).reset_index(drop=True)
    elif "datetime" in out.columns:
        out = out.sort_values("datetime").reset_index(drop=True)
    elif "symbol" in out.columns:
        out = out.sort_values("symbol").reset_index(drop=True)
    labels = ["label_open_next", "label_close_288", "label_min_288", "label_max_288", "label_max_before_min"]
    present = [c for c in labels if c in out.columns]
    out = out.dropna(subset=present).reset_index(drop=True)
    non_features = set(labels) | {"datetime", "symbol", "dataset_type", "_symbol_bar_index"}
    feature_cols = [c for c in out.columns if c not in non_features]
    out[feature_cols] = out[feature_cols].fillna(0)
    if "symbol" in out.columns:
        out["_symbol_bar_index"] = out.groupby("symbol").cumcount()
    else:
        out["_symbol_bar_index"] = np.arange(len(out))
    return downcast_numeric_df(out)


def _load_scoring_frames(train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return prepared training and validation frames."""
    return _prepare_scoring_frame(train_df), _prepare_scoring_frame(val_df)


def _mask_for(rule: dict, train_like_df: pd.DataFrame, valid_df: pd.DataFrame) -> np.ndarray:
    try:
        m1 = _build_rule_signal_mask(train_like_df, list(rule.get("conditions", [])))
        m2 = _build_rule_signal_mask(valid_df, list(rule.get("conditions", [])))
        return np.concatenate([m1.astype(bool, copy=False), m2.astype(bool, copy=False)])
    except Exception:
        return np.zeros(len(train_like_df) + len(valid_df), dtype=bool)


def _pair_overlap(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return 0.0
    ca = int(np.sum(a)); cb = int(np.sum(b))
    if ca == 0 or cb == 0:
        return 0.0
    inter = int(np.sum(a & b))
    return inter / max(1, min(ca, cb))


def _max_overlap(candidate: CandidateRecord, selected: list[CandidateRecord]) -> float:
    if not selected:
        return 0.0
    return max(_pair_overlap(candidate.mask, s.mask) for s in selected)


def _evaluate_ruleset(train_engine: CPUBacktestEngine, valid_engine: CPUBacktestEngine, rules: list[dict]) -> tuple[dict, dict, float]:
    fmt = [_rule_to_engine(r) for r in rules]
    train_m = train_engine.simulate_rule_set(fmt)
    valid_m = valid_engine.simulate_rule_set(fmt)
    score = _score_metrics(
        train_m,
        valid_m,
        min_train_trades=int(getattr(_cfg, "RB_RULESET_MIN_TRAIN_TRADES", getattr(_cfg, "RB_MIN_TRAIN_TRADES", 25))),
        min_valid_trades=int(getattr(_cfg, "RB_RULESET_MIN_VALID_TRADES", getattr(_cfg, "RB_MIN_VALID_TRADES", 15))),
    )
    return train_m, valid_m, score


def _filter_good_rules(
    pool: list[dict],
    train_like_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    direction: str,
) -> list[CandidateRecord]:
    train_engine = CPUBacktestEngine(train_like_df, {}, direction)
    valid_engine = CPUBacktestEngine(valid_df, {}, direction)
    records: list[CandidateRecord] = []
    seen: set[tuple[str, ...]] = set()
    limit = int(getattr(_cfg, "RB_MAX_POOL_RULES_TO_EVALUATE", 700))
    symbols = _available_symbols(train_like_df, valid_df)

    for raw in pool[:limit]:
        for rule in _symbol_specialized_variants(raw, train_engine, valid_engine, symbols):
            rule = _rule_to_engine(rule)
            key = tuple(sorted(str(c) for c in rule.get("conditions", [])))
            if key in seen:
                continue
            seen.add(key)
            try:
                train_m = train_engine.simulate_rule_set([rule])
                valid_m = valid_engine.simulate_rule_set([rule])
            except Exception:
                continue
            if not _is_positive_good(train_m, valid_m):
                continue
            score = _score_metrics(train_m, valid_m)
            rec = CandidateRecord(rule=rule, train_metrics=train_m, valid_metrics=valid_m, score=score)
            rec.mask = _mask_for(rule, train_like_df, valid_df)
            records.append(rec)

    records.sort(key=lambda r: r.score, reverse=True)
    keep = int(getattr(_cfg, "RB_KEEP_TOP_RULES", 120))
    logger.info(
        "RB [%s]: kept %d/%d single rules positive on training and validation.",
        direction, min(len(records), keep), len(seen),
    )
    return records[:keep]


def _compose_ruleset(
    candidates: list[CandidateRecord],
    train_engine: CPUBacktestEngine,
    valid_engine: CPUBacktestEngine,
    direction: str,
) -> tuple[list[CandidateRecord], dict, dict, float, list[dict]]:
    if not candidates:
        raise ValueError(f"No rb-positive rules available for {direction}")

    selected: list[CandidateRecord] = [candidates[0]]
    cur_train, cur_test, cur_score = _evaluate_ruleset(train_engine, valid_engine, [selected[0].rule])
    history = [{
        "step": 1,
        "action": "seed",
        "score": cur_score,
        "train_return_pct": _f(cur_train, "total_return_pct"),
        "valid_return_pct": _f(cur_test, "total_return_pct"),
        "rules": 1,
    }]

    max_rules = int(getattr(_cfg, "RB_MAX_RULES", getattr(_cfg, "PHASE3_MAX_RULES", 5)))
    max_overlap = float(getattr(_cfg, "RB_MAX_PAIR_OVERLAP", 0.22))
    min_score_improve = float(getattr(_cfg, "RB_MIN_SCORE_IMPROVEMENT", 0.05))
    min_train_ret_improve = float(getattr(_cfg, "RB_MIN_TRAIN_RETURN_IMPROVEMENT", 0.01))
    min_valid_ret_improve = float(getattr(_cfg, "RB_MIN_VALID_RETURN_IMPROVEMENT", 0.01))
    require_subset_improve = bool(getattr(_cfg, "RB_RULESET_MUST_BEAT_SUBSETS", True))
    return_only_add = bool(getattr(_cfg, "RB_RULE_ADD_BY_RETURN_ONLY", False))
    ignore_overlap = bool(getattr(_cfg, "RB_RULE_ADD_IGNORE_OVERLAP", False))
    if return_only_add:
        require_subset_improve = False if bool(getattr(_cfg, "RB_RULE_ADD_IGNORE_SUBSET_BEAT", True)) else require_subset_improve
        cur_return_score = _combined_return_score(cur_train, cur_test)
        min_return_improve = float(getattr(_cfg, "RB_MIN_COMBINED_RETURN_IMPROVEMENT", 0.05))

    used = {_rule_key(r.rule) for r in selected}
    while len(selected) < max_rules:
        best: tuple[float, CandidateRecord, dict, dict] | None = None
        for cand in candidates:
            key = _rule_key(cand.rule)
            if key in used:
                continue
            ov = _max_overlap(cand, selected)
            if (not ignore_overlap) and ov > max_overlap:
                continue
            trial_recs = selected + [cand]
            trial_rules = [r.rule for r in trial_recs]
            train_m, valid_m, score = _evaluate_ruleset(train_engine, valid_engine, trial_rules)
            if return_only_add:
                if not _positive_returns(train_m, valid_m):
                    continue
                ret_score = _combined_return_score(train_m, valid_m)
                if ret_score <= cur_return_score + min_return_improve:
                    continue
                choose_score = ret_score
            else:
                if not _is_positive_good(
                    train_m,
                    valid_m,
                    min_train_trades=int(getattr(_cfg, "RB_RULESET_MIN_TRAIN_TRADES", getattr(_cfg, "RB_MIN_TRAIN_TRADES", 25))),
                    min_valid_trades=int(getattr(_cfg, "RB_RULESET_MIN_VALID_TRADES", getattr(_cfg, "RB_MIN_VALID_TRADES", 15))),
                ):
                    continue
                if require_subset_improve:
                    if _f(train_m, "total_return_pct") <= max(_f(cur_train, "total_return_pct"), _f(cand.train_metrics, "total_return_pct")) + min_train_ret_improve:
                        continue
                    if _f(valid_m, "total_return_pct") <= max(_f(cur_test, "total_return_pct"), _f(cand.valid_metrics, "total_return_pct")) + min_valid_ret_improve:
                        continue
                if score <= cur_score + min_score_improve:
                    continue
                choose_score = score
            if best is None or choose_score > best[0]:
                best = (choose_score, cand, train_m, valid_m)

        if best is None:
            logger.info("RB [%s]: no further positive/improving low-overlap extension found at %d rules.", direction, len(selected))
            break

        chosen_score, chosen, cur_train, cur_test = best
        cur_score = _score_metrics(
            cur_train,
            cur_test,
            min_train_trades=int(getattr(_cfg, "RB_RULESET_MIN_TRAIN_TRADES", getattr(_cfg, "RB_MIN_TRAIN_TRADES", 25))),
            min_valid_trades=int(getattr(_cfg, "RB_RULESET_MIN_VALID_TRADES", getattr(_cfg, "RB_MIN_VALID_TRADES", 15))),
        )
        if return_only_add:
            cur_return_score = _combined_return_score(cur_train, cur_test)
        selected.append(chosen)
        used.add(_rule_key(chosen.rule))
        history.append({
            "step": len(selected),
            "action": "add_rule",
            "score": cur_score,
            "train_return_pct": _f(cur_train, "total_return_pct"),
            "valid_return_pct": _f(cur_test, "total_return_pct"),
            "train_pf": _f(cur_train, "profit_factor"),
            "valid_pf": _f(cur_test, "profit_factor"),
            "train_dd": _f(cur_train, "max_drawdown_pct"),
            "valid_dd": _f(cur_test, "max_drawdown_pct"),
            "rules": len(selected),
            "combined_return_score": _combined_return_score(cur_train, cur_test),
        })
        logger.info(
            "RB [%s]: grew to %d rules | score=%.2f train_ret=%.2f%% valid_ret=%.2f%%",
            direction, len(selected), cur_score, _f(cur_train, "total_return_pct"), _f(cur_test, "total_return_pct"),
        )

    return selected, cur_train, cur_test, cur_score, history


def _optimize_risk(
    selected: list[CandidateRecord],
    train_engine: CPUBacktestEngine,
    valid_engine: CPUBacktestEngine,
    direction: str,
) -> tuple[list[dict], dict, dict, float, list[dict]]:
    rules = [_rule_to_engine(r.rule) for r in selected]
    cur_train, cur_test, cur_score = _evaluate_ruleset(train_engine, valid_engine, rules)
    best_rules = [dict(r) for r in rules]
    hist: list[dict] = [{
        "pass": 0,
        "rule_index": -1,
        "score": cur_score,
        "train_return_pct": _f(cur_train, "total_return_pct"),
        "valid_return_pct": _f(cur_test, "total_return_pct"),
        "train_pf": _f(cur_train, "profit_factor"),
        "valid_pf": _f(cur_test, "profit_factor"),
    }]

    tp_grid = tuple(float(x) for x in getattr(_cfg, "RB_TP_GRID", getattr(_cfg, "PHASE4_TP_GRID", (1.5, 2.0, 2.5, 3.0))))
    sl_grid = tuple(float(x) for x in getattr(_cfg, "RB_SL_GRID", getattr(_cfg, "PHASE4_SL_GRID", (0.8, 1.0, 1.2, 1.5))))
    cap_grid = tuple(float(x) for x in getattr(_cfg, "RB_CAPITAL_GRID", getattr(_cfg, "PHASE4_CAPITAL_GRID", (5.0, 7.5, 10.0, 12.5))))
    max_total_cap = float(getattr(_cfg, "RB_MAX_TOTAL_CAPITAL", 65.0))
    passes = int(getattr(_cfg, "RB_RISK_OPT_PASSES", 2))
    min_improve = float(getattr(_cfg, "RB_RISK_MIN_IMPROVEMENT", 0.02))

    for p in range(1, passes + 1):
        improved = False
        for idx in range(len(best_rules)):
            local_best: tuple[float, list[dict], dict, dict] | None = None
            for tp in tp_grid:
                for sl in sl_grid:
                    for cap in cap_grid:
                        trial = [dict(r) for r in best_rules]
                        trial[idx]["tp"] = tp
                        trial[idx]["sl"] = sl
                        trial[idx]["capital_pct"] = cap
                        if sum(float(r.get("capital_pct", 0.0)) for r in trial) > max_total_cap:
                            continue
                        train_m, valid_m, score = _evaluate_ruleset(train_engine, valid_engine, trial)
                        if not _is_positive_good(
                            train_m,
                            valid_m,
                            min_train_trades=int(getattr(_cfg, "RB_RULESET_MIN_TRAIN_TRADES", getattr(_cfg, "RB_MIN_TRAIN_TRADES", 25))),
                            min_valid_trades=int(getattr(_cfg, "RB_RULESET_MIN_VALID_TRADES", getattr(_cfg, "RB_MIN_VALID_TRADES", 15))),
                        ):
                            continue
                        if local_best is None or score > local_best[0]:
                            local_best = (score, trial, train_m, valid_m)
            if local_best is not None and local_best[0] > cur_score + min_improve:
                cur_score, best_rules, cur_train, cur_test = local_best
                improved = True
                hist.append({
                    "pass": p,
                    "rule_index": idx + 1,
                    "score": cur_score,
                    "train_return_pct": _f(cur_train, "total_return_pct"),
                    "valid_return_pct": _f(cur_test, "total_return_pct"),
                    "train_pf": _f(cur_train, "profit_factor"),
                    "valid_pf": _f(cur_test, "profit_factor"),
                    "train_dd": _f(cur_train, "max_drawdown_pct"),
                    "valid_dd": _f(cur_test, "max_drawdown_pct"),
                    "tp": best_rules[idx]["tp"],
                    "sl": best_rules[idx]["sl"],
                    "capital_pct": best_rules[idx]["capital_pct"],
                })
                logger.info(
                    "RB [%s]: risk improve pass=%d rule=%d score=%.2f train=%.2f%% test=%.2f%%",
                    direction, p, idx + 1, cur_score, _f(cur_train, "total_return_pct"), _f(cur_test, "total_return_pct"),
                )
        if not improved:
            break
    return best_rules, cur_train, cur_test, cur_score, hist


def _write_clean_evaluator(strategy: dict, output_path: Path) -> None:
    rules = []
    for r in strategy.get("rules_set", []):
        conditions = list(r.get("conditions", []))
        if bool(getattr(_cfg, "RB_SYMBOL_STRICT_OUTPUT_CHECK", True)) and bool(getattr(_cfg, "RB_REQUIRE_SYMBOL_FILTERS", False)):
            if not any(_is_symbol_condition(str(c)) for c in conditions):
                raise ValueError(f"RB output refused: rule has no symbol filter: {conditions}")
        rules.append({
            "conditions": conditions,
            "tp": float(r.get("tp", 2.0)),
            "sl": float(r.get("sl", 1.0)),
            "capital_pct": float(r.get("capital_pct", 1.0)),
        })
    clean = {
        "direction": strategy.get("direction"),
        "rules_set": rules,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(clean, fh, indent=2)


def run_rb_governor_pipeline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    pools: dict[str, list[dict]],
    directions: list[str] | tuple[str, ...],
    *,
    output_dir: str | os.PathLike[str] | None = None,
) -> dict[str, dict]:
    """Build and optimize rb strategies for each direction and write outputs."""
    out_dir = Path(output_dir or _cfg.OUTPUTS_DIR)
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_like, valid_df = _load_scoring_frames(train_df, val_df)
    results: dict[str, dict] = {}

    for direction in directions:
        pool = pools.get(direction, [])
        if not pool:
            logger.warning("RB [%s]: empty Phase 2 pool; skipping.", direction)
            continue
        train_engine = CPUBacktestEngine(train_like, {}, direction)
        valid_engine = CPUBacktestEngine(valid_df, {}, direction)
        candidates = _filter_good_rules(pool, train_like, valid_df, direction)
        if not candidates:
            logger.warning("RB [%s]: no single rules positive on both training and validation; falling back to best raw governor score.", direction)
            candidates = []
            symbols = _available_symbols(train_like, valid_df)
            for raw in pool[: int(getattr(_cfg, "RB_MAX_POOL_RULES_TO_EVALUATE", 700))]:
                for rule in _symbol_specialized_variants(raw, train_engine, valid_engine, symbols):
                    rule = _rule_to_engine(rule)
                    try:
                        tr = train_engine.simulate_rule_set([rule])
                        te = valid_engine.simulate_rule_set([rule])
                    except Exception:
                        continue
                    rec = CandidateRecord(rule=rule, train_metrics=tr, valid_metrics=te, score=_score_metrics(tr, te))
                    rec.mask = _mask_for(rule, train_like, valid_df)
                    candidates.append(rec)
            candidates.sort(key=lambda r: r.score, reverse=True)
            candidates = candidates[: int(getattr(_cfg, "RB_KEEP_TOP_RULES", 120))]
            if not candidates:
                continue

        selected, sel_train, sel_test, sel_score, compose_history = _compose_ruleset(candidates, train_engine, valid_engine, direction)
        opt_rules, opt_train, opt_test, opt_score, risk_history = _optimize_risk(selected, train_engine, valid_engine, direction)

        strategy = _strategy(
            direction,
            opt_rules,
            risk_optimized=True,
            extra={
                "rb_score": opt_score,
                "rb_train_return_pct": _f(opt_train, "total_return_pct"),
                "rb_valid_return_pct": _f(opt_test, "total_return_pct"),
                "rb_valid_profit_factor": _f(opt_test, "profit_factor"),
                "rb_valid_max_drawdown_pct": _f(opt_test, "max_drawdown_pct"),
                "rb_valid_executed_trades": _i(opt_test, "executed_trades"),
                "rb_train_minus_valid_return_pct": _f(opt_train, "total_return_pct") - _f(opt_test, "total_return_pct"),
                "rb_train_valid_ratio": _f(opt_train, "total_return_pct") / max(_f(opt_test, "total_return_pct"), 1e-9),
            },
        )

        strategy_path = out_dir / f"{direction}.json"
        with strategy_path.open("w", encoding="utf-8") as fh:
            json.dump(strategy, fh, indent=2)
        _write_clean_evaluator(strategy, out_dir / "evaluator_clean" / f"{direction}_evaluator_clean.json")

        report = {
            "direction": direction,
            "rb_score": opt_score,
            "train_metrics": opt_train,
            "valid_metrics": opt_test,
            "train_minus_valid_return_pct": _f(opt_train, "total_return_pct") - _f(opt_test, "total_return_pct"),
            "train_valid_ratio": _f(opt_train, "total_return_pct") / max(_f(opt_test, "total_return_pct"), 1e-9),
            "n_positive_single_rules": len(candidates),
            "selected_rules": len(opt_rules),
            "compose_history": compose_history,
            "risk_history": risk_history,
            "top_single_rules": [
                {
                    "rank": i + 1,
                    "score": c.score,
                    "train_return_pct": _f(c.train_metrics, "total_return_pct"),
                    "valid_return_pct": _f(c.valid_metrics, "total_return_pct"),
                    "train_pf": _f(c.train_metrics, "profit_factor"),
                    "valid_pf": _f(c.valid_metrics, "profit_factor"),
                    "valid_dd": _f(c.valid_metrics, "max_drawdown_pct"),
                    "valid_trades": _i(c.valid_metrics, "executed_trades"),
                    "conditions": list(c.rule.get("conditions", [])),
                    "rule": _rule_to_engine(c.rule),
                }
                for i, c in enumerate(candidates[:25])
            ],
        }
        with (reports_dir / f"rb_governor_{direction}_report.json").open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)

        logger.info(
            "RB [%s]: saved %d-rule strategy | score=%.2f train=%.2f%% test=%.2f%% pf=%.2f dd=%.2f%% -> %s",
            direction,
            len(opt_rules),
            opt_score,
            _f(opt_train, "total_return_pct"),
            _f(opt_test, "total_return_pct"),
            _f(opt_test, "profit_factor"),
            _f(opt_test, "max_drawdown_pct"),
            strategy_path,
        )
        results[direction] = strategy
    return results


def evaluate_strategy_governor(train_df: pd.DataFrame, val_df: pd.DataFrame, strategy: dict, direction: str | None = None) -> dict[str, Any]:
    """Evaluate a strategy on training and validation frames."""
    direction = direction or str(strategy.get("direction", "long"))
    train_like, valid_df = _load_scoring_frames(train_df, val_df)
    rules = [_rule_to_engine(r) for r in strategy.get("rules_set", [])]
    train_engine = CPUBacktestEngine(train_like, {}, direction)
    valid_engine = CPUBacktestEngine(valid_df, {}, direction)
    train_m = train_engine.simulate_rule_set(rules)
    valid_m = valid_engine.simulate_rule_set(rules)
    score = _score_metrics(
        train_m,
        valid_m,
        min_train_trades=int(getattr(_cfg, "RB_RULESET_MIN_TRAIN_TRADES", getattr(_cfg, "RB_MIN_TRAIN_TRADES", 25))),
        min_valid_trades=int(getattr(_cfg, "RB_RULESET_MIN_VALID_TRADES", getattr(_cfg, "RB_MIN_VALID_TRADES", 15))),
    )
    return {"direction": direction, "rules": len(rules), "internal_score": score, "train_metrics": train_m, "valid_metrics": valid_m}

def evaluate_strategy_file_governor(train_df: pd.DataFrame, val_df: pd.DataFrame, strategy_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Evaluate a saved strategy file."""
    with Path(strategy_path).open("r", encoding="utf-8") as fh:
        strategy = json.load(fh)
    return evaluate_strategy_governor(train_df, val_df, strategy, strategy.get("direction"))


def _load_bank(bank_path: Path) -> list[dict]:
    if not bank_path.exists():
        return []
    try:
        data = json.load(open(bank_path, "r", encoding="utf-8"))
        return list(data.get("rules", data if isinstance(data, list) else []))
    except Exception:
        return []


def _save_bank(bank_path: Path, direction: str, rows: list[dict]) -> None:
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "direction": direction,
        "rules": rows,
        "count": len(rows),
        "note": "RB global rule bank built from all previous auto-search runs.",
    }
    with bank_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)


def _bank_candidate_rows_from_run(out_dir: Path, direction: str, run_no: int | None) -> list[dict]:
    rows: list[dict] = []
    strategy_path = out_dir / f"{direction}.json"
    if strategy_path.exists():
        try:
            strategy = json.load(open(strategy_path, "r", encoding="utf-8"))
            for idx, r in enumerate(strategy.get("rules_set", [])):
                rows.append({"rule": _rule_to_engine(r), "source": "final_strategy", "run_no": run_no, "rank": idx + 1})
        except Exception:
            pass

    report_path = out_dir / "reports" / f"rb_governor_{direction}_report.json"
    limit = int(getattr(_cfg, "RB_GLOBAL_BANK_IMPORT_TOP_SINGLE_RULES", 80))
    if report_path.exists():
        try:
            report = json.load(open(report_path, "r", encoding="utf-8"))
            for i, item in enumerate(report.get("top_single_rules", [])[:limit]):
                rule = item.get("rule") or {"conditions": item.get("conditions", [])}
                rows.append({"rule": _rule_to_engine(rule), "source": "top_single_rule", "run_no": run_no, "rank": i + 1})
        except Exception:
            pass
    return rows


def _evaluate_bank_rule(rule: dict, train_engine: CPUBacktestEngine, valid_engine: CPUBacktestEngine) -> tuple[dict, dict, float, float]:
    train_m = train_engine.simulate_rule_set([_rule_to_engine(rule)])
    valid_m = valid_engine.simulate_rule_set([_rule_to_engine(rule)])
    return train_m, valid_m, _score_metrics(train_m, valid_m), _combined_return_score(train_m, valid_m)


def _refresh_bank(
    root: Path,
    out_dir: Path,
    direction: str,
    train_like: pd.DataFrame,
    valid_df: pd.DataFrame,
    run_no: int | None,
) -> list[dict]:
    bank_dir = root / str(getattr(_cfg, "RB_GLOBAL_BANK_DIRNAME", "rb_bank"))
    bank_path = bank_dir / f"{direction}_rules_bank.json"
    existing = _load_bank(bank_path)
    incoming = _bank_candidate_rows_from_run(out_dir, direction, run_no)

    train_engine = CPUBacktestEngine(train_like, {}, direction)
    valid_engine = CPUBacktestEngine(valid_df, {}, direction)
    symbols = _available_symbols(train_like, valid_df)

    by_key: dict[tuple[str, ...], dict] = {}
    for row in existing + incoming:
        raw_rule = row.get("rule", {})
        for rule in _symbol_specialized_variants(raw_rule, train_engine, valid_engine, symbols):
            rule = _rule_to_engine(rule)
            if not rule.get("conditions"):
                continue
            key = _rule_key(rule)
            try:
                tr, te, rb_score, ret_score = _evaluate_bank_rule(rule, train_engine, valid_engine)
            except Exception:
                continue
            if bool(getattr(_cfg, "RB_GLOBAL_REQUIRE_POSITIVE_TRAIN_TEST", True)) and not _positive_returns(tr, te):
                continue
            item = {
                "direction": direction,
                "rule": rule,
                "key": list(key),
                "rb_score": rb_score,
                "combined_return_score": ret_score,
                "train_return_pct": _f(tr, "total_return_pct"),
                "valid_return_pct": _f(te, "total_return_pct"),
                "train_pf": _f(tr, "profit_factor"),
                "valid_pf": _f(te, "profit_factor"),
                "train_dd": _f(tr, "max_drawdown_pct"),
                "valid_dd": _f(te, "max_drawdown_pct"),
                "train_trades": _i(tr, "executed_trades"),
                "valid_trades": _i(te, "executed_trades"),
                "sources": sorted(set((row.get("sources") or []) + [str(row.get("source", "unknown"))])),
                "run_nos": sorted(set((row.get("run_nos") or []) + ([int(run_no)] if run_no is not None else []))),
            }
            old = by_key.get(key)
            if old is None or ret_score > float(old.get("combined_return_score", -1e18)):
                by_key[key] = item

    rows = sorted(by_key.values(), key=lambda x: float(x.get("combined_return_score", -1e18)), reverse=True)
    rows = rows[: int(getattr(_cfg, "RB_GLOBAL_BANK_MAX_RULES_PER_DIRECTION", 700))]
    _save_bank(bank_path, direction, rows)
    logger.info("RB global bank [%s]: %d symbol-filtered rules saved to %s", direction, len(rows), bank_path)
    return rows

def _candidate_records_from_bank(rows: list[dict], train_like: pd.DataFrame, valid_df: pd.DataFrame) -> list[CandidateRecord]:
    records: list[CandidateRecord] = []
    for row in rows:
        rule = _rule_to_engine(row.get("rule", {}))
        tr = {
            "total_return_pct": row.get("train_return_pct", 0.0),
            "profit_factor": row.get("train_pf", 0.0),
            "max_drawdown_pct": row.get("train_dd", 0.0),
            "executed_trades": row.get("train_trades", 0),
        }
        te = {
            "total_return_pct": row.get("valid_return_pct", 0.0),
            "profit_factor": row.get("valid_pf", 0.0),
            "max_drawdown_pct": row.get("valid_dd", 0.0),
            "executed_trades": row.get("valid_trades", 0),
        }
        rec = CandidateRecord(rule=rule, train_metrics=tr, valid_metrics=te, score=float(row.get("rb_score", 0.0)))
        rec.mask = _mask_for(rule, train_like, valid_df)
        records.append(rec)
    records.sort(key=lambda r: _combined_return_score(r.train_metrics, r.valid_metrics), reverse=True)
    return records


def _compose_ruleset_return_only(
    candidates: list[CandidateRecord],
    train_engine: CPUBacktestEngine,
    valid_engine: CPUBacktestEngine,
    direction: str,
    *,
    max_rules: int,
    min_improve: float,
) -> tuple[list[CandidateRecord], dict, dict, float, list[dict]]:
    if not candidates:
        raise ValueError("No bank candidates available")
    selected = [candidates[0]]
    cur_train, cur_test, _ = _evaluate_ruleset(train_engine, valid_engine, [selected[0].rule])
    cur_score = _combined_return_score(cur_train, cur_test)
    used = {_rule_key(selected[0].rule)}
    history = [{
        "step": 1,
        "action": "seed",
        "combined_return_score": cur_score,
        "train_return_pct": _f(cur_train, "total_return_pct"),
        "valid_return_pct": _f(cur_test, "total_return_pct"),
        "rules": 1,
    }]
    while len(selected) < max_rules:
        best: tuple[float, CandidateRecord, dict, dict] | None = None
        for cand in candidates:
            key = _rule_key(cand.rule)
            if key in used:
                continue
            rules = [r.rule for r in selected] + [cand.rule]
            train_m, valid_m, _ = _evaluate_ruleset(train_engine, valid_engine, rules)
            if not _positive_returns(train_m, valid_m):
                continue
            ret_score = _combined_return_score(train_m, valid_m)
            if ret_score <= cur_score + min_improve:
                continue
            if best is None or ret_score > best[0]:
                best = (ret_score, cand, train_m, valid_m)
        if best is None:
            logger.info("RB global [%s]: no further profit-improving rule found at %d rules.", direction, len(selected))
            break
        cur_score, cand, cur_train, cur_test = best
        selected.append(cand)
        used.add(_rule_key(cand.rule))
        history.append({
            "step": len(selected),
            "action": "add_rule_by_profit",
            "combined_return_score": cur_score,
            "rb_score": _score_metrics(cur_train, cur_test),
            "train_return_pct": _f(cur_train, "total_return_pct"),
            "valid_return_pct": _f(cur_test, "total_return_pct"),
            "train_pf": _f(cur_train, "profit_factor"),
            "valid_pf": _f(cur_test, "profit_factor"),
            "train_dd": _f(cur_train, "max_drawdown_pct"),
            "valid_dd": _f(cur_test, "max_drawdown_pct"),
            "rules": len(selected),
        })
        logger.info("RB global [%s]: grew to %d rules by profit | train=%.2f%% test=%.2f%% combined=%.2f",
                    direction, len(selected), _f(cur_train, "total_return_pct"), _f(cur_test, "total_return_pct"), cur_score)
    return selected, cur_train, cur_test, cur_score, history


def _optimize_risk_return_only(
    rules: list[dict],
    train_engine: CPUBacktestEngine,
    valid_engine: CPUBacktestEngine,
    direction: str,
) -> tuple[list[dict], dict, dict, float, list[dict]]:
    best_rules = [_rule_to_engine(r) for r in rules]
    cur_train, cur_test, _ = _evaluate_ruleset(train_engine, valid_engine, best_rules)
    cur_score = _combined_return_score(cur_train, cur_test)
    hist = [{
        "pass": 0,
        "rule_index": -1,
        "combined_return_score": cur_score,
        "train_return_pct": _f(cur_train, "total_return_pct"),
        "valid_return_pct": _f(cur_test, "total_return_pct"),
    }]
    tp_grid = tuple(float(x) for x in getattr(_cfg, "RB_GLOBAL_TP_GRID", getattr(_cfg, "RB_TP_GRID", (2.0,))))
    sl_grid = tuple(float(x) for x in getattr(_cfg, "RB_GLOBAL_SL_GRID", getattr(_cfg, "RB_SL_GRID", (1.0,))))
    cap_grid = tuple(float(x) for x in getattr(_cfg, "RB_GLOBAL_CAPITAL_GRID", getattr(_cfg, "RB_CAPITAL_GRID", (12.5,))))
    max_total_cap = float(getattr(_cfg, "RB_GLOBAL_MAX_TOTAL_CAPITAL", getattr(_cfg, "RB_MAX_TOTAL_CAPITAL", 100.0)))
    passes = int(getattr(_cfg, "RB_GLOBAL_RISK_OPT_PASSES", 1))
    min_improve = float(getattr(_cfg, "RB_GLOBAL_MIN_COMBINED_RETURN_IMPROVEMENT", 0.05))
    for pno in range(1, passes + 1):
        improved = False
        for idx in range(len(best_rules)):
            local: tuple[float, list[dict], dict, dict] | None = None
            for tp in tp_grid:
                for sl in sl_grid:
                    for cap in cap_grid:
                        trial = [dict(r) for r in best_rules]
                        trial[idx]["tp"] = tp
                        trial[idx]["sl"] = sl
                        trial[idx]["capital_pct"] = cap
                        if sum(float(r.get("capital_pct", 0.0)) for r in trial) > max_total_cap:
                            continue
                        train_m, valid_m, _ = _evaluate_ruleset(train_engine, valid_engine, trial)
                        if not _positive_returns(train_m, valid_m):
                            continue
                        ret_score = _combined_return_score(train_m, valid_m)
                        if local is None or ret_score > local[0]:
                            local = (ret_score, trial, train_m, valid_m)
            if local is not None and local[0] > cur_score + min_improve:
                cur_score, best_rules, cur_train, cur_test = local
                improved = True
                hist.append({
                    "pass": pno,
                    "rule_index": idx + 1,
                    "combined_return_score": cur_score,
                    "train_return_pct": _f(cur_train, "total_return_pct"),
                    "valid_return_pct": _f(cur_test, "total_return_pct"),
                    "train_pf": _f(cur_train, "profit_factor"),
                    "valid_pf": _f(cur_test, "profit_factor"),
                    "train_dd": _f(cur_train, "max_drawdown_pct"),
                    "valid_dd": _f(cur_test, "max_drawdown_pct"),
                    "tp": best_rules[idx]["tp"],
                    "sl": best_rules[idx]["sl"],
                    "capital_pct": best_rules[idx]["capital_pct"],
                })
        if not improved:
            break
    return best_rules, cur_train, cur_test, cur_score, hist


def update_global_bank_and_compose(root: Path, out_dir: Path, direction: str, run_no: int | None = None) -> None:
    """Update RB rule bank from the current run and compose best_global.

    Unlike best/ (which stores the best single run), best_global/ is built from
    the accumulated bank of rules across all completed runs.  New rules are
    added mainly when the combined training and validation return improves.
    """
    if not bool(getattr(_cfg, "RB_GLOBAL_BANK_ENABLED", True)):
        return
    train_like, valid_df = _load_scoring_frames(train_df, val_df)

    rows = _refresh_bank(root, out_dir, direction, train_like, valid_df, run_no)
    candidates = _candidate_records_from_bank(rows, train_like, valid_df)
    if not candidates:
        return
    train_engine = CPUBacktestEngine(train_like, {}, direction)
    valid_engine = CPUBacktestEngine(valid_df, {}, direction)
    selected, train_m, valid_m, profit_score, compose_hist = _compose_ruleset_return_only(
        candidates,
        train_engine,
        valid_engine,
        direction,
        max_rules=int(getattr(_cfg, "RB_GLOBAL_MAX_RULES", 12)),
        min_improve=float(getattr(_cfg, "RB_GLOBAL_MIN_COMBINED_RETURN_IMPROVEMENT", 0.05)),
    )
    rules = [_rule_to_engine(r.rule) for r in selected]
    rules, train_m, valid_m, profit_score, risk_hist = _optimize_risk_return_only(rules, train_engine, valid_engine, direction)
    rb_score = _score_metrics(train_m, valid_m)

    best_dir = root / str(getattr(_cfg, "RB_GLOBAL_BEST_DIRNAME", "best_global"))
    best_dir.mkdir(parents=True, exist_ok=True)
    meta_path = best_dir / f"best_global_{direction}_meta.json"
    old_profit = -1e18
    if meta_path.exists():
        try:
            old_profit = float(json.load(open(meta_path, "r", encoding="utf-8")).get("combined_return_score", -1e18))
        except Exception:
            old_profit = -1e18
    if profit_score <= old_profit:
        logger.info("RB global [%s]: composed score %.2f did not beat best %.2f", direction, profit_score, old_profit)
        return

    strategy = _strategy(
        direction,
        rules,
        risk_optimized=True,
        extra={
            "rb_global_bank": True,
            "rb_global_rules": len(rules),
            "rb_score": rb_score,
            "combined_return_score": profit_score,
            "rb_train_return_pct": _f(train_m, "total_return_pct"),
            "rb_valid_return_pct": _f(valid_m, "total_return_pct"),
            "rb_train_minus_valid_return_pct": _f(train_m, "total_return_pct") - _f(valid_m, "total_return_pct"),
        },
    )
    strategy_path = best_dir / f"best_global_{direction}.json"
    with strategy_path.open("w", encoding="utf-8") as fh:
        json.dump(strategy, fh, indent=2, default=str)
    _write_clean_evaluator(strategy, best_dir / f"best_global_{direction}_evaluator_clean.json")
    meta = {
        "run_no": run_no,
        "direction": direction,
        "rules": len(rules),
        "combined_return_score": profit_score,
        "rb_score": rb_score,
        "train_return_pct": _f(train_m, "total_return_pct"),
        "valid_return_pct": _f(valid_m, "total_return_pct"),
        "train_minus_valid_return_pct": _f(train_m, "total_return_pct") - _f(valid_m, "total_return_pct"),
        "valid_profit_factor": _f(valid_m, "profit_factor"),
        "valid_max_drawdown_pct": _f(valid_m, "max_drawdown_pct"),
        "valid_executed_trades": _i(valid_m, "executed_trades"),
        "bank_rules": len(rows),
        "compose_history": compose_hist,
        "risk_history": risk_hist,
        "note": "Global RB rule-set composed from all previous run rules. Rule addition objective is combined training and validation return.",
    }
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    logger.info("RB global [%s]: new best_global %d rules combined=%.2f train=%.2f%% test=%.2f%% -> %s",
                direction, len(rules), profit_score, _f(train_m, "total_return_pct"), _f(valid_m, "total_return_pct"), strategy_path)

def update_global_best(root: Path, out_dir: Path, direction: str, score_payload: dict[str, Any], run_no: int | None = None) -> None:
    """Copy best-so-far strategy per direction into root/best/ when score improves."""
    try:
        score = float(score_payload.get("internal_score", -1e18))
    except Exception:
        return
    best_dir = root / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    meta_path = best_dir / f"best_{direction}_meta.json"
    old_score = -1e18
    if meta_path.exists():
        try:
            old_score = float(json.load(open(meta_path)).get("internal_score", -1e18))
        except Exception:
            old_score = -1e18
    if score <= old_score:
        return
    src = out_dir / f"{direction}.json"
    if not src.exists():
        return
    dst = best_dir / f"best_{direction}.json"
    shutil.copy2(src, dst)
    clean_src = out_dir / "evaluator_clean" / f"{direction}_evaluator_clean.json"
    if clean_src.exists():
        shutil.copy2(clean_src, best_dir / f"best_{direction}_evaluator_clean.json")
    meta = {
        "run_no": run_no,
        "direction": direction,
        "internal_score": score,
        "output_dir": str(out_dir),
        "train_return_pct": (score_payload.get("train_metrics") or {}).get("total_return_pct"),
        "valid_return_pct": (score_payload.get("valid_metrics") or {}).get("total_return_pct"),
        "valid_profit_factor": (score_payload.get("valid_metrics") or {}).get("profit_factor"),
        "valid_max_drawdown_pct": (score_payload.get("valid_metrics") or {}).get("max_drawdown_pct"),
        "train_minus_valid_return_pct": float((score_payload.get("train_metrics") or {}).get("total_return_pct", 0.0)) - float((score_payload.get("valid_metrics") or {}).get("total_return_pct", 0.0)),
        "updated_note": "Best-so-far under rb governor scoring; uses validation frame; prefers training return slightly above validation return.",
    }
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    logger.info("RB auto-search: new best %s score=%.2f copied to %s", direction, score, dst)
