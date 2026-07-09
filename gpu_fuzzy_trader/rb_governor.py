
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
from gpu_fuzzy_trader.backtest.cpu_engine import (
    CPUBacktestEngine,
    _build_rule_signal_mask,
)
from gpu_fuzzy_trader.backtest.df_slim import downcast_numeric_df
from gpu_fuzzy_trader.data.loader import Data_Loader
from gpu_fuzzy_trader.scoring import return_to_drawdown, profit_factor_term
from gpu_fuzzy_trader.validation.monthly_windows import (
    build_monthly_windows,
    summarize_monthly_metrics,
    MonthlyWindowSummary,
)

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


def _symbol_concentration_stats(metrics: dict | None) -> tuple[float, float, str]:
    """Return (hhi_abs_pnl, top_symbol_share, top_symbol) from per_symbol_metrics."""
    per_sym = (metrics or {}).get("per_symbol_metrics", {}) or {}
    pnls: list[float] = []
    top_symbol = ""
    top_abs = 0.0
    for sym, v in per_sym.items():
        if not isinstance(v, dict):
            continue
        val = float(v.get("net_pnl", 0.0))
        abs_val = abs(val)
        pnls.append(abs_val)
        if abs_val > top_abs:
            top_abs = abs_val
            top_symbol = str(sym)
    total_abs = float(np.sum(pnls)) if pnls else 0.0
    if total_abs <= 0.0:
        return 0.0, 0.0, top_symbol
    shares = np.asarray(pnls, dtype=np.float64) / total_abs
    return float(np.sum(shares * shares)), float(np.max(shares)), top_symbol


def _passes_symbol_concentration_gate(valid_m: dict) -> tuple[bool, dict[str, Any]]:
    """Hard gate: reject strategies dominated by one symbol on validation."""
    hhi, top_share, top_sym = _symbol_concentration_stats(valid_m)
    max_share = float(getattr(_cfg, "RB_MAX_SYMBOL_SHARE_ABS_PNL", 0.50))
    max_hhi = float(getattr(_cfg, "RB_MAX_SYMBOL_HHI", 0.55))
    ok = top_share <= max_share + 1e-12 and hhi <= max_hhi + 1e-12
    return ok, {
        "hhi_abs_pnl": hhi,
        "top_symbol_share_abs_pnl": top_share,
        "top_symbol": top_sym,
        "max_share": max_share,
        "max_hhi": max_hhi,
        "passed": bool(ok),
    }


def _passes_tail_holdout_gate(
    risk_history: list[dict],
) -> tuple[bool, dict[str, Any]]:
    """Hard gate on risk-grid tail holdout return (when enabled and present)."""
    if not bool(getattr(_cfg, "RB_TAIL_HOLDOUT_HARD_GATE", True)):
        return True, {"enabled": False, "passed": True}
    if not risk_history:
        return True, {"enabled": True, "available": False, "passed": True}
    final = risk_history[-1]
    if "risk_tail_holdout_return_pct" not in final:
        return True, {"enabled": True, "available": False, "passed": True}
    tail_ret = float(final.get("risk_tail_holdout_return_pct", 0.0))
    min_ret = float(getattr(_cfg, "RB_TAIL_HOLDOUT_MIN_RETURN_PCT", 0.0))
    ok = tail_ret >= min_ret - 1e-12
    return ok, {
        "enabled": True,
        "available": True,
        "tail_return_pct": tail_ret,
        "min_return_pct": min_ret,
        "passed": bool(ok),
    }


def _i(metrics: dict | None, key: str, default: int = 0) -> int:
    try:
        return int((metrics or {}).get(key, default))
    except Exception:
        return default


def _train_valid_shape(train_ret: float, valid_ret: float) -> tuple[bool, float, float]:
    """Return (ok, bonus, penalty) for the desired train-valid balance shape.

    In rb/governor mode the user wants the scoring shape to stay within the configured train-valid range.  This helper
    makes that preference explicit in both filtering and scoring.
    """
    if not bool(getattr(_cfg, "RB_REQUIRE_TRAIN_SLIGHTLY_ABOVE_VALID", False)):
        return True, 0.0, 0.0
    if valid_ret <= 0.0 or train_ret <= 0.0:
        return False, 0.0, 0.0

    min_ratio = float(getattr(_cfg, "RB_TRAIN_VALID_MIN_RATIO", 1.03))
    max_ratio = float(getattr(_cfg, "RB_TRAIN_VALID_MAX_RATIO", 1.35))
    min_abs_gap = float(getattr(_cfg, "RB_TRAIN_VALID_MIN_ABS_GAP", 0.20))
    max_abs_gap = float(getattr(_cfg, "RB_TRAIN_VALID_MAX_ABS_GAP", 12.0))
    below_weight = float(getattr(_cfg, "RB_TRAIN_BELOW_VALID_PENALTY", 900.0))
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
            penalty += (skip_ratio - max_skip) * skip_weight * (1.5 if role == "holdout" else 1.0)
        if exec_ratio < min_exec:
            penalty += (min_exec - exec_ratio) * exec_weight * (1.5 if role == "holdout" else 1.0)
    if max_pos > pos_limit:
        penalty += (max_pos - pos_limit) * pos_weight * (1.2 if role == "holdout" else 1.0)
    return float(penalty)


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


def _enforce_capital_budget(
    rules: list[dict],
    *,
    max_total: float | None = None,
) -> list[dict]:
    """Normalize rule capital_pct so sum <= RB_MAX_TOTAL_CAPITAL."""
    cap_limit = float(
        max_total
        if max_total is not None
        else getattr(_cfg, "RB_MAX_TOTAL_CAPITAL", 100.0)
    )
    cleaned = [_rule_to_engine(r) for r in rules]
    total = sum(float(r["capital_pct"]) for r in cleaned)
    if total <= cap_limit + 1e-9:
        return cleaned
    scale = cap_limit / max(total, 1e-12)
    for rule in cleaned:
        rule["capital_pct"] = round(float(rule["capital_pct"]) * scale, 6)
    return cleaned


def _assert_capital_budget(rules: list[dict], *, max_total: float | None = None) -> None:
    cap_limit = float(
        max_total
        if max_total is not None
        else getattr(_cfg, "RB_MAX_TOTAL_CAPITAL", 100.0)
    )
    total = sum(float(r.get("capital_pct", 0.0)) for r in rules)
    if total > cap_limit + 1e-6:
        raise ValueError(
            f"RB capital budget exceeded: sum(capital_pct)={total:.4f} > {cap_limit:.4f}"
        )


def _strategy(direction: str, rules: list[dict], *, risk_optimized: bool = False, extra: dict | None = None) -> dict:
    clean_rules = _enforce_capital_budget(rules)
    _assert_capital_budget(clean_rules)
    out = {
        "direction": direction,
        "rules_set": clean_rules,
        "risk_optimized": bool(risk_optimized),
    }
    if extra:
        out.update(extra)
    return out


def _score_metrics(train_m: dict, valid_m: dict, *, min_train_trades: int | None = None, min_valid_trades: int | None = None, cv_fold_returns: list[float] | None = None) -> float:
    """Dominant objective: return/DD with train-valid balance, plus CV-fold consistency."""
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
        60.0 * valid_ratio
        + 60.0 * train_ratio
        + 3.0 * valid_ret
        + 3.0 * train_ret
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
    if bool(getattr(_cfg, "RB_REQUIRE_TRAIN_SLIGHTLY_ABOVE_VALID", False)) and not shape_ok:
        score -= 250.0

    score -= max(0.0, train_ratio - valid_ratio) * float(getattr(_cfg, "RB_TRAIN_VALID_RATIO_GAP_WEIGHT", 30.0))
    score -= max(0.0, train_ret - valid_ret) * float(getattr(_cfg, "RB_TRAIN_VALID_RETURN_GAP_WEIGHT", 4.0))

    # CV-fold consistency penalty (when per-fold returns are available)
    if cv_fold_returns and len(cv_fold_returns) > 1:
        cv_min = min(cv_fold_returns)
        cv_max = max(cv_fold_returns)
        cv_range = cv_max - cv_min
        cv_mean = sum(cv_fold_returns) / len(cv_fold_returns)
        # Penalize rules where worst fold is negative
        if cv_min < 0:
            score -= abs(cv_min) * 5.0
        # Penalize high variance across folds (inconsistent OOS)
        if abs(cv_mean) > 0.01 and cv_range > abs(cv_mean) * 2.0:
            score -= (cv_range / max(abs(cv_mean), 0.01) - 2.0) * 5.0

    return float(score)




def _combined_return_score(train_m: dict, valid_m: dict, *, prev_pf: float | None = None, prev_dd: float | None = None) -> float:
    """Profit objective for lenient rule addition, but now evaluator_v5 aware.

    A new rule can still be added mainly when it increases profit, but it must
    not create the  failure mode where most raw signals are skipped by the
    evaluator.  Therefore the score subtracts execution-health penalties.

    When prev_pf and prev_dd are provided, penalties are applied for
    profit-factor degradation and drawdown increase (M7 fix).
    """
    train_ret = _f(train_m, "total_return_pct")
    valid_ret = _f(valid_m, "total_return_pct")
    score = train_ret + valid_ret
    # Penalize edge-quality erosion (M7)
    if prev_pf is not None:
        new_pf = _f(valid_m, "profit_factor", 0.0)
        score -= 2.0 * max(0.0, prev_pf - new_pf)
    if prev_dd is not None:
        new_dd = _f(valid_m, "max_drawdown_pct", 0.0)
        score -= 3.0 * max(0.0, new_dd - prev_dd)
    score -= _evaluator_health_penalty(train_m, role="train") / 35.0
    score -= _evaluator_health_penalty(valid_m, role="valid") / 35.0
    return float(score)


def _positive_returns(train_m: dict, valid_m: dict) -> bool:
    if not bool(getattr(_cfg, "RB_GLOBAL_REQUIRE_POSITIVE_TRAIN_VALID", True)):
        return True
    return _f(train_m, "total_return_pct") > 0.0 and _f(valid_m, "total_return_pct") > 0.0


def _symbols_in_rules(rules: list[dict]) -> set[str]:
    symbols: set[str] = set()
    for rule in rules:
        for cond in rule.get("conditions", []):
            text = str(cond).strip().lower()
            if text.startswith("symbol is "):
                symbols.add(text[len("symbol is "):].strip())
            elif text.startswith("[symbol] is "):
                symbols.add(text[len("[symbol] is "):].strip())
    return symbols


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
    max_variants = max(1, int(getattr(_cfg, "RB_SYMBOL_MAX_VARIANTS_PER_RULE", 10)))
    min_train_trades = int(getattr(_cfg, "RB_SYMBOL_MIN_TRAIN_TRADES", 10))
    min_valid_trades = int(getattr(_cfg, "RB_SYMBOL_MIN_VALID_TRADES", 6))

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
    eligible_syms = [sym for _score, sym, _tr, _te in scored_singles]
    candidate_symbol_sets: list[tuple[str, ...]] = [(sym,) for sym in eligible_syms]

    if bool(getattr(_cfg, "RB_SYMBOL_USE_COMBINATIONS", True)):
        for k in range(2, min(max_symbols, len(eligible_syms)) + 1):
            for combo in combinations(eligible_syms, k):
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
    if bool(getattr(_cfg, "RB_REQUIRE_TRAIN_SLIGHTLY_ABOVE_VALID", False)) and not shape_ok:
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


def _eval_cv_fold_returns(
    rule: dict,
    fold_engines: list[CPUBacktestEngine] | None,
) -> list[float] | None:
    """Evaluate *rule* on each CV fold engine and return per-fold returns.

    Each fold is simulated independently so a single failing fold does not
    drop the entire CV signal.  Returns ``None`` when *fold_engines* is
    ``None``, empty, or all folds fail.
    """
    if not fold_engines:
        return None
    returns: list[float] = []
    for idx, fold_engine in enumerate(fold_engines):
        try:
            m = fold_engine.simulate_rule_set([rule])
            ret = _f(m, "total_return_pct")
            returns.append(ret)
        except Exception:
            logger.warning(
                "CV fold %d simulation failed for rule %s; skipping fold.",
                idx, _rule_key(rule),
            )
            continue
    return returns if returns else None


def _filter_good_rules(
    pool: list[dict],
    train_like_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    direction: str,
    fold_engines: list[CPUBacktestEngine] | None = None,
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
            # Evaluate on CV folds if available (C4)
            cv_fold_returns = _eval_cv_fold_returns(rule, fold_engines)
            score = _score_metrics(train_m, valid_m, cv_fold_returns=cv_fold_returns)
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
    cur_train, cur_valid, cur_score = _evaluate_ruleset(train_engine, valid_engine, [selected[0].rule])
    history = [{
        "step": 1,
        "action": "seed",
        "score": cur_score,
        "train_return_pct": _f(cur_train, "total_return_pct"),
        "valid_return_pct": _f(cur_valid, "total_return_pct"),
        "rules": 1,
    }]

    max_rules = int(_cfg.RB_MAX_RULES)
    min_distinct_symbols = int(getattr(_cfg, "RB_MIN_DISTINCT_SYMBOLS", 0))
    max_overlap = float(getattr(_cfg, "RB_MAX_PAIR_OVERLAP", 0.22))
    min_score_improve = float(getattr(_cfg, "RB_MIN_SCORE_IMPROVEMENT", 0.05))
    min_train_ret_improve = float(getattr(_cfg, "RB_MIN_TRAIN_RETURN_IMPROVEMENT", 0.01))
    min_valid_ret_improve = float(getattr(_cfg, "RB_MIN_VALID_RETURN_IMPROVEMENT", 0.01))
    require_subset_improve = bool(getattr(_cfg, "RB_RULESET_MUST_BEAT_SUBSETS", True))
    return_only_add = bool(getattr(_cfg, "RB_RULE_ADD_BY_RETURN_ONLY", False))
    ignore_overlap = bool(getattr(_cfg, "RB_RULE_ADD_IGNORE_OVERLAP", False))
    if return_only_add:
        require_subset_improve = False if bool(getattr(_cfg, "RB_RULE_ADD_IGNORE_SUBSET_BEAT", True)) else require_subset_improve
        cur_return_score = _combined_return_score(cur_train, cur_valid, prev_pf=None, prev_dd=None)
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
            if min_distinct_symbols > 0:
                selected_syms = _symbols_in_rules([r.rule for r in selected])
                cand_syms = _symbols_in_rules([cand.rule])
                if len(selected_syms) < min_distinct_symbols and not (cand_syms - selected_syms):
                    continue
            train_m, valid_m, score = _evaluate_ruleset(train_engine, valid_engine, trial_rules)
            if return_only_add:
                if not _positive_returns(train_m, valid_m):
                    continue
                prev_pf = _f(cur_valid, "profit_factor", 0.0)
                prev_dd = _f(cur_valid, "max_drawdown_pct", 0.0)
                ret_score = _combined_return_score(train_m, valid_m, prev_pf=prev_pf, prev_dd=prev_dd)
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
                    if _f(valid_m, "total_return_pct") <= max(_f(cur_valid, "total_return_pct"), _f(cand.valid_metrics, "total_return_pct")) + min_valid_ret_improve:
                        continue
                if score <= cur_score + min_score_improve:
                    continue
                choose_score = score
            if best is None or choose_score > best[0]:
                best = (choose_score, cand, train_m, valid_m)

        if best is None:
            logger.info("RB [%s]: no further positive/improving low-overlap extension found at %d rules.", direction, len(selected))
            break

        chosen_score, chosen, cur_train, cur_valid = best
        cur_score = _score_metrics(
            cur_train,
            cur_valid,
            min_train_trades=int(getattr(_cfg, "RB_RULESET_MIN_TRAIN_TRADES", getattr(_cfg, "RB_MIN_TRAIN_TRADES", 25))),
            min_valid_trades=int(getattr(_cfg, "RB_RULESET_MIN_VALID_TRADES", getattr(_cfg, "RB_MIN_VALID_TRADES", 15))),
        )
        if return_only_add:
            cur_return_score = _combined_return_score(cur_train, cur_valid, prev_pf=None, prev_dd=None)
        selected.append(chosen)
        used.add(_rule_key(chosen.rule))
        history.append({
            "step": len(selected),
            "action": "add_rule",
            "score": cur_score,
            "train_return_pct": _f(cur_train, "total_return_pct"),
            "valid_return_pct": _f(cur_valid, "total_return_pct"),
            "train_pf": _f(cur_train, "profit_factor"),
            "valid_pf": _f(cur_valid, "profit_factor"),
            "train_dd": _f(cur_train, "max_drawdown_pct"),
            "valid_dd": _f(cur_valid, "max_drawdown_pct"),
            "rules": len(selected),
            "combined_return_score": _combined_return_score(cur_train, cur_valid, prev_pf=None, prev_dd=None),
        })
        logger.info(
            "RB [%s]: grew to %d rules | score=%.2f train_ret=%.2f%% valid_ret=%.2f%%",
            direction, len(selected), cur_score, _f(cur_train, "total_return_pct"), _f(cur_valid, "total_return_pct"),
        )

    return selected, cur_train, cur_valid, cur_score, history


def _make_walk_forward_fold_engines(
    val_selection_df: pd.DataFrame,
    n_splits: int,
    tail_holdout_frac: float,
    direction: str,
) -> tuple[list[CPUBacktestEngine], CPUBacktestEngine | None]:
    # → fixes audit finding #3 (RB Governor risk-grid overfits val_selection)
    # → fixes audit finding #12 (PHASE4_TAIL_HOLDOUT_FRACTION orphan, now wired)
    """Split val_selection into n_splits chronological folds + optional tail holdout.

    Per-symbol chronological split (matches data/splitter.py convention).
    Returns (fold_engines, tail_holdout_engine_or_None).
    """
    if "symbol" not in val_selection_df.columns or "datetime" not in val_selection_df.columns:
        # Single-symbol or no-symbol data: treat entire df as one symbol
        symbols = ["_all"]
        sym_data = {"_all": val_selection_df.copy().sort_values("datetime").reset_index(drop=True)}
    else:
        sym_data = {}
        df_sorted = val_selection_df.copy().sort_values(["symbol", "datetime"]).reset_index(drop=True)
        for sym, grp in df_sorted.groupby("symbol", sort=False):
            sym_data[str(sym)] = grp.reset_index(drop=True)
        symbols = sorted(sym_data.keys())

    fold_dfs: list[pd.DataFrame] = [pd.DataFrame() for _ in range(n_splits)]
    tail_dfs: list[pd.DataFrame] = []

    for sym in symbols:
        sym_df = sym_data[sym]
        n = len(sym_df)
        if n < n_splits + 1:
            # Too few rows; duplicate to make splitting possible
            sym_df = pd.concat([sym_df] * (n_splits + 1), ignore_index=True)
            n = len(sym_df)

        # Reserve tail holdout fraction
        if tail_holdout_frac > 0.0 and n_splits >= 1:
            tail_n = max(1, int(round(n * tail_holdout_frac)))
            head_n = n - tail_n
            head_df = sym_df.iloc[:head_n].reset_index(drop=True)
            tail_df = sym_df.iloc[head_n:].reset_index(drop=True)
            tail_dfs.append(tail_df)
        else:
            head_df = sym_df.copy()
            tail_dfs.append(pd.DataFrame())

        # Split head into n_splits contiguous chunks
        head_len = len(head_df)
        chunk_size = max(1, head_len // n_splits)
        for i in range(n_splits):
            start = i * chunk_size
            if i < n_splits - 1:
                end = start + chunk_size
            else:
                end = head_len
            if start < head_len:
                fold_dfs[i] = pd.concat([fold_dfs[i], head_df.iloc[start:end]], ignore_index=True)

    # Build fold engines
    fold_engines: list[CPUBacktestEngine] = []
    for fold_df in fold_dfs:
        if len(fold_df) == 0:
            # Empty fold — pad with a copy of another fold's data
            fallback = next((fd for fd in fold_dfs if len(fd) > 0), val_selection_df)
            fold_df = fallback.copy()
        prepared = _prepare_scoring_frame(fold_df)
        fold_engines.append(CPUBacktestEngine(prepared, {}, direction))

    # Build tail holdout engine (if any tail data)
    tail_engine: CPUBacktestEngine | None = None
    if tail_holdout_frac > 0.0:
        combined_tail = pd.concat(tail_dfs, ignore_index=True)
        if len(combined_tail) > 0:
            prepared_tail = _prepare_scoring_frame(combined_tail)
            tail_engine = CPUBacktestEngine(prepared_tail, {}, direction)
        # Build tail holdout engine even from the fallback head portion
        if tail_engine is None:
            prepared_tail = _prepare_scoring_frame(fold_dfs[0].head(1))
            tail_engine = CPUBacktestEngine(prepared_tail, {}, direction)

    return fold_engines, tail_engine


def _optimize_risk(
    selected: list[CandidateRecord],
    train_engine: CPUBacktestEngine,
    valid_engine: CPUBacktestEngine,
    direction: str,
    fold_engines: list[CPUBacktestEngine] | None = None,
    tail_holdout_engine: CPUBacktestEngine | None = None,
) -> tuple[list[dict], dict, dict, float, list[dict]]:
    rules = [_rule_to_engine(r.rule) for r in selected]
    cur_train, cur_valid, cur_score = _evaluate_ruleset(train_engine, valid_engine, rules)
    best_rules = [dict(r) for r in rules]

    use_walk_forward = fold_engines is not None and len(fold_engines) > 1

    hist: list[dict] = [{
        "pass": 0,
        "rule_index": -1,
        "score": cur_score,
        "train_return_pct": _f(cur_train, "total_return_pct"),
        "valid_return_pct": _f(cur_valid, "total_return_pct"),
        "train_pf": _f(cur_train, "profit_factor"),
        "valid_pf": _f(cur_valid, "profit_factor"),
    }]
    if use_walk_forward:
        # Compute initial fold scores
        init_fold_scores: list[float] = []
        for fold_engine in fold_engines:
            _, fold_m, fold_s = _evaluate_ruleset(train_engine, fold_engine, rules)
            init_fold_scores.append(fold_s)
        hist[0]["fold_scores"] = init_fold_scores
        hist[0]["min_fold_score"] = min(init_fold_scores)
        # fix(task-3): recompute cur_score as min(init_fold_scores) so the
        # improvement threshold compares fold-min against fold-min (same scale).
        # Without this, cur_score remains the full-validation score (higher),
        # and walk-forward becomes overly conservative.
        cur_score = min(init_fold_scores)
        hist[0]["score"] = cur_score

    tp_grid = tuple(float(x) for x in getattr(_cfg, "RB_TP_GRID", getattr(_cfg, "PHASE4_TP_GRID", (1.5, 2.0, 2.5, 3.0))))
    sl_grid = tuple(float(x) for x in getattr(_cfg, "RB_SL_GRID", getattr(_cfg, "PHASE4_SL_GRID", (0.8, 1.0, 1.2, 1.5))))
    cap_grid = tuple(float(x) for x in getattr(_cfg, "RB_CAPITAL_GRID", getattr(_cfg, "PHASE4_CAPITAL_GRID", (5.0, 7.5, 10.0, 12.5))))
    max_total_cap = float(getattr(_cfg, "RB_MAX_TOTAL_CAPITAL", 65.0))
    passes = int(getattr(_cfg, "RB_RISK_OPT_PASSES", 2))
    min_improve = float(getattr(_cfg, "RB_RISK_MIN_IMPROVEMENT", 0.02))
    min_train_trades = int(getattr(_cfg, "RB_RULESET_MIN_TRAIN_TRADES", getattr(_cfg, "RB_MIN_TRAIN_TRADES", 25)))
    min_valid_trades = int(getattr(_cfg, "RB_RULESET_MIN_VALID_TRADES", getattr(_cfg, "RB_MIN_VALID_TRADES", 15)))

    for p in range(1, passes + 1):
        improved = False
        for idx in range(len(best_rules)):
            local_best: tuple[float, list[dict], dict, dict, list[float] | None] | None = None
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
                        if not _is_positive_good(train_m, valid_m, min_train_trades=min_train_trades, min_valid_trades=min_valid_trades):
                            continue

                        if use_walk_forward:
                            # Score on each fold engine and compute min fold score
                            fold_scores_local: list[float] = []
                            all_folds_pass = True
                            for fold_engine in fold_engines:
                                _, fold_m, fold_s = _evaluate_ruleset(train_engine, fold_engine, trial)
                                if not _is_positive_good(train_m, fold_m, min_train_trades=min_train_trades, min_valid_trades=min_valid_trades):
                                    all_folds_pass = False
                                    break
                                fold_scores_local.append(fold_s)
                            if not all_folds_pass:
                                continue
                            selection_score = min(fold_scores_local)
                        else:
                            selection_score = score
                            fold_scores_local = None

                        if local_best is None or selection_score > local_best[0]:
                            local_best = (selection_score, trial, train_m, valid_m, fold_scores_local)
            if local_best is not None and local_best[0] > cur_score + min_improve:
                cur_score, best_rules, cur_train, cur_valid, fold_scores_improved = local_best
                improved = True
                entry: dict[str, Any] = {
                    "pass": p,
                    "rule_index": idx + 1,
                    "score": cur_score,
                    "train_return_pct": _f(cur_train, "total_return_pct"),
                    "valid_return_pct": _f(cur_valid, "total_return_pct"),
                    "train_pf": _f(cur_train, "profit_factor"),
                    "valid_pf": _f(cur_valid, "profit_factor"),
                    "train_dd": _f(cur_train, "max_drawdown_pct"),
                    "valid_dd": _f(cur_valid, "max_drawdown_pct"),
                    "tp": best_rules[idx]["tp"],
                    "sl": best_rules[idx]["sl"],
                    "capital_pct": best_rules[idx]["capital_pct"],
                }
                if use_walk_forward and fold_scores_improved is not None:
                    entry["fold_scores"] = fold_scores_improved
                    entry["min_fold_score"] = min(fold_scores_improved)
                hist.append(entry)
                logger.info(
                    "RB [%s]: risk improve pass=%d rule=%d score=%.2f train=%.2f%% valid=%.2f%%",
                    direction, p, idx + 1, cur_score, _f(cur_train, "total_return_pct"), _f(cur_valid, "total_return_pct"),
                )
        if not improved:
            break

    # Tail holdout scoring on final selected combo (not used during search)
    if tail_holdout_engine is not None and hist:
        _, tail_m, _ = _evaluate_ruleset(train_engine, tail_holdout_engine, best_rules)
        final_entry = hist[-1]
        final_entry["risk_tail_holdout_return_pct"] = _f(tail_m, "total_return_pct")
        final_entry["risk_tail_holdout_pf"] = _f(tail_m, "profit_factor")
        final_entry["risk_tail_holdout_dd"] = _f(tail_m, "max_drawdown_pct")

    return best_rules, cur_train, cur_valid, cur_score, hist


def _profit_amp_objective(train_m: dict, valid_m: dict) -> float:
    """Return the profit-first objective after a ruleset has passed robustness gates."""
    train_ret = _f(train_m, "total_return_pct")
    valid_ret = _f(valid_m, "total_return_pct")
    valid_w = float(getattr(_cfg, "RB_PROFIT_AMP_VALID_WEIGHT", 1.55))
    train_w = float(getattr(_cfg, "RB_PROFIT_AMP_TRAIN_WEIGHT", 1.00))
    balance_w = float(getattr(_cfg, "RB_PROFIT_AMP_BALANCE_WEIGHT", 0.20))
    dd_w = float(getattr(_cfg, "RB_PROFIT_AMP_DD_WEIGHT", 0.02))
    health_w = float(getattr(_cfg, "RB_PROFIT_AMP_HEALTH_WEIGHT", 0.030))
    score = train_w * train_ret + valid_w * valid_ret + balance_w * min(train_ret, valid_ret)
    score -= dd_w * (_f(train_m, "max_drawdown_pct", 100.0) + 1.35 * _f(valid_m, "max_drawdown_pct", 100.0))
    score -= health_w * (_evaluator_health_penalty(train_m, role="train") + _evaluator_health_penalty(valid_m, role="valid"))
    return float(score)


def _profit_amp_monthly_summary(valid_df: pd.DataFrame, rules: list[dict], direction: str) -> tuple[MonthlyWindowSummary | None, list[dict]]:
    """Evaluate a ruleset on validation-only chronological windows for the amplifier certificate."""
    if not bool(getattr(_cfg, "RB_PROFIT_AMP_MONTHLY_ENABLED", True)):
        return None, []
    windows = build_monthly_windows(valid_df)
    min_windows = int(getattr(_cfg, "RB_PROFIT_AMP_MIN_MONTHLY_WINDOWS", 2))
    if len(windows) < min_windows:
        return None, []
    metrics: list[dict] = []
    formatted = [_rule_to_engine(r) for r in rules]
    for part in windows:
        try:
            prepared = _prepare_scoring_frame(part)
            engine = CPUBacktestEngine(prepared, {}, direction)
            metrics.append(engine.simulate_rule_set(formatted))
        except Exception:
            metrics.append({
                "total_return_pct": -100.0,
                "profit_factor": 0.0,
                "max_drawdown_pct": 100.0,
                "executed_trades": 0,
            })
    return summarize_monthly_metrics(metrics, n_rows=len(valid_df)), metrics


def _profit_amp_certificate(
    train_m: dict,
    valid_m: dict,
    monthly_summary: MonthlyWindowSummary | None = None,
) -> tuple[bool, dict]:
    """Return whether a ruleset is robust enough to enter profit-first selection."""
    min_train_trades = int(getattr(_cfg, "RB_RULESET_MIN_TRAIN_TRADES", getattr(_cfg, "RB_MIN_TRAIN_TRADES", 25)))
    min_valid_trades = int(getattr(_cfg, "RB_RULESET_MIN_VALID_TRADES", getattr(_cfg, "RB_MIN_VALID_TRADES", 15)))
    ok = _is_positive_good(train_m, valid_m, min_train_trades=min_train_trades, min_valid_trades=min_valid_trades)
    reasons: list[str] = []
    if not ok:
        reasons.append("full_sample_positive_good_failed")
    if _f(valid_m, "max_drawdown_pct", 100.0) > float(getattr(_cfg, "RB_PROFIT_AMP_MAX_VALID_DD", 12.0)):
        ok = False
        reasons.append("valid_drawdown_too_high")
    if _f(train_m, "max_drawdown_pct", 100.0) > float(getattr(_cfg, "RB_PROFIT_AMP_MAX_TRAIN_DD", 18.0)):
        ok = False
        reasons.append("train_drawdown_too_high")
    if monthly_summary is not None and monthly_summary.windows >= int(getattr(_cfg, "RB_PROFIT_AMP_MIN_MONTHLY_WINDOWS", 2)):
        if monthly_summary.profitable_ratio < float(getattr(_cfg, "RB_PROFIT_AMP_MIN_MONTHLY_PROFITABLE_RATIO", 0.55)):
            ok = False
            reasons.append("monthly_profitable_ratio_low")
        if monthly_summary.worst_return_pct < float(getattr(_cfg, "RB_PROFIT_AMP_WORST_MONTHLY_RETURN_FLOOR", -2.0)):
            ok = False
            reasons.append("monthly_worst_return_low")
        if monthly_summary.worst_profit_factor < float(getattr(_cfg, "RB_PROFIT_AMP_WORST_MONTHLY_PF_FLOOR", 0.80)):
            ok = False
            reasons.append("monthly_worst_pf_low")
        if monthly_summary.worst_drawdown_pct > float(getattr(_cfg, "RB_PROFIT_AMP_MAX_MONTHLY_DD", 10.0)):
            ok = False
            reasons.append("monthly_drawdown_too_high")
    detail = {
        "passed": bool(ok),
        "reasons": reasons,
        "train_return_pct": _f(train_m, "total_return_pct"),
        "valid_return_pct": _f(valid_m, "total_return_pct"),
        "train_profit_factor": _f(train_m, "profit_factor"),
        "valid_profit_factor": _f(valid_m, "profit_factor"),
        "train_drawdown_pct": _f(train_m, "max_drawdown_pct"),
        "valid_drawdown_pct": _f(valid_m, "max_drawdown_pct"),
        "train_trades": _i(train_m, "executed_trades"),
        "valid_trades": _i(valid_m, "executed_trades"),
    }
    if monthly_summary is not None:
        detail["monthly"] = asdict(monthly_summary)
    return bool(ok), detail


def _profit_amp_evaluate_candidate(
    train_engine: CPUBacktestEngine,
    valid_engine: CPUBacktestEngine,
    valid_df: pd.DataFrame,
    rules: list[dict],
    direction: str,
    *,
    check_monthly: bool = True,
) -> tuple[dict, dict, float, tuple[bool, dict], list[dict]]:
    """Evaluate a trial ruleset and attach the hard trust certificate used by the amplifier."""
    train_m, valid_m, _ = _evaluate_ruleset(train_engine, valid_engine, rules)
    monthly_summary, monthly_rows = _profit_amp_monthly_summary(valid_df, rules, direction) if check_monthly else (None, [])
    certificate = _profit_amp_certificate(train_m, valid_m, monthly_summary)
    objective = _profit_amp_objective(train_m, valid_m)
    return train_m, valid_m, objective, certificate, monthly_rows


def _records_from_rules(
    rules: list[dict],
    train_engine: CPUBacktestEngine,
    valid_engine: CPUBacktestEngine,
    train_like_df: pd.DataFrame,
    valid_df: pd.DataFrame,
) -> list[CandidateRecord]:
    """Convert existing strategy rules into candidate records for profit amplification."""
    records: list[CandidateRecord] = []
    seen: set[tuple[str, ...]] = set()
    for rule in rules:
        formatted = _rule_to_engine(rule)
        key = _rule_key(formatted)
        if key in seen:
            continue
        seen.add(key)
        try:
            train_m = train_engine.simulate_rule_set([formatted])
            valid_m = valid_engine.simulate_rule_set([formatted])
        except Exception:
            continue
        rec = CandidateRecord(rule=formatted, train_metrics=train_m, valid_metrics=valid_m, score=_score_metrics(train_m, valid_m))
        rec.mask = _mask_for(formatted, train_like_df, valid_df)
        records.append(rec)
    return records


def _profit_amp_rank_candidates(
    candidates: list[CandidateRecord],
    baseline_rules: list[dict],
    train_engine: CPUBacktestEngine,
    valid_engine: CPUBacktestEngine,
    train_like_df: pd.DataFrame,
    valid_df: pd.DataFrame,
) -> list[CandidateRecord]:
    """Build the bounded candidate list used by the profit amplifier."""
    out: list[CandidateRecord] = []
    seen: set[tuple[str, ...]] = set()
    for rec in candidates:
        rule = _rule_to_engine(rec.rule)
        key = _rule_key(rule)
        if key in seen:
            continue
        seen.add(key)
        rec.rule = rule
        if rec.mask is None:
            rec.mask = _mask_for(rule, train_like_df, valid_df)
        out.append(rec)
    for rec in _records_from_rules(baseline_rules, train_engine, valid_engine, train_like_df, valid_df):
        key = _rule_key(rec.rule)
        if key not in seen:
            seen.add(key)
            out.append(rec)
    out.sort(key=lambda r: _profit_amp_objective(r.train_metrics, r.valid_metrics), reverse=True)
    limit = int(getattr(_cfg, "RB_PROFIT_AMP_MAX_CANDIDATES", 90))
    return out[:limit]


def _profit_amp_select_rules(
    candidates: list[CandidateRecord],
    train_engine: CPUBacktestEngine,
    valid_engine: CPUBacktestEngine,
    train_like_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    direction: str,
) -> tuple[list[dict], dict, dict, float, tuple[bool, dict], list[dict], list[dict]] | None:
    """Greedily select rules by marginal profit while keeping every trial certificate-safe."""
    max_rules = int(_cfg.RB_MAX_RULES)
    max_overlap = float(getattr(_cfg, "RB_PROFIT_AMP_MAX_PAIR_OVERLAP", 0.55))
    min_improve = float(getattr(_cfg, "RB_PROFIT_AMP_MIN_OBJECTIVE_IMPROVEMENT", 0.10))
    min_return_improve = float(getattr(_cfg, "RB_PROFIT_AMP_MIN_RETURN_IMPROVEMENT", 0.05))
    overlap_penalty = float(getattr(_cfg, "RB_PROFIT_AMP_OVERLAP_PENALTY", 2.5))
    ranked = sorted(candidates, key=lambda r: _profit_amp_objective(r.train_metrics, r.valid_metrics), reverse=True)
    seed_best: tuple[float, CandidateRecord, dict, dict, float, tuple[bool, dict], list[dict]] | None = None
    for cand in ranked:
        train_m, valid_m, objective, cert, monthly_rows = _profit_amp_evaluate_candidate(train_engine, valid_engine, valid_df, [cand.rule], direction, check_monthly=False)
        if not cert[0]:
            continue
        if seed_best is None or objective > seed_best[0]:
            seed_best = (objective, cand, train_m, valid_m, objective, cert, monthly_rows)
    if seed_best is None:
        return None
    cur_objective, seed, cur_train, cur_valid, _, cur_cert, cur_monthly_rows = seed_best
    selected = [seed]
    used = {_rule_key(seed.rule)}
    history = [{
        "step": 1,
        "action": "seed_by_profit_certificate",
        "profit_amp_objective": cur_objective,
        "train_return_pct": _f(cur_train, "total_return_pct"),
        "valid_return_pct": _f(cur_valid, "total_return_pct"),
        "valid_profit_factor": _f(cur_valid, "profit_factor"),
        "valid_drawdown_pct": _f(cur_valid, "max_drawdown_pct"),
        "rules": 1,
        "certificate": cur_cert[1],
    }]
    while len(selected) < max_rules:
        best: tuple[float, CandidateRecord, dict, dict, float, tuple[bool, dict], list[dict], float] | None = None
        cur_return_sum = _f(cur_train, "total_return_pct") + _f(cur_valid, "total_return_pct")
        for cand in ranked:
            key = _rule_key(cand.rule)
            if key in used:
                continue
            ov = _max_overlap(cand, selected)
            if ov > max_overlap:
                continue
            trial_rules = [r.rule for r in selected] + [cand.rule]
            train_m, valid_m, objective, cert, monthly_rows = _profit_amp_evaluate_candidate(train_engine, valid_engine, valid_df, trial_rules, direction, check_monthly=False)
            if not cert[0]:
                continue
            trial_return_sum = _f(train_m, "total_return_pct") + _f(valid_m, "total_return_pct")
            if objective <= cur_objective + min_improve:
                continue
            if trial_return_sum <= cur_return_sum + min_return_improve:
                continue
            choose_score = objective - cur_objective - overlap_penalty * ov
            if best is None or choose_score > best[0]:
                best = (choose_score, cand, train_m, valid_m, objective, cert, monthly_rows, ov)
        if best is None:
            break
        _, cand, cur_train, cur_valid, cur_objective, cur_cert, cur_monthly_rows, ov = best
        selected.append(cand)
        used.add(_rule_key(cand.rule))
        history.append({
            "step": len(selected),
            "action": "add_by_marginal_profit_certificate",
            "profit_amp_objective": cur_objective,
            "train_return_pct": _f(cur_train, "total_return_pct"),
            "valid_return_pct": _f(cur_valid, "total_return_pct"),
            "valid_profit_factor": _f(cur_valid, "profit_factor"),
            "valid_drawdown_pct": _f(cur_valid, "max_drawdown_pct"),
            "max_overlap": ov,
            "rules": len(selected),
            "certificate": cur_cert[1],
        })
    final_rules = [r.rule for r in selected]
    cur_train, cur_valid, cur_objective, cur_cert, cur_monthly_rows = _profit_amp_evaluate_candidate(train_engine, valid_engine, valid_df, final_rules, direction, check_monthly=True)
    return final_rules, cur_train, cur_valid, cur_objective, cur_cert, history, cur_monthly_rows


def _profit_amp_reallocate_capital(
    rules: list[dict],
    train_engine: CPUBacktestEngine,
    valid_engine: CPUBacktestEngine,
    valid_df: pd.DataFrame,
    direction: str,
) -> tuple[list[dict], dict, dict, float, tuple[bool, dict], list[dict], list[dict]]:
    """Shift capital toward profit-contributing certified rules without changing rule conditions."""
    best_rules = [_rule_to_engine(r) for r in rules]
    cur_train, cur_valid, cur_objective, cur_cert, cur_monthly_rows = _profit_amp_evaluate_candidate(train_engine, valid_engine, valid_df, best_rules, direction)
    history = [{
        "pass": 0,
        "rule_index": -1,
        "profit_amp_objective": cur_objective,
        "train_return_pct": _f(cur_train, "total_return_pct"),
        "valid_return_pct": _f(cur_valid, "total_return_pct"),
        "certificate": cur_cert[1],
    }]
    if not cur_cert[0]:
        return best_rules, cur_train, cur_valid, cur_objective, cur_cert, history, cur_monthly_rows
    cap_grid = tuple(float(x) for x in getattr(_cfg, "RB_PROFIT_AMP_CAPITAL_GRID", getattr(_cfg, "RB_CAPITAL_GRID", (5.0, 12.5, 25.0))))
    max_total_cap = float(getattr(_cfg, "RB_MAX_TOTAL_CAPITAL", 95.0))
    passes = int(getattr(_cfg, "RB_PROFIT_AMP_CAPITAL_PASSES", 2))
    min_improve = float(getattr(_cfg, "RB_PROFIT_AMP_MIN_OBJECTIVE_IMPROVEMENT", 0.10))
    for pno in range(1, passes + 1):
        improved = False
        for idx in range(len(best_rules)):
            local: tuple[float, list[dict], dict, dict, tuple[bool, dict], list[dict]] | None = None
            for cap in cap_grid:
                trial = [dict(r) for r in best_rules]
                trial[idx]["capital_pct"] = cap
                if sum(float(r.get("capital_pct", 0.0)) for r in trial) > max_total_cap:
                    continue
                train_m, valid_m, objective, cert, monthly_rows = _profit_amp_evaluate_candidate(train_engine, valid_engine, valid_df, trial, direction, check_monthly=False)
                if not cert[0]:
                    continue
                if local is None or objective > local[0]:
                    local = (objective, trial, train_m, valid_m, cert, monthly_rows)
            if local is not None and local[0] > cur_objective + min_improve:
                cur_objective, best_rules, cur_train, cur_valid, cur_cert, cur_monthly_rows = local
                improved = True
                history.append({
                    "pass": pno,
                    "rule_index": idx + 1,
                    "profit_amp_objective": cur_objective,
                    "train_return_pct": _f(cur_train, "total_return_pct"),
                    "valid_return_pct": _f(cur_valid, "total_return_pct"),
                    "valid_profit_factor": _f(cur_valid, "profit_factor"),
                    "valid_drawdown_pct": _f(cur_valid, "max_drawdown_pct"),
                    "capital_pct": best_rules[idx]["capital_pct"],
                    "certificate": cur_cert[1],
                })
        if not improved:
            break
    cur_train, cur_valid, cur_objective, cur_cert, cur_monthly_rows = _profit_amp_evaluate_candidate(train_engine, valid_engine, valid_df, best_rules, direction, check_monthly=True)
    return best_rules, cur_train, cur_valid, cur_objective, cur_cert, history, cur_monthly_rows


def _run_profit_amplifier(
    baseline_rules: list[dict],
    baseline_train: dict,
    baseline_valid: dict,
    candidates: list[CandidateRecord],
    train_engine: CPUBacktestEngine,
    valid_engine: CPUBacktestEngine,
    train_like_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    direction: str,
) -> tuple[list[dict], dict, dict, float, dict]:
    """Run robust-certified profit amplification and keep the baseline unless profit improves."""
    baseline_monthly, baseline_monthly_rows = _profit_amp_monthly_summary(valid_df, baseline_rules, direction)
    baseline_cert = _profit_amp_certificate(baseline_train, baseline_valid, baseline_monthly)
    baseline_objective = _profit_amp_objective(baseline_train, baseline_valid)
    meta = {
        "enabled": bool(getattr(_cfg, "RB_PROFIT_AMPLIFIER_ENABLED", True)),
        "accepted": False,
        "baseline_profit_amp_objective": baseline_objective,
        "baseline_certificate": baseline_cert[1],
        "baseline_monthly_metrics": baseline_monthly_rows,
        "reason": "disabled_or_not_improved",
    }
    if not bool(getattr(_cfg, "RB_PROFIT_AMPLIFIER_ENABLED", True)):
        meta["reason"] = "disabled"
        return baseline_rules, baseline_train, baseline_valid, baseline_objective, meta
    ranked_candidates = _profit_amp_rank_candidates(candidates, baseline_rules, train_engine, valid_engine, train_like_df, valid_df)
    selection = _profit_amp_select_rules(ranked_candidates, train_engine, valid_engine, train_like_df, valid_df, direction)
    if selection is None:
        meta["reason"] = "no_certified_profit_seed"
        return baseline_rules, baseline_train, baseline_valid, baseline_objective, meta
    amp_rules, amp_train, amp_valid, amp_objective, amp_cert, select_history, amp_monthly_rows = selection
    capital_history: list[dict] = []
    if bool(getattr(_cfg, "RB_PROFIT_AMP_CAPITAL_REALLOCATION_ENABLED", True)):
        amp_rules, amp_train, amp_valid, amp_objective, amp_cert, capital_history, amp_monthly_rows = _profit_amp_reallocate_capital(amp_rules, train_engine, valid_engine, valid_df, direction)
    min_improve = float(getattr(_cfg, "RB_PROFIT_AMP_MIN_OBJECTIVE_IMPROVEMENT", 0.10))
    keep_baseline = bool(getattr(_cfg, "RB_PROFIT_AMP_KEEP_BASELINE_UNLESS_BETTER", True))
    if keep_baseline and amp_objective <= baseline_objective + min_improve:
        meta.update({
            "reason": "amplified_not_better_than_baseline",
            "amplified_profit_amp_objective": amp_objective,
            "amplified_certificate": amp_cert[1],
            "selection_history": select_history,
            "capital_history": capital_history,
            "amplified_monthly_metrics": amp_monthly_rows,
        })
        return baseline_rules, baseline_train, baseline_valid, baseline_objective, meta
    if not amp_cert[0]:
        meta.update({
            "reason": "amplified_certificate_failed",
            "amplified_profit_amp_objective": amp_objective,
            "amplified_certificate": amp_cert[1],
            "selection_history": select_history,
            "capital_history": capital_history,
            "amplified_monthly_metrics": amp_monthly_rows,
        })
        return baseline_rules, baseline_train, baseline_valid, baseline_objective, meta
    meta.update({
        "accepted": True,
        "reason": "profit_objective_improved_after_certificate",
        "amplified_profit_amp_objective": amp_objective,
        "amplified_certificate": amp_cert[1],
        "selection_history": select_history,
        "capital_history": capital_history,
        "amplified_monthly_metrics": amp_monthly_rows,
    })
    return amp_rules, amp_train, amp_valid, amp_objective, meta


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
    rules = _enforce_capital_budget(rules)
    _assert_capital_budget(rules)
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
    cv_folds: list | None = None,
    val_selection_df: pd.DataFrame | None = None,
) -> dict[str, dict]:
    """Build and optimize rb strategies for each direction and write outputs."""
    out_dir = Path(output_dir or _cfg.OUTPUTS_DIR)
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_like, _ = _load_scoring_frames(train_df, val_df)
    scoring_val = val_selection_df if val_selection_df is not None else val_df
    _, valid_df = _load_scoring_frames(train_df, scoring_val)
    results: dict[str, dict] = {}

    for direction in directions:
        pool = pools.get(direction, [])
        if not pool:
            logger.warning("RB [%s]: empty Phase 2 pool; skipping.", direction)
            continue
        train_engine = CPUBacktestEngine(train_like, {}, direction)
        valid_engine = CPUBacktestEngine(valid_df, {}, direction)

        # Build per-fold engines for CV-fold consistency (C4)
        fold_engines: list[CPUBacktestEngine] | None = None
        if cv_folds:
            from gpu_fuzzy_trader.validation.rolling_cv import cv_folds_only
            try:
                fold_engines = [
                    CPUBacktestEngine(fold.valid_df, {}, direction)
                    for fold in cv_folds_only(cv_folds)
                ]
            except Exception:
                logger.warning("RB [%s]: failed to build CV-fold engines; skipping CV term.", direction)
                fold_engines = None

        candidates = _filter_good_rules(pool, train_like, valid_df, direction, fold_engines=fold_engines)
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
                    # Evaluate on CV folds if available (C4)
                    cv_fold_returns = _eval_cv_fold_returns(rule, fold_engines)
                    rec = CandidateRecord(rule=rule, train_metrics=tr, valid_metrics=te, score=_score_metrics(tr, te, cv_fold_returns=cv_fold_returns))
                    rec.mask = _mask_for(rule, train_like, valid_df)
                    candidates.append(rec)
            candidates.sort(key=lambda r: r.score, reverse=True)
            candidates = candidates[: int(getattr(_cfg, "RB_KEEP_TOP_RULES", 120))]
            if not candidates:
                continue

        selected, sel_train, sel_test, sel_score, compose_history = _compose_ruleset(candidates, train_engine, valid_engine, direction)

        # Build walk-forward fold engines for risk grid (task-3: 2-fold)
        # → fixes audit finding #3 (RB Governor risk-grid overfits val_selection)
        # → fixes audit finding #12 (PHASE4_TAIL_HOLDOUT_FRACTION orphan, now wired)
        wf_splits = int(getattr(_cfg, "RB_RISK_GRID_WF_SPLITS", 1))
        use_tail = bool(getattr(_cfg, "RB_RISK_GRID_USE_TAIL_HOLDOUT", False))
        tail_frac = float(getattr(_cfg, "PHASE4_TAIL_HOLDOUT_FRACTION", 0.0))
        wf_fold_engines: list[CPUBacktestEngine] | None = None
        wf_tail_engine: CPUBacktestEngine | None = None
        if wf_splits > 1 or use_tail:
            wf_fold_engines, wf_tail_engine = _make_walk_forward_fold_engines(
                scoring_val, wf_splits, tail_frac if use_tail else 0.0, direction,
            )

        opt_rules, opt_train, opt_test, opt_score, risk_history = _optimize_risk(
            selected, train_engine, valid_engine, direction,
            fold_engines=wf_fold_engines, tail_holdout_engine=wf_tail_engine,
        )
        profit_rules, profit_train, profit_test, profit_objective, profit_meta = _run_profit_amplifier(
            opt_rules,
            opt_train,
            opt_test,
            candidates,
            train_engine,
            valid_engine,
            train_like,
            valid_df,
            direction,
        )
        opt_rules = profit_rules
        opt_train = profit_train
        opt_test = profit_test
        opt_score = _score_metrics(
            opt_train,
            opt_test,
            min_train_trades=int(getattr(_cfg, "RB_RULESET_MIN_TRAIN_TRADES", getattr(_cfg, "RB_MIN_TRAIN_TRADES", 25))),
            min_valid_trades=int(getattr(_cfg, "RB_RULESET_MIN_VALID_TRADES", getattr(_cfg, "RB_MIN_VALID_TRADES", 15))),
        )

        val_ret = _f(opt_test, "total_return_pct")
        val_pf = _f(opt_test, "profit_factor")
        ret_gate = float(_cfg.PHASE5_VALIDATION_RETURN_GATE_PCT)
        pf_gate = float(_cfg.PHASE5_VALIDATION_PROFIT_FACTOR_GATE)
        sym_ok, sym_gate = _passes_symbol_concentration_gate(opt_test)
        tail_ok, tail_gate = _passes_tail_holdout_gate(risk_history)
        deployable = (
            val_ret >= (ret_gate - 1e-9)
            and val_pf >= (pf_gate - 1e-9)
            and sym_ok
            and tail_ok
        )
        if not sym_ok:
            logger.warning(
                "RB [%s]: symbol-concentration gate failed "
                "(top_share=%.3f hhi=%.3f top=%s)",
                direction,
                sym_gate["top_symbol_share_abs_pnl"],
                sym_gate["hhi_abs_pnl"],
                sym_gate["top_symbol"],
            )
        if not tail_ok:
            logger.warning(
                "RB [%s]: tail-holdout hard gate failed "
                "(tail_ret=%.2f%% min=%.2f%%)",
                direction,
                float(tail_gate.get("tail_return_pct", 0.0)),
                float(tail_gate.get("min_return_pct", 0.0)),
            )

        strategy = _strategy(
            direction,
            opt_rules,
            risk_optimized=bool(deployable),
            extra={
                "deployment_accepted": bool(deployable),
                "validation_gate": {
                    "return_pct": val_ret,
                    "profit_factor": val_pf,
                    "required_return_pct": ret_gate,
                    "required_profit_factor": pf_gate,
                },
                "symbol_concentration_gate": sym_gate,
                "tail_holdout_gate": tail_gate,
                "rb_score": opt_score,
                "rb_train_return_pct": _f(opt_train, "total_return_pct"),
                "rb_valid_return_pct": val_ret,
                "rb_valid_profit_factor": val_pf,
                "rb_valid_max_drawdown_pct": _f(opt_test, "max_drawdown_pct"),
                "rb_valid_executed_trades": _i(opt_test, "executed_trades"),
                "rb_train_minus_valid_return_pct": _f(opt_train, "total_return_pct") - val_ret,
                "rb_train_valid_ratio": _f(opt_train, "total_return_pct") / max(val_ret, 1e-9),
                "rb_profit_amp_objective": profit_objective,
                "rb_profit_amp_accepted": bool(profit_meta.get("accepted", False)),
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
            "profit_amplifier": profit_meta,
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
            "RB [%s]: saved %d-rule strategy | score=%.2f train=%.2f%% valid=%.2f%% pf=%.2f dd=%.2f%% -> %s",
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
            if bool(getattr(_cfg, "RB_GLOBAL_REQUIRE_POSITIVE_TRAIN_VALID", True)) and not _positive_returns(tr, te):
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
    cur_train, cur_valid, _ = _evaluate_ruleset(train_engine, valid_engine, [selected[0].rule])
    cur_score = _combined_return_score(cur_train, cur_valid, prev_pf=None, prev_dd=None)
    used = {_rule_key(selected[0].rule)}
    history = [{
        "step": 1,
        "action": "seed",
        "combined_return_score": cur_score,
        "train_return_pct": _f(cur_train, "total_return_pct"),
        "valid_return_pct": _f(cur_valid, "total_return_pct"),
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
            prev_pf = _f(cur_valid, "profit_factor", 0.0)
            prev_dd = _f(cur_valid, "max_drawdown_pct", 0.0)
            ret_score = _combined_return_score(train_m, valid_m, prev_pf=prev_pf, prev_dd=prev_dd)
            if ret_score <= cur_score + min_improve:
                continue
            if best is None or ret_score > best[0]:
                best = (ret_score, cand, train_m, valid_m)
        if best is None:
            logger.info("RB global [%s]: no further profit-improving rule found at %d rules.", direction, len(selected))
            break
        cur_score, cand, cur_train, cur_valid = best
        selected.append(cand)
        used.add(_rule_key(cand.rule))
        history.append({
            "step": len(selected),
            "action": "add_rule_by_profit",
            "combined_return_score": cur_score,
            "rb_score": _score_metrics(cur_train, cur_valid),
            "train_return_pct": _f(cur_train, "total_return_pct"),
            "valid_return_pct": _f(cur_valid, "total_return_pct"),
            "train_pf": _f(cur_train, "profit_factor"),
            "valid_pf": _f(cur_valid, "profit_factor"),
            "train_dd": _f(cur_train, "max_drawdown_pct"),
            "valid_dd": _f(cur_valid, "max_drawdown_pct"),
            "rules": len(selected),
        })
        logger.info("RB global [%s]: grew to %d rules by profit | train=%.2f%% valid=%.2f%% combined=%.2f",
                    direction, len(selected), _f(cur_train, "total_return_pct"), _f(cur_valid, "total_return_pct"), cur_score)
    return selected, cur_train, cur_valid, cur_score, history


def _optimize_risk_return_only(
    rules: list[dict],
    train_engine: CPUBacktestEngine,
    valid_engine: CPUBacktestEngine,
    direction: str,
) -> tuple[list[dict], dict, dict, float, list[dict]]:
    best_rules = [_rule_to_engine(r) for r in rules]
    cur_train, cur_valid, _ = _evaluate_ruleset(train_engine, valid_engine, best_rules)
    cur_score = _combined_return_score(cur_train, cur_valid, prev_pf=None, prev_dd=None)
    hist = [{
        "pass": 0,
        "rule_index": -1,
        "combined_return_score": cur_score,
        "train_return_pct": _f(cur_train, "total_return_pct"),
        "valid_return_pct": _f(cur_valid, "total_return_pct"),
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
                        prev_pf = _f(cur_valid, "profit_factor", 0.0)
                        prev_dd = _f(cur_valid, "max_drawdown_pct", 0.0)
                        ret_score = _combined_return_score(train_m, valid_m, prev_pf=prev_pf, prev_dd=prev_dd)
                        if local is None or ret_score > local[0]:
                            local = (ret_score, trial, train_m, valid_m)
            if local is not None and local[0] > cur_score + min_improve:
                cur_score, best_rules, cur_train, cur_valid = local
                improved = True
                hist.append({
                    "pass": pno,
                    "rule_index": idx + 1,
                    "combined_return_score": cur_score,
                    "train_return_pct": _f(cur_train, "total_return_pct"),
                    "valid_return_pct": _f(cur_valid, "total_return_pct"),
                    "train_pf": _f(cur_train, "profit_factor"),
                    "valid_pf": _f(cur_valid, "profit_factor"),
                    "train_dd": _f(cur_train, "max_drawdown_pct"),
                    "valid_dd": _f(cur_valid, "max_drawdown_pct"),
                    "tp": best_rules[idx]["tp"],
                    "sl": best_rules[idx]["sl"],
                    "capital_pct": best_rules[idx]["capital_pct"],
                })
        if not improved:
            break
    return best_rules, cur_train, cur_valid, cur_score, hist


def _load_internal_split_frames_for_rb() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the internal train and validation split without reading the final test file."""
    from gpu_fuzzy_trader.data.splitter import Data_Splitter, load_cached_split_if_fresh

    cached = load_cached_split_if_fresh()
    if cached is not None:
        train_df, val_df, _, _, _ = cached
        return train_df, val_df

    full = Data_Loader().load_dataset(_cfg.TRAIN_CSV_PATH)
    train_df, valid_df, _ = Data_Splitter().split_and_persist(full)
    return train_df, valid_df


def update_global_bank_and_compose(
    root: Path,
    out_dir: Path,
    direction: str,
    run_no: int | None = None,
    train_df: pd.DataFrame | None = None,
    val_df: pd.DataFrame | None = None,
) -> None:
    """Update RB rule bank from the current run and compose best_global.

    Unlike best/ (which stores the best single run), best_global/ is built from
    the accumulated bank of rules across all completed runs.  New rules are
    added mainly when the combined training and validation return improves.
    """
    if not bool(getattr(_cfg, "RB_GLOBAL_BANK_ENABLED", True)):
        return
    if train_df is None or val_df is None:
        train_df, val_df = _load_internal_split_frames_for_rb()
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
    profit_rules, profit_train, profit_valid, amp_objective, amp_meta = _run_profit_amplifier(
        rules,
        train_m,
        valid_m,
        candidates,
        train_engine,
        valid_engine,
        train_like,
        valid_df,
        direction,
    )
    rules = profit_rules
    train_m = profit_train
    valid_m = profit_valid
    profit_score = _combined_return_score(train_m, valid_m)
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
            "rb_profit_amp_objective": amp_objective,
            "rb_profit_amp_accepted": bool(amp_meta.get("accepted", False)),
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
        "profit_amplifier": amp_meta,
        "note": "Global RB rule-set composed from all previous run rules. Rule addition objective is combined training and validation return.",
    }
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    logger.info("RB global [%s]: new best_global %d rules combined=%.2f train=%.2f%% valid=%.2f%% -> %s",
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
