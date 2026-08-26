
from __future__ import annotations

import json
import logging
import math
import os
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
from gpu_fuzzy_trader.features.encoder import encode_condition, get_dont_care
from gpu_fuzzy_trader.scoring import return_to_drawdown, profit_factor_term
from gpu_fuzzy_trader.scoring.gates import (
    PositiveGoodThresholds,
    gate_positive_good,
    positive_good_reject_reasons,
)
from gpu_fuzzy_trader.validation.monthly_windows import (
    build_monthly_windows,
    summarize_monthly_metrics,
    MonthlyWindowSummary,
)
from gpu_fuzzy_trader.phases.rule_identity import (
    feature_conditions_only,
    phase2_rule_id,
    strategy_id,
)
from gpu_fuzzy_trader.phases.phase2_support import expectancy_lcb_pct
from gpu_fuzzy_trader.portfolio.clustering import (
    adjusted_quality,
    greedy_adjusted_quality,
    threshold_graph_clusters,
)
from gpu_fuzzy_trader.portfolio.redundancy import (
    redundancy_matrix,
    stable_corr,
)
from gpu_fuzzy_trader.portfolio.marginal import (
    effective_rule_count,
    marginal_contribution,
)

logger = logging.getLogger(__name__)


@dataclass
class CandidateRecord:
    rule: dict
    train_metrics: dict
    valid_metrics: dict
    score: float
    mask: np.ndarray | None = None
    # A recency candidate is allowed to have a bounded loss on the older
    # training regime when both chronological validation halves are positive.
    # Keep this marker explicit so normal RB composition/risk code never
    # silently treats a rescued rule as an ordinary positive-good rule.
    recency: bool = False
    recency_fitness_metrics: dict | None = None
    # Optional fold/return evidence used by correlation-aware composition.
    # Legacy callers may leave these fields unset.
    pnl_series: np.ndarray | None = None
    fold_returns: list[float] | None = None
    fold_pnl_series: list[np.ndarray] | None = None


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


def _passes_symbol_concentration_gate(
    valid_m: dict,
    *,
    max_share: float | None = None,
    max_hhi: float | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Hard gate: reject strategies dominated by one symbol on validation."""
    hhi, top_share, top_sym = _symbol_concentration_stats(valid_m)
    share_limit = float(
        getattr(_cfg, "RB_MAX_SYMBOL_SHARE_ABS_PNL", 0.50)
        if max_share is None else max_share
    )
    hhi_limit = float(
        getattr(_cfg, "RB_MAX_SYMBOL_HHI", 0.55)
        if max_hhi is None else max_hhi
    )
    ok = top_share <= share_limit + 1e-12 and hhi <= hhi_limit + 1e-12
    return ok, {
        "hhi_abs_pnl": hhi,
        "top_symbol_share_abs_pnl": top_share,
        "top_symbol": top_sym,
        "max_share": share_limit,
        "max_hhi": hhi_limit,
        "passed": bool(ok),
    }


def _passes_symbol_contribution_certificate(
    valid_m: dict | None,
    *,
    min_symbols: int | None = None,
    min_trades: int | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Require positive, supported validation PnL from multiple symbols.

    Symbol presence is not enough for a balanced team: a symbol must have
    positive net validation PnL and at least the configured validation trade
    floor.  The helper is strict about missing evidence; callers that support
    legacy test doubles can explicitly skip enforcement when the metrics field
    is absent.
    """
    required_symbols = int(
        min_symbols
        if min_symbols is not None
        else getattr(_cfg, "RB_MIN_DISTINCT_SYMBOLS", 2)
    )
    required_trades = int(
        min_trades
        if min_trades is not None
        else getattr(_cfg, "RB_MIN_VALID_TRADES", 6)
    )
    per_symbol = (valid_m or {}).get("per_symbol_metrics", {}) or {}
    available = isinstance(per_symbol, dict) and bool(per_symbol)
    qualifying: dict[str, dict[str, float | int]] = {}
    rejected: dict[str, str] = {}
    if isinstance(per_symbol, dict):
        for symbol, values in per_symbol.items():
            if not isinstance(values, dict):
                rejected[str(symbol)] = "invalid_metrics"
                continue
            symbol_name = str(symbol)
            try:
                trades = int(values.get("trade_count", 0))
                net_pnl = float(values.get("net_pnl", 0.0))
            except (TypeError, ValueError):
                rejected[symbol_name] = "invalid_metrics"
                continue
            if trades < required_trades:
                rejected[symbol_name] = "insufficient_validation_trades"
            elif net_pnl <= 0.0:
                rejected[symbol_name] = "non_positive_validation_pnl"
            else:
                qualifying[symbol_name] = {
                    "trade_count": trades,
                    "net_pnl": net_pnl,
                    "win_rate": float(values.get("win_rate", 0.0)),
                }
    missing_count = max(0, required_symbols - len(qualifying))
    ok = available and len(qualifying) >= required_symbols
    reasons: list[str] = []
    if not available:
        reasons.append("per_symbol_validation_metrics_missing")
    if missing_count:
        reasons.append("insufficient_positive_validation_symbols")
    return bool(ok), {
        "passed": bool(ok),
        "available": bool(available),
        "required_symbols": required_symbols,
        "min_validation_trades_per_symbol": required_trades,
        "qualifying_symbols": sorted(qualifying),
        "qualifying": qualifying,
        "rejected_symbols": rejected,
        "missing_symbol_count": missing_count,
        "reasons": reasons,
    }


def _portfolio_selection_certificate(
    valid_m: dict | None,
    *,
    min_symbols: int | None = None,
    concentration_max_share: float | None = None,
    concentration_max_hhi: float | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Return the certificate used by compose, risk, and profit selection."""
    metrics = valid_m or {}
    has_symbol_field = (
        isinstance(metrics, dict) and "per_symbol_metrics" in metrics
    )
    per_symbol = metrics.get("per_symbol_metrics", {}) or {}
    # Actual CPUBacktestEngine marks the field as available even when the
    # result has no trades.  Keep compatibility with lightweight legacy test
    # doubles that omit the availability marker, while treating an explicit
    # CPU ``False``/empty result as missing certificate evidence.
    evidence_available = bool(
        metrics.get("per_symbol_metrics_available", False)
    ) if isinstance(metrics, dict) else False
    if not has_symbol_field or (not per_symbol and not evidence_available):
        return True, {
            "passed": True,
            "available": False,
            "reason": "per_symbol_validation_metrics_unavailable",
            "symbol_contribution": {
                "passed": False,
                "available": False,
                "reasons": ["per_symbol_validation_metrics_missing"],
            },
        }
    contribution_ok, contribution = _passes_symbol_contribution_certificate(
        valid_m,
        min_symbols=min_symbols,
    )
    concentration_ok, concentration = _passes_symbol_concentration_gate(
        valid_m or {},
        max_share=concentration_max_share,
        max_hhi=concentration_max_hhi,
    )
    return contribution_ok and concentration_ok, {
        "passed": bool(contribution_ok and concentration_ok),
        "available": True,
        "symbol_contribution": contribution,
        "symbol_concentration": concentration,
        "reasons": [
            *([] if contribution_ok else ["symbol_contribution"]),
            *([] if concentration_ok else ["symbol_concentration"]),
        ],
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
    if final.get("risk_tail_holdout_error"):
        return False, {
            "enabled": True,
            "available": True,
            "passed": False,
            "reason": "tail_holdout_evaluation_error",
            "error": str(final["risk_tail_holdout_error"]),
        }
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


def _passes_tail_selection_gate(
    tail_holdout_engine: CPUBacktestEngine | None,
    rules: list[dict],
) -> tuple[bool, dict[str, Any]]:
    """Validate a trial ruleset on the reserved chronological validation tail.

    The tail is an inner validation split, never Phase 5 test data.  Making it
    part of composition and risk-grid feasibility prevents a post-hoc hard
    gate from rejecting a direction when a tail-positive alternative existed.
    """
    enabled = bool(getattr(_cfg, "RB_TAIL_HOLDOUT_SELECTION_GATE", True))
    if not enabled or tail_holdout_engine is None:
        return True, {"enabled": enabled, "available": False, "passed": True}
    try:
        tail_metrics = tail_holdout_engine.simulate_rule_set(
            [_rule_to_engine(rule) for rule in rules]
        )
    except Exception as exc:
        return False, {
            "enabled": True,
            "available": True,
            "passed": False,
            "reason": "tail_holdout_evaluation_error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    min_ret = float(getattr(_cfg, "RB_TAIL_HOLDOUT_MIN_RETURN_PCT", 0.0))
    min_trades = int(getattr(_cfg, "RB_TAIL_HOLDOUT_MIN_TRADES", 0))
    ret = _f(tail_metrics, "total_return_pct")
    trades = _i(tail_metrics, "executed_trades")
    ok = ret >= min_ret - 1e-12 and trades >= min_trades
    return bool(ok), {
        "enabled": True,
        "available": True,
        "passed": bool(ok),
        "tail_return_pct": ret,
        "tail_trades": trades,
        "min_return_pct": min_ret,
        "min_trades": min_trades,
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
    out = {
        "conditions": list(rule.get("conditions", [])),
        "tp": tp,
        "sl": sl,
        "capital_pct": float(rule.get("capital_pct", getattr(_cfg, "RB_DEFAULT_CAPITAL_PCT", 12.5))),
    }
    # Preserve Phase 2 provenance through every RB scoring/risk copy.  These
    # fields are metadata, not feature conditions, and therefore do not alter
    # evaluator matching semantics.
    feature_conditions = feature_conditions_only(out["conditions"])
    out["feature_conditions"] = list(
        rule.get("feature_conditions", feature_conditions)
    )
    out["phase2_rule_id"] = str(
        rule.get(
            "phase2_rule_id",
            phase2_rule_id(
                out["conditions"],
                direction=rule.get("direction"),
                source_symbols=rule.get("source_symbols", []),
            ),
        )
    )
    out["rule_id"] = str(rule.get("rule_id", out["phase2_rule_id"]))
    for key in (
        "source_symbols",
        "island_symbols",
        "origin_symbol",
        "migration_history",
        "eligible_symbols",
    ):
        if key in rule:
            out[key] = rule[key]
    return out


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


def _assert_mandatory_context(direction: str, rules: list[dict]) -> None:
    """Fail closed if the fixed trend-context conditions were lost (legacy only)."""
    if not bool(getattr(_cfg, "REQUIRE_CONTEXT_IN_STRATEGY", False)):
        return
    mandatory = _cfg.mandatory_context_conditions(direction)
    ctx_cols = set(_cfg.CONTEXT_COLUMNS)

    def _ctx_feature(cond: str) -> str | None:
        if not (cond.startswith("[") and " IS " in cond):
            return None
        feat = cond.split(" IS ", 1)[0][1:].strip()
        if feat.endswith("]"):
            feat = feat[:-1].rstrip()
        return feat if feat in ctx_cols else None

    for idx, rule in enumerate(rules, start=1):
        present = [str(c).strip() for c in rule.get("conditions", [])]
        seen: set[str] = set()
        for c in present:
            feat = _ctx_feature(c)
            if feat:
                if c in seen:
                    raise AssertionError(
                        f"Rule {idx}: duplicate mandatory context "
                        f"condition {c!r} after RB composition.")
                seen.add(c)
        if _cfg.REQUIRE_CONTEXT_IN_STRATEGY:
            for ctx_condition in mandatory:
                if ctx_condition not in present:
                    raise AssertionError(
                        f"Rule {idx}: mandatory context condition "
                        f"{ctx_condition!r} was lost during RB composition "
                        f"for direction {direction!r}.")


def _strategy(direction: str, rules: list[dict], *, risk_optimized: bool = False, extra: dict | None = None) -> dict:
    clean_rules = _enforce_capital_budget(rules)
    _assert_capital_budget(clean_rules)
    _assert_mandatory_context(direction, clean_rules)
    package_id = strategy_id(
        direction=direction,
        rules=clean_rules,
        horizon_bars=int(getattr(_cfg, "MAX_HOLD_CANDLES", 0)),
        cost_model_id=str(getattr(_cfg, "COST_MODEL_ID", "unknown")),
    )
    out = {
        "direction": direction,
        "rules_set": clean_rules,
        "risk_optimized": bool(risk_optimized),
        "strategy_id": package_id,
        "strategy_contract": {
            "identity_includes": [
                "entry_conditions",
                "eligible_symbols",
                "take_profit",
                "stop_loss",
                "horizon_bars",
                "cost_model_id",
            ],
            "horizon_bars": int(getattr(_cfg, "MAX_HOLD_CANDLES", 0)),
            "cost_model_id": str(getattr(_cfg, "COST_MODEL_ID", "unknown")),
            "capital_is_sizing_only": True,
        },
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
    train_lcb = expectancy_lcb_pct(train_m)
    valid_lcb = expectancy_lcb_pct(valid_m)
    valid_es = _f(valid_m, "expected_shortfall_pct", 0.0)

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
        + 12.0 * min(train_lcb, valid_lcb)
        - 0.75 * max(0.0, -valid_es)
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


def _traded_symbols_from_metrics(metrics: dict | None) -> set[str]:
    """Symbols with executed trades in backtest ``per_symbol_metrics``."""
    per_sym = (metrics or {}).get("per_symbol_metrics", {}) or {}
    out: set[str] = set()
    for sym, payload in per_sym.items():
        if not isinstance(payload, dict):
            continue
        trades = int(payload.get("trade_count", 0) or 0)
        if trades > 0:
            out.add(str(sym).strip().lower())
    return out


def _candidate_coverage_symbols(rec: CandidateRecord) -> set[str]:
    """Symbol coverage for compose diversity.

    Prefer **traded** symbols from backtest metrics. Island Mode A attaches
    ``source_symbols`` OR-filters (often 3–4 symbols) which previously made a
    single specialist look fully diversified for ``RB_MIN_DISTINCT_SYMBOLS``
    even when abs-PnL was concentrated on one name — then concentration
    fail-closed emptied the strategy. Fall back to explicit filters only when
    no trades were recorded.
    """
    traded = _traded_symbols_from_metrics(rec.train_metrics) | _traded_symbols_from_metrics(
        rec.valid_metrics
    )
    if traded:
        return traded
    explicit = _symbols_in_rules([rec.rule])
    return {s.lower() for s in explicit}


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


def _source_symbols_from_rule(rule: dict) -> list[str]:
    """Island/cluster symbols carried on Phase 2 pool entries."""
    raw = rule.get("source_symbols") or rule.get("island_symbols") or []
    out: list[str] = []
    seen: set[str] = set()
    for sym in raw:
        text = str(sym).strip()
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        out.append(text)
    return out


def _attach_source_symbol_filters(conditions: list[str], source_symbols: list[str]) -> list[str]:
    """Feature ANDs + island symbol ORs (engine ORs multiple ``symbol is X``)."""
    feats = _strip_symbol_conditions(list(conditions))
    if not source_symbols:
        return feats
    return feats + [_symbol_condition(s) for s in source_symbols]


def _ensure_symbol_filtered_rule(rule: dict, symbols: list[str]) -> dict:
    """Return rule with an explicit symbol filter when required.

    This is a safety net for output files.  Scoring normally specializes rules
    before evaluation.
    """
    out = dict(rule)
    conditions = list(out.get("conditions", []))
    if not bool(getattr(_cfg, "RB_REQUIRE_SYMBOL_FILTERS", False)):
        # Mode A: keep Phase-2 island scope when present; otherwise generalist.
        src = _source_symbols_from_rule(rule)
        out["conditions"] = _attach_source_symbol_filters(conditions, src)
        return out
    if any(_is_symbol_condition(str(c)) for c in conditions):
        out["conditions"] = conditions
        return out
    max_symbols = int(getattr(_cfg, "RB_SYMBOL_MAX_SYMBOLS_PER_RULE", 3))
    scoped = _source_symbols_from_rule(rule)
    use_pool = scoped or symbols
    use_symbols = use_pool[:max(1, max_symbols)] if use_pool else []
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

    Multi-symbol team mode (``RB_REQUIRE_SYMBOL_FILTERS=False``):
    - If the pool rule has ``source_symbols`` (Phase 2 island/cluster), attach
      those as OR filters so the rule only fires on the universe it was trained
      on. Bare generalists (no symbol scope) were deeply negative on full
      train+val_selection even when Phase 2 pool metrics looked healthy.
    - If no ``source_symbols``, strip orphan single-symbol filters and keep a
      true cross-symbol generalist.
    """
    base = _rule_to_engine(rule)
    if not bool(getattr(_cfg, "RB_REQUIRE_SYMBOL_FILTERS", False)):
        explicit_symbols = _symbols_in_rules([base])
        # Deterministic univariate baselines intentionally carry their scope.
        # Also preserve a matching filter when the supplied universe has only
        # that one symbol; in that case it is not an orphan filter.  Other
        # explicit filters are stripped so a generalist candidate cannot retain
        # an accidental scope from an incompatible island.
        baseline_source = str(rule.get("source", "")).strip().lower()
        sole_symbol_scope = (
            len(symbols) == 1
            and bool(explicit_symbols)
            and explicit_symbols.issubset({str(symbols[0]).strip().lower()})
        )
        if baseline_source == "rb_univariate_baseline" or sole_symbol_scope:
            # Preserve the caller's lightweight rule shape here.  The RB
            # candidate loop applies ``_rule_to_engine`` immediately after
            # specialization, while direct callers/tests may intentionally
            # compare the variant to the supplied dictionary.
            return [dict(rule)]
        src = _source_symbols_from_rule(rule)
        out = dict(base)
        out["conditions"] = _attach_source_symbol_filters(
            list(out.get("conditions", [])), src
        )
        if src:
            out["source_symbols"] = list(src)
        return [out]
    if _has_symbol_condition(base):
        return [base]
    if not symbols:
        return [base]

    base_conditions = _strip_symbol_conditions(list(base.get("conditions", [])))
    scoped_symbols = _source_symbols_from_rule(rule)
    if scoped_symbols:
        allowed = {str(sym).strip().lower() for sym in scoped_symbols}
        symbols = [
            sym for sym in symbols
            if str(sym).strip().lower() in allowed
        ] or symbols
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
    thresholds = PositiveGoodThresholds(
        min_train_return=float(getattr(_cfg, "RB_MIN_TRAIN_RETURN", 0.0)),
        min_valid_return=float(getattr(_cfg, "RB_MIN_VALID_RETURN", 0.0)),
        min_train_profit_factor=float(getattr(_cfg, "RB_MIN_TRAIN_PF", 1.0)),
        min_valid_profit_factor=float(getattr(_cfg, "RB_MIN_VALID_PF", 1.0)),
        min_train_trades=int(
            min_train_trades
            if min_train_trades is not None
            else getattr(_cfg, "RB_MIN_TRAIN_TRADES", 25)
        ),
        min_valid_trades=int(
            min_valid_trades
            if min_valid_trades is not None
            else getattr(_cfg, "RB_MIN_VALID_TRADES", 15)
        ),
        require_execution_health=bool(
            getattr(_cfg, "RB_REQUIRE_EXECUTION_HEALTH_ON_SINGLES", False)
        ),
    )
    if not gate_positive_good(train_m, valid_m, thresholds):
        return False
    lcb_floor = float(getattr(_cfg, "RB_EXPECTANCY_LCB_MARGIN_PCT", 0.0))
    return min(
        expectancy_lcb_pct(train_m),
        expectancy_lcb_pct(valid_m),
    ) >= lcb_floor


def _positive_good_reject_reasons(
    train_m: dict,
    valid_m: dict,
    *,
    min_train_trades: int | None = None,
    min_valid_trades: int | None = None,
) -> list[str]:
    """Human-readable reasons why ``_is_positive_good`` failed (diagnostics)."""
    thresholds = PositiveGoodThresholds(
        min_train_return=float(getattr(_cfg, "RB_MIN_TRAIN_RETURN", 0.0)),
        min_valid_return=float(getattr(_cfg, "RB_MIN_VALID_RETURN", 0.0)),
        min_train_profit_factor=float(getattr(_cfg, "RB_MIN_TRAIN_PF", 1.0)),
        min_valid_profit_factor=float(getattr(_cfg, "RB_MIN_VALID_PF", 1.0)),
        min_train_trades=int(
            min_train_trades
            if min_train_trades is not None
            else getattr(_cfg, "RB_MIN_TRAIN_TRADES", 25)
        ),
        min_valid_trades=int(
            min_valid_trades
            if min_valid_trades is not None
            else getattr(_cfg, "RB_MIN_VALID_TRADES", 15)
        ),
        require_execution_health=bool(
            getattr(_cfg, "RB_REQUIRE_EXECUTION_HEALTH_ON_SINGLES", False)
        ),
    )
    reasons = positive_good_reject_reasons(train_m, valid_m, thresholds)
    if not reasons:
        lcb_floor = float(getattr(_cfg, "RB_EXPECTANCY_LCB_MARGIN_PCT", 0.0))
        lcb = min(expectancy_lcb_pct(train_m), expectancy_lcb_pct(valid_m))
        if lcb < lcb_floor:
            reasons.append("expectancy_lcb")
    return reasons


def _recency_validation_score(fitness_m: dict, selection_m: dict) -> float:
    """Score the weaker chronological validation half.

    Recency rescue is intentionally return-first: the candidate must make
    money on *both* halves, so the minimum half-return is the useful
    out-of-regime robustness statistic.  PF and drawdown are hard gates in
    ``_is_recency_good`` rather than soft knobs that could let a high-return
    but fragile candidate win.
    """
    return float(min(
        _f(fitness_m, "total_return_pct", -1.0e9),
        _f(selection_m, "total_return_pct", -1.0e9),
    ))


def _is_recency_good(
    train_m: dict,
    fitness_m: dict,
    selection_m: dict,
) -> bool:
    """Check the bounded validation-only recency rescue certificate.

    This is deliberately stricter than a single validation gate: the older
    training regime may lose, but both chronological validation halves must
    clear return/PF/trade floors and show activity on at least two symbols.
    Concentration is checked with the explicit, slightly wider recency limits
    at final deployment; per-symbol losses are bounded in evaluator PnL units
    to prevent one symbol from carrying an otherwise weak signal.
    """
    if not bool(getattr(_cfg, "RB_RECENCY_RESCUE_ENABLED", False)):
        return False
    if _i(train_m, "executed_trades") < int(
        getattr(_cfg, "RB_RECENCY_MIN_TRAIN_TRADES", 25)
    ):
        return False
    if _f(train_m, "total_return_pct") < -float(
        getattr(_cfg, "RB_RECENCY_MAX_TRAIN_LOSS_PCT", 12.0)
    ):
        return False
    if _f(train_m, "profit_factor") < float(
        getattr(_cfg, "RB_RECENCY_MIN_TRAIN_PF", 0.80)
    ):
        return False
    if _f(train_m, "max_drawdown_pct", 100.0) > float(
        getattr(_cfg, "RB_RECENCY_MAX_TRAIN_DD_PCT", 25.0)
    ):
        return False

    min_return = float(getattr(_cfg, "RB_RECENCY_MIN_VALID_RETURN", 0.50))
    min_pf = float(getattr(_cfg, "RB_RECENCY_MIN_VALID_PF", 1.05))
    min_trades = int(getattr(_cfg, "RB_RECENCY_MIN_VALID_TRADES", 15))
    max_symbol_loss = float(
        getattr(_cfg, "RB_RECENCY_MAX_SYMBOL_LOSS_PCT", 15.0)
    )
    for metrics in (fitness_m, selection_m):
        if _f(metrics, "total_return_pct") < min_return:
            return False
        if _f(metrics, "profit_factor") < min_pf:
            return False
        if _i(metrics, "executed_trades") < min_trades:
            return False
        per_symbol = (metrics or {}).get("per_symbol_metrics", {}) or {}
        # Real evaluator output always includes this field.  Lightweight unit
        # doubles may omit it; those remain eligible for the helper-level
        # compatibility path, while production candidates require evidence.
        if not isinstance(per_symbol, dict) or not per_symbol:
            if bool((metrics or {}).get("per_symbol_metrics_available", False)):
                return False
            continue
        traded = 0
        for values in per_symbol.values():
            if not isinstance(values, dict):
                continue
            trades = int(values.get("trade_count", 0) or 0)
            net_pnl = _f(values, "net_pnl")
            if trades > 0:
                traded += 1
            if net_pnl < -max_symbol_loss:
                return False
        if traded < 2:
            return False
    return True


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
    # Keep the source-tape bar identity when the loader supplied it.  Exact
    # first-touch barrier offsets are measured on that complete tape; resetting
    # a validation slice to zero would make every release target point at the
    # wrong bar (usually the final flush).  Only legacy frames without the
    # internal identity need a local fallback.
    if "_symbol_bar_index" not in out.columns:
        if "symbol" in out.columns:
            out["_symbol_bar_index"] = out.groupby(
                "symbol", observed=False,
            ).cumcount()
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


def _cost_stress_gate(
    train_engine: CPUBacktestEngine,
    valid_engine: CPUBacktestEngine,
    rules: list[dict],
) -> tuple[bool, list[dict]]:
    """Require the final package to survive a configured cost stress."""
    if not (
        bool(getattr(_cfg, "RB_CANONICAL_PIPELINE_ACTIVE", False))
        and bool(getattr(_cfg, "RB_COST_STRESS_ENABLED", False))
    ):
        return True, []
    results: list[dict] = []
    base_fee = float(getattr(_cfg, "FEE_PCT", 0.0))
    min_return = float(getattr(_cfg, "RB_COST_STRESS_MIN_RETURN_PCT", 0.0))
    passed = True
    for multiplier in getattr(_cfg, "RB_COST_STRESS_MULTIPLIERS", (1.0,)):
        factor = float(multiplier)
        if factor <= 1.0:
            continue
        stressed_train = CPUBacktestEngine(
            train_engine.df,
            {},
            train_engine.trade_direction,
            fee_pct=base_fee * factor,
        ).simulate_rule_set([_rule_to_engine(rule) for rule in rules])
        stressed_valid = CPUBacktestEngine(
            valid_engine.df,
            {},
            valid_engine.trade_direction,
            fee_pct=base_fee * factor,
        ).simulate_rule_set([_rule_to_engine(rule) for rule in rules])
        ok = (
            _f(stressed_train, "total_return_pct") >= min_return
            and _f(stressed_valid, "total_return_pct") >= min_return
            and min(
                expectancy_lcb_pct(stressed_train),
                expectancy_lcb_pct(stressed_valid),
            ) >= float(getattr(_cfg, "RB_EXPECTANCY_LCB_MARGIN_PCT", 0.0))
        )
        passed = passed and ok
        results.append({
            "multiplier": factor,
            "train_return_pct": _f(stressed_train, "total_return_pct"),
            "valid_return_pct": _f(stressed_valid, "total_return_pct"),
            "train_profit_factor": _f(stressed_train, "profit_factor"),
            "valid_profit_factor": _f(stressed_valid, "profit_factor"),
            "passed": bool(ok),
        })
    return passed, results


def _monthly_selection_certificate(
    valid_engine: CPUBacktestEngine,
    rules: list[dict],
    direction: str,
) -> tuple[bool, dict[str, Any]]:
    """Require a composed team to be mostly non-loss across calendar windows."""
    if not bool(getattr(_cfg, "RB_MONTHLY_CERTIFICATE_ENABLED", False)):
        return True, {"enabled": False}
    windows = build_monthly_windows(valid_engine.df)
    minimum_windows = int(
        getattr(_cfg, "PHASE2_MONTHLY_ADMISSION_MIN_MONTHS", 2)
    )
    if len(windows) < minimum_windows:
        return True, {
            "enabled": True,
            "passed": True,
            "evidence": "insufficient_windows",
            "windows": len(windows),
            "required_windows": minimum_windows,
        }
    metrics: list[dict] = []
    for window in windows:
        try:
            metrics.append(CPUBacktestEngine(
                window,
                {},
                direction,
            ).simulate_rule_set([_rule_to_engine(rule) for rule in rules]))
        except Exception as exc:
            metrics.append({
                "total_return_pct": -100.0,
                "profit_factor": 0.0,
                "max_drawdown_pct": 100.0,
                "executed_trades": 0,
                "error": f"{type(exc).__name__}: {exc}",
            })
    summary = summarize_monthly_metrics(metrics, n_rows=len(valid_engine.df))
    passed = (
        summary.profitable_ratio
        >= float(getattr(_cfg, "RB_MONTHLY_MIN_PROFITABLE_RATIO", 0.55))
        and summary.active_ratio
        >= float(getattr(_cfg, "PHASE2_MONTHLY_MIN_ACTIVE_RATIO", 0.60))
        and summary.bearish_ratio
        <= float(getattr(_cfg, "RB_MONTHLY_MAX_BEARISH_RATIO", 0.35))
    )
    return bool(passed), {
        "enabled": True,
        "passed": bool(passed),
        "windows": int(summary.windows),
        "profitable_ratio": float(summary.profitable_ratio),
        "active_ratio": float(summary.active_ratio),
        "bearish_ratio": float(summary.bearish_ratio),
        "worst_return_pct": float(summary.worst_return_pct),
        "equity_slope": float(summary.equity_slope),
        "window_metrics": metrics,
    }


def _candidate_positive_symbols(candidate: CandidateRecord) -> set[str]:
    """Return supported positive validation symbols for one candidate."""
    per_symbol = (candidate.valid_metrics or {}).get(
        "per_symbol_metrics", {},
    ) or {}
    if isinstance(per_symbol, dict) and per_symbol:
        return {
            str(symbol)
            for symbol, values in per_symbol.items()
            if isinstance(values, dict)
            and int(values.get("trade_count", 0)) >= int(
                getattr(_cfg, "RB_MIN_VALID_TRADES", 6)
            )
            and float(values.get("net_pnl", 0.0)) > 0.0
        }
    return _candidate_coverage_symbols(candidate)


def _symbol_gate_policy(
    candidates: list[CandidateRecord],
    train_like: pd.DataFrame,
    valid_df: pd.DataFrame,
    tail_holdout_engine: CPUBacktestEngine | None = None,
) -> dict[str, Any]:
    """Build the per-direction symbol certificate policy.

    Specialist islands are intentionally asymmetric: a direction can have a
    healthy BTC island and no healthy ETH island (or vice versa).  Requiring
    the configured universe floor in that case turns a valid direction into
    an empty strategy before Phase 5.  We therefore lower the *effective*
    coverage floor only when the post-admission candidate set itself has no
    positive, tail-eligible candidate for a missing symbol.  Ordinary
    multi-symbol teams
    retain the configured floor and concentration limits.

    This helper is validation-only.  It never inspects the Phase 5/test frame.
    """
    configured = max(1, int(getattr(_cfg, "RB_MIN_DISTINCT_SYMBOLS", 1)))
    active_symbols = {
        str(symbol).strip()
        for symbol in _available_symbols(train_like, valid_df)
        if str(symbol).strip()
    }
    active_by_lower = {symbol.lower(): symbol for symbol in active_symbols}
    candidate_symbols: set[str] = set()
    candidate_symbols_before_tail: set[str] = set()
    for candidate in candidates:
        raw_symbols = {
            str(symbol).strip()
            for symbol in _candidate_positive_symbols(candidate)
            if str(symbol).strip()
        }
        scoped_symbols = {
            active_by_lower[symbol.lower()]
            for symbol in raw_symbols
            if symbol.lower() in active_by_lower
        }
        candidate_symbols_before_tail |= scoped_symbols
        if tail_holdout_engine is not None:
            tail_ok, _ = _passes_tail_selection_gate(
                tail_holdout_engine, [candidate.rule],
            )
            if not tail_ok:
                continue
        candidate_symbols |= scoped_symbols

    specialist_mode = bool(getattr(_cfg, "RB_REQUIRE_SYMBOL_FILTERS", False))
    allow_partial = bool(
        getattr(_cfg, "RB_ALLOW_PARTIAL_SPECIALIST_COVERAGE", False)
    )
    # A multi-symbol release is a product contract, not a soft preference.
    # Never lower its required universe or concentration limits just because
    # one symbol failed to produce a candidate.
    if (
        bool(getattr(_cfg, "RB_MULTI_SYMBOL_RELEASE", True))
        and len(active_symbols) > 1
    ):
        allow_partial = False
    partial = bool(
        specialist_mode
        and allow_partial
        and 0 < len(candidate_symbols) < configured
    )
    effective = len(candidate_symbols) if partial else configured
    effective = max(1, effective)
    return {
        "configured_min_symbols": configured,
        "effective_min_symbols": effective,
        "partial_specialist_coverage": partial,
        "specialist_mode": specialist_mode,
        "allow_partial": allow_partial,
        "candidate_positive_symbols": sorted(candidate_symbols),
        "candidate_positive_symbols_before_tail": sorted(
            candidate_symbols_before_tail
        ),
        # Symbols represented by the positive candidate pool but with no
        # candidate surviving the reserved tail gate.
        "tail_rejected_candidate_symbols": sorted(
            candidate_symbols_before_tail - candidate_symbols
        ),
        "active_symbols": sorted(active_symbols),
        "missing_candidate_symbols": sorted(active_symbols - candidate_symbols),
        # Concentration limits remain the configured multi-symbol limits. They
        # are never relaxed to 1.0 by missing candidate coverage.
        "concentration_max_share": (
            float(getattr(_cfg, "RB_MAX_SYMBOL_SHARE_ABS_PNL", 1.0))
        ),
        "concentration_max_hhi": (
            float(getattr(_cfg, "RB_MAX_SYMBOL_HHI", 1.0))
        ),
    }


def _diversification_shortlist(
    candidates: list[CandidateRecord],
) -> list[CandidateRecord]:
    """Keep global plus score and return leaders for each positive symbol."""
    global_count = int(getattr(_cfg, "RB_DIVERSIFICATION_GLOBAL_LEADERS", 6))
    per_symbol_count = int(
        getattr(_cfg, "RB_DIVERSIFICATION_SYMBOL_LEADERS", 2)
    )
    return_leader_count = int(
        getattr(_cfg, "RB_DIVERSIFICATION_RETURN_LEADERS", 4)
    )
    ranked = sorted(candidates, key=lambda rec: rec.score, reverse=True)
    shortlist: list[CandidateRecord] = []
    seen: set[tuple[str, ...]] = set()

    def add(record: CandidateRecord) -> None:
        key = _rule_key(record.rule)
        if key not in seen:
            seen.add(key)
            shortlist.append(record)

    for record in ranked[:max(1, global_count)]:
        add(record)
    symbols = sorted({
        symbol
        for record in ranked
        for symbol in _candidate_positive_symbols(record)
    })
    for symbol in symbols:
        leaders = [
            record for record in ranked
            if symbol in _candidate_positive_symbols(record)
        ]
        for record in leaders[:max(1, per_symbol_count)]:
            add(record)
        return_leaders = sorted(
            leaders,
            key=lambda record: (
                _f(record.valid_metrics, "total_return_pct"),
                _f(record.train_metrics, "total_return_pct"),
            ),
            reverse=True,
        )
        for record in return_leaders[:max(1, return_leader_count)]:
            add(record)
    return shortlist


def _diversification_beam(
    candidates: list[CandidateRecord],
    train_engine: CPUBacktestEngine,
    valid_engine: CPUBacktestEngine,
    direction: str,
    *,
    min_distinct_symbols: int,
    tail_holdout_engine: CPUBacktestEngine | None = None,
) -> tuple[
    list[CandidateRecord], dict, dict, float, list[dict]
] | None:
    """Find a certificate-safe diversified seed with a bounded beam."""
    if min_distinct_symbols <= 1:
        return None
    if not any(
        isinstance((candidate.valid_metrics or {}).get(
            "per_symbol_metrics"), dict)
        and bool((candidate.valid_metrics or {}).get("per_symbol_metrics"))
        for candidate in candidates
    ):
        return None
    shortlist = _diversification_shortlist(candidates)
    if len(shortlist) < 2:
        return None
    beam_width = max(
        1, int(getattr(_cfg, "RB_DIVERSIFICATION_BEAM_WIDTH", 6))
    )
    n_steps = max(
        1, int(getattr(_cfg, "RB_DIVERSIFICATION_STEPS", 4))
    )
    max_overlap = float(getattr(_cfg, "RB_MAX_PAIR_OVERLAP", 0.35))

    states: list[dict[str, Any]] = []
    for candidate in shortlist:
        try:
            train_m, valid_m, score = _evaluate_ruleset(
                train_engine, valid_engine, [candidate.rule],
            )
        except Exception:
            continue
        if not _is_positive_good(
            train_m,
            valid_m,
            min_train_trades=int(getattr(
                _cfg, "RB_RULESET_MIN_TRAIN_TRADES",
                getattr(_cfg, "RB_MIN_TRAIN_TRADES", 25),
            )),
            min_valid_trades=int(getattr(
                _cfg, "RB_RULESET_MIN_VALID_TRADES",
                getattr(_cfg, "RB_MIN_VALID_TRADES", 15),
            )),
        ):
            continue
        tail_ok, _ = _passes_tail_selection_gate(
            tail_holdout_engine, [candidate.rule],
        )
        if not tail_ok:
            continue
        states.append({
            "selected": [candidate],
            "train": train_m,
            "valid": valid_m,
            "score": score,
        })
    if not states:
        return None

    def coverage(state: dict[str, Any]) -> set[str]:
        valid = state["valid"]
        per_symbol = (valid or {}).get("per_symbol_metrics", {}) or {}
        if isinstance(per_symbol, dict) and per_symbol:
            return set(
                _passes_symbol_contribution_certificate(
                    valid,
                    min_symbols=0,
                )[1].get("qualifying_symbols", [])
            )
        out: set[str] = set()
        for record in state["selected"]:
            out |= _candidate_positive_symbols(record)
        return out

    def state_sort_key(state: dict[str, Any]) -> tuple:
        covered = coverage(state)
        # Until the target is covered, symbol breadth has priority over raw
        # score.  Once covered, score determines which state continues.
        return (
            min(len(covered), min_distinct_symbols),
            float(state["score"]),
            -len(state["selected"]),
        )

    # Keep the highest-scoring state for every positive symbol, then fill the
    # remainder by score.  This prevents an ETH-only leader from crowding out
    # every BTC seed before diversification starts.
    seeded: list[dict[str, Any]] = []
    used_state_keys: set[tuple[tuple[str, ...], ...]] = set()
    symbol_universe = sorted({
        symbol for state in states for symbol in coverage(state)
    })
    for symbol in symbol_universe:
        symbol_states = [
            state for state in states if symbol in coverage(state)
        ]
        if symbol_states:
            state = max(symbol_states, key=lambda item: float(item["score"]))
            state_key = tuple(
                _rule_key(record.rule) for record in state["selected"]
            )
            if state_key not in used_state_keys:
                used_state_keys.add(state_key)
                seeded.append(state)
    for state in sorted(states, key=state_sort_key, reverse=True):
        if len(seeded) >= beam_width:
            break
        state_key = tuple(_rule_key(record.rule)
                          for record in state["selected"])
        if state_key not in used_state_keys:
            used_state_keys.add(state_key)
            seeded.append(state)
    states = seeded[:beam_width]
    all_states = list(states)
    history: list[dict] = []

    for step in range(1, n_steps + 1):
        expanded: list[dict[str, Any]] = list(states)
        for state in states:
            used = {_rule_key(record.rule) for record in state["selected"]}
            for candidate in shortlist:
                if _rule_key(candidate.rule) in used:
                    continue
                if _max_overlap(candidate, state["selected"]) > max_overlap:
                    continue
                trial = state["selected"] + [candidate]
                try:
                    train_m, valid_m, score = _evaluate_ruleset(
                        train_engine, valid_engine,
                        [record.rule for record in trial],
                    )
                except Exception:
                    continue
                if not _is_positive_good(
                    train_m,
                    valid_m,
                    min_train_trades=int(getattr(
                        _cfg, "RB_RULESET_MIN_TRAIN_TRADES",
                        getattr(_cfg, "RB_MIN_TRAIN_TRADES", 25),
                    )),
                    min_valid_trades=int(getattr(
                        _cfg, "RB_RULESET_MIN_VALID_TRADES",
                        getattr(_cfg, "RB_MIN_VALID_TRADES", 15),
                    )),
                ):
                    continue
                tail_ok, _ = _passes_tail_selection_gate(
                    tail_holdout_engine,
                    [record.rule for record in trial],
                )
                if not tail_ok:
                    continue
                expanded.append({
                    "selected": trial,
                    "train": train_m,
                    "valid": valid_m,
                    "score": score,
                })
        deduped: dict[tuple[tuple[str, ...], ...], dict[str, Any]] = {}
        for state in expanded:
            key = tuple(
                sorted(_rule_key(record.rule) for record in state["selected"])
            )
            incumbent = deduped.get(key)
            if incumbent is None or state_sort_key(state) > state_sort_key(incumbent):
                deduped[key] = state
        states = sorted(
            deduped.values(), key=state_sort_key, reverse=True,
        )[:beam_width]
        all_states.extend(states)
        history.append({
            "step": step,
            "action": "diversification_beam",
            "beam_width": beam_width,
            "beam_states": len(states),
            "max_covered_symbols": max(
                (len(coverage(state)) for state in states),
                default=0,
            ),
        })

    certified: list[dict[str, Any]] = []
    for state in all_states:
        if not isinstance(
            (state["valid"] or {}).get("per_symbol_metrics"),
            dict,
        ) or not (state["valid"] or {}).get("per_symbol_metrics"):
            continue
        if len(coverage(state)) < min_distinct_symbols:
            continue
        # Composition establishes independently positive symbol coverage;
        # risk optimization subsequently chooses each rule's capital weight
        # and is responsible for the final concentration certificate. A pair
        # can be genuinely diversifying yet fail concentration temporarily at
        # equal seed weights, so rejecting it here would prevent that tuning.
        passed, _certificate = _passes_symbol_contribution_certificate(
            state["valid"],
            min_symbols=min_distinct_symbols,
        )
        tail_ok, _ = _passes_tail_selection_gate(
            tail_holdout_engine,
            [record.rule for record in state["selected"]],
        )
        if passed and tail_ok:
            certified.append(state)
    if not certified:
        return None
    best = max(certified, key=lambda state: float(state["score"]))
    history.append({
        "step": len(history) + 1,
        "action": "diversification_certificate_passed",
        "score": float(best["score"]),
        "rules": len(best["selected"]),
        "covered_symbols": sorted(coverage(best)),
    })
    return (
        list(best["selected"]),
        best["train"],
        best["valid"],
        float(best["score"]),
        history,
    )


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


def _risk_envelope_admission_variant(
    rule: dict,
    train_engine: CPUBacktestEngine,
    valid_engine: CPUBacktestEngine,
) -> tuple[dict, dict, dict] | None:
    """Find a validation-only TP/SL profile for a rule rejected at default risk.

    Phase 2 uses one fixed TP/SL pair so the evolutionary objective remains
    comparable.  RB, however, owns risk selection and already has a bounded
    TP/SL grid.  Applying the positive-single gate *before* that grid meant a
    rule could be profitable at a legitimate RB profile yet be discarded at
    the Phase 2 profile.  This helper reuses the existing grid only as an
    admission envelope; the ordinary walk-forward risk optimizer remains the
    authority for the final profile.

    Returns the best (rule, train metrics, validation metrics) triplet, or
    ``None`` when no grid point clears the same positive-good thresholds.
    No Phase 5/test frame is reachable from this function.
    """
    if not bool(getattr(_cfg, "RB_CANDIDATE_RISK_ADMISSION_ENABLED", True)):
        return None

    current_tp = float(rule.get("tp", getattr(_cfg, "RB_DEFAULT_TP", 2.0)))
    current_sl = float(rule.get("sl", getattr(_cfg, "RB_DEFAULT_SL", 1.2)))
    profiles: list[tuple[float, float]] = []
    seen_profiles: set[tuple[float, float]] = set()
    for raw_tp in getattr(_cfg, "RB_TP_GRID", (current_tp,)):
        for raw_sl in getattr(_cfg, "RB_SL_GRID", (current_sl,)):
            try:
                tp = float(raw_tp)
                sl = float(raw_sl)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(tp) or not math.isfinite(sl):
                continue
            if bool(getattr(_cfg, "RB_REQUIRE_TP_SL_ABOVE_ONE", True)):
                tp = max(tp, float(getattr(_cfg, "RB_MIN_TP", 1.01)))
                sl = max(sl, float(getattr(_cfg, "RB_MIN_SL", 1.01)))
            profile = (tp, sl)
            if profile not in seen_profiles:
                seen_profiles.add(profile)
                profiles.append(profile)
    if not profiles:
        return None

    best: tuple[tuple[float, float, float], dict, dict, dict] | None = None
    for tp, sl in profiles:
        candidate = dict(rule)
        candidate["tp"] = tp
        candidate["sl"] = sl
        try:
            train_m = train_engine.simulate_rule_set([candidate])
            valid_m = valid_engine.simulate_rule_set([candidate])
        except Exception:
            continue
        if not _is_positive_good(train_m, valid_m):
            continue
        # The weakest split is the primary tie-breaker.  This avoids choosing
        # a profile solely for a large historical return while still using the
        # shared RB score as a secondary quality/risk discriminator.
        min_return = min(
            _f(train_m, "total_return_pct"),
            _f(valid_m, "total_return_pct"),
        )
        min_pf = min(
            _f(train_m, "profit_factor"),
            _f(valid_m, "profit_factor"),
        )
        score = _score_metrics(train_m, valid_m)
        rank = (min_return, min_pf, score)
        if best is None or rank > best[0]:
            best = (rank, candidate, train_m, valid_m)

    if best is None:
        return None
    return best[1], best[2], best[3]


def _balanced_phase2_shortlist(pool: list[dict], limit: int) -> list[dict]:
    """Take a bounded, symbol-balanced Phase 2 shortlist.

    Global rank alone can consume the whole RB budget with one specialist and
    make the missing island impossible to compose.  Round-robin selection is
    deterministic and leaves the original ordering intact within each scope.
    """
    if limit <= 0 or len(pool) <= limit:
        return list(pool)
    groups: dict[str, list[dict]] = {}
    for entry in pool:
        symbols = entry.get("source_symbols", entry.get("island_symbols", []))
        if isinstance(symbols, str):
            symbols = [symbols]
        label = ",".join(sorted({
            str(symbol).strip() for symbol in (symbols or [])
            if str(symbol).strip()
        })) or "__global__"
        groups.setdefault(label, []).append(entry)
    ordered_groups = [
        groups[key] for key in sorted(groups)
    ]
    selected: list[dict] = []
    cursor = 0
    while len(selected) < limit and ordered_groups:
        progressed = False
        for group in ordered_groups:
            if cursor < len(group):
                selected.append(group[cursor])
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
        cursor += 1
    return selected


def _filter_good_rules(
    pool: list[dict],
    train_like_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    direction: str,
    fold_engines: list[CPUBacktestEngine] | None = None,
    *,
    recency_fitness_engine: CPUBacktestEngine | None = None,
    recency_sink: list[CandidateRecord] | None = None,
) -> list[CandidateRecord]:
    train_engine = CPUBacktestEngine(train_like_df, {}, direction)
    valid_engine = CPUBacktestEngine(valid_df, {}, direction)
    records: list[CandidateRecord] = []
    seen: set[tuple[str, ...]] = set()
    reject_counts: dict[str, int] = {}
    risk_rescued = 0
    limit = int(getattr(_cfg, "RB_MAX_POOL_RULES_TO_EVALUATE", 700))
    symbols = _available_symbols(train_like_df, valid_df)

    for raw in _balanced_phase2_shortlist(pool, limit):
        for rule in _symbol_specialized_variants(raw, train_engine, valid_engine, symbols):
            # Variants already include island ``symbol is X`` OR filters in Mode A.
            rule = _rule_to_engine(rule)
            key = tuple(sorted(str(c) for c in rule.get("conditions", [])))
            if key in seen:
                continue
            seen.add(key)
            try:
                train_m = train_engine.simulate_rule_set([rule])
                valid_m = valid_engine.simulate_rule_set([rule])
            except Exception:
                reject_counts["simulate_error"] = reject_counts.get(
                    "simulate_error", 0) + 1
                continue
            if not _is_positive_good(train_m, valid_m):
                # A fixed Phase 2 profile is a discovery anchor, not RB's
                # final risk choice.  Before rejecting the feature rule,
                # search the existing RB TP/SL envelope on train and the
                # validation-selection frame.  This does not read test data
                # and does not bypass any later walk-forward/tail gate.
                rescued = _risk_envelope_admission_variant(
                    rule, train_engine, valid_engine,
                )
                if rescued is not None:
                    rule, train_m, valid_m = rescued
                    risk_rescued += 1
                else:
                    if (
                        recency_fitness_engine is not None
                        and recency_sink is not None
                        and bool(getattr(_cfg, "RB_RECENCY_RESCUE_ENABLED", False))
                    ):
                        try:
                            fitness_m = recency_fitness_engine.simulate_rule_set([rule])
                        except Exception:
                            fitness_m = None
                        if fitness_m is not None and _is_recency_good(
                            train_m, fitness_m, valid_m,
                        ):
                            rec = CandidateRecord(
                                rule=rule,
                                train_metrics=train_m,
                                valid_metrics=valid_m,
                                score=_recency_validation_score(
                                    fitness_m, valid_m,
                                ),
                                recency=True,
                                recency_fitness_metrics=fitness_m,
                            )
                            rec.mask = _mask_for(rule, train_like_df, valid_df)
                            recency_sink.append(rec)
                    for reason in _positive_good_reject_reasons(train_m, valid_m) or ["unknown"]:
                        # Bucket by primary token (before '=')
                        bucket = reason.split("=", 1)[0]
                        reject_counts[bucket] = reject_counts.get(bucket, 0) + 1
                    continue
            # Evaluate on CV folds if available (C4)
            cv_fold_returns = _eval_cv_fold_returns(rule, fold_engines)
            score = _score_metrics(
                train_m, valid_m, cv_fold_returns=cv_fold_returns)
            # Prefer cross-symbol coverage when building multi-symbol teams
            # (generalist mode). Explicit single-symbol filters get no bonus.
            if not bool(getattr(_cfg, "RB_REQUIRE_SYMBOL_FILTERS", False)):
                n_cov = len(
                    _traded_symbols_from_metrics(train_m)
                    | _traded_symbols_from_metrics(valid_m)
                )
                score += float(getattr(_cfg, "RB_MULTI_SYMBOL_COVERAGE_BONUS", 8.0)) * max(
                    0, n_cov - 1
                )
            rec = CandidateRecord(
                rule=rule,
                train_metrics=train_m,
                valid_metrics=valid_m,
                score=score,
                fold_returns=cv_fold_returns,
                pnl_series=(
                    np.asarray(cv_fold_returns, dtype=float)
                    if cv_fold_returns
                    else None
                ),
            )
            rec.mask = _mask_for(rule, train_like_df, valid_df)
            records.append(rec)

    records.sort(key=lambda r: r.score, reverse=True)
    if recency_sink is not None:
        recency_sink.sort(key=lambda r: r.score, reverse=True)
        max_recency = max(
            0, int(getattr(_cfg, "RB_RECENCY_MAX_CANDIDATES", 40))
        )
        del recency_sink[max_recency:]
    keep = int(getattr(_cfg, "RB_KEEP_TOP_RULES", 120))
    logger.info(
        "RB [%s]: kept %d/%d single rules positive on training and validation "
        "(%d recovered by TP/SL admission envelope).",
        direction, min(len(records), keep), len(seen), risk_rescued,
    )
    if not records and reject_counts:
        top = sorted(reject_counts.items(),
                     key=lambda kv: kv[1], reverse=True)[:8]
        logger.warning(
            "RB [%s]: positive-good reject breakdown (unique rules=%d): %s",
            direction,
            len(seen),
            {k: v for k, v in top},
        )
    return records[:keep]


def _univariate_baseline_pool(
    pool: list[dict],
    *frames: pd.DataFrame,
) -> tuple[list[dict], int]:
    """Add deterministic single-condition RB candidates.

    Evolution is deliberately biased toward 4–5 condition rules. This compact
    complement enumerates only the fuzzy modes already exposed by the current
    Phase-2 pool, so it cannot introduce raw columns or a representation not
    selected in Phase 1.  A generalist pass is emitted first, followed by the
    symbol-specialized pass.  This ordering ensures the bounded budget cannot
    truncate a late-but-useful condition such as a momentum state merely
    because early alphabetical features consumed all slots.  Every candidate
    still passes exact train, validation, tail, concentration, and Phase-5
    gates before deployment.
    """
    if not bool(getattr(_cfg, "RB_UNIVARIATE_BASELINE_ENABLED", True)):
        return list(pool), 0
    max_rules = max(0, int(getattr(
        _cfg, "RB_UNIVARIATE_BASELINE_MAX_RULES", 400,
    )))
    if max_rules == 0:
        return list(pool), 0

    mode_values: dict[str, set[str]] = {}
    for mode in (
        "binary",
        "ternary",
        "positive",
        "sparse_positive",
        "sparse_signed",
        "signed",
    ):
        values = {
            encode_condition("feature", gene, mode).split(" IS ", 1)[1]
            for gene in range(get_dont_care(mode))
        }
        mode_values[mode] = values

    possible_modes: dict[str, set[str]] = {}
    for raw in pool:
        for condition in raw.get("conditions", []) if isinstance(raw, dict) else []:
            text = str(condition)
            if not text.startswith("[") or "] IS " not in text:
                continue
            feature, value = text[1:].split("] IS ", 1)
            supported = {
                mode for mode, values in mode_values.items() if value in values
            }
            if not feature or not supported:
                continue
            if feature in possible_modes:
                intersection = possible_modes[feature] & supported
                if intersection:
                    possible_modes[feature] = intersection
            else:
                possible_modes[feature] = supported

    # Phase-2 chromosomes are sparse: a feature can be selected in Phase 1
    # yet never appear as an active gene in the retained pool.  Recover its
    # evaluator mode from the already-pruned train frame so the deterministic
    # complement remains a true Phase-1 feature search rather than a random
    # pool-dependent subset.  This is train-frame metadata only; validation and
    # test values are never inspected to choose the mode.
    try:
        from gpu_fuzzy_trader.features.detector import Feature_Detector

        for frame in frames:
            if not isinstance(frame, pd.DataFrame):
                continue
            feature_cols = [
                name for name in frame.columns
                if str(name).startswith("ff_")
            ]
            if not feature_cols:
                continue
            detected = Feature_Detector().detect_all_modes(frame, feature_cols)
            for feature, mode in detected.items():
                possible_modes.setdefault(feature, set()).add(mode)
            # The first frame is the train-like frame in every caller; do not
            # let a later validation frame introduce a previously unseen mode.
            break
    except Exception:
        # Pool-derived modes remain a safe fallback for lightweight test
        # doubles or unusual frames.
        pass

    symbols = _available_symbols(*frames)
    if not possible_modes or not symbols:
        return list(pool), 0
    capital_grid = tuple(float(value) for value in getattr(
        _cfg, "RB_CAPITAL_GRID", (5.0,),
    ))
    baseline_capital = min(capital_grid) if capital_grid else 5.0
    out = list(pool)
    seen = {
        _rule_key(raw) for raw in pool if isinstance(raw, dict)
    }
    added = 0
    mode_order = (
        "binary",
        "ternary",
        "positive",
        "sparse_positive",
        "sparse_signed",
        "signed",
    )

    def add_rule(conditions: list[str]) -> bool:
        nonlocal added
        rule = {
            "conditions": conditions,
            "tp": float(getattr(_cfg, "RB_DEFAULT_TP", 2.0)),
            "sl": float(getattr(_cfg, "RB_DEFAULT_SL", 1.2)),
            # A frequent singleton is evaluated at the smallest permitted
            # allocation first. The risk grid can increase it only if that
            # remains robust and balanced.
            "capital_pct": baseline_capital,
            "source": "rb_univariate_baseline",
        }
        key = _rule_key(rule)
        if key in seen:
            return True
        seen.add(key)
        out.append(rule)
        added += 1
        return added < max_rules

    ordered_features: list[tuple[str, str]] = []
    for feature in sorted(possible_modes):
        mode = next(
            (item for item in mode_order if item in possible_modes[feature]),
            None,
        )
        if mode is not None:
            ordered_features.append((feature, mode))

    # First reserve the complete generalist complement.  In Mode A a rule is
    # allowed to trade both symbols, and this is also the only way to discover
    # a condition whose useful state was absent from the stochastic pool.
    if bool(getattr(_cfg, "RB_UNIVARIATE_GENERALIST_ENABLED", True)):
        for feature, mode in ordered_features:
            for gene in range(get_dont_care(mode)):
                if not add_rule([encode_condition(feature, gene, mode)]):
                    return out, added

    # Then add specialists, preserving the historical symbol-scoped behavior
    # for multi-asset diversification and specialist mode.
    for feature, mode in ordered_features:
        for gene in range(get_dont_care(mode)):
            condition = encode_condition(feature, gene, mode)
            for symbol in symbols:
                if not add_rule([condition, _symbol_condition(symbol)]):
                    return out, added
    return out, added


def _correlation_control(
    canonical_name: str,
    legacy_name: str,
    default: float,
) -> object:
    """Read a canonical correlation control with legacy-config support."""
    canonical = getattr(_cfg, canonical_name, default)
    legacy = getattr(_cfg, legacy_name, default)
    if canonical != default or legacy == default:
        return canonical
    return legacy


def _redundancy_lambda() -> float:
    """Resolve the configured zero/low/medium redundancy penalty."""
    raw = _correlation_control(
        "RB_REDUNDANCY_PENALTY",
        "RB_CORRELATION_LAMBDA",
        0.0,
    )
    resolver = getattr(_cfg, "resolve_redundancy_penalty", None)
    if callable(resolver):
        return float(resolver(raw))
    return float(raw)


def _candidate_pnl_evidence(candidate: CandidateRecord) -> np.ndarray | None:
    """Return the best available candidate-major PnL evidence."""
    explicit = getattr(candidate, "pnl_series", None)
    if explicit is not None:
        try:
            values = np.asarray(explicit, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            values = np.asarray([], dtype=float)
        if values.size:
            return values
    fold_returns = getattr(candidate, "fold_returns", None)
    if fold_returns:
        try:
            return np.asarray(fold_returns, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            pass
    for metrics in (candidate.valid_metrics, candidate.train_metrics):
        if not isinstance(metrics, dict):
            continue
        for key in ("pnl_series", "returns", "trade_returns", "pnl"):
            values = metrics.get(key)
            if values is None or np.isscalar(values):
                continue
            try:
                array = np.asarray(values, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                continue
            if array.size:
                return array
    return None


def _normalised_candidate_masks(candidates: list[CandidateRecord]) -> list[np.ndarray]:
    """Make missing or legacy masks safe for a report-only correlation pass."""
    raw_masks: list[np.ndarray] = []
    for candidate in candidates:
        try:
            raw = np.asarray(candidate.mask, dtype=bool).reshape(-1)
        except (TypeError, ValueError):
            raw = np.asarray([], dtype=bool)
        raw_masks.append(raw)
    target_size = max((mask.size for mask in raw_masks), default=0)
    masks: list[np.ndarray] = []
    for mask in raw_masks:
        if mask.size == target_size:
            masks.append(mask)
            continue
        padded = np.zeros(target_size, dtype=bool)
        padded[: min(mask.size, target_size)] = mask[:target_size]
        masks.append(padded)
    return masks


def _correlation_profile(
    candidates: list[CandidateRecord],
) -> tuple[dict[str, Any], np.ndarray, list[list[int]], dict[int, int]]:
    """Build the correlation report and the arrays used by compose."""
    if not candidates:
        empty = {
            "enabled": bool(getattr(_cfg, "RB_CORRELATION_AWARE_SELECTION", False)),
            "report_only": not bool(
                getattr(_cfg, "RB_CORRELATION_AWARE_SELECTION", False)
            ),
            "formula": "Rij=0.5*Overlap+0.5*max(0,PnLCorr)",
            "matrix": [],
            "clusters": [],
            "selected_order": [],
        }
        return empty, np.zeros((0, 0), dtype=float), [], {}

    masks = _normalised_candidate_masks(candidates)
    pnl = [_candidate_pnl_evidence(candidate) for candidate in candidates]
    signal_weight = float(
        _correlation_control(
            "RB_SIGNAL_OVERLAP_WEIGHT",
            "RB_CORRELATION_SIGNAL_WEIGHT",
            0.5,
        )
    )
    pnl_weight = float(
        _correlation_control(
            "RB_PNL_CORR_WEIGHT",
            "RB_CORRELATION_PNL_WEIGHT",
            0.5,
        )
    )
    penalty_lambda = _redundancy_lambda()
    matrix = redundancy_matrix(
        masks,
        pnl,
        signal_weight=signal_weight,
        pnl_weight=pnl_weight,
    )

    # If per-fold PnL vectors are supplied, construct one matrix per fold and
    # apply the stable median-plus-alpha rule.  The ordinary RB path currently
    # supplies candidate-major fold returns, so its one-matrix result is still
    # passed through the same stable formula.
    fold_evidence = [getattr(candidate, "fold_pnl_series", None) for candidate in candidates]
    fold_count = max(
        (len(evidence) for evidence in fold_evidence if evidence),
        default=0,
    )
    if fold_count and all(
        evidence is not None and len(evidence) == fold_count
        for evidence in fold_evidence
    ):
        fold_matrices: list[np.ndarray] = []
        for fold_index in range(fold_count):
            fold_pnl = [evidence[fold_index] for evidence in fold_evidence]
            fold_matrices.append(
                redundancy_matrix(
                    masks,
                    fold_pnl,
                    signal_weight=signal_weight,
                    pnl_weight=pnl_weight,
                )
            )
        stable_input = np.stack(fold_matrices, axis=0)
    else:
        stable_input = np.expand_dims(matrix, axis=0)
    stable_matrix = np.asarray(
        stable_corr(
            stable_input,
            alpha=float(getattr(_cfg, "RB_CORRELATION_STABILITY_ALPHA", 0.25)),
            axis=0,
        ),
        dtype=float,
    )
    stable_matrix = np.clip(stable_matrix, 0.0, 1.0)
    clusters = threshold_graph_clusters(
        stable_matrix,
        threshold=float(getattr(_cfg, "RB_CORRELATION_CLUSTER_THRESHOLD", 0.70)),
    )
    cluster_map = {
        index: cluster_id
        for cluster_id, cluster in enumerate(clusters)
        for index in cluster
    }
    enabled = bool(getattr(_cfg, "RB_CORRELATION_AWARE_SELECTION", False))
    order = greedy_adjusted_quality(
        [float(candidate.score) for candidate in candidates],
        stable_matrix,
        lambda_=penalty_lambda,
        max_items=len(candidates),
        clusters=clusters,
        require_cross_cluster=enabled,
    )
    report = {
        "enabled": enabled,
        "report_only": not enabled,
        "formula": "Rij=0.5*Overlap+0.5*max(0,PnLCorr)",
        "stable_formula": "median+alpha*std",
        "signal_weight": signal_weight,
        "pnl_weight": pnl_weight,
        "stability_alpha": float(
            getattr(_cfg, "RB_CORRELATION_STABILITY_ALPHA", 0.25)
        ),
        "lambda": penalty_lambda,
        "redundancy_penalty": penalty_lambda,
        "cluster_threshold": float(
            getattr(_cfg, "RB_CORRELATION_CLUSTER_THRESHOLD", 0.70)
        ),
        "matrix": stable_matrix.tolist(),
        "clusters": clusters,
        "selected_order": order,
        "candidate_count": len(candidates),
        "fold_count": fold_count,
    }
    return report, stable_matrix, clusters, cluster_map


def _marginal_metric_has_worst_month(metrics: object) -> bool:
    """Return whether a backtest result already carries a worst-month value."""
    if not isinstance(metrics, dict):
        return False
    return any(
        key in metrics
        for key in (
            "WorstMonth",
            "worstmonth",
            "worst_month",
            "worst_month_return",
            "worst_month_return_pct",
            "worst_return_pct",
        )
    )


def _marginal_exact_metrics(
    engine: CPUBacktestEngine,
    rules: list[dict],
    direction: str,
) -> dict:
    """Run one exact CPU evaluation and attach a monthly worst-return value."""
    formatted = [_rule_to_engine(rule) for rule in rules]
    try:
        result = engine.simulate_rule_set(formatted)
    except Exception as exc:
        logger.warning(
            "RB [%s]: marginal CPU evaluation failed: %s",
            direction,
            exc,
        )
        return {}
    if isinstance(result, tuple):
        result = result[0] if result and isinstance(result[0], dict) else {}
    metrics = dict(result) if isinstance(result, dict) else {}
    if _marginal_metric_has_worst_month(metrics):
        return metrics

    # CPUBacktestEngine does not need to calculate monthly summaries for its
    # ordinary path.  Build them only for this report, and only when the engine
    # exposes a real frame.  Small fixtures may have no eligible monthly
    # windows; in that case zero is an explicit, non-fabricated fallback.
    metrics["WorstMonth"] = 0.0
    frame = getattr(engine, "df", None)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return metrics
    try:
        windows = build_monthly_windows(frame)
    except Exception as exc:
        logger.debug("RB [%s]: cannot build marginal monthly windows: %s", direction, exc)
        return metrics
    if not windows:
        return metrics

    monthly_returns: list[float] = []
    for window in windows:
        try:
            window_metrics = CPUBacktestEngine(
                window,
                {},
                direction,
            ).simulate_rule_set(formatted)
            if isinstance(window_metrics, tuple):
                window_metrics = window_metrics[0]
            if isinstance(window_metrics, dict):
                monthly_returns.append(_f(window_metrics, "total_return_pct"))
        except Exception as exc:
            logger.debug(
                "RB [%s]: marginal monthly evaluation failed: %s",
                direction,
                exc,
            )
    if monthly_returns:
        metrics["WorstMonth"] = float(min(monthly_returns))
    return metrics


def _marginal_redundancy_matrix(
    selected: list[CandidateRecord],
) -> np.ndarray:
    """Build the dependence matrix used by the effective rule count report."""
    if not selected:
        return np.zeros((0, 0), dtype=float)
    masks = _normalised_candidate_masks(selected)
    pnl = [_candidate_pnl_evidence(candidate) for candidate in selected]
    return redundancy_matrix(
        masks,
        pnl,
        signal_weight=float(
            _correlation_control(
                "RB_SIGNAL_OVERLAP_WEIGHT",
                "RB_CORRELATION_SIGNAL_WEIGHT",
                0.5,
            )
        ),
        pnl_weight=float(
            _correlation_control(
                "RB_PNL_CORR_WEIGHT",
                "RB_CORRELATION_PNL_WEIGHT",
                0.5,
            )
        ),
    )


def _empty_marginal_report(
    direction: str,
    *,
    reason: str = "no_selected_rules",
) -> dict[str, Any]:
    """Return a stable report shape for empty or fail-closed directions."""
    return {
        "direction": direction,
        "pruning_enabled": bool(getattr(_cfg, "RB_MARGINAL_PRUNING", False)),
        "effective_rule_count_enabled": bool(
            getattr(_cfg, "RB_EFFECTIVE_RULE_COUNT_ENABLED", True)
        ),
        "passes": [],
        "per_rule": [],
        "rules": [],
        "selected_rule_count": 0,
        "removed_rule_count": 0,
        "effective_independent_rules": 0.0,
        "effective_rule_count": 0.0,
        "stable": True,
        "pruned": False,
        "reason": reason,
    }


def _marginal_prune_ruleset(
    selected: list[CandidateRecord],
    train_engine: CPUBacktestEngine,
    valid_engine: CPUBacktestEngine,
    direction: str,
) -> tuple[list[CandidateRecord], dict, dict, dict[str, Any]]:
    """Report and optionally prune negative leave-one-out rule contributions.

    The validation engine is the selection evidence used for the marginal
    decision.  The train engine is re-evaluated after pruning so downstream RB
    risk tuning always receives exact metrics for the actual selected rules.
    At most two pruning passes are allowed; report-only mode performs one
    diagnostic pass and never changes the greedy result.
    """
    if not selected:
        empty = _empty_marginal_report(direction)
        return [], {}, {}, empty

    current = list(selected)
    original_positions = {
        id(record): index + 1 for index, record in enumerate(selected)
    }
    pruning_enabled = bool(getattr(_cfg, "RB_MARGINAL_PRUNING", False))
    effective_enabled = bool(
        getattr(_cfg, "RB_EFFECTIVE_RULE_COUNT_ENABLED", True)
    )
    configured_min_rules = max(0, int(getattr(_cfg, "RB_MIN_RULES", 1)))
    entries_by_identity: dict[int, dict[str, Any]] = {}
    pass_reports: list[dict[str, Any]] = []
    max_passes = 2 if pruning_enabled else 1
    stable = False

    for pass_number in range(1, max_passes + 1):
        if not current:
            stable = True
            break
        current_rules = [record.rule for record in current]
        full_metrics = _marginal_exact_metrics(
            valid_engine,
            current_rules,
            direction,
        )
        pass_entries: list[dict[str, Any]] = []
        negative_records: list[tuple[float, int, CandidateRecord]] = []
        for local_index, record in enumerate(current):
            without_rules = [
                other.rule
                for other_index, other in enumerate(current)
                if other_index != local_index
            ]
            without_metrics = _marginal_exact_metrics(
                valid_engine,
                without_rules,
                direction,
            )
            contribution = marginal_contribution(full_metrics, without_metrics)
            original_index = original_positions.get(id(record), local_index + 1)
            rule = record.rule
            rule_id = str(
                rule.get("phase2_rule_id")
                or rule.get("rule_id")
                or f"rule-{original_index}"
            )
            entry: dict[str, Any] = {
                "rule_index": int(original_index),
                "current_index": int(local_index + 1),
                "rule_id": rule_id,
                "phase2_rule_id": str(rule.get("phase2_rule_id", "")),
                "conditions": list(rule.get("conditions", [])),
                **contribution,
                "pass": int(pass_number),
                "removed": False,
                "retained": True,
            }
            pass_entries.append(entry)
            entries_by_identity[id(record)] = entry
            if not bool(contribution["is_beneficial"]):
                negative_records.append(
                    (
                        float(contribution["ΔReturn"]),
                        int(original_index),
                        record,
                    )
                )

        removable_count = max(0, len(current) - configured_min_rules)
        to_remove: list[CandidateRecord] = []
        if pruning_enabled and removable_count and negative_records:
            # Remove the most negative contributors first.  This makes the
            # bounded pass deterministic and preserves RB_MIN_RULES.
            negative_records.sort(key=lambda item: (item[0], item[1]))
            to_remove = [
                record
                for _delta_return, _index, record in negative_records[
                    :removable_count
                ]
            ]
            remove_ids = {id(record) for record in to_remove}
            for record in to_remove:
                entry = entries_by_identity[id(record)]
                entry["removed"] = True
                entry["retained"] = False
                entry["removal_reason"] = "negative_marginal_contribution"
            current = [record for record in current if id(record) not in remove_ids]

        pass_reports.append({
            "pass": int(pass_number),
            "rules_before": len(pass_entries),
            "rules_removed": len(to_remove),
            "rules_after": len(current),
            "full_metrics": {
                "Return": _f(full_metrics, "total_return_pct"),
                "Sortino": _f(full_metrics, "sortino_ratio"),
                "MDD": _f(full_metrics, "max_drawdown_pct"),
                "PF": _f(full_metrics, "profit_factor"),
                "WorstMonth": _f(full_metrics, "WorstMonth"),
            },
            "per_rule": pass_entries,
        })

        if not to_remove:
            stable = True
            break

    final_rules = [record.rule for record in current]
    final_train = _marginal_exact_metrics(train_engine, final_rules, direction)
    final_valid = _marginal_exact_metrics(valid_engine, final_rules, direction)
    matrix = _marginal_redundancy_matrix(current)
    effective = (
        effective_rule_count(matrix) if effective_enabled else None
    )
    per_rule = list(entries_by_identity.values())
    per_rule.sort(key=lambda row: int(row.get("rule_index", 0)))
    report: dict[str, Any] = {
        "direction": direction,
        "metric_delta_contract": "full_minus_without_i",
        "pruning_enabled": pruning_enabled,
        "effective_rule_count_enabled": effective_enabled,
        "passes": pass_reports,
        "per_rule": per_rule,
        # ``rules`` is a descriptive alias used by report consumers.
        "rules": per_rule,
        "selected_rule_count": len(current),
        "removed_rule_count": len(selected) - len(current),
        "effective_independent_rules": effective,
        "effective_rule_count": effective,
        "redundancy_matrix": matrix.tolist(),
        "stable": bool(stable),
        "pruned": bool(len(current) != len(selected)),
    }
    return current, final_train, final_valid, report


# Public descriptive alias for callers that want to run the bounded pass
# outside the pipeline, while the underscore name keeps the governor API
# private for existing integrations.
prune_negative_marginal = _marginal_prune_ruleset


def _compose_ruleset(
    candidates: list[CandidateRecord],
    train_engine: CPUBacktestEngine,
    valid_engine: CPUBacktestEngine,
    direction: str,
    *,
    tail_holdout_engine: CPUBacktestEngine | None = None,
    min_distinct_symbols: int | None = None,
    concentration_max_share: float | None = None,
    concentration_max_hhi: float | None = None,
) -> tuple[list[CandidateRecord], dict, dict, float, list[dict]]:
    if not candidates:
        raise ValueError(f"No rb-positive rules available for {direction}")

    correlation_report, correlation_matrix, correlation_clusters, cluster_map = (
        _correlation_profile(candidates)
    )
    candidate_indices = {
        _rule_key(candidate.rule): index
        for index, candidate in enumerate(candidates)
    }
    correlation_enabled = bool(
        getattr(_cfg, "RB_CORRELATION_AWARE_SELECTION", False)
    )
    redundancy_lambda = _redundancy_lambda()

    tail_eligible = [
        candidate for candidate in candidates
        if _passes_tail_selection_gate(tail_holdout_engine, [candidate.rule])[0]
    ]
    if tail_holdout_engine is not None and not tail_eligible:
        logger.warning(
            "RB [%s]: no single candidate passed the reserved tail gate; "
            "retaining the ordinary seed so the final hard gate fails closed.",
            direction,
        )
    seed_candidates = tail_eligible or candidates
    selected: list[CandidateRecord] = [seed_candidates[0]]
    cur_train, cur_valid, cur_score = _evaluate_ruleset(
        train_engine, valid_engine, [selected[0].rule])
    history = [{
        "step": 1,
        "action": "seed",
        "score": cur_score,
        "train_return_pct": _f(cur_train, "total_return_pct"),
        "valid_return_pct": _f(cur_valid, "total_return_pct"),
        "rules": 1,
    }]

    max_rules = int(_cfg.RB_MAX_RULES)
    if min_distinct_symbols is not None:
        min_distinct_symbols = max(0, int(min_distinct_symbols))
    elif bool(getattr(_cfg, "DEBUG_SYMBOL_SCOPE_ENABLED", False)):
        active_symbol_count = len(
            _available_symbols(
                getattr(train_engine, "df", None),
                getattr(valid_engine, "df", None),
            )
        )
        min_distinct_symbols = int(
            _cfg.effective_rb_min_distinct_symbols(active_symbol_count)
        )
    else:
        min_distinct_symbols = int(getattr(_cfg, "RB_MIN_DISTINCT_SYMBOLS", 0))

    beam_result = _diversification_beam(
        candidates,
        train_engine,
        valid_engine,
        direction,
        min_distinct_symbols=min_distinct_symbols,
        tail_holdout_engine=tail_holdout_engine,
    )
    if beam_result is not None:
        (
            selected,
            cur_train,
            cur_valid,
            cur_score,
            beam_history,
        ) = beam_result
        history = beam_history
        logger.info(
            "RB [%s]: diversification beam selected %d-rule certified seed "
            "covering %s",
            direction,
            len(selected),
            sorted(_passes_symbol_contribution_certificate(
                cur_valid,
                min_symbols=0,
            )[1].get("qualifying_symbols", [])),
        )

    # Preserve the report in the first history row for both the ordinary and
    # beam paths.  Disabled correlation remains explicitly report-only.
    if history:
        history[0]["correlation_report"] = correlation_report

    max_overlap = float(getattr(_cfg, "RB_MAX_PAIR_OVERLAP", 0.22))
    min_score_improve = float(getattr(_cfg, "RB_MIN_SCORE_IMPROVEMENT", 0.05))
    min_train_ret_improve = float(
        getattr(_cfg, "RB_MIN_TRAIN_RETURN_IMPROVEMENT", 0.01))
    min_valid_ret_improve = float(
        getattr(_cfg, "RB_MIN_VALID_RETURN_IMPROVEMENT", 0.01))
    require_subset_improve = bool(
        getattr(_cfg, "RB_RULESET_MUST_BEAT_SUBSETS", True))
    return_only_add = bool(getattr(_cfg, "RB_RULE_ADD_BY_RETURN_ONLY", False))
    ignore_overlap = bool(getattr(_cfg, "RB_RULE_ADD_IGNORE_OVERLAP", False))
    if return_only_add:
        require_subset_improve = False if bool(getattr(
            _cfg, "RB_RULE_ADD_IGNORE_SUBSET_BEAT", True)) else require_subset_improve
        cur_return_score = _combined_return_score(
            cur_train, cur_valid, prev_pf=None, prev_dd=None)
        min_return_improve = float(
            getattr(_cfg, "RB_MIN_COMBINED_RETURN_IMPROVEMENT", 0.05))

    used = {_rule_key(r.rule) for r in selected}
    ordered_indices = list(range(len(candidates)))
    if correlation_enabled:
        ordered_indices = [
            int(index) for index in correlation_report.get("selected_order", [])
            if 0 <= int(index) < len(candidates)
        ]
        ordered_indices.extend(
            index for index in range(len(candidates))
            if index not in ordered_indices
        )
    while len(selected) < max_rules:
        best: tuple[float, CandidateRecord, dict, dict] | None = None
        best_cross: tuple[float, CandidateRecord, dict, dict] | None = None
        selected_indices = {
            candidate_indices.get(_rule_key(record.rule))
            for record in selected
        }
        selected_indices.discard(None)
        selected_cluster_ids = {
            cluster_map[index] for index in selected_indices
            if index in cluster_map
        }
        unrepresented_cluster_ids = {
            cluster_id for cluster_id in cluster_map.values()
            if cluster_id not in selected_cluster_ids
        }
        for candidate_index in ordered_indices:
            cand = candidates[candidate_index]
            key = _rule_key(cand.rule)
            if key in used:
                continue
            candidate_cluster = cluster_map.get(candidate_index)
            is_cross_cluster = (
                candidate_cluster is not None
                and candidate_cluster not in selected_cluster_ids
            )
            ov = _max_overlap(cand, selected)
            if (not ignore_overlap) and ov > max_overlap:
                continue
            trial_recs = selected + [cand]
            trial_rules = [r.rule for r in trial_recs]
            if min_distinct_symbols > 0:
                # Specialist mode: count explicit ``symbol is X`` filters.
                # Generalist mode: count traded symbols from backtest metrics
                # (pool rules have no filters; filter-only logic blocked all adds).
                if bool(getattr(_cfg, "RB_REQUIRE_SYMBOL_FILTERS", False)):
                    selected_syms = _symbols_in_rules(
                        [r.rule for r in selected])
                    cand_syms = _symbols_in_rules([cand.rule])
                else:
                    selected_syms = set()
                    for rec in selected:
                        selected_syms |= _candidate_coverage_symbols(rec)
                    cand_syms = _candidate_coverage_symbols(cand)
                if len(selected_syms) < min_distinct_symbols and not (cand_syms - selected_syms):
                    continue
            train_m, valid_m, score = _evaluate_ruleset(
                train_engine, valid_engine, trial_rules)
            tail_ok, _ = _passes_tail_selection_gate(
                tail_holdout_engine, trial_rules,
            )
            if not tail_ok:
                continue
            if return_only_add:
                if not _positive_returns(train_m, valid_m):
                    continue
                prev_pf = _f(cur_valid, "profit_factor", 0.0)
                prev_dd = _f(cur_valid, "max_drawdown_pct", 0.0)
                ret_score = _combined_return_score(
                    train_m, valid_m, prev_pf=prev_pf, prev_dd=prev_dd)
                if ret_score <= cur_return_score + min_return_improve:
                    continue
                choose_score = ret_score
            else:
                if not _is_positive_good(
                    train_m,
                    valid_m,
                    min_train_trades=int(getattr(_cfg, "RB_RULESET_MIN_TRAIN_TRADES", getattr(
                        _cfg, "RB_MIN_TRAIN_TRADES", 25))),
                    min_valid_trades=int(getattr(_cfg, "RB_RULESET_MIN_VALID_TRADES", getattr(
                        _cfg, "RB_MIN_VALID_TRADES", 15))),
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
            correlation_penalty = 0.0
            if correlation_enabled and selected_indices:
                correlation_penalty = float(
                    np.max(correlation_matrix[candidate_index, list(selected_indices)])
                )
                choose_score = adjusted_quality(
                    choose_score,
                    correlation_penalty,
                    lambda_=redundancy_lambda,
                )
            # Once the beam has found a certified seed, score growth must not
            # reintroduce a symbol imbalance or discard a positive contributor.
            cert_ok, _cert_detail = _portfolio_selection_certificate(
                valid_m,
                min_symbols=min_distinct_symbols,
                concentration_max_share=concentration_max_share,
                concentration_max_hhi=concentration_max_hhi,
            )
            if not cert_ok:
                continue
            if best is None or choose_score > best[0]:
                best = (choose_score, cand, train_m, valid_m)
            if (
                correlation_enabled
                and unrepresented_cluster_ids
                and is_cross_cluster
                and (best_cross is None or choose_score > best_cross[0])
            ):
                best_cross = (choose_score, cand, train_m, valid_m)

        if best_cross is not None:
            best = best_cross

        if best is None:
            logger.info(
                "RB [%s]: no further positive/improving low-overlap extension found at %d rules.", direction, len(selected))
            break

        chosen_score, chosen, cur_train, cur_valid = best
        cur_score = _score_metrics(
            cur_train,
            cur_valid,
            min_train_trades=int(getattr(_cfg, "RB_RULESET_MIN_TRAIN_TRADES", getattr(
                _cfg, "RB_MIN_TRAIN_TRADES", 25))),
            min_valid_trades=int(getattr(_cfg, "RB_RULESET_MIN_VALID_TRADES", getattr(
                _cfg, "RB_MIN_VALID_TRADES", 15))),
        )
        if return_only_add:
            cur_return_score = _combined_return_score(
                cur_train, cur_valid, prev_pf=None, prev_dd=None)
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
            direction, len(selected), cur_score, _f(
                cur_train, "total_return_pct"), _f(cur_valid, "total_return_pct"),
        )

    return selected, cur_train, cur_valid, cur_score, history


def _make_walk_forward_fold_engines(
    val_selection_df: pd.DataFrame,
    n_splits: int,
    tail_holdout_frac: float,
    direction: str,
) -> tuple[list[CPUBacktestEngine], CPUBacktestEngine | None]:
    # → fixes audit finding #3 (RB Governor risk-grid overfits val_selection)
    # The risk grid uses the active RB tail-holdout contract.
    """Split val_selection into n_splits chronological folds + optional tail holdout.

    Per-symbol chronological split (matches data/splitter.py convention).
    Returns (fold_engines, tail_holdout_engine_or_None).
    """
    if "symbol" not in val_selection_df.columns or "datetime" not in val_selection_df.columns:
        # Single-symbol or no-symbol data: treat entire df as one symbol
        symbols = ["_all"]
        sym_data = {"_all": val_selection_df.copy().sort_values(
            "datetime").reset_index(drop=True)}
    else:
        sym_data = {}
        df_sorted = val_selection_df.copy().sort_values(
            ["symbol", "datetime"]).reset_index(drop=True)
        for sym, grp in df_sorted.groupby(
            "symbol", sort=False, observed=False,
        ):
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
                fold_dfs[i] = pd.concat(
                    [fold_dfs[i], head_df.iloc[start:end]], ignore_index=True)

    # Build fold engines
    fold_engines: list[CPUBacktestEngine] = []
    for fold_df in fold_dfs:
        if len(fold_df) == 0:
            # Empty fold — pad with a copy of another fold's data
            fallback = next(
                (fd for fd in fold_dfs if len(fd) > 0), val_selection_df)
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
    *,
    min_distinct_symbols: int | None = None,
    concentration_max_share: float | None = None,
    concentration_max_hhi: float | None = None,
    immutable_exits: bool | None = None,
) -> tuple[list[dict], dict, dict, float, list[dict]]:
    rules = [_rule_to_engine(r.rule) for r in selected]
    cur_train, cur_valid, cur_score = _evaluate_ruleset(
        train_engine, valid_engine, rules)
    best_rules = [dict(r) for r in rules]
    rejected_certificates: list[dict[str, Any]] = []

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
    _, initial_cert = _portfolio_selection_certificate(
        cur_valid,
        min_symbols=min_distinct_symbols,
        concentration_max_share=concentration_max_share,
        concentration_max_hhi=concentration_max_hhi,
    )
    hist[0]["portfolio_certificate"] = initial_cert
    if use_walk_forward:
        # Compute initial fold scores
        init_fold_scores: list[float] = []
        for fold_engine in fold_engines:
            _, fold_m, fold_s = _evaluate_ruleset(
                train_engine, fold_engine, rules)
            init_fold_scores.append(fold_s)
        hist[0]["fold_scores"] = init_fold_scores
        hist[0]["min_fold_score"] = min(init_fold_scores)
        # fix(task-3): recompute cur_score as min(init_fold_scores) so the
        # improvement threshold compares fold-min against fold-min (same scale).
        # Without this, cur_score remains the full-validation score (higher),
        # and walk-forward becomes overly conservative.
        cur_score = min(init_fold_scores)
        hist[0]["score"] = cur_score

    tp_grid = tuple(float(x) for x in _cfg.RB_TP_GRID)
    sl_grid = tuple(float(x) for x in _cfg.RB_SL_GRID)
    cap_grid = tuple(float(x) for x in _cfg.RB_CAPITAL_GRID)
    max_total_cap = float(_cfg.RB_MAX_TOTAL_CAPITAL)
    passes = int(_cfg.RB_RISK_OPT_PASSES)
    min_improve = float(getattr(_cfg, "RB_RISK_MIN_IMPROVEMENT", 0.02))
    min_train_trades = int(getattr(
        _cfg, "RB_RULESET_MIN_TRAIN_TRADES", getattr(_cfg, "RB_MIN_TRAIN_TRADES", 25)))
    min_valid_trades = int(getattr(
        _cfg, "RB_RULESET_MIN_VALID_TRADES", getattr(_cfg, "RB_MIN_VALID_TRADES", 15)))

    strict_exit_contract = (
        bool(getattr(_cfg, "RB_RISK_OPTIMIZE_EXITS", False))
        if immutable_exits is None
        else bool(immutable_exits)
    )
    for p in range(1, passes + 1):
        improved = False
        for idx in range(len(best_rules)):
            local_best: tuple[float, list[dict], dict,
                              dict, list[float] | None] | None = None
            if not strict_exit_contract:
                candidate_tp_grid = tp_grid
                candidate_sl_grid = sl_grid
            else:
                # TP/SL are immutable entry-strategy semantics. RB still
                # searches capital allocation, but never changes the exit
                # geometry discovered and admitted by Phase 2.
                candidate_tp_grid = (
                    float(best_rules[idx].get("tp", _cfg.RB_DEFAULT_TP)),
                )
                candidate_sl_grid = (
                    float(best_rules[idx].get("sl", _cfg.RB_DEFAULT_SL)),
                )
            for tp in candidate_tp_grid:
                for sl in candidate_sl_grid:
                    for cap in cap_grid:
                        trial = [dict(r) for r in best_rules]
                        trial[idx]["tp"] = tp
                        trial[idx]["sl"] = sl
                        trial[idx]["capital_pct"] = cap
                        if sum(float(r.get("capital_pct", 0.0)) for r in trial) > max_total_cap:
                            continue
                        train_m, valid_m, score = _evaluate_ruleset(
                            train_engine, valid_engine, trial)
                        if not _is_positive_good(train_m, valid_m, min_train_trades=min_train_trades, min_valid_trades=min_valid_trades):
                            continue
                        cert_ok, cert_detail = _portfolio_selection_certificate(
                            valid_m,
                            min_symbols=min_distinct_symbols,
                            concentration_max_share=concentration_max_share,
                            concentration_max_hhi=concentration_max_hhi,
                        )
                        if not cert_ok:
                            rejected_certificates.append({
                                "stage": "risk_grid",
                                "pass": p,
                                "rule_index": idx + 1,
                                "tp": tp,
                                "sl": sl,
                                "capital_pct": cap,
                                "certificate": cert_detail,
                            })
                            continue

                        tail_ok, _tail_detail = _passes_tail_selection_gate(
                            tail_holdout_engine, trial,
                        )
                        if not tail_ok:
                            continue

                        if use_walk_forward:
                            # Score on each fold engine and compute min fold score
                            fold_scores_local: list[float] = []
                            all_folds_pass = True
                            for fold_engine in fold_engines:
                                _, fold_m, fold_s = _evaluate_ruleset(
                                    train_engine, fold_engine, trial)
                                # A WF fold is intentionally shorter than the
                                # complete validation-selection frame.  Scale
                                # the absolute validation trade floor by its
                                # row fraction (with a small floor) instead of
                                # making every 3-month fold fail solely because
                                # it cannot produce the full-frame count.
                                fold_rows = len(getattr(fold_engine, "df", ()))
                                valid_rows = max(
                                    1, len(getattr(valid_engine, "df", ()))
                                )
                                fold_min_trades = max(
                                    3,
                                    int(math.ceil(
                                        min_valid_trades
                                        * min(1.0, fold_rows / valid_rows)
                                    )),
                                )
                                if not _is_positive_good(
                                    train_m,
                                    fold_m,
                                    min_train_trades=min_train_trades,
                                    min_valid_trades=fold_min_trades,
                                ):
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
                            local_best = (selection_score, trial,
                                          train_m, valid_m, fold_scores_local)
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
                    direction, p, idx +
                    1, cur_score, _f(cur_train, "total_return_pct"), _f(
                        cur_valid, "total_return_pct"),
                )
        if not improved:
            break

    # Tail holdout scoring on final selected combo (not used during search)
    if tail_holdout_engine is not None and hist:
        _, tail_m, _ = _evaluate_ruleset(
            train_engine, tail_holdout_engine, best_rules)
        final_entry = hist[-1]
        final_entry["risk_tail_holdout_return_pct"] = _f(
            tail_m, "total_return_pct")
        final_entry["risk_tail_holdout_pf"] = _f(tail_m, "profit_factor")
        final_entry["risk_tail_holdout_dd"] = _f(tail_m, "max_drawdown_pct")

    hist[0]["rejected_portfolio_certificates"] = rejected_certificates
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
    score = train_w * train_ret + valid_w * \
        valid_ret + balance_w * min(train_ret, valid_ret)
    score -= dd_w * (_f(train_m, "max_drawdown_pct", 100.0) +
                     1.35 * _f(valid_m, "max_drawdown_pct", 100.0))
    score -= health_w * (_evaluator_health_penalty(train_m, role="train") +
                         _evaluator_health_penalty(valid_m, role="valid"))
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
    *,
    min_symbols: int | None = None,
    concentration_max_share: float | None = None,
    concentration_max_hhi: float | None = None,
) -> tuple[bool, dict]:
    """Return whether a ruleset is robust enough to enter profit-first selection."""
    min_train_trades = int(getattr(
        _cfg, "RB_RULESET_MIN_TRAIN_TRADES", getattr(_cfg, "RB_MIN_TRAIN_TRADES", 25)))
    min_valid_trades = int(getattr(
        _cfg, "RB_RULESET_MIN_VALID_TRADES", getattr(_cfg, "RB_MIN_VALID_TRADES", 15)))
    ok = _is_positive_good(
        train_m, valid_m, min_train_trades=min_train_trades, min_valid_trades=min_valid_trades)
    reasons: list[str] = []
    if not ok:
        reasons.append("full_sample_positive_good_failed")
    portfolio_ok, portfolio_certificate = _portfolio_selection_certificate(
        valid_m,
        min_symbols=min_symbols,
        concentration_max_share=concentration_max_share,
        concentration_max_hhi=concentration_max_hhi,
    )
    if not portfolio_ok:
        ok = False
        reasons.extend(portfolio_certificate.get("reasons", []))
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
        "portfolio_certificate": portfolio_certificate,
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
    min_symbols: int | None = None,
    concentration_max_share: float | None = None,
    concentration_max_hhi: float | None = None,
) -> tuple[dict, dict, float, tuple[bool, dict], list[dict]]:
    """Evaluate a trial ruleset and attach the hard trust certificate used by the amplifier."""
    train_m, valid_m, _ = _evaluate_ruleset(train_engine, valid_engine, rules)
    monthly_summary, monthly_rows = _profit_amp_monthly_summary(valid_df, rules, direction) if check_monthly else (None, [])
    certificate = _profit_amp_certificate(
        train_m,
        valid_m,
        monthly_summary,
        min_symbols=min_symbols,
        concentration_max_share=concentration_max_share,
        concentration_max_hhi=concentration_max_hhi,
    )
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
    rejection_sink: list[dict[str, Any]] | None = None,
    *,
    min_symbols: int | None = None,
    concentration_max_share: float | None = None,
    concentration_max_hhi: float | None = None,
) -> tuple[list[dict], dict, dict, float, tuple[bool, dict], list[dict], list[dict]] | None:
    """Greedily select rules by marginal profit while keeping every trial certificate-safe."""
    max_rules = int(_cfg.RB_MAX_RULES)
    max_overlap = float(getattr(_cfg, "RB_PROFIT_AMP_MAX_PAIR_OVERLAP", 0.55))
    min_improve = float(getattr(_cfg, "RB_PROFIT_AMP_MIN_OBJECTIVE_IMPROVEMENT", 0.10))
    min_return_improve = float(getattr(_cfg, "RB_PROFIT_AMP_MIN_RETURN_IMPROVEMENT", 0.05))
    overlap_penalty = float(getattr(_cfg, "RB_PROFIT_AMP_OVERLAP_PENALTY", 2.5))
    ranked = sorted(candidates, key=lambda r: _profit_amp_objective(r.train_metrics, r.valid_metrics), reverse=True)
    seed_best: tuple[float, CandidateRecord, dict, dict, float, tuple[bool, dict], list[dict]] | None = None
    rejected_certificates: list[dict[str, Any]] = []
    for cand in ranked:
        train_m, valid_m, objective, cert, monthly_rows = _profit_amp_evaluate_candidate(
            train_engine,
            valid_engine,
            valid_df,
            [cand.rule],
            direction,
            check_monthly=False,
            min_symbols=min_symbols,
            concentration_max_share=concentration_max_share,
            concentration_max_hhi=concentration_max_hhi,
        )
        if not cert[0]:
            rejected_certificates.append({
                "stage": "profit_amplifier_seed",
                "rule": _rule_to_engine(cand.rule),
                "certificate": cert[1],
            })
            continue
        if seed_best is None or objective > seed_best[0]:
            seed_best = (objective, cand, train_m, valid_m, objective, cert, monthly_rows)
    if seed_best is None:
        if rejection_sink is not None:
            rejection_sink.extend(rejected_certificates)
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
            train_m, valid_m, objective, cert, monthly_rows = _profit_amp_evaluate_candidate(
                train_engine,
                valid_engine,
                valid_df,
                trial_rules,
                direction,
                check_monthly=False,
                min_symbols=min_symbols,
                concentration_max_share=concentration_max_share,
                concentration_max_hhi=concentration_max_hhi,
            )
            if not cert[0]:
                rejected_certificates.append({
                    "stage": "profit_amplifier_add",
                    "rule": _rule_to_engine(cand.rule),
                    "certificate": cert[1],
                })
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
    cur_train, cur_valid, cur_objective, cur_cert, cur_monthly_rows = _profit_amp_evaluate_candidate(
        train_engine,
        valid_engine,
        valid_df,
        final_rules,
        direction,
        check_monthly=True,
        min_symbols=min_symbols,
        concentration_max_share=concentration_max_share,
        concentration_max_hhi=concentration_max_hhi,
    )
    history[0]["rejected_portfolio_certificates"] = rejected_certificates
    if rejection_sink is not None:
        rejection_sink.extend(rejected_certificates)
    return final_rules, cur_train, cur_valid, cur_objective, cur_cert, history, cur_monthly_rows


def _profit_amp_reallocate_capital(
    rules: list[dict],
    train_engine: CPUBacktestEngine,
    valid_engine: CPUBacktestEngine,
    valid_df: pd.DataFrame,
    direction: str,
    *,
    min_symbols: int | None = None,
    concentration_max_share: float | None = None,
    concentration_max_hhi: float | None = None,
) -> tuple[list[dict], dict, dict, float, tuple[bool, dict], list[dict], list[dict]]:
    """Shift capital toward profit-contributing certified rules without changing rule conditions."""
    best_rules = [_rule_to_engine(r) for r in rules]
    cur_train, cur_valid, cur_objective, cur_cert, cur_monthly_rows = _profit_amp_evaluate_candidate(
        train_engine,
        valid_engine,
        valid_df,
        best_rules,
        direction,
        min_symbols=min_symbols,
        concentration_max_share=concentration_max_share,
        concentration_max_hhi=concentration_max_hhi,
    )
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
    cap_grid = tuple(float(x) for x in _cfg.RB_CAPITAL_GRID)
    max_total_cap = float(getattr(_cfg, "RB_MAX_TOTAL_CAPITAL", 95.0))
    passes = int(getattr(_cfg, "RB_PROFIT_AMP_CAPITAL_PASSES", 2))
    min_improve = float(getattr(_cfg, "RB_PROFIT_AMP_MIN_OBJECTIVE_IMPROVEMENT", 0.10))
    rejected_certificates: list[dict[str, Any]] = []
    for pno in range(1, passes + 1):
        improved = False
        for idx in range(len(best_rules)):
            local: tuple[float, list[dict], dict, dict, tuple[bool, dict], list[dict]] | None = None
            for cap in cap_grid:
                trial = [dict(r) for r in best_rules]
                trial[idx]["capital_pct"] = cap
                if sum(float(r.get("capital_pct", 0.0)) for r in trial) > max_total_cap:
                    continue
                train_m, valid_m, objective, cert, monthly_rows = _profit_amp_evaluate_candidate(
                    train_engine,
                    valid_engine,
                    valid_df,
                    trial,
                    direction,
                    check_monthly=False,
                    min_symbols=min_symbols,
                    concentration_max_share=concentration_max_share,
                    concentration_max_hhi=concentration_max_hhi,
                )
                if not cert[0]:
                    rejected_certificates.append({
                        "stage": "profit_amplifier_capital",
                        "pass": pno,
                        "rule_index": idx + 1,
                        "capital_pct": cap,
                        "certificate": cert[1],
                    })
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
    cur_train, cur_valid, cur_objective, cur_cert, cur_monthly_rows = _profit_amp_evaluate_candidate(
        train_engine,
        valid_engine,
        valid_df,
        best_rules,
        direction,
        check_monthly=True,
        min_symbols=min_symbols,
        concentration_max_share=concentration_max_share,
        concentration_max_hhi=concentration_max_hhi,
    )
    history[0]["rejected_portfolio_certificates"] = rejected_certificates
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
    *,
    min_symbols: int | None = None,
    concentration_max_share: float | None = None,
    concentration_max_hhi: float | None = None,
) -> tuple[list[dict], dict, dict, float, dict]:
    """Run robust-certified profit amplification and keep the baseline unless profit improves."""
    baseline_monthly, baseline_monthly_rows = _profit_amp_monthly_summary(valid_df, baseline_rules, direction)
    baseline_cert = _profit_amp_certificate(
        baseline_train,
        baseline_valid,
        baseline_monthly,
        min_symbols=min_symbols,
        concentration_max_share=concentration_max_share,
        concentration_max_hhi=concentration_max_hhi,
    )
    baseline_objective = _profit_amp_objective(baseline_train, baseline_valid)
    meta = {
        "enabled": bool(getattr(_cfg, "RB_PROFIT_AMPLIFIER_ENABLED", True)),
        "accepted": False,
        "baseline_profit_amp_objective": baseline_objective,
        "baseline_certificate": baseline_cert[1],
        "baseline_monthly_metrics": baseline_monthly_rows,
        "rejected_portfolio_certificates": [],
        "reason": "disabled_or_not_improved",
    }
    if not baseline_cert[0]:
        meta["rejected_portfolio_certificates"].append({
            "stage": "profit_amplifier_baseline",
            "certificate": baseline_cert[1],
        })
    if not bool(getattr(_cfg, "RB_PROFIT_AMPLIFIER_ENABLED", True)):
        meta["reason"] = "disabled"
        return baseline_rules, baseline_train, baseline_valid, baseline_objective, meta
    ranked_candidates = _profit_amp_rank_candidates(candidates, baseline_rules, train_engine, valid_engine, train_like_df, valid_df)
    selection_rejections: list[dict[str, Any]] = []
    selection = _profit_amp_select_rules(
        ranked_candidates,
        train_engine,
        valid_engine,
        train_like_df,
        valid_df,
        direction,
        rejection_sink=selection_rejections,
        min_symbols=min_symbols,
        concentration_max_share=concentration_max_share,
        concentration_max_hhi=concentration_max_hhi,
    )
    if selection is None:
        meta["reason"] = "no_certified_profit_seed"
        meta["rejected_portfolio_certificates"].extend(selection_rejections)
        return baseline_rules, baseline_train, baseline_valid, baseline_objective, meta
    amp_rules, amp_train, amp_valid, amp_objective, amp_cert, select_history, amp_monthly_rows = selection
    capital_history: list[dict] = []
    if bool(getattr(_cfg, "RB_PROFIT_AMP_CAPITAL_REALLOCATION_ENABLED", True)):
        amp_rules, amp_train, amp_valid, amp_objective, amp_cert, capital_history, amp_monthly_rows = _profit_amp_reallocate_capital(
            amp_rules,
            train_engine,
            valid_engine,
            valid_df,
            direction,
            min_symbols=min_symbols,
            concentration_max_share=concentration_max_share,
            concentration_max_hhi=concentration_max_hhi,
        )
    rejected_certificates: list[dict] = list(
        meta.get("rejected_portfolio_certificates", [])
    )
    for row in [*select_history, *capital_history]:
        rejected_certificates.extend(
            row.get("rejected_portfolio_certificates", [])
            if isinstance(row, dict) else []
        )
    if not amp_cert[0]:
        rejected_certificates.append({
            "stage": "profit_amplifier_final",
            "certificate": amp_cert[1],
        })
    min_improve = float(getattr(_cfg, "RB_PROFIT_AMP_MIN_OBJECTIVE_IMPROVEMENT", 0.10))
    keep_baseline = bool(getattr(_cfg, "RB_PROFIT_AMP_KEEP_BASELINE_UNLESS_BETTER", True))
    if keep_baseline and amp_objective <= baseline_objective + min_improve:
        meta.update({
            "reason": "amplified_not_better_than_baseline",
            "amplified_profit_amp_objective": amp_objective,
            "amplified_certificate": amp_cert[1],
            "selection_history": select_history,
            "capital_history": capital_history,
            "rejected_portfolio_certificates": rejected_certificates,
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
            "rejected_portfolio_certificates": rejected_certificates,
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
        "rejected_portfolio_certificates": rejected_certificates,
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


def _write_marginal_report(
    reports_dir: Path,
    report: dict[str, Any],
) -> None:
    """Write the aggregate marginal contribution report."""
    directions = report.get("directions", {})
    if isinstance(directions, dict):
        if len(directions) == 1:
            only_direction = next(iter(directions.values()))
            if isinstance(only_direction, dict):
                # Keep the single-direction shape convenient for lightweight
                # report consumers while retaining the direction map for
                # normal long/short pipeline runs.
                report["per_rule"] = only_direction.get("per_rule", [])
                report["rules"] = only_direction.get("rules", [])
                report["marginal_contributions"] = report["per_rule"]
                report["effective_independent_rules"] = only_direction.get(
                    "effective_independent_rules",
                    0.0,
                )
                report["effective_rule_count"] = only_direction.get(
                    "effective_rule_count",
                    0.0,
                )
        else:
            report["per_rule"] = {
                str(direction): detail.get("per_rule", [])
                for direction, detail in directions.items()
                if isinstance(detail, dict)
            }
            report["rules"] = report["per_rule"]
            report["marginal_contributions"] = report["per_rule"]
            report["effective_independent_rules"] = {
                str(direction): detail.get("effective_independent_rules", 0.0)
                for direction, detail in directions.items()
                if isinstance(detail, dict)
            }
            report["effective_rule_count"] = report[
                "effective_independent_rules"
            ]
    reports_dir.mkdir(parents=True, exist_ok=True)
    with (reports_dir / "marginal_contribution.json").open(
        "w",
        encoding="utf-8",
    ) as fh:
        json.dump(report, fh, indent=2, default=str)


def _write_fail_closed_strategy(
    out_dir: Path,
    reports_dir: Path,
    direction: str,
    reason: str,
    *,
    phase2_status: dict | None = None,
) -> dict:
    """Persist an explicit empty strategy and diagnostic report."""

    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    strategy = _strategy(
        direction,
        [],
        risk_optimized=False,
        extra={
            "deployment_accepted": False,
            "fail_closed": True,
            "reason": reason,
            "phase2_status": phase2_status or {},
        },
    )
    strategy_path = out_dir / f"{direction}.json"
    with strategy_path.open("w", encoding="utf-8") as fh:
        json.dump(strategy, fh, indent=2)
    _write_clean_evaluator(
        strategy,
        out_dir / "evaluator_clean" / f"{direction}_evaluator_clean.json",
    )
    report = {
        "direction": direction,
        "rb_score": 0.0,
        "train_metrics": {},
        "valid_metrics": {},
        "n_positive_single_rules": 0,
        "selected_rules": 0,
        "compose_history": [],
        "risk_history": [],
        "profit_amplifier": {"accepted": False, "reason": reason},
        "top_single_rules": [],
        "fail_closed": True,
        "reason": reason,
        "phase2_status": phase2_status or {},
    }
    with (reports_dir / f"rb_governor_{direction}_report.json").open(
        "w", encoding="utf-8"
    ) as fh:
        json.dump(report, fh, indent=2, default=str)
    logger.info(
        "RB [%s]: fail-closed empty strategy written (reason=%s).",
        direction,
        reason,
    )
    return strategy


def _select_recency_candidate(
    candidates: list[CandidateRecord],
    normal_rules: list[dict],
    normal_fitness_m: dict | None,
    normal_valid_m: dict,
    *,
    normal_certificate_ok: bool,
    tail_holdout_engine: CPUBacktestEngine | None = None,
    min_symbols: int | None = None,
    concentration_max_share: float | None = None,
    concentration_max_hhi: float | None = None,
) -> tuple[CandidateRecord | None, dict[str, Any]]:
    """Choose a bounded recency candidate using validation-only evidence.

    The baseline is scored by its weaker chronological validation half.  A
    rescue can replace it when the baseline is not certificate-safe/deployable
    or when the rescue has a better weak-half return by the configured margin.
    The candidate itself must still satisfy the normal Phase 5 validation
    return/PF gates, a relaxed but explicit concentration certificate, and the
    reserved tail gate.
    """
    detail: dict[str, Any] = {
        "enabled": bool(getattr(_cfg, "RB_RECENCY_RESCUE_ENABLED", False)),
        "candidates": len(candidates),
        "selected": False,
        "reason": "disabled_or_empty",
    }
    if not detail["enabled"] or not candidates:
        return None, detail

    baseline_score = -1.0e9
    if normal_fitness_m is not None:
        baseline_score = _recency_validation_score(
            normal_fitness_m, normal_valid_m,
        )
    validation_return_gate = float(_cfg.PHASE5_VALIDATION_RETURN_GATE_PCT)
    validation_pf_gate = float(_cfg.PHASE5_VALIDATION_PROFIT_FACTOR_GATE)
    baseline_gate_ok = (
        _f(normal_valid_m, "total_return_pct") >= validation_return_gate - 1e-9
        and _f(normal_valid_m, "profit_factor") >= validation_pf_gate - 1e-9
    )
    margin = float(getattr(_cfg, "RB_RECENCY_MIN_SCORE_MARGIN", 0.0))
    max_share = float(
        concentration_max_share
        if concentration_max_share is not None
        else getattr(_cfg, "RB_RECENCY_MAX_SYMBOL_SHARE_ABS_PNL", 0.85)
    )
    max_hhi = float(
        concentration_max_hhi
        if concentration_max_hhi is not None
        else getattr(_cfg, "RB_RECENCY_MAX_SYMBOL_HHI", 0.75)
    )

    ranked = sorted(candidates, key=lambda rec: rec.score, reverse=True)
    rejected: list[dict[str, Any]] = []
    for candidate in ranked[:max(
        0, int(getattr(_cfg, "RB_RECENCY_MAX_CANDIDATES", 40))
    )]:
        valid_m = candidate.valid_metrics
        score = float(candidate.score)
        if score < baseline_score + margin - 1e-12 and baseline_gate_ok and normal_certificate_ok:
            rejected.append({
                "conditions": list(candidate.rule.get("conditions", [])),
                "reason": "weak_half_score_not_better",
                "score": score,
            })
            continue
        if _f(valid_m, "total_return_pct") < validation_return_gate - 1e-9:
            rejected.append({
                "conditions": list(candidate.rule.get("conditions", [])),
                "reason": "selection_return_gate",
            })
            continue
        if _f(valid_m, "profit_factor") < validation_pf_gate - 1e-9:
            rejected.append({
                "conditions": list(candidate.rule.get("conditions", [])),
                "reason": "selection_profit_factor_gate",
            })
            continue
        certificate_ok, certificate = _portfolio_selection_certificate(
            valid_m,
            min_symbols=min_symbols,
            concentration_max_share=max_share,
            concentration_max_hhi=max_hhi,
        )
        if not certificate_ok:
            rejected.append({
                "conditions": list(candidate.rule.get("conditions", [])),
                "reason": "recency_portfolio_certificate",
                "certificate": certificate,
            })
            continue
        tail_ok, tail_detail = _passes_tail_selection_gate(
            tail_holdout_engine, [candidate.rule],
        )
        if not tail_ok:
            rejected.append({
                "conditions": list(candidate.rule.get("conditions", [])),
                "reason": "recency_tail_gate",
                "tail_gate": tail_detail,
            })
            continue
        detail.update({
            "selected": True,
            "reason": "weak_half_return_certificate",
            "baseline_weak_half_score": baseline_score,
            "candidate_weak_half_score": score,
            "baseline_gate_ok": baseline_gate_ok,
            "baseline_certificate_ok": bool(normal_certificate_ok),
            "certificate": certificate,
            "tail_gate": tail_detail,
            "rejected": rejected,
            "conditions": list(candidate.rule.get("conditions", [])),
        })
        return candidate, detail
    detail.update({
        "reason": "no_candidate_passed_recency_comparison",
        "baseline_weak_half_score": baseline_score,
        "baseline_gate_ok": baseline_gate_ok,
        "baseline_certificate_ok": bool(normal_certificate_ok),
        "rejected": rejected,
    })
    return None, detail


def run_rb_governor_pipeline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    pools: dict[str, list[dict]],
    directions: list[str] | tuple[str, ...],
    *,
    output_dir: str | os.PathLike[str] | None = None,
    cv_folds: list | None = None,
    val_selection_df: pd.DataFrame | None = None,
    failure_reasons: dict[str, str] | None = None,
    _allow_full_validation_recovery: bool = True,
) -> dict[str, dict]:
    """Build and optimize rb strategies for each direction and write outputs."""
    out_dir = Path(output_dir or _cfg.OUTPUTS_DIR)
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_like, _ = _load_scoring_frames(train_df, val_df)
    scoring_val = val_selection_df if val_selection_df is not None else val_df
    _, valid_df = _load_scoring_frames(train_df, scoring_val)
    results: dict[str, dict] = {}
    marginal_report: dict[str, Any] = {
        "schema_version": "1.0",
        "metric_delta_contract": "full_minus_without_i",
        "pruning_enabled": bool(
            getattr(_cfg, "RB_MARGINAL_PRUNING", False)
        ),
        "effective_rule_count_enabled": bool(
            getattr(_cfg, "RB_EFFECTIVE_RULE_COUNT_ENABLED", True)
        ),
        "directions": {},
    }

    for direction in directions:
        direction_marginal_report = _empty_marginal_report(
            direction,
            reason="not_evaluated",
        )
        pool = pools.get(direction, [])
        if not pool:
            reason = (failure_reasons or {}).get(direction, "empty_phase2_pool")
            logger.warning("RB [%s]: empty Phase 2 pool; fail closed (%s).", direction, reason)
            direction_marginal_report["reason"] = reason
            marginal_report["directions"][direction] = direction_marginal_report
            results[direction] = _write_fail_closed_strategy(
                out_dir,
                reports_dir,
                direction,
                reason,
                phase2_status={"reason": reason},
            )
            continue
        # Normalize provenance for pools produced by current or compatible
        # older writers.  The identity is based on immutable feature logic and
        # island scope; RB risk copies must carry it unchanged.
        phase2_pool_ids: set[str] = set()
        for pool_entry in pool:
            if not isinstance(pool_entry, dict):
                continue
            feature_conditions = feature_conditions_only(
                pool_entry.get("conditions", [])
            )
            pool_entry["feature_conditions"] = feature_conditions
            pool_entry["phase2_rule_id"] = str(
                pool_entry.get("phase2_rule_id")
                or phase2_rule_id(
                    pool_entry.get("conditions", []),
                    direction=direction,
                    source_symbols=pool_entry.get("source_symbols", []),
                )
            )
            phase2_pool_ids.add(pool_entry["phase2_rule_id"])
        train_engine = CPUBacktestEngine(train_like, {}, direction)
        valid_engine = CPUBacktestEngine(valid_df, {}, direction)

        # Recency rescue uses the older chronological half of the complete
        # validation holdout as a second, independent certificate.  The
        # active ``valid_df`` remains the RB selection frame; Phase 5 test data
        # is never involved in this branch.
        recency_fitness_engine: CPUBacktestEngine | None = None
        recency_candidates: list[CandidateRecord] = []
        if bool(getattr(_cfg, "RB_RECENCY_RESCUE_ENABLED", False)):
            try:
                from gpu_fuzzy_trader.data.splitter import (
                    split_validation_fitness_selection,
                )

                if "symbol" in val_df.columns:
                    recency_fitness_raw, _ = split_validation_fitness_selection(
                        val_df,
                    )
                else:
                    midpoint = max(1, len(val_df) // 2)
                    recency_fitness_raw = val_df.iloc[:midpoint].copy()
                recency_fitness = _prepare_scoring_frame(recency_fitness_raw)
                if len(recency_fitness) > 0:
                    recency_fitness_engine = CPUBacktestEngine(
                        recency_fitness, {}, direction,
                    )
            except Exception as exc:
                logger.warning(
                    "RB [%s]: recency fitness frame unavailable; skipping rescue: %s",
                    direction,
                    exc,
                )

        # Build per-fold engines for CV-fold consistency (C4)
        fold_engines: list[CPUBacktestEngine] | None = None
        if cv_folds:
            try:
                fold_engines = [
                    CPUBacktestEngine(fold.valid_df, {}, direction)
                    for fold in cv_folds
                    if not bool(getattr(fold, "is_holdout", False))
                ]
            except Exception:
                logger.warning("RB [%s]: failed to build CV-fold engines; skipping CV term.", direction)
                fold_engines = None

        # Build the chronological inner-validation tail before composing a
        # strategy. It is a feasibility gate for selection and risk tuning;
        # Phase 5 test data remains completely untouched.
        wf_splits = int(getattr(_cfg, "RB_RISK_GRID_WF_SPLITS", 1))
        use_tail = bool(getattr(_cfg, "RB_RISK_GRID_USE_TAIL_HOLDOUT", False))
        tail_frac = float(_cfg.RB_TAIL_HOLDOUT_FRACTION)
        wf_fold_engines: list[CPUBacktestEngine] | None = None
        wf_tail_engine: CPUBacktestEngine | None = None
        if wf_splits > 1 or use_tail:
            wf_fold_engines, wf_tail_engine = _make_walk_forward_fold_engines(
                scoring_val, wf_splits, tail_frac if use_tail else 0.0, direction,
            )

        candidate_pool, n_univariate_baselines = _univariate_baseline_pool(
            pool, train_like, valid_df,
        )
        if n_univariate_baselines:
            logger.info(
                "RB [%s]: added %d deterministic univariate symbol baselines.",
                direction,
                n_univariate_baselines,
            )
        candidates = _filter_good_rules(
            candidate_pool,
            train_like,
            valid_df,
            direction,
            fold_engines=fold_engines,
            recency_fitness_engine=recency_fitness_engine,
            recency_sink=recency_candidates,
        )
        # Also probe the highest-ranked ordinary candidates on the older
        # validation half.  A rule can be positive on the historical train
        # split yet still be the better *recent-regime* choice when its weak
        # validation half beats the composed baseline.  Limit this probe to
        # the bounded RB shortlist so runtime remains linear in the existing
        # candidate budget rather than adding a second full pool scan.
        if recency_fitness_engine is not None and candidates:
            seen_recency = {_rule_key(rec.rule) for rec in recency_candidates}
            probe_limit = max(
                1,
                int(getattr(_cfg, "RB_RECENCY_MAX_CANDIDATES", 40)) * 3,
            )
            for candidate in candidates[:probe_limit]:
                key = _rule_key(candidate.rule)
                if key in seen_recency:
                    continue
                try:
                    fitness_m = recency_fitness_engine.simulate_rule_set(
                        [candidate.rule],
                    )
                except Exception:
                    continue
                if not _is_recency_good(
                    candidate.train_metrics,
                    fitness_m,
                    candidate.valid_metrics,
                ):
                    continue
                recency_candidates.append(CandidateRecord(
                    rule=candidate.rule,
                    train_metrics=candidate.train_metrics,
                    valid_metrics=candidate.valid_metrics,
                    score=_recency_validation_score(
                        fitness_m, candidate.valid_metrics,
                    ),
                    mask=candidate.mask,
                    recency=True,
                    recency_fitness_metrics=fitness_m,
                ))
                seen_recency.add(key)
            recency_candidates.sort(key=lambda rec: rec.score, reverse=True)
            del recency_candidates[
                max(0, int(getattr(_cfg, "RB_RECENCY_MAX_CANDIDATES", 40))):
            ]
        if not candidates:
            logger.warning(
                "RB [%s]: no positive-good single rules; fail closed.",
                direction,
            )
            direction_marginal_report["reason"] = "no_positive_good_candidates"
            marginal_report["directions"][direction] = direction_marginal_report
            results[direction] = _write_fail_closed_strategy(
                out_dir,
                reports_dir,
                direction,
                "no_positive_good_candidates",
            )
            continue

        symbol_policy = _symbol_gate_policy(
            candidates,
            train_like,
            valid_df,
            tail_holdout_engine=wf_tail_engine,
        )
        effective_min_symbols = int(symbol_policy["effective_min_symbols"])
        concentration_max_share = float(
            symbol_policy["concentration_max_share"]
        )
        concentration_max_hhi = float(symbol_policy["concentration_max_hhi"])
        if symbol_policy["partial_specialist_coverage"]:
            logger.warning(
                "RB [%s]: allowing partial specialist coverage for %s; "
                "no positive candidate survived for %s",
                direction,
                symbol_policy["candidate_positive_symbols"],
                symbol_policy["missing_candidate_symbols"],
            )

        selected, sel_train, sel_test, sel_score, compose_history = _compose_ruleset(
            candidates,
            train_engine,
            valid_engine,
            direction,
            tail_holdout_engine=wf_tail_engine,
            min_distinct_symbols=effective_min_symbols,
            concentration_max_share=concentration_max_share,
            concentration_max_hhi=concentration_max_hhi,
        )

        # The greedy composition is the first point where a complete RB team
        # exists.  Compare that team with every exact validation leave-one-out
        # portfolio before risk tuning or profit amplification can change its
        # membership.  Report-only is the default; enabling the flag applies
        # the bounded, deterministic removal pass.
        try:
            (
                selected,
                sel_train,
                sel_test,
                direction_marginal_report,
            ) = _marginal_prune_ruleset(
                selected,
                train_engine,
                valid_engine,
                direction,
            )
        except Exception as exc:
            logger.warning(
                "RB [%s]: marginal contribution pass failed; retaining greedy "
                "selection: %s",
                direction,
                exc,
            )
            direction_marginal_report = _empty_marginal_report(
                direction,
                reason=f"evaluation_error: {type(exc).__name__}: {exc}",
            )
        marginal_report["directions"][direction] = direction_marginal_report

        opt_rules, opt_train, opt_test, opt_score, risk_history = _optimize_risk(
            selected, train_engine, valid_engine, direction,
            fold_engines=wf_fold_engines, tail_holdout_engine=wf_tail_engine,
            min_distinct_symbols=effective_min_symbols,
            concentration_max_share=concentration_max_share,
            concentration_max_hhi=concentration_max_hhi,
            immutable_exits=bool(
                getattr(_cfg, "RB_CANONICAL_PIPELINE_ACTIVE", False)
            ),
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
            min_symbols=effective_min_symbols,
            concentration_max_share=concentration_max_share,
            concentration_max_hhi=concentration_max_hhi,
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

        # Evaluate the normal final ruleset on the older validation half and
        # let a bounded recency candidate replace it only under the explicit
        # validation-only comparison certificate.  This happens before the
        # tail refresh so the tail gate is applied to whichever ruleset is
        # actually persisted.
        recency_active = False
        recency_detail: dict[str, Any] = {
            "enabled": bool(getattr(_cfg, "RB_RECENCY_RESCUE_ENABLED", False)),
            "candidates": len(recency_candidates),
            "selected": False,
        }
        normal_fitness_metrics: dict | None = None
        if recency_fitness_engine is not None:
            try:
                normal_fitness_metrics = recency_fitness_engine.simulate_rule_set(
                    [_rule_to_engine(rule) for rule in opt_rules],
                )
            except Exception as exc:
                logger.warning(
                    "RB [%s]: normal recency baseline evaluation failed: %s",
                    direction,
                    exc,
                )
        normal_certificate_ok, _normal_certificate = (
            _portfolio_selection_certificate(
                opt_test,
                min_symbols=effective_min_symbols,
                concentration_max_share=concentration_max_share,
                concentration_max_hhi=concentration_max_hhi,
            )
        )
        recency_choice, recency_detail = _select_recency_candidate(
            recency_candidates,
            opt_rules,
            normal_fitness_metrics,
            opt_test,
            normal_certificate_ok=normal_certificate_ok,
            tail_holdout_engine=wf_tail_engine,
            min_symbols=effective_min_symbols,
            concentration_max_share=concentration_max_share,
            concentration_max_hhi=concentration_max_hhi,
        )
        if recency_choice is not None:
            opt_rules = [_rule_to_engine(recency_choice.rule)]
            opt_train = recency_choice.train_metrics
            opt_test = recency_choice.valid_metrics
            opt_score = _score_metrics(
                opt_train,
                opt_test,
                min_train_trades=int(getattr(
                    _cfg, "RB_RULESET_MIN_TRAIN_TRADES",
                    getattr(_cfg, "RB_MIN_TRAIN_TRADES", 25),
                )),
                min_valid_trades=int(getattr(
                    _cfg, "RB_RULESET_MIN_VALID_TRADES",
                    getattr(_cfg, "RB_MIN_VALID_TRADES", 15),
                )),
            )
            recency_active = True
            risk_history.append({
                "pass": "recency_rescue",
                "rule_index": 1,
                "score": opt_score,
                "train_return_pct": _f(opt_train, "total_return_pct"),
                "valid_return_pct": _f(opt_test, "total_return_pct"),
                "recency": True,
            })
            if isinstance(profit_meta, dict):
                profit_meta = dict(profit_meta)
                profit_meta["recency_rescue"] = recency_detail
            logger.info(
                "RB [%s]: recency rescue selected %d-rule candidate | "
                "weak-half=%.2f%% selection=%.2f%%",
                direction,
                len(opt_rules),
                float(recency_detail.get("candidate_weak_half_score", 0.0)),
                _f(opt_test, "total_return_pct"),
            )

        # Profit amplification runs after the risk grid, so refresh the tail
        # result for the actual final ruleset before applying the hard gate.
        if wf_tail_engine is not None:
            tail_entry = dict(risk_history[-1]) if risk_history else {
                "pass": "final",
                "rule_index": -1,
            }
            try:
                _, final_tail_metrics, _ = _evaluate_ruleset(
                    train_engine,
                    wf_tail_engine,
                    opt_rules,
                )
                tail_entry.update({
                    "risk_tail_holdout_return_pct": _f(
                        final_tail_metrics, "total_return_pct",
                    ),
                    "risk_tail_holdout_pf": _f(
                        final_tail_metrics, "profit_factor",
                    ),
                    "risk_tail_holdout_dd": _f(
                        final_tail_metrics, "max_drawdown_pct",
                    ),
                })
                tail_entry.pop("risk_tail_holdout_error", None)
            except Exception as exc:
                # A requested tail check that cannot be evaluated is not a
                # pass.  Use a finite sentinel so JSON remains portable and
                # the normal tail gate fails closed.
                tail_entry.update({
                    "risk_tail_holdout_return_pct": -1.0e9,
                    "risk_tail_holdout_pf": 0.0,
                    "risk_tail_holdout_dd": 100.0,
                    "risk_tail_holdout_error": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                })
            if risk_history:
                risk_history[-1] = tail_entry
            else:
                risk_history.append(tail_entry)
        if recency_active:
            portfolio_cert_ok, portfolio_certificate = (
                _portfolio_selection_certificate(
                    opt_test,
                    min_symbols=effective_min_symbols,
                    concentration_max_share=(
                        concentration_max_share
                        if symbol_policy["partial_specialist_coverage"]
                        else float(getattr(
                            _cfg, "RB_RECENCY_MAX_SYMBOL_SHARE_ABS_PNL", 0.85,
                        ))
                    ),
                    concentration_max_hhi=(
                        concentration_max_hhi
                        if symbol_policy["partial_specialist_coverage"]
                        else float(getattr(
                            _cfg, "RB_RECENCY_MAX_SYMBOL_HHI", 0.75,
                        ))
                    ),
                )
            )
        else:
            portfolio_cert_ok, portfolio_certificate = (
                _portfolio_selection_certificate(
                    opt_test,
                    min_symbols=effective_min_symbols,
                    concentration_max_share=concentration_max_share,
                    concentration_max_hhi=concentration_max_hhi,
                )
            )
        rejected_portfolio_certificates: list[dict] = []
        for risk_entry in risk_history:
            if isinstance(risk_entry, dict):
                risk_rejections = risk_entry.get(
                    "rejected_portfolio_certificates", []
                )
                if isinstance(risk_rejections, list):
                    rejected_portfolio_certificates.extend(risk_rejections)
        if isinstance(profit_meta, dict):
            profit_rejections = profit_meta.get(
                "rejected_portfolio_certificates", []
            )
            if isinstance(profit_rejections, list):
                rejected_portfolio_certificates.extend(profit_rejections)
        if not portfolio_cert_ok:
            rejected_portfolio_certificates.append({
                "stage": "final_portfolio",
                "certificate": portfolio_certificate,
            })

        # ── Hard gate: minimum distinct symbols on final output ──────────────
        if (
            bool(getattr(_cfg, "RB_CANONICAL_PIPELINE_ACTIVE", False))
            and bool(getattr(_cfg, "RB_PHASE2_PROVENANCE_ONLY", False))
        ):
            invalid_rules = []
            for rule_index, rule in enumerate(opt_rules, start=1):
                feature_conditions = _strip_symbol_conditions(
                    list(rule.get("conditions", []))
                )
                n_conditions = len(feature_conditions)
                rule_id = str(rule.get("phase2_rule_id", ""))
                expected_features = list(rule.get("feature_conditions", []))
                invalid_reason = None
                if rule_id not in phase2_pool_ids:
                    invalid_reason = "not_phase2_candidate"
                elif expected_features and expected_features != feature_conditions:
                    invalid_reason = "feature_conditions_changed"
                elif not (
                    int(getattr(_cfg, "MIN_CONDITIONS", 1))
                    <= n_conditions
                    <= int(getattr(_cfg, "MAX_CONDITIONS", n_conditions))
                ):
                    invalid_reason = "condition_count"
                if invalid_reason is not None:
                    invalid_rules.append({
                        "rule_index": rule_index,
                        "condition_count": n_conditions,
                        "conditions": feature_conditions,
                        "phase2_rule_id": rule_id,
                        "reason": invalid_reason,
                    })
            if invalid_rules:
                logger.warning(
                    "RB [%s]: final rule contract failed (%d invalid rules); "
                    "clearing strategy",
                    direction,
                    len(invalid_rules),
                )
                strategy = _strategy(
                    direction,
                    [],
                    risk_optimized=False,
                    extra={
                        "deployment_accepted": False,
                        "fail_closed": True,
                        "reason": "phase2_condition_contract",
                        "invalid_rules": invalid_rules,
                    },
                )
                strategy_path = out_dir / f"{direction}.json"
                with strategy_path.open("w", encoding="utf-8") as fh:
                    json.dump(strategy, fh, indent=2)
                _write_clean_evaluator(
                    strategy,
                    out_dir / "evaluator_clean" /
                    f"{direction}_evaluator_clean.json",
                )
                report = {
                    "direction": direction,
                    "rb_score": 0.0,
                    "train_metrics": {},
                    "valid_metrics": {},
                    "selected_rules": 0,
                    "n_positive_single_rules": len(candidates),
                    "fail_closed": True,
                    "fail_closed_reason": "phase2_condition_contract",
                    "invalid_rules": invalid_rules,
                    "marginal_contribution": direction_marginal_report,
                }
                with (reports_dir / f"rb_governor_{direction}_report.json").open(
                    "w", encoding="utf-8",
                ) as fh:
                    json.dump(report, fh, indent=2, default=str)
                results[direction] = strategy
                continue

        if bool(getattr(_cfg, "RB_REQUIRE_SYMBOL_FILTERS", False)):
            active_symbol_count = len(_available_symbols(train_like, valid_df))
            if (
                bool(getattr(_cfg, "DEBUG_SYMBOL_SCOPE_ENABLED", False))
                and not symbol_policy["partial_specialist_coverage"]
            ):
                min_distinct = int(
                    _cfg.effective_rb_min_distinct_symbols(active_symbol_count)
                )
            else:
                min_distinct = effective_min_symbols
            if min_distinct > 0:
                n_symbols = len(_symbols_in_rules(opt_rules))
                if n_symbols < min_distinct:
                    logger.warning(
                        "RB [%s]: insufficient distinct symbols in final ruleset "
                        "(%d < %d required); failing closed.",
                        direction, n_symbols, min_distinct,
                    )
                    opt_rules = []
                    opt_train = {}
                    opt_test = {}
                    opt_score = 0.0
                    strategy = _strategy(
                        direction, [],
                        risk_optimized=False,
                        extra={
                            "deployment_accepted": False,
                            "fail_closed": True,
                            "reason": "insufficient_distinct_symbols",
                            "n_symbols": n_symbols,
                            "required": min_distinct,
                            "symbol_coverage_policy": symbol_policy,
                            "symbol_contribution_certificate": portfolio_certificate,
                            "marginal_contribution": direction_marginal_report,
                        },
                    )
                    strategy_path = out_dir / f"{direction}.json"
                    with strategy_path.open("w", encoding="utf-8") as fh:
                        json.dump(strategy, fh, indent=2)
                    _write_clean_evaluator(strategy, out_dir / "evaluator_clean" / f"{direction}_evaluator_clean.json")
                    report = {
                        "direction": direction,
                        "rb_score": 0.0,
                        "train_metrics": {},
                        "valid_metrics": {},
                        "train_minus_valid_return_pct": 0.0,
                        "train_valid_ratio": 0.0,
                        "n_positive_single_rules": len(candidates),
                        "selected_rules": 0,
                        "compose_history": compose_history,
                        "risk_history": risk_history,
                        "profit_amplifier": profit_meta,
                        "rejected_portfolio_certificates": rejected_portfolio_certificates,
                        "symbol_contribution_certificate": portfolio_certificate,
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
                        "fail_closed": True,
                        "fail_closed_reason": "insufficient_distinct_symbols",
                        "n_symbols": n_symbols,
                        "required_symbols": min_distinct,
                        "symbol_coverage_policy": symbol_policy,
                        "marginal_contribution": direction_marginal_report,
                    }
                    with (reports_dir / f"rb_governor_{direction}_report.json").open("w", encoding="utf-8") as fh:
                        json.dump(report, fh, indent=2, default=str)
                    logger.info(
                        "RB [%s]: fail-closed empty strategy written "
                        "(insufficient_distinct_symbols: %d < %d).",
                        direction, n_symbols, min_distinct,
                    )
                    results[direction] = strategy
                    continue
        # ── End hard gate ────────────────────────────────────────────────────

        val_ret = _f(opt_test, "total_return_pct")
        val_pf = _f(opt_test, "profit_factor")
        ret_gate = float(_cfg.PHASE5_VALIDATION_RETURN_GATE_PCT)
        pf_gate = float(_cfg.PHASE5_VALIDATION_PROFIT_FACTOR_GATE)
        if recency_active:
            sym_ok, sym_gate = _passes_symbol_concentration_gate(
                opt_test,
                max_share=(
                    concentration_max_share
                    if symbol_policy["partial_specialist_coverage"]
                    else float(getattr(
                        _cfg, "RB_RECENCY_MAX_SYMBOL_SHARE_ABS_PNL", 0.85,
                    ))
                ),
                max_hhi=(
                    concentration_max_hhi
                    if symbol_policy["partial_specialist_coverage"]
                    else float(getattr(_cfg, "RB_RECENCY_MAX_SYMBOL_HHI", 0.75))
                ),
            )
        else:
            sym_ok, sym_gate = _passes_symbol_concentration_gate(
                opt_test,
                max_share=concentration_max_share,
                max_hhi=concentration_max_hhi,
            )
        tail_ok, tail_gate = _passes_tail_holdout_gate(risk_history)
        cost_stress_ok, cost_stress = _cost_stress_gate(
            train_engine,
            valid_engine,
            opt_rules,
        )
        monthly_ok, monthly_certificate = _monthly_selection_certificate(
            valid_engine,
            opt_rules,
            direction,
        )
        deployable = (
            val_ret >= (ret_gate - 1e-9)
            and val_pf >= (pf_gate - 1e-9)
            and portfolio_cert_ok
            and sym_ok
            and tail_ok
            and cost_stress_ok
            and monthly_ok
        )
        if not portfolio_cert_ok:
            logger.warning(
                "RB [%s]: symbol-contribution certificate failed (%s)",
                direction,
                portfolio_certificate.get("reasons", []),
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
        if not cost_stress_ok:
            logger.warning(
                "RB [%s]: cost-stress gate failed (%s)",
                direction,
                cost_stress,
            )
        if not monthly_ok:
            logger.warning(
                "RB [%s]: monthly selection certificate failed (%s)",
                direction,
                monthly_certificate,
            )

        # Hard fail-closed: concentration / tail reject clears ruleset (do not
        # persist rejected teams for Phase 5). Return/PF-only soft path below
        # still saves rules with deployment_accepted=False.
        if (
            not portfolio_cert_ok
            or not sym_ok
            or not tail_ok
            or not cost_stress_ok
            or not monthly_ok
        ):
            reasons: list[str] = []
            if not portfolio_cert_ok:
                certificate_reasons = portfolio_certificate.get("reasons", [])
                if "symbol_contribution" in certificate_reasons:
                    reasons.append("symbol_contribution")
                elif "symbol_concentration" in certificate_reasons and sym_ok:
                    reasons.append("symbol_concentration")
                elif not certificate_reasons:
                    reasons.append("symbol_contribution")
            if not sym_ok:
                reasons.append("symbol_concentration")
            if not tail_ok:
                reasons.append("tail_holdout")
            if not cost_stress_ok:
                reasons.append("cost_stress")
            if not monthly_ok:
                reasons.append("monthly_stability")
            fail_reason = "+".join(reasons)
            strategy = _strategy(
                direction,
                [],
                risk_optimized=False,
                extra={
                    "deployment_accepted": False,
                    "fail_closed": True,
                    "reason": fail_reason,
                    "validation_gate": {
                        "return_pct": val_ret,
                        "profit_factor": val_pf,
                        "required_return_pct": ret_gate,
                        "required_profit_factor": pf_gate,
                    },
                    "symbol_concentration_gate": sym_gate,
                    "tail_holdout_gate": tail_gate,
                    "cost_stress_gate": cost_stress,
                    "monthly_certificate": monthly_certificate,
                    "symbol_coverage_policy": symbol_policy,
                    "symbol_contribution_certificate": portfolio_certificate,
                    "rb_score": 0.0,
                    "rb_profit_amp_objective": profit_objective,
                    "rb_profit_amp_accepted": bool(profit_meta.get("accepted", False)),
                    "recency_rescue": recency_detail,
                },
            )
            strategy_path = out_dir / f"{direction}.json"
            with strategy_path.open("w", encoding="utf-8") as fh:
                json.dump(strategy, fh, indent=2)
            _write_clean_evaluator(
                strategy,
                out_dir / "evaluator_clean" /
                f"{direction}_evaluator_clean.json",
            )
            report = {
                "direction": direction,
                "rb_score": 0.0,
                "train_metrics": {},
                "valid_metrics": {},
                "train_minus_valid_return_pct": 0.0,
                "train_valid_ratio": 0.0,
                "n_positive_single_rules": len(candidates),
                "selected_rules": 0,
                "compose_history": compose_history,
                "risk_history": risk_history,
                "profit_amplifier": profit_meta,
                "rejected_portfolio_certificates": rejected_portfolio_certificates,
                "symbol_contribution_certificate": portfolio_certificate,
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
                "fail_closed": True,
                "fail_closed_reason": fail_reason,
                "symbol_concentration_gate": sym_gate,
                "tail_holdout_gate": tail_gate,
                "cost_stress_gate": cost_stress,
                "monthly_certificate": monthly_certificate,
                "symbol_coverage_policy": symbol_policy,
                "validation_gate": {
                    "return_pct": val_ret,
                    "profit_factor": val_pf,
                    "required_return_pct": ret_gate,
                    "required_profit_factor": pf_gate,
                },
            }
            with (reports_dir / f"rb_governor_{direction}_report.json").open(
                "w", encoding="utf-8",
            ) as fh:
                json.dump(report, fh, indent=2, default=str)
            logger.info(
                "RB [%s]: fail-closed empty strategy written (%s).",
                direction,
                fail_reason,
            )
            results[direction] = strategy
            continue

        strategy = _strategy(
            direction,
            opt_rules,
            risk_optimized=bool(deployable),
            extra={
                "deployment_accepted": bool(deployable),
                "deployment_reason": None if deployable else "validation_gate",
                "validation_gate": {
                    "return_pct": val_ret,
                    "profit_factor": val_pf,
                    "required_return_pct": ret_gate,
                    "required_profit_factor": pf_gate,
                },
                "symbol_concentration_gate": sym_gate,
                "tail_holdout_gate": tail_gate,
                "symbol_coverage_policy": symbol_policy,
                "symbol_contribution_certificate": portfolio_certificate,
                "marginal_contribution": direction_marginal_report,
                "recency_rescue": recency_detail,
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
            "deployment_accepted": bool(deployable),
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
            "recency_rescue": recency_detail,
            "rejected_portfolio_certificates": rejected_portfolio_certificates,
            "symbol_coverage_policy": symbol_policy,
            "symbol_contribution_certificate": portfolio_certificate,
            "marginal_contribution": direction_marginal_report,
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
    # A half-window can be too sparse for a balanced two-symbol certificate
    # even when the complete, still-unseen validation holdout contains a
    # stable team.  Retry only directions that did not produce an accepted
    # deployment, and only when an actual selection subset was supplied.  The
    # recursive call is explicitly disabled so recovery is bounded to one
    # additional validation-only attempt and cannot loop.
    recovery_enabled = bool(
        getattr(_cfg, "RB_FULL_VALIDATION_RECOVERY_ENABLED", False)
    )
    selection_is_subset = (
        val_selection_df is not None
        and len(val_selection_df) < len(val_df)
    )
    recovery_directions = [
        direction
        for direction in directions
        if pools.get(direction)
        if not (
            isinstance(results.get(direction), dict)
            and results[direction].get("rules_set")
            and results[direction].get("deployment_accepted") is True
        )
    ]
    if (
        _allow_full_validation_recovery
        and recovery_enabled
        and selection_is_subset
        and recovery_directions
    ):
        logger.info(
            "RB: retrying fail-closed directions %s on the complete "
            "validation holdout (validation-only recovery).",
            recovery_directions,
        )
        recovered = run_rb_governor_pipeline(
            train_df,
            val_df,
            {direction: pools.get(direction, []) for direction in recovery_directions},
            recovery_directions,
            output_dir=out_dir,
            cv_folds=cv_folds,
            val_selection_df=val_df,
            failure_reasons=failure_reasons,
            _allow_full_validation_recovery=False,
        )
        for direction in recovery_directions:
            candidate = recovered.get(direction)
            if not isinstance(candidate, dict):
                continue
            accepted = bool(candidate.get("rules_set")) and (
                candidate.get("deployment_accepted") is True
            )
            if accepted:
                candidate["validation_recovery"] = {
                    "used": True,
                    "selection_frame": "complete_validation_holdout",
                }
                strategy_path = out_dir / f"{direction}.json"
                try:
                    with strategy_path.open("w", encoding="utf-8") as fh:
                        json.dump(candidate, fh, indent=2)
                except OSError as exc:
                    logger.warning(
                        "RB [%s]: could not annotate recovery strategy: %s",
                        direction,
                        exc,
                    )
                results[direction] = candidate
                logger.info(
                    "RB [%s]: validation-only recovery accepted a %d-rule "
                    "strategy.",
                    direction,
                    len(candidate.get("rules_set", [])),
                )
    _write_marginal_report(reports_dir, marginal_report)
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
