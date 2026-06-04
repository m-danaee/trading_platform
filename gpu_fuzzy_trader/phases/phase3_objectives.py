"""
phase3_objectives.py — Unified Phase 3 penalty and objective computation.
"""

from __future__ import annotations

import numpy as np

from gpu_fuzzy_trader import config as _cfg


def conditions_key(conditions: list[str]) -> frozenset[str]:
    return frozenset(conditions)


def has_duplicate_rules(rule_set: list[dict]) -> bool:
    seen: set[frozenset] = set()
    for rule in rule_set:
        key = conditions_key(rule["conditions"])
        if key in seen:
            return True
        seen.add(key)
    return False


def count_symbols_with_trades(metrics: dict) -> int:
    per_sym = metrics.get("per_symbol_metrics", {})
    return sum(1 for v in per_sym.values() if v.get("trade_count", 0) > 0)


def symbols_with_trades(metrics: dict) -> set:
    per_sym = metrics.get("per_symbol_metrics", {})
    return {
        s for s, v in per_sym.items()
        if v.get("trade_count", 0) > 0
    }


def symbol_consistency_penalty(train_metrics: dict, val_metrics: dict) -> float:
    train_syms = symbols_with_trades(train_metrics)
    val_syms = symbols_with_trades(val_metrics)
    if not train_syms or not val_syms:
        return 0.0
    overlap = len(train_syms & val_syms) / len(train_syms | val_syms)
    return (1.0 - overlap) * _cfg.PHASE3_SYMBOL_CONSISTENCY_WEIGHT


def per_symbol_pnl_vector(metrics: dict, symbols: list) -> np.ndarray:
    per = metrics.get("per_symbol_metrics", {}) or {}
    out = np.zeros(len(symbols), dtype=np.float64)
    for i, sym in enumerate(symbols):
        v = per.get(sym, per.get(str(sym), {}))
        out[i] = float(v.get("net_pnl", 0.0)) if isinstance(v, dict) else 0.0
    return out


def train_val_gap_penalty(train_metrics: dict, val_metrics: dict) -> float:
    """Penalise large train/val return gaps (overfitting in either direction)."""
    train_ret = float(train_metrics.get("total_return_pct", 0.0))
    val_ret = float(val_metrics.get("total_return_pct", 0.0))
    penalty = 0.0
    train_over = train_ret - val_ret
    if train_over > _cfg.PHASE3_TRAIN_VAL_GAP_MAX_PCT:
        penalty += (train_over - _cfg.PHASE3_TRAIN_VAL_GAP_MAX_PCT) * (
            _cfg.PHASE3_GAP_PENALTY_WEIGHT
        )
    val_over = val_ret - train_ret
    if val_over > _cfg.PHASE3_VAL_TRAIN_GAP_MAX_PCT:
        penalty += (val_over - _cfg.PHASE3_VAL_TRAIN_GAP_MAX_PCT) * (
            _cfg.PHASE3_GAP_PENALTY_WEIGHT
        )
    return penalty


def train_val_corr_penalty(train_metrics: dict, val_metrics: dict) -> float:
    train_per = train_metrics.get("per_symbol_metrics", {}) or {}
    val_per = val_metrics.get("per_symbol_metrics", {}) or {}
    common = set(train_per.keys()) & set(val_per.keys())
    if len(common) < 3:
        return 0.0
    symbols = sorted(common, key=lambda s: str(s))
    a = per_symbol_pnl_vector(train_metrics, symbols)
    b = per_symbol_pnl_vector(val_metrics, symbols)
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    corr = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(corr):
        return 0.0
    return (1.0 - corr) * 0.5 * _cfg.PHASE3_TRAIN_VAL_CORR_WEIGHT


def _symbol_robustness_penalty(metrics: dict) -> float:
    """Penalty for poor cross-symbol PnL dispersion."""
    per_sym = metrics.get("per_symbol_metrics", {}) or {}
    if not per_sym:
        return 0.0
    pnl_vec = []
    profitable = 0
    for v in per_sym.values():
        if not isinstance(v, dict):
            continue
        pnl = float(v.get("net_pnl", 0.0))
        pnl_pct = (pnl / _cfg.INITIAL_CAPITAL) * 100.0
        pnl_vec.append(pnl_pct)
        if pnl > 0.0:
            profitable += 1
    if not pnl_vec:
        return 0.0
    penalty = 0.0
    med = float(np.median(np.asarray(pnl_vec, dtype=np.float64)))
    if med < _cfg.PHASE3_SYMBOL_MEDIAN_RETURN_FLOOR_PCT:
        penalty += abs(_cfg.PHASE3_SYMBOL_MEDIAN_RETURN_FLOOR_PCT - med)
    shortfall = max(0, _cfg.PHASE3_MIN_PROFITABLE_SYMBOLS - profitable)
    penalty += float(shortfall) * 3.0
    return penalty


def min_per_symbol_trades_from_metrics(metrics: dict) -> int:
    """Minimum per-symbol trade_count for a single-rule simulation."""
    per = metrics.get("per_symbol_metrics", {}) or {}
    if not per:
        return 0
    worst = float("inf")
    for v in per.values():
        tc = int(v.get("trade_count", 0)) if isinstance(v, dict) else 0
        if tc < worst:
            worst = tc
    return 0 if worst == float("inf") else int(worst)


def per_rule_min_symbol_trades_cached(
    rule_set: list[dict],
    per_rule_min_val_trades: dict[frozenset, int] | None,
) -> int:
    """Lookup worst per-symbol min trade count from precomputed val cache."""
    if not per_rule_min_val_trades:
        return 0
    worst = float("inf")
    for rule in rule_set:
        key = conditions_key(rule.get("conditions", []))
        if key not in per_rule_min_val_trades:
            return 0
        tc = per_rule_min_val_trades[key]
        if tc < worst:
            worst = tc
    return 0 if worst == float("inf") else int(worst)


def _incremental_trade_penalty(
    rule_set: list[dict],
    val_masks_by_key: dict[frozenset, np.ndarray] | None,
    n_rows_val: int,
    min_incremental_trades: int,
) -> float:
    if not val_masks_by_key or len(rule_set) <= 1 or n_rows_val <= 0:
        return 0.0

    assigned = np.zeros(n_rows_val, dtype=bool)
    penalty = 0.0
    for idx, rule in enumerate(rule_set):
        key = conditions_key(rule.get("conditions", []))
        mask = val_masks_by_key.get(key)
        if mask is None:
            continue
        incremental = int(np.count_nonzero(mask & (~assigned)))
        assigned |= mask
        if idx > 0 and incremental < min_incremental_trades:
            shortfall = min_incremental_trades - incremental
            ratio = shortfall / max(min_incremental_trades, 1)
            penalty += _cfg.PHASE3_INCREMENTAL_GATE_PENALTY * ratio
    return penalty


def _jaccard_similarity_penalty(
    rule_set: list[dict],
    val_masks_by_key: dict[frozenset, np.ndarray] | None,
) -> float:
    if not val_masks_by_key or len(rule_set) <= 1:
        return 0.0

    masks: list[np.ndarray] = []
    for rule in rule_set:
        key = conditions_key(rule.get("conditions", []))
        mask = val_masks_by_key.get(key)
        if mask is None:
            continue
        masks.append(mask)
    if len(masks) <= 1:
        return 0.0

    penalty = 0.0
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            union = int(np.count_nonzero(masks[i] | masks[j]))
            if union <= 0:
                continue
            inter = int(np.count_nonzero(masks[i] & masks[j]))
            jaccard = inter / union
            if jaccard > _cfg.PHASE3_JACCARD_SIMILARITY_GATE:
                penalty += (
                    (jaccard - _cfg.PHASE3_JACCARD_SIMILARITY_GATE)
                    * _cfg.PHASE3_JACCARD_PENALTY_WEIGHT
                )
    return penalty


def effective_symbol_coverage_target(pool_size: int | None) -> int:
    """Scale coverage requirement when the Phase 2 pool is small."""
    base = int(_cfg.PHASE3_MIN_SYMBOL_COVERAGE)
    if pool_size is None or pool_size >= int(_cfg.PHASE3_SMALL_POOL_THRESHOLD):
        return base
    return max(3, min(base, pool_size))


def compute_phase3_objectives(
    train_metrics: dict,
    val_metrics: dict,
    rule_set: list[dict],
    *,
    per_rule_min_val_trades: dict[frozenset, int] | None = None,
    val_masks_by_key: dict[frozenset, np.ndarray] | None = None,
    n_rows_val: int = 0,
    pool_size: int | None = None,
) -> np.ndarray:
    """
    Compute minimised objectives [f1, f2, f3] with all penalties applied.

    When ``PHASE3_USE_TRAIN_TARGET`` is True, objectives use train metrics;
    validation drives gate penalties only.
    """
    dup_penalty = 50.0 if has_duplicate_rules(rule_set) else 0.0

    train_sortino = float(train_metrics.get(
        "sortino_ratio", train_metrics.get("total_return_pct", 0.0)))
    train_dd = float(train_metrics.get("max_drawdown_pct", 100.0))
    train_wr = float(train_metrics.get("win_rate", 0.0))
    train_trades = int(train_metrics.get("executed_trades", 0))

    val_sortino = float(val_metrics.get(
        "sortino_ratio", val_metrics.get("total_return_pct", 0.0)))
    val_dd = float(val_metrics.get("max_drawdown_pct", 100.0))
    val_wr = float(val_metrics.get("win_rate", 0.0))
    val_trades = int(val_metrics.get("executed_trades", 0))
    val_ret = float(val_metrics.get("total_return_pct", 0.0))
    val_pf = float(val_metrics.get("profit_factor", 0.0))
    train_ret = float(train_metrics.get("total_return_pct", 0.0))
    train_pf = float(train_metrics.get("profit_factor", 0.0))

    zero_penalty = 100.0 if (train_trades == 0 or val_trades == 0) else 0.0

    val_symbols_with_trades = count_symbols_with_trades(val_metrics)
    coverage_target = effective_symbol_coverage_target(pool_size)
    coverage_penalty = 0.0
    if val_symbols_with_trades < coverage_target:
        coverage_penalty = (
            (coverage_target - val_symbols_with_trades) * 5.0
        )

    symbol_consistency_penalty_val = symbol_consistency_penalty(
        train_metrics, val_metrics)
    corr_penalty = train_val_corr_penalty(train_metrics, val_metrics)
    incremental_penalty = _incremental_trade_penalty(
        rule_set,
        val_masks_by_key,
        n_rows_val,
        _cfg.PHASE3_MIN_INCREMENTAL_TRADES,
    )
    jaccard_penalty = _jaccard_similarity_penalty(rule_set, val_masks_by_key)

    gate_penalty = 0.0
    if train_sortino > 0.0:
        min_val = _cfg.PHASE3_VAL_SORTINO_RATIO_GATE * train_sortino
        if val_sortino < min_val:
            gate_penalty += _cfg.PHASE3_VAL_GATE_PENALTY

    max_val_dd = _cfg.PHASE3_VAL_DRAWDOWN_RATIO_GATE * max(train_dd, 1.0)
    if val_dd > max_val_dd:
        gate_penalty += _cfg.PHASE3_VAL_GATE_PENALTY

    if per_rule_min_val_trades is not None:
        min_per_rule = per_rule_min_symbol_trades_cached(
            rule_set, per_rule_min_val_trades)
        if min_per_rule < _cfg.PHASE3_PER_RULE_MIN_VAL_TRADES_PER_SYMBOL:
            gate_penalty += _cfg.PHASE3_VAL_GATE_PENALTY
    if val_ret < _cfg.PHASE3_VAL_RETURN_FLOOR_PCT:
        gate_penalty += _cfg.PHASE3_VAL_GATE_PENALTY
    if val_pf < _cfg.PHASE3_VAL_PROFIT_FACTOR_FLOOR:
        gate_penalty += _cfg.PHASE3_VAL_GATE_PENALTY
    if train_ret < _cfg.PHASE3_TRAIN_RETURN_FLOOR_PCT:
        gate_penalty += _cfg.PHASE3_VAL_GATE_PENALTY
    if train_pf < _cfg.PHASE3_TRAIN_PROFIT_FACTOR_FLOOR:
        gate_penalty += _cfg.PHASE3_VAL_GATE_PENALTY

    total_penalty = (
        zero_penalty + coverage_penalty + dup_penalty
        + symbol_consistency_penalty_val + corr_penalty + gate_penalty
        + incremental_penalty + jaccard_penalty
        + train_val_gap_penalty(train_metrics, val_metrics)
        + _symbol_robustness_penalty(train_metrics)
        + _symbol_robustness_penalty(val_metrics)
    )

    if _cfg.PHASE3_USE_TRAIN_TARGET:
        return np.array(
            [-train_sortino + total_penalty,
             train_dd + total_penalty,
             -train_wr + total_penalty],
            dtype=np.float64,
        )

    eff_sortino = min(train_sortino, val_sortino)
    eff_dd = max(train_dd, val_dd)
    eff_wr = min(train_wr, val_wr)
    return np.array(
        [-eff_sortino + total_penalty,
         eff_dd + total_penalty,
         -eff_wr + total_penalty],
        dtype=np.float64,
    )
