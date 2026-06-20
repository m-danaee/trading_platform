
from __future__ import annotations

import json
import logging
import os
import random
import time
from typing import Optional

import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.backtest.cpu_engine import CPUBacktestEngine, _build_rule_signal_mask
from gpu_fuzzy_trader.validation.monthly_windows import evaluate_rule_set_monthly, monthly_penalty
from gpu_fuzzy_trader.scoring import return_to_drawdown, robust_ratio_score
from gpu_fuzzy_trader.log_progress import maybe_log_generation
from gpu_fuzzy_trader.phases.phase3_greedy import greedy_rule_set_search
from gpu_fuzzy_trader.reporting.reporter import Reporter
from gpu_fuzzy_trader.rules.repair import repair_rule_set, repair_rule
from gpu_fuzzy_trader.rules.clustering import smart_population_from_pool

logger = logging.getLogger(__name__)


_OUTPUT_PATHS = {
    "long": os.path.join(_cfg.OUTPUTS_DIR, "long.json"),
    "short": os.path.join(_cfg.OUTPUTS_DIR, "short.json"),
}



def _validate_rule_set_schema(data: object, path: str) -> None:
    """
    Validate the structure of a loaded rule set JSON.

    Raises ValueError if the schema is invalid.
    """
    if not isinstance(data, dict):
        raise ValueError(
            f"Rule set must be a JSON object, got {type(data).__name__}: {path}"
        )
    required_top = {"direction", "rules_set"}
    missing = required_top - set(data.keys())
    if missing:
        raise ValueError(
            f"Rule set missing top-level keys {missing}: {path}"
        )
    if data["direction"] not in ("long", "short"):
        raise ValueError(
            f"Rule set 'direction' must be 'long' or 'short': {path}"
        )
    rules_set = data["rules_set"]
    if not isinstance(rules_set, list):
        raise ValueError(
            f"Rule set 'rules_set' must be a list: {path}"
        )
    if not (2 <= len(rules_set) <= 5):
        raise ValueError(
            f"Rule set 'rules_set' must have 2–5 rules, got {len(rules_set)}: {path}"
        )
    for i, rule in enumerate(rules_set):
        if not isinstance(rule, dict):
            raise ValueError(
                f"Rule set entry {i} must be a dict: {path}"
            )
        required_rule_keys = {"tp", "sl", "capital_pct", "conditions"}
        missing_rule = required_rule_keys - set(rule.keys())
        if missing_rule:
            raise ValueError(
                f"Rule set entry {i} missing keys {missing_rule}: {path}"
            )
        if not isinstance(rule["conditions"], list) or len(rule["conditions"]) == 0:
            raise ValueError(
                f"Rule set entry {i} 'conditions' must be a non-empty list: {path}"
            )
    if "risk_optimized" in data and not isinstance(data["risk_optimized"], bool):
        raise ValueError(
            f"Rule set 'risk_optimized' must be a bool if present: {path}"
        )



def _conditions_key(conditions: list[str]) -> frozenset[str]:
    """Order-independent condition set key for duplicate detection."""
    return frozenset(conditions)


def _has_duplicate_rules(rule_set: list[dict]) -> bool:
    """Return True if any two rules have identical condition sets (order-independent)."""
    seen: set[frozenset] = set()
    for rule in rule_set:
        key = _conditions_key(rule["conditions"])
        if key in seen:
            return True
        seen.add(key)
    return False


def _count_symbols_with_trades(metrics: dict) -> int:
    """Count how many symbols have at least one trade."""
    per_sym = metrics.get("per_symbol_metrics", {})
    return sum(1 for v in per_sym.values() if v.get("trade_count", 0) > 0)


def _symbols_with_trades(metrics: dict) -> set:
    """Return symbol IDs with at least one trade."""
    per_sym = metrics.get("per_symbol_metrics", {})
    return {
        s for s, v in per_sym.items()
        if v.get("trade_count", 0) > 0
    }



def _metric_gap_penalty(train_metrics: dict, val_metrics: dict) -> float:
    """Penalise train/validation gaps across return, PF, DD and win rate."""
    if not train_metrics or not val_metrics:
        return 10.0
    train_ret = float(train_metrics.get("total_return_pct", 0.0))
    val_ret = float(val_metrics.get("total_return_pct", 0.0))
    train_pf = float(train_metrics.get("profit_factor", 0.0))
    val_pf = float(val_metrics.get("profit_factor", 0.0))
    train_dd = float(train_metrics.get("max_drawdown_pct", 0.0))
    val_dd = float(val_metrics.get("max_drawdown_pct", 0.0))
    ret_gap = max(0.0, train_ret - val_ret)
    pf_gap = max(0.0, train_pf - val_pf)
    dd_gap = max(0.0, val_dd - train_dd)
    return float(
        ret_gap * float(getattr(_cfg, "PHASE3_GAP_RETURN_WEIGHT", 0.20))
        + pf_gap * float(getattr(_cfg, "PHASE3_GAP_PF_WEIGHT", 2.0))
        + dd_gap * float(getattr(_cfg, "PHASE3_GAP_DD_WEIGHT", 0.15))
    )


def _signal_overlap_penalty(rule_set: list[dict], val_engine) -> float:
    """Penalise rule sets whose rules fire on the same symbol/time rows."""
    if not getattr(_cfg, "PHASE3_SIGNAL_OVERLAP_ENABLED", False) or len(rule_set) < 2:
        return 0.0
    df = getattr(val_engine, "df", None)
    if df is None or len(df) == 0:
        return 0.0
    masks: list[np.ndarray] = []
    for i, rule in enumerate(rule_set, start=1):
        try:
            masks.append(_build_rule_signal_mask(df, list(rule.get("conditions", []))))
        except Exception as exc:
            logger.debug("signal mask failed for overlap rule %d: %s", i, exc)
            return 5.0
    overlaps: list[float] = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            union = int(np.sum(masks[i] | masks[j]))
            if union <= 0:
                continue
            inter = int(np.sum(masks[i] & masks[j]))
            overlaps.append(inter / union)
    if not overlaps:
        return 0.0
    max_overlap = max(overlaps)
    threshold = float(getattr(_cfg, "PHASE3_MAX_PAIR_OVERLAP", 0.35))
    if max_overlap <= threshold:
        return 0.0
    return float((max_overlap - threshold) * float(getattr(_cfg, "PHASE3_OVERLAP_WEIGHT", 18.0)))


def _per_symbol_survival_penalty(metrics: dict) -> float:
    """Penalise rule sets that rely on too few symbols or one dominant symbol."""
    if not getattr(_cfg, "PHASE3_PER_SYMBOL_SURVIVAL_ENABLED", False):
        return 0.0
    per_sym = metrics.get("per_symbol_metrics", {}) if isinstance(metrics, dict) else {}
    if not per_sym:
        return float(getattr(_cfg, "PHASE3_PER_SYMBOL_WEIGHT", 8.0))
    profitable = 0
    positive_pnl: list[float] = []
    total_abs = 0.0
    for v in per_sym.values():
        trades = int(v.get("trade_count", 0))
        pnl = float(v.get("net_pnl", 0.0))
        pf = float(v.get("profit_factor", 0.0))
        if trades > 0 and (pnl > 0 or pf > 1.0):
            profitable += 1
            positive_pnl.append(max(0.0, pnl))
        total_abs += abs(pnl)
    min_profitable = int(getattr(_cfg, "PHASE3_MIN_PROFITABLE_SYMBOLS", 6))
    penalty = max(0, min_profitable - profitable) * float(getattr(_cfg, "PHASE3_PER_SYMBOL_WEIGHT", 8.0))
    if total_abs > 0 and positive_pnl:
        max_share = max(positive_pnl) / total_abs
        limit = float(getattr(_cfg, "PHASE3_MAX_SINGLE_SYMBOL_PNL_SHARE", 0.45))
        if max_share > limit:
            penalty += (max_share - limit) * float(getattr(_cfg, "PHASE3_PER_SYMBOL_WEIGHT", 8.0))
    return float(penalty)


def _purged_cv_rule_set_penalty(rule_set: list[dict], val_engine) -> float:
    """Worst-fold penalty for robust rolling validation engines."""
    if not getattr(_cfg, "PHASE3_PURGED_CV_ENABLED", False):
        return 0.0
    fold_engines = getattr(val_engine, "_purged_fold_engines", [])
    if not fold_engines:
        return 0.0
    try:
        from gpu_fuzzy_trader.validation.rolling_cv import evaluate_rule_set_on_fold_engines
        summary = evaluate_rule_set_on_fold_engines(rule_set, fold_engines)
    except Exception as exc:
        logger.debug("purged cv penalty failed: %s", exc)
        return 10.0
    penalty = 0.0
    if summary.worst_return_pct < float(_cfg.PHASE3_WORST_RETURN_FLOOR):
        penalty += (float(_cfg.PHASE3_WORST_RETURN_FLOOR) - summary.worst_return_pct)
    if summary.worst_profit_factor < float(_cfg.PHASE3_WORST_PF_FLOOR):
        penalty += (float(_cfg.PHASE3_WORST_PF_FLOOR) - summary.worst_profit_factor) * 8.0
    if summary.worst_drawdown_pct > float(_cfg.PHASE3_WORST_DD_CEIL):
        penalty += (summary.worst_drawdown_pct - float(_cfg.PHASE3_WORST_DD_CEIL)) * 0.4
    if summary.min_trades < int(_cfg.PHASE3_MIN_FOLD_TRADES):
        penalty += (int(_cfg.PHASE3_MIN_FOLD_TRADES) - summary.min_trades) * 0.15
    return float(penalty * float(getattr(_cfg, "PHASE3_CV_PENALTY_WEIGHT", 1.0)))

def _symbol_consistency_penalty(train_metrics: dict, val_metrics: dict) -> float:
    """Penalise rule sets that trade different symbols on train vs validation."""
    train_syms = _symbols_with_trades(train_metrics)
    val_syms = _symbols_with_trades(val_metrics)
    if not train_syms or not val_syms:
        return 0.0
    overlap = len(train_syms & val_syms) / len(train_syms | val_syms)
    return (1.0 - overlap) * _cfg.PHASE3_SYMBOL_CONSISTENCY_WEIGHT


def _build_hidden_like_window_engines(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    direction: str,
) -> list[CPUBacktestEngine]:
    """Build a small set of rolling 3-4 month CPU engines for Phase 3 penalties."""
    if not getattr(_cfg, "PHASE3_HIDDEN_VALIDATION_ENABLED", False):
        return []
    if "datetime" not in train_df.columns or "datetime" not in val_df.columns:
        return []

    combined = pd.concat([train_df, val_df], ignore_index=True)
    combined = combined.copy()
    combined["datetime"] = pd.to_datetime(combined["datetime"], errors="coerce")
    combined = combined.dropna(subset=["datetime"]).sort_values(["datetime", "symbol"]).reset_index(drop=True)
    if combined.empty:
        return []

    min_dt = combined["datetime"].min().normalize()
    max_dt = combined["datetime"].max()
    window = pd.Timedelta(days=int(_cfg.PHASE3_HIDDEN_WINDOW_DAYS))
    stride = pd.Timedelta(days=int(_cfg.PHASE3_HIDDEN_STRIDE_DAYS))
    max_windows = int(_cfg.PHASE3_HIDDEN_MAX_WINDOWS)
    min_rows = int(_cfg.PHASE3_HIDDEN_MIN_ROWS)

    candidates: list[tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]] = []
    start = min_dt
    while start + window <= max_dt and len(candidates) < max_windows * 3:
        end = start + window
        part = combined[(combined["datetime"] >= start) & (combined["datetime"] <= end)].copy()
        if len(part) >= min_rows and part.get("symbol", pd.Series([0])).nunique() >= 2:
            candidates.append((start, end, part))
        start += stride

    if len(candidates) > max_windows:
        idx = np.linspace(0, len(candidates) - 1, max_windows).round().astype(int)
        candidates = [candidates[int(i)] for i in idx]

    engines: list[CPUBacktestEngine] = []
    for start, end, part in candidates:
        try:
            engine = CPUBacktestEngine(part.reset_index(drop=True), {}, direction)
            setattr(engine, "_window_start", start)
            setattr(engine, "_window_end", end)
            engines.append(engine)
        except Exception as exc:
            logger.debug("Phase 3 hidden-like window engine failed: %s", exc)

    logger.info(
        "Phase 3 [%s]: built %d hidden-like rolling validation windows",
        direction, len(engines),
    )
    return engines


def _hidden_like_window_penalty(rule_set: list[dict], val_engine) -> float:
    """Penalty for rule sets that collapse on rolling hidden-like windows."""
    engines = getattr(val_engine, "_hidden_like_engines", [])
    if not engines:
        return 0.0

    returns: list[float] = []
    sortinos: list[float] = []
    drawdowns: list[float] = []
    for engine in engines:
        try:
            m = engine.simulate_rule_set(rule_set)
        except Exception as exc:
            logger.debug("hidden-like window simulate failed: %s", exc)
            m = {"total_return_pct": -100.0, "sortino_ratio": -10.0, "max_drawdown_pct": 100.0}
        returns.append(float(m.get("total_return_pct", 0.0)))
        sortinos.append(float(m.get("sortino_ratio", m.get("total_return_pct", 0.0))))
        drawdowns.append(float(m.get("max_drawdown_pct", 0.0)))

    if not returns:
        return 0.0

    worst_ret = float(np.min(returns))
    median_sortino = float(np.median(sortinos))
    worst_dd = float(np.max(drawdowns))

    penalty = 0.0
    floor = float(_cfg.PHASE3_HIDDEN_WORST_RETURN_FLOOR)
    if worst_ret < floor:
        penalty += (floor - worst_ret) * float(_cfg.PHASE3_HIDDEN_WORST_RETURN_WEIGHT)
    if median_sortino < 0.0:
        penalty += abs(median_sortino) * float(_cfg.PHASE3_HIDDEN_SORTINO_WEIGHT)
    penalty += max(0.0, worst_dd - 10.0) * float(_cfg.PHASE3_HIDDEN_DRAWDOWN_WEIGHT)
    return float(penalty)


def _build_phase3_engines(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    direction: str,
) -> tuple[CPUBacktestEngine | object, CPUBacktestEngine | object, bool]:
    """Create validation/train engines; GPU batch when configured and available."""
    feature_modes: dict[str, str] = {}
    use_gpu = bool(_cfg.PHASE3_USE_GPU)

    if use_gpu:
        try:
            from gpu_fuzzy_trader.backtest.gpu_engine import GPUBacktestEngine

            val_engine = GPUBacktestEngine(val_df, feature_modes, direction)
            train_engine = GPUBacktestEngine(
                train_df, feature_modes, direction)
            logger.info(
                "Phase 3 using GPUBacktestEngine (batch rule-set eval)")
            return val_engine, train_engine, True
        except ImportError:
            logger.warning(
                "PHASE3_USE_GPU=True but JAX unavailable; using CPU.")

    val_engine = CPUBacktestEngine(val_df, feature_modes, direction)
    train_engine = CPUBacktestEngine(train_df, feature_modes, direction)
    try:
        setattr(
            val_engine,
            "_hidden_like_engines",
            _build_hidden_like_window_engines(train_df, val_df, direction),
        )
    except Exception as exc:
        logger.warning("Phase 3 [%s]: hidden-like validation disabled: %s", direction, exc)
    try:
        if getattr(_cfg, "PHASE3_PURGED_CV_ENABLED", False):
            from gpu_fuzzy_trader.validation.rolling_cv import build_fold_engines
            combined = pd.concat([train_df, val_df], ignore_index=True)
            feature_names = [
                c for c in combined.columns
                if c not in set(_cfg.LABEL_COLUMNS) | set(_cfg.META_COLUMNS) | set(_cfg.INTERNAL_COLUMNS)
                and not str(c).startswith("_")
            ]
            setattr(
                val_engine,
                "_purged_fold_engines",
                build_fold_engines(combined, direction, feature_names=feature_names),
            )
    except Exception as exc:
        logger.warning("Phase 3 [%s]: purged rolling CV disabled: %s", direction, exc)
    try:
        if getattr(_cfg, "MONTHLY_VALIDATION_ENABLED", False):
            combined = pd.concat([train_df, val_df], ignore_index=True)
            feature_names = [
                c for c in combined.columns
                if c not in set(_cfg.LABEL_COLUMNS) | set(_cfg.META_COLUMNS) | set(_cfg.INTERNAL_COLUMNS)
                and not str(c).startswith("_")
            ]
            setattr(val_engine, "_monthly_combined_df", combined)
            setattr(val_engine, "_monthly_feature_names", feature_names)
    except Exception as exc:
        logger.warning("Phase 3 [%s]: monthly validation disabled: %s", direction, exc)
    return val_engine, train_engine, False


def _evaluate_rule_set(
    rule_set: list[dict],
    val_engine,
    train_engine,
) -> tuple[np.ndarray, dict]:
    """
    Evaluate a candidate rule set and return (objectives, val_metrics).

    objectives = [f1, f2, f3] (all minimised, with penalties applied).
    """
    dup_penalty = 0.0
    if _has_duplicate_rules(rule_set):
        dup_penalty = 50.0

    try:
        val_metrics = val_engine.simulate_rule_set(rule_set)
    except Exception as exc:
        logger.debug("val simulate_rule_set failed: %s", exc)
        val_metrics = {
            "sortino_ratio": 0.0,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 100.0,
            "win_rate": 0.0,
            "executed_trades": 0,
            "per_symbol_metrics": {},
        }

    val_sortino = float(val_metrics.get(
        "sortino_ratio", val_metrics.get("total_return_pct", 0.0)))
    val_dd = float(val_metrics.get("max_drawdown_pct", 100.0))
    val_wr = float(val_metrics.get("win_rate", 0.0))
    val_trades = int(val_metrics.get("executed_trades", 0))

    zero_penalty = 0.0
    if val_trades == 0:
        zero_penalty = 100.0

    coverage_penalty = 0.0
    symbols_with_trades = _count_symbols_with_trades(val_metrics)
    if symbols_with_trades < _cfg.PHASE3_MIN_SYMBOL_COVERAGE:
        coverage_penalty = (
            (_cfg.PHASE3_MIN_SYMBOL_COVERAGE - symbols_with_trades) * 5.0
        )

    overfitting_penalty = 0.0
    train_metrics: dict = {}
    try:
        train_metrics = train_engine.simulate_rule_set(rule_set)
        train_sortino = float(train_metrics.get(
            "sortino_ratio", train_metrics.get("total_return_pct", 0.0)))
        overfitting_penalty = abs(
            train_sortino - val_sortino) / max(abs(train_sortino), 1.0)
    except Exception as exc:
        logger.debug("train simulate_rule_set failed: %s", exc)
        overfitting_penalty = 10.0

    symbol_consistency_penalty = _symbol_consistency_penalty(
        train_metrics, val_metrics)
    hidden_like_penalty = _hidden_like_window_penalty(rule_set, val_engine)
    cv_penalty = _purged_cv_rule_set_penalty(rule_set, val_engine)
    overlap_penalty = _signal_overlap_penalty(rule_set, val_engine)
    metric_gap_penalty = _metric_gap_penalty(train_metrics, val_metrics)
    symbol_survival_penalty = _per_symbol_survival_penalty(val_metrics)
    monthly_window_penalty = 0.0
    try:
        monthly_df = getattr(val_engine, "_monthly_combined_df", None)
        if monthly_df is not None and getattr(_cfg, "MONTHLY_VALIDATION_ENABLED", False):
            monthly_summary, _ = evaluate_rule_set_monthly(
                monthly_df, rule_set, val_engine.trade_direction,
                feature_names=getattr(val_engine, "_monthly_feature_names", None),
            )
            monthly_window_penalty = monthly_penalty(monthly_summary) * float(getattr(_cfg, "PHASE3_MONTHLY_PENALTY_WEIGHT", 1.0))
    except Exception as exc:
        logger.debug("monthly validation penalty failed: %s", exc)
        monthly_window_penalty = 10.0

    total_penalty = (
        zero_penalty + coverage_penalty + overfitting_penalty
        + dup_penalty + symbol_consistency_penalty + hidden_like_penalty
        + cv_penalty + overlap_penalty + metric_gap_penalty + symbol_survival_penalty
        + monthly_window_penalty
    )

    val_return = float(val_metrics.get("total_return_pct", 0.0))
    val_pf = float(val_metrics.get("profit_factor", 0.0))
    val_ratio = return_to_drawdown(val_return, val_dd, float(getattr(_cfg, "RETURN_DD_FLOOR", 1.0)))

    f1 = -val_ratio + total_penalty
    f2 = val_dd + total_penalty
    f3 = -val_pf + total_penalty

    objectives = np.array([f1, f2, f3], dtype=np.float64)
    return objectives, val_metrics



def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Return True if solution *a* dominates *b* (all ≤, at least one <)."""
    return bool(np.all(a <= b) and np.any(a < b))


def _non_dominated_sort(objectives: np.ndarray) -> list[list[int]]:
    """
    NSGA-II non-dominated sorting.

    Returns fronts in order: fronts[0] is the Pareto front.
    """
    n = len(objectives)
    if n == 0:
        return [[]]

    domination_count = np.zeros(n, dtype=int)
    dominated_by: list[list[int]] = [[] for _ in range(n)]
    first_front: list[int] = []

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if _dominates(objectives[i], objectives[j]):
                dominated_by[i].append(j)
            elif _dominates(objectives[j], objectives[i]):
                domination_count[i] += 1
        if domination_count[i] == 0:
            first_front.append(i)

    fronts: list[list[int]] = [first_front]
    current_front = 0

    while fronts[current_front]:
        next_front: list[int] = []
        for i in fronts[current_front]:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        current_front += 1
        if next_front:
            fronts.append(next_front)
        else:
            break

    return fronts


def _crowding_distance(objectives: np.ndarray, front: list[int]) -> np.ndarray:
    """Compute crowding distance for solutions in *front*."""
    n = len(front)
    if n <= 2:
        return np.full(n, np.inf)

    distances = np.zeros(n)
    front_obj = objectives[front]
    M = front_obj.shape[1]

    for m in range(M):
        order = np.argsort(front_obj[:, m])
        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf
        obj_range = front_obj[order[-1], m] - front_obj[order[0], m]
        if obj_range == 0:
            continue
        for k in range(1, n - 1):
            distances[order[k]] += (
                front_obj[order[k + 1], m] - front_obj[order[k - 1], m]
            ) / obj_range

    return distances



def _random_rule_set(
    pool: list[dict],
    rng: random.Random,
    min_rules: int,
    max_rules: int,
) -> list[dict]:
    """
    Sample a random rule set of 2–5 rules from the pool with no duplicates.

    Returns a list of rule dicts (subset of pool entries, with only the
    fields needed by CPUBacktestEngine: conditions, tp, sl, capital_pct).
    """
    n_rules = rng.randint(min_rules, min(max_rules, len(pool)))
    chosen = rng.sample(pool, n_rules)
    seen: set[frozenset] = set()
    unique: list[dict] = []
    for rule in chosen:
        key = _conditions_key(rule["conditions"])
        if key not in seen:
            seen.add(key)
            unique.append(rule)
    remaining = [r for r in pool if _conditions_key(
        r["conditions"]) not in seen]
    while len(unique) < min_rules and remaining:
        r = rng.choice(remaining)
        key = _conditions_key(r["conditions"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
            remaining = [x for x in remaining if _conditions_key(
                x["conditions"]) != key]
    return unique


def _rule_set_to_engine_format(rule_set: list[dict]) -> list[dict]:
    """
    Convert pool entries to the format expected by CPUBacktestEngine and repair
    any GA artifacts before evaluation.
    """
    repaired = repair_rule_set(
        rule_set,
        min_rules=1,
        max_rules=int(getattr(_cfg, "PHASE3_MAX_RULES", 5)),
    )
    return [
        {
            "conditions": list(rule["conditions"]),
            "tp": float(rule.get("tp", _cfg.PHASE2_TP)),
            "sl": float(rule.get("sl", _cfg.PHASE2_SL)),
            "capital_pct": float(rule.get("capital_pct", _cfg.PHASE2_CAPITAL_PCT)),
        }
        for rule in repaired
    ]


def _crossover_rule_sets(
    parent_a: list[dict],
    parent_b: list[dict],
    pool: list[dict],
    rng: random.Random,
    min_rules: int,
    max_rules: int,
) -> list[dict]:
    """
    Produce one child by combining rules from two parents.

    Strategy: take a random subset of rules from both parents, deduplicate,
    then trim or pad to a valid size.
    """
    combined = list(parent_a) + list(parent_b)
    rng.shuffle(combined)

    seen: set[frozenset] = set()
    unique: list[dict] = []
    for rule in combined:
        key = _conditions_key(rule["conditions"])
        if key not in seen:
            seen.add(key)
            unique.append(rule)

    if len(unique) > max_rules:
        unique = unique[:max_rules]

    remaining = [r for r in pool if _conditions_key(
        r["conditions"]) not in seen]
    while len(unique) < min_rules and remaining:
        r = rng.choice(remaining)
        key = _conditions_key(r["conditions"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
            remaining = [x for x in remaining if _conditions_key(
                x["conditions"]) != key]

    return repair_rule_set(unique, min_rules=min_rules, max_rules=max_rules)


def _mutate_rule_set(
    rule_set: list[dict],
    pool: list[dict],
    rng: random.Random,
    min_rules: int,
    max_rules: int,
    mutation_rate: float = 0.3,
) -> list[dict]:
    """
    Mutate a rule set by randomly replacing, adding, or removing rules.
    """
    child = list(rule_set)
    seen = {_conditions_key(r["conditions"]) for r in child}
    remaining = [r for r in pool if _conditions_key(
        r["conditions"]) not in seen]

    if rng.random() < mutation_rate:
        if child and remaining:
            idx = rng.randrange(len(child))
            new_rule = rng.choice(remaining)
            seen.discard(_conditions_key(child[idx]["conditions"]))
            child[idx] = new_rule
            seen.add(_conditions_key(new_rule["conditions"]))
            remaining = [r for r in pool if _conditions_key(
                r["conditions"]) not in seen]

    if rng.random() < mutation_rate and len(child) < max_rules and remaining:
        new_rule = rng.choice(remaining)
        child.append(new_rule)
        seen.add(_conditions_key(new_rule["conditions"]))

    if rng.random() < mutation_rate and len(child) > min_rules:
        idx = rng.randrange(len(child))
        child.pop(idx)

    while len(child) < min_rules and remaining:
        r = rng.choice(remaining)
        key = _conditions_key(r["conditions"])
        if key not in seen:
            seen.add(key)
            child.append(r)
            remaining = [x for x in remaining if _conditions_key(
                x["conditions"]) != key]

    if len(child) > max_rules:
        child = child[:max_rules]

    return repair_rule_set(child, min_rules=min_rules, max_rules=max_rules)



def _seed_population_from_greedy(
    greedy_set: list[dict],
    pool: list[dict],
    pop_size: int,
    min_rules: int,
    max_rules: int,
    rng: random.Random,
) -> list[list[dict]]:
    """Build initial population using greedy, clusters and cross-style sampling."""
    repaired_greedy = repair_rule_set(greedy_set, min_rules=min_rules, max_rules=max_rules)
    if int(pop_size) <= 0:
        return [repaired_greedy]
    population: list[list[dict]] = [repaired_greedy]
    if getattr(_cfg, "PHASE3_SMART_POPULATION_ENABLED", True):
        try:
            smart = smart_population_from_pool(pool, pop_size - len(population), min_rules, max_rules, rng)
            population.extend(smart)
        except Exception as exc:
            logger.debug("smart population sampler failed: %s", exc)
    while len(population) < pop_size:
        child = _mutate_rule_set(list(greedy_set), pool, rng, min_rules, max_rules)
        population.append(child)
    return population[:pop_size]


def _run_nsga2_combinatorial(
    pool: list[dict],
    val_engine,
    train_engine,
    pop_size: int,
    n_generations: int,
    min_rules: int,
    max_rules: int,
    seed: int = 42,
    initial_population: list[list[dict]] | None = None,
    use_batch: bool = False,
    log_tag: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Run NSGA-II combinatorial search over rule set combinations.

    Parameters
    ----------
    pool : list[dict]
        Phase 2 pool entries (each has "conditions", "tp", "sl", "capital_pct").
    val_engine : CPUBacktestEngine
        Engine initialised on the validation split.
    train_engine : CPUBacktestEngine
        Engine initialised on the training split.
    pop_size : int
    n_generations : int
    min_rules : int
    max_rules : int
    seed : int

    Returns
    -------
    pareto_front_rule_sets : list[list[dict]]
        Rule sets on the Pareto front (each is a list of rule dicts).
    history : list[dict]
        Per-generation metrics.
    """
    rng = random.Random(seed)

    if int(n_generations) <= 0 or int(pop_size) <= 0:
        if initial_population:
            return [list(initial_population[0])], []
        return [_random_rule_set(pool, rng, min_rules, max_rules)], []

    if len(pool) < min_rules:
        raise ValueError(
            f"Pool has only {len(pool)} rules, need at least {min_rules}."
        )

    effective_pop = min(pop_size, max(4, len(pool) * 2))

    if initial_population:
        population = [
            list(rs) for rs in initial_population[:effective_pop]
        ]
        while len(population) < effective_pop:
            population.append(
                _random_rule_set(pool, rng, min_rules, max_rules)
            )
    else:
        population = [
            _random_rule_set(pool, rng, min_rules, max_rules)
            for _ in range(effective_pop)
        ]

    objectives = np.full((effective_pop, 3), np.inf)
    history: list[dict] = []

    tag = log_tag or "Phase 3 NSGA-II"
    logger.info(
        "%s: pool=%d, pop=%d, gen=%d",
        tag, len(pool), effective_pop, n_generations,
    )
    gen_loop_start = time.monotonic()

    for gen in range(n_generations):
        pending = [i for i in range(effective_pop)
                   if np.any(np.isinf(objectives[i]))]
        if pending and use_batch and hasattr(val_engine, "simulate_rule_set_batch"):
            fmts = [_rule_set_to_engine_format(population[i]) for i in pending]
            val_list = val_engine.simulate_rule_set_batch(fmts)
            if hasattr(train_engine, "simulate_rule_set_batch"):
                train_list = train_engine.simulate_rule_set_batch(fmts)
            else:
                train_list = [
                    train_engine.simulate_rule_set(f) for f in fmts
                ]
            from gpu_fuzzy_trader.phases.phase3_greedy import _objectives_from_metrics

            for j, i in enumerate(pending):
                obj = _objectives_from_metrics(
                    val_list[j], train_list[j], population[i]
                )
                objectives[i] = obj
        else:
            for i in pending:
                engine_fmt = _rule_set_to_engine_format(population[i])
                obj, _ = _evaluate_rule_set(
                    engine_fmt, val_engine, train_engine)
                objectives[i] = obj

        fronts = _non_dominated_sort(objectives)
        pareto_indices = fronts[0]

        pareto_obj = objectives[pareto_indices]
        history.append({
            "generation": gen,
            "pareto_size": len(pareto_indices),
            "mean_f1": float(np.mean(pareto_obj[:, 0])),
            "mean_f2": float(np.mean(pareto_obj[:, 1])),
            "mean_f3": float(np.mean(pareto_obj[:, 2])),
        })

        mean_f1 = float(np.mean(pareto_obj[:, 0])) if len(pareto_obj) else 0.0
        maybe_log_generation(
            logger, tag, gen, n_generations, len(pareto_indices), mean_f1,
            loop_start=gen_loop_start,
        )

        if gen == n_generations - 1:
            break

        new_population: list[list[dict]] = []
        new_objectives = np.full((effective_pop, 3), np.inf)

        elite_count = min(len(pareto_indices), effective_pop // 2)
        cd = _crowding_distance(objectives, pareto_indices)
        cd_order = np.argsort(-cd)
        elite_indices = [pareto_indices[j] for j in cd_order[:elite_count]]

        for j, idx in enumerate(elite_indices):
            new_population.append(list(population[idx]))
            new_objectives[j] = objectives[idx].copy()

        all_indices = list(range(effective_pop))
        while len(new_population) < effective_pop:
            cands_a = rng.sample(all_indices, min(2, len(all_indices)))
            cands_b = rng.sample(all_indices, min(2, len(all_indices)))
            pa = cands_a[0] if objectives[cands_a[0],
                                          0] <= objectives[cands_a[-1], 0] else cands_a[-1]
            pb = cands_b[0] if objectives[cands_b[0],
                                          0] <= objectives[cands_b[-1], 0] else cands_b[-1]

            child = _crossover_rule_sets(
                population[pa], population[pb], pool, rng, min_rules, max_rules
            )
            child = _mutate_rule_set(child, pool, rng, min_rules, max_rules)
            new_population.append(child)

        population = new_population[:effective_pop]
        objectives = new_objectives

    fronts = _non_dominated_sort(objectives)
    pareto_indices = fronts[0]
    pareto_rule_sets = [population[i] for i in pareto_indices]

    return pareto_rule_sets, history


def _select_best_from_pareto(
    pareto_rule_sets: list[list[dict]],
    val_engine: CPUBacktestEngine,
    train_engine: CPUBacktestEngine,
) -> list[dict]:
    """Select the best rule set using return/maxDD plus validation robustness."""
    if not pareto_rule_sets:
        raise ValueError(
            "Pareto front is empty — cannot select best rule set.")

    best_idx = 0
    best_score = -np.inf

    for i, rs in enumerate(pareto_rule_sets):
        engine_fmt = _rule_set_to_engine_format(rs)
        try:
            train_m = train_engine.simulate_rule_set(engine_fmt)
            val_m = val_engine.simulate_rule_set(engine_fmt)
            score = robust_ratio_score(
                train_m, val_m, None, None,
                min_trades=int(getattr(_cfg, "AUTO_SEARCH_SCORE_MIN_TRADES", 80)),
                min_fold_trades=int(getattr(_cfg, "PHASE3_MIN_FOLD_TRADES", 20)),
                dd_floor=float(getattr(_cfg, "RETURN_DD_FLOOR", 1.0)),
            )
        except Exception:
            obj, _ = _evaluate_rule_set(engine_fmt, val_engine, train_engine)
            score = -float(obj[0])
        if score > best_score:
            best_score = score
            best_idx = i

    logger.info("Phase 3 Pareto selector: best return/DD score=%.4f", best_score)
    return pareto_rule_sets[best_idx]



def _build_output_dict(rule_set: list[dict], direction: str) -> dict:
    """
    Build the evaluator_v3.ipynb-compatible output dict.

    Uses Phase 2 static TP/SL/capital_pct values (Phase 4 will update them).
    """
    rules_list = []
    for rule in repair_rule_set(rule_set, direction=direction, min_rules=1, max_rules=int(getattr(_cfg, "PHASE3_MAX_RULES", 5))):
        rr = repair_rule(rule, direction=direction)
        rules_list.append({
            "tp": float(rr.get("tp", _cfg.PHASE2_TP)),
            "sl": float(rr.get("sl", _cfg.PHASE2_SL)),
            "capital_pct": float(rr.get("capital_pct", _cfg.PHASE2_CAPITAL_PCT)),
            "conditions": list(rr["conditions"]),
        })
    return {
        "direction": direction,
        "risk_optimized": False,
        "rules_set": rules_list,
    }



class Rule_Set_Selector:
    """
    Phase 3: greedy construction + Pareto refinement over rule set combinations.

    Selects the best ordered combination of 2–5 rules from the Phase 2 pool,
    evaluated on the validation split using CPUBacktestEngine.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training split DataFrame (already prepared).
    val_df : pd.DataFrame
        Validation split DataFrame (already prepared).
    pool : list[dict]
        Phase 2 pool entries.  Each entry must have at minimum:
        "conditions" (list[str]).  "tp", "sl", "capital_pct" are optional
        (defaults to Phase 2 static values if absent).
    direction : str
        "long" or "short".
    refine_pop_size : int, optional
        Override PHASE3_REFINE_POP_SIZE (useful for testing).
    refine_generations : int, optional
        Override PHASE3_REFINE_GENERATIONS (useful for testing).
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        pool: list[dict],
        direction: str,
        refine_pop_size: int | None = None,
        refine_generations: int | None = None,
        seed: int = 42,
    ) -> None:
        if direction not in ("long", "short"):
            raise ValueError(
                f"direction must be 'long' or 'short', got {direction!r}")
        if not pool:
            raise ValueError("pool must not be empty.")
        if len(pool) < _cfg.PHASE3_MIN_RULES:
            raise ValueError(
                f"pool must have at least {_cfg.PHASE3_MIN_RULES} rules, "
                f"got {len(pool)}."
            )

        self.direction = direction
        self.pool = pool
        self.refine_pop_size = (
            refine_pop_size if refine_pop_size is not None else _cfg.PHASE3_REFINE_POP_SIZE
        )
        self.refine_generations = (
            refine_generations
            if refine_generations is not None
            else _cfg.PHASE3_REFINE_GENERATIONS
        )
        self.seed = seed

        self._val_engine, self._train_engine, self._use_gpu_batch = _build_phase3_engines(
            train_df, val_df, direction
        )


    def run(self) -> dict:
        """
        Run greedy construction and Pareto refinement.

        Returns the best rule set dict (evaluator_v3.ipynb compatible format).
        Also persists to outputs/{direction}.json.

        Returns
        -------
        dict
            {"direction": ..., "rules_set": [...]}
        """
        logger.info(
            "Phase 3 [%s]: pool=%d, refine_pop=%d, refine_gen=%d, gpu_batch=%s",
            self.direction,
            len(self.pool),
            self.refine_pop_size,
            self.refine_generations,
            self._use_gpu_batch,
        )

        greedy_set, n_greedy_evals = greedy_rule_set_search(
            pool=self.pool,
            val_engine=self._val_engine,
            train_engine=self._train_engine,
            min_rules=_cfg.PHASE3_MIN_RULES,
            max_rules=_cfg.PHASE3_MAX_RULES,
            use_batch=self._use_gpu_batch,
        )
        logger.info(
            "Phase 3 [%s]: greedy done (%d evals), refining...",
            self.direction,
            n_greedy_evals,
        )
        initial_pop = _seed_population_from_greedy(
            greedy_set,
            self.pool,
            max(1, int(self.refine_pop_size)),
            _cfg.PHASE3_MIN_RULES,
            _cfg.PHASE3_MAX_RULES,
            random.Random(self.seed),
        )

        refine_tag = "Phase 3 [%s] refine" % self.direction
        if int(self.refine_generations) <= 0 or int(self.refine_pop_size) <= 0:
            pareto_rule_sets, history = [greedy_set], []
        else:
            pareto_rule_sets, history = _run_nsga2_combinatorial(
                pool=self.pool,
                val_engine=self._val_engine,
                train_engine=self._train_engine,
                pop_size=self.refine_pop_size,
                n_generations=self.refine_generations,
                min_rules=_cfg.PHASE3_MIN_RULES,
                max_rules=_cfg.PHASE3_MAX_RULES,
                seed=self.seed,
                initial_population=initial_pop,
                use_batch=self._use_gpu_batch,
                log_tag=refine_tag,
            )
        logger.info(
            "Phase 3 [%s]: refine complete, pareto_front=%d rule sets",
            self.direction, len(pareto_rule_sets),
        )

        if not pareto_rule_sets:
            logger.warning(
                "Phase 3 [%s]: Pareto front empty, using first %d pool rules.",
                self.direction, _cfg.PHASE3_MIN_RULES,
            )
            best_rule_set = self.pool[: _cfg.PHASE3_MIN_RULES]
        else:
            best_rule_set = _select_best_from_pareto(
                pareto_rule_sets, self._val_engine, self._train_engine
            )

        output_dict = _build_output_dict(best_rule_set, self.direction)

        os.makedirs(_cfg.OUTPUTS_DIR, exist_ok=True)
        output_path = _OUTPUT_PATHS[self.direction]
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(output_dict, fh, indent=2)

        logger.info(
            "Phase 3 [%s]: best rule set has %d rules, saved to %s",
            self.direction, len(output_dict["rules_set"]), output_path,
        )

        try:
            engine_fmt = _rule_set_to_engine_format(best_rule_set)
            train_metrics, train_log = self._train_engine.simulate_rule_set(
                engine_fmt, return_logs=True
            )
            Reporter().plot_equity_curve(train_log, "train", self.direction)
            Reporter().write_per_symbol_csv(train_metrics, "train")
        except Exception as exc:
            logger.warning(
                "Reporter train equity/csv failed (non-fatal): %s", exc
            )

        try:
            engine_fmt = _rule_set_to_engine_format(best_rule_set)
            val_metrics, val_log = self._val_engine.simulate_rule_set(
                engine_fmt, return_logs=True
            )
            Reporter().plot_equity_curve(val_log, "validation", self.direction)
            Reporter().write_per_symbol_csv(val_metrics, "validation")
        except Exception as exc:
            logger.warning(
                "Reporter validation equity/csv failed (non-fatal): %s", exc
            )

        return output_dict

    @staticmethod
    def load_rule_set(direction: str) -> Optional[dict]:
        """
        Load existing rule set if valid, return None if missing.

        Parameters
        ----------
        direction : str
            "long" or "short".

        Returns
        -------
        dict | None
            Loaded rule set if file exists and is valid, None if missing.

        Raises
        ------
        ValueError
            If the file exists but is corrupted or has an invalid schema.
        """
        if direction not in _OUTPUT_PATHS:
            raise ValueError(
                f"direction must be 'long' or 'short', got {direction!r}")

        path = _OUTPUT_PATHS[direction]
        if not os.path.exists(path):
            return None

        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(
                f"Rule set file is unreadable or corrupted: {path}"
            ) from exc

        _validate_rule_set_schema(data, path)
        return data

    @staticmethod
    def skip_if_valid() -> Optional[dict[str, dict]]:
        """
        Return loaded rule sets if both valid, None if need to run.

        Per spec:
          - If both long.json and short.json exist and pass validation → return both.
          - If only one exists → return the available one (partial dict).
          - If neither exists → return None.

        Returns
        -------
        dict[str, dict] | None
            {"long": ..., "short": ...} if both valid,
            {"long": ...} or {"short": ...} if only one valid,
            None if neither exists.
        """
        result: dict[str, dict] = {}

        for direction in ("long", "short"):
            try:
                loaded = Rule_Set_Selector.load_rule_set(direction)
                if loaded is not None:
                    result[direction] = loaded
            except ValueError:
                pass

        if not result:
            return None
        return result
