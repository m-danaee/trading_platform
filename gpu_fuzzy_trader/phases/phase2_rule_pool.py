"""
phase2_rule_pool.py — Rule_Pool_Generator (Phase 2)

GPU-accelerated multi-objective evolutionary search for fuzzy trading rules.

Uses NSGA-III via EvoX when available; falls back to NumPy NSGA-II when EvoX
is not installed.

Chromosome encoding:
    chromosome = [gene_0, gene_1, ..., gene_{K-1}]
    gene_i ∈ {0, ..., num_classes_i - 1, dont_care_i}
    dont_care_i = num_classes_i  (inactive condition)

Three objectives (all minimised):
    f1 = -sortino_ratio
    f2 = max_drawdown_pct
    f3 = -win_rate

Penalties:
    support_penalty        — if executed_trades < MIN_TRADE_SUPPORT
    diversity_penalty      — Hamming distance in chromosome space
    condition_count_penalty — active conditions outside [MIN_CONDITIONS, MAX_CONDITIONS]

Static risk parameters during Phase 2:
    TP = PHASE2_TP, SL = PHASE2_SL, capital_pct = PHASE2_CAPITAL_PCT
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.features.encoder import decode_chromosome, get_dont_care
from gpu_fuzzy_trader.phases.phase2_sparse_encoding import (
    chromosome_key,
    dense_to_sparse,
    is_sparse_chromosome,
    max_slots,
    sparse_hamming,
    sparse_to_dense,
    use_sparse_slots,
)
from gpu_fuzzy_trader.log_progress import maybe_log_generation
from gpu_fuzzy_trader.phases.phase2_support import (
    compute_support_penalty_and_specialist,
    passes_pool_admission_gate,
    passes_pool_trade_floor,
    regime_row_fractions,
    trade_support_penalty as _regime_trade_support_penalty,
)
from gpu_fuzzy_trader.reporting.reporter import Reporter

logger = logging.getLogger(__name__)


def trade_support_penalty(executed: int, **kwargs) -> float:
    """Backward-compatible wrapper returning penalty only."""
    penalty, _, _ = _regime_trade_support_penalty(executed, **kwargs)
    return penalty


def _prepare_regime_context(
    sampled_df: pd.DataFrame,
) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    """
    Assign regime labels to *sampled_df* rows using the Phase 1 artifact.

    Returns (regime_ids, regime_row_fractions_arr, n_regimes).
    """
    if not _cfg.PHASE2_REGIME_SUPPORT_ENABLED:
        return None, None, 0

    from gpu_fuzzy_trader.features.regime_cluster import (
        assign_regime_labels,
        load_regime_model,
    )

    try:
        bundle = load_regime_model(_cfg.PHASE2_REGIME_MODEL_PATH)
    except FileNotFoundError:
        logger.warning(
            "Phase 2 regime support: model not found at %s; using static penalty",
            _cfg.PHASE2_REGIME_MODEL_PATH,
        )
        return None, None, 0

    missing = [c for c in bundle["regime_features"]
               if c not in sampled_df.columns]
    if missing:
        logger.warning(
            "Phase 2 regime support: missing columns %s; using static penalty",
            missing,
        )
        return None, None, 0

    try:
        labels = assign_regime_labels(sampled_df, bundle)
    except Exception as exc:
        logger.warning(
            "Phase 2 regime support: label assignment failed (%s); static penalty",
            exc,
        )
        return None, None, 0

    n_regimes = int(bundle.get("n_clusters", labels.nunique()))
    regime_ids = labels.reindex(sampled_df.index).fillna(
        0).astype(np.int32).values
    fracs = regime_row_fractions(regime_ids, n_regimes)
    logger.info(
        "Phase 2 regime support: n_regimes=%d, row_counts=%s",
        n_regimes,
        np.bincount(regime_ids.astype(np.int64), minlength=n_regimes).tolist(),
    )
    return regime_ids, fracs, n_regimes


def _saturating_sortino(raw: float) -> float:
    """tanh-saturated Sortino so the best-front member moves with progress.

    The previous flat cap pinned best_sortino at the SORTINO_CAP sentinel from
    generation 0 (visible in phase2_long_history.json: best=10.0 at gen 0..99).
    """
    scale = max(_cfg.SORTINO_SCALE, 1e-6)
    cap = _cfg.SORTINO_CAP
    return float(np.tanh(raw / scale) * cap)


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

# These are set dynamically by run_pipeline._temporary_output_paths()
# to support run-specific output directories.
# Legacy direction keys ("long"/"short") or symbol keys ("long", "SYM_A").
_POOL_PATHS: dict = {
    "long": _cfg.PHASE2_POOL_PATHS["long"],
    "short": _cfg.PHASE2_POOL_PATHS["short"],
}
_HISTORY_PATHS: dict = {
    "long": _cfg.PHASE2_HISTORY_PATHS["long"],
    "short": _cfg.PHASE2_HISTORY_PATHS["short"],
}
# Archive stays persistent in project root (not run-specific)
_ARCHIVE_PATHS = dict(_cfg.PHASE2_ARCHIVE_PATHS)


def _pool_path_key(direction: str, symbol: str | None = None):
    if symbol is not None:
        return (direction, symbol)
    return direction


def rebind_symbol_phase2_paths(
    output_root: str,
    symbols: list[str],
) -> None:
    """Register per-symbol pool/history paths for the active run output root."""
    global _POOL_PATHS, _HISTORY_PATHS
    if not _cfg.PHASE2_SYMBOL_SPECIALIST_ENABLED:
        return
    for direction in ("long", "short"):
        for symbol in symbols:
            key = _pool_path_key(direction, symbol)
            _POOL_PATHS[key] = _cfg.phase2_symbol_pool_path(
                direction, symbol, output_root,
            )
            _HISTORY_PATHS[key] = _cfg.phase2_symbol_history_path(
                direction, symbol, output_root,
            )


def _resolve_pool_path(direction: str, symbol: str | None = None) -> str:
    key = _pool_path_key(direction, symbol)
    if key in _POOL_PATHS:
        return _POOL_PATHS[key]
    if symbol is not None:
        return _cfg.phase2_symbol_pool_path(direction, symbol)
    return _cfg.PHASE2_POOL_PATHS[direction]


def _resolve_history_path(direction: str, symbol: str | None = None) -> str:
    key = _pool_path_key(direction, symbol)
    if key in _HISTORY_PATHS:
        return _HISTORY_PATHS[key]
    if symbol is not None:
        return _cfg.phase2_symbol_history_path(direction, symbol)
    return _cfg.PHASE2_HISTORY_PATHS[direction]


# ---------------------------------------------------------------------------
# EvoX optional import
# ---------------------------------------------------------------------------

try:
    from evox.core import Algorithm as EvoxAlgorithm  # type: ignore
    _EVOX_AVAILABLE = True
except ImportError:
    EvoxAlgorithm = None  # type: ignore[misc, assignment]
    _EVOX_AVAILABLE = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_dont_cares(feature_infos: list[dict]) -> np.ndarray:
    """Return array of dont_care sentinels for each feature."""
    return np.array([get_dont_care(fi["mode"]) for fi in feature_infos], dtype=np.int32)


def _count_active_conditions(chromosome: np.ndarray, dont_cares: np.ndarray) -> int:
    """Count active rule conditions (sparse slots or dense dont_care encoding)."""
    if is_sparse_chromosome(chromosome):
        from gpu_fuzzy_trader.phases.phase2_sparse_encoding import count_active_slots
        return count_active_slots(chromosome)
    return int(np.sum(chromosome != dont_cares))


def _chromosome_batch(chromosome: np.ndarray) -> np.ndarray:
    """Shape (1, ...) batch for simulate_rule_batch."""
    chrom = np.asarray(chromosome, dtype=np.int32)
    if is_sparse_chromosome(chrom):
        return chrom[None, :, :]
    return chrom[None, :]


def _chromosome_for_pool_export(
    chromosome: np.ndarray,
    dont_cares: np.ndarray,
) -> np.ndarray:
    """Dense K-vector for pool JSON / decode_chromosome."""
    if is_sparse_chromosome(chromosome):
        return sparse_to_dense(chromosome, dont_cares)
    return np.asarray(chromosome, dtype=np.int32)


def _hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Hamming distance between two chromosomes (active pairs when sparse)."""
    if is_sparse_chromosome(a) or is_sparse_chromosome(b):
        return sparse_hamming(a, b)
    return int(np.sum(a != b))


def _pareto_sortino_stats(
    pareto_indices: list[int],
    metrics_cache: list[dict],
) -> dict[str, float]:
    """Aggregate raw Sortino and return health over the current Pareto front."""
    if not pareto_indices:
        return {
            "mean_sortino_ratio": 0.0,
            "best_sortino_ratio": 0.0,
            "mean_raw_train_return_pct": 0.0,
            "mean_val_return_pct": 0.0,
        }
    sortinos = [
        float(metrics_cache[i].get("sortino_ratio",
              metrics_cache[i].get("total_return_pct", 0.0)))
        for i in pareto_indices
    ]
    train_returns = [
        float(metrics_cache[i].get("total_return_pct", 0.0))
        for i in pareto_indices
    ]
    val_returns = [
        float(metrics_cache[i].get("val_total_return_pct", 0.0))
        for i in pareto_indices
        if metrics_cache[i].get("val_total_return_pct") is not None
    ]
    return {
        "mean_sortino_ratio": float(np.mean(sortinos)),
        "best_sortino_ratio": float(np.max(sortinos)),
        "mean_raw_train_return_pct": float(np.mean(train_returns)),
        "mean_val_return_pct": float(np.mean(val_returns)) if val_returns else 0.0,
    }


_pareto_return_stats = _pareto_sortino_stats


def _symbol_robustness_penalty(metrics: dict) -> float:
    """Penalty for weak cross-symbol robustness on one split."""
    per_sym = metrics.get("per_symbol_metrics", {}) or {}
    if not per_sym:
        return 0.0
    pnl_vec: list[float] = []
    profitable = 0
    for row in per_sym.values():
        if not isinstance(row, dict):
            continue
        pnl = float(row.get("net_pnl", 0.0))
        pnl_pct = (pnl / _cfg.INITIAL_CAPITAL) * 100.0
        pnl_vec.append(pnl_pct)
        if pnl > 0.0:
            profitable += 1
    if not pnl_vec:
        return 0.0
    penalty = 0.0
    med = float(np.median(np.asarray(pnl_vec, dtype=np.float64)))
    if med < _cfg.PHASE2_SYMBOL_MEDIAN_RETURN_FLOOR_PCT:
        penalty += abs(_cfg.PHASE2_SYMBOL_MEDIAN_RETURN_FLOOR_PCT - med)
    shortfall = max(0, _cfg.PHASE2_MIN_PROFITABLE_SYMBOLS - profitable)
    penalty += float(shortfall) * 2.0
    return penalty


def _sort_chronological(df: pd.DataFrame) -> pd.DataFrame:
    """Preserve per-symbol time order required by exposure/release simulation."""
    if df.empty:
        return df
    if "symbol" in df.columns and "datetime" in df.columns:
        return df.sort_values(
            ["symbol", "datetime"], kind="mergesort",
        ).reset_index(drop=True)
    if "symbol" in df.columns and "_symbol_bar_index" in df.columns:
        return df.sort_values(
            ["symbol", "_symbol_bar_index"], kind="mergesort",
        ).reset_index(drop=True)
    if "datetime" in df.columns:
        return df.sort_values("datetime", kind="mergesort").reset_index(drop=True)
    if "_symbol_bar_index" in df.columns:
        return df.sort_values("_symbol_bar_index", kind="mergesort").reset_index(
            drop=True,
        )
    return df.reset_index(drop=True)


def _downsample_chronological(df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    """Evenly spaced chronological downsampling (deterministic, order-preserving)."""
    ordered = _sort_chronological(df)
    total = len(ordered)
    if total <= n_rows:
        return ordered
    idx = np.linspace(0, total - 1, num=n_rows, dtype=np.int64)
    idx = np.unique(idx)
    return ordered.iloc[idx].reset_index(drop=True)


def _sample_df(
    df: pd.DataFrame,
    total_rows: int,
    random_state: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    """
    Sample up to *total_rows* rows from *df*, distributed equally across symbols.

    Rows are taken in chronological order per symbol (deterministic stride
    downsampling). Random sampling is intentionally avoided because backtest
    engines rely on row order and ``_symbol_bar_index`` for exposure release,
    matching ``evaluator_v3.ipynb`` semantics.

    *random_state* is accepted for API compatibility but ignored.
    """
    del random_state  # chronology-preserving sampling is deterministic
    if "symbol" not in df.columns:
        return _downsample_chronological(df, min(total_rows, len(df)))

    symbols = df["symbol"].unique()
    n_sym = len(symbols)
    rows_per_sym = max(1, total_rows // n_sym)

    parts = []
    for sym in symbols:
        sym_df = df[df["symbol"] == sym]
        n = min(rows_per_sym, len(sym_df))
        parts.append(_downsample_chronological(sym_df, n))

    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Fitness evaluation
# ---------------------------------------------------------------------------

def _val_trade_floor_for_objectives() -> int:
    """Minimum validation trades before joint Sortino is trusted."""
    if str(_cfg.SPLIT_MODE).strip().lower() == "purged_rolling_cv":
        return max(int(_cfg.PHASE2_CV_MIN_VAL_TRADES), 1)
    return max(int(_cfg.MIN_TRADE_POOL_FLOOR) // 4, 10)


def compute_phase2_objectives_from_metrics(
    chromosome: np.ndarray,
    dont_cares: np.ndarray,
    metrics: dict,
    pareto_front: list[np.ndarray],
    *,
    val_metrics: dict | None = None,
    regime_row_fractions_arr: np.ndarray | None = None,
    val_regime_row_counts: np.ndarray | None = None,
    diversity_reference: list[np.ndarray] | None = None,
    stage_params=None,
    symbol_scope: str | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Build Phase 2 minimisation objectives from precomputed train/val metrics.

    Shared by single-chromosome evaluation and EvoX batch assignment so penalty
    logic stays identical across code paths.
    """
    active = _count_active_conditions(chromosome, dont_cares)

    cond_penalty = 0.0
    if active < _cfg.MIN_CONDITIONS:
        cond_penalty = (_cfg.MIN_CONDITIONS - active) * 10.0
    elif active > _cfg.MAX_CONDITIONS:
        cond_penalty = (active - _cfg.MAX_CONDITIONS) * 10.0

    raw_sortino = float(metrics.get(
        "sortino_ratio", metrics.get("total_return_pct", 0.0)))
    total_return = float(metrics.get("total_return_pct", 0.0))
    profit_factor = float(metrics.get("profit_factor", 0.0))
    sortino_train = _saturating_sortino(raw_sortino)
    max_dd = float(metrics.get("max_drawdown_pct", 100.0))
    win_rate = float(metrics.get("win_rate", 0.0))
    executed = int(metrics.get("executed_trades", 0))

    sortino_for_obj = sortino_train
    val_floor_penalty = 0.0
    val_trade_floor = _val_trade_floor_for_objectives()

    if val_metrics is not None:
        raw_val_sortino = float(val_metrics.get(
            "sortino_ratio", val_metrics.get("total_return_pct", 0.0)))
        val_total_return = float(val_metrics.get("total_return_pct", 0.0))
        val_profit_factor = float(val_metrics.get("profit_factor", 0.0))
        sortino_val = _saturating_sortino(raw_val_sortino)
        val_executed = int(val_metrics.get("executed_trades", 0))
        metrics["val_sortino_ratio"] = raw_val_sortino
        metrics["val_total_return_pct"] = val_total_return
        metrics["val_executed_trades"] = val_executed
        metrics["val_profit_factor"] = val_profit_factor
        metrics["val_max_drawdown_pct"] = float(
            val_metrics.get("max_drawdown_pct", 0.0))
        if _cfg.PHASE2_JOINT_TRAIN_VAL:
            if val_executed < val_trade_floor:
                sortino_for_obj = min(sortino_train, 0.0)
            else:
                sortino_for_obj = min(sortino_train, sortino_val)
        if val_total_return < _cfg.PHASE2_VAL_RETURN_FLOOR_PCT:
            val_floor_penalty += _cfg.SUPPORT_PENALTY_MAX
        if val_profit_factor < _cfg.PHASE2_PROFIT_FACTOR_FLOOR:
            val_floor_penalty += (
                _cfg.PHASE2_PROFIT_FACTOR_FLOOR - val_profit_factor
            ) * 5.0

    support_penalty, is_specialist, dominant_regime = (
        compute_support_penalty_and_specialist(
            metrics,
            regime_row_fractions_arr,
            val_metrics=val_metrics,
            val_regime_row_counts=val_regime_row_counts,
        )
    )
    if (
        val_metrics is not None
        and int(val_metrics.get("executed_trades", 0)) < val_trade_floor
    ):
        support_penalty = max(support_penalty, _cfg.SUPPORT_PENALTY_MAX)
        if _cfg.PHASE2_JOINT_TRAIN_VAL:
            sortino_for_obj = min(sortino_train, 0.0)
        is_specialist = False

    if is_specialist:
        metrics["regime_specialist"] = True
        metrics["dominant_regime"] = dominant_regime

    if total_return < _cfg.PHASE2_RETURN_FLOOR_PCT:
        support_penalty = max(support_penalty, _cfg.SUPPORT_PENALTY_MAX)
    if profit_factor < _cfg.PHASE2_PROFIT_FACTOR_FLOOR:
        support_penalty += (_cfg.PHASE2_PROFIT_FACTOR_FLOOR - profit_factor) * 5.0

    if symbol_scope is None:
        support_penalty += _symbol_robustness_penalty(metrics)
        if val_metrics is not None:
            support_penalty += _symbol_robustness_penalty(val_metrics)
    else:
        from gpu_fuzzy_trader.phases.phase2_support import (
            passes_symbol_island_robustness_gate,
        )

        if not passes_symbol_island_robustness_gate(metrics, val_metrics):
            support_penalty = max(support_penalty, _cfg.SUPPORT_PENALTY_MAX)
    support_penalty += val_floor_penalty

    dd_gate = getattr(_cfg, "PHASE2_MAX_DRAWDOWN_GATE", 20.0)
    drawdown_gate_penalty = 0.0
    if max_dd > dd_gate:
        excess = max_dd - dd_gate
        drawdown_gate_penalty = excess * 2.0

    diversity_penalty = 0.0
    diversity_refs = list(pareto_front)
    if diversity_reference:
        seen_keys: set[tuple[int, ...]] = set()
        from gpu_fuzzy_trader.phases.phase2_sparse_encoding import chromosome_key
        merged_refs: list[np.ndarray] = []
        for ref in diversity_refs + list(diversity_reference):
            key = chromosome_key(ref)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            merged_refs.append(ref)
        diversity_refs = merged_refs
    diversity_hamming_threshold = (
        int(stage_params.diversity_hamming_threshold)
        if stage_params is not None
        else int(_cfg.PHASE2_DIVERSITY_HAMMING_THRESHOLD)
    )
    diversity_penalty_weight = (
        float(stage_params.diversity_penalty)
        if stage_params is not None
        else float(_cfg.PHASE2_DIVERSITY_PENALTY)
    )
    if diversity_refs:
        min_hamming = min(_hamming_distance(chromosome, pf)
                          for pf in diversity_refs)
        if min_hamming <= diversity_hamming_threshold:
            diversity_penalty = diversity_penalty_weight

    from gpu_fuzzy_trader.phases.phase2_support import (
        feasibility_violation_score,
        robust_return_pct,
    )

    f3_val = win_rate
    if _cfg.PHASE2_USE_TOTAL_RETURN_OBJ:
        if getattr(_cfg, "PHASE2_USE_ROBUST_RETURN_OBJ", True) and val_metrics is not None:
            f3_val = robust_return_pct(metrics, val_metrics)
        else:
            f3_val = total_return

    infeasible_penalty = 0.0
    if val_metrics is not None and _cfg.PHASE2_POOL_REQUIRE_POSITIVE_SPLITS:
        violation = feasibility_violation_score(
            metrics,
            val_metrics,
            cv_fold=str(_cfg.SPLIT_MODE).strip(
            ).lower() == "purged_rolling_cv",
        )
        if violation > 0.0:
            infeasible_penalty = (
                float(_cfg.PHASE2_INFEASIBLE_OBJECTIVE_PENALTY)
                + violation * float(_cfg.PHASE2_FEASIBILITY_VIOLATION_WEIGHT)
            )
            metrics["feasibility_violation"] = violation

    trade_penalty = 0.0
    trade_floor = (
        _cfg.PHASE2_CV_MIN_TRADE_POOL_FLOOR
        if str(_cfg.SPLIT_MODE).strip().lower() == "purged_rolling_cv"
        else _cfg.MIN_TRADE_POOL_FLOOR
    )
    if executed < trade_floor:
        max_dd = 100.0
        sortino_for_obj = 0.0
        f3_val = 0.0
        trade_penalty = 50.0

    f1 = (
        -sortino_for_obj
        + (_cfg.PHASE2_SUPPORT_PENALTY_WEIGHT_F1 * support_penalty)
        + diversity_penalty
        + cond_penalty
        + trade_penalty
        + drawdown_gate_penalty
        + infeasible_penalty
    )
    f2 = (
        max_dd
        + (_cfg.PHASE2_SUPPORT_PENALTY_WEIGHT_F2 * support_penalty)
        + diversity_penalty
        + cond_penalty
        + trade_penalty
        + drawdown_gate_penalty
        + infeasible_penalty
    )
    f3 = (
        -f3_val
        + (_cfg.PHASE2_SUPPORT_PENALTY_WEIGHT_F3 * support_penalty)
        + diversity_penalty
        + cond_penalty
        + trade_penalty
        + drawdown_gate_penalty
        + infeasible_penalty
    )

    if val_metrics is not None:
        metrics["robust_return_pct"] = robust_return_pct(metrics, val_metrics)

    objectives = np.array([f1, f2, f3], dtype=np.float64)
    return objectives, metrics


def _evaluate_chromosome(
    chromosome: np.ndarray,
    dont_cares: np.ndarray,
    engine,  # GPUBacktestEngine or CPUBacktestEngine
    pareto_front: list[np.ndarray],
    val_engine=None,  # optional second engine for joint train+val objective
    regime_row_fractions_arr: np.ndarray | None = None,
    val_regime_row_counts: np.ndarray | None = None,
    diversity_reference: list[np.ndarray] | None = None,
    stage_params=None,
    symbol_scope: str | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Evaluate a single chromosome and return (objectives, metrics).

    objectives = [f1, f2, f3] (all minimised, with penalties applied).

    When PHASE2_JOINT_TRAIN_VAL is enabled and *val_engine* is provided, f1 uses
    min(saturated_train_sortino, saturated_val_sortino) so the search prefers
    rules that hold up out-of-sample.
    """
    try:
        metrics_list = engine.simulate_rule_batch(
            chromosomes=_chromosome_batch(chromosome),
            tp=_cfg.PHASE2_TP,
            sl=_cfg.PHASE2_SL,
            capital_pct=_cfg.PHASE2_CAPITAL_PCT,
        )
        metrics = metrics_list[0]
    except Exception as exc:
        logger.debug("simulate_rule_batch failed: %s", exc)
        metrics = {
            "sortino_ratio": 0.0,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 100.0,
            "win_rate": 0.0,
            "executed_trades": 0,
        }

    val_metrics: dict | None = None
    if val_engine is not None:
        try:
            val_list = val_engine.simulate_rule_batch(
                chromosomes=_chromosome_batch(chromosome),
                tp=_cfg.PHASE2_TP,
                sl=_cfg.PHASE2_SL,
                capital_pct=_cfg.PHASE2_CAPITAL_PCT,
            )
            val_metrics = val_list[0]
        except Exception as exc:
            logger.debug("val simulate_rule_batch failed: %s", exc)
            val_metrics = None

    if regime_row_fractions_arr is None:
        regime_row_fractions_arr = getattr(
            engine, "_regime_row_fractions", None)
    if val_regime_row_counts is None and val_engine is not None:
        val_regime_row_counts = getattr(val_engine, "_regime_row_counts", None)

    symbol_scope = getattr(engine, "_symbol_scope", None)
    return compute_phase2_objectives_from_metrics(
        chromosome,
        dont_cares,
        metrics,
        pareto_front,
        val_metrics=val_metrics,
        regime_row_fractions_arr=regime_row_fractions_arr,
        val_regime_row_counts=val_regime_row_counts,
        diversity_reference=diversity_reference,
        stage_params=stage_params,
        symbol_scope=symbol_scope,
    )


# ---------------------------------------------------------------------------
# NSGA-II helpers (non-dominated sorting + crowding distance)
# ---------------------------------------------------------------------------

def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Return True if solution *a* dominates *b* (all objectives ≤, at least one <)."""
    return bool(np.all(a <= b) and np.any(a < b))


def _non_dominated_sort(objectives: np.ndarray) -> list[list[int]]:
    """
    NSGA-II non-dominated sorting.

    Parameters
    ----------
    objectives : np.ndarray
        Shape (N, M) — N solutions, M objectives (all minimised).

    Returns
    -------
    list[list[int]]
        Fronts in order: fronts[0] is the Pareto front, fronts[1] is the
        second front, etc.  Always returns at least one (possibly empty) front.
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
    """
    Compute crowding distance for solutions in *front*.

    Parameters
    ----------
    objectives : np.ndarray
        Shape (N, M).
    front : list[int]
        Indices of solutions in this front.

    Returns
    -------
    np.ndarray
        Shape (len(front),) crowding distances.
    """
    n = len(front)
    if n <= 2:
        return np.full(n, np.inf)

    distances = np.zeros(n)
    front_obj = objectives[front]  # (n, M)
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


# ---------------------------------------------------------------------------
# Population initialisation
# ---------------------------------------------------------------------------

def _init_population(
    pop_size: int,
    feature_infos: list[dict],
    rng: np.random.Generator,
    dont_care_prob: float = 0.5,
    seeded_chromosomes: np.ndarray | list[np.ndarray] | None = None,
    seed_fraction: float | None = None,
    *,
    init_strategy: str | None = None,
    stratum_fractions: tuple[float, float] | None = None,
    feature_probs: np.ndarray | None = None,
) -> np.ndarray:
    """
    Initialise a population of chromosomes.

    *init_strategy* ``"stratified_sparse"`` (default from config) enforces
    ``MIN_CONDITIONS``–``MAX_CONDITIONS`` active genes via Phase 1–guided strata.
    ``"legacy"`` uses independent per-gene *dont_care_prob* sampling.
    """
    from gpu_fuzzy_trader.phases.phase2_init import (
        assign_strata_to_indices,
        build_feature_sampling_probs,
        pick_active_count,
        repair_active_count,
        sample_sparse_chromosome,
    )

    K = len(feature_infos)
    dont_cares = _get_dont_cares(feature_infos)
    if use_sparse_slots():
        population = np.full(
            (pop_size, max_slots(), 2),
            -1,
            dtype=np.int32,
        )
        population[:, :, 1] = 0
    else:
        population = np.zeros((pop_size, K), dtype=np.int32)
    if seed_fraction is None:
        seed_fraction = _cfg.PHASE2_ARCHIVE_SEED_FRACTION
    if init_strategy is None:
        init_strategy = _cfg.PHASE2_INIT_STRATEGY
    if stratum_fractions is None:
        stratum_fractions = _cfg.PHASE2_INIT_STRATUM_FRACTIONS

    seed_rows: list[np.ndarray] = []
    if seeded_chromosomes is not None:
        seed_array = np.asarray(seeded_chromosomes, dtype=np.int32)
        if seed_array.ndim == 1:
            seed_array = seed_array[None, :]
        if use_sparse_slots() and seed_array.ndim == 3:
            if seed_array.shape[1:] != (max_slots(), 2):
                raise ValueError(
                    f"sparse seeded_chromosomes must have shape (_, {max_slots()}, 2)")
        elif seed_array.ndim != 2:
            raise ValueError(
                "seeded_chromosomes must be a 1D, 2D, or sparse 3D array-like value")
        elif seed_array.shape[1] != K:
            raise ValueError(
                f"seeded_chromosomes must have {K} genes per chromosome, "
                f"got {seed_array.shape[1]}"
            )

        seen: set[tuple[int, ...]] = set()
        for row in seed_array:
            if use_sparse_slots():
                if row.ndim == 1:
                    repaired = dense_to_sparse(row, dont_cares)
                else:
                    from gpu_fuzzy_trader.phases.phase2_sparse_encoding import (
                        repair_sparse_slots,
                    )
                    repaired = repair_sparse_slots(
                        row.astype(np.int32, copy=True),
                        feature_infos,
                        dont_cares,
                        rng,
                    )
                key = chromosome_key(repaired)
            else:
                repaired = row.astype(np.int32, copy=True)
                for k, dc in enumerate(dont_cares):
                    gene = int(repaired[k])
                    if gene < 0:
                        repaired[k] = 0
                    elif gene > int(dc):
                        repaired[k] = int(dc)
                key = tuple(int(v) for v in repaired.tolist())
            if key in seen:
                continue
            seen.add(key)
            seed_rows.append(repaired)

    seed_count = 0
    if seed_rows and pop_size > 0 and seed_fraction > 0:
        seed_count = min(
            pop_size,
            max(1, int(round(pop_size * seed_fraction))),
            len(seed_rows),
        )

    seeded_mask = np.zeros(pop_size, dtype=bool)
    if seed_count > 0:
        pick = min(seed_count, len(seed_rows))
        if pick >= len(seed_rows):
            chosen_rows = seed_rows[:pick]
        else:
            chosen_idx = rng.choice(len(seed_rows), size=pick, replace=False)
            chosen_rows = [seed_rows[int(i)] for i in chosen_idx]
        seed_positions = rng.choice(pop_size, size=pick, replace=False)
        for position, chrom in zip(seed_positions, chosen_rows):
            population[int(position)] = chrom
        seeded_mask[seed_positions] = True

    if init_strategy == "legacy":
        if not use_sparse_slots():
            for k, fi in enumerate(feature_infos):
                dc = dont_cares[k]
                num_classes = dc
                for i in np.where(~seeded_mask)[0]:
                    if rng.random() < dont_care_prob:
                        population[i, k] = dc
                    else:
                        population[i, k] = int(rng.integers(0, num_classes))
            return population

        from gpu_fuzzy_trader.phases.phase2_sparse_encoding import (
            empty_slots,
            sample_sparse_slots_chromosome,
        )
        if feature_probs is None:
            feature_probs = build_feature_sampling_probs(feature_infos)
        for i in np.where(~seeded_mask)[0]:
            if rng.random() < dont_care_prob:
                population[i] = empty_slots()
            else:
                k_active = pick_active_count(rng)
                population[i] = sample_sparse_slots_chromosome(
                    rng, feature_infos, dont_cares, k_active,
                    "explorer", feature_probs,
                )
        return population

    if feature_probs is None:
        feature_probs = build_feature_sampling_probs(feature_infos)

    fresh_indices = np.where(~seeded_mask)[0]
    strata = assign_strata_to_indices(
        fresh_indices,
        stratum_fractions,
        rng,
    )
    for row_idx, stratum in zip(fresh_indices, strata):
        k_active = pick_active_count(rng)
        if use_sparse_slots():
            from gpu_fuzzy_trader.phases.phase2_sparse_encoding import (
                sample_sparse_slots_chromosome,
            )
            population[row_idx] = sample_sparse_slots_chromosome(
                rng,
                feature_infos,
                dont_cares,
                k_active,
                stratum,
                feature_probs,
            )
        else:
            population[row_idx] = sample_sparse_chromosome(
                rng,
                feature_infos,
                dont_cares,
                k_active,
                stratum,
                feature_probs,
            )
            population[row_idx] = repair_active_count(
                population[row_idx],
                feature_infos,
                dont_cares,
                rng,
                feature_probs,
            )

    return population


# ---------------------------------------------------------------------------
# Crossover and mutation
# ---------------------------------------------------------------------------

def _crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform crossover (dense per-gene or sparse per-slot)."""
    if use_sparse_slots() or is_sparse_chromosome(parent_a):
        from gpu_fuzzy_trader.phases.phase2_sparse_encoding import crossover_sparse
        return crossover_sparse(parent_a, parent_b, rng)
    K = len(parent_a)
    mask = rng.random(K) < 0.5
    child_a = np.where(mask, parent_a, parent_b).astype(np.int32)
    child_b = np.where(mask, parent_b, parent_a).astype(np.int32)
    return child_a, child_b


def _mutate(
    chromosome: np.ndarray,
    feature_infos: list[dict],
    dont_cares: np.ndarray,
    rng: np.random.Generator,
    mutation_rate: float = 0.1,
    feature_probs: np.ndarray | None = None,
    *,
    weighted_activate_prob: float | None = None,
) -> np.ndarray:
    """
    Mutate a chromosome (returns a copy).

    When activating a dont_care gene, feature index is chosen with probability
    *weighted_activate_prob* from *feature_probs* and otherwise uniformly.
    Active count is repaired to [MIN_CONDITIONS, MAX_CONDITIONS] afterward.
    """
    from gpu_fuzzy_trader.phases.phase2_init import (
        _pick_inactive_index,
        _random_active_class,
        repair_active_count,
    )

    if weighted_activate_prob is None:
        weighted_activate_prob = _cfg.PHASE2_MUTATION_WEIGHTED_ACTIVATE_PROB

    if use_sparse_slots() or is_sparse_chromosome(chromosome):
        from gpu_fuzzy_trader.phases.phase2_sparse_encoding import mutate_sparse
        chrom = np.asarray(chromosome, dtype=np.int32)
        if use_sparse_slots() and chrom.ndim == 1:
            chrom = dense_to_sparse(chrom, dont_cares)
        return mutate_sparse(
            chrom,
            feature_infos,
            dont_cares,
            rng,
            mutation_rate=mutation_rate,
            feature_probs=feature_probs,
            weighted_activate_prob=weighted_activate_prob,
        )

    child = chromosome.copy()
    K = len(child)
    for k in range(K):
        if rng.random() < mutation_rate:
            dc = int(dont_cares[k])
            num_classes = dc
            if child[k] == dc:
                if feature_probs is not None and rng.random() < weighted_activate_prob:
                    idx = _pick_inactive_index(
                        rng, child, dont_cares, feature_probs, 1.0,
                    )
                    if idx is not None:
                        mode = feature_infos[idx]["mode"]
                        child[idx] = _random_active_class(
                            rng, mode, int(dont_cares[idx]),
                        )
                else:
                    child[k] = _random_active_class(
                        rng, feature_infos[k]["mode"], num_classes,
                    )
            else:
                if rng.random() < 0.3:
                    child[k] = dc
                else:
                    child[k] = _random_active_class(
                        rng, feature_infos[k]["mode"], num_classes,
                    )

    return repair_active_count(
        child,
        feature_infos,
        dont_cares,
        rng,
        feature_probs,
        weighted_activate_prob=weighted_activate_prob,
    )


# ---------------------------------------------------------------------------
# Fallback evolutionary algorithm (NSGA-II style)
# ---------------------------------------------------------------------------

def _metrics_dict_from_population(
    population: np.ndarray,
    metrics_cache: list[dict],
) -> dict[tuple, dict]:
    """Map chromosome tuples to cached fitness metrics."""
    out: dict[tuple, dict] = {}
    for i, met in enumerate(metrics_cache):
        if met:
            out[chromosome_key(population[i])] = met
    return out


# ---------------------------------------------------------------------------
# Pool construction
# ---------------------------------------------------------------------------

def _simulate_val_metrics_for_chrom(
    chrom: np.ndarray,
    val_engine,
) -> dict | None:
    """Full validation backtest for one chromosome (pool admission)."""
    if val_engine is None:
        return None
    try:
        val_list = val_engine.simulate_rule_batch(
            chromosomes=_chromosome_batch(chrom),
            tp=_cfg.PHASE2_TP,
            sl=_cfg.PHASE2_SL,
            capital_pct=_cfg.PHASE2_CAPITAL_PCT,
        )
        return val_list[0]
    except Exception:
        return None


def _build_pool_from_archive(
    archive: list[np.ndarray],
    feature_infos: list[dict],
    dont_cares: np.ndarray,
    engine,
    metrics_by_chrom: dict[tuple, dict] | None = None,
    regime_row_fractions_arr: np.ndarray | None = None,
    val_engine=None,
    direction: str = "",
) -> list[dict]:
    """
    Convert a list of Pareto-front chromosomes into pool JSON entries.

    When running with purged CV engines the entire archive is evaluated in one
    ``simulate_rule_batch`` call per fold (via
    ``evaluate_purged_cv_pool_admission_batch``) instead of calling the engine
    once per chromosome — eliminating the silent hang that previously occurred
    after gen 80/80.

    Each pool entry schema:
    {
        "chromosome": [...],
        "conditions": [...],
        "objectives": {"sortino_ratio": ..., "max_drawdown_pct": ..., "win_rate": ...},
        "executed_trades": ...
    }
    """
    from gpu_fuzzy_trader.phases.phase2_cv import (
        PurgedCVTrainEngine,
        PurgedCVValEngine,
        evaluate_purged_cv_pool_admission_batch,
    )

    use_cv_admission = (
        isinstance(engine, PurgedCVTrainEngine)
        and isinstance(val_engine, PurgedCVValEngine)
    )

    # Deduplicate and filter by condition count before any expensive backtest.
    unique_chroms: list[np.ndarray] = []
    seen: set[tuple] = set()
    for chrom in archive:
        key = chromosome_key(chrom)
        if key in seen:
            continue
        seen.add(key)
        active = _count_active_conditions(chrom, dont_cares)
        if active < _cfg.MIN_CONDITIONS or active > _cfg.MAX_CONDITIONS:
            continue
        unique_chroms.append(chrom)

    if not unique_chroms:
        return []

    logger.info(
        "Phase 2 [%s] pool builder: %d unique eligible chromosomes from %d archive entries",
        direction, len(unique_chroms), len(archive),
    )

    pool: list[dict] = []

    if use_cv_admission:
        # --- Batched CV path: one simulate_rule_batch call per fold ---
        cv_folds_total = len(engine._fold_engines)
        chroms_arr = np.stack(unique_chroms, axis=0)
        batch_results = evaluate_purged_cv_pool_admission_batch(
            engine, val_engine, chroms_arr, direction=direction,
        )

        if regime_row_fractions_arr is None:
            regime_row_fractions_arr = getattr(engine, "_regime_row_fractions", None)

        for chrom, (admitted, metrics, val_metrics, folds_passing) in zip(
            unique_chroms, batch_results
        ):
            if not admitted or not metrics:
                continue

            executed = int(metrics.get("executed_trades", 0))
            if not passes_pool_trade_floor(
                executed, metrics, regime_row_fractions_arr=regime_row_fractions_arr,
            ):
                continue

            try:
                dense_chrom = _chromosome_for_pool_export(chrom, dont_cares)
                conditions = decode_chromosome(dense_chrom, feature_infos)
            except Exception:
                continue
            if not conditions:
                continue

            pool_entry: dict = {
                "chromosome": dense_chrom.tolist(),
                "conditions": conditions,
                "objectives": {
                    "sortino_ratio": float(metrics.get("sortino_ratio", metrics.get("total_return_pct", 0.0))),
                    "total_return_pct": float(metrics.get("total_return_pct", 0.0)),
                    "profit_factor": float(metrics.get("profit_factor", 0.0)),
                    "max_drawdown_pct": float(metrics.get("max_drawdown_pct", 0.0)),
                    "win_rate": float(metrics.get("win_rate", 0.0)),
                },
                "executed_trades": executed,
                "cv_folds_passing": int(folds_passing),
                "cv_folds_total": int(cv_folds_total),
            }
            if val_metrics is not None:
                pool_entry["val_objectives"] = {
                    "sortino_ratio": float(val_metrics.get(
                        "sortino_ratio", val_metrics.get("total_return_pct", 0.0))),
                    "total_return_pct": float(val_metrics.get("total_return_pct", 0.0)),
                    "profit_factor": float(val_metrics.get("profit_factor", 0.0)),
                    "max_drawdown_pct": float(val_metrics.get("max_drawdown_pct", 0.0)),
                    "win_rate": float(val_metrics.get("win_rate", 0.0)),
                }
                pool_entry["val_executed_trades"] = int(
                    val_metrics.get("executed_trades", 0))
            if metrics.get("regime_specialist"):
                pool_entry["regime_specialist"] = True
                pool_entry["dominant_regime"] = int(metrics.get("dominant_regime", -1))
            if metrics.get("regime_trade_counts") is not None:
                pool_entry["regime_trade_counts"] = list(metrics["regime_trade_counts"])
            pool.append(pool_entry)

        target_min = int(_cfg.PHASE2_CV_POOL_TARGET_MIN)
        if len(pool) < target_min:
            rank_min_folds = int(_cfg.PHASE2_CV_RANK_MIN_FOLDS_PASS)
            rank_candidates: list[tuple[float,
                                        np.ndarray, dict, dict | None, int]] = []
            for chrom, (admitted, metrics, val_metrics, folds_passing) in zip(
                unique_chroms, batch_results
            ):
                if admitted or not metrics:
                    continue
                if int(folds_passing) < rank_min_folds:
                    continue
                executed = int(metrics.get("executed_trades", 0))
                if not passes_pool_trade_floor(
                    executed,
                    metrics,
                    regime_row_fractions_arr=regime_row_fractions_arr,
                ):
                    continue
                val_ret = 0.0
                if val_metrics is not None:
                    val_ret = float(val_metrics.get("total_return_pct", 0.0))
                rank_candidates.append(
                    (val_ret, chrom, metrics, val_metrics, int(folds_passing))
                )
            rank_candidates.sort(key=lambda row: row[0], reverse=True)
            seen_pool = {tuple(e["chromosome"]) for e in pool}
            top_k = int(_cfg.PHASE2_CV_POOL_RANK_ADMIT_TOP_K)
            added = 0
            for val_ret, chrom, metrics, val_metrics, folds_passing in rank_candidates[:top_k]:
                if (
                    bool(_cfg.PHASE2_CV_MERGED_GATE_HARD)
                    and val_metrics is not None
                    and not passes_pool_admission_gate(metrics, val_metrics)
                ):
                    continue
                key = chromosome_key(chrom)
                if key in seen_pool:
                    continue
                try:
                    dense_chrom = _chromosome_for_pool_export(
                        chrom, dont_cares)
                    conditions = decode_chromosome(dense_chrom, feature_infos)
                except Exception:
                    continue
                if not conditions:
                    continue
                executed = int(metrics.get("executed_trades", 0))
                pool_entry = {
                    "chromosome": dense_chrom.tolist(),
                    "conditions": conditions,
                    "objectives": {
                        "sortino_ratio": float(metrics.get(
                            "sortino_ratio",
                            metrics.get("total_return_pct", 0.0),
                        )),
                        "total_return_pct": float(metrics.get("total_return_pct", 0.0)),
                        "profit_factor": float(metrics.get("profit_factor", 0.0)),
                        "max_drawdown_pct": float(metrics.get("max_drawdown_pct", 0.0)),
                        "win_rate": float(metrics.get("win_rate", 0.0)),
                    },
                    "executed_trades": executed,
                    "cv_folds_passing": folds_passing,
                    "cv_folds_total": int(cv_folds_total),
                    "rank_fallback": True,
                }
                if val_metrics is not None:
                    pool_entry["val_objectives"] = {
                        "sortino_ratio": float(val_metrics.get(
                            "sortino_ratio",
                            val_metrics.get("total_return_pct", 0.0),
                        )),
                        "total_return_pct": float(val_metrics.get("total_return_pct", 0.0)),
                        "profit_factor": float(val_metrics.get("profit_factor", 0.0)),
                        "max_drawdown_pct": float(val_metrics.get("max_drawdown_pct", 0.0)),
                        "win_rate": float(val_metrics.get("win_rate", 0.0)),
                    }
                    pool_entry["val_executed_trades"] = int(
                        val_metrics.get("executed_trades", 0))
                pool.append(pool_entry)
                seen_pool.add(key)
                added += 1
                if len(pool) >= target_min:
                    break
            if added:
                logger.info(
                    "Phase 2 [%s] CV rank fallback: added %d rules "
                    "(pool %d → target %d)",
                    direction,
                    added,
                    len(pool) - added,
                    target_min,
                )

    else:
        # --- Non-CV path: one chromosome at a time (unchanged) ---
        if regime_row_fractions_arr is None:
            regime_row_fractions_arr = getattr(engine, "_regime_row_fractions", None)

        chrom_keys = [chromosome_key(c) for c in unique_chroms]
        for chrom, key in zip(unique_chroms, chrom_keys):
            metrics = None
            if metrics_by_chrom is not None:
                metrics = metrics_by_chrom.get(key)
            if metrics is None:
                try:
                    metrics_list = engine.simulate_rule_batch(
                        chromosomes=_chromosome_batch(chrom),
                        tp=_cfg.PHASE2_TP,
                        sl=_cfg.PHASE2_SL,
                        capital_pct=_cfg.PHASE2_CAPITAL_PCT,
                    )
                    metrics = metrics_list[0]
                except Exception:
                    continue

            val_metrics = _simulate_val_metrics_for_chrom(chrom, val_engine)
            if not passes_pool_admission_gate(metrics, val_metrics):
                continue

            executed = int(metrics.get("executed_trades", 0))
            if not passes_pool_trade_floor(
                executed, metrics, regime_row_fractions_arr=regime_row_fractions_arr,
            ):
                continue

            try:
                dense_chrom = _chromosome_for_pool_export(chrom, dont_cares)
                conditions = decode_chromosome(dense_chrom, feature_infos)
            except Exception:
                continue
            if not conditions:
                continue

            pool_entry = {
                "chromosome": dense_chrom.tolist(),
                "conditions": conditions,
                "objectives": {
                    "sortino_ratio": float(metrics.get("sortino_ratio", metrics.get("total_return_pct", 0.0))),
                    "total_return_pct": float(metrics.get("total_return_pct", 0.0)),
                    "profit_factor": float(metrics.get("profit_factor", 0.0)),
                    "max_drawdown_pct": float(metrics.get("max_drawdown_pct", 0.0)),
                    "win_rate": float(metrics.get("win_rate", 0.0)),
                },
                "executed_trades": executed,
            }
            if val_metrics is not None:
                pool_entry["val_objectives"] = {
                    "sortino_ratio": float(val_metrics.get(
                        "sortino_ratio", val_metrics.get("total_return_pct", 0.0))),
                    "total_return_pct": float(val_metrics.get("total_return_pct", 0.0)),
                    "profit_factor": float(val_metrics.get("profit_factor", 0.0)),
                    "max_drawdown_pct": float(val_metrics.get("max_drawdown_pct", 0.0)),
                    "win_rate": float(val_metrics.get("win_rate", 0.0)),
                }
                pool_entry["val_executed_trades"] = int(
                    val_metrics.get("executed_trades", 0))
            if metrics.get("regime_specialist"):
                pool_entry["regime_specialist"] = True
                pool_entry["dominant_regime"] = int(metrics.get("dominant_regime", -1))
            if metrics.get("regime_trade_counts") is not None:
                pool_entry["regime_trade_counts"] = list(metrics["regime_trade_counts"])
            pool.append(pool_entry)

    return pool


def _archive_feature_signature(feature_infos: list[dict]) -> list[dict[str, str]]:
    """Return the ordered feature signature used to validate archive reuse."""
    return [
        {"name": fi["name"], "mode": fi["mode"]}
        for fi in feature_infos
    ]


def _read_json_payload(path: str) -> object | None:
    """Read JSON from *path* and return None when the file cannot be loaded."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def _archive_objective_vector(entry: dict) -> np.ndarray:
    """Convert an archive entry into minimisation objectives for ranking."""
    from gpu_fuzzy_trader.phases.phase2_support import deployability_rank_score

    objectives = entry.get("objectives", {}) or {}
    val_obj = entry.get("val_objectives")
    val_metrics = None
    if isinstance(val_obj, dict):
        val_metrics = {
            "total_return_pct": float(val_obj.get("total_return_pct", 0.0)),
            "profit_factor": float(val_obj.get("profit_factor", 1.0)),
            "executed_trades": int(entry.get("val_executed_trades", 0)),
            "sortino_ratio": float(
                val_obj.get("sortino_ratio", val_obj.get(
                    "total_return_pct", 0.0))
            ),
            "max_drawdown_pct": float(val_obj.get("max_drawdown_pct", 0.0)),
        }
    train_metrics = {
        "total_return_pct": float(objectives.get("total_return_pct", 0.0)),
        "profit_factor": float(objectives.get("profit_factor", 1.0)),
        "executed_trades": int(entry.get("executed_trades", 0)),
        "sortino_ratio": float(
            objectives.get("sortino_ratio", objectives.get(
                "total_return_pct", 0.0))
        ),
        "max_drawdown_pct": float(objectives.get("max_drawdown_pct", 0.0)),
    }
    rank = deployability_rank_score(
        train_metrics,
        val_metrics,
        folds_passing=int(entry.get("cv_folds_passing", 0)),
    )
    return np.array(
        [
            -rank,
            float(objectives.get("max_drawdown_pct", 0.0)),
            -float(objectives.get("total_return_pct", 0.0)),
        ],
        dtype=np.float64,
    )


def _is_better_archive_entry(candidate: dict, incumbent: dict) -> bool:
    """Return True when *candidate* should replace *incumbent* for the same chromosome."""
    candidate_vec = _archive_objective_vector(candidate)
    incumbent_vec = _archive_objective_vector(incumbent)
    if _dominates(candidate_vec, incumbent_vec):
        return True
    if _dominates(incumbent_vec, candidate_vec):
        return False
    return tuple(candidate_vec.tolist()) < tuple(incumbent_vec.tolist())


def _pool_entry_passes_admission(entry: dict) -> bool:
    """Check stored train/val metrics on a pool JSON entry."""
    from gpu_fuzzy_trader.phases.phase2_support import passes_pool_entry_admission

    return passes_pool_entry_admission(entry)


def _filter_pool_by_admission(pool: list[dict]) -> list[dict]:
    if not _cfg.PHASE2_POOL_REQUIRE_POSITIVE_SPLITS:
        return pool
    return [e for e in pool if _pool_entry_passes_admission(e)]


def _merge_archive_entries(
    entries: list[dict],
    max_size: int = _cfg.PHASE2_ARCHIVE_MAX_SIZE,
) -> list[dict]:
    """Deduplicate and rank archive entries, keeping the best *max_size* rules."""
    if not entries:
        return []

    deduped: dict[tuple, dict] = {}
    for entry in entries:
        key = tuple(entry["chromosome"])
        current = deduped.get(key)
        if current is None or _is_better_archive_entry(entry, current):
            deduped[key] = entry

    unique_entries = list(deduped.values())
    if not unique_entries:
        return []

    objectives = np.vstack([
        _archive_objective_vector(entry) for entry in unique_entries
    ])
    from gpu_fuzzy_trader.evolution.numba_ops import (
        crowding_distance as _crowding_distance_fast,
        non_dominated_sort as _non_dominated_sort_fast,
    )

    fronts = _non_dominated_sort_fast(objectives)

    selected: list[int] = []
    for front in fronts:
        if not front:
            continue
        if len(selected) + len(front) <= max_size:
            selected.extend(front)
        else:
            crowding = _crowding_distance_fast(objectives, front)
            order = np.argsort(-crowding)
            need = max_size - len(selected)
            selected.extend(int(front[j]) for j in order[:need])
            break

    return [unique_entries[i] for i in selected[:max_size]]


def _stack_chromosome_rows(rows: list[np.ndarray]) -> np.ndarray:
    """Stack chromosome rows into a batch (dense 2D or sparse 3D)."""
    if not rows:
        raise ValueError("rows must be non-empty")
    first = rows[0]
    if use_sparse_slots() or is_sparse_chromosome(first):
        return np.stack(rows, axis=0)
    return np.vstack(rows)


def _pool_seed_chromosomes(
    pool: list[dict],
    dont_cares: np.ndarray | None = None,
) -> np.ndarray | None:
    """Extract deduplicated chromosomes from a Phase 2 pool for population seeding."""
    if not pool:
        return None

    rows: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    for entry in pool:
        chrom = entry.get("chromosome")
        if not isinstance(chrom, list) or not chrom:
            continue
        chrom_arr = np.asarray(chrom, dtype=np.int32)
        if use_sparse_slots() and dont_cares is not None:
            chrom_arr = dense_to_sparse(chrom_arr, dont_cares)
        key = chromosome_key(chrom_arr)
        if key in seen:
            continue
        seen.add(key)
        rows.append(chrom_arr)

    if not rows:
        return None
    return _stack_chromosome_rows(rows)


def _stage_b_seed_chromosomes(
    stage_a_pool: list[dict],
    base_seeds: np.ndarray | None,
    dont_cares: np.ndarray | None,
    top_k: int,
) -> np.ndarray | None:
    """Pick top Stage A deployable rules and merge with optional base seeds."""
    if not stage_a_pool and base_seeds is None:
        return None

    ranked = _merge_archive_entries(stage_a_pool, max_size=max(1, int(top_k)))
    stage_seeds = _pool_seed_chromosomes(ranked, dont_cares)
    if base_seeds is None:
        return stage_seeds
    if stage_seeds is None:
        return base_seeds

    from gpu_fuzzy_trader.phases.phase2_sparse_encoding import chromosome_key

    rows: list[np.ndarray] = [row.copy() for row in stage_seeds]
    seen = {chromosome_key(row) for row in rows}
    for row in base_seeds:
        key = chromosome_key(row)
        if key in seen:
            continue
        seen.add(key)
        rows.append(row.copy())
    return _stack_chromosome_rows(rows)


def _condition_feature_names(conditions: list[str]) -> set[str]:
    """
    Extract feature names from textual conditions like: "[feat] IS Value".
    Returns an empty set when parsing fails for a condition.
    """
    names: set[str] = set()
    for cond in conditions:
        if not isinstance(cond, str):
            continue
        left = cond.split(" IS ", 1)[0].strip()
        if left.startswith("[") and left.endswith("]") and len(left) >= 3:
            names.add(left[1:-1].strip())
    return names


def _filter_compatible_previous_pool(
    pool: list[dict],
    feature_infos: list[dict],
) -> list[dict]:
    """
    Keep only previous pool entries compatible with current feature selection.

    Compatibility rules:
      - chromosome length matches current feature count
      - condition feature names are a subset of current feature names
    """
    if not pool:
        return []
    feature_names = {fi["name"] for fi in feature_infos}
    expected_k = len(feature_infos)
    filtered: list[dict] = []

    for entry in pool:
        chrom = entry.get("chromosome")
        conditions = entry.get("conditions")
        if not isinstance(chrom, list) or len(chrom) != expected_k:
            continue
        if not isinstance(conditions, list):
            continue
        cond_features = _condition_feature_names(conditions)
        if not cond_features.issubset(feature_names):
            continue
        if not _pool_entry_passes_admission(entry):
            continue
        filtered.append(entry)
    return filtered


def _validate_archive_payload(
    payload: object,
    path: str,
    feature_infos: list[dict],
) -> None:
    """Validate the archive JSON structure and feature compatibility."""
    if not isinstance(payload, dict):
        raise ValueError(
            f"Phase 2 archive must be a JSON object, got {type(payload).__name__}: {path}"
        )

    required_keys = {"version", "direction", "feature_signature", "rules"}
    missing = required_keys - set(payload.keys())
    if missing:
        raise ValueError(f"Phase 2 archive missing keys {missing}: {path}")

    if payload["direction"] not in ("long", "short"):
        raise ValueError(
            f"Phase 2 archive has invalid direction {payload['direction']!r}: {path}"
        )

    expected_signature = _archive_feature_signature(feature_infos)
    if payload["feature_signature"] != expected_signature:
        raise ValueError(f"Phase 2 archive feature signature mismatch: {path}")

    if not isinstance(payload["rules"], list):
        raise ValueError(f"Phase 2 archive 'rules' must be a list: {path}")

    _validate_pool_schema(payload["rules"], path)

    dont_cares = _get_dont_cares(feature_infos)
    for i, entry in enumerate(payload["rules"]):
        chromosome = entry["chromosome"]
        if len(chromosome) != len(feature_infos):
            raise ValueError(
                f"Phase 2 archive entry {i} chromosome length mismatch: {path}"
            )
        for j, gene in enumerate(chromosome):
            if not isinstance(gene, (int, np.integer)):
                raise ValueError(
                    f"Phase 2 archive entry {i} gene {j} must be an int: {path}"
                )
            if gene < 0 or gene > int(dont_cares[j]):
                raise ValueError(
                    f"Phase 2 archive entry {i} gene {j} out of range: {path}"
                )


# ---------------------------------------------------------------------------
# EvoX-compatible optimizer (optional)
# ---------------------------------------------------------------------------

if _EVOX_AVAILABLE:
    class FuzzyRuleOptimizer(EvoxAlgorithm):  # type: ignore[misc]
        """
        EvoX-compatible multi-objective optimizer for fuzzy rule evolution.

        Wraps the NSGA-II fallback as an EvoX Algorithm for interface
        compatibility. When EvoX is available, this class can be used
        directly with EvoX workflows.
        """

        def __init__(
            self,
            feature_infos: list[dict],
            engine,
            pop_size: int = _cfg.PHASE2_POPULATION_SIZE,
            n_generations: int = _cfg.PHASE2_GENERATIONS,
            seed: int | None = None,
        ):
            self.feature_infos = feature_infos
            self.engine = engine
            self.pop_size = pop_size
            self.n_generations = n_generations
            self.seed = seed
            self._rng = np.random.default_rng(seed)
            self._population: Optional[np.ndarray] = None
            self._objectives: Optional[np.ndarray] = None

        def setup(self, key=None):
            """Initialise population with dont_care sparsity."""
            self._population = _init_population(
                self.pop_size, self.feature_infos, self._rng
            )
            dont_cares = _get_dont_cares(self.feature_infos)
            self._objectives = np.full((self.pop_size, 3), np.inf)
            self._dont_cares = dont_cares
            return self

        def step(self, state=None):
            """Run one generation step."""
            if self._population is None:
                self.setup()
            # Evaluate
            for i in range(self.pop_size):
                if np.any(np.isinf(self._objectives[i])):
                    obj, _ = _evaluate_chromosome(
                        self._population[i], self._dont_cares, self.engine, []
                    )
                    self._objectives[i] = obj
            return self

# ---------------------------------------------------------------------------
# Rule_Pool_Generator
# ---------------------------------------------------------------------------


class Rule_Pool_Generator:
    """
    Phase 2: GPU-accelerated multi-objective evolutionary rule pool generation.

    Generates separate Pareto-front rule pools for long and short directions.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training split DataFrame (already prepared by Data_Loader + Data_Splitter).
    feature_infos : list[dict]
        Output of Feature_Selector.select_features(): list of
        {"name": str, "mode": str, "score": float}.
    direction : str
        "long" or "short".
    pop_size : int, optional
        Override PHASE2_POPULATION_SIZE (useful for testing).
    n_generations : int, optional
        Override PHASE2_GENERATIONS (useful for testing).
    seed : int | None, optional
        Random seed for reproducibility. If None, a fresh random seed is used
        for each run.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        feature_infos: list[dict],
        direction: str,
        pop_size: int | None = None,
        n_generations: int | None = None,
        seed: int | None = None,
        val_df: pd.DataFrame | None = None,
        cv_folds: list | None = None,
        symbol_scope: str | None = None,
    ) -> None:
        if direction not in ("long", "short"):
            raise ValueError(
                f"direction must be 'long' or 'short', got {direction!r}")
        if not feature_infos:
            raise ValueError("feature_infos must not be empty.")

        self.direction = direction
        self.symbol_scope = str(symbol_scope) if symbol_scope is not None else None
        self.feature_infos = feature_infos
        self.pop_size = pop_size if pop_size is not None else _cfg.PHASE2_POPULATION_SIZE
        self.n_generations = n_generations if n_generations is not None else _cfg.PHASE2_GENERATIONS
        self.seed = seed  # preserved as-is (None when not provided, per docstring)
        self._feature_signature = _archive_feature_signature(feature_infos)
        self._regime_row_fractions: np.ndarray | None = None
        self._n_regimes = 0
        self._val_regime_row_counts: np.ndarray | None = None
        self._evolution_state = None
        self._island_history: list[dict] = []

        scoped_train_df = train_df
        scoped_val_df = val_df
        if self.symbol_scope is not None:
            scoped_train_df = _cfg.filter_df_to_symbol(
                train_df, self.symbol_scope)
            if val_df is not None:
                scoped_val_df = _cfg.filter_df_to_symbol(
                    val_df, self.symbol_scope)

        # Sample training data to budget, then slim to backtest-only columns
        sample_seed = seed if seed is not None else _cfg.PHASE2_SEED
        sampled = _sample_df(
            scoped_train_df, _cfg.PHASE1_SAMPLING_TOTAL, random_state=sample_seed,
        )
        feature_names = [fi["name"] for fi in feature_infos]
        from gpu_fuzzy_trader.backtest.df_slim import slim_backtest_df

        train_regime_ids, self._regime_row_fractions, self._n_regimes = (
            _prepare_regime_context(sampled)
        )
        self._train_df = slim_backtest_df(sampled, feature_names)
        self._train_regime_ids = train_regime_ids
        self._feature_names = feature_names
        self._cv_folds = cv_folds
        self._scoped_val_df = scoped_val_df
        self._sample_seed = sample_seed

        # Build feature_modes dict for engine
        self._feature_modes = {fi["name"]: fi["mode"] for fi in feature_infos}

        self._engine = None
        self._val_engine = None
        self._val_regime_row_counts = None
        self._build_engines()

    # ------------------------------------------------------------------
    # Engine construction
    # ------------------------------------------------------------------

    def _uses_cv_engines(self) -> bool:
        return bool(
            self._cv_folds
            and len(self._cv_folds) > 0
            and str(_cfg.SPLIT_MODE).strip().lower() == "purged_rolling_cv"
        )

    def _build_engines(self) -> None:
        """Build train/val (or CV facade) backtest engines."""
        if self._uses_cv_engines():
            from gpu_fuzzy_trader.phases.phase2_cv import build_cv_fold_engines

            cv_train, cv_val = build_cv_fold_engines(
                self._cv_folds,
                self.feature_infos,
                self.direction,
                seed=self.seed,
                builder=self,
            )
            self._engine = cv_train
            self._val_engine = cv_val
            if cv_val is not None:
                self._val_regime_row_counts = getattr(
                    cv_val, "_regime_row_counts", None)
        else:
            self._engine = self._build_engine(
                regime_ids=self._train_regime_ids,
                n_regimes=self._n_regimes,
            )
            self._val_engine = None
            if self._scoped_val_df is not None and _cfg.PHASE2_JOINT_TRAIN_VAL:
                try:
                    val_sampled = _sample_df(
                        self._scoped_val_df,
                        _cfg.PHASE1_SAMPLING_TOTAL,
                        random_state=self._sample_seed,
                    )
                    val_regime_ids, _val_fracs, val_n_regimes = (
                        _prepare_regime_context(val_sampled)
                    )
                    if val_regime_ids is not None:
                        self._val_regime_row_counts = np.bincount(
                            val_regime_ids.astype(np.int64),
                            minlength=val_n_regimes,
                        ).astype(np.int64)
                    slim_val = slim_backtest_df(val_sampled, self._feature_names)
                    self._val_engine = self._build_engine_for_df(
                        slim_val,
                        regime_ids=val_regime_ids,
                        n_regimes=val_n_regimes,
                    )
                    if self._val_regime_row_counts is not None:
                        self._val_engine._regime_row_counts = (
                            self._val_regime_row_counts
                        )
                    logger.info(
                        "Phase 2 [%s]: joint train+val objective enabled "
                        "(val_rows=%d)",
                        self.direction,
                        len(slim_val),
                    )
                except Exception as exc:
                    logger.warning(
                        "Phase 2 [%s]: failed to build val engine, "
                        "falling back to train-only objective: %s",
                        self.direction,
                        exc,
                    )
                    self._val_engine = None

        from gpu_fuzzy_trader._gpu_runtime import configure_phase2_gpu_runtime
        from gpu_fuzzy_trader._memory import log_memory_rss

        configure_phase2_gpu_runtime(self._engine, val_engine=self._val_engine)
        log_memory_rss(f"Phase2 [{self.direction}] engine init")

    def _ensure_engines(self) -> None:
        """Rebuild engines after ``park_engines`` dropped GPU state."""
        if self._engine is not None:
            return
        self._build_engines()

    def park_engines(self) -> None:
        """Release GPU engines between island scheduler epochs."""
        self._engine = None
        self._val_engine = None
        from gpu_fuzzy_trader._memory import log_memory_rss, release_phase2_resources

        log_memory_rss(
            f"Phase2 [{self.direction}]"
            f"{f'/{self.symbol_scope}' if self.symbol_scope else ''} parked",
        )
        release_phase2_resources()

    def _build_engine(
        self,
        regime_ids: np.ndarray | None = None,
        n_regimes: int = 0,
    ):
        """Build GPUBacktestEngine if JAX available, else CPUBacktestEngine."""
        return self._build_engine_for_df(
            self._train_df,
            regime_ids=regime_ids,
            n_regimes=n_regimes,
        )

    def _build_engine_for_df(
        self,
        df: pd.DataFrame,
        regime_ids: np.ndarray | None = None,
        n_regimes: int = 0,
    ):
        """Build an engine on *df* using the same backend selection logic."""
        engine_kwargs: dict = {}
        if regime_ids is not None and n_regimes > 0:
            engine_kwargs["regime_ids"] = regime_ids
            engine_kwargs["n_regimes"] = n_regimes
        engine_kwargs["fee_pct"] = _cfg.FEE_PCT

        from gpu_fuzzy_trader.backtest.cpu_engine import CPUBacktestEngine

        if _cfg.PHASE2_USE_GPU:
            from gpu_fuzzy_trader.backtest.jax_compat import (
                get_gpu_backtest_engine_class,
            )

            GPUBacktestEngine = get_gpu_backtest_engine_class()
            if GPUBacktestEngine is not None:
                engine = GPUBacktestEngine(
                    df,
                    self._feature_modes,
                    self.direction,
                    **engine_kwargs,
                )
                if self._regime_row_fractions is not None:
                    engine._regime_row_fractions = self._regime_row_fractions
                symbol_scope = getattr(self, "symbol_scope", None)
                if symbol_scope is not None:
                    engine._symbol_scope = symbol_scope
                logger.info(
                    "Phase 2 using GPUBacktestEngine (backend: %s)",
                    engine.backend,
                )
                return engine
            logger.warning(
                "PHASE2_USE_GPU=True but JAX/GPU backtest unavailable; "
                "falling back to CPUBacktestEngine for Phase 2.",
            )
        else:
            logger.info(
                "Phase 2 using CPUBacktestEngine (PHASE2_USE_GPU=False).")

        engine = CPUBacktestEngine(
            df,
            self._feature_modes,
            self.direction,
            **engine_kwargs,
        )
        if self._regime_row_fractions is not None:
            engine._regime_row_fractions = self._regime_row_fractions
        symbol_scope = getattr(self, "symbol_scope", None)
        if symbol_scope is not None:
            engine._symbol_scope = symbol_scope
        return engine

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> list[dict]:
        """
        Run the evolutionary search and return the Pareto-front pool.

        Also persists results to:
                    pools/phase2_{direction}_pool.json
                    pools/phase2_{direction}_history.json

        Returns
        -------
        list[dict]
            Pareto-front rules in pool JSON schema.
        """
        rng = np.random.default_rng(self.seed)

        logger.info(
            "Phase 2 [%s]: pop=%d, gen=%d, features=%d",
            self.direction, self.pop_size, self.n_generations, len(
                self.feature_infos)
        )

        from gpu_fuzzy_trader.evolution.evox_runner import run_phase2_evolution

        logger.info(
            "Phase 2 [%s]: algorithm=%s",
            self.direction, _cfg.PHASE2_ALGORITHM,
        )

        previous_pool: list[dict] = []
        try:
            loaded_pool = Rule_Pool_Generator.load_pool(
                self.direction, self.symbol_scope,
            )
            if loaded_pool:
                previous_pool = loaded_pool
        except ValueError:
            logger.warning(
                "Phase 2 [%s]: existing pool file invalid; starting without seeds",
                self.direction,
            )

        if previous_pool:
            compatible_pool = _filter_compatible_previous_pool(
                previous_pool,
                self.feature_infos,
            )
            dropped = len(previous_pool) - len(compatible_pool)
            if dropped > 0:
                logger.info(
                    "Phase 2 [%s]: dropped %d incompatible previous pool rules "
                    "(feature count/signature mismatch)",
                    self.direction,
                    dropped,
                )
            previous_pool = compatible_pool

        dont_cares = _get_dont_cares(self.feature_infos)
        seed_chromosomes = _pool_seed_chromosomes(previous_pool, dont_cares)
        if seed_chromosomes is not None:
            seed_slots = min(
                self.pop_size,
                max(1, int(round(
                    self.pop_size * _cfg.PHASE2_ARCHIVE_SEED_FRACTION))),
                len(seed_chromosomes),
            )
            logger.info(
                "Phase 2 [%s]: seeding %d/%d population slots (%.0f%%) from "
                "pool with %d unique chromosomes",
                self.direction,
                seed_slots,
                self.pop_size,
                _cfg.PHASE2_ARCHIVE_SEED_FRACTION * 100.0,
                len(seed_chromosomes),
            )

        from gpu_fuzzy_trader.phases.phase2_init import (
            build_feature_sampling_probs,
        )

        feature_probs = build_feature_sampling_probs(self.feature_infos)

        progress_tag = "Phase 2 [%s] NSGA-III" % self.direction
        use_two_stage = (
            bool(getattr(_cfg, "PHASE2_TWO_STAGE_ENABLED", False))
            and self.n_generations == _cfg.PHASE2_GENERATIONS
            and self.pop_size == _cfg.PHASE2_POPULATION_SIZE
        )

        evo_kwargs = dict(
            feature_infos=self.feature_infos,
            engine=self._engine,
            pop_size=self.pop_size,
            rng=rng,
            seed_chromosomes=seed_chromosomes,
            val_engine=self._val_engine,
            regime_row_fractions=self._regime_row_fractions,
            val_regime_row_counts=self._val_regime_row_counts,
            feature_probs=feature_probs,
            init_strategy=_cfg.PHASE2_INIT_STRATEGY,
            stratum_fractions=_cfg.PHASE2_INIT_STRATUM_FRACTIONS,
        )

        if use_two_stage:
            stage_a_gens = int(_cfg.PHASE2_STAGE_A_GENERATIONS)
            stage_b_gens = int(_cfg.PHASE2_STAGE_B_GENERATIONS)
            stage_b_top_k = int(_cfg.PHASE2_STAGE_B_SEED_TOP_K)
            logger.info(
                "Phase 2 [%s]: two-stage search enabled "
                "(Stage A=%d gen, Stage B=%d gen, seed_top_k=%d)",
                self.direction,
                stage_a_gens,
                stage_b_gens,
                stage_b_top_k,
            )
            new_pool_a, history_a = run_phase2_evolution(
                n_generations=stage_a_gens,
                log_tag=f"{progress_tag} Stage A",
                stage="A",
                **evo_kwargs,
            )
            stage_b_seeds = _stage_b_seed_chromosomes(
                list(new_pool_a),
                seed_chromosomes,
                dont_cares,
                stage_b_top_k,
            )
            if stage_b_seeds is not None:
                logger.info(
                    "Phase 2 [%s]: Stage B seeding from %d chromosomes "
                    "(top %d Stage A + archive seeds)",
                    self.direction,
                    stage_b_seeds.shape[0],
                    stage_b_top_k,
                )
            new_pool_b, history_b = run_phase2_evolution(
                n_generations=stage_b_gens,
                log_tag=f"{progress_tag} Stage B",
                stage="B",
                seed_fraction=float(_cfg.PHASE2_STAGE_B_SEED_FRACTION),
                reset_plateau=True,
                **{**evo_kwargs, "seed_chromosomes": stage_b_seeds},
            )
            for entry in history_a:
                entry["stage"] = "A"
            for entry in history_b:
                entry["stage"] = "B"
            new_pool = list(new_pool_a) + list(new_pool_b)
            history = history_a + history_b
        else:
            new_pool, history = run_phase2_evolution(
                n_generations=self.n_generations,
                log_tag=progress_tag,
                **evo_kwargs,
            )

        pool = _merge_archive_entries(previous_pool + list(new_pool))
        pool_before_admission = len(pool)
        pool = _filter_pool_by_admission(pool)
        if pool_before_admission != len(pool):
            logger.info(
                "Phase 2 [%s]: pool admission filter %d → %d rules "
                "(train+val return>0, PF>=%.2f)",
                self.direction,
                pool_before_admission,
                len(pool),
                _cfg.PHASE2_PROFIT_FACTOR_FLOOR,
            )
        logger.info(
            "Phase 2 [%s]: merged pool %d previous + %d new → %d retained",
            self.direction,
            len(previous_pool),
            len(new_pool),
            len(pool),
        )

        pool_path = _resolve_pool_path(self.direction, self.symbol_scope)
        history_path = _resolve_history_path(self.direction, self.symbol_scope)
        pool_dir = os.path.dirname(pool_path)
        history_dir = os.path.dirname(history_path)
        if pool_dir:
            os.makedirs(pool_dir, exist_ok=True)
        if history_dir and history_dir != pool_dir:
            os.makedirs(history_dir, exist_ok=True)

        with open(pool_path, "w", encoding="utf-8") as fh:
            json.dump(pool, fh, indent=2)
        with open(history_path, "w", encoding="utf-8") as fh:
            json.dump(history, fh, indent=2)

        logger.info(
            "Phase 2 [%s]: pool_size=%d, saved to %s",
            self.direction, len(pool), pool_path
        )

        # Reporter: plot Phase 2 generation metrics
        try:
            reporter = Reporter()
            reporter.plot_phase2_metrics(history, self.direction)
            reporter.plot_phase2_pnl(history, self.direction)
        except Exception as exc:
            logger.warning(
                "Reporter Phase 2 plots failed (non-fatal): %s", exc)

        try:
            saved = Rule_Pool_Generator.save_archive(
                self.direction,
                self.feature_infos,
                pool,
                symbol_scope=self.symbol_scope,
            )
            archive_path = Rule_Pool_Generator._archive_path_for(
                self.direction, self.symbol_scope,
            )
            logger.info(
                "Phase 2 [%s]: archive saved with %d rules to %s",
                self.direction, len(saved), archive_path,
            )
        except Exception as exc:
            logger.warning(
                "Phase 2 [%s]: archive save failed (non-fatal): %s",
                self.direction, exc,
            )

        self._release_resources()
        return pool

    def _release_resources(self) -> None:
        """Drop engine and sampled data to free RAM before the next direction."""
        self._engine = None
        self._val_engine = None
        self._train_df = None
        from gpu_fuzzy_trader._memory import log_memory_rss, release_phase2_resources

        log_memory_rss(f"Phase2 [{self.direction}] after release")
        release_phase2_resources()

    @staticmethod
    def load_pool(
        direction: str,
        symbol: str | None = None,
    ) -> Optional[list[dict]]:
        """
        Load existing pool if valid, return None if missing.
        """
        if direction not in ("long", "short"):
            raise ValueError(
                f"direction must be 'long' or 'short', got {direction!r}")
        path = _resolve_pool_path(direction, symbol)
        if not os.path.exists(path):
            return None

        try:
            with open(path, "r", encoding="utf-8") as fh:
                pool = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(
                f"Phase 2 pool file is unreadable or corrupted: {path}"
            ) from exc

        _validate_pool_schema(pool, path)
        return pool

    @staticmethod
    def load_archive(
        direction: str,
        feature_infos: list[dict],
    ) -> Optional[dict]:
        """
        Load a compatible persistent archive if it exists, otherwise return None.

        Archive files are ignored when they are corrupt or were built from an
        incompatible feature signature.
        """
        if direction not in _ARCHIVE_PATHS:
            raise ValueError(
                f"direction must be 'long' or 'short', got {direction!r}")

        path = _ARCHIVE_PATHS[direction]
        if not os.path.exists(path):
            return None

        payload = _read_json_payload(path)
        if payload is None:
            logger.warning(
                "Phase 2 archive file is unreadable or corrupted: %s", path)
            return None

        try:
            _validate_archive_payload(payload, path, feature_infos)
        except ValueError as exc:
            logger.info("Ignoring Phase 2 archive at %s: %s", path, exc)
            return None

        rules = _merge_archive_entries(payload["rules"])
        return {
            "version": int(payload.get("version", 1)),
            "direction": direction,
            "feature_signature": _archive_feature_signature(feature_infos),
            "rules": rules,
        }

    @staticmethod
    def _archive_path_for(
        direction: str,
        symbol: str | None = None,
        *,
        shared: bool = False,
    ) -> str:
        if shared:
            return _cfg.phase2_shared_archive_path(direction)
        if symbol is not None:
            return _cfg.phase2_symbol_archive_path(direction, symbol)
        return _ARCHIVE_PATHS[direction]

    @staticmethod
    def _annotate_archive_entries(
        rules: list[dict],
        *,
        symbol_scope: str | None = None,
        shared_archive: bool = False,
        source_symbols: list[str] | None = None,
    ) -> list[dict]:
        annotated: list[dict] = []
        for entry in rules:
            row = dict(entry)
            if symbol_scope is not None:
                row["symbol_scope"] = symbol_scope
            if shared_archive:
                row["shared_archive"] = True
            if source_symbols:
                row["source_symbols"] = sorted(set(source_symbols))
            val_obj = row.get("val_objectives") or {}
            train_obj = row.get("objectives") or {}
            from gpu_fuzzy_trader.phases.phase2_support import compute_robust_score

            row["robust_score"] = compute_robust_score(
                {
                    "total_return_pct": float(train_obj.get("total_return_pct", 0.0)),
                    "profit_factor": float(train_obj.get("profit_factor", 1.0)),
                    "executed_trades": int(row.get("executed_trades", 0)),
                    "sortino_ratio": float(train_obj.get("sortino_ratio", 0.0)),
                    "max_drawdown_pct": float(train_obj.get("max_drawdown_pct", 0.0)),
                },
                {
                    "total_return_pct": float(val_obj.get("total_return_pct", 0.0)),
                    "profit_factor": float(val_obj.get("profit_factor", 1.0)),
                    "executed_trades": int(row.get("val_executed_trades", 0)),
                },
                source_symbols=source_symbols,
            )
            annotated.append(row)
        return annotated

    @staticmethod
    def load_local_symbol_archive(
        direction: str,
        symbol: str,
        feature_infos: list[dict],
    ) -> Optional[dict]:
        path = Rule_Pool_Generator._archive_path_for(direction, symbol)
        return Rule_Pool_Generator._load_archive_at(path, direction, feature_infos)

    @staticmethod
    def load_shared_archive(
        direction: str,
        feature_infos: list[dict],
    ) -> Optional[dict]:
        path = Rule_Pool_Generator._archive_path_for(direction, shared=True)
        return Rule_Pool_Generator._load_archive_at(path, direction, feature_infos)

    @staticmethod
    def _load_archive_at(
        path: str,
        direction: str,
        feature_infos: list[dict],
    ) -> Optional[dict]:
        if not os.path.exists(path):
            return None
        payload = _read_json_payload(path)
        if payload is None:
            return None
        try:
            _validate_archive_payload(payload, path, feature_infos)
        except ValueError as exc:
            logger.info("Ignoring Phase 2 archive at %s: %s", path, exc)
            return None
        rules = _merge_archive_entries(payload["rules"])
        return {
            "version": int(payload.get("version", 1)),
            "direction": direction,
            "feature_signature": _archive_feature_signature(feature_infos),
            "rules": rules,
        }

    @staticmethod
    def save_archive(
        direction: str,
        feature_infos: list[dict],
        rules: list[dict],
        symbol_scope: str | None = None,
        *,
        shared: bool = False,
        source_symbols: list[str] | None = None,
    ) -> list[dict]:
        """Merge the latest pool into a persistent archive and write atomically."""
        path = Rule_Pool_Generator._archive_path_for(
            direction, symbol_scope, shared=shared,
        )
        existing_rules: list[dict] = []
        raw_payload = _read_json_payload(path)
        if raw_payload is not None:
            try:
                _validate_archive_payload(raw_payload, path, feature_infos)
            except ValueError as exc:
                logger.info(
                    "Replacing invalid Phase 2 archive at %s: %s", path, exc)
            else:
                existing_rules = list(raw_payload["rules"])

        merged = _merge_archive_entries(existing_rules + list(rules))
        if not merged:
            return []

        merged = Rule_Pool_Generator._annotate_archive_entries(
            merged,
            symbol_scope=symbol_scope,
            shared_archive=shared,
            source_symbols=source_symbols,
        )
        payload = {
            "version": 1,
            "direction": direction,
            "feature_signature": _archive_feature_signature(feature_infos),
            "rules": merged,
        }
        if symbol_scope is not None:
            payload["symbol_scope"] = symbol_scope
        if shared:
            payload["shared_archive"] = True

        archive_dir = os.path.dirname(path)
        if archive_dir:
            os.makedirs(archive_dir, exist_ok=True)

        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp_path, path)
        return merged

    @staticmethod
    def collect_shared_archive_candidates(
        direction: str,
        feature_infos: list[dict],
        symbol_pools: dict[str, list[dict]],
    ) -> list[dict]:
        """Promote broadly robust rules into the direction-level shared archive."""
        from gpu_fuzzy_trader.phases.phase2_support import compute_robust_score

        by_chrom: dict[tuple, list[dict]] = {}
        for symbol, pool in symbol_pools.items():
            for entry in pool:
                chrom = entry.get("chromosome")
                if not isinstance(chrom, list):
                    continue
                key = tuple(int(v) for v in chrom)
                tagged = dict(entry)
                tagged["symbol_scope"] = symbol
                by_chrom.setdefault(key, []).append(tagged)

        promoted: list[dict] = []
        min_score = float(_cfg.PHASE2_SHARED_ARCHIVE_MIN_ROBUST_SCORE)
        min_symbols = int(_cfg.PHASE2_SHARED_ARCHIVE_MIN_SYMBOLS)
        for entries in by_chrom.values():
            passing: list[dict] = []
            for entry in entries:
                train_obj = entry.get("objectives") or {}
                val_obj = entry.get("val_objectives") or {}
                score = compute_robust_score(
                    {
                        "total_return_pct": float(
                            train_obj.get("total_return_pct", 0.0)),
                        "profit_factor": float(train_obj.get("profit_factor", 1.0)),
                        "executed_trades": int(entry.get("executed_trades", 0)),
                        "sortino_ratio": float(train_obj.get("sortino_ratio", 0.0)),
                        "max_drawdown_pct": float(
                            train_obj.get("max_drawdown_pct", 0.0)),
                    },
                    {
                        "total_return_pct": float(
                            val_obj.get("total_return_pct", 0.0)),
                        "profit_factor": float(val_obj.get("profit_factor", 1.0)),
                        "executed_trades": int(entry.get("val_executed_trades", 0)),
                    },
                    source_symbols=[str(entry.get("symbol_scope", ""))],
                )
                if score >= min_score:
                    row = dict(entry)
                    row["robust_score"] = score
                    passing.append(row)
            source_symbols = sorted({
                str(e.get("symbol_scope"))
                for e in passing
                if e.get("symbol_scope")
            })
            if len(source_symbols) < min_symbols:
                continue
            best = max(passing, key=lambda e: float(e.get("robust_score", 0.0)))
            best = dict(best)
            best["shared_archive"] = True
            best["source_symbols"] = source_symbols
            promoted.append(best)
        return _merge_archive_entries(promoted)

    def _assemble_epoch_seed_entries(self) -> list[dict]:
        """Merge local pool, symbol archive, and shared archive (dominant seeds)."""
        seeds: list[dict] = []
        local_pool = Rule_Pool_Generator.load_pool(
            self.direction, self.symbol_scope,
        ) or []
        seeds.extend(_filter_compatible_previous_pool(local_pool, self.feature_infos))

        if self.symbol_scope is not None:
            local_archive = Rule_Pool_Generator.load_local_symbol_archive(
                self.direction, self.symbol_scope, self.feature_infos,
            )
            if local_archive:
                seeds.extend(local_archive["rules"])
            shared_archive = Rule_Pool_Generator.load_shared_archive(
                self.direction, self.feature_infos,
            )
            if shared_archive:
                seeds.extend(shared_archive["rules"])
        return _merge_archive_entries(seeds)

    def validate_migrants_on_target(
        self,
        migrant_entries: list[dict],
    ) -> tuple[list[dict], list[dict]]:
        """Re-score migrants on this island and split accepted vs rejected."""
        from gpu_fuzzy_trader.phases.phase2_support import passes_migrant_target_gate

        dont_cares = _get_dont_cares(self.feature_infos)
        accepted: list[dict] = []
        rejected: list[dict] = []
        for entry in migrant_entries:
            chrom = np.asarray(entry["chromosome"], dtype=np.int32)
            if use_sparse_slots():
                chrom = dense_to_sparse(chrom, dont_cares)
            _, metrics = _evaluate_chromosome(
                chrom,
                dont_cares,
                self._engine,
                [],
                val_engine=self._val_engine,
                regime_row_fractions_arr=self._regime_row_fractions,
                val_regime_row_counts=self._val_regime_row_counts,
            )
            val_metrics = {
                "total_return_pct": float(metrics.get("val_total_return_pct", 0.0)),
                "profit_factor": float(metrics.get("val_profit_factor", 1.0)),
                "executed_trades": int(metrics.get("val_executed_trades", 0)),
                "max_drawdown_pct": float(metrics.get("val_max_drawdown_pct", 0.0)),
            }
            train_metrics = {
                "total_return_pct": float(metrics.get("total_return_pct", 0.0)),
                "profit_factor": float(metrics.get("profit_factor", 1.0)),
                "executed_trades": int(metrics.get("executed_trades", 0)),
                "max_drawdown_pct": float(metrics.get("max_drawdown_pct", 0.0)),
            }
            if passes_migrant_target_gate(train_metrics, val_metrics):
                accepted.append(entry)
            else:
                rejected.append({
                    **entry,
                    "rejection_reason": "target_gate_failed",
                })
        return accepted, rejected

    def run_epoch(
        self,
        n_generations: int | None = None,
        *,
        migrant_entries: list[dict] | None = None,
    ) -> list[dict]:
        """Evolve this island for one scheduler epoch."""
        self._ensure_engines()
        from gpu_fuzzy_trader.evolution.evox_runner import (
            extract_deployable_migrants,
            run_phase2_evolution_epoch,
        )
        from gpu_fuzzy_trader.phases.phase2_init import build_feature_sampling_probs

        epoch_gens = int(
            n_generations if n_generations is not None
            else _cfg.PHASE2_ISLAND_EPOCH_GENERATIONS
        )
        seed_entries = self._assemble_epoch_seed_entries()
        dont_cares = _get_dont_cares(self.feature_infos)
        migrant_fraction = float(_cfg.PHASE2_MIGRATION_SEED_FRACTION)
        local_fraction = max(0.0, 1.0 - migrant_fraction)
        local_cap = max(1, int(round(self.pop_size * local_fraction)))
        migrant_cap = max(0, self.pop_size - local_cap)
        local_seeds = seed_entries[:local_cap]
        migrant_seeds = (
            migrant_entries[:migrant_cap] if migrant_entries else []
        )
        seed_chromosomes = _pool_seed_chromosomes(local_seeds + migrant_seeds, dont_cares)
        rng = np.random.default_rng(self.seed)
        feature_probs = build_feature_sampling_probs(self.feature_infos)
        tag = f"Phase 2 [{self.direction}"
        if self.symbol_scope:
            tag += f"/{self.symbol_scope}"
        tag += "]"

        self._evolution_state, epoch_history = run_phase2_evolution_epoch(
            feature_infos=self.feature_infos,
            engine=self._engine,
            pop_size=self.pop_size,
            n_generations=epoch_gens,
            rng=rng,
            state=self._evolution_state,
            seed_chromosomes=seed_chromosomes,
            log_tag=tag,
            val_engine=self._val_engine,
            regime_row_fractions=self._regime_row_fractions,
            val_regime_row_counts=self._val_regime_row_counts,
            feature_probs=feature_probs,
            init_strategy=_cfg.PHASE2_INIT_STRATEGY,
            stratum_fractions=_cfg.PHASE2_INIT_STRATUM_FRACTIONS,
            seed_fraction=migrant_fraction if migrant_entries else _cfg.PHASE2_ARCHIVE_SEED_FRACTION,
        )
        self._island_history.extend(epoch_history)
        return extract_deployable_migrants(self._evolution_state)

    def snapshot_migrants(self, top_k: int = 5) -> list[dict]:
        from gpu_fuzzy_trader.evolution.evox_runner import extract_deployable_migrants

        if self._evolution_state is None:
            return []
        return extract_deployable_migrants(self._evolution_state, top_k=top_k)

    def finalize_island(self) -> list[dict]:
        """Build, filter, and persist the final pool for this island."""
        from gpu_fuzzy_trader.evolution.evox_runner import run_phase2_evolution
        from gpu_fuzzy_trader.phases.phase2_init import build_feature_sampling_probs

        self._ensure_engines()
        rng = np.random.default_rng(self.seed)
        feature_probs = build_feature_sampling_probs(self.feature_infos)
        tag = f"Phase 2 [{self.direction}"
        if self.symbol_scope:
            tag += f"/{self.symbol_scope}"
        tag += "] finalize"

        if self._evolution_state is None:
            return self.run()

        result = run_phase2_evolution(
            feature_infos=self.feature_infos,
            engine=self._engine,
            pop_size=self.pop_size,
            n_generations=0,
            rng=rng,
            state=self._evolution_state,
            build_pool=True,
            return_state=False,
            log_tag=tag,
            val_engine=self._val_engine,
            regime_row_fractions=self._regime_row_fractions,
            val_regime_row_counts=self._val_regime_row_counts,
            feature_probs=feature_probs,
        )
        new_pool = result[0] if isinstance(result, tuple) else []
        previous_pool = Rule_Pool_Generator.load_pool(
            self.direction, self.symbol_scope,
        ) or []
        pool = _merge_archive_entries(previous_pool + list(new_pool))
        pool = _filter_pool_by_admission(pool)
        pool = Rule_Pool_Generator._annotate_archive_entries(
            pool, symbol_scope=self.symbol_scope,
        )

        pool_path = _resolve_pool_path(self.direction, self.symbol_scope)
        history_path = _resolve_history_path(self.direction, self.symbol_scope)
        for path in (pool_path, history_path):
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
        with open(pool_path, "w", encoding="utf-8") as fh:
            json.dump(pool, fh, indent=2)
        with open(history_path, "w", encoding="utf-8") as fh:
            json.dump(self._island_history, fh, indent=2)

        if self.symbol_scope is not None:
            Rule_Pool_Generator.save_archive(
                self.direction,
                self.feature_infos,
                pool,
                symbol_scope=self.symbol_scope,
            )
        else:
            Rule_Pool_Generator.save_archive(
                self.direction, self.feature_infos, pool,
            )
        self._release_resources()
        return pool

    @staticmethod
    def skip_if_valid(
        direction: str,
        symbol: str | None = None,
    ) -> Optional[list[dict]]:
        """Return loaded pool if valid, None if need to run."""
        try:
            return Rule_Pool_Generator.load_pool(direction, symbol)
        except ValueError:
            return None


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def _validate_pool_schema(pool: object, path: str) -> None:
    """
    Validate the structure of a loaded pool JSON.

    Raises ValueError if the schema is invalid.
    """
    if not isinstance(pool, list):
        raise ValueError(
            f"Phase 2 pool must be a JSON array, got {type(pool).__name__}: {path}"
        )

    for i, entry in enumerate(pool):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Phase 2 pool entry {i} must be a dict: {path}"
            )
        required_keys = {"chromosome", "conditions",
                         "objectives", "executed_trades"}
        missing = required_keys - set(entry.keys())
        if missing:
            raise ValueError(
                f"Phase 2 pool entry {i} missing keys {missing}: {path}"
            )
        if not isinstance(entry["chromosome"], list):
            raise ValueError(
                f"Phase 2 pool entry {i} 'chromosome' must be a list: {path}"
            )
        if not isinstance(entry["conditions"], list):
            raise ValueError(
                f"Phase 2 pool entry {i} 'conditions' must be a list: {path}"
            )
        if not isinstance(entry["objectives"], dict):
            raise ValueError(
                f"Phase 2 pool entry {i} 'objectives' must be a dict: {path}"
            )
        required_obj_keys = {"sortino_ratio",
                             "max_drawdown_pct", "win_rate"}
        missing_obj = required_obj_keys - set(entry["objectives"].keys())
        if missing_obj:
            raise ValueError(
                f"Phase 2 pool entry {i} 'objectives' missing keys {missing_obj}: {path}"
            )
        if not isinstance(entry["executed_trades"], int):
            raise ValueError(
                f"Phase 2 pool entry {i} 'executed_trades' must be an int: {path}"
            )
