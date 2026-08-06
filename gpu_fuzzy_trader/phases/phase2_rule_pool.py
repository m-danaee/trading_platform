"""
phase2_rule_pool.py — Rule_Pool_Generator (Phase 2)

GPU-accelerated multi-objective evolutionary search for fuzzy trading rules.

Uses NSGA-III via EvoX when available; falls back to NumPy NSGA-II when EvoX
is not installed.

Chromosome encoding:
    chromosome = [gene_0, gene_1, ..., gene_{K-1}]
    gene_i ∈ {0, ..., num_classes_i - 1, dont_care_i}
    dont_care_i = num_classes_i  (inactive condition)

Three objectives (all minimised, decoupled per Task 3 + OOS fix):
    f1 = -sortino_ratio + support
    f2 = max_drawdown_pct + support + dd_gate + trade_penalty
    f3 = -(robust_return_pct) + support + cond_penalty + overfit_gap
         (robust_return = min(train_return, val_return) when JOINT_TRAIN_VAL).
    f4 = concentration (+ diversity when PHASE2_DIVERSITY_ON_F4)

Penalties (NOT identically applied to all objectives to avoid Pareto collapse):
    support_penalty        — if executed_trades < MIN_TRADE_SUPPORT (weighted per obj)
    diversity_penalty      — Hamming + phenotype crowding (f4 by default; legacy f1+f3)
    trade_penalty          — if executed < MIN_TRADE_POOL_FLOOR (f2 only)
    drawdown_gate_penalty  — if dd > PHASE2_MAX_DRAWDOWN_GATE (f2 only)
    overfit_gap_penalty    — train_ret - val_ret > threshold (f3 only)
    cond_penalty           — active conditions outside [MIN_CONDITIONS, MAX_CONDITIONS] (f3 only)

Static risk parameters during Phase 2:
    TP = PHASE2_TP, SL = PHASE2_SL, capital_pct = PHASE2_CAPITAL_PCT
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.backtest.cpu_engine import CPUBacktestEngine
from gpu_fuzzy_trader.backtest.df_slim import slim_backtest_df
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
    trade_support_penalty as _trade_support_penalty,
)
from gpu_fuzzy_trader.phases.rule_identity import (
    feature_conditions_only,
    phase2_rule_id,
)
from gpu_fuzzy_trader.reporting.reporter import Reporter
from gpu_fuzzy_trader.validation.monthly_windows import (
    build_monthly_windows,
    monthly_return_counts_as_good,
)

logger = logging.getLogger(__name__)

# Increment this whenever chromosome classes or their source representation
# change meaning. Version 2 used an incompatible adaptive encoding; version 3
# restored fixed bins but still used unscaled ordinal source codes. Version 4
# uses train-fitted ordinal scaling and must not seed from either older archive.
_ARCHIVE_SCHEMA_VERSION = 4


def trade_support_penalty(executed: int, **kwargs) -> float:
    """Backward-compatible wrapper returning penalty only."""
    penalty, _, _ = _trade_support_penalty(executed, **kwargs)
    return penalty


def _saturating_sortino(raw: float) -> float:
    """tanh-saturated Sortino so the best-front member moves with progress.

    The previous flat cap pinned best_sortino at the SORTINO_CAP sentinel from
    generation 0 (visible in phase2_long_history.json: best=10.0 at gen 0..99).
    """
    scale = max(_cfg.SORTINO_SCALE, 1e-6)
    cap = _cfg.SORTINO_CAP
    return float(np.tanh(raw / scale) * cap)


def _derive_val_sample_seed(train_sample_seed: int) -> int:
    """Derive a deterministic validation sample seed from the training seed.

    This ensures train and validation sampling windows use different RNG
    states, avoiding the scenario where both splits select the same relative
    chronological window.  The derivation is a simple deterministic offset
    outside the typical seed range so the two seeds never alias.

    Returns
    -------
    int
        A value in ``[0, 2**31 - 1)``, guaranteed different from the input.
    """
    offset = 1_000_003  # large prime offset
    return (train_sample_seed + offset) % (2**31)


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


def _pool_path_key(direction: str):
    return direction


def _resolve_pool_path(direction: str) -> str:
    key = _pool_path_key(direction)
    if key in _POOL_PATHS:
        return _POOL_PATHS[key]
    return _cfg.PHASE2_POOL_PATHS[direction]


def _resolve_history_path(direction: str) -> str:
    key = _pool_path_key(direction)
    if key in _HISTORY_PATHS:
        return _HISTORY_PATHS[key]
    return _cfg.PHASE2_HISTORY_PATHS[direction]


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


def _phenotype_bucket_key(
    sortino_for_obj: float,
    dd_for_obj: float,
    f3_val: float,
) -> tuple[int, int, int]:
    """Discretise objective-relevant metrics for behavioral diversity."""
    sortino_step = float(getattr(_cfg, "PHASE2_PHENOTYPE_SORTINO_STEP", 0.5))
    dd_step = float(getattr(_cfg, "PHASE2_PHENOTYPE_DD_STEP", 5.0))
    f3_step = float(getattr(_cfg, "PHASE2_PHENOTYPE_F3_STEP", 10.0))
    sortino_step = max(sortino_step, 1e-9)
    dd_step = max(dd_step, 1e-9)
    f3_step = max(f3_step, 1e-9)
    return (
        int(sortino_for_obj / sortino_step),
        int(dd_for_obj / dd_step),
        int(f3_val / f3_step),
    )


def _diversity_penalty_blended(
    chromosome: np.ndarray,
    diversity_refs: list[np.ndarray],
    *,
    sortino_for_obj: float,
    dd_for_obj: float,
    f3_val: float,
    hamming_threshold: int,
    penalty_weight: float,
    diversity_metrics_by_key: dict[tuple[int, ...], dict] | None,
) -> float:
    """Hamming OR phenotype-bucket crowding penalty (same weight on both)."""
    from gpu_fuzzy_trader.phases.phase2_sparse_encoding import chromosome_key

    hamming_penalty = 0.0
    if diversity_refs:
        hammings = [_hamming_distance(chromosome, pf) for pf in diversity_refs]
        other_hammings = [h for h in hammings if h > 0]
        if other_hammings:
            min_hamming = min(other_hammings)
            if min_hamming <= hamming_threshold:
                hamming_penalty = penalty_weight

    phenotype_penalty = 0.0
    if diversity_refs and diversity_metrics_by_key:
        my_bucket = _phenotype_bucket_key(sortino_for_obj, dd_for_obj, f3_val)
        for pf in diversity_refs:
            if _hamming_distance(chromosome, pf) == 0:
                continue
            ref_row = diversity_metrics_by_key.get(chromosome_key(pf))
            if not ref_row:
                continue
            ref_bucket = ref_row.get("phenotype_bucket")
            if ref_bucket is not None and tuple(ref_bucket) == my_bucket:
                phenotype_penalty = penalty_weight
                break

    return max(hamming_penalty, phenotype_penalty)


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


def _symbol_robustness_penalty(metrics: dict) -> float:
    """Penalty for weak cross-symbol robustness on one split."""
    if not bool(metrics.get(
        "per_symbol_metrics_available",
        "per_symbol_metrics" in metrics,
    )):
        # Missing enrichment is unknown evidence, not an all-symbol win.
        return 0.0
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
    min_profitable = _cfg.effective_min_profitable_symbols(len(pnl_vec))
    shortfall = max(0, min_profitable - profitable)
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


def _downsample_chronological(
    df: pd.DataFrame,
    n_rows: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Pick a contiguous chronological slice of *n_rows* from *df*.

    Critical for trading backtests: bars are sequential temporal events, so the
    downsampler must NOT skip rows (stride sampling breaks position management
    and intraday pattern recognition). A uniform random start is chosen bounded
    so that ``n_rows`` always fits forward; if the symbol has fewer than
    ``n_rows`` bars the full chronological frame is returned.

    Args:
        df: Chronologically ordered DataFrame (per symbol).
        n_rows: Number of contiguous rows to take.
        rng: ``np.random.Generator`` used to pick the start index.
    """
    ordered = _sort_chronological(df)
    total = len(ordered)
    n_rows = min(n_rows, total)
    if n_rows >= total:
        return ordered.reset_index(drop=True)
    start = int(rng.integers(0, total - n_rows + 1))
    return ordered.iloc[start : start + n_rows].reset_index(drop=True)


def _largest_safe_range(
    total_len: int,
    forbidden: list[tuple[int, int]],
) -> tuple[int, int]:
    """Return (start, end) of the largest contiguous bar range not in
    *forbidden*. The forbidden list may be unsorted/overlapping; it is
    merged internally. ``end`` is inclusive. Returns ``(0, -1)`` if no
    safe range exists (caller must handle)."""
    if not forbidden:
        return (0, total_len - 1)
    # Merge forbidden intervals
    sorted_f = sorted(forbidden, key=lambda x: x[0])
    merged: list[tuple[int, int]] = []
    for s, e in sorted_f:
        s, e = max(0, int(s)), min(total_len - 1, int(e))
        if s > e:
            continue
        if merged and s <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    # Find largest gap
    best_start, best_end = 0, -1
    cursor = 0
    for s, e in merged:
        if s > cursor:
            gap_end = s - 1
            if (gap_end - cursor + 1) > (best_end - best_start + 1):
                best_start, best_end = cursor, gap_end
        cursor = max(cursor, e + 1)
    if cursor <= total_len - 1:
        gap_end = total_len - 1
        if (gap_end - cursor + 1) > (best_end - best_start + 1):
            best_start, best_end = cursor, gap_end
    return (best_start, best_end)


def _resolve_sample_total_rows(
    df: pd.DataFrame,
    total_rows: int,
    forbidden_ranges: list[tuple[int, int]] | None = None,
) -> int:
    """Cap *total_rows* so the per-symbol request fits within the safe range.

    When :attr:`config.PHASE2_PER_EPOCH_WINDOW_ROTATION` is enabled, the
    per-symbol row count is capped to a fraction of the largest safe
    (non-forbidden) contiguous bar range (and to
    :attr:`config.PHASE2_SAMPLE_MAX_BARS_PER_SYMBOL`).  Without this cap,
    ``_sample_df`` falls back to ``start = safe_start`` for every call,
    preventing window rotation.

    The cap ensures ``n_per_sym < safe_len`` so the RNG start-bar branch in
    ``_sample_df`` fires and epochs can see different chronological windows.

    Args:
        df: Input DataFrame with a ``symbol`` column.
        total_rows: Target row count across all symbols.
        forbidden_ranges: Per-symbol ``(start_bar, end_bar)`` intervals
            that the sampled slice must NOT overlap.

    Returns
    -------
    int
        Potentially reduced total_rows that allows rotated start bars.
    """
    if not _cfg.PHASE2_PER_EPOCH_WINDOW_ROTATION:
        return total_rows
    if "symbol" not in df.columns or df.empty:
        return total_rows
    symbols = df["symbol"].unique()
    n_sym = len(symbols)
    if n_sym == 0:
        return total_rows
    sym_groups = {sym: df[df["symbol"] == sym] for sym in symbols}
    min_sym_len = min(len(g) for g in sym_groups.values())
    if min_sym_len <= 0:
        return total_rows
    safe_start, safe_end = _largest_safe_range(
        min_sym_len, forbidden_ranges or []
    )
    safe_len = safe_end - safe_start + 1 if safe_end >= safe_start else 0
    if safe_len <= 0:
        return total_rows

    rotation_frac = float(
        getattr(_cfg, "PHASE2_SAMPLE_ROTATION_FRACTION", 0.65)
    )
    max_bars_cap = int(
        getattr(_cfg, "PHASE2_SAMPLE_MAX_BARS_PER_SYMBOL", 60_000)
    )
    # Reserve a small margin so n_per_sym stays strictly below safe_len.
    margin = max(1, safe_len // 100)
    max_per_sym = min(
        max(1, int(safe_len * rotation_frac)),
        max_bars_cap,
        max(1, safe_len - margin),
    )
    capped = min(total_rows, n_sym * max_per_sym)
    if capped < total_rows:
        logger.debug(
            "_resolve_sample_total_rows: capped %d → %d "
            "(safe_len=%d, max_per_sym=%d, n_sym=%d, rotation_frac=%.2f)",
            total_rows, capped, safe_len, max_per_sym, n_sym, rotation_frac,
        )
    return capped


def sample_df_for_phase2(
    df: pd.DataFrame,
    total_rows: int | None = None,
    random_state: int | np.random.Generator | None = None,
    forbidden_ranges: list[tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """Resolve Phase 2 row budget then sample with aligned symbol windows."""
    budget = int(
        total_rows if total_rows is not None else _cfg.PHASE1_SAMPLING_TOTAL)
    capped = _resolve_sample_total_rows(df, budget, forbidden_ranges)
    return _sample_df(
        df,
        capped,
        random_state=random_state,
        forbidden_ranges=forbidden_ranges,
    )


def _sample_df(
    df: pd.DataFrame,
    total_rows: int,
    random_state: int | np.random.Generator | None = None,
    forbidden_ranges: list[tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """Sample up to *total_rows* rows, distributed equally across symbols.

    A single random start bar is chosen and applied to **every** symbol, so
    the resulting slices are temporally aligned (all symbols see the same
    clock). The per-symbol slice is contiguous and chronological from that
    shared start. The start is bounded so the slice fits in the smallest
    symbol's history.

    Temporal alignment is required because the backtest engine assumes all
    symbols share the same timestamps — if BTC starts at bar 2,500 and ETH
    starts at bar 6,200 the engine sees them at different times, breaking
    cross-symbol signals (relative strength, pair trading) and intraday
    patterns.

    The caller-supplied ``random_state`` (int or ``np.random.Generator``)
    seeds the start selection for reproducibility; ``None`` uses
    ``PHASE2_SEED`` (process seed unless ``GLOBAL_SEED`` is set).

    Contiguity within each symbol is required because the backtest engine
    processes bars sequentially for position management, exposure release,
    and intraday pattern recognition — stride sampling silently drops
    intermediate candles.

    ``forbidden_ranges`` (per-symbol bar indices) excludes CV/holdout valid
    regions and their embargo buffer from the sampled slice to prevent
    training data from leaking into validation. If the largest safe range
    is smaller than the requested sample, the function returns whatever
    fits and logs a warning.

    Args:
        df: Input DataFrame (must contain a ``symbol`` column when
            ``len(symbols) > 1``).
        total_rows: Target row count across all symbols.
        random_state: Seed (int), ``np.random.Generator``, or ``None`` to use
            ``PHASE2_SEED``.
        forbidden_ranges: Per-symbol ``(start_bar, end_bar)`` intervals
            (inclusive) that the sampled slice must NOT overlap. Bars are
            0-indexed within each symbol. ``None`` or empty = no
            constraint.
    """
    rng = np.random.default_rng(_cfg.PHASE2_SEED) if random_state is None else (
        random_state if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )

    if "symbol" not in df.columns or df.empty:
        return _downsample_chronological(df, min(total_rows, len(df)), rng)

    symbols = df["symbol"].unique()
    n_sym = len(symbols)
    if n_sym == 0:
        return df.iloc[0:0].reset_index(drop=True)

    base, rem = divmod(total_rows, n_sym)
    sizes = [base + 1] * rem + [base] * (n_sym - rem)
    n_per_sym = sizes[0]

    sym_groups = {sym: df[df["symbol"] == sym] for sym in symbols}
    min_sym_len = min(len(g) for g in sym_groups.values())
    if min_sym_len <= 0:
        return df.iloc[0:0].reset_index(drop=True)

    safe_start, safe_end = _largest_safe_range(
        min_sym_len, forbidden_ranges or []
    )
    safe_len = safe_end - safe_start + 1 if safe_end >= safe_start else 0

    if safe_len <= 0:
        logger.warning(
            "_sample_df: no safe range available; returning empty")
        return df.iloc[0:0].reset_index(drop=True)

    if n_per_sym <= safe_len:
        max_start = safe_end - n_per_sym + 1
        start = int(rng.integers(safe_start, max_start + 1))
        sizes_to_use = sizes
    else:
        logger.warning(
            "_sample_df: requested %d bars/sym exceeds largest safe range "
            "%d; using the entire safe range (%d bars/sym)",
            n_per_sym, safe_len, safe_len,
        )
        start = safe_start
        # Shrink: each symbol contributes the full safe range
        sizes_to_use = [safe_len] * n_sym

    parts = []
    for sym, target_n in zip(symbols, sizes_to_use):
        sym_df = sym_groups[sym]
        avail = max(0, len(sym_df) - start)
        take = min(target_n, avail)
        if take <= 0:
            continue
        ordered = _sort_chronological(sym_df)
        parts.append(ordered.iloc[start : start + take].reset_index(drop=True))

    if not parts:
        return df.iloc[0:0].reset_index(drop=True)
    result = pd.concat(parts, ignore_index=True)
    bars_per_sym = int(sizes_to_use[0]) if sizes_to_use else 0
    logger.info(
        "_sample_df: bars/sym=%d start=%d safe_len=%d n_sym=%d sampled_rows=%d",
        bars_per_sym,
        start,
        safe_len,
        n_sym,
        len(result),
    )
    return result


# ---------------------------------------------------------------------------
# Purged CV fold evaluator (CPU, sequential per fold)
# ---------------------------------------------------------------------------


class CvFoldValEvaluator:
    """
    Evaluate chromosomes on purged CV validation folds (excluding holdout).

    Aggregates per-fold metrics with ``aggregate_fold_metrics`` for fitness.
    Builds one fold engine at a time to limit RAM use.
    """

    def __init__(
        self,
        cv_folds: list,
        feature_modes: dict[str, str],
        feature_names: list[str],
        direction: str,
    ) -> None:
        from gpu_fuzzy_trader.validation.rolling_cv import (
            aggregate_fold_metrics,
            cv_folds_only,
        )

        self._folds = cv_folds_only(cv_folds)
        self._feature_modes = feature_modes
        self._feature_names = feature_names
        self._direction = direction
        self._aggregate_fn = aggregate_fold_metrics
        fold_rows = [int(f.n_valid_rows) for f in self._folds]
        self.n_valid_rows = min(fold_rows) if fold_rows else 0
        # Cache: build fold engines once, reuse across all evolution calls
        self._cached_fold_engines: list | None = None

    def _ensure_fold_engines(self) -> list:
        """Build and cache one CPUBacktestEngine per CV fold (one-time cost)."""
        if self._cached_fold_engines is not None:
            return self._cached_fold_engines
        engines: list = []
        for fold in self._folds:
            slim = slim_backtest_df(fold.valid_df, self._feature_names)
            engine = CPUBacktestEngine(
                slim,
                self._feature_modes,
                self._direction,
                fee_pct=_cfg.FEE_PCT,
            )
            engines.append(engine)
        self._cached_fold_engines = engines
        return engines

    def simulate_rule_batch(
        self,
        chromosomes: np.ndarray,
        *,
        tp: float,
        sl: float,
        capital_pct: float,
    ) -> list[dict]:
        if not self._folds:
            return [
                self._aggregate_fn([])
                for _ in range(len(chromosomes))
            ]

        n_chrom = len(chromosomes)

        def _eval_one_fold(engine) -> list[dict]:
            try:
                return engine.simulate_rule_batch(
                    chromosomes=chromosomes,
                    tp=tp,
                    sl=sl,
                    capital_pct=capital_pct,
                )
            except Exception as exc:
                logger.debug(
                    "CvFoldValEvaluator fold engine failed: %s",
                    exc,
                )
                return [
                    {
                        "total_return_pct": -100.0,
                        "profit_factor": 0.0,
                        "sortino_ratio": 0.0,
                        "max_drawdown_pct": 100.0,
                        "executed_trades": 0,
                        "win_rate": 0.0,
                    }
                    for _ in range(n_chrom)
                ]

        engines = self._ensure_fold_engines()
        if len(engines) <= 1:
            all_fold_metrics = [_eval_one_fold(e) for e in engines]
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=len(engines)) as pool:
                all_fold_metrics = list(pool.map(_eval_one_fold, engines))

        per_chrom: list[list[dict]] = [[] for _ in range(n_chrom)]
        for fold_metrics in all_fold_metrics:
            for i, metrics in enumerate(fold_metrics):
                per_chrom[i].append(metrics)

        mode = str(getattr(_cfg, "PURGED_WF_AGGREGATION", "worst"))
        return [self._aggregate_fn(fms, mode=mode) for fms in per_chrom]


# ---------------------------------------------------------------------------
# Fitness evaluation
# ---------------------------------------------------------------------------

def _val_trade_floor_for_objectives(n_valid_rows: int | None = None) -> int:
    """Minimum validation trades before joint Sortino is trusted."""
    return int(_cfg.effective_val_trade_floor_for_objectives(n_valid_rows))


def compute_phase2_objectives_from_metrics(
    chromosome: np.ndarray,
    dont_cares: np.ndarray,
    metrics: dict,
    pareto_front: list[np.ndarray],
    *,
    val_metrics: dict | None = None,
    diversity_reference: list[np.ndarray] | None = None,
    diversity_metrics_by_key: dict[tuple[int, ...], dict] | None = None,
    stage_params=None,
    n_valid_rows: int | None = None,
    island_hyperparams: _cfg.IslandHyperparams | None = None,
    direction: str | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Build Phase 2 minimisation objectives from precomputed train/val metrics.

    Shared by single-chromosome evaluation and EvoX batch assignment so penalty
    logic stays identical across code paths.
    """
    from gpu_fuzzy_trader.phases.phase2_support import (
        _raw_feasibility_violation_score,
        expectancy_lcb_pct,
        resolve_evolution_floors,
        robust_return_pct,
    )

    floors = resolve_evolution_floors(
        stage_params, n_rows=n_valid_rows, island_hyperparams=island_hyperparams,
    )
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
    dd_for_obj = max_dd
    win_rate = float(metrics.get("win_rate", 0.0))
    executed = int(metrics.get("executed_trades", 0))
    uncertainty_available = bool(
        metrics.get(
            "_uncertainty_metrics_available",
            any(
                key in metrics
                for key in (
                    "expectancy_lcb_pct_per_trade",
                    "trade_return_std_pct",
                    "expected_shortfall_pct",
                )
            ),
        )
    )
    metrics["_uncertainty_metrics_available"] = uncertainty_available
    expectancy_lcb = expectancy_lcb_pct(metrics)
    metrics["expectancy_lcb_pct_per_trade"] = float(expectancy_lcb)
    expected_shortfall = float(
        metrics.get("expected_shortfall_pct", 0.0) or 0.0
    )
    metrics["expected_shortfall_pct"] = expected_shortfall
    es_penalty = max(0.0, -expected_shortfall) * float(
        getattr(_cfg, "PHASE2_EXPECTED_SHORTFALL_WEIGHT", 0.0)
    )
    cv_returns = [
        float(value) for value in metrics.get("_cv_fold_returns", [])
        if value is not None
    ]
    fold_instability = (
        float(np.std(cv_returns, ddof=1))
        if len(cv_returns) > 1 else 0.0
    )
    metrics["fold_return_std_pct"] = fold_instability

    sortino_for_obj = sortino_train
    val_floor_penalty = 0.0
    val_trade_floor = (
        int(island_hyperparams.val_trade_floor)
        if island_hyperparams is not None
        else _val_trade_floor_for_objectives(n_valid_rows)
    )

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
            val_max_dd = float(val_metrics.get("max_drawdown_pct", 0.0))
            dd_for_obj = max(max_dd, val_max_dd)
            if val_executed < val_trade_floor:
                sortino_for_obj = min(sortino_train, 0.0)
            else:
                sortino_for_obj = min(sortino_train, sortino_val)
        # C6: Gate val-derived floor penalties behind JOINT_TRAIN_VAL or VAL_IN_FITNESS_PENALTY
        if _cfg.PHASE2_JOINT_TRAIN_VAL or getattr(_cfg, "PHASE2_VAL_IN_FITNESS_PENALTY", False):
            val_return_floor = _cfg.effective_phase2_val_return_floor_pct(
                direction)
            if val_total_return < val_return_floor:
                val_floor_penalty += _cfg.SUPPORT_PENALTY_MAX
            if val_profit_factor < _cfg.PHASE2_PROFIT_FACTOR_FLOOR_EVOLUTION:
                val_floor_penalty += (
                    _cfg.PHASE2_PROFIT_FACTOR_FLOOR_EVOLUTION - val_profit_factor
                ) * 5.0

    support_penalty, _, _ = compute_support_penalty_and_specialist(
        metrics,
        min_trade_support=floors.min_trade_support,
    )
    if (
        val_metrics is not None
        and int(val_metrics.get("executed_trades", 0)) < val_trade_floor
    ):
        # C6: Gate val trade-floor support cap behind JOINT_TRAIN_VAL or VAL_IN_FITNESS_PENALTY
        if _cfg.PHASE2_JOINT_TRAIN_VAL or getattr(_cfg, "PHASE2_VAL_IN_FITNESS_PENALTY", False):
            support_penalty = max(support_penalty, _cfg.SUPPORT_PENALTY_MAX)
        if _cfg.PHASE2_JOINT_TRAIN_VAL:
            sortino_for_obj = min(sortino_train, 0.0)

    if total_return < floors.return_floor_pct:
        support_penalty = max(support_penalty, _cfg.SUPPORT_PENALTY_MAX)
    if profit_factor < _cfg.PHASE2_PROFIT_FACTOR_FLOOR_EVOLUTION:
        support_penalty += (_cfg.PHASE2_PROFIT_FACTOR_FLOOR_EVOLUTION - profit_factor) * 5.0

    # C5: Symbol-spread penalty — penalize when few symbols are profitable.
    # One-symbol islands must use island-scaled targets (else every rule is
    # permanently penalised for failing a 3-symbol bar it can never clear).
    if bool(metrics.get(
        "per_symbol_metrics_available",
        "per_symbol_metrics" in metrics,
    )):
        per_sym = metrics.get("per_symbol_metrics", {}) or {}
        n_profitable_symbols = sum(
            1 for v in per_sym.values()
            if isinstance(v, dict) and float(v.get("net_pnl", 0.0)) > 0.0
        )
        if island_hyperparams is not None:
            min_symbols = int(island_hyperparams.min_profitable_symbols)
        else:
            min_symbols = int(
                getattr(_cfg, "PHASE2_MIN_PROFITABLE_SYMBOLS_PENALTY", 3))
        if n_profitable_symbols < min_symbols:
            support_penalty += float(min_symbols - n_profitable_symbols) * 2.0

    if not (island_hyperparams is not None and island_hyperparams.skip_symbol_robustness_penalty):
        support_penalty += _symbol_robustness_penalty(metrics)
        if val_metrics is not None:
            # C6: Gate val symbol_robustness behind JOINT_TRAIN_VAL or VAL_IN_FITNESS_PENALTY
            if _cfg.PHASE2_JOINT_TRAIN_VAL or getattr(_cfg, "PHASE2_VAL_IN_FITNESS_PENALTY", False):
                support_penalty += _symbol_robustness_penalty(val_metrics)
    support_penalty += val_floor_penalty

    dd_gate = getattr(_cfg, "PHASE2_MAX_DRAWDOWN_GATE", 20.0)
    drawdown_gate_penalty = 0.0
    if dd_for_obj > dd_gate:
        excess = dd_for_obj - dd_gate
        drawdown_gate_penalty = excess * 2.0

    # H2: f3_val based on PHASE2_F3_OBJECTIVE
    # → fixes audit finding #5 (dead f3 profit_factor branch;
    # USE_TOTAL_RETURN_OBJ takes precedence when True).
    if not _cfg.PHASE2_USE_TOTAL_RETURN_OBJ:
        f3_objective = str(getattr(_cfg, "PHASE2_F3_OBJECTIVE", "profit_factor"))
        if f3_objective == "cv_fold_min":
            cv_fold_returns = metrics.get("_cv_fold_returns", [])
            if cv_fold_returns:
                f3_val = min(cv_fold_returns)
            else:
                # Fallback when CV not available: use profit_factor
                f3_val = profit_factor
        elif f3_objective == "profit_factor":
            f3_val = profit_factor
            if val_metrics is not None and _cfg.PHASE2_JOINT_TRAIN_VAL:
                val_pf = float(val_metrics.get("profit_factor", profit_factor))
                if int(val_metrics.get("executed_trades", 0)) < val_trade_floor:
                    f3_val = min(profit_factor, 0.0)
                else:
                    f3_val = min(profit_factor, val_pf)
        else:  # "win_rate" legacy
            f3_val = win_rate
            if val_metrics is not None and _cfg.PHASE2_JOINT_TRAIN_VAL:
                val_wr = float(val_metrics.get("win_rate", 0.0))
                if int(val_metrics.get("executed_trades", 0)) < val_trade_floor:
                    f3_val = min(win_rate, 0.0)
                else:
                    f3_val = min(win_rate, val_wr)

    # PHASE2_USE_TOTAL_RETURN_OBJ=True (default): f3 uses robust return
    # (min train/val return) — overrides the PHASE2_F3_OBJECTIVE setting.
    # Set USE_TOTAL_RETURN_OBJ=False to opt into the legacy
    # profit_factor / cv_fold_min / win_rate path.
    # (fixes audit finding #5)
    if _cfg.PHASE2_USE_TOTAL_RETURN_OBJ:
        joint_return = (
            bool(_cfg.PHASE2_JOINT_TRAIN_VAL)
            and floors.use_robust_return_obj
            and val_metrics is not None
        )
        f3_val = robust_return_pct(
            metrics, val_metrics, joint=joint_return,
        )

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
    diversity_hamming_threshold: int
    if bool(getattr(_cfg, "PHASE2_DIVERSITY_HAMMING_THRESHOLD_AUTO", True)):
        k_active = _count_active_conditions(chromosome, dont_cares)
        diversity_hamming_threshold = max(3, k_active // 5)
    else:
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
    diversity_penalty = _diversity_penalty_blended(
        chromosome,
        diversity_refs,
        sortino_for_obj=sortino_for_obj,
        dd_for_obj=dd_for_obj,
        f3_val=f3_val,
        hamming_threshold=diversity_hamming_threshold,
        penalty_weight=diversity_penalty_weight,
        diversity_metrics_by_key=diversity_metrics_by_key,
    )

    from gpu_fuzzy_trader.phases.phase2_support import _val_terms_in_fitness

    include_val = _val_terms_in_fitness()
    # Stage A soft_feasibility: skip heavy feasibility_violation weight;
    # graduated return/support penalties still apply via floors above.
    raw_violation = 0.0
    if not floors.soft_feasibility:
        raw_violation = _raw_feasibility_violation_score(
            metrics,
            val_metrics,
            include_val=include_val,
            island_hyperparams=island_hyperparams,
        )
    # Keep a stable scalar for constrained selection even when the soft stage
    # intentionally disables the hard feasibility penalty.
    metrics["constraint_violation"] = float(raw_violation)
    if raw_violation > 0.0:
        metrics["feasibility_violation"] = raw_violation
        support_penalty += (
            raw_violation * float(_cfg.PHASE2_FEASIBILITY_VIOLATION_WEIGHT)
        )

    trade_penalty = 0.0
    if island_hyperparams is not None:
        trade_floor = int(island_hyperparams.min_trade_pool_floor)
    else:
        trade_floor = _cfg.effective_min_trade_pool_floor(n_valid_rows)
    # Soft Stage A: do not hard-kill low-trade rules (support penalty remains).
    if executed < trade_floor and not floors.soft_feasibility:
        dd_for_obj = 100.0
        sortino_for_obj = 0.0
        f3_val = 0.0
        trade_penalty = float(_cfg.PHASE2_INFEASIBLE_OBJECTIVE_PENALTY)

    metrics["phenotype_bucket"] = _phenotype_bucket_key(
        sortino_for_obj, dd_for_obj, f3_val,
    )

    overfit_gap_penalty = 0.0
    if (
        val_metrics is not None
        and float(getattr(_cfg, "PHASE2_OVERFIT_GAP_PENALTY_WEIGHT", 0.0)) > 0.0
        and (_cfg.PHASE2_JOINT_TRAIN_VAL or getattr(_cfg, "PHASE2_VAL_IN_FITNESS_PENALTY", False))
    ):
        train_ret = float(metrics.get("total_return_pct", 0.0))
        val_ret = float(val_metrics.get("total_return_pct", 0.0))
        gap_pct = train_ret - val_ret
        gap_threshold = float(
            getattr(_cfg, "PHASE2_OVERFIT_GAP_PCT_THRESHOLD", 8.0),
        )
        if gap_pct > gap_threshold:
            overfit_gap_penalty = (
                (gap_pct - gap_threshold)
                * float(_cfg.PHASE2_OVERFIT_GAP_PENALTY_WEIGHT)
            )

    # Decoupled objectives:
    #   f1 = -Sortino + support                    (no diversity — avoids f1↔f3 collapse)
    #   f2 = DD + support + trade_penalty + DD_gate
    #   f3 = -robust_return + support + cond + overfit_gap
    #   f4 = concentration (+ diversity when PHASE2_DIVERSITY_ON_F4)
    diversity_on_f4 = bool(getattr(_cfg, "PHASE2_DIVERSITY_ON_F4", True))
    diversity_f1_f3 = 0.0 if diversity_on_f4 else diversity_penalty
    diversity_f4 = diversity_penalty if diversity_on_f4 else 0.0

    f1 = (
        -sortino_for_obj
        + (_cfg.PHASE2_SUPPORT_PENALTY_WEIGHT_F1 * support_penalty)
        + diversity_f1_f3
    )
    f2 = (
        dd_for_obj
        + (_cfg.PHASE2_SUPPORT_PENALTY_WEIGHT_F2 * support_penalty)
        + drawdown_gate_penalty
        + trade_penalty
        + es_penalty
        + 0.25 * fold_instability
        + (0.0 if getattr(_cfg, "PHASE2_F4_ENABLED", False) else diversity_f4)
    )
    f3 = (
        -f3_val
        + (_cfg.PHASE2_SUPPORT_PENALTY_WEIGHT_F3 * support_penalty)
        + diversity_f1_f3
        + cond_penalty
        + overfit_gap_penalty
        - (
            0.0
            if (
                bool(_cfg.PHASE2_USE_TOTAL_RETURN_OBJ)
                or not uncertainty_available
            )
            else float(getattr(_cfg, "PHASE2_EXPECTANCY_LCB_WEIGHT", 0.0))
            * float(np.tanh(expectancy_lcb))
        )
    )

    if val_metrics is not None:
        joint_ret = bool(_cfg.PHASE2_JOINT_TRAIN_VAL) and bool(
            _cfg.PHASE2_USE_ROBUST_RETURN_OBJ,
        )
        metrics["robust_return_pct"] = robust_return_pct(
            metrics, val_metrics, joint=joint_ret,
        )

    # f4: return-concentration ratio (Task 2)
    #   f4 = max_single_trade_pnl / max(sum_positive_trade_pnl, ε)
    #   High f4 → rule edge depends on one outlier trade (bad).
    #   Diversity is added to the NSGA objective only (not the stored metric).
    f4_concentration = 0.0
    if getattr(_cfg, "PHASE2_F4_ENABLED", False):
        max_tr = float(metrics.get("max_single_trade_pnl", 0.0))
        sum_pos = float(metrics.get("sum_positive_trade_pnl", 0.0))
        _f4_eps = float(getattr(_cfg, "PHASE2_F4_EPSILON", 1e-6))
        f4_concentration = max_tr / max(sum_pos, _f4_eps)

        if val_metrics is not None and bool(getattr(_cfg, "PHASE2_JOINT_TRAIN_VAL", False)):
            max_tr_v = float(val_metrics.get("max_single_trade_pnl", 0.0))
            sum_pos_v = float(val_metrics.get("sum_positive_trade_pnl", 0.0))
            f4_v = max_tr_v / max(sum_pos_v, _f4_eps)
            f4_concentration = min(f4_concentration, f4_v)

        if executed < trade_floor:
            # A low-support rule is not concentration-safe.  Zero used to
            # reward sparse rules on the fourth objective.
            f4_concentration = 1.0

    metrics["f4_concentration"] = f4_concentration
    f4 = f4_concentration + diversity_f4

    if bool(getattr(_cfg, "PHASE2_F4_ENABLED", False)):
        objectives = np.array([f1, f2, f3, f4], dtype=np.float64)
    else:
        objectives = np.array([f1, f2, f3], dtype=np.float64)
    return objectives, metrics


def _evaluate_chromosome(
    chromosome: np.ndarray,
    dont_cares: np.ndarray,
    engine,  # GPUBacktestEngine or CPUBacktestEngine
    pareto_front: list[np.ndarray],
    val_engine=None,  # optional second engine for joint train+val objective
    diversity_reference: list[np.ndarray] | None = None,
    diversity_metrics_by_key: dict[tuple[int, ...], dict] | None = None,
    stage_params=None,
    island_hyperparams: _cfg.IslandHyperparams | None = None,
    cv_fold_evaluator: CvFoldValEvaluator | None = None,
    run_val: bool = True,
    generation: int | None = None,
    is_last_gen: bool = False,
) -> tuple[np.ndarray, dict]:
    """
    Evaluate a single chromosome and return (objectives, metrics).

    objectives = [f1, f2, f3] (all minimised, with penalties applied).

    When PHASE2_JOINT_TRAIN_VAL is enabled and *val_engine* is provided, f1 uses
    min(saturated_train_sortino, saturated_val_sortino) so the search prefers
    rules that hold up out-of-sample.

    When *cv_fold_evaluator* is provided and PHASE2_F3_OBJECTIVE is "cv_fold_min",
    per-fold returns are computed and stored as "_cv_fold_returns" in metrics.
    """
    key_suffix = str(chromosome[:8].tolist())
    try:
        metrics_list = engine.simulate_rule_batch(
            chromosomes=_chromosome_batch(chromosome),
            tp=_cfg.PHASE2_TP,
            sl=_cfg.PHASE2_SL,
            capital_pct=_cfg.PHASE2_CAPITAL_PCT,
            generation=generation,
            is_last_gen=is_last_gen,
        )
        metrics = metrics_list[0]
    except Exception as exc:
        logger.warning("simulate_rule_batch failed for chromosome %s: %s", key_suffix, exc)
        metrics = {
            "sortino_ratio": 0.0,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 100.0,
            "win_rate": 0.0,
            "executed_trades": 0,
        }

    val_metrics: dict | None = None
    if val_engine is not None and run_val:
        try:
            val_list = val_engine.simulate_rule_batch(
                chromosomes=_chromosome_batch(chromosome),
                tp=_cfg.PHASE2_TP,
                sl=_cfg.PHASE2_SL,
                capital_pct=_cfg.PHASE2_CAPITAL_PCT,
                generation=generation,
                is_last_gen=is_last_gen,
            )
            val_metrics = val_list[0]
        except Exception as exc:
            logger.warning("val simulate_rule_batch failed for chromosome %s: %s", key_suffix, exc)
            val_metrics = None

    # H2: Compute CV fold returns for cv_fold_min objective
    if (
        cv_fold_evaluator is not None
        and str(getattr(_cfg, "PHASE2_F3_OBJECTIVE", "profit_factor")) == "cv_fold_min"
    ):
        try:
            # Access internal fold engines to get per-fold returns
            # Use the same pattern as CvFoldValEvaluator.simulate_rule_batch
            engines = cv_fold_evaluator._ensure_fold_engines()
            chrom_batch = _chromosome_batch(chromosome)
            cv_returns: list[float] = []
            for fe in engines:
                fold_metrics_list = fe.simulate_rule_batch(
                    chromosomes=chrom_batch,
                    tp=_cfg.PHASE2_TP,
                    sl=_cfg.PHASE2_SL,
                    capital_pct=_cfg.PHASE2_CAPITAL_PCT,
                )
                cv_returns.append(
                    float(fold_metrics_list[0].get("total_return_pct", 0.0))
                )
            metrics["_cv_fold_returns"] = cv_returns
        except Exception as exc:
            logger.debug(
                "CV fold evaluation failed for chromosome %s: %s",
                key_suffix, exc,
            )
            metrics["_cv_fold_returns"] = []

    if island_hyperparams is None:
        island_hyperparams = getattr(engine, "_island_hyperparams", None)

    n_valid_rows = (
        int(getattr(val_engine, "n_valid_rows"))
        if val_engine is not None and getattr(val_engine, "n_valid_rows", None)
        else None
    )

    direction = getattr(engine, "trade_direction", None)
    return compute_phase2_objectives_from_metrics(
        chromosome,
        dont_cares,
        metrics,
        pareto_front,
        val_metrics=val_metrics,
        diversity_reference=diversity_reference,
        diversity_metrics_by_key=diversity_metrics_by_key,
        stage_params=stage_params,
        n_valid_rows=n_valid_rows,
        island_hyperparams=island_hyperparams,
        direction=direction,
    )


def attach_cv_fold_returns_batch(
    metrics_list: list[dict],
    chromosomes: np.ndarray,
    cv_fold_evaluator: "CvFoldValEvaluator | None",
) -> None:
    """Attach per-fold returns for cv_fold_min f3 on batched metrics."""
    if cv_fold_evaluator is None:
        return
    if str(getattr(_cfg, "PHASE2_F3_OBJECTIVE", "profit_factor")) != "cv_fold_min":
        return
    try:
        engines = cv_fold_evaluator._ensure_fold_engines()
        for idx, metrics in enumerate(metrics_list):
            chrom = chromosomes[idx]
            chrom_batch = _chromosome_batch(chrom)
            cv_returns: list[float] = []
            for fe in engines:
                fold_metrics_list = fe.simulate_rule_batch(
                    chromosomes=chrom_batch,
                    tp=_cfg.PHASE2_TP,
                    sl=_cfg.PHASE2_SL,
                    capital_pct=_cfg.PHASE2_CAPITAL_PCT,
                )
                cv_returns.append(
                    float(fold_metrics_list[0].get("total_return_pct", 0.0))
                )
            metrics["_cv_fold_returns"] = cv_returns
    except Exception as exc:
        logger.debug("attach_cv_fold_returns_batch failed: %s", exc)
        for metrics in metrics_list:
            metrics.setdefault("_cv_fold_returns", [])


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
    # C5: Symbol gene dont_care bias
    # Note: "symbol" is in META_COLUMNS and rarely in feature_infos, so this
    # bias may silently do nothing if no symbol-named feature exists.
    symbol_gene_prob = float(getattr(_cfg, "PHASE2_SYMBOL_GENE_DONT_CARE_PROB", 0.4))
    for k in range(K):
        # If this gene is a symbol gene, force dont_care with probability
        if "symbol" in str(feature_infos[k].get("name", "")).lower():
            if rng.random() < symbol_gene_prob:
                child[k] = int(dont_cares[k])
                continue
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


def _archive_direction(direction: str, *engines: object) -> str:
    """Resolve a plain long/short direction from an evolution log tag."""
    for value in (direction, *(getattr(e, "trade_direction", None) for e in engines)):
        text = str(value or "").strip().lower()
        if text in {"long", "short"}:
            return text
        if "short" in text:
            return "short"
        if "long" in text:
            return "long"
    return "long"


def _build_cpu_archive_engine(source_engine: object | None, direction: str):
    """Build the mandatory CPU evaluator from a Phase 2 engine.

    GPU batch metrics are useful for evolutionary search, but they are not a
    sufficient admission certificate because symbol-level metrics are
    optional on that path.  The final archive pass therefore uses the exact
    sampled frame owned by the Phase 2 engine and always constructs a CPU
    engine when possible.
    """
    if source_engine is None:
        return None
    if isinstance(source_engine, CPUBacktestEngine):
        return source_engine
    frame = getattr(source_engine, "df", None)
    feature_modes = getattr(source_engine, "feature_modes", None)
    if not isinstance(frame, pd.DataFrame) or not isinstance(feature_modes, dict):
        return None
    try:
        return CPUBacktestEngine(
            frame,
            dict(feature_modes),
            _archive_direction(direction, source_engine),
            fee_pct=float(getattr(source_engine, "fee_pct", _cfg.FEE_PCT)),
        )
    except Exception as exc:
        logger.warning("Phase 2 CPU archive evaluator unavailable: %s", exc)
        return None


def _snapshot_per_symbol_metrics(metrics: dict | None) -> dict[str, dict[str, float | int]]:
    """Return a JSON-safe, stable snapshot of CPU per-symbol metrics."""
    raw = (metrics or {}).get("per_symbol_metrics", {}) or {}
    if not isinstance(raw, dict):
        return {}
    snapshot: dict[str, dict[str, float | int]] = {}
    for symbol, values in raw.items():
        if not isinstance(values, dict):
            continue
        try:
            snapshot[str(symbol)] = {
                "trade_count": int(values.get("trade_count", 0)),
                "win_rate": float(values.get("win_rate", 0.0)),
                "net_pnl": float(values.get("net_pnl", 0.0)),
            }
        except (TypeError, ValueError):
            continue
    return snapshot


def _positive_contributor_symbols(
    metrics: dict | None,
    *,
    min_trades: int | None = None,
) -> set[str]:
    """Return symbols with positive PnL and enough validation support."""
    trade_floor = int(
        min_trades
        if min_trades is not None
        else getattr(_cfg, "RB_MIN_VALID_TRADES", 6)
    )
    per_symbol = _snapshot_per_symbol_metrics(metrics)
    return {
        symbol
        for symbol, values in per_symbol.items()
        if float(values.get("net_pnl", 0.0)) > 0.0
        and int(values.get("trade_count", 0)) >= trade_floor
    }


def _entry_validation_per_symbol_metrics(entry: dict) -> dict:
    """Read validation per-symbol metrics across pool schema revisions."""
    for key in (
        "valid_per_symbol_metrics",
        "val_per_symbol_metrics",
        "validation_per_symbol_metrics",
    ):
        value = entry.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _pool_entry_rank(entry: dict) -> float:
    """Compute the existing deployability rank from a pool entry."""
    from gpu_fuzzy_trader.phases.phase2_support import deployability_rank_score

    objectives = entry.get("objectives", {}) or {}
    train_metrics = {
        "total_return_pct": float(objectives.get("total_return_pct", 0.0)),
        "profit_factor": float(objectives.get("profit_factor", 1.0)),
        "executed_trades": int(entry.get("executed_trades", 0)),
        "sortino_ratio": float(objectives.get("sortino_ratio", 0.0)),
        "max_drawdown_pct": float(objectives.get("max_drawdown_pct", 0.0)),
    }
    val_obj = entry.get("val_objectives")
    val_metrics = None
    if isinstance(val_obj, dict):
        val_metrics = {
            "total_return_pct": float(val_obj.get("total_return_pct", 0.0)),
            "profit_factor": float(val_obj.get("profit_factor", 1.0)),
            "executed_trades": int(entry.get("val_executed_trades", 0)),
            "sortino_ratio": float(val_obj.get("sortino_ratio", 0.0)),
            "max_drawdown_pct": float(val_obj.get("max_drawdown_pct", 0.0)),
        }
    return float(deployability_rank_score(train_metrics, val_metrics))


def _reserve_symbol_pool_candidates(
    pool: list[dict],
    *,
    keep_top: int,
    coverage_report: dict[str, Any] | None = None,
) -> list[dict]:
    """Cap a pool while reserving admitted candidates for positive symbols.

    The reservation pass runs only on candidates that already passed every
    admission gate.  A rule can satisfy more than one symbol, but is retained
    only once.  The remaining capacity is filled by the established global
    deployability rank.
    """
    admitted_pool = [
        entry for entry in pool
        if entry.get("admission_passed", True)
        and entry.get("pool_admission_passed", True)
    ]
    if keep_top <= 0:
        if coverage_report is not None:
            coverage_report["reservations"] = {}
            coverage_report["reservation_counts"] = {}
            coverage_report["pool_before_reservation"] = len(pool)
            coverage_report["pool_after_reservation"] = 0
        return []
    if len(pool) <= keep_top:
        if coverage_report is not None:
            coverage_report["reservations"] = {}
            coverage_report["reservation_counts"] = {}
            coverage_report["pool_before_reservation"] = len(pool)
            coverage_report["pool_after_reservation"] = len(admitted_pool)
        # The helper is also used after merging persisted pools.  Do not let
        # a stale/rejected row survive merely because the cap is not tight.
        return list(admitted_pool)

    ranked = sorted(
        admitted_pool,
        key=lambda entry: (
            _pool_entry_rank(entry),
            tuple(int(v) for v in entry.get("chromosome", [])),
        ),
        reverse=True,
    )
    min_trades = int(getattr(_cfg, "RB_MIN_VALID_TRADES", 6))
    by_symbol: dict[str, list[dict]] = {}
    for entry in ranked:
        for symbol in _positive_contributor_symbols(
            {"per_symbol_metrics": _entry_validation_per_symbol_metrics(
                entry)},
            min_trades=min_trades,
        ):
            by_symbol.setdefault(symbol, []).append(entry)

    max_per_symbol = int(
        getattr(_cfg, "PHASE2_MAX_RESERVED_RULES_PER_SYMBOL", 10)
    )
    reserved: list[dict] = []
    reserved_keys: set[tuple[int, ...]] = set()
    reservation_details: dict[str, list[list[int]]] = {}
    # Scarcer contributors get first access to the bounded capacity.
    symbols = sorted(by_symbol, key=lambda sym: (len(by_symbol[sym]), sym))
    for symbol in symbols:
        selected_for_symbol: list[list[int]] = []
        for entry in by_symbol[symbol][:max_per_symbol]:
            key = tuple(int(v) for v in entry.get("chromosome", []))
            if key in reserved_keys:
                selected_for_symbol.append(list(key))
                continue
            if len(reserved) >= keep_top:
                continue
            reserved.append(entry)
            reserved_keys.add(key)
            selected_for_symbol.append(list(key))
        if selected_for_symbol:
            reservation_details[symbol] = selected_for_symbol

    for entry in ranked:
        if len(reserved) >= keep_top:
            break
        key = tuple(int(v) for v in entry.get("chromosome", []))
        if key not in reserved_keys:
            reserved.append(entry)
            reserved_keys.add(key)

    if coverage_report is not None:
        coverage_report["reservations"] = reservation_details
        coverage_report["reservation_counts"] = {
            symbol: len(chromosomes)
            for symbol, chromosomes in reservation_details.items()
        }
        coverage_report["pool_before_reservation"] = len(pool)
        coverage_report["pool_after_reservation"] = len(reserved)
    return reserved


def _build_pool_from_archive(
    archive: list[np.ndarray],
    feature_infos: list[dict],
    dont_cares: np.ndarray,
    engine,
    metrics_by_chrom: dict[tuple, dict] | None = None,
    val_engine=None,
    cpu_engine=None,
    cpu_val_engine=None,
    cv_fold_evaluator: CvFoldValEvaluator | None = None,
    holdout_n_valid_rows: int | None = None,
    train_n_rows: int | None = None,
    island_hyperparams: _cfg.IslandHyperparams | None = None,
    direction: str = "",
    coverage_report: dict[str, Any] | None = None,
) -> list[dict]:
    """
    Convert a list of Pareto-front chromosomes into pool JSON entries.

    Each pool entry schema:
    {
        "chromosome": [...],
        "conditions": [...],
        "objectives": {"sortino_ratio": ..., "max_drawdown_pct": ..., "win_rate": ...},
        "executed_trades": ...
    }

    ``metrics_by_chrom`` is deliberately not used when a CPU archive
    evaluator can be built.  Evolution metrics may be GPU-only snapshots and
    can lack current per-symbol evidence.
    """
    report: dict[str, Any] | None = coverage_report
    if report is not None:
        report.clear()
        report.update({
            "direction": _archive_direction(direction, engine, val_engine),
            "cpu_reevaluation": False,
            "cpu_validation_reevaluation": False,
            "archive_candidates": 0,
            "cpu_evaluated": 0,
            "cpu_validation_evaluated": 0,
            "admitted_rules": 0,
            "retained_rules": 0,
            "final_eligible_rules": 0,
            "eligible_rules": [],
            "positive_contributors": {},
            "rejection_counts": {},
            "rejections": [],
            "reservations": {},
            "available_symbols": [],
            "symbol_diagnostics": {},
            "symbol_rejections": {},
            "btc_rejections": [],
            "rejected_btc_candidates": [],
        })

    def _reject(chromosome: np.ndarray, *reasons: str) -> None:
        if report is None:
            return
        clean_reasons = [str(reason) for reason in reasons if reason]
        if not clean_reasons:
            clean_reasons = ["unknown"]
        for reason in clean_reasons:
            counts = report["rejection_counts"]
            counts[reason] = int(counts.get(reason, 0)) + 1
        report["rejections"].append({
            "chromosome": np.asarray(chromosome, dtype=np.int32).ravel().tolist(),
            "reasons": clean_reasons,
        })

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
            _reject(chrom, "condition_count")
            continue
        unique_chroms.append(chrom)

    if report is not None:
        report["archive_candidates"] = len(unique_chroms)
    if not unique_chroms:
        return []

    logger.info(
        "Phase 2 [%s] pool builder: %d unique eligible chromosomes from %d archive entries",
        direction, len(unique_chroms), len(archive),
    )

    resolved_direction = _archive_direction(direction, engine, val_engine)
    train_cpu = cpu_engine or _build_cpu_archive_engine(
        engine, resolved_direction)
    valid_cpu = cpu_val_engine or _build_cpu_archive_engine(
        val_engine, resolved_direction,
    )
    if report is not None:
        valid_frame = getattr(valid_cpu, "df", None)
        if isinstance(valid_frame, pd.DataFrame) and "symbol" in valid_frame:
            report["available_symbols"] = sorted(
                {str(symbol)
                 for symbol in valid_frame["symbol"].dropna().unique()}
            )
        report["cpu_reevaluation"] = bool(train_cpu is not None)
        report["cpu_validation_reevaluation"] = bool(valid_cpu is not None)
        report["cpu_backend"] = (
            type(train_cpu).__name__ if train_cpu is not None else None
        )

    def _record_symbol_results(
        chromosome: np.ndarray,
        val_metrics: dict | None,
        *,
        admitted: bool,
        admission_reasons: list[str] | None = None,
    ) -> None:
        if report is None:
            return
        observed = _snapshot_per_symbol_metrics(val_metrics)
        symbols = set(report.get("available_symbols", [])) | set(observed)
        for symbol in sorted(symbols):
            values = observed.get(symbol, {})
            if not values or int(values.get("trade_count", 0)) <= 0:
                base_reason = "no_validation_trades"
            elif int(values.get("trade_count", 0)) < int(
                getattr(_cfg, "RB_MIN_VALID_TRADES", 6)
            ):
                base_reason = "insufficient_validation_trades"
            elif float(values.get("net_pnl", 0.0)) <= 0.0:
                base_reason = "non_positive_validation_pnl"
            else:
                base_reason = "positive_validation_contributor"

            reasons = [] if admitted else list(admission_reasons or [])
            if base_reason != "positive_validation_contributor":
                reasons.insert(0, base_reason)
            reason = "+".join(dict.fromkeys(reasons)) or "admitted"
            detail = report["symbol_diagnostics"].setdefault(
                symbol,
                {
                    "candidate_count": 0,
                    "positive_candidate_count": 0,
                    "rejected_candidate_count": 0,
                    "reason_counts": {},
                    "rejected_chromosomes": [],
                },
            )
            detail["candidate_count"] += 1
            detail["reason_counts"][reason] = int(
                detail["reason_counts"].get(reason, 0)
            ) + 1
            if admitted and base_reason == "positive_validation_contributor":
                detail["positive_candidate_count"] += 1
            if not admitted or base_reason != "positive_validation_contributor":
                detail["rejected_candidate_count"] += 1
                chromosome_list = np.asarray(
                    chromosome, dtype=np.int32,
                ).ravel().tolist()
                detail["rejected_chromosomes"].append(chromosome_list)
                rejection = report["symbol_rejections"].setdefault(symbol, [])
                rejection.append({
                    "chromosome": chromosome_list,
                    "reason": reason,
                })
                if str(symbol).upper().startswith("BTC"):
                    report["btc_rejections"].append({
                        "chromosome": chromosome_list,
                        "reason": reason,
                    })
        report["rejected_btc_candidates"] = report["btc_rejections"]

    # Final archive admission is deliberately fail-closed when either exact
    # CPU split is unavailable.  Cached GPU rows can be useful for evolution,
    # but they are not an admission certificate and may not contain symbol-
    # level validation evidence.  In particular, do not let the relaxed
    # ``PHASE2_POOL_REQUIRE_POSITIVE_SPLITS`` compatibility flag turn a
    # missing validation re-evaluation into an eligible pool entry.
    if train_cpu is None or valid_cpu is None:
        missing_reason = (
            "cpu_reevaluation_unavailable"
            if train_cpu is None
            else "cpu_validation_reevaluation_unavailable"
        )
        logger.error(
            "Phase 2 [%s] archive admission failed closed: %s",
            direction,
            missing_reason,
        )
        for chrom in unique_chroms:
            _reject(chrom, missing_reason)
            _record_symbol_results(
                chrom,
                None,
                admitted=False,
                admission_reasons=[missing_reason],
            )
        return []

    pool: list[dict] = []
    for chrom in unique_chroms:
        dense_chrom = _chromosome_for_pool_export(chrom, dont_cares)
        try:
            metrics_list = train_cpu.simulate_rule_batch(
                chromosomes=_chromosome_batch(dense_chrom),
                tp=_cfg.PHASE2_TP,
                sl=_cfg.PHASE2_SL,
                capital_pct=_cfg.PHASE2_CAPITAL_PCT,
            )
            metrics = metrics_list[0] if metrics_list else None
        except Exception as exc:
            logger.debug("Phase 2 CPU train re-evaluation failed: %s", exc)
            _reject(chrom, "cpu_train_simulation_error")
            _record_symbol_results(
                chrom,
                None,
                admitted=False,
                admission_reasons=["cpu_train_simulation_error"],
            )
            continue
        if not isinstance(metrics, dict):
            _reject(chrom, "cpu_train_metrics_missing")
            _record_symbol_results(
                chrom,
                None,
                admitted=False,
                admission_reasons=["cpu_train_metrics_missing"],
            )
            continue
        if report is not None:
            report["cpu_evaluated"] += 1

        try:
            val_list = valid_cpu.simulate_rule_batch(
                chromosomes=_chromosome_batch(dense_chrom),
                tp=_cfg.PHASE2_TP,
                sl=_cfg.PHASE2_SL,
                capital_pct=_cfg.PHASE2_CAPITAL_PCT,
            )
            val_metrics = val_list[0] if val_list else None
        except Exception as exc:
            logger.debug(
                "Phase 2 CPU validation re-evaluation failed: %s", exc)
            _reject(chrom, "cpu_validation_simulation_error")
            _record_symbol_results(
                chrom,
                None,
                admitted=False,
                admission_reasons=["cpu_validation_simulation_error"],
            )
            continue
        if not isinstance(val_metrics, dict):
            _reject(chrom, "cpu_validation_metrics_missing")
            _record_symbol_results(
                chrom,
                None,
                admitted=False,
                admission_reasons=["cpu_validation_metrics_missing"],
            )
            continue
        if report is not None:
            report["cpu_validation_evaluated"] += 1

        if not passes_pool_admission_gate(
            metrics,
            val_metrics,
            n_valid_rows=holdout_n_valid_rows,
            island_hyperparams=island_hyperparams,
        ):
            reasons = ["pool_admission_gate"]
            if val_metrics is None:
                reasons.append("validation_metrics_missing")
            else:
                if float(metrics.get("total_return_pct", 0.0)) <= 0.0:
                    reasons.append("train_return_floor")
                if float(val_metrics.get("total_return_pct", 0.0)) <= 0.0:
                    reasons.append("validation_return_floor")
                if float(metrics.get("profit_factor", 0.0)) < float(
                    getattr(_cfg, "PHASE2_PROFIT_FACTOR_FLOOR_ADMISSION", 1.15)
                ):
                    reasons.append("train_pf_floor")
                if float(val_metrics.get("profit_factor", 0.0)) < float(
                    getattr(_cfg, "PHASE2_PROFIT_FACTOR_FLOOR_ADMISSION", 1.15)
                ):
                    reasons.append("validation_pf_floor")
            _reject(chrom, *reasons)
            _record_symbol_results(
                chrom,
                val_metrics,
                admitted=False,
                admission_reasons=reasons,
            )
            continue

        if (
            cv_fold_evaluator is not None
            and bool(getattr(_cfg, "PURGED_WF_REQUIRE_ALL_CV_FOLDS", False))
        ):
            try:
                cv_metrics_list = cv_fold_evaluator.simulate_rule_batch(
                    chromosomes=_chromosome_batch(chrom),
                    tp=_cfg.PHASE2_TP,
                    sl=_cfg.PHASE2_SL,
                    capital_pct=_cfg.PHASE2_CAPITAL_PCT,
                )
                cv_summary = cv_metrics_list[0] if cv_metrics_list else {}
            except Exception:
                cv_summary = {}
            if not passes_pool_admission_gate(
                metrics,
                cv_summary,
                n_valid_rows=cv_fold_evaluator.n_valid_rows,
                island_hyperparams=island_hyperparams,
            ):
                _reject(chrom, "cv_admission_gate")
                _record_symbol_results(
                    chrom,
                    val_metrics,
                    admitted=False,
                    admission_reasons=["cv_admission_gate"],
                )
                continue

        executed = int(metrics.get("executed_trades", 0))
        if not passes_pool_trade_floor(
            executed,
            metrics,
            n_rows=train_n_rows,
            island_hyperparams=island_hyperparams,
        ):
            _reject(chrom, "train_trade_floor")
            _record_symbol_results(
                chrom,
                val_metrics,
                admitted=False,
                admission_reasons=["train_trade_floor"],
            )
            continue

        try:
            conditions = decode_chromosome(dense_chrom, feature_infos)
        except Exception:
            _reject(chrom, "condition_decode_error")
            _record_symbol_results(
                chrom,
                val_metrics,
                admitted=False,
                admission_reasons=["condition_decode_error"],
            )
            continue
        if not conditions:
            _reject(chrom, "empty_conditions")
            _record_symbol_results(
                chrom,
                val_metrics,
                admitted=False,
                admission_reasons=["empty_conditions"],
            )
            continue

        # Mandatory trend-context conditions are injected after chromosome
        # decoding and are never part of the chromosome / evolved feature set.
        # MIN_CONDITIONS / MAX_CONDITIONS count only the evolved conditions
        # above; the fixed context contract is policy, not discovery.
        n_evolved_conditions = len(conditions)
        if direction in ("long", "short"):
            evolved_set = set(conditions)
            for ctx_condition in _cfg.mandatory_context_conditions(direction):
                if ctx_condition not in evolved_set:
                    conditions.append(ctx_condition)

        train_per_symbol = _snapshot_per_symbol_metrics(metrics)
        pool_entry = {
            "chromosome": dense_chrom.tolist(),
            "conditions": conditions,
            "n_evolved_conditions": n_evolved_conditions,
            "context_contract_digest": _cfg.context_contract_digest(),
            "objectives": {
                "sortino_ratio": float(metrics.get("sortino_ratio", metrics.get("total_return_pct", 0.0))),
                "total_return_pct": float(metrics.get("total_return_pct", 0.0)),
                "profit_factor": float(metrics.get("profit_factor", 0.0)),
                "max_drawdown_pct": float(metrics.get("max_drawdown_pct", 0.0)),
                "win_rate": float(metrics.get("win_rate", 0.0)),
            },
            "executed_trades": executed,
            "tp": float(_cfg.PHASE2_TP),
            "sl": float(_cfg.PHASE2_SL),
            "capital_pct": float(_cfg.PHASE2_CAPITAL_PCT),
            "admission_passed": True,
            "pool_admission_passed": True,
        }
        pool_entry["per_symbol_metrics"] = train_per_symbol
        pool_entry["train_per_symbol_metrics"] = train_per_symbol
        if val_metrics is not None:
            valid_per_symbol = _snapshot_per_symbol_metrics(val_metrics)
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
            pool_entry["val_per_symbol_metrics"] = valid_per_symbol
            pool_entry["valid_per_symbol_metrics"] = valid_per_symbol
            if report is not None:
                positive_symbols = sorted(
                    _positive_contributor_symbols(
                        {"per_symbol_metrics": valid_per_symbol},
                    )
                )
                report["eligible_rules"].append({
                    "chromosome": dense_chrom.tolist(),
                    "positive_validation_symbols": positive_symbols,
                })
                for symbol in positive_symbols:
                    symbol_report = report["positive_contributors"].setdefault(
                        symbol, {"candidate_count": 0, "chromosomes": []},
                    )
                    symbol_report["candidate_count"] += 1
                    symbol_report["chromosomes"].append(dense_chrom.tolist())
        elif report is not None:
            report["eligible_rules"].append({
                "chromosome": dense_chrom.tolist(),
                "positive_validation_symbols": [],
            })
        pool.append(pool_entry)
        _record_symbol_results(chrom, val_metrics, admitted=True)

    # --- Cap pool size while reserving positive contributors per symbol. ---
    keep_top = int(getattr(_cfg, "PHASE2_KEEP_TOP_RULES", 140))
    admitted_count = len(pool)
    if len(pool) > keep_top:
        pool = _reserve_symbol_pool_candidates(
            pool,
            keep_top=keep_top,
            coverage_report=report,
        )
        logger.info(
            "Phase 2 [%s] pool capped to %d rules with symbol reservations",
            direction, keep_top,
        )

    if report is not None:
        report["admitted_rules"] = admitted_count
        report["retained_rules"] = len(pool)
        report["final_eligible_rules"] = len(pool)
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


def _pool_entry_passes_admission(
    entry: dict,
    *,
    island_hyperparams: _cfg.IslandHyperparams | None = None,
) -> bool:
    """Check stored train/val metrics on a pool JSON entry."""
    from gpu_fuzzy_trader.phases.phase2_support import passes_pool_entry_admission

    return passes_pool_entry_admission(
        entry,
        island_hyperparams=island_hyperparams,
    )


def _filter_pool_by_admission(
    pool: list[dict],
    *,
    island_hyperparams: _cfg.IslandHyperparams | None = None,
) -> list[dict]:
    if not _cfg.PHASE2_POOL_REQUIRE_POSITIVE_SPLITS:
        return pool
    return [
        e for e in pool
        if _pool_entry_passes_admission(
            e,
            island_hyperparams=island_hyperparams,
        )
    ]


def _merge_archive_entries(
    entries: list[dict],
    max_size: int = _cfg.PHASE2_ARCHIVE_MAX_SIZE,
) -> list[dict]:
    """Deduplicate and rank archive entries, keeping the best *max_size* rules."""
    if not entries:
        return []

    deduped: dict[tuple, dict] = {}
    preserve_scope = bool(
        getattr(_cfg, "PHASE2_SYMBOL_SPECIALISTS_ENABLED", False)
    )
    for entry in entries:
        chromosome_key = tuple(
            int(v) for v in np.asarray(entry["chromosome"], dtype=np.int32).ravel().tolist()
        )
        scope_key = (
            tuple(sorted(str(s) for s in entry.get("source_symbols", [])))
            if preserve_scope
            else ()
        )
        key = (chromosome_key, scope_key)
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
        if use_sparse_slots() and dont_cares is not None and not is_sparse_chromosome(
            chrom_arr
        ):
            chrom_arr = dense_to_sparse(chrom_arr, dont_cares)
        elif use_sparse_slots() and is_sparse_chromosome(chrom_arr):
            from gpu_fuzzy_trader.phases.phase2_sparse_encoding import (
                canonicalize_slots,
            )
            chrom_arr = canonicalize_slots(chrom_arr)
        key = chromosome_key(chrom_arr)
        if key in seen:
            continue
        seen.add(key)
        rows.append(chrom_arr)

    if not rows:
        return None
    return _stack_chromosome_rows(rows)


def _deployable_archive_pool_entries(deployable_archive: dict) -> list[dict]:
    """Convert deployable archive rows into pool-shaped entries for Stage B seeding."""
    entries: list[dict] = []
    for item in deployable_archive.values():
        chrom = item.get("chromosome")
        metrics = item.get("metrics", {})
        if chrom is None or not metrics:
            continue
        chrom_list = (
            chrom.tolist() if hasattr(chrom, "tolist") else list(chrom)
        )
        entries.append({
            "chromosome": chrom_list,
            "objectives": {
                "total_return_pct": float(metrics.get("total_return_pct", 0.0)),
                "max_drawdown_pct": float(metrics.get("max_drawdown_pct", 0.0)),
                "profit_factor": float(metrics.get("profit_factor", 1.0)),
                "sortino_ratio": float(metrics.get("sortino_ratio", 0.0)),
            },
            "val_objectives": {
                "total_return_pct": float(metrics.get("val_total_return_pct", 0.0)),
                "max_drawdown_pct": float(metrics.get("val_max_drawdown_pct", 0.0)),
                "profit_factor": float(metrics.get("val_profit_factor", 1.0)),
            },
            "executed_trades": int(metrics.get("executed_trades", 0)),
            "val_executed_trades": int(metrics.get("val_executed_trades", 0)),
        })
    return entries


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
    Returns an empty set when parsing fails for a condition.  Mandatory
    trend-context columns are fixed execution policy, not evolved features,
    and are never part of the feature-compatibility check.
    """
    names: set[str] = set()
    ctx_columns = set(_cfg.CONTEXT_COLUMNS)
    for cond in conditions:
        if not isinstance(cond, str):
            continue
        left = cond.split(" IS ", 1)[0].strip()
        if left.startswith("[") and left.endswith("]") and len(left) >= 3:
            feature = left[1:-1].strip()
            if feature in ctx_columns:
                continue
            names.add(feature)
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

    if int(payload.get("version", -1)) != _ARCHIVE_SCHEMA_VERSION:
        raise ValueError(
            "Phase 2 archive schema version mismatch: "
            f"expected {_ARCHIVE_SCHEMA_VERSION}, got {payload.get('version')}: {path}"
        )

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
# Monthly-window helper (Task 13)
# ---------------------------------------------------------------------------


def _evaluate_rule_on_window(
    pool_entry: dict,
    window_df: pd.DataFrame,
    direction: str,
) -> dict:
    """Evaluate a single pool rule on a single monthly window.

    Returns the full window metrics dict.  Uses the existing
    ``CPUBacktestEngine.simulate_rule_set`` with Phase 2 static risk
    parameters (``PHASE2_TP``, ``PHASE2_SL``, ``PHASE2_CAPITAL_PCT``).
    On engine errors (missing columns, misconfigured data, type errors),
    returns ``-100.0``, which the calling gate counts as non-profitable.

    Parameters
    ----------
    pool_entry:
        Pool JSON entry with a ``conditions`` key (list of condition strings).
    window_df:
        Monthly-window DataFrame (must contain label and meta columns).
    direction:
        ``"long"`` or ``"short"``.

    Returns
    -------
    dict
        Backtest metrics, or a finite failure sentinel on error.
    """
    rule = {
        "conditions": pool_entry["conditions"],
        "tp": _cfg.PHASE2_TP,
        "sl": _cfg.PHASE2_SL,
        "capital_pct": _cfg.PHASE2_CAPITAL_PCT,
    }
    try:
        engine = CPUBacktestEngine(
            window_df,
            {},
            direction,
            fee_pct=_cfg.FEE_PCT,
        )
        return dict(engine.simulate_rule_set([rule]))
    except (KeyError, ValueError, AttributeError, TypeError):
        return {
            "total_return_pct": -100.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 100.0,
            "executed_trades": 0,
        }


def _monthly_window_metrics(
    result: object,
    pool_entry: dict,
) -> dict:
    """Normalize a window evaluator result for the monthly gate.

    The float fallback keeps small downstream integrations and older unit
    doubles compatible while production evaluation always returns the exact
    CPU metrics, including executed trade support.
    """
    if isinstance(result, dict):
        return result
    source = pool_entry if isinstance(pool_entry, dict) else {}
    try:
        return {
            "total_return_pct": float(result),
            "executed_trades": int(source.get("executed_trades", 0)),
            "profit_factor": float(source.get("profit_factor", 0.0)),
            "max_drawdown_pct": float(source.get("max_drawdown_pct", 0.0)),
        }
    except (TypeError, ValueError):
        return {
            "total_return_pct": -100.0,
            "executed_trades": 0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 100.0,
        }


# ---------------------------------------------------------------------------
# Monthly-admission gate (Task 13)
# ---------------------------------------------------------------------------


def _apply_monthly_admission_gate(
    pool: list[dict],
    monthly_windows: list[pd.DataFrame],
    direction: str,
    island_hyperparams: _cfg.IslandHyperparams | None = None,
) -> list[dict]:
    """Apply the monthly-window shadow-test gate to a pool of rules.

    Each rule is evaluated on every monthly window via
    ``_evaluate_rule_on_window``.  A month counts as good only when the rule
    has enough executed trades and its return clears
    ``PHASE2_MONTHLY_GOOD_RETURN_MIN_PCT``.  Inactive months are neutral
    evidence, never good evidence; the active-month coverage and bearish-month
    limits are hard gates alongside the good-month ratio.
    resulting per-rule diagnostics are retained in the pool artifact so a
    fail-closed outcome can be investigated without rerunning evolution.

    By default the gate is fail-closed: if every rule fails, an empty pool is
    returned. The legacy keep-original behavior is available only through
    ``PHASE2_MONTHLY_ADMISSION_FAIL_CLOSED=False``.

    Parameters
    ----------
    pool:
        List of pool-entry dicts, each with a ``conditions`` key.
    monthly_windows:
        Pre-built monthly windows (list of DataFrames) from
        ``build_monthly_windows``.
    direction:
        ``"long"`` or ``"short"`` (used for logging only).
    island_hyperparams:
        Optional island-specific hyperparams. When provided, uses its
        ``monthly_admission_min_profitable_ratio`` instead of the global config.

    Returns
    -------
        list[dict]
        Filtered pool, possibly empty when no rule passes.
    """
    if island_hyperparams is not None:
        min_profitable_ratio = float(
            getattr(island_hyperparams, "monthly_admission_min_profitable_ratio", 0.667))
    else:
        min_profitable_ratio = float(
            getattr(_cfg, "PHASE2_MONTHLY_ADMISSION_MIN_RATIO", 0.667))
    pre_filter_count = len(pool)
    profitable_ratios: list[float] = []
    active_ratios: list[float] = []
    keep: list[dict] = []
    monthly_trade_floor = max(
        1, int(getattr(_cfg, "PHASE2_MONTHLY_MIN_TRADES", 3))
    )
    min_active_ratio = float(
        getattr(_cfg, "PHASE2_MONTHLY_MIN_ACTIVE_RATIO", 0.60)
    )
    max_bearish_ratio = float(
        getattr(_cfg, "PHASE2_MONTHLY_MAX_BEARISH_RATIO", 0.50)
    )
    flat_tolerance = float(
        getattr(_cfg, "MONTHLY_FLAT_TOLERANCE_PCT", 0.50)
    )
    for entry in pool:
        ret_pcts: list[float] = []
        executed_trades: list[int] = []
        bearish = 0
        for w in monthly_windows:
            raw_metrics = _evaluate_rule_on_window(entry, w, direction)
            metrics = _monthly_window_metrics(raw_metrics, entry)
            ret = float(metrics.get("total_return_pct", -100.0))
            trades = int(metrics.get("executed_trades", 0))
            ret_pcts.append(ret)
            executed_trades.append(trades)
            if trades >= monthly_trade_floor and ret < -flat_tolerance:
                bearish += 1
        min_good_return = float(
            getattr(_cfg, "PHASE2_MONTHLY_GOOD_RETURN_MIN_PCT", 0.0))
        active = sum(t >= monthly_trade_floor for t in executed_trades)
        profitable = sum(
            1 for r, t in zip(ret_pcts, executed_trades)
            if t >= monthly_trade_floor and monthly_return_counts_as_good(
                r, min_good_return, strict_above_zero=False)
        )
        ratio = profitable / max(1, len(ret_pcts))
        active_ratio = active / max(1, len(ret_pcts))
        bearish_ratio = bearish / max(1, len(ret_pcts))
        passed = (
            ratio >= min_profitable_ratio
            and active_ratio >= min_active_ratio
            and bearish_ratio <= max_bearish_ratio
        )
        if isinstance(entry, dict):
            entry["monthly_admission"] = {
                "window_returns_pct": ret_pcts,
                "window_executed_trades": executed_trades,
                "good_windows": profitable,
                "active_windows": active,
                "bearish_windows": bearish,
                "windows": len(ret_pcts),
                "good_return_min_pct": min_good_return,
                "monthly_trade_floor": monthly_trade_floor,
                "profitable_ratio": ratio,
                "active_ratio": active_ratio,
                "bearish_ratio": bearish_ratio,
                "min_profitable_ratio": min_profitable_ratio,
                "min_active_ratio": min_active_ratio,
                "max_bearish_ratio": max_bearish_ratio,
                "passed": passed,
            }
        profitable_ratios.append(ratio)
        active_ratios.append(active_ratio)
        if passed:
            keep.append(entry)

    post_filter_count = len(keep)
    if profitable_ratios:
        median_ratio = float(np.median(profitable_ratios))
        p10_ratio = float(np.percentile(profitable_ratios, 10))
        median_active_ratio = float(np.median(active_ratios))
    else:
        median_ratio = 0.0
        p10_ratio = 0.0
        median_active_ratio = 0.0

    if post_filter_count == 0:
        if bool(getattr(_cfg, "PHASE2_MONTHLY_ADMISSION_FAIL_CLOSED", True)):
            logger.error(
                "Phase 2 [%s]: monthly-admission gate emptied the pool "
                "(%d → 0); failing closed",
                direction,
                pre_filter_count,
            )
            return []
        logger.warning(
            "Phase 2 [%s]: monthly-admission gate emptied the pool "
            "(%d → 0); retaining the legacy compatibility fallback",
            direction,
            pre_filter_count,
        )
        return list(pool)

    logger.info(
        "Phase 2 [%s]: monthly-admission gate %d → %d rules "
        "(median_profitable_ratio=%.3f, median_active_ratio=%.3f, "
        "p10=%.3f, min_ratio=%.3f, min_active=%.3f)",
        direction,
        pre_filter_count,
        post_filter_count,
        median_ratio,
        median_active_ratio,
        p10_ratio,
        min_profitable_ratio,
        min_active_ratio,
    )
    return keep


def _monthly_admission_source_df(gen: "Rule_Pool_Generator") -> pd.DataFrame | None:
    """Prefer unsampled monthly val; fall back to sampled slim val."""
    monthly = getattr(gen, "_cached_monthly_val", None)
    if monthly is not None and len(monthly) > 0:
        return monthly
    slim = getattr(gen, "_cached_slim_val", None)
    if slim is not None and len(slim) > 0:
        return slim
    return None


def _run_monthly_admission_on_pool(
    pool: list[dict],
    gen: "Rule_Pool_Generator",
) -> list[dict]:
    """Build monthly windows from unsampled val and apply the admission gate.

    Never silently skips when at least one monthly window exists. If window
    count is below ``MIN_MONTHS``, still runs the gate (degraded) with a warning.
    """
    if not bool(getattr(_cfg, "PHASE2_MONTHLY_ADMISSION_ENABLED", True)):
        return pool

    val_df = _monthly_admission_source_df(gen)
    if val_df is None:
        logger.warning(
            "Phase 2 [%s]: no val DataFrame available; skipping "
            "monthly-admission gate",
            gen.direction,
        )
        return pool

    monthly_windows = build_monthly_windows(val_df)
    min_months = (
        gen.island_hyperparams.monthly_admission_min_months
        if gen.island_hyperparams is not None
        else _cfg.PHASE2_MONTHLY_ADMISSION_MIN_MONTHS
    )
    if len(monthly_windows) == 0:
        logger.warning(
            "Phase 2 [%s]: zero monthly windows on val; skipping "
            "monthly-admission gate",
            gen.direction,
        )
        return pool

    if len(monthly_windows) < int(min_months):
        logger.warning(
            "Phase 2 [%s]: only %d monthly windows (< MIN_MONTHS=%d); "
            "running monthly-admission gate in degraded mode",
            gen.direction,
            len(monthly_windows),
            min_months,
        )

    return _apply_monthly_admission_gate(
        pool,
        monthly_windows,
        gen.direction,
        island_hyperparams=gen.island_hyperparams,
    )


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
        island_id: str | None = None,
        source_symbols: list[str] | None = None,
        island_hyperparams: _cfg.IslandHyperparams | None = None,
        island_profile: str = "global",
        reference_rows: int | None = None,
        pending_migrant_seeds: list[dict] | None = None,
        defer_warmup: bool = False,
        run_id: str | None = None,
    ) -> None:
        if direction not in ("long", "short"):
            raise ValueError(
                f"direction must be 'long' or 'short', got {direction!r}")
        if not feature_infos:
            raise ValueError("feature_infos must not be empty.")

        self.direction = direction
        self.feature_infos = feature_infos
        self.pop_size = pop_size if pop_size is not None else _cfg.PHASE2_POPULATION_SIZE
        self.n_generations = n_generations if n_generations is not None else _cfg.PHASE2_GENERATIONS
        self.seed = seed  # preserved as-is (None when not provided, per docstring)
        self._rng = np.random.default_rng(seed if seed is not None else _cfg.PHASE2_SEED)
        self._feature_signature = _archive_feature_signature(feature_infos)
        self._evolution_state = None
        self._island_history: list[dict] = []
        self._island_generations_done = 0
        self.island_id = island_id
        self.source_symbols = list(source_symbols or [])
        self.island_hyperparams = island_hyperparams
        self.island_profile = str(island_profile)
        self.reference_rows = reference_rows
        self._pending_migrant_seeds = list(pending_migrant_seeds or [])
        self._defer_warmup = defer_warmup
        self.run_id = str(run_id) if run_id else None

        self._scoped_train_df = train_df
        self._scoped_val_df = val_df
        self._cv_folds = cv_folds
        self._cv_val_evaluator: CvFoldValEvaluator | None = None
        self._holdout_n_valid_rows = (
            int(len(val_df)) if val_df is not None else None
        )

        # Build forbidden bar ranges from CV folds so the sampled training
        # slice never overlaps a CV/holdout valid region (prevents data
        # leakage into the CV fitness signal).
        from gpu_fuzzy_trader.validation.rolling_cv import build_forbidden_ranges

        forbidden_ranges = (
            build_forbidden_ranges(self._cv_folds) if self._cv_folds else []
        )
        self._forbidden_ranges = forbidden_ranges

        feature_names = [fi["name"] for fi in feature_infos]
        self._feature_names = feature_names

        # Sample training data to budget, then slim to backtest-only columns
        sample_seed = seed if seed is not None else _cfg.PHASE2_SEED
        sampled = sample_df_for_phase2(
            self._scoped_train_df,
            random_state=sample_seed,
            forbidden_ranges=forbidden_ranges,
        )

        self._train_df = slim_backtest_df(sampled, feature_names)
        self._sample_seed = sample_seed
        # Derive a distinct but deterministic validation sample seed so
        # train and validation windows do not select the same relative
        # chronological regime (Task 4).
        self._val_sample_seed = _derive_val_sample_seed(sample_seed)

        # Build feature_modes dict for engine
        self._feature_modes = {fi["name"]: fi["mode"] for fi in feature_infos}

        self._engine = None
        self._val_engine = None
        # Exact full validation evaluator used only for final pool admission.
        # Evolution keeps its sampled validation engine for throughput, while
        # archive eligibility must be based on the complete validation frame.
        self._pool_val_engine = None
        # Cached val data for rebuilds after park_engines
        self._cached_slim_val = None
        # Unsampled val (slimmed) for monthly admission — full calendar span
        self._cached_monthly_val = None
        self._build_engines()
        if self._cv_folds:
            self._cv_val_evaluator = CvFoldValEvaluator(
                self._cv_folds,
                self._feature_modes,
                self._feature_names,
                self.direction,
            )
            logger.info(
                "Phase 2 [%s]: purged CV fitness evaluator (%d folds, "
                "min_fold_valid_rows=%d)",
                self.direction,
                len(self._cv_val_evaluator._folds),
                self._cv_val_evaluator.n_valid_rows,
            )

        # Keep slimmed copy for park/unpark cycles
        self._cached_slim_train = self._train_df
        # Keep the full scoped train for per-epoch window re-sampling
        # (freed in park_engines if rotation is disabled).
        if _cfg.PHASE2_PER_EPOCH_WINDOW_ROTATION:
            self._cached_scoped_train_df = self._scoped_train_df
            self._scoped_train_df = None
        else:
            self._cached_scoped_train_df = None
            self._scoped_train_df = None
        self._scoped_val_df = None

        if self.island_hyperparams is not None:
            hp = self.island_hyperparams
            tag = f"{self.direction}"
            if self.island_id is not None:
                tag += f" cluster_{self.island_id}"
            logger.info(
                "Phase 2 [%s]: island hyperparams profile=%s rows=%d "
                "min_trade_support=%d pool_floor=%d min_profitable_symbols=%d",
                tag,
                hp.profile,
                hp.n_rows,
                hp.min_trade_support,
                hp.min_trade_pool_floor,
                hp.min_profitable_symbols,
            )

    def set_pending_migrant_seeds(self, seeds: list[dict]) -> None:
        """Inject guarded migration seeds for the next epoch."""
        self._pending_migrant_seeds = list(seeds)

    # ------------------------------------------------------------------
    # Engine construction
    # ------------------------------------------------------------------

    def _build_engines(self) -> None:
        """Build train/val backtest engines."""
        self._engine = self._build_engine()
        self._val_engine = None
        if self._scoped_val_df is not None:
            # First-time build from scoped val df
            try:
                # Keep unsampled slim val for monthly admission (full calendar span).
                # Sampled slim val is used only for joint fitness / pool engines.
                self._cached_monthly_val = slim_backtest_df(
                    self._scoped_val_df, self._feature_names,
                )
                self._pool_val_engine = CPUBacktestEngine(
                    self._cached_monthly_val,
                    self._feature_modes,
                    self.direction,
                    fee_pct=_cfg.FEE_PCT,
                )
                self._pool_val_engine.n_valid_rows = len(
                    self._cached_monthly_val
                )
                val_sampled = sample_df_for_phase2(
                    self._scoped_val_df,
                    random_state=self._val_sample_seed,
                )
                slim_val = slim_backtest_df(val_sampled, self._feature_names)
                self._val_engine = self._build_engine_for_df(slim_val)
                self._holdout_n_valid_rows = len(slim_val)
                self._val_engine.n_valid_rows = len(slim_val)
                # Snapshot to cache for rebuilds after park_engines
                self._cached_slim_val = slim_val
                if _cfg.PHASE2_JOINT_TRAIN_VAL:
                    logger.info(
                        "Phase 2 [%s]: joint train+val fitness enabled "
                        "(val_rows=%d, monthly_val_rows=%d)",
                        self.direction,
                        len(slim_val),
                        len(self._cached_monthly_val),
                    )
                else:
                    logger.info(
                        "Phase 2 [%s]: val engine built for pool admission "
                        "only (joint_train_val=False; val_rows=%d)",
                        self.direction,
                        len(slim_val),
                    )
            except Exception as exc:
                logger.warning(
                    "Phase 2 [%s]: failed to build val engine, "
                    "falling back to train-only admission: %s",
                    self.direction,
                    exc,
                )
                self._val_engine = None
                self._pool_val_engine = None
                self._cached_monthly_val = None
        elif self._cached_slim_val is not None:
            # Rebuild from cached data after park_engines (no re-sampling needed)
            try:
                self._val_engine = self._build_engine_for_df(
                    self._cached_slim_val,
                )
                if self._cached_monthly_val is not None:
                    self._pool_val_engine = CPUBacktestEngine(
                        self._cached_monthly_val,
                        self._feature_modes,
                        self.direction,
                        fee_pct=_cfg.FEE_PCT,
                    )
                    self._pool_val_engine.n_valid_rows = len(
                        self._cached_monthly_val
                    )
                self._holdout_n_valid_rows = len(self._cached_slim_val)
                self._val_engine.n_valid_rows = len(self._cached_slim_val)
                if _cfg.PHASE2_JOINT_TRAIN_VAL:
                    logger.info(
                        "Phase 2 [%s]: joint train+val fitness enabled "
                        "(val_rows=%d)",
                        self.direction,
                        len(self._cached_slim_val),
                    )
                else:
                    logger.info(
                        "Phase 2 [%s]: val engine built for pool admission "
                        "only (joint_train_val=False; val_rows=%d)",
                        self.direction,
                        len(self._cached_slim_val),
                    )
            except Exception as exc:
                logger.warning(
                    "Phase 2 [%s]: failed to build val engine, "
                    "falling back to train-only admission: %s",
                    self.direction,
                    exc,
                )
                self._val_engine = None
                self._pool_val_engine = None

        from gpu_fuzzy_trader._memory import log_memory_rss

        if not self._defer_warmup:
            from gpu_fuzzy_trader._gpu_runtime import configure_phase2_gpu_runtime

            configure_phase2_gpu_runtime(
                self._engine,
                val_engine=self._val_engine,
                cluster_id=self.island_id,
            )
        log_memory_rss(f"Phase2 [{self.direction}] engine init")

    def _rebuild_train_df(self) -> None:
        """Restore slimmed training data from cache (no re-sampling needed)."""
        self._train_df = self._cached_slim_train

    def resample_train_for_epoch(self, epoch_idx: int) -> None:
        """Re-sample training data with a per-epoch rotated window.

        Each epoch gets a different contiguous sub-window of the training
        data by deriving a deterministic seed from ``(self._sample_seed,
        epoch_idx)`` and passing it to :func:`_sample_df`.  The total row
        count is capped so the per-symbol request fits within the largest
        safe bar range, ensuring the RNG start-bar branch fires.

        When :attr:`config.PHASE2_PER_EPOCH_WINDOW_ROTATION` is ``False``,
        this is a no-op (legacy single-sample behavior preserved).

        .. note::
           The first epoch (epoch_idx=0) uses the sample taken during
           ``__init__``; this method is intended for *subsequent* epochs
           (called from the scheduler loop between ``run_epoch`` calls).
        """
        if not _cfg.PHASE2_PER_EPOCH_WINDOW_ROTATION:
            return
        if self._cached_scoped_train_df is None:
            logger.warning(
                "Phase 2 [%s]: _cached_scoped_train_df is None; "
                "cannot resample. Using cached slim train.",
                self.direction,
            )
            return

        from gpu_fuzzy_trader.phases.phase2_island_scheduler import (
            _derive_epoch_seed,
        )
        from gpu_fuzzy_trader.backtest.df_slim import slim_backtest_df

        epoch_seed = _derive_epoch_seed(self._sample_seed, epoch_idx)
        sampled = sample_df_for_phase2(
            self._cached_scoped_train_df,
            random_state=epoch_seed,
            forbidden_ranges=self._forbidden_ranges,
        )
        self._cached_slim_train = slim_backtest_df(
            sampled, self._feature_names,
        )
        logger.debug(
            "Phase 2 [%s]: resampled train window for epoch %d "
            "(sampled_rows=%d, seed=%s)",
            self.direction, epoch_idx, len(sampled), epoch_seed,
        )

    def _ensure_engines(self) -> None:
        """Rebuild engines after ``park_engines`` dropped GPU state."""
        if self._train_df is None:
            self._rebuild_train_df()
        if self._engine is not None:
            return
        self._build_engines()

    def _pool_admission_context(self) -> dict[str, Any]:
        """Return the full current-run context required for pool admission."""
        pool_val_engine = (
            getattr(self, "_pool_val_engine", None)
            if getattr(self, "_pool_val_engine", None) is not None
            else getattr(self, "_val_engine", None)
        )
        pool_valid_rows = (
            len(getattr(self, "_cached_monthly_val", None))
            if getattr(self, "_cached_monthly_val", None) is not None
            else getattr(self, "_holdout_n_valid_rows", None)
        )
        train_n_rows = (
            len(getattr(self, "_train_df", None))
            if getattr(self, "_train_df", None) is not None else None
        )
        return {
            "pool_val_engine": pool_val_engine,
            "cv_fold_evaluator": getattr(self, "_cv_val_evaluator", None),
            "holdout_n_valid_rows": pool_valid_rows,
            "train_n_rows": train_n_rows,
        }

    def _admission_evidence_metadata(self) -> dict[str, Any]:
        """Describe the frames used by this generator's admission evidence."""
        def _frame_rows(engine) -> int | None:
            frame = getattr(engine, "df", None)
            return int(len(frame)) if isinstance(frame, pd.DataFrame) else None

        context = self._pool_admission_context()
        pool_rows = (
            int(len(getattr(self, "_cached_monthly_val", None)))
            if getattr(self, "_cached_monthly_val", None) is not None
            else context["holdout_n_valid_rows"]
        )
        return {
            "run_id": getattr(self, "run_id", None),
            "direction": self.direction,
            "island_id": getattr(self, "island_id", None),
            "source_symbols": list(getattr(self, "source_symbols", [])),
            "train_rows": context["train_n_rows"],
            "fitness_validation_rows": _frame_rows(
                getattr(self, "_val_engine", None)
            ),
            "pool_validation_rows": pool_rows,
            "pool_validation_frame": (
                "full_unsampled_monthly_validation"
                if getattr(self, "_cached_monthly_val", None) is not None
                else "fallback_validation_engine"
            ),
            "cv_validation_rows": (
                int(self._cv_val_evaluator.n_valid_rows)
                if getattr(self, "_cv_val_evaluator", None) is not None
                else None
            ),
        }

    def _write_admission_report(
        self,
        coverage_report: dict[str, Any],
        *,
        final_pool_size: int,
    ) -> str:
        """Persist current-run archive admission evidence."""
        coverage_report["run_id"] = getattr(self, "run_id", None)
        coverage_report["admission_evidence"] = (
            self._admission_evidence_metadata()
        )
        coverage_report["final_pool_size"] = int(final_pool_size)
        report_root = Path(
            os.path.dirname(_resolve_pool_path(self.direction))
            or _cfg.OUTPUTS_DIR
        ) / "reports"
        island_id = getattr(self, "island_id", None)
        if island_id is None:
            suffix = "coverage"
        else:
            safe_island = "".join(
                char if char.isalnum() or char in "-_" else "_"
                for char in str(island_id)
            )
            suffix = f"island_{safe_island}_coverage"
        path = report_root / f"phase2_{self.direction}_{suffix}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(coverage_report, indent=2, default=str),
            encoding="utf-8",
        )
        return str(path)

    def park_engines(self) -> None:
        """Release GPU engines and slim in-memory data between island epochs."""
        self._engine = None
        self._val_engine = None
        self._pool_val_engine = None
        self._train_df = None
        if getattr(self, "_evolution_state", None) is not None:
            from gpu_fuzzy_trader.evolution.evox_runner import (
                trim_evolution_state_memory,
            )

            trim_evolution_state_memory(
                self._evolution_state,
                pop_size=self.pop_size,
            )
        from gpu_fuzzy_trader._memory import log_memory_rss, release_phase2_resources

        log_memory_rss(
            f"Phase2 [{self.direction}] parked",
        )
        release_phase2_resources()

    def _build_engine(self):
        """Build the selected Phase 2 backend for the sampled train frame."""
        return self._build_engine_for_df(self._train_df)

    def _prefer_cpu_phase2_backend(self) -> bool:
        """Return whether this generator should avoid allocating a JAX engine.

        ``GPUBacktestEngine`` can route a large *batch* back to the CPU, but
        constructing that wrapper still uploads its full input frame and
        initializes JAX arrays first.  On the small-RAM hosts targeted by the
        large-window CPU policy, that defeated the policy and could OOM before
        the first batch was evaluated.  Select the backend once from the train
        window instead, and use it consistently for its companion validation
        engine as well.
        """
        if not bool(getattr(_cfg, "PHASE2_GPU_CPU_ROUTE_LARGE_DATA", True)):
            return False
        min_bars = max(
            0,
            int(getattr(_cfg, "PHASE2_GPU_CPU_ROUTE_MIN_BARS", 20_000)),
        )
        max_batch = max(
            1,
            int(getattr(_cfg, "PHASE2_GPU_CPU_ROUTE_MAX_BATCH", 256)),
        )
        population_size = int(
            getattr(self, "pop_size", _cfg.PHASE2_POPULATION_SIZE)
        )
        train_df = getattr(self, "_train_df", None)
        train_rows = len(train_df) if train_df is not None else 0
        return train_rows >= min_bars and population_size <= max_batch

    @staticmethod
    def _set_island_engine_context(engine, owner) -> None:
        """Attach optional island metadata; safe when *owner* is a partial mock."""
        hp = getattr(owner, "island_hyperparams", None)
        if hp is not None:
            engine._island_hyperparams = hp
        engine._island_profile = getattr(owner, "island_profile", "global")

    def _build_engine_for_df(self, df: pd.DataFrame):
        """Build an engine on *df* using the same backend selection logic."""
        engine_kwargs: dict = {
            "fee_pct": _cfg.FEE_PCT,
        }

        from gpu_fuzzy_trader.backtest.cpu_engine import CPUBacktestEngine

        if _cfg.PHASE2_USE_GPU and Rule_Pool_Generator._prefer_cpu_phase2_backend(self):
            logger.info(
                "Phase 2 using CPUBacktestEngine before JAX allocation "
                "(large train window=%d rows, population=%d).",
                len(self._train_df) if self._train_df is not None else len(df),
                int(getattr(self, "pop_size", _cfg.PHASE2_POPULATION_SIZE)),
            )
        elif _cfg.PHASE2_USE_GPU:
            from gpu_fuzzy_trader.backtest.jax_compat import (
                get_gpu_backtest_engine_class,
            )

            GPUBacktestEngine = get_gpu_backtest_engine_class()
            if GPUBacktestEngine is not None:
                try:
                    engine = GPUBacktestEngine(
                        df,
                        self._feature_modes,
                        self.direction,
                        **engine_kwargs,
                    )
                except Exception as exc:
                    # Importing JAX successfully does not guarantee that the
                    # runtime can compile kernels (for example, a missing
                    # ptxas/nvlink toolchain).  A Phase 2 backend issue must
                    # not turn a valid fail-closed selection into a pipeline
                    # exception; use the exact CPU evaluator instead.
                    logger.warning(
                        "Phase 2 GPU engine initialization failed (%s: %s); "
                        "falling back to CPUBacktestEngine.",
                        type(exc).__name__,
                        exc,
                    )
                else:
                    Rule_Pool_Generator._set_island_engine_context(engine, self)
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
        Rule_Pool_Generator._set_island_engine_context(engine, self)
        return engine

    def _attach_island_engine_context(self, engine) -> None:
        Rule_Pool_Generator._set_island_engine_context(engine, self)

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
        rng = self._rng

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
        if not (
            self.island_id is not None
            and bool(getattr(_cfg, "PHASE2_SYMBOL_SPECIALISTS_ENABLED", False))
        ):
            try:
                loaded_pool = Rule_Pool_Generator.load_pool(
                    self.direction,
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
        coverage_report: dict[str, Any] = {}

        progress_tag = "Phase 2 [%s] NSGA-III" % self.direction
        use_two_stage = (
            bool(getattr(_cfg, "PHASE2_TWO_STAGE_ENABLED", False))
            and self.island_profile == "global"
            and self.n_generations == _cfg.PHASE2_GENERATIONS
            and self.pop_size == _cfg.PHASE2_POPULATION_SIZE
        )

        # Purged CV: fitness on aggregated CV folds; pool admission on holdout val.
        fitness_val_engine = self._val_engine
        if (
            self._cv_val_evaluator is not None
            and _cfg.PHASE2_JOINT_TRAIN_VAL
            and _cfg.split_mode_is_purged_walk_forward()
        ):
            fitness_val_engine = self._cv_val_evaluator

        admission_context = self._pool_admission_context()

        evo_kwargs = dict(
            feature_infos=self.feature_infos,
            engine=self._engine,
            pop_size=self.pop_size,
            rng=rng,
            seed_chromosomes=seed_chromosomes,
            val_engine=fitness_val_engine,
            **admission_context,
            feature_probs=feature_probs,
            init_strategy=_cfg.PHASE2_INIT_STRATEGY,
            stratum_fractions=_cfg.PHASE2_INIT_STRATUM_FRACTIONS,
            island_profile=self.island_profile,
            island_hyperparams=self.island_hyperparams,
            coverage_report=coverage_report,
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

            # --- RAM/VRAM cleanup before Stage B XLA recompile ---
            # Stage B triggers a new JAX kernel compile which spikes host RAM.
            # Clearing JAX caches and running GC here prevents Colab SIGKILL.
            import gc as _gc
            _gc.collect()
            try:
                import jax as _jax
                _jax.clear_caches()
            except Exception:
                pass
            _gc.collect()
            logger.info("Phase 2 [%s]: Stage A memory released — starting Stage B", self.direction)

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
        pool = _filter_pool_by_admission(
            pool,
            island_hyperparams=self.island_hyperparams,
        )
        if pool_before_admission != len(pool):
            logger.info(
                "Phase 2 [%s]: pool admission filter %d → %d rules "
                "(train+val return>0, PF>=%.2f)",
                self.direction,
                pool_before_admission,
                len(pool),
                _cfg.PHASE2_PROFIT_FACTOR_FLOOR_ADMISSION,
            )
        logger.info(
            "Phase 2 [%s]: merged pool %d previous + %d new → %d retained",
            self.direction,
            len(previous_pool),
            len(new_pool),
            len(pool),
        )

        # --- Monthly-window shadow-test gate ---
        # Uses unsampled validation_fitness (full calendar span). Never silent-skip
        # when ≥1 monthly window exists (degraded mode if < MIN_MONTHS).
        pool = _run_monthly_admission_on_pool(pool, self)
        # The evolution stages and the persistent pool can contribute entries
        # from more than one archive pass.  Rebuild the coverage summary from
        # the final monthly-admitted population so it describes what can
        # actually reach RB, while preserving detailed rejection diagnostics
        # collected during CPU re-evaluation.
        coverage_report["eligible_rules"] = []
        coverage_report["positive_contributors"] = {}
        for entry in pool:
            chromosome = list(entry.get("chromosome", []))
            positive_symbols = sorted(
                _positive_contributor_symbols(
                    {
                        "per_symbol_metrics": (
                            _entry_validation_per_symbol_metrics(entry)
                        )
                    }
                )
            )
            coverage_report["eligible_rules"].append({
                "chromosome": chromosome,
                "positive_validation_symbols": positive_symbols,
            })
            for symbol in positive_symbols:
                symbol_report = coverage_report[
                    "positive_contributors"
                ].setdefault(
                    symbol,
                    {"candidate_count": 0, "chromosomes": []},
                )
                symbol_report["candidate_count"] += 1
                symbol_report["chromosomes"].append(chromosome)
        coverage_report["final_eligible_rules"] = len(pool)
        coverage_report["admitted_rules"] = len(pool)
        pool = _reserve_symbol_pool_candidates(
            list(pool),
            keep_top=int(getattr(_cfg, "PHASE2_KEEP_TOP_RULES", 140)),
            coverage_report=coverage_report,
        )
        coverage_report["retained_rules"] = len(pool)

        if self.island_id is not None:
            pre_final_filter_count = len(pool)
            pool = _filter_pool_by_admission(
                list(pool),
                island_hyperparams=self.island_hyperparams,
            )
            pool = Rule_Pool_Generator._annotate_archive_entries(
                pool,
                source_symbols=self.source_symbols or None,
                direction=self.direction,
            )
            coverage_report["final_filter_input_rules"] = (
                pre_final_filter_count
            )
            coverage_report["final_filter_output_rules"] = len(pool)
            report_path = self._write_admission_report(
                coverage_report,
                final_pool_size=len(pool),
            )
            logger.info(
                "Phase 2 [%s %s]: admission evidence saved to %s",
                self.direction,
                self.island_id,
                report_path,
            )
            self._release_resources()
            return pool

        pool_path = _resolve_pool_path(self.direction)
        history_path = _resolve_history_path(self.direction)
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

        coverage_path = self._write_admission_report(
            coverage_report,
            final_pool_size=len(pool),
        )

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
            )
            archive_path = Rule_Pool_Generator._archive_path_for(
                self.direction,
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
    ) -> Optional[list[dict]]:
        """
        Load existing pool if valid, return None if missing.
        """
        if direction not in ("long", "short"):
            raise ValueError(
                f"direction must be 'long' or 'short', got {direction!r}")
        path = _resolve_pool_path(direction)
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
            "version": _ARCHIVE_SCHEMA_VERSION,
            "direction": direction,
            "feature_signature": _archive_feature_signature(feature_infos),
            "rules": rules,
        }

    @staticmethod
    def _archive_path_for(
        direction: str,
        *,
        shared: bool = False,
    ) -> str:
        if shared:
            return _cfg.phase2_shared_archive_path(direction)
        return _ARCHIVE_PATHS[direction]

    @staticmethod
    def _annotate_archive_entries(
        rules: list[dict],
        *,
        shared_archive: bool = False,
        source_symbols: list[str] | None = None,
        direction: str | None = None,
    ) -> list[dict]:
        annotated: list[dict] = []
        for entry in rules:
            row = dict(entry)
            if shared_archive:
                row["shared_archive"] = True
            if source_symbols:
                row["source_symbols"] = sorted(set(source_symbols))
            row["feature_conditions"] = feature_conditions_only(
                row.get("conditions", [])
            )
            row["phase2_rule_id"] = phase2_rule_id(
                row.get("conditions", []),
                direction=direction,
                source_symbols=row.get("source_symbols", source_symbols),
            )
            row["rule_id"] = row["phase2_rule_id"]
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
            "version": _ARCHIVE_SCHEMA_VERSION,
            "direction": direction,
            "feature_signature": _archive_feature_signature(feature_infos),
            "rules": rules,
        }

    @staticmethod
    def save_archive(
        direction: str,
        feature_infos: list[dict],
        rules: list[dict],
        *,
        shared: bool = False,
        source_symbols: list[str] | None = None,
    ) -> list[dict]:
        """Merge the latest pool into a persistent archive and write atomically."""
        path = Rule_Pool_Generator._archive_path_for(
            direction, shared=shared,
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
            shared_archive=shared,
            source_symbols=source_symbols,
            direction=direction,
        )
        payload = {
            "version": _ARCHIVE_SCHEMA_VERSION,
            "direction": direction,
            "feature_signature": _archive_feature_signature(feature_infos),
            "rules": merged,
        }
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

    def _assemble_epoch_seed_entries(self) -> list[dict]:
        """Merge local pool, symbol archive, and shared archive (dominant seeds)."""
        # Singleton specialist islands must not seed from the direction-level
        # pool written by a previous island.  That would silently turn an
        # independent BTC/ETH search back into a cross-symbol warm start.
        if (
            self.source_symbols
            and bool(getattr(_cfg, "PHASE2_SYMBOL_SPECIALISTS_ENABLED", False))
        ):
            return []
        seeds: list[dict] = []
        local_pool = Rule_Pool_Generator.load_pool(
            self.direction,
        ) or []
        seeds.extend(_filter_compatible_previous_pool(local_pool, self.feature_infos))
        return _merge_archive_entries(seeds)

    def run_epoch(
        self,
        n_generations: int | None = None,
    ) -> list[dict]:
        """Evolve this island for one scheduler epoch."""
        self._ensure_engines()
        from gpu_fuzzy_trader.evolution.evox_runner import (
            extract_deployable_migrants,
            run_phase2_evolution_epoch,
        )
        from gpu_fuzzy_trader.phases.phase2_init import build_feature_sampling_probs
        from gpu_fuzzy_trader.phases.phase2_stage import resolve_island_stage

        requested_epoch_gens = int(
            n_generations if n_generations is not None
            else _cfg.PHASE2_ISLAND_EPOCH_GENERATIONS
        )
        stage_plan = resolve_island_stage(
            self._island_generations_done,
            self.n_generations,
        )
        if stage_plan.remaining_in_stage <= 0:
            if self._evolution_state is None:
                return []
            return extract_deployable_migrants(self._evolution_state)

        epoch_gens = requested_epoch_gens
        if stage_plan.two_stage_active:
            epoch_gens = min(requested_epoch_gens,
                             stage_plan.remaining_in_stage)

        dont_cares = _get_dont_cares(self.feature_infos)
        first_epoch = self._evolution_state is None
        entering_stage_b = stage_plan.entering_stage_b
        apply_seeds = first_epoch or entering_stage_b
        # Each island epoch is a fresh plateau context: carrying the prior
        # epoch's best/streak makes every subsequent epoch instantly plateau
        # at min_gen because the streak is already 5-17 from epoch 1.
        # Reset unconditionally — Stage B seeding (below) is independent.
        # NOTE: a global "island fully converged" kill-switch, if ever needed,
        # should be a SEPARATE counter, NOT the per-epoch streak.
        reset_plateau = True

        seed_chromosomes = None
        seed_fraction = None
        if entering_stage_b and self._evolution_state is not None:
            stage_a_pool = _deployable_archive_pool_entries(
                self._evolution_state.deployable_archive,
            )
            seed_chromosomes = _stage_b_seed_chromosomes(
                stage_a_pool,
                None,
                dont_cares,
                int(_cfg.PHASE2_STAGE_B_SEED_TOP_K),
            )
            seed_fraction = float(_cfg.PHASE2_STAGE_B_SEED_FRACTION)
            logger.info(
                "Phase 2 [%s]: entering Stage B (%d gen remaining, %d seeds)",
                self.direction,
                stage_plan.remaining_in_stage,
                0 if seed_chromosomes is None else len(seed_chromosomes),
            )
        elif apply_seeds or self._pending_migrant_seeds:
            seed_entries: list[dict] = []
            if first_epoch:
                seed_entries.extend(self._assemble_epoch_seed_entries())
            migrant_seeds_present = bool(self._pending_migrant_seeds)
            if self._pending_migrant_seeds:
                seed_entries.extend(self._pending_migrant_seeds)
                self._pending_migrant_seeds = []
            if seed_entries:
                if migrant_seeds_present:
                    migrant_cap = max(
                        1,
                        int(round(self.pop_size * float(_cfg.PHASE2_MIGRATION_SEED_FRACTION))),
                    )
                    migrant_entries = [
                        e for e in seed_entries
                        if e.get("migrant_rank_score") is not None
                    ]
                    archive_entries = [
                        e for e in seed_entries
                        if e.get("migrant_rank_score") is None
                    ]
                    local_seeds = _merge_archive_entries(
                        migrant_entries[:migrant_cap] + archive_entries,
                    )
                    seed_chromosomes = _pool_seed_chromosomes(local_seeds, dont_cares)
                    seed_fraction = float(_cfg.PHASE2_MIGRATION_SEED_FRACTION)
                else:
                    local_cap = max(
                        1,
                        int(round(self.pop_size * float(_cfg.PHASE2_ARCHIVE_SEED_FRACTION))),
                    )
                    local_seeds = _merge_archive_entries(seed_entries)[:local_cap]
                    seed_chromosomes = _pool_seed_chromosomes(
                        local_seeds, dont_cares)
                    seed_fraction = (
                        float(_cfg.PHASE2_STAGE_A_ARCHIVE_SEED_FRACTION)
                        if stage_plan.two_stage_active
                        else float(_cfg.PHASE2_ARCHIVE_SEED_FRACTION)
                    )
                apply_seeds = True

        # --- H5: Clear global_metrics_cache for seeded keys; cap HoF carry-over ---
        if not first_epoch and self._evolution_state is not None and seed_chromosomes is not None:
            from gpu_fuzzy_trader.phases.phase2_sparse_encoding import chromosome_key
            seeded_keys = {chromosome_key(c) for c in seed_chromosomes}
            cache = self._evolution_state.global_metrics_cache
            for key in list(cache.keys()):
                if key in seeded_keys:
                    cache.pop(key, None)
            # Cap hall-of-fame carry-over to PHASE2_HOF_EPOCH_CARRYOVER entries
            hof = self._evolution_state.hall_of_fame
            max_carry = int(getattr(_cfg, "PHASE2_HOF_EPOCH_CARRYOVER", 10))
            if len(hof) > max_carry:
                keys = list(hof.keys())[:max_carry]
                self._evolution_state.hall_of_fame = {
                    k: hof[k] for k in keys
                }

        rng = self._rng
        feature_probs = build_feature_sampling_probs(self.feature_infos)
        tag = f"Phase 2 [{self.direction}]"
        if stage_plan.two_stage_active and stage_plan.stage is not None:
            tag += f" Stage {stage_plan.stage}"

        # --- Task 2: Refresh objectives on island resume to prevent stale fitness ---
        # Refresh on resume only when the train window changes per epoch
        # (PHASE2_PER_EPOCH_WINDOW_ROTATION=True, the default after task-1).
        # When rotation is off (legacy fixed-window), the cache is valid and
        # the wipe is wasteful — cache_hit_rate drops to 0.
        # → fixes audit finding #8 (cache refresh was wasteful in fixed-window mode)
        refresh_objectives_on_resume = (
            not first_epoch
            and bool(getattr(_cfg, "PHASE2_PER_EPOCH_WINDOW_ROTATION", True))
        )
        admission_context = self._pool_admission_context()
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
            **admission_context,
            feature_probs=feature_probs,
            init_strategy=_cfg.PHASE2_INIT_STRATEGY,
            stratum_fractions=_cfg.PHASE2_INIT_STRATUM_FRACTIONS,
            seed_fraction=seed_fraction,
            stage=stage_plan.stage if stage_plan.two_stage_active else None,
            apply_seed_chromosomes=apply_seeds,
            reset_plateau=reset_plateau,
            island_profile=self.island_profile,
            island_hyperparams=self.island_hyperparams,
            refresh_objectives_on_resume=refresh_objectives_on_resume,
        )
        for entry in epoch_history:
            if stage_plan.two_stage_active and stage_plan.stage is not None:
                entry["stage"] = stage_plan.stage
        self._island_history.extend(epoch_history)
        # Charge actual generations run (epoch_history length), not the
        # requested epoch_gens, because the evolution loop may early-stop
        # before exhausting the full budget.
        self._island_generations_done += len(epoch_history)
        return extract_deployable_migrants(self._evolution_state)

    def finalize_island(self) -> list[dict]:
        """Build, filter, and persist the final pool for this island."""
        from gpu_fuzzy_trader.evolution.evox_runner import run_phase2_evolution
        from gpu_fuzzy_trader.phases.phase2_init import build_feature_sampling_probs

        self._ensure_engines()
        rng = self._rng
        feature_probs = build_feature_sampling_probs(self.feature_infos)
        tag = f"Phase 2 [{self.direction}] finalize"
        admission_context = self._pool_admission_context()
        coverage_report: dict[str, Any] = {}

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
            **admission_context,
            feature_probs=feature_probs,
            island_profile=self.island_profile,
            island_hyperparams=self.island_hyperparams,
            coverage_report=coverage_report,
        )
        new_pool = result[0] if isinstance(result, tuple) else []
        if self.island_id is not None:
            pre_admission_count = len(new_pool)
            pool = _filter_pool_by_admission(
                list(new_pool),
                island_hyperparams=self.island_hyperparams,
            )
            post_admission_count = len(pool)
            logger.info(
                "Phase 2 [%s %s]: final admission %d -> %d rules",
                self.direction,
                self.island_id,
                pre_admission_count,
                post_admission_count,
            )

            # --- Monthly-window gate for islands ---
            pool = _run_monthly_admission_on_pool(pool, self)

            pool = Rule_Pool_Generator._annotate_archive_entries(
                pool,
                source_symbols=self.source_symbols or None,
                direction=self.direction,
            )
            coverage_report["pre_admission_rules"] = pre_admission_count
            coverage_report["monthly_gate_input_rules"] = post_admission_count
            coverage_report["final_filter_output_rules"] = len(pool)
            report_path = self._write_admission_report(
                coverage_report,
                final_pool_size=len(pool),
            )
            logger.info(
                "Phase 2 [%s %s]: admission evidence saved to %s",
                self.direction,
                self.island_id,
                report_path,
            )
            self._release_resources()
            return pool

        previous_pool = Rule_Pool_Generator.load_pool(
            self.direction,
        ) or []
        pool = _merge_archive_entries(previous_pool + list(new_pool))
        pool = _filter_pool_by_admission(
            pool,
            island_hyperparams=self.island_hyperparams,
        )
        pool = Rule_Pool_Generator._annotate_archive_entries(
            pool,
            direction=self.direction,
        )

        pool_path = _resolve_pool_path(self.direction)
        history_path = _resolve_history_path(self.direction)
        for path in (pool_path, history_path):
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
        with open(pool_path, "w", encoding="utf-8") as fh:
            json.dump(pool, fh, indent=2)
        with open(history_path, "w", encoding="utf-8") as fh:
            json.dump(self._island_history, fh, indent=2)

        Rule_Pool_Generator.save_archive(
                self.direction,
                self.feature_infos,
                pool,
            )
        self._release_resources()
        return pool

    @staticmethod
    def skip_if_valid(
        direction: str,
    ) -> Optional[list[dict]]:
        """Return loaded pool if valid, None if need to run."""
        try:
            return Rule_Pool_Generator.load_pool(direction)
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
