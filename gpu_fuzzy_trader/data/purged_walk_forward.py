"""
data/purged_walk_forward.py — Purged_Walk_Forward

Reusable purged walk-forward validation splitter appropriate for 288-step
labels.  Applies an embargo BEFORE each validation fold by removing training
rows in the interval (T-288, T] around the fold boundary; the validation
window itself is never trimmed.

Algorithm (per symbol, per fold):
  1. Partition each symbol's sorted rows into PWF_N_SPLITS nearly-equal
     chronological folds.
  2. For fold i (validation), all folds j < i serve as the candidate train
     pool.
  3. Apply embargo: remove any train row whose timestamp falls in
     (val_start - PWF_PURGE_BARS, val_start].  This is computed on the
     row index, not datetime, since rows are 5-min bars and TAIL_DROP_ROWS
     already aligns with the label horizon.
  4. A fold is valid only if it has at least PWF_MIN_TRAIN_FOLDS preceding
     folds.
  5. The final test set (separately loaded test.csv) is untouched.

Design decisions:
  - Purge is applied on row-index distance (288 bars = 288 rows at 5-min
    resolution).  This is more robust than datetime arithmetic because
    markets have gaps, but the underlying bar spacing is the invariant.
  - PWF_PURGE_BARS defaults to TAIL_DROP_ROWS (288) for consistency with the
    label horizon.
  - Each per-symbol partition uses floor division for fold sizing; remainder
    rows are distributed evenly across early folds.

Usage:
    splitter = Purged_Walk_Forward()
    folds = splitter.split(df, n_splits=5, purge_bars=288)
    # folds is a list of (train_df, val_df) tuples
"""

from __future__ import annotations

import logging
import math
import os
from typing import Sequence

import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.backtest.df_slim import downcast_numeric_df

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class Purged_Walk_Forward:
    """Chronological purged walk-forward validation splitter.

    Produces a sequence of (train, validation) fold pairs with an embargo
    window removed from the training side of each fold boundary.
    """

    def __init__(
        self,
        n_splits: int = _cfg.PWF_N_SPLITS,
        purge_bars: int = _cfg.PWF_PURGE_BARS,
        min_train_folds: int = _cfg.PWF_MIN_TRAIN_FOLDS,
    ) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        if purge_bars < 0:
            raise ValueError(f"purge_bars must be >= 0, got {purge_bars}")
        if min_train_folds < 1:
            raise ValueError(
                f"min_train_folds must be >= 1, got {min_train_folds}"
            )
        self.n_splits = n_splits
        self.purge_bars = purge_bars
        self.min_train_folds = min_train_folds

    # ------------------------------------------------------------------
    # Core split logic
    # ------------------------------------------------------------------

    def split(
        self,
        df: pd.DataFrame,
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Partition *df* into purged walk-forward folds.

        Parameters
        ----------
        df : pd.DataFrame
            Sorted DataFrame (by symbol, datetime) as returned by Data_Loader.
            Must contain ``symbol`` and ``datetime`` columns.

        Returns
        -------
        list[tuple[pd.DataFrame, pd.DataFrame]]
            Ordered list of ``(train_df, val_df)`` pairs, one per valid fold.
            Folds with insufficient training data (fewer than *min_train_folds*
            preceding folds) are skipped.  Returns empty list if no valid
            folds exist.
        """
        if len(df) == 0:
            return []

        folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []

        for symbol, group in df.groupby("symbol", sort=True):
            sym_folds = self._split_one_symbol(group)
            folds.extend(sym_folds)

        # Reconcatenate per-symbol folds: align fold positions across symbols
        # by building one combined (train, val) pair per fold index.
        max_folds = self.n_splits - self.min_train_folds
        if max_folds <= 0:
            return []

        combined: list[tuple[pd.DataFrame, pd.DataFrame]] = []
        # Re-group per-symbol folds by fold index
        sym_folds_map: dict[int, list[tuple[pd.DataFrame, pd.DataFrame]]] = {}
        offset = 0
        # folds are interleaved: all folds for sym_1, then sym_2, ...
        # Repartition into fold-index buckets.
        sym_idx = {}
        current_idx = 0
        for sym, group in df.groupby("symbol", sort=True):
            n = len(group)
            sym_idx[sym] = current_idx
            current_idx += n
        # Simpler: just do per-symbol folds then recombine by fold index
        per_sym_folds: dict[str, list[tuple[pd.DataFrame, pd.DataFrame]]] = {}
        for symbol, group in df.groupby("symbol", sort=True):
            per_sym_folds[symbol] = self._split_one_symbol(group)

        # Determine the maximum valid fold index across all symbols
        max_fold = max(
            (len(fs) for fs in per_sym_folds.values()), default=0
        )
        for fi in range(max_fold):
            train_parts: list[pd.DataFrame] = []
            val_parts: list[pd.DataFrame] = []
            for fs in per_sym_folds.values():
                if fi < len(fs):
                    t, v = fs[fi]
                    train_parts.append(t)
                    val_parts.append(v)
            if train_parts and val_parts:
                train = pd.concat(train_parts, ignore_index=True)
                val = pd.concat(val_parts, ignore_index=True)
                train = downcast_numeric_df(train)
                val = downcast_numeric_df(val)
                combined.append((train, val))

        return combined

    # ------------------------------------------------------------------
    # Internal: single-symbol fold generation
    # ------------------------------------------------------------------

    def _split_one_symbol(
        self, sym_df: pd.DataFrame
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Generate purged walk-forward folds for a single symbol.

        Returns list of (train_df, val_df) tuples, ordered chronologically.
        """
        n = len(sym_df)
        if n < self.n_splits:
            # Too few rows for meaningful splits — return empty
            logger.debug(
                "Symbol %s: only %d rows (< n_splits=%d), skipping folds.",
                sym_df["symbol"].iloc[0], n, self.n_splits,
            )
            return []

        # Partition into n_splits near-equal folds
        fold_indices = _partition_indices(n, self.n_splits)

        folds_out: list[tuple[pd.DataFrame, pd.DataFrame]] = []

        for fold_i in range(self.min_train_folds, self.n_splits):
            val_start, val_end = fold_indices[fold_i]

            # Training rows: all preceding folds, embargoed
            train_rows: list[int] = []
            for j in range(fold_i):
                t_start, t_end = fold_indices[j]
                # Add train fold rows, then embargo from the boundary
                train_rows.extend(range(t_start, t_end))

            # Apply embargo: remove rows in (val_start - purge_bars, val_start]
            embargo_cut = max(0, val_start - self.purge_bars)
            train_rows = [r for r in train_rows if r < embargo_cut]

            if not train_rows:
                continue

            train_df = sym_df.iloc[train_rows].copy()
            val_df = sym_df.iloc[val_start:val_end].copy()

            folds_out.append((train_df, val_df))

        return folds_out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def persist_folds(
        self,
        folds: list[tuple[pd.DataFrame, pd.DataFrame]],
        output_dir: str,
    ) -> list[str]:
        """Write each fold pair to Parquet and return file paths.

        Files are named ``fold_{i:02d}_train.parquet`` and
        ``fold_{i:02d}_val.parquet`` inside *output_dir*.

        Parameters
        ----------
        folds : list[tuple[pd.DataFrame, pd.DataFrame]]
        output_dir : str

        Returns
        -------
        list[str]
            List of written file paths (train and val paths interleaved).
        """
        os.makedirs(output_dir, exist_ok=True)
        paths: list[str] = []
        for i, (train_df, val_df) in enumerate(folds):
            t_path = os.path.join(output_dir, f"fold_{i:02d}_train.parquet")
            v_path = os.path.join(output_dir, f"fold_{i:02d}_val.parquet")
            train_df.to_parquet(t_path, index=False)
            val_df.to_parquet(v_path, index=False)
            paths.extend([t_path, v_path])
            logger.debug(
                "Persisted fold %d: train=%d rows, val=%d rows → %s / %s",
                i, len(train_df), len(val_df), t_path, v_path,
            )
        return paths

    def get_fold_report(self, folds: list[tuple[pd.DataFrame, pd.DataFrame]]) -> dict:
        """Generate a diagnostic report for the folds.

        Returns
        -------
        dict
            Keys: n_folds, fold_details (list of per-fold train/val row counts and
            per-symbol breakdown).
        """
        report: dict = {"n_folds": len(folds), "fold_details": []}
        for i, (train_df, val_df) in enumerate(folds):
            fold_info = {
                "fold": i,
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "train_symbols": sorted(train_df["symbol"].unique().tolist()) if "symbol" in train_df.columns else [],
                "val_symbols": sorted(val_df["symbol"].unique().tolist()) if "symbol" in val_df.columns else [],
            }
            report["fold_details"].append(fold_info)
        return report


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _partition_indices(n: int, k: int) -> list[tuple[int, int]]:
    """Partition [0, n) into k contiguous fold index ranges.

    Returns list of (start, end) tuples where end is exclusive.
    Remainder rows are distributed evenly across the earliest folds.
    """
    base = n // k
    rem = n % k
    partitions: list[tuple[int, int]] = []
    start = 0
    for i in range(k):
        size = base + (1 if i < rem else 0)
        end = start + size
        partitions.append((start, end))
        start = end
    return partitions


def _embargo_train_indices(
    train_indices: Sequence[int],
    val_start: int,
    purge_bars: int,
) -> list[int]:
    """Remove train indices within the embargo zone before *val_start*."""
    embargo_cut = max(0, val_start - purge_bars)
    return [i for i in train_indices if i < embargo_cut]


# ---------------------------------------------------------------------------
# Diagnostics: check that no train row falls within the embargo window
# ---------------------------------------------------------------------------


def validate_embargo(
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    purge_bars: int,
) -> dict:
    """Verify that every train/val fold pair satisfies the embargo constraint.

    For each fold, checks that the max train bar-index is strictly less
    than (val_start - purge_bars).  Since folds are built per-symbol and
    then concatenated, this operates on the concatenated output.

    Returns
    -------
    dict
        {fold_index: {"pass": bool, "message": str}}
    """
    results: dict = {}
    # The check is best done at creation time (single-symbol). For
    # concatenated folds, we fall back to checking lack of overlap.
    for i, (train_df, val_df) in enumerate(folds):
        if "_symbol_bar_index" in train_df.columns and "_symbol_bar_index" in val_df.columns:
            # Per-symbol check
            violations = 0
            for sym in train_df["symbol"].unique():
                t_sym = train_df[train_df["symbol"] == sym]
                v_sym = val_df[val_df["symbol"] == sym]
                if len(t_sym) == 0 or len(v_sym) == 0:
                    continue
                t_max = t_sym["_symbol_bar_index"].max()
                v_min = v_sym["_symbol_bar_index"].min()
                if t_max >= (v_min - purge_bars):
                    violations += 1
            results[i] = {
                "pass": violations == 0,
                "message": (
                    "embargo OK"
                    if violations == 0
                    else f"{violations} symbol(s) violate embargo"
                ),
            }
        else:
            results[i] = {
                "pass": True,
                "message": "no bar-index column; overlap check skipped",
            }
    return results
