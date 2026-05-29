"""
Purged chronological rolling folds (per symbol) with label-horizon embargo.

Used by Phase 2/3 when ``SPLIT_MODE == "purged_rolling_cv"``. Phase 4/5 still
use the last fold's train/val blocks persisted as train_75 / validation_25.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config as _cfg


@dataclass(frozen=True)
class PurgedFold:
    """One expanding-window fold: train is strictly before validation (purged)."""

    fold_index: int
    train_df: pd.DataFrame
    val_df: pd.DataFrame


def cv_min_train_bars() -> int:
    """Minimum training rows per symbol per fold (default ≈ 2 calendar months)."""
    months = float(_cfg.CV_MIN_TRAIN_MONTHS)
    return int(months * 30.0 * float(_cfg.CV_BARS_PER_DAY))


def cv_embargo_bars() -> int:
    return int(_cfg.CV_EMBARGO_BARS)


def _symbol_fold_slices(
    n: int,
    *,
    n_folds: int,
    embargo: int,
    min_train: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Build (train_indices, val_indices) per fold for one symbol of length *n*.

    Evaluation region starts at ``min_train``. That region is split into
  ``n_folds`` contiguous validation segments. Training for fold *i* is
    ``[0, val_start - embargo)`` (expanding window, purged).
    """
    # First validation window must start after min_train + embargo so the
    # purged training prefix still has at least ``min_train`` rows.
    if n < min_train + embargo + 2:
        return []

    eval_start = min_train + embargo
    eval_len = n - eval_start
    if eval_len < n_folds:
        return []

    boundaries = np.linspace(eval_start, n, num=n_folds + 1, dtype=int)
    out: list[tuple[np.ndarray, np.ndarray]] = []

    for i in range(n_folds):
        vs = int(boundaries[i])
        ve = int(boundaries[i + 1])
        if ve <= vs:
            continue

        purge_end = max(0, vs - embargo)
        if purge_end < min_train:
            continue

        train_idx = np.arange(0, purge_end, dtype=np.int64)
        val_idx = np.arange(vs, ve, dtype=np.int64)
        if len(train_idx) < min_train or len(val_idx) == 0:
            continue
        out.append((train_idx, val_idx))

    return out


def build_purged_rolling_folds(df: pd.DataFrame) -> list[PurgedFold]:
    """
    Build purged rolling folds across all symbols.

    Parameters
    ----------
    df:
        Chronologically sorted rows per symbol (as from ``Data_Loader``).

    Returns
    -------
    list[PurgedFold]
        Up to ``CV_N_FOLDS`` folds. Empty if constraints cannot be met.
    """
    if "symbol" not in df.columns:
        raise ValueError("df must contain a 'symbol' column")

    n_folds = int(_cfg.CV_N_FOLDS)
    embargo = cv_embargo_bars()
    min_train = cv_min_train_bars()

    if n_folds < 1:
        raise ValueError(f"CV_N_FOLDS must be >= 1, got {n_folds}")

    sort_col = "datetime" if "datetime" in df.columns else None
    per_symbol_slices: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}

    for symbol, group in df.groupby("symbol", sort=True):
        g = group.sort_values(sort_col) if sort_col else group
        g = g.reset_index(drop=True)
        slices = _symbol_fold_slices(
            len(g),
            n_folds=n_folds,
            embargo=embargo,
            min_train=min_train,
        )
        if len(slices) < n_folds:
            continue
        per_symbol_slices[str(symbol)] = slices

    if not per_symbol_slices:
        return []

    folds: list[PurgedFold] = []
    for fold_i in range(n_folds):
        train_parts: list[pd.DataFrame] = []
        val_parts: list[pd.DataFrame] = []
        for symbol, group in df.groupby("symbol", sort=True):
            sym = str(symbol)
            if sym not in per_symbol_slices:
                continue
            if fold_i >= len(per_symbol_slices[sym]):
                continue
            g = group.sort_values(sort_col) if sort_col else group
            g = g.reset_index(drop=True)
            train_idx, val_idx = per_symbol_slices[sym][fold_i]
            train_parts.append(g.iloc[train_idx])
            val_parts.append(g.iloc[val_idx])
        if not train_parts or not val_parts:
            continue
        folds.append(
            PurgedFold(
                fold_index=fold_i,
                train_df=pd.concat(train_parts, ignore_index=True),
                val_df=pd.concat(val_parts, ignore_index=True),
            )
        )

    return folds


def primary_holdout_from_folds(
    folds: list[PurgedFold],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Last chronological fold → (train, val) for Phase 4/5 persistence."""
    if not folds:
        raise ValueError("folds must not be empty")
    last = folds[-1]
    return last.train_df, last.val_df


def holdout_75_25_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Legacy per-symbol 75/25 chronological split."""
    import math

    train_parts: list[pd.DataFrame] = []
    validation_parts: list[pd.DataFrame] = []

    for _, group in df.groupby("symbol", sort=True):
        n = len(group)
        split_point = math.floor(n * 0.75)
        train_parts.append(group.iloc[:split_point])
        validation_parts.append(group.iloc[split_point:])

    train_df = (
        pd.concat(train_parts, ignore_index=True)
        if train_parts
        else pd.DataFrame(columns=df.columns)
    )
    validation_df = (
        pd.concat(validation_parts, ignore_index=True)
        if validation_parts
        else pd.DataFrame(columns=df.columns)
    )
    return train_df, validation_df
