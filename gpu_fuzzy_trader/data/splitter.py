"""
data/splitter.py — Data_Splitter

Per-symbol chronological split for Phases 2–4:

- ``holdout``: single holdout split + 288-bar embargo gap (active).
  Train/val fraction determined by ``HOLDOUT_TRAIN_FRACTION`` (default 65/35).
- ``purged_walk_forward``: expanding CV folds + primary tail holdout with embargo (deprecated).
"""

from __future__ import annotations

import math
import logging
import os
from typing import TYPE_CHECKING

import pandas as pd

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.backtest.df_slim import downcast_numeric_df
from gpu_fuzzy_trader.config import (
    TRAIN_70_PATH,
    VALIDATION_30_PATH,
    VALIDATION_FITNESS_PATH,
    VALIDATION_SELECTION_PATH,
)

if TYPE_CHECKING:
    from gpu_fuzzy_trader.validation.rolling_cv import PurgedFold

logger = logging.getLogger(__name__)


def _holdout_embargo_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-symbol chronological split with embargo gap.

    Train: first ``HOLDOUT_TRAIN_FRACTION`` of each symbol's bars.
    Embargo: next ``HOLDOUT_EMBARGO_CANDLES`` bars — DROPPED.
    Validation: remaining bars after embargo.
    """
    train_frac = float(_cfg.HOLDOUT_TRAIN_FRACTION)
    embargo = int(_cfg.HOLDOUT_EMBARGO_CANDLES)
    train_parts: list[pd.DataFrame] = []
    validation_parts: list[pd.DataFrame] = []

    for _, group in df.groupby("symbol", sort=True):
        n = len(group)
        train_end = math.floor(n * train_frac)
        embargo_end = min(train_end + embargo, n)
        train_parts.append(group.iloc[:train_end])
        if embargo_end < n:
            validation_parts.append(group.iloc[embargo_end:])

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


def _purged_walk_forward_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    from gpu_fuzzy_trader.validation.rolling_cv import (
        build_purged_walk_forward_folds,
        derive_primary_holdout,
    )

    _cfg.set_purged_wf_reference_rows(len(df))
    folds = build_purged_walk_forward_folds(df)
    if not folds:
        logger.warning(
            "Purged walk-forward produced no folds; falling back to holdout_embargo"
        )
        train_df, val_df = _holdout_embargo_split(df)
        return train_df, val_df, []

    train_df, val_df = derive_primary_holdout(folds)
    logger.info(
        "Purged walk-forward: %d folds, train=%d rows, holdout val=%d rows",
        len(folds),
        len(train_df),
        len(val_df),
    )
    return train_df, val_df, folds


def _chronological_half_split(
    df: pd.DataFrame,
    *,
    first_half: bool,
) -> pd.DataFrame:
    """Per-symbol chronological first or second half of *df*."""
    parts: list[pd.DataFrame] = []
    sort_col = "datetime" if "datetime" in df.columns else None
    for _, group in df.groupby("symbol", sort=True):
        g = group.sort_values(sort_col) if sort_col else group
        g = g.reset_index(drop=True)
        n = len(g)
        if n <= 1:
            parts.append(g if first_half else g.iloc[0:0])
            continue
        split_point = n // 2  # first half gets floor; second half gets ceil
        if first_half:
            parts.append(g.iloc[:split_point])
        else:
            parts.append(g.iloc[split_point:])
    if not parts:
        return pd.DataFrame(columns=df.columns)
    return pd.concat(parts, ignore_index=True)


def split_validation_fitness_selection(
    validation_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split holdout validation into fitness vs selection halves per symbol."""
    val_fitness = _chronological_half_split(validation_df, first_half=True)
    val_selection = _chronological_half_split(validation_df, first_half=False)
    return val_fitness, val_selection


def load_cached_split_if_fresh() -> (
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
          pd.DataFrame, list | None] | None
):
    """Load cached split parquets when they are newer than the source CSV.

    Validates manifest ``split_mode``, ``config_fingerprint``, all four parquet
    mtimes, and (purged mode) ``reference_rows`` against a fresh CSV load.

    Returns
    -------
    tuple or None
        ``(train, val, val_fitness, val_selection, cv_folds)`` on cache hit,
        else ``None``.
    """
    from gpu_fuzzy_trader.data.loader import Data_Loader
    from gpu_fuzzy_trader.validation.rolling_cv import (
        build_purged_walk_forward_folds,
        load_cv_folds_manifest,
        purged_config_fingerprint,
    )

    csv_path = _cfg.TRAIN_CSV_PATH
    train_path = _cfg.TRAIN_70_PATH
    val_path = _cfg.VALIDATION_30_PATH
    fitness_path = _cfg.VALIDATION_FITNESS_PATH
    selection_path = _cfg.VALIDATION_SELECTION_PATH
    manifest_path = getattr(_cfg, "CV_FOLDS_MANIFEST_PATH", "")

    required_paths = (
        csv_path,
        train_path,
        val_path,
        fitness_path,
        selection_path,
        manifest_path,
    )
    if not all(os.path.exists(p) for p in required_paths):
        return None

    try:
        csv_mtime = os.path.getmtime(csv_path)
        cache_mtime = min(
            os.path.getmtime(train_path),
            os.path.getmtime(val_path),
            os.path.getmtime(fitness_path),
            os.path.getmtime(selection_path),
            os.path.getmtime(manifest_path),
        )
        manifest = load_cv_folds_manifest(manifest_path)
        if manifest is None:
            return None
        if manifest.get("split_mode") != _cfg.SPLIT_MODE:
            return None
        if manifest.get("config_fingerprint") != purged_config_fingerprint():
            return None
    except OSError:
        return None

    if cache_mtime < csv_mtime:
        return None

    train_df = downcast_numeric_df(pd.read_parquet(train_path))
    val_df = downcast_numeric_df(pd.read_parquet(val_path))
    val_fitness = downcast_numeric_df(pd.read_parquet(fitness_path))
    val_selection = downcast_numeric_df(pd.read_parquet(selection_path))

    cv_folds: list | None = None
    if _cfg.split_mode_is_purged_walk_forward():
        loader = Data_Loader()
        train_full = loader.load_dataset(csv_path)
        ref = manifest.get("reference_rows")
        if ref is None or int(ref) != len(train_full):
            return None
        _cfg.set_purged_wf_reference_rows(len(train_full))
        folds = build_purged_walk_forward_folds(train_full)
        cv_folds = folds if folds else None

    return train_df, val_df, val_fitness, val_selection, cv_folds


class Data_Splitter:
    """Chronological train/validation splitter."""

    def split_and_persist(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, list | None]:
        """
        Build train/validation DataFrames and persist to Parquet.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, list[PurgedFold] | None]
            ``(train_df, validation_df, cv_folds)``. ``cv_folds`` is non-None
            only when ``SPLIT_MODE == purged_walk_forward``.
        """
        from gpu_fuzzy_trader.validation.rolling_cv import write_cv_folds_manifest

        mode = str(_cfg.SPLIT_MODE).strip().lower()
        cv_folds: list | None = None

        if mode == "purged_walk_forward":
            train_df, validation_df, cv_folds = _purged_walk_forward_split(df)
        else:
            train_df, validation_df = _holdout_embargo_split(df)
            _cfg.set_purged_wf_reference_rows(len(df))

        train_df = downcast_numeric_df(train_df)
        validation_df = downcast_numeric_df(validation_df)
        val_fitness_df, val_selection_df = split_validation_fitness_selection(
            validation_df,
        )
        val_fitness_df = downcast_numeric_df(val_fitness_df)
        val_selection_df = downcast_numeric_df(val_selection_df)

        train_df.to_parquet(TRAIN_70_PATH, index=False)
        validation_df.to_parquet(VALIDATION_30_PATH, index=False)
        val_fitness_df.to_parquet(VALIDATION_FITNESS_PATH, index=False)
        val_selection_df.to_parquet(VALIDATION_SELECTION_PATH, index=False)

        manifest_path = write_cv_folds_manifest(
            cv_folds,
            reference_rows=len(df),
        )
        logger.info(
            "Persisted train/validation splits and manifest=%s (mode=%s)",
            manifest_path,
            _cfg.SPLIT_MODE,
        )

        return train_df, validation_df, cv_folds


def split_and_persist(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list | None]:
    """Module-level wrapper around ``Data_Splitter.split_and_persist``."""
    return Data_Splitter().split_and_persist(df)
