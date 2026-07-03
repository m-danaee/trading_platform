"""
data/splitter.py — Data_Splitter

Per-symbol chronological split for Phases 2–4:

- ``holdout_70_30``: single 70/30 holdout (legacy).
- ``purged_walk_forward``: expanding CV folds + primary tail holdout with embargo.
"""

from __future__ import annotations

import math
import logging
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


def _holdout_70_30_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-symbol 70/30 chronological split."""
    train_parts: list[pd.DataFrame] = []
    validation_parts: list[pd.DataFrame] = []

    for _, group in df.groupby("symbol", sort=True):
        n = len(group)
        split_point = math.floor(n * 0.70)
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


def _purged_walk_forward_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    from gpu_fuzzy_trader.validation.rolling_cv import (
        build_purged_walk_forward_folds,
        derive_primary_holdout,
        write_cv_folds_manifest,
    )

    _cfg.set_purged_wf_reference_rows(len(df))
    folds = build_purged_walk_forward_folds(df)
    if not folds:
        logger.warning(
            "Purged walk-forward produced no folds; falling back to holdout_70_30"
        )
        train_df, val_df = _holdout_70_30_split(df)
        return train_df, val_df, []

    train_df, val_df = derive_primary_holdout(folds)
    manifest_path = write_cv_folds_manifest(folds, reference_rows=len(df))
    logger.info(
        "Purged walk-forward: %d folds, train=%d rows, holdout val=%d rows, manifest=%s",
        len(folds),
        len(train_df),
        len(val_df),
        manifest_path,
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
        split_point = max(1, n // 2)
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
        mode = str(_cfg.SPLIT_MODE).strip().lower()
        cv_folds: list | None = None

        if mode == "purged_walk_forward":
            train_df, validation_df, cv_folds = _purged_walk_forward_split(df)
        else:
            train_df, validation_df = _holdout_70_30_split(df)

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

        return train_df, validation_df, cv_folds


def split_and_persist(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list | None]:
    """Module-level wrapper around ``Data_Splitter.split_and_persist``."""
    return Data_Splitter().split_and_persist(df)
