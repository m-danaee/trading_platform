"""
data/splitter.py — Data_Splitter

Per-symbol chronological 70/30 train/validation split (holdout mode).
"""

from __future__ import annotations

import math
import logging

import pandas as pd

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.backtest.df_slim import downcast_numeric_df
from gpu_fuzzy_trader.config import TRAIN_70_PATH, VALIDATION_30_PATH

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


class Data_Splitter:
    """Chronological train/validation splitter using holdout_70_30."""

    def split_and_persist(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build train/validation DataFrames and persist to Parquet.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            ``(train_df, validation_df)``.
        """
        train_df, validation_df = _holdout_70_30_split(df)

        train_df = downcast_numeric_df(train_df)
        validation_df = downcast_numeric_df(validation_df)

        train_df.to_parquet(TRAIN_70_PATH, index=False)
        validation_df.to_parquet(VALIDATION_30_PATH, index=False)

        return train_df, validation_df


def split_and_persist(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Module-level wrapper around ``Data_Splitter.split_and_persist``."""
    return Data_Splitter().split_and_persist(df)
