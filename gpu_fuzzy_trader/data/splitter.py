"""
data/splitter.py — Data_Splitter

Per-symbol chronological 75/25 train/validation split.

Algorithm:
  For each symbol independently:
    1. Rows are already sorted by datetime (guaranteed by Data_Loader).
    2. Compute split_point = floor(N * 0.75) where N = number of rows for that symbol.
    3. First split_point rows → train subset.
    4. Remaining rows → validation subset.
  Concatenate all symbols' train subsets → train_75.
  Concatenate all symbols' validation subsets → validation_25.
  Persist to TRAIN_75_PATH and VALIDATION_25_PATH (Parquet format).
  Returns (train_df, validation_df).

Design decisions:
  - Split is strictly per-symbol. A symbol with 1000 rows gets 750 train + 250
    validation, regardless of other symbols' time ranges.
  - floor(N * 0.75) is used (not round or ceil) to match the spec exactly.
  - The caller (Data_Loader) is responsible for sorting; the splitter trusts
    the incoming order.
  - Parquet persistence uses the default pyarrow engine for compatibility.
"""

from __future__ import annotations

import math

import pandas as pd

from gpu_fuzzy_trader.backtest.df_slim import downcast_numeric_df
from gpu_fuzzy_trader.config import TRAIN_75_PATH, VALIDATION_25_PATH


class Data_Splitter:
    """Per-symbol chronological 75/25 train/validation splitter."""

    def split_and_persist(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        For each symbol independently:
          - Sort rows by datetime (already sorted by loader)
          - Take first floor(N * 0.75) rows as train
          - Take remaining rows as validation
        Concatenate all symbols' train rows → train_75
        Concatenate all symbols' validation rows → validation_25
        Persist to TRAIN_75_PATH and VALIDATION_25_PATH.
        Returns (train_df, validation_df).

        Parameters
        ----------
        df:
            Prepared DataFrame as returned by ``Data_Loader.load_dataset``.
            Must contain a ``symbol`` column.  Rows within each symbol must
            already be sorted chronologically (Data_Loader guarantees this).

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            ``(train_df, validation_df)`` — concatenated across all symbols,
            with the original index reset.
        """
        train_parts: list[pd.DataFrame] = []
        validation_parts: list[pd.DataFrame] = []

        for symbol, group in df.groupby("symbol", sort=True):
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

        train_df = downcast_numeric_df(train_df)
        validation_df = downcast_numeric_df(validation_df)

        # Persist to Parquet (float32 schema for lower RAM on reload)
        train_df.to_parquet(TRAIN_75_PATH, index=False)
        validation_df.to_parquet(VALIDATION_25_PATH, index=False)

        return train_df, validation_df


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def split_and_persist(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Module-level wrapper around ``Data_Splitter.split_and_persist``."""
    return Data_Splitter().split_and_persist(df)
