
from __future__ import annotations

import pandas as pd

from gpu_fuzzy_trader.backtest.df_slim import downcast_numeric_df
from gpu_fuzzy_trader.config import (
    LABEL_COLUMNS,
    META_COLUMNS,
    TAIL_DROP_ROWS,
)


class Data_Loader:
    """Stateless data loader for the GPU-Fuzzy Trading Pipeline."""

    def load_dataset(
        self,
        path: str,
        feature_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Load a CSV dataset with full preparation pipeline:

        1. Read CSV with comma separator
        2. Parse datetime column
        3. Sort by (datetime, symbol) to match evaluator_v5
        4. Optionally drop last TAIL_DROP_ROWS rows per symbol
        5. Drop rows where any LABEL_COLUMNS value is NaN
        6. Fill NaN in feature columns with 0
        7. Compute _symbol_bar_index per symbol

        Parameters
        ----------
        path:
            Path to the CSV file.
        feature_cols:
            Explicit list of feature column names.  When *None* the feature
            columns are inferred as all columns that are neither in
            LABEL_COLUMNS nor in META_COLUMNS.

        Returns
        -------
        pd.DataFrame
            Prepared DataFrame with an additional ``_symbol_bar_index`` column.
        """
        df = pd.read_csv(path, sep=",")

        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
        df["datetime"] = df["datetime"].dt.tz_localize(None)

        df = df.sort_values(["datetime", "symbol"]).reset_index(drop=True)

        tail_count = df.groupby("symbol").cumcount(ascending=False)
        df = df[tail_count >= TAIL_DROP_ROWS].reset_index(drop=True)

        label_cols_present = [c for c in LABEL_COLUMNS if c in df.columns]
        df = df.dropna(subset=label_cols_present).reset_index(drop=True)

        if feature_cols is None:
            non_feature = set(LABEL_COLUMNS) | set(META_COLUMNS)
            feature_cols = [c for c in df.columns if c not in non_feature]

        existing_feature_cols = [c for c in feature_cols if c in df.columns]
        df[existing_feature_cols] = df[existing_feature_cols].fillna(0)

        df["_symbol_bar_index"] = df.groupby("symbol").cumcount()

        return downcast_numeric_df(df)



def load_dataset(
    path: str,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Module-level wrapper around ``Data_Loader.load_dataset``."""
    return Data_Loader().load_dataset(path, feature_cols=feature_cols)
