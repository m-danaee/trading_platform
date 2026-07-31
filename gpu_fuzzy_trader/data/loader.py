"""
data/loader.py — Data_Loader

Stateless CSV loading with full preparation pipeline:
  1. Read CSV (comma-separated)
  2. Parse datetime column
  3. Derive forward labels from OHLCV when the CSV has no labels
  4. Sort by (datetime, symbol) — matches evaluator_v5.ipynb row order
  5. Drop last TAIL_DROP_ROWS rows per symbol
  6. Drop rows where any LABEL_COLUMNS value is NaN
  7. Fill NaN in feature columns with 0
  8. Compute _symbol_bar_index via groupby("symbol").cumcount()

No caching or persistence — caller is responsible for that.
"""

from __future__ import annotations

import pandas as pd

from gpu_fuzzy_trader.backtest.df_slim import downcast_numeric_df
from gpu_fuzzy_trader.config import (
    LABEL_COLUMNS,
    META_COLUMNS,
    TAIL_DROP_ROWS,
)
from gpu_fuzzy_trader.data.labels import compute_labels


_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


def _ensure_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Keep supplied labels or derive all labels from raw OHLCV columns.

    The original datasets stored the five forward labels in the CSV. The
    replacement datasets intentionally contain raw OHLCV plus ``ff_*``
    features instead, so labels must be generated before tail rows are
    removed. A partially labelled dataset is rejected to avoid silently
    mixing incompatible label definitions.
    """
    present = [column for column in LABEL_COLUMNS if column in df.columns]
    if len(present) == len(LABEL_COLUMNS):
        return df
    if present:
        missing = [column for column in LABEL_COLUMNS if column not in df.columns]
        raise ValueError(
            "Dataset contains only some required label columns; missing: "
            f"{missing}. Provide all labels or raw OHLCV columns."
        )

    missing_ohlcv = [column for column in _OHLCV_COLUMNS if column not in df.columns]
    if missing_ohlcv:
        raise ValueError(
            "Dataset has no label columns and cannot derive them because "
            f"OHLCV columns are missing: {missing_ohlcv}."
        )

    label_frame = compute_labels(
        df[["datetime", "symbol", *_OHLCV_COLUMNS]]
    )
    try:
        return df.merge(
            label_frame,
            on=["datetime", "symbol"],
            how="left",
            sort=False,
            validate="one_to_one",
        )
    except pd.errors.MergeError as exc:
        raise ValueError(
            "Dataset must contain at most one row per (datetime, symbol) "
            "when labels are derived from OHLCV."
        ) from exc


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
        3. Derive labels from OHLCV when labels are not supplied
        4. Sort by (datetime, symbol)
        5. Drop last TAIL_DROP_ROWS rows per symbol
        6. Drop rows where any LABEL_COLUMNS value is NaN
        7. Fill NaN in feature columns with 0
        8. Compute _symbol_bar_index per symbol

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
        # ------------------------------------------------------------------
        # 1. Read CSV
        # ------------------------------------------------------------------
        df = pd.read_csv(path, sep=",")

        # ------------------------------------------------------------------
        # 2. Parse datetime column
        # ------------------------------------------------------------------
        df["datetime"] = pd.to_datetime(df["datetime"])

        # ------------------------------------------------------------------
        # 3. Derive labels from raw OHLCV when needed
        # ------------------------------------------------------------------
        df = _ensure_labels(df)

        # ------------------------------------------------------------------
        # 4. Sort by (datetime, symbol)
        # ------------------------------------------------------------------
        df = df.sort_values(["datetime", "symbol"]).reset_index(drop=True)

        # ------------------------------------------------------------------
        # 5. Drop last TAIL_DROP_ROWS rows per symbol
        # ------------------------------------------------------------------
        # Build a boolean mask: keep all rows except the last TAIL_DROP_ROWS
        # per symbol.  Using cumcount from the tail avoids groupby/apply
        # issues with pandas 3.x index handling.
        tail_count = df.groupby(
            "symbol", observed=False,
        ).cumcount(ascending=False)
        df = df[tail_count >= TAIL_DROP_ROWS].reset_index(drop=True)

        # ------------------------------------------------------------------
        # 6. Drop rows where any LABEL_COLUMNS value is NaN
        # ------------------------------------------------------------------
        df = df.dropna(subset=LABEL_COLUMNS).reset_index(drop=True)

        # ------------------------------------------------------------------
        # 7. Fill NaN in feature columns with 0
        # ------------------------------------------------------------------
        if feature_cols is None:
            non_feature = set(LABEL_COLUMNS) | set(META_COLUMNS)
            feature_cols = [c for c in df.columns if c not in non_feature]

        # Only fill columns that actually exist in the DataFrame
        existing_feature_cols = [c for c in feature_cols if c in df.columns]
        df[existing_feature_cols] = df[existing_feature_cols].fillna(0)

        # ------------------------------------------------------------------
        # 8. Compute _symbol_bar_index per symbol (after all drops)
        # ------------------------------------------------------------------
        df["_symbol_bar_index"] = df.groupby(
            "symbol", observed=False,
        ).cumcount()

        return downcast_numeric_df(df)


# ---------------------------------------------------------------------------
# Module-level convenience function (mirrors the class interface)
# ---------------------------------------------------------------------------

def load_dataset(
    path: str,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Module-level wrapper around ``Data_Loader.load_dataset``."""
    return Data_Loader().load_dataset(path, feature_cols=feature_cols)
