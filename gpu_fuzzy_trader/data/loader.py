"""
data/loader.py — Data_Loader

Stateless CSV loading with full preparation pipeline:
  1. Read CSV (comma-separated)
  2. Parse datetime column
  3. Derive forward labels from OHLCV when the CSV has no labels
  4. Sort by (datetime, symbol) — matches evaluator_v5.ipynb row order
  5. Optionally attach exact first-touch barrier outcomes
  6. Drop last TAIL_DROP_ROWS rows per symbol
  7. Drop rows where any LABEL_COLUMNS value is NaN
  8. Fill NaN in feature columns with 0
  9. Compute _symbol_bar_index via groupby("symbol").cumcount()

No caching or persistence — caller is responsible for that.
"""

from __future__ import annotations

import pandas as pd

from gpu_fuzzy_trader import config as _cfg
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
        *,
        drop_tail: bool = True,
        include_barrier_outcomes: bool = False,
    ) -> pd.DataFrame:
        """
        Load a CSV dataset with full preparation pipeline:

        1. Read CSV with comma separator
        2. Parse datetime column
        3. Derive labels from OHLCV when labels are not supplied
        4. Sort by (datetime, symbol)
        5. Optionally attach exact first-touch barrier outcomes
        6. Optionally drop last TAIL_DROP_ROWS rows per symbol
        7. Drop rows where any LABEL_COLUMNS value is NaN
        8. Fill NaN in feature columns with 0
        9. Compute _symbol_bar_index per symbol

        Parameters
        ----------
        path:
            Path to the CSV file.
        feature_cols:
            Explicit list of feature column names.  When *None* the feature
            columns are inferred as all columns that are neither in
            LABEL_COLUMNS nor in META_COLUMNS (and, by default, raw OHLCV
            columns when ``PHASE1_EXCLUDE_RAW_OHLCV`` is enabled).
        drop_tail:
            Drop the final label horizon rows per symbol. Set to ``False``
            only when a caller needs the full source tape.
        include_barrier_outcomes:
            Attach exact first-touch return/offset columns before trimming the
            source tail.

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
        # 5. Attach exact outcomes before trimming the source tail.
        # ------------------------------------------------------------------
        if include_barrier_outcomes:
            # Label-only synthetic fixtures (used by compatibility tests and
            # legacy callers) have no OHLC tape from which a first-touch
            # result can be reconstructed.  Production train/test CSVs do,
            # and are the only inputs for which the exact contract is enabled.
            if set(_OHLCV_COLUMNS).issubset(df.columns):
                from gpu_fuzzy_trader.backtest.barrier import (
                    attach_barrier_outcomes,
                )

                df = attach_barrier_outcomes(df, horizon=TAIL_DROP_ROWS)

        # ------------------------------------------------------------------
        # 6. Drop last TAIL_DROP_ROWS rows per symbol (optional)
        # ------------------------------------------------------------------
        if drop_tail:
            # Build a boolean mask: keep all rows except the last
            # TAIL_DROP_ROWS per symbol. Using cumcount from the tail avoids
            # groupby/apply issues with pandas 3.x index handling.
            tail_count = df.groupby(
                "symbol", observed=False,
            ).cumcount(ascending=False)
            df = df[tail_count >= TAIL_DROP_ROWS].reset_index(drop=True)

        # ------------------------------------------------------------------
        # 7. Drop rows where any LABEL_COLUMNS value is NaN
        # ------------------------------------------------------------------
        df = df.dropna(subset=LABEL_COLUMNS).reset_index(drop=True)

        # ------------------------------------------------------------------
        # 8. Fill NaN in feature columns with 0
        # ------------------------------------------------------------------
        if feature_cols is None:
            non_feature = set(LABEL_COLUMNS) | set(META_COLUMNS)
            non_feature.update(c for c in df.columns if str(c).startswith("_"))
            if bool(getattr(_cfg, "PHASE1_EXCLUDE_RAW_OHLCV", True)):
                # Raw price levels are retained for barrier construction, but
                # are not evaluator feature candidates.  Explicit
                # ``feature_cols`` still overrides this default for callers
                # that intentionally need a raw column.
                non_feature.update(_OHLCV_COLUMNS)
            feature_cols = [c for c in df.columns if c not in non_feature]

        # Only fill columns that actually exist in the DataFrame
        existing_feature_cols = [c for c in feature_cols if c in df.columns]
        df[existing_feature_cols] = df[existing_feature_cols].fillna(0)

        # ------------------------------------------------------------------
        # 9. Compute _symbol_bar_index per symbol (after all drops)
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
    *,
    drop_tail: bool = True,
    include_barrier_outcomes: bool = False,
) -> pd.DataFrame:
    """Module-level wrapper around ``Data_Loader.load_dataset``."""
    return Data_Loader().load_dataset(
        path,
        feature_cols=feature_cols,
        drop_tail=drop_tail,
        include_barrier_outcomes=include_barrier_outcomes,
    )
