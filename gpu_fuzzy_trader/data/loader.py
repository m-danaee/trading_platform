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

import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.backtest.df_slim import downcast_numeric_df
from gpu_fuzzy_trader.config import (
    CONTEXT_COLUMNS,
    LABEL_COLUMNS,
    META_COLUMNS,
    TAIL_DROP_ROWS,
)
from gpu_fuzzy_trader.data.labels import compute_labels


_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")

_CONTEXT_STATE_COLUMNS = ("hwc_state", "mwc_state", "lwc_state")
_CONTEXT_PERMISSION_COLUMNS = ("tf_permission_long", "tf_permission_short")
_CONTEXT_TRIGGER_COLUMNS = (
    "lwc_pullback_reversal_long",
    "lwc_pullback_reversal_short",
)
_VALID_STATE_CODES = frozenset(_cfg.CONTEXT_STATE_CODES.values())


def validate_context_columns(df: pd.DataFrame) -> None:
    """Validate the trend-context contract on an enriched frame if present.

    Fails closed on violations if required, or returns gracefully if context
    columns are not present on raw/clean OHLCV frames.
    """
    missing = [c for c in CONTEXT_COLUMNS if c not in df.columns]
    if missing:
        if bool(getattr(_cfg, "REQUIRE_CONTEXT_COLUMNS", False)):
            raise ValueError(
                "Enriched input is missing required context columns: "
                f"{missing}."
            )
        return

    if df[list(CONTEXT_COLUMNS)].isna().any().any():
        raise ValueError(
            "Enriched input contains missing values in context columns; "
            "refusing to interpret them as zero or Range."
        )

    for col in _CONTEXT_STATE_COLUMNS:
        bad = set(df[col].dropna().unique()) - _VALID_STATE_CODES
        if bad:
            raise ValueError(
                f"Context column {col!r} contains invalid state codes {sorted(bad)}; "
                f"valid codes are {sorted(_VALID_STATE_CODES)} "
                "(-1 Bearish, 0 Range, 1 Bullish, 2 Noisy)."
            )

    for col in _CONTEXT_PERMISSION_COLUMNS + _CONTEXT_TRIGGER_COLUMNS:
        bad = set(df[col].dropna().unique()) - {0, 1}
        if bad:
            raise ValueError(
                f"Context column {col!r} must be binary (0/1), got {sorted(bad)}."
            )

    hwc = df["hwc_state"].to_numpy()
    mwc = df["mwc_state"].to_numpy()
    perm_long = df["tf_permission_long"].to_numpy()
    perm_short = df["tf_permission_short"].to_numpy()
    bullish = _cfg.CONTEXT_STATE_CODES["bullish"]
    bearish = _cfg.CONTEXT_STATE_CODES["bearish"]
    range_ = _cfg.CONTEXT_STATE_CODES["range"]

    # Must mirror the actual enrichment policy (config.CONTEXT_ALLOW_MWC_RANGE_
    # PERMISSION): a neutral MWC consolidation is a valid, intentional part of
    # the permission contract when the flag is set, not a stale carryover.
    mwc_range_allowed = bool(_cfg.CONTEXT_ALLOW_MWC_RANGE_PERMISSION)
    mwc_long_ok = (mwc == bullish) | (mwc_range_allowed & (mwc == range_))
    mwc_short_ok = (mwc == bearish) | (mwc_range_allowed & (mwc == range_))
    expected_long = ((hwc == bullish) & mwc_long_ok).astype(np.int8)
    expected_short = ((hwc == bearish) & mwc_short_ok).astype(np.int8)
    if np.any(perm_long != expected_long) or np.any(perm_short != expected_short):
        mwc_clause = (
            "mwc_state==Bullish-or-Range" if mwc_range_allowed else "mwc_state==Bullish"
        )
        mwc_clause_short = (
            "mwc_state==Bearish-or-Range" if mwc_range_allowed else "mwc_state==Bearish"
        )
        raise ValueError(
            "Context permission truth table violated: tf_permission_long must "
            f"be 1 iff hwc_state==Bullish AND {mwc_clause}, and "
            f"tf_permission_short must be 1 iff hwc_state==Bearish AND "
            f"{mwc_clause_short}."
        )
    if np.any((perm_long == 1) & (perm_short == 1)):
        raise ValueError(
            "Context permissions must be mutually exclusive: long and short "
            "cannot both be active on the same row."
        )


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
        require_context: bool | None = None,
    ) -> pd.DataFrame:
        """
        Load a CSV dataset with full preparation pipeline:

        1. Read CSV with comma separator
        2. Parse datetime column
        3. Derive labels from OHLCV when labels are not supplied
        4. Sort by (datetime, symbol)
        5. Optionally attach exact first-touch barrier outcomes
        5b. Validate the mandatory trend-context contract when the tape is
            enriched (context columns present) or ``require_context`` is set
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
        require_context:
            When *None*, context columns are validated whenever they are
            present in the CSV (enriched inputs).  When ``True`` the full
            context contract is required and rejected if missing.  When
            ``False`` the context validation is skipped entirely.

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
        # Normalize every input to the repository's UTC-naive storage
        # convention before labels, splits, or fixed candle boundaries are
        # computed. This prevents host-local timezone inference from changing
        # the temporal contract.
        df["datetime"] = pd.to_datetime(
            df["datetime"], errors="raise", utc=True
        ).dt.tz_localize(None)

        # ------------------------------------------------------------------
        # 3. Derive labels from raw OHLCV when needed
        # ------------------------------------------------------------------
        df = _ensure_labels(df)

        # ------------------------------------------------------------------
        # 4. Sort by (datetime, symbol)
        # ------------------------------------------------------------------
        df = df.sort_values(["datetime", "symbol"]).reset_index(drop=True)

        # ------------------------------------------------------------------
        # 4b. Validate mandatory trend-context contract if requested.
        # ------------------------------------------------------------------
        context_present = all(c in df.columns for c in CONTEXT_COLUMNS)
        enforce = (
            bool(getattr(_cfg, "REQUIRE_CONTEXT_COLUMNS", False))
            if require_context is None
            else bool(require_context)
        )
        if enforce or (context_present and require_context is not False):
            validate_context_columns(df)

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
    require_context: bool | None = None,
) -> pd.DataFrame:
    """Module-level wrapper around ``Data_Loader.load_dataset``."""
    return Data_Loader().load_dataset(
        path,
        feature_cols=feature_cols,
        drop_tail=drop_tail,
        include_barrier_outcomes=include_barrier_outcomes,
        require_context=require_context,
    )
