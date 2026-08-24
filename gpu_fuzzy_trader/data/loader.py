"""
data/loader.py — Data_Loader

Stateless CSV loading with full preparation pipeline:
  1. Read CSV (comma-separated with PyArrow engine fallback)
  2. Parse datetime column
  3. Derive forward labels from OHLCV when the CSV has no labels
  4. Sort by (datetime, symbol) — matches evaluator_v5.ipynb row order
  5. Optionally attach exact first-touch barrier outcomes (cached by tape SHA-256)
  6. Drop last TAIL_DROP_ROWS rows per symbol
  7. Drop rows where any LABEL_COLUMNS value is NaN
  8. Fill NaN in feature columns with 0
  9. Compute _symbol_bar_index via groupby("symbol").cumcount()
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Mapping

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
from gpu_fuzzy_trader.research_integrity import sha256_file


_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")

_CONTEXT_STATE_COLUMNS = ("hwc_state", "mwc_state", "lwc_state")
_CONTEXT_PERMISSION_COLUMNS = ("tf_permission_long", "tf_permission_short")
_CONTEXT_TRIGGER_COLUMNS = (
    "lwc_pullback_reversal_long",
    "lwc_pullback_reversal_short",
)
_VALID_STATE_CODES = frozenset(_cfg.CONTEXT_STATE_CODES.values())
_BARRIER_CACHE_MANIFEST_VERSION = 1

logger = logging.getLogger(__name__)


def _barrier_cache_manifest_path(cache_file: Path) -> Path:
    """Return the integrity manifest path for one barrier cache file."""
    return cache_file.with_suffix(f"{cache_file.suffix}.json")


def _barrier_frame_sha256(frame: pd.DataFrame) -> str:
    """Return a stable content hash for cached barrier rows and their order."""
    columns = sorted(str(column) for column in frame.columns)
    canonical = frame.loc[:, columns]
    schema = [(column, str(canonical[column].dtype)) for column in columns]
    values = pd.util.hash_pandas_object(
        canonical,
        index=True,
    ).to_numpy().tobytes()
    return hashlib.sha256(
        json.dumps(schema, separators=(",", ":")).encode("utf-8") + values
    ).hexdigest()


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
        try:
            df = pd.read_csv(path, sep=",", engine="pyarrow")
        except Exception:
            df = pd.read_csv(path, sep=",")

        if "symbol" in df.columns and str(df["symbol"].dtype) != "category":
            df["symbol"] = df["symbol"].astype("category")

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
        duplicate_rows = df.duplicated(["datetime", "symbol"], keep=False)
        if duplicate_rows.any():
            raise ValueError(
                "Dataset contains duplicate (datetime, symbol) rows; "
                "refusing ambiguous bars."
            )

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
                    barrier_cache_filename,
                    barrier_cache_identity,
                    configured_barrier_pairs,
                    required_barrier_columns,
                )

                # Barrier outcomes can be cached per tape content hash
                cached_barrier_df = None
                cached_barrier_manifest = None
                tape_hash = None
                cache_file = None
                cache_manifest_path = None
                barrier_pairs = configured_barrier_pairs()
                expected_barrier_columns = required_barrier_columns(
                    barrier_pairs,
                )
                cache_identity = barrier_cache_identity(
                    horizon=TAIL_DROP_ROWS,
                    pairs=barrier_pairs,
                )
                try:
                    if isinstance(path, (str, Path, os.PathLike)) and os.path.exists(str(path)):
                        tape_hash = sha256_file(path)
                        cache_dir = Path(_cfg.OUTPUTS_DIR) / ".cache" / "barriers"
                        cache_file = cache_dir / barrier_cache_filename(
                            tape_hash,
                            horizon=TAIL_DROP_ROWS,
                            pairs=barrier_pairs,
                        )
                        cache_manifest_path = _barrier_cache_manifest_path(
                            cache_file,
                        )
                        if cache_file.exists() and cache_manifest_path.exists():
                            cached_barrier_df = pd.read_parquet(cache_file)
                            cached_barrier_manifest = json.loads(
                                cache_manifest_path.read_text(encoding="utf-8"),
                            )
                except Exception as exc:
                    logger.warning(
                        "Rejecting unreadable barrier cache %s (%s)",
                        cache_file,
                        exc,
                    )
                    cached_barrier_df = None
                    cached_barrier_manifest = None

                cache_is_valid = (
                    cached_barrier_df is not None
                    and isinstance(cached_barrier_manifest, dict)
                    and len(cached_barrier_df) == len(df)
                    and set(cached_barrier_df.columns)
                    == expected_barrier_columns
                    and cached_barrier_manifest.get("format_version")
                    == _BARRIER_CACHE_MANIFEST_VERSION
                    and cached_barrier_manifest.get("source_sha256") == tape_hash
                    and cached_barrier_manifest.get("cache_identity")
                    == cache_identity
                    and cached_barrier_manifest.get("row_count") == len(df)
                    and cached_barrier_manifest.get("columns")
                    == sorted(expected_barrier_columns)
                    and cached_barrier_manifest.get("content_sha256")
                    == _barrier_frame_sha256(
                        cached_barrier_df.loc[
                            :, sorted(expected_barrier_columns),
                        ],
                    )
                )
                if cache_is_valid:
                    # Attach cached barrier columns directly
                    for c in sorted(expected_barrier_columns):
                        df[c] = cached_barrier_df[c].values
                else:
                    df = attach_barrier_outcomes(
                        df,
                        horizon=TAIL_DROP_ROWS,
                        pairs=barrier_pairs,
                    )
                    if (
                        tape_hash is not None
                        and cache_file is not None
                        and cache_manifest_path is not None
                    ):
                        try:
                            cache_dir = cache_file.parent
                            cache_dir.mkdir(parents=True, exist_ok=True)
                            cache_frame = df.loc[
                                :, sorted(expected_barrier_columns),
                            ]
                            cache_manifest = {
                                "format_version": _BARRIER_CACHE_MANIFEST_VERSION,
                                "source_sha256": tape_hash,
                                "cache_identity": cache_identity,
                                "row_count": len(cache_frame),
                                "columns": sorted(expected_barrier_columns),
                                "content_sha256": _barrier_frame_sha256(
                                    cache_frame,
                                ),
                            }
                            cache_temp_path = cache_file.with_name(
                                f".{cache_file.name}.{os.getpid()}.tmp",
                            )
                            manifest_temp_path = cache_manifest_path.with_name(
                                f".{cache_manifest_path.name}.{os.getpid()}.tmp",
                            )
                            cache_frame.to_parquet(cache_temp_path, index=False)
                            os.replace(cache_temp_path, cache_file)
                            manifest_temp_path.write_text(
                                json.dumps(cache_manifest, sort_keys=True),
                                encoding="utf-8",
                            )
                            os.replace(manifest_temp_path, cache_manifest_path)
                        except Exception as exc:
                            logger.warning(
                                "Could not persist barrier cache %s (%s)",
                                cache_file,
                                exc,
                            )

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
        # 7. Drop rows where any LABEL_COLUMNS value is NaN (only when drop_tail=True)
        # ------------------------------------------------------------------
        if drop_tail:
            df = df.dropna(subset=LABEL_COLUMNS).reset_index(drop=True)

        # ------------------------------------------------------------------
        # 8. Fill NaN in feature columns with 0 (optional, controlled by config)
        # ------------------------------------------------------------------
        if bool(getattr(_cfg, "FILL_NA_WITH_ZERO", False)):
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
