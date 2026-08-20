"""
data/splitter.py — Data_Splitter

Per-symbol chronological split for Phase 2, RB, and Phase 5 preparation:

- ``holdout``: single holdout split + embargo gap of ``HOLDOUT_EMBARGO_CANDLES``
  bars (active). Train/val fraction determined by ``HOLDOUT_TRAIN_FRACTION``
  (default 65/35).
- ``purged_walk_forward``: expanding CV folds + primary tail holdout with embargo (deprecated).
"""

from __future__ import annotations

import logging
import os
import hashlib
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.backtest.barrier import required_barrier_columns
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


def _file_sha256(path: str) -> str:
    """Return the content digest used to bind a split cache to its source."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _holdout_embargo_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-symbol chronological split with embargo gap.

    Train: first ``HOLDOUT_TRAIN_FRACTION`` of each symbol's bars.
    Embargo: next ``HOLDOUT_EMBARGO_CANDLES`` bars — DROPPED from scoring partitions.
    Validation: remaining bars after embargo.

    Note: Feature indicators before validation/testing should be computed on full
    tape (e.g. Data_Loader with drop_tail=False) to preserve indicator history.
    """
    embargo = int(_cfg.HOLDOUT_EMBARGO_CANDLES)
    train_parts: list[pd.DataFrame] = []
    validation_parts: list[pd.DataFrame] = []

    for _, group in df.groupby("symbol", sort=True, observed=False):
        n = len(group)
        train_end = _cfg.train_prefix_row_count(n)
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


def _holdout_embargo_split_with_mask(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Per-symbol chronological split preserving full tape with boolean scoring masks.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]
        (full_df, train_mask, embargo_mask, validation_mask)
    """
    embargo = int(_cfg.HOLDOUT_EMBARGO_CANDLES)
    n_total = len(df)
    train_mask = np.zeros(n_total, dtype=bool)
    embargo_mask = np.zeros(n_total, dtype=bool)
    val_mask = np.zeros(n_total, dtype=bool)

    # Use index positions per symbol
    for _, group in df.groupby("symbol", sort=True, observed=False):
        indices = group.index.to_numpy()
        n = len(indices)
        train_end = _cfg.train_prefix_row_count(n)
        embargo_end = min(train_end + embargo, n)

        train_mask[indices[:train_end]] = True
        embargo_mask[indices[train_end:embargo_end]] = True
        if embargo_end < n:
            val_mask[indices[embargo_end:]] = True

    return (
        df.copy(),
        pd.Series(train_mask, index=df.index, name="_train_mask"),
        pd.Series(embargo_mask, index=df.index, name="_embargo_mask"),
        pd.Series(val_mask, index=df.index, name="_val_mask"),
    )


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
    purge_rows: int = 0,
) -> pd.DataFrame:
    """Per-symbol chronological first or second half of *df*.

    ``purge_rows`` is removed on both sides of the internal boundary.  This
    is separate from the main train/validation embargo: labels at the end of
    the fitness half otherwise inspect prices that belong to selection.
    """
    parts: list[pd.DataFrame] = []
    sort_col = "datetime" if "datetime" in df.columns else None
    purge_rows = max(0, int(purge_rows))
    for _, group in df.groupby("symbol", sort=True, observed=False):
        g = group.sort_values(sort_col) if sort_col else group
        g = g.reset_index(drop=True)
        n = len(g)
        if n <= 1:
            parts.append(g if first_half else g.iloc[0:0])
            continue
        split_point = n // 2  # first half gets floor; second half gets ceil
        if first_half:
            parts.append(g.iloc[:max(0, split_point - purge_rows)])
        else:
            parts.append(g.iloc[min(n, split_point + purge_rows):])
    if not parts:
        return pd.DataFrame(columns=df.columns)
    return pd.concat(parts, ignore_index=True)


def split_validation_fitness_selection(
    validation_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split validation into purged fitness and selection halves per symbol.

    The gap is intentionally present even in ordinary holdout mode.  A
    validation label opened near the end of fitness can otherwise use future
    prices from selection, allowing the same bar path to influence both
    evolutionary scoring and downstream model selection.
    """
    purge_rows = max(
        int(getattr(_cfg, "VALIDATION_HALF_PURGE_CANDLES", 0)),
        int(getattr(_cfg, "HOLDOUT_EMBARGO_CANDLES", 0)),
        int(getattr(_cfg, "PURGED_WF_EMBARGO_CANDLES", 0)),
    )
    val_fitness = _chronological_half_split(
        validation_df, first_half=True, purge_rows=purge_rows,
    )
    val_selection = _chronological_half_split(
        validation_df, first_half=False, purge_rows=purge_rows,
    )
    return val_fitness, val_selection


def _validation_half_geometry_matches(
    validation_df: pd.DataFrame,
    val_fitness: pd.DataFrame,
    val_selection: pd.DataFrame,
) -> bool:
    """Return whether cached internal halves match the purged geometry."""
    purge_rows = max(
        int(getattr(_cfg, "VALIDATION_HALF_PURGE_CANDLES", 0)),
        int(getattr(_cfg, "HOLDOUT_EMBARGO_CANDLES", 0)),
        int(getattr(_cfg, "PURGED_WF_EMBARGO_CANDLES", 0)),
    )
    if "symbol" not in validation_df.columns:
        return False

    def positions(frame: pd.DataFrame, symbol: object) -> list[object]:
        part = frame[frame["symbol"].astype(str) == str(symbol)]
        if "_symbol_bar_index" in part.columns:
            return part.sort_values("_symbol_bar_index")["_symbol_bar_index"].tolist()
        if "datetime" in part.columns:
            return part.sort_values("datetime")["datetime"].tolist()
        return part.index.tolist()

    for symbol, group in validation_df.groupby("symbol", sort=True, observed=False):
        if "datetime" in group.columns:
            group = group.sort_values("datetime")
        elif "_symbol_bar_index" in group.columns:
            group = group.sort_values("_symbol_bar_index")
        n = len(group)
        split_point = n // 2
        expected_fitness = group.iloc[:max(0, split_point - purge_rows)]
        expected_selection = group.iloc[min(n, split_point + purge_rows):]
        if positions(val_fitness, symbol) != positions(expected_fitness, symbol):
            return False
        if positions(val_selection, symbol) != positions(expected_selection, symbol):
            return False
    return True


def load_cached_split_if_fresh() -> (
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
          pd.DataFrame, list | None] | None
):
    """Load cached split parquets when they are newer than the source CSV.

    Validates manifest ``split_mode``, ``config_fingerprint``, source content
    hash, all four parquet mtimes, and (purged mode) ``reference_rows`` against
    a fresh CSV load.

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
        source_sha256 = manifest.get("source_sha256")
        if not isinstance(source_sha256, str) or not source_sha256:
            logger.warning("Rejecting split cache: manifest has no source hash")
            return None
        if _file_sha256(csv_path) != source_sha256:
            logger.warning("Rejecting split cache: source CSV content changed")
            return None
    except OSError:
        return None

    if cache_mtime < csv_mtime:
        return None

    train_df = downcast_numeric_df(pd.read_parquet(train_path))
    val_df = downcast_numeric_df(pd.read_parquet(val_path))
    val_fitness = downcast_numeric_df(pd.read_parquet(fitness_path))
    val_selection = downcast_numeric_df(pd.read_parquet(selection_path))

    # The four parquet files are a single cache unit.  A previous test run (or
    # an interrupted copy) can leave train/validation parquets from one data
    # set beside fitness/selection parquets from another.  Mtime and the
    # split-mode fingerprint cannot detect that case, but passing the tiny
    # mismatched fitness frame into Phase 2 silently removes validation from
    # the search.  Validate the cheap structural invariants before accepting
    # a cache hit.
    required_columns = (
        set(_cfg.META_COLUMNS)
        | set(_cfg.LABEL_COLUMNS)
        | set(getattr(_cfg, "INTERNAL_COLUMNS", ()))
    )
    # Exact first-touch columns are mandatory for raw OHLCV production
    # sources.  Keep label-only compatibility caches usable for small legacy
    # fixtures that cannot generate those columns in the first place.
    try:
        source_columns = set(pd.read_csv(csv_path, nrows=0).columns)
    except (OSError, pd.errors.ParserError):
        source_columns = set()
    if {"open", "high", "low", "close", "volume"}.issubset(source_columns):
        required_columns.update(required_barrier_columns())
    cached_frames = {
        "train": train_df,
        "validation": val_df,
        "validation_fitness": val_fitness,
        "validation_selection": val_selection,
    }
    for frame_name, frame in cached_frames.items():
        missing = required_columns.difference(frame.columns)
        if missing:
            logger.warning(
                "Rejecting split cache: %s is missing required columns %s",
                frame_name,
                sorted(missing),
            )
            return None

    if not _validation_half_geometry_matches(
        val_df, val_fitness, val_selection,
    ):
        logger.warning(
            "Rejecting split cache: validation fitness/selection halves do "
            "not match the current purged geometry",
        )
        return None

    # In holdout mode the manifest's reference_rows is the prepared source
    # length.  When every symbol has a non-empty validation tail, it can be
    # checked exactly from the cached partitions: train + validation + the
    # fixed embargo per symbol.  This catches stale caches without rereading
    # and relabelling the full CSV on every normal run.
    if not _cfg.split_mode_is_purged_walk_forward():
        reference_rows = manifest.get("reference_rows")
        if reference_rows is not None and "symbol" in train_df.columns:
            train_symbols = set(train_df["symbol"].astype(str).unique())
            val_symbols = set(val_df["symbol"].astype(str).unique())
            symbols = train_symbols | val_symbols
            val_counts = val_df["symbol"].astype(str).value_counts()
            has_empty_val = any(
                int(val_counts.get(sym, 0)) == 0 for sym in symbols
            ) if symbols else False
            if has_empty_val:
                # Debug/undersampled universe where the embargo consumes the
                # entire tail for at least one symbol: the holdout geometry
                # is not recoverable from partitions alone. Fall back to a
                # lightweight exact check by rereading the CSV row count.
                try:
                    from gpu_fuzzy_trader.data.loader import Data_Loader as _DL
                    _loader = _DL()
                    _full = _loader.load_dataset(csv_path, drop_tail=False,
                                                  include_barrier_outcomes=False)
                    # dropna on labels mirrors loader's tail/label logic
                    _full = _full.dropna(subset=list(_cfg.LABEL_COLUMNS))
                    if int(reference_rows) != len(_full):
                        logger.warning(
                            "Rejecting split cache: manifest reference_rows=%s, "
                            "full CSV implies %s (debug/zero-val universe)",
                            reference_rows, len(_full),
                        )
                        return None
                except Exception as exc:
                    logger.warning(
                        "Rejecting split cache: cannot verify reference_rows "
                        "for debug/zero-val universe (%s)", exc,
                    )
                    return None
            elif symbols and all(int(val_counts.get(sym, 0)) > 0 for sym in symbols):
                expected_reference_rows = (
                    len(train_df)
                    + len(val_df)
                    + int(_cfg.HOLDOUT_EMBARGO_CANDLES) * len(symbols)
                )
                if int(reference_rows) != expected_reference_rows:
                    logger.warning(
                        "Rejecting split cache: manifest reference_rows=%s, "
                        "cached holdout geometry implies %s",
                        reference_rows,
                        expected_reference_rows,
                    )
                    return None

    cv_folds: list | None = None
    if _cfg.split_mode_is_purged_walk_forward():
        loader = Data_Loader()
        # Only the prepared row count is needed to rebuild CV fold geometry;
        # the cache-schema check above already enforces exact barriers for raw
        # OHLCV sources.  Keep this lightweight call compatible with custom
        # loader doubles used by callers/tests.
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

        try:
            source_sha256 = _file_sha256(_cfg.TRAIN_CSV_PATH)
        except OSError:
            source_sha256 = None
            logger.warning(
                "Could not hash split source %s; cache will not be reusable",
                _cfg.TRAIN_CSV_PATH,
            )

        manifest_path = write_cv_folds_manifest(
            cv_folds,
            reference_rows=len(df),
            source_sha256=source_sha256,
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
