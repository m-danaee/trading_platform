"""
data/splitter.py — Data_Splitter

Per-symbol chronological split for Phase 2, RB, and Phase 5 preparation.

The pipeline uses one development/validation holdout with an embargo gap.
The train/validation fraction is determined by ``HOLDOUT_TRAIN_FRACTION``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os

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

logger = logging.getLogger(__name__)


def _file_sha256(path: str) -> str:
    """Return the content digest used to bind a split cache to its source."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_split_manifest(
    path: str,
    *,
    reference_rows: int,
    source_sha256: str | None,
    artifact_sha256: dict[str, str] | None,
) -> str:
    """Persist the holdout cache contract and content hashes."""
    manifest = {
        "schema_version": 1,
        "holdout_train_fraction": float(_cfg.HOLDOUT_TRAIN_FRACTION),
        "embargo_candles": int(_cfg.HOLDOUT_EMBARGO_CANDLES),
        "purge_candles": int(
            getattr(_cfg, "VALIDATION_PURGE_CANDLES", _cfg.MAX_HOLD_CANDLES)
        ),
        "reference_rows": int(reference_rows),
        "source_sha256": source_sha256,
        "artifact_sha256": artifact_sha256,
    }
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    return path


def _load_split_manifest(path: str) -> dict | None:
    """Load a split manifest, returning ``None`` for invalid JSON."""
    try:
        with open(path, encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, ValueError):
        return None
    return manifest if isinstance(manifest, dict) else None


def _validation_purge_candles() -> int:
    """Return the single gap used between validation fitness and selection."""
    return int(
        getattr(_cfg, "VALIDATION_PURGE_CANDLES", _cfg.MAX_HOLD_CANDLES)
    )


def _configured_split_path(
    canonical_name: str,
    legacy_name: str | None,
    default: str,
) -> str:
    """Resolve a path while honoring temporary patches of either name."""
    canonical = str(getattr(_cfg, canonical_name, default))
    if legacy_name is None:
        return canonical
    legacy = str(getattr(_cfg, legacy_name, canonical))
    return legacy if legacy != default else canonical


def _split_manifest_matches_config(manifest: dict) -> bool:
    """Return whether a manifest was built with the current split geometry."""
    try:
        return (
            float(manifest["holdout_train_fraction"])
            == float(_cfg.HOLDOUT_TRAIN_FRACTION)
            and int(manifest["embargo_candles"])
            == int(_cfg.HOLDOUT_EMBARGO_CANDLES)
            and int(manifest["purge_candles"]) == _validation_purge_candles()
        )
    except (KeyError, TypeError, ValueError):
        return False


def _persisted_split_paths() -> dict[str, str]:
    """Resolve write paths while keeping old module-level patches working."""
    defaults = {
        "train": "data/development_train.parquet",
        "validation": "data/validation.parquet",
        "validation_fitness": "data/validation_fitness.parquet",
        "validation_selection": "data/validation_selection.parquet",
    }
    module_paths = {
        "train": TRAIN_70_PATH,
        "validation": VALIDATION_30_PATH,
        "validation_fitness": VALIDATION_FITNESS_PATH,
        "validation_selection": VALIDATION_SELECTION_PATH,
    }
    config_names = {
        "train": ("DEVELOPMENT_TRAIN_PATH", "TRAIN_70_PATH"),
        "validation": ("VALIDATION_PATH", "VALIDATION_30_PATH"),
        "validation_fitness": ("VALIDATION_FITNESS_PATH", None),
        "validation_selection": ("VALIDATION_SELECTION_PATH", None),
    }
    paths: dict[str, str] = {}
    for name, module_path in module_paths.items():
        if module_path != defaults[name]:
            paths[name] = module_path
            continue
        canonical_name, legacy_name = config_names[name]
        paths[name] = _configured_split_path(
            canonical_name,
            legacy_name,
            defaults[name],
        )
    return paths


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


def _chronological_half_split(
    df: pd.DataFrame,
    *,
    first_half: bool,
    purge_rows: int = 0,
) -> pd.DataFrame:
    """Per-symbol chronological first or second half of *df*.

    ``purge_rows`` is removed between the halves.  This is separate from the
    main train/validation embargo: labels at the end of the fitness half
    otherwise inspect prices that belong to selection.
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
            parts.append(g.iloc[split_point:])
    if not parts:
        return pd.DataFrame(columns=df.columns)
    return pd.concat(parts, ignore_index=True)


def split_validation_fitness_selection(
    validation_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split validation into fitness and selection halves per symbol.

    The gap is intentionally present even in ordinary holdout mode.  A
    validation label opened near the end of fitness can otherwise use future
    prices from selection, allowing the same bar path to influence both
    evolutionary scoring and downstream model selection.
    """
    purge_rows = _validation_purge_candles()
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
    """Return whether cached internal halves match the single-gap geometry."""
    purge_rows = _validation_purge_candles()
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
        expected_selection = group.iloc[split_point:]
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

    Validates the holdout geometry, source and persisted-split content hashes,
    and all four parquet mtimes.  Legacy artifact names are accepted as a
    one-release read fallback, while new writes use the development/validation
    names.

    Returns
    -------
    tuple or None
        ``(train, val, val_fitness, val_selection, cv_folds)`` on cache hit,
        else ``None``.
    """
    csv_path = _cfg.TRAIN_CSV_PATH
    train_path = _configured_split_path(
        "DEVELOPMENT_TRAIN_PATH",
        "TRAIN_70_PATH",
        "data/development_train.parquet",
    )
    val_path = _configured_split_path(
        "VALIDATION_PATH",
        "VALIDATION_30_PATH",
        "data/validation.parquet",
    )
    fitness_path = _cfg.VALIDATION_FITNESS_PATH
    selection_path = _cfg.VALIDATION_SELECTION_PATH
    manifest_path = _configured_split_path(
        "SPLIT_MANIFEST_PATH",
        None,
        "data/split_manifest.json",
    )

    current_paths = (
        csv_path,
        train_path,
        val_path,
        fitness_path,
        selection_path,
        manifest_path,
    )
    if not all(os.path.exists(p) for p in current_paths):
        legacy_paths = (
            csv_path,
            "data/train_70.parquet",
            "data/validation_30.parquet",
            fitness_path,
            selection_path,
            "data/cv_folds_manifest.json",
        )
        if all(os.path.exists(p) for p in legacy_paths):
            _, train_path, val_path, fitness_path, selection_path, manifest_path = (
                legacy_paths
            )
        else:
            return None

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
        manifest = _load_split_manifest(manifest_path)
        if not isinstance(manifest, dict):
            return None
        if not _split_manifest_matches_config(manifest):
            return None
        source_sha256 = manifest.get("source_sha256")
        if not isinstance(source_sha256, str) or not source_sha256:
            logger.warning("Rejecting split cache: manifest has no source hash")
            return None
        if _file_sha256(csv_path) != source_sha256:
            logger.warning("Rejecting split cache: source CSV content changed")
            return None
        artifact_sha256 = manifest.get("artifact_sha256")
        cache_paths = {
            "train": train_path,
            "validation": val_path,
            "validation_fitness": fitness_path,
            "validation_selection": selection_path,
        }
        if not isinstance(artifact_sha256, dict):
            logger.warning("Rejecting split cache: manifest has no split hashes")
            return None
        for artifact_name, artifact_path in cache_paths.items():
            expected_hash = artifact_sha256.get(artifact_name)
            if not isinstance(expected_hash, str) or not expected_hash:
                logger.warning(
                    "Rejecting split cache: manifest has no hash for %s",
                    artifact_name,
                )
                return None
            if _file_sha256(artifact_path) != expected_hash:
                logger.warning(
                    "Rejecting split cache: %s content changed",
                    artifact_name,
                )
                return None
    except OSError:
        return None

    if cache_mtime < csv_mtime:
        return None

    try:
        train_df = downcast_numeric_df(pd.read_parquet(train_path))
        val_df = downcast_numeric_df(pd.read_parquet(val_path))
        val_fitness = downcast_numeric_df(pd.read_parquet(fitness_path))
        val_selection = downcast_numeric_df(pd.read_parquet(selection_path))
    except Exception as exc:
        logger.warning("Rejecting split cache: cannot read persisted frames (%s)", exc)
        return None

    # The four parquet files are a single cache unit.  A previous test run (or
    # an interrupted copy) can leave train/validation parquets from one data
    # set beside fitness/selection parquets from another.  Mtime and the
    # manifest cannot detect that case, but passing the mismatched fitness frame
    # into Phase 2 silently removes validation from the search.  Validate the
    # cheap structural invariants before accepting a cache hit.
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
            "not match the current single-gap geometry",
        )
        return None

    # When every symbol has a non-empty validation tail, the holdout source
    # length is recoverable from the cached partitions: train + validation +
    # the fixed embargo per symbol.  This catches stale caches without loading
    # and relabelling the full CSV on every normal run.
    reference_rows = manifest.get("reference_rows")
    if reference_rows is not None and "symbol" in train_df.columns:
        train_symbols = set(train_df["symbol"].astype(str).unique())
        val_symbols = set(val_df["symbol"].astype(str).unique())
        symbols = train_symbols | val_symbols
        val_counts = val_df["symbol"].astype(str).value_counts()
        if symbols and all(int(val_counts.get(sym, 0)) > 0 for sym in symbols):
            expected_reference_rows = (
                len(train_df)
                + len(val_df)
                + int(_cfg.HOLDOUT_EMBARGO_CANDLES) * len(symbols)
            )
            try:
                reference_matches = int(reference_rows) == expected_reference_rows
            except (TypeError, ValueError):
                reference_matches = False
            if not reference_matches:
                logger.warning(
                    "Rejecting split cache: manifest reference_rows=%s, "
                    "cached holdout geometry implies %s",
                    reference_rows,
                    expected_reference_rows,
                )
                return None

    return train_df, val_df, val_fitness, val_selection, None


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
        tuple[pd.DataFrame, pd.DataFrame, list | None]
            ``(train_df, validation_df, cv_folds)``.  ``cv_folds`` remains in
            the return value for backwards compatibility and is always None.
        """
        cv_folds: list | None = None
        train_df, validation_df = _holdout_embargo_split(df)

        train_df = downcast_numeric_df(train_df)
        validation_df = downcast_numeric_df(validation_df)
        val_fitness_df, val_selection_df = split_validation_fitness_selection(
            validation_df,
        )
        val_fitness_df = downcast_numeric_df(val_fitness_df)
        val_selection_df = downcast_numeric_df(val_selection_df)

        persisted_paths = _persisted_split_paths()
        train_df.to_parquet(persisted_paths["train"], index=False)
        validation_df.to_parquet(persisted_paths["validation"], index=False)
        val_fitness_df.to_parquet(
            persisted_paths["validation_fitness"], index=False,
        )
        val_selection_df.to_parquet(
            persisted_paths["validation_selection"], index=False,
        )
        try:
            source_sha256 = _file_sha256(_cfg.TRAIN_CSV_PATH)
            artifact_sha256 = {
                name: _file_sha256(path)
                for name, path in persisted_paths.items()
            }
        except OSError:
            source_sha256 = None
            artifact_sha256 = None
            logger.warning(
                "Could not hash split source or artifacts; cache will not be reusable (%s)",
                _cfg.TRAIN_CSV_PATH,
            )

        manifest_path = _write_split_manifest(
            _configured_split_path(
                "SPLIT_MANIFEST_PATH",
                None,
                "data/split_manifest.json",
            ),
            reference_rows=len(df),
            source_sha256=source_sha256,
            artifact_sha256=artifact_sha256,
        )
        logger.info(
            "Persisted development/validation splits and manifest=%s",
            manifest_path,
        )

        return train_df, validation_df, cv_folds


def split_and_persist(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list | None]:
    """Module-level wrapper around ``Data_Splitter.split_and_persist``."""
    return Data_Splitter().split_and_persist(df)
