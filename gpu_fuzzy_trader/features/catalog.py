"""Deterministic, train-only rule-feature catalog for Phase 2."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config
from gpu_fuzzy_trader.features.detector import Feature_Detector


_RAW_OHLCV_FEATURES = frozenset({"open", "high", "low", "close", "volume"})
_RAW_LEVEL_FEATURE_SUFFIXES = ("_atr_14", "_kama_10")


def _is_raw_level_feature(name: str) -> bool:
    """Return true for absolute indicator levels, not bounded rule inputs."""
    normalized = str(name)
    return normalized in {"atr_14", "kama_10"} or normalized.endswith(
        _RAW_LEVEL_FEATURE_SUFFIXES,
    )


def candidate_rule_feature_columns(train_df: pd.DataFrame) -> list[str]:
    """Return deterministic columns that are safe for rule evolution."""
    allowed_ff = config.RULE_ALLOWED_FF_FEATURES
    excluded = (
        set(config.LABEL_COLUMNS)
        | set(config.META_COLUMNS)
        | set(config.INTERNAL_COLUMNS)
        | set(config.CONTEXT_COLUMNS)
    )
    if config.RULE_EXCLUDE_RAW_OHLCV:
        excluded |= _RAW_OHLCV_FEATURES

    return [
        column
        for column in train_df.columns
        if column not in excluded
        and not column.startswith("_")
        and not _is_raw_level_feature(column)
        and (
            not str(column).startswith("ff_")
            or column in allowed_ff
        )
    ]


def _remove_low_dispersion(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    threshold: float,
) -> list[str]:
    """Remove a column only when one observed value exceeds *threshold*."""
    if not feature_columns or train_df.empty:
        return []

    kept: list[str] = []
    for column in feature_columns:
        values = train_df[column].to_numpy(copy=False)
        _, counts = np.unique(values, return_counts=True)
        if float(counts.max()) / float(len(train_df)) <= threshold:
            kept.append(column)
    return kept


def build_rule_feature_specs(train_df: pd.DataFrame) -> list[dict[str, str]]:
    """Build ordered rule specs from safe train-only candidates, without labels."""
    feature_columns = _remove_low_dispersion(
        train_df,
        candidate_rule_feature_columns(train_df),
        config.RULE_DISPERSION_THRESHOLD,
    )
    modes = Feature_Detector().detect_all_modes(train_df, feature_columns)
    return [
        {"name": column, "mode": modes[column]}
        for column in feature_columns
    ]


def rule_feature_specs_from_rules(
    train_df: pd.DataFrame,
    rules: list[dict],
) -> list[dict[str, str]]:
    """Return train-derived modes for non-context features used by frozen rules."""
    excluded = (
        set(config.LABEL_COLUMNS)
        | set(config.META_COLUMNS)
        | set(config.INTERNAL_COLUMNS)
        | set(config.CONTEXT_COLUMNS)
    )
    feature_columns: list[str] = []
    seen: set[str] = set()
    for rule in rules:
        for condition in rule.get("conditions", []):
            match = re.match(r"^\s*\[([^\]]+)\]\s+is\s+", str(condition), re.I)
            if match is None:
                continue
            column = match.group(1).strip()
            if (
                not column
                or column in seen
                or column in excluded
                or column.startswith("_")
                or column not in train_df.columns
            ):
                continue
            seen.add(column)
            feature_columns.append(column)

    modes = Feature_Detector().detect_all_modes(train_df, feature_columns)
    return [
        {"name": column, "mode": modes[column]}
        for column in feature_columns
    ]
