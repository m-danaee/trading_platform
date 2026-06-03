"""
condition_cache.py — Cached boolean masks for textual rule conditions.

Avoids re-parsing ``[feature] IS value`` strings on every backtest when the
same condition set is evaluated repeatedly (Phase 3/4).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def conditions_cache_key(conditions: list[str]) -> tuple[str, ...]:
    """Stable hashable key for a list of condition strings."""
    return tuple(conditions)


def get_or_build_rule_mask(
    df: "pd.DataFrame",
    conditions: list[str],
    cache: dict[tuple[str, ...], np.ndarray] | None,
) -> np.ndarray:
    """
  Return a boolean row mask for *conditions*, using *cache* when provided.

    The cache is updated in place when *cache* is not None.
    """
    from gpu_fuzzy_trader.backtest.cpu_engine import _compute_rule_signal_mask

    key = conditions_cache_key(conditions)
    if cache is not None and key in cache:
        return cache[key]
    mask = _compute_rule_signal_mask(df, conditions)
    if cache is not None:
        cache[key] = mask
    return mask
