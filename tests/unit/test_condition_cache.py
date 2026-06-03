"""Unit tests for condition mask caching."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gpu_fuzzy_trader.backtest.condition_cache import get_or_build_rule_mask
from gpu_fuzzy_trader.backtest.cpu_engine import _compute_rule_signal_mask


def _tiny_df(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "feat_a": rng.uniform(0, 1, size=n),
        "feat_b": rng.uniform(-1, 1, size=n),
    })


class TestConditionCache:
    def test_cached_mask_matches_direct_build(self):
        df = _tiny_df()
        conditions = ["[feat_a] IS Very High", "[feat_b] IS Weak Positive"]
        cache: dict[tuple[str, ...], np.ndarray] = {}

        direct = _compute_rule_signal_mask(df, conditions)
        cached = get_or_build_rule_mask(df, conditions, cache)
        again = get_or_build_rule_mask(df, conditions, cache)

        assert np.array_equal(direct, cached)
        assert np.array_equal(cached, again)
        assert len(cache) == 1
