"""Unit tests for NSGA-III environmental selection."""

from __future__ import annotations

import numpy as np
import pytest

from gpu_fuzzy_trader.evolution.evox_runner import (
    _EVOX_AVAILABLE,
    _nsga3_environmental_selection,
)


@pytest.mark.skipif(not _EVOX_AVAILABLE, reason="EvoX not installed")
class TestNsga3Selection:
    def test_returns_pop_size_survivors(self):
        rng = np.random.default_rng(0)
        pop_size = 12
        k = 4
        merge_pop = rng.integers(0, 3, size=(pop_size * 2, k), dtype=np.int32)
        merge_fit = rng.uniform(0, 10, size=(
            pop_size * 2, 3)).astype(np.float64)
        ref = np.eye(3, dtype=np.float64)
        feature_infos = [{"name": f"f{i}", "mode": "binary"} for i in range(k)]
        from gpu_fuzzy_trader.phases.phase2_rule_pool import _get_dont_cares

        dont_cares = _get_dont_cares(feature_infos)

        pop, fit, sel_idx = _nsga3_environmental_selection(
            merge_pop, merge_fit, ref, pop_size, feature_infos, dont_cares,
        )
        assert pop.shape == (pop_size, k)
        assert fit.shape == (pop_size, 3)
        assert len(sel_idx) == pop_size
