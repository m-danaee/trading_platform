"""Numba warm-up for Phase 3 NSGA-II (non-gating benchmark helper)."""

from __future__ import annotations

import numpy as np
import pytest

from gpu_fuzzy_trader.evolution.numba_ops import numba_enabled
from gpu_fuzzy_trader.phases.phase3_rule_set import _non_dominated_sort


@pytest.mark.benchmark
def test_phase3_numba_non_dominated_sort_warmup() -> None:
    if not numba_enabled():
        pytest.skip("Numba disabled or unavailable")
    rng = np.random.default_rng(0)
    obj = rng.random((80, 3))
    _non_dominated_sort(obj)
    _non_dominated_sort(obj)
