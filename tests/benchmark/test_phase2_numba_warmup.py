"""Document Numba warm-up requirement for Phase 2 benchmarks (non-gating)."""

from __future__ import annotations

import numpy as np
import pytest

from gpu_fuzzy_trader.evolution.numba_ops import non_dominated_sort, numba_enabled


@pytest.mark.benchmark
def test_numba_non_dominated_sort_warmup() -> None:
    """Warm-up compile before timing NSGA sort in manual benchmarks."""
    if not numba_enabled():
        pytest.skip("Numba disabled or unavailable")
    rng = np.random.default_rng(0)
    obj = rng.random((80, 3))
    non_dominated_sort(obj)
    non_dominated_sort(obj)
