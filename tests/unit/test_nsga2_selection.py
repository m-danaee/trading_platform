"""Unit tests for canonical NSGA-II environmental selection."""

from __future__ import annotations

import numpy as np

from gpu_fuzzy_trader.evolution.evox_runner import environmental_selection_nsga2


class TestEnvironmentalSelectionNsga2:
    def test_prefers_lower_front(self):
        # Non-dominated: 0 and 1; 2 is dominated; 3 is dominated by all
        merge_fit = np.array(
            [
                [0.0, 10.0],
                [10.0, 0.0],
                [5.0, 5.0],
                [20.0, 20.0],
            ],
            dtype=np.float64,
        )
        merge_pop = np.arange(4, dtype=np.int32)[:, None]
        pop, fit, idx = environmental_selection_nsga2(
            merge_pop, merge_fit, pop_size=2)
        assert len(idx) == 2
        assert set(idx) == {0, 1}

    def test_crowding_breaks_tie_on_same_front(self):
        merge_fit = np.array(
            [
                [0.0, 0.0],
                [1.0, 9.0],
                [9.0, 1.0],
            ],
            dtype=np.float64,
        )
        merge_pop = np.array([[0], [1], [2]], dtype=np.int32)
        _, _, idx = environmental_selection_nsga2(
            merge_pop, merge_fit, pop_size=2)
        assert 0 in idx
