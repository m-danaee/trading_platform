"""Tests for constrained Pareto ordering used by Phase 2 selection."""

from __future__ import annotations

import numpy as np

from gpu_fuzzy_trader.evolution.constraints import (
    constrained_dominates,
    constrained_non_dominated_sort,
)


def test_feasible_candidate_dominates_infeasible_candidate() -> None:
    assert constrained_dominates(
        np.array([100.0, 100.0]),
        np.array([0.0, 0.0]),
        0.0,
        1.0,
    )


def test_lower_violation_wins_when_both_are_infeasible() -> None:
    assert constrained_dominates(
        np.array([100.0, 100.0]),
        np.array([0.0, 0.0]),
        0.5,
        2.0,
    )


def test_sort_places_feasible_candidates_in_first_front() -> None:
    objectives = np.array(
        [
            [10.0, 10.0],  # feasible
            [0.0, 0.0],    # infeasible but attractive objectives
            [12.0, 12.0],  # feasible, dominated by row 0
        ],
    )
    fronts = constrained_non_dominated_sort(objectives, np.array([0.0, 1.0, 0.0]))
    assert fronts[0] == [0]
    assert 1 not in fronts[0]

