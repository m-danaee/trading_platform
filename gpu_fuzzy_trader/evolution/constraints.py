"""Constraint-aware Pareto ordering for Phase 2 evolution.

Objectives alone are not enough when an infeasible rule receives a favorable
trade-off on another axis. This module implements Deb-style constrained
dominance without changing the stored objective values:

* any feasible candidate dominates any infeasible candidate;
* among infeasible candidates, the smaller total violation wins;
* candidates with equal feasibility status and violation use ordinary Pareto
  dominance on the minimisation objectives.
"""

from __future__ import annotations

import math

import numpy as np


def _clean_violation(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        return math.inf
    return max(0.0, value)


def _pareto_dominates(left: np.ndarray, right: np.ndarray) -> bool:
    """Return whether *left* Pareto-dominates *right* (minimisation)."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        return False
    return bool(np.all(left <= right) and np.any(left < right))


def constrained_dominates(
    left_objectives: np.ndarray,
    right_objectives: np.ndarray,
    left_violation: float,
    right_violation: float,
    *,
    epsilon: float = 1e-12,
) -> bool:
    """Return whether the left candidate dominates the right candidate."""
    left_cv = _clean_violation(left_violation)
    right_cv = _clean_violation(right_violation)
    left_feasible = left_cv <= epsilon
    right_feasible = right_cv <= epsilon

    if left_feasible != right_feasible:
        return left_feasible
    if not left_feasible:
        if left_cv < right_cv - epsilon:
            return True
        if right_cv < left_cv - epsilon:
            return False
    return _pareto_dominates(left_objectives, right_objectives)


def constrained_non_dominated_sort(
    objectives: np.ndarray,
    violations: np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> list[list[int]]:
    """Sort candidates into fronts using constrained dominance."""
    obj = np.asarray(objectives, dtype=np.float64)
    cv = np.asarray(violations, dtype=np.float64).reshape(-1)
    if obj.ndim != 2:
        raise ValueError("objectives must be a 2D array")
    if len(obj) != len(cv):
        raise ValueError("violations must have one value per objective row")
    if len(obj) == 0:
        return [[]]

    dominates: list[list[int]] = [[] for _ in range(len(obj))]
    dominated_count = np.zeros(len(obj), dtype=np.int64)
    first_front: list[int] = []
    for i in range(len(obj)):
        for j in range(i + 1, len(obj)):
            if constrained_dominates(obj[i], obj[j], cv[i], cv[j], epsilon=epsilon):
                dominates[i].append(j)
                dominated_count[j] += 1
            elif constrained_dominates(obj[j], obj[i], cv[j], cv[i], epsilon=epsilon):
                dominates[j].append(i)
                dominated_count[i] += 1
        if dominated_count[i] == 0:
            first_front.append(i)

    fronts: list[list[int]] = [first_front]
    current = first_front
    while current:
        next_front: list[int] = []
        for i in current:
            for j in dominates[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    next_front.append(j)
        if next_front:
            fronts.append(next_front)
        current = next_front
    return fronts

