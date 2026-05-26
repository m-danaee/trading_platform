"""Unit tests for the Numba NSGA helpers."""

from __future__ import annotations

import numpy as np

from gpu_fuzzy_trader.evolution import numba_ops


def test_non_dominated_sort_falls_back_when_numba_fails(monkeypatch) -> None:
    monkeypatch.setattr(numba_ops, "_NUMBA_SORT_AVAILABLE", True)
    monkeypatch.setattr(numba_ops, "numba_enabled", lambda: True)

    def _boom(objectives: np.ndarray):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        numba_ops, "_non_dominated_sort_numba", _boom, raising=False)

    objectives = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float64)

    assert numba_ops.non_dominated_sort(objectives) == [[0], [1]]
    assert numba_ops._NUMBA_SORT_AVAILABLE is False


def test_non_dominated_sort_empty_returns_single_empty_front() -> None:
    empty = np.zeros((0, 3), dtype=np.float64)
    assert numba_ops.non_dominated_sort(empty) == [[]]


def test_non_dominated_sort_numba_compiles() -> None:
    if not numba_ops._NUMBA_AVAILABLE or not numba_ops.numba_enabled():
        return
    rng = np.random.default_rng(0)
    objectives = rng.random((5, 3))
    fronts = numba_ops.non_dominated_sort(objectives)
    assert fronts
    assert all(isinstance(front, list) for front in fronts)
