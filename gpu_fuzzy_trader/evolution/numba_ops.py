"""
numba_ops.py — Numba-accelerated NSGA helpers and batch support penalties.

All entry points accept host NumPy arrays only (never JAX DeviceArray).
"""

from __future__ import annotations

import logging
import os
import numpy as np

from gpu_fuzzy_trader import config as _cfg

_NUMBA_AVAILABLE = False
_NUMBA_SORT_AVAILABLE = False
try:
    import numba
    from numba import njit

    _NUMBA_AVAILABLE = True
    _NUMBA_SORT_AVAILABLE = True
    _numba_threads = int(os.environ.get("NUMBA_NUM_THREADS", "0") or 0)
    if _numba_threads > 0:
        try:
            numba.set_num_threads(_numba_threads)
        except Exception:
            pass
except ImportError:
    njit = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


def numba_enabled() -> bool:
    return _NUMBA_AVAILABLE and bool(_cfg.PHASE2_NUMBA_ENABLED)


def _dominates_py(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def _disable_numba_sort(reason: str) -> None:
    global _NUMBA_SORT_AVAILABLE
    if _NUMBA_SORT_AVAILABLE:
        logger.warning("Disabling Numba non-dominated sort: %s", reason)
    _NUMBA_SORT_AVAILABLE = False


def _non_dominated_sort_py(objectives: np.ndarray) -> list[list[int]]:
    n = len(objectives)
    if n == 0:
        return [[]]

    domination_count = np.zeros(n, dtype=int)
    dominated_by: list[list[int]] = [[] for _ in range(n)]
    first_front: list[int] = []

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if _dominates_py(objectives[i], objectives[j]):
                dominated_by[i].append(j)
            elif _dominates_py(objectives[j], objectives[i]):
                domination_count[i] += 1
        if domination_count[i] == 0:
            first_front.append(i)

    fronts: list[list[int]] = [first_front]
    current_front = 0

    while fronts[current_front]:
        next_front: list[int] = []
        for i in fronts[current_front]:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        current_front += 1
        if next_front:
            fronts.append(next_front)
        else:
            break

    return fronts


def _crowding_distance_py(objectives: np.ndarray, front: list[int]) -> np.ndarray:
    n = len(front)
    if n <= 2:
        return np.full(n, np.inf)

    distances = np.zeros(n)
    front_obj = objectives[front]
    m = front_obj.shape[1]

    for mi in range(m):
        order = np.argsort(front_obj[:, mi])
        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf
        obj_range = front_obj[order[-1], mi] - front_obj[order[0], mi]
        if obj_range == 0:
            continue
        for k in range(1, n - 1):
            distances[order[k]] += (
                front_obj[order[k + 1], mi] - front_obj[order[k - 1], mi]
            ) / obj_range

    return distances


if _NUMBA_AVAILABLE:

    @njit(cache=True)
    def _dominates_numba(a, b):
        better = False
        for i in range(len(a)):
            if a[i] > b[i]:
                return False
            if a[i] < b[i]:
                better = True
        return better

    @njit(cache=True)
    def _non_dominated_sort_numba(objectives):
        """
        NSGA non-dominated sort using O(N + E) memory instead of O(N²).

        dominated_by[i] is a flat list of solutions that i dominates.
        domination_count[i] counts how many solutions dominate i.
        Only the (typically sparse) domination edges are stored.
        """
        n = objectives.shape[0]
        domination_count = np.zeros(n, dtype=np.int64)

        # Flat ragged storage: dominated_by_start[i] points into dominated_by_flat.
        # We first count, then fill to avoid dynamic appends in nopython mode.
        edge_counts = np.zeros(n, dtype=np.int64)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if _dominates_numba(objectives[i], objectives[j]):
                    edge_counts[i] += 1
                elif _dominates_numba(objectives[j], objectives[i]):
                    domination_count[i] += 1

        total_edges = np.sum(edge_counts)
        dominated_by_flat = np.empty(total_edges, dtype=np.int64)
        dominated_by_start = np.zeros(n + 1, dtype=np.int64)
        for i in range(n):
            dominated_by_start[i + 1] = dominated_by_start[i] + edge_counts[i]

        write_pos = dominated_by_start.copy()
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if _dominates_numba(objectives[i], objectives[j]):
                    dominated_by_flat[write_pos[i]] = j
                    write_pos[i] += 1

        # BFS-style front extraction
        first_front = []
        for i in range(n):
            if domination_count[i] == 0:
                first_front.append(i)

        fronts = []
        fronts.append(first_front)
        current = 0
        while len(fronts[current]) > 0:
            next_front = []
            for i in fronts[current]:
                start = dominated_by_start[i]
                end = dominated_by_start[i + 1]
                for k in range(start, end):
                    j = dominated_by_flat[k]
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current += 1
            if len(next_front) == 0:
                break
            fronts.append(next_front)

        out = []
        for f in fronts:
            if len(f) > 0:
                out.append(f)
        return out

    @njit(cache=True)
    def _crowding_distance_numba(front_obj):
        n = front_obj.shape[0]
        if n <= 2:
            out = np.empty(n, dtype=np.float64)
            for i in range(n):
                out[i] = np.inf
            return out

        m = front_obj.shape[1]
        distances = np.zeros(n, dtype=np.float64)
        for mi in range(m):
            order = np.argsort(front_obj[:, mi])
            distances[order[0]] = np.inf
            distances[order[n - 1]] = np.inf
            obj_range = front_obj[order[n - 1], mi] - front_obj[order[0], mi]
            if obj_range == 0.0:
                continue
            for k in range(1, n - 1):
                idx = order[k]
                distances[idx] += (
                    front_obj[order[k + 1], mi] - front_obj[order[k - 1], mi]
                ) / obj_range
        return distances

def non_dominated_sort(objectives: np.ndarray) -> list[list[int]]:
    """Non-dominated sorting with optional Numba acceleration."""
    obj = np.asarray(objectives, dtype=np.float64)
    if obj.shape[0] == 0:
        return [[]]
    if _NUMBA_SORT_AVAILABLE and numba_enabled():
        try:
            return _non_dominated_sort_numba(obj)  # type: ignore[misc]
        except Exception as exc:
            _disable_numba_sort(f"{exc.__class__.__name__}: {exc}")
    return _non_dominated_sort_py(obj)


def crowding_distance(objectives: np.ndarray, front: list[int]) -> np.ndarray:
    """Crowding distance for a front with optional Numba acceleration."""
    if not front:
        return np.array([], dtype=np.float64)
    front_obj = np.asarray(objectives[front], dtype=np.float64)
    if numba_enabled():
        return _crowding_distance_numba(front_obj)  # type: ignore[misc]
    return _crowding_distance_py(objectives, front)


def batch_hamming_min(
    chromosome: np.ndarray,
    archive: list[np.ndarray],
) -> int:
    """
    Minimum Hamming distance from *chromosome* to any member of *archive*.

    Vectorized over the archive in one NumPy broadcast instead of a Python loop,
    reducing the per-generation cost from O(|archive| × K) Python calls to a
    single (|archive|, K) boolean matrix comparison.

    Returns ``len(chromosome) + 1`` when *archive* is empty (no penalty triggers).
    """
    if not archive:
        return len(chromosome) + 1
    arr = np.stack(archive, axis=0)   # (A, K)
    dists = np.sum(arr != chromosome[None, :], axis=1)  # (A,) int
    return int(dists.min())
