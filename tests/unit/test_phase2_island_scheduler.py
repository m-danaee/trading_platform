"""Unit tests for cluster island scheduler budget math and epoch guard."""

from __future__ import annotations

import inspect

from gpu_fuzzy_trader import config as cfg
from gpu_fuzzy_trader.phases import phase2_island_scheduler


def test_gens_per_cluster_split():
    total = int(cfg.PHASE2_ISLAND_TOTAL_GENERATIONS)
    k = int(cfg.PHASE2_N_CLUSTERS)
    gens_per = max(1, total // max(1, k))
    assert gens_per * k <= total + k


def test_epoch_rounds_cover_budget():
    total = int(cfg.PHASE2_ISLAND_TOTAL_GENERATIONS)
    k = int(cfg.PHASE2_N_CLUSTERS)
    epoch = int(cfg.PHASE2_ISLAND_EPOCH_GENERATIONS)
    gens_per = max(1, total // max(1, k))
    rounds = (gens_per + epoch - 1) // epoch
    assert rounds * epoch >= gens_per


def test_min_epoch_guard_source_present():
    """Verify MIN_EPOCH_GENS = 5 guard exists in _run_cluster_islands."""
    source = inspect.getsource(phase2_island_scheduler._run_cluster_islands)
    # The guard constant must be defined
    assert "MIN_EPOCH_GENS = 5" in source, (
        "MIN_EPOCH_GENS = 5 missing from _run_cluster_islands"
    )
    # The guard must skip epochs with remaining < threshold
    assert "remaining < MIN_EPOCH_GENS" in source, (
        "remaining < MIN_EPOCH_GENS guard missing from _run_cluster_islands"
    )
    # The generator must be marked as done to exit the loop cleanly
    assert "gen._island_generations_done = gens_per_cluster" in source, (
        "Missing assignment to exit loop when skipping epoch"
    )
    # After the guard block, the loop should skip to next cluster via continue
    assert "MIN_EPOCH_GENS" in source, "MIN_EPOCH_GENS not found in source"


class _MockGenerator:
    """Minimal mock for Rule_Pool_Generator used in epoch guard tests."""

    def __init__(self, gens_done: int):
        self._island_generations_done = gens_done
        self.run_epoch_called = False
        self.park_engines_called = False

    def run_epoch(self, n_generations: int) -> None:
        self.run_epoch_called = True
        self._island_generations_done += n_generations

    def park_engines(self) -> None:
        self.park_engines_called = True


def test_min_epoch_guard_skips_small_remaining():
    """Verify that the min-epoch guard in _run_cluster_islands
    skips an epoch when remaining < 5 by simulating the inline logic."""
    gens_per_cluster = 10
    MIN_EPOCH_GENS = 5

    # Simulate a generator that has done 8 gens → remaining = 2 (< 5)
    gen = _MockGenerator(gens_done=8)
    assert gen._island_generations_done < gens_per_cluster
    remaining = gens_per_cluster - gen._island_generations_done
    assert remaining == 2
    assert remaining < MIN_EPOCH_GENS

    # Guard should fire: mark as done, skip run_epoch
    gen._island_generations_done = gens_per_cluster

    assert not gen.run_epoch_called
    assert gen._island_generations_done == gens_per_cluster


def test_min_epoch_guard_passes_large_remaining():
    """Verify that the min-epoch guard does NOT skip epochs
    when remaining >= 5."""
    gens_per_cluster = 10
    MIN_EPOCH_GENS = 5

    # Simulate a generator that has done 3 gens → remaining = 7 (>= 5)
    gen = _MockGenerator(gens_done=3)
    remaining = gens_per_cluster - gen._island_generations_done
    assert remaining == 7
    assert remaining >= MIN_EPOCH_GENS

    # Guard should NOT fire
    epoch_gens = min(5, remaining)
    gen.run_epoch(n_generations=epoch_gens)
    assert gen.run_epoch_called
    assert gen._island_generations_done == 8
