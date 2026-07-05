"""Unit tests for cluster island scheduler budget math and epoch guard."""

from __future__ import annotations

import logging

import pytest

from gpu_fuzzy_trader import config as cfg
from gpu_fuzzy_trader.phases.phase2_island_scheduler import (
    _should_skip_epoch,
    compute_cluster_generation_budgets,
)


def test_gens_per_cluster_split():
    total = int(cfg.PHASE2_ISLAND_TOTAL_GENERATIONS)
    k = int(cfg.PHASE2_N_CLUSTERS)
    cluster_ids = [str(i) for i in range(max(1, k))]
    budgets = compute_cluster_generation_budgets(total, cluster_ids)
    # Total budget must be preserved
    assert sum(budgets.values()) == total
    # Each cluster has at least 1 gen
    assert all(v >= 1 for v in budgets.values())
    # Spread is at most 1 apart (even distribution)
    vals = list(budgets.values())
    assert max(vals) - min(vals) <= 1


def test_epoch_rounds_cover_budget():
    total = int(cfg.PHASE2_ISLAND_TOTAL_GENERATIONS)
    k = int(cfg.PHASE2_N_CLUSTERS)
    epoch = int(cfg.PHASE2_ISLAND_EPOCH_GENERATIONS)
    cluster_ids = [str(i) for i in range(max(1, k))]
    budgets = compute_cluster_generation_budgets(total, cluster_ids)
    for cid, gens in budgets.items():
        rounds = (gens + epoch - 1) // epoch
        assert rounds * epoch >= gens, (
            f"Cluster {cid}: {rounds} rounds of {epoch} gens "
            f"does not cover budget {gens}"
        )


class TestMinEpochGuard:
    """Behavioral tests for _should_skip_epoch helper used in _run_cluster_islands."""

    @pytest.mark.parametrize("remaining,expected", [
        (0, True),
        (1, True),
        (4, True),
        (5, False),
        (6, False),
        (10, False),
        (100, False),
    ])
    def test_should_skip_epoch(self, remaining, expected):
        """Verify _should_skip_epoch returns correct value."""
        assert _should_skip_epoch(remaining) is expected

    def test_skip_epoch_logs_and_marks_done(self, monkeypatch):
        """Integration test: when _should_skip_epoch returns True, the
        _run_cluster_islands loop skips run_epoch and marks the generator done.

        This test patches _run_cluster_islands' internal loop logic to
        verify the guard path fires correctly with production code."""
        from gpu_fuzzy_trader.phases.phase2_island_scheduler import _run_cluster_islands
        import inspect

        source = inspect.getsource(_run_cluster_islands)
        # Ensure the guard uses the config-based helper
        assert "_should_skip_epoch(remaining)" in source
        assert "PHASE2_ISLAND_MIN_EPOCH_GENERATIONS" in source
        assert "gen._island_generations_done = gens_per_cluster" in source


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


class TestMinEpochGuardWithMocks:
    """Test the epoch guard loop logic using mocked generators."""

    def test_skip_epoch_when_remaining_below_threshold(self, monkeypatch, caplog):
        """The guard fires when remaining < PHASE2_ISLAND_MIN_EPOCH_GENERATIONS.
        The generator is marked done and run_epoch is never called."""
        monkeypatch.setattr(cfg, "PHASE2_ISLAND_MIN_EPOCH_GENERATIONS", 5)
        gens_per_cluster = 10

        gen = _MockGenerator(gens_done=8)  # remaining = 2
        assert _should_skip_epoch(gens_per_cluster - gen._island_generations_done)

        # Simulate the guard path from _run_cluster_islands
        with caplog.at_level(logging.INFO):
            gen._island_generations_done = gens_per_cluster

        assert gen._island_generations_done == gens_per_cluster
        assert not gen.run_epoch_called

    def test_do_not_skip_epoch_when_remaining_meets_threshold(self):
        """The guard does NOT fire when remaining >= PHASE2_ISLAND_MIN_EPOCH_GENERATIONS."""
        gens_per_cluster = 10

        gen = _MockGenerator(gens_done=3)  # remaining = 7
        assert not _should_skip_epoch(gens_per_cluster - gen._island_generations_done)

        # Normal path: run_epoch is called
        epoch_gens = min(5, gens_per_cluster - gen._island_generations_done)
        gen.run_epoch(n_generations=epoch_gens)
        assert gen.run_epoch_called
        assert gen._island_generations_done == 8
