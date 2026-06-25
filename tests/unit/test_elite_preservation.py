"""Unit tests for elite preservation under (μ+λ) selection.

Verifies that top-K deployable-archive elites survive mid-epoch erosion
from recomputed dynamic diversity/support penalties.
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.evolution.evox_runner import (
    _preserve_deployable_elites,
    _build_rank_and_crowding,
    environmental_selection_nsga2,
    non_dominated_sort,
)


def _make_chromosome(
    n_features: int = 10,
    seed: int = 0,
) -> np.ndarray:
    """Build a dense chromosome with some active genes."""
    rng = np.random.default_rng(seed)
    chrom = rng.integers(0, 3, size=n_features, dtype=np.int32)
    return chrom


def _make_deployable_entry(
    chromosome: np.ndarray,
    rank_score: float = 10.0,
) -> dict:
    """Create a deployable archive entry dict."""
    return {
        "chromosome": chromosome.copy(),
        "rank_score": rank_score,
        "metrics": {
            "sortino_ratio": 2.0,
            "total_return_pct": 15.0,
            "max_drawdown_pct": 5.0,
            "profit_factor": 1.5,
            "executed_trades": 100,
            "val_total_return_pct": 10.0,
            "val_profit_factor": 1.4,
            "val_executed_trades": 50,
        },
    }


class TestPreserveDeployableElites:
    """Unit tests for the _preserve_deployable_elites helper."""

    def test_disabled_no_op(self):
        """When preservation is disabled, the population is unchanged."""
        pop = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        obj = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        archive = {
            (0, 1, 2): _make_deployable_entry(pop[0], rank_score=10.0),
        }
        metrics = [{"a": 1}, {"b": 2}]
        pop_copy = pop.copy()

        with mock.patch.object(_cfg, "PHASE2_ELITE_PRESERVATION_ENABLED", False):
            _preserve_deployable_elites(pop, obj, archive, metrics, _cfg, 5)

        assert np.array_equal(pop, pop_copy)

    def test_below_min_gen_no_op(self):
        """When current_gen < MIN_GEN, the population is unchanged."""
        pop = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        obj = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        archive = {
            (0, 1, 2): _make_deployable_entry(pop[0], rank_score=10.0),
        }
        metrics = [{"a": 1}, {"b": 2}]
        pop_copy = pop.copy()

        with mock.patch.object(_cfg, "PHASE2_ELITE_PRESERVATION_MIN_GEN", 10):
            _preserve_deployable_elites(pop, obj, archive, metrics, _cfg, 5)

        assert np.array_equal(pop, pop_copy)

    def test_empty_archive_no_op(self):
        """When the archive is empty, the population is unchanged."""
        pop = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        obj = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        metrics = [{"a": 1}, {"b": 2}]
        pop_copy = pop.copy()

        _preserve_deployable_elites(pop, obj, {}, metrics, _cfg, 5)

        assert np.array_equal(pop, pop_copy)

    def test_champion_survives_penalty_drift_when_enabled(self):
        """A champion in deployable_archive survives 15 gens of penalty drift."""
        n_features = 6
        pop_size = 10

        # Champion chromosome — not initially in population
        champion = _make_chromosome(n_features, seed=999)
        rng = np.random.default_rng(42)

        # Initialize population with distinct chromosomes (none matching champion)
        population = rng.integers(0, 3, size=(pop_size, n_features), dtype=np.int32)
        objectives = np.full((pop_size, 3), 0.5, dtype=np.float64)
        # Make first individual best (lowest objectives) so it's rank 1
        objectives[0] = np.array([0.1, 0.1, 0.1])

        # Deployable archive with champion at top rank_score
        other_entry = _make_deployable_entry(
            _make_chromosome(n_features, seed=1), rank_score=5.0,
        )
        champion_entry = _make_deployable_entry(champion, rank_score=50.0)
        archive = {
            tuple(champion): champion_entry,
            tuple(_make_chromosome(n_features, seed=1)): other_entry,
        }

        metrics_cache = [{} for _ in range(pop_size)]

        # Run preservation for 15 generations (penalty drift simulated by
        # growing objectives each call, mimicking diversity/support drift)
        with mock.patch.object(
            _cfg, "PHASE2_ELITE_PRESERVATION_MIN_GEN", 0,
        ):
            for gen in range(15):
                # Simulate penalty drift: objectives get slightly worse each gen
                objectives += np.random.default_rng(gen).normal(0, 0.01, size=objectives.shape)

                _preserve_deployable_elites(
                    population, objectives, archive, metrics_cache, _cfg, gen,
                )

                # Verify champion is in population
                found = any(
                    np.array_equal(population[i], champion)
                    for i in range(pop_size)
                )
                assert found, f"Champion evicted at gen {gen} (preservation ON)"

    def test_champion_evicted_when_disabled(self):
        """Without preservation, champion is evicted by gen ~8 under drift."""
        n_features = 6
        pop_size = 10

        champion = _make_chromosome(n_features, seed=999)
        champion_entry = _make_deployable_entry(champion, rank_score=50.0)
        archive = {tuple(champion): champion_entry}

        rng = np.random.default_rng(42)
        population = rng.integers(0, 3, size=(pop_size, n_features), dtype=np.int32)
        objectives = np.full((pop_size, 3), 0.5, dtype=np.float64)
        objectives[0] = np.array([0.1, 0.1, 0.1])
        metrics_cache = [{} for _ in range(pop_size)]

        evicted_gen = None
        with mock.patch.object(_cfg, "PHASE2_ELITE_PRESERVATION_ENABLED", False):
            for gen in range(15):
                if gen < 1:
                    _preserve_deployable_elites(
                        population, objectives, archive, metrics_cache, _cfg, gen,
                    )
                    continue

                # Simulate (μ+λ) selection: create merged pop, select, then preserve
                # Objective drift
                objectives += np.random.default_rng(gen).normal(0, 0.02, size=objectives.shape)

                # Simulate offspring (slightly better on avg)
                offspring = rng.integers(0, 3, size=(pop_size, n_features), dtype=np.int32)
                off_obj = objectives + np.random.default_rng(gen+100).normal(-0.1, 0.05, size=objectives.shape)

                merge_pop = np.vstack([population, offspring])
                merge_fit = np.vstack([objectives, off_obj])

                population, objectives, sel_idx = environmental_selection_nsga2(
                    merge_pop, merge_fit, pop_size,
                )
                metrics_cache = [{} for _ in range(pop_size)]

                _preserve_deployable_elites(
                    population, objectives, archive, metrics_cache, _cfg, gen,
                )

                found = any(
                    np.array_equal(population[i], champion)
                    for i in range(pop_size)
                )
                if not found and evicted_gen is None:
                    evicted_gen = gen

        assert evicted_gen is not None, (
            "Champion was never evicted even with preservation OFF "
            "(test may need stronger drift)"
        )
        assert evicted_gen < 12, (
            f"Champion survived until gen {evicted_gen} with preservation OFF "
            "(drift may be too weak)"
        )

    def test_preservation_never_exceeds_top_k(self):
        """At most TOP_K slots are overwritten by elite preservation."""
        n_features = 4
        pop_size = 20
        top_k = int(_cfg.PHASE2_ELITE_PRESERVATION_TOP_K)

        rng = np.random.default_rng(42)
        population = rng.integers(0, 3, size=(pop_size, n_features), dtype=np.int32)
        objectives = np.ones((pop_size, 3), dtype=np.float64)
        metrics_cache = [{} for _ in range(pop_size)]

        # Archive with many entries (more than TOP_K)
        archive = {}
        for i in range(top_k + 5):
            chrom = rng.integers(0, 3, size=n_features, dtype=np.int32)
            key = tuple(chrom)
            # Ensure no duplicate keys
            while key in archive:
                chrom = rng.integers(0, 3, size=n_features, dtype=np.int32)
                key = tuple(chrom)
            archive[key] = _make_deployable_entry(chrom, rank_score=float(20 - i))

        pop_before = population.copy()
        _preserve_deployable_elites(
            population, objectives, archive, metrics_cache, _cfg, 5,
        )

        # Count how many slots changed
        changed = 0
        for i in range(pop_size):
            if not np.array_equal(population[i], pop_before[i]):
                changed += 1

        assert changed <= top_k, (
            f"Preservation modified {changed} slots, but TOP_K={top_k}"
        )

    def test_preserved_elite_objectives_reset_to_inf(self):
        """Preserved elite's objectives are reset to inf (forces re-eval)."""
        n_features = 4
        pop_size = 5

        champion = _make_chromosome(n_features, seed=999)
        archive = {tuple(champion): _make_deployable_entry(champion, rank_score=50.0)}

        population = np.zeros((pop_size, n_features), dtype=np.int32)
        for i in range(pop_size):
            population[i] = _make_chromosome(n_features, seed=i)

        objectives = np.full((pop_size, 3), 0.5, dtype=np.float64)
        objectives[0] = np.array([0.1, 0.1, 0.1])  # rank 1
        metrics_cache = [{"a": 1} for _ in range(pop_size)]

        _preserve_deployable_elites(
            population, objectives, archive, metrics_cache, _cfg, 5,
        )

        # Find where champion was placed and verify objectives are inf
        for i in range(pop_size):
            if np.array_equal(population[i], champion):
                assert np.all(np.isinf(objectives[i])), (
                    f"Preserved elite at index {i} has non-inf objectives: {objectives[i]}"
                )
                assert metrics_cache[i] == {}, (
                    f"Preserved elite at index {i} has non-empty metrics_cache"
                )
                return

        pytest.fail("Champion was not placed into population")

    def test_never_evicts_rank_one(self):
        """Preservation never evicts a rank-1 (Pareto front) member."""
        n_features = 4
        pop_size = 8

        champion = _make_chromosome(n_features, seed=999)
        archive = {tuple(champion): _make_deployable_entry(champion, rank_score=50.0)}

        # Population where first 3 members are rank 1 (different Pareto fronts)
        population = np.zeros((pop_size, n_features), dtype=np.int32)
        population[0] = _make_chromosome(n_features, seed=0)
        population[1] = _make_chromosome(n_features, seed=1)
        population[2] = _make_chromosome(n_features, seed=2)
        for i in range(3, pop_size):
            population[i] = _make_chromosome(n_features, seed=i + 10)

        # First 3 are Pareto-optimal (objectives are best in one dimension each)
        objectives = np.full((pop_size, 3), 10.0, dtype=np.float64)
        objectives[0] = np.array([1.0, 10.0, 10.0])
        objectives[1] = np.array([10.0, 1.0, 10.0])
        objectives[2] = np.array([10.0, 10.0, 1.0])
        # Rest are worse
        for i in range(3, pop_size):
            objectives[i] = np.array([15.0, 15.0, 15.0])

        metrics_cache = [{} for _ in range(pop_size)]
        fronts = non_dominated_sort(objectives)
        ranks, _ = _build_rank_and_crowding(objectives, fronts)

        pop_before = population.copy()
        _preserve_deployable_elites(
            population, objectives, archive, metrics_cache, _cfg, 5,
        )

        # Verify none of the rank-1 slots (0,1,2) were overwritten
        for i in range(3):
            assert np.array_equal(population[i], pop_before[i]), (
                f"Rank-1 member at index {i} was evicted by preservation"
            )

    def test_byte_identical_when_disabled(self):
        """Snapshot: with ENABLED=False, evolution is unchanged (2-gen)."""
        n_features = 6
        pop_size = 10

        rng = np.random.default_rng(42)
        population = rng.integers(0, 3, size=(pop_size, n_features), dtype=np.int32)
        objectives = np.full((pop_size, 3), 0.5, dtype=np.float64)
        objectives[0] = np.array([0.1, 0.1, 0.1])
        metrics_cache = [{} for _ in range(pop_size)]
        archive = {
            tuple(_make_chromosome(n_features, seed=99)):
                _make_deployable_entry(_make_chromosome(n_features, seed=99), rank_score=20.0),
        }

        # Run 2 generations with preservation disabled
        with mock.patch.object(_cfg, "PHASE2_ELITE_PRESERVATION_ENABLED", False):
            for gen in range(2):
                # Create offspring
                offspring = rng.integers(0, 3, size=(pop_size, n_features), dtype=np.int32)
                off_obj = objectives + np.random.default_rng(gen+100).normal(0, 0.05, size=objectives.shape)

                merge_pop = np.vstack([population, offspring])
                merge_fit = np.vstack([objectives, off_obj])

                population, objectives, sel_idx = environmental_selection_nsga2(
                    merge_pop, merge_fit, pop_size,
                )
                metrics_cache = [{} for _ in range(pop_size)]

                _preserve_deployable_elites(
                    population, objectives, archive, metrics_cache, _cfg, gen,
                )

        # Run 2 generations WITHOUT calling _preserve_deployable_elites at all
        rng2 = np.random.default_rng(42)
        population2 = rng2.integers(0, 3, size=(pop_size, n_features), dtype=np.int32)
        objectives2 = np.full((pop_size, 3), 0.5, dtype=np.float64)
        objectives2[0] = np.array([0.1, 0.1, 0.1])
        metrics_cache2 = [{} for _ in range(pop_size)]

        for gen in range(2):
            offspring2 = rng2.integers(0, 3, size=(pop_size, n_features), dtype=np.int32)
            off_obj2 = objectives2 + np.random.default_rng(gen+100).normal(0, 0.05, size=objectives2.shape)

            merge_pop2 = np.vstack([population2, offspring2])
            merge_fit2 = np.vstack([objectives2, off_obj2])

            population2, objectives2, sel_idx2 = environmental_selection_nsga2(
                merge_pop2, merge_fit2, pop_size,
            )
            metrics_cache2 = [{} for _ in range(pop_size)]

        # Verify byte-for-byte identical
        assert np.array_equal(population, population2), (
            "Population differs when preservation disabled vs not calling it"
        )
        assert np.array_equal(objectives, objectives2), (
            "Objectives differ when preservation disabled vs not calling it"
        )
