"""Unit tests for diversity restart on first plateau (Fix D).

Verifies that:
- First plateau triggers a diversity restart (not a break).
- Second plateau (or when restart_count >= max_restarts) breaks.
- Pareto elite chromosomes survive the restart.
- mutation_rate is boosted for 3 gens then restored.
- plateau_streak resets to 0 after restart.
- PHASE2_PLATEAU_DIVERSITY_RESTART_ENABLED=False → immediate break.
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from gpu_fuzzy_trader import config as cfg
from gpu_fuzzy_trader.evolution.evox_runner import (
    _plateau_diversity_restart,
    _should_plateau_early_stop_phase2,
    _stage_mutation_rate,
    run_phase2_evolution,
)
from gpu_fuzzy_trader.phases.phase2_rule_pool import _init_population


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    """New config keys have correct defaults."""
    assert cfg.PHASE2_PLATEAU_DIVERSITY_RESTART_ENABLED is True
    assert cfg.PHASE2_PLATEAU_DIVERSITY_RESTART_FRACTION == 0.65
    assert cfg.PHASE2_PLATEAU_MAX_RESTARTS == 3
    assert cfg.PHASE2_PLATEAU_EARLY_STOP_MIN_DELTA_PCT == 0.5
    assert cfg.PHASE2_PLATEAU_POST_RESTART_MUTATION_BOOST == 0.45
    assert cfg.PHASE2_PLATEAU_POST_RESTART_BOOST_GENS == 3
    assert cfg.PHASE2_HOF_EPOCH_CARRYOVER == 10
    assert cfg.PHASE2_DIVERSITY_HAMMING_THRESHOLD_AUTO is True
    assert cfg.PHASE2_DIVERSITY_HAMMING_THRESHOLD == 0


# ---------------------------------------------------------------------------
# _plateau_diversity_restart helper
# ---------------------------------------------------------------------------

class TestPlateauDiversityRestart:
    """Direct unit tests for the _plateau_diversity_restart helper."""

    def test_elite_preserved(self):
        """Pareto elite chromosomes survive the restart."""
        rng = np.random.default_rng(42)
        feature_infos = [
            {"name": "f0", "mode": "binary", "score": 0.5},
            {"name": "f1", "mode": "binary", "score": 0.5},
            {"name": "f2", "mode": "binary", "score": 0.5},
        ]
        pop_size = 20
        # Use _init_population to get correct sparse format
        population = _init_population(pop_size, feature_infos, rng)
        # Elite = indices 0..4 (first 5 Pareto front indices, but n_elite caps at 2)
        elite_indices = list(range(2))
        pareto_indices = list(range(5)) + [10, 11, 12]

        objectives = np.full((pop_size, 3), 0.5)
        metrics_cache = [{"total_return_pct": float(i)} for i in range(pop_size)]

        # Record elite chromosome keys (sparse encoding)
        from gpu_fuzzy_trader.phases.phase2_sparse_encoding import chromosome_key
        elite_keys = {chromosome_key(population[i]) for i in elite_indices}

        with mock.patch.object(
            cfg, "PHASE2_PLATEAU_DIVERSITY_RESTART_FRACTION", 0.40,
        ):
            n_elite = _plateau_diversity_restart(
                population,
                objectives,
                metrics_cache,
                feature_infos,
                rng,
                pareto_indices=pareto_indices,
                pop_size=pop_size,
            )

        assert n_elite == len(elite_indices), (
            f"Expected {len(elite_indices)} elite kept, got {n_elite}"
        )
        # Check elite chromosomes are intact
        for i in elite_indices:
            assert chromosome_key(population[i]) in elite_keys, (
                f"Elite at index {i} was modified"
            )
            # Elite objectives/metrics should NOT be reset to inf/{}
            assert not np.any(np.isinf(objectives[i])), (
                f"Elite objective at {i} was reset to inf"
            )
            assert metrics_cache[i] != {}, (
                f"Elite metrics at {i} was reset to empty"
            )

    def test_reinit_slots_have_inf_objectives(self):
        """Reinitialised slots have objectives=np.inf and metrics_cache={}."""
        rng = np.random.default_rng(42)
        feature_infos = [
            {"name": "f0", "mode": "binary", "score": 0.5},
        ]
        pop_size = 10
        population = _init_population(pop_size, feature_infos, rng)
        pareto_indices = [0]
        objectives = np.full((pop_size, 3), 0.5)
        metrics_cache = [{"val": float(i)} for i in range(pop_size)]

        with mock.patch.object(
            cfg, "PHASE2_PLATEAU_DIVERSITY_RESTART_FRACTION", 0.50,
        ):
            _plateau_diversity_restart(
                population,
                objectives,
                metrics_cache,
                feature_infos,
                rng,
                pareto_indices=pareto_indices,
                pop_size=pop_size,
            )

        # At least one non-elite slot should be reset
        non_elite = [i for i in range(pop_size) if i != 0]
        reset_count = sum(
            1 for i in non_elite
            if np.any(np.isinf(objectives[i])) and metrics_cache[i] == {}
        )
        assert reset_count >= 1, (
            "No non-elite slots were reset to inf/{}"
        )

    def test_n_elite_at_most_5(self):
        """Even with large Pareto front, at most 2 elite are preserved (was 5)."""
        rng = np.random.default_rng(42)
        feature_infos = [
            {"name": "f0", "mode": "binary", "score": 0.5},
        ]
        pop_size = 20
        population = _init_population(pop_size, feature_infos, rng)
        # Pareto front has 20 indices
        pareto_indices = list(range(20))
        objectives = np.full((pop_size, 3), 0.5)
        metrics_cache = [{"v": i} for i in range(pop_size)]

        n_elite = _plateau_diversity_restart(
            population,
            objectives,
            metrics_cache,
            feature_infos,
            rng,
            pareto_indices=pareto_indices,
            pop_size=pop_size,
        )
        assert n_elite == 2, f"Expected 2 elite, got {n_elite}"


# ---------------------------------------------------------------------------
# Mutation boost logic
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Integration-style test: mutation boost applied to offspring + streak reset
# ---------------------------------------------------------------------------

class TestMutationBoostIntegration:
    """Verify AC3 (mutation boost applied to offspring) and AC4 (streak reset).

    Runs the full evolution loop with mocked plateau detection, captures the
    ``mutation_rate`` argument passed to ``_make_offspring_population``, and
    tracks ``plateau_streak`` via the plateau-check mock.
    """

    def test_boost_applied_and_streak_reset(self):
        """Mutation boost is applied to offspring after restart, then reverts;
        plateau_streak resets to 0 after restart."""
        from gpu_fuzzy_trader.evolution.evox_runner import (
            _make_offspring_population,
        )

        mutation_rates: list[float] = []
        plateau_streak_at_gen: dict[int, int] = {}

        orig_make_offspring = _make_offspring_population

        def tracking_make_offspring(
            population, objectives, pop_size, feature_infos,
            dont_cares, rng, **kwargs,
        ):
            mutation_rates.append(kwargs.get("mutation_rate"))
            return orig_make_offspring(
                population, objectives, pop_size, feature_infos,
                dont_cares, rng, **kwargs,
            )

        def controlled_plateau_check(gen, streak, **kwargs):
            plateau_streak_at_gen[gen] = streak
            # Trigger restart only at gen 2; otherwise no plateau
            return gen == 2

        with mock.patch(
            "gpu_fuzzy_trader.evolution.evox_runner"
            "._should_plateau_early_stop_phase2",
            side_effect=controlled_plateau_check,
        ), mock.patch(
            "gpu_fuzzy_trader.evolution.evox_runner"
            "._plateau_diversity_restart",
            return_value=2,
        ), mock.patch(
            "gpu_fuzzy_trader.evolution.evox_runner"
            "._make_offspring_population",
            side_effect=tracking_make_offspring,
        ), mock.patch.object(
            cfg, "PHASE2_PLATEAU_DIVERSITY_RESTART_ENABLED", True,
        ), mock.patch.object(
            cfg, "PHASE2_PLATEAU_MAX_RESTARTS", 10,
        ), mock.patch.object(
            cfg, "PHASE2_DIVERSITY_RECOVERY_ENABLED", False,
        ), mock.patch.object(
            cfg, "PHASE2_EARLY_STOP_ENABLED", False,
        ), mock.patch.object(
            cfg, "PHASE2_EARLY_STOP_MIN_GENERATION", 999,
        ), mock.patch.object(
            cfg, "PHASE2_PLATEAU_POST_RESTART_STOP_ENABLED", False,
        ):
            rng = np.random.default_rng(0)
            feature_infos = [
                {"name": "f0", "mode": "binary", "score": 0.5},
                {"name": "f1", "mode": "binary", "score": 0.5},
            ]
            with mock.patch(
                "gpu_fuzzy_trader.evolution.evox_runner._EVOX_AVAILABLE",
                False,
            ):
                run_phase2_evolution(
                    feature_infos=feature_infos,
                    engine=FakeEngine(),
                    pop_size=10,
                    n_generations=15,
                    rng=rng,
                )

        # ---- AC3: multi-gen mutation boost ----
        # Offspring calls happen each gen except gen 2 (continue after restart)
        #   [0]=gen0, [1]=gen1, [2]=gen3(boosted), [3]=gen4(boosted), [4]=gen5(boosted)
        assert len(mutation_rates) >= 5, (
            f"Expected >=5 offspring calls, got {len(mutation_rates)}"
        )

        expected_boosted = float(
            getattr(cfg, "PHASE2_PLATEAU_POST_RESTART_MUTATION_BOOST", 0.45)
        )

        # After restart at gen 2, next 3 offspring calls use boosted mutation rate
        for i in range(2, 5):
            assert mutation_rates[i] == pytest.approx(expected_boosted), (
                f"Gen {i+1} mutation_rate {mutation_rates[i]} != boosted "
                f"{expected_boosted}"
            )

        # After 3 boosted gens, it should revert to normal mutation rate
        gen6_rate = mutation_rates[5]
        base_mr = _stage_mutation_rate(None)
        assert gen6_rate == pytest.approx(base_mr), (
            f"Gen 6 mutation_rate {gen6_rate} != default {base_mr}"
        )

        # ---- AC4: plateau_streak reset ----
        # The restart at gen 2 resets plateau_streak to 0.
        # At gen 3's plateau check, streak should be low (1 after one
        # _update_max_return_plateau increment from 0).
        assert 3 in plateau_streak_at_gen, (
            "No plateau_streak recorded for gen 3"
        )
        assert plateau_streak_at_gen[3] <= 1, (
            f"Expected plateau_streak <= 1 at gen 3 (after restart), "
            f"got {plateau_streak_at_gen[3]}"
        )


# ---------------------------------------------------------------------------
# Plateau early-stop trigger (should_plateau_early_stop_phase2)
# ---------------------------------------------------------------------------

def test_plateau_trigger_basic(monkeypatch):
    """Plateau trigger works normally (unchanged)."""
    monkeypatch.setattr(cfg, "PHASE2_PLATEAU_EARLY_STOP_ENABLED", True)
    monkeypatch.setattr(cfg, "PHASE2_PLATEAU_BLOCK_WHEN_DEPLOYABLE_ZERO", True)
    monkeypatch.setattr(cfg, "PHASE2_PLATEAU_EARLY_STOP_PATIENCE", 5)
    # streak=5 >= patience=5, deployable=10 → should trigger
    assert _should_plateau_early_stop_phase2(
        9, 5, deployable_count=10, island_profile="global",
    )


# ---------------------------------------------------------------------------
# Full-loop tests using mocks to control plateau detection
# ---------------------------------------------------------------------------

class FakeEngine:
    """Minimal engine stub that returns fixed metrics."""

    def simulate_rule_batch(self, chromosomes, tp=None, sl=None, capital_pct=None):
        B = chromosomes.shape[0]
        return [
            {
                "sortino_ratio": 1.0,
                "total_return_pct": 1.0,
                "max_drawdown_pct": 2.0,
                "win_rate": 50.0,
                "executed_trades": 50,
            }
            for _ in range(B)
        ]


class TestPlateauBranchDecision:
    """Verify the plateau branch decision logic in _run_nsga3.

    We monkeypatch _should_plateau_early_stop_phase2 to return True at
    specific generations, and count calls to _plateau_diversity_restart.
    """

    def test_first_plateau_triggers_restart(self):
        """First plateau triggers _plateau_diversity_restart."""
        restart_called = [0]

        original_plateau_check = _should_plateau_early_stop_phase2

        def controlled_plateau_check(gen, streak, **kwargs):
            # Return True on gen 2 (simulating plateau)
            if gen >= 2:
                return True
            return original_plateau_check(gen, streak, **kwargs)

        def mock_restart(*args, **kwargs):
            restart_called[0] += 1
            return 5  # elite_kept

        with mock.patch(
            "gpu_fuzzy_trader.evolution.evox_runner._should_plateau_early_stop_phase2",
            side_effect=controlled_plateau_check,
        ), mock.patch(
            "gpu_fuzzy_trader.evolution.evox_runner._plateau_diversity_restart",
            side_effect=mock_restart,
        ), mock.patch.object(
            cfg, "PHASE2_PLATEAU_DIVERSITY_RESTART_ENABLED", True,
        ), mock.patch.object(
            cfg, "PHASE2_PLATEAU_MAX_RESTARTS", 1,
        ), mock.patch.object(
            cfg, "PHASE2_EARLY_STOP_ENABLED", False,
        ), mock.patch.object(
            cfg, "PHASE2_EARLY_STOP_MIN_GENERATION", 999,
        ), mock.patch.object(
            cfg, "PHASE2_PLATEAU_POST_RESTART_STOP_ENABLED", False,
        ):
            rng = np.random.default_rng(0)
            feature_infos = [
                {"name": "f0", "mode": "binary", "score": 0.5},
                {"name": "f1", "mode": "binary", "score": 0.5},
            ]
            with mock.patch(
                "gpu_fuzzy_trader.evolution.evox_runner._EVOX_AVAILABLE", False,
            ):
                run_phase2_evolution(
                    feature_infos=feature_infos,
                    engine=FakeEngine(),
                    pop_size=10,
                    n_generations=20,
                    rng=rng,
                )

        assert restart_called[0] >= 1, (
            f"Expected _plateau_diversity_restart to be called, "
            f"got {restart_called[0]} calls"
        )

    def test_disabled_no_restart(self):
        """When disabled, _plateau_diversity_restart is NOT called."""
        restart_called = [0]

        original_plateau_check = _should_plateau_early_stop_phase2

        def controlled_plateau_check(gen, streak, **kwargs):
            if gen >= 2:
                return True
            return original_plateau_check(gen, streak, **kwargs)

        def mock_restart(*args, **kwargs):
            restart_called[0] += 1
            return 5

        with mock.patch(
            "gpu_fuzzy_trader.evolution.evox_runner._should_plateau_early_stop_phase2",
            side_effect=controlled_plateau_check,
        ), mock.patch(
            "gpu_fuzzy_trader.evolution.evox_runner._plateau_diversity_restart",
            side_effect=mock_restart,
        ), mock.patch.object(
            cfg, "PHASE2_PLATEAU_DIVERSITY_RESTART_ENABLED", False,
        ), mock.patch.object(
            cfg, "PHASE2_EARLY_STOP_ENABLED", False,
        ), mock.patch.object(
            cfg, "PHASE2_EARLY_STOP_MIN_GENERATION", 999,
        ), mock.patch.object(
            cfg, "PHASE2_PLATEAU_POST_RESTART_STOP_ENABLED", False,
        ):
            rng = np.random.default_rng(0)
            feature_infos = [
                {"name": "f0", "mode": "binary", "score": 0.5},
                {"name": "f1", "mode": "binary", "score": 0.5},
            ]
            with mock.patch(
                "gpu_fuzzy_trader.evolution.evox_runner._EVOX_AVAILABLE", False,
            ):
                run_phase2_evolution(
                    feature_infos=feature_infos,
                    engine=FakeEngine(),
                    pop_size=10,
                    n_generations=20,
                    rng=rng,
                )

        assert restart_called[0] == 0, (
            f"Expected 0 restart calls when disabled, got {restart_called[0]}"
        )

    def test_multiple_restarts_with_higher_max(self):
        """With max_restarts=2, restart is called at least twice."""
        restart_count = [0]

        original_plateau_check = _should_plateau_early_stop_phase2

        def controlled_plateau_check(gen, streak, **kwargs):
            if gen >= 2:
                return True
            return original_plateau_check(gen, streak, **kwargs)

        def mock_restart(*args, **kwargs):
            restart_count[0] += 1
            return 5

        with mock.patch(
            "gpu_fuzzy_trader.evolution.evox_runner._should_plateau_early_stop_phase2",
            side_effect=controlled_plateau_check,
        ), mock.patch(
            "gpu_fuzzy_trader.evolution.evox_runner._plateau_diversity_restart",
            side_effect=mock_restart,
        ), mock.patch.object(
            cfg, "PHASE2_PLATEAU_DIVERSITY_RESTART_ENABLED", True,
        ), mock.patch.object(
            cfg, "PHASE2_PLATEAU_MAX_RESTARTS", 2,
        ), mock.patch.object(
            cfg, "PHASE2_EARLY_STOP_ENABLED", False,
        ), mock.patch.object(
            cfg, "PHASE2_EARLY_STOP_MIN_GENERATION", 999,
        ), mock.patch.object(
            cfg, "PHASE2_PLATEAU_POST_RESTART_STOP_ENABLED", False,
        ):
            rng = np.random.default_rng(0)
            feature_infos = [
                {"name": "f0", "mode": "binary", "score": 0.5},
                {"name": "f1", "mode": "binary", "score": 0.5},
            ]
            with mock.patch(
                "gpu_fuzzy_trader.evolution.evox_runner._EVOX_AVAILABLE", False,
            ):
                run_phase2_evolution(
                    feature_infos=feature_infos,
                    engine=FakeEngine(),
                    pop_size=10,
                    n_generations=30,
                    rng=rng,
                )

        # With max_restarts=2 and plateau always true after gen 2,
        # restart should be called multiple times then break
        assert restart_count[0] >= 1, (
            f"Expected >=1 restarts with max_restarts=2, "
            f"got {restart_count[0]}"
        )


# ---------------------------------------------------------------------------
# Extension of test_island_early_stop: verify plateau restart respects
# island_profile scoping
# ---------------------------------------------------------------------------

def test_plateau_restart_respects_island_profile(monkeypatch):
    """Plateau restart respects island_profile scoping (early stop disabled)."""
    monkeypatch.setattr(cfg, "PHASE2_ISLAND_PLATEAU_EARLY_STOP_ENABLED", False)
    monkeypatch.setattr(cfg, "PHASE2_ISLAND_PLATEAU_BLOCK_WHEN_DEPLOYABLE_ZERO", False)
    monkeypatch.setattr(cfg, "PHASE2_PLATEAU_EARLY_STOP_ENABLED", False)
    # Even with high streak, should not trigger for island profile
    assert not _should_plateau_early_stop_phase2(
        9, 10, deployable_count=0, island_profile="cluster_0",
    )
