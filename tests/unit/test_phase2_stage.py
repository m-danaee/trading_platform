"""Unit tests for Phase 2 stage-specific hyperparameter profiles."""

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.phases.phase2_stage import (
    island_stage_budgets,
    resolve_island_stage,
    resolve_phase2_stage_params,
)
from gpu_fuzzy_trader.phases.phase2_rule_pool import compute_phase2_objectives_from_metrics
from gpu_fuzzy_trader.evolution.evox_runner import _should_inject_diversity_recovery
import numpy as np


class TestResolvePhase2StageParams:
    def test_stage_a_is_more_explorative_than_stage_b(self):
        a = resolve_phase2_stage_params("A")
        b = resolve_phase2_stage_params("B")

        assert a.mutation_rate > b.mutation_rate
        assert a.diversity_penalty > b.diversity_penalty
        assert a.diversity_hamming_threshold > b.diversity_hamming_threshold
        assert (
            a.diversity_recovery_min_unique_ratio
            > b.diversity_recovery_min_unique_ratio
        )
        assert (
            a.diversity_recovery_inject_fraction
            > b.diversity_recovery_inject_fraction
        )
        assert a.plateau_early_stop_patience > b.plateau_early_stop_patience
        assert a.seed_fraction < b.seed_fraction

    def test_single_stage_uses_global_defaults(self):
        single = resolve_phase2_stage_params(None)

        assert single.mutation_rate == _cfg.PHASE2_MUTATION_RATE
        assert single.diversity_penalty == _cfg.PHASE2_DIVERSITY_PENALTY
        assert single.seed_fraction == _cfg.PHASE2_ARCHIVE_SEED_FRACTION


class TestStageObjectivePenalties:
    def test_stage_a_applies_stronger_diversity_penalty(self):
        chromosome = np.array([0, 1, 2], dtype=np.int32)
        dont_cares = np.array([3, 3, 3], dtype=np.int32)
        ref = chromosome.copy()
        metrics = {
            "sortino_ratio": 2.0,
            "total_return_pct": 5.0,
            "profit_factor": 1.5,
            "max_drawdown_pct": 4.0,
            "win_rate": 55.0,
            "executed_trades": 200,
        }
        stage_a = resolve_phase2_stage_params("A")
        stage_b = resolve_phase2_stage_params("B")

        obj_a, _ = compute_phase2_objectives_from_metrics(
            chromosome,
            dont_cares,
            metrics,
            [ref],
            diversity_reference=[ref],
            stage_params=stage_a,
        )
        obj_b, _ = compute_phase2_objectives_from_metrics(
            chromosome,
            dont_cares,
            metrics,
            [ref],
            diversity_reference=[ref],
            stage_params=stage_b,
        )

        assert obj_a[0] > obj_b[0]
        assert obj_a[1] > obj_b[1]
        assert obj_a[2] > obj_b[2]


class TestIslandStageBudgets:
    def test_island_budgets_scale_total_generations(self, monkeypatch):
        monkeypatch.setattr(_cfg, "PHASE2_TWO_STAGE_ENABLED", True)
        monkeypatch.setattr(_cfg, "PHASE2_STAGE_A_GENERATIONS", 80)
        monkeypatch.setattr(_cfg, "PHASE2_STAGE_B_GENERATIONS", 40)

        stage_a, stage_b = island_stage_budgets(80)
        assert stage_a + stage_b == 80
        assert stage_a > stage_b

    def test_resolve_island_stage_transitions_to_b(self, monkeypatch):
        monkeypatch.setattr(_cfg, "PHASE2_TWO_STAGE_ENABLED", True)
        monkeypatch.setattr(_cfg, "PHASE2_STAGE_A_GENERATIONS", 80)
        monkeypatch.setattr(_cfg, "PHASE2_STAGE_B_GENERATIONS", 40)

        stage_a, stage_b = island_stage_budgets(80)
        plan_a = resolve_island_stage(0, 80)
        assert plan_a.stage == "A"
        assert plan_a.entering_stage_b is False

        plan_b = resolve_island_stage(stage_a, 80)
        assert plan_b.stage == "B"
        assert plan_b.entering_stage_b is True
        assert plan_b.remaining_in_stage == stage_b


class TestParetoCollapseDiversityRecovery:
    def test_triggers_when_pareto_collapses_despite_unique_population(self):
        assert _should_inject_diversity_recovery(
            1.0,
            pareto_size=1,
            plateau_streak=2,
            pop_size=200,
        )

    def test_does_not_trigger_on_fresh_population(self):
        assert not _should_inject_diversity_recovery(
            1.0,
            pareto_size=1,
            plateau_streak=0,
            pop_size=200,
        )
