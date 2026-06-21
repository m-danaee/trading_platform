"""Unit tests for Phase 2 stage-specific hyperparameter profiles."""

import numpy as np

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.phases.phase2_rule_pool import (
    _diversity_penalty_for_chromosome,
    _stage_b_seed_chromosomes,
)
from gpu_fuzzy_trader.phases.phase2_stage import resolve_phase2_stage_params


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

    def test_stage_a_floor_overrides(self):
        stage_a = resolve_phase2_stage_params("A")
        stage_b = resolve_phase2_stage_params("B")

        assert stage_a.return_floor_pct == _cfg.PHASE2_STAGE_A_RETURN_FLOOR_PCT
        assert stage_a.min_trade_support == _cfg.PHASE2_STAGE_A_MIN_TRADE_SUPPORT
        assert stage_a.use_robust_return_obj == _cfg.PHASE2_STAGE_A_USE_ROBUST_RETURN_OBJ
        assert stage_a.soft_feasibility is True
        assert stage_a.pool_require_positive_splits is False

        assert stage_b.return_floor_pct == _cfg.PHASE2_RETURN_FLOOR_PCT
        assert stage_b.min_trade_support == _cfg.MIN_TRADE_SUPPORT
        assert stage_b.use_robust_return_obj == _cfg.PHASE2_USE_ROBUST_RETURN_OBJ
        assert stage_b.soft_feasibility is False
        assert stage_b.pool_require_positive_splits == _cfg.PHASE2_POOL_REQUIRE_POSITIVE_SPLITS


class TestStageDiversityPenalty:
    def test_stage_a_applies_stronger_diversity_penalty(self):
        chromosome = np.array([0, 1, 2], dtype=np.int32)
        ref = np.array([0, 1, 0], dtype=np.int32)
        stage_a = resolve_phase2_stage_params("A")
        stage_b = resolve_phase2_stage_params("B")

        pen_a = _diversity_penalty_for_chromosome(
            chromosome, [ref], stage_params=stage_a,
        )
        pen_b = _diversity_penalty_for_chromosome(
            chromosome, [ref], stage_params=stage_b,
        )

        assert pen_a > pen_b
        assert pen_a == _cfg.PHASE2_STAGE_A_DIVERSITY_PENALTY
        assert pen_b == _cfg.PHASE2_STAGE_B_DIVERSITY_PENALTY

    def test_no_penalty_when_hamming_above_threshold(self):
        chromosome = np.array([0, 0, 0, 0, 0], dtype=np.int32)
        ref = np.array([1, 1, 1, 1, 1], dtype=np.int32)
        stage_a = resolve_phase2_stage_params("A")

        pen = _diversity_penalty_for_chromosome(
            chromosome, [ref], stage_params=stage_a,
        )
        assert pen == 0.0


class TestStageBSeedChromosomes:
    def test_merges_stage_a_elites_with_base_seeds(self):
        stage_a = [
            {
                "chromosome": [0, 1, 2],
                "objectives": {"sortino_ratio": 3.0},
                "executed_trades": 100,
            },
            {
                "chromosome": [1, 2, 0],
                "objectives": {"sortino_ratio": 2.0},
                "executed_trades": 100,
            },
        ]
        base = np.array([[9, 9, 9]], dtype=np.int32)
        seeds = _stage_b_seed_chromosomes(stage_a, base, top_k=1)
        assert seeds is not None
        assert seeds.shape[0] == 2
        keys = {tuple(row.tolist()) for row in seeds}
        assert (0, 1, 2) in keys
        assert (9, 9, 9) in keys


class TestConfigBudget:
    def test_stage_generations_sum_to_total(self):
        assert (
            _cfg.PHASE2_STAGE_A_GENERATIONS + _cfg.PHASE2_STAGE_B_GENERATIONS
            == _cfg.PHASE2_GENERATIONS
        )
