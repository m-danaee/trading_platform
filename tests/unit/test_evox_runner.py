"""Unit tests for Phase 2 NSGA-III runner."""

from __future__ import annotations

import logging
from unittest import mock

import numpy as np
import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.evolution.evox_runner import (
    _EVOX_AVAILABLE,
    _deduplicate_selection_indices,
    _evaluate_population_indices,
    _harvest_archive_chromosomes,
    _normalize_for_association,
    _nsga3_environmental_selection,
    _pareto_robust_stats,
    _plateau_progress_metric,
    _population_unique_chromosome_ratio,
    _resolve_plateau_min_generation,
    _resolve_plateau_patience,
    _survivors_missing_cached_validation,
    _should_inject_diversity_recovery,
    _should_plateau_early_stop_phase2,
    _update_deployable_archive,
    _update_hall_of_fame,
    _update_max_return_plateau,
    Phase2EvolutionState,
    run_phase2_evolution,
    run_phase2_evolution_epoch,
)


def test_survivor_validation_refresh_only_targets_missing_snapshots():
    metrics_cache = [
        {"val_total_return_pct": 0.0},
        {"val_total_return_pct": None},
        {},
    ]
    assert _survivors_missing_cached_validation(
        [0, 1, 2], metrics_cache,
    ) == [1, 2]


def _chromosome_with_min_active(
    n_features: int = 10,
    dont_care: int = 5,
) -> np.ndarray:
    """Build a dense chromosome with exactly MIN_CONDITIONS active genes."""
    chrom = np.full(n_features, dont_care, dtype=np.int32)
    chrom[: int(_cfg.MIN_CONDITIONS)] = 0
    return chrom


class TestHallOfFame:
    def test_update_hall_of_fame_accumulates_unique_pareto(self):
        hall: dict[tuple[int, ...], np.ndarray] = {}
        pop_gen0 = np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int32)
        _update_hall_of_fame(hall, pop_gen0, [0, 2])

        pop_gen1 = np.array([[0, 1], [6, 7], [8, 9]], dtype=np.int32)
        _update_hall_of_fame(hall, pop_gen1, [0, 1, 2])

        assert set(hall.keys()) == {
            (0, 1),
            (4, 5),
            (6, 7),
            (8, 9),
        }


@pytest.mark.skipif(not _EVOX_AVAILABLE, reason="EvoX not installed")
class TestRunPhase2EvolutionSmoke:
    def test_nsga3_one_generation(self):
        feature_infos = [
            {"name": "feat_0", "mode": "binary", "score": 0.5},
            {"name": "feat_1", "mode": "binary", "score": 0.5},
        ]

        class FakeEngine:
            def simulate_rule_batch(
                self, chromosomes, tp, sl, capital_pct, **kwargs,
            ):
                B = chromosomes.shape[0]
                return [
                    {
                        "sortino_ratio": 1.0,
                        "total_return_pct": 1.0,
                        "max_drawdown_pct": 2.0,
                        "win_rate": 50.0,
                        "executed_trades": 25,
                    }
                    for _ in range(B)
                ]

        rng = np.random.default_rng(0)
        pool, history = run_phase2_evolution(
            feature_infos=feature_infos,
            engine=FakeEngine(),
            pop_size=8,
            n_generations=2,
            rng=rng,
        )
        assert isinstance(pool, list)
        assert len(history) == 2
        assert history[0]["algorithm"] == "NSGA-III"


class TestRunPhase2EvolutionFallback:
    def test_missing_evox_uses_nsga2(self):
        feature_infos = [
            {"name": "feat_0", "mode": "binary", "score": 0.5},
        ]

        class FakeEngine:
            def simulate_rule_batch(
                self, chromosomes, tp, sl, capital_pct, **kwargs,
            ):
                B = chromosomes.shape[0]
                return [
                    {
                        "sortino_ratio": 1.0,
                        "total_return_pct": 1.0,
                        "max_drawdown_pct": 2.0,
                        "win_rate": 50.0,
                        "executed_trades": 25,
                    }
                    for _ in range(B)
                ]

        rng = np.random.default_rng(1)
        with mock.patch("gpu_fuzzy_trader.evolution.evox_runner._EVOX_AVAILABLE", False):
            pool, history = run_phase2_evolution(
                feature_infos=feature_infos,
                engine=FakeEngine(),
                pop_size=6,
                n_generations=1,
                rng=rng,
            )
        assert len(history) == 1
        assert history[0].get("algorithm") == "NSGA-II (fallback)"
        assert "sortino_cap_hit_fraction" in history[0]
        assert "objective_std_f1" in history[0]
        assert "objective_std_f2" in history[0]
        assert "objective_std_f3" in history[0]

    def test_seed_chromosomes_forwarded_to_fallback(self):
        feature_infos = [
            {"name": "feat_0", "mode": "binary", "score": 0.5},
        ]

        class FakeEngine:
            def simulate_rule_batch(
                self, chromosomes, tp, sl, capital_pct, **kwargs,
            ):
                B = chromosomes.shape[0]
                return [
                    {
                        "sortino_ratio": 1.0,
                        "total_return_pct": 1.0,
                        "max_drawdown_pct": 2.0,
                        "win_rate": 50.0,
                        "executed_trades": 25,
                    }
                    for _ in range(B)
                ]

        rng = np.random.default_rng(2)
        seeds = np.array([[0]], dtype=np.int32)
        captured = {}

        def fake_fallback(
            feature_infos,
            engine,
            pop_size,
            n_generations,
            rng,
            seed_chromosomes=None,
            log_tag=None,
            val_engine=None,
            **kwargs,
        ):
            captured["seed_chromosomes"] = seed_chromosomes
            return [], []

        with mock.patch("gpu_fuzzy_trader.evolution.evox_runner._EVOX_AVAILABLE", False), \
                mock.patch("gpu_fuzzy_trader.evolution.evox_runner._run_nsga2_fallback", side_effect=fake_fallback):
            run_phase2_evolution(
                feature_infos=feature_infos,
                engine=FakeEngine(),
                pop_size=6,
                n_generations=1,
                rng=rng,
                seed_chromosomes=seeds,
            )

        assert np.array_equal(captured["seed_chromosomes"], seeds)

    def test_fallback_attaches_cv_returns_to_offspring(self, monkeypatch):
        """The fallback must not switch f3 from CV return to PF after gen 0."""
        feature_infos = [
            {"name": "feat_0", "mode": "binary", "score": 0.5},
            {"name": "feat_1", "mode": "binary", "score": 0.5},
        ]

        class FakeEngine:
            def simulate_rule_batch(
                self, chromosomes, tp, sl, capital_pct, **kwargs,
            ):
                return [
                    {
                        "sortino_ratio": 1.0,
                        "total_return_pct": 1.0,
                        "max_drawdown_pct": 2.0,
                        "win_rate": 50.0,
                        "profit_factor": 1.2,
                        "executed_trades": 25,
                    }
                    for _ in range(len(chromosomes))
                ]

        class FakeCvEvaluator:
            def __init__(self):
                self.calls = 0

            def _ensure_fold_engines(self):
                self.calls += 1
                return [FakeEngine()]

        cv = FakeCvEvaluator()
        monkeypatch.setattr(_cfg, "PHASE2_USE_TOTAL_RETURN_OBJ", False)
        monkeypatch.setattr(_cfg, "PHASE2_F3_OBJECTIVE", "cv_fold_min")
        with mock.patch("gpu_fuzzy_trader.evolution.evox_runner._EVOX_AVAILABLE", False):
            run_phase2_evolution(
                feature_infos=feature_infos,
                engine=FakeEngine(),
                pop_size=6,
                n_generations=2,
                rng=np.random.default_rng(3),
                cv_fold_evaluator=cv,
            )

        assert cv.calls >= 2


def test_low_trade_drawdown_penalty():
    # Set trade floor to 25
    orig_floor = _cfg.MIN_TRADE_POOL_FLOOR
    orig_support = _cfg.MIN_TRADE_SUPPORT
    try:
        _cfg.MIN_TRADE_POOL_FLOOR = 25
        _cfg.MIN_TRADE_SUPPORT = 5
        
        # population of 1 candidate with valid active count (no cond penalty)
        pop = _chromosome_with_min_active()[np.newaxis, :]
        dont_cares = np.ones(10, dtype=np.int32) * 5
        objectives = np.full((1, 4), np.inf)
        metrics_cache = [{}]
        
        class MockEngine:
            def simulate_rule_batch(self, chromosomes, **kwargs):
                # return metrics with 5 executed trades and 0.0 drawdown
                return [{"executed_trades": 5, "total_return_pct": 1.0, "sortino_ratio": 0.5, "max_drawdown_pct": 0.0, "win_rate": 0.5}]
                
        engine = MockEngine()
        _evaluate_population_indices(
            pop, [0], dont_cares, engine, [], objectives, metrics_cache
        )
        
        # Task 3: trade_penalty is now on f2 only (decoupled objectives).
        # f1 gets no trade_penalty, f2 gets the full penalty (DD + trade),
        # f3 gets no trade_penalty.
        # f1 has PF-floor and min-symbols penalties from missing mock fields.
        assert objectives[0, 0] > 0.0  # support penalty from pf/min_symbols, no trade_penalty
        assert objectives[0, 1] >= 150.0  # dd_for_obj(100) + penalty(50)
        assert objectives[0, 2] < objectives[0, 1]  # f3 < f2 (f3 has no trade_penalty, f2 does)
    finally:
        _cfg.MIN_TRADE_POOL_FLOOR = orig_floor
        _cfg.MIN_TRADE_SUPPORT = orig_support


def test_phase2_use_total_return_obj():
    has_orig = hasattr(_cfg, "PHASE2_USE_TOTAL_RETURN_OBJ")
    orig_val = getattr(_cfg, "PHASE2_USE_TOTAL_RETURN_OBJ", False)
    orig_floor = _cfg.MIN_TRADE_SUPPORT
    orig_val_penalty = bool(
        getattr(_cfg, "PHASE2_VAL_IN_FITNESS_PENALTY", False))
    orig_pool_pos = bool(
        getattr(_cfg, "PHASE2_POOL_REQUIRE_POSITIVE_SPLITS", True))

    try:
        _cfg.PHASE2_USE_TOTAL_RETURN_OBJ = True
        _cfg.MIN_TRADE_SUPPORT = 5
        # Isolate f3 identity: missing val_metrics otherwise adds a fixed
        # feasibility penalty when VAL_IN_FITNESS_PENALTY is on.
        _cfg.PHASE2_VAL_IN_FITNESS_PENALTY = False
        _cfg.PHASE2_POOL_REQUIRE_POSITIVE_SPLITS = False

        pop = _chromosome_with_min_active()[np.newaxis, :]
        dont_cares = np.ones(10, dtype=np.int32) * 5
        objectives = np.full((1, 4), np.inf)
        metrics_cache = [{}]

        class MockEngine:
            def simulate_rule_batch(self, chromosomes, **kwargs):
                return [{
                    "executed_trades": 100,
                    "total_return_pct": 15.0,
                    "sortino_ratio": 0.5,
                    "max_drawdown_pct": 2.0,
                    "win_rate": 50.0,
                    "profit_factor": 5.0,
                    "per_symbol_metrics": {
                        "SYM1": {"net_pnl": 100.0},
                        "SYM2": {"net_pnl": 200.0},
                        "SYM3": {"net_pnl": 300.0},
                    },
                }]

        engine = MockEngine()
        _evaluate_population_indices(
            pop, [0], dont_cares, engine, [], objectives, metrics_cache
        )

        # With total return obj enabled: f3 = -total_return_pct = -15.0 (plus penalties)
        # Objectives are: [-sortino + pen, dd + pen, -total_return + pen]
        # Sortino obj should be -0.5 + pen, DD should be 2.0 + pen, F3 should be -15.0 + pen
        assert np.isclose(objectives[0, 2], -15.0)

        # Disable it -> should use PHASE2_F3_OBJECTIVE (profit_factor = 5.0)
        _cfg.PHASE2_USE_TOTAL_RETURN_OBJ = False
        _cfg.PHASE2_F3_OBJECTIVE = "profit_factor"
        objectives[0] = np.inf
        _evaluate_population_indices(
            pop, [0], dont_cares, engine, [], objectives, metrics_cache
        )
        # With profit_factor: f3 = -profit_factor = -5.0 (plus penalties)
        assert np.isclose(objectives[0, 2], -5.0)

        # Set PHASE2_F3_OBJECTIVE to "win_rate" -> should use win_rate = 50.0
        _cfg.PHASE2_F3_OBJECTIVE = "win_rate"
        objectives[0] = np.inf
        _evaluate_population_indices(
            pop, [0], dont_cares, engine, [], objectives, metrics_cache
        )
        assert np.isclose(objectives[0, 2], -50.0)

    finally:
        _cfg.MIN_TRADE_SUPPORT = orig_floor
        _cfg.PHASE2_VAL_IN_FITNESS_PENALTY = orig_val_penalty
        _cfg.PHASE2_POOL_REQUIRE_POSITIVE_SPLITS = orig_pool_pos
        if has_orig:
            _cfg.PHASE2_USE_TOTAL_RETURN_OBJ = orig_val
        else:
            if hasattr(_cfg, "PHASE2_USE_TOTAL_RETURN_OBJ"):
                delattr(_cfg, "PHASE2_USE_TOTAL_RETURN_OBJ")
        # Restore PHASE2_F3_OBJECTIVE if it was modified
        if hasattr(_cfg, "PHASE2_F3_OBJECTIVE"):
            _cfg.PHASE2_F3_OBJECTIVE = "profit_factor"


def _default_eval_metrics(executed_trades: int = 25) -> dict:
    return {
        "executed_trades": executed_trades,
        "sortino_ratio": 1.0,
        "max_drawdown_pct": 2.0,
        "win_rate": 50.0,
        "total_return_pct": 1.0,
    }


class TestEvalOptimizations:
    def test_batch_dedup_evaluates_unique_chromosomes_once(self):
        call_sizes: list[int] = []

        class CountingEngine:
            def simulate_rule_batch(
                self, chromosomes, tp, sl, capital_pct, **kwargs,
            ):
                call_sizes.append(chromosomes.shape[0])
                return [
                    _default_eval_metrics()
                    for _ in range(chromosomes.shape[0])
                ]

        pop = np.array([[0, 1], [0, 1], [2, 3]], dtype=np.int32)
        objectives = np.full((3, 4), np.inf)
        metrics_cache: list[dict] = [{}, {}, {}]
        dont_cares = np.array([2, 2], dtype=np.int32)

        _evaluate_population_indices(
            pop,
            [0, 1, 2],
            dont_cares,
            CountingEngine(),
            [],
            objectives,
            metrics_cache,
            global_metrics_cache={},
        )

        assert call_sizes == [2]
        assert not np.any(np.isinf(objectives))

    def test_global_cache_skips_second_gpu_eval(self):
        call_count = 0

        class CountingEngine:
            def simulate_rule_batch(
                self, chromosomes, tp, sl, capital_pct, **kwargs,
            ):
                nonlocal call_count
                call_count += 1
                return [
                    _default_eval_metrics()
                    for _ in range(chromosomes.shape[0])
                ]

        pop = np.array([[1, 0], [1, 0]], dtype=np.int32)
        dont_cares = np.array([2, 2], dtype=np.int32)
        engine = CountingEngine()
        global_cache: dict[tuple[int, ...], dict] = {}

        objectives_first = np.full((1, 4), np.inf)
        metrics_first: list[dict] = [{}]
        _evaluate_population_indices(
            pop[:1],
            [0],
            dont_cares,
            engine,
            [],
            objectives_first,
            metrics_first,
            global_metrics_cache=global_cache,
        )

        objectives_second = np.full((1, 4), np.inf)
        metrics_second: list[dict] = [{}]
        _evaluate_population_indices(
            pop[1:2],
            [0],
            dont_cares,
            engine,
            [],
            objectives_second,
            metrics_second,
            global_metrics_cache=global_cache,
        )

        assert call_count == 1
        assert np.allclose(objectives_first[0], objectives_second[0])
        assert metrics_first[0]["executed_trades"] == 25
        assert metrics_second[0]["executed_trades"] == 25

    def test_global_cache_recomputes_diversity_penalty(self):
        call_count = 0

        class CountingEngine:
            def simulate_rule_batch(
                self, chromosomes, tp, sl, capital_pct, **kwargs,
            ):
                nonlocal call_count
                call_count += 1
                return [_default_eval_metrics()]

        pop = np.array([[1, 0]], dtype=np.int32)
        dont_cares = np.array([2, 2], dtype=np.int32)
        engine = CountingEngine()
        global_cache: dict[tuple[int, ...], dict] = {}
        objectives = np.full((1, 4), np.inf)
        metrics_cache: list[dict] = [{}]

        _evaluate_population_indices(
            pop,
            [0],
            dont_cares,
            engine,
            [],
            objectives,
            metrics_cache,
            global_metrics_cache=global_cache,
        )
        obj_no_archive = objectives[0].copy()

        objectives[0] = np.inf
        near_duplicate = np.array([0, 0], dtype=np.int32)
        _evaluate_population_indices(
            pop,
            [0],
            dont_cares,
            engine,
            [near_duplicate],
            objectives,
            metrics_cache,
            global_metrics_cache=global_cache,
        )
        obj_with_archive = objectives[0]

        assert call_count == 1
        # Diversity lands on f4 when PHASE2_DIVERSITY_ON_F4 (default); f1 unchanged.
        assert obj_with_archive[3] > obj_no_archive[3]
        assert np.isclose(obj_with_archive[0], obj_no_archive[0])


class TestPlateauEarlyStop:
    def test_update_resets_streak_on_improvement(self):
        best, streak = _update_max_return_plateau(5.0, -np.inf, 3)
        assert best == 5.0
        assert streak == 0

        best, streak = _update_max_return_plateau(5.0, best, streak)
        assert best == 5.0
        assert streak == 1

        delta = float(_cfg.PHASE2_PLATEAU_EARLY_STOP_MIN_DELTA_PCT)
        best, streak = _update_max_return_plateau(
            5.0 + delta + 0.01, best, streak)
        assert best == 5.0 + delta + 0.01
        assert streak == 0

    def test_should_stop_after_patience(self, monkeypatch):
        monkeypatch.setattr(_cfg, "PHASE2_PLATEAU_EARLY_STOP_ENABLED", True)
        monkeypatch.setattr(
            _cfg, "PHASE2_PLATEAU_EARLY_STOP_MIN_GENERATION", 5)
        monkeypatch.setattr(_cfg, "PHASE2_PLATEAU_EARLY_STOP_PATIENCE", 3)
        monkeypatch.setattr(
            _cfg, "PHASE2_PLATEAU_BLOCK_WHEN_DEPLOYABLE_ZERO", True)
        monkeypatch.setattr(
            _cfg, "PHASE2_PLATEAU_BLOCK_WHEN_DIVERSITY_LOW", True)

        assert not _should_plateau_early_stop_phase2(3, 3)
        assert not _should_plateau_early_stop_phase2(10, 2)
        assert not _should_plateau_early_stop_phase2(
            10, 3, deployable_count=0, unique_chromosome_ratio=1.0,
        )
        assert not _should_plateau_early_stop_phase2(
            10,
            3,
            deployable_count=2,
            unique_chromosome_ratio=1.0,
            population_unique_ratio=0.1,
        )
        assert _should_plateau_early_stop_phase2(
            10,
            3,
            deployable_count=2,
            unique_chromosome_ratio=1.0,
            population_unique_ratio=0.5,
        )


class TestNormalizeForAssociation:
    def test_inf_objectives_do_not_raise(self):
        merge_fit = np.array(
            [[1.0, 2.0, 3.0], [np.inf, 2.0, 3.0], [2.0, 3.0, 4.0]],
            dtype=np.float64,
        )
        ref = np.array([[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]], dtype=np.float64)
        fit_n, ref_n = _normalize_for_association(merge_fit, ref)
        assert np.all(np.isfinite(fit_n))
        assert np.all(np.isfinite(ref_n))


class TestPopulationDiversityMetrics:
    def test_population_unique_ratio_counts_full_population(self):
        pop = np.array(
            [[0, 1], [0, 1], [2, 3], [4, 5]],
            dtype=np.int32,
        )
        assert _population_unique_chromosome_ratio(pop) == 0.75

    def test_diversity_recovery_pads_when_archive_smaller_than_inject(
        self, monkeypatch,
    ):
        """Tiny deployable archive must not IndexError on viability recovery."""
        from gpu_fuzzy_trader.evolution.evox_runner import _inject_diversity_recovery
        from gpu_fuzzy_trader.phases.phase2_rule_pool import _init_population
        from gpu_fuzzy_trader.phases.phase2_stage import resolve_phase2_stage_params

        monkeypatch.setattr(
            _cfg, "PHASE2_VIABILITY_RECOVERY_DEPLOYABLE_MUTATE_FRACTION", 0.8,
        )
        pop_size = 60
        feature_infos = [
            {"name": f"f{i}", "n_terms": 3, "mode": "ternary"} for i in range(8)
        ]
        rng = np.random.default_rng(0)
        population = _init_population(pop_size, feature_infos, rng)
        objectives = np.ones((pop_size, 4))
        metrics_cache = [{} for _ in range(pop_size)]
        dont_cares = np.full(8, 3, dtype=np.int32)
        archive = {
            "k1": {"chromosome": population[0].copy(), "rank_score": 1.0},
        }
        stage = resolve_phase2_stage_params("A")
        injected = _inject_diversity_recovery(
            population,
            objectives,
            metrics_cache,
            feature_infos,
            dont_cares,
            rng,
            stage_params=stage,
            deployable_archive=archive,
            viability_recovery=True,
        )
        n_inject = max(
            1,
            int(round(pop_size * float(stage.diversity_recovery_inject_fraction))),
        )
        assert len(injected) == n_inject
        for i in injected:
            assert np.all(np.isinf(objectives[i]))

    def test_diversity_recovery_uses_population_not_pareto(self, monkeypatch):
        monkeypatch.setattr(_cfg, "PHASE2_DIVERSITY_RECOVERY_ENABLED", True)
        monkeypatch.setattr(
            _cfg, "PHASE2_DIVERSITY_RECOVERY_MIN_UNIQUE_RATIO", 0.30)
        assert not _should_inject_diversity_recovery(1.0)
        assert _should_inject_diversity_recovery(0.1)

    def test_phenotype_collapse_trigger_fires_on_small_pareto_with_streak(self, monkeypatch):
        """Test that Check 3 fires when pareto_size=3, plateau_streak=2, pop_size=100.

        For pop_size=100, Check 2 threshold is max(2, 100 // 40) = 2.
        So pareto_size=3 does NOT match Check 2 (3 > 2), but DOES match Check 3 (3 <= 3).
        This isolates the phenotype-collapse trigger.
        """
        monkeypatch.setattr(_cfg, "PHASE2_DIVERSITY_RECOVERY_ENABLED", True)
        # pop_size=100 → Check 2 threshold = 2, so pareto_size=3 bypasses Check 2
        # Check 3 should fire: pareto_size=3 <= 3 and plateau_streak=2 >= 2
        assert _should_inject_diversity_recovery(
            population_unique_ratio=1.0,  # high genetic uniqueness
            pareto_size=3,
            plateau_streak=2,
            pop_size=100,
            valid_count=50,
        ) is True

    def test_phenotype_collapse_does_not_fire_without_streak(self, monkeypatch):
        """Check 3 requires plateau_streak >= 2 (isolated with pop_size=100)."""
        monkeypatch.setattr(_cfg, "PHASE2_DIVERSITY_RECOVERY_ENABLED", True)
        # pop_size=100 → Check 2 threshold = 2
        # pareto_size=3 → 3 > 2, so Check 2 does NOT fire
        # plateau_streak=1 → < 2, so Check 3 does NOT fire
        assert _should_inject_diversity_recovery(
            population_unique_ratio=1.0,
            pareto_size=3,
            plateau_streak=1,  # below Check 3 threshold
            pop_size=100,
            valid_count=50,
        ) is False

    def test_phenotype_collapse_boundary_pop_size_100(self, monkeypatch):
        """For pop_size=100, Check 2 threshold=2, so pareto_size=4 should NOT trigger."""
        monkeypatch.setattr(_cfg, "PHASE2_DIVERSITY_RECOVERY_ENABLED", True)
        # Check 2: pareto_size=4 > threshold=2 → False
        # Check 3: pareto_size=4 > 3 → False
        # Should return False (assuming no other checks fire)
        assert _should_inject_diversity_recovery(
            population_unique_ratio=1.0,
            pareto_size=4,
            plateau_streak=2,
            pop_size=100,
            valid_count=50,
        ) is False

    def test_phenotype_collapse_respects_master_switch(self, monkeypatch):
        """Check 3 respects PHASE2_DIVERSITY_RECOVERY_ENABLED."""
        monkeypatch.setattr(_cfg, "PHASE2_DIVERSITY_RECOVERY_ENABLED", False)
        assert _should_inject_diversity_recovery(
            population_unique_ratio=1.0,
            pareto_size=2,
            plateau_streak=3,
            pop_size=100,
            valid_count=50,
        ) is False


class TestDeployableArchive:
    def test_harvest_prefers_deployable_archive(self):
        deployable = {
            (0, 1): {"chromosome": np.array([0, 1], dtype=np.int32), "rank_score": 5.0},
        }
        hall = {(2, 3): np.array([2, 3], dtype=np.int32)}
        pareto = [np.array([4, 5], dtype=np.int32)]
        harvested = _harvest_archive_chromosomes(deployable, hall, pareto)
        assert len(harvested) == 1
        assert np.array_equal(harvested[0], np.array([0, 1], dtype=np.int32))

    def test_update_deployable_archive_keeps_best_rank(self, monkeypatch):
        monkeypatch.setattr(_cfg, "PHASE2_DEPLOYABLE_ARCHIVE_MAX_SIZE", 10)
        monkeypatch.setattr(_cfg, "PHASE2_POOL_REQUIRE_POSITIVE_SPLITS", True)
        monkeypatch.setattr(_cfg, "MIN_TRADE_POOL_FLOOR", 10)

        archive: dict = {}
        pop = np.array([[0, 1], [0, 1]], dtype=np.int32)
        metrics_cache = [
            {
                "executed_trades": 50,
                "total_return_pct": 3.0,
                "profit_factor": 1.2,
                "sortino_ratio": 1.0,
                "max_drawdown_pct": 2.0,
                "val_total_return_pct": 2.0,
                "val_profit_factor": 1.1,
                "val_executed_trades": 20,
            },
            {
                "executed_trades": 50,
                "total_return_pct": 5.0,
                "profit_factor": 1.3,
                "sortino_ratio": 1.5,
                "max_drawdown_pct": 2.0,
                "val_total_return_pct": 4.0,
                "val_profit_factor": 1.2,
                "val_executed_trades": 20,
            },
        ]
        _update_deployable_archive(archive, pop, [0, 1], metrics_cache)
        assert len(archive) == 1
        assert float(archive[(0, 1)]["rank_score"]) > 0.0

    def test_plateau_progress_uses_robust_return(self, monkeypatch):
        monkeypatch.setattr(_cfg, "PHASE2_PLATEAU_USE_ROBUST_RETURN", True)
        monkeypatch.setattr(_cfg, "PHASE2_JOINT_TRAIN_VAL", True)
        metrics_cache = [
            {
                "total_return_pct": 10.0,
                "val_total_return_pct": 2.0,
            },
            {
                "total_return_pct": 4.0,
                "val_total_return_pct": 3.5,
            },
        ]
        progress = _plateau_progress_metric([0, 1], metrics_cache)
        assert progress == pytest.approx(3.5)


class TestBatchSingleObjectiveAlignment:
    def test_return_floor_penalty_matches_single_path(self, monkeypatch):
        from gpu_fuzzy_trader.phases.phase2_rule_pool import (
            _evaluate_chromosome,
            compute_phase2_objectives_from_metrics,
        )

        monkeypatch.setattr(_cfg, "PHASE2_RETURN_FLOOR_PCT", 2.0)
        monkeypatch.setattr(_cfg, "PHASE2_JOINT_TRAIN_VAL", True)

        pop = _chromosome_with_min_active()[np.newaxis, :]
        dont_cares = np.ones(10, dtype=np.int32) * 5
        metrics = {
            "executed_trades": 100,
            "total_return_pct": 1.0,
            "sortino_ratio": 0.5,
            "max_drawdown_pct": 2.0,
            "win_rate": 50.0,
            "profit_factor": 1.2,
        }
        val_metrics = {
            "executed_trades": 50,
            "total_return_pct": 3.0,
            "sortino_ratio": 1.0,
            "max_drawdown_pct": 1.0,
            "win_rate": 55.0,
            "profit_factor": 1.3,
        }

        batch_obj, _ = compute_phase2_objectives_from_metrics(
            pop[0], dont_cares, dict(metrics), [],
            val_metrics=dict(val_metrics),
        )

        class MockEngine:
            def simulate_rule_batch(self, chromosomes, **kwargs):
                return [dict(metrics)]

        class MockValEngine:
            def simulate_rule_batch(self, chromosomes, **kwargs):
                return [dict(val_metrics)]

        single_obj, _ = _evaluate_chromosome(
            pop[0], dont_cares, MockEngine(), [],
            val_engine=MockValEngine(),
        )

        assert np.allclose(batch_obj, single_obj)


class TestDuplicateSuppression:
    def test_deduplicate_replaces_clone_with_unique_genotype(self):
        merge_pop = np.array([[0], [0], [1], [2]], dtype=np.int32)
        merge_fit = np.array([
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0],
        ])
        selected = np.array([0, 0, 1], dtype=np.intp)
        idx = _deduplicate_selection_indices(selected, merge_pop, merge_fit, 3)
        keys = {tuple(int(x) for x in merge_pop[i]) for i in idx}
        assert len(keys) == 3

    @pytest.mark.skipif(not _EVOX_AVAILABLE, reason="EvoX not installed")
    def test_nsga3_survivors_limit_duplicate_genotypes(self):
        feature_infos = [
            {"name": "feat_0", "mode": "binary", "score": 0.5},
        ]
        dont_cares = np.array([2], dtype=np.int32)
        merge_pop = np.vstack([
            np.array([[0]], dtype=np.int32),
            np.array([[0]], dtype=np.int32),
            np.array([[1]], dtype=np.int32),
            np.array([[2]], dtype=np.int32),
        ])
        merge_fit = np.array([
            [1.0, 5.0, 5.0],
            [1.0, 4.0, 4.0],
            [2.0, 2.0, 2.0],
            [3.0, 1.0, 1.0],
        ], dtype=np.float64)
        ref = np.array([[1.0, 1.0, 1.0]], dtype=np.float64)
        pop, _, sel_idx = _nsga3_environmental_selection(
            merge_pop, merge_fit, ref, 3, feature_infos, dont_cares,
        )
        assert len(pop) == 3
        keys = {tuple(int(x) for x in row) for row in pop}
        assert len(keys) == 3
        assert len(sel_idx) == 3


class TestParetoRobustStats:
    def test_uses_min_train_val_return(self, monkeypatch):
        monkeypatch.setattr(_cfg, "PHASE2_JOINT_TRAIN_VAL", True)
        metrics_cache = [{
            "total_return_pct": 5.0,
            "sortino_ratio": 2.0,
            "val_total_return_pct": 1.0,
            "val_sortino_ratio": 0.5,
            "val_profit_factor": 1.1,
            "val_executed_trades": 20,
            "val_max_drawdown_pct": 1.0,
        }]
        stats = _pareto_robust_stats([0], metrics_cache)
        assert stats["max_robust_return_pct"] == pytest.approx(1.0)
        assert stats["mean_robust_return_pct"] == pytest.approx(1.0)


class TestResolvePlateauPatience:
    """Tests for _resolve_plateau_patience helper."""

    def test_global_profile_uses_global_patience(self, monkeypatch):
        monkeypatch.setattr(_cfg, "PHASE2_PLATEAU_EARLY_STOP_PATIENCE", 8)
        assert _resolve_plateau_patience(None) == 8

    def _make_stage_params(self, patience: int):
        """Helper to create a Phase2StageParams with controlled patience."""
        from gpu_fuzzy_trader.phases.phase2_stage import Phase2StageParams
        return Phase2StageParams(
            stage="A",
            mutation_rate=0.1,
            mutation_weighted_activate_prob=0.3,
            diversity_penalty=1.0,
            diversity_hamming_threshold=3,
            diversity_recovery_min_unique_ratio=0.3,
            diversity_recovery_inject_fraction=0.25,
            diversity_recovery_mutation_boost=1.5,
            plateau_early_stop_patience=patience,
            plateau_early_stop_min_generation=20,
            early_stop_min_generation=20,
            seed_fraction=0.3,
            return_floor_pct=-50.0,
            min_trade_support=10,
            use_robust_return_obj=True,
            soft_feasibility=True,
            pool_require_positive_splits=True,
        )

    def test_stage_params_overrides_global_patience(self, monkeypatch):
        monkeypatch.setattr(_cfg, "PHASE2_PLATEAU_EARLY_STOP_PATIENCE", 8)
        stage_params = self._make_stage_params(patience=4)
        assert _resolve_plateau_patience(stage_params) == 4


class TestPlateauEarlyStopBehavior:
    """Verify decision logic uses correct patience values."""

    @staticmethod
    def _dummy_state():
        return {
            "deployable_count": 5,
            "unique_chromosome_ratio": 0.5,
            "population_unique_ratio": 0.5,
        }

    def test_global_triggers_at_global_patience(self, monkeypatch):
        """Global profile: streak=6 does NOT trigger when global_patience=8."""
        monkeypatch.setattr(_cfg, "PHASE2_PLATEAU_EARLY_STOP_PATIENCE", 8)
        monkeypatch.setattr(_cfg, "PHASE2_PLATEAU_EARLY_STOP_ENABLED", True)
        monkeypatch.setattr(_cfg, "PHASE2_PLATEAU_EARLY_STOP_MIN_GENERATION", 1)
        monkeypatch.setattr(_cfg, "PHASE2_PLATEAU_BLOCK_WHEN_DEPLOYABLE_ZERO", False)

        # Global patience=8: streak=6 should NOT trigger
        assert not _should_plateau_early_stop_phase2(
            10, 6, stage_params=None,
            **self._dummy_state(),
        )
        # Global patience=8: streak=8 should trigger
        assert _should_plateau_early_stop_phase2(
            10, 8, stage_params=None,
            **self._dummy_state(),
        )



class TestValMetricInheritance:
    def test_inherit_val_from_global_cache_by_genotype(self):
        from gpu_fuzzy_trader.evolution.evox_runner import (
            _inherit_val_metrics_from_global_cache,
        )

        key = (0, 1, 2)
        global_cache = {
            key: {
                "total_return_pct": 10.0,
                "val_total_return_pct": 5.0,
                "val_sortino_ratio": 1.2,
            },
        }
        metrics = {"total_return_pct": 12.0}
        _inherit_val_metrics_from_global_cache(
            metrics, key, global_cache, run_val=False,
        )
        assert metrics["val_total_return_pct"] == 5.0
        assert metrics["val_sortino_ratio"] == 1.2

    def test_no_inherit_when_val_runs(self):
        from gpu_fuzzy_trader.evolution.evox_runner import (
            _inherit_val_metrics_from_global_cache,
        )

        key = (0, 1, 2)
        global_cache = {key: {"val_total_return_pct": 5.0}}
        metrics = {"total_return_pct": 12.0}
        _inherit_val_metrics_from_global_cache(
            metrics, key, global_cache, run_val=True,
        )
        assert "val_total_return_pct" not in metrics


class TestParetoTrainValGapStats:
    def test_gap_stats_on_pareto_front(self):
        from gpu_fuzzy_trader.evolution.evox_runner import (
            _pareto_train_val_gap_stats,
        )

        metrics_cache = [
            {"total_return_pct": 100.0, "val_total_return_pct": 40.0},
            {"total_return_pct": 50.0},
        ]
        stats = _pareto_train_val_gap_stats([0, 1], metrics_cache)
        assert stats["max_train_val_gap_pct"] == 60.0
        assert abs(stats["max_train_val_gap_ratio"] - 2.5) < 1e-6


class TestPhase2EvolutionStateRestartCount:
    def test_restart_count_field_defaults_zero(self):
        from gpu_fuzzy_trader.evolution.evox_runner import Phase2EvolutionState

        state = Phase2EvolutionState(
            population=np.zeros((1, 4), dtype=np.int32),
            objectives=np.zeros((1, 4)),
            metrics_cache=[{}],
            pareto_archive=[],
            hall_of_fame={},
            deployable_archive={},
            global_metrics_cache={},
        )
        assert state.restart_count == 0


class TestResetPlateauRecoveryCounters:
    """Verify reset_plateau clears restart counters on resumed state."""

    @staticmethod
    def _make_state_with_nonzero_counters(
        pop_size: int = 6,
        restart_count: int = 2,
        post_restart_gens_remaining: int = 1,
        post_restart_no_improve_streak: int = 3,
        post_restart_best_progress: float = 5.0,
    ) -> "Phase2EvolutionState":
        from gpu_fuzzy_trader.evolution.evox_runner import Phase2EvolutionState

        return Phase2EvolutionState(
            population=np.zeros((pop_size, 1), dtype=np.int32),
            objectives=np.full((pop_size, 4), np.inf),
            metrics_cache=[{} for _ in range(pop_size)],
            pareto_archive=[],
            hall_of_fame={},
            deployable_archive={},
            global_metrics_cache={},
            restart_count=restart_count,
            post_restart_gens_remaining=post_restart_gens_remaining,
            post_restart_no_improve_streak=post_restart_no_improve_streak,
            post_restart_best_progress=post_restart_best_progress,
        )

    class _FakeEngine:
        def simulate_rule_batch(self, chromosomes, **kwargs):
            B = chromosomes.shape[0]
            return [
                {
                    "sortino_ratio": 1.0,
                    "total_return_pct": 1.0,
                    "max_drawdown_pct": 2.0,
                    "win_rate": 50.0,
                    "executed_trades": 25,
                }
                for _ in range(B)
            ]

    def test_reset_plateau_true_clears_restart_counters(self):
        """AC: resumed island epoch with reset_plateau=True clears restart counters."""
        from gpu_fuzzy_trader.evolution.evox_runner import run_phase2_evolution_epoch

        feature_infos = [
            {"name": "feat_0", "mode": "binary", "score": 0.5},
        ]
        state = self._make_state_with_nonzero_counters(
            pop_size=6,
            restart_count=2,
            post_restart_gens_remaining=1,
            post_restart_no_improve_streak=3,
            post_restart_best_progress=5.0,
        )
        rng = np.random.default_rng(42)

        new_state, history = run_phase2_evolution_epoch(
            feature_infos=feature_infos,
            engine=self._FakeEngine(),
            pop_size=6,
            n_generations=1,
            rng=rng,
            state=state,
            reset_plateau=True,
        )

        assert new_state.restart_count == 0, (
            f"restart_count should be 0 after reset_plateau, got {new_state.restart_count}"
        )
        assert new_state.post_restart_gens_remaining == 0, (
            f"post_restart_gens_remaining should be 0 after reset_plateau, "
            f"got {new_state.post_restart_gens_remaining}"
        )
        assert new_state.post_restart_no_improve_streak == 0, (
            f"post_restart_no_improve_streak should be 0 after reset_plateau, "
            f"got {new_state.post_restart_no_improve_streak}"
        )
        assert new_state.post_restart_best_progress == -np.inf, (
            f"post_restart_best_progress should be -inf after reset_plateau, "
            f"got {new_state.post_restart_best_progress}"
        )
        assert new_state.plateau_streak == 0, (
            f"plateau_streak should be 0 after reset_plateau, "
            f"got {new_state.plateau_streak}"
        )
        # Note: plateau_best_progress may be updated by the evolution loop
        # during generation 0; only restart counters are guaranteed reset.

    def test_reset_plateau_false_preserves_restart_counters(self):
        """Global/non-island mode: reset_plateau=False preserves counters.

        Uses restart_count=0 so post-restart counters are not mutated
        by the generation loop (which only tracks post-restart progress
        when restart_count > 0).
        """
        from gpu_fuzzy_trader.evolution.evox_runner import run_phase2_evolution_epoch

        feature_infos = [
            {"name": "feat_0", "mode": "binary", "score": 0.5},
        ]
        state = self._make_state_with_nonzero_counters(
            pop_size=6,
            restart_count=0,
            post_restart_gens_remaining=1,
            post_restart_no_improve_streak=3,
            post_restart_best_progress=5.0,
        )
        rng = np.random.default_rng(43)

        new_state, history = run_phase2_evolution_epoch(
            feature_infos=feature_infos,
            engine=self._FakeEngine(),
            pop_size=6,
            n_generations=1,
            rng=rng,
            state=state,
            reset_plateau=False,
        )

        assert new_state.restart_count == 0, (
            f"restart_count should be preserved (0) with reset_plateau=False, "
            f"got {new_state.restart_count}"
        )
        assert new_state.post_restart_gens_remaining == 1, (
            f"post_restart_gens_remaining should be preserved (1) with "
            f"reset_plateau=False, got {new_state.post_restart_gens_remaining}"
        )
        assert new_state.post_restart_no_improve_streak == 3, (
            f"post_restart_no_improve_streak should be preserved (3) with "
            f"reset_plateau=False, got {new_state.post_restart_no_improve_streak}"
        )
        assert new_state.post_restart_best_progress == 5.0, (
            f"post_restart_best_progress should be preserved (5.0) with "
            f"reset_plateau=False, got {new_state.post_restart_best_progress}"
        )


class TestRefreshObjectivesOnResume:
    """Task 2: Verify refresh_objectives_on_resume resets stale objectives
    on resumed evolution states without clearing population/archives."""

    @staticmethod
    def _make_resumable_state(
        pop_size: int = 6,
        n_features: int = 1,
        objectives_fill: float = 5.0,
    ) -> Phase2EvolutionState:
        """Create a state with non-inf objectives and non-empty metrics_cache."""
        return Phase2EvolutionState(
            population=np.zeros((pop_size, n_features), dtype=np.int32),
            objectives=np.full((pop_size, 4), objectives_fill, dtype=np.float64),
            metrics_cache=[
                {"total_return_pct": 10.0, "sortino_ratio": 1.0}
                for _ in range(pop_size)
            ],
            pareto_archive=[],
            hall_of_fame={
                (0,): np.array([0], dtype=np.int32),
            },
            deployable_archive={
                (0,): {
                    "chromosome": np.array([0], dtype=np.int32),
                    "rank_score": 1.0,
                },
            },
            global_metrics_cache={
                (0,): {"total_return_pct": 10.0},
            },
        )

    class _FakeEngine:
        def simulate_rule_batch(self, chromosomes, tp, sl, capital_pct, **kwargs):
            B = chromosomes.shape[0]
            return [
                {
                    "sortino_ratio": 1.0,
                    "total_return_pct": 1.0,
                    "max_drawdown_pct": 2.0,
                    "win_rate": 50.0,
                    "executed_trades": 25,
                }
                for _ in range(B)
            ]

    def test_refresh_true_with_state_resets_objectives(self):
        """refresh_objectives_on_resume=True with state≠None sets objectives
        to inf and clears metrics_cache (verified with n_generations=0)."""
        feature_infos = [{"name": "feat_0", "mode": "binary", "score": 0.5}]
        state = self._make_resumable_state(
            pop_size=6, n_features=1, objectives_fill=5.0,
        )
        rng = np.random.default_rng(42)

        new_state, history = run_phase2_evolution_epoch(
            feature_infos=feature_infos,
            engine=self._FakeEngine(),
            pop_size=6,
            n_generations=0,
            rng=rng,
            state=state,
            refresh_objectives_on_resume=True,
        )

        # Objectives should all be inf (reset by refresh, no gen loop to re-evaluate)
        assert np.all(np.isinf(new_state.objectives)), (
            "All objectives should be inf after refresh with n_generations=0"
        )
        # metrics_cache should be cleared
        assert all(len(m) == 0 for m in new_state.metrics_cache), (
            "All metrics_cache entries should be cleared after refresh"
        )
        # Population preserved
        assert new_state.population.shape == (6, 1)
        assert np.array_equal(new_state.population, np.zeros((6, 1), dtype=np.int32))

    def test_refresh_preserves_population_and_archives(self):
        """Population, hall_of_fame, deployable_archive, and global_metrics_cache
        are preserved after refresh (unless cache is clearly stale)."""
        feature_infos = [{"name": "feat_0", "mode": "binary", "score": 0.5}]
        state = self._make_resumable_state(
            pop_size=6, n_features=1, objectives_fill=5.0,
        )
        rng = np.random.default_rng(42)

        new_state, history = run_phase2_evolution_epoch(
            feature_infos=feature_infos,
            engine=self._FakeEngine(),
            pop_size=6,
            n_generations=0,
            rng=rng,
            state=state,
            refresh_objectives_on_resume=True,
        )

        # Population shape and values preserved
        assert new_state.population.shape == state.population.shape
        assert np.array_equal(new_state.population, state.population)

        # Hall of fame preserved
        assert len(new_state.hall_of_fame) == len(state.hall_of_fame)
        for key, chrom in state.hall_of_fame.items():
            assert key in new_state.hall_of_fame
            assert np.array_equal(new_state.hall_of_fame[key], chrom)

        # Deployable archive preserved
        assert len(new_state.deployable_archive) == len(state.deployable_archive)
        for key, entry in state.deployable_archive.items():
            assert key in new_state.deployable_archive
            assert np.array_equal(
                new_state.deployable_archive[key]["chromosome"],
                entry["chromosome"],
            )

        # Pareo archive preserved
        assert len(new_state.pareto_archive) == len(state.pareto_archive)

    def test_refresh_false_preserves_old_objectives(self):
        """When refresh_objectives_on_resume=False (default), resumed
        objectives keep their old values (verified with n_generations=0)."""
        feature_infos = [{"name": "feat_0", "mode": "binary", "score": 0.5}]
        state = self._make_resumable_state(
            pop_size=6, n_features=1, objectives_fill=5.0,
        )
        rng = np.random.default_rng(42)

        new_state, history = run_phase2_evolution_epoch(
            feature_infos=feature_infos,
            engine=self._FakeEngine(),
            pop_size=6,
            n_generations=0,
            rng=rng,
            state=state,
            refresh_objectives_on_resume=False,
        )

        # Objectives should be unchanged (still 5.0, not inf)
        assert np.allclose(new_state.objectives, 5.0), (
            "Objectives should be preserved when refresh_objectives_on_resume=False"
        )
        # metrics_cache should be unchanged
        assert all(
            m.get("total_return_pct") == 10.0 for m in new_state.metrics_cache
        ), "metrics_cache should be preserved when refresh_objectives_on_resume=False"

    def test_refresh_without_state_noop(self):
        """When state is None (fresh run), refresh_objectives_on_resume has
        no effect — behavior is identical to default."""
        feature_infos = [{"name": "feat_0", "mode": "binary", "score": 0.5}]
        rng = np.random.default_rng(42)

        new_state, history = run_phase2_evolution_epoch(
            feature_infos=feature_infos,
            engine=self._FakeEngine(),
            pop_size=6,
            n_generations=2,
            rng=rng,
            state=None,
            refresh_objectives_on_resume=True,
        )

        # Should have run successfully with 2 generations of history
        assert len(history) == 2
        # Objectives should be finite (evaluated during gen loop)
        assert np.all(np.isfinite(new_state.objectives))
        # metrics_cache should be populated
        assert all(len(m) > 0 for m in new_state.metrics_cache)

        # Compare with default (no refresh flag) — should produce same result
        new_state_default, history_default = run_phase2_evolution_epoch(
            feature_infos=feature_infos,
            engine=self._FakeEngine(),
            pop_size=6,
            n_generations=2,
            rng=rng,
            state=None,
        )
        assert len(history_default) == 2
        assert np.all(np.isfinite(new_state_default.objectives))
