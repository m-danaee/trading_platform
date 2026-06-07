"""Unit tests for Phase 2 NSGA-III runner."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.evolution.evox_runner import (
    _EVOX_AVAILABLE,
    _evaluate_population_indices,
    _should_plateau_early_stop_phase2,
    _update_hall_of_fame,
    _update_max_return_plateau,
    run_phase2_evolution,
)


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
            def simulate_rule_batch(self, chromosomes, tp, sl, capital_pct):
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
            def simulate_rule_batch(self, chromosomes, tp, sl, capital_pct):
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
            def simulate_rule_batch(self, chromosomes, tp, sl, capital_pct):
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


def test_low_trade_drawdown_penalty():
    # Set CV fold mode trade floor to 25
    orig_mode = _cfg.SPLIT_MODE
    orig_floor = _cfg.PHASE2_CV_MIN_TRADE_POOL_FLOOR
    orig_support = _cfg.MIN_TRADE_SUPPORT
    try:
        _cfg.SPLIT_MODE = "purged_rolling_cv"
        _cfg.PHASE2_CV_MIN_TRADE_POOL_FLOOR = 25
        _cfg.MIN_TRADE_SUPPORT = 5
        
        # population of 1 candidate
        # dont_cares = 5, we make 3 active conditions, so cond_penalty is 0.0
        pop = np.full((1, 10), 5, dtype=np.int32)
        pop[0, :3] = 0  # 3 active conditions
        dont_cares = np.ones(10, dtype=np.int32) * 5
        objectives = np.full((1, 3), np.inf)
        metrics_cache = [{}]
        
        class MockEngine:
            def simulate_rule_batch(self, chromosomes, **kwargs):
                # return metrics with 5 executed trades and 0.0 drawdown
                return [{"executed_trades": 5, "total_return_pct": 1.0, "sortino_ratio": 0.5, "max_drawdown_pct": 0.0, "win_rate": 0.5}]
                
        engine = MockEngine()
        _evaluate_population_indices(
            pop, [0], dont_cares, engine, [], objectives, metrics_cache
        )
        
        # Assert that all 3 objectives receive the full dominating penalty
        assert objectives[0, 0] >= 50.0  # Sortino penalized to 0.0 + penalty >= 50.0
        assert objectives[0, 1] >= 150.0 # Drawdown penalized to 100.0 + penalty >= 150.0
        assert objectives[0, 2] >= 50.0  # Win rate penalized to 0.0 + penalty >= 50.0
    finally:
        _cfg.SPLIT_MODE = orig_mode
        _cfg.PHASE2_CV_MIN_TRADE_POOL_FLOOR = orig_floor
        _cfg.MIN_TRADE_SUPPORT = orig_support


def test_phase2_use_total_return_obj():
    has_orig = hasattr(_cfg, "PHASE2_USE_TOTAL_RETURN_OBJ")
    orig_val = getattr(_cfg, "PHASE2_USE_TOTAL_RETURN_OBJ", False)
    orig_floor = _cfg.MIN_TRADE_SUPPORT
    
    try:
        _cfg.PHASE2_USE_TOTAL_RETURN_OBJ = True
        _cfg.MIN_TRADE_SUPPORT = 5
        
        pop = np.full((1, 10), 5, dtype=np.int32)
        pop[0, :3] = 0  # 3 active conditions
        dont_cares = np.ones(10, dtype=np.int32) * 5
        objectives = np.full((1, 3), np.inf)
        metrics_cache = [{}]
        
        class MockEngine:
            def simulate_rule_batch(self, chromosomes, **kwargs):
                return [{
                    "executed_trades": 100,
                    "total_return_pct": 15.0,
                    "sortino_ratio": 0.5,
                    "max_drawdown_pct": 2.0,
                    "win_rate": 50.0,
                    "profit_factor": 1.0,
                }]
                
        engine = MockEngine()
        _evaluate_population_indices(
            pop, [0], dont_cares, engine, [], objectives, metrics_cache
        )
        
        # With total return obj enabled: f3 = -total_return_pct = -15.0 (plus penalties)
        # Objectives are: [-sortino + pen, dd + pen, -total_return + pen]
        # Sortino obj should be -0.5 + pen, DD should be 2.0 + pen, F3 should be -15.0 + pen
        assert np.isclose(objectives[0, 2], -15.0)
        
        # Disable it -> should use win_rate = 50.0
        _cfg.PHASE2_USE_TOTAL_RETURN_OBJ = False
        objectives[0] = np.inf
        _evaluate_population_indices(
            pop, [0], dont_cares, engine, [], objectives, metrics_cache
        )
        assert np.isclose(objectives[0, 2], -50.0)
        
    finally:
        _cfg.MIN_TRADE_SUPPORT = orig_floor
        if has_orig:
            _cfg.PHASE2_USE_TOTAL_RETURN_OBJ = orig_val
        else:
            if hasattr(_cfg, "PHASE2_USE_TOTAL_RETURN_OBJ"):
                delattr(_cfg, "PHASE2_USE_TOTAL_RETURN_OBJ")


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
            def simulate_rule_batch(self, chromosomes, tp, sl, capital_pct):
                call_sizes.append(chromosomes.shape[0])
                return [
                    _default_eval_metrics()
                    for _ in range(chromosomes.shape[0])
                ]

        pop = np.array([[0, 1], [0, 1], [2, 3]], dtype=np.int32)
        objectives = np.full((3, 3), np.inf)
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
            def simulate_rule_batch(self, chromosomes, tp, sl, capital_pct):
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

        objectives_first = np.full((1, 3), np.inf)
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

        objectives_second = np.full((1, 3), np.inf)
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
            def simulate_rule_batch(self, chromosomes, tp, sl, capital_pct):
                nonlocal call_count
                call_count += 1
                return [_default_eval_metrics()]

        pop = np.array([[1, 0]], dtype=np.int32)
        dont_cares = np.array([2, 2], dtype=np.int32)
        engine = CountingEngine()
        global_cache: dict[tuple[int, ...], dict] = {}
        objectives = np.full((1, 3), np.inf)
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
        _evaluate_population_indices(
            pop,
            [0],
            dont_cares,
            engine,
            [pop[0].copy()],
            objectives,
            metrics_cache,
            global_metrics_cache=global_cache,
        )
        obj_with_archive = objectives[0]

        assert call_count == 1
        assert obj_with_archive[0] > obj_no_archive[0]


class TestPlateauEarlyStop:
    def test_update_resets_streak_on_improvement(self):
        best, streak = _update_max_return_plateau(5.0, -np.inf, 3)
        assert best == 5.0
        assert streak == 0

        best, streak = _update_max_return_plateau(5.0, best, streak)
        assert best == 5.0
        assert streak == 1

        best, streak = _update_max_return_plateau(5.02, best, streak)
        assert best == 5.02
        assert streak == 0

    def test_should_stop_after_patience(self, monkeypatch):
        monkeypatch.setattr(_cfg, "PHASE2_PLATEAU_EARLY_STOP_ENABLED", True)
        monkeypatch.setattr(
            _cfg, "PHASE2_PLATEAU_EARLY_STOP_MIN_GENERATION", 5)
        monkeypatch.setattr(_cfg, "PHASE2_PLATEAU_EARLY_STOP_PATIENCE", 3)
        monkeypatch.setattr(
            _cfg, "PHASE2_PLATEAU_EARLY_STOP_DISABLED_IN_CV", False)

        assert not _should_plateau_early_stop_phase2(3, 3)
        assert not _should_plateau_early_stop_phase2(10, 2)
        assert _should_plateau_early_stop_phase2(10, 3)


class TestBatchSingleObjectiveAlignment:
    def test_return_floor_penalty_matches_single_path(self, monkeypatch):
        from gpu_fuzzy_trader.phases.phase2_rule_pool import (
            _evaluate_chromosome,
            compute_phase2_objectives_from_metrics,
        )

        monkeypatch.setattr(_cfg, "PHASE2_RETURN_FLOOR_PCT", 2.0)
        monkeypatch.setattr(_cfg, "PHASE2_JOINT_TRAIN_VAL", True)

        pop = np.full((1, 10), 5, dtype=np.int32)
        pop[0, :3] = 0
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
