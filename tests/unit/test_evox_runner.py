"""Unit tests for Phase 2 NSGA-III runner."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from gpu_fuzzy_trader.evolution.evox_runner import (
    _EVOX_AVAILABLE,
    _update_hall_of_fame,
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
