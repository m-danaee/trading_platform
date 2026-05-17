"""Unit tests for Phase 2 EvoX / RVEA runner."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.evolution.evox_runner import (
    _EVOX_AVAILABLE,
    resolve_phase2_runner,
    run_phase2_evolution,
)


class TestResolvePhase2Runner:
    def test_default_rvea(self):
        assert resolve_phase2_runner("RVEA", 200) == "rvea"

    def test_nsga2_explicit(self):
        assert resolve_phase2_runner("NSGA2", 200) == "nsga2"

    def test_nsga3_large_pop(self):
        assert resolve_phase2_runner("NSGA3", 1500) == "nsga3"

    def test_nsga3_small_pop_falls_back_to_rvea(self):
        assert resolve_phase2_runner("NSGA3", 200) == "rvea"


@pytest.mark.skipif(not _EVOX_AVAILABLE, reason="EvoX not installed")
class TestRunPhase2EvolutionSmoke:
    def test_rvea_one_generation(self):
        feature_infos = [
            {"name": "feat_0", "mode": "binary", "score": 0.5},
            {"name": "feat_1", "mode": "binary", "score": 0.5},
        ]

        class FakeEngine:
            def simulate_rule_batch(self, chromosomes, tp, sl, capital_pct):
                B = chromosomes.shape[0]
                return [
                    {
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
            algorithm="RVEA",
        )
        assert isinstance(pool, list)
        assert len(history) == 2
        assert "algorithm" in history[0]


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
                algorithm="RVEA",
            )
        assert len(history) == 1
        assert "algorithm" not in history[0] or history[0].get(
            "algorithm") != "RVEA"
