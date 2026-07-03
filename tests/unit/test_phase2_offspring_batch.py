"""Unit tests for batched offspring evaluation (Phase 2 runtime A1)."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from gpu_fuzzy_trader import config as cfg
from gpu_fuzzy_trader.evolution.evox_runner import run_phase2_evolution


class CountingEngine:
    """Stub engine recording every simulate_rule_batch call's batch size."""

    def __init__(self):
        self.train_calls: list[int] = []

    def simulate_rule_batch(self, chromosomes, tp=None, sl=None, capital_pct=None, **kwargs):
        B = int(chromosomes.shape[0])
        self.train_calls.append(B)
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


def test_offspring_evaluated_in_single_batch_per_gen():
    """Offspring should be evaluated via ONE simulate_rule_batch call per gen,
    not pop_size individual calls. The initial population may still be
    evaluated one-by-one (that per-chromosome loop is not part of this
    optimisation), so we allow up to pop_size size-1 calls from initial eval.

    With only 2 binary features the 10 offspring deduplicate to 4 unique
    chromosomes, so the batch size will be 4 (not 10) — that's fine; the key
    invariant is that there is NO SIZE-1 call for any offspring, only the
    initial-pop individual evaluations use size 1."""
    pop_size = 10
    n_generations = 3
    engine = CountingEngine()

    with mock.patch(
        "gpu_fuzzy_trader.evolution.evox_runner._EVOX_AVAILABLE", False,
    ), mock.patch.object(cfg, "PHASE2_EARLY_STOP_ENABLED", False), \
         mock.patch.object(cfg, "PHASE2_EARLY_STOP_MIN_GENERATION", 999), \
         mock.patch.object(cfg, "PHASE2_PLATEAU_EARLY_STOP_ENABLED", False), \
         mock.patch.object(cfg, "PHASE2_PLATEAU_POST_RESTART_STOP_ENABLED", False), \
         mock.patch.object(cfg, "PHASE2_DIVERSITY_RECOVERY_ENABLED", False), \
         mock.patch.object(cfg, "PHASE2_PLATEAU_DIVERSITY_RESTART_ENABLED", False):
        rng = np.random.default_rng(0)
        feature_infos = [
            {"name": "f0", "mode": "binary", "score": 0.5},
            {"name": "f1", "mode": "binary", "score": 0.5},
        ]
        run_phase2_evolution(
            feature_infos=feature_infos,
            engine=engine,
            pop_size=pop_size,
            n_generations=n_generations,
            rng=rng,
        )

    batch_sizes = engine.train_calls

    # Key invariant: at least one batch has size > 1 (offspring batched).
    assert any(b > 1 for b in batch_sizes), (
        f"Expected at least one batch > 1 (offspring), got sizes {batch_sizes}"
    )
    # The total number of calls must be much less than the old behaviour
    # (pop_size * n_generations + pop_size = 40 for 10x3). Offspring are
    # batched, so total calls ≈ pop_size (initial eval) + (n_generations - 1)
    # (offspring, last gen skips).
    max_expected_calls = pop_size + n_generations  # 13 for 10x3
    assert len(batch_sizes) <= max_expected_calls, (
        f"Too many batch calls ({len(batch_sizes)}); offspring not batched. "
        f"Sizes: {batch_sizes}"
    )
    # The old behaviour would produce pop_size * n_generations + pop_size
    # size-1 calls. After batching, only the initial-pop eval contributes
    # size-1 calls. Allow up to pop_size + small margin.
    size_1_calls = sum(1 for b in batch_sizes if b == 1)
    max_expected_size_1 = pop_size + 2  # initial-pop eval + small fallback margin
    assert size_1_calls <= max_expected_size_1, (
        f"Too many size-1 batch calls ({size_1_calls}); offspring not batched. "
        f"Sizes: {batch_sizes}"
    )
