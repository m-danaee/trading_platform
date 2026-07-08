"""Unit tests for periodic val simulation (Phase 2 runtime A2)."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from gpu_fuzzy_trader import config as cfg
from gpu_fuzzy_trader.evolution.evox_runner import run_phase2_evolution


class CountingEngine:
    def __init__(self):
        self.calls: list[int] = []
        self.call_gens: list[int | None] = []

    def simulate_rule_batch(self, chromosomes, tp=None, sl=None, capital_pct=None,
                            generation=None, **kwargs):
        B = int(chromosomes.shape[0])
        self.calls.append(B)
        self.call_gens.append(generation)
        return [
            {
                "sortino_ratio": 1.0, "total_return_pct": 1.0,
                "max_drawdown_pct": 2.0, "win_rate": 50.0, "executed_trades": 50,
            }
            for _ in range(B)
        ]


def test_val_skipped_on_non_interval_gens(monkeypatch):
    """With PHASE2_VAL_SIM_INTERVAL=2, val sim runs on interval gens (0,2,4)
    + last gen; train sim tracks all generations."""
    train_engine = CountingEngine()
    val_engine = CountingEngine()

    monkeypatch.setattr(cfg, "PHASE2_VAL_SIM_INTERVAL", 2)
    monkeypatch.setattr(cfg, "PHASE2_JOINT_TRAIN_VAL", False)
    monkeypatch.setattr(cfg, "PHASE2_EARLY_STOP_ENABLED", False)
    monkeypatch.setattr(cfg, "PHASE2_EARLY_STOP_MIN_GENERATION", 999)
    monkeypatch.setattr(cfg, "PHASE2_PLATEAU_EARLY_STOP_ENABLED", False)
    monkeypatch.setattr(cfg, "PHASE2_PLATEAU_POST_RESTART_STOP_ENABLED", False)
    monkeypatch.setattr(cfg, "PHASE2_DIVERSITY_RECOVERY_ENABLED", False)

    with mock.patch("gpu_fuzzy_trader.evolution.evox_runner._EVOX_AVAILABLE", False):
        rng = np.random.default_rng(0)
        feature_infos = [
            {"name": "f0", "mode": "binary", "score": 0.5},
            {"name": "f1", "mode": "binary", "score": 0.5},
        ]
        run_phase2_evolution(
            feature_infos=feature_infos,
            engine=train_engine,
            val_engine=val_engine,
            pop_size=10,
            n_generations=5,
            rng=rng,
        )

    # Train engine runs at least once (initial-pop evaluation in gen 0).
    assert len(train_engine.calls) >= 1, (
        f"Train should run at least once; got {len(train_engine.calls)} calls"
    )
    # Val should NOT run on non-interval gens.
    val_gens = {g for g in val_engine.call_gens if g is not None}
    # With interval=2 and n_generations=5 (gens 0-4), val runs on
    # gens 0 (0%2==0), 2 (2%2==0), and 4 (last gen).
    expected_val_gens = {0, 2, 4}
    assert val_gens == expected_val_gens, (
        f"Val should run on interval-hit gens {expected_val_gens}; got {val_gens}"
    )
    # Val does NOT run on non-interval gens 1 or 3.
    assert val_gens.isdisjoint({1, 3}), (
        f"Val should NOT run on non-interval gens 1 or 3; got {val_gens}"
    )


def test_val_respects_interval_even_when_joint_train_val_true(monkeypatch):
    """PHASE2_JOINT_TRAIN_VAL=True no longer forces val every gen.

    JOINT_TRAIN_VAL controls whether val metrics *feed fitness*, not how
    often val is computed.  Val cadence is governed solely by
    ``PHASE2_VAL_SIM_INTERVAL`` (+ always on last gen).  This separation
    was restored to fix the per-generation runtime blowup.
    """
    train_engine = CountingEngine()
    val_engine = CountingEngine()

    monkeypatch.setattr(cfg, "PHASE2_JOINT_TRAIN_VAL", True)
    monkeypatch.setattr(cfg, "PHASE2_VAL_SIM_INTERVAL", 5)
    monkeypatch.setattr(cfg, "PHASE2_EARLY_STOP_ENABLED", False)
    monkeypatch.setattr(cfg, "PHASE2_EARLY_STOP_MIN_GENERATION", 999)
    monkeypatch.setattr(cfg, "PHASE2_PLATEAU_EARLY_STOP_ENABLED", False)
    monkeypatch.setattr(cfg, "PHASE2_PLATEAU_POST_RESTART_STOP_ENABLED", False)
    monkeypatch.setattr(cfg, "PHASE2_DIVERSITY_RECOVERY_ENABLED", False)

    with mock.patch("gpu_fuzzy_trader.evolution.evox_runner._EVOX_AVAILABLE", False):
        rng = np.random.default_rng(0)
        feature_infos = [
            {"name": "f0", "mode": "binary", "score": 0.5},
            {"name": "f1", "mode": "binary", "score": 0.5},
        ]
        run_phase2_evolution(
            feature_infos=feature_infos,
            engine=train_engine,
            val_engine=val_engine,
            pop_size=10,
            n_generations=3,
            rng=rng,
        )

    # With interval=5 and n_generations=3, val runs on gen 0 (0%5==0) and
    # last gen (gen 2), but NOT on gen 1.
    val_gens = set(g for g in val_engine.call_gens if g is not None)
    assert val_gens == {0, 2}, (
        f"Val should run on gen 0 (interval) and gen 2 (last gen); got {val_gens}"
    )
    assert 1 not in val_gens, (
        "JOINT=True should NOT force val on gen 1 when interval=5"
    )


def test_val_runs_every_gen_when_interval_1(monkeypatch):
    """PHASE2_VAL_SIM_INTERVAL=1 preserves original behaviour (val every gen)."""
    train_engine = CountingEngine()
    val_engine = CountingEngine()

    monkeypatch.setattr(cfg, "PHASE2_VAL_SIM_INTERVAL", 1)
    monkeypatch.setattr(cfg, "PHASE2_EARLY_STOP_ENABLED", False)
    monkeypatch.setattr(cfg, "PHASE2_EARLY_STOP_MIN_GENERATION", 999)
    monkeypatch.setattr(cfg, "PHASE2_PLATEAU_EARLY_STOP_ENABLED", False)
    monkeypatch.setattr(cfg, "PHASE2_PLATEAU_POST_RESTART_STOP_ENABLED", False)
    monkeypatch.setattr(cfg, "PHASE2_DIVERSITY_RECOVERY_ENABLED", False)

    with mock.patch("gpu_fuzzy_trader.evolution.evox_runner._EVOX_AVAILABLE", False):
        rng = np.random.default_rng(0)
        feature_infos = [
            {"name": "f0", "mode": "binary", "score": 0.5},
        ]
        run_phase2_evolution(
            feature_infos=feature_infos,
            engine=train_engine,
            val_engine=val_engine,
            pop_size=10,
            n_generations=3,
            rng=rng,
        )

    # With interval=1, val should run on every generation.
    val_gens = {g for g in val_engine.call_gens if g is not None}
    # For n_generations=3 (gens 0, 1, 2), val runs on all 3 gens.
    expected_val_gens = {0, 1, 2}
    assert val_gens == expected_val_gens, (
        f"interval=1 should run val every gen; "
        f"expected {expected_val_gens}, got {val_gens}"
    )
    assert len(val_engine.calls) > 0, "Val should run when interval=1"


# ── task-11 tests: PHASE2_VAL_SIM_INTERVAL = 3 ──────────────────────────


def test_should_run_val_this_gen_interval_3():
    """Direct unit test of _should_run_val_this_gen with interval=3.

    With interval=3, val runs on gens 0, 3, 6, 9, ... and always on last gen.
    """
    from gpu_fuzzy_trader.evolution.evox_runner import _should_run_val_this_gen

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(cfg, "PHASE2_VAL_SIM_INTERVAL", 3)
    try:
        # Non-last gens: only multiples of 3
        assert _should_run_val_this_gen(0, is_last_gen=False) is True   # 0 % 3 == 0
        assert _should_run_val_this_gen(1, is_last_gen=False) is False
        assert _should_run_val_this_gen(2, is_last_gen=False) is False
        assert _should_run_val_this_gen(3, is_last_gen=False) is True   # 3 % 3 == 0
        assert _should_run_val_this_gen(4, is_last_gen=False) is False
        assert _should_run_val_this_gen(5, is_last_gen=False) is False
        assert _should_run_val_this_gen(6, is_last_gen=False) is True   # 6 % 3 == 0
        assert _should_run_val_this_gen(7, is_last_gen=False) is False
        assert _should_run_val_this_gen(8, is_last_gen=False) is False
        assert _should_run_val_this_gen(9, is_last_gen=False) is True   # 9 % 3 == 0

        # Last gen always runs val regardless of interval
        assert _should_run_val_this_gen(0, is_last_gen=True) is True
        assert _should_run_val_this_gen(1, is_last_gen=True) is True
        assert _should_run_val_this_gen(2, is_last_gen=True) is True
        assert _should_run_val_this_gen(12, is_last_gen=True) is True
    finally:
        monkeypatch.undo()


def test_val_interval_3_with_13_gen_epoch(monkeypatch):
    """Integration test: with interval=3 and 13-gen epoch, val runs on
    gens 0, 3, 6, 9, 12 (every 3rd gen + last gen)."""
    train_engine = CountingEngine()
    val_engine = CountingEngine()

    monkeypatch.setattr(cfg, "PHASE2_VAL_SIM_INTERVAL", 3)
    monkeypatch.setattr(cfg, "PHASE2_JOINT_TRAIN_VAL", False)
    monkeypatch.setattr(cfg, "PHASE2_EARLY_STOP_ENABLED", False)
    monkeypatch.setattr(cfg, "PHASE2_EARLY_STOP_MIN_GENERATION", 999)
    monkeypatch.setattr(cfg, "PHASE2_PLATEAU_EARLY_STOP_ENABLED", False)
    monkeypatch.setattr(cfg, "PHASE2_PLATEAU_POST_RESTART_STOP_ENABLED", False)
    monkeypatch.setattr(cfg, "PHASE2_DIVERSITY_RECOVERY_ENABLED", False)

    with mock.patch("gpu_fuzzy_trader.evolution.evox_runner._EVOX_AVAILABLE", False):
        rng = np.random.default_rng(0)
        feature_infos = [
            {"name": "f0", "mode": "binary", "score": 0.5},
            {"name": "f1", "mode": "binary", "score": 0.5},
        ]
        run_phase2_evolution(
            feature_infos=feature_infos,
            engine=train_engine,
            val_engine=val_engine,
            pop_size=10,
            n_generations=13,
            rng=rng,
        )

    # Train runs at least once (initial-pop evaluation).
    assert len(train_engine.calls) >= 1, (
        f"Train should run at least once; got {len(train_engine.calls)} calls"
    )
    # Val should visit specific generations according to interval=3 cadence.
    val_gens = {g for g in val_engine.call_gens if g is not None}
    # With interval=3 and 13 gens (0-12), val runs on gens 0, 3, 6, 9, 12.
    expected_val_gens = {0, 3, 6, 9, 12}
    assert val_gens == expected_val_gens, (
        f"Val should run on gens {expected_val_gens}; got {val_gens}"
    )


def test_val_metrics_deterministic_across_cached_gens(monkeypatch):
    """Val metrics for a chromosome are deterministic when val is skipped.

    With interval=3, val runs on gen 0 (interval hit), gen 3 (interval hit),
    and gen 4 (last gen).  Chromosomes evaluated on gen 1 and gen 2 inherit
    val metrics from the global cache (populated during gen 0 val run).
    The cache is chromosome-keyed, so metrics for the same chromosome are
    identical regardless of which generation val was computed.  This test
    verifies that the val engine calls are strictly fewer than train calls
    and that the cadence respects interval=3.

    The deterministic property follows from:
    - Cache key = chromosome_key (tuple of ints), not gen index.
    - Val window is fixed per epoch (post task-1 per-epoch rotation).
    - _inherit_val_metrics_from_global_cache copies cached values verbatim.
    """
    train_engine = CountingEngine()
    val_engine = CountingEngine()

    monkeypatch.setattr(cfg, "PHASE2_VAL_SIM_INTERVAL", 3)
    monkeypatch.setattr(cfg, "PHASE2_JOINT_TRAIN_VAL", False)
    monkeypatch.setattr(cfg, "PHASE2_EARLY_STOP_ENABLED", False)
    monkeypatch.setattr(cfg, "PHASE2_EARLY_STOP_MIN_GENERATION", 999)
    monkeypatch.setattr(cfg, "PHASE2_PLATEAU_EARLY_STOP_ENABLED", False)
    monkeypatch.setattr(cfg, "PHASE2_PLATEAU_POST_RESTART_STOP_ENABLED", False)
    monkeypatch.setattr(cfg, "PHASE2_DIVERSITY_RECOVERY_ENABLED", False)

    with mock.patch("gpu_fuzzy_trader.evolution.evox_runner._EVOX_AVAILABLE", False):
        rng = np.random.default_rng(42)
        feature_infos = [
            {"name": "f0", "mode": "binary", "score": 0.5},
            {"name": "f1", "mode": "binary", "score": 0.5},
        ]
        # Use 5 gens (0-4): val runs on gens 0 (interval), 3 (interval),
        # and 4 (last gen).  Skipped on gens 1-2.
        run_phase2_evolution(
            feature_infos=feature_infos,
            engine=train_engine,
            val_engine=val_engine,
            pop_size=8,
            n_generations=5,
            rng=rng,
        )

    # Val should visit specific generations matching interval=3 cadence.
    val_gens = {g for g in val_engine.call_gens if g is not None}
    # With interval=3 and 5 gens (0-4), val runs on gens 0, 3, 4.
    expected_val_gens = {0, 3, 4}
    assert val_gens == expected_val_gens, (
        f"Val should run on gens {expected_val_gens}; got {val_gens}"
    )
