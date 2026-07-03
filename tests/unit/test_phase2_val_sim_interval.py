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

    def simulate_rule_batch(self, chromosomes, tp=None, sl=None, capital_pct=None, **kwargs):
        B = int(chromosomes.shape[0])
        self.calls.append(B)
        return [
            {
                "sortino_ratio": 1.0, "total_return_pct": 1.0,
                "max_drawdown_pct": 2.0, "win_rate": 50.0, "executed_trades": 50,
            }
            for _ in range(B)
        ]


def test_val_skipped_on_non_interval_gens(monkeypatch):
    """With PHASE2_VAL_SIM_INTERVAL=2, val sim runs only on even gens (0,2,4)
    + last gen; train sim runs every gen."""
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

    # Train should run every gen (>=5 calls, one per gen minimum).
    assert len(train_engine.calls) >= 5, (
        f"Train should run every gen; got {len(train_engine.calls)} calls"
    )
    # Val should run LESS often than train (skipped on odd gens).
    assert len(val_engine.calls) < len(train_engine.calls), (
        f"Val should be skipped on non-interval gens; "
        f"val={len(val_engine.calls)} train={len(train_engine.calls)}"
    )
    # Val must run at least on gen 0 (interval) and last gen (forced).
    assert len(val_engine.calls) >= 2, (
        f"Val should run on gen 0 and last gen at minimum; got {len(val_engine.calls)}"
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
    # last gen (gen 2), but NOT on gen 1.  Train runs every gen (3 times)
    # plus initial-pop evaluation calls.  Val should therefore be *fewer*
    # than train calls.
    assert len(val_engine.calls) < len(train_engine.calls), (
        f"JOINT=True should NOT force val every gen when interval=5; "
        f"val={len(val_engine.calls)} train={len(train_engine.calls)}"
    )
    # Val must still run at least on gen 0 (interval) and last gen (forced).
    assert len(val_engine.calls) >= 2, (
        f"Val should run on gen 0 and last gen at minimum; got {len(val_engine.calls)}"
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

    assert len(val_engine.calls) > 0, "Val should run when interval=1"
    # With interval=1, val count should be close to train count (every gen).
    assert len(val_engine.calls) >= len(train_engine.calls) - 1, (
        f"interval=1 should run val every gen; val={len(val_engine.calls)} "
        f"train={len(train_engine.calls)}"
    )
