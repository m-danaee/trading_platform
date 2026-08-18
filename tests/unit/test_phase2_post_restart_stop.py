"""Unit tests for post-restart no-improvement early stop (Phase 2 runtime)."""

from __future__ import annotations

from gpu_fuzzy_trader import config as cfg
from gpu_fuzzy_trader.evolution.evox_runner import (
    _should_post_restart_early_stop_phase2,
)


def test_config_defaults():
    assert cfg.PHASE2_PLATEAU_POST_RESTART_STOP_ENABLED is False
    assert cfg.PHASE2_PLATEAU_POST_RESTART_STOP_PATIENCE == 5


def test_global_uses_global_knobs(monkeypatch):
    monkeypatch.setattr(cfg, "PHASE2_PLATEAU_POST_RESTART_STOP_ENABLED", True)
    monkeypatch.setattr(cfg, "PHASE2_PLATEAU_POST_RESTART_STOP_PATIENCE", 2)
    assert not _should_post_restart_early_stop_phase2(1)
    assert _should_post_restart_early_stop_phase2(2)


def test_global_disabled_no_stop(monkeypatch):
    monkeypatch.setattr(cfg, "PHASE2_PLATEAU_POST_RESTART_STOP_ENABLED", False)
    assert not _should_post_restart_early_stop_phase2(10)

