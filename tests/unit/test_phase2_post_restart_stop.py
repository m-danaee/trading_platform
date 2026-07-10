"""Unit tests for post-restart no-improvement early stop (Phase 2 runtime)."""

from __future__ import annotations

from gpu_fuzzy_trader import config as cfg
from gpu_fuzzy_trader.evolution.evox_runner import (
    _should_post_restart_early_stop_phase2,
)


def test_config_defaults():
    assert cfg.PHASE2_PLATEAU_POST_RESTART_STOP_ENABLED is True
    assert cfg.PHASE2_PLATEAU_POST_RESTART_STOP_PATIENCE == 5
    assert cfg.PHASE2_ISLAND_PLATEAU_POST_RESTART_STOP_ENABLED is True
    assert cfg.PHASE2_ISLAND_PLATEAU_POST_RESTART_STOP_PATIENCE == 8
    assert cfg.PHASE2_ISLAND_PLATEAU_EARLY_STOP_PATIENCE == 10


def test_island_streak_below_patience_no_stop():
    assert not _should_post_restart_early_stop_phase2(
        2, island_profile="cluster_0",
    )


def test_island_streak_at_patience_stops():
    assert _should_post_restart_early_stop_phase2(
        8, island_profile="cluster_0",
    )


def test_island_disabled_no_stop(monkeypatch):
    monkeypatch.setattr(
        cfg, "PHASE2_ISLAND_PLATEAU_POST_RESTART_STOP_ENABLED", False,
    )
    assert not _should_post_restart_early_stop_phase2(
        10, island_profile="cluster_0",
    )


def test_global_uses_global_knobs(monkeypatch):
    monkeypatch.setattr(cfg, "PHASE2_PLATEAU_POST_RESTART_STOP_ENABLED", True)
    monkeypatch.setattr(cfg, "PHASE2_PLATEAU_POST_RESTART_STOP_PATIENCE", 2)
    assert not _should_post_restart_early_stop_phase2(1, island_profile="global")
    assert _should_post_restart_early_stop_phase2(2, island_profile="global")


def test_global_disabled_no_stop(monkeypatch):
    monkeypatch.setattr(cfg, "PHASE2_PLATEAU_POST_RESTART_STOP_ENABLED", False)
    assert not _should_post_restart_early_stop_phase2(
        10, island_profile="global",
    )


def test_orphan_uses_island_knobs():
    assert not _should_post_restart_early_stop_phase2(
        7, island_profile="orphan",
    )
    assert _should_post_restart_early_stop_phase2(
        8, island_profile="orphan",
    )
