"""Unit tests for island early-stop bypass."""

from __future__ import annotations

from gpu_fuzzy_trader import config as cfg
from gpu_fuzzy_trader.evolution.evox_runner import (
    _should_early_stop_phase2,
    _should_plateau_early_stop_phase2,
)


def test_cluster_profile_disables_early_stop(monkeypatch):
    monkeypatch.setattr(cfg, "PHASE2_ISLAND_EARLY_STOP_ENABLED", False)
    monkeypatch.setattr(cfg, "PHASE2_EARLY_STOP_ENABLED", True)
    assert not _should_early_stop_phase2(
        100, -10.0, 0, island_profile="cluster",
    )


def test_cluster_profile_disables_plateau(monkeypatch):
    monkeypatch.setattr(cfg, "PHASE2_ISLAND_PLATEAU_EARLY_STOP_ENABLED", False)
    monkeypatch.setattr(cfg, "PHASE2_PLATEAU_EARLY_STOP_ENABLED", True)
    assert not _should_plateau_early_stop_phase2(
        100, 20, island_profile="cluster",
    )


def test_orphan_profile_disables_early_stop(monkeypatch):
    monkeypatch.setattr(cfg, "PHASE2_ISLAND_MODE", "cluster")
    monkeypatch.setattr(cfg, "PHASE2_ISLAND_EARLY_STOP_ENABLED", False)
    monkeypatch.setattr(cfg, "PHASE2_EARLY_STOP_ENABLED", True)
    assert not _should_early_stop_phase2(
        100, -10.0, 0, island_profile="orphan",
    )
