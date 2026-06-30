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


def test_island_patience_uses_island_knob_not_stage_params(monkeypatch):
    """Regression: island patience must come from
    PHASE2_ISLAND_PLATEAU_EARLY_STOP_PATIENCE, NOT stage_params.

    Islands run single-stage (stage=None) → resolve_phase2_stage_params(None)
    bakes in PHASE2_PLATEAU_EARLY_STOP_PATIENCE=8 into stage_params. Before the
    fix, stage_params.plateau_early_stop_patience (=8) was read instead of the
    island knob (=6), making the island knob dead code.
    """
    from gpu_fuzzy_trader.phases.phase2_stage import resolve_phase2_stage_params

    monkeypatch.setattr(cfg, "PHASE2_ISLAND_PLATEAU_EARLY_STOP_ENABLED", True)
    monkeypatch.setattr(cfg, "PHASE2_ISLAND_PLATEAU_BLOCK_WHEN_DEPLOYABLE_ZERO", False)
    monkeypatch.setattr(cfg, "PHASE2_ISLAND_PLATEAU_EARLY_STOP_PATIENCE", 6)
    monkeypatch.setattr(cfg, "PHASE2_PLATEAU_EARLY_STOP_PATIENCE", 8)

    # stage_params has patience=8 (the GLOBAL default baked into None profile).
    stage_params = resolve_phase2_stage_params(None)
    assert stage_params.plateau_early_stop_patience == 8, (
        "Test precondition: stage_params patience should be the global 8"
    )

    # Island profile: streak=6 must trigger stop (island patience=6 wins).
    assert _should_plateau_early_stop_phase2(
        9, 6, deployable_count=5, island_profile="cluster_0",
        stage_params=stage_params,
    ), "streak=6 should stop with island patience=6 (not wait for 8)"

    # And streak=5 must NOT trigger (below island patience=6).
    assert not _should_plateau_early_stop_phase2(
        9, 5, deployable_count=5, island_profile="cluster_0",
        stage_params=stage_params,
    ), "streak=5 should not stop with island patience=6"
