"""Unit tests for island plateau early-stop safety net (task-3).

Verifies that dead islands (deployable=0, plateaued) can early-stop
instead of churning for the full generation budget, while global mode
and healthy islands remain unaffected.
"""

from __future__ import annotations

from gpu_fuzzy_trader import config as cfg
from gpu_fuzzy_trader.evolution.evox_runner import (
    _should_plateau_early_stop_phase2,
)


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    """Island plateau early-stop is off for short full-budget one-symbol runs."""
    assert cfg.PHASE2_ISLAND_PLATEAU_EARLY_STOP_ENABLED is False
    assert cfg.PHASE2_ISLAND_PLATEAU_BLOCK_WHEN_DEPLOYABLE_ZERO is True
    assert cfg.PHASE2_ISLAND_PLATEAU_EARLY_STOP_PATIENCE == 10


# ---------------------------------------------------------------------------
# AC-T3.1: Dead island stops at plateau
# ---------------------------------------------------------------------------

def test_dead_island_stops_when_plateaued(monkeypatch):
    """Dead island (deployable=0, plateau_streak >= patience) → stops."""
    monkeypatch.setattr(cfg, "PHASE2_ISLAND_PLATEAU_EARLY_STOP_ENABLED", True)
    monkeypatch.setattr(cfg, "PHASE2_ISLAND_PLATEAU_BLOCK_WHEN_DEPLOYABLE_ZERO", False)
    monkeypatch.setattr(cfg, "PHASE2_ISLAND_PLATEAU_EARLY_STOP_PATIENCE", 8)
    # gen=9 so min_gen (3) is satisfied; streak=8 >= patience=8
    assert _should_plateau_early_stop_phase2(
        9, 8, deployable_count=0, island_profile="cluster_0",
    )


def test_dead_island_below_patience_does_not_stop(monkeypatch):
    """Dead island with streak < patience should not stop."""
    monkeypatch.setattr(cfg, "PHASE2_ISLAND_PLATEAU_EARLY_STOP_ENABLED", True)
    monkeypatch.setattr(cfg, "PHASE2_ISLAND_PLATEAU_BLOCK_WHEN_DEPLOYABLE_ZERO", False)
    monkeypatch.setattr(cfg, "PHASE2_ISLAND_PLATEAU_EARLY_STOP_PATIENCE", 8)
    # streak=7 < patience=8
    assert not _should_plateau_early_stop_phase2(
        9, 7, deployable_count=0, island_profile="cluster_0",
    )


# ---------------------------------------------------------------------------
# AC-T3.2: Healthy island does NOT stop early
# ---------------------------------------------------------------------------

def test_healthy_island_does_not_stop_early(monkeypatch):
    """Healthy island (deployable>0, improving = low streak) does not stop."""
    monkeypatch.setattr(cfg, "PHASE2_ISLAND_PLATEAU_EARLY_STOP_ENABLED", True)
    monkeypatch.setattr(cfg, "PHASE2_ISLAND_PLATEAU_BLOCK_WHEN_DEPLOYABLE_ZERO", False)
    monkeypatch.setattr(cfg, "PHASE2_ISLAND_PLATEAU_EARLY_STOP_PATIENCE", 8)
    # still improving (streak=1 << patience=8)
    assert not _should_plateau_early_stop_phase2(
        9, 1, deployable_count=5, island_profile="cluster_0",
    )


def test_healthy_island_can_stop_if_plateaued_long_enough(monkeypatch):
    """Healthy island can stop if it has plateaued long enough."""
    monkeypatch.setattr(cfg, "PHASE2_ISLAND_PLATEAU_EARLY_STOP_ENABLED", True)
    monkeypatch.setattr(cfg, "PHASE2_ISLAND_PLATEAU_BLOCK_WHEN_DEPLOYABLE_ZERO", False)
    monkeypatch.setattr(cfg, "PHASE2_ISLAND_PLATEAU_EARLY_STOP_PATIENCE", 8)
    # healthy deployable but plateaued >= patience
    assert _should_plateau_early_stop_phase2(
        9, 8, deployable_count=5, island_profile="cluster_0",
    )


# ---------------------------------------------------------------------------
# AC-T3.3: Global mode unchanged — deployable=0 still blocks
# ---------------------------------------------------------------------------

def test_global_mode_still_blocks_when_deployable_zero(monkeypatch):
    """Global mode: deployable=0 blocks plateau stop (unchanged behaviour)."""
    monkeypatch.setattr(cfg, "PHASE2_PLATEAU_EARLY_STOP_ENABLED", True)
    monkeypatch.setattr(cfg, "PHASE2_PLATEAU_BLOCK_WHEN_DEPLOYABLE_ZERO", True)
    monkeypatch.setattr(cfg, "PHASE2_PLATEAU_EARLY_STOP_PATIENCE", 5)
    assert not _should_plateau_early_stop_phase2(
        9, 5, deployable_count=0, island_profile="global",
    )


def test_global_mode_works_normally_with_deployable(monkeypatch):
    """Global mode: deployable>0 and plateaued → stops normally."""
    monkeypatch.setattr(cfg, "PHASE2_PLATEAU_EARLY_STOP_ENABLED", True)
    monkeypatch.setattr(cfg, "PHASE2_PLATEAU_BLOCK_WHEN_DEPLOYABLE_ZERO", True)
    monkeypatch.setattr(cfg, "PHASE2_PLATEAU_EARLY_STOP_PATIENCE", 5)
    assert _should_plateau_early_stop_phase2(
        9, 5, deployable_count=10, island_profile="global",
    )


# ---------------------------------------------------------------------------
# AC-T3.4: Island patience knob is respected (8 vs global's 5)
# ---------------------------------------------------------------------------

def test_island_patience_respected_at_8(monkeypatch):
    """Island patience=8: streak=7 → False, streak=8 → True."""
    monkeypatch.setattr(cfg, "PHASE2_ISLAND_PLATEAU_EARLY_STOP_ENABLED", True)
    monkeypatch.setattr(cfg, "PHASE2_ISLAND_PLATEAU_BLOCK_WHEN_DEPLOYABLE_ZERO", False)
    monkeypatch.setattr(cfg, "PHASE2_ISLAND_PLATEAU_EARLY_STOP_PATIENCE", 8)
    monkeypatch.setattr(cfg, "PHASE2_PLATEAU_EARLY_STOP_PATIENCE", 5)

    # island uses patience=8, so streak=7 is not enough
    assert not _should_plateau_early_stop_phase2(
        9, 7, deployable_count=2, island_profile="cluster_1",
    )
    # streak=8 meets island patience=8
    assert _should_plateau_early_stop_phase2(
        9, 8, deployable_count=2, island_profile="cluster_1",
    )


def test_global_patience_still_5(monkeypatch):
    """Global patience remains 5."""
    monkeypatch.setattr(cfg, "PHASE2_PLATEAU_EARLY_STOP_ENABLED", True)
    monkeypatch.setattr(cfg, "PHASE2_PLATEAU_BLOCK_WHEN_DEPLOYABLE_ZERO", True)
    monkeypatch.setattr(cfg, "PHASE2_PLATEAU_EARLY_STOP_PATIENCE", 5)

    # global: streak=5 with deployable>0 should stop
    assert _should_plateau_early_stop_phase2(
        9, 5, deployable_count=3, island_profile="global",
    )
    # global: streak=4 should not stop
    assert not _should_plateau_early_stop_phase2(
        9, 4, deployable_count=3, island_profile="global",
    )


# ---------------------------------------------------------------------------
# Island profile disabled explicitly
# ---------------------------------------------------------------------------

def test_island_early_stop_disabled_explicitly(monkeypatch):
    """When island early-stop is explicitly disabled, dead islands still run."""
    monkeypatch.setattr(cfg, "PHASE2_ISLAND_PLATEAU_EARLY_STOP_ENABLED", False)
    monkeypatch.setattr(cfg, "PHASE2_PLATEAU_EARLY_STOP_ENABLED", True)
    # even with plateaued dead island, island early-stop is off
    assert not _should_plateau_early_stop_phase2(
        9, 8, deployable_count=0, island_profile="cluster_0",
    )


def test_orphan_profile_uses_island_knobs(monkeypatch):
    """Orphan profile follows island-scoped settings."""
    monkeypatch.setattr(cfg, "PHASE2_ISLAND_PLATEAU_EARLY_STOP_ENABLED", True)
    monkeypatch.setattr(cfg, "PHASE2_ISLAND_PLATEAU_BLOCK_WHEN_DEPLOYABLE_ZERO", False)
    monkeypatch.setattr(cfg, "PHASE2_ISLAND_PLATEAU_EARLY_STOP_PATIENCE", 8)

    # orphan is treated as non-global → uses island settings
    assert _should_plateau_early_stop_phase2(
        9, 8, deployable_count=0, island_profile="orphan",
    )
