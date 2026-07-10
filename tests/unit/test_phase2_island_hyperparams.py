"""Unit tests for island hyperparameter scaling."""

from __future__ import annotations

import pytest

from gpu_fuzzy_trader import config as cfg


def test_cluster_k4_scaling(monkeypatch):
    monkeypatch.setattr(cfg, "MIN_TRADE_SUPPORT", 45)
    monkeypatch.setattr(cfg, "MIN_TRADE_POOL_FLOOR", 17)
    monkeypatch.setattr(cfg, "PHASE2_SORTINO_MIN_TRADE_THRESHOLD", 50)
    hp = cfg.resolve_island_hyperparams(
        "cluster", n_rows=175_000, reference_rows=700_000, n_symbols=3,
    )
    assert hp.min_trade_support == 11
    assert hp.min_trade_pool_floor >= 8
    assert hp.min_profitable_symbols == 2  # 3-sym → max(2, (3+1)//2=2) = 2
    assert hp.monthly_admission_min_months == cfg.PHASE2_ISLAND_MONTHLY_MIN_MONTHS


def test_orphan_floors(monkeypatch):
    monkeypatch.setattr(cfg, "PHASE2_ORPHAN_MIN_TRADE_SUPPORT", 8)
    hp = cfg.resolve_island_hyperparams(
        "orphan", n_rows=70_000, reference_rows=700_000, n_symbols=1,
    )
    assert hp.min_trade_support == 8
    assert hp.min_profitable_symbols == 1
    assert hp.monthly_admission_min_months == max(
        2, int(cfg.PHASE2_ISLAND_MONTHLY_MIN_MONTHS) - 1,
    )


def test_scale_trade_floor_absolute_min():
    assert cfg.scale_trade_floor_by_universe(
        45, 1000, 10000, absolute_min=8) == 8


# ── min_profitable_symbols scaling tests (Fix E2) ──────────────────────

@pytest.mark.parametrize("n_symbols,expected_min_profitable", [
    (1, 1),   # orphan-like: max(1, round(0.5)) = 1
    (2, 1),   # max(1, round(1.0)) = 1
    (3, 2),   # cluster floor: max(2, 2) = 2
    (4, 2),   # max(2, 2) = 2
    (5, 3),   # max(2, 3) = 3
    (6, 3),   # max(2, 3) = 3
    (7, 4),   # max(2, 4) = 4
])
def test_min_profitable_symbols_scales_with_cluster_size(
    monkeypatch, n_symbols, expected_min_profitable,
):
    """Verify cluster islands scale min_profitable = max(2, ceil-half)."""
    # Ensure PHASE2_MIN_PROFITABLE_SYMBOLS is high enough not to cap
    monkeypatch.setattr(cfg, "PHASE2_MIN_PROFITABLE_SYMBOLS", 99)
    hp = cfg.resolve_island_hyperparams(
        "cluster",
        n_rows=100_000,
        reference_rows=500_000,
        n_symbols=n_symbols,
    )
    assert hp.min_profitable_symbols == expected_min_profitable, (
        f"n_symbols={n_symbols}: expected {expected_min_profitable}, "
        f"got {hp.min_profitable_symbols}"
    )


def test_min_profitable_symbols_capped_by_config(monkeypatch):
    """The config constant PHASE2_MIN_PROFITABLE_SYMBOLS is the upper bound."""
    monkeypatch.setattr(cfg, "PHASE2_MIN_PROFITABLE_SYMBOLS", 2)
    hp = cfg.resolve_island_hyperparams(
        "cluster",
        n_rows=100_000,
        reference_rows=500_000,
        n_symbols=10,
    )
    # round(10 * 0.5) = 5, capped by PHASE2_MIN_PROFITABLE_SYMBOLS = 2
    assert hp.min_profitable_symbols == 2


def test_orphan_min_profitable_unchanged(monkeypatch):
    """Orphan profile always has min_profitable_symbols=1."""
    monkeypatch.setattr(cfg, "PHASE2_ORPHAN_MIN_TRADE_SUPPORT", 8)
    hp = cfg.resolve_island_hyperparams(
        "orphan", n_rows=70_000, reference_rows=700_000, n_symbols=1,
    )
    assert hp.min_profitable_symbols == 1
