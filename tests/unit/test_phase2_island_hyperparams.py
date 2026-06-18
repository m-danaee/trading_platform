"""Unit tests for island hyperparameter scaling."""

from __future__ import annotations

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
    assert hp.min_profitable_symbols <= 2


def test_orphan_floors(monkeypatch):
    monkeypatch.setattr(cfg, "PHASE2_ORPHAN_MIN_TRADE_SUPPORT", 8)
    hp = cfg.resolve_island_hyperparams(
        "orphan", n_rows=70_000, reference_rows=700_000, n_symbols=1,
    )
    assert hp.min_trade_support == 8
    assert hp.min_profitable_symbols == 1


def test_scale_trade_floor_absolute_min():
    assert cfg.scale_trade_floor_by_universe(
        45, 1000, 10000, absolute_min=8) == 8
