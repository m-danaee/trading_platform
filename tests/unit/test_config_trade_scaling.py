"""Tests for purged-WF trade-floor scaling helpers."""

from __future__ import annotations

import pytest

from gpu_fuzzy_trader import config as cfg


@pytest.fixture
def purged_mode(monkeypatch):
    monkeypatch.setattr(cfg, "SPLIT_MODE", "purged_walk_forward")
    monkeypatch.setattr(cfg, "PURGED_WF_SCALE_TRADE_FLOORS", True)
    monkeypatch.setattr(cfg, "PURGED_WF_MIN_TRADE_FLOOR_ABSOLUTE", 5)
    cfg.set_purged_wf_reference_rows(700_000)


@pytest.fixture
def holdout_mode(monkeypatch):
    monkeypatch.setattr(cfg, "SPLIT_MODE", "holdout_70_30")
    cfg.set_purged_wf_reference_rows(700_000)


class TestScaleTradeFloor:
    def test_holdout_mode_returns_base(self, holdout_mode):
        assert cfg.scale_trade_floor(45, 40_000, 700_000) == 45

    def test_purged_scales_proportionally(self, purged_mode):
        scaled = cfg.scale_trade_floor(45, 175_000, 700_000)
        assert scaled == int(round(45 * 0.25))

    def test_absolute_minimum_clamp(self, purged_mode):
        scaled = cfg.scale_trade_floor(17, 1_000, 700_000)
        assert scaled >= cfg.PURGED_WF_MIN_TRADE_FLOOR_ABSOLUTE

    def test_monotonic_with_rows(self, purged_mode):
        small = cfg.scale_trade_floor(45, 50_000)
        large = cfg.scale_trade_floor(45, 150_000)
        assert large >= small

    def test_effective_wrappers_none_rows(self, holdout_mode):
        assert cfg.effective_min_trade_support(None) == cfg.MIN_TRADE_SUPPORT
        assert cfg.effective_pool_min_val_trades(None) == max(
            cfg.MIN_TRADE_POOL_FLOOR // 4, 10
        )
