"""Unit tests for regime-aware Phase 2 support penalties."""

from __future__ import annotations

import numpy as np
import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.phases.phase2_support import (
    compute_support_penalty_and_specialist,
    passes_pool_admission_gate,
    passes_pool_trade_floor,
    trade_support_penalty,
    val_regime_confirmation,
)


class TestPoolAdmissionGate:
    def test_rejects_negative_train_return(self) -> None:
        train = {
            "total_return_pct": -1.0,
            "profit_factor": 1.2,
            "executed_trades": _cfg.MIN_TRADE_POOL_FLOOR,
        }
        val = {"total_return_pct": 2.0,
               "profit_factor": 1.1, "executed_trades": 50}
        assert passes_pool_admission_gate(train, val) is False

    def test_accepts_positive_train_and_val(self) -> None:
        train = {
            "total_return_pct": 3.0,
            "profit_factor": 1.2,
            "executed_trades": _cfg.MIN_TRADE_POOL_FLOOR,
        }
        val = {"total_return_pct": 2.0,
               "profit_factor": 1.1, "executed_trades": 50}
        assert passes_pool_admission_gate(train, val) is True


class TestTradeSupportPenaltyStatic:
    def test_at_threshold_zero_penalty(self) -> None:
        pen, spec, dom = trade_support_penalty(_cfg.MIN_TRADE_SUPPORT)
        assert pen == 0.0
        assert spec is False
        assert dom == -1

    def test_below_floor_hard_reject(self) -> None:
        pen, _, _ = trade_support_penalty(0)
        assert pen == 2.0 * _cfg.SUPPORT_PENALTY_MAX

    def test_graduated_between_floor_and_support(self) -> None:
        executed = (_cfg.MIN_TRADE_POOL_FLOOR + _cfg.MIN_TRADE_SUPPORT) // 2
        pen, _, _ = trade_support_penalty(executed)
        assert 0.0 < pen <= _cfg.SUPPORT_PENALTY_MAX


class TestRegimeSpecialist:
    def _regime_metrics(
        self,
        executed: int,
        dominant_trades: int,
        dominant_wins: int,
        dominant_pnl: float,
        n_regimes: int = 3,
    ) -> dict:
        counts = [0] * n_regimes
        wins = [0] * n_regimes
        pnl = [0.0] * n_regimes
        counts[1] = dominant_trades
        wins[1] = dominant_wins
        pnl[1] = dominant_pnl
        other = executed - dominant_trades
        if other > 0:
            counts[0] = other
        return {
            "executed_trades": executed,
            "regime_trade_counts": counts,
            "regime_win_counts": wins,
            "regime_net_pnl": pnl,
        }

    def test_specialist_waives_penalty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_cfg, "PHASE2_REGIME_SUPPORT_ENABLED", True)
        fracs = np.array([0.1, 0.8, 0.1], dtype=np.float64)
        metrics = self._regime_metrics(80, 76, 35, 100.0)
        pen, spec, dom = trade_support_penalty(
            80,
            regime_trade_counts=np.array(metrics["regime_trade_counts"]),
            regime_win_counts=np.array(metrics["regime_win_counts"]),
            regime_net_pnl=np.array(metrics["regime_net_pnl"]),
            regime_row_fractions_arr=fracs,
        )
        assert spec is True
        assert dom == 1
        assert pen == 0.0

    def test_scattered_trades_not_specialist(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_cfg, "PHASE2_REGIME_SUPPORT_ENABLED", True)
        fracs = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3], dtype=np.float64)
        metrics = {
            "executed_trades": 90,
            "regime_trade_counts": [30, 30, 30],
            "regime_win_counts": [20, 20, 20],
            "regime_net_pnl": [1.0, 1.0, 1.0],
        }
        pen, spec, _ = trade_support_penalty(
            90,
            regime_trade_counts=np.array(metrics["regime_trade_counts"]),
            regime_win_counts=np.array(metrics["regime_win_counts"]),
            regime_net_pnl=np.array(metrics["regime_net_pnl"]),
            regime_row_fractions_arr=fracs,
        )
        assert spec is False
        assert pen > 0.0

    def test_pool_floor_waived_for_specialist(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_cfg, "PHASE2_REGIME_SUPPORT_ENABLED", True)
        fracs = np.array([0.05, 0.9, 0.05], dtype=np.float64)
        metrics = self._regime_metrics(80, 76, 35, 50.0)
        assert passes_pool_trade_floor(
            80, metrics, regime_row_fractions_arr=fracs)


class TestValRegimeConfirmation:
    def test_missing_val_regime_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            _cfg, "PHASE2_REGIME_REQUIRE_VAL_CONFIRMATION", True)
        val_metrics = {
            "executed_trades": 20,
            "regime_trade_counts": [20, 0, 0],
            "regime_win_counts": [10, 0, 0],
            "regime_net_pnl": [1.0, 0.0, 0.0],
        }
        assert val_regime_confirmation(
            1, val_metrics, val_regime_row_counts=np.array([50, 0, 0]),
        )

    def test_val_confirmation_fails_low_concentration(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            _cfg, "PHASE2_REGIME_REQUIRE_VAL_CONFIRMATION", True)
        val_metrics = {
            "executed_trades": 50,
            "regime_trade_counts": [10, 18, 22],
            "regime_win_counts": [5, 9, 11],
            "regime_net_pnl": [1.0, -1.0, -1.0],
        }
        assert not val_regime_confirmation(
            1,
            val_metrics,
            val_regime_row_counts=np.array([100, 100, 100]),
        )
