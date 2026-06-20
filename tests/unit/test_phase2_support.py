"""Unit tests for Phase 2 support penalties."""

from __future__ import annotations

import numpy as np
import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.phases.phase2_support import (
    _raw_feasibility_violation_score,
    compute_support_penalty_and_specialist,
    deployability_rank_score,
    feasibility_violation_score,
    passes_evolution_deployability_preview,
    passes_pool_admission_gate,
    passes_pool_entry_admission,
    passes_pool_trade_floor,
    robust_return_pct,
    trade_support_penalty,
)
from gpu_fuzzy_trader.phases.phase2_stage import resolve_phase2_stage_params


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

    def test_accepts_positive_train_and_val(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_cfg, "PHASE2_STRICT_POSITIVE_GOOD", False)
        train = {
            "total_return_pct": 3.0,
            "profit_factor": 1.2,
            "executed_trades": _cfg.MIN_TRADE_POOL_FLOOR,
        }
        val = {"total_return_pct": 2.0,
               "profit_factor": 1.1, "executed_trades": 50}
        assert passes_pool_admission_gate(train, val) is True

    def test_requires_validation_even_when_joint_train_val_disabled(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(_cfg, "PHASE2_JOINT_TRAIN_VAL", False)
        train = {
            "total_return_pct": 5.0,
            "profit_factor": 1.5,
            "executed_trades": _cfg.MIN_TRADE_POOL_FLOOR,
        }
        val = {
            "total_return_pct": -1.0,
            "profit_factor": 0.9,
            "executed_trades": 50,
        }
        assert passes_pool_admission_gate(train, val) is False

    def test_rejects_missing_validation_metrics(self) -> None:
        train = {
            "total_return_pct": 5.0,
            "profit_factor": 1.5,
            "executed_trades": _cfg.MIN_TRADE_POOL_FLOOR,
        }
        assert passes_pool_admission_gate(train, None) is False

    def test_rejects_excessive_train_val_gap(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(_cfg, "PHASE2_STRICT_POSITIVE_GOOD", False)
        monkeypatch.setattr(_cfg, "PHASE2_MAX_TRAIN_VAL_GAP_PCT", 20.0)
        train = {
            "total_return_pct": 25.0,
            "profit_factor": 1.2,
            "executed_trades": _cfg.MIN_TRADE_POOL_FLOOR,
        }
        val = {
            "total_return_pct": 3.0,
            "profit_factor": 1.1,
            "executed_trades": 50,
        }
        assert passes_pool_admission_gate(train, val) is False

    def test_accepts_within_train_val_gap(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(_cfg, "PHASE2_STRICT_POSITIVE_GOOD", False)
        monkeypatch.setattr(_cfg, "PHASE2_MAX_TRAIN_VAL_GAP_PCT", 20.0)
        train = {
            "total_return_pct": 10.0,
            "profit_factor": 1.2,
            "executed_trades": _cfg.MIN_TRADE_POOL_FLOOR,
        }
        val = {
            "total_return_pct": 8.0,
            "profit_factor": 1.1,
            "executed_trades": 50,
        }
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


class TestDeployabilityHelpers:
    def test_robust_return_uses_min_train_val(self) -> None:
        train = {"total_return_pct": 5.0}
        val = {"total_return_pct": 2.0}
        assert robust_return_pct(train, val) == pytest.approx(2.0)

    def test_feasibility_violation_zero_when_metrics_ok(self) -> None:
        train = {
            "total_return_pct": 2.0,
            "profit_factor": 1.2,
            "executed_trades": _cfg.MIN_TRADE_POOL_FLOOR,
        }
        val = {
            "total_return_pct": 1.0,
            "profit_factor": 1.1,
            "executed_trades": 50,
        }
        assert feasibility_violation_score(train, val) == 0.0

    def test_stage_a_soft_feasibility_returns_zero_for_marginal_rule(self) -> None:
        stage_a = resolve_phase2_stage_params("A")
        train = {
            "total_return_pct": 0.5,
            "profit_factor": 0.9,
            "executed_trades": _cfg.MIN_TRADE_POOL_FLOOR,
        }
        val = {
            "total_return_pct": -0.5,
            "profit_factor": 0.8,
            "executed_trades": 50,
        }
        assert _raw_feasibility_violation_score(train, val) > 0.0
        assert feasibility_violation_score(
            train, val, stage_params=stage_a,
        ) == 0.0

    def test_stage_b_still_rejects_marginal_rule(self) -> None:
        stage_b = resolve_phase2_stage_params("B")
        train = {
            "total_return_pct": 0.5,
            "profit_factor": 0.9,
            "executed_trades": _cfg.MIN_TRADE_POOL_FLOOR,
        }
        val = {
            "total_return_pct": -0.5,
            "profit_factor": 0.8,
            "executed_trades": 50,
        }
        assert feasibility_violation_score(
            train, val, stage_params=stage_b,
        ) > 0.0

    def test_deployability_preview_requires_trade_floor(self) -> None:
        train = {
            "total_return_pct": 5.0,
            "profit_factor": 1.5,
            "executed_trades": 1,
        }
        val = {
            "total_return_pct": 2.0,
            "profit_factor": 1.2,
            "executed_trades": 50,
        }
        assert passes_evolution_deployability_preview(train, val) is False

    def test_deployability_rank_prefers_higher_robust_return(self) -> None:
        orig_use_ret = _cfg.PHASE2_USE_TOTAL_RETURN_OBJ
        try:
            _cfg.PHASE2_USE_TOTAL_RETURN_OBJ = True
            low = deployability_rank_score(
                {"total_return_pct": 1.0, "sortino_ratio": 1.0,
                    "max_drawdown_pct": 5.0},
                {"total_return_pct": 0.5, "sortino_ratio": 0.5,
                    "max_drawdown_pct": 4.0},
            )
            high = deployability_rank_score(
                {"total_return_pct": 4.0, "sortino_ratio": 2.0,
                    "max_drawdown_pct": 5.0},
                {"total_return_pct": 3.0, "sortino_ratio": 1.5,
                    "max_drawdown_pct": 4.0},
            )
            assert high > low
        finally:
            _cfg.PHASE2_USE_TOTAL_RETURN_OBJ = orig_use_ret

    def test_deployability_rank_prefers_higher_win_rate_when_wr_mode(self) -> None:
        orig_use_ret = _cfg.PHASE2_USE_TOTAL_RETURN_OBJ
        try:
            _cfg.PHASE2_USE_TOTAL_RETURN_OBJ = False
            low = deployability_rank_score(
                {"win_rate": 40.0, "sortino_ratio": 1.0, "max_drawdown_pct": 5.0},
                {"win_rate": 35.0, "sortino_ratio": 0.5, "max_drawdown_pct": 4.0},
            )
            high = deployability_rank_score(
                {"win_rate": 60.0, "sortino_ratio": 2.0, "max_drawdown_pct": 5.0},
                {"win_rate": 55.0, "sortino_ratio": 1.5, "max_drawdown_pct": 4.0},
            )
            assert high > low
        finally:
            _cfg.PHASE2_USE_TOTAL_RETURN_OBJ = orig_use_ret


class TestPoolAdmissionScaledFloors:
    def test_scaled_min_val_trades_on_small_slice(self, monkeypatch) -> None:
        from gpu_fuzzy_trader.phases.phase2_support import _pool_admission_floors

        monkeypatch.setattr(_cfg, "SPLIT_MODE", "purged_walk_forward")
        monkeypatch.setattr(_cfg, "PURGED_WF_SCALE_TRADE_FLOORS", True)
        _cfg.set_purged_wf_reference_rows(700_000)

        _, _, _, _, full_min = _pool_admission_floors(None)
        _, _, _, _, small_min = _pool_admission_floors(40_000)
        assert small_min < full_min
        assert small_min >= _cfg.PURGED_WF_MIN_TRADE_FLOOR_ABSOLUTE
