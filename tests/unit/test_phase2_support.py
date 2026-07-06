"""Unit tests for Phase 2 support penalties."""

from __future__ import annotations

import numpy as np
import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.phases.phase2_support import (
    _feasibility_gate_failures,
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
               "profit_factor": 1.21, "executed_trades": 50}
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
            "profit_factor": 1.21,
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
        assert 0.0 <= pen <= _cfg.SUPPORT_PENALTY_MAX


class TestDeployabilityHelpers:
    def test_robust_return_uses_min_train_val(self) -> None:
        train = {"total_return_pct": 5.0}
        val = {"total_return_pct": 2.0}
        # Explicit joint=True: returns min(train, val) = 2.0.
        assert robust_return_pct(train, val, joint=True) == pytest.approx(2.0)

    def test_robust_return_joint_false_returns_train_only(self) -> None:
        train = {"total_return_pct": 5.0}
        val = {"total_return_pct": 2.0}
        # With joint=False, val must NOT pull return down — train-only is used.
        assert robust_return_pct(train, val, joint=False) == pytest.approx(5.0)
        # Also confirm it works without val_metrics (e.g. CV mode).
        assert robust_return_pct(train, None, joint=False) == pytest.approx(5.0)

    def test_feasibility_violation_zero_when_metrics_ok(self) -> None:
        train = {
            "total_return_pct": 2.0,
            "profit_factor": 1.2,
            "executed_trades": _cfg.MIN_TRADE_POOL_FLOOR,
        }
        val = {
            "total_return_pct": 1.0,
            "profit_factor": 1.21,
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

    def test_feasibility_violation_catches_train_val_gap(self, monkeypatch) -> None:
        """Regression: train=90%/val=10% (gap=80pp >> 16pp threshold) must produce
        a non-zero violation score and fail deployability preview."""
        from gpu_fuzzy_trader.phases.phase2_support import (
            _raw_feasibility_violation_score,
            passes_evolution_deployability_preview,
        )

        monkeypatch.setattr(_cfg, "PHASE2_POOL_REQUIRE_POSITIVE_SPLITS", True)
        monkeypatch.setattr(_cfg, "MIN_TRADE_POOL_FLOOR", 1)
        monkeypatch.setattr(_cfg, "PHASE2_RETURN_FLOOR_PCT", -100.0)
        monkeypatch.setattr(_cfg, "PHASE2_VAL_RETURN_FLOOR_PCT", -100.0)
        monkeypatch.setattr(_cfg, "PHASE2_PROFIT_FACTOR_FLOOR", 0.0)
        monkeypatch.setattr(_cfg, "PHASE2_MAX_TRAIN_VAL_GAP_PCT", 16.0)

        train = {
            "total_return_pct": 90.0,
            "profit_factor": 1.5,
            "executed_trades": 100,
        }
        val = {
            "total_return_pct": 10.0,
            "profit_factor": 1.2,
            "executed_trades": 50,
        }

        score = _raw_feasibility_violation_score(train, val)
        assert score > 0.0, (
            f"Expected positive violation for 80pp gap, got {score}"
        )
        assert passes_evolution_deployability_preview(train, val) is False, (
            "Should not pass deployability preview with 80pp train-val gap"
        )


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


class TestFeasibilityGateFailures:
    """Tests for _feasibility_gate_failures — per-gate breakdown."""

    @pytest.fixture
    def high_metrics(self) -> dict:
        """A rule that should pass all 9 gates."""
        return {
            "executed_trades": 100,
            "total_return_pct": 5.0,
            "profit_factor": 2.0,
        }

    @pytest.fixture
    def high_val_metrics(self) -> dict:
        return {
            "executed_trades": 50,
            "total_return_pct": 3.0,
            "profit_factor": 1.5,
        }

    @pytest.fixture
    def low_trade_metrics(self) -> dict:
        """A rule with too few train trades."""
        return {
            "executed_trades": 5,
            "total_return_pct": 5.0,
            "profit_factor": 2.0,
        }

    def test_all_pass(
        self, high_metrics: dict, high_val_metrics: dict,
    ) -> None:
        """A rule passing all gates returns all-zero dict."""
        result = _feasibility_gate_failures(high_metrics, high_val_metrics)
        assert all(v == 0 for v in result.values())
        assert len(result) == 9

    def test_val_none(self, high_metrics: dict) -> None:
        """When val_metrics is None, only val_required=1, others=0."""
        result = _feasibility_gate_failures(high_metrics, None)
        assert result["val_required"] == 1
        # Train gates still evaluated
        assert result["train_trade_floor"] == 0
        assert result["train_return_floor"] == 0
        assert result["train_pf_floor"] == 0
        # Val gates never reached (returned early)
        assert result["val_ret_positive"] == 0
        assert result["val_trade_floor"] == 0
        assert result["val_return_floor"] == 0
        assert result["val_pf_floor"] == 0
        assert result["train_val_gap"] == 0
        assert len(result) == 9

    def test_train_trade_floor(
        self, low_trade_metrics: dict, high_val_metrics: dict,
    ) -> None:
        """A rule with too few train trades fails train_trade_floor."""
        result = _feasibility_gate_failures(
            low_trade_metrics, high_val_metrics,
        )
        assert result["train_trade_floor"] == 1
        # Other train gates still pass
        assert result["train_return_floor"] == 0
        assert result["train_pf_floor"] == 0
        # Val gates left at 0 (not reached in this case since train_trade_floor
        # is a soft gate — unlike _passes_pool_admission_impl it does NOT
        # short-circuit the rest, so val gates are still evaluated)
        assert result["val_required"] == 0

    def test_val_ret_positive(
        self, high_metrics: dict, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When PHASE2_REQUIRE_LAST_FOLD_POSITIVE=True and val_ret <= 0,
        val_ret_positive = 1. (val_return_floor also triggers because
        val_ret <= val_ret_min=0.0 — both gates fire simultaneously.)"""
        monkeypatch.setattr(
            _cfg, "PHASE2_REQUIRE_LAST_FOLD_POSITIVE", True,
        )
        val = {"executed_trades": 50, "total_return_pct": -1.0, "profit_factor": 1.5}
        result = _feasibility_gate_failures(high_metrics, val)
        assert result["val_ret_positive"] == 1
        # val_return_floor also fires because val_ret=-1.0 <= val_ret_min=0.0
        assert result["val_return_floor"] == 1
        # Other gates should pass
        assert result["val_trade_floor"] == 0
        assert result["val_pf_floor"] == 0

    def test_val_return_floor(
        self, high_metrics: dict, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When val_ret <= val_ret_min (default 0.0), val_return_floor=1."""
        monkeypatch.setattr(_cfg, "PHASE2_REQUIRE_LAST_FOLD_POSITIVE", False)
        val = {
            "executed_trades": 50,
            "total_return_pct": -0.5,
            "profit_factor": 1.5,
        }
        result = _feasibility_gate_failures(high_metrics, val)
        # val_ret_positive should be 0 (PHASE2_REQUIRE_LAST_FOLD_POSITIVE=False)
        assert result["val_ret_positive"] == 0
        # but val_return_floor should be 1 (val_ret=-0.5 <= 0.0)
        assert result["val_return_floor"] == 1

    def test_all_fail(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Multi-gate failure: low trades, low return, low pf, and val_ret <= 0."""
        monkeypatch.setattr(
            _cfg, "PHASE2_REQUIRE_LAST_FOLD_POSITIVE", True,
        )
        train = {
            "executed_trades": 5,
            "total_return_pct": -2.0,
            "profit_factor": 0.5,
        }
        val = {
            "executed_trades": 2,
            "total_return_pct": -3.0,
            "profit_factor": 0.8,
        }
        result = _feasibility_gate_failures(train, val)
        assert result["train_trade_floor"] == 1
        assert result["train_return_floor"] == 1
        assert result["train_pf_floor"] == 1
        assert result["val_ret_positive"] == 1
        assert result["val_trade_floor"] == 1
        assert result["val_return_floor"] == 1
        assert result["val_pf_floor"] == 1
        # train_val_gap = train_ret - val_ret = -2 - (-3) = 1, max_gap=20, so 0
        assert result["train_val_gap"] == 0
        assert len(result) == 9
