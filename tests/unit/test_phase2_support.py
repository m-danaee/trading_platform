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
    resolve_evolution_floors,
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


class TestResolveEvolutionFloorsIslandTwoStage:
    """Stage A soft floors must survive island_hyperparams (cluster two-stage)."""

    @staticmethod
    def _island_hp(*, support: int = 80, pool_floor: int = 25):
        return _cfg.IslandHyperparams(
            profile="cluster",
            min_trade_support=support,
            min_trade_pool_floor=pool_floor,
            sortino_min_trade_threshold=20,
            val_trade_floor=10,
            min_profitable_symbols=2,
            monthly_admission_min_months=3,
            monthly_admission_min_profitable_ratio=0.4,
            skip_symbol_robustness_penalty=False,
            n_rows=200_000,
            n_symbols=3,
        )

    def test_stage_a_soft_floors_with_island_hyperparams(self) -> None:
        stage_a = resolve_phase2_stage_params("A")
        island = self._island_hp(support=80)
        floors = resolve_evolution_floors(
            stage_a, island_hyperparams=island,
        )
        assert floors.soft_feasibility is True
        assert floors.return_floor_pct == _cfg.PHASE2_STAGE_A_RETURN_FLOOR_PCT
        assert floors.min_trade_support == min(
            _cfg.PHASE2_STAGE_A_MIN_TRADE_SUPPORT, 80,
        )
        assert floors.pool_require_positive_splits is False

    def test_stage_b_uses_island_scaled_support(self) -> None:
        stage_b = resolve_phase2_stage_params("B")
        island = self._island_hp(support=80)
        floors = resolve_evolution_floors(
            stage_b, island_hyperparams=island,
        )
        assert floors.soft_feasibility is False
        assert floors.return_floor_pct == _cfg.PHASE2_RETURN_FLOOR_PCT
        assert floors.min_trade_support == 80
        assert floors.pool_require_positive_splits is True

    def test_island_only_keeps_strict_floors(self) -> None:
        island = self._island_hp(support=80)
        floors = resolve_evolution_floors(None, island_hyperparams=island)
        assert floors.soft_feasibility is False
        assert floors.return_floor_pct == _cfg.PHASE2_RETURN_FLOOR_PCT
        assert floors.min_trade_support == 80


class TestDeployabilityHelpersTail:
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

    def test_pf_floor_uses_admission_not_evolution(self, monkeypatch) -> None:
        """_pool_admission_floors returns the ADMISSION floor (1.15),
        not the EVOLUTION floor (1.0). This is the hard gate regression guard."""
        from gpu_fuzzy_trader.phases.phase2_support import _pool_admission_floors

        _, _, _, pf_floor, _ = _pool_admission_floors(None)
        assert pf_floor == pytest.approx(1.15), (
            f"_pool_admission_floors returned pf_floor={pf_floor}, "
            f"expected 1.15 (ADMISSION floor)"
        )


class TestEvolutionFeasibilityFloors:
    """Tests for _evolution_feasibility_floors — EVOLUTION PF 1.0 vs ADMISSION PF 1.15."""

    def test_evolution_feasibility_floors_uses_evolution_pf(self, monkeypatch) -> None:
        """_evolution_feasibility_floors returns PF=1.0, not ADMISSION 1.15."""
        from gpu_fuzzy_trader.phases.phase2_support import (
            _evolution_feasibility_floors,
            _pool_admission_floors,
        )

        _, _, _, evo_pf, _ = _evolution_feasibility_floors(None)
        assert evo_pf == pytest.approx(1.0), (
            f"_evolution_feasibility_floors returned pf_floor={evo_pf}, "
            f"expected 1.0 (EVOLUTION floor)"
        )

        _, _, _, adm_pf, _ = _pool_admission_floors(None)
        assert adm_pf == pytest.approx(1.15), (
            f"_pool_admission_floors returned pf_floor={adm_pf}, "
            f"expected 1.15 (ADMISSION floor)"
        )

    def test_raw_violation_score_uses_evolution_pf(self, monkeypatch) -> None:
        """_raw_feasibility_violation_score uses EVOLUTION PF (1.0) so
        a rule with PF between 1.0 and 1.15 has zero PF violation,
        while pool admission would reject it."""
        from gpu_fuzzy_trader.phases.phase2_support import (
            _raw_feasibility_violation_score,
        )

        monkeypatch.setattr(_cfg, "PHASE2_POOL_REQUIRE_POSITIVE_SPLITS", True)
        monkeypatch.setattr(_cfg, "MIN_TRADE_POOL_FLOOR", 1)
        monkeypatch.setattr(_cfg, "PHASE2_RETURN_FLOOR_PCT", -100.0)
        monkeypatch.setattr(_cfg, "PHASE2_VAL_RETURN_FLOOR_PCT", -100.0)
        monkeypatch.setattr(_cfg, "PHASE2_MAX_TRAIN_VAL_GAP_PCT", 99.0)

        # Rule with PF=1.10 — above EVOLUTION floor (1.0) but below ADMISSION (1.15)
        train = {
            "total_return_pct": 3.0,
            "profit_factor": 1.10,
            "executed_trades": 100,
        }
        val = {
            "total_return_pct": 2.0,
            "profit_factor": 1.10,
            "executed_trades": 50,
        }
        score = _raw_feasibility_violation_score(train, val)
        # The violation score should be 0 because PF 1.10 >= 1.0 (evolution floor)
        assert score == 0.0, (
            f"Expected score=0 for PF=1.10 >= evolution floor 1.0, got {score}"
        )

        # Pool admission still checks against ADMISSION floor 1.15
        from gpu_fuzzy_trader.phases.phase2_support import (
            passes_pool_admission_gate,
        )
        assert passes_pool_admission_gate(train, val) is False, (
            "PF=1.10 should fail pool admission (requires 1.15)"
        )


class TestTrainOnlyFitnessValGating:
    """Val terms in _raw_feasibility_violation_score respect include_val flag."""

    def test_train_only_fitness_ignores_bad_val(self, monkeypatch) -> None:
        monkeypatch.setattr(_cfg, "PHASE2_POOL_REQUIRE_POSITIVE_SPLITS", True)
        monkeypatch.setattr(_cfg, "MIN_TRADE_POOL_FLOOR", 1)
        monkeypatch.setattr(_cfg, "PHASE2_RETURN_FLOOR_PCT", -100.0)
        monkeypatch.setattr(_cfg, "PHASE2_MAX_TRAIN_VAL_GAP_PCT", 99.0)

        train = {
            "total_return_pct": 3.0,
            "profit_factor": 1.2,
            "executed_trades": 100,
        }
        val = {
            "total_return_pct": -5.0,
            "profit_factor": 0.5,
            "executed_trades": 2,
        }

        train_only = _raw_feasibility_violation_score(
            train, val, include_val=False,
        )
        full = _raw_feasibility_violation_score(
            train, val, include_val=True,
        )

        assert train_only == 0.0
        assert full > 0.0

    def test_deployability_preview_still_checks_val(self, monkeypatch) -> None:
        monkeypatch.setattr(_cfg, "PHASE2_POOL_REQUIRE_POSITIVE_SPLITS", True)
        monkeypatch.setattr(_cfg, "MIN_TRADE_POOL_FLOOR", 1)
        monkeypatch.setattr(_cfg, "PHASE2_RETURN_FLOOR_PCT", -100.0)
        monkeypatch.setattr(_cfg, "PHASE2_MAX_TRAIN_VAL_GAP_PCT", 99.0)

        train = {
            "total_return_pct": 3.0,
            "profit_factor": 1.2,
            "executed_trades": 100,
        }
        val = {
            "total_return_pct": -5.0,
            "profit_factor": 0.5,
            "executed_trades": 50,
        }

        assert passes_evolution_deployability_preview(train, val) is False


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
        assert len(result) == 11

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
        assert result["overfit_ratio"] == 0
        assert len(result) == 11

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
        # overfit_ratio = train_ret / max(val_ret, 0.1) = -2.0 / 0.1 = -20.0, not > 3.0
        assert result["overfit_ratio"] == 0
        assert len(result) == 11

    def test_f4_gate_uses_joint_min_when_joint_train_val_enabled(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When PHASE2_JOINT_TRAIN_VAL=True, the f4 gate must use
        min(train_f4, val_f4) — matching the objective computation exactly.

        Train f4 is high (0.9 > 0.5 floor), but val f4 is low (0.2).
        Without joint-min the gate would reject; with joint-min it passes.
        """
        monkeypatch.setattr(_cfg, "PHASE2_F4_ENABLED", True)
        monkeypatch.setattr(_cfg, "PHASE2_JOINT_TRAIN_VAL", True)
        monkeypatch.setattr(_cfg, "PHASE2_F4_EPSILON", 1e-6)
        monkeypatch.setattr(_cfg, "PHASE2_F4_CONCENTRATION_FLOOR", 0.5)
        monkeypatch.setattr(_cfg, "PHASE2_REQUIRE_LAST_FOLD_POSITIVE", False)

        train = {
            "executed_trades": 100,
            "total_return_pct": 5.0,
            "profit_factor": 2.0,
            "max_single_trade_pnl": 90.0,
            "sum_positive_trade_pnl": 100.0,  # f4 = 0.9
        }
        val = {
            "executed_trades": 50,
            "total_return_pct": 3.0,
            "profit_factor": 1.5,
            "max_single_trade_pnl": 10.0,
            "sum_positive_trade_pnl": 50.0,  # f4 = 0.2
        }
        result = _feasibility_gate_failures(train, val)
        # Train-only f4 would be 0.9 > 0.5 → gate would fail.
        # Joint min(train=0.9, val=0.2) = 0.2 ≤ 0.5 → gate passes.
        assert result["f4_concentration"] == 0, (
            f"Expected f4 gate to pass (joint min=0.2 ≤ 0.5), "
            f"but got f4_concentration={result['f4_concentration']}"
        )

    def test_overfit_ratio_fires_when_high_ratio(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When train_ret / max(val_ret, 0.1) > PHASE2_OVERFIT_RATIO_FLOOR,
        overfit_ratio=1.
        """
        monkeypatch.setattr(_cfg, "PHASE2_OVERFIT_RATIO_FLOOR", 3.0)
        train = {
            "executed_trades": 100,
            "total_return_pct": 15.0,
            "profit_factor": 2.0,
        }
        val = {
            "executed_trades": 50,
            "total_return_pct": 4.0,  # 15/4 = 3.75 > 3.0
            "profit_factor": 1.5,
        }
        result = _feasibility_gate_failures(train, val)
        assert result["overfit_ratio"] == 1, (
            f"Expected overfit_ratio=1 for 15%/4% (3.75×), "
            f"got {result['overfit_ratio']}"
        )

    def test_overfit_ratio_zero_when_floor_disabled(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When PHASE2_OVERFIT_RATIO_FLOOR=0.0, the gate is disabled
        and overfit_ratio stays 0 (pre-task-6 regression guard).
        """
        monkeypatch.setattr(_cfg, "PHASE2_OVERFIT_RATIO_FLOOR", 0.0)
        train = {
            "executed_trades": 100,
            "total_return_pct": 15.0,
            "profit_factor": 2.0,
        }
        val = {
            "executed_trades": 50,
            "total_return_pct": 4.0,
            "profit_factor": 1.5,
        }
        result = _feasibility_gate_failures(train, val)
        assert result["overfit_ratio"] == 0, (
            f"Expected overfit_ratio=0 with floor=0.0, "
            f"got {result['overfit_ratio']}"
        )

    def test_overfit_ratio_zero_when_moderate_ratio(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When ratio is below PHASE2_OVERFIT_RATIO_FLOOR, overfit_ratio=0."""
        monkeypatch.setattr(_cfg, "PHASE2_OVERFIT_RATIO_FLOOR", 3.0)
        train = {
            "executed_trades": 100,
            "total_return_pct": 15.0,
            "profit_factor": 2.0,
        }
        val = {
            "executed_trades": 50,
            "total_return_pct": 10.0,  # 15/10 = 1.5 < 3.0
            "profit_factor": 1.5,
        }
        result = _feasibility_gate_failures(train, val)
        assert result["overfit_ratio"] == 0, (
            f"Expected overfit_ratio=0 for 15%/10% (1.5×), "
            f"got {result['overfit_ratio']}"
        )
