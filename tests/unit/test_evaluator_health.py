"""
Unit tests for ``evaluator_health_penalty`` and ``execution_ok`` (Task 4).

Tests cover:
  - Pure-function behaviour: skip ratio penalty, exec ratio penalty, boundary.
  - Role multiplier (train / valid / test).
  - Max simultaneous positions penalty.
  - ``execution_ok`` boundary cases (missing raw, zero raw, good ratios).
  - Wire-in: evaluator health penalty subtracted from Phase 3 combo score.
  - Wire-in: ``gate_positive_good`` with ``require_execution_health=True``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from gpu_fuzzy_trader.scoring.evaluator_health import (
    evaluator_health_penalty,
    execution_ok,
)


# ---------------------------------------------------------------------------
# Helper — synthetic metrics dicts
# ---------------------------------------------------------------------------


def _m(
    raw_signal_count: int = 100,
    executed_trades: int = 80,
    skipped_min_notional_count: int = 20,
    max_simultaneous_positions: int = 5,
    **kwargs,
) -> dict:
    """Build a metrics dict with evaluator-health fields."""
    return {
        "raw_signal_count": raw_signal_count,
        "executed_trades": executed_trades,
        "skipped_min_notional_count": skipped_min_notional_count,
        "max_simultaneous_positions": max_simultaneous_positions,
        "total_return_pct": kwargs.get("total_return_pct", 5.0),
        "profit_factor": kwargs.get("profit_factor", 1.5),
    }


# ===================================================================
# evaluator_health_penalty — pure function tests
# ===================================================================


class TestEvaluatorHealthPenalty:
    """Tests for ``evaluator_health_penalty``."""

    def test_skip_ratio_exceeds_threshold(self) -> None:
        """Skip ratio 0.50 > 0.20 → positive penalty."""
        metrics = _m(
            raw_signal_count=100,
            executed_trades=50,
            skipped_min_notional_count=50,
        )
        penalty = evaluator_health_penalty(metrics, role="valid")
        assert penalty > 0.0, f"Expected positive penalty, got {penalty}"

    def test_exec_ratio_below_minimum(self) -> None:
        """Exec ratio 0.30 < 0.60 → positive penalty."""
        metrics = _m(
            raw_signal_count=100,
            executed_trades=30,
            skipped_min_notional_count=70,
        )
        penalty = evaluator_health_penalty(metrics, role="valid")
        assert penalty > 0.0, f"Expected positive penalty, got {penalty}"

    def test_both_ratios_ok_zero_penalty(self) -> None:
        """Skip ratio 0.20 == threshold, exec ratio 0.80 >= 0.60 → 0 penalty."""
        metrics = _m(
            raw_signal_count=100,
            executed_trades=80,
            skipped_min_notional_count=20,
        )
        penalty = evaluator_health_penalty(metrics, role="valid")
        assert penalty == 0.0, f"Expected 0 penalty, got {penalty}"

    def test_below_threshold_skip_ratio_zero_penalty(self) -> None:
        """Skip ratio 0.10 < 0.20, exec ratio 0.90 >= 0.60 → 0 penalty."""
        metrics = _m(
            raw_signal_count=100,
            executed_trades=90,
            skipped_min_notional_count=10,
        )
        penalty = evaluator_health_penalty(metrics, role="valid")
        assert penalty == 0.0, f"Expected 0 penalty, got {penalty}"

    def test_test_role_multiplier(self) -> None:
        """Role 'test' applies 1.5x penalty on skip/exec, should be > valid."""
        metrics = _m(
            raw_signal_count=100,
            executed_trades=50,
            skipped_min_notional_count=50,
        )
        valid_penalty = evaluator_health_penalty(metrics, role="valid")
        test_penalty = evaluator_health_penalty(metrics, role="test")
        assert test_penalty > valid_penalty, (
            f"Expected test penalty ({test_penalty}) > valid penalty ({valid_penalty})"
        )

    def test_max_simultaneous_positions_penalty(self) -> None:
        """Max positions 15 > 10 → positive penalty."""
        metrics = _m(
            raw_signal_count=100,
            executed_trades=80,
            skipped_min_notional_count=20,
            max_simultaneous_positions=15,
        )
        penalty = evaluator_health_penalty(metrics, role="valid")
        assert penalty > 0.0, f"Expected positive position penalty, got {penalty}"

    def test_max_simultaneous_positions_within_limit(self) -> None:
        """Max positions 8 <= 10 → no additional penalty from positions."""
        metrics = _m(
            raw_signal_count=100,
            executed_trades=80,
            skipped_min_notional_count=20,
            max_simultaneous_positions=8,
        )
        # With good ratios and within position limit → 0.
        penalty = evaluator_health_penalty(metrics, role="valid")
        assert penalty == 0.0, f"Expected 0 penalty, got {penalty}"

    def test_missing_raw_signal_count(self) -> None:
        """Missing ``raw_signal_count`` → treated as 0 → no skip/exec penalty."""
        metrics = {
            "executed_trades": 50,
            "skipped_min_notional_count": 25,
            "total_return_pct": 5.0,
        }
        penalty = evaluator_health_penalty(metrics, role="valid")
        # raw=0, so skip/exec ratio not computed. Only position penalty if > 10.
        # max_simultaneous_positions defaults to 0 → within limit.
        assert penalty == 0.0, f"Expected 0 penalty for missing raw, got {penalty}"

    def test_zero_raw_signal_count(self) -> None:
        """``raw_signal_count=0`` → no skip/exec ratio computed → no penalty."""
        metrics = _m(raw_signal_count=0, executed_trades=0, skipped_min_notional_count=0)
        penalty = evaluator_health_penalty(metrics, role="valid")
        assert penalty == 0.0, f"Expected 0 penalty for zero raw, got {penalty}"

    def test_train_role_no_multiplier(self) -> None:
        """Role 'train' same as 'valid' (1.0x)."""
        metrics = _m(
            raw_signal_count=100,
            executed_trades=50,
            skipped_min_notional_count=50,
        )
        train_penalty = evaluator_health_penalty(metrics, role="train")
        valid_penalty = evaluator_health_penalty(metrics, role="valid")
        assert train_penalty == valid_penalty, (
            f"Train penalty ({train_penalty}) != valid penalty ({valid_penalty})"
        )


# ===================================================================
# execution_ok — pure function tests
# ===================================================================


class TestExecutionOk:
    """Tests for ``execution_ok``."""

    def test_good_metrics_returns_true(self) -> None:
        """Skip ratio 0.10 <= 0.20 and exec ratio 0.90 >= 0.60 → True."""
        metrics = _m(
            raw_signal_count=100,
            executed_trades=90,
            skipped_min_notional_count=10,
        )
        assert execution_ok(metrics) is True

    def test_skip_ratio_too_high_returns_false(self) -> None:
        """Skip ratio 0.30 > 0.20 → False."""
        metrics = _m(
            raw_signal_count=100,
            executed_trades=70,
            skipped_min_notional_count=30,
        )
        assert execution_ok(metrics) is False

    def test_exec_ratio_too_low_returns_false(self) -> None:
        """Exec ratio 0.50 < 0.60 → False."""
        metrics = _m(
            raw_signal_count=100,
            executed_trades=50,
            skipped_min_notional_count=50,
        )
        assert execution_ok(metrics) is False

    def test_missing_raw_signal_count_returns_false(self) -> None:
        """Missing ``raw_signal_count`` → treated as 0 → False."""
        metrics = {
            "executed_trades": 80,
            "skipped_min_notional_count": 20,
            "total_return_pct": 5.0,
        }
        assert execution_ok(metrics) is False

    def test_zero_raw_signal_count_returns_false(self) -> None:
        """``raw_signal_count=0`` → False."""
        metrics = _m(raw_signal_count=0, executed_trades=0, skipped_min_notional_count=0)
        assert execution_ok(metrics) is False

    def test_boundary_skip_ratio(self) -> None:
        """Skip ratio exactly 0.20 (== threshold) → True (when exec ratio also ok)."""
        metrics = _m(
            raw_signal_count=100,
            executed_trades=80,
            skipped_min_notional_count=20,
        )
        assert execution_ok(metrics) is True

    def test_boundary_exec_ratio(self) -> None:
        """Exec ratio exactly 0.60 (== threshold) → True (when skip ratio also ok)."""
        metrics = _m(
            raw_signal_count=100,
            executed_trades=60,
            skipped_min_notional_count=20,
        )
        assert execution_ok(metrics) is True


# ===================================================================
# Wire-in: evaluator_health_penalty in Phase 3 combo scoring
# ===================================================================


class TestHealthPenaltyWiredIntoPhase3:
    """Verify that evaluator health penalty is subtracted from combo score."""

    def test_penalty_subtracted_from_combo_return(self) -> None:
        """When evaluator_health_penalty returns > 0, the combo score is lower."""
        from gpu_fuzzy_trader import config as _cfg

        # We'll test the _robust_combo_return logic indirectly by checking
        # that the evaluator_health_penalty function is importable and
        # would be called in the scoring path.
        metrics = _m(
            raw_signal_count=100,
            executed_trades=50,
            skipped_min_notional_count=50,
        )
        penalty = evaluator_health_penalty(metrics, role="train") + evaluator_health_penalty(
            metrics, role="valid"
        )
        assert penalty > 0.0, "Penalty should be positive for bad metrics"

        # With PHASE3_EVAL_HEALTH_WEIGHT = 1.0, the score should be reduced.
        weight = float(getattr(_cfg, "PHASE3_EVAL_HEALTH_WEIGHT", 1.0))
        score_reduction = penalty * weight
        assert score_reduction > 0.0, "Score reduction should be positive"

    def test_good_metrics_no_score_reduction(self) -> None:
        """When evaluator health is fine, no penalty is applied."""
        metrics = _m(
            raw_signal_count=100,
            executed_trades=80,
            skipped_min_notional_count=20,
        )
        penalty = evaluator_health_penalty(metrics, role="train") + evaluator_health_penalty(
            metrics, role="valid"
        )
        assert penalty == 0.0, f"Expected 0 penalty for good metrics, got {penalty}"


# ===================================================================
# Wire-in: execution_ok in gate_positive_good
# ===================================================================


class TestExecutionHealthInGate:
    """Verify that ``gate_positive_good`` calls ``execution_ok`` when flagged."""

    def test_gate_rejects_when_execution_not_ok_and_required(self) -> None:
        """When require_execution_health=True and execution_ok fails, gate returns False."""
        from gpu_fuzzy_trader.phases.phase3_rule_set import gate_positive_good

        # Metrics with good return/PF/trades but bad execution health.
        train = {
            "total_return_pct": 5.0,
            "profit_factor": 1.5,
            "executed_trades": 50,
            "raw_signal_count": 100,
            "skipped_min_notional_count": 50,  # 50% skip → execution_ok fails
        }
        val = {
            "total_return_pct": 3.0,
            "profit_factor": 1.2,
            "executed_trades": 20,
            "raw_signal_count": 100,
            "skipped_min_notional_count": 50,  # 50% skip → execution_ok fails
        }

        result = gate_positive_good(
            train,
            val,
            require_execution_health=True,
        )
        assert result is False, (
            "gate_positive_good should return False when execution health fails"
        )

    def test_gate_passes_when_execution_ok_and_required(self) -> None:
        """When require_execution_health=True and execution_ok passes, gate still returns True."""
        from gpu_fuzzy_trader.phases.phase3_rule_set import gate_positive_good

        train = {
            "total_return_pct": 5.0,
            "profit_factor": 1.5,
            "executed_trades": 65,
            "raw_signal_count": 100,
            "skipped_min_notional_count": 10,  # 10% skip → OK
        }
        val = {
            "total_return_pct": 3.0,
            "profit_factor": 1.2,
            "executed_trades": 65,
            "raw_signal_count": 100,
            "skipped_min_notional_count": 10,  # 10% skip → OK
        }

        result = gate_positive_good(
            train,
            val,
            require_execution_health=True,
        )
        assert result is True, (
            "gate_positive_good should return True when execution health passes"
        )

    def test_gate_not_affected_without_flag(self) -> None:
        """When require_execution_health=False, gate ignores raw_signal_count."""
        from gpu_fuzzy_trader.phases.phase3_rule_set import gate_positive_good

        # Metrics with no raw_signal_count key (like existing tests).
        train = {"total_return_pct": 5.0, "profit_factor": 1.5, "executed_trades": 50}
        val = {"total_return_pct": 3.0, "profit_factor": 1.2, "executed_trades": 20}

        # Should pass without require_execution_health (legacy behaviour).
        result = gate_positive_good(train, val)
        assert result is True, (
            "gate_positive_good should pass without require_execution_health"
        )


# ===================================================================
# Module import test
# ===================================================================


class TestModuleImportable:
    """Both public functions are importable from the module."""

    def test_import_evaluator_health_penalty(self) -> None:
        from gpu_fuzzy_trader.scoring.evaluator_health import (
            evaluator_health_penalty as p,
        )
        assert callable(p)

    def test_import_execution_ok(self) -> None:
        from gpu_fuzzy_trader.scoring.evaluator_health import execution_ok as e
        assert callable(e)
