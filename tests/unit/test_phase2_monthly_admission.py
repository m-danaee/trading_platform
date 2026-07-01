"""Unit tests for the Phase 2 monthly-window shadow-test gate (Task 13).

Tests ``_apply_monthly_admission_gate`` directly by monkeypatching
``_evaluate_rule_on_window`` to return deterministic values,
so no real backtest engine or data is needed.
"""

from __future__ import annotations

import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.phases.phase2_rule_pool import (
    _apply_monthly_admission_gate,
)


# ---------------------------------------------------------------------------
# Deterministic evaluator monkeypatch helper
# ---------------------------------------------------------------------------


class _DeterministicEvaluator:
    """Maps (rule_index, window_index) → ``total_return_pct``.

    Usage in tests::

        monkeypatch.setattr(
            "gpu_fuzzy_trader.phases.phase2_rule_pool._evaluate_rule_on_window",
            _DeterministicEvaluator(returns_by_rule),
        )

    Each call advances an internal counter; *returns_by_rule[r][w]* is returned
    for the *r*-th rule on the *w*-th window (rules evaluated sequentially,
    all windows per rule before moving to the next rule).
    """

    def __init__(self, returns_by_rule: list[list[float]]):
        self.returns_by_rule = returns_by_rule
        self.call_count = 0

    def __call__(
        self, pool_entry: dict, window_df: object, direction: str,
    ) -> float:
        n_windows = len(self.returns_by_rule[0]) if self.returns_by_rule else 1
        rule_idx = self.call_count // n_windows
        window_idx = self.call_count % n_windows
        self.call_count += 1
        if rule_idx >= len(self.returns_by_rule):
            return 0.0
        return float(self.returns_by_rule[rule_idx][window_idx])


# ---------------------------------------------------------------------------
# Shared fixture: a pool of 3 rules
# ---------------------------------------------------------------------------

POOL_THREE = [
    {
        "conditions": ["[feat_a] IS high", "symbol is 1"],
        "chromosome": [0, 1, 1],
        "objectives": {"total_return_pct": 5.0, "profit_factor": 1.5},
        "executed_trades": 50,
    },
    {
        "conditions": ["[feat_b] IS low", "symbol is 2"],
        "chromosome": [1, 0, 1],
        "objectives": {"total_return_pct": 2.0, "profit_factor": 1.2},
        "executed_trades": 40,
    },
    {
        "conditions": ["[feat_c] IS medium", "symbol is 3"],
        "chromosome": [2, 2, 2],
        "objectives": {"total_return_pct": -3.0, "profit_factor": 0.8},
        "executed_trades": 30,
    },
]


# ===========================================================================
# Tests
# ===========================================================================


class TestMonthlyAdmissionGate:
    """Verify the gate keeps / rejects rules based on profitable_ratio."""

    # ------------------------------------------------------------------
    # Test 1: basic profitability filtering
    # ------------------------------------------------------------------

    def test_keeps_profitable_and_half_profitable(self, monkeypatch):
        """Rule 0 profitable on all 6 months (ratio=1.0) passes at >=0.5;
        rule 1 on half (ratio=0.5) also passes at >=0.5; rule 2 (ratio=0.0)
        dropped."""
        returns = [
            [1.0, 0.5, 2.0, 1.5, 0.8, 1.2],    # rule 0: 6/6 = 1.0
            [1.0, -0.5, 2.0, -1.0, 0.5, -0.3],  # rule 1: 3/6 = 0.5
            [-1.0, -2.0, -0.5, -1.5, -3.0, -0.8],  # rule 2: 0/6 = 0.0
        ]
        monkeypatch.setattr(
            "gpu_fuzzy_trader.phases.phase2_rule_pool._evaluate_rule_on_window",
            _DeterministicEvaluator(returns),
        )

        result = _apply_monthly_admission_gate(POOL_THREE, list(range(6)), "long")
        # Rule 0 (1.0) passes >=0.5; rule 1 (0.5) passes >=0.5; rule 2 (0.0) dropped.
        assert len(result) == 2
        assert result[0]["conditions"] == POOL_THREE[0]["conditions"]
        assert result[1]["conditions"] == POOL_THREE[1]["conditions"]

    # ------------------------------------------------------------------
    # Test 2: all rules just below threshold are rejected,
    #         triggering graceful degradation
    # ------------------------------------------------------------------

    def test_rejects_below_threshold_graceful_degradation(self, monkeypatch):
        """All rules have ratio < 0.5 → gate empties pool → graceful
        degradation keeps original pool."""
        returns = [
            [-1.0, -0.5, -2.0, -1.5, -0.8, -1.2],  # rule 0: 0/6 = 0.0
            [-1.0, -0.5, -2.0, -1.0, -0.5, -0.3],  # rule 1: 0/6 = 0.0
            [-1.0, -2.0, -0.5, -1.5, -3.0, -0.8],  # rule 2: 0/6 = 0.0
        ]
        monkeypatch.setattr(
            "gpu_fuzzy_trader.phases.phase2_rule_pool._evaluate_rule_on_window",
            _DeterministicEvaluator(returns),
        )

        result = _apply_monthly_admission_gate(POOL_THREE, list(range(6)), "long")
        # Graceful degradation: original pool returned (3 rules)
        assert len(result) == 3
        assert result == POOL_THREE

    # ------------------------------------------------------------------
    # Test 3: boundary — ratio exactly at 0.5 threshold
    # ------------------------------------------------------------------

    def test_boundary_exact_half_profitable(self, monkeypatch):
        """Rule with exactly 3/6 = 0.5 passes at >=0.5 threshold."""
        returns = [
            [1.0, -1.0, 1.0, -1.0, 1.0, -1.0],  # rule 0: 3/6 = 0.5 (passes)
            [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],  # rule 1: 0/6
            [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6],  # rule 2: 0/6
        ]
        monkeypatch.setattr(
            "gpu_fuzzy_trader.phases.phase2_rule_pool._evaluate_rule_on_window",
            _DeterministicEvaluator(returns),
        )

        result = _apply_monthly_admission_gate(POOL_THREE, list(range(6)), "long")
        # Rule 0 (0.5) passes at >=0.5; rules 1+2 rejected.
        assert len(result) == 1
        assert result[0]["conditions"] == POOL_THREE[0]["conditions"]

    # ------------------------------------------------------------------
    # Test 4: boundary — ratio just below 0.5 threshold
    # ------------------------------------------------------------------

    def test_boundary_just_below_threshold(self, monkeypatch):
        """Rule with 2/6 ≈ 0.33 is below 0.5 threshold; rule with 3/6 = 0.5 passes."""
        returns = [
            [1.0, -1.0, 1.0, -1.0, -1.0, -1.0],  # rule 0: 2/6 ≈ 0.33 (rejected)
            [1.0, -1.0, 1.0, -1.0, 1.0, -1.0],   # rule 1: 3/6 = 0.5 (passes)
            [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],  # rule 2: 0/6
        ]
        monkeypatch.setattr(
            "gpu_fuzzy_trader.phases.phase2_rule_pool._evaluate_rule_on_window",
            _DeterministicEvaluator(returns),
        )

        result = _apply_monthly_admission_gate(POOL_THREE, list(range(6)), "long")
        # Rule 1 (0.5) passes; rule 0 rejected; rule 2 rejected.
        assert len(result) == 1
        assert result[0]["conditions"] == POOL_THREE[1]["conditions"]

    def test_zero_threshold_strict_profit_excludes_flat_months(
        self, monkeypatch,
    ) -> None:
        """Phase 2 at min=0 uses strict >0; flat months do not count."""
        returns = [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # rule 0: 1/6 strict profit
            [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
            [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
        ]
        monkeypatch.setattr(
            "gpu_fuzzy_trader.phases.phase2_rule_pool._evaluate_rule_on_window",
            _DeterministicEvaluator(returns),
        )
        monkeypatch.setattr(_cfg, "PHASE2_MONTHLY_GOOD_RETURN_MIN_PCT", 0.0)

        result = _apply_monthly_admission_gate(
            POOL_THREE, list(range(6)), "long")
        assert len(result) == 3
        assert result == POOL_THREE

    def test_positive_threshold_requires_min_return(self, monkeypatch) -> None:
        """With PHASE2_MONTHLY_GOOD_RETURN_MIN_PCT=2, months need return >= 2%.
        Rule 0 has 5/6 >= 2% (ratio=0.833) which passes 0.667; rule 1,2 fail."""
        returns = [
            [3.0, 2.0, 2.0, 2.5, 2.0, 0.0],  # rule 0: 5/6 >= 2% (0.833)
            [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
            [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
        ]
        monkeypatch.setattr(
            "gpu_fuzzy_trader.phases.phase2_rule_pool._evaluate_rule_on_window",
            _DeterministicEvaluator(returns),
        )
        monkeypatch.setattr(_cfg, "PHASE2_MONTHLY_GOOD_RETURN_MIN_PCT", 2.0)

        result = _apply_monthly_admission_gate(
            POOL_THREE, list(range(6)), "long")
        assert len(result) == 1
        assert result[0]["conditions"] == POOL_THREE[0]["conditions"]

    def test_multiple_rules_pass_gate(self, monkeypatch):
        """Rules with >= 0.5 ratio are kept; rule below 0.5 threshold dropped."""
        returns = [
            [1.0, 0.5, 2.0, 1.5, 0.8, 1.2],    # rule 0: 6/6 = 1.0 (passes)
            [1.0, -0.5, 2.0, -1.0, 0.5, -1.0],  # rule 1: 3/6 = 0.5 (passes)
            [1.0, -0.5, -2.0, -1.0, -0.5, -0.3],  # rule 2: 1/6 ≈ 0.17 (rejected)
        ]
        monkeypatch.setattr(
            "gpu_fuzzy_trader.phases.phase2_rule_pool._evaluate_rule_on_window",
            _DeterministicEvaluator(returns),
        )

        result = _apply_monthly_admission_gate(POOL_THREE, list(range(6)), "long")
        assert len(result) == 2
        assert result[0]["conditions"] == POOL_THREE[0]["conditions"]
        assert result[1]["conditions"] == POOL_THREE[1]["conditions"]
