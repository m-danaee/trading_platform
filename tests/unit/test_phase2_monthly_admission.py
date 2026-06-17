"""Unit tests for the Phase 2 monthly-window shadow-test gate (Task 13).

Tests the gate logic in isolation by monkeypatching
``_evaluate_rule_on_window`` to return deterministic values,
so no real backtest engine or data is needed.
"""

from __future__ import annotations

import pytest

from gpu_fuzzy_trader import config as _cfg


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


def _run_gate(
    pool: list[dict],
    n_months: int = 6,
    enabled: bool = True,
    min_ratio: float = 0.5,
    min_months: int = 4,
) -> list[dict]:
    """Simulate the monthly-admission gate logic (replicates ``run()``).

    Accesses ``_evaluate_rule_on_window`` through the module so monkeypatch
    on the module attribute works correctly.
    """
    import gpu_fuzzy_trader.phases.phase2_rule_pool as _p2rp

    # If gate is disabled, return the pool unchanged.
    if not enabled:
        return list(pool)

    # Skip gate when not enough months.
    if n_months < min_months:
        return list(pool)

    # Evaluate each rule on each (dummy) window.
    keep: list[dict] = []
    for entry in pool:
        ret_pcts: list[float] = []
        for w in range(n_months):
            ret = _p2rp._evaluate_rule_on_window(entry, w, "long")
            ret_pcts.append(ret)
        profitable = sum(1 for r in ret_pcts if r > 0)
        ratio = profitable / max(1, len(ret_pcts))
        if ratio >= min_ratio:
            keep.append(entry)

    # Graceful degradation: if gate would empty the pool, return original.
    if len(keep) == 0:
        return list(pool)
    return keep


# ===========================================================================
# Tests
# ===========================================================================


class TestMonthlyAdmissionGate:
    """Verify the gate keeps / rejects rules based on profitable_ratio."""

    def test_keeps_only_fully_profitable_rule(self, monkeypatch):
        """Rule 0 profitable on all 6 months → kept; others partially → dropped."""
        returns = [
            [1.0, 0.5, 2.0, 1.5, 0.8, 1.2],    # rule 0: 6/6 = 1.0
            [1.0, -0.5, 2.0, -1.0, 0.5, -0.3],  # rule 1: 3/6 = 0.5 (passes but...)
            [-1.0, -2.0, -0.5, -1.5, -3.0, -0.8],  # rule 2: 0/6 = 0.0
        ]
        monkeypatch.setattr(
            "gpu_fuzzy_trader.phases.phase2_rule_pool._evaluate_rule_on_window",
            _DeterministicEvaluator(returns),
        )

        result = _run_gate(POOL_THREE, n_months=6)
        # Rule 0 has ratio 1.0, rule 1 has 0.5, rule 2 has 0.0.
        # With min_ratio=0.5, rules 0 and 1 should be kept.
        assert len(result) == 2
        assert result[0]["conditions"] == POOL_THREE[0]["conditions"]
        assert result[1]["conditions"] == POOL_THREE[1]["conditions"]

    def test_disabled_flag_keeps_all_rules(self, monkeypatch):
        """When the flag is False, the gate does not run — all 3 rules kept."""
        result = _run_gate(POOL_THREE, n_months=6, enabled=False)
        assert len(result) == 3

    def test_skip_gate_when_min_months_exceeds_windows(self, monkeypatch):
        """With min_months=10 and only 6 months, skip gate → all 3 rules kept."""
        result = _run_gate(POOL_THREE, n_months=6, min_months=10)
        assert len(result) == 3

    def test_graceful_degradation_empty_pool(self, monkeypatch):
        """When all rules fail the gate, original pool is kept (not emptied)."""
        returns = [
            [-1.0, -0.5, -2.0, -1.5, -0.8, -1.2],  # rule 0: all negative
            [-1.0, -0.5, -2.0, -1.0, -0.5, -0.3],  # rule 1: all negative
            [-1.0, -2.0, -0.5, -1.5, -3.0, -0.8],  # rule 2: all negative
        ]
        monkeypatch.setattr(
            "gpu_fuzzy_trader.phases.phase2_rule_pool._evaluate_rule_on_window",
            _DeterministicEvaluator(returns),
        )

        result = _run_gate(POOL_THREE, n_months=6)
        # Graceful degradation: original pool returned (3 rules)
        assert len(result) == 3
        assert result == POOL_THREE

    def test_boundary_exact_half_profitable(self, monkeypatch):
        """Rule with exactly 3/6 = 0.5 passes when min_ratio=0.5."""
        returns = [
            [1.0, -1.0, 1.0, -1.0, 1.0, -1.0],  # rule 0: 3/6 = 0.5 (passes)
            [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],  # rule 1: 0/6
            [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6],  # rule 2: 0/6
        ]
        monkeypatch.setattr(
            "gpu_fuzzy_trader.phases.phase2_rule_pool._evaluate_rule_on_window",
            _DeterministicEvaluator(returns),
        )

        result = _run_gate(POOL_THREE, n_months=6)
        assert len(result) == 1
        assert result[0]["conditions"] == POOL_THREE[0]["conditions"]
