"""Tests for marginal rule contribution diagnostics."""

from __future__ import annotations

import numpy as np

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.rb_governor import (
    CandidateRecord,
    _marginal_prune_ruleset,
)
from gpu_fuzzy_trader.portfolio.marginal import (
    effective_rule_count,
    marginal_contribution,
)


def test_marginal_contribution_uses_full_minus_leave_one_out() -> None:
    full = {
        "Return": 10.0,
        "Sortino": 1.5,
        "MDD": 4.0,
        "PF": 1.8,
        "WorstMonth": -1.0,
    }
    without_rule = {
        "Return": 8.0,
        "Sortino": 1.0,
        "MDD": 5.0,
        "PF": 1.4,
        "WorstMonth": -2.0,
    }

    result = marginal_contribution(full, without_rule)

    assert result == {
        "ΔReturn": 2.0,
        "ΔSortino": 0.5,
        "ΔMDD": -1.0,
        "ΔPF": 0.4,
        "ΔWorstMonth": 1.0,
        "is_beneficial": True,
    }


def test_marginal_contribution_rejects_sortino_and_drawdown_degradation() -> None:
    result = marginal_contribution(
        {"Return": 9.0, "Sortino": 0.5, "MDD": 6.0, "PF": 1.1, "WorstMonth": -4.0},
        {"Return": 10.0, "Sortino": 1.0, "MDD": 4.0, "PF": 1.3, "WorstMonth": -2.0},
    )

    assert result["ΔReturn"] == -1.0
    assert result["ΔSortino"] == -0.5
    assert result["ΔMDD"] == 2.0
    assert result["is_beneficial"] is False


def test_marginal_contribution_rejects_severe_return_loss() -> None:
    result = marginal_contribution(
        {"Return": -5.0, "Sortino": 1.2, "MDD": 4.0},
        {"Return": 10.0, "Sortino": 0.8, "MDD": 6.0},
    )

    assert result["ΔSortino"] == 0.4
    assert result["ΔMDD"] == -2.0
    assert result["ΔReturn"] == -15.0
    assert result["is_beneficial"] is False


def test_effective_rule_count_is_participation_ratio() -> None:
    assert effective_rule_count(np.eye(3)) == 3.0
    assert effective_rule_count(np.ones((3, 3))) == 1.0
    assert effective_rule_count(np.asarray([1.0, 1.0, 2.0])) == 16.0 / 6.0


def test_effective_rule_count_handles_empty_and_single_rule_inputs() -> None:
    assert effective_rule_count(np.empty((0, 0))) == 0.0
    assert effective_rule_count(np.asarray([[1.0]])) == 1.0
    assert effective_rule_count(np.asarray([3.0])) == 1.0


class _MarginalEngine:
    """Small exact-engine double keyed by the rule names it receives."""

    def simulate_rule_set(self, rules: list[dict]) -> dict:
        names = {str(rule["conditions"][0]) for rule in rules}
        if len(names) == 3:
            return {
                "total_return_pct": 10.0,
                "sortino_ratio": 0.4,
                "max_drawdown_pct": 7.0,
                "profit_factor": 1.5,
                "executed_trades": 30,
            }
        if "B" in names:
            return {
                "total_return_pct": 9.0,
                "sortino_ratio": 0.3,
                "max_drawdown_pct": 8.0,
                "profit_factor": 1.5,
                "executed_trades": 30,
            }
        metrics = {
            "total_return_pct": 12.0,
            "sortino_ratio": 1.0,
            "max_drawdown_pct": 4.0,
            "profit_factor": 1.5,
            "executed_trades": 30,
        }
        return metrics


def test_enabled_marginal_pruning_removes_negative_rule_in_two_pass_bound() -> None:
    records = [
        CandidateRecord(
            rule={"conditions": [name]},
            train_metrics={},
            valid_metrics={},
            score=1.0,
            mask=np.asarray([index == 0 for index in range(3)]),
        )
        for name in ("A", "B", "C")
    ]
    engine = _MarginalEngine()
    old_pruning = _cfg.RB_MARGINAL_PRUNING
    old_min_rules = _cfg.RB_MIN_RULES
    try:
        _cfg.RB_MARGINAL_PRUNING = True
        _cfg.RB_MIN_RULES = 1
        selected, _train, _valid, report = _marginal_prune_ruleset(
            records,
            engine,
            engine,
            "long",
        )
    finally:
        _cfg.RB_MARGINAL_PRUNING = old_pruning
        _cfg.RB_MIN_RULES = old_min_rules

    assert [record.rule["conditions"][0] for record in selected] == ["A", "C"]
    assert report["pruned"] is True
    assert report["selected_rule_count"] == 2
    assert report["removed_rule_count"] == 1
    assert len(report["passes"]) <= 2
    removed = [row for row in report["per_rule"] if row["removed"]]
    assert [row["conditions"] for row in removed] == [["B"]]
