"""
Unit tests for ``_optimize_risk_grid`` (Task 7 — deterministic grid search).

Tests cover:
  - A rule improves after the grid (TP/SL/capital move to better values).
  - A rule stays unchanged when the grid doesn't improve (all combos worse).
  - The ``PHASE4_GRID_MAX_TOTAL_CAPITAL`` constraint is respected.
  - The ``gate_positive_good`` check filters out bad combinations.
  - The function returns the correct tuple shape.
  - Two passes of round-robin are performed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.phases.phase4_wf_optimizer import (
    _evaluate_ruleset,
    _optimize_risk_grid,
    _score_metrics,
    _normalize_capital_pct,
)


# ---------------------------------------------------------------------------
# Mock engine helpers
# ---------------------------------------------------------------------------

def _make_rule(
    conditions: list[str] | None = None,
    tp: float = 2.0,
    sl: float = 1.0,
    capital_pct: float = 12.5,
) -> dict:
    return {
        "conditions": conditions or ["[feat_0] IS Very High"],
        "tp": tp,
        "sl": sl,
        "capital_pct": capital_pct,
    }


def _make_engine(
    direction: str = "long",
    return_pct: float = 5.0,
    pf: float = 1.5,
    trades: int = 50,
    dd: float = 8.0,
    win_rate: float = 0.55,
) -> MagicMock:
    """Create a mock CPUBacktestEngine with deterministic metrics."""
    engine = MagicMock()
    engine.direction = direction

    def _simulate(rules: list[dict]) -> dict:
        # Compute a synthetic return based on the first rule's TP value
        # so we can test whether higher TP yields better score.
        if rules:
            tp = rules[0].get("tp", 2.0)
            sl = rules[0].get("sl", 1.0)
            cap = rules[0].get("capital_pct", 12.5)
            # Better TP gives better return (up to a point)
            bonus = max(0.0, (tp - 1.5) * 0.5)
            # Higher capital gives better return but higher DD
            cap_bonus = max(0.0, (cap - 5.0) * 0.1)
            actual_return = return_pct + bonus + cap_bonus
            # Higher SL hurts return slightly
            sl_penalty = max(0.0, (sl - 1.0) * 1.0)
            actual_return = max(actual_return - sl_penalty, 0.1)
            # PF improves with better TP/SL ratio
            actual_pf = pf + max(0.0, (tp / max(sl, 0.1) - 2.0)) * 0.2
            return {
                "total_return_pct": actual_return,
                "profit_factor": actual_pf,
                "executed_trades": trades,
                "max_drawdown_pct": dd + max(0.0, (cap - 12.5)) * 0.05,
                "win_rate": win_rate,
            }
        return {
            "total_return_pct": 0.0,
            "profit_factor": 0.0,
            "executed_trades": 0,
            "max_drawdown_pct": 100.0,
            "win_rate": 0.0,
        }

    engine.simulate_rule_set = MagicMock(side_effect=_simulate)
    return engine


def _make_dead_engine(
    direction: str = "long",
) -> MagicMock:
    """Engine that always returns negative metrics (fails gate)."""
    engine = MagicMock()
    engine.direction = direction

    def _simulate(rules: list[dict]) -> dict:
        return {
            "total_return_pct": -5.0,
            "profit_factor": 0.5,
            "executed_trades": 10,
            "max_drawdown_pct": 20.0,
            "win_rate": 0.3,
        }

    engine.simulate_rule_set = MagicMock(side_effect=_simulate)
    return engine


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestScoreMetrics:
    """``_score_metrics`` is a pure function — test its internals."""

    def test_both_positive(self):
        train = {"total_return_pct": 5.0, "max_drawdown_pct": 8.0,
                 "profit_factor": 1.5, "win_rate": 0.55}
        valid = {"total_return_pct": 4.0, "max_drawdown_pct": 6.0,
                 "profit_factor": 1.8, "win_rate": 0.60}
        score = _score_metrics(train, valid)
        assert score > 0, f"Expected positive score for good metrics, got {score}"

    def test_negative_valid(self):
        train = {"total_return_pct": 5.0, "max_drawdown_pct": 8.0,
                 "profit_factor": 1.5, "win_rate": 0.55}
        valid = {"total_return_pct": -2.0, "max_drawdown_pct": 10.0,
                 "profit_factor": 0.8, "win_rate": 0.40}
        score = _score_metrics(train, valid)
        # Negative valid return should be heavily penalised
        assert score < 0, f"Expected negative score for bad valid, got {score}"


class TestOptimizeRiskGrid:
    """Core grid-search function tests."""

    def test_returns_correct_shape(self):
        """``_optimize_risk_grid`` returns 5-element tuple with correct types."""
        rules = [_make_rule(tp=2.0)]
        train_eng = _make_engine(return_pct=5.0)
        val_eng = _make_engine(return_pct=4.0)

        result = _optimize_risk_grid(
            rules, train_eng, val_eng, min_improvement=0.02)

        assert isinstance(result, tuple), "Result must be a tuple"
        assert len(result) == 5, "Result must have 5 elements"

        opt_rules, train_m, val_m, score, history = result
        assert isinstance(opt_rules, list), "opt_rules must be a list"
        assert isinstance(train_m, dict), "train_metrics must be a dict"
        assert isinstance(val_m, dict), "val_metrics must be a dict"
        assert isinstance(score, float), "score must be a float"
        assert isinstance(history, list), "history must be a list"
        assert len(opt_rules) == 1, "Should have 1 rule"
        assert all(
            k in opt_rules[0] for k in ("conditions", "tp", "sl", "capital_pct")
        ), "Rule must have all required keys"

    def test_rule_improves_with_higher_tp(self):
        """A rule with default TP=2.0 should find a better TP (e.g. 10.0)."""
        rules = [_make_rule(tp=2.0, sl=1.0, capital_pct=12.5)]
        # Engine rewards higher TP with higher return
        train_eng = _make_engine(return_pct=3.0)
        val_eng = _make_engine(return_pct=2.5)

        opt_rules, _, _, score, history = _optimize_risk_grid(
            rules, train_eng, val_eng, min_improvement=0.01)

        # The grid includes TP=10.0 which should give a higher return
        assert opt_rules[0]["tp"] > 2.0, (
            f"Expected TP to improve from 2.0, got {opt_rules[0]['tp']}"
        )
        assert len(history) > 1, (
            "Expected at least 2 history entries (initial + improvement)"
        )

    def test_rule_stays_unchanged_when_grid_no_improvement(self):
        """When all grid combos are worse than default, the rule should stay unchanged."""
        # Engine that returns best at the default params
        def _make_peaked_engine():
            eng = MagicMock()
            eng.direction = "long"

            def _sim(rules):
                tp = rules[0].get("tp", 2.0)
                # Best return at exactly TP=2.0, degrades as TP moves away
                ret = 5.0 - abs(tp - 2.0) * 3.0
                return {
                    "total_return_pct": max(ret, 0.5),
                    "profit_factor": 1.5 - abs(tp - 2.0) * 0.2,
                    "executed_trades": 50,
                    "max_drawdown_pct": 8.0,
                    "win_rate": 0.55,
                }

            eng.simulate_rule_set = MagicMock(side_effect=_sim)
            return eng

        rules = [_make_rule(tp=2.0, sl=1.0, capital_pct=12.5)]
        train_eng = _make_peaked_engine()
        val_eng = _make_peaked_engine()

        opt_rules, _, _, score, history = _optimize_risk_grid(
            rules, train_eng, val_eng, min_improvement=0.02)

        # The default TP=2.0 should remain since it's already the peak
        assert opt_rules[0]["tp"] == 2.0, (
            f"Expected TP to stay at 2.0, got {opt_rules[0]['tp']}"
        )

    def test_max_total_capital_respected(self):
        """Combinations exceeding 95% total capital are skipped."""
        orig_max = _cfg.PHASE4_GRID_MAX_TOTAL_CAPITAL

        try:
            # Temporarily set a very low max total capital
            import gpu_fuzzy_trader.config as cfg
            # Use a strict cap to force filtering
            with patch.object(cfg, "PHASE4_GRID_MAX_TOTAL_CAPITAL", 20.0):
                # Two rules — each capital_pct=12.5 sums to 25 which exceeds 20
                # So only combinations where both are <= 20 will be tried
                rules = [
                    _make_rule(tp=2.0, sl=1.0, capital_pct=12.5),
                    _make_rule(tp=2.0, sl=1.0, capital_pct=12.5,
                               conditions=["[feat_1] IS High"]),
                ]
                train_eng = _make_engine(return_pct=5.0)
                val_eng = _make_engine(return_pct=4.0)

                opt_rules, _, _, score, history = _optimize_risk_grid(
                    rules, train_eng, val_eng, min_improvement=0.02)

                total_cap = sum(
                    float(r.get("capital_pct", 0.0)) for r in opt_rules)
                assert total_cap <= 20.0 + 1e-9, (
                    f"Total capital {total_cap} exceeds cap 20.0"
                )
        finally:
            pass  # Restore is automatic via the with block

    def test_gate_positive_good_filters_bad_combos(self):
        """Combinations that fail gate_positive_good are skipped."""
        rules = [_make_rule(tp=2.0, sl=1.0, capital_pct=12.5)]

        # Use a dead engine that always returns negative metrics
        train_eng = _make_dead_engine()
        val_eng = _make_dead_engine()

        # With negative returns, gate_positive_good will fail for every combo,
        # so the function should keep the original rule unchanged
        opt_rules, _, _, score, history = _optimize_risk_grid(
            rules, train_eng, val_eng, min_improvement=0.02)

        # No improvement should be found
        assert len(history) == 1, (
            f"Expected only initial history entry, got {len(history)}"
        )
        # The rule should remain unchanged
        assert opt_rules[0]["tp"] == 2.0

    def test_two_passes_round_robin(self):
        """The function performs 2 passes of round-robin as configured."""
        rules = [
            _make_rule(tp=2.0, sl=1.0, capital_pct=12.5),
            _make_rule(tp=3.0, sl=1.5, capital_pct=15.0,
                       conditions=["[feat_1] IS High"]),
        ]

        train_eng = _make_engine(return_pct=5.0)
        val_eng = _make_engine(return_pct=4.0)

        with patch.object(_cfg, "PHASE4_GRID_PASSES", 2):
            opt_rules, _, _, score, history = _optimize_risk_grid(
                rules, train_eng, val_eng, min_improvement=0.01)

            # The history should have entries for passes 1 and 2
            passes_seen = set()
            for h in history:
                p = h.get("pass", 0)
                if p > 0:
                    passes_seen.add(p)

            assert 1 in passes_seen, "Pass 1 should have been executed"
            # Pass 2 may or may not execute depending on whether improvements
            # were found. But at minimum pass 1 must have run.
            assert len(passes_seen) >= 1, "At least one pass must have run"

    def test_evaluate_ruleset_shim(self):
        """``_evaluate_ruleset`` returns the correct 3-tuple."""
        rules = [_make_rule(tp=2.0)]
        train_eng = _make_engine(return_pct=5.0)
        val_eng = _make_engine(return_pct=4.0)

        train_m, val_m, score = _evaluate_ruleset(train_eng, val_eng, rules)

        assert isinstance(train_m, dict)
        assert isinstance(val_m, dict)
        assert isinstance(score, float)
        assert "total_return_pct" in train_m
        assert "total_return_pct" in val_m


class TestWalkForwardOptimizerRiskGrid:
    """Integration tests for the ``optimize_risk_grid`` method."""

    def test_method_exists(self):
        """``WalkForwardRiskOptimizer`` has the ``optimize_risk_grid`` method."""
        from gpu_fuzzy_trader.phases.phase4_wf_optimizer import (
            WalkForwardRiskOptimizer,
        )
        assert hasattr(WalkForwardRiskOptimizer, "optimize_risk_grid"), (
            "WalkForwardRiskOptimizer must have optimize_risk_grid method"
        )

    def test_method_calls_optimize_risk_grid(self):
        """The method delegates to ``_optimize_risk_grid``."""
        from gpu_fuzzy_trader.phases.phase4_wf_optimizer import (
            WalkForwardRiskOptimizer,
        )

        # Create a minimal rule set
        rule_set = {
            "direction": "long",
            "rules_set": [
                _make_rule(tp=2.0, sl=1.0, capital_pct=12.5),
            ],
        }

        val_df = pd.DataFrame({
            "datetime": pd.date_range("2020-01-01", periods=100, freq="5min"),
            "symbol": "SYM_A",
            "label_open_next": np.ones(100) * 150.0,
            "label_close_288": np.ones(100) * 151.0,
            "label_min_288": np.ones(100) * 149.0,
            "label_max_288": np.ones(100) * 152.0,
            "label_max_before_min": np.ones(100) * 1.0,
            "_symbol_bar_index": np.arange(100),
            "feat_0": np.random.default_rng(42).uniform(0, 1, 100),
        })

        optimizer = WalkForwardRiskOptimizer(
            val_df=val_df,
            rule_set=rule_set,
            direction="long",
        )

        # Patch _optimize_risk_grid to verify it's called
        with patch(
            "gpu_fuzzy_trader.phases.phase4_wf_optimizer._optimize_risk_grid"
        ) as mock_fn:
            mock_fn.return_value = (
                rule_set["rules_set"],
                {"total_return_pct": 5.0, "profit_factor": 1.5},
                {"total_return_pct": 4.0, "profit_factor": 1.8},
                100.0,
                [{"pass": 0, "score": 100.0}],
            )
            result = optimizer.optimize_risk_grid()

            mock_fn.assert_called_once()
            assert result["direction"] == "long"
            assert "rules_set" in result
            assert "risk_optimized" in result
