"""Unit tests for RB Governor tail-holdout path in risk grid.

Covers:
- ``_make_walk_forward_fold_engines`` with tail holdout fraction > 0
- ``_optimize_risk`` with ``tail_holdout_engine`` produces
  ``risk_tail_holdout_return_pct``, ``risk_tail_holdout_pf``,
  ``risk_tail_holdout_dd`` in the final history entry.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.rb_governor import (
    _score_metrics,
    _make_walk_forward_fold_engines,
    _optimize_risk,
    _evaluate_ruleset,
    _rule_to_engine,
    CandidateRecord,
)
from gpu_fuzzy_trader.backtest.cpu_engine import CPUBacktestEngine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _train_metrics(
    *,
    return_pct: float = 5.0,
    dd: float = 2.0,
    pf: float = 1.5,
    wr: float = 55.0,
    trades: int = 30,
) -> dict:
    raw = max(trades + 5, 40)
    return {
        "total_return_pct": return_pct,
        "max_drawdown_pct": dd,
        "profit_factor": pf,
        "win_rate": wr,
        "executed_trades": trades,
        "raw_signal_count": raw,
        "skipped_min_notional_count": 2,
        "max_simultaneous_positions": 2,
        "eval_v5_zoo_health_pct": 100.0,
        "eval_v5_success_rate_pct": 100.0,
    }


def _valid_metrics(
    *,
    return_pct: float = 4.0,
    dd: float = 2.0,
    pf: float = 1.5,
    wr: float = 55.0,
    trades: int = 20,
) -> dict:
    raw = max(trades + 5, 40)
    return {
        "total_return_pct": return_pct,
        "max_drawdown_pct": dd,
        "profit_factor": pf,
        "win_rate": wr,
        "executed_trades": trades,
        "raw_signal_count": raw,
        "skipped_min_notional_count": 2,
        "max_simultaneous_positions": 2,
        "eval_v5_zoo_health_pct": 100.0,
        "eval_v5_success_rate_pct": 100.0,
    }


class _MockEngine:
    """Minimal mock that mimics CPUBacktestEngine for testing _optimize_risk."""
    def __init__(self, metrics: dict | None = None):
        self._metrics = metrics or _valid_metrics()
        self.called_with: list = []

    def simulate_rule_set(self, rules):
        self.called_with.append(copy.deepcopy(rules))
        return dict(self._metrics)


# ---------------------------------------------------------------------------
# Tail-holdout engine builder tests
# ---------------------------------------------------------------------------


class TestMakeFoldEnginesTailHoldout:
    """Verify tail holdout engine is created when fraction > 0."""

    def _make_df(self, symbols: list[str], rows_per_sym: int) -> pd.DataFrame:
        rows: list[dict] = []
        for sym in symbols:
            for i in range(rows_per_sym):
                rows.append({
                    "symbol": sym,
                    "datetime": pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5 * i),
                    "_symbol_bar_index": i,
                    "label_open_next": 100.0,
                    "label_max_288": 101.0,
                    "label_min_288": 99.0,
                    "label_close_288": 100.5,
                    "label_max_before_min": 1,
                    "feat_a": 0.5,
                })
        return pd.DataFrame(rows)

    def test_tail_holdout_engine_created(self):
        """With tail_holdout_frac=0.25, a tail engine is returned with ~25% of data."""
        df = self._make_df(["SYM1", "SYM2"], rows_per_sym=100)
        folds, tail = _make_walk_forward_fold_engines(df, n_splits=2, tail_holdout_frac=0.25, direction="long")
        assert tail is not None
        assert isinstance(tail, CPUBacktestEngine)
        # fold engines should also be CPUBacktestEngine
        assert len(folds) == 2

    def test_tail_holdout_engine_none_when_frac_zero(self):
        """With tail_holdout_frac=0.0, no tail engine."""
        df = self._make_df(["SYM1"], rows_per_sym=100)
        folds, tail = _make_walk_forward_fold_engines(df, n_splits=2, tail_holdout_frac=0.0, direction="long")
        assert tail is None

    def test_tail_holdout_with_single_symbol(self):
        """Single symbol with tail holdout still works."""
        df = self._make_df(["SYM1"], rows_per_sym=40)
        folds, tail = _make_walk_forward_fold_engines(df, n_splits=2, tail_holdout_frac=0.25, direction="long")
        assert tail is not None
        assert len(folds) == 2


# ---------------------------------------------------------------------------
# Tail-holdout fields in history
# ---------------------------------------------------------------------------


class TestOptimizeRiskTailHoldoutFields:
    """Verify _optimize_risk with tail_holdout_engine adds tail fields to final history entry."""

    def _make_selected(self) -> list[CandidateRecord]:
        rule = {"conditions": ["symbol is SYM1"], "tp": 2.0, "sl": 1.2, "capital_pct": 20.0}
        return [CandidateRecord(rule=rule, train_metrics=_train_metrics(), valid_metrics=_valid_metrics(), score=100.0)]

    def test_tail_fields_appear_in_final_history_entry(self, monkeypatch):
        """When tail_holdout_engine is provided, the final history entry
        contains risk_tail_holdout_return_pct, risk_tail_holdout_pf,
        and risk_tail_holdout_dd."""
        from gpu_fuzzy_trader import rb_governor as _rg

        selected = self._make_selected()
        train_mock = _MockEngine(_train_metrics(return_pct=5.0))
        valid_mock = _MockEngine(_valid_metrics(return_pct=4.0))
        fold1_mock = _MockEngine(_valid_metrics(return_pct=3.0))
        fold2_mock = _MockEngine(_valid_metrics(return_pct=4.5))

        # Tail engine returns specific metrics
        tail_mock = _MockEngine(_valid_metrics(return_pct=2.5, pf=1.3, dd=3.0))

        original_eval = _rg._evaluate_ruleset

        def mock_eval(train_eng, valid_eng, rules):
            train_m = train_eng.simulate_rule_set(rules)
            valid_m = valid_eng.simulate_rule_set(rules)
            score = _score_metrics(train_m, valid_m, min_train_trades=5, min_valid_trades=5)
            return train_m, valid_m, score

        monkeypatch.setattr(_rg, "_evaluate_ruleset", mock_eval)
        monkeypatch.setattr(_cfg, "RB_TP_GRID", (2.0,))
        monkeypatch.setattr(_cfg, "RB_SL_GRID", (1.2,))
        monkeypatch.setattr(_cfg, "RB_CAPITAL_GRID", (20.0,))
        monkeypatch.setattr(_cfg, "RB_RISK_OPT_PASSES", 1)
        monkeypatch.setattr(_cfg, "RB_MAX_TOTAL_CAPITAL", 100.0)
        monkeypatch.setattr(_cfg, "RB_RULESET_MIN_TRAIN_TRADES", 5)
        monkeypatch.setattr(_cfg, "RB_RULESET_MIN_VALID_TRADES", 5)
        monkeypatch.setattr(_cfg, "RB_MIN_TRAIN_TRADES", 5)
        monkeypatch.setattr(_cfg, "RB_MIN_VALID_TRADES", 5)
        monkeypatch.setattr(_cfg, "RB_RISK_MIN_IMPROVEMENT", 0.01)
        monkeypatch.setattr(_cfg, "RB_REQUIRE_TRAIN_SLIGHTLY_ABOVE_VALID", False)

        rules, train, valid, score, hist = _optimize_risk(
            selected, train_mock, valid_mock, "long",
            fold_engines=[fold1_mock, fold2_mock],
            tail_holdout_engine=tail_mock,
        )

        # The final history entry should have the tail fields
        final_entry = hist[-1]
        assert "risk_tail_holdout_return_pct" in final_entry, (
            f"Final entry missing risk_tail_holdout_return_pct: {final_entry}"
        )
        assert "risk_tail_holdout_pf" in final_entry
        assert "risk_tail_holdout_dd" in final_entry

        # The tail metrics should match what our mock returned
        assert final_entry["risk_tail_holdout_return_pct"] == 2.5
        assert final_entry["risk_tail_holdout_pf"] == 1.3
        assert final_entry["risk_tail_holdout_dd"] == 3.0

    def test_no_tail_fields_when_tail_engine_none(self, monkeypatch):
        """When tail_holdout_engine=None, NO tail fields in history."""
        from gpu_fuzzy_trader import rb_governor as _rg

        selected = self._make_selected()
        train_mock = _MockEngine(_train_metrics(return_pct=5.0))
        valid_mock = _MockEngine(_valid_metrics(return_pct=4.0))
        fold1_mock = _MockEngine(_valid_metrics(return_pct=3.0))
        fold2_mock = _MockEngine(_valid_metrics(return_pct=4.5))

        original_eval = _rg._evaluate_ruleset

        def mock_eval(train_eng, valid_eng, rules):
            train_m = train_eng.simulate_rule_set(rules)
            valid_m = valid_eng.simulate_rule_set(rules)
            score = _score_metrics(train_m, valid_m, min_train_trades=5, min_valid_trades=5)
            return train_m, valid_m, score

        monkeypatch.setattr(_rg, "_evaluate_ruleset", mock_eval)
        monkeypatch.setattr(_cfg, "RB_TP_GRID", (2.0,))
        monkeypatch.setattr(_cfg, "RB_SL_GRID", (1.2,))
        monkeypatch.setattr(_cfg, "RB_CAPITAL_GRID", (20.0,))
        monkeypatch.setattr(_cfg, "RB_RISK_OPT_PASSES", 1)
        monkeypatch.setattr(_cfg, "RB_MAX_TOTAL_CAPITAL", 100.0)
        monkeypatch.setattr(_cfg, "RB_RULESET_MIN_TRAIN_TRADES", 5)
        monkeypatch.setattr(_cfg, "RB_RULESET_MIN_VALID_TRADES", 5)
        monkeypatch.setattr(_cfg, "RB_MIN_TRAIN_TRADES", 5)
        monkeypatch.setattr(_cfg, "RB_MIN_VALID_TRADES", 5)
        monkeypatch.setattr(_cfg, "RB_RISK_MIN_IMPROVEMENT", 0.01)
        monkeypatch.setattr(_cfg, "RB_REQUIRE_TRAIN_SLIGHTLY_ABOVE_VALID", False)

        rules, train, valid, score, hist = _optimize_risk(
            selected, train_mock, valid_mock, "long",
            fold_engines=[fold1_mock, fold2_mock],
            tail_holdout_engine=None,
        )

        for entry in hist:
            assert "risk_tail_holdout_return_pct" not in entry
            assert "risk_tail_holdout_pf" not in entry
            assert "risk_tail_holdout_dd" not in entry
