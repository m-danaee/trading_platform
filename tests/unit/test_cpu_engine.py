"""
Unit tests for CPUBacktestEngine.

Tests verify exact evaluator_v3.ipynb semantics:
  - apply_dynamic_rule threshold logic
  - Priority-based rule assignment
  - Trade outcome logic (long/short, TP/SL/time-exit, max_before_min)
  - Capital management and position sizing
  - Exposure reservation and release
  - Fee deduction
  - Equity tracking and drawdown
  - Account ruin detection
  - Per-symbol metrics (Requirement 15.1)
  - return_logs DataFrame format
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gpu_fuzzy_trader.backtest.cpu_engine import (
    CPUBacktestEngine,
    _apply_dynamic_rule,
    _build_entries_from_rule_set,
    _normalize_direction,
    _safe_profit_factor,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_df(
    n: int = 10,
    symbol: str = "SYM",
    entry: float = 100.0,
    label_max: float = 105.0,
    label_min: float = 97.0,
    label_close: float = 102.0,
    max_before_min: int = 1,
    feature_val: float = 0.9,
) -> pd.DataFrame:
    """Build a minimal DataFrame for testing."""
    return pd.DataFrame(
        {
            "symbol": [symbol] * n,
            "datetime": pd.date_range("2024-01-01", periods=n, freq="5min"),
            "_symbol_bar_index": list(range(n)),
            "label_open_next": [entry] * n,
            "label_max_288": [label_max] * n,
            "label_min_288": [label_min] * n,
            "label_close_288": [label_close] * n,
            "label_max_before_min": [max_before_min] * n,
            "feat_a": [feature_val] * n,
        }
    )


def _make_engine(df: pd.DataFrame, direction: str = "long", **kw) -> CPUBacktestEngine:
    return CPUBacktestEngine(df, feature_modes={}, direction=direction, **kw)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestNormalizeDirection:
    def test_long(self):
        assert _normalize_direction("long") == "long"

    def test_short(self):
        assert _normalize_direction("SHORT") == "short"

    def test_invalid(self):
        with pytest.raises(ValueError):
            _normalize_direction("buy")


class TestSafeProfitFactor:
    def test_normal(self):
        assert _safe_profit_factor(100.0, 50.0) == pytest.approx(2.0)

    def test_no_losses_with_wins(self):
        assert _safe_profit_factor(100.0, 0.0) == 99.0

    def test_no_losses_no_wins(self):
        assert _safe_profit_factor(0.0, 0.0) == 0.0


class TestApplyDynamicRule:
    """Test all threshold branches of _apply_dynamic_rule."""

    def _df(self, vals):
        return pd.DataFrame({"f": vals})

    def test_active_1(self):
        df = self._df([0, 1, 0, 1])
        result = _apply_dynamic_rule(df, "[f] IS Active (1)")
        np.testing.assert_array_equal(result, [False, True, False, True])

    def test_inactive_0(self):
        df = self._df([0, 1])
        result = _apply_dynamic_rule(df, "[f] IS Inactive (0)")
        np.testing.assert_array_equal(result, [True, False])

    def test_positive_1(self):
        df = self._df([-1, 0, 1])
        result = _apply_dynamic_rule(df, "[f] IS Positive (1)")
        np.testing.assert_array_equal(result, [False, False, True])

    def test_neutral_0(self):
        df = self._df([-1, 0, 1])
        result = _apply_dynamic_rule(df, "[f] IS Neutral (0)")
        np.testing.assert_array_equal(result, [False, True, False])

    def test_negative_minus1(self):
        df = self._df([-1, 0, 1])
        result = _apply_dynamic_rule(df, "[f] IS Negative (-1)")
        np.testing.assert_array_equal(result, [True, False, False])

    def test_strong_negative(self):
        df = self._df([-0.5, -0.25, -0.1, 0.0])
        result = _apply_dynamic_rule(df, "[f] IS Strong Negative")
        np.testing.assert_array_equal(result, [True, True, False, False])

    def test_weak_negative(self):
        # Weak Negative: (s > -0.25) & (s <= -1e-5)
        # -0.3 is NOT > -0.25 → False; -0.1 is in range → True; -1e-5 is the boundary (included) → True; 0.0 is not <= -1e-5 → False
        df = self._df([-0.3, -0.1, -1e-5, 0.0])
        result = _apply_dynamic_rule(df, "[f] IS Weak Negative")
        np.testing.assert_array_equal(result, [False, True, True, False])

    def test_exactly_zero(self):
        # Exactly Zero: (s > -1e-5) & (s <= 1e-5)
        # -1e-5 is NOT > -1e-5 → False; 0.0 is in range → True; 1e-5 is the boundary (included) → True; 0.1 is not <= 1e-5 → False
        df = self._df([-1e-5, 0.0, 1e-5, 0.1])
        result = _apply_dynamic_rule(df, "[f] IS Exactly Zero")
        np.testing.assert_array_equal(result, [False, True, True, False])

    def test_weak_positive(self):
        df = self._df([0.0, 1e-4, 0.1, 0.25, 0.3])
        result = _apply_dynamic_rule(df, "[f] IS Weak Positive")
        np.testing.assert_array_equal(result, [False, True, True, True, False])

    def test_strong_positive(self):
        df = self._df([0.25, 0.26, 1.0])
        result = _apply_dynamic_rule(df, "[f] IS Strong Positive")
        np.testing.assert_array_equal(result, [False, True, True])

    def test_very_low(self):
        df = self._df([0.0, 0.2, 0.21])
        result = _apply_dynamic_rule(df, "[f] IS Very Low")
        np.testing.assert_array_equal(result, [True, True, False])

    def test_low(self):
        df = self._df([0.2, 0.3, 0.4, 0.41])
        result = _apply_dynamic_rule(df, "[f] IS Low")
        np.testing.assert_array_equal(result, [False, True, True, False])

    def test_medium(self):
        df = self._df([0.4, 0.5, 0.6, 0.61])
        result = _apply_dynamic_rule(df, "[f] IS Medium")
        np.testing.assert_array_equal(result, [False, True, True, False])

    def test_high(self):
        df = self._df([0.6, 0.7, 0.8, 0.81])
        result = _apply_dynamic_rule(df, "[f] IS High")
        np.testing.assert_array_equal(result, [False, True, True, False])

    def test_very_high(self):
        df = self._df([0.8, 0.81, 1.0])
        result = _apply_dynamic_rule(df, "[f] IS Very High")
        np.testing.assert_array_equal(result, [False, True, True])

    def test_extreme_bearish(self):
        df = self._df([-1.0, -0.8, -0.79])
        result = _apply_dynamic_rule(df, "[f] IS Extreme Bearish")
        np.testing.assert_array_equal(result, [True, True, False])

    def test_extreme_bullish(self):
        df = self._df([0.8, 0.81, 1.0])
        result = _apply_dynamic_rule(df, "[f] IS Extreme Bullish")
        np.testing.assert_array_equal(result, [False, True, True])

    def test_unknown_value_raises(self):
        df = self._df([0.5])
        with pytest.raises(ValueError, match="not recognized"):
            _apply_dynamic_rule(df, "[f] IS Unknown Value")

    def test_unknown_feature_raises(self):
        df = self._df([0.5])
        with pytest.raises(ValueError, match="Unknown feature"):
            _apply_dynamic_rule(df, "[missing_col] IS Very High")

    def test_invalid_condition_format_raises(self):
        df = self._df([0.5])
        with pytest.raises(ValueError):
            _apply_dynamic_rule(df, "f = Very High")



# ---------------------------------------------------------------------------
# Priority-based rule assignment
# ---------------------------------------------------------------------------

class TestBuildEntriesFromRuleSet:
    def test_empty_rule_set_returns_empty(self):
        df = _make_df(5, feature_val=0.9)
        entries = _build_entries_from_rule_set(df, [])
        assert entries == []

    def test_single_rule_all_match(self):
        df = _make_df(3, feature_val=0.9)
        rule_set = [{"conditions": ["[feat_a] IS Very High"], "tp": 2.0, "sl": 1.0, "capital_pct": 50.0}]
        entries = _build_entries_from_rule_set(df, rule_set)
        assert len(entries) == 3
        assert all(e["rule_index"] == 1 for e in entries)

    def test_priority_first_rule_wins(self):
        """Row matching both rules should be assigned to rule 1 only."""
        df = _make_df(3, feature_val=0.9)
        rule_set = [
            {"conditions": ["[feat_a] IS Very High"], "tp": 2.0, "sl": 1.0, "capital_pct": 50.0},
            {"conditions": ["[feat_a] IS Very High"], "tp": 3.0, "sl": 1.5, "capital_pct": 30.0},
        ]
        entries = _build_entries_from_rule_set(df, rule_set)
        # All rows match rule 1; rule 2 gets nothing
        assert all(e["rule_index"] == 1 for e in entries)
        assert len(entries) == 3

    def test_second_rule_gets_unmatched_rows(self):
        """Rows not matching rule 1 should be assigned to rule 2."""
        vals = [0.9, 0.9, 0.5, 0.5]  # first 2 → Very High, last 2 → Medium
        df = pd.DataFrame({
            "symbol": ["S"] * 4,
            "datetime": pd.date_range("2024-01-01", periods=4, freq="5min"),
            "_symbol_bar_index": list(range(4)),
            "label_open_next": [100.0] * 4,
            "label_max_288": [105.0] * 4,
            "label_min_288": [97.0] * 4,
            "label_close_288": [102.0] * 4,
            "label_max_before_min": [1] * 4,
            "feat_a": vals,
        })
        rule_set = [
            {"conditions": ["[feat_a] IS Very High"], "tp": 2.0, "sl": 1.0, "capital_pct": 50.0},
            {"conditions": ["[feat_a] IS Medium"], "tp": 3.0, "sl": 1.5, "capital_pct": 30.0},
        ]
        entries = _build_entries_from_rule_set(df, rule_set)
        assert len(entries) == 4
        rule_indices = {e["idx"]: e["rule_index"] for e in entries}
        assert rule_indices[0] == 1
        assert rule_indices[1] == 1
        assert rule_indices[2] == 2
        assert rule_indices[3] == 2

    def test_entries_sorted_by_row_index(self):
        df = _make_df(5, feature_val=0.9)
        rule_set = [{"conditions": ["[feat_a] IS Very High"], "tp": 2.0, "sl": 1.0, "capital_pct": 50.0}]
        entries = _build_entries_from_rule_set(df, rule_set)
        indices = [e["idx"] for e in entries]
        assert indices == sorted(indices)

    def test_no_matching_rows_returns_empty(self):
        df = _make_df(3, feature_val=0.5)  # Medium, not Very High
        rule_set = [{"conditions": ["[feat_a] IS Very High"], "tp": 2.0, "sl": 1.0, "capital_pct": 50.0}]
        entries = _build_entries_from_rule_set(df, rule_set)
        assert entries == []

    def test_invalid_capital_pct_raises(self):
        df = _make_df(3, feature_val=0.9)
        rule_set = [{"conditions": ["[feat_a] IS Very High"], "tp": 2.0, "sl": 1.0, "capital_pct": 0.0}]
        with pytest.raises(ValueError, match="invalid capital_pct"):
            _build_entries_from_rule_set(df, rule_set)



# ---------------------------------------------------------------------------
# Trade outcome logic
# ---------------------------------------------------------------------------

class TestTradeOutcomeLong:
    """Test _build_trade_outcome_single for long direction."""

    def _engine(self, max_288, min_288, close_288, mbm):
        df = pd.DataFrame({
            "symbol": ["S"],
            "datetime": pd.date_range("2024-01-01", periods=1, freq="5min"),
            "_symbol_bar_index": [0],
            "label_open_next": [100.0],
            "label_max_288": [max_288],
            "label_min_288": [min_288],
            "label_close_288": [close_288],
            "label_max_before_min": [mbm],
            "feat_a": [0.9],
        })
        return _make_engine(df, direction="long")

    def test_tp_only(self):
        eng = self._engine(max_288=105.0, min_288=99.0, close_288=103.0, mbm=1)
        # max_ret = 5%, min_ret = -1%, tp=4%, sl=2% → hit_tp=True, hit_sl=False
        ret, reason = eng._build_trade_outcome_single(0, tp=4.0, sl=2.0)
        assert ret == pytest.approx(4.0)
        assert reason == "TP"

    def test_sl_only(self):
        eng = self._engine(max_288=101.0, min_288=97.0, close_288=99.0, mbm=1)
        # max_ret=1%, min_ret=-3%, tp=4%, sl=2% → hit_tp=False, hit_sl=True
        ret, reason = eng._build_trade_outcome_single(0, tp=4.0, sl=2.0)
        assert ret == pytest.approx(-2.0)
        assert reason == "SL"

    def test_both_hit_max_before_min_1_tp_first(self):
        eng = self._engine(max_288=106.0, min_288=97.0, close_288=103.0, mbm=1)
        # Both hit; mbm=1 → TP first
        ret, reason = eng._build_trade_outcome_single(0, tp=4.0, sl=2.0)
        assert ret == pytest.approx(4.0)
        assert reason == "TP"

    def test_both_hit_max_before_min_0_sl_first(self):
        eng = self._engine(max_288=106.0, min_288=97.0, close_288=103.0, mbm=0)
        # Both hit; mbm=0 → SL first
        ret, reason = eng._build_trade_outcome_single(0, tp=4.0, sl=2.0)
        assert ret == pytest.approx(-2.0)
        assert reason == "SL"

    def test_time_exit(self):
        eng = self._engine(max_288=101.0, min_288=99.0, close_288=101.5, mbm=1)
        # Neither TP nor SL hit; close_ret = 1.5%
        ret, reason = eng._build_trade_outcome_single(0, tp=4.0, sl=2.0)
        assert ret == pytest.approx(1.5)
        assert reason == "Time_288"


class TestTradeOutcomeShort:
    """Test _build_trade_outcome_single for short direction."""

    def _engine(self, max_288, min_288, close_288, mbm):
        df = pd.DataFrame({
            "symbol": ["S"],
            "datetime": pd.date_range("2024-01-01", periods=1, freq="5min"),
            "_symbol_bar_index": [0],
            "label_open_next": [100.0],
            "label_max_288": [max_288],
            "label_min_288": [min_288],
            "label_close_288": [close_288],
            "label_max_before_min": [mbm],
            "feat_a": [0.9],
        })
        return _make_engine(df, direction="short")

    def test_tp_only_short(self):
        # Short TP: min_ret <= -tp → min_288 <= entry*(1-tp/100) = 96
        eng = self._engine(max_288=101.0, min_288=95.0, close_288=98.0, mbm=1)
        # min_ret = -5%, tp=4% → hit_tp: -5 <= -4 True; max_ret=1%, sl=2% → hit_sl: 1>=2 False
        ret, reason = eng._build_trade_outcome_single(0, tp=4.0, sl=2.0)
        assert ret == pytest.approx(4.0)
        assert reason == "TP"

    def test_sl_only_short(self):
        # Short SL: max_ret >= sl → max_288 >= entry*(1+sl/100) = 102
        eng = self._engine(max_288=103.0, min_288=99.0, close_288=101.0, mbm=1)
        # max_ret=3%, sl=2% → hit_sl True; min_ret=-1%, tp=4% → hit_tp: -1<=-4 False
        ret, reason = eng._build_trade_outcome_single(0, tp=4.0, sl=2.0)
        assert ret == pytest.approx(-2.0)
        assert reason == "SL"

    def test_both_hit_short_mbm_1_sl_first(self):
        # Both hit; mbm=1 → SL first for short
        eng = self._engine(max_288=103.0, min_288=95.0, close_288=99.0, mbm=1)
        ret, reason = eng._build_trade_outcome_single(0, tp=4.0, sl=2.0)
        assert ret == pytest.approx(-2.0)
        assert reason == "SL"

    def test_both_hit_short_mbm_0_tp_first(self):
        # Both hit; mbm=0 → TP first for short
        eng = self._engine(max_288=103.0, min_288=95.0, close_288=99.0, mbm=0)
        ret, reason = eng._build_trade_outcome_single(0, tp=4.0, sl=2.0)
        assert ret == pytest.approx(4.0)
        assert reason == "TP"

    def test_time_exit_short(self):
        # Neither hit; short time exit = -close_ret
        eng = self._engine(max_288=101.0, min_288=99.0, close_288=101.5, mbm=1)
        # close_ret = 1.5%, short returns -1.5%
        ret, reason = eng._build_trade_outcome_single(0, tp=4.0, sl=2.0)
        assert ret == pytest.approx(-1.5)
        assert reason == "Time_288"



# ---------------------------------------------------------------------------
# Release index precomputation
# ---------------------------------------------------------------------------

class TestPrecomputeReleaseIndices:
    def test_release_index_within_symbol(self):
        """Release index should point to the row where bar_index + 288 is reached."""
        n = 600
        df = pd.DataFrame({
            "symbol": ["S"] * n,
            "datetime": pd.date_range("2024-01-01", periods=n, freq="5min"),
            "_symbol_bar_index": list(range(n)),
            "label_open_next": [100.0] * n,
            "label_max_288": [105.0] * n,
            "label_min_288": [97.0] * n,
            "label_close_288": [102.0] * n,
            "label_max_before_min": [1] * n,
            "feat_a": [0.9] * n,
        })
        eng = _make_engine(df, max_hold_candles=288)
        # Row 0 has bar_index=0; target_bar=288; row 288 has bar_index=288
        assert eng.release_index[0] == 288
        # Row 100 → target_bar=388 → row 388
        assert eng.release_index[100] == 388

    def test_release_index_at_end_when_no_future_bar(self):
        """Rows near the end should get release_index = len(df)."""
        n = 300
        df = pd.DataFrame({
            "symbol": ["S"] * n,
            "datetime": pd.date_range("2024-01-01", periods=n, freq="5min"),
            "_symbol_bar_index": list(range(n)),
            "label_open_next": [100.0] * n,
            "label_max_288": [105.0] * n,
            "label_min_288": [97.0] * n,
            "label_close_288": [102.0] * n,
            "label_max_before_min": [1] * n,
            "feat_a": [0.9] * n,
        })
        eng = _make_engine(df, max_hold_candles=288)
        # Row 299 → target_bar=587, not reachable → release_index = 300
        assert eng.release_index[299] == 300

    def test_release_index_multi_symbol(self):
        """Each symbol's release indices are computed independently."""
        n = 20
        df = pd.DataFrame({
            "symbol": ["A"] * n + ["B"] * n,
            "datetime": pd.date_range("2024-01-01", periods=2 * n, freq="5min"),
            "_symbol_bar_index": list(range(n)) + list(range(n)),
            "label_open_next": [100.0] * (2 * n),
            "label_max_288": [105.0] * (2 * n),
            "label_min_288": [97.0] * (2 * n),
            "label_close_288": [102.0] * (2 * n),
            "label_max_before_min": [1] * (2 * n),
            "feat_a": [0.9] * (2 * n),
        })
        eng = _make_engine(df, max_hold_candles=10)
        # Symbol A: row 0 → bar 0, target bar 10 → row 10 (bar_index=10)
        assert eng.release_index[0] == 10
        # Symbol B starts at row 20: row 20 → bar 0, target bar 10 → row 30
        assert eng.release_index[20] == 30



# ---------------------------------------------------------------------------
# Capital management and position sizing
# ---------------------------------------------------------------------------

class TestPositionSizing:
    def test_target_notional(self):
        """position_notional = equity * capital_pct/100 * leverage when exposure is free."""
        df = _make_df(1)
        eng = _make_engine(df, initial_capital=1000.0, leverage=1.0)
        notional, info = eng._calculate_position_notional(
            equity=1000.0, capital_pct=50.0, open_total_exposure=0.0
        )
        assert notional == pytest.approx(500.0)
        assert info["target_position_notional"] == pytest.approx(500.0)

    def test_capped_by_remaining_exposure(self):
        """position_notional is capped when exposure is nearly full."""
        df = _make_df(1)
        eng = _make_engine(df, initial_capital=1000.0, leverage=1.0, max_total_exposure_pct=100.0)
        # max_exposure = 1000; already 800 open → remaining = 200
        notional, info = eng._calculate_position_notional(
            equity=1000.0, capital_pct=50.0, open_total_exposure=800.0
        )
        assert notional == pytest.approx(200.0)
        assert info["remaining_total_capacity"] == pytest.approx(200.0)

    def test_zero_remaining_exposure(self):
        """When exposure is full, position_notional = 0."""
        df = _make_df(1)
        eng = _make_engine(df, initial_capital=1000.0, leverage=1.0, max_total_exposure_pct=100.0)
        notional, _ = eng._calculate_position_notional(
            equity=1000.0, capital_pct=50.0, open_total_exposure=1000.0
        )
        assert notional == pytest.approx(0.0)

    def test_leverage_multiplier(self):
        """Leverage multiplies both target and max_exposure."""
        df = _make_df(1)
        eng = _make_engine(df, initial_capital=1000.0, leverage=2.0, max_total_exposure_pct=100.0)
        notional, info = eng._calculate_position_notional(
            equity=1000.0, capital_pct=50.0, open_total_exposure=0.0
        )
        # target = 1000 * 0.5 * 2 = 1000; max_exposure = 1000 * 1.0 * 2 = 2000
        assert notional == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# Fee deduction
# ---------------------------------------------------------------------------

class TestFeeDeduction:
    def test_fee_deducted_from_gross_pnl(self):
        """net_pnl = gross_pnl - fee; fee = position_notional * fee_rate."""
        df = _make_df(
            n=5,
            entry=100.0,
            label_max=105.0,
            label_min=99.0,
            label_close=103.0,
            max_before_min=1,
            feature_val=0.9,
        )
        rule_set = [{"conditions": ["[feat_a] IS Very High"], "tp": 4.0, "sl": 2.0, "capital_pct": 50.0}]
        eng = _make_engine(df, initial_capital=1000.0, fee_pct=0.20, max_hold_candles=2)
        metrics, logs = eng.simulate_rule_set(rule_set, return_logs=True)

        for _, row in logs.iterrows():
            expected_fee = row["Position_Notional"] * 0.20 / 100.0
            assert row["Fee"] == pytest.approx(expected_fee, rel=1e-6)
            assert row["Net_PnL"] == pytest.approx(row["Gross_PnL"] - row["Fee"], rel=1e-6)


# ---------------------------------------------------------------------------
# Equity tracking and drawdown
# ---------------------------------------------------------------------------

class TestEquityTracking:
    def test_equity_increases_on_win(self):
        """After a winning trade, final_equity > initial_capital."""
        df = _make_df(
            n=5,
            entry=100.0,
            label_max=106.0,
            label_min=99.0,
            label_close=104.0,
            max_before_min=1,
            feature_val=0.9,
        )
        rule_set = [{"conditions": ["[feat_a] IS Very High"], "tp": 4.0, "sl": 2.0, "capital_pct": 50.0}]
        eng = _make_engine(df, initial_capital=1000.0, max_hold_candles=2)
        metrics = eng.simulate_rule_set(rule_set)
        assert metrics["final_equity"] > 1000.0
        assert metrics["total_return_pct"] > 0.0

    def test_equity_decreases_on_loss(self):
        """After a losing trade, final_equity < initial_capital."""
        df = _make_df(
            n=5,
            entry=100.0,
            label_max=101.0,
            label_min=97.0,
            label_close=99.0,
            max_before_min=0,
            feature_val=0.9,
        )
        rule_set = [{"conditions": ["[feat_a] IS Very High"], "tp": 4.0, "sl": 2.0, "capital_pct": 50.0}]
        eng = _make_engine(df, initial_capital=1000.0, max_hold_candles=2)
        metrics = eng.simulate_rule_set(rule_set)
        assert metrics["final_equity"] < 1000.0
        assert metrics["total_return_pct"] < 0.0

    def test_max_drawdown_non_negative(self):
        df = _make_df(n=10, feature_val=0.9)
        rule_set = [{"conditions": ["[feat_a] IS Very High"], "tp": 4.0, "sl": 2.0, "capital_pct": 50.0}]
        eng = _make_engine(df, max_hold_candles=2)
        metrics = eng.simulate_rule_set(rule_set)
        assert metrics["max_drawdown_pct"] >= 0.0

    def test_zero_trades_returns_initial_capital(self):
        df = _make_df(n=5, feature_val=0.5)  # Medium, not Very High
        rule_set = [{"conditions": ["[feat_a] IS Very High"], "tp": 4.0, "sl": 2.0, "capital_pct": 50.0}]
        eng = _make_engine(df, initial_capital=1000.0)
        metrics = eng.simulate_rule_set(rule_set)
        assert metrics["final_equity"] == pytest.approx(1000.0)
        assert metrics["total_return_pct"] == pytest.approx(0.0)
        assert metrics["executed_trades"] == 0



# ---------------------------------------------------------------------------
# Account ruin detection
# ---------------------------------------------------------------------------

class TestAccountRuin:
    def test_account_ruined_when_equity_drops_to_zero(self):
        """Simulate catastrophic losses to trigger account ruin.

        With min_288=0 (entry=100), min_ret = -100%. With sl=50% and capital_pct=100%,
        position_notional = equity. gross_pnl = -equity * 0.5. fee = equity * 0.002.
        net_pnl = -equity * 0.502. equity after = equity * 0.498.
        This converges but never reaches 0.

        Instead, use sl > 100% equivalent: make min_ret = -100% and sl=100%.
        Then gross_pnl = -notional, fee = notional * 0.002, net_pnl = -notional * 1.002.
        With capital_pct=100%, notional = equity, so equity becomes equity - equity*1.002 < 0.
        """
        n = 5
        df = pd.DataFrame({
            "symbol": ["S"] * n,
            "datetime": pd.date_range("2024-01-01", periods=n, freq="5min"),
            "_symbol_bar_index": list(range(n)),
            "label_open_next": [100.0] * n,
            "label_max_288": [100.1] * n,   # tiny max, no TP
            "label_min_288": [0.0] * n,     # min_ret = -100%
            "label_close_288": [0.0] * n,
            "label_max_before_min": [0] * n,  # SL first
            "feat_a": [0.9] * n,
        })
        # sl=100% → hit_sl when min_ret <= -100 → -100 <= -100 True
        # gross_pnl = notional * (-100/100) = -notional
        # net_pnl = -notional - notional*0.002 = -notional * 1.002
        # With capital_pct=100%, notional=equity → equity becomes equity - equity*1.002 < 0
        rule_set = [{"conditions": ["[feat_a] IS Very High"], "tp": 0.05, "sl": 100.0, "capital_pct": 100.0}]
        eng = _make_engine(df, initial_capital=1000.0, max_hold_candles=1)
        metrics = eng.simulate_rule_set(rule_set)
        assert metrics["account_ruined"] is True

    def test_account_not_ruined_on_normal_trades(self):
        df = _make_df(n=5, feature_val=0.9)
        rule_set = [{"conditions": ["[feat_a] IS Very High"], "tp": 4.0, "sl": 2.0, "capital_pct": 50.0}]
        eng = _make_engine(df, max_hold_candles=2)
        metrics = eng.simulate_rule_set(rule_set)
        assert metrics["account_ruined"] is False


# ---------------------------------------------------------------------------
# Min position notional filter
# ---------------------------------------------------------------------------

class TestMinPositionNotional:
    def test_skipped_when_below_min_notional(self):
        """Trades with tiny equity should be skipped."""
        df = _make_df(n=3, feature_val=0.9)
        rule_set = [{"conditions": ["[feat_a] IS Very High"], "tp": 4.0, "sl": 2.0, "capital_pct": 0.001}]
        # capital_pct=0.001% of 1000 = 0.01 < MIN_POSITION_NOTIONAL=1.0
        eng = _make_engine(df, initial_capital=1000.0, min_position_notional=1.0, max_hold_candles=2)
        metrics = eng.simulate_rule_set(rule_set)
        assert metrics["executed_trades"] == 0
        assert metrics["skipped_min_notional_count"] == 3


# ---------------------------------------------------------------------------
# Per-symbol metrics (Requirement 15.1)
# ---------------------------------------------------------------------------

class TestPerSymbolMetrics:
    def test_per_symbol_trade_counts(self):
        """Per-symbol metrics should reflect actual trade distribution."""
        n = 5
        df = pd.DataFrame({
            "symbol": ["A"] * n + ["B"] * n,
            "datetime": pd.date_range("2024-01-01", periods=2 * n, freq="5min"),
            "_symbol_bar_index": list(range(n)) + list(range(n)),
            "label_open_next": [100.0] * (2 * n),
            "label_max_288": [106.0] * (2 * n),
            "label_min_288": [99.0] * (2 * n),
            "label_close_288": [104.0] * (2 * n),
            "label_max_before_min": [1] * (2 * n),
            "feat_a": [0.9] * (2 * n),
        })
        # Use capital_pct=10% so exposure doesn't fill up (max 10 trades open at once)
        rule_set = [{"conditions": ["[feat_a] IS Very High"], "tp": 4.0, "sl": 2.0, "capital_pct": 10.0}]
        eng = _make_engine(df, initial_capital=1000.0, max_hold_candles=2)
        metrics, logs = eng.simulate_rule_set(rule_set, return_logs=True)

        per_sym = metrics["per_symbol_metrics"]
        assert "A" in per_sym
        assert "B" in per_sym
        assert per_sym["A"]["trade_count"] == n
        assert per_sym["B"]["trade_count"] == n

    def test_per_symbol_net_pnl_sign(self):
        """Winning trades should produce positive net_pnl per symbol."""
        n = 3
        df = pd.DataFrame({
            "symbol": ["WIN"] * n,
            "datetime": pd.date_range("2024-01-01", periods=n, freq="5min"),
            "_symbol_bar_index": list(range(n)),
            "label_open_next": [100.0] * n,
            "label_max_288": [106.0] * n,
            "label_min_288": [99.0] * n,
            "label_close_288": [104.0] * n,
            "label_max_before_min": [1] * n,
            "feat_a": [0.9] * n,
        })
        rule_set = [{"conditions": ["[feat_a] IS Very High"], "tp": 4.0, "sl": 2.0, "capital_pct": 50.0}]
        eng = _make_engine(df, initial_capital=1000.0, max_hold_candles=2)
        metrics, _ = eng.simulate_rule_set(rule_set, return_logs=True)
        assert metrics["per_symbol_metrics"]["WIN"]["net_pnl"] > 0.0


# ---------------------------------------------------------------------------
# return_logs DataFrame format
# ---------------------------------------------------------------------------

class TestReturnLogs:
    def test_logs_df_has_expected_columns(self):
        df = _make_df(n=3, feature_val=0.9)
        rule_set = [{"conditions": ["[feat_a] IS Very High"], "tp": 4.0, "sl": 2.0, "capital_pct": 50.0}]
        eng = _make_engine(df, max_hold_candles=2)
        metrics, logs_df = eng.simulate_rule_set(rule_set, return_logs=True)

        expected_cols = [
            "Trade_Number", "Direction", "Rule_Index", "Symbol",
            "Entry_Time", "Entry_Index", "Entry_Price",
            "Exit_Reason", "Price_Return_Pct",
            "Position_Notional", "Gross_PnL", "Fee", "Net_PnL",
        ]
        for col in expected_cols:
            assert col in logs_df.columns, f"Missing column: {col}"

    def test_logs_df_row_count_matches_executed_trades(self):
        df = _make_df(n=5, feature_val=0.9)
        rule_set = [{"conditions": ["[feat_a] IS Very High"], "tp": 4.0, "sl": 2.0, "capital_pct": 50.0}]
        eng = _make_engine(df, max_hold_candles=2)
        metrics, logs_df = eng.simulate_rule_set(rule_set, return_logs=True)
        assert len(logs_df) == metrics["executed_trades"]

    def test_no_logs_returns_metrics_only(self):
        df = _make_df(n=3, feature_val=0.9)
        rule_set = [{"conditions": ["[feat_a] IS Very High"], "tp": 4.0, "sl": 2.0, "capital_pct": 50.0}]
        eng = _make_engine(df, max_hold_candles=2)
        result = eng.simulate_rule_set(rule_set, return_logs=False)
        assert isinstance(result, dict)

    def test_empty_rule_set_returns_empty_df(self):
        df = _make_df(n=3, feature_val=0.9)
        eng = _make_engine(df)
        metrics, logs_df = eng.simulate_rule_set([], return_logs=True)
        assert isinstance(logs_df, pd.DataFrame)
        assert len(logs_df) == 0
        assert metrics["executed_trades"] == 0


# ---------------------------------------------------------------------------
# Metrics correctness
# ---------------------------------------------------------------------------

class TestMetricsCorrectness:
    def test_win_rate_all_wins(self):
        """All TP trades → win_rate = 100%."""
        df = _make_df(
            n=4,
            entry=100.0,
            label_max=106.0,
            label_min=99.0,
            label_close=104.0,
            max_before_min=1,
            feature_val=0.9,
        )
        rule_set = [{"conditions": ["[feat_a] IS Very High"], "tp": 4.0, "sl": 2.0, "capital_pct": 50.0}]
        eng = _make_engine(df, initial_capital=1000.0, max_hold_candles=2)
        metrics = eng.simulate_rule_set(rule_set)
        assert metrics["win_rate"] == pytest.approx(100.0)

    def test_profit_factor_no_losses(self):
        """No losing trades → profit_factor = 99.0."""
        df = _make_df(
            n=3,
            entry=100.0,
            label_max=106.0,
            label_min=99.0,
            label_close=104.0,
            max_before_min=1,
            feature_val=0.9,
        )
        rule_set = [{"conditions": ["[feat_a] IS Very High"], "tp": 4.0, "sl": 2.0, "capital_pct": 50.0}]
        eng = _make_engine(df, initial_capital=1000.0, max_hold_candles=2)
        metrics = eng.simulate_rule_set(rule_set)
        assert metrics["profit_factor"] == pytest.approx(99.0)

    def test_raw_signal_count_equals_matched_rows(self):
        df = _make_df(n=5, feature_val=0.9)
        rule_set = [{"conditions": ["[feat_a] IS Very High"], "tp": 4.0, "sl": 2.0, "capital_pct": 50.0}]
        eng = _make_engine(df, max_hold_candles=2)
        metrics = eng.simulate_rule_set(rule_set)
        assert metrics["raw_signal_count"] == 5

    def test_direction_in_metrics(self):
        df = _make_df(n=2, feature_val=0.9)
        rule_set = [{"conditions": ["[feat_a] IS Very High"], "tp": 4.0, "sl": 2.0, "capital_pct": 50.0}]
        eng_long = _make_engine(df, direction="long", max_hold_candles=2)
        eng_short = _make_engine(df, direction="short", max_hold_candles=2)
        assert eng_long.simulate_rule_set(rule_set)["direction"] == "long"
        assert eng_short.simulate_rule_set(rule_set)["direction"] == "short"

    def test_invalid_entry_price_raises(self):
        df = _make_df(n=3, feature_val=0.9)
        df["label_open_next"] = [100.0, 0.0, 100.0]  # zero entry price is invalid
        with pytest.raises(ValueError, match="Invalid label_open_next"):
            _make_engine(df)

