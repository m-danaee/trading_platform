"""
Unit tests for gpu_fuzzy_trader.reporting.reporter.Reporter

Tests cover:
  - plot_phase2_metrics: normal case, empty history, single entry
  - plot_phase2_pnl: normal case, empty history, single entry
  - plot_equity_curve: normal case, empty trade_log, missing column, None trade_log
  - write_per_symbol_csv: normal case, empty metrics, missing key
  - plot_rl_curve: normal case, empty validation_returns, elbow_idx clamping
  - plot_per_rule_breakdown: file creation, path, direction validation, empty/None logs
"""

from __future__ import annotations
from gpu_fuzzy_trader.reporting.reporter import Reporter
import pytest
import pandas as pd
import numpy as np

import os

import matplotlib
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_history(n: int = 5) -> list[dict]:
    """Create a minimal Phase 2 history list."""
    return [
        {
            "generation": i,
            "mean_f1": -float(i) * 0.5,
            "mean_f2": float(i) * 0.1,
            "mean_f3": -float(i) * 0.3,
        }
        for i in range(n)
    ]


def _make_pnl_history(n: int = 5) -> list[dict]:
    """Create a minimal Phase 2 history list with PnL fields."""
    return [
        {
            "generation": i,
            "mean_sortino_ratio": float(i) * 2.0,
            "best_sortino_ratio": float(i) * 3.0 + 5.0,
        }
        for i in range(n)
    ]


def _make_trade_log(n: int = 20, seed: int = 42) -> pd.DataFrame:
    """Create a minimal trade log DataFrame with Equity_After column."""
    rng = np.random.default_rng(seed)
    equity = 1000.0 + np.cumsum(rng.uniform(-10, 15, size=n))
    return pd.DataFrame({"Equity_After": equity})


def _make_per_symbol_metrics(symbols: list[str] | None = None) -> dict:
    """Create a metrics dict with per_symbol_metrics."""
    if symbols is None:
        symbols = ["SYM_A", "SYM_B", "SYM_C"]
    return {
        "per_symbol_metrics": {
            sym: {
                "trade_count": 10,
                "win_rate": 55.0,
                "net_pnl": 25.0,
            }
            for sym in symbols
        }
    }


# ---------------------------------------------------------------------------
# Tests: plot_phase2_metrics
# ---------------------------------------------------------------------------

class TestPlotPhase2Metrics:
    def test_creates_png_file(self, tmp_path):
        reporter = Reporter()
        reporter.plot_phase2_metrics(
            _make_history(), "long", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "phase2_long_metrics.png")

    def test_creates_short_png_file(self, tmp_path):
        reporter = Reporter()
        reporter.plot_phase2_metrics(
            _make_history(), "short", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "phase2_short_metrics.png")

    def test_returns_correct_path(self, tmp_path):
        reporter = Reporter()
        result = reporter.plot_phase2_metrics(
            _make_history(), "long", output_dir=str(tmp_path))
        expected = os.path.join(str(tmp_path), "phase2_long_metrics.png")
        assert result == expected

    def test_empty_history_still_creates_file(self, tmp_path):
        reporter = Reporter()
        reporter.plot_phase2_metrics([], "long", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "phase2_long_metrics.png")

    def test_single_entry_history(self, tmp_path):
        reporter = Reporter()
        history = [{"generation": 0, "mean_f1": -
                    1.0, "mean_f2": 0.5, "mean_f3": -0.8}]
        reporter.plot_phase2_metrics(history, "long", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "phase2_long_metrics.png")

    def test_file_is_nonzero_size(self, tmp_path):
        reporter = Reporter()
        reporter.plot_phase2_metrics(_make_history(
            10), "long", output_dir=str(tmp_path))
        size = os.path.getsize(tmp_path / "phase2_long_metrics.png")
        assert size > 0

    def test_creates_parent_dirs(self, tmp_path):
        reporter = Reporter()
        nested_dir = str(tmp_path / "a" / "b" / "c")
        reporter.plot_phase2_metrics(
            _make_history(), "long", output_dir=nested_dir)
        assert os.path.exists(os.path.join(
            nested_dir, "phase2_long_metrics.png"))

    def test_missing_keys_in_history_entries(self, tmp_path):
        """History entries with missing keys should not raise."""
        reporter = Reporter()
        history = [{"generation": 0}, {"generation": 1, "mean_f1": -0.5}]
        reporter.plot_phase2_metrics(history, "long", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "phase2_long_metrics.png")


# ---------------------------------------------------------------------------
# Tests: plot_phase2_pnl
# ---------------------------------------------------------------------------

class TestPlotPhase2Pnl:
    def test_creates_long_png_file(self, tmp_path):
        reporter = Reporter()
        reporter.plot_phase2_pnl(
            _make_pnl_history(), "long", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "phase2_long_pnl.png")

    def test_creates_short_png_file(self, tmp_path):
        reporter = Reporter()
        reporter.plot_phase2_pnl(
            _make_pnl_history(), "short", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "phase2_short_pnl.png")

    def test_returns_correct_path(self, tmp_path):
        reporter = Reporter()
        result = reporter.plot_phase2_pnl(
            _make_pnl_history(), "long", output_dir=str(tmp_path))
        expected = os.path.join(str(tmp_path), "phase2_long_pnl.png")
        assert result == expected

    def test_empty_history_still_creates_file(self, tmp_path):
        reporter = Reporter()
        reporter.plot_phase2_pnl([], "long", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "phase2_long_pnl.png")

    def test_single_entry_history(self, tmp_path):
        reporter = Reporter()
        history = [{
            "generation": 0,
            "mean_sortino_ratio": 10.0,
            "best_sortino_ratio": 15.0,
        }]
        reporter.plot_phase2_pnl(history, "long", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "phase2_long_pnl.png")

    def test_file_is_nonzero_size(self, tmp_path):
        reporter = Reporter()
        reporter.plot_phase2_pnl(_make_pnl_history(
            10), "long", output_dir=str(tmp_path))
        size = os.path.getsize(tmp_path / "phase2_long_pnl.png")
        assert size > 0

    def test_creates_parent_dirs(self, tmp_path):
        reporter = Reporter()
        nested_dir = str(tmp_path / "a" / "b" / "c")
        reporter.plot_phase2_pnl(
            _make_pnl_history(), "long", output_dir=nested_dir)
        assert os.path.exists(os.path.join(nested_dir, "phase2_long_pnl.png"))

    def test_missing_keys_in_history_entries(self, tmp_path):
        """History entries with missing keys should not raise."""
        reporter = Reporter()
        history = [{"generation": 0}, {
            "generation": 1, "mean_sortino_ratio": 5.0}]
        reporter.plot_phase2_pnl(history, "long", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "phase2_long_pnl.png")


# ---------------------------------------------------------------------------
# Tests: plot_equity_curve
# ---------------------------------------------------------------------------

class TestPlotEquityCurve:
    def test_creates_train_long_png(self, tmp_path):
        reporter = Reporter()
        reporter.plot_equity_curve(
            _make_trade_log(), "train", "long", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "train_long_equity.png")

    def test_creates_validation_short_png(self, tmp_path):
        reporter = Reporter()
        reporter.plot_equity_curve(
            _make_trade_log(), "validation", "short", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "validation_short_equity.png")

    def test_creates_test_long_png(self, tmp_path):
        reporter = Reporter()
        reporter.plot_equity_curve(
            _make_trade_log(), "test", "long", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "test_long_equity.png")

    def test_returns_correct_path(self, tmp_path):
        reporter = Reporter()
        result = reporter.plot_equity_curve(
            _make_trade_log(), "train", "long", output_dir=str(tmp_path)
        )
        expected = os.path.join(str(tmp_path), "train_long_equity.png")
        assert result == expected

    def test_empty_dataframe_creates_file(self, tmp_path):
        reporter = Reporter()
        reporter.plot_equity_curve(
            pd.DataFrame(), "train", "long", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "train_long_equity.png")

    def test_none_trade_log_creates_file(self, tmp_path):
        reporter = Reporter()
        reporter.plot_equity_curve(
            None, "train", "long", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "train_long_equity.png")

    def test_missing_equity_column_creates_file(self, tmp_path):
        reporter = Reporter()
        df = pd.DataFrame({"some_other_col": [1, 2, 3]})
        reporter.plot_equity_curve(
            df, "train", "long", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "train_long_equity.png")

    def test_file_is_nonzero_size(self, tmp_path):
        reporter = Reporter()
        reporter.plot_equity_curve(_make_trade_log(
            50), "train", "long", output_dir=str(tmp_path))
        size = os.path.getsize(tmp_path / "train_long_equity.png")
        assert size > 0

    def test_creates_parent_dirs(self, tmp_path):
        reporter = Reporter()
        nested_dir = str(tmp_path / "x" / "y")
        reporter.plot_equity_curve(
            _make_trade_log(), "test", "short", output_dir=nested_dir)
        assert os.path.exists(os.path.join(
            nested_dir, "test_short_equity.png"))

    def test_single_row_trade_log(self, tmp_path):
        reporter = Reporter()
        df = pd.DataFrame({"Equity_After": [1050.0]})
        reporter.plot_equity_curve(
            df, "train", "long", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "train_long_equity.png")


# ---------------------------------------------------------------------------
# Tests: write_per_symbol_csv
# ---------------------------------------------------------------------------

class TestWritePerSymbolCsv:
    def test_creates_train_csv(self, tmp_path):
        reporter = Reporter()
        reporter.write_per_symbol_csv(
            _make_per_symbol_metrics(), "train", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "train_per_symbol_performance.csv")

    def test_creates_validation_csv(self, tmp_path):
        reporter = Reporter()
        reporter.write_per_symbol_csv(
            _make_per_symbol_metrics(), "validation", output_dir=str(tmp_path))
        assert os.path.exists(
            tmp_path / "validation_per_symbol_performance.csv")

    def test_creates_test_csv(self, tmp_path):
        reporter = Reporter()
        reporter.write_per_symbol_csv(
            _make_per_symbol_metrics(), "test", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "test_per_symbol_performance.csv")

    def test_returns_correct_path(self, tmp_path):
        reporter = Reporter()
        result = reporter.write_per_symbol_csv(
            _make_per_symbol_metrics(), "train", output_dir=str(tmp_path)
        )
        expected = os.path.join(
            str(tmp_path), "train_per_symbol_performance.csv")
        assert result == expected

    def test_csv_has_required_columns(self, tmp_path):
        reporter = Reporter()
        reporter.write_per_symbol_csv(
            _make_per_symbol_metrics(), "train", output_dir=str(tmp_path))
        df = pd.read_csv(tmp_path / "train_per_symbol_performance.csv")
        for col in ("symbol", "trade_count", "win_rate", "net_pnl"):
            assert col in df.columns, f"Missing column: {col}"

    def test_csv_row_count_matches_symbols(self, tmp_path):
        reporter = Reporter()
        symbols = ["SYM_A", "SYM_B", "SYM_C", "SYM_D"]
        reporter.write_per_symbol_csv(
            _make_per_symbol_metrics(symbols), "train", output_dir=str(tmp_path)
        )
        df = pd.read_csv(tmp_path / "train_per_symbol_performance.csv")
        assert len(df) == len(symbols)

    def test_csv_values_match_input(self, tmp_path):
        reporter = Reporter()
        metrics = {
            "per_symbol_metrics": {
                "SYM_X": {"trade_count": 7, "win_rate": 71.4, "net_pnl": 15.5},
            }
        }
        reporter.write_per_symbol_csv(
            metrics, "train", output_dir=str(tmp_path))
        df = pd.read_csv(tmp_path / "train_per_symbol_performance.csv")
        row = df[df["symbol"] == "SYM_X"].iloc[0]
        assert row["trade_count"] == 7
        assert abs(row["win_rate"] - 71.4) < 1e-4
        assert abs(row["net_pnl"] - 15.5) < 1e-4

    def test_empty_per_symbol_metrics_creates_empty_csv(self, tmp_path):
        reporter = Reporter()
        reporter.write_per_symbol_csv(
            {"per_symbol_metrics": {}}, "train", output_dir=str(tmp_path))
        df = pd.read_csv(tmp_path / "train_per_symbol_performance.csv")
        assert len(df) == 0
        for col in ("symbol", "trade_count", "win_rate", "net_pnl"):
            assert col in df.columns

    def test_missing_per_symbol_metrics_key_creates_empty_csv(self, tmp_path):
        reporter = Reporter()
        reporter.write_per_symbol_csv({}, "train", output_dir=str(tmp_path))
        df = pd.read_csv(tmp_path / "train_per_symbol_performance.csv")
        assert len(df) == 0

    def test_creates_parent_dirs(self, tmp_path):
        reporter = Reporter()
        nested_dir = str(tmp_path / "p" / "q")
        reporter.write_per_symbol_csv(
            _make_per_symbol_metrics(), "test", output_dir=nested_dir)
        assert os.path.exists(os.path.join(
            nested_dir, "test_per_symbol_performance.csv"))

    def test_partial_sym_metrics_keys(self, tmp_path):
        """Symbols with missing sub-keys should default to 0."""
        reporter = Reporter()
        metrics = {
            "per_symbol_metrics": {
                "SYM_A": {},  # all keys missing
            }
        }
        reporter.write_per_symbol_csv(
            metrics, "train", output_dir=str(tmp_path))
        df = pd.read_csv(tmp_path / "train_per_symbol_performance.csv")
        assert len(df) == 1
        assert df.iloc[0]["trade_count"] == 0
        assert df.iloc[0]["win_rate"] == 0.0
        assert df.iloc[0]["net_pnl"] == 0.0


# ---------------------------------------------------------------------------
# Tests: plot_rl_curve
# ---------------------------------------------------------------------------

class TestPlotRlCurve:
    def test_creates_long_png(self, tmp_path):
        reporter = Reporter()
        returns = [1.0, 2.0, 3.5, 3.6, 3.7, 3.8]
        reporter.plot_rl_curve(returns, elbow_idx=2,
                               direction="long", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "phase4_long_rl_curve.png")

    def test_creates_short_png(self, tmp_path):
        reporter = Reporter()
        returns = [0.5, 1.0, 1.5, 2.0]
        reporter.plot_rl_curve(returns, elbow_idx=1,
                               direction="short", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "phase4_short_rl_curve.png")

    def test_returns_correct_path(self, tmp_path):
        reporter = Reporter()
        returns = [1.0, 2.0, 3.0]
        result = reporter.plot_rl_curve(
            returns, elbow_idx=1, direction="long", output_dir=str(tmp_path))
        expected = os.path.join(str(tmp_path), "phase4_long_rl_curve.png")
        assert result == expected

    def test_empty_validation_returns_creates_file(self, tmp_path):
        reporter = Reporter()
        reporter.plot_rl_curve(
            [], elbow_idx=0, direction="long", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "phase4_long_rl_curve.png")

    def test_file_is_nonzero_size(self, tmp_path):
        reporter = Reporter()
        returns = list(range(20))
        reporter.plot_rl_curve(returns, elbow_idx=5,
                               direction="long", output_dir=str(tmp_path))
        size = os.path.getsize(tmp_path / "phase4_long_rl_curve.png")
        assert size > 0

    def test_elbow_idx_clamped_when_too_large(self, tmp_path):
        """elbow_idx beyond list length should not raise."""
        reporter = Reporter()
        returns = [1.0, 2.0, 3.0]
        reporter.plot_rl_curve(returns, elbow_idx=999,
                               direction="long", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "phase4_long_rl_curve.png")

    def test_elbow_idx_clamped_when_negative(self, tmp_path):
        """Negative elbow_idx should not raise."""
        reporter = Reporter()
        returns = [1.0, 2.0, 3.0]
        reporter.plot_rl_curve(returns, elbow_idx=-5,
                               direction="long", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "phase4_long_rl_curve.png")

    def test_single_return_value(self, tmp_path):
        reporter = Reporter()
        reporter.plot_rl_curve(
            [5.0], elbow_idx=0, direction="long", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "phase4_long_rl_curve.png")

    def test_creates_parent_dirs(self, tmp_path):
        reporter = Reporter()
        nested_dir = str(tmp_path / "r" / "s" / "t")
        reporter.plot_rl_curve([1.0, 2.0], elbow_idx=0,
                               direction="short", output_dir=nested_dir)
        assert os.path.exists(os.path.join(
            nested_dir, "phase4_short_rl_curve.png"))

    def test_elbow_idx_zero(self, tmp_path):
        reporter = Reporter()
        returns = [3.0, 3.1, 3.2, 3.3]
        reporter.plot_rl_curve(returns, elbow_idx=0,
                               direction="long", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "phase4_long_rl_curve.png")

    def test_elbow_idx_last(self, tmp_path):
        reporter = Reporter()
        returns = [1.0, 2.0, 3.0, 4.0]
        reporter.plot_rl_curve(returns, elbow_idx=3,
                               direction="long", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "phase4_long_rl_curve.png")


# ---------------------------------------------------------------------------
# Helpers for write_strategy_evaluation_table tests
# ---------------------------------------------------------------------------

def _make_eval_rule_set(n_rules: int = 2, conditions_per_rule: int = 3) -> list[dict]:
    """Create a minimal rule set for evaluation table tests."""
    return [
        {
            "conditions": [f"feat_{i}_{j} == High" for j in range(conditions_per_rule)],
            "tp": 2.0,
            "sl": 1.0,
            "capital_pct": 48.0,
        }
        for i in range(n_rules)
    ]


def _make_metrics_by_split(
    win_rate: float = 0.6,
    max_drawdown_pct: float = 5.0,
    total_return_pct: float = 12.0,
    sortino_ratio: float = 1.5,
    profit_factor: float = 1.8,
) -> dict:
    """Create a metrics_by_split dict with all three splits populated."""
    m = {
        "win_rate": win_rate,
        "max_drawdown_pct": max_drawdown_pct,
        "total_return_pct": total_return_pct,
        "sortino_ratio": sortino_ratio,
        "profit_factor": profit_factor,
    }
    return {"train": m.copy(), "validation": m.copy(), "test": m.copy()}


def _make_full_trade_log(n: int = 20, seed: int = 42) -> pd.DataFrame:
    """Create a trade log with all columns needed for sharpe computation."""
    rng = np.random.default_rng(seed)
    equity_before = 1000.0 + np.cumsum(rng.uniform(0, 5, size=n))
    net_pnl = rng.uniform(-10, 20, size=n)
    equity_after = equity_before + net_pnl
    return pd.DataFrame({
        "Net_PnL": net_pnl,
        "Equity_Before_Entry": equity_before,
        "Equity_After": equity_after,
        "Rule_Index": rng.integers(0, 2, size=n),
        "Entry_Index": np.arange(n),
        "Release_Index": np.arange(n) + 5,
    })


def _make_trade_logs_by_split(n: int = 20) -> dict:
    """Create a trade_logs_by_split dict with all three splits populated."""
    return {
        "train": _make_full_trade_log(n, seed=1),
        "validation": _make_full_trade_log(n, seed=2),
        "test": _make_full_trade_log(n, seed=3),
    }


# ---------------------------------------------------------------------------
# Tests: write_strategy_evaluation_table
# ---------------------------------------------------------------------------

class TestWriteStrategyEvaluationTable:
    def test_creates_csv_file(self, tmp_path):
        reporter = Reporter()
        reporter.write_strategy_evaluation_table(
            _make_metrics_by_split(),
            _make_trade_logs_by_split(),
            _make_eval_rule_set(),
            "long",
            output_dir=str(tmp_path),
        )
        assert os.path.exists(tmp_path / "strategy_evaluation_long.csv")

    def test_creates_short_csv_file(self, tmp_path):
        reporter = Reporter()
        reporter.write_strategy_evaluation_table(
            _make_metrics_by_split(),
            _make_trade_logs_by_split(),
            _make_eval_rule_set(),
            "short",
            output_dir=str(tmp_path),
        )
        assert os.path.exists(tmp_path / "strategy_evaluation_short.csv")

    def test_returns_correct_path(self, tmp_path):
        reporter = Reporter()
        result = reporter.write_strategy_evaluation_table(
            _make_metrics_by_split(),
            _make_trade_logs_by_split(),
            _make_eval_rule_set(),
            "long",
            output_dir=str(tmp_path),
        )
        expected = os.path.abspath(
            os.path.join(str(tmp_path), "strategy_evaluation_long.csv")
        )
        assert result == expected

    def test_csv_has_required_columns(self, tmp_path):
        reporter = Reporter()
        reporter.write_strategy_evaluation_table(
            _make_metrics_by_split(),
            _make_trade_logs_by_split(),
            _make_eval_rule_set(),
            "long",
            output_dir=str(tmp_path),
        )
        df = pd.read_csv(tmp_path / "strategy_evaluation_long.csv")
        expected_cols = {
            "split", "win_rate", "mdd_pct", "total_return_pct",
            "num_rules", "num_conditions", "sortino_ratio",
            "profit_factor", "sharpe_ratio",
        }
        assert set(df.columns) == expected_cols

    def test_csv_has_three_rows(self, tmp_path):
        reporter = Reporter()
        reporter.write_strategy_evaluation_table(
            _make_metrics_by_split(),
            _make_trade_logs_by_split(),
            _make_eval_rule_set(),
            "long",
            output_dir=str(tmp_path),
        )
        df = pd.read_csv(tmp_path / "strategy_evaluation_long.csv")
        assert len(df) == 3

    def test_csv_split_values(self, tmp_path):
        reporter = Reporter()
        reporter.write_strategy_evaluation_table(
            _make_metrics_by_split(),
            _make_trade_logs_by_split(),
            _make_eval_rule_set(),
            "long",
            output_dir=str(tmp_path),
        )
        df = pd.read_csv(tmp_path / "strategy_evaluation_long.csv")
        assert list(df["split"]) == ["train", "validation", "test"]

    def test_num_rules_matches_rule_set_length(self, tmp_path):
        reporter = Reporter()
        rule_set = _make_eval_rule_set(n_rules=3)
        reporter.write_strategy_evaluation_table(
            _make_metrics_by_split(),
            _make_trade_logs_by_split(),
            rule_set,
            "long",
            output_dir=str(tmp_path),
        )
        df = pd.read_csv(tmp_path / "strategy_evaluation_long.csv")
        assert (df["num_rules"] == 3).all()

    def test_num_conditions_matches_sum(self, tmp_path):
        reporter = Reporter()
        # 2 rules × 4 conditions each = 8 total
        rule_set = _make_eval_rule_set(n_rules=2, conditions_per_rule=4)
        reporter.write_strategy_evaluation_table(
            _make_metrics_by_split(),
            _make_trade_logs_by_split(),
            rule_set,
            "long",
            output_dir=str(tmp_path),
        )
        df = pd.read_csv(tmp_path / "strategy_evaluation_long.csv")
        assert (df["num_conditions"] == 8).all()

    def test_sharpe_zero_for_single_trade(self, tmp_path):
        reporter = Reporter()
        single_trade_log = pd.DataFrame({
            "Net_PnL": [10.0],
            "Equity_Before_Entry": [1000.0],
            "Equity_After": [1010.0],
        })
        logs = {"train": single_trade_log, "validation": None, "test": None}
        reporter.write_strategy_evaluation_table(
            _make_metrics_by_split(),
            logs,
            _make_eval_rule_set(),
            "long",
            output_dir=str(tmp_path),
        )
        df = pd.read_csv(tmp_path / "strategy_evaluation_long.csv")
        train_row = df[df["split"] == "train"].iloc[0]
        assert train_row["sharpe_ratio"] == 0.0

    def test_sharpe_zero_for_none_trade_log(self, tmp_path):
        reporter = Reporter()
        logs = {"train": None, "validation": None, "test": None}
        reporter.write_strategy_evaluation_table(
            _make_metrics_by_split(),
            logs,
            _make_eval_rule_set(),
            "long",
            output_dir=str(tmp_path),
        )
        df = pd.read_csv(tmp_path / "strategy_evaluation_long.csv")
        assert (df["sharpe_ratio"] == 0.0).all()

    def test_invalid_direction_raises(self, tmp_path):
        reporter = Reporter()
        with pytest.raises(ValueError):
            reporter.write_strategy_evaluation_table(
                _make_metrics_by_split(),
                _make_trade_logs_by_split(),
                _make_eval_rule_set(),
                "sideways",
                output_dir=str(tmp_path),
            )

    def test_invalid_direction_raises_before_file_created(self, tmp_path):
        reporter = Reporter()
        with pytest.raises(ValueError):
            reporter.write_strategy_evaluation_table(
                _make_metrics_by_split(),
                _make_trade_logs_by_split(),
                _make_eval_rule_set(),
                "up",
                output_dir=str(tmp_path),
            )
        # No CSV should have been created
        assert not any(tmp_path.iterdir())

    def test_missing_metrics_defaults_to_zero(self, tmp_path):
        reporter = Reporter()
        reporter.write_strategy_evaluation_table(
            {"train": None, "validation": None, "test": None},
            _make_trade_logs_by_split(),
            _make_eval_rule_set(),
            "long",
            output_dir=str(tmp_path),
        )
        df = pd.read_csv(tmp_path / "strategy_evaluation_long.csv")
        assert (df["win_rate"] == 0.0).all()
        assert (df["mdd_pct"] == 0.0).all()
        assert (df["total_return_pct"] == 0.0).all()
        assert (df["sortino_ratio"] == 0.0).all()
        assert (df["profit_factor"] == 0.0).all()

    def test_empty_rule_set(self, tmp_path):
        reporter = Reporter()
        reporter.write_strategy_evaluation_table(
            _make_metrics_by_split(),
            _make_trade_logs_by_split(),
            [],
            "long",
            output_dir=str(tmp_path),
        )
        df = pd.read_csv(tmp_path / "strategy_evaluation_long.csv")
        assert (df["num_rules"] == 0).all()
        assert (df["num_conditions"] == 0).all()

    def test_sharpe_computed_correctly(self, tmp_path):
        """Sharpe ratio should equal mean(r)/std(r, ddof=1) for a known log."""
        reporter = Reporter()
        net_pnl = np.array([10.0, -5.0, 8.0, 12.0, -3.0])
        equity_before = np.array([1000.0, 1010.0, 1005.0, 1013.0, 1025.0])
        log = pd.DataFrame({
            "Net_PnL": net_pnl,
            "Equity_Before_Entry": equity_before,
        })
        logs = {"train": log, "validation": None, "test": None}
        reporter.write_strategy_evaluation_table(
            _make_metrics_by_split(),
            logs,
            _make_eval_rule_set(),
            "long",
            output_dir=str(tmp_path),
        )
        df = pd.read_csv(tmp_path / "strategy_evaluation_long.csv")
        r = net_pnl / equity_before
        expected_sharpe = r.mean() / r.std(ddof=1)
        actual_sharpe = df[df["split"] == "train"].iloc[0]["sharpe_ratio"]
        assert abs(actual_sharpe - expected_sharpe) < 1e-9

    def test_metrics_sourced_from_metrics_dict(self, tmp_path):
        """win_rate, mdd_pct etc. should come from metrics_by_split."""
        reporter = Reporter()
        metrics = {
            "train": {
                "win_rate": 0.75,
                "max_drawdown_pct": 8.5,
                "total_return_pct": 20.0,
                "sortino_ratio": 2.1,
                "profit_factor": 2.5,
            },
            "validation": None,
            "test": None,
        }
        reporter.write_strategy_evaluation_table(
            metrics,
            _make_trade_logs_by_split(),
            _make_eval_rule_set(),
            "long",
            output_dir=str(tmp_path),
        )
        df = pd.read_csv(tmp_path / "strategy_evaluation_long.csv")
        train_row = df[df["split"] == "train"].iloc[0]
        assert abs(train_row["win_rate"] - 0.75) < 1e-9
        assert abs(train_row["mdd_pct"] - 8.5) < 1e-9
        assert abs(train_row["total_return_pct"] - 20.0) < 1e-9
        assert abs(train_row["sortino_ratio"] - 2.1) < 1e-9
        assert abs(train_row["profit_factor"] - 2.5) < 1e-9


# ---------------------------------------------------------------------------
# Helpers for plot_per_rule_breakdown
# ---------------------------------------------------------------------------

def _make_rule_set(n: int = 3) -> list:
    """Create a minimal rule set with *n* rules."""
    return [
        {
            "conditions": [f"feat_{i} == High", f"feat_{i} == Low"],
            "tp": 2.0,
            "sl": 1.0,
            "capital_pct": 48.0,
        }
        for i in range(n)
    ]


def _make_full_trade_log(n_rules: int = 3, trades_per_rule: int = 5, seed: int = 0) -> pd.DataFrame:
    """Create a trade log with Rule_Index, Net_PnL, Equity_After columns."""
    rng = np.random.default_rng(seed)
    rows = []
    equity = 1000.0
    for rule_idx in range(1, n_rules + 1):
        for _ in range(trades_per_rule):
            pnl = float(rng.uniform(-20, 30))
            equity += pnl
            rows.append({
                "Rule_Index": rule_idx,
                "Net_PnL": pnl,
                "Equity_After": equity,
                "Equity_Before_Entry": equity - pnl,
                "Entry_Index": 0,
                "Release_Index": 1,
            })
    return pd.DataFrame(rows)


def _make_split_logs(n_rules: int = 3) -> dict:
    """Create trade_logs_by_split dict with all three splits populated."""
    return {
        "train": _make_full_trade_log(n_rules=n_rules, trades_per_rule=5, seed=1),
        "validation": _make_full_trade_log(n_rules=n_rules, trades_per_rule=3, seed=2),
        "test": _make_full_trade_log(n_rules=n_rules, trades_per_rule=4, seed=3),
    }


# ---------------------------------------------------------------------------
# Tests: plot_per_rule_breakdown
# ---------------------------------------------------------------------------

class TestPlotPerRuleBreakdown:
    def test_creates_png_file(self, tmp_path):
        reporter = Reporter()
        reporter.plot_per_rule_breakdown(
            _make_rule_set(3),
            _make_split_logs(3),
            "long",
            output_dir=str(tmp_path),
        )
        assert os.path.exists(tmp_path / "per_rule_breakdown_long.png")

    def test_creates_short_png_file(self, tmp_path):
        reporter = Reporter()
        reporter.plot_per_rule_breakdown(
            _make_rule_set(2),
            _make_split_logs(2),
            "short",
            output_dir=str(tmp_path),
        )
        assert os.path.exists(tmp_path / "per_rule_breakdown_short.png")

    def test_returns_correct_path(self, tmp_path):
        reporter = Reporter()
        result = reporter.plot_per_rule_breakdown(
            _make_rule_set(2),
            _make_split_logs(2),
            "long",
            output_dir=str(tmp_path),
        )
        expected = os.path.abspath(
            os.path.join(str(tmp_path), "per_rule_breakdown_long.png")
        )
        assert result == expected

    def test_invalid_direction_raises(self, tmp_path):
        reporter = Reporter()
        with pytest.raises(ValueError):
            reporter.plot_per_rule_breakdown(
                _make_rule_set(2),
                _make_split_logs(2),
                "sideways",
                output_dir=str(tmp_path),
            )

    def test_invalid_direction_does_not_create_file(self, tmp_path):
        reporter = Reporter()
        with pytest.raises(ValueError):
            reporter.plot_per_rule_breakdown(
                _make_rule_set(2),
                _make_split_logs(2),
                "sideways",
                output_dir=str(tmp_path),
            )
        # No PNG should have been created
        assert not any(tmp_path.iterdir())

    def test_none_trade_log_does_not_raise(self, tmp_path):
        reporter = Reporter()
        logs = {"train": None, "validation": None, "test": None}
        # Should not raise
        result = reporter.plot_per_rule_breakdown(
            _make_rule_set(2),
            logs,
            "long",
            output_dir=str(tmp_path),
        )
        assert os.path.exists(result)

    def test_empty_trade_log_does_not_raise(self, tmp_path):
        reporter = Reporter()
        logs = {
            "train": pd.DataFrame(),
            "validation": pd.DataFrame(),
            "test": pd.DataFrame(),
        }
        result = reporter.plot_per_rule_breakdown(
            _make_rule_set(2),
            logs,
            "long",
            output_dir=str(tmp_path),
        )
        assert os.path.exists(result)

    def test_mixed_none_and_valid_logs(self, tmp_path):
        """One split None, others valid — should not raise."""
        reporter = Reporter()
        logs = {
            "train": _make_full_trade_log(n_rules=2, trades_per_rule=5),
            "validation": None,
            "test": _make_full_trade_log(n_rules=2, trades_per_rule=3),
        }
        result = reporter.plot_per_rule_breakdown(
            _make_rule_set(2),
            logs,
            "long",
            output_dir=str(tmp_path),
        )
        assert os.path.exists(result)

    def test_zero_rule_trades_renders_zero_bar(self, tmp_path):
        """A rule that has no trades in any split should produce 0.0 metrics."""
        reporter = Reporter()
        # Rule set has 3 rules but trade log only has trades for rule 1
        log = pd.DataFrame([
            {"Rule_Index": 1, "Net_PnL": 10.0, "Equity_After": 1010.0,
             "Equity_Before_Entry": 1000.0, "Entry_Index": 0, "Release_Index": 1},
        ])
        logs = {"train": log, "validation": log, "test": log}
        # Should not raise even though rules 2 and 3 have zero trades
        result = reporter.plot_per_rule_breakdown(
            _make_rule_set(3),
            logs,
            "long",
            output_dir=str(tmp_path),
        )
        assert os.path.exists(result)

    def test_file_is_nonzero_size(self, tmp_path):
        reporter = Reporter()
        reporter.plot_per_rule_breakdown(
            _make_rule_set(3),
            _make_split_logs(3),
            "long",
            output_dir=str(tmp_path),
        )
        size = os.path.getsize(tmp_path / "per_rule_breakdown_long.png")
        assert size > 0

    def test_creates_parent_dirs(self, tmp_path):
        reporter = Reporter()
        nested_dir = str(tmp_path / "a" / "b" / "c")
        reporter.plot_per_rule_breakdown(
            _make_rule_set(2),
            _make_split_logs(2),
            "long",
            output_dir=nested_dir,
        )
        assert os.path.exists(
            os.path.join(nested_dir, "per_rule_breakdown_long.png")
        )

    def test_empty_rule_set_does_not_raise(self, tmp_path):
        """An empty rule set should produce a figure with no bars."""
        reporter = Reporter()
        result = reporter.plot_per_rule_breakdown(
            [],
            _make_split_logs(0),
            "long",
            output_dir=str(tmp_path),
        )
        assert os.path.exists(result)

    def test_single_rule(self, tmp_path):
        reporter = Reporter()
        result = reporter.plot_per_rule_breakdown(
            _make_rule_set(1),
            _make_split_logs(1),
            "short",
            output_dir=str(tmp_path),
        )
        assert os.path.exists(result)
        assert os.path.getsize(result) > 0


# ---------------------------------------------------------------------------
# Helpers for Spearman tests
# ---------------------------------------------------------------------------

def _make_dataset_with_label(
    n: int = 50,
    feature_names: list[str] | None = None,
    seed: int = 0,
    include_label: bool = True,
) -> pd.DataFrame:
    """Create a dataset DataFrame with feature columns and label_close_288."""
    rng = np.random.default_rng(seed)
    if feature_names is None:
        feature_names = ["feat_a", "feat_b"]
    data = {name: rng.uniform(-1, 1, size=n) for name in feature_names}
    if include_label:
        data["label_close_288"] = rng.uniform(-5, 5, size=n)
    return pd.DataFrame(data)


def _make_datasets_by_split(
    feature_names: list[str] | None = None,
    n: int = 50,
) -> dict:
    """Create datasets_by_split dict with all three splits."""
    return {
        "train": _make_dataset_with_label(n=n, feature_names=feature_names, seed=1),
        "validation": _make_dataset_with_label(n=n, feature_names=feature_names, seed=2),
        "test": _make_dataset_with_label(n=n, feature_names=feature_names, seed=3),
    }


def _make_selected_features(names: list[str] | None = None) -> list[dict]:
    """Create a selected_features list."""
    if names is None:
        names = ["feat_a", "feat_b"]
    return [{"name": name, "mode": "positive", "score": 1.0} for name in names]


# ---------------------------------------------------------------------------
# Tests: write_spearman_correlation_report
# ---------------------------------------------------------------------------

class TestWriteSpearmanCorrelationReport:
    def test_creates_csv_file(self, tmp_path):
        reporter = Reporter()
        reporter.write_spearman_correlation_report(
            _make_datasets_by_split(),
            _make_selected_features(),
            "long",
            output_dir=str(tmp_path),
        )
        assert os.path.exists(tmp_path / "spearman_correlation_long.csv")

    def test_creates_short_csv_file(self, tmp_path):
        reporter = Reporter()
        reporter.write_spearman_correlation_report(
            _make_datasets_by_split(),
            _make_selected_features(),
            "short",
            output_dir=str(tmp_path),
        )
        assert os.path.exists(tmp_path / "spearman_correlation_short.csv")

    def test_returns_correct_path(self, tmp_path):
        reporter = Reporter()
        result = reporter.write_spearman_correlation_report(
            _make_datasets_by_split(),
            _make_selected_features(),
            "long",
            output_dir=str(tmp_path),
        )
        expected = os.path.abspath(
            os.path.join(str(tmp_path), "spearman_correlation_long.csv")
        )
        assert result == expected

    def test_returns_absolute_path(self, tmp_path):
        reporter = Reporter()
        result = reporter.write_spearman_correlation_report(
            _make_datasets_by_split(),
            _make_selected_features(),
            "long",
            output_dir=str(tmp_path),
        )
        assert os.path.isabs(result)

    def test_csv_has_required_columns(self, tmp_path):
        reporter = Reporter()
        result = reporter.write_spearman_correlation_report(
            _make_datasets_by_split(),
            _make_selected_features(),
            "long",
            output_dir=str(tmp_path),
        )
        df = pd.read_csv(result)
        assert set(df.columns) == {
            "feature", "train_spearman", "validation_spearman", "test_spearman"
        }

    def test_one_row_per_feature(self, tmp_path):
        reporter = Reporter()
        features = _make_selected_features(["feat_a", "feat_b", "feat_c"])
        result = reporter.write_spearman_correlation_report(
            _make_datasets_by_split(["feat_a", "feat_b", "feat_c"]),
            features,
            "long",
            output_dir=str(tmp_path),
        )
        df = pd.read_csv(result)
        assert len(df) == 3
        assert set(df["feature"]) == {"feat_a", "feat_b", "feat_c"}

    def test_absent_feature_column_records_nan(self, tmp_path):
        """Feature not in dataset → NaN for that split."""
        reporter = Reporter()
        # Dataset only has feat_a, but we request feat_a and feat_missing
        datasets = _make_datasets_by_split(["feat_a"])
        features = _make_selected_features(["feat_a", "feat_missing"])
        result = reporter.write_spearman_correlation_report(
            datasets,
            features,
            "long",
            output_dir=str(tmp_path),
        )
        df = pd.read_csv(result)
        missing_row = df[df["feature"] == "feat_missing"].iloc[0]
        assert pd.isna(missing_row["train_spearman"])
        assert pd.isna(missing_row["validation_spearman"])
        assert pd.isna(missing_row["test_spearman"])

    def test_absent_label_column_records_nan(self, tmp_path):
        """Dataset without label_close_288 → NaN for all features on that split."""
        reporter = Reporter()
        datasets = {
            "train": _make_dataset_with_label(include_label=False),
            "validation": _make_dataset_with_label(include_label=True, seed=2),
            "test": _make_dataset_with_label(include_label=True, seed=3),
        }
        features = _make_selected_features(["feat_a"])
        result = reporter.write_spearman_correlation_report(
            datasets,
            features,
            "long",
            output_dir=str(tmp_path),
        )
        df = pd.read_csv(result)
        row = df[df["feature"] == "feat_a"].iloc[0]
        assert pd.isna(row["train_spearman"])
        # validation and test should have valid values
        assert not pd.isna(row["validation_spearman"])
        assert not pd.isna(row["test_spearman"])

    def test_fewer_than_two_rows_records_nan(self, tmp_path):
        """Dataset with only 1 non-NaN paired row → NaN."""
        reporter = Reporter()
        # Single-row dataset
        single_row = pd.DataFrame({
            "feat_a": [0.5],
            "label_close_288": [1.0],
        })
        datasets = {
            "train": single_row,
            "validation": single_row,
            "test": single_row,
        }
        features = _make_selected_features(["feat_a"])
        result = reporter.write_spearman_correlation_report(
            datasets,
            features,
            "long",
            output_dir=str(tmp_path),
        )
        df = pd.read_csv(result)
        row = df[df["feature"] == "feat_a"].iloc[0]
        assert pd.isna(row["train_spearman"])
        assert pd.isna(row["validation_spearman"])
        assert pd.isna(row["test_spearman"])

    def test_sorted_by_abs_train_spearman(self, tmp_path):
        """Rows must be sorted by abs(train_spearman) descending."""
        reporter = Reporter()
        # Create datasets where feat_a has high correlation and feat_b has low
        rng = np.random.default_rng(99)
        n = 100
        label = rng.uniform(-5, 5, size=n)
        # feat_a: strongly correlated with label
        feat_a = label + rng.uniform(-0.1, 0.1, size=n)
        # feat_b: weakly correlated (random)
        feat_b = rng.uniform(-1, 1, size=n)
        df_data = pd.DataFrame({
            "feat_a": feat_a,
            "feat_b": feat_b,
            "label_close_288": label,
        })
        datasets = {"train": df_data, "validation": df_data, "test": df_data}
        features = _make_selected_features(["feat_a", "feat_b"])
        result = reporter.write_spearman_correlation_report(
            datasets,
            features,
            "long",
            output_dir=str(tmp_path),
        )
        df = pd.read_csv(result)
        abs_train = df["train_spearman"].abs().values
        # Each value should be >= the next (non-increasing)
        for i in range(len(abs_train) - 1):
            assert abs_train[i] >= abs_train[i + 1], (
                f"Row {i} abs={abs_train[i]} < row {i+1} abs={abs_train[i+1]}"
            )

    def test_invalid_direction_raises(self, tmp_path):
        reporter = Reporter()
        with pytest.raises(ValueError):
            reporter.write_spearman_correlation_report(
                _make_datasets_by_split(),
                _make_selected_features(),
                "sideways",
                output_dir=str(tmp_path),
            )

    def test_invalid_direction_does_not_create_file(self, tmp_path):
        reporter = Reporter()
        with pytest.raises(ValueError):
            reporter.write_spearman_correlation_report(
                _make_datasets_by_split(),
                _make_selected_features(),
                "sideways",
                output_dir=str(tmp_path),
            )
        assert not any(tmp_path.iterdir())

    def test_none_dataset_records_nan(self, tmp_path):
        """None dataset for a split → NaN for all features on that split."""
        reporter = Reporter()
        datasets = {
            "train": None,
            "validation": _make_dataset_with_label(seed=2),
            "test": _make_dataset_with_label(seed=3),
        }
        features = _make_selected_features(["feat_a"])
        result = reporter.write_spearman_correlation_report(
            datasets,
            features,
            "long",
            output_dir=str(tmp_path),
        )
        df = pd.read_csv(result)
        row = df[df["feature"] == "feat_a"].iloc[0]
        assert pd.isna(row["train_spearman"])
        assert not pd.isna(row["validation_spearman"])

    def test_empty_features_list_produces_empty_csv(self, tmp_path):
        """Empty selected_features → CSV with header only."""
        reporter = Reporter()
        result = reporter.write_spearman_correlation_report(
            _make_datasets_by_split(),
            [],
            "long",
            output_dir=str(tmp_path),
        )
        df = pd.read_csv(result)
        assert len(df) == 0
        assert list(df.columns) == [
            "feature", "train_spearman", "validation_spearman", "test_spearman"
        ]

    def test_spearman_values_in_valid_range(self, tmp_path):
        """All non-NaN Spearman values must be in [-1.0, 1.0]."""
        reporter = Reporter()
        result = reporter.write_spearman_correlation_report(
            _make_datasets_by_split(["feat_a", "feat_b"]),
            _make_selected_features(["feat_a", "feat_b"]),
            "long",
            output_dir=str(tmp_path),
        )
        df = pd.read_csv(result)
        for col in ["train_spearman", "validation_spearman", "test_spearman"]:
            valid = df[col].dropna()
            assert (valid >= -1.0).all() and (valid <= 1.0).all(), (
                f"Column {col} has values outside [-1, 1]: {valid.values}"
            )


# ---------------------------------------------------------------------------
# Helpers for plot_distribution_and_equity tests
# ---------------------------------------------------------------------------

def _make_dist_trade_log(n: int = 20, seed: int = 42) -> pd.DataFrame:
    """Create a trade log with all columns needed for distribution/equity plots."""
    rng = np.random.default_rng(seed)
    entry_indices = np.sort(rng.integers(0, 200, size=n))
    release_indices = entry_indices + rng.integers(1, 20, size=n)
    net_pnl = rng.uniform(-15, 25, size=n)
    equity = 1000.0 + np.cumsum(net_pnl)
    return pd.DataFrame({
        "Entry_Index": entry_indices,
        "Release_Index": release_indices,
        "Net_PnL": net_pnl,
        "Equity_After": equity,
        "Equity_Before_Entry": equity - net_pnl,
        "Rule_Index": rng.integers(1, 3, size=n),
    })


def _make_dist_logs_by_split(n: int = 20) -> dict:
    """Create trade_logs_by_split dict with all three splits populated."""
    return {
        "train": _make_dist_trade_log(n=n, seed=1),
        "validation": _make_dist_trade_log(n=n, seed=2),
        "test": _make_dist_trade_log(n=n, seed=3),
    }


# ---------------------------------------------------------------------------
# Tests: plot_distribution_and_equity
# ---------------------------------------------------------------------------

class TestPlotDistributionAndEquity:
    def test_creates_png_per_split(self, tmp_path):
        reporter = Reporter()
        reporter.plot_distribution_and_equity(
            _make_dist_logs_by_split(),
            "long",
            output_dir=str(tmp_path),
        )
        for split in ("train", "validation", "test"):
            assert os.path.exists(
                tmp_path / f"distribution_equity_{split}_long.png"
            ), f"Missing PNG for split={split}"

    def test_returns_list_of_paths(self, tmp_path):
        reporter = Reporter()
        result = reporter.plot_distribution_and_equity(
            _make_dist_logs_by_split(),
            "long",
            output_dir=str(tmp_path),
        )
        assert isinstance(result, list)
        assert len(result) == 3

    def test_returns_absolute_paths(self, tmp_path):
        reporter = Reporter()
        result = reporter.plot_distribution_and_equity(
            _make_dist_logs_by_split(),
            "long",
            output_dir=str(tmp_path),
        )
        for path in result:
            assert os.path.isabs(path), f"Path is not absolute: {path}"

    def test_none_split_skipped(self, tmp_path):
        reporter = Reporter()
        logs = {
            "train": _make_dist_trade_log(seed=1),
            "validation": None,
            "test": _make_dist_trade_log(seed=3),
        }
        result = reporter.plot_distribution_and_equity(
            logs,
            "long",
            output_dir=str(tmp_path),
        )
        assert len(result) == 2
        assert not os.path.exists(
            tmp_path / "distribution_equity_validation_long.png"
        )

    def test_empty_split_skipped(self, tmp_path):
        reporter = Reporter()
        logs = {
            "train": _make_dist_trade_log(seed=1),
            "validation": pd.DataFrame(),
            "test": _make_dist_trade_log(seed=3),
        }
        result = reporter.plot_distribution_and_equity(
            logs,
            "long",
            output_dir=str(tmp_path),
        )
        assert len(result) == 2
        assert not os.path.exists(
            tmp_path / "distribution_equity_validation_long.png"
        )

    def test_all_splits_none_returns_empty_list(self, tmp_path):
        reporter = Reporter()
        logs = {"train": None, "validation": None, "test": None}
        result = reporter.plot_distribution_and_equity(
            logs,
            "long",
            output_dir=str(tmp_path),
        )
        assert result == []

    def test_return_list_length_matches_nonempty_splits(self, tmp_path):
        reporter = Reporter()
        logs = {
            "train": _make_dist_trade_log(seed=1),
            "validation": None,
            "test": None,
        }
        result = reporter.plot_distribution_and_equity(
            logs,
            "short",
            output_dir=str(tmp_path),
        )
        assert len(result) == 1

    def test_file_is_nonzero_size(self, tmp_path):
        reporter = Reporter()
        result = reporter.plot_distribution_and_equity(
            _make_dist_logs_by_split(n=30),
            "long",
            output_dir=str(tmp_path),
        )
        for path in result:
            assert os.path.getsize(path) > 0, f"File is empty: {path}"

    def test_invalid_direction_raises(self, tmp_path):
        reporter = Reporter()
        with pytest.raises(ValueError):
            reporter.plot_distribution_and_equity(
                _make_dist_logs_by_split(),
                "sideways",
                output_dir=str(tmp_path),
            )

    def test_invalid_direction_does_not_create_file(self, tmp_path):
        reporter = Reporter()
        with pytest.raises(ValueError):
            reporter.plot_distribution_and_equity(
                _make_dist_logs_by_split(),
                "up",
                output_dir=str(tmp_path),
            )
        assert not any(tmp_path.iterdir())

    def test_short_direction_creates_correct_filenames(self, tmp_path):
        reporter = Reporter()
        result = reporter.plot_distribution_and_equity(
            _make_dist_logs_by_split(),
            "short",
            output_dir=str(tmp_path),
        )
        for path in result:
            assert "_short.png" in path

    def test_paths_point_to_existing_files(self, tmp_path):
        reporter = Reporter()
        result = reporter.plot_distribution_and_equity(
            _make_dist_logs_by_split(),
            "long",
            output_dir=str(tmp_path),
        )
        for path in result:
            assert os.path.exists(path), f"File does not exist: {path}"

    def test_single_trade_log_does_not_raise(self, tmp_path):
        """A trade log with a single trade should not raise."""
        reporter = Reporter()
        single = pd.DataFrame({
            "Entry_Index": [5],
            "Release_Index": [10],
            "Net_PnL": [15.0],
            "Equity_After": [1015.0],
            "Equity_Before_Entry": [1000.0],
            "Rule_Index": [1],
        })
        logs = {"train": single, "validation": None, "test": None}
        result = reporter.plot_distribution_and_equity(
            logs,
            "long",
            output_dir=str(tmp_path),
        )
        assert len(result) == 1
        assert os.path.exists(result[0])


# ---------------------------------------------------------------------------
# Helpers for write_feature_stratified_performance tests
# ---------------------------------------------------------------------------

def _make_stratified_dataset(
    n: int = 30,
    feature_names: list[str] | None = None,
    fuzzy_values: list[str] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Create a dataset with fuzzy-valued feature columns."""
    rng = np.random.default_rng(seed)
    if feature_names is None:
        feature_names = ["feat_a"]
    if fuzzy_values is None:
        fuzzy_values = ["High", "Low", "Medium"]
    data = {}
    for name in feature_names:
        data[name] = rng.choice(fuzzy_values, size=n)
    data["label_close_288"] = rng.uniform(-5, 5, size=n)
    return pd.DataFrame(data)


def _make_stratified_trade_log(
    n: int = 20,
    dataset_len: int = 30,
    seed: int = 0,
) -> pd.DataFrame:
    """Create a trade log with Entry_Index values within dataset bounds."""
    rng = np.random.default_rng(seed)
    entry_indices = rng.integers(0, dataset_len, size=n)
    net_pnl = rng.uniform(-15, 25, size=n)
    equity = 1000.0 + np.cumsum(net_pnl)
    return pd.DataFrame({
        "Entry_Index": entry_indices,
        "Release_Index": entry_indices + rng.integers(1, 10, size=n),
        "Net_PnL": net_pnl,
        "Equity_After": equity,
        "Equity_Before_Entry": equity - net_pnl,
        "Rule_Index": rng.integers(1, 3, size=n),
    })


def _make_stratified_datasets_by_split(
    n: int = 30,
    feature_names: list[str] | None = None,
    fuzzy_values: list[str] | None = None,
) -> dict:
    return {
        "train": _make_stratified_dataset(n=n, feature_names=feature_names,
                                          fuzzy_values=fuzzy_values, seed=1),
        "validation": _make_stratified_dataset(n=n, feature_names=feature_names,
                                               fuzzy_values=fuzzy_values, seed=2),
        "test": _make_stratified_dataset(n=n, feature_names=feature_names,
                                         fuzzy_values=fuzzy_values, seed=3),
    }


def _make_stratified_logs_by_split(n: int = 20, dataset_len: int = 30) -> dict:
    return {
        "train": _make_stratified_trade_log(n=n, dataset_len=dataset_len, seed=1),
        "validation": _make_stratified_trade_log(n=n, dataset_len=dataset_len, seed=2),
        "test": _make_stratified_trade_log(n=n, dataset_len=dataset_len, seed=3),
    }


# ---------------------------------------------------------------------------
# Tests: write_feature_stratified_performance
# ---------------------------------------------------------------------------

class TestWriteFeatureStratifiedPerformance:
    _REQUIRED_COLS = {
        "feature", "fuzzy_value", "split", "num_trades",
        "total_return_pct", "win_rate", "sharpe_ratio",
    }

    def test_creates_csv_per_split(self, tmp_path):
        reporter = Reporter()
        reporter.write_feature_stratified_performance(
            _make_stratified_logs_by_split(),
            [],
            _make_selected_features(["feat_a"]),
            _make_stratified_datasets_by_split(feature_names=["feat_a"]),
            "long",
            output_dir=str(tmp_path),
        )
        for split in ("train", "validation", "test"):
            assert os.path.exists(
                tmp_path / f"feature_stratified_{split}_long.csv"
            ), f"Missing CSV for split={split}"

    def test_returns_list_of_paths(self, tmp_path):
        reporter = Reporter()
        result = reporter.write_feature_stratified_performance(
            _make_stratified_logs_by_split(),
            [],
            _make_selected_features(["feat_a"]),
            _make_stratified_datasets_by_split(feature_names=["feat_a"]),
            "long",
            output_dir=str(tmp_path),
        )
        assert isinstance(result, list)
        assert len(result) == 3

    def test_returns_absolute_paths(self, tmp_path):
        reporter = Reporter()
        result = reporter.write_feature_stratified_performance(
            _make_stratified_logs_by_split(),
            [],
            _make_selected_features(["feat_a"]),
            _make_stratified_datasets_by_split(feature_names=["feat_a"]),
            "long",
            output_dir=str(tmp_path),
        )
        for path in result:
            assert os.path.isabs(path), f"Path is not absolute: {path}"

    def test_csv_has_required_columns(self, tmp_path):
        reporter = Reporter()
        result = reporter.write_feature_stratified_performance(
            _make_stratified_logs_by_split(),
            [],
            _make_selected_features(["feat_a"]),
            _make_stratified_datasets_by_split(feature_names=["feat_a"]),
            "long",
            output_dir=str(tmp_path),
        )
        for path in result:
            df = pd.read_csv(path)
            assert set(df.columns) == self._REQUIRED_COLS, (
                f"Columns mismatch in {path}: {set(df.columns)}"
            )

    def test_invalid_direction_raises(self, tmp_path):
        reporter = Reporter()
        with pytest.raises(ValueError):
            reporter.write_feature_stratified_performance(
                _make_stratified_logs_by_split(),
                [],
                _make_selected_features(["feat_a"]),
                _make_stratified_datasets_by_split(feature_names=["feat_a"]),
                "sideways",
                output_dir=str(tmp_path),
            )

    def test_invalid_direction_does_not_create_file(self, tmp_path):
        reporter = Reporter()
        with pytest.raises(ValueError):
            reporter.write_feature_stratified_performance(
                _make_stratified_logs_by_split(),
                [],
                _make_selected_features(["feat_a"]),
                _make_stratified_datasets_by_split(feature_names=["feat_a"]),
                "up",
                output_dir=str(tmp_path),
            )
        assert not any(tmp_path.iterdir())

    def test_absent_feature_column_skipped(self, tmp_path):
        """Feature not in dataset → that feature is skipped, no error raised."""
        reporter = Reporter()
        # Dataset only has feat_a, but we request feat_a and feat_missing
        datasets = _make_stratified_datasets_by_split(feature_names=["feat_a"])
        features = _make_selected_features(["feat_a", "feat_missing"])
        result = reporter.write_feature_stratified_performance(
            _make_stratified_logs_by_split(),
            [],
            features,
            datasets,
            "long",
            output_dir=str(tmp_path),
        )
        # Should still produce 3 CSVs
        assert len(result) == 3
        for path in result:
            df = pd.read_csv(path)
            # feat_missing should not appear in any CSV
            assert "feat_missing" not in df["feature"].values

    def test_zero_trade_stratum_row_has_zero_metrics(self, tmp_path):
        """A fuzzy value that no trade maps to should produce a row with all zeros."""
        reporter = Reporter()
        # Dataset has "High", "Low", "Medium" but trade log entries only map to "High"
        n_dataset = 30
        # Build dataset where first 10 rows are "High", rest are "Low"/"Medium"
        feat_vals = ["High"] * 10 + ["Low"] * 10 + ["Medium"] * 10
        dataset = pd.DataFrame({
            "feat_a": feat_vals,
            "label_close_288": np.zeros(n_dataset),
        })
        # Trade log: all Entry_Index values point to rows 0-9 (all "High")
        trade_log = pd.DataFrame({
            "Entry_Index": list(range(10)),
            "Release_Index": list(range(1, 11)),
            "Net_PnL": [5.0] * 10,
            "Equity_After": [1000.0 + 5.0 * (i + 1) for i in range(10)],
            "Equity_Before_Entry": [1000.0 + 5.0 * i for i in range(10)],
            "Rule_Index": [1] * 10,
        })
        datasets = {"train": dataset, "validation": dataset, "test": dataset}
        logs = {"train": trade_log, "validation": trade_log, "test": trade_log}
        result = reporter.write_feature_stratified_performance(
            logs,
            [],
            _make_selected_features(["feat_a"]),
            datasets,
            "long",
            output_dir=str(tmp_path),
        )
        df = pd.read_csv(result[0])  # train CSV
        # "Low" and "Medium" strata should have zero trades and zero metrics
        for zero_val in ("Low", "Medium"):
            row = df[(df["feature"] == "feat_a") & (df["fuzzy_value"] == zero_val)]
            assert len(row) == 1, f"Expected 1 row for fuzzy_value={zero_val}"
            assert row.iloc[0]["num_trades"] == 0
            assert row.iloc[0]["total_return_pct"] == 0.0
            assert row.iloc[0]["win_rate"] == 0.0
            assert row.iloc[0]["sharpe_ratio"] == 0.0

    def test_out_of_bounds_entry_index_skipped(self, tmp_path):
        """Out-of-bounds Entry_Index values should be skipped without error."""
        reporter = Reporter()
        n_dataset = 10
        dataset = pd.DataFrame({
            "feat_a": ["High"] * n_dataset,
            "label_close_288": np.zeros(n_dataset),
        })
        # Mix of valid (0-9) and out-of-bounds (100, 200) Entry_Index values
        trade_log = pd.DataFrame({
            "Entry_Index": [0, 1, 2, 100, 200],
            "Release_Index": [1, 2, 3, 101, 201],
            "Net_PnL": [5.0, -3.0, 8.0, 10.0, 12.0],
            "Equity_After": [1005.0, 1002.0, 1010.0, 1020.0, 1032.0],
            "Equity_Before_Entry": [1000.0, 1005.0, 1002.0, 1010.0, 1020.0],
            "Rule_Index": [1, 1, 1, 1, 1],
        })
        datasets = {"train": dataset, "validation": dataset, "test": dataset}
        logs = {"train": trade_log, "validation": trade_log, "test": trade_log}
        result = reporter.write_feature_stratified_performance(
            logs,
            [],
            _make_selected_features(["feat_a"]),
            datasets,
            "long",
            output_dir=str(tmp_path),
        )
        df = pd.read_csv(result[0])
        row = df[(df["feature"] == "feat_a") & (df["fuzzy_value"] == "High")]
        assert len(row) == 1
        # Only 3 valid trades (indices 0, 1, 2) should be counted
        assert row.iloc[0]["num_trades"] == 3

    def test_none_dataset_writes_header_only_csv(self, tmp_path):
        """None dataset for a split → header-only CSV."""
        reporter = Reporter()
        datasets = {
            "train": None,
            "validation": _make_stratified_dataset(feature_names=["feat_a"], seed=2),
            "test": _make_stratified_dataset(feature_names=["feat_a"], seed=3),
        }
        logs = _make_stratified_logs_by_split()
        result = reporter.write_feature_stratified_performance(
            logs,
            [],
            _make_selected_features(["feat_a"]),
            datasets,
            "long",
            output_dir=str(tmp_path),
        )
        assert len(result) == 3
        train_path = next(p for p in result if "train" in p)
        df = pd.read_csv(train_path)
        assert len(df) == 0
        assert set(df.columns) == self._REQUIRED_COLS

    def test_none_trade_log_writes_header_only_csv(self, tmp_path):
        """None trade log for a split → header-only CSV."""
        reporter = Reporter()
        datasets = _make_stratified_datasets_by_split(feature_names=["feat_a"])
        logs = {
            "train": None,
            "validation": _make_stratified_trade_log(seed=2),
            "test": _make_stratified_trade_log(seed=3),
        }
        result = reporter.write_feature_stratified_performance(
            logs,
            [],
            _make_selected_features(["feat_a"]),
            datasets,
            "long",
            output_dir=str(tmp_path),
        )
        assert len(result) == 3
        train_path = next(p for p in result if "train" in p)
        df = pd.read_csv(train_path)
        assert len(df) == 0
        assert set(df.columns) == self._REQUIRED_COLS

    def test_short_direction_creates_correct_filenames(self, tmp_path):
        reporter = Reporter()
        result = reporter.write_feature_stratified_performance(
            _make_stratified_logs_by_split(),
            [],
            _make_selected_features(["feat_a"]),
            _make_stratified_datasets_by_split(feature_names=["feat_a"]),
            "short",
            output_dir=str(tmp_path),
        )
        for path in result:
            assert "_short.csv" in path

    def test_metric_correctness(self, tmp_path):
        """Verify total_return_pct, win_rate, num_trades for a known stratum."""
        from gpu_fuzzy_trader import config as _cfg
        reporter = Reporter()
        # Build a controlled dataset: all rows are "High"
        n_dataset = 5
        dataset = pd.DataFrame({
            "feat_a": ["High"] * n_dataset,
            "label_close_288": np.zeros(n_dataset),
        })
        # 4 trades: 3 wins (+10, +20, +5) and 1 loss (-8)
        net_pnl = np.array([10.0, 20.0, 5.0, -8.0])
        equity_before = np.array([1000.0, 1010.0, 1030.0, 1035.0])
        trade_log = pd.DataFrame({
            "Entry_Index": [0, 1, 2, 3],
            "Release_Index": [1, 2, 3, 4],
            "Net_PnL": net_pnl,
            "Equity_After": equity_before + net_pnl,
            "Equity_Before_Entry": equity_before,
            "Rule_Index": [1, 1, 1, 1],
        })
        datasets = {"train": dataset, "validation": dataset, "test": dataset}
        logs = {"train": trade_log, "validation": trade_log, "test": trade_log}
        result = reporter.write_feature_stratified_performance(
            logs,
            [],
            _make_selected_features(["feat_a"]),
            datasets,
            "long",
            output_dir=str(tmp_path),
        )
        df = pd.read_csv(result[0])
        row = df[(df["feature"] == "feat_a") & (df["fuzzy_value"] == "High")].iloc[0]

        assert row["num_trades"] == 4
        expected_return = net_pnl.sum() / _cfg.INITIAL_CAPITAL * 100
        assert abs(row["total_return_pct"] - expected_return) < 1e-9
        expected_win_rate = 3 / 4
        assert abs(row["win_rate"] - expected_win_rate) < 1e-9
