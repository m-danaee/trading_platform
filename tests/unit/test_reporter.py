"""
Unit tests for gpu_fuzzy_trader.reporting.reporter.Reporter

Tests cover:
  - plot_phase2_metrics: normal case, empty history, single entry
  - plot_phase2_pnl: normal case, empty history, single entry
  - plot_equity_curve: normal case, empty trade_log, missing column, None trade_log
  - write_per_symbol_csv: normal case, empty metrics, missing key
  - plot_rl_curve: normal case, empty validation_returns, elbow_idx clamping
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from gpu_fuzzy_trader.reporting.reporter import Reporter


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
            "mean_total_return_pct": float(i) * 2.0,
            "best_total_return_pct": float(i) * 3.0 + 5.0,
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
        reporter.plot_phase2_metrics(_make_history(), "long", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "phase2_long_metrics.png")

    def test_creates_short_png_file(self, tmp_path):
        reporter = Reporter()
        reporter.plot_phase2_metrics(_make_history(), "short", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "phase2_short_metrics.png")

    def test_returns_correct_path(self, tmp_path):
        reporter = Reporter()
        result = reporter.plot_phase2_metrics(_make_history(), "long", output_dir=str(tmp_path))
        expected = os.path.join(str(tmp_path), "phase2_long_metrics.png")
        assert result == expected

    def test_empty_history_still_creates_file(self, tmp_path):
        reporter = Reporter()
        reporter.plot_phase2_metrics([], "long", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "phase2_long_metrics.png")

    def test_single_entry_history(self, tmp_path):
        reporter = Reporter()
        history = [{"generation": 0, "mean_f1": -1.0, "mean_f2": 0.5, "mean_f3": -0.8}]
        reporter.plot_phase2_metrics(history, "long", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "phase2_long_metrics.png")

    def test_file_is_nonzero_size(self, tmp_path):
        reporter = Reporter()
        reporter.plot_phase2_metrics(_make_history(10), "long", output_dir=str(tmp_path))
        size = os.path.getsize(tmp_path / "phase2_long_metrics.png")
        assert size > 0

    def test_creates_parent_dirs(self, tmp_path):
        reporter = Reporter()
        nested_dir = str(tmp_path / "a" / "b" / "c")
        reporter.plot_phase2_metrics(_make_history(), "long", output_dir=nested_dir)
        assert os.path.exists(os.path.join(nested_dir, "phase2_long_metrics.png"))

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
            "mean_total_return_pct": 10.0,
            "best_total_return_pct": 15.0,
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
            "generation": 1, "mean_total_return_pct": 5.0}]
        reporter.plot_phase2_pnl(history, "long", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "phase2_long_pnl.png")


# ---------------------------------------------------------------------------
# Tests: plot_equity_curve
# ---------------------------------------------------------------------------

class TestPlotEquityCurve:
    def test_creates_train_long_png(self, tmp_path):
        reporter = Reporter()
        reporter.plot_equity_curve(_make_trade_log(), "train", "long", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "train_long_equity.png")

    def test_creates_validation_short_png(self, tmp_path):
        reporter = Reporter()
        reporter.plot_equity_curve(_make_trade_log(), "validation", "short", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "validation_short_equity.png")

    def test_creates_test_long_png(self, tmp_path):
        reporter = Reporter()
        reporter.plot_equity_curve(_make_trade_log(), "test", "long", output_dir=str(tmp_path))
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
        reporter.plot_equity_curve(pd.DataFrame(), "train", "long", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "train_long_equity.png")

    def test_none_trade_log_creates_file(self, tmp_path):
        reporter = Reporter()
        reporter.plot_equity_curve(None, "train", "long", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "train_long_equity.png")

    def test_missing_equity_column_creates_file(self, tmp_path):
        reporter = Reporter()
        df = pd.DataFrame({"some_other_col": [1, 2, 3]})
        reporter.plot_equity_curve(df, "train", "long", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "train_long_equity.png")

    def test_file_is_nonzero_size(self, tmp_path):
        reporter = Reporter()
        reporter.plot_equity_curve(_make_trade_log(50), "train", "long", output_dir=str(tmp_path))
        size = os.path.getsize(tmp_path / "train_long_equity.png")
        assert size > 0

    def test_creates_parent_dirs(self, tmp_path):
        reporter = Reporter()
        nested_dir = str(tmp_path / "x" / "y")
        reporter.plot_equity_curve(_make_trade_log(), "test", "short", output_dir=nested_dir)
        assert os.path.exists(os.path.join(nested_dir, "test_short_equity.png"))

    def test_single_row_trade_log(self, tmp_path):
        reporter = Reporter()
        df = pd.DataFrame({"Equity_After": [1050.0]})
        reporter.plot_equity_curve(df, "train", "long", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "train_long_equity.png")


# ---------------------------------------------------------------------------
# Tests: write_per_symbol_csv
# ---------------------------------------------------------------------------

class TestWritePerSymbolCsv:
    def test_creates_train_csv(self, tmp_path):
        reporter = Reporter()
        reporter.write_per_symbol_csv(_make_per_symbol_metrics(), "train", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "train_per_symbol_performance.csv")

    def test_creates_validation_csv(self, tmp_path):
        reporter = Reporter()
        reporter.write_per_symbol_csv(_make_per_symbol_metrics(), "validation", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "validation_per_symbol_performance.csv")

    def test_creates_test_csv(self, tmp_path):
        reporter = Reporter()
        reporter.write_per_symbol_csv(_make_per_symbol_metrics(), "test", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "test_per_symbol_performance.csv")

    def test_returns_correct_path(self, tmp_path):
        reporter = Reporter()
        result = reporter.write_per_symbol_csv(
            _make_per_symbol_metrics(), "train", output_dir=str(tmp_path)
        )
        expected = os.path.join(str(tmp_path), "train_per_symbol_performance.csv")
        assert result == expected

    def test_csv_has_required_columns(self, tmp_path):
        reporter = Reporter()
        reporter.write_per_symbol_csv(_make_per_symbol_metrics(), "train", output_dir=str(tmp_path))
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
        reporter.write_per_symbol_csv(metrics, "train", output_dir=str(tmp_path))
        df = pd.read_csv(tmp_path / "train_per_symbol_performance.csv")
        row = df[df["symbol"] == "SYM_X"].iloc[0]
        assert row["trade_count"] == 7
        assert abs(row["win_rate"] - 71.4) < 1e-4
        assert abs(row["net_pnl"] - 15.5) < 1e-4

    def test_empty_per_symbol_metrics_creates_empty_csv(self, tmp_path):
        reporter = Reporter()
        reporter.write_per_symbol_csv({"per_symbol_metrics": {}}, "train", output_dir=str(tmp_path))
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
        reporter.write_per_symbol_csv(_make_per_symbol_metrics(), "test", output_dir=nested_dir)
        assert os.path.exists(os.path.join(nested_dir, "test_per_symbol_performance.csv"))

    def test_partial_sym_metrics_keys(self, tmp_path):
        """Symbols with missing sub-keys should default to 0."""
        reporter = Reporter()
        metrics = {
            "per_symbol_metrics": {
                "SYM_A": {},  # all keys missing
            }
        }
        reporter.write_per_symbol_csv(metrics, "train", output_dir=str(tmp_path))
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
        reporter.plot_rl_curve(returns, elbow_idx=2, direction="long", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "phase4_long_rl_curve.png")

    def test_creates_short_png(self, tmp_path):
        reporter = Reporter()
        returns = [0.5, 1.0, 1.5, 2.0]
        reporter.plot_rl_curve(returns, elbow_idx=1, direction="short", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "phase4_short_rl_curve.png")

    def test_returns_correct_path(self, tmp_path):
        reporter = Reporter()
        returns = [1.0, 2.0, 3.0]
        result = reporter.plot_rl_curve(returns, elbow_idx=1, direction="long", output_dir=str(tmp_path))
        expected = os.path.join(str(tmp_path), "phase4_long_rl_curve.png")
        assert result == expected

    def test_empty_validation_returns_creates_file(self, tmp_path):
        reporter = Reporter()
        reporter.plot_rl_curve([], elbow_idx=0, direction="long", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "phase4_long_rl_curve.png")

    def test_file_is_nonzero_size(self, tmp_path):
        reporter = Reporter()
        returns = list(range(20))
        reporter.plot_rl_curve(returns, elbow_idx=5, direction="long", output_dir=str(tmp_path))
        size = os.path.getsize(tmp_path / "phase4_long_rl_curve.png")
        assert size > 0

    def test_elbow_idx_clamped_when_too_large(self, tmp_path):
        """elbow_idx beyond list length should not raise."""
        reporter = Reporter()
        returns = [1.0, 2.0, 3.0]
        reporter.plot_rl_curve(returns, elbow_idx=999, direction="long", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "phase4_long_rl_curve.png")

    def test_elbow_idx_clamped_when_negative(self, tmp_path):
        """Negative elbow_idx should not raise."""
        reporter = Reporter()
        returns = [1.0, 2.0, 3.0]
        reporter.plot_rl_curve(returns, elbow_idx=-5, direction="long", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "phase4_long_rl_curve.png")

    def test_single_return_value(self, tmp_path):
        reporter = Reporter()
        reporter.plot_rl_curve([5.0], elbow_idx=0, direction="long", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "phase4_long_rl_curve.png")

    def test_creates_parent_dirs(self, tmp_path):
        reporter = Reporter()
        nested_dir = str(tmp_path / "r" / "s" / "t")
        reporter.plot_rl_curve([1.0, 2.0], elbow_idx=0, direction="short", output_dir=nested_dir)
        assert os.path.exists(os.path.join(nested_dir, "phase4_short_rl_curve.png"))

    def test_elbow_idx_zero(self, tmp_path):
        reporter = Reporter()
        returns = [3.0, 3.1, 3.2, 3.3]
        reporter.plot_rl_curve(returns, elbow_idx=0, direction="long", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "phase4_long_rl_curve.png")

    def test_elbow_idx_last(self, tmp_path):
        reporter = Reporter()
        returns = [1.0, 2.0, 3.0, 4.0]
        reporter.plot_rl_curve(returns, elbow_idx=3, direction="long", output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "phase4_long_rl_curve.png")
