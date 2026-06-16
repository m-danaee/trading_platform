"""
Unit tests for date-based x-axis in Reporter.plot_equity_curve (Task 10.3).

Tests cover:
  - When Entry_Time is present and valid, the x-axis label is "Date"
    (not "Trade #").
  - When Entry_Time is missing, the x-axis label falls back to "Trade #".
  - When Entry_Time is all-NaN, the x-axis label falls back to "Trade #".
  - The initial-capital reference line appears in both date and trade-# modes.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from gpu_fuzzy_trader.reporting.reporter import Reporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trade_log(
    n: int = 20,
    seed: int = 42,
    with_entry_time: bool = True,
) -> pd.DataFrame:
    """Create a minimal trade log with Equity_After (and optionally Entry_Time)."""
    rng = np.random.default_rng(seed)
    equity = 1000.0 + np.cumsum(rng.uniform(-10, 15, size=n))
    df = pd.DataFrame({"Equity_After": equity})
    if with_entry_time:
        # Use timezone-naive datetime for simpler testing
        df["Entry_Time"] = pd.date_range(
            "2024-01-01", periods=n, freq="D",
        )
    return df


# ---------------------------------------------------------------------------
# Tests: axis label via mocked matplotlib
# ---------------------------------------------------------------------------


class TestEquityCurveDateAxis:
    """Test the date-based x-axis (Entry_Time path) via mocked plt.subplots."""

    @pytest.fixture
    def reporter(self):
        return Reporter()

    @pytest.fixture
    def mock_ax(self):
        """Return a MagicMock that stands in for a matplotlib Axes."""
        ax = MagicMock()
        ax.xaxis = MagicMock()
        return ax

    @pytest.fixture
    def mock_fig(self, mock_ax):
        fig = MagicMock()
        fig.add_subplot.return_value = mock_ax
        return fig

    def _call_with_patched_subplots(
        self, reporter, trade_log, mock_ax, tmp_path,
    ):
        """Call plot_equity_curve with plt.subplots patched to return mock_ax."""
        with patch("gpu_fuzzy_trader.reporting.reporter.plt.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            reporter.plot_equity_curve(
                trade_log, "train", "long", output_dir=str(tmp_path),
            )
        return mock_ax

    def test_xlabel_date_when_entry_time_present(
        self, reporter, mock_ax, tmp_path,
    ):
        """When Entry_Time is present and valid, x-axis label is 'Date'."""
        trade_log = _make_trade_log(with_entry_time=True)
        self._call_with_patched_subplots(reporter, trade_log, mock_ax, tmp_path)

        # Find the set_xlabel call and check its argument
        xlabel_calls = [
            c.args[0] for c in mock_ax.set_xlabel.call_args_list
        ]
        assert "Date" in xlabel_calls, (
            f"Expected 'Date' in set_xlabel calls, got {xlabel_calls}"
        )

    def test_xlabel_trade_number_when_entry_time_missing(
        self, reporter, mock_ax, tmp_path,
    ):
        """When Entry_Time is absent, x-axis label is 'Trade #'."""
        trade_log = _make_trade_log(with_entry_time=False)
        self._call_with_patched_subplots(reporter, trade_log, mock_ax, tmp_path)

        xlabel_calls = [
            c.args[0] for c in mock_ax.set_xlabel.call_args_list
        ]
        assert "Trade #" in xlabel_calls, (
            f"Expected 'Trade #' in set_xlabel calls, got {xlabel_calls}"
        )

    def test_xlabel_trade_number_when_entry_time_all_nan(
        self, reporter, mock_ax, tmp_path,
    ):
        """When Entry_Time is all-NaN, x-axis label is 'Trade #'."""
        trade_log = _make_trade_log(with_entry_time=True)
        trade_log["Entry_Time"] = pd.NaT
        self._call_with_patched_subplots(reporter, trade_log, mock_ax, tmp_path)

        xlabel_calls = [
            c.args[0] for c in mock_ax.set_xlabel.call_args_list
        ]
        assert "Trade #" in xlabel_calls, (
            f"Expected 'Trade #' in set_xlabel calls, got {xlabel_calls}"
        )

    def test_axhline_called_for_initial_capital(
        self, reporter, mock_ax, tmp_path,
    ):
        """axhline is called (initial-capital line) in date mode."""
        trade_log = _make_trade_log(with_entry_time=True)
        self._call_with_patched_subplots(reporter, trade_log, mock_ax, tmp_path)

        assert mock_ax.axhline.called, "axhline should be called for initial capital"

    def test_axhline_called_in_trade_number_mode(
        self, reporter, mock_ax, tmp_path,
    ):
        """axhline is called (initial-capital line) in Trade # mode."""
        trade_log = _make_trade_log(with_entry_time=False)
        self._call_with_patched_subplots(reporter, trade_log, mock_ax, tmp_path)

        assert mock_ax.axhline.called, "axhline should be called for initial capital"

    def test_date_formatter_set_when_entry_time_present(
        self, reporter, mock_ax, tmp_path,
    ):
        """DateFormatter is set when Entry_Time is present."""
        trade_log = _make_trade_log(with_entry_time=True)
        self._call_with_patched_subplots(reporter, trade_log, mock_ax, tmp_path)

        assert mock_ax.xaxis.set_major_formatter.called, (
            "set_major_formatter should be called for date formatting"
        )

    def test_no_date_formatter_when_entry_time_missing(
        self, reporter, mock_ax, tmp_path,
    ):
        """DateFormatter is NOT set when Entry_Time is missing."""
        trade_log = _make_trade_log(with_entry_time=False)
        self._call_with_patched_subplots(reporter, trade_log, mock_ax, tmp_path)

        assert not mock_ax.xaxis.set_major_formatter.called, (
            "set_major_formatter should NOT be called for Trade # mode"
        )


class TestEquityCurveBasic:
    """Basic file-creation tests that call the real plot_equity_curve."""

    def test_creates_file_with_entry_time(self, tmp_path):
        """File is created when Entry_Time is present."""
        reporter = Reporter()
        trade_log = _make_trade_log(with_entry_time=True)
        result = reporter.plot_equity_curve(
            trade_log, "train", "long", output_dir=str(tmp_path),
        )
        expected = os.path.join(str(tmp_path), "train_long_equity.png")
        assert result == expected
        assert os.path.exists(expected)

    def test_creates_file_without_entry_time(self, tmp_path):
        """File is created when Entry_Time is absent."""
        reporter = Reporter()
        trade_log = _make_trade_log(with_entry_time=False)
        result = reporter.plot_equity_curve(
            trade_log, "validation", "short", output_dir=str(tmp_path),
        )
        expected = os.path.join(str(tmp_path), "validation_short_equity.png")
        assert result == expected
        assert os.path.exists(expected)

    def test_creates_file_with_all_nan_entry_time(self, tmp_path):
        """File is created when Entry_Time is all-NaN."""
        reporter = Reporter()
        trade_log = _make_trade_log(with_entry_time=True)
        trade_log["Entry_Time"] = pd.NaT
        result = reporter.plot_equity_curve(
            trade_log, "test", "long", output_dir=str(tmp_path),
        )
        expected = os.path.join(str(tmp_path), "test_long_equity.png")
        assert result == expected
        assert os.path.exists(expected)
