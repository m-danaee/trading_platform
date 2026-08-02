"""
Unit tests for validation/monthly_windows.py.

These tests use small synthetic DataFrames and do NOT load train.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.validation.monthly_windows import (
    MonthlyWindowSummary,
    build_monthly_windows,
    monthly_penalty,
    summarize_monthly_metrics,
)


class TestMonthlyPenalty:
    """monthly_penalty edge cases."""

    def test_monthly_penalty_zero_windows(self) -> None:
        """Zero-window summary always returns 100.0."""
        summary = MonthlyWindowSummary(
            windows=0,
            profitable_windows=0,
            profitable_ratio=0.0,
            mean_return_pct=0.0,
            median_return_pct=0.0,
            worst_return_pct=0.0,
            latest_return_pct=0.0,
            recency_weighted_return_pct=0.0,
            mean_profit_factor=0.0,
            worst_profit_factor=0.0,
            worst_drawdown_pct=0.0,
            min_trades=0,
            mean_trades=0.0,
            equity_slope=0.0,
            max_equity_dip_pct=0.0,
            score=0.0,
        )
        assert monthly_penalty(summary) == 100.0

    def test_monthly_penalty_negative_windows(self) -> None:
        """Negative windows also return 100.0 (defensive)."""
        summary = MonthlyWindowSummary(
            windows=-1,
            profitable_windows=-1,
            profitable_ratio=0.0,
            mean_return_pct=0.0,
            median_return_pct=0.0,
            worst_return_pct=0.0,
            latest_return_pct=0.0,
            recency_weighted_return_pct=0.0,
            mean_profit_factor=0.0,
            worst_profit_factor=0.0,
            worst_drawdown_pct=0.0,
            min_trades=0,
            mean_trades=0.0,
            equity_slope=0.0,
            max_equity_dip_pct=0.0,
            score=0.0,
        )
        assert monthly_penalty(summary) == 100.0


class TestSummarizeMonthlyMetrics:
    """summarize_monthly_metrics edge cases."""

    def test_summarize_empty_metrics(self) -> None:
        """Empty metrics produces a summary with windows == 0 and score == -1e6."""
        summary = summarize_monthly_metrics([])
        assert summary.windows == 0
        assert summary.score == -1e6

    def test_zero_threshold_requires_activity_for_flat_months(self, monkeypatch) -> None:
        """Flat months count only when the strategy has trade support."""
        monkeypatch.setattr(_cfg, "MONTHLY_GOOD_RETURN_MIN_PCT", 0.0)
        metrics = [
            {"total_return_pct": 1.0, "profit_factor": 1.2,
             "max_drawdown_pct": 2.0, "executed_trades": 30},
            {"total_return_pct": 0.0, "profit_factor": 0.0,
             "max_drawdown_pct": 0.0, "executed_trades": 0},
            {"total_return_pct": -1.0, "profit_factor": 0.8,
             "max_drawdown_pct": 5.0, "executed_trades": 20},
        ]
        summary = summarize_monthly_metrics(metrics)
        assert summary.profitable_windows == 1
        assert summary.profitable_ratio == pytest.approx(1 / 3)
        assert summary.active_windows == 2
        assert summary.inactive_windows == 1

    def test_positive_threshold_requires_min_return(self, monkeypatch) -> None:
        """With MONTHLY_GOOD_RETURN_MIN_PCT=2, only months with return >= 2% count."""
        monkeypatch.setattr(_cfg, "MONTHLY_GOOD_RETURN_MIN_PCT", 2.0)
        metrics = [
            {"total_return_pct": 3.0, "profit_factor": 1.2,
             "max_drawdown_pct": 2.0, "executed_trades": 30},
            {"total_return_pct": 1.0, "profit_factor": 1.0,
             "max_drawdown_pct": 0.0, "executed_trades": 10},
            {"total_return_pct": 2.0, "profit_factor": 1.1,
             "max_drawdown_pct": 5.0, "executed_trades": 20},
        ]
        summary = summarize_monthly_metrics(metrics)
        assert summary.profitable_windows == 2
        assert summary.profitable_ratio == pytest.approx(2 / 3)


class TestBuildMonthlyWindows:
    """build_monthly_windows input validation."""

    def test_build_monthly_windows_requires_datetime(self) -> None:
        """ValueError raised when datetime column is missing."""
        df = pd.DataFrame({"foo": [1, 2, 3]})
        with pytest.raises(ValueError, match="datetime"):
            build_monthly_windows(df)

    def test_build_empty_df_returns_empty(self) -> None:
        """Empty DataFrame returns empty list."""
        df = pd.DataFrame({"datetime": pd.to_datetime([])})
        assert build_monthly_windows(df) == []

    def test_build_monthly_windows_uses_none_pattern(self) -> None:
        """Passing min_rows=0 gives more windows than passing min_rows=2500.

        This verifies the 'is not None' fix: 0 is a valid value that disables
        the min-rows filter.
        """
        np.random.seed(42)
        n_rows = 5000
        start = pd.Timestamp("2024-01-01")
        df = pd.DataFrame(
            {
                "datetime": [
                    start + pd.Timedelta(minutes=5 * i)
                    for i in range(n_rows)
                ],
                "symbol": [1] * n_rows,
                "label_open_next": np.random.randn(n_rows).tolist(),
                "label_close_288": np.random.randn(n_rows).tolist(),
                "label_min_288": np.random.randn(n_rows).tolist(),
                "label_max_288": np.random.randn(n_rows).tolist(),
                "label_max_before_min": np.random.randn(n_rows).tolist(),
            }
        )

        # With min_rows=0 (no filter), expect 2 sequential 30-day windows.
        windows_no_filter = build_monthly_windows(
            df, window_days=30, min_rows=0, max_windows=24
        )

        # With min_rows=2500 (strict filter), expect fewer or same windows.
        windows_filtered = build_monthly_windows(
            df, window_days=30, min_rows=2500, max_windows=24
        )

        assert len(windows_no_filter) >= len(windows_filtered), (
            f"min_rows=0 should give at least as many windows "
            f"as min_rows=2500 ({len(windows_no_filter)} vs {len(windows_filtered)})"
        )

    def test_build_monthly_windows_are_sequential_non_overlapping(self) -> None:
        """Each window starts where the previous one ended (no overlap)."""
        n_rows = 10_000
        start = pd.Timestamp("2024-01-01")
        df = pd.DataFrame(
            {
                "datetime": [
                    start + pd.Timedelta(minutes=5 * i)
                    for i in range(n_rows)
                ],
                "symbol": [1] * n_rows,
                "label_open_next": [0.0] * n_rows,
                "label_close_288": [0.0] * n_rows,
                "label_min_288": [0.0] * n_rows,
                "label_max_288": [0.0] * n_rows,
                "label_max_before_min": [0.0] * n_rows,
            }
        )
        windows = build_monthly_windows(
            df, window_days=30, min_rows=0, max_windows=24
        )
        assert len(windows) >= 2

        prev_end: pd.Timestamp | None = None
        for w in windows:
            dt = pd.to_datetime(w["datetime"])
            w_start = dt.min()
            w_end = dt.max()
            if prev_end is not None:
                assert w_start >= prev_end, (
                    "window start must not precede previous window's last bar"
                )
            prev_end = w_end


class TestSmoke:
    """Quick smoke checks for basic functionality."""

    def test_monthly_window_summary_fields(self) -> None:
        """MonthlyWindowSummary constructor works."""
        s = MonthlyWindowSummary(
            windows=1,
            profitable_windows=1,
            profitable_ratio=1.0,
            mean_return_pct=2.0,
            median_return_pct=2.0,
            worst_return_pct=2.0,
            latest_return_pct=2.0,
            recency_weighted_return_pct=2.0,
            mean_profit_factor=1.5,
            worst_profit_factor=1.5,
            worst_drawdown_pct=5.0,
            min_trades=10,
            mean_trades=10.0,
            equity_slope=0.5,
            max_equity_dip_pct=3.0,
            score=42.0,
        )
        assert s.windows == 1
        assert s.score == 42.0
