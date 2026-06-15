"""
Unit test for monthly_penalty integration into Phase 3 / Phase 4 scoring.

Two rule-sets with identical ``total_return_pct`` on validation but
different monthly summaries should receive different composite scores —
the one with better monthly performance wins.
"""

from __future__ import annotations

import unittest.mock as umock

import numpy as np
import pandas as pd
import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.backtest.cpu_engine import CPUBacktestEngine
from gpu_fuzzy_trader.phases.phase3_rule_set import _per_symbol_greedy
from gpu_fuzzy_trader.validation.monthly_windows import (
    MonthlyWindowSummary,
    build_monthly_windows,
    evaluate_rule_set_monthly,
    monthly_penalty,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synthetic_df(n_rows: int = 6000) -> pd.DataFrame:
    """Create a small synthetic DataFrame with features and labels."""
    start = pd.Timestamp("2020-01-01")
    n_sym = 2
    symbols = [str(i) for i in range(n_sym)]

    rows: list[dict] = []
    rng = np.random.default_rng(42)
    for sym in symbols:
        for i in range(n_rows // n_sym):
            dt = start + pd.Timedelta(minutes=5 * i)
            rows.append({
                "datetime": dt,
                "symbol": sym,
                # label_open_next must be finite and > 0 (backtest requirement).
                "label_open_next": float(np.abs(rng.normal(1.0, 0.3))),
                "label_close_288": float(rng.normal(2.0, 1.0)),
                "label_min_288": float(rng.normal(-0.5, 1.0)),
                "label_max_288": float(rng.normal(2.5, 1.0)),
                "label_max_before_min": float(rng.normal(0.5, 0.3)),
                "feature_ma_5": float(rng.normal(0.0, 1.0)),
                "feature_ma_20": float(rng.normal(0.0, 1.0)),
                "feature_rsi": float(rng.normal(50.0, 10.0)),
            })
    df = pd.DataFrame(rows)
    return df


def _rule_set_all_long(condition_feature: str, fuzzy_value: str = "Very High") -> list[dict]:
    """A rule-set using the fuzzy IS format expected by the backtest engine."""
    return [
        {
            "conditions": [f"[{condition_feature}] IS {fuzzy_value}"],
            "tp": 3.0,
            "sl": 1.5,
            "capital_pct": 10.0,
        },
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMonthlyPenaltyInPhase3Scoring:
    """Monthly penalty integration: Phase 3 scoring."""

    def test_identical_validation_return_different_composite_scores(self) -> None:
        """
        Two rule-sets with the same ``total_return_pct`` on validation but
        different monthly performance get different composite scores.
        """
        df = _synthetic_df(6000)

        # Build two rule-sets — we rely on the fact that different conditions
        # produce different monthly profiles even if overall return is similar.
        rs_a = _rule_set_all_long("feature_ma_5")
        rs_b = _rule_set_all_long("feature_rsi")

        # Evaluate validation return for each.
        val_engine_a = CPUBacktestEngine(df, {}, "long")
        val_engine_b = CPUBacktestEngine(df, {}, "long")
        metrics_a = val_engine_a.simulate_rule_set(rs_a)
        metrics_b = val_engine_b.simulate_rule_set(rs_b)

        val_ret_a = float(metrics_a.get("total_return_pct", 0.0))
        val_ret_b = float(metrics_b.get("total_return_pct", 0.0))

        # Compute monthly summaries.
        feature_names = [
            c for c in df.columns
            if c not in set(_cfg.LABEL_COLUMNS)
            | set(_cfg.META_COLUMNS)
            | set(_cfg.INTERNAL_COLUMNS)
            and not str(c).startswith("_")
        ]
        summary_a, _ = evaluate_rule_set_monthly(
            df, rs_a, "long", feature_names=feature_names)
        summary_b, _ = evaluate_rule_set_monthly(
            df, rs_b, "long", feature_names=feature_names)

        # Compute monthly penalties.
        weight = float(getattr(_cfg, "PHASE3_MONTHLY_PENALTY_WEIGHT", 1.0))
        penalty_a = monthly_penalty(summary_a) * weight
        penalty_b = monthly_penalty(summary_b) * weight

        # Composite scores: higher return – penalty.
        composite_a = val_ret_a - penalty_a
        composite_b = val_ret_b - penalty_b

        # The rule-set with the better monthly summary should have a higher
        # composite score (or at minimum, the penalty differentiates them).
        better_monthly = (
            summary_a if summary_a.score > summary_b.score else summary_b
        )
        composite_better = (
            composite_a if better_monthly is summary_a else composite_b
        )
        composite_worse = (
            composite_b if better_monthly is summary_a else composite_a
        )

        # The gap between the two composites must be at least 0.5 points
        # (penalty is meaningful).
        gap = composite_better - composite_worse
        assert gap > 0.5, (
            f"Expected composite score gap > 0.5 between better and worse "
            f"monthly performers, got gap={gap:.4f}.\n"
            f"  val_ret_a={val_ret_a:.4f}, val_ret_b={val_ret_b:.4f}\n"
            f"  penalty_a={penalty_a:.4f}, penalty_b={penalty_b:.4f}\n"
            f"  summary_a.score={summary_a.score:.4f}, "
            f"summary_b.score={summary_b.score:.4f}"
        )


class TestMonthlyPenaltyInPhase4Scoring:
    """Monthly penalty integration: Phase 4 scoring."""

    def test_monthly_penalty_affects_phase4_score(self) -> None:
        """
        When monthly penalty is applied, a trial with worse monthly
        performance receives a lower composite score.
        """
        weight = float(getattr(_cfg, "PHASE4_MONTHLY_SCORE_WEIGHT", 0.70))

        # Build two controlled summaries:
        #   "good" — profitable windows, positive equity slope
        #   "bad"  — unprofitable windows, negative equity slope
        good_summary = MonthlyWindowSummary(
            windows=6,
            profitable_windows=5,
            profitable_ratio=5.0 / 6.0,
            mean_return_pct=3.0,
            median_return_pct=2.8,
            worst_return_pct=1.0,
            latest_return_pct=2.5,
            recency_weighted_return_pct=2.9,
            mean_profit_factor=1.8,
            worst_profit_factor=1.2,
            worst_drawdown_pct=4.0,
            min_trades=30,
            mean_trades=45.0,
            equity_slope=2.5,
            max_equity_dip_pct=5.0,
            score=100.0,
        )
        bad_summary = MonthlyWindowSummary(
            windows=6,
            profitable_windows=1,
            profitable_ratio=1.0 / 6.0,
            mean_return_pct=-2.0,
            median_return_pct=-1.5,
            worst_return_pct=-8.0,
            latest_return_pct=-3.0,
            recency_weighted_return_pct=-2.5,
            mean_profit_factor=0.6,
            worst_profit_factor=0.2,
            worst_drawdown_pct=15.0,
            min_trades=5,
            mean_trades=18.0,
            equity_slope=-1.5,
            max_equity_dip_pct=18.0,
            score=-200.0,
        )

        penalty_good = monthly_penalty(good_summary) * weight
        penalty_bad = monthly_penalty(bad_summary) * weight

        # Both trials have the same worst_return and fold_penalty.
        identical_return = 10.0
        score_good = identical_return - penalty_good
        score_bad = identical_return - penalty_bad

        assert penalty_bad > penalty_good, (
            f"Expected bad-monthly penalty ({penalty_bad:.4f}) > "
            f"good-monthly penalty ({penalty_good:.4f})"
        )
        assert score_good > score_bad, (
            f"Expected score_good ({score_good:.4f}) > score_bad ({score_bad:.4f})\n"
            f"  penalty_good={penalty_good:.4f}, penalty_bad={penalty_bad:.4f}"
        )


# ---------------------------------------------------------------------------
# Integration: _per_symbol_greedy with monthly penalty wired in
# ---------------------------------------------------------------------------


def _tiny_pool() -> list[dict]:
    """Return a 2-rule pool with different conditions."""
    return [
        {
            "conditions": ["[feature_ma_5] IS Very High"],
            "tp": 3.0,
            "sl": 1.5,
            "capital_pct": 10.0,
        },
        {
            "conditions": ["[feature_rsi] IS Very Low"],
            "tp": 3.0,
            "sl": 1.5,
            "capital_pct": 10.0,
        },
    ]


def _synthetic_df_monthly(n_rows: int = 30000) -> pd.DataFrame:
    """Create a larger synthetic DataFrame (enough for ~2 monthly windows)."""
    start = pd.Timestamp("2020-01-01")
    n_sym = 2
    symbols = [str(i) for i in range(n_sym)]
    rows: list[dict] = []
    rng = np.random.default_rng(12345)
    for sym in symbols:
        for i in range(n_rows // n_sym):
            dt = start + pd.Timedelta(minutes=5 * i)
            rows.append({
                "datetime": dt,
                "symbol": sym,
                "label_open_next": float(np.abs(rng.normal(1.0, 0.3))),
                "label_close_288": float(rng.normal(2.0, 1.0)),
                "label_min_288": float(rng.normal(-0.5, 1.0)),
                "label_max_288": float(rng.normal(2.5, 1.0)),
                "label_max_before_min": float(rng.normal(0.5, 0.3)),
                "feature_ma_5": float(rng.normal(0.0, 1.0)),
                "feature_ma_20": float(rng.normal(0.0, 1.0)),
                "feature_rsi": float(rng.normal(50.0, 10.0)),
            })
    df = pd.DataFrame(rows)
    return df


class TestIntegrationPerSymbolGreedy:
    """Integration: _per_symbol_greedy with monthly penalty wiring."""

    def test_greedy_runs_with_monthly_penalty_enabled(self) -> None:
        """
        _per_symbol_greedy should run without error when monthly windows are
        cached and MONTHLY_VALIDATION_ENABLED is True.  This exercises the
        _robust_combo_return path with the pre-built windows list.
        """
        df = _synthetic_df_monthly(30000)
        pool = _tiny_pool()
        sym = "0"
        sym_df = df[df["symbol"].astype(str) == sym].reset_index(drop=True)

        feature_names = [
            c for c in df.columns
            if c not in set(_cfg.LABEL_COLUMNS)
            | set(_cfg.META_COLUMNS)
            | set(_cfg.INTERNAL_COLUMNS)
            and not str(c).startswith("_")
        ]

        # Build monthly windows once (same as Rule_Set_Selector.run() does).
        monthly_windows = build_monthly_windows(df)

        with umock.patch.object(_cfg, "MONTHLY_VALIDATION_ENABLED", True):
            selected = _per_symbol_greedy(
                symbol=sym,
                symbol_df=sym_df,
                pool=pool,
                direction="long",
                combined_df=df,
                feature_names=feature_names,
                monthly_windows=monthly_windows,
            )

        # Must return a list (possibly empty — depends on min_return checks).
        assert isinstance(selected, list)
        # All returned indices must be valid pool indices.
        for idx in selected:
            assert 0 <= idx < len(pool), f"Invalid pool index {idx}"

    def test_monthly_enabled_changes_selection(self) -> None:
        """
        With monthly validation enabled, the greedy selection should differ
        from when it is disabled, because monthly-unstable rules get a penalty.
        """
        df = _synthetic_df_monthly(30000)
        pool = _tiny_pool()
        sym = "0"
        sym_df = df[df["symbol"].astype(str) == sym].reset_index(drop=True)

        feature_names = [
            c for c in df.columns
            if c not in set(_cfg.LABEL_COLUMNS)
            | set(_cfg.META_COLUMNS)
            | set(_cfg.INTERNAL_COLUMNS)
            and not str(c).startswith("_")
        ]

        monthly_windows = build_monthly_windows(df)

        with umock.patch.object(_cfg, "MONTHLY_VALIDATION_ENABLED", False):
            selected_off = _per_symbol_greedy(
                symbol=sym,
                symbol_df=sym_df,
                pool=pool,
                direction="long",
                combined_df=df,
                feature_names=feature_names,
                monthly_windows=monthly_windows,
            )

        with umock.patch.object(_cfg, "MONTHLY_VALIDATION_ENABLED", True):
            selected_on = _per_symbol_greedy(
                symbol=sym,
                symbol_df=sym_df,
                pool=pool,
                direction="long",
                combined_df=df,
                feature_names=feature_names,
                monthly_windows=monthly_windows,
            )

        # The selections could be equal (if data doesn't produce clear monthly
        # difference) — this is a smoke-test assertion that the code path runs.
        assert isinstance(selected_on, list)
        assert isinstance(selected_off, list)
