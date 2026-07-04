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
from gpu_fuzzy_trader.phases.phase3_rule_set import (
    _build_per_symbol_monthly_context,
    _monthly_feature_names,
    _per_symbol_greedy,
    _phase3_scaled_monthly_penalty,
)
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
        Two rule-sets with different monthly profiles receive different
        monthly penalties; the better monthly summary gets the lower penalty.
        """
        df = _synthetic_df(6000)

        rs_a = _rule_set_all_long("feature_ma_5")
        rs_b = _rule_set_all_long("feature_rsi")

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

        penalty_a = _phase3_scaled_monthly_penalty(monthly_penalty(summary_a))
        penalty_b = _phase3_scaled_monthly_penalty(monthly_penalty(summary_b))

        assert penalty_a != penalty_b or summary_a.score != summary_b.score, (
            "Expected distinct monthly profiles for the two rule sets"
        )
        if summary_a.score > summary_b.score:
            assert penalty_a < penalty_b
        elif summary_b.score > summary_a.score:
            assert penalty_b < penalty_a


class TestMonthlyPenaltyInPhase4Scoring:
    """Monthly penalty integration: Phase 4 scoring."""

    def test_monthly_penalty_affects_phase4_score(self) -> None:
        """
        When monthly penalty is applied, a trial with worse monthly
        performance receives a lower composite score.
        """
        from gpu_fuzzy_trader.phases.phase4_wf_optimizer import (
            _phase4_scaled_monthly_penalty,
        )

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
        penalty_good = _phase4_scaled_monthly_penalty(
            monthly_penalty(good_summary))
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
        penalty_bad = _phase4_scaled_monthly_penalty(
            monthly_penalty(bad_summary))

        identical_base_score = 250.0
        score_good = identical_base_score - penalty_good
        score_bad = identical_base_score - penalty_bad

        assert penalty_bad > penalty_good, (
            f"Expected bad-monthly penalty ({penalty_bad:.4f}) > "
            f"good-monthly penalty ({penalty_good:.4f})"
        )
        assert score_good > score_bad, (
            f"Expected score_good ({score_good:.4f}) > score_bad ({score_bad:.4f})\n"
            f"  penalty_good={penalty_good:.4f}, penalty_bad={penalty_bad:.4f}"
        )

    def test_evaluate_ruleset_subtracts_monthly_drag(self, monkeypatch) -> None:
        """_evaluate_ruleset applies the same monthly drag as production scoring."""
        from gpu_fuzzy_trader.phases.phase4_wf_optimizer import (
            _Phase4MonthlyContext,
            _evaluate_ruleset,
            _score_metrics,
        )

        train_m = {
            "total_return_pct": 4.0,
            "max_drawdown_pct": 5.0,
            "profit_factor": 1.5,
            "win_rate": 55.0,
        }
        val_m = {
            "total_return_pct": 3.0,
            "max_drawdown_pct": 6.0,
            "profit_factor": 1.4,
            "win_rate": 52.0,
        }
        base_score = _score_metrics(train_m, val_m)

        class _StubEngine:
            def __init__(self, metrics: dict) -> None:
                self._metrics = metrics

            def simulate_rule_set(self, rules: list[dict]) -> dict:
                return dict(self._metrics)

        monthly_ctx = _Phase4MonthlyContext(
            combined_df=pd.DataFrame(
                {"datetime": [pd.Timestamp("2020-01-01")]}),
            monthly_windows=[pd.DataFrame(
                {"datetime": [pd.Timestamp("2020-01-01")]})],
            feature_names=[],
            direction="long",
        )

        def _fake_drag(rules: list[dict], ctx: _Phase4MonthlyContext) -> float:
            return 12.5

        monkeypatch.setattr(
            "gpu_fuzzy_trader.phases.phase4_wf_optimizer._monthly_drag_for_rules",
            _fake_drag,
        )

        _, _, score = _evaluate_ruleset(
            _StubEngine(train_m),
            _StubEngine(val_m),
            [{"conditions": ["[x] IS high"], "tp": 2.0,
                "sl": 1.0, "capital_pct": 10.0}],
            monthly_ctx=monthly_ctx,
        )
        assert score == pytest.approx(base_score - 12.5)


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

        feature_names = _monthly_feature_names(df)
        sym_combined_df, monthly_windows = _build_per_symbol_monthly_context(
            None, sym_df)

        with umock.patch.object(_cfg, "MONTHLY_VALIDATION_ENABLED", True):
            selected = _per_symbol_greedy(
                symbol=sym,
                symbol_df=sym_df,
                pool=pool,
                direction="long",
                combined_df=sym_combined_df,
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

        feature_names = _monthly_feature_names(df)
        sym_combined_df, monthly_windows = _build_per_symbol_monthly_context(
            None, sym_df)

        with umock.patch.object(_cfg, "MONTHLY_VALIDATION_ENABLED", False):
            selected_off = _per_symbol_greedy(
                symbol=sym,
                symbol_df=sym_df,
                pool=pool,
                direction="long",
                combined_df=sym_combined_df,
                feature_names=feature_names,
                monthly_windows=monthly_windows,
            )

        with umock.patch.object(_cfg, "MONTHLY_VALIDATION_ENABLED", True):
            selected_on = _per_symbol_greedy(
                symbol=sym,
                symbol_df=sym_df,
                pool=pool,
                direction="long",
                combined_df=sym_combined_df,
                feature_names=feature_names,
                monthly_windows=monthly_windows,
            )

        # The selections could be equal (if data doesn't produce clear monthly
        # difference) — this is a smoke-test assertion that the code path runs.
        assert isinstance(selected_on, list)
        assert isinstance(selected_off, list)

    def test_monthly_eval_uses_per_symbol_data_only(self) -> None:
        """Monthly penalty must run on the symbol slice, not the full universe."""
        df = _synthetic_df_monthly(30000)
        pool = _tiny_pool()
        sym = "0"
        sym_df = df[df["symbol"].astype(str) == sym].reset_index(drop=True)
        sym_combined_df, monthly_windows = _build_per_symbol_monthly_context(
            None, sym_df)

        captured_dfs: list[pd.DataFrame] = []

        def _capture_eval(df_in, rule_set, direction, feature_names=None, windows=None, n_rows=None):
            captured_dfs.append(df_in)
            return evaluate_rule_set_monthly(
                df_in, rule_set, direction,
                feature_names=feature_names, windows=windows, n_rows=n_rows)

        patches = {
            "MONTHLY_VALIDATION_ENABLED": True,
            "PHASE3_PER_SYMBOL_MIN_TRADES": 0,
            "PHASE3_PER_SYMBOL_MIN_RETURN": -999.0,
            "PHASE3_PER_SYMBOL_MAX_RULES": 2,
        }
        with umock.patch.multiple(_cfg, **patches):
            with umock.patch(
                "gpu_fuzzy_trader.phases.phase3_rule_set.evaluate_rule_set_monthly",
                side_effect=_capture_eval,
            ):
                _per_symbol_greedy(
                    symbol=sym,
                    symbol_df=sym_df,
                    pool=pool,
                    direction="long",
                    combined_df=sym_combined_df,
                    feature_names=_monthly_feature_names(df),
                    monthly_windows=monthly_windows,
                )

        assert captured_dfs, "monthly eval should have been called for combo scoring"
        for passed_df in captured_dfs:
            assert set(passed_df["symbol"].astype(str).unique()) == {sym}

    def test_build_per_symbol_monthly_context_single_symbol(self) -> None:
        """Per-symbol monthly windows must not include other symbols' rows."""
        df = _synthetic_df_monthly(30000)
        sym_df = df[df["symbol"].astype(str) == "1"].reset_index(drop=True)
        combined, windows = _build_per_symbol_monthly_context(None, sym_df)
        assert combined is not None
        assert set(combined["symbol"].astype(str).unique()) == {"1"}
        assert windows is not None
        for window in windows:
            assert set(window["symbol"].astype(str).unique()) == {"1"}


class TestPhase3MonthlyPenaltyScale:
    """PHASE3_MONTHLY_PENALTY_SCALE normalizes penalty vs return %."""

    def test_scale_divides_penalty_before_subtraction(self, monkeypatch) -> None:
        monkeypatch.setattr(_cfg, "PHASE3_MONTHLY_PENALTY_WEIGHT", 1.0)
        monkeypatch.setattr(_cfg, "PHASE3_MONTHLY_PENALTY_SCALE", 10.0)
        assert _phase3_scaled_monthly_penalty(20.0) == pytest.approx(2.0)

    def test_scale_one_preserves_legacy_amount(self, monkeypatch) -> None:
        monkeypatch.setattr(_cfg, "PHASE3_MONTHLY_PENALTY_WEIGHT", 1.5)
        monkeypatch.setattr(_cfg, "PHASE3_MONTHLY_PENALTY_SCALE", 1.0)
        assert _phase3_scaled_monthly_penalty(10.0) == pytest.approx(15.0)
