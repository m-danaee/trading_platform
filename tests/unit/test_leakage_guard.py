"""
Unit tests for gpu_fuzzy_trader.validation.leakage_guard.Leakage_Guard

Tests cover:
  - Probe injection (adds _leakage_probe column matching label_max_288)
  - Probe removal
  - Diagnostic: probe_detected=True when probe is selected (healthy/expected)
  - Diagnostic: probe_detected=False when probe is NOT selected (failure!)
  - Disabled guardrail (inject_probe and diagnose are no-ops)
  - rank and score reporting in details
  - LeakageAlert: raised by caller only on probe_detected=False
  - Module-level convenience functions
"""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from gpu_fuzzy_trader.validation.leakage_guard import (
    Leakage_Guard,
    LeakageAlert,
    inject_leakage_probe,
    diagnose_leakage_guard,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_train_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "symbol": ["SYM_A"] * n,
        "datetime": pd.date_range("2020-01-01", periods=n, freq="5min"),
        "label_max_288": rng.uniform(0.95, 1.10, size=n),
        "label_open_next": 1.0,
        "label_close_288": 1.0,
        "label_min_288": 0.99,
        "label_max_before_min": 1.0,
        "feature_a": rng.random(size=n),
        "feature_b": rng.random(size=n),
    })


def _make_selected_features(include_probe: bool = True) -> list[dict]:
    features = [
        {"name": "feature_a", "mode": "positive", "score": 0.8},
        {"name": "feature_b", "mode": "positive", "score": 0.7},
    ]
    if include_probe:
        features.insert(0, {"name": "_leakage_probe", "mode": "positive", "score": 1.0})
    return features


# ---------------------------------------------------------------------------
# Tests: injection
# ---------------------------------------------------------------------------


class TestInjectProbe:
    def test_injects_probe_column(self):
        df = _make_train_df(n=50)
        guard = Leakage_Guard()
        result = guard.inject_probe(df)
        assert "_leakage_probe" in result.columns

    def test_probe_matches_label_max_288(self):
        df = _make_train_df(n=50)
        guard = Leakage_Guard()
        result = guard.inject_probe(df)
        pd.testing.assert_series_equal(
            result["_leakage_probe"],
            result["label_max_288"],
            check_names=False,
        )

    def test_original_df_unchanged(self):
        df = _make_train_df(n=50)
        guard = Leakage_Guard()
        _ = guard.inject_probe(df)
        assert "_leakage_probe" not in df.columns

    def test_no_side_effects(self):
        df = _make_train_df(n=50)
        original_cols = set(df.columns)
        guard = Leakage_Guard()
        result = guard.inject_probe(df)
        assert len(result.columns) == len(original_cols) + 1
        assert original_cols.issubset(set(result.columns))

    def test_disabled_guard_returns_unchanged(self):
        df = _make_train_df(n=50)
        guard = Leakage_Guard(enabled=False)
        result = guard.inject_probe(df)
        assert "_leakage_probe" not in result.columns
        pd.testing.assert_frame_equal(result, df)

    def test_label_max_288_missing(self):
        df = pd.DataFrame({"feature_a": [1, 2, 3]})
        guard = Leakage_Guard()
        result = guard.inject_probe(df)
        assert "_leakage_probe" not in result.columns


# ---------------------------------------------------------------------------
# Tests: removal
# ---------------------------------------------------------------------------


class TestRemoveProbe:
    def test_removes_probe_column(self):
        df = _make_train_df(n=50)
        guard = Leakage_Guard()
        injected = guard.inject_probe(df)
        cleaned = guard.remove_probe(injected)
        assert "_leakage_probe" not in cleaned.columns

    def test_safe_when_no_probe(self):
        df = _make_train_df(n=50)
        guard = Leakage_Guard()
        result = guard.remove_probe(df)
        pd.testing.assert_frame_equal(result, df)


# ---------------------------------------------------------------------------
# Tests: diagnose — probe detected (healthy/expected outcome)
# ---------------------------------------------------------------------------


class TestDiagnoseProbeDetected:
    def test_probe_detected_when_probe_selected(self):
        """Probe IS selected → probe_detected=True (healthy, expected)."""
        guard = Leakage_Guard()
        selected = _make_selected_features(include_probe=True)
        report = guard.diagnose(selected, selected)
        assert report["probe_detected"] is True
        assert "probe" in report["message"].lower()

    def test_reports_rank_and_score(self):
        guard = Leakage_Guard()
        selected = _make_selected_features(include_probe=True)
        report = guard.diagnose(selected, selected)
        assert report["details"]["long_rank"] == 1
        assert report["details"]["long_score"] == 1.0

    def test_does_not_raise_by_itself(self):
        """diagnose() never raises; the CALLER raises LeakageAlert."""
        guard = Leakage_Guard()
        selected = _make_selected_features(include_probe=True)
        report = guard.diagnose(selected, selected)
        assert report["probe_detected"] is True
        # No exception raised

    def test_disabled_guard_reports_probe_detected_true(self):
        guard = Leakage_Guard(enabled=False)
        selected = _make_selected_features(include_probe=False)
        report = guard.diagnose(selected, selected)
        assert report["probe_detected"] is True


# ---------------------------------------------------------------------------
# Tests: diagnose — probe NOT detected (failure!)
# ---------------------------------------------------------------------------


class TestDiagnoseProbeNotDetected:
    def test_probe_not_detected_when_not_selected(self):
        """Probe NOT selected → probe_detected=False (selector is blind)."""
        guard = Leakage_Guard()
        selected = _make_selected_features(include_probe=False)
        report = guard.diagnose(selected, selected)
        assert report["probe_detected"] is False

    def test_probe_not_detected_when_only_one_direction_detects(self):
        guard = Leakage_Guard()
        selected_long = _make_selected_features(include_probe=True)
        selected_short = _make_selected_features(include_probe=False)
        report = guard.diagnose(selected_long, selected_short)
        assert report["probe_detected"] is False

    def test_details_show_missing(self):
        guard = Leakage_Guard()
        selected = _make_selected_features(include_probe=False)
        report = guard.diagnose(selected, selected)
        assert report["details"]["appeared_long"] is False
        assert report["details"]["appeared_short"] is False
        assert report["details"]["long_rank"] is None

    def test_caller_can_raise_leakage_alert(self):
        """The correct usage pattern: caller checks probe_detected and raises."""
        guard = Leakage_Guard()
        selected = _make_selected_features(include_probe=False)
        report = guard.diagnose(selected, selected)
        with pytest.raises(LeakageAlert, match="leak"):
            if not report["probe_detected"]:
                raise LeakageAlert("leak diagnostic failed")


# ---------------------------------------------------------------------------
# Tests: LeakageAlert exception
# ---------------------------------------------------------------------------


class TestLeakageAlert:
    def test_can_raise_and_catch(self):
        with pytest.raises(LeakageAlert, match="leak"):
            raise LeakageAlert("test leak detected")

    def test_is_runtime_error(self):
        assert issubclass(LeakageAlert, RuntimeError)


# ---------------------------------------------------------------------------
# Tests: convenience functions
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    def test_inject_leakage_probe(self):
        df = _make_train_df(n=20)
        result = inject_leakage_probe(df)
        assert "_leakage_probe" in result.columns

    def test_diagnose_leakage_guard(self):
        selected = _make_selected_features(include_probe=True)
        report = diagnose_leakage_guard(selected, selected)
        assert report["probe_detected"] is True
