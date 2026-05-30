"""Unit tests for gpu_fuzzy_trader.features.regime_cluster."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gpu_fuzzy_trader.features.regime_cluster import (
    _enforce_min_duration,
    assign_regime_labels,
    fit_regime_labels,
    persist_regime_model,
    load_regime_model,
)


def _regime_df(n: int = 100) -> pd.DataFrame:
    """Generates a synthetic DataFrame with a price column and date info."""
    dates = pd.date_range(start="2024-01-01", periods=n, freq="D")
    df_a = pd.DataFrame({
        "datetime": dates,
        "symbol": "A",
        "label_open_next": np.sin(np.linspace(0, 4 * np.pi, n)) + 1.0,
    })
    df_b = pd.DataFrame({
        "datetime": dates,
        "symbol": "B",
        "label_open_next": np.cos(np.linspace(0, 4 * np.pi, n)) + 1.0,
    })
    return pd.concat([df_a, df_b], ignore_index=True)


class TestEnforceMinDuration:
    def test_basic_merging(self) -> None:
        # A sequence with short blocks: 5 days of 0, 3 days of 1, 20 days of 2
        arr = np.array([0]*5 + [1]*3 + [2]*20)
        enforced = _enforce_min_duration(arr, min_days=10)
        # The block of 1s (3 days) is shorter than 10, so it gets merged.
        # The block of 0s (5 days) is shorter than 10, so it gets merged.
        # They should merge into the longest block (2s).
        assert np.all(enforced == 2)

    def test_already_valid(self) -> None:
        arr = np.array([0]*15 + [1]*20)
        enforced = _enforce_min_duration(arr, min_days=10)
        assert np.array_equal(enforced, arr)


class TestFitRegimeLabels:
    def test_returns_labels_and_bundle(self) -> None:
        df = _regime_df(100)
        result = fit_regime_labels(df)
        assert result is not None
        labels, bundle = result
        assert len(labels) == len(df)
        assert bundle["clusterer"] == "rolling_regression"
        assert bundle["min_days"] == 14

    def test_assign_matches_fit(self) -> None:
        df = _regime_df(100)
        result = fit_regime_labels(df)
        assert result is not None
        labels, bundle = result
        reassigned = assign_regime_labels(df, bundle)
        pd.testing.assert_series_equal(labels, reassigned)


class TestPersist:
    def test_round_trip(self, tmp_path) -> None:
        df = _regime_df(50)
        result = fit_regime_labels(df)
        assert result is not None
        _, bundle = result
        path = str(tmp_path / "regime.joblib")
        persist_regime_model(path, bundle)
        loaded = load_regime_model(path)
        assert loaded["clusterer"] == bundle["clusterer"]
        assert loaded["fast_window"] == bundle["fast_window"]
