"""Unit tests for gpu_fuzzy_trader.features.regime_cluster."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.mixture import GaussianMixture

from gpu_fuzzy_trader import config
from gpu_fuzzy_trader.features.regime_cluster import (
    _safe_standardize_per_symbol,
    assign_regime_labels,
    fit_regime_labels,
    persist_regime_model,
    load_regime_model,
)


def _regime_df(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_a, n_b = n // 2, n - n // 2
    base = {col: rng.normal(size=n) for col in config.PHASE1_REGIME_FEATURES}
    df_a = pd.DataFrame(base)
    df_a = df_a.iloc[:n_a].copy()
    df_a["symbol"] = "A"
    df_b = pd.DataFrame({col: rng.normal(size=n_b) +
                        2.0 for col in config.PHASE1_REGIME_FEATURES})
    df_b["symbol"] = "B"
    return pd.concat([df_a, df_b], ignore_index=True)


class TestSafeStandardize:
    def test_constant_feature_no_nan(self) -> None:
        df = _regime_df(100)
        df["realized_vol_20"] = 1.0
        X, stats = _safe_standardize_per_symbol(
            df, config.PHASE1_REGIME_FEATURES, config.PHASE1_REGIME_ZERO_VAR_EPS,
        )
        assert not np.isnan(X).any()
        assert not np.isinf(X).any()
        assert "A" in stats or "__all__" in stats

    def test_per_symbol_zscore_differs_by_symbol(self) -> None:
        df = _regime_df(200)
        X, _ = _safe_standardize_per_symbol(
            df, ["realized_vol_20"], config.PHASE1_REGIME_ZERO_VAR_EPS,
        )
        a_mask = df["symbol"] == "A"
        b_mask = df["symbol"] == "B"
        assert abs(X[a_mask, 0].mean()) < 0.5
        assert abs(X[b_mask, 0].mean()) < 0.5


class TestFitRegimeLabels:
    def test_returns_labels_and_bundle(self) -> None:
        df = _regime_df(400)
        result = fit_regime_labels(df, n_clusters=3)
        assert result is not None
        labels, bundle = result
        assert len(labels) == len(df)
        assert labels.nunique() >= 2
        assert bundle["clusterer"] == config.PHASE1_REGIME_CLUSTERER

    def test_gmm_uses_kmeans_init_and_reg_covar(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict = {}

        class RecordingGMM(GaussianMixture):
            def __init__(self, *args, **kwargs):
                captured.update(kwargs)
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(
            "gpu_fuzzy_trader.features.regime_cluster.GaussianMixture",
            RecordingGMM,
        )
        df = _regime_df(300)
        fit_regime_labels(df, n_clusters=3, clusterer="gmm")
        assert captured.get("init_params") == "k-means++"
        assert captured.get("reg_covar") == config.PHASE1_REGIME_GMM_REG_COVAR

    def test_missing_column_returns_none(self) -> None:
        df = _regime_df(100).drop(columns=["realized_vol_20"])
        assert fit_regime_labels(df, n_clusters=3) is None

    def test_assign_matches_fit(self) -> None:
        df = _regime_df(300)
        result = fit_regime_labels(df, n_clusters=3)
        assert result is not None
        labels, bundle = result
        reassigned = assign_regime_labels(df, bundle)
        pd.testing.assert_series_equal(labels, reassigned)


class TestPersist:
    def test_round_trip(self, tmp_path) -> None:
        df = _regime_df(200)
        result = fit_regime_labels(df, n_clusters=3)
        assert result is not None
        _, bundle = result
        path = str(tmp_path / "regime.joblib")
        persist_regime_model(path, bundle)
        loaded = load_regime_model(path)
        assert loaded["regime_features"] == bundle["regime_features"]
