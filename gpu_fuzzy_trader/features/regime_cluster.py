"""
regime_cluster.py — Market-regime labeling for Phase 1 stationarity.

Fits a pooled cluster model on per-symbol z-scored regime indicators (train only).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from gpu_fuzzy_trader import config

logger = logging.getLogger(__name__)

RegimeBundle = dict[str, Any]


def _safe_standardize_per_symbol(
    df: pd.DataFrame,
    regime_features: list[str],
    zero_var_eps: float,
) -> tuple[np.ndarray, dict[str, dict[str, tuple[float, float]]]]:
    """
    Z-score regime features within each symbol; constant columns → 0.0.

    Returns (X_scaled shape (n_rows, n_features), per_symbol_stats).
    per_symbol_stats[sym][col] = (mean, std) with std floored to 1.0 when zero-var.
    """
    if "symbol" not in df.columns:
        symbols = [None]
        groups = [(None, df)]
    else:
        groups = [(sym, grp) for sym, grp in df.groupby("symbol", sort=False)]

    n_features = len(regime_features)
    X = np.zeros((len(df), n_features), dtype=np.float64)
    index_to_row = {idx: i for i, idx in enumerate(df.index)}
    per_symbol_stats: dict[str, dict[str, tuple[float, float]]] = {}

    for sym, grp in groups:
        sym_key = str(sym) if sym is not None else "__all__"
        per_symbol_stats[sym_key] = {}
        row_indices = [index_to_row[i] for i in grp.index]

        for j, col in enumerate(regime_features):
            vals = grp[col].astype(np.float64).values
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals, ddof=0))
            store_std = std_val

            if std_val < zero_var_eps or len(np.unique(vals)) == 1:
                logger.warning(
                    "Regime z-score: zero variance for symbol=%s column=%s; using 0.0",
                    sym_key,
                    col,
                )
                scaled = np.zeros(len(vals), dtype=np.float64)
                store_std = 1.0
            else:
                scaled = (vals - mean_val) / std_val

            per_symbol_stats[sym_key][col] = (mean_val, store_std)
            X[row_indices, j] = scaled

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, per_symbol_stats


def _transform_with_stats(
    df: pd.DataFrame,
    regime_features: list[str],
    per_symbol_stats: dict[str, dict[str, tuple[float, float]]],
    zero_var_eps: float,
) -> np.ndarray:
    """Apply stored per-symbol mean/std to new rows (inference)."""
    n_features = len(regime_features)
    X = np.zeros((len(df), n_features), dtype=np.float64)
    index_to_row = {idx: i for i, idx in enumerate(df.index)}

    if "symbol" not in df.columns:
        sym_key = "__all__"
        stats = per_symbol_stats.get(sym_key, {})
        for j, col in enumerate(regime_features):
            mean_val, std_val = stats.get(col, (0.0, 1.0))
            vals = df[col].astype(np.float64).values
            if std_val < zero_var_eps:
                scaled = np.zeros(len(vals), dtype=np.float64)
            else:
                scaled = (vals - mean_val) / std_val
            X[:, j] = scaled
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    for sym, grp in df.groupby("symbol", sort=False):
        sym_key = str(sym)
        stats = per_symbol_stats.get(
            sym_key, per_symbol_stats.get("__all__", {}))
        row_indices = [index_to_row[i] for i in grp.index]
        for j, col in enumerate(regime_features):
            mean_val, std_val = stats.get(col, (0.0, 1.0))
            vals = grp[col].astype(np.float64).values
            if std_val < zero_var_eps:
                scaled = np.zeros(len(vals), dtype=np.float64)
            else:
                scaled = (vals - mean_val) / std_val
            X[row_indices, j] = scaled

    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def _fit_clusterer(
    X: np.ndarray,
    n_clusters: int,
    clusterer: str,
    random_state: int,
) -> Any:
    if clusterer == "kmeans":
        model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
        )
    else:
        model = GaussianMixture(
            n_components=n_clusters,
            random_state=random_state,
            init_params="k-means++",
            reg_covar=config.PHASE1_REGIME_GMM_REG_COVAR,
        )
    model.fit(X)
    return model


def _predict_labels(model: Any, X: np.ndarray, clusterer: str) -> np.ndarray:
    if clusterer == "kmeans":
        return model.predict(X)
    return model.predict(X)


def fit_regime_labels(
    df: pd.DataFrame,
    regime_features: Optional[list[str]] = None,
    n_clusters: Optional[int] = None,
    clusterer: Optional[str] = None,
    random_state: int = 42,
    zero_var_eps: Optional[float] = None,
) -> Optional[tuple[pd.Series, RegimeBundle]]:
    """
    Fit regime clustering on train rows; return labels aligned to df.index.

    Returns None if regime columns are missing or clustering fails.
    """
    regime_features = regime_features or list(config.PHASE1_REGIME_FEATURES)
    n_clusters = n_clusters if n_clusters is not None else config.PHASE1_REGIME_N_CLUSTERS
    clusterer = (clusterer or config.PHASE1_REGIME_CLUSTERER).lower()
    zero_var_eps = zero_var_eps if zero_var_eps is not None else config.PHASE1_REGIME_ZERO_VAR_EPS

    missing = [c for c in regime_features if c not in df.columns]
    if missing:
        logger.warning(
            "Regime clustering skipped: missing columns %s", missing,
        )
        return None

    if n_clusters < 2:
        return None

    try:
        X, per_symbol_stats = _safe_standardize_per_symbol(
            df, regime_features, zero_var_eps,
        )
        model = _fit_clusterer(X, n_clusters, clusterer, random_state)
        labels = _predict_labels(model, X, clusterer)
    except Exception as exc:
        logger.warning(
            "Regime clustering failed (%s); caller should fall back to chronological",
            exc,
        )
        return None

    label_series = pd.Series(labels.astype(
        np.int32), index=df.index, name="regime")
    bundle: RegimeBundle = {
        "regime_features": list(regime_features),
        "n_clusters": n_clusters,
        "clusterer": clusterer,
        "per_symbol_stats": per_symbol_stats,
        "model": model,
        "random_state": random_state,
        "zero_var_eps": zero_var_eps,
    }
    counts = label_series.value_counts().sort_index()
    logger.info(
        "Regime clustering: %d clusters, counts=%s",
        n_clusters,
        counts.to_dict(),
    )
    return label_series, bundle


def assign_regime_labels(df: pd.DataFrame, bundle: RegimeBundle) -> pd.Series:
    """Assign regime labels using a fitted bundle (no refit)."""
    regime_features = bundle["regime_features"]
    X = _transform_with_stats(
        df,
        regime_features,
        bundle["per_symbol_stats"],
        bundle.get("zero_var_eps", config.PHASE1_REGIME_ZERO_VAR_EPS),
    )
    labels = _predict_labels(
        bundle["model"], X, bundle.get("clusterer", "gmm"),
    )
    return pd.Series(labels.astype(np.int32), index=df.index, name="regime")


def persist_regime_model(path: str, bundle: RegimeBundle) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(bundle, path)
    logger.info("Saved regime cluster model to %s", path)


def load_regime_model(path: str) -> RegimeBundle:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Regime model not found: {path}")
    return joblib.load(path)
