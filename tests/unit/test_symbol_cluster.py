"""Unit tests for hybrid symbol clustering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gpu_fuzzy_trader.features.symbol_cluster import build_hybrid_symbol_clusters


def _make_train_df(symbols: list[str], rows: int = 120) -> pd.DataFrame:
    parts = []
    for sym in symbols:
        parts.append(pd.DataFrame({
            "symbol": [sym] * rows,
            "datetime": pd.date_range("2024-01-01", periods=rows, freq="5min"),
            "label_open_next": np.linspace(100, 110, rows) + hash(sym) % 5,
            "feat_a": np.random.default_rng(abs(hash(sym)) % (2**32)).random(rows),
            "feat_b": np.random.default_rng((abs(hash(sym)) + 1) % (2**32)).random(rows),
        }))
    return pd.concat(parts, ignore_index=True)


def test_hybrid_cluster_covers_all_symbols():
    df = _make_train_df(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
    feat = [{"name": "feat_a", "mode": "positive", "score": 1.0},
            {"name": "feat_b", "mode": "positive", "score": 0.5}]
    payload = build_hybrid_symbol_clusters(
        df, feat, feat, n_clusters=4, random_state=42,
    )
    assigned = {s for syms in payload["clusters"].values() for s in syms}
    assert assigned == set(payload["symbols"])
    assert len(payload["clusters"]) <= 4


def test_single_symbol_skips_clustering():
    df = _make_train_df(["1"], rows=50)
    feat = [{"name": "feat_a", "mode": "positive", "score": 1.0}]
    payload = build_hybrid_symbol_clusters(df, feat, feat, n_clusters=4)
    assert payload["clusters"] == {"0": ["1"]}


def test_balanced_10_symbols_k3_no_small_cluster():
    """Fix E1: With 10 symbols and K=3, no cluster has fewer than 3 symbols."""
    symbols = [str(i) for i in range(1, 11)]
    df = _make_train_df(symbols, rows=120)
    feat = [{"name": "feat_a", "mode": "positive", "score": 1.0},
            {"name": "feat_b", "mode": "positive", "score": 0.5}]

    # Test with several seeds to ensure balance is robust
    for seed in [0, 1, 42, 123, 999]:
        payload = build_hybrid_symbol_clusters(
            df, feat, feat, n_clusters=3, random_state=seed, balance=True,
        )
        # All symbols covered
        assigned = {s for syms in payload["clusters"].values() for s in syms}
        assert assigned == set(payload["symbols"]), f"seed={seed}: missing symbols"
        # Cluster counts
        sizes = [len(v) for v in payload["clusters"].values()]
        assert all(s >= 3 for s in sizes), (
            f"seed={seed}: cluster sizes {sizes} have a cluster < 3"
        )
        assert sum(sizes) == 10, f"seed={seed}: total symbols {sum(sizes)} != 10"
        # Payload shape unchanged
        assert set(payload.keys()) == {"clusters", "method", "n_clusters", "symbols"}
        assert payload["method"] == "hybrid_v1"


def test_balanced_k_equals_n_symbols():
    """When k == n_symbols, each symbol gets its own cluster (unchanged path)."""
    symbols = [str(i) for i in range(1, 6)]
    df = _make_train_df(symbols, rows=60)
    feat = [{"name": "feat_a", "mode": "positive", "score": 1.0}]
    payload = build_hybrid_symbol_clusters(
        df, feat, feat, n_clusters=5, random_state=42, balance=True,
    )
    for cid, syms in payload["clusters"].items():
        assert len(syms) == 1, f"cluster {cid} has {len(syms)} symbols, expected 1"


def test_balance_disabled_falls_back_to_kmeans():
    """When balance=False, assignment follows raw KMeans labels."""
    symbols = [str(i) for i in range(1, 11)]
    df = _make_train_df(symbols, rows=120)
    feat = [{"name": "feat_a", "mode": "positive", "score": 1.0},
            {"name": "feat_b", "mode": "positive", "score": 0.5}]
    # KMeans with a fixed seed may produce imbalanced clusters without balancing
    payload = build_hybrid_symbol_clusters(
        df, feat, feat, n_clusters=3, random_state=42, balance=False,
    )
    # All symbols still covered
    assigned = {s for syms in payload["clusters"].values() for s in syms}
    assert assigned == set(payload["symbols"])
    # Payload shape unchanged
    assert set(payload.keys()) == {"clusters", "method", "n_clusters", "symbols"}


def test_corr_hybrid_path_uses_hybrid_corr_v1_method(monkeypatch):
    """When PHASE2_CLUSTER_USE_RETURN_CORR=True, method tag is hybrid_corr_v1."""
    from gpu_fuzzy_trader import config
    monkeypatch.setattr(config, "PHASE2_CLUSTER_USE_RETURN_CORR", True)
    monkeypatch.setattr(config, "PHASE2_CLUSTER_FEATURE_WEIGHT", 0.5)
    monkeypatch.setattr(config, "PHASE2_CLUSTER_CORR_WEIGHT", 0.5)

    symbols = [str(i) for i in range(1, 10)]
    df = _make_train_df(symbols, rows=120)
    feat = [{"name": "feat_a", "mode": "positive", "score": 1.0},
            {"name": "feat_b", "mode": "positive", "score": 0.5}]
    payload = build_hybrid_symbol_clusters(
        df, feat, feat, n_clusters=3, random_state=42, balance=True,
    )
    assert payload["method"] == "hybrid_corr_v1", (
        f"Expected hybrid_corr_v1, got {payload['method']}"
    )
    # All symbols covered
    assigned = {s for syms in payload["clusters"].values() for s in syms}
    assert assigned == set(payload["symbols"])
    # Balanced: with 9 symbols and K=3, no cluster smaller than 3
    sizes = [len(v) for v in payload["clusters"].values()]
    assert all(s >= 3 for s in sizes), (
        f"Cluster sizes {sizes} have a cluster < 3"
    )
    assert sum(sizes) == 9


def test_corr_hybrid_path_fallback_when_no_price_col(monkeypatch):
    """When the DataFrame has no price-like columns, corr path still works
    (falls back to first numeric column)."""
    from gpu_fuzzy_trader import config
    monkeypatch.setattr(config, "PHASE2_CLUSTER_USE_RETURN_CORR", True)
    monkeypatch.setattr(config, "PHASE2_CLUSTER_FEATURE_WEIGHT", 0.5)
    monkeypatch.setattr(config, "PHASE2_CLUSTER_CORR_WEIGHT", 0.5)

    # Build a DataFrame without label_open_next or close columns
    parts = []
    for sym in ["1", "2", "3"]:
        parts.append(pd.DataFrame({
            "symbol": [sym] * 50,
            "feat_a": np.random.default_rng(abs(hash(sym)) % (2**32)).random(50),
        }))
    df = pd.concat(parts, ignore_index=True)
    feat = [{"name": "feat_a", "mode": "positive", "score": 1.0}]
    payload = build_hybrid_symbol_clusters(
        df, feat, feat, n_clusters=2, random_state=42, balance=True,
    )
    assert payload["method"] == "hybrid_corr_v1"
    assigned = {s for syms in payload["clusters"].values() for s in syms}
    assert assigned == {"1", "2", "3"}
