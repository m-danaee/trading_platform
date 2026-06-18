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
