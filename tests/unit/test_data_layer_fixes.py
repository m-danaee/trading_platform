"""Unit tests for the bugfixes implemented in loader, labels, and splitter."""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.data.labels import compute_labels
from gpu_fuzzy_trader.data.loader import Data_Loader, load_dataset
from gpu_fuzzy_trader.data.splitter import _holdout_embargo_split_with_mask


def test_labels_edge_cases():
    """Verify compute_labels handles small datasets (n in 0, 1, 49, 80, 95, 96, 97) without crashing."""
    for n in [0, 1, 49, 80, 95, 96, 97]:
        df = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01", periods=n, freq="15min"),
            "symbol": ["BTCUSDT"] * n,
            "open": np.linspace(100, 200, n) if n > 0 else [],
            "high": np.linspace(105, 205, n) if n > 0 else [],
            "low": np.linspace(95, 195, n) if n > 0 else [],
            "close": np.linspace(102, 202, n) if n > 0 else [],
            "volume": np.linspace(10, 20, n) if n > 0 else [],
        })
        res = compute_labels(df)
        assert len(res) == n
        if n > 0:
            assert res["label_close_288"].isna().all() if n <= _cfg.TAIL_DROP_ROWS else not res["label_close_288"].iloc[:n - _cfg.TAIL_DROP_ROWS].isna().any()


def test_loader_drop_tail_preservation():
    """Verify drop_tail=False preserves full tape including tail rows."""
    n = 200
    df = pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=n, freq="15min"),
        "symbol": ["BTCUSDT"] * n,
        "open": np.linspace(100, 200, n),
        "high": np.linspace(105, 205, n),
        "low": np.linspace(95, 195, n),
        "close": np.linspace(102, 202, n),
        "volume": np.linspace(10, 20, n),
        "ff_feature": np.random.randn(n),
    })
    df.loc[10:15, "ff_feature"] = np.nan
    labels = compute_labels(df)
    for col in labels.columns:
        if col not in df.columns:
            df[col] = labels[col]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        df.to_csv(f, index=False)
        csv_path = f.name

    try:
        loader = Data_Loader()
        loaded_drop_true = loader.load_dataset(csv_path, drop_tail=True, require_context=False)
        loaded_drop_false = loader.load_dataset(csv_path, drop_tail=False, require_context=False)

        assert len(loaded_drop_true) == 200 - _cfg.TAIL_DROP_ROWS
        assert len(loaded_drop_false) == 200
        # Check feature NaN is preserved
        assert loaded_drop_false["ff_feature"].isna().sum() == 6
    finally:
        os.unlink(csv_path)


def test_splitter_holdout_mask():
    """Verify _holdout_embargo_split_with_mask returns correct boolean masks."""
    n = 300
    df = pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=n, freq="15min"),
        "symbol": ["BTCUSDT"] * n,
        "feat": np.arange(n),
    })
    full_df, train_mask, embargo_mask, val_mask = _holdout_embargo_split_with_mask(df)
    assert len(full_df) == n
    train_count = _cfg.train_prefix_row_count(n)
    embargo_count = min(int(_cfg.HOLDOUT_EMBARGO_CANDLES), n - train_count)
    val_count = max(0, n - (train_count + embargo_count))

    assert train_mask.sum() == train_count
    assert embargo_mask.sum() == embargo_count
    assert val_mask.sum() == val_count
    assert (train_mask & embargo_mask).sum() == 0
    assert (train_mask & val_mask).sum() == 0
    assert (embargo_mask & val_mask).sum() == 0
    assert (train_mask | embargo_mask | val_mask).sum() == n
