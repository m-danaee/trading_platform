"""Unit tests for backtest DataFrame slimming."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gpu_fuzzy_trader.backtest.df_slim import (
    downcast_numeric_df,
    prune_train_columns,
    slim_backtest_df,
)
from gpu_fuzzy_trader.config import LABEL_COLUMNS, META_COLUMNS


def _wide_df(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = {
        "datetime": pd.date_range("2020-01-01", periods=n, freq="5min"),
        "symbol": ["A"] * n,
        "label_open_next": rng.uniform(100, 200, n),
        "label_close_288": rng.uniform(100, 200, n),
        "label_min_288": rng.uniform(90, 100, n),
        "label_max_288": rng.uniform(100, 110, n),
        "label_max_before_min": rng.integers(0, 2, n),
        "_symbol_bar_index": np.arange(n),
        "feat_a": rng.standard_normal(n),
        "feat_b": rng.standard_normal(n),
        "feat_unused": rng.standard_normal(n),
    }
    return pd.DataFrame(data)


def test_slim_backtest_df_keeps_only_requested_features():
    df = _wide_df()
    slim = slim_backtest_df(df, ["feat_a"])
    assert "feat_a" in slim.columns
    assert "feat_unused" not in slim.columns
    for col in META_COLUMNS + LABEL_COLUMNS:
        if col in df.columns:
            assert col in slim.columns


def test_downcast_numeric_df_uses_float32_labels():
    df = _wide_df()
    slim = downcast_numeric_df(df)
    for col in LABEL_COLUMNS:
        if col in slim.columns:
            assert slim[col].dtype == np.float32


def test_prune_train_columns():
    df = _wide_df()
    pruned = prune_train_columns(df, ["feat_a", "feat_b"])
    assert "feat_unused" not in pruned.columns
    assert "feat_a" in pruned.columns
