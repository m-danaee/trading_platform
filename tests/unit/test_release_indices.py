"""Parity between standalone release_index precompute and CPUBacktestEngine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gpu_fuzzy_trader.backtest.cpu_engine import (
    CPUBacktestEngine,
    precompute_release_indices,
)
from gpu_fuzzy_trader import config as _cfg


def _mini_df(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "datetime": pd.date_range("2020-01-01", periods=n, freq="5min"),
        "symbol": ["A"] * (n // 2) + ["B"] * (n - n // 2),
        "label_open_next": rng.uniform(100, 200, n).astype("float32"),
        "label_close_288": rng.uniform(100, 200, n).astype("float32"),
        "label_min_288": rng.uniform(90, 100, n).astype("float32"),
        "label_max_288": rng.uniform(100, 110, n).astype("float32"),
        "label_max_before_min": rng.integers(0, 2, n).astype("float32"),
        "_symbol_bar_index": np.concatenate([
            np.arange(n // 2),
            np.arange(n - n // 2),
        ]),
    })


def test_precompute_release_indices_matches_cpu_engine():
    df = _mini_df()
    engine = CPUBacktestEngine(df, {}, "long")
    sym = df["symbol"].astype(str).values
    bars = df["_symbol_bar_index"].values.astype(int)
    standalone = precompute_release_indices(
        sym, bars, len(df), _cfg.MAX_HOLD_CANDLES
    )
    np.testing.assert_array_equal(standalone, engine.release_index)
