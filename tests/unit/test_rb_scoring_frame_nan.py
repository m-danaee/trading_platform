"""RB scoring-frame NaN fill must honor FILL_NA_WITH_ZERO."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.rb_governor import _prepare_scoring_frame


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=3, freq="15min"),
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "open": [100.0, np.nan, 102.0],
            "high": [101.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 102.0, 103.0],
            "volume": [10.0, 11.0, 12.0],
            "label_open_next": [100.5, 102.0, 103.0],
            "label_close_288": [101.0, 103.0, 104.0],
            "label_min_288": [99.0, 100.0, 101.0],
            "label_max_288": [102.0, 104.0, 105.0],
            "label_max_before_min": [1, 1, 1],
            "ff_rsi": [0.2, np.nan, 0.4],
            "_barrier_long_tp_2p0_1p2_return_pct": [0.5, np.nan, -0.2],
        }
    )


def test_prepare_scoring_frame_keeps_warmup_and_internal_nans(monkeypatch) -> None:
    monkeypatch.setattr(_cfg, "FILL_NA_WITH_ZERO", False)
    out = _prepare_scoring_frame(_frame())
    assert pd.isna(out["ff_rsi"].iloc[1])
    assert pd.isna(out["open"].iloc[1])
    assert pd.isna(out["_barrier_long_tp_2p0_1p2_return_pct"].iloc[1])


def test_prepare_scoring_frame_fills_features_only_when_enabled(monkeypatch) -> None:
    monkeypatch.setattr(_cfg, "FILL_NA_WITH_ZERO", True)
    out = _prepare_scoring_frame(_frame())
    assert float(out["ff_rsi"].iloc[1]) == 0.0
    assert pd.isna(out["open"].iloc[1])
    assert pd.isna(out["_barrier_long_tp_2p0_1p2_return_pct"].iloc[1])
