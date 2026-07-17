"""Unit tests for forward-window label semantics (``gpu_fuzzy_trader.data.labels``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gpu_fuzzy_trader.config import TAIL_DROP_ROWS
from gpu_fuzzy_trader.data.labels import compute_labels


def _raw_from_ohlc(
    symbol: str,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> pd.DataFrame:
    n = len(open_)
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=n, freq="5min"),
            "symbol": [symbol] * n,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
        }
    )


class TestForwardWindowLabels:
    def test_label_max_288_uses_future_highs_only(self):
        n = TAIL_DROP_ROWS + 20
        hi = np.linspace(100.0, 100.0 + n - 1, n)
        hi[50] = 999.0
        raw = _raw_from_ohlc(
            "A",
            open_=hi,
            high=hi,
            low=hi - 1.0,
            close=hi - 0.5,
        )
        labels = compute_labels(raw)
        t = 0
        expected = float(np.max(hi[t + 1: t + 1 + TAIL_DROP_ROWS]))
        actual = float(labels.loc[labels.index[0], "label_max_288"])
        assert actual == expected
        assert actual == 999.0

    def test_label_min_288_uses_future_lows_only(self):
        n = TAIL_DROP_ROWS + 20
        lo = np.linspace(100.0, 100.0 + n - 1, n)
        lo[50] = 1.0
        raw = _raw_from_ohlc(
            "A",
            open_=lo + 1.0,
            high=lo + 2.0,
            low=lo,
            close=lo + 0.5,
        )
        labels = compute_labels(raw)
        t = 0
        expected = float(np.min(lo[t + 1: t + 1 + TAIL_DROP_ROWS]))
        actual = float(labels.loc[labels.index[0], "label_min_288"])
        assert actual == expected
        assert actual == 1.0

    def test_label_max_before_min_forward_window(self):
        n = TAIL_DROP_ROWS + 10
        hi = np.full(n, 50.0)
        lo = np.full(n, 40.0)
        hi[3] = 100.0
        lo[8] = 1.0
        raw = _raw_from_ohlc("A", hi, hi, lo, (hi + lo) / 2.0)
        labels = compute_labels(raw)
        t = 0
        fwd_hi = hi[t + 1: t + 1 + TAIL_DROP_ROWS]
        fwd_lo = lo[t + 1: t + 1 + TAIL_DROP_ROWS]
        expected = 1.0 if int(np.argmax(fwd_hi)) < int(
            np.argmin(fwd_lo)) else 0.0
        actual = float(labels.loc[labels.index[0], "label_max_before_min"])
        assert actual == expected

    def test_tail_rows_are_nan(self):
        n = TAIL_DROP_ROWS + 5
        px = np.arange(n, dtype=float) + 100.0
        raw = _raw_from_ohlc("A", px, px + 1.0, px - 1.0, px)
        labels = compute_labels(raw)
        sym = labels[labels["symbol"] == "A"]
        tail = sym.tail(TAIL_DROP_ROWS)
        for col in (
            "label_close_288",
            "label_min_288",
            "label_max_288",
            "label_max_before_min",
        ):
            assert tail[col].isna().all(), col

    def test_not_equal_to_backward_shifted_bug(self):
        """Regression: old backward rolling must not match corrected forward labels."""
        n = TAIL_DROP_ROWS + 50
        rng = np.random.default_rng(0)
        hi = 100.0 + np.cumsum(rng.normal(0.0, 0.2, n))
        lo = hi - rng.uniform(0.5, 2.0, n)
        raw = _raw_from_ohlc("A", hi, hi, lo, (hi + lo) / 2.0)

        labels = compute_labels(raw)
        hi_shifted = np.roll(hi, -1)
        hi_shifted[-1] = np.nan
        buggy = (
            pd.Series(hi_shifted)
            .rolling(TAIL_DROP_ROWS, min_periods=TAIL_DROP_ROWS)
            .max()
            .to_numpy()
        )
        t = 10
        fixed = float(labels.loc[t, "label_max_288"])
        assert fixed == float(np.max(hi[t + 1: t + 1 + TAIL_DROP_ROWS]))
        if not np.isnan(buggy[t]):
            assert fixed != float(buggy[t]) or np.allclose(
                hi[t + 1: t + 1 + TAIL_DROP_ROWS],
                hi[max(0, t - TAIL_DROP_ROWS + 1): t + 2],
            )
