"""Forward-window label computation for OHLCV bars.

Horizon is ``TAIL_DROP_ROWS`` (forward window for max/min/close labels).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gpu_fuzzy_trader.config import TAIL_DROP_ROWS

LABEL_COLS = [
    "label_open_next",
    "label_close_288",
    "label_min_288",
    "label_max_288",
    "label_max_before_min",
]


def compute_labels(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 5 label columns per symbol.

    - label_open_next:      open[t+1]
    - label_close_288:      close[t+288]
    - label_min_288:        min(low[t+1 : t+289])
    - label_max_288:        max(high[t+1 : t+289])
    - label_max_before_min: 1 if argmax(high) occurs before argmin(low) in window, else 0

    Rows where the forward window extends beyond the data are set to NaN
    (the loader drops them via ``TAIL_DROP_ROWS``).
    """
    raw = raw.sort_values(["symbol", "datetime"]).reset_index(drop=True)
    parts: list[pd.DataFrame] = []
    for _sym, g in raw.groupby("symbol", sort=True, observed=False):
        g = g.reset_index(drop=True)
        n = len(g)

        o = g["open"].to_numpy()
        hi = g["high"].to_numpy()
        lo = g["low"].to_numpy()
        c = g["close"].to_numpy()

        lab_open = np.full(n, np.nan, dtype=np.float64)
        lab_open[: n - 1] = o[1:]

        lab_close = np.full(n, np.nan, dtype=np.float64)
        lab_close[: n - TAIL_DROP_ROWS] = c[TAIL_DROP_ROWS:]

        lab_min = np.full(n, np.nan, dtype=np.float64)
        if n > TAIL_DROP_ROWS:
            rolling_min = (
                pd.Series(lo)
                .rolling(TAIL_DROP_ROWS, min_periods=TAIL_DROP_ROWS)
                .min()
                .shift(-TAIL_DROP_ROWS)
                .to_numpy()
            )
            lab_min[: n - TAIL_DROP_ROWS] = rolling_min[: n - TAIL_DROP_ROWS]

        lab_max = np.full(n, np.nan, dtype=np.float64)
        if n > TAIL_DROP_ROWS:
            rolling_max = (
                pd.Series(hi)
                .rolling(TAIL_DROP_ROWS, min_periods=TAIL_DROP_ROWS)
                .max()
                .shift(-TAIL_DROP_ROWS)
                .to_numpy()
            )
            lab_max[: n - TAIL_DROP_ROWS] = rolling_max[: n - TAIL_DROP_ROWS]

        lab_mbm = np.full(n, np.nan, dtype=np.float64)
        if n > TAIL_DROP_ROWS:
            from numpy.lib.stride_tricks import sliding_window_view

            hi_windows = sliding_window_view(hi, TAIL_DROP_ROWS)
            lo_windows = sliding_window_view(lo, TAIL_DROP_ROWS)
            valid_rows = n - TAIL_DROP_ROWS
            hi_fwd = hi_windows[1: valid_rows + 1]
            lo_fwd = lo_windows[1: valid_rows + 1]
            argmax_idx = np.argmax(hi_fwd, axis=1)
            argmin_idx = np.argmin(lo_fwd, axis=1)
            lab_mbm[:valid_rows] = (argmax_idx < argmin_idx).astype(float)

        parts.append(
            pd.DataFrame(
                {
                    "datetime": g["datetime"].to_numpy(),
                    "symbol": g["symbol"].to_numpy(),
                    "label_open_next": lab_open,
                    "label_close_288": lab_close,
                    "label_min_288": lab_min,
                    "label_max_288": lab_max,
                    "label_max_before_min": lab_mbm,
                }
            )
        )

    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(["datetime", "symbol"])
        .reset_index(drop=True)
    )
