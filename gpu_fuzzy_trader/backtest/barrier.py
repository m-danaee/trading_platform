"""Exact first-touch barrier outcomes for prepared market tapes.

The original pipeline stored only the maximum/minimum excursion over the
holding window and a global ``argmax < argmin`` bit.  That is not enough to
know which barrier was hit first for a particular TP/SL pair.  This module
materialises exact gross outcomes and exit offsets for the finite TP/SL grids
used by Phase 2 and the RB risk search.

The columns are internal (their names start with ``_``) and are deliberately
kept separate from evaluator-facing feature columns.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config as _cfg

try:  # Numba is part of the production requirements, but keep a safe fallback.
    from numba import njit
except Exception:  # pragma: no cover - exercised only in minimal installs
    njit = None


def _number_token(value: float) -> str:
    text = format(float(value), ".8g")
    return text.replace("-", "m").replace(".", "p")


def barrier_column_names(direction: str, tp: float, sl: float) -> tuple[str, str]:
    """Return stable internal return/offset column names for one risk pair."""
    direction = str(direction).strip().lower()
    if direction not in {"long", "short"}:
        raise ValueError(f"Unsupported direction: {direction!r}")
    token = f"{_number_token(tp)}_{_number_token(sl)}"
    prefix = f"_barrier_{direction}_tp_{token}"
    return f"{prefix}_return_pct", f"{prefix}_offset"


def configured_barrier_pairs() -> list[tuple[float, float]]:
    """Return the deterministic risk pairs required by all production stages."""
    tps = {float(getattr(_cfg, "PHASE2_TP", 2.0))}
    sls = {float(getattr(_cfg, "PHASE2_SL", 1.2))}
    tps.update(float(v) for v in getattr(_cfg, "RB_TP_GRID", ()))
    sls.update(float(v) for v in getattr(_cfg, "RB_SL_GRID", ()))
    return [(tp, sl) for tp in sorted(tps) for sl in sorted(sls)]


def required_barrier_columns() -> set[str]:
    columns: set[str] = set()
    for direction in ("long", "short"):
        for tp, sl in configured_barrier_pairs():
            columns.update(barrier_column_names(direction, tp, sl))
    return columns


def _first_touch_for_symbol_python(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    horizon: int,
    pairs: list[tuple[float, float]],
    direction: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact gross return and offset arrays for one symbol.

    The implementation intentionally uses a straightforward row/bar loop.
    It runs once per source tape, while all later rule evaluations reuse the
    materialised columns.  A future Numba implementation can replace this
    helper without changing the column contract.
    """
    n = len(opens)
    returns = np.full((n, len(pairs)), np.nan, dtype=np.float32)
    offsets = np.full((n, len(pairs)), -1, dtype=np.int16)
    direction = str(direction).lower()

    for i in range(max(0, n - horizon)):
        entry = float(opens[i + 1])
        if not np.isfinite(entry) or entry <= 0.0:
            continue
        end = i + horizon
        for pair_idx, (tp, sl) in enumerate(pairs):
            exit_return = float("nan")
            exit_offset = horizon
            for offset in range(1, horizon + 1):
                high = float(highs[i + offset])
                low = float(lows[i + offset])
                if direction == "long":
                    hit_tp = high >= entry * (1.0 + tp / 100.0)
                    hit_sl = low <= entry * (1.0 - sl / 100.0)
                    if hit_tp and hit_sl:
                        # OHLC cannot reveal intrabar order.  Conservative
                        # stop-first treatment is deterministic and avoids
                        # rewarding ambiguity.
                        exit_return = -float(sl)
                        exit_offset = offset
                        break
                    if hit_tp:
                        exit_return = float(tp)
                        exit_offset = offset
                        break
                    if hit_sl:
                        exit_return = -float(sl)
                        exit_offset = offset
                        break
                else:
                    hit_tp = low <= entry * (1.0 - tp / 100.0)
                    hit_sl = high >= entry * (1.0 + sl / 100.0)
                    if hit_tp and hit_sl:
                        exit_return = -float(sl)
                        exit_offset = offset
                        break
                    if hit_tp:
                        exit_return = float(tp)
                        exit_offset = offset
                        break
                    if hit_sl:
                        exit_return = -float(sl)
                        exit_offset = offset
                        break

            if not np.isfinite(exit_return):
                close = float(closes[end])
                exit_return = (
                    (close - entry) / entry * 100.0
                    if direction == "long"
                    else (entry - close) / entry * 100.0
                )
            returns[i, pair_idx] = np.float32(exit_return)
            offsets[i, pair_idx] = np.int16(exit_offset)

    return returns, offsets


if njit is not None:

    @njit(cache=True)
    def _first_touch_for_symbol_numba(
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        horizon: int,
        pair_values: np.ndarray,
        is_long: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(opens)
        n_pairs = len(pair_values)
        returns = np.full((n, n_pairs), np.nan, dtype=np.float32)
        offsets = np.full((n, n_pairs), -1, dtype=np.int16)
        limit = n - horizon
        for i in range(max(0, limit)):
            entry = float(opens[i + 1])
            if not np.isfinite(entry) or entry <= 0.0:
                continue
            for pair_idx in range(n_pairs):
                tp = float(pair_values[pair_idx, 0])
                sl = float(pair_values[pair_idx, 1])
                exit_return = np.nan
                exit_offset = horizon
                for offset in range(1, horizon + 1):
                    high = float(highs[i + offset])
                    low = float(lows[i + offset])
                    if is_long == 1:
                        hit_tp = high >= entry * (1.0 + tp / 100.0)
                        hit_sl = low <= entry * (1.0 - sl / 100.0)
                    else:
                        hit_tp = low <= entry * (1.0 - tp / 100.0)
                        hit_sl = high >= entry * (1.0 + sl / 100.0)
                    if hit_tp and hit_sl:
                        exit_return = -sl
                        exit_offset = offset
                        break
                    if hit_tp:
                        exit_return = tp
                        exit_offset = offset
                        break
                    if hit_sl:
                        exit_return = -sl
                        exit_offset = offset
                        break
                if not np.isfinite(exit_return):
                    close = float(closes[i + horizon])
                    if is_long == 1:
                        exit_return = (close - entry) / entry * 100.0
                    else:
                        exit_return = (entry - close) / entry * 100.0
                returns[i, pair_idx] = np.float32(exit_return)
                offsets[i, pair_idx] = np.int16(exit_offset)
        return returns, offsets


def _first_touch_for_symbol(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    horizon: int,
    pairs: list[tuple[float, float]],
    direction: str,
) -> tuple[np.ndarray, np.ndarray]:
    if njit is not None:
        pair_array = np.asarray(pairs, dtype=np.float64)
        return _first_touch_for_symbol_numba(
            opens,
            highs,
            lows,
            closes,
            int(horizon),
            pair_array,
            1 if str(direction).lower() == "long" else 0,
        )
    return _first_touch_for_symbol_python(
        opens, highs, lows, closes, horizon, pairs, direction,
    )


def attach_barrier_outcomes(
    df: pd.DataFrame,
    *,
    horizon: int | None = None,
    pairs: Iterable[tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Attach exact barrier outcomes to a full, chronologically ordered tape.

    The input must still contain the final ``horizon`` rows per symbol.  Those
    rows are later removed by :class:`Data_Loader`, but are necessary to compute
    outcomes for entries near the source-tape boundary.
    """
    required = {"symbol", "datetime", "open", "high", "low", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Barrier outcomes require columns: {sorted(missing)}")

    horizon = int(horizon or getattr(_cfg, "MAX_HOLD_CANDLES", 96))
    if horizon < 1:
        raise ValueError("Barrier horizon must be positive")
    pairs = list(pairs or configured_barrier_pairs())
    if not pairs:
        return df

    out = df.sort_values(["datetime", "symbol"]).reset_index(drop=True).copy()

    # Compute one pair at a time.  Holding a (rows × all-pairs) result for both
    # directions is needlessly expensive on low-RAM hosts, and assigning 160
    # columns one by one fragments the pandas frame.  The temporary column
    # arrays are compact; a single concat creates one contiguous barrier block
    # at the end.
    barrier_columns: dict[str, np.ndarray] = {}
    symbol_groups = [
        (str(symbol), group.sort_values("datetime"))
        for symbol, group in out.groupby(
            "symbol", sort=False, observed=False,
        )
    ]
    for direction in ("long", "short"):
        for tp, sl in pairs:
            ret_name, off_name = barrier_column_names(direction, tp, sl)
            ret_column = np.full(len(out), np.nan, dtype=np.float32)
            off_column = np.full(len(out), -1, dtype=np.int16)
            for symbol, group in symbol_groups:
                ret, off = _first_touch_for_symbol(
                    group["open"].to_numpy(dtype=np.float64),
                    group["high"].to_numpy(dtype=np.float64),
                    group["low"].to_numpy(dtype=np.float64),
                    group["close"].to_numpy(dtype=np.float64),
                    horizon,
                    [(tp, sl)],
                    direction,
                )
                order = group.index.to_numpy()
                ret_column[order] = ret[:, 0]
                off_column[order] = off[:, 0]
            barrier_columns[ret_name] = ret_column
            barrier_columns[off_name] = off_column

    barrier_frame = pd.DataFrame(barrier_columns, index=out.index)
    return pd.concat([out, barrier_frame], axis=1, copy=False)
