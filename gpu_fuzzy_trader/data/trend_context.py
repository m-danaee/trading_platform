"""trend_context.py — deterministic causal HWC/MWC/LWC enrichment.

Generates enriched research CSVs from raw OHLCV tapes without overwriting the
raw source.  Implements the frozen four-state regime contract:

    -1 = Bearish
     0 = Range
     1 = Bullish
     2 = Noisy

Semantics
---------
- CSV timestamps are 15m bar-open times.
- A 15m candle opened at 10:45 closes at 11:00 and can affect the 11:00 entry
  (the pipeline executes entries at ``label_open_next`` = next bar's open).
- A 1h candle opened at 10:00 and closed at 11:00 can affect the 11:00 entry.
- A 4h candle opened at 08:00 and closed at 12:00 can first affect 12:00.
- Higher-timeframe state is only published *after* that bar completes, and is
  aligned back to 15m rows with backward-causal semantics.
- Incomplete higher-timeframe candles never affect an earlier entry.

Thresholds are fitted once from the Phase-0 training prefix only, then frozen,
independently per timeframe (LWC/MWC/HWC each get their own threshold set
fitted from that timeframe's own bar distribution).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config as _cfg

logger = logging.getLogger(__name__)

# Fixed, documented state codes.
STATE_BEARISH = _cfg.CONTEXT_STATE_CODES["bearish"]  # -1
STATE_RANGE = _cfg.CONTEXT_STATE_CODES["range"]  # 0
STATE_BULLISH = _cfg.CONTEXT_STATE_CODES["bullish"]  # 1
STATE_NOISY = _cfg.CONTEXT_STATE_CODES["noisy"]  # 2

CONTEXT_OUTPUT_COLUMNS = (
    "hwc_state",
    "mwc_state",
    "lwc_state",
    "tf_permission_long",
    "tf_permission_short",
    "lwc_pullback_reversal_long",
    "lwc_pullback_reversal_short",
)

_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
_MINUTE = pd.Timedelta(minutes=1)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_input_frame(df: pd.DataFrame) -> None:
    """Validate the raw 15m tape contract.

    Raises ``ValueError`` on:
    - missing datetime / symbol columns
    - duplicated ``(datetime, symbol)`` rows
    - irregular / ambiguous timestamps (must be a closed 15m grid per symbol)
    """
    missing = [c for c in ("datetime", "symbol") if c not in df.columns]
    if missing:
        raise ValueError(
            f"Raw tape must contain {missing}; got columns {list(df.columns)}")

    if df["datetime"].isna().any():
        raise ValueError("Raw tape contains NaN datetimes")
    dupes = df.duplicated(subset=["datetime", "symbol"])
    if dupes.any():
        n_dupes = int(dupes.sum())
        preview = df.loc[dupes, ["datetime", "symbol"]].head(5)
        raise ValueError(
            f"Raw tape must have unique (datetime, symbol) rows; found "
            f"{n_dupes} duplicates, e.g. {preview.to_dict('records')}")

    grid = pd.Timedelta(minutes=int(_cfg.LWC_TIMEFRAME_MINUTES))
    for symbol, group in df.groupby("symbol", sort=False, observed=False):
        g = group.sort_values("datetime").reset_index(drop=True)
        dt = g["datetime"]
        diffs = dt.diff().iloc[1:]
        if not pd.api.types.is_datetime64_any_dtype(dt):
            raise ValueError(
                f"Symbol {symbol!r}: datetime column must be datetime, got "
                f"{dt.dtype!r}")
        bad = diffs[diffs != grid]
        if len(bad):
            preview = g.loc[bad.index, ["datetime"]].head(5)
            raise ValueError(
                f"Symbol {symbol!r}: timestamps must form a regular 15m grid; "
                f"found {len(bad)} irregular gaps, e.g. "
                f"{preview['datetime'].tolist()}")
        # Bar-open timestamps must be exactly aligned to the 15m grid
        # (:00/:15/:30/:45 with zero seconds and sub-second components).
        # Regular spacing alone is not enough: a sequence at 10:44/10:59/11:14
        # is 15 minutes apart yet off-grid.
        minute = dt.dt.minute.to_numpy()
        second = dt.dt.second.to_numpy()
        microsecond = dt.dt.microsecond.to_numpy()
        off_grid = (minute % 15 != 0) | (second != 0) | (microsecond != 0)
        if off_grid.any():
            preview = g.loc[np.flatnonzero(off_grid), ["datetime"]].head(5)
            raise ValueError(
                f"Symbol {symbol!r}: bar-open timestamps must be aligned to "
                f"the 15m grid (minute % 15 == 0, second == 0, "
                f"microsecond == 0); found off-grid timestamps, e.g. "
                f"{preview['datetime'].tolist()}")
        if g["datetime"].dt.tz is not None:
            raise ValueError(
                f"Symbol {symbol!r}: bar-open timestamps must be timezone-naive")


# ---------------------------------------------------------------------------
# Structural inputs (per completed bar)
# ---------------------------------------------------------------------------

def signed_price_efficiency(close: pd.Series, lookback: int) -> pd.Series:
    """Signed price efficiency: directional displacement / total movement.

    Positive → up-trend displacement; near zero → choppy/range.
    NaN for the first ``lookback`` bars (warm-up).
    """
    close = close.astype(float)
    if lookback < 1 or len(close) <= lookback:
        return pd.Series(np.full(len(close), np.nan), index=close.index)
    displacement = close.diff(lookback)
    abs_diff = close.diff().abs()
    total = abs_diff.rolling(lookback, min_periods=lookback).sum()
    out = displacement / total
    return out.replace([np.inf, -np.inf], np.nan)


def average_true_range(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int,
) -> pd.Series:
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def normalized_ema_spread(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    span_fast: int,
    span_slow: int,
    atr_period: int,
) -> pd.Series:
    """EMA spread normalized by ATR: trend direction and structural separation."""
    fast = close.ewm(span=span_fast, adjust=False).mean()
    slow = close.ewm(span=span_slow, adjust=False).mean()
    atr = average_true_range(high, low, close, atr_period)
    spread = (fast - slow) / atr.replace(0.0, np.nan)
    return spread.replace([np.inf, -np.inf], np.nan)


def realized_volatility(close: pd.Series, window: int) -> pd.Series:
    """Realized volatility: rolling std of log returns (disorder detection)."""
    logret = np.log(close.astype(float) / close.astype(float).shift(1))
    return logret.rolling(window, min_periods=window).std()


def structural_inputs(
    df: pd.DataFrame,
    *,
    lookback: int | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute the three structural inputs on completed bars (per symbol)."""
    lookback = int(_cfg.CONTEXT_STRUCTURAL_LOOKBACK) if lookback is None else int(lookback)
    vol_window = max(1, int(_cfg.CONTEXT_VOL_WINDOW))
    eff: list[pd.Series] = []
    spread: list[pd.Series] = []
    rv: list[pd.Series] = []
    for _sym, g in df.sort_values("datetime").groupby(
        "symbol", sort=False, observed=False,
    ):
        g = g.sort_values("datetime")
        span_fast = max(2, lookback // 2)
        span_slow = max(4, lookback)
        e = signed_price_efficiency(g["close"], lookback)
        s = normalized_ema_spread(
            g["close"], g["high"], g["low"], span_fast, span_slow, lookback,
        )
        r = realized_volatility(g["close"], min(vol_window, lookback))
        e.index = g.index
        s.index = g.index
        r.index = g.index
        eff.append(e)
        spread.append(s)
        rv.append(r)
    return (
        pd.concat(eff).reindex(df.index),
        pd.concat(spread).reindex(df.index),
        pd.concat(rv).reindex(df.index),
    )


# ---------------------------------------------------------------------------
# Four-state classification
# ---------------------------------------------------------------------------

def classify_regime(
    eff: np.ndarray,
    spread: np.ndarray,
    rv: np.ndarray,
    threshold_eff: float,
    threshold_spread: float,
    threshold_rv: float,
) -> np.ndarray:
    """Classify each row into one of the four fixed state codes.

    Order of precedence (contract):
      1. Bullish:  eff >= +trend threshold AND spread >= +spread threshold
      2. Bearish:  eff <= -trend threshold AND spread <= -spread threshold
      3. Range:    |eff| weak AND rv below compression threshold
      4. Noisy:    everything else (including unavailable / warm-up / NaN)
    """
    eff = np.asarray(eff, dtype=float)
    spread = np.asarray(spread, dtype=float)
    rv = np.asarray(rv, dtype=float)
    state = np.full(len(eff), STATE_NOISY, dtype=np.int8)

    bullish = (eff >= threshold_eff) & (spread >= threshold_spread)
    bearish = (eff <= -threshold_eff) & (spread <= -threshold_spread)
    rng = (np.abs(eff) < threshold_eff) & (rv < threshold_rv)
    # Guard against a non-positive compression threshold making Range
    # impossible or overlapping Bearish.
    rng &= rv >= 0.0

    state[bullish] = STATE_BULLISH
    state[bearish] = STATE_BEARISH
    state[rng] = STATE_RANGE
    return state


def fit_thresholds(
    train_prefix: pd.DataFrame,
    *,
    eff_quantile: float | None = None,
    spread_quantile: float | None = None,
    rv_quantile: float | None = None,
    lookback: int | None = None,
) -> dict[str, float]:
    """Fit train-only pooled percentile thresholds from the Phase-0 prefix.

    The quantiles are frozen before validation/test/forward results are
    reviewed and must never be refitted outside the training prefix.
    """
    eff, spread, rv = structural_inputs(train_prefix, lookback=lookback)
    eff_abs = np.abs(eff.to_numpy(dtype=float))
    spread_abs = np.abs(spread.to_numpy(dtype=float))
    rv_np = rv.to_numpy(dtype=float)
    eff_abs = eff_abs[np.isfinite(eff_abs)]
    spread_abs = spread_abs[np.isfinite(spread_abs)]
    rv_np = rv_np[np.isfinite(rv_np)]

    eff_quantile = (
        float(_cfg.CONTEXT_EFFICIENCY_TREND_THRESHOLD_QUANTILE)
        if eff_quantile is None else float(eff_quantile))
    spread_quantile = (
        float(_cfg.CONTEXT_EMA_SPREAD_TREND_THRESHOLD_QUANTILE)
        if spread_quantile is None else float(spread_quantile))
    rv_quantile = (
        float(_cfg.CONTEXT_VOLATILITY_COMPRESSION_QUANTILE)
        if rv_quantile is None else float(rv_quantile))

    if eff_abs.size == 0 or spread_abs.size == 0 or rv_np.size == 0:
        raise ValueError(
            "Cannot fit trend-context thresholds: no completed bars in the "
            "training prefix (need at least CONTEXT_STRUCTURAL_LOOKBACK rows "
            "per symbol).")

    return {
        "efficiency_abs_trend_threshold": float(np.quantile(eff_abs, eff_quantile)),
        "ema_spread_abs_trend_threshold": float(np.quantile(spread_abs, spread_quantile)),
        "volatility_compression_threshold": float(np.quantile(rv_np, rv_quantile)),
        "eff_quantile": eff_quantile,
        "spread_quantile": spread_quantile,
        "rv_quantile": rv_quantile,
    }


def build_train_prefix(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Return only the per-symbol leading rows in the Phase-0 training prefix.

    Uses the exact same per-symbol row boundary as ``Data_Splitter``
    (``config.train_prefix_row_count``), so trend-context thresholds are
    fitted strictly before the rows that later become the validation split
    and never leak validation-period price distributions into the regime
    classifier.
    """
    parts: list[pd.DataFrame] = []
    for _symbol, g in raw_df.groupby("symbol", sort=True, observed=False):
        g = g.sort_values("datetime")
        train_end = _cfg.train_prefix_row_count(len(g))
        parts.append(g.iloc[:train_end])
    if not parts:
        return raw_df.iloc[0:0]
    return pd.concat(parts, ignore_index=True)


def fit_all_thresholds(train_prefix: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Fit independent, frozen thresholds per timeframe from the train prefix.

    Realized volatility, efficiency, and EMA-spread distributions all shift
    materially with bar duration, so a single 15m-fitted threshold set must
    never be reused for 1h/4h classification (it can make higher-timeframe
    Range nearly impossible or produce badly unbalanced Noisy coverage).
    Each timeframe is fitted from its own *complete* bars (see
    :func:`build_higher_bars`) built from the same train-only prefix.
    """
    mwc_bars = build_higher_bars(train_prefix, int(_cfg.MWC_TIMEFRAME_MINUTES))
    hwc_bars = build_higher_bars(train_prefix, int(_cfg.HWC_TIMEFRAME_MINUTES))
    return {
        "lwc": fit_thresholds(train_prefix),
        "mwc": fit_thresholds(mwc_bars),
        "hwc": fit_thresholds(hwc_bars),
    }


# ---------------------------------------------------------------------------
# Higher-timeframe bars and causal alignment
# ---------------------------------------------------------------------------

def build_higher_bars(
    df: pd.DataFrame, timeframe_minutes: int, *, require_complete: bool = True,
) -> pd.DataFrame:
    """Aggregate 15m rows into independent per-symbol higher-timeframe bars.

    Returns a frame with columns ``datetime`` (bar-open time), ``symbol``,
    ``open``, ``high``, ``low``, ``close``, ``volume``.

    When ``require_complete`` (default), a bucket is only kept when it
    contains exactly ``timeframe_minutes // LWC_TIMEFRAME_MINUTES`` 15m rows.
    A tape that does not start/end on a higher-timeframe boundary otherwise
    produces a partial leading or trailing bucket (e.g. a tape starting at
    05:00 has only 12 of the 16 15m rows needed for the 04:00-08:00 4h
    bucket); treating it as a complete bar would corrupt the early rolling
    structural indicators and any threshold fitted from it.
    """
    tf = pd.Timedelta(minutes=int(timeframe_minutes))
    expected_rows = int(timeframe_minutes) // int(_cfg.LWC_TIMEFRAME_MINUTES)
    parts: list[pd.DataFrame] = []
    for symbol, g in df.sort_values("datetime").groupby(
        "symbol", sort=False, observed=False,
    ):
        g = g.sort_values("datetime")
        bucket = g["datetime"].dt.floor(tf)
        agg = g.groupby(bucket, sort=True).agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        if require_complete:
            complete = bucket.value_counts() == expected_rows
            agg = agg.loc[agg.index.map(complete).fillna(False)]
        out = agg.reset_index(drop=False)
        out = out.rename(columns={out.columns[0]: "datetime"})
        out["symbol"] = symbol
        parts.append(out)
    if not parts:
        return pd.DataFrame(
            columns=["datetime", "symbol", *_OHLCV_COLUMNS],
        )
    return pd.concat(parts, ignore_index=True).sort_values(
        ["datetime", "symbol"]).reset_index(drop=True)


def align_completed_states_to_rows(
    rows: pd.DataFrame,
    hf_bars: pd.DataFrame,
    thresholds: dict[str, float],
    timeframe_minutes: int,
) -> pd.Series:
    """Publish a higher-timeframe state only after that bar completes.

    Each row with bar-open time ``D`` is a signal row whose completed 15m bar
    executes at ``label_open_next`` at ``D + 15m``.  The aligned state is
    therefore the most recent higher-timeframe bar whose close time is
    ``<= D + 15m``.  A bar opened at ``T`` closes at ``T + timeframe``.
    Rows before the first completed bar are *unavailable* → Noisy.

    ``hf_bars`` must be the per-symbol higher-timeframe bars built from the
    same rows (see :func:`build_higher_bars`).
    """
    tf = pd.Timedelta(minutes=int(timeframe_minutes))
    close_times = hf_bars["datetime"] + tf
    hf = hf_bars.assign(_close_time=close_times)
    hf_state_by_symbol: dict[object, np.ndarray] = {}

    for symbol, g in hf.sort_values("_close_time").groupby(
        "symbol", sort=False, observed=False,
    ):
        close_arr = g["_close_time"].to_numpy()
        state_arr = _classify_hf_bars(g, thresholds)
        hf_state_by_symbol[symbol] = (close_arr, state_arr)

    aligned = pd.Series(
        np.full(len(rows), STATE_NOISY, dtype=np.int8), index=rows.index,
    )
    for symbol, g in rows.groupby("symbol", sort=False, observed=False):
        if symbol not in hf_state_by_symbol:
            continue
        close_arr, state_arr = hf_state_by_symbol[symbol]
        close_ns = close_arr.astype("int64")
        execution_times = (
            g["datetime"]
            + pd.Timedelta(minutes=int(_cfg.LWC_TIMEFRAME_MINUTES))
        )
        execution_ns = execution_times.to_numpy().astype("int64")
        # The signal on row D executes at the next 15m open.  Use every HTF
        # candle completed by that execution instant, never an incomplete one.
        idx = np.searchsorted(close_ns, execution_ns, side="right") - 1
        out = np.full(len(execution_ns), STATE_NOISY, dtype=np.int8)
        valid = idx >= 0
        out[valid] = state_arr[np.clip(idx[valid], 0, len(state_arr) - 1)]
        aligned.loc[g.index] = out
    return aligned


def _classify_hf_bars(hf: pd.DataFrame, thresholds: dict[str, float]) -> np.ndarray:
    """Classify aggregated higher-timeframe bars into state codes."""
    lookback = max(1, int(_cfg.CONTEXT_STRUCTURAL_LOOKBACK))
    span_fast = max(2, lookback // 2)
    span_slow = max(4, lookback)
    vol_window = max(1, int(_cfg.CONTEXT_VOL_WINDOW))
    eff = signed_price_efficiency(hf["close"], lookback).to_numpy()
    spread = normalized_ema_spread(
        hf["close"], hf["high"], hf["low"], span_fast, span_slow, lookback,
    ).to_numpy()
    rv = realized_volatility(hf["close"], min(vol_window, lookback)).to_numpy()
    return classify_regime(
        eff,
        spread,
        rv,
        thresholds["efficiency_abs_trend_threshold"],
        thresholds["ema_spread_abs_trend_threshold"],
        thresholds["volatility_compression_threshold"],
    )


# ---------------------------------------------------------------------------
# Permissions and LWC pullback-reversal triggers
# ---------------------------------------------------------------------------

def _prior_window_count(mark: np.ndarray, lookback: int) -> np.ndarray:
    """Count True occurrences of ``mark`` strictly before each position
    within the preceding ``lookback`` completed bars.  Per block, so it never
    crosses a symbol boundary.  ``mark`` is an arbitrary boolean condition
    supplied by the caller (e.g. "LWC was Bearish AND long permission was
    active"), not merely a raw state match."""
    rolled = pd.Series(mark.astype(np.int8)).rolling(lookback, min_periods=1).sum()
    prior = rolled.shift(1).fillna(0).to_numpy()
    return prior


def compute_permissions_and_triggers(
    hwc: pd.Series,
    mwc: pd.Series,
    lwc: pd.Series,
    symbols: pd.Series,
    lookback: int | None = None,
) -> pd.DataFrame:
    """Return the four execution columns.

    - ``tf_permission_long``  = hwc Bullish AND (mwc Bullish OR Range)
    - ``tf_permission_short`` = hwc Bearish AND (mwc Bearish OR Range)
    - ``lwc_pullback_reversal_long``  = current LWC Bullish AND a completed
      LWC Bearish print occurred within the previous ``lookback`` bars.
    - ``lwc_pullback_reversal_short`` = current LWC Bearish AND a completed
      LWC Bullish print occurred within the previous ``lookback`` bars.

    The trigger is a pure LTF-timing signal: it is intentionally NOT ANDed
    with the current-row permission, so ``permission_only``/``trigger_only``
    coverage diagnostics are meaningful rather than structurally impossible.
    Callers requiring a tradeable signal must AND both
    ``tf_permission_<dir>`` and ``lwc_pullback_reversal_<dir>`` (this is the
    mandatory-condition contract enforced elsewhere).

    When ``CONTEXT_REQUIRE_PERMISSION_ON_PULLBACK_PRINT`` is True, the
    historical opposite-LWC print only counts if same-direction permission was
    also active on that bar (stricter, can starve Phase 2 coverage). The
    default is False: any opposite LWC print in the lookback window counts.

    A current Noisy state never triggers.  MWC Range is treated as a valid
    consolidation while HWC retains direction; the policy is configurable via
    ``CONTEXT_ALLOW_MWC_RANGE_PERMISSION``.  Per-symbol: the pullback window
    is computed independently for each symbol, regardless of row ordering
    (symbols may be interleaved).
    """
    lookback = int(_cfg.LWC_PULLBACK_LOOKBACK) if lookback is None else int(lookback)
    n = len(lwc)
    hwc_np = np.asarray(hwc, dtype=np.int8)
    mwc_np = np.asarray(mwc, dtype=np.int8)
    lwc_np = np.asarray(lwc, dtype=np.int8)
    trig_long = np.zeros(n, dtype=np.int8)
    trig_short = np.zeros(n, dtype=np.int8)

    mwc_range_allowed = bool(_cfg.CONTEXT_ALLOW_MWC_RANGE_PERMISSION)
    mwc_long = (mwc_np == STATE_BULLISH) | (
        mwc_range_allowed & (mwc_np == STATE_RANGE)
    )
    mwc_short = (mwc_np == STATE_BEARISH) | (
        mwc_range_allowed & (mwc_np == STATE_RANGE)
    )
    perm_long = ((hwc_np == STATE_BULLISH) & mwc_long).astype(np.int8)
    perm_short = ((hwc_np == STATE_BEARISH) & mwc_short).astype(np.int8)
    gate_pullback = bool(_cfg.CONTEXT_REQUIRE_PERMISSION_ON_PULLBACK_PRINT)

    block = pd.DataFrame(
        {
            "symbol": symbols.to_numpy(),
            "lwc": lwc_np,
            "long": perm_long.astype(bool),
            "short": perm_short.astype(bool),
        }
    )
    for _sym, g in block.groupby("symbol", sort=False, observed=False):
        idxs = g.index.to_numpy()
        lw = g["lwc"].to_numpy()
        pl = g["long"].to_numpy()
        ps = g["short"].to_numpy()
        prior_bear = (lw == STATE_BEARISH)
        prior_bull = (lw == STATE_BULLISH)
        if gate_pullback:
            prior_bear = prior_bear & pl
            prior_bull = prior_bull & ps
        prior_bear_count = _prior_window_count(prior_bear, lookback)
        prior_bull_count = _prior_window_count(prior_bull, lookback)
        trig_long[idxs] = (
            (lw == STATE_BULLISH) & (prior_bear_count > 0)
        ).astype(np.int8)
        trig_short[idxs] = (
            (lw == STATE_BEARISH) & (prior_bull_count > 0)
        ).astype(np.int8)

    return pd.DataFrame(
        {
            "tf_permission_long": perm_long,
            "tf_permission_short": perm_short,
            "lwc_pullback_reversal_long": trig_long,
            "lwc_pullback_reversal_short": trig_short,
        },
        index=lwc.index,
    )


# ---------------------------------------------------------------------------
# Enrichment assembly
# ---------------------------------------------------------------------------

def generate_enriched_frame(
    raw_df: pd.DataFrame,
    thresholds: dict[str, dict[str, float]],
    *,
    emit_warmup: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Enrich a raw 15m tape with the full context contract.

    ``thresholds`` must carry one independently fitted threshold set per
    timeframe: ``{"lwc": {...}, "mwc": {...}, "hwc": {...}}`` (see
    :func:`fit_all_thresholds`). A single 15m-fitted threshold set must never
    be reused for 1h/4h classification.

    Returns
    -------
    (enriched_df, warmup_mask)
        enriched_df has all original columns plus the seven context columns.
        warmup_mask is True for rows whose structural indicators are not yet
        computable (causal warm-up).  Rows flagged warm-up are never scored.
    """
    validate_input_frame(raw_df)
    df = raw_df.sort_values(["datetime", "symbol"]).reset_index(drop=True)

    lwc_th = thresholds["lwc"]
    eff, spread, rv = structural_inputs(df)
    lwc_state = pd.Series(
        classify_regime(
            eff.to_numpy(), spread.to_numpy(), rv.to_numpy(),
            lwc_th["efficiency_abs_trend_threshold"],
            lwc_th["ema_spread_abs_trend_threshold"],
            lwc_th["volatility_compression_threshold"],
        ),
        index=df.index,
    )

    mwc_bars = build_higher_bars(df, int(_cfg.MWC_TIMEFRAME_MINUTES))
    mwc_state = align_completed_states_to_rows(
        df, mwc_bars, thresholds["mwc"], int(_cfg.MWC_TIMEFRAME_MINUTES))

    hwc_bars = build_higher_bars(df, int(_cfg.HWC_TIMEFRAME_MINUTES))
    hwc_state = align_completed_states_to_rows(
        df, hwc_bars, thresholds["hwc"], int(_cfg.HWC_TIMEFRAME_MINUTES))

    exec_cols = compute_permissions_and_triggers(
        hwc_state, mwc_state, lwc_state, df["symbol"],
        lookback=int(_cfg.LWC_PULLBACK_LOOKBACK),
    )

    out = df.copy()
    out["hwc_state"] = hwc_state.to_numpy().astype(np.int8)
    out["mwc_state"] = mwc_state.to_numpy().astype(np.int8)
    out["lwc_state"] = lwc_state.to_numpy().astype(np.int8)
    for col in ("tf_permission_long", "tf_permission_short",
                "lwc_pullback_reversal_long", "lwc_pullback_reversal_short"):
        out[col] = exec_cols[col].to_numpy().astype(np.int8)

    warmup = (
        pd.isna(eff) | pd.isna(spread) | pd.isna(rv)
    ).to_numpy()
    if not emit_warmup:
        out = out.loc[~warmup].reset_index(drop=True)
        warmup_out = warmup[~warmup]
        warmup = warmup_out
    return out, pd.Series(warmup, index=out.index)


def enrich_tape(
    target_df: pd.DataFrame,
    history_df: pd.DataFrame | None,
    thresholds: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Enrich *target_df*, using *history_df* only for causal warm-up.

    The combined stream is enriched, the true warm-up rows (structural
    indicators not computable) are dropped, and only rows belonging to the
    target range are returned.  Thresholds are never refitted here.

    The target's own rows are selected by exact ``(datetime, symbol)`` key
    membership rather than by a global minimum-time cutoff, so a symbol whose
    target range starts later than another symbol's is never polluted by
    earlier history rows.  Duplicate or history-overlapping target keys are
    ambiguous and rejected.
    """
    target_dt = pd.to_datetime(target_df["datetime"], errors="raise")
    target_symbols = target_df["symbol"].astype(str).to_numpy(dtype=object)

    if target_df.duplicated(subset=["datetime", "symbol"]).any():
        dupes = target_df[target_df.duplicated(
            subset=["datetime", "symbol"], keep=False)]
        preview = dupes[["datetime", "symbol"]].head(5).to_dict("records")
        raise ValueError(
            "enrich_tape target has duplicate (datetime, symbol) keys, e.g. "
            f"{preview}; a target row must be uniquely identifiable.")

    if history_df is not None and len(history_df):
        hist_dt = pd.to_datetime(history_df["datetime"], errors="raise")
        hist_symbols = history_df["symbol"].astype(str).to_numpy(dtype=object)
        overlap = set(zip(target_dt.to_numpy(), target_symbols)) & set(
            zip(hist_dt.to_numpy(), hist_symbols))
        if overlap:
            raise ValueError(
                "enrich_tape target keys overlap the history warm-up prefix, "
                f"e.g. {sorted(map(str, overlap))[:5]}; the same "
                "(datetime, symbol) cannot be both history and target.")
        combined = pd.concat([history_df, target_df], ignore_index=True)
    else:
        combined = target_df

    enriched, _warmup = generate_enriched_frame(combined, thresholds)
    # Keep exactly the target's own rows via key membership, never history
    # rows, even when symbols have different target start times.
    target_keys = set(zip(target_dt.to_numpy(), target_symbols))
    enriched_dt = pd.to_datetime(enriched["datetime"], errors="raise").to_numpy()
    enriched_symbols = enriched["symbol"].astype(str).to_numpy(dtype=object)
    own_mask = np.fromiter(
        (
            (dt, sym) in target_keys
            for dt, sym in zip(enriched_dt, enriched_symbols)
        ),
        dtype=bool,
        count=len(enriched),
    )
    own = enriched.loc[own_mask].reset_index(drop=True)
    return own


# ---------------------------------------------------------------------------
# Hashing / manifest
# ---------------------------------------------------------------------------

def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_meta(path: str | None) -> dict[str, str | int | None] | None:
    if not path or not os.path.exists(path):
        return None
    frame = pd.read_csv(path, usecols=lambda c: c in {"datetime", "symbol"})
    dates = pd.to_datetime(frame["datetime"], errors="coerce")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": int(len(frame)),
        "interval": (
            [dates.min().isoformat(), dates.max().isoformat()]
            if not dates.isna().all() else None
        ),
    }


def _floatify(obj: object) -> object:
    """Recursively cast numeric leaves to plain ``float`` for JSON output."""
    if isinstance(obj, dict):
        return {k: _floatify(v) for k, v in obj.items()}
    return float(obj)


def build_manifest(
    thresholds: dict[str, object],
    *,
    train_source: str | None = None,
    tapes: dict[str, str] | None = None,
    tape_sources: dict[str, str] | None = None,
    history_sources: dict[str, str] | None = None,
    prefix_fitting_rows: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build the versioned enrichment manifest.

    Every enriched tape entry records both the enriched file's own hash and the
    actual raw source tape (``path``/``sha256``) it was enriched from.  Sources
    are passed explicitly via ``tape_sources``; they are *never* inferred from
    config so custom CLI ``--train/--test/--forward`` paths are recorded
    faithfully (a ``forward`` enriched tape is attributed to its forward raw
    source, not to the test path).  Enriched output hashes are preserved in the
    entry's ``sha256`` field.  ``thresholds`` carries one set per timeframe
    (``{"lwc": {...}, "mwc": {...}, "hwc": {...}}``).  ``prefix_fitting_rows``
    records the exact per-symbol row count/interval used to fit those
    thresholds, so the training-prefix boundary is auditable.
    """
    tape_entries: dict[str, object] = {}
    for name, path in (tapes or {}).items():
        if not path or not os.path.exists(path):
            continue
        entry: dict[str, object] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
        source = (tape_sources or {}).get(name)
        entry["source"] = _source_meta(source)
        tape_entries[name] = entry
    history_entries: dict[str, object] = {}
    for name, path in (history_sources or {}).items():
        if path and os.path.exists(path):
            history_entries[name] = _source_meta(path)
    return {
        "context_algorithm_version": str(_cfg.CONTEXT_ALGORITHM_VERSION),
        "context_contract": _cfg.context_contract(),
        "thresholds": _floatify(thresholds),
        "threshold_fitting": {
            "source": _source_meta(train_source),
            "train_only": True,
            "per_symbol": prefix_fitting_rows or {},
        },
        "timeframes_minutes": {
            "hwc": int(_cfg.HWC_TIMEFRAME_MINUTES),
            "mwc": int(_cfg.MWC_TIMEFRAME_MINUTES),
            "lwc": int(_cfg.LWC_TIMEFRAME_MINUTES),
        },
        "bar_open_timestamps": True,
        "state_codes": dict(_cfg.CONTEXT_STATE_CODES),
        "lwc_pullback_lookback": int(_cfg.LWC_PULLBACK_LOOKBACK),
        "horizon_bars_15m": int(_cfg.MAX_HOLD_CANDLES),
        "tapes": tape_entries,
        "history_sources": history_entries,
    }


def write_manifest(payload: dict[str, object], path: str) -> str:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(dest)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _write_enriched_csv(
    raw_path: str,
    thresholds: dict[str, dict[str, float]],
    out_path: str,
    history_paths: tuple[str, ...] = (),
) -> pd.DataFrame:
    raw = pd.read_csv(raw_path)
    raw["datetime"] = pd.to_datetime(raw["datetime"])
    history_frames = []
    for history_path in history_paths:
        if history_path and os.path.exists(history_path):
            h = pd.read_csv(history_path)
            h["datetime"] = pd.to_datetime(h["datetime"])
            history_frames.append(h)
    history = (
        pd.concat(history_frames, ignore_index=True) if history_frames else None
    )
    enriched = enrich_tape(raw, history, thresholds)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(out_path, index=False)
    logger.info("Wrote %d enriched rows to %s", len(enriched), out_path)
    return enriched


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m gpu_fuzzy_trader.data.trend_context",
        description="Deterministic causal HWC/MWC/LWC enrichment.",
    )
    parser.add_argument("--train", default=_cfg.TRAIN_CSV_PATH)
    parser.add_argument("--test", default=_cfg.TEST_CSV_PATH)
    parser.add_argument("--forward", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)

    if not os.path.exists(args.train):
        logger.error("Training source tape not found: %s", args.train)
        return 1

    raw_train = pd.read_csv(args.train)
    raw_train["datetime"] = pd.to_datetime(raw_train["datetime"])
    validate_input_frame(raw_train)

    # Thresholds are fitted strictly from the per-symbol training prefix
    # (never the rows that later become the validation split), with an
    # independent threshold set per timeframe.
    train_prefix = build_train_prefix(raw_train)
    thresholds = fit_all_thresholds(train_prefix)
    prefix_fitting_rows = {
        str(symbol): {
            "rows": int(len(g)),
            "interval": [
                g["datetime"].min().isoformat(),
                g["datetime"].max().isoformat(),
            ],
        }
        for symbol, g in train_prefix.groupby("symbol", observed=False)
    }

    out_dir = args.out_dir or _cfg.ENRICHED_DIR
    train_out = os.path.join(out_dir, "train_new_hwc_mwc_lwc.csv")
    test_out = os.path.join(out_dir, "test_new_hwc_mwc_lwc.csv")
    forward_out = os.path.join(out_dir, "forward_hwc_mwc_lwc.csv")

    has_test = bool(args.test) and os.path.exists(args.test)
    _write_enriched_csv(args.train, thresholds, train_out)
    if has_test:
        _write_enriched_csv(
            args.test, thresholds, test_out, history_paths=(args.train,))
    if args.forward and os.path.exists(args.forward):
        # Forward enrichment may use trailing train AND test history so a
        # forward tape starting after the test period never leaves a
        # timestamp gap that validate_input_frame would reject.
        forward_history = (args.train, args.test) if has_test else (args.train,)
        _write_enriched_csv(
            args.forward, thresholds, forward_out,
            history_paths=forward_history)

    manifest = build_manifest(
        thresholds,
        train_source=args.train,
        tapes={
            "train": train_out,
            "test": test_out if os.path.exists(test_out) else None,
            "forward": forward_out if os.path.exists(forward_out) else None,
        },
        tape_sources={
            "train": args.train,
            "test": args.test if has_test else None,
            "forward": (
                args.forward
                if args.forward and os.path.exists(args.forward)
                else None
            ),
        },
        history_sources={
            "test_history_train": args.train if has_test else None,
            "forward_history_train": (
                args.train
                if args.forward and os.path.exists(args.forward)
                else None
            ),
            "forward_history_test": (
                args.test
                if args.forward and os.path.exists(args.forward) and has_test
                else None
            ),
        },
        prefix_fitting_rows=prefix_fitting_rows,
    )
    manifest_path = os.path.join(out_dir, "trend_context_manifest.json")
    write_manifest(manifest, manifest_path)
    logger.info("Wrote enrichment manifest to %s", manifest_path)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
