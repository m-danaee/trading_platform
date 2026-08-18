"""gpu_fuzzy_trader/data/multi_timeframe.py — Causal Multi-Timeframe Data Engine.

Constructs complete, UTC-aligned higher timeframe (MWC: 1H / 60m, HWC: 4H / 240m)
bars from raw 15m (LWC) OHLCV data, calculates independent technical features
per timeframe, and aligns HTF features back to 15m execution rows with strict
point-in-time causality.
"""

from __future__ import annotations

import logging
from typing import Sequence
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


def build_complete_higher_bars(
    df: pd.DataFrame,
    timeframe_minutes: int,
    base_timeframe_minutes: int = 15,
) -> pd.DataFrame:
    """Aggregate 15m rows into complete, UTC-aligned higher-timeframe bars.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or constituent DataFrame containing columns:
        ``datetime``, ``symbol``, ``open``, ``high``, ``low``, ``close``, ``volume``.
    timeframe_minutes : int
        Target higher timeframe in minutes (e.g. 60 for 1H, 240 for 4H).
    base_timeframe_minutes : int, default 15
        Base timeframe in minutes of the input bars.

    Returns
    -------
    pd.DataFrame
        DataFrame of complete higher-timeframe bars with columns:
        ``datetime`` (bar-open timestamp), ``symbol``, ``open``, ``high``,
        ``low``, ``close``, ``volume``.

    Notes
    -----
    - Candle boundaries are aligned to UTC standard intervals (e.g. 00:00, 01:00 for 1H;
      00:00, 04:00, 08:00, 12:00, 16:00, 20:00 for 4H).
    - An HTF bar is kept ONLY if it contains exactly
      ``timeframe_minutes // base_timeframe_minutes`` constituent rows.
    - Incomplete leading/trailing boundary buckets or buckets with missing
      constituent bars are strictly dropped.
    """
    if df.empty:
        return pd.DataFrame(columns=["datetime", "symbol", *_OHLCV_COLUMNS])

    missing_cols = [c for c in ("datetime", "symbol", *_OHLCV_COLUMNS) if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Input DataFrame is missing required columns: {missing_cols}")

    df_work = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_work["datetime"]):
        df_work["datetime"] = pd.to_datetime(df_work["datetime"])

    tf = pd.Timedelta(minutes=int(timeframe_minutes))
    expected_rows = int(timeframe_minutes) // int(base_timeframe_minutes)

    parts: list[pd.DataFrame] = []
    for symbol, g in df_work.sort_values("datetime").groupby(
        "symbol", sort=False, observed=False
    ):
        g = g.sort_values("datetime")
        # Fixed UTC anchoring via dt.floor
        bucket = g["datetime"].dt.floor(tf)
        
        # Verify completeness: bucket must contain exactly expected_rows
        counts = bucket.value_counts()
        complete_buckets = counts[counts == expected_rows].index

        if complete_buckets.empty:
            continue

        g_complete = g[bucket.isin(complete_buckets)]
        bucket_complete = bucket[bucket.isin(complete_buckets)]

        agg = g_complete.groupby(bucket_complete, sort=True).agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        out = agg.reset_index()
        out = out.rename(columns={out.columns[0]: "datetime"})
        out["symbol"] = symbol
        parts.append(out)

    if not parts:
        return pd.DataFrame(columns=["datetime", "symbol", *_OHLCV_COLUMNS])

    result = pd.concat(parts, ignore_index=True)
    return result.sort_values(["datetime", "symbol"]).reset_index(drop=True)


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute relative strength index (RSI)."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # Where avg_loss is 0 and avg_gain > 0, RSI = 100; where both 0, RSI = 50
    rsi = rsi.where(~(avg_loss == 0.0) | (avg_gain == 0.0), 100.0)
    rsi = rsi.where(~((avg_loss == 0.0) & (avg_gain == 0.0)), 50.0)
    return rsi


def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Compute average true range (ATR)."""
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def _compute_kama(close: pd.Series, period: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    """Compute Kaufman Adaptive Moving Average (KAMA)."""
    n = len(close)
    if n <= period:
        return pd.Series(np.full(n, np.nan), index=close.index)

    close_arr = close.to_numpy(dtype=float)
    kama = np.full(n, np.nan, dtype=float)

    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)

    # First KAMA point initialized at period index
    kama[period - 1] = close_arr[period - 1]

    diff = np.abs(np.diff(close_arr))
    for i in range(period, n):
        change = abs(close_arr[i] - close_arr[i - period])
        volatility = np.sum(diff[i - period : i])
        if volatility == 0.0:
            er = 0.0
        else:
            er = change / volatility
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close_arr[i] - kama[i - 1])

    return pd.Series(kama, index=close.index)


def compute_timeframe_features(
    df_bars: pd.DataFrame,
    timeframe_minutes: int,
) -> pd.DataFrame:
    """Compute technical features independently on completed bars of a timeframe.

    Parameters
    ----------
    df_bars : pd.DataFrame
        DataFrame of completed bars (from ``build_complete_higher_bars`` or raw tape)
        containing ``datetime``, ``symbol``, ``open``, ``high``, ``low``, ``close``, ``volume``.
    timeframe_minutes : int
        Timeframe of the bars in minutes.

    Returns
    -------
    pd.DataFrame
        DataFrame containing ``datetime``, ``symbol``, original OHLCV, and
        computed technical features (RSI, ATR, KAMA, Bollinger Bands, Realized
        Volatility, Momentum, Volume metrics, EMA spreads).
    """
    if df_bars.empty:
        return df_bars.copy()

    df_work = df_bars.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_work["datetime"]):
        df_work["datetime"] = pd.to_datetime(df_work["datetime"])

    parts: list[pd.DataFrame] = []
    for symbol, g in df_work.sort_values("datetime").groupby(
        "symbol", sort=False, observed=False
    ):
        g = g.sort_values("datetime").copy()
        close = g["close"].astype(float)
        high = g["high"].astype(float)
        low = g["low"].astype(float)
        open_ = g["open"].astype(float)
        volume = g["volume"].astype(float)

        # 1. RSI
        rsi_14 = _compute_rsi(close, period=14)
        g["rsi_14"] = rsi_14
        g["rsi_14_midline"] = rsi_14 - 50.0

        # 2. ATR
        atr_14 = _compute_atr(high, low, close, period=14)
        g["atr_14"] = atr_14
        g["atr_14_ratio"] = atr_14 / close.replace(0.0, np.nan)

        # 3. KAMA
        kama_10 = _compute_kama(close, period=10)
        g["kama_10"] = kama_10
        g["kama_distance_10"] = (close - kama_10) / atr_14.replace(0.0, np.nan)

        # 4. Bollinger Bands (20, 2)
        sma_20 = close.rolling(20, min_periods=20).mean()
        std_20 = close.rolling(20, min_periods=20).std()
        upper_bb = sma_20 + 2.0 * std_20
        lower_bb = sma_20 - 2.0 * std_20
        bb_width = upper_bb - lower_bb
        g["bollinger_pct_b"] = (close - lower_bb) / bb_width.replace(0.0, np.nan)
        g["bollinger_bandwidth"] = bb_width / sma_20.replace(0.0, np.nan)

        # 5. Realized Volatility (rolling std of log returns)
        log_ret = np.log(close / close.shift(1).replace(0.0, np.nan))
        g["realized_volatility"] = log_ret.rolling(20, min_periods=20).std()

        # 6. Momentum / ROC / Efficiency
        g["momentum_roc"] = (close - close.shift(10)) / close.shift(10).replace(0.0, np.nan)
        disp_10 = close.diff(10)
        path_10 = close.diff(1).abs().rolling(10, min_periods=10).sum()
        g["price_efficiency_10"] = disp_10 / path_10.replace(0.0, np.nan)

        # 7. EMA Spreads
        ema_8 = close.ewm(span=8, adjust=False).mean()
        ema_21 = close.ewm(span=21, adjust=False).mean()
        g["ema_spread_8_21"] = (ema_8 - ema_21) / atr_14.replace(0.0, np.nan)

        # 8. Volume Features
        vol_sma_20 = volume.rolling(20, min_periods=20).mean()
        vol_std_20 = volume.rolling(20, min_periods=20).std()
        g["relative_volume_20"] = volume / vol_sma_20.replace(0.0, np.nan)
        g["volume_spike"] = (volume > (vol_sma_20 + 2.0 * vol_std_20)).astype(float)

        parts.append(g)

    result = pd.concat(parts, ignore_index=True)
    # Restore original index mapping / order
    return result.sort_values(["datetime", "symbol"]).reset_index(drop=True)


def align_htf_features_causal(
    lwc_df: pd.DataFrame,
    htf_features: pd.DataFrame,
    timeframe_minutes: int,
    base_timeframe_minutes: int = 15,
) -> pd.DataFrame:
    """Align higher-timeframe features causally to base (15m) execution rows.

    For an LWC candle opened at timestamp D:
    - Execution occurs at ``D + base_timeframe_minutes``.
    - An HTF candle opened at T closes at ``T + timeframe_minutes``.
    - The causal HTF candle aligned to row D is the most recent completed HTF
      bar whose close time satisfies:
      ``HTF Close <= D + base_timeframe_minutes``.
    - If no completed HTF bar has closed yet, feature values are NaN.

    Parameters
    ----------
    lwc_df : pd.DataFrame
        Base execution rows containing at least ``datetime`` and ``symbol``.
    htf_features : pd.DataFrame
        DataFrame of higher-timeframe features containing ``datetime``, ``symbol``,
        and feature columns.
    timeframe_minutes : int
        Higher timeframe duration in minutes (e.g. 60 or 240).
    base_timeframe_minutes : int, default 15
        Base timeframe duration in minutes.

    Returns
    -------
    pd.DataFrame
        Copy of ``lwc_df`` with all feature columns from ``htf_features`` attached
        via causal point-in-time alignment.
    """
    if lwc_df.empty:
        return lwc_df.copy()

    if not pd.api.types.is_datetime64_any_dtype(lwc_df["datetime"]):
        lwc_work = lwc_df.copy()
        lwc_work["datetime"] = pd.to_datetime(lwc_work["datetime"])
    else:
        lwc_work = lwc_df.copy()

    if not pd.api.types.is_datetime64_any_dtype(htf_features["datetime"]):
        htf_work = htf_features.copy()
        htf_work["datetime"] = pd.to_datetime(htf_work["datetime"])
    else:
        htf_work = htf_features.copy()

    # Identify feature columns to align (all columns except datetime and symbol)
    feature_cols = [c for c in htf_work.columns if c not in ("datetime", "symbol")]

    tf = pd.Timedelta(minutes=int(timeframe_minutes))
    base_tf = pd.Timedelta(minutes=int(base_timeframe_minutes))

    # Pre-calculate HTF close timestamps
    htf_work["_htf_close"] = htf_work["datetime"] + tf

    # Prepare output container matching lwc_df index
    aligned_df = lwc_df.copy()
    for col in feature_cols:
        if col not in aligned_df.columns:
            aligned_df[col] = np.nan

    # Group by symbol for point-in-time searchsorted alignment
    for symbol, lwc_grp in lwc_work.groupby("symbol", sort=False, observed=False):
        htf_grp = htf_work[htf_work["symbol"] == symbol].sort_values("_htf_close")
        if htf_grp.empty:
            continue

        htf_close_ns = htf_grp["_htf_close"].to_numpy(dtype="int64")
        lwc_exec_ns = (lwc_grp["datetime"] + base_tf).to_numpy(dtype="int64")

        # searchsorted side="right" finds the number of HTF close times <= execution time
        # subtracting 1 gives the index of the latest completed HTF candle
        match_idx = np.searchsorted(htf_close_ns, lwc_exec_ns, side="right") - 1

        valid_mask = match_idx >= 0
        if not np.any(valid_mask):
            continue

        valid_match_idx = match_idx[valid_mask]
        target_indices = lwc_grp.index[valid_mask]

        for col in feature_cols:
            htf_vals = htf_grp[col].to_numpy()
            aligned_df.loc[target_indices, col] = htf_vals[valid_match_idx]

    return aligned_df
