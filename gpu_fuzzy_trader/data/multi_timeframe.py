"""gpu_fuzzy_trader/data/multi_timeframe.py — Causal Multi-Timeframe Data Engine.

Constructs complete, UTC-aligned higher timeframe (MWC: 1H / 60m, HWC: 4H / 240m)
bars from raw 15m (LWC) OHLCV data, calculates independent technical features
per timeframe, and aligns HTF features back to 15m execution rows with strict
point-in-time causality.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config as _cfg

logger = logging.getLogger(__name__)

_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


def _as_utc_datetime(values: pd.Series | pd.Index) -> pd.Series | pd.DatetimeIndex:
    """Parse timestamps as UTC without accepting a local-time interpretation."""
    parsed = pd.to_datetime(values, errors="raise", utc=True)
    # The repository's persisted tapes use timezone-naive timestamps.  Keep
    # that storage convention while making the values unambiguously UTC after
    # conversion (including inputs carrying a non-UTC offset).
    if isinstance(parsed, pd.Series):
        return parsed.dt.tz_localize(None)
    return parsed.tz_localize(None)


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

    timeframe_minutes = int(timeframe_minutes)
    base_timeframe_minutes = int(base_timeframe_minutes)
    if timeframe_minutes <= 0 or base_timeframe_minutes <= 0:
        raise ValueError("timeframe_minutes and base_timeframe_minutes must be positive")
    if timeframe_minutes % base_timeframe_minutes != 0:
        raise ValueError(
            "timeframe_minutes must be an integer multiple of base_timeframe_minutes"
        )

    df_work = df.copy()
    df_work["datetime"] = _as_utc_datetime(df_work["datetime"])
    if df_work.duplicated(["datetime", "symbol"]).any():
        duplicate_count = int(df_work.duplicated(["datetime", "symbol"]).sum())
        raise ValueError(
            "Input contains duplicate constituent bars for the same symbol and "
            f"timestamp ({duplicate_count} duplicates)."
        )

    tf = pd.Timedelta(minutes=timeframe_minutes)
    base_tf = pd.Timedelta(minutes=base_timeframe_minutes)
    expected_rows = timeframe_minutes // base_timeframe_minutes

    parts: list[pd.DataFrame] = []
    for symbol, g in df_work.sort_values("datetime").groupby(
        "symbol", sort=False, observed=False
    ):
        g = g.sort_values("datetime")
        # ``floor`` is safe only after timestamps have been normalized to UTC.
        # Row counts alone are not continuity validation: a duplicate row can
        # otherwise replace a missing constituent and create a false bar.
        bucket = g["datetime"].dt.floor(tf)
        complete_bucket_values: list[pd.Timestamp] = []
        for bucket_start, bucket_rows in g.groupby(bucket, sort=True, observed=False):
            expected = pd.DatetimeIndex(
                bucket_start + np.arange(expected_rows) * base_tf
            )
            actual = pd.DatetimeIndex(bucket_rows["datetime"].sort_values())
            if len(actual) == expected_rows and actual.equals(expected):
                complete_bucket_values.append(bucket_start)

        if not complete_bucket_values:
            continue

        complete_index = pd.Index(complete_bucket_values)
        g_complete = g[bucket.isin(complete_index)]
        bucket_complete = bucket[bucket.isin(complete_index)]

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
    df_work["_orig_order"] = np.arange(len(df_bars))
    df_work["datetime"] = _as_utc_datetime(df_work["datetime"])

    parts: list[pd.DataFrame] = []
    for _symbol, g in df_work.sort_values("datetime").groupby(
        "symbol", sort=False, observed=False
    ):
        g = g.sort_values("datetime").copy()
        close = g["close"].astype(float)
        high = g["high"].astype(float)
        low = g["low"].astype(float)
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
    result = result.sort_values("_orig_order").drop(columns=["_orig_order"])
    result.index = df_bars.index
    return result



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

    lwc_work = lwc_df.copy()
    lwc_work["datetime"] = _as_utc_datetime(lwc_work["datetime"])
    if lwc_work.duplicated(["datetime", "symbol"]).any():
        raise ValueError("LWC data contains duplicate (datetime, symbol) rows.")
    htf_work = htf_features.copy()
    missing_htf = [c for c in ("datetime", "symbol") if c not in htf_work.columns]
    if missing_htf:
        raise ValueError(f"HTF features are missing required columns: {missing_htf}")
    htf_work["datetime"] = _as_utc_datetime(htf_work["datetime"])
    if htf_work.duplicated(["datetime", "symbol"]).any():
        raise ValueError("HTF features contain duplicate (datetime, symbol) rows.")

    # Only derived features are aligned.  HTF OHLCV must never overwrite the
    # execution tape's OHLCV columns, and an existing LWC column is preserved.
    feature_cols = [
        c
        for c in htf_work.columns
        if c not in ("datetime", "symbol", *_OHLCV_COLUMNS)
        and c not in lwc_work.columns
    ]

    timeframe_minutes = int(timeframe_minutes)
    base_timeframe_minutes = int(base_timeframe_minutes)
    if timeframe_minutes <= 0 or base_timeframe_minutes <= 0:
        raise ValueError("timeframe_minutes and base_timeframe_minutes must be positive")
    tf = pd.Timedelta(minutes=timeframe_minutes)
    base_tf = pd.Timedelta(minutes=base_timeframe_minutes)

    # Pre-calculate HTF close timestamps
    htf_work["_htf_close"] = htf_work["datetime"] + tf

    # Prepare output container matching lwc_df index
    # Persist the normalized UTC-naive timestamps used for the match.  An
    # offset-aware input must not remain in local time downstream.
    aligned_df = lwc_work.copy()
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

        # Check staleness: if LWC candle is too far past HTF close timestamp
        # (e.g. data gaps), discard the match to avoid carrying forward stale context.
        max_staleness = getattr(_cfg, "MTF_MAX_STALENESS_CANDLES", None)
        if max_staleness is not None and int(max_staleness) > 0:
            max_staleness_ns = int(max_staleness) * base_tf.value
            matched_htf_close_ns = htf_close_ns[valid_match_idx]
            lwc_matched_exec_ns = lwc_exec_ns[valid_mask]
            not_stale = (lwc_matched_exec_ns - matched_htf_close_ns) <= max_staleness_ns
            valid_match_idx = valid_match_idx[not_stale]
            target_indices = target_indices[not_stale]

        for col in feature_cols:
            htf_vals = htf_grp[col].to_numpy()
            aligned_df.loc[target_indices, col] = htf_vals[valid_match_idx]

    return aligned_df
