"""
regime_cluster.py — Market-regime labeling for Phase 1 stationarity.

Implements dual-window rolling linear regression regime detection
with 9-day median pre-filtering and a strict 14-day minimum duration constraint.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config

logger = logging.getLogger(__name__)

RegimeBundle = dict[str, Any]


def _enforce_min_duration(reg_array: np.ndarray, min_days: int = 14) -> np.ndarray:
    """Iteratively merges any block shorter than min_days with its longer neighbor."""
    arr = np.array(reg_array)
    n = len(arr)
    if n == 0:
        return arr
    while True:
        blocks = []
        curr_val = arr[0]
        curr_start = 0
        for i in range(1, n):
            if arr[i] != curr_val:
                blocks.append((curr_val, curr_start, i - 1, i - curr_start))
                curr_val = arr[i]
                curr_start = i
        blocks.append((curr_val, curr_start, n - 1, n - curr_start))
        
        shortest_idx = -1
        shortest_len = min_days
        for idx, (val, start, end, length) in enumerate(blocks):
            if length < shortest_len:
                shortest_len = length
                shortest_idx = idx
                
        if shortest_idx == -1:
            break
            
        val, start, end, length = blocks[shortest_idx]
        
        left_val = blocks[shortest_idx - 1][0] if shortest_idx > 0 else None
        left_len = blocks[shortest_idx - 1][3] if shortest_idx > 0 else 0
        right_val = blocks[shortest_idx + 1][0] if shortest_idx < len(blocks) - 1 else None
        right_len = blocks[shortest_idx + 1][3] if shortest_idx < len(blocks) - 1 else 0
        
        if left_val is not None and right_val is not None:
            if left_len >= right_len:
                merge_val = left_val
            else:
                merge_val = right_val
        elif left_val is not None:
            merge_val = left_val
        elif right_val is not None:
            merge_val = right_val
        else:
            break
            
        arr[start : end + 1] = merge_val
    return arr


def _compute_rolling_regimes_for_symbol(
    df_sym: pd.DataFrame,
    fast_window: int,
    slow_window: int,
    fast_r2_threshold: float,
    slow_r2_threshold: float,
    fast_slope_threshold: float,
    slow_slope_threshold: float,
    med_window: int,
    min_days: int,
) -> pd.Series:
    """Computes daily rolling regression regimes for a single symbol and maps back to original index."""
    if len(df_sym) == 0:
        return pd.Series(dtype=np.int32)
        
    if "datetime" in df_sym.columns:
        dates = pd.to_datetime(df_sym["datetime"])
    elif isinstance(df_sym.index, pd.DatetimeIndex):
        dates = df_sym.index
    else:
        # Fallback to dummy daily dates if no time information is available (e.g. unit tests)
        dates = pd.date_range(start="2024-01-01", periods=len(df_sym), freq="D")
        
    temp = df_sym.copy()
    temp["_dates"] = dates
    temp = temp.set_index("_dates")
    
    price_col = "label_open_next"
    if price_col not in temp.columns:
        # Fallback to the first available numeric column, otherwise return all sideways
        numeric_cols = temp.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            price_col = numeric_cols[0]
        else:
            return pd.Series(0, index=df_sym.index, name="regime", dtype=np.int32)
            
    # Resample price to daily (using last value)
    price_daily = temp[price_col].resample("D").last()
    price_daily = price_daily.ffill().bfill()
    price_smooth = price_daily.rolling(window=3, min_periods=1).mean()
    
    fast_slopes = []
    fast_r2_values = []
    slow_slopes = []
    slow_r2_values = []
    
    n_days = len(price_daily)
    for i in range(n_days):
        # Fast regression
        if i < fast_window:
            fast_slopes.append(0.0)
            fast_r2_values.append(0.0)
        else:
            y = price_smooth.iloc[i-fast_window+1 : i+1].values
            x = np.arange(fast_window)
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * x + intercept
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            ss_res = np.sum((y - y_pred) ** 2)
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            # Normalize slope as a percentage of price so thresholds like 0.0016 (0.16% per day) work universally
            mean_y = np.mean(y)
            norm_slope = slope / mean_y if mean_y != 0 else 0.0
            
            fast_slopes.append(norm_slope)
            fast_r2_values.append(r2)
            
        # Slow regression
        if i < slow_window:
            slow_slopes.append(0.0)
            slow_r2_values.append(0.0)
        else:
            y = price_smooth.iloc[i-slow_window+1 : i+1].values
            x = np.arange(slow_window)
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * x + intercept
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            ss_res = np.sum((y - y_pred) ** 2)
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            # Normalize slow slope as a percentage of price
            mean_y = np.mean(y)
            norm_slope = slope / mean_y if mean_y != 0 else 0.0
            
            slow_slopes.append(norm_slope)
            slow_r2_values.append(r2)
            
    fast_slopes = np.array(fast_slopes)
    fast_r2_values = np.array(fast_r2_values)
    slow_slopes = np.array(slow_slopes)
    slow_r2_values = np.array(slow_r2_values)
    
    regimes = []
    for i in range(n_days):
        fs = fast_slopes[i]
        fr2 = fast_r2_values[i]
        ss = slow_slopes[i]
        sr2 = slow_r2_values[i]
        
        is_bull = (fr2 >= fast_r2_threshold and fs >= fast_slope_threshold and 
                   (sr2 < slow_r2_threshold or ss >= -slow_slope_threshold / 2.0))
                   
        is_bear = (fr2 >= fast_r2_threshold and fs <= -fast_slope_threshold and 
                   (sr2 < slow_r2_threshold or ss <= slow_slope_threshold / 2.0))
                   
        if is_bull:
            regimes.append(2)
        elif is_bear:
            regimes.append(1)
        else:
            regimes.append(0)
            
    regimes = np.array(regimes)
    
    if med_window > 0:
        regimes = pd.Series(regimes).rolling(window=med_window, center=True, min_periods=1).median().astype(int).values
        
    regimes = _enforce_min_duration(regimes, min_days=min_days)
    
    daily_series = pd.Series(regimes, index=price_daily.index)
    row_dates = dates.dt.normalize() if hasattr(dates, "dt") else pd.Series(dates).dt.normalize()
    mapped_regimes = row_dates.map(daily_series)
    mapped_regimes = mapped_regimes.ffill().bfill().fillna(0).astype(np.int32)
    
    return pd.Series(mapped_regimes.values, index=df_sym.index, name="regime", dtype=np.int32)


def fit_regime_labels(
    df: pd.DataFrame,
    regime_features: Optional[list[str]] = None,
    n_clusters: Optional[int] = None,
    clusterer: Optional[str] = None,
    random_state: Optional[int] = None,
    zero_var_eps: Optional[float] = None,
) -> Optional[tuple[pd.Series, RegimeBundle]]:
    """Fits dual-window regression regime labels on train rows; returns labels aligned to df.index."""
    if random_state is None:
        random_state = config.get_seed()
    if len(df) == 0:
        return None

    # Load parameters from config as single source of truth
    fast_window = config.PHASE1_REGIME_FAST_WINDOW
    slow_window = config.PHASE1_REGIME_SLOW_WINDOW
    fast_r2_threshold = config.PHASE1_REGIME_FAST_R2_THRESHOLD
    slow_r2_threshold = config.PHASE1_REGIME_SLOW_R2_THRESHOLD
    fast_slope_threshold = config.PHASE1_REGIME_FAST_SLOPE_THRESHOLD
    slow_slope_threshold = config.PHASE1_REGIME_SLOW_SLOPE_THRESHOLD
    med_window = config.PHASE1_REGIME_MED_WINDOW
    min_days = config.PHASE1_REGIME_MIN_DAYS
    
    try:
        if "symbol" in df.columns:
            symbols = df["symbol"].unique()
            all_series = []
            for sym in symbols:
                df_sym = df[df["symbol"] == sym]
                sym_series = _compute_rolling_regimes_for_symbol(
                    df_sym, fast_window, slow_window,
                    fast_r2_threshold, slow_r2_threshold,
                    fast_slope_threshold, slow_slope_threshold,
                    med_window, min_days
                )
                all_series.append(sym_series)
            label_series = pd.concat(all_series).reindex(df.index)
        else:
            label_series = _compute_rolling_regimes_for_symbol(
                df, fast_window, slow_window,
                fast_r2_threshold, slow_r2_threshold,
                fast_slope_threshold, slow_slope_threshold,
                med_window, min_days
            )
    except Exception as exc:
        logger.warning("Regime detection failed: %s", exc)
        return None

    bundle: RegimeBundle = {
        "regime_features": ["label_open_next"],
        "n_clusters": 3,
        "clusterer": "rolling_regression",
        "fast_window": fast_window,
        "slow_window": slow_window,
        "fast_r2_threshold": fast_r2_threshold,
        "slow_r2_threshold": slow_r2_threshold,
        "fast_slope_threshold": fast_slope_threshold,
        "slow_slope_threshold": slow_slope_threshold,
        "med_window": med_window,
        "min_days": min_days,
    }
    
    return label_series, bundle


def assign_regime_labels(df: pd.DataFrame, bundle: RegimeBundle) -> pd.Series:
    """Assign regime labels using parameters stored in a bundle (no refit)."""
    # Assign parameters using bundle, falling back to config defaults
    fast_window = bundle.get("fast_window", config.PHASE1_REGIME_FAST_WINDOW)
    slow_window = bundle.get("slow_window", config.PHASE1_REGIME_SLOW_WINDOW)
    fast_r2_threshold = bundle.get("fast_r2_threshold", config.PHASE1_REGIME_FAST_R2_THRESHOLD)
    slow_r2_threshold = bundle.get("slow_r2_threshold", config.PHASE1_REGIME_SLOW_R2_THRESHOLD)
    fast_slope_threshold = bundle.get("fast_slope_threshold", config.PHASE1_REGIME_FAST_SLOPE_THRESHOLD)
    slow_slope_threshold = bundle.get("slow_slope_threshold", config.PHASE1_REGIME_SLOW_SLOPE_THRESHOLD)
    med_window = bundle.get("med_window", config.PHASE1_REGIME_MED_WINDOW)
    min_days = bundle.get("min_days", config.PHASE1_REGIME_MIN_DAYS)
    
    if "symbol" in df.columns:
        symbols = df["symbol"].unique()
        all_series = []
        for sym in symbols:
            df_sym = df[df["symbol"] == sym]
            sym_series = _compute_rolling_regimes_for_symbol(
                df_sym, fast_window, slow_window,
                fast_r2_threshold, slow_r2_threshold,
                fast_slope_threshold, slow_slope_threshold,
                med_window, min_days
            )
            all_series.append(sym_series)
        label_series = pd.concat(all_series).reindex(df.index)
    else:
        label_series = _compute_rolling_regimes_for_symbol(
            df, fast_window, slow_window,
            fast_r2_threshold, slow_r2_threshold,
            fast_slope_threshold, slow_slope_threshold,
            med_window, min_days
        )
    return label_series


def persist_regime_model(path: str, bundle: RegimeBundle) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(bundle, path)
    logger.info("Saved regime cluster model to %s", path)


def load_regime_model(path: str) -> RegimeBundle:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Regime model not found: {path}")
    return joblib.load(path)
