"""
Monthly rolling-window validation for rule sets.

Exports
-------
MonthlyWindowSummary
    Dataclass summarising a rule set across 30-day windows.
build_monthly_windows
    Slice a DataFrame into chronological windows.
summarize_monthly_metrics
    Collapse a list of per-window metrics dicts into a summary.
evaluate_rule_set_monthly
    Convenience: build windows, run CPUBacktestEngine, return summary + raw.
monthly_penalty
    Non-negative penalty for Phase 3/4 objective functions.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Iterable

import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.backtest.cpu_engine import CPUBacktestEngine
from gpu_fuzzy_trader.backtest.df_slim import slim_backtest_df

logger = logging.getLogger(__name__)


@dataclass
class MonthlyWindowSummary:
    """Aggregate statistics across rolling monthly windows."""

    windows: int
    profitable_windows: int
    profitable_ratio: float
    mean_return_pct: float
    median_return_pct: float
    worst_return_pct: float
    latest_return_pct: float
    recency_weighted_return_pct: float
    mean_profit_factor: float
    worst_profit_factor: float
    worst_drawdown_pct: float
    min_trades: int
    mean_trades: float
    equity_slope: float
    max_equity_dip_pct: float
    score: float


def _datetime_series(df: pd.DataFrame) -> pd.Series:
    """Extract a numeric or datetime column for chronological ordering."""
    if "datetime" not in df.columns:
        return pd.Series(np.arange(len(df)), index=df.index)
    return pd.to_datetime(df["datetime"], errors="coerce")


def build_monthly_windows(
    df: pd.DataFrame,
    window_days: int | None = None,
    stride_days: int | None = None,
    min_rows: int | None = None,
    max_windows: int | None = None,
) -> list[pd.DataFrame]:
    """Return chronological rolling validation windows.

    Parameters
    ----------
    df:
        DataFrame with at least a ``datetime`` column (or fall back to row
        index).  Must contain label columns needed by the backtest engine.
    window_days:
        Width of each window in calendar days.  Default ``MONTHLY_WINDOW_DAYS``
        (30).
    stride_days:
        Step between window starts.  Default ``MONTHLY_WINDOW_STRIDE_DAYS``
        (also 30, i.e. non-overlapping).
    min_rows:
        Minimum number of rows a window must contain to be kept.
        Default ``MONTHLY_WINDOW_MIN_ROWS`` (2500).
    max_windows:
        Maximum number of windows returned.  Default
        ``MONTHLY_WINDOW_MAX_WINDOWS`` (24).

    Returns
    -------
    list[pd.DataFrame]
        Chronologically ordered windows.  Each is a copy reset with a new
        default index.
    """
    if len(df) == 0:
        return []

    if "datetime" not in df.columns:
        raise ValueError("DataFrame must contain a 'datetime' column")

    window_days = int(
        window_days if window_days is not None
        else getattr(_cfg, "MONTHLY_WINDOW_DAYS", 30)
    )
    stride_days = int(
        stride_days if stride_days is not None
        else getattr(_cfg, "MONTHLY_WINDOW_STRIDE_DAYS", window_days)
    )
    min_rows = int(
        min_rows if min_rows is not None
        else getattr(_cfg, "MONTHLY_WINDOW_MIN_ROWS", 2500)
    )
    max_windows = int(
        max_windows if max_windows is not None
        else getattr(_cfg, "MONTHLY_WINDOW_MAX_WINDOWS", 24)
    )

    work = df.copy()
    dt = _datetime_series(work)
    work = work.assign(__dt=dt).dropna(subset=["__dt"])
    sort_cols = (
        ["__dt", "symbol"]
        if "symbol" in work.columns
        else ["__dt"]
    )
    work = work.sort_values(sort_cols).reset_index(drop=True)

    if work.empty:
        return []

    start = work["__dt"].min()
    end = work["__dt"].max()
    windows: list[pd.DataFrame] = []
    cur = start
    delta = pd.Timedelta(days=window_days)
    stride = pd.Timedelta(days=stride_days)

    while cur <= end and len(windows) < max_windows:
        nxt = cur + delta
        mask = (work["__dt"] >= cur) & (work["__dt"] < nxt)
        part = work.loc[mask].drop(columns=["__dt"])
        if len(part) >= min_rows:
            windows.append(part.reset_index(drop=True))
        cur += stride

    return windows


def summarize_monthly_metrics(metrics: Iterable[dict]) -> MonthlyWindowSummary:
    """Aggregate per-window metrics into a ``MonthlyWindowSummary``.

    Parameters
    ----------
    metrics:
        Iterable of dicts, each containing at least ``total_return_pct``,
        ``profit_factor``, ``max_drawdown_pct``, and ``executed_trades``.

    Returns
    -------
    MonthlyWindowSummary
        Zero-window summary when *metrics* is empty.
    """
    rows = list(metrics)
    if not rows:
        return MonthlyWindowSummary(
            windows=0,
            profitable_windows=0,
            profitable_ratio=0.0,
            mean_return_pct=0.0,
            median_return_pct=0.0,
            worst_return_pct=-100.0,
            latest_return_pct=-100.0,
            recency_weighted_return_pct=-100.0,
            mean_profit_factor=0.0,
            worst_profit_factor=0.0,
            worst_drawdown_pct=100.0,
            min_trades=0,
            mean_trades=0.0,
            equity_slope=-100.0,
            max_equity_dip_pct=100.0,
            score=-1e6,
        )

    returns = np.asarray(
        [float(m.get("total_return_pct", 0.0)) for m in rows], dtype=float
    )
    pfs = np.asarray(
        [float(m.get("profit_factor", 0.0)) for m in rows], dtype=float
    )
    dds = np.asarray(
        [float(m.get("max_drawdown_pct", 100.0)) for m in rows], dtype=float
    )
    trades = np.asarray(
        [int(m.get("executed_trades", 0)) for m in rows], dtype=np.int64
    )

    # Recency-weighted return: later windows get higher weight.
    recency_weight = float(getattr(_cfg, "MONTHLY_RECENCY_WEIGHT", 2.0))
    weights = np.linspace(1.0, recency_weight, len(returns))
    recency_return = float(np.average(returns, weights=weights))

    # Equity slope from cumulative product of returns.
    equity = np.cumprod(1.0 + returns / 100.0)
    if len(equity) >= 2:
        x = np.arange(len(equity), dtype=float)
        slope = float(np.polyfit(x, equity, 1)[0] * 100.0)
    else:
        slope = float(returns[-1])

    peak = np.maximum.accumulate(equity)
    dips = (peak - equity) / np.maximum(peak, 1e-12) * 100.0

    profitable = int(np.sum(returns > 0.0))
    profitable_ratio = profitable / max(1, len(returns))

    # Composite score (used internally for ranking).
    score = (
        0.65 * recency_return
        + 0.35 * float(np.mean(returns))
        + 0.75 * float(np.min(returns))
        + 5.0 * float(np.min(pfs))
        + 6.0 * profitable_ratio
        + 1.5 * slope
        - 0.65 * float(np.max(dds))
        - 0.20 * float(np.max(dips))
        - max(
            0.0,
            float(getattr(_cfg, "MONTHLY_MIN_TRADES", 20))
            - float(np.min(trades)),
        )
        * 0.08
    )

    return MonthlyWindowSummary(
        windows=len(returns),
        profitable_windows=profitable,
        profitable_ratio=float(profitable_ratio),
        mean_return_pct=float(np.mean(returns)),
        median_return_pct=float(np.median(returns)),
        worst_return_pct=float(np.min(returns)),
        latest_return_pct=float(returns[-1]),
        recency_weighted_return_pct=recency_return,
        mean_profit_factor=float(np.mean(pfs)),
        worst_profit_factor=float(np.min(pfs)),
        worst_drawdown_pct=float(np.max(dds)),
        min_trades=int(np.min(trades)),
        mean_trades=float(np.mean(trades)),
        equity_slope=slope,
        max_equity_dip_pct=float(np.max(dips)),
        score=float(score),
    )


def evaluate_rule_set_monthly(
    df: pd.DataFrame,
    rule_set: list[dict],
    direction: str,
    feature_names: list[str] | None = None,
) -> tuple[MonthlyWindowSummary, list[dict]]:
    """Evaluate a rule set on chronological monthly windows.

    Parameters
    ----------
    df:
        Full DataFrame (train + validation, or just the combined set).  Must
        contain ``datetime`` and label columns.
    rule_set:
        List of rule dicts as produced by Phase 3 (keys ``conditions``,
        ``tp``, ``sl``, ``capital_pct``).
    direction:
        ``"long"`` or ``"short"``.
    feature_names:
        Optional column whitelist for ``slim_backtest_df``.

    Returns
    -------
    tuple[MonthlyWindowSummary, list[dict]]
        Summary plus the raw per-window metrics list.
    """
    windows = build_monthly_windows(df)
    metrics: list[dict] = []
    for part in windows:
        try:
            slim = (
                slim_backtest_df(part, feature_names)
                if feature_names is not None
                else part
            )
            eng = CPUBacktestEngine(slim, {}, direction)
            metrics.append(eng.simulate_rule_set(rule_set))
        except Exception as exc:
            logger.debug("monthly window simulation failed: %s", exc)
            metrics.append(
                {
                    "total_return_pct": -100.0,
                    "profit_factor": 0.0,
                    "max_drawdown_pct": 100.0,
                    "executed_trades": 0,
                }
            )
    return summarize_monthly_metrics(metrics), metrics


def monthly_penalty(summary: MonthlyWindowSummary) -> float:
    """Compute a non-negative penalty for Phase 3/4 objective functions.

    Lower is better.  Returns ``100.0`` when *summary* has zero windows.

    Parameters
    ----------
    summary:
        Monthly window summary to penalise.

    Returns
    -------
    float
        Non-negative penalty value.
    """
    if summary.windows <= 0:
        return 100.0

    penalty = 0.0

    worst_return_floor = float(
        getattr(_cfg, "MONTHLY_WORST_RETURN_FLOOR", -2.0)
    )
    worst_return_weight = float(
        getattr(_cfg, "MONTHLY_WORST_RETURN_WEIGHT", 1.2)
    )
    penalty += (
        max(0.0, worst_return_floor - summary.worst_return_pct)
        * worst_return_weight
    )

    worst_pf_floor = float(getattr(_cfg, "MONTHLY_WORST_PF_FLOOR", 1.0))
    worst_pf_weight = float(getattr(_cfg, "MONTHLY_WORST_PF_WEIGHT", 8.0))
    penalty += (
        max(0.0, worst_pf_floor - summary.worst_profit_factor)
        * worst_pf_weight
    )

    max_dd = float(getattr(_cfg, "MONTHLY_MAX_DD", 8.0))
    dd_weight = float(getattr(_cfg, "MONTHLY_DD_WEIGHT", 0.7))
    penalty += (
        max(0.0, summary.worst_drawdown_pct - max_dd) * dd_weight
    )

    min_profitable_ratio = float(
        getattr(_cfg, "MONTHLY_MIN_PROFITABLE_RATIO", 0.55)
    )
    profitable_ratio_weight = float(
        getattr(_cfg, "MONTHLY_PROFITABLE_RATIO_WEIGHT", 15.0)
    )
    penalty += (
        max(0.0, min_profitable_ratio - summary.profitable_ratio)
        * profitable_ratio_weight
    )

    trend_weight = float(getattr(_cfg, "MONTHLY_TREND_WEIGHT", 2.0))
    penalty += max(0.0, -summary.equity_slope) * trend_weight

    latest_weight = float(getattr(_cfg, "MONTHLY_LATEST_WEIGHT", 0.6))
    penalty += max(0.0, -summary.latest_return_pct) * latest_weight

    min_trades = int(getattr(_cfg, "MONTHLY_MIN_TRADES", 20))
    penalty += max(0.0, min_trades - summary.min_trades) * 0.08

    return float(penalty)
