
from __future__ import annotations

import math
from typing import Any


def _f(metrics: dict, key: str, default: float = 0.0) -> float:
    try:
        value = float(metrics.get(key, default))
    except Exception:
        value = default
    if not math.isfinite(value):
        return default
    return value


def _i(metrics: dict, key: str, default: int = 0) -> int:
    try:
        return int(metrics.get(key, default))
    except Exception:
        return default


def return_to_drawdown(return_pct: float, drawdown_pct: float, dd_floor: float = 1.0) -> float:
    """Return divided by max drawdown; higher is better.

    A small drawdown floor avoids exploding scores for tiny/no-trade drawdowns.
    Negative returns stay negative, which is what we want.
    """
    try:
        ret = float(return_pct)
        dd = abs(float(drawdown_pct))
    except Exception:
        return -1e6
    if not math.isfinite(ret) or not math.isfinite(dd):
        return -1e6
    return ret / max(dd, float(dd_floor), 1e-9)


def profit_factor_term(pf: float, cap: float = 3.0) -> float:
    try:
        pf = float(pf)
    except Exception:
        pf = 0.0
    if not math.isfinite(pf):
        pf = cap
    return max(0.0, min(pf, cap))


def trade_penalty(trades: int, min_trades: int, weight: float = 0.04) -> float:
    return max(0, int(min_trades) - int(trades)) * float(weight)


def robust_ratio_score(
    train_metrics: dict,
    valid_metrics: dict,
    fold_summary: Any | None = None,
    monthly_summary: Any | None = None,
    *,
    min_trades: int = 80,
    min_fold_trades: int = 20,
    dd_floor: float = 1.0,
    include_train_gap: bool = True,
) -> float:
    """Unified score used for Phase 3/4 and auto-search.

    The dominant term is validation return / validation max drawdown.  Worst-fold
    and monthly ratios are included to reduce candidates that only work in one
    contiguous segment.  Large train-vs-validation gaps are penalized heavily.
    """
    val_ret = _f(valid_metrics, "total_return_pct")
    val_dd = _f(valid_metrics, "max_drawdown_pct", 100.0)
    val_pf = _f(valid_metrics, "profit_factor", 0.0)
    val_wr = _f(valid_metrics, "win_rate", 0.0)
    val_trades = _i(valid_metrics, "executed_trades", 0)

    train_ret = _f(train_metrics or {}, "total_return_pct", 0.0)
    train_dd = _f(train_metrics or {}, "max_drawdown_pct", 100.0)
    train_pf = _f(train_metrics or {}, "profit_factor", 0.0)

    worst_ret = val_ret
    worst_dd = val_dd
    worst_pf = val_pf
    min_tr = val_trades
    if fold_summary is not None:
        worst_ret = float(getattr(fold_summary, "worst_return_pct", worst_ret))
        worst_dd = float(getattr(fold_summary, "worst_drawdown_pct", worst_dd))
        worst_pf = float(getattr(fold_summary, "worst_profit_factor", worst_pf))
        min_tr = int(getattr(fold_summary, "min_trades", min_tr))

    val_ratio = return_to_drawdown(val_ret, val_dd, dd_floor)
    worst_ratio = return_to_drawdown(worst_ret, worst_dd, dd_floor)
    train_ratio = return_to_drawdown(train_ret, train_dd, dd_floor)

    score = (
        18.0 * val_ratio
        + 10.0 * worst_ratio
        + 1.20 * val_ret
        + 0.60 * worst_ret
        + 5.0 * profit_factor_term(val_pf)
        + 6.0 * profit_factor_term(worst_pf)
        + 0.035 * val_wr
        - 0.20 * val_dd
        - 0.35 * worst_dd
        - trade_penalty(val_trades, min_trades, 0.035)
        - trade_penalty(min_tr, min_fold_trades, 0.09)
    )

    if include_train_gap:
        score -= max(0.0, train_ret - val_ret) * 0.45
        score -= max(0.0, train_pf - val_pf) * 3.0
        score -= max(0.0, train_ratio - val_ratio) * 8.0

    if monthly_summary is not None:
        m_score = float(getattr(monthly_summary, "score", 0.0))
        m_worst_ret = float(getattr(monthly_summary, "worst_return_pct", 0.0))
        m_worst_dd = float(getattr(monthly_summary, "worst_drawdown_pct", 100.0))
        m_worst_pf = float(getattr(monthly_summary, "worst_profit_factor", 0.0))
        m_prof = float(getattr(monthly_summary, "profitable_ratio", 0.0))
        m_slope = float(getattr(monthly_summary, "equity_slope", 0.0))
        m_latest = float(getattr(monthly_summary, "latest_return_pct", 0.0))
        m_ratio = return_to_drawdown(m_worst_ret, m_worst_dd, dd_floor)
        score += 0.65 * m_score + 8.0 * m_ratio + 0.60 * m_latest + 4.0 * m_prof + 0.75 * m_slope
        score += 3.0 * profit_factor_term(m_worst_pf)
        score -= max(0.0, 0.60 - m_prof) * 16.0
        score -= max(0.0, 0.85 - m_worst_pf) * 10.0
        score -= max(0.0, -m_slope) * 3.0

    return float(score)
