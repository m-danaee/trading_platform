"""Directional and conditional evaluators for Hierarchical MTF rule discovery."""

from __future__ import annotations

from typing import Union
import numpy as np
import pandas as pd


def compute_forward_movement_labels(
    close: Union[np.ndarray, pd.Series],
    atr: Union[np.ndarray, pd.Series],
    horizon_bars: int,
) -> np.ndarray:
    """Compute ATR-normalized forward price movement labels.

    Formula:
        move_{t, K} = (Close_{t + K} - Close_t) / ATR_t

    Args:
        close: 1D array of bar close prices.
        atr: 1D array of Average True Range values.
        horizon_bars: Forward lookahead horizon in bars (K >= 1).

    Returns:
        1D np.ndarray of float64 forward movement values with trailing NaNs.
    """
    c = np.asarray(close, dtype=np.float64)
    a = np.asarray(atr, dtype=np.float64)
    n = len(c)

    moves = np.full(n, np.nan, dtype=np.float64)
    if horizon_bars <= 0 or n <= horizon_bars:
        return moves

    valid_atr = np.isfinite(a[:-horizon_bars]) & (a[:-horizon_bars] > 1e-12)
    diff = c[horizon_bars:] - c[:-horizon_bars]
    valid_idx = np.where(valid_atr)[0]

    moves[valid_idx] = diff[valid_idx] / a[valid_idx]
    return moves


def fit_directional_threshold(
    move: Union[np.ndarray, pd.Series],
    quantile: float = 0.60,
) -> float:
    """Fit positive classification threshold theta via quantile on train set movements.

    Args:
        move: 1D array of continuous forward movement values.
        quantile: Quantile in (0, 1) for symmetric thresholding.

    Returns:
        Positive float threshold theta.
    """
    m = np.asarray(move, dtype=np.float64)
    clean_moves = np.abs(m[np.isfinite(m)])
    if len(clean_moves) == 0:
        return 1.0

    q = float(np.clip(quantile, 0.01, 0.99))
    theta = float(np.quantile(clean_moves, q))
    return max(theta, 1e-6)


def classify_directional_labels(
    move: Union[np.ndarray, pd.Series],
    theta: float,
) -> np.ndarray:
    """Classify continuous normalized movement into discrete directional labels.

    - +1: LONG_FAVORABLE (move > +theta)
    - -1: SHORT_FAVORABLE (move < -theta)
    -  0: NEUTRAL / INDETERMINATE

    Args:
        move: 1D array of continuous normalized movements.
        theta: Positive directional threshold.

    Returns:
        1D np.ndarray of int32 labels.
    """
    m = np.asarray(move, dtype=np.float64)
    labels = np.zeros(len(m), dtype=np.int32)
    pos_mask = np.isfinite(m) & (m > theta)
    neg_mask = np.isfinite(m) & (m < -theta)
    labels[pos_mask] = 1
    labels[neg_mask] = -1
    return labels


def compute_conditional_mwc_labels(
    move: Union[np.ndarray, pd.Series],
    hwc_score: Union[np.ndarray, pd.Series],
    theta_mwc: float,
    direction: str = "long",
    support_threshold: float = 0.20,
) -> np.ndarray:
    """Compute conditional target labels for MWC conditioned on upstream HWC score.

    Args:
        move: 1D array of forward movements on MWC timeframe.
        hwc_score: 1D array of upstream HWC directional scores in [-1, +1].
        theta_mwc: MWC directional threshold.
        direction: "long" or "short".
        support_threshold: Minimum HWC score magnitude required for directional support.

    Returns:
        1D np.ndarray of int32 conditional target labels.
    """
    m = np.asarray(move, dtype=np.float64)
    h = np.asarray(hwc_score, dtype=np.float64)
    labels = np.zeros(len(m), dtype=np.int32)

    is_long = direction.lower() in ("long", "1", "buy")
    if is_long:
        eligible = np.isfinite(h) & (h >= support_threshold) & np.isfinite(m)
        target = eligible & (m > theta_mwc)
        labels[target] = 1
    else:
        eligible = np.isfinite(h) & (h <= -support_threshold) & np.isfinite(m)
        target = eligible & (m < -theta_mwc)
        labels[target] = -1

    return labels


def evaluate_directional_rule(
    active_mask: Union[np.ndarray, pd.Series],
    labels: Union[np.ndarray, pd.Series],
    direction: str = "long",
    target_coverage: tuple[float, float] = (0.20, 0.60),
    base_rate: float | None = None,
) -> tuple[float, float, float]:
    """Vectorized evaluation of a directional rule.

    Calculates:
    1. Directional Edge = Precision - Base Rate
    2. Matthews Correlation Coefficient (MCC) on the active mask
    3. Continuous Soft Coverage Penalty outside [C_min, C_max]

    Args:
        active_mask: Boolean array indicating when the rule triggers.
        labels: 1D array of directional labels (+1, -1, 0).
        direction: "long" or "short".
        target_coverage: Tuple of (C_min, C_max) bounds in [0, 1].
        base_rate: Optional baseline prevalence. If None, computed from labels.

    Returns:
        Tuple of (directional_edge, mcc, coverage_penalty).
    """
    mask = np.asarray(active_mask, dtype=bool)
    lbl = np.asarray(labels, dtype=np.int32)
    n_total = len(mask)

    if n_total == 0 or len(lbl) != n_total:
        return 0.0, 0.0, 1.0

    target_label = 1 if direction.lower() in ("long", "1", "buy") else -1
    target_mask = lbl == target_label

    n_active = int(np.sum(mask))
    coverage = n_active / n_total

    if base_rate is None:
        base_rate = float(np.mean(target_mask)) if n_total > 0 else 0.0

    # 1. Directional Edge
    if n_active == 0:
        directional_edge = 0.0
    else:
        tp = int(np.sum(mask & target_mask))
        precision = tp / n_active
        directional_edge = precision - base_rate

    # 2. Matthews Correlation Coefficient (MCC)
    tp = int(np.sum(mask & target_mask))
    fp = int(np.sum(mask & ~target_mask))
    fn = int(np.sum(~mask & target_mask))
    tn = int(np.sum(~mask & ~target_mask))

    numerator = float(tp * tn - fp * fn)
    denominator = float(
        np.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
    )
    mcc = (numerator / denominator) if denominator > 0.0 else 0.0

    # 3. Soft Continuous Coverage Penalty
    c_min, c_max = target_coverage
    if coverage < c_min:
        cov_penalty = (c_min - coverage) / c_min if c_min > 0.0 else 0.0
    elif coverage > c_max:
        cov_penalty = (coverage - c_max) / (1.0 - c_max) if c_max < 1.0 else 0.0
    else:
        cov_penalty = 0.0
    cov_penalty = float(np.clip(cov_penalty, 0.0, 1.0))

    return float(directional_edge), float(mcc), float(cov_penalty)


def evaluate_conditional_directional_rule(
    active_mask: Union[np.ndarray, pd.Series],
    move: Union[np.ndarray, pd.Series],
    hwc_score: Union[np.ndarray, pd.Series],
    theta_mwc: float,
    direction: str = "long",
    support_threshold: float = 0.20,
    target_coverage: tuple[float, float] = (0.10, 0.40),
) -> tuple[float, float, float]:
    """Evaluate candidate MWC rule under conditional HWC guidance.

    Args:
        active_mask: Boolean array indicating when the rule triggers.
        move: 1D array of forward movements on MWC timeframe.
        hwc_score: 1D array of upstream HWC directional scores in [-1, +1].
        theta_mwc: MWC directional threshold.
        direction: "long" or "short".
        support_threshold: Minimum HWC score magnitude required for directional support.
        target_coverage: Tuple of (C_min, C_max) bounds for MWC coverage.

    Returns:
        Tuple of (directional_edge, mcc, coverage_penalty).
    """
    cond_labels = compute_conditional_mwc_labels(
        move=move,
        hwc_score=hwc_score,
        theta_mwc=theta_mwc,
        direction=direction,
        support_threshold=support_threshold,
    )
    return evaluate_directional_rule(
        active_mask=active_mask,
        labels=cond_labels,
        direction=direction,
        target_coverage=target_coverage,
    )
