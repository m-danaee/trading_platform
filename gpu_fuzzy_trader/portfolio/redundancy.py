"""Redundancy matrices for candidate rules."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from gpu_fuzzy_trader.portfolio.correlation import pnl_correlation, signal_overlap


def _ordered_values(values: Iterable[object] | Mapping[object, object]) -> list[object]:
    if isinstance(values, Mapping):
        return [values[key] for key in values]
    return list(values)


def _safe_pnl_correlation(left: object, right: object) -> float:
    try:
        return float(pnl_correlation(left, right))
    except (TypeError, ValueError):
        # Fold evidence can be absent for one candidate.  Signal overlap still
        # remains useful, but missing PnL evidence must not become correlation.
        return 0.0


def pair_redundancy(
    left_signal: Iterable[object] | np.ndarray,
    right_signal: Iterable[object] | np.ndarray,
    left_pnl: Iterable[object] | np.ndarray | None = None,
    right_pnl: Iterable[object] | np.ndarray | None = None,
    *,
    signal_weight: float = 0.5,
    pnl_weight: float = 0.5,
) -> float:
    """Return ``Rij = 0.5*Overlap + 0.5*max(0, PnLCorr)`` by default."""
    signal_weight = float(signal_weight)
    pnl_weight = float(pnl_weight)
    if (
        not np.isfinite(signal_weight)
        or not np.isfinite(pnl_weight)
        or signal_weight < 0.0
        or pnl_weight < 0.0
    ):
        raise ValueError("redundancy weights must be non-negative")
    total_weight = signal_weight + pnl_weight
    if total_weight <= 0.0:
        raise ValueError("at least one redundancy weight must be positive")

    overlap = signal_overlap(left_signal, right_signal)
    pnl_corr = 0.0
    if left_pnl is not None and right_pnl is not None:
        pnl_corr = max(0.0, _safe_pnl_correlation(left_pnl, right_pnl))
    value = (
        signal_weight * overlap + pnl_weight * pnl_corr
    ) / total_weight
    return float(np.clip(value, 0.0, 1.0))


def redundancy_matrix(
    signal_masks: Sequence[Iterable[object] | np.ndarray]
    | Mapping[object, Iterable[object] | np.ndarray],
    pnl_series: Sequence[Iterable[object] | np.ndarray | None]
    | Mapping[object, Iterable[object] | np.ndarray | None]
    | None = None,
    *,
    signal_weight: float = 0.5,
    pnl_weight: float = 0.5,
) -> np.ndarray:
    """Build a symmetric candidate redundancy matrix.

    ``signal_masks`` and ``pnl_series`` are candidate-major.  PnL series are
    optional; when they are absent, the signal term remains available and the
    PnL term is zero.  The diagonal is always one because a rule is fully
    redundant with itself.
    """
    signals = _ordered_values(signal_masks)
    if isinstance(pnl_series, Mapping):
        if isinstance(signal_masks, Mapping):
            pnl = [pnl_series.get(key) for key in signal_masks]
        else:
            pnl = list(pnl_series.values())
    elif pnl_series is None:
        pnl = [None] * len(signals)
    else:
        pnl = list(pnl_series)
    if len(pnl) != len(signals):
        raise ValueError("signal_masks and pnl_series must have the same length")

    size = len(signals)
    matrix = np.zeros((size, size), dtype=float)
    for left_index in range(size):
        matrix[left_index, left_index] = 1.0
        for right_index in range(left_index + 1, size):
            value = pair_redundancy(
                signals[left_index],
                signals[right_index],
                pnl[left_index],
                pnl[right_index],
                signal_weight=signal_weight,
                pnl_weight=pnl_weight,
            )
            matrix[left_index, right_index] = value
            matrix[right_index, left_index] = value
    return matrix


def stable_corr(
    correlations: Iterable[float] | np.ndarray,
    alpha: float = 0.25,
    axis: int | None = 0,
) -> float | np.ndarray:
    """Return the stable correlation ``median + alpha * standard deviation``.

    A two-dimensional or higher-dimensional input is reduced along ``axis``.
    Non-finite observations are ignored.  Empty evidence returns zero rather
    than an invalid NaN that could silently change selection.
    """
    alpha_value = float(alpha)
    if alpha_value < 0.0 or not np.isfinite(alpha_value):
        raise ValueError("alpha must be finite and non-negative")
    array = np.asarray(correlations, dtype=float)
    if array.ndim == 0:
        return float(array) if np.isfinite(array) else 0.0
    with np.errstate(invalid="ignore"):
        median = np.nanmedian(array, axis=axis)
        deviation = np.nanstd(array, axis=axis)
    result = np.asarray(median + alpha_value * deviation, dtype=float)
    result = np.where(np.isfinite(result), result, 0.0)
    if result.ndim == 0:
        return float(result)
    return result


def stable_redundancy_matrix(
    fold_signal_masks: Sequence[
        Sequence[Iterable[object] | np.ndarray]
    ],
    fold_pnl_series: Sequence[
        Sequence[Iterable[object] | np.ndarray | None] | None
    ]
    | None = None,
    *,
    alpha: float = 0.25,
    signal_weight: float = 0.5,
    pnl_weight: float = 0.5,
) -> np.ndarray:
    """Build one redundancy matrix per fold and combine them stably."""
    signal_folds = list(fold_signal_masks)
    if not signal_folds:
        return np.zeros((0, 0), dtype=float)
    if fold_pnl_series is not None and len(fold_pnl_series) != len(signal_folds):
        raise ValueError("fold signal and PnL evidence must have the same length")
    matrices: list[np.ndarray] = []
    for fold_index, masks in enumerate(signal_folds):
        pnl = None if fold_pnl_series is None else fold_pnl_series[fold_index]
        matrices.append(
            redundancy_matrix(
                masks,
                pnl,
                signal_weight=signal_weight,
                pnl_weight=pnl_weight,
            )
        )
    stacked = np.stack(matrices, axis=0)
    stable = np.asarray(stable_corr(stacked, alpha=alpha, axis=0), dtype=float)
    # Stable uncertainty can move a value above one.  Redundancy is a bounded
    # edge weight, so keep the matrix a valid threshold-graph input.
    return np.clip(stable, 0.0, 1.0)


# Descriptive aliases used by reports and callers.
build_redundancy_matrix = redundancy_matrix
compute_redundancy_matrix = redundancy_matrix
calculate_redundancy = pair_redundancy
stable_redundancy = stable_redundancy_matrix


__all__ = [
    "compute_redundancy_matrix",
    "build_redundancy_matrix",
    "calculate_redundancy",
    "pair_redundancy",
    "redundancy_matrix",
    "stable_corr",
    "stable_redundancy",
    "stable_redundancy_matrix",
]
