"""Small, deterministic correlation primitives for RB portfolio selection.

The functions in this module do not know about RB records.  They accept simple
one-dimensional array-like values so they can be used by reports and tests as
well as by the governor.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def _as_1d(values: Iterable[object] | np.ndarray, *, dtype: object) -> np.ndarray:
    """Return a one-dimensional array without changing item order."""
    array = np.asarray(values, dtype=dtype)
    return array.reshape(-1)


def _finite_pair(
    left: Iterable[object] | np.ndarray,
    right: Iterable[object] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite numeric pairs and reject silent length mismatches."""
    left_array = _as_1d(left, dtype=float)
    right_array = _as_1d(right, dtype=float)
    if left_array.size != right_array.size:
        raise ValueError(
            "correlation inputs must have the same number of observations"
        )
    finite = np.isfinite(left_array) & np.isfinite(right_array)
    return left_array[finite], right_array[finite]


def signal_overlap(
    left: Iterable[object] | np.ndarray,
    right: Iterable[object] | np.ndarray,
) -> float:
    """Return the Jaccard overlap of two signal masks.

    A true value means that a rule emits a signal at that observation.  An
    empty union has no observed overlap and therefore returns ``0.0``.  Input
    lengths must match because a signal at one timestamp cannot be compared to
    a different timestamp.
    """
    left_array = _as_1d(left, dtype=bool)
    right_array = _as_1d(right, dtype=bool)
    if left_array.size != right_array.size:
        raise ValueError("signal masks must have the same number of observations")
    union = left_array | right_array
    union_count = int(np.count_nonzero(union))
    if union_count == 0:
        return 0.0
    intersection_count = int(np.count_nonzero(left_array & right_array))
    return float(intersection_count / union_count)


def pnl_correlation(
    left: Iterable[object] | np.ndarray,
    right: Iterable[object] | np.ndarray,
    *,
    min_periods: int = 2,
) -> float:
    """Return Pearson correlation for aligned PnL observations.

    Missing or non-finite pairs are ignored.  Pearson correlation is undefined
    for fewer than ``min_periods`` observations or a constant series; both
    cases return ``0.0`` so a missing correlation does not create a false
    redundancy signal.
    """
    if int(min_periods) < 2:
        raise ValueError("min_periods must be at least 2")
    left_array, right_array = _finite_pair(left, right)
    if left_array.size < int(min_periods):
        return 0.0
    left_centered = left_array - float(np.mean(left_array))
    right_centered = right_array - float(np.mean(right_array))
    left_norm = float(np.sqrt(np.dot(left_centered, left_centered)))
    right_norm = float(np.sqrt(np.dot(right_centered, right_centered)))
    if left_norm <= np.finfo(float).eps or right_norm <= np.finfo(float).eps:
        return 0.0
    value = float(np.dot(left_centered, right_centered) / (left_norm * right_norm))
    if abs(value - 1.0) <= 1e-12:
        return 1.0
    if abs(value + 1.0) <= 1e-12:
        return -1.0
    return float(np.clip(value, -1.0, 1.0))


def downside_dependence(
    left: Iterable[object] | np.ndarray,
    right: Iterable[object] | np.ndarray,
    *,
    threshold: float = 0.0,
) -> float:
    """Return co-movement of losses below ``threshold``.

    The measure is the cosine similarity of the two lower-partial-loss
    vectors.  It is in ``[0, 1]``: zero means that no common downside evidence
    exists, while one means that the observed downside magnitudes are
    proportional.  This is stable when one series has no downside observations
    and avoids treating two profitable observations as downside agreement.
    """
    left_array, right_array = _finite_pair(left, right)
    if left_array.size == 0:
        return 0.0
    threshold_value = float(threshold)
    left_loss = np.maximum(threshold_value - left_array, 0.0)
    right_loss = np.maximum(threshold_value - right_array, 0.0)
    left_norm = float(np.sqrt(np.dot(left_loss, left_loss)))
    right_norm = float(np.sqrt(np.dot(right_loss, right_loss)))
    if left_norm <= np.finfo(float).eps or right_norm <= np.finfo(float).eps:
        return 0.0
    value = float(np.dot(left_loss, right_loss) / (left_norm * right_norm))
    return float(np.clip(value, 0.0, 1.0))


__all__ = ["downside_dependence", "pnl_correlation", "signal_overlap"]
