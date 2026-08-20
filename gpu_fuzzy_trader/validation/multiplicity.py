"""Selection-multiplicity diagnostics for strategy research artifacts."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def estimate_pbo(
    in_sample_scores: Sequence[Sequence[float]],
    out_of_sample_scores: Sequence[Sequence[float]],
) -> float | None:
    """Estimate the fraction of folds where the IS winner misses OOS median.

    Inputs are ``fold -> candidate scores``.  A single candidate is reported as
    unavailable rather than pretending PBO can be estimated.
    """
    if len(in_sample_scores) != len(out_of_sample_scores):
        raise ValueError("IS and OOS fold counts must match")
    if not in_sample_scores or any(
        len(row) < 2 for row in in_sample_scores
    ):
        return None
    misses = 0
    used = 0
    for is_row, oos_row in zip(in_sample_scores, out_of_sample_scores, strict=False):
        if len(is_row) != len(oos_row) or len(oos_row) < 2:
            continue
        winner = int(np.argmax(np.asarray(is_row, dtype=float)))
        median_oos = float(np.median(np.asarray(oos_row, dtype=float)))
        misses += float(oos_row[winner]) < median_oos
        used += 1
    return float(misses / used) if used else None


def deflated_sharpe_ratio(
    observed_sharpe: float,
    *,
    n_trials: int,
    n_observations: int,
    skewness: float = 0.0,
    excess_kurtosis: float = 0.0,
) -> float:
    """Return a normal-approximation deflated Sharpe probability.

    This is a conservative diagnostic, not a substitute for a full CSCV/PBO
    implementation.  The expected maximum null Sharpe grows with the number
    of trials, so the observed score is discounted before conversion to a
    probability.
    """
    n = max(2, int(n_observations))
    trials = max(1, int(n_trials))
    expected_max = math.sqrt(2.0 * math.log(trials)) / math.sqrt(n)
    variance = max(
        1.0e-12,
        (1.0 - skewness * observed_sharpe
         + ((excess_kurtosis - 1.0) / 4.0) * observed_sharpe**2)
        / n,
    )
    z = (float(observed_sharpe) - expected_max) / math.sqrt(variance)
    return float(0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))


def summarize_multiplicity(
    *,
    fold_returns: Iterable[float],
    n_trials: int,
) -> dict[str, float | int | None]:
    """Create a compact ledger-ready multiplicity report."""
    returns = np.asarray(list(fold_returns), dtype=float)
    if returns.size == 0:
        return {
            "trial_count": int(n_trials),
            "observations": 0,
            "observed_sharpe_proxy": 0.0,
            "deflated_sharpe_probability": 0.0,
            "pbo": None,
        }
    scale = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
    observed = float(np.mean(returns) / scale) if scale > 0.0 else float(
        np.mean(returns)
    )
    return {
        "trial_count": int(n_trials),
        "observations": int(returns.size),
        "observed_sharpe_proxy": observed,
        "deflated_sharpe_probability": deflated_sharpe_ratio(
            observed,
            n_trials=int(n_trials),
            n_observations=int(returns.size),
        ),
        "pbo": None,
        "pbo_note": "requires candidate-by-fold score matrix",
    }
