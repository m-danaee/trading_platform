"""Report-only uncertainty diagnostics for realized trade returns.

The intervals are deliberately not used for strategy selection.  Trades can
overlap and are not independent, so the report calls the result a diagnostic
and uses a moving-block bootstrap instead of claiming a formal backtest test.
"""

from __future__ import annotations

from math import ceil, lgamma, log, sqrt
from typing import Any

import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config as _cfg


def _finite_trade_returns_pct(trade_log: pd.DataFrame | None) -> np.ndarray:
    """Extract net account-return percentages from realized trade logs."""
    if trade_log is None or trade_log.empty:
        return np.empty(0, dtype=float)
    if "Net_PnL" not in trade_log.columns:
        return np.empty(0, dtype=float)

    frame = trade_log
    if "Realized" in frame.columns:
        frame = frame.loc[frame["Realized"].fillna(False).astype(bool)]
    pnl = pd.to_numeric(frame["Net_PnL"], errors="coerce").to_numpy(dtype=float)
    if "Equity_Before_Entry" in frame.columns:
        denominator = pd.to_numeric(
            frame["Equity_Before_Entry"], errors="coerce",
        ).to_numpy(dtype=float)
        valid = np.isfinite(pnl) & np.isfinite(denominator) & (denominator > 0.0)
        return (pnl[valid] / denominator[valid]) * 100.0
    return np.empty(0, dtype=float)


def _two_sided_sign_test_pvalue(values: np.ndarray) -> float:
    """Return an exact two-sided sign-test p-value for non-zero values."""
    nonzero = values[np.abs(values) > 0.0]
    n = len(nonzero)
    if n == 0:
        return 1.0
    wins = int(np.sum(nonzero > 0.0))
    k = min(wins, n - wins)
    # Sum the lower binomial tail in log space. This remains stable for large
    # trade counts and avoids requiring SciPy for a report-only statistic.
    terms = np.asarray(
        [
            lgamma(n + 1.0)
            - lgamma(i + 1.0)
            - lgamma(n - i + 1.0)
            - n * log(2.0)
            for i in range(k + 1)
        ],
        dtype=float,
    )
    maximum = float(np.max(terms))
    lower_tail = float(np.exp(maximum) * np.exp(terms - maximum).sum())
    return float(min(1.0, 2.0 * lower_tail))


def _bootstrap_sample(
    values: np.ndarray,
    rng: np.random.Generator,
    block_length: int,
) -> np.ndarray:
    """Draw one circular moving-block bootstrap sample."""
    n = len(values)
    n_blocks = ceil(n / block_length)
    starts = rng.integers(0, n, size=n_blocks)
    offsets = np.arange(block_length, dtype=np.int64)
    sample = np.concatenate(
        [values[(start + offsets) % n] for start in starts]
    )
    return sample[:n]


def compute_trade_uncertainty(
    trade_log: pd.DataFrame | None,
    *,
    seed: int = 42,
    samples: int | None = None,
    block_length: int | None = None,
) -> dict[str, Any]:
    """Compute deterministic trade-level uncertainty diagnostics.

    ``mean_trade_return_pct`` and its percentile interval are account-return
    percentages. ``compound_return_pct`` is the sequential compounded return
    of the sampled trade-return sequence. The interval is descriptive only;
    it does not account for all market, execution, or model-selection risks.
    """
    values = _finite_trade_returns_pct(trade_log)
    n_trades = len(values)
    if n_trades == 0:
        return {
            "status": "no_realized_trades",
            "trade_count": 0,
            "mean_trade_return_pct": 0.0,
            "mean_trade_return_ci95_pct": [0.0, 0.0],
            "compound_return_ci95_pct": [0.0, 0.0],
            "sign_test_p_value": 1.0,
            "bootstrap_samples": 0,
            "block_length": 0,
        }

    requested_samples = int(
        getattr(_cfg, "REPORT_BOOTSTRAP_SAMPLES", 1000)
        if samples is None else samples
    )
    requested_samples = max(0, requested_samples)
    configured_block = int(
        getattr(_cfg, "REPORT_BOOTSTRAP_BLOCK_LENGTH", 0)
        if block_length is None else block_length
    )
    resolved_block = configured_block if configured_block > 0 else max(
        1, min(20, int(sqrt(n_trades)))
    )
    resolved_block = min(resolved_block, n_trades)
    point_mean = float(np.mean(values))

    if requested_samples == 0:
        return {
            "status": "point_estimate_only",
            "trade_count": n_trades,
            "mean_trade_return_pct": point_mean,
            "mean_trade_return_ci95_pct": [point_mean, point_mean],
            "compound_return_ci95_pct": [
                float(np.prod(1.0 + values / 100.0) - 1.0) * 100.0,
            ] * 2,
            "sign_test_p_value": _two_sided_sign_test_pvalue(values),
            "bootstrap_samples": 0,
            "block_length": resolved_block,
        }

    rng = np.random.default_rng(int(seed))
    means = np.empty(requested_samples, dtype=float)
    compounded = np.empty(requested_samples, dtype=float)
    for index in range(requested_samples):
        sample = _bootstrap_sample(values, rng, resolved_block)
        means[index] = float(np.mean(sample))
        growth = 1.0 + sample / 100.0
        if np.any(growth <= 0.0) or not np.isfinite(growth).all():
            compounded[index] = -100.0
        else:
            compounded[index] = float(np.prod(growth) - 1.0) * 100.0

    quantiles = (2.5, 97.5)
    return {
        "status": "diagnostic",
        "trade_count": n_trades,
        "mean_trade_return_pct": point_mean,
        "mean_trade_return_ci95_pct": [
            float(value) for value in np.percentile(means, quantiles)
        ],
        "compound_return_ci95_pct": [
            float(value) for value in np.percentile(compounded, quantiles)
        ],
        "sign_test_p_value": _two_sided_sign_test_pvalue(values),
        "bootstrap_samples": requested_samples,
        "block_length": resolved_block,
        "dependence_note": (
            "Descriptive moving-block bootstrap; overlapping trades and model "
            "selection mean this is not a formal independent-trades test."
        ),
    }


__all__ = ["compute_trade_uncertainty"]
