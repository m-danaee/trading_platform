"""Marginal contribution and effective rule-count helpers.

The helpers in this module are deliberately independent of a backtest engine.
The RB governor supplies exact full-portfolio and leave-one-out metrics, while
this module only defines the comparison contract used by reports and pruning.

Metric deltas use ``full - without_i``.  Therefore a negative MDD delta means
that the rule reduced drawdown, and a positive WorstMonth delta means that it
improved the weakest monthly return.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


# A small negative return change can be accepted when the rule improves risk.
# Returns in the repository are percentage points, so this is also expressed
# in percentage points.  Callers can override it for a stricter experiment.
DEFAULT_MAX_RETURN_DEGRADATION_PCT = 5.0
_EPSILON = 1.0e-12

_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "Return": (
        "Return",
        "return",
        "total_return_pct",
        "total_return",
        "return_pct",
    ),
    "Sortino": (
        "Sortino",
        "sortino",
        "sortino_ratio",
    ),
    "MDD": (
        "MDD",
        "mdd",
        "max_drawdown_pct",
        "max_drawdown",
        "drawdown_pct",
    ),
    "PF": (
        "PF",
        "pf",
        "profit_factor",
    ),
    "WorstMonth": (
        "WorstMonth",
        "worstmonth",
        "worst_month",
        "worst_month_return",
        "worst_month_return_pct",
        "worst_return_pct",
    ),
}


def _normalise_key(key: object) -> str:
    """Return a comparison-friendly metric key."""
    return str(key).strip().lower().replace("-", " ").replace("_", " ")


def _mapping_or_attributes(value: object) -> Mapping[object, object] | None:
    if isinstance(value, Mapping):
        return value
    as_dict = getattr(value, "to_dict", None)
    if callable(as_dict):
        try:
            converted = as_dict()
        except Exception:
            converted = None
        if isinstance(converted, Mapping):
            return converted
    try:
        attributes = vars(value)
    except TypeError:
        return None
    return attributes if isinstance(attributes, Mapping) else None


def _finite_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def _lookup_metric(metrics: object, canonical_name: str) -> float:
    """Read a metric from engine output or a report-style mapping.

    The engine uses snake-case names, while research reports use compact names
    such as ``Return`` and ``MDD``.  Supporting both keeps this pure helper
    useful for persisted reports and for direct CPU engine output.
    """
    mapping = _mapping_or_attributes(metrics)
    aliases = _METRIC_ALIASES[canonical_name]
    if mapping is not None:
        for alias in aliases:
            if alias in mapping:
                return _finite_float(mapping[alias])
        normalised = {
            _normalise_key(key): value for key, value in mapping.items()
        }
        for alias in aliases:
            value = normalised.get(_normalise_key(alias))
            if value is not None:
                return _finite_float(value)

        # WorstMonth is often nested in the monthly certificate.  Accept both
        # a summary object and a list of per-window metric dictionaries.
        if canonical_name == "WorstMonth":
            for nested_name in (
                "monthly",
                "monthly_summary",
                "monthly_metrics",
                "window_metrics",
            ):
                nested = mapping.get(nested_name)
                if nested is None:
                    continue
                nested_mapping = _mapping_or_attributes(nested)
                if nested_mapping is not None:
                    value = _lookup_metric(nested_mapping, canonical_name)
                    if value != 0.0 or any(
                        _normalise_key(key)
                        in {_normalise_key(alias) for alias in aliases}
                        for key in nested_mapping
                    ):
                        return value
                if isinstance(nested, (list, tuple)):
                    values = [
                        _lookup_metric(row, "Return")
                        for row in nested
                        if _mapping_or_attributes(row) is not None
                    ]
                    if values:
                        return float(np.min(values))
        return 0.0

    for alias in aliases:
        value = getattr(metrics, alias, None)
        if value is not None:
            return _finite_float(value)
    return 0.0


def marginal_contribution(
    portfolio_metrics: Mapping[object, object] | object,
    without_i: Mapping[object, object] | object,
    *,
    max_return_degradation_pct: float | None = None,
) -> dict[str, float | bool]:
    """Return the leave-one-out contribution of one rule.

    Parameters
    ----------
    portfolio_metrics:
        Exact metrics for the full ruleset.
    without_i:
        Exact metrics for the same ruleset with rule ``i`` removed.
    max_return_degradation_pct:
        Maximum accepted negative ``ΔReturn`` when a risk metric improves.
        The default is five percentage points.  A Sortino or MDD improvement
        is beneficial only when the return loss is not severe.

    Returns
    -------
    dict
        ``ΔReturn``, ``ΔSortino``, ``ΔMDD``, ``ΔPF``, ``ΔWorstMonth`` and the
        boolean ``is_beneficial``.  Every delta is ``full - without_i``.
    """
    deltas: dict[str, float] = {}
    for name in ("Return", "Sortino", "MDD", "PF", "WorstMonth"):
        full_value = _lookup_metric(portfolio_metrics, name)
        without_value = _lookup_metric(without_i, name)
        # Rounding removes binary floating point noise from JSON diagnostics
        # without changing the metric's meaningful precision.
        deltas[f"Δ{name}"] = round(full_value - without_value, 12)

    if max_return_degradation_pct is None:
        tolerance = DEFAULT_MAX_RETURN_DEGRADATION_PCT
    else:
        tolerance = _finite_float(max_return_degradation_pct)
    tolerance = max(0.0, tolerance)
    sortino_improved = deltas["ΔSortino"] >= -_EPSILON
    drawdown_improved = deltas["ΔMDD"] <= _EPSILON
    return_not_severely_degraded = deltas["ΔReturn"] >= -tolerance - _EPSILON
    deltas["is_beneficial"] = bool(
        (sortino_improved or drawdown_improved)
        and return_not_severely_degraded
    )
    return deltas


def effective_rule_count(R: object) -> float:
    """Return the participation-ratio effective number of rules.

    ``R`` may be a non-negative contribution vector or a square redundancy /
    correlation matrix.  For a matrix, the participation ratio is applied to
    its non-negative eigenvalues, which is equivalent to
    ``trace(R)**2 / trace(R @ R)`` for a valid symmetric positive-semidefinite
    matrix.  A one-rule input is always reported as one, including a zero
    contribution, and an empty input is reported as zero.
    """
    try:
        values = np.asarray(R, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("R must be a numeric vector or square matrix") from exc

    if values.ndim == 0:
        return 1.0 if np.isfinite(values.item()) else 0.0
    if values.ndim == 1:
        if values.size == 0:
            return 0.0
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return 1.0 if values.size == 1 else 0.0
        if values.size == 1:
            return 1.0
        # Participation is a count of active magnitudes; signed PnL
        # contributions must not cancel one another in the count.
        weights = np.abs(finite)
        denominator = float(np.sum(np.square(weights)))
        if denominator <= _EPSILON:
            return 0.0
        return float(np.sum(weights) ** 2 / denominator)

    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("R must be a vector or a square matrix")
    size = int(values.shape[0])
    if size == 0:
        return 0.0
    if size == 1:
        return 1.0

    # Non-finite matrix cells do not carry usable dependence evidence.  Treat
    # them as zero rather than allowing NaN to enter an audit report.
    clean = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    symmetric = (clean + clean.T) / 2.0
    try:
        eigenvalues = np.linalg.eigvalsh(symmetric)
    except np.linalg.LinAlgError as exc:
        raise ValueError("R must be numerically diagonalizable") from exc
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    denominator = float(np.sum(np.square(eigenvalues)))
    if denominator <= _EPSILON:
        return 0.0
    return float(np.sum(eigenvalues) ** 2 / denominator)


# Descriptive aliases for callers that use the long form.
compute_marginal_contribution = marginal_contribution
participation_ratio = effective_rule_count
effective_independent_rules = effective_rule_count


__all__ = [
    "DEFAULT_MAX_RETURN_DEGRADATION_PCT",
    "compute_marginal_contribution",
    "effective_independent_rules",
    "effective_rule_count",
    "marginal_contribution",
    "participation_ratio",
]
