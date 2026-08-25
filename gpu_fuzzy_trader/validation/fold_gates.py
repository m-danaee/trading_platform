"""Fold-aware count gates.

Count evidence changes with the slice that is being scored.  This module keeps
that change separate from quality thresholds: count gates scale with effective
row exposure, while quality gates such as profit factor and MCC remain fixed.

``build_fold_gate_context`` must be called for the data used by the gate.  For
train gates, pass the train scoring frame; for OOF gates, pass the OOF scoring
frame.  Do not build one context from train data and reuse it for OOF gates.
The reference frame is normally the full reference universe for that same
stage.

The exposure is row based by contract.  Duration and per-symbol counts are
included for audit output and for future policies, but they do not silently
replace row exposure in the count-gate formula.  Pooled and macro aggregation
may use the same context, but the selected aggregation policy must be explicit
at the call site.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import ceil, isfinite
from typing import Any

import pandas as pd

from gpu_fuzzy_trader.mtf.cross_fitting import FoldExposure


# Keep this allowlist explicit.  In particular, do not scale every key whose
# name happens to contain ``min``: quality floors may also use that prefix.
COUNT_GATE_ALLOWLIST = frozenset(
    {
        "MinTrades",
        "MinSignals",
        "MinSupport",
        "MinCandidateSupport",
        "MinTradesPerSymbol",
        "min_trades",
        "min_signals",
        "min_support",
        "min_candidate_support",
        "min_trades_per_symbol",
        "min_trade_support",
        "min_trade_pool_floor",
        "min_train_trades",
        "min_valid_trades",
        "min_val_trades",
        "min_candidate_trades",
        "MIN_TRADES",
        "MIN_SIGNALS",
        "MIN_SUPPORT",
        "MIN_CANDIDATE_SUPPORT",
        "MIN_TRADES_PER_SYMBOL",
    }
)

# This list is documentation and an audit aid.  Resolution is deliberately
# driven by COUNT_GATE_ALLOWLIST; every other key is copied unchanged.
QUALITY_GATE_ALLOWLIST = frozenset(
    {
        "PF",
        "MCC",
        "MDD",
        "Edge",
        "WinRate",
        "Sortino",
        "Return",
        "profit_factor",
        "profit_factor_floor",
        "mcc",
        "mcc_floor",
        "edge",
        "edge_floor",
        "win_rate",
        "win_rate_floor",
        "sortino",
        "sortino_ratio",
        "sortino_floor",
        "mdd",
        "max_drawdown",
        "max_drawdown_pct",
        "mdd_floor",
        "return",
        "return_pct",
        "return_pct_floor",
        "total_return_pct",
    }
)


def _default_absolute_min() -> int:
    """Read the absolute count floor lazily to avoid a config import cycle."""
    try:
        from gpu_fuzzy_trader import config as cfg

        return int(getattr(cfg, "FOLD_ABSOLUTE_MIN_TRADES", 5))
    except (ImportError, AttributeError, TypeError, ValueError):
        return 5


def _effective_rows(exposure: FoldExposure | Mapping[str, Any] | Any) -> int:
    """Return non-negative effective rows from an exposure-like value."""
    if isinstance(exposure, Mapping):
        value = exposure.get("effective_rows", exposure.get("rows", 0))
    else:
        value = getattr(exposure, "effective_rows", None)
        if value is None:
            value = getattr(exposure, "rows", 0)
        if callable(value):
            value = value()
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("fold exposure rows must be an integer") from exc


def scale_count_gate(
    base: int,
    fold_exposure: FoldExposure,
    reference_exposure: FoldExposure,
    absolute_min: int | None = None,
    *,
    rounding: str = "ceil",
) -> int:
    """Scale a count gate by fold exposure.

    The normal contract is::

        max(absolute_min, ceil(base * Ef / Eref))

    where ``Ef`` is the fold (or scoring) effective row count and ``Eref`` is
    the reference row count.  A non-positive fold has no count evidence and
    therefore resolves to the absolute floor.  A non-positive reference has
    no meaningful ratio; in that case the base is retained, subject to the
    floor, instead of dividing by zero.

    ``rounding="legacy"`` is only for the deprecated config shim.  It keeps
    the historical ``round`` result for old callers while all new callers use
    the required ceiling formula.
    """
    floor = _default_absolute_min() if absolute_min is None else int(absolute_min)
    base_count = int(base)
    fold_rows = _effective_rows(fold_exposure)
    reference_rows = _effective_rows(reference_exposure)

    if reference_rows <= 0:
        return max(floor, base_count)
    if fold_rows <= 0:
        return floor

    scaled_value = base_count * fold_rows / reference_rows
    if rounding == "ceil":
        scaled = ceil(scaled_value)
    elif rounding == "legacy":
        scaled = int(round(scaled_value))
    else:
        raise ValueError("rounding must be 'ceil' or 'legacy'")
    return max(floor, int(scaled))


def required_folds(eligible: int, ratio: float) -> int:
    """Return the minimum fold support for an eligible-fold count.

    The lower bound of two keeps a support ratio from admitting a single-fold
    result.  Values below zero are treated as zero eligible folds; configured
    ratios are validated separately by :func:`gpu_fuzzy_trader.config.validate_config`.
    """
    eligible_count = max(0, int(eligible))
    ratio_value = float(ratio)
    if not isfinite(ratio_value):
        raise ValueError("fold support ratio must be finite")
    return max(2, ceil(eligible_count * ratio_value))


def _datetime_series(df: pd.DataFrame, datetime_col: str | None = None) -> pd.Series | None:
    """Find a usable datetime series without requiring a fixed frame schema."""
    candidates = (
        (datetime_col,) if datetime_col is not None else ("datetime", "timestamp", "date")
    )
    for column in candidates:
        if column and column in df.columns:
            values = pd.to_datetime(df[column], errors="coerce", utc=True)
            if values.notna().any():
                return values
    if isinstance(df.index, pd.DatetimeIndex):
        values = pd.to_datetime(df.index, errors="coerce", utc=True)
        if values.notna().any():
            return pd.Series(values, index=df.index)
    return None


def _duration_bars(df: pd.DataFrame, datetime_col: str | None = None) -> int:
    """Estimate elapsed bars from timestamps, with a row-count fallback."""
    if df.empty:
        return 0
    values = _datetime_series(df, datetime_col)
    if values is None:
        return int(len(df))

    valid = values.dropna().drop_duplicates().sort_values()
    if len(valid) <= 1:
        return 1
    deltas = valid.diff().dropna()
    positive = deltas[deltas > pd.Timedelta(0)]
    if positive.empty:
        return int(len(valid))
    bar_delta = positive.median()
    if bar_delta <= pd.Timedelta(0):
        return int(len(valid))
    span = valid.iloc[-1] - valid.iloc[0]
    return max(1, int(ceil(span / bar_delta)) + 1)


def _per_symbol_rows(df: pd.DataFrame) -> dict[str, int]:
    """Return deterministic per-symbol row counts when a symbol column exists."""
    symbol_column = next(
        (column for column in ("symbol", "asset", "ticker") if column in df.columns),
        None,
    )
    if symbol_column is None:
        return {}
    grouped = df.groupby(symbol_column, dropna=False, sort=True).size()
    return {str(symbol): int(rows) for symbol, rows in grouped.items()}


def _frame_exposure(df: pd.DataFrame, datetime_col: str | None = None) -> FoldExposure:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("scoring_df and reference_df must be pandas DataFrames")
    return FoldExposure(
        rows=int(len(df)),
        duration_bars=_duration_bars(df, datetime_col),
        per_symbol_rows=_per_symbol_rows(df),
    )


def build_fold_gate_context(
    scoring_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    *,
    datetime_col: str | None = None,
    stage: str | None = None,
) -> dict[str, Any]:
    """Build the exposure context used by :func:`resolve_fold_gates`.

    ``scoring_df`` is intentionally not inferred as train or OOF.  The caller
    must pass the train scoring frame for train gates and the OOF scoring frame
    for OOF gates.  ``stage`` is optional metadata that can be used when a
    caller stores train and OOF contexts together.
    """
    scoring_exposure = _frame_exposure(scoring_df, datetime_col)
    reference_exposure = _frame_exposure(reference_df, datetime_col)
    scoring_rows = _effective_rows(scoring_exposure)
    reference_rows = _effective_rows(reference_exposure)
    exposure_ratio = (
        float(scoring_rows / reference_rows) if reference_rows > 0 else 0.0
    )
    reference_duration = float(reference_exposure.duration_bars)
    duration_ratio = (
        float(scoring_exposure.duration_bars / reference_duration)
        if reference_duration > 0
        else 0.0
    )
    normalized_stage = None if stage is None else str(stage).strip().lower()
    if normalized_stage not in {None, "train", "oof"}:
        raise ValueError("stage must be 'train', 'oof', or None")

    return {
        "stage": normalized_stage,
        "exposure": scoring_exposure,
        "fold_exposure": scoring_exposure,
        "scoring_exposure": scoring_exposure,
        "reference_exposure": reference_exposure,
        "exposure_ratio": exposure_ratio,
        "duration_ratio": duration_ratio,
        "effective_rows": scoring_rows,
        "scoring_rows": scoring_rows,
        "reference_rows": reference_rows,
        "per_symbol_rows": dict(scoring_exposure.per_symbol_rows),
        "reference_per_symbol_rows": dict(reference_exposure.per_symbol_rows),
        "absolute_min": _default_absolute_min(),
        "count_gate_allowlist": tuple(sorted(COUNT_GATE_ALLOWLIST)),
    }


def _context_for_stage(
    gate_context: Mapping[str, Any],
    stage: str | None,
) -> Mapping[str, Any]:
    """Select a nested train/OOF context when one was supplied."""
    selected_stage = stage
    if selected_stage is None:
        selected_stage = gate_context.get("stage")
    if selected_stage is None:
        return gate_context
    normalized = str(selected_stage).strip().lower()
    if normalized not in {"train", "oof"}:
        raise ValueError("stage must be 'train' or 'oof'")
    for key in (normalized, f"{normalized}_context"):
        value = gate_context.get(key)
        if isinstance(value, Mapping):
            return value
    # A context built by build_fold_gate_context has stage metadata but does
    # not need nesting.  Use it directly when it already carries an exposure.
    if any(key in gate_context for key in ("exposure", "fold_exposure", "scoring_exposure")):
        return gate_context
    raise ValueError(f"gate context has no {normalized!r} exposure")


def _context_exposures(
    gate_context: Mapping[str, Any],
) -> tuple[Any, Any]:
    """Extract fold and reference exposures, accepting audit-friendly aliases."""
    fold_exposure = next(
        (
            gate_context[key]
            for key in ("fold_exposure", "exposure", "scoring_exposure")
            if key in gate_context
        ),
        None,
    )
    reference_exposure = gate_context.get("reference_exposure")

    if fold_exposure is None and "effective_rows" in gate_context:
        fold_exposure = FoldExposure(
            rows=int(gate_context["effective_rows"]),
            duration_bars=int(gate_context.get("duration_bars", 0)),
            per_symbol_rows=dict(gate_context.get("per_symbol_rows", {})),
        )
    if reference_exposure is None and "reference_rows" in gate_context:
        reference_exposure = FoldExposure(
            rows=int(gate_context["reference_rows"]),
            duration_bars=int(gate_context.get("reference_duration_bars", 0)),
            per_symbol_rows=dict(gate_context.get("reference_per_symbol_rows", {})),
        )
    if fold_exposure is None or reference_exposure is None:
        raise ValueError(
            "gate context must contain fold and reference exposure; "
            "call build_fold_gate_context for the scoring stage"
        )
    return fold_exposure, reference_exposure


def resolve_fold_gates(
    base_gates: Mapping[str, Any],
    gate_context: Mapping[str, Any],
    stage: str | None = None,
) -> dict[str, Any]:
    """Resolve count gates for one scoring stage, leaving quality gates fixed.

    Pass a context built from train data when resolving train gates and a
    context built from OOF data when resolving OOF gates.  If train and OOF
    contexts are stored under ``{"train": ..., "oof": ...}``, pass ``stage``
    to select the intended one.
    """
    if not isinstance(base_gates, Mapping):
        raise TypeError("base_gates must be a mapping")
    if not isinstance(gate_context, Mapping):
        raise TypeError("gate_context must be a mapping")

    context = _context_for_stage(gate_context, stage)
    fold_exposure, reference_exposure = _context_exposures(context)
    absolute_min = context.get("absolute_min", gate_context.get("absolute_min"))
    if absolute_min is None:
        absolute_min = _default_absolute_min()

    resolved: dict[str, Any] = {}
    for name, value in base_gates.items():
        if name in COUNT_GATE_ALLOWLIST:
            resolved[name] = scale_count_gate(
                int(value),
                fold_exposure,
                reference_exposure,
                int(absolute_min),
            )
        else:
            resolved[name] = value
    return resolved


__all__ = [
    "COUNT_GATE_ALLOWLIST",
    "QUALITY_GATE_ALLOWLIST",
    "FoldExposure",
    "build_fold_gate_context",
    "required_folds",
    "resolve_fold_gates",
    "scale_count_gate",
]
