"""Scaling contracts for ordinal and rule-facing fuzzy features.

The replacement data stores many fuzzy features as integer ordinal codes
(``1..4``, ``-2..2``, or ``-5..5``), while the evaluator contract expresses
their rule thresholds on ``[0, 1]`` or ``[-1, 1]``.  This module translates
that documented representation.

MTF technical features use :func:`causal_positive_scale` and
:func:`causal_signed_scale`.  These transforms use only the current value and
a trailing, past-only magnitude reference.  They do not fit a distribution on
the complete tape, so an OOF row cannot inherit scale information from its
future fold.  The resulting values match the evaluator's fixed fuzzy bins.

The ordinal ``ff_*`` path remains train-fitted and is reused unchanged for
validation and held-out test.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


from gpu_fuzzy_trader import config as _cfg

_FORMAT_VERSION = 2
_DEFAULT_CAUSAL_SCALE_WINDOW = 128


def _numeric_series(values: pd.Series | np.ndarray) -> pd.Series:
    """Return finite-compatible floating-point values with the source index."""
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce").astype(float)
    return pd.Series(pd.to_numeric(values, errors="coerce"), dtype=float)


def _causal_magnitude_reference(
    values: pd.Series | np.ndarray,
    *,
    window: int = _DEFAULT_CAUSAL_SCALE_WINDOW,
) -> tuple[pd.Series, pd.Series]:
    """Return numeric values and a trailing median absolute magnitude.

    ``rolling`` includes the current observation and never includes a later
    observation.  ``min_periods=1`` avoids inventing a global warm-up scale.
    Missing current values remain missing in the returned transformed value.
    """
    window = int(window)
    if window < 1:
        raise ValueError("causal scaling window must be positive")
    numeric = _numeric_series(values)
    reference = numeric.abs().rolling(window=window, min_periods=1).median()
    return numeric, reference


def causal_positive_scale(
    values: pd.Series | np.ndarray,
    *,
    window: int = _DEFAULT_CAUSAL_SCALE_WINDOW,
) -> pd.Series:
    """Map a non-negative feature to ``[0, 1]`` with a past-only reference.

    For a current value ``x`` and trailing magnitude ``m`` the result is
    ``x / (x + m)``.  A value near its recent median is therefore near 0.5,
    while unusually large values approach 1.  This is monotone and does not
    use future or cross-symbol observations.
    """
    numeric, reference = _causal_magnitude_reference(values, window=window)
    current = numeric.to_numpy(dtype=float, copy=False)
    scale = reference.to_numpy(dtype=float, copy=False)
    result = np.full(len(numeric), np.nan, dtype=float)
    valid = np.isfinite(current) & np.isfinite(scale)
    denominator = current[valid] + scale[valid]
    nonzero = denominator > 0.0
    result_indices = np.flatnonzero(valid)
    result[result_indices[nonzero]] = (
        current[result_indices[nonzero]] / denominator[nonzero]
    )
    # Both current and reference are non-negative by contract.  Clipping also
    # makes this helper fail-safe for a malformed caller.
    return pd.Series(np.clip(result, 0.0, 1.0), index=numeric.index)


def causal_signed_scale(
    values: pd.Series | np.ndarray,
    *,
    window: int = _DEFAULT_CAUSAL_SCALE_WINDOW,
) -> pd.Series:
    """Map a signed feature to ``[-1, 1]`` with a past-only magnitude scale."""
    numeric, reference = _causal_magnitude_reference(values, window=window)
    current = numeric.to_numpy(dtype=float, copy=False)
    scale = reference.to_numpy(dtype=float, copy=False)
    result = np.full(len(numeric), np.nan, dtype=float)
    valid = np.isfinite(current) & np.isfinite(scale)
    denominator = np.abs(current[valid]) + scale[valid]
    nonzero = denominator > 0.0
    result_indices = np.flatnonzero(valid)
    result[result_indices[nonzero]] = (
        current[result_indices[nonzero]] / denominator[nonzero]
    )
    return pd.Series(np.clip(result, -1.0, 1.0), index=numeric.index)


def validate_rule_feature_ranges(
    df: pd.DataFrame,
    *,
    prefixes: tuple[str, ...] = ("lwc_", "mwc_", "hwc_"),
    columns: tuple[str, ...] | None = None,
) -> None:
    """Fail closed if rule-facing technical features leave ``[-1, 1]``.

    NaN is allowed during indicator warm-up.  Raw OHLCV and labels are not
    inspected because they are not rule-facing fuzzy features.
    """
    selected = set(columns) if columns is not None else None
    for name in df.columns:
        if selected is not None:
            if name not in selected:
                continue
        elif not any(str(name).startswith(prefix) for prefix in prefixes):
            continue
        values = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size and (np.min(finite) < -1.0 - 1e-9 or np.max(finite) > 1.0 + 1e-9):
            raise ValueError(
                f"Rule-facing feature {name!r} is outside [-1, 1]; "
                "the fuzzy representation contract is invalid."
            )


def fit_fuzzy_feature_scaling(train_df: pd.DataFrame) -> dict[str, Any]:
    """Build a train-only scaling contract for ordinal ``ff_*`` columns.

    If config.FEATURE_SCALE_MANIFEST is provided, it is used directly.
    Otherwise the scaling is inferred from the training split.
    """
    manifest_override = getattr(_cfg, "FEATURE_SCALE_MANIFEST", None)
    if isinstance(manifest_override, dict) and "features" in manifest_override:
        return manifest_override

    features: dict[str, dict[str, float | str]] = {}
    for name in train_df.columns:
        if not name.startswith("ff_") or not pd.api.types.is_numeric_dtype(
            train_df[name]
        ):
            continue
        values = train_df[name].to_numpy(dtype=np.float64, copy=False)
        finite = values[np.isfinite(values)]
        if finite.size == 0 or not np.allclose(finite, np.rint(finite)):
            continue
        minimum = float(np.min(finite))
        maximum = float(np.max(finite))
        max_abs = max(abs(minimum), abs(maximum))
        if minimum >= 0.0 and maximum <= 4.0 and maximum > 1.0:
            # Positive fuzzy codes have the documented five levels 0..4.
            features[name] = {"kind": "positive_ordinal", "scale": 4.0}
        elif minimum < 0.0 and max_abs <= 5.0 and max_abs > 1.0:
            # Signed code families are -2..2 or -5..5.  The observed train
            # range identifies the family without looking at validation/test.
            features[name] = {
                "kind": "signed_ordinal",
                "scale": 2.0 if max_abs <= 2.0 else 5.0,
            }
    return {"version": _FORMAT_VERSION, "features": features}


def apply_fuzzy_feature_scaling(
    df: pd.DataFrame,
    scaling: dict[str, Any],
) -> pd.DataFrame:
    """Apply an existing train-fitted scaling contract in place and return *df*."""
    feature_specs = scaling.get("features", {}) if isinstance(scaling, dict) else {}
    if not isinstance(feature_specs, dict):
        return df
    for name, spec in feature_specs.items():
        if name not in df.columns or not isinstance(spec, dict):
            continue
        try:
            scale = float(spec["scale"])
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(scale) or scale <= 0.0:
            continue
        values = pd.to_numeric(df[name], errors="coerce") / scale
        # Column assignment intentionally permits integer source columns to
        # become floats; ``.loc`` rejects that upcast under pandas 3.
        df[name] = values.clip(lower=-1.0, upper=1.0)
    return df
