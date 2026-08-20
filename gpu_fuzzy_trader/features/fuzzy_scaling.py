"""Train-fitted normalization for ordinal ``ff_*`` fuzzy feature codes.

The replacement data stores many fuzzy features as integer ordinal codes
(``1..4``, ``-2..2``, or ``-5..5``), while the evaluator contract expresses
their rule thresholds on ``[0, 1]`` or ``[-1, 1]``.  This module translates
only that documented representation.  Its parameters are fitted from the
training split and then reused unchanged for validation and held-out test.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


from gpu_fuzzy_trader import config as _cfg

_FORMAT_VERSION = 1


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
