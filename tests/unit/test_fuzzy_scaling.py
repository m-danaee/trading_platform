"""Tests for train-fitted ordinal fuzzy feature scaling."""

from __future__ import annotations

import pandas as pd

from gpu_fuzzy_trader.features.fuzzy_scaling import (
    apply_fuzzy_feature_scaling,
    fit_fuzzy_feature_scaling,
)


def test_ordinal_ff_codes_match_evaluator_threshold_scale() -> None:
    train = pd.DataFrame(
        {
            "ff_positive": [1, 2, 3, 4],
            "ff_signed_two": [-2, -1, 1, 2],
            "ff_signed_five": [-5, -1, 1, 5],
            "close": [100.0, 101.0, 102.0, 103.0],
        }
    )
    validation = train.copy()
    scaling = fit_fuzzy_feature_scaling(train)

    assert scaling["features"] == {
        "ff_positive": {"kind": "positive_ordinal", "scale": 4.0},
        "ff_signed_two": {"kind": "signed_ordinal", "scale": 2.0},
        "ff_signed_five": {"kind": "signed_ordinal", "scale": 5.0},
    }

    apply_fuzzy_feature_scaling(validation, scaling)

    assert validation["ff_positive"].tolist() == [0.25, 0.5, 0.75, 1.0]
    assert validation["ff_signed_two"].tolist() == [-1.0, -0.5, 0.5, 1.0]
    assert validation["ff_signed_five"].tolist() == [-1.0, -0.2, 0.2, 1.0]
    assert validation["close"].tolist() == train["close"].tolist()
