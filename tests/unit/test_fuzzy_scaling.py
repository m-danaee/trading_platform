"""Tests for train-fitted ordinal fuzzy feature scaling."""

from __future__ import annotations

import pandas as pd

from gpu_fuzzy_trader.features.fuzzy_scaling import (
    apply_fuzzy_feature_scaling,
    causal_positive_scale,
    causal_signed_scale,
    fit_fuzzy_feature_scaling,
    validate_rule_feature_ranges,
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


def test_causal_scaling_does_not_use_later_observations() -> None:
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    changed_future = values.copy()
    changed_future.iloc[-1] = 500.0

    base_positive = causal_positive_scale(values, window=3)
    changed_positive = causal_positive_scale(changed_future, window=3)
    base_signed = causal_signed_scale(values - 3.0, window=3)
    changed_signed = causal_signed_scale(changed_future - 3.0, window=3)

    assert base_positive.iloc[:-1].tolist() == changed_positive.iloc[:-1].tolist()
    assert base_signed.iloc[:-1].tolist() == changed_signed.iloc[:-1].tolist()
    assert base_positive.between(0.0, 1.0).all()
    assert base_signed.between(-1.0, 1.0).all()


def test_rule_feature_range_contract_fails_closed() -> None:
    valid = pd.DataFrame({"lwc_signal": [-1.0, 0.0, 1.0]})
    validate_rule_feature_ranges(valid)

    invalid = pd.DataFrame({"lwc_kama_10": [100.0]})
    try:
        validate_rule_feature_ranges(invalid)
    except ValueError as exc:
        assert "lwc_kama_10" in str(exc)
    else:
        raise AssertionError("raw rule-facing values must be rejected")
