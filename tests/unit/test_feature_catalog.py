"""Tests for the deterministic rule-feature catalog."""

from __future__ import annotations

import pandas as pd

from gpu_fuzzy_trader.features.catalog import (
    build_rule_feature_specs,
    rule_feature_specs_from_rules,
)


def test_catalog_keeps_only_safe_candidates_and_training_modes() -> None:
    """The catalog excludes non-rule columns without using labels to rank inputs."""
    frame = pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=4, freq="15min"),
        "symbol": ["BTCUSDT"] * 4,
        "label_open_next": [100.0, 101.0, 102.0, 103.0],
        "open": [100.0, 101.0, 102.0, 103.0],
        "high": [101.0, 102.0, 103.0, 104.0],
        "low": [99.0, 100.0, 101.0, 102.0],
        "close": [100.5, 101.5, 102.5, 103.5],
        "volume": [10.0, 20.0, 30.0, 40.0],
        "ff_donchian_width_20": [0.1, 0.2, 0.3, 0.4],
        "ff_ppo_sign": [1.0, -1.0, 1.0, -1.0],
        "binary_signal": [0.0, 1.0, 0.0, 1.0],
        "constant_signal": [1.0] * 4,
        "lwc_atr_14": [100.0, 101.0, 102.0, 103.0],
        "hwc_state": [1.0, 1.0, 1.0, 1.0],
        "_scratch": [1.0, 2.0, 3.0, 4.0],
    })

    assert build_rule_feature_specs(frame) == [
        {"name": "ff_donchian_width_20", "mode": "positive"},
        {"name": "binary_signal", "mode": "binary"},
    ]


def test_rule_feature_specs_come_from_frozen_rule_conditions() -> None:
    """Phase 5 reports only active, non-context rule inputs from train data."""
    train_df = pd.DataFrame({
        "binary_signal": [0.0, 1.0, 0.0, 1.0],
        "ff_donchian_width_20": [0.1, 0.2, 0.3, 0.4],
        "tf_permission_long": [1.0, 1.0, 1.0, 1.0],
    })
    rules = [
        {
            "conditions": [
                "[binary_signal] IS High",
                "[tf_permission_long] IS Allowed",
            ],
        },
        {
            "conditions": [
                "[ff_donchian_width_20] IS Low",
                "[symbol] IS BTCUSDT",
            ],
        },
    ]

    assert rule_feature_specs_from_rules(train_df, rules) == [
        {"name": "binary_signal", "mode": "binary"},
        {"name": "ff_donchian_width_20", "mode": "positive"},
    ]


def test_catalog_is_invariant_to_label_values() -> None:
    """Changing labels cannot change the catalog or its feature modes."""
    frame = pd.DataFrame({
        "label_open_next": [100.0, 101.0, 102.0, 103.0],
        "binary_signal": [0.0, 1.0, 0.0, 1.0],
        "ff_roc_8": [-0.4, 0.2, -0.1, 0.3],
    })
    changed_labels = frame.copy()
    changed_labels["label_open_next"] = [-999.0, 500.0, 0.0, 42.0]

    assert build_rule_feature_specs(frame) == build_rule_feature_specs(
        changed_labels,
    )
