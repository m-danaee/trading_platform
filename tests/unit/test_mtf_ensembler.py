"""Unit tests for Decoupled Direction & Strength Ensembling and Rule Archives."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from gpu_fuzzy_trader.mtf.ensembler import (
    compute_rule_weights,
    compute_ensemble_direction_and_strength,
    deduplicate_rules,
)
from gpu_fuzzy_trader.mtf.archives import (
    compute_archive_hash,
    compute_rule_hash,
    get_default_archive_path,
    load_mtf_archive_payload,
    load_mtf_rule_archive,
    save_mtf_rule_archive,
    validate_archive_schema,
    validate_rule_schema,
)


def test_decoupled_direction_and_strength():
    # 2 Long rules, 1 Short rule
    rules = [
        {"direction": "long", "directional_edge": 0.15, "stability": 0.8},  # w = 0.12
        {"direction": "long", "directional_edge": 0.10, "stability": 0.6},  # w = 0.06
        {"direction": "short", "directional_edge": 0.20, "stability": 0.9},  # w = 0.18
    ]
    weights = compute_rule_weights(rules)
    assert (weights >= 0).all()
    assert np.isclose(weights[0], 0.12)
    assert np.isclose(weights[1], 0.06)
    assert np.isclose(weights[2], 0.18)

    # 3 timestamps:
    # t0: only short rule active -> Direction = -1.0, Strength = 0.18 / 0.36 = 0.50
    # t1: 1 long and 1 short active -> Direction = (0.12 - 0.18)/(0.12 + 0.18) = -0.20, Strength = 0.30/0.36 = 0.8333333333333334
    # t2: no rules active -> Direction = 0.0, Strength = 0.0
    active_matrix = np.array([
        [False, False, True],
        [True, False, True],
        [False, False, False],
    ])
    directions = ["long", "long", "short"]
    direction_score, strength_score = compute_ensemble_direction_and_strength(
        active_matrix, directions, weights
    )
    assert np.isclose(direction_score[0], -1.0)
    assert np.isclose(strength_score[0], 0.5)
    assert np.isclose(direction_score[1], -0.2)
    assert np.isclose(strength_score[1], 0.30 / 0.36)
    assert np.isclose(direction_score[2], 0.0)
    assert np.isclose(strength_score[2], 0.0)


def test_rule_weight_admission_and_edge_cases():
    rules = [
        # Valid positive rule
        {"direction": "long", "directional_edge": 0.10, "stability": 0.5, "mcc": 0.2},  # w = 0.05
        # Negative directional edge -> w = 0
        {"direction": "long", "directional_edge": -0.05, "stability": 0.9, "mcc": 0.1},  # w = 0.0
        # Negative stability -> w = 0
        {"direction": "short", "directional_edge": 0.15, "stability": -0.2, "mcc": 0.3},  # w = 0.0
        # Non-positive MCC -> w = 0
        {"direction": "short", "directional_edge": 0.20, "stability": 0.8, "mcc": 0.0},  # w = 0.0
        {"direction": "short", "directional_edge": 0.20, "stability": 0.8, "mcc": -0.1},  # w = 0.0
        # Non-positive skill -> w = 0
        {"direction": "long", "directional_edge": 0.12, "stability": 0.7, "mcc": 0.2, "skill": 0.0},  # w = 0.0
        # Fallback aliases
        {"direction": "long", "edge": 0.08, "stability_score": 0.5, "oof_mcc": 0.15},  # w = 0.04
    ]

    weights = compute_rule_weights(rules)
    assert len(weights) == len(rules)
    assert np.isclose(weights[0], 0.05)
    assert np.isclose(weights[1], 0.0)
    assert np.isclose(weights[2], 0.0)
    assert np.isclose(weights[3], 0.0)
    assert np.isclose(weights[4], 0.0)
    assert np.isclose(weights[5], 0.0)
    assert np.isclose(weights[6], 0.04)

    # Empty rules list
    empty_weights = compute_rule_weights([])
    assert isinstance(empty_weights, np.ndarray)
    assert len(empty_weights) == 0


def test_ensemble_scoring_edge_cases():
    # Empty active matrix
    dir_s, str_s = compute_ensemble_direction_and_strength(
        active_matrix=np.empty((0, 2), dtype=bool),
        directions=["long", "short"],
        weights=np.array([0.1, 0.2]),
    )
    assert len(dir_s) == 0
    assert len(str_s) == 0

    # Zero total weights (all rules zero weight)
    active_matrix = np.array([[True, True], [False, True]])
    dir_s, str_s = compute_ensemble_direction_and_strength(
        active_matrix=active_matrix,
        directions=["long", "short"],
        weights=np.array([0.0, 0.0]),
    )
    assert (dir_s == 0.0).all()
    assert (str_s == 0.0).all()

    # Direction aliases (BUY / SELL, uppercase)
    active_matrix = np.array([[True, False], [False, True]])
    dir_s, str_s = compute_ensemble_direction_and_strength(
        active_matrix=active_matrix,
        directions=["BUY", "SELL"],
        weights=np.array([0.5, 0.5]),
    )
    assert np.isclose(dir_s[0], 1.0)
    assert np.isclose(str_s[0], 0.5)
    assert np.isclose(dir_s[1], -1.0)
    assert np.isclose(str_s[1], 0.5)


def test_deduplicate_rules():
    rules = [
        {
            "timeframe": "hwc",
            "direction": "long",
            "conditions": ["rsi_14 < 30", "close > ema_50"],
            "directional_edge": 0.15,
            "mcc": 0.20,
        },
        # Duplicate with reversed condition order but lower edge
        {
            "timeframe": "hwc",
            "direction": "long",
            "conditions": ["close > ema_50", "rsi_14 < 30"],
            "directional_edge": 0.10,
            "mcc": 0.15,
        },
        # Distinct rule (short direction)
        {
            "timeframe": "hwc",
            "direction": "short",
            "conditions": ["rsi_14 < 30", "close > ema_50"],
            "directional_edge": 0.12,
            "mcc": 0.18,
        },
        # Distinct rule (different condition)
        {
            "timeframe": "hwc",
            "direction": "long",
            "conditions": ["rsi_14 < 25", "close > ema_50"],
            "directional_edge": 0.18,
            "mcc": 0.22,
        },
    ]

    deduped = deduplicate_rules(rules)
    assert len(deduped) == 3
    # Verify that the best duplicate was kept (edge 0.15 instead of 0.10)
    long_dup = [r for r in deduped if r["direction"] == "long" and len(r["conditions"]) == 2 and "rsi_14 < 30" in r["conditions"]]
    assert len(long_dup) == 1
    assert np.isclose(long_dup[0]["directional_edge"], 0.15)


def test_rule_hash_and_archive_hash():
    rule1 = {
        "timeframe": "hwc",
        "direction": "long",
        "conditions": ["rsi_14 < 30", "close > ema_50"],
    }
    rule2 = {
        "timeframe": "hwc",
        "direction": "long",
        "conditions": ["close > ema_50", "rsi_14 < 30"],
    }
    rule3 = {
        "timeframe": "hwc",
        "direction": "short",
        "conditions": ["rsi_14 < 30", "close > ema_50"],
    }

    hash1 = compute_rule_hash(rule1)
    hash2 = compute_rule_hash(rule2)
    hash3 = compute_rule_hash(rule3)

    assert isinstance(hash1, str) and len(hash1) == 64
    # Condition order invariance
    assert hash1 == hash2
    # Direction sensitivity
    assert hash1 != hash3

    arch_hash1 = compute_archive_hash([rule1, rule3])
    arch_hash2 = compute_archive_hash([rule2, rule3])
    assert arch_hash1 == arch_hash2


def test_save_and_load_mtf_rule_archive(tmp_path: Path):
    rules = [
        {
            "timeframe": "hwc",
            "direction": "long",
            "conditions": ["rsi_14 < 30", "vol_ratio > 1.2"],
            "complexity": 2,
            "coverage": 0.35,
            "directional_edge": 0.15,
            "stability": 0.85,
            "oof_metrics": {
                "directional_edge": 0.15,
                "mcc": 0.22,
                "stability": 0.85,
                "coverage": 0.35,
            },
            "data_hash": "d1a2b3c4",
        },
        {
            "timeframe": "hwc",
            "direction": "short",
            "conditions": ["rsi_14 > 70"],
            "complexity": 1,
            "coverage": 0.28,
            "directional_edge": 0.18,
            "stability": 0.90,
            "oof_metrics": {
                "directional_edge": 0.18,
                "mcc": 0.25,
                "stability": 0.90,
                "coverage": 0.28,
            },
            "data_hash": "d1a2b3c4",
        },
    ]

    archive_path = tmp_path / "rule_archives" / "hwc" / "hwc_rules.json"
    archive_hash = save_mtf_rule_archive(
        timeframe="hwc",
        rules=rules,
        path=archive_path,
        metadata={"data_hash": "d1a2b3c4", "n_folds": 4},
    )

    assert archive_path.exists()
    assert isinstance(archive_hash, str) and len(archive_hash) == 64

    # Load rules list
    loaded_rules = load_mtf_rule_archive(archive_path)
    assert len(loaded_rules) == 2
    assert loaded_rules[0]["direction"] == "long"
    assert loaded_rules[1]["direction"] == "short"
    assert "rule_hash" in loaded_rules[0]

    # Load full archive payload
    payload = load_mtf_archive_payload(archive_path)
    assert payload["schema_version"] == "2.0.0"
    assert payload["timeframe"] == "hwc"
    assert payload["archive_hash"] == archive_hash
    assert payload["rule_count"] == 2
    assert payload["metadata"]["data_hash"] == "d1a2b3c4"


def test_rule_archive_validation_failures(tmp_path: Path):
    # Missing required field 'conditions'
    invalid_rule = {
        "timeframe": "hwc",
        "direction": "long",
        "coverage": 0.3,
    }
    with pytest.raises(ValueError, match="Missing required rule field"):
        validate_rule_schema(invalid_rule, raise_error=True)

    assert validate_rule_schema(invalid_rule, raise_error=False) is False

    # Invalid timeframe
    invalid_tf_rule = {
        "timeframe": "invalid_tf",
        "direction": "long",
        "conditions": ["rsi < 30"],
        "coverage": 0.3,
        "oof_metrics": {"directional_edge": 0.1},
        "complexity": 1,
        "data_hash": "abc",
    }
    with pytest.raises(ValueError, match="Invalid timeframe"):
        validate_rule_schema(invalid_tf_rule, raise_error=True)

    # Corrupt payload validation
    corrupt_payload = {
        "schema_version": "1.0.0",  # wrong version
        "timeframe": "hwc",
        "rules": [],
    }
    with pytest.raises(ValueError, match="schema_version"):
        validate_archive_schema(corrupt_payload, raise_error=True)

    # Attempt to load non-existent file
    with pytest.raises(FileNotFoundError):
        load_mtf_rule_archive(tmp_path / "non_existent.json")


def test_get_default_archive_path():
    path_hwc = get_default_archive_path("hwc")
    assert "rule_archives/hwc" in str(path_hwc)
    assert path_hwc.suffix == ".json"

    path_mwc = get_default_archive_path("60m")
    assert "rule_archives/mwc" in str(path_mwc)

    path_lwc = get_default_archive_path("15m")
    assert "rule_archives/lwc" in str(path_lwc)
