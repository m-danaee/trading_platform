"""
Tests for Phase 5 rule filtering (_remove_negative_pnl_rules).

Covers the 8 required test scenarios from the code review:
  1. Basic removal
  2. Empty trade_log
  3. All negative with minimum-rule safeguard
  4. All positive → no rewrite
  5. Single negative with safeguard → no rewrite
  6. Rule_Index 1-based alignment
  7. File-on-disk rewrite + evaluator_clean written
  8. WRITE_EVALUATOR_CLEAN=False → no evaluator_clean file
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.output.writer import Output_Writer
from gpu_fuzzy_trader.phases import phase5_oos
from gpu_fuzzy_trader.phases.phase5_oos import OOS_Evaluator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def three_rules() -> list[dict]:
    """Three simple rules with different tp/sl."""
    return [
        {"tp": 2.0, "sl": 1.0, "capital_pct": 10.0,
         "conditions": ["[feature_a] IS High"]},
        {"tp": 3.0, "sl": 1.5, "capital_pct": 8.0,
         "conditions": ["[feature_b] IS Low"]},
        {"tp": 1.5, "sl": 1.0, "capital_pct": 5.0,
         "conditions": ["[feature_c] IS Mid"]},
    ]


@pytest.fixture
def long_strategy(three_rules) -> dict:
    return {"direction": "long", "rules_set": list(three_rules)}


@pytest.fixture
def short_strategy() -> dict:
    return {
        "direction": "short",
        "rules_set": [
            {"tp": 2.0, "sl": 1.0, "capital_pct": 10.0,
             "conditions": ["[feature] IS High"]},
        ],
    }


# =======================================================
# Test 1: Basic removal
# =======================================================

def test_basic_removal(long_strategy):
    """Rules 1,2,3 with Net_PnL +10, -20, +5 → keep rules 1 and 3."""
    trade_log = pd.DataFrame({
        "Rule_Index": [1, 1, 2, 2, 3],
        "Net_PnL": [10.0, -5.0, -20.0, -5.0, 5.0],
    })
    result, cleaned = OOS_Evaluator._remove_negative_pnl_rules(
        long_strategy, trade_log, "long")
    assert cleaned is True
    assert len(result["rules_set"]) == 2
    kept_tps = [r["tp"] for r in result["rules_set"]]
    assert 2.0 in kept_tps, "Rule 1 (positive PnL) should be kept"
    assert 1.5 in kept_tps, "Rule 3 (positive PnL) should be kept"
    assert 3.0 not in kept_tps, "Rule 2 (negative PnL) should be removed"


# =======================================================
# Test 2: Empty trade_log
# =======================================================

def test_empty_trade_log(long_strategy):
    """No trades at all → all rules kept, no rewrite."""
    trade_log = pd.DataFrame()
    result, cleaned = OOS_Evaluator._remove_negative_pnl_rules(
        long_strategy, trade_log, "long")
    assert cleaned is False
    assert len(result["rules_set"]) == 3


# =======================================================
# Test 3: All negative + PHASE3_GLOBAL_MIN_RULES=2
# =======================================================

def test_all_negative_with_min_safeguard(long_strategy):
    """All rules negative, min=2 → strategy unchanged, returns False."""
    trade_log = pd.DataFrame({
        "Rule_Index": [1, 2, 3],
        "Net_PnL": [-10.0, -5.0, -3.0],
    })
    orig_min = _cfg.PHASE3_GLOBAL_MIN_RULES
    try:
        _cfg.PHASE3_GLOBAL_MIN_RULES = 2
        result, cleaned = OOS_Evaluator._remove_negative_pnl_rules(
            long_strategy, trade_log, "long")
        assert cleaned is False
        assert len(result["rules_set"]) == 3
    finally:
        _cfg.PHASE3_GLOBAL_MIN_RULES = orig_min


# =======================================================
# Test 4: All positive → no rewrite
# =======================================================

def test_all_positive_no_removal(long_strategy):
    """All rules have positive PnL → nothing to remove."""
    trade_log = pd.DataFrame({
        "Rule_Index": [1, 2, 3],
        "Net_PnL": [10.0, 5.0, 3.0],
    })
    result, cleaned = OOS_Evaluator._remove_negative_pnl_rules(
        long_strategy, trade_log, "long")
    assert cleaned is False
    assert len(result["rules_set"]) == 3


# =======================================================
# Test 5: Single negative rule with min=2 → safeguarded
# =======================================================

def test_single_negative_with_min_safeguard():
    """One negative rule out of 2, min=2 → can't drop below min."""
    strategy = {
        "direction": "short",
        "rules_set": [
            {"tp": 2.0, "sl": 1.0, "capital_pct": 10.0,
             "conditions": ["[feature] IS High"]},
            {"tp": 3.0, "sl": 1.5, "capital_pct": 8.0,
             "conditions": ["[feature] IS Low"]},
        ],
    }
    trade_log = pd.DataFrame({
        "Rule_Index": [1, 2],
        "Net_PnL": [-5.0, 10.0],
    })
    orig_min = _cfg.PHASE3_GLOBAL_MIN_RULES
    try:
        _cfg.PHASE3_GLOBAL_MIN_RULES = 2
        result, cleaned = OOS_Evaluator._remove_negative_pnl_rules(
            strategy, trade_log, "short")
        assert cleaned is False
        assert len(result["rules_set"]) == 2
    finally:
        _cfg.PHASE3_GLOBAL_MIN_RULES = orig_min


# =======================================================
# Test 6: Rule_Index 1-based alignment
# =======================================================

def test_rule_index_one_based_alignment(long_strategy):
    """Rule_Index is 1-based; verify correct PnL attribution."""
    trade_log = pd.DataFrame({
        "Rule_Index": [1, 2, 3],
        "Net_PnL": [100.0, -1.0, 0.5],
    })
    orig_min = _cfg.PHASE3_GLOBAL_MIN_RULES
    try:
        _cfg.PHASE3_GLOBAL_MIN_RULES = 1
        result, cleaned = OOS_Evaluator._remove_negative_pnl_rules(
            long_strategy, trade_log, "long")
        assert cleaned is True
        # Rule 2 (index 2) has net -1.0 → removed
        assert len(result["rules_set"]) == 2
        kept_tps = [r["tp"] for r in result["rules_set"]]
        assert 2.0 in kept_tps  # Rule 1: +100
        assert 1.5 in kept_tps  # Rule 3: +0.5
        assert 3.0 not in kept_tps  # Rule 2: -1
    finally:
        _cfg.PHASE3_GLOBAL_MIN_RULES = orig_min


# =======================================================
# Test 7: File on disk rewritten + evaluator_clean written
# =======================================================

def test_file_rewrite_and_evaluator_clean(long_strategy):
    """Verify that disk file is rewritten and evaluator-clean is created."""
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = os.path.join(tmpdir, "outputs")
        os.makedirs(outputs_dir, exist_ok=True)

        # Override _STRATEGY_PATHS for long to point inside tmpdir
        orig_paths = dict(phase5_oos._STRATEGY_PATHS)
        strategy_path = os.path.join(outputs_dir, "long.json")
        phase5_oos._STRATEGY_PATHS["long"] = strategy_path

        # Write initial strategy to disk
        Output_Writer().write(long_strategy, strategy_path)

        trade_log = pd.DataFrame({
            "Rule_Index": [1, 2, 3],
            "Net_PnL": [10.0, -20.0, 5.0],
        })

        try:
            result, cleaned = OOS_Evaluator._remove_negative_pnl_rules(
                long_strategy, trade_log, "long"
            )
            assert cleaned is True
            assert len(result["rules_set"]) == 2

            # Main file should have been rewritten on disk
            with open(strategy_path) as f:
                disk_data = json.load(f)
            assert len(disk_data["rules_set"]) == 2

            # Evaluator-clean file should exist
            clean_path = os.path.join(
                outputs_dir, "evaluator_clean", "long_evaluator_clean.json"
            )
            assert os.path.exists(clean_path), \
                "Evaluator-clean file should exist after rewrite"
            with open(clean_path) as f:
                clean_data = json.load(f)
            assert clean_data["direction"] == "long"
            assert len(clean_data["rules_set"]) == 2
            # Extra keys stripped
            assert list(clean_data.keys()) == ["direction", "rules_set"]
        finally:
            phase5_oos._STRATEGY_PATHS.clear()
            phase5_oos._STRATEGY_PATHS.update(orig_paths)


# =======================================================
# Test 8: WRITE_EVALUATOR_CLEAN=False → no evaluator_clean
# =======================================================

def test_write_evaluator_clean_false_suppresses_file():
    """When WRITE_EVALUATOR_CLEAN=False, no evaluator_clean file is created."""
    orig_flag = _cfg.WRITE_EVALUATOR_CLEAN
    orig_min = _cfg.PHASE3_GLOBAL_MIN_RULES

    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = os.path.join(tmpdir, "outputs")
        os.makedirs(outputs_dir, exist_ok=True)

        orig_paths = dict(phase5_oos._STRATEGY_PATHS)
        strategy_path = os.path.join(outputs_dir, "long.json")
        phase5_oos._STRATEGY_PATHS["long"] = strategy_path

        strategy = {
            "direction": "long",
            "rules_set": [
                {"tp": 2.0, "sl": 1.0, "capital_pct": 10.0,
                 "conditions": ["[feature] IS High"]},
                {"tp": 3.0, "sl": 1.5, "capital_pct": 8.0,
                 "conditions": ["[feature] IS Low"]},
            ],
        }

        # Write initial strategy to disk manually (no evaluator-clean)
        os.makedirs(outputs_dir, exist_ok=True)
        with open(strategy_path, "w") as f:
            json.dump(strategy, f, indent=2)

        trade_log = pd.DataFrame({
            "Rule_Index": [1, 2],
            "Net_PnL": [5.0, -10.0],
        })

        try:
            _cfg.WRITE_EVALUATOR_CLEAN = False
            _cfg.PHASE3_GLOBAL_MIN_RULES = 1

            result, cleaned = OOS_Evaluator._remove_negative_pnl_rules(
                strategy, trade_log, "long"
            )
            assert cleaned is True
            assert len(result["rules_set"]) == 1

            # Main file should be rewritten
            with open(strategy_path) as f:
                disk_data = json.load(f)
            assert len(disk_data["rules_set"]) == 1

            # Evaluator-clean file should NOT exist
            clean_path = os.path.join(
                outputs_dir, "evaluator_clean", "long_evaluator_clean.json"
            )
            assert not os.path.exists(clean_path), \
                "Evaluator-clean file should NOT exist when flag is False"
        finally:
            _cfg.WRITE_EVALUATOR_CLEAN = orig_flag
            _cfg.PHASE3_GLOBAL_MIN_RULES = orig_min
            phase5_oos._STRATEGY_PATHS.clear()
            phase5_oos._STRATEGY_PATHS.update(orig_paths)
