"""Tests for _merge_per_symbol_strategies and related helpers."""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.run_pipeline import (
    _merge_per_symbol_strategies,
    _normalise_directions,
    Pipeline_Orchestrator,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_output_dir():
    """Yield a temporary directory bound to _cfg.OUTPUTS_DIR."""
    orig = _cfg.OUTPUTS_DIR
    with tempfile.TemporaryDirectory() as tmp:
        _cfg.OUTPUTS_DIR = tmp
        yield tmp
    _cfg.OUTPUTS_DIR = orig


@pytest.fixture
def directions():
    return _normalise_directions("both")


@pytest.fixture
def sample_strategy_a():
    """Two rules for symbol A (long direction)."""
    return {
        "direction": "long",
        "rules_set": [
            {"conditions": ["feat_0 IS High", "symbol is A"], "tp": 2.0, "sl": 1.2, "capital_pct": 12.5},
            {"conditions": ["feat_1 IS Low", "symbol is A"], "tp": 2.0, "sl": 1.2, "capital_pct": 12.5},
        ],
        "risk_optimized": True,
        "rb_score": 1.5,
    }


@pytest.fixture
def sample_strategy_b():
    """One rule for symbol B (long direction)."""
    return {
        "direction": "long",
        "rules_set": [
            {"conditions": ["feat_0 IS High", "symbol is B"], "tp": 2.0, "sl": 1.2, "capital_pct": 12.5},
        ],
        "risk_optimized": True,
        "rb_score": 2.0,
    }


# ---------------------------------------------------------------------------
# Capital distribution tests
# ---------------------------------------------------------------------------

class TestCapitalDistribution:
    """Verify capital_pct is computed correctly."""

    def test_two_symbols_four_rules(self, temp_output_dir, directions,
                                     sample_strategy_a, sample_strategy_b):
        """Two symbols → 3 rules → capital_pct = min(12.5, 35/3) ≈ 11.67."""
        per_symbol = {
            "A": [sample_strategy_a],  # 2 rules
            "B": [sample_strategy_b],  # 1 rule → total 3
        }
        merged = _merge_per_symbol_strategies(per_symbol, directions)
        long_rules = merged["long"]["rules_set"]
        assert len(long_rules) == 3
        expected_cap = min(12.5, 35.0 / 3)
        for r in long_rules:
            assert r["capital_pct"] == pytest.approx(expected_cap, abs=1e-9)
        total_cap = sum(r["capital_pct"] for r in long_rules)
        assert total_cap == pytest.approx(35.0, abs=1e-6)

    def test_one_symbol_two_rules(self, temp_output_dir, directions,
                                   sample_strategy_a):
        """Single symbol → 2 rules → capital_pct = min(12.5, 35/2) = 12.5."""
        per_symbol = {"A": [sample_strategy_a]}
        merged = _merge_per_symbol_strategies(per_symbol, directions)
        long_rules = merged["long"]["rules_set"]
        assert len(long_rules) == 2
        for r in long_rules:
            assert r["capital_pct"] == 12.5

    def test_one_symbol_one_rule(self, temp_output_dir, directions):
        """Single rule → capital_pct = 12.5 (no need to split)."""
        strategy = {
            "direction": "long",
            "rules_set": [
                {"conditions": ["x IS High"], "tp": 2.0, "sl": 1.2, "capital_pct": 12.5},
            ],
            "risk_optimized": True,
        }
        per_symbol = {"X": [strategy]}
        merged = _merge_per_symbol_strategies(per_symbol, directions)
        long_rules = merged["long"]["rules_set"]
        assert len(long_rules) == 1
        assert long_rules[0]["capital_pct"] == 12.5

    def test_many_rules_caps_total(self, temp_output_dir, directions):
        """10 rules → each gets 3.5 (35/10 = 3.5 < 12.5)."""
        rules = [
            {"conditions": [f"feat_{i} IS High"], "tp": 2.0, "sl": 1.2, "capital_pct": 12.5}
            for i in range(10)
        ]
        strategy = {"direction": "long", "rules_set": rules}
        per_symbol = {"S": [strategy]}
        merged = _merge_per_symbol_strategies(per_symbol, directions)
        long_rules = merged["long"]["rules_set"]
        assert len(long_rules) == 10
        for r in long_rules:
            assert r["capital_pct"] == 3.5
        total = sum(r["capital_pct"] for r in long_rules)
        assert total == pytest.approx(35.0, abs=1e-6)

    def test_capital_never_exceeds_max_per_rule(self, temp_output_dir, directions):
        """Even with 1 rule, capital_pct ≤ RB_DEFAULT_CAPITAL_PCT."""
        strategy = {
            "direction": "long",
            "rules_set": [{"conditions": ["x IS High"], "tp": 2.0, "sl": 1.2, "capital_pct": 99.0}],
        }
        per_symbol = {"X": [strategy]}
        merged = _merge_per_symbol_strategies(per_symbol, directions)
        long_rules = merged["long"]["rules_set"]
        assert long_rules[0]["capital_pct"] <= 12.5

    def test_capital_formula(self, temp_output_dir, directions):
        """Explicitly verify the formula: min(max_per_rule, max_total / n_rules)."""
        rules = [
            {"conditions": [f"feat_{i}"], "tp": 2.0, "sl": 1.2, "capital_pct": 12.5}
            for i in range(5)
        ]
        strategy = {"direction": "long", "rules_set": rules}
        per_symbol = {"S": [strategy]}
        merged = _merge_per_symbol_strategies(per_symbol, directions)
        n = 5
        expected = min(12.5, 35.0 / n)
        assert merged["long"]["rules_set"][0]["capital_pct"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Empty / edge-case tests
# ---------------------------------------------------------------------------

class TestEmptyStrategies:
    """Verify behaviour with empty or missing strategies."""

    def test_empty_dict(self, temp_output_dir, directions):
        """Empty per_symbol_strategies → empty direction dicts."""
        merged = _merge_per_symbol_strategies({}, directions)
        assert "long" in merged
        assert "short" in merged
        assert merged["long"]["rules_set"] == []
        assert merged["short"]["rules_set"] == []

    def test_symbol_with_no_direction_match(self, temp_output_dir, directions):
        """Symbol has only 'short' strategy, 'long' direction should be empty."""
        strategy = {"direction": "short", "rules_set": [{"conditions": ["x"], "tp": 2.0, "sl": 1.2}]}
        per_symbol = {"S": [strategy]}
        merged = _merge_per_symbol_strategies(per_symbol, directions)
        assert merged["long"]["rules_set"] == []
        assert len(merged["short"]["rules_set"]) == 1

    def test_empty_rules_set(self, temp_output_dir, directions):
        """Strategy with empty rules_set contributes nothing."""
        strategy = {"direction": "long", "rules_set": []}
        per_symbol = {"S": [strategy]}
        merged = _merge_per_symbol_strategies(per_symbol, directions)
        assert merged["long"]["rules_set"] == []

    def test_output_file_written(self, temp_output_dir, directions):
        """Check that outputs/{direction}.json files are created."""
        strategy = {"direction": "long", "rules_set": [{"conditions": ["x"], "tp": 2.0, "sl": 1.2}]}
        per_symbol = {"S": [strategy]}
        _merge_per_symbol_strategies(per_symbol, directions)
        long_path = os.path.join(_cfg.OUTPUTS_DIR, "long.json")
        short_path = os.path.join(_cfg.OUTPUTS_DIR, "short.json")
        assert os.path.exists(long_path)
        assert os.path.exists(short_path)
        with open(long_path) as f:
            data = json.load(f)
        assert data["direction"] == "long"
        assert len(data["rules_set"]) == 1


# ---------------------------------------------------------------------------
# Metadata tests
# ---------------------------------------------------------------------------

class TestMetadata:
    """Verify metadata first-wins behaviour."""

    def test_metadata_from_first_symbol(self, temp_output_dir, directions):
        """Metadata (rb_score, risk_optimized) should come from first symbol."""
        strategy_a = {"direction": "long", "rules_set": [{"conditions": ["a"]}], "rb_score": 1.0, "risk_optimized": True}
        strategy_b = {"direction": "long", "rules_set": [{"conditions": ["b"]}], "rb_score": 9.0, "risk_optimized": False}
        per_symbol = {"A": [strategy_a], "B": [strategy_b]}
        merged = _merge_per_symbol_strategies(per_symbol, directions)
        assert merged["long"]["rb_score"] == 1.0   # first-wins
        assert merged["long"]["risk_optimized"] is True


# ---------------------------------------------------------------------------
# Integration / structural tests
# ---------------------------------------------------------------------------

class TestIntegration:
    """Verify that Pipeline_Orchestrator exposes the method and merge is wired."""

    def test_orchestrator_has_method(self):
        """Pipeline_Orchestrator exposes _run_per_symbol_rb_governor."""
        assert hasattr(Pipeline_Orchestrator, "_run_per_symbol_rb_governor")
        method = getattr(Pipeline_Orchestrator, "_run_per_symbol_rb_governor")
        assert callable(method)

    def test_merge_function_reference(self):
        """Verify _merge_per_symbol_strategies is a callable function."""
        assert callable(_merge_per_symbol_strategies)
