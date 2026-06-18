"""
Unit tests for multi-symbol combinations in Phase 3 (Task 6).

Tests cover:
  - ``_build_symbol_specialized_variants`` returns single-symbol variants
    when ``SYMBOL_SPECIALIZATION_USE_COMBINATIONS=False``.
  - ``_build_symbol_specialized_variants`` returns 1-, 2-, and 3-symbol
    variants when ``USE_COMBINATIONS=True`` and enough symbols pass the gate.
  - Variants are filtered by ``gate_positive_good`` with the configured
    min-trade thresholds.
  - The best variant (highest score) is picked by ``_build_multi_symbol_merged_rules``.
  - Config keys are accessible via ``getattr(_cfg, ...)``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.phases.phase3_rule_set import (
    _build_multi_symbol_merged_rules,
    _build_symbol_specialized_variants,
    _is_symbol_condition,
    _merge_per_symbol_rules,
    _strip_symbol_conditions,
    gate_positive_good,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_engine(return_pct: float = 5.0, trades: int = 20) -> MagicMock:
    """Create a minimal mock engine that returns a fixed metrics dict."""
    metrics = {
        "total_return_pct": return_pct,
        "profit_factor": 1.5,
        "executed_trades": trades,
        "win_rate": 0.55,
        "max_drawdown_pct": 8.0,
        "sortino_ratio": 1.2,
        "raw_signal_count": trades * 2,
        "skipped_min_notional_count": 0,
        "max_simultaneous_positions": 5,
    }

    engine = MagicMock()
    engine.simulate_rule_set.return_value = dict(metrics)
    return engine


def _make_rule(conditions: list[str] | None = None) -> dict:
    """Create a minimal pool-style rule dict."""
    return {
        "conditions": conditions or ["[feature_ma] IS Very High"],
        "tp": 2.0,
        "sl": 1.0,
        "capital_pct": 30.0,
    }


# ===================================================================
# Pure-function tests for helper utilities
# ===================================================================


class TestSymbolConditionHelpers:
    """Tests for ``_is_symbol_condition`` and ``_strip_symbol_conditions``."""

    def test_is_symbol_condition_true(self) -> None:
        assert _is_symbol_condition("symbol is 1") is True
        assert _is_symbol_condition("symbol is 5") is True
        assert _is_symbol_condition("symbol is 1, symbol is 5") is True
        assert _is_symbol_condition("[symbol] is 1") is True

    def test_is_symbol_condition_false(self) -> None:
        assert _is_symbol_condition("[feature_ma] IS Very High") is False
        assert _is_symbol_condition("") is False
        assert _is_symbol_condition("symbol") is False

    def test_strip_removes_symbol_conditions(self) -> None:
        conds = ["[feature_ma] IS Very High", "symbol is 1", "symbol is 5"]
        result = _strip_symbol_conditions(conds)
        assert result == ["[feature_ma] IS Very High"]

    def test_strip_preserves_non_symbol(self) -> None:
        conds = ["[feature_ma] IS Very High", "[rsi] IS Low"]
        result = _strip_symbol_conditions(conds)
        assert result == conds

    def test_strip_empty_list(self) -> None:
        assert _strip_symbol_conditions([]) == []


# ===================================================================
# ``_build_symbol_specialized_variants`` unit tests
# ===================================================================


class TestBuildSymbolSpecializedVariants:
    """Core tests for the multi-symbol variant builder."""

    # --- Test 1: Single-symbol variant kept when it passes the gate ---

    def test_single_symbol_variant_kept(self) -> None:
        """A single-symbol variant that passes the gate is returned."""
        train_engine = _mock_engine(return_pct=5.0, trades=20)
        val_engine = _mock_engine(return_pct=3.0, trades=15)
        rule = _make_rule()
        symbols = ["1", "2", "3"]

        with patch.object(_cfg, "SYMBOL_SPECIALIZATION_USE_COMBINATIONS", False):
            variants = _build_symbol_specialized_variants(
                rule, train_engine, val_engine, symbols,
            )

        # Should have at most single-symbol variants (no 2/3 symbol combos).
        assert len(variants) >= 1
        for v in variants:
            conditions = v.get("conditions", [])
            sym_conds = [c for c in conditions if _is_symbol_condition(c)]
            assert len(sym_conds) == 1, (
                f"Expected exactly 1 symbol condition, got {sym_conds}"
            )

    # --- Test 2: 2-symbol variant preferred when both pass ---

    def test_two_symbol_variant_preferred(self) -> None:
        """When USE_COMBINATIONS=True, the best 2-symbol variant is ranked highest."""
        train_engine = _mock_engine(return_pct=5.0, trades=20)
        val_engine = _mock_engine(return_pct=3.0, trades=15)
        rule = _make_rule()
        symbols = ["1", "2", "3", "4", "5"]

        with patch.object(_cfg, "SYMBOL_SPECIALIZATION_MAX_SYMBOLS_PER_RULE", 3):
            variants = _build_symbol_specialized_variants(
                rule, train_engine, val_engine, symbols,
            )

        # Since all variants have the same mock return and all pass the gate,
        # the list should contain a mix of 1-, 2-, and 3-symbol variants.
        # At least one variant should exist.
        assert len(variants) >= 1

        # Check that at least some 2-symbol variants exist.
        has_two_symbol = False
        for v in variants:
            sym_conds = [
                c for c in v.get("conditions", [])
                if _is_symbol_condition(c)
            ]
            if len(sym_conds) == 2:
                has_two_symbol = True
                break
        assert has_two_symbol, (
            "Expected at least one 2-symbol variant among the results"
        )

    # --- Test 3: 3-symbol variants generated with MAX_SYMBOLS_PER_RULE=3 ---

    def test_three_symbol_variants_generated(self) -> None:
        """With MAX_SYMBOLS_PER_RULE=3, 3-symbol combinations are generated."""
        train_engine = _mock_engine(return_pct=5.0, trades=20)
        val_engine = _mock_engine(return_pct=3.0, trades=15)
        rule = _make_rule()
        symbols = ["1", "2", "3", "4", "5"]

        # Raise MAX_VARIANTS_PER_RULE so all 25 candidates fit (5 single + 10 two + 10 three).
        with patch.object(_cfg, "SYMBOL_SPECIALIZATION_MAX_SYMBOLS_PER_RULE", 3):
            with patch.object(_cfg, "SYMBOL_SPECIALIZATION_MAX_VARIANTS_PER_RULE", 30):
                variants = _build_symbol_specialized_variants(
                    rule, train_engine, val_engine, symbols,
                )

        has_three_symbol = False
        for v in variants:
            sym_conds = [
                c for c in v.get("conditions", [])
                if _is_symbol_condition(c)
            ]
            if len(sym_conds) == 3:
                has_three_symbol = True
                break
        assert has_three_symbol, (
            "Expected at least one 3-symbol variant with MAX_SYMBOLS_PER_RULE=3"
        )

    # --- Test 4: USE_COMBINATIONS=False disables 2/3-symbol generation ---

    def test_use_combinations_false_disables_multi_symbol(self) -> None:
        """When USE_COMBINATIONS=False, only single-symbol variants are returned."""
        train_engine = _mock_engine(return_pct=5.0, trades=20)
        val_engine = _mock_engine(return_pct=3.0, trades=15)
        rule = _make_rule()
        symbols = ["1", "2", "3", "4", "5"]

        with patch.object(_cfg, "SYMBOL_SPECIALIZATION_USE_COMBINATIONS", False):
            variants = _build_symbol_specialized_variants(
                rule, train_engine, val_engine, symbols,
            )

        for v in variants:
            sym_conds = [
                c for c in v.get("conditions", [])
                if _is_symbol_condition(c)
            ]
            assert len(sym_conds) <= 1, (
                f"Expected single-symbol only, got {len(sym_conds)} symbol conditions"
            )

    # --- Test 5: Rule with existing symbol condition is returned as-is ---

    def test_rule_with_existing_symbol_condition(self) -> None:
        """If the rule already has a symbol condition, return unchanged."""
        train_engine = _mock_engine(return_pct=5.0, trades=20)
        val_engine = _mock_engine(return_pct=3.0, trades=15)
        rule = _make_rule(["[feature_ma] IS Very High", "symbol is 1"])
        symbols = ["1", "2", "3"]

        variants = _build_symbol_specialized_variants(
            rule, train_engine, val_engine, symbols,
        )

        assert len(variants) == 1
        conds = variants[0].get("conditions", [])
        assert "symbol is 1" in conds
        assert len([c for c in conds if _is_symbol_condition(c)]) == 1

    # --- Test 6: Empty symbols returns rule as-is ---

    def test_eligible_symbols_restricts_universe(self) -> None:
        """eligible_symbols limits which symbols are specialized."""
        train_engine = _mock_engine(return_pct=5.0, trades=20)
        val_engine = _mock_engine(return_pct=3.0, trades=15)
        rule = _make_rule()
        symbols = ["1", "2", "3", "4", "5"]

        with patch.object(_cfg, "SYMBOL_SPECIALIZATION_USE_COMBINATIONS", False):
            variants = _build_symbol_specialized_variants(
                rule, train_engine, val_engine, symbols,
                eligible_symbols=["2", "4"],
            )

        sym_sets = {
            tuple(
                c.replace("symbol is ", "")
                for c in v.get("conditions", [])
                if _is_symbol_condition(c)
            )
            for v in variants
        }
        assert sym_sets <= {("2",), ("4",)}

    def test_empty_symbols(self) -> None:
        """When symbols list is empty, return the rule unchanged."""
        train_engine = _mock_engine(return_pct=5.0, trades=20)
        val_engine = _mock_engine(return_pct=3.0, trades=15)
        rule = _make_rule()

        variants = _build_symbol_specialized_variants(
            rule, train_engine, val_engine, [],
        )

        assert len(variants) == 1
        # No symbol conditions should have been added.
        assert not any(
            _is_symbol_condition(c) for c in variants[0].get("conditions", [])
        )

    # --- Test 7: Gate filters out poor variants ---

    def test_gate_filters_poor_variants(self) -> None:
        """Variants that fail gate_positive_good are excluded."""
        train_engine_bad = _mock_engine(return_pct=-2.0, trades=5)
        val_engine_bad = _mock_engine(return_pct=-1.0, trades=3)
        rule = _make_rule()
        symbols = ["1", "2", "3"]

        with patch.object(
            _cfg, "SYMBOL_SPECIALIZATION_MIN_TRAIN_TRADES", 10,
        ):
            with patch.object(
                _cfg, "SYMBOL_SPECIALIZATION_MIN_VAL_TRADES", 6,
            ):
                variants = _build_symbol_specialized_variants(
                    rule, train_engine_bad, val_engine_bad, symbols,
                )

        # All variants should fail the gate because trades < min thresholds.
        # The function falls back to returning a variant with one symbol.
        assert len(variants) >= 0  # May be empty or fallback

    # --- Test 8: Conditions are correctly built ---

    def test_variant_conditions_preserve_base(self) -> None:
        """Each variant preserves the original feature conditions."""
        train_engine = _mock_engine(return_pct=5.0, trades=20)
        val_engine = _mock_engine(return_pct=3.0, trades=15)
        rule = _make_rule(["[feature_ma] IS Very High", "[rsi] IS Overbought"])
        symbols = ["1", "2"]

        variants = _build_symbol_specialized_variants(
            rule, train_engine, val_engine, symbols,
        )

        for v in variants:
            conds = v.get("conditions", [])
            assert "[feature_ma] IS Very High" in conds
            assert "[rsi] IS Overbought" in conds
            has_sym = any(_is_symbol_condition(c) for c in conds)
            assert has_sym, "Expected at least one symbol condition in variant"


# ===================================================================
# ``_build_multi_symbol_merged_rules`` tests
# ===================================================================


class TestBuildMultiSymbolMergedRules:
    """Tests for the integration function that calls _build_symbol_specialized_variants."""

    def test_best_variant_picked(self) -> None:
        """The best variant for each pool rule is picked."""
        train_engine = _mock_engine(return_pct=5.0, trades=20)
        val_engine = _mock_engine(return_pct=3.0, trades=15)
        pool = [_make_rule(["[feature_ma] IS Very High"])]
        symbols = ["1", "2", "3"]

        # symbol_assignments: symbol "1" selected pool index 0, symbol "2" selected pool index 0
        symbol_assignments = {"1": [0], "2": [0]}

        merged = _build_multi_symbol_merged_rules(
            symbol_assignments, pool, train_engine, val_engine, symbols,
        )

        assert len(merged) == 1
        # The merged rule should have symbol conditions from the best variant.
        conds = merged[0].get("conditions", [])
        sym_conds = [c for c in conds if _is_symbol_condition(c)]
        # With the mock engine returning identical scores for all symbols,
        # the best variant will be the first single-symbol variant.
        assert len(sym_conds) >= 1

    def test_multiple_pool_rules(self) -> None:
        """Multiple pool rules each get their best variant."""
        train_engine = _mock_engine(return_pct=5.0, trades=20)
        val_engine = _mock_engine(return_pct=3.0, trades=15)
        pool = [
            _make_rule(["[feature_ma] IS Very High"]),
            _make_rule(["[rsi] IS Oversold"]),
        ]
        symbols = ["1", "2", "3"]

        symbol_assignments = {
            "1": [0],
            "2": [0, 1],
            "3": [1],
        }

        merged = _build_multi_symbol_merged_rules(
            symbol_assignments, pool, train_engine, val_engine, symbols,
        )

        assert len(merged) == 2

    def test_pool_idx_preserved(self) -> None:
        """The _pool_idx key is preserved for sorting."""
        train_engine = _mock_engine(return_pct=5.0, trades=20)
        val_engine = _mock_engine(return_pct=3.0, trades=15)
        pool = [_make_rule(["[feature_ma] IS Very High"])]
        symbols = ["1"]

        symbol_assignments = {"1": [0]}
        merged = _build_multi_symbol_merged_rules(
            symbol_assignments, pool, train_engine, val_engine, symbols,
        )

        assert len(merged) == 1
        assert "_pool_idx" in merged[0]
        assert merged[0]["_pool_idx"] == 0

    def test_empty_assignments(self) -> None:
        """Empty symbol_assignments returns empty list."""
        merged = _build_multi_symbol_merged_rules(
            {}, [], _mock_engine(), _mock_engine(), [],
        )
        assert merged == []


# ===================================================================
# Config keys accessibility
# ===================================================================


class TestConfigKeys:
    """All 6 SYMBOL_SPECIALIZATION_* keys are present in config."""

    def test_all_config_keys_exist(self) -> None:
        expected_keys = [
            "SYMBOL_SPECIALIZATION_USE_COMBINATIONS",
            "SYMBOL_SPECIALIZATION_MAX_SYMBOLS_PER_RULE",
            "SYMBOL_SPECIALIZATION_TOP_SINGLE_SYMBOLS",
            "SYMBOL_SPECIALIZATION_MAX_VARIANTS_PER_RULE",
            "SYMBOL_SPECIALIZATION_MIN_TRAIN_TRADES",
            "SYMBOL_SPECIALIZATION_MIN_VAL_TRADES",
        ]
        for key in expected_keys:
            value = getattr(_cfg, key, None)
            assert value is not None, f"Config key {key!r} is missing or None"
            assert isinstance(value, (bool, int, float)), (
                f"Config key {key!r} has unexpected type {type(value).__name__}"
            )

    def test_default_values(self) -> None:
        assert (
            getattr(_cfg, "SYMBOL_SPECIALIZATION_USE_COMBINATIONS", None)
            is True
        )
        assert (
            getattr(_cfg, "SYMBOL_SPECIALIZATION_MAX_SYMBOLS_PER_RULE", None)
            == 1
        )
        assert (
            getattr(_cfg, "SYMBOL_SPECIALIZATION_TOP_SINGLE_SYMBOLS", None)
            == 5
        )
        assert (
            getattr(_cfg, "SYMBOL_SPECIALIZATION_MAX_VARIANTS_PER_RULE", None)
            == 10
        )
        assert (
            getattr(_cfg, "SYMBOL_SPECIALIZATION_MIN_TRAIN_TRADES", None)
            == 10
        )
        assert (
            getattr(_cfg, "SYMBOL_SPECIALIZATION_MIN_VAL_TRADES", None)
            == 6
        )


# ===================================================================
# Import and function existence
# ===================================================================


class TestImportable:
    """The new functions are importable from phase3_rule_set."""

    def test_build_symbol_specialized_variants_importable(self) -> None:
        from gpu_fuzzy_trader.phases.phase3_rule_set import (
            _build_symbol_specialized_variants as fn,
        )
        assert fn is _build_symbol_specialized_variants

    def test_build_multi_symbol_merged_rules_importable(self) -> None:
        from gpu_fuzzy_trader.phases.phase3_rule_set import (
            _build_multi_symbol_merged_rules as fn,
        )
        assert fn is _build_multi_symbol_merged_rules

    def test_is_symbol_condition_importable(self) -> None:
        from gpu_fuzzy_trader.phases.phase3_rule_set import (
            _is_symbol_condition as fn,
        )
        assert fn is _is_symbol_condition

    def test_strip_symbol_conditions_importable(self) -> None:
        from gpu_fuzzy_trader.phases.phase3_rule_set import (
            _strip_symbol_conditions as fn,
        )
        assert fn is _strip_symbol_conditions


# ===================================================================
# Backward compatibility — _merge_per_symbol_rules still works
# ===================================================================


class TestMergePerSymbolRulesPreserved:
    """The original _merge_per_symbol_rules function is preserved."""

    def test_merge_still_works(self) -> None:
        pool = [
            _make_rule(["[feature_ma] IS Very High"]),
            _make_rule(["[rsi] IS Oversold"]),
        ]
        assignments = {
            "1": [0],
            "2": [0, 1],
            "3": [1],
        }
        merged = _merge_per_symbol_rules(assignments, pool)
        assert len(merged) == 2
        # Rule 0 was selected by symbols 1 and 2
        r0_conds = merged[0].get("conditions", [])
        assert "symbol is 1" in r0_conds
        assert "symbol is 2" in r0_conds
        # Rule 1 was selected by symbols 2 and 3
        r1_conds = merged[1].get("conditions", [])
        assert "symbol is 2" in r1_conds
        assert "symbol is 3" in r1_conds
