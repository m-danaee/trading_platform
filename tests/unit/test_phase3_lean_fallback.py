"""
Unit tests for ``_try_lean_fallback`` in phase3_rule_set.py (Task 10.2).

Tests cover:
  - Returns top-2 rules when pool has ≥ 2 rules.
  - Returns ``None`` when pool has < 2 rules.
  - Returned rules are sorted by ``_pool_rule_val_score`` descending.
  - A WARNING is logged explaining the relaxation.
"""

from __future__ import annotations

import logging

import pytest

from gpu_fuzzy_trader.phases.phase3_rule_set import (
    _pool_rule_val_score,
    _try_lean_fallback,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pool_rule(
    conditions: list[str] | None = None,
    train_ret: float = 5.0,
    val_ret: float = 4.0,
) -> dict:
    """Create a minimal pool rule dict with objectives for scoring."""
    if conditions is None:
        conditions = ["[feature_a] IS High"]
    return {
        "conditions": list(conditions),
        "objectives": {
            "total_return_pct": train_ret,
        },
        "val_objectives": {
            "total_return_pct": val_ret,
        },
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTryLeanFallback:
    """Tests for _try_lean_fallback."""

    def test_returns_top_2_rules_when_pool_has_4_rules(self):
        """With 4 rules, returns exactly the top 2 by _pool_rule_val_score."""
        pool = [
            _make_pool_rule(["rule_a"], train_ret=10.0, val_ret=8.0),    # min=8.0
            _make_pool_rule(["rule_b"], train_ret=5.0, val_ret=12.0),    # min=5.0
            _make_pool_rule(["rule_c"], train_ret=3.0, val_ret=2.0),    # min=2.0
            _make_pool_rule(["rule_d"], train_ret=7.0, val_ret=6.0),    # min=6.0
        ]
        result = _try_lean_fallback(pool, "long", global_min=2)
        assert result is not None
        assert len(result) == 2

        # Should be top 2 by _pool_rule_val_score
        scores = [_pool_rule_val_score(r) for r in result]
        assert scores == sorted(scores, reverse=True), (
            f"Rules not sorted by score descending: {scores}"
        )
        # rule_a (min=8.0) and rule_d (min=6.0) should be the top 2
        top_conditions = {str(r["conditions"][0]) for r in result}
        assert "rule_a" in top_conditions
        assert "rule_d" in top_conditions

    def test_returns_none_when_pool_has_1_rule(self):
        """With only 1 rule, returns None because we need at least 2."""
        pool = [_make_pool_rule(["only_rule"], train_ret=5.0, val_ret=5.0)]
        result = _try_lean_fallback(pool, "short", global_min=2)
        assert result is None

    def test_returns_none_when_pool_empty(self):
        """With empty pool, returns None."""
        result = _try_lean_fallback([], "long", global_min=2)
        assert result is None

    def test_custom_global_min(self):
        """With global_min=3 and 5 rules, returns 3 rules."""
        pool = [
            _make_pool_rule(["r1"], train_ret=10.0, val_ret=10.0),
            _make_pool_rule(["r2"], train_ret=9.0, val_ret=9.0),
            _make_pool_rule(["r3"], train_ret=8.0, val_ret=8.0),
            _make_pool_rule(["r4"], train_ret=7.0, val_ret=7.0),
            _make_pool_rule(["r5"], train_ret=6.0, val_ret=6.0),
        ]
        result = _try_lean_fallback(pool, "long", global_min=3)
        assert result is not None
        assert len(result) == 3
        # Should be the top 3 rules
        top_scores = [_pool_rule_val_score(r) for r in result]
        assert top_scores == [10.0, 9.0, 8.0], (
            f"Expected top 3 scores [10, 9, 8], got {top_scores}"
        )

    def test_rules_sorted_descending(self):
        """Returned rules are sorted by _pool_rule_val_score descending."""
        pool = [
            _make_pool_rule(["low"], train_ret=1.0, val_ret=1.0),
            _make_pool_rule(["high"], train_ret=15.0, val_ret=15.0),
            _make_pool_rule(["medium"], train_ret=8.0, val_ret=8.0),
        ]
        result = _try_lean_fallback(pool, "long", global_min=2)
        assert result is not None
        assert len(result) == 2
        scores = [_pool_rule_val_score(r) for r in result]
        assert scores == [15.0, 8.0], (
            f"Expected [15.0, 8.0], got {scores}"
        )

    def test_warning_logged_at_warning_level(self, caplog):
        """A WARNING is logged explaining the relaxation."""
        pool = [
            _make_pool_rule(["a"], train_ret=10.0, val_ret=10.0),
            _make_pool_rule(["b"], train_ret=8.0, val_ret=8.0),
        ]
        with caplog.at_level(logging.WARNING, logger="gpu_fuzzy_trader.phases.phase3_rule_set"):
            result = _try_lean_fallback(pool, "long", global_min=2)
            assert result is not None

        # Check that the warning message exists
        warning_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and "_try_lean_fallback" in r.getMessage()
        ]
        assert len(warning_records) >= 1, (
            f"Expected at least 1 WARNING with '_try_lean_fallback', "
            f"got {len(warning_records)}: {[r.message for r in caplog.records]}"
        )
        # The message should mention the relaxation
        msg = warning_records[0].getMessage()
        assert "relaxed" in msg.lower() or "relaxation" in msg.lower(), (
            f"Warning should mention relaxation: {msg}"
        )

    def test_pool_with_exactly_2_rules(self):
        """With exactly 2 rules, returns both."""
        pool = [
            _make_pool_rule(["x"], train_ret=8.0, val_ret=8.0),
            _make_pool_rule(["y"], train_ret=6.0, val_ret=6.0),
        ]
        result = _try_lean_fallback(pool, "long", global_min=2)
        assert result is not None
        assert len(result) == 2
        # Both should be returned
        conditions = {str(r["conditions"][0]) for r in result}
        assert conditions == {"x", "y"}

    def test_returns_none_when_all_below_floor(self, monkeypatch):
        """When all pool rules have min(train,val) below val_floor, returns None."""
        from gpu_fuzzy_trader import config as _cfg
        pool = [
            _make_pool_rule(["a"], train_ret=1.0, val_ret=1.0),   # score = 1.0
            _make_pool_rule(["b"], train_ret=0.5, val_ret=0.5),   # score = 0.5
        ]
        monkeypatch.setattr(_cfg, "effective_phase3_val_return_floor_pct", lambda: 5.0)

        result = _try_lean_fallback(pool, "long", global_min=2)
        assert result is None

    def test_filters_mixed_above_below_floor(self, monkeypatch):
        """When some rules are above val_floor and some below, returns None if
        too few survive the floor filter."""
        from gpu_fuzzy_trader import config as _cfg
        pool = [
            _make_pool_rule(["a"], train_ret=10.0, val_ret=10.0),  # score = 10.0 (above)
            _make_pool_rule(["b"], train_ret=1.0, val_ret=1.0),    # score = 1.0 (below)
        ]
        monkeypatch.setattr(_cfg, "effective_phase3_val_return_floor_pct", lambda: 5.0)

        result = _try_lean_fallback(pool, "long", global_min=2)
        # Only 1 rule above floor, but global_min=2, so returns None
        assert result is None
