"""Mode A must keep Phase 2 island ``source_symbols`` as OR filters."""

from __future__ import annotations

from unittest.mock import MagicMock

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.rb_governor import (
    _ensure_symbol_filtered_rule,
    _is_positive_good,
    _source_symbols_from_rule,
    _symbol_specialized_variants,
)


def test_source_symbols_parsed_deduped():
    rule = {"source_symbols": ["3", "4", "3", " 7 "]}
    assert _source_symbols_from_rule(rule) == ["3", "4", "7"]


def test_mode_a_attaches_island_symbol_or_filters():
    raw = {
        "conditions": ["[atr_pct_14] IS High", "symbol is 9"],
        "tp": 2.0,
        "sl": 1.2,
        "capital_pct": 15.0,
        "source_symbols": ["3", "4", "7", "8"],
    }
    train_engine = MagicMock()
    valid_engine = MagicMock()
    with __import__("unittest").mock.patch.object(_cfg, "RB_REQUIRE_SYMBOL_FILTERS", False):
        variants = _symbol_specialized_variants(
            raw, train_engine, valid_engine, ["1", "2"])
    assert len(variants) == 1
    conds = variants[0]["conditions"]
    assert "[atr_pct_14] IS High" in conds
    assert "symbol is 9" not in conds  # orphan filter stripped
    for sym in ("3", "4", "7", "8"):
        assert f"symbol is {sym}" in conds


def test_mode_a_without_source_symbols_stays_generalist():
    raw = {
        "conditions": ["[atr_pct_14] IS High", "symbol is 1"],
        "tp": 2.0,
        "sl": 1.2,
        "capital_pct": 15.0,
    }
    with __import__("unittest").mock.patch.object(_cfg, "RB_REQUIRE_SYMBOL_FILTERS", False):
        variants = _symbol_specialized_variants(
            raw, MagicMock(), MagicMock(), ["1", "2"])
    assert variants[0]["conditions"] == ["[atr_pct_14] IS High"]


def test_ensure_symbol_filtered_preserves_island_scope_in_mode_a():
    rule = {
        "conditions": ["[x] IS Low"],
        "source_symbols": ["1", "6", "2"],
        "tp": 2.0,
        "sl": 1.2,
        "capital_pct": 10.0,
    }
    with __import__("unittest").mock.patch.object(_cfg, "RB_REQUIRE_SYMBOL_FILTERS", False):
        out = _ensure_symbol_filtered_rule(rule, ["1", "2", "3"])
    assert "symbol is 1" in out["conditions"]
    assert "symbol is 6" in out["conditions"]
    assert "symbol is 2" in out["conditions"]


def test_execution_health_not_hard_gate_by_default():
    train_m = {
        "total_return_pct": 5.0,
        "profit_factor": 1.2,
        "executed_trades": 40,
        "raw_signal_count": 100,
        "skipped_min_notional_count": 40,  # 0.40 > default 0.20
    }
    valid_m = {
        "total_return_pct": 2.0,
        "profit_factor": 1.15,
        "executed_trades": 20,
        "raw_signal_count": 50,
        "skipped_min_notional_count": 20,
    }
    with __import__("unittest").mock.patch.object(
        _cfg, "RB_REQUIRE_EXECUTION_HEALTH_ON_SINGLES", False
    ):
        assert _is_positive_good(train_m, valid_m) is True
    with __import__("unittest").mock.patch.object(
        _cfg, "RB_REQUIRE_EXECUTION_HEALTH_ON_SINGLES", True
    ):
        assert _is_positive_good(train_m, valid_m) is False
