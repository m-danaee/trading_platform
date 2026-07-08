"""RB governor capital budget normalization tests."""

from __future__ import annotations

import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.rb_governor import (
    _assert_capital_budget,
    _enforce_capital_budget,
    _strategy,
)


def test_enforce_capital_budget_scales_down_when_over_cap():
    rules = [
        {"conditions": ["a"], "tp": 2.0, "sl": 1.0, "capital_pct": 30.0},
        {"conditions": ["b"], "tp": 2.0, "sl": 1.0, "capital_pct": 30.0},
        {"conditions": ["c"], "tp": 2.0, "sl": 1.0, "capital_pct": 30.0},
        {"conditions": ["d"], "tp": 2.0, "sl": 1.0, "capital_pct": 30.0},
    ]
    normalized = _enforce_capital_budget(rules, max_total=100.0)
    total = sum(r["capital_pct"] for r in normalized)
    assert total == pytest.approx(100.0, rel=1e-6, abs=1e-6)


def test_assert_capital_budget_raises_when_over_cap():
    rules = [{"capital_pct": 60.0}, {"capital_pct": 50.0}]
    with pytest.raises(ValueError, match="capital budget exceeded"):
        _assert_capital_budget(rules, max_total=100.0)


def test_strategy_output_respects_rb_max_total_capital(monkeypatch):
    monkeypatch.setattr(_cfg, "RB_MAX_TOTAL_CAPITAL", 100.0)
    strategy = _strategy(
        "long",
        [
            {"conditions": ["a"], "tp": 2.0, "sl": 1.0, "capital_pct": 40.0},
            {"conditions": ["b"], "tp": 2.0, "sl": 1.0, "capital_pct": 40.0},
            {"conditions": ["c"], "tp": 2.0, "sl": 1.0, "capital_pct": 40.0},
        ],
    )
    total = sum(r["capital_pct"] for r in strategy["rules_set"])
    assert total == pytest.approx(100.0, rel=1e-6, abs=1e-6)
