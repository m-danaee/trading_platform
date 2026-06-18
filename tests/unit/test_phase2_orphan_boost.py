"""Unit tests for guarded migration and orphan detection helpers."""

from __future__ import annotations

import pandas as pd

from gpu_fuzzy_trader.phases.phase2_island_scheduler import (
    _score_rule_on_symbol_val,
    _symbol_has_viable_pool_rule,
)


def test_orphan_detection_empty_pool():
    val_df = pd.DataFrame({
        "symbol": ["A"] * 10,
        "datetime": pd.date_range("2024-01-01", periods=10, freq="5min"),
        "label_open_next": range(10),
    })
    assert not _symbol_has_viable_pool_rule([], "A", val_df, "long")


def test_score_rule_on_symbol_returns_trades():
    val_df = pd.DataFrame({
        "symbol": ["A"] * 5,
        "datetime": pd.date_range("2024-01-01", periods=5, freq="5min"),
        "label_open_next": [1.0, 1.1, 1.2, 1.3, 1.4],
    })
    rule = {"conditions": [], "tp": 2.0, "sl": 1.0, "capital_pct": 30.0}
    score = _score_rule_on_symbol_val(rule, val_df, "long")
    assert "trades" in score
    assert "return_pct" in score
