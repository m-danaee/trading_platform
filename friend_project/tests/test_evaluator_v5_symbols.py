import numpy as np
import pandas as pd

from gpu_fuzzy_trader.rb_evaluator_v5 import _build_entries, build_rule_signal_mask, parse_symbol_condition


def test_symbol_condition_formats_are_normalized():
    """Validate accepted symbol condition formats."""
    assert parse_symbol_condition("symbol is 3") == ["3"]
    assert parse_symbol_condition("[symbol] is 2, 6") == ["2", "6"]


def test_symbol_filter_restricts_rule_signal_mask():
    """Validate that symbol filters reduce matching rows."""
    df = pd.DataFrame({"symbol": ["1", "2", "3"], "feature_a": [0.5, 0.5, 0.5]})
    mask = build_rule_signal_mask(df, ["[feature_a] IS Medium", "symbol is 2"])
    assert mask.tolist() == [False, True, False]


def test_rule_entries_use_symbol_priority_inside_same_time():
    """Validate evaluator ordering for symbols inside the same timestamp."""
    df = pd.DataFrame({
        "datetime": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-01"]),
        "symbol": ["1", "2", "3"],
        "feature_a": [0.5, 0.5, 0.5],
    })
    rule_set = [{
        "conditions": ["[feature_a] IS Medium", "symbol is 2", "symbol is 1"],
        "tp": 2.0,
        "sl": 1.0,
        "capital_pct": 10.0,
    }]
    entries = _build_entries(df, rule_set, row_priority=np.array([0, 0, 0]))
    assert [entry["idx"] for entry in entries] == [1, 0]
