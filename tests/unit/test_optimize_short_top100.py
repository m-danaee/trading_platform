import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.optimize_short_top100 import (  # noqa: E402
    compute_profit_factor,
    pin_rule,
    rule_score,
    strip_symbol_conditions,
    symbol_passes_gate,
)


def test_compute_profit_factor_basic():
    assert compute_profit_factor([10.0, -5.0]) == pytest.approx(2.0)


def test_compute_profit_factor_no_losses_returns_inf_sentinel_for_gate_then_capped_in_score():
    pf = compute_profit_factor([3.0, 2.0])
    assert math.isinf(pf)
    assert rule_score(5.0, pf) == pytest.approx(50.0)  # 5 * 10 cap


def test_symbol_passes_gate_b():
    assert symbol_passes_gate(10.0, 1.2, 5) is True
    assert symbol_passes_gate(10.0, 1.19, 5) is False
    assert symbol_passes_gate(10.0, 1.2, 4) is False
    assert symbol_passes_gate(0.0, 2.0, 10) is False
    assert symbol_passes_gate(-1.0, 2.0, 10) is False


def test_strip_and_pin_rule():
    rule = {
        "conditions": ["[rsi_centered_14] IS Bearish", "symbol is 1,2"],
        "tp": 2.0,
        "sl": 1.0,
        "capital_pct": 10.0,
    }
    assert strip_symbol_conditions(rule["conditions"]) == [
        "[rsi_centered_14] IS Bearish"]
    pinned = pin_rule(rule, [3, 1, 10])
    assert pinned["conditions"][-1] == "symbol is 1,3,10"
    assert pinned["conditions"][0] == "[rsi_centered_14] IS Bearish"
    assert rule["conditions"][-1] == "symbol is 1,2"  # original untouched
