from gpu_fuzzy_trader.rb_governor import _has_symbol_condition, _rule_risk_ok


def test_rule_risk_accepts_one_percent_values():
    """Validate the accepted TP and SL boundary."""
    rule = {"conditions": ["symbol is 1"], "tp": 1.0, "sl": 1.0, "capital_pct": 10.0}
    assert _rule_risk_ok(rule) is True


def test_rule_risk_rejects_values_below_one_percent():
    """Validate rejection below the TP and SL boundary."""
    assert _rule_risk_ok({"conditions": ["symbol is 1"], "tp": 0.9, "sl": 1.0, "capital_pct": 10.0}) is False
    assert _rule_risk_ok({"conditions": ["symbol is 1"], "tp": 1.0, "sl": 0.9, "capital_pct": 10.0}) is False


def test_symbol_detection_contract():
    """Validate symbol condition detection in generated rules."""
    assert _has_symbol_condition({"conditions": ["[x] IS Medium", "symbol is 4"]}) is True
    assert _has_symbol_condition({"conditions": ["[x] IS Medium"]}) is False
