from gpu_fuzzy_trader import config as cfg
from gpu_fuzzy_trader.rb_governor import _has_symbol_condition, _rule_risk_ok


def test_rule_risk_accepts_config_minimum_values():
    """Validate the accepted TP and SL boundary from config."""
    rule = {
        "conditions": ["symbol is 1"],
        "tp": cfg.RB_MIN_TP,
        "sl": cfg.RB_MIN_SL,
        "capital_pct": 10.0,
    }
    assert _rule_risk_ok(rule) is True


def test_rule_risk_rejects_values_below_config_minimums():
    """Validate rejection below RB_MIN_TP / RB_MIN_SL."""
    assert _rule_risk_ok({
        "conditions": ["symbol is 1"],
        "tp": cfg.RB_MIN_TP - 0.1,
        "sl": cfg.RB_MIN_SL,
        "capital_pct": 10.0,
    }) is False
    assert _rule_risk_ok({
        "conditions": ["symbol is 1"],
        "tp": cfg.RB_MIN_TP,
        "sl": cfg.RB_MIN_SL - 0.1,
        "capital_pct": 10.0,
    }) is False
    assert _rule_risk_ok({
        "conditions": ["symbol is 1"],
        "tp": 1.0,
        "sl": cfg.RB_MIN_SL,
        "capital_pct": 10.0,
    }) is False


def test_symbol_detection_contract():
    """Validate symbol condition detection in generated rules."""
    assert _has_symbol_condition({"conditions": ["[x] IS Medium", "symbol is 4"]}) is True
    assert _has_symbol_condition({"conditions": ["[x] IS Medium"]}) is False
