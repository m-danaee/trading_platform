from gpu_fuzzy_trader import config as cfg


def test_tp_sl_minimums_include_one_percent():
    """Validate that one percent TP and SL are accepted."""
    assert cfg.RB_MIN_TP == 1.0
    assert cfg.RB_MIN_SL == 1.0
    assert min(cfg.RB_TP_GRID) >= 1.0
    assert min(cfg.RB_SL_GRID) >= 1.0


def test_symbol_filters_are_required_for_rb_outputs():
    """Validate that final RB rules must include symbol filters."""
    assert cfg.RB_REQUIRE_SYMBOL_FILTERS is True
    assert cfg.RB_SYMBOL_STRICT_OUTPUT_CHECK is True
    assert cfg.RB_SYMBOL_MAX_SYMBOLS_PER_RULE >= 1
