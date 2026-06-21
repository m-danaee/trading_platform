from gpu_fuzzy_trader import config as cfg


def test_tp_sl_minimums_match_config():
    """Validate RB Governor TP/SL floors and grid lower bounds."""
    assert cfg.RB_MIN_TP == 1.5
    assert cfg.RB_MIN_SL == 1.0
    assert min(cfg.RB_TP_GRID) >= cfg.RB_MIN_TP
    assert min(cfg.RB_SL_GRID) >= cfg.RB_MIN_SL


def test_symbol_filters_are_required_for_rb_outputs():
    """Validate that final RB rules must include symbol filters."""
    assert cfg.RB_REQUIRE_SYMBOL_FILTERS is True
    assert cfg.RB_SYMBOL_STRICT_OUTPUT_CHECK is True
    assert cfg.RB_SYMBOL_MAX_SYMBOLS_PER_RULE >= 1
