from gpu_fuzzy_trader import config as c


def test_effective_min_profitable_symbols_caps_debug_universe(monkeypatch):
    monkeypatch.setattr(c, "DEBUG_SYMBOL_SCOPE_ENABLED", True)
    monkeypatch.setattr(c, "DEBUG_SYMBOL_COUNT", 2)
    monkeypatch.setattr(c, "PHASE2_MIN_PROFITABLE_SYMBOLS", 5)
    assert c.effective_min_profitable_symbols() == 2
    assert c.effective_min_profitable_symbols(symbol_count=7) == 5


def test_new_config_parameters_exist():
    assert hasattr(c, 'DEBUG_SYMBOL_SCOPE_ENABLED')
    assert c.DEBUG_SYMBOL == "BTCUSDT"
    assert hasattr(c, 'DEBUG_SYMBOL_COUNT')
    assert c.PHASE1_REQUIRE_SIGN_CONSISTENCY is True
    assert c.PHASE1_SIGN_CONSISTENCY_MIN_FOLDS == 4
    assert c.PHASE2_REQUIRE_LAST_FOLD_POSITIVE is False
    assert c.PHASE2_SCAN_UNROLL == 32
    assert c.RB_MIN_RULES == 1
    assert c.RB_MAX_RULES == 20
    assert c.RB_MAX_TOTAL_CAPITAL == 100.0
    assert c.RB_CAPITAL_GRID[0] == 5.0
    assert not hasattr(c, "PHASE2_MIGRATION_EPOCH_INTERVAL")
    assert not hasattr(c, "MONTHLY_VALIDATION_ENABLED")
    assert not hasattr(c, "RB_PROFIT_AMP_MAX_RULES")



def test_default_pipeline_inputs_are_raw_sources():
    assert c.TRAIN_CSV_PATH == c.RAW_TRAIN_CSV_PATH
    assert c.TEST_CSV_PATH == c.RAW_TEST_CSV_PATH
    assert c.RAW_TRAIN_CSV_PATH.endswith("train_new.csv")
    assert c.RAW_TEST_CSV_PATH.endswith("test_new.csv")

