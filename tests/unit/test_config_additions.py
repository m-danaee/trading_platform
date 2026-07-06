from gpu_fuzzy_trader import config as c


def test_effective_min_profitable_symbols_caps_debug_universe(monkeypatch):
    monkeypatch.setattr(c, "DEBUG_SYMBOL_SCOPE_ENABLED", True)
    monkeypatch.setattr(c, "DEBUG_SYMBOL_COUNT", 2)
    monkeypatch.setattr(c, "PHASE2_MIN_PROFITABLE_SYMBOLS", 5)
    assert c.effective_min_profitable_symbols() == 2
    assert c.effective_min_profitable_symbols(symbol_count=7) == 5


def test_effective_phase3_thresholds_relax_in_debug_scope(monkeypatch):
    monkeypatch.setattr(c, "DEBUG_SYMBOL_SCOPE_ENABLED", True)
    monkeypatch.setattr(c, "DEBUG_SYMBOL_COUNT", 2)
    monkeypatch.setattr(c, "PHASE3_PER_SYMBOL_MIN_TRADES", 35)
    monkeypatch.setattr(c, "PHASE3_PER_SYMBOL_MIN_RETURN", 3.0)
    monkeypatch.setattr(c, "PHASE3_VAL_RETURN_FLOOR_PCT", 5.0)
    assert c.effective_phase3_per_symbol_min_trades() == 15
    assert c.effective_phase3_per_symbol_min_return() == 2.5
    assert c.effective_phase3_val_return_floor_pct() == 4.0


def test_effective_phase3_thresholds_unchanged_full_universe(monkeypatch):
    monkeypatch.setattr(c, "DEBUG_SYMBOL_SCOPE_ENABLED", False)
    monkeypatch.setattr(c, "PHASE3_PER_SYMBOL_MIN_TRADES", 35)
    monkeypatch.setattr(c, "PHASE3_PER_SYMBOL_MIN_RETURN", 3.0)
    monkeypatch.setattr(c, "PHASE3_VAL_RETURN_FLOOR_PCT", 5.0)
    assert c.effective_phase3_per_symbol_min_trades() == 35
    assert c.effective_phase3_per_symbol_min_return() == 3.0
    assert c.effective_phase3_val_return_floor_pct() == 5.0


def test_new_config_parameters_exist():
    assert hasattr(c, 'DEBUG_SYMBOL_SCOPE_ENABLED')
    assert c.DEBUG_SYMBOL == "1"
    assert hasattr(c, 'DEBUG_SYMBOL_COUNT')
    assert hasattr(c, 'PHASE2_ISLAND_EPOCH_GENERATIONS')
    assert c.PHASE2_ISLAND_EPOCH_GENERATIONS == 13
    assert c.PHASE2_ISLAND_MODE in ("global", "cluster")
    assert hasattr(c, 'PHASE2_MIGRATION_SEED_FRACTION')
    assert c.PHASE2_MIGRATION_SEED_FRACTION == 0.10
    assert hasattr(c, 'PHASE2_SHARED_ARCHIVE_MIN_SYMBOLS') is False or 'PHASE2_SHARED_ARCHIVE_MIN_SYMBOLS' not in dir(c)
    assert c.PHASE1_REQUIRE_SIGN_CONSISTENCY is True
    assert c.PHASE1_SIGN_CONSISTENCY_MIN_FOLDS == 2
    assert c.PHASE2_REQUIRE_LAST_FOLD_POSITIVE is False
    assert c.PHASE2_SCAN_UNROLL == 32

