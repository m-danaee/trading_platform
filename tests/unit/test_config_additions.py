from gpu_fuzzy_trader import config as c

def test_new_config_parameters_exist():
    assert c.DEBUG_SYMBOL_SCOPE_ENABLED is False
    assert c.DEBUG_SYMBOL == "1"
    assert c.DEBUG_SYMBOL_COUNT == 1
    assert c.PHASE2_ISLAND_EPOCH_GENERATIONS >= 1
    assert c.PHASE2_MIGRATION_SEED_FRACTION < 1.0
    assert c.PHASE2_SHARED_ARCHIVE_MIN_SYMBOLS >= 1
    assert c.PHASE2_REGIME_PROFITABILITY_GATE is True
    assert c.PHASE2_REGIME_MIN_RETURN_PER_REGIME == 0.25
    assert c.PHASE1_REQUIRE_SIGN_CONSISTENCY is True
    assert c.PHASE1_SIGN_CONSISTENCY_MIN_FOLDS == 2
    assert c.PHASE2_RECENCY_WEIGHT_ENABLED is True
    assert c.PHASE2_RECENCY_WEIGHT_FRACTION == 0.25
    assert c.PHASE2_RECENCY_WEIGHT_MULTIPLIER == 2.0
    assert c.PHASE2_REQUIRE_LAST_FOLD_POSITIVE is False
    assert c.PHASE2_SCAN_UNROLL == 32

