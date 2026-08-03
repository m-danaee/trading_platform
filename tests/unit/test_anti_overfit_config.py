"""Regression tests for anti-overfit / exploration-retune config bundle."""

from __future__ import annotations

from gpu_fuzzy_trader import config as cfg


def test_anti_overfit_config_bundle():
    assert cfg.PHASE2_JOINT_TRAIN_VAL is False
    assert cfg.PHASE2_VAL_SIM_INTERVAL == 2
    assert cfg.PHASE2_VAL_IN_FITNESS_PENALTY is False
    assert cfg.PHASE2_CAPITAL_PCT == 18.0
    assert cfg.PHASE2_MAX_TRAIN_VAL_GAP_PCT == 10.0
    assert cfg.PHASE2_OVERFIT_RATIO_FLOOR == 3.0
    assert cfg.PHASE2_OVERFIT_GAP_PENALTY_WEIGHT == 15.0
    assert cfg.PHASE2_OVERFIT_GAP_PCT_THRESHOLD == 8.0
    assert cfg.PHASE2_MONTHLY_ADMISSION_MIN_RATIO == 0.50
    assert cfg.PHASE2_MONTHLY_GOOD_RETURN_MIN_PCT == 0.0
    assert cfg.PHASE2_DIVERSITY_PENALTY == 3.0
    assert cfg.PHASE2_MUTATION_RATE == 0.32
    assert cfg.PHASE2_PLATEAU_EARLY_STOP_PATIENCE == 7
    assert cfg.PHASE2_PLATEAU_MAX_RESTARTS == 1
    assert cfg.PHASE2_PROFIT_FACTOR_FLOOR_ADMISSION == 1.15
    assert cfg.PHASE2_PROFIT_FACTOR_FLOOR_EVOLUTION == 1.0
    assert cfg.PHASE2_RETURN_FLOOR_PCT == 0.25
    assert cfg.PHASE2_GENERATIONS == 40
    assert cfg.PHASE2_POPULATION_SIZE == 200
    assert cfg.PHASE2_TWO_STAGE_ENABLED is True
    assert cfg.PHASE2_SAMPLE_MAX_BARS_PER_SYMBOL == 60_000
    assert cfg.PHASE2_SAMPLE_ROTATION_FRACTION == 0.65
    assert cfg.PHASE2_TP == 2.0
    assert cfg.PHASE2_MONTHLY_ADMISSION_MIN_MONTHS == 2
    assert cfg.PHASE1_DISABLED is False
    assert cfg.PHASE2_DIVERSITY_ON_F4 is True
    assert cfg.PHASE2_USE_TOTAL_RETURN_OBJ is False
    assert cfg.PHASE2_MIN_PROFITABLE_SYMBOLS == 2
    assert cfg.RB_TAIL_HOLDOUT_HARD_GATE is True
    assert cfg.RB_MAX_SYMBOL_SHARE_ABS_PNL == 0.67
    assert cfg.RB_MAX_SYMBOL_HHI == 0.60
    assert cfg.RB_MIN_TRAIN_RETURN == 0.25
    assert cfg.RB_MIN_VALID_RETURN == 0.25
    assert cfg.RB_MIN_SCORE_IMPROVEMENT == 0.01
    assert cfg.RB_MIN_TRAIN_RETURN_IMPROVEMENT == 0.002
    assert cfg.RB_MIN_VALID_RETURN_IMPROVEMENT == 0.002
    assert cfg.RB_MULTI_SYMBOL_COVERAGE_BONUS == 15.0
    assert cfg.RB_MAX_RULES == 20
    assert cfg.RB_MAX_TOTAL_CAPITAL == 100.0
    assert cfg.RB_TRAIN_VALID_MAX_RATIO == 2.00
    assert cfg.RB_MIN_DISTINCT_SYMBOLS == 2
    assert cfg.PHASE2_VAL_RETURN_FLOOR_PCT == 0.25
    assert cfg.PHASE2_VAL_RETURN_FLOOR_PCT_SHORT == 0.25
    assert cfg.effective_phase2_val_return_floor_pct("short") == 0.25
    assert cfg.effective_phase2_val_return_floor_pct("long") == 0.25
    assert cfg.MIN_TRADE_SUPPORT == 60
    assert cfg.MIN_TRADE_POOL_FLOOR == 15
    assert cfg.PHASE2_SUPPORT_PENALTY_WEIGHT_F1 == 0.25
    assert cfg.PHASE2_ISLAND_TWO_STAGE_ENABLED is True
    assert cfg.PHASE2_ONE_SYMBOL_ISLANDS is False
    assert cfg.PHASE2_MIGRATION_ENABLED is True
    assert cfg.PHASE2_EARLY_STOP_ENABLED is False
    assert cfg.PHASE2_PLATEAU_EARLY_STOP_ENABLED is False
    assert cfg.PHASE2_ISLAND_PLATEAU_EARLY_STOP_ENABLED is False
    assert cfg.PHASE2_STAGE_A_GENERATIONS == 20
    assert cfg.PHASE2_STAGE_B_GENERATIONS == 20
    assert cfg.PHASE2_STAGE_A_MUTATION_WEIGHTED_ACTIVATE_PROB == 0.70
    assert cfg.RB_MIN_RULES == 1
    assert cfg.RB_MAX_RULES == 20
    assert cfg.RB_MIN_COMBINED_RETURN_IMPROVEMENT == 3.5
    assert cfg.RB_MAX_PAIR_OVERLAP == 0.35
    assert cfg.RB_CAPITAL_GRID[0] == 5.0
    assert cfg.RB_CAPITAL_GRID[-1] == 18.0
    assert cfg.RB_MAX_TOTAL_CAPITAL == cfg.MAX_TOTAL_EXPOSURE_PCT
    assert cfg.RB_MAX_TOTAL_CAPITAL >= cfg.RB_MAX_RULES * cfg.RB_CAPITAL_GRID[0]


def test_cluster_island_symbol_robustness_enabled():
    hp = cfg.resolve_island_hyperparams(
        "cluster", n_rows=175_000, reference_rows=700_000, n_symbols=4,
    )
    assert hp.skip_symbol_robustness_penalty is False
    assert hp.min_profitable_symbols >= 2


def test_one_symbol_island_hyperparams_target_one_profitable():
    hp = cfg.resolve_island_hyperparams(
        "cluster", n_rows=60_000, reference_rows=600_000, n_symbols=1,
    )
    assert hp.min_profitable_symbols == 1
