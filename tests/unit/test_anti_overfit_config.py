"""Regression tests for anti-overfit / exploration-retune config bundle."""

from __future__ import annotations

from gpu_fuzzy_trader import config as cfg


def test_anti_overfit_config_bundle():
    assert cfg.PHASE2_JOINT_TRAIN_VAL is False
    assert cfg.PHASE2_VAL_SIM_INTERVAL == 2
    assert cfg.PHASE2_VAL_IN_FITNESS_PENALTY is False
    assert cfg.PHASE2_CAPITAL_PCT == 18.0
    assert cfg.PHASE2_MAX_TRAIN_VAL_GAP_PCT == 10.0
    assert cfg.PHASE2_OVERFIT_RATIO_FLOOR == 2.5
    assert cfg.PHASE2_OVERFIT_GAP_PENALTY_WEIGHT == 15.0
    assert cfg.PHASE2_OVERFIT_GAP_PCT_THRESHOLD == 8.0
    assert cfg.PHASE2_MONTHLY_ADMISSION_MIN_RATIO == 0.50
    assert cfg.PHASE2_DIVERSITY_PENALTY == 3.0
    assert cfg.PHASE2_MUTATION_RATE == 0.32
    assert cfg.PHASE2_PLATEAU_EARLY_STOP_PATIENCE == 7
    assert cfg.PHASE2_PLATEAU_MAX_RESTARTS == 3
    assert cfg.PHASE2_PROFIT_FACTOR_FLOOR_ADMISSION == 1.15
    assert cfg.PHASE2_GENERATIONS == 96
    assert cfg.PHASE2_TWO_STAGE_ENABLED is True
    assert cfg.PHASE2_SAMPLE_MAX_BARS_PER_SYMBOL == 60_000
    assert cfg.PHASE2_SAMPLE_ROTATION_FRACTION == 0.65
    assert cfg.PHASE2_TP == 2.0
    assert cfg.PHASE2_MONTHLY_ADMISSION_MIN_MONTHS == 3
    assert cfg.PHASE1_DISABLED is False
    assert cfg.PHASE2_DIVERSITY_ON_F4 is True
    assert cfg.RB_TAIL_HOLDOUT_HARD_GATE is True
    assert cfg.RB_MAX_SYMBOL_SHARE_ABS_PNL == 0.50
    assert cfg.RB_MAX_RULES == 10
    assert cfg.RB_TRAIN_VALID_MAX_RATIO == 1.15
    assert cfg.PHASE2_VAL_RETURN_FLOOR_PCT_SHORT == 2.0
    assert cfg.effective_phase2_val_return_floor_pct("short") == 2.0
    assert cfg.effective_phase2_val_return_floor_pct("long") == 1.0
    assert cfg.RB_MIN_COMBINED_RETURN_IMPROVEMENT == 3.5
    assert cfg.RB_MAX_PAIR_OVERLAP == 0.25
    assert cfg.RB_CAPITAL_GRID[-1] == 25.0


def test_cluster_island_symbol_robustness_enabled():
    hp = cfg.resolve_island_hyperparams(
        "cluster", n_rows=175_000, reference_rows=700_000, n_symbols=4,
    )
    assert hp.skip_symbol_robustness_penalty is False
    assert hp.min_profitable_symbols >= 2
