"""Regression tests for anti-overfit config bundle (inflated-returns fix)."""

from __future__ import annotations

from gpu_fuzzy_trader import config as cfg


def test_anti_overfit_config_bundle():
    assert cfg.PHASE2_JOINT_TRAIN_VAL is True
    assert cfg.PHASE2_VAL_SIM_INTERVAL == 1
    assert cfg.PHASE2_VAL_IN_FITNESS_PENALTY is True
    assert cfg.PHASE2_CAPITAL_PCT == 18.0
    assert cfg.PHASE2_MAX_TRAIN_VAL_GAP_PCT == 10.0
    assert cfg.PHASE2_OVERFIT_RATIO_FLOOR == 2.5
    assert cfg.PHASE2_MONTHLY_ADMISSION_MIN_RATIO == 0.65
    assert cfg.PHASE2_DIVERSITY_PENALTY == 6.0
    assert cfg.PHASE2_GENERATIONS == 96
    assert cfg.PHASE2_TWO_STAGE_ENABLED is True
    assert cfg.RB_MAX_RULES == 5
    assert cfg.RB_MIN_COMBINED_RETURN_IMPROVEMENT == 3.5
    assert cfg.RB_MAX_PAIR_OVERLAP == 0.25
    assert cfg.RB_CAPITAL_GRID[-1] == 25.0


def test_cluster_island_symbol_robustness_enabled():
    hp = cfg.resolve_island_hyperparams(
        "cluster", n_rows=175_000, reference_rows=700_000, n_symbols=4,
    )
    assert hp.skip_symbol_robustness_penalty is False
    assert hp.min_profitable_symbols >= 3
