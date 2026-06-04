"""
Search space and resource profiles for config-level Optuna tuning.
"""

from __future__ import annotations

from typing import Any

import optuna

from gpu_fuzzy_trader import config as _cfg

# Fixed caps for 2-core / 4GB runs (applied before trial-specific suggestions).
LOW_RAM_PROFILE: dict[str, Any] = {
    "PHASE2_POPULATION_SIZE": 100,
    "PHASE2_GENERATIONS": 50,
    "PHASE3_REFINE_POP_SIZE": 48,
    "PHASE3_REFINE_GENERATIONS": 25,
    "PHASE1_SAMPLING_TOTAL": 150_000,
    "PHASE4_N_TRIALS": 40,
    "PHASE2_USE_GPU": False,
    "PHASE3_USE_GPU": False,
    "PHASE2_CV_FOLD_WORKERS": 1,
    "PHASE3_BATCH_WORKERS": 2,
    "PHASE4_N_JOBS": 1,
    "PHASE2_NUMBA_ENABLED": True,
    "PHASE2_EARLY_STOP_ENABLED": False,
}

PROFILE_NAMES = ("low_ram",)


def get_profile_params(profile: str) -> dict[str, Any]:
    """Return fixed config overrides for a named resource profile."""
    if profile == "low_ram":
        return dict(LOW_RAM_PROFILE)
    raise ValueError(
        f"Unknown profile {profile!r}; choose from {PROFILE_NAMES}"
    )


def suggest_trial_params(trial: optuna.Trial, profile: str) -> dict[str, Any]:
    """
    Build full config override dict for one Optuna trial.

    Merges profile fixed caps with sampled generalization / budget knobs.
    """
    params: dict[str, Any] = get_profile_params(profile)

    params["CV_N_FOLDS"] = trial.suggest_categorical("CV_N_FOLDS", [2, 3])
    cv_folds = int(params["CV_N_FOLDS"])

    # --- Phase 2: CV admission & pool breadth ---
    params["PHASE2_CV_POOL_MIN_FOLDS_PASS"] = trial.suggest_int(
        "PHASE2_CV_POOL_MIN_FOLDS_PASS", 1, cv_folds,
    )
    params["PHASE2_CV_RANK_MIN_FOLDS_PASS"] = trial.suggest_int(
        "PHASE2_CV_RANK_MIN_FOLDS_PASS", 1, cv_folds,
    )
    params["PHASE2_CV_PROFIT_FACTOR_FLOOR"] = trial.suggest_float(
        "PHASE2_CV_PROFIT_FACTOR_FLOOR", 0.9, 1.1,
    )
    params["PHASE2_CV_POOL_VAL_RETURN_MIN_PCT"] = trial.suggest_float(
        "PHASE2_CV_POOL_VAL_RETURN_MIN_PCT", 0.0, 1.5,
    )
    params["PHASE2_CV_POOL_TRAIN_RETURN_MIN_PCT"] = trial.suggest_float(
        "PHASE2_CV_POOL_TRAIN_RETURN_MIN_PCT", 0.0, 1.0,
    )
    params["PHASE2_MIN_PROFITABLE_SYMBOLS"] = trial.suggest_int(
        "PHASE2_MIN_PROFITABLE_SYMBOLS", 4, 8,
    )
    params["PHASE2_SYMBOL_MEDIAN_RETURN_FLOOR_PCT"] = trial.suggest_float(
        "PHASE2_SYMBOL_MEDIAN_RETURN_FLOOR_PCT", -1.0, 0.5,
    )
    params["PHASE2_CV_POOL_RANK_ADMIT_TOP_K"] = trial.suggest_int(
        "PHASE2_CV_POOL_RANK_ADMIT_TOP_K", 20, 80,
    )
    params["PHASE2_REGIME_MIN_WIN_RATE"] = trial.suggest_float(
        "PHASE2_REGIME_MIN_WIN_RATE", 0.35, 0.50,
    )
    params["PHASE2_REQUIRE_LAST_FOLD_POSITIVE"] = trial.suggest_categorical(
        "PHASE2_REQUIRE_LAST_FOLD_POSITIVE", [True, False],
    )
    params["MIN_TRADE_SUPPORT"] = trial.suggest_int(
        "MIN_TRADE_SUPPORT", 100, 250,
    )
    params["PHASE2_JOINT_TRAIN_VAL"] = trial.suggest_categorical(
        "PHASE2_JOINT_TRAIN_VAL", [True, False],
    )

    # --- Phase 3: team selection & anti-overfit ---
    params["PHASE3_VAL_SORTINO_RATIO_GATE"] = trial.suggest_float(
        "PHASE3_VAL_SORTINO_RATIO_GATE", 0.35, 0.65,
    )
    params["PHASE3_VAL_DRAWDOWN_RATIO_GATE"] = trial.suggest_float(
        "PHASE3_VAL_DRAWDOWN_RATIO_GATE", 1.0, 1.25,
    )
    params["PHASE3_VAL_RETURN_FLOOR_PCT"] = trial.suggest_float(
        "PHASE3_VAL_RETURN_FLOOR_PCT", 0.0, 2.0,
    )
    params["PHASE3_VAL_PROFIT_FACTOR_FLOOR"] = trial.suggest_float(
        "PHASE3_VAL_PROFIT_FACTOR_FLOOR", 1.0, 1.15,
    )
    params["PHASE3_MIN_PROFITABLE_SYMBOLS"] = trial.suggest_int(
        "PHASE3_MIN_PROFITABLE_SYMBOLS", 5, 8,
    )
    params["PHASE3_MIN_SYMBOL_COVERAGE"] = trial.suggest_int(
        "PHASE3_MIN_SYMBOL_COVERAGE", 6, 9,
    )
    params["PHASE3_SYMBOL_CONSISTENCY_WEIGHT"] = trial.suggest_float(
        "PHASE3_SYMBOL_CONSISTENCY_WEIGHT", 5.0, 15.0,
    )
    params["PHASE3_JACCARD_SIMILARITY_GATE"] = trial.suggest_float(
        "PHASE3_JACCARD_SIMILARITY_GATE", 0.65, 0.85,
    )
    params["PHASE3_VAL_TRAIN_GAP_MAX_PCT"] = trial.suggest_float(
        "PHASE3_VAL_TRAIN_GAP_MAX_PCT", 5.0, 20.0,
    )
    params["PHASE3_TRAIN_VAL_GAP_MAX_PCT"] = trial.suggest_float(
        "PHASE3_TRAIN_VAL_GAP_MAX_PCT", 8.0, 25.0,
    )
    params["PHASE3_GAP_PENALTY_WEIGHT"] = trial.suggest_float(
        "PHASE3_GAP_PENALTY_WEIGHT", 2.0, 8.0,
    )
    params["PHASE3_VAL_GATE_PENALTY"] = trial.suggest_float(
        "PHASE3_VAL_GATE_PENALTY", 5.0, 15.0,
    )
    params["PHASE3_MIN_INCREMENTAL_TRADES"] = trial.suggest_int(
        "PHASE3_MIN_INCREMENTAL_TRADES", 40, 80,
    )

    # --- Phase 4: walk-forward feasibility ---
    params["PHASE4_MIN_WORST_FOLD_RETURN_PCT"] = trial.suggest_float(
        "PHASE4_MIN_WORST_FOLD_RETURN_PCT", -5.0, 0.0,
    )
    params["PHASE4_MIN_WORST_FOLD_PF"] = trial.suggest_float(
        "PHASE4_MIN_WORST_FOLD_PF", 0.95, 1.05,
    )
    params["PHASE4_MAX_WORST_DRAWDOWN_PCT"] = trial.suggest_float(
        "PHASE4_MAX_WORST_DRAWDOWN_PCT", 12.0, 20.0,
    )
    params["PHASE4_MIN_WORST_TRADES"] = trial.suggest_int(
        "PHASE4_MIN_WORST_TRADES", 10, 25,
    )

    # --- Objective deployment gate (validation only) ---
    params["PHASE5_VALIDATION_RETURN_GATE_PCT"] = trial.suggest_float(
        "PHASE5_VALIDATION_RETURN_GATE_PCT", 0.0, 5.0,
    )

    return params
