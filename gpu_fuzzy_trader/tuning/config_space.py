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
    "PHASE2_GENERATIONS": 30,
    "PHASE3_REFINE_POP_SIZE": 48,
    "PHASE3_REFINE_GENERATIONS": 25,
    "PHASE1_SAMPLING_TOTAL": 150_000,
    "CV_N_FOLDS": 2,
    "PHASE4_N_TRIALS": 40,
    "PHASE3_USE_GPU": False,
    "PHASE2_CV_FOLD_WORKERS": 1,
    "PHASE3_BATCH_WORKERS": 2,
    "PHASE4_N_JOBS": 1,
    "PHASE2_NUMBA_ENABLED": True,
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

    cv_folds = int(params.get("CV_N_FOLDS", _cfg.CV_N_FOLDS))
    params["PHASE2_CV_POOL_MIN_FOLDS_PASS"] = trial.suggest_int(
        "PHASE2_CV_POOL_MIN_FOLDS_PASS", 1, cv_folds,
    )
    params["PHASE2_CV_PROFIT_FACTOR_FLOOR"] = trial.suggest_float(
        "PHASE2_CV_PROFIT_FACTOR_FLOOR", 0.9, 1.1,
    )
    params["MIN_TRADE_SUPPORT"] = trial.suggest_int(
        "MIN_TRADE_SUPPORT", 100, 250,
    )
    params["PHASE2_JOINT_TRAIN_VAL"] = trial.suggest_categorical(
        "PHASE2_JOINT_TRAIN_VAL", [True, False],
    )

    params["PHASE3_VAL_SORTINO_RATIO_GATE"] = trial.suggest_float(
        "PHASE3_VAL_SORTINO_RATIO_GATE", 0.35, 0.65,
    )
    params["PHASE3_VAL_TRAIN_GAP_MAX_PCT"] = trial.suggest_float(
        "PHASE3_VAL_TRAIN_GAP_MAX_PCT", 5.0, 20.0,
    )
    params["PHASE3_TRAIN_VAL_GAP_MAX_PCT"] = trial.suggest_float(
        "PHASE3_TRAIN_VAL_GAP_MAX_PCT", 8.0, 25.0,
    )
    params["PHASE3_VAL_GATE_PENALTY"] = trial.suggest_float(
        "PHASE3_VAL_GATE_PENALTY", 5.0, 15.0,
    )

    params["PHASE2_EARLY_STOP_MIN_GENERATION"] = trial.suggest_int(
        "PHASE2_EARLY_STOP_MIN_GENERATION", 15, 40,
    )
    params["PHASE2_EARLY_STOP_MEAN_RETURN_PCT"] = trial.suggest_float(
        "PHASE2_EARLY_STOP_MEAN_RETURN_PCT", -8.0, -3.0,
    )

    params["PHASE2_GENERATIONS"] = trial.suggest_int(
        "PHASE2_GENERATIONS", 20, 40,
    )
    params["PHASE2_POPULATION_SIZE"] = trial.suggest_int(
        "PHASE2_POPULATION_SIZE", 80, 150,
    )

    return params
