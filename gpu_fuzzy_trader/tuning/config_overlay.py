"""
Temporarily override gpu_fuzzy_trader.config module attributes for one trial.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from gpu_fuzzy_trader import config as _cfg

# Config keys the tuner is allowed to patch (must exist on the config module).
_TUNABLE_NAMES = frozenset({
    "GLOBAL_SEED",
    "PHASE2_SEED",
    "PHASE4_SEED",
    "CV_N_FOLDS",
    "PHASE1_SAMPLING_TOTAL",
    "PHASE2_POPULATION_SIZE",
    "PHASE2_GENERATIONS",
    "PHASE2_CV_POOL_MIN_FOLDS_PASS",
    "PHASE2_CV_PROFIT_FACTOR_FLOOR",
    "MIN_TRADE_SUPPORT",
    "PHASE2_JOINT_TRAIN_VAL",
    "PHASE3_VAL_SORTINO_RATIO_GATE",
    "PHASE3_VAL_TRAIN_GAP_MAX_PCT",
    "PHASE3_TRAIN_VAL_GAP_MAX_PCT",
    "PHASE3_VAL_GATE_PENALTY",
    "PHASE2_EARLY_STOP_MIN_GENERATION",
    "PHASE2_EARLY_STOP_MEAN_RETURN_PCT",
    "PHASE3_REFINE_POP_SIZE",
    "PHASE3_REFINE_GENERATIONS",
    "PHASE4_N_TRIALS",
    "PHASE3_USE_GPU",
    "PHASE2_CV_FOLD_WORKERS",
    "PHASE3_BATCH_WORKERS",
    "PHASE4_N_JOBS",
    "PHASE2_NUMBA_ENABLED",
})


def _validate_params(params: dict[str, Any]) -> None:
    unknown = set(params) - _TUNABLE_NAMES
    if unknown:
        raise ValueError(
            f"Unsupported config overrides: {sorted(unknown)}"
        )
    for name in params:
        if not hasattr(_cfg, name):
            raise AttributeError(f"config has no attribute {name!r}")


@contextmanager
def apply_trial_params(
    params: dict[str, Any],
    *,
    trial_seed: int | None = None,
) -> Iterator[None]:
    """
    Monkeypatch config attributes for the duration of one pipeline trial.

    Parameters
    ----------
    params : dict
        Config attribute names → values (see ``_TUNABLE_NAMES``).
    trial_seed : int, optional
        If set, also assigns ``GLOBAL_SEED``, ``PHASE2_SEED``, and
        ``PHASE4_SEED`` so evolution and Phase 4 Optuna are reproducible.
    """
    merged = dict(params)
    if trial_seed is not None:
        merged["GLOBAL_SEED"] = trial_seed
        merged["PHASE2_SEED"] = trial_seed
        merged["PHASE4_SEED"] = trial_seed

    _validate_params(merged)

    saved: dict[str, Any] = {}
    for name, value in merged.items():
        saved[name] = getattr(_cfg, name)
        setattr(_cfg, name, value)

    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(_cfg, name, value)
