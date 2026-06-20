#!/usr/bin/env python3
"""Optuna hyperparameter search for GPU Fuzzy Trader config.

Usage:
    python -m gpu_fuzzy_trader.optuna_search --n-trials 50
    python -m gpu_fuzzy_trader.optuna_search --fast --n-trials 30
    python -m gpu_fuzzy_trader.optuna_search --debug --n-trials 10
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from typing import Any

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Search space: 12 hyperparameters from config.py
# Each maps to a list of categorical candidate values.
# ---------------------------------------------------------------------------

SEARCH_SPACE: dict[str, list[Any]] = {
    "PHASE1_TOP_K_FEATURES": [10, 15, 20, 25, 30],
    "MIN_TRADE_SUPPORT": [30, 60, 90, 120, 150],
    "MIN_TRADE_POOL_FLOOR": [8, 12, 17, 25, 35],
    "PHASE2_MAX_DRAWDOWN_GATE": [12.0, 18.0, 25.0, 30.0, 35.0],
    "PHASE2_MAX_TRAIN_VAL_GAP_PCT": [4.0, 6.0, 8.0, 10.0, 15.0],
    "PHASE2_KEEP_TOP_RULES": [40, 60, 80, 100, 120],
    "PHASE2_POPULATION_SIZE": [100, 150, 200, 250],
    "PHASE2_MUTATION_RATE": [0.12, 0.17, 0.22, 0.27, 0.32],
    "PHASE2_STAGE_A_GENERATIONS": [50, 70, 85, 100, 120],
    "PHASE3_VAL_RETURN_FLOOR_PCT": [2.0, 4.0, 5.0, 7.0, 10.0],
    "RB_MIN_TRAIN_RETURN": [1.0, 2.0, 3.0, 5.0],
    "RB_MIN_VALID_RETURN": [1.0, 2.0, 3.0, 5.0],
}

# ---------------------------------------------------------------------------
# Module-level flags (set by CLI, read by objective())
# ---------------------------------------------------------------------------

_fast_mode: bool = False
_debug_mode: bool = False


# ---------------------------------------------------------------------------
# Config patching helpers
# ---------------------------------------------------------------------------


def apply_trial_config(trial_params: dict[str, Any]) -> list:
    """Patch config module globals with trial hyperparameters.

    Uses ``unittest.mock.patch`` to temporarily override attribute values
    on ``gpu_fuzzy_trader.config``.  Returns a list of patcher objects
    that must be stopped after the trial (via ``p.stop()``).
    """
    patchers: list = []
    for key, value in trial_params.items():
        patchers.append(patch(f"gpu_fuzzy_trader.config.{key}", value))
    for p in patchers:
        p.start()
    return patchers


# ---------------------------------------------------------------------------
# Metrics collection
# ---------------------------------------------------------------------------


def collect_phase5_metrics(results: dict) -> dict[str, float]:
    """Extract test-split metrics from the pipeline's Phase 5 results.

    Parameters
    ----------
    results : dict
        The full results dict returned by ``Pipeline_Orchestrator.run()``.

    Returns
    -------
    dict[str, float]
        Flattened metrics keyed by ``test_long_return_pct``,
        ``test_long_dd_pct``, ``test_short_return_pct``, etc.
    """
    phase5 = results.get("phase5", {})
    metrics: dict[str, float] = {}

    for direction in ("long", "short"):
        dir_metrics = phase5.get(direction, {})

        # Phase 5 may nest metrics under a "test" key
        if "test" in dir_metrics:
            dir_metrics = dir_metrics["test"]

        metrics[f"test_{direction}_return_pct"] = float(
            dir_metrics.get("total_return_pct", -999)
        )
        metrics[f"test_{direction}_dd_pct"] = float(
            dir_metrics.get("max_drawdown_pct", 100)
        )
        metrics[f"test_{direction}_pf"] = float(
            dir_metrics.get("profit_factor", 0)
        )

    return metrics


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------


def compute_score(metrics: dict[str, float]) -> float:
    """Compute composite objective score from test return and drawdown.

    Formula
    -------
    combined_return = (long_return + short_return) / 2.0
    max_dd = max(long_dd, short_dd)
    dd_penalty = DD_PENALTY_WEIGHT * max(0, max_dd - DD_THRESHOLD)
    score = combined_return - dd_penalty
    """
    long_return = metrics.get("test_long_return_pct", -999.0)
    short_return = metrics.get("test_short_return_pct", -999.0)
    long_dd = metrics.get("test_long_dd_pct", 100.0)
    short_dd = metrics.get("test_short_dd_pct", 100.0)

    combined_return = (long_return + short_return) / 2.0
    max_dd = max(long_dd, short_dd)

    DD_THRESHOLD = 8.0
    DD_PENALTY_WEIGHT = 3.0

    dd_penalty = DD_PENALTY_WEIGHT * max(0.0, max_dd - DD_THRESHOLD)
    score = combined_return - dd_penalty
    return score


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------


def run_pipeline_for_trial(
    output_dir: str,
    fast_mode: bool,
    debug_mode: bool,
) -> dict:
    """Run the pipeline with currently patched config values.

    Parameters
    ----------
    output_dir : str
        Base output directory for pipeline artifacts.
    fast_mode : bool
        If True, pass ``force=False`` (resume mode — skip valid cached phases).
    debug_mode : bool
        If True, enable ``DEBUG_SYMBOL_SCOPE`` with 4 symbols.

    Returns
    -------
    dict
        Full results dict from ``Pipeline_Orchestrator.run()``.
    """
    # Apply debug scope *before* creating the orchestrator so that the
    # pipeline picks up the narrowed symbol universe from the start.
    if debug_mode:
        import gpu_fuzzy_trader.config as cfg

        cfg.DEBUG_SYMBOL_SCOPE_ENABLED = True
        cfg.DEBUG_SYMBOL_COUNT = 4

    # Import pipeline modules *after* config has been patched so that
    # module-level references to ``_cfg.XXX`` are still dynamic lookups.
    from gpu_fuzzy_trader.run_pipeline import Pipeline_Orchestrator

    orchestrator = Pipeline_Orchestrator(output_dir=output_dir)

    # ``force=True``  → rerun all phases (default).
    # ``force=False`` → resume mode, skip phases with valid on-disk outputs.
    force = not fast_mode

    return orchestrator.run(force=force)


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------


def objective(trial: optuna.Trial) -> float:
    """Optuna objective: maximize test return with drawdown penalty.

    For each trial:
      1. Sample candidate hyperparameters from ``SEARCH_SPACE``.
      2. Patch ``gpu_fuzzy_trader.config`` with the trial values.
      3. Run the full pipeline.
      4. Compute a composite score from test returns and drawdowns.
      5. Restore original config values.
      6. Return the score.

    Failed trials return ``-999.0`` (heavy penalty) rather than crashing.
    """
    # --- 1. Sample hyperparameters ---
    trial_params: dict[str, Any] = {}
    for param_name, values in SEARCH_SPACE.items():
        trial_params[param_name] = trial.suggest_categorical(param_name, values)

    # --- 2. Apply config patches ---
    patchers = apply_trial_config(trial_params)

    # Use a stable output directory so resume-mode can reuse cached phases.
    output_dir = "outputs"

    try:
        # --- 3. Run pipeline ---
        results = run_pipeline_for_trial(
            output_dir=output_dir,
            fast_mode=_fast_mode,
            debug_mode=_debug_mode,
        )

        # --- 4. Collect & score metrics ---
        metrics = collect_phase5_metrics(results)
        score = compute_score(metrics)

        # Store rich metrics as trial user attributes for later analysis.
        long_ret = metrics.get("test_long_return_pct", 0.0)
        short_ret = metrics.get("test_short_return_pct", 0.0)
        long_dd = metrics.get("test_long_dd_pct", 0.0)
        short_dd = metrics.get("test_short_dd_pct", 0.0)

        trial.set_user_attr("long_return", long_ret)
        trial.set_user_attr("short_return", short_ret)
        trial.set_user_attr("long_dd", long_dd)
        trial.set_user_attr("short_dd", short_dd)
        trial.set_user_attr("long_pf", metrics.get("test_long_pf", 0.0))
        trial.set_user_attr("short_pf", metrics.get("test_short_pf", 0.0))
        trial.set_user_attr("combined_return", (long_ret + short_ret) / 2.0)
        trial.set_user_attr("max_dd", max(long_dd, short_dd))
        trial.set_user_attr("score", score)

        return score

    except Exception:
        # --- 5. Handle failures gracefully ---
        trial.set_user_attr("error", traceback.format_exc())
        traceback.print_exc()
        return -999.0  # heavy penalty for failed trials

    finally:
        # --- 6. Always restore config to prevent cross-trial contamination ---
        for p in patchers:
            p.stop()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments, create/load an Optuna study, and run the search."""
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search for GPU Fuzzy Trader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --n-trials 50\n"
            "  %(prog)s --fast --n-trials 30\n"
            "  %(prog)s --debug --n-trials 10\n"
        ),
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials (default: 50)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Fast/resume mode: run pipeline with force=False so that "
            "phases with valid cached outputs are skipped.  Useful when "
            "iterating on a subset of hyperparameters."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Debug mode: enable DEBUG_SYMBOL_SCOPE with 4 symbols so "
            "the pipeline runs on a tiny data subset (fast but not "
            "representative)."
        ),
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="gpu_fuzzy_optuna",
        help="Optuna study name (default: gpu_fuzzy_optuna)",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///outputs/optuna_study.db",
        help=(
            "Storage URL for the Optuna study. "
            "(default: sqlite:///outputs/optuna_study.db)"
        ),
    )
    args = parser.parse_args()

    # Ensure output directories exist
    os.makedirs("outputs", exist_ok=True)

    # Set module-level flags for use in objective()
    global _fast_mode, _debug_mode
    _fast_mode = args.fast
    _debug_mode = args.debug

    # ------------------------------------------------------------------
    # Create / load Optuna study
    # ------------------------------------------------------------------
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    # ------------------------------------------------------------------
    # Run optimisation
    # ------------------------------------------------------------------
    print(f"Starting Optuna search: study={args.study_name!r}, "
          f"n_trials={args.n_trials}, "
          f"fast={args.fast}, debug={args.debug}")
    print(f"Storage: {args.storage}")
    print()

    t0 = time.monotonic()

    study.optimize(
        lambda trial: objective(trial),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    elapsed = time.monotonic() - t0
    print(f"\nSearch completed in {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Print & persist best results
    # ------------------------------------------------------------------
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best score: {study.best_value:.4f}")
    print("Best params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save best params to JSON
    best_path = "outputs/optuna_best_params.json"
    with open(best_path, "w") as f:
        json.dump(
            {
                "best_trial": study.best_trial.number,
                "best_score": study.best_value,
                "best_params": study.best_params,
                "user_attrs": study.best_trial.user_attrs,
            },
            f,
            indent=2,
        )
    print(f"\nBest params saved to {best_path}")

    # Print top 5 trials by score
    ranked = [t for t in study.trials if t.value is not None]
    ranked.sort(key=lambda t: t.value, reverse=True)
    print("\nTop 5 trials:")
    for trial in ranked[:5]:
        print(f"  Trial {trial.number}: score={trial.value:.4f}")


if __name__ == "__main__":
    main()
