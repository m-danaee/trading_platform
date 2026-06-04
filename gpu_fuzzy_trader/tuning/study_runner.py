"""
Optuna study driver: Phases 2–5 per trial with cached Phase 1 features.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

import optuna

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.run_pipeline import Pipeline_Orchestrator
from gpu_fuzzy_trader.tuning._bootstrap import configure_tuning_cpu_env
from gpu_fuzzy_trader.tuning.config_overlay import apply_trial_params
from gpu_fuzzy_trader.tuning.config_space import get_profile_params, suggest_trial_params
from gpu_fuzzy_trader.tuning.objective import compute_validation_objective

logger = logging.getLogger(__name__)

_PHASE1_ARTIFACTS = (
    "selected_features_long.json",
    "selected_features_short.json",
    "phase1_regime_cluster.joblib",
)


def copy_phase1_artifacts(baseline_output_dir: str, trial_output_dir: str) -> None:
    """Copy Phase 1 JSON (and regime model if present) into a trial output tree."""
    os.makedirs(trial_output_dir, exist_ok=True)
    baseline = os.path.abspath(baseline_output_dir)

    for name in _PHASE1_ARTIFACTS:
        src = os.path.join(baseline, name)
        if not os.path.isfile(src):
            if name.endswith(".json"):
                raise FileNotFoundError(
                    f"Missing baseline Phase 1 artifact: {src}. "
                    "Run: python -m gpu_fuzzy_trader.run_pipeline --phase 1"
                )
            continue
        shutil.copy2(src, os.path.join(trial_output_dir, name))


def validate_baseline_prerequisites(baseline_output_dir: str) -> None:
    """Ensure baseline has Phase 1 feature files."""
    baseline = os.path.abspath(baseline_output_dir)
    for name in ("selected_features_long.json", "selected_features_short.json"):
        path = os.path.join(baseline, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Baseline output missing {path}. "
                "Run Phase 1 first: python -m gpu_fuzzy_trader.run_pipeline "
                f"--output {baseline} --phase 1"
            )


def build_merged_trial_config(
    trial_params: dict[str, Any],
    profile: str,
) -> dict[str, Any]:
    """Profile fixed caps plus Optuna-suggested knobs (trial overlay dict)."""
    merged = get_profile_params(profile)
    merged.update(trial_params)
    return merged


def _log_tuning_runtime(*, force_cpu: bool, profile: str) -> None:
    """Log JAX platform and Phase 2/3 engine flags at study start."""
    profile_caps = get_profile_params(profile)
    p2_gpu = profile_caps.get("PHASE2_USE_GPU", _cfg.PHASE2_USE_GPU)
    p3_gpu = profile_caps.get("PHASE3_USE_GPU", _cfg.PHASE3_USE_GPU)
    logger.info(
        "Tuning runtime: JAX_PLATFORMS=%s force_cpu=%s profile=%s | "
        "trial PHASE2_USE_GPU=%s PHASE3_USE_GPU=%s",
        os.environ.get("JAX_PLATFORMS", "(unset)"),
        force_cpu,
        profile,
        p2_gpu,
        p3_gpu,
    )


def export_best_config(
    study: optuna.Study,
    path: str,
    *,
    profile: str = "low_ram",
) -> None:
    """Write winning trial params to JSON for pasting into config.py."""
    if study.best_trial is None:
        raise RuntimeError("Study has no completed trials")

    merged = build_merged_trial_config(
        dict(study.best_trial.params), profile)

    payload = {
        "trial_number": study.best_trial.number,
        "value": study.best_value,
        "params": dict(study.best_trial.params),
        "merged_config": merged,
        "user_attrs": dict(study.best_trial.user_attrs),
        "note": (
            "Paste generalization knobs from params or merged_config into "
            "config.py. On Colab GPU runs keep PHASE2_USE_GPU=True (default) "
            "and omit tuning-only CPU caps unless you want a slow run."
        ),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def write_trials_summary_csv(study: optuna.Study, path: str) -> None:
    """Export all trials to CSV for offline analysis."""
    fieldnames = [
        "number",
        "value",
        "state",
        "duration_sec",
        "val_return_long",
        "val_return_short",
        "test_return_long",
        "test_return_short",
        "phase2_pool_long",
        "phase2_pool_short",
    ]
    param_names = sorted(
        {k for t in study.trials for k in t.params}
    )
    fieldnames.extend(param_names)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for trial in study.trials:
            row: dict[str, Any] = {
                "number": trial.number,
                "value": trial.value,
                "state": trial.state.name,
                "duration_sec": trial.user_attrs.get("duration_sec"),
                "val_return_long": trial.user_attrs.get("val_return_long"),
                "val_return_short": trial.user_attrs.get("val_return_short"),
                "test_return_long": trial.user_attrs.get("test_return_long"),
                "test_return_short": trial.user_attrs.get("test_return_short"),
                "phase2_pool_long": trial.user_attrs.get("phase2_pool_long"),
                "phase2_pool_short": trial.user_attrs.get("phase2_pool_short"),
            }
            row.update(trial.params)
            writer.writerow(row)


def run_study(
    *,
    baseline_output_dir: str,
    study_dir: str,
    n_trials: int = 20,
    profile: str = "low_ram",
    seed: int = 42,
    storage_url: str | None = None,
    study_name: str = "config_tuner",
    force_cpu: bool = True,
) -> optuna.Study:
    """
    Run an Optuna study over pipeline config hyperparameters.

    Each trial executes Phases 2–5 in an isolated output directory under
    ``study_dir/trial_N``.
    """
    configure_tuning_cpu_env(force=force_cpu)
    _log_tuning_runtime(force_cpu=force_cpu, profile=profile)

    validate_baseline_prerequisites(baseline_output_dir)
    study_path = Path(study_dir)
    study_path.mkdir(parents=True, exist_ok=True)

    if storage_url is None:
        storage_url = f"sqlite:///{study_path / 'optuna.db'}"

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
    )

    def objective(trial: optuna.Trial) -> float:
        trial_seed = seed + trial.number
        params = suggest_trial_params(trial, profile)
        trial_dir = str(study_path / f"trial_{trial.number}")
        copy_phase1_artifacts(baseline_output_dir, trial_dir)

        t0 = time.monotonic()
        with apply_trial_params(params, trial_seed=trial_seed):
            orchestrator = Pipeline_Orchestrator(output_dir=trial_dir)
            try:
                results = orchestrator.run_from_phase2(force=True)
            except Exception as exc:
                logger.error(
                    "Trial %d failed: %s", trial.number, exc, exc_info=True)
                raise

        phase5 = results.get("phase5", {})
        phase2 = results.get("phase2", {})
        pool_long = len(phase2.get("long", []))
        pool_short = len(phase2.get("short", []))
        score, details = compute_validation_objective(
            phase5,
            phase2_pool_long=pool_long,
            phase2_pool_short=pool_short,
        )

        elapsed = time.monotonic() - t0
        trial.set_user_attr("duration_sec", round(elapsed, 2))
        trial.set_user_attr("val_return_long", details.get("val_return_long"))
        trial.set_user_attr("val_return_short",
                            details.get("val_return_short"))
        trial.set_user_attr("test_return_long",
                            details.get("test_return_long"))
        trial.set_user_attr("test_return_short",
                            details.get("test_return_short"))
        trial.set_user_attr("phase2_pool_long", pool_long)
        trial.set_user_attr("phase2_pool_short", pool_short)
        trial.set_user_attr("pool_penalty", details.get("pool_penalty"))
        trial.set_user_attr("score_details", json.dumps(details))

        logger.info(
            "Trial %d done: score=%.4f val_long=%.2f%% val_short=%.2f%% "
            "test_long=%.2f%% test_short=%.2f%% (%.1fs)",
            trial.number,
            score,
            details.get("val_return_long", 0.0),
            details.get("val_return_short", 0.0),
            details.get("test_return_long", 0.0),
            details.get("test_return_short", 0.0),
            elapsed,
        )
        return score

    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    export_best_config(
        study, str(study_path / "best_config.json"), profile=profile)
    write_trials_summary_csv(study, str(study_path / "trials_summary.csv"))

    logger.info(
        "Study complete: best trial #%s value=%.4f",
        study.best_trial.number if study.best_trial else "n/a",
        study.best_value if study.best_value is not None else float("nan"),
    )
    return study


def main(argv: list[str] | None = None) -> None:
    """CLI entry for config tuning."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        prog="python -m gpu_fuzzy_trader.tuning",
        description=(
            "Optuna tuner for gpu_fuzzy_trader.config (Phases 2–5, "
            "cached Phase 1)."
        ),
    )
    parser.add_argument(
        "--baseline-output",
        default=_cfg.OUTPUTS_DIR,
        help="Directory with Phase 1 selected_features_*.json (default: outputs/).",
    )
    parser.add_argument(
        "--study-dir",
        default="tuning_studies/low_ram",
        help="Directory for optuna.db, trial outputs, and exports.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of Optuna trials (default: 20).",
    )
    parser.add_argument(
        "--profile",
        default="low_ram",
        choices=("low_ram",),
        help="Resource profile with fixed budget caps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (each trial uses seed + trial number).",
    )
    parser.add_argument(
        "--force-cpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Pin JAX to CPU and use low_ram CPU engine caps (default: on). "
            "Use --no-force-cpu to allow GPU JAX if installed."
        ),
    )

    args = parser.parse_args(argv)
    study = run_study(
        baseline_output_dir=args.baseline_output,
        study_dir=args.study_dir,
        n_trials=args.n_trials,
        profile=args.profile,
        seed=args.seed,
        force_cpu=args.force_cpu,
    )

    print("\n=== Tuning complete ===")
    if study.best_trial is not None:
        print(f"  Best trial: #{study.best_trial.number}")
        print(f"  Score: {study.best_value:.4f}")
        print(f"  Params: {study.best_trial.params}")
    print(f"  Study dir: {os.path.abspath(args.study_dir)}")
    print(f"  best_config.json and trials_summary.csv written there.")
