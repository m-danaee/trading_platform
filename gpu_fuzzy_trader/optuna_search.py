#!/usr/bin/env python3
"""Validation-only Optuna search for the active Phase 2/RB pipeline.

The test/OOS split is deliberately absent from this module.  Phase 5 remains
the final evaluator, but its metrics can never select hyperparameters.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import traceback
from pathlib import Path
from typing import Any
from unittest.mock import patch

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from gpu_fuzzy_trader import config as _cfg


# Independent, live configuration parameters.  Stage B totals are
# derived in sample_trial_params() so every trial satisfies validate_config().
SEARCH_SPACE: dict[str, list[Any]] = {
    "PHASE1_TOP_K_FEATURES": [10, 15, 20, 25, 30],
    "MIN_TRADE_SUPPORT": [30, 45, 60, 90, 120],
    "MIN_TRADE_POOL_FLOOR": [8, 12, 15, 20, 25],
    "PHASE2_MAX_DRAWDOWN_GATE": [12.0, 18.0, 25.0, 30.0],
    "PHASE2_MAX_TRAIN_VAL_GAP_PCT": [4.0, 6.0, 8.0, 10.0, 15.0],
    "PHASE2_KEEP_TOP_RULES": [80, 120, 150, 200],
    "PHASE2_POPULATION_SIZE": [100, 150, 200, 250],
    "PHASE2_GENERATIONS": [20, 30, 40, 50, 60],
    "PHASE2_MUTATION_RATE": [0.16, 0.22, 0.28, 0.32, 0.38],
    "PHASE2_STAGE_A_MUTATION_RATE": [0.25, 0.30, 0.35, 0.40],
    "PHASE2_RETURN_FLOOR_PCT": [0.0, 0.25, 0.5, 1.0, 2.0],
    "PHASE2_VAL_RETURN_FLOOR_PCT": [0.0, 0.25, 0.5, 1.0, 2.0],
    "PHASE2_PROFIT_FACTOR_FLOOR_ADMISSION": [1.05, 1.10, 1.15, 1.20],
    "PHASE2_MONTHLY_ADMISSION_MIN_RATIO": [0.3, 0.4, 0.5, 0.6],
    "RB_MIN_TRAIN_RETURN": [0.25, 0.5, 1.0, 2.0],
    "RB_MIN_VALID_RETURN": [0.25, 0.5, 1.0, 2.0],
    "RB_MIN_TRAIN_PF": [1.0, 1.02, 1.05, 1.10],
    "RB_MIN_VALID_PF": [1.0, 1.02, 1.05, 1.10],
    "RB_KEEP_TOP_RULES": [80, 120, 150, 200],
    "RB_MAX_PAIR_OVERLAP": [0.20, 0.25, 0.30, 0.35, 0.40],
    "RB_MIN_SCORE_IMPROVEMENT": [0.005, 0.01, 0.02, 0.05],
    "RB_RISK_MIN_IMPROVEMENT": [0.01, 0.02, 0.05, 0.10],
}

_fast_mode: bool = False
_debug_mode: bool = False


def _active_search_space() -> dict[str, list[Any]]:
    """Return the search space after removing frozen fast-mode parameters."""

    if not _fast_mode:
        return dict(SEARCH_SPACE)
    return {
        key: values
        for key, values in SEARCH_SPACE.items()
        if not key.startswith("PHASE2_")
    }


def sample_trial_params(trial: optuna.Trial) -> dict[str, Any]:
    """Sample live settings and derive a coherent two-stage budget."""

    params: dict[str, Any] = {
        name: trial.suggest_categorical(name, values)
        for name, values in _active_search_space().items()
    }
    if not _fast_mode:
        total = int(params["PHASE2_GENERATIONS"])
        min_stage_a = max(
            1,
            int(_cfg.PHASE2_STAGE_A_PLATEAU_EARLY_STOP_MIN_GENERATION),
            int(_cfg.PHASE2_STAGE_A_EARLY_STOP_MIN_GENERATION),
        )
        min_stage_b = max(
            1,
            int(_cfg.PHASE2_STAGE_B_PLATEAU_EARLY_STOP_MIN_GENERATION),
            int(_cfg.PHASE2_STAGE_B_EARLY_STOP_MIN_GENERATION),
        )
        if total < min_stage_a + min_stage_b:
            raise _cfg.ConfigError(
                "PHASE2_GENERATIONS is too small for the configured stage early-stop geometry"
            )
        stage_a = trial.suggest_int(
            "PHASE2_STAGE_A_GENERATIONS",
            min_stage_a,
            total - min_stage_b,
        )
        params["PHASE2_STAGE_A_GENERATIONS"] = stage_a
        params["PHASE2_STAGE_B_GENERATIONS"] = total - stage_a
        # Stage A is the exploration stage; derive a non-increasing Stage B

        # rate so independent sampling can never violate that contract.
        params["PHASE2_STAGE_B_MUTATION_RATE"] = min(
            0.20,
            float(params["PHASE2_STAGE_A_MUTATION_RATE"]),
        )
    return params


def apply_trial_config(trial_params: dict[str, Any]) -> list:
    """Patch config globals and validate the complete trial contract."""

    patchers: list = []
    try:
        for key, value in trial_params.items():
            if not hasattr(_cfg, key):
                raise _cfg.ConfigError(f"Optuna parameter is not live: {key}")
            patcher = patch(f"gpu_fuzzy_trader.config.{key}", value)
            patcher.start()
            patchers.append(patcher)
        _cfg.validate_config()
    except Exception:
        for patcher in reversed(patchers):
            patcher.stop()
        raise
    return patchers


def _read_rb_report(output_dir: str, direction: str) -> dict[str, Any]:
    path = Path(output_dir) / "reports" / f"rb_governor_{direction}_report.json"
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {"fail_closed": True, "reason": "missing_rb_report"}
    return data if isinstance(data, dict) else {"fail_closed": True}


def collect_validation_metrics(
    results: dict[str, Any],
    output_dir: str,
) -> dict[str, float]:
    """Collect train/validation/tail metrics only from RB reports."""

    metrics: dict[str, float] = {}
    for direction in ("long", "short"):
        report = _read_rb_report(output_dir, direction)
        valid = report.get("valid_metrics") or {}
        risk_history = report.get("risk_history") or []
        tail_return = 0.0
        if risk_history and isinstance(risk_history[-1], dict):
            tail_return = float(
                risk_history[-1].get("risk_tail_holdout_return_pct", 0.0)
            )
        metrics[f"valid_{direction}_return_pct"] = float(
            valid.get("total_return_pct", -1000.0)
        )
        metrics[f"valid_{direction}_dd_pct"] = float(
            valid.get("max_drawdown_pct", 1000.0)
        )
        metrics[f"valid_{direction}_pf"] = float(
            valid.get("profit_factor", 0.0)
        )
        metrics[f"tail_{direction}_return_pct"] = tail_return
        metrics[f"fail_closed_{direction}"] = float(
            bool(report.get("fail_closed"))
            or (
                "deployment_accepted" in report
                and not bool(report.get("deployment_accepted"))
            )
        )
    return metrics


def compute_score(metrics: dict[str, float]) -> float:
    """Score balanced validation/tail robustness without OOS leakage."""

    if any(
        metrics.get(f"fail_closed_{direction}", 1.0) > 0.0
        for direction in ("long", "short")
    ):
        return -1_000_000.0

    direction_scores: list[float] = []
    returns: list[float] = []
    for direction in ("long", "short"):
        valid_return = metrics.get(f"valid_{direction}_return_pct", -1000.0)
        tail_return = metrics.get(f"tail_{direction}_return_pct", valid_return)
        valid_dd = metrics.get(f"valid_{direction}_dd_pct", 1000.0)
        valid_pf = metrics.get(f"valid_{direction}_pf", 0.0)
        returns.append(valid_return)
        direction_scores.append(
            0.45 * valid_return
            + 0.35 * tail_return
            + 0.20 * min(valid_pf, 3.0)
            - 0.10 * max(valid_dd, 0.0)
        )
    if not direction_scores:
        return -1_000_000.0
    balance_penalty = 0.05 * abs(returns[0] - returns[1]) if len(returns) == 2 else 0.0
    return float(sum(direction_scores) / len(direction_scores) - balance_penalty)


_FAST_MODE_COPY_FILES: tuple[str, ...] = (
    "selected_features_long.json",
    "selected_features_short.json",
    "phase2_long_pool.json",
    "phase2_short_pool.json",
    "phase2_long_history.json",
    "phase2_short_history.json",
)



def _copy_phase1_2_outputs(src_dir: str, dst_dir: str) -> None:
    for filename in _FAST_MODE_COPY_FILES:
        source = os.path.join(src_dir, filename)
        target = os.path.join(dst_dir, filename)
        if os.path.isfile(source) and not os.path.exists(target):
            shutil.copy2(source, target)


def run_pipeline_for_trial(
    output_dir: str,
    fast_mode: bool,
    debug_mode: bool,
    baseline_outputs_dir: str = "outputs",
) -> dict:
    """Run one isolated trial; fast mode only reuses Phase 1/2 artifacts."""

    os.makedirs(output_dir, exist_ok=True)
    if debug_mode:
        _cfg.DEBUG_SYMBOL_SCOPE_ENABLED = True
    if fast_mode:
        _copy_phase1_2_outputs(baseline_outputs_dir, output_dir)

    from gpu_fuzzy_trader.run_pipeline import Pipeline_Orchestrator

    return Pipeline_Orchestrator(output_dir=output_dir).run(force=not fast_mode)


def _write_trial_correlation_report(study: optuna.Study, path: str) -> None:
    """Persist Spearman parameter/score correlations from validation trials."""

    completed = [
        trial for trial in study.trials
        if trial.value is not None and trial.state == optuna.trial.TrialState.COMPLETE
    ]
    report: dict[str, Any] = {"n_trials": len(completed), "score_correlations": {}}
    if len(completed) >= 3:
        scores = [float(trial.value) for trial in completed]
        all_names = {
            key
            for trial in completed
            for key in (
                set(trial.params)
                | set(trial.user_attrs.get("derived_trial_params", {}))
            )
        }
        for name in sorted(all_names):
            pairs = [
                (
                    trial.params.get(
                        name,
                        trial.user_attrs.get("derived_trial_params", {}).get(name),
                    ),
                    float(trial.value),
                )
                for trial in completed
                if name in trial.params
                or name in trial.user_attrs.get("derived_trial_params", {})
            ]
            if len(pairs) < 3:
                continue
            categories = sorted(
                {str(value) for value, _ in pairs}
            )
            category_codes = {
                value: float(index)
                for index, value in enumerate(categories)
            }
            param_values = [
                float(value)
                if isinstance(value, (int, float)) and not isinstance(value, bool)
                else category_codes[str(value)]
                for value, _ in pairs
            ]
            score_values = [score for _, score in pairs]
            param_ranks = _rank(param_values)
            score_ranks = _rank(score_values)
            report["score_correlations"][name] = _pearson(param_ranks, score_ranks)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    for rank, index in enumerate(order):
        ranks[index] = float(rank)
    return ranks


def _pearson(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    numerator = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    denom_a = sum((x - mean_a) ** 2 for x in a) ** 0.5
    denom_b = sum((y - mean_b) ** 2 for y in b) ** 0.5
    return float(numerator / (denom_a * denom_b)) if denom_a and denom_b else 0.0


def objective(trial: optuna.Trial) -> float:
    """Maximize robust validation/tail score in an isolated trial."""

    try:
        trial_params = sample_trial_params(trial)
    except _cfg.ConfigError as exc:
        trial.set_user_attr("rejection_reason", str(exc))
        return -1_000_000.0
    trial.set_user_attr("trial_params", trial_params)
    trial.set_user_attr(
        "derived_trial_params",
        {
            key: value
            for key, value in trial_params.items()
            if key not in trial.params
        },
    )
    patchers: list = []
    trial_output_dir = os.path.join("outputs", f"trial_{trial.number}")
    try:
        patchers = apply_trial_config(trial_params)
        effective = _cfg.effective_config_snapshot()
        trial.set_user_attr("derived_effective_values", effective)
        results = run_pipeline_for_trial(
            trial_output_dir,
            fast_mode=_fast_mode,
            debug_mode=_debug_mode,
            baseline_outputs_dir="outputs",
        )
        metrics = collect_validation_metrics(results, trial_output_dir)
        score = compute_score(metrics)
        for key, value in metrics.items():
            trial.set_user_attr(key, value)
        trial.set_user_attr("score_source", "rb_validation_and_tail")
        trial.set_user_attr("effective_config", effective)
        trial.set_user_attr("score", score)
        failed = [
            direction
            for direction in ("long", "short")
            if metrics.get(f"fail_closed_{direction}", 1.0) > 0.0
        ]
        if failed:
            trial.set_user_attr(
                "rejection_reason",
                "rb_fail_closed:" + ",".join(failed),
            )
        return score
    except _cfg.ConfigError as exc:
        trial.set_user_attr("rejection_reason", str(exc))
        trial.set_user_attr("error", traceback.format_exc())
        return -1_000_000.0
    except Exception as exc:
        trial.set_user_attr("rejection_reason", f"execution_error:{exc}")
        trial.set_user_attr("error", traceback.format_exc())
        return -1_000_000.0
    finally:
        for patcher in reversed(patchers):
            patcher.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only RB Optuna search")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--study-name", default="gpu_fuzzy_optuna")
    parser.add_argument("--storage", default="sqlite:///outputs/optuna_study.db")
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)
    global _fast_mode, _debug_mode
    _fast_mode = bool(args.fast)
    _debug_mode = bool(args.debug)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        load_if_exists=True,
    )
    start = time.monotonic()
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
    print(f"Search completed in {time.monotonic() - start:.1f}s")
    print(f"Best score: {study.best_value:.4f}")
    with open("outputs/optuna_best_params.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "best_trial": study.best_trial.number,
                "best_score": study.best_value,
                "best_params": study.best_params,
                "user_attrs": study.best_trial.user_attrs,
            },
            handle,
            indent=2,
        )
    _write_trial_correlation_report(
        study,
        "outputs/reports/hyperparameter_correlation.json",
    )


if __name__ == "__main__":
    main()
