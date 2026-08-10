"""
run_pipeline.py — Pipeline_Orchestrator

Top-level orchestrator for the GPU-Fuzzy Trading Pipeline.

Execution order:
  1. Create output directories (outputs/, outputs/reports/)
  2. Load and prepare data (Data_Loader + Data_Splitter)
  3. Phase 1: Feature_Selector
  4. Phase 2: Rule_Pool_Generator for both directions
  5. RB Governor: unified selection + risk tuning
  6. Phase 5: OOS_Evaluator (always runs)

Default CLI (``python -m gpu_fuzzy_trader.run_pipeline``) always re-runs Phase 1,
Phase 2, and the RB Governor.
Pass ``--resume`` to skip phases whose on-disk outputs are already valid.

Logging:
  - Log start time, end time, and elapsed duration for each phase.
  - Save structured log to outputs/pipeline.log as JSON lines.

Entry point:
  python -m gpu_fuzzy_trader.run_pipeline

Requirements: 13.1, 13.2, 13.3, 13.4, 13.5
"""

from __future__ import annotations
from gpu_fuzzy_trader.reporting import reporter as _reporter_module
from gpu_fuzzy_trader.phases import phase5_oos as _phase5_module
from gpu_fuzzy_trader.phases import phase2_rule_pool as _phase2_module
from gpu_fuzzy_trader import rb_governor as _rb_governor_module
from gpu_fuzzy_trader.features import selector as _selector_module
from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.data.loader import Data_Loader, validate_context_columns
from gpu_fuzzy_trader.data.splitter import Data_Splitter, load_cached_split_if_fresh
from gpu_fuzzy_trader.data.trend_context import sha256_file
from gpu_fuzzy_trader.features.selector import Feature_Selector
from gpu_fuzzy_trader.features.fuzzy_scaling import (
    apply_fuzzy_feature_scaling,
    fit_fuzzy_feature_scaling,
)
from gpu_fuzzy_trader.validation.rolling_cv import (
    build_forbidden_ranges,
    mask_df_to_safe_region,
)
from gpu_fuzzy_trader.phases.phase2_rule_pool import Rule_Pool_Generator
from gpu_fuzzy_trader.phases.phase2_rule_pool import (
    _derive_val_sample_seed,
    sample_df_for_phase2,
)
from gpu_fuzzy_trader.phases.phase2_island_scheduler import (
    compute_cluster_generation_budgets,
    _derive_island_seed,
)
from gpu_fuzzy_trader.context_diagnostics import (
    context_coverage_for_direction,
    context_coverage_report,
    context_floor_failures,
)
from gpu_fuzzy_trader.phases.phase5_oos import OOS_Evaluator
from gpu_fuzzy_trader.research_integrity import (
    ExperimentLedger,
    count_trials,
    write_dataset_manifests,
)
from gpu_fuzzy_trader.research_profile import ResearchProfile

import argparse
from contextlib import contextmanager
import pandas as pd
from typing import Any
from datetime import datetime, timezone
import uuid
import time
import sys
import os
import logging
import json
import numpy as np
from pathlib import Path

from gpu_fuzzy_trader._jax_env import configure_jax_env

configure_jax_env()


# ---------------------------------------------------------------------------
# Logging setup (must run before phase imports — EvoX adds a root handler that
# would otherwise make basicConfig a no-op and leave the root level at WARNING)
# ---------------------------------------------------------------------------

_LOG_LEVEL_NAME = os.environ.get("GPU_FUZZY_LOG_LEVEL", "INFO").upper()
_LOG_LEVEL = getattr(logging, _LOG_LEVEL_NAME, logging.INFO)

logging.basicConfig(
    level=_LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)


logging.getLogger().setLevel(_LOG_LEVEL)
logger = logging.getLogger(__name__)

_PIPELINE_LOG_PATH = os.path.join(_cfg.OUTPUTS_DIR, "pipeline.log")


def _resolve_output_root(output_dir: str | None) -> str:
    """Return the active output root for this run."""
    if output_dir is None:
        return _cfg.OUTPUTS_DIR
    return os.path.abspath(os.path.expanduser(output_dir))


@contextmanager
def _temporary_output_paths(output_dir: str | None):
    """Temporarily rebind all cached output paths for one pipeline run."""
    global _PIPELINE_LOG_PATH

    output_root = _resolve_output_root(output_dir)
    reports_root = os.path.join(output_root, "reports")

    previous_state = {
        "cfg_outputs": _cfg.OUTPUTS_DIR,
        "cfg_reports": _cfg.REPORTS_DIR,
        "cfg_run_log_path": _cfg.RUN_LOG_PATH,
        "cfg_phase2_pool_paths": _cfg.PHASE2_POOL_PATHS.copy(),
        "cfg_phase2_history_paths": _cfg.PHASE2_HISTORY_PATHS.copy(),
        "pipeline_log_path": _PIPELINE_LOG_PATH,
        "selector_long": _selector_module._LONG_PATH,
        "selector_short": _selector_module._SHORT_PATH,
        "selector_paths": _selector_module._DIRECTION_PATHS,
        "phase2_pool_paths": _phase2_module._POOL_PATHS.copy(),
        "phase2_history_paths": _phase2_module._HISTORY_PATHS.copy(),
        "phase5_strategy_paths": _phase5_module._STRATEGY_PATHS,
        "phase5_feature_paths": _phase5_module._FEATURE_PATHS,
        "phase5_report_paths": _phase5_module._REPORT_PATHS,
        "reporter_reports_dir": _reporter_module._REPORTS_DIR,
    }

    try:
        _cfg.OUTPUTS_DIR = output_root
        _cfg.REPORTS_DIR = reports_root
        _cfg.RUN_LOG_PATH = os.path.join(output_root, "run.log")
        _PIPELINE_LOG_PATH = os.path.join(output_root, "pipeline.log")

        # Phase 2 pool files now in run-specific output directory
        _cfg.PHASE2_POOL_PATHS = {
            "long": os.path.join(output_root, "phase2_long_pool.json"),
            "short": os.path.join(output_root, "phase2_short_pool.json"),
        }
        _cfg.PHASE2_HISTORY_PATHS = {
            "long": os.path.join(output_root, "phase2_long_history.json"),
            "short": os.path.join(output_root, "phase2_short_history.json"),
        }

        _selector_module._LONG_PATH = os.path.join(
            output_root, "selected_features_long.json"
        )
        _selector_module._SHORT_PATH = os.path.join(
            output_root, "selected_features_short.json"
        )
        _selector_module._DIRECTION_PATHS = {
            "long": _selector_module._LONG_PATH,
            "short": _selector_module._SHORT_PATH,
        }

        # Update phase2 module's cached paths
        _phase2_module._POOL_PATHS = _cfg.PHASE2_POOL_PATHS.copy()
        _phase2_module._HISTORY_PATHS = _cfg.PHASE2_HISTORY_PATHS.copy()

        _phase5_module._STRATEGY_PATHS = {
            "long": os.path.join(output_root, "long.json"),
            "short": os.path.join(output_root, "short.json"),
        }
        _phase5_module._FEATURE_PATHS = {
            "long": os.path.join(output_root, "selected_features_long.json"),
            "short": os.path.join(output_root, "selected_features_short.json"),
        }
        _phase5_module._REPORT_PATHS = {
            "long": os.path.join(reports_root, "test_long_report.json"),
            "short": os.path.join(reports_root, "test_short_report.json"),
            "per_symbol": os.path.join(
                reports_root, "test_per_symbol_performance.csv"
            ),
            "joint": os.path.join(
                reports_root, "test_joint_portfolio_report.json"
            ),
            "forward_long": os.path.join(
                reports_root, "forward_long_report.json"
            ),
            "forward_short": os.path.join(
                reports_root, "forward_short_report.json"
            ),
            "forward_joint": os.path.join(
                reports_root, "forward_joint_portfolio_report.json"
            ),
        }

        _reporter_module._REPORTS_DIR = reports_root

        yield output_root
    finally:
        _cfg.OUTPUTS_DIR = previous_state["cfg_outputs"]
        _cfg.REPORTS_DIR = previous_state["cfg_reports"]
        _cfg.RUN_LOG_PATH = previous_state["cfg_run_log_path"]
        _cfg.PHASE2_POOL_PATHS = previous_state["cfg_phase2_pool_paths"]
        _cfg.PHASE2_HISTORY_PATHS = previous_state["cfg_phase2_history_paths"]
        _PIPELINE_LOG_PATH = previous_state["pipeline_log_path"]
        _selector_module._LONG_PATH = previous_state["selector_long"]
        _selector_module._SHORT_PATH = previous_state["selector_short"]
        _selector_module._DIRECTION_PATHS = previous_state["selector_paths"]
        _phase2_module._POOL_PATHS = previous_state["phase2_pool_paths"]
        _phase2_module._HISTORY_PATHS = previous_state["phase2_history_paths"]
        _phase5_module._STRATEGY_PATHS = previous_state["phase5_strategy_paths"]
        _phase5_module._FEATURE_PATHS = previous_state["phase5_feature_paths"]
        _phase5_module._REPORT_PATHS = previous_state["phase5_report_paths"]
        _reporter_module._REPORTS_DIR = previous_state["reporter_reports_dir"]


# ---------------------------------------------------------------------------
# Phase timing helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


_CONTEXT_COVERAGE_SPLITS = (
    "train",
    "validation_fitness",
    "validation_selection",
)
_CONTEXT_PREFLIGHT_SPLITS = ("train", "validation_fitness")


def _context_coverage_for_direction(
    frame: pd.DataFrame,
    direction: str,
) -> dict[str, object]:
    """Return shared permission/trigger/conjunction coverage diagnostics."""
    return context_coverage_for_direction(frame, direction)


def _context_coverage_report(
    train_df: pd.DataFrame,
    val_fitness_df: pd.DataFrame,
    val_selection_df: pd.DataFrame,
) -> dict[str, dict[str, dict[str, object]]]:
    """Return split-aware context coverage for both trading directions."""
    return context_coverage_report({
        "train": train_df,
        "validation_fitness": val_fitness_df,
        "validation_selection": val_selection_df,
    })


def _validate_enriched_context_contract() -> None:
    """Reject a mixed, stale, or altered enriched train/test input pair."""
    input_paths = {
        "train": Path(_cfg.TRAIN_CSV_PATH),
        "test": Path(_cfg.TEST_CSV_PATH),
    }
    enriched_tapes = {
        name: path
        for name, path in input_paths.items()
        if path.name.endswith("_hwc_mwc_lwc.csv")
    }
    if not enriched_tapes:
        return

    raw_tapes = [
        f"{name}={path}"
        for name, path in input_paths.items()
        if name not in enriched_tapes
    ]
    if raw_tapes:
        raise RuntimeError(
            "The canonical pipeline requires an enriched train/test pair; "
            f"non-enriched input(s): {', '.join(raw_tapes)}. Re-enrich the "
            "raw tapes or supply a matching enriched pair and manifest."
        )

    manifest_path = Path(_cfg.ENRICHED_MANIFEST_PATH)
    if not manifest_path.exists():
        raise RuntimeError(
            f"Enriched inputs have no context manifest at "
            f"{manifest_path}. Re-enrich the raw train_new.csv before running "
            "the pipeline."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cannot read enriched context manifest {manifest_path}: {exc}. "
            "Re-enrich the raw train_new.csv before running the pipeline."
        ) from exc

    actual_version = manifest.get("context_algorithm_version")
    if actual_version is None:
        contract = manifest.get("context_contract", {})
        if isinstance(contract, dict):
            actual_version = contract.get("algorithm_version")
    expected_version = str(_cfg.CONTEXT_ALGORITHM_VERSION)
    if str(actual_version) != expected_version:
        raise RuntimeError(
            f"Enriched inputs use context contract "
            f"{actual_version!r}, but the pipeline requires "
            f"{expected_version!r}. Re-enrich raw train_new.csv and rebuild "
            "the cached splits before running."
        )

    tapes = manifest.get("tapes")
    if not isinstance(tapes, dict):
        raise RuntimeError(
            f"Enriched context manifest {manifest_path} has no tape hashes. "
            "Re-enrich the raw train/test pair before running the pipeline."
        )
    for name, input_path in input_paths.items():
        tape = tapes.get(name)
        expected_hash = tape.get("sha256") if isinstance(tape, dict) else None
        if not isinstance(expected_hash, str) or not expected_hash:
            raise RuntimeError(
                f"Enriched context manifest {manifest_path} has no {name} "
                "tape hash. Re-enrich the raw train/test pair before running "
                "the pipeline."
            )
        if not input_path.exists():
            raise RuntimeError(
                f"Configured enriched {name} tape is missing: {input_path}."
            )
        actual_hash = sha256_file(input_path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Configured enriched {name} tape {input_path} does not match "
                f"the hash recorded in {manifest_path}. Re-enrich the raw "
                "train/test pair and rebuild cached splits before running."
            )


def _context_island_sample_report(
    train_df: pd.DataFrame,
    val_fitness_df: pd.DataFrame,
    *,
    cv_folds: list | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Report context support on the windows Phase 2 islands actually sample.

    The production default uses singleton specialist islands.  For a generic
    clustered configuration this is a conservative per-universe proxy; the
    scheduler repeats the same check after it has resolved the exact cluster
    map.
    """
    base_seed = int(seed if seed is not None else _cfg.PHASE2_SEED)
    forbidden_ranges = (
        build_forbidden_ranges(cv_folds) if cv_folds else []
    )
    reference_sample = sample_df_for_phase2(
        train_df,
        random_state=base_seed,
    )
    reference_rows = len(reference_sample)

    if "symbol" in train_df.columns:
        symbols = sorted(
            {str(value) for value in train_df["symbol"].dropna().unique()}
        )
    else:
        symbols = []
    singleton_mode = bool(
        getattr(_cfg, "PHASE2_SYMBOL_SPECIALISTS_ENABLED", False)
        or getattr(_cfg, "PHASE2_ONE_SYMBOL_ISLANDS", False)
    )
    if symbols and singleton_mode:
        scopes = [
            (str(index), [symbol])
            for index, symbol in enumerate(symbols)
        ]
    else:
        scopes = [("universe", symbols)]

    report: dict[str, Any] = {
        "mode": "singleton_proxy" if singleton_mode else "universe_proxy",
        "reference_rows": int(reference_rows),
        "sampling_total": int(_cfg.PHASE1_SAMPLING_TOTAL),
        "islands": {},
        "failures": [],
        "passed_islands_by_direction": {
            "long": [],
            "short": [],
        },
        "failed_islands_by_direction": {
            "long": [],
            "short": [],
        },
    }
    for island_id, scope_symbols in scopes:
        if scope_symbols:
            train_scope = train_df[
                train_df["symbol"].astype(str).isin(scope_symbols)
            ]
            val_scope = val_fitness_df[
                val_fitness_df["symbol"].astype(str).isin(scope_symbols)
            ]
        else:
            train_scope = train_df
            val_scope = val_fitness_df

        island_data: dict[str, Any] = {
            "island_id": str(island_id),
            "symbols": list(scope_symbols),
            "directions": {},
        }
        for direction in ("long", "short"):
            island_seed = _derive_island_seed(
                base_seed,
                f"{direction}_{island_id}",
            )
            if island_seed is None:
                island_seed = base_seed
            train_sample = sample_df_for_phase2(
                train_scope,
                random_state=island_seed,
                forbidden_ranges=forbidden_ranges,
            )
            val_sample = sample_df_for_phase2(
                val_scope,
                random_state=_derive_val_sample_seed(island_seed),
            )
            hp = _cfg.resolve_island_hyperparams(
                "cluster",
                len(train_sample),
                max(1, reference_rows),
                n_symbols=max(1, len(scope_symbols)),
            )
            train_stats = _context_coverage_for_direction(
                train_sample, direction,
            )
            val_stats = _context_coverage_for_direction(
                val_sample, direction,
            )
            direction_data: dict[str, Any] = {
                "seed": int(island_seed),
                "train_rows": int(len(train_sample)),
                "validation_fitness_rows": int(len(val_sample)),
                "floors": {
                    "min_trade_support": int(hp.min_trade_support),
                    "min_trade_pool_floor": int(hp.min_trade_pool_floor),
                    "validation_trade_floor": int(hp.val_trade_floor),
                },
                "coverage": {
                    "train_sample": train_stats,
                    "validation_fitness_sample": val_stats,
                },
            }
            island_data["directions"][direction] = direction_data
            train_failures = context_floor_failures(
                train_stats,
                support_floor=hp.min_trade_support,
                pool_floor=hp.min_trade_pool_floor,
            )
            val_failures = context_floor_failures(
                val_stats,
                validation_floor=hp.val_trade_floor,
            )
            island_failures: list[str] = []
            for split_name, split_failures in (
                ("train_sample", train_failures),
                ("validation_fitness_sample", val_failures),
            ):
                for reason in split_failures:
                    detail = (
                        f"{direction}/{island_id}/{split_name}: {reason}"
                    )
                    island_failures.append(detail)
                    report["failures"].append(detail)
            if island_failures:
                report["failed_islands_by_direction"][direction].append(
                    {
                        "island_id": str(island_id),
                        "symbols": list(scope_symbols),
                        "failures": island_failures,
                    }
                )
            else:
                report["passed_islands_by_direction"][direction].append(
                    {
                        "island_id": str(island_id),
                        "symbols": list(scope_symbols),
                    }
                )
        report["islands"][str(island_id)] = island_data
    return report


def _write_context_preflight_report(
    report: dict[str, Any],
    failures: list[str],
    run_id: str | None,
) -> None:
    """Persist the diagnostic even when preflight blocks the pipeline."""
    if run_id is None:
        return
    preflight_path = Path(_cfg.REPORTS_DIR) / "phase2_context_preflight.json"
    preflight_path.parent.mkdir(parents=True, exist_ok=True)
    preflight_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "failures": failures,
                "coverage": report,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def _context_coverage_preflight(
    train_df: pd.DataFrame,
    val_fitness_df: pd.DataFrame,
    val_selection_df: pd.DataFrame,
    *,
    cv_folds: list | None = None,
    floor_aware: bool = True,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Log coverage and block only directions that cannot meet their floors."""
    report = _context_coverage_report(
        train_df, val_fitness_df, val_selection_df,
    )
    frames = {
        "train": train_df,
        "validation_fitness": val_fitness_df,
        "validation_selection": val_selection_df,
    }
    any_context_columns = any(
        any(column in frame.columns for column in _cfg.CONTEXT_COLUMNS)
        for frame in frames.values()
    )

    for split_name in _CONTEXT_COVERAGE_SPLITS:
        for direction in ("long", "short"):
            stats = report[split_name][direction]
            if stats["eligible_rows"] is None:
                logger.warning(
                    "Context coverage [%s/%s] unavailable; missing columns=%s",
                    split_name,
                    direction,
                    stats["missing_columns"],
                )
                continue
            logger.info(
                "Context coverage [%s/%s]: %.2f%% (%d / %d rows); "
                "permission=%s trigger=%s per_symbol=%s",
                split_name,
                direction,
                stats["coverage_pct"],
                stats["eligible_rows"],
                stats["total_rows"],
                stats.get("permission_rows"),
                stats.get("trigger_rows"),
                stats["by_symbol"],
            )

    # Keep empty compatibility fixtures usable for orchestration tests.  A
    # non-empty production frame without context columns is raw/unusable and
    # must fail before Phase 2 rather than silently disabling the fixed gate.
    if not any_context_columns:
        if not all(frame.empty for frame in frames.values()):
            _write_context_preflight_report(
                report,
                ["no context columns in non-empty split frames"],
                run_id,
            )
            raise RuntimeError(
                "Context coverage preflight failed before Phase 2: no context "
                "columns are present in the non-empty split frames. Enrich "
                "all tapes with the active 24-bar contract first."
            )
        logger.warning(
            "Context coverage preflight skipped for empty compatibility "
            "fixtures: no context columns are present."
        )
        return report

    failures: list[str] = []
    direction_failures: dict[str, list[str]] = {
        "long": [],
        "short": [],
    }

    def _record_failure(direction: str | None, reason: str) -> None:
        failures.append(reason)
        if direction in direction_failures:
            direction_failures[direction].append(reason)
        else:
            for fallback_direction in direction_failures:
                direction_failures[fallback_direction].append(reason)

    for split_name in _CONTEXT_PREFLIGHT_SPLITS:
        for direction in ("long", "short"):
            stats = report[split_name][direction]
            if stats["eligible_rows"] is None:
                _record_failure(
                    direction,
                    f"{split_name}/{direction} missing "
                    f"{stats['missing_columns']}",
                )
            elif int(stats["eligible_rows"]) == 0:
                _record_failure(
                    direction,
                    f"{split_name}/{direction} has zero eligible rows",
                )

    if floor_aware and not train_df.empty and not val_fitness_df.empty:
        report["run_id"] = run_id
        floor_requirements: dict[str, dict[str, dict[str, int]]] = {}
        for split_name, frame in (
            ("train", train_df),
            ("validation_fitness", val_fitness_df),
        ):
            floor_requirements[split_name] = {}
            for direction in ("long", "short"):
                if split_name == "train":
                    floors = {
                        "min_trade_support": int(
                            _cfg.effective_min_trade_support(len(frame))
                        ),
                        "min_trade_pool_floor": int(
                            _cfg.effective_min_trade_pool_floor(len(frame))
                        ),
                    }
                else:
                    floors = {
                        "validation_trade_floor": int(
                            _cfg.effective_pool_min_val_trades(len(frame))
                        ),
                    }
                floor_requirements[split_name][direction] = floors
                for reason in context_floor_failures(
                    report[split_name][direction],
                    support_floor=floors.get("min_trade_support"),
                    pool_floor=floors.get("min_trade_pool_floor"),
                    validation_floor=floors.get(
                        "validation_trade_floor"
                    ),
                ):
                    _record_failure(
                        direction,
                        f"{split_name}/{direction}: {reason}",
                    )
        report["floor_requirements"] = floor_requirements

        if _cfg.phase2_island_mode_enabled():
            island_report = _context_island_sample_report(
                train_df,
                val_fitness_df,
                cv_folds=cv_folds,
                seed=_cfg.PHASE2_SEED,
            )
            report["island_samples"] = island_report
            skip_starved = bool(
                getattr(_cfg, "PHASE2_SKIP_CONTEXT_STARVED_ISLANDS", True)
            )
            skipped_islands: list[str] = []
            for direction in ("long", "short"):
                passed = list(
                    island_report["passed_islands_by_direction"].get(
                        direction, [],
                    )
                )
                failed = list(
                    island_report["failed_islands_by_direction"].get(
                        direction, [],
                    )
                )
                if failed and passed and skip_starved:
                    for item in failed:
                        detail = (
                            f"{direction}/{item['island_id']} "
                            f"symbols={item['symbols']}: "
                            + "; ".join(item["failures"])
                        )
                        skipped_islands.append(detail)
                        logger.warning(
                            "Context support: skipping starved island %s "
                            "(direction still has %d supported island(s))",
                            detail,
                            len(passed),
                        )
                    continue
                for item in failed:
                    for failure in item["failures"]:
                        _record_failure(
                            direction,
                            f"island sample: {failure}",
                        )
            report["skipped_context_starved_islands"] = skipped_islands
            for island_id, island in island_report["islands"].items():
                for direction, direction_data in island["directions"].items():
                    train_stats = direction_data["coverage"]["train_sample"]
                    val_stats = direction_data[
                        "coverage"
                    ]["validation_fitness_sample"]
                    if (
                        train_stats["eligible_rows"] is None
                        or val_stats["eligible_rows"] is None
                    ):
                        logger.warning(
                            "Context support [%s/%s] unavailable; "
                            "missing train=%s validation=%s",
                            direction,
                            island_id,
                            train_stats.get("missing_columns", []),
                            val_stats.get("missing_columns", []),
                        )
                        continue
                    logger.info(
                        "Context support [%s/%s]: train %.2f%% (%d/%d), "
                        "validation fitness %.2f%% (%d/%d), floors=%s",
                        direction,
                        island_id,
                        train_stats["coverage_pct"],
                        train_stats["eligible_rows"],
                        train_stats["total_rows"],
                        val_stats["coverage_pct"],
                        val_stats["eligible_rows"],
                        val_stats["total_rows"],
                        direction_data["floors"],
                    )

    blocked_directions = sorted(
        direction
        for direction, direction_reasons in direction_failures.items()
        if direction_reasons
    )
    report["blocked_directions"] = blocked_directions
    report["direction_failures"] = direction_failures
    _write_context_preflight_report(report, failures, run_id)

    if len(blocked_directions) == len(direction_failures):
        raise RuntimeError(
            "Context coverage preflight blocked all directions before Phase 2: "
            + "; ".join(failures)
            + ". Repair context enrichment/sampling or recalibrate the "
            "direction-specific floors before rerunning."
        )
    if blocked_directions:
        supported_directions = sorted(
            set(direction_failures) - set(blocked_directions)
        )
        logger.warning(
            "Context coverage preflight blocked directions=%s; continuing "
            "supported directions=%s. Blocked directions will not enter "
            "Phase 2 or Phase 5.",
            blocked_directions,
            supported_directions,
        )
    return report


def _log_pipeline_config() -> None:
    """Log key hyperparameters at pipeline start."""
    debug_suffix = ""
    if _cfg.DEBUG_SYMBOL_SCOPE_ENABLED:
        debug_suffix = (
            f" | DEBUG start={_cfg.DEBUG_SYMBOL!r} "
            f"count={_cfg.DEBUG_SYMBOL_COUNT}"
        )
    if _cfg.phase2_island_mode_enabled():
        specialists_enabled = bool(
            getattr(_cfg, "PHASE2_SYMBOL_SPECIALISTS_ENABLED", False)
        )
        migration_effective = bool(
            _cfg.PHASE2_MIGRATION_ENABLED
        )
        total_gens = int(_cfg.PHASE2_ISLAND_TOTAL_GENERATIONS)
        cluster_ids = [str(i) for i in range(max(1, int(_cfg.PHASE2_N_CLUSTERS)))]
        budgets = compute_cluster_generation_budgets(
            total_gens,
            cluster_ids,
            shared_budget=bool(
                getattr(_cfg, "PHASE2_SHARED_ISLAND_GENERATION_BUDGET", False)
            ),
        )
        gens_per_cluster = budgets[cluster_ids[0]]
        epoch_gens = int(_cfg.PHASE2_ISLAND_EPOCH_GENERATIONS)
        phase2_fmt = (
            "PHASE2 algo=%s pop=%d island_total=%d configured_cluster_gens=%d "
            "epoch=%d shared_budget=%s joint_train_val=%s migration=%s "
            "specialists=%s"
        )
        phase2_args = (
            _cfg.PHASE2_ALGORITHM,
            _cfg.PHASE2_POPULATION_SIZE,
            total_gens,
            gens_per_cluster,
            epoch_gens,
            _cfg.PHASE2_SHARED_ISLAND_GENERATION_BUDGET,
            _cfg.PHASE2_JOINT_TRAIN_VAL,
            migration_effective,
            specialists_enabled,
        )
    else:
        phase2_fmt = "PHASE2 algo=%s pop=%d gen=%d joint_train_val=%s"
        phase2_args = (
            _cfg.PHASE2_ALGORITHM,
            _cfg.PHASE2_POPULATION_SIZE,
            _cfg.PHASE2_GENERATIONS,
            _cfg.PHASE2_JOINT_TRAIN_VAL,
        )
    logger.info(
        "Pipeline config: PHASE1 top_k=%d | "
        + phase2_fmt
        + " | RB Governor=canonical | %s",
        _cfg.PHASE1_TOP_K_FEATURES,
        *phase2_args,
        debug_suffix,
    )


class _NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that converts NumPy/JAX scalar and array types to native Python types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        try:
            import jax
            if isinstance(obj, jax.Array):
                return obj.tolist()
        except (ImportError, TypeError):
            pass
        try:
            import jax.numpy as jnp
            if isinstance(obj, jnp.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        return super().default(obj)


def _log_phase_entry(
    log_path: str,
    phase_name: str,
    start_time: str,
    end_time: str,
    elapsed_seconds: float,
    skipped: bool,
    result_summary: dict | None = None,
) -> None:
    """
    Append a structured JSON line to the pipeline log file.

    Parameters
    ----------
    log_path : str
        Path to the pipeline.log file.
    phase_name : str
        Human-readable phase name.
    start_time : str
        ISO 8601 start timestamp.
    end_time : str
        ISO 8601 end timestamp.
    elapsed_seconds : float
        Wall-clock duration in seconds.
    skipped : bool
        Whether the phase was skipped (outputs already valid).
    result_summary : dict | None
        Optional summary of phase results (e.g. pool size, rule count).
    """
    entry = {
        "phase": phase_name,
        "start_time": start_time,
        "end_time": end_time,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "skipped": skipped,
    }
    if result_summary:
        entry["result_summary"] = result_summary

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, cls=_NumpyJSONEncoder) + "\n")

    status = "SKIPPED" if skipped else "COMPLETED"
    logger.info(
        "[%s] %s in %.2fs",
        status,
        phase_name,
        elapsed_seconds,
    )


def _phase5_test_metrics(metrics: dict) -> dict:
    """Return test-split metrics (supports nested Phase 5 result shape)."""
    if "test" in metrics:
        return metrics["test"]
    return metrics


def _print_run_summary(results: dict[str, Any], phase: int | None, log_path: str) -> None:
    """Print a concise CLI summary for a full or single-phase run."""
    print("\n=== Pipeline Summary ===")

    if phase is None:
        phase5 = results.get("phase5", {})
        if phase5:
            for direction, metrics in phase5.items():
                if direction == "acceptance":
                    continue
                test_m = _phase5_test_metrics(metrics)
                print(
                    f"  {direction.upper()}: "
                    f"return={test_m.get('total_return_pct', 0.0):.2f}%, "
                    f"trades={test_m.get('executed_trades', 0)}, "
                    f"drawdown={test_m.get('max_drawdown_pct', 0.0):.2f}%"
                )
            acceptance = phase5.get("acceptance")
            if acceptance:
                print(
                    "  Acceptance: "
                    f"{acceptance.get('status', 'unknown')} "
                    "(test period remains diagnostic-only)"
                )
        else:
            print(f"  No OOS results (check {log_path} for details)")
        return

    if phase == 1:
        phase_result = results.get("phase1", {})
        print(
            "  Phase 1 selected features: "
            f"long={len(phase_result.get('long', []))}, "
            f"short={len(phase_result.get('short', []))}"
        )
        return

    if phase == 2:
        phase_result = results.get("phase2", {})
        print(
            "  Phase 2 pools: "
            f"long={len(phase_result.get('long', []))}, "
            f"short={len(phase_result.get('short', []))}"
        )
        return

    if phase in {3, 4}:
        phase_result = results.get("rb_governor", {})
        if not phase_result:
            print("  RB Governor produced no rule sets")
            return
        print(
            "  RB Governor compatibility run rule sets: "
            + ", ".join(
                f"{direction}={len(rule_set.get('rules_set', []))} rules"
                for direction, rule_set in phase_result.items()
            )
        )
        return

    phase5 = results.get("phase5", {})
    if phase5:
        for direction, metrics in phase5.items():
            if direction == "acceptance":
                continue
            test_m = _phase5_test_metrics(metrics)
            print(
                f"  {direction.upper()}: "
                f"return={test_m.get('total_return_pct', 0.0):.2f}%, "
                f"trades={test_m.get('executed_trades', 0)}, "
                f"drawdown={test_m.get('max_drawdown_pct', 0.0):.2f}%"
            )
        acceptance = phase5.get("acceptance")
        if acceptance:
            print(
                "  Acceptance: "
                f"{acceptance.get('status', 'unknown')} "
                "(test period remains diagnostic-only)"
            )
    else:
        print(f"  No OOS results (check {log_path} for details)")


# ---------------------------------------------------------------------------
# Pipeline_Orchestrator
# ---------------------------------------------------------------------------

class Pipeline_Orchestrator:
    """
    Top-level orchestrator for the GPU-Fuzzy Trading Pipeline.

    Runs all five phases in order, skipping phases whose outputs are already
    valid on disk.  Logs timing information for each phase to
    ``outputs/pipeline.log`` as JSON lines.

    Usage
    -----
    ::

        orchestrator = Pipeline_Orchestrator()
        results = orchestrator.run()

    Or via the command line::

        python -m gpu_fuzzy_trader.run_pipeline
    """

    def __init__(self, output_dir: str | None = None) -> None:
        self._output_dir = _resolve_output_root(output_dir)
        self._log_path = os.path.join(self._output_dir, "pipeline.log")
        self._cv_folds: list | None = None
        self._val_fitness_df: pd.DataFrame | None = None
        self._val_selection_df: pd.DataFrame | None = None
        self._phase2_status: dict[str, dict[str, Any]] = {}
        self._run_id: str | None = None
        self._run_started_at: str | None = None
        self._run_mode: str | None = None
        self._run_status: str | None = None

    def _begin_run(
        self,
        mode: str,
        *,
        clear_derived: bool,
        clear_phase2: bool = False,
    ) -> None:
        """Create a run identity and remove artifacts that cannot be trusted."""
        self._run_started_at = _now_iso()
        self._run_mode = str(mode)
        self._run_status = "running"
        self._run_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid.uuid4().hex[:10]
        )
        if clear_derived:
            try:
                OOS_Evaluator._clear_previous_reports()
            except Exception as exc:
                logger.warning(
                    "Failed to clear stale Phase 5 reports at run start: %s",
                    exc,
                )
            report_root = Path(_cfg.REPORTS_DIR)
            for pattern in (
                "phase2_*_coverage.json",
                "phase2_*_island_*_coverage.json",
                "phase2_*_context_support.json",
                "phase2_context_preflight.json",
            ):
                for path in report_root.glob(pattern):
                    path.unlink(missing_ok=True)
        if clear_phase2:
            output_root = Path(_cfg.OUTPUTS_DIR)
            for name in (
                "phase2_long_pool.json",
                "phase2_short_pool.json",
                "phase2_long_history.json",
                "phase2_short_history.json",
                "long.json",
                "short.json",
            ):
                (output_root / name).unlink(missing_ok=True)
            for path in Path(_cfg.REPORTS_DIR).glob(
                "rb_governor_*_report.json"
            ):
                path.unlink(missing_ok=True)
            if mode == "full":
                for name in (
                    "selected_features_long.json",
                    "selected_features_short.json",
                ):
                    (output_root / name).unlink(missing_ok=True)

        run_manifest = {
            "run_id": self._run_id,
            "mode": self._run_mode,
            "status": "running",
            "started_at": self._run_started_at,
            "output_dir": str(_cfg.OUTPUTS_DIR),
            "context_contract_digest": _cfg.context_contract_digest(),
        }
        manifest_path = Path(_cfg.REPORTS_DIR) / "run_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(run_manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _finish_run(self, status: str, *, error: str | None = None) -> None:
        """Persist the final status for the current run identity."""
        if not self._run_id:
            return
        self._run_status = str(status)
        manifest_path = Path(_cfg.REPORTS_DIR) / "run_manifest.json"
        payload: dict[str, Any] = {
            "run_id": self._run_id,
            "mode": self._run_mode,
            "status": str(status),
            "started_at": self._run_started_at,
            "finished_at": _now_iso(),
            "output_dir": str(_cfg.OUTPUTS_DIR),
            "context_contract_digest": _cfg.context_contract_digest(),
        }
        if error:
            payload["error"] = str(error)
        manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @contextmanager
    def _run_identity_guard(self):
        """Mark standalone phase runs failed when an exception escapes."""
        try:
            yield
        except Exception as exc:
            if self._run_status == "running":
                self._finish_run("failed", error=str(exc))
            raise
        else:
            if self._run_status == "running":
                self._finish_run("completed")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, force: bool = False) -> dict:
        """
        Run all pipeline phases in order.

        Returns
        -------
        dict
            Results from each phase, keyed by phase name.  Each value is
            the primary output of that phase (feature lists, pools, rule
            sets, OOS metrics, etc.).
        """
        with _temporary_output_paths(self._output_dir), self._run_identity_guard():
            logger.info("=" * 60)
            logger.info("GPU-Fuzzy Trading Pipeline — starting")
            logger.info("=" * 60)
            _log_pipeline_config()

            pipeline_start = time.monotonic()
            results: dict[str, Any] = {}

            # ------------------------------------------------------------------
            # Step 0: Create output directories
            # ------------------------------------------------------------------
            self._create_output_dirs()
            self._begin_run(
                "full",
                clear_derived=bool(force),
                clear_phase2=bool(force),
            )
            results["run_id"] = self._run_id

            # ------------------------------------------------------------------
            # Attach run.log FileHandler (must come after output dirs exist)
            # ------------------------------------------------------------------
            _run_log_handler = self._attach_run_log_handler()
            _sep = "=" * 80
            _start_ts = datetime.now(
                timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            _run_log_handler.stream.write(
                f"{_sep}\n[{_start_ts}] Pipeline run START\n{_sep}\n")
            _run_log_handler.stream.flush()

            try:
                # ------------------------------------------------------------------
                # Step 1: Load and prepare data
                # ------------------------------------------------------------------
                train_df, val_df = self._load_and_split_data()
                self._validate_active_configuration(train_df)
                val_fitness_df, val_selection_df = self._validation_scoring_frames(val_df)
                context_report = _context_coverage_preflight(
                    train_df,
                    val_fitness_df,
                    val_selection_df,
                    cv_folds=self._cv_folds,
                    floor_aware=True,
                    run_id=self._run_id,
                )
                results["data"] = {
                    "train_rows": len(train_df),
                    "val_rows": len(val_df),
                    "val_fitness_rows": len(val_fitness_df),
                    "val_selection_rows": len(val_selection_df),
                    "context_coverage": context_report,
                }
                blocked_directions = frozenset(
                    context_report.get("blocked_directions", [])
                )

                # ------------------------------------------------------------------
                # Phase 1: Feature Selection
                # ------------------------------------------------------------------
                phase1_train_df = self._mask_train_df_for_phase1(train_df)
                phase1_result = self._run_phase1(
                    phase1_train_df, force=force, val_df=None,
                )
                results["phase1"] = phase1_result
                train_df, val_df = self._prune_splits_after_phase1(
                    train_df, val_df, phase1_result)
                val_fitness_df, val_selection_df = self._prune_splits_after_phase1(
                    val_fitness_df, val_selection_df, phase1_result)
                self._cv_folds = self._prune_cv_folds_after_phase1(
                    self._cv_folds, phase1_result)

                # ------------------------------------------------------------------
                # Phase 2: Rule Pool Generation
                # ------------------------------------------------------------------
                phase2_result = self._run_phase2(
                    train_df,
                    phase1_result,
                    force=force,
                    val_df=val_fitness_df,
                    blocked_directions=blocked_directions,
                )
                results["phase2"] = phase2_result

                self._release_between_phases("RB Governor")
                rb_result = self._run_rb_governor(
                    train_df,
                    val_df,
                    phase2_result,
                    cv_folds=self._cv_folds,
                    val_selection_df=val_selection_df,
                )
                results["rb_governor"] = rb_result
                # Compatibility aliases: phase 3 and phase 4 now mean the
                # same canonical RB result and never dispatch legacy code.
                results["phase3"] = rb_result
                results["phase4"] = rb_result
                results["phase_status"] = {
                    "phase2": self._phase2_status,
                    "rb_governor": self._rb_status_summary(rb_result),
                }
                results["nested_validation"] = self._run_nested_validation(
                    train_df,
                    rb_result,
                    trial_count=count_trials(
                        phase2=phase2_result,
                        rb=rb_result,
                    ),
                )
                phase5_directions = frozenset(
                    direction
                    for direction, strategy in rb_result.items()
                    if strategy.get("rules_set")
                    and strategy.get("deployment_accepted") is True
                )

                # ------------------------------------------------------------------
                # Phase 5: Out-of-Sample Evaluation (always runs)
                # ------------------------------------------------------------------
                self._release_between_phases("Phase 5")

                phase5_result = self._run_phase5(
                    allowed_directions=phase5_directions)
                results["phase5"] = phase5_result

                # ------------------------------------------------------------------
                # Summary
                # ------------------------------------------------------------------
                total_elapsed = time.monotonic() - pipeline_start
                self._record_research_integrity(results, total_elapsed)
                self._finish_run("completed")
                logger.info("=" * 60)
                logger.info("Pipeline complete in %.2fs", total_elapsed)
                logger.info("=" * 60)

                return results

            finally:
                if self._run_status == "running":
                    self._finish_run("failed", error="pipeline aborted")
                _end_ts = datetime.now(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ")
                _run_log_handler.stream.write(
                    f"{_sep}\n[{_end_ts}] Pipeline run END\n{_sep}\n")
                _run_log_handler.stream.flush()
                self._detach_run_log_handler(_run_log_handler)

    def run_from_phase2(self, force: bool = True) -> dict:
        """
        Run Phase 2, the RB Governor, and Phase 5 using Phase 1 artifacts
        already on disk.

        Expects ``selected_features_{long,short}.json`` under this run's output
        directory (copy from a baseline run before calling). Does not re-run
        feature selection.

        Parameters
        ----------
        force : bool
            When True, always rebuild Phase 2 and RB outputs.

        Returns
        -------
        dict
            Keys: ``data``, ``phase1`` (loaded paths only), ``phase2`` … ``phase5``.
        """
        with _temporary_output_paths(self._output_dir), self._run_identity_guard():
            logger.info("=" * 60)
            logger.info("GPU-Fuzzy Trading Pipeline — Phase 2, RB Governor, Phase 5")
            logger.info("=" * 60)
            _log_pipeline_config()

            pipeline_start = time.monotonic()
            results: dict[str, Any] = {}

            self._create_output_dirs()
            self._begin_run(
                "from_phase2",
                clear_derived=bool(force),
                clear_phase2=bool(force),
            )
            results["run_id"] = self._run_id
            _run_log_handler = self._attach_run_log_handler()
            _sep = "=" * 80
            _start_ts = datetime.now(
                timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            _run_log_handler.stream.write(
                f"{_sep}\n[{_start_ts}] Pipeline run START (from phase 2)\n{_sep}\n")
            _run_log_handler.stream.flush()

            try:
                train_df, val_df = self._load_and_split_data()
                self._validate_active_configuration(train_df)
                val_fitness_df, val_selection_df = self._validation_scoring_frames(val_df)
                context_report = _context_coverage_preflight(
                    train_df,
                    val_fitness_df,
                    val_selection_df,
                    cv_folds=self._cv_folds,
                    floor_aware=True,
                    run_id=self._run_id,
                )
                results["data"] = {
                    "train_rows": len(train_df),
                    "val_rows": len(val_df),
                    "val_fitness_rows": len(val_fitness_df),
                    "val_selection_rows": len(val_selection_df),
                    "context_coverage": context_report,
                }
                blocked_directions = frozenset(
                    context_report.get("blocked_directions", [])
                )

                phase1_result = self._load_phase1_outputs()
                results["phase1"] = phase1_result
                train_df, val_df = self._prune_splits_after_phase1(
                    train_df, val_df, phase1_result)
                val_fitness_df, val_selection_df = self._prune_splits_after_phase1(
                    val_fitness_df, val_selection_df, phase1_result)
                self._cv_folds = self._prune_cv_folds_after_phase1(
                    self._cv_folds, phase1_result)

                phase2_result = self._run_phase2(
                    train_df,
                    phase1_result,
                    force=force,
                    val_df=val_fitness_df,
                    blocked_directions=blocked_directions,
                )
                results["phase2"] = phase2_result

                self._release_between_phases("RB Governor")
                rb_result = self._run_rb_governor(
                    train_df,
                    val_df,
                    phase2_result,
                    cv_folds=self._cv_folds,
                    val_selection_df=val_selection_df,
                )
                results["rb_governor"] = rb_result
                results["phase3"] = rb_result
                results["phase4"] = rb_result
                results["phase_status"] = {
                    "phase2": self._phase2_status,
                    "rb_governor": self._rb_status_summary(rb_result),
                }
                results["nested_validation"] = self._run_nested_validation(
                    train_df,
                    rb_result,
                    trial_count=count_trials(
                        phase2=phase2_result,
                        rb=rb_result,
                    ),
                )
                phase5_directions = frozenset(
                    direction
                    for direction, strategy in rb_result.items()
                    if strategy.get("rules_set")
                    and strategy.get("deployment_accepted") is True
                )

                self._release_between_phases("Phase 5")
                phase5_result = self._run_phase5(
                    allowed_directions=phase5_directions)
                results["phase5"] = phase5_result

                total_elapsed = time.monotonic() - pipeline_start
                self._record_research_integrity(results, total_elapsed)
                self._finish_run("completed")
                logger.info("=" * 60)
                logger.info(
                    "Pipeline (Phase 2, RB Governor, Phase 5) complete in %.2fs",
                    total_elapsed,
                )
                logger.info("=" * 60)

                return results

            finally:
                if self._run_status == "running":
                    self._finish_run("failed", error="pipeline aborted")
                _end_ts = datetime.now(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ")
                _run_log_handler.stream.write(
                    f"{_sep}\n[{_end_ts}] Pipeline run END\n{_sep}\n")
                _run_log_handler.stream.flush()
                self._detach_run_log_handler(_run_log_handler)

    def run_phase(self, phase: int) -> dict:
        """
        Run a single pipeline phase from disk-backed prerequisites.

        The selected phase is forced to re-run even if its outputs already
        exist. Earlier phases are not auto-run.
        """
        if phase not in {1, 2, 3, 4, 5}:
            raise ValueError(
                "phase must be one of 1, 2, 3, 4, or 5; "
                f"got {phase!r}"
            )

        with _temporary_output_paths(self._output_dir), self._run_identity_guard():
            logger.info("=" * 60)
            logger.info("GPU-Fuzzy Trading Pipeline — phase %d", phase)
            logger.info("=" * 60)
            _log_pipeline_config()

            pipeline_start = time.monotonic()
            results: dict[str, Any] = {}

            self._create_output_dirs()
            self._begin_run(
                f"phase_{phase}",
                clear_derived=phase >= 2,
                clear_phase2=phase == 2,
            )
            results["run_id"] = self._run_id

            if phase == 1:
                train_df, val_df = self._load_and_split_data()
                self._validate_active_configuration(train_df)
                results["data"] = {
                    "train_rows": len(train_df),
                    "val_rows": len(val_df),
                }
                phase1_train_df = self._mask_train_df_for_phase1(train_df)
                results["phase1"] = self._run_phase1(
                    phase1_train_df, force=True, val_df=None,
                )

            elif phase == 2:
                train_df, val_df = self._load_and_split_data()
                self._validate_active_configuration(train_df)
                val_fitness_df, val_selection_df = self._validation_scoring_frames(val_df)
                context_report = _context_coverage_preflight(
                    train_df,
                    val_fitness_df,
                    val_selection_df,
                    cv_folds=self._cv_folds,
                    floor_aware=True,
                    run_id=self._run_id,
                )
                results["data"] = {
                    "train_rows": len(train_df),
                    "val_rows": len(val_df),
                    "val_fitness_rows": len(val_fitness_df),
                    "val_selection_rows": len(val_selection_df),
                    "context_coverage": context_report,
                }
                blocked_directions = frozenset(
                    context_report.get("blocked_directions", [])
                )
                phase1_result = self._load_phase1_outputs()
                train_df, val_df = self._prune_splits_after_phase1(
                    train_df, val_df, phase1_result)
                val_fitness_df, val_selection_df = self._prune_splits_after_phase1(
                    val_fitness_df, val_selection_df, phase1_result)
                self._cv_folds = self._prune_cv_folds_after_phase1(
                    self._cv_folds, phase1_result)
                results["phase2"] = self._run_phase2(
                    train_df,
                    phase1_result,
                    force=True,
                    val_df=val_fitness_df,
                    blocked_directions=blocked_directions,
                )

            elif phase in {3, 4}:
                train_df, val_df = self._load_and_split_data()
                self._validate_active_configuration(train_df)
                _, val_selection_df = self._validation_scoring_frames(val_df)
                results["data"] = {
                    "train_rows": len(train_df),
                    "val_rows": len(val_df),
                }
                phase2_result = self._load_phase2_outputs()
                self._release_between_phases("RB Governor")
                rb_result = self._run_rb_governor(
                    train_df,
                    val_df,
                    phase2_result,
                    cv_folds=self._cv_folds,
                    val_selection_df=val_selection_df,
                )
                results["rb_governor"] = rb_result
                results["phase3"] = rb_result
                results["phase4"] = rb_result
                results["phase_status"] = {
                    "phase2": self._phase2_status,
                    "rb_governor": self._rb_status_summary(rb_result),
                }

            else:
                self._ensure_phase5_inputs()
                self._release_between_phases("Phase 5")
                results["phase5"] = self._run_phase5(
                    # Standalone Phase 5 is explicitly diagnostic: load the
                    # valid on-disk strategy files and evaluate them on the
                    # consumed test period.  Current-run direction scoping is
                    # only used by the full pipeline after RB returns.
                    allowed_directions=None
                )
                self._record_research_integrity(
                    results, time.monotonic() - pipeline_start,
                )

            total_elapsed = time.monotonic() - pipeline_start
            self._finish_run("completed")
            logger.info("=" * 60)
            logger.info("Phase %d complete in %.2fs", phase, total_elapsed)
            logger.info("=" * 60)
            return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _release_between_phases(self, before: str) -> None:
        """Release GPU/host resources between heavy phases (best-effort)."""
        logger.info("Releasing GPU resources before %s ...", before)
        try:
            from gpu_fuzzy_trader._memory import release_phase2_resources

            release_phase2_resources()
        except Exception as exc:
            logger.warning("Memory release before %s failed: %s", before, exc)

    def _attach_run_log_handler(self) -> logging.FileHandler:
        """Attach a FileHandler to the root logger writing to RUN_LOG_PATH.

        Returns the handler so it can be detached later via
        ``_detach_run_log_handler``.
        """
        os.makedirs(os.path.dirname(_cfg.RUN_LOG_PATH) or ".", exist_ok=True)
        handler = logging.FileHandler(
            _cfg.RUN_LOG_PATH, mode="a", encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        ))
        logging.getLogger().addHandler(handler)
        return handler

    def _detach_run_log_handler(self, handler: logging.FileHandler) -> None:
        """Remove *handler* from the root logger and close it."""
        logging.getLogger().removeHandler(handler)
        handler.close()

    def _create_output_dirs(self) -> None:
        """Create outputs/ and outputs/reports/ directories if they don't exist."""
        os.makedirs(_cfg.OUTPUTS_DIR, exist_ok=True)
        os.makedirs(_cfg.REPORTS_DIR, exist_ok=True)
        if bool(getattr(_cfg, "DATASET_MANIFEST_ENABLED", True)):
            try:
                write_dataset_manifests(
                    _cfg.OUTPUTS_DIR,
                    {
                        "train": _cfg.TRAIN_CSV_PATH,
                        "test_diagnostic": _cfg.TEST_CSV_PATH,
                        "forward_acceptance": getattr(
                            _cfg, "FORWARD_CSV_PATH", None,
                        ),
                    },
                )
            except Exception as exc:
                logger.warning("Dataset manifest failed (non-fatal): %s", exc)
        logger.info(
            "Output directories ready: %s, %s",
            _cfg.OUTPUTS_DIR,
            _cfg.REPORTS_DIR,
        )

    def _record_research_integrity(
        self,
        results: dict[str, Any],
        elapsed_seconds: float,
    ) -> None:
        """Append one auditable record after a pipeline evaluation completes."""
        if not bool(getattr(_cfg, "EXPERIMENT_LEDGER_ENABLED", True)):
            return
        phase5 = results.get("phase5", {})
        diagnostic: dict[str, dict[str, Any]] = {}
        if isinstance(phase5, dict):
            for direction, value in phase5.items():
                if direction == "acceptance" or not isinstance(value, dict):
                    continue
                metrics = value.get("test", value)
                if not isinstance(metrics, dict):
                    continue
                diagnostic[direction] = {
                    "total_return_pct": float(
                        metrics.get("total_return_pct", 0.0) or 0.0
                    ),
                    "profit_factor": float(
                        metrics.get("profit_factor", 0.0) or 0.0
                    ),
                    "max_drawdown_pct": float(
                        metrics.get("max_drawdown_pct", 0.0) or 0.0
                    ),
                    "executed_trades": int(
                        metrics.get("executed_trades", 0) or 0
                    ),
                }
        rb = results.get("rb_governor", {})
        phase2 = results.get("phase2", {})
        config_keys = (
            "GLOBAL_SEED",
            "PHASE2_GENERATIONS",
            "PHASE2_POPULATION_SIZE",
            "PHASE2_TP",
            "PHASE2_SL",
            "RB_RISK_OPTIMIZE_EXITS",
            "RB_CANDIDATE_RISK_ADMISSION_ENABLED",
            "RB_PHASE2_PROVENANCE_ONLY",
            "RB_ALLOW_PARTIAL_SPECIALIST_COVERAGE",
            "PHASE2_VAL_IN_FITNESS_PENALTY",
        )
        config_delta = {
            key: getattr(_cfg, key)
            for key in config_keys
            if hasattr(_cfg, key)
        }
        trial_count = count_trials(phase2=phase2, rb=rb)
        ExperimentLedger(_cfg.OUTPUTS_DIR).append({
            "record_type": "pipeline_run",
            "run_id": self._run_id,
            "run_mode": self._run_mode,
            "elapsed_seconds": round(float(elapsed_seconds), 3),
            "phase2_pool_sizes": {
                direction: len(value) if isinstance(value, list) else 0
                for direction, value in (
                    phase2.items() if isinstance(phase2, dict) else []
                )
            },
            "rb_status": {
                direction: {
                    "accepted": bool(
                        isinstance(value, dict)
                        and value.get("deployment_accepted") is True
                        and value.get("rules_set")
                    ),
                    "rules": len(value.get("rules_set", []))
                    if isinstance(value, dict) else 0,
                }
                for direction, value in (
                    rb.items() if isinstance(rb, dict) else []
                )
            },
            "diagnostic_test_metrics": diagnostic,
            "context_coverage": results.get("data", {}).get(
                "context_coverage", {}
            ),
            "acceptance": (
                phase5.get("acceptance", {})
                if isinstance(phase5, dict) else {}
            ),
            "trial_count_estimate": trial_count,
            "config": config_delta,
            "research_profile": {
                "profile_id": ResearchProfile.from_config(_cfg).profile_id,
                **ResearchProfile.from_config(_cfg).as_dict(),
            },
        })
        try:
            from gpu_fuzzy_trader.validation.multiplicity import (
                summarize_multiplicity,
            )
            nested = results.get("nested_validation", {})
            fold_returns = [
                float(report.get("median_return_pct", 0.0))
                for report in nested.values()
                if isinstance(report, dict)
            ] if isinstance(nested, dict) else []
            multiplicity = summarize_multiplicity(
                fold_returns=fold_returns,
                n_trials=trial_count,
            )
            Path(_cfg.REPORTS_DIR, "multiplicity_summary.json").write_text(
                json.dumps(multiplicity, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning(
                "Multiplicity report failed (non-fatal): %s",
                exc,
            )

    @staticmethod
    def _validate_active_configuration(train_df: pd.DataFrame) -> None:
        """Validate data-dependent configuration before any expensive phase."""
        n_symbols = (
            int(train_df["symbol"].astype(str).nunique())
            if "symbol" in train_df.columns
            else None
        )
        _cfg.validate_config(n_rows=len(train_df), n_symbols=n_symbols)
        _cfg.write_config_audit_report(
            _cfg.OUTPUTS_DIR,
            n_rows=len(train_df),
            n_symbols=n_symbols,
        )

    @staticmethod
    def _rb_status_summary(rb_result: dict[str, dict]) -> dict[str, dict[str, str]]:
        """Return explicit per-direction RB deployment status and reason."""
        summary: dict[str, dict[str, str]] = {}
        for direction, strategy in rb_result.items():
            accepted = bool(strategy.get("deployment_accepted")) and bool(
                strategy.get("rules_set")
            )
            summary[direction] = {
                "status": "accepted" if accepted else "fail_closed",
                "reason": (
                    str(strategy.get("reason"))
                    if strategy.get("reason")
                    else str(
                        strategy.get("deployment_reason", "accepted" if accepted else "no_strategy")
                    )
                ),
            }
        return summary

    def _load_phase1_outputs(self) -> dict[str, list[dict]]:
        """Load Phase 1 feature selection outputs for both directions."""
        result: dict[str, list[dict]] = {}
        missing: list[str] = []

        for direction, path in _selector_module._DIRECTION_PATHS.items():
            try:
                result[direction] = Feature_Selector.load_and_validate(path)
            except ValueError as exc:
                missing.append(f"{direction}: {path} ({exc})")

        if missing:
            raise FileNotFoundError(
                "Phase 2 requires Phase 1 outputs for both directions. "
                f"Missing or invalid: {', '.join(missing)}"
            )

        return result

    def _load_phase2_outputs(self) -> dict[str, list[dict]]:
        """Load Phase 2 pools for both directions from the persistent cache."""
        result: dict[str, list[dict]] = {}
        missing: list[str] = []
        self._phase2_status = {}

        for direction in ("long", "short"):
            try:
                pool = Rule_Pool_Generator.load_pool(direction)
            except Exception as exc:
                missing.append(
                    f"{direction}: {type(exc).__name__}: {exc}"
                )
                continue

            if pool is None:
                missing.append(
                    f"{direction}: {_phase2_module._POOL_PATHS[direction]}")
                continue

            result[direction] = pool
            self._phase2_status[direction] = {
                "status": "ok" if pool else "empty",
                "reason": "loaded_pool" if pool else "empty_pool",
                "pool_size": len(pool),
            }

        if missing:
            logger.warning(
                "RB Governor will fail closed for unavailable Phase 2 pools: %s",
                "; ".join(missing),
            )
            for entry in missing:
                direction = entry.split(":", 1)[0]
                result.setdefault(direction, [])
                self._phase2_status[direction] = {
                    "status": "error",
                    "reason": "missing_phase2_output",
                    "detail": entry,
                    "pool_size": 0,
                }

        return result

    def _ensure_phase5_inputs(self) -> None:
        """Ensure at least one valid strategy exists before standalone Phase 5."""
        evaluator = OOS_Evaluator()
        strategies = evaluator.load_strategies()
        if not strategies:
            raise FileNotFoundError(
                "Phase 5 requires at least one valid RB strategy output "
                "(long.json or short.json)."
            )

    def _accepted_strategy_directions(self) -> frozenset[str]:
        """Return only non-empty RB strategies accepted for deployment."""
        strategies = OOS_Evaluator.load_strategies()
        return frozenset(
            direction
            for direction, strategy in strategies.items()
            if strategy.get("rules_set")
            and strategy.get("deployment_accepted") is True
        )

    def _load_and_split_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load train_new.csv and split into train/validation DataFrames.

        When ``SPLIT_MODE`` is ``purged_walk_forward``, also rebuilds CV folds
        (stored on ``self._cv_folds``) for Phase 2 and RB.
        """
        _validate_enriched_context_contract()
        cached_split = load_cached_split_if_fresh()
        if cached_split is not None:
            train_df, val_df, val_fitness, val_selection, cv_folds = cached_split
            # A cache predating the enriched-input contract must never let the
            # canonical pipeline bypass the loader's fail-closed validation.
            for frame in (train_df, val_df, val_fitness, val_selection):
                validate_context_columns(frame)
            scaling = fit_fuzzy_feature_scaling(train_df)
            for frame in (train_df, val_df, val_fitness, val_selection):
                apply_fuzzy_feature_scaling(frame, scaling)
            self._cv_folds = cv_folds
            self._preloaded_val_fitness = val_fitness
            self._preloaded_val_selection = val_selection
            logger.info(
                "Using cached train/validation split from %s and %s (mode=%s)",
                _cfg.TRAIN_70_PATH,
                _cfg.VALIDATION_30_PATH,
                _cfg.SPLIT_MODE,
            )
            return self._apply_debug_symbol_scope(train_df, val_df)

        self._preloaded_val_fitness = None
        self._preloaded_val_selection = None

        logger.info("Loading training data from %s …", _cfg.TRAIN_CSV_PATH)
        loader = Data_Loader()
        # Materialise exact first-touch outcomes from the full source tape
        # before the loader removes the final horizon rows.  The internal
        # columns survive splitting and are the execution contract consumed by
        # CPU/GPU admission and Phase 5.
        train_full = loader.load_dataset(
            _cfg.TRAIN_CSV_PATH,
            drop_tail=False,
            include_barrier_outcomes=True,
            require_context=True,
        )
        logger.info(
            "Loaded %d rows, %d symbols",
            len(train_full),
            train_full["symbol"].nunique()
            if "symbol" in train_full.columns
            else 0,
        )
        for _dir, _perm, _trig in [
            ("long", _cfg.context_permission_column("long"),
             _cfg.context_trigger_column("long")),
            ("short", _cfg.context_permission_column("short"),
             _cfg.context_trigger_column("short")),
        ]:
            _mask = (
                (train_full[_perm].to_numpy() == 1)
                & (train_full[_trig].to_numpy() == 1)
            )
            _pct = _mask.sum() / max(len(_mask), 1)
            _log = logger.warning if _pct < 0.03 else logger.info
            _log(
                "Context mask [%s]: %.2f%% of rows active "
                "(%d / %d); perm=%s trig=%s",
                _dir, _pct * 100, int(_mask.sum()),
                len(_mask), _perm, _trig,
            )

        splitter = Data_Splitter()
        split_label = (
            "purged walk-forward"
            if _cfg.split_mode_is_purged_walk_forward()
            else f"holdout {_cfg.holdout_train_val_label()}"
        )
        logger.info("Splitting %s (%s) …", _cfg.TRAIN_CSV_PATH, split_label)

        train_df, val_df, cv_folds = splitter.split_and_persist(train_full)
        self._cv_folds = cv_folds

        from gpu_fuzzy_trader.data.splitter import split_validation_fitness_selection

        val_fitness, val_selection = split_validation_fitness_selection(val_df)
        scaling = fit_fuzzy_feature_scaling(train_df)
        for frame in (train_df, val_df, val_fitness, val_selection):
            apply_fuzzy_feature_scaling(frame, scaling)
        self._preloaded_val_fitness = val_fitness
        self._preloaded_val_selection = val_selection

        logger.info(
            "Split complete: train=%d rows, val=%d rows, cv_folds=%s",
            len(train_df),
            len(val_df),
            len(cv_folds) if cv_folds else "n/a",
        )
        return self._apply_debug_symbol_scope(train_df, val_df)

    def _validation_scoring_frames(
        self,
        val_df: pd.DataFrame,
        val_fitness: pd.DataFrame | None = None,
        val_selection: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load or derive validation fitness/selection halves for Phase 2 / RB."""
        from gpu_fuzzy_trader.backtest.df_slim import downcast_numeric_df
        from gpu_fuzzy_trader.data.splitter import split_validation_fitness_selection

        if val_fitness is None:
            val_fitness = getattr(self, "_preloaded_val_fitness", None)
        if val_selection is None:
            val_selection = getattr(self, "_preloaded_val_selection", None)

        if val_fitness is None or val_selection is None:
            val_fitness, val_selection = split_validation_fitness_selection(val_df)
        else:
            val_fitness = downcast_numeric_df(val_fitness)
            val_selection = downcast_numeric_df(val_selection)

        symbols = _cfg.resolve_debug_symbols(val_df)
        if symbols is not None:
            val_fitness = _cfg.filter_df_to_symbols(val_fitness, symbols)
            val_selection = _cfg.filter_df_to_symbols(val_selection, symbols)

        self._val_fitness_df = val_fitness
        self._val_selection_df = val_selection
        return val_fitness, val_selection

    def _apply_debug_symbol_scope(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Filter train/val to first N symbols when debug scope is enabled."""
        symbols = _cfg.resolve_debug_symbols(train_df)
        if symbols is None:
            return train_df, val_df

        scoped_train = _cfg.filter_df_to_symbols(train_df, symbols)
        scoped_val = _cfg.filter_df_to_symbols(val_df, symbols)

        if self._cv_folds:
            from dataclasses import replace

            from gpu_fuzzy_trader.validation.rolling_cv import PurgedFold

            sym_set = set(str(s) for s in symbols)
            scoped_folds: list[PurgedFold] = []
            for fold in self._cv_folds:
                train_part = fold.train_df[
                    fold.train_df["symbol"].astype(str).isin(sym_set)
                ].reset_index(drop=True)
                valid_part = fold.valid_df[
                    fold.valid_df["symbol"].astype(str).isin(sym_set)
                ].reset_index(drop=True)
                scoped_folds.append(
                    replace(
                        fold,
                        train_df=train_part,
                        valid_df=valid_part,
                        n_train_rows=len(train_part),
                        n_valid_rows=len(valid_part),
                    )
                )
            self._cv_folds = scoped_folds

        logger.info(
            "DEBUG SYMBOL SCOPE: enabled, count=%d symbols=%s, "
            "train=%d val=%d",
            len(symbols),
            symbols,
            len(scoped_train),
            len(scoped_val),
        )
        return scoped_train, scoped_val

    @staticmethod
    def _phase1_keep_feature_names(
        phase1_result: dict[str, list[dict]],
    ) -> list[str]:
        """Selected fuzzy features for Phase 2."""
        names: list[str] = []
        for direction in ("long", "short"):
            for fi in phase1_result.get(direction, []):
                n = fi.get("name")
                if n and n not in names:
                    names.append(n)
        return names

    @staticmethod
    def _prune_splits_after_phase1(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None,
        phase1_result: dict[str, list[dict]],
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Drop unused feature columns from train/val splits to reduce RAM."""
        from gpu_fuzzy_trader.backtest.df_slim import prune_train_columns

        names = Pipeline_Orchestrator._phase1_keep_feature_names(phase1_result)
        if not names:
            return train_df, val_df

        pruned_train = prune_train_columns(train_df, names)
        logger.info(
            "Pruned train_df columns after Phase 1: %d -> %d columns",
            len(train_df.columns),
            len(pruned_train.columns),
        )
        pruned_val = val_df
        if val_df is not None:
            pruned_val = prune_train_columns(val_df, names)
            logger.info(
                "Pruned val_df columns after Phase 1: %d -> %d columns",
                len(val_df.columns),
                len(pruned_val.columns),
            )
        from gpu_fuzzy_trader._memory import log_memory_rss

        log_memory_rss("after Phase 1 column prune")
        return pruned_train, pruned_val

    @staticmethod
    def _prune_cv_folds_after_phase1(
        cv_folds: list | None,
        phase1_result: dict[str, list[dict]],
    ) -> list | None:
        """Drop unused feature columns from CV fold DataFrames after Phase 1."""
        if not cv_folds:
            return cv_folds

        from dataclasses import replace

        from gpu_fuzzy_trader.backtest.df_slim import prune_train_columns
        from gpu_fuzzy_trader.validation.rolling_cv import PurgedFold

        names = Pipeline_Orchestrator._phase1_keep_feature_names(phase1_result)
        if not names:
            return cv_folds

        pruned: list[PurgedFold] = []
        for fold in cv_folds:
            pruned_train = prune_train_columns(fold.train_df, names)
            pruned_valid = prune_train_columns(fold.valid_df, names)
            pruned.append(
                replace(
                    fold,
                    train_df=pruned_train,
                    valid_df=pruned_valid,
                    n_train_rows=len(pruned_train),
                    n_valid_rows=len(pruned_valid),
                )
            )
        logger.info(
            "Pruned cv_folds columns after Phase 1 (%d folds)",
            len(pruned),
        )
        return pruned

    @staticmethod
    def _prune_train_df_after_phase1(
        train_df: pd.DataFrame,
        phase1_result: dict[str, list[dict]],
    ) -> pd.DataFrame:
        """Drop unused feature columns from train split (legacy single-split API)."""
        pruned_train, _ = Pipeline_Orchestrator._prune_splits_after_phase1(
            train_df, None, phase1_result,
        )
        return pruned_train

    # ------------------------------------------------------------------
    # Phase 1
    # ------------------------------------------------------------------

    def _mask_train_df_for_phase1(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Exclude CV/holdout valid bars (+ embargo) from Phase 1 training data."""
        if not self._cv_folds or train_df.empty:
            return train_df
        forbidden = build_forbidden_ranges(self._cv_folds)
        if not forbidden:
            return train_df
        masked = mask_df_to_safe_region(train_df, forbidden)
        logger.info(
            "Phase 1: masked train_df to safe region (%d -> %d rows)",
            len(train_df),
            len(masked),
        )
        return masked

    def _run_phase1(
        self,
        train_df: pd.DataFrame,
        force: bool = False,
        val_df: pd.DataFrame | None = None,
    ) -> dict[str, list[dict]]:
        """
        Run Phase 1 (Feature Selection) or skip if valid outputs exist.

        Returns
        -------
        dict[str, list[dict]]
            {"long": [...], "short": [...]}
        """
        phase_name = "Phase 1: Feature Selection"
        start_ts = _now_iso()
        t0 = time.monotonic()

        if not force:
            existing = Feature_Selector.skip_if_valid()
            if existing is not None:
                long_path = os.path.join(
                    _cfg.OUTPUTS_DIR, "selected_features_long.json")
                short_path = os.path.join(
                    _cfg.OUTPUTS_DIR, "selected_features_short.json")
                logger.info(
                    "Skipping %s: valid outputs at %s and %s (%d long, %d short features). "
                    "Delete those files to force Phase 1 to recompute.",
                    phase_name, long_path, short_path,
                    len(existing.get("long", [])),
                    len(existing.get("short", [])),
                )
                elapsed = time.monotonic() - t0
                _log_phase_entry(
                    self._log_path, phase_name, start_ts, _now_iso(),
                    elapsed, skipped=True,
                    result_summary={
                        "long_features": len(existing.get("long", [])),
                        "short_features": len(existing.get("short", [])),
                    },
                )
                return existing

        # Run Phase 1
        logger.info("Running %s …", phase_name)
        try:
            selector = Feature_Selector()
            result = selector.run(train_df, val_df=val_df)
        except Exception as exc:
            logger.error("Phase 1 failed: %s", exc, exc_info=True)
            raise

        elapsed = time.monotonic() - t0
        _log_phase_entry(
            self._log_path, phase_name, start_ts, _now_iso(),
            elapsed, skipped=False,
            result_summary={
                "long_features": len(result.get("long", [])),
                "short_features": len(result.get("short", [])),
            },
        )
        return result

    # ------------------------------------------------------------------
    # Phase 2
    # ------------------------------------------------------------------

    def _run_phase2(
        self,
        train_df: pd.DataFrame,
        phase1_result: dict[str, list[dict]],
        force: bool = False,
        val_df: pd.DataFrame | None = None,
        blocked_directions: frozenset[str] | None = None,
    ) -> dict:
        """
        Run Phase 2 (Rule Pool Generation) or skip if valid outputs exist.

        Returns
        -------
        dict
            ``{"long": [...], "short": [...]}``; blocked directions are empty.
        """

        phase_name = "Phase 2: Rule Pool Generation"
        start_ts = _now_iso()
        t0 = time.monotonic()

        pools: dict[str, list[dict]] = {}
        self._phase2_status = {}

        for direction in ("long", "short"):
            dir_phase_name = f"{phase_name} [{direction}]"
            dir_start_ts = _now_iso()
            dir_t0 = time.monotonic()

            if blocked_directions and direction in blocked_directions:
                logger.warning(
                    "Phase 2 [%s]: blocked by context-support preflight; "
                    "no rules will be generated.",
                    direction,
                )
                pools[direction] = []
                self._phase2_status[direction] = {
                    "status": "blocked",
                    "reason": "context_support_preflight",
                    "pool_size": 0,
                }
                dir_elapsed = time.monotonic() - dir_t0
                _log_phase_entry(
                    self._log_path,
                    dir_phase_name,
                    dir_start_ts,
                    _now_iso(),
                    dir_elapsed,
                    skipped=False,
                    result_summary=self._phase2_status[direction],
                )
                continue

            if not force:
                existing_pool = Rule_Pool_Generator.skip_if_valid(direction)
                if (
                    existing_pool
                    and bool(getattr(_cfg, "PHASE2_SYMBOL_SPECIALISTS_ENABLED", False))
                    and any(
                        not list(entry.get("source_symbols", []))
                        or not entry.get("phase2_rule_id")
                        or not list(entry.get("feature_conditions", []))
                        for entry in existing_pool
                        if isinstance(entry, dict)
                    )
                ):
                    logger.info(
                        "Discarding legacy global Phase 2 %s pool: "
                        "specialist provenance is required",
                        direction,
                    )
                    existing_pool = None
                if existing_pool is not None:
                    pool_path = _phase2_module._POOL_PATHS[direction]
                    logger.info(
                        "Skipping %s: valid pool at %s (%d rules)",
                        dir_phase_name, pool_path, len(existing_pool),
                    )
                    dir_elapsed = time.monotonic() - dir_t0
                    _log_phase_entry(
                        self._log_path, dir_phase_name, dir_start_ts, _now_iso(),
                        dir_elapsed, skipped=True,
                        result_summary={"pool_size": len(existing_pool)},
                    )
                    pools[direction] = existing_pool
                    self._phase2_status[direction] = {
                        "status": "ok" if existing_pool else "empty",
                        "reason": "resumed_pool" if existing_pool else "empty_pool",
                        "pool_size": len(existing_pool),
                    }
                    continue

            # Get feature infos for this direction
            feature_infos = phase1_result.get(direction, [])
            if not feature_infos:
                logger.warning(
                    "Phase 2 [%s]: no features from Phase 1; skipping direction.",
                    direction,
                )
                pools[direction] = []
                self._phase2_status[direction] = {
                    "status": "empty",
                    "reason": "no_phase1_features",
                    "pool_size": 0,
                }
                continue

            # Run Phase 2 for this direction
            logger.info(
                "Running %s … (%d features from Phase 1)",
                dir_phase_name, len(feature_infos),
            )
            try:
                if _cfg.phase2_island_mode_enabled():
                    from gpu_fuzzy_trader.phases.phase2_island_scheduler import (
                        run_cluster_phase2,
                    )
                    pool = run_cluster_phase2(
                        train_df=train_df,
                        val_df=val_df,
                        feature_infos=feature_infos,
                        direction=direction,
                        cv_folds=self._cv_folds,
                        seed=_cfg.PHASE2_SEED,
                        run_id=self._run_id,
                    )
                else:
                    generator = Rule_Pool_Generator(
                        train_df=train_df,
                        feature_infos=feature_infos,
                        direction=direction,
                        val_df=val_df,
                        cv_folds=self._cv_folds,
                        seed=_cfg.PHASE2_SEED,
                        run_id=self._run_id,
                    )
                    pool = generator.run()
            except Exception as exc:
                logger.error(
                    "Phase 2 [%s] failed: %s", direction, exc, exc_info=True
                )
                pool = []
                self._phase2_status[direction] = {
                    "status": "error",
                    "reason": "phase2_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "pool_size": 0,
                }
            else:
                self._phase2_status[direction] = {
                    "status": "ok" if pool else "empty",
                    "reason": "generated" if pool else "empty_pool",
                    "pool_size": len(pool),
                }

            dir_elapsed = time.monotonic() - dir_t0
            _log_phase_entry(
                self._log_path, dir_phase_name, dir_start_ts, _now_iso(),
                dir_elapsed, skipped=False,
                result_summary=self._phase2_status[direction],
            )
            pools[direction] = pool

            # Release JAX compilation cache and GC between directions to
            # avoid host RAM exhaustion when the next direction recompiles.
            from gpu_fuzzy_trader._memory import release_phase2_resources
            release_phase2_resources()

        # Log overall Phase 2 timing
        elapsed = time.monotonic() - t0
        _log_phase_entry(
            self._log_path, phase_name, start_ts, _now_iso(),
            elapsed, skipped=False,
            result_summary={
                "long_pool_size": len(pools.get("long", [])),
                "short_pool_size": len(pools.get("short", [])),
            },
        )
        return pools

    # ------------------------------------------------------------------
    # RB Governor
    # ------------------------------------------------------------------

    def _run_rb_governor(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        phase2_result: dict[str, list[dict]],
        *,
        cv_folds: list | None = None,
        val_selection_df: pd.DataFrame | None = None,
    ) -> dict[str, dict]:
        """Run the canonical RB Governor pipeline.

        Returns a dict keyed by direction, each value being the strategy dict
        written to disk (containing ``direction`` and ``rules_set``).  The
        shape matches the evaluator-facing strategy output so Phase 5 can load
        the generated ``{direction}.json`` files unchanged.
        """
        phase_name = "RB Governor"
        start_ts = _now_iso()
        t0 = time.monotonic()

        directions = ("long", "short")
        logger.info(
            "Running %s … (directions=%s, pools=%s)",
            phase_name,
            list(directions),
            {d: len(phase2_result.get(d, [])) for d in directions},
        )

        n_symbols = (
            int(train_df["symbol"].astype(str).nunique())
            if "symbol" in train_df.columns
            else None
        )
        # The production contract is deliberately narrower than the legacy
        # compatibility API: RB may only select/compose Phase 2 discoveries,
        # and specialist islands must remain explicitly symbol-scoped.  Apply
        # these policy overrides only around the canonical pipeline call so
        # small unit/diagnostic callers can still exercise legacy helpers.
        # Partial specialist coverage stays under the config flag (needed when
        # PHASE2_SKIP_CONTEXT_STARVED_ISLANDS drops an ETH island).
        rb_policy_attrs = {
            "RB_REQUIRE_SYMBOL_FILTERS": bool(
                getattr(_cfg, "PHASE2_SYMBOL_SPECIALISTS_ENABLED", False)
            ),
            "RB_UNIVARIATE_BASELINE_ENABLED": False,
            "RB_PHASE2_PROVENANCE_ONLY": True,
            "RB_RECENCY_RESCUE_ENABLED": False,
            "RB_FULL_VALIDATION_RECOVERY_ENABLED": False,
            "RB_CANDIDATE_RISK_ADMISSION_ENABLED": False,
            "RB_RISK_OPTIMIZE_EXITS": False,
            "RB_CANONICAL_PIPELINE_ACTIVE": True,
        }
        rb_policy_previous = {
            name: getattr(_cfg, name) for name in rb_policy_attrs
        }
        for name, value in rb_policy_attrs.items():
            setattr(_cfg, name, value)
        _cfg.write_config_audit_report(
            _cfg.OUTPUTS_DIR,
            n_rows=len(train_df),
            n_symbols=n_symbols,
        )
        try:
            strategies = _rb_governor_module.run_rb_governor_pipeline(
                train_df=train_df,
                val_df=val_df,
                pools=phase2_result,
                directions=directions,
                output_dir=_cfg.OUTPUTS_DIR,
                cv_folds=cv_folds,
                val_selection_df=val_selection_df,
                failure_reasons={
                    direction: status.get("reason", "phase2_empty_pool")
                    for direction, status in self._phase2_status.items()
                    if status.get("status") != "ok"
                },
            )
        except Exception as exc:
            logger.error("RB Governor failed: %s", exc, exc_info=True)
            strategies = {
                direction: _rb_governor_module._write_fail_closed_strategy(
                    Path(_cfg.OUTPUTS_DIR),
                    Path(_cfg.REPORTS_DIR),
                    direction,
                    "rb_governor_error",
                    phase2_status={
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                for direction in ("long", "short")
            }
        finally:
            for name, value in rb_policy_previous.items():
                setattr(_cfg, name, value)

        elapsed = time.monotonic() - t0
        _log_phase_entry(
            self._log_path, phase_name, start_ts, _now_iso(),
            elapsed, skipped=False,
            result_summary={
                d: len(s.get("rules_set", []))
                for d, s in strategies.items()
            },
        )
        return strategies

    # ------------------------------------------------------------------
    # Phase 5
    # ------------------------------------------------------------------

    def _run_phase5(
        self,
        allowed_directions: frozenset[str] | None = None,
    ) -> dict[str, dict]:
        """
        Run Phase 5 (Out-of-Sample Evaluation). Always runs.

        Parameters
        ----------
        allowed_directions : frozenset[str] | None
            Directions produced in the current run's RB stage. An empty frozenset
            skips all OOS evaluation so stale on-disk strategies are not scored.
            ``None`` evaluates every valid strategy file (standalone Phase 5).

        Returns
        -------
        dict[str, dict]
            OOS metrics keyed by direction.
        """
        phase_name = "Phase 5: Out-of-Sample Evaluation"
        start_ts = _now_iso()
        t0 = time.monotonic()

        if allowed_directions is not None:
            logger.info(
                "Running %s … (directions from current run: %s)",
                phase_name,
                ", ".join(sorted(allowed_directions)) or "none",
            )
        else:
            logger.info("Running %s …", phase_name)
        try:
            evaluator = OOS_Evaluator(run_id=self._run_id)
            result = evaluator.run(allowed_directions=allowed_directions)
        except Exception as exc:
            logger.error("Phase 5 failed: %s", exc, exc_info=True)
            raise RuntimeError("Phase 5 evaluation failed") from exc

        elapsed = time.monotonic() - t0
        _log_phase_entry(
            self._log_path, phase_name, start_ts, _now_iso(),
            elapsed, skipped=False,
            result_summary={
                d: {
                    "total_return_pct": _phase5_test_metrics(m).get(
                        "total_return_pct", 0.0),
                    "executed_trades": _phase5_test_metrics(m).get(
                        "executed_trades", 0),
                }
                for d, m in result.items()
                if d != "acceptance"
            },
        )
        return result

    @staticmethod
    def _run_nested_validation(
        train_df: pd.DataFrame,
        strategies: dict[str, dict],
        *,
        trial_count: int = 1,
    ) -> dict[str, dict]:
        """Evaluate current packages on purged outer folds only."""
        if not bool(getattr(_cfg, "NESTED_VALIDATION_ENABLED", True)):
            return {}
        try:
            from gpu_fuzzy_trader.validation.nested_walk_forward import (
                write_nested_reports,
            )
            nested_strategies = {
                direction: {
                    **strategy,
                    "trial_count": int(trial_count),
                }
                for direction, strategy in strategies.items()
            }
            nested = write_nested_reports(
                _cfg.OUTPUTS_DIR,
                nested_strategies,
                train_df,
                n_outer=int(getattr(
                    _cfg, "NESTED_VALIDATION_OUTER_FOLDS", 3,
                )),
            )
            from gpu_fuzzy_trader.validation.baselines import (
                write_baseline_reports,
            )
            baseline = write_baseline_reports(
                _cfg.OUTPUTS_DIR,
                train_df,
                nested_strategies,
            )
            for direction, report in nested.items():
                report["baselines"] = baseline.get(direction, {})
            return nested
        except Exception as exc:
            logger.warning(
                "Nested validation failed (non-fatal to pipeline): %s",
                exc,
            )
            return {}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """Run the full pipeline from the command line."""
    parser = argparse.ArgumentParser(
        prog="python -m gpu_fuzzy_trader.run_pipeline",
        description="Run the GPU-Fuzzy Trading pipeline.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Base output directory for this run (defaults to outputs/).",
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=(1, 2, 3, 4, 5),
        default=None,
        help=(
            "Run one phase instead of the full pipeline. "
            "3 and 4 are RB Governor compatibility aliases."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse valid Phase 1/2 artifacts when available (default: full rerun).",
    )
    parser.add_argument(
        "--from-phase",
        type=int,
        choices=(2,),
        default=None,
        help="Start at phase 2 (requires Phase 1 outputs in --output).",
    )

    args = parser.parse_args([] if argv is None else argv)
    orchestrator = Pipeline_Orchestrator(output_dir=args.output)
    try:
        if args.from_phase == 2:
            results = orchestrator.run_from_phase2(force=True)
        elif args.phase is None:
            results = orchestrator.run(force=not args.resume)
        else:
            results = orchestrator.run_phase(args.phase)

        _print_run_summary(results, args.phase, orchestrator._log_path)
        print(f"\nStructured log saved to: {orchestrator._log_path}")
    except Exception as exc:
        logger.error("Pipeline failed with unhandled exception: %s",
                     exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
