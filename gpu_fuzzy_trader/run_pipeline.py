"""
run_pipeline.py — Pipeline_Orchestrator

Top-level orchestrator for the GPU-Fuzzy Trading Pipeline.

Execution order:
  1. Create output directories (outputs/, outputs/reports/)
  2. Load and prepare data (Data_Loader + Data_Splitter)
  3. Phase 1: Feature_Selector
  4. Phase 2: Rule_Pool_Generator for both directions
  5. Phase 3: Rule_Set_Selector for both directions
  6. Phase 4: WalkForwardRiskOptimizer for both directions
  7. Phase 5: OOS_Evaluator (always runs)

Default CLI (``python -m gpu_fuzzy_trader.run_pipeline``) always re-runs Phases 1–4.
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
from gpu_fuzzy_trader.phases import phase4_wf_optimizer as _phase4_module
from gpu_fuzzy_trader.phases import phase3_rule_set as _phase3_module
from gpu_fuzzy_trader.phases import phase2_rule_pool as _phase2_module
from gpu_fuzzy_trader import rb_governor as _rb_governor_module
from gpu_fuzzy_trader.features import selector as _selector_module
from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.data.loader import Data_Loader
from gpu_fuzzy_trader.data.splitter import Data_Splitter
from gpu_fuzzy_trader.features.selector import Feature_Selector
from gpu_fuzzy_trader.phases.phase2_rule_pool import Rule_Pool_Generator
from gpu_fuzzy_trader.phases.phase3_rule_set import Rule_Set_Selector
from gpu_fuzzy_trader.phases.phase4_wf_optimizer import WalkForwardRiskOptimizer
from gpu_fuzzy_trader.phases.phase5_oos import OOS_Evaluator

import argparse
from contextlib import contextmanager
import pandas as pd
from typing import Any
from datetime import datetime, timezone
import time
import sys
import os
import logging
import json
import numpy as np

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


def _phase2_result_has_rules(phase2_result: dict) -> bool:
    """True when Phase 2 produced at least one rule."""
    return bool(phase2_result.get("long")) or bool(phase2_result.get("short"))


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
        "phase3_output_paths": _phase3_module._OUTPUT_PATHS,
        "phase4_output_paths": _phase4_module._OUTPUT_PATHS,
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

        _phase3_module._OUTPUT_PATHS = {
            "long": os.path.join(output_root, "long.json"),
            "short": os.path.join(output_root, "short.json"),
        }

        _phase4_module._OUTPUT_PATHS = {
            "long": os.path.join(output_root, "long.json"),
            "short": os.path.join(output_root, "short.json"),
        }

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
        _phase3_module._OUTPUT_PATHS = previous_state["phase3_output_paths"]
        _phase4_module._OUTPUT_PATHS = previous_state["phase4_output_paths"]
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


def _log_pipeline_config() -> None:
    """Log key hyperparameters at pipeline start."""
    debug_suffix = ""
    if _cfg.DEBUG_SYMBOL_SCOPE_ENABLED:
        debug_suffix = (
            f" | DEBUG start={_cfg.DEBUG_SYMBOL!r} "
            f"count={_cfg.DEBUG_SYMBOL_COUNT}"
        )
    logger.info(
        "Pipeline config: PHASE1 top_k=%d | "
        "PHASE2 algo=%s pop=%d gen=%d joint_train_val=%s | "
        "PHASE3 per-symbol greedy | "
        "PHASE4 grid_search=True | %s",
            _cfg.PHASE1_TOP_K_FEATURES,
            _cfg.PHASE2_ALGORITHM,
            _cfg.PHASE2_POPULATION_SIZE,
            _cfg.PHASE2_GENERATIONS,
            _cfg.PHASE2_JOINT_TRAIN_VAL,
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
                test_m = _phase5_test_metrics(metrics)
                print(
                    f"  {direction.upper()}: "
                    f"return={test_m.get('total_return_pct', 0.0):.2f}%, "
                    f"trades={test_m.get('executed_trades', 0)}, "
                    f"drawdown={test_m.get('max_drawdown_pct', 0.0):.2f}%"
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

    if phase == 3:
        phase_result = results.get("phase3", {})
        if not phase_result:
            print("  No Phase 3 rule sets were produced")
            return
        print(
            "  Phase 3 rule sets: "
            + ", ".join(
                f"{direction}={len(rule_set.get('rules_set', []))} rules"
                for direction, rule_set in phase_result.items()
            )
        )
        return

    if phase == 4:
        phase_result = results.get("phase4", {})
        if not phase_result:
            print("  No Phase 4 optimized strategies were produced")
            return
        print(
            "  Phase 4 optimized strategies: "
            + ", ".join(
                f"{direction}={len(rule_set.get('rules_set', []))} rules"
                for direction, rule_set in phase_result.items()
            )
        )
        return

    phase5 = results.get("phase5", {})
    if phase5:
        for direction, metrics in phase5.items():
            test_m = _phase5_test_metrics(metrics)
            print(
                f"  {direction.upper()}: "
                f"return={test_m.get('total_return_pct', 0.0):.2f}%, "
                f"trades={test_m.get('executed_trades', 0)}, "
                f"drawdown={test_m.get('max_drawdown_pct', 0.0):.2f}%"
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
        with _temporary_output_paths(self._output_dir):
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
                results["data"] = {
                    "train_rows": len(train_df),
                    "val_rows": len(val_df),
                }

                # ------------------------------------------------------------------
                # Phase 1: Feature Selection
                # ------------------------------------------------------------------
                phase1_result = self._run_phase1(train_df, force=force, val_df=val_df)
                results["phase1"] = phase1_result
                train_df, val_df = self._prune_splits_after_phase1(
                    train_df, val_df, phase1_result)
                self._cv_folds = self._prune_cv_folds_after_phase1(
                    self._cv_folds, phase1_result)

                # ------------------------------------------------------------------
                # Phase 2: Rule Pool Generation
                # ------------------------------------------------------------------
                phase2_result = self._run_phase2(
                    train_df, phase1_result, force=force, val_df=val_df)
                results["phase2"] = phase2_result

                # Check if Phase 2 produced any rules; if not, skip Phases 3 and 4
                pool_empty = not _phase2_result_has_rules(phase2_result)

                if pool_empty:
                    logger.warning(
                        "Phase 2 produced no rules for any island. "
                        "Skipping Phases 3 and 4."
                    )
                    results["phase3"] = {}
                    results["phase4"] = {}
                    phase5_directions: frozenset[str] = frozenset()
                elif bool(getattr(_cfg, "RB_GOVERNOR_ENABLED", False)):
                    rb_result = self._run_rb_governor(
                        train_df, val_df, phase2_result)
                    results["phase3"] = rb_result
                    results["phase4"] = rb_result
                    phase5_directions = frozenset(rb_result.keys())
                else:
                    # ------------------------------------------------------------------
                    # Phase 3: Rule Set Selection
                    # ------------------------------------------------------------------
                    phase3_result = self._run_phase3(
                        train_df, val_df, phase2_result, force=force)
                    results["phase3"] = phase3_result

                    # ------------------------------------------------------------------
                    # Phase 4: RL Risk Optimization
                    # ------------------------------------------------------------------
                    phase4_result = self._run_phase4(
                        train_df, val_df, phase3_result, force=force)
                    results["phase4"] = phase4_result
                    phase5_directions = frozenset(phase3_result.keys())

                # ------------------------------------------------------------------
                # Phase 5: Out-of-Sample Evaluation (always runs)
                # ------------------------------------------------------------------
                # Free GPU memory before CPU-only Phase 5 to avoid OOM
                try:
                    import jax
                    jax.clear_caches()
                except Exception:
                    pass
                import gc
                gc.collect()

                phase5_result = self._run_phase5(
                    allowed_directions=phase5_directions)
                results["phase5"] = phase5_result

                # ------------------------------------------------------------------
                # Summary
                # ------------------------------------------------------------------
                total_elapsed = time.monotonic() - pipeline_start
                logger.info("=" * 60)
                logger.info("Pipeline complete in %.2fs", total_elapsed)
                logger.info("=" * 60)

                return results

            finally:
                _end_ts = datetime.now(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ")
                _run_log_handler.stream.write(
                    f"{_sep}\n[{_end_ts}] Pipeline run END\n{_sep}\n")
                _run_log_handler.stream.flush()
                self._detach_run_log_handler(_run_log_handler)

    def run_from_phase2(self, force: bool = True) -> dict:
        """
        Run Phases 2–5 using Phase 1 artifacts already on disk.

        Expects ``selected_features_{long,short}.json`` under this run's output
        directory (copy from a baseline run before calling). Does not re-run
        feature selection.

        Parameters
        ----------
        force : bool
            When True, always rebuild Phases 2–4 outputs.

        Returns
        -------
        dict
            Keys: ``data``, ``phase1`` (loaded paths only), ``phase2`` … ``phase5``.
        """
        with _temporary_output_paths(self._output_dir):
            logger.info("=" * 60)
            logger.info("GPU-Fuzzy Trading Pipeline — Phases 2–5")
            logger.info("=" * 60)
            _log_pipeline_config()

            pipeline_start = time.monotonic()
            results: dict[str, Any] = {}

            self._create_output_dirs()
            _run_log_handler = self._attach_run_log_handler()
            _sep = "=" * 80
            _start_ts = datetime.now(
                timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            _run_log_handler.stream.write(
                f"{_sep}\n[{_start_ts}] Pipeline run START (from phase 2)\n{_sep}\n")
            _run_log_handler.stream.flush()

            try:
                train_df, val_df = self._load_and_split_data()
                results["data"] = {
                    "train_rows": len(train_df),
                    "val_rows": len(val_df),
                }

                phase1_result = self._load_phase1_outputs()
                results["phase1"] = phase1_result
                train_df, val_df = self._prune_splits_after_phase1(
                    train_df, val_df, phase1_result)
                self._cv_folds = self._prune_cv_folds_after_phase1(
                    self._cv_folds, phase1_result)

                phase2_result = self._run_phase2(
                    train_df, phase1_result, force=force, val_df=val_df)
                results["phase2"] = phase2_result

                pool_empty = not _phase2_result_has_rules(phase2_result)

                if pool_empty:
                    logger.warning(
                        "Phase 2 produced no rules for any island. "
                        "Skipping Phases 3 and 4."
                    )
                    results["phase3"] = {}
                    results["phase4"] = {}
                    phase5_directions: frozenset[str] = frozenset()
                elif bool(getattr(_cfg, "RB_GOVERNOR_ENABLED", False)):
                    rb_result = self._run_rb_governor(
                        train_df, val_df, phase2_result)
                    results["phase3"] = rb_result
                    results["phase4"] = rb_result
                    phase5_directions = frozenset(rb_result.keys())
                else:
                    phase3_result = self._run_phase3(
                        train_df, val_df, phase2_result, force=force)
                    results["phase3"] = phase3_result

                    phase4_result = self._run_phase4(
                        train_df, val_df, phase3_result, force=force)
                    results["phase4"] = phase4_result
                    phase5_directions = frozenset(phase3_result.keys())

                phase5_result = self._run_phase5(
                    allowed_directions=phase5_directions)
                results["phase5"] = phase5_result

                total_elapsed = time.monotonic() - pipeline_start
                logger.info("=" * 60)
                logger.info(
                    "Pipeline (phases 2–5) complete in %.2fs", total_elapsed)
                logger.info("=" * 60)

                return results

            finally:
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
            raise ValueError(f"phase must be between 1 and 5, got {phase!r}")

        with _temporary_output_paths(self._output_dir):
            logger.info("=" * 60)
            logger.info("GPU-Fuzzy Trading Pipeline — phase %d", phase)
            logger.info("=" * 60)
            _log_pipeline_config()

            pipeline_start = time.monotonic()
            results: dict[str, Any] = {}

            self._create_output_dirs()

            if phase == 1:
                train_df, val_df = self._load_and_split_data()
                results["data"] = {
                    "train_rows": len(train_df),
                    "val_rows": len(val_df),
                }
                results["phase1"] = self._run_phase1(train_df, force=True, val_df=val_df)

            elif phase == 2:
                train_df, val_df = self._load_and_split_data()
                results["data"] = {
                    "train_rows": len(train_df),
                    "val_rows": len(val_df),
                }
                phase1_result = self._load_phase1_outputs()
                train_df, val_df = self._prune_splits_after_phase1(
                    train_df, val_df, phase1_result)
                self._cv_folds = self._prune_cv_folds_after_phase1(
                    self._cv_folds, phase1_result)
                results["phase2"] = self._run_phase2(
                    train_df, phase1_result, force=True, val_df=val_df)

            elif phase == 3:
                train_df, val_df = self._load_and_split_data()
                results["data"] = {
                    "train_rows": len(train_df),
                    "val_rows": len(val_df),
                }
                phase2_result = self._load_phase2_outputs()
                if bool(getattr(_cfg, "RB_GOVERNOR_ENABLED", False)):
                    rb_result = self._run_rb_governor(
                        train_df, val_df, phase2_result)
                    results["phase3"] = rb_result
                    results["phase4"] = rb_result
                else:
                    results["phase3"] = self._run_phase3(
                        train_df, val_df, phase2_result, force=True)

            elif phase == 4:
                train_df, val_df = self._load_and_split_data()
                results["data"] = {
                    "train_rows": len(train_df),
                    "val_rows": len(val_df),
                }
                if bool(getattr(_cfg, "RB_GOVERNOR_ENABLED", False)):
                    phase2_result = self._load_phase2_outputs()
                    rb_result = self._run_rb_governor(
                        train_df, val_df, phase2_result)
                    results["phase3"] = rb_result
                    results["phase4"] = rb_result
                else:
                    phase3_result = self._load_phase3_outputs()
                    results["phase4"] = self._run_phase4(
                        train_df, val_df, phase3_result, force=True)

            else:
                self._ensure_phase5_inputs()
                results["phase5"] = self._run_phase5()

            total_elapsed = time.monotonic() - pipeline_start
            logger.info("=" * 60)
            logger.info("Phase %d complete in %.2fs", phase, total_elapsed)
            logger.info("=" * 60)
            return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
        logger.info(
            "Output directories ready: %s, %s",
            _cfg.OUTPUTS_DIR,
            _cfg.REPORTS_DIR,
        )

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

        for direction in ("long", "short"):
            try:
                pool = Rule_Pool_Generator.load_pool(direction)
            except ValueError as exc:
                missing.append(f"{direction}: {exc}")
                continue

            if pool is None:
                missing.append(
                    f"{direction}: {_phase2_module._POOL_PATHS[direction]}")
                continue

            result[direction] = pool

        if missing:
            raise FileNotFoundError(
                "Phase 3 requires Phase 2 pool outputs for both directions. "
                f"Missing or invalid: {', '.join(missing)}"
            )

        return result

    def _load_phase3_outputs(self) -> dict[str, dict]:
        """Load Phase 3 rule sets for both directions."""
        result: dict[str, dict] = {}
        missing: list[str] = []

        for direction in ("long", "short"):
            try:
                rule_set = Rule_Set_Selector.load_rule_set(direction)
            except ValueError as exc:
                missing.append(f"{direction}: {exc}")
                continue

            if rule_set is None:
                missing.append(
                    f"{direction}: {_phase3_module._OUTPUT_PATHS[direction]}")
                continue

            result[direction] = rule_set

        if missing:
            raise FileNotFoundError(
                "Phase 4 requires Phase 3 rule-set outputs for both directions. "
                f"Missing or invalid: {', '.join(missing)}"
            )

        return result

    def _ensure_phase5_inputs(self) -> None:
        """Ensure Phase 4 outputs exist before running Phase 5 alone."""
        evaluator = OOS_Evaluator()
        strategies = evaluator.load_strategies()
        missing = [
            direction for direction in ("long", "short")
            if direction not in strategies
        ]
        if missing:
            raise FileNotFoundError(
                "Phase 5 requires Phase 4 optimized strategies for both directions. "
                f"Missing or invalid: {', '.join(missing)}"
            )

    def _load_and_split_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load train.csv and split into train/validation DataFrames.

        When ``SPLIT_MODE`` is ``purged_walk_forward``, also rebuilds CV folds
        (stored on ``self._cv_folds``) for Phases 2–3.
        """
        cached_split = self._load_cached_split_if_fresh()
        if cached_split is not None:
            train_df, val_df = cached_split
            self._cv_folds = self._rebuild_cv_folds_if_needed()
            logger.info(
                "Using cached train/validation split from %s and %s (mode=%s)",
                _cfg.TRAIN_70_PATH,
                _cfg.VALIDATION_30_PATH,
                _cfg.SPLIT_MODE,
            )
            return self._apply_debug_symbol_scope(train_df, val_df)

        logger.info("Loading training data from %s …", _cfg.TRAIN_CSV_PATH)
        loader = Data_Loader()
        train_full = loader.load_dataset(_cfg.TRAIN_CSV_PATH)
        logger.info(
            "Loaded %d rows, %d symbols",
            len(train_full),
            train_full["symbol"].nunique()
            if "symbol" in train_full.columns
            else 0,
        )

        splitter = Data_Splitter()
        split_label = (
            "purged walk-forward"
            if _cfg.split_mode_is_purged_walk_forward()
            else "holdout 70/30"
        )
        logger.info("Splitting train.csv (%s) …", split_label)

        train_df, val_df, cv_folds = splitter.split_and_persist(train_full)
        self._cv_folds = cv_folds

        logger.info(
            "Split complete: train=%d rows, val=%d rows, cv_folds=%s",
            len(train_df),
            len(val_df),
            len(cv_folds) if cv_folds else "n/a",
        )
        return self._apply_debug_symbol_scope(train_df, val_df)

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
    def _load_cached_split_if_fresh() -> tuple[pd.DataFrame, pd.DataFrame] | None:
        """Load cached split files when they are newer than the source CSV."""
        csv_path = _cfg.TRAIN_CSV_PATH
        train_path = _cfg.TRAIN_70_PATH
        val_path = _cfg.VALIDATION_30_PATH
        manifest_path = getattr(_cfg, "CV_FOLDS_MANIFEST_PATH", "")

        if not (
            os.path.exists(csv_path)
            and os.path.exists(train_path)
            and os.path.exists(val_path)
        ):
            return None

        try:
            csv_mtime = os.path.getmtime(csv_path)
            cache_mtime = min(
                os.path.getmtime(train_path),
                os.path.getmtime(val_path),
            )
            if _cfg.split_mode_is_purged_walk_forward():
                if not os.path.exists(manifest_path):
                    return None
                manifest = json.load(open(manifest_path, encoding="utf-8"))
                if manifest.get("split_mode") != _cfg.SPLIT_MODE:
                    return None
                cache_mtime = min(cache_mtime, os.path.getmtime(manifest_path))
        except OSError:
            return None

        if cache_mtime < csv_mtime:
            return None

        from gpu_fuzzy_trader.backtest.df_slim import downcast_numeric_df

        train_df = downcast_numeric_df(pd.read_parquet(train_path))
        val_df = downcast_numeric_df(pd.read_parquet(val_path))
        return train_df, val_df

    @staticmethod
    def _rebuild_cv_folds_if_needed() -> list | None:
        """Rebuild in-memory CV folds when using a cached purged split."""
        if not _cfg.split_mode_is_purged_walk_forward():
            return None

        from gpu_fuzzy_trader.validation.rolling_cv import (
            build_purged_walk_forward_folds,
            load_cv_folds_manifest,
        )

        manifest = load_cv_folds_manifest()
        if manifest is not None:
            ref = manifest.get("reference_rows")
            if ref is not None:
                _cfg.set_purged_wf_reference_rows(int(ref))

        loader = Data_Loader()
        train_full = loader.load_dataset(_cfg.TRAIN_CSV_PATH)
        if manifest is None or manifest.get("reference_rows") is None:
            _cfg.set_purged_wf_reference_rows(len(train_full))

        folds = build_purged_walk_forward_folds(train_full)
        return folds if folds else None

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
    ) -> dict:
        """
        Run Phase 2 (Rule Pool Generation) or skip if valid outputs exist.

        Returns
        -------
        dict
            ``{"long": [...], "short": [...]}``
        """

        phase_name = "Phase 2: Rule Pool Generation"
        start_ts = _now_iso()
        t0 = time.monotonic()

        pools: dict[str, list[dict]] = {}

        for direction in ("long", "short"):
            dir_phase_name = f"{phase_name} [{direction}]"
            dir_start_ts = _now_iso()
            dir_t0 = time.monotonic()

            if not force:
                existing_pool = Rule_Pool_Generator.skip_if_valid(direction)
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
                    continue

            # Get feature infos for this direction
            feature_infos = phase1_result.get(direction, [])
            if not feature_infos:
                logger.warning(
                    "Phase 2 [%s]: no features from Phase 1; skipping direction.",
                    direction,
                )
                pools[direction] = []
                continue

            # Run Phase 2 for this direction
            logger.info(
                "Running %s … (%d features from Phase 1)",
                dir_phase_name, len(feature_infos),
            )
            try:
                if _cfg.PHASE2_ISLAND_MODE == "cluster":
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
                    )
                else:
                    generator = Rule_Pool_Generator(
                        train_df=train_df,
                        feature_infos=feature_infos,
                        direction=direction,
                        val_df=val_df,
                        cv_folds=self._cv_folds,
                        seed=_cfg.PHASE2_SEED,
                    )
                    pool = generator.run()
            except Exception as exc:
                logger.error(
                    "Phase 2 [%s] failed: %s", direction, exc, exc_info=True
                )
                pool = []

            dir_elapsed = time.monotonic() - dir_t0
            _log_phase_entry(
                self._log_path, dir_phase_name, dir_start_ts, _now_iso(),
                dir_elapsed, skipped=False,
                result_summary={"pool_size": len(pool)},
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
    # Phase 3
    # ------------------------------------------------------------------

    def _run_phase3(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        phase2_result: dict[str, list[dict]],
        force: bool = False,
    ) -> dict[str, dict]:
        phase_name = "Phase 3: Rule Set Selection"
        start_ts = _now_iso()
        t0 = time.monotonic()

        if not force:
            existing = Rule_Set_Selector.skip_if_valid()
            if existing is not None:
                long_path = os.path.join(_cfg.OUTPUTS_DIR, "long.json")
                short_path = os.path.join(_cfg.OUTPUTS_DIR, "short.json")
                logger.info(
                    "Skipping %s: valid rule sets at %s and %s (%s)",
                    phase_name, long_path, short_path,
                    ", ".join(
                        "%s=%d rules" % (d, len(rs.get("rules_set", [])))
                        for d, rs in existing.items()
                    ),
                )
                elapsed = time.monotonic() - t0
                _log_phase_entry(
                    self._log_path, phase_name, start_ts, _now_iso(),
                    elapsed, skipped=True,
                    result_summary={
                        d: len(rs.get("rules_set", []))
                        for d, rs in existing.items()
                    },
                )
                return existing

        # Run Phase 3 per direction
        rule_sets: dict[str, dict] = {}

        for direction in ("long", "short"):
            dir_phase_name = f"{phase_name} [{direction}]"
            dir_start_ts = _now_iso()
            dir_t0 = time.monotonic()

            pool = phase2_result.get(direction, [])
            if not pool:
                logger.warning(
                    "Phase 3 [%s]: empty pool from Phase 2; skipping direction.",
                    direction,
                )
                continue

            # Add Phase 2 static TP/SL/capital_pct to pool entries if missing
            enriched_pool = [
                {
                    **entry,
                    "tp": float(entry.get("tp", _cfg.PHASE2_TP)),
                    "sl": float(entry.get("sl", _cfg.PHASE2_SL)),
                    "capital_pct": float(entry.get("capital_pct", _cfg.PHASE2_CAPITAL_PCT)),
                }
                for entry in pool
            ]

            logger.info(
                "Running %s … (pool_size=%d from Phase 2)",
                dir_phase_name, len(enriched_pool),
            )
            try:
                selector = Rule_Set_Selector(
                    train_df=train_df,
                    val_df=val_df,
                    pool=enriched_pool,
                    direction=direction,
                    cv_folds=self._cv_folds,
                )
                rule_set = selector.run()
            except Exception as exc:
                logger.error(
                    "Phase 3 [%s] failed: %s", direction, exc, exc_info=True
                )
                rule_set = None

            dir_elapsed = time.monotonic() - dir_t0
            if rule_set is not None:
                rule_sets[direction] = rule_set
                _log_phase_entry(
                    self._log_path, dir_phase_name, dir_start_ts, _now_iso(),
                    dir_elapsed, skipped=False,
                    result_summary={"rules": len(
                        rule_set.get("rules_set", []))},
                )
            else:
                _log_phase_entry(
                    self._log_path, dir_phase_name, dir_start_ts, _now_iso(),
                    dir_elapsed, skipped=False,
                    result_summary={"error": "phase failed"},
                )

        elapsed = time.monotonic() - t0
        _log_phase_entry(
            self._log_path, phase_name, start_ts, _now_iso(),
            elapsed, skipped=False,
            result_summary={
                d: len(rs.get("rules_set", []))
                for d, rs in rule_sets.items()
            },
        )
        return rule_sets

    # ------------------------------------------------------------------
    # Phase 4
    # ------------------------------------------------------------------

    def _run_phase4(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        phase3_result: dict[str, dict],
        force: bool = False,
    ) -> dict[str, dict]:
        """
        Run Phase 4 (Walk-Forward Risk Optimization) or skip if valid outputs exist.

        Returns
        -------
        dict[str, dict]
            {"long": optimized_rule_set, "short": optimized_rule_set}
        """
        phase_name = "Phase 4: Walk-Forward Risk Optimization"
        start_ts = _now_iso()
        t0 = time.monotonic()

        optimized: dict[str, dict] = {}

        for direction in ("long", "short"):
            dir_phase_name = f"{phase_name} [{direction}]"
            dir_start_ts = _now_iso()
            dir_t0 = time.monotonic()

            if not force:
                existing = WalkForwardRiskOptimizer.skip_if_valid(direction)
                if existing is not None:
                    out_path = os.path.join(
                        _cfg.OUTPUTS_DIR, "%s.json" % direction)
                    logger.info(
                        "Skipping %s: risk-optimized rules at %s (%d rules)",
                        dir_phase_name, out_path,
                        len(existing.get("rules_set", [])),
                    )
                    dir_elapsed = time.monotonic() - dir_t0
                    _log_phase_entry(
                        self._log_path, dir_phase_name, dir_start_ts, _now_iso(),
                        dir_elapsed, skipped=True,
                        result_summary={"rules": len(
                            existing.get("rules_set", []))},
                    )
                    optimized[direction] = existing
                    continue

            # Get Phase 3 rule set for this direction
            rule_set = phase3_result.get(direction)
            if rule_set is None:
                logger.warning(
                    "Phase 4 [%s]: no rule set from Phase 3; skipping direction.",
                    direction,
                )
                continue

            n_rules = len(rule_set.get("rules_set", []))
            if n_rules == 0:
                logger.warning(
                    "Phase 4 [%s]: Phase 3 produced 0 rules; skipping direction.",
                    direction,
                )
                _log_phase_entry(
                    self._log_path, dir_phase_name, dir_start_ts, _now_iso(),
                    time.monotonic() - dir_t0, skipped=True,
                    result_summary={"rules": 0,
                                    "reason": "empty_phase3_rules"},
                )
                continue

            use_grid = bool(getattr(_cfg, "PHASE4_GRID_ENABLED", True))
            logger.info(
                "Running %s … (%d rules from Phase 3, method=%s)",
                dir_phase_name, n_rules,
                "grid",
            )
            try:
                optimizer = WalkForwardRiskOptimizer(
                    val_df=val_df,
                    train_df=train_df,
                    rule_set=rule_set,
                    direction=direction,
                    
                )
                result = optimizer.train()
            except Exception as exc:
                logger.error(
                    "Phase 4 [%s] failed: %s", direction, exc, exc_info=True
                )
                result = None

            dir_elapsed = time.monotonic() - dir_t0
            if result is not None:
                optimized[direction] = result
                _log_phase_entry(
                    self._log_path, dir_phase_name, dir_start_ts, _now_iso(),
                    dir_elapsed, skipped=False,
                    result_summary={"rules": len(result.get("rules_set", []))},
                )
            else:
                _log_phase_entry(
                    self._log_path, dir_phase_name, dir_start_ts, _now_iso(),
                    dir_elapsed, skipped=False,
                    result_summary={"error": "phase failed"},
                )

        elapsed = time.monotonic() - t0
        _log_phase_entry(
            self._log_path, phase_name, start_ts, _now_iso(),
            elapsed, skipped=False,
            result_summary={
                d: len(rs.get("rules_set", []))
                for d, rs in optimized.items()
            },
        )
        return optimized

    # ------------------------------------------------------------------
    # RB Governor (replaces Phase 3 + Phase 4)
    # ------------------------------------------------------------------

    def _run_rb_governor(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        phase2_result: dict[str, list[dict]],
    ) -> dict[str, dict]:
        """Run the RB Governor pipeline (Phase 3 + Phase 4 replacement).

        Returns a dict keyed by direction, each value being the strategy dict
        written to disk (containing ``direction`` and ``rules_set``).  The
        shape matches the legacy Phase 4 output so Phase 5 can load the
        generated ``{direction}.json`` files unchanged.
        """
        phase_name = "RB Governor (Phase 3+4 replacement)"
        start_ts = _now_iso()
        t0 = time.monotonic()

        directions = tuple(
            d for d in ("long", "short") if phase2_result.get(d)
        )
        logger.info(
            "Running %s … (directions=%s, pools=%s)",
            phase_name,
            list(directions),
            {d: len(phase2_result.get(d, [])) for d in directions},
        )

        try:
            strategies = _rb_governor_module.run_rb_governor_pipeline(
                train_df=train_df,
                val_df=val_df,
                pools=phase2_result,
                directions=directions,
                output_dir=_cfg.OUTPUTS_DIR,
            )
        except Exception as exc:
            logger.error("RB Governor failed: %s", exc, exc_info=True)
            strategies = {}

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
            Directions produced in the current run's Phase 3. An empty frozenset
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
            evaluator = OOS_Evaluator()
            result = evaluator.run(allowed_directions=allowed_directions)
        except Exception as exc:
            logger.error("Phase 5 failed: %s", exc, exc_info=True)
            result = {}

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
            },
        )
        return result


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
        help="Run only one phase instead of the full pipeline.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip phases 1–4 when valid outputs already exist (default: full rerun).",
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
