"""
run_pipeline.py — Pipeline_Orchestrator

Top-level orchestrator for the GPU-Fuzzy Trading Pipeline.

Execution order:
  1. Create output directories (outputs/, outputs/reports/)
  2. Load and prepare data (Data_Loader + Data_Splitter)
  3. Phase 1: Feature_Selector (skip if valid outputs exist)
  4. Phase 2: Rule_Pool_Generator for both directions (skip if valid outputs exist)
  5. Phase 3: Rule_Set_Selector for both directions (skip if valid outputs exist)
  6. Phase 4: RL_Agent for both directions (skip if valid outputs exist)
  7. Phase 5: OOS_Evaluator (always runs)

Skip logic per phase:
  - Validate output files before skipping.
  - Re-run phase if validation fails.

Logging:
  - Log start time, end time, and elapsed duration for each phase.
  - Save structured log to outputs/pipeline.log as JSON lines.

Entry point:
  python -m gpu_fuzzy_trader.run_pipeline

Requirements: 13.1, 13.2, 13.3, 13.4, 13.5
"""

from __future__ import annotations

from gpu_fuzzy_trader._jax_env import configure_jax_env

configure_jax_env()

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.data.loader import Data_Loader
from gpu_fuzzy_trader.data.splitter import Data_Splitter
from gpu_fuzzy_trader.features.selector import Feature_Selector
from gpu_fuzzy_trader.phases.phase2_rule_pool import Rule_Pool_Generator
from gpu_fuzzy_trader.phases.phase3_rule_set import Rule_Set_Selector
from gpu_fuzzy_trader.phases.phase4_rl_optimizer import RL_Agent
from gpu_fuzzy_trader.phases.phase5_oos import OOS_Evaluator

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_PIPELINE_LOG_PATH = os.path.join(_cfg.OUTPUTS_DIR, "pipeline.log")


# ---------------------------------------------------------------------------
# Phase timing helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


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
        fh.write(json.dumps(entry) + "\n")

    status = "SKIPPED" if skipped else "COMPLETED"
    logger.info(
        "[%s] %s in %.2fs",
        status,
        phase_name,
        elapsed_seconds,
    )


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

    def __init__(self) -> None:
        self._log_path = _PIPELINE_LOG_PATH

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """
        Run all pipeline phases in order.

        Returns
        -------
        dict
            Results from each phase, keyed by phase name.  Each value is
            the primary output of that phase (feature lists, pools, rule
            sets, OOS metrics, etc.).
        """
        logger.info("=" * 60)
        logger.info("GPU-Fuzzy Trading Pipeline — starting")
        logger.info("=" * 60)

        pipeline_start = time.monotonic()
        results: dict[str, Any] = {}

        # ------------------------------------------------------------------
        # Step 0: Create output directories
        # ------------------------------------------------------------------
        self._create_output_dirs()

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
        phase1_result = self._run_phase1(train_df)
        results["phase1"] = phase1_result

        # ------------------------------------------------------------------
        # Phase 2: Rule Pool Generation
        # ------------------------------------------------------------------
        phase2_result = self._run_phase2(train_df, phase1_result)
        results["phase2"] = phase2_result

        # Check if Phase 2 produced any rules; if not, skip Phases 3 and 4
        long_pool = phase2_result.get("long", [])
        short_pool = phase2_result.get("short", [])
        pool_empty = (not long_pool) and (not short_pool)

        if pool_empty:
            logger.warning(
                "Phase 2 produced no rules for either direction. "
                "Skipping Phases 3 and 4."
            )
            results["phase3"] = {}
            results["phase4"] = {}
        else:
            # ------------------------------------------------------------------
            # Phase 3: Rule Set Selection
            # ------------------------------------------------------------------
            phase3_result = self._run_phase3(train_df, val_df, phase2_result)
            results["phase3"] = phase3_result

            # ------------------------------------------------------------------
            # Phase 4: RL Risk Optimization
            # ------------------------------------------------------------------
            phase4_result = self._run_phase4(train_df, val_df, phase3_result)
            results["phase4"] = phase4_result

        # ------------------------------------------------------------------
        # Phase 5: Out-of-Sample Evaluation (always runs)
        # ------------------------------------------------------------------
        phase5_result = self._run_phase5()
        results["phase5"] = phase5_result

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        total_elapsed = time.monotonic() - pipeline_start
        logger.info("=" * 60)
        logger.info("Pipeline complete in %.2fs", total_elapsed)
        logger.info("=" * 60)

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_output_dirs(self) -> None:
        """Create outputs/ and outputs/reports/ directories if they don't exist."""
        os.makedirs(_cfg.OUTPUTS_DIR, exist_ok=True)
        os.makedirs(_cfg.REPORTS_DIR, exist_ok=True)
        logger.info(
            "Output directories ready: %s, %s",
            _cfg.OUTPUTS_DIR,
            _cfg.REPORTS_DIR,
        )

    def _load_and_split_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load train.csv and split into train/validation DataFrames.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            (train_df, val_df)
        """
        logger.info("Loading training data from %s …", _cfg.TRAIN_CSV_PATH)
        loader = Data_Loader()
        train_full = loader.load_dataset(_cfg.TRAIN_CSV_PATH)
        logger.info(
            "Loaded %d rows, %d symbols",
            len(train_full),
            train_full["symbol"].nunique() if "symbol" in train_full.columns else 0,
        )

        logger.info("Splitting into train (75%%) / validation (25%%) …")
        splitter = Data_Splitter()
        train_df, val_df = splitter.split_and_persist(train_full)
        logger.info(
            "Split complete: train=%d rows, val=%d rows",
            len(train_df),
            len(val_df),
        )
        return train_df, val_df

    # ------------------------------------------------------------------
    # Phase 1
    # ------------------------------------------------------------------

    def _run_phase1(self, train_df: pd.DataFrame) -> dict[str, list[dict]]:
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

        # Try to skip
        existing = Feature_Selector.skip_if_valid()
        if existing is not None:
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
            result = selector.run(train_df)
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
    ) -> dict[str, list[dict]]:
        """
        Run Phase 2 (Rule Pool Generation) or skip if valid outputs exist.

        Returns
        -------
        dict[str, list[dict]]
            {"long": [...], "short": [...]}
        """
        phase_name = "Phase 2: Rule Pool Generation"
        start_ts = _now_iso()
        t0 = time.monotonic()

        pools: dict[str, list[dict]] = {}

        for direction in ("long", "short"):
            dir_phase_name = f"{phase_name} [{direction}]"
            dir_start_ts = _now_iso()
            dir_t0 = time.monotonic()

            # Try to skip
            existing_pool = Rule_Pool_Generator.skip_if_valid(direction)
            if existing_pool is not None:
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
            logger.info("Running %s …", dir_phase_name)
            try:
                generator = Rule_Pool_Generator(
                    train_df=train_df,
                    feature_infos=feature_infos,
                    direction=direction,
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
    ) -> dict[str, dict]:
        """
        Run Phase 3 (Rule Set Selection) or skip if valid outputs exist.

        Returns
        -------
        dict[str, dict]
            {"long": rule_set_dict, "short": rule_set_dict}
            (only directions with valid pools are included)
        """
        phase_name = "Phase 3: Rule Set Selection"
        start_ts = _now_iso()
        t0 = time.monotonic()

        # Try to skip (both or partial)
        existing = Rule_Set_Selector.skip_if_valid()
        if existing is not None:
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

            logger.info("Running %s …", dir_phase_name)
            try:
                selector = Rule_Set_Selector(
                    train_df=train_df,
                    val_df=val_df,
                    pool=enriched_pool,
                    direction=direction,
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
                    result_summary={"rules": len(rule_set.get("rules_set", []))},
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
    ) -> dict[str, dict]:
        """
        Run Phase 4 (RL Risk Optimization) or skip if valid outputs exist.

        Returns
        -------
        dict[str, dict]
            {"long": optimized_rule_set, "short": optimized_rule_set}
        """
        phase_name = "Phase 4: RL Risk Optimization"
        start_ts = _now_iso()
        t0 = time.monotonic()

        optimized: dict[str, dict] = {}

        for direction in ("long", "short"):
            dir_phase_name = f"{phase_name} [{direction}]"
            dir_start_ts = _now_iso()
            dir_t0 = time.monotonic()

            # Try to skip
            existing = RL_Agent.skip_if_valid(direction)
            if existing is not None:
                dir_elapsed = time.monotonic() - dir_t0
                _log_phase_entry(
                    self._log_path, dir_phase_name, dir_start_ts, _now_iso(),
                    dir_elapsed, skipped=True,
                    result_summary={"rules": len(existing.get("rules_set", []))},
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

            logger.info("Running %s …", dir_phase_name)
            try:
                agent = RL_Agent(
                    train_df=train_df,
                    val_df=val_df,
                    rule_set=rule_set,
                    direction=direction,
                )
                result = agent.train()
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
    # Phase 5
    # ------------------------------------------------------------------

    def _run_phase5(self) -> dict[str, dict]:
        """
        Run Phase 5 (Out-of-Sample Evaluation). Always runs.

        Returns
        -------
        dict[str, dict]
            OOS metrics keyed by direction.
        """
        phase_name = "Phase 5: Out-of-Sample Evaluation"
        start_ts = _now_iso()
        t0 = time.monotonic()

        logger.info("Running %s …", phase_name)
        try:
            evaluator = OOS_Evaluator()
            result = evaluator.run()
        except Exception as exc:
            logger.error("Phase 5 failed: %s", exc, exc_info=True)
            result = {}

        elapsed = time.monotonic() - t0
        _log_phase_entry(
            self._log_path, phase_name, start_ts, _now_iso(),
            elapsed, skipped=False,
            result_summary={
                d: {
                    "total_return_pct": m.get("total_return_pct", 0.0),
                    "executed_trades": m.get("executed_trades", 0),
                }
                for d, m in result.items()
            },
        )
        return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full pipeline from the command line."""
    orchestrator = Pipeline_Orchestrator()
    try:
        results = orchestrator.run()
        # Print a brief summary to stdout
        print("\n=== Pipeline Summary ===")
        phase5 = results.get("phase5", {})
        if phase5:
            for direction, metrics in phase5.items():
                print(
                    f"  {direction.upper()}: "
                    f"return={metrics.get('total_return_pct', 0.0):.2f}%, "
                    f"trades={metrics.get('executed_trades', 0)}, "
                    f"drawdown={metrics.get('max_drawdown_pct', 0.0):.2f}%"
                )
        else:
            print("  No OOS results (check outputs/pipeline.log for details)")
        print(f"\nStructured log saved to: {_PIPELINE_LOG_PATH}")
    except Exception as exc:
        logger.error("Pipeline failed with unhandled exception: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
