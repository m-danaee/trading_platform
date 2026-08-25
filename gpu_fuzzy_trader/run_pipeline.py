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

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from typing import Any
import uuid
from pathlib import Path

from gpu_fuzzy_trader._jax_env import configure_jax_env

configure_jax_env()

import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader import rb_governor as _rb_governor_module
from gpu_fuzzy_trader.data.loader import Data_Loader, validate_context_columns
from gpu_fuzzy_trader.data.splitter import Data_Splitter, load_cached_split_if_fresh
from gpu_fuzzy_trader.features import selector as _selector_module
from gpu_fuzzy_trader.features.fuzzy_scaling import (
    apply_fuzzy_feature_scaling,
    fit_fuzzy_feature_scaling,
)
from gpu_fuzzy_trader.features.selector import Feature_Selector
from gpu_fuzzy_trader.mtf import (
    DEFAULT_MIN_EVIDENCE_STRENGTH,
    DEFAULT_RETENTION_FLOOR,
    DEFAULT_RETENTION_TARGET,
    DEFAULT_V_HWC_LONG,
    DEFAULT_V_HWC_SHORT,
    DEFAULT_V_MWC_LONG,
    DEFAULT_V_MWC_SHORT,
    HierarchicalStrategyCandidate,
    build_master_temporal_folds,
    export_fold_boundaries,
    load_mtf_rule_archive,
    save_mtf_rule_archive,
)
from gpu_fuzzy_trader.mtf.discovery import (
    LayerDiscoveryResult,
    discovery_purge_minutes,
    discovery_search_identity,
    discover_directional_layer,
)
from gpu_fuzzy_trader.mtf.runtime import (
    attach_frozen_layer_scores,
    attach_oof_layer_scores,
    evaluate_candidate_frame,
)
from gpu_fuzzy_trader.phases import phase2_rule_pool as _phase2_module
from gpu_fuzzy_trader.phases import phase5_oos as _phase5_module
from gpu_fuzzy_trader.phases.phase2_rule_pool import Rule_Pool_Generator
from gpu_fuzzy_trader.phases.phase5_oos import OOS_Evaluator
from gpu_fuzzy_trader.reporting import reporter as _reporter_module
from gpu_fuzzy_trader.research_integrity import (
    ExperimentLedger,
    count_trials,
    sha256_file,
    write_dataset_manifests,
)
from gpu_fuzzy_trader.research_profile import ResearchProfile


def _dataframe_sha256(frame: pd.DataFrame | None) -> str:
    """Return a stable hash of a dataframe's schema and values."""
    if frame is None:
        return ""
    columns = sorted(str(column) for column in frame.columns)
    canonical = frame.loc[:, columns]
    payload = pd.util.hash_pandas_object(canonical, index=True).to_numpy().tobytes()
    schema = [(column, str(canonical[column].dtype)) for column in columns]
    return hashlib.sha256(
        json.dumps(schema, sort_keys=True).encode("utf-8") + payload
    ).hexdigest()


def _dataframe_schema_sha256(frame: pd.DataFrame | None) -> str:
    """Hash column names and dtypes without including row values."""
    if frame is None:
        return ""
    schema = [
        (str(column), str(frame[column].dtype))
        for column in frame.columns
    ]
    return hashlib.sha256(
        json.dumps(schema, sort_keys=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _git_commit_id() -> str:
    """Read the current commit without making git state part of execution."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


_MTF_SCORE_COLUMNS = (
    "mtf_hwc_direction",
    "mtf_hwc_strength",
    "mtf_mwc_direction",
    "mtf_mwc_strength",
    "_mtf_oof_available",
)


def _merge_mtf_score_columns(
    base: pd.DataFrame,
    scored: pd.DataFrame,
) -> pd.DataFrame:
    """Merge computed MTF score columns by normalized (symbol, datetime) keys."""
    required = {"datetime", "symbol", *_MTF_SCORE_COLUMNS}
    missing = sorted(required - set(scored.columns))
    if missing:
        raise ValueError(f"MTF score frame is missing columns: {missing}")
    if base.empty:
        return base.copy()

    left = base.copy()
    right = scored.loc[:, ["datetime", "symbol", *_MTF_SCORE_COLUMNS]].copy()
    for frame in (left, right):
        frame["datetime"] = pd.to_datetime(
            frame["datetime"], errors="raise", utc=True
        ).dt.tz_localize(None)
        if str(frame["symbol"].dtype) != "category":
            frame["symbol"] = frame["symbol"].astype("category")
    if left.duplicated(["datetime", "symbol"]).any():
        raise ValueError("MTF input contains duplicate (datetime, symbol) rows")
    if right.duplicated(["datetime", "symbol"]).any():
        raise ValueError("MTF score frame contains duplicate (datetime, symbol) rows")

    left["_mtf_original_order"] = np.arange(len(left), dtype=np.int64)
    left = left.drop(columns=list(_MTF_SCORE_COLUMNS), errors="ignore")
    merged = left.merge(
        right,
        on=["datetime", "symbol"],
        how="left",
        sort=False,
        validate="one_to_one",
    )
    return (
        merged.sort_values("_mtf_original_order", kind="mergesort")
        .drop(columns=["_mtf_original_order"])
        .reset_index(drop=True)
    )


def _merge_mtf_lwc_runtime_columns(
    base: pd.DataFrame,
    scored: pd.DataFrame,
) -> pd.DataFrame:
    """Merge causal LWC features and MTF scores into a raw research frame.

    The runtime builder also computes HWC/MWC feature columns for applying
    their frozen rules. Those raw higher-timeframe features must not leak into
    LWC discovery; LWC receives its own causal features plus upstream scores.
    """
    required = {"datetime", "symbol", *_MTF_SCORE_COLUMNS}
    missing = sorted(required - set(scored.columns))
    if missing:
        raise ValueError(f"MTF runtime frame is missing columns: {missing}")
    if base.empty:
        return base.copy()

    left = base.copy()
    # Discard precomputed feature columns from legacy/enriched loaders. The
    # canonical MTF path derives the LWC feature family below from raw OHLCV,
    # while labels, exact barrier outcomes, and loader internals remain
    # available to the backtest and split contracts.
    base_columns = [
        column
        for column in left.columns
        if column in {"datetime", "symbol", "open", "high", "low", "close", "volume"}
        or str(column).startswith("label_")
        or str(column).startswith("_")
    ]
    left = left.loc[:, base_columns].copy()
    right_columns = [
        column
        for column in scored.columns
        if column in _MTF_SCORE_COLUMNS or str(column).startswith("lwc_")
    ]
    right = scored.loc[:, ["datetime", "symbol", *right_columns]].copy()
    for frame in (left, right):
        frame["datetime"] = pd.to_datetime(
            frame["datetime"], errors="raise", utc=True
        ).dt.tz_localize(None)
        if str(frame["symbol"].dtype) != "category":
            frame["symbol"] = frame["symbol"].astype("category")
    if left.duplicated(["datetime", "symbol"]).any():
        raise ValueError("MTF input contains duplicate (datetime, symbol) rows")
    if right.duplicated(["datetime", "symbol"]).any():
        raise ValueError("MTF runtime frame contains duplicate (datetime, symbol) rows")

    runtime_columns = list(dict.fromkeys(
        list(_MTF_SCORE_COLUMNS)
        + [column for column in right_columns if str(column).startswith("lwc_")]
    ))
    left = left.drop(columns=runtime_columns, errors="ignore")
    left["_mtf_original_order"] = np.arange(len(left), dtype=np.int64)
    merged = left.merge(
        right,
        on=["datetime", "symbol"],
        how="left",
        sort=False,
        validate="one_to_one",
    )
    return (
        merged.sort_values("_mtf_original_order", kind="mergesort")
        .drop(columns=["_mtf_original_order"])
        .reset_index(drop=True)
    )


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
        "cfg_mtf_archive_dir": _cfg.MTF_ARCHIVE_DIR,
        "cfg_mtf_archive_paths": _cfg.MTF_ARCHIVE_PATHS.copy(),
        "cfg_mtf_manifest_path": _cfg.MTF_MANIFEST_PATH,
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
        _cfg.MTF_ARCHIVE_DIR = os.path.join(output_root, "rule_archives")
        _cfg.MTF_ARCHIVE_PATHS = {
            timeframe: os.path.join(
                _cfg.MTF_ARCHIVE_DIR, timeframe, f"{timeframe}_rules.json"
            )
            for timeframe in ("hwc", "mwc", "lwc")
        }
        _cfg.MTF_MANIFEST_PATH = os.path.join(output_root, "mtf_manifest.json")

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
        _cfg.MTF_ARCHIVE_DIR = previous_state["cfg_mtf_archive_dir"]
        _cfg.MTF_ARCHIVE_PATHS = previous_state["cfg_mtf_archive_paths"]
        _cfg.MTF_MANIFEST_PATH = previous_state["cfg_mtf_manifest_path"]
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


_PHASE2_RESUME_IDENTITY_VERSION = 1
_PHASE2_RESUME_CODE_PATHS = (
    "run_pipeline.py",
    "backtest/cpu_engine.py",
    "backtest/gpu_engine.py",
    "evolution/evox_runner.py",
    "phases/phase2_rule_pool.py",
    "phases/phase2_sparse_encoding.py",
    "phases/phase2_support.py",
    "phases/rule_identity.py",
    "validation/fold_gates.py",
    "mtf/cross_fitting.py",
)



def _identity_value(value: Any) -> Any:
    """Convert configuration values to a stable, JSON-safe identity form."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else repr(numeric)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _identity_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_identity_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_identity_value(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, np.ndarray):
        return _identity_value(value.tolist())
    return str(value)


def _phase2_frame_identity(frame: pd.DataFrame | None) -> str | None:
    """Return a canonical content hash for a Phase 2 split frame."""
    if frame is None:
        return None
    sort_columns = [
        column for column in ("datetime", "symbol") if column in frame.columns
    ]
    canonical = (
        frame.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
        if sort_columns
        else frame.reset_index(drop=True)
    )
    schema = {
        "columns": [str(column) for column in canonical.columns],
        "dtypes": [str(dtype) for dtype in canonical.dtypes],
        "rows": int(len(canonical)),
    }
    digest = hashlib.sha256(
        json.dumps(schema, sort_keys=True).encode("utf-8")
    )
    row_hashes = pd.util.hash_pandas_object(
        canonical,
        index=False,
        categorize=True,
    ).to_numpy(dtype=np.uint64, copy=False)
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def _phase2_cv_structure(cv_folds: list | None) -> list[dict[str, Any]]:
    """Capture CV boundaries without duplicating full frame contents in RAM."""
    if not cv_folds:
        return []
    fields = (
        "fold_id",
        "train_end_bar",
        "valid_start_bar",
        "valid_end_bar",
        "n_train_rows",
        "n_valid_rows",
        "is_holdout",
    )
    return [
        {
            field: _identity_value(
                fold.get(field) if isinstance(fold, dict) else getattr(fold, field)
            )
            for field in fields
        }
        for fold in cv_folds
    ]


def _phase2_resume_identity(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    feature_infos: list[dict],
    direction: str,
    cv_folds: list | None,
) -> str:
    """Bind reusable Phase 2 pools to data, selection, config, and code."""
    package_root = Path(__file__).resolve().parent
    config_snapshot = {
        name: _identity_value(getattr(_cfg, name))
        for name in sorted(dir(_cfg))
        if name.isupper() and not name.startswith("_")
        and not callable(getattr(_cfg, name))
    }
    payload = {
        "version": _PHASE2_RESUME_IDENTITY_VERSION,
        "direction": direction,
        "train": _phase2_frame_identity(train_df),
        "validation": _phase2_frame_identity(val_df),
        "feature_infos": _identity_value(feature_infos),
        "cv_structure": _phase2_cv_structure(cv_folds),
        "context_contract_digest": _cfg.context_contract_digest(),
        "config": config_snapshot,
        "code": {
            relative_path: sha256_file(package_root / relative_path)
            for relative_path in _PHASE2_RESUME_CODE_PATHS
        },
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    perm = _cfg.context_permission_column(direction)
    trig = _cfg.context_trigger_column(direction)
    missing = [c for c in (perm, trig) if c not in frame.columns]
    if missing:
        return {
            "eligible_rows": None,
            "total_rows": int(len(frame)),
            "coverage_pct": None,
            "permission_rows": None,
            "trigger_rows": None,
            "permission_only_rows": None,
            "trigger_only_rows": None,
            "neither_rows": None,
            "by_symbol": {},
            "by_symbol_detail": {},
            "missing_columns": missing,
        }
    perm_mask = frame[perm].to_numpy() == 1
    trig_mask = frame[trig].to_numpy() == 1
    eligible = perm_mask & trig_mask
    permission_only = perm_mask & ~trig_mask
    trigger_only = ~perm_mask & trig_mask
    neither = ~perm_mask & ~trig_mask

    by_symbol: dict[str, int] = {}
    by_symbol_detail: dict[str, dict[str, Any]] = {}
    if "symbol" in frame.columns:
        grouped = frame.groupby("symbol", sort=True, observed=False)
        for symbol, group in grouped:
            symbol_permission = group[perm].to_numpy() == 1
            symbol_trigger = group[trig].to_numpy() == 1
            symbol_eligible = symbol_permission & symbol_trigger
            symbol_name = str(symbol)
            by_symbol[symbol_name] = int(symbol_eligible.sum())
            by_symbol_detail[symbol_name] = {
                "total_rows": int(len(group)),
                "permission_rows": int(symbol_permission.sum()),
                "trigger_rows": int(symbol_trigger.sum()),
                "eligible_rows": int(symbol_eligible.sum()),
                "permission_only_rows": int(
                    (symbol_permission & ~symbol_trigger).sum()
                ),
                "trigger_only_rows": int(
                    (~symbol_permission & symbol_trigger).sum()
                ),
                "coverage_pct": (
                    int(symbol_eligible.sum()) / max(len(group), 1) * 100.0
                ),
            }
    else:
        by_symbol["<all>"] = int(eligible.sum())
        by_symbol_detail["<all>"] = {
            "total_rows": int(len(frame)),
            "permission_rows": int(perm_mask.sum()),
            "trigger_rows": int(trig_mask.sum()),
            "eligible_rows": int(eligible.sum()),
            "permission_only_rows": int(permission_only.sum()),
            "trigger_only_rows": int(trigger_only.sum()),
            "coverage_pct": (
                int(eligible.sum()) / max(len(frame), 1) * 100.0
            ),
        }

    eligible_rows = int(eligible.sum())
    return {
        "eligible_rows": eligible_rows,
        "total_rows": int(len(frame)),
        "coverage_pct": eligible_rows / max(len(frame), 1) * 100.0,
        "permission_rows": int(perm_mask.sum()),
        "trigger_rows": int(trig_mask.sum()),
        "permission_only_rows": int(permission_only.sum()),
        "trigger_only_rows": int(trigger_only.sum()),
        "neither_rows": int(neither.sum()),
        "by_symbol": by_symbol,
        "by_symbol_detail": by_symbol_detail,
        "missing_columns": [],
    }


def _context_coverage_report(
    train_df: pd.DataFrame,
    val_fitness_df: pd.DataFrame,
    val_selection_df: pd.DataFrame,
) -> dict[str, dict[str, dict[str, object]]]:
    """Return split-aware context coverage for both trading directions."""
    return {
        split_name: {
            direction: _context_coverage_for_direction(frame, direction)
            for direction in ("long", "short")
        }
        for split_name, frame in {
            "train": train_df,
            "validation_fitness": val_fitness_df,
            "validation_selection": val_selection_df,
        }.items()
    }


def context_floor_failures(
    stats: dict[str, Any],
    *,
    support_floor: int | None = None,
    pool_floor: int | None = None,
    validation_floor: int | None = None,
) -> list[str]:
    """Return mathematically impossible trade-floor failures for coverage."""
    eligible = stats.get("eligible_rows")
    if eligible is None:
        return [
            "missing_context_columns:"
            + ",".join(str(value) for value in stats.get("missing_columns", []))
        ]
    eligible_rows = int(eligible)
    failures: list[str] = []
    if support_floor is not None and eligible_rows < int(support_floor):
        failures.append(
            f"eligible_rows={eligible_rows}<min_trade_support={int(support_floor)}"
        )
    if pool_floor is not None and eligible_rows < int(pool_floor):
        failures.append(
            f"eligible_rows={eligible_rows}<min_trade_pool_floor={int(pool_floor)}"
        )
    if validation_floor is not None and eligible_rows < int(validation_floor):
        failures.append(
            f"eligible_rows={eligible_rows}<validation_trade_floor={int(validation_floor)}"
        )
    return failures


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
            "The legacy path must use context contract with an enriched "
            "train/test pair; "
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
    """Return test-split metrics from a direct or wrapped Phase 5 result."""
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
                "phase2_long_pool.json.identity.json",
                "phase2_short_pool.json.identity.json",
                "long.json",
                "short.json",
            ):
                (output_root / name).unlink(missing_ok=True)
            for path in Path(_cfg.REPORTS_DIR).glob(
                "rb_governor_*_report.json"
            ):
                path.unlink(missing_ok=True)
            for timeframe in ("hwc", "mwc", "lwc"):
                archive_path = (
                    output_root / "rule_archives" / timeframe
                    / f"{timeframe}_rules.json"
                )
                archive_path.unlink(missing_ok=True)
            (output_root / "mtf_manifest.json").unlink(missing_ok=True)
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
                if self._should_use_mtf_pipeline(train_df):
                    mtf_results = self._run_mtf_pipeline(
                        train_df=train_df,
                        val_df=val_df,
                        force=force,
                    )
                    results.update(mtf_results)
                    total_elapsed = time.monotonic() - pipeline_start
                    self._record_research_integrity(results, total_elapsed)
                    self._finish_run("completed")
                    logger.info("Hierarchical MTF pipeline complete in %.2fs", total_elapsed)
                    return results
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
                matrix_rows = self._write_candidate_fold_matrix(
                    train_df,
                    phase2_result,
                    rb_result,
                )
                trial_counters = self._trial_counters(
                    phase1_result,
                    phase2_result,
                    rb_result,
                )
                results["strategy_stability"] = self.run_stability_report(
                    train_df,
                    rb_result,
                    trial_count=max(1, sum(trial_counters.values())),
                    trial_count_ledger=sum(trial_counters.values()),
                    candidate_fold_matrix=matrix_rows,
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
        directory.  The artifacts must match the current masked training input
        and Phase 1 contract; stale or copied artifacts fail closed.  Does not
        re-run feature selection.

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
                if self._should_use_mtf_pipeline(train_df):
                    mtf_results = self._run_mtf_from_phase2(
                        train_df=train_df,
                        val_df=val_df,
                        force=force,
                    )
                    results.update(mtf_results)
                    total_elapsed = time.monotonic() - pipeline_start
                    self._record_research_integrity(results, total_elapsed)
                    self._finish_run("completed")
                    logger.info(
                        "Hierarchical MTF pipeline (from phase 2) complete in %.2fs", total_elapsed)
                    return results
                val_fitness_df, val_selection_df = self._validation_scoring_frames(
                    val_df)

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

                phase1_train_df = self._mask_train_df_for_phase1(train_df)
                phase1_result = self._load_phase1_outputs(phase1_train_df)
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
                matrix_rows = self._write_candidate_fold_matrix(
                    train_df,
                    phase2_result,
                    rb_result,
                )
                trial_counters = self._trial_counters(
                    phase1_result,
                    phase2_result,
                    rb_result,
                )
                results["strategy_stability"] = self.run_stability_report(
                    train_df,
                    rb_result,
                    trial_count=max(1, sum(trial_counters.values())),
                    trial_count_ledger=sum(trial_counters.values()),
                    candidate_fold_matrix=matrix_rows,
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
        Run a single pipeline phase from identity-validated disk prerequisites.

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
                if self._should_use_mtf_pipeline(train_df):
                    mtf_folds = build_master_temporal_folds(
                        train_df,
                        n_folds=int(_cfg.MTF_MAX_FOLDS),
                        embargo_minutes=discovery_purge_minutes("hwc"),
                    )
                    hwc_rules = self.run_phase1_hwc(
                        train_df, folds=mtf_folds, force=True)
                    mwc_rules = self.run_phase1_mwc(
                        train_df, hwc_rules=hwc_rules, folds=mtf_folds, force=True)
                    hwc_discovery = getattr(self, "_mtf_hwc_discovery", None)
                    mwc_discovery = getattr(self, "_mtf_mwc_discovery", None)
                    lwc_train_scored = self._build_mtf_lwc_training_frame(
                        train_df, hwc_discovery, mwc_discovery)
                    oof_avail = lwc_train_scored["_mtf_oof_available"].fillna(
                        False).astype(bool)
                    lwc_train_df = lwc_train_scored.loc[oof_avail].reset_index(
                        drop=True)
                    phase1_train_df = self._mask_train_df_for_phase1(
                        lwc_train_df)
                    results["phase1"] = self._run_phase1(
                        phase1_train_df, force=True, val_df=None)
                    results["mtf_hwc"] = hwc_rules
                    results["mtf_mwc"] = mwc_rules
                else:
                    phase1_train_df = self._mask_train_df_for_phase1(train_df)
                    results["phase1"] = self._run_phase1(
                        phase1_train_df, force=True, val_df=None,
                    )

            elif phase == 2:
                train_df, val_df = self._load_and_split_data()
                self._validate_active_configuration(train_df)
                if self._should_use_mtf_pipeline(train_df):
                    mtf_folds = build_master_temporal_folds(
                        train_df,
                        n_folds=int(_cfg.MTF_MAX_FOLDS),
                        embargo_minutes=discovery_purge_minutes("hwc"),
                    )
                    hwc_rules = self.run_phase1_hwc(
                        train_df, folds=mtf_folds, force=False)
                    mwc_rules = self.run_phase1_mwc(
                        train_df, hwc_rules=hwc_rules, folds=mtf_folds, force=False)
                    hwc_discovery = getattr(self, "_mtf_hwc_discovery", None)
                    mwc_discovery = getattr(self, "_mtf_mwc_discovery", None)
                    val_fitness_df, val_selection_df = self._validation_scoring_frames(
                        val_df)
                    lwc_train_scored_all = self._build_mtf_lwc_training_frame(
                        train_df, hwc_discovery, mwc_discovery)
                    lwc_val_fitness_scored = self._build_mtf_lwc_validation_frame(
                        val_fitness_df, hwc_rules, mwc_rules, history_df=train_df)
                    oof_available = lwc_train_scored_all["_mtf_oof_available"].fillna(
                        False).astype(bool)
                    lwc_train_scored = lwc_train_scored_all.loc[oof_available].reset_index(
                        drop=True)
                    phase1_train_df = self._mask_train_df_for_phase1(
                        lwc_train_scored)
                    try:
                        phase1_result = self._load_phase1_outputs(
                            phase1_train_df)
                    except Exception:
                        phase1_result = self._run_phase1(
                            phase1_train_df, force=False, val_df=lwc_val_fitness_scored)
                    results["phase2"] = self._run_phase2(
                        phase1_train_df,
                        phase1_result,
                        force=True,
                        val_df=lwc_val_fitness_scored,
                        blocked_directions=frozenset(),
                    )
                    self._write_candidate_fold_matrix(
                        train_df,
                        results["phase2"],
                        None,
                    )
                else:
                    val_fitness_df, val_selection_df = self._validation_scoring_frames(
                        val_df)
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
                    phase1_train_df = self._mask_train_df_for_phase1(train_df)
                    phase1_result = self._load_phase1_outputs(phase1_train_df)
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
                    self._write_candidate_fold_matrix(
                        train_df,
                        results["phase2"],
                        None,
                    )

            elif phase in {3, 4}:
                train_df, val_df = self._load_and_split_data()
                self._validate_active_configuration(train_df)
                results["data"] = {
                    "train_rows": len(train_df),
                    "val_rows": len(val_df),
                }
                if self._should_use_mtf_pipeline(train_df):
                    mtf_folds = build_master_temporal_folds(
                        train_df,
                        n_folds=int(_cfg.MTF_MAX_FOLDS),
                        embargo_minutes=discovery_purge_minutes("hwc"),
                    )
                    hwc_rules = self.run_phase1_hwc(
                        train_df, folds=mtf_folds, force=False)
                    mwc_rules = self.run_phase1_mwc(
                        train_df, hwc_rules=hwc_rules, folds=mtf_folds, force=False)
                    lwc_path = Path(self._output_dir) / \
                        "rule_archives" / "lwc" / "lwc_rules.json"
                    from gpu_fuzzy_trader.mtf.archives import load_mtf_rule_archive
                    lwc_rules = load_mtf_rule_archive(
                        lwc_path) if lwc_path.exists() else []
                    candidates = self.run_mtf_composition(
                        lwc_rules=lwc_rules,
                        hwc_rules=hwc_rules,
                        mwc_rules=mwc_rules,
                        df=train_df,
                        folds=mtf_folds,
                    )
                    val_fitness_df, val_selection_df = self._validation_scoring_frames(
                        val_df)
                    rb_result = self._run_mtf_rb_governor(
                        train_df=train_df,
                        val_df=val_selection_df,
                        candidates=candidates,
                    )
                    results["rb_governor"] = rb_result
                    results["phase3"] = rb_result
                    results["phase4"] = rb_result
                else:
                    val_fitness_df, val_selection_df = self._validation_scoring_frames(
                        val_df)
                    phase1_train_df = self._mask_train_df_for_phase1(train_df)
                    phase1_result = self._load_phase1_outputs(phase1_train_df)
                    train_df, val_df = self._prune_splits_after_phase1(
                        train_df, val_df, phase1_result)
                    val_fitness_df, val_selection_df = self._prune_splits_after_phase1(
                        val_fitness_df, val_selection_df, phase1_result)
                    self._cv_folds = self._prune_cv_folds_after_phase1(
                        self._cv_folds, phase1_result)
                    phase2_result = self._load_phase2_outputs(
                        train_df, val_fitness_df, phase1_result,
                    )
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
                    "rb_governor": self._rb_status_summary(results.get("rb_governor", {})),
                }
                self._write_candidate_fold_matrix(
                    train_df,
                    results.get("phase2"),
                    results.get("rb_governor"),
                )

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

    @staticmethod
    def _trial_counters(
        phase1: dict[str, Any] | None = None,
        phase2: dict[str, Any] | None = None,
        rb: dict[str, Any] | None = None,
    ) -> dict[str, int]:
        """Build explicit trial counters for the research ledger.

        These counters describe alternatives that reached each report stage.
        They are kept separate so DSR does not silently reuse a rough artifact
        size estimate when the ledger is available.
        """
        feature_alternatives = sum(
            len(value) for value in (phase1 or {}).values()
            if isinstance(value, list)
        )
        rules_tested = sum(
            len(value) for value in (phase2 or {}).values()
            if isinstance(value, list)
        )
        selection_candidates = rules_tested
        hyperparameter_configs = 0
        if phase2:
            hyperparameter_configs += 1
        if rb:
            hyperparameter_configs += 1
        return {
            "feature_alternatives": int(feature_alternatives),
            "rules_tested": int(rules_tested),
            "hyperparameter_configs": int(hyperparameter_configs),
            "selection_candidates": int(selection_candidates),
        }

    @staticmethod
    def _write_candidate_fold_matrix(
        train_df: pd.DataFrame,
        phase2: dict[str, Any] | None,
        rb: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Persist deterministic Phase 2/RB candidate×fold score rows."""
        if not bool(getattr(_cfg, "MULTIPLICITY_REPORT_ENABLED", True)):
            return []
        from gpu_fuzzy_trader.validation.multiplicity import (
            write_candidate_fold_matrix,
        )
        from gpu_fuzzy_trader.validation.walk_forward_stability_report import (
            build_candidate_fold_matrix,
        )

        candidates: list[dict[str, Any]] = []

        def add_candidate(
            source: str,
            direction: str,
            candidate: Any,
        ) -> None:
            if not isinstance(candidate, dict):
                return
            entry = dict(candidate)
            entry["source"] = source
            entry["direction"] = direction
            if not entry.get("candidate_id"):
                identity = {
                    "chromosome": entry.get("chromosome"),
                    "conditions": entry.get("conditions"),
                    "rules_set": entry.get("rules_set"),
                    "strategy_id": entry.get("strategy_id"),
                }
                digest = hashlib.sha256(
                    json.dumps(
                        identity,
                        sort_keys=True,
                        default=str,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()[:16]
                entry["candidate_id"] = f"{source}:{direction}:{digest}"
            candidates.append(entry)

        for direction in sorted((phase2 or {})):
            values = (phase2 or {}).get(direction)
            if isinstance(values, list):
                for candidate in values:
                    add_candidate("phase2", str(direction), candidate)
        for direction in sorted((rb or {})):
            add_candidate("rb", str(direction), (rb or {}).get(direction))

        rows = build_candidate_fold_matrix(
            train_df,
            candidates,
            n_windows=int(getattr(_cfg, "MULTIPLICITY_STABILITY_WINDOWS", 3)),
        )
        if not rows:
            # Legacy/resumed artifacts can contain only their admission metrics.
            # Keep one auditable fold rather than dropping the candidate from
            # the ledger when a full frame re-evaluation is not possible.
            for candidate in candidates:
                objectives = candidate.get("objectives", {})
                validation = candidate.get("val_objectives", {})
                if isinstance(objectives, dict) and isinstance(validation, dict):
                    rows.append({
                        "candidate_id": candidate["candidate_id"],
                        "fold_id": 0,
                        "is_score": float(
                            objectives.get("total_return_pct", 0.0) or 0.0
                        ),
                        "oos_score": float(
                            validation.get("total_return_pct", 0.0) or 0.0
                        ),
                        "source": candidate.get("source", "phase2"),
                        "direction": candidate.get("direction", ""),
                    })
                    continue
                split_metrics = candidate.get("split_metrics", {})
                if isinstance(split_metrics, dict):
                    train_metrics = split_metrics.get("train", {})
                    valid_metrics = split_metrics.get("validation", {})
                    if isinstance(train_metrics, dict) and isinstance(
                        valid_metrics, dict
                    ):
                        rows.append({
                            "candidate_id": candidate["candidate_id"],
                            "fold_id": 0,
                            "is_score": float(
                                train_metrics.get("total_return_pct", 0.0) or 0.0
                            ),
                            "oos_score": float(
                                valid_metrics.get("total_return_pct", 0.0) or 0.0
                            ),
                            "source": candidate.get("source", "rb"),
                            "direction": candidate.get("direction", ""),
                        })

        configured_matrix_path = getattr(
            _cfg,
            "CANDIDATE_FOLD_MATRIX_PATH",
            os.path.join(_cfg.REPORTS_DIR, "candidate_fold_matrix.jsonl"),
        )
        # Keep temporary-output tests and --output runs bound to the active
        # report directory even when the module-level default was imported
        # before OUTPUTS_DIR was overridden.
        if str(configured_matrix_path).endswith(
            os.path.join("outputs", "reports", "candidate_fold_matrix.jsonl")
        ):
            configured_matrix_path = os.path.join(
                _cfg.REPORTS_DIR, "candidate_fold_matrix.jsonl"
            )
        write_candidate_fold_matrix(configured_matrix_path, rows)
        return rows

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
            "RB_MULTI_SYMBOL_RELEASE",
            "PHASE2_VAL_IN_FITNESS_PENALTY",
        )
        config_delta = {
            key: getattr(_cfg, key)
            for key in config_keys
            if hasattr(_cfg, key)
        }
        trial_counters = self._trial_counters(
            results.get("phase1"),
            phase2,
            rb,
        )
        from gpu_fuzzy_trader.validation.multiplicity import (
            read_candidate_fold_matrix,
            read_ledger_trial_count,
            summarize_multiplicity,
            trial_count_from_counters,
        )
        trial_count = trial_count_from_counters(trial_counters)
        if trial_count <= 0:
            trial_count = count_trials(phase2=phase2, rb=rb)
        matrix_path = getattr(
            _cfg,
            "CANDIDATE_FOLD_MATRIX_PATH",
            os.path.join(_cfg.REPORTS_DIR, "candidate_fold_matrix.jsonl"),
        )
        if str(matrix_path).endswith(
            os.path.join("outputs", "reports", "candidate_fold_matrix.jsonl")
        ):
            matrix_path = os.path.join(
                _cfg.REPORTS_DIR, "candidate_fold_matrix.jsonl"
            )
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
            "trial_count_ledger": int(trial_count),
            "trial_counters": trial_counters,
            "candidate_fold_matrix": matrix_path,
            "config": config_delta,
            "research_profile": {
                "profile_id": ResearchProfile.from_config(_cfg).profile_id,
                **ResearchProfile.from_config(_cfg).as_dict(),
            },
        })
        try:
            stability = results.get("strategy_stability", {})
            fold_returns = [
                float(report.get("median_return_pct", 0.0))
                for report in stability.values()
                if isinstance(report, dict)
            ] if isinstance(stability, dict) else []
            ledger_trial_count = read_ledger_trial_count(
                _cfg.OUTPUTS_DIR,
                run_id=self._run_id,
            )
            matrix_rows = read_candidate_fold_matrix(matrix_path)
            multiplicity = summarize_multiplicity(
                fold_returns=fold_returns,
                n_trials=trial_count,
                matrix=matrix_rows,
                trial_count_ledger=(
                    ledger_trial_count
                    if ledger_trial_count is not None
                    else trial_count
                ),
                ledger_counters=trial_counters,
            )
            configured_summary_path = getattr(
                _cfg,
                "MULTIPLICITY_SUMMARY_PATH",
                Path(_cfg.REPORTS_DIR) / "multiplicity_summary.json",
            )
            if str(configured_summary_path).endswith(
                os.path.join("outputs", "reports", "multiplicity_summary.json")
            ):
                configured_summary_path = Path(
                    _cfg.REPORTS_DIR, "multiplicity_summary.json"
                )
            Path(configured_summary_path).write_text(
                json.dumps(multiplicity, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning(
                "Multiplicity report failed (non-fatal): %s",
                exc,
            )

    @staticmethod
    def _should_use_mtf_pipeline(train_df: pd.DataFrame) -> bool:
        """Select the canonical MTF path only for real OHLCV tapes."""
        required = {"datetime", "symbol", "open",
                    "high", "low", "close", "volume"}
        return bool(
            getattr(_cfg, "MTF_PIPELINE_ENABLED", False)
            and required.issubset(train_df.columns)
        )

    @staticmethod
    def _build_mtf_lwc_training_frame(
        train_df: pd.DataFrame,
        hwc_discovery: LayerDiscoveryResult,
        mwc_discovery: LayerDiscoveryResult,
    ) -> pd.DataFrame:
        """Attach only causal HWC/MWC OOF scores to the LWC train tape."""
        scored = attach_oof_layer_scores(
            train_df,
            hwc_scores=hwc_discovery.oof_scores,
            mwc_scores=mwc_discovery.oof_scores,
        )
        return _merge_mtf_lwc_runtime_columns(train_df, scored)

    @staticmethod
    def _build_mtf_lwc_validation_frame(
        val_df: pd.DataFrame,
        hwc_rules: list[dict[str, Any]],
        mwc_rules: list[dict[str, Any]],
        history_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Attach final full-train HWC/MWC scores to validation only."""
        scored = attach_frozen_layer_scores(
            val_df,
            hwc_rules,
            mwc_rules,
            history_df=history_df,
        )
        return _merge_mtf_lwc_runtime_columns(val_df, scored)

    def _attach_mtf_scores_to_cv_folds(
        self,
        train_oof_df: pd.DataFrame,
        validation_frozen_df: pd.DataFrame,
    ) -> None:
        """Bind OOF score columns to legacy LWC CV folds by timestamp keys."""
        if not self._cv_folds:
            return
        from dataclasses import replace

        updated = []
        for fold in self._cv_folds:
            fold_train = _merge_mtf_lwc_runtime_columns(
                fold.train_df, train_oof_df)
            fold_source = (
                validation_frozen_df if bool(getattr(fold, "is_holdout", False))
                else train_oof_df
            )
            fold_valid = _merge_mtf_lwc_runtime_columns(
                fold.valid_df, fold_source)
            if not bool(getattr(fold, "is_holdout", False)):
                available_train = fold_train["_mtf_oof_available"].fillna(
                    False).astype(bool)
                available_valid = fold_valid["_mtf_oof_available"].fillna(
                    False).astype(bool)
                fold_train = fold_train.loc[available_train].reset_index(
                    drop=True)
                fold_valid = fold_valid.loc[available_valid].reset_index(
                    drop=True)
            if fold_train.empty or fold_valid.empty:
                logger.warning(
                    "Dropping LWC CV fold %s after OOF score support filtering",
                    getattr(fold, "fold_id", "unknown"),
                )
                continue
            updated.append(
                replace(
                    fold,
                    train_df=fold_train,
                    valid_df=fold_valid,
                    n_train_rows=len(fold_train),
                    n_valid_rows=len(fold_valid),
                )
            )
        self._cv_folds = updated or None

    def _run_mtf_pipeline(
        self,
        *,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        force: bool,
    ) -> dict[str, Any]:
        """Run the connected HWC -> OOF -> MWC -> LWC -> RB -> OOS path."""
        logger.info(
            "Canonical input is raw OHLCV; running hierarchical MTF pipeline")
        mtf_folds = build_master_temporal_folds(
            train_df,
            n_folds=int(_cfg.MTF_MAX_FOLDS),
            embargo_minutes=discovery_purge_minutes("hwc"),
        )
        self._mtf_folds = mtf_folds

        hwc_rules = self.run_phase1_hwc(train_df, folds=mtf_folds, force=force)
        if not hwc_rules:
            raise RuntimeError(
                "MTF HWC discovery produced no admissible OOF-backed rules"
            )
        mwc_rules = self.run_phase1_mwc(
            train_df,
            hwc_rules=hwc_rules,
            folds=mtf_folds,
            force=force,
        )
        if not mwc_rules:
            raise RuntimeError(
                "MTF MWC discovery produced no admissible HWC-conditioned rules"
            )

        hwc_discovery = getattr(self, "_mtf_hwc_discovery", None)
        mwc_discovery = getattr(self, "_mtf_mwc_discovery", None)
        if hwc_discovery is None or mwc_discovery is None:
            raise RuntimeError(
                "MTF discovery did not produce HWC and MWC OOF artifacts")

        val_fitness_df, val_selection_df = self._validation_scoring_frames(val_df)

        # LWC discovery is restricted to rows carrying both upstream OOF
        # scores. Validation fitness receives the final full-train ensembles only
        # after discovery, so no OOF value is replaced by an in-sample score.
        lwc_train_scored_all = self._build_mtf_lwc_training_frame(
            train_df, hwc_discovery, mwc_discovery
        )
        lwc_val_fitness_scored = self._build_mtf_lwc_validation_frame(
            val_fitness_df, hwc_rules, mwc_rules, history_df=train_df
        )
        oof_available = lwc_train_scored_all["_mtf_oof_available"].fillna(
            False).astype(bool)
        lwc_train_scored = lwc_train_scored_all.loc[oof_available].reset_index(
            drop=True)
        if lwc_train_scored.empty:
            raise RuntimeError(
                "MTF LWC discovery has no rows with both HWC and MWC OOF scores"
            )
        self._attach_mtf_scores_to_cv_folds(
            lwc_train_scored_all,
            lwc_val_fitness_scored,
        )

        # Keep the existing NSGA-III/plateau LWC search as the execution-rule
        # generator. It receives only causal OOF HWC/MWC score features.
        phase1_train_df = self._mask_train_df_for_phase1(lwc_train_scored)
        lwc_phase1 = self._run_phase1(
            phase1_train_df,
            force=force,
            val_df=lwc_val_fitness_scored,
        )
        lwc_rules = self._run_phase2(
            phase1_train_df,
            lwc_phase1,
            force=force,
            val_df=lwc_val_fitness_scored,
            blocked_directions=frozenset(),
        )
        self._mtf_lwc_phase1 = lwc_phase1

        candidates = self.run_mtf_composition(
            lwc_rules=lwc_rules,
            hwc_rules=hwc_rules,
            mwc_rules=mwc_rules,
            df=train_df,
            folds=mtf_folds,
            metadata={
                "dataset_hashes": {
                    "train": _dataframe_sha256(train_df),
                    "validation": _dataframe_sha256(val_df),
                    "oos": sha256_file(_cfg.TEST_CSV_PATH)
                    if Path(_cfg.TEST_CSV_PATH).exists()
                    else "",
                },
                "fold_boundaries": export_fold_boundaries(mtf_folds, df=train_df),
                "labels": {
                    "theta_per_oof_fold": {
                        "hwc": hwc_discovery.theta_per_oof_fold,
                        "mwc": mwc_discovery.theta_per_oof_fold,
                    },
                    "theta_final_train": {
                        "hwc": hwc_discovery.theta_final_train,
                        "mwc": mwc_discovery.theta_final_train,
                    },
                },
                "features": {
                    "hwc": {
                        "schema_hash": hwc_discovery.feature_schema_hash,
                        "data_hash": hwc_discovery.data_hash,
                    },
                    "mwc": {
                        "schema_hash": mwc_discovery.feature_schema_hash,
                        "data_hash": mwc_discovery.data_hash,
                    },
                    "lwc": {
                        "schema_hash": _dataframe_schema_sha256(lwc_train_scored),
                        "data_hash": _dataframe_sha256(lwc_train_scored),
                        "oof_score_columns": [
                            "mtf_hwc_direction", "mtf_hwc_strength",
                            "mtf_mwc_direction", "mtf_mwc_strength",
                        ],
                        "source": "causal_upstream_oof_only",
                    },
                },
                "search": {
                    "hwc": hwc_discovery.search_metadata,
                    "mwc": mwc_discovery.search_metadata,
                    "lwc": {
                        "algorithm": "existing_phase2_nsga3_with_plateau_restarts",
                        "source": "legacy_lwc_rule_pool_generator",
                    },
                },
                "release_policy": {
                    "fit": "train_only",
                    "validation": "frozen_candidate_only",
                    "oos": "one_shot_no_refit",
                },
            },
        )
        rb_result = self._run_mtf_rb_governor(
            train_df=train_df,
            val_df=val_selection_df,
            candidates=candidates,
        )
        self._release_between_phases("Phase 5")
        accepted = frozenset(
            direction
            for direction, strategy in rb_result.items()
            if strategy.get("rules_set")
            and strategy.get("deployment_accepted") is True
        )
        phase5_result = self._run_phase5(allowed_directions=accepted)
        matrix_rows = self._write_candidate_fold_matrix(
            train_df,
            lwc_rules,
            rb_result,
        )
        trial_counters = self._trial_counters(
            lwc_phase1,
            lwc_rules,
            rb_result,
        )
        stability_report = self.run_stability_report(
            train_df,
            rb_result,
            trial_count=max(1, sum(trial_counters.values())),
            trial_count_ledger=sum(trial_counters.values()),
            candidate_fold_matrix=matrix_rows,
        )
        return {
            "data": {
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "val_fitness_rows": len(val_fitness_df),
                "val_selection_rows": len(val_selection_df),
                "source_contract": "raw_ohlcv_15m",
                "timezone": "UTC",
            },
            "phase1": lwc_phase1,
            "phase2": lwc_rules,
            "mtf_hwc": hwc_rules,
            "mtf_mwc": mwc_rules,
            "mtf_candidates": candidates,
            "rb_governor": rb_result,
            "phase3": rb_result,
            "phase4": rb_result,
            "phase_status": {"mtf": "completed", "rb_governor": self._rb_status_summary(rb_result)},
            "strategy_stability": stability_report or {
                "status": "not_run",
                "reason": "no_frozen_strategy",
            },
            "phase5": phase5_result,
        }
    def _run_mtf_from_phase2(
        self,
        *,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        force: bool,
    ) -> dict[str, Any]:
        """Run Phase 2 onwards in MTF mode using existing or generated HWC/MWC artifacts."""
        logger.info("Running hierarchical MTF pipeline from Phase 2")
        mtf_folds = build_master_temporal_folds(
            train_df,
            n_folds=int(_cfg.MTF_MAX_FOLDS),
            embargo_minutes=discovery_purge_minutes("hwc"),
        )
        self._mtf_folds = mtf_folds

        hwc_rules = self.run_phase1_hwc(train_df, folds=mtf_folds, force=False)
        mwc_rules = self.run_phase1_mwc(
            train_df, hwc_rules=hwc_rules, folds=mtf_folds, force=False)
        hwc_discovery = getattr(self, "_mtf_hwc_discovery", None)
        mwc_discovery = getattr(self, "_mtf_mwc_discovery", None)
        if hwc_discovery is None or mwc_discovery is None:
            raise RuntimeError(
                "MTF discovery did not produce HWC and MWC artifacts")

        val_fitness_df, val_selection_df = self._validation_scoring_frames(val_df)

        lwc_train_scored_all = self._build_mtf_lwc_training_frame(
            train_df, hwc_discovery, mwc_discovery
        )
        lwc_val_fitness_scored = self._build_mtf_lwc_validation_frame(
            val_fitness_df, hwc_rules, mwc_rules, history_df=train_df
        )
        oof_available = lwc_train_scored_all["_mtf_oof_available"].fillna(
            False).astype(bool)
        lwc_train_scored = lwc_train_scored_all.loc[oof_available].reset_index(
            drop=True)
        if lwc_train_scored.empty:
            raise RuntimeError(
                "MTF LWC discovery has no rows with both HWC and MWC OOF scores"
            )
        self._attach_mtf_scores_to_cv_folds(
            lwc_train_scored_all,
            lwc_val_fitness_scored,
        )

        phase1_train_df = self._mask_train_df_for_phase1(lwc_train_scored)
        try:
            lwc_phase1 = self._load_phase1_outputs(phase1_train_df)
        except Exception:
            lwc_phase1 = self._run_phase1(
                phase1_train_df,
                force=False,
                val_df=lwc_val_fitness_scored,
            )
        self._mtf_lwc_phase1 = lwc_phase1

        lwc_rules = self._run_phase2(
            phase1_train_df,
            lwc_phase1,
            force=force,
            val_df=lwc_val_fitness_scored,
            blocked_directions=frozenset(),
        )

        candidates = self.run_mtf_composition(
            lwc_rules=lwc_rules,
            hwc_rules=hwc_rules,
            mwc_rules=mwc_rules,
            df=train_df,
            folds=mtf_folds,
            metadata={
                "dataset_hashes": {
                    "train": _dataframe_sha256(train_df),
                    "validation": _dataframe_sha256(val_df),
                    "oos": sha256_file(_cfg.TEST_CSV_PATH)
                    if Path(_cfg.TEST_CSV_PATH).exists()
                    else "",
                },
                "fold_boundaries": export_fold_boundaries(mtf_folds, df=train_df),
                "labels": {
                    "theta_per_oof_fold": {
                        "hwc": hwc_discovery.theta_per_oof_fold,
                        "mwc": mwc_discovery.theta_per_oof_fold,
                    },
                    "theta_final_train": {
                        "hwc": hwc_discovery.theta_final_train,
                        "mwc": mwc_discovery.theta_final_train,
                    },
                },
                "features": {
                    "hwc": {
                        "schema_hash": hwc_discovery.feature_schema_hash,
                        "data_hash": hwc_discovery.data_hash,
                    },
                    "mwc": {
                        "schema_hash": mwc_discovery.feature_schema_hash,
                        "data_hash": mwc_discovery.data_hash,
                    },
                    "lwc": {
                        "schema_hash": _dataframe_schema_sha256(lwc_train_scored),
                        "data_hash": _dataframe_sha256(lwc_train_scored),
                        "oof_score_columns": [
                            "mtf_hwc_direction", "mtf_hwc_strength",
                            "mtf_mwc_direction", "mtf_mwc_strength",
                        ],
                        "source": "causal_upstream_oof_only",
                    },
                },
                "search": {
                    "hwc": hwc_discovery.search_metadata,
                    "mwc": mwc_discovery.search_metadata,
                    "lwc": {
                        "algorithm": "existing_phase2_nsga3_with_plateau_restarts",
                        "source": "legacy_lwc_rule_pool_generator",
                    },
                },
                "release_policy": {
                    "fit": "train_only",
                    "validation": "frozen_candidate_only",
                    "oos": "one_shot_no_refit",
                },
            },
        )
        rb_result = self._run_mtf_rb_governor(
            train_df=train_df,
            val_df=val_selection_df,
            candidates=candidates,
        )
        self._release_between_phases("Phase 5")
        accepted = frozenset(
            direction
            for direction, strategy in rb_result.items()
            if strategy.get("rules_set")
            and strategy.get("deployment_accepted") is True
        )
        phase5_result = self._run_phase5(allowed_directions=accepted)
        matrix_rows = self._write_candidate_fold_matrix(
            train_df,
            lwc_rules,
            rb_result,
        )
        trial_counters = self._trial_counters(
            lwc_phase1,
            lwc_rules,
            rb_result,
        )
        stability_report = self.run_stability_report(
            train_df,
            rb_result,
            trial_count=max(1, sum(trial_counters.values())),
            trial_count_ledger=sum(trial_counters.values()),
            candidate_fold_matrix=matrix_rows,
        )
        return {
            "data": {
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "val_fitness_rows": len(val_fitness_df),
                "val_selection_rows": len(val_selection_df),
                "source_contract": "raw_ohlcv_15m",
                "timezone": "UTC",
            },
            "phase1": lwc_phase1,
            "phase2": lwc_rules,
            "mtf_hwc": hwc_rules,
            "mtf_mwc": mwc_rules,
            "mtf_candidates": candidates,
            "rb_governor": rb_result,
            "phase3": rb_result,
            "phase4": rb_result,
            "phase_status": {"mtf": "completed", "rb_governor": self._rb_status_summary(rb_result)},
            "strategy_stability": stability_report or {
                "status": "not_run",
                "reason": "no_frozen_strategy",
            },
            "phase5": phase5_result,
        }

    def _run_mtf_rb_governor(

        self,
        *,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        candidates: dict[str, HierarchicalStrategyCandidate],
    ) -> dict[str, dict[str, Any]]:
        """Validate frozen composed signals and write MTF-aware strategy packages."""
        from gpu_fuzzy_trader.backtest.cpu_engine import CPUBacktestEngine

        output_root = Path(_cfg.OUTPUTS_DIR)
        output_root.mkdir(parents=True, exist_ok=True)
        reports_root = Path(_cfg.REPORTS_DIR)
        reports_root.mkdir(parents=True, exist_ok=True)
        results: dict[str, dict[str, Any]] = {}
        for direction in ("long", "short"):
            candidate = candidates.get(direction)
            if candidate is None or not candidate.lwc_rules:
                results[direction] = {
                    "direction": direction,
                    "rules_set": [],
                    "deployment_accepted": False,
                    "fail_closed": True,
                    "reason": "empty_mtf_lwc_pool",
                }
                continue

            first_rule = candidate.lwc_rules[0]
            tp = float(first_rule.get(
                "tp", getattr(_cfg, "RB_DEFAULT_TP", 2.0)))
            sl = float(first_rule.get(
                "sl", getattr(_cfg, "RB_DEFAULT_SL", 1.2)))
            capital_pct = float(first_rule.get(
                "capital_pct", getattr(_cfg, "RB_DEFAULT_CAPITAL_PCT", 18.0)
            ))

            split_metrics: dict[str, dict[str, Any]] = {}
            retention: dict[str, Any] = {}
            for split_name, split_df in (("train", train_df), ("validation", val_df)):
                signals, stats, audit = evaluate_candidate_frame(
                    candidate,
                    split_df,
                    history_df=train_df if split_name == "validation" else None,
                )
                retention[split_name] = stats.get("retention_diagnostics", {})
                engine = CPUBacktestEngine(split_df, {}, direction)
                metrics = engine.simulate_signal_mask(
                    signals != 0,
                    tp=tp,
                    sl=sl,
                    capital_pct=capital_pct,
                )
                split_metrics[split_name] = metrics

            retention_ok = all(
                bool(value.get("passes_floor", False))
                for value in retention.values()
            )
            train_metrics = split_metrics.get("train", {})
            validation_metrics = split_metrics.get("validation", {})
            performance_ok = all(
                float(metrics.get("total_return_pct", 0.0))
                >= float(getattr(_cfg, threshold_name, 0.0))
                and float(metrics.get("profit_factor", 0.0))
                >= float(getattr(_cfg, pf_name, 0.0))
                and int(metrics.get("executed_trades", 0))
                >= int(getattr(_cfg, trades_name, 0))
                for metrics, threshold_name, pf_name, trades_name in (
                    (
                        train_metrics,
                        "RB_MIN_TRAIN_RETURN",
                        "RB_MIN_TRAIN_PF",
                        "RB_MIN_TRAIN_TRADES",
                    ),
                    (
                        validation_metrics,
                        "RB_MIN_VALID_RETURN",
                        "RB_MIN_VALID_PF",
                        "RB_MIN_VALID_TRADES",
                    ),
                )
            )
            accepted = bool(
                retention_ok and performance_ok and candidate.lwc_rules)
            strategy = {
                "direction": direction,
                "rules_set": [dict(rule) for rule in candidate.lwc_rules],
                "deployment_accepted": accepted,
                "fail_closed": not accepted,
                "reason": (
                    "mtf_retention_floor"
                    if not retention_ok
                    else (
                        "mtf_train_validation_gate"
                        if not performance_ok
                        else "mtf_candidate_accepted"
                    )
                ),
                "strategy_id": candidate.strategy_id,
                "mtf_candidate": candidate.to_dict(),
                "mtf_manifest": candidate.mtf_manifest,
                "mtf_runtime": {
                    "frozen": True,
                    "split_metrics": split_metrics,
                    "retention": retention,
                    "tp": tp,
                    "sl": sl,
                    "capital_pct": capital_pct,
                    "acceptance_gates": {
                        "retention_floor": retention_ok,
                        "train_validation_performance": performance_ok,
                    },
                },
                "provenance": {
                    "mtf_manifest_hash": hashlib.sha256(
                        json.dumps(candidate.mtf_manifest or {},
                                   sort_keys=True).encode("utf-8")
                    ).hexdigest(),
                },
            }
            (output_root / f"{direction}.json").write_text(
                json.dumps(strategy, indent=2, sort_keys=True, default=str),
                encoding="utf-8",
            )
            (reports_root / f"mtf_{direction}_retention.json").write_text(
                json.dumps(retention, indent=2, sort_keys=True, default=str),
                encoding="utf-8",
            )
            results[direction] = strategy
        return results

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
                        strategy.get("deployment_reason",
                                     "accepted" if accepted else "no_strategy")
                    )
                ),
            }
        return summary

    def _load_phase1_outputs(
        self,
        train_df: pd.DataFrame | None = None,
        val_df: pd.DataFrame | None = None,
    ) -> dict[str, list[dict]]:
        """Load Phase 1 outputs only when they match current inputs."""
        if train_df is None:
            raise FileNotFoundError(
                "Phase 2 requires Phase 1 outputs matching current Phase 1 input; "
                "rerun Phase 1."
            )

        try:
            result = Feature_Selector.skip_if_valid(train_df, val_df=val_df)
        except ValueError as exc:
            raise FileNotFoundError(
                "Phase 2 requires valid Phase 1 outputs matching current "
                "Phase 1 input; rerun Phase 1."
            ) from exc

        if result is None:
            raise FileNotFoundError(
                "Phase 2 requires Phase 1 outputs matching current Phase 1 input; "
                "rerun Phase 1."
            )
        return result

    def _load_phase2_outputs(
        self,
        train_df: pd.DataFrame | None = None,
        val_df: pd.DataFrame | None = None,
        phase1_result: dict[str, list[dict]] | None = None,
    ) -> dict[str, list[dict]]:
        """Load Phase 2 pools only when they match current prerequisites."""
        result: dict[str, list[dict]] = {}
        missing: list[str] = []
        self._phase2_status = {}

        if train_df is None or phase1_result is None:
            logger.warning(
                "RB Governor will fail closed: Phase 2 input identity is unavailable"
            )
            for direction in ("long", "short"):
                result[direction] = []
                self._phase2_status[direction] = {
                    "status": "error",
                    "reason": "phase2_identity_unavailable",
                    "detail": (
                        "Phase 3/4 requires current Phase 1 outputs and "
                        "current Phase 2 input frames"
                    ),
                    "pool_size": 0,
                }
            return result

        for direction in ("long", "short"):
            expected_identity = _phase2_resume_identity(
                train_df,
                val_df,
                phase1_result.get(direction, []),
                direction,
                self._cv_folds,
            )
            pool = Rule_Pool_Generator.skip_if_valid(
                direction,
                expected_identity=expected_identity,
            )

            if pool is None:
                missing.append(
                    f"{direction}: {_phase2_module._POOL_PATHS[direction]} "
                    "(missing, invalid, or identity mismatch)"
                )
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
        """
        mtf_mode = bool(getattr(_cfg, "MTF_PIPELINE_ENABLED", False))
        if not mtf_mode:
            _validate_enriched_context_contract()
        elif any(
            Path(path).name.endswith("_hwc_mwc_lwc.csv")
            for path in (_cfg.TRAIN_CSV_PATH, _cfg.TEST_CSV_PATH)
        ):
            raise RuntimeError(
                "The canonical MTF pipeline requires raw 15m tapes; enriched "
                "HWC/MWC/LWC inputs are supported only with MTF_PIPELINE_ENABLED=False."
            )
        # Cached parquet splits created by the removed context pipeline are not
        # valid MTF inputs.  Rebuild the split from the raw tape so its identity
        # and feature schema are tied to this run's source data.
        cached_split = None if mtf_mode else load_cached_split_if_fresh()
        if cached_split is not None:
            train_df, val_df, val_fitness, val_selection, cv_folds = cached_split
            # A cache predating the enriched-input contract must never let the
            # canonical pipeline bypass the loader's fail-closed validation.
            if not mtf_mode:
                for frame in (train_df, val_df, val_fitness, val_selection):
                    validate_context_columns(frame)
            scaling = fit_fuzzy_feature_scaling(train_df)
            for frame in (train_df, val_df, val_fitness, val_selection):
                apply_fuzzy_feature_scaling(frame, scaling)
            self._cv_folds = cv_folds
            self._preloaded_val_fitness = val_fitness
            self._preloaded_val_selection = val_selection
            logger.info(
                "Using cached train/validation split from %s and %s",
                _cfg.DEVELOPMENT_TRAIN_PATH,
                _cfg.VALIDATION_PATH,
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
            require_context=False if mtf_mode else True,
        )
        # Drop rows where any LABEL_COLUMNS is NaN before splitting (e.g. tail rows
        # retained by drop_tail=False for exact barrier outcome calculation).
        # This keeps the full source tape for barrier outcome construction while
        # ensuring downstream splits and backtest engines receive valid labels.
        from gpu_fuzzy_trader.config import LABEL_COLUMNS
        train_full = train_full.dropna(subset=list(LABEL_COLUMNS)).reset_index(drop=True)
        logger.info(
            "Loaded %d rows, %d symbols",
            len(train_full),
            train_full["symbol"].nunique()
            if "symbol" in train_full.columns
            else 0,
        )
        if not mtf_mode:
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
        split_label = f"holdout {_cfg.holdout_train_val_label()}"
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
            val_fitness, val_selection = split_validation_fitness_selection(
                val_df)
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

            sym_set = set(str(s) for s in symbols)
            scoped_folds: list = []
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

        names = Pipeline_Orchestrator._phase1_keep_feature_names(phase1_result)
        if not names:
            return cv_folds

        pruned: list = []
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
        """Return the canonical holdout training frame without legacy CV masking."""
        if not self._cv_folds or train_df.empty:
            return train_df
        if "_symbol_bar_index" not in train_df.columns:
            return train_df
        forbidden = [
            (
                max(0, int(getattr(fold, "valid_start_bar")) - int(_cfg.MAX_HOLD_CANDLES)),
                int(getattr(fold, "valid_end_bar")),
            )
            for fold in self._cv_folds
            if getattr(fold, "valid_start_bar", None) is not None
            and getattr(fold, "valid_end_bar", None) is not None
        ]
        if not forbidden:
            return train_df
        bar_index = train_df["_symbol_bar_index"].to_numpy()
        safe = np.ones(len(train_df), dtype=bool)
        for start, end in forbidden:
            safe &= ~((bar_index >= start) & (bar_index <= end))
        masked = train_df.loc[safe].reset_index(drop=True)
        logger.info(
            "Phase 1: masked train_df to safe fold region (%d -> %d rows)",
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
            existing = Feature_Selector.skip_if_valid(train_df, val_df=val_df)
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

            # Bind a resumed pool to the exact selected features, split data,
            # CV boundaries, configuration, and evaluator code used today.
            feature_infos = phase1_result.get(direction, [])
            resume_identity = _phase2_resume_identity(
                train_df,
                val_df,
                feature_infos,
                direction,
                self._cv_folds,
            )

            if not force:
                existing_pool = Rule_Pool_Generator.skip_if_valid(
                    direction,
                    expected_identity=resume_identity,
                )
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

                # A stale pool may otherwise be merged back into the new
                # population by Rule_Pool_Generator.run().  Remove only the
                # cache artifacts that failed this run's identity contract.
                pool_path = _phase2_module._POOL_PATHS[direction]
                identity_path = f"{pool_path}.identity.json"
                if os.path.exists(pool_path) or os.path.exists(identity_path):
                    logger.info(
                        "Discarding stale Phase 2 %s cache before regeneration",
                        direction,
                    )
                    Rule_Pool_Generator.discard_cached_pool(direction)

            if force:
                pool_path = _phase2_module._POOL_PATHS[direction]
                identity_path = f"{pool_path}.identity.json"
                if os.path.exists(pool_path) or os.path.exists(identity_path):
                    logger.info(
                        "Discarding Phase 2 %s cache for a forced regeneration",
                        direction,
                    )
                    Rule_Pool_Generator.discard_cached_pool(direction)

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
                pool_path = _phase2_module._POOL_PATHS[direction]
                if os.path.isfile(pool_path):
                    try:
                        Rule_Pool_Generator.write_pool_resume_identity(
                            direction,
                            resume_identity,
                        )
                    except (OSError, ValueError) as exc:
                        logger.warning(
                            "Phase 2 [%s]: pool was generated but cannot be "
                            "resumed safely: %s",
                            direction,
                            exc,
                        )
                else:
                    logger.warning(
                        "Phase 2 [%s]: generator returned without writing %s; "
                        "the result will not be resumable",
                        direction,
                        pool_path,
                    )

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
        # Canonical RB may only select/compose Phase 2 discoveries. Mode A
        # (no required symbol filters) is the global two-symbol contract.
        rb_policy_attrs = {
            "RB_REQUIRE_SYMBOL_FILTERS": False,
            "RB_ALLOW_PARTIAL_SPECIALIST_COVERAGE": False,
            "RB_MULTI_SYMBOL_RELEASE": True,
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
        test_csv_path: str | None = None,
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
            evaluator = OOS_Evaluator(
                test_csv_path=test_csv_path,
                run_id=self._run_id,
            )
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
    def run_stability_report(
        train_df: pd.DataFrame,
        strategies: dict[str, dict],
        *,
        trial_count: int = 1,
        trial_count_ledger: int | None = None,
        candidate_fold_matrix: Any | None = None,
    ) -> dict[str, dict]:
        """Write a diagnostic stability report for frozen strategy packages."""
        try:
            from gpu_fuzzy_trader.validation.walk_forward_stability_report import (
                write_strategy_stability_reports,
            )
            stability_strategies = {
                direction: {
                    **strategy,
                    "trial_count": int(trial_count),
                }
                for direction, strategy in strategies.items()
            }
            stability = write_strategy_stability_reports(
                _cfg.OUTPUTS_DIR,
                stability_strategies,
                train_df,
                n_windows=int(
                    getattr(_cfg, "MULTIPLICITY_STABILITY_WINDOWS", 3)
                ),
                candidate_fold_matrix=candidate_fold_matrix,
                trial_count_ledger=trial_count_ledger,
            )
            from gpu_fuzzy_trader.validation.baselines import (
                write_baseline_reports,
            )
            baseline = write_baseline_reports(
                _cfg.OUTPUTS_DIR,
                train_df,
                stability_strategies,
            )
            for direction, report in stability.items():
                report["baselines"] = baseline.get(direction, {})
            return stability
        except Exception as exc:
            logger.warning(
                "Strategy stability report failed (non-fatal to pipeline): %s",
                exc,
            )
            return {}

    # ------------------------------------------------------------------
    # Hierarchical MTF Discovery & Composition Phase Methods
    # ------------------------------------------------------------------

    @staticmethod
    def _load_mtf_source_tape(path: str | None = None) -> pd.DataFrame:
        """Load only raw OHLCV columns for standalone MTF discovery calls."""
        source_path = path or getattr(
            _cfg, "RAW_TRAIN_CSV_PATH", _cfg.TRAIN_CSV_PATH)
        columns = ["datetime", "symbol", "open",
                   "high", "low", "close", "volume"]
        return pd.read_csv(source_path, usecols=columns)

    def build_mtf_manifest(
        self,
        hwc_archive_hash: str = "",
        mwc_archive_hash: str = "",
        lwc_archive_hash: str = "",
        composer_params: dict[str, Any] | None = None,
        output_path: str | Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build and write the frozen mtf_manifest.json file."""
        target_path = Path(output_path) if output_path else Path(
            self._output_dir) / "mtf_manifest.json"
        target_path.parent.mkdir(parents=True, exist_ok=True)

        c_params = {
            "v_hwc_long": float(getattr(_cfg, "MTF_V_HWC_LONG", DEFAULT_V_HWC_LONG)),
            "v_hwc_short": float(getattr(_cfg, "MTF_V_HWC_SHORT", DEFAULT_V_HWC_SHORT)),
            "v_mwc_long": float(getattr(_cfg, "MTF_V_MWC_LONG", DEFAULT_V_MWC_LONG)),
            "v_mwc_short": float(getattr(_cfg, "MTF_V_MWC_SHORT", DEFAULT_V_MWC_SHORT)),
            "min_evidence_strength_hwc": float(getattr(_cfg, "MTF_MIN_EVIDENCE_STRENGTH_HWC", DEFAULT_MIN_EVIDENCE_STRENGTH)),
            "min_evidence_strength_mwc": float(getattr(_cfg, "MTF_MIN_EVIDENCE_STRENGTH_MWC", DEFAULT_MIN_EVIDENCE_STRENGTH)),
            "retention_floor": float(getattr(_cfg, "MTF_RETENTION_FLOOR", DEFAULT_RETENTION_FLOOR)),
            "retention_target": float(getattr(_cfg, "MTF_RETENTION_TARGET", DEFAULT_RETENTION_TARGET)),
        }
        if composer_params:
            c_params.update(composer_params)

        raw_metadata = dict(metadata or {})
        dataset_hashes = dict(raw_metadata.pop("dataset_hashes", {}))
        labels_metadata = dict(raw_metadata.pop("labels", {}))
        cross_metadata = dict(raw_metadata.pop("cross_fitting", {}))
        feature_metadata = dict(raw_metadata.pop("features", {}))
        search_metadata = dict(raw_metadata.pop("search", {}))
        feature_metadata.setdefault(
            "warmup_bars",
            {"lwc": 20, "mwc": 20, "hwc": 20},
        )
        feature_metadata.setdefault(
            "frozen_transforms",
            {"lwc": "causal_indicator_nan_to_zero", "mwc": "none", "hwc": "none"},
        )
        if "fold_boundaries" in raw_metadata:
            cross_metadata["fold_boundaries"] = raw_metadata.pop(
                "fold_boundaries")
        if "theta_per_oof_fold" in raw_metadata:
            labels_metadata["theta_per_oof_fold"] = raw_metadata.pop(
                "theta_per_oof_fold")
        if "theta_final_train" in raw_metadata:
            labels_metadata["theta_final_train"] = raw_metadata.pop(
                "theta_final_train")
        if "feature_schema_hash" in raw_metadata:
            feature_metadata["schema_hash"] = raw_metadata.pop(
                "feature_schema_hash")
        config_fields = {
            name: getattr(_cfg, name)
            for name in (
                "HOLDOUT_TRAIN_FRACTION", "MAX_HOLD_CANDLES",
                "VALIDATION_PURGE_CANDLES",
                "MTF_PIPELINE_ENABLED", "MTF_MAX_FOLDS", "MTF_MIN_FOLDS",
                "MTF_MIN_FOLD_SUPPORT_RATIO", "FOLD_MIN_EFFECTIVE_ROWS",
                "FOLD_MIN_ROWS_PER_SYMBOL", "FOLD_ABSOLUTE_MIN_TRADES",
                "FOLD_MIN_DURATION_BARS", "FOLD_MIN_SYMBOL_COVERAGE",
                "LWC_TIMEFRAME_MINUTES", "MWC_TIMEFRAME_MINUTES",
                "HWC_TIMEFRAME_MINUTES", "MTF_DISCOVERY_MAX_RULES_PER_LAYER",
                "MTF_HWC_HORIZON_BARS", "MTF_MWC_HORIZON_BARS",
                "MTF_V_HWC_LONG", "MTF_V_HWC_SHORT",
                "MTF_V_MWC_LONG", "MTF_V_MWC_SHORT",
                "MTF_MIN_EVIDENCE_STRENGTH_HWC", "MTF_MIN_EVIDENCE_STRENGTH_MWC",
                "MTF_RETENTION_FLOOR", "MTF_RETENTION_TARGET",
            )
            if hasattr(_cfg, name)
        }
        config_hash = hashlib.sha256(
            json.dumps(config_fields, sort_keys=True,
                       default=str).encode("utf-8")
        ).hexdigest()
        manifest = {
            "schema_version": "2.0.0",
            "dataset_hashes": dataset_hashes,
            "timeframes": {
                "lwc_minutes": int(getattr(_cfg, "LWC_TIMEFRAME_MINUTES", 15)),
                "mwc_minutes": int(getattr(_cfg, "MWC_TIMEFRAME_MINUTES", 60)),
                "hwc_minutes": int(getattr(_cfg, "HWC_TIMEFRAME_MINUTES", 240)),
                "timezone": "UTC",
                "candle_boundary": "fixed_utc",
                "missing_bar_policy": "invalidate_htf_bucket",
                "partial_candle_policy": "drop_unclosed_bucket",
                "release_time_policy": "htf_close_at_or_before_lwc_execution",
                "warmup_policy": {
                    "lwc": "causal_indicator_nan_to_zero",
                    "mwc": "feature_nan_remains_unavailable",
                    "hwc": "feature_nan_remains_unavailable",
                },
            },
            "labels": {
                "hwc_horizon_bars": int(getattr(_cfg, "MTF_HWC_HORIZON_BARS", 6)),
                "mwc_horizon_bars": int(getattr(_cfg, "MTF_MWC_HORIZON_BARS", 4)),
                "mwc_hwc_support_threshold": 0.20,
                **labels_metadata,
            },
            "cross_fitting": {
                "purge_durations_minutes": {
                    "hwc": discovery_purge_minutes("hwc"),
                    "mwc": discovery_purge_minutes("mwc"),
                    "lwc": int(_cfg.MAX_HOLD_CANDLES)
                    * int(_cfg.LWC_TIMEFRAME_MINUTES),
                },
                "embargo_policy": "strict_label_or_trade_horizon_before_prediction_start",
                **cross_metadata,
            },
            "features": feature_metadata,
            "search": search_metadata,
            "archives": {
                "hwc_archive_hash": str(hwc_archive_hash),
                "mwc_archive_hash": str(mwc_archive_hash),
                "lwc_archive_hash": str(lwc_archive_hash),
            },
            "composer_parameters": c_params,
            "composer": c_params,
            "execution": {
                "fee_pct_round_trip": float(getattr(_cfg, "FEE_PCT", 0.0)),
                "slippage_bps_per_side": float(
                    getattr(_cfg, "SLIPPAGE_BPS_PER_SIDE", 0.0)
                ),
                "default_tp": float(getattr(_cfg, "RB_DEFAULT_TP", 2.0)),
                "default_sl": float(getattr(_cfg, "RB_DEFAULT_SL", 1.2)),
                "same_bar_ambiguity": "stop_first_conservative",
            },
            "frozen_runtime": True,
            "release_policy": raw_metadata.pop(
                "release_policy",
                {
                    "oos": "one_shot_no_refit",
                    "weights": "frozen",
                    "thresholds": "frozen",
                },
            ),
            "reproducibility": {
                "git_commit": _git_commit_id(),
                "config_hash": config_hash,
                "global_seed": getattr(_cfg, "GLOBAL_SEED", None),
                "phase2_seed": getattr(_cfg, "PHASE2_SEED", None),
            },
            "created_at": _now_iso(),
            "metadata": raw_metadata,
        }
        temp_path = target_path.with_suffix(target_path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        temp_path.replace(target_path)
        return manifest

    def run_phase1_hwc(
        self,
        train_df: pd.DataFrame | None = None,
        folds: list | None = None,
        force: bool = False,
    ) -> list[dict[str, Any]]:
        """Run Phase 1H: HWC macro directional rule discovery and ensembling."""
        phase_name = "Phase 1H: HWC Rule Discovery"
        start_ts = _now_iso()
        t0 = time.monotonic()
        logger.info("Running %s …", phase_name)

        source = train_df if train_df is not None else self._load_mtf_source_tape()
        if folds is None:
            folds = build_master_temporal_folds(
                source,
                n_folds=int(_cfg.MTF_MAX_FOLDS),
                embargo_minutes=discovery_purge_minutes("hwc"),
            )
        archive_path = Path(self._output_dir) / \
            "rule_archives" / "hwc" / "hwc_rules.json"
        oof_path = archive_path.with_name("hwc_oof_scores.json")
        if not force and archive_path.exists() and oof_path.exists():
            try:
                from gpu_fuzzy_trader.mtf.archives import load_mtf_archive_payload
                from gpu_fuzzy_trader.mtf.discovery import (
                    _build_layer_frame,
                    _frame_hash,
                    _schema_hash,
                    canonicalize_oof_scores,
                    hash_oof_scores,
                )
                payload = load_mtf_archive_payload(archive_path)
                meta = payload.get("metadata", {})
                bars = _build_layer_frame(source, 240, "hwc")
                current_data_hash = _frame_hash(
                    bars.drop(columns=["_move"], errors="ignore"))
                current_schema_hash = _schema_hash(
                    bars.drop(columns=["_move"], errors="ignore"))
                current_fold_boundaries = export_fold_boundaries(folds, df=source)
                current_search_identity = discovery_search_identity("hwc")
                if meta.get("dataset_hash") != current_data_hash:
                    raise ValueError(
                        f"Archive dataset_hash mismatch: expected {current_data_hash}, got {meta.get('dataset_hash')}"
                    )
                if meta.get("feature_schema_hash") != current_schema_hash:
                    raise ValueError(
                        f"Archive feature_schema_hash mismatch: expected {current_schema_hash}, got {meta.get('feature_schema_hash')}"
                    )
                if meta.get("fold_boundaries") != current_fold_boundaries:
                    raise ValueError(
                        "Archive fold_boundaries do not match current temporal folds")
                search_metadata = meta.get("search")
                if (
                    not isinstance(search_metadata, dict)
                    or search_metadata.get("identity") != current_search_identity
                ):
                    raise ValueError(
                        "Archive search identity does not match active HWC settings"
                    )

                oof_records = json.loads(oof_path.read_text(encoding="utf-8"))
                oof_scores = canonicalize_oof_scores(
                    pd.DataFrame.from_records(oof_records))
                current_oof_hash = hash_oof_scores(oof_scores)
                if meta.get("oof_score_hash") != current_oof_hash:
                    raise ValueError(
                        f"Archive oof_score_hash mismatch: expected {current_oof_hash}, got {meta.get('oof_score_hash')}"
                    )

                discovery = LayerDiscoveryResult(
                    timeframe="hwc",
                    rules=payload.get("rules", []),
                    oof_scores=oof_scores,
                    bars=bars,
                    feature_schema_hash=current_schema_hash,
                    data_hash=current_data_hash,
                    theta_per_oof_fold=meta.get("theta_per_oof_fold", {}),
                    theta_final_train=float(
                        meta.get("theta_final_train", 0.0)),
                    fold_rules={},
                    search_metadata=meta.get("search", {}),
                    oof_score_hash=current_oof_hash,
                )
                self._mtf_hwc_discovery = discovery
                logger.info("Reusing verified HWC rule archive from %s (%d rules)",
                            archive_path, len(discovery.rules))
                elapsed = time.monotonic() - t0
                _log_phase_entry(
                    self._log_path, phase_name, start_ts, _now_iso(), elapsed, skipped=True,
                    result_summary={"hwc_rules": len(
                        discovery.rules), "archive_hash": payload.get("archive_hash", "")},
                )
                return discovery.rules
            except Exception as exc:
                logger.warning(
                    "Failed to validate and resume HWC archive from %s (%s); running discovery", archive_path, exc)

        discovery = discover_directional_layer(source, role="hwc", folds=folds)
        self._mtf_hwc_discovery = discovery
        metadata = {
            "role": "hwc",
            "dataset_hash": discovery.data_hash,
            "feature_schema_hash": discovery.feature_schema_hash,
            "fold_boundaries": export_fold_boundaries(folds, df=source),
            "theta_per_oof_fold": discovery.theta_per_oof_fold,
            "theta_final_train": discovery.theta_final_train,
            "oof_score_hash": discovery.oof_score_hash,
            "purge_minutes": discovery_purge_minutes("hwc"),
            "search": discovery.search_metadata,
        }
        archive_hash = save_mtf_rule_archive(
            timeframe="hwc",
            rules=discovery.rules,
            path=archive_path,
            metadata=metadata,
            require_provenance=True,
        )
        oof_path = archive_path.with_name("hwc_oof_scores.json")
        oof_path.write_text(
            json.dumps(
                discovery.oof_scores.to_dict(orient="records"),
                indent=2,
                sort_keys=True,
                default=str,
            ),
            encoding="utf-8",
        )

        elapsed = time.monotonic() - t0
        _log_phase_entry(
            self._log_path, phase_name, start_ts, _now_iso(), elapsed, skipped=False,
            result_summary={"hwc_rules": len(
                discovery.rules), "archive_hash": archive_hash},
        )
        return discovery.rules

    def run_phase1_mwc(
        self,
        train_df: pd.DataFrame | None = None,
        hwc_rules: list[dict[str, Any]] | None = None,
        folds: list | None = None,
        force: bool = False,
    ) -> list[dict[str, Any]]:
        """Run Phase 1M: MWC setup confirmation rule discovery (OOF-conditioned) and ensembling."""
        phase_name = "Phase 1M: MWC Rule Discovery"
        start_ts = _now_iso()
        t0 = time.monotonic()
        logger.info("Running %s …", phase_name)

        source = train_df if train_df is not None else self._load_mtf_source_tape()
        if folds is None:
            folds = build_master_temporal_folds(
                source,
                n_folds=int(_cfg.MTF_MAX_FOLDS),
                embargo_minutes=discovery_purge_minutes("hwc"),
            )
        upstream = getattr(self, "_mtf_hwc_discovery", None)
        if upstream is None:
            logger.warning(
                "MWC requires current HWC OOF scores; resolving HWC before "
                "MWC archive reuse",
            )
            self.run_phase1_hwc(source, folds=list(folds), force=force)
            upstream = getattr(self, "_mtf_hwc_discovery", None)
        if upstream is None:
            raise RuntimeError(
                "Cannot run MWC discovery without verified HWC OOF scores.",
            )
        current_hwc_oof_hash = getattr(upstream, "oof_score_hash", "")
        if not isinstance(current_hwc_oof_hash, str) or not current_hwc_oof_hash:
            raise RuntimeError(
                "Cannot run MWC discovery without an HWC OOF score hash.",
            )
        archive_path = Path(self._output_dir) / \
            "rule_archives" / "mwc" / "mwc_rules.json"
        oof_path = archive_path.with_name("mwc_oof_scores.json")
        if not force and archive_path.exists() and oof_path.exists():
            try:
                from gpu_fuzzy_trader.mtf.archives import load_mtf_archive_payload
                from gpu_fuzzy_trader.mtf.discovery import (
                    _build_layer_frame,
                    _frame_hash,
                    _schema_hash,
                    canonicalize_oof_scores,
                    hash_oof_scores,
                )
                payload = load_mtf_archive_payload(archive_path)
                meta = payload.get("metadata", {})
                bars = _build_layer_frame(source, 60, "mwc")
                current_data_hash = _frame_hash(
                    bars.drop(columns=["_move"], errors="ignore"))
                current_schema_hash = _schema_hash(
                    bars.drop(columns=["_move"], errors="ignore"))
                current_fold_boundaries = export_fold_boundaries(folds, df=source)
                current_search_identity = discovery_search_identity("mwc")
                if meta.get("dataset_hash") != current_data_hash:
                    raise ValueError(
                        f"Archive dataset_hash mismatch: expected {current_data_hash}, got {meta.get('dataset_hash')}"
                    )
                if meta.get("feature_schema_hash") != current_schema_hash:
                    raise ValueError(
                        f"Archive feature_schema_hash mismatch: expected {current_schema_hash}, got {meta.get('feature_schema_hash')}"
                    )
                if meta.get("fold_boundaries") != current_fold_boundaries:
                    raise ValueError(
                        "Archive fold_boundaries do not match current temporal folds")
                search_metadata = meta.get("search")
                if (
                    not isinstance(search_metadata, dict)
                    or search_metadata.get("identity") != current_search_identity
                ):
                    raise ValueError(
                        "Archive search identity does not match active MWC settings"
                    )

                oof_records = json.loads(oof_path.read_text(encoding="utf-8"))
                oof_scores = canonicalize_oof_scores(
                    pd.DataFrame.from_records(oof_records))
                current_oof_hash = hash_oof_scores(oof_scores)
                if meta.get("oof_score_hash") != current_oof_hash:
                    raise ValueError(
                        f"Archive oof_score_hash mismatch: expected {current_oof_hash}, got {meta.get('oof_score_hash')}"
                    )
                if meta.get("hwc_oof_score_hash") != current_hwc_oof_hash:
                    raise ValueError(
                        "Archive hwc_oof_score_hash does not match current "
                        "HWC OOF scores"
                    )

                discovery = LayerDiscoveryResult(
                    timeframe="mwc",
                    rules=payload.get("rules", []),
                    oof_scores=oof_scores,
                    bars=bars,
                    feature_schema_hash=current_schema_hash,
                    data_hash=current_data_hash,
                    theta_per_oof_fold=meta.get("theta_per_oof_fold", {}),
                    theta_final_train=float(
                        meta.get("theta_final_train", 0.0)),
                    fold_rules={},
                    search_metadata=meta.get("search", {}),
                    oof_score_hash=current_oof_hash,
                )
                self._mtf_mwc_discovery = discovery
                logger.info("Reusing verified MWC rule archive from %s (%d rules)",
                            archive_path, len(discovery.rules))
                elapsed = time.monotonic() - t0
                _log_phase_entry(
                    self._log_path, phase_name, start_ts, _now_iso(), elapsed, skipped=True,
                    result_summary={"mwc_rules": len(
                        discovery.rules), "archive_hash": payload.get("archive_hash", "")},
                )
                return discovery.rules
            except Exception as exc:
                logger.warning(
                    "Failed to validate and resume MWC archive from %s (%s); running discovery", archive_path, exc)

        discovery = discover_directional_layer(
            source,
            role="mwc",
            folds=folds,
            upstream_oof_scores=upstream.oof_scores,
        )
        self._mtf_mwc_discovery = discovery
        metadata = {
            "role": "mwc",
            "conditioned_on": "hwc_oof_scores_only",
            "dataset_hash": discovery.data_hash,
            "feature_schema_hash": discovery.feature_schema_hash,
            "fold_boundaries": export_fold_boundaries(folds, df=source),
            "theta_per_oof_fold": discovery.theta_per_oof_fold,
            "theta_final_train": discovery.theta_final_train,
            "oof_score_hash": discovery.oof_score_hash,
            "hwc_oof_score_hash": current_hwc_oof_hash,
            "purge_minutes": discovery_purge_minutes("mwc"),
            "search": discovery.search_metadata,
        }
        archive_hash = save_mtf_rule_archive(
            timeframe="mwc",
            rules=discovery.rules,
            path=archive_path,
            metadata=metadata,
            require_provenance=True,
        )
        archive_path.with_name("mwc_oof_scores.json").write_text(
            json.dumps(
                discovery.oof_scores.to_dict(orient="records"),
                indent=2,
                sort_keys=True,
                default=str,
            ),
            encoding="utf-8",
        )

        elapsed = time.monotonic() - t0
        _log_phase_entry(
            self._log_path, phase_name, start_ts, _now_iso(), elapsed, skipped=False,
            result_summary={"mwc_rules": len(discovery.rules), "archive_hash": archive_hash},
        )
        return discovery.rules


    def run_phase2(
        self,
        train_df: pd.DataFrame,
        phase1_result: dict[str, list[dict]] | None = None,
        force: bool = False,
        val_df: pd.DataFrame | None = None,
        blocked_directions: frozenset[str] | None = None,
    ) -> dict:
        """Run Phase 2 (LWC Rule Pool Generation) and persist LWC archive."""
        if phase1_result is None:
            phase1_result = {"long": [], "short": []}
        res = self._run_phase2(
            train_df=train_df,
            phase1_result=phase1_result,
            force=force,
            val_df=val_df,
            blocked_directions=blocked_directions,
        )
        lwc_archive_path = Path(self._output_dir) / "rule_archives" / "lwc" / "lwc_rules.json"
        all_lwc_rules = []
        for direction, rules in res.items():
            for r in rules:
                rule_copy = dict(r)
                rule_copy["timeframe"] = "lwc"
                rule_copy["direction"] = direction
                rule_copy["coverage"] = float(r.get("coverage", 0.15))
                all_lwc_rules.append(rule_copy)
        if all_lwc_rules:
            save_mtf_rule_archive(timeframe="lwc", rules=all_lwc_rules, path=lwc_archive_path)
        return res

    def run_mtf_composition(
        self,
        lwc_rules: dict[str, list[dict]] | list[dict] | None = None,
        hwc_rules: list[dict] | None = None,
        mwc_rules: list[dict] | None = None,
        composer_params: dict[str, Any] | None = None,
        df: pd.DataFrame | None = None,
        folds: list | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, HierarchicalStrategyCandidate]:
        """Run MTF signal composition, apply asymmetric soft veto, verify retention, and generate manifest."""
        phase_name = "Phase MTF: Signal Composition & Manifest"
        start_ts = _now_iso()
        t0 = time.monotonic()
        logger.info("Running %s …", phase_name)

        hwc_path = Path(self._output_dir) / "rule_archives" / "hwc" / "hwc_rules.json"
        mwc_path = Path(self._output_dir) / "rule_archives" / "mwc" / "mwc_rules.json"
        lwc_path = Path(self._output_dir) / "rule_archives" / "lwc" / "lwc_rules.json"

        hwc_loaded_from_archive = False
        if hwc_rules is None:
            if hwc_path.exists():
                hwc_rules = load_mtf_rule_archive(hwc_path)
                hwc_loaded_from_archive = True
            else:
                hwc_rules = self.run_phase1_hwc()

        mwc_loaded_from_archive = False
        if mwc_rules is None:
            if mwc_path.exists():
                mwc_rules = load_mtf_rule_archive(mwc_path)
                mwc_loaded_from_archive = True
            else:
                mwc_rules = self.run_phase1_mwc(hwc_rules=hwc_rules)

        lwc_long: list[dict] = []
        lwc_short: list[dict] = []
        lwc_loaded_from_archive = False
        if isinstance(lwc_rules, dict):
            lwc_long = lwc_rules.get("long", [])
            lwc_short = lwc_rules.get("short", [])
        elif isinstance(lwc_rules, list):
            for r in lwc_rules:
                rule_direction = str(r.get("direction", "")).strip().lower()
                if rule_direction == "long":
                    lwc_long.append(r)
                elif rule_direction == "short":
                    lwc_short.append(r)
                else:
                    raise ValueError(
                        f"LWC rule has invalid direction {r.get('direction')!r}"
                    )
        elif lwc_path.exists():
            lwc_loaded_from_archive = True
            for r in load_mtf_rule_archive(lwc_path):
                rule_direction = str(r.get("direction", "")).strip().lower()
                if rule_direction == "long":
                    lwc_long.append(r)
                elif rule_direction == "short":
                    lwc_short.append(r)
                else:
                    raise ValueError(
                        f"LWC archive rule has invalid direction {r.get('direction')!r}"
                    )

        hwc_discovery = getattr(self, "_mtf_hwc_discovery", None)
        mwc_discovery = getattr(self, "_mtf_mwc_discovery", None)
        all_lwc = lwc_long + lwc_short

        def loaded_archive_hash(path: Path) -> str:
            from gpu_fuzzy_trader.mtf.archives import load_mtf_archive_payload

            archive_hash = load_mtf_archive_payload(path).get("archive_hash")
            if not isinstance(archive_hash, str) or not archive_hash:
                raise ValueError(f"MTF archive has no archive hash: {path}")
            return archive_hash

        if not hwc_rules:
            hwc_hash = ""
        elif hwc_loaded_from_archive:
            hwc_hash = loaded_archive_hash(hwc_path)
        else:
            hwc_metadata = None
            if hwc_discovery is not None:
                hwc_metadata = {
                    "role": "hwc",
                    "dataset_hash": hwc_discovery.data_hash,
                    "feature_schema_hash": hwc_discovery.feature_schema_hash,
                    "fold_boundaries": export_fold_boundaries(
                        folds or [], df=df,
                    ),
                    "theta_per_oof_fold": hwc_discovery.theta_per_oof_fold,
                    "theta_final_train": hwc_discovery.theta_final_train,
                    "oof_score_hash": hwc_discovery.oof_score_hash,
                    "purge_minutes": discovery_purge_minutes("hwc"),
                    "search": hwc_discovery.search_metadata,
                }
            hwc_hash = save_mtf_rule_archive(
                "hwc",
                hwc_rules,
                path=hwc_path,
                metadata=hwc_metadata,
                require_provenance=hwc_metadata is not None,
            )

        if not mwc_rules:
            mwc_hash = ""
        elif mwc_loaded_from_archive:
            mwc_hash = loaded_archive_hash(mwc_path)
        else:
            mwc_metadata = None
            if mwc_discovery is not None:
                if hwc_discovery is None:
                    raise RuntimeError(
                        "Cannot serialize MWC discovery without HWC OOF provenance.",
                    )
                mwc_metadata = {
                    "role": "mwc",
                    "conditioned_on": "hwc_oof_scores_only",
                    "dataset_hash": mwc_discovery.data_hash,
                    "feature_schema_hash": mwc_discovery.feature_schema_hash,
                    "fold_boundaries": export_fold_boundaries(
                        folds or [], df=df,
                    ),
                    "theta_per_oof_fold": mwc_discovery.theta_per_oof_fold,
                    "theta_final_train": mwc_discovery.theta_final_train,
                    "oof_score_hash": mwc_discovery.oof_score_hash,
                    "hwc_oof_score_hash": hwc_discovery.oof_score_hash,
                    "purge_minutes": discovery_purge_minutes("mwc"),
                    "search": mwc_discovery.search_metadata,
                }
            mwc_hash = save_mtf_rule_archive(
                "mwc",
                mwc_rules,
                path=mwc_path,
                metadata=mwc_metadata,
                require_provenance=mwc_metadata is not None,
            )

        if not all_lwc:
            lwc_hash = ""
        elif lwc_loaded_from_archive:
            lwc_hash = loaded_archive_hash(lwc_path)
        else:
            lwc_hash = save_mtf_rule_archive("lwc", all_lwc, path=lwc_path)

        candidates: dict[str, HierarchicalStrategyCandidate] = {}
        for direction, dir_lwc in (("long", lwc_long), ("short", lwc_short)):
            candidate = HierarchicalStrategyCandidate(
                direction=direction,
                lwc_rules=dir_lwc,
                # Both directions must remain in the ensemble so Direction
                # and Evidence Strength are computed from the complete archive.
                hwc_rules=hwc_rules,
                mwc_rules=mwc_rules,
                composer_params=composer_params,
            )
            candidates[direction] = candidate

        runtime_retention: dict[str, Any] = {}
        if df is not None:
            from gpu_fuzzy_trader.mtf.diagnostics import compute_granular_retention_diagnostics

            for direction, candidate in candidates.items():
                try:
                    _, stats, audit = evaluate_candidate_frame(candidate, df)
                    if folds:
                        fold_starts = np.asarray(
                            [pd.Timestamp(fold.test_start).value for fold in folds],
                            dtype=np.int64,
                        )
                        audit_times = pd.to_datetime(
                            audit["datetime"], errors="raise", utc=True
                        ).astype("int64").to_numpy()
                        audit["fold_id"] = np.maximum(
                            1,
                            np.searchsorted(
                                fold_starts, audit_times, side="right"
                            ),
                        )
                    else:
                        audit["fold_id"] = 0
                    runtime_retention[direction] = {
                        "funnel": stats.get("retention_diagnostics", {}),
                        "granular": compute_granular_retention_diagnostics(audit),
                    }
                    candidate.metadata.update({
                        "retention_diagnostics": runtime_retention[direction],
                        "runtime_evaluated": True,
                    })
                except Exception as exc:
                    logger.error("MTF composition evaluation failed for %s: %s", direction, exc)
                    runtime_retention[direction] = {
                        "status": "ERROR",
                        "error": f"{type(exc).__name__}: {exc}",
                    }

        manifest_metadata = dict(metadata or {})
        manifest_metadata["retention_diagnostics"] = runtime_retention
        manifest = self.build_mtf_manifest(
            hwc_archive_hash=hwc_hash,
            mwc_archive_hash=mwc_hash,
            lwc_archive_hash=lwc_hash,
            composer_params=composer_params,
            metadata=manifest_metadata,
        )
        for candidate in candidates.values():
            candidate.mtf_manifest = manifest
        self._mtf_candidates = candidates
        self._mtf_manifest = manifest

        elapsed = time.monotonic() - t0
        _log_phase_entry(
            self._log_path, phase_name, start_ts, _now_iso(), elapsed, skipped=False,
            result_summary={"candidates": list(candidates.keys())},
        )
        return candidates

    def run_rb_governor(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        phase2_result: dict[str, list[dict]],
        cv_folds: list | None = None,
        val_selection_df: pd.DataFrame | None = None,
    ) -> dict:
        """Run RB Governor portfolio selection and risk tuning."""
        return self._run_rb_governor(
            train_df=train_df,
            val_df=val_df,
            phase2_result=phase2_result,
            cv_folds=cv_folds,
            val_selection_df=val_selection_df,
        )

    def run_phase5_oos(
        self,
        allowed_directions: frozenset[str] | None = None,
        test_csv_path: str | None = None,
    ) -> dict:
        """Run Phase 5 Out-of-Sample Evaluation."""
        return self._run_phase5(
            allowed_directions=allowed_directions,
            test_csv_path=test_csv_path,
        )


Pipeline_Runner = Pipeline_Orchestrator


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
