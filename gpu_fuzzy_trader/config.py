"""
Single source of truth for pipeline hyperparameters.

All modules import from here; do not duplicate defaults elsewhere.

File layout
-----------
  1. Global randomness (``GLOBAL_SEED``, ``get_seed``)
  2. Phase 0 — paths, schema, train/val split (holdout+embargo), backtest, logging
  3. Phase 1 — feature selection + GPU row budget (Phase 1→2 bridge)
  4. Phase 2 — rule evolution (NSGA-III): risk, genome, gates
  5. RB Governor — unified rule selection + risk tuning
  6. Monthly windows — shared validation penalties
  7. Phase 5 — consumed test diagnostics plus optional untouched forward acceptance
  8. Helpers — path resolvers, trade-floor scaling, Phase 2 floors
  9. Configuration validation + Colab runtime defaults

Pipeline phases
---------------
  Phase 0  Paths, schema, train/val split (holdout+embargo), backtest constants
  Phase 1  Feature selection (train_new.csv only)
  Phase 2  NSGA-III rule-pool evolution (GPU backtests)
  RB       Rule-team selection + walk-forward TP/SL/capital optimization
  Phase 5  Test diagnostics; optional FORWARD_CSV_PATH acceptance

Detailed behaviour and formulas: README.md and RUN.md; evaluator parity is
defined by the read-only evaluator_v5.ipynb notebook.

Tuning cheat-sheet (symptom → knob)
-----------------------------------
  Short OOS / overfitting          RB_* gates, PHASE2_JOINT_TRAIN_VAL
  GPU OOM                          PHASE1_SAMPLING_TOTAL ↓, PHASE2_GPU_BATCH_SIZE ↓,
                                   PHASE2_SCAN_UNROLL ↓
  Phase 2 too slow                 PHASE2_GENERATIONS ↓, PHASE2_USE_GPU
  Empty Phase 2 pool               MIN_TRADE_SUPPORT ↓, PHASE2_*_FLOOR ↓
  Too many weak / noisy rules      MIN_TRADE_SUPPORT ↑, MIN_CONDITIONS ↑,
                                   PHASE2_*_FLOOR ↑, PHASE2_MAX_DRAWDOWN_GATE ↓
  RB finds no teams                RB_MIN_* ↓, RB_KEEP_TOP_RULES ↑
  Fees / horizon mismatch          FEE_PCT, TAIL_DROP_ROWS, MAX_HOLD_CANDLES,
                                   HOLDOUT_EMBARGO_CANDLES
                                   (must match evaluator_v5.ipynb)
  RB Governor too strict           RB_MIN_* ↓, RB_KEEP_TOP_RULES ↑

Environment overrides: DATA_ROOT, RAW_TRAIN_CSV_PATH, RAW_TEST_CSV_PATH,
                       ENRICHED_DIR, ENRICHED_TRAIN_PATH, ENRICHED_TEST_PATH,
                       ENRICHED_MANIFEST_PATH, TRAIN_CSV_PATH, TEST_CSV_PATH,
                       FORWARD_CSV_PATH,
                       PHASE2_GPU_BATCH_SIZE, PHASE2_GPU_BATCH_SIZE_AUTO
"""


from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

# Repo root (parent of gpu_fuzzy_trader/) — paths outside per-run OUTPUTS_DIR.
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir))

_logger = logging.getLogger(__name__)


# =============================================================================
# Global randomness
# =============================================================================

# GLOBAL_SEED
#   None  → one cryptographically random seed per process (explicit exploration).
#   int   → fully reproducible runs.
# Higher/lower: N/A — only None vs fixed integer matters for reproducibility.
GLOBAL_SEED: int | None = 42

_PROCESS_SEED: int | None = None


def get_seed() -> int:
    """Return a per-process random seed (stable within a run, different across runs).

    If ``GLOBAL_SEED`` is set to an integer, that value is always returned.
    Otherwise a cryptographically random seed is drawn once and reused for the
    lifetime of the process so that all modules share the same seed.
    """
    global _PROCESS_SEED
    if GLOBAL_SEED is not None:
        return int(GLOBAL_SEED)
    if _PROCESS_SEED is None:
        _PROCESS_SEED = int.from_bytes(os.urandom(4), "big")
    return _PROCESS_SEED


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


# =============================================================================
# Phase 0 — Paths & outputs
# =============================================================================
# Relative paths resolve from process cwd (usually repo root).
# run_pipeline.py may rewrite OUTPUTS_DIR and Phase 2 pool paths per --output.

DATA_ROOT = os.environ.get("DATA_ROOT", "").strip()
# Raw market data is the canonical production input. The hierarchical MTF
# pipeline builds HWC/MWC/LWC bars and features from these tapes in memory.
RAW_TRAIN_CSV_PATH = _env_str(
    "RAW_TRAIN_CSV_PATH",
    os.path.join(DATA_ROOT, "train_new.csv") if DATA_ROOT else "data/train_new.csv",
)
RAW_TEST_CSV_PATH = _env_str(
    "RAW_TEST_CSV_PATH",
    os.path.join(DATA_ROOT, "test_new.csv") if DATA_ROOT else "data/test_new.csv",
)

# ---------------------------------------------------------------------------
# Deprecated enrichment paths retained only for explicit legacy diagnostics.
# The canonical hierarchical MTF pipeline consumes the raw tapes below and
# builds complete higher-timeframe bars in memory from the frozen source data.
# ---------------------------------------------------------------------------
ENRICHED_DIR = _env_str(
    "ENRICHED_DIR",
    os.path.join(DATA_ROOT, "enriched") if DATA_ROOT else "data/enriched",
)
ENRICHED_TRAIN_PATH = _env_str(
    "ENRICHED_TRAIN_PATH",
    os.path.join(ENRICHED_DIR, "train_new_hwc_mwc_lwc.csv"),
)
ENRICHED_TEST_PATH = _env_str(
    "ENRICHED_TEST_PATH",
    os.path.join(ENRICHED_DIR, "test_new_hwc_mwc_lwc.csv"),
)
ENRICHED_MANIFEST_PATH = _env_str(
    "ENRICHED_MANIFEST_PATH",
    os.path.join(ENRICHED_DIR, "trend_context_manifest.json"),
)

# Production inputs default to raw source tapes.  Explicit path overrides are
# allowed for a controlled raw OHLCV pair; legacy enriched inputs are accepted
# only when the MTF pipeline is explicitly disabled.
TRAIN_CSV_PATH = _env_str("TRAIN_CSV_PATH", RAW_TRAIN_CSV_PATH)
TEST_CSV_PATH = _env_str("TEST_CSV_PATH", RAW_TEST_CSV_PATH)
# The checked-in test tape is a consumed diagnostic holdout.  A strategy is
# never marked deployment-accepted from it; provide a strictly newer untouched
# future tape through FORWARD_CSV_PATH for release acceptance.
FORWARD_CSV_PATH = os.environ.get("FORWARD_CSV_PATH", "").strip() or None

# Research-integrity artifacts.  The consumed test tape is never a selection
# input; these files make dataset lineage and adaptive trial counts auditable.
EXPERIMENT_LEDGER_ENABLED: bool = True
DATASET_MANIFEST_ENABLED: bool = True
NESTED_VALIDATION_ENABLED: bool = True
NESTED_VALIDATION_OUTER_FOLDS: int = 3
# A forward tape is a release acceptance observation, not a reusable
# validation set.  A given output directory may consume it only once.
FORWARD_ACCEPTANCE_ONCE: bool = True

# Cached splits from train_new.csv (Phases 2–5). Rebuilt when train_new.csv is newer.
TRAIN_70_PATH = "data/train_70.parquet"
VALIDATION_30_PATH = "data/validation_30.parquet"
VALIDATION_FITNESS_PATH = "data/validation_fitness.parquet"
VALIDATION_SELECTION_PATH = "data/validation_selection.parquet"

OUTPUTS_DIR = "outputs"
RUN_LOG_PATH = os.path.join(OUTPUTS_DIR, "run.log")
REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")

PHASE2_POOL_PATHS = {
    "long": os.path.join(OUTPUTS_DIR, "phase2_long_pool.json"),
    "short": os.path.join(OUTPUTS_DIR, "phase2_short_pool.json"),
}
PHASE2_HISTORY_PATHS = {
    "long": os.path.join(OUTPUTS_DIR, "phase2_long_history.json"),
    "short": os.path.join(OUTPUTS_DIR, "phase2_short_history.json"),
}

# Cross-run warm-start (not cleared by --output).
PHASE2_ARCHIVE_DIR = os.path.join(_PROJECT_ROOT, "phase2_rule_archive")
PHASE2_ARCHIVE_PATHS = {
    "long": os.path.join(PHASE2_ARCHIVE_DIR, "phase2_long_archive.json"),
    "short": os.path.join(PHASE2_ARCHIVE_DIR, "phase2_short_archive.json"),
}

# Multi-timeframe rule archives & manifest
MTF_ARCHIVE_DIR = os.path.join(OUTPUTS_DIR, "rule_archives")
MTF_ARCHIVE_PATHS = {
    "hwc": os.path.join(MTF_ARCHIVE_DIR, "hwc", "hwc_rules.json"),
    "mwc": os.path.join(MTF_ARCHIVE_DIR, "mwc", "mwc_rules.json"),
    "lwc": os.path.join(MTF_ARCHIVE_DIR, "lwc", "lwc_rules.json"),
}
MTF_MANIFEST_PATH = os.path.join(OUTPUTS_DIR, "mtf_manifest.json")

# Debug: scope pipeline to N symbols starting at DEBUG_SYMBOL (sorted universe).
DEBUG_SYMBOL_SCOPE_ENABLED = False
DEBUG_SYMBOL = "BTCUSDT"
DEBUG_SYMBOL_COUNT = 1
# =============================================================================
# Phase 0 — Schema & labels
# =============================================================================
# LABEL_* / META_* never enter feature matrices. INTERNAL_* is loader-only.

LABEL_COLUMNS = [
    "label_open_next",
    "label_close_288",
    "label_min_288",
    "label_max_288",
    "label_max_before_min",
]
META_COLUMNS = ["datetime", "symbol"]
INTERNAL_COLUMNS = ("_symbol_bar_index",)

# Raw OHLCV price levels are not fuzzy indicators: evaluator-v5's fixed
# ``[-1, 1]`` condition thresholds would turn them into permanent signals.
# The replacement data provides the bounded ``ff_*`` features for rule search.
PHASE1_EXCLUDE_RAW_OHLCV = True

# FILL_NA_WITH_ZERO — whether Data_Loader fills NaN feature values with 0.
# Defaults to False so unavailable warmup periods are preserved as NaN and not
# converted into artificial neutral/zero signals.
FILL_NA_WITH_ZERO = False

# TAIL_DROP_ROWS — bars dropped per symbol at dataset tail (label horizon).
# Must equal MAX_HOLD_CANDLES (96 = 24 h at 15-minute bars).
# The forward-label window is now 96 bars; the ``_288`` label column names are
# retained temporarily for schema compatibility with evaluator_v5.ipynb and
# persisted artifacts, but the actual horizon everywhere is 96 bars.
#   Higher → more rows removed, safer labels, less training data.
#   Lower  → more rows kept, risk of NaN / lookahead leakage at symbol tails.
TAIL_DROP_ROWS = 96


# =============================================================================
# Phase 0 — Train / validation split (Phase 2 and RB)
# =============================================================================
# Phases 4–5 always use persisted train_70 + validation_30 (see splitter.py).

# SPLIT_MODE — how train_new.csv is divided before Phase 2.
#   holdout             → single per-symbol chronological split with embargo
#                         (HOLDOUT_EMBARGO_CANDLES bars dropped between train
#                         and val — 96, not the legacy 288-bar horizon).
#                         The actual train/val fraction is set by
#                         HOLDOUT_TRAIN_FRACTION (see below).
#   purged_walk_forward → expanding CV folds + primary tail holdout with embargo
#                         (deprecated — use holdout instead).
SPLIT_MODE = "holdout"

# HOLDOUT_TRAIN_FRACTION — fraction of each symbol's bars reserved for training.
# The remaining (1 - HOLDOUT_TRAIN_FRACTION) is validation, with an embargo gap
# of HOLDOUT_EMBARGO_CANDLES bars dropped between them to prevent label lookahead.
HOLDOUT_TRAIN_FRACTION = 0.65

# HOLDOUT_EMBARGO_CANDLES — bars dropped between train and val (label horizon).
# Must equal MAX_HOLD_CANDLES so a position opened at the last train bar cannot
# look into the validation window.
HOLDOUT_EMBARGO_CANDLES = 96


def train_prefix_row_count(n_rows: int, train_frac: float | None = None) -> int:
    """Number of leading per-symbol rows belonging to the training prefix.

    Shared by ``Data_Splitter`` (Phase-0 train/validation split) and
    ``trend_context`` (threshold fitting) so both always agree on exactly
    which rows are "train": trend-context thresholds must never be fitted on
    rows that later become the validation split.
    """
    frac = float(HOLDOUT_TRAIN_FRACTION) if train_frac is None else float(train_frac)
    return math.floor(int(n_rows) * frac)


def holdout_train_val_label(frac: float = HOLDOUT_TRAIN_FRACTION) -> str:
    """Return a human-friendly 'train/val' label derived from *frac*.

    Example: ``frac=0.65`` → ``"65/35"``, ``frac=0.70`` → ``"70/30"``.

    Intended for log / status messages so the percentage is always in sync
    with ``HOLDOUT_TRAIN_FRACTION`` instead of being baked into a string.
    """
    train_pct = int(round(frac * 100))
    val_pct = 100 - train_pct
    return f"{train_pct}/{val_pct}"


# --- Purged walk-forward (when SPLIT_MODE == purged_walk_forward) ---
# NOTE: Purged walk-forward CV is deprecated. SPLIT_MODE is now "holdout"
# with HOLDOUT_TRAIN_FRACTION + HOLDOUT_EMBARGO_CANDLES (see above).
# The purged-WF keys below are retained for reference but are inactive when
# SPLIT_MODE != "purged_walk_forward".

# PURGED_WF_N_SPLITS — number of CV folds on the train prefix (K).
#   K CV folds are built on the first (1 - HOLDOUT_FRACTION) of each symbol;
#   a separate primary holdout (validation_30) is always appended (K+1 total).
PURGED_WF_N_SPLITS = 2

# PURGED_WF_HOLDOUT_FRACTION — tail fraction per symbol reserved for val parquet.
PURGED_WF_HOLDOUT_FRACTION = 0.3

# PURGED_WF_EMBARGO_CANDLES — purge gap between train and valid (label horizon).
PURGED_WF_EMBARGO_CANDLES = 96

# PURGED_WF_MIN_TRAIN_FRACTION — minimum train prefix before first CV valid block.
# Set to 0.4 so the strict no-leak safe region (prefix - embargo) is wide enough
# to fit PHASE1_SAMPLING_TOTAL per-symbol bars without overlap.
PURGED_WF_MIN_TRAIN_FRACTION = 0.4

# PURGED_WF_MIN_VALID_ROWS — minimum rows in a CV valid block (holdout exempt).
PURGED_WF_MIN_VALID_ROWS = 3000

# PURGED_WF_AGGREGATION — combine per-fold metrics: worst | mean.
# mean averages return/PF/Sortino; still uses max drawdown and min trades.
# Prefer mean when per-fold trade floors are low (scaled slices) so one noisy
# fold does not dominate fitness; worst is more conservative for large folds.
PURGED_WF_AGGREGATION = "mean"

# PURGED_WF_REQUIRE_ALL_CV_FOLDS — pool admission also checks every CV fold.
PURGED_WF_REQUIRE_ALL_CV_FOLDS = False

# PURGED_WF_SCALE_TRADE_FLOORS — scale trade-count gates by slice row fraction.
PURGED_WF_SCALE_TRADE_FLOORS = True

# PURGED_WF_MIN_TRADE_FLOOR_ABSOLUTE — floor after proportional scaling.
PURGED_WF_MIN_TRADE_FLOOR_ABSOLUTE = 5

CV_FOLDS_MANIFEST_PATH = "data/cv_folds_manifest.json"

# Set at split time; used by scale_trade_floor when purged mode is active.
_PURGED_WF_REFERENCE_ROWS: int | None = None


# =============================================================================
# Phase 0 — Backtest simulation (must match evaluator_v5.ipynb)
# =============================================================================
# Used by cpu_engine / gpu_engine in all phases.

# INITIAL_CAPITAL — starting equity for simulated PnL / return %.
#   Higher → absolute PnL scales; relative metrics (return %, Sortino) unchanged.
INITIAL_CAPITAL = 1000.0

# LEVERAGE — position size multiplier on notional.
#   Higher → larger gains and losses per trade; drawdown % grows proportionally.
#   Lower  → more conservative sizing (1.0 = no leverage).
LEVERAGE = 1.0

# FEE_PCT — round-trip fee as % of notional per trade.
#   Higher → penalizes high-turnover rules; net return and PF drop for active rules.
#   Lower  → optimistic backtest; must match evaluator_v5.ipynb for valid OOS.
FEE_PCT = 0.20
# SPREAD_BPS — bid-ask spread cost in basis points (1 bp = 0.01% = 0.0001).
SPREAD_BPS = 0.0
# SLIPPAGE_BPS — execution slippage in basis points (1 bp = 0.01% = 0.0001).
SLIPPAGE_BPS = 0.0
# Identifier included in strategy packages so a fee/execution-model change
# cannot silently reuse an old economic strategy identity.
COST_MODEL_ID: str = "crypto_bar_v2"

# MAX_HOLD_CANDLES — force-exit horizon (bars) when neither TP nor SL hits.
#   Higher → longer holds, larger label window, must match TAIL_DROP_ROWS.
#   Lower  → quicker time exits, more fee drag if rules fire often.
# 96 bars = 24 hours at 15-minute bars (was 288 = 72 h).
MAX_HOLD_CANDLES = 96

# VALIDATION_HALF_PURGE_CANDLES — bars removed on both sides of the internal
# validation fitness/selection boundary.  The forward labels consume the next
# MAX_HOLD_CANDLES bars, so the same horizon is required here as at the main
# train/validation boundary.
VALIDATION_HALF_PURGE_CANDLES = MAX_HOLD_CANDLES

# MAX_TOTAL_EXPOSURE_PCT — cap on sum of concurrent rule capital allocations.
# Must remain aligned with evaluator_v5.ipynb and RB_MAX_TOTAL_CAPITAL.
MAX_TOTAL_EXPOSURE_PCT = 100.0

# MIN_POSITION_NOTIONAL — skip trades below this dollar size.
#   Higher → filters dust trades; may reduce trade count on small capital.
#   Lower  → more micro-trades counted toward support metrics.
MIN_POSITION_NOTIONAL = 1.0

def resolve_backtest_workers(requested: int | None = None) -> int:
    """Resolve the CPU worker cap for batched rule-set simulations.

    When ``cpu_count <= 2`` (e.g. 2-core cloud/CI environments), workers are capped at 2.
    Otherwise, capped at min(8, cpu_count).
    If an explicit worker count is requested, it is respected but bounded by the hardware cap.
    """
    cpus = os.cpu_count() or 1
    if cpus <= 2:
        hw_cap = min(2, cpus)
    else:
        hw_cap = min(8, cpus)
    
    if requested is not None:
        return max(1, min(int(requested), hw_cap))
    return max(1, hw_cap)


# Shared CPU worker cap for batched rule-set simulations used by RB and tests.
# Exact rule-set/RB evaluation is Python-heavy and each process carries a
# prepared copy of the scoring arrays. Capping workers prevents SMT thread
# contention and RAM pressure.
BACKTEST_BATCH_WORKERS = resolve_backtest_workers()


# =============================================================================
# Phase 0 — Logging
# =============================================================================

# LOG_GENERATION_INTERVAL — Phase 2 progress log frequency (generations).
#   0   → auto (~10% of PHASE2_GENERATIONS).
#   N>0 → log every N generations; lower N = more verbose, slight I/O overhead.
LOG_GENERATION_INTERVAL = 0


# =============================================================================
# Phase 0 — Trend context (HWC/MWC/LWC causal enrichment)
# =============================================================================
# Deterministic, auditable four-state regime contract.  State codes are fixed
# and documented:
#   -1 = Bearish
#    0 = Range
#    1 = Bullish
#    2 = Noisy
# Higher-timeframe state is only published for completed bars.  15m timestamps
# are bar-open times; a completed bar's state first affects the next-bar entry.
CONTEXT_STATE_CODES: dict[str, int] = {
    "bearish": -1,
    "range": 0,
    "bullish": 1,
    "noisy": 2,
}
# Structural classifier formulas and version (part of strategy identity).
# Version 7 widens LWC pullback prints to treat Range as consolidation
# pullback, softens train-prefix classifier quantiles (0.55/0.55/0.45), and
# keeps the ungated historical pullback-permission policy from v6. Tradeable
# entries still require current-row permission AND trigger. Enriched tapes
# built under v6 are stale.
CONTEXT_ALGORITHM_VERSION: str = "regime_v7_range_pullback_coverage"
# CSV timestamps are 15m bar-open times.
CONTEXT_BAR_SECONDS: int = 15 * 60
# The frozen timeframe hierarchy: HWC = 4h, MWC = 1h, LWC = 15m.
HWC_TIMEFRAME_MINUTES: int = 240
MWC_TIMEFRAME_MINUTES: int = 60
LWC_TIMEFRAME_MINUTES: int = 15
# The pullback lookback over completed 15m LWC states (previous N bars).
# Frozen at 24 per PLAN.md / README.md wave-cycle contract (previous 24
# completed LWC states). Changing this value rewrites the regime trigger
# identity and must be accompanied by a contract version bump and full tape
# re-enrichment.
LWC_PULLBACK_LOOKBACK: int = 24
# Default train-only pooled percentile thresholds (frozen before any
# validation / test / forward results are reviewed).
# Softened so LWC prints more directional states under HTF permission.
CONTEXT_EFFICIENCY_TREND_THRESHOLD_QUANTILE: float = 0.55
CONTEXT_EMA_SPREAD_TREND_THRESHOLD_QUANTILE: float = 0.55
CONTEXT_VOLATILITY_COMPRESSION_QUANTILE: float = 0.45
# Common structural lookback (bars per cycle) used for the rolling indicators.
CONTEXT_STRUCTURAL_LOOKBACK: int = 20
# Minimum realised-volatility window; warm-up/unavailable context is Noisy.
CONTEXT_VOL_WINDOW: int = 20
# A neutral MWC state is a valid consolidation while HWC retains direction.
# Noisy MWC states remain excluded.  This policy is part of the enrichment
# contract and requires full tape re-enrichment when changed.
CONTEXT_ALLOW_MWC_RANGE_PERMISSION: bool = True
# When True, an opposite LWC print only counts as a pullback if same-direction
# HTF permission was already active on that historical bar. When False
# (default), any opposite LWC print in the lookback window counts; current-row
# permission still gates tradeable entries via the mandatory conditions.
# Changing this flag requires a contract version bump and full re-enrichment.
CONTEXT_REQUIRE_PERMISSION_ON_PULLBACK_PRINT: bool = False
# When True (default), LWC Range counts as a pullback print alongside the
# strict opposite state (Bearish for long / Bullish for short).
CONTEXT_PULLBACK_INCLUDE_RANGE: bool = True
# The mandatory direction-specific context + LWC trigger conditions.  These
# are fixed execution conditions, never ordinary NSGA genes.
CONTEXT_PERMISSION_COLUMNS: tuple[str, str] = (
    "tf_permission_long",
    "tf_permission_short",
)
CONTEXT_TRIGGER_COLUMNS: tuple[str, str] = (
    "lwc_pullback_reversal_long",
    "lwc_pullback_reversal_short",
)
# All context columns, kept out of ordinary Phase 1 / NSGA feature inference.
CONTEXT_COLUMNS: tuple[str, ...] = (
    "hwc_state",
    "mwc_state",
    "lwc_state",
    "tf_permission_long",
    "tf_permission_short",
    "lwc_pullback_reversal_long",
    "lwc_pullback_reversal_short",
)
# ---------------------------------------------------------------------------
# Hierarchical Multi-Timeframe (MTF) System Parameters
# ---------------------------------------------------------------------------
MTF_HWC_HORIZON_BARS: int = 6
MTF_MWC_HORIZON_BARS: int = 4
MTF_V_HWC_LONG: float = 0.65
MTF_V_HWC_SHORT: float = 0.60
MTF_V_MWC_LONG: float = 0.60
MTF_V_MWC_SHORT: float = 0.55
MTF_MIN_EVIDENCE_STRENGTH: float = 0.15
MTF_MIN_EVIDENCE_STRENGTH_HWC: float = 0.15
MTF_MIN_EVIDENCE_STRENGTH_MWC: float = 0.15
MTF_RETENTION_FLOOR: float = 0.50
MTF_RETENTION_TARGET: float = 0.60
MTF_PIPELINE_ENABLED: bool = True
MTF_N_FOLDS: int = 4
MTF_DISCOVERY_MAX_RULES_PER_LAYER: int = 8
MTF_MIN_FOLD_SUPPORT: int = 2

# Maximum allowed candle staleness for forward-filled HTF features on data gaps
MTF_MAX_STALENESS_CANDLES: int = 5

# Explicit feature scale manifest configuration (None = infer from train data)
FEATURE_SCALE_MANIFEST: dict[str, Any] | None = None

# Generic loader callers may explicitly work with raw fixture data.
REQUIRE_CONTEXT_COLUMNS: bool = False
# Legacy hardcoded context requirement is disabled for the new MTF architecture.
REQUIRE_CONTEXT_IN_STRATEGY: bool = False
# Mandatory, fixed conditions injected into every exported rule per direction.
# MIN_CONDITIONS / MAX_CONDITIONS count evolved confirmations, NOT these.
CONTEXT_MANDATORY_CONDITIONS: dict[str, tuple[str, ...]] = {    "long": (
        "[tf_permission_long] IS Active (1)",
        "[lwc_pullback_reversal_long] IS Active (1)",
    ),
    "short": (
        "[tf_permission_short] IS Active (1)",
        "[lwc_pullback_reversal_short] IS Active (1)",
    ),
}


def context_permission_column(direction: str) -> str:
    """Return the direction-specific permission column name."""
    direction = str(direction).strip().lower()
    if direction == "long":
        return CONTEXT_PERMISSION_COLUMNS[0]
    if direction == "short":
        return CONTEXT_PERMISSION_COLUMNS[1]
    raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")


def context_trigger_column(direction: str) -> str:
    """Return the direction-specific LWC pullback-reversal trigger column."""
    direction = str(direction).strip().lower()
    if direction == "long":
        return CONTEXT_TRIGGER_COLUMNS[0]
    if direction == "short":
        return CONTEXT_TRIGGER_COLUMNS[1]
    raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")


def mandatory_context_conditions(direction: str) -> tuple[str, ...]:
    """Return the fixed, mandatory context conditions for *direction*."""
    direction = str(direction).strip().lower()
    if direction not in CONTEXT_MANDATORY_CONDITIONS:
        raise ValueError(
            f"direction must be 'long' or 'short', got {direction!r}")
    return CONTEXT_MANDATORY_CONDITIONS[direction]


def context_contract() -> dict[str, object]:
    """Return the full context contract for strategy/dataset identity."""
    return {
        "algorithm_version": str(CONTEXT_ALGORITHM_VERSION),
        "state_codes": dict(CONTEXT_STATE_CODES),
        "bar_open_seconds": int(CONTEXT_BAR_SECONDS),
        "timeframes_minutes": {
            "hwc": int(HWC_TIMEFRAME_MINUTES),
            "mwc": int(MWC_TIMEFRAME_MINUTES),
            "lwc": int(LWC_TIMEFRAME_MINUTES),
        },
        "efficiency_trend_threshold_quantile": float(
            CONTEXT_EFFICIENCY_TREND_THRESHOLD_QUANTILE),
        "ema_spread_trend_threshold_quantile": float(
            CONTEXT_EMA_SPREAD_TREND_THRESHOLD_QUANTILE),
        "volatility_compression_quantile": float(
            CONTEXT_VOLATILITY_COMPRESSION_QUANTILE),
        "structural_lookback": int(CONTEXT_STRUCTURAL_LOOKBACK),
        "lwc_pullback_lookback": int(LWC_PULLBACK_LOOKBACK),
        "permission_policy": {
            "long": (
                "hwc_bullish AND mwc_bullish_or_range"
                if CONTEXT_ALLOW_MWC_RANGE_PERMISSION
                else "hwc_bullish AND mwc_bullish"
            ),
            "short": (
                "hwc_bearish AND mwc_bearish_or_range"
                if CONTEXT_ALLOW_MWC_RANGE_PERMISSION
                else "hwc_bearish AND mwc_bearish"
            ),
            "mwc_range_allowed": bool(CONTEXT_ALLOW_MWC_RANGE_PERMISSION),
        },
        "trigger_policy": {
            "require_permission_on_pullback_print": bool(
                CONTEXT_REQUIRE_PERMISSION_ON_PULLBACK_PRINT
            ),
            "pullback_include_range": bool(CONTEXT_PULLBACK_INCLUDE_RANGE),
        },
        "horizon_bars_15m": int(MAX_HOLD_CANDLES),
        "context_columns": list(CONTEXT_COLUMNS),
    }


def context_contract_digest() -> str:
    """Return a stable hash of the static contract and fitted enrichment."""
    import hashlib

    manifest_path = Path(ENRICHED_MANIFEST_PATH)
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Invalid trend-context manifest {manifest_path}: {exc}"
            ) from exc
        enrichment_identity: object = manifest
    elif REQUIRE_CONTEXT_COLUMNS:
        raise FileNotFoundError(
            f"Required trend-context manifest not found: {manifest_path}"
        )
    else:
        enrichment_identity = {"status": "missing"}

    payload = json.dumps(
        {
            "contract": context_contract(),
            "enrichment_manifest": enrichment_identity,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# =============================================================================
# Phase 1 — Feature selection (train_new.csv only)
# =============================================================================

# --- Ranking & shortlist ---

# PHASE1_DISPERSION_THRESHOLD — drop features whose mode value exceeds this freq.
#   Higher (→1.0) → keep near-constant columns; noisier Phase 2 gene space.
#   Lower (→0.5)  → aggressive pruning; may drop weak but real signals.
PHASE1_DISPERSION_THRESHOLD = 0.95

# PHASE1_TOP_K_FEATURES — shortlist size per direction (long / short).
#   Higher → wider Phase 2 search space, slower evolution, more combinations.
#   Lower  → faster search, risk of missing predictive features.
PHASE1_TOP_K_FEATURES = 20

# PHASE1_DISABLED — bypass MI ranking, sign-consistency, stationarity, and top-K
# selection. When True, Feature_Selector.run returns ALL features that pass the
# dispersion filter (PHASE1_DISPERSION_THRESHOLD) for both directions, with modes
# detected by Feature_Detector. Phase 2 then evolves over the full feature set.
#   True  → larger GA search space, more GPU RAM per chromosome, no MI prefilter.
#   False → normal top-K MI-ranked selection (PHASE1_TOP_K_FEATURES=20).
# Current default: bypass Phase 1 (full dispersion-filtered feature set) so the
# configured Phase 2 budget explores the broader genome with context gates.
PHASE1_DISABLED: bool = True

# PHASE1_MAX_FEATURE_OVERLAP — max shared feature names between long & short lists.
#   Enforced as int(TOP_K × overlap) shared names (e.g. 25 × 0.8 → 20 shared).
#   Higher → more shared features across directions; smaller combined gene space.
#   Lower  → more direction-specific lists; better asymmetry, less redundancy.
PHASE1_MAX_FEATURE_OVERLAP = 0.8

# PHASE1_ASYMMETRIC_TARGET — separate MI targets for long vs short.
#   True  → direction-specific feature rankings (recommended).
#   False → shared target; long/short pools share more structure.
PHASE1_ASYMMETRIC_TARGET = True

# PHASE1_USE_EXACT_BARRIER — use exact first-touch barrier outcomes for Phase 1 target.
#   True  → check exact barrier columns (_barrier_{direction}_tp_{tp}_{sl}_return_pct)
#           matching PHASE2_TP / PHASE2_SL to avoid max_before_min both-hit mislabeling.
#   False → fallback to legacy label_max_before_min heuristic.
PHASE1_USE_EXACT_BARRIER: bool = True
# --- Sign consistency across stationarity folds ---

# PHASE1_REQUIRE_SIGN_CONSISTENCY — drop features whose Spearman sign flips.
#   True  → fewer unstable features; stricter shortlist.
#   False → keep flip-flopping features; more noise in Phase 2.
PHASE1_REQUIRE_SIGN_CONSISTENCY: bool = True

# PHASE1_SIGN_CONSISTENCY_MIN_FOLDS — folds that must agree on correlation sign.
#   Higher → stricter; features must be stable across more sub-periods.
#   Lower  → more features pass; must be ≤ PHASE1_STATIONARITY_FOLDS.
PHASE1_SIGN_CONSISTENCY_MIN_FOLDS: int = 4

# PHASE1_SIGN_CONSISTENCY_MIN_ABS_CORR — ignore sign flips below this |ρ|.
#   Higher → only strong correlations must be consistent; more features kept.
#   Lower  → even weak correlations must be stable; stricter pruning.
PHASE1_SIGN_CONSISTENCY_MIN_ABS_CORR: float = 0.03

# --- Stationarity (reduce time-varying features) ---

# PHASE1_STATIONARITY_FOLDS — chronological chunks for stability tests.
#   Higher → more robust stationarity check, fewer features pass.
#   Lower  → faster, looser stationarity filter.
PHASE1_STATIONARITY_FOLDS = 5

# PHASE1_STATIONARITY_CV_MAX — max coefficient-of-variation across fold MI ranks.
#   Higher → allow more rank instability; keep more features.
#   Lower  → drop features whose importance swings across folds.
PHASE1_STATIONARITY_CV_MAX = 1.0

# PHASE1_STATIONARITY_RANK_DRIFT_MAX — max allowed rank change between folds.
#   Higher → tolerate large rank jumps; more features survive.
#   Lower  → only consistently top-ranked features kept.
PHASE1_STATIONARITY_RANK_DRIFT_MAX = 8

# =============================================================================
# Phase 1 → Phase 2 bridge — GPU row budget & JAX performance
# =============================================================================

# PHASE1_SAMPLING_TOTAL — max rows subsampled for Phase 2 GPU backtests.
# Peak GPU RAM scales ~linearly with this value (largest VRAM lever).
#   Higher → more statistical power, slower, OOM risk on small GPUs.
#   Lower  → faster, less RAM; trade/support floors may need proportional cut.
PHASE1_SAMPLING_TOTAL = 701_000

# PHASE2_PER_EPOCH_WINDOW_ROTATION — rotate train-window start per epoch
#   True  → each epoch samples a different contiguous sub-window from the
#           training data, using a deterministic per-epoch seed derived
#           from (sample_seed, epoch_idx). The per-sym request is capped
#           to fit within the largest safe range so the RNG start bar
#           branch in _sample_df fires.
#   False → sample once at generator init (A/B comparison / regression guard).
PHASE2_PER_EPOCH_WINDOW_ROTATION = True

# PHASE2_PER_EPOCH_WINDOW_SEED_MODE — how to derive the per-epoch seed.
#   "hash_epoch" → hash(sample_seed, epoch_idx) via SHA-256,
#                  deterministic, no RNG state leak.
PHASE2_PER_EPOCH_WINDOW_SEED_MODE = "hash_epoch"

# PHASE2_SAMPLE_MAX_BARS_PER_SYMBOL — hard cap on per-symbol bars in Phase 2
# sampling.  Keeps windows rotatable (n_per_sym < safe_len) on long histories.
#   Higher → more statistical power per epoch, less rotation headroom.
#   Lower  → more epoch-to-epoch diversity, faster backtests.
PHASE2_SAMPLE_MAX_BARS_PER_SYMBOL = 60_000

# PHASE2_SAMPLE_ROTATION_FRACTION — when rotation is enabled, request at most
# this fraction of the largest safe contiguous range per symbol (before the
# hard cap above).  Must be < 1.0 so start-bar RNG has room to vary.
PHASE2_SAMPLE_ROTATION_FRACTION = 0.65

# PHASE2_GPU_BATCH_SIZE — chromosomes per JAX vmap chunk in simulate_rule_batch.
# Peak VRAM scales ~linearly (rule matching is O(batch × rows × conditions)).
# Used directly when PHASE2_GPU_BATCH_SIZE_AUTO is False; otherwise VRAM/RAM-capped.
#   Higher → faster throughput until OOM; 64–128 is fine on Colab T4 with headroom.
#   Lower  → safer on small GPUs / 12 GiB RAM hosts, more kernel launches, slower.
PHASE2_GPU_BATCH_SIZE = 256

# PHASE2_GPU_BATCH_SIZE_AUTO — cap batch size by detected GPU VRAM and host RAM.
#   True  → apply tiers in _gpu_runtime (standard ~12 GiB Colab RAM → 64;
#           T4 ≤16 GiB VRAM → 256 before the host-RAM cap).
#   False → use PHASE2_GPU_BATCH_SIZE exactly (env PHASE2_GPU_BATCH_SIZE still wins).
PHASE2_GPU_BATCH_SIZE_AUTO = True

# PHASE2_SCAN_UNROLL — lax.scan unroll for equity simulation.
#   Higher → fewer kernel launches, longer XLA compile, slightly more VRAM.
#   Lower  → more launches, shorter compile; use 8 on memory-constrained GPUs.
PHASE2_SCAN_UNROLL = 32

# PHASE2_EVAL_BATCH_DEDUP — deduplicate identical chromosomes per batch call.
#   True  → skip redundant backtests; faster, no metric change.
#   False → evaluate every individual; use for debugging parity.
PHASE2_EVAL_BATCH_DEDUP = True

# PHASE2_EVAL_GLOBAL_CACHE — run-wide cache chromosome_key → metrics.
#   True  → large speedup when elites recur; safe for production.
#   False → always re-simulate; use when debugging cache staleness.
PHASE2_EVAL_GLOBAL_CACHE = True

# PHASE2_EVAL_GLOBAL_CACHE_MAX_SIZE — hard cap on the global eval cache.
#   Adaptive default based on host RAM: 600 if ram <= 13 GiB, else 900.
#   Higher → more cache hits; less RAM.
#   Lower  → less RAM; more re-evaluations.
# Evaluated adaptively to limit Colab / 2-core RAM footprint while maximizing hits.
def _resolve_global_cache_max_size() -> int:
    try:
        from gpu_fuzzy_trader._gpu_runtime import detect_system_ram_gb
        ram = detect_system_ram_gb()
        if ram is not None and ram > 13.0:
            return 900
    except Exception:
        pass
    return 600

PHASE2_EVAL_GLOBAL_CACHE_MAX_SIZE = _resolve_global_cache_max_size()

# PHASE2_SKIP_ZERO_SIGNAL_SCAN — skip equity scan when rule matches 0 bars.
#   True  → faster; infeasible rules get penalty without full scan.
#   False → full scan always; slightly slower, easier to debug.
PHASE2_SKIP_ZERO_SIGNAL_SCAN = True

# PHASE2_SKIP_INFEASIBLE_SIGNAL_SCAN — skip scan when raw matches < trade floor.
#   True  → faster evolution; rules below floor never get full equity path.
#   False → full metrics even for doomed chromosomes.
PHASE2_SKIP_INFEASIBLE_SIGNAL_SCAN = True

# PHASE2_GPU_USE_FP32 — float32 on GPU for Phase 2 ranking (CPU stays reference).
#   True  → ~2× faster on T4; tiny numeric drift vs float64.
#   False → slower, exact parity with CPU dtypes.
PHASE2_GPU_USE_FP32 = True

# PHASE2_GPU_DATA_INT8 — store discretized feature matrix as int8 on GPU.
#   True  → lower VRAM for feature tensor; classes fit in 0..10.
#   False → wider dtypes; slightly more VRAM.
PHASE2_GPU_DATA_INT8 = True

# PHASE2_GPU_CPU_ROUTE_LARGE_DATA — route Phase 2 ranking batches to the
# vectorized CPU engine when the time-series window is large.  On an 8-core
# CPU + 6-GiB RTX 4050, host-side sparse/event simulation is faster than the
# GPU's full-row scan for the default ~90k-bar train window.
#   True  → lower wall time on the RTX 4050 target; exact CPU ranking metrics.
#   False → force the JAX batch path (useful for A/B benchmarks or a faster GPU).
PHASE2_GPU_CPU_ROUTE_LARGE_DATA = True

# Minimum bars and maximum population size for the CPU route.  Small windows
# and unusually large batches remain on GPU because launch amortization can
# reverse the crossover on stronger hardware.
PHASE2_GPU_CPU_ROUTE_MIN_BARS = 20_000
PHASE2_GPU_CPU_ROUTE_MAX_BATCH = 256

# PHASE2_CPU_BATCH_SIZE — chromosomes per vectorized CPU signal chunk.
# The CPU evaluator's sparse gather has a temporary shape of
# ``batch × bars × active_slots``.  A conservative chunk keeps the entire
# Phase 2 run below the 7.2-GiB WSL dynamic-memory ceiling while retaining
# vectorized matching.  It changes neither chromosome fitness nor ranking.
PHASE2_CPU_BATCH_SIZE = 16

# PHASE2_GPU_EVENT_DRIVEN — use an event-only equity scan for sparse-slot
# chromosomes.  Sparse rules usually match a tiny fraction of bars; scanning
# every bar with lax.scan wastes GPU time while the exact CPU path visits only
# signal/release events.  The event kernel keeps the same accounting model but
# scans a fixed, padded event buffer so XLA does not recompile every rule.
PHASE2_GPU_EVENT_DRIVEN = True

# PHASE2_GPU_EVENT_MAX_EVENTS — maximum unique signal/release rows in one
# chromosome before the regular full-row GPU scan is used.  This is a shape
# and memory guard, not a quality threshold.  4096 is ample for the default
# 4–5-condition sparse genome on a 6-GiB RTX 4050.
PHASE2_GPU_EVENT_MAX_EVENTS = 4096


# =============================================================================
# Phase 2 — Fixed risk during rule search (RB tunes TP/SL/capital later)
# =============================================================================

# PHASE2_TP — take-profit % used when scoring rules in Phase 2 (and Phase 1 targets).
#   Higher → fewer "wins" in labels/objectives; rules must catch larger moves.
#   Lower  → more wins, higher turnover, may favor noisy frequent signals.
PHASE2_TP = 2.0

# PHASE2_SL — stop-loss % during Phase 2 scoring.
#   Higher → wider stops, fewer stop-outs, larger per-trade risk.
#   Lower  → tighter stops; forces higher precision rules to survive.
PHASE2_SL = 1.2

# PHASE2_CAPITAL_PCT — % of equity allocated per rule signal in Phase 2.
#   Higher → larger simulated positions; drawdown and return scale up.
#   Lower  → conservative sizing; may understate overlap effects until RB risk tuning.
PHASE2_CAPITAL_PCT = 18.0


# =============================================================================
# Phase 2 — Rule genome
# =============================================================================

# MIN_CONDITIONS / MAX_CONDITIONS — active fuzzy conditions per rule.
#   Higher MIN → stricter rules, fewer matching bars, higher precision target.
#   Lower MIN → broader rules, more trades, risk of weak patterns.
#   Higher MAX → allow complex rules (if encoding supports variable count).
#   Lower MAX → force simplicity; more generalization, less specificity.
MIN_CONDITIONS = 2
MAX_CONDITIONS = 4

# PHASE2_ENCODING — chromosome memory layout during evolution.
#   "dense"        — length-K vector with per-feature dont_care.
#   "sparse_slots" — fixed slots (MAX_CONDITIONS, 2); dynamic active count.
# Pool JSON / archives remain dense K-vectors for RB compatibility.
PHASE2_ENCODING = "sparse_slots"
# Preserve active feature building blocks during sparse crossover.  Slot-wise
# crossover can exchange unrelated features merely because they share a row.
PHASE2_FEATURE_SET_CROSSOVER: bool = True


# =============================================================================
# Phase 2 — Trade support & pool admission
# =============================================================================

# MIN_TRADE_SUPPORT — target executed trades before support penalty vanishes.
#   Higher → penalize low-frequency rules harder; pool favors robust sample size.
#   Lower  → allow rare-pattern rules; noisier Sortino/return estimates.
# Global BTC/ETH search uses the pooled-universe target (not the old
# singleton-island 60-trade relaxation).
MIN_TRADE_SUPPORT = 120

# SUPPORT_PENALTY_MAX — cap on quadratic support shortfall penalty.
#   Higher → stronger push away from under-supported rules on all objectives.
#   Lower  → evolution tolerates thin trade counts longer.
SUPPORT_PENALTY_MAX = 5.0

# TRADE_SUPPORT_PENALTY_EXPONENT — exponent for trade-support penalty between
# pool_floor and min_support. Higher = steeper, harsher penalty for low-trade-
# count rules. Default 3.0 (was implicitly 2.0 before this parameter existed).
TRADE_SUPPORT_PENALTY_EXPONENT = 3.0

# MIN_TRADE_POOL_FLOOR — hard reject below this executed trade count.
#   Higher → archive/pool never keeps very rare rules.
#   Lower  → extremely sparse rules can survive if other metrics excel.
MIN_TRADE_POOL_FLOOR = 25

# PHASE2_SUPPORT_PENALTY_WEIGHT_F1/F2/F3 — per-objective support penalty scale.
#   Higher → that objective punishes low support more (steer Sortino vs DD vs return).
#   Lower  → support matters less for that objective.
# F1 0.1→0.45 — zero-trade rules were barely punished on Sortino and
# cluttered the Pareto front (median_return=0.00% in run.log).
# F1 0.45→0.25 — do not obsess over fat singles; keep some pressure
# so zero-trade junk still loses on Sortino.
PHASE2_SUPPORT_PENALTY_WEIGHT_F1 = 0.25  # Sortino objective
PHASE2_SUPPORT_PENALTY_WEIGHT_F2 = 0.6  # drawdown objective
PHASE2_SUPPORT_PENALTY_WEIGHT_F3 = 0.6  # return / win-rate objective

# PHASE2_USE_TOTAL_RETURN_OBJ — f3 uses robust return (min train, val) instead
# of the configured profit-factor or win-rate objective.
#   True  → f3 = -robust_return_pct (min of train/val return); aligns with OOS PnL.
#   False → f3 uses PHASE2_F3_OBJECTIVE (profit_factor or win_rate).
# With PHASE2_JOINT_TRAIN_VAL=False, "robust" return collapses to train-only and
# f1 (Sortino) ≈ f3 (return) → objective_corr_f1_f3≈1.0 (Pareto collapse in run.log).
# Keep this False: the configured profit_factor f3 is materially different from
# Sortino and preserves a useful third exploration axis.
PHASE2_USE_TOTAL_RETURN_OBJ = False

# --- Task 2: Return-concentration 4th NSGA objective -------------------------
# PHASE2_F4_ENABLED — adds f4 = max_single_trade_pnl / max(sum_positive_trade_pnl, ε)
#   as a 4th NSGA-III objective to penalise rules whose edge comes from a single
#   outlier trade. True = 4-objective; False = 3-objective (regression guard).
# → fixes audit finding #2 (outlier-driven f3 from uncapped time-exit returns)
PHASE2_F4_ENABLED = True
# PHASE2_F4_CONCENTRATION_FLOOR — rules with f4 > this floor are rejected at pool
#   admission via the f4_concentration gate in _feasibility_gate_failures.
PHASE2_F4_CONCENTRATION_FLOOR = 0.35
# PHASE2_F4_EPSILON — small constant to avoid division by zero in the f4 ratio.
PHASE2_F4_EPSILON = 1e-6
# PHASE2_N_OBJECTIVES — number of NSGA-III objectives (3 or 4). Used to size
#   objective arrays and reference vector calls in evox_runner.py.
PHASE2_N_OBJECTIVES = 4

# Lower-confidence edge estimates keep sparse PF/Sortino winners from
# dominating the archive. Exact per-trade dispersion is used when available;
# aggregate metrics use a conservative fallback.
PHASE2_EXPECTANCY_LCB_Z = 1.645
PHASE2_EXPECTANCY_LCB_WEIGHT = 8.0
PHASE2_RANK_USE_LCB_EXPECTANCY: bool = True
PHASE2_EXPECTED_SHORTFALL_Q: float = 0.10
PHASE2_EXPECTED_SHORTFALL_WEIGHT: float = 1.5
PHASE2_BEHAVIORAL_ARCHIVE_ENABLED: bool = True

# PHASE2_F3_OBJECTIVE — third objective: "profit_factor" (default when
# PHASE2_USE_TOTAL_RETURN_OBJ=False),
# "cv_fold_min" (min of CV fold returns), or "win_rate" (legacy).
#   profit_factor → f3 = -profit_factor (aligns with edge quality over noise).
#   cv_fold_min  → f3 = -min(CV fold returns); requires CvFoldValEvaluator
#                  which is too expensive for NSGA-III inner loop — disabled.
#   win_rate     → f3 = -win_rate (degenerate, not recommended).
# With the default PHASE2_USE_TOTAL_RETURN_OBJ=False, f3 uses this setting.
# If total-return mode is explicitly enabled, robust_return_pct takes precedence.
# CV-fold robustness is enforced at the pool-admission gate and RB Governor
# scoring stages instead.
PHASE2_F3_OBJECTIVE = "profit_factor"

# PHASE2_MIN_PROFITABLE_SYMBOLS_PENALTY — min profitable symbols before penalty.
#   Soft evolution nudge: adds to support_penalty when
#   n_profitable_symbols < this during fitness. The target is capped by the
#   measured symbol universe.
PHASE2_MIN_PROFITABLE_SYMBOLS_PENALTY = 1

# PHASE2_SYMBOL_GENE_DONT_CARE_PROB — probability of forcing a symbol gene to
#   dont_care during mutation. Higher → more cross-symbol rules; prevents
#   symbol-locked evolution.
PHASE2_SYMBOL_GENE_DONT_CARE_PROB = 0.75

# PHASE2_USE_ROBUST_RETURN_OBJ — store min(train_return, val_return) as
#   robust_return_pct on metrics when PHASE2_JOINT_TRAIN_VAL=True.
#   When PHASE2_USE_TOTAL_RETURN_OBJ=True, this controls whether the joint
#   return uses min(train, val) or just train-only.
#   True  → robust_return_pct = min(train_return, val_return).
#   False → robust_return_pct = train_return (equivalent to no robustness).
PHASE2_USE_ROBUST_RETURN_OBJ = True

# PHASE2_SORTINO_MIN_TRADE_THRESHOLD — trade count below which Sortino is scaled down.
#   Used in Approach 2 to penalize low-trade-count rules.
PHASE2_SORTINO_MIN_TRADE_THRESHOLD = 20

# --- Return / quality floors (evolution + pool filtering) ---

# PHASE2_RETURN_FLOOR_PCT — min train return % to avoid feasibility penalty.
#   Higher → only profitable-on-train rules stay feasible; emptier search.
#   Lower  → more exploration; weak rules linger until other gates remove them.
# 0.5→0.25 — ease early feasibility collapse (train_return_floor ~90%).
PHASE2_RETURN_FLOOR_PCT = 0.25

# PHASE2_VAL_RETURN_FLOOR_PCT — min validation return % for feasibility.
#   Higher → stricter OOS alignment during evolution.
#   Lower  → allow negative val return during search (gates may still catch later).
# 1.0→0.25 — val_return_floor was failing ~95–98% of the pop early.
PHASE2_VAL_RETURN_FLOOR_PCT = 0.25

# Aligned with long. Was 2.0 then 1.0; still starved short islands early.
PHASE2_VAL_RETURN_FLOOR_PCT_SHORT = 0.25

# PHASE2_PROFIT_FACTOR_FLOOR_EVOLUTION — soft penalty threshold during NSGA-III fitness.
#   Lower than the admission floor so the feasible set isn't artificially collapsed
#   when val trade counts are thin (random rules rarely have val PF > 1.15).
#   Higher → fewer rules pass the soft penalty during evolution.
#   Lower  → more rules explore; admission gate (1.15) is the real filter.
#   → fixes audit finding #9 (feasibility collapse is val-driven, not objective-design)
PHASE2_PROFIT_FACTOR_FLOOR_EVOLUTION = 1.0

# PHASE2_PROFIT_FACTOR_FLOOR_ADMISSION — hard gate at pool admission.
#   This is intentionally stricter than the evolution floor.
#   → fixes audit finding #9
PHASE2_PROFIT_FACTOR_FLOOR_ADMISSION = 1.15

# PHASE2_SYMBOL_MEDIAN_RETURN_FLOOR_PCT — min median return across symbols.
#   Higher → rules must work on typical symbols, not one outlier.
#   Lower  → single-symbol heroes can survive longer.
PHASE2_SYMBOL_MEDIAN_RETURN_FLOOR_PCT = 0.0

# PHASE2_MIN_PROFITABLE_SYMBOLS — target count of symbols with positive PnL for
# the cross-symbol robustness penalty. The active two-symbol train_new.csv
# universe makes 2 the strongest achievable generalist target. Runtime
# validation rejects this value when a full (non-debug) universe is smaller;
# debug scopes cap the effective target to their scoped universe.
PHASE2_MIN_PROFITABLE_SYMBOLS = 2

# PHASE2_MAX_DRAWDOWN_GATE — soft DD % cap; excess DD adds penalty on f2 only.
#   Lower  → Pareto front pushed toward low-drawdown rules; may cut high return.
#   Higher → allow aggressive rules with large equity swings.
PHASE2_MAX_DRAWDOWN_GATE = 20.0

# PHASE2_POOL_REQUIRE_POSITIVE_SPLITS — require non-negative train & val returns.
#   True  → infeasible penalty on negative-split rules during evolution.
#   False → negative val allowed at fitness stage (CV gates may still filter).
PHASE2_POOL_REQUIRE_POSITIVE_SPLITS = True
PHASE2_POOL_TRAIN_RETURN_MIN_PCT = 0.0
PHASE2_POOL_VAL_RETURN_MIN_PCT = 0.0

# PHASE2_MAX_TRAIN_VAL_GAP_PCT — reject pool admission when train return exceeds
# val return by more than this threshold (classic overfit signal).
#   Higher → more lenient; only extreme train>>val gaps rejected.
#   Lower  → stricter alignment between train and val required.
PHASE2_MAX_TRAIN_VAL_GAP_PCT = 10.0

# PHASE2_OVERFIT_WARNING_RATIO — threshold ratio (max_return / max_robust_return)
# above which a WARNING is logged during evolution. Signals a rule that performs
# well on train but poorly on validation — likely overfit to the training window.
#   Higher → only extreme overfit gaps trigger a warning.
#   Lower  → more sensitive; smaller gaps also trigger warnings.
PHASE2_OVERFIT_WARNING_RATIO = 3.0

# PHASE2_OVERFIT_GAP_PENALTY_WEIGHT — fitness penalty when train_return - val_return
#   exceeds PHASE2_OVERFIT_GAP_PCT_THRESHOLD (applied to f1 and f3).
#   Default 15.0 (raised from 5.0 in task-6) so the soft penalty is comparable to the
#   return signal — kills rules where train>>val even before pool admission.
# → fixes audit finding #7
PHASE2_OVERFIT_GAP_PENALTY_WEIGHT = 15.0

# PHASE2_OVERFIT_GAP_PCT_THRESHOLD — train/val return gap (in percentage points)
#   above which the overfit-gap penalty starts accumulating. Subtraction-based,
#   well-defined for any sign of val_ret (unlike the old ratio-based definition).
#   Default 8.0pp is below PHASE2_MAX_TRAIN_VAL_GAP_PCT=10.0 so the soft penalty
#   starts ramping before the hard gate rejects outright.
PHASE2_OVERFIT_GAP_PCT_THRESHOLD = 8.0

# PHASE2_OVERFIT_RATIO_FLOOR — hard reject pool admission when train_return /
#   max(val_return, 0.1) exceeds this ratio. Catches cases where the absolute-pp
#   gate (PHASE2_MAX_TRAIN_VAL_GAP_PCT) misses high-ratio / low-absolute-gap
#   situations, e.g., train=15% / val=4% (gap=11pp < 16pp, ratio=3.75×).
#   Higher → more lenient; only extreme ratio mismatches rejected.
#   Lower  → stricter; smaller ratios also trigger.
#   0.0 or float('inf') → disables the ratio gate (regression guard).
# Slightly eased 2.5→3.0: island windows inflate train/val ratio noise.
PHASE2_OVERFIT_RATIO_FLOOR = 3.0

# PHASE2_OBJECTIVE_CORR_WARN_THRESHOLD — log a debug warning when Pareto-front
#   objective pairwise correlation exceeds this (Pareto collapse risk).
PHASE2_OBJECTIVE_CORR_WARN_THRESHOLD = 0.9

# PHASE2_OBJECTIVE_CORR_MIN_PARETO_SIZE — minimum Pareto front size before
#   the "Pareto collapse risk" warning fires. 2-point Pearson correlations
#   are degenerate (trivially ±1.0 by construction); the warning is only
#   meaningful when the front has enough rules.
# → fixes audit finding #13 (noisy 2-point correlation warning)
PHASE2_OBJECTIVE_CORR_MIN_PARETO_SIZE = 5

# PHASE2_KEEP_TOP_RULES — max rules kept in the final Phase 2 pool after
# admission filtering, sorted by deployability_rank_score descending.
#   Higher → larger pool for RB candidate selection.
#   Lower  → smaller pool; faster RB selection, fewer combinations.
# 80→150 — one-symbol islands → many moderate specialists.
PHASE2_KEEP_TOP_RULES = 150

# PHASE2_MAX_RESERVED_RULES_PER_SYMBOL — maximum number of admitted pool
# candidates reserved for each symbol before the global deployability ranking
# fills the remaining slots.  Reservations are evidence-based: a candidate
# must have positive validation PnL and the RB validation trade floor for that
# symbol, so this cannot keep a failing rule just to satisfy a quota.
PHASE2_MAX_RESERVED_RULES_PER_SYMBOL = 10

# PHASE2_REQUIRE_LAST_FOLD_POSITIVE — in the holdout pool-admission path,
# require the (single) validation fold to have positive total return.
#   True  → pool admission rejects rules with val_return <= 0%.
#   False → val_return can be non-positive (other gates still apply).
# (Originally named for the multi-fold CV era; the check itself runs in the
# holdout path where cv_fold=False, so it remains meaningful without CV.)
PHASE2_REQUIRE_LAST_FOLD_POSITIVE: bool = False

# =============================================================================
# Phase 2 — Monthly-window shadow test for pool admission (Task 13)
# =============================================================================
# These flags add a hard pool-admission gate after Phase 2 evolution: each
# candidate rule must be profitable on at least 50% of monthly rolling windows
# in the train split.  This addresses the regime-shift problem identified in
# Task 12's diagnostic CSV: per-symbol rules that pass validation can bleed on
# test because they are not stable across time.  The gate is additive — when
# PHASE2_MONTHLY_ADMISSION_ENABLED is False, the existing pool path is
# unchanged.

# PHASE2_MONTHLY_ADMISSION_ENABLED — toggle the monthly-window gate.
#   True  → rules must pass the monthly profitable-ratio filter to enter the pool.
#   False → skip the gate (zero behaviour change vs. pre-Task-13 code).
PHASE2_MONTHLY_ADMISSION_ENABLED = True

# PHASE2_MONTHLY_ADMISSION_FAIL_CLOSED — behavior when every candidate fails
# the monthly validation gate.
#   True  → return an empty pool and let downstream deployment fail closed.
#   False → retain the legacy compatibility fallback (not recommended).
PHASE2_MONTHLY_ADMISSION_FAIL_CLOSED = True

# PHASE2_MONTHLY_GOOD_RETURN_MIN_PCT — minimum total_return_pct (%) for a monthly
# window to count as "good" in the pool-admission gate.
#   0.0  → non-loss months count (flat range months OK — desired equity shape).
#   2.0  → month must earn at least +2% to count as good.
#   -1.0 → month counts if return >= -1% (more lenient non-loss bar).
PHASE2_MONTHLY_GOOD_RETURN_MIN_PCT = 0.0

# PHASE2_MONTHLY_ADMISSION_MIN_RATIO — fraction of monthly windows that must
# be profitable for a rule to be admitted.
#   0.500 → minimum feasible evidence on the short validation calendar while
#           still requiring half the windows to be non-loss.
PHASE2_MONTHLY_ADMISSION_MIN_RATIO = 0.50

# PHASE2_MONTHLY_ADMISSION_MIN_MONTHS — minimum number of monthly windows
# required before the gate is applied. validation_fitness is ~110 calendar days
# (~3×30d windows), so 4 was structurally incompatible and silently skipped.
# When fewer windows exist but at least one is available, the gate still runs
# (degraded) instead of being skipped.
# 3→2 — Colab holdout val (~2 months) always logged
# "only 2 monthly windows (< MIN_MONTHS=3); degraded mode". Align floor with
# actual window count so the gate is non-degraded when 2 windows exist.
#   Higher → skip/degrade more often on short data.
#   Lower  → require monthly evidence even on short trains.
PHASE2_MONTHLY_ADMISSION_MIN_MONTHS = 2

# A flat month is only neutral when the rule actually traded.  These controls
# prevent zero-signal rules from passing the monthly gate by accumulating
# artificial 0% months.
PHASE2_MONTHLY_MIN_TRADES = 3
PHASE2_MONTHLY_MIN_ACTIVE_RATIO = 0.60
PHASE2_MONTHLY_MAX_BEARISH_RATIO = 0.50


# =============================================================================
# Phase 2 — Fitness objectives & joint evaluation
# =============================================================================

# SORTINO_CAP — maximum saturated Sortino after tanh compression.
#   Higher → more differentiation among top Sortino rules on f1.
#   Lower  → flatter f1 landscape; diversity across other objectives easier.
SORTINO_CAP = 20.0

# SORTINO_SCALE — divisor inside tanh(raw_sortino / scale); controls saturation.
#   Higher → less compression; extreme Sortino values still differentiate f1.
#   Lower  → aggressive compression; reduces Sortino-driven dominance.
SORTINO_SCALE = 10.0

# PHASE2_JOINT_TRAIN_VAL — fitness uses min(train, val) Sortino/return where applicable.
#   True  → slower (eval val every gen) but aligned with deployment; less overfit.
#   False → train-only fitness; faster; holdout remains clean for model selection
#           (RB Governor) & OOS (Phase 5). Robustness via the active CV evaluator.
# Changed True→False (task-4): PHASE2_JOINT_TRAIN_VAL=True was double-counting
# the val split — both joint fitness and pool-admission gates (monthly windows,
# val return floors, overfit gaps) used the same val window, creating a leak.
# Train-only fitness + val-only gates is the anti-overfit design (RB Governor
# select-and-tune uses its own val split, not Phase 2's).
PHASE2_JOINT_TRAIN_VAL = False

# PHASE2_VAL_IN_FITNESS_PENALTY — when True, val-derived penalties
#   (val_floor_penalty, val symbol_robustness, val trade-floor support cap,
#   overfit_gap_penalty, and val terms inside _raw_feasibility_violation_score)
#   enter fitness even when PHASE2_JOINT_TRAIN_VAL=False.
#   Val metrics are always stored for reporting, deployability preview, and
#   pool admission. Joint stays False to avoid double-counting the same holdout
#   in joint Sortino/return *and* admission gates (anti-leak).
#   If the feasible set starves under these penalties, retune floors/weights —
#   do not flip JOINT_TRAIN_VAL=True without a separate holdout design.
#   → Phase 2 feasible-search item 4 / OOS plan 003
# Production keeps validation as an admission/archive screen rather than
# applying the same holdout as continuous evolutionary selection pressure.
PHASE2_VAL_IN_FITNESS_PENALTY = False

# PHASE2_VAL_SIM_INTERVAL — fallback validation cadence when validation is
# report-only. If PHASE2_JOINT_TRAIN_VAL or PHASE2_VAL_IN_FITNESS_PENALTY is
# enabled, the evolution loop evaluates validation every generation so newly
# evaluated chromosomes have stable objective semantics. Otherwise, with
# per-epoch window rotation, validation is deterministic and can be throttled.
# Val ALWAYS runs on the epoch's last gen (pool-admission freshness).
#   1 → val every gen (legacy, expensive).
#   2 → val every other gen (current fallback; ~50% maximum val-call savings).
# → fixes audit finding #10 (val every gen is wasteful when
# window is fixed; cache is safe)
PHASE2_VAL_SIM_INTERVAL = 2

# PHASE2_DIVERSITY_HAMMING_THRESHOLD — min Hamming distance for "unique" rule.
#   Higher → demand more genetic distance; wider Pareto spread, slower convergence.
#   Lower  → allow near-duplicate rules; risk of niche collapse.
#   0 = auto-scale via PHASE2_DIVERSITY_HAMMING_THRESHOLD_AUTO (max(3, k_active // 5)).
PHASE2_DIVERSITY_HAMMING_THRESHOLD = 0

# PHASE2_DIVERSITY_HAMMING_THRESHOLD_AUTO — auto-scale Hamming threshold.
#   True  → threshold = max(3, k_active // 5) computed at runtime.
#   False → use PHASE2_DIVERSITY_HAMMING_THRESHOLD value directly.
PHASE2_DIVERSITY_HAMMING_THRESHOLD_AUTO = True

# PHASE2_HOF_EPOCH_CARRYOVER — max hall-of-fame entries carried across epochs.
#   Higher → more prior-generation chromosomes kept for diversity reference.
#   Lower  → epoch boundary acts as harder reset of the diversity reference set.
PHASE2_HOF_EPOCH_CARRYOVER = 10

# PHASE2_DIVERSITY_ON_F4 — when True (default), diversity_penalty is applied to
#   f4 only (or f2 if f4 disabled). When False, legacy behavior applies diversity
#   to both f1 and f3 (causes objective_corr_f1_f3 collapse).
PHASE2_DIVERSITY_ON_F4: bool = True

# PHASE2_DIVERSITY_PENALTY — objective penalty when crowding near existing rules.
#   Higher → stronger push toward novel chromosomes.
#   Lower  → convergence to similar high performers allowed.
# Increased 0.5→2.0: 0.5 was negligible vs 50+ infeasible penalties, allowing
# phenotype collapse. 2.0 provides meaningful push toward diversity without
# overwhelming feasible objectives on the Pareto front.
PHASE2_DIVERSITY_PENALTY = 3.0

# PHASE2_PHENOTYPE_SORTINO_STEP — Sortino bucket width for behavioral diversity.
# Tightened 0.5→0.3→0.15: with compressed Sortino in ~0–20 and pop 200, finer
# buckets give broader Pareto spread.
PHASE2_PHENOTYPE_SORTINO_STEP = 0.15

# PHASE2_PHENOTYPE_DD_STEP — drawdown % bucket width for behavioral diversity.
# Tightened 5→4: DD typically lives in 5–25%; 5.0 gave ~4 buckets, 4.0 gives ~5.
PHASE2_PHENOTYPE_DD_STEP = 4.0

# PHASE2_PHENOTYPE_F3_STEP — f3-axis bucket width (win rate % or return %).
# Tightened 10→5→2: f3 (return) rarely exceeds ±15%; 5.0 gave 4–6 buckets;
# 2.0 gives ~15 buckets for finer behavioral differentiation.
PHASE2_PHENOTYPE_F3_STEP = 2.0

# PHASE2_EARLY_STOP_ENABLED — stop evolution on poor mean/median return trend.
#   True  → save generations when search is clearly failing.
#   False → always run full PHASE2_GENERATIONS budget.
# False — short 20-gen one-symbol runs must use the full budget.
PHASE2_EARLY_STOP_ENABLED = False

# PHASE2_EARLY_STOP_MIN_GENERATION — earliest gen for return-based early stop.
#   Higher → more exploration before stop can fire.
#   Lower  → may stop before diversity recovery mechanisms run.
PHASE2_EARLY_STOP_MIN_GENERATION = 40

# PHASE2_EARLY_STOP_MEAN_RETURN_PCT — stop if mean Pareto return below this %.
#   Higher (less negative) → stop sooner on mediocre populations.
#   Lower (more negative) → tolerate longer periods of poor mean return.
PHASE2_EARLY_STOP_MEAN_RETURN_PCT = -5.0

# PHASE2_EARLY_STOP_USE_MEDIAN_RETURN — use median instead of mean for stop.
#   True  → robust to one outlier bad rule on Pareto front.
#   False → sensitive to mean; one bad elite can prevent stop.
PHASE2_EARLY_STOP_USE_MEDIAN_RETURN = True

# PHASE2_EARLY_STOP_MIN_VALID_RULES — require at least this many valid Pareto rules.
#   Higher → don't early-stop while front is still sparse.
#   Lower  → stop even with tiny front.
PHASE2_EARLY_STOP_MIN_VALID_RULES = 3

# --- Plateau early stop (no improvement in best return) ---

# False — no early/plateau stop on short one-symbol budgets.
PHASE2_PLATEAU_EARLY_STOP_ENABLED = False

# PHASE2_PLATEAU_EARLY_STOP_MIN_GENERATION — earliest gen for plateau stop.
#   Higher → more exploration in Stage A before plateau can end run.
#   Lower  → may stop during initial transient; should be ≤ STAGE_A_GENERATIONS.
PHASE2_PLATEAU_EARLY_STOP_MIN_GENERATION = 6

# PHASE2_PLATEAU_EARLY_STOP_PATIENCE — gens without improvement before stop.
#   Higher → wait longer for breakthrough; uses more compute.
#   Lower  → stop quickly when progress stalls.
PHASE2_PLATEAU_EARLY_STOP_PATIENCE = 7

# PHASE2_PLATEAU_EARLY_STOP_MIN_DELTA_PCT — min return improvement to reset patience.
#   Higher → need larger gains to count as progress.
#   Lower  → tiny improvements reset plateau counter.
# Increased 0.5→1.0: require meaningful improvement to reset patience;
# 0.5% noise-level fluctuations kept resetting the plateau counter.
PHASE2_PLATEAU_EARLY_STOP_MIN_DELTA_PCT = 1.0

# PHASE2_PLATEAU_USE_ROBUST_RETURN — track min(train,val) return for plateau.
#   True  → plateau reflects deployable return, not train-only spikes.
#   False → train max return can mask val stagnation.
PHASE2_PLATEAU_USE_ROBUST_RETURN = True

# PHASE2_PLATEAU_BLOCK_WHEN_* — suppress plateau stop while front is unhealthy.
PHASE2_PLATEAU_BLOCK_WHEN_DEPLOYABLE_ZERO = True
PHASE2_PLATEAU_BLOCK_WHEN_DIVERSITY_LOW = False

# PHASE2_PLATEAU_DIVERSITY_RESTART_ENABLED — reinit part of the pop on plateau
#   instead of immediately breaking when restarts remain.
#   True  → up to PHASE2_PLATEAU_MAX_RESTARTS diversity restarts, then break.
#   False → immediate break on first plateau (original behaviour).
PHASE2_PLATEAU_DIVERSITY_RESTART_ENABLED = True

# PHASE2_PLATEAU_DIVERSITY_RESTART_FRACTION — share of pop reinitialised on restart.
#   Higher  → more fresh chromosomes; may lose more elite progress.
#   Lower   → gentler restart; may not escape attractor.
PHASE2_PLATEAU_DIVERSITY_RESTART_FRACTION = 0.35

# PHASE2_PLATEAU_POST_RESTART_MUTATION_BOOST — mutation rate (not multiplier) used
#   for PHASE2_PLATEAU_POST_RESTART_BOOST_GENS after a plateau restart, then
#   anneals back to the stage mutation rate.
#   Higher → more exploration in the gens immediately after restart.
#   Lower  → gentler post-restart mutation; less disruptive to surviving elites.
PHASE2_PLATEAU_POST_RESTART_MUTATION_BOOST = 0.45

# PHASE2_PLATEAU_POST_RESTART_BOOST_GENS — number of generations to hold the
#   post-restart mutation boost before decaying to the normal stage rate.
#   Higher → longer boosted exploration after restart.
#   Lower  → quicker return to normal mutation; less search diversity.
# Increased 3→4: more boost time before evaluation; pairs with patience=5
# so boosted window gets full chance before stop can fire.
PHASE2_PLATEAU_POST_RESTART_BOOST_GENS = 4

# PHASE2_PLATEAU_POST_RESTART_STOP_ENABLED — break the epoch when the best
#   metric fails to improve for PHASE2_PLATEAU_POST_RESTART_STOP_PATIENCE gens
#   AFTER a plateau restart.  A restart already signals a stall; if fresh blood
#   + boosted mutation yields no progress within the boost window, further gens
#   are very unlikely to help.  Cuts only provably-unproductive generations.
#   True  → stop after a failed restart (default; safe runtime win).
#   False → always run the full epoch budget after a restart (original behaviour).
# False — user requested full 20-gen budget, no early stop.
PHASE2_PLATEAU_POST_RESTART_STOP_ENABLED = False

# PHASE2_PLATEAU_POST_RESTART_STOP_PATIENCE — gens of no improvement after a
#   restart before breaking.  Should be >= PHASE2_PLATEAU_POST_RESTART_BOOST_GENS
#   so the boosted-mutation window gets a full chance to recover.
#   Higher → more conservative (give restart more time); slower.
#   Lower  → stop sooner after a failed restart; faster, tiny risk of cutting a
#            late breakthrough (an improvement resets the streak, so no false stop).
# Increased 3→5: less aggressive early stop; patience=3 killed epochs before
# restart could recover (was < boost_gens=4 now > boost_gens).
PHASE2_PLATEAU_POST_RESTART_STOP_PATIENCE = 5

# PHASE2_PLATEAU_MAX_RESTARTS — restarts per epoch before final break.
#   3       → up to 3 diversity restarts, then break on the next plateau.
#   0       → immediately break (disables restart regardless of ENABLED flag).
PHASE2_PLATEAU_MAX_RESTARTS = 3

# PHASE2_VIABILITY_COLLAPSE_THRESHOLD — pop_viable fraction below which viability
#   is considered collapsed (triggers forced restart after streak).
#   Higher → more sensitive to viable-population shrinkage.
#   Lower  → tolerate very small viable populations before restart.
PHASE2_VIABILITY_COLLAPSE_THRESHOLD = 0.5

# PHASE2_VIABILITY_COLLAPSE_STREAK — consecutive generations of viability collapse
#   before a forced diversity restart is triggered.
#   Higher → more tolerance before restart; may stall longer.
#   Lower  → aggressive restart on any prolonged viability dip.
PHASE2_VIABILITY_COLLAPSE_STREAK = 3

# PHASE2_FEASIBILITY_VIOLATION_WEIGHT — scales soft penalty for floor violations.
#   Higher → infeasible rules pushed far down on all objectives.
#   Lower  → borderline rules compete with feasible ones longer.
# Lowered 25→15: this is multiplied into support_penalty which is then re-weighted
# by PHASE2_SUPPORT_PENALTY_WEIGHT_F2/F3 (=0.6), so one violation becomes
# 0.6×25=15 added to f2/f3 — i.e. 60-300% of natural DD scale, too harsh.
# 15.0 yields an effective 0.6×15=9 per violation: meaningful but not crushing.
PHASE2_FEASIBILITY_VIOLATION_WEIGHT = 15.0

# PHASE2_INFEASIBLE_OBJECTIVE_PENALTY — flat penalty added when hard infeasible.
#   Higher → clear separation feasible vs infeasible on Pareto front.
#   Lower  → infeasible rules may linger in ranking.
# Lowered 100→50: with compressed Sortino in 0–3, 100 was 30× the natural f1
# scale — overkill. 50 is 15× Sortino which still buries any hard-infeasible
# rule on the Pareto front. Note: the hardcoded trade_penalty=50 (below
# MIN_TRADE_POOL_FLOOR) is the actual death sentence in the current code.
PHASE2_INFEASIBLE_OBJECTIVE_PENALTY = 50.0

# PHASE2_DEPLOYABLE_ARCHIVE_MAX_SIZE — cap on stored deployable-elite archive.
#   Higher → more warm-start diversity across runs; more disk/RAM.
#   Lower  → smaller cross-run memory.
# History: 200 → 100 (Colab RAM) → 75 (debug-scope economy). For the current
# two-symbol profile and two directions, 75 deployable elites provide enough
# warm-start diversity while limiting cross-run memory.
PHASE2_DEPLOYABLE_ARCHIVE_MAX_SIZE = 75

# --- elite-preservation guard (prevents mid-epoch erosion) ---
# When enabled, the top-K deployable-archive elites are force-preserved in
# the live population each generation under (μ+λ) selection. This prevents
# a non-dominated elite from being evicted purely by recomputed dynamic
# diversity/support penalties as hall_of_fame and pareto_archive grow.
PHASE2_ELITE_PRESERVATION_ENABLED: bool = True
PHASE2_ELITE_PRESERVATION_TOP_K: int = 5
PHASE2_ELITE_PRESERVATION_MIN_GEN: int = 1

# --- Diversity recovery (inject randomness when unique ratio collapses) ---

PHASE2_DIVERSITY_RECOVERY_ENABLED = True

# PHASE2_DIVERSITY_RECOVERY_MIN_UNIQUE_RATIO — trigger when uniqueness below this.
#   Higher → recovery fires sooner; more aggressive anti-collapse.
#   Lower  → tolerate more duplicate-heavy populations.
PHASE2_DIVERSITY_RECOVERY_MIN_UNIQUE_RATIO = 0.30

# PHASE2_DIVERSITY_RECOVERY_INJECT_FRACTION — fraction of pop replaced on recovery.
#   Higher → bigger shock to population; more exploration, disrupts elites.
#   Lower  → mild injection.
PHASE2_DIVERSITY_RECOVERY_INJECT_FRACTION = 0.30

# PHASE2_DIVERSITY_RECOVERY_MUTATION_BOOST — temporary mutation rate multiplier.
#   Higher → more radical offspring during recovery.
#   Lower  → gentler nudge away from local niche.
PHASE2_DIVERSITY_RECOVERY_MUTATION_BOOST = 1.75

# --- Two-stage evolution: wide exploration → lower-mutation refinement ---
# With PHASE2_JOINT_TRAIN_VAL=False and PHASE2_VAL_IN_FITNESS_PENALTY=False,
# both stages optimize train-space fitness. Validation is an admission screen,
# not a Stage-B objective; do not describe this schedule as val refinement.

PHASE2_TWO_STAGE_ENABLED = True

# PHASE2_STAGE_A_GENERATIONS — Stage A (exploration) generation budget.
#   Higher → more diverse initial Pareto before lower-mutation Stage B.
#   Lower  → quicker handoff; Stage B may miss good regions.
# Scaled to PHASE2_GENERATIONS=100 (A:B = 65:35).
PHASE2_STAGE_A_GENERATIONS = 65

# PHASE2_STAGE_B_GENERATIONS — Stage B (refinement) generation budget.
#   Higher → more polishing after exploration; total time = A + B gens.
#   Lower  → less refinement after exploration.
# Matched to 100-gen budget (A:B = 65:35).
PHASE2_STAGE_B_GENERATIONS = 35

# PHASE2_STAGE_B_SEED_TOP_K — elites from Stage A seeded into Stage B.
#   Higher → broader refinement starting set; slower Stage B per gen.
#   Lower  → refine only top performers; risk missing dark horses.
# 50→20 — pop=60 short Stage B.
PHASE2_STAGE_B_SEED_TOP_K = 20

# PHASE2_STAGE_B_SEED_FRACTION — fraction of Stage B pop seeded from Stage A elites.
#   Higher → more refinement around known good regions; risk of clone collapse.
#   Lower  → more random exploration in Stage B.
PHASE2_STAGE_B_SEED_FRACTION = 0.35

# --- Stage A hyperparameters (exploration: higher mutation, stronger diversity) ---

# PHASE2_STAGE_A_MUTATION_RATE — per-gene mutation in Stage A.
#   Higher → more genetic exploration before Stage B refinement.
# 0.25→0.35 — short Stage A needs aggressive exploration.
PHASE2_STAGE_A_MUTATION_RATE = 0.35

# PHASE2_STAGE_A_MUTATION_WEIGHTED_ACTIVATE_PROB — bias toward activating genes in A.
# 0.50→0.65 — more active conditions → more trades on island windows.
PHASE2_STAGE_A_MUTATION_WEIGHTED_ACTIVATE_PROB = 0.70

# PHASE2_STAGE_A_DIVERSITY_PENALTY — crowding penalty on objectives in Stage A.
PHASE2_STAGE_A_DIVERSITY_PENALTY = 8.0

# PHASE2_STAGE_A_DIVERSITY_HAMMING_THRESHOLD — min Hamming distance before penalty in A.
PHASE2_STAGE_A_DIVERSITY_HAMMING_THRESHOLD = 3

# PHASE2_STAGE_A_DIVERSITY_RECOVERY_MIN_UNIQUE_RATIO — trigger diversity injection in A.
PHASE2_STAGE_A_DIVERSITY_RECOVERY_MIN_UNIQUE_RATIO = 0.40

# PHASE2_STAGE_A_DIVERSITY_RECOVERY_INJECT_FRACTION — pop replaced on recovery in A.
PHASE2_STAGE_A_DIVERSITY_RECOVERY_INJECT_FRACTION = 0.30

# PHASE2_STAGE_A_DIVERSITY_RECOVERY_MUTATION_BOOST — mutation multiplier after recovery in A.
PHASE2_STAGE_A_DIVERSITY_RECOVERY_MUTATION_BOOST = 2.0

# PHASE2_STAGE_A_PLATEAU_EARLY_STOP_PATIENCE — gens without progress before stop in A.
PHASE2_STAGE_A_PLATEAU_EARLY_STOP_PATIENCE = 12

# PHASE2_STAGE_A_PLATEAU_EARLY_STOP_MIN_GENERATION — earliest plateau stop gen in A.
# Must be ≤ PHASE2_STAGE_A_GENERATIONS (asserted at import).
PHASE2_STAGE_A_PLATEAU_EARLY_STOP_MIN_GENERATION = 8

# PHASE2_STAGE_A_EARLY_STOP_MIN_GENERATION — earliest return-based early stop in A.
PHASE2_STAGE_A_EARLY_STOP_MIN_GENERATION = 10

# PHASE2_STAGE_A_ARCHIVE_SEED_FRACTION — warm-start fraction from prior pool in Stage A.
PHASE2_STAGE_A_ARCHIVE_SEED_FRACTION = 0.25

# --- Stage A evolution floor overrides (loose fitness gates; pool export stays strict) ---

# PHASE2_STAGE_A_RETURN_FLOOR_PCT — min train return % during Stage A fitness only.
PHASE2_STAGE_A_RETURN_FLOOR_PCT = 0.0

# PHASE2_STAGE_A_MIN_TRADE_SUPPORT — trade-count target before support penalty vanishes in Stage A.
# 15→8 — sparse context (permission∧trigger) needs a reachable exploration floor.
PHASE2_STAGE_A_MIN_TRADE_SUPPORT = 8

# PHASE2_STAGE_A_USE_ROBUST_RETURN_OBJ — applies only when the optional
# total-return f3 objective is enabled; the default profit-factor f3 ignores it.
PHASE2_STAGE_A_USE_ROBUST_RETURN_OBJ = True

# PHASE2_STAGE_A_SOFT_FEASIBILITY — Stage A uses soft penalties instead of hard infeasible block.
PHASE2_STAGE_A_SOFT_FEASIBILITY = True

# --- Viability-aware diversity recovery (Stage A; complements uniqueness-based recovery) ---

# PHASE2_VIABILITY_RECOVERY_ENABLED — inject deployable-archive seeds when valid_rules collapse.
PHASE2_VIABILITY_RECOVERY_ENABLED = True

# PHASE2_VIABILITY_RECOVERY_MIN_VALID — trigger when Pareto valid_rules falls below this.
PHASE2_VIABILITY_RECOVERY_MIN_VALID = 5

# PHASE2_VIABILITY_RECOVERY_DEPLOYABLE_MUTATE_FRACTION — share of injected slots from archive mutate.
PHASE2_VIABILITY_RECOVERY_DEPLOYABLE_MUTATE_FRACTION = 0.5

# --- Stage B hyperparameters (refinement: lower mutation, allow clustering) ---

# PHASE2_STAGE_B_MUTATION_RATE — per-gene mutation in Stage B.
PHASE2_STAGE_B_MUTATION_RATE = 0.20

# PHASE2_STAGE_B_MUTATION_WEIGHTED_ACTIVATE_PROB — conservative gene activation in B.
PHASE2_STAGE_B_MUTATION_WEIGHTED_ACTIVATE_PROB = 0.45

# PHASE2_STAGE_B_DIVERSITY_PENALTY — weaker crowding penalty during refinement.
PHASE2_STAGE_B_DIVERSITY_PENALTY = 4.0

# PHASE2_STAGE_B_DIVERSITY_HAMMING_THRESHOLD — allow nearer-duplicate elites in B.
PHASE2_STAGE_B_DIVERSITY_HAMMING_THRESHOLD = 2

# PHASE2_STAGE_B_DIVERSITY_RECOVERY_MIN_UNIQUE_RATIO — later diversity recovery in B.
PHASE2_STAGE_B_DIVERSITY_RECOVERY_MIN_UNIQUE_RATIO = 0.25

# PHASE2_STAGE_B_DIVERSITY_RECOVERY_INJECT_FRACTION — smaller injection shock in B.
PHASE2_STAGE_B_DIVERSITY_RECOVERY_INJECT_FRACTION = 0.20

# PHASE2_STAGE_B_DIVERSITY_RECOVERY_MUTATION_BOOST — milder post-recovery mutation in B.
PHASE2_STAGE_B_DIVERSITY_RECOVERY_MUTATION_BOOST = 1.4

# PHASE2_STAGE_B_PLATEAU_EARLY_STOP_PATIENCE — shorter patience while polishing in B.
PHASE2_STAGE_B_PLATEAU_EARLY_STOP_PATIENCE = 8

# PHASE2_STAGE_B_PLATEAU_EARLY_STOP_MIN_GENERATION — earliest plateau stop gen in B.
PHASE2_STAGE_B_PLATEAU_EARLY_STOP_MIN_GENERATION = 5

# PHASE2_STAGE_B_EARLY_STOP_MIN_GENERATION — earliest return-based early stop in B.
PHASE2_STAGE_B_EARLY_STOP_MIN_GENERATION = 6

# PHASE2_GPU_ENRICH_SYMBOL_METRICS — merge CPU per-symbol metrics after GPU batch eval.
PHASE2_GPU_ENRICH_SYMBOL_METRICS = True

# PHASE2_ENRICH_SYMBOL_METRICS_EVERY_N_GENS — throttle CPU enrichment cadence.
#   CPU full re-simulation (per_symbol_metrics) is expensive (~60s/gen).
#   Only run every N generations (always on last gen) to keep symbol-spread
#   penalty fresh enough without paying the full cost every generation.
#   1 = every gen (original behavior).
#   20 = every 20th gen (default — keeps evolution GPU-bound; the mandatory
#   exact CPU archive pass still refreshes the final pool metrics).
PHASE2_ENRICH_SYMBOL_METRICS_EVERY_N_GENS = 20
# =============================================================================
# Phase 2 — NSGA-III search budget & archive
# =============================================================================

# PHASE2_POPULATION_SIZE — individuals per generation.
#   Higher → better Pareto coverage, ~linear GPU cost per generation.
#   Lower  → faster gens, risk of premature convergence.
PHASE2_POPULATION_SIZE = 500

# PHASE2_GENERATIONS — generation budget for the global Phase 2 search.
PHASE2_GENERATIONS = 100

PHASE2_ALGORITHM = "NSGA3"

# PHASE2_ARCHIVE_MAX_SIZE — max stored non-dominated solutions across gens.
#   Higher → richer elite memory; more memory, slower non-dominated sorting.
#   Lower  → leaner archive; may lose good rules found early.
# 120→300 — scaled with pop=200.
PHASE2_ARCHIVE_MAX_SIZE = 800

# PHASE2_ARCHIVE_SEED_FRACTION — fraction of initial pop from cross-run archive.
#   Higher → more warm-start from past runs; less fresh random exploration.
#   Lower  → more random init; slower reuse of known good rules.
PHASE2_ARCHIVE_SEED_FRACTION = 0.25

PHASE2_SEED: int = get_seed()


# =============================================================================
# Phase 2 — Search topology
# =============================================================================
# Global Phase 2: unified multi-objective evolutionary search across all symbols.

# =============================================================================
# Phase 2 — Engine, initialization & mutation
# =============================================================================

# PHASE2_USE_GPU — JAX GPU backtest during evolution.
#   True  → fast Phase 2 (default).  False → CPU only (low_ram tuning profile).
PHASE2_USE_GPU = True

# PHASE2_NUMBA_ENABLED — Numba-accelerated NSGA helper kernels.
#   True  → faster non-dominated sort / crowding on CPU.
#   False → pure NumPy; easier debugging.
PHASE2_NUMBA_ENABLED = True

# PHASE2_INIT_STRATEGY — initial population construction.
#   "stratified_sparse" → biased toward valid condition counts (recommended).
#   "legacy"            → older uniform random init.
PHASE2_INIT_STRATEGY = "stratified_sparse"

# PHASE2_INIT_STRATUM_FRACTIONS — mix of init strata (explore vs exploit seeds).
#   Shift toward first fraction → more random sparse rules.
#   Shift toward second → more archive-biased / structured seeds.
PHASE2_INIT_STRATUM_FRACTIONS = (0.4, 0.6)

# PHASE2_INIT_SOFTMAX_TEMP — temperature for weighted feature activation in init.
#   Higher → more uniform random feature picks.
#   Lower  → strongly favor high-MI features in initial conditions.
PHASE2_INIT_SOFTMAX_TEMP = 2.0

PHASE2_INIT_SCORE_EPS = 1e-6

# PHASE2_INIT_UNIFORM_MIX — probability of uniform random gene vs structured init.
#   Higher → more random chromosomes in initial population.
#   Lower  → more MI-guided structured rules at gen 0.
PHASE2_INIT_UNIFORM_MIX = 0.1

# PHASE2_MUTATION_RATE — per-gene mutation probability.
#   Higher → more exploration, noisier convergence, better escape local optima.
#   Lower  → finer local search, risk of premature convergence.
# Increased 0.3→0.35: more exploration to compensate for fewer generations
# (132→100); helps escape local optima in tighter budget.
PHASE2_MUTATION_RATE = 0.32

# PHASE2_MUTATION_WEIGHTED_ACTIVATE_PROB — bias mutations toward activating genes.
#   Higher → mutations tend to add conditions rather than dont_care.
#   Lower  → mutations more often deactivate or flip existing conditions.
PHASE2_MUTATION_WEIGHTED_ACTIVATE_PROB = 0.4


# =============================================================================
# Shared monthly validation
# =============================================================================

MONTHLY_WINDOW_DAYS = 30
MONTHLY_WINDOW_MIN_ROWS = 2500
MONTHLY_WINDOW_MAX_WINDOWS = 24
MONTHLY_RECENCY_WEIGHT = 2.2
MONTHLY_MIN_TRADES = 20
# The Phase 2 gate uses a smaller per-window floor because its validation
# windows are short; the broader monthly summary keeps the historical floor
# above for score penalties and reporting.
MONTHLY_ACTIVE_MIN_TRADES = 3
# Active returns in this band are treated as range-market noise rather than
# bearish months when the monthly stability diagnostics are computed.
MONTHLY_FLAT_TOLERANCE_PCT = 0.50
MONTHLY_MIN_ACTIVE_RATIO = 0.60
MONTHLY_MAX_BEARISH_RATIO = 0.50
MONTHLY_GOOD_RETURN_MIN_PCT = 0.5
MONTHLY_MIN_PROFITABLE_RATIO = 0.60
MONTHLY_WORST_RETURN_FLOOR = -1.5
MONTHLY_WORST_PF_FLOOR = 1.0
MONTHLY_MAX_DD = 8.0
MONTHLY_WORST_RETURN_WEIGHT = 1.2
MONTHLY_WORST_PF_WEIGHT = 8.0
MONTHLY_DD_WEIGHT = 0.7
MONTHLY_PROFITABLE_RATIO_WEIGHT = 15.0
MONTHLY_ACTIVE_RATIO_WEIGHT = 15.0
MONTHLY_BEARISH_RATIO_WEIGHT = 15.0
MONTHLY_TREND_WEIGHT = 2.0
MONTHLY_LATEST_WEIGHT = 0.6

# Phase 5 — consumed test diagnostics plus optional untouched forward acceptance
# =============================================================================

# PHASE5_VALIDATION_RETURN_GATE_PCT — min val return % for deployment flag.
#   Higher → fewer strategies marked deployable after pipeline.
#   Lower  → marginal val performers still flagged OK (risky for live).
PHASE5_VALIDATION_RETURN_GATE_PCT = 2.0

# PHASE5_VALIDATION_PROFIT_FACTOR_GATE — min val PF for deployment flag.
#   Higher → stricter deployment filter on gross win/loss ratio.
#   Lower  → strategies with thin edge pass deployment check.
PHASE5_VALIDATION_PROFIT_FACTOR_GATE = 1.05

# Phase 5 is report-only: it never mutates a strategy using held-out test
# performance. Test-set pruning would turn the OOS report into a tuning step.


# =============================================================================
# RB Governor — rule selection and risk tuning
# =============================================================================
# RB is the only production selection/risk path. There is intentionally no
# enable/disable switch: two competing implementations caused configuration
# drift and made the evaluator contract ambiguous.

# --- Rule scoring / gating ---

# RB_MIN_TRAIN_RETURN / RB_MIN_VALID_RETURN — return-% floors below which a
#   single rule fails ``_is_positive_good`` (dual positivity on train+val).
# 0.5→0.25 — match PHASE2_* return floors so island rules that
#   clear Phase 2 admission are not wiped on RB val_selection resim
#   (Colab: kept 1/15 long, 4/23 short → tiny teams → concentration fail-closed).
#   Correlated with RB_MAX_PAIR_OVERLAP / score-improvement easing below:
#   more survivors only help if compose can grow multi-island teams.
#   Keep dual-positivity; RB has no fallback path and fails closed on rejection.
RB_MIN_TRAIN_RETURN: float = 0.25
RB_MIN_VALID_RETURN: float = 0.25

# RB_MIN_TRAIN_PF / RB_MIN_VALID_PF — minimum profit factor for each split.
#   1.0 = break-even before fees.
RB_MIN_TRAIN_PF: float = 1.02
RB_MIN_VALID_PF: float = 1.02

# RB_MIN_TRAIN_TRADES / RB_MIN_VALID_TRADES — per-rule trade-count floors
#   used by ``gate_positive_good`` and ``_score_metrics`` for single rules.
#   Kept moderate because optional cluster/specialist paths can see thin
#   validation slices even when the default global path uses both symbols.
RB_MIN_TRAIN_TRADES: int = 10
RB_MIN_VALID_TRADES: int = 6

# RB_CANDIDATE_RISK_ADMISSION_ENABLED — legacy post-discovery TP/SL rescue.
# False is the production research contract: a Phase 2 candidate must pass at
# its discovery exit geometry. Enabling this flag creates a new exit-policy
# search family and must only be used by an explicitly nested experiment.
RB_CANDIDATE_RISK_ADMISSION_ENABLED: bool = False
# Internal policy marker set only by Pipeline_Orchestrator.  Small compatibility
# callers may exercise legacy helper behavior without bypassing the canonical
# production contract.
RB_CANONICAL_PIPELINE_ACTIVE: bool = False

# RB_RULESET_MIN_* — trade-count floors applied to the composed team (all
#   rules together).  Should be larger than the per-rule floors because the
#   combined team fires more frequently than any single rule.
RB_RULESET_MIN_TRAIN_TRADES: int = 12
RB_RULESET_MIN_VALID_TRADES: int = 8


# --- Pool & candidate limits ---

# RB_MAX_POOL_RULES_TO_EVALUATE — cap on Phase 2 pool rules passed through
#   symbol-specialization in ``_filter_good_rules``.
#   Higher → more candidates, slower filtering. 400 covers the default
#   evolved pool plus its bounded deterministic univariate complement.
RB_MAX_POOL_RULES_TO_EVALUATE: int = 400

# RB_KEEP_TOP_RULES — how many positive-good candidates survive the
#   single-rule ranking to feed ``_compose_ruleset``.
#   Should be comfortably larger than PHASE2_KEEP_TOP_RULES.
RB_KEEP_TOP_RULES: int = 150

# RB is a governor over Phase 2 discoveries, not a second rule generator.
# Keep the historical univariate complement available as an explicitly
# opt-in diagnostic helper, but never add it to the production candidate pool.
RB_UNIVARIATE_BASELINE_ENABLED: bool = False
# Every production RB candidate/final rule must preserve the Phase 2
# feature-condition contract. Legacy callers may override this only in
# isolated tests.
RB_PHASE2_PROVENANCE_ONLY: bool = True
RB_UNIVARIATE_BASELINE_MAX_RULES: int = 400
# Include a generalist form of each one-condition baseline in addition to the
# symbol-specialized forms.  The former is important when a condition has a
# stable cross-asset edge but the Phase-2 chromosome happened not to emit the
# corresponding value for both symbols.  Both forms are still evaluated by
# the exact RB gates before deployment.
RB_UNIVARIATE_GENERALIST_ENABLED: bool = True

# A bounded recency-regime rescue for non-stationary markets.  Ordinary RB
# candidates must remain positive on the historical train split.  A rescue is
# considered only when *both* chronological validation halves are positive,
# have adequate PF/trade support, and the older train loss/drawdown are within
# explicit limits.  It never reads Phase-5 test data and is recorded in the
# strategy/report when selected.
RB_RECENCY_RESCUE_ENABLED: bool = False
RB_RECENCY_MIN_VALID_RETURN: float = 0.50
RB_RECENCY_MIN_VALID_PF: float = 1.05
# The chronological halves are intentionally short on this data.  Ten trades
# per half is still enough for a bounded rescue while allowing a compact rule
# to survive the sparse older/newer split.
RB_RECENCY_MIN_VALID_TRADES: int = 10
RB_RECENCY_MIN_TRAIN_TRADES: int = 25
RB_RECENCY_MIN_TRAIN_PF: float = 0.80
RB_RECENCY_MAX_TRAIN_LOSS_PCT: float = 12.0
RB_RECENCY_MAX_TRAIN_DD_PCT: float = 25.0
RB_RECENCY_MAX_SYMBOL_LOSS_PCT: float = 15.0
RB_RECENCY_MAX_SYMBOL_SHARE_ABS_PNL: float = 0.85
RB_RECENCY_MAX_SYMBOL_HHI: float = 0.75
RB_RECENCY_MIN_SCORE_MARGIN: float = 0.0
RB_RECENCY_MAX_CANDIDATES: int = 40

# When a direction is fail-closed after the chronological validation-selection
# frame, retry that direction once on the complete train/validation holdout.
# This remains validation-only (Phase 5 test data is never read) and avoids
# discarding a direction merely because the half-window was too sparse for a
# balanced portfolio certificate.  Accepted directions do not pay this cost.
RB_FULL_VALIDATION_RECOVERY_ENABLED: bool = False


# --- Team composition ---

# RB_MIN_RULES / RB_MAX_RULES — output bounds for accepted strategies.
RB_MIN_RULES: int = 1
# To reach RB_MAX_RULES with the default 5% capital-grid minimum, the
# evaluator-compatible 100% total-capital cap remains feasible.
# 10→20 — many-moderate-rules package.
RB_MAX_RULES: int = 20

# RB_MAX_PAIR_OVERLAP — max Hamming-style overlap between any two rules in
#   the team. Lower = more diverse team, harder to grow.
# 0.25→0.35 — with traded-symbol coverage (not island OR filters),
#   compose must pull rules from other islands; mild condition overlap is OK
#   if traded symbols differ. Correlated with MIN_DISTINCT + coverage fix.
RB_MAX_PAIR_OVERLAP: float = 0.35

# Bounded certificate-first diversification search.  The beam is deliberately
# small because each state requires a full CPU train/validation simulation.
RB_DIVERSIFICATION_BEAM_WIDTH: int = 6
RB_DIVERSIFICATION_STEPS: int = 4
RB_DIVERSIFICATION_GLOBAL_LEADERS: int = 6
RB_DIVERSIFICATION_SYMBOL_LEADERS: int = 2
# Keep a few return leaders per symbol in addition to score leaders. High
# frequency candidates can carry strong net PnL yet receive health penalties
# before the risk grid lowers their allocation.
RB_DIVERSIFICATION_RETURN_LEADERS: int = 4

# RB_RULESET_MUST_BEAT_SUBSETS — a candidate team must beat both its parent
#   subset and the standalone candidate on both train and val return.
RB_RULESET_MUST_BEAT_SUBSETS: bool = False

# RB_MIN_SCORE_IMPROVEMENT — minimum delta in the governor score to add a
#   new rule in ``_compose_ruleset``.
# 0.03→0.01 — ease team growth once more singles survive; still
#   require positive score delta (not zero).
RB_MIN_SCORE_IMPROVEMENT: float = 0.01

# RB_MIN_TRAIN_RETURN_IMPROVEMENT / RB_MIN_VALID_RETURN_IMPROVEMENT — min
#   return-% uplift required from adding a candidate rule.
# 0.005→0.002 — correlated with score-improvement ease so adding
#   a diversifying island rule is not blocked by tiny combined-return noise.
RB_MIN_TRAIN_RETURN_IMPROVEMENT: float = 0.002
RB_MIN_VALID_RETURN_IMPROVEMENT: float = 0.002

# RB_RETURN_DD_FLOOR — drawdown floor (%) used when converting return to a
#   return/drawdown ratio inside ``_score_metrics``.
RB_RETURN_DD_FLOOR: float = 0.50

# RB_TRADE_PENALTY — per-trade penalty weight applied when a rule falls
#   below the minimum trade-count floors.
RB_TRADE_PENALTY: float = 0.70

# RB_TRAIN_VALID_RATIO_GAP_WEIGHT / RB_TRAIN_VALID_RETURN_GAP_WEIGHT —
#   overfit penalties applied to the train>>val gap in the score.
RB_TRAIN_VALID_RATIO_GAP_WEIGHT: float = 30.0
RB_TRAIN_VALID_RETURN_GAP_WEIGHT: float = 4.0


# --- Lenient-add compatibility mode -----------------------------------------
# RB_RULE_ADD_IGNORE_SUBSET_BEAT and RB_MIN_COMBINED_RETURN_IMPROVEMENT are
# active only when RB_RULE_ADD_BY_RETURN_ONLY=True.

# RB_RULE_ADD_BY_RETURN_ONLY — add rules purely on combined-return uplift
#   (skips the stricter subset-beat and overlap checks when paired with
#   RB_RULE_ADD_IGNORE_OVERLAP=True).  Profit amplifier still re-checks.
RB_RULE_ADD_BY_RETURN_ONLY: bool = False
RB_RULE_ADD_IGNORE_OVERLAP: bool = False
RB_RULE_ADD_IGNORE_SUBSET_BEAT: bool = True
# Active only when RB_RULE_ADD_BY_RETURN_ONLY=True.
# min combined return-% uplift to add a new rule
RB_MIN_COMBINED_RETURN_IMPROVEMENT: float = 3.5


# --- Train-valid shape prior (anti-overfit) ---

# RB_REQUIRE_TRAIN_SLIGHTLY_ABOVE_VALID — soft shape preference in scoring only
#   (bonus/penalty). Hard reject was removed from ``_is_positive_good`` because
#   the old narrow band (ratio 1.03–1.15 + min abs gap 0.20) rejected every
#   healthy Phase 2 pool rule (run.log: 0 positive-good → fail-closed empty).
RB_REQUIRE_TRAIN_SLIGHTLY_ABOVE_VALID: bool = True
RB_TRAIN_VALID_MIN_RATIO: float = 0.90
RB_TRAIN_VALID_MAX_RATIO: float = 2.00
RB_TRAIN_VALID_MIN_ABS_GAP: float = 0.0
RB_TRAIN_VALID_MAX_ABS_GAP: float = 12.0
RB_TRAIN_BELOW_VALID_PENALTY: float = 120.0
RB_TRAIN_TOO_HIGH_PENALTY: float = 180.0
RB_TRAIN_VALID_SHAPE_BONUS: float = 80.0


# --- Default risk parameters (initial TP/SL/capital_pct embedded in rules
#     before risk optimization, and minimum allowed values) ---

RB_DEFAULT_TP: float = 2.0
RB_DEFAULT_SL: float = 1.2
RB_DEFAULT_CAPITAL_PCT: float = 18.0
RB_REQUIRE_TP_SL_ABOVE_ONE: bool = True
RB_MIN_TP: float = 1.0
RB_MIN_SL: float = 1.0
# RB_RISK_OPTIMIZE_EXITS — exit barriers are part of strategy identity.
# False means RB may size capital but cannot silently rewrite TP/SL discovered
# by Phase 2. An exit experiment must create a new strategy family.
RB_RISK_OPTIMIZE_EXITS: bool = False
RB_EXPECTANCY_LCB_MARGIN_PCT: float = 0.0
RB_COST_STRESS_MULTIPLIERS: tuple[float, ...] = (1.0, 1.5)
RB_COST_STRESS_ENABLED: bool = True
RB_COST_STRESS_MIN_RETURN_PCT: float = 0.0
RB_MONTHLY_CERTIFICATE_ENABLED: bool = True
RB_MONTHLY_MIN_PROFITABLE_RATIO: float = 0.55
RB_MONTHLY_MAX_BEARISH_RATIO: float = 0.35


# --- RB risk-grid search -----------------------------------------------------

# RB_TP_GRID / RB_SL_GRID / RB_CAPITAL_GRID — experimental exit and capital
# profiles. Production uses the capital grid only because TP/SL are immutable
# strategy identity; exit grids remain available for explicitly nested studies.
# The minimum is 5% so the maximum 20-rule team fits under the evaluator's
# 100% exposure contract before normalization.
RB_TP_GRID: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0)
RB_SL_GRID: tuple[float, ...] = (1.0, 1.2, 1.5, 2.0)
RB_CAPITAL_GRID: tuple[float, ...] = (5.0, 7.5, 10.0, 12.0, 15.0, 18.0)

# RB_RISK_OPT_PASSES — round-robin passes through all rules.
RB_RISK_OPT_PASSES: int = 1

# RB_RISK_MIN_IMPROVEMENT — min score delta to accept a new TP/SL/cap combo.
RB_RISK_MIN_IMPROVEMENT: float = 0.02

# RB_MAX_TOTAL_CAPITAL — hard cap on sum(capital_pct) across all rules.
# This must match MAX_TOTAL_EXPOSURE_PCT and evaluator_v5.ipynb.
RB_MAX_TOTAL_CAPITAL: float = 100.0

# RB_RISK_GRID_WF_SPLITS — walk-forward folds for risk grid (1 = legacy single-fold).
#   Score every TP/SL/capital combo on N chronological folds of val_selection,
#   pick the combo with the best min(fold1, fold2, ...) score (worst-case selection).
#   → fixes audit finding #3 (RB Governor risk-grid overfits val_selection)
RB_RISK_GRID_WF_SPLITS: int = 3

# Final fraction of validation-selection data reserved for the untouched RB
# tail check. This is separate from Phase 2 fitness data and is never used for
# Optuna parameter selection.
RB_TAIL_HOLDOUT_FRACTION: float = 0.25

# RB_RISK_GRID_USE_TAIL_HOLDOUT — reserve final RB_TAIL_HOLDOUT_FRACTION
# of val_selection for chronological robustness validation.  Set False to use
# the full val_selection for folds.
RB_RISK_GRID_USE_TAIL_HOLDOUT: bool = True

# RB_TAIL_HOLDOUT_HARD_GATE — when True, strategies whose tail-holdout return
#   is below RB_TAIL_HOLDOUT_MIN_RETURN_PCT fail closed: empty ruleset written
#   with deployment_accepted=False (not a soft flag with rules retained).
RB_TAIL_HOLDOUT_HARD_GATE: bool = True
RB_TAIL_HOLDOUT_MIN_RETURN_PCT: float = 0.0

# Enforce the same reserved chronological tail while composing and tuning a
# ruleset.  Applying this only after greedy selection could discard the whole
# direction despite other tail-positive validation candidates being available.
# This remains validation-only; Phase 5 test data is never read here.
RB_TAIL_HOLDOUT_SELECTION_GATE: bool = True
RB_TAIL_HOLDOUT_MIN_TRADES: int = 4

# RB_MAX_SYMBOL_SHARE_ABS_PNL — max fraction of abs PnL from a single symbol on
#   the RB validation frame; above this → fail-closed empty strategy
#   (deployment_accepted=False, rules_set cleared).
# A two-symbol portfolio can never have a top share below 0.50 except at exact
# equality. 0.67 requires a material secondary contributor while allowing the
# risk grid to certify a realistic 2:1 split.
RB_MAX_SYMBOL_SHARE_ABS_PNL: float = 0.67
# RB_MAX_SYMBOL_HHI — max Herfindahl index of abs PnL across symbols on valid;
#   above this → same fail-closed empty strategy as share gate.
RB_MAX_SYMBOL_HHI: float = 0.60


# --- Symbol strategy: two modes (do not mix) ---------------------------------
#
# Mode A — multi-symbol TEAM (recommended / current):
#   RB_REQUIRE_SYMBOL_FILTERS=False
#   Global Phase 2 rules are scored on pooled BTC/ETH data. Compose uses
#   **traded** symbols from metrics, not discovery-time symbol partitions.
#
# Mode B — per-symbol SPECIALISTS (compatibility path; avoid unless intentional):
#   RB_REQUIRE_SYMBOL_FILTERS=True
#   Each rule locks to ``symbol is X``; compose many specialists for coverage.
#   Tends to jagged single-symbol equity; needs thick per-symbol pools.

# RB_REQUIRE_SYMBOL_FILTERS — Mode B when True; Mode A when False.
RB_REQUIRE_SYMBOL_FILTERS: bool = False

# RB_MIN_DISTINCT_SYMBOLS — target symbol coverage while composing.
#   Mode A: traded symbols from metrics must expand toward this.
#   Mode B: distinct ``symbol is X`` filters on final rules (hard gate).
RB_MIN_DISTINCT_SYMBOLS: int = 2

# RB_ALLOW_PARTIAL_SPECIALIST_COVERAGE — Mode B only. When True, a direction
# may deploy with fewer symbols than RB_MIN_DISTINCT_SYMBOLS if a missing
# symbol has no positive candidate. Kept False for the global two-symbol
# product: a BTC+ETH release must not silently become BTC-only.
RB_ALLOW_PARTIAL_SPECIALIST_COVERAGE: bool = False
# A configured two-symbol release must cover the configured universe.
RB_MULTI_SYMBOL_RELEASE: bool = True

# Soft score bonus per extra traded symbol beyond the first (Mode A ranking).
# 8→15 — stronger preference for multi-symbol traded coverage
#   when ranking singles into the compose pool (correlated with coverage fix).
RB_MULTI_SYMBOL_COVERAGE_BONUS: float = 15.0

# RB_SYMBOL_USE_COMBINATIONS — Mode B only: also try 2-/3-symbol filter combos.
RB_SYMBOL_USE_COMBINATIONS: bool = False

# RB_SYMBOL_MAX_SYMBOLS_PER_RULE — Mode B only: max ``symbol is X`` per rule.
RB_SYMBOL_MAX_SYMBOLS_PER_RULE: int = 1

RB_SYMBOL_MAX_VARIANTS_PER_RULE: int = 10
RB_SYMBOL_MIN_TRAIN_TRADES: int = 10
RB_SYMBOL_MIN_VALID_TRADES: int = 4
RB_SYMBOL_STRICT_OUTPUT_CHECK: bool = True


# --- Evaluator-health penalties (mirror evaluator_v5 execution checks) ---

RB_MAX_SKIPPED_SIGNAL_RATIO: float = 0.20
RB_MIN_EXECUTED_RAW_RATIO: float = 0.60
# RB_REQUIRE_EXECUTION_HEALTH_ON_SINGLES — hard-gate singles on skip/exec ratios.
#   False (default): health stays a soft score penalty only; hard skip gates
#   emptied otherwise-profitable island rules (skip≈0.40 with +return).
#   True: restore legacy hard gate inside ``_is_positive_good``.
RB_REQUIRE_EXECUTION_HEALTH_ON_SINGLES: bool = False
RB_SKIPPED_RATIO_PENALTY: float = 3500.0
RB_EXECUTED_RATIO_PENALTY: float = 2500.0
RB_MAX_SIMULTANEOUS_POSITIONS: int = 10
RB_MAX_POSITIONS_PENALTY: float = 120.0


# --- Profit amplifier (post-risk-optimization refinement stage) ---

# RB_PROFIT_AMPLIFIER_ENABLED — when True, after risk grid search a final
#   refinement pass swaps/adds rules from the candidate pool and reallocates
#   capital to maximize a blended valid-return objective.  Keep enabled for
#   maximum performance; disable for faster but slightly weaker results.
RB_PROFIT_AMPLIFIER_ENABLED: bool = False

RB_PROFIT_AMP_MAX_CANDIDATES: int = 50
# RB_PROFIT_AMP_MIN_OBJECTIVE_IMPROVEMENT — minimum objective delta to
#   accept a profit-amplifier candidate.  Tuned for thin per-symbol returns.
RB_PROFIT_AMP_MIN_OBJECTIVE_IMPROVEMENT: float = 0.02
RB_PROFIT_AMP_MIN_RETURN_IMPROVEMENT: float = 0.005
RB_PROFIT_AMP_VALID_WEIGHT: float = 1.60
RB_PROFIT_AMP_TRAIN_WEIGHT: float = 1.00
RB_PROFIT_AMP_BALANCE_WEIGHT: float = 0.30
RB_PROFIT_AMP_DD_WEIGHT: float = 0.04
RB_PROFIT_AMP_HEALTH_WEIGHT: float = 0.06
RB_PROFIT_AMP_OVERLAP_PENALTY: float = 2.0
RB_PROFIT_AMP_MAX_PAIR_OVERLAP: float = 0.45
RB_PROFIT_AMP_MAX_VALID_DD: float = 20.0
RB_PROFIT_AMP_MAX_TRAIN_DD: float = 30.0
RB_PROFIT_AMP_MONTHLY_ENABLED: bool = True
RB_PROFIT_AMP_MIN_MONTHLY_WINDOWS: int = 2
RB_PROFIT_AMP_MIN_MONTHLY_PROFITABLE_RATIO: float = 0.50
RB_PROFIT_AMP_WORST_MONTHLY_RETURN_FLOOR: float = -1.5
RB_PROFIT_AMP_WORST_MONTHLY_PF_FLOOR: float = 0.75
RB_PROFIT_AMP_MAX_MONTHLY_DD: float = 15.0
RB_PROFIT_AMP_CAPITAL_REALLOCATION_ENABLED: bool = True
RB_PROFIT_AMP_CAPITAL_PASSES: int = 1
RB_PROFIT_AMP_KEEP_BASELINE_UNLESS_BETTER: bool = True


# =============================================================================
# Helpers & resolvers
# =============================================================================
# Pure functions and dataclasses. Constants live in the sections above; these
# resolve paths, scale trade floors, and adapt gates to debug symbol scope.


def filter_df_to_symbols(df: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """Return rows for the given symbols; raises if column missing or no rows."""
    if "symbol" not in df.columns:
        raise ValueError(
            "DataFrame must contain a 'symbol' column for symbol scope")
    sym_set = {str(s) for s in symbols}
    scoped = df[df["symbol"].astype(str).isin(sym_set)]
    if scoped.empty:
        raise ValueError(f"No rows for symbols {list(symbols)!r}")
    return scoped.reset_index(drop=True)


def resolve_debug_symbols(train_df) -> list[str] | None:
    """Return N debug symbols starting at DEBUG_SYMBOL when scope is enabled."""
    if not DEBUG_SYMBOL_SCOPE_ENABLED:
        return None
    count = int(DEBUG_SYMBOL_COUNT)
    if count < 1:
        raise ValueError(
            "DEBUG_SYMBOL_SCOPE_ENABLED is True but DEBUG_SYMBOL_COUNT < 1"
        )
    start_symbol = str(DEBUG_SYMBOL).strip()
    if not start_symbol:
        raise ValueError(
            "DEBUG_SYMBOL_SCOPE_ENABLED is True but DEBUG_SYMBOL is empty"
        )
    if train_df is None or getattr(train_df, "empty", True):
        raise ValueError(
            "DEBUG_SYMBOL_SCOPE_ENABLED but train data is empty"
        )
    if "symbol" not in train_df.columns:
        raise ValueError(
            "DEBUG_SYMBOL_SCOPE_ENABLED but train data has no 'symbol' column"
        )
    available = sorted(
        train_df["symbol"].dropna().astype(str).unique().tolist())
    if not available:
        raise ValueError(
            "DEBUG_SYMBOL_SCOPE_ENABLED but train data has no symbols"
        )
    if start_symbol not in available:
        raise ValueError(
            f"DEBUG_SYMBOL {start_symbol!r} not in train data; "
            f"available symbols: {available}"
        )
    start_idx = available.index(start_symbol)
    return available[start_idx:start_idx + count]


def _debug_symbol_universe_size() -> int | None:
    """Active symbol count when debug scope is on; None for full-universe runs."""
    if not DEBUG_SYMBOL_SCOPE_ENABLED:
        return None
    return max(1, int(DEBUG_SYMBOL_COUNT))


def effective_min_profitable_symbols(symbol_count: int | None = None) -> int:
    """Cap cross-symbol profitability gate to the active universe size.

    With DEBUG_SYMBOL_COUNT=2 and PHASE2_MIN_PROFITABLE_SYMBOLS=5, evolution
    always pays a shortfall penalty (impossible target) and long pools collapse.
    """
    target = int(PHASE2_MIN_PROFITABLE_SYMBOLS)
    universe = symbol_count if symbol_count is not None else _debug_symbol_universe_size()
    if universe is None:
        return target
    return min(target, max(1, int(universe)))


def effective_rb_min_distinct_symbols(symbol_count: int | None = None) -> int:
    """Return the RB coverage target for the active debug universe.

    Full runs keep ``RB_MIN_DISTINCT_SYMBOLS`` unchanged. An explicit debug
    scope may contain fewer symbols, so RB composition and its specialist
    output gate use the achievable scoped target instead of failing before a
    diagnostic result can be produced.
    """
    target = int(RB_MIN_DISTINCT_SYMBOLS)
    if not DEBUG_SYMBOL_SCOPE_ENABLED:
        return target
    universe = symbol_count if symbol_count is not None else _debug_symbol_universe_size()
    if universe is None:
        return target
    return min(target, max(1, int(universe)))


def effective_phase2_val_return_floor_pct(direction: str | None = None) -> float:
    """Direction-aware Phase 2 validation return floor for fitness penalties."""
    if direction == "short":
        return float(PHASE2_VAL_RETURN_FLOOR_PCT_SHORT)
    return float(PHASE2_VAL_RETURN_FLOOR_PCT)


def phase2_pool_path(
    direction: str,
    outputs_dir: str | None = None,
) -> str:
    """Resolve Phase 2 pool path."""
    if outputs_dir is None:
        return PHASE2_POOL_PATHS[direction]
    return os.path.join(outputs_dir, f"phase2_{direction}_pool.json")


def phase2_history_path(
    direction: str,
    outputs_dir: str | None = None,
) -> str:
    """Resolve Phase 2 history path."""
    if outputs_dir is None:
        return PHASE2_HISTORY_PATHS[direction]
    return os.path.join(outputs_dir, f"phase2_{direction}_history.json")


def phase2_should_enrich_symbol_metrics(
    engine: object | None = None,
    generation: int | None = None,
    is_last_gen: bool = False,
) -> bool:
    """Return True when GPU batch eval should run a follow-up CPU enrichment pass.

    When *generation* is provided, enrichment is throttled to every N generations
    (``PHASE2_ENRICH_SYMBOL_METRICS_EVERY_N_GENS``) plus always on the final
    generation.  This avoids paying the expensive CPU full re-simulation cost on
    every generation while keeping the symbol-spread penalty signal fresh enough.
    """
    if not PHASE2_GPU_ENRICH_SYMBOL_METRICS:
        return False
    if generation is not None:
        interval = max(1, int(PHASE2_ENRICH_SYMBOL_METRICS_EVERY_N_GENS))
        if is_last_gen or (generation % interval == 0):
            return True
        return False
    return True


def phase2_shared_archive_path(direction: str) -> str:
    """Shared Phase 2 archive path (warm-start across runs)."""
    return os.path.join(PHASE2_ARCHIVE_DIR, direction, "shared_archive.json")


def split_mode_is_purged_walk_forward() -> bool:
    """True when the active split mode is purged walk-forward."""
    return str(SPLIT_MODE).strip().lower() == "purged_walk_forward"



def set_purged_wf_reference_rows(n_rows: int) -> None:
    """Store full train_new.csv row count after loader prep (split time)."""
    global _PURGED_WF_REFERENCE_ROWS
    _PURGED_WF_REFERENCE_ROWS = max(0, int(n_rows))



def scale_trade_floor(
    base: int,
    n_rows: int,
    reference_rows: int | None = None,
) -> int:
    """Scale an integer trade floor by slice size vs reference universe."""
    if not split_mode_is_purged_walk_forward():
        return int(base)
    if not PURGED_WF_SCALE_TRADE_FLOORS:
        return int(base)
    ref = reference_rows if reference_rows is not None else _PURGED_WF_REFERENCE_ROWS
    if ref is None or ref <= 0:
        _logger.warning(
            "scale_trade_floor: purged walk-forward active but reference_rows "
            "unset; using unscaled base=%s (call set_purged_wf_reference_rows)",
            base,
        )
        return int(base)
    scaled = int(round(int(base) * int(n_rows) / int(ref)))
    return max(int(PURGED_WF_MIN_TRADE_FLOOR_ABSOLUTE), scaled)



def effective_min_trade_support(n_rows: int | None = None) -> int:
    base = int(MIN_TRADE_SUPPORT)
    if n_rows is None:
        return base
    return scale_trade_floor(base, n_rows)



def effective_min_trade_pool_floor(n_rows: int | None = None) -> int:
    base = int(MIN_TRADE_POOL_FLOOR)
    if n_rows is None:
        return base
    return scale_trade_floor(base, n_rows)



def effective_val_trade_floor_for_objectives(n_rows: int | None = None) -> int:
    base = max(int(MIN_TRADE_POOL_FLOOR) // 4, 10)
    if n_rows is None:
        return base
    return scale_trade_floor(base, n_rows)



def effective_pool_min_val_trades(n_rows: int | None = None) -> int:
    base = max(int(MIN_TRADE_POOL_FLOOR) // 4, 10)
    if n_rows is None:
        return base
    return scale_trade_floor(base, n_rows)



def effective_monthly_min_trades(n_rows: int | None = None) -> int:
    base = int(MONTHLY_MIN_TRADES)
    if n_rows is None:
        return base
    return scale_trade_floor(base, n_rows)



def scale_trade_floor_by_universe(
    base: int,
    n_rows: int,
    reference_rows: int,
    *,
    absolute_min: int | None = None,
) -> int:
    """Scale integer trade floors by slice size vs full-universe reference."""
    ref = int(reference_rows)
    if ref <= 0:
        return int(base)
    floor_min = int(
        absolute_min if absolute_min is not None else 8
    )
    scaled = int(round(int(base) * int(n_rows) / ref))
    return max(floor_min, scaled)

@dataclass(frozen=True)
class IslandHyperparams:
    """Optional Phase 2 floor overrides (tests and diagnostics)."""

    min_trade_support: int
    min_trade_pool_floor: int
    sortino_min_trade_threshold: int
    val_trade_floor: int
    min_profitable_symbols: int
    monthly_admission_min_months: int
    monthly_admission_min_profitable_ratio: float
    n_rows: int
    n_symbols: int


# =============================================================================
# Cross-parameter sanity is implemented by validate_config() below. It is
# callable after runtime/Optuna overrides and raises ConfigError with the
# violated relationship instead of relying on opaque import-time assertions.
# Runtime — Colab GPU defaults
# =============================================================================


def is_colab_runtime() -> bool:
    """True when running on Google Colab (/content runtime)."""
    return (
        os.environ.get("COLAB_RELEASE_TAG") is not None
        or os.path.isdir("/content")
    )


def is_t4_runtime() -> bool:
    """True when running on an NVIDIA Tesla T4 GPU or explicit T4 env override."""
    try:
        from gpu_fuzzy_trader._gpu_runtime import is_t4_runtime as _is_t4
        return _is_t4()
    except Exception:
        env_t4 = os.environ.get("GPU_OPT_T4", "").strip().lower()
        return env_t4 in ("1", "true", "yes")


def _apply_hardware_gpu_defaults() -> None:
    """
    Apply hardware-aware GPU defaults for Colab and generic T4 runtimes.

    T4 / Colab path:
    - PHASE2_GPU_BATCH_SIZE_AUTO = True
    - PHASE2_GPU_CPU_ROUTE_LARGE_DATA = False (GPU-first ranking path)
    - PHASE2_SCAN_UNROLL = min(PHASE2_SCAN_UNROLL, 16) (bounds XLA compile memory)

    Gated by GPU_OPT_DISABLE=1 escape hatch.
    """
    global PHASE2_GPU_BATCH_SIZE_AUTO
    global PHASE2_GPU_CPU_ROUTE_LARGE_DATA, PHASE2_SCAN_UNROLL

    disable_opt = os.environ.get("GPU_OPT_DISABLE", "").strip().lower()
    if disable_opt in ("1", "true", "yes"):
        return

    if not (is_colab_runtime() or is_t4_runtime()):
        return

    # Colab / generic T4 has enough VRAM for the JAX ranking path, while the local
    # RTX 4050 policy intentionally routes the default long window to CPU.
    PHASE2_GPU_BATCH_SIZE_AUTO = True
    PHASE2_GPU_CPU_ROUTE_LARGE_DATA = False

    # Keep XLA's host-side compilation footprint bounded on T4 / Colab
    # runtimes. Respect an even smaller value supplied before config import.
    PHASE2_SCAN_UNROLL = min(int(PHASE2_SCAN_UNROLL), 16)


# Backwards compatibility wrappers
_apply_colab_gpu_defaults = _apply_hardware_gpu_defaults
_apply_t4_gpu_defaults = _apply_hardware_gpu_defaults

_apply_hardware_gpu_defaults()


# =============================================================================
# Configuration validation and audit snapshots
# =============================================================================


class ConfigError(ValueError):
    """Raised when a configuration violates a cross-parameter contract."""


def _config_check(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _finite_config_number(name: str, value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be numeric, got {value!r}") from exc
    if not pd.notna(result) or result in (float("inf"), float("-inf")):
        raise ConfigError(f"{name} must be finite, got {value!r}")
    return result


def _validate_config_grid(name: str, values: object, *, minimum: float = 0.0) -> tuple[float, ...]:
    if not isinstance(values, (tuple, list)) or not values:
        raise ConfigError(f"{name} must be a non-empty tuple/list")
    parsed = tuple(_finite_config_number(name, value) for value in values)
    _config_check(
        all(value >= minimum for value in parsed),
        f"{name} values must be >= {minimum}, got {parsed!r}",
    )
    _config_check(
        tuple(sorted(set(parsed))) == parsed,
        f"{name} must be sorted and duplicate-free, got {parsed!r}",
    )
    return parsed


def validate_config(
    *,
    n_rows: int | None = None,
    n_symbols: int | None = None,
) -> None:
    """Validate all high-impact hyperparameter relationships.

    The function is intentionally callable after Optuna temporarily patches
    module globals.  It performs no data loading and raises ``ConfigError``
    before a long evolution/backtest can start with an incoherent contract.
    """

    _config_check(str(SPLIT_MODE).lower() in {"holdout", "purged_walk_forward"},
                  f"Unsupported SPLIT_MODE={SPLIT_MODE!r}")
    _config_check(int(PHASE2_MIN_PROFITABLE_SYMBOLS) >= 1,
                  "PHASE2_MIN_PROFITABLE_SYMBOLS must be positive")
    _config_check(int(DEBUG_SYMBOL_COUNT) >= 1,
                  "DEBUG_SYMBOL_COUNT must be positive")
    _config_check(bool(str(DEBUG_SYMBOL).strip()),
                  "DEBUG_SYMBOL must not be empty")
    _config_check(0.0 < float(HOLDOUT_TRAIN_FRACTION) < 1.0,
                  "HOLDOUT_TRAIN_FRACTION must be between 0 and 1")
    _config_check(int(TAIL_DROP_ROWS) == int(MAX_HOLD_CANDLES),
                  "TAIL_DROP_ROWS must equal MAX_HOLD_CANDLES")
    _config_check(int(HOLDOUT_EMBARGO_CANDLES) == int(MAX_HOLD_CANDLES),
                  "HOLDOUT_EMBARGO_CANDLES must equal MAX_HOLD_CANDLES")
    _config_check(int(VALIDATION_HALF_PURGE_CANDLES) >= int(MAX_HOLD_CANDLES),
                  "VALIDATION_HALF_PURGE_CANDLES must cover MAX_HOLD_CANDLES")
    # The 96-bar horizon contract must stay coherent across every linked
    # embargo/tail knob (24 h at 15-minute bars).
    _config_check(
        int(TAIL_DROP_ROWS) == int(HOLDOUT_EMBARGO_CANDLES)
        == int(PURGED_WF_EMBARGO_CANDLES) == int(MAX_HOLD_CANDLES),
        "TAIL_DROP_ROWS, HOLDOUT_EMBARGO_CANDLES, PURGED_WF_EMBARGO_CANDLES, "
        "and MAX_HOLD_CANDLES must all be equal",
    )
    _config_check(
        int(VALIDATION_HALF_PURGE_CANDLES) == int(MAX_HOLD_CANDLES),
        "VALIDATION_HALF_PURGE_CANDLES must equal MAX_HOLD_CANDLES",
    )
    # Trend-context contract coherence.
    _config_check(
        int(CONTEXT_BAR_SECONDS) == 15 * 60,
        "CONTEXT_BAR_SECONDS must be 900 (15-minute bar-open timestamps)",
    )
    _config_check(
        int(LWC_TIMEFRAME_MINUTES) * 4 == int(MWC_TIMEFRAME_MINUTES)
        and int(MWC_TIMEFRAME_MINUTES) * 4 == int(HWC_TIMEFRAME_MINUTES),
        "LWC/MWC/HWC must be 15m/1h/4h (×4 hierarchy)",
    )
    _config_check(
        1 <= int(LWC_PULLBACK_LOOKBACK) <= int(MAX_HOLD_CANDLES),
        "LWC_PULLBACK_LOOKBACK must be in [1, MAX_HOLD_CANDLES] (frozen at 24)",
    )
    _config_check(
        int(LWC_PULLBACK_LOOKBACK) == 24,
        "LWC_PULLBACK_LOOKBACK is frozen at 24 per the wave-cycle contract; "
        "bump CONTEXT_ALGORITHM_VERSION and re-enrich all tapes to change it",
    )
    _config_check(
        int(CONTEXT_STRUCTURAL_LOOKBACK) >= 1,
        "CONTEXT_STRUCTURAL_LOOKBACK must be positive",
    )
    _config_check(
        0.0 < float(CONTEXT_EFFICIENCY_TREND_THRESHOLD_QUANTILE) < 1.0
        and 0.0 < float(CONTEXT_EMA_SPREAD_TREND_THRESHOLD_QUANTILE) < 1.0,
        "trend threshold quantiles must be in (0, 1)",
    )
    _config_check(
        0.0 < float(CONTEXT_VOLATILITY_COMPRESSION_QUANTILE) < 1.0,
        "CONTEXT_VOLATILITY_COMPRESSION_QUANTILE must be in (0, 1)",
    )
    for direction in ("long", "short"):
        _config_check(
            len(CONTEXT_MANDATORY_CONDITIONS.get(direction, ())) == 2,
            f"direction {direction!r} must have exactly 2 mandatory context "
            "conditions (permission + LWC trigger)",
        )
    _config_check(int(PHASE2_MONTHLY_MIN_TRADES) >= 1,
                  "PHASE2_MONTHLY_MIN_TRADES must be positive")
    _config_check(
        0.0 <= float(PHASE2_MONTHLY_MIN_ACTIVE_RATIO) <= 1.0,
        "PHASE2_MONTHLY_MIN_ACTIVE_RATIO must be in [0, 1]",
    )
    _config_check(
        0.0 <= float(PHASE2_MONTHLY_MAX_BEARISH_RATIO) <= 1.0,
        "PHASE2_MONTHLY_MAX_BEARISH_RATIO must be in [0, 1]",
    )
    fee_pct = _finite_config_number("FEE_PCT", FEE_PCT)
    spread_bps = _finite_config_number("SPREAD_BPS", SPREAD_BPS)
    slippage_bps = _finite_config_number("SLIPPAGE_BPS", SLIPPAGE_BPS)
    initial_capital = _finite_config_number("INITIAL_CAPITAL", INITIAL_CAPITAL)
    leverage = _finite_config_number("LEVERAGE", LEVERAGE)
    max_exposure = _finite_config_number(
        "MAX_TOTAL_EXPOSURE_PCT", MAX_TOTAL_EXPOSURE_PCT
    )
    min_notional = _finite_config_number(
        "MIN_POSITION_NOTIONAL", MIN_POSITION_NOTIONAL
    )
    _config_check(fee_pct >= 0.0, "FEE_PCT must be non-negative")
    _config_check(spread_bps >= 0.0, "SPREAD_BPS must be non-negative")
    _config_check(slippage_bps >= 0.0, "SLIPPAGE_BPS must be non-negative")
    _config_check(initial_capital > 0.0, "INITIAL_CAPITAL must be positive")
    _config_check(leverage > 0.0, "LEVERAGE must be positive")
    _config_check(max_exposure > 0.0,
                  "MAX_TOTAL_EXPOSURE_PCT must be positive")
    _config_check(min_notional > 0.0,
                  "MIN_POSITION_NOTIONAL must be positive")

    _config_check(int(PHASE1_TOP_K_FEATURES) >= int(MAX_CONDITIONS),
                  "PHASE1_TOP_K_FEATURES must cover MAX_CONDITIONS")
    _config_check(0.0 < float(PHASE1_MAX_FEATURE_OVERLAP) <= 1.0,
                  "PHASE1_MAX_FEATURE_OVERLAP must be in (0, 1]")
    _config_check(1 <= int(PHASE1_SIGN_CONSISTENCY_MIN_FOLDS)
                  <= int(PHASE1_STATIONARITY_FOLDS),
                  "Phase 1 sign-consistency folds must fit stationarity folds")
    _config_check(int(PHASE1_SAMPLING_TOTAL) > 0,
                  "PHASE1_SAMPLING_TOTAL must be positive")
    _config_check(int(PHASE2_VAL_SIM_INTERVAL) >= 1,
                  "PHASE2_VAL_SIM_INTERVAL must be positive")

    _config_check(int(MIN_CONDITIONS) <= int(MAX_CONDITIONS),
                  "MIN_CONDITIONS must be <= MAX_CONDITIONS")
    _config_check(int(PHASE2_POPULATION_SIZE) >= 2,
                  "PHASE2_POPULATION_SIZE must be at least 2")
    _config_check(int(PHASE2_GENERATIONS) >= 2,
                  "PHASE2_GENERATIONS must be at least 2")
    _config_check(int(PHASE2_ARCHIVE_MAX_SIZE) >= int(PHASE2_POPULATION_SIZE),
                  "PHASE2_ARCHIVE_MAX_SIZE must be >= PHASE2_POPULATION_SIZE")
    _config_check(int(PHASE2_STAGE_A_GENERATIONS) > 0
                  and int(PHASE2_STAGE_B_GENERATIONS) > 0,
                  "Phase 2 stage generation budgets must be positive")
    _config_check(
        int(PHASE2_STAGE_A_GENERATIONS) + int(PHASE2_STAGE_B_GENERATIONS)
        == int(PHASE2_GENERATIONS),
        "PHASE2_STAGE_A_GENERATIONS + PHASE2_STAGE_B_GENERATIONS must equal PHASE2_GENERATIONS",
    )
    _config_check(
        float(PHASE2_STAGE_A_MUTATION_RATE) > 0.0
        and float(PHASE2_STAGE_B_MUTATION_RATE) > 0.0,
        "Phase 2 mutation rates must be positive",
    )
    _config_check(
        0.0 < float(PHASE2_STAGE_A_MUTATION_RATE) <= 0.5
        and 0.0 < float(PHASE2_STAGE_B_MUTATION_RATE) <= 0.5,
        "Phase 2 mutation rates must be in (0, 0.5]",
    )
    _config_check(
        float(PHASE2_STAGE_A_MUTATION_RATE)
        >= float(PHASE2_STAGE_B_MUTATION_RATE),
        "Stage A mutation must be >= Stage B mutation",
    )
    _config_check(
        0.0 < float(PHASE2_MUTATION_RATE) <= 0.5,
        "PHASE2_MUTATION_RATE must be in (0, 0.5]",
    )
    _config_check(
        0 <= int(PHASE2_STAGE_B_SEED_TOP_K) <= int(PHASE2_POPULATION_SIZE),
        "PHASE2_STAGE_B_SEED_TOP_K must fit the population size",
    )
    _config_check(
        0.0 <= float(PHASE2_STAGE_B_SEED_FRACTION) <= 1.0,
        "PHASE2_STAGE_B_SEED_FRACTION must be in [0, 1]",
    )
    _config_check(
        0.0 <= float(PHASE2_ARCHIVE_SEED_FRACTION) <= 1.0
        and 0.0 <= float(PHASE2_STAGE_A_ARCHIVE_SEED_FRACTION) <= 1.0,
        "Phase 2 archive seed fractions must be in [0, 1]",
    )
    _config_check(
        len(PHASE2_INIT_STRATUM_FRACTIONS) == 2
        and all(0.0 <= float(value) <= 1.0 for value in PHASE2_INIT_STRATUM_FRACTIONS)
        and abs(sum(float(value) for value in PHASE2_INIT_STRATUM_FRACTIONS) - 1.0) <= 1e-9,
        "PHASE2_INIT_STRATUM_FRACTIONS must contain two values summing to 1",
    )
    for name, value, budget in (
        ("PHASE2_STAGE_A_PLATEAU_EARLY_STOP_MIN_GENERATION",
         PHASE2_STAGE_A_PLATEAU_EARLY_STOP_MIN_GENERATION,
         PHASE2_STAGE_A_GENERATIONS),
        ("PHASE2_STAGE_A_EARLY_STOP_MIN_GENERATION",
         PHASE2_STAGE_A_EARLY_STOP_MIN_GENERATION,
         PHASE2_STAGE_A_GENERATIONS),
        ("PHASE2_STAGE_B_PLATEAU_EARLY_STOP_MIN_GENERATION",
         PHASE2_STAGE_B_PLATEAU_EARLY_STOP_MIN_GENERATION,
         PHASE2_STAGE_B_GENERATIONS),
        ("PHASE2_STAGE_B_EARLY_STOP_MIN_GENERATION",
         PHASE2_STAGE_B_EARLY_STOP_MIN_GENERATION,
         PHASE2_STAGE_B_GENERATIONS),
    ):
        _config_check(0 <= int(value) <= int(budget),
                      f"{name} must fit its stage budget")
    _config_check(0.0 < float(PHASE2_SAMPLE_ROTATION_FRACTION) <= 1.0,
                  "PHASE2_SAMPLE_ROTATION_FRACTION must be in (0, 1]")
    _config_check(
        0.0 <= float(PHASE2_SYMBOL_GENE_DONT_CARE_PROB) <= 1.0,
        "PHASE2_SYMBOL_GENE_DONT_CARE_PROB must be in [0, 1]",
    )
    _config_check(int(PHASE2_SAMPLE_MAX_BARS_PER_SYMBOL) > 0,
                  "PHASE2_SAMPLE_MAX_BARS_PER_SYMBOL must be positive")
    _config_check(
        int(PHASE2_N_OBJECTIVES) == (4 if bool(PHASE2_F4_ENABLED) else 3),
        "PHASE2_N_OBJECTIVES must match PHASE2_F4_ENABLED",
    )
    _config_check(
        str(PHASE2_F3_OBJECTIVE) in {"profit_factor", "cv_fold_min", "win_rate"},
        "PHASE2_F3_OBJECTIVE must be profit_factor, cv_fold_min, or win_rate",
    )
    _config_check(
        0.0 <= float(PHASE2_F4_CONCENTRATION_FLOOR) <= 1.0,
        "PHASE2_F4_CONCENTRATION_FLOOR must be in [0, 1]",
    )
    _config_check(float(PHASE2_F4_EPSILON) > 0.0,
                  "PHASE2_F4_EPSILON must be positive")
    _config_check(float(PHASE2_EXPECTANCY_LCB_Z) > 0.0,
                  "PHASE2_EXPECTANCY_LCB_Z must be positive")
    _config_check(float(PHASE2_EXPECTANCY_LCB_WEIGHT) >= 0.0,
                  "PHASE2_EXPECTANCY_LCB_WEIGHT must be non-negative")
    _config_check(int(MIN_TRADE_POOL_FLOOR) <= int(MIN_TRADE_SUPPORT),
                  "MIN_TRADE_POOL_FLOOR must be <= MIN_TRADE_SUPPORT")
    _config_check(int(MIN_TRADE_SUPPORT) > 0,
                  "MIN_TRADE_SUPPORT must be positive")
    _config_check(
        int(PHASE2_MONTHLY_ADMISSION_MIN_MONTHS) >= 2,
        "PHASE2_MONTHLY_ADMISSION_MIN_MONTHS must support the two-window holdout",
    )
    _config_check(int(MONTHLY_WINDOW_MIN_ROWS) > 0,
                  "MONTHLY_WINDOW_MIN_ROWS must be positive")
    _config_check(int(MONTHLY_WINDOW_MAX_WINDOWS) >= 2,
                  "MONTHLY_WINDOW_MAX_WINDOWS must support two windows")
    _config_check(0.0 <= float(PHASE2_MONTHLY_ADMISSION_MIN_RATIO) <= 1.0,
                  "PHASE2_MONTHLY_ADMISSION_MIN_RATIO must be in [0, 1]")
    _config_check(
        float(PHASE2_PROFIT_FACTOR_FLOOR_EVOLUTION) >= 1.0
        and float(PHASE2_PROFIT_FACTOR_FLOOR_ADMISSION) >= 1.0
        and float(PHASE2_PROFIT_FACTOR_FLOOR_ADMISSION)
        >= float(PHASE2_PROFIT_FACTOR_FLOOR_EVOLUTION),
        "Phase 2 profit-factor floors must be ordered at or above 1.0",
    )
    _config_check(
        float(PHASE2_RETURN_FLOOR_PCT) >= 0.0
        and float(PHASE2_VAL_RETURN_FLOOR_PCT) >= 0.0
        and float(PHASE2_VAL_RETURN_FLOOR_PCT_SHORT) >= 0.0,
        "Phase 2 return floors must be non-negative",
    )
    _config_check(float(PHASE2_MAX_DRAWDOWN_GATE) > 0.0,
                  "PHASE2_MAX_DRAWDOWN_GATE must be positive")
    _config_check(float(PHASE2_MAX_TRAIN_VAL_GAP_PCT) >= 0.0,
                  "PHASE2_MAX_TRAIN_VAL_GAP_PCT must be non-negative")

    tp_grid = _validate_config_grid("RB_TP_GRID", RB_TP_GRID, minimum=0.0)
    sl_grid = _validate_config_grid("RB_SL_GRID", RB_SL_GRID, minimum=0.0)
    capital_grid = _validate_config_grid("RB_CAPITAL_GRID", RB_CAPITAL_GRID, minimum=0.0)
    _config_check(min(capital_grid) > 0.0,
                  "RB_CAPITAL_GRID must contain only positive capital values")
    _config_check(int(RB_MAX_RULES) >= 1,
                  "RB_MAX_RULES must be positive")
    _config_check(int(RB_MIN_RULES) >= 0,
                  "RB_MIN_RULES must be non-negative")
    _config_check(int(RB_MIN_RULES) <= int(RB_MAX_RULES),
                  "RB_MIN_RULES must be <= RB_MAX_RULES")
    _config_check(int(RB_KEEP_TOP_RULES) >= 1,
                  "RB_KEEP_TOP_RULES must be positive")
    _config_check(int(RB_MAX_POOL_RULES_TO_EVALUATE) >= 1,
                  "RB_MAX_POOL_RULES_TO_EVALUATE must be positive")
    _config_check(int(RB_MAX_POOL_RULES_TO_EVALUATE) >= int(RB_KEEP_TOP_RULES),
                  "RB_MAX_POOL_RULES_TO_EVALUATE must be >= RB_KEEP_TOP_RULES")
    _config_check(int(RB_MIN_TRAIN_TRADES) <= int(RB_RULESET_MIN_TRAIN_TRADES),
                  "RB per-rule train floor must be <= team train floor")
    _config_check(int(RB_MIN_VALID_TRADES) <= int(RB_RULESET_MIN_VALID_TRADES),
                  "RB per-rule validation floor must be <= team validation floor")
    _config_check(
        float(RB_MIN_TRAIN_RETURN) >= 0.0
        and float(RB_MIN_VALID_RETURN) >= 0.0,
        "RB per-rule return floors must be non-negative",
    )
    _config_check(
        float(RB_MIN_TRAIN_PF) >= 1.0
        and float(RB_MIN_VALID_PF) >= 1.0,
        "RB per-rule profit-factor floors must be >= 1.0",
    )
    _config_check(
        int(RB_MIN_TRAIN_TRADES) >= 0
        and int(RB_MIN_VALID_TRADES) >= 0
        and int(RB_RULESET_MIN_TRAIN_TRADES) >= 0
        and int(RB_RULESET_MIN_VALID_TRADES) >= 0,
        "RB trade floors must be non-negative",
    )
    _config_check(float(RB_MAX_TOTAL_CAPITAL) == float(MAX_TOTAL_EXPOSURE_PCT),
                  "RB_MAX_TOTAL_CAPITAL must equal MAX_TOTAL_EXPOSURE_PCT")
    _config_check(
        max(capital_grid) <= float(RB_MAX_TOTAL_CAPITAL) + 1e-9,
        "RB_CAPITAL_GRID values must fit RB_MAX_TOTAL_CAPITAL",
    )
    _config_check(
        int(RB_MAX_RULES) * float(min(capital_grid))
        <= float(RB_MAX_TOTAL_CAPITAL) + 1e-9,
        "RB_MAX_RULES multiplied by minimum RB capital must fit RB_MAX_TOTAL_CAPITAL",
    )
    _config_check(float(RB_DEFAULT_TP) in tp_grid,
                  "RB_DEFAULT_TP must be present in RB_TP_GRID")
    _config_check(float(RB_DEFAULT_SL) in sl_grid,
                  "RB_DEFAULT_SL must be present in RB_SL_GRID")
    _config_check(float(RB_DEFAULT_CAPITAL_PCT) in capital_grid,
                  "RB_DEFAULT_CAPITAL_PCT must be present in RB_CAPITAL_GRID")
    _config_check(float(RB_MIN_TP) >= 0.0 and float(RB_MIN_TP) <= min(tp_grid),
                  "RB_MIN_TP must not exceed the smallest TP grid value")
    _config_check(float(RB_MIN_SL) >= 0.0 and float(RB_MIN_SL) <= min(sl_grid),
                  "RB_MIN_SL must not exceed the smallest SL grid value")
    _config_check(int(RB_RISK_OPT_PASSES) >= 1,
                  "RB_RISK_OPT_PASSES must be positive")
    _config_check(float(RB_RISK_MIN_IMPROVEMENT) >= 0.0,
                  "RB_RISK_MIN_IMPROVEMENT must be non-negative")
    _config_check(int(RB_RISK_GRID_WF_SPLITS) >= 1,
                  "RB_RISK_GRID_WF_SPLITS must be >= 1")
    _config_check(float(RB_EXPECTANCY_LCB_MARGIN_PCT) >= -100.0,
                  "RB_EXPECTANCY_LCB_MARGIN_PCT is invalid")
    _config_check(
        all(float(value) >= 1.0 for value in RB_COST_STRESS_MULTIPLIERS),
        "RB_COST_STRESS_MULTIPLIERS must be >= 1",
    )
    _config_check(int(NESTED_VALIDATION_OUTER_FOLDS) >= 1,
                  "NESTED_VALIDATION_OUTER_FOLDS must be positive")
    _config_check(0.0 < float(RB_TAIL_HOLDOUT_FRACTION) < 1.0,
                  "RB_TAIL_HOLDOUT_FRACTION must be in (0, 1)")
    _config_check(float(RB_TAIL_HOLDOUT_MIN_RETURN_PCT) >= 0.0,
                  "RB_TAIL_HOLDOUT_MIN_RETURN_PCT must be non-negative")
    _config_check(int(RB_TAIL_HOLDOUT_MIN_TRADES) >= 0,
                  "RB_TAIL_HOLDOUT_MIN_TRADES must be non-negative")
    _config_check(0.0 <= float(RB_MAX_PAIR_OVERLAP) <= 1.0,
                  "RB_MAX_PAIR_OVERLAP must be in [0, 1]")
    _config_check(0.0 <= float(RB_MAX_SYMBOL_SHARE_ABS_PNL) <= 1.0,
                  "RB_MAX_SYMBOL_SHARE_ABS_PNL must be in [0, 1]")
    _config_check(0.0 <= float(RB_MAX_SYMBOL_HHI) <= 1.0,
                  "RB_MAX_SYMBOL_HHI must be in [0, 1]")
    _config_check(int(RB_MIN_DISTINCT_SYMBOLS) >= 1,
                  "RB_MIN_DISTINCT_SYMBOLS must be at least 1")
    _config_check(int(RB_PROFIT_AMP_CAPITAL_PASSES) >= 1,
                  "RB_PROFIT_AMP_CAPITAL_PASSES must be positive")
    _config_check(
        0.0 <= float(PHASE5_VALIDATION_RETURN_GATE_PCT)
        and float(PHASE5_VALIDATION_PROFIT_FACTOR_GATE) >= 1.0,
        "Phase 5 deployment gates must be non-negative and PF >= 1.0",
    )
    if n_symbols is not None:
        active_symbols = int(n_symbols)
        _config_check(active_symbols >= 1, "n_symbols must be positive")
        debug_scope = bool(DEBUG_SYMBOL_SCOPE_ENABLED)
        if debug_scope:
            effective_symbol_floor = effective_min_profitable_symbols(active_symbols)
            effective_rb_symbol_floor = effective_rb_min_distinct_symbols(active_symbols)
        else:
            _config_check(
                int(PHASE2_MIN_PROFITABLE_SYMBOLS) <= active_symbols,
                "PHASE2_MIN_PROFITABLE_SYMBOLS exceeds the active universe",
            )
            effective_symbol_floor = int(PHASE2_MIN_PROFITABLE_SYMBOLS)
            effective_rb_symbol_floor = int(RB_MIN_DISTINCT_SYMBOLS)
        _config_check(
            effective_symbol_floor <= active_symbols,
            "effective Phase 2 profitable-symbol floor exceeds the active universe",
        )
        _config_check(
            effective_rb_symbol_floor <= active_symbols,
            "effective RB_MIN_DISTINCT_SYMBOLS exceeds the active universe",
        )

    if n_rows is not None:
        _config_check(int(n_rows) > 0, "n_rows must be positive")
    estimated_rows = int(PHASE1_SAMPLING_TOTAL)
    if n_rows is not None:
        estimated_rows = min(estimated_rows, int(n_rows))
    if n_symbols is not None:
        estimated_rows = min(
            estimated_rows,
            int(PHASE2_SAMPLE_MAX_BARS_PER_SYMBOL) * int(n_symbols),
        )
    _config_check(int(MIN_TRADE_POOL_FLOOR) <= estimated_rows,
                  "MIN_TRADE_POOL_FLOOR exceeds the effective sample-row budget")
    _config_check(int(MIN_TRADE_SUPPORT) <= estimated_rows,
                  "MIN_TRADE_SUPPORT exceeds the effective sample-row budget")


def effective_config_snapshot(
    *,
    n_rows: int | None = None,
    n_symbols: int | None = None,
) -> dict[str, object]:
    """Return resolved values and derived constraints for audit/reporting."""

    validate_config(n_rows=n_rows, n_symbols=n_symbols)
    estimated_rows = int(PHASE1_SAMPLING_TOTAL)
    if n_rows is not None:
        estimated_rows = min(estimated_rows, int(n_rows))
    if n_symbols is not None:
        estimated_rows = min(
            estimated_rows,
            int(PHASE2_SAMPLE_MAX_BARS_PER_SYMBOL) * int(n_symbols),
        )
    active_symbol_count = int(n_symbols) if n_symbols is not None else None
    effective_rb_symbol_floor = effective_rb_min_distinct_symbols(active_symbol_count)
    min_capital = min(float(value) for value in RB_CAPITAL_GRID)
    return {
        "active_universe": {
            "n_rows": int(n_rows) if n_rows is not None else None,
            "n_symbols": active_symbol_count,
            "debug_scope_enabled": bool(DEBUG_SYMBOL_SCOPE_ENABLED),
            "debug_symbol": str(DEBUG_SYMBOL),
            "debug_symbol_count": int(DEBUG_SYMBOL_COUNT),
        },
        "evaluator_contract": {
            "fee_pct": float(FEE_PCT),
            "max_hold_candles": int(MAX_HOLD_CANDLES),
            "initial_capital": float(INITIAL_CAPITAL),
            "leverage": float(LEVERAGE),
            "max_total_exposure_pct": float(MAX_TOTAL_EXPOSURE_PCT),
            "min_position_notional": float(MIN_POSITION_NOTIONAL),
            "execution_policy": {
                "hardware_routing": {
                    "large_window_cpu_route": bool(PHASE2_GPU_CPU_ROUTE_LARGE_DATA),
                    "route_min_bars": int(PHASE2_GPU_CPU_ROUTE_MIN_BARS),
                    "route_max_batch": int(PHASE2_GPU_CPU_ROUTE_MAX_BATCH),
                },
                "numerics": {
                    "gpu_fp32": bool(PHASE2_GPU_USE_FP32),
                    "gpu_data_int8": bool(PHASE2_GPU_DATA_INT8),
                    "per_symbol_available": None,
                },
            },
        },
        "split": {
            "mode": str(SPLIT_MODE),
            "holdout_train_fraction": float(HOLDOUT_TRAIN_FRACTION),
            "embargo_candles": int(HOLDOUT_EMBARGO_CANDLES),
            "validation_half_purge_candles": int(VALIDATION_HALF_PURGE_CANDLES),
            "tail_drop_rows": int(TAIL_DROP_ROWS),
        },
        "context": context_contract(),
        "phase2": {
            "population_size": int(PHASE2_POPULATION_SIZE),
            "generations": int(PHASE2_GENERATIONS),
            "stage_a_generations": int(PHASE2_STAGE_A_GENERATIONS),
            "stage_b_generations": int(PHASE2_STAGE_B_GENERATIONS),
            "two_stage_enabled": bool(PHASE2_TWO_STAGE_ENABLED),
            "sampling_total": int(PHASE1_SAMPLING_TOTAL),
            "estimated_effective_rows": estimated_rows,
            "min_trade_support": int(MIN_TRADE_SUPPORT),
            "min_trade_pool_floor": int(MIN_TRADE_POOL_FLOOR),
            "joint_train_val": bool(PHASE2_JOINT_TRAIN_VAL),
            "val_in_fitness_penalty": bool(PHASE2_VAL_IN_FITNESS_PENALTY),
            "use_total_return_objective": bool(PHASE2_USE_TOTAL_RETURN_OBJ),
            "f3_objective": str(PHASE2_F3_OBJECTIVE),
            "effective_f3_objective": (
                "total_return"
                if bool(PHASE2_USE_TOTAL_RETURN_OBJ)
                else str(PHASE2_F3_OBJECTIVE)
            ),
            "effective_min_profitable_symbols": effective_min_profitable_symbols(n_symbols),
            "min_profitable_symbols_required": int(PHASE2_MIN_PROFITABLE_SYMBOLS),
            "symbol_gene_dont_care_prob": float(PHASE2_SYMBOL_GENE_DONT_CARE_PROB),
            "expectancy_lcb_z": float(PHASE2_EXPECTANCY_LCB_Z),
            "expected_shortfall_q": float(PHASE2_EXPECTED_SHORTFALL_Q),
            "phase1_disabled": bool(PHASE1_DISABLED),
            "monthly_min_trades": int(PHASE2_MONTHLY_MIN_TRADES),
            "monthly_min_active_ratio": float(PHASE2_MONTHLY_MIN_ACTIVE_RATIO),
            "monthly_max_bearish_ratio": float(PHASE2_MONTHLY_MAX_BEARISH_RATIO),
            "sample_max_bars_per_symbol": int(PHASE2_SAMPLE_MAX_BARS_PER_SYMBOL),
            "admission_min_val_trades": int(effective_pool_min_val_trades(n_rows)),
        },
        "rb": {
            "min_rules": int(RB_MIN_RULES),
            "max_rules": int(RB_MAX_RULES),
            "max_total_capital": float(RB_MAX_TOTAL_CAPITAL),
            "capital_grid": [float(value) for value in RB_CAPITAL_GRID],
            "max_feasible_rules_at_min_capital": int(
                float(RB_MAX_TOTAL_CAPITAL) // min_capital
            ),
            "risk_grid_wf_splits": int(RB_RISK_GRID_WF_SPLITS),
            "risk_optimize_exits": bool(RB_RISK_OPTIMIZE_EXITS),
            "candidate_risk_admission": bool(
                RB_CANDIDATE_RISK_ADMISSION_ENABLED
            ),
            "cost_stress_enabled": bool(RB_COST_STRESS_ENABLED),
            "cost_stress_multipliers": [
                float(value) for value in RB_COST_STRESS_MULTIPLIERS
            ],
            "tail_holdout_fraction": float(RB_TAIL_HOLDOUT_FRACTION),
            "tail_holdout_selection_gate": bool(RB_TAIL_HOLDOUT_SELECTION_GATE),
            "tail_holdout_min_trades": int(RB_TAIL_HOLDOUT_MIN_TRADES),
            "min_distinct_symbols": int(RB_MIN_DISTINCT_SYMBOLS),
            "effective_min_distinct_symbols": effective_rb_symbol_floor,
            "risk_min_improvement": float(RB_RISK_MIN_IMPROVEMENT),
            "tail_min_return_pct": float(RB_TAIL_HOLDOUT_MIN_RETURN_PCT),
            "max_pair_overlap": float(RB_MAX_PAIR_OVERLAP),
            "max_symbol_share_abs_pnl": float(RB_MAX_SYMBOL_SHARE_ABS_PNL),
            "max_symbol_hhi": float(RB_MAX_SYMBOL_HHI),
            "symbol_filters_required": bool(RB_REQUIRE_SYMBOL_FILTERS),
            "allow_partial_specialist_coverage": bool(
                RB_ALLOW_PARTIAL_SPECIALIST_COVERAGE
            ),
            "multi_symbol_release": bool(RB_MULTI_SYMBOL_RELEASE),
            "phase2_provenance_only": bool(
                globals().get("RB_PHASE2_PROVENANCE_ONLY", False)
            ),
            "strategy_identity_exit_immutable": not bool(
                RB_RISK_OPTIMIZE_EXITS
            ),
            "monthly_certificate": bool(RB_MONTHLY_CERTIFICATE_ENABLED),
            "monthly_min_profitable_ratio": float(
                RB_MONTHLY_MIN_PROFITABLE_RATIO
            ),
            "monthly_max_bearish_ratio": float(RB_MONTHLY_MAX_BEARISH_RATIO),
            "univariate_baseline_enabled": bool(
                RB_UNIVARIATE_BASELINE_ENABLED
            ),
            "full_validation_recovery": bool(RB_FULL_VALIDATION_RECOVERY_ENABLED),
            "univariate_generalist": bool(RB_UNIVARIATE_GENERALIST_ENABLED),
            "recency_rescue": bool(RB_RECENCY_RESCUE_ENABLED),
            "recency_min_validation_return": float(RB_RECENCY_MIN_VALID_RETURN),
            "recency_min_validation_pf": float(RB_RECENCY_MIN_VALID_PF),
            "recency_min_validation_trades": int(RB_RECENCY_MIN_VALID_TRADES),
            "recency_max_train_loss_pct": float(RB_RECENCY_MAX_TRAIN_LOSS_PCT),
            "recency_max_train_dd_pct": float(RB_RECENCY_MAX_TRAIN_DD_PCT),
            "recency_max_symbol_loss_pct": float(RB_RECENCY_MAX_SYMBOL_LOSS_PCT),
            "recency_max_symbol_share_abs_pnl": float(RB_RECENCY_MAX_SYMBOL_SHARE_ABS_PNL),
            "recency_max_symbol_hhi": float(RB_RECENCY_MAX_SYMBOL_HHI),
        },
        "gates": {
            "phase2_train_return_min": float(PHASE2_POOL_TRAIN_RETURN_MIN_PCT),
            "phase2_valid_return_min": float(PHASE2_POOL_VAL_RETURN_MIN_PCT),
            "phase2_valid_return_min_short": float(PHASE2_VAL_RETURN_FLOOR_PCT_SHORT),
            "phase2_evolution_pf_floor": float(PHASE2_PROFIT_FACTOR_FLOOR_EVOLUTION),
            "phase2_admission_pf_floor": float(PHASE2_PROFIT_FACTOR_FLOOR_ADMISSION),
            "phase2_max_drawdown_pct": float(PHASE2_MAX_DRAWDOWN_GATE),
            "phase2_max_train_valid_gap_pct": float(PHASE2_MAX_TRAIN_VAL_GAP_PCT),
            "phase2_monthly_min_months": int(PHASE2_MONTHLY_ADMISSION_MIN_MONTHS),
            "phase2_monthly_min_profitable_ratio": float(PHASE2_MONTHLY_ADMISSION_MIN_RATIO),
            "phase2_monthly_min_trades": int(PHASE2_MONTHLY_MIN_TRADES),
            "phase2_monthly_min_active_ratio": float(PHASE2_MONTHLY_MIN_ACTIVE_RATIO),
            "phase2_monthly_max_bearish_ratio": float(PHASE2_MONTHLY_MAX_BEARISH_RATIO),
            "phase2_monthly_fail_closed": bool(PHASE2_MONTHLY_ADMISSION_FAIL_CLOSED),
            "rb_min_train_return": float(RB_MIN_TRAIN_RETURN),
            "rb_min_valid_return": float(RB_MIN_VALID_RETURN),
            "rb_min_train_pf": float(RB_MIN_TRAIN_PF),
            "rb_min_valid_pf": float(RB_MIN_VALID_PF),
            "rb_min_train_trades": int(RB_MIN_TRAIN_TRADES),
            "rb_min_valid_trades": int(RB_MIN_VALID_TRADES),
            "rb_ruleset_min_train_trades": int(RB_RULESET_MIN_TRAIN_TRADES),
            "rb_ruleset_min_valid_trades": int(RB_RULESET_MIN_VALID_TRADES),
            "phase5_validation_return_gate": float(PHASE5_VALIDATION_RETURN_GATE_PCT),
            "phase5_validation_pf_gate": float(PHASE5_VALIDATION_PROFIT_FACTOR_GATE),
        },
    }


def write_config_audit_report(
    output_dir: str | None = None,
    *,
    n_rows: int | None = None,
    n_symbols: int | None = None,
) -> str:
    """Write the effective configuration snapshot and return its path."""

    root = output_dir or OUTPUTS_DIR
    reports_dir = os.path.join(root, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    path = os.path.join(reports_dir, "config_audit.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            effective_config_snapshot(n_rows=n_rows, n_symbols=n_symbols),
            handle,
            indent=2,
            sort_keys=True,
        )
    return path


validate_config()
