"""
Single source of truth for pipeline hyperparameters.

All modules import from here; do not duplicate defaults elsewhere.
Tuning guide: docs/hyperparameters/README.md (Phase 0–5 per-phase docs).
"""

from __future__ import annotations

import os

# Repo root (parent of gpu_fuzzy_trader/) — used for paths outside the run output dir.
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir))

# =============================================================================
# Phase 0 — Shared (paths, schema, backtest simulation, logging)
# =============================================================================

# --- Paths -------------------------------------------------------------------
# Relative paths resolve from the process cwd (typically repo root).
# run_pipeline.py may override OUTPUTS_DIR and Phase 2 pool paths per run.

TRAIN_CSV_PATH = "data/train.csv"
TEST_CSV_PATH = "data/test.csv"  # Phase 5 OOS only — never use in Phases 1–4
TRAIN_75_PATH = "data/train_75.parquet"
VALIDATION_25_PATH = "data/validation_25.parquet"
OUTPUTS_DIR = "outputs"
RUN_LOG_PATH = os.path.join(OUTPUTS_DIR, "run.log")
REPORTS_DIR = "outputs/reports"

# Per-run pools/history; run_pipeline.py rewrites these under --output.
PHASE2_POOL_DIR = OUTPUTS_DIR
PHASE2_POOL_PATHS = {
    "long": os.path.join(OUTPUTS_DIR, "phase2_long_pool.json"),
    "short": os.path.join(OUTPUTS_DIR, "phase2_short_pool.json"),
}
PHASE2_HISTORY_PATHS = {
    "long": os.path.join(OUTPUTS_DIR, "phase2_long_history.json"),
    "short": os.path.join(OUTPUTS_DIR, "phase2_short_history.json"),
}

# Cross-run warm-start archive (not cleared by --output).
PHASE2_ARCHIVE_DIR = os.path.join(_PROJECT_ROOT, "phase2_rule_archive")
PHASE2_ARCHIVE_PATHS = {
    "long": os.path.join(PHASE2_ARCHIVE_DIR, "phase2_long_archive.json"),
    "short": os.path.join(PHASE2_ARCHIVE_DIR, "phase2_short_archive.json"),
}

# --- Schema ------------------------------------------------------------------
# LABEL_* and META_* must never enter feature matrices. INTERNAL_* are loader-only.

LABEL_COLUMNS = [
    "label_open_next",
    "label_close_288",
    "label_min_288",
    "label_max_288",
    "label_max_before_min",
]
META_COLUMNS = ["datetime", "symbol"]
INTERNAL_COLUMNS = ("_symbol_bar_index",)
# Drop trailing bars per symbol where 288-bar labels are undefined.
TAIL_DROP_ROWS = 288

# --- Backtest (must match evaluator_v3.ipynb) --------------------------------
# Used by cpu_engine / gpu_engine for all phase fitness evaluations.

INITIAL_CAPITAL = 1000.0
LEVERAGE = 1.0
FEE_PCT = 0.20  # round-trip fee %; penalizes high-turnover rules
MAX_HOLD_CANDLES = 288  # aligned with label horizon
MAX_TOTAL_EXPOSURE_PCT = 100.0
MIN_POSITION_NOTIONAL = 1.0

# --- Logging -----------------------------------------------------------------
# 0 = auto-throttle generation logs; N > 0 = log every N generations.

LOG_GENERATION_INTERVAL = 0

# =============================================================================
# Phase 1 — Feature selection (train only; features/selector.py)
# =============================================================================

# MI ranking: drop near-constant columns, cap list size, limit long/short overlap.
PHASE1_DISPERSION_THRESHOLD = 0.95
PHASE1_TOP_K_FEATURES = 22
PHASE1_MAX_FEATURE_OVERLAP = 0.50
# Signed 3-class PnL surrogate so long/short shortlists diverge (vs binary success).
PHASE1_ASYMMETRIC_TARGET = True

# Stationarity: per-fold MI stability across folds (regime or chronological).
PHASE1_STATIONARITY_FOLDS = 3
PHASE1_STATIONARITY_CV_MAX = 1.0
PHASE1_STATIONARITY_RANK_DRIFT_MAX = 10
PHASE1_STATIONARITY_STRATIFY = "regime"  # "regime" | "chronological"

# Regime clustering inputs (train rows only when STRATIFY == "regime").
PHASE1_REGIME_FEATURES = [
    "realized_vol_20",
    "parkinson_vol_20",
    "atr_pct_14",
    "vol_regime_pct_120",
    "efficiency_ratio_20",
    "ret_autocorr_1_30",
    "amihud_illiquidity_20",
    "vol_ratio_20_100",
]
PHASE1_REGIME_N_CLUSTERS = 3  # defaults to STATIONARITY_FOLDS when unset at runtime
PHASE1_REGIME_MIN_SAMPLES = 100
PHASE1_REGIME_CLUSTERER = "gmm"  # "gmm" | "kmeans"
PHASE1_REGIME_GMM_REG_COVAR = 1e-6
PHASE1_REGIME_ZERO_VAR_EPS = 1e-12
PHASE1_REGIME_MODEL_PATH = os.path.join(
    OUTPUTS_DIR, "phase1_regime_cluster.joblib")

# Phase 2 backtest row budget (equal per symbol). Primary GPU memory knob.
# Raising this grows JAX arrays roughly linearly; on memory-limited GPUs keep ≤ 150_000.
PHASE1_SAMPLING_TOTAL = 600_000

# =============================================================================
# Phase 2 — Rule pool / NSGA-III (phases/phase2_rule_pool.py)
# =============================================================================

# Fixed TP/SL/capital during rule search — isolates rule logic from risk tuning (Phase 4).
PHASE2_TP = 3.0
PHASE2_SL = 1.5
PHASE2_CAPITAL_PCT = 32.0

# Rule genome: number of active fuzzy conditions per chromosome.
MIN_CONDITIONS = 3
MAX_CONDITIONS = 4

# Trade-count gates and support penalty (noisy Sortino when executed << support).
MIN_TRADE_SUPPORT = 300
SUPPORT_PENALTY_MAX = 50.0
# hard reject below this in archive (~ MIN_TRADE_SUPPORT // 4)
MIN_TRADE_POOL_FLOOR = 75

# Fitness transform: tanh(sortino / SCALE) * CAP (avoids sentinel pinning at gen 0).
SORTINO_CAP = 5.0
SORTINO_SCALE = 3.0

# Worst-case train/val Sortino when both splits are evaluated.
PHASE2_JOINT_TRAIN_VAL = True

# Population diversity: penalize chromosomes within Hamming distance ≤ threshold.
PHASE2_DIVERSITY_HAMMING_THRESHOLD = 2
PHASE2_DIVERSITY_PENALTY = 5.0

# NSGA-III search budget and persistence.
PHASE2_POPULATION_SIZE = 200
PHASE2_GENERATIONS = 200
PHASE2_ALGORITHM = "NSGA3"
PHASE2_ARCHIVE_MAX_SIZE = 500
# fraction seeded from pool + cross-run archive
PHASE2_ARCHIVE_SEED_FRACTION = 0.35

# Regime-stratified trade support (Phase 1 GMM artifact).
PHASE2_REGIME_SUPPORT_ENABLED = True
PHASE2_REGIME_MODEL_PATH = PHASE1_REGIME_MODEL_PATH
PHASE2_REGIME_CONCENTRATION_MIN = 0.90
PHASE2_REGIME_MIN_WIN_RATE = 0.40
PHASE2_REGIME_USE_PNL_GATE = True
PHASE2_REGIME_MIN_TRADE_FRACTION = 1.0
PHASE2_REGIME_REQUIRE_VAL_CONFIRMATION = False

# Numba-accelerated NSGA helpers (warm-up compile on first call; cache=True).
PHASE2_NUMBA_ENABLED = True

# =============================================================================
# Phase 3 — Rule set selection (phases/phase3_rule_set.py, phase3_greedy.py)
# =============================================================================

PHASE3_MIN_RULES = 2
PHASE3_MAX_RULES = 3
PHASE3_MIN_SYMBOL_COVERAGE = 7  # of 10 symbols must have ≥ 1 trade on validation
PHASE3_USE_GPU = False  # JAX mask path + batched eval; enable after parity tests pass
# ProcessPool batch on CPU (independent of GPU)
PHASE3_USE_PARALLEL_BATCH = True
PHASE3_BATCH_WORKERS = min(32, os.cpu_count() or 4)
PHASE3_NUMBA_ENABLED = True  # NSGA-II sort/crowding via evolution.numba_ops
PHASE3_REFINE_GENERATIONS = 80
PHASE3_REFINE_POP_SIZE = 100
PHASE3_GREEDY_WEIGHTS = (1.0, 0.7, 0.5)  # sortino, drawdown, win_rate
PHASE3_SYMBOL_CONSISTENCY_WEIGHT = 10.0

# Train-as-target / validation-as-gate (reduce validation leakage in objectives).
PHASE3_USE_TRAIN_TARGET = True
# reject if val_sortino < ratio * train_sortino
PHASE3_VAL_SORTINO_RATIO_GATE = 0.5
PHASE3_VAL_DRAWDOWN_RATIO_GATE = 1.5  # reject if val_dd > ratio * train_dd
PHASE3_PER_RULE_MIN_VAL_TRADES_PER_SYMBOL = 5
# penalty on low corr(per-symbol PnL train, val)
PHASE3_TRAIN_VAL_CORR_WEIGHT = 5.0
PHASE3_VAL_GATE_PENALTY = 75.0

# =============================================================================
# Phase 4 — Walk-forward risk optimization (phases/phase4_wf_optimizer.py)
# =============================================================================

PHASE4_TP_MIN = 2.0
PHASE4_TP_MAX = 4.0
PHASE4_SL_MIN = 1.0
PHASE4_SL_MAX = 2.0
PHASE4_CAPITAL_PCT_MIN = 10.0
PHASE4_CAPITAL_PCT_MAX = 50.0
PHASE4_TP_STEP = 0.2
PHASE4_SL_STEP = 0.2
PHASE4_CAPITAL_STEP = 5.0
PHASE4_TOTAL_CAP_PENALTY = 2.0
PHASE4_N_TRIALS = 1000
PHASE4_WF_SPLITS = 2
PHASE4_MAX_WORST_DRAWDOWN_PCT = 15.0
PHASE4_SAMPLER = "nsga2"  # alternative: "tpe"
PHASE4_SEED = 42
PHASE4_N_JOBS = 1
# Scale per-rule capital_pct so sum ≤ MAX_TOTAL_EXPOSURE_PCT after optimization.
PHASE4_HARD_CAP_NORMALIZE = True

# =============================================================================
# Phase 5 — Out-of-sample evaluation (phases/phase5_oos.py)
# =============================================================================
# No phase-specific constants: uses TEST_CSV_PATH, REPORTS_DIR, and backtest block above.
