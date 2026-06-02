"""
Single source of truth for pipeline hyperparameters.

All modules import from here; do not duplicate defaults elsewhere.

Docs (per-phase behaviour and formulas):
  docs/phase0_shared.md … docs/phase5_oos.md

Quick tuning map
----------------
  Generalization (short OOS failures)  → SPLIT_MODE, CV_*, PHASE3_* gates, PHASE2_JOINT_TRAIN_VAL
  GPU RAM                              → PHASE1_SAMPLING_TOTAL, PHASE2_POPULATION_SIZE
  Search budget                        → PHASE2_GENERATIONS, PHASE3_REFINE_*, PHASE4_N_TRIALS
  Trade frequency / support            → MIN_TRADE_SUPPORT, MIN_CONDITIONS, MAX_CONDITIONS
  Risk after rules are fixed           → PHASE4_TP_*, PHASE4_SL_*, PHASE4_CAPITAL_*
  Fees / horizon (must match notebook) → FEE_PCT, TAIL_DROP_ROWS, MAX_HOLD_CANDLES

Environment overrides: DATA_ROOT, TRAIN_CSV_PATH, TEST_CSV_PATH
"""

from __future__ import annotations

import os

# Repo root (parent of gpu_fuzzy_trader/) — paths outside per-run OUTPUTS_DIR.
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir))


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


# =============================================================================
# Phase 0 — Paths & outputs
# =============================================================================
# Relative paths resolve from process cwd (usually repo root).
# run_pipeline.py may rewrite OUTPUTS_DIR and Phase 2 pool paths per --output.

DATA_ROOT = os.environ.get("DATA_ROOT", "").strip()
TRAIN_CSV_PATH = _env_str(
    "TRAIN_CSV_PATH",
    os.path.join(DATA_ROOT, "train.csv") if DATA_ROOT else "data/train.csv",
)
TEST_CSV_PATH = _env_str(
    "TEST_CSV_PATH",
    os.path.join(DATA_ROOT, "test.csv") if DATA_ROOT else "data/test.csv",
)

# Cached splits from train.csv (Phases 2–5). Rebuilt when train.csv is newer.
TRAIN_75_PATH = "data/train_75.parquet"
VALIDATION_25_PATH = "data/validation_25.parquet"
CV_FOLDS_MANIFEST_PATH = "data/cv_folds_manifest.json"

OUTPUTS_DIR = "outputs"
RUN_LOG_PATH = os.path.join(OUTPUTS_DIR, "run.log")
REPORTS_DIR = "outputs/reports"

# Per-run Phase 2 artifacts (rewritten under --output).
PHASE2_POOL_DIR = OUTPUTS_DIR
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

# Bars dropped per symbol at dataset tail (288-bar label horizon).
# RECOMMENDATION: keep equal to MAX_HOLD_CANDLES and CV_EMBARGO_BARS.
TAIL_DROP_ROWS = 288


# =============================================================================
# Phase 0 — Train / validation split (Phases 2–3)
# =============================================================================
# Phases 4–5 always use persisted train_75 + validation_25 (see splitter.py).
#
# SPLIT_MODE options:
#   "purged_rolling_cv" — K expanding-window folds, 288-bar embargo, ≥2 months
#                         train per fold; Phase 2/3 score worst fold. Default.
#   "holdout_75_25"     — legacy single 75/25 per symbol; faster, easier to
#                         overfit one validation season (risky for short).
#
# RECOMMENDATION: keep purged_rolling_cv for short; use holdout_75_25 only for
# fast debugging or apples-to-apples with older runs.

SPLIT_MODE = "purged_rolling_cv"

CV_N_FOLDS = 3
CV_EMBARGO_BARS = TAIL_DROP_ROWS
CV_BARS_PER_DAY = 288  # 5-minute bars
CV_MIN_TRAIN_MONTHS = 2.0  # per symbol, per fold; raise if folds feel too noisy

# =============================================================================
# Phase 0 — Backtest simulation (must match evaluator_v3.ipynb)
# =============================================================================
# Used by cpu_engine / gpu_engine in all phases.

INITIAL_CAPITAL = 1000.0
LEVERAGE = 1.0
FEE_PCT = 0.20  # round-trip % of notional; ↑ penalizes high turnover
MAX_HOLD_CANDLES = 288
MAX_TOTAL_EXPOSURE_PCT = 100.0
MIN_POSITION_NOTIONAL = 1.0


# =============================================================================
# Phase 0 — Logging
# =============================================================================

LOG_GENERATION_INTERVAL = 0  # 0 = auto ~10% of generations; N = every N gens


# =============================================================================
# Phase 1 — Feature selection (train.csv only)
# =============================================================================

# --- Ranking & shortlist ---
PHASE1_DISPERSION_THRESHOLD = 0.95  # drop near-constant columns
PHASE1_TOP_K_FEATURES = 25
PHASE1_MAX_FEATURE_OVERLAP = 0.50  # max Jaccard overlap long vs short lists
PHASE1_ASYMMETRIC_TARGET = True  # separate MI targets for long / short

# --- Stationarity (reduce regime-specific features) ---
PHASE1_STATIONARITY_FOLDS = 3
PHASE1_STATIONARITY_CV_MAX = 1.0
PHASE1_STATIONARITY_RANK_DRIFT_MAX = 5
PHASE1_STATIONARITY_STRATIFY = "regime"  # "regime" | "chronological"

# --- Regime detection (rolling regression when STRATIFY == "regime") ---
PHASE1_REGIME_FAST_WINDOW = 10
PHASE1_REGIME_SLOW_WINDOW = 24
PHASE1_REGIME_FAST_R2_THRESHOLD = 0.20
PHASE1_REGIME_SLOW_R2_THRESHOLD = 0.25
PHASE1_REGIME_FAST_SLOPE_THRESHOLD = 0.0016
PHASE1_REGIME_SLOW_SLOPE_THRESHOLD = 0.0010
PHASE1_REGIME_MED_WINDOW = 9
PHASE1_REGIME_MIN_DAYS = 14
# Minimum rows per stationarity fold (chronological or per-regime MI).
PHASE1_REGIME_MIN_SAMPLES = 100
PHASE1_REGIME_MODEL_PATH = os.path.join(
    OUTPUTS_DIR, "phase1_regime_cluster.joblib")

# Row budget for Phase 2 GPU backtests (equal sample per symbol).
# RECOMMENDATION: primary GPU RAM knob; try ≤150_000 on small GPUs.
PHASE1_SAMPLING_TOTAL = 701_500


# =============================================================================
# Phase 2 — Rule pool / NSGA-III
# =============================================================================

# --- Fixed risk during rule search (Phase 4 tunes TP/SL/capital) ---
PHASE2_TP = 2.0
PHASE2_SL = 1.5
PHASE2_CAPITAL_PCT = 32.0

# --- Rule genome (shared with Phase 3 team size) ---
MIN_CONDITIONS = 3
MAX_CONDITIONS = 4

# --- Trade support & pool admission ---
MIN_TRADE_SUPPORT = 120  # target executed trades for support penalty
SUPPORT_PENALTY_MAX = 12.0
MIN_TRADE_POOL_FLOOR = 50  # hard reject below this in archive
PHASE2_SUPPORT_PENALTY_WEIGHT_F1 = 0.8  # Sortino objective
PHASE2_SUPPORT_PENALTY_WEIGHT_F2 = 0.6  # drawdown objective
PHASE2_SUPPORT_PENALTY_WEIGHT_F3 = 0.5  # win-rate objective

PHASE2_RETURN_FLOOR_PCT = 0.0
PHASE2_VAL_RETURN_FLOOR_PCT = 0.0
PHASE2_PROFIT_FACTOR_FLOOR = 1.0
PHASE2_SYMBOL_MEDIAN_RETURN_FLOOR_PCT = -0.5
PHASE2_MIN_PROFITABLE_SYMBOLS = 5

PHASE2_POOL_REQUIRE_POSITIVE_SPLITS = True
PHASE2_POOL_TRAIN_RETURN_MIN_PCT = 0.0
PHASE2_POOL_VAL_RETURN_MIN_PCT = 0.0
# Holdout mode: require non-negative train/val return (see floors above).

# Purged CV pool admission (per-fold gates; see phase2_cv.evaluate_purged_cv_pool_admission)
PHASE2_CV_POOL_MIN_FOLDS_PASS = 1  # of CV_N_FOLDS must pass train+val gates
PHASE2_CV_MIN_TRADE_POOL_FLOOR = 25
PHASE2_CV_POOL_TRAIN_RETURN_MIN_PCT = 0.0
PHASE2_CV_POOL_VAL_RETURN_MIN_PCT = 0.0
PHASE2_CV_PROFIT_FACTOR_FLOOR = 0.95
PHASE2_CV_MIN_VAL_TRADES = 10

# --- Fitness & joint evaluation ---
SORTINO_CAP = 3.0
SORTINO_SCALE = 3.0
PHASE2_JOINT_TRAIN_VAL = True  # f1 uses min(train_sortino, val_sortino)
# With purged_rolling_cv, val side is worst across CV folds.

# --- Diversity & early stop ---
PHASE2_DIVERSITY_HAMMING_THRESHOLD = 2
PHASE2_DIVERSITY_PENALTY = 4.0
PHASE2_EARLY_STOP_ENABLED = True
PHASE2_EARLY_STOP_MIN_GENERATION = 50
PHASE2_EARLY_STOP_MEAN_RETURN_PCT = -5.0
PHASE2_EARLY_STOP_USE_MEDIAN_RETURN = True  # robust vs one bad Pareto member
PHASE2_EARLY_STOP_DISABLED_IN_CV = False  # enable early stop in purged CV mode

# --- NSGA-III budget ---
PHASE2_POPULATION_SIZE = 450
PHASE2_GENERATIONS = 80
PHASE2_ALGORITHM = "NSGA3"
PHASE2_ARCHIVE_MAX_SIZE = 500
PHASE2_ARCHIVE_SEED_FRACTION = 0.35
PHASE2_SEED = 42

# --- Regime-stratified support (uses PHASE1_REGIME_MODEL_PATH) ---
PHASE2_REGIME_SUPPORT_ENABLED = True
PHASE2_REGIME_MODEL_PATH = PHASE1_REGIME_MODEL_PATH
PHASE2_REGIME_CONCENTRATION_MIN = 0.70
PHASE2_REGIME_MIN_WIN_RATE = 0.35
PHASE2_REGIME_USE_PNL_GATE = True
PHASE2_REGIME_MIN_TRADE_FRACTION = 1.0
PHASE2_REGIME_REQUIRE_VAL_CONFIRMATION = False

# --- Engine & initialization ---
PHASE2_NUMBA_ENABLED = True
PHASE2_INIT_STRATEGY = "stratified_sparse"  # "stratified_sparse" | "legacy"
PHASE2_INIT_STRATUM_FRACTIONS = (0.50, 0.30, 0.20)
PHASE2_INIT_SOFTMAX_TEMP = 0.5
PHASE2_INIT_SCORE_EPS = 1e-6
PHASE2_INIT_UNIFORM_MIX = 0.05
PHASE2_MUTATION_WEIGHTED_ACTIVATE_PROB = 0.70


# =============================================================================
# Phase 3 — Rule set selection (greedy + NSGA-II)
# =============================================================================

# --- Team shape ---
PHASE3_MIN_RULES = 2
PHASE3_MAX_RULES = 5
PHASE3_MIN_SYMBOL_COVERAGE = 7  # of 10 symbols with ≥1 val trade
PHASE3_MAX_CAPITAL_PCT_PER_RULE = 50.0

# --- Engines ---
PHASE3_USE_GPU = False  # JAX path; enable after parity checks
PHASE3_USE_PARALLEL_BATCH = True
PHASE3_BATCH_WORKERS = min(32, os.cpu_count() or 4)
PHASE3_NUMBA_ENABLED = True

# --- Refinement budget ---
PHASE3_REFINE_GENERATIONS = 80
PHASE3_REFINE_POP_SIZE = 100
PHASE3_GREEDY_WEIGHTS = (1.0, 0.7, 0.5)  # sortino, drawdown, win_rate

# --- Objectives & anti-overfit gates ---
# RECOMMENDATION: keep USE_TRAIN_TARGET True; rely on CV + gates, not val-only fit.
PHASE3_USE_TRAIN_TARGET = True
PHASE3_USE_MAXIMIN_SCORE = True
PHASE3_SYMBOL_CONSISTENCY_WEIGHT = 10.0
PHASE3_TRAIN_VAL_CORR_WEIGHT = 8.0
PHASE3_VAL_GATE_PENALTY = 100.0

PHASE3_VAL_SORTINO_RATIO_GATE = 0.5  # val_sortino ≥ ratio × train_sortino
PHASE3_VAL_DRAWDOWN_RATIO_GATE = 1.15  # val_dd ≤ ratio × train_dd
PHASE3_PER_RULE_MIN_VAL_TRADES_PER_SYMBOL = 8

PHASE3_VAL_RETURN_FLOOR_PCT = 0.0
PHASE3_VAL_PROFIT_FACTOR_FLOOR = 1.0
PHASE3_TRAIN_RETURN_FLOOR_PCT = 0.0
PHASE3_TRAIN_PROFIT_FACTOR_FLOOR = 1.0
PHASE3_MIN_PROFITABLE_SYMBOLS = 5
PHASE3_SYMBOL_MEDIAN_RETURN_FLOOR_PCT = -0.5

# Penalise val >> train or train >> val (classic short/long overfit signatures).
PHASE3_TRAIN_VAL_GAP_MAX_PCT = 15.0
PHASE3_VAL_TRAIN_GAP_MAX_PCT = 10.0
PHASE3_GAP_PENALTY_WEIGHT = 4.0

# --- Rule-team orthogonality (validation masks) ---
PHASE3_MIN_INCREMENTAL_TRADES = 60
PHASE3_INCREMENTAL_GATE_PENALTY = 60.0
PHASE3_JACCARD_PENALTY_WEIGHT = 35.0
PHASE3_JACCARD_SIMILARITY_GATE = 0.75


# =============================================================================
# Phase 4 — Walk-forward risk (TP / SL / capital on validation_25)
# =============================================================================
# Rule conditions are frozen; only risk params are optimized.
# PHASE4_WF_SPLITS slices validation_25 only (not full CV folds).

# --- Search space ---
PHASE4_TP_MIN = 2.0
PHASE4_TP_MAX = 5.0
PHASE4_SL_MIN = 1.0
PHASE4_SL_MAX = 2.0
PHASE4_MIN_TP_SL_RATIO = 1.2  # tp must exceed sl (trend-following risk/reward)
PHASE4_CAPITAL_PCT_MIN = 30.0
PHASE4_CAPITAL_PCT_MAX = 50.0
PHASE4_TP_STEP = 0.5
PHASE4_SL_STEP = 0.5
PHASE4_CAPITAL_STEP = 5.0

# --- Optuna ---
PHASE4_N_TRIALS = 200
PHASE4_SAMPLER = "tpe"  # "tpe" | "nsga2"
PHASE4_SEED = 42
PHASE4_N_JOBS = 1
PHASE4_HARD_CAP_NORMALIZE = True  # sum capital_pct ≤ MAX_TOTAL_EXPOSURE_PCT

# --- Walk-forward on validation split ---
PHASE4_WF_SPLITS = 2
PHASE4_INCLUDE_TAIL_HOLDOUT = True
PHASE4_TAIL_HOLDOUT_FRACTION = 0.25
PHASE4_WORST_RETURN_WEIGHT = 1.0
PHASE4_WORST_DRAWDOWN_WEIGHT = 1.0
PHASE4_WORST_TURNOVER_WEIGHT = 1.0

# --- Feasibility filters (trial must pass all) ---
PHASE4_MAX_WORST_DRAWDOWN_PCT = 15.0
PHASE4_MIN_WORST_TRADES = 30
PHASE4_MIN_WORST_FOLD_RETURN_PCT = 0.5
PHASE4_MIN_WORST_FOLD_PF = 1.05
# RECOMMENDATION: raise WF_SPLITS to 4–6 if short still fails OOS after CV in P2/P3.


# =============================================================================
# Phase 5 — Out-of-sample (test.csv only; never used in Phases 1–4)
# =============================================================================

PHASE5_VALIDATION_RETURN_GATE_PCT = 0.0  # deployment flag on val metrics only
PHASE5_VALIDATION_PROFIT_FACTOR_GATE = 1.0
# RECOMMENDATION: treat test metrics in reports as truth; do not tune on TEST_CSV_PATH.
