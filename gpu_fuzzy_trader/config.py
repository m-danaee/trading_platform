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

# ---------------------------------------------------------------------------
# Global seed — None means a fresh random seed is drawn once per process.
# Set GLOBAL_SEED to an integer (e.g. 42) to reproduce a specific run.
# ---------------------------------------------------------------------------
GLOBAL_SEED: int | None = None

# One seed per process, lazily initialised on first call to get_seed().
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

# Increased from 3→5 folds: more folds give better estimate of out-of-fold
# generalisation, reducing the chance of the short strategy overfitting one season.
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
# max Jaccard overlap long vs short lists (was 0.50 → 12 shared)
PHASE1_MAX_FEATURE_OVERLAP = 0.65
PHASE1_ASYMMETRIC_TARGET = True  # separate MI targets for long / short

# --- Stationarity (reduce regime-specific features) ---
PHASE1_STATIONARITY_FOLDS = 3
PHASE1_STATIONARITY_CV_MAX = 1.0
PHASE1_STATIONARITY_RANK_DRIFT_MAX = 8
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
# Reduced 701_500→350_000: with CV_N_FOLDS=5 and PHASE2_CV_FOLD_WORKERS=2,
# the peak GPU allocation was 8.26 GiB (OOM). Halving the row budget is the
# single largest lever — GPU memory scales linearly with this value.
PHASE1_SAMPLING_TOTAL = 701_000


# =============================================================================
# Phase 2 — Rule pool / NSGA-III
# =============================================================================

# --- Fixed risk during rule search (Phase 4 tunes TP/SL/capital) ---
# Tighter SL (1.0→0.8) forces Phase 2 to find rules with higher precision;
# keeps raw TP/SL ratio at 2.5 which is more selective than 1.33 previously.
PHASE2_TP = 2.0
PHASE2_SL = 1.0
PHASE2_CAPITAL_PCT = 30.0

# --- Rule genome (shared with Phase 3 team size) ---
MIN_CONDITIONS = 3
MAX_CONDITIONS = 4

# --- Trade support & pool admission ---
MIN_TRADE_SUPPORT = 150  # target executed trades for support penalty
SUPPORT_PENALTY_MAX = 12.0
MIN_TRADE_POOL_FLOOR = 50  # hard reject below this in archive
PHASE2_SUPPORT_PENALTY_WEIGHT_F1 = 0.8  # Sortino objective
PHASE2_SUPPORT_PENALTY_WEIGHT_F2 = 0.6  # drawdown objective
PHASE2_SUPPORT_PENALTY_WEIGHT_F3 = 0.5  # win-rate objective
PHASE2_USE_TOTAL_RETURN_OBJ = True  # True: f3 is total_return_pct, False: f3 is win_rate

# Run 3 analysis: strict stacked floors collapsed pool to only 17 long / 22 short
# rules. Quality filtering should happen at the CV majority-vote level, not by
# stacking floor requirements. Relaxed back toward run-1 levels.
PHASE2_RETURN_FLOOR_PCT = 0.0
PHASE2_VAL_RETURN_FLOOR_PCT = 0.0
PHASE2_PROFIT_FACTOR_FLOOR = 1.0
PHASE2_SYMBOL_MEDIAN_RETURN_FLOOR_PCT = -0.5
PHASE2_MIN_PROFITABLE_SYMBOLS = 5

PHASE2_POOL_REQUIRE_POSITIVE_SPLITS = True
PHASE2_POOL_TRAIN_RETURN_MIN_PCT = 0.0
PHASE2_POOL_VAL_RETURN_MIN_PCT = 0.0
# Holdout mode: require non-negative train/val return (see floors above).

# Purged CV pool admission (per-fold gates).
# Changed 3→2 (majority vote): unanimity across 3 folds gave only 17/22 rules.
# 2-of-3 still rejects season-specific overfitters while building a usable pool.
PHASE2_CV_POOL_MIN_FOLDS_PASS = 2
PHASE2_CV_MIN_TRADE_POOL_FLOOR = 25
PHASE2_CV_POOL_TRAIN_RETURN_MIN_PCT = 0.0
PHASE2_CV_POOL_VAL_RETURN_MIN_PCT = 0.0
PHASE2_CV_PROFIT_FACTOR_FLOOR = 1.0
PHASE2_CV_MIN_VAL_TRADES = 12
# Rank fallback when strict CV admission starves Phase 3 (run log: 13/123 long).
PHASE2_CV_POOL_TARGET_MIN = 25
PHASE2_CV_POOL_RANK_ADMIT_TOP_K = 50
PHASE2_CV_RANK_MIN_FOLDS_PASS = 1

# --- Fitness & joint evaluation ---
SORTINO_CAP = 5.0
SORTINO_SCALE = 3.0
PHASE2_JOINT_TRAIN_VAL = True  # f1 uses min(train_sortino, val_sortino)
# With purged_rolling_cv, val side is worst across CV folds.

# --- Diversity & early stop ---
# Run log: from gen 57 onward the long Pareto was fully saturated (450/450)
# and mean return was flat/worsening — wasted ~25 generations.
# Raised Hamming threshold 2→3 and diversity penalty 4→6 to maintain spread.
PHASE2_DIVERSITY_HAMMING_THRESHOLD = 3
PHASE2_DIVERSITY_PENALTY = 6.0
PHASE2_EARLY_STOP_ENABLED = True
PHASE2_EARLY_STOP_MIN_GENERATION = 40
# Tightened -5.0 → -3.5: short run showed -10% by gen 65 yet ran to completion.
PHASE2_EARLY_STOP_MEAN_RETURN_PCT = -5.0
PHASE2_EARLY_STOP_USE_MEDIAN_RETURN = True  # robust vs one bad Pareto member
# also require sparse valid_rules on Pareto
PHASE2_EARLY_STOP_MIN_VALID_RULES = 5
PHASE2_EARLY_STOP_DISABLED_IN_CV = False  # enable early stop in purged CV mode

# --- Parallel fold evaluation ---
# Number of threads used to evaluate CV folds simultaneously in Phase 2.
# Set to 0 to match CV_N_FOLDS automatically; set to 1 to disable parallelism.
# Capped at 2 (was 0=auto=5): with 5 folds each running a GPU backtest in
# parallel the peak VRAM demand was 5× a single fold. 2 workers keeps peak
# usage to ~2× while still providing meaningful parallelism.
PHASE2_CV_FOLD_WORKERS = 1

# --- NSGA-III budget ---
# Population reduced 600→400: with 350k sampling rows and 2 parallel fold
# workers the per-generation GPU allocation must fit in available VRAM.
# 400 still provides strong diversity (previous successful run used 450).
PHASE2_POPULATION_SIZE = 400
PHASE2_GENERATIONS = 100
PHASE2_ALGORITHM = "NSGA3"
# Archive adjusted to match new population size.
PHASE2_ARCHIVE_MAX_SIZE = 400
PHASE2_ARCHIVE_SEED_FRACTION = 0.25
PHASE2_SEED: int = get_seed()  # per-run random seed; set GLOBAL_SEED=42 to reproduce

# --- Regime-stratified support (uses PHASE1_REGIME_MODEL_PATH) ---
PHASE2_REGIME_SUPPORT_ENABLED = True
PHASE2_REGIME_MODEL_PATH = PHASE1_REGIME_MODEL_PATH
PHASE2_REGIME_CONCENTRATION_MIN = 0.70
# Raised min win rate 0.35→0.40: short OOS failure shows rules had insufficient
# edge across regimes — requiring 40% win rate raises quality of admitted rules.
PHASE2_REGIME_MIN_WIN_RATE = 0.40
PHASE2_REGIME_USE_PNL_GATE = True
PHASE2_REGIME_MIN_TRADE_FRACTION = 1.0
# Enable val confirmation: prevents regime-specific overfit (key for short side).
PHASE2_REGIME_REQUIRE_VAL_CONFIRMATION = True

# --- Engine & initialization ---
# When False, Phase 2 uses CPUBacktestEngine only (no JAX). Tuning low_ram sets False.
PHASE2_USE_GPU = True
PHASE2_NUMBA_ENABLED = True
PHASE2_INIT_STRATEGY = "stratified_sparse"  # "stratified_sparse" | "legacy"
PHASE2_INIT_STRATUM_FRACTIONS = (0.67, 0.33)
PHASE2_INIT_SOFTMAX_TEMP = 0.5
PHASE2_INIT_SCORE_EPS = 1e-6
PHASE2_INIT_UNIFORM_MIX = 0.05
PHASE2_MUTATION_WEIGHTED_ACTIVATE_PROB = 0.70


# =============================================================================
# Phase 3 — Rule set selection (greedy + NSGA-II)
# =============================================================================

# --- Team shape ---
PHASE3_MIN_RULES = 2
PHASE3_MAX_RULES = 3
PHASE3_MIN_SYMBOL_COVERAGE = 7  # of 10 symbols with ≥1 val trade
PHASE3_MAX_CAPITAL_PCT_PER_RULE = 50.0

# --- Engines ---
PHASE3_USE_GPU = False  # JAX path; enable after parity checks
PHASE3_USE_PARALLEL_BATCH = True
PHASE3_BATCH_WORKERS = min(32, os.cpu_count() or 4)
PHASE3_NUMBA_ENABLED = True

# --- Refinement budget ---
# Run log: both long and short refinements showed zero improvement through 80 gens
# with pop=100. Raising to pop=300, gen=250 provides more exploration capacity.
# Previous config already raised to pop=200/gen=200 which is reasonable; keep gen=250
# but cap pop at 300 to avoid 5×budget blowout on large pools.
PHASE3_REFINE_GENERATIONS = 250
PHASE3_REFINE_POP_SIZE = 300
PHASE3_SMALL_POOL_THRESHOLD = 20
PHASE3_SMALL_POOL_POP = 100
PHASE3_SMALL_POOL_GEN = 60
PHASE3_MIN_PARETO_FRONT = 3
PHASE3_REFINE_EARLY_STOP_PARETO_ONE_GENS = 15
PHASE3_GREEDY_STOP_ON_WORSEN = True
PHASE3_GREEDY_WEIGHTS = (0.8, 0.6, 0.5)  # sortino, drawdown, win_rate

# --- Objectives & anti-overfit gates ---
# RECOMMENDATION: keep USE_TRAIN_TARGET True; rely on CV + gates, not val-only fit.
PHASE3_USE_TRAIN_TARGET = True
PHASE3_USE_MAXIMIN_SCORE = True
PHASE3_SYMBOL_CONSISTENCY_WEIGHT = 10.0
PHASE3_TRAIN_VAL_CORR_WEIGHT = 8.0
PHASE3_VAL_GATE_PENALTY = 10.0

# Tightened val/train gates: run log shows Phase 3 mean_return was deeply
# negative (-150% long, -117% short) meaning the greedy seed was never forced
# toward positive val territory. Tightening these gates acts as a hard floor.
PHASE3_VAL_SORTINO_RATIO_GATE = 0.6  # val_sortino ≥ ratio × train_sortino
PHASE3_VAL_DRAWDOWN_RATIO_GATE = 1.10  # val_dd ≤ ratio × train_dd (tighter)
PHASE3_PER_RULE_MIN_VAL_TRADES_PER_SYMBOL = 6

PHASE3_VAL_RETURN_FLOOR_PCT = 0.5
PHASE3_VAL_PROFIT_FACTOR_FLOOR = 1.05
PHASE3_TRAIN_RETURN_FLOOR_PCT = 1.0
PHASE3_TRAIN_PROFIT_FACTOR_FLOOR = 1.05
PHASE3_MIN_PROFITABLE_SYMBOLS = 6
PHASE3_SYMBOL_MEDIAN_RETURN_FLOOR_PCT = 0.0

# Penalise val >> train or train >> val (classic short/long overfit signatures).
# Narrowed gap from 15→10% on train side to detect overfit earlier.
PHASE3_TRAIN_VAL_GAP_MAX_PCT = 10.0
PHASE3_VAL_TRAIN_GAP_MAX_PCT = 10.0
PHASE3_GAP_PENALTY_WEIGHT = 4.0

# --- Rule-team orthogonality (validation masks) ---
PHASE3_MIN_INCREMENTAL_TRADES = 60
PHASE3_INCREMENTAL_GATE_PENALTY = 60.0
PHASE3_JACCARD_PENALTY_WEIGHT = 35.0
PHASE3_JACCARD_SIMILARITY_GATE = 0.75


# =============================================================================
# Phase 4 — Walk-forward risk (TP / SL / capital; all CV val folds when available)
# =============================================================================
# Rule conditions are frozen; only risk params are optimized.
# With purged_rolling_cv, WF windows are built on every fold's val block (not only
# validation_25). Final deployment gate still uses validation_25 (last fold).

# --- Search space ---
PHASE4_TP_MIN = 2.0
PHASE4_TP_MAX = 4.0
PHASE4_SL_MIN = 1.0
PHASE4_SL_MAX = 2.0
PHASE4_MIN_TP_SL_RATIO = 1.2  # tp must exceed sl (trend-following risk/reward)
PHASE4_CAPITAL_PCT_MIN = 30.0
PHASE4_CAPITAL_PCT_MAX = 30.0
PHASE4_TP_STEP = 0.5
PHASE4_SL_STEP = 0.5
PHASE4_CAPITAL_STEP = 5.0

# --- Optuna ---
PHASE4_N_TRIALS = 200
PHASE4_SAMPLER = "tpe"  # "tpe" | "nsga2"
PHASE4_SEED: int = get_seed()  # per-run random seed; set GLOBAL_SEED=42 to reproduce
PHASE4_N_JOBS = 1
PHASE4_HARD_CAP_NORMALIZE = True  # sum capital_pct ≤ MAX_TOTAL_EXPOSURE_PCT

# --- Walk-forward on validation split ---
# Reduced 4→2: with val=174k rows and only 2-4 rules, wf_splits=4 creates
# windows too small to accumulate 20 trades — Phase 4 rejected 100% of trials
# in runs 3 and 4. 2 splits gives each window ~87k rows (~300 bars) and
# enough trades to evaluate properly.
PHASE4_WF_SPLITS = 2
PHASE4_INCLUDE_TAIL_HOLDOUT = True
PHASE4_TAIL_HOLDOUT_FRACTION = 0.25
PHASE4_WORST_RETURN_WEIGHT = 1.5
PHASE4_WORST_DRAWDOWN_WEIGHT = 2.0
PHASE4_WORST_TURNOVER_WEIGHT = 0.5

# --- Feasibility filters (trial must pass all) ---
# Run 3: BOTH directions had zero feasible Phase 4 trials.
# MIN_WORST_TRADES=40 is unreachable with only 2-3 rules per team;
# MIN_WORST_FOLD_RETURN_PCT=1.5% demands consistent positive return per WF window.
# Goal: reject strategies with consistent losses or blow-up drawdowns, not
# demand strong positive returns at the per-fold level.
PHASE4_MAX_WORST_DRAWDOWN_PCT = 15.0
PHASE4_MIN_WORST_TRADES = 15
PHASE4_MIN_WORST_FOLD_RETURN_PCT = -2.0
PHASE4_MIN_WORST_FOLD_PF = 1.0
# RECOMMENDATION: raise WF_SPLITS to 4–6 if short still fails OOS after CV in P2/P3.


# =============================================================================
# Phase 5 — Out-of-sample (test.csv only; never used in Phases 1–4)
# =============================================================================

# Tightened: Phase 5 gate was 0.0% return / 1.0 PF — the short strategy passed
# this trivially while having -16.91% OOS. Raise to filter out marginal strategies.
PHASE5_VALIDATION_RETURN_GATE_PCT = 2.0  # deployment flag on val metrics only
PHASE5_VALIDATION_PROFIT_FACTOR_GATE = 1.05
# RECOMMENDATION: treat test metrics in reports as truth; do not tune on TEST_CSV_PATH.

# --- Trading Regime and Refinement Fixes (2026-06-04) ---
# require profit > 0 in >=2 of 3 regimes
PHASE2_REGIME_PROFITABILITY_GATE: bool = True
PHASE2_REGIME_MIN_RETURN_PER_REGIME: float = 0.0  # per-regime return floor
# drop features with Spearman sign flip across folds
PHASE1_REQUIRE_SIGN_CONSISTENCY: bool = True
# must have same sign in >= N folds
PHASE1_SIGN_CONSISTENCY_MIN_FOLDS: int = 2
# Ignore sign flips when |Spearman rho| is below this (noise-level correlations).
PHASE1_SIGN_CONSISTENCY_MIN_ABS_CORR: float = 0.02
# bars in the last 25% of training period count 2x in return
PHASE2_RECENCY_WEIGHT_ENABLED: bool = True
PHASE2_RECENCY_WEIGHT_FRACTION: float = 0.25      # last 25% of training bars
PHASE2_RECENCY_WEIGHT_MULTIPLIER: float = 2.0     # these bars count double
# rule must be profitable on validation split of most recent fold
PHASE2_REQUIRE_LAST_FOLD_POSITIVE: bool = True
