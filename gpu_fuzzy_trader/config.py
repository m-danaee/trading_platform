"""
config.py — Single source of truth for all hyperparameters.

No module may define its own defaults that override these values.
All paths, constants, and behavioral settings live here.
"""

from __future__ import annotations

import os

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRAIN_CSV_PATH = "data/train.csv"
TEST_CSV_PATH = "data/test.csv"
TRAIN_75_PATH = "data/train_75.parquet"
VALIDATION_25_PATH = "data/validation_25.parquet"
OUTPUTS_DIR = "outputs"
RUN_LOG_PATH = os.path.join(OUTPUTS_DIR, "run.log")
REPORTS_DIR = "outputs/reports"
# Phase 2 pool files now live in the run-specific output directory
# (set dynamically by run_pipeline.py via _temporary_output_paths)
PHASE2_POOL_DIR = OUTPUTS_DIR  # Will be overridden at runtime
PHASE2_POOL_PATHS = {
    "long": os.path.join(OUTPUTS_DIR, "phase2_long_pool.json"),
    "short": os.path.join(OUTPUTS_DIR, "phase2_short_pool.json"),
}
PHASE2_HISTORY_PATHS = {
    "long": os.path.join(OUTPUTS_DIR, "phase2_long_history.json"),
    "short": os.path.join(OUTPUTS_DIR, "phase2_short_history.json"),
}
PHASE2_ARCHIVE_DIR = os.path.join(_PROJECT_ROOT, "phase2_rule_archive")
PHASE2_ARCHIVE_PATHS = {
    "long": os.path.join(PHASE2_ARCHIVE_DIR, "phase2_long_archive.json"),
    "short": os.path.join(PHASE2_ARCHIVE_DIR, "phase2_short_archive.json"),
}

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
LABEL_COLUMNS = [
    "label_open_next",
    "label_close_288",
    "label_min_288",
    "label_max_288",
    "label_max_before_min",
]
META_COLUMNS = ["datetime", "symbol"]
# Loader / pipeline internal columns — not trading features (excluded from Phase 1, etc.)
INTERNAL_COLUMNS = ("_symbol_bar_index",)
TAIL_DROP_ROWS = 288  # rows dropped per symbol (no labels available)

# ---------------------------------------------------------------------------
# Backtest Constants (must match evaluator_v3.ipynb exactly)
# ---------------------------------------------------------------------------
INITIAL_CAPITAL = 1000.0
LEVERAGE = 1.0
FEE_PCT = 0.20
MAX_HOLD_CANDLES = 288
MAX_TOTAL_EXPOSURE_PCT = 100.0
MIN_POSITION_NOTIONAL = 1.0

# ---------------------------------------------------------------------------
# Phase 2 Static Risk (isolates predictive alpha from risk tuning)
# ---------------------------------------------------------------------------
PHASE2_TP = 2.0
PHASE2_SL = 1.0
PHASE2_CAPITAL_PCT = 48.0

# ---------------------------------------------------------------------------
# Phase 2 Rule Constraints
# ---------------------------------------------------------------------------
MIN_CONDITIONS = 3
MAX_CONDITIONS = 4
# minimum number of trades across all symbols (after applying condition filters)
# Raised from 150 → 300 (~3 trades per symbol per month over 8 months) to drop
# noisy rules; pool analysis showed 14/18 long rules below 150.
MIN_TRADE_SUPPORT = 300
# Raised from 10 → 50 so the support penalty actually dominates noisy Sortino.
SUPPORT_PENALTY_MAX = 50.0
# Hard rejection floor for archive entries (~ MIN_TRADE_SUPPORT // 4).
MIN_TRADE_POOL_FLOOR = 75

# Saturating Sortino transform (used in evaluation): tanh(sortino / SORTINO_SCALE) * SORTINO_CAP.
# The previous flat cap pinned best_sortino at the sentinel from generation 0.
SORTINO_CAP = 5.0
SORTINO_SCALE = 3.0
# Train+Val joint objective: Phase 2 evaluates each chromosome on both splits and
# uses min(train, val) for f1 (worst-case Sortino) when ENABLED.
PHASE2_JOINT_TRAIN_VAL = True
# Hamming threshold below which the diversity penalty is applied (was: only ==0).
PHASE2_DIVERSITY_HAMMING_THRESHOLD = 2
PHASE2_DIVERSITY_PENALTY = 5.0
PHASE2_POPULATION_SIZE = 200
# Raised from 100 → 200; the history shows the Pareto front stops improving
# around gen 50 with the previous setting.
PHASE2_GENERATIONS = 200
PHASE2_ARCHIVE_MAX_SIZE = 500
# Fraction of Phase 2 population seeded from pools/phase2_{direction}_pool.json
PHASE2_ARCHIVE_SEED_FRACTION = 0.35
PHASE2_ALGORITHM = "NSGA3"  # NSGA-III

# ---------------------------------------------------------------------------
# Phase 3 Rule Set Selection
# ---------------------------------------------------------------------------
PHASE3_MIN_RULES = 2
PHASE3_MAX_RULES = 3
PHASE3_MIN_SYMBOL_COVERAGE = 7  # out of 10 symbols must have >= 1 trade
PHASE3_USE_GPU = False  # set True after GPU rule-set batch parity tests pass
PHASE3_REFINE_GENERATIONS = 80
PHASE3_REFINE_POP_SIZE = 100
PHASE3_GREEDY_WEIGHTS = (1.0, 0.7, 0.5)  # sortino, drawdown, win_rate
PHASE3_SYMBOL_CONSISTENCY_WEIGHT = 10.0
# --- Train-as-target / validation-as-gate (anti-leakage redesign) ---
# When enabled, Phase 3 optimizes objectives on TRAIN and uses validation only
# as a gate (rejecting candidates whose validation degrades disproportionately).
PHASE3_USE_TRAIN_TARGET = True
# Reject candidates whose val_sortino < ratio * train_sortino (when train > 0)
PHASE3_VAL_SORTINO_RATIO_GATE = 0.5
# Reject candidates whose validation drawdown > ratio * train drawdown
PHASE3_VAL_DRAWDOWN_RATIO_GATE = 1.5
# Per-rule per-symbol minimum trade count on validation; rejects rules that
# fire on too few symbols and would not survive a regime change.
PHASE3_PER_RULE_MIN_VAL_TRADES_PER_SYMBOL = 5
# Penalty weight for low corr(train_per_symbol_pnl, val_per_symbol_pnl).
PHASE3_TRAIN_VAL_CORR_WEIGHT = 5.0
# Soft-gate penalty for failing the validation gates (in addition to dominating).
PHASE3_VAL_GATE_PENALTY = 75.0

# ---------------------------------------------------------------------------
# Phase 2 MOME (deferred — future native 4×10 descriptor grid)
# ---------------------------------------------------------------------------
# Descriptors: n_active_conditions (2–5) × symbol_coverage (0–10).
# Each cell holds a local Pareto archive; Phase 3 samples one rule per cell.
# See plan milestone 4; not implemented in this release.

# ---------------------------------------------------------------------------
# Phase 4 RL Risk Optimization
# ---------------------------------------------------------------------------
PHASE4_RL_ALGORITHM = "DDPG"  # alternative: "PPO"
PHASE4_TP_MIN = 2.0
PHASE4_TP_MAX = 4.0
PHASE4_SL_MIN = 1.0
PHASE4_SL_MAX = 2.0
PHASE4_CAPITAL_PCT_MIN = 10.0
PHASE4_CAPITAL_PCT_MAX = 50.0
PHASE4_TOTAL_CAP_PENALTY = 2.0
PHASE4_RL_EVAL_WINDOW = 288
# Raised from 0.2 → 1.0: validation Sortino is now a primary signal, not a tiebreaker.
PHASE4_VAL_SORTINO_WEIGHT = 1.0
PHASE4_VAL_SORTINO_BONUS_CAP = 5.0
PHASE4_TOTAL_TIMESTEPS = 500_000
PHASE4_ELBOW_WINDOW = 15
# Hard normalize per-rule capital_pct so the sum never exceeds the limit.
# Phase 4 outputs were summing to 75% (long) and 108% (short) — short violates 100%.
PHASE4_HARD_CAP_NORMALIZE = True

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
# 0 = auto-throttle generation progress; positive int = log every N generations
LOG_GENERATION_INTERVAL = 0

# ---------------------------------------------------------------------------
# Phase 1 Feature Selection
# ---------------------------------------------------------------------------
PHASE1_DISPERSION_THRESHOLD = 0.95
# Lowered 25 → 15: scores below 0.01 are MI floor noise; padding the gene with
# 25 features lets the GA exploit noise.
PHASE1_TOP_K_FEATURES = 15
PHASE1_MAX_FEATURE_OVERLAP = 0.50
# Stationarity filter: per-fold MI (chronological or regime-stratified); drop
# features with high CV or unstable rank across folds.
PHASE1_STATIONARITY_FOLDS = 3
PHASE1_STATIONARITY_CV_MAX = 1.0
PHASE1_STATIONARITY_RANK_DRIFT_MAX = 30
# "regime" = cluster train rows by vol/trend/liquidity indicators; "chronological" = time folds
PHASE1_STATIONARITY_STRATIFY = "regime"
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
# when STRATIFY=regime, defaults to STATIONARITY_FOLDS if unset at runtime
PHASE1_REGIME_N_CLUSTERS = 3
PHASE1_REGIME_MIN_SAMPLES = 100
PHASE1_REGIME_CLUSTERER = "gmm"  # "gmm" | "kmeans"
PHASE1_REGIME_GMM_REG_COVAR = 1e-6
PHASE1_REGIME_ZERO_VAR_EPS = 1e-12
PHASE1_REGIME_MODEL_PATH = os.path.join(
    OUTPUTS_DIR, "phase1_regime_cluster.joblib")
# Asymmetric scoring target — instead of a binary success flag, use a signed
# expected-PnL surrogate so long/short feature lists actually differ.
PHASE1_ASYMMETRIC_TARGET = True
# Primary memory control knob for Phase 2. Raising this value increases JAX device array size proportionally. On WSL with limited GPU memory, keep at or below 150_000.
PHASE1_SAMPLING_TOTAL = 150_000

# Emergency RAM knobs (last resort; prefer code slimming in df_slim.py):
# PHASE2_POPULATION_SIZE = 100
# PHASE2_GENERATIONS = 50
