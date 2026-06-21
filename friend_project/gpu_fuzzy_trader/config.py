"""
Central configuration for the GPU-Fuzzy trading pipeline.

All tunable constants live here as module-level names so phases, tests, and
notebooks can import or override them (e.g. ``from gpu_fuzzy_trader import config as cfg``).

Sections follow the pipeline order: data paths → economics → Phase 1 … Phase 5 →
RB Governor → runtime helpers. Cross-phase settings (purged CV, monthly windows)
are grouped with the phase that primarily consumes them.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Project root (friend_project/)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir))

# ---------------------------------------------------------------------------
# Data paths & outputs
# ---------------------------------------------------------------------------

# Raw CSV inputs (relative to cwd when the pipeline is launched).
TRAIN_CSV_PATH = "data/train.csv"
TEST_CSV_PATH = "data/test.csv"

# Evaluator_v5 reference schema and OOS evaluation file (same files by default).
EVALUATOR_REFERENCE_SCHEMA_PATH = "data/train.csv"
EVALUATOR_EVALUATION_FILE_PATH = "data/test.csv"

# Chronological 75/25 split persisted as parquet for faster reloads.
TRAIN_75_PATH = "data/train_75.parquet"
VALIDATION_25_PATH = "data/validation_25.parquet"

# Run artefacts: logs, reports, and per-run output root.
OUTPUTS_DIR = "outputs"
RUN_LOG_PATH = os.path.join(OUTPUTS_DIR, "run.log")
REPORTS_DIR = "outputs/reports"

# Phase 2 rule pools, evolution history, and archived elites (absolute paths).
PHASE2_POOL_DIR = os.path.join(_PROJECT_ROOT, "pools")
PHASE2_POOL_PATHS = {
    "long": os.path.join(PHASE2_POOL_DIR, "phase2_long_pool.json"),
    "short": os.path.join(PHASE2_POOL_DIR, "phase2_short_pool.json"),
}
PHASE2_HISTORY_PATHS = {
    "long": os.path.join(PHASE2_POOL_DIR, "phase2_long_history.json"),
    "short": os.path.join(PHASE2_POOL_DIR, "phase2_short_history.json"),
}
PHASE2_ARCHIVE_DIR = os.path.join(_PROJECT_ROOT, "phase2_rule_archive")
PHASE2_ARCHIVE_PATHS = {
    "long": os.path.join(PHASE2_ARCHIVE_DIR, "phase2_long_archive.json"),
    "short": os.path.join(PHASE2_ARCHIVE_DIR, "phase2_short_archive.json"),
}

# ---------------------------------------------------------------------------
# Data schema
# ---------------------------------------------------------------------------

# Forward-looking label columns required for supervised fitness and backtests.
LABEL_COLUMNS = [
    "label_open_next",
    "label_close_288",
    "label_min_288",
    "label_max_288",
    "label_max_before_min",
]

# Index/metadata columns preserved through loading and splitting.
META_COLUMNS = ["datetime", "symbol"]
INTERNAL_COLUMNS = ("_symbol_bar_index",)

# Rows dropped from the tail of each symbol after sort (0 = keep all).
TAIL_DROP_ROWS = 0

# ---------------------------------------------------------------------------
# Trading economics (shared across backtest engines)
# ---------------------------------------------------------------------------

INITIAL_CAPITAL = 1000.0
LEVERAGE = 1.0
FEE_PCT = 0.20  # round-trip fee as percent of notional

# Maximum bars a position may remain open (must match label horizon).
MAX_HOLD_CANDLES = 288

# Portfolio exposure caps used by CPU/GPU simulators.
MAX_TOTAL_EXPOSURE_PCT = 100.0
MIN_POSITION_NOTIONAL = 1.0

# Default TP/SL/capital for evolved single rules before Phase 4 tuning.
PHASE2_TP = 2.0
PHASE2_SL = 1.0
PHASE2_CAPITAL_PCT = 30.0

# ---------------------------------------------------------------------------
# Rule structure & fitness guards (Phase 2+)
# ---------------------------------------------------------------------------

# Number of fuzzy conditions per chromosome.
MIN_CONDITIONS = 3
MAX_CONDITIONS = 5

# Minimum executed trades before a rule is considered statistically supported.
MIN_TRADE_SUPPORT = 100
SUPPORT_PENALTY_MAX = 10.0
MIN_TRADE_POOL_FLOOR = 38

# Cap Sortino used in multi-objective ranking to limit outlier influence.
SORTINO_CAP = 10.0

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

# Generations between progress log lines (0 = log every generation).
LOG_GENERATION_INTERVAL = 0

# ---------------------------------------------------------------------------
# Phase 1 — feature selection
# ---------------------------------------------------------------------------

# Drop features whose value dispersion falls below this threshold.
PHASE1_DISPERSION_THRESHOLD = 0.95
PHASE1_TOP_K_FEATURES = 25

# Reject feature sets with pairwise overlap above this Jaccard-like bound.
PHASE1_MAX_FEATURE_OVERLAP = 0.50

# Row budget for Phase 1 subsampling (also reused as Phase 2 row sample cap).
PHASE1_SAMPLING_TOTAL = 701_000
PHASE2_ROW_SAMPLE_TOTAL = PHASE1_SAMPLING_TOTAL

# Penalise features whose label correlation sign flips across CV folds.
PHASE1_SIGN_CONSISTENCY_ENABLED = True
PHASE1_SIGN_CONSISTENCY_FOLDS = 4
PHASE1_SIGN_CONSISTENCY_WEIGHT = 0.35
PHASE1_SIGN_CONSISTENCY_MIN_ABS_CORR = 1e-5

# ---------------------------------------------------------------------------
# Purged cross-validation (shared embargo settings)
# ---------------------------------------------------------------------------

PURGED_CV_ENABLED = True
PURGED_CV_N_SPLITS = 4
PURGED_CV_EMBARGO_CANDLES = MAX_HOLD_CANDLES
PURGED_CV_MIN_TRAIN_FRACTION = 0.45
PURGED_CV_MIN_VALID_ROWS = 5_000

# ---------------------------------------------------------------------------
# Phase 2 — rule-pool evolution (NSGA-III)
# ---------------------------------------------------------------------------

PHASE2_POPULATION_SIZE = 250
PHASE2_GENERATIONS = 50
PHASE2_ARCHIVE_MAX_SIZE = 500
PHASE2_ARCHIVE_SEED_FRACTION = 0.35
PHASE2_ALGORITHM = "NSGA3"

# --- Single-stage defaults (fallback when two-stage is off; see phase2_stage.py) ---

PHASE2_MUTATION_RATE = 0.10
PHASE2_MUTATION_WEIGHTED_ACTIVATE_PROB = 0.40
PHASE2_DIVERSITY_PENALTY = 5.0
PHASE2_DIVERSITY_HAMMING_THRESHOLD = 2
PHASE2_DIVERSITY_RECOVERY_MIN_UNIQUE_RATIO = 0.30
PHASE2_DIVERSITY_RECOVERY_INJECT_FRACTION = 0.30
PHASE2_DIVERSITY_RECOVERY_MUTATION_BOOST = 1.75
PHASE2_PLATEAU_EARLY_STOP_PATIENCE = 5
PHASE2_PLATEAU_EARLY_STOP_MIN_GENERATION = 3
PHASE2_EARLY_STOP_MIN_GENERATION = 25
PHASE2_RETURN_FLOOR_PCT = 0.0
PHASE2_POOL_REQUIRE_POSITIVE_SPLITS = False
PHASE2_USE_ROBUST_RETURN_OBJ = False

# --- Two-stage search: wide exploration (A) → refinement (B) ---

PHASE2_TWO_STAGE_ENABLED = True

PHASE2_STAGE_A_GENERATIONS = 30
PHASE2_STAGE_B_GENERATIONS = 20
PHASE2_STAGE_B_SEED_TOP_K = 30
PHASE2_STAGE_B_SEED_FRACTION = 0.30

# Stage A — higher mutation / diversity, relaxed trade support, robust return obj.
PHASE2_STAGE_A_MUTATION_RATE = 0.25
PHASE2_STAGE_A_MUTATION_WEIGHTED_ACTIVATE_PROB = 0.50
PHASE2_STAGE_A_DIVERSITY_PENALTY = 10.0
PHASE2_STAGE_A_DIVERSITY_HAMMING_THRESHOLD = 4
PHASE2_STAGE_A_DIVERSITY_RECOVERY_MIN_UNIQUE_RATIO = 0.35
PHASE2_STAGE_A_DIVERSITY_RECOVERY_INJECT_FRACTION = 0.35
PHASE2_STAGE_A_DIVERSITY_RECOVERY_MUTATION_BOOST = 2.0
PHASE2_STAGE_A_PLATEAU_EARLY_STOP_PATIENCE = 28
PHASE2_STAGE_A_PLATEAU_EARLY_STOP_MIN_GENERATION = 20
PHASE2_STAGE_A_EARLY_STOP_MIN_GENERATION = 22
PHASE2_STAGE_A_ARCHIVE_SEED_FRACTION = 0.20
PHASE2_STAGE_A_RETURN_FLOOR_PCT = 0.0
PHASE2_STAGE_A_MIN_TRADE_SUPPORT = 30
PHASE2_STAGE_A_USE_ROBUST_RETURN_OBJ = True
PHASE2_STAGE_A_SOFT_FEASIBILITY = True

# Stage B — lower mutation, tighter diversity, seeds from Stage A elites.
PHASE2_STAGE_B_MUTATION_RATE = 0.18
PHASE2_STAGE_B_MUTATION_WEIGHTED_ACTIVATE_PROB = 0.40
PHASE2_STAGE_B_DIVERSITY_PENALTY = 5.0
PHASE2_STAGE_B_DIVERSITY_HAMMING_THRESHOLD = 2
PHASE2_STAGE_B_DIVERSITY_RECOVERY_MIN_UNIQUE_RATIO = 0.25
PHASE2_STAGE_B_DIVERSITY_RECOVERY_INJECT_FRACTION = 0.20
PHASE2_STAGE_B_DIVERSITY_RECOVERY_MUTATION_BOOST = 1.4
PHASE2_STAGE_B_PLATEAU_EARLY_STOP_PATIENCE = 15
PHASE2_STAGE_B_PLATEAU_EARLY_STOP_MIN_GENERATION = 12
PHASE2_STAGE_B_EARLY_STOP_MIN_GENERATION = 15

# --- Per-symbol Phase 2 ---

# True  → evolve separate pools per symbol (rules tagged ``symbol is S``).
# False → single combined pool across all symbols.
PER_SYMBOL_PHASE2 = True
PER_SYMBOL_MIN_ROWS = 1000
PHASE2_PER_SYMBOL_POOL_DIR = os.path.join(PHASE2_POOL_DIR, "per_symbol")

# --- Population strata (elite / explorer / regime-aware) ---

PHASE2_THREE_STRATA_ENABLED = True
PHASE2_STRATA_ELITE_FRAC = 0.40
PHASE2_STRATA_EXPLORER_FRAC = 0.35
PHASE2_STRATA_REGIME_FRAC = 0.25
PHASE2_FEATURE_SOFTMAX_TEMPERATURE = 0.35
PHASE2_REGIME_FEATURE_KEYWORDS = (
    "vol", "atr", "bb_width", "compression", "range", "trend", "regime",
    "breakout", "drawdown", "channel", "adx", "dmi", "semivol",
)

# --- Archive & fallback when the live pool is thin ---

PHASE2_MULTI_ARCHIVE_ENABLED = True
PHASE2_MULTI_ARCHIVE_PER_METRIC = 40
PHASE2_MULTI_ARCHIVE_METRICS = (
    "sortino_ratio",
    "total_return_pct",
    "max_drawdown_pct",
    "win_rate",
    "executed_trades",
)
PHASE2_FALLBACK_ENABLED = True
PHASE2_FALLBACK_MIN_POOL_SIZE = 40
PHASE2_FALLBACK_MAX_RULES = 200

# --- Post-evolution CV filter on pool candidates ---

PHASE2_CV_FILTER_ENABLED = True
PHASE2_CV_MAX_RULES_TO_EVALUATE = 350
PHASE2_CV_MIN_WORST_RETURN = -8.0
PHASE2_CV_MIN_WORST_PF = 0.80
PHASE2_CV_MAX_WORST_DD = 18.0
PHASE2_CV_MIN_FOLD_TRADES = 10

# --- GPU backtest during evolution ---

# Chromosomes per JAX vmap chunk in simulate_rule_batch.
# Used directly when PHASE2_GPU_BATCH_SIZE_AUTO is False; otherwise VRAM/RAM
# tiers clamp at runtime (see _gpu_runtime.resolve_phase2_gpu_batch_size).
PHASE2_GPU_BATCH_SIZE = 198

# True  → cap batch size by detected GPU VRAM and host RAM.
# False → use PHASE2_GPU_BATCH_SIZE exactly (env override still applies).
PHASE2_GPU_BATCH_SIZE_AUTO = True

# lax.scan unroll for equity simulation (higher = longer compile, less loop overhead).
PHASE2_SCAN_UNROLL = 32

# float32 on GPU for Phase 2 ranking (CPU path remains reference precision).
PHASE2_GPU_USE_FP32 = True

# Store discretized feature matrix as int8 on device to save VRAM.
PHASE2_GPU_DATA_INT8 = True

# Master switch for JAX GPU backtest during Phase 2 evolution.
PHASE2_USE_GPU = True

# ---------------------------------------------------------------------------
# Condition support — penalise ultra-rare or overly broad fuzzy predicates
# ---------------------------------------------------------------------------

CONDITION_SUPPORT_ENABLED = True
CONDITION_SUPPORT_WINDOW_DAYS = 30
CONDITION_SUPPORT_ULTRA_RARE_PCT = 0.0005
CONDITION_SUPPORT_RARE_PCT = 0.005
CONDITION_SUPPORT_BROAD_PCT = 0.20
CONDITION_SUPPORT_VERY_BROAD_PCT = 0.40
CONDITION_SUPPORT_DROP_ULTRA_RARE = True
CONDITION_SUPPORT_RARE_WEIGHT = 0.35
CONDITION_SUPPORT_BROAD_WEIGHT = 0.45
CONDITION_SUPPORT_VERY_BROAD_WEIGHT = 0.15
CONDITION_SUPPORT_MIN_STABILITY_WEIGHT = 0.20

# ---------------------------------------------------------------------------
# Rule generation helpers (seeds for Phase 2 / Phase 3 populations)
# ---------------------------------------------------------------------------

RULE_GENERATION_ENABLED = True
RULE_GEN_TOP_SEEDS = 40
RULE_GEN_EXPANSIONS_PER_SEED = 2
RULE_GEN_MIN_FEATURE_CONDITIONS = 5
RULE_GEN_MAX_FEATURE_CONDITIONS = 9
RULE_GEN_MAX_AND_PAIRS = 100
RULE_GEN_MAX_BACKTEST_CANDIDATES = 320
RULE_GEN_MAX_RAW_SIGNALS = 9000
RULE_REPAIR_MIN_FEATURE_CONDITIONS = 4
RULE_REPAIR_MAX_FEATURE_CONDITIONS = 9
RULE_REPAIR_MAX_PER_FAMILY = 2
TEMPLATE_GENERATOR_MAX_RULES = 80
APRIORI_MAX_CONDITION_BANK = 140
APRIORI_MAX_SEED_RULES = 100
APRIORI_MIN_SINGLE_SUPPORT = 0.002
APRIORI_MAX_SINGLE_SUPPORT = 0.35
APRIORI_MIN_PAIR_SUPPORT = 0.001
APRIORI_MIN_LIFT_LIKE = 1.05

# ---------------------------------------------------------------------------
# Phase 3 — greedy rule-set assembly
# ---------------------------------------------------------------------------

PHASE3_MIN_RULES = 1
PHASE3_MAX_RULES = 5
PHASE3_MIN_SYMBOL_COVERAGE = 7
PHASE3_USE_GPU = False
PHASE3_REFINE_GENERATIONS = 0
PHASE3_REFINE_POP_SIZE = 0
PHASE3_GREEDY_WEIGHTS = (1.0, 0.7, 0.5)
PHASE3_SYMBOL_CONSISTENCY_WEIGHT = 10.0
PHASE3_SMART_POPULATION_ENABLED = True
PHASE3_RULE_CLUSTER_COUNT = 32

# Greedy early-stop when marginal score gain falls below threshold.
RETURN_DD_FLOOR = 1.0
PHASE3_GREEDY_EARLY_STOP = True
PHASE3_GREEDY_MIN_IMPROVEMENT = 0.0

# Purged-CV penalties applied while assembling the rule set.
PHASE3_PURGED_CV_ENABLED = True
PHASE3_CV_PENALTY_WEIGHT = 1.0
PHASE3_WORST_RETURN_FLOOR = -2.0
PHASE3_WORST_PF_FLOOR = 1.00
PHASE3_WORST_DD_CEIL = 12.0
PHASE3_MIN_FOLD_TRADES = 20

# Signal overlap and train/valid gap penalties between candidate rules.
PHASE3_SIGNAL_OVERLAP_ENABLED = True
PHASE3_MAX_PAIR_OVERLAP = 0.35
PHASE3_OVERLAP_WEIGHT = 18.0
PHASE3_GAP_RETURN_WEIGHT = 0.45
PHASE3_GAP_PF_WEIGHT = 3.00
PHASE3_GAP_DD_WEIGHT = 0.15

# Per-symbol survival: avoid sets dominated by one symbol's PnL.
PHASE3_PER_SYMBOL_SURVIVAL_ENABLED = True
PHASE3_MIN_PROFITABLE_SYMBOLS = 6
PHASE3_MAX_SINGLE_SYMBOL_PNL_SHARE = 0.45
PHASE3_PER_SYMBOL_WEIGHT = 8.0

# Hidden rolling validation windows (not used for training labels).
PHASE3_HIDDEN_VALIDATION_ENABLED = True
PHASE3_HIDDEN_WINDOW_DAYS = 90
PHASE3_HIDDEN_STRIDE_DAYS = 30
PHASE3_HIDDEN_MAX_WINDOWS = 6
PHASE3_HIDDEN_MIN_ROWS = 10_000
PHASE3_HIDDEN_WORST_RETURN_FLOOR = -5.0
PHASE3_HIDDEN_WORST_RETURN_WEIGHT = 0.35
PHASE3_HIDDEN_SORTINO_WEIGHT = 0.20
PHASE3_HIDDEN_DRAWDOWN_WEIGHT = 0.05

# Minimum rules retained after Phase 5 negative-PnL pruning safeguard.
PHASE3_GLOBAL_MIN_RULES = 2

# ---------------------------------------------------------------------------
# Monthly rolling validation (Phase 3 penalties & Phase 4 scoring)
# ---------------------------------------------------------------------------

MONTHLY_VALIDATION_ENABLED = True
MONTHLY_WINDOW_DAYS = 30
MONTHLY_WINDOW_STRIDE_DAYS = 30
MONTHLY_WINDOW_MIN_ROWS = 2500
MONTHLY_WINDOW_MAX_WINDOWS = 36
MONTHLY_RECENCY_WEIGHT = 2.2
MONTHLY_MIN_TRADES = 20
MONTHLY_MIN_PROFITABLE_RATIO = 0.60
MONTHLY_WORST_RETURN_FLOOR = -1.5
MONTHLY_WORST_PF_FLOOR = 0.85
MONTHLY_MAX_DD = 8.0
MONTHLY_WORST_RETURN_WEIGHT = 1.2
MONTHLY_WORST_PF_WEIGHT = 8.0
MONTHLY_DD_WEIGHT = 0.7
MONTHLY_PROFITABLE_RATIO_WEIGHT = 15.0
MONTHLY_TREND_WEIGHT = 2.0
MONTHLY_LATEST_WEIGHT = 0.6
PHASE3_MONTHLY_PENALTY_WEIGHT = 1.0
PHASE4_MONTHLY_SCORE_WEIGHT = 0.70

# ---------------------------------------------------------------------------
# Symbol specialization (post-process rules for symbol-specific filters)
# ---------------------------------------------------------------------------

SYMBOL_SPECIALIZATION_ENABLED = True
SYMBOL_SPECIALIZATION_MAX_SYMBOLS_PER_RULE = 3
SYMBOL_SPECIALIZATION_MIN_TRADES = 20
SYMBOL_SPECIALIZATION_MIN_PF = 1.02
SYMBOL_SPECIALIZATION_MIN_SCORE = -2.0

# ---------------------------------------------------------------------------
# Phase 4 — risk parameter search (grid and optional RL)
# ---------------------------------------------------------------------------

# RL-based tuner (used when the RL code path is active).
PHASE4_RL_ALGORITHM = "DDPG"
PHASE4_TP_MIN = 2.0
PHASE4_TP_MAX = 5.0
PHASE4_SL_MIN = 1.0
PHASE4_SL_MAX = 2.5
PHASE4_CAPITAL_PCT_MIN = 1.0
PHASE4_CAPITAL_PCT_MAX = 12.5
PHASE4_TOTAL_CAP_PENALTY = 2.0
PHASE4_RL_EVAL_WINDOW = 288
PHASE4_VAL_SORTINO_WEIGHT = 0.2
PHASE4_VAL_SORTINO_BONUS_CAP = 5.0
PHASE4_TOTAL_TIMESTEPS = 500_000
PHASE4_ELBOW_WINDOW = 15

# Robust grid search over TP / SL / capital (default Phase 4 path).
PHASE4_OPTIMIZER = "robust_grid"
PHASE4_TP_GRID = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
PHASE4_SL_GRID = (0.8, 1.0, 1.2, 1.5, 2.0)
PHASE4_CAPITAL_GRID = (1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 12.5)
PHASE4_ROBUST_MAX_TRIALS = 180
PHASE4_ROBUST_MIN_WORST_PF = 0.65
PHASE4_ROBUST_MAX_WORST_DD = 25.0
PHASE4_ROBUST_MIN_FOLD_TRADES = 5
PHASE4_ROBUST_EXPOSURE_WEIGHT = 0.002
PHASE4_MAX_TOTAL_CAPITAL = 35.0

# ---------------------------------------------------------------------------
# Phase 5 — out-of-sample evaluator
# ---------------------------------------------------------------------------

# Remove rules with Net_PnL <= 0 on test before writing final strategy JSON.
PHASE5_REMOVE_NEGATIVE_PNL_RULES = True

# Write stripped strategy JSON (direction + rules_set) for evaluator_v5 import.
WRITE_EVALUATOR_CLEAN = True

# ---------------------------------------------------------------------------
# Auto-search orchestration
# ---------------------------------------------------------------------------

AUTO_SEARCH_RUNS = 2
AUTO_SEARCH_START_DIRECTION = "long"
AUTO_SEARCH_OUTPUT_ROOT = os.path.join(OUTPUTS_DIR, "auto_search")
AUTO_SEARCH_HOURS = 24.0
AUTO_SEARCH_RUN_FINAL_TEST = False
AUTO_SEARCH_SCORE_MIN_TRADES = 60

# ---------------------------------------------------------------------------
# RB Governor — rule-bank composition & evaluator_v5 alignment
# ---------------------------------------------------------------------------

RB_ENGINE_ENABLED = True

# Default risk params for RB single-rule backtests before grid search.
RB_DEFAULT_TP = 2.0
RB_DEFAULT_SL = 1.2
RB_DEFAULT_CAPITAL_PCT = 12.5

# Per-rule train/valid acceptance floors.
RB_MIN_TRAIN_RETURN = 0.0
RB_MIN_VALID_RETURN = 0.0
RB_MIN_TRAIN_PF = 1.00
RB_MIN_VALID_PF = 1.00
RB_MIN_TRAIN_TRADES = 20
RB_MIN_VALID_TRADES = 12

# Rule-set composition limits and scoring.
RB_RULESET_MIN_TRAIN_TRADES = 30
RB_RULESET_MIN_VALID_TRADES = 20
RB_MAX_POOL_RULES_TO_EVALUATE = 900
RB_KEEP_TOP_RULES = 140
RB_MAX_RULES = 5
RB_MAX_PAIR_OVERLAP = 0.24
RB_RULESET_MUST_BEAT_SUBSETS = True
RB_MIN_SCORE_IMPROVEMENT = 0.05
RB_MIN_TRAIN_RETURN_IMPROVEMENT = 0.01
RB_MIN_VALID_RETURN_IMPROVEMENT = 0.01
RB_RETURN_DD_FLOOR = 0.50
RB_TRADE_PENALTY = 0.70
RB_TRAIN_VALID_RATIO_GAP_WEIGHT = 6.0
RB_TRAIN_VALID_RETURN_GAP_WEIGHT = 0.25

# Train should modestly outperform valid (anti-overfit shape bonus / penalties).
RB_REQUIRE_TRAIN_SLIGHTLY_ABOVE_VALID = True
RB_TRAIN_VALID_MIN_RATIO = 1.03
RB_TRAIN_VALID_MAX_RATIO = 1.35
RB_TRAIN_VALID_MIN_ABS_GAP = 0.20
RB_TRAIN_VALID_MAX_ABS_GAP = 12.0
RB_TRAIN_BELOW_VALID_PENALTY = 900.0
RB_TRAIN_TOO_HIGH_PENALTY = 220.0
RB_TRAIN_VALID_SHAPE_BONUS = 160.0

# Per-rule and global risk grids searched by the RB Governor.
RB_TP_GRID = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0)
RB_SL_GRID = (1.0, 1.5, 2.0, 2.5, 3.0)
RB_CAPITAL_GRID = (15.0, 20.0, 25.0, 35.0, 40.0)
RB_MAX_TOTAL_CAPITAL = 100.0
RB_RISK_OPT_PASSES = 2
RB_RISK_MIN_IMPROVEMENT = 0.02

# Evaluator_v5 contract: scoring, TP/SL floors, execution-ratio guards.
RB_EVALUATOR_ORDER_COMPAT = True
RB_USE_EVALUATOR_V5_SCORING = True
RB_USE_EVALUATOR_V5_FILES = True
RB_REQUIRE_TP_SL_ABOVE_ONE = True
RB_MIN_TP = 1.5
RB_MIN_SL = 1.0
RB_MAX_SKIPPED_SIGNAL_RATIO = 0.20
RB_MIN_EXECUTED_RAW_RATIO = 0.60
RB_SKIPPED_RATIO_PENALTY = 3500.0
RB_EXECUTED_RATIO_PENALTY = 2500.0
RB_MAX_SIMULTANEOUS_POSITIONS = 10
RB_MAX_POSITIONS_PENALTY = 120.0

# Greedy rule-add heuristics during RB composition.
RB_RULE_ADD_BY_RETURN_ONLY = True
RB_RULE_ADD_IGNORE_OVERLAP = True
RB_RULE_ADD_IGNORE_SUBSET_BEAT = True
RB_MIN_COMBINED_RETURN_IMPROVEMENT = 0.05

# Cross-run global rule bank (merge elites from multiple auto-search runs).
RB_GLOBAL_BANK_ENABLED = True
RB_GLOBAL_COMPOSE_AFTER_EACH_RUN = True
RB_GLOBAL_BANK_DIRNAME = "rb_bank"
RB_GLOBAL_BANK_MAX_RULES_PER_DIRECTION = 700
RB_GLOBAL_BANK_IMPORT_TOP_SINGLE_RULES = 80
RB_GLOBAL_MAX_RULES = 12
RB_GLOBAL_MIN_COMBINED_RETURN_IMPROVEMENT = 0.05
RB_GLOBAL_REQUIRE_POSITIVE_TRAIN_VALID = True
RB_GLOBAL_RISK_OPT_PASSES = 2
RB_GLOBAL_BEST_DIRNAME = "best_global"
RB_GLOBAL_TP_GRID = (1.5, 2.0, 3.0, 5.0, 8.0)
RB_GLOBAL_SL_GRID = (1.2, 1.5, 2.0, 2.5)
RB_GLOBAL_CAPITAL_GRID = (5.0, 12.5, 25.0, 50.0)
RB_GLOBAL_MAX_TOTAL_CAPITAL = 100.0

# Profit amplifier — optional second pass to boost return while guarding DD/monthly stats.
RB_PROFIT_AMPLIFIER_ENABLED = True
RB_PROFIT_AMP_MAX_CANDIDATES = 90
RB_PROFIT_AMP_MAX_RULES = 5
RB_PROFIT_AMP_MIN_OBJECTIVE_IMPROVEMENT = 0.10
RB_PROFIT_AMP_MIN_RETURN_IMPROVEMENT = 0.05
RB_PROFIT_AMP_VALID_WEIGHT = 1.55
RB_PROFIT_AMP_TRAIN_WEIGHT = 1.00
RB_PROFIT_AMP_BALANCE_WEIGHT = 0.20
RB_PROFIT_AMP_DD_WEIGHT = 0.02
RB_PROFIT_AMP_HEALTH_WEIGHT = 0.030
RB_PROFIT_AMP_OVERLAP_PENALTY = 2.5
RB_PROFIT_AMP_MAX_PAIR_OVERLAP = 0.55
RB_PROFIT_AMP_MAX_VALID_DD = 12.0
RB_PROFIT_AMP_MAX_TRAIN_DD = 18.0
RB_PROFIT_AMP_MONTHLY_ENABLED = True
RB_PROFIT_AMP_MIN_MONTHLY_WINDOWS = 2
RB_PROFIT_AMP_MIN_MONTHLY_PROFITABLE_RATIO = 0.55
RB_PROFIT_AMP_WORST_MONTHLY_RETURN_FLOOR = -2.0
RB_PROFIT_AMP_WORST_MONTHLY_PF_FLOOR = 0.80
RB_PROFIT_AMP_MAX_MONTHLY_DD = 10.0
RB_PROFIT_AMP_CAPITAL_REALLOCATION_ENABLED = True
RB_PROFIT_AMP_CAPITAL_PASSES = 2
RB_PROFIT_AMP_CAPITAL_GRID = RB_CAPITAL_GRID
RB_PROFIT_AMP_KEEP_BASELINE_UNLESS_BETTER = True

# Symbol filters required on every RB output rule (evaluator_v5 contract).
RB_REQUIRE_SYMBOL_FILTERS = True
RB_SYMBOL_MAX_SYMBOLS_PER_RULE = 3
RB_SYMBOL_TOP_SINGLE_SYMBOLS = 5
RB_SYMBOL_MAX_VARIANTS_PER_RULE = 10
RB_SYMBOL_MIN_TRAIN_TRADES = 10
RB_SYMBOL_MIN_VALID_TRADES = 6
RB_SYMBOL_USE_COMBINATIONS = True
RB_SYMBOL_STRICT_OUTPUT_CHECK = True

# ---------------------------------------------------------------------------
# Config invariants (fail fast on inconsistent stage / mutation settings)
# ---------------------------------------------------------------------------

assert PHASE2_STAGE_A_GENERATIONS + PHASE2_STAGE_B_GENERATIONS == PHASE2_GENERATIONS, (
    "Stage A + Stage B generations must equal PHASE2_GENERATIONS"
)
assert PHASE2_STAGE_A_PLATEAU_EARLY_STOP_MIN_GENERATION <= PHASE2_STAGE_A_GENERATIONS
assert PHASE2_STAGE_B_PLATEAU_EARLY_STOP_MIN_GENERATION <= PHASE2_STAGE_B_GENERATIONS
assert PHASE2_STAGE_A_EARLY_STOP_MIN_GENERATION <= PHASE2_STAGE_A_GENERATIONS
assert PHASE2_STAGE_B_EARLY_STOP_MIN_GENERATION <= PHASE2_STAGE_B_GENERATIONS
assert 0.0 < PHASE2_STAGE_A_MUTATION_RATE <= 0.5
assert 0.0 < PHASE2_STAGE_B_MUTATION_RATE <= 0.5
assert PHASE2_STAGE_A_MUTATION_RATE >= PHASE2_STAGE_B_MUTATION_RATE


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------

def is_colab_runtime() -> bool:
    """True when running on Google Colab (/content runtime)."""
    import importlib.util
    return (
        importlib.util.find_spec("google.colab") is not None
        or os.environ.get("COLAB_RELEASE_TAG") is not None
    )


def _apply_colab_gpu_defaults() -> None:
    """
    Colab T4 optimizations for notebook runs.

    VRAM auto batch sizing uses the T4-friendly 128 cap when enabled.
    """
    global PHASE2_GPU_BATCH_SIZE_AUTO
    if not is_colab_runtime():
        return
    PHASE2_GPU_BATCH_SIZE_AUTO = True


_apply_colab_gpu_defaults()
