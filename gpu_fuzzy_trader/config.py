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
REPORTS_DIR = "outputs/reports"
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
MIN_TRADE_SUPPORT = 150
SUPPORT_PENALTY_MAX = 10.0
MIN_TRADE_POOL_FLOOR = 38  # archive hard filter (~ MIN // 2)

# Maximum Sortino ratio (prevents sentinel-driven Pareto distortion)
SORTINO_CAP = 10.0
PHASE2_POPULATION_SIZE = 200
PHASE2_GENERATIONS = 100
PHASE2_ARCHIVE_MAX_SIZE = 500
PHASE2_ARCHIVE_SEED_FRACTION = 0.35
PHASE2_ALGORITHM = "NSGA3"  # NSGA-III

# ---------------------------------------------------------------------------
# Phase 3 Rule Set Selection
# ---------------------------------------------------------------------------
PHASE3_MIN_RULES = 2
PHASE3_MAX_RULES = 3
PHASE3_MIN_SYMBOL_COVERAGE = 7  # out of 10 symbols must have >= 1 trade
PHASE3_USE_GPU = False  # set True after GPU rule-set batch parity tests pass
PHASE3_REFINE_GENERATIONS = 40
PHASE3_REFINE_POP_SIZE = 100
PHASE3_GREEDY_WEIGHTS = (1.0, 0.7, 0.5)  # sortino, drawdown, win_rate
PHASE3_SYMBOL_CONSISTENCY_WEIGHT = 10.0

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
PHASE4_TP_MAX = 5.0
PHASE4_SL_MIN = 1.0
PHASE4_SL_MAX = 2.5
PHASE4_CAPITAL_PCT_MIN = 10.0
PHASE4_CAPITAL_PCT_MAX = 50.0
PHASE4_TOTAL_CAP_PENALTY = 2.0
PHASE4_RL_EVAL_WINDOW = 288
PHASE4_VAL_SORTINO_WEIGHT = 0.2
PHASE4_VAL_SORTINO_BONUS_CAP = 5.0
PHASE4_TOTAL_TIMESTEPS = 500_000
PHASE4_ELBOW_WINDOW = 15

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
# 0 = auto-throttle generation progress; positive int = log every N generations
LOG_GENERATION_INTERVAL = 0

# ---------------------------------------------------------------------------
# Phase 1 Feature Selection
# ---------------------------------------------------------------------------
PHASE1_DISPERSION_THRESHOLD = 0.95
PHASE1_TOP_K_FEATURES = 25
PHASE1_MAX_FEATURE_OVERLAP = 0.50
# Rows sampled per Phase 2 backtest engine (distributed across symbols).
# Emergency RAM knob if OOM persists after df slimming: try 150_000.
PHASE1_SAMPLING_TOTAL = 300_000

# Emergency RAM knobs (last resort; prefer code slimming in df_slim.py):
# PHASE2_POPULATION_SIZE = 100
# PHASE2_GENERATIONS = 50
