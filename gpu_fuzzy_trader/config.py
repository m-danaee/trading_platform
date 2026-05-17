"""
config.py — Single source of truth for all hyperparameters.

No module may define its own defaults that override these values.
All paths, constants, and behavioral settings live here.
"""

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRAIN_CSV_PATH = "data/train.csv"
TEST_CSV_PATH = "data/test.csv"
TRAIN_75_PATH = "data/train_75.parquet"
VALIDATION_25_PATH = "data/validation_25.parquet"
OUTPUTS_DIR = "outputs"
REPORTS_DIR = "outputs/reports"

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
PHASE2_TP = 4.0
PHASE2_SL = 2.0
PHASE2_CAPITAL_PCT = 50.0

# ---------------------------------------------------------------------------
# Phase 2 Rule Constraints
# ---------------------------------------------------------------------------
MIN_CONDITIONS = 2
MAX_CONDITIONS = 5
MIN_TRADE_SUPPORT = 20
PHASE2_POPULATION_SIZE = 200
PHASE2_GENERATIONS = 500
PHASE2_ALGORITHM = "MOEAD"  # alternatives: "MOPSO", "RVEA"

# ---------------------------------------------------------------------------
# Phase 3 Rule Set Selection
# ---------------------------------------------------------------------------
PHASE3_POPULATION_SIZE = 100
PHASE3_GENERATIONS = 200
PHASE3_MIN_RULES = 2
PHASE3_MAX_RULES = 5
PHASE3_MIN_SYMBOL_COVERAGE = 7  # out of 10 symbols must have >= 1 trade

# ---------------------------------------------------------------------------
# Phase 4 RL Risk Optimization
# ---------------------------------------------------------------------------
PHASE4_RL_ALGORITHM = "DDPG"  # alternative: "PPO"
PHASE4_TP_MIN = 1.0
PHASE4_TP_MAX = 10.0
PHASE4_SL_MIN = 0.5
PHASE4_SL_MAX = 5.0
PHASE4_CAPITAL_PCT_MIN = 10.0
PHASE4_CAPITAL_PCT_MAX = 100.0
PHASE4_TOTAL_TIMESTEPS = 500_000
PHASE4_ELBOW_WINDOW = 20

# ---------------------------------------------------------------------------
# Phase 1 Feature Selection
# ---------------------------------------------------------------------------
PHASE1_DISPERSION_THRESHOLD = 0.95
PHASE1_TOP_K_FEATURES = 30
PHASE1_SAMPLING_TOTAL = 300_000  # shared across symbols (e.g. 30k per symbol for 10)
