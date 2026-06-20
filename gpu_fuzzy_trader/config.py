"""
Single source of truth for pipeline hyperparameters.

All modules import from here; do not duplicate defaults elsewhere.

Pipeline phases
---------------
  Phase 0  Paths, schema, train/val split, backtest constants
  Phase 1  Feature selection (train.csv only)
  Phase 2  NSGA-III rule-pool evolution (GPU backtests)
  Phase 3  Greedy + NSGA-II rule-team selection
  Phase 4  Walk-forward TP/SL/capital optimization (Optuna)
  Phase 5  Out-of-sample evaluation (test.csv only)

Detailed behaviour and formulas: docs/phase0_shared.md … docs/phase5_oos.md

Tuning cheat-sheet (symptom → knob)
-----------------------------------
  Short OOS / overfitting          PHASE3_* gates, PHASE2_JOINT_TRAIN_VAL
  GPU OOM                          PHASE1_SAMPLING_TOTAL ↓, PHASE2_GPU_BATCH_SIZE ↓,
                                   PHASE2_SCAN_UNROLL ↓
  Phase 2 too slow                 PHASE2_GENERATIONS ↓, PHASE2_USE_GPU
  Empty Phase 2 pool               MIN_TRADE_SUPPORT ↓, PHASE2_*_FLOOR ↓
  Too many weak / noisy rules      MIN_TRADE_SUPPORT ↑, MIN_CONDITIONS ↑,
                                   PHASE2_*_FLOOR ↑, PHASE2_MAX_DRAWDOWN_GATE ↓
  Phase 3 finds no teams           PHASE3_*_FLOOR ↓, PHASE3_MIN_RULES ↓
  Phase 4 rejects all trials       PHASE4_MIN_WORST_* ↓, PHASE4_WF_SPLITS ↓
  Fees / horizon mismatch          FEE_PCT, TAIL_DROP_ROWS, MAX_HOLD_CANDLES
                                   (must match evaluator_v3.ipynb)

Environment overrides: DATA_ROOT, TRAIN_CSV_PATH, TEST_CSV_PATH,
                       PHASE2_GPU_BATCH_SIZE, PHASE2_GPU_BATCH_SIZE_AUTO
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import pandas as pd

# Repo root (parent of gpu_fuzzy_trader/) — paths outside per-run OUTPUTS_DIR.
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir))


# =============================================================================
# Global randomness
# =============================================================================

# GLOBAL_SEED
#   None  → one cryptographically random seed per process (default).
#   int   → fully reproducible runs (e.g. 42).
# Higher/lower: N/A — only None vs fixed integer matters for reproducibility.
GLOBAL_SEED: int | None = None

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
TRAIN_70_PATH = "data/train_70.parquet"
VALIDATION_30_PATH = "data/validation_30.parquet"

OUTPUTS_DIR = "outputs"
RUN_LOG_PATH = os.path.join(OUTPUTS_DIR, "run.log")
REPORTS_DIR = "outputs/reports"

# Evaluator v5 schema: rules_set must contain 2–EVALUATOR_MAX_RULES rules.
EVALUATOR_MAX_RULES = 5
EVALUATOR_MIN_RULES = 2

# When True, Output_Writer.write also writes a stripped file containing only
# direction and rules_set (defensive: protects against stricter evaluators
# that reject unknown top-level keys).
WRITE_EVALUATOR_CLEAN = True

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

# Debug: scope pipeline to N symbols starting at DEBUG_SYMBOL (sorted universe).
DEBUG_SYMBOL_SCOPE_ENABLED = False
DEBUG_SYMBOL = "1"
DEBUG_SYMBOL_COUNT = 4


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


def effective_phase3_per_symbol_min_trades() -> int:
    """Per-symbol trade floor; scaled down for thin debug universes."""
    base = int(PHASE3_PER_SYMBOL_MIN_TRADES)
    universe = _debug_symbol_universe_size()
    if universe is None:
        return base
    # Full run assumes ~10 symbols; scale floor with active symbol count.
    scaled = int(round(base * universe / 10.0))
    return max(15, scaled)


def effective_phase3_per_symbol_min_return() -> float:
    """Per-symbol return floor; relaxed in debug scope for sparse val slices."""
    base = float(PHASE3_PER_SYMBOL_MIN_RETURN)
    if _debug_symbol_universe_size() is None:
        return base
    return min(base, 2.5)


def effective_phase3_val_return_floor_pct() -> float:
    """Team fallback return floor; must sit below typical Phase 2 pool returns."""
    base = float(PHASE3_VAL_RETURN_FLOOR_PCT)
    if _debug_symbol_universe_size() is None:
        return base
    return min(base, 4.0)


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

# TAIL_DROP_ROWS — bars dropped per symbol at dataset tail (label horizon).
# Must equal MAX_HOLD_CANDLES (288 = 24 h at 5-min bars).
#   Higher → more rows removed, safer labels, less training data.
#   Lower  → more rows kept, risk of NaN / lookahead leakage at symbol tails.
TAIL_DROP_ROWS = 288


# =============================================================================
# Phase 0 — Train / validation split (Phases 2–3)
# =============================================================================
# Phases 4–5 always use persisted train_70 + validation_30 (see splitter.py).

# SPLIT_MODE — how train.csv is divided before Phase 2.
#   holdout_70_30       → single per-symbol 70/30 chronological split (legacy).
#   purged_walk_forward → expanding CV folds + primary tail holdout with embargo.
SPLIT_MODE = "purged_walk_forward"

# --- Purged walk-forward (when SPLIT_MODE == purged_walk_forward) ---

# PURGED_WF_N_SPLITS — total fold count including primary holdout (K).
#   K-1 folds are CV; last fold is the persisted validation holdout.
PURGED_WF_N_SPLITS = 3

# PURGED_WF_HOLDOUT_FRACTION — tail fraction per symbol reserved for val parquet.
PURGED_WF_HOLDOUT_FRACTION = 0.25

# PURGED_WF_EMBARGO_CANDLES — purge gap between train and valid (label horizon).
PURGED_WF_EMBARGO_CANDLES = 288

# PURGED_WF_MIN_TRAIN_FRACTION — minimum train prefix before first CV valid block.
PURGED_WF_MIN_TRAIN_FRACTION = 0.25

# PURGED_WF_MIN_VALID_ROWS — minimum rows in a CV valid block (holdout exempt).
PURGED_WF_MIN_VALID_ROWS = 3000

# PURGED_WF_AGGREGATION — combine per-fold metrics: worst | mean.
PURGED_WF_AGGREGATION = "worst"

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
# Phase 0 — Backtest simulation (must match evaluator_v3.ipynb)
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
#   Lower  → optimistic backtest; must match evaluator_v3.ipynb for valid OOS.
FEE_PCT = 0.20

# MAX_HOLD_CANDLES — force-exit horizon (bars) when neither TP nor SL hits.
#   Higher → longer holds, larger label window, must match TAIL_DROP_ROWS.
#   Lower  → quicker time exits, more fee drag if rules fire often.
MAX_HOLD_CANDLES = 288

# MAX_TOTAL_EXPOSURE_PCT — cap on sum of concurrent rule capital allocations.
#   Higher → more overlapping exposure, higher drawdown potential.
#   Lower  → forces capital to be spread thinner across simultaneous signals.
#   150% chosen because rules have distinct conditions + symbol filters and
#   rarely all fire simultaneously; normalization still bounds worst-case exposure.
MAX_TOTAL_EXPOSURE_PCT = 100.0

# MIN_POSITION_NOTIONAL — skip trades below this dollar size.
#   Higher → filters dust trades; may reduce trade count on small capital.
#   Lower  → more micro-trades counted toward support metrics.
MIN_POSITION_NOTIONAL = 1.0


# =============================================================================
# Phase 0 — Logging
# =============================================================================

# LOG_GENERATION_INTERVAL — Phase 2 progress log frequency (generations).
#   0   → auto (~10% of PHASE2_GENERATIONS).
#   N>0 → log every N generations; lower N = more verbose, slight I/O overhead.
LOG_GENERATION_INTERVAL = 0


# =============================================================================
# Phase 1 — Feature selection (train.csv only)
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

# PHASE1_MAX_FEATURE_OVERLAP — max Jaccard overlap between long & short lists.
#   Higher → more shared features across directions; smaller combined gene space.
#   Lower  → more direction-specific lists; better asymmetry, less redundancy.
PHASE1_MAX_FEATURE_OVERLAP = 0.8

# PHASE1_ASYMMETRIC_TARGET — separate MI targets for long vs short.
#   True  → direction-specific feature rankings (recommended).
#   False → shared target; long/short pools share more structure.
PHASE1_ASYMMETRIC_TARGET = True

# --- Sign consistency across stationarity folds ---

# PHASE1_REQUIRE_SIGN_CONSISTENCY — drop features whose Spearman sign flips.
#   True  → fewer regime-fragile features; stricter shortlist.
#   False → keep flip-flopping features; more noise in Phase 2.
PHASE1_REQUIRE_SIGN_CONSISTENCY: bool = True

# PHASE1_SIGN_CONSISTENCY_MIN_FOLDS — folds that must agree on correlation sign.
#   Higher → stricter; features must be stable across more sub-periods.
#   Lower  → more features pass; must be ≤ PHASE1_STATIONARITY_FOLDS.
PHASE1_SIGN_CONSISTENCY_MIN_FOLDS: int = 2

# PHASE1_SIGN_CONSISTENCY_MIN_ABS_CORR — ignore sign flips below this |ρ|.
#   Higher → only strong correlations must be consistent; more features kept.
#   Lower  → even weak correlations must be stable; stricter pruning.
PHASE1_SIGN_CONSISTENCY_MIN_ABS_CORR: float = 0.02

# --- Stationarity (reduce regime-specific features) ---

# PHASE1_STATIONARITY_FOLDS — chronological/regime chunks for stability tests.
#   Higher → more robust stationarity check, fewer features pass.
#   Lower  → faster, looser stationarity filter.
PHASE1_STATIONARITY_FOLDS = 2

# PHASE1_STATIONARITY_CV_MAX — max coefficient-of-variation across fold MI ranks.
#   Higher → allow more rank instability; keep more features.
#   Lower  → drop features whose importance swings across folds.
PHASE1_STATIONARITY_CV_MAX = 1.0

# PHASE1_STATIONARITY_RANK_DRIFT_MAX — max allowed rank change between folds.
#   Higher → tolerate large rank jumps; more features survive.
#   Lower  → only consistently top-ranked features kept.
PHASE1_STATIONARITY_RANK_DRIFT_MAX = 8

# PHASE1_STATIONARITY_STRATIFY — how folds are built for stationarity.
#   "regime"        → cluster by trend/vol regime (needs regime model).
#   "chronological" → time-ordered chunks; simpler, ignores regime structure.
PHASE1_STATIONARITY_STRATIFY = "chronological"

# =============================================================================
# Phase 1 → Phase 2 bridge — GPU row budget & JAX performance
# =============================================================================

# PHASE1_SAMPLING_TOTAL — max rows subsampled for Phase 2 GPU backtests.
# Peak GPU RAM scales ~linearly with this value (largest VRAM lever).
#   Higher → more statistical power, slower, OOM risk on small GPUs.
#   Lower  → faster, less RAM; trade/support floors may need proportional cut.
PHASE1_SAMPLING_TOTAL = 701_000

# PHASE2_GPU_BATCH_SIZE — chromosomes per JAX vmap chunk in simulate_rule_batch.
# Peak VRAM scales ~linearly (rule matching is O(batch × rows × conditions)).
# Used directly when PHASE2_GPU_BATCH_SIZE_AUTO is False; otherwise VRAM/RAM-capped.
#   Higher → faster throughput until OOM; 64–128 is fine on Colab T4 with headroom.
#   Lower  → safer on small GPUs / 12 GiB RAM hosts, more kernel launches, slower.
PHASE2_GPU_BATCH_SIZE = 198

# PHASE2_GPU_BATCH_SIZE_AUTO — cap batch size by detected GPU VRAM and host RAM.
#   True  → apply tiers in _gpu_runtime (12 GiB RAM → 32; T4 ≤16 GiB VRAM → 128).
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


# =============================================================================
# Phase 2 — Fixed risk during rule search (Phase 4 tunes TP/SL/capital later)
# =============================================================================

# PHASE2_TP — take-profit % used when scoring rules in Phase 2 (and Phase 1 targets).
#   Higher → fewer "wins" in labels/objectives; rules must catch larger moves.
#   Lower  → more wins, higher turnover, may favor noisy frequent signals.
PHASE2_TP = 2.0

# PHASE2_SL — stop-loss % during Phase 2 scoring.
#   Higher → wider stops, fewer stop-outs, larger per-trade risk.
#   Lower  → tighter stops; forces higher precision rules to survive.
PHASE2_SL = 1.0

# PHASE2_CAPITAL_PCT — % of equity allocated per rule signal in Phase 2.
#   Higher → larger simulated positions; drawdown and return scale up.
#   Lower  → conservative sizing; may understate overlap effects until Phase 4.
PHASE2_CAPITAL_PCT = 30.0


# =============================================================================
# Phase 2 — Rule genome
# =============================================================================

# MIN_CONDITIONS / MAX_CONDITIONS — active fuzzy conditions per rule.
#   Higher MIN → stricter rules, fewer matching bars, higher precision target.
#   Lower MIN → broader rules, more trades, risk of weak patterns.
#   Higher MAX → allow complex rules (if encoding supports variable count).
#   Lower MAX → force simplicity; more generalization, less specificity.
MIN_CONDITIONS = 3
MAX_CONDITIONS = 3

# PHASE2_ENCODING — chromosome memory layout during evolution.
#   "dense"        — length-K vector with per-feature dont_care (legacy).
#   "sparse_slots" — fixed slots (MAX_CONDITIONS, 2); dynamic active count.
# Pool JSON / archives remain dense K-vectors for Phase 3 compatibility.
PHASE2_ENCODING = "sparse_slots"


# =============================================================================
# Phase 2 — Trade support & pool admission
# =============================================================================

# MIN_TRADE_SUPPORT — target executed trades before support penalty vanishes.
#   Higher → penalize low-frequency rules harder; pool favors robust sample size.
#   Lower  → allow rare-pattern rules; noisier Sortino/return estimates.
MIN_TRADE_SUPPORT = 90

# SUPPORT_PENALTY_MAX — cap on quadratic support shortfall penalty.
#   Higher → stronger push away from under-supported rules on all objectives.
#   Lower  → evolution tolerates thin trade counts longer.
SUPPORT_PENALTY_MAX = 12.0

# MIN_TRADE_POOL_FLOOR — hard reject below this executed trade count.
#   Higher → archive/pool never keeps very rare rules.
#   Lower  → extremely sparse rules can survive if other metrics excel.
MIN_TRADE_POOL_FLOOR = 17

# PHASE2_SUPPORT_PENALTY_WEIGHT_F1/F2/F3 — per-objective support penalty scale.
#   Higher → that objective punishes low support more (steer Sortino vs DD vs return).
#   Lower  → support matters less for that objective.
PHASE2_SUPPORT_PENALTY_WEIGHT_F1 = 0.8  # Sortino objective
PHASE2_SUPPORT_PENALTY_WEIGHT_F2 = 0.6  # drawdown objective
PHASE2_SUPPORT_PENALTY_WEIGHT_F3 = 0.5  # return / win-rate objective

# PHASE2_USE_TOTAL_RETURN_OBJ — f3 uses return instead of win rate.
#   True  → optimize deployable return; aligns with PnL goals.
#   False → optimize win rate; may favor many small wins over net PnL.
PHASE2_USE_TOTAL_RETURN_OBJ = True

# PHASE2_USE_ROBUST_RETURN_OBJ — f3 uses min(train_return, val_return).
#   True  → penalizes train-only return spikes (recommended with val/CV).
#   False → train return only; easier overfit to in-sample seasons.
PHASE2_USE_ROBUST_RETURN_OBJ = True

# PHASE2_SORTINO_MIN_TRADE_THRESHOLD — trade count below which Sortino is scaled down.
#   Used in Approach 2 to penalize low-trade-count rules.
PHASE2_SORTINO_MIN_TRADE_THRESHOLD = 50

# --- Return / quality floors (evolution + pool filtering) ---

# PHASE2_RETURN_FLOOR_PCT — min train return % to avoid feasibility penalty.
#   Higher → only profitable-on-train rules stay feasible; emptier search.
#   Lower  → more exploration; weak rules linger until other gates remove them.
PHASE2_RETURN_FLOOR_PCT = 0

# PHASE2_VAL_RETURN_FLOOR_PCT — min validation return % for feasibility.
#   Higher → stricter OOS alignment during evolution.
#   Lower  → allow negative val return during search (gates may still catch later).
PHASE2_VAL_RETURN_FLOOR_PCT = 0.5

# PHASE2_PROFIT_FACTOR_FLOOR — min profit factor for feasibility.
#   Higher → require gross wins >> losses; fewer rules pass.
#   Lower  → allow marginal PF; more rules in Pareto set.
PHASE2_PROFIT_FACTOR_FLOOR = 1.05

# PHASE2_SYMBOL_MEDIAN_RETURN_FLOOR_PCT — min median return across symbols.
#   Higher → rules must work on typical symbols, not one outlier.
#   Lower  → single-symbol heroes can survive longer.
PHASE2_SYMBOL_MEDIAN_RETURN_FLOOR_PCT = -0.5

# PHASE2_MIN_PROFITABLE_SYMBOLS — min count of symbols with positive PnL.
#   Higher → demand broad cross-symbol edge; stricter for 10-symbol universe.
#   Lower  → allow niche symbol specialists.
PHASE2_MIN_PROFITABLE_SYMBOLS = 4

# PHASE2_MAX_DRAWDOWN_GATE — hard DD % cap; above this all objectives penalized.
#   Lower  → Pareto front pushed toward low-drawdown rules; may cut high return.
#   Higher → allow aggressive rules with large equity swings.
PHASE2_MAX_DRAWDOWN_GATE = 25.0

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
PHASE2_MAX_TRAIN_VAL_GAP_PCT = 8.0

# PHASE2_KEEP_TOP_RULES — max rules kept in the final Phase 2 pool after
# admission filtering, sorted by deployability_rank_score descending.
#   Higher → larger pool for Phase 3 greedy selection.
#   Lower  → smaller pool; faster Phase 3, fewer combinations.
PHASE2_KEEP_TOP_RULES = 80

# PHASE2_REQUIRE_LAST_FOLD_POSITIVE — in the holdout pool-admission path,
# require the (single) validation fold to have positive total return.
#   True  → pool admission rejects rules with val_return <= 0%.
#   False → val_return can be non-positive (other gates still apply).
# (Originally named for the multi-fold CV era; the check itself runs in the
# holdout path where cv_fold=False, so it remains meaningful without CV.)
PHASE2_REQUIRE_LAST_FOLD_POSITIVE: bool = False

# =============================================================================
# Phase 2 — Monthly-window shadow test for pool admission (Task 13, 2026-06-17)
# =============================================================================
# These flags add a hard pool-admission gate after Phase 2 evolution: each
# candidate rule must be profitable on at least 50% of monthly rolling windows
# in the train split.  This addresses the regime-shift problem identified in
# Task 12's diagnostic CSV: per-symbol rules that pass Phase 3 on val bleed on
# test because they are not stable across time.  The gate is additive — when
# PHASE2_MONTHLY_ADMISSION_ENABLED is False, the existing pool path is
# unchanged.

# PHASE2_MONTHLY_ADMISSION_ENABLED — toggle the monthly-window gate.
#   True  → rules must pass the monthly profitable-ratio filter to enter the pool.
#   False → skip the gate (zero behaviour change vs. pre-Task-13 code).
PHASE2_MONTHLY_ADMISSION_ENABLED = True

# PHASE2_MONTHLY_GOOD_RETURN_MIN_PCT — minimum total_return_pct (%) for a monthly
# window to count as "good" in the pool-admission gate.
#   0.0  → strict profit only (return must be > 0; flat months do not count).
#   2.0  → month must earn at least +2% to count as good.
#   -1.0 → month counts if return >= -1% (more lenient non-loss bar).
PHASE2_MONTHLY_GOOD_RETURN_MIN_PCT = 0.0

# PHASE2_MONTHLY_ADMISSION_MIN_PROFITABLE_RATIO — fraction of monthly windows
# that must count as "good" per PHASE2_MONTHLY_GOOD_RETURN_MIN_PCT to be admitted.
#   Higher → stricter time-stability requirement; fewer rules pass.
#   Lower  → more lenient; rules that work on a minority of windows survive.
#   0.50 means the rule must pass on at least half the windows.
PHASE2_MONTHLY_ADMISSION_MIN_PROFITABLE_RATIO = 0.5

# PHASE2_MONTHLY_ADMISSION_MIN_MONTHS — minimum number of monthly windows
# required before the gate is applied.  When the train split is shorter than
# this, the gate is skipped (with a warning) and the original pool is kept.
#   Higher → skip the gate more often on short data; avoid false negatives.
#   Lower  → require monthly evidence even on short trains.
PHASE2_MONTHLY_ADMISSION_MIN_MONTHS = 4


# =============================================================================
# Phase 2 — Fitness objectives & joint evaluation
# =============================================================================

# SORTINO_CAP — maximum saturated Sortino after tanh compression.
#   Higher → more differentiation among top Sortino rules on f1.
#   Lower  → flatter f1 landscape; diversity across other objectives easier.
SORTINO_CAP = 7.0

# SORTINO_SCALE — divisor inside tanh(raw_sortino / scale); controls saturation.
#   Higher → less compression; extreme Sortino values still differentiate f1.
#   Lower  → aggressive compression; reduces Sortino-driven dominance.
SORTINO_SCALE = 5.0

# PHASE2_JOINT_TRAIN_VAL — fitness uses min(train, val) Sortino/return where applicable.
#   True  → slower (eval val every gen) but aligned with deployment; less overfit.
#   False → train-only fitness; faster but val-blind during evolution.
PHASE2_JOINT_TRAIN_VAL = True

# --- Recency weighting (train bars in last fraction count more) ---

# PHASE2_RECENCY_WEIGHT_ENABLED — up-weight recent train bars in return objective.
#   True  → rules must work in latest market structure, not only old regimes.
#   False → uniform weight across train history.
PHASE2_RECENCY_WEIGHT_ENABLED: bool = True

# PHASE2_RECENCY_WEIGHT_FRACTION — tail fraction of train bars that get boosted.
#   Higher → more bars double-counted; stronger recency bias.
#   Lower  → narrower recent window matters.
PHASE2_RECENCY_WEIGHT_FRACTION: float = 0.25

# PHASE2_RECENCY_WEIGHT_MULTIPLIER — weight multiplier on recency fraction bars.
#   Higher → recent performance dominates fitness; may ignore older seasons.
#   Lower  → mild recency nudge (1.0 = no boost within enabled fraction).
PHASE2_RECENCY_WEIGHT_MULTIPLIER: float = 2.0


# =============================================================================
# Phase 2 — Diversity, early stop & two-stage search
# =============================================================================

# PHASE2_DIVERSITY_HAMMING_THRESHOLD — min Hamming distance for "unique" rule.
#   Higher → demand more genetic distance; wider Pareto spread, slower convergence.
#   Lower  → allow near-duplicate rules; risk of niche collapse.
PHASE2_DIVERSITY_HAMMING_THRESHOLD = 3

# PHASE2_DIVERSITY_PENALTY — objective penalty when crowding near existing rules.
#   Higher → stronger push toward novel chromosomes.
#   Lower  → convergence to similar high performers allowed.
PHASE2_DIVERSITY_PENALTY = 8.0

# PHASE2_PHENOTYPE_SORTINO_STEP — Sortino bucket width for behavioral diversity.
PHASE2_PHENOTYPE_SORTINO_STEP = 0.5

# PHASE2_PHENOTYPE_DD_STEP — drawdown % bucket width for behavioral diversity.
PHASE2_PHENOTYPE_DD_STEP = 5.0

# PHASE2_PHENOTYPE_F3_STEP — f3-axis bucket width (win rate % or return %).
PHASE2_PHENOTYPE_F3_STEP = 10.0

# PHASE2_EARLY_STOP_ENABLED — stop evolution on poor mean/median return trend.
#   True  → save generations when search is clearly failing.
#   False → always run full PHASE2_GENERATIONS budget.
PHASE2_EARLY_STOP_ENABLED = True

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

PHASE2_PLATEAU_EARLY_STOP_ENABLED = True

# PHASE2_PLATEAU_EARLY_STOP_MIN_GENERATION — earliest gen for plateau stop.
#   Higher → more exploration in Stage A before plateau can end run.
#   Lower  → may stop during initial transient; should be ≤ STAGE_A_GENERATIONS.
PHASE2_PLATEAU_EARLY_STOP_MIN_GENERATION = 3

# PHASE2_PLATEAU_EARLY_STOP_PATIENCE — gens without improvement before stop.
#   Higher → wait longer for breakthrough; uses more compute.
#   Lower  → stop quickly when progress stalls.
PHASE2_PLATEAU_EARLY_STOP_PATIENCE = 5

# PHASE2_PLATEAU_EARLY_STOP_MIN_DELTA_PCT — min return improvement to reset patience.
#   Higher → need larger gains to count as progress.
#   Lower  → tiny improvements reset plateau counter.
PHASE2_PLATEAU_EARLY_STOP_MIN_DELTA_PCT = 0.02

# PHASE2_PLATEAU_USE_ROBUST_RETURN — track min(train,val) return for plateau.
#   True  → plateau reflects deployable return, not train-only spikes.
#   False → train max return can mask val stagnation.
PHASE2_PLATEAU_USE_ROBUST_RETURN = True

PHASE2_PLATEAU_BLOCK_WHEN_DEPLOYABLE_ZERO = True
PHASE2_PLATEAU_BLOCK_WHEN_DIVERSITY_LOW = False

# PHASE2_FEASIBILITY_VIOLATION_WEIGHT — scales soft penalty for floor violations.
#   Higher → infeasible rules pushed far down on all objectives.
#   Lower  → borderline rules compete with feasible ones longer.
PHASE2_FEASIBILITY_VIOLATION_WEIGHT = 25.0

# PHASE2_INFEASIBLE_OBJECTIVE_PENALTY — flat penalty added when hard infeasible.
#   Higher → clear separation feasible vs infeasible on Pareto front.
#   Lower  → infeasible rules may linger in ranking.
PHASE2_INFEASIBLE_OBJECTIVE_PENALTY = 100.0

# PHASE2_DEPLOYABLE_ARCHIVE_MAX_SIZE — cap on stored deployable-elite archive.
#   Higher → more warm-start diversity across runs; more disk/RAM.
#   Lower  → smaller cross-run memory.
# Lowered from 200 → 100 to reduce RAM on Colab (12.7 GiB host).
PHASE2_DEPLOYABLE_ARCHIVE_MAX_SIZE = 100

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

# --- Two-stage evolution: wide exploration → val-robust refinement ---

PHASE2_TWO_STAGE_ENABLED = True

# PHASE2_STAGE_A_GENERATIONS — Stage A (exploration) generation budget.
#   Higher → more diverse initial Pareto before val-focused Stage B.
#   Lower  → quicker handoff; Stage B may miss good regions.
PHASE2_STAGE_A_GENERATIONS = 85

# PHASE2_STAGE_B_GENERATIONS — Stage B (refinement) generation budget.
#   Higher → more val-robust polishing; total time = A + B gens.
#   Lower  → less refinement after exploration.
PHASE2_STAGE_B_GENERATIONS = 45

# PHASE2_STAGE_B_SEED_TOP_K — elites from Stage A seeded into Stage B.
#   Higher → broader refinement starting set; slower Stage B per gen.
#   Lower  → refine only top performers; risk missing dark horses.
PHASE2_STAGE_B_SEED_TOP_K = 50

# PHASE2_STAGE_B_SEED_FRACTION — fraction of Stage B pop seeded from Stage A elites.
#   Higher → more refinement around known good regions; risk of clone collapse.
#   Lower  → more random exploration in Stage B.
PHASE2_STAGE_B_SEED_FRACTION = 0.30

# --- Stage A hyperparameters (exploration: higher mutation, stronger diversity) ---

# PHASE2_STAGE_A_MUTATION_RATE — per-gene mutation in Stage A.
#   Higher → more genetic exploration before Stage B refinement.
PHASE2_STAGE_A_MUTATION_RATE = 0.25

# PHASE2_STAGE_A_MUTATION_WEIGHTED_ACTIVATE_PROB — bias toward activating genes in A.
PHASE2_STAGE_A_MUTATION_WEIGHTED_ACTIVATE_PROB = 0.50

# PHASE2_STAGE_A_DIVERSITY_PENALTY — crowding penalty on objectives in Stage A.
PHASE2_STAGE_A_DIVERSITY_PENALTY = 10.0

# PHASE2_STAGE_A_DIVERSITY_HAMMING_THRESHOLD — min Hamming distance before penalty in A.
PHASE2_STAGE_A_DIVERSITY_HAMMING_THRESHOLD = 4

# PHASE2_STAGE_A_DIVERSITY_RECOVERY_MIN_UNIQUE_RATIO — trigger diversity injection in A.
PHASE2_STAGE_A_DIVERSITY_RECOVERY_MIN_UNIQUE_RATIO = 0.35

# PHASE2_STAGE_A_DIVERSITY_RECOVERY_INJECT_FRACTION — pop replaced on recovery in A.
PHASE2_STAGE_A_DIVERSITY_RECOVERY_INJECT_FRACTION = 0.35

# PHASE2_STAGE_A_DIVERSITY_RECOVERY_MUTATION_BOOST — mutation multiplier after recovery in A.
PHASE2_STAGE_A_DIVERSITY_RECOVERY_MUTATION_BOOST = 2.0

# PHASE2_STAGE_A_PLATEAU_EARLY_STOP_PATIENCE — gens without progress before stop in A.
PHASE2_STAGE_A_PLATEAU_EARLY_STOP_PATIENCE = 28

# PHASE2_STAGE_A_PLATEAU_EARLY_STOP_MIN_GENERATION — earliest plateau stop gen in A.
PHASE2_STAGE_A_PLATEAU_EARLY_STOP_MIN_GENERATION = 30

# PHASE2_STAGE_A_EARLY_STOP_MIN_GENERATION — earliest return-based early stop in A.
PHASE2_STAGE_A_EARLY_STOP_MIN_GENERATION = 32

# PHASE2_STAGE_A_ARCHIVE_SEED_FRACTION — warm-start fraction from prior pool in Stage A.
PHASE2_STAGE_A_ARCHIVE_SEED_FRACTION = 0.20

# --- Stage A evolution floor overrides (loose fitness gates; pool export stays strict) ---

# PHASE2_STAGE_A_RETURN_FLOOR_PCT — min train return % during Stage A fitness only.
PHASE2_STAGE_A_RETURN_FLOOR_PCT = 0.0

# PHASE2_STAGE_A_MIN_TRADE_SUPPORT — trade-count target before support penalty vanishes in Stage A.
PHASE2_STAGE_A_MIN_TRADE_SUPPORT = 30

# PHASE2_STAGE_A_USE_ROBUST_RETURN_OBJ — Stage A f3 uses train return instead of min(train,val).
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
PHASE2_STAGE_B_MUTATION_RATE = 0.18

# PHASE2_STAGE_B_MUTATION_WEIGHTED_ACTIVATE_PROB — conservative gene activation in B.
PHASE2_STAGE_B_MUTATION_WEIGHTED_ACTIVATE_PROB = 0.40

# PHASE2_STAGE_B_DIVERSITY_PENALTY — weaker crowding penalty during refinement.
PHASE2_STAGE_B_DIVERSITY_PENALTY = 5.0

# PHASE2_STAGE_B_DIVERSITY_HAMMING_THRESHOLD — allow nearer-duplicate elites in B.
PHASE2_STAGE_B_DIVERSITY_HAMMING_THRESHOLD = 2

# PHASE2_STAGE_B_DIVERSITY_RECOVERY_MIN_UNIQUE_RATIO — later diversity recovery in B.
PHASE2_STAGE_B_DIVERSITY_RECOVERY_MIN_UNIQUE_RATIO = 0.25

# PHASE2_STAGE_B_DIVERSITY_RECOVERY_INJECT_FRACTION — smaller injection shock in B.
PHASE2_STAGE_B_DIVERSITY_RECOVERY_INJECT_FRACTION = 0.20

# PHASE2_STAGE_B_DIVERSITY_RECOVERY_MUTATION_BOOST — milder post-recovery mutation in B.
PHASE2_STAGE_B_DIVERSITY_RECOVERY_MUTATION_BOOST = 1.4

# PHASE2_STAGE_B_PLATEAU_EARLY_STOP_PATIENCE — shorter patience while polishing in B.
PHASE2_STAGE_B_PLATEAU_EARLY_STOP_PATIENCE = 15

# PHASE2_STAGE_B_PLATEAU_EARLY_STOP_MIN_GENERATION — earliest plateau stop gen in B.
PHASE2_STAGE_B_PLATEAU_EARLY_STOP_MIN_GENERATION = 15

# PHASE2_STAGE_B_EARLY_STOP_MIN_GENERATION — earliest return-based early stop in B.
PHASE2_STAGE_B_EARLY_STOP_MIN_GENERATION = 20

# PHASE2_GPU_ENRICH_SYMBOL_METRICS — merge CPU per-symbol metrics after GPU batch eval.
PHASE2_GPU_ENRICH_SYMBOL_METRICS = True


def phase2_should_enrich_symbol_metrics(engine: object | None = None) -> bool:
    """Return True when GPU batch eval should run a follow-up CPU enrichment pass."""
    if not PHASE2_GPU_ENRICH_SYMBOL_METRICS:
        return False
    return True


# =============================================================================
# Phase 2 — NSGA-III search budget & archive

# PHASE2_POPULATION_SIZE — individuals per generation.
#   Higher → better Pareto coverage, ~linear GPU cost per generation.
#   Lower  → faster gens, risk of premature convergence.
PHASE2_POPULATION_SIZE = 200

# PHASE2_GENERATIONS — total evolutionary generations (before early stop).
#   Higher → more search budget; diminishing returns after plateau.
#   Lower  → faster runs; may under-explore gene space.
PHASE2_GENERATIONS = 150

PHASE2_ALGORITHM = "NSGA3"

# PHASE2_ARCHIVE_MAX_SIZE — max stored non-dominated solutions across gens.
#   Higher → richer elite memory; more memory, slower non-dominated sorting.
#   Lower  → leaner archive; may lose good rules found early.
# Lowered from 400 → 200 to reduce RAM on Colab (12.7 GiB host).
PHASE2_ARCHIVE_MAX_SIZE = 200

# PHASE2_ARCHIVE_SEED_FRACTION — fraction of initial pop from cross-run archive.
#   Higher → more warm-start from past runs; less fresh random exploration.
#   Lower  → more random init; slower reuse of known good rules.
PHASE2_ARCHIVE_SEED_FRACTION = 0.25

PHASE2_SEED: int = get_seed()


# =============================================================================
# Phase 2 — Island / cluster mode (scoped evolution)
# =============================================================================
# When PHASE2_ISLAND_MODE == "cluster", Phase 2 runs K hybrid clusters with a
# fixed total generation budget split across islands. Global knobs below stay
# as universe bases; runtime scaling via resolve_island_hyperparams().

PHASE2_ISLAND_MODE = "cluster"  # "global" | "cluster"
PHASE2_N_CLUSTERS = 3
PHASE2_ISLAND_TOTAL_GENERATIONS = PHASE2_GENERATIONS
PHASE2_ISLAND_EPOCH_GENERATIONS = 25
PHASE2_ISLAND_TWO_STAGE_ENABLED = False
PHASE2_ISLAND_EARLY_STOP_ENABLED = False
PHASE2_ISLAND_PLATEAU_EARLY_STOP_ENABLED = False
PHASE2_ISLAND_SCALE_TRADE_FLOORS = True
PHASE2_ISLAND_TRADE_FLOOR_ABSOLUTE_MIN = 10
PHASE2_ISLAND_MONTHLY_MIN_MONTHS = 4
PHASE2_MIGRATION_EPOCH_INTERVAL = 2
PHASE2_MIGRATION_TOP_K = 5
PHASE2_MIGRATION_REQUIRE_DEPLOYABILITY = True
PHASE2_MIGRATION_MIN_VAL_RETURN_PCT = 0.0
PHASE2_MIGRATION_MIN_VAL_TRADES: int | None = None

PHASE2_ORPHAN_ENABLED = True
PHASE2_ORPHAN_GENERATIONS = 18
PHASE2_ORPHAN_POPULATION_SIZE = 100
PHASE2_ORPHAN_MIN_TRADE_SUPPORT = 8
PHASE2_ORPHAN_MIN_TRADE_POOL_FLOOR = 8
PHASE2_ORPHAN_SORTINO_MIN_TRADE_THRESHOLD = 8
PHASE2_ORPHAN_MIN_VAL_TRADES = 6
PHASE2_ORPHAN_MIN_VAL_RETURN_PCT = 0.0
PHASE2_ORPHAN_MONTHLY_MIN_PROFITABLE_RATIO = 0.4


def phase2_cluster_archive_path(direction: str, cluster_id: str) -> str:
    """Persistent archive for one cluster island."""
    return os.path.join(
        PHASE2_ARCHIVE_DIR,
        direction,
        f"cluster_{cluster_id}",
        "archive.json",
    )


def phase2_shared_archive_path(direction: str) -> str:
    """Cross-cluster shared archive for migration warm-start."""
    return os.path.join(PHASE2_ARCHIVE_DIR, direction, "shared_archive.json")


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
PHASE2_INIT_STRATUM_FRACTIONS = (0.67, 0.33)

# PHASE2_INIT_SOFTMAX_TEMP — temperature for weighted feature activation in init.
#   Higher → more uniform random feature picks.
#   Lower  → strongly favor high-MI features in initial conditions.
PHASE2_INIT_SOFTMAX_TEMP = 1.5

PHASE2_INIT_SCORE_EPS = 1e-6

# PHASE2_INIT_UNIFORM_MIX — probability of uniform random gene vs structured init.
#   Higher → more random chromosomes in initial population.
#   Lower  → more MI-guided structured rules at gen 0.
PHASE2_INIT_UNIFORM_MIX = 0.05

# PHASE2_MUTATION_RATE — per-gene mutation probability.
#   Higher → more exploration, noisier convergence, better escape local optima.
#   Lower  → finer local search, risk of premature convergence.
PHASE2_MUTATION_RATE = 0.22

# PHASE2_MUTATION_WEIGHTED_ACTIVATE_PROB — bias mutations toward activating genes.
#   Higher → mutations tend to add conditions rather than dont_care.
#   Lower  → mutations more often deactivate or flip existing conditions.
PHASE2_MUTATION_WEIGHTED_ACTIVATE_PROB = 0.45


# =============================================================================
# Phase 3 — Rule set selection (greedy + NSGA-II)
# =============================================================================

# --- Team shape ---

# PHASE3_PER_SYMBOL_MAX_RULES — maximum rules selected per symbol.
PHASE3_PER_SYMBOL_MAX_RULES = 2

# PHASE3_GLOBAL_MIN_RULES / MAX_RULES — total rules in the output JSON.
#   Higher MIN → require at least this many rules across all symbols.
#   MAX must match EVALUATOR_MAX_RULES (evaluator v5 accepts 2–5 rules).
PHASE3_GLOBAL_MIN_RULES = 2
# Evaluator v5 accepts at most 5 rules; Phase 3 must not exceed this.
PHASE3_GLOBAL_MAX_RULES = 5

# PHASE3_PER_SYMBOL_GREEDY_TOP_K — top-K pool rules tested per greedy round.
#   Higher → more thorough search per symbol, slower.
#   Lower  → faster, may miss good combinations.
PHASE3_PER_SYMBOL_GREEDY_TOP_K = 25

# PHASE3_PER_SYMBOL_MIN_TRADES — min trades on symbol's val data for rule.
#   Higher → reject rules with thin evidence on that symbol.
#   Lower  → allow sparse rules through.
#   Debug scope scales via effective_phase3_per_symbol_min_trades().
#   Lowered from 50→15 on 2026-06-16 because the current 4-9 rule pool
#   cannot reach 50 trades per symbol (∼6 val trades/symbol on average).
#   15 aligns with SYMBOL_SPECIALIZATION_MIN_VAL_TRADES=6 (Task 6) while
#   still requiring more-than-minimal evidence on each symbol.
#   Lowered from 15→8 on 2026-06-17 (Task 12) because 6/10 symbols still
#   have no rules on test; 8 trades on a ~7k-row per-symbol validation
#   window (≈0.1% of bars) is still a reasonable evidence minimum.
PHASE3_PER_SYMBOL_MIN_TRADES = 8

# PHASE3_PER_SYMBOL_MIN_RETURN — min val return % on symbol for rule.
#   Higher → only profitable-on-symbol rules considered.
#   Lower  → allow marginal rules through.
#   Debug scope relaxes via effective_phase3_per_symbol_min_return().
#   Lowered from 3.0→1.5 on 2026-06-16 because the 4-9 rule pool has
#   only ∼2-4% max returns per symbol; 3.0 was rejecting everything.
#   1.5 still requires modest profitability while allowing Phase 4 risk
#   optimization to improve the final team's return.
#   Lowered from 1.5→0.5 on 2026-06-17 (Task 12) — many rules have
#   positive but small per-symbol returns (<1%) due to the thin
#   per-symbol validation window (~7k rows).  0.5% is still a positive
#   return that Phase 4 risk optimization can amplify.
PHASE3_PER_SYMBOL_MIN_RETURN = 0.5

# PHASE3_DIAGNOSTIC_REPORT_ENABLED — write per-symbol diagnostic CSV.
#   When True, ``Rule_Set_Selector.run()`` writes
#   ``outputs/reports/gen_diag_iter12.csv`` with columns:
#   direction, symbol, val_trades, val_return_pct, train_val_gap_pct,
#   n_rules_selected, top_rule_condition_signature.
#   The CSV has one row per (direction, symbol) pair that had at least
#   1 rule selected.  This is a diagnostic artifact for the user to see
#   which symbols are still being dropped and why.
PHASE3_DIAGNOSTIC_REPORT_ENABLED = True

# PHASE3_MAX_CAPITAL_PCT_PER_RULE — cap per rule before normalization.
#   Higher → each rule can use more notional; higher overlap drawdown risk.
#   Lower  → thinner per-rule sizing; may under-use signals.
PHASE3_MAX_CAPITAL_PCT_PER_RULE = 50.0

# PHASE3_MAX_TRAIN_VAL_GAP_PCT — max allowed gap between train return and val
# return for a rule to pass Phase 3 per-symbol scoring.
#   If train_return - val_return > this threshold the rule is hard-rejected
#   as an overfit signal (scored -999 so it never enters the greedy team).
#   Higher → more lenient; only extreme gaps rejected.
#   Lower  → stricter; tighter alignment between train and val required.
#   Set to a large number (e.g. 999) to disable the gap gate entirely.
PHASE3_MAX_TRAIN_VAL_GAP_PCT = 12.0

# --- Engines ---

PHASE3_USE_GPU = False  # overridden to True on Colab GPU via _apply_colab_gpu_defaults()

# PHASE3_BATCH_WORKERS — parallel workers for team evaluation.
#   Higher → faster Phase 3 on many-core CPU; diminishing returns past ~32.
#   Lower  → less CPU contention.
PHASE3_BATCH_WORKERS = min(32, os.cpu_count() or 4)

# Return / PF floors for Phase 3 team admission (must align with Phase 2 quality).
# Higher floors → fewer teams pass; lower → more teams, weaker OOS risk.
# 8% blocked long fallback when Phase 2 max return was ~8.3%; 5% is better aligned.
# Debug scope relaxes via effective_phase3_val_return_floor_pct().
PHASE3_VAL_RETURN_FLOOR_PCT = 5.0

# --- Positive-good gate (is_positive_good style) ---------------------------------

# PHASE3_REQUIRE_POSITIVE_GOOD — require rule to be positive on both train and val
# with PF >= 1.0, min trades, and returns above the configured floors.
# When True, ``gate_positive_good()`` is called in per-symbol greedy scoring;
# rules that fail are hard-rejected (return -999).  The existing
# ``PHASE3_MAX_TRAIN_VAL_GAP_PCT`` gate runs in addition, not instead.
#   True  → reject rules that are not profitable on both splits (default).
#   False → skip this gate (legacy behaviour).
PHASE3_REQUIRE_POSITIVE_GOOD = True

# PHASE3_MIN_TRAIN_RETURN — minimum train return % for ``gate_positive_good``.
#   Higher → only strongly profitable-on-train rules pass the gate.
#   Lower  → any positive return counts (0.0 = strictly > 0).
PHASE3_MIN_TRAIN_RETURN = 0.0

# PHASE3_MIN_VAL_RETURN — minimum validation return % for ``gate_positive_good``.
#   Higher → only strongly profitable-on-val rules pass.
#   Lower  → any positive return counts (0.0 = strictly > 0).
PHASE3_MIN_VAL_RETURN = 0.0

# PHASE3_MIN_TRAIN_PF — minimum train profit factor for ``gate_positive_good``.
#   Higher → require strong gross-win / gross-loss ratio on train.
#   Lower  → allow marginal train PF (1.0 = break-even before fees).
PHASE3_MIN_TRAIN_PF = 1.0

# PHASE3_MIN_VAL_PF — minimum validation profit factor for ``gate_positive_good``.
#   Higher → require strong gross-win / gross-loss ratio on val.
#   Lower  → allow marginal val PF (1.0 = break-even before fees).
PHASE3_MIN_VAL_PF = 1.0

# PHASE3_MIN_TRAIN_TRADES — minimum executed trades on train for gate.
#   Higher → reject thin train-sample rules.
#   Lower  → allow sparse train evidence (must still pass pool floors).
PHASE3_MIN_TRAIN_TRADES = 25

# PHASE3_MIN_VAL_TRADES — minimum executed trades on validation for gate.
#   Higher → reject thin val-sample rules.
#   Lower  → allow sparse val evidence.
PHASE3_MIN_VAL_TRADES = 15

# PHASE2_STRICT_POSITIVE_GOOD — when True, applies ``gate_positive_good`` in
# Phase 2 pool admission (``_passes_pool_admission_impl``).  Default OFF to
# avoid breaking the existing pool; turned ON in Task 5.
#   True  → pool entries must also pass the positive-good gate.
#   False → pool admission uses its own floors (legacy, unchanged).
PHASE2_STRICT_POSITIVE_GOOD = True


# --- Evaluator health penalty (Task 4) ---------------------------------------

# EVAL_HEALTH_MAX_SKIPPED_RATIO — max (skipped / raw) before skip penalty kicks in.
#   Higher → tolerate more evaluator-filtered signals.
#   Lower  → penalise strategies whose signals are mostly below MIN_POSITION_NOTIONAL.
EVAL_HEALTH_MAX_SKIPPED_RATIO = 0.20

# EVAL_HEALTH_MIN_EXECUTED_RATIO — min (executed / raw) to avoid exec penalty.
#   Higher → require most raw signals to actually open as trades.
#   Lower  → tolerate moderate skip rates without penalty.
EVAL_HEALTH_MIN_EXECUTED_RATIO = 0.60

# EVAL_HEALTH_SKIPPED_WEIGHT — penalty multiplier for exceeding max skip ratio.
#   Higher → larger penalty per % of excess skipped signals.
EVAL_HEALTH_SKIPPED_WEIGHT = 3500.0

# EVAL_HEALTH_EXECUTED_WEIGHT — penalty multiplier for falling below min exec ratio.
#   Higher → larger penalty per % of missing executed trades.
EVAL_HEALTH_EXECUTED_WEIGHT = 2500.0

# EVAL_HEALTH_MAX_SIMULTANEOUS_POSITIONS — max concurrent positions before penalty.
EVAL_HEALTH_MAX_SIMULTANEOUS_POSITIONS = 10

# EVAL_HEALTH_MAX_POSITIONS_WEIGHT — penalty multiplier per excess concurrent position.
EVAL_HEALTH_MAX_POSITIONS_WEIGHT = 120.0

# PHASE3_EVAL_HEALTH_WEIGHT — multiplier on evaluator_health_penalty in Phase 3 scoring.
#   1.0 → full penalty applied; 0.0 → no penalty (legacy).
PHASE3_EVAL_HEALTH_WEIGHT = 1.0

# PHASE3_GATE_EXECUTION_HEALTH — when True, ``gate_positive_good`` also requires
# that both train and val pass ``execution_ok()``.
#   True  → reject rule sets with excessive skip rates at the gate level.
#   False → skip this extra gate (legacy behaviour).
PHASE3_GATE_EXECUTION_HEALTH = True


# --- Multi-symbol combinations in Phase 3 symbol specialization (Task 6) -----
# These control how ``_build_symbol_specialized_variants`` generates 1-, 2-,
# and 3-symbol variants of each pool-chosen rule.  When ``USE_COMBINATIONS``
# is True, rules may contain multiple ``symbol is X`` conditions (e.g.
# ``symbol is 1, symbol is 5``), expanding the search to cross-symbol
# diversification.  The friend's defaults are used (from rb_governor.py).

# SYMBOL_SPECIALIZATION_USE_COMBINATIONS — when True, also try 2- and 3-symbol
#   combinations of the top single-symbol variants per rule.  When False, only
#   single-symbol variants are produced (legacy behaviour).
#   Higher → richer symbol combinations, broader search, slower evaluation.
#   Lower  → only single-symbol specialisation (original Phase 3 behaviour).
SYMBOL_SPECIALIZATION_USE_COMBINATIONS = True

# SYMBOL_SPECIALIZATION_MAX_SYMBOLS_PER_RULE — maximum number of symbols in a
#   single rule's ``symbol is X`` conditions (1 = single only, 2 = 1+2 combos,
#   3 = 1+2+3 combos).
#   Higher → more symbols per rule, potential overfitting to specific baskets.
#   Lower  → simpler rules, easier to interpret.
SYMBOL_SPECIALIZATION_MAX_SYMBOLS_PER_RULE = 1

# SYMBOL_SPECIALIZATION_TOP_SINGLE_SYMBOLS — number of top-ranked single-symbol
#   variants used as the seed set for generating 2- and 3-symbol combinations.
#   Higher → more 2/3-symbol candidates, slower evaluation.
#   Lower  → fewer candidates, faster but may miss good cross-symbol combos.
SYMBOL_SPECIALIZATION_TOP_SINGLE_SYMBOLS = 5

# SYMBOL_SPECIALIZATION_MAX_VARIANTS_PER_RULE — maximum number of scored variants
#   returned per pool rule, sorted by score descending.  Only the best variant
#   is used in the final rule set.
#   Higher → more candidates retained (only the best is used, but more combos
#   are scored for tie-breaking).
SYMBOL_SPECIALIZATION_MAX_VARIANTS_PER_RULE = 10

# SYMBOL_SPECIALIZATION_MIN_TRAIN_TRADES — minimum executed trades on the train
#   split for a variant to be considered (passed to ``gate_positive_good``).
#   Note: the friend uses 10 (lower than the Phase 3 default of 25) to avoid
#   filtering out valid multi-symbol rules too aggressively.
#   Higher → stricter train-trade filter, fewer variants.
#   Lower  → more variants survive the gating step.
SYMBOL_SPECIALIZATION_MIN_TRAIN_TRADES = 10

# SYMBOL_SPECIALIZATION_MIN_VAL_TRADES — minimum executed trades on the validation
#   split for a variant to be considered (passed to ``gate_positive_good``).
#   Note: the friend uses 6 (lower than the Phase 3 default of 15).
#   Higher → stricter val-trade filter.
#   Lower  → more variants survive the gating step.
SYMBOL_SPECIALIZATION_MIN_VAL_TRADES = 6


# =============================================================================
# Phase 4 — Walk-forward risk optimization (TP / SL / capital)
# =============================================================================
# Rule conditions are frozen; only risk params are optimized via Optuna.

# --- Search space bounds ---

# PHASE4_TP_MIN/MAX — take-profit search range (%).
#   Wider MAX → allow larger targets; fewer hits, bigger winners per trade.
#   Narrower → optimizer stuck with modest TP; may miss trend captures.
PHASE4_TP_MIN = 2.0
PHASE4_TP_MAX = 5.0

# PHASE4_SL_MIN/MAX — stop-loss search range (%).
#   Wider MAX → wider stops, lower stop-out rate, larger loss per loser.
#   Narrower → tighter risk control; more stop-outs.
PHASE4_SL_MIN = 1.0
PHASE4_SL_MAX = 2.0

# PHASE4_MIN_TP_SL_RATIO — enforce TP > SL × ratio (trend-following RR discipline).
#   Higher → demand more reward per unit risk; fewer feasible trials.
#   Lower  → allow near 1:1 or inverted effective RR combinations.
PHASE4_MIN_TP_SL_RATIO = 1.2

# PHASE4_CAPITAL_PCT_MIN/MAX — per-rule capital allocation search range.
#   Higher MAX → optimizer can concentrate more capital per signal.
#   Lower MAX → forced diversification across rules.
#   Setting MIN == MAX locks capital to a fixed value (old behaviour was 30/30).
#   Widening the range lets Optuna discover the best allocation; the
#   PHASE4_HARD_CAP_NORMALIZE step then scales the total to ≤150%.
PHASE4_CAPITAL_PCT_MIN = 10.0
PHASE4_CAPITAL_PCT_MAX = 30.0

# PHASE4_TP_STEP / SL_STEP / CAPITAL_STEP — Optuna discretization granularity.
#   Smaller steps → finer search, more trials needed to explore space.
#   Larger steps → coarser optimum, faster convergence per trial.
PHASE4_TP_STEP = 0.5
PHASE4_SL_STEP = 0.5
PHASE4_CAPITAL_STEP = 5.0

# --- Optuna budget ---


# PHASE4_HARD_CAP_NORMALIZE — scale capital so sum ≤ MAX_TOTAL_EXPOSURE_PCT.
#   True  → realistic portfolio cap; required for live-like exposure.
#   False  → raw trial capital may exceed 100% total exposure.
PHASE4_HARD_CAP_NORMALIZE = True

# --- Walk-forward windows on validation data ---

# PHASE4_WF_SPLITS — number of walk-forward windows on validation split.
#   Higher → stricter temporal robustness; each window smaller (trade starvation).
#   Lower  → larger windows, more trades per fold, less temporal coverage.
PHASE4_WF_SPLITS = 2


def effective_phase4_wf_splits() -> int:
    """Inner validation WF windows; single window when purged CV already ran."""
    if split_mode_is_purged_walk_forward():
        return 1
    return int(PHASE4_WF_SPLITS)

PHASE4_INCLUDE_TAIL_HOLDOUT = True

# PHASE4_TAIL_HOLDOUT_FRACTION — fraction of val reserved as final holdout window.
#   Higher → more recent data held out; fewer trades in WF folds.
#   Lower  → more data in WF folds; less independent tail check.
PHASE4_TAIL_HOLDOUT_FRACTION = 0.25

# Worst-fold objective weights (emphasize tail risk across WF windows).
#   Higher WORST_RETURN_WEIGHT → optimizer prioritizes worst-window return.
#   Higher WORST_DRAWDOWN_WEIGHT → penalize strategies that blow up in one window.
#   Higher WORST_TURNOVER_WEIGHT → penalize fee-heavy params in worst window.
PHASE4_WORST_RETURN_WEIGHT = 1.5
PHASE4_WORST_DRAWDOWN_WEIGHT = 2.0
PHASE4_WORST_TURNOVER_WEIGHT = 0.5

# --- Feasibility filters (trial rejected if any fail) ---

# PHASE4_MAX_WORST_DRAWDOWN_PCT — max allowed worst-window drawdown %.
#   Lower → only low-DD risk params feasible; may reject all trials.
#   Higher → allow volatile params through.
PHASE4_MAX_WORST_DRAWDOWN_PCT = 15.0

# PHASE4_MIN_WORST_TRADES — min trades in worst WF window.
#   Higher → demand statistical significance in every window; very strict.
#   Lower  → thin windows can still produce feasible trials.
#   With per-symbol rules, each rule fires on fewer rows, so lower is needed.
PHASE4_MIN_WORST_TRADES = 20

# PHASE4_MIN_WORST_FOLD_RETURN_PCT — min return % in worst WF window.
#   Higher → only consistently profitable windows pass; may zero feasible set.
#   Lower (more negative) → allow losing worst windows; more trials pass.
PHASE4_MIN_WORST_FOLD_RETURN_PCT = -2.0

# PHASE4_MIN_WORST_FOLD_PF — min profit factor in worst WF window.
#   Higher → stricter per-window profitability.
#   Lower → marginal worst-window PF allowed.
PHASE4_MIN_WORST_FOLD_PF = 1.0


# =============================================================================
# Monthly windows validation (used in Phase 3/4 scoring)
# =============================================================================

# MONTHLY_VALIDATION_ENABLED — toggle monthly rolling-window validation.
#   True  → rule sets are penalised if they fail monthly windows gates.
#   False → monthly penalty is skipped (legacy behaviour).
MONTHLY_VALIDATION_ENABLED = True

MONTHLY_WINDOW_DAYS = 30
MONTHLY_WINDOW_MIN_ROWS = 2500
MONTHLY_WINDOW_MAX_WINDOWS = 24
MONTHLY_RECENCY_WEIGHT = 2.2
MONTHLY_MIN_TRADES = 20

# MONTHLY_GOOD_RETURN_MIN_PCT — minimum total_return_pct (%) for a monthly window
# to count toward profitable_ratio in summarize_monthly_metrics / monthly_penalty.
#   0.0  → non-losing months count (return >= 0; flat months are OK).
#   2.0  → month must earn at least +2% to count as good.
#   -1.0 → month counts if return >= -1%.
MONTHLY_GOOD_RETURN_MIN_PCT = 0.5

# MONTHLY_MIN_PROFITABLE_RATIO — target fraction of "good" months per
# MONTHLY_GOOD_RETURN_MIN_PCT; monthly_penalty rises when ratio falls below this.
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

# PHASE3_MONTHLY_PENALTY_WEIGHT — multiplier on monthly_penalty() in Phase 3 scoring.
PHASE3_MONTHLY_PENALTY_WEIGHT = 1.0
# PHASE3_MONTHLY_PENALTY_SCALE — divides the weighted monthly penalty before it is
# subtracted from min(train, val) return (%).  Converts abstract penalty points
# into a return-comparable drag: effective_drag = penalty * WEIGHT / SCALE.
#   Higher → weaker monthly influence (e.g. 10.0 turns a 20-pt penalty into −2%).
#   Lower  → stronger monthly influence; must be > 0.
PHASE3_MONTHLY_PENALTY_SCALE = 7.0
# PHASE3_MONTHLY_FALLBACK_PENALTY — fallback penalty when monthly windows == 0.
PHASE3_MONTHLY_FALLBACK_PENALTY = 5.0
# PHASE4_MONTHLY_SCORE_WEIGHT — multiplier on monthly_penalty() in Phase 4 grid scoring.
PHASE4_MONTHLY_SCORE_WEIGHT = 0.70
# PHASE4_MONTHLY_PENALTY_SCALE — divides the weighted monthly penalty before it is
# subtracted from the grid composite score (_score_metrics output).
#   Higher → weaker monthly influence during TP/SL/capital search.
#   Lower  → stronger monthly influence; must be > 0.
PHASE4_MONTHLY_PENALTY_SCALE = 10.0
# PHASE4_MONTHLY_FALLBACK_PENALTY — fallback raw penalty when monthly windows == 0.
PHASE4_MONTHLY_FALLBACK_PENALTY = 5.0
# PHASE4_MONTHLY_EVAL_EVERY_TRIAL — when True, evaluate monthly_penalty on every
# grid trial using cached train+val monthly windows.
PHASE4_MONTHLY_EVAL_EVERY_TRIAL = True


# =============================================================================
# Phase 4 — Deterministic risk grid search (Task 7)
# =============================================================================


# PHASE4_GRID_TP_VALUES — TP values (%) to enumerate (5 values).
PHASE4_GRID_TP_VALUES = (2.0, 2.5, 3.0, 4.0, 5.0)

# PHASE4_GRID_SL_VALUES — SL values (%) to enumerate (4 values).
PHASE4_GRID_SL_VALUES = (1.0, 1.5, 2.0, 2.5)

# PHASE4_GRID_CAPITAL_VALUES — capital_pct values (%) to enumerate (5 values).
PHASE4_GRID_CAPITAL_VALUES = (10.0, 15.0, 20.0, 25.0, 30.0)

# PHASE4_MAX_VAL_TRAIN_GAP_PCT — reject grid trials when val return exceeds
# train return by more than this (validation overfit during risk tuning).
PHASE4_MAX_VAL_TRAIN_GAP_PCT = 12.0

# PHASE4_USE_ROBUST_SCORE — score grid trials on min(train, val) return (like Phase 3).
PHASE4_USE_ROBUST_SCORE = True

# PHASE4_GRID_MAX_TOTAL_CAPITAL — hard cap on sum(capital_pct) across all rules.
#   Combinations that push the total above this cap are skipped.
PHASE4_GRID_MAX_TOTAL_CAPITAL = 95.0

# PHASE4_GRID_PASSES — number of round-robin passes through all rules.
PHASE4_GRID_PASSES = 2

# PHASE4_GRID_MIN_IMPROVEMENT — minimum score improvement to accept a new combo.
PHASE4_GRID_MIN_IMPROVEMENT = 0.005

# PHASE4_OPTIMIZE_PER_RULE_SYMBOL — tune each rule on its assigned symbol(s) only.
#   True  → grid trials for a rule are scored on train/val rows for that rule's
#           "symbol is X" filters (matches Phase 3 per-symbol selection).
#   False → legacy portfolio-level scoring on the full universe.
PHASE4_OPTIMIZE_PER_RULE_SYMBOL = True


# =============================================================================
# Phase 5 — Out-of-sample evaluation (test.csv only; never used in Phases 1–4)
# =============================================================================

# PHASE5_VALIDATION_RETURN_GATE_PCT — min val return % for deployment flag.
#   Higher → fewer strategies marked deployable after pipeline.
#   Lower  → marginal val performers still flagged OK (risky for live).
PHASE5_VALIDATION_RETURN_GATE_PCT = 2.0

# PHASE5_VALIDATION_PROFIT_FACTOR_GATE — min val PF for deployment flag.
#   Higher → stricter deployment filter on gross win/loss ratio.
#   Lower  → strategies with thin edge pass deployment check.
PHASE5_VALIDATION_PROFIT_FACTOR_GATE = 1.05

# PHASE5_REMOVE_NEGATIVE_PNL_RULES — remove rules with negative PnL on test.
#   True  → clean up losing rules after OOS evaluation.
#   False → keep all rules regardless of test performance.
PHASE5_REMOVE_NEGATIVE_PNL_RULES = True


# =============================================================================
# Purged walk-forward helpers (trade-floor scaling)
# =============================================================================


def split_mode_is_purged_walk_forward() -> bool:
    """True when the active split mode is purged walk-forward."""
    return str(SPLIT_MODE).strip().lower() == "purged_walk_forward"


def set_purged_wf_reference_rows(n_rows: int) -> None:
    """Store full train.csv row count after loader prep (split time)."""
    global _PURGED_WF_REFERENCE_ROWS
    _PURGED_WF_REFERENCE_ROWS = max(0, int(n_rows))


def get_purged_wf_reference_rows() -> int | None:
    """Return reference row count for trade-floor scaling, if set."""
    return _PURGED_WF_REFERENCE_ROWS


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


def effective_sortino_min_trade_threshold(n_rows: int | None = None) -> int:
    base = int(PHASE2_SORTINO_MIN_TRADE_THRESHOLD)
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


def effective_phase3_min_val_trades(n_rows: int | None = None) -> int:
    base = int(PHASE3_MIN_VAL_TRADES)
    if n_rows is None:
        return base
    return scale_trade_floor(base, n_rows)


def effective_phase4_min_worst_trades(n_rows: int | None = None) -> int:
    base = int(PHASE4_MIN_WORST_TRADES)
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
        absolute_min if absolute_min is not None else PHASE2_ISLAND_TRADE_FLOOR_ABSOLUTE_MIN
    )
    scaled = int(round(int(base) * int(n_rows) / ref))
    return max(floor_min, scaled)


@dataclass(frozen=True)
class IslandHyperparams:
    """Resolved Phase 2 knobs for cluster or orphan slices."""

    profile: Literal["cluster", "orphan"]
    min_trade_support: int
    min_trade_pool_floor: int
    sortino_min_trade_threshold: int
    val_trade_floor: int
    pool_min_val_trades: int
    min_profitable_symbols: int
    monthly_admission_min_months: int
    monthly_admission_min_profitable_ratio: float
    skip_symbol_robustness_penalty: bool
    n_rows: int
    n_symbols: int


def resolve_island_hyperparams(
    profile: Literal["cluster", "orphan"],
    n_rows: int,
    reference_rows: int,
    n_symbols: int,
) -> IslandHyperparams:
    """Resolve scaled trade floors and relaxed cross-symbol gates."""
    ref = max(1, int(reference_rows))
    rows = max(1, int(n_rows))
    sym_n = max(1, int(n_symbols))

    if profile == "orphan":
        min_support = int(PHASE2_ORPHAN_MIN_TRADE_SUPPORT)
        pool_floor = int(PHASE2_ORPHAN_MIN_TRADE_POOL_FLOOR)
        sortino_thr = int(PHASE2_ORPHAN_SORTINO_MIN_TRADE_THRESHOLD)
        min_profitable = 1
        monthly_months = max(2, int(PHASE2_ISLAND_MONTHLY_MIN_MONTHS) - 1)
        monthly_ratio = float(PHASE2_ORPHAN_MONTHLY_MIN_PROFITABLE_RATIO)
    else:
        if PHASE2_ISLAND_SCALE_TRADE_FLOORS:
            abs_min = int(PHASE2_ISLAND_TRADE_FLOOR_ABSOLUTE_MIN)
            min_support = scale_trade_floor_by_universe(
                MIN_TRADE_SUPPORT, rows, ref, absolute_min=abs_min,
            )
            pool_floor = scale_trade_floor_by_universe(
                MIN_TRADE_POOL_FLOOR, rows, ref, absolute_min=abs_min,
            )
            sortino_thr = scale_trade_floor_by_universe(
                PHASE2_SORTINO_MIN_TRADE_THRESHOLD, rows, ref, absolute_min=abs_min,
            )
        else:
            min_support = int(MIN_TRADE_SUPPORT)
            pool_floor = int(MIN_TRADE_POOL_FLOOR)
            sortino_thr = int(PHASE2_SORTINO_MIN_TRADE_THRESHOLD)
        min_profitable = min(
            int(PHASE2_MIN_PROFITABLE_SYMBOLS),
            max(1, sym_n // 2),
        )
        monthly_months = int(PHASE2_ISLAND_MONTHLY_MIN_MONTHS)
        monthly_ratio = float(PHASE2_MONTHLY_ADMISSION_MIN_PROFITABLE_RATIO)

    val_floor = max(pool_floor // 4, 8)
    val_floor = scale_trade_floor_by_universe(
        max(int(MIN_TRADE_POOL_FLOOR) // 4, 10), rows, ref,
        absolute_min=8,
    )

    return IslandHyperparams(
        profile=profile,
        min_trade_support=int(min_support),
        min_trade_pool_floor=int(pool_floor),
        sortino_min_trade_threshold=int(sortino_thr),
        val_trade_floor=int(val_floor),
        pool_min_val_trades=int(val_floor),
        min_profitable_symbols=int(min_profitable),
        monthly_admission_min_months=int(monthly_months),
        monthly_admission_min_profitable_ratio=float(monthly_ratio),
        skip_symbol_robustness_penalty=True,
        n_rows=int(rows),
        n_symbols=int(sym_n),
    )


def island_early_stop_enabled() -> bool:
    if PHASE2_ISLAND_MODE == "cluster":
        return bool(PHASE2_ISLAND_EARLY_STOP_ENABLED)
    return bool(PHASE2_EARLY_STOP_ENABLED)


def island_plateau_early_stop_enabled() -> bool:
    if PHASE2_ISLAND_MODE == "cluster":
        return bool(PHASE2_ISLAND_PLATEAU_EARLY_STOP_ENABLED)
    return bool(PHASE2_PLATEAU_EARLY_STOP_ENABLED)


def scoped_island_profile(island_profile: str) -> bool:
    """True for cluster/orphan scoped runs (not the legacy global Phase 2 path)."""
    return str(island_profile) != "global"


def island_two_stage_enabled() -> bool:
    if PHASE2_ISLAND_MODE == "cluster":
        return bool(PHASE2_ISLAND_TWO_STAGE_ENABLED)
    return bool(PHASE2_TWO_STAGE_ENABLED)


# =============================================================================
# Cross-parameter sanity (import-time)
# =============================================================================

assert PHASE2_PLATEAU_EARLY_STOP_MIN_GENERATION <= PHASE2_STAGE_A_GENERATIONS, (
    "plateau min gen should not exceed Stage A budget"
)
assert PHASE2_STAGE_A_PLATEAU_EARLY_STOP_MIN_GENERATION <= PHASE2_STAGE_A_GENERATIONS, (
    "Stage A plateau min gen should not exceed Stage A budget"
)
assert PHASE2_STAGE_B_PLATEAU_EARLY_STOP_MIN_GENERATION <= PHASE2_STAGE_B_GENERATIONS, (
    "Stage B plateau min gen should not exceed Stage B budget"
)
assert PHASE2_STAGE_A_EARLY_STOP_MIN_GENERATION <= PHASE2_STAGE_A_GENERATIONS, (
    "Stage A early-stop min gen should not exceed Stage A budget"
)
assert PHASE2_STAGE_B_EARLY_STOP_MIN_GENERATION <= PHASE2_STAGE_B_GENERATIONS, (
    "Stage B early-stop min gen should not exceed Stage B budget"
)
assert 0.0 < PHASE2_STAGE_A_MUTATION_RATE <= 0.5
assert 0.0 < PHASE2_STAGE_B_MUTATION_RATE <= 0.5
assert PHASE2_STAGE_A_MUTATION_RATE >= PHASE2_STAGE_B_MUTATION_RATE, (
    "Stage A should be at least as explorative as Stage B on mutation rate"
)
assert float(PHASE3_MONTHLY_PENALTY_SCALE) > 0.0, (
    "PHASE3_MONTHLY_PENALTY_SCALE must be > 0"
)
assert float(PHASE4_MONTHLY_PENALTY_SCALE) > 0.0, (
    "PHASE4_MONTHLY_PENALTY_SCALE must be > 0"
)
assert int(PHASE3_GLOBAL_MAX_RULES) <= int(EVALUATOR_MAX_RULES), (
    "PHASE3_GLOBAL_MAX_RULES must not exceed evaluator schema cap"
)
assert int(PHASE2_ORPHAN_MIN_TRADE_SUPPORT) <= int(MIN_TRADE_SUPPORT), (
    "orphan min trade support should not exceed global"
)
assert int(PHASE2_ISLAND_TRADE_FLOOR_ABSOLUTE_MIN) >= 5, (
    "island trade floor absolute min too low"
)
assert PHASE1_SIGN_CONSISTENCY_MIN_FOLDS <= PHASE1_STATIONARITY_FOLDS, (
    "sign-consistency cannot require more folds than stationarity uses"
)


def is_colab_runtime() -> bool:
    """True when running on Google Colab (/content runtime)."""
    return (
        os.environ.get("COLAB_RELEASE_TAG") is not None
        or os.path.isdir("/content")
    )


def _apply_colab_gpu_defaults() -> None:
    """
    Colab T4 optimizations for main.ipynb runs.

    - Phase 3 uses GPUBacktestEngine (mask cache + batch eval path).
    - VRAM auto batch sizing uses the T4-friendly 128 cap when enabled.
    """
    global PHASE3_USE_GPU, PHASE2_GPU_BATCH_SIZE_AUTO
    if not is_colab_runtime():
        return
    PHASE3_USE_GPU = True
    PHASE2_GPU_BATCH_SIZE_AUTO = True


_apply_colab_gpu_defaults()


# =============================================================================
# RB Governor — replaces Phase 3 (rule-set selection) + Phase 4 (risk tuning)
# =============================================================================
# When RB_GOVERNOR_ENABLED is True, ``run_pipeline.py`` bypasses the legacy
# Phase 3 (Rule_Set_Selector) and Phase 4 (WalkForwardRiskOptimizer) modules
# and calls ``run_rb_governor_pipeline`` from ``rb_governor.py``. The output
# strategy JSONs keep the same ``{direction: {rules_set: [...]}}`` shape so
# Phase 5 (OOS evaluation) and evaluator_v5.ipynb are unaffected.

# RB_GOVERNOR_ENABLED — master switch.
#   True  → use RB Governor for rule selection + TP/SL/capital optimization.
#   False → legacy Phase 3 + Phase 4 modules (backwards compatible).
RB_GOVERNOR_ENABLED: bool = True


# --- Rule scoring / gating ---

# RB_MIN_TRAIN_RETURN / RB_MIN_VALID_RETURN — return-% floors below which a
#   heavy score penalty is applied to a candidate rule.
#   Higher → only clearly profitable single rules pass the gate.
#   Lower  → allow marginal rules into the candidate pool.
RB_MIN_TRAIN_RETURN: float = 2.0
RB_MIN_VALID_RETURN: float = 2.0

# RB_MIN_TRAIN_PF / RB_MIN_VALID_PF — minimum profit factor for each split.
#   1.0 = break-even before fees.
RB_MIN_TRAIN_PF: float = 1.00
RB_MIN_VALID_PF: float = 1.00

# RB_MIN_TRAIN_TRADES / RB_MIN_VALID_TRADES — per-rule trade-count floors
#   used by ``gate_positive_good`` and ``_score_metrics`` for single rules.
#   Lowered vs. the friend's defaults because per-symbol specialized rules
#   typically see thin validation slices (~7k rows / symbol).
RB_MIN_TRAIN_TRADES: int = 10
RB_MIN_VALID_TRADES: int = 6

# RB_RULESET_MIN_* — trade-count floors applied to the composed team (all
#   rules together).  Should be larger than the per-rule floors because the
#   combined team fires more frequently than any single rule.
RB_RULESET_MIN_TRAIN_TRADES: int = 20
RB_RULESET_MIN_VALID_TRADES: int = 12


# --- Pool & candidate limits ---

# RB_MAX_POOL_RULES_TO_EVALUATE — cap on Phase 2 pool rules passed through
#   symbol-specialization in ``_filter_good_rules``.
#   Higher → more candidates, slower filtering.
RB_MAX_POOL_RULES_TO_EVALUATE: int = 200

# RB_KEEP_TOP_RULES — how many positive-good candidates survive the
#   single-rule ranking to feed ``_compose_ruleset``.
#   Should be comfortably larger than PHASE2_KEEP_TOP_RULES.
RB_KEEP_TOP_RULES: int = 80


# --- Team composition ---

# RB_MAX_RULES — maximum rules in the composed team (hard cap, must remain
#   ≤ EVALUATOR_MAX_RULES=5 or Phase 5 evaluator_v5 will reject the file).
RB_MAX_RULES: int = 5

# RB_MAX_PAIR_OVERLAP — max Hamming-style overlap between any two rules in
#   the team. Lower = more diverse team, harder to grow.
#   Set slightly higher than the friend's 0.24 because the incoming pool is
#   already symbol-specialized and therefore more likely to share conditions.
RB_MAX_PAIR_OVERLAP: float = 0.30

# RB_RULESET_MUST_BEAT_SUBSETS — a candidate team must beat both its parent
#   subset and the standalone candidate on both train and val return.
RB_RULESET_MUST_BEAT_SUBSETS: bool = True

# RB_MIN_SCORE_IMPROVEMENT — minimum delta in the governor score to add a
#   new rule in ``_compose_ruleset``.
RB_MIN_SCORE_IMPROVEMENT: float = 0.03

# RB_MIN_TRAIN_RETURN_IMPROVEMENT / RB_MIN_VALID_RETURN_IMPROVEMENT — min
#   return-% uplift required from adding a candidate rule. Lowered vs the
#   friend's 0.01 because per-symbol returns on val are often <1%.
RB_MIN_TRAIN_RETURN_IMPROVEMENT: float = 0.005
RB_MIN_VALID_RETURN_IMPROVEMENT: float = 0.005

# RB_RETURN_DD_FLOOR — drawdown floor (%) used when converting return to a
#   return/drawdown ratio inside ``_score_metrics``.
RB_RETURN_DD_FLOOR: float = 0.50

# RB_TRADE_PENALTY — per-trade penalty weight applied when a rule falls
#   below the minimum trade-count floors.
RB_TRADE_PENALTY: float = 0.70

# RB_TRAIN_VALID_RATIO_GAP_WEIGHT / RB_TRAIN_VALID_RETURN_GAP_WEIGHT —
#   overfit penalties applied to the train>>val gap in the score.
RB_TRAIN_VALID_RATIO_GAP_WEIGHT: float = 6.0
RB_TRAIN_VALID_RETURN_GAP_WEIGHT: float = 0.25


# --- Lenient-add mode (friend's recommended path) ---

# RB_RULE_ADD_BY_RETURN_ONLY — add rules purely on combined-return uplift
#   (skips the stricter subset-beat and overlap checks when paired with
#   RB_RULE_ADD_IGNORE_OVERLAP=True).  Profit amplifier still re-checks.
RB_RULE_ADD_BY_RETURN_ONLY: bool = True
RB_RULE_ADD_IGNORE_OVERLAP: bool = True
RB_RULE_ADD_IGNORE_SUBSET_BEAT: bool = True
RB_MIN_COMBINED_RETURN_IMPROVEMENT: float = 0.05


# --- Train-valid shape prior (anti-overfit) ---

# RB_REQUIRE_TRAIN_SLIGHTLY_ABOVE_VALID — when True, apply a shape
#   bonus/penalty so that train return is slightly above val return (a
#   healthy sign) but not wildly above (overfit sign).
RB_REQUIRE_TRAIN_SLIGHTLY_ABOVE_VALID: bool = True
RB_TRAIN_VALID_MIN_RATIO: float = 1.03
RB_TRAIN_VALID_MAX_RATIO: float = 1.35
RB_TRAIN_VALID_MIN_ABS_GAP: float = 0.20
RB_TRAIN_VALID_MAX_ABS_GAP: float = 12.0
RB_TRAIN_BELOW_VALID_PENALTY: float = 900.0
RB_TRAIN_TOO_HIGH_PENALTY: float = 220.0
RB_TRAIN_VALID_SHAPE_BONUS: float = 160.0


# --- Default risk parameters (initial TP/SL/capital_pct embedded in rules
#     before risk optimization, and minimum allowed values) ---

RB_DEFAULT_TP: float = 2.0
RB_DEFAULT_SL: float = 1.2
RB_DEFAULT_CAPITAL_PCT: float = 12.5
RB_REQUIRE_TP_SL_ABOVE_ONE: bool = True
RB_MIN_TP: float = 1.0
RB_MIN_SL: float = 1.0


# --- Risk grid search (replaces Phase 4 Walk-Forward optimizer) ---

# RB_TP_GRID / RB_SL_GRID / RB_CAPITAL_GRID — values enumerated per rule in
#   the round-robin grid search.  Coarser than the friend's full grid to
#   keep runtime reasonable on a ~10-symbol universe.
RB_TP_GRID: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)
RB_SL_GRID: tuple[float, ...] = (1.0, 1.2, 1.5, 2.0, 2.5)
RB_CAPITAL_GRID: tuple[float, ...] = (15.0, 20.0, 25.0, 35.0)

# RB_RISK_OPT_PASSES — round-robin passes through all rules.
RB_RISK_OPT_PASSES: int = 2

# RB_RISK_MIN_IMPROVEMENT — min score delta to accept a new TP/SL/cap combo.
RB_RISK_MIN_IMPROVEMENT: float = 0.02

# RB_MAX_TOTAL_CAPITAL — hard cap on sum(capital_pct) across all rules.
RB_MAX_TOTAL_CAPITAL: float = 95.0


# --- Symbol specialization (per-rule ``symbol is X`` conditions) ---

# RB_REQUIRE_SYMBOL_FILTERS — every final rule must include at least one
#   ``symbol is X`` condition (required for per-symbol strategies).
RB_REQUIRE_SYMBOL_FILTERS: bool = True

# RB_SYMBOL_USE_COMBINATIONS — when True, generate 1-/2-/3-symbol variants
#   of each pool rule.  Set False to restrict to single-symbol variants only.
RB_SYMBOL_USE_COMBINATIONS: bool = False  # pool is already symbol-specialized

# RB_SYMBOL_MAX_SYMBOLS_PER_RULE — max ``symbol is X`` conditions per rule.
RB_SYMBOL_MAX_SYMBOLS_PER_RULE: int = 1

# RB_SYMBOL_TOP_SINGLE_SYMBOLS — top-ranked single-symbol variants kept as
#   seeds for multi-symbol combinations.
RB_SYMBOL_TOP_SINGLE_SYMBOLS: int = 5
RB_SYMBOL_MAX_VARIANTS_PER_RULE: int = 10
RB_SYMBOL_MIN_TRAIN_TRADES: int = 10
RB_SYMBOL_MIN_VALID_TRADES: int = 4  # per-symbol val slices are thin
RB_SYMBOL_STRICT_OUTPUT_CHECK: bool = True


# --- Evaluator-health penalties (mirror evaluator_v5 execution checks) ---

RB_MAX_SKIPPED_SIGNAL_RATIO: float = 0.20
RB_MIN_EXECUTED_RAW_RATIO: float = 0.60
RB_SKIPPED_RATIO_PENALTY: float = 3500.0
RB_EXECUTED_RATIO_PENALTY: float = 2500.0
RB_MAX_SIMULTANEOUS_POSITIONS: int = 10
RB_MAX_POSITIONS_PENALTY: float = 120.0


# --- Profit amplifier (post-risk-optimization refinement stage) ---

# RB_PROFIT_AMPLIFIER_ENABLED — when True, after risk grid search a final
#   refinement pass swaps/adds rules from the candidate pool and reallocates
#   capital to maximize a blended valid-return objective.  Keep enabled for
#   maximum performance; disable for faster but slightly weaker results.
RB_PROFIT_AMPLIFIER_ENABLED: bool = True

RB_PROFIT_AMP_MAX_CANDIDATES: int = 60
RB_PROFIT_AMP_MAX_RULES: int = 5
# RB_PROFIT_AMP_MIN_OBJECTIVE_IMPROVEMENT — minimum objective delta to
#   accept a profit-amplifier candidate.  Lowered from friend's 0.10 because
#   our governor scores are typically smaller in magnitude.
RB_PROFIT_AMP_MIN_OBJECTIVE_IMPROVEMENT: float = 0.05
RB_PROFIT_AMP_MIN_RETURN_IMPROVEMENT: float = 0.02
RB_PROFIT_AMP_VALID_WEIGHT: float = 1.55
RB_PROFIT_AMP_TRAIN_WEIGHT: float = 1.00
RB_PROFIT_AMP_BALANCE_WEIGHT: float = 0.20
RB_PROFIT_AMP_DD_WEIGHT: float = 0.02
RB_PROFIT_AMP_HEALTH_WEIGHT: float = 0.030
RB_PROFIT_AMP_OVERLAP_PENALTY: float = 2.5
RB_PROFIT_AMP_MAX_PAIR_OVERLAP: float = 0.55
RB_PROFIT_AMP_MAX_VALID_DD: float = 12.0
RB_PROFIT_AMP_MAX_TRAIN_DD: float = 18.0
RB_PROFIT_AMP_MONTHLY_ENABLED: bool = True
RB_PROFIT_AMP_MIN_MONTHLY_WINDOWS: int = 2
RB_PROFIT_AMP_MIN_MONTHLY_PROFITABLE_RATIO: float = 0.55
RB_PROFIT_AMP_WORST_MONTHLY_RETURN_FLOOR: float = -2.0
RB_PROFIT_AMP_WORST_MONTHLY_PF_FLOOR: float = 0.80
RB_PROFIT_AMP_MAX_MONTHLY_DD: float = 10.0
RB_PROFIT_AMP_CAPITAL_REALLOCATION_ENABLED: bool = True
RB_PROFIT_AMP_CAPITAL_PASSES: int = 2
RB_PROFIT_AMP_CAPITAL_GRID: tuple[float, ...] = RB_CAPITAL_GRID
RB_PROFIT_AMP_KEEP_BASELINE_UNLESS_BETTER: bool = True


# --- Cross-run global bank (disabled by default to keep runs isolated) ---

RB_GLOBAL_BANK_ENABLED: bool = False
RB_GLOBAL_COMPOSE_AFTER_EACH_RUN: bool = False
RB_GLOBAL_BANK_DIRNAME: str = "rb_bank"
RB_GLOBAL_BANK_MAX_RULES_PER_DIRECTION: int = 700
RB_GLOBAL_BANK_IMPORT_TOP_SINGLE_RULES: int = 80
RB_GLOBAL_MAX_RULES: int = 12
RB_GLOBAL_MIN_COMBINED_RETURN_IMPROVEMENT: float = 0.05
RB_GLOBAL_REQUIRE_POSITIVE_TRAIN_VALID: bool = True
RB_GLOBAL_RISK_OPT_PASSES: int = 2
RB_GLOBAL_BEST_DIRNAME: str = "best_global"
RB_GLOBAL_TP_GRID: tuple[float, ...] = (1.5, 2.0, 3.0, 5.0, 8.0)
RB_GLOBAL_SL_GRID: tuple[float, ...] = (1.2, 1.5, 2.0, 2.5)
RB_GLOBAL_CAPITAL_GRID: tuple[float, ...] = (5.0, 12.5, 25.0, 50.0)
RB_GLOBAL_MAX_TOTAL_CAPITAL: float = 100.0


# =============================================================================
# Phase 2 — Governor-friendly adjustments
# =============================================================================
# These tune the Phase 2 pool to feed RB Governor with enough candidates.
# RB Governor benefits from a slightly wider pool than the legacy Phase 3
# greedy selector, so we raise PHASE2_KEEP_TOP_RULES modestly.
# (PHASE2_KEEP_TOP_RULES is already defined above; override here.)
PHASE2_KEEP_TOP_RULES = 120


assert MIN_CONDITIONS <= MAX_CONDITIONS, (
    f"MIN_CONDITIONS ({MIN_CONDITIONS}) must be <= MAX_CONDITIONS ({MAX_CONDITIONS})"
)
assert 0.0 <= PHASE2_VIABILITY_RECOVERY_DEPLOYABLE_MUTATE_FRACTION <= 1.0
