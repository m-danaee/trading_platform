"""
Single source of truth for pipeline hyperparameters.

All modules import from here; do not duplicate defaults elsewhere.

File layout
-----------
  1. Global randomness (``GLOBAL_SEED``, ``get_seed``)
  2. Phase 0 — paths, schema, train/val split (holdout+embargo), backtest, logging
  3. Phase 1 — feature selection + GPU row budget (Phase 1→2 bridge)
  4. Phase 2 — rule evolution (NSGA-III): risk, genome, gates, islands
  5. Phase 3 — rule-team selection (legacy greedy path)
  6. Phase 4 — walk-forward / grid risk tuning (legacy Optuna path)
  7. Monthly windows — shared Phase 3/4 penalty knobs
  8. Phase 5 — out-of-sample evaluation (test.csv only)
  9. RB Governor — unified rule selection + risk tuning (replaces Phase 3+4)
 10. Helpers — path resolvers, trade-floor scaling, island hyperparams
 11. Import-time assertions + Colab runtime defaults

Pipeline phases
---------------
  Phase 0  Paths, schema, train/val split (holdout+embargo), backtest constants
  Phase 1  Feature selection (train.csv only)
  Phase 2  NSGA-III rule-pool evolution (GPU backtests)
  Phase 3  Greedy + NSGA-II rule-team selection  *(skipped when RB Governor on)*
  Phase 4  Walk-forward TP/SL/capital optimization  *(skipped when RB Governor on)*
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
  Fees / horizon mismatch          FEE_PCT, TAIL_DROP_ROWS, MAX_HOLD_CANDLES,
                                   HOLDOUT_EMBARGO_CANDLES
                                   (must match evaluator_v5.ipynb)
  RB Governor too strict           RB_MIN_* ↓, RB_KEEP_TOP_RULES ↑

Environment overrides: DATA_ROOT, TRAIN_CSV_PATH, TEST_CSV_PATH,
                       PHASE2_GPU_BATCH_SIZE, PHASE2_GPU_BATCH_SIZE_AUTO
"""


from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Literal

import pandas as pd

# Repo root (parent of gpu_fuzzy_trader/) — paths outside per-run OUTPUTS_DIR.
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir))

_logger = logging.getLogger(__name__)


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
    os.path.join(
        DATA_ROOT, "train.csv") if DATA_ROOT else "data/train.csv",
)
TEST_CSV_PATH = _env_str(
    "TEST_CSV_PATH",
    os.path.join(DATA_ROOT, "test.csv") if DATA_ROOT else "data/test.csv",
)

# Cached splits from train.csv (Phases 2–5). Rebuilt when train.csv is newer.
TRAIN_70_PATH = "data/train_70.parquet"
VALIDATION_30_PATH = "data/validation_30.parquet"
VALIDATION_FITNESS_PATH = "data/validation_fitness.parquet"
VALIDATION_SELECTION_PATH = "data/validation_selection.parquet"

OUTPUTS_DIR = "outputs"
RUN_LOG_PATH = os.path.join(OUTPUTS_DIR, "run.log")
REPORTS_DIR = "outputs/reports"

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
#   holdout             → single per-symbol chronological split with embargo
#                         (288 bars dropped between train and val).
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
HOLDOUT_EMBARGO_CANDLES = 288


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
PURGED_WF_EMBARGO_CANDLES = 288

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

# PHASE1_DISABLED — bypass MI ranking, sign-consistency, stationarity, and top-K
# selection. When True, Feature_Selector.run returns ALL features that pass the
# dispersion filter (PHASE1_DISPERSION_THRESHOLD) for both directions, with modes
# detected by Feature_Detector. Phase 2 then evolves over the full feature set.
#   True  → larger GA search space, more GPU RAM per chromosome, no MI prefilter.
#   False → normal top-K MI-ranked selection (PHASE1_TOP_K_FEATURES=20).
# 2026-07-16 (plan 004): False — full feature set under-searches with pop=60/gens=20.
PHASE1_DISABLED: bool = False

# PHASE1_MAX_FEATURE_OVERLAP — max shared feature names between long & short lists.
#   Enforced as int(TOP_K × overlap) shared names (e.g. 25 × 0.8 → 20 shared).
#   Higher → more shared features across directions; smaller combined gene space.
#   Lower  → more direction-specific lists; better asymmetry, less redundancy.
PHASE1_MAX_FEATURE_OVERLAP = 0.8

# PHASE1_ASYMMETRIC_TARGET — separate MI targets for long vs short.
#   True  → direction-specific feature rankings (recommended).
#   False → shared target; long/short pools share more structure.
PHASE1_ASYMMETRIC_TARGET = True

# --- Sign consistency across stationarity folds ---

# PHASE1_REQUIRE_SIGN_CONSISTENCY — drop features whose Spearman sign flips.
#   True  → fewer unstable features; stricter shortlist.
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

# --- Stationarity (reduce time-varying features) ---

# PHASE1_STATIONARITY_FOLDS — chronological chunks for stability tests.
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

# =============================================================================
# Phase 1 → Phase 2 bridge — GPU row budget & JAX performance
# =============================================================================

# PHASE1_SAMPLING_TOTAL — max rows subsampled for Phase 2 GPU backtests.
# Peak GPU RAM scales ~linearly with this value (largest VRAM lever).
#   Higher → more statistical power, slower, OOM risk on small GPUs.
#   Lower  → faster, less RAM; trade/support floors may need proportional cut.
PHASE1_SAMPLING_TOTAL = 1_500_000

# PHASE2_PER_EPOCH_WINDOW_ROTATION — rotate train-window start per epoch
#   True  → each epoch samples a different contiguous sub-window from the
#           training data, using a deterministic per-epoch seed derived
#           from (island_seed, epoch_idx). The per-sym request is capped
#           to fit within the largest safe range so the RNG start bar
#           branch in _sample_df fires.
#   False → preserve pre-task-1 behavior: sample once at cluster init
#           (useful for A/B comparison / regression guard).
#   → fixes audit finding #1 (per-epoch window resampling is dead),
#     implements N2
PHASE2_PER_EPOCH_WINDOW_ROTATION = True

# PHASE2_PER_EPOCH_WINDOW_SEED_MODE — how to derive the per-epoch seed.
#   "hash_island_epoch" → hash(island_seed, epoch_idx) via SHA-256,
#                          deterministic, no RNG state leak.
PHASE2_PER_EPOCH_WINDOW_SEED_MODE = "hash_island_epoch"

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
#   True  → apply tiers in _gpu_runtime (12 GiB RAM → 32; T4 ≤16 GiB VRAM → 256).
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
#   200 = 200 (population) × 1 (cache generation) × 1.0.
#   Higher → more cache hits; less RAM.
#   Lower  → less RAM; more re-evaluations.
# Halved twice (1200 → 600 → 200) to reduce Colab RAM footprint (~0.4 GB per cut).
# Rationale: cache hit rate was observed at 0-4% in the 2026-07-05 22:09 log,
# so successive cuts are near-free — the working set is ~1 gen at 200 pop.
PHASE2_EVAL_GLOBAL_CACHE_MAX_SIZE = 200

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
PHASE2_SL = 1.2

# PHASE2_CAPITAL_PCT — % of equity allocated per rule signal in Phase 2.
#   Higher → larger simulated positions; drawdown and return scale up.
#   Lower  → conservative sizing; may understate overlap effects until Phase 4.
PHASE2_CAPITAL_PCT = 18.0


# =============================================================================
# Phase 2 — Rule genome
# =============================================================================

# MIN_CONDITIONS / MAX_CONDITIONS — active fuzzy conditions per rule.
#   Higher MIN → stricter rules, fewer matching bars, higher precision target.
#   Lower MIN → broader rules, more trades, risk of weak patterns.
#   Higher MAX → allow complex rules (if encoding supports variable count).
#   Lower MAX → force simplicity; more generalization, less specificity.
MIN_CONDITIONS = 4
MAX_CONDITIONS = 5

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
# 2026-07-11: 200→120 — island runs (180k–240k rows) were failing train_trade_floor
# for ~70%+ of the pop; softer support target lifts viable / trading rules.
# 2026-07-11b: 120→60 — prefer many moderate-support rules over few fat specialists
# (portfolio support via larger rule sets; one-symbol islands scale further down).
MIN_TRADE_SUPPORT = 60

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
# 2026-07-11: 35→25 — fewer hard kills of thin-but-real island rules.
# 2026-07-11b: 25→15 — one-symbol islands + many-moderate-rules package.
MIN_TRADE_POOL_FLOOR = 15

# PHASE2_SUPPORT_PENALTY_WEIGHT_F1/F2/F3 — per-objective support penalty scale.
#   Higher → that objective punishes low support more (steer Sortino vs DD vs return).
#   Lower  → support matters less for that objective.
# 2026-07-11: F1 0.1→0.45 — zero-trade rules were barely punished on Sortino and
# cluttered the Pareto front (median_return=0.00% in run.log).
# 2026-07-11b: F1 0.45→0.25 — do not obsess over fat singles; keep some pressure
# so zero-trade junk still loses on Sortino.
PHASE2_SUPPORT_PENALTY_WEIGHT_F1 = 0.25  # Sortino objective
PHASE2_SUPPORT_PENALTY_WEIGHT_F2 = 0.6  # drawdown objective
PHASE2_SUPPORT_PENALTY_WEIGHT_F3 = 0.6  # return / win-rate objective

# PHASE2_USE_TOTAL_RETURN_OBJ — f3 uses robust return (min train, val) instead
# of profit_factor or win rate.
#   True  → f3 = -robust_return_pct (min of train/val return); aligns with OOS PnL.
#   False → f3 uses PHASE2_F3_OBJECTIVE (profit_factor or win_rate).
# With PHASE2_JOINT_TRAIN_VAL=False, "robust" return collapses to train-only and
# f1 (Sortino) ≈ f3 (return) → objective_corr_f1_f3≈1.0 (Pareto collapse in run.log).
# Use profit_factor for f3 so the front stays multi-objective.
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

# PHASE2_F3_OBJECTIVE — third objective: "profit_factor" (default when
# PHASE2_USE_TOTAL_RETURN_OBJ=False),
# "cv_fold_min" (min of CV fold returns), or "win_rate" (legacy).
#   profit_factor → f3 = -profit_factor (aligns with edge quality over noise).
#   cv_fold_min  → f3 = -min(CV fold returns); requires CvFoldValEvaluator
#                  which is too expensive for NSGA-III inner loop — disabled.
#   win_rate     → f3 = -win_rate (degenerate, not recommended).
# NOTE: PHASE2_USE_TOTAL_RETURN_OBJ=True (now default) takes precedence, so f3
# uses robust_return_pct instead of PHASE2_F3_OBJECTIVE.  The legacy F3_OBJECTIVE
# setting only takes effect when USE_TOTAL_RETURN_OBJ is False.
# CV-fold robustness is enforced at the pool-admission gate and RB Governor
# scoring stages instead.
PHASE2_F3_OBJECTIVE = "profit_factor"

# PHASE2_MIN_PROFITABLE_SYMBOLS_PENALTY — min profitable symbols before penalty.
#   Softer evolution nudge (default 3): adds to support_penalty when
#   n_profitable_symbols < this during fitness.  Stricter pool gate
#   PHASE2_MIN_PROFITABLE_SYMBOLS (default 3) is enforced separately at
#   admission via _symbol_robustness_penalty.
#   One-symbol islands override this via island_hyperparams.min_profitable_symbols.
PHASE2_MIN_PROFITABLE_SYMBOLS_PENALTY = 1

# PHASE2_SYMBOL_GENE_DONT_CARE_PROB — probability of forcing a symbol gene to
#   dont_care during mutation. Higher → more cross-symbol rules; prevents
#   symbol-locked evolution.
# 2026-07-11b: 0.75→0.15 — one-symbol islands should stay symbol-anchored;
# source_symbols OR-filters still attach at pool export.
PHASE2_SYMBOL_GENE_DONT_CARE_PROB = 0.15

# PHASE2_USE_ROBUST_RETURN_OBJ — store min(train_return, val_return) as
#   robust_return_pct on metrics when PHASE2_JOINT_TRAIN_VAL=True.
#   When PHASE2_USE_TOTAL_RETURN_OBJ=True (now default), this controls whether
#   the joint return uses min(train, val) or just train-only.
#   True  → robust_return_pct = min(train_return, val_return).
#   False → robust_return_pct = train_return (equivalent to no robustness).
PHASE2_USE_ROBUST_RETURN_OBJ = True

# PHASE2_SORTINO_MIN_TRADE_THRESHOLD — trade count below which Sortino is scaled down.
#   Used in Approach 2 to penalize low-trade-count rules.
# 2026-07-11b: 50→20 — one-symbol windows + moderate-support package.
PHASE2_SORTINO_MIN_TRADE_THRESHOLD = 20

# --- Return / quality floors (evolution + pool filtering) ---

# PHASE2_RETURN_FLOOR_PCT — min train return % to avoid feasibility penalty.
#   Higher → only profitable-on-train rules stay feasible; emptier search.
#   Lower  → more exploration; weak rules linger until other gates remove them.
# 2026-07-11: 0.5→0.25 — ease early feasibility collapse (train_return_floor ~90%).
PHASE2_RETURN_FLOOR_PCT = 0.25

# PHASE2_VAL_RETURN_FLOOR_PCT — min validation return % for feasibility.
#   Higher → stricter OOS alignment during evolution.
#   Lower  → allow negative val return during search (gates may still catch later).
# 2026-07-11: 1.0→0.25 — val_return_floor was failing ~95–98% of the pop early.
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
#   The original PHASE2_PROFIT_FACTOR_FLOOR=1.15; kept for the hard gate.
#   → fixes audit finding #9
PHASE2_PROFIT_FACTOR_FLOOR_ADMISSION = 1.15

# PHASE2_PROFIT_FACTOR_FLOOR — DEPRECATED alias for PHASE2_PROFIT_FACTOR_FLOOR_ADMISSION.
#   Kept for backward compat; do not use in new code. Tracks ADMISSION automatically.
#   → fixes audit finding #9: split into EVOLUTION (soft penalty) and ADMISSION (hard gate).
PHASE2_PROFIT_FACTOR_FLOOR = PHASE2_PROFIT_FACTOR_FLOOR_ADMISSION

# PHASE2_SYMBOL_MEDIAN_RETURN_FLOOR_PCT — min median return across symbols.
#   Higher → rules must work on typical symbols, not one outlier.
#   Lower  → single-symbol heroes can survive longer.
PHASE2_SYMBOL_MEDIAN_RETURN_FLOOR_PCT = 0.0

# PHASE2_MIN_PROFITABLE_SYMBOLS — min count of symbols with positive PnL.
#   Higher → demand broad cross-symbol edge; stricter for 10-symbol universe.
#   Lower  → allow niche symbol specialists.
# Island clusters are 3–4 symbols with min_profitable≈2; a global floor of 5
# is unreachable for island-evolved rules and starves the merged pool.
PHASE2_MIN_PROFITABLE_SYMBOLS = 3

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
#   Higher → larger pool for Phase 3 greedy selection.
#   Lower  → smaller pool; faster Phase 3, fewer combinations.
# widened for RB Governor candidate pool (was 80 for legacy Phase 3)
# 2026-07-11b: 80→150 — one-symbol islands → many moderate specialists.
PHASE2_KEEP_TOP_RULES = 150

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
#   0.0  → non-loss months count (flat range months OK — desired equity shape).
#   2.0  → month must earn at least +2% to count as good.
#   -1.0 → month counts if return >= -1% (more lenient non-loss bar).
PHASE2_MONTHLY_GOOD_RETURN_MIN_PCT = 0.0

# PHASE2_MONTHLY_ADMISSION_MIN_RATIO — fraction of monthly windows that must
# be profitable for a rule to be admitted (non-island path; island uses
# island_hyperparams.monthly_admission_min_profitable_ratio from this value).
#   0.400 → allow some flat/weak months; thicker pool for RB multi-symbol teams.
#   0.667 → rule must be profitable in two-thirds of windows; tighter stability.
# run.log: monthly gate emptied clusters (1→0, 2→0) then graceful-degraded to
# weak originals — ease slightly so stable non-loss rules survive.
PHASE2_MONTHLY_ADMISSION_MIN_RATIO = 0.40

# PHASE2_MONTHLY_ADMISSION_MIN_MONTHS — minimum number of monthly windows
# required before the gate is applied. validation_fitness is ~110 calendar days
# (~3×30d windows), so 4 was structurally incompatible and silently skipped.
# When fewer windows exist but at least one is available, the gate still runs
# (degraded) instead of being skipped.
# 2026-07-17: 3→2 — Colab holdout val (~2 months) always logged
# "only 2 monthly windows (< MIN_MONTHS=3); degraded mode". Align floor with
# actual window count so the gate is non-degraded when 2 windows exist.
#   Higher → skip/degrade more often on short data.
#   Lower  → require monthly evidence even on short trains.
PHASE2_MONTHLY_ADMISSION_MIN_MONTHS = 2


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
#           (Phase 3) & OOS (Phase 5). Robustness via purged 4-fold CV evaluator.
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
PHASE2_VAL_IN_FITNESS_PENALTY = True

# PHASE2_VAL_SIM_INTERVAL — run val backtest every N generations during
# evolution (default 3, was 1 before task-11).  With per-epoch window
# rotation (PHASE2_PER_EPOCH_WINDOW_ROTATION, post task-1), the val
# window is fixed across epochs, so val metrics for a given chromosome
# are deterministic and safe to cache.  Skipping val on gens 2/3 cuts
# GPU work ~33% with zero metric drift.
# Val ALWAYS runs on the epoch's last gen (pool-admission freshness).
#   1 → val every gen (legacy, expensive).
#   3 → val every 3rd gen (default post task-11; ~33% GPU savings).
# → fixes audit finding #10 (val every gen is wasteful when
# window is fixed; cache is safe)
PHASE2_VAL_SIM_INTERVAL = 2
assert PHASE2_VAL_SIM_INTERVAL >= 1, "PHASE2_VAL_SIM_INTERVAL must be >= 1"

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
# 2026-07-11b: False — short 20-gen one-symbol runs must use the full budget.
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

# 2026-07-11b: False — no early/plateau stop on short one-symbol budgets.
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
# 2026-07-11b: False — user requested full 20-gen budget, no early stop.
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

# Island-scoped variants (mirror PHASE2_ISLAND_PLATEAU_EARLY_STOP_* scoping).
PHASE2_ISLAND_PLATEAU_POST_RESTART_STOP_ENABLED = False
PHASE2_ISLAND_PLATEAU_POST_RESTART_STOP_PATIENCE = 8

# PHASE2_PLATEAU_MAX_RESTARTS — restarts per epoch before final break.
#   3       → up to 3 diversity restarts, then break on the next plateau.
#   0       → immediately break (disables restart regardless of ENABLED flag).
# 2026-07-11b: 3→1 — 20-gen one-symbol islands cannot afford triple restarts.
PHASE2_PLATEAU_MAX_RESTARTS = 1

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
# History: 200 → 100 (Colab RAM) → 75 (debug-scope economy). With ~9 symbols
# and 2 directions, 75 deployable elites provide enough warm-start diversity
# while saving ~25% RAM on the 12.7 GiB Colab host.
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

# --- Two-stage evolution: wide exploration → val-robust refinement ---

PHASE2_TWO_STAGE_ENABLED = True

# PHASE2_STAGE_A_GENERATIONS — Stage A (exploration) generation budget.
#   Higher → more diverse initial Pareto before val-focused Stage B.
#   Lower  → quicker handoff; Stage B may miss good regions.
# 2026-07-11b: 60→12 — scaled to PHASE2_GENERATIONS=20 (A:B ≈ 12:8).
PHASE2_STAGE_A_GENERATIONS = 12

# PHASE2_STAGE_B_GENERATIONS — Stage B (refinement) generation budget.
#   Higher → more val-robust polishing; total time = A + B gens.
#   Lower  → less refinement after exploration.
# 2026-07-11b: 36→8 — matches 20-gen one-symbol island budget.
PHASE2_STAGE_B_GENERATIONS = 8

# PHASE2_STAGE_B_SEED_TOP_K — elites from Stage A seeded into Stage B.
#   Higher → broader refinement starting set; slower Stage B per gen.
#   Lower  → refine only top performers; risk missing dark horses.
# 2026-07-11b: 50→20 — pop=60 short Stage B.
PHASE2_STAGE_B_SEED_TOP_K = 20

# PHASE2_STAGE_B_SEED_FRACTION — fraction of Stage B pop seeded from Stage A elites.
#   Higher → more refinement around known good regions; risk of clone collapse.
#   Lower  → more random exploration in Stage B.
PHASE2_STAGE_B_SEED_FRACTION = 0.35

# --- Stage A hyperparameters (exploration: higher mutation, stronger diversity) ---

# PHASE2_STAGE_A_MUTATION_RATE — per-gene mutation in Stage A.
#   Higher → more genetic exploration before Stage B refinement.
# 2026-07-11b: 0.25→0.35 — short Stage A needs aggressive exploration.
PHASE2_STAGE_A_MUTATION_RATE = 0.35

# PHASE2_STAGE_A_MUTATION_WEIGHTED_ACTIVATE_PROB — bias toward activating genes in A.
# 2026-07-11: 0.50→0.65 — more active conditions → more trades on island windows.
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
# 2026-07-11b: 30→15 — one-symbol moderate-support exploration.
PHASE2_STAGE_A_MIN_TRADE_SUPPORT = 15

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
#   5 = every 5th gen (default — saves ~80% of CPU enrichment cost).
PHASE2_ENRICH_SYMBOL_METRICS_EVERY_N_GENS = 5
# =============================================================================
# Phase 2 — NSGA-III search budget & archive
# =============================================================================

# PHASE2_POPULATION_SIZE — individuals per generation.
#   Higher → better Pareto coverage, ~linear GPU cost per generation.
#   Lower  → faster gens, risk of premature convergence.
# 2026-07-11b: 100→60 — one-symbol × many islands; keep GPU time manageable.
PHASE2_POPULATION_SIZE = 60

# PHASE2_GENERATIONS — per-island generation budget when ONE_SYMBOL_ISLANDS
#   (each symbol runs this many gens; wall-clock ≈ gens × n_symbols).
#   When clustering (legacy), this total is split across K clusters.
# 2026-07-11b: 96→20 — short full-budget runs (early stop disabled).
PHASE2_GENERATIONS = 20

PHASE2_ALGORITHM = "NSGA3"

# PHASE2_ARCHIVE_MAX_SIZE — max stored non-dominated solutions across gens.
#   Higher → richer elite memory; more memory, slower non-dominated sorting.
#   Lower  → leaner archive; may lose good rules found early.
# 2026-07-11b: 300→120 — pop=60 / short gens.
PHASE2_ARCHIVE_MAX_SIZE = 120

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

# PHASE2_ISLAND_MODE — scoped evolution layout.
#   "global"  → single universe-wide NSGA-III run (default).
#   "cluster" → K symbol clusters evolved as separate islands with migration.
PHASE2_ISLAND_MODE = "cluster"  # "global" | "cluster"
# PHASE2_ONE_SYMBOL_ISLANDS — when True, skip KMeans/corr clustering and give
#   each symbol its own island. Generation budget is per-island full
#   PHASE2_ISLAND_TOTAL_GENERATIONS for both one-symbol and multi-symbol
#   cluster modes (splitting a short budget across K clusters starves Stage B).
# 2026-07-16 (plan 004): False — one-symbol specialists conflict with RB
#   concentration gates; restore hybrid multi-symbol clusters (N_CLUSTERS=3).
PHASE2_ONE_SYMBOL_ISLANDS = False
# PHASE2_N_CLUSTERS — number of hybrid symbol clusters when island mode is active
#   and PHASE2_ONE_SYMBOL_ISLANDS is False (legacy clustering path).
PHASE2_N_CLUSTERS = 3
# PHASE2_CLUSTER_USE_RETURN_CORR — when True, build return-correlation embedding
#   and blend with feature-means (weights below) for a hybrid clustering that
#   groups symbols with similar return patterns.  Set False for legacy
#   feature-mean-only clustering.
#   Default True (feasible-search item 3): islands should group co-movers.
# Unused when PHASE2_ONE_SYMBOL_ISLANDS=True.
PHASE2_CLUSTER_USE_RETURN_CORR = True
# PHASE2_CLUSTER_FEATURE_WEIGHT / CORR_WEIGHT — blend weights for the
#   feature-mean block and the return-correlation embedding.  Normalised to
#   sum=1 internally.  Corr-heavy so co-movement dominates; features break ties.
PHASE2_CLUSTER_FEATURE_WEIGHT = 0.3
PHASE2_CLUSTER_CORR_WEIGHT = 0.7
# PHASE2_ISLAND_TOTAL_GENERATIONS — per-island gens when ONE_SYMBOL; else split.
PHASE2_ISLAND_TOTAL_GENERATIONS = PHASE2_GENERATIONS
# PHASE2_ISLAND_EPOCH_GENERATIONS — generations per island epoch before migration.
# 2026-07-11b: 20→10 — two epochs fit Stage A(12)+B(8) handoff cleanly.
PHASE2_ISLAND_EPOCH_GENERATIONS = 10
# PHASE2_ISLAND_MIN_EPOCH_GENERATIONS — skip epochs with fewer remaining gens
# than this threshold (engine rebuild ~30s with negligible benefit for <5 gens).
PHASE2_ISLAND_MIN_EPOCH_GENERATIONS = 4
# Island overrides — two-stage exploration then refinement in cluster mode.
# Stage A soft floors (RETURN_FLOOR_PCT=0, SOFT_FEASIBILITY, support=15) apply
# when PHASE2_ISLAND_TWO_STAGE_ENABLED=True; Stage B uses global floors.
PHASE2_ISLAND_TWO_STAGE_ENABLED = True
PHASE2_ISLAND_EARLY_STOP_ENABLED = False
# 2026-07-11b: False — full 20-gen budget, no plateau early stop.
PHASE2_ISLAND_PLATEAU_EARLY_STOP_ENABLED = False
PHASE2_ISLAND_PLATEAU_BLOCK_WHEN_DEPLOYABLE_ZERO: bool = True
PHASE2_ISLAND_PLATEAU_EARLY_STOP_PATIENCE: int = 10
# PHASE2_ISLAND_SCALE_TRADE_FLOORS — scale support floors to island row count.
PHASE2_ISLAND_SCALE_TRADE_FLOORS = True
# 2026-07-11b: 10→8 — one-symbol absolute floor for moderate-support rules.
PHASE2_ISLAND_TRADE_FLOOR_ABSOLUTE_MIN = 8
PHASE2_ISLAND_MONTHLY_MIN_MONTHS = 3
# Migration — exchange top elites between islands every N epochs.
# PHASE2_MIGRATION_ENABLED — master switch for inter-island elite exchange.
# 2026-07-16 (plan 004): True — multi-symbol clusters share elites across islands.
PHASE2_MIGRATION_ENABLED: bool = True
# PHASE2_MIGRATION_EPOCH_INTERVAL — DEPRECATED.  This is a no-op in
# the current code path.  Migration fires once after each cluster
# finishes (sequential post-cluster chain, see _run_cluster_islands).
# Retained for backward compat only; do not use in new code.
# → fixes audit finding #6 (dead migration helper was misleading
# future maintainers about how migration actually works)
PHASE2_MIGRATION_EPOCH_INTERVAL = 1
PHASE2_MIGRATION_TOP_K = 5
PHASE2_MIGRATION_REQUIRE_DEPLOYABILITY = True
PHASE2_MIGRATION_MIN_VAL_RETURN_PCT = 1.0
PHASE2_MIGRATION_MIN_VAL_TRADES = None          # None = use island trade floor

# PHASE2_MIGRATION_SEED_FRACTION — fraction of the live population overwritten by
# migrants at epoch boundaries. Decoupled from PHASE2_ARCHIVE_SEED_FRACTION (which
# governs cross-run warm-start and stays 0.25). 0.10 = 20 of 200 slots, so
# migrants displace ≤10% of converged locals.
PHASE2_MIGRATION_SEED_FRACTION: float = 0.10

# PHASE2_ORPHAN_* — relaxed hyperparams for low-row symbol slices left out of clusters.
# Disabled True→False: consistently fails with viability collapse; Symbol '7'
# orphan run hit 3 viability collapses in 10 gens, wasting ~180s.
PHASE2_ORPHAN_ENABLED = False
PHASE2_ORPHAN_GENERATIONS = 18
PHASE2_ORPHAN_POPULATION_SIZE = 150
PHASE2_ORPHAN_MIN_TRADE_SUPPORT = 20
PHASE2_ORPHAN_MIN_TRADE_POOL_FLOOR = 8
PHASE2_ORPHAN_SORTINO_MIN_TRADE_THRESHOLD = 8
PHASE2_ORPHAN_MIN_VAL_TRADES = 6
PHASE2_ORPHAN_MIN_VAL_RETURN_PCT = 0.0
PHASE2_ORPHAN_MONTHLY_MIN_PROFITABLE_RATIO = 0.5
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
# Phase 3 — Rule set selection (greedy + NSGA-II)
# =============================================================================

# --- Team shape ---

# PHASE3_PER_SYMBOL_MAX_RULES — maximum rules selected per symbol.
# 2026-07-11b: 2→4 — many-moderate-rules package.
PHASE3_PER_SYMBOL_MAX_RULES = 4

# PHASE3_GLOBAL_MIN_RULES / MAX_RULES — total rules in the output JSON.
#   Higher MIN → require at least this many rules across all symbols.
#   MAX caps the final strategy size written by Phase 3 / validated on load.
PHASE3_GLOBAL_MIN_RULES = 1
# 2026-07-11b: 20→25 — allow larger composed teams (aligned with RB_MAX_RULES).
PHASE3_GLOBAL_MAX_RULES = 25

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
#   combinations of every single-symbol variant that passes trade floors.
#   When False, only single-symbol variants are produced (legacy behaviour).
#   Higher → richer symbol combinations, broader search, slower evaluation.
#   Lower  → only single-symbol specialisation (original Phase 3 behaviour).
SYMBOL_SPECIALIZATION_USE_COMBINATIONS = True

# SYMBOL_SPECIALIZATION_MAX_SYMBOLS_PER_RULE — maximum number of symbols in a
#   single rule's ``symbol is X`` conditions (1 = single only, 2 = 1+2 combos,
#   3 = 1+2+3 combos).
#   Must be >= 2 when USE_COMBINATIONS=True or combinations are a no-op.
#   Legacy Phase 3 only (inactive while RB_GOVERNOR_ENABLED=True).
SYMBOL_SPECIALIZATION_MAX_SYMBOLS_PER_RULE = 3

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
# PHASE4_INCLUDE_TAIL_HOLDOUT — reserve a final tail window on validation data.
PHASE4_INCLUDE_TAIL_HOLDOUT = True

# PHASE4_TAIL_HOLDOUT_FRACTION — fraction of val reserved as final holdout window.
#   Used by RB Governor risk grid when RB_RISK_GRID_USE_TAIL_HOLDOUT=True
#   (the final tie-break holdout reported but not searched).
#   Higher → more recent data held out; fewer trades in WF folds.
#   Lower  → more data in WF folds; less independent tail check.
PHASE4_TAIL_HOLDOUT_FRACTION = 0.25

# --- Feasibility filters (trial rejected if any fail) ---
# Legacy Phase 4 WF gates; inactive when RB_GOVERNOR_ENABLED=True.
# Retained for the legacy Optuna/grid path when RB Governor is off.

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
# Monthly windows validation (Phase 3 / 4 scoring)
# =============================================================================

# MONTHLY_VALIDATION_ENABLED — toggle monthly rolling-window validation.
#   True  → rule sets are penalised if they fail monthly windows gates.
#   False → monthly penalty is skipped (legacy behaviour).
MONTHLY_VALIDATION_ENABLED = True

MONTHLY_WINDOW_DAYS = 30
# MONTHLY_WINDOW_MIN_ROWS — minimum rows per rolling window before it is skipped.
MONTHLY_WINDOW_MIN_ROWS = 2500
# MONTHLY_WINDOW_MAX_WINDOWS — cap on windows evaluated per backtest slice.
MONTHLY_WINDOW_MAX_WINDOWS = 24
# MONTHLY_RECENCY_WEIGHT — up-weight recent windows in summarize_monthly_metrics
# recency_weighted_return (not used directly inside monthly_penalty()).
MONTHLY_RECENCY_WEIGHT = 2.2
# MONTHLY_MIN_TRADES — minimum executed trades per window (scaled in purged WF).
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
# MONTHLY_WORST_* / MONTHLY_*_WEIGHT — penalty terms in monthly_penalty().
MONTHLY_WORST_RETURN_FLOOR = -1.5
MONTHLY_WORST_PF_FLOOR = 1.0
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
#   Portfolio-level grid trials skip combos above this cap. Per-rule symbol
#   trials score one rule in isolation, so this cap is enforced only when
#   writing the final ruleset (see PHASE4_HARD_CAP_NORMALIZE).
PHASE4_GRID_MAX_TOTAL_CAPITAL = 95.0

# PHASE4_GRID_PASSES — number of round-robin passes through all rules.
PHASE4_GRID_PASSES = 2

# PHASE4_GRID_MIN_IMPROVEMENT — minimum score improvement to accept a new combo.
PHASE4_GRID_MIN_IMPROVEMENT = 0.005

# PHASE4_GRID_ENABLED — when False, Phase 4 grid search is skipped (RB Governor replaces Ph 3+4).
PHASE4_GRID_ENABLED: bool = False

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
PHASE5_REMOVE_NEGATIVE_PNL_RULES = False


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

# RB_ALLOW_FALLBACK — when no positive-good single rules exist, fall back to
#   the best raw-score candidates (legacy behaviour).
#   False → fail closed: write an empty strategy with deployment_accepted=false
#           and reason "no_positive_good_candidates".  Skip compose/risk/amp.
#   True  → preserve legacy raw-score fallback path (rebuild candidates from
#           all specialized variants ranked by raw score).
RB_ALLOW_FALLBACK: bool = False


# --- Rule scoring / gating ---

# RB_MIN_TRAIN_RETURN / RB_MIN_VALID_RETURN — return-% floors below which a
#   single rule fails ``_is_positive_good`` (dual positivity on train+val).
# 2026-07-17: 0.5→0.25 — match PHASE2_* return floors so island rules that
#   clear Phase 2 admission are not wiped on RB val_selection resim
#   (Colab: kept 1/15 long, 4/23 short → tiny teams → concentration fail-closed).
#   Correlated with RB_MAX_PAIR_OVERLAP / score-improvement easing below:
#   more survivors only help if compose can grow multi-island teams.
#   Keep dual-positivity; do NOT enable RB_ALLOW_FALLBACK.
RB_MIN_TRAIN_RETURN: float = 0.25
RB_MIN_VALID_RETURN: float = 0.25

# RB_MIN_TRAIN_PF / RB_MIN_VALID_PF — minimum profit factor for each split.
#   1.0 = break-even before fees.
RB_MIN_TRAIN_PF: float = 1.02
RB_MIN_VALID_PF: float = 1.02

# RB_MIN_TRAIN_TRADES / RB_MIN_VALID_TRADES — per-rule trade-count floors
#   used by ``gate_positive_good`` and ``_score_metrics`` for single rules.
#   Lowered vs. the friend's defaults because per-symbol specialized rules
#   typically see thin validation slices (~7k rows / symbol).
RB_MIN_TRAIN_TRADES: int = 10
RB_MIN_VALID_TRADES: int = 6

# RB_RULESET_MIN_* — trade-count floors applied to the composed team (all
#   rules together).  Should be larger than the per-rule floors because the
#   combined team fires more frequently than any single rule.
RB_RULESET_MIN_TRAIN_TRADES: int = 12
RB_RULESET_MIN_VALID_TRADES: int = 8


# --- Pool & candidate limits ---

# RB_MAX_POOL_RULES_TO_EVALUATE — cap on Phase 2 pool rules passed through
#   symbol-specialization in ``_filter_good_rules``.
#   Higher → more candidates, slower filtering.
RB_MAX_POOL_RULES_TO_EVALUATE: int = 200

# RB_KEEP_TOP_RULES — how many positive-good candidates survive the
#   single-rule ranking to feed ``_compose_ruleset``.
#   Should be comfortably larger than PHASE2_KEEP_TOP_RULES.
RB_KEEP_TOP_RULES: int = 150


# --- Team composition ---

# RB_MAX_RULES — maximum rules in the composed team (hard cap; keep aligned
#   with PHASE3_GLOBAL_MAX_RULES).  To reach RB_MAX_RULES with the default
#   RB_CAPITAL_GRID min (15%), lower grid min or raise RB_MAX_TOTAL_CAPITAL.
# 2026-07-11b: 10→20 — many-moderate-rules package.
RB_MAX_RULES: int = 20

# RB_MAX_PAIR_OVERLAP — max Hamming-style overlap between any two rules in
#   the team. Lower = more diverse team, harder to grow.
# 2026-07-17: 0.25→0.35 — with traded-symbol coverage (not island OR filters),
#   compose must pull rules from other islands; mild condition overlap is OK
#   if traded symbols differ. Correlated with MIN_DISTINCT + coverage fix.
RB_MAX_PAIR_OVERLAP: float = 0.35

# RB_RULESET_MUST_BEAT_SUBSETS — a candidate team must beat both its parent
#   subset and the standalone candidate on both train and val return.
RB_RULESET_MUST_BEAT_SUBSETS: bool = False

# RB_MIN_SCORE_IMPROVEMENT — minimum delta in the governor score to add a
#   new rule in ``_compose_ruleset``.
# 2026-07-17: 0.03→0.01 — ease team growth once more singles survive; still
#   require positive score delta (not zero).
RB_MIN_SCORE_IMPROVEMENT: float = 0.01

# RB_MIN_TRAIN_RETURN_IMPROVEMENT / RB_MIN_VALID_RETURN_IMPROVEMENT — min
#   return-% uplift required from adding a candidate rule.
# 2026-07-17: 0.005→0.002 — correlated with score-improvement ease so adding
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


# --- Lenient-add mode (friend's recommended path) ---
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


# --- Risk grid search (replaces Phase 4 Walk-Forward optimizer) ---

# RB_TP_GRID / RB_SL_GRID / RB_CAPITAL_GRID — values enumerated per rule in
#   the round-robin grid search.  Coarser than the friend's full grid to
#   keep runtime reasonable on a ~10-symbol universe.
# 2026-07-11b: grid min 15→7.5 so RB_MAX_RULES=20 fits under TOTAL=150
# (20 × 7.5 = 150). Without this, large teams starve for capital.
RB_TP_GRID: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)
RB_SL_GRID: tuple[float, ...] = (1.0, 1.2, 1.5, 2.0, 2.5)
RB_CAPITAL_GRID: tuple[float, ...] = (7.5, 10.0, 12.0, 15.0, 18.0)

# RB_RISK_OPT_PASSES — round-robin passes through all rules.
RB_RISK_OPT_PASSES: int = 1

# RB_RISK_MIN_IMPROVEMENT — min score delta to accept a new TP/SL/cap combo.
RB_RISK_MIN_IMPROVEMENT: float = 0.02

# RB_MAX_TOTAL_CAPITAL — hard cap on sum(capital_pct) across all rules.
# 2026-07-11b: 100→150 — with capital grid min 7.5%, funds up to RB_MAX_RULES=20.
RB_MAX_TOTAL_CAPITAL: float = 150.0

# RB_RISK_GRID_WF_SPLITS — walk-forward folds for risk grid (1 = legacy single-fold).
#   Score every TP/SL/capital combo on N chronological folds of val_selection,
#   pick the combo with the best min(fold1, fold2, ...) score (worst-case selection).
#   → fixes audit finding #3 (RB Governor risk-grid overfits val_selection)
RB_RISK_GRID_WF_SPLITS: int = 3

# RB_RISK_GRID_USE_TAIL_HOLDOUT — reserve final PHASE4_TAIL_HOLDOUT_FRACTION
#   of val_selection as an untouched tie-break holdout (reported but NOT used
#   during search).  Set False to use the full val_selection for folds.
#   → fixes audit finding #12 (PHASE4_TAIL_HOLDOUT_FRACTION orphan)
RB_RISK_GRID_USE_TAIL_HOLDOUT: bool = True

# RB_TAIL_HOLDOUT_HARD_GATE — when True, strategies whose tail-holdout return
#   is below RB_TAIL_HOLDOUT_MIN_RETURN_PCT fail closed: empty ruleset written
#   with deployment_accepted=False (not a soft flag with rules retained).
RB_TAIL_HOLDOUT_HARD_GATE: bool = True
RB_TAIL_HOLDOUT_MIN_RETURN_PCT: float = 0.0

# RB_MAX_SYMBOL_SHARE_ABS_PNL — max fraction of abs PnL from a single symbol on
#   the RB validation frame; above this → fail-closed empty strategy
#   (deployment_accepted=False, rules_set cleared).
# 2026-07-17: 0.50→0.55 — Colab fail-closed at top_share=0.556 / 0.615 on
#   1–2 rule teams. Slight ease only; real fix is multi-island compose
#   (traded coverage + looser overlap/return floors). Do not raise alone.
RB_MAX_SYMBOL_SHARE_ABS_PNL: float = 0.55
# RB_MAX_SYMBOL_HHI — max Herfindahl index of abs PnL across symbols on valid;
#   above this → same fail-closed empty strategy as share gate.
# 2026-07-17: 0.55→0.60 — correlated with share ease (Colab HHI~0.46–0.46
#   passed HHI but failed share; headroom for slightly larger teams).
RB_MAX_SYMBOL_HHI: float = 0.60


# --- Symbol strategy: two modes (do not mix) ---------------------------------
#
# Mode A — multi-symbol TEAM (recommended / current):
#   RB_REQUIRE_SYMBOL_FILTERS=False
#   Phase 2 islands learn fuzzy rules on 3–4 symbol clusters.
#   RB attaches each rule's ``source_symbols`` as OR filters (cluster scope),
#   then composes rules from different islands into a team covering the book.
#   Compose diversity uses **traded** symbols from metrics (not OR filters):
#   island filters alone must not satisfy RB_MIN_DISTINCT_SYMBOLS.
#   Bare "fire on all 10 symbols" generalists destroy island rules on
#   full train + val_selection (run.log: kept 0/15, 0/12 → empty strategies).
#
# Mode B — per-symbol SPECIALISTS (friend_project path; avoid unless intentional):
#   RB_REQUIRE_SYMBOL_FILTERS=True
#   Each rule locks to ``symbol is X``; compose many specialists for coverage.
#   Tends to jagged single-symbol equity; needs thick per-symbol pools.
#
# Mixing A+B (filters off but compose counting only ``symbol is X``) blocks
# team growth — that was a real bug with MIN_DISTINCT>0.

# RB_REQUIRE_SYMBOL_FILTERS — Mode B when True; Mode A when False.
RB_REQUIRE_SYMBOL_FILTERS: bool = False

# RB_MIN_DISTINCT_SYMBOLS — target symbol coverage while composing.
#   Mode A: traded symbols from metrics must expand toward this.
#   Mode B: distinct ``symbol is X`` filters on final rules (hard gate).
RB_MIN_DISTINCT_SYMBOLS: int = 3

# Soft score bonus per extra traded symbol beyond the first (Mode A ranking).
# 2026-07-17: 8→15 — stronger preference for multi-symbol traded coverage
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
RB_PROFIT_AMP_MAX_RULES: int = 8
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
RB_PROFIT_AMP_CAPITAL_GRID: tuple[float, ...] = RB_CAPITAL_GRID
RB_PROFIT_AMP_KEEP_BASELINE_UNLESS_BETTER: bool = True


# --- Cross-run global bank (disabled by default to keep runs isolated) ---

RB_GLOBAL_BANK_ENABLED: bool = False
RB_GLOBAL_BANK_DIRNAME: str = "rb_bank"
RB_GLOBAL_BANK_MAX_RULES_PER_DIRECTION: int = 700
RB_GLOBAL_BANK_IMPORT_TOP_SINGLE_RULES: int = 80
RB_GLOBAL_MAX_RULES: int = 12
RB_GLOBAL_MIN_COMBINED_RETURN_IMPROVEMENT: float = 0.05
RB_GLOBAL_REQUIRE_POSITIVE_TRAIN_VALID: bool = True
RB_GLOBAL_RISK_OPT_PASSES: int = 1
RB_GLOBAL_BEST_DIRNAME: str = "best_global"
RB_GLOBAL_TP_GRID: tuple[float, ...] = (1.5, 2.0, 3.0, 5.0, 8.0)
RB_GLOBAL_SL_GRID: tuple[float, ...] = (1.2, 1.5, 2.0, 2.5)
RB_GLOBAL_CAPITAL_GRID: tuple[float, ...] = (5.0, 12.5, 25.0, 50.0)
RB_GLOBAL_MAX_TOTAL_CAPITAL: float = 100.0


# =============================================================================
# Helpers & resolvers
# =============================================================================
# Pure functions and dataclasses. Constants live in the sections above; these
# resolve paths, scale trade floors, and adapt gates to debug / island scope.


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


def effective_phase4_wf_splits() -> int:
    """Inner validation WF windows; single window when purged CV already ran."""
    if split_mode_is_purged_walk_forward():
        return 1
    return int(PHASE4_WF_SPLITS)



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
    """Legacy Phase 4 WF gate floor; unwired when RB_GOVERNOR_ENABLED=True."""
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
        if sym_n >= 3:
            scaled = max(2, (sym_n + 1) // 2)  # 3-sym → 2, 4-sym → 2, 5-sym → 3
        else:
            scaled = max(1, (sym_n + 1) // 2)
        min_profitable = min(
            int(PHASE2_MIN_PROFITABLE_SYMBOLS),
            scaled,
        )
        monthly_months = int(PHASE2_ISLAND_MONTHLY_MIN_MONTHS)
        monthly_ratio = float(PHASE2_MONTHLY_ADMISSION_MIN_RATIO)

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
        min_profitable_symbols=int(min_profitable),
        monthly_admission_min_months=int(monthly_months),
        monthly_admission_min_profitable_ratio=float(monthly_ratio),
        skip_symbol_robustness_penalty=(profile == "orphan"),
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

assert PHASE2_STAGE_A_GENERATIONS + PHASE2_STAGE_B_GENERATIONS == PHASE2_GENERATIONS, (
    "Stage A+B budgets must equal PHASE2_GENERATIONS for two-stage handoff"
)
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
assert int(RB_MAX_RULES) <= int(PHASE3_GLOBAL_MAX_RULES), (
    "RB_MAX_RULES must not exceed Phase 3 output cap"
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
assert float(PURGED_WF_HOLDOUT_FRACTION) + float(PURGED_WF_MIN_TRAIN_FRACTION) <= 1.0, (
    "holdout + min train prefix must fit within each symbol timeline"
)
assert int(TAIL_DROP_ROWS) == int(MAX_HOLD_CANDLES), (
    "TAIL_DROP_ROWS must equal MAX_HOLD_CANDLES (label horizon)"
)
assert int(HOLDOUT_EMBARGO_CANDLES) == int(MAX_HOLD_CANDLES), (
    "HOLDOUT_EMBARGO_CANDLES must equal MAX_HOLD_CANDLES (label horizon)"
)
# =============================================================================
assert MIN_CONDITIONS <= MAX_CONDITIONS, (
    f"MIN_CONDITIONS ({MIN_CONDITIONS}) must be <= MAX_CONDITIONS ({MAX_CONDITIONS})"
)
assert 0.0 <= PHASE2_VIABILITY_RECOVERY_DEPLOYABLE_MUTATE_FRACTION <= 1.0

# =============================================================================
# Runtime — Colab GPU defaults
# =============================================================================


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

