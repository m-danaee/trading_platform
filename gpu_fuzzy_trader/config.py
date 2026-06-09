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
  Short OOS / overfitting          SPLIT_MODE, CV_N_FOLDS, PHASE3_* gates,
                                   PHASE2_JOINT_TRAIN_VAL, PHASE2_CV_* gates
  GPU OOM                          PHASE1_SAMPLING_TOTAL ↓, PHASE2_GPU_BATCH_SIZE ↓,
                                   PHASE2_SCAN_UNROLL ↓, PHASE2_CV_FOLD_WORKERS = 1
  Phase 2 too slow                 PHASE2_GENERATIONS ↓, CV_N_FOLDS ↓, PHASE2_USE_GPU
  Empty Phase 2 pool               MIN_TRADE_SUPPORT ↓, PHASE2_*_FLOOR ↓,
                                   PHASE2_CV_POOL_MIN_FOLDS_PASS ↓
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

# Symbol-specialist Phase 2: one island per direction × symbol.
PHASE2_SYMBOL_SPECIALIST_ENABLED = True
PHASE2_SYMBOL_UNIVERSE_MODE = "train_present"
PHASE2_ISLAND_EPOCH_GENERATIONS = 27
PHASE2_MIGRATION_EPOCH_INTERVAL = 3
PHASE2_MIGRATION_SEED_FRACTION = 0.15
PHASE2_SHARED_ARCHIVE_MIN_SYMBOLS = 3
PHASE2_SHARED_ARCHIVE_MIN_ROBUST_SCORE = 1.0
PHASE3_SYMBOL_RULE_MIN_TRAIN_TRADES = 17
PHASE3_SYMBOL_RULE_MIN_VAL_TRADES = 5

# Debug: scope pipeline to N symbols starting at DEBUG_SYMBOL (sorted universe).
DEBUG_SYMBOL_SCOPE_ENABLED = True
DEBUG_SYMBOL = "1"
DEBUG_SYMBOL_COUNT = 2


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


def filter_df_to_symbol(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Return rows for a single symbol; raises if column missing or no rows."""
    return filter_df_to_symbols(df, [symbol])


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


def phase2_symbol_pool_path(
    direction: str,
    symbol: str,
    outputs_dir: str | None = None,
) -> str:
    """Per-symbol Phase 2 pool under ``outputs/phase2/{direction}/{symbol}/``."""
    root = outputs_dir or OUTPUTS_DIR
    return os.path.join(root, "phase2", direction, symbol, "pool.json")


def phase2_symbol_history_path(
    direction: str,
    symbol: str,
    outputs_dir: str | None = None,
) -> str:
    """Per-symbol Phase 2 history JSON path."""
    root = outputs_dir or OUTPUTS_DIR
    return os.path.join(root, "phase2", direction, symbol, "history.json")


def phase2_symbol_archive_path(direction: str, symbol: str) -> str:
    """Persistent local symbol archive (cross-run warm start)."""
    return os.path.join(PHASE2_ARCHIVE_DIR, direction, symbol, "archive.json")


def phase2_shared_archive_path(direction: str) -> str:
    """Direction-level shared archive of broadly robust rules."""
    return os.path.join(PHASE2_ARCHIVE_DIR, direction, "shared_archive.json")


def enumerate_phase2_symbols(train_df) -> list[str]:
    """Return sorted symbol universe for symbol-specialist Phase 2."""
    if PHASE2_SYMBOL_UNIVERSE_MODE != "train_present":
        raise ValueError(
            f"Unsupported PHASE2_SYMBOL_UNIVERSE_MODE: {PHASE2_SYMBOL_UNIVERSE_MODE!r}"
        )
    if train_df is None or getattr(train_df, "empty", True):
        return []
    if "symbol" not in train_df.columns:
        return []
    return sorted(train_df["symbol"].dropna().astype(str).unique().tolist())


def phase2_pool_path(
    direction: str,
    symbol: str | None = None,
    outputs_dir: str | None = None,
) -> str:
    """Resolve Phase 2 pool path (symbol-scoped or legacy direction-level)."""
    if PHASE2_SYMBOL_SPECIALIST_ENABLED and symbol is not None:
        return phase2_symbol_pool_path(direction, symbol, outputs_dir)
    if symbol is None:
        if outputs_dir is None:
            return PHASE2_POOL_PATHS[direction]
        return os.path.join(outputs_dir, f"phase2_{direction}_pool.json")
    return phase2_symbol_pool_path(direction, symbol, outputs_dir)


def phase2_history_path(
    direction: str,
    symbol: str | None = None,
    outputs_dir: str | None = None,
) -> str:
    """Resolve Phase 2 history path (symbol-scoped or legacy direction-level)."""
    if PHASE2_SYMBOL_SPECIALIST_ENABLED and symbol is not None:
        return phase2_symbol_history_path(direction, symbol, outputs_dir)
    if symbol is None:
        if outputs_dir is None:
            return PHASE2_HISTORY_PATHS[direction]
        return os.path.join(outputs_dir, f"phase2_{direction}_history.json")
    return phase2_symbol_history_path(direction, symbol, outputs_dir)


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
# Must equal MAX_HOLD_CANDLES and CV_EMBARGO_BARS (288 = 24 h at 5-min bars).
#   Higher → more rows removed, safer labels, less training data.
#   Lower  → more rows kept, risk of NaN / lookahead leakage at symbol tails.
TAIL_DROP_ROWS = 288


# =============================================================================
# Phase 0 — Train / validation split (Phases 2–3)
# =============================================================================
# Phases 4–5 always use persisted train_70 + validation_30 (see splitter.py).

# SPLIT_MODE
#   "purged_rolling_cv" — K expanding-window folds, embargo, worst-fold scoring.
#                         Stricter generalization; ~K× slower Phase 2/3.
#   "holdout_70_30"     — single 70/30 per symbol; fast but easier to overfit
#                         one validation season (risky for short direction).
SPLIT_MODE = "purged_rolling_cv"

# CV_N_FOLDS — number of purged rolling validation windows per symbol.
#   Higher → stricter season coverage, slower eval, pool gates need more folds.
#   Lower  → faster runs, less robust to regime change (2 is debug-friendly).
# Coupled: PHASE2_CV_POOL_MIN_FOLDS_PASS auto-scales via _CV_POOL_MIN_FOLDS_PASS.
CV_N_FOLDS = 2

# CV_EMBARGO_BARS — gap between train end and val start (label-horizon purge).
#   Higher → less leakage, shorter effective train per fold.
#   Lower  → more train rows, risk of label overlap across split boundary.
CV_EMBARGO_BARS = TAIL_DROP_ROWS

# CV_BARS_PER_DAY — bars per calendar day (288 for 5-minute data).
# Used only for fold sizing / month calculations; wrong value mis-sizes folds.
CV_BARS_PER_DAY = 288

# CV_MIN_TRAIN_MONTHS — minimum train history before first val window.
#   Higher → folds start later, more stable train stats, fewer usable folds.
#   Lower  → more folds on short histories, noisier train estimates.
CV_MIN_TRAIN_MONTHS = 2.0

# True majority for pool admission: even K requires K/2+1; odd K uses ceil(K/2).


def _cv_pool_min_folds_pass(n_folds: int) -> int:
    if n_folds <= 1:
        return 1
    if n_folds % 2 == 0:
        return n_folds // 2 + 1
    return n_folds // 2 + 1


_CV_POOL_MIN_FOLDS_PASS = _cv_pool_min_folds_pass(CV_N_FOLDS)
_CV_RANK_MIN_FOLDS_PASS = max(1, _CV_POOL_MIN_FOLDS_PASS - 1)


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
PHASE1_TOP_K_FEATURES = 15

# PHASE1_MAX_FEATURE_OVERLAP — max Jaccard overlap between long & short lists.
#   Higher → more shared features across directions; smaller combined gene space.
#   Lower  → more direction-specific lists; better asymmetry, less redundancy.
PHASE1_MAX_FEATURE_OVERLAP = 0.65

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
PHASE1_STATIONARITY_FOLDS = 3

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
PHASE1_STATIONARITY_STRATIFY = "regime"

# --- Regime detection (when STRATIFY == "regime") ---

# PHASE1_REGIME_FAST_WINDOW / SLOW_WINDOW — rolling regression window lengths.
#   Higher → smoother regime labels, slower regime switches, fewer regimes.
#   Lower  → more reactive labels, noisier regime boundaries.
PHASE1_REGIME_FAST_WINDOW = 10
PHASE1_REGIME_SLOW_WINDOW = 24

# PHASE1_REGIME_FAST_R2_THRESHOLD / SLOW_R2_THRESHOLD — min R² for trend regime.
#   Higher → only strong trends labeled as trending; more "chop" regimes.
#   Lower  → weak trends count as trending; fewer chop labels.
PHASE1_REGIME_FAST_R2_THRESHOLD = 0.20
PHASE1_REGIME_SLOW_R2_THRESHOLD = 0.25

# PHASE1_REGIME_FAST_SLOPE_THRESHOLD / SLOW_SLOPE_THRESHOLD — min |slope|.
#   Higher → steeper move required to call a trend; stricter trend detection.
#   Lower  → flat markets may still be labeled trending.
PHASE1_REGIME_FAST_SLOPE_THRESHOLD = 0.0016
PHASE1_REGIME_SLOW_SLOPE_THRESHOLD = 0.0010

# PHASE1_REGIME_MED_WINDOW — median filter on regime labels (smooth flicker).
#   Higher → stabler regime IDs; delayed regime transitions.
#   Lower  → faster regime switches; noisier per-bar labels.
PHASE1_REGIME_MED_WINDOW = 9

# PHASE1_REGIME_MIN_DAYS — minimum calendar days to fit regime model.
#   Higher → regime model trained on longer history; may miss recent structure.
#   Lower  → fits on less data; unstable clusters on short histories.
PHASE1_REGIME_MIN_DAYS = 14

# PHASE1_REGIME_MIN_SAMPLES — minimum rows per stationarity fold.
#   Higher → skip thin folds; stricter MI estimates.
#   Lower  → allow sparse folds; noisier stationarity scores.
PHASE1_REGIME_MIN_SAMPLES = 100

PHASE1_REGIME_MODEL_PATH = os.path.join(
    OUTPUTS_DIR, "phase1_regime_cluster.joblib")


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
PHASE2_GPU_BATCH_SIZE_AUTO = False

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
MAX_CONDITIONS = 4

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
MIN_TRADE_SUPPORT = 60

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

# --- Return / quality floors (evolution + pool filtering) ---

# PHASE2_RETURN_FLOOR_PCT — min train return % to avoid feasibility penalty.
#   Higher → only profitable-on-train rules stay feasible; emptier search.
#   Lower  → more exploration; weak rules linger until other gates remove them.
PHASE2_RETURN_FLOOR_PCT = 1.0

# PHASE2_VAL_RETURN_FLOOR_PCT — min validation return % for feasibility.
#   Higher → stricter OOS alignment during evolution.
#   Lower  → allow negative val return during search (gates may still catch later).
PHASE2_VAL_RETURN_FLOOR_PCT = 0.0

# PHASE2_PROFIT_FACTOR_FLOOR — min profit factor for feasibility.
#   Higher → require gross wins >> losses; fewer rules pass.
#   Lower  → allow marginal PF; more rules in Pareto set.
PHASE2_PROFIT_FACTOR_FLOOR = 1.0

# PHASE2_SYMBOL_MEDIAN_RETURN_FLOOR_PCT — min median return across symbols.
#   Higher → rules must work on typical symbols, not one outlier.
#   Lower  → single-symbol heroes can survive longer.
PHASE2_SYMBOL_MEDIAN_RETURN_FLOOR_PCT = -0.5

# PHASE2_MIN_PROFITABLE_SYMBOLS — min count of symbols with positive PnL.
#   Higher → demand broad cross-symbol edge; stricter for 10-symbol universe.
#   Lower  → allow niche symbol specialists.
PHASE2_MIN_PROFITABLE_SYMBOLS = 5

# PHASE2_MAX_DRAWDOWN_GATE — hard DD % cap; above this all objectives penalized.
#   Lower  → Pareto front pushed toward low-drawdown rules; may cut high return.
#   Higher → allow aggressive rules with large equity swings.
PHASE2_MAX_DRAWDOWN_GATE = 20.0

# PHASE2_POOL_REQUIRE_POSITIVE_SPLITS — require non-negative train & val returns.
#   True  → infeasible penalty on negative-split rules during evolution.
#   False → negative val allowed at fitness stage (CV gates may still filter).
PHASE2_POOL_REQUIRE_POSITIVE_SPLITS = True
PHASE2_POOL_TRAIN_RETURN_MIN_PCT = 0.0
PHASE2_POOL_VAL_RETURN_MIN_PCT = 0.0

# --- Purged CV pool admission (per-fold gates) ---

# PHASE2_CV_POOL_MIN_FOLDS_PASS — folds that must pass per-fold checks.
# Auto-coupled to CV_N_FOLDS (~50%). Do not set to CV_N_FOLDS unless you want
# 100% fold pass (often starves Phase 3).
#   Higher → stricter deployability across seasons; smaller pool.
#   Lower  → rules can enter pool with fewer good seasons.
PHASE2_CV_POOL_MIN_FOLDS_PASS = _CV_POOL_MIN_FOLDS_PASS

# PHASE2_CV_MERGED_GATE_HARD — merged worst-case metrics also hard-reject.
#   True  → double gate (fold pass AND merged metrics); very strict pool.
#   False → fold majority is the hard gate; merged metrics rank only.
PHASE2_CV_MERGED_GATE_HARD = True

# PHASE2_CV_MIN_TRADE_POOL_FLOOR — per-fold hard trade floor (lower than global).
#   Higher → each fold must show more trades; rejects seasonal one-offs.
#   Lower  → thin seasonal rules can pass a fold.
PHASE2_CV_MIN_TRADE_POOL_FLOOR = 7

PHASE2_CV_POOL_TRAIN_RETURN_MIN_PCT = 0.0
PHASE2_CV_POOL_VAL_RETURN_MIN_PCT = 0.0

# PHASE2_CV_PROFIT_FACTOR_FLOOR — per-fold minimum PF for pool admission.
#   Higher → stricter per-season profitability.
#   Lower  → marginal PF allowed in some folds.
PHASE2_CV_PROFIT_FACTOR_FLOOR = 1.0

# PHASE2_CV_MIN_VAL_TRADES — min validation trades per fold for admission.
#   Higher → fold val metrics are statistically meaningful.
#   Lower  → folds with few trades can still admit rules.
PHASE2_CV_MIN_VAL_TRADES = 4

# PHASE2_CV_POOL_TARGET_MIN — soft target pool size before rank fallback kicks in.
#   Higher → pipeline tries harder to fill a large pool via looser rank admit.
#   Lower  → satisfied with smaller pool; less rank fallback pressure.
PHASE2_CV_POOL_TARGET_MIN = 40

# PHASE2_CV_POOL_RANK_ADMIT_TOP_K — max rules admitted via rank fallback.
#   Higher → larger pool when strict gates are too tight.
#   Lower  → smaller pool; Phase 3 has fewer combinations to search.
PHASE2_CV_POOL_RANK_ADMIT_TOP_K = 80

# PHASE2_CV_RANK_MIN_FOLDS_PASS — looser fold pass for rank fallback (≤ pool gate).
PHASE2_CV_RANK_MIN_FOLDS_PASS = _CV_RANK_MIN_FOLDS_PASS

# PHASE2_REQUIRE_LAST_FOLD_POSITIVE — last CV fold val return must be > 0.
#   True  → emphasize most recent season; can shrink pool sharply.
#   False → last fold can be weak if earlier folds pass (recommended with CV=2).
PHASE2_REQUIRE_LAST_FOLD_POSITIVE: bool = False


# =============================================================================
# Phase 2 — Fitness objectives & joint evaluation
# =============================================================================

# SORTINO_CAP — maximum saturated Sortino after tanh compression.
#   Higher → more differentiation among top Sortino rules on f1.
#   Lower  → flatter f1 landscape; diversity across other objectives easier.
SORTINO_CAP = 5.0

# SORTINO_SCALE — divisor inside tanh(raw_sortino / scale); controls saturation.
#   Higher → less compression; extreme Sortino values still differentiate f1.
#   Lower  → aggressive compression; reduces Sortino-driven dominance.
SORTINO_SCALE = 3.0

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
PHASE2_EARLY_STOP_MIN_VALID_RULES = 5

# PHASE2_EARLY_STOP_DISABLED_IN_CV — disable return early-stop in purged CV mode.
#   True  → always run full gens under CV (slower).
#   False → early stop active under CV (default).
PHASE2_EARLY_STOP_DISABLED_IN_CV = False

# --- Plateau early stop (no improvement in best return) ---

PHASE2_PLATEAU_EARLY_STOP_ENABLED = True

# PHASE2_PLATEAU_EARLY_STOP_MIN_GENERATION — earliest gen for plateau stop.
#   Higher → more exploration in Stage A before plateau can end run.
#   Lower  → may stop during initial transient; should be ≤ STAGE_A_GENERATIONS.
PHASE2_PLATEAU_EARLY_STOP_MIN_GENERATION = 28

# PHASE2_PLATEAU_EARLY_STOP_PATIENCE — gens without improvement before stop.
#   Higher → wait longer for breakthrough; uses more compute.
#   Lower  → stop quickly when progress stalls.
PHASE2_PLATEAU_EARLY_STOP_PATIENCE = 12

# PHASE2_PLATEAU_EARLY_STOP_MIN_DELTA_PCT — min return improvement to reset patience.
#   Higher → need larger gains to count as progress.
#   Lower  → tiny improvements reset plateau counter.
PHASE2_PLATEAU_EARLY_STOP_MIN_DELTA_PCT = 0.02

PHASE2_PLATEAU_EARLY_STOP_DISABLED_IN_CV = False

# PHASE2_PLATEAU_USE_ROBUST_RETURN — track min(train,val) return for plateau.
#   True  → plateau reflects deployable return, not train-only spikes.
#   False → train max return can mask val stagnation.
PHASE2_PLATEAU_USE_ROBUST_RETURN = True

PHASE2_PLATEAU_BLOCK_WHEN_DEPLOYABLE_ZERO = True
PHASE2_PLATEAU_BLOCK_WHEN_DIVERSITY_LOW = True

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
PHASE2_DEPLOYABLE_ARCHIVE_MAX_SIZE = 200

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
PHASE2_STAGE_A_GENERATIONS = 80

# PHASE2_STAGE_B_GENERATIONS — Stage B (refinement) generation budget.
#   Higher → more val-robust polishing; total time = A + B gens.
#   Lower  → less refinement after exploration.
PHASE2_STAGE_B_GENERATIONS = 40

# PHASE2_STAGE_B_SEED_TOP_K — elites from Stage A seeded into Stage B.
#   Higher → broader refinement starting set; slower Stage B per gen.
#   Lower  → refine only top performers; risk missing dark horses.
PHASE2_STAGE_B_SEED_TOP_K = 100

# PHASE2_STAGE_B_SEED_FRACTION — fraction of Stage B pop seeded from Stage A elites.
#   Higher → more refinement around known good regions; risk of clone collapse.
#   Lower  → more random exploration in Stage B.
PHASE2_STAGE_B_SEED_FRACTION = 0.50

# --- Stage A hyperparameters (exploration: higher mutation, stronger diversity) ---

# PHASE2_STAGE_A_MUTATION_RATE — per-gene mutation in Stage A.
#   Higher → more genetic exploration before Stage B refinement.
PHASE2_STAGE_A_MUTATION_RATE = 0.18

# PHASE2_STAGE_A_MUTATION_WEIGHTED_ACTIVATE_PROB — bias toward activating genes in A.
PHASE2_STAGE_A_MUTATION_WEIGHTED_ACTIVATE_PROB = 0.75

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
PHASE2_STAGE_A_PLATEAU_EARLY_STOP_PATIENCE = 14

# PHASE2_STAGE_A_PLATEAU_EARLY_STOP_MIN_GENERATION — earliest plateau stop gen in A.
PHASE2_STAGE_A_PLATEAU_EARLY_STOP_MIN_GENERATION = 20

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
PHASE2_STAGE_A_USE_ROBUST_RETURN_OBJ = False

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
PHASE2_STAGE_B_MUTATION_RATE = 0.10

# PHASE2_STAGE_B_MUTATION_WEIGHTED_ACTIVATE_PROB — conservative gene activation in B.
PHASE2_STAGE_B_MUTATION_WEIGHTED_ACTIVATE_PROB = 0.60

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
PHASE2_STAGE_B_PLATEAU_EARLY_STOP_PATIENCE = 8

# PHASE2_STAGE_B_PLATEAU_EARLY_STOP_MIN_GENERATION — earliest plateau stop gen in B.
PHASE2_STAGE_B_PLATEAU_EARLY_STOP_MIN_GENERATION = 1

# PHASE2_STAGE_B_EARLY_STOP_MIN_GENERATION — earliest return-based early stop in B.
PHASE2_STAGE_B_EARLY_STOP_MIN_GENERATION = 20

# PHASE2_GPU_ENRICH_SYMBOL_METRICS — merge CPU per-symbol metrics after GPU batch eval.
# Skipped automatically when symbol_scope is set (symbol-island mode).
PHASE2_GPU_ENRICH_SYMBOL_METRICS = True


def phase2_should_enrich_symbol_metrics(engine: object | None = None) -> bool:
    """Return True when GPU batch eval should run a follow-up CPU enrichment pass."""
    if not PHASE2_GPU_ENRICH_SYMBOL_METRICS:
        return False
    if engine is not None and getattr(engine, "_symbol_scope", None) is not None:
        return False
    return True


# =============================================================================
# Phase 2 — Parallel CV fold evaluation
# =============================================================================

# PHASE2_CV_FOLD_WORKERS — threads evaluating CV folds simultaneously.
#   0 → auto (= CV_N_FOLDS).  1 → sequential (safest on single GPU).
#   Higher → faster on multi-GPU hosts; on one GPU can increase peak VRAM.
PHASE2_CV_FOLD_WORKERS = 1


# =============================================================================
# Phase 2 — NSGA-III search budget & archive
# =============================================================================

# PHASE2_POPULATION_SIZE — individuals per generation.
#   Higher → better Pareto coverage, ~linear GPU cost per generation.
#   Lower  → faster gens, risk of premature convergence.
PHASE2_POPULATION_SIZE = 200

# PHASE2_GENERATIONS — total evolutionary generations (before early stop).
#   Higher → more search budget; diminishing returns after plateau.
#   Lower  → faster runs; may under-explore gene space.
PHASE2_GENERATIONS = 80

PHASE2_ALGORITHM = "NSGA3"

# PHASE2_ARCHIVE_MAX_SIZE — max stored non-dominated solutions across gens.
#   Higher → richer elite memory; more memory, slower non-dominated sorting.
#   Lower  → leaner archive; may lose good rules found early.
PHASE2_ARCHIVE_MAX_SIZE = 400

# PHASE2_ARCHIVE_SEED_FRACTION — fraction of initial pop from cross-run archive.
#   Higher → more warm-start from past runs; less fresh random exploration.
#   Lower  → more random init; slower reuse of known good rules.
PHASE2_ARCHIVE_SEED_FRACTION = 0.25

PHASE2_SEED: int = get_seed()


# =============================================================================
# Phase 2 — Regime-stratified support & profitability
# =============================================================================

PHASE2_REGIME_SUPPORT_ENABLED = True
PHASE2_REGIME_MODEL_PATH = PHASE1_REGIME_MODEL_PATH

# PHASE2_REGIME_CONCENTRATION_MIN — fraction of trades in dominant regime.
#   Higher → only sharp regime specialists bypass global support penalty.
#   Lower  → diffuse rules can claim specialist status more easily.
PHASE2_REGIME_CONCENTRATION_MIN = 0.70

# PHASE2_REGIME_MIN_WIN_RATE — min win rate in dominant regime for specialist.
#   Higher → specialists must show stronger edge in their niche.
#   Lower  → marginal win rate allowed for regime bypass.
PHASE2_REGIME_MIN_WIN_RATE = 0.40

PHASE2_REGIME_USE_PNL_GATE = True

# PHASE2_REGIME_MIN_TRADE_FRACTION — scales per-regime trade thresholds.
#   Higher → more trades required per regime slice.
#   Lower  → easier regime specialist qualification.
PHASE2_REGIME_MIN_TRADE_FRACTION = 1.0

# PHASE2_REGIME_REQUIRE_VAL_CONFIRMATION — specialist must pass on val too.
#   True  → blocks train-only regime overfit (important for short).
#   False → train regime stats alone can qualify specialist.
PHASE2_REGIME_REQUIRE_VAL_CONFIRMATION = True

# PHASE2_REGIME_PROFITABILITY_GATE — require profit > 0 in enough regimes.
#   True  → rules must work in multiple regimes, not one lucky slice.
#   False → single-regime profitability sufficient.
PHASE2_REGIME_PROFITABILITY_GATE: bool = True

# PHASE2_REGIME_MIN_RETURN_PER_REGIME — min return % per regime to count as profitable.
#   Higher → stricter multi-regime edge; fewer rules pass gate.
#   Lower  → tiny positive return per regime counts as OK.
PHASE2_REGIME_MIN_RETURN_PER_REGIME: float = 0.25


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
PHASE2_INIT_SOFTMAX_TEMP = 0.5

PHASE2_INIT_SCORE_EPS = 1e-6

# PHASE2_INIT_UNIFORM_MIX — probability of uniform random gene vs structured init.
#   Higher → more random chromosomes in initial population.
#   Lower  → more MI-guided structured rules at gen 0.
PHASE2_INIT_UNIFORM_MIX = 0.05

# PHASE2_MUTATION_RATE — per-gene mutation probability.
#   Higher → more exploration, noisier convergence, better escape local optima.
#   Lower  → finer local search, risk of premature convergence.
PHASE2_MUTATION_RATE = 0.14

# PHASE2_MUTATION_WEIGHTED_ACTIVATE_PROB — bias mutations toward activating genes.
#   Higher → mutations tend to add conditions rather than dont_care.
#   Lower  → mutations more often deactivate or flip existing conditions.
PHASE2_MUTATION_WEIGHTED_ACTIVATE_PROB = 0.70


# =============================================================================
# Phase 3 — Rule set selection (greedy + NSGA-II)
# =============================================================================

# --- Team shape ---

# PHASE3_MIN_RULES / MAX_RULES — team size for combined rule set.
#   Higher MIN → need more rules before team is valid; stricter diversification.
#   Higher MAX → larger teams, more capital overlap, richer interaction effects.
#   Lower MIN → accept single-rule teams; less diversification.
PHASE3_MIN_RULES = 2
PHASE3_MAX_RULES = 3

# PHASE3_MIN_SYMBOL_COVERAGE — symbols with ≥1 val trade required (of 10).
#   Higher → team must fire across more symbols; rejects niche teams.
#   Lower  → symbol-concentrated teams can pass.
PHASE3_MIN_SYMBOL_COVERAGE = 7

# PHASE3_MAX_CAPITAL_PCT_PER_RULE — cap per rule before normalization.
#   Higher → each rule can use more notional; higher overlap drawdown risk.
#   Lower  → thinner per-rule sizing; may under-use signals.
PHASE3_MAX_CAPITAL_PCT_PER_RULE = 50.0

# --- Engines ---

PHASE3_USE_GPU = False  # overridden to True on Colab GPU via _apply_colab_gpu_defaults()
PHASE3_USE_PARALLEL_BATCH = True

# PHASE3_BATCH_WORKERS — parallel workers for team evaluation.
#   Higher → faster Phase 3 on many-core CPU; diminishing returns past ~32.
#   Lower  → less CPU contention.
PHASE3_BATCH_WORKERS = min(32, os.cpu_count() or 4)

PHASE3_NUMBA_ENABLED = True

# --- Refinement budget (NSGA-II on team composition) ---

# PHASE3_REFINE_GENERATIONS — NSGA-II generations for team refinement.
#   Higher → more team combinatorial search; slower Phase 3.
#   Lower  → greedy result may not be polished by evolution.
PHASE3_REFINE_GENERATIONS = 250

# PHASE3_REFINE_POP_SIZE — population size during team refinement.
#   Higher → explore more team compositions per generation.
#   Lower  → faster refinement, may miss optimal team mixes.
PHASE3_REFINE_POP_SIZE = 300

# PHASE3_SMALL_POOL_THRESHOLD — pool size below which reduced budget applies.
PHASE3_SMALL_POOL_POP = 100
PHASE3_SMALL_POOL_GEN = 60
PHASE3_SMALL_POOL_THRESHOLD = 20

# PHASE3_MIN_PARETO_FRONT — minimum front size before accepting refinement result.
#   Higher → insist on diverse team options; may fail on tiny pools.
#   Lower  → accept thin fronts.
PHASE3_MIN_PARETO_FRONT = 3

# PHASE3_REFINE_EARLY_STOP_PARETO_ONE_GENS — stop if Pareto size == 1 this many gens.
#   Higher → wait longer before declaring unique optimum.
#   Lower  → stop quickly when single team dominates.
PHASE3_REFINE_EARLY_STOP_PARETO_ONE_GENS = 15

PHASE3_GREEDY_STOP_ON_WORSEN = True

# PHASE3_GREEDY_WEIGHTS — (sortino, drawdown, win_rate) weights for greedy seed.
#   Raise sortino weight → favor risk-adjusted return in greedy pass.
#   Raise drawdown weight → favor lower DD teams early.
PHASE3_GREEDY_WEIGHTS = (0.8, 0.6, 0.5)

# --- Objectives & anti-overfit gates ---

# PHASE3_USE_TRAIN_TARGET — fit teams primarily on train metrics.
#   True  → faster but overfit risk; not recommended for production.
#   False → val-aware objectives (recommended with CV).
PHASE3_USE_TRAIN_TARGET = False

# PHASE3_USE_MAXIMIN_SCORE — use worst-symbol metric as robustness term.
#   True  → teams must not rely on one symbol for all PnL.
#   False → average metrics dominate; weak symbols may hide in team score.
PHASE3_USE_MAXIMIN_SCORE = True

# PHASE3_SYMBOL_CONSISTENCY_WEIGHT — penalty when train/val symbol sets diverge.
#   Higher → punish teams that trade different symbols in train vs val.
#   Lower  → allow symbol shift between splits.
PHASE3_SYMBOL_CONSISTENCY_WEIGHT = 10.0

# PHASE3_TRAIN_VAL_CORR_WEIGHT — penalty for low per-symbol PnL correlation.
#   Higher → demand similar symbol ranking train vs val.
#   Lower  → allow reordering of symbol profitability OOS.
PHASE3_TRAIN_VAL_CORR_WEIGHT = 8.0

# PHASE3_VAL_GATE_PENALTY — base penalty when val gates fail.
#   Higher → hard push toward val-feasible teams on objectives.
#   Lower  → borderline val teams remain competitive longer.
PHASE3_VAL_GATE_PENALTY = 10.0

# PHASE3_VAL_SORTINO_RATIO_GATE — require val_sortino ≥ ratio × train_sortino.
#   Higher → stricter val Sortino relative to train; anti-overfit.
#   Lower  → allow val Sortino collapse vs train.
PHASE3_VAL_SORTINO_RATIO_GATE = 0.5

# PHASE3_VAL_DRAWDOWN_RATIO_GATE — require val_dd ≤ ratio × train_dd.
#   Higher (e.g. 1.5) → allow val drawdown to exceed train more.
#   Lower (e.g. 1.0) → val DD must not worsen vs train.
PHASE3_VAL_DRAWDOWN_RATIO_GATE = 1.20

# PHASE3_PER_RULE_MIN_VAL_TRADES_PER_SYMBOL — min val trades per rule per symbol.
#   Higher → each rule must prove itself on every symbol in val.
#   Lower  → sparse per-symbol val activity allowed.
PHASE3_PER_RULE_MIN_VAL_TRADES_PER_SYMBOL = 5

# Return / PF floors for Phase 3 team admission (must align with Phase 2 quality).
# Higher floors → fewer teams pass; lower → more teams, weaker OOS risk.
PHASE3_VAL_RETURN_FLOOR_PCT = 0.5
PHASE3_VAL_PROFIT_FACTOR_FLOOR = 1.05
PHASE3_TRAIN_RETURN_FLOOR_PCT = 1.0
PHASE3_TRAIN_PROFIT_FACTOR_FLOOR = 1.05
PHASE3_MIN_PROFITABLE_SYMBOLS = 5
PHASE3_SYMBOL_MEDIAN_RETURN_FLOOR_PCT = 0.0

# PHASE3_TRAIN_VAL_GAP_MAX_PCT / VAL_TRAIN_GAP_MAX_PCT — allowed return gap %.
#   Higher → tolerate larger train/val divergence (more teams, more overfit risk).
#   Lower  → strict alignment; may empty Pareto on 2-fold CV.
PHASE3_TRAIN_VAL_GAP_MAX_PCT = 8.0
PHASE3_VAL_TRAIN_GAP_MAX_PCT = 8.0

# PHASE3_GAP_PENALTY_WEIGHT — multiplier on excess gap beyond allowed max.
#   Higher → large gaps heavily penalized on all objectives.
#   Lower  → gaps matter less in team ranking.
PHASE3_GAP_PENALTY_WEIGHT = 6.0

# --- Rule-team orthogonality (incremental value of adding a rule) ---

# PHASE3_MIN_INCREMENTAL_TRADES — min new val trades when adding a rule to team.
#   Higher → each rule must add meaningful new signal coverage.
#   Lower  → redundant rules can slip into teams.
PHASE3_MIN_INCREMENTAL_TRADES = 45

# PHASE3_INCREMENTAL_GATE_PENALTY — penalty when incremental trades below min.
#   Higher → strong penalty for redundant rules.
#   Lower  → teams may stack similar rules.
PHASE3_INCREMENTAL_GATE_PENALTY = 60.0

# PHASE3_JACCARD_PENALTY_WEIGHT — penalty for overlapping entry masks between rules.
#   Higher → force orthogonal entry timing across team members.
#   Lower  → allow correlated rules in same team.
PHASE3_JACCARD_PENALTY_WEIGHT = 35.0

# PHASE3_JACCARD_SIMILARITY_GATE — hard reject if Jaccard similarity exceeds this.
#   Higher → allow more overlap between rules.
#   Lower  → strict de-duplication of signal timing.
PHASE3_JACCARD_SIMILARITY_GATE = 0.75


# =============================================================================
# Phase 4 — Walk-forward risk optimization (TP / SL / capital)
# =============================================================================
# Rule conditions are frozen; only risk params are optimized via Optuna.
# With purged_rolling_cv, WF windows use every fold's val block.

# --- Search space bounds ---

# PHASE4_TP_MIN/MAX — take-profit search range (%).
#   Wider MAX → allow larger targets; fewer hits, bigger winners per trade.
#   Narrower → optimizer stuck with modest TP; may miss trend captures.
PHASE4_TP_MIN = 2.0
PHASE4_TP_MAX = 4.0

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
PHASE4_CAPITAL_PCT_MIN = 30.0
PHASE4_CAPITAL_PCT_MAX = 30.0

# PHASE4_TP_STEP / SL_STEP / CAPITAL_STEP — Optuna discretization granularity.
#   Smaller steps → finer search, more trials needed to explore space.
#   Larger steps → coarser optimum, faster convergence per trial.
PHASE4_TP_STEP = 0.5
PHASE4_SL_STEP = 0.5
PHASE4_CAPITAL_STEP = 5.0

# --- Optuna budget ---

# PHASE4_N_TRIALS — number of Optuna trials per direction.
#   Higher → better risk param fit, slower Phase 4.
#   Lower  → may miss optimal TP/SL; fast but coarse.
PHASE4_N_TRIALS = 200

PHASE4_SAMPLER = "tpe"  # "tpe" | "nsga2"
PHASE4_SEED: int = get_seed()

# PHASE4_N_JOBS — parallel Optuna workers.
#   Higher → faster on multi-core; trials are independent.
#   Lower  → sequential; reproducible ordering easier.
PHASE4_N_JOBS = 1

# PHASE4_HARD_CAP_NORMALIZE — scale capital so sum ≤ MAX_TOTAL_EXPOSURE_PCT.
#   True  → realistic portfolio cap; required for live-like exposure.
#   False  → raw trial capital may exceed 100% total exposure.
PHASE4_HARD_CAP_NORMALIZE = True

# --- Walk-forward windows on validation data ---

# PHASE4_WF_SPLITS — number of walk-forward windows on validation split.
#   Higher → stricter temporal robustness; each window smaller (trade starvation).
#   Lower  → larger windows, more trades per fold, less temporal coverage.
PHASE4_WF_SPLITS = 2

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
PHASE4_MIN_WORST_TRADES = 15

# PHASE4_MIN_WORST_FOLD_RETURN_PCT — min return % in worst WF window.
#   Higher → only consistently profitable windows pass; may zero feasible set.
#   Lower (more negative) → allow losing worst windows; more trials pass.
PHASE4_MIN_WORST_FOLD_RETURN_PCT = -2.0

# PHASE4_MIN_WORST_FOLD_PF — min profit factor in worst WF window.
#   Higher → stricter per-window profitability.
#   Lower → marginal worst-window PF allowed.
PHASE4_MIN_WORST_FOLD_PF = 1.0


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


# =============================================================================
# Cross-parameter sanity (import-time; catches CV / fold-gate drift)
# =============================================================================

assert CV_N_FOLDS >= 1, f"CV_N_FOLDS must be >= 1, got {CV_N_FOLDS}"
assert 1 <= PHASE2_CV_POOL_MIN_FOLDS_PASS <= CV_N_FOLDS, (
    "PHASE2_CV_POOL_MIN_FOLDS_PASS must be in [1, CV_N_FOLDS]; "
    f"got {PHASE2_CV_POOL_MIN_FOLDS_PASS} with CV_N_FOLDS={CV_N_FOLDS}"
)
assert 1 <= PHASE2_CV_RANK_MIN_FOLDS_PASS <= CV_N_FOLDS, (
    "PHASE2_CV_RANK_MIN_FOLDS_PASS must be in [1, CV_N_FOLDS]"
)
assert PHASE2_CV_RANK_MIN_FOLDS_PASS <= PHASE2_CV_POOL_MIN_FOLDS_PASS, (
    "rank fallback must be looser than or equal to strict pool gate"
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

    - Keep ``PHASE2_CV_FOLD_WORKERS=1`` — parallel fold threads spike peak
      GPU/RAM and trigger Linux OOM kills (SIGKILL) on 12 GiB Colab hosts.
    - Phase 3 uses GPUBacktestEngine (mask cache + batch eval path).
    - VRAM auto batch sizing uses the T4-friendly 128 cap when enabled.
    """
    global PHASE3_USE_GPU, PHASE2_GPU_BATCH_SIZE_AUTO
    if not is_colab_runtime():
        return
    PHASE3_USE_GPU = True
    PHASE2_GPU_BATCH_SIZE_AUTO = True


_apply_colab_gpu_defaults()


assert MIN_CONDITIONS <= MAX_CONDITIONS, (
    f"MIN_CONDITIONS ({MIN_CONDITIONS}) must be <= MAX_CONDITIONS ({MAX_CONDITIONS})"
)
assert PHASE2_ISLAND_EPOCH_GENERATIONS >= 1
assert 0.0 <= PHASE2_VIABILITY_RECOVERY_DEPLOYABLE_MUTATE_FRACTION <= 1.0
assert PHASE2_MIGRATION_EPOCH_INTERVAL >= 1
assert 0.0 < PHASE2_MIGRATION_SEED_FRACTION < 1.0
assert PHASE2_SHARED_ARCHIVE_MIN_SYMBOLS >= 1
assert PHASE3_SYMBOL_RULE_MIN_TRAIN_TRADES >= 1
assert PHASE3_SYMBOL_RULE_MIN_VAL_TRADES >= 1
