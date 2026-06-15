# Documentation Index

Technical documentation for the GPU-Fuzzy Trading Pipeline. Each document covers one phase in depth: algorithm details, function-by-function explanations, and the effect of hyperparameters in `gpu_fuzzy_trader/config.py`.

| Document | Phase | Key topics |
|---|---|---|
| [phase0_shared.md](phase0_shared.md) | Phase 0 — Shared | Data loading, train/val split modes (purged CV vs 75/25), backtest engine, shared config |
| [phase1_feature_selection.md](phase1_feature_selection.md) | Phase 1 | Feature mode detection, MI scoring, stationarity filter, regime clustering |
| [phase2_rule_pool.md](phase2_rule_pool.md) | Phase 2 | NSGA-III evolution, joint train/val fitness, **purged CV worst-fold scoring** |
| [phase3_rule_set.md](phase3_rule_set.md) | Phase 3 | Greedy + NSGA-II, train-target objectives, CV fold gates, orthogonality penalties |
| [phase4_wf_risk.md](phase4_wf_risk.md) | Phase 4 | Walk-forward on `validation_25`, Deterministic Grid TP/SL/capital search |
| [phase5_oos.md](phase5_oos.md) | Phase 5 | True OOS on `test.csv`, cross-split reports, generalization diagnostics |

## Data flow (three time horizons)

```
data/train.csv
    │
    ├─► Phases 1–4 (in-sample)
    │       SPLIT_MODE controls Phases 2–3:
    │         purged_rolling_cv → K folds, worst-fold fitness
    │         holdout_75_25     → single 75/25 per symbol
    │       Persisted for Phase 4–5 reports:
    │         train_75.parquet + validation_25.parquet
    │         (last CV fold when using purged_rolling_cv)
    │
    └─► data/test.csv  →  Phase 5 only (never tune on this file)
```

## Quick hyperparameter index

| Goal | Parameter | Phase doc |
|---|---|---|
| **Stronger generalization (short OOS)** | `SPLIT_MODE`, `CV_N_FOLDS`, `CV_MIN_TRAIN_MONTHS` | Phase 0, 2, 3 |
| Faster iteration / legacy behaviour | `SPLIT_MODE = "holdout_75_25"` | Phase 0 |
| Reduce GPU memory | `PHASE1_SAMPLING_TOTAL` | Phase 1, 2 |
| More trades per rule | `MIN_TRADE_SUPPORT` | Phase 2 |
| Rule complexity | `MIN_CONDITIONS`, `MAX_CONDITIONS` | Phase 2 |
| Evolution budget | `PHASE2_POPULATION_SIZE`, `PHASE2_GENERATIONS` | Phase 2 |
| Warm-start from prior runs | `PHASE2_ARCHIVE_SEED_FRACTION` | Phase 2 |
| Symbol coverage in teams | `PHASE3_MIN_SYMBOL_COVERAGE` | Phase 3 |
| Anti-overlap rule teams | `PHASE3_JACCARD_SIMILARITY_GATE` | Phase 3 |
| Train/val stability gates | `PHASE3_VAL_SORTINO_RATIO_GATE`, `PHASE3_*_GAP_*` | Phase 3 |
| Refinement budget | `PHASE3_REFINE_POP_SIZE`, `PHASE3_REFINE_GENERATIONS` | Phase 3 |
| Risk search budget | `PHASE4_GRID_PASSES`, `PHASE4_N_TRIALS` (Optuna fallback) | Phase 4 |
| WF windows on validation | `PHASE4_WF_SPLITS`, `PHASE4_INCLUDE_TAIL_HOLDOUT` | Phase 4 |
| TP/SL search range | `PHASE4_GRID_TP_VALUES`, `PHASE4_GRID_SL_VALUES` | Phase 4 |
| Feature count per direction | `PHASE1_TOP_K_FEATURES` | Phase 1 |
| Long/short feature divergence | `PHASE1_MAX_FEATURE_OVERLAP` | Phase 1 |
| Fees / label horizon | `FEE_PCT`, `TAIL_DROP_ROWS`, `MAX_HOLD_CANDLES`, `CV_EMBARGO_BARS` | Phase 0 |

## Config file

All defaults and tuning notes live in **`gpu_fuzzy_trader/config.py`** (module docstring includes a quick tuning map).
