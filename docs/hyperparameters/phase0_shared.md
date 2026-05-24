# Phase 0 — Shared Constants

Constants used across multiple pipeline phases: paths, data schema, backtest simulation, and logging. Defined at the top of [`gpu_fuzzy_trader/config.py`](../../gpu_fuzzy_trader/config.py).

[← Back to index](README.md)

---

## Purpose

These settings define **where data lives**, **which columns are features vs labels**, and **how trades are simulated**. They must stay aligned with `evaluator_v3.ipynb` so optimization scores match final evaluation.

Phases 1–5 all inherit backtest constants. Changing them changes the objective landscape for every downstream phase.

---

## Paths

| Constant               | Default                            | Used by            | Performance effect                                           |
| ---------------------- | ---------------------------------- | ------------------ | ------------------------------------------------------------ |
| `TRAIN_CSV_PATH`       | `"data/train.csv"`                 | Loader, Phases 1–4 | Training universe; larger/different data changes all metrics |
| `TEST_CSV_PATH`        | `"data/test.csv"`                  | Phase 5 only       | True OOS; never use for tuning                               |
| `TRAIN_75_PATH`        | `"data/train_75.parquet"`          | Splitter cache     | Auto-written; speeds re-runs                                 |
| `VALIDATION_25_PATH`   | `"data/validation_25.parquet"`     | Splitter cache     | Auto-written                                                 |
| `OUTPUTS_DIR`          | `"outputs"`                        | All phases         | Overridable via `--output` CLI flag                          |
| `REPORTS_DIR`          | `"outputs/reports"`                | Reporter, Phase 5  | Plots and CSV reports                                        |
| `RUN_LOG_PATH`         | `outputs/run.log`                  | Pipeline           | Operational logging                                          |
| `PHASE2_POOL_DIR`      | `OUTPUTS_DIR`                      | Phase 2            | Runtime override for run-specific dirs                       |
| `PHASE2_POOL_PATHS`    | `phase2_{long,short}_pool.json`    | Phase 2 → 3        | Per-run rule pools                                           |
| `PHASE2_HISTORY_PATHS` | `phase2_{long,short}_history.json` | Phase 2            | Generation metrics for diagnostics                           |
| `PHASE2_ARCHIVE_DIR`   | `phase2_rule_archive/`             | Phase 2            | **Persistent** across runs                                   |
| `PHASE2_ARCHIVE_PATHS` | `phase2_{long,short}_archive.json` | Phase 2            | Warm-start seed for evolution                                |

**Archive note:** `PHASE2_ARCHIVE_*` is the only path that persists when you use `--output`. Compatible archived chromosomes seed ~35% of the next Phase 2 population (`PHASE2_ARCHIVE_SEED_FRACTION`), improving convergence across runs without changing fitness semantics.

---

## Schema

| Constant           | Default              | Performance effect                                                                                                                                                           |
| ------------------ | -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `LABEL_COLUMNS`    | 5 look-ahead columns | Define 288-bar horizon labels; must not be used as features                                                                                                                  |
| `META_COLUMNS`     | `datetime`, `symbol` | Excluded from Phase 1; used for splitting and grouping                                                                                                                       |
| `INTERNAL_COLUMNS` | `_symbol_bar_index`  | Bar index within symbol; excluded from features                                                                                                                              |
| `TAIL_DROP_ROWS`   | `288`                | Last 288 rows per symbol dropped (labels unavailable). **Must match label generation horizon.** Lowering without regenerating labels misaligns Phase 1 targets and backtests |

### Label columns (reference)

| Column                            | Role                                |
| --------------------------------- | ----------------------------------- |
| `label_open_next`                 | Entry reference price               |
| `label_close_288`                 | Price at horizon end                |
| `label_min_288` / `label_max_288` | Path extremes (TP/SL simulation)    |
| `label_max_before_min`            | Tie-breaker when both TP and SL hit |

---

## Backtest constants

Must match `evaluator_v3.ipynb` exactly. Implemented in [`cpu_engine.py`](../../gpu_fuzzy_trader/backtest/cpu_engine.py) and [`gpu_engine.py`](../../gpu_fuzzy_trader/backtest/gpu_engine.py).

| Constant                 | Default  | ↑ increase effect                  | ↓ decrease effect   | Notes                                                    |
| ------------------------ | -------- | ---------------------------------- | ------------------- | -------------------------------------------------------- |
| `INITIAL_CAPITAL`        | `1000.0` | Scales absolute PnL only           | Same                | Sortino ratio unchanged; return % unchanged              |
| `LEVERAGE`               | `1.0`    | Larger positions, higher variance  | Smaller positions   | Multiplies notional exposure                             |
| `FEE_PCT`                | `0.20`   | Penalizes high-turnover rules more | Favors more trades  | Round-trip fee %; dominant filter for noisy rules        |
| `MAX_HOLD_CANDLES`       | `288`    | Longer hold window                 | Shorter holds       | Tied to label horizon; affects Phase 1 asymmetric target |
| `MAX_TOTAL_EXPOSURE_PCT` | `100.0`  | Allows more concurrent exposure    | Caps portfolio heat | Interacts with Phase 4 capital allocation                |
| `MIN_POSITION_NOTIONAL`  | `1.0`    | Fewer dust trades filtered         | More tiny trades    | Prevents negligible positions                            |

### Interactions

- **`FEE_PCT` × trade frequency:** High fees make sparse, high-conviction rules dominate fitness — same effect as raising `MIN_TRADE_SUPPORT` indirectly.
- **`MAX_HOLD_CANDLES` × Phase 1 target:** Asymmetric target uses TP/SL hit logic over the hold window; changing hold without updating labels breaks MI scoring.
- **`MAX_TOTAL_EXPOSURE_PCT` × Phase 4:** Phase 4 can assign per-rule `capital_pct` summing above 100%; `PHASE4_HARD_CAP_NORMALIZE` scales down post-hoc.

---

## Logging

| Constant                  | Default | Effect                                                                                   |
| ------------------------- | ------- | ---------------------------------------------------------------------------------------- |
| `LOG_GENERATION_INTERVAL` | `0`     | `0` = auto-throttle generation logs; positive int = log every N generations in Phase 2/3 |

No impact on model performance; affects observability during long evolutionary runs.

---

## Related phase docs

Phase-specific static risk during rule mining is documented under Phase 2 (`PHASE2_TP`, `PHASE2_SL`, `PHASE2_CAPITAL_PCT`) even though it lives in the shared config block — those values intentionally isolate **alpha search** from **risk tuning** (Phase 4).

- [Phase 1 — Feature selection](phase1_feature_selection.md)
- [Phase 2 — Rule pool](phase2_rule_pool.md)
