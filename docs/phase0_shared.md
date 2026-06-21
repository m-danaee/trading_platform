# Phase 0 — Data, split, backtest

Shared infrastructure used by every phase.

**Source code:**
- Data loader: [`gpu_fuzzy_trader/data/loader.py`](../gpu_fuzzy_trader/data/loader.py)
- Splitter: [`gpu_fuzzy_trader/data/splitter.py`](../gpu_fuzzy_trader/data/splitter.py)
- Rolling CV: [`gpu_fuzzy_trader/validation/rolling_cv.py`](../gpu_fuzzy_trader/validation/rolling_cv.py)
- CPU backtest engine: [`gpu_fuzzy_trader/backtest/cpu_engine.py`](../gpu_fuzzy_trader/backtest/cpu_engine.py)
- GPU backtest engine: [`gpu_fuzzy_trader/backtest/gpu_engine.py`](../gpu_fuzzy_trader/backtest/gpu_engine.py)

**Hyperparameter reference:** [README.md §3](../README.md#3-phase-0--data-split-backtest)

## Caching

Phase 0 persists outputs to skip recomputation when the split/feature configuration is unchanged:

| Path | Written by | Consumed by |
|------|-----------|-------------|
| `data/train_70.parquet` | splitter (holdout_70_30) | Phases 1–4 |
| `data/validation_30.parquet` | splitter (holdout_70_30) | Phases 2–4 |
| `data/cv_folds_manifest.json` | rolling_cv (purged_walk_forward) | Phases 2–4 |

**Cache invalidation rule:** delete these files whenever `SPLIT_MODE` or any `PURGED_WF_*` parameter changes — stale parquets silently override new config.

## Evaluator parity

The following constants in [`gpu_fuzzy_trader/config.py`](../gpu_fuzzy_trader/config.py) **must match** `evaluator_v5.ipynb` (per AGENTS.md):

`FEE_PCT`, `MAX_HOLD_CANDLES`, `INITIAL_CAPITAL`, `LEVERAGE`, `MAX_TOTAL_EXPOSURE_PCT`, `MIN_POSITION_NOTIONAL`. See [README.md §12](../README.md#12-evaluator-parity-checklist) for the full checklist.
