# Task 4: Hot-path data & transfer optimization (loader, MTF, features, caches)
> id: task-4
> slug: t4-data-pipeline
> commit: bae3102 (bae3102470c445d9907f226e080dca72aa0d4500)
> base_branch: main
> effort: M
> confidence: MEDIUM
> depends_on: task-3
> branch: feature/task-4-t4-data-pipeline

## Evidence
- `gpu_fuzzy_trader/data/loader.py:120-210` – `_ensure_labels` merges labels, `validate_context_columns`, `drop_tail` per symbol, `fillna(0)` object dtypes; single-threaded pandas
- `gpu_fuzzy_trader/backtest/df_slim.py` – `slim_backtest_df`, `downcast_numeric_df` (numeric downcast but not always early)
- `gpu_fuzzy_trader/data/multi_timeframe.py:1-401` – causal HTF resampling + `merge` per phase/repeated
- `gpu_fuzzy_trader/run_pipeline.py:280-450` – `_merge_mtf_score_columns` / `_merge_mtf_lwc_runtime_columns` merge on `(datetime,symbol)` with sort-heavy path; drops MTF columns then merges
- `gpu_fuzzy_trader/features/selector.py:400-800` – `mutual_info_classif`, sign-consistency, stationarity folds (sklearn `n_jobs` defaults to 1, python loops for pruning)
- `gpu_fuzzy_trader/backtest/barrier.py` – Numba `first_touch` O(N*bars) already fast; risk of recompute per fold
- `gpu_fuzzy_trader/backtest/condition_cache.py` – per-engine rule mask cache

## Scope
- In: `gpu_fuzzy_trader/data/loader.py`, `gpu_fuzzy_trader/backtest/df_slim.py`, `gpu_fuzzy_trader/data/multi_timeframe.py`, `gpu_fuzzy_trader/features/selector.py`, `gpu_fuzzy_trader/features/encoder.py`, `gpu_fuzzy_trader/run_pipeline.py` (merge helpers), `gpu_fuzzy_trader/mtf/runtime.py`
- Out (do NOT touch): `evaluator_v5.ipynb`, `config` thresholds, `cpu_engine.py`/`gpu_engine.py` risk math, RB gating thresholds
- Related callers (blast): `run_pipeline.py` orchestrator calls loader→splitter→selector→phase2→mtf; every downstream phase reads prepared DF – run `nexus impact --json --targets gpu_fuzzy_trader/data/loader.py,gpu_fuzzy_trader/data/multi_timeframe.py,gpu_fuzzy_trader/features/selector.py,gpu_fuzzy_trader/run_pipeline.py`

## Acceptance criteria
- [ ] `Data_Loader.load_dataset` uses `dtype` pinning / `pyarrow` engine where possible, converts `symbol` to `category` early, ensures `datetime` is `datetime64[ns]` via vectorized `pd.to_datetime(...,utc=True).dt.tz_localize(None)` without per-row Python loops; micro-bench `loader_ms` improves vs baseline in `scripts/benchmark_t4.py`
- [ ] `downcast_numeric_df`/`slim_backtest_df` called once immediately after load (not repeatedly) and barrier outcomes computed once and cached per tape hash (avoid recompute per fold); cache key via `sha256_file` on CSV path
- [ ] MTF merges use `sort=False` + `validate=one_to_one` (already) plus ensure `datetime` is int64-ish and `symbol` is `category` before merge to avoid object-merges; duplicate-key error paths preserved
- [ ] `Feature_Selector` MI ranking uses `n_jobs=min(2,cpu_count)` when estimator allows, dispersion pruning vectorized (no python loops over columns), same `selected_features_{long,short}.json` output for fixed seed on same synthetic tape
- [ ] No wasted/duplicate code remains after change (old dead branches removed per AGENTS.md clean-project rule)

## Verification gates (exact commands)
1. `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest -q tests/unit/test_feature_selector_smoke.py tests/unit/test_multi_timeframe.py` – expected: passing (adjust to actual file names under `tests/unit/test_*`; if file missing, use closest `tests/unit/test_*selector*` and `tests/unit/test_*multi*`)
2. `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest -q tests/property/test_data_loader_properties.py tests/property/test_feature_selector_properties.py` – expected: passing
3. `.venv/bin/python scripts/benchmark_t4.py --component loader --rows 50000` – expected: JSON with `loader_ms` < baseline and dtype checks (`symbol` is category, `datetime` not object)
4. `git diff main...feature/task-4-t4-data-pipeline --stat` – only `data/loader.py`, `df_slim.py`, `multi_timeframe.py`, `features/selector.py`, `encoder.py`, `run_pipeline.py`, `mtf/runtime.py` touched
5. `.venv/bin/python -c "from gpu_fuzzy_trader.data.loader import Data_Loader; print('loader ok')"` – expected: `loader ok` (no import regression)

## STOP conditions
- STOP if any property test requiring `TAIL_DROP_ROWS==96` / `MAX_HOLD_CANDLES==96` fails (geometry corrupted; tail rows dropped incorrectly)
- STOP if `Feature_Selector.run` returns different `selected_features_long.json` content for fixed seed on same synthetic tape (accuracy regression – feature selection must be deterministic-speedup only)
- STOP if merge helpers raise new `MergeError` on valid one-to-one frames or swallow existing `ValueError("MTF score frame is missing columns")` paths (contract violation)
- STOP if `downcast_numeric_df` downcast changes barrier outcome dtype semantics (must preserve int/float precision for barrier columns)

## Implementation sketch
- In `loader.py`: add `engine="pyarrow"` branch with explicit dtypes (`symbol` category, OHLCV float32), vectorized `pd.to_datetime`, immediate `downcast_numeric_df`, cache barrier via tape hash file under `/tmp` or `outputs/`
- In `run_pipeline.py`: cast `symbol` to `category` before merge, keep `sort=False` and `validate`, document int64 merge key, remove redundant drops when empty
- In `features/selector.py`: wire `n_jobs=min(2,cpu_count)` into `mutual_info_classif` call, vectorize dispersion via `df.nunique`/`value_counts` masks, memoize per-fold MI if random_state fixed

## Graph context
- Hubs: `data/loader.py` (feeds `run_pipeline.py`, `splitter.py`), `run_pipeline.py` (orchestrator), `features/selector.py` (Phase1)
- Blast: `nexus impact --json --targets gpu_fuzzy_trader/data/loader.py,gpu_fuzzy_trader/data/multi_timeframe.py,gpu_fuzzy_trader/features/selector.py`
