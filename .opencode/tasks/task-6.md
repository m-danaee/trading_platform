# Task 6: Remove SPLIT_MODE + simplify splitter/cache + fix validation purge
> id: task-6
> slug: splitter-cleanup
> branch: feat/task-6-splitter-cleanup
> base_branch: main
> commit: d151be4 (d151be48c23dad8f02c1a852cc2bf1f1a1b72ea3)
> generated_at: 2026-08-25T15:06:56+03:30
> effort: M
> confidence: MEDIUM
> depends_on: task-5
> drift_threshold: STOP if >50 commits

## Goal
Remove `SPLIT_MODE` abstraction, delete `_purged_walk_forward_split`, simplify validation fitness/selection to single 96-candle gap, and rename stale 70/30 artifacts.

## Evidence
- `gpu_fuzzy_trader/config.py:251` — `SPLIT_MODE = "holdout"`
- `gpu_fuzzy_trader/data/splitter.py:114-139` — `_purged_walk_forward_split()` to delete; mode branch in `split_and_persist`
- `gpu_fuzzy_trader/data/splitter.py:173-210` — `split_validation_fitness_selection()` uses `max(VALIDATION_HALF_PURGE,HOLDOUT_EMBARGO,PURGED_WF_EMBARGO)` with double-gap 96+96 waste
- `gpu_fuzzy_trader/config.py:367-371` — `VALIDATION_HALF_PURGE_CANDLES = MAX_HOLD_CANDLES =96`
- `gpu_fuzzy_trader/config.py:165-166` — `TRAIN_70_PATH` / `VALIDATION_30_PATH` stale names

## Scope
- In: `gpu_fuzzy_trader/config.py` (remove `SPLIT_MODE`, define `VALIDATION_PURGE_CANDLES=MAX_HOLD_CANDLES`), `gpu_fuzzy_trader/data/splitter.py` (simplify `split_validation_fitness_selection`, `load_cached_split_if_fresh`, `split_and_persist`), path renames, tests
- Out: `gpu_fuzzy_trader/validation/nested_walk_forward.py` (Task 7)
- Related callers: `Data_Splitter`, manifest fingerprint, reporting

## Acceptance criteria
- [ ] `SPLIT_MODE` and `_purged_walk_forward_split` removed; only `_holdout_embargo_split` remains
- [ ] `split_validation_fitness_selection` uses single gap `VALIDATION_PURGE_CANDLES = MAX_HOLD_CANDLES` (96) — `fitness |----96----| selection`
- [ ] Cache manifest no longer contains `split_mode`/`purged_config_fingerprint`; holds `holdout_train_fraction`, `embargo_candles`, `purge_candles`
- [ ] Artifact names renamed: `data/train_70.parquet` → `data/development_train.parquet`, `data/validation_30.parquet` → `data/validation.parquet` (constants `DEVELOPMENT_TRAIN_PATH`/`VALIDATION_PATH` with fallback aliases)
- [ ] `test_data_splitter.py` updated and passes with new geometry

## Verification gates
1. `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest -q tests/unit/test_data_splitter.py` — pass
2. `grep -R "SPLIT_MODE\|_purged_walk_forward\|PURGED_WF" --include="*.py" gpu_fuzzy_trader/ tests/` — 0 hits (outside docs)
3. `ls data/development_train.parquet data/validation.parquet 2>&1 || ls data/train_70.parquet data/validation_30.parquet` — new paths or fallback

## STOP conditions
- STOP if `_holdout_embargo_split` not at line 43
- STOP if changing `VALIDATION_HALF_PURGE_CANDLES` breaks `validate_config` invariant `TAIL_DROP==MAX_HOLD`
- STOP if `load_cached_split_if_fresh` cannot be simplified without heavy CSV re-read

## Implementation sketch
- Step 1: Remove `SPLIT_MODE`, replace with `VALIDATION_PURGE_CANDLES = MAX_HOLD_CANDLES`, add `DEVELOPMENT_TRAIN_PATH`.
- Step 2: Delete purged branch, simplify `split_validation_fitness_selection` to `purge_rows=VALIDATION_PURGE_CANDLES` single gap.
- Step 3: Update `load_cached_split_if_fresh` manifest checks; add fallback read for legacy parquet paths.
- Step 4: Update `write_cv_folds_manifest` naming if present.

