# Task 2: Unified FoldContext + FoldEligibility + Adaptive K
> id: task-2
> slug: fold-eligibility
> branch: feat/task-2-fold-eligibility
> base_branch: main
> commit: d151be4 (d151be48c23dad8f02c1a852cc2bf1f1a1b72ea3)
> generated_at: 2026-08-25T15:06:56+03:30
> effort: M
> confidence: MEDIUM
> depends_on: task-1
> drift_threshold: >50 commits or base_branch changed → STOP

## Goal
Introduce `FoldContext`/`FoldEligibility`, adaptive `K` (try 4→3→2, fail closed), equal-time partitions, and per-symbol audit rows so tiny folds can never be accepted.

## Evidence
- `gpu_fuzzy_trader/mtf/cross_fitting.py:30-56` — `TemporalFold(fold_id, train_start, train_end, test_start, test_end, is_seed, purge_minutes)`
- `gpu_fuzzy_trader/config.py:510-513` — `MTF_N_FOLDS=4`, `MTF_MIN_FOLD_SUPPORT=2`, `MTF_HWC_HORIZON_BARS=6`
- `gpu_fuzzy_trader/mtf/cross_fitting.py:89-168` — current fixed-K equal-time split
- `gpu_fuzzy_trader/config.py:289-332` — deprecated purged configs to be distinguished from new folding configs

## Scope
- In: `gpu_fuzzy_trader/mtf/cross_fitting.py`, `gpu_fuzzy_trader/config.py` (new: `MTF_MAX_FOLDS`, `MTF_MIN_FOLDS`, `MTF_MIN_FOLD_SUPPORT_RATIO`, `FOLD_MIN_EFFECTIVE_ROWS`, `FOLD_MIN_ROWS_PER_SYMBOL`, `FOLD_ABSOLUTE_MIN_TRADES`, `FOLD_MIN_DURATION_BARS`, `FOLD_MIN_SYMBOL_COVERAGE`), `gpu_fuzzy_trader/validation/fold_manifest.py` (optional)
- Out: `gpu_fuzzy_trader/validation/rolling_cv.py`, `gpu_fuzzy_trader/data/splitter.py`, `gpu_fuzzy_trader/validation/nested_walk_forward.py`
- Related callers: `generate_oof_scores`, future `validation/fold_gates.py`, reporting, `tests/unit/test_unified_folds.py`

## Acceptance criteria
- [ ] `TemporalFold` geometry-only (remove `purge_minutes` here; `is_seed` deferred to Task 4 but prepare)
- [ ] New dataclasses: `FoldContext(train_rows,test_rows,per_symbol_rows,train_duration,test_duration,symbols_available,eligible)`, `FoldExposure`
- [ ] `build_master_temporal_folds(df, max_folds=4, ...)` adaptive: try K=4 then 3 then 2; fail closed if even min eligible insufficient
- [ ] Equal-time partitions preserved (time-based, not row-equalized); eligibility checks rows + duration + per-symbol coverage + symbol count
- [ ] Manifest records per-fold `train_rows/test_rows/per_symbol_*` for audit
- [ ] `validate_master_temporal_folds` still passes monotonic/expanding/contiguous checks
- [ ] Task 1 `test_unified_folds` passes with real adaptive logic

## Verification gates
1. `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest -q tests/unit/test_unified_folds.py` — expected: pass (adaptive K)
2. `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest -q tests/unit/test_mtf_cross_fitting.py` — expected: pass (updated expectations)
3. `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest -q tests/unit/test_config_validation.py` — expected: pass

## STOP conditions
- STOP if `gpu_fuzzy_trader/mtf/cross_fitting.py` no longer contains `TemporalFold` at line 30
- STOP if `gpu_fuzzy_trader/config.py` already has `MTF_MAX_FOLDS` from another branch
- STOP if `grep -R validate_master_temporal_folds --include="*.py" gpu_fuzzy_trader/` reveals downstream callers that cannot tolerate field removal

## Implementation sketch
- Step 1: Add config knobs with `validate_config` checks; keep `MTF_N_FOLDS` as deprecated alias pointing to `MTF_MAX_FOLDS`.
- Step 2: Add `FoldContext`/`FoldEligibility` and `assess_fold_eligibility(fold, df)` computing per-symbol rows/duration.
- Step 3: Refactor `build_master_temporal_folds` to loop K descending, build equal-time boundaries, assess eligibility, return eligible set or raise fail-closed.
- Step 4: Remove `TemporalFold.purge_minutes`, update `to_dict`/`get_train_slice` to require explicit purge param.

