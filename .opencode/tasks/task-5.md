# Task 5: Remove deprecated purged_walk_forward system
> id: task-5
> slug: remove-rolling-cv
> branch: feat/task-5-remove-rolling-cv
> base_branch: main
> commit: d151be4 (d151be48c23dad8f02c1a852cc2bf1f1a1b72ea3)
> generated_at: 2026-08-25T15:06:56+03:30
> effort: S
> confidence: HIGH
> depends_on: task-3, task-4
> drift_threshold: STOP if >50 commits

## Goal
Delete `validation/rolling_cv.py` and all `PURGED_WF_*` configs/helpers after migration to `fold_gates`.

## Evidence
- `gpu_fuzzy_trader/validation/rolling_cv.py:1-538` — entire file to delete
- `gpu_fuzzy_trader/config.py:289-332` — `PURGED_WF_N_SPLITS/HOLDOUT_FRACTION/EMBARGO/MIN_TRAIN/MIN_VALID/AGGREGATION/REQUIRE_ALL/SCALE_TRADE_FLOORS/MIN_TRADE_FLOOR_ABSOLUTE`, `CV_FOLDS_MANIFEST_PATH`, `_PURGED_WF_REFERENCE_ROWS`
- `gpu_fuzzy_trader/config.py:2240-2272` — `split_mode_is_purged_walk_forward`, `set_purged_wf_reference_rows`, legacy `scale_trade_floor`

## Scope
- In: `gpu_fuzzy_trader/validation/rolling_cv.py` (delete), `gpu_fuzzy_trader/config.py` (remove purged knobs/helpers), update tests
- Out: `gpu_fuzzy_trader/mtf/cross_fitting.py`, `gpu_fuzzy_trader/data/splitter.py` (handled in Task 6)
- Related callers: `gpu_fuzzy_trader/data/splitter.py:_purged_walk_forward_split`, `tests/unit/test_config_trade_scaling.py`

## Acceptance criteria
- [ ] `validation/rolling_cv.py` deleted; `grep -R rolling_cv` 0 hits
- [ ] `config.py` no longer contains `PURGED_WF_*`, `CV_FOLDS_MANIFEST_PATH` (replaced), `_PURGED_WF_REFERENCE_ROWS`, `split_mode_is_purged_walk_forward`, `set_purged_wf_reference_rows`, legacy `scale_trade_floor` (or only shim)
- [ ] `validate_config` no longer checks purged mode branch
- [ ] Migrated tests pass (`test_config_trade_scaling` migrated to fold_gates)

## Verification gates
1. `ls gpu_fuzzy_trader/validation/rolling_cv.py` — expected: not found
2. `grep -R "PURGED_WF_\|split_mode_is_purged\|_PURGED_WF_REFERENCE" --include="*.py" gpu_fuzzy_trader/` — 0 hits
3. `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest -q tests/unit/test_data_splitter.py tests/unit/test_mtf_cross_fitting.py` — pass

## STOP conditions
- STOP if `grep -R "from gpu_fuzzy_trader.validation.rolling_cv"` still finds imports (audit before delete)
- STOP if config block at 289-332 not as expected (plan drift)
- STOP if `test_config_trade_scaling.py` asserts old API without replacement

## Implementation sketch
- Step 1: `grep -R "rolling_cv\|PURGED_WF\|CV_FOLDS_MANIFEST"` migrate needed helpers to `fold_gates.py`.
- Step 2: Delete file, remove config entries/helpers, update `validate_config`.
- Step 3: Update tests to use `fold_gates`.

