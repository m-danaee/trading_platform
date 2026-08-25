# Task 4: Single-source Purge + role-specific eligibility
> id: task-4
> slug: purge-unified
> branch: feat/task-4-purge-unified
> base_branch: main
> commit: d151be4 (d151be48c23dad8f02c1a852cc2bf1f1a1b72ea3)
> generated_at: 2026-08-25T15:06:56+03:30
> effort: M
> confidence: MEDIUM
> depends_on: task-2, task-3
> drift_threshold: STOP if >50 commits

## Goal
Make Purge a single source of truth derived from horizons/timeframes and replace generic `is_seed` with role-specific `eligible_for_role`.

## Evidence
- `gpu_fuzzy_trader/mtf/cross_fitting.py:19-21` — `DEFAULT_HWC_PURGE=1440`, `DEFAULT_MWC=240`, `DEFAULT_LWC=1440` vs stale `48*15m=12h` comment vs actual `MAX_HOLD_CANDLES=96` → 1440
- `gpu_fuzzy_trader/config.py:499-500,361` — `MTF_HWC_HORIZON_BARS=6*240=1440`, `MTF_MWC=4*60=240`, `MAX_HOLD_CANDLES=96*15=1440`
- `gpu_fuzzy_trader/mtf/cross_fitting.py:236-280` — `generate_oof_scores(exclude_seed=True)` generic skip

## Scope
- In: `gpu_fuzzy_trader/config.py` (derive `HWC_PURGE_MINUTES`, `MWC_PURGE_MINUTES`, `LWC_PURGE_MINUTES`, `purge_for_role()`), `gpu_fuzzy_trader/mtf/cross_fitting.py` (remove `is_seed`, add `eligible_for_role`), `gpu_fuzzy_trader/mtf/*`, `gpu_fuzzy_trader/scoring/*`
- Out: `gpu_fuzzy_trader/validation/rolling_cv.py`, `gpu_fuzzy_trader/data/splitter.py`
- Related callers: `generate_oof_scores`, HWC/MWC/LWC discovery

## Acceptance criteria
- [ ] `config.py` derives `HWC_PURGE=MTF_HWC_HORIZON_BARS*240`, `MWC_PURGE=MTF_MWC_HORIZON_BARS*60`, `LWC_PURGE=MAX_HOLD_CANDLES*15`; hardcoded defaults removed
- [ ] `TemporalFold` has no `is_seed` nor `purge_minutes`; geometry-only
- [ ] `eligible_for_role(fold, "hwc"|"mwc"|"lwc")` implemented: HWC all eligible folds, MWC excludes Fold1 lacking HWC OOF, LWC similarly
- [ ] `TemporalFold.get_train_slice(df, role=...)` or `purge_minutes=purge_for_role(role)` — purge at retrieval
- [ ] `generate_oof_scores` accepts `role=` param, delegates to `eligible_for_role`; `exclude_seed` deprecated or removed
- [ ] Tests assert HWC Fold1 usable, MWC Fold1 unavailable, no in-sample leakage

## Verification gates
1. `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest -q tests/unit/test_mtf_cross_fitting.py tests/unit/test_purge_leakage.py` — expected: pass
2. `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest -q tests/unit/test_mtf_audit_fixes.py tests/unit/test_mtf_composer.py` — expected: pass
3. `grep -R "is_seed" --include="*.py" gpu_fuzzy_trader/ tests/` — 0 production hits

## STOP conditions
- STOP if `gpu_fuzzy_trader/config.py` missing `MTF_HWC_HORIZON_BARS` at 499
- STOP if `validate_master_temporal_folds` still requires `is_seed` for downstream tests
- STOP if LWC purge derivation changes value and tests mismatch `MAX_HOLD_CANDLES`

## Implementation sketch
- Step 1: Add `purge_for_role` and derived constants in config.py.
- Step 2: Replace `is_seed` with `eligible_for_role` in cross_fitting.py.
- Step 3: Update `generate_oof_scores` signature.
- Step 4: Audit callers; update manifest export to drop `is_seed`/`purge_minutes`.

