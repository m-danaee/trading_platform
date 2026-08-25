# Task 1: Fold/Gate regression & leakage tests first
> id: task-1
> slug: tests-first
> branch: feat/task-1-tests-first
> base_branch: main
> commit: d151be4 (d151be48c23dad8f02c1a852cc2bf1f1a1b72ea3)
> generated_at: 2026-08-25T15:06:56+03:30
> effort: M
> confidence: HIGH
> depends_on: none
> drift_threshold: >50 commits or base_branch changed → STOP

## Goal
Write the Fold + Gate + Purge leakage regression tests BEFORE any production code changes, locking the spec for adaptive folds, gate scaling, and role-specific OOF.

## Evidence
- `tests/unit/test_mtf_cross_fitting.py:1-80` — existing fold structure/purge tests (seed, expanding train, contiguous tests)
- `gpu_fuzzy_trader/mtf/cross_fitting.py:89-168` — current `build_master_temporal_folds` expanding logic to be characterized
- `gpu_fuzzy_trader/config.py:2253-2272` — current `scale_trade_floor()` formula `max(absolute, ceil(base * E_f/E_ref))`
- `gpu_fuzzy_trader/mtf/cross_fitting.py:236-271` — `generate_oof_scores` with `exclude_seed` / `is_seed` skip to be replaced

## Scope
- In: `tests/unit/test_unified_folds.py` (new), `tests/unit/test_fold_gates.py` (new), `tests/unit/test_purge_leakage.py` (new), shared helper `_make_synthetic_df`
- Out: `gpu_fuzzy_trader/config.py`, `gpu_fuzzy_trader/mtf/cross_fitting.py`, `gpu_fuzzy_trader/validation/*`, `gpu_fuzzy_trader/data/splitter.py`
- Related callers / blast radius: future `validation/fold_gates.py`, `FoldEligibility`, `eligible_for_role`

## Acceptance criteria
- [ ] `test_unified_folds` covers: train precedes test, no overlap, monotonic expanding train, test intervals contiguous, adaptive-K tiny-fold rejection, per-symbol coverage (NEWCOIN half-history)
- [ ] `test_fold_gates` covers: ref 100k base 40 → 100k→40, 50k→20, 25k→10, 5k→absolute_min=5; PF/MCC/MDD unchanged with fold size; Train-gate uses train exposure, OOF-gate uses OOF exposure
- [ ] `test_purge_leakage` covers: synthetic future spike, HWC/MWC/LWC last usable train label never reaches test_start; HWC Fold1 usable, MWC Fold1 unavailable
- [ ] Tests import inline spec (not requiring future module) so they pass on `main` before refactor
- [ ] All new tests pass on `main`

## Verification gates (exact commands)
1. `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest -q tests/unit/test_mtf_cross_fitting.py tests/unit/test_data_splitter.py` — expected: pass on main
2. `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest -q tests/unit/test_unified_folds.py tests/unit/test_fold_gates.py tests/unit/test_purge_leakage.py` — expected: all pass
3. `git diff main...HEAD --stat` — only `tests/` files changed

## STOP conditions
- STOP if `tests/unit/test_mtf_cross_fitting.py` already failing on `main` (run baseline first)
- STOP if `gpu_fuzzy_trader/mtf/cross_fitting.py:build_master_temporal_folds` does not contain expanding logic at 89-168
- STOP if `git log main..HEAD -- tests/` shows another task already modified tests

## Implementation sketch
- Step 1: Helper `_make_synthetic_df(symbols, freq, missing)` with datetime + symbol + close.
- Step 2: Fold geometry tests vs `build_master_temporal_folds` / `validate_master_temporal_folds`.
- Step 3: Gate scaling tests using inline `max(abs, ceil(base * Ef/Eref))` spec (so Task 1 independent of Task 3 module).
- Step 4: Leakage tests: synthetic clear future pattern, assert purged train max datetime ≤ test_start - purge.
- Step 5: Role-specific OOF test: HWC Fold1 predicted, MWC Fold1 empty.

## Notes
- Use `PYTEST_LOW_MEMORY=1` per AGENTS.md; .venv required.
- Keep tests deterministic (UTC-normalized timestamps).

