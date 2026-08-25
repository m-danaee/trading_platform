# Task 7: Remove/rename pseudo-nested + final cleanup + auditability
> id: task-7
> slug: nested-rename-audit
> branch: feat/task-7-nested-rename-audit
> base_branch: main
> commit: d151be4 (d151be48c23dad8f02c1a852cc2bf1f1a1b72ea3)
> generated_at: 2026-08-25T15:06:56+03:30
> effort: S
> confidence: MEDIUM
> depends_on: task-5, task-6
> drift_threshold: STOP if >50 commits

## Goal
Remove or rename pseudo `nested_walk_forward`, finalize simplified config shape, and ensure per-fold auditability (train/test rows, per-symbol rows, purge per role, base/scaled gates, eligible/reason) in `mtf_manifest.json`.

## Evidence
- `gpu_fuzzy_trader/validation/nested_walk_forward.py:1-50` — `evaluate_nested_strategy()` not true nested (evaluates final strategy over outer folds)
- `gpu_fuzzy_trader/config.py:158-159` — `NESTED_VALIDATION_ENABLED`, `NESTED_VALIDATION_OUTER_FOLDS`
- `gpu_fuzzy_trader/config.py:499-516` — final config should be `HOLDOUT_TRAIN_FRACTION`, `MAX_HOLD_CANDLES`, `MTF_MAX_FOLDS`, `MTF_MIN_FOLDS`, `FOLD_MIN_*`, `MTF_MIN_FOLD_SUPPORT_RATIO`, `MTF_HWC_HORIZON_BARS`, `MTF_MWC_HORIZON_BARS`

## Scope
- In: `gpu_fuzzy_trader/validation/nested_walk_forward.py` (delete or rename to `walk_forward_stability_report.py` with `evaluate_strategy_stability`), `gpu_fuzzy_trader/config.py` (remove `NESTED_VALIDATION_*` or gate under research profile), `gpu_fuzzy_trader/reporting/*`, `README.md`/`RUN.md`, `gpu_fuzzy_trader/mtf/cross_fitting.py` manifest export
- Out: `gpu_fuzzy_trader/validation/fold_gates.py`, `gpu_fuzzy_trader/data/splitter.py`
- Related callers: `run_pipeline.py`, `research_integrity.py`

## Acceptance criteria
- [ ] `nested_walk_forward.py` removed from canonical pipeline (no import in `run_pipeline.py`) or renamed to `walk_forward_stability_report.py` with `evaluate_strategy_stability`
- [ ] `NESTED_VALIDATION_*` removed or moved behind `research_profile.py` opt-in
- [ ] Final config shape simplified to `HOLDOUT_TRAIN_FRACTION`, `MAX_HOLD_CANDLES`, `MTF_MAX_FOLDS`/`MTF_MIN_FOLDS`, `FOLD_MIN_*`, `MTF_MIN_FOLD_SUPPORT_RATIO`, `MTF_HWC_HORIZON_BARS`/`MTF_MWC_HORIZON_BARS`; derived purge documented; no `PURGED_WF_*`
- [ ] Per-fold audit: `train_start/end`, `test_start/end`, `train_rows`, `test_rows`, `per_symbol_*`, `purge_hwc/mwc/lwc`, `base_min_trades/scaled_min_trades`, `eligible/reason` in `mtf_manifest.json`
- [ ] Full CPU test slice passes after all tasks

## Verification gates
1. `grep -R "nested_walk_forward\|NESTED_VALIDATION" --include="*.py" gpu_fuzzy_trader/` — 0 canonical hits (or only walk_forward_stability)
2. `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest -q tests/unit/test_mtf_cross_fitting.py tests/unit/test_mtf_composer.py tests/unit/test_data_splitter.py tests/unit/test_fold_gates.py tests/unit/test_purge_leakage.py` — all pass
3. `cat outputs/mtf_manifest.json 2>/dev/null | head -n 80 || grep -n "MTF_MAX_FOLDS" gpu_fuzzy_trader/config.py` — audit fields present

## STOP conditions
- STOP if `run_pipeline.py` still imports `nested_walk_forward` after claimed removal
- STOP if removing `NESTED_VALIDATION_*` breaks `research_integrity.py` (`grep -R NESTED_VALIDATION`)
- STOP if any `PURGED_WF`/`SPLIT_MODE` remains after all tasks (`grep` must be 0)

## Implementation sketch
- Step 1: Grep and decide delete vs rename; create `walk_forward_stability_report.py` if keeping diagnostic.
- Step 2: Remove/gate `NESTED_VALIDATION_*` from config; update docs.
- Step 3: Extend `export_fold_boundaries` + eligibility + gate contexts to include audit fields.
- Step 4: Update README/RUN folding section to single unified architecture.

