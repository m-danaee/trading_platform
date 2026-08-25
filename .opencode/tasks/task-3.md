# Task 3: Fold-aware Count Gate system
> id: task-3
> slug: fold-gates
> branch: feat/task-3-fold-gates
> base_branch: main
> commit: d151be4 (d151be48c23dad8f02c1a852cc2bf1f1a1b72ea3)
> generated_at: 2026-08-25T15:06:56+03:30
> effort: M
> confidence: MEDIUM
> depends_on: task-2
> drift_threshold: STOP if >50 commits

## Goal
Create single unified `validation/fold_gates.py` that scales only count-based gates by effective exposure with absolute floor; quality gates remain fixed; ratio support gate replaces fixed count.

## Evidence
- `gpu_fuzzy_trader/config.py:2253-2272` — `scale_trade_floor()` with `PURGED_WF_*` absolute floor =5
- `gpu_fuzzy_trader/config.py:321-327` — `PURGED_WF_SCALE_TRADE_FLOORS`, `PURGED_WF_MIN_TRADE_FLOOR_ABSOLUTE`
- `gpu_fuzzy_trader/config.py:513` — `MTF_MIN_FOLD_SUPPORT=2` (fixed) vs desired ratio 0.67

## Scope
- In: `gpu_fuzzy_trader/validation/fold_gates.py` (new), `gpu_fuzzy_trader/config.py` (wire `FOLD_ABSOLUTE_MIN_TRADES`, `MTF_MIN_FOLD_SUPPORT_RATIO`), `gpu_fuzzy_trader/phases/*`, `gpu_fuzzy_trader/evolution/*`, `gpu_fuzzy_trader/rb_governor.py`
- Out: `gpu_fuzzy_trader/validation/rolling_cv.py`, `gpu_fuzzy_trader/data/splitter.py`
- Related callers: all `scale_trade_floor` consumers

## Acceptance criteria
- [ ] `validation/fold_gates.py` with `FoldExposure(rows,duration_bars,per_symbol_rows)`, `scale_count_gate(base, exposure, reference, absolute_min)=max(abs, ceil(base*Ef/Eref))`, `build_fold_gate_context(scoring_df, reference_df)`, `resolve_fold_gates(base_gates, ctx)` scaling only count gates (MinTrades/MinSignals/MinSupport/MinCandidate/min_trades_per_symbol) leaving PF/MCC/Edge/WinRate/Sortino/MDD/Return fixed
- [ ] Train-gate uses train exposure, OOF-gate uses OOF exposure (explicitly tested)
- [ ] `MTF_MIN_FOLD_SUPPORT` → `MTF_MIN_FOLD_SUPPORT_RATIO=0.67` with `RequiredFolds=max(2, ceil(eligible*ratio))`
- [ ] Task 1 gate tests import real module and pass; legacy `scale_trade_floor` migrated/removed
- [ ] Manifest-ready gate output dict (`exposure_ratio`, `min_trades`, `min_signals`, `profit_factor_floor` unchanged etc.)

## Verification gates
1. `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest -q tests/unit/test_fold_gates.py tests/unit/test_config_trade_scaling.py` — expected: pass
2. `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest -q tests/unit/test_mtf_cross_fitting.py tests/unit/test_mtf_composer.py` — expected: pass
3. `grep -R "scale_trade_floor" --include="*.py" gpu_fuzzy_trader/` — only shim or fold_gates remains

## STOP conditions
- STOP if `gpu_fuzzy_trader/config.py:scale_trade_floor` not at 2253 (assumption broken)
- STOP if Task 2 `FoldExposure` not present (dependency not met)
- STOP if quality-gate invariance tests fail

## Implementation sketch
- Step 1: Create `fold_gates.py` with absolute floor and exposure helpers.
- Step 2: Add `MTF_MIN_FOLD_SUPPORT_RATIO` config, deprecate fixed count.
- Step 3: Migrate callers from `config.scale_trade_floor` to `fold_gates.scale_count_gate`.
- Step 4: Document pooled vs macro guidance (Phase 17) in module docstring.

