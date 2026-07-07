# Task 3: RB Governor 2-Fold Walk-Forward Risk Grid

## Task ID
`task-3` (third of 12 tasks in the 2026-07-07 audit fix plan)

## Title
RB Governor 2-Fold Walk-Forward Risk Grid

## Goal
Fix audit finding #3 (RB Governor risk-grid overfits val_selection;
log shows long jumped train 32.63%→58.21%, valid 29.76%→49.92%
purely from the risk grid, then test collapsed 49.92%→22.22%) and
audit finding #12 (`PHASE4_TAIL_HOLDOUT_FRACTION=0.25` is defined
in config but never read by rb_governor.py). Score every TP/SL/
capital combo on TWO chronological folds of val_selection, pick the
combo with the best `min(fold1_score, fold2_score)`. Optionally
reserve a final 25% of val_selection as an untouched tie-break
holdout (uses the existing orphan `PHASE4_TAIL_HOLDOUT_FRACTION`).

## Audit Citation
- Confirmed by static inspection:
  - `rb_governor.py:1188` uses `scoring_val = val_selection_df if val_selection_df is not None else val_df` for BOTH selection AND risk grid.
  - `rb_governor.py:703 _optimize_risk` iterates the full TP×SL×capital grid on a single `valid_engine`.
  - `grep PHASE4_TAIL_HOLDOUT_FRACTION gpu_fuzzy_trader/rb_governor.py` returns 0 matches.
- Run log evidence (2026-07-07): "RB [long]: risk improve pass=1 rule=1 score=3463.83 train=44.70% valid=36.86%" then "saved 3-rule strategy | score=4620.73 train=58.21% valid=49.92%" — the +25pp train and +13pp valid came from risk grid picking TP/SL/capital combos that fit val_selection's outlier trades, not real OOS robustness.

## Target Files
- `gpu_fuzzy_trader/rb_governor.py`
  - `_optimize_risk` (line 703): add a 2-fold walk-forward variant or
    generalize the function to accept a list of valid engines.
  - The caller of `_optimize_risk` (around line 1188): pass 2 fold
    engines instead of 1, plus an optional tail-holdout engine for
    final reporting.
  - Add helper `_make_walk_forward_fold_engines(val_selection_df,
    n_splits, tail_holdout_frac, direction)` that returns 2 fold
    engines + 1 optional tail-holdout engine.
  - Add helper `_score_on_engines(train_engine, fold_engines,
    tail_holdout_engine, rules)` that returns
    `(train_m, fold_metrics_list, tail_metrics, min_fold_score)`.
- `gpu_fuzzy_trader/config.py`
  - Add `RB_RISK_GRID_WF_SPLITS=2` (default; 1 = legacy single-fold
    behavior).
  - Add `RB_RISK_GRID_USE_TAIL_HOLDOUT=True` (default; uses
    `PHASE4_TAIL_HOLDOUT_FRACTION` to reserve tail).
  - Confirm `PHASE4_TAIL_HOLDOUT_FRACTION=0.25` already exists
    (it does at line 1493); do NOT redefine.
- `tests/unit/test_rb_governor_cv_folds.py`
  - Add a 2-fold WF test with synthetic val_selection and assert:
    - The two best train-scoring combos that disagree on the folds
      produce different selections vs legacy.
    - The min(fold1, fold2) score is reported in the history.
- `tests/unit/test_rb_governor_tail_holdout.py` (new file)
  - Test the tail-holdout path with synthetic fold splits.

## Current Behavior
- `rb_governor.py:1188 scoring_val = val_selection_df if ... else val_df` uses
  ONE validation set for the entire risk grid.
- `rb_governor.py:703 _optimize_risk` signature is
  `(selected, train_engine, valid_engine, direction)`. The inner loop
  calls `_evaluate_ruleset(train_engine, valid_engine, trial)` for
  every TP/SL/capital combo and picks the highest score.
- `PHASE4_TAIL_HOLDOUT_FRACTION=0.25` is defined at config.py:1493
  with the comment "fraction of val reserved as final holdout window
  for risk-grid validation" but is never read.

## Scope
1. **Walk-forward split helper** (`rb_governor.py`):
   - New function `_make_walk_forward_fold_engines(val_selection_df,
     n_splits, tail_holdout_frac, direction) -> tuple[list[CPUBacktestEngine], CPUBacktestEngine | None]`.
   - Splits `val_selection_df` per-symbol chronologically into
     `n_splits` folds (each fold is a contiguous per-symbol slice).
   - If `tail_holdout_frac > 0`, reserves the final
     `tail_holdout_frac` of each symbol as a tail holdout (returned
     separately).
   - Returns a list of fold engines and an optional tail-holdout engine.
2. **Generalized risk-grid scoring** (`_optimize_risk`):
   - Either (preferred) generalize the function to accept a list of
     fold engines and a tail-holdout engine:
     `def _optimize_risk(selected, train_engine, fold_engines,
      tail_holdout_engine, direction) -> ...`
   - Or (alternative) add a new function `_optimize_risk_walk_forward`
     and keep the legacy `_optimize_risk` for the regression guard.
   - Choose (a) — the implementer may deviate if cleaner.
3. **Walk-forward selection criterion** (in the inner loop):
   - For each TP/SL/capital combo, score on ALL fold engines and
     compute `min_fold_score = min(score_fold_1, score_fold_2, ...)`.
   - Pick the combo with the highest `min_fold_score` (not the
     highest single-fold score).
   - The first fold's metrics (or aggregate) becomes the "primary"
     metrics reported in the history dict.
4. **Tail-holdout usage**:
   - The tail-holdout engine is NOT used during the risk-grid
     search. It is used ONLY for the final post-search tie-break
     report: re-score the chosen combo on the tail holdout and add
     `risk_tail_holdout_return_pct`, `risk_tail_holdout_pf`,
     `risk_tail_holdout_dd` to the output JSON.
5. **Config flags** (config.py):
   - Add `RB_RISK_GRID_WF_SPLITS=2` (default).
   - Add `RB_RISK_GRID_USE_TAIL_HOLDOUT=True` (default).
   - Confirm `PHASE4_TAIL_HOLDOUT_FRACTION=0.25` exists; do NOT
     redefine. If the comment doesn't mention the RB Governor,
     update the comment.
6. **Regression guard**:
   - `RB_RISK_GRID_WF_SPLITS=1` and `RB_RISK_GRID_USE_TAIL_HOLDOUT=False`
     must preserve pre-task-3 behavior bit-identically.
   - The legacy `_optimize_risk` is kept for the single-fold case.
7. **Do NOT touch**:
   - The 2026-07-07 profit amplifier (RB_PROFIT_AMPLIFIER_ENABLED=False;
     that's task-S1 from the audit, deferred from the plan).
   - `_profit_amp_*` helpers at rb_governor.py:778+.
   - The selection logic at rb_governor.py:540-600 (extending from
     Phase 2 pool). The risk grid comes AFTER selection.
   - Any other file outside `rb_governor.py` and `config.py`.

## Acceptance Criteria
1. With `RB_RISK_GRID_WF_SPLITS=2` and a synthetic val_selection
   that has 2 folds with DIFFERENT optimal TP/SL combos, the
   2-fold version picks the combo that performs well on BOTH folds
   (worst-case selection), not the one that wins fold-1 alone.
2. The history dict entries gain a `min_fold_score` field and a
   `fold_scores` list (length = `RB_RISK_GRID_WF_SPLITS`).
3. When `RB_RISK_GRID_USE_TAIL_HOLDOUT=True`, the output JSON
   contains `risk_tail_holdout_return_pct`,
   `risk_tail_holdout_pf`, `risk_tail_holdout_dd` for the final
   selected combo.
4. With `RB_RISK_GRID_WF_SPLITS=1` and
   `RB_RISK_GRID_USE_TAIL_HOLDOUT=False`, behavior matches
   pre-task-3 exactly (regression test).
5. `PHASE4_TAIL_HOLDOUT_FRACTION=0.25` is now read by rb_governor.py
   (grep `PHASE4_TAIL_HOLDOUT_FRACTION gpu_fuzzy_trader/rb_governor.py`
   returns ≥1 match).
6. The 2-fold selection with the 2026-07-07 log data would NOT
   pick the `score=4620.73 train=58.21% valid=49.92%` combo if
   fold-1 and fold-2 disagree (this is an aspirational test — the
   implementer should add a synthetic test that demonstrates the
   rejection of the overfit combo).
7. All existing `test_rb_governor_*.py` tests still pass (no
   regression in the RB Governor test suite).

## Verification
Run only related unit tests with `PYTEST_LOW_MEMORY=1` and `.venv`:

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_rb_governor_cv_folds.py -q
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_rb_governor_tail_holdout.py -q
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_rb_governor_data_load.py -q
```

Mark any test that requires JAX or full pipeline data with
`@pytest.mark.uses_jax`.

## Notes
- Do NOT modify `evaluator_v5.ipynb`.
- Do NOT run the full project or full test suite locally (OOM risk
  per AGENTS.md; user runs on Colab GPU).
- The walk-forward split is per-symbol chronological (matches the
  existing val_selection split convention in `data/splitter.py`).
- If `RB_RISK_GRID_WF_SPLITS > 2`, the implementation should still
  work but is not required by the spec; aim for `n_splits >= 2`
  generality but only test with `n_splits=2`.
- The 2-fold scoring adds ~2× the per-combo evaluation cost. With
  the existing 8 TP × 5 SL × 4 capital = 160 combos × 2 folds =
  320 evaluations per rule. With RB_RISK_OPT_PASSES=2 and ~3 rules,
  this is ~1920 extra evaluations. Acceptable.
- If implementation reveals a deeper issue with `_evaluate_ruleset`
  (e.g., it caches by chromosome key and fold-engine swap breaks
  the cache), return BLOCKED with the exact error and a minimal
  repro rather than shipping a partial fix.
