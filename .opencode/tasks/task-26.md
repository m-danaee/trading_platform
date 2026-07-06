# Task 26 — f3 train+val blend (B / Item 10)

## Branch
`fix/f3-train-val-blend` (from `main`)

## Problem
The NSGA-III `f3` objective (currently `profit_factor` by default) uses
**train-only** data. The `win_rate` branch in the same code block
blends train+val correctly (`min(win_rate, val_wr)`) but the
`profit_factor` branch does not. This allows rules with great train PF
and poor val PF to dominate the Pareto front, contributing to
`corr_f1_f3=1.00` warnings and high `max_train_val_gap_ratio` values.

This is the third attempt at this fix — the prior plan flagged it as
"Item 10" and deferred it twice. Today's re-run data confirms the
gap is still present, so per the plan's own decision criteria, this
is the indicated next fix.

## Files to Edit
- `gpu_fuzzy_trader/phases/phase2_rule_pool.py` (function `compute_phase2_objectives_from_metrics`, the `f3_objective` selection block)
- `tests/unit/test_phase2_rule_pool.py` (add a new test in `TestEvaluateChromosome`)

## Required Behavior

### Source change
In `phase2_rule_pool.py` at the `f3_objective` selection block
(currently lines ~720-737), change the `elif f3_objective == "profit_factor":`
branch from:

```python
elif f3_objective == "profit_factor":
    f3_val = profit_factor
```

to (mirror the `win_rate` branch's blend pattern):

```python
elif f3_objective == "profit_factor":
    f3_val = profit_factor
    if val_metrics is not None and _cfg.PHASE2_JOINT_TRAIN_VAL:
        val_pf = float(val_metrics.get("val_profit_factor", profit_factor))
        if int(val_metrics.get("executed_trades", 0)) < val_trade_floor:
            f3_val = min(profit_factor, 0.0)
        else:
            f3_val = min(profit_factor, val_pf)
```

### Key facts to verify before implementing
1. The `val_profit_factor` key IS already written into `val_metrics` at
   `phase2_rule_pool.py:660` (in the `if val_metrics is not None:` block
   near `compute_phase2_objectives_from_metrics`). Confirmed.
2. The `val_trade_floor` local is already in scope at the f3 block
   (declared earlier in the function). Confirmed by reading the
   surrounding code.
3. The `PHASE2_JOINT_TRAIN_VAL` config flag already gates the
   `win_rate` blend — using the same gate preserves the
   "off-by-default, opt-in" semantics.
4. The `cv_fold_min` branch (line 720-727) and the
   `PHASE2_USE_TOTAL_RETURN_OBJ` override (lines 738-748) are
   downstream of the f3 selection block — they re-assign `f3_val`
   after this code runs, so this change only affects the
   `profit_factor` branch's contribution to `f3_val` before
   downstream overrides.

### Test change
Add a new test in `tests/unit/test_phase2_rule_pool.py` in the
`TestEvaluateChromosome` class (the test class containing
`test_evaluate_chromosome_use_total_return_obj` at line 1526).

The new test must cover **all three branches** of the new behavior:
1. When `JOINT_TRAIN_VAL=True` and `val_metrics.profit_factor` is provided:
   `f3_val = min(profit_factor, val_profit_factor)` (i.e., `-f3` is the
   **larger** of `-train_pf` and `-val_pf`, since f3 is minimized)
2. When `JOINT_TRAIN_VAL=True` and `val_metrics.executed_trades < val_trade_floor`:
   `f3_val = min(profit_factor, 0.0)` (i.e., val gate fails, penalize to 0)
3. When `JOINT_TRAIN_VAL=False`: `f3_val = profit_factor` (unchanged from
   current behavior — train-only)

Mirror the style of the existing test at line 1526: it uses a `MockEngine`
class, sets config attributes, calls `_evaluate_chromosome`, and
restores config in a `finally` block.

## Acceptance criteria
1. The `profit_factor` branch in the `f3_objective` selection block
   blends train+val the same way the `win_rate` branch does
2. The new test exists in `TestEvaluateChromosome` and covers all
   three branches (blend, val-gate, off)
3. All existing tests in `test_phase2_rule_pool.py` still pass with
   `PYTEST_LOW_MEMORY=1` (in particular, `test_evaluate_chromosome_use_total_return_obj`
   must still pass — it asserts specific values for f3 with the
   `profit_factor` branch, but uses `_evaluate_chromosome` which only
   sets `f3_val` from `profit_factor` when JOINT_TRAIN_VAL is False
   by default, so the test should be unaffected)
4. The `cv_fold_min` branch is unchanged
5. The `PHASE2_USE_TOTAL_RETURN_OBJ` override is unchanged
6. The `win_rate` branch is unchanged
7. No other code is touched

## Out of scope
- Do NOT change `PHASE2_F3_OBJECTIVE` default value
- Do NOT change `PHASE2_JOINT_TRAIN_VAL` default value
- Do NOT change the `_raw_feasibility_violation_score` function
  (which has its own train/val gap check, fixed in commit 8ff3328)
- Do NOT change any per-generation log lines
- Do NOT change the `val_profit_factor` key name in val_metrics
  (it's used elsewhere in the codebase — see grep results in
  `phase2_rule_pool.py` for downstream readers)
