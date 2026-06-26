# Task 2 — `fix/holdout-fitness-leak` (Fix C)

## Branch
`fix/holdout-fitness-leak` (from latest `main`, after task-1 merge).

## Problem
`PHASE2_JOINT_TRAIN_VAL=True` (`config.py:570`) folds the holdout `val`
(209590 rows) into Phase 2 evolution fitness via
`robust_return_pct = min(train, val)` (`phase2_support.py:314-330`). The
holdout is meant for model selection (Phase 3) — using it in Phase 2 fitness
means evolution optimizes against it, so it stops being an honest holdout and
`test.csv` (Phase 5 OOS) suffers.

## Required Changes

### Primary fix — disable joint train+val fitness
**File:** `gpu_fuzzy_trader/config.py` (line 570)
```python
# BEFORE
PHASE2_JOINT_TRAIN_VAL = True
# AFTER
PHASE2_JOINT_TRAIN_VAL = False
```
With `joint=False`, `robust_return_pct` returns train-only return (the val
simulation still runs for reporting metrics like `max_robust_return` in logs —
this is acceptable but see optional optimization below). Robustness during
evolution comes from the already-wired purged 4-fold CV evaluator
(`cv_fold_evaluator`, logged as "purged CV fitness evaluator (3 folds)").

### Verification of val reuse (no code change unless a leak is found)
- Confirm `val_df` / `val_engine` in Phase 3 (`phase3_rule_set.py`,
  `phase3_cache.py`) is used for *rule-set selection* — this is the PROPER use
  of a holdout and should be KEPT.
- Confirm Phase 4 (`phase4_wf_optimizer.py`) does not use `val_df` for tuning
  in a way that would make val a third training set. If it does, report it but
  do NOT change Phase 4 behavior in this task (out of scope — flag to user).
- Confirm `test.csv` is consumed ONLY in Phase 5 (`phase5_oos.py`).

### Optional optimization (only if low-risk and clearly isolated)
When `PHASE2_JOINT_TRAIN_VAL=False`, the per-chromosome `val_engine.simulate_*`
calls in `evox_runner.py` (~lines 1172, 1485, 1676) still run purely for
reporting. If straightforward, guard them behind `if _cfg.PHASE2_JOINT_TRAIN_VAL:`
to save GPU time. If this risks changing reported metrics or breaks tests, SKIP
it and leave a `# TODO` comment instead. Do not over-engineer.

### Docs
Update `README.md` config table entry for `PHASE2_JOINT_TRAIN_VAL` to reflect
the new default and the rationale (holdout must stay clean for OOS).

## Acceptance Criteria
1. `PHASE2_JOINT_TRAIN_VAL == False`.
2. Unit test: `robust_return_pct(train_m, val_m, joint=False)` returns
   `train_m["total_return_pct"]` (train-only) — i.e. val does not pull it down.
3. Phase 3 still uses `val_engine` for selection (no regression in phase3 tests).
4. `test.csv` referenced only in Phase 5 (grep verification, documented in PR).
5. `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit -q` passes.

## Target Files
- `gpu_fuzzy_trader/config.py`
- `gpu_fuzzy_trader/phases/phase2_support.py` (optional guard) — only if safe
- `README.md`
- `tests/unit/test_phase2_support.py` (add joint=False test) or new test file.

## Verification
```
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit -q
grep -rnE "test\.csv|load_test|OOS_Evaluator" gpu_fuzzy_trader/   # confirm test.csv scope
```
Do NOT run the full pipeline.
