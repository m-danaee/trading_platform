# Task-22: Restore Phase2→RB-Governor→OOS Objective Continuity

**Branch:** `fix/objective-continuity`
**Priority:** 🔴 Critical
**Depends on:** none (may conflict with task-21 on evox_runner.py / phase2_rule_pool.py — coordinate if both land in same PR)

## Problem

The 2026-07-01 root-cause audit found three breaks in the objective chain from
Phase 2's multi-objective search through RB Governor to Phase 5 OOS:

### Problem 1: `PHASE2_F3_OBJECTIVE="cv_fold_min"` never actually executes
`config.py:459` sets `PHASE2_F3_OBJECTIVE = "cv_fold_min"`, but:

- The batched NSGA-III evaluation path (`_evaluate_population_indices` in
  `evox_runner.py`) never populates `_cv_fold_returns` in metrics — it's only
  populated in the slow single-chromosome fallback path (`_evaluate_chromosome`
  in `phase2_rule_pool.py:783-810`).

- `compute_phase2_objectives_from_metrics` (line 600) correctly checks for
  `_cv_fold_returns` but finds it empty for 100% of batch-evaluated individuals,
  silently falling back to `profit_factor`.

- Result: Phase 2 advertised as optimizing cross-validation robustness
  but actually optimizes training-set profit factor. All NSGA-III search
  decisions were made under the wrong objective.

### Problem 2: Phase 2 pool entries never include `tp`/`sl`/`capital_pct`
- `_build_pool_from_archive` (line 1384) builds pool entries with
  `chromosome`, `conditions`, `objectives`, `val_objectives`, `executed_trades`
  — but NOT `tp`, `sl`, `capital_pct`.

- `extract_deployable_migrants` (line 2748) also omits these fields.

- RB Governor's `_rule_to_engine` (line 137) reads them with
  `rule.get("tp", RB_DEFAULT_TP)`, `rule.get("sl", RB_DEFAULT_SL)`,
  `rule.get("capital_pct", RB_DEFAULT_CAPITAL_PCT)` — since they're always
  missing, RB Governor silently re-parametrizes every rule to defaults
  that Phase 2 never optimized for (e.g., SL 1.0%→1.2%, capital 30%→20%).

### Problem 3: RB Governor's scoring lacks CV-fold consistency
RB Governor's `_score_metrics` (line 160) uses return-to-drawdown ratio on
train/valid split but does not incorporate CV-fold return consistency — rules
can score well on a single train/valid split while having highly variable
per-fold returns. The `cv_fold_returns` plumbing exists (lines 498, 554, 1200)
but is only used for filtering, not for influencing the grid-search objective.

## Files to Modify

1. `gpu_fuzzy_trader/config.py` — revert `PHASE2_F3_OBJECTIVE` to `"profit_factor"`
2. `gpu_fuzzy_trader/phases/phase2_rule_pool.py` — persist `tp`/`sl`/`capital_pct` on pool entries
3. `gpu_fuzzy_trader/evolution/evox_runner.py` — persist `tp`/`sl`/`capital_pct` on migrant entries
4. `gpu_fuzzy_trader/rb_governor.py` — add CV-fold consistency to scoring

## Detailed Changes

### R1: Make F3 objective label truthful

In `config.py`, change:
```python
PHASE2_F3_OBJECTIVE = "cv_fold_min"
```
to:
```python
PHASE2_F3_OBJECTIVE = "profit_factor"
```

Update the config comment to explain: `cv_fold_min` requires batched CV-fold
evaluation which is prohibitively expensive in the NSGA-III inner loop;
`profit_factor` (what actually executes today) is the truthful config label.
CV-fold robustness is instead enforced at the pool-admission gate and RB
Governor scoring stages.

### R2: Persist `tp`/`sl`/`capital_pct` on pool entries

In `_build_pool_from_archive` (`phase2_rule_pool.py`, after line 1384), add
these fields to the `pool_entry` dict:
```python
"tp": float(_cfg.PHASE2_TP),
"sl": float(_cfg.PHASE2_SL),
"capital_pct": float(_cfg.PHASE2_CAPITAL_PCT),
```

In `extract_deployable_migrants` (`evox_runner.py`, line 2748), add the same:
```python
"tp": float(_cfg.PHASE2_TP),
"sl": float(_cfg.PHASE2_SL),
"capital_pct": float(_cfg.PHASE2_CAPITAL_PCT),
```

This ensures RB Governor's `_rule_to_engine` picks up Phase 2's validated
operating point instead of unrelated `RB_DEFAULT_*` values.

### R3: RB Governor CV-fold consistency in scoring

In `rb_governor.py`, `_score_metrics` already accepts `cv_fold_returns` as an
optional parameter (line 160) but doesn't use it. The callers at lines 554
and 1200 pass `cv_fold_returns` but the function ignores them.

Add a CV-fold consistency penalty to `_score_metrics`:
```python
if cv_fold_returns and len(cv_fold_returns) > 1:
    cv_min = min(cv_fold_returns)
    cv_range = max(cv_fold_returns) - cv_min
    cv_mean = sum(cv_fold_returns) / len(cv_fold_returns)
    # Penalize rules where worst fold is negative or fold range is extreme
    if cv_min < 0:
        score -= abs(cv_min) * 5.0
    # Penalize high variance across folds (inconsistent OOS)
    if cv_range > abs(cv_mean) * 2.0:
        score -= (cv_range - abs(cv_mean)) * 2.0
```

## Acceptance Criteria

- [ ] R1: `PHASE2_F3_OBJECTIVE` reverted to `"profit_factor"` in config.py with truthful comment.
- [ ] R2: Pool entries and migrant entries both include `tp`, `sl`, `capital_pct` with Phase 2's actual values.
- [ ] R3: RB Governor's `_score_metrics` incorporates `cv_fold_returns` when available.
- [ ] Existing tests pass: `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_rule_pool.py tests/unit/test_evox_runner.py tests/unit/test_rb_governor.py -x -q`
- [ ] New/updated test asserting pool entries contain tp/sl/capital_pct.
- [ ] `evaluator_v5.ipynb` NOT modified.

## Verification

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_rule_pool.py tests/unit/test_evox_runner.py tests/unit/test_rb_governor.py -x -q
```
