# Task 3: Stage 3 — Fitness-function gap fixes (the core fix)

## Source plan
`/home/danaee/.claude/plans/you-are-a-senior-pure-cupcake.md` — Stage 3, items 8-9 (item 10 deferred)

## Branch
`fix/phase2-stage3-fitness-gap` (from `main`)

## Files to touch
- `gpu_fuzzy_trader/phases/phase2_rule_pool.py` (Item 8)
- `gpu_fuzzy_trader/phases/phase2_support.py` (Item 9)
- `gpu_fuzzy_trader/config.py` (rename threshold parameter)
- `tests/unit/test_phase2_rule_pool.py` (new tests for both items)
- `tests/unit/test_phase2_support.py` (new test for Item 9)
- `tests/unit/test_config_additions.py` (if needed for renamed config key)
- `gpu_fuzzy_trader/optuna_search.py` (update if it references the renamed config)

## Changes

### Item 8: Fix `overfit_gap_penalty`'s blind spot

**Current bug** (`phases/phase2_rule_pool.py:818-834`):
- Ratio-based: `gap_ratio = train_ret / max(val_ret, 1e-6)`
- Gated on `if val_ret > 0.0` — a rule with train=99%/val≤0% gets **zero** penalty
- The worst overfit case dodges the only in-loop check

**Fix:** Switch to subtraction-based (`train_ret - val_ret`), matching the final pool-admission gate's check. This unifies the two different "gap" definitions in the codebase (ratio in-loop vs. subtraction at final gate) into one.

Replace the current block in `phases/phase2_rule_pool.py` with:
```python
overfit_gap_penalty = 0.0
if val_metrics is not None and float(_cfg.PHASE2_OVERFIT_GAP_PENALTY_WEIGHT) > 0.0:
    train_ret = float(metrics.get("total_return_pct", 0.0))
    val_ret = float(val_metrics.get("total_return_pct", 0.0))
    gap_pct = train_ret - val_ret
    if gap_pct > float(_cfg.PHASE2_OVERFIT_GAP_PCT_THRESHOLD):
        overfit_gap_penalty = (
            (gap_pct - _cfg.PHASE2_OVERFIT_GAP_PCT_THRESHOLD)
            * float(_cfg.PHASE2_OVERFIT_GAP_PENALTY_WEIGHT)
        )
```

**Config rename** in `config.py`:
- `PHASE2_OVERFIT_GAP_RATIO_THRESHOLD` → `PHASE2_OVERFIT_GAP_PCT_THRESHOLD` (units change from ratio to pct-points)
- Keep `PHASE2_OVERFIT_GAP_PENALTY_WEIGHT`
- Update the comment block at `config.py:585-591` to reflect subtraction-based semantics
- Suggested new default: ~8-10pp (below the hard gate `PHASE2_MAX_TRAIN_VAL_GAP_PCT=16.0`). The plan says "this is a tunable you should sanity-check against your own risk tolerance, not a value I'd fix unilaterally" — pick a reasonable default like 8.0 and note it in the handoff.

**New test** in `tests/unit/test_phase2_rule_pool.py` (extend near `TestRobustReturnObjective` at line 1657):
- Assert penalty for `train=99%/val=-10%` is **larger** than for `train=99%/val=1%` (monotonicity / no-blind-spot)
- Do NOT assert exact numbers — assert ordering/monotonicity
- Add a test that asserts the penalty is **non-zero** when `val_ret <= 0` (the bug case)

### Item 9: Add same gap check to `_raw_feasibility_violation_score`

**Why this is highest-leverage:** `_raw_feasibility_violation_score` is the single choke point — it feeds:
- `passes_evolution_deployability_preview` (→ `_update_deployable_archive` in `evolution/evox_runner.py:571-614`, which gates `deployable_archive` membership and therefore what `_preserve_deployable_elites` can force-pin into the population every generation)
- The real objectives via `support_penalty` (weighted 0.4/0.6/0.6 into f1/f2/f3 in `phase2_rule_pool.py:793-801`)

It currently checks trade-count/return/PF floors but never the train-vs-val gap, unlike the final pool-admission gate (phase2_support.py:179-181).

**Fix in `phases/phase2_support.py:_raw_feasibility_violation_score`** (~line 329-369):
After the existing val_pf check (~line 367), add:
```python
gap = train_ret - val_ret
max_gap = float(getattr(_cfg, "PHASE2_MAX_TRAIN_VAL_GAP_PCT", 16.0))
if gap > max_gap:
    score += (gap - max_gap) * 1.0  # weight tunable, same order as existing PF term
```

**New test** in `tests/unit/test_phase2_support.py` (extend `TestDeployabilityHelpers` at ~line 122-194):
- `train_ret=90%`, `val_ret=10%` (gap=80pp, over threshold), but all individual floors otherwise passing
- Assert `_raw_feasibility_violation_score(...) > 0.0` (was 0.0 pre-fix)
- Assert `passes_evolution_deployability_preview(...) is False` (was True pre-fix)
- This directly encodes the bug as a regression test

### Item 10: DEFERRED per plan
f1/f3 asymmetry (making f3 worst-of-train/val like f1) is explicitly out of scope. The plan says: "land #8/#9 first, re-run, and only pursue this if the gap is still insufficiently controlled."

## Acceptance criteria
- [ ] `overfit_gap_penalty` uses subtraction (`train_ret - val_ret`), not ratio
- [ ] `overfit_gap_penalty` is well-defined for `val_ret <= 0` (no `if val_ret > 0` gate)
- [ ] `PHASE2_OVERFIT_GAP_PCT_THRESHOLD` (new name) exists in config; old `PHASE2_OVERFIT_GAP_RATIO_THRESHOLD` removed
- [ ] No remaining references to `PHASE2_OVERFIT_GAP_RATIO_THRESHOLD` (grep whole repo)
- [ ] `optuna_search.py` updated to use new name (if it references the old one)
- [ ] New monotonicity test for `overfit_gap_penalty` passes (penalty at val=-10% > penalty at val=1%)
- [ ] New test for blind spot: penalty non-zero when val_ret <= 0
- [ ] `_raw_feasibility_violation_score` includes train-vs-val gap check using `PHASE2_MAX_TRAIN_VAL_GAP_PCT`
- [ ] New test: train_ret=90%/val_ret=10% with otherwise-passing floors → `_raw_feasibility_violation_score > 0` AND `passes_evolution_deployability_preview is False`
- [ ] Pre-existing test `test_f3_uses_min_train_val_return` should now PASS (was failing in Tasks 1-2)
- [ ] All touched test suites pass with `PYTEST_LOW_MEMORY=1`

## Hard rules
- Do NOT change the `_derive_island_seed` signature (unrelated to this task).
- Do NOT push to remote, do NOT merge to main.
- Use `.venv/bin/python` for any test command.
- Use `PYTEST_LOW_MEMORY=1`.
- Only run touched test suites, not full suite (OOM risk per AGENTS.md).
- Commit message prefix: `fix(task-3): <item summary>`. One commit per item, or one consolidated commit.

## Verification command
```
cd /home/danaee/trading_platform
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_rule_pool.py tests/unit/test_phase2_support.py tests/unit/test_phase2_island_scheduler.py tests/unit/test_island_scheduler_migration.py tests/unit/test_migration_safety.py tests/unit/test_elite_preservation.py tests/unit/test_evox_runner.py tests/unit/test_config_additions.py -v
```

If `test_f3_uses_min_train_val_return` was failing in Tasks 1-2 due to the `overfit_gap_penalty` shape change, the new subtraction-based logic should make it pass.
