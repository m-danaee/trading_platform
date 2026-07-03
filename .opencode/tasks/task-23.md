# Task-23: Config/Logging Anomaly Cleanup

**Branch:** `fix/config-logging-anomalies`
**Priority:** 🟢 Low
**Depends on:** none (purely cosmetic/consistency, no runtime-behavior dependency)

## Problem

Three minor config/logging inconsistencies identified in the root-cause audit:

### Issue 1: Startup banner always claims legacy Phase 3/4
`run_pipeline.py` module docstring (lines 2-16) lists execution order as:
```
5. Phase 3: Rule_Set_Selector for both directions
6. Phase 4: WalkForwardRiskOptimizer for both directions
7. Phase 5: OOS_Evaluator (always runs)
```

This is stale — RB Governor replaces Phase 3+4, but the docstring never
mentions it. The pipeline summary (`_print_run_summary`) also always prints
"Phase 3 rule sets" / "Phase 4 optimized strategies" without indicating
whether RB Governor was active.

### Issue 2: `skip_symbol_robustness_penalty` always True
In `config.py:2107`, `resolve_island_hyperparams` unconditionally sets
`skip_symbol_robustness_penalty=True` for both cluster and orphan profiles.
This field is typed as `bool` on `IslandHyperparams` but is never configurable,
making it dead config. The comment should clarify why it's always True for
island-scoped runs (islands have fewer symbols → min_profitable_symbols check
is already more lenient, so the robustness penalty would be redundant/too
aggressive on small symbol sets).

### Issue 3 (optional): Generation-budget remainder dropping
In `phase2_island_scheduler.py:370`, `_should_skip_epoch` drops remaining
generations less than `PHASE2_ISLAND_MIN_EPOCH_GENERATIONS`. When the budget
has a remainder (e.g., 33 total gens / 3 clusters = 11 each, with epoch=5 →
2 remaining gens are dropped per cluster), this wastes computation budget.
A minor fix: accumulate remainder and assign to one cluster, or use
`math.ceil` division.

## Files to Modify

1. `gpu_fuzzy_trader/run_pipeline.py` — update module docstring; add RB Governor mention to summary
2. `gpu_fuzzy_trader/config.py` — clarify `skip_symbol_robustness_penalty` comment; optionally fix budget rounding
3. `gpu_fuzzy_trader/phases/phase2_island_scheduler.py` — optionally fix generation-budget remainder

## Detailed Changes

### C1: Update pipeline docstring and summary

In `run_pipeline.py` (lines 2-16), update:
```
3. Phase 1: Feature_Selector
4. Phase 2: Rule_Pool_Generator for both directions
5. Phase 3: Rule_Set_Selector (legacy — skipped when RB Governor active)
6. Phase 4: WalkForwardRiskOptimizer (legacy — skipped when RB Governor active)
7. RB Governor: unified selection + risk tuning (replaces Phase 3+4)
8. Phase 5: OOS_Evaluator (always runs)
```

In `_print_run_summary` (lines 381, 395), add a note when RB Governor is active:
```python
if phase == 3:
    if _cfg.RB_GOVERNOR_ENABLED:
        print("  Phase 3: SKIPPED (RB Governor replaces Phase 3+4)")
    else:
        # existing Phase 3 summary code
```

### C2: Clarify skip_symbol_robustness_penalty

In `config.py`, add a comment above line 2107:
```python
# skip_symbol_robustness_penalty is always True for island-scoped
# (cluster/orphan) runs because the per-symbol median profit check
# (PHASE2_SYMBOL_MEDIAN_RETURN_FLOOR_PCT) is too aggressive for small
# symbol sets (most clusters have only 2-5 symbols). The
# min_profitable_symbols gate already provides cross-symbol quality
# control at an appropriate threshold.
skip_symbol_robustness_penalty=True,
```

### C3 (optional): Fix generation-budget remainder

In `phase2_island_scheduler.py`, change the budget division to avoid remainder dropping:
- Use `math.ceil` to distribute remainder across first N clusters, OR
- Accumulate remainder and assign to one cluster

This is optional — only implement if it can be done cleanly without changing behavior for the no-remainder case.

## Acceptance Criteria

- [ ] C1: Module docstring in `run_pipeline.py` mentions RB Governor replacing Phase 3+4.
- [ ] C1: `_print_run_summary` indicates when RB Governor is active vs legacy Phase 3/4.
- [ ] C2: `skip_symbol_robustness_penalty` has a comment explaining why it's always True for island-scoped runs.
- [ ] Existing tests pass: `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_island_scheduler.py -x -q`
- [ ] `evaluator_v5.ipynb` NOT modified.

## Verification

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_island_scheduler.py -x -q
```
