# Task 1 — `fix/plateau-state-leak` (Fixes A + B) — CRITICAL

## Branch
`fix/plateau-state-leak` (from latest `main`).

## Problem
Plateau state (`plateau_best_progress`, `plateau_streak`) leaks across island
epochs via `Phase2EvolutionState`. Because `PHASE2_ISLAND_TWO_STAGE_ENABLED=False`,
`reset_plateau = entering_stage_b` is always `False`, so `plateau_streak` carries
5–17 into epoch 2+ and every epoch dies at `min_gen=3`. Separately, the island
generation budget charges *requested* gens (10) not *actual* gens run (3).

## Required Changes

### Fix A — Reset plateau per epoch
**File:** `gpu_fuzzy_trader/phases/phase2_rule_pool.py` (around line 2672)

Current:
```python
reset_plateau = entering_stage_b
```
Change to: always reset plateau at the start of each epoch (each epoch is a fresh
plateau context; migrants/seeds may improve from a new starting point). Keep the
`entering_stage_b` seeding logic below it intact.

```python
# Each island epoch is a fresh plateau context: carrying the prior epoch's
# best/streak makes every subsequent epoch instantly plateau at min_gen.
reset_plateau = True
```
If a future two-stage island mode is enabled, the Stage A/B transition should
still reset (the `reset_plateau=True` already covers it; `entering_stage_b`
seeding remains separate). Add a short comment noting that a global
"island fully converged" kill-switch, if ever needed, should be a separate
counter — NOT the per-epoch streak.

### Fix B — Charge actual generations to the budget
**File:** `gpu_fuzzy_trader/phases/phase2_rule_pool.py` (around line 2765)

Current:
```python
self._island_generations_done += epoch_gens
```
Change to charge the number of generations actually executed
(`epoch_history` has one entry per gen run; the evolution loop `break`s on
early-stop):
```python
self._island_generations_done += len(epoch_history)
```
`epoch_history` is already returned by `run_phase2_evolution_epoch` and is in
scope at that line. Verify the variable name in the enclosing `run_epoch`
method (it is assigned from the return tuple).

## Acceptance Criteria
1. `plateau_streak` at gen 1 of epoch 2 is `0` when no improvement threshold is
   crossed *within epoch 2* (regression test).
2. `_island_generations_done` increments by the number of generations actually
   executed, not the requested `epoch_gens` (regression test simulating an
   early-stop at gen 3 of a 10-gen epoch → budget += 3).
3. `reset_plateau` is `True` for every epoch regardless of two-stage config.
4. No behavioral change to Stage B seeding or migration logic.
5. `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit -q` passes.
6. Prior migration/elite/early-stop tests still pass.

## Target Files
- `gpu_fuzzy_trader/phases/phase2_rule_pool.py`
- `tests/unit/test_island_early_stop.py` (add 2-3 regression tests) — or a new
  `tests/unit/test_plateau_state_leak.py` if cleaner.

## Verification
```
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_island_early_stop.py tests/unit/test_plateau_state_leak.py -q
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit -q
```
Do NOT run the full pipeline.

## Notes
- Do not touch `evox_runner.py` state-restore logic beyond what Fix A requires;
  the `reset_plateau` block at `evox_runner.py:1829-1831` already resets both
  `plateau_best_progress` and `plateau_streak` when `reset_plateau=True`.
- Clean up any now-dead code paths introduced (per AGENTS.md).
