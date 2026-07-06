# Task 1: Clarify Island Epoch State And Reset Recovery Counters

## Task ID
`task-1`

## Title
Clarify Island Epoch State And Reset Recovery Counters

## Goal
Prevent island epochs from inheriting stale recovery counters (`restart_count`, post-restart counters) while preserving intentional population/archive continuity. Make resumed epoch behavior clearer in logs/history so run logs do not look like fresh independent runs when they are resumed epochs.

## Target Files
- `gpu_fuzzy_trader/evolution/evox_runner.py`
- `gpu_fuzzy_trader/phases/phase2_rule_pool.py`
- Related tests under `tests/unit/`, especially `tests/unit/test_evox_runner.py` and/or Phase 2 rule-pool tests.

## Current Evidence From Run Log
- Multiple island epochs start at `gen 1/13` but retain metrics from the previous epoch.
- Later epochs begin with `restarts=1/3`, `2/3`, or `3/3`, which means recovery capacity is depleted across epoch boundaries.
- `Rule_Pool_Generator.run_epoch()` sets `reset_plateau=True`, but `_run_nsga3()` only resets plateau fields and leaves restart/post-restart counters restored from `Phase2EvolutionState`.

## Scope
- Add explicit support for resetting per-epoch recovery counters when resuming an island state.
- Reset or isolate these counters when requested:
  - `restart_count`
  - `post_restart_gens_remaining`
  - `post_restart_no_improve_streak`
  - `post_restart_best_progress`
- Keep population, objectives, hall of fame, deployable archive, and global metrics cache continuity unless explicitly needed for the counter reset.
- Improve logging/history minimally to make resumed island epochs distinguishable from true fresh runs. Prefer adding existing `generation_offset`/global generation context rather than broad logging rewrites.
- Do not change global/non-island behavior unless the new option is explicitly enabled.

## Acceptance Criteria
- A resumed island epoch no longer starts with stale `restarts=N/3` from prior epochs when `reset_plateau=True` is used by island scheduling.
- Global/non-island evolution behavior remains unchanged.
- Tests cover resumed state with nonzero restart/post-restart counters and verify counters reset only when requested.
- Related tests pass with `.venv` and `PYTEST_LOW_MEMORY=1`.

## Verification
Run only related tests, for example:

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_evox_runner.py -q
```

If a narrower existing test target is more appropriate, use that and report it.

## Notes
- Do not modify `evaluator_v5.ipynb`.
- Do not run the full project or full test suite locally.
- Keep changes minimal and focused on Task 1; later tasks will handle objective refresh, objective decoupling, and sampling/migration semantics.
