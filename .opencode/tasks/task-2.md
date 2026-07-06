# Task 2: Recompute Resumed Objectives At Safe Epoch Boundaries

## Task ID
`task-2`

## Title
Recompute Resumed Objectives At Safe Epoch Boundaries

## Goal
Prevent resumed island epochs from reusing stale objective values when epoch context changes. The population/archive may continue, but objective values should be explicitly refreshed when context-sensitive inputs such as migrants, Stage B entry, rebuilt engines, validation sidecars, or diversity references can change fitness.

## Target Files
- `gpu_fuzzy_trader/evolution/evox_runner.py`
- `gpu_fuzzy_trader/phases/phase2_rule_pool.py`
- Related tests under `tests/unit/`, especially `tests/unit/test_evox_runner.py` and/or `tests/unit/test_phase2_rule_pool.py`.

## Evidence From Run Log / Task 1 Analysis
- Island epochs resume the prior `Phase2EvolutionState` and can start with identical Pareto metrics at `gen 1/13` after prior `gen 13/13`.
- Task 1 reset recovery counters, but objectives/metrics remain resumed unless a chromosome slot was overwritten or marked `inf`.
- Diversity references, validation freshness, migration seeds, and rebuilt engines can make old objectives stale even when chromosomes are unchanged.

## Scope
- Add an explicit option such as `refresh_objectives_on_resume` to the runner epoch path.
- When enabled and `state is not None`, set live `objectives` to `inf` and clear/recompute associated live `metrics_cache` entries as needed so the first generation re-evaluates the current population under current context.
- Wire this option from `Rule_Pool_Generator.run_epoch()` for safe island boundaries.
- Trigger/enable refresh at least when:
  - migrants are applied to an existing state,
  - entering Stage B,
  - an island epoch starts after prior parked/rebuilt engines or context may have changed.
- Preserve population, hall of fame, deployable archive, and global metrics cache unless there is a clearly necessary targeted invalidation.
- Avoid changing fresh-run behavior.
- Avoid unnecessary global-mode recomputation.

## Acceptance Criteria
- Migrant-seeded resumed epochs do not reuse stale objective values for overwritten or context-sensitive chromosomes.
- Resumed island epochs can force a full live-population objective refresh without clearing population/archives.
- Existing fresh-run behavior is unchanged.
- Tests verify objective refresh occurs only when requested and does not clear population/archives.
- Related tests pass with `.venv` and `PYTEST_LOW_MEMORY=1`.

## Verification
Run only related tests, for example:

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_evox_runner.py tests/unit/test_phase2_rule_pool.py -q
```

If narrower related tests are sufficient, use them and report exactly what ran.

## Notes
- Do not modify `evaluator_v5.ipynb`.
- Do not run the full project or full test suite locally.
- Keep changes focused on objective refresh/state consistency only. Objective decoupling and sampling/migration semantics are later tasks.
