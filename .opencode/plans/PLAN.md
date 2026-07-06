# Plan: Phase 2 Island/NSGA-III Robustness Fixes

## Goal
Improve Phase 2 island evolution reliability and OOS-focused optimization after the latest run-log analysis. Address misleading island epoch state, objective collapse, sampling/validation leakage risk, and migration/config mismatches without changing `evaluator_v5.ipynb` or running the full project locally.

## Constraints
- Use `.venv` for commands.
- Do not run the full pipeline locally; it can OOM.
- Do not modify `evaluator_v5.ipynb`.
- Run only related tests with `PYTEST_LOW_MEMORY=1`.
- Branch policy: isolated feature branch per task from `main`.
- Execution mode: checkpoint; stop after each completed task until user says `continue task N`.

## Task 1: Clarify Island Epoch State And Reset Recovery Counters

### Target Files
- `gpu_fuzzy_trader/evolution/evox_runner.py`
- `gpu_fuzzy_trader/phases/phase2_rule_pool.py`
- Related tests under `tests/unit/` for EvoX runner / Phase 2 rule pool.

### Scope
- Add explicit support for resetting per-epoch recovery counters when resuming an island state.
- Reset or isolate `restart_count`, `post_restart_gens_remaining`, `post_restart_no_improve_streak`, and `post_restart_best_progress` at island epoch boundaries when requested by `Rule_Pool_Generator.run_epoch()`.
- Keep population/archive continuity unless explicitly reinitializing; do not silently change the island's search continuity model.
- Improve logging so resumed island epochs are not mistaken for fresh independent runs. Include `generation_offset` or total generation progress in generation logs/history where minimally invasive.

### Acceptance Criteria
- A resumed island epoch no longer starts with stale `restarts=N/3` from prior epochs when `reset_plateau=True` is used by island scheduling.
- Existing global/non-island evolution behavior remains unchanged.
- Tests cover resumed state with nonzero restart/post-restart counters and verify counters reset only when requested.
- Related tests pass with `PYTEST_LOW_MEMORY=1` and `.venv/bin/python`.

### Verification
- `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_evox_runner.py -q`

## Task 2: Recompute Resumed Objectives At Safe Epoch Boundaries

### Target Files
- `gpu_fuzzy_trader/evolution/evox_runner.py`
- `gpu_fuzzy_trader/phases/phase2_rule_pool.py`
- Related tests under `tests/unit/`.

### Scope
- Add an explicit `refresh_objectives_on_resume` or similar flag for island epochs.
- When an epoch resumes with changed epoch context, set live objectives to `inf` so parent population is re-evaluated with current validation/diversity/cache context.
- Trigger refresh when migrants are applied, entering Stage B, or an island epoch starts after parked/rebuilt engines, while avoiding unnecessary global-mode recomputation.
- Ensure metrics cache consistency after objective refresh.

### Acceptance Criteria
- Migrant-seeded resumed epochs do not reuse stale objective values for overwritten or context-sensitive chromosomes.
- Existing fresh-run behavior is unchanged.
- Tests verify objective refresh occurs only when requested and does not clear population/archives.

### Verification
- `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_evox_runner.py tests/unit/test_phase2_rule_pool.py -q`

## Task 3: Decouple Phase 2 Objectives And Prefer Robust Return For OOS

### Target Files
- `gpu_fuzzy_trader/phases/phase2_rule_pool.py`
- `gpu_fuzzy_trader/phases/phase2_support.py`
- `gpu_fuzzy_trader/config.py`
- Related tests under `tests/unit/test_phase2_rule_pool.py` and `tests/unit/test_phase2_support.py`.

### Scope
- Reduce shared penalty dominance across `f1`, `f2`, and `f3` so objective correlations do not trivially collapse to +/-1.
- Prefer robust return as the default f3 signal for OOS when joint train+val metrics exist, while keeping PF as a quality gate/penalty.
- Preserve existing gates unless tests and config comments are updated to reflect a deliberate change.
- Add diagnostics or metrics that expose penalty component contributions if already compatible with the logging patterns.

### Acceptance Criteria
- Objective calculation tests show penalties are not identically added to all objectives.
- f3 can use robust return in the configured OOS-focused mode.
- PF floor still affects feasibility/admission behavior.
- No evaluator notebook changes.

### Verification
- `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_rule_pool.py tests/unit/test_phase2_support.py -q`

## Task 4: Make Sampling And Migration Semantics Honest

### Target Files
- `gpu_fuzzy_trader/phases/phase2_rule_pool.py`
- `gpu_fuzzy_trader/phases/phase2_island_scheduler.py`
- `gpu_fuzzy_trader/config.py`
- Related tests under `tests/unit/` or `tests/property/` if existing.

### Scope
- Update logging/config comments so current sequential post-cluster migration is not described as epoch-interval migration, or wire the existing interval helper into a true round-robin epoch scheduler if small enough.
- Make train and validation sampling seeds distinct but deterministic.
- Add/prepare timestamp-aligned contiguous sampling safeguards where possible without broad engine changes.
- Ensure `_sample_df()` warnings remain useful and do not hide no-op sampling.

### Acceptance Criteria
- Logs accurately describe whether migration is post-cluster chain migration or epoch-round migration.
- Train and validation sampling windows are not selected with the identical RNG seed by default.
- Existing contiguous per-symbol sampling behavior remains deterministic.
- Related sampling/scheduler tests pass.

### Verification
- `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_rule_pool.py tests/unit/test_phase2_island_scheduler.py -q`
