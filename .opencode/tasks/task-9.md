# Task 9: Remove Dead Migration Helper + Deprecate Interval Config

## Task ID
`task-9` (ninth of 12 tasks in the 2026-07-07 audit fix plan)

## Title
Remove Dead Migration Helper + Deprecate Interval Config

## Goal
Fix audit finding #6: `_should_migrate_this_round` is dead code
(marked "legacy/unused" in its own docstring, never called from
`_run_cluster_islands`). The migration log line at the top of the
cluster section is already correct ("enabled sequential
post-cluster chain"), but the dead helper at line 412 is a
maintenance trap. Also: `PHASE2_MIGRATION_EPOCH_INTERVAL` is a
no-op config retained for backward compat — its docstring
already says so, but the field's continued existence invites
confusion.

## Audit Citation
- Confirmed by static inspection:
  - `phases/phase2_island_scheduler.py:412-431` — `_should_migrate_this_round`
    helper. Its own docstring says: "This helper is **legacy/unused**
    in the active sequential scheduler."
  - `grep _should_migrate_this_round` returns 1 hit (the definition
    only; no callers).
  - `phases/phase2_island_scheduler.py:675` — log line already says
    "enabled sequential post-cluster chain" (correct).
  - `config.py:1206-1210` — `PHASE2_MIGRATION_EPOCH_INTERVAL` is
    already docstringed as a no-op.
- The migration is actually a 1-shot sequential chain (cluster N → N+1
  after cluster N finishes), not round-robin.

## Target Files
- `gpu_fuzzy_trader/phases/phase2_island_scheduler.py`
  - Delete `_should_migrate_this_round` (lines 412-431).
  - Update the docstring of `_run_cluster_islands` to remove any
    references to the deleted helper (if any).
- `gpu_fuzzy_trader/config.py`
  - Add a `# DEPRECATED` prefix to the `PHASE2_MIGRATION_EPOCH_INTERVAL`
    docstring/comment to make the no-op status more obvious.
  - Optionally: rename the config flag to `PHASE2_MIGRATION_EPOCH_INTERVAL_DEPRECATED`
    — but this is a breaking change for any external consumer, so
    prefer keeping the name and just improving the docstring.
- `tests/unit/test_phase2_island_scheduler.py`
  - The `test_phase2_migration_gate.py` test currently exists
    (315B file). Verify it still passes after the deletion.
  - Add a regression test that confirms the deleted helper is
    no longer importable from `phase2_island_scheduler`.

## Current Behavior
- `phases/phase2_island_scheduler.py:412-431`: defines
  `_should_migrate_this_round(round_index, interval) -> bool`.
  Never called. Its own docstring admits it.
- `phases/phase2_island_scheduler.py:675`: logs "island mode
  migration=enabled sequential post-cluster chain" — already
  accurate.
- `config.py:1206-1210`: `PHASE2_MIGRATION_EPOCH_INTERVAL = 1`
  with a docstring that already says "no-op in the current code path".

## Scope
1. **Delete the dead helper** (`_should_migrate_this_round`):
   - Remove lines 412-431 (the function definition).
   - Verify no callers exist (grep the codebase).
   - Verify no test imports it (grep `tests/`).
   - If a test does import it, update the test to import something
     else or remove the test.
2. **Improve the config docstring** (config.py:1206-1210):
   - Add a more prominent DEPRECATED marker:
     ```python
     # PHASE2_MIGRATION_EPOCH_INTERVAL — DEPRECATED.  This is a no-op in
     # the current code path.  Migration fires once after each cluster
     # finishes (sequential post-cluster chain).  Retained for backward
     # compat only; do not use in new code.  See task-9 in
     # .opencode/plans/PLAN.md for context.
     PHASE2_MIGRATION_EPOCH_INTERVAL = 1
     ```
3. **Add audit-finding linkage**:
   - Add `# → fixes audit finding #6 (dead migration helper;
     sequential chain is already correctly named in logs)` in the
     config.py comment block.
4. **Add regression test**:
   - In `tests/unit/test_phase2_island_scheduler.py` (or a new file),
     add a test that asserts the deleted helper is no longer
     importable:
     ```python
     def test_should_migrate_this_round_removed():
         with pytest.raises(ImportError):
             from gpu_fuzzy_trader.phases.phase2_island_scheduler import (
                 _should_migrate_this_round,
             )
     ```
5. **Do NOT change**:
   - The actual migration logic in `_run_cluster_islands` (lines
     around 540-580). It's already correct (sequential post-cluster
     chain).
   - The log line at line 675 ("island mode migration=enabled
     sequential post-cluster chain"). Already correct.
   - Any other file outside `phases/phase2_island_scheduler.py`,
     `config.py`, and the test file.

## Acceptance Criteria
1. `_should_migrate_this_round` is deleted from
   `phases/phase2_island_scheduler.py`.
2. `grep _should_migrate_this_round gpu_fuzzy_trader/` returns 0 matches.
3. `from gpu_fuzzy_trader.phases.phase2_island_scheduler import _should_migrate_this_round`
   raises `ImportError` (verified by regression test).
4. `PHASE2_MIGRATION_EPOCH_INTERVAL` config docstring prominently
   says DEPRECATED.
5. The migration log line at line ~675 is unchanged ("enabled
   sequential post-cluster chain").
6. The actual migration logic (cluster N → N+1 chain) is unchanged.
7. All existing tests pass: `test_phase2_island_scheduler.py`,
   `test_phase2_migration_gate.py`, `test_phase2_window_rotation.py`,
   `test_phase2_rule_pool.py`, etc.

## Verification
Run only related unit tests with `PYTEST_LOW_MEMORY=1` and `.venv`:

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_island_scheduler.py -q
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_migration_gate.py -q
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_window_rotation.py -q
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_rule_pool.py -q
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_monthly_admission.py -q
```

Also confirm:
```bash
grep -rn _should_migrate_this_round gpu_fuzzy_trader/ tests/
# Should return 0 matches
```

## Notes
- Do NOT modify `evaluator_v5.ipynb`.
- Do NOT run the full project or full test suite locally (OOM risk
  per AGENTS.md).
- This is the smallest task in the plan (~20 lines of source code
  removed, ~5 lines docstring improved, ~10 lines in 1 test file).
- The actual migration behavior is unchanged (sequential post-cluster
  chain). Only the dead helper is removed.
- This task is purely code hygiene; no behavior change.
