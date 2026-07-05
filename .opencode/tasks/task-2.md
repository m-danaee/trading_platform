# Task 2: Stage 2 — Seed collision + migration cadence bugs

## Source plan
`/home/danaee/.claude/plans/you-are-a-senior-pure-cupcake.md` — Stage 2, items 6-7

## Branch
`fix/phase2-stage2-seed-migration` (from `main`)

## Files to touch
- `gpu_fuzzy_trader/phases/phase2_island_scheduler.py` (both fixes)
- `tests/unit/test_phase2_island_scheduler.py` (new tests)
- `tests/unit/test_island_scheduler_migration.py` (possibly — keep existing tests passing)
- `tests/unit/test_migration_safety.py` (keep passing)

## Changes

### Item 6: Long/short seed collision
- `_run_cluster_islands` (~line 380 in `phase2_island_scheduler.py`): change `_derive_island_seed(seed, cid)` → `_derive_island_seed(seed, f"{direction}_{cid}")`. The `direction` variable is already in scope (function parameter).
- Orphan-boost path (~line 296): change `_derive_island_seed(seed, f"orphan_{sym}")` → `_derive_island_seed(seed, f"{direction}_orphan_{sym}")`. The `direction` variable is already in scope (loop variable from the outer `_run_cluster_phase2` flow).
- DO NOT change the `_derive_island_seed` signature — the plan explicitly says it doesn't need to change. Direction is included in the `island_id` string at the call sites.
- New test in `tests/unit/test_phase2_island_scheduler.py`: assert that `_derive_island_seed(seed, "long_0") != _derive_island_seed(seed, "short_0")` (and similar for orphan path).

### Item 7: Migration cadence bug
- The `epoch_counter` currently increments once per cluster inside the `for cid in cluster_ids:` loop, but the modulo gate is checked once per outer `while` round. This is invisible only because `interval=1` makes any `x % 1 == 0` trivially true.
- Extract a small pure helper, e.g.:
  ```python
  def _should_migrate_this_round(round_index: int, interval: int) -> bool:
      """True if migration should fire on this outer round (1-indexed)."""
      if interval <= 0:
          return False
      return round_index % interval == 0
  ```
  Use this helper inside `_run_cluster_islands` (replace the inline `epoch_counter % int(_cfg.PHASE2_MIGRATION_EPOCH_INTERVAL) == 0`).
- Move the `epoch_counter += 1` outside the `for cid in cluster_ids:` loop, so it increments once per outer round (not once per cluster). Rename to `round_counter` for clarity.
- New test: drive `_run_cluster_islands` (or just `_should_migrate_this_round` directly) with `n_clusters=3`, `interval=2` across several rounds, and assert migration fires on round cadence (every 2 rounds), not every 2 cluster-epoch calls. The test should:
  - Either call `_should_migrate_this_round(0, 2) is True`, `_should_migrate_this_round(1, 2) is False`, `_should_migrate_this_round(2, 2) is True` ...
  - Or run `_run_cluster_islands` with a stub generator that records the guard value at each round, and verify it's True every 2 rounds (not every 2 cluster epochs, which would fire ~3x too often with n_clusters=3).
- The existing tests in `test_migration_safety.py::TestMigrationEnabledByDefault` and `test_island_scheduler_migration.py::TestMigrationGuard` test the formula in isolation — they hand-set `epoch_counter = 2` and check the modulo arithmetic. They won't catch this regression either way, so the new test is purely additive.

## Acceptance criteria
- [ ] `_derive_island_seed` is called with `direction` in the `island_id` at both call sites (cluster and orphan)
- [ ] New test in `test_phase2_island_scheduler.py` asserts `_derive_island_seed(seed, "long_0") != _derive_island_seed(seed, "short_0")`
- [ ] New test in `test_phase2_island_scheduler.py` asserts orphan path uniqueness across directions
- [ ] `_should_migrate_this_round(round_index, interval)` helper exists in `phase2_island_scheduler.py`
- [ ] `_run_cluster_islands` uses the helper; the round counter increments once per outer `while` round (not per cluster)
- [ ] New test asserts migration fires on round cadence, not cluster-epoch cadence (with `n_clusters=3`, `interval=2`)
- [ ] All existing tests in `test_migration_safety.py` and `test_island_scheduler_migration.py` still pass
- [ ] All touched test suites pass with `PYTEST_LOW_MEMORY=1`

## Hard rules
- Do NOT change the `_derive_island_seed` signature (the plan explicitly says it doesn't need to change).
- Do NOT push to remote, do NOT merge to main.
- Use `.venv/bin/python` for any test command.
- Use `PYTEST_LOW_MEMORY=1`.
- Only run touched test suites, not full suite (OOM risk per AGENTS.md).
- Commit message prefix: `fix(task-2): <item summary>`. One commit per item, or one consolidated commit if tightly coupled.

## Verification command
```
cd /home/danaee/trading_platform
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_island_scheduler.py tests/unit/test_island_scheduler_migration.py tests/unit/test_migration_safety.py tests/unit/test_phase2_rule_pool.py -v
```
