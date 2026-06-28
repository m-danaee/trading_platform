# Task-17: Island Migration & Rule Structure

**Branch:** `fix/island-migration-rule-structure`
**Priority:** 🟠 High
**Fixes:** H4, H6
**Depends on:** task-15 merged (config ordering)

## Problem

`PHASE2_MIGRATION_ENABLED = False` (config.py line 921) — the "island model" is fake. 3 cluster islands run fully independent NSGA-III searches with no elite exchange. Cluster_0 (easy, big, 194k rows) finds 21.99% returns that don't generalize; clusters 1&2 are starved of good genetic material. Additionally, `MIN_CONDITIONS=3, MAX_CONDITIONS=4` makes the symbol gene 25–33% of the rule, biasing toward symbol-locking.

## Files to Modify

1. `gpu_fuzzy_trader/config.py` — migration + conditions config
2. `gpu_fuzzy_trader/phases/phase2_island_scheduler.py` — startup logging
3. `gpu_fuzzy_trader/phases/phase2_rule_pool.py` — verify MIN/MAX_CONDITIONS usage (no change needed if already read from config)

## Detailed Changes

### H4: Enable migration (Option A — preferred)

**config.py line 921:**
```python
# BEFORE:
PHASE2_MIGRATION_ENABLED: bool = False
# AFTER:
PHASE2_MIGRATION_ENABLED: bool = True
```

**Verify the following migration params have reasonable values (set if missing):**
```python
PHASE2_MIGRATION_EPOCH_INTERVAL = 1      # migrate every epoch (15 gens)
PHASE2_MIGRATION_TOP_K = 5               # top 5 migrants per source
PHASE2_MIGRATION_MIN_VAL_RETURN_PCT = 1.0  # min val return to migrate
PHASE2_MIGRATION_MIN_VAL_TRADES = None    # None = use island trade floor
PHASE2_MIGRATION_SEED_FRACTION = 0.10     # fraction of pop seeded from migrants
PHASE2_MIGRATION_REQUIRE_DEPLOYABILITY = True
```

**phase2_island_scheduler.py — add startup log:**
- In `run_cluster_phase2` (after the K=3 log line):
  ```python
  logger.info(
      "Phase 2 [%s]: island mode migration=%s (interval=%d, top_k=%d, seed_frac=%.2f)",
      direction,
      "enabled" if _cfg.PHASE2_MIGRATION_ENABLED else "disabled (independent islands)",
      int(_cfg.PHASE2_MIGRATION_EPOCH_INTERVAL),
      int(_cfg.PHASE2_MIGRATION_TOP_K),
      float(_cfg.PHASE2_MIGRATION_SEED_FRACTION),
  )
  ```

**Fallback (Option B — only if GPU re-eval cost prohibitive):**
- If implementer discovers migration's `_migrant_to_metrics` re-evaluation on receiver cluster is too slow (profile: >30% of epoch time), set `PHASE2_N_CLUSTERS = 1` and short-circuit `_run_cluster_islands` to run a single island on the full symbol set. Document the choice in the task handoff.

### H6: Raise MIN/MAX_CONDITIONS

**config.py lines 405–406:**
```python
# BEFORE:
MIN_CONDITIONS = 3
MAX_CONDITIONS = 4
# AFTER:
MIN_CONDITIONS = 4
MAX_CONDITIONS = 5
```

**Verify** config assertion at line 2078 still passes:
```python
assert MIN_CONDITIONS <= MAX_CONDITIONS
```

**Verify** sparse encoding (`PHASE2_ENCODING = "sparse_slots"`) uses `MAX_CONDITIONS` for slot count — it should auto-pick up the new value. Check `phase2_sparse_encoding.py` for any hardcoded `4` that should be `MAX_CONDITIONS`.

## Acceptance Criteria

- [ ] `PHASE2_MIGRATION_ENABLED = True` OR documented `PHASE2_N_CLUSTERS = 1` fallback.
- [ ] Migration config params present and documented (epoch_interval, top_k, min_val_return, seed_fraction).
- [ ] Startup log line shows migration status.
- [ ] `MIN_CONDITIONS = 4`, `MAX_CONDITIONS = 5`.
- [ ] Config assertion passes.
- [ ] Sparse encoding picks up new MAX_CONDITIONS (no hardcoded `4`).
- [ ] `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_island*.py tests/unit/test_island_scheduler_migration.py -x -q` passes.
- [ ] No regressions in `test_phase2_island_hyperparams.py`.

## Verification

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/ -x -q -k "island or migration or cluster or conditions"
```
