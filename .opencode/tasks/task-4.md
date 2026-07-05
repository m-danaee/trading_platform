# Task 4: RAM quick wins (3 fixes)

## Source plan
`.opencode/plans/PLAN.md` — Task 4

## Branch
`fix/ram-quick-wins` (from `main`)

## Files to touch
- `gpu_fuzzy_trader/config.py` (Fix 1: cache size)
- `gpu_fuzzy_trader/phases/phase2_island_scheduler.py` (Fix 2: cache clear + Fix 3: engine teardown)
- `gpu_fuzzy_trader/evolution/evox_runner.py` (Fix 4: more frequent gc)

## Changes

### Fix 1: Halve global metrics cache
**File:** `gpu_fuzzy_trader/config.py:413`
**Change:** `PHASE2_EVAL_GLOBAL_CACHE_MAX_SIZE = 1200` → `PHASE2_EVAL_GLOBAL_CACHE_MAX_SIZE = 600`
**Justification:** Cache hit rate observed at 0-4% in the 2026-07-05 22:09 log. 50% size reduction barely affects hit rate.
**Expected saving:** ~0.4 GB

### Fix 2: Clear global metrics cache between clusters
**File:** `gpu_fuzzy_trader/phases/phase2_island_scheduler.py`
**Location:** Inside `_run_cluster_islands`, after the outer `while` loop completes and before orphan-boost / pool merge (around line 502).

Add a clear-cache block. The cache is `global_metrics_cache` in `evox_runner.py` (module-level). Approach:
- Import `_trim_global_metrics_cache` or expose a public clear helper from `evox_runner.py`
- Or add a public helper `clear_global_metrics_cache()` to `evox_runner.py` that does `global_metrics_cache.clear(); gc.collect()`
- Then call it after each cluster's epochs complete

**Recommended approach (minimize API change):**
- Add a new public function `clear_global_metrics_cache()` in `gpu_fuzzy_trader/evolution/evox_runner.py` that does:
  ```python
  def clear_global_metrics_cache() -> None:
      """Clear the global eval cache. Used to free RAM between cluster runs."""
      global global_metrics_cache
      if "global_metrics_cache" in globals():
          globals()["global_metrics_cache"].clear()
      import gc as _gc
      _gc.collect()
  ```
- In `phase2_island_scheduler.py:_run_cluster_islands`, import and call it after the outer while loop, before the cluster_pools merge block

**Expected saving:** ~0.4-0.8 GB (cache size is already 600 after Fix 1, but clearing it stops it from growing)

### Fix 3: Explicit engine teardown after each cluster
**File:** `gpu_fuzzy_trader/phases/phase2_island_scheduler.py:_run_cluster_islands`
**Location:** The for-loop at line ~498 that builds `cluster_pools`:
```python
for cid, gen in generators.items():
    pool_part = gen.finalize_island()
    annotated = Rule_Pool_Generator._annotate_archive_entries(...)
    cluster_pools.extend(annotated)
```

**Change:** After collecting `pool_part` and `annotated` for a cluster, explicitly delete the generator and call `gc.collect()` to free DataFrame + JAX wrapper memory. The generator's pool data has been copied into `cluster_pools`, so the original can be released.

```python
for cid, gen in generators.items():
    pool_part = gen.finalize_island()
    annotated = Rule_Pool_Generator._annotate_archive_entries(
        pool_part,
        source_symbols=cluster_map.get(cid, []),
    )
    cluster_pools.extend(annotated)
    # Release the generator's dataframes and JAX wrappers (saves ~0.5 GB/cluster)
    import gc as _gc
    del generators[cid]
    _gc.collect()
```

**Expected saving:** ~0.5 GB per cluster released early. At any time, only 1-2 clusters' data is in memory instead of 3.

**Important:** Verify that `finalize_island()` returns a fully independent copy of the pool data (no references back to the generator's internal DataFrames). Read `finalize_island` in `phase2_rule_pool.py` to confirm.

### Fix 4: More frequent `gc.collect()`
**File:** `gpu_fuzzy_trader/evolution/evox_runner.py:2777`
**Current:**
```python
if gen % 10 == 0 and gen > 0:
    import gc as _gc
    _gc.collect()
```
**Change:** `gen % 10 == 0` → `gen % 3 == 0`
**Justification:** Colab 12 GB RAM is tight; gen % 10 lets fragmentation accumulate. gc.collect() is cheap (sub-second) when there's not much to free.
**Expected saving:** ~0.3-0.5 GB (mostly fragmentation recovery, hard to predict)

## Acceptance criteria
- [ ] `PHASE2_EVAL_GLOBAL_CACHE_MAX_SIZE = 600` in `config.py:413`
- [ ] New `clear_global_metrics_cache()` function exists in `evox_runner.py` and is callable
- [ ] `_run_cluster_islands` calls `clear_global_metrics_cache()` between cluster epochs and pool merge
- [ ] `_run_cluster_islands` calls `del generators[cid]; gc.collect()` after each cluster's `finalize_island()`
- [ ] `gc.collect()` in `evox_runner.py:2777` runs every 3 generations (gen % 3 == 0)
- [ ] All touched test suites pass with `PYTEST_LOW_MEMORY=1`:
  ```
  cd /home/danaee/trading_platform
  PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_island_scheduler.py tests/unit/test_phase2_rule_pool.py tests/unit/test_evox_runner.py tests/unit/test_migration_safety.py tests/unit/test_island_scheduler_migration.py -v
  ```

## Hard rules
- Do NOT change behavior. All 4 fixes are pure memory-management changes.
- Do NOT change Stages 1-3 fixes (overfit-gap behavior must remain).
- Do NOT push to remote, do NOT merge to main.
- Use `.venv/bin/python` for any test command.
- Use `PYTEST_LOW_MEMORY=1`.
- Only run touched test suites, not full suite (OOM risk per AGENTS.md).
- Commit message prefix: `fix(task-4): <fix summary>`. Single consolidated commit is OK (4 tightly coupled changes).

## Verification command
```
cd /home/danaee/trading_platform
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_island_scheduler.py tests/unit/test_phase2_rule_pool.py tests/unit/test_evox_runner.py tests/unit/test_migration_safety.py tests/unit/test_island_scheduler_migration.py -v
```

## Implementation hints

### For Fix 2 — verify `finalize_island` returns independent data
Read `gpu_fuzzy_trader/phases/phase2_rule_pool.py` and find the `finalize_island` method. Verify that it returns a list of dicts (pool entries) that don't reference internal DataFrames. If they do, you'll need to deep-copy the pool data before deleting the generator.

```python
# In _run_cluster_islands, the safe pattern:
for cid, gen in generators.items():
    pool_part = gen.finalize_island()
    annotated = Rule_Pool_Generator._annotate_archive_entries(...)
    cluster_pools.extend(annotated)
    del generators[cid]
    import gc as _gc
    _gc.collect()
```

If `pool_part` contains references to `gen._train_df` or `gen._val_df`, the `del` won't actually free the memory. You may need:
```python
pool_part = list(gen.finalize_island())  # materialize as list
# Or deep-copy if the entries are dicts referencing DataFrames
```

Test by reading the `finalize_island` code, not by running the project (OOM risk per AGENTS.md).
