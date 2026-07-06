# Task 5: Sequential cluster warmup (drop 3/4 signatures)

## Source plan
`.opencode/plans/PLAN.md` — Task 5

## Branch
`fix/ram-sequential-clusters` (from `main` after Task 4)

## Files to touch
- `gpu_fuzzy_trader/_gpu_runtime.py` — add `_evict_cluster_signatures()` helper
- `gpu_fuzzy_trader/phases/phase2_island_scheduler.py` — call the helper between clusters
- `gpu_fuzzy_trader/evolution/evox_runner.py` — may need to expose a per-cluster cache eviction
- `tests/unit/test_phase2_island_scheduler.py` — new test for the eviction path
- `tests/unit/test_evox_runner.py` (possibly) — test the helper

## Changes

### Background
Currently `_gpu_runtime.py:warmup_phase2_gpu_kernels` is called ONCE at startup with all 3 cluster engines + their val engines. The result: `signatures=4` stays alive for the entire run. JAX keeps the compiled programs in memory as Python objects + buffers (~3 GB across 4 signatures).

The fix: warm only the current cluster's engines, run it, evict its signatures, warm the next cluster. At any time, only 1 cluster's signatures are alive.

### Implementation

**1. New helper in `gpu_fuzzy_trader/_gpu_runtime.py`:**

```python
def evict_cluster_signatures(cluster_id: str | int | None = None) -> int:
    """Evict JAX compiled signatures for a completed cluster.
    
    Removes entries from ``_WARMED_SIGNATURES`` that belong to a specific
    cluster and tries to free the JAX-compiled programs. Returns the number
    of signatures evicted.
    
    Strategy:
    - If ``cluster_id`` is provided, evict only that cluster's signatures
    - If ``cluster_id`` is None, evict ALL signatures
    - Use ``jax.clear_caches()`` if available (JAX ≥ 0.4.x)
    - Fallback: best-effort GC + cache clear
    
    The signatures are tagged with cluster_id via the ``_warmup_signature``
    helper, which must be updated to include it (see below).
    """
    global _WARMED_SIGNATURES
    if cluster_id is not None:
        before = len(_WARMED_SIGNATURES)
        _WARMED_SIGNATURES = {
            sig for sig in _WARMED_SIGNATURES
            if not (isinstance(sig, tuple) and len(sig) >= 2 and sig[-1] == cluster_id)
        }
        evicted = before - len(_WARMED_SIGNATURES)
    else:
        evicted = len(_WARMED_SIGNATURES)
        _WARMED_SIGNATURES = set()
    
    if evicted > 0:
        try:
            import jax
            if hasattr(jax, "clear_caches"):
                jax.clear_caches()
        except Exception:
            pass
        import gc as _gc
        _gc.collect()
    
    return evicted
```

**2. Update `_warmup_signature()` to include `cluster_id` tag:**

The current signature is `(n_rows, n_features, batch_size)` (or similar — read the actual function). Append a cluster_id marker:
```python
def _warmup_signature(target, batch_size, cluster_id=None):
    base_sig = ...  # existing logic
    if cluster_id is not None:
        return base_sig + (cluster_id,)
    return base_sig
```

This way the eviction helper can filter by cluster.

**3. Update `warmup_phase2_gpu_kernels()` to accept `cluster_id`:**

```python
def warmup_phase2_gpu_kernels(
    engine: object,
    val_engine: object | None = None,
    cluster_id: str | int | None = None,
) -> None:
    """..."""
    batch_size = resolve_phase2_gpu_batch_size()
    targets = _iter_warmup_targets(engine, val_engine)
    if not targets:
        logger.warning("Phase 2 JAX warmup: no engines to warm")
        return

    warmed = 0
    skipped = 0
    for target in targets:
        sig = _warmup_signature(target, batch_size, cluster_id=cluster_id)
        if sig in _WARMED_SIGNATURES:
            skipped += 1
            continue
        _warmup_engine(target, batch_size=batch_size)
        warmed += 1

    used = detect_gpu_memory_used_gb()
    used_str = f"{used:.2f} GiB" if used is not None else "unknown"
    logger.info(
        "Phase 2 JAX warmup complete (%d engines warmed, %d skipped, "
        "batch_size=%d, gpu_used=%s, signatures=%d, cluster_id=%s)",
        warmed, skipped, batch_size, used_str,
        len(_WARMED_SIGNATURES), cluster_id,
    )
```

**4. Wire the per-cluster warmup in `phase2_island_scheduler.py:_run_cluster_islands`:**

The current flow:
1. Build all 3 cluster generators (each calls `_gpu_runtime.configure_phase2_gpu_runtime` or similar)
2. Run the while loop, processing each cluster per epoch

New flow (per-cluster warmup):
1. Build all 3 cluster generators **without** warming
2. For each cluster, in order:
   a. Warm this cluster's engines via `warmup_phase2_gpu_kernels(..., cluster_id=cid)`
   b. Run its epochs
   c. After all epochs done, `evict_cluster_signatures(cluster_id=cid)`
3. Continue to next cluster

This means:
- The `configure_phase2_gpu_runtime` call currently in `Rule_Pool_Generator.__init__` (or wherever) should accept a `cluster_id` parameter
- OR the warmup is moved out of generator init and into `_run_cluster_islands`

**The simpler refactor:** keep the warmup inside generator init, but tag signatures with cluster_id, and call `evict_cluster_signatures` at the end of each cluster's epochs in `_run_cluster_islands`.

Look at the `Rule_Pool_Generator` constructor or wherever `configure_phase2_gpu_runtime` is called from. It should pass `cluster_id` to the warmup.

### Required changes summary

1. `_gpu_runtime.py`:
   - Update `_warmup_signature` to accept and embed `cluster_id`
   - Add `evict_cluster_signatures(cluster_id=None)` helper
   - Update `warmup_phase2_gpu_kernels` to accept and log `cluster_id`

2. `phase2_rule_pool.py` (or wherever warmup is called from generator init):
   - Pass `cluster_id=self._cluster_id` (or similar) to `warmup_phase2_gpu_kernels` and `configure_phase2_gpu_runtime`

3. `phase2_island_scheduler.py:_run_cluster_islands`:
   - After each cluster's epochs are done and BEFORE `del generators[cid]` (the Task 4 fix), call `evict_cluster_signatures(cluster_id=cid)`
   - Log how many signatures were evicted

### Expected behavior after fix

Re-run log should show:
```
Phase 2 [long] cluster_0: JAX warmup complete (2 engines warmed, 0 skipped, ..., signatures=2, cluster_id=0)
Phase 2 [long] cluster_0: evicted 2 signatures
Phase 2 [long] cluster_1: JAX warmup complete (2 engines warmed, 0 skipped, ..., signatures=2, cluster_id=1)
Phase 2 [long] cluster_1: evicted 2 signatures
Phase 2 [long] cluster_2: JAX warmup complete (2 engines warmed, 0 skipped, ..., signatures=2, cluster_id=2)
Phase 2 [long] cluster_2: evicted 2 signatures
```

Instead of:
```
Phase 2 [long] cluster_0: JAX warmup complete (..., signatures=2)
Phase 2 [long] cluster_1: JAX warmup complete (..., signatures=4)
Phase 2 [long] cluster_2: JAX warmup complete (0 warmed, 2 skipped, ..., signatures=4)
```

## Acceptance criteria
- [ ] `_evict_cluster_signatures()` helper exists in `_gpu_runtime.py`
- [ ] `_warmup_signature()` embeds `cluster_id` in the signature tuple
- [ ] `warmup_phase2_gpu_kernels()` accepts and logs `cluster_id`
- [ ] `phase2_island_scheduler.py` calls `evict_cluster_signatures(cluster_id=cid)` after each cluster's epochs
- [ ] `Rule_Pool_Generator` (or whatever calls warmup from init) passes cluster_id
- [ ] New test in `test_phase2_island_scheduler.py` (or `test_evox_runner.py`) exercises `evict_cluster_signatures` and asserts `_WARMED_SIGNATURES` shrinks
- [ ] All touched test suites pass with `PYTEST_LOW_MEMORY=1`
- [ ] No regressions in existing overfit-gap behavior (Stages 1-3) or RAM quick wins (Task 4)

## Hard rules
- Do NOT change behavior of existing tests (they pass on `_WARMED_SIGNATURES` shape, may need updating)
- Do NOT change the `warmup_phase2_gpu_kernels` return type or existing call sites — only add a new optional parameter
- Do NOT push to remote, do NOT merge to main
- Use `.venv/bin/python` for any test command
- Use `PYTEST_LOW_MEMORY=1`
- Only run touched test suites, not full suite
- Commit message prefix: `fix(task-5): <item summary>`

## Verification command
```
cd /home/danaee/trading_platform
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_island_scheduler.py tests/unit/test_phase2_rule_pool.py tests/unit/test_evox_runner.py tests/unit/test_migration_safety.py tests/unit/test_island_scheduler_migration.py -v
```

## Implementation hints

### Read these files first
- `gpu_fuzzy_trader/_gpu_runtime.py` — the full warmup flow
- `gpu_fuzzy_trader/phases/phase2_rule_pool.py` — Rule_Pool_Generator.__init__ and where warmup is called
- `gpu_fuzzy_trader/phases/phase2_island_scheduler.py` — _run_cluster_islands (where eviction should be added)
- `gpu_fuzzy_trader/evolution/evox_runner.py` — confirm warmup is NOT called from evox_runner (it's only _gpu_runtime)

### Likely implementation order
1. Update `_warmup_signature` to add cluster_id
2. Add `evict_cluster_signatures` helper
3. Update `warmup_phase2_gpu_kernels` to accept cluster_id
4. Update `Rule_Pool_Generator.__init__` to pass cluster_id to warmup
5. Update `_run_cluster_islands` to call evict after each cluster
6. Add new test for the eviction

### Backward compatibility
- Old call sites of `warmup_phase2_gpu_kernels` (without cluster_id) should still work
- The signature tuple shape changes — but signatures are internal (in `_WARMED_SIGNATURES` set), not exposed

## Important risks
- Changing signature tuple shape may break existing tests that check signature count
- `jax.clear_caches()` may not exist in older JAX versions (check version)
- Per-cluster recompile cost: each cluster pays ~60s JIT compile time (per the log timing). Total overhead: ~3 min for K=3 clusters. Acceptable trade.
