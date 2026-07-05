# Plan: Phase 2 RAM Optimization (avoid Colab OOM)

**Created:** 2026-07-05
**Status:** active
**base_branch:** `main`
**branch_policy:** isolated
**execution_mode:** checkpoint

## Goal

Bring Phase 2 RAM usage from **9.9 / 12.7 GB → ~6-7 GB** on Colab so the run completes without OOM, while preserving correctness and not regressing the Stages 1-3 overfit fixes.

## Background

Post-Stages 1-3 re-run (2026-07-05 22:09) reached 9.9 / 12.7 GB RAM at gen 6 and is at risk of OOM. RAM breakdown:

| Consumer | Est. RAM | File:line |
|---|---|---|
| 4× JAX compiled programs (kept alive entire run) | ~3.0 GB | `_gpu_runtime.py:286-303` |
| 3× cluster DataFrames held in `generators` dict | ~1.5 GB | `phase2_island_scheduler.py:444-502` |
| Global metrics cache (size 1200, never cleared) | ~0.8 GB | `config.py:413`, `evox_runner.py:1476` |
| pandas train+val base + per-symbol slices | ~1.2 GB | loaders, splitters |
| Population state, archives, Python objects | ~1.5 GB | evox_runner state |
| JAX pre-allocation, traces, misc | ~1.9 GB | compiled artifacts |
| **Total observed** | **~9.9 GB** | matches log |

## Tasks

### Task 4: RAM quick wins (3 low-risk fixes)
**Branch:** `fix/ram-quick-wins` (from `main`)
**Risk:** Low (config + cleanup, no behavior change)
**Est. savings:** 1.3-2.0 GB

Changes:
1. **Halve global metrics cache** in `config.py:413` — `PHASE2_EVAL_GLOBAL_CACHE_MAX_SIZE = 600` (was 1200). Cache hit rate was 0-4% so the 50% size cut barely affects it.
2. **Clear global metrics cache between clusters** in `phase2_island_scheduler.py` — after the outer `while` loop in `_run_cluster_islands`, clear the cache before orphan-boost and pool merge.
3. **Explicit engine teardown after each cluster** in `phase2_island_scheduler.py` — `del generators[cid]; gc.collect()` after `gen.finalize_island()` for each cluster. The `park_engines()` at line 460 only releases GPU buffers; Python objects (DataFrames, JAX wrappers) stay alive.
4. **More frequent `gc.collect()`** in `evox_runner.py:2777` — change `gen % 10 == 0` → `gen % 3 == 0` to combat Colab fragmentation.

**Acceptance criteria:**
- `PHASE2_EVAL_GLOBAL_CACHE_MAX_SIZE = 600` in `config.py:413`
- Global metrics cache is cleared between clusters (verified by inspecting code + log statement at the clear site)
- `_run_cluster_islands` calls `del generators[cid]; gc.collect()` after `finalize_island()` for each cluster
- `gc.collect()` runs every 3 generations in `evox_runner.py:2777`
- All touched test suites pass with `PYTEST_LOW_MEMORY=1`
- No regressions in existing overfit-gap behavior (Stages 1-3 still working)

---

### Task 5: Sequential cluster warmup (drop 3 of 4 JAX signatures)
**Branch:** `fix/ram-sequential-clusters` (from `main` after Task 4)
**Risk:** Medium (changes warmup flow, may add ~60s per cluster for JIT recompile)
**Est. additional savings:** 1.5-2.0 GB (at peak; at any moment only 1-2 signatures alive)

Current behavior (per the log): `_gpu_runtime.py:286-303` warms all engines for all clusters upfront → `signatures=4` stays alive for the entire run.

New behavior: warm cluster_0's engines, run its epochs, teardown, warm cluster_1's engines, etc. Only the active cluster's signatures are alive.

**Files to touch:**
- `gpu_fuzzy_trader/_gpu_runtime.py` — change warmup to accept per-cluster scope, OR provide a teardown helper to drop signatures for completed clusters
- `gpu_fuzzy_trader/phases/phase2_island_scheduler.py:_run_cluster_islands` — call teardown after each cluster's epochs (extends Task 4's `del generators[cid]` with a JAX-level cache eviction)
- `gpu_fuzzy_trader/evolution/evox_runner.py` — may need to expose cache eviction per cluster (or use existing `_trim_global_metrics_cache` pattern)

**Implementation approach (recommended):**
- Add a `_evict_cluster_signatures(cluster_id)` helper in `_gpu_runtime.py` that pops the cluster's signatures from `_WARMED_SIGNATURES` and calls `jax.clear_caches()` if available
- In `_run_cluster_islands`, after each cluster's `del generators[cid]; gc.collect()` (from Task 4), call `_evict_cluster_signatures(cid)` and another `gc.collect()`
- Each subsequent cluster pays one JIT recompile cost (~60s per the warmup timing in the log) but only the active cluster's signatures are alive

**Acceptance criteria:**
- New helper `_evict_cluster_signatures(cluster_id)` exists in `_gpu_runtime.py` and is called between clusters
- After teardown, `_WARMED_SIGNATURES` no longer contains the evicted cluster's signatures (verifiable via debug log or assertion)
- Re-warming for the next cluster produces the same `signatures=N+2` count, proving recompile is happening
- All touched test suites pass with `PYTEST_LOW_MEMORY=1`
- No regressions in existing overfit-gap behavior

---

## Verification (after both tasks merged)

- Re-run Phase 2 on Colab
- Watch for: `Phase 2 JAX warmup complete (2 engines warmed, 0 skipped, ..., signatures=2)` repeating 3 times (once per cluster), instead of `signatures=4` once
- Expected peak RAM: ~6-7 GB (down from 9.9 GB)
- Expected runtime impact: +60s per cluster from JIT recompile = +3 min total (acceptable trade)

## Out of scope

- Item 10 (f1/f3 asymmetry) — still deferred
- Feasibility collapse (valid_rules=2-4) — separate plan when re-run data is in
- Pareto collapse (corr_f1_f3=1.00) — separate plan
- `_sample_df` random-start silent bypass — separate small task (per the previous analysis)
