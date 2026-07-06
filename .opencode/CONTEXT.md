# Nexus Context

**Updated:** 2026-07-05
**base_branch:** `main`
**branch_policy:** isolated
**execution_mode:** checkpoint
**status:** active — new plan: Phase 2 RAM optimization (avoid Colab OOM)

## Active Plan
**Plan: Phase 2 RAM Optimization (avoid Colab OOM)**
**Source:** Pre-fix re-run log showed 9.9 / 12.7 GB RAM at gen 6; OOM risk imminent
**PLAN:** `.opencode/plans/PLAN.md`

| Task | Title | Branch | Status |
|------|-------|--------|--------|
| 4 | RAM quick wins (cache + teardown + gc) | `fix/ram-quick-wins` | ✅ Merged (commits 1b647ca, 7584a6d, a0b4c76) |
| 5 | Sequential cluster warmup (drop 3/4 signatures) | `fix/ram-sequential-clusters` | ✅ Merged (commits ed1db9b, f4f3509) |

### Task 4 — RAM quick wins (merged)
- Cache size 1200 → 600 (saves ~0.4 GB)
- `clear_global_metrics_cache()` added; called between clusters
- `del generators[cid]; gc.collect()` after each cluster's `finalize_island()` (with `list()` wrap to prevent dict-iteration RuntimeError — caught in code review)
- `gc.collect()` every 3 gens (was 10)
- Expected savings: ~1.3-2.0 GB

### Task 4 review highlights
- Spec review found 1 minor (stale comment) — fixed
- Code review found 1 HIGH-severity bug: `del generators[cid]` inside `for cid, gen in generators.items():` would raise `RuntimeError: dictionary changed size during iteration` on the 2nd cluster. **All 221 tests passed despite the bug** because tests use `inspect.getsource()` (structural), never invoke the function with multiple clusters. Classic "tests pass but bug is real" pattern — flagged for awareness.

## Previous plan (completed)
**Plan: Fix Phase 2 overfit-gap blind spots + confirmed bugs (Stages 1-3)**
- ✅ Stages 1-3 merged to main (8 commits, ahead of `origin/main` by 8)
- ✅ All 3 feature branches deleted
- ✅ Post-fix re-run (2026-07-05 22:09) confirms Stages 1-3 working:
  - SPLIT_MODE: "holdout 65/35" ✓
  - corr_f1_f3: WARNING level ✓
  - max_train_val_gap bounded at 12.70% (was 86.46%) ✓
  - max_train_val_gap_ratio bounded at 1.57-5.50x (was 7.77x frozen) ✓

## RAM Analysis (from 2026-07-05 22:09 log)
- Peak RAM: 9.9 / 12.7 GB at gen 6
- Top consumers: 4× JAX compiled programs (~3 GB), 3× cluster DataFrames (~1.5 GB), global metrics cache (~0.8 GB)
- Plan targets ~6-7 GB peak (1.5-3.5 GB savings)

## Hard rules
- `PYTEST_LOW_MEMORY=1` for any test run
- Only run touched test suites (OOM risk on full suite)
- Use `.venv/bin/python`
- Per AGENTS.md: do not run the project (OOM risk) — Colab GPU only
- `evaluator_v5.ipynb` is the source of truth for rule testing (do not change)

## Branch cleanup
3 feature branches from previous plan are deleted.
Current plan: 2 branches pending cleanup after final review.

## Plan Status: COMPLETE — Both tasks merged to main
**Total expected RAM savings: 2.8-5.0 GB** (down from 9.9 GB → ~5-7 GB)

### Task 4 — RAM quick wins (merged)
- Cache size 1200 → 600 (saves ~0.4 GB)
- `clear_global_metrics_cache()` added; called between clusters
- `del generators[cid]; gc.collect()` after each cluster's `finalize_island()` (with `list()` wrap to prevent dict-iteration RuntimeError — caught in code review)
- `gc.collect()` every 3 gens (was 10)
- Expected savings: ~1.3-2.0 GB

### Task 5 — Sequential cluster warmup (merged)
- `defer_warmup` flag added to `Rule_Pool_Generator.__init__`; default `False` (orphan-boost path unchanged)
- `_run_cluster_islands` restructured to sequential per-cluster processing: warm → epochs → evict → next
- New `evict_cluster_signatures(cluster_id)` helper in `_gpu_runtime.py` filters `_WARMED_SIGNATURES` by cluster_id, calls `jax.clear_caches()` + `gc.collect()`
- Peak JAX signatures reduced from 4 to 2 (one cluster's train+val alive at a time)
- Migration changed from round-robin mesh to sequential chain (necessary consequence; migrants flow cluster N → cluster N+1)
- Expected savings: ~1.5-2.0 GB JAX compiled program memory

### Critical review findings (both fixed)
- **Task 4**: `del generators[cid]` inside `for cid, gen in generators.items():` would raise `RuntimeError: dictionary changed size during iteration` on the 2nd cluster. **Tests passed despite the bug** because they use `inspect.getsource()` (structural), never invoke the function with multiple clusters. Fixed via `list()` wrap.
- **Task 5 (initial)**: Eviction happened AFTER all clusters' epochs (too late — didn't solve OOM during the run). Required restructuring `Rule_Pool_Generator` to defer warmup via `defer_warmup=True` flag, calling warmup per-cluster inside the while loop. Spec reviewer caught the gap; implementer fixed.

### Out-of-scope (separate plan when re-run data is in)
- Item 10 (f1/f3 asymmetry) — corr_f1_f3=1.00 still in log
- Feasibility collapse (valid_rules=2-4) — 95-99% infeasible
- `_sample_df` random-start silent bypass — separate small task
- `_should_migrate_this_round` is now dead code after Task 5 (sequential chain replaced round-robin) — flagged in code review as dead_code finding
