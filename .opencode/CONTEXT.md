# Nexus Context

**Updated:** 2026-07-06
**base_branch:** `main`
**branch_policy:** isolated
**execution_mode:** checkpoint
**status:** COMPLETE — all 4 tasks (A, B, C, D) merged to main, all branches cleaned up

## Active Plan
**Plan: Post-RAM-fix run analysis — 4 verified improvements**
**Source:** 2026-07-06 re-run log + screenshot showing RAM at 9.6/12.7 GB and high max_return. Cross-referenced with prior analysis at `~/.claude/plans/i-m-partially-run-my-temporal-treehouse.md`
**PLAN:** `.opencode/plans/PLAN.md`

### Just completed
**Task 25 — RAM low-cost knobs (A1 + A2 + A3)** — ✅ MERGED
- Branch: `fix/ram-knobs-final` (merged, ready for cleanup)
- Commit: a605d80 (merge), 24d68d8 (polish), 88d3c66 (impl)
- Tests: 222 passed

### Just completed
**Task 26 — f3 train+val blend (B / Item 10)** — ✅ MERGED
- Branch: `fix/f3-train-val-blend` (merged, ready for cleanup)
- Commit: d70d06f (merge), 2e36db6 (impl)
- Tests: 134 passed in test_phase2_rule_pool.py
- Code-reviewer notes: 2 LOW-severity findings (not blocking):
  - Fallback in get() differs from win_rate branch (uses profit_factor vs 0.0)
  - Empty val_metrics dict triggers val-gate (theoretical edge case)

### Just completed
**Task 27 — Document evaluator_v5 parity (C, docs-only)** — ✅ MERGED
- Branch: `docs/time-exit-evaluator-parity` (merged, ready for cleanup)
- Commit: 9091a48 (merge), 95ec002 (impl)
- Change: 4 comment blocks, 40 insertions, 0 deletions
- Code-reviewer note (LOW): comment uses 'close_ret' conceptual name; local var is 's_close' (harmless)

### All 4 tasks COMPLETE — branches cleaned up
**Cleanup complete:** 4 feature branches deleted via  (all confirmed merged). See  (2026-07-06T12:22:22Z).


**Task 25 — RAM low-cost knobs (A1 + A2 + A3)** — ✅ MERGED
- Branch: `fix/ram-knobs-final` (merged, ready for cleanup)
- Commit: a605d80 (merge), 24d68d8 (polish), 88d3c66 (impl)
- Tests: 222 passed

**Task 26 — f3 train+val blend (B / Item 10)** — ✅ MERGED
- Branch: `fix/f3-train-val-blend` (merged, ready for cleanup)
- Commit: d70d06f (merge), 2e36db6 (impl)
- Tests: 134 passed in test_phase2_rule_pool.py

**Task 27 — Document evaluator_v5 parity (C, docs-only)** — ✅ MERGED
- Branch: `docs/time-exit-evaluator-parity` (merged, ready for cleanup)
- Commit: 9091a48 (merge), 95ec002 (impl)
- Change: 4 comment blocks, 40 insertions, 0 deletions

**Task 28 — Feasibility collapse observability (D, log-only)** — ✅ MERGED
- Branch: `feat/feasibility-observability` (merged, ready for cleanup)
- Commit: c0e910e (merge), 1d17268 (impl)
- Tests: 60 passed (25 in test_phase2_support.py + 35 in test_evox_runner.py)
- Code-reviewer: 3 LOW-severity non-blocking notes (duplication, missing test, log format)

### Verification verdicts
- A (RAM knobs): IMPLEMENT
- B (f3 blend): IMPLEMENT
- C (time-exit cap): DO NOT re-implement (would diverge from evaluator_v5.ipynb per AGENTS.md); document only
- D (feasibility collapse): Implement observability only; defer floor relaxation

### Completed plan (reference)
**Plan: Phase 2 RAM Optimization (avoid Colab OOM)** — original Task 4/5
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

## Post-merge re-run analysis (2026-07-06)

After the RAM Task 4/5 fixes were merged (commits up to 39618bd), a fresh
`run_pipeline` run was started. Two issues remain visible in the log:

### Issue 1 — RAM still climbing early (not a regression)
- 9.6 / 12.7 GB at gen 9 (cluster_0, first epoch, ~26 min in)
- Task 4/5 fixes target **between-cluster** steady state; they don't help
  within a single cluster's run before that cluster's own eviction fires
- Likely root cause: glibc allocator fragmentation (gc.collect() reclaims
  Python objects, not OS arena pages)
- Recommended low-cost knobs in `config.py`:
  1. `PHASE2_ISLAND_EPOCH_GENERATIONS`: 25 → ~10-13 (line 1104) — more
     frequent `trim_evolution_state_memory`; respects the existing
     `PHASE2_ISLAND_MIN_EPOCH_GENERATIONS=5` floor; total budget unchanged
  2. `PHASE2_EVAL_GLOBAL_CACHE_MAX_SIZE`: 600 → 200-300 (line 414) — cache
     hit rate is already 0-4%, near-unused
  3. `malloc_trim(0)` after each `gc.collect()` (Colab is Linux glibc) —
     separate small patch, not applied yet
  4. `PHASE2_POPULATION_SIZE`: 200 → 128-150 (line 1061) — last-resort
     lever, directly trades quality for RAM

### Issue 2 — Elevated returns: not a calculation bug
Verified the previous Claude Code analysis plan at
`~/.claude/plans/i-m-partially-run-my-temporal-treehouse.md` against current
source — claims about line numbers, f3 behavior, and overfit-gap fixes
all check out.

- Stage 1-3 overfit fixes (commits 8ff3328, etc.) are working: today's
  `max_train_val_gap_ratio` is bounded at 2.7x-4.5x vs the previously
  unbounded 7.77x
- The remaining overfit lever (deliberately deferred, "Item 10" in the
  prior plan) is making `f3` (profit_factor) blend train+val the way `f1`
  (Sortino) already does. The `win_rate` branch at
  `phase2_rule_pool.py:720-737` shows the pattern; the `profit_factor`
  branch sets `f3_val = profit_factor` (train-only)
- A separate, independent concern surfaced: the backtest engine's
  time-exit return was once capped (`MAX_TIME_EXIT_RETURN_PCT`) and the
  cap was reverted the same day (commits 46cb88a → 072c527) with no
  recorded reason. **Re-verified: evaluator_v5.ipynb:958, 971 has no
  cap on the time-exit return** — re-implementing would diverge from
  the user's ground-truth evaluator (per AGENTS.md). Decision: document
  only, do not re-implement
- Feasibility collapse (valid_rules=2-4 out of 200) flagged in repo
  notes as open and untraced — small surviving population inflates noise
  Plan: add observability first; defer floor relaxation until we have
  evidence from the new logs

## Hard rules
- `PYTEST_LOW_MEMORY=1` for any test run
- Only run touched test suites (OOM risk on full suite)
- Use `.venv/bin/python`
- Per AGENTS.md: do not run the project (OOM risk) — Colab GPU only
- `evaluator_v5.ipynb` is the source of truth for rule testing (do not change)

## Branch cleanup
All feature branches from previous plans are deleted. No branches pending
cleanup as of this update.

4 feature branches from this plan deleted: fix/ram-knobs-final, fix/f3-train-val-blend, docs/time-exit-evaluator-parity, feat/feasibility-observability

## Plan Status: COMPLETE — Both tasks merged to main
**Total expected RAM savings: 2.8-5.0 GB** (down from 9.9 GB → ~5-7 GB)
Post-merge re-run shows 9.6/12.7 GB at gen 9 (in-cluster fragmentation not
addressed by between-cluster cleanup) — additional knobs proposed, awaiting
user decision.
