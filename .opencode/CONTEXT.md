# Nexus Context

**Updated:** 2026-07-05
**base_branch:** `main`
**branch_policy:** isolated
**execution_mode:** checkpoint
**status:** COMPLETED — All 3 tasks merged to main

## Completed Plan
**Plan: Fix Phase 2 overfit-gap blind spots + confirmed bugs (Stages 1-3)**
**Source:** `/home/danaee/.claude/plans/you-are-a-senior-pure-cupcake.md`
**PLAN:** `.opencode/plans/PLAN.md`

| Task | Title | Branch | Commit | Status |
|------|-------|--------|--------|--------|
| 1 | Stage 1: Mechanical fixes (5 items) | `fix/phase2-stage1-mechanical` | `8d81c8f` | ✅ Merged |
| 2 | Stage 2: Seed collision + migration cadence | `fix/phase2-stage2-seed-migration` | `0c8d014` | ✅ Merged |
| 3 | Stage 3: Fitness-function gap fixes (core) | `fix/phase2-stage3-fitness-gap` | `8ff3328` | ✅ Merged |

### Merge flow
```
main: 6583671 (pre-plan) 
  → 8d81c8f (Task 1: Stage 1 mechanical)
  → 0c8d014 (Task 2: n_clusters fix + seed+migration)
  → 8ff3328 (Task 3: Stage 3 fitness gap fixes)
  → 14f859e (Task 3 handoff commit)
```

### What changed (cumulative)

**Task 1 — Stage 1 (5 items, zero behavior risk)**
- `SPLIT_MODE` `"holdout_70_30"` → `"holdout"` (atomic, all 16 .py files; 0 grep hits)
- `holdout_train_val_label(frac)` helper added in `config.py` (dynamic from `HOLDOUT_TRAIN_FRACTION`)
- `compute_cluster_generation_budgets()` helper extracted, used by both call sites
- Log key `per_cluster=` → `per_cluster_gens=` (no longer reads as population split)
- `corr_f1_f3` warning: `logger.debug` → `logger.warning`
- `phases/phase2_rule_pool.py` docstring reflects `PHASE2_F3_OBJECTIVE` semantics
- `PHASE2_MIGRATION_ENABLED` comment rewritten (re-enabled-with-guards narrative; value stays `True`)

**Task 2 — Stage 2 (2 items + critical fix)**
- Long/short seed collision: `_derive_island_seed` call sites now include `direction` prefix at both cluster (`f"{direction}_{cid}"`) and orphan (`f"{direction}_orphan_{sym}"`) paths
- Migration cadence bug: extracted `_should_migrate_this_round()` pure helper; `round_counter` increments once per outer `while` round (was per cluster, masked by `interval=1`)
- **CRITICAL FIX (caught in spec review):** `n_clusters` was undefined inside `_run_cluster_islands` after Task 1's refactor — would have caused `NameError` at runtime when migration fired. Added `n_clusters = len(cluster_ids)` at line 403.

**Task 3 — Stage 3 (2 items, THE CORE FIX)**
- `overfit_gap_penalty` switched from ratio-based to subtraction-based (`train_ret - val_ret`), matching the final pool-admission gate
- Removed `if val_ret > 0.0` gate (penalty now well-defined for any sign of `val_ret`); replaced with meta-gate `PHASE2_JOINT_TRAIN_VAL or PHASE2_VAL_IN_FITNESS_PENALTY` to preserve existing C6 test contract
- Renamed `PHASE2_OVERFIT_GAP_RATIO_THRESHOLD` → `PHASE2_OVERFIT_GAP_PCT_THRESHOLD` (default 8.0pp, below hard gate 16.0pp)
- Added train-vs-val gap check to `_raw_feasibility_violation_score` (the highest-leverage single change — flows into deployability preview, real objectives, and elite preservation)
- Pre-existing `test_f3_uses_min_train_val_return` now passes (was failing in Tasks 1-2)

### Item 10 — DEFERRED
f1/f3 asymmetry (making `f3` worst-of-train/val like `f1`) is explicitly out of scope per the plan. Land Stages 1-3 first, re-run, and only pursue if the gap is still insufficiently controlled.

## Branch Cleanup
Three feature branches are still local:
- `fix/phase2-stage1-mechanical` — can be deleted
- `fix/phase2-stage2-seed-migration` — can be deleted
- `fix/phase2-stage3-fitness-gap` — can be deleted

**Will be dispatched to implementer for cleanup via branch-cleanup-prompt.md per orchestrating skill.**

## Test Status
- All 8 specified test suites pass (253 tests total)
- Pre-existing failure `test_f3_uses_min_train_val_return` is now RESOLVED
- No regressions introduced

## Verification (recommended post-merge)
- Run full touched suites: `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_support.py tests/unit/test_phase2_rule_pool.py tests/unit/test_phase2_island_scheduler.py tests/unit/test_island_scheduler_migration.py tests/unit/test_migration_safety.py tests/unit/test_elite_preservation.py tests/unit/test_evox_runner.py -v`
- Re-run Phase 2 on the same seed/data and compare the `max_train_val_gap_ratio` trajectory (should stay bounded, not climb to 7-8x)
- Check final pool size after admission gating (may shrink if previously-"deployable" overfit rules no longer qualify mid-run)

## Next Action
- **AWAITING RE-RUN** on Colab with merged Stages 1-3 — user will provide new log for post-fix analysis
- 8 commits on local main ahead of `origin/main` (needs `git pull` on Colab before re-run)
- Pre-existing analysis (`/home/danaee/.claude/plans/you-are-a-senior-pure-cupcake.md` audit + Stages 1-3 + post-run analysis) flagged Stage 4 + Item 10 + feasibility-collapse fix as the next likely plan, contingent on the re-run data
- 3 merged feature branches already deleted via `git branch -d`
