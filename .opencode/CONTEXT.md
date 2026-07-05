# Nexus Context

**Updated:** 2026-07-05
**base_branch:** `main`
**branch_policy:** isolated
**execution_mode:** checkpoint
**status:** active — new plan: Phase 2 overfit-gap blind spots + bugs (Stages 1-3)

## Active Plan
**Plan: Fix Phase 2 overfit-gap blind spots + confirmed bugs (Stages 1-3)**
**Source:** `/home/danaee/.claude/plans/you-are-a-senior-pure-cupcake.md`
**PLAN:** `.opencode/plans/PLAN.md`

| Task | Title | Branch | Status |
|------|-------|--------|--------|
| 1 | Stage 1: Mechanical fixes (5 items) | `fix/phase2-stage1-mechanical` | ⏳ Pending |
| 2 | Stage 2: Seed collision + migration cadence | `fix/phase2-stage2-seed-migration` | ⏳ Pending |
| 3 | Stage 3: Fitness-function gap fixes (core fix) | `fix/phase2-stage3-fitness-gap` | ⏳ Pending |

## Plan Verification Notes

Audit performed 2026-07-05 by reading every file/line cited in the plan:
- ✅ `config.py:176-180` — `SPLIT_MODE = "holdout_70_30"` is a real string compared via `==`
- ✅ `data/splitter.py:6` — module docstring already silently admits 65/35
- ✅ `phases/phase2_rule_pool.py:14-17` — stale `f3 = -win_rate` docstring confirmed
- ✅ `evolution/evox_runner.py:2660-2673` — `corr_f1_f3` warning at DEBUG confirmed
- ✅ `run_pipeline.py:227-232` and `phases/phase2_island_scheduler.py:353-358` — duplicated budget math confirmed
- ✅ `config.py:1098-1108` — stale "overhead without benefit" comment + live `True` confirmed
- ✅ `phases/phase2_island_scheduler.py:40-58` — `_derive_island_seed` signature does NOT include `direction`
- ✅ `phases/phase2_island_scheduler.py:296, 380` — both call sites have `direction` in scope
- ✅ `phases/phase2_island_scheduler.py:389-419` — `epoch_counter += 1` is INSIDE `for cid` loop (the cadence bug)
- ✅ `phases/phase2_rule_pool.py:818-834` — `if val_ret > 0.0` gate confirms the blind spot
- ✅ `phases/phase2_support.py:329-369` — `_raw_feasibility_violation_score` confirmed missing gap check
- ✅ `phases/phase2_support.py:179-181` — final pool-admission gate does check `PHASE2_MAX_TRAIN_VAL_GAP_PCT`

Plan is accurate. No deviations needed.

## Stage 4 (deferred)
Resampling train/val per island-epoch — deferred to follow-up plan per user choice.

## Test Discipline
- `PYTEST_LOW_MEMORY=1` required (per AGENTS.md)
- Only run touched test suites, not full suite (OOM risk)
- New regression tests added *before* corresponding fix where practical
