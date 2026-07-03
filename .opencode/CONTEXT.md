# Nexus Context — Phase 2 Runtime Blowup & Cross-Phase Objective Continuity

**Updated:** 2026-07-03
**base_branch:** `main`
**branch_policy:** isolated
**execution_mode:** checkpoint
**plan:** `.opencode/plans/PLAN.md` (4 tasks: 20-23)

## Active Objective
The 2026-07-01 Colab run (post tasks 1-3) still shows 60-230s/generation and
weak OOS results. Root-cause audit found the dominant costs are an
unconditional duplicate CPU re-simulation per generation and a broken
Phase2→RB-Governor objective handoff (RB Governor silently re-parametrizes
SL/capital_pct and never validates against Phase 2's fitness criteria).
See `.opencode/tasks/task-20.md`..`task-23.md` for full analysis.

## Constraints (AGENTS.md)
- Use `.venv` for all commands; tests with `PYTEST_LOW_MEMORY=1` only.
- Do NOT run the pipeline. Do NOT touch `evaluator_v5.ipynb`.
- Remove dead/obsolete code after edits.

## Previous Plans (COMPLETE)
- **A1** ✅ MERGED (96e8c08) — batched offspring eval
- **A2** ✅ MERGED (ed4f183) — periodic val sim
- **A3** ✅ MERGED (ce3706a) — island patience dead-code fix
- **Task 1** ✅ MERGED (cd69462) — Config parameter tuning (runtime + OOS)
- **Task 2** ✅ MERGED (feat/phase2-evox-fixes) — EvoX runner code fixes (FIFO cache + phenotype-collapse trigger)
- **Task 3** ✅ MERGED (feat/phase2-island-fixes) — Island scheduler + pool admission fixes (min epoch guard, patience helper, config param)
- **OOM Fix** ✅ MERGED (fix/oom-rb-governor-transition) — Memory cleanup between phases to prevent OOM at RB Governor boundary

## Current Plan Tasks
- **Task 20** 🔄 IN PROGRESS — Kill per-generation runtime blowup
- **Task 21** 🔲 TODO — Island RNG state leakage + generation-budget realization
- **Task 22** 🔲 TODO — Restore Phase2→RB-Governor→OOS objective continuity
- **Task 23** 🔲 TODO — Config/logging anomaly cleanup

## Current State
**Task 20 in progress** on branch `fix/phase2-runtime-blowup`.
Recommended order: Task 20 first (unblocks fast iteration), then Task 21 + Task 22 in parallel, Task 23 any time.
