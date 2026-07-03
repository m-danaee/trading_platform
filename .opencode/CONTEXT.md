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
- **Task 2** ✅ MERGED (feat/phase2-evox-fixes) — EvoX runner code fixes
- **Task 3** ✅ MERGED (feat/phase2-island-fixes) — Island scheduler + pool admission
- **OOM Fix** ✅ MERGED (fix/oom-rb-governor-transition) — Memory cleanup
- **Task 20** ✅ COMPLETE (on fix/phase2-runtime-blowup) — Kill per-generation runtime blowup
- **Task 21** ✅ COMPLETE (on fix/island-rng-and-budget) — Island RNG state leakage

## Current Plan Tasks
- **Task 22** ✅ COMPLETE (ac918c4 on fix/objective-continuity) — Restore Phase2→RB-Governor→OOS objective continuity
- **Task 23** 🔲 TODO — Config/logging anomaly cleanup

## Current State
**Task 22 completed.** Branch `fix/objective-continuity` with 2 commits, passed spec review + code review. Awaiting checkpoint resume for Task 23.
