# Nexus Context — Phase 2 Runtime & OOS Quality Overhaul

**Updated:** 2026-07-01
**base_branch:** `main`
**branch_policy:** isolated
**execution_mode:** checkpoint
**plan:** `.opencode/plans/PLAN.md` (3 tasks)

## Active Objective
Fix 7 critical + 5 moderate issues from latest Colab run analysis.
Target: ~60% runtime reduction + improved OOS generalization.

## Constraints (AGENTS.md)
- Use `.venv` for all commands; tests with `PYTEST_LOW_MEMORY=1` only.
- Do NOT run the pipeline. Do NOT touch `evaluator_v5.ipynb`.
- Remove dead/obsolete code after edits.

## Previous Plan (COMPLETE)
- **A1** ✅ MERGED (96e8c08) — batched offspring eval
- **A2** ✅ MERGED (ed4f183) — periodic val sim
- **A3** ✅ MERGED (ce3706a) — island patience dead-code fix

## Current Plan Tasks
- **Task 1** ✅ MERGED (cd69462) — Config parameter tuning (runtime + OOS)
- **Task 2** ✅ MERGED (feat/phase2-evox-fixes) — EvoX runner code fixes (FIFO cache + phenotype-collapse trigger)
- **Task 3** ✅ MERGED (feat/phase2-island-fixes) — Island scheduler + pool admission fixes (min epoch guard, patience helper, config param)

## Current State
**All 3 tasks complete and merged to main.** Ready for next Colab run to validate improvements.
