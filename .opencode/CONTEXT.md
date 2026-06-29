# Nexus Context — Phase 2 Runtime Reduction (post-restart early stop)

**Updated:** 2026-06-29
**base_branch:** `main`
**branch_policy:** isolated
**execution_mode:** checkpoint
**plan:** `.opencode/plans/PLAN.md` (1 task)

## Active Objective

Reduce Phase 2 (rule-pool evolution) wall-clock time by cutting
provably-unproductive generations that run *after a plateau restart fails to
yield any improvement*.  No reduction to search budget, population, or epoch
count — only to generations that produce zero improvement.

## Diagnosis (from 2026-06-29 run log)

- Phase 2 island epochs are 15 gens each, ~80–280 s/gen.
- `PHASE2_ISLAND_PLATEAU_EARLY_STOP_PATIENCE = 8` ⇒ restart fires late in an
  epoch (gen 10–14).  After restart, `plateau_streak` resets and needs 8 more
  no-improvement gens to stop — but only 1–5 gens remain, so the stop **never
  fires** and dead generations run to the end of the epoch.
- Example (epoch 3, long cluster_2): restart at gen 10, then gens 11–15 stuck
  at 6.11% return = ~5 min wasted.

## Approach

Add a **post-restart no-improvement early stop**: a separate, short patience
(default 3 gens, == existing `PHASE2_PLATEAU_POST_RESTART_BOOST_GENS`) that
activates *only* after a plateau restart.  If the restart + boosted mutation
yields no improvement beyond the pre-restart best within the patience window,
break the epoch.  Plus lower `PHASE2_ISLAND_PLATEAU_EARLY_STOP_PATIENCE` 8→6 so
restarts happen sooner (more post-restart evaluation room).

## Constraints (AGENTS.md)

- Use `.venv` for all commands.
- Run tests with `PYTEST_LOW_MEMORY=1` only — never plain `pytest` (OOM risk).
- Do NOT run the full pipeline (OOM on local; user runs on Colab GPU).
- Do NOT modify `evaluator_v5.ipynb`.
- Remove dead/obsolete code after edits.

## Current Task

**task-1** — post-restart early stop — ✅ IMPLEMENTED + SPEC-REVIEW APPROVED + CODE-REVIEW APPROVED
- Branch: `feature/task-1-post-restart-early-stop` (commit `ad2708e`, not merged)
- Awaiting user confirmation to finalize/merge (execution_mode: checkpoint).
