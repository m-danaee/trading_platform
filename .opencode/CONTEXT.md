# Nexus Context — Phase 2 Priority A Runtime Fixes (Path 1: safe ~30%)

**Updated:** 2026-06-30
**base_branch:** `main`
**branch_policy:** isolated
**execution_mode:** checkpoint
**plan:** `.opencode/plans/PLAN.md` (3 tasks)

## Active Objective
Path 1 (user-chosen, safe ~30% runtime win): A1 (consistency, done) + A2
(periodic val sim) + A3 (patience bug fix). No generation-budget cut.

## IMPORTANT DIAGNOSTIC CORRECTION
- `_run_nsga3` (production path used on Colab) ALREADY batches offspring via
  `_evaluate_population_indices`. A1 was a consistency fix for the fallback
  path only — ~0% production runtime win, but clean tested code worth merging.
- Real production per-gen cost is genuine batched backtest compute (~50-120s).
- A2 (skip val most gens) is the real ~25-30% win. A3 is a correctness fix.

## Constraints (AGENTS.md)
- Use `.venv` for all commands; tests with `PYTEST_LOW_MEMORY=1` only.
- Do NOT run the pipeline. Do NOT touch `evaluator_v5.ipynb`.
- Remove dead/obsolete code after edits.

## Tasks
- **A1** ✅ MERGED (96e8c08) — batched offspring eval (fallback path consistency + test)
- **A2** ✅ MERGED (ed4f183) — periodic val sim (~25-30% win) + JOINT contract fix + interval guard
- **A3** ✅ MERGED (ce3706a) — island patience dead-code fix (restarts fire at streak 6, not 8)

## Current State
**ALL PRIORITY A TASKS COMPLETE + MERGED to main.**
Ready for next Colab run to confirm runtime reduction.
