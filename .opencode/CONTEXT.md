# Nexus Context — Phase 2 Island Plateau & OOS Leakage Fixes

**Updated:** 2026-06-26
**base_branch:** `main`
**branch_policy:** isolated
**execution_mode:** auto-continue

## Active Objective

Fix the premature-convergence cascade diagnosed from the 2026-06-26 run log.
Root cause: plateau state (`plateau_best_progress`, `plateau_streak`) leaks across
island epochs because `reset_plateau = entering_stage_b` and
`PHASE2_ISLAND_TWO_STAGE_ENABLED=False` makes `entering_stage_b` always False.
Compounded by: holdout val folded into Phase 2 fitness (`JOINT_TRAIN_VAL=True`),
frozen-elite attractor (no diversity restart), imbalanced K=3 clusters, and
over-twitchy plateau config (patience=5, min_gen=3, min_delta=0.02).

## Task Summary (5 tasks, sequential, isolated branches)

| # | Branch | Fixes | Status |
|---|--------|-------|--------|
| 1 | `fix/plateau-state-leak` | A (reset plateau per epoch) + B (charge actual gens) | ✅ MERGED |
| 2 | `fix/holdout-fitness-leak` | C (`JOINT_TRAIN_VAL=False`, verify Phase 3/4 reuse) | ✅ MERGED |
| 3 | `fix/diversity-restart-on-plateau` | D (diversity restart instead of immediate break) | ✅ MERGED |
| 4 | `fix/cluster-balancing` | E (balanced clustering + relax large-cluster gates) | ✅ MERGED |
| 5 | `fix/plateau-config-tuning-and-banner` | F (banner) + config knobs + task-4 README | ✅ MERGED |

## Hard Rules (from AGENTS.md)

- Always use `.venv` for running commands.
- Run tests with `PYTEST_LOW_MEMORY=1` (OOM risk on local/WSL).
- Do NOT run the full pipeline locally (runs on Colab GPU).
- Do NOT modify `evaluator_v5.ipynb`.
- After changing code, remove wasted/old implementation to keep project clean.

## Workflow

- Dispatch ONE implementer at a time per task.
- Two-stage review: spec-reviewer → code-reviewer.
- Branch from latest `main` (rebase after each merge).
- At completion, delegate branch cleanup to implementer (never delete branches directly).
