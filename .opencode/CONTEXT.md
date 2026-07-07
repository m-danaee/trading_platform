# Nexus Context

Active objective: Fix the 13 CONFIRMED/SUSPECTED findings from the
gpu_fuzzy_trader OOS audit (long 58.21%→49.92%→22.22%, short
60.57%→50.47%→15.11% collapse on 2026-07-07 run).
Current phase: task-7 reviews complete, awaiting user merge
Current task: task-7 COMPLETE (spec APPROVED, code APPROVED)
  - Implementer commit 1: 0b866d4 (feat — split floor flags)
  - Fix commit:          19df0a1 (true alias + test aligned with spec)
  - 304/304 tests pass (3 new + 301 prior)
  - 2 INFO code-review findings (optuna_search.py patches old alias;
    ~16 tests monkeypatch old alias; both out of scope per spec)
base_branch: main
feature_branch: feature/task-7-pf-floor-split
branch_policy: isolated
execution_mode: checkpoint
Pending blockers: user must merge feature/task-7-pf-floor-split
  to main before task-8 can dispatch.
Next action: user merges the feature branch; then "continue" or
"start task-8" dispatches the val_df usage fix in Phase 1 sign
consistency filter (audit fix #11).
