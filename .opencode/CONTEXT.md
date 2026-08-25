# Nexus Context

workflow: default
execution_mode: delegated
branch_cleanup_policy: always
active_objective: unified-fold-gates — unify Folds/Purge/Gate scaling, remove deprecated purged_walk_forward/SPLIT_MODE/rolling_cv, achieve single adaptive expanding Master Temporal Fold system
current_phase: PLANNED
base_branch: main
branch_policy: isolated
execution_mode: checkpoint
verification_baseline:
  build: none detected
  test: PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest -q --ignore=tests/benchmark
  lint: none detected
  typecheck: none detected
  baseline_run: pending (to be run at first implementer handoff)
plan_commit: d151be4 (d151be48c23dad8f02c1a852cc2bf1f1a1b72ea3)
generated_at: 2026-08-25T15:06:56+03:30
impact: pending (nexus impact to be run per-task pre-impact)
pending_blockers: none
next_action: nexus run transition --to PLANNED then TASK_IMPACT_READY for task-1

