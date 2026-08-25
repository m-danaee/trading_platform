# Nexus Context

workflow: default
execution_mode: delegated
branch_cleanup_policy: always
active_objective: fix-fold-gates — address verdict blockers: ratio support, count-gate scaling, rolling_cv cleanup, SPLIT_MODE tests, low cleanup
current_phase: PLANNED
base_branch: main
branch_policy: isolated
execution_mode: checkpoint
verification_baseline:
  build: none detected
  test: PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest -q --ignore=tests/benchmark
  lint: none detected
  typecheck: none detected
  baseline_run: pending
plan_commit: 3d0ce31 (3d0ce312bd043594180fd82b7f3a106e264ca3c1)
generated_at: 2026-08-25T16:30:00+03:30
impact: pending (nexus impact to be run per-task pre-impact)
pending_blockers: none
next_action: nexus run transition --to PLANNED
