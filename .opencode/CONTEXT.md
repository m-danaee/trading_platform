# Nexus Context

Active objective: Fix the 13 CONFIRMED/SUSPECTED findings from the
gpu_fuzzy_trader OOS audit (long 58.21%→49.92%→22.22%, short
60.57%→50.47%→15.11% collapse on 2026-07-07 run).
Current phase: task-9 implementer dispatching
Current task: task-9 (DRAFTED; spec committed, implementer dispatching)
  - task-1..8 MERGED
  - task-9 in flight (remove dead migration helper + deprecate
    PHASE2_MIGRATION_EPOCH_INTERVAL config — audit fix #6)
base_branch: main
feature_branch: feature/task-9-remove-dead-migration-helper
branch_policy: isolated
execution_mode: checkpoint
Pending blockers: none
Next action: dispatch implementer subagent for task-9; await handoff
JSON at .opencode/handoffs/task-9-implementer.json; then spec-review
then code-review before user merges and proceeds to task-10
(disable cache refresh when window is fixed — depends on task-1,
audit fix #8).
