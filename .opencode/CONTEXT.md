# Nexus Context

Active objective: Fix the 13 CONFIRMED/SUSPECTED findings from the
gpu_fuzzy_trader OOS audit (long 58.21%→49.92%→22.22%, short
60.57%→50.47%→15.11% collapse on 2026-07-07 run).
Current phase: task-4 implementer dispatching
Current task: task-4 (DRAFTED; spec committed, implementer dispatching)
  - task-1 MERGED (per-epoch window rotation; commit 44f8631)
  - task-2 MERGED (4th NSGA objective; commit 703a777)
  - task-3 MERGED (RB walk-forward; commit 1d9521d)
  - task-4 in flight (monthly gate on val — smallest diff, surgical fix)
base_branch: main
feature_branch: feature/task-4-monthly-gate-on-val
branch_policy: isolated
execution_mode: checkpoint
Pending blockers: none
Next action: dispatch implementer subagent for task-4; await handoff
JSON at .opencode/handoffs/task-4-implementer.json; then spec-review
then code-review before user merges and proceeds to task-5
(delete dead f3 profit_factor branch).
