# Nexus Context

Active objective: Fix the 13 CONFIRMED/SUSPECTED findings from the
gpu_fuzzy_trader OOS audit (long 58.21%→49.92%→22.22%, short
60.57%→50.47%→15.11% collapse on 2026-07-07 run).
Current phase: task-8 implementer dispatching
Current task: task-8 (DRAFTED; spec committed, implementer dispatching)
  - task-1 MERGED (per-epoch window rotation)
  - task-2 MERGED (4th NSGA objective)
  - task-3 MERGED (RB walk-forward)
  - task-4 MERGED (monthly gate on val)
  - task-5 MERGED (dead f3 branch)
  - task-6 MERGED (overfit ratio gate)
  - task-7 MERGED (PF floor split)
  - task-8 in flight (use val_df in Phase 1 sign consistency)
base_branch: main
feature_branch: feature/task-8-sign-consistency-val
branch_policy: isolated
execution_mode: checkpoint
Pending blockers: none
Next action: dispatch implementer subagent for task-8; await handoff
JSON at .opencode/handoffs/task-8-implementer.json; then spec-review
then code-review before user merges and proceeds to task-9
(rename migration to "sequential chain" in logs — audit fix #6).
