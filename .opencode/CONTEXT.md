# Nexus Context

Active objective: Fix the 13 CONFIRMED/SUSPECTED findings from the
gpu_fuzzy_trader OOS audit (long 58.21%→49.92%→22.22%, short
60.57%→50.47%→15.11% collapse on 2026-07-07 run).
Current phase: task-7 implementer dispatching
Current task: task-7 (DRAFTED; spec committed, implementer dispatching)
  - task-1 MERGED (per-epoch window rotation)
  - task-2 MERGED (4th NSGA objective)
  - task-3 MERGED (RB walk-forward)
  - task-4 MERGED (monthly gate on val)
  - task-5 MERGED (dead f3 branch)
  - task-6 MERGED (overfit ratio gate)
  - task-7 in flight (PF floor split — 1.05 evolution, 1.15 admission)
base_branch: main
feature_branch: feature/task-7-pf-floor-split
branch_policy: isolated
execution_mode: checkpoint
Pending blockers: none
Next action: dispatch implementer subagent for task-7; await handoff
JSON at .opencode/handoffs/task-7-implementer.json; then spec-review
then code-review before user merges and proceeds to task-8
(use val_df in Phase 1 sign consistency filter).
