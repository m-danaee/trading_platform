# Nexus Context

Active objective: Fix the 13 CONFIRMED/SUSPECTED findings from the
gpu_fuzzy_trader OOS audit (long 58.21%→49.92%→22.22%, short
60.57%→50.47%→15.11% collapse on 2026-07-07 run).
Current phase: task-11 implementer dispatching
Current task: task-11 (DRAFTED; spec committed, implementer dispatching)
  - task-1..10 MERGED
  - task-11 in flight (raise PHASE2_VAL_SIM_INTERVAL 1 -> 3,
    audit fix #10, depends on task-1's fixed val window)
base_branch: main
feature_branch: feature/task-11-raise-val-interval
branch_policy: isolated
execution_mode: checkpoint
Pending blockers: none
Next action: dispatch implementer subagent for task-11; await
handoff JSON at .opencode/handoffs/task-11-implementer.json; then
spec-review then code-review before user merges and proceeds to
the final task-12 (gate Pareto-collapse warning to pareto_size
>= 5, audit fix #13).
