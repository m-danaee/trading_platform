# Nexus Context

Active objective: Fix the 13 CONFIRMED/SUSPECTED findings from the
gpu_fuzzy_trader OOS audit (long 58.21%→49.92%→22.22%, short
60.57%→50.47%→15.11% collapse on 2026-07-07 run). Each fix mapped
1:1 to an audit finding; full task list in `.opencode/plans/PLAN.md`.
Current phase: planning (plan written, awaiting execution start)
base_branch: main
branch_policy: isolated (one feature branch per task)
execution_mode: checkpoint (Nexus stops after each task for review)
Pending blockers: none
Next action: user confirms "execute task-1" to dispatch the
implementer subagent for the highest-leverage fix (per-epoch
window rotation, fixes audit finding #1).
