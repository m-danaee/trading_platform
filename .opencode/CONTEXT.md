# Nexus Context

Active objective: OOS failure diagnosis fixes — PLAN COMPLETE.

## Status: PLAN COMPLETE — all 5 tasks merged to main

### Workflow
- base_branch: main
- branch_policy: isolated
- execution_mode: continuous (user override)

### Merged tasks
| Task | Summary | Feature commits |
|------|---------|-----------------|
| task-1 | Fail-closed RB fallback (`RB_ALLOW_FALLBACK=False`) | 8e49383, 615f265 |
| task-2 | Hard-require min distinct symbols on final output | 3381927 |
| task-3 | Phase 5 train/val equity curves | 3ea7454 |
| task-4 | Config anti-leak: joint train/val off, symbol filters off | 841bf80 |
| task-5 | Pool quality: return floor 1.0, monthly ratio 0.50 | 81e606c |

### Key defaults now on main
- `RB_ALLOW_FALLBACK = False`
- `PHASE2_JOINT_TRAIN_VAL = False` (val penalties still on)
- `RB_REQUIRE_SYMBOL_FILTERS = False`
- `PHASE2_RETURN_FLOOR_PCT = 1.0`
- `PHASE2_MONTHLY_ADMISSION_MIN_RATIO = 0.50`
- Min-symbols gate + fail-closed empty strategies loadable when `deployment_accepted=False`
- Phase 5 plots train/validation/test equity

### Next action
User runs pipeline on Colab. Compare OOS vs prior 1-rule HHI=1.0 failure.
Do not run full project locally (OOM per AGENTS.md).

main is ahead of origin/main (local merges not pushed).
