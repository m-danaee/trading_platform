# Nexus Context

**Updated:** 2026-07-04
**base_branch:** `main`
**branch_policy:** isolated
**execution_mode:** direct
**status:** COMPLETED — Phase 2 Gen-4/5 Return Blow-Up Fix

## Completed Plan
**Plan: Diagnose & Fix Phase 2 Gen-4/5 Return Blow-Up** (`.opencode/plans/PLAN.md`)

| Task | Branch | Status |
|------|--------|--------|
| Task 1: Cap time-exit return | `fix/time-exit-return-cap` | ✅ Spec+Code approved (`46cb88a`) |
| Task 2: GPU entry-price guard | `fix/gpu-entry-price-guard` | ✅ Spec approved + Code fix (`f451216`) |
| Task 3: Tighten trade-support penalty | `fix/tighten-trade-support-penalty` | ✅ Spec+Code approved (`466610b`) |
| Task 4: Overfit gap in logs | `fix/overfit-gap-logging` | ✅ Spec+Code approved (`8556e8d`) |
| Task 5: Extended tests | Not implemented | ℹ️ Core test coverage done by Tasks 1-4; remaining property-test updates are low-severity follow-up |

## Previous Active Task (separate)
- **Task 24** 🔄 in progress — Fix `_sample_df` stride sampling bug on branch `feature/fix-contiguous-sampling`

## Next Steps
- Merge all 4 branches to `main` (user confirmation required)
- Run a Colab pipeline job to verify gen-4/5 return spikes are eliminated
- Optional follow-up: switch NSGA-III primary return objective from raw train-only return to `robust_return` (min(train, val))
