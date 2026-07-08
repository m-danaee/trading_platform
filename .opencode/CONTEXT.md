# Nexus Context

Active objective: Fix the 13 CONFIRMED/SUSPECTED findings from the
gpu_fuzzy_trader OOS audit (long 58.21%→49.92%→22.22%, short
60.57%→50.47%→15.11% collapse on 2026-07-07 run).

## Status: PLAN COMPLETE

All 12 tasks merged to main, all 12 feature branches deleted.
Only the `main` branch remains. The user is running the project
on Colab to validate the OOS improvement.

## Summary of merged changes (12 tasks, 60+ commits)

### Phase 1 (data + features)
- (No Phase 1 changes; audit found no Phase 1 issues beyond #11)

### Phase 2 (rule evolution)
- **task-1**: per-epoch window rotation (`PHASE2_PER_EPOCH_WINDOW_ROTATION`)
- **task-2**: return-concentration as 4th NSGA objective (`PHASE2_F4_*`)
- **task-5**: dead f3 profit_factor branch guarded behind `USE_TOTAL_RETURN_OBJ`
- **task-7**: PF floor split (1.05 evolution, 1.15 admission)
- **task-8**: val_df actually used in Phase 1 sign consistency
- **task-10**: cache refresh gated on rotation flag
- **task-11**: `PHASE2_VAL_SIM_INTERVAL` 1 → 3 (~33% GPU savings)
- **task-12**: Pareto-collapse warning gated on pareto_size >= 5

### Phase 2 admission / monthly gates
- **task-4**: monthly admission gate runs on val, not train
- **task-6**: hard overfit ratio gate + raised penalty weight (5 → 15)
- **task-9**: removed dead migration helper + deprecated interval config

### RB Governor
- **task-3**: 2-fold walk-forward risk grid + PHASE4_TAIL_HOLDOUT_FRACTION wiring

### Test coverage
- 12 new test files created or extended
- ~60 new tests added
- All existing tests still pass (no regressions)

## Audit findings status (all 13)

| # | Finding | Status |
|---|---------|--------|
| 1 | Per-epoch resampling dead | ✅ FIXED (task-1) |
| 2 | Uncapped time-exit drives f3 | ✅ FIXED (task-2, f4) |
| 3 | RB risk-grid overfits val_selection | ✅ FIXED (task-3) |
| 4 | Monthly gate on train | ✅ FIXED (task-4) |
| 5 | Dead f3 profit_factor branch | ✅ FIXED (task-5) |
| 6 | Migration interval dead | ✅ FIXED (task-9) |
| 7 | Overfit-gap penalty too weak | ✅ FIXED (task-6) |
| 8 | Cache refresh wipes cache every epoch | ✅ FIXED (task-10) |
| 9 | Val PF floor too high | ✅ FIXED (task-7) |
| 10 | Val sim every gen is wasteful | ✅ FIXED (task-11) |
| 11 | val_df dead in sign consistency | ✅ FIXED (task-8) |
| 12 | PHASE4_TAIL_HOLDOUT_FRACTION orphan | ✅ FIXED (task-3) |
| 13 | Pareto collapse warning on 2 points | ✅ FIXED (task-12) |
| S1 | Profit amplifier dormant | ⏸️ DEFERRED (RB_PROFIT_AMPLIFIER_ENABLED=False) |

## Next action
User runs the project on Colab. Compare new test results against
the 2026-07-07 baseline:
- long: 22.22% return, 80 trades, 2.50% drawdown
- short: 15.11% return, 108 trades, 3.32% drawdown

If the new run shows reduced train→val→test gap (i.e., val and
test returns closer to train returns), the plan is validated.
If the new run shows worse OOS, rollback is straightforward:
each task is a self-contained commit on main.
