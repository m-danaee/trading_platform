# task-3: Phase 5 train/val equity curves (P2)

## Goal
Regenerate train and validation equity PNGs in Phase 5 so reports match
the evaluated strategy (RB path bypasses Phase 3 which previously wrote
stale train/val equity).

## Scope

### phase5_oos.py
After evaluating train/validation/test, call `reporter.plot_equity_curve`
for **train** and **validation** using the same non-fatal try/except
pattern already used for test (around the existing test plot call).

Use trade_logs_by_split.get("train") and .get("validation").

### Tests
- Extend tests/unit/test_phase5_oos.py
- Mock Reporter; assert plot_equity_curve called with train, validation, and test
- Empty trade log must not crash

## Out of scope
- Config changes (task-4/5)
- evaluator_v5, full pipeline, outputs/ commits

## Acceptance
- [ ] Phase 5 invokes plot for train, validation, and test
- [ ] No crash on empty trade logs
- [ ] Tests pass under .venv + PYTEST_LOW_MEMORY=1 + PYTHONPATH=.
- [ ] Commit on feature/task-3-phase5-equity only

## Branch
- base: main
- feature: feature/task-3-phase5-equity
