# task-4: Config anti-leak defaults (P0 design)

## Goal
Reduce optimizer leakage: stop dual-using validation in Phase 2 joint
fitness while RB also selects on val; stop single-symbol specialization
until multi-symbol teams work.

## Scope — config.py defaults only (+ fix tests that pin old values)

1. Set `PHASE2_JOINT_TRAIN_VAL = False`
   - Keep `PHASE2_VAL_IN_FITNESS_PENALTY = True` so val penalties still
     regularize without full joint fitness.
2. Set `RB_REQUIRE_SYMBOL_FILTERS = False`
   - Document: re-enable only after multi-symbol specialized composition
     works with task-2 gate + thick pool.
3. Keep `PHASE2_DIVERSITY_ON_F4 = True` (no change if already True).
4. Update config comments to explain anti-overfit rationale.
5. Fix any unit tests that hard-assert the old True defaults.

## Out of scope
- Algorithm changes, pool knobs (task-5), evaluator_v5, pipeline run

## Acceptance
- [ ] Defaults match above
- [ ] config.py bottom asserts still hold
- [ ] Related tests updated and green
- [ ] Commit on feature/task-4-config-anti-leak only

## Branch
- base: main
- feature: feature/task-4-config-anti-leak

## Verification
```bash
source .venv/bin/activate
PYTHONPATH=. PYTEST_LOW_MEMORY=1 python -m pytest tests/unit/ -k "config or joint or symbol_filter or RB_REQUIRE" -q
# also any test files that fail due to default change
```
