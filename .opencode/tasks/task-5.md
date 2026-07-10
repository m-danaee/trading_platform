# task-5: Phase2→RB pool quality knobs (P1)

## Goal
Slightly improve Phase 2 pool quality/size so RB has more robust
cross-symbol candidates to compose (after fail-closed gates are on).

## Scope — config.py defaults only (+ fix tests that pin old values)

1. `PHASE2_RETURN_FLOOR_PCT = 1.0` (was 0.0) — restore >0 floor
2. `PHASE2_MONTHLY_ADMISSION_MIN_RATIO = 0.50` (was 0.55) — slightly ease
3. Keep `PHASE2_MONTHLY_ADMISSION_MIN_MONTHS = 3`
4. Document intent: grow robust pool for RB; do not chase OOS via TP/SL
5. Fix any tests hard-coding old 0.0 / 0.55 values
6. Optionally fix pre-existing `test_rb_compose_gates.py` expectation
   `RB_MAX_PAIR_OVERLAP == 0.35` → match actual config (0.25) if still failing
   (tiny drive-by only if one-line assert; improves suite health)

## Out of scope
- Algorithm changes, evaluator_v5, full pipeline, TP/SL retune

## Acceptance
- [ ] Defaults as above
- [ ] Related tests green
- [ ] Commit on feature/task-5-pool-quality only

## Branch
- base: main
- feature: feature/task-5-pool-quality

## Verification
```bash
source .venv/bin/activate
PYTHONPATH=. PYTEST_LOW_MEMORY=1 python -m pytest tests/unit/ -k "return_floor or monthly or compose_gates or anti_overfit" -q
```
