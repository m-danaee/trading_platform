# PLAN: OOS Failure Diagnosis Fixes

## Goal
Stop shipping overfit single-symbol RB strategies that fail OOS. Fix three
code/design bugs (fail-open fallback, soft min-symbols, stale train/val
equity) and tighten config against validation reuse + thin pools.

Source diagnosis: `~/.cursor/plans/oos_failure_diagnosis_c17359ce.plan.md`
(verified against `rb_governor.py`, `phase5_oos.py`, `config.py` 2026-07-10).

Do NOT touch `evaluator_v5.ipynb` (per AGENTS.md). Do not run the full
pipeline locally (OOM). Validate with targeted unit tests only.

## Context
- Active objective: fail closed on non-positive-good / single-symbol teams;
  make Phase 5 reports match the evaluated strategy; reduce train/val reuse
  and improve Phase2→RB pool quality.
- base_branch: main
- branch_policy: isolated
- execution_mode: continuous (user override 2026-07-10: merge+continue all tasks)

## Execution order

### task-1: Fail-closed RB fallback (P0) — DONE (merged)
- `RB_ALLOW_FALLBACK=False`; empty positive-good → empty rules_set;
  loadable when deployment_accepted=False.

### task-2: Hard-require min distinct symbols on final output (P0)
- After compose/risk, if `RB_REQUIRE_SYMBOL_FILTERS` and
  `len(_symbols_in_rules(opt_rules)) < RB_MIN_DISTINCT_SYMBOLS`: fail closed
  (empty rules_set, deployment_accepted=false, reason
  `insufficient_distinct_symbols`). Skip gate when filters disabled.
- Targets: `rb_governor.py`, tests.
- Acceptance: 1-rule symbol team rejected when min=5; multi-symbol OK;
  gate skipped when filters off.

### task-3: Phase 5 train/val equity curves (P2)
- In `phase5_oos.py`, plot train + validation equity (same pattern as test).
- Targets: `phase5_oos.py`, `tests/unit/test_phase5_oos.py`.
- Acceptance: plot_equity_curve called for train, validation, test.

### task-4: Config anti-leak (P0 design)
- `PHASE2_JOINT_TRAIN_VAL=False` (keep `PHASE2_VAL_IN_FITNESS_PENALTY=True`).
- `RB_REQUIRE_SYMBOL_FILTERS=False`.
- Keep `PHASE2_DIVERSITY_ON_F4=True`.
- Targets: `config.py`, tests that pin old defaults.

### task-5: Phase2→RB pool quality knobs (P1)
- `PHASE2_RETURN_FLOOR_PCT=1.0` (was 0.0).
- `PHASE2_MONTHLY_ADMISSION_MIN_RATIO=0.50` (was 0.55).
- Targets: `config.py`, tests that pin old values.

## Out of scope
- Nested holdout, profit amplifier, full pipeline, evaluator_v5, TP/SL retune.

## Verification
```bash
source .venv/bin/activate
PYTEST_LOW_MEMORY=1 PYTHONPATH=. python -m pytest <related> -q
```
