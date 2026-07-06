# Task 3: Decouple Phase 2 Objectives And Prefer Robust Return For OOS

## Task ID
`task-3`

## Title
Decouple Phase 2 Objectives And Prefer Robust Return For OOS

## Goal
Reduce NSGA-III Pareto collapse caused by nearly identical objective penalties and align Phase 2 optimization more directly with OOS/robust return. The current run log repeatedly shows `objective_corr_f1_f3=1.00`, tiny Pareto fronts, and very low valid-rule counts, indicating that shared penalties dominate the objective surface.

## Target Files
- `gpu_fuzzy_trader/phases/phase2_rule_pool.py`
- `gpu_fuzzy_trader/phases/phase2_support.py`
- `gpu_fuzzy_trader/config.py`
- Related tests under `tests/unit/test_phase2_rule_pool.py`, `tests/unit/test_phase2_support.py`, and any existing objective/config tests.

## Evidence From Run Log / Analysis
- Repeated warnings: `objective_corr_f1_f3=1.00`, `objective_corr_f1_f2=-1.00`, `objective_corr_f2_f3=-1.00`.
- Current objective construction adds many same-signed shared penalties to all objectives:
  - support penalty,
  - diversity penalty,
  - trade penalty,
  - overfit gap penalty.
- `PHASE2_USE_TOTAL_RETURN_OBJ=False` with `PHASE2_F3_OBJECTIVE="profit_factor"` means f3 is not directly optimizing robust return, despite OOS return being the main user goal.
- PF should remain a quality/feasibility gate, but robust return should be a first-class objective signal for OOS.

## Scope
- Reduce shared penalty dominance across `f1`, `f2`, and `f3` so the objectives encode distinct trade-offs.
- Prefer robust return as the default f3 signal for OOS-focused mode when joint train+val metrics exist.
- Keep profit factor as a quality gate or penalty; do not remove PF floor/admission behavior.
- Update config defaults/comments if the intended default changes from PF-first to robust-return-first.
- Add/adjust tests proving:
  - f3 can use robust return in configured OOS-focused mode,
  - PF floor still affects feasibility/penalty behavior,
  - penalties are not identically applied to all objectives.
- Keep final pool/admission gates conceptually intact unless tests and comments are updated to reflect a deliberate change.

## Non-Goals
- Do not modify `evaluator_v5.ipynb`.
- Do not change sampling/migration semantics; that is Task 4.
- Do not broadly relax all feasibility gates without evidence; if relaxation is needed, keep it targeted and tested.
- Do not run full pipeline or full tests locally.

## Acceptance Criteria
- Objective calculation tests show penalties are not identically added to all objectives.
- f3 can use robust return in the configured OOS-focused mode.
- PF floor still affects feasibility/admission behavior.
- Existing tests for Phase 2 objective/gate behavior are updated and pass.
- No evaluator notebook changes.

## Verification
Run only related tests, for example:

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_rule_pool.py tests/unit/test_phase2_support.py tests/unit/test_config_additions.py -q
```

If narrower or additional related tests are needed, use them and report exactly what ran.
