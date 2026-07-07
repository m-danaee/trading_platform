# Task 5: Delete (or Guard) the Dead f3 Profit-Factor Branch

## Task ID
`task-5` (fifth of 12 tasks in the 2026-07-07 audit fix plan)

## Title
Delete (or Guard) the Dead f3 Profit-Factor Branch

## Goal
Fix audit finding #5: the f3 objective calculation has a dead-code
branch (lines ~806-830 in `phase2_rule_pool.py`) that computes
`f3_val` from `PHASE2_F3_OBJECTIVE` (one of `profit_factor`,
`cv_fold_min`, `win_rate`), but the immediately-following block
(`if _cfg.PHASE2_USE_TOTAL_RETURN_OBJ:` at line ~834) overwrites
`f3_val` with `robust_return_pct(...)` whenever
`PHASE2_USE_TOTAL_RETURN_OBJ=True` (the default since Task 3). The
dead branch misleads anyone reading the code into thinking f3 is not
val-blended (it IS — via `_joint_primary_metric` min(train, val)).

## Audit Citation
- Confirmed by static inspection:
  - `phases/phase2_rule_pool.py:806-830` — the f3_objective switch
  - `phases/phase2_rule_pool.py:832-840` — the `PHASE2_USE_TOTAL_RETURN_OBJ`
    block that overwrites f3_val
  - `config.py:507-525` — `PHASE2_USE_TOTAL_RETURN_OBJ = True` (default
    since Task 3, was False before)
- The val-blend is real: `phases/phase2_support.py:371-381
  _joint_primary_metric` returns `min(train_val, val_val)` when joint.
- Behavior at runtime: the f3_objective switch computes f3_val, but
  the `if _cfg.PHASE2_USE_TOTAL_RETURN_OBJ:` block overwrites it on
  the next ~10 lines. The switch is dead code.

## Target Files
- `gpu_fuzzy_trader/phases/phase2_rule_pool.py`
  - Wrap the f3_objective switch (lines ~806-830) in
    `if not _cfg.PHASE2_USE_TOTAL_RETURN_OBJ:` so the legacy path
    is only taken when the user explicitly opts in.
  - Update the comment on line ~806 to reference audit finding #5.
  - Update the comment on line ~832 to clarify the precedence
    (USE_TOTAL_RETURN_OBJ wins when True).
  - Keep the `PHASE2_F3_OBJECTIVE` config flag for the legacy path
    (do NOT delete the config — that's a separate decision).
- `tests/unit/test_phase2_rule_pool.py`
  - Add a parametrized test that exercises both code paths:
    - `PHASE2_USE_TOTAL_RETURN_OBJ=True` (default): f3_val is
      `robust_return_pct(...)` (the val-blended min).
    - `PHASE2_USE_TOTAL_RETURN_OBJ=False` AND
      `PHASE2_F3_OBJECTIVE="profit_factor"`: f3_val is
      `min(profit_factor, val_profit_factor)` when JOINT.
    - `PHASE2_USE_TOTAL_RETURN_OBJ=False` AND
      `PHASE2_F3_OBJECTIVE="win_rate"`: f3_val is win_rate.
    - `PHASE2_USE_TOTAL_RETURN_OBJ=False` AND
      `PHASE2_F3_OBJECTIVE="cv_fold_min"`: f3_val is
      `min(_cv_fold_returns)` when available, else profit_factor.

## Current Behavior
- Line ~806: `# H2: f3_val based on PHASE2_F3_OBJECTIVE` (this comment
  is misleading — f3_val is overwritten 10 lines later when
  USE_TOTAL_RETURN_OBJ=True).
- Line ~807: `f3_objective = str(getattr(_cfg, "PHASE2_F3_OBJECTIVE", "profit_factor"))`.
- Line ~808-830: switch on `f3_objective` (cv_fold_min / profit_factor / win_rate).
- Line ~832: `# PHASE2_USE_TOTAL_RETURN_OBJ (now default True): f3 uses robust return`.
- Line ~834: `if _cfg.PHASE2_USE_TOTAL_RETURN_OBJ: ... f3_val = robust_return_pct(...)`.
- At runtime (with `USE_TOTAL_RETURN_OBJ=True`): the switch computes
  f3_val, then it's immediately overwritten. The switch is dead code.

## Scope
1. **Wrap the dead branch** (phase2_rule_pool.py:806-830):
   - Add `if not _cfg.PHASE2_USE_TOTAL_RETURN_OBJ:` at the top of the
     f3_objective switch (line ~807). Indent the switch body.
   - Add `# → fixes audit finding #5 (dead f3 profit_factor branch;
     USE_TOTAL_RETURN_OBJ takes precedence when True)` in the
     comment block above.
   - Update the comment on line ~832 to clarify:
     `PHASE2_USE_TOTAL_RETURN_OBJ=True (default): f3 uses robust return
     (overrides PHASE2_F3_OBJECTIVE setting). Set to False to opt into
     the legacy PHASE2_F3_OBJECTIVE path (profit_factor/cv_fold_min/win_rate).`
2. **Add parametrized tests** in `tests/unit/test_phase2_rule_pool.py`:
   - Cover both `USE_TOTAL_RETURN_OBJ=True` and `False` paths.
   - For `False`, cover all three `F3_OBJECTIVE` values.
   - Confirm f3_val is the val-blended robust return when True, and
     the legacy value when False.
3. **Do NOT change**:
   - The `PHASE2_F3_OBJECTIVE` config flag (keep it for backward
     compat with the legacy path).
   - The `PHASE2_USE_TOTAL_RETURN_OBJ` config flag (keep it as the
     master switch).
   - The `_joint_primary_metric` helper (correct as-is).
   - Any other file outside `phases/phase2_rule_pool.py` and the test.

## Acceptance Criteria
1. The f3_objective switch in `phase2_rule_pool.py:806-830` is
   wrapped in `if not _cfg.PHASE2_USE_TOTAL_RETURN_OBJ:`.
2. A code comment references "audit finding #5" near the wrapped branch.
3. A parametrized test covers the 4 cases:
   - `USE_TOTAL_RETURN_OBJ=True` (default) → robust return
   - `USE_TOTAL_RETURN_OBJ=False, F3_OBJECTIVE="profit_factor"` → min(PF, val_PF)
   - `USE_TOTAL_RETURN_OBJ=False, F3_OBJECTIVE="win_rate"` → win_rate
   - `USE_TOTAL_RETURN_OBJ=False, F3_OBJECTIVE="cv_fold_min"` → min(cv_folds) or PF fallback
4. Default config behavior (USE_TOTAL_RETURN_OBJ=True) is bit-identical
   to pre-task-5.
5. Legacy opt-in (`USE_TOTAL_RETURN_OBJ=False` + `F3_OBJECTIVE` choice)
   still works (regression guard for any user who has set these).
6. All existing tests pass: `test_phase2_rule_pool.py`,
   `test_phase2_monthly_admission.py`, `test_phase2_window_rotation.py`,
   `test_phase2_island_scheduler.py`, `test_evox_runner.py`, etc.

## Verification
Run only related unit tests with `PYTEST_LOW_MEMORY=1` and `.venv`:

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_rule_pool.py -q
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_monthly_admission.py -q
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_window_rotation.py -q
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_island_scheduler.py -q
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_evox_runner.py -q
```

## Notes
- Do NOT modify `evaluator_v5.ipynb`.
- Do NOT run the full project or full test suite locally (OOM risk
  per AGENTS.md).
- This is the smallest diff in the plan (~10 lines of source code,
  1 test file). Keep it minimal.
- The `cv_fold_min` branch is technically reachable (via the legacy
  opt-in path), so the parametrized test must include it to prove the
  legacy path actually works.
- The `win_rate` branch is even more legacy (the docstring at line
  820 says "win_rate — f3 = -win_rate (degenerate, not recommended)").
  But it's still reachable via the legacy opt-in, so the test should
  cover it.
- This task is the cleanup twin of task-4 (monthly gate on val).
  Both are small, focused fixes with no behavior change in default
  config.
