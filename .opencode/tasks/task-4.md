# Task 4: Monthly Admission Gate Runs on Val, Not Train

## Task ID
`task-4` (fourth of 12 tasks in the 2026-07-07 audit fix plan)

## Title
Monthly Admission Gate Runs on Val, Not Train

## Goal
Fix audit finding #4 (monthly admission gate runs on train windows
— near no-op; filters 0-8 of 75 rules per log). The gate was designed
to catch regime-shift rules that bleed on test, but testing on train
months cannot detect that. Switch the gate's source DataFrame from
`self._train_df` (line 2836) and `self._train_df` (line 3316) to
`self._scoped_val_df` so the gate actually exercises out-of-sample
stability.

## Audit Citation
- Confirmed by static inspection:
  - `phases/phase2_rule_pool.py:2836` and `:3316` both call
    `build_monthly_windows(self._train_df)`.
  - `phases/phase2_rule_pool.py:2079 _evaluate_rule_on_window`
    needs the window df to have label and meta columns — `self._train_df`
    has these (so the current call works), but the gate is meaningless
    on train.
- Run log evidence (2026-07-07): "monthly-admission gate 75 → 75 rules"
  (long first call), "75 → 75 rules" (long second call), "75 → 67 rules"
  (long third call), "75 → 75 rules", "75 → 72 rules", "75 → 74 rules"
  (short calls) — total filtering is 0/0/8/0/3/1 of 75. The gate is
  not catching anything useful.

## Target Files
- `gpu_fuzzy_trader/phases/phase2_rule_pool.py`
  - Line 2836: change `self._train_df` → `self._scoped_val_df`
  - Line 3316: change `self._train_df` → `self._scoped_val_df`
  - Update the surrounding comment to be accurate (it currently
    says "monthly rolling windows in the train split" — that's the
    bug we're fixing).
  - Update the warning log message to say "val split" instead of
    "monthly-admission gate" implicit context.
- `tests/unit/test_phase2_monthly_admission.py`
  - Add a test that the gate is called with the val DataFrame,
    not the train DataFrame.
  - The test should monkeypatch `build_monthly_windows` to record
    the DataFrame it was called with and assert it's the val df
    (using a marker column to distinguish train vs val).

## Current Behavior
- Line 2836 (within `_finalize_pool_after_evolution` or similar):
  ```python
  if _cfg.PHASE2_MONTHLY_ADMISSION_ENABLED:
      monthly_windows = build_monthly_windows(self._train_df)
  ```
- Line 3316 (within `_finalize_pool_with_admission` or similar — the
  late-stage finalize for the RB-bound pool): same call on train.
- `self._train_df` is the slimmed train DataFrame (features only —
  but `build_monthly_windows` likely doesn't need features; it needs
  datetime and label columns for profitability computation).
- `self._scoped_val_df` is the val DataFrame passed in via
  `__init__(val_df=val_df)` and stored at line 2293. It has
  datetime, labels, and meta columns.

## Scope
1. **Fix the two call sites** (phase2_rule_pool.py:2836, :3316):
   - Change `self._train_df` → `self._scoped_val_df` in both places.
   - Update the docstring/comment that says "monthly rolling windows
     in the train split" to say "val split" or "out-of-sample
     stability check".
2. **Defensive guard**:
   - If `self._scoped_val_df is None` (degenerate case where no val
     was provided at init), the gate should be skipped with a warning,
     not crash.
3. **Audit-finding linkage**:
   - Add `# → fixes audit finding #4 (monthly gate on train was no-op)`
     in the comment block at the two call sites.
4. **Do NOT change**:
   - The `_apply_monthly_admission_gate` function (it already takes
     pre-built windows; the fix is upstream in the call site).
   - The `_evaluate_rule_on_window` helper (it correctly takes any
     df with label/meta columns).
   - The `build_monthly_windows` function in
     `validation/monthly_windows.py` (it's a generic splitter).
   - Any other file outside `phases/phase2_rule_pool.py` and the test.

## Acceptance Criteria
1. The two call sites in `phase2_rule_pool.py:2836` and `:3316` use
   `self._scoped_val_df` (not `self._train_df`).
2. A new test in `test_phase2_monthly_admission.py` confirms that
   `build_monthly_windows` is called with the val DataFrame (using
   a marker column to distinguish).
3. With `PHASE2_MONTHLY_ADMISSION_ENABLED=True`, the log line
   "monthly-admission gate N → M rules" should now show non-trivial
   filtering (e.g., 75 → 30+ rules on val vs the prior 0-8 on train)
   — verifiable in the test by counting rejected rules.
4. If `self._scoped_val_df is None`, a warning is logged and the
   gate is skipped (no crash).
5. The surrounding comment/docstring accurately describes the
   behavior (val, not train).
6. All existing tests pass: `test_phase2_monthly_admission.py`,
   `test_phase2_rule_pool.py`, `test_phase2_window_rotation.py`,
   `test_phase2_island_scheduler.py`, etc.

## Verification
Run only related unit tests with `PYTEST_LOW_MEMORY=1` and `.venv`:

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_monthly_admission.py -q
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_rule_pool.py -q
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_window_rotation.py -q
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_island_scheduler.py -q
```

## Notes
- Do NOT modify `evaluator_v5.ipynb`.
- Do NOT run the full project or full test suite locally (OOM risk
  per AGENTS.md).
- This is a small surgical fix (2 lines of code + 1 test + comment
  updates). Keep the diff minimal.
- If `self._scoped_val_df` is not in the expected format (e.g.,
  missing the `datetime` column), the gate's `build_monthly_windows`
  call may fail — handle this gracefully with a warning.
- The change should NOT affect the `PHASE2_MONTHLY_ADMISSION_ENABLED`
  flag semantics (still gates whether the gate runs at all).
- This task is intentionally small to keep the review diff tight.
  Task-5 (delete dead f3 branch) and task-6 (overfit ratio gate) will
  follow in the same cleanup spirit.
