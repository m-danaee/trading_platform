# Task 6: Hard Overfit Ratio Gate + Raise Penalty Weight

## Task ID
`task-6` (sixth of 12 tasks in the 2026-07-07 audit fix plan)

## Title
Hard Overfit Ratio Gate + Raise Penalty Weight

## Goal
Fix audit finding #7: the soft overfit-gap penalty
`(gap_pct - 8) × 5` is too weak relative to the return signal.
With train=76% / val=7.6% (gap=68.4pp), the penalty adds 302 to
f3, but a "clean" rule (train=20%, val=15%, gap=5pp) gets
penalty=0 and dominates. The outlier rule still wins the Pareto
front on f1 (Sortino) because nothing else beats its train-side
Sortino. The hard gate `PHASE2_MAX_TRAIN_VAL_GAP_PCT=16.0pp` only
checks absolute pp, not ratio — a 10× ratio with 7.6% val passes
through.

## Audit Citation
- Confirmed by static inspection:
  - `phases/phase2_rule_pool.py:912-921` — overfit_gap_penalty calc.
  - `phases/phase2_support.py:177-179` — pool admission uses
    absolute pp gap (`train_ret - val_ret > max_gap`), not ratio.
  - `config.py:626-650` — `PHASE2_MAX_TRAIN_VAL_GAP_PCT=16.0`,
    `PHASE2_OVERFIT_GAP_PENALTY_WEIGHT=5.0`,
    `PHASE2_OVERFIT_GAP_PCT_THRESHOLD=8.0`.
- Run log evidence (2026-07-07): `max_train_val_gap_ratio` reaches
  2.5-10.3× persistently across clusters. The 10×-ratio case with
  val=7.6% passes the absolute gap gate (gap=68pp > 16pp should
  reject... but this gate runs at pool admission, and rules
  still entered the pool because the monthly gate (task #4) was
  a no-op). After task-4 merges, this will be partially fixed
  by the monthly-on-val gate, but a hard ratio gate is still
  needed for cases the monthly gate misses.

## Target Files
- `gpu_fuzzy_trader/phases/phase2_support.py`
  - Add a 10th gate in `_passes_pool_admission_impl` (after the
    existing 9+1 gates) for the overfit ratio.
  - Add the same gate to `_feasibility_gate_failures` (the
    diagnostic mirror).
- `gpu_fuzzy_trader/config.py`
  - Add `PHASE2_OVERFIT_RATIO_FLOOR = 3.0` (default; reject if
    `train_return / max(val_return, 0.1) > 3.0`).
  - Raise `PHASE2_OVERFIT_GAP_PENALTY_WEIGHT` from 5.0 → 15.0.
  - Update the comment to reference audit finding #7.
- `tests/unit/test_phase2_rule_pool.py`
  - Add a test asserting the ratio gate rejects a rule with
    train=30% / val=5% (6× ratio, was admitted before).
  - Add a test asserting a rule with train=15% / val=10% (1.5×
    ratio) is admitted.
- `tests/unit/test_phase2_support.py`
  - Add a test asserting `_feasibility_gate_failures` includes
    the `overfit_ratio` key when the ratio gate fires.

## Current Behavior
- `phases/phase2_rule_pool.py:912-921`:
  ```python
  overfit_gap_penalty = 0.0
  if val_metrics is not None and PHASE2_OVERFIT_GAP_PENALTY_WEIGHT > 0 and (...):
      train_ret = float(metrics.get("total_return_pct", 0.0))
      val_ret = float(val_metrics.get("total_return_pct", 0.0))
      gap_pct = train_ret - val_ret
      gap_threshold = float(PHASE2_OVERFIT_GAP_PCT_THRESHOLD)  # 8.0
      if gap_pct > gap_threshold:
          overfit_gap_penalty = (gap_pct - gap_threshold) * weight  # weight=5
  ```
  Penalty adds to f3 but is dominated by the return signal.
- `phases/phase2_support.py:177-179` (pool admission):
  ```python
  max_gap = float(PHASE2_MAX_TRAIN_VAL_GAP_PCT)  # 16.0
  if train_ret - val_ret > max_gap:
      return False
  ```
  Only checks absolute pp, not ratio.
- `config.py:641`: `PHASE2_OVERFIT_GAP_PENALTY_WEIGHT = 5.0`

## Scope
1. **Add hard ratio gate** (phase2_support.py):
   - In `_passes_pool_admission_impl` (after the existing absolute
     pp gap gate at line 177-179), add:
     ```python
     max_ratio = float(getattr(_cfg, "PHASE2_OVERFIT_RATIO_FLOOR", 3.0))
     val_ret_safe = max(val_ret, 0.1)  # avoid div-by-near-zero
     if max_ratio > 0 and train_ret / val_ret_safe > max_ratio:
         return False
     ```
   - In `_feasibility_gate_failures` (the diagnostic mirror), add
     an `overfit_ratio` key that returns 1 if the gate fires.
   - Update the docstring "The 9 gates mirror..." to "The 10 gates
     mirror..." (or 11 if you count the f4 gate from task-2).
2. **Raise penalty weight** (config.py):
   - Change `PHASE2_OVERFIT_GAP_PENALTY_WEIGHT` from 5.0 to 15.0.
   - Update the comment to reference audit finding #7 and explain
     the 3× raise rationale.
3. **Add new config flag** (config.py):
   - Add `PHASE2_OVERFIT_RATIO_FLOOR = 3.0` with a clear docstring.
4. **Regression guard**:
   - `PHASE2_OVERFIT_RATIO_FLOOR = 0.0` or `= float('inf')` must
     preserve pre-task-6 behavior (no ratio gate).
   - The default `= 3.0` enables the new gate.
5. **Add audit-finding linkage**:
   - Add `# → fixes audit finding #7 (overfit-gap penalty too weak
     vs return signal; absolute-pp gate missed high-ratio cases)`
     in the comment block at the gate site.
6. **Do NOT change**:
   - The existing absolute-pp gate (`PHASE2_MAX_TRAIN_VAL_GAP_PCT=16.0`).
     The two gates are complementary: absolute pp catches huge gaps;
     ratio catches smaller gaps that are still suspect.
   - The existing `PHASE2_OVERFIT_GAP_PCT_THRESHOLD=8.0` (the soft
     penalty threshold).
   - Any other file outside `phases/phase2_support.py`, `config.py`,
     and the test files.

## Acceptance Criteria
1. A rule with train=30% / val=5% (6× ratio) is REJECTED at pool
   admission (was admitted previously because gap=25pp > 16pp but
   the gate at line 177-179 only triggers when gap > 16pp; with
   25pp it should fire; but the 10× case with val=7.6% has gap=68pp
   which DOES fire the absolute gate; the test should target a
   case where absolute pp is small but ratio is high, e.g.,
   train=15% / val=4% (3.75× ratio, gap=11pp < 16pp, ratio
   gate should fire).
2. A rule with train=15% / val=10% (1.5× ratio) is ADMITTED.
3. A rule with train=20% / val=8% (2.5× ratio) is ADMITTED (under
   the 3.0 default).
4. `_feasibility_gate_failures` includes the `overfit_ratio` key
   when the ratio gate fires.
5. The soft overfit-gap penalty is now 3× stronger (weight=15.0
   vs 5.0); the same rule gets a penalty of (gap-8)×15 instead
   of (gap-8)×5.
6. With `PHASE2_OVERFIT_RATIO_FLOOR = 0.0`, behavior matches
   pre-task-6 (regression guard).
7. All existing tests pass: `test_phase2_rule_pool.py`,
   `test_phase2_support.py`, `test_phase2_monthly_admission.py`,
   `test_phase2_window_rotation.py`, `test_phase2_island_scheduler.py`,
   `test_evox_runner.py`, etc.

## Verification
Run only related unit tests with `PYTEST_LOW_MEMORY=1` and `.venv`:

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_rule_pool.py -q
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_support.py -q
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_monthly_admission.py -q
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_window_rotation.py -q
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_island_scheduler.py -q
```

## Notes
- Do NOT modify `evaluator_v5.ipynb`.
- Do NOT run the full project or full test suite locally (OOM risk
  per AGENTS.md).
- The 3× penalty weight raise is a heuristic; the OOS gain will
  be validated by the user's next Colab run (post all tasks).
- The ratio gate is a hard reject at pool admission; the soft
  penalty applies to all rules (in evolution) regardless of
  pool admission. They serve different purposes.
- This is a small surgical fix (similar in scope to task-4 and
  task-5). Keep the diff minimal.
- The ratio gate and the absolute-pp gate are complementary;
  do NOT remove the absolute-pp gate.
- The `val_ret_safe = max(val_ret, 0.1)` guards against
  division-by-near-zero when val_ret is tiny positive.
