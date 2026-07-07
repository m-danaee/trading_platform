# Task 7: Lower Val PF Floor During Evolution (Split Floor Flags)

## Task ID
`task-7` (seventh of 12 tasks in the 2026-07-07 audit fix plan)

## Title
Lower Val PF Floor During Evolution (Split Floor Flags)

## Goal
Fix audit finding #9: feasibility collapse is val-driven, not
objective-design-driven. The val PF floor (currently
`PHASE2_PROFIT_FACTOR_FLOOR=1.15`) rejects 102-113/120 chromosomes
at gen 1 and 40-70/120 mid-run. Random rules rarely have val
PF > 1.15 with thin val trade counts (some clusters have
val_rows=95733 ≈ 32k/sym). This causes Pareto fronts of 2-10
rules and 5+ plateau restarts per cluster in the log. Split the
single floor into two: a permissive evolution floor (1.05) used
during NSGA-III fitness calc, and the original strict admission
floor (1.15) used at pool admission.

## Audit Citation
- Confirmed by static inspection:
  - `config.py:597-600` — `PHASE2_PROFIT_FACTOR_FLOOR = 1.15` (single
    floor used everywhere).
  - `phases/phase2_support.py:124` — `_pool_admission_floors` returns
    this floor for pool admission (correct use).
  - `phases/phase2_rule_pool.py:758-760, 779-780` — the floor is also
    used in the soft penalty during evolution (causes the collapse).
- Run log evidence (2026-07-07): `feasibility collapse breakdown`
  shows `val_pf_floor` rejecting 102-113/120 at gen 1, persisting
  40-70/120 throughout. Pareto fronts stay at 2-10 rules.

## Target Files
- `gpu_fuzzy_trader/config.py`
  - Replace `PHASE2_PROFIT_FACTOR_FLOOR = 1.15` with two flags:
    - `PHASE2_PROFIT_FACTOR_FLOOR_EVOLUTION = 1.05` (default; soft
      penalty threshold during NSGA-III)
    - `PHASE2_PROFIT_FACTOR_FLOOR_ADMISSION = 1.15` (default; hard
      gate at pool admission)
  - Keep `PHASE2_PROFIT_FACTOR_FLOOR` as a deprecated alias
    (`= PHASE2_PROFIT_FACTOR_FLOOR_ADMISSION`) for backward compat.
  - Add `# → fixes audit finding #9` comment.
- `gpu_fuzzy_trader/phases/phase2_support.py`
  - `_pool_admission_floors` should return the ADMISSION floor
    (1.15), not the EVOLUTION floor.
- `gpu_fuzzy_trader/phases/phase2_rule_pool.py`
  - Lines 758-760 (val floor penalty) and 779-780 (train floor
    penalty) should use the EVOLUTION floor (1.05).
  - The log message at line 2825 should reference the ADMISSION
    floor (1.15) since that's what pool admission uses.
- `tests/unit/test_phase2_rule_pool.py`
  - Add a test for `PHASE2_PROFIT_FACTOR_FLOOR_EVOLUTION=1.05`:
    a rule with train_pf=1.10, val_pf=1.10 gets a SMALLER penalty
    than under the old 1.15 floor.
  - Add a test for `PHASE2_PROFIT_FACTOR_FLOOR_ADMISSION=1.15`:
    pool admission still rejects val_pf < 1.15.
  - Add a regression test: `PHASE2_PROFIT_FACTOR_FLOOR` deprecated
    alias still works (returns 1.15 by default).
- `tests/unit/test_phase2_support.py`
  - Add a test asserting `_pool_admission_floors` returns the
    ADMISSION floor (1.15), not the EVOLUTION floor (1.05).

## Current Behavior
- `phases/phase2_support.py:124`: `_pool_admission_floors` returns
  `_cfg.PHASE2_PROFIT_FACTOR_FLOOR` (= 1.15) for use in
  `_passes_pool_admission_impl` (line 176: `if val_pf < pf_floor`).
- `phases/phase2_rule_pool.py:758-760`: in the soft penalty loop
  for val, uses `_cfg.PHASE2_PROFIT_FACTOR_FLOOR` (= 1.15). This is
  too high for evolution.
- `phases/phase2_rule_pool.py:779-780`: in the soft penalty loop
  for train, uses `_cfg.PHASE2_PROFIT_FACTOR_FLOOR` (= 1.15). Also
  too high for evolution.
- `phases/phase2_rule_pool.py:2825`: log message references
  `_cfg.PHASE2_PROFIT_FACTOR_FLOOR` (= 1.15) for the pool admission
  context. Correct.
- The collapse: at gen 1, val_pf rarely > 1.15 for random rules;
  most rules get rejected before they can be evaluated by NSGA-III.
  The search collapses.

## Scope
1. **Add new config flags** (config.py):
   - `PHASE2_PROFIT_FACTOR_FLOOR_EVOLUTION = 1.05` (lower, for the
     soft penalty during evolution).
   - `PHASE2_PROFIT_FACTOR_FLOOR_ADMISSION = 1.15` (kept, for the
     hard gate at pool admission).
   - `PHASE2_PROFIT_FACTOR_FLOOR = PHASE2_PROFIT_FACTOR_FLOOR_ADMISSION`
     (deprecated alias for backward compat — same value, so any
     direct use of the old flag still works).
2. **Update `_pool_admission_floors`** (phase2_support.py:118-125):
   - Return `PHASE2_PROFIT_FACTOR_FLOOR_ADMISSION` instead of
     `PHASE2_PROFIT_FACTOR_FLOOR` (so the pool admission uses 1.15
     even though evolution uses 1.05).
3. **Update evolution penalty** (phase2_rule_pool.py:758-780):
   - Replace `_cfg.PHASE2_PROFIT_FACTOR_FLOOR` with
     `_cfg.PHASE2_PROFIT_FACTOR_FLOOR_EVOLUTION` in the soft penalty
     loops (lines 758, 760, 779, 780).
4. **Update log message** (phase2_rule_pool.py:2825):
   - The log message about pool admission should reference the
     ADMISSION floor (1.15), not the EVOLUTION floor. Use
     `_cfg.PHASE2_PROFIT_FACTOR_FLOOR_ADMISSION` in the format string.
5. **Audit-finding linkage**:
   - Add `# → fixes audit finding #9` comment near the new config
     flags in config.py and at the `_pool_admission_floors` site.
6. **Do NOT change**:
   - The existing pool admission gates (they should remain at
     1.15, just sourcing from the new ADMISSION flag).
   - The `min_trade_support` or `min_trade_pool_floor` settings
     (separate concept).
   - Any other file outside `config.py`, `phases/phase2_support.py`,
     `phases/phase2_rule_pool.py`, and the test files.

## Acceptance Criteria
1. `PHASE2_PROFIT_FACTOR_FLOOR_EVOLUTION = 1.05` (default).
2. `PHASE2_PROFIT_FACTOR_FLOOR_ADMISSION = 1.15` (default).
3. `PHASE2_PROFIT_FACTOR_FLOOR = 1.15` (deprecated alias, still
   works for backward compat).
4. `_pool_admission_floors` returns 1.15 (ADMISSION), not 1.05
   (EVOLUTION).
5. The evolution soft penalty uses 1.05 (EVOLUTION), not 1.15.
6. Pool admission still rejects val_pf < 1.15 (regression guard).
7. A rule with train_pf=1.10, val_pf=1.10 now gets a SMALLER soft
   penalty (using 1.05) than under the old behavior (using 1.15).
8. All existing tests pass: `test_phase2_rule_pool.py`,
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
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_evox_runner.py -q
```

## Notes
- Do NOT modify `evaluator_v5.ipynb`.
- Do NOT run the full project or full test suite locally (OOM risk
  per AGENTS.md).
- The deprecated `PHASE2_PROFIT_FACTOR_FLOOR` alias is critical for
  backward compat — any external user (or test) that reads the old
  flag should still get 1.15.
- The 1.05 evolution floor is a heuristic; the OOS gain will be
  validated by the user's next Colab run (post all tasks).
- This is a small surgical fix (similar in scope to task-4, task-5,
  task-6). Keep the diff minimal.
- The new flag values will be applied immediately to all users —
  no migration needed since the default values are reasonable.
