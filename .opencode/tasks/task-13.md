# Task 13 — Add monthly-window shadow test to Phase 2 pool admission

## Goal

Add a **hard pool-admission gate** to Phase 2: each candidate rule
must be profitable on at least 50% of monthly windows in the train
split. This addresses the regime-shift problem that Task 12's
diagnostic CSV revealed: the per-symbol rules that passed Phase 3
on val (Jan-Aug 2024) bleed on test (Sep 2024 - Feb 2025) because
they are not stable across time.

The full plan is in `.opencode/plans/PLAN.md` (Task 13 section).
The CONTEXT.md is in `.opencode/CONTEXT.md`.

## Background

The post-Task-12 pipeline re-run showed:
- **LONG: -8.09% test return** (was -1.02%, -7.1pp worse)
- **SHORT: -7.43% test return** (was -2.56%, -4.8pp worse)
- **Long: 7/10 symbols with rules on test** (up from 4/10)
- **Short: 5/10 symbols with rules on test** (up from 4/10)

The diagnostic CSV (`outputs/reports/gen_diag_iter12.csv`) revealed
that the new per-symbol rules fire on val but bleed on test,
confirming the regime-shift hypothesis. The right fix is to
reject Phase 2 pool rules that don't survive across multiple
monthly windows of train data — these are the rules most likely
to be regime-specific.

`monthly_windows.py` (Task 1) and `monthly_penalty` (Task 2)
already exist; we just need to make the **monthly-window
profitable ratio** a **hard pool-admission gate** (not a soft
penalty in NSGA-III fitness).

## Branch

- base_branch: `main` @ `336e29f`
- feature branch: `feature/task-13-phase2-monthly-admission`
- branch_policy: isolated (one branch per task)
- execution_mode: checkpoint (stop after implementer for user review)

## Workflow

1. Implementer creates the branch, adds the gate to Phase 2, and
   writes the unit test.
2. Spec-reviewer verifies acceptance criteria.
3. Code-reviewer checks for unintended side effects.
4. User is asked for confirmation before merging to `main`.
5. After merge, the full pipeline is re-run and the user is shown
   the `gen_diag_iter13.csv` summary (also `gen_diag_iter12.csv`
   for comparison).

## Acceptance criteria

1. **Three new config flags** in `gpu_fuzzy_trader/config.py`:
   - `PHASE2_MONTHLY_ADMISSION_ENABLED = True` (default on)
   - `PHASE2_MONTHLY_ADMISSION_MIN_PROFITABLE_RATIO = 0.5`
     (rule must be profitable on ≥50% of monthly windows)
   - `PHASE2_MONTHLY_ADMISSION_MIN_MONTHS = 4`
     (skip the gate if train is shorter than 4 months)
   - All three have comment blocks explaining the rationale
     and the date of introduction (2026-06-17, Task 13).

2. **Gate is applied AFTER Phase 2 evolution** in
   `gpu_fuzzy_trader/phases/phase2_rule_pool.py`:
   - After the existing pool builder produces `merged_pool`
     (currently at line ~1690 — search for `merged_pool`),
     add a post-filter block:
     ```python
     if _cfg.PHASE2_MONTHLY_ADMISSION_ENABLED:
         monthly_windows = build_monthly_windows(train_df)
         if len(monthly_windows) < _cfg.PHASE2_MONTHLY_ADMISSION_MIN_MONTHS:
             logger.warning(
                 "Phase 2 [%s]: only %d monthly windows; skipping gate",
                 self.direction, len(monthly_windows))
         else:
             # ... evaluate each rule on each window ...
             # ... filter by profitable_ratio ...
     ```
   - The exact insertion point must be in the
     `Rule_Pool_Generator.run()` method, after the existing
     `merged_pool = ...` assignment and before the pool is
     saved to `phase2_*_pool.json`.

3. **Helper function** for evaluating a single rule on a single
   window. Use the existing `CPUBacktestEngine.simulate_rule_set`
   (the same one used in Phase 3). Pseudocode:
   ```python
   def _evaluate_rule_on_window(rule, window_df, direction):
       engine = CPUBacktestEngine(window_df, {}, direction, fee_pct=_cfg.FEE_PCT)
       metrics = engine.simulate_rule_set([rule])
       return float(metrics.get("total_return_pct", 0.0))
   ```
   - This function should live in `phase2_rule_pool.py` (or in
     `validation/monthly_windows.py` if it fits the module's
     scope better — implementer's choice).
   - The function should be **fast** (no heavy pre-processing).

4. **Graceful degradation**: if the gate filters out everything
   (e.g., `merged_pool` becomes empty), log a clear warning and
   keep the **original** `merged_pool` (degrade gracefully — do
   not produce an empty pool, which would crash Phase 3).

5. **Log gate statistics**: pre-filter count, post-filter count,
   median profitable_ratio, p10 profitable_ratio. The log line
   should be at INFO level and clearly labeled with the direction.

6. **Unit test** in `tests/unit/test_phase2_monthly_admission.py`:
   - Construct a pool of 3 rules: one profitable on all months,
     one profitable on half the months, one profitable on no
     months.
   - Construct a 6-month train split with synthetic positive and
     negative PnL per month (use monkeypatched
     `_evaluate_rule_on_window` to return deterministic values).
   - Assert the gate keeps only the first rule.
   - Assert that disabling the flag keeps all 3 rules.
   - Assert that with `min_months=10` and 6 months of data, the
     gate is skipped and the original pool is kept.
   - Assert that when the gate would empty the pool, the original
     pool is kept (graceful degradation).

7. **All existing unit tests still pass**:
   ```bash
   .venv/bin/python -m pytest \
     tests/unit/test_phase2_monthly_admission.py \
     tests/unit/test_phase2_rule_pool.py \
     tests/unit/test_phase2_support.py \
     -v
   ```
   (Do NOT run all tests per AGENTS.md — RAM limits.)

8. **No `evaluator_v5.ipynb` modification** (per AGENTS.md).
9. **No `friend_project/` modification**.

10. **One commit on the feature branch** (clear message; the
    gate logic + helper + test all in one commit).

11. **Pipeline re-run**:
    - The implementer should NOT re-run the full pipeline (that
      takes ~3 hours and is the orchestrator's job post-merge).
    - The implementer SHOULD verify the unit tests pass and the
      new code is importable.
    - The orchestrator will re-run the pipeline after merge.

## Files to modify

- `gpu_fuzzy_trader/config.py` — add 3 constants with comment
  blocks
- `gpu_fuzzy_trader/phases/phase2_rule_pool.py` — add the
  helper function, the gate block, and the import for
  `build_monthly_windows`
- `tests/unit/test_phase2_monthly_admission.py` — new file
- `.opencode/handoffs/task-13-implementer.json` — handoff JSON

## Verification commands (run by the implementer)

```bash
# 1. AST / import sanity
.venv/bin/python -c "import ast; ast.parse(open('gpu_fuzzy_trader/config.py').read()); print('config ok')"
.venv/bin/python -c "import ast; ast.parse(open('gpu_fuzzy_trader/phases/phase2_rule_pool.py').read()); print('phase2 ok')"

# 2. Imports work
.venv/bin/python -c "from gpu_fuzzy_trader.phases.phase2_rule_pool import Rule_Pool_Generator; print('pool import ok')"
.venv/bin/python -c "from gpu_fuzzy_trader.validation.monthly_windows import build_monthly_windows; print('monthly_windows ok')"

# 3. New unit test passes
.venv/bin/python -m pytest tests/unit/test_phase2_monthly_admission.py -v

# 4. Existing Phase 2 tests still pass
.venv/bin/python -m pytest tests/unit/test_phase2_rule_pool.py tests/unit/test_phase2_support.py -v
```

**Do NOT run the full pipeline** in this step (orchestrator
will do it after merge). The unit tests are sufficient.

## Skip / proceed decision (for the orchestrator / user)

After the implementer merges, the user will be shown:

1. The pre/post-filter counts logged in the pipeline output
2. The pool size for both directions (was 78 long, 122 short
   in the post-Task-12 run; should be smaller after the gate)
3. The test PnL for both directions

**Decision rule**:

- **If test PnL improves by ≥3pp on either direction** vs.
  Task 12's result (-8.09% long, -7.43% short): declare Task 13
  a success; proceed to Task 14.
- **If pool size drops below 30 rules per direction** AND
  test PnL is still negative: relax the gate to
  `MIN_PROFITABLE_RATIO = 0.4` and re-run.
- **If test PnL does not improve at all**: record Task 13 as
  inconclusive; proceed to Task 14 anyway (Task 14 is
  independent of Phase 2).

## Constraints

- Per AGENTS.md: **always use `.venv/bin/python`** for commands.
- Per AGENTS.md: **do not run all tests** — only the four
  Phase 2 test files listed.
- Per AGENTS.md: **no `evaluator_v5.ipynb` modification**.
- Per AGENTS.md: **no `friend_project/` modification**.
- Per AGENTS.md: **remove additional (wasted) parts from old
  implementation** — if the gate makes any existing code
  redundant, remove it.
- The `outputs/long.json` and `outputs/short.json` shape must
  stay compatible with `evaluator_v5.ipynb`.
- The gate is **additive** when the flag is False (zero
  behavior change). Implement this as an `if` block at the
  top of the post-pool-build step.
- **Minimal diff** — do not refactor unrelated code.
