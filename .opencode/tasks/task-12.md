# Task 12 — Lower per-symbol Phase 3 thresholds + diagnostic CSV

## Goal

Reduce `PHASE3_PER_SYMBOL_MIN_TRADES` and `PHASE3_PER_SYMBOL_MIN_RETURN`
so that more symbols (out of 10) get at least one rule assigned on
test, and add a per-symbol diagnostic CSV so the user can see exactly
which symbols are still being dropped and why.

The full plan is in `.opencode/plans/PLAN.md` (Task 12 section).
The CONTEXT.md is in `.opencode/CONTEXT.md`.

## Background

The current run (with `PHASE3_PER_SYMBOL_MIN_TRADES = 15` and
`PHASE3_PER_SYMBOL_MIN_RETURN = 1.5`) produces only 4/10 symbols
with rules on the test split, and the surviving rules have
negative PnL on 3 of those 4 symbols. Our analysis suggests
these thresholds are too strict for the ~7k-row per-symbol
validation windows (15 trades on a 7k-row window is ~0.2%
of bars, which is the same effective density as ~150 trades
on a 70k-row cross-symbol window — the latter is currently
the Phase 2 pool floor, not a per-symbol floor).

The user's primary concern is **regime shift between train
(Jan 2024 – Jun 2024) and test (Sep 2024 – Feb 2025)**. Task 12
is a *diagnostic*: if the number of symbols with rules on test
does not rise after this change, the regime-shift hypothesis is
the dominant cause and we need Tasks 13-14 (regime-aware gates)
to address it.

## Branch

- base_branch: `main`
- feature branch: `feature/task-12-lower-phase3-thresholds`
- branch_policy: isolated (one branch per task)
- execution_mode: checkpoint (stop after implementer for user review)

## Workflow

1. Implementer creates the branch (already done by orchestrator),
   lowers the two constants, adds the diagnostic CSV, and writes
   the unit test.
2. Spec-reviewer verifies acceptance criteria.
3. Code-reviewer checks for unintended side effects.
4. User is asked for confirmation before merging to `main`.
5. After merge, the full pipeline is re-run and the user is shown
   the `gen_diag_iter12.csv` summary.

## Acceptance criteria

1. **`PHASE3_PER_SYMBOL_MIN_TRADES` lowered from 15 to 8** in
   `gpu_fuzzy_trader/config.py`. The `effective_phase3_per_symbol_min_trades()`
   helper must still respect the new value.
2. **`PHASE3_PER_SYMBOL_MIN_RETURN` lowered from 1.5 to 0.5** in
   `gpu_fuzzy_trader/config.py`. The `effective_phase3_per_symbol_min_return()`
   helper must still respect the new value.
3. **`PHASE3_DIAGNOSTIC_REPORT_ENABLED` flag added** (default
   `True`) in `gpu_fuzzy_trader/config.py`.
4. **`outputs/reports/gen_diag_iter12.csv` is written** by
   `Rule_Set_Selector.run()` when the flag is enabled. Columns
   are: `direction, symbol, val_trades, val_return_pct,
   train_val_gap_pct, n_rules_selected, top_rule_condition_signature`.
   The CSV must have one row per `(direction, symbol)` pair that
   had at least 1 rule selected.
5. **Unit test added** in `tests/unit/test_phase3_threshold_diagnostic.py`
   that:
   - Constructs a small pool (5 rules) and a small train+val
     split (2 symbols, 1k rows each).
   - Runs `Rule_Set_Selector.run()` and asserts the diagnostic
     CSV is written.
   - Asserts that a rule with `8` trades and `0.5%` val return
     is selected for a symbol.
   - Asserts that a rule with `7` trades is NOT selected.
   - Asserts that a rule with `0.4%` val return is NOT selected.
6. **All existing unit tests still pass**:
   ```bash
   .venv/bin/python -m pytest \
     tests/unit/test_phase3_rule_set.py \
     tests/unit/test_phase3_lean_fallback.py \
     tests/unit/test_phase3_threshold_diagnostic.py \
     -v
   ```
7. **No `evaluator_v5.ipynb` modification.**
8. **No `friend_project/` modification.**
9. **The Phase 2 pool is reused from the existing archive**
   (`outputs/phase2_*_pool.json`); only Phase 3-4-5 should
   re-run. The full pipeline re-run time should be ~10 min
   (Phase 3-5 only, no Phase 2).

## Files to modify

- `gpu_fuzzy_trader/config.py` — change 2 constants, add 1 flag
- `gpu_fuzzy_trader/phases/phase3_rule_set.py` — add diagnostic
  CSV writing logic in `Rule_Set_Selector.run()`
- `tests/unit/test_phase3_threshold_diagnostic.py` — new file
- `.opencode/handoffs/task-12-implementer.json` — handoff JSON

## Verification commands

The implementer must run these and record results in the handoff:

```bash
# 1. AST / import sanity
.venv/bin/python -c "import ast; ast.parse(open('gpu_fuzzy_trader/config.py').read()); print('config ok')"
.venv/bin/python -c "from gpu_fuzzy_trader.phases.phase3_rule_set import Rule_Set_Selector; print('phase3 ok')"

# 2. New unit test passes
.venv/bin/python -m pytest tests/unit/test_phase3_threshold_diagnostic.py -v

# 3. Existing Phase 3 tests still pass
.venv/bin/python -m pytest tests/unit/test_phase3_rule_set.py tests/unit/test_phase3_lean_fallback.py -v

# 4. Full pipeline re-run (Phase 2-3-4-5; reuses Phase 2 pool from cache)
.venv/bin/python -m gpu_fuzzy_trader.run_pipeline

# 5. Verify diagnostic CSV is written
head outputs/reports/gen_diag_iter12.csv
```

## Skip / proceed decision (for the orchestrator / user)

After the implementer merges, the user will be shown:

1. `outputs/reports/gen_diag_iter12.csv` — the per-symbol diagnostic
2. `outputs/reports/test_{long,short}_per_symbol_performance.csv` —
   the test per-symbol performance
3. The test `total_return_pct` for both directions

**Decision rule**:

- **If test PnL improves by ≥3pp on either direction** AND
  **the number of symbols with rules on test reaches ≥7/10**:
  declare Task 12 a success; consider skipping Task 13 and
  going straight to Task 14.
- **If test PnL is still negative on both directions** but
  **the number of symbols with rules on test rises to ≥6/10**:
  Task 12 is a partial improvement; proceed to Task 13
  (monthly-window shadow test).
- **If the number of symbols with rules on test does not rise**:
  the regime-shift hypothesis is confirmed; proceed to
  Task 13 immediately.

## Constraints

- Per AGENTS.md: **always use `.venv/bin/python`** for commands.
- Per AGENTS.md: **remove additional (wasted) parts from old
  implementation** to keep the project clean.
- Per AGENTS.md: **do not run all tests** (RAM limits) — only
  the unit tests listed above.
- Per AGENTS.md: **evaluator_v5.ipynb is sacred** — do not
  modify it.
- The `outputs/long.json` and `outputs/short.json` shape must
  stay compatible with `evaluator_v5.ipynb` (i.e. the legacy
  `tp/sl/capital_pct` fields are still at the top level of each
  rule).
- One commit on the feature branch (no need to split into
  multiple commits for a 1-line threshold change + diagnostic
  CSV).
