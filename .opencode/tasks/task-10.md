# Task 10 — Fix empty rules + Date-based equity plot

## Goal

Restore non-empty `outputs/long.json` and `outputs/short.json` (≥ 2 rules
each, evaluator-valid) and switch equity-curve plots from `Trade #` to
`Date`. This unblocks the hidden-test pipeline (currently produces 0
rules → 0 OOS results).

## Background

The current run (Jun 16) produced:

- Phase 1: 20 features per direction ✓
- Phase 2 long: pool=4, short: pool=9 ✓
- Phase 3: 0 rules for any symbol (per-symbol threshold too strict)
- Phase 3 team fallback: failed `_is_positive_good` gate
- Phase 4: skipped
- Phase 5: "rules_set must have ≥ 2 rules" — skipped
- Result: empty `{direction}.json`, empty `evaluator_clean/`, no OOS
  report

User also asked for `Date` on the x-axis of `*_equity.png` to make
monthly-window reasoning easier.

## Sub-tasks

### 10.1 — Lower per-symbol thresholds

In `gpu_fuzzy_trader/config.py`:
- `PHASE3_PER_SYMBOL_MIN_TRADES`: `50` → `15`
- `PHASE3_PER_SYMBOL_MIN_RETURN`: `3.0` → `1.5`

Both currently unreachable given a 4–9 rule pool. The friend project
uses similar floors (`SYMBOL_SPECIALIZATION_MIN_TRADES=20`,
`AUTO_SEARCH_SCORE_MIN_TRADES=60` for tighter cases).

### 10.2 — Add `_try_lean_fallback` to Phase 3

In `gpu_fuzzy_trader/phases/phase3_rule_set.py`:

When per-symbol selection fails AND the strict team fallback fails,
add a third path: `_try_lean_fallback` that picks the **top-2 pool
rules by `min(train, val) return`** without requiring PF or evaluator
health on the team (those gates were calibrated for the friend
project's 140-rule pool, not ours). The Phase 4 risk optimization
and evaluator parity checks in Phase 5 will filter out weak rules.

Document the path clearly so future readers understand why it's
deliberate.

The new path should:
- Pick the top 2 pool rules by `_pool_rule_val_score` (already
  exists; `min(train_ret, val_ret)`).
- Skip the positive-good gate, the PF floor, and the evaluator
  health check.
- Apply only the `val_floor` (from `effective_phase3_val_return_floor_pct`)
  as a sanity check.
- Log a WARNING explaining the relaxation.
- Return the 2-rule list (or `None` if `min(train, val) <= val_floor`).

The fallback should be tried in this order:
1. Per-symbol greedy (current)
2. `_try_global_pool_fallback` (current, strict)
3. `_try_lean_fallback` (new, relaxed) — only if (1) and (2) both fail

### 10.3 — Date-based x-axis in `plot_equity_curve`

In `gpu_fuzzy_trader/reporting/reporter.py`, update
`Reporter.plot_equity_curve` (line 494):
- Use `trade_log["Entry_Time"]` for x-axis (already in the log).
- Sort the trade log by `Entry_Time` first (trades can interleave
  across symbols).
- Set `ax.set_xlabel("Date")` and use matplotlib date formatter
  (`mdates.DateFormatter("%Y-%m-%d")`).
- Fall back to `range(len(equity))` + `"Trade #"` if `Entry_Time`
  is missing or all-NaN (so existing test fixtures don't break).
- Keep the initial-capital reference line and the title format.

### 10.4 — Document `outputs/evaluator_clean/`

Add `outputs/evaluator_clean/README.md` (≈ 30 lines) explaining:
- What the folder is (defensive stripped copy of the strategy files)
- When it's written (Phase 3 / 4 / 5 via `_maybe_write_evaluator_clean`)
- Why it exists (a stricter evaluator might reject extra keys like
  `risk_optimized`, `selection_accepted`, `selection_rejection_reason`)
- How to interpret the `rules_set: []` case (Phase 3 rejected all
  rules — see the `selection_rejection_reason` in the metadata JSON)

### 10.5 — Add tests

Add 2 new test files (per AGENTS.md: small, fast tests only — DO NOT
run the full test suite):
- `tests/unit/test_reporter_equity_date_axis.py` — verify date
  formatting is applied when `Entry_Time` is present, and falls
  back to "Trade #" when missing.
- `tests/unit/test_phase3_lean_fallback.py` — verify the new lean
  fallback returns 2 rules when called and `None` when the team
  return is below the floor.

Use `pytest` patterns already in the test suite.

## Acceptance criteria

1. `outputs/long.json` and `outputs/short.json` each have
   `len(rules_set) >= 2` after a re-run on the user's data.
2. Each rule in the final `rules_set` still passes either:
   - `_is_positive_good` (per-symbol or team fallback path), OR
   - The new lean fallback (deliberate relaxation, documented).
3. `outputs/evaluator_clean/{long,short}_evaluator_clean.json` are
   both present after a run, with the strict `{direction, rules_set}`
   shape.
4. `outputs/reports/{train,validation,test}_{long,short}_equity.png`
   show `Date` on the x-axis (or `Trade #` for the no-Entry_Time
   fallback path, e.g. test fixtures).
5. `outputs/evaluator_clean/README.md` exists and documents the
   folder.
6. New unit tests pass: `pytest tests/unit/test_reporter_equity_date_axis.py
   tests/unit/test_phase3_lean_fallback.py -v`.
7. **No new files outside the agreed scope** — no junk, no stale
   code paths, no orphaned imports (per AGENTS.md: "remove additional
   wasted parts from old implementation"). If you find any unused
   import or dead code, remove it.
8. All commits are on the assigned feature branch, never on `main`.

## Target files

| File | Change |
|---|---|
| `gpu_fuzzy_trader/config.py` | Lower 2 per-symbol thresholds |
| `gpu_fuzzy_trader/phases/phase3_rule_set.py` | Add `_try_lean_fallback` + wire into `Rule_Set_Selector.run` |
| `gpu_fuzzy_trader/reporting/reporter.py` | Date-based x-axis |
| `outputs/evaluator_clean/README.md` | **new** — documentation |
| `tests/unit/test_reporter_equity_date_axis.py` | **new** — test |
| `tests/unit/test_phase3_lean_fallback.py` | **new** — test |
| `.opencode/CONTEXT.md` | Update task ledger (last row of task-10 → DONE) |
| `.opencode/handoffs/task-10-implementer.json` | **new** — handoff JSON |

## Verification steps

1. `python -c "import ast; ast.parse(open('gpu_fuzzy_trader/config.py').read())"`
2. `python -c "from gpu_fuzzy_trader.phases.phase3_rule_set import _try_lean_fallback; print('OK')"`
3. `python -c "from gpu_fuzzy_trader.reporting.reporter import Reporter; print('OK')"`
4. `python -m pytest tests/unit/test_reporter_equity_date_axis.py tests/unit/test_phase3_lean_fallback.py -v`
5. `git diff main...HEAD --stat` — confirm the diff is scoped to task 10

## Constraints (per AGENTS.md)

- Use `.venv` for any Python commands.
- Do NOT run the full test suite (user has RAM limits).
- Do NOT modify `evaluator_v5.ipynb`.
- Do NOT modify the working `outputs/` (they'll be regenerated on
  the user's next run).
- Keep the venv workflow.

## Risks

- **Lean fallback bypasses the positive-good gate**. This is
  deliberate (the gate was too strict for our 4-rule pool) but
  means a 2-rule team with weak PF could leak through.
  **Mitigation**: Phase 4 risk optimization will not optimize risk
  for a 0-PF team (it'll keep the Phase 2 defaults), and Phase 5's
  evaluator will score it as-is.
- **Date formatting** may not be the user's exact format.
  **Mitigation**: the implementation uses `YYYY-MM-DD` and falls
  back to `Trade #` if the trade log has no `Entry_Time`.
- **Lower per-symbol thresholds** may admit noisy rules.
  **Mitigation**: the Phase 3 monthly penalty + Phase 4 risk grid
  search will still down-weight bad months.

## Out of scope (deferred)

- Changing `SPLIT_MODE` to `purged_rolling_cv` (would re-run Phase 2
  for ~2 hours, user already approved `holdout_70_30` for Task 5).
- Improving the Phase 2 pool size (Task 5 already raised the cap
  to 140; the current 4-rule pool is from a different filter path).
- Re-tuning `PHASE3_MAX_TRAIN_VAL_GAP_PCT` (separate concern; will
  be addressed if test still shows overfit after this fix).
