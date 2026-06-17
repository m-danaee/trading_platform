# Plan — Task 12..15: Diagnostic-first path for test-set generalization

## Goal

Improve out-of-sample (OOS) generalization of the trading pipeline — the
current run produces `LONG = -1.0%` and `SHORT = -2.6%` on test despite
`+19.5%` / `+18.5%` on train and `+20.6%` / `+29.0%` on validation —
through a **diagnostic-first, low-risk** sequence of interventions,
with explicit skip criteria between them. The full per-symbol Phase 2
refactor proposed in the brainstorming session is **deferred** to
Task 15, and only attempted if Tasks 12–14 fail to close the gap.

## Background (from brainstorming + data review)

The current pipeline fails on test (Sept 2024 → Feb 2025, 5-month OOS
window) for these reasons, in order of evidence strength:

1. **Phase 3 per-symbol selection has 6/10 symbols with no rules on
   test.** `PHASE3_PER_SYMBOL_MIN_TRADES = 15` and
   `PHASE3_PER_SYMBOL_MIN_RETURN = 1.5%` may be too strict for the
   7k-row per-symbol validation windows. The pipeline currently
   silently drops these symbols, and the surviving rules are biased
   toward whichever symbol happened to win on validation.
2. **Phase 2 pool admission is fit on min(train, val) only**, with
   no concept of "stable across time". A rule that does well on the
   last 3 months of train but fails on the first 3 months is
   indistinguishable from a robust rule.
3. **Phase 4 risk params are global per rule**, not per-(rule,
   symbol). A rule whose TP/SL fits symbol 9 on val may be wrong
   for symbol 8.
4. (Hypothesis, will be measured) **The regime shift between train
   (Jan 2024 – Jun 2024) and test (Sep 2024 – Feb 2025) is large
   enough** that no amount of model sophistication trained on the
   available window will close the gap; the user has to accept this
   or add external data. Task 12's first iteration will tell us.

The previous `friend_project/` comparison work already added:
- `monthly_windows.py` (Task 1, merged)
- `monthly_penalty` wired into Phase 3/4 (Task 2, merged)
- `_is_positive_good`-style gate (Task 3, merged)
- Evaluator health penalty (Task 4, merged)
- Expanded Phase 2 pool admission (Task 5, merged)
- Multi-symbol combinations in Phase 3 (Task 6, merged)
- Risk-optimization grid search (Task 7, merged)
- Regime-keyword stratum init (Task 8, merged)
- Evaluator-clean writer (Task 9, merged)
- `_try_lean_fallback` (Task 10, merged)
- Purged CV removed (Task 11, merged)

The `monthly_penalty` is already integrated but is a *soft*
penalty. There is no **hard pool-admission gate** based on
monthly-window profitability ratio, and there is no per-symbol
risk-param search in Phase 4.

## Current scope (per-phase intervention)

| Phase | Current behaviour | Where to intervene |
|---|---|---|
| Phase 1 (feature selection) | Sign-consistency + MI + redundancy + stationarity | No change |
| Phase 2 (rule pool) | NSGA-III on min(train, val) with cross-symbol robustness penalty | **Task 13**: add post-evolution monthly-window pool-admission gate |
| Phase 3 (rule set) | Per-symbol greedy with `MIN_TRADES=15, MIN_RETURN=1.5%` | **Task 12**: lower per-symbol thresholds; preserve positive-good gate |
| Phase 4 (risk) | Global grid per rule on TP/SL/capital | **Task 14**: optional per-symbol risk tuning |
| Phase 5 (OOS) | CPU backtest on test | No code change; only used to measure improvement |

## Skip / proceed decision rules

Each task has an **explicit skip criterion**. If the skip criterion
is met, that task's branch is closed and the next task is attempted.
If a task is closed, its results (improvement vs. regression) MUST
be recorded in `outputs/reports/gen_diag_iterN.csv` so the user
can see what was tried.

- **After Task 12**: if the test PnL on either direction improves
  by ≥3pp **AND** the number of symbols with rules on test
  increases to ≥7/10, proceed to Task 13. Otherwise, treat Task 12
  as inconclusive; record results and proceed to Task 13 anyway
  (the per-symbol threshold change is harmless).
- **After Task 13**: if the test PnL on either direction improves
  by ≥3pp vs. Task 12, proceed to Task 14. Otherwise close the
  branch; record results.
- **After Task 14**: if test PnL improves by ≥3pp vs. Task 13,
  stop. Otherwise consider Task 15 (the full per-symbol refactor).
- **Task 15 is the last resort** — it is the user's original idea
  and carries the highest run-time cost (≈10× Phase 2 cost). It is
  not started unless Tasks 12-14 collectively fail to bring test
  PnL above −3% per direction.

## Acceptance criteria (overall)

1. The final `outputs/long.json` and `outputs/short.json` each have
   `len(rules_set) >= 2`.
2. `outputs/reports/test_{long,short}_per_symbol_performance.csv`
   shows **at least 6 distinct symbols** (out of 10) with non-empty
   rows in the test split, AND at least one symbol with positive net
   PnL per direction.
3. Test `total_return_pct` improves by **at least 3 percentage
   points** per direction vs. the current baseline
   (`LONG=-1.0%, SHORT=-2.6%`). This is the primary success metric.
4. Train+validation `total_return_pct` does **not regress by more
   than 5pp** per direction (we still want the in-sample behaviour
   to be healthy).
5. `evaluator_v5.ipynb` is not modified.
6. The pipeline is reproducible: a fresh `git clone` + the documented
   commands produce the same outputs.

## Baseline numbers to beat

| Direction | Train | Val | Test | Top test symbol | # symbols on test |
|---|---|---|---|---|---|
| Long | +19.5% | +20.6% | **−1.0%** | 8 (+26.2) | 4/10 |
| Short | +18.5% | +29.0% | **−2.6%** | 2 (+15.9) | 4/10 |

These are the target numbers to beat; the success criterion is
"both directions test PnL improves by ≥3pp".

---

## Task 12 — Lower per-symbol Phase 3 thresholds + add diagnostic reporting

**Priority: high**
**Risk: very low** (one numeric change + one diagnostic flag)
**Runtime cost: ~75s (Phase 3 re-run only)**

### Why this is the right first step

6 out of 10 symbols have no rules on test. The most likely
explanation is that the per-symbol thresholds
(`PHASE3_PER_SYMBOL_MIN_TRADES = 15`,
`PHASE3_PER_SYMBOL_MIN_RETURN = 1.5%`) are too strict for the 7k-row
per-symbol validation windows. With ~10x fewer rows than the
cross-symbol pool, a per-symbol min-trades of 15 is approximately
equivalent to the pool-level min-trades of 150, which is
unreasonably high.

### Files

- `gpu_fuzzy_trader/config.py` (lower the two constants, add a
  `PHASE3_DIAGNOSTIC_REPORT_ENABLED` flag)
- `gpu_fuzzy_trader/phases/phase3_rule_set.py` (no logic change
  unless thresholds are referenced; only logging)
- `outputs/reports/gen_diag_iter12.csv` (new — see below)

### Implementation steps (for the implementer)

1. **Create the feature branch from `main`**:
   ```bash
   git checkout main && git pull
   git checkout -b feature/task-12-lower-phase3-thresholds
   ```

2. **In `gpu_fuzzy_trader/config.py`**:
   - Lower `PHASE3_PER_SYMBOL_MIN_TRADES` from `15` to `8`.
   - Lower `PHASE3_PER_SYMBOL_MIN_RETURN` from `1.5` to `0.5`.
   - Add `PHASE3_DIAGNOSTIC_REPORT_ENABLED = True`.
   - Add a brief comment explaining why these numbers (one sentence
     each) — see `effective_phase3_per_symbol_min_trades` for the
     debug-scope pattern to mimic.

3. **In `gpu_fuzzy_trader/phases/phase3_rule_set.py`**:
   - In the `Rule_Set_Selector.run()` loop, after each
     `_per_symbol_greedy` call, accumulate a
     `dict[sym, dict[trades, return_pct, n_selected, gap_pct]]` in
     `self._diag_per_symbol`.
   - After Phase 3 completes, if
     `_cfg.PHASE3_DIAGNOSTIC_REPORT_ENABLED`, write
     `outputs/reports/gen_diag_iter12.csv` with columns:
     `direction, symbol, val_trades, val_return_pct, train_val_gap_pct,
     n_rules_selected, top_rule_condition_signature`.
   - The CSV is a **diagnostic artifact** for the user; it is not
     consumed by downstream phases.

4. **In `tests/unit/test_phase3_rule_set.py`** (or a new
   `test_phase3_threshold_diagnostic.py`):
   - Add a unit test that:
     - Constructs a small pool (5 rules) and a small train+val
       split.
     - Runs `Rule_Set_Selector.run()` and asserts that the
       diagnostic CSV is written and contains one row per symbol
       that has at least 1 rule in val.
     - Asserts that with `MIN_TRADES=8, MIN_RETURN=0.5`, a rule
       with `8` trades and `0.5%` val return is selected for a
       symbol, but a rule with `7` trades or `0.4%` return is
       not.

5. **Re-run the pipeline** (Phase 2-3-4-5 only; the existing
   Phase 2 pool is reused because of the archive cache):
   ```bash
   .venv/bin/python -m gpu_fuzzy_trader.run_pipeline
   ```
   Capture `outputs/reports/test_{long,short}_*.csv` and
   `outputs/reports/gen_diag_iter12.csv`.

6. **Commit the changes**:
   ```bash
   git add -A
   git commit -m "task-12: lower per-symbol Phase 3 thresholds + diagnostic CSV

   - PHASE3_PER_SYMBOL_MIN_TRADES 15 -> 8
   - PHASE3_PER_SYMBOL_MIN_RETURN 1.5 -> 0.5
   - Add PHASE3_DIAGNOSTIC_REPORT_ENABLED flag
   - Write outputs/reports/gen_diag_iter12.csv with per-symbol
     trades, return, gap, n_selected, top_rule_signature
   - Add unit test for the threshold diagnostic
   - Re-run pipeline; record results in gen_diag_iter12.csv
   "
   ```

7. **Write handoff JSON** to
   `.opencode/handoffs/task-12-implementer.json` with:
   - `branch`: `feature/task-12-lower-phase3-thresholds`
   - `commit`: the implementer commit SHA
   - `verification`: dict with the test outputs and the test PnL
     numbers (both directions).

### Acceptance criteria for Task 12

- `outputs/reports/gen_diag_iter12.csv` exists and has one row per
  (direction, symbol) where at least 1 rule was selected.
- `outputs/reports/test_{long,short}_per_symbol_performance.csv`
  has rows for at least 6 distinct symbols per direction (currently
  4/10).
- All unit tests pass:
  ```bash
  .venv/bin/python -m pytest \
    tests/unit/test_phase3_rule_set.py \
    tests/unit/test_phase3_threshold_diagnostic.py \
    -v
  ```
- The new test must show: rule with 8 trades / 0.5% return IS
  selected; rule with 7 trades / 0.4% return is NOT selected.

### Skip / proceed decision for Task 12

- **If test PnL improves by ≥3pp on either direction** AND **the
  number of symbols with rules on test reaches ≥7/10**: declare
  Task 12 the fix, skip Task 13, and proceed to Task 14.
- **Otherwise**: Task 12 is recorded as a partial improvement
  (lower threshold but no test improvement). Record results, mark
  task as DONE, and proceed to Task 13.

---

## Task 13 — Add monthly-window shadow test to Phase 2 pool admission

**Priority: high** (if Task 12 is inconclusive)
**Risk: medium** (touches Phase 2 pool admission; must not break
existing pool size)
**Runtime cost: +5–10 min on top of Phase 2** (post-evolution
re-evaluation, not per-generation)

### Why this is the right second step

Even with relaxed Phase 3 thresholds, the surviving pool may still
contain rules that are good on the *last month* of train (which is
what the user is using as "validation") but fail on the *first
5 months* of train. This is the canonical overfit pattern that
purged CV was supposed to catch — but we removed purged CV in
Task 11. We need a cheaper replacement: a **monthly-window shadow
test** evaluated on the **final pool**, not during evolution.

`monthly_windows.py` (Task 1) already builds 30-day rolling
windows; `monthly_penalty()` (Task 2) already exists. We just
need to make it a **hard pool-admission gate** (not a soft
penalty).

### Files

- `gpu_fuzzy_trader/config.py` (add 2 constants)
- `gpu_fuzzy_trader/phases/phase2_rule_pool.py` (add a post-build
  filter step)
- `gpu_fuzzy_trader/validation/monthly_windows.py` (no change
  expected; read-only usage)
- `tests/unit/test_phase2_monthly_admission.py` (new)

### Implementation steps

1. **Create the feature branch from `main` after Task 12 is
   merged**:
   ```bash
   git checkout main && git pull
   git checkout -b feature/task-13-phase2-monthly-admission
   ```

2. **In `gpu_fuzzy_trader/config.py`**:
   - Add `PHASE2_MONTHLY_ADMISSION_ENABLED = True`.
   - Add `PHASE2_MONTHLY_ADMISSION_MIN_PROFITABLE_RATIO = 0.5`
     (rule must be profitable on ≥50% of monthly windows in the
     train split).
   - Add `PHASE2_MONTHLY_ADMISSION_MIN_MONTHS = 4` (need at least
     4 months of data; reject if train is shorter).
   - Add a brief comment block explaining the gate.

3. **In `gpu_fuzzy_trader/phases/phase2_rule_pool.py`**:
   - After the existing pool-builder produces the merged pool
     (currently at `pool_size=78` for long, `122` for short on
     this dataset), add a post-filter:
     ```python
     if _cfg.PHASE2_MONTHLY_ADMISSION_ENABLED:
         monthly_windows = build_monthly_windows(train_df)
         # min-months gate
         if len(monthly_windows) < _cfg.PHASE2_MONTHLY_ADMISSION_MIN_MONTHS:
             logger.warning(
                 "Phase 2 [%s]: only %d monthly windows; skipping gate",
                 self.direction, len(monthly_windows))
         else:
             keep = []
             for rule in merged_pool:
                 ret_pcts = [
                     evaluate_rule_on_window(rule, w)["total_return_pct"]
                     for w in monthly_windows
                 ]
                 profitable = sum(1 for r in ret_pcts if r > 0)
                 if profitable / len(ret_pcts) >= _cfg.PHASE2_MONTHLY_ADMISSION_MIN_PROFITABLE_RATIO:
                     keep.append(rule)
             merged_pool = keep
     ```
   - `evaluate_rule_on_window(rule, w)` is a small helper that
     uses the existing `CPUBacktestEngine.simulate_rule_set()` to
     backtest a single rule on a single window. Reuse the same
     logic that `_simulate_team` uses.
   - If the gate filters out everything, log a clear warning and
     keep the original pool (degrade gracefully).
   - Log the gate stats: `pre_filter_count, post_filter_count,
     median_profitable_ratio, p10_profitable_ratio`.

4. **In `tests/unit/test_phase2_monthly_admission.py`** (new):
   - Construct a pool of 3 rules: one profitable on all months,
     one profitable on half the months, one profitable on no
     months.
   - Construct a 6-month train split with synthetic positive and
     negative PnL per month.
   - Assert the gate keeps only the first.
   - Assert that disabling the flag keeps all 3.
   - Assert that with `min_months=10` and 6 months of data, the
     gate is skipped and the original pool is kept.

5. **Re-run the pipeline** (full run, ~3 hours).

6. **Commit the changes** with a clear message; update
   `outputs/reports/gen_diag_iter13.csv` with pre/post-filter
   counts and final test PnL.

7. **Write handoff JSON** to
   `.opencode/handoffs/task-13-implementer.json`.

### Acceptance criteria for Task 13

- The pool after Task 13 is **strictly smaller** than the pool
  after Task 12 (we are filtering, not adding).
- Pool size is still ≥30 rules per direction (so Phase 3 has
  enough candidates).
- All unit tests pass:
  ```bash
  .venv/bin/python -m pytest \
    tests/unit/test_phase2_monthly_admission.py \
    -v
  ```
- `outputs/reports/gen_diag_iter13.csv` exists with the gate
  statistics.

### Skip / proceed decision for Task 13

- **If test PnL improves by ≥3pp vs. Task 12** AND
  **pool size is still ≥30**: declare Task 13 a success and
  proceed to Task 14.
- **If pool is too small** (<30) but test PnL improved: relax
  the gate to `MIN_PROFITABLE_RATIO = 0.4` and re-run.
- **If test PnL did not improve**: close Task 13 as
  inconclusive; record results; proceed to Task 14 anyway
  (Task 14 is independent of Phase 2).

---

## Task 14 — Per-(rule, symbol) Phase 4 risk tuning

**Priority: medium**
**Risk: low** (additive; when flag is False, behaviour is
unchanged)
**Runtime cost: +20–30s on top of Phase 4** (Phase 4 currently
takes 2.2s for 6,480 trials; 10 symbols × per-symbol grid is
64,800 trials ≈ 22s)

### Why this is the right third step

The user's idea of "per-symbol training" is partially
implementable as "per-symbol risk tuning": keep the global rule
set (Phase 2/3) but allow each (rule, symbol) pair to have its
own TP/SL/capital_pct. This isolates the per-symbol risk
exposure from the per-symbol *signal* (which is what the
expensive per-symbol training would change).

### Files

- `gpu_fuzzy_trader/config.py` (add 3 constants)
- `gpu_fuzzy_trader/phases/phase4_wf_optimizer.py` (add a
  per-symbol grid branch)
- `gpu_fuzzy_trader/output/writer.py` (extend the output
  schema to support per-(rule, symbol) risk — see "Output
  schema" below)
- `tests/unit/test_phase4_per_symbol_risk.py` (new)

### Output schema (per-(rule, symbol) risk)

Currently the output rule looks like:
```json
{
  "tp": 2.0,
  "sl": 1.0,
  "capital_pct": 16.66,
  "conditions": ["[feature] IS Value", "symbol is 4"]
}
```

With per-symbol risk, the new shape is:
```json
{
  "conditions": ["[feature] IS Value"],
  "risk_per_symbol": {
    "4": {"tp": 2.0, "sl": 1.0, "capital_pct": 16.66},
    "9": {"tp": 1.5, "sl": 1.2, "capital_pct": 10.0}
  }
}
```

**Critical:** `evaluator_v5.ipynb` reads `tp`, `sl`,
`capital_pct` directly from the rule (not from a
`risk_per_symbol` sub-dict). So when `risk_per_symbol` is
present, Phase 5 must:
- For each rule, find the symbol the rule fires on (via
  `symbol is X` conditions).
- Look up that symbol in `risk_per_symbol`.
- Inject `{tp, sl, capital_pct}` at the top level of the
  rule.

Phase 5 already filters negative-PnL rules. The new path is
additive: when `risk_per_symbol` is present, the filter is
applied **per (rule, symbol)** rather than per rule.

### Implementation steps

1. **Create the feature branch from `main` after Task 13 is
   merged**:
   ```bash
   git checkout main && git pull
   git checkout -b feature/task-14-per-symbol-risk
   ```

2. **In `gpu_fuzzy_trader/config.py`**:
   - Add `PHASE4_PER_SYMBOL_RISK_ENABLED = False` (default off
     — user opts in only when they want it).
   - Add `PHASE4_PER_SYMBOL_RISK_TOP_K_SYMBOLS = 3` (only tune
     the top 3 most-active symbols per rule, to keep grid
     tractable).
   - Add `PHASE4_PER_SYMBOL_RISK_MAX_DELTA_PCT = 20.0` (the
     per-symbol risk must differ from the global risk by at
     most 20%, to prevent overfit).
   - Add a brief comment block explaining the three constants.

3. **In `gpu_fuzzy_trader/phases/phase4_wf_optimizer.py`**:
   - When `PHASE4_PER_SYMBOL_RISK_ENABLED`:
     - After the global grid is done, for each rule, identify
       the top-K symbols (by `trade_count` on val).
     - For each (rule, top_symbol), run a local grid of
       `±20%` around the global best TP/SL/capital and keep
       the per-symbol best.
     - Store the result in a new
       `risk_per_symbol[symbol]` dict.
   - The per-symbol grid is much smaller (5×5×5 = 125 cells
     vs. 540 global cells).
   - When the flag is False, the new code path is not
     executed at all (zero behavior change).

4. **In `gpu_fuzzy_trader/phases/phase5_oos.py`**:
   - When loading `outputs/long.json` / `outputs/short.json`,
     if a rule has `risk_per_symbol`:
       - Find the rule's symbols from `symbol is X` conditions.
       - For each symbol, look up the risk in `risk_per_symbol`.
       - Inject the risk at the top level of the rule.
     - If a symbol is not in `risk_per_symbol`, fall back to
       the global TP/SL/capital.

5. **In `tests/unit/test_phase4_per_symbol_risk.py`** (new):
   - Construct a small rule set with 2 rules, each on 2
     symbols.
   - Run `Phase4WalkForwardOptimizer` with
     `PHASE4_PER_SYMBOL_RISK_ENABLED=True`.
   - Assert each rule has a `risk_per_symbol` dict with the
     expected symbols.
   - Assert the per-symbol risk differs from the global risk
     by at most `MAX_DELTA_PCT`.

6. **Re-run the pipeline** with the flag enabled (since it
   defaults to False, the user must opt in for the run).

7. **Commit the changes**; update
   `outputs/reports/gen_diag_iter14.csv`.

8. **Write handoff JSON** to
   `.opencode/handoffs/task-14-implementer.json`.

### Acceptance criteria for Task 14

- All unit tests pass:
  ```bash
  .venv/bin/python -m pytest \
    tests/unit/test_phase4_per_symbol_risk.py \
    -v
  ```
- `outputs/long.json` and `outputs/short.json` have
  `risk_per_symbol` sub-dicts when the flag is enabled, and
  the top-level `tp/sl/capital_pct` are the **fallback**
  values for symbols not in `risk_per_symbol`.
- `outputs/reports/test_{long,short}_*.csv` shows that at
  least one symbol's net PnL improved by ≥3pp vs. Task 13.
- `evaluator_v5.ipynb` was not modified.

### Skip / proceed decision for Task 14

- **If test PnL improves by ≥3pp on either direction** vs.
  Task 13: declare success. The diagnostic-first path is
  complete; Tasks 15+ are not needed.
- **If test PnL is still negative on both directions** after
  Tasks 12+13+14: consider Task 15 (full per-symbol refactor)
  only if the user explicitly asks for it. The cost is ≈10×
  Phase 2 runtime, with no guarantee of improvement.

---

## Task 15 (last resort) — Full per-symbol Phase 2 refactor

**Priority: low** — only attempted if the user explicitly
requests it and Tasks 12-14 are collectively insufficient.
**Risk: high** (touches the most expensive phase)
**Runtime cost: ≈10× Phase 2 cost** (one NSGA-III run per
symbol, total ~28 hours for both directions on this dataset)

This task is **NOT started by default**. It is documented here
for completeness so the user has a clear path if the
diagnostic-first sequence fails.

### When to attempt

- Tasks 12, 13, 14 all merged.
- Test PnL still ≤ −3% on both directions.
- User explicitly requests the full per-symbol refactor.

### High-level design (sketch only — not a plan yet)

1. **Phase 2 per-symbol**:
   - For each symbol, run NSGA-III on a symbol-scoped train
     split.
   - Lower `MIN_TRADE_SUPPORT` from 45 to 12 to account for
     the smaller per-symbol data.
   - Lower `MIN_CONDITIONS` from 3 to 2 to keep the search
     space explorable.
   - Add a **shared-pool post-filter**: from the union of
     per-symbol pool rules, keep rules that are profitable on
     **≥5 symbols** with at least `8` trades per symbol
     (this is a hard minimum to avoid spurious shared rules).
   - The shared pool and the per-symbol pool are both kept
     and passed to Phase 3.

2. **Phase 3 (per-symbol, with shared pool injection)**:
   - For each symbol, run greedy selection on the
     concatenation of (local pool ∪ shared pool), giving
     shared rules a slight priority.
   - Add a `top_k` cap on how many shared rules can be
     selected per symbol (default 1) to prevent a single
     shared rule from dominating.

3. **Phase 4 (per-symbol risk, always on)**:
   - Task 14 is folded in: `PHASE4_PER_SYMBOL_RISK_ENABLED`
     becomes `True` by default.

4. **Skip criteria for this task** (in case it does not help):
   - If the test PnL still does not improve vs. Tasks 12-14,
     revert and accept that the regime shift is too large for
     the available data.

### Why this is the LAST resort

The brainstorming analysis showed:
- Per-symbol training does not fix the regime-shift problem.
- It 10x's the runtime for unclear gain.
- The "≥5 symbols" shared-pool filter is statistically
  dangerous at N=10 symbols (you cannot reliably detect
  cross-sectional robustness from 10 symbols).

The diagnostic-first path (Tasks 12-14) is much cheaper and
addresses the actual failure modes identified in the data.

---

## Task ledger

| # | Title | Status | Branch |
|---|---|---|---|
| 12 | Lower per-symbol Phase 3 thresholds + diagnostic CSV | **TODO** | `feature/task-12-lower-phase3-thresholds` |
| 13 | Monthly-window shadow test in Phase 2 pool admission | **TODO** | `feature/task-13-phase2-monthly-admission` |
| 14 | Per-(rule, symbol) Phase 4 risk tuning | **TODO** | `feature/task-14-per-symbol-risk` |
| 15 | (Last resort) Full per-symbol Phase 2 refactor | **DEFERRED** | n/a |

## Implementation order

Tasks 12 → 13 → 14 in that order. Each task produces a
re-runnable pipeline, a handoff JSON, a unit test, and a
diagnostic CSV. The next task is only started after the
previous one is merged and the user is shown the
`gen_diag_iterN.csv` summary.

## Constraints (carry-over from CONTEXT.md)

- **Do NOT modify `evaluator_v5.ipynb`.**
- **Do NOT** touch `friend_project/` (committed reference).
- **Use `.venv/bin/python`** for all commands.
- **Do NOT run all tests** — pick the relevant unit tests per
  task (full pipeline run is the integration test).
- **Keep the `outputs/long.json` and `outputs/short.json`
  shape** compatible with `evaluator_v5.ipynb`; the new
  `risk_per_symbol` sub-dict is additive (the legacy
  `tp/sl/capital_pct` fields stay as fallback values).
- **Use feature branches** `feature/task-N-*`; one implementer
  per task; spec-reviewer, then code-reviewer; merge to `main`.
- **Update `.opencode/CONTEXT.md` task ledger** after each
  task merges.
