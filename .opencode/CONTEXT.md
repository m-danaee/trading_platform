# Context — Friend project comparison & improvement plan

## Goal
Side-by-side compare `gpu_fuzzy_trader/` (mine) vs `friend_project/gpu_fuzzy_trader/`
to identify *why* the friend gets good hidden-test results while mine is
negative. The friend is the GitHub project `trader_bigdata_governor` — they
adopt an additional "RB Equity Governor" layer on top of the standard
Phase 1-5 pipeline.

## What evaluator_v5 expects
The final evaluator only reads `outputs/long.json` and `outputs/short.json`.
Each must contain a `direction` (`"long"` or `"short"`) and a `rules_set`
list of `{conditions, tp, sl, capital_pct}`. Conditions must use the exact
`[feature] IS <value>` format (plus optional `symbol is X` filters).
No other top-level key is required. The evaluator tracks
`raw_signal_count`, `executed_trades`, `skipped_min_notional_count`,
`max_simultaneous_positions`, `max_total_open_exposure`, `final_equity`,
`account_ruined`, etc. — these are *outputs* of the evaluator, not
*inputs* the user provides.

## Why my last run is bad (LONG -4.23%, SHORT -3.14% test)
- Rule selection: I only keep 2 rules per direction, both heavily filtered by
  per-symbol greedy + 20% gap-reject threshold. The pool is also tiny
  (5 long + 8 short rules) because Stage A → Stage B with mutation=0.18
  collapses to "more of the same chromosomes".
- Both directions have TP ≤ 4 and SL ≤ 1.5, so any single position is
  capped to ~4% net move. With 22.5% / 17.5% drawdowns, the equity
  curve is degrading in 2024-25 test — the rules are basically symmetric
  long-bias/short-bias that don't survive regime shift.
- My scoring in Phase 2/3/4 has no concept of "monthly robustness" and no
  per-fold monthly profit factor, so the rule set ends up overfit to
  one validation window.
- The rules have `symbol is X` filters, but with a 20% gap-reject in
  Phase 3 they were too restrictive — they only fire for symbols 1, 5
  in practice.

## High-level differences (mine vs friend)

| Concern | Mine | Friend |
|---|---|---|
| Regime detection | Per-bar `regime` label (0/1/2) from rolling OLS of `label_open_next`; used in Phase 2 to evaluate each rule per regime, but not as a discrete feature in the chromosome | **No** discrete regime label. Instead, `PHASE2_REGIME_FEATURE_KEYWORDS` reserves 25% of the initial population for chromosomes anchored on volatility/trend features (vol, atr, bb_width, adx, semivol, etc.) — a "feature-space" proxy for regime-aware rules. |
| Selection criterion | min(train, val) return + 20% gap-reject | `_is_positive_good`: rule must be **positive on both** train and val with min trades; combined with min-PF ≥ 1.0 on both legs; per-month profitable-ratio penalty |
| Walk-forward robustness | 2 windows on validation; PURGED_CV_N_SPLITS=4 for Phase 2 only | 4 purged rolling CV folds for Phase 2; **PHASE3_HIDDEN_VALIDATION** uses 6 hidden 90-day windows with worst-return floor −5%; **MONTHLY_VALIDATION** evaluates the rule set on 30-day rolling windows for `profitable_ratio ≥ 60%`, `worst_pf ≥ 0.85`, `equity_slope > 0` |
| Rule composition | Greedy on validation return | Greedy on `combined_return_score = train_ret + val_ret - evaluator_health_penalty`; requires *every* new rule to keep `_positive_returns()` true and improve the combined return |
| Symbol specialization | Phase 3 adds `symbol is X` per-symbol | **Both** Phase 3 (`SYMBOL_SPECIALIZATION_ENABLED=True`) **and** the RB governor does it again: every pool rule is expanded into the top-5 single-symbol variants + 2-/3-symbol combinations, evaluated on the data, and the best variant is kept |
| Risk optimization | Optuna multi-objective walk-forward | The governor does deterministic grid-search over TP/SL/capital with a 95% cap and a tight 1.5–10 TP / 1.0–3.0 SL / 5–50 capital grid, requiring *every* combination to keep `_is_positive_good()` |
| Evaluator parity | `CPUBacktestEngine` mirrors evaluator_v3 (slightly older) | `rb_evaluator_v5.py` contains a faithful clone of evaluator_v5 (`EvaluatorV5BacktestEngine`) used for *all* governor scoring. The governor explicitly penalizes for the evaluator's failure modes (`skipped_min_notional_count`, `max_simultaneous_positions`, `executed_trades/raw_signal_count` ratio) |
| Pipeline structure | Phase 1 → 2 → 3 → 4 → 5 always | If `RB_ENGINE_ENABLED=True`, Phase 3+4 are *replaced* by the RB governor. Phase 5 (OOS test) is also replaced with `update_global_bank_and_compose` (multi-run aggregator) |
| Final outputs | `outputs/long.json`, `outputs/short.json` (with extra metadata like `risk_optimized`, `deployment_accepted`, `validation_gate`) | `outputs/long.json`, `outputs/short.json` (same shape), **plus** `outputs/evaluator_clean/{long,short}_evaluator_clean.json` and `best_global/` aggregator outputs |

## Mine: things that are good and should be kept
- **My regime detection** is better than the friend's keyword-based proxy.
  Rolling OLS with R² + slope + median pre-filter + 14-day min duration
  produces a per-bar discrete regime (0=sideways, 1=bear, 2=bull). It is
  used in `phase2_support.py` to weight chromosomes by their per-regime
  return. The friend has nothing equivalent. **KEEP it** and extend its
  use downstream (see recommendations).
- The per-bar regime label is saved in the dataset (`regime` column)
  and used in the `evox_runner` regime-aware mask. This is a strong
  feature that the friend lacks. **KEEP it**.
- `regime_cluster` is a `rolling_regression` clusterer with 9-day median
  pre-filter, 14-day min duration — these smoothing choices are sound.
- My CPU/GPU engines already handle symbol conditions correctly (the
  same `split_feature_and_symbol_conditions` logic as the friend).
- The Phase 2 purged CV and the spec-consistency filter are good.

## Mine: things that are likely hurting hidden-test performance
- **No monthly validation**. The single most likely cause of negative
  test results is that my rules are great on one validation window and
  fail on the rest of the year. The friend's `monthly_windows.py` is
  cheap to run and explicitly forces `profitable_ratio ≥ 60%` and
  `equity_slope > 0` across 30-day windows.
- **No min(train, val) PF guardrail** in Phase 3/4. The friend requires
  PF ≥ 1.0 on both legs of the rule (`RB_MIN_TRAIN_PF`, `RB_MIN_VALID_PF`).
  My selection only checks the return, so a rule with great return but
  PF < 1 can still be selected.
- **No evaluator-failure-mode awareness** in scoring. The friend
  penalizes `(skipped/raw) > 0.20` and `(executed/raw) < 0.60`, and
  caps `max_simultaneous_positions ≤ 10`. My Phase 3/4 doesn't track
  these metrics. The evaluator WILL skip signals that are below
  `MIN_POSITION_NOTIONAL = 1.0`, so a high skip rate means a rule is
  effectively generating 0 PnL.
- **Tiny Phase 2 pool** (5 long, 8 short). The friend uses
  `RB_KEEP_TOP_RULES = 140` *after* symbol specialization, and runs
  the full rule pool through the evaluator-aware filter. My
  `_build_pool_from_archive` filters out everything that doesn't pass
  the train-val gap ≤ 20% — way too aggressive for a noisy problem.
- **No `symbol is X` combinations**, only single-symbol. The friend
  explores 1-, 2-, and 3-symbol combinations per rule (`RB_SYMBOL_USE_COMBINATIONS=True`).
  My phase 3 only picks a single symbol per rule (or merges identical
  conditions across symbols).

## Recommendations (in priority order)
1. **Add `validation/monthly_windows.py`** to my project (copy from
   friend). Add `MONTHLY_VALIDATION_ENABLED=True` and a `monthly_penalty`
   to my Phase 3/4 scoring. This is the single biggest gap.
2. **Adopt `_is_positive_good`-style gate in Phase 3/4**: rule must be
   positive on *both* train and val with PF ≥ 1.0 on both.
3. **Adopt evaluator-aware health penalty**: track `raw_signal_count`,
   `executed_trades`, `skipped_min_notional_count`, `max_simultaneous_positions`
   in my engine and add a penalty in scoring. Use `rb_evaluator_v5.py` as
   a reference to ensure parity.
4. **Expand rule pool by relaxing pool-admission thresholds**. The
   current 20% gap-reject is too aggressive; relax to a 50% floor
   similar to `PHASE2_CV_MIN_WORST_RETURN=-8%` so the pool has more
   diverse candidates.
5. **Add `_optimize_risk` style grid search** to replace (or supplement)
   Phase 4 Optuna for the final TP/SL/capital. The friend's grid
   `(1.0,1.2,1.5,2.0,2.5,3.0,3.5,4.0,5.0,6.0,8.0,10.0) × (1.0,1.2,1.5,2.0,2.5,3.0)
   × (5,7.5,10,12.5,15,20,25,35,50)` is much more aggressive on the
   upside, which gives the hidden-test room to find a good TP.
6. **Add 2/3-symbol combinations** in Phase 3 symbol specialization
   (currently only single-symbol).
7. **Keep my regime detection** and *additionally* implement
   `PHASE2_REGIME_FEATURE_KEYWORDS` "regime" stratum initialization
   in Phase 2 (the friend does this; the keywords are a feature-space
   proxy, but it's complementary to my bar-level regime).
8. **Add an `evaluator_clean` writer** that strips metadata and keeps
   the strict `{direction, rules_set}` shape, in case the evaluator
   gets a stricter version.

## JSON output contract — must keep
`outputs/long.json` and `outputs/short.json` must keep this exact shape:
```json
{
  "direction": "long",
  "risk_optimized": true,
  "rules_set": [
    {
      "conditions": ["[feature] IS Value", "symbol is X"],
      "tp": <float>,
      "sl": <float>,
      "capital_pct": <float>
    }
  ]
}
```
Extra keys (`deployment_accepted`, `validation_gate`, etc.) are
tolerated by the evaluator (it only reads `direction` and `rules_set`).
The current run already has that shape — we must preserve it.

## Constraints
- **Do NOT modify `evaluator_v5.ipynb`**.
- Keep the venv workflow; the user has RAM limits.
- The outputs and reports in `outputs/` are the **only** files used for
  hidden-test scoring — the run logs / pipeline.log are not graded.

## Active orchestration state

- base_branch: `main`
- current_task: Task 6 DONE; checkpoint for user review/merge before Task 7
- active_branch: `feature/task-6-multi-symbol-combinations` (1 commit on top of main)
- dispatch_mode: one implementer at a time, then spec-reviewer, then code-reviewer
- user_chose: option (a) — implement all 10 tasks; branch policy = reviewable in isolation; checkpoint per task
- handoff_dir: `.opencode/handoffs/`

## Task ledger
| # | Title | Status | Branch | Commit | Merged? |
|---|---|---|---|---|---|
| 1 | Add `validation/monthly_windows.py` + `rolling_cv.py` | DONE / APPROVED | `feature/task-1-monthly-validation` | `f930104` | **YES** (`9957cc1` on `main`) |
| 2 | Wire `monthly_penalty` into Phase 3/4 scoring | DONE / APPROVED | `feature/task-2-monthly-penalty-scoring` | `ca028ba` | **YES** (`9b49fba` on `main`) |
| 3 | Add `_is_positive_good`-style gate | DONE / APPROVED | `feature/task-3-positive-good-gate` | `f96addd` | **YES** (`eb502a1` on `main`) |
| 4 | Evaluator-failure-mode awareness | DONE / APPROVED | `feature/task-4-evaluator-health-penalty` | `9271bc7` | **YES** (`eb37ee5` on `main`) |
| 5 | Expand Phase 2 pool admission | DONE / APPROVED | `feature/task-5-expand-phase2-pool` | `d01c8d7` | **YES** (`f5a44ea` on `main`) |
| 6 | Multi-symbol combinations in Phase 3 | DONE / APPROVED | `feature/task-6-multi-symbol-combinations` | `585dc39` | **pending user merge** |
