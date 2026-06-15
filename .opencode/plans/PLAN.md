# Plan — Add friend's robust ideas to my pipeline (preserve regime detection)

This plan is the result of a side-by-side audit of
`gpu_fuzzy_trader/` (mine, bad test results) vs
`friend_project/gpu_fuzzy_trader/` (good test results). The friend added
an "RB Equity Governor" layer that is, in essence, a re-implementation
of Phase 3/4/5 with stricter selection criteria and a faithful clone
of the evaluator's backtest engine. We are not copying the governor
verbatim — instead we are adopting the *concepts* that explain the
performance gap, while keeping my regime detection (which the friend
lacks).

## Acceptance criteria
- Final `outputs/long.json` and `outputs/short.json` keep their current
  shape (`direction`, `rules_set`, `tp`, `sl`, `capital_pct`,
  `conditions` with `[feature] IS <value>` and `symbol is X`).
- Test-set evaluation (in `evaluator_v5.ipynb`) shows positive total
  return for *both* long and short (target: long ≥ 0% and short ≥ 0%,
  ideally both > 5%).
- New code lives next to existing code with config flags for safe
  enable/disable.
- Regime detection (mine) is preserved and its bar-level label is
  still used by Phase 2.

## Tasks

### Task 1 — Add `validation/monthly_windows.py` and `validation/rolling_cv.py`
*Priority: high. This is the largest single gap.*

Copy the friend's `monthly_windows.py` and `rolling_cv.py` into
`gpu_fuzzy_trader/validation/`. Expose them via the package `__init__.py`
that I'll create. Adjust imports to use my `config as _cfg` and my
`backtest.cpu_engine.CPUBacktestEngine`. Add the corresponding
`MONTHLY_*` and `PURGED_CV_*` config keys to my `config.py` (with the
same defaults the friend uses).

**Acceptance**: `from gpu_fuzzy_trader.validation.monthly_windows import
evaluate_rule_set_monthly` works; `evaluate_rule_set_monthly(df, rs,
"long")` returns `(MonthlyWindowSummary, list[dict])`.

### Task 2 — Wire `monthly_penalty` into Phase 3 and Phase 4 scoring
*Priority: high.*

In `phases/phase3_rule_set.py`, build a `combined_df = pd.concat([train_df, val_df])`
once and pass it to `evaluate_rule_set_monthly(...)` per candidate
rule-set. Add `monthly_window_penalty = monthly_penalty(summary) * PHASE3_MONTHLY_PENALTY_WEIGHT`
to the per-rule-set score. The friend uses `PHASE3_MONTHLY_PENALTY_WEIGHT=1.0`
and `PHASE4_MONTHLY_SCORE_WEIGHT=0.70`. Use the same defaults.

In `phases/phase4_wf_optimizer.py`, add the same monthly penalty to
each Optuna trial's score (so the chosen TP/SL/capital must also be
robust across monthly windows).

**Acceptance**: When `MONTHLY_VALIDATION_ENABLED=True`, a strategy
that beats the validation set but is negative on >40% of monthly
windows gets a worse score than a slightly worse validation-return
strategy that is positive on >60% of months.

### Task 3 — Add `_is_positive_good`-style gate
*Priority: high.*

Add a `gate_positive_good(train_metrics, val_metrics)` helper in
`phases/phase3_rule_set.py` (and reuse in Phase 4 / RB-style phase)
that returns `True` only when:
- `train_ret > 0`, `val_ret > 0`
- `train_pf ≥ PHASE3_MIN_TRAIN_PF` (default 1.0), `val_pf ≥ PHASE3_MIN_VAL_PF` (default 1.0)
- `train_trades ≥ PHASE3_MIN_TRAIN_TRADES` (default 25), `val_trades ≥ PHASE3_MIN_VAL_TRADES` (default 15)

Wire it into the per-symbol greedy and the global pool fallback so a
failing rule never reaches the final strategy file. Add the four
config keys with the friend's defaults.

**Acceptance**: After Phase 3, every rule in the final `rules_set`
passes the gate on the train/val split. (Check by re-running my engine.)

### Task 4 — Add evaluator-failure-mode awareness
*Priority: high.*

Extend my `CPUBacktestEngine.simulate_rule_set` return dict to include:
`raw_signal_count`, `executed_trades`, `skipped_min_notional_count`,
`max_simultaneous_positions`, `max_total_open_exposure`, `loss_count`,
`time_closed_count` — exactly the fields the evaluator reports. These
are already computed internally; just expose them.

Then add a `_evaluator_health_penalty(metrics)` function (port from
friend's `rb_governor.py`) and add it to Phase 3/4 scoring. Penalties:
- `(skipped / raw) > 0.20` → penalty
- `(executed / raw) < 0.60` → penalty
- `max_simultaneous_positions > 10` → penalty

**Acceptance**: My engine returns the evaluator's metric names; the
penalty is 0 for any rule-set that does not trigger skip/exposure
issues.

### Task 5 — Expand Phase 2 pool admission
*Priority: medium.*

Relax the per-rule pool-admission thresholds. Current: 20% gap-reject
between train and val. Replace with the friend's logic: a rule is
admitted if it passes `_is_positive_good` *or* a weaker fallback
(`PHASE2_CV_MIN_WORST_RETURN=-8%`, `PHASE2_CV_MAX_WORST_DD=18%`).
Add `PHASE2_KEEP_TOP_RULES=140` (or similar) to cap pool size after
filtering. Increase `PHASE2_ARCHIVE_MAX_SIZE=500` (already similar in
mine) and `PHASE2_GENERATIONS=50` is fine.

**Acceptance**: The post-Phase-2 pool has at least 15 long and 15 short
rules on the current dataset (up from 5 / 8).

### Task 6 — Add multi-symbol combinations
*Priority: medium.*

In Phase 3 (`_per_symbol_greedy` / `_merge_per_symbol_rules`), instead
of only merging the *same* rule selected by multiple single-symbol
greedy runs, also try the top-K 2-symbol and 3-symbol combinations of
each pool rule's best per-symbol scores. The friend does this in
`_symbol_specialized_variants`. Add a `SYMBOL_SPECIALIZATION_USE_COMBINATIONS=True`
flag and `SYMBOL_SPECIALIZATION_MAX_SYMBOLS_PER_RULE=3`.

**Acceptance**: Some rules in the final `rules_set` contain more than
one `symbol is X` condition (e.g. `symbol is 1, symbol is 5, symbol
is 9`).

### Task 7 — Add `_optimize_risk` grid search
*Priority: medium.*

Replace (or augment) the Optuna walk-forward in Phase 4 with a
deterministic grid search from the friend: TP in
`(1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0)`, SL in
`(1.0, 1.2, 1.5, 2.0, 2.5, 3.0)`, capital_pct in
`(5, 7.5, 10, 12.5, 15, 20, 25, 35, 50)`. For each combination, check
`sum(capital_pct) ≤ 95%` and `_is_positive_good`. Pick the combo with
the highest `_score_metrics`. Run 2 passes (per-rule round-robin).

Keep Optuna as a *secondary* path behind a config flag
(`PHASE4_USE_OPTUNA=True` default). Both paths must produce the same
JSON shape.

**Acceptance**: A dry-run on the current long.json/short.json
(2 rules, 1.5×1×2 capital) re-optimizes to a higher combined return
on the validation split.

### Task 8 — Add `regime_feature_keyword` stratum initialization
*Priority: low (complement to existing regime detection).*

Copy the friend's `_regime_feature_indices` and add a "regime"
stratum to my Phase 2 init (currently I have only "elite" and
"explorer"). When `PHASE2_REGIME_STRATUM_FRAC=0.25`, 25% of new
chromosomes are forced to have their first active gene be a
"regime/volatility/trend" feature (vol, atr, bb_width, range,
compression, adx, dmi, semivol, etc.). This is a feature-space proxy
that complements my bar-level regime label.

Add `PHASE2_REGIME_STRATUM_FRAC=0.25` and `PHASE2_REGIME_FEATURE_KEYWORDS`
to `config.py`.

**Acceptance**: `phase2_long_history.json` shows that ≥20% of the
initial population has a first-gene feature whose name matches one of
the regime keywords.

### Task 9 — Add `evaluator_clean` writer
*Priority: low (defensive).*

Add a small helper that, after Phase 4, writes
`outputs/evaluator_clean/{long,short}_evaluator_clean.json` with
*only* `{direction, rules_set}`. The friend does this. It protects
against a hypothetical stricter evaluator that rejects unknown
top-level keys.

**Acceptance**: `outputs/evaluator_clean/long_evaluator_clean.json`
exists, parses with `json.load`, and contains only `direction` and
`rules_set`.

### Task 10 — Verify on a real run
*Priority: high. Final gate.*

Run the full pipeline on the current dataset (or a smaller subset for
RAM-safety) and check that:
- `outputs/long.json` and `outputs/short.json` keep their shape.
- `outputs/reports/test_long_report.json` and `test_short_report.json`
  show positive `total_return_pct` for *both* directions.
- `outputs/reports/strategy_evaluation_long.csv` and `_short.csv`
  show `total_return_pct` matches the JSON report.
- The new monthly-summary report (if I add one) shows ≥ 60%
  `profitable_ratio` and positive `equity_slope` for both directions.

## Risk analysis

- **Backward compatibility** of JSON output: I'm only adding
  *additional* fields to the engine return dict; the existing
  `total_return_pct`, `max_drawdown_pct`, `win_rate`, `profit_factor`,
  `executed_trades`, `per_symbol_metrics` stay. The strategy JSON keeps
  its current shape with optional new keys. The evaluator reads only
  `direction` and `rules_set`, so adding extras is safe.
- **Runtime**: monthly validation runs 24 short backtests per
  rule-set. With 2 directions × 1 phase3 call × 5 candidates × 24
  monthly windows × 2 rules = ~480 backtests. Each backtest on the
  validation split (~210k rows) is ~2s on the CPU. Total ~16 minutes.
  Acceptable.
- **RAM**: the friend does not use GPU; we keep GPU on Phase 2 and
  switch to CPU for Phase 3/4/5 (same as the friend). I will not run
  the GPU-engine in Phase 3/4 to keep parity with the friend and to
  stay under the user's 12.7 GiB RAM budget.
- **Removing old code**: per `AGENTS.md`, "remove additional (wasted)
  parts from old implementation". After Task 7, the old Optuna path
  is behind a flag; if it is unused I will delete it. After Task 3,
  the existing `PHASE3_MAX_TRAIN_VAL_GAP_PCT=40.0` is replaced by the
  stricter `_is_positive_good` gate — I will remove the old fallback
  paths that are no longer reachable.

## Out of scope
- Touching `evaluator_v5.ipynb`.
- Modifying the GPU fitness path (the `EvoX` NSGA-III). We keep
  existing logic.
- Building a multi-run auto-search loop. The friend has
  `auto_search.py` that runs multiple directions and aggregates a
  `best_global/`. We can add this later, but it is not blocking the
  hidden-test improvement.
- Any new feature engineering (features, detector, encoder are out of
  scope).
