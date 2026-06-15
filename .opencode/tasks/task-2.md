# Task 2 — Wire `monthly_penalty` into Phase 3 and Phase 4 scoring

## Why
Task 1 added `validation/monthly_windows.py` and its `monthly_penalty`
function. This task actually *uses* the penalty in the rule-set
selection and risk-optimization scoring. The friend does this
(`PHASE3_MONTHLY_PENALTY_WEIGHT=1.0`, `PHASE4_MONTHLY_SCORE_WEIGHT=0.70`)
and that is the single most impactful robustness gate. Without it,
Task 1's monthly module is dead code from the orchestrator's POV.

## Required reading
- `.opencode/tasks/task-1.md` (context for the monthly module)
- `.opencode/plans/PLAN.md` (overall plan)
- `.opencode/CONTEXT.md` (JSON output contract; evaluator_v5 contract)
- The new `validation/monthly_windows.py` already merged in `main`
- The friend's reference: `friend_project/gpu_fuzzy_trader/phases/phase3_rule_set.py` (lines 371-465) and `friend_project/gpu_fuzzy_trader/phases/phase4_rl_optimizer.py` (lines 928-932, 819-927)

## Files I will touch
- `gpu_fuzzy_trader/config.py` (add the two `PHASE3_MONTHLY_PENALTY_WEIGHT` / `PHASE4_MONTHLY_SCORE_WEIGHT` constants)
- `gpu_fuzzy_trader/phases/phase3_rule_set.py` (add `monthly_window_penalty` to per-rule-set scoring)
- `gpu_fuzzy_trader/phases/phase4_wf_optimizer.py` (add `monthly_summary` to the per-trial score)
- `tests/unit/test_monthly_wired_into_scoring.py` (new) — small unit test that confirms a candidate with bad monthly performance scores lower than one with good monthly performance, even when validation return is identical.

## Behavior changes

### Phase 3 — `phases/phase3_rule_set.py`

In the per-symbol greedy loop (`_per_symbol_greedy` and the final
`Rule_Set_Selector.run`), build a `combined_df = pd.concat([train_df, val_df], ignore_index=True)`
once at the start of `run()` (if not already built) and pass it to
`evaluate_rule_set_monthly(combined_df, rule_set, direction,
feature_names=...)`. Compute
`monthly_window_penalty = monthly_penalty(summary) * PHASE3_MONTHLY_PENALTY_WEIGHT`
and add it to the per-rule-set score. The friend uses
`PHASE3_MONTHLY_PENALTY_WEIGHT=1.0`.

Specifically:
- The penalty is added in `_evaluate_rule_set` (or equivalent inner
  function) so it affects every candidate evaluation.
- A `monthly_summary` is computed once per `run()` call when
  `MONTHLY_VALIDATION_ENABLED=True`; for inner greedy loops that
  don't change the data, cache it.
- If `evaluate_rule_set_monthly` raises or returns `windows == 0`, do
  NOT crash; treat the penalty as the small constant
  `PHASE3_MONTHLY_FALLBACK_PENALTY` (default 5.0) and continue.

### Phase 4 — `phases/phase4_wf_optimizer.py`

In the Optuna trial objective (or whatever scoring function the WF
optimizer uses), accept an optional `monthly_summary` argument and
add a `PHASE4_MONTHLY_SCORE_WEIGHT` (default 0.70) multiplier of the
`monthly_penalty` to the trial's composite score. The friend uses
`PHASE4_MONTHLY_SCORE_WEIGHT=0.70`.

The friend computes a fresh `monthly_summary` per trial. I will do the
same but with a config flag to cap the number of trials that get a
fresh monthly evaluation (default: every trial). The flag is
`PHASE4_MONTHLY_EVAL_EVERY_TRIAL=True`.

### Config additions to `config.py`
```python
PHASE3_MONTHLY_PENALTY_WEIGHT = 1.0
PHASE3_MONTHLY_FALLBACK_PENALTY = 5.0
PHASE4_MONTHLY_SCORE_WEIGHT = 0.70
PHASE4_MONTHLY_EVAL_EVERY_TRIAL = True
```

## Out of scope
- Do NOT add the `_is_positive_good` gate (Task 3).
- Do NOT add evaluator-failure-mode awareness (Task 4).
- Do NOT touch the GPU engine or EvoX runner.
- Do NOT change the JSON output format.
- Do NOT modify `evaluator_v5.ipynb`.

## Acceptance criteria
1. After Task 2, calling `Rule_Set_Selector.run(...)` (or whatever the
   public entry is) with `MONTHLY_VALIDATION_ENABLED=True` causes a
   monthly penalty to be added to the per-rule-set score.
2. After Task 2, the Optuna trial objective in Phase 4 (or the
   walk-forward grid, if Task 7 is not done) adds a `monthly_penalty`
   to the trial score.
3. A unit test confirms that two rule-sets with the same
   `total_return_pct` on validation but very different monthly
   performance get different composite scores. The one with better
   monthly performance (higher `profitable_ratio`, positive
   `equity_slope`) wins.
4. All existing imports / smoke tests pass.
5. New config keys `PHASE3_MONTHLY_PENALTY_WEIGHT`,
   `PHASE3_MONTHLY_FALLBACK_PENALTY`,
   `PHASE4_MONTHLY_SCORE_WEIGHT`, `PHASE4_MONTHLY_EVAL_EVERY_TRIAL`
   are present in `config.py` and accessible.
6. New unit test `tests/unit/test_monthly_wired_into_scoring.py`
   exists and passes.
7. No changes to `evaluator_v5.ipynb` or to the existing public
   function signatures that other code depends on.

## Constraints
- Stay on `feature/task-2-monthly-penalty-scoring`.
- 12.7 GiB RAM total — do not load the full `train.csv` in tests.
- PEP 8, type hints, module logger.
- Use only existing third-party deps (numpy, pandas, pytest, jax, evox).
- Branch off `main` (this task is reviewable in isolation, not stacked
  on task-1).
