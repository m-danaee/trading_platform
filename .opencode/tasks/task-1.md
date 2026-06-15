# Task 1 — Add `validation/monthly_windows.py` and `validation/rolling_cv.py`

## Why
The friend has a `validation/monthly_windows.py` module that evaluates
a rule-set on 30-day rolling windows and a `validation/rolling_cv.py`
that builds purged expanding-window folds. Together they form the
"robustness" backbone of the friend's pipeline: a candidate rule-set
is rejected unless it is profitable on >60% of months, has worst
monthly PF ≥ 0.85, and has a positive equity slope across windows.
This is the single largest gap in my pipeline.

I am copying these two files *conceptually* (NOT a verbatim file copy)
and adapting the imports / config keys to my codebase.

## Source for reference
The friend has these at:
- `friend_project/gpu_fuzzy_trader/validation/monthly_windows.py`
- `friend_project/gpu_fuzzy_trader/validation/rolling_cv.py`

My current `gpu_fuzzy_trader/validation/` is empty (only a
`__pycache__/`). I will create the new files there.

## Required behavior

### `validation/monthly_windows.py`
Exports:
- `MonthlyWindowSummary` dataclass with fields:
  `windows, profitable_windows, profitable_ratio, mean_return_pct,
   median_return_pct, worst_return_pct, latest_return_pct,
   recency_weighted_return_pct, mean_profit_factor,
   worst_profit_factor, worst_drawdown_pct, min_trades,
   mean_trades, equity_slope, max_equity_dip_pct, score`
- `build_monthly_windows(df, window_days=None, stride_days=None,
   min_rows=None, max_windows=None) -> list[pd.DataFrame]`
  Builds 30-day rolling windows over `df["datetime"]` (sorted
  chronologically). Each window must have at least `min_rows` rows
  (default 2500). At most `max_windows` windows (default 24).
- `summarize_monthly_metrics(metrics: Iterable[dict]) -> MonthlyWindowSummary`
- `evaluate_rule_set_monthly(df, rule_set, direction,
   feature_names=None) -> tuple[MonthlyWindowSummary, list[dict]]`
  Calls `CPUBacktestEngine.simulate_rule_set(rule_set)` on each
  monthly window and returns a `MonthlyWindowSummary` plus the raw
  per-window metrics.
- `monthly_penalty(summary: MonthlyWindowSummary) -> float`
  Computes a non-negative penalty from the summary, weighted by
  `MONTHLY_*` config keys (see below). If `summary.windows <= 0`,
  returns 100.0.

### `validation/rolling_cv.py`
Exports:
- `PurgedFold` dataclass with: `fold_id, train_df, valid_df,
  train_end_bar, valid_start_bar, valid_end_bar`
- `FoldMetricsSummary` dataclass with: `folds, worst_return_pct,
  worst_profit_factor, worst_sortino_ratio, worst_drawdown_pct,
  min_trades, mean_return_pct, mean_profit_factor, metrics`
- `build_purged_rolling_folds(df, n_splits=None, embargo_candles=None,
   min_train_frac=None, min_valid_rows=None) -> list[PurgedFold]`
  Per-symbol expanding-window folds with embargo. Uses config
  defaults `PURGED_CV_N_SPLITS=4`, `PURGED_CV_EMBARGO_CANDLES=288`,
  `PURGED_CV_MIN_TRAIN_FRACTION=0.45`, `PURGED_CV_MIN_VALID_ROWS=5000`.
- `build_fold_engines(df, direction, feature_names=None, n_splits=None,
   embargo_candles=None) -> list[CPUBacktestEngine]`
  Returns a `CPUBacktestEngine` per fold with the fold id stored on
  `engine._fold_id`.
- `summarize_fold_metrics(metrics) -> FoldMetricsSummary`
- `evaluate_rule_set_on_fold_engines(rule_set, fold_engines) -> FoldMetricsSummary`

## Config keys to add to `gpu_fuzzy_trader/config.py`
Add (or update) the following constants with the same defaults the
friend uses:

```python
# Monthly windows validation
MONTHLY_VALIDATION_ENABLED = True
MONTHLY_WINDOW_DAYS = 30
MONTHLY_WINDOW_STRIDE_DAYS = 30
MONTHLY_WINDOW_MIN_ROWS = 2500
MONTHLY_WINDOW_MAX_WINDOWS = 24
MONTHLY_RECENCY_WEIGHT = 2.2
MONTHLY_MIN_TRADES = 20
MONTHLY_MIN_PROFITABLE_RATIO = 0.60
MONTHLY_WORST_RETURN_FLOOR = -1.5
MONTHLY_WORST_PF_FLOOR = 0.85
MONTHLY_MAX_DD = 8.0
MONTHLY_WORST_RETURN_WEIGHT = 1.2
MONTHLY_WORST_PF_WEIGHT = 8.0
MONTHLY_DD_WEIGHT = 0.7
MONTHLY_PROFITABLE_RATIO_WEIGHT = 15.0
MONTHLY_TREND_WEIGHT = 2.0
MONTHLY_LATEST_WEIGHT = 0.6
```

The purged-CV keys (`PURGED_CV_*`) already exist in my config; I will
not duplicate them.

## Adaptations to my codebase
- Import path: `from gpu_fuzzy_trader import config as _cfg`
- Engine: use my existing
  `from gpu_fuzzy_trader.backtest.cpu_engine import CPUBacktestEngine`
  (no need to import the friend's `rb_evaluator_v5.py`).
- Use my existing `slim_backtest_df` from
  `gpu_fuzzy_trader.backtest.df_slim` when `feature_names` is given.
- Empty `validation/__init__.py` (currently missing) must be created.

## Files I will touch
- `gpu_fuzzy_trader/validation/__init__.py` (new, empty)
- `gpu_fuzzy_trader/validation/monthly_windows.py` (new)
- `gpu_fuzzy_trader/validation/rolling_cv.py` (new)
- `gpu_fuzzy_trader/config.py` (add the `MONTHLY_*` constants)

## Out of scope for this task
- Do NOT wire `monthly_penalty` into Phase 3/4 yet (Task 2).
- Do NOT add the `_is_positive_good` gate (Task 3).
- Do NOT modify Phase 2 or Phase 3 logic in any way.
- Do NOT modify `evaluator_v5.ipynb`.

## Acceptance criteria
1. `from gpu_fuzzy_trader.validation.monthly_windows import
   evaluate_rule_set_monthly, monthly_penalty, MonthlyWindowSummary`
   works without error.
2. `from gpu_fuzzy_trader.validation.rolling_cv import
   build_purged_rolling_folds, evaluate_rule_set_on_fold_engines,
   PurgedFold, FoldMetricsSummary` works.
3. Calling `evaluate_rule_set_monthly` on a small synthetic
   `pd.DataFrame` (with `datetime`, `symbol`, `label_open_next`,
   `label_close_288`, `label_min_288`, `label_max_288`,
   `label_max_before_min`) and a 1-rule set returns
   `(MonthlyWindowSummary, list[dict])` with at least one window
   in the list. The summary's `windows` field is ≥ 1.
4. Calling `build_purged_rolling_folds` on a synthetic 1000-row
   DataFrame returns a list of at least 2 `PurgedFold` objects when
   `PURGED_CV_N_SPLITS=4` and `min_train_frac=0.45`. Each fold has
   non-empty `train_df` and `valid_df`.
5. `monthly_penalty(MonthlyWindowSummary(windows=0, ...))` returns
   exactly `100.0`.
6. Running `python -c "from gpu_fuzzy_trader.validation import
   monthly_windows, rolling_cv"` from the project root succeeds
   (no ImportError).
7. The new `MONTHLY_*` config keys are present in `config.py` and
   accessible via `import gpu_fuzzy_trader.config as cfg; cfg.MONTHLY_VALIDATION_ENABLED`
   returns `True`.
8. Existing unit tests under `tests/` that import the package still
   import cleanly (do not break the import graph). Do NOT run the
   full test suite (per `AGENTS.md` RAM limit) — only a smoke import.

## Notes
- The user has 12.7 GiB RAM. Avoid loading the full `train.csv` in
  the synthetic test — use a small `pd.DataFrame(range(3000))` with
  the required label columns filled with 0 or NaN-safe values.
- The implementer may use a `__main__` block guarded by `if __name__
  == "__main__":` for a quick smoke test that builds a small
  DataFrame and calls the new functions, but this is optional.
- Do not introduce any new third-party dependencies.
- Keep code style consistent with the rest of the project (PEP 8,
  type hints, module-level logger).
