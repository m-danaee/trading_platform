# Phase 5 — Out-of-Sample Evaluation

**Module:** `gpu_fuzzy_trader/phases/phase5_oos.py` → `OOS_Evaluator`

Phase 5 is the final, honest evaluation of the strategy. It loads the optimized strategies from Phase 4 and evaluates them on `data/test.csv` — data that has never been seen by any phase of the pipeline. This is the only number that should be treated as a true estimate of live performance.

---

## 1. Why Phase 5 is Special

Every other phase uses `data/train.csv` (split into training and validation). The test set is held out completely. This means:

- Phase 1 feature selection was done on training data only.
- Phase 2 rule evolution used training data with validation/CV gates (`SPLIT_MODE` controls single split vs purged rolling CV).
- Phase 3 rule set selection used training-target objectives with validation/CV gates.
- Phase 4 risk optimization was done on the persisted `validation_25` split (last CV fold when using `purged_rolling_cv`).

The test set has never influenced any decision. Phase 5 is therefore a genuine out-of-sample evaluation.

**Critical rule:** Never use `TEST_CSV_PATH` in Phases 1–4. The config comment explicitly marks it as "Phase 5 OOS only — never use in Phases 1–4."

---

## 2. Data Preparation

Phase 5 prepares the test data using the **identical pipeline** as training:

1. Load `data/test.csv` via `Data_Loader.load_dataset`
2. Sort by (symbol, datetime)
3. Drop last 288 rows per symbol (`TAIL_DROP_ROWS`)
4. Drop rows where any label column is NaN
5. Fill feature NaN with 0
6. Compute `_symbol_bar_index`

This is critical for consistency: if the test data were prepared differently (e.g., without the tail drop), the backtest results would not be comparable to the training/validation results.

Phase 5 also loads `train_75.parquet` and `validation_25.parquet` to evaluate on train, validation, and test for comparison. Reports include:

- `outputs/reports/strategy_evaluation_{long,short}.csv` — side-by-side metrics
- `outputs/reports/generalization_diagnostics_{long,short}.json` — sign flips, train→val→test deltas, feature bucket concentration
- Equity curves per split (`train_*`, `validation_*`, `test_*`)

**Reading results with purged CV:** Strong train + val but weak test usually means **regime shift** into `test.csv` (e.g. calendar period after `train.csv`), not just a bad 75/25 split. Check `generalization_diagnostics_*.json` for `train_to_test_sign_flip`.

---

## 3. Strategy Loading — `OOS_Evaluator.load_strategies`

Loads `outputs/long.json` and `outputs/short.json` via `Output_Writer.load_and_validate`. If a file is missing or fails schema validation, that direction is silently skipped (with a WARNING log). Phase 5 can run with only one direction if the other failed.

---

## 4. Evaluation — `_evaluate_strategy`

For each strategy and each split (train, validation, test):

```python
metrics, trade_log = engine.simulate_rule_set(rule_set, return_logs=True)
```

`return_logs=True` returns a detailed trade log DataFrame in addition to the summary metrics. This is used for equity curve plots, per-rule breakdowns, and distribution analysis.

### Zero-trade handling

If `executed_trades == 0`:
- `total_return_pct` is set to 0.0 (not left as whatever the engine returned).
- `account_ruined` is set to `False` (a strategy that never trades cannot ruin the account).

This prevents misleading reports for strategies that happen to generate no signals on the test set.

### Negative return warning

If `test_return < −5.0%`, a WARNING is logged:
```
Phase 5 [long]: FAIL — test return -8.23% is negative. Strategy does not generalize.
```

This is a diagnostic signal, not a hard failure. The pipeline continues and saves the report.

---

## 5. Metrics Reported

All metrics come from `CPUBacktestEngine.simulate_rule_set`:

| Metric | Formula | Interpretation |
|---|---|---|
| `total_return_pct` | `(final_equity / INITIAL_CAPITAL − 1) × 100` | Overall return on the test period |
| `max_drawdown_pct` | `max((peak − equity) / peak × 100)` | Worst peak-to-trough decline |
| `win_rate` | `wins / executed_trades × 100` | Fraction of trades that were profitable |
| `profit_factor` | `gross_profit / gross_loss` | Ratio of total wins to total losses (99.0 if no losses with wins) |
| `executed_trades` | Count | Total number of trades executed |
| `account_status` | `"survived"` or `"ruined"` | Whether equity reached zero |
| `final_equity` | Currency | Ending equity |
| `per_symbol_metrics` | Dict | Per-symbol trade count, win rate, net PnL |

### Per-symbol breakdown

For each symbol, the report includes:
- `trade_count`: Number of trades on this symbol
- `win_rate`: Win rate on this symbol
- `net_pnl`: Net profit/loss on this symbol

This breakdown is saved to `outputs/reports/test_per_symbol_performance.csv` and is essential for diagnosing whether the strategy works uniformly across the portfolio or is driven by one or two symbols.

---

## 6. Cross-Split Reporting

Phase 5 evaluates the strategy on all three splits and generates comparative reports:

### `write_strategy_evaluation_table`
A table comparing train/validation/test metrics side by side. This is the primary tool for diagnosing overfitting:
- If train >> validation >> test: severe overfitting.
- If train ≈ validation ≈ test: good generalization.
- If test > validation: lucky test set (possible, but treat with caution).

### `plot_equity_curve`
Equity curves for each split. A strategy that shows a smooth upward curve on training but a flat or declining curve on test is overfitting.

### `plot_per_rule_breakdown`
Per-rule contribution to the strategy's performance. If one rule accounts for 90% of trades and the other rules contribute nothing, the strategy is effectively a single-rule strategy.

### `plot_distribution_and_equity`
Distribution of per-trade returns (histogram) and equity curve. A healthy distribution has more positive returns than negative, with the positive tail extending further.

### `write_spearman_correlation_report`
Spearman rank correlation between selected features and trade outcomes across splits. This helps identify which features are driving the strategy's performance and whether their predictive power is consistent across time periods.

### `write_feature_stratified_performance`
Performance broken down by feature value. For each selected feature, shows how the strategy performs when that feature is in each fuzzy state. This is useful for understanding which market conditions the strategy exploits.

---

## 7. Interpreting Results

### Good signs
- `total_return_pct > 0` on test
- `max_drawdown_pct < 20%` on test
- `win_rate > 45%` on test
- `profit_factor > 1.2` on test
- Per-symbol performance is relatively uniform (no single symbol driving all returns)
- Test metrics are within 50% of validation metrics (not dramatically worse)

### Warning signs
- `total_return_pct < 0` on test: strategy does not generalize
- `max_drawdown_pct > 30%` on test: excessive risk
- `executed_trades < 50` on test: too few trades for statistical significance
- One symbol accounts for >50% of net PnL: strategy is not diversified
- Test metrics are dramatically worse than validation: overfitting to validation

### Common failure modes

**Overfitting to validation (Phase 3/4 leakage):** If Phase 3 or Phase 4 was run many times with different hyperparameters and the best result was selected, the validation set has been implicitly used for model selection. The test set will show worse performance.

**Regime shift:** The test period may be in a different market regime than the training/validation period. Check the regime cluster model to see if the test data falls in a regime that was well-represented in training.

**Low trade count:** If the strategy generates very few trades on the test set, the metrics are noisy. A strategy with 20 trades and 60% win rate is not statistically different from 50% win rate.

---

## 8. No Phase-Specific Config Parameters

Phase 5 has no dedicated config parameters. It uses:
- `TEST_CSV_PATH` — path to the test CSV
- `REPORTS_DIR` — output directory for reports
- All backtest constants from Phase 0 (`INITIAL_CAPITAL`, `FEE_PCT`, etc.)

---

## 9. Outputs

| File | Content |
|---|---|
| `outputs/reports/test_long_report.json` | Summary metrics for the long strategy on test data |
| `outputs/reports/test_short_report.json` | Summary metrics for the short strategy on test data |
| `outputs/reports/test_per_symbol_performance.csv` | Per-symbol metrics for both directions |
| `outputs/reports/test_long_equity.png` | Equity curve on test data (long) |
| `outputs/reports/test_short_equity.png` | Equity curve on test data (short) |
| `outputs/reports/strategy_evaluation_long.csv` | Cross-split comparison table (long) |
| `outputs/reports/strategy_evaluation_short.csv` | Cross-split comparison table (short) |
| `outputs/reports/per_rule_breakdown_long.png` | Per-rule contribution (long) |
| `outputs/reports/per_rule_breakdown_short.png` | Per-rule contribution (short) |
| `outputs/reports/distribution_equity_long.png` | Return distribution + equity curve (long) |
| `outputs/reports/distribution_equity_short.png` | Return distribution + equity curve (short) |
| `outputs/reports/spearman_correlation_long.csv` | Feature-outcome Spearman correlations (long) |
| `outputs/reports/spearman_correlation_short.csv` | Feature-outcome Spearman correlations (short) |
| `outputs/reports/feature_stratified_long.csv` | Performance by feature value (long) |
| `outputs/reports/feature_stratified_short.csv` | Performance by feature value (short) |
