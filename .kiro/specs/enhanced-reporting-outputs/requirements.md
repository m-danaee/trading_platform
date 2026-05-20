# Requirements Document

## Introduction

This feature extends the GPU-Fuzzy Trading Pipeline's `Reporter` class with five new reporting outputs that provide deeper insight into strategy quality, feature relevance, and trade distribution. The new reports are generated after Phase 5 (out-of-sample evaluation) and cover all three data splits: train, validation, and test. All outputs are saved to `outputs/reports/` and follow the existing project conventions (PNG for plots, CSV for tabular data).

## Glossary

- **Reporter**: The class in `gpu_fuzzy_trader/reporting/reporter.py` responsible for generating all visual and tabular reports.
- **CPUBacktestEngine**: The backtest engine in `gpu_fuzzy_trader/backtest/cpu_engine.py` that simulates a rule set and returns metrics and a trade log.
- **Rule**: A single entry in a strategy's `rules_set` list, containing `tp`, `sl`, `capital_pct`, and `conditions`.
- **Rule_Set**: The full list of rules in a strategy file (`long.json` or `short.json`).
- **Strategy**: A validated JSON object with a `direction` and a `rules_set`, as defined by `Output_Writer`.
- **Split**: One of the three dataset partitions: `train` (train_75.parquet), `validation` (validation_25.parquet), or `test` (test.csv).
- **Trade_Log**: The `pd.DataFrame` returned by `CPUBacktestEngine.simulate_rule_set(return_logs=True)`, containing one row per executed trade with columns including `Rule_Index`, `Net_PnL`, `Equity_After`, `Equity_Before_Entry`, `Entry_Index`, `Release_Index`, `Entry_Time`, `Close_Time`, `Symbol`, and `Exit_Reason`.
- **Forward_Return**: The per-row value of `label_close_288` in the dataset, representing the percentage price change over the next 288 candles from the open.
- **Selected_Features**: The list of feature dicts loaded from `outputs/selected_features_{direction}.json`, each with keys `name`, `mode`, and `score`.
- **MDD (Max Drawdown Pct)**: The maximum percentage drop from peak equity to trough equity within a given trade log, computed as `(peak - trough) / peak * 100`.
- **Sortino_Ratio**: A risk-adjusted return metric computed from per-trade returns using only downside deviation, as implemented in `_sortino_ratio_from_returns` in `cpu_engine.py`.
- **Sharpe_Ratio**: Mean per-trade net return rate divided by the standard deviation of per-trade net return rates, where per-trade net return rate = `Net_PnL / Equity_Before_Entry`.
- **Profit_Factor**: Gross winning PnL divided by gross losing PnL, as computed by `_safe_profit_factor` in `cpu_engine.py`.
- **Win_Rate**: The fraction of executed trades with positive net PnL.
- **Spearman_Correlation**: A rank-based correlation coefficient measuring the monotonic relationship between a feature and forward returns, computed via `scipy.stats.spearmanr` after dropping NaN-paired rows.
- **Concurrent_Open_Positions**: The number of trades simultaneously open at a given candle index, derived from overlapping `Entry_Index` and `Release_Index` intervals in the Trade_Log.
- **Time_Between_Trades**: The number of candles between the `Entry_Index` of consecutive trades in the Trade_Log, sorted by `Entry_Index` ascending; the first trade is excluded.
- **Equity_Curve**: A time-series plot of `Equity_After` values from the Trade_Log (y-axis) against trade sequence number (x-axis), annotated with trade entry and exit markers.
- **Feature_Stratum**: A subset of trades where the dataset row at `Entry_Index` has a given feature column equal to a specific fuzzy value string (e.g., "High", "Low", "Medium").

---

## Requirements

### Requirement 1: Per-Rule Breakdown Image

**User Story:** As a strategy analyst, I want a visual breakdown of each rule's individual performance across train, validation, and test splits, so that I can identify which rules contribute positively and which degrade on unseen data.

#### Acceptance Criteria

1. WHEN `Reporter.plot_per_rule_breakdown` is called with a `Rule_Set`, a `Trade_Log` per split, and a `direction` of either `"long"` or `"short"`, THE `Reporter` SHALL produce one PNG image saved to `outputs/reports/per_rule_breakdown_{direction}.png` and return its absolute path.
2. THE `Reporter` SHALL display, for each rule in the `Rule_Set`, a grouped bar chart with one group per rule index showing four metrics: total PnL, win rate, number of trades, and Max Drawdown Pct (computed as the maximum percentage drop from peak to trough equity within the filtered trade log for that rule).
3. THE `Reporter` SHALL render each metric group with three bars — one per split — using the fixed colors `"#4C72B0"` for train, `"#DD8452"` for validation, and `"#55A868"` for test, applied consistently across all metric subplots.
4. THE `Reporter` SHALL label each bar group on the x-axis as `"Rule 1"`, `"Rule 2"`, etc., using 1-based indexing matching the position in the `Rule_Set`.
5. THE `Reporter` SHALL compute per-rule metrics by filtering the `Trade_Log` on the `Rule_Index` column (0-based integer) for each split independently.
6. IF a rule generates zero trades on a given split, THEN THE `Reporter` SHALL render that bar with a value of `0.0` and SHALL NOT raise an error.
7. THE `Reporter` SHALL include a legend identifying the three splits and set the figure title to `"Per-Rule Breakdown — {direction.capitalize()}"`.
8. THE `Reporter` SHALL save the figure at 100 DPI, close the matplotlib figure with `plt.close(fig)` after saving, and return the absolute path to the saved PNG.
9. IF `direction` is not `"long"` or `"short"`, THEN THE `Reporter` SHALL raise a `ValueError` before creating any figure.
10. IF a split's `Trade_Log` is `None` or an empty `pd.DataFrame`, THEN THE `Reporter` SHALL treat all rules as having zero trades for that split and SHALL NOT raise an error.

---

### Requirement 2: Strategy Evaluation Table

**User Story:** As a strategy analyst, I want a comprehensive performance table for the final long and short strategies evaluated across all three splits, so that I can assess overfitting and overall strategy quality at a glance.

#### Acceptance Criteria

1. WHEN `Reporter.write_strategy_evaluation_table` is called with a metrics dict per split, a `Trade_Log` per split, a `Rule_Set`, and a `direction` of either `"long"` or `"short"`, THE `Reporter` SHALL produce one CSV file saved to `outputs/reports/strategy_evaluation_{direction}.csv` and return its absolute path.
2. THE `Reporter` SHALL include one row per split (train, validation, test) with the following columns: `split`, `win_rate`, `mdd_pct`, `total_return_pct`, `num_rules`, `num_conditions`, `sortino_ratio`, `profit_factor`, `sharpe_ratio`.
3. THE `Reporter` SHALL compute `num_rules` as `len(Rule_Set)`.
4. THE `Reporter` SHALL compute `num_conditions` as the sum of `len(rule["conditions"])` for each rule in the `Rule_Set`.
5. THE `Reporter` SHALL compute `sharpe_ratio` as `mean(trade_returns) / std(trade_returns)` where `trade_returns = Net_PnL / Equity_Before_Entry` for each trade in the split's `Trade_Log`; IF fewer than two trades exist, THEN THE `Reporter` SHALL record `sharpe_ratio` as `0.0`.
6. THE `Reporter` SHALL source `win_rate`, `mdd_pct` (from key `max_drawdown_pct`), `total_return_pct`, `sortino_ratio`, and `profit_factor` from the metrics dict returned by `CPUBacktestEngine.simulate_rule_set`.
7. IF a split's metrics dict is absent or the split produced zero trades, THEN THE `Reporter` SHALL set `win_rate`, `mdd_pct`, `total_return_pct`, `sortino_ratio`, `profit_factor`, and `sharpe_ratio` to `0.0` for that split, while still populating `num_rules` and `num_conditions` from the `Rule_Set`, and SHALL NOT raise an error.
8. WHEN metrics data is unavailable for all splits, THE `Reporter` SHALL still create the CSV file with zero-valued financial metric rows and return its absolute path.
9. IF `direction` is not `"long"` or `"short"`, THEN THE `Reporter` SHALL raise a `ValueError` before creating any file.

---

### Requirement 3: Spearman Correlation Report

**User Story:** As a feature engineer, I want to see the Spearman correlation between each selected feature and forward returns on each data split, so that I can verify that feature relevance holds out-of-sample.

#### Acceptance Criteria

1. WHEN `Reporter.write_spearman_correlation_report` is called with a dataset per split, a list of `Selected_Features`, and a `direction` of either `"long"` or `"short"`, THE `Reporter` SHALL produce one CSV file saved to `outputs/reports/spearman_correlation_{direction}.csv`, where `direction` is used solely to construct the output filename.
2. THE `Reporter` SHALL compute the Spearman correlation coefficient between each feature's column values and the `label_close_288` column independently for each split by dropping all rows where either the feature column or the `label_close_288` column contains a NaN value before computing the correlation.
3. THE `Reporter` SHALL include one row per feature with the following columns: `feature`, `train_spearman`, `validation_spearman`, `test_spearman`.
4. THE `Reporter` SHALL sort rows by the absolute value of `train_spearman` in descending order, using the feature name as a secondary sort key in ascending lexicographic order to break ties.
5. IF a feature column is absent from a split's DataFrame, THEN THE `Reporter` SHALL record `NaN` for that split's correlation value and SHALL NOT raise an error.
6. IF a split's DataFrame contains fewer than two non-NaN paired rows for a given feature after dropping NaN values, THEN THE `Reporter` SHALL record `NaN` for that feature on that split and SHALL NOT raise an error.
7. IF the `label_close_288` column is absent from a split's DataFrame, THEN THE `Reporter` SHALL record `NaN` for all features on that split and SHALL NOT raise an error.
8. THE `Reporter` SHALL return the absolute path to the saved CSV file.
9. FOR ALL features present in all three splits with at least two non-NaN paired rows, the Spearman correlation values in the saved CSV SHALL be finite floats in the range `[-1.0, 1.0]` inclusive.

---

### Requirement 4: Distribution and Annotated Equity Curve

**User Story:** As a risk analyst, I want histograms of concurrent open positions and time-between-trades, plus an equity curve annotated with trade entry and exit points, so that I can understand the strategy's exposure profile and timing behavior.

#### Acceptance Criteria

1. WHEN `Reporter.plot_distribution_and_equity` is called with a `Trade_Log` per split and a `direction` of either `"long"` or `"short"`, THE `Reporter` SHALL produce one PNG image per split saved to `outputs/reports/distribution_equity_{split}_{direction}.png`.
2. THE `Reporter` SHALL arrange each figure with three subplots: (top-left) histogram of `Concurrent_Open_Positions`, (top-right) histogram of `Time_Between_Trades`, and (bottom) the annotated equity curve.
3. THE `Reporter` SHALL compute `Concurrent_Open_Positions` for each candle index in the range `[0, max(Release_Index)]` inclusive by counting the number of trades in the `Trade_Log` whose `Entry_Index` is less than or equal to the candle index and whose `Release_Index` is greater than the candle index.
4. THE `Reporter` SHALL compute `Time_Between_Trades` as the difference in `Entry_Index` between consecutive trades sorted by `Entry_Index` ascending; the first trade SHALL be excluded from this series.
5. THE `Reporter` SHALL plot the equity curve with trade sequence number (1 to N) on the x-axis and `Equity_After` on the y-axis.
6. THE `Reporter` SHALL annotate the equity curve with upward-pointing triangle markers (`^`) at each trade entry point and downward-pointing triangle markers (`v`) at each trade exit point, using green (`"#55A868"`) for trades where `Net_PnL > 0` and red (`"#C44E52"`) for trades where `Net_PnL <= 0`.
7. IF the `Trade_Log` for a split is `None` or an empty `pd.DataFrame`, THEN THE `Reporter` SHALL skip PNG generation for that split, log a WARNING, and SHALL NOT raise an error; the return list SHALL omit the path for that split.
8. THE `Reporter` SHALL save each figure at 100 DPI, close the matplotlib figure with `plt.close(fig)` after saving, and return a list of absolute paths to the saved PNG files (one per split that had trades).

---

### Requirement 5: Feature-Stratified Performance

**User Story:** As a strategy analyst, I want to see strategy performance broken down by the fuzzy value strata of each selected feature, so that I can understand under which market conditions the strategy performs best and worst.

#### Acceptance Criteria

1. WHEN `Reporter.write_feature_stratified_performance` is called with a `Trade_Log` per split, a `Rule_Set`, a list of `Selected_Features`, the full dataset per split, and a `direction` of either `"long"` or `"short"`, THE `Reporter` SHALL produce one CSV file per split saved to `outputs/reports/feature_stratified_{split}_{direction}.csv`.
2. THE `Reporter` SHALL produce one row per `(feature, fuzzy_value)` combination that appears in the dataset, with the following columns: `feature`, `fuzzy_value`, `split`, `num_trades`, `total_return_pct`, `win_rate`, `sharpe_ratio`.
3. THE `Reporter` SHALL determine the set of fuzzy values for each feature as the unique non-NaN string values present in that feature's column across the dataset for that split; a trade belongs to a stratum if the dataset row at `Entry_Index` has the feature column equal to the fuzzy value string.
4. THE `Reporter` SHALL compute `total_return_pct` for a stratum as `sum(Net_PnL) / config.INITIAL_CAPITAL * 100`; IF `config.INITIAL_CAPITAL` is zero, THEN THE `Reporter` SHALL record `total_return_pct` as `0.0`.
5. THE `Reporter` SHALL compute `win_rate` for a stratum as `count(Net_PnL > 0) / num_trades`; IF `num_trades` is zero, THEN THE `Reporter` SHALL record `win_rate` as `0.0`.
6. THE `Reporter` SHALL compute `sharpe_ratio` for a stratum as `mean(Net_PnL / Equity_Before_Entry) / std(Net_PnL / Equity_Before_Entry)`; IF fewer than two trades exist in the stratum, THEN THE `Reporter` SHALL record `sharpe_ratio` as `0.0`.
7. IF a feature column is absent from the dataset for a given split, THEN THE `Reporter` SHALL skip that feature for that split and SHALL NOT raise an error.
8. IF a stratum contains zero trades, THEN THE `Reporter` SHALL include a row for that stratum with `num_trades = 0` and all metric columns set to `0.0`.
9. THE `Reporter` SHALL return a list of absolute paths to the saved CSV files (one per split).
10. IF `Entry_Index` values in the `Trade_Log` are out of bounds for the dataset index, THEN THE `Reporter` SHALL skip those trades for stratification, log a WARNING, and SHALL NOT raise an error.

---

### Requirement 6: Report Output Path Consistency

**User Story:** As a pipeline operator, I want all new report files to follow the existing naming and directory conventions, so that the output directory remains predictable and organized.

#### Acceptance Criteria

1. THE `Reporter` SHALL write all new report files to the directory specified by `config.REPORTS_DIR` (`outputs/reports/`) when no `output_dir` override is provided.
2. WHEN no `output_dir` override is provided, THE `Reporter` SHALL call `_ensure_dir(config.REPORTS_DIR)` before writing any file to create the directory if it does not already exist.
3. WHEN an `output_dir` override is provided, THE `Reporter` SHALL NOT call `_ensure_dir`; IF the directory does not exist, the resulting `FileNotFoundError` SHALL propagate to the caller.
4. THE `Reporter` SHALL name PNG files using the pattern `{report_type}_{split}_{direction}.png`.
5. THE `Reporter` SHALL name CSV files that aggregate metrics across all splits using the pattern `{report_type}_{direction}.csv`, and CSV files that contain per-split data using the pattern `{report_type}_{split}_{direction}.csv`.
6. WHEN an `output_dir` parameter is provided to any new `Reporter` method, THE `Reporter` SHALL use that directory for all file writes in that method call instead of `config.REPORTS_DIR`.
7. THE `Reporter` SHALL log the absolute path of each saved file at INFO level using the module-level `logger`.

---

### Requirement 7: Graceful Handling of Missing or Incomplete Data

**User Story:** As a pipeline operator, I want the new reporting methods to handle missing data, empty trade logs, and absent splits without crashing the pipeline, so that partial pipeline runs still produce whatever reports are possible.

#### Acceptance Criteria

1. IF a `Trade_Log` passed to any new `Reporter` method is `None` or an empty `pd.DataFrame`, THEN THE `Reporter` SHALL write the output file to disk — saving a blank axes figure for plot methods or a header-only zero-row file for CSV methods — return its absolute path as a non-empty string, and log a WARNING.
2. IF a dataset split DataFrame passed to any new `Reporter` method is `None` or empty, THEN THE `Reporter` SHALL skip computation for that split, fill numeric metric fields with `0.0` and aggregated fields that have no meaningful zero value with `NaN`, and SHALL NOT raise an error.
3. IF a required column (`Rule_Index`, `Net_PnL`, `Equity_After`, `Equity_Before_Entry`, `Entry_Index`, or `Release_Index`) is absent from the `Trade_Log`, THEN THE `Reporter` SHALL write an empty output file to disk, return its absolute path as a non-empty string, and log an ERROR without raising an unhandled exception.
4. WHEN all required columns are present in the `Trade_Log`, THE `Reporter` SHALL validate that numeric columns (`Net_PnL`, `Equity_After`, `Equity_Before_Entry`, `Entry_Index`, `Release_Index`) contain numeric data types.
5. IF a numeric column (`Net_PnL`, `Equity_After`, `Equity_Before_Entry`, `Entry_Index`, or `Release_Index`) contains non-numeric data, THEN THE `Reporter` SHALL write an empty output file to disk, return its absolute path as a non-empty string, and log an ERROR without raising an unhandled exception.
6. THE `Reporter` SHALL NOT call `plt.show()` in any new method; all figures SHALL be saved to disk and closed with `plt.close(fig)`.
7. IF both the `Trade_Log` and all split DataFrames are empty simultaneously, THEN THE `Reporter` SHALL apply the per-method empty-input behavior defined in criteria 1 and 2 for each affected method and SHALL NOT raise an error.
