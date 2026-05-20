# Implementation Plan: Enhanced Reporting Outputs

## Overview

Add five new methods to the existing `Reporter` class in `gpu_fuzzy_trader/reporting/reporter.py`, along with unit tests in `tests/unit/test_reporter.py` and property-based tests in a new `tests/property/test_reporter_properties.py`. All methods follow the existing 8-step pattern and project conventions.

## Tasks

- [x] 1. Add scipy import and implement `plot_per_rule_breakdown`
  - [x] 1.1 Add `from scipy.stats import spearmanr` import to `reporter.py`
    - Insert after the existing `matplotlib` imports, before the `logger` definition
    - _Requirements: 3.2_

  - [x] 1.2 Implement `plot_per_rule_breakdown` method in `Reporter`
    - Add `_compute_mdd` static helper method to `Reporter`
    - Implement direction validation (raise `ValueError` for non-`"long"`/`"short"`)
    - For each split, filter trade log by `Rule_Index` (1-based) and compute: `total_pnl`, `win_rate`, `num_trades`, `mdd_pct`
    - Build 2×2 subplot figure with grouped bars using fixed colors `"#4C72B0"` (train), `"#DD8452"` (validation), `"#55A868"` (test)
    - Label x-axis groups as `"Rule 1"`, `"Rule 2"`, etc.; set figure title to `f"Per-Rule Breakdown — {direction.capitalize()}"`
    - Save at 100 DPI, close with `plt.close(fig)`, log path at INFO, return absolute path
    - Handle `None`/empty trade logs gracefully (treat as zero trades)
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10_

  - [x] 1.3 Write unit tests for `plot_per_rule_breakdown` (`TestPlotPerRuleBreakdown`)
    - `test_creates_png_file`, `test_returns_correct_path`, `test_invalid_direction_raises`
    - `test_none_trade_log_does_not_raise`, `test_empty_trade_log_does_not_raise`
    - `test_zero_rule_trades_renders_zero_bar`, `test_file_is_nonzero_size`, `test_creates_parent_dirs`
    - _Requirements: 1.1, 1.6, 1.8, 1.9, 1.10_

- [x] 2. Implement `write_strategy_evaluation_table`
  - [x] 2.1 Implement `write_strategy_evaluation_table` method in `Reporter`
    - Validate `direction`; raise `ValueError` if invalid
    - Compute `num_rules = len(rule_set)` and `num_conditions = sum(len(r.get("conditions", [])) for r in rule_set)`
    - For each split: source `win_rate`, `mdd_pct` (from `max_drawdown_pct`), `total_return_pct`, `sortino_ratio`, `profit_factor` from `metrics_by_split`; default to `0.0` if absent/None
    - Compute `sharpe_ratio` from trade log: `r = Net_PnL / Equity_Before_Entry`; `mean(r) / std(r, ddof=1)` if `len(r) >= 2` else `0.0`
    - Write CSV with columns: `split`, `win_rate`, `mdd_pct`, `total_return_pct`, `num_rules`, `num_conditions`, `sortino_ratio`, `profit_factor`, `sharpe_ratio`; `index=False`
    - Save to `strategy_evaluation_{direction}.csv`, log path at INFO, return absolute path
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9_

  - [x] 2.2 Write unit tests for `write_strategy_evaluation_table` (`TestWriteStrategyEvaluationTable`)
    - `test_creates_csv_file`, `test_returns_correct_path`, `test_csv_has_required_columns`
    - `test_csv_has_three_rows`, `test_num_rules_matches_rule_set_length`, `test_num_conditions_matches_sum`
    - `test_sharpe_zero_for_single_trade`, `test_invalid_direction_raises`, `test_missing_metrics_defaults_to_zero`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.7, 2.9_

- [x] 3. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implement `write_spearman_correlation_report`
  - [x] 4.1 Implement `write_spearman_correlation_report` method in `Reporter`
    - Add `_spearman` static helper: drop NaN-paired rows, return `NaN` if fewer than 2 rows remain, call `spearmanr`, use `.statistic` with fallback to `.correlation`
    - Validate `direction`; raise `ValueError` if invalid
    - For each feature × split: check dataset/column presence; compute Spearman vs. `label_close_288`; record `NaN` on missing column, missing label, or insufficient rows
    - Build DataFrame with columns `feature`, `train_spearman`, `validation_spearman`, `test_spearman`
    - Sort by `abs(train_spearman)` descending, then by `feature` ascending (stable sort)
    - Write CSV with `index=False`, save to `spearman_correlation_{direction}.csv`, log path at INFO, return absolute path
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9_

  - [x] 4.2 Write unit tests for `write_spearman_correlation_report` (`TestWriteSpearmanCorrelationReport`)
    - `test_creates_csv_file`, `test_returns_correct_path`, `test_csv_has_required_columns`
    - `test_one_row_per_feature`, `test_absent_feature_column_records_nan`, `test_absent_label_column_records_nan`
    - `test_fewer_than_two_rows_records_nan`, `test_sorted_by_abs_train_spearman`, `test_invalid_direction_raises`
    - _Requirements: 3.1, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8_

- [x] 5. Implement `plot_distribution_and_equity`
  - [x] 5.1 Implement `plot_distribution_and_equity` method in `Reporter`
    - Validate `direction`; raise `ValueError` if invalid
    - For each split in `["train", "validation", "test"]`: skip (log WARNING, omit from return list) if trade log is `None` or empty
    - Compute `Concurrent_Open_Positions`: for each `idx` in `range(0, max(Release_Index) + 1)`, count trades where `Entry_Index <= idx < Release_Index`
    - Compute `Time_Between_Trades`: sort by `Entry_Index` ascending, take `.diff().dropna()`
    - Build 3-panel figure: `fig.add_subplot(2, 2, 1)` (concurrent histogram), `fig.add_subplot(2, 2, 2)` (time-between histogram), `fig.add_subplot(2, 1, 2)` (equity curve full-width)
    - Annotate equity curve with `^` (green `"#55A868"`) for `Net_PnL > 0` trades and `v` (red `"#C44E52"`) for `Net_PnL <= 0` trades
    - Save each figure at 100 DPI to `distribution_equity_{split}_{direction}.png`, close with `plt.close(fig)`, log path at INFO, append to return list
    - Return list of absolute paths (one per non-empty split)
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8_

  - [x] 5.2 Write unit tests for `plot_distribution_and_equity` (`TestPlotDistributionAndEquity`)
    - `test_creates_png_per_split`, `test_returns_list_of_paths`, `test_none_split_skipped`
    - `test_empty_split_skipped`, `test_return_list_length_matches_nonempty_splits`
    - `test_file_is_nonzero_size`, `test_invalid_direction_raises`
    - _Requirements: 4.1, 4.7, 4.8_

- [x] 6. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Implement `write_feature_stratified_performance`
  - [x] 7.1 Implement `write_feature_stratified_performance` method in `Reporter`
    - Validate `direction`; raise `ValueError` if invalid
    - For each split: if dataset is `None`/empty or trade log is `None`/empty, write header-only CSV and append path
    - For each feature × split: skip feature if column absent (log WARNING)
    - Build stratum lookup using vectorised approach: `valid_mask = trade_log["Entry_Index"].between(0, len(dataset) - 1)`; log WARNING for out-of-bounds indices
    - For each `fuzzy_value` (unique non-NaN strings in feature column): compute `num_trades`, `total_return_pct = sum(Net_PnL) / config.INITIAL_CAPITAL * 100`, `win_rate`, `sharpe_ratio`
    - Include rows for zero-trade strata with all metrics set to `0.0`
    - Write CSV per split with columns: `feature`, `fuzzy_value`, `split`, `num_trades`, `total_return_pct`, `win_rate`, `sharpe_ratio`; `index=False`
    - Save to `feature_stratified_{split}_{direction}.csv`, log path at INFO, return list of absolute paths
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 5.10_

  - [x] 7.2 Write unit tests for `write_feature_stratified_performance` (`TestWriteFeatureStratifiedPerformance`)
    - `test_creates_csv_per_split`, `test_returns_list_of_paths`, `test_csv_has_required_columns`
    - `test_absent_feature_column_skipped`, `test_zero_trade_stratum_row_has_zero_metrics`
    - `test_out_of_bounds_entry_index_skipped`, `test_invalid_direction_raises`
    - _Requirements: 5.1, 5.2, 5.7, 5.8, 5.9, 5.10_

- [x] 8. Create property-based test file
  - [x] 8.1 Create `tests/property/test_reporter_properties.py` with Hypothesis strategies
    - Add composite strategies: `rule_set_strategy`, `trade_log_strategy`, `metrics_strategy`, `dataset_with_features_strategy`, `split_logs_strategy`, `stratification_scenario_strategy`
    - All strategies generate valid inputs matching the data models defined in the design
    - _Requirements: 1.1, 2.1, 3.1, 4.1, 5.1_

  - [x] 8.2 Write property test for Property 1 — File creation round-trip
    - **Property 1: File creation round-trip (single-return methods)**
    - For valid inputs to `plot_per_rule_breakdown`, `write_strategy_evaluation_table`, `write_spearman_correlation_report`: returned path is non-empty, file exists at that path, path is absolute
    - `@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example])`
    - **Validates: Requirements 1.1, 1.8, 2.1, 3.1, 3.8, 6.6, 7.1**

  - [x] 8.3 Write property test for Property 2 — Invalid direction raises ValueError
    - **Property 2: Invalid direction raises ValueError**
    - For any string not in `("long", "short")`, all five new methods SHALL raise `ValueError` before creating any file
    - `@given(direction=st.text().filter(lambda s: s not in ("long", "short")))`
    - `@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example])`
    - **Validates: Requirements 1.9, 2.9**

  - [x] 8.4 Write property test for Property 3 — output_dir override is respected
    - **Property 3: output_dir override is respected**
    - For any valid inputs and any `output_dir`, all returned paths SHALL start with the provided `output_dir`
    - `@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example])`
    - **Validates: Requirements 6.4, 6.5, 6.6**

  - [x] 8.5 Write property test for Property 4 — Strategy evaluation table schema and rule_set counts
    - **Property 4: Strategy evaluation table schema and rule_set counts**
    - For any `rule_set` of length N with total condition count C: CSV has exactly 3 rows, exactly the required columns, `num_rules == N` and `num_conditions == C` in every row
    - `@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example])`
    - **Validates: Requirements 2.2, 2.3, 2.4**

  - [x] 8.6 Write property test for Property 5 — Sharpe ratio computation correctness
    - **Property 5: Sharpe ratio computation correctness**
    - For trade logs with ≥2 trades: `sharpe_ratio == mean(r) / std(r, ddof=1)` within 1e-9 tolerance; for <2 trades: `sharpe_ratio == 0.0`
    - `@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example])`
    - **Validates: Requirements 2.5, 5.6**

  - [x] 8.7 Write property test for Property 6 — Spearman correlation correctness and range invariant
    - **Property 6: Spearman correlation correctness and range invariant**
    - For ≥2 non-NaN paired rows: Spearman value matches `scipy.stats.spearmanr` directly and is a finite float in `[-1.0, 1.0]`
    - `@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example])`
    - **Validates: Requirements 3.2, 3.9**

  - [x] 8.8 Write property test for Property 7 — Spearman output sorted by absolute train correlation
    - **Property 7: Spearman output sorted by absolute train correlation**
    - Output CSV rows are sorted such that `abs(train_spearman)` is non-increasing, ties broken by `feature` ascending
    - `@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example])`
    - **Validates: Requirements 3.4**

  - [x] 8.9 Write property test for Property 8 — Distribution and equity skips empty splits
    - **Property 8: Distribution and equity skips empty splits**
    - For K non-empty splits: return list has exactly K paths, each pointing to an existing file
    - `@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example])`
    - **Validates: Requirements 4.1, 4.7, 4.8**

  - [x] 8.10 Write property test for Property 9 — Feature stratification metric correctness
    - **Property 9: Feature stratification metric correctness**
    - For any stratum of N trades: `num_trades == N`, `total_return_pct == sum(Net_PnL) / INITIAL_CAPITAL * 100`, `win_rate == count(Net_PnL > 0) / N` (or `0.0` if N == 0)
    - `@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example])`
    - **Validates: Requirements 5.3, 5.4, 5.5**

- [x] 9. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- The `_compute_mdd` and `_spearman` helpers are added as static methods on `Reporter` alongside the new public methods
- All five methods follow the existing 8-step pattern: resolve dir → validate direction → validate inputs → compute → save → close → log → return
- Property tests use `tmp_path` pytest fixture for `output_dir` override, consistent with existing unit tests
- Run tests with: `PYTHONPATH=. pytest tests/ --hypothesis-seed=42`

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "2.1"] },
    { "id": 2, "tasks": ["1.3", "2.2", "4.1"] },
    { "id": 3, "tasks": ["4.2", "5.1"] },
    { "id": 4, "tasks": ["5.2", "7.1"] },
    { "id": 5, "tasks": ["7.2", "8.1"] },
    { "id": 6, "tasks": ["8.2", "8.3", "8.4", "8.5", "8.6", "8.7", "8.8", "8.9", "8.10"] }
  ]
}
```
