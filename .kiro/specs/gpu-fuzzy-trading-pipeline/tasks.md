# Implementation Plan: GPU-Fuzzy Trading Pipeline

## Overview

Implement the `gpu_fuzzy_trader` Python package end-to-end: project scaffold and config, data loading and splitting, feature detection and encoding, CPU and GPU backtest engines, five pipeline phases (feature selection, rule pool generation, rule set selection, RL risk optimization, out-of-sample test), output writing, reporting, and the top-level orchestrator. Each task builds on the previous and ends with all components wired together.

## Tasks

- [x] 1. Scaffold project structure and implement `config.py`
  - Create the `gpu_fuzzy_trader/` package directory with `__init__.py` files for all sub-packages: `data/`, `features/`, `backtest/`, `phases/`, `output/`, `reporting/`
  - Implement `gpu_fuzzy_trader/config.py` with all constants and paths listed in the design: `TRAIN_CSV_PATH`, `TEST_CSV_PATH`, `TRAIN_75_PATH`, `VALIDATION_25_PATH`, `OUTPUTS_DIR`, `REPORTS_DIR`, `LABEL_COLUMNS`, `META_COLUMNS`, `TAIL_DROP_ROWS`, `INITIAL_CAPITAL`, `LEVERAGE`, `FEE_PCT`, `MAX_HOLD_CANDLES`, `MAX_TOTAL_EXPOSURE_PCT`, `MIN_POSITION_NOTIONAL`, `PHASE2_TP`, `PHASE2_SL`, `PHASE2_CAPITAL_PCT`, `MIN_CONDITIONS`, `MAX_CONDITIONS`, `MIN_TRADE_SUPPORT`, `PHASE2_POPULATION_SIZE`, `PHASE2_GENERATIONS`, `PHASE2_ALGORITHM`, `PHASE3_*`, `PHASE4_*`, `PHASE1_*`
  - Create `outputs/` and `outputs/reports/` directories
  - _Requirements: 1.1, 1.2, 1.3, 13.5_

- [x] 2. Implement data loading and splitting
  - [x] 2.1 Implement `gpu_fuzzy_trader/data/loader.py` — `Data_Loader`
    - `load_dataset(path)`: read CSV, parse datetime, sort by (symbol, datetime), drop last `TAIL_DROP_ROWS` rows per symbol, drop NaN label rows, fill feature NaN with 0, compute `_symbol_bar_index` via `groupby("symbol").cumcount()`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x] 2.2 Write property test for `Data_Loader` — per-symbol chronological sort
    - **Property 1: Per-Symbol Chronological Sort**
    - **Validates: Requirements 2.2**

  - [x] 2.3 Write property test for `Data_Loader` — last-288-row drop
    - **Property 2: Last-288-Row Drop**
    - **Validates: Requirements 2.3**

  - [x] 2.4 Write property test for `Data_Loader` — no NaN labels after loading
    - **Property 3: No NaN Labels After Loading**
    - **Validates: Requirements 2.4**

  - [x] 2.5 Write property test for `Data_Loader` — no NaN features after loading
    - **Property 4: No NaN Features After Loading**
    - **Validates: Requirements 2.5**

  - [x] 2.6 Implement `gpu_fuzzy_trader/data/splitter.py` — `Data_Splitter`
    - `split_and_persist(df)`: per-symbol chronological 75/25 split using `floor(N * 0.75)`, concatenate, persist to Parquet at `TRAIN_75_PATH` and `VALIDATION_25_PATH`
    - _Requirements: 2.6, 2.7, 2.8_

  - [x] 2.7 Write property test for `Data_Splitter` — per-symbol split ratio and no overlap
    - **Property 5: Per-Symbol Chronological Split Ratio**
    - **Validates: Requirements 2.6, 2.7**

- [x] 3. Implement feature detection and encoding
  - [x] 3.1 Implement `gpu_fuzzy_trader/features/detector.py` — `Feature_Detector`
    - `detect_feature_mode(series)`: exact six-branch logic from design (binary → ternary → sparse_signed → signed → sparse_positive → positive), using `zero_ratio` on full series including zeros
    - `detect_all_modes(df, feature_cols)`: apply per column, return `dict[str, str]`
    - _Requirements: 3.1, 3.2, 3.3_

  - [x] 3.2 Write property test for `Feature_Detector` — mode classification completeness
    - **Property 6: Feature Mode Classification Completeness**
    - **Validates: Requirements 3.1**

  - [x] 3.3 Write property test for `Feature_Detector` — mode classification correctness
    - **Property 7: Feature Mode Classification Correctness**
    - **Validates: Requirements 3.2**

  - [x] 3.4 Implement `gpu_fuzzy_trader/features/encoder.py` — `Encoder`
    - `get_dont_care(mode)`: return sentinel per mode (2/3/5/10)
    - `encode_condition(feature_name, gene, mode)`: map gene to fuzzy value name, format `[feature_name] IS Fuzzy Value Name`; raise `ConfigurationError` if gene equals dont_care
    - `decode_chromosome(chromosome, feature_infos)`: skip dont_care genes, return list of condition strings
    - Implement all exact fuzzy value name mappings from design for all six modes
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

  - [x] 3.5 Write property test for `Encoder` — fuzzy value name encoding round-trip
    - **Property 8: Fuzzy Value Name Encoding Round-Trip**
    - **Validates: Requirements 4.1, 4.2**

  - [x] 3.6 Write property test for `Encoder` — don't-care sentinel correctness
    - **Property 9: Don't-Care Sentinel Correctness**
    - **Validates: Requirements 4.3**

- [x] 4. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement CPU backtest engine
  - [x] 5.1 Implement `gpu_fuzzy_trader/backtest/cpu_engine.py` — `CPUBacktestEngine`
    - `__init__(df, feature_modes, direction, **constants)`: store dataset and config
    - `simulate_rule_set(rule_set, return_logs=False)`: full priority-based assignment, capital-managed simulation, exposure reservation, fee deduction, equity tracking, account ruin detection, per-symbol metrics — exactly mirroring `evaluator_v3.ipynb` `CapitalManagedTradeSimulator`
    - Implement long and short trade outcome logic (TP/SL/time-exit, `max_before_min` tie-breaking) exactly as specified in design
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 5.10, 15.1_

  - [x] 5.2 Write property test for `CPUBacktestEngine` — priority-based rule assignment exclusivity
    - **Property 10: Priority-Based Rule Assignment Exclusivity**
    - **Validates: Requirements 5.1**

  - [x] 5.3 Write property test for `CPUBacktestEngine` — trade outcome correctness
    - **Property 11: Trade Outcome Correctness**
    - **Validates: Requirements 5.2**

  - [x] 5.4 Write property test for `CPUBacktestEngine` — capital-managed position sizing
    - **Property 12: Capital-Managed Position Sizing**
    - **Validates: Requirements 5.4, 5.9**

  - [x] 5.5 Write property test for `CPUBacktestEngine` — exposure reservation invariant
    - **Property 13: Exposure Reservation Invariant**
    - **Validates: Requirements 5.5**

  - [x] 5.6 Write property test for `CPUBacktestEngine` — fee deduction correctness
    - **Property 14: Fee Deduction Correctness**
    - **Validates: Requirements 5.6**

  - [x] 5.7 Write property test for `CPUBacktestEngine` — equity tracking consistency
    - **Property 15: Equity Tracking Consistency**
    - **Validates: Requirements 5.7**

  - [x] 5.8 Write property test for `CPUBacktestEngine` — per-symbol metrics consistency
    - **Property 28: Per-Symbol Metrics Consistency**
    - **Validates: Requirements 15.1**

- [x] 6. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Implement GPU backtest engine
  - [x] 7.1 Implement `gpu_fuzzy_trader/backtest/gpu_engine.py` — `GPUBacktestEngine`
    - `__init__(df, feature_modes, direction, **constants)`: initialize JAX arrays; detect GPU availability; fall back to CPU transparently; raise `ImportError` if JAX cannot be imported
    - `compute_rule_signals(data_matrix, chromosome, dont_cares)`: JAX-jitted vectorized rule matching
    - `compute_trade_outcomes_batch(max_ret, min_ret, close_ret, max_before_min, tp, sl, direction)`: JAX-jitted vectorized trade outcome computation
    - `simulate_equity_sequential(entries, release_indices, net_pnls, initial_capital)`: `jax.lax.scan`-based sequential equity simulation with exposure management
    - `simulate_rule_batch(chromosomes, tp, sl, capital_pct)`: evaluate a batch of chromosomes simultaneously; return list of metrics dicts
    - `simulate_rule_set(rule_set, return_logs=False)`: same interface as `CPUBacktestEngine`
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

  - [x] 7.2 Write property test for `GPUBacktestEngine` — GPU-CPU numerical parity
    - **Property 16: GPU-CPU Numerical Parity**
    - **Validates: Requirements 6.1**

- [x] 8. Implement Phase 1 — Feature Selector
  - [x] 8.1 Implement `gpu_fuzzy_trader/features/selector.py` — `Feature_Selector`
    - Exclude `LABEL_COLUMNS` and `META_COLUMNS`; detect modes from training split only
    - Remove near-zero dispersion features (>95% identical values)
    - Build direction-specific binary success targets (long: max_288 ≥ entry*(1+TP/100) before min_288 ≤ entry*(1-SL/100); short: inverse)
    - Score features per symbol using mutual information; compute cross-symbol stability score; final score = relevance × stability
    - Within-mode redundancy removal (pairwise correlation > 0.95)
    - Select top `PHASE1_TOP_K_FEATURES` per direction
    - Persist to `outputs/selected_features_long.json` and `outputs/selected_features_short.json`
    - Skip logic: if output files exist and are valid JSON with required schema, skip; if missing/corrupted, fail immediately
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9_

  - [x] 8.2 Write property test for `Feature_Selector` — label and meta column exclusion
    - **Property 17: Label and Meta Column Exclusion from Feature Selection**
    - **Validates: Requirements 7.2**

  - [x] 8.3 Write property test for `Feature_Selector` — low-dispersion feature exclusion
    - **Property 18: Low-Dispersion Feature Exclusion**
    - **Validates: Requirements 7.5**

- [x] 9. Implement Phase 2 — Rule Pool Generator
  - [x] 9.1 Implement `gpu_fuzzy_trader/phases/phase2_rule_pool.py` — `Rule_Pool_Generator`
    - `evolution/evox_runner.py`: `run_phase2_evolution()` NSGA-III loop with EvoX reference vectors and niche selection; NumPy NSGA-II fallback when EvoX is missing
    - Fitness function: three objectives (−total_return_pct, max_drawdown_pct, −win_rate) evaluated via `GPUBacktestEngine.simulate_rule_batch()` using static `PHASE2_TP`, `PHASE2_SL`, `PHASE2_CAPITAL_PCT`
    - Penalties: support penalty (trades < `MIN_TRADE_SUPPORT`), diversity penalty (Hamming distance in chromosome space + crowding distance in objective space), condition count penalty (active conditions outside [MIN_CONDITIONS, MAX_CONDITIONS])
    - Sampling: distribute `PHASE1_SAMPLING_TOTAL` rows equally across symbols
    - Run separate pools for long and short directions
    - Persist Pareto-front pool to `outputs/phase2_long_pool.json` / `outputs/phase2_short_pool.json` and history to `outputs/phase2_long_history.json` / `outputs/phase2_short_history.json`
    - Skip logic: if pool files exist and are valid, skip Phase 2
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 8.10, 8.11, 8.12_

  - [x] 9.2 Write property test for `Rule_Pool_Generator` — Phase 2 static risk parameters
    - **Property 19: Phase 2 Static Risk Parameters**
    - **Validates: Requirements 8.4**

  - [x] 9.3 Write property test for `Rule_Pool_Generator` — rule condition count bounds
    - **Property 20: Rule Condition Count Bounds**
    - **Validates: Requirements 8.6**

- [x] 10. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 11. Implement Phase 3 — Rule Set Selector
  - [x] 11.1 Implement `gpu_fuzzy_trader/phases/phase3_rule_set.py` — `Rule_Set_Selector`
    - NSGA-II combinatorial search over ordered combinations of 2–5 rules from Phase 2 pool; no duplicate rules (order-independent condition set equality)
    - Evaluate candidate rule sets using `CPUBacktestEngine` on validation split
    - Fitness: three objectives (−validation_return, validation_drawdown, −validation_win_rate) plus penalties: coverage penalty (symbols_with_trades < `PHASE3_MIN_SYMBOL_COVERAGE`), zero-trade penalty, overfitting penalty (|train_return − val_return| / max(|train_return|, 1.0)), duplicate rule penalty
    - Select best Pareto-front rule set; write to `outputs/long.json` and `outputs/short.json` with Phase 2 static TP/SL/capital_pct
    - Skip logic: if both files exist and pass schema validation, skip; if only one exists, skip and proceed with available file
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 15.4_

  - [x] 11.2 Write property test for `Rule_Set_Selector` — rule set size bounds
    - **Property 21: Rule Set Size Bounds**
    - **Validates: Requirements 9.1, 12.8**

  - [x] 11.3 Write property test for `Rule_Set_Selector` — rule set uniqueness
    - **Property 22: Rule Set Uniqueness**
    - **Validates: Requirements 9.4**

  - [x] 11.4 Write property test for `Rule_Set_Selector` — symbol coverage penalty application
    - **Property 29: Symbol Coverage Penalty Application**
    - **Validates: Requirements 9.5, 15.4**

- [x] 12. Implement output writer
  - [x] 12.1 Implement `gpu_fuzzy_trader/output/writer.py` — `Output_Writer`
    - `write(rule_set, path)`: serialize `RuleSet` to JSON with exact schema (`"direction"`, `"rules_set"` with `"tp"`, `"sl"`, `"capital_pct"`, `"conditions"`)
    - Schema enforcement: truncate `rules_set` to 5 if > 5 rules (log WARNING); reject rules with all-zero tp/sl/capital_pct (log ERROR); validate condition strings match `[feature_name] IS Fuzzy Value Name` pattern; raise `ValidationError` on schema violations
    - `load_and_validate(path)`: load JSON and run full schema validation; raise `ValidationError` if invalid
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9_

  - [x] 12.2 Write property test for `Output_Writer` — JSON output schema validity
    - **Property 23: JSON Output Schema Validity**
    - **Validates: Requirements 12.1–12.9**

- [x] 13. Implement Phase 4 — RL Risk Optimizer
  - [x] 13.1 Implement `gpu_fuzzy_trader/phases/phase4_rl_optimizer.py` — `RL_Agent`
    - Implement `TradingEnv` (gym-compatible): state vector = [K market features, R rule activation strengths (fraction of conditions satisfied), equity_normalized, open_exposure_normalized]; action vector = [tp_i, sl_i, capital_pct_i] per rule, clipped to config bounds; reward = net_pnl_normalized − drawdown_penalty
    - Implement `find_elbow_point(validation_returns)`: normalize curve, compute perpendicular distances from line connecting first and last point, return index of maximum distance; handle edge cases (monotonically increasing → last point; immediately plateauing → first point)
    - Implement DDPG or PPO agent (stable-baselines3 or custom PyTorch): train on training split for `PHASE4_TOTAL_TIMESTEPS` steps; evaluate on validation split every `PHASE4_ELBOW_WINDOW` episodes; apply Elbow Method during training to identify optimal checkpoint; save checkpoint at elbow point
    - Train separate agents for long and short strategies
    - Load elbow checkpoint; extract optimized TP/SL/capital_pct per rule; update `outputs/long.json` and `outputs/short.json` via `Output_Writer`
    - Skip logic: if output files exist and TP/SL/capital_pct values are within valid ranges, skip Phase 4
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8_

  - [x] 13.2 Write property test for `RL_Agent` — RL action bounds
    - **Property 24: RL Action Bounds**
    - **Validates: Requirements 10.3**

  - [x] 13.3 Write property test for `RL_Agent` — RL state vector completeness
    - **Property 25: RL State Vector Completeness**
    - **Validates: Requirements 10.2**

  - [x] 13.4 Write property test for `find_elbow_point` — elbow method correctness
    - **Property 26: Elbow Method Correctness**
    - **Validates: Requirements 10.5**

- [x] 14. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 15. Implement Phase 5 — Out-of-Sample Evaluator
  - [x] 15.1 Implement `gpu_fuzzy_trader/phases/phase5_oos.py` — `OOS_Evaluator`
    - Load `outputs/long.json` and `outputs/short.json` via `Output_Writer.load_and_validate()`
    - Prepare `data/test.csv` with identical steps as training: sort by (symbol, datetime), drop last 288 rows per symbol, drop NaN label rows, fill feature NaN with 0, compute `_symbol_bar_index`
    - Run `CPUBacktestEngine.simulate_rule_set()` with `return_logs=True` for both strategies
    - Compute per-symbol breakdowns from trade logs
    - Handle zero-trade case: report 0% total return, do not report account ruin unless equity actually reached zero
    - Save `outputs/reports/test_long_report.json`, `outputs/reports/test_short_report.json`, `outputs/reports/test_per_symbol_performance.csv`
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6_

  - [x] 15.2 Write property test for `OOS_Evaluator` — test data preparation consistency
    - **Property 27: Test Data Preparation Consistency**
    - **Validates: Requirements 11.2**

- [x] 16. Implement reporting and visualization
  - [x] 16.1 Implement `gpu_fuzzy_trader/reporting/reporter.py` — `Reporter`
    - `plot_phase2_metrics(history, direction)`: objectives vs. generation plot → `outputs/reports/phase2_{direction}_metrics.png`
    - `plot_equity_curve(trade_log, split, direction)`: equity curve plot → `outputs/reports/{split}_{direction}_equity.png` (train, validation, test)
    - `write_per_symbol_csv(metrics, split)`: per-symbol performance CSV → `outputs/reports/{split}_per_symbol_performance.csv`
    - `plot_rl_curve(validation_returns, elbow_idx, direction)`: RL training curve with elbow point marked → `outputs/reports/phase4_{direction}_rl_curve.png`
    - _Requirements: 14.1, 14.2, 14.3, 14.4_

  - [x] 16.2 Wire `Reporter` calls into Phase 2, Phase 3, Phase 4, and Phase 5 modules
    - Call `plot_phase2_metrics` at end of Phase 2 for both directions
    - Call `plot_equity_curve` and `write_per_symbol_csv` at end of Phase 3 (train + validation) and Phase 5 (test)
    - Call `plot_rl_curve` at end of Phase 4 for both directions
    - _Requirements: 14.1, 14.2, 14.3, 14.4_

- [x] 17. Implement pipeline orchestrator and wire everything together
  - [x] 17.1 Implement `gpu_fuzzy_trader/run_pipeline.py` — `Pipeline_Orchestrator`
    - Entry point: `python -m gpu_fuzzy_trader.run_pipeline`
    - Execution order: create output dirs → load and prepare data (Data_Loader + Data_Splitter) → Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5
    - Skip logic per phase: validate output files before skipping; re-run phase if validation fails
    - Log start time, end time, and elapsed duration for each phase; save structured log to `outputs/pipeline.log`
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5_

  - [x] 17.2 Implement `gpu_fuzzy_trader/__main__.py` to support `python -m gpu_fuzzy_trader.run_pipeline` invocation
    - _Requirements: 13.4_

- [x] 18. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at logical phase boundaries
- Property tests validate universal correctness properties using Hypothesis (Python); run with `pytest tests/property/ --hypothesis-seed=42`
- Unit tests validate specific examples and edge cases; run with `pytest tests/unit/`
- Integration tests (GPU-CPU parity benchmark, pipeline skip logic, end-to-end smoke test) should be added under `tests/integration/` after all phases are implemented
- The GPU engine falls back to CPU transparently when no GPU is available; all tests must pass on CPU-only environments
- Phase 2 uses `GPUBacktestEngine`; Phases 3 and 5 use `CPUBacktestEngine` exclusively
- `config.py` is the single source of truth — no module may define overriding defaults

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["2.1", "3.1"] },
    { "id": 1, "tasks": ["2.2", "2.3", "2.4", "2.5", "3.2", "3.3"] },
    { "id": 2, "tasks": ["2.6", "3.4"] },
    { "id": 3, "tasks": ["2.7", "3.5", "3.6"] },
    { "id": 4, "tasks": ["5.1"] },
    { "id": 5, "tasks": ["5.2", "5.3", "5.4", "5.5", "5.6", "5.7", "5.8"] },
    { "id": 6, "tasks": ["7.1"] },
    { "id": 7, "tasks": ["7.2"] },
    { "id": 8, "tasks": ["8.1"] },
    { "id": 9, "tasks": ["8.2", "8.3"] },
    { "id": 10, "tasks": ["9.1"] },
    { "id": 11, "tasks": ["9.2", "9.3"] },
    { "id": 12, "tasks": ["11.1", "12.1"] },
    { "id": 13, "tasks": ["11.2", "11.3", "11.4", "12.2"] },
    { "id": 14, "tasks": ["13.1"] },
    { "id": 15, "tasks": ["13.2", "13.3", "13.4"] },
    { "id": 16, "tasks": ["15.1"] },
    { "id": 17, "tasks": ["15.2"] },
    { "id": 18, "tasks": ["16.1"] },
    { "id": 19, "tasks": ["16.2"] },
    { "id": 20, "tasks": ["17.1", "17.2"] }
  ]
}
```
