# IMPLEMENTED

This repository implements a rule-mining trading pipeline rather than a conventional predictive model. The core idea is:

1. Start with engineered, already-discretized feature columns.
2. Use future labels only for scoring and backtesting, never as model inputs.
3. Select stable features per output type.
4. Evolve single-rule pools with Phase 1.
5. Assemble rule teams with Phase 2.
6. Replay the exported strategy out of sample with Phase 3.

From a data mining perspective, the project is a search-and-selection system for robust fuzzy rules that should generalize across symbols, not a standard forecasting stack.

## Data Model

The dataset is organized around three column groups:

- Meta columns: `datetime`, `symbol`
- Label columns: `label_open_next`, `label_close_288`, `label_min_288`, `label_max_288`, `label_max_before_min`
- Feature columns: all engineered indicators and derived signals

The labels are look-ahead values used to simulate entry, take-profit, stop-loss, and time-based exits. They are excluded from feature selection and rule encoding. The pipeline also removes the last look-ahead window of each symbol when labels are incomplete.

The feature space is not treated as continuous. Columns are grouped into fuzzy output modes such as binary, ternary, sparse signed, sparse positive, and signed. That matters because the encoding, rule matching, and feature selection all depend on the output type.

## End-to-End Pipeline

### Stage 0: Data Preparation

The raw training file is split chronologically by symbol into training and validation sets. The split is symbol-aware, so each symbol is divided independently instead of cutting the whole dataset at one global index. That avoids leaking symbol-specific time structure.

Implemented behavior:

- Load `data/train.parquet` by default, with CSV fallback if needed.
- Sort by time.
- Split each symbol into roughly 75 percent train and 25 percent validation.
- Drop rows with missing labels.
- Write `data/train_75.parquet` and `data/validation_25.parquet`.

Specialist note: this is a time-series aware split, not a random shuffle. It preserves the chronological order needed for realistic trade evaluation.

### Stage 1: Feature Selection

Feature selection is output-type aware. It does not simply rank all features globally. Instead it scores columns using the train and validation splits, checks symbol coverage, measures stability, and groups features by mode.

Implemented behavior:

- Exclude label and meta columns.
- Detect each feature's fuzzy mode from value patterns.
- Filter out low-dispersion columns.
- Score features per symbol across train and validation.
- Rank by relationship, stability, and coverage.
- Produce a single global JSON artifact for downstream phases.

Outputs:

- `outputs/selected_features_global.json`
- `outputs/feature_selection_report_global.csv`

Specialist note: this stage is doing feature mining, not model fitting. It is looking for columns that are both predictive and stable across symbols and time.

### Stage 2: Phase 1 Rule Pool Generation

Phase 1 evolves individual fuzzy rules using a MOEA/D search. Each candidate rule has TP, SL, capital sizing, direction, and up to five condition slots. The goal is to generate a diverse pool of promising long and short rules rather than a single final strategy.

Implemented behavior:

- Decode the selected features and their fuzzy modes.
- Build a chromosome for each candidate rule.
- Score rules with the backtesting engine.
- Optimize multiple objectives, including PnL, drawdown, and win rate.
- Apply penalties for weak trade support, instability, and structural issues.
- Optionally use inner walk-forward folds for more robust scoring.
- Persist phase 1 history, audits, and optional pools.

Typical outputs:

- `outputs/phase1_long_generation_metrics.png`
- `outputs/phase1_short_generation_metrics.png`
- `outputs/phase1_long_generation_history.json`
- `outputs/phase1_short_generation_history.json`
- `outputs/phase1_long_pool_audit.json`
- `outputs/phase1_short_pool_audit.json`

Specialist note: the key product of Phase 1 is not the final strategy. It is a candidate pool with enough diversity and support to make Phase 2 selection meaningful.

### Stage 3: Phase 2 Ensemble Selection

Phase 2 selects ordered teams of rules from the Phase 1 pools. It uses the same backtest engine, but now the question is which combination of rules works best as a priority stack.

Implemented behavior:

- Consume long and short rule pools from Phase 1.
- Build ordered teams with unique rule signatures.
- Evaluate teams with the same priority reservation logic as production replay.
- Use greedy selection by default, with MOEA/D fallback available.
- Enforce coverage and trade-support floors.
- Prefer teams that are not only profitable on validation, but also aligned with train performance.
- Export the final long and short strategies as submission-style JSON files.

Outputs:

- `outputs/long.json`
- `outputs/short.json`
- `outputs/reports/train_trade_dashboard.png`
- `outputs/reports/train_per_symbol_dashboard.png`
- `outputs/reports/validation_trade_dashboard.png`
- `outputs/reports/validation_per_symbol_dashboard.png`
- `outputs/reports/validation_per_symbol_performance.csv` when CSV reporting is enabled

Specialist note: this is a combinatorial search over rule sets. The important question is not whether one rule is good in isolation, but whether a compact ordered team stays stable across symbols and splits.

### Stage 4: Phase 3 Out-of-Sample Test

Phase 3 loads the exported `long.json` and `short.json` files and replays them on the held-out test set. This is the final evidence of whether the mined strategy generalizes.

Implemented behavior:

- Load only the test rows needed by the exported rules.
- Re-run the strategy on `data/test.parquet` or CSV fallback.
- Compute the same trade metrics used earlier.
- Generate test dashboards and per-symbol summaries.
- Record run-debug artifacts.

Outputs:

- `outputs/reports/test_trade_dashboard.png`
- `outputs/reports/test_trade_dashboard_long.png`
- `outputs/reports/test_trade_dashboard_short.png`
- `outputs/reports/test_per_symbol_dashboard.png`
- `outputs/reports/test_per_symbol_performance.csv` when CSV reporting is enabled

Specialist note: this stage is the only one that should be treated as out-of-sample truth. Validation is for search; test is for final assessment.

## File-by-File Implementation Map

### `config.py`

Single source of truth for the project. It stores paths, schema constants, stage settings, and hyperparameters. The file is intentionally central because the pipeline avoids runtime flags and instead reads behavior from configuration.

What it controls:

- Dataset paths for raw, split, and test files
- Label and meta column definitions
- TP/SL bounds and capital sizing limits
- Backtest and risk metric constants
- Phase 1 and Phase 2 MOEA/D settings
- Reporting, run bundle, ablation, and Optuna settings

Specialist note: this is the control plane for the entire system. If a stage behaves unexpectedly, this is the first place to inspect.

### `bigdata_trader/__init__.py`

Package manifest and module guide. It documents the intended top-to-bottom run order and exposes the main submodules.

Specialist note: useful as a lightweight map of the package, but not part of the computational pipeline.

### `bigdata_trader/dataset_io.py`

Unified dataset loader and writer for Parquet and CSV.

Implemented behavior:

- Read and write tabular data with extension-aware logic.
- Prefer Parquet when available.
- Fall back to CSV when needed.
- Optionally materialize Parquet from CSV so future runs are faster.
- Parse `datetime` columns consistently.

Specialist note: this module reduces friction during dataset migration and keeps the rest of the codebase format-agnostic.

### `bigdata_trader/convert_datasets.py`

One-time conversion script for migrating the standard datasets to Parquet.

Implemented behavior:

- Convert `train.csv` to the raw training Parquet file.
- Convert `train_75.csv` and `validation_25.csv` if they exist.
- Convert `test.csv` to the test Parquet file.

Specialist note: this is an ingestion helper, not a modeling stage.

### `bigdata_trader/data_prep.py`

Chronological train/validation splitter.

Implemented behavior:

- Split by symbol and time.
- Preserve per-symbol chronology.
- Remove rows with incomplete label windows.
- Save the split datasets for downstream use.

Specialist note: this stage is critical for leakage control. The split is designed to respect each symbol's temporal structure.

### `bigdata_trader/select_features.py`

Feature scoring and ranking engine.

Implemented behavior:

- Detect feature mode from value distribution.
- Build direction-specific success targets from the label columns.
- Score each feature with relationship, stability, and coverage metrics.
- Optionally use tree-based importance when enabled.
- Remove low-dispersion and weak columns.
- Write the selection JSON consumed by the encoding layer.

Specialist note: this module is doing data mining in the strict sense. It looks for features with predictive structure and temporal consistency, not just high correlation on one split.

### `bigdata_trader/encoding.py`

Chromosome and fuzzy-state encoding/decoding.

Implemented behavior:

- Map feature outputs to canonical fuzzy state labels.
- Convert numeric chromosomes to rule dictionaries.
- Load selected features and their detected modes.
- Provide a numeric decode path for the compiled runtime.
- Compute per-feature quantile thresholds when needed.

Specialist note: this is the bridge between mining results and backtestable rules. It is the structural representation layer for the search.

### `bigdata_trader/backtest_engine.py`

Trade simulation and metric engine.

Implemented behavior:

- Translate condition strings into boolean masks.
- Combine rule conditions with logical AND.
- Simulate priority-based rule assignment.
- Support both standard trade logs and capital-managed replay.
- Handle TP, SL, time exit, transaction cost, drawdown, liquidation, and exposure reservation.
- Produce metrics such as win rate, PnL, drawdown, profit factor, recovery factor, symbol stability, and temporal robustness.

Specialist note: this is the core evaluator. Every stage ultimately depends on its semantics, so it must stay aligned with the intended trading rules.

### `bigdata_trader/compiled_backtest.py`

Packed-mask fast path for the backtester.

Implemented behavior:

- Represent data as array context plus packed masks.
- Run a compiled version of the strategy evaluation.
- Mirror the capital-managed semantics of the legacy engine.

Specialist note: this is an acceleration layer. Its value is only real if it matches the legacy engine closely.

### `bigdata_trader/benchmark_compiled.py`

Parity and speed benchmark for the compiled runtime.

Implemented behavior:

- Sample the training data.
- Generate random rules.
- Compare legacy and compiled metrics.
- Report a pass/fail parity table and a speedup ratio.

Specialist note: use this before trusting the compiled path in production search runs.

### `bigdata_trader/phase1.py`

MOEA/D rule-pool generator.

Implemented behavior:

- Evolve single-rule chromosomes.
- Score candidates through the backtester.
- Apply penalties for weak support, instability, and infeasible structures.
- Use optional walk-forward folds.
- Maintain diversity and archive behavior across generations.
- Persist generation histories and audits.

Specialist note: Phase 1 is a controlled exploration stage. It should maximize useful rule diversity, not prematurely collapse to one lucky pattern.

### `bigdata_trader/phase2.py`

MOEA/D ensemble selector.

Implemented behavior:

- Consume Phase 1 rule pools.
- Build ordered rule teams.
- Evaluate portfolio-like rule stacks with priority reservation.
- Apply trade-support, coverage, overlap, train/validation agreement, and robustness penalties.
- Export final long and short JSON strategies.
- Generate training and validation dashboards plus history files.

Specialist note: Phase 2 is where the project turns candidate rules into a deployable strategy. It is the main combinatorial search layer.

### `bigdata_trader/phase3.py`

Out-of-sample replay stage.

Implemented behavior:

- Load the exported long and short strategies.
- Collect only required feature columns for the test set.
- Backtest the final strategies without clipping submission capital settings.
- Generate test dashboards and per-symbol reports.

Specialist note: this is the final generalization check.

### `bigdata_trader/reporting.py`

Visualization and report generation.

Implemented behavior:

- Build dataset-level trade dashboards.
- Build per-symbol summary charts.
- Build per-rule reports when needed.
- Save PNGs and optional CSV summaries under `outputs/reports`.

Specialist note: this module is diagnostic, not computational. It helps inspect whether the mined strategy is concentrated, diverse, or brittle.

### `bigdata_trader/live_logging.py`

Structured logging and phase timing.

Implemented behavior:

- Emit compact timestamped log lines.
- Track phase start/end and elapsed time.
- Support a run-level file handler for saved logs.

Specialist note: this is the audit trail for long optimization runs.

### `bigdata_trader/run_pipeline.py`

Top-level pipeline orchestrator.

Implemented behavior:

- Run data prep, feature selection, Phase 1, Phase 2, and Phase 3 in order.
- Skip stages when their outputs already exist.
- Support per-phase execution.
- Support alternate output directories for experiment comparisons.
- Integrate run reporting.

Specialist note: this is the canonical command entry point for the project.

### `bigdata_trader/run_reporting.py`

Per-run artifact bundler.

Implemented behavior:

- Create `outputs/runs/<run_id>/` folders.
- Capture the pipeline log.
- Save phase timings and a timing chart.
- Mirror selected artifacts from the active output directory.
- Write a manifest with config subset, metrics, paths, and timing data.

Specialist note: this is the experiment archive layer. It makes runs reproducible and easier to compare.

### `bigdata_trader/optuna_tuner.py`

Hyperparameter search driver.

Implemented behavior:

- Build a tuning context from train and validation data.
- Search across Phase 1 and Phase 2 settings.
- Write trial logs and study outputs.
- Optionally tee the process output into a file.
- Optionally write the best values back to config.

Specialist note: this is a meta-optimizer on top of the mining pipeline, useful when the default knobs are not good enough.

### `bigdata_trader/ablation_runner.py`

Experiment ladder runner.

Implemented behavior:

- Execute the configured ablation presets.
- Run the full pipeline for each preset.
- Evaluate train, validation, and test performance.
- Compare risk-adjusted scores and Phase 1 diversity.
- Write both per-experiment and aggregate comparison JSON files.

Specialist note: this is how you measure whether an idea actually improves the system, rather than only changing one metric.

## Important Artifacts

The main outputs produced by the pipeline are:

- `outputs/selected_features_global.json`
- `outputs/feature_selection_report_global.csv`
- `outputs/phase1_long_generation_history.json`
- `outputs/phase1_short_generation_history.json`
- `outputs/phase1_long_pool_audit.json`
- `outputs/phase1_short_pool_audit.json`
- `outputs/long.json`
- `outputs/short.json`
- `outputs/reports/*.png`
- `outputs/reports/*.csv` when the CSV reporting toggles are enabled
- `outputs/runs/<run_id>/` for per-run bundles

There is also an `archive_p1/` directory used for Phase 1 archive persistence and warm starts.

## Practical Reading Order

If you want to understand the system quickly, read in this order:

1. `config.py`
2. `data_prep.py`
3. `select_features.py`
4. `encoding.py`
5. `backtest_engine.py`
6. `phase1.py`
7. `phase2.py`
8. `phase3.py`
9. `reporting.py`
10. `run_pipeline.py`

That sequence follows the actual data flow from raw input to final test replay.

## Operational Summary

The project is implemented as a disciplined search pipeline for fuzzy trading rules. The strongest parts of the design are the symbol-aware split, the mode-aware feature selection, the explicit rule encoding, and the evaluator-aligned capital-managed backtest. The main risk in this kind of system is overfitting to validation. The code therefore separates feature mining, rule generation, ensemble selection, and out-of-sample replay into distinct stages so each layer can be inspected independently.
