# GPU-Fuzzy Trading Pipeline

A ground-up, GPU-accelerated fuzzy rule mining pipeline that discovers, optimizes, and evaluates trading strategies across 10 symbols. The system mines interpretable fuzzy rules from a discretized feature dataset, refines risk parameters via deep reinforcement learning, and produces two JSON strategy files (`long.json` and `short.json`) that are fully compatible with `evaluator_v3.ipynb`.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Dataset](#3-dataset)
4. [Five-Phase Pipeline](#4-five-phase-pipeline)
5. [Module Reference](#5-module-reference)
6. [Configuration](#6-configuration)
7. [Output Files](#7-output-files)
8. [Strategy Format](#8-strategy-format)
9. [Backtest Engine Semantics](#9-backtest-engine-semantics)
10. [Feature Modes and Fuzzy Encoding](#10-feature-modes-and-fuzzy-encoding)
11. [Running the Pipeline](#11-running-the-pipeline)
12. [Testing](#12-testing)
13. [Design Principles](#13-design-principles)

---

## 1. Project Overview

This project is a **rule-mining trading system**, not a conventional predictive model. The core idea is:

- Start with pre-engineered, already-discretized feature columns.
- Use future labels **only** for scoring and backtesting — never as model inputs.
- Select stable, direction-specific features per output type.
- Evolve a large pool of candidate fuzzy rules using GPU-accelerated **NSGA-III** multi-objective search (EvoX).
- Assemble compact rule teams (default: 2–3 rules; strategy schema supports up to 5) via greedy construction and Pareto refinement.
- Fine-tune risk parameters (TP, SL, capital allocation) using a deep RL agent.
- Evaluate the final strategy on a held-out test set.

The strongest design choices are the **symbol-aware chronological split**, **mode-aware feature selection**, **explicit fuzzy rule encoding**, and a **backtest engine that exactly mirrors `evaluator_v3.ipynb`** so optimization scores match final evaluation scores.

---

## 2. Architecture

```
gpu_fuzzy_trader/
├── config.py                    # Single source of truth for all hyperparameters
├── run_pipeline.py              # Top-level orchestrator
├── __main__.py                  # python -m gpu_fuzzy_trader.run_pipeline entry point
│
├── data/
│   ├── loader.py                # Data_Loader: CSV loading, datetime parsing, NaN handling
│   └── splitter.py              # Data_Splitter: per-symbol chronological 75/25 split
│
├── features/
│   ├── detector.py              # Feature_Detector: mode classification (binary/ternary/etc.)
│   ├── selector.py              # Feature_Selector: direction-specific scoring and ranking
│   └── encoder.py               # Encoder: gene → fuzzy value name, condition string formatting
│
├── backtest/
│   ├── cpu_engine.py            # CPUBacktestEngine: exact evaluator_v3.ipynb semantics
│   └── gpu_engine.py            # GPUBacktestEngine: JAX-accelerated, numerically equivalent
│
├── evolution/
│   └── evox_runner.py           # Phase 2 NSGA-III loop (EvoX ranking + reference vectors)
│
├── phases/
│   ├── phase2_rule_pool.py      # Rule_Pool_Generator: orchestrates Phase 2 evolution
│   ├── phase3_rule_set.py       # Rule_Set_Selector: greedy + refinement on validation
│   ├── phase4_rl_optimizer.py   # RL_Agent: DDPG/PPO with Elbow Method stopping
│   └── phase5_oos.py            # OOS_Evaluator: final test.csv evaluation
│
├── output/
│   └── writer.py                # Output_Writer: JSON serialization and schema validation
│
└── reporting/
    └── reporter.py              # Reporter: equity curves, per-symbol CSVs, phase metrics
```

### Data Flow

```
data/train.csv ──► Data_Loader ──► Data_Splitter ──► train_75.parquet
                                                  └──► validation_25.parquet
                                                           │
                                              Feature_Detector (modes)
                                                           │
                                              Feature_Selector (Phase 1)
                                                           │
                                         ┌─────────────────┴──────────────────┐
                                    long features                        short features
                                         │                                     │
                                  Rule_Pool_Generator (Phase 2, GPU)           │
                                         │                                     │
                                      pools/phase2_long_pool.json      pools/phase2_short_pool.json
                                         │                                     │
                                  Rule_Set_Selector (Phase 3, CPU validation)  │
                                         │                                     │
                                    long.json                           short.json
                                         │                                     │
                                  RL_Agent (Phase 4, Elbow Method)             │
                                         │                                     │
                                    long.json (updated TP/SL/cap)    short.json (updated)
                                         │                                     │
                                  OOS_Evaluator (Phase 5, data/test.csv)
                                         │
                                  outputs/reports/test_*.json / *.png / *.csv
```

---

## 3. Dataset

### Files

| File                         | Purpose                                                                                               |
| ---------------------------- | ----------------------------------------------------------------------------------------------------- |
| `data/train.csv`             | Training data — used for feature selection, rule pool generation, rule set selection, and RL training |
| `data/test.csv`              | Held-out test data — used **only** in Phase 5 (out-of-sample evaluation)                              |
| `data/train_75.parquet`      | Auto-generated: 75% chronological training split per symbol                                           |
| `data/validation_25.parquet` | Auto-generated: 25% chronological validation split per symbol                                         |

### Column Groups

**Meta columns** (excluded from all modeling):

- `datetime` — timestamp of the candle
- `symbol` — one of 10 trading instruments

**Label columns** (look-ahead values, used only for trade simulation):

- `label_open_next` — entry price (next candle open)
- `label_close_288` — close price 288 candles ahead (24 hours at 5-min bars)
- `label_min_288` — minimum price over the next 288 candles
- `label_max_288` — maximum price over the next 288 candles
- `label_max_before_min` — 1 if the max was reached before the min (used for TP/SL tie-breaking)

**Feature columns** — hundreds of pre-engineered, discretized indicators (momentum, volatility, mean-reversion, trend, etc.). Values are integers representing discrete fuzzy states, not raw floats.

### Important: Last-288-Row Drop

The last 288 rows per symbol have no valid labels (the look-ahead window extends beyond the dataset). These rows are **always dropped** before any processing in both train and test data.

### Prediction Horizon

The `_288` suffix means a 24-hour look-ahead window: 288 bars × 5 minutes = 1,440 minutes. This captures daily price cycles and significant intraday trends.

---

## 4. Five-Phase Pipeline

### Phase 1 — Direction-Specific Feature Selection

**Module:** `gpu_fuzzy_trader/features/selector.py` → `Feature_Selector`

Produces two independent feature lists (long and short) from the training split. The selection is mode-aware and symbol-aware.

**Algorithm:**

1. Exclude all label and meta columns.
2. Detect each feature's fuzzy mode from its value distribution (training split only).
3. Remove near-zero dispersion features (>95% identical values).
4. Build a direction-specific binary success target:
   - **Long:** `label_max_288 ≥ entry × (1 + TP%)` before `label_min_288 ≤ entry × (1 − SL%)`
   - **Short:** `label_min_288 ≤ entry × (1 − TP%)` before `label_max_288 ≥ entry × (1 + SL%)`
5. Score each feature per symbol using mutual information.
6. Compute cross-symbol stability = `1 − (std / mean)` of per-symbol scores.
7. Final score = `relevance × stability`.
8. Within-mode redundancy removal (pairwise correlation > 0.95 → drop lower-scored).
9. Select top `PHASE1_TOP_K_FEATURES` (default: 25) per direction.

**Outputs:**

- `outputs/selected_features_long.json`
- `outputs/selected_features_short.json`

**Skip logic:** The `skip_if_valid()` helper can validate these files for programmatic use, but the default CLI full run forces Phase 1 to rerun.

---

### Phase 2 — GPU-Accelerated Rule Pool Generation

**Modules:**

- `gpu_fuzzy_trader/phases/phase2_rule_pool.py` → `Rule_Pool_Generator` (orchestration, persistence, reporting)
- `gpu_fuzzy_trader/evolution/evox_runner.py` → `run_phase2_evolution` (NSGA-III evolutionary loop)

Evolves a large, diverse pool of candidate fuzzy rules using **NSGA-III** (Non-dominated Sorting Genetic Algorithm III). Each generation:

1. Evaluates the population with `GPUBacktestEngine` (JAX) when available, else `CPUBacktestEngine`.
2. Builds offspring via rank/crowding tournament mating, crossover, and mutation on integer chromosomes.
3. Merges parents + offspring and applies **NSGA-III environmental selection** (EvoX non-dominated ranking + reference-vector niche filling on the last front).

**Requires `evox`** (and `torch`) for the NSGA-III survivor step. If EvoX is not installed, Phase 2 logs a warning and falls back to a built-in **NumPy NSGA-II** loop (history records `"NSGA-II (fallback)"`).

**Chromosome encoding:**

```
chromosome = [gene_0, gene_1, ..., gene_{K-1}]
gene_i ∈ {0, ..., num_classes_i − 1, dont_care_i}
dont_care_i = num_classes_i  (inactive condition)
```

**Three objectives (all minimized):**

- `f1 = −sortino_ratio`
- `f2 = max_drawdown_pct`
- `f3 = −win_rate`

**Penalties applied to all objectives:**

- **Support penalty** — if `executed_trades < MIN_TRADE_SUPPORT` (default: 200)
- **Diversity penalty** — Hamming distance to nearest Pareto-front member
- **Condition count penalty** — if active conditions outside `[MIN_CONDITIONS, MAX_CONDITIONS]` (default: 3–4)

**Static risk parameters during Phase 2** (isolates predictive alpha from risk tuning):

- TP = 4.0%, SL = 2.0%, capital_pct = 50.0%

**Outputs:**

- `outputs/phase2_long_pool.json` / `outputs/phase2_short_pool.json` (per-run, overwritten)
- `outputs/phase2_long_history.json` / `outputs/phase2_short_history.json`
- `phase2_rule_archive/phase2_long_archive.json` / `phase2_rule_archive/phase2_short_archive.json` (persistent)
- `outputs/reports/phase2_long_metrics.png` / `outputs/reports/phase2_short_metrics.png`

**Skip logic:** The `skip_if_valid()` helper can validate the pool files for programmatic use, but the default CLI full run forces Phase 2 to rerun. The root-level archive is persistent across output directories; compatible archived rules still seed part of the next Phase 2 population before the rest is initialized randomly.

---

### Phase 3 — Rule Set Selection

**Module:** `gpu_fuzzy_trader/phases/phase3_rule_set.py` → `Rule_Set_Selector`

Selects the best ordered combination of rules from the Phase 2 pool using greedy construction followed by short Pareto refinement. By default, the search uses 2–3 rules (`PHASE3_MIN_RULES`–`PHASE3_MAX_RULES`), while the output schema remains compatible with 2–5 rules. Evaluated on the **validation split** using `CPUBacktestEngine`.

**Search space:** All ordered combinations of `PHASE3_MIN_RULES`–`PHASE3_MAX_RULES` rules from the pool, with no duplicate rules (order-independent condition set equality).

**Three objectives (all minimized):**

- `f1 = −validation_sortino_ratio`
- `f2 = validation_max_drawdown_pct`
- `f3 = −validation_win_rate`

**Penalties:**

- **Coverage penalty** — if `symbols_with_trades < PHASE3_MIN_SYMBOL_COVERAGE` (default: 7 of 10)
- **Zero-trade penalty** — if no trades executed at all
- **Overfitting penalty** — `|train_return − val_return| / max(|train_return|, 1.0)`
- **Duplicate rule penalty** — if any two rules share identical condition sets

**Outputs:**

- `outputs/long.json` / `outputs/short.json` (Phase 2 static TP/SL/capital_pct at this stage)
- `outputs/reports/train_long_equity.png` / `outputs/reports/validation_long_equity.png`
- `outputs/reports/train_per_symbol_performance.csv` / `outputs/reports/validation_per_symbol_performance.csv`

**Skip logic:** The `skip_if_valid()` helper can validate these files for programmatic use, but the default CLI full run forces Phase 3 to rerun. The `--phase 3` command expects the Phase 2 pool files to already exist.

---

### Phase 4 — RL-Based Risk Optimization

**Module:** `gpu_fuzzy_trader/phases/phase4_rl_optimizer.py` → `RL_Agent`

Fine-tunes TP, SL, and `capital_pct` for each rule using a DDPG or PPO agent (stable-baselines3 when available; random search with Elbow Method stopping as fallback).

**State vector:**

```
[K market features, R rule activation strengths, equity_normalized, open_exposure_normalized]
```

**Action vector (continuous, per rule):**

```
[tp_i, sl_i, capital_pct_i]  for i in 0..R-1
```

**Action bounds (from `config.py`):**

- TP: [1.0%, 10.0%]
- SL: [0.5%, 5.0%]
- capital_pct: [10.0%, 100.0%]

**Reward:** `net_pnl_normalized − drawdown_penalty`

**Elbow Method stopping:** Finds the optimal training checkpoint by computing perpendicular distances from the line connecting the first and last point of the validation returns curve. Returns the index of maximum curvature — preventing overfitting to the training split.

**Outputs:**

- `outputs/long.json` / `outputs/short.json` (updated with RL-optimized TP/SL/capital_pct)
- `outputs/reports/phase4_long_rl_curve.png` / `outputs/reports/phase4_short_rl_curve.png`

**Skip logic:** The `skip_if_valid()` helper can validate these files for programmatic use, but the default CLI full run forces Phase 4 to rerun. The `--phase 4` command expects the Phase 3 strategy files to already exist.

---

### Phase 5 — Out-of-Sample Evaluation

**Module:** `gpu_fuzzy_trader/phases/phase5_oos.py` → `OOS_Evaluator`

Loads the final strategies and evaluates them on the held-out `data/test.csv`. This is the only phase that should be treated as out-of-sample truth.

**Data preparation** (identical to training pipeline):

1. Load `data/test.csv`
2. Sort by (symbol, datetime)
3. Drop last 288 rows per symbol
4. Drop NaN label rows
5. Fill feature NaN with 0
6. Compute `_symbol_bar_index`

**Metrics reported:**

- Total return %, max drawdown %, win rate %, profit factor
- Executed trades, account status (survived / ruined)
- Per-symbol: trade count, win rate, net PnL

**Zero-trade handling:** Reports 0% total return; does NOT report account ruin unless equity actually reached zero.

**Outputs:**

- `outputs/reports/test_long_report.json` / `outputs/reports/test_short_report.json`
- `outputs/reports/test_per_symbol_performance.csv`
- `outputs/reports/test_long_equity.png` / `outputs/reports/test_short_equity.png`

---

## 5. Module Reference

### `gpu_fuzzy_trader/config.py`

Single source of truth. No module may define its own defaults that override these values. All paths, constants, and behavioral settings live here. See [Configuration](#6-configuration) for the full parameter table.

### `gpu_fuzzy_trader/data/loader.py` — `Data_Loader`

Stateless CSV loader with a 7-step preparation pipeline:

1. Read CSV (comma-separated)
2. Parse `datetime` column
3. Sort by `(symbol, datetime)`
4. Drop last `TAIL_DROP_ROWS` (288) rows per symbol
5. Drop rows where any label column is NaN
6. Fill NaN in feature columns with 0
7. Compute `_symbol_bar_index` via `groupby("symbol").cumcount()`

### `gpu_fuzzy_trader/data/splitter.py` — `Data_Splitter`

Per-symbol chronological 75/25 split. Uses `floor(N × 0.75)` for the split point. Persists to `data/train_75.parquet` and `data/validation_25.parquet`.

### `gpu_fuzzy_trader/features/detector.py` — `Feature_Detector`

Classifies each feature column into one of six modes using the exact logic from `evaluator_v3.ipynb`. `zero_ratio` is computed on the full series including zeros.

### `gpu_fuzzy_trader/features/encoder.py` — `Encoder`

Maps integer gene values to fuzzy value names and formats condition strings. Defines don't-care sentinels per mode. Raises `ConfigurationError` if a gene equals the don't-care sentinel.

### `gpu_fuzzy_trader/features/selector.py` — `Feature_Selector`

Direction-specific feature scoring. Produces `selected_features_long.json` and `selected_features_short.json`. Includes `skip_if_valid()` and `load_and_validate()` static methods.

### `gpu_fuzzy_trader/backtest/cpu_engine.py` — `CPUBacktestEngine`

The canonical reference implementation. Exactly mirrors `evaluator_v3.ipynb`'s `CapitalManagedTradeSimulator`. All other engines must produce numerically equivalent results. Supports `return_logs=True` for detailed trade log DataFrames.

### `gpu_fuzzy_trader/backtest/gpu_engine.py` — `GPUBacktestEngine`

JAX-accelerated backtest engine used during Phase 2 (and optionally Phase 3 when `PHASE3_USE_GPU=True`). Produces results within 1e-4 relative tolerance of `CPUBacktestEngine`. Falls back to CPU transparently when no GPU is available. Raises `ImportError` if JAX cannot be imported.

### `gpu_fuzzy_trader/evolution/evox_runner.py` — `run_phase2_evolution`

Phase 2 multi-objective evolutionary search. Implements NSGA-III with EvoX reference vectors and niche-based truncation on the critical front. Shared helpers in the same module cover offspring generation (tournament mating on Pareto rank/crowding), integer chromosome repair, and NSGA-II environmental selection used only when EvoX is missing.

### `gpu_fuzzy_trader/phases/phase2_rule_pool.py` — `Rule_Pool_Generator`

Loads Phase 1 features, runs `run_phase2_evolution` separately for long and short directions, and writes pool/history JSON plus generation metric plots.

### `gpu_fuzzy_trader/output/writer.py` — `Output_Writer`

Serializes rule sets to JSON with full schema enforcement. Truncates to 5 rules if > 5 (logs WARNING). Rejects rules with all-zero TP/SL/capital_pct (logs ERROR, raises `ValidationError`). Validates condition string format.

### `gpu_fuzzy_trader/reporting/reporter.py` — `Reporter`

Generates all visual and tabular reports. Uses matplotlib with the `Agg` backend (non-interactive). All methods accept an `output_dir` override for testability.

### `gpu_fuzzy_trader/run_pipeline.py` — `Pipeline_Orchestrator`

Top-level orchestrator. Runs all five phases in order, or a single requested phase, with forced full rebuilds on the default CLI path and structured JSON-lines logging to `outputs/pipeline.log`.

---

## 6. Configuration

All hyperparameters live in [`gpu_fuzzy_trader/config.py`](gpu_fuzzy_trader/config.py). Edit that file to tune the pipeline — no runtime flags are used.

**Detailed reference:** For per-phase explanations, default values, and how each knob affects out-of-sample performance, generalization, and compute cost, see **[docs/hyperparameters/](docs/hyperparameters/README.md)** (one guide per pipeline phase, written for data scientists).

Quick index:

| Doc | Covers |
|-----|--------|
| [Phase 0 — Shared](docs/hyperparameters/phase0_shared.md) | Paths, schema, backtest constants |
| [Phase 1](docs/hyperparameters/phase1_feature_selection.md) | `PHASE1_*` feature selection |
| [Phase 2](docs/hyperparameters/phase2_rule_pool.md) | Rule pool evolution, support penalties, archive |
| [Phase 3](docs/hyperparameters/phase3_rule_set.md) | Rule set selection, validation gates |
| [Phase 4](docs/hyperparameters/phase4_rl_risk.md) | RL risk optimization |
| [Phase 5](docs/hyperparameters/phase5_oos.md) | OOS evaluation and metric interpretation |

---

## 7. Output Files

```
outputs/
├── pipeline.log                          # JSON-lines phase timing log
├── selected_features_long.json           # Phase 1: selected features for long direction
├── selected_features_short.json          # Phase 1: selected features for short direction
├── phase2_long_pool.json                 # Phase 2: Pareto-front rule pool (long) — per-run
├── phase2_short_pool.json                # Phase 2: Pareto-front rule pool (short) — per-run
├── phase2_long_history.json              # Phase 2: per-generation metrics (long)
├── phase2_short_history.json             # Phase 2: per-generation metrics (short)
├── long.json                             # Phase 3/4: final long strategy
├── short.json                            # Phase 3/4: final short strategy
└── reports/
    ├── phase2_long_metrics.png           # Phase 2: objectives vs. generation (long)
    ├── phase2_short_metrics.png          # Phase 2: objectives vs. generation (short)
    ├── train_long_equity.png             # Phase 3: equity curve on training split (long)
    ├── train_short_equity.png            # Phase 3: equity curve on training split (short)
    ├── validation_long_equity.png        # Phase 3: equity curve on validation split (long)
    ├── validation_short_equity.png       # Phase 3: equity curve on validation split (short)
    ├── train_per_symbol_performance.csv  # Phase 3: per-symbol metrics on training split
    ├── validation_per_symbol_performance.csv  # Phase 3: per-symbol metrics on validation
    ├── phase4_long_rl_curve.png          # Phase 4: RL training curve with elbow point (long)
    ├── phase4_short_rl_curve.png         # Phase 4: RL training curve with elbow point (short)
    ├── test_long_report.json             # Phase 5: OOS metrics (long)
    ├── test_short_report.json            # Phase 5: OOS metrics (short)
    ├── test_per_symbol_performance.csv   # Phase 5: per-symbol OOS metrics
    ├── test_long_equity.png              # Phase 5: equity curve on test set (long)
    └── test_short_equity.png             # Phase 5: equity curve on test set (short)

phase2_rule_archive/                      # Persistent across runs (project root)
├── phase2_long_archive.json              # Phase 2: persistent best-rule archive (long)
└── phase2_short_archive.json             # Phase 2: persistent best-rule archive (short)
```

---

## 8. Strategy Format

The final output files (`long.json` and `short.json`) are fully compatible with `evaluator_v3.ipynb`:

```json
{
  "direction": "long",
  "rules_set": [
    {
      "tp": 2.5,
      "sl": 1.2,
      "capital_pct": 15.0,
      "conditions": [
        "[dmi_balance_14] IS Bearish",
        "[vol_ratio_20_100] IS Very Low"
      ]
    },
    {
      "tp": 3.1,
      "sl": 1.8,
      "capital_pct": 12.0,
      "conditions": [
        "[amihud_illiquidity_20] IS Very High",
        "[macd_hist_atr] IS Extreme Bearish",
        "[return_skew_30] IS Strong Bearish"
      ]
    }
  ]
}
```

**Schema constraints:**

- `direction`: `"long"` or `"short"` (lowercase)
- `rules_set`: array of 2–5 rule objects
- Each rule: exactly `"tp"`, `"sl"`, `"capital_pct"`, `"conditions"`
- `tp`, `sl`: floats representing percentages (e.g., `2.5` = 2.5%)
- `capital_pct`: float representing % of equity to allocate (e.g., `50.0` = 50%)
- `conditions`: non-empty array of `"[feature_name] IS Fuzzy Value Name"` strings
- At least one of `tp`, `sl`, `capital_pct` must be non-zero per rule

---

## 9. Backtest Engine Semantics

The `CPUBacktestEngine` exactly mirrors `evaluator_v3.ipynb`'s `CapitalManagedTradeSimulator`. This alignment is critical — optimization scores during Phases 2 and 3 must match the final evaluation scores.

### Priority-Based Rule Assignment

For each candle, the first rule in the ordered rule set whose conditions all match is assigned. Subsequent rules are skipped for that candle. This prevents duplicate entries on the same symbol/time.

### Trade Outcome Logic

**Long direction:**

- TP hit: `label_max_288 ≥ entry × (1 + tp/100)`
- SL hit: `label_min_288 ≤ entry × (1 − sl/100)`
- Both hit: `label_max_before_min == 1` → TP first; else SL first
- Neither hit: time exit at `label_close_288`

**Short direction:**

- TP hit: `label_min_288 ≤ entry × (1 − tp/100)`
- SL hit: `label_max_288 ≥ entry × (1 + sl/100)`
- Both hit: `label_max_before_min == 1` → SL first; else TP first
- Neither hit: time exit at `−close_ret`

### Capital Management

```
position_notional = min(
    equity × (capital_pct / 100) × leverage,
    max(0, equity × MAX_TOTAL_EXPOSURE_PCT/100 × leverage − open_total_exposure)
)
```

Trades are skipped if `position_notional < MIN_POSITION_NOTIONAL` (1.0).

### Exposure Reservation

Each open trade reserves exposure until `entry_symbol_bar_index + MAX_HOLD_CANDLES`. PnL is realized at the conservative release point, not at entry. This prevents using future PnL to size new trades.

### Fee Deduction

```
fee = position_notional × FEE_PCT / 100
net_pnl = gross_pnl − fee
```

### Account Ruin

Simulation stops and marks the account as ruined when `equity ≤ 0`.

### Performance Metrics

| Metric         | Formula                                                           |
| -------------- | ----------------------------------------------------------------- |
| Total Return % | `(final_equity / INITIAL_CAPITAL − 1) × 100`                      |
| Win Rate       | `wins / executed_trades × 100`                                    |
| Profit Factor  | `gross_profit_sum / gross_loss_sum` (99.0 if no losses with wins) |
| Max Drawdown % | `max((peak_equity − equity) / peak_equity × 100)`                 |

---

## 10. Feature Modes and Fuzzy Encoding

Feature columns are not treated as continuous values. Each column is classified into one of six discrete modes, and values are mapped to human-readable fuzzy labels.

### Mode Detection

```python
def detect_feature_mode(series):
    unique_vals = series.dropna().unique()
    n_unique = len(unique_vals)

    if n_unique <= 2 and set(unique_vals).issubset({0, 1}):
        return "binary"
    if n_unique <= 3 and set(unique_vals).issubset({-1, 0, 1}):
        return "ternary"

    zero_ratio = (series == 0).mean()  # computed on full series including zeros

    if series.min() < 0:
        return "sparse_signed" if zero_ratio > 0.3 else "signed"
    return "sparse_positive" if zero_ratio > 0.3 else "positive"
```

### Fuzzy Value Name Mappings

| Mode                           | Gene → Fuzzy Value Name                                                                                                                                                                                        |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `binary`                       | 0 → "Inactive (0)", 1 → "Active (1)"                                                                                                                                                                           |
| `ternary`                      | 0 → "Negative (-1)", 1 → "Neutral (0)", 2 → "Positive (1)"                                                                                                                                                     |
| `positive` / `sparse_positive` | 0 → "Very Low", 1 → "Low", 2 → "Medium", 3 → "High", 4 → "Very High"                                                                                                                                           |
| `sparse_signed`                | 0 → "Strong Negative", 1 → "Weak Negative", 2 → "Exactly Zero", 3 → "Weak Positive", 4 → "Strong Positive"                                                                                                     |
| `signed`                       | 0 → "Extreme Bearish", 1 → "Strong Bearish", 2 → "Bearish", 3 → "Weak Bearish", 4 → "Neutral Negative", 5 → "Neutral Positive", 6 → "Weak Bullish", 7 → "Bullish", 8 → "Strong Bullish", 9 → "Extreme Bullish" |

### Don't-Care Sentinels

| Mode                                           | num_classes | dont_care |
| ---------------------------------------------- | ----------- | --------- |
| `binary`                                       | 2           | 2         |
| `ternary`                                      | 3           | 3         |
| `positive`, `sparse_positive`, `sparse_signed` | 5           | 5         |
| `signed`                                       | 10          | 10        |

A gene equal to `dont_care` means that condition is inactive (the feature is not part of the rule). This allows the evolutionary algorithm to discover rules with varying numbers of active conditions.

### Condition String Format

All conditions follow the exact format recognized by `evaluator_v3.ipynb`'s `apply_dynamic_rule`:

```
[feature_name] IS Fuzzy Value Name
```

Examples:

- `[amihud_illiquidity_20] IS Very High`
- `[rsi_centered_14] IS Weak Bullish`
- `[mom_tl_break_bull_30] IS Active (1)`

---

## 11. Running the Pipeline

### Full Pipeline

```bash
python -m gpu_fuzzy_trader.run_pipeline
```

This runs all five phases in order and forces a fresh rebuild into `outputs/` by default. Pass `--output DIR` to write into another directory. The CLI does not skip a phase just because cached outputs already exist. Use `--phase 1` through `--phase 5` to run a single phase after its prerequisite files already exist.

### Programmatic Usage

```python
from gpu_fuzzy_trader.run_pipeline import Pipeline_Orchestrator

orchestrator = Pipeline_Orchestrator()
results = orchestrator.run(force=True)

# results keys: "data", "phase1", "phase2", "phase3", "phase4", "phase5"
print(results["phase5"])  # OOS metrics
```

### Running Individual Phases

CLI phase runs:

```bash
python -m gpu_fuzzy_trader.run_pipeline --phase 1
python -m gpu_fuzzy_trader.run_pipeline --phase 2
python -m gpu_fuzzy_trader.run_pipeline --phase 3
python -m gpu_fuzzy_trader.run_pipeline --phase 4
python -m gpu_fuzzy_trader.run_pipeline --phase 5
```

Each phase command expects its prerequisite outputs to already be present on disk. The CLI will not auto-run earlier phases for you.

```python
from gpu_fuzzy_trader.data.loader import Data_Loader
from gpu_fuzzy_trader.data.splitter import Data_Splitter
from gpu_fuzzy_trader.features.selector import Feature_Selector
from gpu_fuzzy_trader.phases.phase2_rule_pool import Rule_Pool_Generator
from gpu_fuzzy_trader.phases.phase3_rule_set import Rule_Set_Selector
from gpu_fuzzy_trader.phases.phase4_rl_optimizer import RL_Agent
from gpu_fuzzy_trader.phases.phase5_oos import OOS_Evaluator

# Load and split data
loader = Data_Loader()
train_full = loader.load_dataset("data/train.csv")

splitter = Data_Splitter()
train_df, val_df = splitter.split_and_persist(train_full)

# Phase 1: Feature selection
selector = Feature_Selector()
features = selector.run(train_df)
# features = {"long": [...], "short": [...]}

# Phase 2: Rule pool generation (long direction)
generator = Rule_Pool_Generator(
    train_df=train_df,
    feature_infos=features["long"],
    direction="long",
)
pool = generator.run()

# Phase 3: Rule set selection
rule_selector = Rule_Set_Selector(
    train_df=train_df,
    val_df=val_df,
    pool=pool,
    direction="long",
)
rule_set = rule_selector.run()

# Phase 4: RL risk optimization
agent = RL_Agent(
    train_df=train_df,
    val_df=val_df,
    rule_set=rule_set,
    direction="long",
)
optimized = agent.train()

# Phase 5: Out-of-sample evaluation
evaluator = OOS_Evaluator()
oos_results = evaluator.run()
```

### Skip Logic

The `skip_if_valid()` helpers still exist for validation and programmatic use, but the default CLI run now forces a rebuild:

```python
# Check if Phase 1 can be skipped
existing = Feature_Selector.skip_if_valid()
if existing:
    print("Phase 1 skipped — using existing outputs")

# Check if Phase 2 pool exists
pool = Rule_Pool_Generator.skip_if_valid("long")

# Check if Phase 3 rule sets exist
rule_sets = Rule_Set_Selector.skip_if_valid()

# Check if Phase 4 outputs are within valid RL bounds
optimized = RL_Agent.skip_if_valid("long")
```

### Pipeline Log

Each run appends structured JSON lines to `outputs/pipeline.log`:

```json
{"phase": "Phase 1: Feature Selection", "start_time": "...", "end_time": "...", "elapsed_seconds": 12.4, "skipped": false, "result_summary": {"long_features": 25, "short_features": 25}}
{"phase": "Phase 2: Rule Pool Generation [long]", "start_time": "...", "end_time": "...", "elapsed_seconds": 847.2, "skipped": false, "result_summary": {"pool_size": 43}}
```

---

## 12. Testing

The project has comprehensive test coverage: **713 tests passing** (56 skipped — GPU-only tests that correctly skip without JAX/GPU hardware).

### Test Structure

```
tests/
├── unit/                          # Unit tests for each module
│   ├── test_data_loader.py
│   ├── test_data_splitter.py
│   ├── test_feature_detector.py
│   ├── test_encoder.py
│   ├── test_feature_selector.py
│   ├── test_cpu_engine.py
│   ├── test_gpu_engine.py         # Skipped without JAX
│   ├── test_phase2_rule_pool.py
│   ├── test_phase3_rule_set.py
│   ├── test_output_writer.py
│   ├── test_phase4_rl_optimizer.py
│   ├── test_phase5_oos.py
│   ├── test_reporter.py
│   └── test_run_pipeline.py
│
└── property/                      # Hypothesis property-based tests
    ├── test_data_loader_properties.py      # Properties 1–4
    ├── test_data_splitter_properties.py    # Property 5
    ├── test_feature_detector_properties.py # Properties 6–7
    ├── test_encoder_properties.py          # Properties 8–9
    ├── test_cpu_engine_properties.py       # Properties 10–15, 28
    ├── test_gpu_engine_properties.py       # Property 16 (skipped without JAX)
    ├── test_feature_selector_properties.py # Properties 17–18
    ├── test_phase2_rule_pool_properties.py # Properties 19–20
    ├── test_phase3_rule_set_properties.py  # Properties 21–22, 29
    ├── test_output_writer_properties.py    # Property 23
    ├── test_phase4_rl_optimizer_properties.py # Properties 24–26
    └── test_phase5_oos_properties.py       # Property 27
```

### Running Tests

```bash
# Run all tests
pytest tests/ --hypothesis-seed=42

# Run only unit tests
pytest tests/unit/

# Run only property-based tests
pytest tests/property/ --hypothesis-seed=42

# Run with verbose output
pytest tests/ --hypothesis-seed=42 -v

# Run a specific test file
pytest tests/unit/test_cpu_engine.py -v
```

### Property-Based Tests

The property tests use [Hypothesis](https://hypothesis.readthedocs.io/) to verify universal correctness properties:

| Property | Description                                | Validates     |
| -------- | ------------------------------------------ | ------------- |
| 1        | Per-symbol chronological sort              | Req 2.2       |
| 2        | Last-288-row drop                          | Req 2.3       |
| 3        | No NaN labels after loading                | Req 2.4       |
| 4        | No NaN features after loading              | Req 2.5       |
| 5        | Per-symbol split ratio and no overlap      | Req 2.6, 2.7  |
| 6        | Feature mode classification completeness   | Req 3.1       |
| 7        | Feature mode classification correctness    | Req 3.2       |
| 8        | Fuzzy value name encoding round-trip       | Req 4.1, 4.2  |
| 9        | Don't-care sentinel correctness            | Req 4.3       |
| 10       | Priority-based rule assignment exclusivity | Req 5.1       |
| 11       | Trade outcome correctness                  | Req 5.2       |
| 12       | Capital-managed position sizing            | Req 5.4, 5.9  |
| 13       | Exposure reservation invariant             | Req 5.5       |
| 14       | Fee deduction correctness                  | Req 5.6       |
| 15       | Equity tracking consistency                | Req 5.7       |
| 16       | GPU-CPU numerical parity                   | Req 6.1       |
| 17       | Label and meta column exclusion            | Req 7.2       |
| 18       | Low-dispersion feature exclusion           | Req 7.5       |
| 19       | Phase 2 static risk parameters             | Req 8.4       |
| 20       | Rule condition count bounds                | Req 8.6       |
| 21       | Rule set size bounds                       | Req 9.1, 12.8 |
| 22       | Rule set uniqueness                        | Req 9.4       |
| 23       | JSON output schema validity                | Req 12.1–12.9 |
| 24       | RL action bounds                           | Req 10.3      |
| 25       | RL state vector completeness               | Req 10.2      |
| 26       | Elbow method correctness                   | Req 10.5      |
| 27       | Test data preparation consistency          | Req 11.2      |
| 28       | Per-symbol metrics consistency             | Req 15.1      |
| 29       | Symbol coverage penalty application        | Req 9.5, 15.4 |

---

## 13. Design Principles

### Single Source of Truth

All hyperparameters live in `config.py`. No module defines its own defaults that override these values. No runtime flags are used — change `config.py` to tune the pipeline.

### Label Isolation

Label columns (`label_*`) are used **only** for trade simulation, never as model inputs. This is enforced at every stage: feature selection explicitly excludes them, and the encoder never encodes them.

### Temporal Integrity

The train/validation split is per-symbol chronological (75/25). The last 288 rows per symbol are always dropped. This prevents any form of temporal leakage.

### Evaluator Parity

The internal `CPUBacktestEngine` exactly mirrors `evaluator_v3.ipynb`'s `CapitalManagedTradeSimulator` semantics. This means optimization scores during Phases 2 and 3 are directly comparable to final evaluation scores.

### GPU-First with Transparent Fallback

**Phase 2:** JAX batch backtests when available; **NSGA-III** via EvoX for survivor selection. Without EvoX, Phase 2 falls back to NumPy **NSGA-II** (same objectives and mating operators). **Phase 3** refinement still uses NSGA-II over rule-set combinations. CPU-only environments are supported; GPU/JAX tests skip automatically when JAX is not installed.

### Phase Isolation

Each phase produces persisted artifacts. Completed phases are skipped on re-runs. This makes long optimization runs resumable and individual phases independently inspectable.

### Symbol-Aware Evaluation

All backtest evaluations track per-symbol metrics. Feature selection scores features per symbol and measures cross-symbol stability. Rule set selection penalizes strategies that fail to generate trades on most symbols. The goal is rules that generalize across all 10 instruments, not rules that overfit to one.

### Separation of Concerns

The pipeline separates:

1. **Feature mining** (Phase 1) — which features are predictive and stable
2. **Rule generation** (Phase 2) — what individual rules look like
3. **Ensemble selection** (Phase 3) — which combination of rules works best as a team
4. **Risk tuning** (Phase 4) — what TP/SL/capital allocation maximizes risk-adjusted return
5. **Out-of-sample truth** (Phase 5) — does the strategy actually generalize

Each layer can be inspected independently, making it easier to diagnose where the pipeline succeeds or fails.

---

## Relationship to Previous Implementation

This package (`gpu_fuzzy_trader`) is a complete ground-up rewrite of the previous `bigdata_trader` package, incorporating lessons learned from that implementation:

| Aspect            | `bigdata_trader` (old)   | `gpu_fuzzy_trader` (new)                             |
| ----------------- | ------------------------ | ---------------------------------------------------- |
| GPU acceleration  | Optional, partial        | JAX-first, transparent CPU fallback                  |
| Feature selection | Global, single direction | Direction-specific (long/short)                      |
| Rule encoding     | Mixed                    | Strict chromosome encoding with don't-care sentinels |
| Risk optimization | Static TP/SL             | RL agent with Elbow Method stopping                  |
| Test coverage     | Limited                  | 713 tests, 29 property-based properties              |
| Config            | Mixed flags + config     | Single `config.py`, no runtime flags                 |
| Evaluator parity  | Approximate              | Exact mirror of `evaluator_v3.ipynb`                 |
| Skip logic        | Basic                    | Validation helpers remain; default CLI forces rerun  |

The output format (`long.json` / `short.json`) is identical between both implementations and fully compatible with `evaluator_v3.ipynb`.
