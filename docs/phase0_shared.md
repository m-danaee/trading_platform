# Phase 0 — Shared Infrastructure

This document covers the foundational components that every phase depends on: data loading, the train/validation split, the backtest engine, and all shared configuration constants. Understanding these is a prerequisite for understanding any individual phase.

---

## 1. Data Loading — `Data_Loader` (`gpu_fuzzy_trader/data/loader.py`)

`Data_Loader.load_dataset(path, feature_cols=None)` is a stateless, 7-step preparation pipeline. It is called identically for both the training CSV and the test CSV, ensuring no data-preparation leakage between splits.

### Step-by-step pipeline

**Step 1 — Read CSV**
Reads the file with `pd.read_csv(path, sep=",")`. No dtype coercion at this stage.

**Step 2 — Parse datetime**
Converts the `datetime` column to `pd.Timestamp` objects via `pd.to_datetime`. This enables chronological sorting and is required for the per-symbol split.

**Step 3 — Sort by (symbol, datetime)**
`df.sort_values(["symbol", "datetime"])` ensures that within each symbol, rows are in strict chronological order. This is the contract that `Data_Splitter` relies on.

**Step 4 — Drop last `TAIL_DROP_ROWS` rows per symbol**
The label columns (`label_close_288`, `label_min_288`, `label_max_288`) require a 288-bar look-ahead window. The last 288 rows of each symbol have no valid labels because the window extends beyond the dataset boundary. These rows are dropped using a reverse cumcount:

```python
tail_count = df.groupby("symbol").cumcount(ascending=False)
df = df[tail_count >= TAIL_DROP_ROWS]
```

`TAIL_DROP_ROWS = 288` (config). At 5-minute bars, 288 bars = 24 hours. Increasing this value would drop more rows but is only necessary if you change the label horizon. Decreasing it risks including rows with NaN labels.

**Step 5 — Drop rows where any label column is NaN**
After the tail drop, any remaining NaN in label columns is dropped. This handles edge cases like gaps in the data.

**Step 6 — Fill feature NaN with 0**
Feature columns (everything that is not a label or meta column) have their NaN values filled with 0. This is a deliberate design choice: the features are discretized integers, and 0 is a valid fuzzy state (e.g., "Very Low" or "Inactive"). Filling with 0 avoids introducing a new sentinel value.

**Step 7 — Compute `_symbol_bar_index`**
`df.groupby("symbol").cumcount()` assigns a sequential integer index (0, 1, 2, …) to each row within its symbol, after all drops. This is used by the backtest engine to compute exposure release windows.

**Final step — `downcast_numeric_df`**
All numeric columns are downcast to the smallest safe dtype (e.g., `float64 → float32`, `int64 → int16`). This reduces RAM by roughly 2×.

---

## 2. Train/Validation Split — `Data_Splitter` & `cv_folds`

**Modules:**
- `gpu_fuzzy_trader/data/splitter.py` → `Data_Splitter.split_and_persist`
- `gpu_fuzzy_trader/data/cv_folds.py` → purged rolling fold construction

`Data_Splitter.split_and_persist(df)` returns `(train_df, validation_df, cv_folds)` and always writes:

- `data/train_75.parquet`
- `data/validation_25.parquet`
- `data/cv_folds_manifest.json` (summary only, when using purged CV)

Behaviour is controlled by **`SPLIT_MODE`** in `config.py`.

### Mode comparison

| `SPLIT_MODE` | Phases 2–3 fitness | Persisted train/val | Speed |
|---|---|---|---|
| `"purged_rolling_cv"` (default) | K expanding-window folds; **worst fold** across metrics | **Last fold** train + val | Slower (~K× eval) |
| `"holdout_75_25"` | Single train + single val per symbol | Classic 75/25 split | Faster |

**Recommendation:** Use `purged_rolling_cv` when validation performance must generalize across seasons (especially short). Use `holdout_75_25` for debugging or reproducing older runs.

### A. Purged rolling CV (`purged_rolling_cv`)

Implemented in `build_purged_rolling_folds()` (`cv_folds.py`).

For each symbol independently (rows already chronological):

1. Require at least **`CV_MIN_TRAIN_MONTHS`** of history before any validation window (default 2 months ≈ 17,280 bars at 5-minute resolution).
2. Split the remaining timeline into **`CV_N_FOLDS`** contiguous validation segments (default 3).
3. For fold *i* with validation `[v_start, v_end)`:
   - **Train:** rows `[0, v_start − CV_EMBARGO_BARS)` (expanding window, purged).
   - **Val:** rows `[v_start, v_end)`.
4. Concatenate the same fold index across all symbols → one `PurgedFold` per index.

**Embargo:** `CV_EMBARGO_BARS = TAIL_DROP_ROWS` (288) prevents label-horizon leakage between train and val.

**Phases 2–3:** Each fold gets its own backtest engine (subsampled). Fitness aggregates with a **conservative merge** (minimum return/Sortino/PF, maximum drawdown) across folds — rules must survive every season, not just the last 25%.

**Phases 4–5:** Still use the **last fold** persisted as `train_75` / `validation_25` (most recent in-sample OOS block before `test.csv`). Phase 4 walk-forward runs on `validation_25` only, not on all CV folds.

If fold construction fails (dataset too short), the splitter logs a warning and **falls back** to `holdout_75_25`.

### B. Legacy holdout (`holdout_75_25`)

`holdout_75_25_split()` — per symbol:

1. `split_point = floor(N × 0.75)`.
2. Rows `[0, split_point)` → train; `[split_point, N)` → validation.

`cv_folds` is empty; Phases 2–3 use one train engine and one val engine.

### Why per-symbol?

A global time cut would leave some symbols entirely in train or val. Per-symbol splits guarantee every symbol appears in both partitions — required for `PHASE3_MIN_SYMBOL_COVERAGE`.

### Parquet cache

- **`holdout_75_25`:** If `train_75.parquet` and `validation_25.parquet` are newer than `train.csv`, the pipeline loads the cache and skips splitting.
- **`purged_rolling_cv`:** Cache is **not** used for the full pipeline load (folds must be rebuilt from `train.csv`). Parquet files are still written for Phase 4–5.

After changing `SPLIT_MODE` or CV parameters, delete `data/train_75.parquet`, `data/validation_25.parquet`, and `data/cv_folds_manifest.json`, then rerun.

---

## 3. Backtest Engine — `CPUBacktestEngine` (`gpu_fuzzy_trader/backtest/cpu_engine.py`)

This is the canonical reference implementation. It exactly mirrors `evaluator_v3.ipynb`'s `CapitalManagedTradeSimulator`. All optimization scores during Phases 2, 3, and 4 must match the final evaluation scores in Phase 5, so this engine is the single source of truth for trade simulation semantics.

### Rule matching — `_apply_dynamic_rule`

Each condition string `"[feature_name] IS Fuzzy Value Name"` is evaluated using **threshold-based logic**, not mode-based discretization. The feature column is treated as a continuous value and compared against fixed thresholds:

| Fuzzy Value Name | Threshold logic |
|---|---|
| `Very Low` | `s <= 0.2` |
| `Low` | `0.2 < s <= 0.4` |
| `Medium` | `0.4 < s <= 0.6` |
| `High` | `0.6 < s <= 0.8` |
| `Very High` | `s > 0.8` |
| `Extreme Bearish` | `s <= -0.8` |
| `Strong Bearish` | `-0.8 < s <= -0.6` |
| … | … |
| `Active (1)` | `s == 1` |
| `Inactive (0)` | `s == 0` |

This is mode-independent: the same threshold logic applies regardless of how the feature was classified in Phase 1. This matches `evaluator_v3.ipynb` exactly.

### Priority-based rule assignment — `_build_entries_from_rule_set`

For each candle (row), the **first rule** in the ordered rule set whose conditions all match is assigned. Subsequent rules are skipped for that candle. This prevents duplicate entries on the same symbol/time. The order of rules in `rules_set` therefore matters: earlier rules have priority.

### Trade outcome logic — `_build_trade_outcome_single`

**Long direction:**
- TP hit: `label_max_288 ≥ entry × (1 + tp/100)`
- SL hit: `label_min_288 ≤ entry × (1 − sl/100)`
- Both hit: `label_max_before_min == 1` → TP first (max came before min); else SL first
- Neither hit: time exit at `label_close_288`

**Short direction:**
- TP hit: `label_min_288 ≤ entry × (1 − tp/100)`
- SL hit: `label_max_288 ≥ entry × (1 + sl/100)`
- Both hit: `label_max_before_min == 1` → SL first (max came before min, so SL was hit first for a short); else TP first
- Neither hit: time exit at `−close_ret`

### Capital management — `_calculate_position_notional`

```
position_notional = min(
    equity × (capital_pct / 100) × leverage,
    max(0, equity × MAX_TOTAL_EXPOSURE_PCT/100 × leverage − open_total_exposure)
)
```

Trades are skipped if `position_notional < MIN_POSITION_NOTIONAL` (1.0). This prevents micro-trades when equity is nearly depleted.

### Exposure reservation — `precompute_release_indices`

Each open trade reserves exposure until `entry_symbol_bar_index + MAX_HOLD_CANDLES`. The release index is precomputed using `np.searchsorted` for efficiency. PnL is realized at the release point, not at entry. This prevents using future PnL to size new trades — a critical anti-lookahead measure.

### Fee deduction

```
fee = position_notional × FEE_PCT / 100
net_pnl = gross_pnl − fee
```

`FEE_PCT = 0.20` is a round-trip fee percentage. At 0.20%, a $1000 position costs $2 in fees. This penalizes high-turnover rules.

All phases use `FEE_PCT` consistently for fee deduction.

### Sortino Ratio computation — `_sortino_ratio_from_returns`

```
sortino = mean(excess_returns) / downside_deviation
```

where `excess_returns = trade_returns − target_return` (target = 0) and `downside_deviation = sqrt(mean(min(excess_returns, 0)²))`.

If `downside_deviation == 0` and `mean_excess_return > 0`, returns `SORTINO_CAP` (5.0). This handles the case of a rule with no losing trades.

The Sortino is **not annualized** — it is computed on per-trade equity returns. This makes it comparable across rules with different trade frequencies.

### Account ruin

Simulation stops and marks `account_ruined = True` when `equity ≤ 0`. All subsequent entries are skipped.

---

## 4. Shared Configuration Constants (`gpu_fuzzy_trader/config.py`)

### Paths

| Parameter | Default | Effect |
|---|---|---|
| `TRAIN_CSV_PATH` | `data/train.csv` | Source for Phases 1–4 |
| `TEST_CSV_PATH` | `data/test.csv` | **Phase 5 only** — never tune on this |
| `TRAIN_75_PATH` | `data/train_75.parquet` | Persisted train block (last CV fold or 75%) |
| `VALIDATION_25_PATH` | `data/validation_25.parquet` | Persisted val block (last CV fold or 25%) |
| `CV_FOLDS_MANIFEST_PATH` | `data/cv_folds_manifest.json` | Fold row counts / date ranges (purged CV) |
| `OUTPUTS_DIR` | `outputs` | Per-run outputs |
| `PHASE2_ARCHIVE_DIR` | `phase2_rule_archive/` | Cross-run warm-start archive |

### Split mode (Phases 2–3)

| Parameter | Default | Effect |
|---|---|---|
| `SPLIT_MODE` | `"purged_rolling_cv"` | `"holdout_75_25"` for legacy single split |
| `CV_N_FOLDS` | `3` | Number of rolling validation windows |
| `CV_EMBARGO_BARS` | `288` | Purge gap before each val block (= label horizon) |
| `CV_BARS_PER_DAY` | `288` | Bars per day for month → bar conversion |
| `CV_MIN_TRAIN_MONTHS` | `2.0` | Minimum train rows per symbol per fold |

### Schema constants

| Parameter | Value | Effect |
|---|---|---|
| `LABEL_COLUMNS` | 5 columns | Never enter feature matrices |
| `META_COLUMNS` | `datetime`, `symbol` | Never enter feature matrices |
| `TAIL_DROP_ROWS` | `288` | Rows dropped per symbol at the end |

### Backtest constants

These must match `evaluator_v3.ipynb` exactly. Changing them will cause optimization scores to diverge from final evaluation scores.

| Parameter | Default | Effect |
|---|---|---|
| `INITIAL_CAPITAL` | `1000.0` | Starting equity for all simulations |
| `LEVERAGE` | `1.0` | Position sizing multiplier. Increasing allows larger positions relative to equity, amplifying both gains and losses. |
| `FEE_PCT` | `0.20` | Round-trip fee as % of notional. Increasing penalizes high-turnover rules more heavily. |
| `MAX_HOLD_CANDLES` | `288` | Maximum bars a position can be held. Aligned with the 288-bar label horizon. Decreasing would force earlier time exits. |
| `MAX_TOTAL_EXPOSURE_PCT` | `100.0` | Maximum % of equity that can be simultaneously deployed. Decreasing reduces risk but also limits upside. |
| `MIN_POSITION_NOTIONAL` | `1.0` | Minimum trade size in currency units. Trades below this are skipped. |

### Logging

| Parameter | Default | Effect |
|---|---|---|
| `LOG_GENERATION_INTERVAL` | `0` | `0` = auto-throttle (logs every ~10% of generations). Set to `N > 0` to log every N generations. |
