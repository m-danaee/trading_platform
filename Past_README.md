# Trading System

## 1. Project Goal

- Uses **5 feature Lables in dataset (no OHLC) to handle open and close trades**
- Uses **feature clustering** to guide the models
- **TP range: 2% to 5% ($20 to $50)**
- **SL range: 1% to 2.5% ($10 to $25)**
- If we have sampling in any stage of training it must be share between all symbols for example for 300/000 sampling in GA it must use 30/000 rows per each symbol for training (we have 10 symbols)
- train.csv dataset is for training and validation (75-25) and test.csv only for testing at the end
- The pipeline now reads the split datasets from Parquet by default (`train_75.parquet`, `validation_25.parquet`, `test.parquet`) and falls back to CSV only if a Parquet file is missing.
- If we have spliting train.csv to 75-25 it not split from start row to 75 then until end because we have 10 symbols and it must split per symbol train.csv file
- I'm now split train.csv to train_75.csv and validation_25.csv files
- You must don't use any feature Lables for training just use these for opening and closing positions because these are look ahead datas!

### Overview

- Selected important features from your dataset
- Activated fuzzy rules using real feature names
- Position decision (Long / Short / No Trade)
- TP / SL levels in percentage
- Final PnL performance
- `At each script you must consider to optimize this script to run on GPU to reduce overall training time`
- **TP and SL :**
  Take profit and stop loss values must be floating-point numbers. For example,
  a TP of 5% and SL of 2.5%, where these percentages are based on initial capital
  (2.5% of initial capital is at risk per trade).
- In addition this project must produce two JSON files (`short` and `long`)
  containing the best strategy discovered by their algorithm
- **Prediction Horizon (The “288” Suffix):**
  All labels with the suffix \_288 (e.g., label_close_288, label_high_288) refer to a look-ahead window of 288 bars. Given the dataset’s 5-minute timeframe, this represents exactly a 24-hour period (288 bars × 5 minutes = 1440 minutes). This horizon is chosen to capture daily price cycles and significant intraday trends, providing the models with a consistent daily target for forecasting and strategy development.

---

## 2. Understanding Your Dataset

Your dataset contains **future labels instead of OHLC**:

- `label_open_next`
- `label_close_288`
- `label_min_288`
- `label_max_288`
- `label_max_before_min`

Plus **hundreds of engineered features** like:

- `mom_stoch_rsi_14_14_3` — momentum indicators
- `mom_tl_break_bull_30` — trend line breaks
- `vol_*` — volatility features
- `mean_rev_*` — mean reversion features
- And many more...

`The last 288 rows for each symbol in the train and test datasets have no labels, so you must handle them (e.g., remove them)`

Each column in my dataset represents a feature (indicator or price behavior). Feature values
are converted into discrete classes rather than floating-point numbers. There are 5 feature types:

1. **Binary:** 2 states (e.g., 0 = inactive, 1 = active)
2. **Ternary:** 3 states (e.g., -1, 0, 1)
3. **Positive / Positive Sparse:** 5 magnitude states (e.g., 0 = very low to 4 = very high)
4. **Signed Sparse:** 5 directional states (from strongly negative to strongly positive)
5. **Signed:** 10 finer states (from strongly bearish to strongly bullish)

- My project must not train on all dataset because it has symbol column and each symbol have different price and etc (main goal is to find best rule sets (one Long, one Short) that has works on all symbols

Code to detect feature mode :

```python
def detect_feature_mode(series):
unique_vals = series.dropna().unique()
n_unique = len(unique_vals)
if n_unique <= 2 and set(unique_vals).issubset({0, 1}):
return "binary"
if n_unique <= 3 and set(unique_vals).issubset({-1, 0, 1}):
return "ternary"

    zero_ratio = (series == 0).mean()
    if series.min() < 0:
        return "sparse_signed" if zero_ratio > 0.3 else "signed"
    else:
        return "sparse_positive" if zero_ratio > 0.3 else "positive"

def get_features_info(df, feature_cols):
feature_info = []
for col in feature_cols:
mode = detect_feature_mode(df[col])
if mode == "binary":
feature_info.append({"col": col, "mode": mode, "num_classes": 2, "dont_care": 2})
elif mode == "ternary":
feature_info.append({"col": col, "mode": mode, "num_classes": 3, "dont_care": 3})
elif mode in ["positive", "sparse_positive"]:
feature_info.append({"col": col, "mode": mode, "num_classes": 5, "dont_care": 5})
elif mode == "sparse_signed":
feature_info.append({"col": col, "mode": mode, "num_classes": 5, "dont_care": 5})
else: # signed
feature_info.append({"col": col, "mode": mode, "num_classes": 10, "dont_care": 10})
return feature_info
```

## 7. Feature Selection (Implemented)

This project now includes an **output-type-aware** feature selector that follows the rules in `FEATURE_SELECTION.md`:

- Detects feature output type (`binary`, `ternary`, `positive / sparse_positive`, `sparse_signed`, `signed`)
- Splits data **per symbol** and **time-based** (train/validation)
- Builds **separate selections for long and short** using a direction-specific trade-success target derived from the 5 label columns
- Removes redundancy **within each output-type category**

Run it:

```bash
python -m bigdata_trader.select_features
```

Outputs:

- `outputs/selected_features_global.json`
- `outputs/feature_selection_report_global.csv`

## 8. Three-Phase MOEA/D Pipeline (Implemented)

Run full pipeline:

```bash
python -m bigdata_trader.run_pipeline
```

Optional phase-only runs:

```bash
python -m bigdata_trader.run_pipeline --phase phase1
python -m bigdata_trader.run_pipeline --phase phase2
python -m bigdata_trader.run_pipeline --phase phase3
```

Outputs:

- Feature selection:
  - `outputs/selected_features_global.json`
- Phase 1:
  - `outputs/reports/phase1_long_generation_metrics.png`
  - `outputs/reports/phase1_short_generation_metrics.png`
- Phase 2:
  - `outputs/long.json`
  - `outputs/short.json`
  - `outputs/reports/train_trade_dashboard.png`
  - `outputs/reports/validation_trade_dashboard.png`
  - `outputs/reports/validation_per_symbol_dashboard.png`
- Phase 3:
  - `outputs/reports/test_trade_dashboard.png`
  - `outputs/reports/test_trade_dashboard_long.png`
  - `outputs/reports/test_trade_dashboard_short.png`
  - `outputs/reports/test_per_symbol_dashboard.png`

## 9. Complete Training Pipeline

```python
if __name__ == "__main__":
    print("Loading data to extract feature info...")
    # ⚠️ Enter the path to your dataset file here.
    df = pd.read_csv("data/final_train_dataset.csv")

    label_cols = ["label_open_next", "label_close_288", "label_min_288", "label_max_288", "label_max_before_min"]
    meta_cols = ["datetime", "symbol"]
    feature_cols = [c for c in df.columns if c not in label_cols + meta_cols]

    # Extract feature metadata.
    feature_info = get_features_info(df, feature_cols)

    # Generate random numerical rules (you will replace this part with your own AI).
    best_rules, best_tps, best_sls = generate_random_strategy(feature_info, max_rules=3)

    # Translate numerical values into human‑readable strings.
    final_human_readable_strategy = decode_to_human_readable(
        best_rules, best_tps, best_sls, feature_info, direction="long"
    )

    # Save to a JSON file.
    output_filename = "student_strategy_submission.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(final_human_readable_strategy, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Strategy generated and saved to '{output_filename}'")
    print("=" * 50)
    print("Preview of your submission format:\n")
    print(json.dumps(final_human_readable_strategy, indent=4))
```

## 10. Backtesting Engine

### Trade Simulation Logic

- Use 5 feature lables for trading :

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import json
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
#  Main Settings
# ==============================================================================
TRAIN_FILE_PATH = "data/final_train_dataset.csv"  # Training file path
TEST_FILE_PATH = "data/final_test_dataset.csv"    # Test file path (out-of-sample)
FEE_PCT = 0.20                                    # Round-trip fee (percent)

# ==============================================================================
#  Discovered Strategy
# ==============================================================================
best_strategy = {
    'direction': 'long',
    'rules_set': [
        {
            'tp': 3.45,
            'sl': 4.11,
            'capital_pct': 40.0,
            'conditions': [
                '[amihud_illiquidity_20] IS Very High',
                '[cand_up_down_vol_ratio_20] IS Low',
                '[dollar_vol_rel_20] IS High',
                '[lower_wick_to_tr] IS Very High',
                '[ret_vol_corr_30] IS Weak Bearish',
            ]
        },
        {
            'tp': 4.08,
            'sl': 1.65,
            'capital_pct': 42.5,
            'conditions': [
                '[ema_gap_atr_20] IS Bullish',
                '[parkinson_vol_20] IS Very Low',
                '[rsi_centered_14] IS Weak Bullish',
                '[rsi_div_persistence] IS Extreme Bullish',
            ]
        },
        {
            'tp': 3.85,
            'sl': 3.20,
            'capital_pct': 35.0,
            'conditions': [
                '[mom_tl_break_bull_30] IS Active (1)',
                '[roc_10] IS Neutral Positive',
                '[vol_ratio_20_100] IS High',
            ]
        },
    ]
}


# ==============================================================================
# 🛠 Base and Helper Functions
# ==============================================================================
def load_and_clean_data(file_path):
    if not os.path.exists(file_path):
        print(f"❌ Error: File '{file_path}' not found.")
        return pd.DataFrame()

    try:
        try:
            df = pd.read_csv(file_path, sep="\t")
            if len(df.columns) < 5:
                df = pd.read_csv(file_path, sep=",")
        except:
            df = pd.read_csv(file_path, sep=",")

        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime").reset_index(drop=True)

        label_cols = [
            "label_open_next",
            "label_close_288",
            "label_min_288",
            "label_max_288",
            "label_max_before_min",
        ]
        df.dropna(subset=label_cols, inplace=True)

        feature_cols = [
            c for c in df.columns if c not in label_cols + ["datetime", "symbol"]
        ]
        df[feature_cols] = df[feature_cols].fillna(0)

        return df.reset_index(drop=True)
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return pd.DataFrame()


def apply_dynamic_rule(df, rule_string):
    try:
        feature_part, value_part = rule_string.split(" IS ")
        feature_name = feature_part.strip()[1:-1]
        value_name = value_part.strip()
    except ValueError:
        return pd.Series([True] * len(df), index=df.index)

    if feature_name not in df.columns:
        return pd.Series([True] * len(df), index=df.index)

    s = df[feature_name]

    if value_name == "Active (1)":
        return s == 1
    if value_name == "Inactive (0)":
        return s == 0
    if value_name in ["Positive", "Positive (1)"]:
        return s == 1
    if value_name in ["Neutral", "Neutral (0)"]:
        return s == 0
    if value_name in ["Negative", "Negative (-1)"]:
        return s == -1

    if (
        value_name == "Strong Negative"
        or value_name == "Strong Negative (e.g. Big Down Gap)"
    ):
        return s <= -0.25
    if value_name == "Weak Negative":
        return (s > -0.25) & (s <= -1e-5)
    if value_name == "Exactly Zero" or value_name == "Exactly Zero (No Gap)":
        return (s > -1e-5) & (s <= 1e-5)
    if value_name == "Weak Positive":
        return (s > 1e-5) & (s <= 0.25)
    if (
        value_name == "Strong Positive"
        or value_name == "Strong Positive (e.g. Big Up Gap)"
    ):
        return s > 0.25

    if value_name == "Very Low":
        return s <= 0.2
    if value_name == "Low":
        return (s > 0.2) & (s <= 0.4)
    if value_name == "Medium":
        return (s > 0.4) & (s <= 0.6)
    if value_name == "High":
        return (s > 0.6) & (s <= 0.8)
    if value_name == "Very High":
        return s > 0.8

    if value_name == "Extreme Bearish":
        return s <= -0.8
    if value_name == "Strong Bearish":
        return (s > -0.8) & (s <= -0.6)
    if value_name == "Bearish":
        return (s > -0.6) & (s <= -0.4)
    if value_name == "Weak Bearish":
        return (s > -0.4) & (s <= -0.2)
    if value_name == "Neutral Negative":
        return (s > -0.2) & (s <= 0.0)
    if value_name == "Neutral Positive":
        return (s > 0.0) & (s <= 0.2)
    if value_name == "Weak Bullish":
        return (s > 0.2) & (s <= 0.4)
    if value_name == "Bullish":
        return (s > 0.4) & (s <= 0.6)
    if value_name == "Strong Bullish":
        return (s > 0.6) & (s <= 0.8)
    if value_name == "Extreme Bullish":
        return s > 0.8

    print(
        f"⚠️ Warning: Rule value '{value_name}' not recognized for feature '{feature_name}'"
    )
    return pd.Series([True] * len(df), index=df.index)


# ==============================================================================
#  Trade Processing Core
# ==============================================================================
def generate_trade_logs(df, signals, tp_series, sl_series, rule_ids, direction="long"):
    trades = df[signals].copy()

    if trades.empty:
        return pd.DataFrame()

    entry_price = trades["label_open_next"]
    max_price = trades["label_max_288"]
    min_price = trades["label_min_288"]
    close_price = trades["label_close_288"]
    max_first = trades["label_max_before_min"]

    tp_pct = tp_series[signals]
    sl_pct = sl_series[signals]
    triggered_rule_ids = rule_ids[signals]

    if direction == "long":
        tp_target = entry_price * (1 + tp_pct)
        sl_target = entry_price * (1 - sl_pct)

        tp_hit = max_price >= tp_target
        sl_hit = min_price <= sl_target

        cond_both_tp_first = tp_hit & sl_hit & (max_first == 1)
        cond_both_sl_first = tp_hit & sl_hit & (max_first == 0)
        time_pnl = (close_price - entry_price) / entry_price

    elif direction == "short":
        tp_target = entry_price * (1 - tp_pct)
        sl_target = entry_price * (1 + sl_pct)

        tp_hit = min_price <= tp_target
        sl_hit = max_price >= sl_target

        cond_both_tp_first = tp_hit & sl_hit & (max_first == 0)
        cond_both_sl_first = tp_hit & sl_hit & (max_first == 1)
        time_pnl = (entry_price - close_price) / entry_price

    cond_tp_only = tp_hit & ~sl_hit
    cond_sl_only = sl_hit & ~tp_hit
    cond_time = ~tp_hit & ~sl_hit & close_price.notna()

    win_condition = cond_tp_only | cond_both_tp_first
    loss_condition = cond_sl_only | cond_both_sl_first

    pnl = np.select(
        [win_condition, loss_condition, cond_time],
        [tp_pct, -sl_pct, time_pnl],
        default=np.nan,
    )

    exit_reason = np.select(
        [win_condition, loss_condition, cond_time],
        ["TP", "SL", "Time_288"],
        default="Unknown",
    )

    trades_df = pd.DataFrame(
        {
            "Symbol": trades["symbol"],
            "Entry_Time": trades["datetime"],
            "Rule_ID": triggered_rule_ids,
            "TP_Set_%": tp_pct * 100,
            "SL_Set_%": sl_pct * 100,
            "Entry_Price": entry_price,
            "PnL_Pct": pnl * 100,
            "Exit_Reason": exit_reason,
        }
    ).dropna(subset=["PnL_Pct"])

    if trades_df.empty:
        return trades_df

    trades_df["PnL_Pct"] = trades_df["PnL_Pct"] - FEE_PCT
    trades_df["Cumulative_PnL"] = trades_df["PnL_Pct"].cumsum()

    liquidation_points = trades_df[trades_df["Cumulative_PnL"] <= -100]
    if not liquidation_points.empty:
        liq_index_label = liquidation_points.index[0]
        trades_df = trades_df.loc[:liq_index_label].copy()
        trades_df.loc[liq_index_label, "Cumulative_PnL"] = -100.0
        trades_df.loc[liq_index_label, "Exit_Reason"] = "LIQUIDATED"

    trades_df["High_Water_Mark"] = trades_df["Cumulative_PnL"].cummax().clip(lower=0)
    trades_df["Drawdown"] = trades_df["High_Water_Mark"] - trades_df["Cumulative_PnL"]

    return trades_df


def run_dynamic_test_and_plot(strategy, df_data, dataset_name="Dataset"):
    print(f"\n🚀 Running Priority-Based Strategy on {dataset_name}...")

    if df_data.empty:
        print(f"❌ No valid data for {dataset_name}.")
        return None, None

    # 🚀 Priority Engine
    triggered = pd.Series([False] * len(df_data), index=df_data.index)
    active_tps = pd.Series([0.0] * len(df_data), index=df_data.index)
    active_sls = pd.Series([0.0] * len(df_data), index=df_data.index)
    rule_ids = pd.Series([0] * len(df_data), index=df_data.index)

    rules_list = strategy.get("rules_set", [])

    for r_idx, rule_dict in enumerate(rules_list):
        subset_signals = pd.Series([True] * len(df_data), index=df_data.index)

        for condition in rule_dict.get("conditions", []):
            subset_signals &= apply_dynamic_rule(df_data, condition)

        # Reserve candles: only signals not already reserved by higher-priority rules are applied
        new_triggers = subset_signals & ~triggered

        active_tps[new_triggers] = rule_dict["tp"] / 100.0
        active_sls[new_triggers] = rule_dict["sl"] / 100.0
        rule_ids[new_triggers] = r_idx + 1  # Store rule number
        triggered |= subset_signals

    trade_logs = generate_trade_logs(
        df_data,
        triggered,
        active_tps,
        active_sls,
        rule_ids,
        direction=strategy.get("direction", "long"),
    )

    if trade_logs.empty:
        print(f"❌ No trades triggered on {dataset_name}.")
        return trade_logs, None

    num_trades = len(trade_logs)
    is_liquidated = "LIQUIDATED" in trade_logs["Exit_Reason"].values
    total_pnl = trade_logs["Cumulative_PnL"].iloc[-1]
    win_rate = (len(trade_logs[trade_logs["PnL_Pct"] > 0]) / num_trades) * 100
    max_drawdown = trade_logs["Drawdown"].max()

    print("=" * 50)
    print(f"📊 {dataset_name.upper()} PERFORMANCE REPORT")
    print("=" * 50)
    if is_liquidated:
        print("🚨🚨 ACCOUNT LIQUIDATED (MARGIN CALL) 🚨🚨")
    print(
        f"Total Trades:      {num_trades} {'(Stopped)' if is_liquidated else '(Completed)'}"
    )
    print(f"Win Rate:          {win_rate:.2f}%")
    print(f"Total Net PnL:     {total_pnl:.2f}%")
    print(f"Average Per Trade: {trade_logs['PnL_Pct'].mean():.4f}%")
    print(f"Max Drawdown:      {max_drawdown:.2f}%")
    print("=" * 50)

    # Performance report per individual rule
    print("📈 Performance by Rule ID:")
    rule_stats = trade_logs.groupby("Rule_ID").agg(
        Trades=("PnL_Pct", "count"),
        Net_PnL=("PnL_Pct", "sum"),
        Win_Rate=("PnL_Pct", lambda x: (x > 0).mean() * 100),
    )
    for rule_id, row in rule_stats.iterrows():
        print(
            f"   Rule {rule_id}: {int(row['Trades'])} Trades | Win: {row['Win_Rate']:.1f}% | PnL: {row['Net_PnL']:.2f}%"
        )
    print("=" * 50)

    metrics = {
        "total_trades": int(num_trades),
        "win_rate": float(win_rate),
        "total_pnl": float(total_pnl),
        "max_drawdown": float(max_drawdown),
        "is_liquidated": bool(is_liquidated),
    }

    plot_trade_analysis(trade_logs, title_prefix=dataset_name)

    return trade_logs, metrics


def plot_trade_analysis(trades_df, title_prefix=""):
    plt.figure(figsize=(12, 8))

    try:
        time_x = pd.to_datetime(trades_df["Entry_Time"])
    except:
        time_x = trades_df.index

    is_liquidated = "LIQUIDATED" in trades_df["Exit_Reason"].values

    plt.subplot(2, 1, 1)
    color_pnl = "darkred" if is_liquidated else "green"
    plt.plot(
        time_x,
        trades_df["Cumulative_PnL"],
        label="Cumulative PnL (%)",
        color=color_pnl,
        linewidth=2,
    )
    plt.fill_between(time_x, trades_df["Cumulative_PnL"], 0, alpha=0.1, color=color_pnl)

    if is_liquidated:
        plt.axhline(y=-100, color="red", linestyle="--", label="Liquidation Level")

    plt.title(f"{title_prefix} - Cumulative PnL Over Time")
    plt.ylabel("Net PnL (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(
        time_x, -trades_df["Drawdown"], label="Drawdown (%)", color="red", linewidth=1.5
    )
    plt.fill_between(time_x, -trades_df["Drawdown"], 0, alpha=0.2, color="red")

    plt.title("Drawdown Over Time (Underwater Chart)")
    plt.ylabel("Drawdown (%)")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()


def save_backtest_results(
    strategy, train_metrics, test_metrics, train_logs, test_logs, base_dir="backtests"
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_path = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(folder_path, exist_ok=True)

    summary = {
        "strategy": strategy,
        "performance": {"train": train_metrics, "test": test_metrics},
    }
    with open(os.path.join(folder_path, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    if train_logs is not None and not train_logs.empty:
        train_logs.to_csv(os.path.join(folder_path, "train_trades.csv"), index=False)
    if test_logs is not None and not test_logs.empty:
        test_logs.to_csv(os.path.join(folder_path, "test_trades.csv"), index=False)

    print(f"\n📂 All results saved to: {folder_path}")
    return folder_path


# ==============================================================================
#  Final Execution
# ==============================================================================
if __name__ == "__main__":
    print("Loading datasets...")
    df_train = load_and_clean_data(TRAIN_FILE_PATH)
    df_test = load_and_clean_data(TEST_FILE_PATH)

    train_logs, train_metrics = run_dynamic_test_and_plot(
        best_strategy, df_train, "Train Data"
    )
    test_logs, test_metrics = run_dynamic_test_and_plot(
        best_strategy, df_test, "Test Data"
    )

    save_backtest_results(
        strategy=best_strategy,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        train_logs=train_logs,
        test_logs=test_logs,
    )
```

---

This code is a **Vectorized Backtesting Engine** designed to evaluate medium-term trading strategies (based on 288 candles or a specific time window). Instead of directly predicting the price, this system tests a set of conditional rules against historical data.

Below is a complete analysis of this code in 5 main sections:

---

## 1. Strategy Structure and Rules (The Strategy Brain)

At the beginning of the code, the `best_strategy` object is defined. This strategy is **Priority-Based**; meaning if multiple rules generate a signal simultaneously, the one positioned higher (with greater priority) is executed.

- **Rules:** Each rule includes a Take Profit (`tp`), a Stop Loss (`sl`), a position size (`capital_pct`), and a set of conditions (`conditions`).
- **Textual Conditions:** Conditions are written as text strings like `[rsi] IS High`, which are easier for humans to work with but require an interpreter to convert them into executable code.

---

## 2. Data Preparation and Cleaning (`load_and_clean_data`)

This is the first execution stage, which loads the Train and Test datasets:

- **Automatic Format Detection:** The code attempts to read the file using either a Tab or Comma delimiter.
- **NaN Management:** For critical columns (like future prices), it drops rows with missing values. For features, it fills them with `0` to prevent mathematical errors during calculations.
- **Temporal Sorting:** Data is sorted by time to ensure the sequence of trades during the backtest is correct.

---

## 3. Dynamic Logic Interpreter (`apply_dynamic_rule`)

This function is the logical heart of the system. Its job is to convert textual expressions into Boolean Masks in Pandas.

- **Value Categorization:** For expressions like "Very High" or "Bullish", numerical thresholds are defined. For instance, a value greater than **0.8** is considered "Very High".
- **Vectorization:** Instead of checking rows one by one using a loop, it compares the entire DataFrame column with the condition at once, drastically increasing the backtest execution speed.

---

## 4. Trade Processing Engine (`generate_trade_logs`)

This is where actual profit and loss are calculated. The exit logic follows a **Triple-Barrier** method:

1.  **Take Profit Hit (TP):** If the price reaches the target within the specified period (e.g., the next 288 candles).
2.  **Stop Loss Hit (SL):** If the price hits the stop loss earlier.
3.  **Time Exit:** If after 288 candles neither barrier is touched, the trade is closed at the current market price at that moment.

**Profit Calculation ($PnL$):**
The profit percentage for each trade is calculated using the formula below, after which the commission fee (`FEE_PCT`) is deducted:
$$PnL_{pct} = \left( \frac{Exit\_Price - Entry\_Price}{Entry\_Price} \right) \times 100 - Fee$$

---

## 5. Main Execution and Performance Analysis (`run_dynamic_test_and_plot`)

This section manages the backtesting process and generates statistical outputs:

- **Priority Engine:** Using a loop over the rules, it reserves candles. If a rule fires at a specific time, it locks that time for lower-priority rules using a mask (`~triggered`).
- **Key Performance Indicator Calculation:**
  - **Win Rate:** Ratio of profitable trades to total trades.
  - **Max Drawdown:** The largest drop in equity from a previous peak (High Water Mark).
  - **Liquidation Check:** If the cumulative sum of profits and losses reaches **-100%**, the simulation stops and declares bankruptcy.
- **Visualization:** Plots the cumulative equity curve and the Drawdown chart (underwater plot) for risk analysis.

---

### Performance Metrics

| Metric        | Formula                                                                       |
| ------------- | ----------------------------------------------------------------------------- |
| Total PnL     | $\sum_t \text{PnL}_t$ (in dollars)                                            |
| Win Rate      | $\frac{\text{Winning Trades}}{\text{Total Trades}}$                           |
| Profit Factor | $\frac{\sum \text{Wins}}{\sum \text{Losses}}$                                 |
| Max Drawdown  | $\max_t \left(\text{Peak}_t - \text{Equity}_t\right)$                         |
| PnL / \|MDD\| | Risk-adjusted return proxy: `total_pnl_pct` over max(\|max_drawdown_pct\|, ε) |
| Avg Win       | Average TP hit profit                                                         |
| Avg Loss      | Average SL hit loss                                                           |
| Risk/Reward   | $\frac{\text{Avg Win}}{\text{Avg Loss}}$                                      |

---

## 11. Final System Output

Preview of submission format:

```json
{
  "direction": "long",
  "rules_set": [
    {
      "tp": 1.21,
      "sl": 4.82,
      "capital_pct": 40.0,
      "conditions": [
        "[range_compression_20_100] IS Very High",
        "[vol_over_ema20] IS Very High",
        "[vol_over_median20] IS Medium"
      ]
    },
    {
      "tp": 2.65,
      "sl": 2.74,
      "capital_pct": 45.0,
      "conditions": [
        "[amihud_illiquidity_20] IS High",
        "[breakdown_strength_20] IS Medium",
        "[drawdown_from_peak_60] IS Low",
        "[realized_vol_20] IS Very High",
        "[tr_to_atr_14] IS Low"
      ]
    },
    {
      "tp": 3.6,
      "sl": 2.5,
      "capital_pct": 37.5,
      "conditions": [
        "[body_signed_to_tr] IS Neutral Positive",
        "[breakout_strength_20] IS Very Low",
        "[log_range_over_vol_100] IS High",
        "[open_gap_atr_14] IS Weak Positive"
      ]
    }
  ]
}
```

## 12. Configuration File

```python
# sample config.py
# create this file for hyperprameters (not use flags for run time, I'm change this hyperprameters each time)

# TP/SL ranges (percentage)
TP_MIN = 2.0   # 2%   → $20 profit
TP_MAX = 3.0   # 3% → $30 profit
SL_MIN = 0.8   # 0.9% → $8 loss
SL_MAX = 1.5   # 1.5% → $15 loss

# Transaction costs
TRANSACTION_COST_PCT = 0.2  # 0.2% per trade

# Fuzzy states
FUZZY_STATES = [
            if mode == "binary":
                val = "Active (1)" if gene == 1 else "Inactive (0)"
            elif mode == "ternary":
                val = ["Negative (-1)", "Neutral (0)", "Positive (1)"][gene]
            elif mode in ["positive", "sparse_positive"]:
                val = ["Very Low", "Low", "Medium", "High", "Very High"][gene]
            elif mode == "sparse_signed":
                val = ["Strong Negative", "Weak Negative", "Exactly Zero", "Weak Positive", "Strong Positive"][gene]
            else: # signed
                val = ["Extreme Bearish", "Strong Bearish", "Bearish", "Weak Bearish", "Neutral Negative", "Neutral Positive", "Weak Bullish", "Bullish", "Strong Bullish", "Extreme Bullish"][gene]
]
```
