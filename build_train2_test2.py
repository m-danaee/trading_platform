#!/usr/bin/env python3
"""
Build train_2.csv and test_2.csv with:
- 27 safe27 features (already normalized per-symbol)
- 5 label columns (raw forward prices)
Structure matches train.csv / test.csv
"""
from __future__ import annotations
from safe27_feature_formula_package.safe27_builder import build_safe27
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

TAIL_DROP_ROWS = 288
LABEL_COLS = [
    "label_open_next",
    "label_close_288",
    "label_min_288",
    "label_max_288",
    "label_max_before_min",
]


def compute_labels(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 5 label columns per symbol.

    - label_open_next:      open[t+1]
    - label_close_288:      close[t+288]
    - label_min_288:        min(low[t+1 : t+289])
    - label_max_288:        max(high[t+1 : t+289])
    - label_max_before_min: 1 if argmax(high) occurs before argmin(low) in window, else 0

    Rows where the forward window extends beyond the data are set to NaN
    (the loader will drop them via TAIL_DROP_ROWS).
    """
    raw = raw.sort_values(["symbol", "datetime"]).reset_index(drop=True)
    parts = []
    for sym, g in raw.groupby("symbol", sort=True):
        g = g.reset_index(drop=True)
        n = len(g)
        h = n - TAIL_DROP_ROWS

        o = g["open"].to_numpy()
        hi = g["high"].to_numpy()
        lo = g["low"].to_numpy()
        c = g["close"].to_numpy()

        # label_open_next: open[t+1]
        lab_open = np.full(n, np.nan, dtype=np.float64)
        lab_open[:n-1] = o[1:]

        # label_close_288: close[t+288]
        lab_close = np.full(n, np.nan, dtype=np.float64)
        lab_close[:n-TAIL_DROP_ROWS] = c[TAIL_DROP_ROWS:]

        # label_min_288: min(low[t+1:t+289])
        lab_min = np.full(n, np.nan, dtype=np.float64)
        if n > TAIL_DROP_ROWS:
            # Use rolling min on low[1:] with window=288
            lo_shifted = np.roll(lo, -1)
            lo_shifted[-1] = np.nan
            rolling_min = pd.Series(lo_shifted).rolling(
                TAIL_DROP_ROWS, min_periods=TAIL_DROP_ROWS).min().to_numpy()
            lab_min[:n-TAIL_DROP_ROWS] = rolling_min[:n-TAIL_DROP_ROWS]

        # label_max_288: max(high[t+1:t+289])
        lab_max = np.full(n, np.nan, dtype=np.float64)
        if n > TAIL_DROP_ROWS:
            hi_shifted = np.roll(hi, -1)
            hi_shifted[-1] = np.nan
            rolling_max = pd.Series(hi_shifted).rolling(
                TAIL_DROP_ROWS, min_periods=TAIL_DROP_ROWS).max().to_numpy()
            lab_max[:n-TAIL_DROP_ROWS] = rolling_max[:n-TAIL_DROP_ROWS]

        # label_max_before_min: 1 if argmax(high) < argmin(low) in window
        lab_mbm = np.full(n, np.nan, dtype=np.float64)
        if n > TAIL_DROP_ROWS:
            # Recompute shifted arrays if not already done (they should be from label_min_288/max blocks)
            if 'hi_shifted' not in locals():
                hi_shifted = np.roll(hi, -1)
                hi_shifted[-1] = np.nan
            if 'lo_shifted' not in locals():
                lo_shifted = np.roll(lo, -1)
                lo_shifted[-1] = np.nan

            hi_s = pd.Series(hi_shifted)
            lo_s = pd.Series(lo_shifted)
            argmax_idx = hi_s.rolling(TAIL_DROP_ROWS, min_periods=TAIL_DROP_ROWS).apply(
                lambda x: x.argmax(), raw=True
            ).to_numpy()
            argmin_idx = lo_s.rolling(TAIL_DROP_ROWS, min_periods=TAIL_DROP_ROWS).apply(
                lambda x: x.argmin(), raw=True
            ).to_numpy()
            valid = ~np.isnan(argmax_idx) & ~np.isnan(argmin_idx)
            lab_mbm[:n-TAIL_DROP_ROWS] = np.where(
                valid[:n-TAIL_DROP_ROWS],
                (argmax_idx[:n-TAIL_DROP_ROWS] <
                 argmin_idx[:n-TAIL_DROP_ROWS]).astype(float),
                np.nan
            )

        sym_labels = pd.DataFrame({
            "datetime": g["datetime"].to_numpy(),
            "symbol": g["symbol"].to_numpy(),
            "label_open_next": lab_open,
            "label_close_288": lab_close,
            "label_min_288": lab_min,
            "label_max_288": lab_max,
            "label_max_before_min": lab_mbm,
        })
        parts.append(sym_labels)
        print(f"  Symbol {sym}: {n} rows, {h} with valid labels")

    return pd.concat(parts, ignore_index=True).sort_values(["datetime", "symbol"]).reset_index(drop=True)


def build_dataset(raw_path: str, out_path: str, specs: dict) -> None:
    print(f"\n{'='*70}")
    print(f"Building {out_path} from {raw_path}")
    print(f"{'='*70}")

    print(f"Loading raw OHLCV...")
    raw = pd.read_csv(raw_path, parse_dates=["datetime"])
    print(f"  Shape: {raw.shape}")
    print(f"  Date range: {raw.datetime.min()} -> {raw.datetime.max()}")

    print(f"\nComputing 27 safe27 features...")
    features_df = build_safe27(raw, specs)

    print(f"\nComputing 5 label columns...")
    labels_df = compute_labels(raw)

    print(f"\nMerging features + labels...")
    # Align on (datetime, symbol) - merge full labels_df which includes datetime/symbol
    merged = features_df.merge(
        labels_df, on=["datetime", "symbol"], how="left")

    # Drop rows where any label is NaN (tail rows)
    before = len(merged)
    merged = merged.dropna(subset=LABEL_COLS).reset_index(drop=True)
    print(f"  Dropped {before - len(merged)} tail rows (labels NaN)")

    # Fill NaN in feature columns with 0 (matches Data_Loader behavior)
    feat_cols = [c for c in merged.columns if c not in [
        "datetime", "symbol"] + LABEL_COLS]
    merged[feat_cols] = merged[feat_cols].fillna(0)

    # Final column order: datetime, symbol, labels..., features...
    final_cols = ["datetime", "symbol"] + LABEL_COLS + feat_cols
    merged = merged[final_cols]

    print(f"\nFinal shape: {merged.shape}")
    print(f"Columns ({len(merged.columns)}):")
    print(f"  meta:   datetime, symbol")
    print(f"  labels: {LABEL_COLS}")
    print(f"  feats:  {len(feat_cols)} features")

    merged.to_csv(out_path, index=False)
    print(f"\n✓ Saved {out_path}")
    print(f"  Date range: {merged.datetime.min()} -> {merged.datetime.max()}")


def main() -> None:
    specs_path = Path("safe27_feature_formula_package/safe27_specs.json")
    with open(specs_path, encoding="utf-8") as f:
        specs = json.load(f)

    build_dataset("data/train_1.csv", "data/train_2.csv", specs)
    build_dataset("data/test_1.csv", "data/test_2.csv", specs)

    # Verification
    print(f"\n{'='*70}")
    print("Verification")
    print(f"{'='*70}")
    for path in ["data/train_2.csv", "data/test_2.csv"]:
        df = pd.read_csv(path)
        print(f"\n{path}:")
        print(f"  Shape: {df.shape}")
        print(f"  Date range: {df.datetime.min()} -> {df.datetime.max()}")
        print(f"  Symbols: {sorted(df.symbol.unique())}")
        for col in LABEL_COLS:
            print(f"  {col:25s}: [{df[col].min():.4f}, {df[col].max():.4f}]")

        # Compare train/test feature parity
        other = "data/test_2.csv" if "train" in path else "data/train_2.csv"
        other_df = pd.read_csv(other, nrows=0)
        print(
            f"  Column parity with {other}: {set(df.columns) == set(other_df.columns)}")
        feat_cols = [c for c in df.columns if c not in [
            "datetime", "symbol"] + LABEL_COLS]
        print(f"  Feature count: {len(feat_cols)} (expected 27)")


if __name__ == "__main__":
    main()
