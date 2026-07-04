#!/usr/bin/env python3
"""
Combine 2024 and 2025 raw OHLCV data, then build extended train_2.csv
covering 2024-01-01 to 2025-09-30.
"""
from __future__ import annotations
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

        lab_open = np.full(n, np.nan, dtype=np.float64)
        lab_open[:n-1] = o[1:]

        lab_close = np.full(n, np.nan, dtype=np.float64)
        lab_close[:n-TAIL_DROP_ROWS] = c[TAIL_DROP_ROWS:]

        lab_min = np.full(n, np.nan, dtype=np.float64)
        lab_max = np.full(n, np.nan, dtype=np.float64)
        lab_mbm = np.full(n, np.nan, dtype=np.float64)

        if n > TAIL_DROP_ROWS:
            lo_shifted = np.roll(lo, -1)
            lo_shifted[-1] = np.nan
            hi_shifted = np.roll(hi, -1)
            hi_shifted[-1] = np.nan

            rolling_min = pd.Series(lo_shifted).rolling(TAIL_DROP_ROWS, min_periods=TAIL_DROP_ROWS).min().to_numpy()
            rolling_max = pd.Series(hi_shifted).rolling(TAIL_DROP_ROWS, min_periods=TAIL_DROP_ROWS).max().to_numpy()
            lab_min[:n-TAIL_DROP_ROWS] = rolling_min[:n-TAIL_DROP_ROWS]
            lab_max[:n-TAIL_DROP_ROWS] = rolling_max[:n-TAIL_DROP_ROWS]

            hi_s = pd.Series(hi_shifted)
            lo_s = pd.Series(lo_shifted)
            argmax_idx = hi_s.rolling(TAIL_DROP_ROWS, min_periods=TAIL_DROP_ROWS).apply(lambda x: x.argmax(), raw=True).to_numpy()
            argmin_idx = lo_s.rolling(TAIL_DROP_ROWS, min_periods=TAIL_DROP_ROWS).apply(lambda x: x.argmin(), raw=True).to_numpy()
            valid = ~np.isnan(argmax_idx) & ~np.isnan(argmin_idx)
            lab_mbm[:n-TAIL_DROP_ROWS] = np.where(
                valid[:n-TAIL_DROP_ROWS],
                (argmax_idx[:n-TAIL_DROP_ROWS] < argmin_idx[:n-TAIL_DROP_ROWS]).astype(float),
                np.nan
            )

        parts.append(pd.DataFrame({
            "datetime": g["datetime"].to_numpy(),
            "symbol": g["symbol"].to_numpy(),
            "label_open_next": lab_open,
            "label_close_288": lab_close,
            "label_min_288": lab_min,
            "label_max_288": lab_max,
            "label_max_before_min": lab_mbm,
        }))
        print(f"  Symbol {sym}: {n} rows, {h} with valid labels")

    return pd.concat(parts, ignore_index=True).sort_values(["datetime", "symbol"]).reset_index(drop=True)


def main() -> None:
    from safe27_feature_formula_package.safe27_builder import build_safe27

    specs_path = Path("safe27_feature_formula_package/safe27_specs.json")
    with open(specs_path, encoding="utf-8") as f:
        specs = json.load(f)

    # Combine 2024 + 2025 raw OHLCV
    print("Loading 2024 raw OHLCV...")
    df_2024 = pd.read_csv("data/train_1_extended_2024.csv", parse_dates=["datetime"])
    print(f"  2024 shape: {df_2024.shape}")

    print("Loading 2025 raw OHLCV...")
    df_2025 = pd.read_csv("data/train_1.csv", parse_dates=["datetime"])
    print(f"  2025 shape: {df_2025.shape}")

    raw = pd.concat([df_2024, df_2025], ignore_index=True)
    raw = raw.drop_duplicates(subset=["datetime", "symbol"]).sort_values(["symbol", "datetime"]).reset_index(drop=True)
    print(f"  Combined shape: {raw.shape}")
    print(f"  Date range: {raw.datetime.min()} -> {raw.datetime.max()}")

    print("\nComputing 27 safe27 features...")
    features_df = build_safe27(raw, specs)

    print("\nComputing 5 label columns...")
    labels_df = compute_labels(raw)

    print("\nMerging features + labels...")
    merged = features_df.merge(labels_df, on=["datetime", "symbol"], how="left")

    before = len(merged)
    merged = merged.dropna(subset=LABEL_COLS).reset_index(drop=True)
    print(f"  Dropped {before - len(merged)} tail rows (labels NaN)")

    feat_cols = [c for c in merged.columns if c not in ["datetime", "symbol"] + LABEL_COLS]
    merged[feat_cols] = merged[feat_cols].fillna(0)

    final_cols = ["datetime", "symbol"] + LABEL_COLS + feat_cols
    merged = merged[final_cols]

    print(f"\nFinal shape: {merged.shape}")
    merged.to_csv("data/train_2.csv", index=False)
    print(f"\n✓ Saved data/train_2.csv")
    print(f"  Date range: {merged.datetime.min()} -> {merged.datetime.max()}")


if __name__ == "__main__":
    main()
