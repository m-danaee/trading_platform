#!/usr/bin/env python3
"""
Combine 2024 and 2025 raw OHLCV data, then build extended train_2.csv
covering 2024-01-01 to 2025-09-30.

Uses the same forward-window label semantics as build_train2_test2.compute_labels
(label_min/max over [t+1, t+288], not inclusive of bar t).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_train2_test2 import LABEL_COLS, compute_labels  # noqa: E402


def main() -> None:
    from safe27_feature_formula_package.safe27_builder import build_safe27

    specs_path = Path("safe27_feature_formula_package/safe27_specs.json")
    with open(specs_path, encoding="utf-8") as f:
        specs = json.load(f)

    print("Loading 2024 raw OHLCV...")
    df_2024 = pd.read_csv("data/train_1_extended_2024.csv", parse_dates=["datetime"])
    print(f"  2024 shape: {df_2024.shape}")

    print("Loading 2025 raw OHLCV...")
    df_2025 = pd.read_csv("data/train_1.csv", parse_dates=["datetime"])
    print(f"  2025 shape: {df_2025.shape}")

    raw = pd.concat([df_2024, df_2025], ignore_index=True)
    raw = (
        raw.drop_duplicates(subset=["datetime", "symbol"])
        .sort_values(["symbol", "datetime"])
        .reset_index(drop=True)
    )
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

    feat_cols = [
        c for c in merged.columns if c not in ["datetime", "symbol"] + LABEL_COLS
    ]
    merged[feat_cols] = merged[feat_cols].fillna(0)

    final_cols = ["datetime", "symbol"] + LABEL_COLS + feat_cols
    merged = merged[final_cols]

    print(f"\nFinal shape: {merged.shape}")
    merged.to_csv("data/train_2.csv", index=False)
    print("\n✓ Saved data/train_2.csv")
    print(f"  Date range: {merged.datetime.min()} -> {merged.datetime.max()}")


if __name__ == "__main__":
    main()
