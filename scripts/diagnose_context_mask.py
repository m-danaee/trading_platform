#!/usr/bin/env python3
"""Diagnose context mask coverage in the enriched training CSV.

Run on Colab (or wherever the data lives):
    python scripts/diagnose_context_mask.py

This shows what fraction of bars are eligible for trading under the
mandatory context gate (tf_permission_long AND lwc_pullback_reversal_long).
If the percentage is <3%, zero-trade collapse in Phase 2 is expected.
"""
import sys
import os

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from gpu_fuzzy_trader import config as _cfg

path = _cfg.TRAIN_CSV_PATH
print(f"Loading: {path}")
try:
    df = pd.read_csv(path, low_memory=False)
except FileNotFoundError:
    print(f"  File not found. Check TRAIN_CSV_PATH or DATA_ROOT env vars.")
    sys.exit(1)

print(f"  Rows: {len(df):,}  Symbols: {df['symbol'].nunique() if 'symbol' in df.columns else 'N/A'}")

ctx_cols = list(_cfg.CONTEXT_COLUMNS)
present = [c for c in ctx_cols if c in df.columns]
missing = [c for c in ctx_cols if c not in df.columns]

if missing:
    print(f"\n[WARNING] Missing context columns: {missing}")
    print("  → The context mask will be all-True (no filtering). This is fine if data has no context.")
    sys.exit(0)

print(f"\n  Context columns present: {present}")
print()

for direction in ("long", "short"):
    perm = _cfg.context_permission_column(direction)
    trig = _cfg.context_trigger_column(direction)
    mask = (df[perm].to_numpy() == 1) & (df[trig].to_numpy() == 1)
    pct = mask.sum() / max(len(mask), 1) * 100
    print(f"[{direction.upper():5s}] context mask: {pct:.2f}%  ({mask.sum():,} / {len(mask):,} rows)")
    if pct < 1.0:
        print(f"         *** CRITICAL: <1% coverage — Phase 2 will find zero trades ***")
    elif pct < 3.0:
        print(f"         *** WARNING: <3% coverage — Phase 2 will likely collapse ***")

    # Per-symbol breakdown
    if "symbol" in df.columns:
        for sym, grp in df.groupby("symbol"):
            m = (grp[perm].to_numpy() == 1) & (grp[trig].to_numpy() == 1)
            print(f"           {sym}: {m.mean()*100:.2f}%  ({m.sum():,} / {len(m):,})")

print()

# State distribution
for col in ("hwc_state", "mwc_state", "lwc_state"):
    if col in df.columns:
        vc = df[col].value_counts().sort_index()
        codes = {-1: "Bearish", 0: "Range", 1: "Bullish", 2: "Noisy"}
        parts = [f"{codes.get(k, k)}={v:,} ({v/len(df)*100:.1f}%)" for k, v in vc.items()]
        print(f"  {col}: {' | '.join(parts)}")

# Long permission (hwc+mwc both bullish)
if "tf_permission_long" in df.columns:
    perm_pct = (df["tf_permission_long"] == 1).mean() * 100
    print(f"\n  tf_permission_long (hwcB+mwcB): {perm_pct:.2f}%")

if "lwc_pullback_reversal_long" in df.columns:
    trig_pct = (df["lwc_pullback_reversal_long"] == 1).mean() * 100
    print(f"  lwc_pullback_reversal_long:      {trig_pct:.2f}%")

print()
print("Recommendation:")
long_pct = ((df["tf_permission_long"] == 1) & (df["lwc_pullback_reversal_long"] == 1)).mean() * 100 if "tf_permission_long" in df.columns else 0
if long_pct < 3.0:
    print("  Context coverage is too low for Phase 2 evolution.")
    print("  Options:")
    print("  1. Check if the enriched CSV was generated correctly (re-run trend_context enrichment)")
    print("  2. Increase LWC_PULLBACK_LOOKBACK in config.py (currently:", _cfg.LWC_PULLBACK_LOOKBACK, ")")
    print("  3. Lower MIN_TRADE_POOL_FLOOR and MIN_TRADE_SUPPORT to match expected trade density")
    expected_per_chrom = long_pct / 100 * 29600  # approx bars per island
    print(f"  4. Expected trades/chromosome at {long_pct:.1f}% coverage with 29600 rows: ~{expected_per_chrom:.0f}")
    print(f"     (current MIN_TRADE_POOL_FLOOR = {_cfg.MIN_TRADE_POOL_FLOOR})")
