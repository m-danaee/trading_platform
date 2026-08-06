#!/usr/bin/env python3
"""Diagnose split-aware context mask coverage in enriched tapes.

Run on Colab (or wherever the data lives):
    python scripts/diagnose_context_mask.py

This reports both long and short eligibility for the training, validation,
validation-fitness, and validation-selection frames, including per-symbol
counts. Zero coverage in train or validation-fitness is a Phase 2 blocker.
"""
import sys
import os

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from gpu_fuzzy_trader import config as _cfg


def _coverage(frame: pd.DataFrame, direction: str) -> dict[str, object]:
    """Return eligible-row counts for one direction and split frame."""
    perm = _cfg.context_permission_column(direction)
    trig = _cfg.context_trigger_column(direction)
    missing = [column for column in (perm, trig) if column not in frame.columns]
    if missing:
        return {
            "eligible_rows": None,
            "total_rows": int(len(frame)),
            "percent": None,
            "by_symbol": {},
            "missing": missing,
        }

    mask = (frame[perm].to_numpy() == 1) & (frame[trig].to_numpy() == 1)
    by_symbol: dict[str, int] = {}
    if "symbol" in frame.columns:
        for symbol, group in frame.groupby("symbol", sort=True, observed=False):
            group_mask = (
                (group[perm].to_numpy() == 1)
                & (group[trig].to_numpy() == 1)
            )
            by_symbol[str(symbol)] = int(group_mask.sum())
    else:
        by_symbol["<all>"] = int(mask.sum())
    eligible = int(mask.sum())
    return {
        "eligible_rows": eligible,
        "total_rows": int(len(frame)),
        "percent": eligible / max(len(frame), 1) * 100,
        "by_symbol": by_symbol,
        "missing": [],
    }


def _print_split_coverage(
    split_name: str,
    frame: pd.DataFrame | None,
) -> dict[str, dict[str, object]]:
    """Print both direction and per-symbol coverage for one split."""
    if frame is None:
        print(f"\n[{split_name}] unavailable (split parquet not found)")
        return {}

    print(f"\n[{split_name}] rows: {len(frame):,}")
    report: dict[str, dict[str, object]] = {}
    for direction in ("long", "short"):
        stats = _coverage(frame, direction)
        report[direction] = stats
        if stats["eligible_rows"] is None:
            print(
                f"  {direction.upper():5s}: unavailable; missing "
                f"{stats['missing']}"
            )
            continue
        by_symbol = ", ".join(
            f"{symbol}={count:,}"
            for symbol, count in stats["by_symbol"].items()
        ) or "<none>"
        print(
            f"  {direction.upper():5s}: "
            f"{stats['percent']:.2f}% "
            f"({stats['eligible_rows']:,} / {stats['total_rows']:,}) "
            f"by_symbol: {by_symbol}"
        )
        if stats["percent"] < 1.0:
            print("         *** CRITICAL: <1% coverage — inspect Phase 2 floors")
        elif stats["percent"] < 3.0:
            print("         *** WARNING: <3% coverage — Phase 2 may be sparse")
    return report


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
    print("  → Coverage cannot be established until the tape is enriched.")

print(f"\n  Context columns present: {present}")

print(
    "  Contract: "
    f"LWC lookback={_cfg.LWC_PULLBACK_LOOKBACK} bars; "
    "threshold quantiles=60th/60th/40th"
)

frames: dict[str, pd.DataFrame | None] = {"train": df}
for split_name, split_path in (
    ("validation", _cfg.VALIDATION_30_PATH),
    ("validation_fitness", _cfg.VALIDATION_FITNESS_PATH),
    ("validation_selection", _cfg.VALIDATION_SELECTION_PATH),
):
    try:
        frames[split_name] = pd.read_parquet(split_path)
    except (FileNotFoundError, OSError, ImportError, ValueError):
        frames[split_name] = None

coverage_reports = {
    split_name: _print_split_coverage(split_name, frame)
    for split_name, frame in frames.items()
}

print()

# State distribution
for col in ("hwc_state", "mwc_state", "lwc_state"):
    if col in df.columns:
        vc = df[col].value_counts().sort_index()
        codes = {-1: "Bearish", 0: "Range", 1: "Bullish", 2: "Noisy"}
        parts = [f"{codes.get(k, k)}={v:,} ({v/len(df)*100:.1f}%)" for k, v in vc.items()]
        print(f"  {col}: {' | '.join(parts)}")

print()
print("Recommendation:")
failures: list[str] = []
for split_name in ("train", "validation_fitness"):
    split_report = coverage_reports.get(split_name, {})
    for direction in ("long", "short"):
        stats = split_report.get(direction, {})
        eligible = stats.get("eligible_rows")
        if eligible is None:
            failures.append(f"{split_name}/{direction}: missing context columns")
        elif int(eligible) == 0:
            failures.append(f"{split_name}/{direction}: zero eligible rows")

if failures:
    print("  [ACTION REQUIRED] Context coverage preflight would stop Phase 2:")
    for failure in failures:
        print(f"    - {failure}")
    print("  Re-run trend_context enrichment with the active 24-bar contract,")
    print("  then rebuild all split and Phase 1/Phase 2 artifacts.")
else:
    low_coverage = [
        f"{split_name}/{direction}"
        for split_name, split_report in coverage_reports.items()
        for direction, stats in split_report.items()
        if stats.get("percent") is not None
        and float(stats["percent"]) < 1.0
    ]
    if low_coverage:
        print(
            "  [WARNING] Nonzero context exists, but coverage is below 1% "
            "in: " + ", ".join(low_coverage)
        )
    print(
        "  Train and validation-fitness have nonzero eligible rows in both "
        "directions; validation-selection is reported above for diagnosis."
    )
