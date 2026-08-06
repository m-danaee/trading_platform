#!/usr/bin/env python3
"""Diagnose split-aware context mask coverage in enriched tapes.

Run on Colab (or wherever the data lives):
    python scripts/diagnose_context_mask.py

This reports both long and short eligibility for the training, validation,
validation-fitness, and validation-selection frames, including per-symbol
counts. Zero coverage in train or validation-fitness is a Phase 2 blocker.
"""
from gpu_fuzzy_trader.phases.phase2_rule_pool import (
    _derive_val_sample_seed,
    sample_df_for_phase2,
)
from gpu_fuzzy_trader.phases.phase2_island_scheduler import _derive_island_seed
from gpu_fuzzy_trader.context_diagnostics import (
    context_coverage_for_direction,
    context_floor_failures,
)
import sys
import os

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from gpu_fuzzy_trader import config as _cfg


def _coverage(frame: pd.DataFrame, direction: str) -> dict[str, object]:
    """Return eligible-row counts for one direction and split frame."""
    stats = context_coverage_for_direction(frame, direction)
    stats["percent"] = stats["coverage_pct"]
    stats["missing"] = stats["missing_columns"]
    return stats


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
            f"permission={stats['permission_rows']:,} "
            f"trigger={stats['trigger_rows']:,} "
            f"permission_only={stats['permission_only_rows']:,} "
            f"trigger_only={stats['trigger_only_rows']:,} "
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


def _print_island_sample_diagnostics(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame | None,
) -> dict[str, object]:
    """Print coverage on deterministic singleton-island train/val windows."""
    if validation_frame is None:
        print("[island samples] validation-fitness frame unavailable")
        return {}

    forbidden_ranges: list[tuple[int, int]] = []
    reference_rows = len(
        sample_df_for_phase2(
            train_frame,
            random_state=_cfg.PHASE2_SEED,
            forbidden_ranges=forbidden_ranges,
        )
    )
    if "symbol" in train_frame.columns:
        symbols = sorted(
            {str(value) for value in train_frame["symbol"].dropna().unique()}
        )
    else:
        symbols = []
    singleton_mode = bool(
        getattr(_cfg, "PHASE2_SYMBOL_SPECIALISTS_ENABLED", False)
        or getattr(_cfg, "PHASE2_ONE_SYMBOL_ISLANDS", False)
    )
    scopes = (
        [(str(index), [symbol]) for index, symbol in enumerate(symbols)]
        if symbols and singleton_mode
        else [("universe", symbols)]
    )
    report: dict[str, object] = {
        "reference_rows": int(reference_rows),
        "islands": {},
        "failures": [],
    }
    print(
        f"[island samples] mode={'singleton' if singleton_mode else 'universe'} "
        f"reference_rows={reference_rows:,}"
    )
    for island_id, scope_symbols in scopes:
        if scope_symbols:
            train_scope = train_frame[
                train_frame["symbol"].astype(str).isin(scope_symbols)
            ]
            val_scope = validation_frame[
                validation_frame["symbol"].astype(str).isin(scope_symbols)
            ]
        else:
            train_scope = train_frame
            val_scope = validation_frame
        island_report: dict[str, object] = {
            "symbols": list(scope_symbols),
            "directions": {},
        }
        for direction in ("long", "short"):
            island_seed = _derive_island_seed(
                _cfg.PHASE2_SEED, f"{direction}_{island_id}",
            )
            if island_seed is None:
                island_seed = _cfg.PHASE2_SEED
            train_sample = sample_df_for_phase2(
                train_scope, random_state=island_seed,
            )
            val_sample = sample_df_for_phase2(
                val_scope,
                random_state=_derive_val_sample_seed(island_seed),
            )
            hp = _cfg.resolve_island_hyperparams(
                "cluster",
                len(train_sample),
                max(1, reference_rows),
                n_symbols=max(1, len(scope_symbols)),
            )
            train_stats = _coverage(train_sample, direction)
            val_stats = _coverage(val_sample, direction)
            direction_report = {
                "train": train_stats,
                "validation_fitness": val_stats,
                "floors": {
                    "min_trade_support": int(hp.min_trade_support),
                    "min_trade_pool_floor": int(hp.min_trade_pool_floor),
                    "validation_trade_floor": int(hp.val_trade_floor),
                },
            }
            island_report["directions"][direction] = direction_report
            if (
                train_stats["eligible_rows"] is None
                or val_stats["eligible_rows"] is None
            ):
                print(
                    f"  {direction.upper():5s} island={island_id}: "
                    "unavailable; missing context columns"
                )
                continue
            train_failures = context_floor_failures(
                train_stats,
                support_floor=hp.min_trade_support,
                pool_floor=hp.min_trade_pool_floor,
            )
            val_failures = context_floor_failures(
                val_stats,
                validation_floor=hp.val_trade_floor,
            )
            for split_name, split_failures in (
                ("train", train_failures),
                ("validation_fitness", val_failures),
            ):
                for reason in split_failures:
                    report["failures"].append(
                        f"{direction}/{island_id}/{split_name}: {reason}"
                    )
            print(
                f"  {direction.upper():5s} island={island_id} "
                f"train={train_stats['eligible_rows']}/{train_stats['total_rows']} "
                f"({train_stats['percent']:.2f}%) "
                f"val={val_stats['eligible_rows']}/{val_stats['total_rows']} "
                f"({val_stats['percent']:.2f}%) "
                f"floors={direction_report['floors']}"
            )
        report["islands"][island_id] = island_report
    return report


island_sample_report = _print_island_sample_diagnostics(
    df,
    frames.get("validation_fitness"),
)

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
        else:
            if split_name == "train":
                failures.extend(
                    f"{split_name}/{direction}: {reason}"
                    for reason in context_floor_failures(
                        stats,
                        support_floor=_cfg.effective_min_trade_support(
                            len(frames[split_name])
                        ),
                        pool_floor=_cfg.effective_min_trade_pool_floor(
                            len(frames[split_name])
                        ),
                    )
                )
            else:
                failures.extend(
                    f"{split_name}/{direction}: {reason}"
                    for reason in context_floor_failures(
                        stats,
                        validation_floor=_cfg.effective_pool_min_val_trades(
                            len(frames[split_name])
                        ),
                    )
                )
for failure in island_sample_report.get("failures", []):
    failures.append(f"island sample: {failure}")

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
        "  Context rows clear the configured upper-bound floors on the "
        "reported windows; fuzzy rule matching can still reduce trades."
    )
