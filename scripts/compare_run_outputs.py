#!/usr/bin/env python3
"""
Compare key Phase 2–5 artifacts across two pipeline output directories.

Usage:
  .venv/bin/python scripts/compare_run_outputs.py outputs_2 outputs_3
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _read_strategy_eval(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            split = row.get("split", "")
            if split:
                out[split] = row
    return out


def _phase2_history_tail(path: Path, n: int = 3) -> list[dict]:
    data = _read_json(path)
    if not isinstance(data, list):
        return []
    return data[-n:]


def compare_runs(baseline: Path, candidate: Path) -> int:
    """Print side-by-side metrics; return 0 if both dirs exist."""
    if not baseline.is_dir():
        print(f"Missing baseline directory: {baseline}", file=sys.stderr)
        return 1
    if not candidate.is_dir():
        print(f"Missing candidate directory: {candidate}", file=sys.stderr)
        return 1

    print(f"Baseline:  {baseline}")
    print(f"Candidate: {candidate}")
    print()

    for direction in ("long", "short"):
        print(f"=== {direction.upper()} ===")
        for split in ("train", "validation", "test"):
            b = _read_strategy_eval(
                baseline / "reports" / f"strategy_evaluation_{direction}.csv"
            ).get(split, {})
            c = _read_strategy_eval(
                candidate / "reports" / f"strategy_evaluation_{direction}.csv"
            ).get(split, {})
            if not b and not c:
                continue
            print(
                f"  {split:12} return%  "
                f"{b.get('total_return_pct', 'n/a'):>8} -> {c.get('total_return_pct', 'n/a'):>8}  "
                f"PF {b.get('profit_factor', 'n/a'):>6} -> {c.get('profit_factor', 'n/a'):>6}"
            )

        b_deploy = _read_json(baseline / f"{direction}.json") or {}
        c_deploy = _read_json(candidate / f"{direction}.json") or {}
        print(
            f"  deployment   "
            f"{b_deploy.get('deployment_accepted', 'n/a')} -> "
            f"{c_deploy.get('deployment_accepted', 'n/a')}  "
            f"risk_opt {b_deploy.get('risk_optimized', 'n/a')} -> "
            f"{c_deploy.get('risk_optimized', 'n/a')}"
        )

        b_hist = _phase2_history_tail(
            baseline / f"phase2_{direction}_history.json")
        c_hist = _phase2_history_tail(
            candidate / f"phase2_{direction}_history.json")
        if b_hist or c_hist:
            print("  Phase 2 last gens (pareto_size, cap_hit_frac, uniq_ratio, med_hamming):")
            for label, rows in (("base", b_hist), ("cand", c_hist)):
                if not rows:
                    print(f"    {label}: (no history)")
                    continue
                for row in rows:
                    gen = row.get("generation", "?")
                    print(
                        f"    {label} g{gen}: pareto={row.get('pareto_size')} "
                        f"cap_hit={row.get('sortino_cap_hit_fraction', 0):.2f} "
                        f"uniq={row.get('unique_chromosome_ratio', 0):.2f} "
                        f"ham={row.get('median_pairwise_hamming', 0):.1f}"
                    )
        print()

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="Reference output dir (e.g. outputs_2)")
    parser.add_argument("candidate", type=Path, help="New output dir (e.g. outputs_3)")
    args = parser.parse_args()
    return compare_runs(args.baseline.resolve(), args.candidate.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
