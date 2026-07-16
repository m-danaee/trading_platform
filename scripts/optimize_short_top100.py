#!/usr/bin/env python3
"""Optimize short.json → top 100 pinned/reordered rules on train_2.csv.

Discovery: solo per-rule matches (symbol filters stripped), score symbols with
Net_PnL × PF (fee-adjusted price returns, evaluator_v5 short TP/SL semantics).
Gate B: net_pnl > 0, PF >= 1.2, trades >= 5.
Pin survivors with ``symbol is a,b,...``, keep top 100 by aggregate score.

Does not modify short.json or evaluator_v5.ipynb.
"""

from __future__ import annotations

import copy
import json
import math
import re
import sys
import time
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from optimize_long_rules import (  # noqa: E402
    FEE_PCT,
    collect_features,
    load_dataset,
    load_evaluator_ns,
    precompute_arrays,
)

DEFAULT_INPUT = ROOT / "short.json"
DEFAULT_DATA = ROOT / "data" / "train_2.csv"
DEFAULT_OUT_DIR = ROOT / "optimized_short"
TARGET_N = 100
MIN_PF = 1.2
MIN_TRADES = 5
PF_CAP = 10.0

SYMBOL_CONDITION_RE = re.compile(
    r"(?i)(?:\[?\s*symbol\s*\]?)\s+is\s+",
)


def compute_profit_factor(net_pnls: list[float] | np.ndarray) -> float:
    arr = np.asarray(net_pnls, dtype=float)
    if arr.size == 0:
        return 0.0
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    gross_profit = float(wins.sum()) if wins.size else 0.0
    gross_loss = float(-losses.sum()) if losses.size else 0.0
    if gross_loss <= 1e-12:
        return math.inf if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def rule_score(net_pnl: float, profit_factor: float, *, pf_cap: float = PF_CAP) -> float:
    pf = float(profit_factor)
    if math.isinf(pf):
        pf = pf_cap
    elif math.isnan(pf):
        pf = 0.0
    else:
        pf = min(pf, pf_cap)
    return float(net_pnl) * pf


def symbol_passes_gate(
    net_pnl: float,
    profit_factor: float,
    trades: int,
    *,
    min_pf: float = MIN_PF,
    min_trades: int = MIN_TRADES,
) -> bool:
    if trades < min_trades:
        return False
    if net_pnl <= 0:
        return False
    pf = float(profit_factor)
    if math.isinf(pf):
        pf = PF_CAP
    if math.isnan(pf) or pf < min_pf:
        return False
    return True


def strip_symbol_conditions(conditions: list[str]) -> list[str]:
    kept: list[str] = []
    for cond in conditions:
        text = str(cond).strip()
        if SYMBOL_CONDITION_RE.search(text):
            continue
        kept.append(text)
    return kept


def pin_rule(rule: dict, symbols: list[str | int]) -> dict:
    pinned = copy.deepcopy(rule)
    conditions = list(pinned.get("conditions") or [])
    feature_only = strip_symbol_conditions(conditions)
    sym_sorted = sorted({int(float(s)) for s in symbols})
    if not sym_sorted:
        raise ValueError("pin_rule requires at least one symbol")
    pin = "symbol is " + ",".join(str(s) for s in sym_sorted)
    feature_only.append(pin)
    pinned["conditions"] = feature_only
    return pinned


def short_price_return(
    idx: np.ndarray,
    tp: float,
    sl: float,
    max_ret: np.ndarray,
    min_ret: np.ndarray,
    close_ret: np.ndarray,
    mbm: np.ndarray,
) -> np.ndarray:
    """Vectorized short TP/SL/time outcome (matches evaluator_v5)."""
    s_max = max_ret[idx]
    s_min = min_ret[idx]
    s_close = close_ret[idx]
    s_mbm = mbm[idx]
    hit_tp = s_min <= -tp
    hit_sl = s_max >= sl
    both = hit_tp & hit_sl
    out = np.where(
        both,
        np.where(s_mbm == 1, -sl, tp),
        np.where(hit_tp, tp, np.where(hit_sl, -sl, -s_close)),
    )
    return out.astype(np.float64)


def score_rule_by_symbol(
    match_idx: np.ndarray,
    net_pct: np.ndarray,
    symbols: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Aggregate fee-adjusted net outcomes per symbol."""
    out: dict[str, dict[str, float]] = {}
    if len(match_idx) == 0:
        return out
    syms = symbols[match_idx]
    for sym in np.unique(syms):
        mask = syms == sym
        nets = net_pct[mask]
        trades = int(nets.size)
        net_pnl = float(nets.sum())
        pf = compute_profit_factor(nets)
        out[str(sym)] = {
            "net_pnl": net_pnl,
            "profit_factor": float(pf) if not math.isinf(pf) else float("inf"),
            "trades": trades,
            "score": rule_score(net_pnl, pf),
        }
    return out


def main(
    input_path: Path = DEFAULT_INPUT,
    data_path: Path = DEFAULT_DATA,
    out_dir: Path = DEFAULT_OUT_DIR,
    target_n: int = TARGET_N,
) -> int:
    t_start = time.time()
    if not input_path.exists():
        raise FileNotFoundError(f"Missing strategy file: {input_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Missing dataset: {data_path}")

    strategy = json.loads(input_path.read_text(encoding="utf-8"))
    direction = str(strategy.get("direction", "short")).lower()
    rules: list[dict] = list(strategy.get("rules_set") or [])
    if not rules:
        raise ValueError("short.json has empty rules_set")
    print(
        f"[load] {input_path.name}: {len(rules)} rules | direction={direction}")

    features = collect_features(rules)
    print(f"[load] {len(features)} features referenced")

    print("[load] evaluator_v5 namespace...")
    ns = load_evaluator_ns()
    build_mask = ns["build_rule_signal_mask"]

    df = load_dataset(data_path, features)
    arrays = precompute_arrays(df)
    symbols = arrays["symbols"]

    # Quiet mask builds (apply_dynamic_rule can be chatty in some paths)
    sink = StringIO()
    per_rule_rows: list[dict[str, Any]] = []
    survivors: list[dict[str, Any]] = []

    print(f"[score] solo discovery on {len(df):,} rows...")
    for i, rule in enumerate(rules):
        orig_1based = i + 1
        discovery = copy.deepcopy(rule)
        discovery["conditions"] = strip_symbol_conditions(
            list(rule.get("conditions") or []))
        tp = float(rule["tp"])
        sl = float(rule["sl"])

        try:
            with redirect_stdout(sink):
                mask = build_mask(df, discovery["conditions"], orig_1based)
            idxs = np.flatnonzero(np.asarray(
                mask, dtype=bool)).astype(np.int64)
            pret = short_price_return(
                idxs,
                tp,
                sl,
                arrays["max_ret"],
                arrays["min_ret"],
                arrays["close_ret"],
                arrays["mbm"],
            )
            net = pret - FEE_PCT
            by_sym = score_rule_by_symbol(idxs, net, symbols)
            kept_syms: list[str] = []
            kept_sym_metrics: dict[str, dict[str, float]] = {}
            for sym, m in by_sym.items():
                pf_raw = m["profit_factor"]
                if symbol_passes_gate(m["net_pnl"], pf_raw, int(m["trades"])):
                    kept_syms.append(sym)
                    kept_sym_metrics[sym] = m
            if not kept_syms:
                per_rule_rows.append(
                    {
                        "original_rule_index": orig_1based,
                        "status": "dropped_no_symbol",
                        "kept_symbols": [],
                        "per_symbol": by_sym,
                        "aggregate_score": None,
                        "kept_rank": None,
                    }
                )
                continue

            agg = float(sum(kept_sym_metrics[s]["score"] for s in kept_syms))
            pinned = pin_rule(rule, kept_syms)
            survivors.append(
                {
                    "original_rule_index": orig_1based,
                    "aggregate_score": agg,
                    "kept_symbols": sorted(kept_syms, key=lambda x: int(float(x))),
                    "per_symbol": kept_sym_metrics,
                    "pinned_rule": pinned,
                }
            )
            per_rule_rows.append(
                {
                    "original_rule_index": orig_1based,
                    "status": "survivor_candidate",
                    "kept_symbols": sorted(kept_syms, key=lambda x: int(float(x))),
                    "per_symbol": {
                        s: {
                            **kept_sym_metrics[s],
                            "profit_factor": (
                                kept_sym_metrics[s]["profit_factor"]
                                if not math.isinf(kept_sym_metrics[s]["profit_factor"])
                                else "inf"
                            ),
                        }
                        for s in kept_syms
                    },
                    "aggregate_score": agg,
                    "kept_rank": None,
                }
            )
        except Exception as exc:  # noqa: BLE001 — continue other rules
            per_rule_rows.append(
                {
                    "original_rule_index": orig_1based,
                    "status": "error",
                    "error": str(exc),
                    "kept_symbols": [],
                    "aggregate_score": None,
                    "kept_rank": None,
                }
            )
            print(f"[warn] rule {orig_1based} failed: {exc}")

        if (i + 1) % 25 == 0 or (i + 1) == len(rules):
            print(
                f"[score] {i+1}/{len(rules)} | survivors so far={len(survivors)}",
                flush=True,
            )

    survivors.sort(key=lambda r: r["aggregate_score"], reverse=True)
    selected = survivors[:target_n]
    for rank, row in enumerate(selected, start=1):
        row["kept_rank"] = rank
        # update report row
        for pr in per_rule_rows:
            if pr["original_rule_index"] == row["original_rule_index"]:
                pr["status"] = "kept"
                pr["kept_rank"] = rank
                break

    # mark non-selected survivors
    selected_ids = {r["original_rule_index"] for r in selected}
    for pr in per_rule_rows:
        if pr.get("status") == "survivor_candidate" and pr["original_rule_index"] not in selected_ids:
            pr["status"] = "dropped_rank"

    out_dir.mkdir(parents=True, exist_ok=True)
    deployable = {
        "direction": "short",
        "rules_set": [r["pinned_rule"] for r in selected],
    }
    deploy_path = out_dir / "short_top100.json"
    deploy_path.write_text(json.dumps(
        deployable, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    elapsed = time.time() - t_start
    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input": str(input_path),
        "dataset": str(data_path),
        "dataset_rows": int(len(df)),
        "scoring": {
            "formula": "sum_over_kept_symbols(net_pnl * min(pf, 10))",
            "fee_pct": FEE_PCT,
            "symbol_gate": {
                "net_pnl_gt": 0,
                "min_profit_factor": MIN_PF,
                "min_trades": MIN_TRADES,
            },
            "discovery": "strip_symbol_conditions_then_solo_match",
        },
        "counts": {
            "evaluated": len(rules),
            "dropped_no_symbol": sum(1 for r in per_rule_rows if r.get("status") == "dropped_no_symbol"),
            "dropped_rank": sum(1 for r in per_rule_rows if r.get("status") == "dropped_rank"),
            "errors": sum(1 for r in per_rule_rows if r.get("status") == "error"),
            "survivor_candidates": len(survivors),
            "kept_after_rank": len(selected),
            "target_n": target_n,
        },
        "elapsed_sec": elapsed,
        "rules": per_rule_rows,
    }
    # JSON-safe inf
    report_path = out_dir / "optimize_report.json"

    def _safe(obj: Any) -> Any:
        if isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
            return "inf" if math.isinf(obj) and obj > 0 else ("-inf" if math.isinf(obj) else None)
        if isinstance(obj, dict):
            return {k: _safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_safe(v) for v in obj]
        return obj

    report_path.write_text(json.dumps(
        _safe(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    readme = out_dir / "README.md"
    readme.write_text(
        "# Optimized short (top 100)\n\n"
        "Generated by `scripts/optimize_short_top100.py` from `short.json` on "
        "`data/train_2.csv`.\n\n"
        "- Score: Net_PnL × PF (PF capped at 10), solo discovery with symbol filters stripped\n"
        "- Symbol gate: Net_PnL > 0, PF ≥ 1.2, trades ≥ 5\n"
        "- Output order: best aggregate score first (evaluator first-match priority)\n\n"
        "Spec: `docs/superpowers/specs/2026-07-16-short-top100-reorder-design.md`\n",
        encoding="utf-8",
    )

    print(
        f"[done] kept={len(selected)} / candidates={len(survivors)} / "
        f"evaluated={len(rules)} in {elapsed:.1f}s"
    )
    print(f"[done] wrote {deploy_path}")
    print(f"[done] wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
