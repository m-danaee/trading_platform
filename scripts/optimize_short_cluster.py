#!/usr/bin/env python3
"""Optimize short.json: pin rules to Phase-2 symbol clusters + balanced overfit prune.

Design (approved):
  1. Assign each rule to its best train cluster (Net PnL on cluster symbols).
  2. Pin survivors with ``symbol is a,b,...`` (OR) for that cluster.
  3. Balanced keep gates with train/test size normalization (~7x row ratio).
  4. Soft per-cluster budget so one cluster cannot dominate.
  5. Write short_optimized.json (original short.json left untouched).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SHORT_PATH = ROOT / "short.json"
CLUSTERS_PATH = ROOT / "outputs" / "symbol_clusters.json"
TRAIN_LOG = ROOT / "student_strategy_evaluation" / \
    "short" / "train" / "student_strategy_trade_logs.csv"
TEST_LOG = ROOT / "student_strategy_evaluation" / \
    "short" / "test" / "student_strategy_trade_logs.csv"
OUT_JSON = ROOT / "short_optimized.json"
OUT_REPORT = ROOT / "student_strategy_evaluation" / \
    "short" / "optimize_cluster_report.json"

# Balanced gates (size-aware). train_2 / test_2 ≈ 7.01 rows.
# Min trades are deliberately lighter than a naive *7 scale: after pinning to a
# cluster, per-rule volume is much smaller than the all-symbol baseline.
ROW_RATIO = 7.0
MIN_TEST_TRADES = 2
MIN_TRAIN_TRADES = 5
# Keep if test_norm >= GAP_FLOOR * train_norm (per 10k bars in that cluster).
GAP_FLOOR = 0.20
# Near-flat allowance on test when PF / win-rate still healthy.
NEAR_FLAT_NET = -3.0
NEAR_FLAT_MIN_WR = 50.0
NEAR_FLAT_MIN_TRAIN_PF = 1.3
MAX_PER_CLUSTER = 15

_SYMBOL_RE = re.compile(
    r"^\s*\[?\s*symbol\s*\]?\s+is\s+(.+?)\s*$",
    flags=re.IGNORECASE,
)


def _is_symbol_condition(text: object) -> bool:
    return isinstance(text, str) and _SYMBOL_RE.match(text) is not None


def _strip_symbol_conditions(conditions: list) -> list[str]:
    return [str(c) for c in conditions if not _is_symbol_condition(c)]


def _pin_cluster(conditions: list, symbols: list[str]) -> list[str]:
    features = _strip_symbol_conditions(conditions)
    # evaluator_v5: comma-separated symbols in one condition are OR-ed
    features.append("symbol is " + ",".join(str(s) for s in symbols))
    return features


def _pf(wins: pd.Series, losses: pd.Series) -> float:
    gp = float(wins.sum()) if len(wins) else 0.0
    gl = float(-losses.sum()) if len(losses) else 0.0
    if gl <= 1e-12:
        return math.inf if gp > 0 else 0.0
    return gp / gl


def _cluster_metrics(logs: pd.DataFrame, rule: int, symbols: set[int]) -> dict:
    g = logs[(logs["Rule_Index"] == rule) & (logs["Symbol"].isin(symbols))]
    if g.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "net_pnl": 0.0,
            "profit_factor": 0.0,
        }
    net = g["Net_PnL"]
    wins = net[net > 0]
    losses = net[net < 0]
    return {
        "trades": int(len(g)),
        "wins": int((net > 0).sum()),
        "losses": int((net < 0).sum()),
        "win_rate": float((net > 0).mean() * 100.0),
        "net_pnl": float(net.sum()),
        "profit_factor": _pf(wins, losses),
    }


def _norm_pnl(net: float, n_bars: int) -> float:
    if n_bars <= 0:
        return 0.0
    return net / n_bars * 10_000.0


def _passes_balanced(train_m: dict, test_m: dict, train_bars: int, test_bars: int) -> tuple[bool, str]:
    if train_m["trades"] < MIN_TRAIN_TRADES:
        return False, "min_train_trades"
    if test_m["trades"] < MIN_TEST_TRADES:
        return False, "min_test_trades"
    if train_m["net_pnl"] <= 0:
        return False, "train_not_positive"

    test_ok = test_m["net_pnl"] > 0
    near_flat = (
        test_m["net_pnl"] >= NEAR_FLAT_NET
        and test_m["win_rate"] >= NEAR_FLAT_MIN_WR
        and train_m["profit_factor"] >= NEAR_FLAT_MIN_TRAIN_PF
    )
    if not (test_ok or near_flat):
        return False, "test_weak"

    train_n = _norm_pnl(train_m["net_pnl"], train_bars)
    test_n = _norm_pnl(test_m["net_pnl"], test_bars)
    if train_n > 0 and test_n < GAP_FLOOR * train_n and not (
        test_ok and test_m["profit_factor"] >= 1.5 and test_m["trades"] >= 4
    ):
        return False, "train_test_gap"

    return True, "keep"


def _score(train_m: dict, test_m: dict, train_bars: int, test_bars: int) -> float:
    """Rank within cluster: prefer robust test, lightly credit train."""
    return (
        _norm_pnl(test_m["net_pnl"], test_bars)
        + 0.2 * _norm_pnl(train_m["net_pnl"], train_bars)
        + 0.05 * min(test_m["profit_factor"], 5.0)
    )


def main() -> None:
    strategy = json.loads(SHORT_PATH.read_text(encoding="utf-8"))
    rules = strategy["rules_set"]
    clusters_raw = json.loads(
        CLUSTERS_PATH.read_text(encoding="utf-8"))["clusters"]
    clusters: dict[int, list[str]] = {
        int(k): [str(s) for s in v] for k, v in clusters_raw.items()
    }
    cluster_syms_int: dict[int, set[int]] = {
        cid: {int(s) for s in syms} for cid, syms in clusters.items()
    }

    # Equal rows per symbol in train_2 / test_2
    train_bars_per_sym = 183_744
    test_bars_per_sym = 26_208
    train_bars_cluster = {
        cid: train_bars_per_sym * len(syms) for cid, syms in clusters.items()
    }
    test_bars_cluster = {
        cid: test_bars_per_sym * len(syms) for cid, syms in clusters.items()
    }

    train_logs = pd.read_csv(TRAIN_LOG)
    test_logs = pd.read_csv(TEST_LOG)
    if "Realized" in train_logs.columns:
        train_logs = train_logs[train_logs["Realized"] == True]  # noqa: E712
    if "Realized" in test_logs.columns:
        test_logs = test_logs[test_logs["Realized"] == True]  # noqa: E712

    candidates: list[dict] = []
    reject_counts: dict[str, int] = {}

    for idx, rule in enumerate(rules, start=1):
        # Best cluster by train Net PnL on cluster symbols
        best_cid = None
        best_train_pnl = float("-inf")
        per_cluster = {}
        for cid, syms in cluster_syms_int.items():
            tr = _cluster_metrics(train_logs, idx, syms)
            te = _cluster_metrics(test_logs, idx, syms)
            per_cluster[cid] = {"train": tr, "test": te}
            if tr["trades"] > 0 and tr["net_pnl"] > best_train_pnl:
                best_train_pnl = tr["net_pnl"]
                best_cid = cid

        if best_cid is None:
            reject_counts["no_train_trades"] = reject_counts.get(
                "no_train_trades", 0) + 1
            continue

        tr = per_cluster[best_cid]["train"]
        te = per_cluster[best_cid]["test"]
        ok, reason = _passes_balanced(
            tr, te, train_bars_cluster[best_cid], test_bars_cluster[best_cid]
        )
        if not ok:
            reject_counts[reason] = reject_counts.get(reason, 0) + 1
            continue

        score = _score(
            tr, te, train_bars_cluster[best_cid], test_bars_cluster[best_cid])
        candidates.append(
            {
                "rule_index": idx,
                "cluster": best_cid,
                "score": score,
                "train": tr,
                "test": te,
                "rule": rule,
            }
        )

    # Soft budget per cluster
    kept: list[dict] = []
    by_cluster: dict[int, list[dict]] = {cid: [] for cid in clusters}
    for c in candidates:
        by_cluster[c["cluster"]].append(c)
    for cid, items in by_cluster.items():
        items.sort(key=lambda x: x["score"], reverse=True)
        selected = items[:MAX_PER_CLUSTER]
        kept.extend(selected)

    kept.sort(key=lambda x: (x["cluster"], -x["score"]))

    optimized_rules = []
    kept_meta = []
    for item in kept:
        cid = item["cluster"]
        pinned = dict(item["rule"])
        pinned["conditions"] = _pin_cluster(
            item["rule"].get("conditions", []), clusters[cid]
        )
        optimized_rules.append(pinned)
        kept_meta.append(
            {
                "original_rule_index": item["rule_index"],
                "cluster": cid,
                "cluster_symbols": clusters[cid],
                "score": item["score"],
                "train_trades": item["train"]["trades"],
                "train_net_pnl": item["train"]["net_pnl"],
                "test_trades": item["test"]["trades"],
                "test_net_pnl": item["test"]["net_pnl"],
                "train_pf": item["train"]["profit_factor"],
                "test_pf": item["test"]["profit_factor"],
            }
        )

    out = {
        "direction": strategy.get("direction", "short"),
        "rules_set": optimized_rules,
        "optimization": {
            "method": "cluster_pin_balanced_v1",
            "source": str(SHORT_PATH.name),
            "clusters": clusters,
            "gates": {
                "min_train_trades": MIN_TRAIN_TRADES,
                "min_test_trades": MIN_TEST_TRADES,
                "gap_floor": GAP_FLOOR,
                "max_per_cluster": MAX_PER_CLUSTER,
                "row_ratio": ROW_RATIO,
            },
        },
    }
    OUT_JSON.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    report = {
        "input_rules": len(rules),
        "candidates_before_budget": len(candidates),
        "kept_rules": len(optimized_rules),
        "kept_per_cluster": {
            str(cid): int(sum(1 for k in kept if k["cluster"] == cid))
            for cid in clusters
        },
        "reject_counts": reject_counts,
        "kept": kept_meta,
        "output": str(OUT_JSON),
    }
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(json.dumps(
        report, indent=2, default=str) + "\n", encoding="utf-8")

    print("OPTIMIZE SHORT (cluster pin + balanced)")
    print(
        f"  input={len(rules)} candidates={len(candidates)} kept={len(optimized_rules)}")
    print(f"  per_cluster={report['kept_per_cluster']}")
    print(f"  rejects={reject_counts}")
    print(f"  wrote {OUT_JSON}")
    print(f"  wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
