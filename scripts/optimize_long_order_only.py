#!/usr/bin/env python3
"""
Select 100 rules from long.json and optimize JSON rule order only.

Uses evaluator_v5 first-match + capital-managed portfolio semantics via the
fast simulator in optimize_long_rules.py. Does NOT change TP / SL / capital_pct.
Does NOT modify long.json or evaluator_v5.ipynb.

Writes:
  optimized_long/long_100_standalone_order.json
  optimized_long/long_100_greedy_order.json
  optimized_long/long_100_order_optimized.json   (final best)
  optimized_long/order_optimize_report.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from optimize_long_rules import (  # noqa: E402
    INITIAL_CAPITAL,
    RuleOpt,
    collect_features,
    load_dataset,
    load_evaluator_ns,
    long_price_return,
    precompute_arrays,
    score_standalone,
    simulate_portfolio,
)

TARGET_N = 100
DEFAULT_DATA = ROOT / "data" / "train_2_full.csv"
DEFAULT_INPUT = ROOT / "long.json"
DEFAULT_OUT_DIR = ROOT / "optimized_long"


def portfolio_score_key(m: dict[str, float]) -> tuple:
    """Lexicographic preference for comparing portfolios."""
    return (
        float(m["score"]),
        float(m["total_return_pct"]),
        float(m["profit_factor"]),
        -float(m["max_drawdown_pct"]),
        float(m["executed_trades"]),
    )


def build_candidates(
    rules: list[dict],
    match_indices: list[np.ndarray],
    arrays: dict[str, Any],
    min_matches: int = 5,
) -> list[RuleOpt]:
    candidates: list[RuleOpt] = []
    for i, rule in enumerate(rules):
        idxs = match_indices[i]
        if len(idxs) < min_matches:
            continue
        tp = float(rule["tp"])
        sl = float(rule["sl"])
        cap = float(rule.get("capital_pct", 12.5))
        pret = long_price_return(
            idxs,
            tp,
            sl,
            arrays["max_ret"],
            arrays["min_ret"],
            arrays["close_ret"],
            arrays["mbm"],
        )
        st = score_standalone(pret, capital_pct=cap)
        candidates.append(
            RuleOpt(
                orig_idx=i,
                conditions=list(rule["conditions"]),
                tp=tp,
                sl=sl,
                capital_pct=cap,
                match_idx=idxs,
                stand_score=float(st["score"]),
                stand_n=float(st["n"]),
                stand_pf=float(st["pf"]),
                stand_mean=float(st["mean_net"]),
            )
        )
    candidates.sort(key=lambda c: c.stand_score, reverse=True)
    return candidates


def sim(
    ordered: list[int],
    candidates: list[RuleOpt],
    arrays: dict[str, Any],
) -> dict[str, float]:
    match_list = [c.match_idx for c in candidates]
    tps = [c.tp for c in candidates]
    sls = [c.sl for c in candidates]
    caps = [c.capital_pct for c in candidates]
    return simulate_portfolio(ordered, match_list, tps, sls, caps, arrays)


def order_by_standalone(selected: list[int], candidates: list[RuleOpt]) -> list[int]:
    return sorted(selected, key=lambda i: candidates[i].stand_score, reverse=True)


def greedy_select_order_only(
    candidates: list[RuleOpt],
    arrays: dict[str, Any],
    n_keep: int,
) -> list[int]:
    """Forward selection keeping original risk params; JSON order = stand_score desc."""
    remaining = list(range(len(candidates)))
    remaining.sort(key=lambda i: candidates[i].stand_score, reverse=True)
    selected: list[int] = []

    # seed best standalone
    selected.append(remaining.pop(0))
    current = sim(order_by_standalone(selected, candidates), candidates, arrays)
    print(
        f"[greedy] seed orig={candidates[selected[0]].orig_idx} "
        f"stand={candidates[selected[0]].stand_score:.4f} "
        f"ret={current['total_return_pct']:.2f}% "
        f"score={current['score']:.2f}"
    )

    while len(selected) < n_keep and remaining:
        best_i = None
        best_m = None
        best_key = portfolio_score_key(current)

        pool_size = 80 if len(selected) < 15 else 50 if len(selected) < 40 else 35
        pool = remaining[: min(pool_size, len(remaining))]

        for i in pool:
            trial = order_by_standalone(selected + [i], candidates)
            m = sim(trial, candidates, arrays)
            key = portfolio_score_key(m)
            if key > best_key:
                best_key = key
                best_i = i
                best_m = m

        if best_i is None:
            # force-add early; later try full scan then fill
            if len(selected) < max(25, n_keep // 3):
                best_i = remaining[0]
                best_m = sim(
                    order_by_standalone(selected + [best_i], candidates),
                    candidates,
                    arrays,
                )
            else:
                improved = False
                for i in remaining:
                    trial = order_by_standalone(selected + [i], candidates)
                    m = sim(trial, candidates, arrays)
                    key = portfolio_score_key(m)
                    if key > best_key:
                        best_key = key
                        best_i = i
                        best_m = m
                        improved = True
                if not improved:
                    floor = current["score"] * 0.92
                    for i in list(remaining):
                        if len(selected) >= n_keep:
                            break
                        trial = order_by_standalone(selected + [i], candidates)
                        m = sim(trial, candidates, arrays)
                        if (
                            m["score"] >= floor
                            or m["total_return_pct"]
                            >= current["total_return_pct"] * 0.95
                        ):
                            selected.append(i)
                            remaining.remove(i)
                            current = m
                            if len(selected) % 10 == 0:
                                print(
                                    f"[greedy] fill n={len(selected):3d} "
                                    f"ret={m['total_return_pct']:.2f}% "
                                    f"pf={m['profit_factor']:.2f} "
                                    f"trades={m['executed_trades']}"
                                )
                    break

        if best_i is None:
            break

        remaining.remove(best_i)
        selected.append(best_i)
        current = best_m  # type: ignore[assignment]
        remaining.sort(key=lambda i: candidates[i].stand_score, reverse=True)
        if len(selected) % 5 == 0 or len(selected) <= 20:
            print(
                f"[greedy] n={len(selected):3d} ret={current['total_return_pct']:.2f}% "
                f"dd={current['max_drawdown_pct']:.2f}% pf={current['profit_factor']:.2f} "
                f"trades={current['executed_trades']} score={current['score']:.2f}"
            )

    remaining.sort(key=lambda i: candidates[i].stand_score, reverse=True)
    while len(selected) < n_keep and remaining:
        selected.append(remaining.pop(0))

    return order_by_standalone(selected, candidates)


def refine_order_local(
    ordered: list[int],
    candidates: list[RuleOpt],
    arrays: dict[str, Any],
    top_k: int = 40,
    rounds: int = 4,
) -> tuple[list[int], dict[str, float]]:
    """Adjacent swaps + relocate moves to improve first-match order."""
    best = list(ordered)
    best_m = sim(best, candidates, arrays)
    print(
        f"[order] start ret={best_m['total_return_pct']:.2f}% "
        f"dd={best_m['max_drawdown_pct']:.2f}% pf={best_m['profit_factor']:.2f} "
        f"trades={best_m['executed_trades']} score={best_m['score']:.2f}"
    )

    n = len(best)
    for rnd in range(rounds):
        improved = False
        # adjacent bubble within top_k and a bit beyond
        limit = min(n, top_k + 15)
        for pos in range(1, limit):
            trial = list(best)
            trial[pos - 1], trial[pos] = trial[pos], trial[pos - 1]
            m = sim(trial, candidates, arrays)
            if portfolio_score_key(m) > portfolio_score_key(best_m):
                best = trial
                best_m = m
                improved = True

        # relocate: move a lower rule into the top section
        for src in range(1, n):
            if src > top_k + 20:
                break
            for dst in range(0, min(src, top_k)):
                trial = list(best)
                item = trial.pop(src)
                trial.insert(dst, item)
                m = sim(trial, candidates, arrays)
                if portfolio_score_key(m) > portfolio_score_key(best_m):
                    best = trial
                    best_m = m
                    improved = True
                    break

        print(
            f"[order] round {rnd+1}: ret={best_m['total_return_pct']:.2f}% "
            f"dd={best_m['max_drawdown_pct']:.2f}% pf={best_m['profit_factor']:.2f} "
            f"trades={best_m['executed_trades']} score={best_m['score']:.2f} "
            f"{'improved' if improved else 'stable'}"
        )
        if not improved:
            break

    return best, best_m


def write_strategy(
    path: Path,
    direction: str,
    ordered: list[int],
    candidates: list[RuleOpt],
) -> None:
    rules = []
    for i in ordered:
        c = candidates[i]
        rules.append(
            {
                "conditions": c.conditions,
                "tp": float(c.tp),
                "sl": float(c.sl),
                "capital_pct": float(c.capital_pct),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"direction": direction, "rules_set": rules}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[write] {path} ({len(rules)} rules)")


def metrics_brief(m: dict[str, float]) -> dict[str, Any]:
    return {
        "total_return_pct": float(m["total_return_pct"]),
        "max_drawdown_pct": float(m["max_drawdown_pct"]),
        "profit_factor": float(m["profit_factor"]),
        "win_rate": float(m["win_rate"]),
        "executed_trades": int(m["executed_trades"]),
        "raw_signal_count": int(m.get("raw_signal_count", 0)),
        "final_equity": float(m["final_equity"]),
        "account_ruined": bool(m["account_ruined"]),
        "score": float(m["score"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--n-rules", type=int, default=TARGET_N)
    args = ap.parse_args()

    t_all = time.time()
    strategy = json.loads(args.input.read_text(encoding="utf-8"))
    rules = strategy["rules_set"]
    direction = strategy.get("direction", "long")
    print(f"[init] {len(rules)} rules from {args.input}")
    print(f"[init] data={args.data}")
    print("[init] order-only: keep original tp/sl/capital_pct")

    features = collect_features(rules)
    df = load_dataset(args.data, features)

    print("[init] loading evaluator_v5 helpers...")
    ns = load_evaluator_ns()
    build_mask = ns["build_rule_signal_mask"]

    print("[init] precomputing arrays...")
    arrays = precompute_arrays(df)

    print("[init] building signal match indices...")
    t0 = time.time()
    match_indices: list[np.ndarray] = []
    for i, rule in enumerate(rules):
        mask = build_mask(df, rule["conditions"], rule_number=i + 1)
        idxs = np.flatnonzero(np.asarray(mask, dtype=bool)).astype(np.int32)
        match_indices.append(idxs)
        if (i + 1) % 50 == 0 or i == 0 or i + 1 == len(rules):
            print(f"  rule {i+1}/{len(rules)} matches={len(idxs)}")
    print(f"[init] masks done in {time.time()-t0:.1f}s")

    del df
    gc.collect()

    print("[stage1] standalone scores (original params)...")
    candidates = build_candidates(rules, match_indices, arrays)
    print(f"[stage1] {len(candidates)} candidates (min_matches>=5)")
    print("  top 10 standalone:")
    for c in candidates[:10]:
        print(
            f"    orig={c.orig_idx:3d} n={c.stand_n:.0f} mean={c.stand_mean:.3f} "
            f"pf={c.stand_pf:.2f} tp={c.tp} sl={c.sl} cap={c.capital_pct} "
            f"score={c.stand_score:.4f}"
        )

    n_keep = min(args.n_rules, len(candidates))

    # Baseline: original first 100 in file order
    by_orig = {c.orig_idx: k for k, c in enumerate(candidates)}
    base_first = [by_orig[i] for i in range(len(rules)) if i in by_orig][:n_keep]
    base_m = sim(base_first, candidates, arrays)
    print(
        f"[baseline] first{len(base_first)} original order: "
        f"ret={base_m['total_return_pct']:.2f}% dd={base_m['max_drawdown_pct']:.2f}% "
        f"pf={base_m['profit_factor']:.2f} trades={base_m['executed_trades']}"
    )

    all_sel = [by_orig[i] for i in range(len(rules)) if i in by_orig]
    all_m = sim(all_sel, candidates, arrays)
    print(
        f"[baseline] all{len(all_sel)} original order: "
        f"ret={all_m['total_return_pct']:.2f}% dd={all_m['max_drawdown_pct']:.2f}% "
        f"pf={all_m['profit_factor']:.2f} trades={all_m['executed_trades']}"
    )

    # A) Top-N by standalone score only (order = quality)
    stand_sel = list(range(n_keep))  # candidates already sorted by stand_score
    stand_m = sim(stand_sel, candidates, arrays)
    print(
        f"[standalone] top{n_keep}: ret={stand_m['total_return_pct']:.2f}% "
        f"dd={stand_m['max_drawdown_pct']:.2f}% pf={stand_m['profit_factor']:.2f} "
        f"trades={stand_m['executed_trades']} score={stand_m['score']:.2f}"
    )
    path_stand = args.out_dir / "long_100_standalone_order.json"
    write_strategy(path_stand, direction, stand_sel, candidates)

    # B) Greedy portfolio selection, ordered by standalone
    print(f"[stage2] greedy select -> {n_keep} rules (order-only)...")
    greedy_sel = greedy_select_order_only(candidates, arrays, n_keep)
    greedy_m = sim(greedy_sel, candidates, arrays)
    print(
        f"[greedy] n={len(greedy_sel)}: ret={greedy_m['total_return_pct']:.2f}% "
        f"dd={greedy_m['max_drawdown_pct']:.2f}% pf={greedy_m['profit_factor']:.2f} "
        f"trades={greedy_m['executed_trades']} score={greedy_m['score']:.2f}"
    )
    path_greedy = args.out_dir / "long_100_greedy_order.json"
    write_strategy(path_greedy, direction, greedy_sel, candidates)

    # C) Local order refinement on the better of A/B as start, also try the other
    print("[stage3] local order refinement...")
    start_a, m_a = refine_order_local(stand_sel, candidates, arrays)
    start_b, m_b = refine_order_local(greedy_sel, candidates, arrays)

    if portfolio_score_key(m_b) >= portfolio_score_key(m_a):
        final_sel, final_m = start_b, m_b
        final_source = "greedy+order"
    else:
        final_sel, final_m = start_a, m_a
        final_source = "standalone+order"

    # final polish pass
    final_sel, final_m = refine_order_local(
        final_sel, candidates, arrays, top_k=50, rounds=3
    )
    print(
        f"[final] source={final_source} ret={final_m['total_return_pct']:.2f}% "
        f"dd={final_m['max_drawdown_pct']:.2f}% pf={final_m['profit_factor']:.2f} "
        f"trades={final_m['executed_trades']} score={final_m['score']:.2f}"
    )

    path_final = args.out_dir / "long_100_order_optimized.json"
    write_strategy(path_final, direction, final_sel, candidates)

    report = {
        "mode": "order_only",
        "dataset": str(args.data),
        "input": str(args.input),
        "n_input_rules": len(rules),
        "n_candidates": len(candidates),
        "n_output_rules": len(final_sel),
        "note": (
            "Only rule selection + JSON order optimized. "
            "TP/SL/capital_pct kept from long.json. "
            "Higher-quality rules placed earlier for evaluator first-match priority."
        ),
        "baseline_first100": metrics_brief(base_m),
        "baseline_all": metrics_brief(all_m),
        "standalone_top100": metrics_brief(stand_m),
        "greedy_top100": metrics_brief(greedy_m),
        "optimized": metrics_brief(final_m),
        "final_source": final_source,
        "selected_orig_indices": [candidates[i].orig_idx for i in final_sel],
        "outputs": {
            "standalone": str(path_stand),
            "greedy": str(path_greedy),
            "optimized": str(path_final),
        },
        "elapsed_sec": round(time.time() - t_all, 2),
        "initial_capital": INITIAL_CAPITAL,
    }
    report_path = args.out_dir / "order_optimize_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[write] {report_path}")

    print("\n" + "=" * 72)
    print("SUMMARY (train_2_full, order-only, 100 rules)")
    print("=" * 72)
    rows = [
        ("baseline first100", base_m),
        ("baseline all", all_m),
        ("standalone top100", stand_m),
        ("greedy top100", greedy_m),
        ("optimized final", final_m),
    ]
    print(
        f"{'variant':22s} {'return%':>9s} {'maxDD%':>8s} {'PF':>6s} "
        f"{'win%':>7s} {'trades':>7s} {'score':>9s}"
    )
    for name, m in rows:
        print(
            f"{name:22s} {m['total_return_pct']:9.2f} {m['max_drawdown_pct']:8.2f} "
            f"{m['profit_factor']:6.2f} {m['win_rate']:7.2f} "
            f"{m['executed_trades']:7d} {m['score']:9.2f}"
        )
    print(f"\nlong.json was NOT modified.")
    print(f"Outputs in: {args.out_dir}")
    print(f"Elapsed: {time.time()-t_all:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
