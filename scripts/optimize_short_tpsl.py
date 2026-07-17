#!/usr/bin/env python3
"""Optimize TP/SL for optimized_short/short_top100.json on train_2_full.csv.

Stage 1: solo TP/SL grid per rule (short evaluator_v5 outcomes).
Stage 2: portfolio accept — keep solo best only if full 100-rule book improves.

Keeps rule order, symbol pins, conditions, and capital_pct fixed.
Does not modify short.json, short_top100.json, or evaluator_v5.ipynb.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from optimize_long_order_only import portfolio_score_key  # noqa: E402
from optimize_long_rules import (  # noqa: E402
    FEE_PCT,
    INITIAL_CAPITAL,
    LEVERAGE,
    MAX_TOTAL_EXPOSURE_PCT,
    MIN_POSITION_NOTIONAL,
    SL_GRID,
    TP_GRID,
    collect_features,
    load_dataset,
    load_evaluator_ns,
    precompute_arrays,
    score_standalone,
)
from optimize_short_top100 import short_price_return  # noqa: E402

DEFAULT_INPUT = ROOT / "optimized_short" / "short_top100.json"
DEFAULT_DATA = ROOT / "data" / "train_2_full.csv"
DEFAULT_OUT_DIR = ROOT / "optimized_short"


def simulate_portfolio_short(
    ordered_rule_indices: list[int],
    rule_match_idx: list[np.ndarray],
    rule_tp: list[float],
    rule_sl: list[float],
    rule_cap: list[float],
    arrays: dict[str, Any],
) -> dict[str, float]:
    """Capital-managed portfolio sim with short TP/SL outcomes (evaluator_v5)."""
    n = arrays["n"]
    assigned = np.full(n, -1, dtype=np.int16)
    for order_pos, ri in enumerate(ordered_rule_indices):
        idxs = rule_match_idx[ri]
        if len(idxs) == 0:
            continue
        free = assigned[idxs] < 0
        take = idxs[free]
        if len(take):
            assigned[take] = order_pos

    entry_rows = np.flatnonzero(assigned >= 0)
    if len(entry_rows) == 0:
        return {
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "executed_trades": 0,
            "raw_signal_count": 0,
            "final_equity": INITIAL_CAPITAL,
            "account_ruined": False,
            "score": -1e18,
        }

    order_pos = assigned[entry_rows]
    ordered_arr = np.asarray(ordered_rule_indices, dtype=np.int32)
    rule_ids = ordered_arr[order_pos]

    etp = arrays["entry_time_priority"][entry_rows]
    rule_order_1based = order_pos + 1
    sort_key = np.lexsort((entry_rows, rule_order_1based, etp))
    entry_rows = entry_rows[sort_key]
    rule_ids = rule_ids[sort_key]

    max_ret = arrays["max_ret"]
    min_ret = arrays["min_ret"]
    close_ret = arrays["close_ret"]
    mbm = arrays["mbm"]
    release_index = arrays["release_index"]
    fee_rate = FEE_PCT / 100.0
    max_exp_rate = MAX_TOTAL_EXPOSURE_PCT / 100.0

    equity = INITIAL_CAPITAL
    peak = INITIAL_CAPITAL
    max_dd = 0.0
    open_positions: list[dict] = []
    open_total_exp = 0.0
    gp = 0.0
    gl = 0.0
    wins = 0
    losses = 0
    executed = 0
    ruined = False

    def release_due(current_index: int) -> None:
        nonlocal equity, peak, max_dd, open_total_exp, gp, gl, wins, losses, ruined
        still = []
        for pos in open_positions:
            if pos["release"] <= current_index:
                equity += pos["net"]
                peak = max(peak, equity)
                dd = (peak - equity) / peak * 100.0 if peak > 0 else 100.0
                max_dd = max(max_dd, dd)
                if pos["net"] > 0:
                    wins += 1
                    gp += pos["net"]
                elif pos["net"] < 0:
                    losses += 1
                    gl += abs(pos["net"])
                open_total_exp -= pos["notional"]
                if equity <= 0:
                    ruined = True
                    equity = 0.0
            else:
                still.append(pos)
        open_positions[:] = still

    for row, rid in zip(entry_rows, rule_ids):
        release_due(int(row))
        if ruined:
            break
        rid = int(rid)
        tp = rule_tp[rid]
        sl = rule_sl[rid]
        cap = rule_cap[rid]
        target = equity * (cap / 100.0) * LEVERAGE
        max_total = equity * max_exp_rate * LEVERAGE
        remaining = max(0.0, max_total - open_total_exp)
        notional = min(target, remaining)
        if notional < MIN_POSITION_NOTIONAL:
            continue

        s_max = max_ret[row]
        s_min = min_ret[row]
        s_close = close_ret[row]
        s_mbm = mbm[row]
        hit_tp = s_min <= -tp
        hit_sl = s_max >= sl
        if hit_tp and hit_sl:
            pret = -sl if s_mbm == 1 else tp
        elif hit_tp:
            pret = tp
        elif hit_sl:
            pret = -sl
        else:
            pret = -s_close

        gross = notional * (pret / 100.0)
        fee = notional * fee_rate
        net = gross - fee
        open_positions.append(
            {
                "release": int(release_index[row]),
                "notional": notional,
                "net": net,
            }
        )
        open_total_exp += notional
        executed += 1

    release_due(n)

    total_ret = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100.0
    if gl <= 0 and gp > 0:
        pf = 99.0
    elif gl <= 0:
        pf = 0.0
    else:
        pf = gp / gl
    wr = (wins / executed * 100.0) if executed else 0.0

    dd_pen = max(max_dd, 0.5)
    score = total_ret * min(pf, 4.0) / (1.0 + dd_pen / 10.0)
    if ruined:
        score = -1e9
    if executed < 50:
        score -= 50.0

    return {
        "total_return_pct": float(total_ret),
        "max_drawdown_pct": float(max_dd),
        "profit_factor": float(pf),
        "win_rate": float(wr),
        "executed_trades": int(executed),
        "raw_signal_count": int(len(entry_rows)),
        "final_equity": float(equity),
        "account_ruined": bool(ruined),
        "score": float(score),
    }


def optimize_rule_tpsl_short(
    match_idx: np.ndarray,
    arrays: dict[str, Any],
    seed_tp: float,
    seed_sl: float,
) -> tuple[float, float, dict]:
    if len(match_idx) == 0:
        st = score_standalone(np.array([], dtype=np.float64))
        return seed_tp, seed_sl, st

    best = None
    best_tp, best_sl = seed_tp, seed_sl
    best_stats: dict = {}

    candidates: list[tuple[float, float]] = []
    for tp in TP_GRID:
        for sl in SL_GRID:
            if tp / sl < 1.0:
                continue
            candidates.append((float(tp), float(sl)))
    candidates.append((float(seed_tp), float(seed_sl)))
    # unique preserve order
    seen: set[tuple[float, float]] = set()
    uniq: list[tuple[float, float]] = []
    for pair in candidates:
        if pair not in seen:
            seen.add(pair)
            uniq.append(pair)

    for tp, sl in uniq:
        pret = short_price_return(
            match_idx,
            tp,
            sl,
            arrays["max_ret"],
            arrays["min_ret"],
            arrays["close_ret"],
            arrays["mbm"],
        )
        st = score_standalone(pret)
        key = (st["score"], st["mean_net"], st["pf"], st["n"])
        if best is None or key > best:
            best = key
            best_tp, best_sl = tp, sl
            best_stats = st

    return best_tp, best_sl, best_stats


def sim(
    tps: list[float],
    sls: list[float],
    caps: list[float],
    match_list: list[np.ndarray],
    arrays: dict[str, Any],
) -> dict[str, float]:
    order = list(range(len(tps)))
    return simulate_portfolio_short(order, match_list, tps, sls, caps, arrays)


def main(
    input_path: Path = DEFAULT_INPUT,
    data_path: Path = DEFAULT_DATA,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> int:
    t0 = time.time()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    strategy = json.loads(input_path.read_text(encoding="utf-8"))
    rules: list[dict] = list(strategy["rules_set"])
    n_rules = len(rules)
    print(f"[load] {input_path.name}: {n_rules} rules")

    features = collect_features(rules)
    print("[load] evaluator_v5...")
    ns = load_evaluator_ns()
    build_mask = ns["build_rule_signal_mask"]

    df = load_dataset(data_path, features)
    arrays = precompute_arrays(df)

    sink = StringIO()
    match_list: list[np.ndarray] = []
    print("[masks] building pinned match indices...")
    for i, rule in enumerate(rules, start=1):
        with redirect_stdout(sink):
            mask = build_mask(df, list(rule["conditions"]), i)
        idxs = np.flatnonzero(np.asarray(mask, dtype=bool)).astype(np.int64)
        match_list.append(idxs)
        if i % 25 == 0 or i == n_rules:
            print(f"[masks] {i}/{n_rules}", flush=True)

    orig_tps = [float(r["tp"]) for r in rules]
    orig_sls = [float(r["sl"]) for r in rules]
    caps = [float(r.get("capital_pct", 12.5)) for r in rules]

    print("[stage1] solo TP/SL grid...")
    solo_tps: list[float] = []
    solo_sls: list[float] = []
    solo_stats: list[dict] = []
    for i in range(n_rules):
        tp, sl, st = optimize_rule_tpsl_short(
            match_list[i], arrays, orig_tps[i], orig_sls[i]
        )
        solo_tps.append(tp)
        solo_sls.append(sl)
        solo_stats.append(st)
        if (i + 1) % 25 == 0 or (i + 1) == n_rules:
            print(f"[stage1] {i+1}/{n_rules}", flush=True)

    print("[stage2] portfolio baseline + accept pass...")
    cur_tps = list(orig_tps)
    cur_sls = list(orig_sls)
    baseline = sim(cur_tps, cur_sls, caps, match_list, arrays)
    current = dict(baseline)
    print(
        f"[stage2] baseline ret={baseline['total_return_pct']:.2f}% "
        f"pf={baseline['profit_factor']:.2f} dd={baseline['max_drawdown_pct']:.2f}% "
        f"trades={baseline['executed_trades']} score={baseline['score']:.2f}"
    )

    per_rule: list[dict[str, Any]] = []
    accepted = 0
    for i in range(n_rules):
        row: dict[str, Any] = {
            "rule_index_1based": i + 1,
            "old_tp": orig_tps[i],
            "old_sl": orig_sls[i],
            "solo_tp": solo_tps[i],
            "solo_sl": solo_sls[i],
            "solo_n": solo_stats[i].get("n"),
            "solo_pf": solo_stats[i].get("pf"),
            "solo_mean_net": solo_stats[i].get("mean_net"),
            "solo_score": solo_stats[i].get("score"),
        }
        if solo_tps[i] == cur_tps[i] and solo_sls[i] == cur_sls[i]:
            row["decision"] = "unchanged_solo_equals_current"
            row["new_tp"] = cur_tps[i]
            row["new_sl"] = cur_sls[i]
            per_rule.append(row)
            continue

        trial_tps = list(cur_tps)
        trial_sls = list(cur_sls)
        trial_tps[i] = solo_tps[i]
        trial_sls[i] = solo_sls[i]
        trial_m = sim(trial_tps, trial_sls, caps, match_list, arrays)
        if portfolio_score_key(trial_m) > portfolio_score_key(current):
            cur_tps[i] = solo_tps[i]
            cur_sls[i] = solo_sls[i]
            current = trial_m
            accepted += 1
            row["decision"] = "accepted"
            row["portfolio_after"] = {
                "total_return_pct": current["total_return_pct"],
                "profit_factor": current["profit_factor"],
                "max_drawdown_pct": current["max_drawdown_pct"],
                "executed_trades": current["executed_trades"],
                "score": current["score"],
            }
        else:
            row["decision"] = "rejected"
        row["new_tp"] = cur_tps[i]
        row["new_sl"] = cur_sls[i]
        per_rule.append(row)

        if (i + 1) % 25 == 0 or (i + 1) == n_rules:
            print(
                f"[stage2] {i+1}/{n_rules} accepted={accepted} "
                f"ret={current['total_return_pct']:.2f}%",
                flush=True,
            )

    out_rules = []
    for i, rule in enumerate(rules):
        r = copy.deepcopy(rule)
        r["tp"] = float(cur_tps[i])
        r["sl"] = float(cur_sls[i])
        out_rules.append(r)

    out_dir.mkdir(parents=True, exist_ok=True)
    deploy = {"direction": "short", "rules_set": out_rules}
    deploy_path = out_dir / "short_top100_tpsl.json"
    deploy_path.write_text(json.dumps(
        deploy, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input": str(input_path),
        "dataset": str(data_path),
        "dataset_rows": int(len(df)),
        "tp_grid": list(TP_GRID),
        "sl_grid": list(SL_GRID),
        "baseline": baseline,
        "final": current,
        "counts": {
            "n_rules": n_rules,
            "accepted_tpsl_changes": accepted,
            "unchanged": sum(1 for r in per_rule if r["decision"].startswith("unchanged")),
            "rejected": sum(1 for r in per_rule if r["decision"] == "rejected"),
        },
        "elapsed_sec": time.time() - t0,
        "rules": per_rule,
    }
    report_path = out_dir / "tpsl_optimize_report.json"
    report_path.write_text(json.dumps(
        report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        f"[done] accepted={accepted}/{n_rules} | "
        f"ret {baseline['total_return_pct']:.2f}% -> {current['total_return_pct']:.2f}% | "
        f"{time.time()-t0:.1f}s"
    )
    print(f"[done] wrote {deploy_path}")
    print(f"[done] wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
