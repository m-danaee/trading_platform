#!/usr/bin/env python3
"""
Optimize long.json (default 100 rules) on a labeled OHLCV/features CSV.

Uses evaluator_v5.ipynb semantics (first-match rule priority, capital-managed
exposure, TP/SL/label outcomes, fees). Does not modify evaluator_v5.ipynb.

Stages:
  1. Per-rule TP/SL grid search (standalone expectancy)
  2. Rank rules; greedy forward selection of N with portfolio sim
  3. Capital_pct grid (uniform + quality-tiered)
  4. Local reorder / drop-add refinement
  5. Write long.json + optimization report
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Constants aligned with evaluator_v5.ipynb
# ---------------------------------------------------------------------------
FEE_PCT = 0.20
MAX_HOLD_CANDLES = 288
INITIAL_CAPITAL = 1000.0
LEVERAGE = 1.0
MAX_TOTAL_EXPOSURE_PCT = 100.0
MIN_POSITION_NOTIONAL = 1.0
TARGET_N_RULES = 100

TP_GRID = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0)
SL_GRID = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)
CAPITAL_GRID = (5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0)

DEFAULT_DATA = ROOT / "data" / "full_dataset_safe27_2024_now.csv"
DEFAULT_INPUT = ROOT / "long.json"
DEFAULT_OUTPUT = ROOT / "long.json"
DEFAULT_REPORT = ROOT / "outputs" / "long_optimize_full_report.json"


def load_evaluator_ns() -> dict:
    """Load evaluator_v5 code cells (skip student input / run cells)."""
    nb_path = ROOT / "evaluator_v5.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    ns: dict = {"__name__": "__main__"}
    skip_markers = (
        "Student Strategy Input",
        "Run Train and Test Evaluation",
        "student_strategy =",
    )
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if any(m in src for m in skip_markers):
            continue
        exec(compile(src, str(nb_path), "exec"), ns)
    return ns


def collect_features(rules: list[dict]) -> list[str]:
    feats: set[str] = set()
    for r in rules:
        for c in r["conditions"]:
            if " IS " not in c:
                continue
            feat = c.split(" IS ", 1)[0].strip().strip("[]")
            if feat.lower() != "symbol":
                feats.add(feat)
    return sorted(feats)


def load_dataset(path: Path, features: list[str]) -> pd.DataFrame:
    label_cols = [
        "label_open_next",
        "label_close_288",
        "label_min_288",
        "label_max_288",
        "label_max_before_min",
    ]
    usecols = ["datetime", "symbol"] + label_cols + features
    print(f"[load] reading {path} ({len(usecols)} cols)...")
    t0 = time.time()
    df = pd.read_csv(path, usecols=usecols)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df["datetime"] = df["datetime"].dt.tz_localize(None)
    df = df.sort_values(["datetime", "symbol"]).reset_index(drop=True)
    df = df.dropna(subset=label_cols).reset_index(drop=True)
    for c in features:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.float32)
    for c in label_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
    df["_symbol_bar_index"] = df.groupby("symbol", sort=False).cumcount().astype(np.int32)
    print(f"[load] {len(df):,} rows in {time.time()-t0:.1f}s")
    return df


def precompute_arrays(df: pd.DataFrame) -> dict[str, Any]:
    entry = df["label_open_next"].to_numpy(dtype=np.float64)
    max_ret = (df["label_max_288"].to_numpy(dtype=np.float64) - entry) / entry * 100.0
    min_ret = (df["label_min_288"].to_numpy(dtype=np.float64) - entry) / entry * 100.0
    close_ret = (df["label_close_288"].to_numpy(dtype=np.float64) - entry) / entry * 100.0
    mbm = df["label_max_before_min"].to_numpy(dtype=np.int8)
    symbols = df["symbol"].astype(str).to_numpy()
    symbol_bar = df["_symbol_bar_index"].to_numpy(dtype=np.int32)
    datetimes = df["datetime"].to_numpy()
    entry_time_priority = pd.factorize(pd.Series(datetimes), sort=False)[0].astype(np.int64)

    # release indices (conservative MAX_HOLD_CANDLES)
    n = len(df)
    release_index = np.full(n, n, dtype=np.int64)
    row_index = np.arange(n, dtype=np.int64)
    sym_codes, _ = pd.factorize(symbols, sort=False)
    for sc in np.unique(sym_codes):
        mask = sym_codes == sc
        rows = row_index[mask]
        bars = symbol_bar[mask].astype(np.int64)
        order = np.argsort(bars, kind="mergesort")
        rows_s = rows[order]
        bars_s = bars[order]
        target = bars_s + MAX_HOLD_CANDLES
        pos = np.searchsorted(bars_s, target, side="left")
        valid = pos < len(rows_s)
        if np.any(valid):
            release_index[rows_s[valid]] = rows_s[pos[valid]]

    return {
        "max_ret": max_ret,
        "min_ret": min_ret,
        "close_ret": close_ret,
        "mbm": mbm,
        "symbols": symbols,
        "release_index": release_index,
        "entry_time_priority": entry_time_priority,
        "n": n,
    }


def long_price_return(
    idx: np.ndarray,
    tp: float,
    sl: float,
    max_ret: np.ndarray,
    min_ret: np.ndarray,
    close_ret: np.ndarray,
    mbm: np.ndarray,
) -> np.ndarray:
    """Vectorized long TP/SL/time outcome (matches evaluator_v5)."""
    s_max = max_ret[idx]
    s_min = min_ret[idx]
    s_close = close_ret[idx]
    s_mbm = mbm[idx]
    hit_tp = s_max >= tp
    hit_sl = s_min <= -sl
    both = hit_tp & hit_sl
    out = np.where(
        both,
        np.where(s_mbm == 1, tp, -sl),
        np.where(hit_tp, tp, np.where(hit_sl, -sl, s_close)),
    )
    return out.astype(np.float64)


def score_standalone(
    price_ret_pct: np.ndarray,
    capital_pct: float = 12.5,
) -> dict[str, float]:
    """Fee-aware standalone score assuming sequential non-overlapping trades.

    Not identical to portfolio sim, but good ranking signal for rule quality.
    """
    if len(price_ret_pct) == 0:
        return {
            "n": 0,
            "mean_net": 0.0,
            "pf": 0.0,
            "wr": 0.0,
            "total_net_R": 0.0,
            "score": -1e18,
        }
    fee = FEE_PCT  # percent of notional
    net = price_ret_pct - fee
    wins = net[net > 0]
    losses = net[net < 0]
    gp = float(wins.sum()) if len(wins) else 0.0
    gl = float(-losses.sum()) if len(losses) else 0.0
    if gl <= 0 and gp > 0:
        pf = 99.0
    elif gl <= 0:
        pf = 0.0
    else:
        pf = gp / gl
    wr = float((net > 0).mean() * 100.0)
    mean_net = float(net.mean())
    total = float(net.sum())
    # Prefer positive expectancy, PF, and enough trades
    n = len(net)
    trade_bonus = min(n, 200) / 200.0
    score = mean_net * (1.0 + 0.5 * min(pf, 5.0)) * (0.5 + 0.5 * trade_bonus)
    if n < 15:
        score -= 5.0
    if mean_net <= 0:
        score -= 10.0
    return {
        "n": float(n),
        "mean_net": mean_net,
        "pf": float(pf),
        "wr": wr,
        "total_net_R": total,
        "score": float(score),
        "capital_pct": capital_pct,
    }


def simulate_portfolio(
    ordered_rule_indices: list[int],
    rule_match_idx: list[np.ndarray],
    rule_tp: list[float],
    rule_sl: list[float],
    rule_cap: list[float],
    arrays: dict[str, Any],
) -> dict[str, float]:
    """Capital-managed portfolio sim matching evaluator_v5 first-match + exposure."""
    n = arrays["n"]
    assigned = np.full(n, -1, dtype=np.int16)
    for order_pos, ri in enumerate(ordered_rule_indices):
        idxs = rule_match_idx[ri]
        if len(idxs) == 0:
            continue
        free = assigned[idxs] < 0
        take = idxs[free]
        if len(take):
            assigned[take] = order_pos  # position in ordered list (0-based rule order)

    # Build entries
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
    # Map order_pos -> original rule id for tp/sl/cap
    ordered_arr = np.asarray(ordered_rule_indices, dtype=np.int32)
    rule_ids = ordered_arr[order_pos]

    # sort: entry_time_priority, rule_index (1-based order), row idx
    etp = arrays["entry_time_priority"][entry_rows]
    rule_order_1based = order_pos + 1
    sort_key = np.lexsort((entry_rows, rule_order_1based, etp))
    entry_rows = entry_rows[sort_key]
    rule_ids = rule_ids[sort_key]

    max_ret = arrays["max_ret"]
    min_ret = arrays["min_ret"]
    close_ret = arrays["close_ret"]
    mbm = arrays["mbm"]
    symbols = arrays["symbols"]
    release_index = arrays["release_index"]
    fee_rate = FEE_PCT / 100.0
    max_exp_rate = MAX_TOTAL_EXPOSURE_PCT / 100.0

    equity = INITIAL_CAPITAL
    peak = INITIAL_CAPITAL
    max_dd = 0.0
    open_positions: list[dict] = []
    open_total_exp = 0.0
    # symbol exposure not used for sizing capacity beyond total (evaluator only tracks it)
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

        # single outcome
        s_max = max_ret[row]
        s_min = min_ret[row]
        s_close = close_ret[row]
        s_mbm = mbm[row]
        hit_tp = s_max >= tp
        hit_sl = s_min <= -sl
        if hit_tp and hit_sl:
            pret = tp if s_mbm == 1 else -sl
        elif hit_tp:
            pret = tp
        elif hit_sl:
            pret = -sl
        else:
            pret = s_close

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

    # composite: return with drawdown penalty and PF preference
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


@dataclass
class RuleOpt:
    orig_idx: int
    conditions: list[str]
    tp: float
    sl: float
    capital_pct: float
    match_idx: np.ndarray
    stand_score: float
    stand_n: float
    stand_pf: float
    stand_mean: float


def optimize_rule_tpsl(
    match_idx: np.ndarray,
    arrays: dict[str, Any],
    seed_tp: float,
    seed_sl: float,
) -> tuple[float, float, dict]:
    best = None
    best_tp, best_sl = seed_tp, seed_sl
    for tp in TP_GRID:
        for sl in SL_GRID:
            if tp / sl < 1.0:
                continue
            pret = long_price_return(
                match_idx,
                tp,
                sl,
                arrays["max_ret"],
                arrays["min_ret"],
                arrays["close_ret"],
                arrays["mbm"],
            )
            st = score_standalone(pret)
            # also try seed if not on grid
            key = (st["score"], st["mean_net"], st["pf"], st["n"])
            if best is None or key > best:
                best = key
                best_tp, best_sl = tp, sl
                best_stats = st
    # include original seed
    pret = long_price_return(
        match_idx,
        seed_tp,
        seed_sl,
        arrays["max_ret"],
        arrays["min_ret"],
        arrays["close_ret"],
        arrays["mbm"],
    )
    st = score_standalone(pret)
    key = (st["score"], st["mean_net"], st["pf"], st["n"])
    if best is None or key > best:
        best_tp, best_sl = seed_tp, seed_sl
        best_stats = st
    return best_tp, best_sl, best_stats


def greedy_select(
    candidates: list[RuleOpt],
    arrays: dict[str, Any],
    n_keep: int,
) -> list[int]:
    """Forward selection: add rule that most improves portfolio score.

    Order among selected is by stand_score (best rules first for first-match).
    """
    remaining = list(range(len(candidates)))
    selected: list[int] = []

    # seed with best standalone rule
    remaining.sort(key=lambda i: candidates[i].stand_score, reverse=True)
    selected.append(remaining.pop(0))
    print(f"[greedy] seed rule orig={candidates[selected[0]].orig_idx} "
          f"score={candidates[selected[0]].stand_score:.4f}")

    def ordered_selected() -> list[int]:
        # JSON order = descending stand_score (important first-match priority)
        return sorted(selected, key=lambda i: candidates[i].stand_score, reverse=True)

    match_list = [c.match_idx for c in candidates]
    tps = [c.tp for c in candidates]
    sls = [c.sl for c in candidates]
    caps = [c.capital_pct for c in candidates]

    current_metrics = simulate_portfolio(
        ordered_selected(), match_list, tps, sls, caps, arrays
    )
    print(f"[greedy] after seed: ret={current_metrics['total_return_pct']:.2f}% "
          f"pf={current_metrics['profit_factor']:.2f} "
          f"trades={current_metrics['executed_trades']} "
          f"score={current_metrics['score']:.2f}")

    while len(selected) < n_keep and remaining:
        best_i = None
        best_m = None
        best_score = current_metrics["score"]
        # only try top-K remaining by standalone to save time, expand if needed
        pool = remaining[: min(40, len(remaining))]
        if len(selected) < 10:
            pool = remaining[: min(80, len(remaining))]

        for i in pool:
            trial = selected + [i]
            trial_ordered = sorted(
                trial, key=lambda j: candidates[j].stand_score, reverse=True
            )
            m = simulate_portfolio(trial_ordered, match_list, tps, sls, caps, arrays)
            if m["score"] > best_score + 1e-9:
                best_score = m["score"]
                best_i = i
                best_m = m

        if best_i is None:
            # no improvement in pool — force add best remaining standalone if early,
            # else stop and fill with next best standalone that doesn't crash score badly
            if len(selected) < max(20, n_keep // 3):
                # force add next standalone
                best_i = remaining[0]
                trial = selected + [best_i]
                trial_ordered = sorted(
                    trial, key=lambda j: candidates[j].stand_score, reverse=True
                )
                best_m = simulate_portfolio(
                    trial_ordered, match_list, tps, sls, caps, arrays
                )
            else:
                # try full remaining scan once every few adds, or fill rest
                improved = False
                for i in remaining:
                    trial = selected + [i]
                    trial_ordered = sorted(
                        trial, key=lambda j: candidates[j].stand_score, reverse=True
                    )
                    m = simulate_portfolio(
                        trial_ordered, match_list, tps, sls, caps, arrays
                    )
                    if m["score"] > best_score + 1e-9:
                        best_score = m["score"]
                        best_i = i
                        best_m = m
                        improved = True
                if not improved:
                    # fill remaining slots with top standalone that keep score >= 90% of best
                    floor = current_metrics["score"] * 0.9
                    for i in list(remaining):
                        if len(selected) >= n_keep:
                            break
                        trial = selected + [i]
                        trial_ordered = sorted(
                            trial, key=lambda j: candidates[j].stand_score, reverse=True
                        )
                        m = simulate_portfolio(
                            trial_ordered, match_list, tps, sls, caps, arrays
                        )
                        if m["score"] >= floor or m["total_return_pct"] >= current_metrics[
                            "total_return_pct"
                        ] * 0.95:
                            selected.append(i)
                            remaining.remove(i)
                            current_metrics = m
                            if len(selected) % 10 == 0:
                                print(
                                    f"[greedy] fill {len(selected)}: "
                                    f"ret={m['total_return_pct']:.2f}% "
                                    f"pf={m['profit_factor']:.2f} "
                                    f"trades={m['executed_trades']}"
                                )
                    break

        if best_i is None:
            break
        remaining.remove(best_i)
        selected.append(best_i)
        current_metrics = best_m
        # re-sort remaining by stand_score for next pool
        remaining.sort(key=lambda i: candidates[i].stand_score, reverse=True)
        if len(selected) % 5 == 0 or len(selected) <= 15:
            print(
                f"[greedy] n={len(selected):3d} ret={current_metrics['total_return_pct']:.2f}% "
                f"dd={current_metrics['max_drawdown_pct']:.2f}% "
                f"pf={current_metrics['profit_factor']:.2f} "
                f"trades={current_metrics['executed_trades']} "
                f"score={current_metrics['score']:.2f}"
            )

    # if still short, pad with best remaining standalone
    remaining.sort(key=lambda i: candidates[i].stand_score, reverse=True)
    while len(selected) < n_keep and remaining:
        selected.append(remaining.pop(0))

    return ordered_selected()


def optimize_capital(
    selected: list[int],
    candidates: list[RuleOpt],
    arrays: dict[str, Any],
) -> tuple[list[float], dict]:
    """Try uniform capital levels and quality-tiered capital."""
    match_list = [c.match_idx for c in candidates]
    tps = [c.tp for c in candidates]
    sls = [c.sl for c in candidates]
    n = len(candidates)
    best_caps = [candidates[i].capital_pct for i in range(n)]
    best_m = None
    best_score = -1e18

    scores = np.array([candidates[i].stand_score for i in selected], dtype=np.float64)
    # rank within selected (0=best)
    order = np.argsort(-scores)
    rank = np.empty(len(selected), dtype=np.int32)
    rank[order] = np.arange(len(selected))

    for cap in CAPITAL_GRID:
        caps = [12.5] * n
        for i in selected:
            caps[i] = cap
        m = simulate_portfolio(selected, match_list, tps, sls, caps, arrays)
        if m["score"] > best_score:
            best_score = m["score"]
            best_m = m
            best_caps = caps
            print(f"[capital] uniform {cap}: ret={m['total_return_pct']:.2f}% "
                  f"pf={m['profit_factor']:.2f} trades={m['executed_trades']}")

    # tiered: top third higher capital, bottom lower
    for hi, mid, lo in (
        (20.0, 12.5, 7.5),
        (25.0, 12.5, 5.0),
        (15.0, 12.5, 10.0),
        (20.0, 15.0, 10.0),
        (12.5, 10.0, 7.5),
    ):
        caps = [12.5] * n
        n_sel = len(selected)
        for j, i in enumerate(selected):
            r = int(rank[j])
            if r < n_sel // 3:
                caps[i] = hi
            elif r < 2 * n_sel // 3:
                caps[i] = mid
            else:
                caps[i] = lo
        m = simulate_portfolio(selected, match_list, tps, sls, caps, arrays)
        if m["score"] > best_score:
            best_score = m["score"]
            best_m = m
            best_caps = caps
            print(
                f"[capital] tiered {hi}/{mid}/{lo}: ret={m['total_return_pct']:.2f}% "
                f"pf={m['profit_factor']:.2f} trades={m['executed_trades']}"
            )

    return best_caps, best_m


def prune_redundant(
    selected: list[int],
    candidates: list[RuleOpt],
    arrays: dict[str, Any],
    caps: list[float],
) -> list[int]:
    """Drop rules that reduce score when present; refill to TARGET_N_RULES later if needed."""
    match_list = [c.match_idx for c in candidates]
    tps = [c.tp for c in candidates]
    sls = [c.sl for c in candidates]
    current = list(selected)
    base = simulate_portfolio(current, match_list, tps, sls, caps, arrays)
    improved = True
    rounds = 0
    while improved and rounds < 3:
        improved = False
        rounds += 1
        # try drop worst-ranked first
        ordered = sorted(current, key=lambda i: candidates[i].stand_score)
        for i in ordered:
            if len(current) <= max(30, TARGET_N_RULES // 2):
                break
            trial = [j for j in current if j != i]
            m = simulate_portfolio(trial, match_list, tps, sls, caps, arrays)
            if m["score"] > base["score"] + 1e-6:
                current = trial
                base = m
                improved = True
                print(
                    f"[prune] drop orig={candidates[i].orig_idx} -> "
                    f"ret={m['total_return_pct']:.2f}% score={m['score']:.2f} n={len(current)}"
                )
                break
    return current


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--n-rules", type=int, default=TARGET_N_RULES)
    ap.add_argument("--baseline-only", action="store_true")
    args = ap.parse_args()

    t_all = time.time()
    strategy = json.loads(args.input.read_text(encoding="utf-8"))
    rules = strategy["rules_set"]
    direction = strategy.get("direction", "long")
    print(f"[init] {len(rules)} rules from {args.input}")

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
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  rule {i+1}/{len(rules)} matches={len(idxs)}")
    print(f"[init] masks done in {time.time()-t0:.1f}s")

    # free df feature columns if possible — keep only what's needed for masks done
    del df
    gc.collect()

    # Stage 1: per-rule TP/SL
    print("[stage1] per-rule TP/SL optimization...")
    candidates: list[RuleOpt] = []
    for i, rule in enumerate(rules):
        idxs = match_indices[i]
        if len(idxs) < 5:
            print(f"  skip rule {i}: only {len(idxs)} matches")
            continue
        tp, sl, st = optimize_rule_tpsl(
            idxs, arrays, float(rule["tp"]), float(rule["sl"])
        )
        candidates.append(
            RuleOpt(
                orig_idx=i,
                conditions=list(rule["conditions"]),
                tp=float(tp),
                sl=float(sl),
                capital_pct=float(rule.get("capital_pct", 12.5)),
                match_idx=idxs,
                stand_score=float(st["score"]),
                stand_n=float(st["n"]),
                stand_pf=float(st["pf"]),
                stand_mean=float(st["mean_net"]),
            )
        )
        if (i + 1) % 25 == 0:
            print(f"  optimized {i+1}/{len(rules)}")

    candidates.sort(key=lambda c: c.stand_score, reverse=True)
    print(f"[stage1] {len(candidates)} candidates kept")
    print("  top 10 standalone:")
    for c in candidates[:10]:
        print(
            f"    orig={c.orig_idx:3d} n={c.stand_n:.0f} mean={c.stand_mean:.3f} "
            f"pf={c.stand_pf:.2f} tp={c.tp} sl={c.sl} score={c.stand_score:.4f}"
        )

    # Baseline: first 100 original order with original risk
    print("[baseline] original first 100 rules...")
    # map orig -> candidate position
    by_orig = {c.orig_idx: k for k, c in enumerate(candidates)}
    base_sel = [by_orig[i] for i in range(min(100, len(rules))) if i in by_orig]
    match_list = [c.match_idx for c in candidates]
    tps = [c.tp for c in candidates]
    sls = [c.sl for c in candidates]
    # use original tp/sl for true baseline
    orig_tps = [float(rules[c.orig_idx]["tp"]) for c in candidates]
    orig_sls = [float(rules[c.orig_idx]["sl"]) for c in candidates]
    orig_caps = [float(rules[c.orig_idx]["capital_pct"]) for c in candidates]
    base_m = simulate_portfolio(base_sel, match_list, orig_tps, orig_sls, orig_caps, arrays)
    print(
        f"[baseline] first100: ret={base_m['total_return_pct']:.2f}% "
        f"dd={base_m['max_drawdown_pct']:.2f}% pf={base_m['profit_factor']:.2f} "
        f"trades={base_m['executed_trades']}"
    )

    # all 167 baseline with orig params (using only candidates that exist)
    all_sel = [by_orig[i] for i in range(len(rules)) if i in by_orig]
    all_m = simulate_portfolio(all_sel, match_list, orig_tps, orig_sls, orig_caps, arrays)
    print(
        f"[baseline] all{len(all_sel)}: ret={all_m['total_return_pct']:.2f}% "
        f"dd={all_m['max_drawdown_pct']:.2f}% pf={all_m['profit_factor']:.2f} "
        f"trades={all_m['executed_trades']}"
    )

    if args.baseline_only:
        return 0

    # Stage 2: greedy select
    print(f"[stage2] greedy select -> {args.n_rules} rules...")
    # reset caps to 12.5 for selection fairness
    for c in candidates:
        c.capital_pct = 12.5
    selected = greedy_select(candidates, arrays, args.n_rules)
    tps = [c.tp for c in candidates]
    sls = [c.sl for c in candidates]
    caps = [c.capital_pct for c in candidates]
    sel_m = simulate_portfolio(selected, match_list, tps, sls, caps, arrays)
    print(
        f"[stage2] selected {len(selected)}: ret={sel_m['total_return_pct']:.2f}% "
        f"dd={sel_m['max_drawdown_pct']:.2f}% pf={sel_m['profit_factor']:.2f} "
        f"trades={sel_m['executed_trades']} score={sel_m['score']:.2f}"
    )

    # Stage 3: capital
    print("[stage3] capital optimization...")
    caps, cap_m = optimize_capital(selected, candidates, arrays)
    for i, cap in enumerate(caps):
        candidates[i].capital_pct = cap
    print(
        f"[stage3] ret={cap_m['total_return_pct']:.2f}% "
        f"dd={cap_m['max_drawdown_pct']:.2f}% pf={cap_m['profit_factor']:.2f} "
        f"trades={cap_m['executed_trades']}"
    )

    # Stage 4: prune + refill
    print("[stage4] prune redundant rules...")
    pruned = prune_redundant(selected, candidates, arrays, caps)
    if len(pruned) < args.n_rules:
        used = set(pruned)
        rest = [
            i
            for i in sorted(
                range(len(candidates)),
                key=lambda j: candidates[j].stand_score,
                reverse=True,
            )
            if i not in used
        ]
        match_list = [c.match_idx for c in candidates]
        tps = [c.tp for c in candidates]
        sls = [c.sl for c in candidates]
        base = simulate_portfolio(pruned, match_list, tps, sls, caps, arrays)
        for i in rest:
            if len(pruned) >= args.n_rules:
                break
            trial = sorted(
                pruned + [i],
                key=lambda j: candidates[j].stand_score,
                reverse=True,
            )
            m = simulate_portfolio(trial, match_list, tps, sls, caps, arrays)
            # accept if not much worse
            if m["score"] >= base["score"] * 0.98 or m["total_return_pct"] >= base[
                "total_return_pct"
            ] * 0.98:
                pruned = trial
                base = m
        selected = pruned
    else:
        selected = pruned

    # re-order by stand_score and re-opt capital
    selected = sorted(selected, key=lambda i: candidates[i].stand_score, reverse=True)
    if len(selected) > args.n_rules:
        selected = selected[: args.n_rules]
    caps, cap_m = optimize_capital(selected, candidates, arrays)

    # Stage 5: optional pairwise order swap for top of list (limited)
    print("[stage5] limited order local search...")
    match_list = [c.match_idx for c in candidates]
    tps = [c.tp for c in candidates]
    sls = [c.sl for c in candidates]
    best_order = list(selected)
    best_m = simulate_portfolio(best_order, match_list, tps, sls, caps, arrays)
    # try moving each of top-20 one position up if helps
    for _ in range(2):
        improved = False
        for pos in range(1, min(30, len(best_order))):
            trial = list(best_order)
            trial[pos - 1], trial[pos] = trial[pos], trial[pos - 1]
            m = simulate_portfolio(trial, match_list, tps, sls, caps, arrays)
            if m["score"] > best_m["score"] + 1e-9:
                best_order = trial
                best_m = m
                improved = True
        if not improved:
            break
    selected = best_order
    print(
        f"[stage5] final: ret={best_m['total_return_pct']:.2f}% "
        f"dd={best_m['max_drawdown_pct']:.2f}% pf={best_m['profit_factor']:.2f} "
        f"trades={best_m['executed_trades']} score={best_m['score']:.2f}"
    )

    # Build output strategy — exactly n_rules
    out_rules = []
    for i in selected[: args.n_rules]:
        c = candidates[i]
        out_rules.append(
            {
                "conditions": c.conditions,
                "tp": float(c.tp),
                "sl": float(c.sl),
                "capital_pct": float(caps[i]),
            }
        )

    # If still short, pad from remaining candidates
    if len(out_rules) < args.n_rules:
        used_orig = {candidates[i].orig_idx for i in selected}
        for c in candidates:
            if len(out_rules) >= args.n_rules:
                break
            if c.orig_idx in used_orig:
                continue
            out_rules.append(
                {
                    "conditions": c.conditions,
                    "tp": float(c.tp),
                    "sl": float(c.sl),
                    "capital_pct": float(c.capital_pct),
                }
            )
            used_orig.add(c.orig_idx)

    out = {"direction": direction, "rules_set": out_rules[: args.n_rules]}
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"[write] {args.output} with {len(out['rules_set'])} rules")

    report = {
        "dataset": str(args.data),
        "n_input_rules": len(rules),
        "n_output_rules": len(out["rules_set"]),
        "baseline_first100": base_m,
        "baseline_all": all_m,
        "optimized": best_m,
        "elapsed_sec": time.time() - t_all,
        "selected_orig_indices": [candidates[i].orig_idx for i in selected[: args.n_rules]],
        "tp_grid": list(TP_GRID),
        "sl_grid": list(SL_GRID),
        "capital_grid": list(CAPITAL_GRID),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[write] report {args.report}")
    print(f"[done] elapsed {time.time()-t_all:.1f}s")
    print(
        f"\nSUMMARY  baseline_all ret={all_m['total_return_pct']:.2f}% "
        f"-> optimized ret={best_m['total_return_pct']:.2f}% "
        f"({len(out['rules_set'])} rules)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
