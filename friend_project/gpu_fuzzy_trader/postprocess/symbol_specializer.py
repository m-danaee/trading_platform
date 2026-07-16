
from __future__ import annotations

import copy
from typing import Iterable

import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.backtest.cpu_engine import CPUBacktestEngine, parse_symbol_condition, normalize_symbol_value
from gpu_fuzzy_trader.backtest.df_slim import slim_backtest_df
from gpu_fuzzy_trader.validation.monthly_windows import evaluate_rule_set_monthly


def _strip_symbol_conditions(conditions: Iterable[str]) -> tuple[list[str], list[str]]:
    features: list[str] = []
    symbols: list[str] = []
    for cond in conditions:
        parsed = parse_symbol_condition(cond) if isinstance(cond, str) else None
        if parsed is None:
            features.append(str(cond))
        else:
            symbols.extend(parsed)
    return features, symbols


def _symbol_list(df: pd.DataFrame) -> list[str]:
    if "symbol" not in df.columns:
        return []
    vals = [normalize_symbol_value(v) for v in pd.unique(df["symbol"])]
    return sorted(set(vals), key=lambda x: (len(str(x)), str(x)))


def _score_rule_on_symbol(df: pd.DataFrame, rule: dict, symbol: str, direction: str, feature_names: list[str] | None) -> dict:
    base_conditions, _ = _strip_symbol_conditions(rule.get("conditions", []))
    candidate = [{
        "conditions": [f"symbol is {symbol}"] + base_conditions,
        "tp": float(rule.get("tp", 2.0)),
        "sl": float(rule.get("sl", 1.0)),
        "capital_pct": float(rule.get("capital_pct", 1.0)),
    }]
    try:
        slim = slim_backtest_df(df, feature_names) if feature_names is not None else df
        metrics = CPUBacktestEngine(slim, {}, direction).simulate_rule_set(candidate)
    except Exception:
        metrics = {"total_return_pct": -100.0, "profit_factor": 0.0, "max_drawdown_pct": 100.0, "executed_trades": 0}
    try:
        monthly, _ = evaluate_rule_set_monthly(df, candidate, direction, feature_names=feature_names)
    except Exception:
        monthly = None
    ret = float(metrics.get("total_return_pct", 0.0))
    pf = float(metrics.get("profit_factor", 0.0))
    dd = float(metrics.get("max_drawdown_pct", 100.0))
    trades = int(metrics.get("executed_trades", 0))
    monthly_score = float(getattr(monthly, "score", 0.0)) if monthly is not None else 0.0
    worst_month = float(getattr(monthly, "worst_return_pct", ret)) if monthly is not None else ret
    prof_ratio = float(getattr(monthly, "profitable_ratio", 0.0)) if monthly is not None else 0.0
    score = ret + 5.0 * pf - 0.45 * dd + 0.35 * monthly_score + 0.50 * worst_month + 3.0 * prof_ratio - max(0, int(getattr(_cfg, "SYMBOL_SPECIALIZATION_MIN_TRADES", 20)) - trades) * 0.10
    return {"symbol": symbol, "score": score, "return": ret, "pf": pf, "dd": dd, "trades": trades, "worst_month": worst_month, "profitable_ratio": prof_ratio}


def specialize_rule_symbols(train_df: pd.DataFrame, val_df: pd.DataFrame, rules: list[dict], direction: str) -> list[dict]:
    """Return a copy of rules with evaluator_v5 symbol filters added.
    """
    if not bool(getattr(_cfg, "SYMBOL_SPECIALIZATION_ENABLED", True)):
        return copy.deepcopy(rules)
    combined = pd.concat([train_df, val_df], ignore_index=True)
    symbols = _symbol_list(combined)
    if not symbols:
        return copy.deepcopy(rules)
    feature_names = [
        c for c in combined.columns
        if c not in set(_cfg.LABEL_COLUMNS) | set(_cfg.META_COLUMNS) | set(_cfg.INTERNAL_COLUMNS)
        and not str(c).startswith("_")
    ]
    max_symbols = int(getattr(_cfg, "SYMBOL_SPECIALIZATION_MAX_SYMBOLS_PER_RULE", 3))
    min_trades = int(getattr(_cfg, "SYMBOL_SPECIALIZATION_MIN_TRADES", 20))
    min_pf = float(getattr(_cfg, "SYMBOL_SPECIALIZATION_MIN_PF", 1.02))
    min_score = float(getattr(_cfg, "SYMBOL_SPECIALIZATION_MIN_SCORE", -2.0))

    out: list[dict] = []
    for rule in rules:
        feature_conditions, existing_symbols = _strip_symbol_conditions(rule.get("conditions", []))
        if existing_symbols:
            out.append(copy.deepcopy(rule))
            continue
        scored = [_score_rule_on_symbol(combined[combined["symbol"].map(normalize_symbol_value) == sym], rule, sym, direction, feature_names) for sym in symbols]
        scored = [x for x in scored if x["trades"] >= min_trades and x["pf"] >= min_pf and x["score"] >= min_score]
        if not scored:
            scored = [_score_rule_on_symbol(combined[combined["symbol"].map(normalize_symbol_value) == sym], rule, sym, direction, feature_names) for sym in symbols]
            scored = sorted(scored, key=lambda x: x["score"], reverse=True)[:max(1, min(2, max_symbols))]
        else:
            scored = sorted(scored, key=lambda x: x["score"], reverse=True)[:max_symbols]
        selected_symbols = [x["symbol"] for x in scored]
        new_rule = copy.deepcopy(rule)
        new_rule["conditions"] = [f"symbol is {s}" for s in selected_symbols] + feature_conditions
        new_rule["symbol_specialization"] = scored
        out.append(new_rule)
    return out
