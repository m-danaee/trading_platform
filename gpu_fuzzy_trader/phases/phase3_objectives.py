"""
phase3_objectives.py — Shared utilities for Phase 3.
"""

from __future__ import annotations

import numpy as np

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.backtest.symbol_conditions import (
    normalize_symbol_value,
    parse_symbol_condition,
)


def conditions_key(conditions: list[str]) -> frozenset[str]:
    return frozenset(conditions)


def feature_conditions_key(conditions: list[str]) -> tuple[str, ...]:
    feature_only = [
        cond for cond in conditions
        if parse_symbol_condition(cond) is None
    ]
    return tuple(sorted(feature_only))


def normalize_export_symbol(symbol: str) -> str:
    return normalize_symbol_value(symbol)


def rule_set_has_symbol_filters(rule_set: list[dict]) -> bool:
    for rule in rule_set:
        for cond in rule.get("conditions", []):
            if parse_symbol_condition(cond) is not None:
                return True
    return False


def has_duplicate_rules(rule_set: list[dict]) -> bool:
    seen: set[frozenset] = set()
    for rule in rule_set:
        key = conditions_key(rule["conditions"])
        if key in seen:
            return True
        seen.add(key)
    return False


def count_symbols_with_trades(metrics: dict) -> int:
    per_sym = metrics.get("per_symbol_metrics", {})
    return sum(1 for v in per_sym.values() if v.get("trade_count", 0) > 0)


def symbols_with_trades(metrics: dict) -> set:
    per_sym = metrics.get("per_symbol_metrics", {})
    return {
        s for s, v in per_sym.items()
        if v.get("trade_count", 0) > 0
    }


def per_symbol_pnl_vector(metrics: dict, symbols: list) -> np.ndarray:
    per = metrics.get("per_symbol_metrics", {}) or {}
    out = np.zeros(len(symbols), dtype=np.float64)
    for i, sym in enumerate(symbols):
        v = per.get(sym, per.get(str(sym), {}))
        out[i] = float(v.get("net_pnl", 0.0)) if isinstance(v, dict) else 0.0
    return out


def min_per_symbol_trades_from_metrics(metrics: dict) -> int:
    per = metrics.get("per_symbol_metrics", {}) or {}
    if not per:
        return 0
    worst = float("inf")
    for v in per.values():
        tc = int(v.get("trade_count", 0)) if isinstance(v, dict) else 0
        if tc < worst:
            worst = tc
    return 0 if worst == float("inf") else int(worst)