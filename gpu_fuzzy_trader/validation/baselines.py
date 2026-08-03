"""Simple baselines and ablations for every research release.

These baselines are deliberately small and evaluator-compatible.  They answer
whether RB sizing, entry conditions, or the exit geometry is responsible for
the observed result before more evolutionary capacity is added.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.backtest.cpu_engine import (
    CPUBacktestEngine,
    compute_entry_time_priority,
)


def _evaluate(
    frame: pd.DataFrame,
    direction: str,
    rules: list[dict],
) -> dict[str, Any]:
    return CPUBacktestEngine(frame, {}, direction).simulate_rule_set(rules)


def _compact(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metrics.get(key, 0.0)
        for key in (
            "total_return_pct",
            "profit_factor",
            "max_drawdown_pct",
            "win_rate",
            "executed_trades",
            "expectancy_lcb_pct_per_trade",
        )
    }


def _fixed_exit(rules: list[dict]) -> list[dict]:
    return [
        {
            **rule,
            "tp": float(getattr(_cfg, "PHASE2_TP", 2.0)),
            "sl": float(getattr(_cfg, "PHASE2_SL", 1.2)),
        }
        for rule in rules
    ]


def _equal_weight(rules: list[dict]) -> list[dict]:
    if not rules:
        return []
    total = sum(float(rule.get("capital_pct", 0.0)) for rule in rules)
    equal = total / len(rules)
    return [{**rule, "capital_pct": equal} for rule in rules]


def _drop_one_condition_variants(rules: list[dict]) -> list[list[dict]]:
    variants: list[list[dict]] = []
    for rule_index, rule in enumerate(rules):
        conditions = list(rule.get("conditions", []))
        feature_conditions = [
            condition for condition in conditions
            if not str(condition).lower().startswith(
                ("symbol is ", "[symbol] is ")
            )
        ]
        for condition in feature_conditions:
            if len(feature_conditions) <= 1:
                continue
            changed = [
                dict(item)
                for item in rules
            ]
            changed[rule_index]["conditions"] = [
                value for value in conditions if value != condition
            ]
            variants.append(changed)
    return variants


def _random_entry(
    frame: pd.DataFrame,
    direction: str,
    rules: list[dict],
    *,
    seed: int,
) -> dict[str, Any]:
    """Evaluate random entries with the same exit and capital contract."""
    if not rules or frame.empty:
        return {"total_return_pct": 0.0, "executed_trades": 0}
    rng = np.random.default_rng(seed)
    n_entries = min(len(frame), max(1, int(len(frame) * 0.03)))
    indices = np.sort(rng.choice(len(frame), size=n_entries, replace=False))
    rule = rules[0]
    if "datetime" in frame.columns:
        priority = compute_entry_time_priority(frame["datetime"].values, len(frame))
    else:
        priority = np.arange(len(frame), dtype=np.int64)
    entries = [
        {
            "idx": int(index),
            "entry_priority": int(priority[index]),
            "rule_index": 1,
            "symbol_priority": 0,
            "tp": float(rule.get("tp", _cfg.PHASE2_TP)),
            "sl": float(rule.get("sl", _cfg.PHASE2_SL)),
            "capital_pct": float(rule.get("capital_pct", _cfg.PHASE2_CAPITAL_PCT)),
        }
        for index in indices
    ]
    engine = CPUBacktestEngine(frame, {}, direction)
    return engine._simulate_rule_set_entries(
        entries,
        return_logs=False,
        initial_capital=engine.initial_capital,
    )


def _feature_shuffle(
    frame: pd.DataFrame,
    direction: str,
    rules: list[dict],
    *,
    seed: int,
) -> dict[str, Any]:
    shuffled = frame.copy()
    rng = np.random.default_rng(seed)
    feature_names = {
        str(condition).split("]", 1)[0].lstrip("[")
        for rule in rules
        for condition in rule.get("conditions", [])
        if str(condition).startswith("[")
    }
    for name in feature_names:
        if name in shuffled.columns:
            values = shuffled[name].to_numpy(copy=True)
            rng.shuffle(values)
            shuffled[name] = values
    return _evaluate(shuffled, direction, rules)


def evaluate_baselines(
    frame: pd.DataFrame,
    strategy: dict[str, Any],
    *,
    seed: int = 17,
) -> dict[str, Any]:
    """Return compact baseline and entry-ablation metrics."""
    direction = str(strategy.get("direction", "long"))
    rules = [dict(rule) for rule in strategy.get("rules_set", [])]
    if not rules:
        return {"direction": direction, "status": "no_strategy"}

    output: dict[str, Any] = {
        "direction": direction,
        "strategy_id": strategy.get("strategy_id"),
        "fixed_phase2_exit": _compact(_evaluate(
            frame, direction, _fixed_exit(rules),
        )),
        "equal_weight_capital": _compact(_evaluate(
            frame, direction, _equal_weight(rules),
        )),
        "feature_shuffle": _compact(_feature_shuffle(
            frame, direction, rules, seed=seed,
        )),
        "random_entry_same_exit": _compact(_random_entry(
            frame, direction, rules, seed=seed,
        )),
    }
    drop_results = [
        _compact(_evaluate(frame, direction, variant))
        for variant in _drop_one_condition_variants(rules)
    ]
    output["drop_one_condition"] = {
        "variants": len(drop_results),
        "median_return_pct": float(
            np.median([
                float(row.get("total_return_pct", 0.0))
                for row in drop_results
            ])
        ) if drop_results else 0.0,
        "worst_return_pct": float(
            np.min([
                float(row.get("total_return_pct", 0.0))
                for row in drop_results
            ])
        ) if drop_results else 0.0,
        "metrics": drop_results,
    }
    return output


def write_baseline_reports(
    output_dir: str,
    frame: pd.DataFrame,
    strategies: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Write baseline reports without consulting any OOS/test frame."""
    reports_dir = Path(output_dir) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, dict[str, Any]] = {}
    for direction, strategy in strategies.items():
        if direction not in {"long", "short"} or not strategy.get("rules_set"):
            continue
        report = evaluate_baselines(frame, strategy)
        reports[direction] = report
        (reports_dir / f"baseline_{direction}.json").write_text(
            json.dumps(report, indent=2, default=str),
            encoding="utf-8",
        )
    return reports
