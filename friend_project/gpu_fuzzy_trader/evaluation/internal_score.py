
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.backtest.cpu_engine import CPUBacktestEngine
from gpu_fuzzy_trader.validation.rolling_cv import build_fold_engines, evaluate_rule_set_on_fold_engines
from gpu_fuzzy_trader.validation.monthly_windows import evaluate_rule_set_monthly
from gpu_fuzzy_trader.scoring import robust_ratio_score, return_to_drawdown


def _engine_rule_set(strategy: dict) -> list[dict]:
    rules = strategy.get("rules_set", [])
    out: list[dict] = []
    for rule in rules:
        out.append(
            {
                "conditions": list(rule["conditions"]),
                "tp": float(rule["tp"]),
                "sl": float(rule["sl"]),
                "capital_pct": float(rule.get("capital_pct", strategy.get("capital_pct", 0.0))),
            }
        )
    return out


def robust_internal_score(train_metrics: dict, valid_metrics: dict, fold_summary: Any, monthly_summary: Any = None) -> float:
    """Scalar score used by auto-search summary; higher is better.

    V6 uses return/max-drawdown as the dominant signal. This makes a strategy
    with +6% return and 4% drawdown rank above one with +10% return and 14%
    drawdown, which is closer to how the final evaluator punishes fragile rules.
    """
    return robust_ratio_score(
        train_metrics,
        valid_metrics,
        fold_summary,
        monthly_summary,
        min_trades=int(getattr(_cfg, "AUTO_SEARCH_SCORE_MIN_TRADES", 80)),
        min_fold_trades=int(getattr(_cfg, "PHASE3_MIN_FOLD_TRADES", 20)),
        dd_floor=float(getattr(_cfg, "RETURN_DD_FLOOR", 1.0)),
        include_train_gap=True,
    )


def evaluate_strategy_internal(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    strategy: dict,
    direction: str | None = None,
) -> dict[str, Any]:
    """Evaluate one exported strategy on train, validation, and purged folds."""
    direction = direction or str(strategy.get("direction", "long"))
    rules = _engine_rule_set(strategy)
    train_engine = CPUBacktestEngine(train_df, {}, direction)
    valid_engine = CPUBacktestEngine(val_df, {}, direction)
    train_metrics = train_engine.simulate_rule_set(rules)
    valid_metrics = valid_engine.simulate_rule_set(rules)

    combined = pd.concat([train_df, val_df], ignore_index=True)
    feature_names = [
        c for c in combined.columns
        if c not in set(_cfg.LABEL_COLUMNS) | set(_cfg.META_COLUMNS) | set(_cfg.INTERNAL_COLUMNS)
        and not str(c).startswith("_")
    ]
    fold_engines = build_fold_engines(combined, direction, feature_names=feature_names)
    fold_summary = evaluate_rule_set_on_fold_engines(rules, fold_engines) if fold_engines else None

    if fold_summary is None:
        from gpu_fuzzy_trader.validation.rolling_cv import summarize_fold_metrics
        fold_summary = summarize_fold_metrics([valid_metrics])

    monthly_summary = None
    try:
        if getattr(_cfg, "MONTHLY_VALIDATION_ENABLED", False):
            monthly_summary, _ = evaluate_rule_set_monthly(combined, rules, direction, feature_names=feature_names)
    except Exception:
        monthly_summary = None

    score = robust_internal_score(train_metrics, valid_metrics, fold_summary, monthly_summary)
    return {
        "direction": direction,
        "rules": len(rules),
        "internal_score": score,
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "fold_summary": {
            "folds": fold_summary.folds,
            "worst_return_pct": fold_summary.worst_return_pct,
            "worst_profit_factor": fold_summary.worst_profit_factor,
            "worst_sortino_ratio": fold_summary.worst_sortino_ratio,
            "worst_drawdown_pct": fold_summary.worst_drawdown_pct,
            "min_trades": fold_summary.min_trades,
            "mean_return_pct": fold_summary.mean_return_pct,
            "mean_profit_factor": fold_summary.mean_profit_factor,
        },
        "monthly_summary": None if monthly_summary is None else dict(monthly_summary.__dict__),
    }


def evaluate_strategy_file_internal(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    strategy_path: str | os.PathLike[str],
) -> dict[str, Any]:
    path = Path(strategy_path)
    with path.open("r", encoding="utf-8") as fh:
        strategy = json.load(fh)
    return evaluate_strategy_internal(train_df, val_df, strategy, strategy.get("direction"))
