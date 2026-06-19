from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.validation.monthly_windows import evaluate_rule_set_monthly
from gpu_fuzzy_trader.rules.repair import repair_rule_set


@dataclass
class FitnessResult:
    score: float
    train_metrics: dict
    valid_metrics: dict
    monthly_summary: dict
    penalties: dict


def fitness_individual(individual: dict | list[dict], *, train_engine, valid_engine, direction: str, monthly_df=None, feature_names=None) -> FitnessResult:
    rules = individual.get("rules_set", individual) if isinstance(individual, dict) else individual
    repaired = repair_rule_set(rules, direction=direction, min_rules=1, max_rules=int(getattr(_cfg, "PHASE3_MAX_RULES", 5)))
    train_metrics = train_engine.simulate_rule_set(repaired)
    valid_metrics = valid_engine.simulate_rule_set(repaired)
    monthly_summary = None
    monthly_score = 0.0
    if monthly_df is not None and getattr(_cfg, "MONTHLY_VALIDATION_ENABLED", False):
        monthly_summary, _ = evaluate_rule_set_monthly(monthly_df, repaired, direction, feature_names=feature_names)
        monthly_score = float(monthly_summary.score)
    ret = float(valid_metrics.get("total_return_pct", 0.0))
    pf = float(valid_metrics.get("profit_factor", 0.0))
    dd = float(valid_metrics.get("max_drawdown_pct", 100.0))
    trades = int(valid_metrics.get("executed_trades", 0))
    train_ret = float(train_metrics.get("total_return_pct", 0.0))
    gap = max(0.0, train_ret - ret)
    penalties = {
        "gap": gap * 0.2,
        "dd": max(0.0, dd - float(getattr(_cfg, "MONTHLY_MAX_DD", 8.0))) * 0.5,
        "trades": max(0, int(getattr(_cfg, "AUTO_SEARCH_SCORE_MIN_TRADES", 80)) - trades) * 0.05,
    }
    score = ret + 7.0 * pf - 0.6 * dd + monthly_score - sum(penalties.values())
    return FitnessResult(score=float(score), train_metrics=train_metrics, valid_metrics=valid_metrics, monthly_summary=asdict(monthly_summary) if monthly_summary else {}, penalties=penalties)
