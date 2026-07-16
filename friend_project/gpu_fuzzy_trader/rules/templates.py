from __future__ import annotations

import random
from dataclasses import dataclass

from gpu_fuzzy_trader.rules.condition_quality import ConditionQualityStore
from gpu_fuzzy_trader.rules.repair import repair_rule


@dataclass(frozen=True)
class RuleTemplate:
    name: str
    families: tuple[str, ...]


DEFAULT_TEMPLATES = (
    RuleTemplate("trend_continuation", ("momentum_trend", "area_structure", "volume_liquidity", "volatility_range", "candle_reaction")),
    RuleTemplate("pullback", ("area_structure", "momentum_trend", "volatility_range", "candle_reaction", "volume_liquidity")),
    RuleTemplate("compression_breakout", ("volatility_range", "area_structure", "momentum_trend", "volume_liquidity", "candle_reaction")),
    RuleTemplate("mean_reversion", ("area_structure", "distribution_flow", "momentum_trend", "candle_reaction", "volatility_range")),
    RuleTemplate("wick_reversal", ("candle_reaction", "area_structure", "momentum_trend", "volatility_range", "volume_liquidity")),
)


def generate_template_rules(store: ConditionQualityStore, direction: str, *, rng: random.Random, max_rules: int = 80) -> list[dict]:
    out: list[dict] = []
    seen: set[tuple[str, ...]] = set()
    per_template = max(1, max_rules // max(1, len(DEFAULT_TEMPLATES)))
    for tmpl in DEFAULT_TEMPLATES:
        for _ in range(per_template):
            conditions: list[str] = []
            for fam in tmpl.families:
                bank = store.best_conditions(fam, limit=25)
                if not bank:
                    continue
                idx = min(len(bank) - 1, int(rng.expovariate(0.35)))
                c = bank[idx]
                if c not in conditions:
                    conditions.append(c)
            if len(conditions) < 4:
                continue
            rule = repair_rule({"conditions": conditions, "tp": 2.0, "sl": 1.0, "capital_pct": 5.0}, direction=direction)
            key = tuple(sorted(rule["conditions"]))
            if key not in seen:
                rule["template"] = tmpl.name
                out.append(rule)
                seen.add(key)
            if len(out) >= max_rules:
                return out
    return out
