from __future__ import annotations

import itertools
import random
from collections import Counter

import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.backtest.cpu_engine import _build_rule_signal_mask
from gpu_fuzzy_trader.rules.repair import repair_rule


def mine_apriori_seed_rules(df: pd.DataFrame, conditions: list[str], direction: str, *, max_rules: int = 80, rng: random.Random | None = None) -> list[dict]:
    rng = rng or random.Random(42)
    max_conditions = min(len(conditions), int(getattr(_cfg, "APRIORI_MAX_CONDITION_BANK", 120)))
    conds = list(conditions)[:max_conditions]
    if not conds:
        return []
    n = max(1, len(df))
    masks: dict[str, np.ndarray] = {}
    supports: dict[str, float] = {}
    for c in conds:
        try:
            m = _build_rule_signal_mask(df, [c])
            sup = float(np.mean(m))
            if sup >= float(getattr(_cfg, "APRIORI_MIN_SINGLE_SUPPORT", 0.002)) and sup <= float(getattr(_cfg, "APRIORI_MAX_SINGLE_SUPPORT", 0.35)):
                masks[c] = m
                supports[c] = sup
        except Exception:
            pass
    ranked = sorted(masks, key=lambda c: supports[c], reverse=True)
    out: list[dict] = []
    seen: set[tuple[str, ...]] = set()
    for a, b in itertools.combinations(ranked[:80], 2):
        inter = masks[a] & masks[b]
        sup = float(np.mean(inter))
        if sup < float(getattr(_cfg, "APRIORI_MIN_PAIR_SUPPORT", 0.001)):
            continue
        lift_like = sup / max(supports[a] * supports[b], 1e-12)
        if lift_like < float(getattr(_cfg, "APRIORI_MIN_LIFT_LIKE", 1.05)):
            continue
        rule = repair_rule({"conditions": [a, b], "tp": 2.0, "sl": 1.0, "capital_pct": 5.0}, direction=direction)
        key = tuple(sorted(rule["conditions"]))
        if key not in seen:
            rule["apriori_lift_like"] = float(lift_like)
            out.append(rule)
            seen.add(key)
        if len(out) >= max_rules:
            break
    return out
