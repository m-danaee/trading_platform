from __future__ import annotations

import random

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.rules.condition_quality import ConditionQualityStore
from gpu_fuzzy_trader.rules.repair import repair_rule, combine_rules_and
from gpu_fuzzy_trader.rules.feature_family import extract_feature_name, feature_family, family_counts
from gpu_fuzzy_trader.rules.templates import generate_template_rules
from gpu_fuzzy_trader.rules.apriori_seeds import mine_apriori_seed_rules


def _quality_score(entry: dict) -> float:
    obj = entry.get("objectives", {}) if isinstance(entry.get("objectives", {}), dict) else {}
    cv = entry.get("cv_summary", {}) if isinstance(entry.get("cv_summary", {}), dict) else {}
    return (
        float(obj.get("total_return_pct", 0.0))
        + 3.0 * float(obj.get("sortino_ratio", 0.0))
        + 5.0 * float(cv.get("worst_profit_factor", 0.0))
        + 0.02 * float(entry.get("executed_trades", 0))
        - 0.2 * float(obj.get("max_drawdown_pct", 0.0))
    )


def _entry_from_rule(rule: dict, metrics: dict, source: str) -> dict:
    return {
        "chromosome": [],
        "conditions": list(rule["conditions"]),
        "tp": float(rule.get("tp", getattr(_cfg, "PHASE2_TP", 2.0))),
        "sl": float(rule.get("sl", getattr(_cfg, "PHASE2_SL", 1.0))),
        "capital_pct": float(rule.get("capital_pct", getattr(_cfg, "PHASE2_CAPITAL_PCT", 5.0))),
        "objectives": {
            "sortino_ratio": float(metrics.get("sortino_ratio", metrics.get("total_return_pct", 0.0))),
            "total_return_pct": float(metrics.get("total_return_pct", 0.0)),
            "max_drawdown_pct": float(metrics.get("max_drawdown_pct", 0.0)),
            "win_rate": float(metrics.get("win_rate", 0.0)),
            "profit_factor": float(metrics.get("profit_factor", 0.0)),
        },
        "executed_trades": int(metrics.get("executed_trades", 0)),
        "source": source,
    }


def _evaluate_rule(engine, rule: dict) -> dict | None:
    try:
        metrics = engine.simulate_rule_set([rule])
        trades = int(metrics.get("executed_trades", 0))
        if trades < int(getattr(_cfg, "MIN_TRADE_POOL_FLOOR", 30)):
            return None
        raw = int(metrics.get("raw_signal_count", trades))
        max_raw = int(getattr(_cfg, "RULE_GEN_MAX_RAW_SIGNALS", 8000))
        if raw > max_raw:
            return None
        return metrics
    except Exception:
        return None


def _expand_seed(seed: dict, store: ConditionQualityStore, direction: str, rng: random.Random) -> list[dict]:
    base = repair_rule(seed, direction=direction)
    out: list[dict] = []
    min_len = int(getattr(_cfg, "RULE_GEN_MIN_FEATURE_CONDITIONS", 5))
    max_len = int(getattr(_cfg, "RULE_GEN_MAX_FEATURE_CONDITIONS", 9))
    max_variants = int(getattr(_cfg, "RULE_GEN_EXPANSIONS_PER_SEED", 3))
    for _ in range(max_variants):
        conds = list(base.get("conditions", []))
        fams = family_counts(conds)
        banks = store.best_conditions(limit=200)
        rng.shuffle(banks)
        for c in banks:
            if c in conds:
                continue
            fam = feature_family(extract_feature_name(c) or "")
            if fams.get(fam, 0) >= int(getattr(_cfg, "RULE_REPAIR_MAX_PER_FAMILY", 2)):
                continue
            conds.append(c)
            fams[fam] = fams.get(fam, 0) + 1
            feature_count = len([x for x in conds if extract_feature_name(x)])
            if feature_count >= rng.randint(min_len, max_len):
                break
        rule = repair_rule({**base, "conditions": conds}, direction=direction)
        if len([x for x in rule["conditions"] if extract_feature_name(x)]) >= min_len:
            out.append(rule)
    return out


def augment_phase2_pool_with_generated_candidates(pool: list[dict], *, engine, train_df, direction: str, rng_seed: int = 42) -> list[dict]:
    if not getattr(_cfg, "RULE_GENERATION_ENABLED", True) or not pool:
        return pool
    rng = random.Random(rng_seed)
    repaired_entries: list[dict] = []
    for e in pool:
        r = repair_rule({"conditions": e.get("conditions", []), "tp": e.get("tp", getattr(_cfg, "PHASE2_TP", 2.0)), "sl": e.get("sl", getattr(_cfg, "PHASE2_SL", 1.0)), "capital_pct": e.get("capital_pct", getattr(_cfg, "PHASE2_CAPITAL_PCT", 5.0))}, direction=direction)
        ee = dict(e); ee["conditions"] = r["conditions"]; ee["tp"] = r["tp"]; ee["sl"] = r["sl"]; ee["capital_pct"] = r["capital_pct"]
        repaired_entries.append(ee)
    pool = repaired_entries
    store = ConditionQualityStore.from_pool_and_df(pool, train_df, direction)
    candidates: list[tuple[dict, str]] = []
    ranked = sorted(pool, key=_quality_score, reverse=True)
    top_n = min(len(ranked), int(getattr(_cfg, "RULE_GEN_TOP_SEEDS", 60)))
    for seed in ranked[:top_n]:
        candidates.append((repair_rule(seed, direction=direction), "repaired_seed"))
        for exp in _expand_seed(seed, store, direction, rng):
            candidates.append((exp, "expanded_seed"))
    max_pairs = int(getattr(_cfg, "RULE_GEN_MAX_AND_PAIRS", 160))
    pairs_done = 0
    for i, a in enumerate(ranked[:top_n]):
        for b in ranked[i+1:top_n]:
            if pairs_done >= max_pairs:
                break
            comb = combine_rules_and(a, b, direction=direction)
            if comb is not None:
                candidates.append((comb, "and_composed"))
                pairs_done += 1
        if pairs_done >= max_pairs:
            break
    for tr in generate_template_rules(store, direction, rng=rng, max_rules=int(getattr(_cfg, "TEMPLATE_GENERATOR_MAX_RULES", 100))):
        candidates.append((tr, "template"))
    all_conditions = store.best_conditions(limit=int(getattr(_cfg, "APRIORI_MAX_CONDITION_BANK", 120)))
    for ar in mine_apriori_seed_rules(train_df, all_conditions, direction, max_rules=int(getattr(_cfg, "APRIORI_MAX_SEED_RULES", 80)), rng=rng):
        candidates.append((ar, "apriori_seed"))

    seen = {tuple(sorted(e.get("conditions", []))) for e in pool}
    added: list[dict] = []
    budget = int(getattr(_cfg, "RULE_GEN_MAX_BACKTEST_CANDIDATES", 450))
    for rule, source in candidates[:budget]:
        key = tuple(sorted(rule.get("conditions", [])))
        if key in seen:
            continue
        metrics = _evaluate_rule(engine, rule)
        if metrics is None:
            continue
        added.append(_entry_from_rule(rule, metrics, source))
        seen.add(key)
    return pool + added
