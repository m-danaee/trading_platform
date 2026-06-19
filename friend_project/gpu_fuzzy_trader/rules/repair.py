from __future__ import annotations

import re
from typing import Iterable

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.backtest.cpu_engine import parse_symbol_condition, normalize_symbol_value
from gpu_fuzzy_trader.rules.feature_family import extract_feature_name, feature_family

_TARGET_TOKENS = ("target", "future", "label", "ret_fwd", "forward", "take_profit", "stop_loss")


def _is_target_like_condition(condition: str) -> bool:
    s = str(condition).lower()
    if s.strip().lower().startswith("symbol") or s.strip().lower().startswith("[symbol]"):
        return False
    feat = extract_feature_name(condition)
    if feat is None:
        return True
    return any(tok in feat.lower() for tok in _TARGET_TOKENS)


def _unique_symbol_values(conditions: Iterable[str]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for c in conditions:
        parsed = parse_symbol_condition(c)
        if parsed:
            for v in parsed:
                nv = normalize_symbol_value(v)
                if nv not in seen:
                    values.append(nv)
                    seen.add(nv)
    return values


def canonical_symbol_conditions(values: Iterable[str]) -> list[str]:
    out = []
    seen: set[str] = set()
    for v in values:
        nv = normalize_symbol_value(v)
        if nv and nv not in seen:
            out.append(f"symbol is {nv}")
            seen.add(nv)
    return out


def split_conditions(conditions: Iterable[str]) -> tuple[list[str], list[str]]:
    syms = _unique_symbol_values(conditions)
    features: list[str] = []
    seen_features: set[str] = set()
    for raw in conditions:
        c = str(raw).strip()
        if not c or parse_symbol_condition(c) is not None:
            continue
        if _is_target_like_condition(c):
            continue
        feat = extract_feature_name(c)
        if feat is None:
            continue
        key = feat.lower()
        if key in seen_features:
            continue
        seen_features.add(key)
        features.append(c)
    return canonical_symbol_conditions(syms), features


def repair_rule(rule: dict, *, direction: str | None = None) -> dict:
    """Return a safe evaluator_v5-compatible rule dict."""
    max_symbols = int(getattr(_cfg, "SYMBOL_SPECIALIZATION_MAX_SYMBOLS_PER_RULE", 3))
    min_feature_conditions = int(getattr(_cfg, "RULE_REPAIR_MIN_FEATURE_CONDITIONS", 4))
    max_feature_conditions = int(getattr(_cfg, "RULE_REPAIR_MAX_FEATURE_CONDITIONS", 9))
    max_per_family = int(getattr(_cfg, "RULE_REPAIR_MAX_PER_FAMILY", 2))

    sym_conds, feat_conds = split_conditions(rule.get("conditions", []))
    if max_symbols > 0:
        sym_conds = sym_conds[:max_symbols]

    kept: list[str] = []
    fam_counts: dict[str, int] = {}
    for c in feat_conds:
        feat = extract_feature_name(c)
        fam = feature_family(feat or "")
        if fam_counts.get(fam, 0) >= max_per_family:
            continue
        kept.append(c)
        fam_counts[fam] = fam_counts.get(fam, 0) + 1
        if len(kept) >= max_feature_conditions:
            break

    if len(kept) < min_feature_conditions:
        kept = feat_conds[:max_feature_conditions]

    tp = float(rule.get("tp", getattr(_cfg, "PHASE2_TP", 2.0)))
    sl = float(rule.get("sl", getattr(_cfg, "PHASE2_SL", 1.0)))
    cap = float(rule.get("capital_pct", getattr(_cfg, "PHASE2_CAPITAL_PCT", 5.0)))
    tp = min(max(tp, float(getattr(_cfg, "PHASE4_TP_MIN", 1.0))), float(getattr(_cfg, "PHASE4_TP_MAX", 12.0)))
    sl = min(max(sl, float(getattr(_cfg, "PHASE4_SL_MIN", 0.5))), float(getattr(_cfg, "PHASE4_SL_MAX", 5.0)))
    cap = min(max(cap, float(getattr(_cfg, "PHASE4_CAPITAL_PCT_MIN", 1.0))), float(getattr(_cfg, "PHASE4_CAPITAL_PCT_MAX", 50.0)))

    out = dict(rule)
    out["conditions"] = sym_conds + kept
    out["tp"] = round(tp, 4)
    out["sl"] = round(sl, 4)
    out["capital_pct"] = round(cap, 4)
    if direction:
        out.setdefault("direction", direction)
    return out


def repair_rule_set(rules: Iterable[dict], *, direction: str | None = None, min_rules: int | None = None, max_rules: int | None = None) -> list[dict]:
    min_rules = int(min_rules if min_rules is not None else getattr(_cfg, "PHASE3_MIN_RULES", 1))
    max_rules = int(max_rules if max_rules is not None else getattr(_cfg, "PHASE3_MAX_RULES", 5))
    out: list[dict] = []
    seen: set[tuple[str, ...]] = set()
    for r in rules:
        rr = repair_rule(r, direction=direction)
        key = tuple(sorted(rr.get("conditions", [])))
        if not rr.get("conditions") or key in seen:
            continue
        out.append(rr)
        seen.add(key)
        if len(out) >= max_rules:
            break
    return out


def combine_rules_and(rule_a: dict, rule_b: dict, *, direction: str | None = None) -> dict | None:
    """AND-combine two rules, intersecting symbol filters when both have them.

    evaluator_v5 ORs multiple symbol filters inside one rule.  For AND-combining
    two existing rules, the correct symbol set is the intersection when both
    rules specify symbols; otherwise it is the specified side.
    """
    a_syms, a_feats = split_conditions(rule_a.get("conditions", []))
    b_syms, b_feats = split_conditions(rule_b.get("conditions", []))
    a_vals = _unique_symbol_values(a_syms)
    b_vals = _unique_symbol_values(b_syms)
    if a_vals and b_vals:
        vals = [v for v in a_vals if v in set(b_vals)]
        if not vals:
            return None
    else:
        vals = a_vals or b_vals
    merged = {
        "conditions": canonical_symbol_conditions(vals) + a_feats + b_feats,
        "tp": float(rule_a.get("tp", rule_b.get("tp", getattr(_cfg, "PHASE2_TP", 2.0)))),
        "sl": float(rule_a.get("sl", rule_b.get("sl", getattr(_cfg, "PHASE2_SL", 1.0)))),
        "capital_pct": min(float(rule_a.get("capital_pct", 5.0)), float(rule_b.get("capital_pct", 5.0))),
    }
    return repair_rule(merged, direction=direction)
