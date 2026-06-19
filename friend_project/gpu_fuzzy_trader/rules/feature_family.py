from __future__ import annotations

import re
from collections import Counter


def extract_feature_name(condition: str) -> str | None:
    if not isinstance(condition, str):
        return None
    s = condition.strip()
    if s.lower().startswith("symbol") or s.lower().startswith("[symbol]"):
        return None
    m = re.match(r"^\s*\[([^\]]+)\]\s+IS\s+.+$", s, flags=re.IGNORECASE)
    return m.group(1).strip() if m else None


def feature_family(feature_name: str) -> str:
    """Map feature names into coarse families for redundancy control."""
    f = str(feature_name).lower()
    if any(k in f for k in ("rsi", "stoch", "momentum", "mom_", "roc", "macd", "dmi", "adx")):
        return "momentum_trend"
    if any(k in f for k in ("vol", "atr", "parkinson", "semivol", "bb_width", "range", "compression")):
        return "volatility_range"
    if any(k in f for k in ("wick", "body", "candle", "inside_bar", "engulf", "doji", "tr")):
        return "candle_reaction"
    if any(k in f for k in ("volume", "dollar_vol", "amihud", "liquid", "vol_over", "up_down_vol")):
        return "volume_liquidity"
    if any(k in f for k in ("bb_percb", "channel", "pos", "breakout", "drawdown", "peak", "support", "resistance")):
        return "area_structure"
    if any(k in f for k in ("skew", "corr", "autocorr", "sign_flip")):
        return "distribution_flow"
    return "other"


def condition_family(condition: str) -> str | None:
    name = extract_feature_name(condition)
    return feature_family(name) if name else None


def family_counts(conditions: list[str]) -> Counter:
    out: Counter = Counter()
    for c in conditions:
        fam = condition_family(c)
        if fam:
            out[fam] += 1
    return out


def non_symbol_conditions(conditions: list[str]) -> list[str]:
    return [c for c in conditions if extract_feature_name(c) is not None]


def symbol_conditions(conditions: list[str]) -> list[str]:
    return [c for c in conditions if extract_feature_name(c) is None and isinstance(c, str) and "symbol" in c.lower()]
