from __future__ import annotations

from dataclasses import dataclass
import math
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.backtest.cpu_engine import _build_rule_signal_mask
from gpu_fuzzy_trader.rules.feature_family import extract_feature_name, feature_family


@dataclass
class ConditionQuality:
    condition: str
    support: float
    trades: int
    proxy_lift: float
    quality: float
    family: str


class ConditionQualityStore:
    def __init__(self, qualities: dict[str, ConditionQuality]):
        self.qualities = qualities
        self.by_family: dict[str, list[ConditionQuality]] = defaultdict(list)
        for q in qualities.values():
            self.by_family[q.family].append(q)
        for fam in list(self.by_family):
            self.by_family[fam].sort(key=lambda q: q.quality, reverse=True)

    @classmethod
    def from_pool_and_df(cls, pool: list[dict], df: pd.DataFrame, direction: str) -> "ConditionQualityStore":
        candidates: set[str] = set()
        for e in pool:
            for c in e.get("conditions", []):
                if extract_feature_name(c):
                    candidates.add(c)
        qualities: dict[str, ConditionQuality] = {}
        n = max(1, len(df))
        label_cols = [c for c in getattr(_cfg, "LABEL_COLUMNS", []) if c in df.columns]
        direction_label = None
        for c in label_cols:
            if direction in c.lower():
                direction_label = c
                break
        base_rate = 0.0
        if direction_label:
            try:
                base_rate = float(pd.to_numeric(df[direction_label], errors="coerce").fillna(0).mean())
            except Exception:
                base_rate = 0.0
        for cond in candidates:
            try:
                mask = _build_rule_signal_mask(df, [cond])
                trades = int(np.sum(mask))
                support = trades / n
                lift = 1.0
                if direction_label and trades > 0 and base_rate > 1e-12:
                    hit = float(pd.to_numeric(df.loc[mask, direction_label], errors="coerce").fillna(0).mean())
                    lift = hit / base_rate if np.isfinite(hit) else 1.0
                broad_penalty = max(0.0, support - float(getattr(_cfg, "CONDITION_SUPPORT_BROAD_PCT", 0.20))) * 3.0
                rare_penalty = max(0.0, float(getattr(_cfg, "CONDITION_SUPPORT_RARE_PCT", 0.005)) - support) * 50.0
                quality = math.log1p(trades) * 0.2 + max(0.0, lift - 1.0) * 3.0 - broad_penalty - rare_penalty
                fam = feature_family(extract_feature_name(cond) or "")
                qualities[cond] = ConditionQuality(cond, support, trades, lift, quality, fam)
            except Exception:
                continue
        return cls(qualities)

    def best_conditions(self, family: str | None = None, limit: int = 20) -> list[str]:
        if family is not None:
            return [q.condition for q in self.by_family.get(family, [])[:limit]]
        vals = sorted(self.qualities.values(), key=lambda q: q.quality, reverse=True)
        return [q.condition for q in vals[:limit]]

    def quality(self, condition: str) -> float:
        q = self.qualities.get(condition)
        return float(q.quality) if q else 0.0
