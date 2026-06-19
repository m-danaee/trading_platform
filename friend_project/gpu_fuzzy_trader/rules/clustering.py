from __future__ import annotations

import random
from collections import defaultdict

import numpy as np

from gpu_fuzzy_trader.rules.feature_family import family_counts


def _metric_vector(entry: dict) -> list[float]:
    obj = entry.get("objectives", {}) if isinstance(entry, dict) else {}
    cv = entry.get("cv_summary", {}) if isinstance(entry.get("cv_summary", {}), dict) else {}
    return [
        float(obj.get("total_return_pct", 0.0)),
        -float(obj.get("max_drawdown_pct", 0.0)),
        float(obj.get("win_rate", 0.0)),
        float(entry.get("executed_trades", 0)),
        float(cv.get("worst_return_pct", 0.0)),
        float(cv.get("worst_profit_factor", 0.0)),
    ]


def assign_rule_clusters(pool: list[dict], n_clusters: int = 32) -> list[dict]:
    if not pool:
        return []
    n_clusters = max(1, min(int(n_clusters), len(pool)))
    X = np.asarray([_metric_vector(e) for e in pool], dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if X.shape[0] <= n_clusters:
        labels = np.arange(X.shape[0])
    else:
        try:
            from sklearn.cluster import KMeans
            Xs = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-9)
            labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(Xs)
        except Exception:
            scores = X[:, 0] + 5 * X[:, 5] + 0.01 * X[:, 3] + X[:, 4]
            ranks = np.argsort(np.argsort(scores))
            labels = np.floor(ranks / max(1, len(pool) / n_clusters)).astype(int)
    out = []
    for e, lab in zip(pool, labels):
        ee = dict(e)
        ee["cluster_id"] = int(lab)
        out.append(ee)
    return out


def smart_population_from_pool(pool: list[dict], pop_size: int, min_rules: int, max_rules: int, rng: random.Random) -> list[list[dict]]:
    if not pool:
        return []
    p = assign_rule_clusters(pool)
    by_cluster: dict[int, list[dict]] = defaultdict(list)
    for e in p:
        by_cluster[int(e.get("cluster_id", 0))].append(e)
    for k in by_cluster:
        by_cluster[k].sort(key=lambda e: float(e.get("objectives", {}).get("total_return_pct", 0.0)) + 3 * float(e.get("cv_summary", {}).get("worst_profit_factor", 0.0)), reverse=True)
    clusters = list(by_cluster)
    population: list[list[dict]] = []
    def unique(rs):
        out=[]; seen=set()
        for r in rs:
            key=tuple(sorted(r.get("conditions", [])))
            if key not in seen:
                out.append(r); seen.add(key)
        return out[:max_rules]
    while len(population) < pop_size:
        frac = len(population) / max(1, pop_size)
        k = rng.randint(min_rules, max_rules)
        if frac < 0.20:         
            candidates = sorted(p, key=lambda e: float(e.get("objectives", {}).get("total_return_pct", 0.0)) + 2*float(e.get("cv_summary", {}).get("worst_profit_factor", 0.0)), reverse=True)
            rs = candidates[:k]
        elif frac < 0.55 and len(clusters) >= k:                 
            rs = [rng.choice(by_cluster[c]) for c in rng.sample(clusters, k)]
        elif frac < 0.75:                              
            c = rng.choice(clusters)
            bank = by_cluster[c]
            rs = rng.sample(bank, min(k, len(bank)))
        elif frac < 0.90:          
            rs = []
            for c in rng.sample(clusters, min(len(clusters), max(1, k//2))):
                rs.append(by_cluster[c][0])
            while len(rs) < k:
                rs.append(rng.choice(p))
        else:           
            rs = rng.sample(p, min(k, len(p)))
        rs = unique(rs)
        if len(rs) >= min_rules:
            population.append(rs)
    return population
