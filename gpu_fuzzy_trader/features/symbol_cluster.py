"""
symbol_cluster.py — Per-symbol clustering for Phase 2 island scheduling using feature profiles.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from gpu_fuzzy_trader import config

logger = logging.getLogger(__name__)

SYMBOL_CLUSTERS_PATH = os.path.join(config.OUTPUTS_DIR, "symbol_clusters.json")


def _feature_names_union(
    feature_infos_long: list[dict],
    feature_infos_short: list[dict],
) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for fi in list(feature_infos_long) + list(feature_infos_short):
        name = str(fi.get("name", ""))
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _feature_profile_block(
    train_df: pd.DataFrame,
    symbols: list[str],
    feature_names: list[str],
) -> np.ndarray:
    """Per-symbol mean of selected features."""
    rows: list[np.ndarray] = []
    usable = [c for c in feature_names if c in train_df.columns]
    if not usable:
        return np.zeros((len(symbols), 1), dtype=np.float64)
    for sym in symbols:
        sym_df = train_df[train_df["symbol"].astype(str) == str(sym)]
        if sym_df.empty:
            rows.append(np.zeros(len(usable), dtype=np.float64))
            continue
        rows.append(sym_df[usable].mean(
            numeric_only=True).to_numpy(dtype=np.float64))
    return np.vstack(rows)


def build_hybrid_symbol_clusters(
    train_df: pd.DataFrame,
    feature_infos_long: list[dict],
    feature_infos_short: list[dict],
    n_clusters: int | None = None,
    random_state: int | None = None,
    balance: bool = True,
) -> dict[str, Any]:
    """
    Cluster symbols by feature profile embedding.

    When ``balance=True`` (default), symbols are greedily assigned to K clusters
    so that no cluster exceeds ``ceil(n_symbols / k)`` symbols.  This prevents
    the degenerate 1-symbol cluster that overfits (Fix E1).

    Returns dict with keys ``clusters`` (id -> symbol list), ``method``, ``n_clusters``.
    """
    if "symbol" not in train_df.columns:
        raise ValueError("train_df must contain a 'symbol' column")

    symbols = sorted(train_df["symbol"].dropna().astype(str).unique().tolist())
    if not symbols:
        raise ValueError("train_df has no symbols")

    k_target = int(
        n_clusters if n_clusters is not None else config.PHASE2_N_CLUSTERS)
    if len(symbols) <= 1:
        return {
            "method": "hybrid_v1",
            "n_clusters": 1,
            "clusters": {"0": symbols},
            "symbols": symbols,
        }

    k = min(k_target, len(symbols))
    feature_names = _feature_names_union(
        feature_infos_long, feature_infos_short)
    block_a = _feature_profile_block(train_df, symbols, feature_names)
    embedding = block_a
    embedding = np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)

    if k == len(symbols):
        clusters = {str(i): [symbols[i]] for i in range(len(symbols))}
        return {
            "method": "hybrid_v1",
            "n_clusters": k,
            "clusters": clusters,
            "symbols": symbols,
        }

    scaled = StandardScaler().fit_transform(embedding)
    seed = int(random_state if random_state is not None else config.PHASE2_SEED)
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(scaled)

    if balance:
        # ── Balanced greedy assignment ──────────────────────────────────
        # Compute distance from each symbol to each centroid.
        centroids = kmeans.cluster_centers_
        dists = pairwise_distances(scaled, centroids)  # (n_symbols, k)

        # Compute the most balanced possible per-cluster target sizes.
        # E.g. 10 symbols, K=3 → targets [4, 3, 3] → min cluster = 3.
        n_sym = len(symbols)
        base = n_sym // k
        remainder = n_sym % k
        targets = [base + 1 if i < remainder else base for i in range(k)]

        # Sort symbols by their minimum distance to any centroid (certainty).
        min_dists = dists.min(axis=1)
        sorted_indices = np.argsort(min_dists)

        cluster_counts: list[int] = [0] * k
        clusters: dict[str, list[str]] = {str(i): [] for i in range(k)}

        for idx in sorted_indices:
            sym = symbols[idx]
            # Try clusters in order of increasing distance.
            for ci in np.argsort(dists[idx]):
                if cluster_counts[int(ci)] < targets[int(ci)]:
                    clusters[str(int(ci))].append(sym)
                    cluster_counts[int(ci)] += 1
                    break
            else:
                # All clusters at target — assign to the nearest anyway.
                ci = int(np.argmin(dists[idx]))
                clusters[str(ci)].append(sym)
                cluster_counts[ci] += 1
    else:
        labels = kmeans.predict(scaled)
        clusters: dict[str, list[str]] = {str(i): [] for i in range(k)}
        for sym, lab in zip(symbols, labels):
            clusters[str(int(lab))].append(sym)

    # Drop empty keys from bad fits
    clusters = {cid: syms for cid, syms in clusters.items() if syms}

    logger.info(
        "symbol_cluster: K=%d assignment=%s",
        len(clusters),
        {cid: syms for cid, syms in clusters.items()},
    )
    return {
        "method": "hybrid_v1",
        "n_clusters": len(clusters),
        "clusters": clusters,
        "symbols": symbols,
    }


def persist_symbol_clusters(path: str, payload: dict[str, Any]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def load_symbol_clusters(path: str | None = None) -> dict[str, Any] | None:
    p = path or SYMBOL_CLUSTERS_PATH
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)
