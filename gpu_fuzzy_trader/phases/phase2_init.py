"""
phase2_init.py — Sparsity-guided stratified population initialization for Phase 2.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from gpu_fuzzy_trader import config as _cfg

Stratum = Literal["elite", "explorer"]


def build_uniform_feature_probs(feature_specs: list[dict]) -> np.ndarray:
    """Return equal feature probabilities without using feature scores."""
    if not feature_specs:
        return np.array([], dtype=np.float64)
    return np.full(len(feature_specs), 1.0 / len(feature_specs), dtype=np.float64)


def _sample_indices_without_replacement(
    rng: np.random.Generator,
    n_pick: int,
    n_features: int,
    weights: np.ndarray | None,
) -> np.ndarray:
    if n_pick <= 0:
        return np.array([], dtype=np.intp)
    if n_pick >= n_features:
        return np.arange(n_features, dtype=np.intp)
    if weights is None:
        return rng.choice(n_features, size=n_pick, replace=False)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape[0] != n_features:
        raise ValueError("weights length must match n_features")
    w = np.maximum(w, 0.0)
    if w.sum() <= 0:
        w = np.ones(n_features, dtype=np.float64)
    return rng.choice(n_features, size=n_pick, replace=False, p=w / w.sum())


def _random_active_class(
    rng: np.random.Generator,
    mode: str,
    num_classes: int,
    *,
    extreme_only: bool = False,
) -> int:
    if num_classes <= 0:
        return 0
    if extreme_only and mode in ("positive", "sparse_positive"):
        return int(rng.choice([0, num_classes - 1]))
    return int(rng.integers(0, num_classes))


def sample_sparse_chromosome(
    rng: np.random.Generator,
    feature_infos: list[dict],
    dont_cares: np.ndarray,
    k: int,
    stratum: Stratum,
    feature_probs: np.ndarray,
) -> np.ndarray:
    """
    Build one chromosome with exactly *k* active genes.

    All genes start at dont_care; *k* indices are activated per stratum rules.
    """
    k = int(k)
    k = max(_cfg.MIN_CONDITIONS, min(
        k, _cfg.MAX_CONDITIONS, len(feature_infos)))
    chrom = np.array(dont_cares, dtype=np.int32, copy=True)
    n_features = len(feature_infos)

    if stratum == "explorer":
        active_indices = _sample_indices_without_replacement(
            rng, k, n_features, None,
        ).tolist()
    else:
        active_indices = _sample_indices_without_replacement(
            rng, k, n_features, feature_probs,
        ).tolist()

    for idx in active_indices:
        dc = int(dont_cares[idx])
        num_classes = dc
        mode = feature_infos[idx]["mode"]
        chrom[idx] = _random_active_class(
            rng, mode, num_classes, extreme_only=False,
        )

    return chrom


def assign_strata_to_indices(
    indices: np.ndarray,
    stratum_fractions: tuple[float, float],
    rng: np.random.Generator,
) -> list[Stratum]:
    """Assign elite / explorer labels to non-seeded population rows."""
    n = len(indices)
    if n == 0:
        return []

    elite_f, explorer_f = stratum_fractions
    total = elite_f + explorer_f
    if total <= 0:
        elite_f, explorer_f = 0.67, 0.33
    else:
        elite_f /= total
        explorer_f /= total

    n_elite = int(round(n * elite_f))
    n_explorer = n - n_elite

    labels: list[Stratum] = (
        ["elite"] * n_elite
        + ["explorer"] * n_explorer
    )
    if len(labels) < n:
        labels.extend(["elite"] * (n - len(labels)))
    elif len(labels) > n:
        labels = labels[:n]

    rng.shuffle(labels)
    return labels


def pick_active_count(rng: np.random.Generator) -> int:
    """Uniform random k in [MIN_CONDITIONS, MAX_CONDITIONS]."""
    lo = _cfg.MIN_CONDITIONS
    hi = _cfg.MAX_CONDITIONS
    return int(rng.integers(lo, hi + 1))


def _pick_inactive_index(
    rng: np.random.Generator,
    chrom: np.ndarray,
    dont_cares: np.ndarray,
    feature_probs: np.ndarray | None,
    weighted_prob: float,
) -> int | None:
    inactive = np.where(chrom == dont_cares)[0]
    if inactive.size == 0:
        return None
    if (
        feature_probs is not None
        and rng.random() < weighted_prob
    ):
        w = feature_probs[inactive]
        w = np.maximum(w, 0.0)
        if w.sum() <= 0:
            return int(rng.choice(inactive))
        return int(rng.choice(inactive, p=w / w.sum()))
    return int(rng.choice(inactive))


def _pick_active_index(
    rng: np.random.Generator,
    chrom: np.ndarray,
    dont_cares: np.ndarray,
) -> int | None:
    active = np.where(chrom != dont_cares)[0]
    if active.size == 0:
        return None
    return int(rng.choice(active))


def repair_active_count(
    chrom: np.ndarray,
    feature_infos: list[dict],
    dont_cares: np.ndarray,
    rng: np.random.Generator,
    feature_probs: np.ndarray | None = None,
    *,
    weighted_activate_prob: float | None = None,
) -> np.ndarray:
    """Enforce MIN_CONDITIONS <= active <= MAX_CONDITIONS on a copy."""
    from gpu_fuzzy_trader.phases.phase2_sparse_encoding import (
        dense_to_sparse,
        repair_sparse_slots,
        sparse_to_dense,
        use_sparse_slots,
    )

    if use_sparse_slots() and np.asarray(chrom).ndim == 1:
        sparse = dense_to_sparse(chrom, dont_cares)
        repaired = repair_sparse_slots(
            sparse,
            feature_infos,
            dont_cares,
            rng,
            feature_probs=feature_probs,
            weighted_activate_prob=weighted_activate_prob,
        )
        return sparse_to_dense(repaired, dont_cares)

    out = chrom.copy()
    weighted_activate_prob = (
        _cfg.PHASE2_MUTATION_WEIGHTED_ACTIVATE_PROB
        if weighted_activate_prob is None
        else weighted_activate_prob
    )

    active = int(np.sum(out != dont_cares))
    while active > _cfg.MAX_CONDITIONS:
        idx = _pick_active_index(rng, out, dont_cares)
        if idx is None:
            break
        out[idx] = int(dont_cares[idx])
        active -= 1

    while active < _cfg.MIN_CONDITIONS:
        idx = _pick_inactive_index(
            rng, out, dont_cares, feature_probs, weighted_activate_prob,
        )
        if idx is None:
            break
        dc = int(dont_cares[idx])
        mode = feature_infos[idx]["mode"]
        out[idx] = _random_active_class(rng, mode, dc)
        active += 1

    return out.astype(np.int32)
