"""
phase2_init.py — Sparsity-guided stratified population initialization for Phase 2.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from gpu_fuzzy_trader import config as _cfg

Stratum = Literal["elite", "explorer", "regime"]


def _regime_feature_indices(feature_infos: list[dict]) -> list[int]:
    """Return indices of features whose names match PHASE2_REGIME_FEATURE_KEYWORDS.

    Matches feature names (case-insensitive, substring) against the keyword list
    from config.  Returns an empty list when no match is found.
    """
    keywords = tuple(
        str(k).lower()
        for k in getattr(_cfg, "PHASE2_REGIME_FEATURE_KEYWORDS", ())
    )
    out: list[int] = []
    for i, fi in enumerate(feature_infos):
        name = str(fi.get("name", "")).lower()
        if any(kw in name for kw in keywords):
            out.append(i)
    return out


def build_feature_sampling_probs(
    feature_infos: list[dict],
    *,
    temp: float | None = None,
    eps: float | None = None,
    mix_uniform: float | None = None,
) -> np.ndarray:
    """Softmax Phase 1 scores with optional uniform floor (length K)."""
    if not feature_infos:
        return np.array([], dtype=np.float64)

    temp = _cfg.PHASE2_INIT_SOFTMAX_TEMP if temp is None else temp
    eps = _cfg.PHASE2_INIT_SCORE_EPS if eps is None else eps
    mix_uniform = _cfg.PHASE2_INIT_UNIFORM_MIX if mix_uniform is None else mix_uniform

    scores = np.array(
        [max(float(fi.get("score", 0.0)), eps) for fi in feature_infos],
        dtype=np.float64,
    )
    if temp <= 0:
        probs = np.zeros(len(scores), dtype=np.float64)
        probs[int(np.argmax(scores))] = 1.0
    else:
        scaled = scores / temp
        scaled -= np.max(scaled)
        exp_s = np.exp(scaled)
        probs = exp_s / exp_s.sum()

    k = len(probs)
    if mix_uniform > 0 and k > 0:
        probs = (1.0 - mix_uniform) * probs + mix_uniform / k

    probs = np.maximum(probs, 0.0)
    total = probs.sum()
    if total <= 0:
        return np.ones(k, dtype=np.float64) / k
    return probs / total


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


def assign_three_strata_to_indices(
    indices: np.ndarray,
    stratum_fractions: tuple[float, float, float],
    rng: np.random.Generator,
) -> list[str]:
    """Assign elite / explorer / regime labels to non-seeded population rows.

    Parameters
    ----------
    indices : np.ndarray
        1-D array of row indices to assign strata.
    stratum_fractions : tuple[float, float, float]
        (elite_frac, explorer_frac, regime_frac).  Normalised to sum to 1.
    rng : np.random.Generator

    Returns
    -------
    list[str]
        One of ``"elite"``, ``"explorer"``, ``"regime"`` per row, shuffled.
    """
    n = len(indices)
    if n == 0:
        return []

    elite_f, explorer_f, regime_f = stratum_fractions
    total = elite_f + explorer_f + regime_f
    if total <= 0:
        elite_f, explorer_f, regime_f = 0.40, 0.35, 0.25
    else:
        elite_f /= total
        explorer_f /= total
        regime_f /= total

    n_elite = int(round(n * elite_f))
    n_explorer = int(round(n * explorer_f))
    # Ensure total matches n exactly
    allocated = n_elite + n_explorer
    if allocated + int(round(n * regime_f)) != n:
        n_regime = n - n_elite - n_explorer
    else:
        n_regime = int(round(n * regime_f))

    labels: list[str] = (
        ["elite"] * n_elite
        + ["explorer"] * n_explorer
        + ["regime"] * n_regime
    )
    if len(labels) < n:
        labels.extend(["elite"] * (n - len(labels)))
    elif len(labels) > n:
        labels = labels[:n]

    rng.shuffle(labels)
    return labels


def sample_regime_stratum_chromosome(
    rng: np.random.Generator,
    feature_infos: list[dict],
    dont_cares: np.ndarray,
    k: int,
    feature_probs: np.ndarray,
) -> np.ndarray:
    """Build one chromosome whose first active gene is a regime-feature index.

    The regime feature is picked uniformly from the keyword-matched indices.
    The remaining active-gene slots are filled from the rest of the feature set
    using the same logic as ``sample_sparse_chromosome`` (*i.e.* weighted by
    *feature_probs* for "elite" stratum, uniform for "explorer").

    If no regime feature is found (empty keyword list or no match), falls back
    to a regular explorer-stratum chromosome.
    """
    k = int(k)
    k = max(_cfg.MIN_CONDITIONS, min(
        k, _cfg.MAX_CONDITIONS, len(feature_infos)))
    chrom = np.array(dont_cares, dtype=np.int32, copy=True)

    regime_indices = _regime_feature_indices(feature_infos)

    if not regime_indices:
        # No regime features available → fall back to explorer
        active_indices = _sample_indices_without_replacement(
            rng, k, len(feature_infos), None,
        ).tolist()
    else:
        # First active gene is a regime feature
        first_idx = int(rng.choice(np.array(regime_indices, dtype=int)))
        active_indices: list[int] = [first_idx]

        remaining_k = k - 1
        if remaining_k > 0:
            other_indices = [i for i in range(len(feature_infos)) if i != first_idx]
            if remaining_k >= len(other_indices):
                extra = other_indices
            else:
                # Use weighted sampling from the remaining features
                other_probs = (
                    feature_probs[other_indices]
                    if feature_probs is not None and len(feature_probs) == len(feature_infos)
                    else None
                )
                extra_ids = _sample_indices_without_replacement(
                    rng, remaining_k, len(other_indices), other_probs,
                ).tolist()
                extra = [other_indices[i] for i in extra_ids]
            active_indices.extend(extra)

    for idx in active_indices:
        dc = int(dont_cares[idx])
        mode = feature_infos[idx]["mode"]
        chrom[idx] = _random_active_class(rng, mode, dc)

    return chrom
