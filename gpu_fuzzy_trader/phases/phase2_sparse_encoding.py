"""
Phase 2 sparse slot chromosome encoding.

Internal evolution representation: (MAX_CONDITIONS, 2) int32 arrays
  column 0 = feature index (>= 0 active, -1 = inactive padded slot)
  column 1 = gene / fuzzy class for that feature

Active count is dynamic in [MIN_CONDITIONS, MAX_CONDITIONS]; unused slots
are padded with feat_idx=-1. Pool JSON and archives still use dense K-vectors
(expanded via sparse_to_dense) for Phase 3 compatibility.
"""

from __future__ import annotations

import numpy as np

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.features.encoder import get_dont_care

INACTIVE_FEAT_IDX = -1


def use_sparse_slots() -> bool:
    return str(getattr(_cfg, "PHASE2_ENCODING", "dense")).strip().lower() == "sparse_slots"


def max_slots() -> int:
    return int(_cfg.MAX_CONDITIONS)


def is_sparse_chromosome(chrom: np.ndarray) -> bool:
    if not use_sparse_slots():
        return False
    chrom = np.asarray(chrom)
    return chrom.ndim == 2 and chrom.shape == (max_slots(), 2)


def is_sparse_batch(
    chromosomes: np.ndarray,
    *,
    n_features: int | None = None,
) -> bool:
    if not use_sparse_slots():
        return False
    chromosomes = np.asarray(chromosomes)
    slot_shape = (max_slots(), 2)
    if chromosomes.ndim == 3:
        return chromosomes.shape[1:] == slot_shape
    if chromosomes.ndim == 2:
        if chromosomes.shape != slot_shape:
            return False
        # Ambiguous: single sparse (S, 2) vs dense batch (B=max_slots, K=n_features)
        if np.any(chromosomes[:, 0] == INACTIVE_FEAT_IDX):
            return True
        if n_features is not None and chromosomes.shape[1] == n_features:
            return False
        return True
    return False


def empty_slots() -> np.ndarray:
    slots = np.full((max_slots(), 2), INACTIVE_FEAT_IDX, dtype=np.int32)
    slots[:, 1] = 0
    return slots


def count_active_slots(slots: np.ndarray) -> int:
    return int(np.sum(np.asarray(slots)[:, 0] >= 0))


def chromosome_key(chrom: np.ndarray) -> tuple[int, ...]:
    """Hashable key for metrics caches (works for dense or sparse)."""
    return tuple(int(v) for v in np.asarray(chrom, dtype=np.int32).ravel().tolist())


def canonicalize_slots(slots: np.ndarray) -> np.ndarray:
    """Sort active slots by feature index; pad inactive at the end."""
    slots = np.asarray(slots, dtype=np.int32).reshape(-1, 2)
    out = empty_slots()
    active = slots[slots[:, 0] >= 0]
    if active.size == 0:
        return out

    order = np.argsort(active[:, 0])
    active = active[order]
    seen: set[int] = set()
    uniq_rows: list[list[int]] = []
    for row in active:
        fi = int(row[0])
        if fi in seen:
            continue
        seen.add(fi)
        uniq_rows.append([fi, int(row[1])])

    n = min(len(uniq_rows), max_slots())
    if n:
        out[:n] = np.asarray(uniq_rows[:n], dtype=np.int32)
    return out


def dense_to_sparse(
    dense: np.ndarray,
    dont_cares: np.ndarray,
) -> np.ndarray:
    dense = np.asarray(dense, dtype=np.int32).ravel()
    rows: list[list[int]] = []
    for k, gene in enumerate(dense):
        if int(gene) != int(dont_cares[k]):
            rows.append([k, int(gene)])
    rows.sort(key=lambda r: r[0])
    out = empty_slots()
    n = min(len(rows), max_slots())
    if n:
        out[:n] = np.asarray(rows[:n], dtype=np.int32)
    return canonicalize_slots(out)


def sparse_to_dense(
    slots: np.ndarray,
    dont_cares: np.ndarray,
) -> np.ndarray:
    dense = np.array(dont_cares, dtype=np.int32, copy=True)
    for feat_idx, gene in np.asarray(slots, dtype=np.int32).reshape(-1, 2):
        fi = int(feat_idx)
        if fi < 0:
            break
        if 0 <= fi < len(dense):
            dense[fi] = int(gene)
    return dense


def sparse_hamming(a: np.ndarray, b: np.ndarray) -> int:
    """Feature-aligned Hamming: count features whose class differs (incl. presence)."""
    ca = canonicalize_slots(a)
    cb = canonicalize_slots(b)
    na = count_active_slots(ca)
    nb = count_active_slots(cb)
    dict_a = {int(ca[i, 0]): int(ca[i, 1]) for i in range(na)}
    dict_b = {int(cb[i, 0]): int(cb[i, 1]) for i in range(nb)}
    keys = set(dict_a) | set(dict_b)
    return sum(1 for k in keys if dict_a.get(k) != dict_b.get(k))


def compute_rule_signals_numpy(
    data_matrix: np.ndarray,
    slots: np.ndarray,
) -> np.ndarray:
    """CPU rule matching for one sparse chromosome. Returns (N,) bool."""
    signals = np.ones(len(data_matrix), dtype=bool)
    for feat_idx, gene in np.asarray(slots, dtype=np.int32).reshape(-1, 2):
        fi = int(feat_idx)
        if fi < 0:
            break
        signals &= data_matrix[:, fi] == int(gene)
    return signals


def _clamp_slot_gene(
    feat_idx: int,
    gene: int,
    feature_infos: list[dict],
    dont_cares: np.ndarray,
    rng: np.random.Generator,
) -> int:
    from gpu_fuzzy_trader.phases.phase2_init import _random_active_class

    dc = int(dont_cares[feat_idx])
    mode = feature_infos[feat_idx]["mode"]
    g = int(gene)
    if g < 0 or g >= dc:
        return _random_active_class(rng, mode, dc)
    return g


def repair_sparse_slots(
    slots: np.ndarray,
    feature_infos: list[dict],
    dont_cares: np.ndarray,
    rng: np.random.Generator,
    feature_probs: np.ndarray | None = None,
    *,
    weighted_activate_prob: float | None = None,
) -> np.ndarray:
    """Enforce MIN/MAX active slots, valid indices, and unique features."""
    from gpu_fuzzy_trader.phases.phase2_init import (
        _random_active_class,
    )

    if weighted_activate_prob is None:
        weighted_activate_prob = _cfg.PHASE2_MUTATION_WEIGHTED_ACTIVATE_PROB

    k_features = len(feature_infos)
    out = canonicalize_slots(slots)

    # Clamp genes and drop invalid feature indices.
    cleaned: list[list[int]] = []
    for feat_idx, gene in out:
        fi = int(feat_idx)
        if fi < 0:
            break
        if fi >= k_features:
            continue
        g = _clamp_slot_gene(fi, int(gene), feature_infos, dont_cares, rng)
        cleaned.append([fi, g])
    out = empty_slots()
    n = min(len(cleaned), max_slots())
    if n:
        out[:n] = np.asarray(cleaned[:n], dtype=np.int32)
    out = canonicalize_slots(out)

    used = {int(out[i, 0]) for i in range(count_active_slots(out))}

    while count_active_slots(out) > _cfg.MAX_CONDITIONS:
        active_idx = [i for i in range(max_slots()) if out[i, 0] >= 0]
        if not active_idx:
            break
        pick = int(rng.choice(active_idx))
        old_fi = int(out[pick, 0])
        out[pick, 0] = INACTIVE_FEAT_IDX
        out[pick, 1] = 0
        used.discard(old_fi)

    while count_active_slots(out) < _cfg.MIN_CONDITIONS:
        available = [i for i in range(k_features) if i not in used]
        if not available:
            break
        if (
            feature_probs is not None
            and rng.random() < weighted_activate_prob
        ):
            w = feature_probs[np.asarray(available, dtype=np.intp)]
            w = np.maximum(w, 0.0)
            if w.sum() <= 0:
                fi = int(rng.choice(available))
            else:
                fi = int(rng.choice(available, p=w / w.sum()))
        else:
            fi = int(rng.choice(available))

        inactive_pos = next(
            (i for i in range(max_slots()) if out[i, 0] < 0),
            None,
        )
        if inactive_pos is None:
            break
        dc = int(dont_cares[fi])
        out[inactive_pos, 0] = fi
        out[inactive_pos, 1] = _random_active_class(
            rng, feature_infos[fi]["mode"], dc,
        )
        used.add(fi)

    return canonicalize_slots(out)


def sample_sparse_slots_chromosome(
    rng: np.random.Generator,
    feature_infos: list[dict],
    dont_cares: np.ndarray,
    k: int,
    stratum: str,
    feature_probs: np.ndarray,
) -> np.ndarray:
    """Build one sparse-slot chromosome with exactly *k* active conditions."""
    from gpu_fuzzy_trader.phases.phase2_init import (
        _random_active_class,
        _sample_indices_without_replacement,
    )

    k = int(k)
    k = max(_cfg.MIN_CONDITIONS, min(
        k, _cfg.MAX_CONDITIONS, len(feature_infos)))
    out = empty_slots()

    if stratum == "explorer":
        active_indices = _sample_indices_without_replacement(
            rng, k, len(feature_infos), None,
        )
    else:
        active_indices = _sample_indices_without_replacement(
            rng, k, len(feature_infos), feature_probs,
        )

    for slot_i, feat_idx in enumerate(active_indices[:max_slots()]):
        fi = int(feat_idx)
        dc = int(dont_cares[fi])
        out[slot_i, 0] = fi
        out[slot_i, 1] = _random_active_class(
            rng, feature_infos[fi]["mode"], dc,
        )
    return canonicalize_slots(out)


def crossover_sparse(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform crossover on slot rows (each slot is one condition)."""
    s = max_slots()
    mask = rng.random(s) < 0.5
    child_a = np.where(mask[:, None], parent_a, parent_b).astype(np.int32)
    child_b = np.where(mask[:, None], parent_b, parent_a).astype(np.int32)
    return child_a, child_b


def mutate_sparse(
    chromosome: np.ndarray,
    feature_infos: list[dict],
    dont_cares: np.ndarray,
    rng: np.random.Generator,
    mutation_rate: float = 0.1,
    feature_probs: np.ndarray | None = None,
    *,
    weighted_activate_prob: float | None = None,
) -> np.ndarray:
    """Mutate sparse slots; repair enforces dynamic MIN/MAX conditions."""
    from gpu_fuzzy_trader.phases.phase2_init import _random_active_class

    if weighted_activate_prob is None:
        weighted_activate_prob = _cfg.PHASE2_MUTATION_WEIGHTED_ACTIVATE_PROB

    child = np.array(chromosome, dtype=np.int32, copy=True)
    k_features = len(feature_infos)
    used = {
        int(child[i, 0])
        for i in range(max_slots())
        if child[i, 0] >= 0
    }

    # C5: Symbol gene dont_care bias — force symbol-gene slots to inactive
    symbol_gene_prob = float(getattr(_cfg, "PHASE2_SYMBOL_GENE_DONT_CARE_PROB", 0.4))
    symbol_feat_idxs = frozenset(
        fi_idx for fi_idx, fi in enumerate(feature_infos)
        if "symbol" in str(fi.get("name", "")).lower()
    )

    for slot_i in range(max_slots()):
        fi = int(child[slot_i, 0])
        # C5 bias: if active slot is a symbol feature, force inactive with probability
        if fi >= 0 and fi in symbol_feat_idxs:
            if rng.random() < symbol_gene_prob:
                child[slot_i, 0] = INACTIVE_FEAT_IDX
                child[slot_i, 1] = 0
                used.discard(fi)
                continue

        if rng.random() >= mutation_rate:
            continue
        if fi < 0:
            available = [i for i in range(k_features) if i not in used]
            if not available:
                continue
            if feature_probs is not None and rng.random() < weighted_activate_prob:
                w = feature_probs[np.asarray(available, dtype=np.intp)]
                w = np.maximum(w, 0.0)
                if w.sum() <= 0:
                    new_fi = int(rng.choice(available))
                else:
                    new_fi = int(rng.choice(available, p=w / w.sum()))
            else:
                new_fi = int(rng.choice(available))
            dc = int(dont_cares[new_fi])
            child[slot_i, 0] = new_fi
            child[slot_i, 1] = _random_active_class(
                rng, feature_infos[new_fi]["mode"], dc,
            )
            used.add(new_fi)
        else:
            dc = int(dont_cares[fi])
            if rng.random() < 0.3:
                child[slot_i, 0] = INACTIVE_FEAT_IDX
                child[slot_i, 1] = 0
                used.discard(fi)
            else:
                child[slot_i, 1] = _random_active_class(
                    rng, feature_infos[fi]["mode"], dc,
                )

    return repair_sparse_slots(
        child,
        feature_infos,
        dont_cares,
        rng,
        feature_probs,
        weighted_activate_prob=weighted_activate_prob,
    )


def repair_chromosome(
    chromosome: np.ndarray,
    feature_infos: list[dict],
    dont_cares: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Repair dense or sparse chromosome based on shape."""
    if is_sparse_chromosome(chromosome):
        if rng is None:
            rng = np.random.default_rng(_cfg.PHASE2_SEED)
        return repair_sparse_slots(
            chromosome, feature_infos, dont_cares, rng,
        )
    out = np.asarray(chromosome, dtype=np.int32).copy()
    for k, fi in enumerate(feature_infos):
        dc = int(dont_cares[k])
        num_classes = dc
        v = int(np.rint(out[k]))
        if v == dc or v >= num_classes:
            out[k] = dc
        elif v < 0:
            out[k] = 0
        else:
            out[k] = v
    return out.astype(np.int32)


def repair_population(
    population: np.ndarray,
    feature_infos: list[dict],
    dont_cares: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(_cfg.PHASE2_SEED)
    return np.stack(
        [
            repair_chromosome(population[i], feature_infos, dont_cares, rng)
            for i in range(len(population))
        ],
        axis=0,
    )
