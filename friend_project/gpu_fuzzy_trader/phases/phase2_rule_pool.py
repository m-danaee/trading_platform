
from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.features.encoder import decode_chromosome, get_dont_care
from gpu_fuzzy_trader.log_progress import maybe_log_generation
from gpu_fuzzy_trader.reporting.reporter import Reporter
from gpu_fuzzy_trader.rules.candidate_generation import augment_phase2_pool_with_generated_candidates

logger = logging.getLogger(__name__)


def trade_support_penalty(executed: int) -> float:
    """Graduated penalty when executed trades fall below MIN_TRADE_SUPPORT."""
    if executed >= _cfg.MIN_TRADE_SUPPORT:
        return 0.0
    shortfall = (_cfg.MIN_TRADE_SUPPORT - executed) / _cfg.MIN_TRADE_SUPPORT
    return min(shortfall ** 2 * _cfg.SUPPORT_PENALTY_MAX, _cfg.SUPPORT_PENALTY_MAX)



_POOL_PATHS = {
    "long": _cfg.PHASE2_POOL_PATHS["long"],
    "short": _cfg.PHASE2_POOL_PATHS["short"],
}
_HISTORY_PATHS = {
    "long": _cfg.PHASE2_HISTORY_PATHS["long"],
    "short": _cfg.PHASE2_HISTORY_PATHS["short"],
}
_ARCHIVE_PATHS = dict(_cfg.PHASE2_ARCHIVE_PATHS)


try:
    from evox.core import Algorithm as EvoxAlgorithm                
    _EVOX_AVAILABLE = True
except ImportError:
    EvoxAlgorithm = None                                  
    _EVOX_AVAILABLE = False



def _get_dont_cares(feature_infos: list[dict]) -> np.ndarray:
    """Return array of dont_care sentinels for each feature."""
    return np.array([get_dont_care(fi["mode"]) for fi in feature_infos], dtype=np.int32)


def _count_active_conditions(chromosome: np.ndarray, dont_cares: np.ndarray) -> int:
    """Count genes that are NOT dont_care (i.e. active conditions)."""
    return int(np.sum(chromosome != dont_cares))


def _hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Hamming distance between two integer arrays."""
    return int(np.sum(a != b))


def _pareto_sortino_stats(
    pareto_indices: list[int],
    metrics_cache: list[dict],
) -> dict[str, float]:
    """Aggregate raw Sortino Ratio over the current Pareto front."""
    if not pareto_indices:
        return {"mean_sortino_ratio": 0.0, "best_sortino_ratio": 0.0}
    returns = [
        float(metrics_cache[i].get("sortino_ratio",
              metrics_cache[i].get("total_return_pct", 0.0)))
        for i in pareto_indices
    ]
    return {
        "mean_sortino_ratio": float(np.mean(returns)),
        "best_sortino_ratio": float(np.max(returns)),
    }


_pareto_return_stats = _pareto_sortino_stats


def _sample_df(df: pd.DataFrame, total_rows: int) -> pd.DataFrame:
    """
    Sample up to *total_rows* rows from *df*, distributed equally across symbols.

    If a symbol has fewer rows than its share, all its rows are used.
    """
    if "symbol" not in df.columns:
        return df.sample(n=min(total_rows, len(df)), random_state=42)

    symbols = df["symbol"].unique()
    n_sym = len(symbols)
    rows_per_sym = max(1, total_rows // n_sym)

    parts = []
    for sym in symbols:
        sym_df = df[df["symbol"] == sym]
        n = min(rows_per_sym, len(sym_df))
        parts.append(sym_df.sample(n=n, random_state=42))

    return pd.concat(parts, ignore_index=True)



def _evaluate_chromosome(
    chromosome: np.ndarray,
    dont_cares: np.ndarray,
    engine,                                          
    pareto_front: list[np.ndarray],
) -> tuple[np.ndarray, dict]:
    """
    Evaluate a single chromosome and return (objectives, metrics).

    objectives = [f1, f2, f3] (all minimised, with penalties applied).
    """
    active = _count_active_conditions(chromosome, dont_cares)

    cond_penalty = 0.0
    if active < _cfg.MIN_CONDITIONS:
        cond_penalty = (_cfg.MIN_CONDITIONS - active) * 10.0
    elif active > _cfg.MAX_CONDITIONS:
        cond_penalty = (active - _cfg.MAX_CONDITIONS) * 10.0

    try:
        metrics_list = engine.simulate_rule_batch(
            chromosomes=chromosome[None, :],
            tp=_cfg.PHASE2_TP,
            sl=_cfg.PHASE2_SL,
            capital_pct=_cfg.PHASE2_CAPITAL_PCT,
        )
        metrics = metrics_list[0]
    except Exception as exc:
        logger.debug("simulate_rule_batch failed: %s", exc)
        metrics = {
            "sortino_ratio": 0.0,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 100.0,
            "win_rate": 0.0,
            "executed_trades": 0,
        }

    sortino_ratio = float(metrics.get(
        "sortino_ratio", metrics.get("total_return_pct", 0.0)))
    max_dd = float(metrics.get("max_drawdown_pct", 100.0))
    win_rate = float(metrics.get("win_rate", 0.0))
    executed = int(metrics.get("executed_trades", 0))

    support_penalty = trade_support_penalty(executed)

    diversity_penalty = 0.0
    if pareto_front:
        min_hamming = min(_hamming_distance(chromosome, pf)
                          for pf in pareto_front)
        if min_hamming == 0:
            diversity_penalty = 5.0                                   

    f1 = -sortino_ratio + support_penalty + diversity_penalty + cond_penalty
    f2 = max_dd + support_penalty + diversity_penalty + cond_penalty
    f3 = -win_rate + support_penalty + diversity_penalty + cond_penalty

    objectives = np.array([f1, f2, f3], dtype=np.float64)
    return objectives, metrics



def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Return True if solution *a* dominates *b* (all objectives ≤, at least one <)."""
    return bool(np.all(a <= b) and np.any(a < b))


def _non_dominated_sort(objectives: np.ndarray) -> list[list[int]]:
    """
    NSGA-II non-dominated sorting.

    Parameters
    ----------
    objectives : np.ndarray
        Shape (N, M) — N solutions, M objectives (all minimised).

    Returns
    -------
    list[list[int]]
        Fronts in order: fronts[0] is the Pareto front, fronts[1] is the
        second front, etc.  Always returns at least one (possibly empty) front.
    """
    n = len(objectives)
    if n == 0:
        return [[]]

    domination_count = np.zeros(n, dtype=int)
    dominated_by: list[list[int]] = [[] for _ in range(n)]
    first_front: list[int] = []

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if _dominates(objectives[i], objectives[j]):
                dominated_by[i].append(j)
            elif _dominates(objectives[j], objectives[i]):
                domination_count[i] += 1
        if domination_count[i] == 0:
            first_front.append(i)

    fronts: list[list[int]] = [first_front]
    current_front = 0

    while fronts[current_front]:
        next_front: list[int] = []
        for i in fronts[current_front]:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        current_front += 1
        if next_front:
            fronts.append(next_front)
        else:
            break

    return fronts


def _crowding_distance(objectives: np.ndarray, front: list[int]) -> np.ndarray:
    """
    Compute crowding distance for solutions in *front*.

    Parameters
    ----------
    objectives : np.ndarray
        Shape (N, M).
    front : list[int]
        Indices of solutions in this front.

    Returns
    -------
    np.ndarray
        Shape (len(front),) crowding distances.
    """
    n = len(front)
    if n <= 2:
        return np.full(n, np.inf)

    distances = np.zeros(n)
    front_obj = objectives[front]          
    M = front_obj.shape[1]

    for m in range(M):
        order = np.argsort(front_obj[:, m])
        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf
        obj_range = front_obj[order[-1], m] - front_obj[order[0], m]
        if obj_range == 0:
            continue
        for k in range(1, n - 1):
            distances[order[k]] += (
                front_obj[order[k + 1], m] - front_obj[order[k - 1], m]
            ) / obj_range

    return distances



def _weighted_gene_choice(
    feature_info: dict,
    rng: np.random.Generator,
    num_classes: int,
) -> int:
    """Sample a valid active gene using support-analysis weights when present."""
    weights = feature_info.get("condition_sample_weights")
    if weights is not None:
        arr = np.asarray(weights, dtype=np.float64)[:num_classes]
        if arr.shape[0] == num_classes and np.isfinite(arr).all() and arr.sum() > 0:
            arr = arr / arr.sum()
            return int(rng.choice(np.arange(num_classes), p=arr))
    return int(rng.integers(0, num_classes))



def _softmax_feature_probs(feature_infos: list[dict]) -> np.ndarray:
    """Softmax probabilities from Phase 1 scores for elite sampling."""
    scores = np.asarray([float(fi.get("score", 0.0)) for fi in feature_infos], dtype=np.float64)
    if not np.isfinite(scores).all() or np.max(scores) <= 0:
        return np.full(len(feature_infos), 1.0 / max(1, len(feature_infos)))
    temp = max(float(getattr(_cfg, "PHASE2_FEATURE_SOFTMAX_TEMPERATURE", 0.35)), 1e-6)
    x = scores / (np.max(scores) + 1e-12) / temp
    x = x - np.max(x)
    p = np.exp(x)
    p = p / np.sum(p)
    return p


def _regime_feature_indices(feature_infos: list[dict]) -> list[int]:
    """Feature indices whose names look regime/volatility/trend related."""
    keywords = tuple(str(k).lower() for k in getattr(_cfg, "PHASE2_REGIME_FEATURE_KEYWORDS", ()))
    out: list[int] = []
    for i, fi in enumerate(feature_infos):
        name = str(fi.get("name", "")).lower()
        if any(k in name for k in keywords):
            out.append(i)
    return out


def _sample_active_indices(
    feature_infos: list[dict],
    rng: np.random.Generator,
    stratum: str,
) -> list[int]:
    """Sample active feature indices for one chromosome under a stratum."""
    k = len(feature_infos)
    min_c = int(_cfg.MIN_CONDITIONS)
    max_c = min(int(_cfg.MAX_CONDITIONS), k)
    n_active = int(rng.integers(min_c, max_c + 1))
    all_idx = np.arange(k)

    if stratum == "elite":
        probs = _softmax_feature_probs(feature_infos)
        return [int(i) for i in rng.choice(all_idx, size=n_active, replace=False, p=probs)]

    if stratum == "regime":
        regime_idx = _regime_feature_indices(feature_infos)
        if regime_idx:
            first = int(rng.choice(np.asarray(regime_idx, dtype=int)))
            remaining = [i for i in range(k) if i != first]
            if n_active > 1 and remaining:
                probs = _softmax_feature_probs([feature_infos[i] for i in remaining])
                extra = rng.choice(np.asarray(remaining, dtype=int), size=min(n_active - 1, len(remaining)), replace=False, p=probs)
                return [first] + [int(i) for i in extra]
            return [first]

    return [int(i) for i in rng.choice(all_idx, size=n_active, replace=False)]


def _sample_stratified_chromosome(
    feature_infos: list[dict],
    dont_cares: np.ndarray,
    rng: np.random.Generator,
    stratum: str,
) -> np.ndarray:
    """Build one chromosome for elite/explorer/regime strata."""
    chrom = dont_cares.astype(np.int32, copy=True)
    for idx in _sample_active_indices(feature_infos, rng, stratum):
        dc = int(dont_cares[idx])
        chrom[idx] = _weighted_gene_choice(feature_infos[idx], rng, dc)
    return chrom
def _init_population(
    pop_size: int,
    feature_infos: list[dict],
    rng: np.random.Generator,
    dont_care_prob: float = 0.5,
    seeded_chromosomes: np.ndarray | list[np.ndarray] | None = None,
    seed_fraction: float | None = None,
) -> np.ndarray:
    """
    Initialise a population of chromosomes.

    Each gene is either a random valid class index or the dont_care sentinel,
    chosen with probability *dont_care_prob*.

    Parameters
    ----------
    pop_size : int
    feature_infos : list[dict]
        Each dict must have "mode" key.
    rng : np.random.Generator
    dont_care_prob : float
        Probability that a gene is set to dont_care (inactive).
    seeded_chromosomes : np.ndarray | list[np.ndarray] | None
        Optional archive chromosomes to seed into the initial population.
    seed_fraction : float | None
        Fraction of the population to seed from *seeded_chromosomes*.

    Returns
    -------
    np.ndarray
        Shape (pop_size, K) int32.
    """
    K = len(feature_infos)
    dont_cares = _get_dont_cares(feature_infos)
    population = np.zeros((pop_size, K), dtype=np.int32)
    if seed_fraction is None:
        seed_fraction = _cfg.PHASE2_ARCHIVE_SEED_FRACTION

    seed_rows: list[np.ndarray] = []
    if seeded_chromosomes is not None:
        seed_array = np.asarray(seeded_chromosomes, dtype=np.int32)
        if seed_array.ndim == 1:
            seed_array = seed_array[None, :]
        if seed_array.ndim != 2:
            raise ValueError(
                "seeded_chromosomes must be a 1D or 2D array-like value")
        if seed_array.shape[1] != K:
            raise ValueError(
                f"seeded_chromosomes must have {K} genes per chromosome, got {seed_array.shape[1]}"
            )

        seen: set[tuple[int, ...]] = set()
        for row in seed_array:
            key = tuple(int(v) for v in row.tolist())
            if key in seen:
                continue
            seen.add(key)
            repaired = row.astype(np.int32, copy=True)
            for k, dc in enumerate(dont_cares):
                gene = int(repaired[k])
                if gene < 0:
                    repaired[k] = 0
                elif gene > int(dc):
                    repaired[k] = int(dc)
            seed_rows.append(repaired)

    seed_count = 0
    if seed_rows and pop_size > 0 and seed_fraction > 0:
        seed_count = min(
            pop_size,
            max(1, int(round(pop_size * seed_fraction))),
            len(seed_rows),
        )

    seeded_mask = np.zeros(pop_size, dtype=bool)
    if seed_count > 0:
        seed_positions = rng.choice(pop_size, size=seed_count, replace=False)
        for position, chrom in zip(seed_positions, seed_rows[:seed_count]):
            population[position] = chrom
        seeded_mask[seed_positions] = True

    nonseeded_positions = np.where(~seeded_mask)[0]
    if getattr(_cfg, "PHASE2_THREE_STRATA_ENABLED", False):
        elite_frac = float(getattr(_cfg, "PHASE2_STRATA_ELITE_FRAC", 0.40))
        explorer_frac = float(getattr(_cfg, "PHASE2_STRATA_EXPLORER_FRAC", 0.35))
        for pos in nonseeded_positions:
            u = rng.random()
            if u < elite_frac:
                stratum = "elite"
            elif u < elite_frac + explorer_frac:
                stratum = "explorer"
            else:
                stratum = "regime"
            population[pos] = _sample_stratified_chromosome(feature_infos, dont_cares, rng, stratum)
    else:
        for k, fi in enumerate(feature_infos):
            dc = dont_cares[k]
            num_classes = dc                           
            for i in nonseeded_positions:
                if rng.random() < dont_care_prob:
                    population[i, k] = dc
                else:
                    population[i, k] = _weighted_gene_choice(feature_infos[k], rng, num_classes)

    return population



def _crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform crossover: each gene independently chosen from either parent."""
    K = len(parent_a)
    mask = rng.random(K) < 0.5
    child_a = np.where(mask, parent_a, parent_b).astype(np.int32)
    child_b = np.where(mask, parent_b, parent_a).astype(np.int32)
    return child_a, child_b


def _mutate(
    chromosome: np.ndarray,
    feature_infos: list[dict],
    dont_cares: np.ndarray,
    rng: np.random.Generator,
    mutation_rate: float = 0.1,
) -> np.ndarray:
    """
    Mutate a chromosome in-place (returns a copy).

    Each gene is mutated with probability *mutation_rate*:
      - If currently dont_care → random valid class index
      - If currently active → dont_care or a different valid class index
    """
    child = chromosome.copy()
    K = len(child)
    for k in range(K):
        if rng.random() < mutation_rate:
            dc = dont_cares[k]
            num_classes = dc
            if child[k] == dc:
                child[k] = _weighted_gene_choice(feature_infos[k], rng, num_classes)
            else:
                if rng.random() < 0.3:
                    child[k] = dc
                else:
                    child[k] = _weighted_gene_choice(feature_infos[k], rng, num_classes)
    return child



def _metrics_dict_from_population(
    population: np.ndarray,
    metrics_cache: list[dict],
) -> dict[tuple, dict]:
    """Map chromosome tuples to cached fitness metrics."""
    out: dict[tuple, dict] = {}
    for i, met in enumerate(metrics_cache):
        if met:
            out[tuple(population[i].tolist())] = met
    return out



def _build_pool_from_archive(
    archive: list[np.ndarray],
    feature_infos: list[dict],
    dont_cares: np.ndarray,
    engine,
    metrics_by_chrom: dict[tuple, dict] | None = None,
) -> list[dict]:
    """
    Convert a list of Pareto-front chromosomes into pool JSON entries.

    Each entry:
    {
        "chromosome": [...],
        "conditions": [...],
        "objectives": {"sortino_ratio": ..., "max_drawdown_pct": ..., "win_rate": ...},
        "executed_trades": ...
    }
    """
    pool: list[dict] = []
    seen: set[tuple] = set()

    for chrom in archive:
        key = tuple(chrom.tolist())
        if key in seen:
            continue
        seen.add(key)

        active = _count_active_conditions(chrom, dont_cares)
        if active < _cfg.MIN_CONDITIONS or active > _cfg.MAX_CONDITIONS:
            continue

        metrics = None
        if metrics_by_chrom is not None:
            metrics = metrics_by_chrom.get(key)
        if metrics is None:
            try:
                metrics_list = engine.simulate_rule_batch(
                    chromosomes=chrom[None, :],
                    tp=_cfg.PHASE2_TP,
                    sl=_cfg.PHASE2_SL,
                    capital_pct=_cfg.PHASE2_CAPITAL_PCT,
                )
                metrics = metrics_list[0]
            except Exception:
                continue

        executed = int(metrics.get("executed_trades", 0))
        if executed < _cfg.MIN_TRADE_POOL_FLOOR:
            continue

        try:
            conditions = decode_chromosome(chrom, feature_infos)
        except Exception:
            continue

        if not conditions:
            continue

        pool.append({
            "chromosome": chrom.tolist(),
            "conditions": conditions,
            "objectives": {
                "sortino_ratio": float(metrics.get("sortino_ratio", metrics.get("total_return_pct", 0.0))),
                "total_return_pct": float(metrics.get("total_return_pct", 0.0)),
                "max_drawdown_pct": float(metrics.get("max_drawdown_pct", 0.0)),
                "win_rate": float(metrics.get("win_rate", 0.0)),
            },
            "executed_trades": executed,
        })

    return pool



def _rule_quality_score(entry: dict) -> float:
    """Scalar fallback score for a single Phase 2 pool/archive entry."""
    obj = entry.get("objectives", {}) if isinstance(entry, dict) else {}
    ret = float(obj.get("total_return_pct", 0.0))
    sortino = float(obj.get("sortino_ratio", ret))
    wr = float(obj.get("win_rate", 0.0))
    dd = float(obj.get("max_drawdown_pct", 100.0))
    trades = int(entry.get("executed_trades", 0))
    cv = entry.get("cv_summary", {}) if isinstance(entry.get("cv_summary", {}), dict) else {}
    cv_bonus = 0.0
    if cv:
        cv_bonus = (
            float(cv.get("worst_return_pct", 0.0))
            + 3.0 * float(cv.get("worst_profit_factor", 0.0))
            - 0.25 * float(cv.get("worst_drawdown_pct", 0.0))
        )
    return float(ret + 2.0 * sortino + 0.05 * wr - 0.25 * dd + 0.01 * trades + cv_bonus)


def _build_fallback_pool_from_population(
    population: np.ndarray,
    feature_infos: list[dict],
    dont_cares: np.ndarray,
    engine,
    metrics_by_chrom: dict[tuple, dict] | None = None,
    max_rules: int | None = None,
) -> list[dict]:
    """Build a ranked safety pool from the final evolutionary population."""
    if max_rules is None:
        max_rules = int(getattr(_cfg, "PHASE2_FALLBACK_MAX_RULES", 200))
    pool = _build_pool_from_archive(
        [np.asarray(row, dtype=np.int32) for row in population],
        feature_infos,
        dont_cares,
        engine,
        metrics_by_chrom=metrics_by_chrom,
    )
    pool.sort(key=_rule_quality_score, reverse=True)
    return pool[:max_rules]

def _archive_feature_signature(feature_infos: list[dict]) -> list[dict[str, str]]:
    """Return the ordered feature signature used to validate archive reuse."""
    return [
        {"name": fi["name"], "mode": fi["mode"]}
        for fi in feature_infos
    ]


def _read_json_payload(path: str) -> object | None:
    """Read JSON from *path* and return None when the file cannot be loaded."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def _archive_objective_vector(entry: dict) -> np.ndarray:
    """Convert an archive entry into the minimisation objectives used for ranking."""
    objectives = entry.get("objectives", {})
    sortino_ratio = float(objectives.get(
        "sortino_ratio", objectives.get("total_return_pct", 0.0)))
    return np.array(
        [
            -sortino_ratio,
            float(objectives.get("max_drawdown_pct", 0.0)),
            -float(objectives.get("win_rate", 0.0)),
        ],
        dtype=np.float64,
    )


def _is_better_archive_entry(candidate: dict, incumbent: dict) -> bool:
    """Return True when *candidate* should replace *incumbent* for the same chromosome."""
    candidate_vec = _archive_objective_vector(candidate)
    incumbent_vec = _archive_objective_vector(incumbent)
    if _dominates(candidate_vec, incumbent_vec):
        return True
    if _dominates(incumbent_vec, candidate_vec):
        return False
    return tuple(candidate_vec.tolist()) < tuple(incumbent_vec.tolist())


def _merge_archive_entries(
    entries: list[dict],
    max_size: int = _cfg.PHASE2_ARCHIVE_MAX_SIZE,
) -> list[dict]:
    """Deduplicate and rank archive entries, keeping the best *max_size* rules."""
    if not entries:
        return []

    deduped: dict[tuple, dict] = {}
    for entry in entries:
        key = tuple(entry["chromosome"])
        current = deduped.get(key)
        if current is None or _is_better_archive_entry(entry, current):
            deduped[key] = entry

    unique_entries = list(deduped.values())
    if not unique_entries:
        return []

    objectives = np.vstack([
        _archive_objective_vector(entry) for entry in unique_entries
    ])

    selected: list[int] = []
    selected_set: set[int] = set()
    if getattr(_cfg, "PHASE2_MULTI_ARCHIVE_ENABLED", False):
        per_metric = int(getattr(_cfg, "PHASE2_MULTI_ARCHIVE_PER_METRIC", 0))
        if per_metric > 0:
            for metric in getattr(_cfg, "PHASE2_MULTI_ARCHIVE_METRICS", ()):                              
                if metric == "max_drawdown_pct":
                    order = sorted(
                        range(len(unique_entries)),
                        key=lambda i: float(unique_entries[i].get("objectives", {}).get(metric, 1e9)),
                    )
                elif metric == "executed_trades":
                    order = sorted(
                        range(len(unique_entries)),
                        key=lambda i: int(unique_entries[i].get("executed_trades", 0)),
                        reverse=True,
                    )
                else:
                    order = sorted(
                        range(len(unique_entries)),
                        key=lambda i: float(unique_entries[i].get("objectives", {}).get(metric, -1e9)),
                        reverse=True,
                    )
                for idx in order[:per_metric]:
                    if idx not in selected_set and len(selected) < max_size:
                        selected.append(idx)
                        selected_set.add(idx)

    fronts = _non_dominated_sort(objectives)

    for front in fronts:
        if not front:
            continue
        front_unique = [int(i) for i in front if int(i) not in selected_set]
        if not front_unique:
            continue
        if len(selected) + len(front_unique) <= max_size:
            selected.extend(front_unique)
            selected_set.update(front_unique)
        else:
            crowding = _crowding_distance(objectives, front_unique)
            order = np.argsort(-crowding)
            need = max_size - len(selected)
            for j in order[:need]:
                idx = int(front_unique[j])
                if idx not in selected_set:
                    selected.append(idx)
                    selected_set.add(idx)
            break

    return [unique_entries[i] for i in selected[:max_size]]


def _pool_seed_chromosomes(pool: list[dict]) -> np.ndarray | None:
    """Extract deduplicated chromosomes from a Phase 2 pool for population seeding."""
    if not pool:
        return None

    rows: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    for entry in pool:
        chrom = entry.get("chromosome")
        if not isinstance(chrom, list) or not chrom:
            continue
        key = tuple(int(v) for v in chrom)
        if key in seen:
            continue
        seen.add(key)
        rows.append(np.asarray(chrom, dtype=np.int32))

    if not rows:
        return None
    return np.vstack(rows)


def _validate_archive_payload(
    payload: object,
    path: str,
    feature_infos: list[dict],
) -> None:
    """Validate the archive JSON structure and feature compatibility."""
    if not isinstance(payload, dict):
        raise ValueError(
            f"Phase 2 archive must be a JSON object, got {type(payload).__name__}: {path}"
        )

    required_keys = {"version", "direction", "feature_signature", "rules"}
    missing = required_keys - set(payload.keys())
    if missing:
        raise ValueError(f"Phase 2 archive missing keys {missing}: {path}")

    if payload["direction"] not in _ARCHIVE_PATHS:
        raise ValueError(
            f"Phase 2 archive has invalid direction {payload['direction']!r}: {path}"
        )

    expected_signature = _archive_feature_signature(feature_infos)
    if payload["feature_signature"] != expected_signature:
        raise ValueError(f"Phase 2 archive feature signature mismatch: {path}")

    if not isinstance(payload["rules"], list):
        raise ValueError(f"Phase 2 archive 'rules' must be a list: {path}")

    _validate_pool_schema(payload["rules"], path)

    dont_cares = _get_dont_cares(feature_infos)
    for i, entry in enumerate(payload["rules"]):
        chromosome = entry["chromosome"]
        if len(chromosome) != len(feature_infos):
            raise ValueError(
                f"Phase 2 archive entry {i} chromosome length mismatch: {path}"
            )
        for j, gene in enumerate(chromosome):
            if not isinstance(gene, (int, np.integer)):
                raise ValueError(
                    f"Phase 2 archive entry {i} gene {j} must be an int: {path}"
                )
            if gene < 0 or gene > int(dont_cares[j]):
                raise ValueError(
                    f"Phase 2 archive entry {i} gene {j} out of range: {path}"
                )



if _EVOX_AVAILABLE:
    class FuzzyRuleOptimizer(EvoxAlgorithm):                      
        """
        EvoX-compatible multi-objective optimizer for fuzzy rule evolution.

        Wraps the NSGA-II fallback as an EvoX Algorithm for interface
        compatibility. When EvoX is available, this class can be used
        directly with EvoX workflows.
        """

        def __init__(
            self,
            feature_infos: list[dict],
            engine,
            pop_size: int = _cfg.PHASE2_POPULATION_SIZE,
            n_generations: int = _cfg.PHASE2_GENERATIONS,
        ):
            self.feature_infos = feature_infos
            self.engine = engine
            self.pop_size = pop_size
            self.n_generations = n_generations
            self._rng = np.random.default_rng(42)
            self._population: Optional[np.ndarray] = None
            self._objectives: Optional[np.ndarray] = None

        def setup(self, key=None):
            """Initialise population with dont_care sparsity."""
            self._population = _init_population(
                self.pop_size, self.feature_infos, self._rng
            )
            dont_cares = _get_dont_cares(self.feature_infos)
            self._objectives = np.full((self.pop_size, 3), np.inf)
            self._dont_cares = dont_cares
            return self

        def step(self, state=None):
            """Run one generation step."""
            if self._population is None:
                self.setup()
            for i in range(self.pop_size):
                if np.any(np.isinf(self._objectives[i])):
                    obj, _ = _evaluate_chromosome(
                        self._population[i], self._dont_cares, self.engine, []
                    )
                    self._objectives[i] = obj
            return self



class Rule_Pool_Generator:
    """
    Phase 2: GPU-accelerated multi-objective evolutionary rule pool generation.

    Generates separate Pareto-front rule pools for long and short directions.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training split DataFrame (already prepared by Data_Loader + Data_Splitter).
    feature_infos : list[dict]
        Output of Feature_Selector.select_features(): list of
        {"name": str, "mode": str, "score": float}.
    direction : str
        "long" or "short".
    pop_size : int, optional
        Override PHASE2_POPULATION_SIZE (useful for testing).
    n_generations : int, optional
        Override PHASE2_GENERATIONS (useful for testing).
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        feature_infos: list[dict],
        direction: str,
        pop_size: int | None = None,
        n_generations: int | None = None,
        seed: int = 42,
    ) -> None:
        if direction not in ("long", "short"):
            raise ValueError(
                f"direction must be 'long' or 'short', got {direction!r}")
        if not feature_infos:
            raise ValueError("feature_infos must not be empty.")

        self.direction = direction
        self.feature_infos = feature_infos
        self.pop_size = pop_size if pop_size is not None else _cfg.PHASE2_POPULATION_SIZE
        self.n_generations = n_generations if n_generations is not None else _cfg.PHASE2_GENERATIONS
        self.seed = seed
        self._feature_signature = _archive_feature_signature(feature_infos)

        sampled = _sample_df(train_df, int(getattr(_cfg, "PHASE2_ROW_SAMPLE_TOTAL", _cfg.PHASE1_SAMPLING_TOTAL)))
        feature_names = [fi["name"] for fi in feature_infos]
        from gpu_fuzzy_trader.backtest.df_slim import slim_backtest_df

        self._train_df = slim_backtest_df(sampled, feature_names)

        self._feature_modes = {fi["name"]: fi["mode"] for fi in feature_infos}

        if getattr(_cfg, "CONDITION_SUPPORT_ENABLED", False):
            try:
                from gpu_fuzzy_trader.features.condition_support import (
                    attach_condition_support_weights,
                    build_condition_support_table,
                    summarise_support_table,
                    write_condition_support_report,
                )
                support_table = build_condition_support_table(self._train_df, self.feature_infos)
                self.feature_infos = attach_condition_support_weights(
                    self.feature_infos, support_table
                )
                summary = summarise_support_table(support_table)
                report_path = write_condition_support_report(
                    support_table, self.direction, _cfg.REPORTS_DIR
                )
                logger.info(
                    "Phase 2 [%s]: condition support analysis rows=%d "
                    "dead=%d ultra_rare=%d rare=%d good=%d broad=%d very_broad=%d report=%s",
                    self.direction, summary.rows, summary.dead, summary.ultra_rare,
                    summary.rare, summary.good, summary.broad, summary.very_broad,
                    report_path,
                )
            except Exception as exc:
                logger.warning(
                    "Phase 2 [%s]: condition support analysis failed; falling back to uniform sampling: %s",
                    self.direction, exc,
                )

        self._engine = self._build_engine()

        from gpu_fuzzy_trader._gpu_runtime import configure_phase2_gpu_runtime
        from gpu_fuzzy_trader._memory import log_memory_rss

        configure_phase2_gpu_runtime(self._engine)
        log_memory_rss(f"Phase2 [{direction}] engine init")


    def _build_engine(self):
        """Build GPUBacktestEngine if JAX available, else CPUBacktestEngine."""
        try:
            from gpu_fuzzy_trader.backtest.gpu_engine import GPUBacktestEngine
            engine = GPUBacktestEngine(
                self._train_df,
                self._feature_modes,
                self.direction,
            )
            logger.info(
                "Phase 2 using GPUBacktestEngine (backend: %s)", engine.backend)
            return engine
        except ImportError:
            logger.warning(
                "JAX not available; falling back to CPUBacktestEngine for Phase 2.")
            from gpu_fuzzy_trader.backtest.cpu_engine import CPUBacktestEngine
            return CPUBacktestEngine(
                self._train_df,
                self._feature_modes,
                self.direction,
                feature_infos=self.feature_infos,
            )


    def run(self) -> list[dict]:
        """
        Run the evolutionary search and return the Pareto-front pool.

        Also persists results to:
                    pools/phase2_{direction}_pool.json
                    pools/phase2_{direction}_history.json

        Returns
        -------
        list[dict]
            Pareto-front rules in pool JSON schema.
        """
        rng = np.random.default_rng(self.seed)

        logger.info(
            "Phase 2 [%s]: pop=%d, gen=%d, features=%d",
            self.direction, self.pop_size, self.n_generations, len(
                self.feature_infos)
        )

        from gpu_fuzzy_trader.evolution.evox_runner import run_phase2_evolution

        logger.info(
            "Phase 2 [%s]: algorithm=%s",
            self.direction, _cfg.PHASE2_ALGORITHM,
        )

        previous_pool: list[dict] = []
        try:
            loaded_pool = Rule_Pool_Generator.load_pool(self.direction)
            if loaded_pool:
                previous_pool = loaded_pool
        except ValueError:
            logger.warning(
                "Phase 2 [%s]: existing pool file invalid; starting without seeds",
                self.direction,
            )

        seed_chromosomes = _pool_seed_chromosomes(previous_pool)
        if seed_chromosomes is not None:
            seed_slots = min(
                self.pop_size,
                max(1, int(round(
                    self.pop_size * _cfg.PHASE2_ARCHIVE_SEED_FRACTION))),
                len(seed_chromosomes),
            )
            logger.info(
                "Phase 2 [%s]: seeding %d/%d population slots (%.0f%%) from "
                "pool with %d unique chromosomes",
                self.direction,
                seed_slots,
                self.pop_size,
                _cfg.PHASE2_ARCHIVE_SEED_FRACTION * 100.0,
                len(seed_chromosomes),
            )

        progress_tag = "Phase 2 [%s] NSGA-III" % self.direction
        new_pool, history = run_phase2_evolution(
            feature_infos=self.feature_infos,
            engine=self._engine,
            pop_size=self.pop_size,
            n_generations=self.n_generations,
            rng=rng,
            log_tag=progress_tag,
            seed_chromosomes=seed_chromosomes,
        )

        pool = _merge_archive_entries(previous_pool + list(new_pool))
        try:
            before_aug = len(pool)
            pool = augment_phase2_pool_with_generated_candidates(
                pool,
                engine=self._engine,
                train_df=self._train_df,
                direction=self.direction,
                rng_seed=self.seed,
            )
            if len(pool) != before_aug:
                logger.info(
                    "Phase 2 [%s]: robust v5 candidate generation augmented pool %d -> %d",
                    self.direction, before_aug, len(pool),
                )
        except Exception as exc:
            logger.warning("Phase 2 [%s]: robust v5 candidate generation skipped: %s", self.direction, exc)
        if getattr(_cfg, "PHASE2_FALLBACK_ENABLED", False) and len(pool) < int(getattr(_cfg, "PHASE2_FALLBACK_MIN_POOL_SIZE", 40)):
            logger.warning(
                "Phase 2 [%s]: pool too small after Pareto merge (%d); fallback entries from evolution were requested upstream.",
                self.direction, len(pool),
            )
        pool = self._apply_purged_cv_filter(pool)
        logger.info(
            "Phase 2 [%s]: merged pool %d previous + %d new → %d retained",
            self.direction,
            len(previous_pool),
            len(new_pool),
            len(pool),
        )

        pool_path = _POOL_PATHS[self.direction]
        history_path = _HISTORY_PATHS[self.direction]
        pool_dir = os.path.dirname(pool_path)
        history_dir = os.path.dirname(history_path)
        if pool_dir:
            os.makedirs(pool_dir, exist_ok=True)
        if history_dir and history_dir != pool_dir:
            os.makedirs(history_dir, exist_ok=True)

        with open(pool_path, "w", encoding="utf-8") as fh:
            json.dump(pool, fh, indent=2)
        with open(history_path, "w", encoding="utf-8") as fh:
            json.dump(history, fh, indent=2)

        logger.info(
            "Phase 2 [%s]: pool_size=%d, saved to %s",
            self.direction, len(pool), pool_path
        )

        try:
            reporter = Reporter()
            reporter.plot_phase2_metrics(history, self.direction)
            reporter.plot_phase2_pnl(history, self.direction)
        except Exception as exc:
            logger.warning(
                "Reporter Phase 2 plots failed (non-fatal): %s", exc)

        try:
            saved = Rule_Pool_Generator.save_archive(
                self.direction, self.feature_infos, pool
            )
            logger.info(
                "Phase 2 [%s]: archive saved with %d rules to %s",
                self.direction, len(saved), _ARCHIVE_PATHS[self.direction],
            )
        except Exception as exc:
            logger.warning(
                "Phase 2 [%s]: archive save failed (non-fatal): %s",
                self.direction, exc,
            )

        self._release_resources()
        return pool


    def _apply_purged_cv_filter(self, pool: list[dict]) -> list[dict]:
        """Annotate and filter the Phase 2 pool with robust rolling folds."""
        if not pool or not getattr(_cfg, "PHASE2_CV_FILTER_ENABLED", False):
            return pool
        try:
            from gpu_fuzzy_trader.validation.rolling_cv import (
                build_fold_engines,
                evaluate_rule_set_on_fold_engines,
            )
            max_eval = int(getattr(_cfg, "PHASE2_CV_MAX_RULES_TO_EVALUATE", len(pool)))
            ranked = sorted(pool, key=_rule_quality_score, reverse=True)
            selected = ranked[:max_eval]
            rest = ranked[max_eval:]
            feature_names = [fi["name"] for fi in self.feature_infos]
            fold_engines = build_fold_engines(
                self._train_df,
                self.direction,
                feature_names=feature_names,
            )
            if not fold_engines:
                return pool

            kept: list[dict] = []
            rejected = 0
            for entry in selected:
                rule = {
                    "conditions": entry["conditions"],
                    "tp": float(entry.get("tp", _cfg.PHASE2_TP)),
                    "sl": float(entry.get("sl", _cfg.PHASE2_SL)),
                    "capital_pct": float(entry.get("capital_pct", _cfg.PHASE2_CAPITAL_PCT)),
                }
                summary = evaluate_rule_set_on_fold_engines([rule], fold_engines)
                entry = dict(entry)
                entry["cv_summary"] = {
                    "folds": summary.folds,
                    "worst_return_pct": summary.worst_return_pct,
                    "worst_profit_factor": summary.worst_profit_factor,
                    "worst_sortino_ratio": summary.worst_sortino_ratio,
                    "worst_drawdown_pct": summary.worst_drawdown_pct,
                    "min_trades": summary.min_trades,
                    "mean_return_pct": summary.mean_return_pct,
                    "mean_profit_factor": summary.mean_profit_factor,
                }
                pass_filter = (
                    summary.worst_return_pct >= float(_cfg.PHASE2_CV_MIN_WORST_RETURN)
                    and summary.worst_profit_factor >= float(_cfg.PHASE2_CV_MIN_WORST_PF)
                    and summary.worst_drawdown_pct <= float(_cfg.PHASE2_CV_MAX_WORST_DD)
                    and summary.min_trades >= int(_cfg.PHASE2_CV_MIN_FOLD_TRADES)
                )
                if pass_filter:
                    kept.append(entry)
                else:
                    rejected += 1

            min_pool = int(getattr(_cfg, "PHASE2_FALLBACK_MIN_POOL_SIZE", 40))
            if len(kept) < min_pool:
                backfill = [e for e in selected if e not in kept]
                backfill.sort(key=_rule_quality_score, reverse=True)
                kept.extend(backfill[: max(0, min_pool - len(kept))])
            kept.extend(rest[: max(0, int(_cfg.PHASE2_ARCHIVE_MAX_SIZE) - len(kept))])
            kept = _merge_archive_entries(kept, max_size=int(_cfg.PHASE2_ARCHIVE_MAX_SIZE))
            logger.info(
                "Phase 2 [%s]: purged-CV annotated %d rules, rejected=%d, retained=%d",
                self.direction, len(selected), rejected, len(kept),
            )
            return kept
        except Exception as exc:
            logger.warning("Phase 2 [%s]: purged-CV filter failed (non-fatal): %s", self.direction, exc)
            return pool

    def _release_resources(self) -> None:
        """Drop engine and sampled data to free RAM before the next direction."""
        self._engine = None
        self._train_df = None
        from gpu_fuzzy_trader._memory import log_memory_rss, release_phase2_resources

        log_memory_rss(f"Phase2 [{self.direction}] after release")
        release_phase2_resources()

    @staticmethod
    def load_pool(direction: str) -> Optional[list[dict]]:
        """
        Load existing pool if valid, return None if missing.

        Parameters
        ----------
        direction : str
            "long" or "short".

        Returns
        -------
        list[dict] | None
            Loaded pool if file exists and is valid JSON, None if missing.

        Raises
        ------
        ValueError
            If the file exists but is corrupted or has an invalid schema.
        """
        if direction not in _POOL_PATHS:
            raise ValueError(
                f"direction must be 'long' or 'short', got {direction!r}")

        path = _POOL_PATHS[direction]
        if not os.path.exists(path):
            return None

        try:
            with open(path, "r", encoding="utf-8") as fh:
                pool = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(
                f"Phase 2 pool file is unreadable or corrupted: {path}"
            ) from exc

        _validate_pool_schema(pool, path)
        return pool

    @staticmethod
    def load_archive(
        direction: str,
        feature_infos: list[dict],
    ) -> Optional[dict]:
        """
        Load a compatible persistent archive if it exists, otherwise return None.

        Archive files are ignored when they are corrupt or were built from an
        incompatible feature signature.
        """
        if direction not in _ARCHIVE_PATHS:
            raise ValueError(
                f"direction must be 'long' or 'short', got {direction!r}")

        path = _ARCHIVE_PATHS[direction]
        if not os.path.exists(path):
            return None

        payload = _read_json_payload(path)
        if payload is None:
            logger.warning(
                "Phase 2 archive file is unreadable or corrupted: %s", path)
            return None

        try:
            _validate_archive_payload(payload, path, feature_infos)
        except ValueError as exc:
            logger.info("Ignoring Phase 2 archive at %s: %s", path, exc)
            return None

        rules = _merge_archive_entries(payload["rules"])
        return {
            "version": int(payload.get("version", 1)),
            "direction": direction,
            "feature_signature": _archive_feature_signature(feature_infos),
            "rules": rules,
        }

    @staticmethod
    def save_archive(
        direction: str,
        feature_infos: list[dict],
        rules: list[dict],
    ) -> list[dict]:
        """Merge the latest pool into the persistent archive and write it atomically."""
        if direction not in _ARCHIVE_PATHS:
            raise ValueError(
                f"direction must be 'long' or 'short', got {direction!r}")

        path = _ARCHIVE_PATHS[direction]
        existing_rules: list[dict] = []
        raw_payload = _read_json_payload(path)
        if raw_payload is not None:
            try:
                _validate_archive_payload(raw_payload, path, feature_infos)
            except ValueError as exc:
                logger.info(
                    "Replacing invalid Phase 2 archive at %s: %s", path, exc)
            else:
                existing_rules = list(raw_payload["rules"])

        merged = _merge_archive_entries(existing_rules + list(rules))
        if not merged:
            return []

        payload = {
            "version": 1,
            "direction": direction,
            "feature_signature": _archive_feature_signature(feature_infos),
            "rules": merged,
        }

        archive_dir = os.path.dirname(path)
        if archive_dir:
            os.makedirs(archive_dir, exist_ok=True)

        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp_path, path)
        return merged

    @staticmethod
    def skip_if_valid(direction: str) -> Optional[list[dict]]:
        """
        Return loaded pool if valid, None if need to run.

        Parameters
        ----------
        direction : str
            "long" or "short".

        Returns
        -------
        list[dict] | None
            Loaded pool if file exists and is valid, None otherwise.
        """
        try:
            return Rule_Pool_Generator.load_pool(direction)
        except ValueError:
            return None



def _validate_pool_schema(pool: object, path: str) -> None:
    """
    Validate the structure of a loaded pool JSON.

    Raises ValueError if the schema is invalid.
    """
    if not isinstance(pool, list):
        raise ValueError(
            f"Phase 2 pool must be a JSON array, got {type(pool).__name__}: {path}"
        )

    for i, entry in enumerate(pool):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Phase 2 pool entry {i} must be a dict: {path}"
            )
        required_keys = {"chromosome", "conditions",
                         "objectives", "executed_trades"}
        missing = required_keys - set(entry.keys())
        if missing:
            raise ValueError(
                f"Phase 2 pool entry {i} missing keys {missing}: {path}"
            )
        if not isinstance(entry["chromosome"], list):
            raise ValueError(
                f"Phase 2 pool entry {i} 'chromosome' must be a list: {path}"
            )
        if not isinstance(entry["conditions"], list):
            raise ValueError(
                f"Phase 2 pool entry {i} 'conditions' must be a list: {path}"
            )
        if not isinstance(entry["objectives"], dict):
            raise ValueError(
                f"Phase 2 pool entry {i} 'objectives' must be a dict: {path}"
            )
        required_obj_keys = {"sortino_ratio",
                             "max_drawdown_pct", "win_rate"}
        missing_obj = required_obj_keys - set(entry["objectives"].keys())
        if missing_obj:
            raise ValueError(
                f"Phase 2 pool entry {i} 'objectives' missing keys {missing_obj}: {path}"
            )
        if not isinstance(entry["executed_trades"], int):
            raise ValueError(
                f"Phase 2 pool entry {i} 'executed_trades' must be an int: {path}"
            )
