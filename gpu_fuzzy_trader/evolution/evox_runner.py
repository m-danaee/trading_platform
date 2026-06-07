"""
evox_runner.py — Phase 2 NSGA-III multi-objective evolution.

Uses EvoX reference-vector sampling and non-dominated ranking with niche
selection on integer fuzzy-rule chromosomes. Falls back to NumPy NSGA-II
environmental selection when EvoX is unavailable.
"""

from __future__ import annotations
from gpu_fuzzy_trader.phases.phase2_rule_pool import (
    _crossover,
    _mutate,
)
from gpu_fuzzy_trader.evolution.numba_ops import (
    batch_hamming_min,
    crowding_distance,
    non_dominated_sort,
)

import copy
import logging
import time

import numpy as np

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.log_progress import maybe_log_generation

logger = logging.getLogger(__name__)

_EVOX_AVAILABLE = False
_EVOX_IMPORT_ERROR: str | None = None
try:
    import torch
    from evox.operators.sampling.uniform import uniform_sampling
    from evox.operators.selection.non_dominate import non_dominate_rank

    _EVOX_AVAILABLE = True
except ImportError as exc:
    _EVOX_IMPORT_ERROR = str(exc)
    torch = None  # type: ignore[assignment]
    uniform_sampling = None  # type: ignore[assignment]
    non_dominate_rank = None  # type: ignore[assignment]


def _build_rank_and_crowding(
    objectives: np.ndarray,
    fronts: list[list[int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Per-individual Pareto rank (lower is better) and crowding distance."""
    n = len(objectives)
    rank = np.full(n, np.inf, dtype=np.float64)
    crowding = np.zeros(n, dtype=np.float64)
    for r, front in enumerate(fronts):
        for i in front:
            rank[i] = r
        if len(front) <= 2:
            for i in front:
                crowding[i] = np.inf
        else:
            cd = crowding_distance(objectives, front)
            for j, i in enumerate(front):
                crowding[i] = cd[j]
    return rank, crowding


def _binary_tournament_pick(
    indices: list[int],
    rank: np.ndarray,
    crowding: np.ndarray,
    rng: np.random.Generator,
) -> int:
    """Pick the better of two random candidates (lower rank, higher crowding)."""
    a, b = rng.choice(indices, size=2, replace=False)
    if rank[a] < rank[b]:
        return int(a)
    if rank[b] < rank[a]:
        return int(b)
    return int(a if crowding[a] >= crowding[b] else b)


def environmental_selection_nsga2(
    merge_pop: np.ndarray,
    merge_fit: np.ndarray,
    pop_size: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Canonical NSGA-II truncation on a 2N merged population."""
    fronts = non_dominated_sort(merge_fit)
    selected: list[int] = []
    for front in fronts:
        if not front:
            continue
        if len(selected) + len(front) <= pop_size:
            selected.extend(front)
        else:
            cd = crowding_distance(merge_fit, front)
            order = np.argsort(-cd)
            need = pop_size - len(selected)
            selected.extend(int(front[j]) for j in order[:need])
            break
    selected = selected[:pop_size]
    idx = np.array(selected, dtype=np.intp)
    return merge_pop[idx], merge_fit[idx], selected


def _repair_chromosome(
    chromosome: np.ndarray,
    feature_infos: list[dict],
    dont_cares: np.ndarray,
) -> np.ndarray:
    """Clamp each gene to a valid class index or dont_care sentinel."""
    out = chromosome.copy()
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


def _repair_population(
    population: np.ndarray,
    feature_infos: list[dict],
    dont_cares: np.ndarray,
) -> np.ndarray:
    """Repair every row in a population matrix."""
    from gpu_fuzzy_trader.phases.phase2_sparse_encoding import (
        repair_population as _repair_population_sparse,
        use_sparse_slots,
    )

    if use_sparse_slots():
        return _repair_population_sparse(
            population, feature_infos, dont_cares,
        )
    return np.stack(
        [
            _repair_chromosome(population[i], feature_infos, dont_cares)
            for i in range(len(population))
        ],
        axis=0,
    )


def _update_hall_of_fame(
    hall_of_fame: dict[tuple[int, ...], np.ndarray],
    population: np.ndarray,
    pareto_indices: list[int],
) -> None:
    """Accumulate unique Pareto chromosomes discovered across generations."""
    from gpu_fuzzy_trader.phases.phase2_sparse_encoding import chromosome_key

    for i in pareto_indices:
        chrom = population[int(i)].copy()
        hall_of_fame[chromosome_key(chrom)] = chrom


def _pareto_mean_return_pct(
    pareto_indices: list[int],
    metrics_cache: list[dict],
) -> float:
    """Mean total_return_pct across the current Pareto front."""
    if not pareto_indices:
        return -100.0
    returns = [
        float(metrics_cache[i].get("total_return_pct", 0.0))
        for i in pareto_indices
    ]
    return float(np.mean(returns))


def _pareto_median_return_pct(
    pareto_indices: list[int],
    metrics_cache: list[dict],
) -> float:
    """Median total_return_pct across the current Pareto front."""
    if not pareto_indices:
        return -100.0
    returns = [
        float(metrics_cache[i].get("total_return_pct", 0.0))
        for i in pareto_indices
    ]
    return float(np.median(returns))


def _pareto_return_pct_for_early_stop(
    pareto_indices: list[int],
    metrics_cache: list[dict],
) -> float:
    if bool(_cfg.PHASE2_EARLY_STOP_USE_MEDIAN_RETURN):
        return _pareto_median_return_pct(pareto_indices, metrics_cache)
    return _pareto_mean_return_pct(pareto_indices, metrics_cache)


def _should_early_stop_phase2(
    gen: int,
    pareto_return_pct: float,
    valid_count: int,
) -> bool:
    if not _cfg.PHASE2_EARLY_STOP_ENABLED:
        return False
    if (
        str(_cfg.SPLIT_MODE).strip().lower() == "purged_rolling_cv"
        and bool(_cfg.PHASE2_EARLY_STOP_DISABLED_IN_CV)
    ):
        return False
    if gen + 1 < int(_cfg.PHASE2_EARLY_STOP_MIN_GENERATION):
        return False
    if pareto_return_pct >= float(_cfg.PHASE2_EARLY_STOP_MEAN_RETURN_PCT):
        return False
    min_valid = int(_cfg.PHASE2_EARLY_STOP_MIN_VALID_RULES)
    if valid_count >= min_valid:
        return False
    return True


def _split_mode_is_purged_cv() -> bool:
    return str(_cfg.SPLIT_MODE).strip().lower() == "purged_rolling_cv"


def _empty_eval_stats() -> dict[str, int]:
    return {"pending": 0, "cache_hits": 0}


def _merge_gen_eval_stats(*stats_list: dict[str, int] | None) -> dict[str, int]:
    merged = _empty_eval_stats()
    for stats in stats_list:
        if not stats:
            continue
        merged["pending"] += int(stats.get("pending", 0))
        merged["cache_hits"] += int(stats.get("cache_hits", 0))
    return merged


def _eval_cache_hit_rate(stats: dict[str, int]) -> float | None:
    pending = int(stats.get("pending", 0))
    if pending <= 0:
        return None
    return float(stats.get("cache_hits", 0)) / float(pending)


def _update_max_return_plateau(
    max_return_pct: float,
    best_max_return: float,
    streak: int,
) -> tuple[float, int]:
    """Return updated (best_max_return, gens_without_improvement)."""
    delta = float(_cfg.PHASE2_PLATEAU_EARLY_STOP_MIN_DELTA_PCT)
    if max_return_pct > best_max_return + delta:
        return max_return_pct, 0
    return best_max_return, streak + 1


def _should_plateau_early_stop_phase2(gen: int, streak: int) -> bool:
    if not _cfg.PHASE2_PLATEAU_EARLY_STOP_ENABLED:
        return False
    if (
        _split_mode_is_purged_cv()
        and bool(_cfg.PHASE2_PLATEAU_EARLY_STOP_DISABLED_IN_CV)
    ):
        return False
    if gen + 1 < int(_cfg.PHASE2_PLATEAU_EARLY_STOP_MIN_GENERATION):
        return False
    return streak >= int(_cfg.PHASE2_PLATEAU_EARLY_STOP_PATIENCE)


def _median_pairwise_hamming(chromosomes: list[np.ndarray]) -> float:
    """Median Hamming distance across unique Pareto chromosomes (sample cap 40)."""
    from gpu_fuzzy_trader.phases.phase2_rule_pool import _hamming_distance
    from gpu_fuzzy_trader.phases.phase2_sparse_encoding import chromosome_key

    if len(chromosomes) < 2:
        return 0.0
    uniq: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    for chrom in chromosomes:
        key = chromosome_key(chrom)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(chrom)
    if len(uniq) < 2:
        return 0.0
    if len(uniq) > 40:
        rng = np.random.default_rng(0)
        pick = rng.choice(len(uniq), size=40, replace=False)
        uniq = [uniq[int(i)] for i in pick]
    dists: list[float] = []
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            dists.append(float(_hamming_distance(uniq[i], uniq[j])))
    return float(np.median(dists)) if dists else 0.0


def _pareto_diagnostics(
    pareto_indices: list[int],
    metrics_cache: list[dict],
    pareto_obj: np.ndarray,
    population: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute compact diagnostics for Phase 2 Pareto-front health.

    - sortino_cap_hit_fraction: fraction of Pareto members whose saturated
      Sortino sits near the configured cap (signal of objective saturation).
    - unique_chromosome_ratio: unique Pareto genotypes / Pareto size.
    - median_pairwise_hamming: median Hamming distance among Pareto chromosomes.
    - objective_std_*: objective dispersion across Pareto members.
    """
    if not pareto_indices:
        return {
            "sortino_cap_hit_fraction": 0.0,
            "unique_chromosome_ratio": 0.0,
            "median_pairwise_hamming": 0.0,
            "objective_std_f1": 0.0,
            "objective_std_f2": 0.0,
            "objective_std_f3": 0.0,
        }

    cap = max(float(_cfg.SORTINO_CAP), 1e-9)
    scale = max(float(_cfg.SORTINO_SCALE), 1e-9)
    cap_hits = 0
    for i in pareto_indices:
        raw = float(
            metrics_cache[i].get(
                "sortino_ratio",
                metrics_cache[i].get("total_return_pct", 0.0),
            )
        )
        sat = float(np.tanh(raw / scale) * cap)
        if abs(sat) >= 0.98 * cap:
            cap_hits += 1

    if len(pareto_obj):
        std_f1, std_f2, std_f3 = np.std(pareto_obj, axis=0)
    else:
        std_f1 = std_f2 = std_f3 = 0.0

    chromosomes: list[np.ndarray] = []
    if population is not None:
        for i in pareto_indices:
            chromosomes.append(population[int(i)].copy())
    unique_ratio = 0.0
    median_hamming = 0.0
    if chromosomes:
        from gpu_fuzzy_trader.phases.phase2_sparse_encoding import chromosome_key

        keys = {chromosome_key(c) for c in chromosomes}
        unique_ratio = float(len(keys) / len(chromosomes))
        median_hamming = _median_pairwise_hamming(chromosomes)

    return {
        "sortino_cap_hit_fraction": float(cap_hits / len(pareto_indices)),
        "unique_chromosome_ratio": unique_ratio,
        "median_pairwise_hamming": median_hamming,
        "objective_std_f1": float(std_f1),
        "objective_std_f2": float(std_f2),
        "objective_std_f3": float(std_f3),
    }


def _make_offspring_population(
    population: np.ndarray,
    objectives: np.ndarray,
    pop_size: int,
    feature_infos: list[dict],
    dont_cares: np.ndarray,
    rng: np.random.Generator,
    feature_probs: np.ndarray | None = None,
    fronts: list[list[int]] | None = None,
) -> np.ndarray:
    """Generate pop_size offspring via binary tournament, crossover, mutation.

    *fronts* may be passed in from the caller to avoid recomputing the sort.
    """
    if fronts is None:
        fronts = non_dominated_sort(objectives)
    rank, crowding = _build_rank_and_crowding(objectives, fronts)
    all_indices = list(range(pop_size))
    offspring_list: list[np.ndarray] = []
    for _ in range(0, pop_size, 2):
        pa = _binary_tournament_pick(all_indices, rank, crowding, rng)
        pb = _binary_tournament_pick(all_indices, rank, crowding, rng)
        child_a, child_b = _crossover(population[pa], population[pb], rng)
        offspring_list.append(
            _mutate(
                child_a, feature_infos, dont_cares, rng,
                mutation_rate=float(_cfg.PHASE2_MUTATION_RATE),
                feature_probs=feature_probs,
            )
        )
        offspring_list.append(
            _mutate(
                child_b, feature_infos, dont_cares, rng,
                mutation_rate=float(_cfg.PHASE2_MUTATION_RATE),
                feature_probs=feature_probs,
            )
        )
    return np.stack(offspring_list[:pop_size], axis=0)


def _get_reference_vectors(
    pop_size: int,
    n_objs: int = 3,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Das-Dennis style reference vectors; uses EvoX uniform_sampling when available."""
    if _EVOX_AVAILABLE and uniform_sampling is not None:
        refs = uniform_sampling(pop_size, n_objs)[0].cpu().numpy()
        while len(refs) < pop_size:
            extra, _ = uniform_sampling(
                pop_size - len(refs), n_objs)  # type: ignore[misc]
            refs = np.vstack([refs, extra.cpu().numpy()])
        return refs[:pop_size]

    refs = np.array(
        [
            [1.0, 1e-6, 1e-6],
            [1e-6, 1.0, 1e-6],
            [1e-6, 1e-6, 1.0],
        ],
        dtype=np.float64,
    )
    fallback_rng = rng if rng is not None else np.random.default_rng()
    while len(refs) < pop_size:
        t = fallback_rng.random()
        refs = np.vstack([refs, np.array([t, (1 - t) / 2, (1 - t) / 2])])
    return refs[:pop_size]


def _assign_eval_result(
    i: int,
    chromosome: np.ndarray,
    metrics: dict,
    val_metrics: dict | None,
    dont_cares: np.ndarray,
    pareto_archive: list[np.ndarray],
    objectives: np.ndarray,
    metrics_cache: list[dict],
    regime_row_fractions: np.ndarray | None,
    val_regime_row_counts: np.ndarray | None,
) -> None:
    """Compute penalties/objectives from metrics; write objectives[i] and metrics_cache[i]."""
    from gpu_fuzzy_trader.phases.phase2_rule_pool import _count_active_conditions
    from gpu_fuzzy_trader.phases.phase2_rule_pool import _saturating_sortino
    from gpu_fuzzy_trader.phases.phase2_support import (
        compute_support_penalty_and_specialist,
    )

    active = _count_active_conditions(chromosome, dont_cares)
    cond_penalty = 0.0
    if active < _cfg.MIN_CONDITIONS:
        cond_penalty = (_cfg.MIN_CONDITIONS - active) * 10.0
    elif active > _cfg.MAX_CONDITIONS:
        cond_penalty = (active - _cfg.MAX_CONDITIONS) * 10.0

    raw_sortino = float(metrics.get(
        "sortino_ratio", metrics.get("total_return_pct", 0.0)))
    sortino_train = _saturating_sortino(raw_sortino)
    max_dd = float(metrics.get("max_drawdown_pct", 100.0))
    win_rate = float(metrics.get("win_rate", 0.0))

    sortino_for_obj = sortino_train
    if val_metrics is not None:
        raw_val_sortino = float(val_metrics.get(
            "sortino_ratio",
            val_metrics.get("total_return_pct", 0.0),
        ))
        sortino_val = _saturating_sortino(raw_val_sortino)
        val_executed = int(val_metrics.get("executed_trades", 0))
        metrics["val_sortino_ratio"] = raw_val_sortino
        metrics["val_executed_trades"] = val_executed
        if val_executed < max(_cfg.MIN_TRADE_POOL_FLOOR // 4, 10):
            sortino_for_obj = min(sortino_train, 0.0)
        else:
            sortino_for_obj = min(sortino_train, sortino_val)

    support_penalty, is_specialist, dominant_regime = (
        compute_support_penalty_and_specialist(
            metrics,
            regime_row_fractions,
            val_metrics=val_metrics,
            val_regime_row_counts=val_regime_row_counts,
        )
    )
    if val_metrics is not None and int(
        val_metrics.get("executed_trades", 0),
    ) < max(_cfg.MIN_TRADE_POOL_FLOOR // 4, 10):
        support_penalty = max(
            support_penalty, _cfg.SUPPORT_PENALTY_MAX)
        sortino_for_obj = min(sortino_train, 0.0)
        is_specialist = False
    if is_specialist:
        metrics["regime_specialist"] = True
        metrics["dominant_regime"] = dominant_regime

    diversity_penalty = 0.0
    if pareto_archive:
        min_hamming = batch_hamming_min(chromosome, pareto_archive)
        if min_hamming <= _cfg.PHASE2_DIVERSITY_HAMMING_THRESHOLD:
            diversity_penalty = _cfg.PHASE2_DIVERSITY_PENALTY

    executed = int(metrics.get("executed_trades", 0))
    dd_val = max_dd
    trade_floor = (
        _cfg.PHASE2_CV_MIN_TRADE_POOL_FLOOR
        if str(_cfg.SPLIT_MODE).strip().lower() == "purged_rolling_cv"
        else _cfg.MIN_TRADE_POOL_FLOOR
    )

    if _cfg.PHASE2_USE_TOTAL_RETURN_OBJ:
        f3_val = float(metrics.get("total_return_pct", 0.0))
    else:
        f3_val = win_rate

    trade_penalty = 0.0
    if executed < trade_floor:
        dd_val = 100.0
        sortino_for_obj = 0.0
        f3_val = 0.0
        trade_penalty = 50.0

    pen = support_penalty + diversity_penalty + cond_penalty + trade_penalty

    objectives[i] = np.array(
        [-sortino_for_obj + pen, dd_val + pen, -f3_val + pen],
        dtype=np.float64,
    )
    metrics_cache[i] = metrics


def _store_global_metrics_cache(
    global_metrics_cache: dict[tuple[int, ...], dict],
    key: tuple[int, ...],
    metrics: dict,
    val_metrics: dict | None,
) -> None:
    """Store processed metrics (and optional val sidecar) in the run-wide cache."""
    entry = copy.deepcopy(metrics)
    if val_metrics is not None:
        entry["_cached_val_metrics"] = copy.deepcopy(val_metrics)
    global_metrics_cache[key] = entry


def _load_global_metrics_cache(
    cached: dict,
) -> tuple[dict, dict | None]:
    """Restore metrics and optional val_metrics from a cache entry."""
    entry = copy.deepcopy(cached)
    val_metrics = entry.pop("_cached_val_metrics", None)
    return entry, val_metrics


def _evaluate_population_indices(
    population: np.ndarray,
    indices: list[int],
    dont_cares: np.ndarray,
    engine,
    pareto_archive: list[np.ndarray],
    objectives: np.ndarray,
    metrics_cache: list[dict],
    val_engine=None,
    regime_row_fractions: np.ndarray | None = None,
    val_regime_row_counts: np.ndarray | None = None,
    global_metrics_cache: dict[tuple[int, ...], dict] | None = None,
) -> dict[str, int]:
    """Evaluate unevaluated individuals, preferring batch simulate_rule_batch."""
    from gpu_fuzzy_trader.phases.phase2_rule_pool import (
        _evaluate_chromosome,
    )
    from gpu_fuzzy_trader.phases.phase2_sparse_encoding import chromosome_key

    pending = [i for i in indices if np.any(np.isinf(objectives[i]))]
    if not pending:
        return _empty_eval_stats()

    if regime_row_fractions is None:
        regime_row_fractions = getattr(
            engine, "_regime_row_fractions", None)
    if val_regime_row_counts is None and val_engine is not None:
        val_regime_row_counts = getattr(
            val_engine, "_regime_row_counts", None)

    gpu_pending: list[int] = []
    cache_hits = 0
    if _cfg.PHASE2_EVAL_GLOBAL_CACHE and global_metrics_cache is not None:
        for i in pending:
            key = chromosome_key(population[i])
            if key in global_metrics_cache:
                metrics, val_metrics = _load_global_metrics_cache(
                    global_metrics_cache[key],
                )
                _assign_eval_result(
                    i,
                    population[i],
                    metrics,
                    val_metrics,
                    dont_cares,
                    pareto_archive,
                    objectives,
                    metrics_cache,
                    regime_row_fractions,
                    val_regime_row_counts,
                )
                cache_hits += 1
            else:
                gpu_pending.append(i)
    else:
        gpu_pending = list(pending)

    if not gpu_pending:
        if cache_hits:
            logger.debug(
                "Phase 2 eval: %d cache hits, 0 GPU pending",
                cache_hits,
            )
        return {"pending": len(pending), "cache_hits": cache_hits}

    try:
        if _cfg.PHASE2_EVAL_BATCH_DEDUP:
            key_to_uj: dict[tuple[int, ...], int] = {}
            unique_rows: list[np.ndarray] = []
            for i in gpu_pending:
                key = chromosome_key(population[i])
                if key not in key_to_uj:
                    key_to_uj[key] = len(unique_rows)
                    unique_rows.append(population[i])
            unique_chroms = np.stack(unique_rows, axis=0)
        else:
            key_to_uj = {
                chromosome_key(population[i]): j
                for j, i in enumerate(gpu_pending)
            }
            unique_chroms = population[gpu_pending]

        metrics_list = engine.simulate_rule_batch(
            chromosomes=unique_chroms,
            tp=_cfg.PHASE2_TP,
            sl=_cfg.PHASE2_SL,
            capital_pct=_cfg.PHASE2_CAPITAL_PCT,
        )

        val_metrics_list = None
        if val_engine is not None and _cfg.PHASE2_JOINT_TRAIN_VAL:
            try:
                val_metrics_list = val_engine.simulate_rule_batch(
                    chromosomes=unique_chroms,
                    tp=_cfg.PHASE2_TP,
                    sl=_cfg.PHASE2_SL,
                    capital_pct=_cfg.PHASE2_CAPITAL_PCT,
                )
            except Exception as exc:
                logger.debug(
                    "val simulate_rule_batch failed in Phase 2 batch path: %s",
                    exc,
                )
                val_metrics_list = None

        unique_count = len(unique_chroms)
        logger.debug(
            "Phase 2 eval: %d cache hits, %d GPU pending, %d unique chromosomes",
            cache_hits,
            len(gpu_pending),
            unique_count,
        )

        for i in gpu_pending:
            key = chromosome_key(population[i])
            uj = key_to_uj[key]
            metrics = copy.deepcopy(metrics_list[uj])
            val_metrics = None
            if val_metrics_list is not None:
                val_metrics = copy.deepcopy(val_metrics_list[uj])

            _assign_eval_result(
                i,
                population[i],
                metrics,
                val_metrics,
                dont_cares,
                pareto_archive,
                objectives,
                metrics_cache,
                regime_row_fractions,
                val_regime_row_counts,
            )

            if _cfg.PHASE2_EVAL_GLOBAL_CACHE and global_metrics_cache is not None:
                _store_global_metrics_cache(
                    global_metrics_cache,
                    key,
                    metrics_cache[i],
                    val_metrics,
                )
    except Exception as exc:
        logger.debug("Batch eval failed, falling back to single: %s", exc)
        for i in gpu_pending:
            obj, metrics = _evaluate_chromosome(
                population[i], dont_cares, engine, pareto_archive,
                val_engine=val_engine,
                regime_row_fractions_arr=regime_row_fractions,
                val_regime_row_counts=val_regime_row_counts,
            )
            objectives[i] = obj
            metrics_cache[i] = metrics
            if _cfg.PHASE2_EVAL_GLOBAL_CACHE and global_metrics_cache is not None:
                _store_global_metrics_cache(
                    global_metrics_cache,
                    chromosome_key(population[i]),
                    metrics,
                    None,
                )
        return {"pending": len(pending), "cache_hits": cache_hits}

    return {"pending": len(pending), "cache_hits": cache_hits}


def _normalize_for_association(
    merge_fit: np.ndarray,
    ref: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Min-max normalize objectives; unit-normalize rows for association."""
    norm_fit = merge_fit - merge_fit.min(axis=0)
    norm_fit = norm_fit / np.maximum(norm_fit.max(axis=0), 1e-6)
    ref_n = ref / np.linalg.norm(ref, axis=1, keepdims=True).clip(1e-10)
    fit_n = norm_fit / \
        np.linalg.norm(norm_fit, axis=1, keepdims=True).clip(1e-10)
    return fit_n, ref_n


def _nsga3_environmental_selection(
    merge_pop: np.ndarray,
    merge_fit: np.ndarray,
    ref: np.ndarray,
    pop_size: int,
    feature_infos: list[dict],
    dont_cares: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """NSGA-III environmental selection (rank + niche on last front)."""
    if not _EVOX_AVAILABLE or non_dominate_rank is None:
        pop, fit, _ = environmental_selection_nsga2(
            merge_pop, merge_fit, pop_size)
        return _repair_population(pop, feature_infos, dont_cares), fit

    f = torch.tensor(merge_fit.astype(np.float32))
    rank = non_dominate_rank(f)
    rank_np = rank.cpu().numpy()
    worst_rank = int(torch.topk(rank, pop_size + 1,
                     largest=False)[0][-1].item())
    candi_idx = np.where(rank_np <= worst_rank)[0]

    if len(candi_idx) <= pop_size:
        idx = candi_idx[:pop_size]
        return (
            _repair_population(merge_pop[idx], feature_infos, dont_cares),
            merge_fit[idx],
        )

    selected = [int(i) for i in candi_idx if rank_np[i] < worst_rank]
    if len(selected) >= pop_size:
        idx = np.array(selected[:pop_size], dtype=np.intp)
        return (
            _repair_population(merge_pop[idx], feature_infos, dont_cares),
            merge_fit[idx],
        )

    need = pop_size - len(selected)
    last_idx = [int(i) for i in candi_idx if rank_np[i] == worst_rank]

    fit_n, ref_n = _normalize_for_association(merge_fit, ref)
    dist = 1.0 - fit_n @ ref_n.T
    assoc = np.argmin(dist, axis=1)

    rho = np.bincount(assoc[selected], minlength=len(ref)) if selected else np.zeros(
        len(ref), dtype=np.int64,
    )
    used_last: set[int] = set()

    while need > 0 and last_idx:
        ref_order = np.argsort(rho)
        picked = False
        for ref_j in ref_order:
            candidates = [
                i for i in last_idx
                if assoc[i] == ref_j and i not in used_last
            ]
            if not candidates:
                continue
            best_i = min(candidates, key=lambda i: dist[i, ref_j])
            selected.append(best_i)
            used_last.add(best_i)
            rho[ref_j] += 1
            need -= 1
            picked = True
            if need <= 0:
                break
        if not picked:
            for i in last_idx:
                if i not in used_last:
                    selected.append(i)
                    used_last.add(i)
                    need -= 1
                    if need <= 0:
                        break
            break

    idx = np.array(selected[:pop_size], dtype=np.intp)
    return (
        _repair_population(merge_pop[idx], feature_infos, dont_cares),
        merge_fit[idx],
    )


def _run_nsga2_fallback(
    feature_infos: list[dict],
    engine,
    pop_size: int,
    n_generations: int,
    rng: np.random.Generator,
    seed_chromosomes: np.ndarray | None = None,
    log_tag: str | None = None,
    val_engine=None,
    regime_row_fractions: np.ndarray | None = None,
    val_regime_row_counts: np.ndarray | None = None,
    feature_probs: np.ndarray | None = None,
    init_strategy: str | None = None,
    stratum_fractions: tuple[float, float, float] | None = None,
) -> tuple[list[dict], list[dict]]:
    """NumPy NSGA-II loop when EvoX is not installed."""
    from gpu_fuzzy_trader.phases.phase2_rule_pool import (
        _build_pool_from_archive,
        _evaluate_chromosome,
        _get_dont_cares,
        _init_population,
        _metrics_dict_from_population,
        _pareto_sortino_stats,
    )
    from gpu_fuzzy_trader.phases.phase2_support import passes_pool_trade_floor

    K = len(feature_infos)
    dont_cares = _get_dont_cares(feature_infos)
    population = _init_population(
        pop_size,
        feature_infos,
        rng,
        seeded_chromosomes=seed_chromosomes,
        init_strategy=init_strategy,
        stratum_fractions=stratum_fractions,
        feature_probs=feature_probs,
    )
    objectives = np.full((pop_size, 3), np.inf)
    metrics_cache: list[dict] = [{} for _ in range(pop_size)]
    pareto_archive: list[np.ndarray] = []
    hall_of_fame: dict[tuple[int, ...], np.ndarray] = {}
    history: list[dict] = []

    tag = log_tag or "NSGA-II (fallback)"
    logger.info("%s: %d features, pop=%d, gen=%d",
                tag, K, pop_size, n_generations)
    gen_loop_start = time.monotonic()
    plateau_best_max_return = -np.inf
    plateau_streak = 0

    for gen in range(n_generations):
        for i in range(pop_size):
            if np.any(np.isinf(objectives[i])):
                obj, metrics = _evaluate_chromosome(
                    population[i], dont_cares, engine, pareto_archive,
                    val_engine=val_engine,
                    regime_row_fractions_arr=regime_row_fractions,
                    val_regime_row_counts=val_regime_row_counts,
                )
                objectives[i] = obj
                metrics_cache[i] = metrics

        fronts = non_dominated_sort(objectives)
        pareto_indices = fronts[0]
        pareto_archive = [population[i].copy() for i in pareto_indices]
        _update_hall_of_fame(hall_of_fame, population, pareto_indices)

        pareto_obj = objectives[pareto_indices]
        pareto_diag = _pareto_diagnostics(
            pareto_indices, metrics_cache, pareto_obj, population,
        )
        history.append({
            "generation": gen,
            "pareto_size": len(pareto_indices),
            "mean_f1": float(np.mean(pareto_obj[:, 0])),
            "mean_f2": float(np.mean(pareto_obj[:, 1])),
            "mean_f3": float(np.mean(pareto_obj[:, 2])),
            "algorithm": "NSGA-II (fallback)",
            **_pareto_sortino_stats(pareto_indices, metrics_cache),
            **pareto_diag,
        })

        returns = [
            float(metrics_cache[i].get("total_return_pct", 0.0))
            for i in pareto_indices
        ]
        mean_ret = float(np.mean(returns)) if returns else -100.0
        median_ret = _pareto_median_return_pct(pareto_indices, metrics_cache)
        max_ret = max(returns) if returns else 0.0
        sortinos = [
            float(metrics_cache[i].get("sortino_ratio", 0.0))
            for i in pareto_indices
        ]
        max_sort = max(sortinos) if sortinos else 0.0
        val_count = sum(
            1 for i in pareto_indices
            if passes_pool_trade_floor(
                int(metrics_cache[i].get("executed_trades", 0)),
                metrics_cache[i],
                regime_row_fractions_arr=regime_row_fractions,
            )
        )
        plateau_best_max_return, plateau_streak = _update_max_return_plateau(
            max_ret, plateau_best_max_return, plateau_streak,
        )
        maybe_log_generation(
            logger, tag, gen, n_generations, len(pareto_indices), mean_ret,
            max_return_pct=max_ret,
            median_return_pct=median_ret,
            max_sortino=max_sort,
            valid_count=val_count,
            unique_chromosome_ratio=float(
                pareto_diag.get("unique_chromosome_ratio", 0.0)),
            loop_start=gen_loop_start,
        )

        pareto_ret = _pareto_return_pct_for_early_stop(
            pareto_indices, metrics_cache,
        )
        if _should_early_stop_phase2(gen, pareto_ret, val_count):
            stat_label = (
                "median_return" if _cfg.PHASE2_EARLY_STOP_USE_MEDIAN_RETURN
                else "mean_return"
            )
            logger.info(
                "%s: early stop at gen %d (%s=%.2f%% < %.2f%%, "
                "valid_rules=%d < %d)",
                tag,
                gen + 1,
                stat_label,
                pareto_ret,
                float(_cfg.PHASE2_EARLY_STOP_MEAN_RETURN_PCT),
                val_count,
                int(_cfg.PHASE2_EARLY_STOP_MIN_VALID_RULES),
            )
            break

        if _should_plateau_early_stop_phase2(gen, plateau_streak):
            logger.info(
                "%s: plateau early stop at gen %d (max_return=%.2f%% unchanged "
                "for %d gens, patience=%d, min_delta=%.2f%%)",
                tag,
                gen + 1,
                max_ret,
                plateau_streak,
                int(_cfg.PHASE2_PLATEAU_EARLY_STOP_PATIENCE),
                float(_cfg.PHASE2_PLATEAU_EARLY_STOP_MIN_DELTA_PCT),
            )
            break

        if gen == n_generations - 1:
            break

        offspring = _make_offspring_population(
            population, objectives, pop_size, feature_infos, dont_cares, rng,
            feature_probs=feature_probs,
            fronts=fronts,
        )
        off_obj = np.full((pop_size, 3), np.inf)
        off_metrics: list[dict] = [{} for _ in range(pop_size)]
        for i in range(pop_size):
            obj, metrics = _evaluate_chromosome(
                offspring[i], dont_cares, engine, pareto_archive,
                val_engine=val_engine,
                regime_row_fractions_arr=regime_row_fractions,
                val_regime_row_counts=val_regime_row_counts,
            )
            off_obj[i] = obj
            off_metrics[i] = metrics

        merge_pop = np.vstack([population, offspring])
        merge_fit = np.vstack([objectives, off_obj])
        merge_metrics = metrics_cache + off_metrics

        population, objectives, sel_idx = environmental_selection_nsga2(
            merge_pop, merge_fit, pop_size,
        )
        metrics_cache = [merge_metrics[i] for i in sel_idx]

    metrics_by_chrom = _metrics_dict_from_population(
        population, metrics_cache
    )
    harvest_archive = list(hall_of_fame.values(
    )) if hall_of_fame else pareto_archive
    pareto_pool = _build_pool_from_archive(
        harvest_archive,
        feature_infos,
        dont_cares,
        engine,
        metrics_by_chrom=metrics_by_chrom,
        regime_row_fractions_arr=regime_row_fractions,
        val_engine=val_engine,
        direction=log_tag or "",
    )
    return pareto_pool, history


def _run_nsga3(
    feature_infos: list[dict],
    engine,
    pop_size: int,
    n_generations: int,
    rng: np.random.Generator,
    seed_chromosomes: np.ndarray | None = None,
    log_tag: str | None = None,
    val_engine=None,
    regime_row_fractions: np.ndarray | None = None,
    val_regime_row_counts: np.ndarray | None = None,
    feature_probs: np.ndarray | None = None,
    init_strategy: str | None = None,
    stratum_fractions: tuple[float, float, float] | None = None,
) -> tuple[list[dict], list[dict]]:
    """NSGA-III evolutionary loop for Phase 2 rule pool generation."""
    from gpu_fuzzy_trader.phases.phase2_rule_pool import (
        _build_pool_from_archive,
        _get_dont_cares,
        _init_population,
        _metrics_dict_from_population,
        _pareto_sortino_stats,
    )
    from gpu_fuzzy_trader.phases.phase2_support import passes_pool_trade_floor

    K = len(feature_infos)
    dont_cares = _get_dont_cares(feature_infos)
    population = _init_population(
        pop_size,
        feature_infos,
        rng,
        seeded_chromosomes=seed_chromosomes,
        init_strategy=init_strategy,
        stratum_fractions=stratum_fractions,
        feature_probs=feature_probs,
    )
    objectives = np.full((pop_size, 3), np.inf)
    metrics_cache: list[dict] = [{} for _ in range(pop_size)]
    pareto_archive: list[np.ndarray] = []
    hall_of_fame: dict[tuple[int, ...], np.ndarray] = {}
    global_metrics_cache: dict[tuple[int, ...], dict] = {}
    history: list[dict] = []

    ref_vec = _get_reference_vectors(pop_size, 3, rng)
    tag = log_tag or "NSGA-III"
    logger.info("%s: %d features, pop=%d, gen=%d",
                tag, K, pop_size, n_generations)
    gen_loop_start = time.monotonic()
    plateau_best_max_return = -np.inf
    plateau_streak = 0

    for gen in range(n_generations):
        parent_stats = _evaluate_population_indices(
            population,
            list(range(pop_size)),
            dont_cares,
            engine,
            pareto_archive,
            objectives,
            metrics_cache,
            val_engine=val_engine,
            regime_row_fractions=regime_row_fractions,
            val_regime_row_counts=val_regime_row_counts,
            global_metrics_cache=global_metrics_cache,
        )

        # Compute fronts once — reused for logging, offspring, and archive.
        fronts = non_dominated_sort(objectives)
        pareto_indices = fronts[0]
        pareto_archive = [population[i].copy() for i in pareto_indices]
        _update_hall_of_fame(hall_of_fame, population, pareto_indices)

        pareto_obj = objectives[pareto_indices]
        pareto_diag = _pareto_diagnostics(
            pareto_indices, metrics_cache, pareto_obj, population,
        )
        history.append({
            "generation": gen,
            "pareto_size": len(pareto_indices),
            "mean_f1": float(np.mean(pareto_obj[:, 0])) if len(pareto_obj) else 0.0,
            "mean_f2": float(np.mean(pareto_obj[:, 1])) if len(pareto_obj) else 0.0,
            "mean_f3": float(np.mean(pareto_obj[:, 2])) if len(pareto_obj) else 0.0,
            "algorithm": "NSGA-III",
            **_pareto_sortino_stats(pareto_indices, metrics_cache),
            **pareto_diag,
        })

        returns = [
            float(metrics_cache[i].get("total_return_pct", 0.0))
            for i in pareto_indices
        ]
        mean_ret = float(np.mean(returns)) if returns else -100.0
        median_ret = _pareto_median_return_pct(pareto_indices, metrics_cache)
        max_ret = max(returns) if returns else 0.0
        sortinos = [
            float(metrics_cache[i].get("sortino_ratio", 0.0))
            for i in pareto_indices
        ]
        max_sort = max(sortinos) if sortinos else 0.0
        val_count = sum(
            1 for i in pareto_indices
            if passes_pool_trade_floor(
                int(metrics_cache[i].get("executed_trades", 0)),
                metrics_cache[i],
                regime_row_fractions_arr=regime_row_fractions,
            )
        )

        plateau_best_max_return, plateau_streak = _update_max_return_plateau(
            max_ret, plateau_best_max_return, plateau_streak,
        )

        is_last_gen = gen == n_generations - 1
        off_stats = _empty_eval_stats()
        if not is_last_gen:
            offspring = _make_offspring_population(
                population, objectives, pop_size, feature_infos, dont_cares, rng,
                feature_probs=feature_probs,
                fronts=fronts,
            )
            off_obj = np.full((pop_size, 3), np.inf)
            off_metrics: list[dict] = [{} for _ in range(pop_size)]
            off_stats = _evaluate_population_indices(
                offspring,
                list(range(pop_size)),
                dont_cares,
                engine,
                pareto_archive,
                off_obj,
                off_metrics,
                val_engine=val_engine,
                regime_row_fractions=regime_row_fractions,
                val_regime_row_counts=val_regime_row_counts,
                global_metrics_cache=global_metrics_cache,
            )

        gen_eval_stats = _merge_gen_eval_stats(parent_stats, off_stats)
        history[-1]["eval_cache_hit_rate"] = _eval_cache_hit_rate(
            gen_eval_stats)

        maybe_log_generation(
            logger, tag, gen, n_generations, len(pareto_indices), mean_ret,
            max_return_pct=max_ret,
            median_return_pct=median_ret,
            max_sortino=max_sort,
            valid_count=val_count,
            unique_chromosome_ratio=float(
                pareto_diag.get("unique_chromosome_ratio", 0.0)),
            cache_hit_rate=_eval_cache_hit_rate(gen_eval_stats),
            loop_start=gen_loop_start,
        )

        pareto_ret = _pareto_return_pct_for_early_stop(
            pareto_indices, metrics_cache,
        )
        if _should_early_stop_phase2(gen, pareto_ret, val_count):
            stat_label = (
                "median_return" if _cfg.PHASE2_EARLY_STOP_USE_MEDIAN_RETURN
                else "mean_return"
            )
            logger.info(
                "%s: early stop at gen %d (%s=%.2f%% < %.2f%%, "
                "valid_rules=%d < %d)",
                tag,
                gen + 1,
                stat_label,
                pareto_ret,
                float(_cfg.PHASE2_EARLY_STOP_MEAN_RETURN_PCT),
                val_count,
                int(_cfg.PHASE2_EARLY_STOP_MIN_VALID_RULES),
            )
            break

        if _should_plateau_early_stop_phase2(gen, plateau_streak):
            logger.info(
                "%s: plateau early stop at gen %d (max_return=%.2f%% unchanged "
                "for %d gens, patience=%d, min_delta=%.2f%%)",
                tag,
                gen + 1,
                max_ret,
                plateau_streak,
                int(_cfg.PHASE2_PLATEAU_EARLY_STOP_PATIENCE),
                float(_cfg.PHASE2_PLATEAU_EARLY_STOP_MIN_DELTA_PCT),
            )
            break

        if is_last_gen:
            break

        merge_pop = np.vstack([population, offspring])
        merge_fit = np.vstack([objectives, off_obj])
        merge_metrics = metrics_cache + off_metrics

        population, objectives = _nsga3_environmental_selection(
            merge_pop, merge_fit, ref_vec, pop_size, feature_infos, dont_cares,
        )

        # Carry objectives directly from merge_fit by index matching instead of
        # expensive tuple-key dict lookup — objectives are already correct after
        # environmental selection since _nsga3_environmental_selection returns
        # the sliced merge_fit directly.
        n_alive = len(population)
        from gpu_fuzzy_trader.phases.phase2_sparse_encoding import chromosome_key

        _merge_metrics_by_key: dict[tuple, dict] = {
            chromosome_key(merge_pop[j]): m
            for j, m in enumerate(merge_metrics)
            if m
        }
        metrics_cache = [
            _merge_metrics_by_key.get(chromosome_key(population[i]), {})
            for i in range(n_alive)
        ]

    metrics_by_chrom = _metrics_dict_from_population(population, metrics_cache)
    harvest_archive = list(hall_of_fame.values(
    )) if hall_of_fame else pareto_archive
    pareto_pool = _build_pool_from_archive(
        harvest_archive,
        feature_infos,
        dont_cares,
        engine,
        metrics_by_chrom=metrics_by_chrom,
        regime_row_fractions_arr=regime_row_fractions,
        val_engine=val_engine,
        direction=log_tag or "",
    )
    return pareto_pool, history


def run_phase2_evolution(
    feature_infos: list[dict],
    engine,
    pop_size: int,
    n_generations: int,
    rng: np.random.Generator,
    seed_chromosomes: np.ndarray | None = None,
    log_tag: str | None = None,
    val_engine=None,
    regime_row_fractions: np.ndarray | None = None,
    val_regime_row_counts: np.ndarray | None = None,
    feature_probs: np.ndarray | None = None,
    init_strategy: str | None = None,
    stratum_fractions: tuple[float, float] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Run Phase 2 NSGA-III evolution. Returns (pareto_pool, history)."""
    evo_kwargs = dict(
        feature_probs=feature_probs,
        init_strategy=init_strategy,
        stratum_fractions=stratum_fractions,
    )
    if not _EVOX_AVAILABLE:
        logger.warning(
            "EvoX not available (%s); falling back to NumPy NSGA-II for Phase 2.",
            _EVOX_IMPORT_ERROR or "import failed",
        )
        fallback_tag = "NSGA-II (fallback)"
        if log_tag:
            fallback_tag = log_tag.replace("NSGA-III", "NSGA-II (fallback)")
        return _run_nsga2_fallback(
            feature_infos, engine, pop_size, n_generations, rng,
            seed_chromosomes=seed_chromosomes,
            log_tag=fallback_tag,
            val_engine=val_engine,
            regime_row_fractions=regime_row_fractions,
            val_regime_row_counts=val_regime_row_counts,
            **evo_kwargs,
        )

    return _run_nsga3(
        feature_infos, engine, pop_size, n_generations, rng,
        seed_chromosomes=seed_chromosomes, log_tag=log_tag,
        val_engine=val_engine,
        regime_row_fractions=regime_row_fractions,
        val_regime_row_counts=val_regime_row_counts,
        **evo_kwargs,
    )
