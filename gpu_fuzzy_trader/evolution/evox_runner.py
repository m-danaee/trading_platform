"""
evox_runner.py — Phase 2 NSGA-III multi-objective evolution.

Uses EvoX reference-vector sampling and non-dominated ranking with niche
selection on integer fuzzy-rule chromosomes. Falls back to NumPy NSGA-II
environmental selection when EvoX is unavailable.
"""

from __future__ import annotations
from gpu_fuzzy_trader.phases.phase2_rule_pool import (
    _crowding_distance,
    _crossover,
    _mutate,
    _non_dominated_sort,
    trade_support_penalty,
)

import logging
import time

import numpy as np

from gpu_fuzzy_trader.log_progress import maybe_log_generation

logger = logging.getLogger(__name__)

_EVOX_AVAILABLE = False
try:
    import torch
    from evox.operators.sampling.uniform import uniform_sampling
    from evox.operators.selection.non_dominate import non_dominate_rank

    _EVOX_AVAILABLE = True
except ImportError:
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
            cd = _crowding_distance(objectives, front)
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
    fronts = _non_dominated_sort(merge_fit)
    selected: list[int] = []
    for front in fronts:
        if not front:
            continue
        if len(selected) + len(front) <= pop_size:
            selected.extend(front)
        else:
            cd = _crowding_distance(merge_fit, front)
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
    return np.stack(
        [
            _repair_chromosome(population[i], feature_infos, dont_cares)
            for i in range(len(population))
        ],
        axis=0,
    )


def _make_offspring_population(
    population: np.ndarray,
    objectives: np.ndarray,
    pop_size: int,
    feature_infos: list[dict],
    dont_cares: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate pop_size offspring via binary tournament, crossover, mutation."""
    fronts = _non_dominated_sort(objectives)
    rank, crowding = _build_rank_and_crowding(objectives, fronts)
    all_indices = list(range(pop_size))
    offspring_list: list[np.ndarray] = []
    for _ in range(0, pop_size, 2):
        pa = _binary_tournament_pick(all_indices, rank, crowding, rng)
        pb = _binary_tournament_pick(all_indices, rank, crowding, rng)
        child_a, child_b = _crossover(population[pa], population[pb], rng)
        offspring_list.append(
            _mutate(child_a, feature_infos, dont_cares, rng)
        )
        offspring_list.append(
            _mutate(child_b, feature_infos, dont_cares, rng)
        )
    return np.stack(offspring_list[:pop_size], axis=0)


def _get_reference_vectors(pop_size: int, n_objs: int = 3) -> np.ndarray:
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
    while len(refs) < pop_size:
        t = np.random.default_rng(42).random()
        refs = np.vstack([refs, np.array([t, (1 - t) / 2, (1 - t) / 2])])
    return refs[:pop_size]


def _evaluate_population_indices(
    population: np.ndarray,
    indices: list[int],
    dont_cares: np.ndarray,
    engine,
    pareto_archive: list[np.ndarray],
    objectives: np.ndarray,
    metrics_cache: list[dict],
) -> None:
    """Evaluate unevaluated individuals, preferring batch simulate_rule_batch."""
    from gpu_fuzzy_trader.phases.phase2_rule_pool import _evaluate_chromosome

    pending = [i for i in indices if np.any(np.isinf(objectives[i]))]
    if not pending:
        return

    try:
        from gpu_fuzzy_trader import config as _cfg
        from gpu_fuzzy_trader.phases.phase2_rule_pool import (
            _count_active_conditions,
            _hamming_distance,
        )

        chroms = population[pending]
        metrics_list = engine.simulate_rule_batch(
            chromosomes=chroms,
            tp=_cfg.PHASE2_TP,
            sl=_cfg.PHASE2_SL,
            capital_pct=_cfg.PHASE2_CAPITAL_PCT,
        )

        for j, i in enumerate(pending):
            chromosome = population[i]
            metrics = metrics_list[j]
            active = _count_active_conditions(chromosome, dont_cares)
            cond_penalty = 0.0
            if active < _cfg.MIN_CONDITIONS:
                cond_penalty = (_cfg.MIN_CONDITIONS - active) * 10.0
            elif active > _cfg.MAX_CONDITIONS:
                cond_penalty = (active - _cfg.MAX_CONDITIONS) * 10.0

            sortino_ratio = float(metrics.get(
                "sortino_ratio", metrics.get("total_return_pct", 0.0)))
            max_dd = float(metrics.get("max_drawdown_pct", 100.0))
            win_rate = float(metrics.get("win_rate", 0.0))
            executed = int(metrics.get("executed_trades", 0))

            support_penalty = trade_support_penalty(executed)

            diversity_penalty = 0.0
            if pareto_archive:
                min_hamming = min(
                    _hamming_distance(chromosome, pf) for pf in pareto_archive
                )
                if min_hamming == 0:
                    diversity_penalty = 5.0

            pen = support_penalty + diversity_penalty + cond_penalty
            objectives[i] = np.array(
                [-sortino_ratio + pen, max_dd + pen, -win_rate + pen],
                dtype=np.float64,
            )
            metrics_cache[i] = metrics
    except Exception as exc:
        logger.debug("Batch eval failed, falling back to single: %s", exc)
        for i in pending:
            obj, met = _evaluate_chromosome(
                population[i], dont_cares, engine, pareto_archive
            )
            objectives[i] = obj
            metrics_cache[i] = met


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

    K = len(feature_infos)
    dont_cares = _get_dont_cares(feature_infos)
    population = _init_population(
        pop_size,
        feature_infos,
        rng,
        seeded_chromosomes=seed_chromosomes,
    )
    objectives = np.full((pop_size, 3), np.inf)
    metrics_cache: list[dict] = [{} for _ in range(pop_size)]
    pareto_archive: list[np.ndarray] = []
    history: list[dict] = []

    tag = log_tag or "NSGA-II (fallback)"
    logger.info("%s: %d features, pop=%d, gen=%d",
                tag, K, pop_size, n_generations)
    gen_loop_start = time.monotonic()

    for gen in range(n_generations):
        for i in range(pop_size):
            if np.any(np.isinf(objectives[i])):
                obj, met = _evaluate_chromosome(
                    population[i], dont_cares, engine, pareto_archive
                )
                objectives[i] = obj
                metrics_cache[i] = met

        fronts = _non_dominated_sort(objectives)
        pareto_indices = fronts[0]
        pareto_archive = [population[i].copy() for i in pareto_indices]

        pareto_obj = objectives[pareto_indices]
        history.append({
            "generation": gen,
            "pareto_size": len(pareto_indices),
            "mean_f1": float(np.mean(pareto_obj[:, 0])),
            "mean_f2": float(np.mean(pareto_obj[:, 1])),
            "mean_f3": float(np.mean(pareto_obj[:, 2])),
            "algorithm": "NSGA-II (fallback)",
            **_pareto_sortino_stats(pareto_indices, metrics_cache),
        })

        mean_f1 = float(np.mean(pareto_obj[:, 0])) if len(pareto_obj) else 0.0
        maybe_log_generation(
            logger, tag, gen, n_generations, len(pareto_indices), mean_f1,
            loop_start=gen_loop_start,
        )

        if gen == n_generations - 1:
            break

        offspring = _make_offspring_population(
            population, objectives, pop_size, feature_infos, dont_cares, rng,
        )
        off_obj = np.full((pop_size, 3), np.inf)
        off_metrics: list[dict] = [{} for _ in range(pop_size)]
        for i in range(pop_size):
            obj, met = _evaluate_chromosome(
                offspring[i], dont_cares, engine, pareto_archive
            )
            off_obj[i] = obj
            off_metrics[i] = met

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
    pareto_pool = _build_pool_from_archive(
        pareto_archive,
        feature_infos,
        dont_cares,
        engine,
        metrics_by_chrom=metrics_by_chrom,
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
) -> tuple[list[dict], list[dict]]:
    """NSGA-III evolutionary loop for Phase 2 rule pool generation."""
    from gpu_fuzzy_trader.phases.phase2_rule_pool import (
        _build_pool_from_archive,
        _get_dont_cares,
        _init_population,
        _metrics_dict_from_population,
        _non_dominated_sort,
        _pareto_sortino_stats,
    )

    K = len(feature_infos)
    dont_cares = _get_dont_cares(feature_infos)
    population = _init_population(
        pop_size,
        feature_infos,
        rng,
        seeded_chromosomes=seed_chromosomes,
    )
    objectives = np.full((pop_size, 3), np.inf)
    metrics_cache: list[dict] = [{} for _ in range(pop_size)]
    pareto_archive: list[np.ndarray] = []
    history: list[dict] = []

    ref_vec = _get_reference_vectors(pop_size, 3)
    tag = log_tag or "NSGA-III"
    logger.info("%s: %d features, pop=%d, gen=%d",
                tag, K, pop_size, n_generations)
    gen_loop_start = time.monotonic()

    for gen in range(n_generations):
        _evaluate_population_indices(
            population,
            list(range(pop_size)),
            dont_cares,
            engine,
            pareto_archive,
            objectives,
            metrics_cache,
        )

        fronts = _non_dominated_sort(objectives)
        pareto_indices = fronts[0]
        pareto_archive = [population[i].copy() for i in pareto_indices]

        pareto_obj = objectives[pareto_indices]
        history.append({
            "generation": gen,
            "pareto_size": len(pareto_indices),
            "mean_f1": float(np.mean(pareto_obj[:, 0])) if len(pareto_obj) else 0.0,
            "mean_f2": float(np.mean(pareto_obj[:, 1])) if len(pareto_obj) else 0.0,
            "mean_f3": float(np.mean(pareto_obj[:, 2])) if len(pareto_obj) else 0.0,
            "algorithm": "NSGA-III",
            **_pareto_sortino_stats(pareto_indices, metrics_cache),
        })

        mean_f1 = float(np.mean(pareto_obj[:, 0])) if len(pareto_obj) else 0.0
        maybe_log_generation(
            logger, tag, gen, n_generations, len(pareto_indices), mean_f1,
            loop_start=gen_loop_start,
        )

        if gen == n_generations - 1:
            break

        offspring = _make_offspring_population(
            population, objectives, pop_size, feature_infos, dont_cares, rng,
        )
        off_obj = np.full((pop_size, 3), np.inf)
        off_metrics: list[dict] = [{} for _ in range(pop_size)]
        _evaluate_population_indices(
            offspring,
            list(range(pop_size)),
            dont_cares,
            engine,
            pareto_archive,
            off_obj,
            off_metrics,
        )

        merge_pop = np.vstack([population, offspring])
        merge_fit = np.vstack([objectives, off_obj])
        merge_metrics = metrics_cache + off_metrics

        population, objectives = _nsga3_environmental_selection(
            merge_pop, merge_fit, ref_vec, pop_size, feature_infos, dont_cares,
        )

        n_alive = len(population)
        metrics_cache = [{} for _ in range(n_alive)]
        for i in range(n_alive):
            key = tuple(population[i].tolist())
            for j, m in enumerate(merge_metrics):
                if m and tuple(merge_pop[j].tolist()) == key:
                    metrics_cache[i] = m
                    break

    metrics_by_chrom = _metrics_dict_from_population(population, metrics_cache)
    pareto_pool = _build_pool_from_archive(
        pareto_archive,
        feature_infos,
        dont_cares,
        engine,
        metrics_by_chrom=metrics_by_chrom,
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
) -> tuple[list[dict], list[dict]]:
    """Run Phase 2 NSGA-III evolution. Returns (pareto_pool, history)."""
    if not _EVOX_AVAILABLE:
        logger.warning(
            "EvoX not available; falling back to NumPy NSGA-II for Phase 2.",
        )
        return _run_nsga2_fallback(
            feature_infos, engine, pop_size, n_generations, rng,
            seed_chromosomes=seed_chromosomes,
            log_tag=log_tag or "NSGA-II (fallback)",
        )

    return _run_nsga3(
        feature_infos, engine, pop_size, n_generations, rng,
        seed_chromosomes=seed_chromosomes, log_tag=log_tag,
    )
