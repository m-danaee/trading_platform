"""
evox_runner.py — Phase 2 multi-objective evolution via EvoX selection operators.

Uses EvoX reference-vector sampling and RVEA / NSGA-III environmental selection
on integer fuzzy-rule chromosomes, with fitness from GPUBacktestEngine batch eval.
Falls back to NumPy NSGA-II when EvoX is unavailable or algorithm is NSGA2.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

from gpu_fuzzy_trader import config as _cfg

logger = logging.getLogger(__name__)

RunnerKind = Literal["nsga2", "rvea", "nsga3"]

_EVOX_AVAILABLE = False
try:
    import torch
    from evox.operators.sampling.uniform import uniform_sampling
    from evox.operators.selection.rvea_selection import ref_vec_guided

    _EVOX_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    uniform_sampling = None  # type: ignore[assignment]
    ref_vec_guided = None  # type: ignore[assignment]


def resolve_phase2_runner(
    algorithm: str | None = None,
    pop_size: int | None = None,
) -> RunnerKind:
    """Map config algorithm name to internal runner kind."""
    algo = (algorithm or _cfg.PHASE2_ALGORITHM).upper().replace(
        "-", "").replace("_", "")
    pop = pop_size if pop_size is not None else _cfg.PHASE2_POPULATION_SIZE

    if algo in ("NSGA2", "NSGAII"):
        return "nsga2"

    use_nsga3 = algo in ("NSGA3", "TENSORNSGA3", "TENSOR_NSGA3")
    if not use_nsga3 and _cfg.PHASE2_TENSOR_NSGA3 and pop >= _cfg.PHASE2_LARGE_POP_THRESHOLD:
        use_nsga3 = True

    if use_nsga3 and pop >= _cfg.PHASE2_LARGE_POP_THRESHOLD:
        return "nsga3"

    if algo in ("MOEAD", "MOPSO"):
        logger.warning(
            "PHASE2_ALGORITHM=%s not fully wired; using RVEA-style search.",
            algorithm,
        )

    return "rvea"


def _get_reference_vectors(pop_size: int, n_objs: int = 3) -> np.ndarray:
    """Das-Dennis style reference vectors; uses EvoX uniform_sampling when available."""
    if _EVOX_AVAILABLE and uniform_sampling is not None:
        w, n_samples = uniform_sampling(pop_size, n_objs)
        return w.cpu().numpy()[:pop_size]

    # Simple 3-objective fallback: corners + uniform face points
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
        chroms = population[pending]
        metrics_list = engine.simulate_rule_batch(
            chromosomes=chroms,
            tp=_cfg.PHASE2_TP,
            sl=_cfg.PHASE2_SL,
            capital_pct=_cfg.PHASE2_CAPITAL_PCT,
        )
        from gpu_fuzzy_trader.phases.phase2_rule_pool import (
            _count_active_conditions,
            _hamming_distance,
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

            total_return = float(metrics.get("total_return_pct", 0.0))
            max_dd = float(metrics.get("max_drawdown_pct", 100.0))
            win_rate = float(metrics.get("win_rate", 0.0))
            executed = int(metrics.get("executed_trades", 0))

            support_penalty = 0.0
            if executed < _cfg.MIN_TRADE_SUPPORT:
                support_penalty = (_cfg.MIN_TRADE_SUPPORT - executed) * 0.5

            diversity_penalty = 0.0
            if pareto_archive:
                min_hamming = min(
                    _hamming_distance(chromosome, pf) for pf in pareto_archive
                )
                if min_hamming == 0:
                    diversity_penalty = 5.0

            pen = support_penalty + diversity_penalty + cond_penalty
            objectives[i] = np.array(
                [-total_return + pen, max_dd + pen, -win_rate + pen],
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


def _rvea_survivors(
    merge_pop: np.ndarray,
    merge_fit: np.ndarray,
    ref_vec: np.ndarray,
    pop_size: int,
    gen: int,
    max_gen: int,
    alpha: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Select pop_size survivors using EvoX ref_vec_guided."""
    if not _EVOX_AVAILABLE or ref_vec_guided is None:
        from gpu_fuzzy_trader.phases.phase2_rule_pool import _non_dominated_sort

        fronts = _non_dominated_sort(merge_fit)
        order: list[int] = []
        for front in fronts:
            order.extend(front)
            if len(order) >= pop_size:
                break
        idx = order[:pop_size]
        return merge_pop[idx], merge_fit[idx]

    theta = (gen / max(max_gen, 1)) ** alpha
    x = torch.tensor(merge_pop.astype(np.float32))
    f = torch.tensor(merge_fit.astype(np.float32))
    v = torch.tensor(ref_vec.astype(np.float32))
    if v.shape[0] < pop_size:
        extra, _ = uniform_sampling(pop_size, f.shape[1])  # type: ignore[misc]
        v = torch.tensor(extra.cpu().numpy().astype(np.float32))

    next_x, next_f = ref_vec_guided(
        x, f, v[: max(v.shape[0], pop_size)], torch.tensor(theta))

    valid_mask = ~torch.isnan(next_x).all(dim=1)
    n_valid = int(valid_mask.sum().item())
    survivors_x = next_x[valid_mask].cpu().numpy()
    survivors_f = next_f[valid_mask].cpu().numpy()

    if n_valid < pop_size:
        remaining = pop_size - n_valid
        order = np.argsort(merge_fit[:, 0])
        fill_idx = [int(idx) for idx in order[:remaining]]
        if fill_idx:
            survivors_x = np.vstack([survivors_x, merge_pop[fill_idx]])
            survivors_f = np.vstack([survivors_f, merge_fit[fill_idx]])

    if len(survivors_x) > pop_size:
        survivors_x = survivors_x[:pop_size]
        survivors_f = survivors_f[:pop_size]
    elif len(survivors_x) < pop_size:
        order = np.argsort(merge_fit[:, 0])[: pop_size - len(survivors_x)]
        survivors_x = np.vstack([survivors_x, merge_pop[order]])
        survivors_f = np.vstack([survivors_f, merge_fit[order]])

    return np.rint(survivors_x).astype(np.int32), survivors_f


def _adapt_reference_vectors(
    ref_vec: np.ndarray,
    pop_fit: np.ndarray,
    init_ref: np.ndarray,
    gen: int,
    adapt_every: int,
) -> np.ndarray:
    """RVEA reference vector adaptation (scale by objective range)."""
    if gen <= 0 or gen % adapt_every != 0:
        return ref_vec
    max_vals = np.nanmax(pop_fit, axis=0)
    min_vals = np.nanmin(pop_fit, axis=0)
    span = np.maximum(max_vals - min_vals, 1e-6)
    return init_ref * span


def _run_reference_vector_moea(
    feature_infos: list[dict],
    engine,
    pop_size: int,
    n_generations: int,
    rng: np.random.Generator,
    runner: RunnerKind,
) -> tuple[list[dict], list[dict]]:
    """RVEA or NSGA-III style loop (NSGA3 uses same mating, RVEA selection for now)."""
    from gpu_fuzzy_trader.phases.phase2_rule_pool import (
        _build_pool_from_archive,
        _crossover,
        _get_dont_cares,
        _init_population,
        _mutate,
        _non_dominated_sort,
    )

    K = len(feature_infos)
    dont_cares = _get_dont_cares(feature_infos)
    population = _init_population(pop_size, feature_infos, rng)
    objectives = np.full((pop_size, 3), np.inf)
    metrics_cache: list[dict] = [{} for _ in range(pop_size)]
    pareto_archive: list[np.ndarray] = []
    history: list[dict] = []

    init_ref = _get_reference_vectors(pop_size, 3)
    ref_vec = init_ref.copy()
    adapt_every = max(1, int(round(1.0 / 0.1)))  # fr=0.1
    alpha = 2.0

    tag = "RVEA" if runner == "rvea" else "NSGA-III"
    logger.info("%s: %d features, pop=%d, gen=%d",
                tag, K, pop_size, n_generations)

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
            "algorithm": tag,
        })

        if gen == n_generations - 1:
            break

        # Mating pool + offspring
        offspring_list: list[np.ndarray] = []
        all_indices = list(range(pop_size))
        for _ in range(0, pop_size, 2):
            cands_a = rng.choice(all_indices, size=2, replace=False)
            cands_b = rng.choice(all_indices, size=2, replace=False)
            pa = (
                cands_a[0]
                if objectives[cands_a[0], 0] <= objectives[cands_a[1], 0]
                else cands_a[1]
            )
            pb = (
                cands_b[0]
                if objectives[cands_b[0], 0] <= objectives[cands_b[1], 0]
                else cands_b[1]
            )
            child_a, child_b = _crossover(population[pa], population[pb], rng)
            offspring_list.append(
                _mutate(child_a, feature_infos, dont_cares, rng)
            )
            offspring_list.append(
                _mutate(child_b, feature_infos, dont_cares, rng)
            )

        offspring = np.stack(offspring_list[:pop_size], axis=0)
        off_obj = np.full((pop_size, 3), np.inf)

        _evaluate_population_indices(
            offspring,
            list(range(pop_size)),
            dont_cares,
            engine,
            pareto_archive,
            off_obj,
            [{} for _ in range(pop_size)],
        )

        merge_pop = np.vstack([population, offspring])
        merge_fit = np.vstack([objectives, off_obj])

        if runner == "nsga3" and _EVOX_AVAILABLE:
            population, objectives = _nsga3_environmental_selection(
                merge_pop, merge_fit, ref_vec, pop_size
            )
        else:
            population, objectives = _rvea_survivors(
                merge_pop, merge_fit, ref_vec, pop_size, gen, n_generations, alpha
            )

        ref_vec = _adapt_reference_vectors(
            ref_vec, objectives, init_ref, gen + 1, adapt_every
        )
        metrics_cache = [{} for _ in range(pop_size)]

    pareto_pool = _build_pool_from_archive(
        pareto_archive, feature_infos, dont_cares, engine
    )
    return pareto_pool, history


def _nsga3_environmental_selection(
    merge_pop: np.ndarray,
    merge_fit: np.ndarray,
    ref: np.ndarray,
    pop_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """NSGA-III environmental selection via EvoX non_dominate_rank + niche count."""
    from evox.operators.selection.non_dominate import non_dominate_rank

    f = torch.tensor(merge_fit.astype(np.float32))
    rank = non_dominate_rank(f)
    rank_np = rank.cpu().numpy()
    worst_rank = int(torch.topk(rank, pop_size + 1,
                     largest=False)[0][-1].item())
    candi_idx = np.where(rank_np <= worst_rank)[0]

    if len(candi_idx) <= pop_size:
        return merge_pop[candi_idx[:pop_size]], merge_fit[candi_idx[:pop_size]]

    selected = [i for i in candi_idx if rank_np[i] < worst_rank]
    if len(selected) >= pop_size:
        return merge_pop[selected[:pop_size]], merge_fit[selected[:pop_size]]

    need = pop_size - len(selected)
    last_idx = [i for i in candi_idx if rank_np[i] == worst_rank]

    norm_fit = merge_fit - merge_fit.min(axis=0)
    norm_fit = norm_fit / np.maximum(norm_fit.max(axis=0), 1e-6)
    ref_n = ref / np.linalg.norm(ref, axis=1, keepdims=True).clip(1e-10)
    fit_n = norm_fit / \
        np.linalg.norm(norm_fit, axis=1, keepdims=True).clip(1e-10)
    dist = 1.0 - fit_n @ ref_n.T
    assoc = np.argmin(dist, axis=1)

    rho = np.bincount(assoc[last_idx], minlength=len(ref))
    order = np.argsort(rho)
    for j in order[:need]:
        if j < len(last_idx):
            selected.append(last_idx[j])

    selected = selected[:pop_size]
    return merge_pop[selected], merge_fit[selected]


def run_phase2_evolution(
    feature_infos: list[dict],
    engine,
    pop_size: int,
    n_generations: int,
    rng: np.random.Generator,
    algorithm: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Run Phase 2 evolution with the configured algorithm.

    Returns (pareto_pool, history).
    """
    kind = resolve_phase2_runner(algorithm, pop_size)

    if kind == "nsga2":
        from gpu_fuzzy_trader.phases.phase2_rule_pool import _run_nsga2

        return _run_nsga2(feature_infos, engine, pop_size, n_generations, rng)

    if not _EVOX_AVAILABLE:
        logger.warning("EvoX not available; falling back to NumPy NSGA-II.")
        from gpu_fuzzy_trader.phases.phase2_rule_pool import _run_nsga2

        return _run_nsga2(feature_infos, engine, pop_size, n_generations, rng)

    return _run_reference_vector_moea(
        feature_infos, engine, pop_size, n_generations, rng, kind
    )
