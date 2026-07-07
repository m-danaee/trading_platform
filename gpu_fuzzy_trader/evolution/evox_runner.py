"""
evox_runner.py — Phase 2 NSGA-III multi-objective evolution.

Uses EvoX reference-vector sampling and non-dominated ranking with niche
selection on integer fuzzy-rule chromosomes. Falls back to NumPy NSGA-II
environmental selection when EvoX is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gpu_fuzzy_trader.phases.phase2_rule_pool import (
    _crossover,
    _mutate,
)
from gpu_fuzzy_trader.evolution.numba_ops import (
    batch_hamming_min,
    crowding_distance,
    non_dominated_sort,
)

import logging
import time

import numpy as np

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.log_progress import maybe_log_generation
from gpu_fuzzy_trader.phases.phase2_stage import (
    Phase2StageParams,
    StageLabel,
    resolve_phase2_stage_params,
)

logger = logging.getLogger(__name__)

# Number of NSGA-III objectives (3 or 4). Controlled by PHASE2_F4_ENABLED.
# All np.full((pop_size, ...), inf) and _get_reference_vectors calls use this.
# Reads the centralized PHASE2_N_OBJECTIVES from config when enabled (→ fixes
# audit finding #2: prevents drift between config and evox_runner array sizes).
N_OBJ = (int(getattr(_cfg, "PHASE2_N_OBJECTIVES", 4))
         if getattr(_cfg, "PHASE2_F4_ENABLED", False)
         else 3)


@dataclass
class Phase2EvolutionState:
    """Resumable NSGA-III state for symbol-island epoch scheduling."""

    population: np.ndarray
    objectives: np.ndarray
    metrics_cache: list[dict]
    pareto_archive: list[np.ndarray]
    hall_of_fame: dict[tuple[int, ...], np.ndarray]
    deployable_archive: dict[tuple[int, ...], dict]
    global_metrics_cache: dict[tuple[int, ...], dict]
    history: list[dict] = field(default_factory=list)
    generation_offset: int = 0
    plateau_best_progress: float = -np.inf
    plateau_streak: int = 0
    ref_vec: np.ndarray | None = None
    mutation_rate: float | None = None
    weighted_activate_prob: float | None = None
    stage: StageLabel = None
    post_restart_gens_remaining: int = 0
    post_restart_no_improve_streak: int = 0
    post_restart_best_progress: float = -np.inf
    restart_count: int = 0


def trim_evolution_state_memory(
    state: Phase2EvolutionState,
    *,
    pop_size: int | None = None,
) -> None:
    """Drop bulky resumable state that is already persisted elsewhere."""
    state.history.clear()
    # Keep only 1x pop_size entries in the global cache after each direction
    # to free RAM before the next direction starts (Colab 12 GiB guard).
    max_cache = max(200, int(pop_size or len(state.population)))
    _trim_global_metrics_cache(state.global_metrics_cache, max_cache)


_EVOX_WARNED = False


def _warn_evox_unavailable() -> None:
    global _EVOX_WARNED
    if not _EVOX_AVAILABLE and not _EVOX_WARNED:
        logger.warning(
            "EvoX unavailable (%s); falling back to NSGA-II. "
            "Pipeline config may specify NSGA3 — verify evox/torch installation.",
            _EVOX_IMPORT_ERROR,
        )
        _EVOX_WARNED = True


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


def _should_run_val_this_gen(gen: int, is_last_gen: bool) -> bool:
    """Shared val-cadence check for both NSGA-II fallback and NSGA-III loops.

    Val simulation always runs on the last generation (pool admission needs
    fresh metrics).  Otherwise, throttle via ``PHASE2_VAL_SIM_INTERVAL``.

    ``PHASE2_JOINT_TRAIN_VAL`` does NOT affect cadence — it controls whether
    val metrics *feed fitness*, not how often val is computed.  This was the
    root cause of the 2x blowup: the old ``or bool(_cfg.PHASE2_JOINT_TRAIN_VAL)``
    short-circuited the throttle whenever joint training was enabled.
    """
    if is_last_gen:
        return True
    interval = max(1, int(_cfg.PHASE2_VAL_SIM_INTERVAL))
    return gen % interval == 0


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
    *,
    max_entries: int = 300,
) -> None:
    """Accumulate unique Pareto chromosomes discovered across generations."""
    from gpu_fuzzy_trader.phases.phase2_sparse_encoding import chromosome_key

    for i in pareto_indices:
        chrom = population[int(i)].copy()
        hall_of_fame[chromosome_key(chrom)] = chrom
    overflow = len(hall_of_fame) - int(max_entries)
    if overflow > 0:
        for key in list(hall_of_fame.keys())[:overflow]:
            del hall_of_fame[key]


def _val_metrics_from_cache(metrics: dict) -> dict | None:
    """Reconstruct validation metrics sidecar stored on train metrics."""
    if metrics.get("val_total_return_pct") is None:
        return None
    return {
        "total_return_pct": float(metrics.get("val_total_return_pct", 0.0)),
        "profit_factor": float(metrics.get("val_profit_factor", 0.0)),
        "executed_trades": int(metrics.get("val_executed_trades", 0)),
        "sortino_ratio": float(
            metrics.get("val_sortino_ratio", metrics.get(
                "val_total_return_pct", 0.0))
        ),
        "max_drawdown_pct": float(metrics.get("val_max_drawdown_pct", 0.0)),
    }


def _inherit_val_metrics_from_global_cache(
    metrics: dict,
    key: tuple[int, ...],
    global_metrics_cache: dict[tuple[int, ...], dict] | None,
    *,
    run_val: bool,
) -> None:
    """Copy val_* from global cache for identical chromosomes when val is skipped.

    Replaces the old slot-index inheritance that copied unrelated parents'
    val metrics into offspring at the same population index.
    """
    if run_val:
        return
    if not _cfg.PHASE2_EVAL_GLOBAL_CACHE or global_metrics_cache is None:
        return
    cached = global_metrics_cache.get(key)
    if cached is None:
        return
    for k, v in cached.items():
        if k.startswith("val_") and k not in metrics:
            metrics[k] = v
    sidecar = cached.get("_cached_val_metrics")
    if isinstance(sidecar, dict):
        mapping = {
            "total_return_pct": "val_total_return_pct",
            "profit_factor": "val_profit_factor",
            "executed_trades": "val_executed_trades",
            "sortino_ratio": "val_sortino_ratio",
            "max_drawdown_pct": "val_max_drawdown_pct",
        }
        for src, dst in mapping.items():
            if src in sidecar and dst not in metrics:
                metrics[dst] = sidecar[src]


def _build_diversity_reference(
    hall_of_fame: dict[tuple[int, ...], np.ndarray],
    pareto_archive: list[np.ndarray],
    *,
    max_refs: int = 500,
) -> list[np.ndarray]:
    """Chromosomes used for Hamming diversity pressure beyond the live Pareto set."""
    refs: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    for chrom in list(hall_of_fame.values()) + list(pareto_archive):
        from gpu_fuzzy_trader.phases.phase2_sparse_encoding import chromosome_key

        key = chromosome_key(chrom)
        if key in seen:
            continue
        seen.add(key)
        refs.append(chrom.copy())
        if len(refs) >= int(max_refs):
            break
    return refs


def _deduplicate_selection_indices(
    selected_idx: np.ndarray,
    merge_pop: np.ndarray,
    merge_fit: np.ndarray,
    pop_size: int,
) -> np.ndarray:
    """Replace duplicate genotypes in survivors with unused unique merge rows."""
    from gpu_fuzzy_trader.phases.phase2_sparse_encoding import chromosome_key

    chosen = [int(i) for i in selected_idx[:pop_size]]
    seen: set[tuple[int, ...]] = set()
    survivors: list[int] = []

    for idx in chosen:
        key = chromosome_key(merge_pop[idx])
        if key not in seen:
            seen.add(key)
            survivors.append(idx)

    if len(survivors) < pop_size:
        chosen_set = set(chosen)
        extras = sorted(
            (i for i in range(len(merge_pop)) if i not in chosen_set),
            key=lambda i: (float(np.sum(merge_fit[i])), i),
        )
        for idx in extras:
            if len(survivors) >= pop_size:
                break
            key = chromosome_key(merge_pop[idx])
            if key not in seen:
                seen.add(key)
                survivors.append(idx)

    if len(survivors) < pop_size:
        for idx in chosen:
            if len(survivors) >= pop_size:
                break
            if idx not in survivors:
                survivors.append(idx)

    return np.asarray(survivors[:pop_size], dtype=np.intp)


def _pareto_robust_stats(
    pareto_indices: list[int],
    metrics_cache: list[dict],
) -> dict[str, float]:
    """Aggregate min(train, val) return and Sortino over the Pareto front."""
    if not pareto_indices:
        return {
            "mean_robust_return_pct": 0.0,
            "max_robust_return_pct": 0.0,
            "max_robust_sortino": 0.0,
        }
    from gpu_fuzzy_trader.phases.phase2_rule_pool import _saturating_sortino
    from gpu_fuzzy_trader.phases.phase2_support import robust_return_pct

    robust_returns: list[float] = []
    robust_sortinos: list[float] = []
    for i in pareto_indices:
        train_m = metrics_cache[i]
        val_m = _val_metrics_from_cache(train_m)
        robust_returns.append(robust_return_pct(train_m, val_m))
        train_s = _saturating_sortino(float(
            train_m.get("sortino_ratio", train_m.get("total_return_pct", 0.0)),
        ))
        if val_m is None:
            robust_sortinos.append(train_s)
        else:
            val_s = _saturating_sortino(float(
                val_m.get("sortino_ratio", val_m.get("total_return_pct", 0.0)),
            ))
            robust_sortinos.append(min(train_s, val_s))

    return {
        "mean_robust_return_pct": float(np.mean(robust_returns)),
        "max_robust_return_pct": float(np.max(robust_returns)),
        "max_robust_sortino": float(np.max(robust_sortinos)),
    }


def _pareto_train_val_gap_stats(
    pareto_indices: list[int],
    metrics_cache: list[dict],
) -> dict[str, float]:
    """Max train-vs-val return gap and ratio over the Pareto front."""
    if not pareto_indices:
        return {
            "max_train_val_gap_pct": 0.0,
            "max_train_val_gap_ratio": 0.0,
        }
    gaps: list[float] = []
    ratios: list[float] = []
    for i in pareto_indices:
        train_ret = float(metrics_cache[i].get("total_return_pct", 0.0))
        val_m = _val_metrics_from_cache(metrics_cache[i])
        if val_m is None:
            continue
        val_ret = float(val_m.get("total_return_pct", 0.0))
        gaps.append(train_ret - val_ret)
        if val_ret > 0.0:
            ratios.append(train_ret / val_ret)
    return {
        "max_train_val_gap_pct": float(max(gaps)) if gaps else 0.0,
        "max_train_val_gap_ratio": float(max(ratios)) if ratios else 0.0,
    }


def _pareto_holdout_stats(
    pareto_indices: list[int],
    population: np.ndarray,
    pool_val_engine,
    *,
    generation: int | None = None,
    is_last_gen: bool = False,
) -> dict[str, float | None]:
    """Holdout-only robust snapshot for logging (not used in selection)."""
    empty: dict[str, float | None] = {
        "max_holdout_return_pct": None,
        "max_holdout_sortino": None,
    }
    if pool_val_engine is None or not pareto_indices:
        return empty
    from gpu_fuzzy_trader.phases.phase2_rule_pool import _saturating_sortino

    chroms = np.stack(
        [population[int(i)].copy() for i in pareto_indices],
        axis=0,
    )
    try:
        holdout_metrics = pool_val_engine.simulate_rule_batch(
            chromosomes=chroms,
            tp=_cfg.PHASE2_TP,
            sl=_cfg.PHASE2_SL,
            capital_pct=_cfg.PHASE2_CAPITAL_PCT,
            generation=generation,
            is_last_gen=is_last_gen,
        )
    except Exception as exc:
        logger.debug("Holdout snapshot simulate_rule_batch failed: %s", exc)
        return empty

    returns: list[float] = []
    sortinos: list[float] = []
    for metrics in holdout_metrics:
        returns.append(float(metrics.get("total_return_pct", 0.0)))
        sortinos.append(_saturating_sortino(float(
            metrics.get("sortino_ratio", metrics.get("total_return_pct", 0.0)),
        )))
    return {
        "max_holdout_return_pct": float(max(returns)) if returns else None,
        "max_holdout_sortino": float(max(sortinos)) if sortinos else None,
    }


def _population_rank_diagnostics(
    fronts: list[list[int]],
    pop_size: int,
) -> dict[str, float]:
    """Non-dominated rank distribution across the live population."""
    if not fronts or pop_size <= 0:
        return {
            "rank0_fraction": 0.0,
            "rank1_fraction": 0.0,
            "max_rank": 0.0,
        }
    ranks = np.full(pop_size, len(fronts), dtype=np.int32)
    for rank, front in enumerate(fronts):
        for i in front:
            ranks[int(i)] = rank
    return {
        "rank0_fraction": float(np.sum(ranks == 0) / pop_size),
        "rank1_fraction": float(np.sum(ranks == 1) / pop_size),
        "max_rank": float(np.max(ranks)),
    }


def _refresh_survivor_val_metrics(
    population: np.ndarray,
    survivor_indices: list[int],
    metrics_cache: list[dict],
    objectives: np.ndarray,
    dont_cares: np.ndarray,
    pareto_archive: list[np.ndarray],
    val_engine,
    *,
    diversity_reference: list[np.ndarray] | None = None,
    diversity_metrics_by_key: dict[tuple[int, ...], dict] | None = None,
    stage_params: Phase2StageParams | None = None,
    global_metrics_cache: dict[tuple[int, ...], dict] | None = None,
    generation: int | None = None,
    is_last_gen: bool = False,
) -> None:
    """Re-run val backtests for unevaluated survivors and refresh objectives."""
    if val_engine is None or not survivor_indices:
        return
    from gpu_fuzzy_trader.phases.phase2_sparse_encoding import chromosome_key

    n_valid_rows = (
        int(getattr(val_engine, "n_valid_rows"))
        if getattr(val_engine, "n_valid_rows", None) is not None
        else None
    )
    chroms = np.stack(
        [population[int(i)].copy() for i in survivor_indices],
        axis=0,
    )
    try:
        val_metrics_list = val_engine.simulate_rule_batch(
            chromosomes=chroms,
            tp=_cfg.PHASE2_TP,
            sl=_cfg.PHASE2_SL,
            capital_pct=_cfg.PHASE2_CAPITAL_PCT,
            generation=generation,
            is_last_gen=is_last_gen,
        )
    except Exception as exc:
        logger.debug(
            "Survivor val refresh simulate_rule_batch failed: %s", exc)
        return

    for j, idx in enumerate(survivor_indices):
        i = int(idx)
        val_metrics = _metrics_snapshot(val_metrics_list[j])
        train_metrics = _metrics_snapshot(metrics_cache[i])
        for k in list(train_metrics.keys()):
            if k.startswith("val_"):
                train_metrics.pop(k, None)
        _assign_eval_result(
            i,
            population[i],
            train_metrics,
            val_metrics,
            dont_cares,
            pareto_archive,
            objectives,
            metrics_cache,
            diversity_reference=diversity_reference,
            diversity_metrics_by_key=diversity_metrics_by_key,
            stage_params=stage_params,
            n_valid_rows=n_valid_rows,
        )
        if _cfg.PHASE2_EVAL_GLOBAL_CACHE and global_metrics_cache is not None:
            _store_global_metrics_cache(
                global_metrics_cache,
                chromosome_key(population[i]),
                metrics_cache[i],
                val_metrics,
            )


def _update_deployable_archive(
    deployable_archive: dict[tuple[int, ...], dict],
    population: np.ndarray,
    indices: list[int],
    metrics_cache: list[dict],
) -> None:
    """Keep best deployability-preview candidates across generations."""
    from gpu_fuzzy_trader.phases.phase2_sparse_encoding import chromosome_key
    from gpu_fuzzy_trader.phases.phase2_support import (
        deployability_rank_score,
        passes_evolution_deployability_preview,
    )

    max_size = int(_cfg.PHASE2_DEPLOYABLE_ARCHIVE_MAX_SIZE)
    for i in indices:
        metrics = metrics_cache[int(i)]
        if not metrics:
            continue
        val_metrics = _val_metrics_from_cache(metrics)
        if not passes_evolution_deployability_preview(
            metrics,
            val_metrics,
        ):
            continue
        chrom = population[int(i)].copy()
        key = chromosome_key(chrom)
        rank = deployability_rank_score(metrics, val_metrics)
        incumbent = deployable_archive.get(key)
        if incumbent is None or rank > float(incumbent.get("rank_score", -np.inf)):
            deployable_archive[key] = {
                "chromosome": chrom,
                "rank_score": rank,
                "metrics": _metrics_snapshot(metrics),
            }

    if len(deployable_archive) <= max_size:
        return
    ranked = sorted(
        deployable_archive.items(),
        key=lambda kv: float(kv[1].get("rank_score", -np.inf)),
        reverse=True,
    )[:max_size]
    deployable_archive.clear()
    deployable_archive.update(dict(ranked))


def _harvest_archive_chromosomes(
    deployable_archive: dict[tuple[int, ...], dict],
    hall_of_fame: dict[tuple[int, ...], np.ndarray],
    pareto_archive: list[np.ndarray],
) -> list[np.ndarray]:
    """Prefer deployable archive; fall back to Pareto hall-of-fame."""
    if deployable_archive:
        ranked = sorted(
            deployable_archive.values(),
            key=lambda entry: float(entry.get("rank_score", -np.inf)),
            reverse=True,
        )
        return [entry["chromosome"] for entry in ranked]
    if hall_of_fame:
        return list(hall_of_fame.values())
    return list(pareto_archive)


def _pareto_robust_return_pct(
    pareto_indices: list[int],
    metrics_cache: list[dict],
) -> float:
    """Best min(train, val) return across the current Pareto front."""
    if not pareto_indices:
        return -100.0
    from gpu_fuzzy_trader.phases.phase2_support import robust_return_pct

    scores = [
        robust_return_pct(
            metrics_cache[i],
            _val_metrics_from_cache(metrics_cache[i]),
        )
        for i in pareto_indices
    ]
    return float(max(scores))


def _count_deployable_preview(
    pareto_indices: list[int],
    metrics_cache: list[dict],
) -> int:
    from gpu_fuzzy_trader.phases.phase2_support import (
        passes_evolution_deployability_preview,
    )

    count = 0
    for i in pareto_indices:
        metrics = metrics_cache[i]
        if not metrics:
            continue
        if passes_evolution_deployability_preview(
            metrics,
            _val_metrics_from_cache(metrics),
        ):
            count += 1
    return count


def _plateau_progress_metric(
    pareto_indices: list[int],
    metrics_cache: list[dict],
) -> float:
    if bool(getattr(_cfg, "PHASE2_PLATEAU_USE_ROBUST_RETURN", True)):
        return _pareto_robust_return_pct(pareto_indices, metrics_cache)
    returns = [
        float(metrics_cache[i].get("total_return_pct", 0.0))
        for i in pareto_indices
    ]
    return max(returns) if returns else -100.0


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
    stage_params: Phase2StageParams | None = None,
    *,
    island_profile: str = "global",
) -> bool:
    if _cfg.scoped_island_profile(island_profile):
        if not _cfg.island_early_stop_enabled():
            return False
    elif not _cfg.PHASE2_EARLY_STOP_ENABLED:
        return False
    min_gen = (
        int(stage_params.early_stop_min_generation)
        if stage_params is not None
        else int(_cfg.PHASE2_EARLY_STOP_MIN_GENERATION)
    )
    if gen + 1 < min_gen:
        return False
    if pareto_return_pct >= float(_cfg.PHASE2_EARLY_STOP_MEAN_RETURN_PCT):
        return False
    min_valid = int(_cfg.PHASE2_EARLY_STOP_MIN_VALID_RULES)
    if valid_count >= min_valid:
        return False
    return True


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


def _resolve_plateau_patience(
    stage_params: Phase2StageParams | None,
    island_profile: str,
) -> int:
    """Resolve the plateau patience value based on profile and stage.

    Islands run single-stage (stage=None); the stage_params patience is
    the GLOBAL default baked into the None profile, NOT the island knob.
    Use the island-scoped config directly so
    PHASE2_ISLAND_PLATEAU_EARLY_STOP_PATIENCE actually takes effect.
    """
    if _cfg.scoped_island_profile(island_profile):
        return int(getattr(
            _cfg, "PHASE2_ISLAND_PLATEAU_EARLY_STOP_PATIENCE",
            _cfg.PHASE2_PLATEAU_EARLY_STOP_PATIENCE,
        ))
    if stage_params is not None:
        return int(stage_params.plateau_early_stop_patience)
    return int(_cfg.PHASE2_PLATEAU_EARLY_STOP_PATIENCE)


def _should_plateau_early_stop_phase2(
    gen: int,
    streak: int,
    *,
    deployable_count: int = 0,
    unique_chromosome_ratio: float = 1.0,
    population_unique_ratio: float | None = None,
    stage_params: Phase2StageParams | None = None,
    island_profile: str = "global",
) -> bool:
    if _cfg.scoped_island_profile(island_profile):
        if not _cfg.island_plateau_early_stop_enabled():
            return False
    elif not _cfg.PHASE2_PLATEAU_EARLY_STOP_ENABLED:
        return False
    min_gen = (
        int(stage_params.plateau_early_stop_min_generation)
        if stage_params is not None
        else int(_cfg.PHASE2_PLATEAU_EARLY_STOP_MIN_GENERATION)
    )
    if gen + 1 < min_gen:
        return False
    if _cfg.scoped_island_profile(island_profile):
        if bool(getattr(_cfg, "PHASE2_ISLAND_PLATEAU_BLOCK_WHEN_DEPLOYABLE_ZERO", False)):
            if deployable_count <= 0:
                return False
    elif bool(getattr(_cfg, "PHASE2_PLATEAU_BLOCK_WHEN_DEPLOYABLE_ZERO", True)):
        if deployable_count <= 0:
            return False
    if bool(getattr(_cfg, "PHASE2_PLATEAU_BLOCK_WHEN_DIVERSITY_LOW", True)):
        diversity_ratio = (
            population_unique_ratio
            if population_unique_ratio is not None
            else unique_chromosome_ratio
        )
        if diversity_ratio < _diversity_recovery_min_unique_ratio(stage_params):
            return False
    patience = _resolve_plateau_patience(stage_params, island_profile)
    return streak >= patience


def _should_post_restart_early_stop_phase2(
    post_restart_streak: int,
    *,
    island_profile: str = "global",
    stage_params: Phase2StageParams | None = None,
) -> bool:
    """Break the epoch when a plateau restart yields no improvement.

    Independent of the main plateau streak/patience so it never interferes
    with the restart decision itself — it only fires on generations AFTER a
    restart, measuring whether the restart produced any progress.
    """
    if _cfg.scoped_island_profile(island_profile):
        if not bool(getattr(
            _cfg, "PHASE2_ISLAND_PLATEAU_POST_RESTART_STOP_ENABLED", True,
        )):
            return False
        patience = int(getattr(
            _cfg, "PHASE2_ISLAND_PLATEAU_POST_RESTART_STOP_PATIENCE",
            getattr(_cfg, "PHASE2_PLATEAU_POST_RESTART_STOP_PATIENCE", 3),
        ))
    else:
        if not bool(getattr(
            _cfg, "PHASE2_PLATEAU_POST_RESTART_STOP_ENABLED", True,
        )):
            return False
        patience = int(getattr(
            _cfg, "PHASE2_PLATEAU_POST_RESTART_STOP_PATIENCE", 3,
        ))
    return post_restart_streak >= patience


def _should_viability_recovery(
    stage_params: Phase2StageParams | None,
    *,
    valid_count: int = 0,
    plateau_streak: int = 0,
) -> bool:
    """True when Stage A viability is critically low and search has plateaued."""
    if not bool(getattr(_cfg, "PHASE2_VIABILITY_RECOVERY_ENABLED", True)):
        return False
    if stage_params is None or stage_params.stage != "A":
        return False
    return (
        int(valid_count) < int(_cfg.PHASE2_VIABILITY_RECOVERY_MIN_VALID)
        and int(plateau_streak) >= 2
    )


def _inject_diversity_recovery(
    population: np.ndarray,
    objectives: np.ndarray,
    metrics_cache: list[dict],
    feature_infos: list[dict],
    dont_cares: np.ndarray,
    rng: np.random.Generator,
    *,
    feature_probs: np.ndarray | None = None,
    stage_params: Phase2StageParams | None = None,
    deployable_archive: dict | None = None,
    viability_recovery: bool = False,
) -> list[int]:
    """Replace a fraction of the population with fresh or archive-mutated seeds."""
    from gpu_fuzzy_trader.phases.phase2_rule_pool import _init_population, _mutate

    pop_size = len(population)
    if pop_size <= 0:
        return []
    fraction = (
        float(stage_params.diversity_recovery_inject_fraction)
        if stage_params is not None
        else float(
            getattr(_cfg, "PHASE2_DIVERSITY_RECOVERY_INJECT_FRACTION", 0.25)
        )
    )
    n_inject = max(1, int(round(pop_size * fraction)))
    positions = rng.choice(pop_size, size=n_inject, replace=False)
    mutation_rate = _stage_mutation_rate(stage_params, diversity_recovery=True)
    weighted_activate_prob = _stage_weighted_activate_prob(stage_params)

    seeds: list[np.ndarray] = []
    if (
        viability_recovery
        and deployable_archive
        and len(deployable_archive) > 0
    ):
        mutate_frac = float(
            _cfg.PHASE2_VIABILITY_RECOVERY_DEPLOYABLE_MUTATE_FRACTION
        )
        n_mutate = max(0, int(round(n_inject * mutate_frac)))
        n_fresh = n_inject - n_mutate
        ranked = sorted(
            deployable_archive.values(),
            key=lambda entry: float(entry.get("rank_score", -np.inf)),
            reverse=True,
        )
        archive_chroms = [entry["chromosome"] for entry in ranked]
        # Select a diverse subset from the top-ranked archive entries
        # to maximise genetic diversity in the injected seeds
        diverse_archive = _select_diverse_subset(archive_chroms, n_mutate)
        for parent in diverse_archive:
            seeds.append(
                _mutate(
                    parent,
                    feature_infos,
                    dont_cares,
                    rng,
                    mutation_rate=mutation_rate,
                    feature_probs=feature_probs,
                    weighted_activate_prob=weighted_activate_prob,
                )
            )
        if n_fresh > 0:
            seeds.extend(
                _init_population(
                    n_fresh,
                    feature_infos,
                    rng,
                    init_strategy=_cfg.PHASE2_INIT_STRATEGY,
                    stratum_fractions=_cfg.PHASE2_INIT_STRATUM_FRACTIONS,
                    feature_probs=None,
                )
            )
    else:
        seeds = list(
            _init_population(
                n_inject,
                feature_infos,
                rng,
                init_strategy=_cfg.PHASE2_INIT_STRATEGY,
                stratum_fractions=_cfg.PHASE2_INIT_STRATUM_FRACTIONS,
                feature_probs=None,
            )
        )

    injected: list[int] = []
    for idx, pos in enumerate(positions):
        population[int(pos)] = seeds[idx]
        objectives[int(pos)] = np.inf
        metrics_cache[int(pos)] = {}
        injected.append(int(pos))
    return injected


def _population_unique_chromosome_ratio(population: np.ndarray) -> float:
    """Unique genotypes in the full population (not just the Pareto front)."""
    from gpu_fuzzy_trader.phases.phase2_sparse_encoding import chromosome_key

    n = len(population)
    if n <= 0:
        return 0.0
    keys = {chromosome_key(population[i]) for i in range(n)}
    return float(len(keys) / n)


def _population_genotype_diversity_ratio(population: np.ndarray) -> float:
    """Median pairwise Hamming distance normalized by mean chromosome length."""
    from gpu_fuzzy_trader.phases.phase2_sparse_encoding import (
        count_active_slots,
        is_sparse_chromosome,
    )

    def _chrom_len(chrom: np.ndarray) -> int:
        chrom = np.asarray(chrom)
        if is_sparse_chromosome(chrom):
            return max(1, count_active_slots(chrom) * 2)
        return max(1, int(chrom.size))

    n = len(population)
    if n < 2:
        return 0.0
    chroms = [population[i] for i in range(n)]
    median_hamming = _median_pairwise_hamming(chroms)
    avg_len = float(np.mean([_chrom_len(c) for c in chroms]))
    return float(median_hamming / avg_len)


def _hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Hamming distance between two chromosomes (sparse-aware)."""
    from gpu_fuzzy_trader.phases.phase2_sparse_encoding import (
        is_sparse_chromosome, sparse_hamming,
    )
    if is_sparse_chromosome(a) or is_sparse_chromosome(b):
        return sparse_hamming(a, b)
    return int(np.sum(a != b))


def _select_diverse_subset(
    chromosomes: list[np.ndarray],
    k: int,
) -> list[np.ndarray]:
    """Max-min Hamming diversity sampling: greedy pick farthest from chosen.

    Returns up to *k* chromosomes from *chromosomes* that maximise pairwise
    Hamming distance.  If *k* >= len(chromosomes), returns a copy of the input.
    """
    if k <= 0:
        return []
    if len(chromosomes) <= k:
        return list(chromosomes)
    chosen = [chromosomes[0]]
    remaining = list(chromosomes[1:])
    while len(chosen) < k and remaining:
        best_idx = max(
            range(len(remaining)),
            key=lambda i: min(
                _hamming_distance(remaining[i], c) for c in chosen
            ),
        )
        chosen.append(remaining.pop(best_idx))
    return chosen


def _plateau_diversity_restart(
    population: np.ndarray,
    objectives: np.ndarray,
    metrics_cache: list[dict],
    feature_infos: list[dict],
    rng: np.random.Generator,
    *,
    pareto_indices: list[int],
    pop_size: int,
    init_strategy: str | None = None,
    stratum_fractions: tuple[float, float] | None = None,
) -> int:
    """Reinit a fraction of the population on plateau, preserving Pareto elite.

    Keeps the top K = min(10, max(3, len(pareto_indices))) elite chromosomes
    ranked by ``robust_return_pct`` (fallback: ``total_return_pct``), then
    reinitialises a configurable fraction of the remaining slots via
    ``_init_population``.  Resets those slots' objectives to ``inf`` and
    ``metrics_cache`` to ``{}``.

    Returns the number of elite chromosomes kept.
    """
    from gpu_fuzzy_trader.phases.phase2_rule_pool import _init_population

    n_elite = min(10, max(3, len(pareto_indices)))

    def _elite_score(idx: int) -> float:
        m = metrics_cache[idx] if idx < len(metrics_cache) else {}
        if m.get("robust_return_pct") is not None:
            return float(m["robust_return_pct"])
        return float(m.get("total_return_pct", -1e9))

    ranked = sorted(pareto_indices, key=_elite_score, reverse=True)
    elite_positions = set(ranked[:n_elite])
    fraction = float(
        getattr(_cfg, "PHASE2_PLATEAU_DIVERSITY_RESTART_FRACTION", 0.40)
    )
    # Determine which slots to reinit: all non-elite slots
    all_slots = list(range(pop_size))
    reinit_slots = [i for i in all_slots if i not in elite_positions]
    n_reinit = max(0, min(len(reinit_slots), int(round(pop_size * fraction))))
    # Pick a random subset of non-elite slots
    chosen = list(
        rng.choice(reinit_slots, size=n_reinit, replace=False)
    ) if n_reinit > 0 else []

    if n_reinit > 0:
        seeds = _init_population(
            n_reinit,
            feature_infos,
            rng,
            init_strategy=init_strategy,
            stratum_fractions=stratum_fractions,
            feature_probs=None,
        )
        for idx, pos in enumerate(chosen):
            population[pos] = seeds[idx]
            objectives[pos] = np.full(N_OBJ, np.inf)
            metrics_cache[pos] = {}

    return n_elite


def _diversity_recovery_min_unique_ratio(
    stage_params: Phase2StageParams | None = None,
) -> float:
    if stage_params is not None:
        return float(stage_params.diversity_recovery_min_unique_ratio)
    return float(
        getattr(_cfg, "PHASE2_DIVERSITY_RECOVERY_MIN_UNIQUE_RATIO", 0.30)
    )


def _should_inject_diversity_recovery(
    population_unique_ratio: float,
    stage_params: Phase2StageParams | None = None,
    *,
    pareto_size: int = 0,
    plateau_streak: int = 0,
    pop_size: int = 0,
    valid_count: int = 0,
) -> bool:
    if not bool(getattr(_cfg, "PHASE2_DIVERSITY_RECOVERY_ENABLED", True)):
        return False
    if (
        population_unique_ratio
        < _diversity_recovery_min_unique_ratio(stage_params)
    ):
        return True
    collapse_threshold = max(2, int(pop_size) // 40)
    if (
        pareto_size > 0
        and pareto_size <= collapse_threshold
        and int(plateau_streak) >= 2
    ):
        return True
    # Check 3: Phenotype collapse — Pareto front ≤3 despite high genetic uniqueness
    # This catches the case where chromosomes are genetically unique but phenotypically
    # identical (same trading behavior, different gene encoding).
    # Note: For pop_size >= 120, Check 2 (threshold = pop_size // 40) already covers
    # pareto_size <= 3. Check 3 adds explicit coverage for smaller populations
    # (e.g., island runs with pop_size < 120) and serves as a safety net if Check 2
    # thresholds are tightened in the future.
    if pareto_size > 0 and pareto_size <= 3 and int(plateau_streak) >= 2:
        return True
    if _should_viability_recovery(
        stage_params,
        valid_count=valid_count,
        plateau_streak=plateau_streak,
    ):
        return True
    return False


def _count_pop_viable(
    pop_size: int,
    metrics_cache: list[dict],
) -> int:
    """Population-wide count passing the pool trade floor."""
    from gpu_fuzzy_trader.phases.phase2_support import passes_pool_trade_floor

    return sum(
        1
        for i in range(pop_size)
        if metrics_cache[i]
        and passes_pool_trade_floor(
            int(metrics_cache[i].get("executed_trades", 0)),
            metrics_cache[i],
        )
    )


def _stage_mutation_rate(
    stage_params: Phase2StageParams | None,
    *,
    diversity_recovery: bool = False,
) -> float:
    base = (
        float(stage_params.mutation_rate)
        if stage_params is not None
        else float(_cfg.PHASE2_MUTATION_RATE)
    )
    if not diversity_recovery:
        return base
    boost = (
        float(stage_params.diversity_recovery_mutation_boost)
        if stage_params is not None
        else float(
            getattr(_cfg, "PHASE2_DIVERSITY_RECOVERY_MUTATION_BOOST", 1.5)
        )
    )
    return min(0.5, base * boost)


def _stage_weighted_activate_prob(
    stage_params: Phase2StageParams | None,
) -> float:
    if stage_params is not None:
        return float(stage_params.mutation_weighted_activate_prob)
    return float(_cfg.PHASE2_MUTATION_WEIGHTED_ACTIVATE_PROB)


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
        std_all = np.std(pareto_obj, axis=0)
        std_f1 = float(std_all[0]) if len(std_all) > 0 else 0.0
        std_f2 = float(std_all[1]) if len(std_all) > 1 else 0.0
        std_f3 = float(std_all[2]) if len(std_all) > 2 else 0.0
        std_f4 = float(std_all[3]) if len(std_all) > 3 else 0.0
    else:
        std_f1 = std_f2 = std_f3 = std_f4 = 0.0

    corr_f1_f2 = corr_f1_f3 = corr_f2_f3 = 0.0
    if len(pareto_obj) >= 2:
        obj = np.asarray(pareto_obj, dtype=np.float64)
        if obj.shape[1] >= 2 and np.std(obj[:, 0]) > 1e-12 and np.std(obj[:, 1]) > 1e-12:
            corr_f1_f2 = float(np.corrcoef(obj[:, 0], obj[:, 1])[0, 1])
        if obj.shape[1] >= 3 and np.std(obj[:, 0]) > 1e-12 and np.std(obj[:, 2]) > 1e-12:
            corr_f1_f3 = float(np.corrcoef(obj[:, 0], obj[:, 2])[0, 1])
        if obj.shape[1] >= 3 and np.std(obj[:, 1]) > 1e-12 and np.std(obj[:, 2]) > 1e-12:
            corr_f2_f3 = float(np.corrcoef(obj[:, 1], obj[:, 2])[0, 1])

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
        "objective_std_f4": float(std_f4) if N_OBJ >= 4 else 0.0,
        "objective_corr_f1_f2": corr_f1_f2,
        "objective_corr_f1_f3": corr_f1_f3,
        "objective_corr_f2_f3": corr_f2_f3,
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
    mutation_rate: float | None = None,
    weighted_activate_prob: float | None = None,
) -> np.ndarray:
    """Generate pop_size offspring via binary tournament, crossover, mutation.

    *fronts* may be passed in from the caller to avoid recomputing the sort.
    """
    if fronts is None:
        fronts = non_dominated_sort(objectives)
    rank, crowding = _build_rank_and_crowding(objectives, fronts)
    all_indices = list(range(pop_size))
    mut_rate = (
        float(mutation_rate)
        if mutation_rate is not None
        else float(_cfg.PHASE2_MUTATION_RATE)
    )
    offspring_list: list[np.ndarray] = []
    for _ in range(0, pop_size, 2):
        pa = _binary_tournament_pick(all_indices, rank, crowding, rng)
        pb = _binary_tournament_pick(all_indices, rank, crowding, rng)
        child_a, child_b = _crossover(population[pa], population[pb], rng)
        offspring_list.append(
            _mutate(
                child_a, feature_infos, dont_cares, rng,
                mutation_rate=mut_rate,
                feature_probs=feature_probs,
                weighted_activate_prob=weighted_activate_prob,
            )
        )
        offspring_list.append(
            _mutate(
                child_b, feature_infos, dont_cares, rng,
                mutation_rate=mut_rate,
                feature_probs=feature_probs,
                weighted_activate_prob=weighted_activate_prob,
            )
        )
    return np.stack(offspring_list[:pop_size], axis=0)


def _das_dennis(n_partitions: int, n_objs: int) -> np.ndarray:
    """Das-Dennis reference vectors on the unit simplex."""
    from itertools import product

    points: list[list[float]] = []
    for combo in product(range(n_partitions + 1), repeat=n_objs):
        if sum(combo) == n_partitions:
            points.append([c / n_partitions for c in combo])
    return np.array(points, dtype=np.float64)


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

    # Das-Dennis fallback: increase partitions until >= pop_size
    n_partitions = n_objs  # start small
    while True:
        refs = _das_dennis(n_partitions, n_objs)
        if len(refs) >= pop_size:
            return refs[:pop_size]
        n_partitions += 1
        if n_partitions > 100:  # safety
            break
    # Pad with random simplex points if still short
    fallback_rng = rng if rng is not None else np.random.default_rng()
    while len(refs) < pop_size:
        r = fallback_rng.random(n_objs - 1)
        r = np.sort(r)
        last = np.concatenate([[r[0]], np.diff(r), [1 - r[-1]]])
        refs = np.vstack([refs, last])
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
    diversity_reference: list[np.ndarray] | None = None,
    diversity_metrics_by_key: dict[tuple[int, ...], dict] | None = None,
    stage_params: Phase2StageParams | None = None,
    engine=None,
    n_valid_rows: int | None = None,
) -> None:
    """Compute penalties/objectives from metrics; write objectives[i] and metrics_cache[i]."""
    from gpu_fuzzy_trader.phases.phase2_rule_pool import (
        compute_phase2_objectives_from_metrics,
    )

    obj, processed = compute_phase2_objectives_from_metrics(
        chromosome,
        dont_cares,
        metrics,
        pareto_archive,
        val_metrics=val_metrics,
        diversity_reference=diversity_reference,
        diversity_metrics_by_key=diversity_metrics_by_key,
        stage_params=stage_params,
        n_valid_rows=n_valid_rows,
    )
    objectives[i] = obj
    metrics_cache[i] = processed


def _metrics_snapshot(metrics: dict) -> dict:
    """Shallow copy for evolution caches (metrics are flat numeric dicts)."""
    return dict(metrics)


def _store_global_metrics_cache(
    global_metrics_cache: dict[tuple[int, ...], dict],
    key: tuple[int, ...],
    metrics: dict,
    val_metrics: dict | None,
) -> None:
    """Store processed metrics (and optional val sidecar) in the run-wide cache."""
    entry = _metrics_snapshot(metrics)
    if val_metrics is not None:
        entry["_cached_val_metrics"] = _metrics_snapshot(val_metrics)
    global_metrics_cache[key] = entry
    if _cfg.PHASE2_EVAL_GLOBAL_CACHE:
        max_cache = int(getattr(_cfg, "PHASE2_EVAL_GLOBAL_CACHE_MAX_SIZE", 575))
        _trim_global_metrics_cache(global_metrics_cache, max_cache)


def _trim_global_metrics_cache(
    global_metrics_cache: dict[tuple[int, ...], dict],
    max_size: int,
) -> None:
    """Bound run-wide eval cache size to limit RAM growth across long runs.

    Uses FIFO eviction (oldest entries first) to preserve recent evaluations
    and maximize cache hit rates across generations.
    """
    overflow = len(global_metrics_cache) - int(max_size)
    if overflow <= 0:
        return
    # Remove oldest entries (dict preserves insertion order)
    keys_to_remove = list(global_metrics_cache.keys())[:overflow]
    for key in keys_to_remove:
        global_metrics_cache.pop(key, None)


def clear_global_metrics_cache(
    cache: dict | None = None,
) -> None:
    """Clear the global eval cache and force GC.

    Used to free RAM between cluster runs in island-scheduled Phase 2.
    If *cache* is provided, calls ``cache.clear()``; otherwise clears the
    module-level cache if one exists.
    """
    if cache is not None:
        cache.clear()
    else:
        # Fallback for any module-level cache (currently none defined).
        pass
    import gc as _gc
    _gc.collect()


def _load_global_metrics_cache(
    cached: dict,
) -> tuple[dict, dict | None]:
    """Restore metrics and optional val_metrics from a cache entry."""
    entry = _metrics_snapshot(cached)
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
    global_metrics_cache: dict[tuple[int, ...], dict] | None = None,
    diversity_reference: list[np.ndarray] | None = None,
    diversity_metrics_by_key: dict[tuple[int, ...], dict] | None = None,
    stage_params: Phase2StageParams | None = None,
    run_val: bool = True,
    generation: int | None = None,
    is_last_gen: bool = False,
    cv_fold_evaluator=None,
) -> dict[str, int]:
    """Evaluate unevaluated individuals, preferring batch simulate_rule_batch."""
    from gpu_fuzzy_trader.phases.phase2_rule_pool import (
        _evaluate_chromosome,
        attach_cv_fold_returns_batch,
    )
    from gpu_fuzzy_trader.phases.phase2_sparse_encoding import chromosome_key

    pending = [i for i in indices if np.any(np.isinf(objectives[i]))]
    if not pending:
        return _empty_eval_stats()

    n_valid_rows = (
        int(getattr(val_engine, "n_valid_rows"))
        if val_engine is not None and getattr(val_engine, "n_valid_rows", None)
        else None
    )

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
                    diversity_reference=diversity_reference,
                    diversity_metrics_by_key=diversity_metrics_by_key,
                    stage_params=stage_params,
                    engine=engine,
                    n_valid_rows=n_valid_rows,
                )
                cache_hits += 1
            else:
                gpu_pending.append(i)
    else:
        gpu_pending = list(pending)

    if not gpu_pending:
        if cache_hits:
            max_cache = int(getattr(_cfg, "PHASE2_EVAL_GLOBAL_CACHE_MAX_SIZE", 575))
            cache_hit_rate = cache_hits / max(1, len(pending))
            logger.debug(
                "Phase 2 eval cache: size=%d capacity=%d (hit_rate this gen: %.3f)",
                len(global_metrics_cache), max_cache, cache_hit_rate,
            )
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
            generation=generation,
            is_last_gen=is_last_gen,
        )
        attach_cv_fold_returns_batch(
            metrics_list,
            unique_chroms,
            cv_fold_evaluator,
        )

        val_metrics_list = None
        if val_engine is not None and run_val:
            try:
                val_metrics_list = val_engine.simulate_rule_batch(
                    chromosomes=unique_chroms,
                    tp=_cfg.PHASE2_TP,
                    sl=_cfg.PHASE2_SL,
                    capital_pct=_cfg.PHASE2_CAPITAL_PCT,
                    generation=generation,
                    is_last_gen=is_last_gen,
                )
            except Exception as exc:
                logger.debug(
                    "val simulate_rule_batch failed in Phase 2 batch path: %s",
                    exc,
                )
                val_metrics_list = None

        unique_count = len(unique_chroms)
        max_cache = int(getattr(_cfg, "PHASE2_EVAL_GLOBAL_CACHE_MAX_SIZE", 575))
        cache_hit_rate = cache_hits / max(1, len(pending))
        logger.debug(
            "Phase 2 eval cache: size=%d capacity=%d (hit_rate this gen: %.3f)",
            len(global_metrics_cache) if global_metrics_cache is not None else 0,
            max_cache, cache_hit_rate,
        )
        logger.debug(
            "Phase 2 eval: %d cache hits, %d GPU pending, %d unique chromosomes",
            cache_hits,
            len(gpu_pending),
            unique_count,
        )

        for i in gpu_pending:
            key = chromosome_key(population[i])
            uj = key_to_uj[key]
            metrics = _metrics_snapshot(metrics_list[uj])
            val_metrics = None
            if val_metrics_list is not None:
                val_metrics = _metrics_snapshot(val_metrics_list[uj])

            # When val was not computed this gen, inherit val_* only from an
            # identical chromosome in the global cache (never from slot index).
            _inherit_val_metrics_from_global_cache(
                metrics,
                key,
                global_metrics_cache,
                run_val=run_val,
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
                diversity_reference=diversity_reference,
                diversity_metrics_by_key=diversity_metrics_by_key,
                stage_params=stage_params,
                engine=engine,
                n_valid_rows=n_valid_rows,
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
                diversity_reference=diversity_reference,
                diversity_metrics_by_key=diversity_metrics_by_key,
                stage_params=stage_params,
                cv_fold_evaluator=cv_fold_evaluator,
                run_val=run_val,
                generation=generation,
                is_last_gen=is_last_gen,
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
    """Rank-based normalization (robust to outliers like trade_penalty=50)."""
    fit = np.asarray(merge_fit, dtype=np.float64)
    # Unevaluated (inf) or corrupt rows must not reach NSGA-III association.
    fit = np.where(np.isfinite(fit), fit, 1e12)
    n = len(fit)
    rank_fit = np.empty_like(fit)
    for j in range(fit.shape[1]):
        # Average-rank normalisation (robust to ties, no scipy dependency)
        order = np.argsort(fit[:, j], kind="mergesort")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(1, n + 1, dtype=np.float64)
        # Adjust ties: replace equal values with mean rank
        sorted_fit = fit[order, j]
        i = 0
        while i < n:
            j_tie = i + 1
            while j_tie < n and sorted_fit[j_tie] == sorted_fit[i]:
                j_tie += 1
            if j_tie - i > 1:
                avg_rank = float(np.mean(ranks[order[i:j_tie]]))
                ranks[order[i:j_tie]] = avg_rank
            i = j_tie
        rank_fit[:, j] = ranks / n  # normalise to [0, 1]
    ref_n = ref / np.linalg.norm(ref, axis=1, keepdims=True).clip(1e-10)
    fit_n = rank_fit / np.linalg.norm(rank_fit, axis=1, keepdims=True).clip(1e-10)
    return fit_n, ref_n


def _reevaluate_infinite_objectives(
    population: np.ndarray,
    objectives: np.ndarray,
    metrics_cache: list[dict],
    indices: list[int] | None,
    dont_cares: np.ndarray,
    engine,
    pareto_archive: list[np.ndarray],
    *,
    val_engine=None,
    global_metrics_cache: dict[tuple[int, ...], dict] | None = None,
    diversity_reference: list[np.ndarray] | None = None,
    diversity_metrics_by_key: dict[tuple[int, ...], dict] | None = None,
    stage_params: Phase2StageParams | None = None,
    run_val: bool = True,
    cv_fold_evaluator=None,
) -> dict[str, int]:
    """Evaluate any individuals still marked with inf objectives."""
    if indices is None:
        pending = [
            i for i in range(len(population))
            if np.any(np.isinf(objectives[i]))
        ]
    else:
        pending = [i for i in indices if np.any(np.isinf(objectives[i]))]
    if not pending:
        return _empty_eval_stats()
    return _evaluate_population_indices(
        population,
        pending,
        dont_cares,
        engine,
        pareto_archive,
        objectives,
        metrics_cache,
        val_engine=val_engine,
        global_metrics_cache=global_metrics_cache,
        diversity_reference=diversity_reference,
        diversity_metrics_by_key=diversity_metrics_by_key,
        stage_params=stage_params,
        run_val=run_val,
        cv_fold_evaluator=cv_fold_evaluator,
    )


def _nsga3_environmental_selection(
    merge_pop: np.ndarray,
    merge_fit: np.ndarray,
    ref: np.ndarray,
    pop_size: int,
    feature_infos: list[dict],
    dont_cares: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NSGA-III environmental selection (rank + niche on last front)."""
    _warn_evox_unavailable()
    if not _EVOX_AVAILABLE or non_dominate_rank is None:
        pop, fit, selected = environmental_selection_nsga2(
            merge_pop, merge_fit, pop_size)
        idx = _deduplicate_selection_indices(
            np.asarray(selected, dtype=np.intp),
            merge_pop,
            merge_fit,
            pop_size,
        )
        return (
            _repair_population(merge_pop[idx], feature_infos, dont_cares),
            merge_fit[idx],
            idx,
        )

    f = torch.tensor(merge_fit.astype(np.float32))
    rank = non_dominate_rank(f)
    rank_np = rank.cpu().numpy()
    worst_rank = int(torch.topk(rank, pop_size + 1,
                     largest=False)[0][-1].item())
    candi_idx = np.where(rank_np <= worst_rank)[0]

    if len(candi_idx) <= pop_size:
        idx = _deduplicate_selection_indices(
            candi_idx[:pop_size], merge_pop, merge_fit, pop_size,
        )
        return (
            _repair_population(merge_pop[idx], feature_infos, dont_cares),
            merge_fit[idx],
            idx,
        )

    selected = [int(i) for i in candi_idx if rank_np[i] < worst_rank]
    if len(selected) >= pop_size:
        idx = _deduplicate_selection_indices(
            np.asarray(selected[:pop_size], dtype=np.intp),
            merge_pop,
            merge_fit,
            pop_size,
        )
        return (
            _repair_population(merge_pop[idx], feature_infos, dont_cares),
            merge_fit[idx],
            idx,
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

    idx = _deduplicate_selection_indices(
        np.asarray(selected[:pop_size], dtype=np.intp),
        merge_pop,
        merge_fit,
        pop_size,
    )
    return (
        _repair_population(merge_pop[idx], feature_infos, dont_cares),
        merge_fit[idx],
        idx,
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
    pool_val_engine=None,
    cv_fold_evaluator=None,
    holdout_n_valid_rows: int | None = None,
    train_n_rows: int | None = None,
    feature_probs: np.ndarray | None = None,
    init_strategy: str | None = None,
    stratum_fractions: tuple[float, float] | None = None,
    seed_fraction: float | None = None,
    reset_plateau: bool = False,
    stage: StageLabel = None,
    island_profile: str = "global",
    island_hyperparams: _cfg.IslandHyperparams | None = None,
) -> tuple[list[dict], list[dict]]:
    """NumPy NSGA-II loop when EvoX is not installed."""
    _warn_evox_unavailable()
    from gpu_fuzzy_trader.phases.phase2_rule_pool import (
        _build_pool_from_archive,
        _evaluate_chromosome,
        _get_dont_cares,
        _init_population,
        _metrics_dict_from_population,
        _pareto_sortino_stats,
    )
    from gpu_fuzzy_trader.phases.phase2_support import passes_pool_trade_floor

    stage_params = resolve_phase2_stage_params(stage)
    K = len(feature_infos)
    dont_cares = _get_dont_cares(feature_infos)
    if seed_fraction is None:
        seed_fraction = stage_params.seed_fraction
    population = _init_population(
        pop_size,
        feature_infos,
        rng,
        seeded_chromosomes=seed_chromosomes,
        seed_fraction=seed_fraction,
        init_strategy=init_strategy,
        stratum_fractions=stratum_fractions,
        feature_probs=feature_probs,
    )
    objectives = np.full((pop_size, N_OBJ), np.inf)
    metrics_cache: list[dict] = [{} for _ in range(pop_size)]
    pareto_archive: list[np.ndarray] = []
    hall_of_fame: dict[tuple[int, ...], np.ndarray] = {}
    deployable_archive: dict[tuple[int, ...], dict] = {}
    global_metrics_cache: dict[tuple[int, ...], dict] = {}
    history: list[dict] = []

    tag = log_tag or "NSGA-II (fallback)"
    logger.info(
        "%s: %d features, pop=%d, gen=%d, mutation=%.3f",
        tag, K, pop_size, n_generations, stage_params.mutation_rate,
    )
    gen_loop_start = time.monotonic()
    if reset_plateau:
        plateau_best_progress = -np.inf
        plateau_streak = 0
        restart_count = 0
        post_restart_gens_remaining = 0
        post_restart_no_improve_streak = 0
        post_restart_best_progress = -np.inf
    else:
        plateau_best_progress = -np.inf
        plateau_streak = 0
        restart_count = 0
        post_restart_gens_remaining = 0
        post_restart_no_improve_streak = 0
        post_restart_best_progress = -np.inf
    mutation_rate = _stage_mutation_rate(stage_params)
    weighted_activate_prob = _stage_weighted_activate_prob(stage_params)
    # restart_count is per-stage (Stage A and Stage B each have their own),
    # deliberately more lenient than per-epoch.
    viability_collapse_streak = 0
    max_restarts = int(getattr(
        _cfg, "PHASE2_PLATEAU_MAX_RESTARTS", 1,
    ))

    for gen in range(n_generations):
        just_restarted = False
        is_last_gen = gen == n_generations - 1
        run_val_this_gen = _should_run_val_this_gen(gen, is_last_gen)
        from gpu_fuzzy_trader.phases.phase2_sparse_encoding import chromosome_key

        pre_survivors = [
            i for i in range(pop_size)
            if not np.any(np.isinf(objectives[i]))
        ]
        for i in range(pop_size):
            if np.any(np.isinf(objectives[i])):
                obj, metrics = _evaluate_chromosome(
                    population[i], dont_cares, engine, pareto_archive,
                    val_engine=val_engine,
                    stage_params=stage_params,
                    cv_fold_evaluator=cv_fold_evaluator,
                    run_val=run_val_this_gen,
                    generation=gen,
                    is_last_gen=is_last_gen,
                )
                objectives[i] = obj
                key = chromosome_key(population[i])
                _inherit_val_metrics_from_global_cache(
                    metrics,
                    key,
                    global_metrics_cache,
                    run_val=run_val_this_gen,
                )
                metrics_cache[i] = metrics
                if _cfg.PHASE2_EVAL_GLOBAL_CACHE:
                    val_m = _val_metrics_from_cache(metrics)
                    _store_global_metrics_cache(
                        global_metrics_cache,
                        key,
                        metrics,
                        val_m,
                    )
        if run_val_this_gen and val_engine is not None and pre_survivors:
            _refresh_survivor_val_metrics(
                population,
                pre_survivors,
                metrics_cache,
                objectives,
                dont_cares,
                pareto_archive,
                val_engine,
                global_metrics_cache=global_metrics_cache,
                stage_params=stage_params,
                generation=gen,
                is_last_gen=is_last_gen,
            )

        fronts = non_dominated_sort(objectives)
        pareto_indices = fronts[0]
        pareto_archive = [population[i].copy() for i in pareto_indices]
        _update_hall_of_fame(hall_of_fame, population, pareto_indices)
        if run_val_this_gen:
            _update_deployable_archive(
                deployable_archive,
                population,
                list(range(pop_size)),
                metrics_cache,
            )

        pareto_obj = objectives[pareto_indices]
        pareto_diag = _pareto_diagnostics(
            pareto_indices, metrics_cache, pareto_obj, population,
        )
        deployable_count = _count_deployable_preview(
            pareto_indices,
            metrics_cache,
        )
        unique_ratio = float(pareto_diag.get("unique_chromosome_ratio", 0.0))
        pop_unique_ratio = _population_unique_chromosome_ratio(population)
        pop_diversity_ratio = _population_genotype_diversity_ratio(population)
        plateau_metric = _plateau_progress_metric(
            pareto_indices, metrics_cache)
        robust_stats = _pareto_robust_stats(pareto_indices, metrics_cache)
        gap_stats = _pareto_train_val_gap_stats(pareto_indices, metrics_cache)
        rank_diag = _population_rank_diagnostics(fronts, pop_size)
        holdout_stats: dict[str, float | None] = {
            "max_holdout_return_pct": None,
            "max_holdout_sortino": None,
        }
        if is_last_gen and pool_val_engine is not None:
            holdout_stats = _pareto_holdout_stats(
                pareto_indices,
                population,
                pool_val_engine,
                generation=gen,
                is_last_gen=is_last_gen,
            )
        history.append({
            "generation": gen,
            "pareto_size": len(pareto_indices),
            "mean_f1": float(np.mean(pareto_obj[:, 0])),
            "mean_f2": float(np.mean(pareto_obj[:, 1])),
            "mean_f3": float(np.mean(pareto_obj[:, 2])),
            "algorithm": "NSGA-II (fallback)",
            "deployable_preview_count": deployable_count,
            "deployable_archive_size": len(deployable_archive),
            "plateau_progress_pct": plateau_metric,
            "pop_genotype_diversity_ratio": pop_diversity_ratio,
            **_pareto_sortino_stats(pareto_indices, metrics_cache),
            **robust_stats,
            **gap_stats,
            **rank_diag,
            **holdout_stats,
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
            )
        )
        pop_viable_count = _count_pop_viable(
            pop_size,
            metrics_cache,
        )
        # --- viability-collapse trigger: forced restart when viable pop is
        #     persistently below threshold ---
        pop_size_for_viability = max(pop_size, 1)
        viability_threshold = int(
            getattr(_cfg, "PHASE2_VIABILITY_COLLAPSE_THRESHOLD", 0.5)
            * pop_size_for_viability
        )
        if pop_viable_count < viability_threshold:
            viability_collapse_streak += 1
        else:
            viability_collapse_streak = 0
        viability_collapse_patience = int(
            getattr(_cfg, "PHASE2_VIABILITY_COLLAPSE_STREAK", 3)
        )
        if (
            viability_collapse_streak >= viability_collapse_patience
            and restart_count < max_restarts
        ):
            n_elite_kept = _plateau_diversity_restart(
                population, objectives, metrics_cache,
                feature_infos, rng,
                pareto_indices=pareto_indices,
                pop_size=pop_size,
                init_strategy=init_strategy,
                stratum_fractions=stratum_fractions,
            )
            restart_count += 1
            pre_reset_streak = viability_collapse_streak
            viability_collapse_streak = 0
            post_restart_gens_remaining = int(getattr(
                _cfg,
                "PHASE2_PLATEAU_POST_RESTART_BOOST_GENS",
                3,
            ))
            post_restart_best_progress = -np.inf
            post_restart_no_improve_streak = 0
            just_restarted = True
            logger.info(
                "Phase 2 [%s]: viability-collapse restart at gen %d "
                "(pop_viable=%d < %d, streak=%d)",
                tag, gen + 1, pop_viable_count, viability_threshold,
                pre_reset_streak,
            )
        plateau_best_progress, plateau_streak = _update_max_return_plateau(
            plateau_metric, plateau_best_progress, plateau_streak,
        )
        if restart_count > 0 and not just_restarted:
            post_restart_best_progress, post_restart_no_improve_streak = (
                _update_max_return_plateau(
                    plateau_metric,
                    post_restart_best_progress,
                    post_restart_no_improve_streak,
                )
            )
        maybe_log_generation(
            logger, tag, gen, n_generations, len(pareto_indices), mean_ret,
            max_return_pct=max_ret,
            median_return_pct=median_ret,
            max_sortino=max_sort,
            mean_robust_return_pct=robust_stats["mean_robust_return_pct"],
            max_robust_return_pct=robust_stats["max_robust_return_pct"],
            max_robust_sortino=robust_stats["max_robust_sortino"],
            max_train_val_gap_pct=gap_stats["max_train_val_gap_pct"],
            max_train_val_gap_ratio=gap_stats["max_train_val_gap_ratio"],
            max_holdout_return_pct=holdout_stats.get("max_holdout_return_pct"),
            max_holdout_sortino=holdout_stats.get("max_holdout_sortino"),
            objective_corr_f1_f3=float(
                pareto_diag.get("objective_corr_f1_f3", 0.0)),
            rank0_fraction=float(rank_diag.get("rank0_fraction", 0.0)),
            valid_count=val_count,
            unique_chromosome_ratio=float(
                pareto_diag.get("unique_chromosome_ratio", 0.0)),
            pop_unique_chromosome_ratio=pop_diversity_ratio,
            deployable_count=deployable_count,
            pop_viable_count=pop_viable_count,
            plateau_streak=plateau_streak,
            restarts="%d/%d" % (restart_count, max_restarts),
            loop_start=gen_loop_start,
        )

        if val_count < 10:
            from gpu_fuzzy_trader.phases.phase2_support import (
                _feasibility_gate_failures,
            )
            breakdown = {k: 0 for k in [
                "train_trade_floor", "train_return_floor", "train_pf_floor",
                "val_required", "val_ret_positive", "val_trade_floor",
                "val_return_floor", "val_pf_floor", "train_val_gap",
                "f4_concentration",
            ]}
            for i in range(len(metrics_cache)):
                train_m = metrics_cache[i]
                val_m = _val_metrics_from_cache(train_m)
                gate_fails = _feasibility_gate_failures(train_m, val_m)
                for k in breakdown:
                    breakdown[k] += gate_fails[k]
            logger.warning(
                "Phase 2 [%s] gen %d/%d: feasibility collapse breakdown "
                "(valid_rules=%d < 10): %s",
                tag, gen + 1, n_generations, val_count, breakdown,
            )

        pareto_ret = _pareto_return_pct_for_early_stop(
            pareto_indices, metrics_cache,
        )
        if _should_early_stop_phase2(
            gen, pareto_ret, val_count, stage_params=stage_params,
            island_profile=island_profile,
        ):
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

        if _should_plateau_early_stop_phase2(
            gen,
            plateau_streak,
            deployable_count=deployable_count,
            unique_chromosome_ratio=unique_ratio,
            population_unique_ratio=pop_diversity_ratio,
            stage_params=stage_params,
            island_profile=island_profile,
        ):
            # --- diversity restart on first plateau, break on second ---
            restart_enabled = bool(getattr(
                _cfg, "PHASE2_PLATEAU_DIVERSITY_RESTART_ENABLED", True,
            ))
            max_restarts = int(getattr(
                _cfg, "PHASE2_PLATEAU_MAX_RESTARTS", 1,
            ))
            if (
                restart_enabled
                and max_restarts > 0
                and restart_count < max_restarts
            ):
                remaining_gens = n_generations - gen - 1
                if remaining_gens > 0:
                    elite_kept = _plateau_diversity_restart(
                        population, objectives, metrics_cache,
                        feature_infos, rng,
                        pareto_indices=pareto_indices,
                        pop_size=pop_size,
                        init_strategy=init_strategy,
                        stratum_fractions=stratum_fractions,
                    )
                    # Set multi-generation mutation boost instead of one-shot
                    post_restart_gens_remaining = int(getattr(
                        _cfg,
                        "PHASE2_PLATEAU_POST_RESTART_BOOST_GENS",
                        3,
                    ))
                    restart_count += 1
                    plateau_streak = 0
                    post_restart_best_progress = -np.inf
                    post_restart_no_improve_streak = 0
                    just_restarted = True
                    logger.info(
                        "Phase 2 [%s]: plateau restart at gen %d "
                        "(restart %d/%d, reinit %.0f%%, elite_kept=%d, mutation=%.2f)",
                        tag,
                        gen + 1,
                        restart_count,
                        max_restarts,
                        float(_cfg.PHASE2_PLATEAU_DIVERSITY_RESTART_FRACTION) * 100,
                        elite_kept,
                        mutation_rate,
                    )
                    continue  # skip env selection; go to next gen
            patience_used = _resolve_plateau_patience(stage_params, island_profile)
            logger.info(
                "%s: plateau early stop at gen %d (progress=%.2f%% unchanged "
                "for %d gens, patience=%d, min_delta=%.2f%%, "
                "deployable_preview=%d, pop_unique=%.2f)",
                tag,
                gen + 1,
                plateau_metric,
                plateau_streak,
                patience_used,
                float(_cfg.PHASE2_PLATEAU_EARLY_STOP_MIN_DELTA_PCT),
                deployable_count,
                pop_diversity_ratio,
            )
            break

        if (
            restart_count > 0
            and _should_post_restart_early_stop_phase2(
                post_restart_no_improve_streak,
                stage_params=stage_params,
                island_profile=island_profile,
            )
        ):
            logger.info(
                "%s: post-restart early stop at gen %d "
                "(no improvement for %d gens after restart %d/%d, "
                "best_progress=%.2f%%, deployable_preview=%d)",
                tag,
                gen + 1,
                post_restart_no_improve_streak,
                restart_count,
                max_restarts,
                plateau_best_progress,
                deployable_count,
            )
            break

        if gen == n_generations - 1:
            break

        injected_positions: list[int] = []
        viability_recovery = _should_viability_recovery(
            stage_params,
            valid_count=val_count,
            plateau_streak=plateau_streak,
        )
        if _should_inject_diversity_recovery(
            pop_unique_ratio,
            stage_params=stage_params,
            pareto_size=len(pareto_indices),
            plateau_streak=plateau_streak,
            pop_size=pop_size,
            valid_count=val_count,
        ):
            injected_positions = _inject_diversity_recovery(
                population,
                objectives,
                metrics_cache,
                feature_infos,
                dont_cares,
                rng,
                feature_probs=feature_probs,
                stage_params=stage_params,
                deployable_archive=deployable_archive,
                viability_recovery=viability_recovery,
            )
            _reevaluate_infinite_objectives(
                population,
                objectives,
                metrics_cache,
                injected_positions,
                dont_cares,
                engine,
                pareto_archive,
                val_engine=val_engine,
                stage_params=stage_params,
                run_val=run_val_this_gen,
                cv_fold_evaluator=cv_fold_evaluator,
            )
            mutation_rate = _stage_mutation_rate(
                stage_params, diversity_recovery=True,
            )
        elif post_restart_gens_remaining > 0:
            mutation_rate = float(getattr(
                _cfg,
                "PHASE2_PLATEAU_POST_RESTART_MUTATION_BOOST",
                0.45,
            ))
            post_restart_gens_remaining -= 1
        else:
            mutation_rate = _stage_mutation_rate(stage_params)

        offspring = _make_offspring_population(
            population, objectives, pop_size, feature_infos, dont_cares, rng,
            feature_probs=feature_probs,
            fronts=fronts,
            mutation_rate=mutation_rate,
            weighted_activate_prob=weighted_activate_prob,
        )
        off_obj = np.full((pop_size, N_OBJ), np.inf)
        off_metrics: list[dict] = [{} for _ in range(pop_size)]
        # Batched offspring eval: a single simulate_rule_batch over all
        # pop_size offspring (vs pop_size batch=1 dispatches). Reuses the same
        # helper as the initial-population eval path. NOTE: this path does not
        # compute CV fold returns; with PHASE2_F3_OBJECTIVE="profit_factor"
        # (active config) this matches the existing initial-pop behaviour.
        # cv_fold_min would need separate batched CV handling (pre-existing
        # limitation).
        _evaluate_population_indices(
            offspring,
            list(range(pop_size)),
            dont_cares,
            engine,
            pareto_archive,
            off_obj,
            off_metrics,
            val_engine=val_engine,
            stage_params=stage_params,
            run_val=run_val_this_gen,
        )

        merge_pop = np.vstack([population, offspring])
        merge_fit = np.vstack([objectives, off_obj])
        merge_metrics = metrics_cache + off_metrics

        population, objectives, sel_idx = environmental_selection_nsga2(
            merge_pop, merge_fit, pop_size,
        )
        metrics_cache = [merge_metrics[i] for i in sel_idx]

        # --- elite preservation ---
        _preserve_deployable_elites(
            population, objectives, deployable_archive, metrics_cache,
            _cfg, gen,
        )

    metrics_by_chrom = _metrics_dict_from_population(
        population, metrics_cache
    )
    harvest_archive = _harvest_archive_chromosomes(
        deployable_archive, hall_of_fame, pareto_archive,
    )
    pareto_pool = _build_pool_from_archive(
        harvest_archive,
        feature_infos,
        dont_cares,
        engine,
        metrics_by_chrom=metrics_by_chrom,
        val_engine=pool_val_engine if pool_val_engine is not None else val_engine,
        cv_fold_evaluator=cv_fold_evaluator,
        holdout_n_valid_rows=holdout_n_valid_rows,
        train_n_rows=train_n_rows,
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
    pool_val_engine=None,
    cv_fold_evaluator=None,
    holdout_n_valid_rows: int | None = None,
    train_n_rows: int | None = None,
    feature_probs: np.ndarray | None = None,
    init_strategy: str | None = None,
    stratum_fractions: tuple[float, float] | None = None,
    seed_fraction: float | None = None,
    reset_plateau: bool = False,
    stage: StageLabel = None,
    state: Phase2EvolutionState | None = None,
    build_pool: bool = True,
    return_state: bool = False,
    apply_seed_chromosomes: bool | None = None,
    island_profile: str = "global",
    island_hyperparams: _cfg.IslandHyperparams | None = None,
    refresh_objectives_on_resume: bool = False,
) -> tuple[list[dict], list[dict]] | tuple[list[dict], list[dict], Phase2EvolutionState]:
    """NSGA-III evolutionary loop for Phase 2 rule pool generation."""
    from gpu_fuzzy_trader.phases.phase2_rule_pool import (
        _build_pool_from_archive,
        _get_dont_cares,
        _init_population,
        _metrics_dict_from_population,
        _pareto_sortino_stats,
    )
    from gpu_fuzzy_trader.phases.phase2_support import passes_pool_trade_floor

    stage_params = resolve_phase2_stage_params(stage)
    K = len(feature_infos)
    dont_cares = _get_dont_cares(feature_infos)
    if seed_fraction is None:
        seed_fraction = stage_params.seed_fraction
    tag = log_tag or "NSGA-III"
    if state is None:
        population = _init_population(
            pop_size,
            feature_infos,
            rng,
            seeded_chromosomes=seed_chromosomes,
            seed_fraction=seed_fraction,
            init_strategy=init_strategy,
            stratum_fractions=stratum_fractions,
            feature_probs=feature_probs,
        )
        objectives = np.full((pop_size, N_OBJ), np.inf)
        metrics_cache: list[dict] = [{} for _ in range(pop_size)]
        pareto_archive: list[np.ndarray] = []
        hall_of_fame: dict[tuple[int, ...], np.ndarray] = {}
        deployable_archive: dict[tuple[int, ...], dict] = {}
        global_metrics_cache: dict[tuple[int, ...], dict] = {}
        history: list[dict] = []
        ref_vec = _get_reference_vectors(pop_size, N_OBJ, rng)
        plateau_best_progress = -np.inf
        plateau_streak = 0
        mutation_rate = _stage_mutation_rate(stage_params)
        weighted_activate_prob = _stage_weighted_activate_prob(stage_params)
        generation_offset = 0
        restart_count = 0
        viability_collapse_streak = 0
        max_restarts = int(getattr(
            _cfg, "PHASE2_PLATEAU_MAX_RESTARTS", 1,
        ))
        post_restart_gens_remaining: int = 0
        post_restart_no_improve_streak: int = 0
        post_restart_best_progress: float = -np.inf
    else:
        population = state.population
        objectives = state.objectives
        metrics_cache = state.metrics_cache
        pareto_archive = list(state.pareto_archive)
        hall_of_fame = dict(state.hall_of_fame)
        deployable_archive = dict(state.deployable_archive)
        global_metrics_cache = dict(state.global_metrics_cache)
        history = list(state.history)
        ref_vec = state.ref_vec
        if ref_vec is None:
            ref_vec = _get_reference_vectors(pop_size, N_OBJ, rng)
        plateau_best_progress = float(state.plateau_best_progress)
        plateau_streak = int(state.plateau_streak)
        mutation_rate = (
            state.mutation_rate
            if state.mutation_rate is not None
            else _stage_mutation_rate(stage_params)
        )
        restart_count = int(getattr(state, "restart_count", 0))
        viability_collapse_streak = 0
        max_restarts = int(getattr(
            _cfg, "PHASE2_PLATEAU_MAX_RESTARTS", 1,
        ))
        post_restart_gens_remaining = int(state.post_restart_gens_remaining)
        post_restart_no_improve_streak = int(
            getattr(state, "post_restart_no_improve_streak", 0)
        )
        post_restart_best_progress = float(
            getattr(state, "post_restart_best_progress", -np.inf)
        )
        weighted_activate_prob = (
            state.weighted_activate_prob
            if state.weighted_activate_prob is not None
            else _stage_weighted_activate_prob(stage_params)
        )
        generation_offset = int(state.generation_offset)
        if apply_seed_chromosomes is None:
            apply_seed_chromosomes = False
        if seed_chromosomes is not None and seed_chromosomes.size > 0 and apply_seed_chromosomes:
            migrant_slots = min(
                pop_size,
                max(1, int(round(pop_size * float(seed_fraction or 0.0)))),
                len(seed_chromosomes),
            )
            for i in range(migrant_slots):
                population[i] = np.asarray(seed_chromosomes[i], dtype=np.int32).copy()
                objectives[i] = np.full(N_OBJ, np.inf)
                metrics_cache[i] = {}

        # --- Task 2: Refresh objectives when resuming with potentially stale context ---
        if refresh_objectives_on_resume and state is not None:
            objectives[:] = np.full((pop_size, N_OBJ), np.inf)
            metrics_cache = [{} for _ in range(pop_size)]
            if _cfg.PHASE2_EVAL_GLOBAL_CACHE and global_metrics_cache is not None:
                global_metrics_cache.clear()
            logger.info(
                "%s: refresh_objectives_on_resume enabled — all objectives reset to inf, "
                "metrics_cache cleared", tag,
            )

    logger.info(
        "%s: %d features, pop=%d, gen=%d, mutation=%.3f%s",
        tag, K, pop_size, n_generations, stage_params.mutation_rate,
        f", offset={generation_offset}" if generation_offset > 0 else "",
    )
    gen_loop_start = time.monotonic()
    if reset_plateau:
        plateau_best_progress = -np.inf
        plateau_streak = 0
        restart_count = 0
        post_restart_gens_remaining = 0
        post_restart_no_improve_streak = 0
        post_restart_best_progress = -np.inf

    hist_before_loop = len(history)
    for gen in range(n_generations):
        just_restarted = False
        is_last_gen = gen == n_generations - 1
        run_val_this_gen = _should_run_val_this_gen(gen, is_last_gen)
        diversity_reference = _build_diversity_reference(
            hall_of_fame, pareto_archive,
        )
        diversity_metrics_by_key = global_metrics_cache
        pre_survivors = [
            i for i in range(pop_size)
            if not np.any(np.isinf(objectives[i]))
        ]
        parent_stats = _evaluate_population_indices(
            population,
            list(range(pop_size)),
            dont_cares,
            engine,
            pareto_archive,
            objectives,
            metrics_cache,
            val_engine=val_engine,
            global_metrics_cache=global_metrics_cache,
            diversity_reference=diversity_reference,
            diversity_metrics_by_key=diversity_metrics_by_key,
            stage_params=stage_params,
            run_val=run_val_this_gen,
            generation=gen,
            is_last_gen=is_last_gen,
            cv_fold_evaluator=cv_fold_evaluator,
        )
        if run_val_this_gen and val_engine is not None and pre_survivors:
            _refresh_survivor_val_metrics(
                population,
                pre_survivors,
                metrics_cache,
                objectives,
                dont_cares,
                pareto_archive,
                val_engine,
                diversity_reference=diversity_reference,
                diversity_metrics_by_key=diversity_metrics_by_key,
                stage_params=stage_params,
                global_metrics_cache=global_metrics_cache,
                generation=gen,
                is_last_gen=is_last_gen,
            )

        # Compute fronts once — reused for logging, offspring, and archive.
        fronts = non_dominated_sort(objectives)
        pareto_indices = fronts[0]
        pareto_archive = [population[i].copy() for i in pareto_indices]
        _update_hall_of_fame(hall_of_fame, population, pareto_indices)
        if run_val_this_gen:
            _update_deployable_archive(
                deployable_archive,
                population,
                list(range(pop_size)),
                metrics_cache,
            )

        pareto_obj = objectives[pareto_indices]
        pareto_diag = _pareto_diagnostics(
            pareto_indices, metrics_cache, pareto_obj, population,
        )
        deployable_count = _count_deployable_preview(
            pareto_indices,
            metrics_cache,
        )
        unique_ratio = float(pareto_diag.get("unique_chromosome_ratio", 0.0))
        pop_unique_ratio = _population_unique_chromosome_ratio(population)
        pop_diversity_ratio = _population_genotype_diversity_ratio(population)
        plateau_metric = _plateau_progress_metric(
            pareto_indices, metrics_cache)
        robust_stats = _pareto_robust_stats(pareto_indices, metrics_cache)
        gap_stats = _pareto_train_val_gap_stats(pareto_indices, metrics_cache)
        rank_diag = _population_rank_diagnostics(fronts, pop_size)
        holdout_stats: dict[str, float | None] = {
            "max_holdout_return_pct": None,
            "max_holdout_sortino": None,
        }
        if is_last_gen and pool_val_engine is not None:
            holdout_stats = _pareto_holdout_stats(
                pareto_indices,
                population,
                pool_val_engine,
                generation=gen,
                is_last_gen=is_last_gen,
            )
        corr_threshold = float(
            getattr(_cfg, "PHASE2_OBJECTIVE_CORR_WARN_THRESHOLD", 0.9),
        )
        for corr_key in (
            "objective_corr_f1_f2",
            "objective_corr_f1_f3",
            "objective_corr_f2_f3",
        ):
            corr_val = float(pareto_diag.get(corr_key, 0.0))
            if abs(corr_val) >= corr_threshold:
                logger.warning(
                    "Phase 2 [%s] gen %d: %s=%.2f (Pareto collapse risk)",
                    tag, gen + 1, corr_key, corr_val,
                )
        history.append({
            "generation": generation_offset + gen,
            "pareto_size": len(pareto_indices),
            "mean_f1": float(np.mean(pareto_obj[:, 0])) if len(pareto_obj) else 0.0,
            "mean_f2": float(np.mean(pareto_obj[:, 1])) if len(pareto_obj) else 0.0,
            "mean_f3": float(np.mean(pareto_obj[:, 2])) if len(pareto_obj) else 0.0,
            "algorithm": "NSGA-III",
            "deployable_preview_count": deployable_count,
            "deployable_archive_size": len(deployable_archive),
            "plateau_progress_pct": plateau_metric,
            "pop_genotype_diversity_ratio": pop_diversity_ratio,
            **_pareto_sortino_stats(pareto_indices, metrics_cache),
            **robust_stats,
            **gap_stats,
            **rank_diag,
            **holdout_stats,
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
            )
        )
        pop_viable_count = _count_pop_viable(
            pop_size,
            metrics_cache,
        )

        # --- viability-collapse trigger: forced restart when viable pop is
        #     persistently below threshold ---
        pop_size_for_viability = max(pop_size, 1)
        viability_threshold = int(
            getattr(_cfg, "PHASE2_VIABILITY_COLLAPSE_THRESHOLD", 0.5)
            * pop_size_for_viability
        )
        if pop_viable_count < viability_threshold:
            viability_collapse_streak += 1
        else:
            viability_collapse_streak = 0
        viability_collapse_patience = int(
            getattr(_cfg, "PHASE2_VIABILITY_COLLAPSE_STREAK", 3)
        )
        if (
            viability_collapse_streak >= viability_collapse_patience
            and restart_count < max_restarts
        ):
            n_elite_kept = _plateau_diversity_restart(
                population, objectives, metrics_cache,
                feature_infos, rng,
                pareto_indices=pareto_indices,
                pop_size=pop_size,
                init_strategy=init_strategy,
                stratum_fractions=stratum_fractions,
            )
            restart_count += 1
            pre_reset_streak = viability_collapse_streak
            viability_collapse_streak = 0
            post_restart_gens_remaining = int(getattr(
                _cfg,
                "PHASE2_PLATEAU_POST_RESTART_BOOST_GENS",
                3,
            ))
            post_restart_best_progress = -np.inf
            post_restart_no_improve_streak = 0
            just_restarted = True
            logger.info(
                "Phase 2 [%s]: viability-collapse restart at gen %d "
                "(pop_viable=%d < %d, streak=%d)",
                tag, gen + 1, pop_viable_count, viability_threshold,
                pre_reset_streak,
            )

        plateau_best_progress, plateau_streak = _update_max_return_plateau(
            plateau_metric, plateau_best_progress, plateau_streak,
        )
        if restart_count > 0 and not just_restarted:
            post_restart_best_progress, post_restart_no_improve_streak = (
                _update_max_return_plateau(
                    plateau_metric,
                    post_restart_best_progress,
                    post_restart_no_improve_streak,
                )
            )

        # Periodically reclaim stale Python objects (metric dicts, offspring
        # arrays) to reduce peak RSS on memory-constrained hosts (Colab 12 GiB).
        if gen % 3 == 0 and gen > 0:
            import gc as _gc
            _gc.collect()
            # Return freed glibc arena memory to the OS. Colab (Linux/glibc)
            # is the primary host; the (OSError, AttributeError) guard makes
            # this a no-op on non-glibc systems (macOS, musl), when ctypes
            # is missing, or when the loaded libc doesn't export malloc_trim.
            try:
                import ctypes
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except (OSError, AttributeError):
                pass

        is_last_gen = gen == n_generations - 1
        off_stats = _empty_eval_stats()
        inject_stats = _empty_eval_stats()
        if not is_last_gen:
            injected_positions: list[int] = []
            viability_recovery = _should_viability_recovery(
                stage_params,
                valid_count=val_count,
                plateau_streak=plateau_streak,
            )
            if _should_inject_diversity_recovery(
                pop_unique_ratio,
                stage_params=stage_params,
                pareto_size=len(pareto_indices),
                plateau_streak=plateau_streak,
                pop_size=pop_size,
                valid_count=val_count,
            ):
                injected_positions = _inject_diversity_recovery(
                    population,
                    objectives,
                    metrics_cache,
                    feature_infos,
                    dont_cares,
                    rng,
                    feature_probs=feature_probs,
                    stage_params=stage_params,
                    deployable_archive=deployable_archive,
                    viability_recovery=viability_recovery,
                )
                inject_stats = _reevaluate_infinite_objectives(
                    population,
                    objectives,
                    metrics_cache,
                    injected_positions,
                    dont_cares,
                    engine,
                    pareto_archive,
                    val_engine=val_engine,
                    global_metrics_cache=global_metrics_cache,
                    diversity_reference=diversity_reference,
                    diversity_metrics_by_key=diversity_metrics_by_key,
                    stage_params=stage_params,
                    run_val=run_val_this_gen,
                    cv_fold_evaluator=cv_fold_evaluator,
                )
                mutation_rate = _stage_mutation_rate(
                    stage_params, diversity_recovery=True,
                )
            elif post_restart_gens_remaining > 0:
                mutation_rate = float(getattr(
                    _cfg,
                    "PHASE2_PLATEAU_POST_RESTART_MUTATION_BOOST",
                    0.45,
                ))
                post_restart_gens_remaining -= 1
            else:
                mutation_rate = _stage_mutation_rate(stage_params)

            offspring = _make_offspring_population(
                population, objectives, pop_size, feature_infos, dont_cares, rng,
                feature_probs=feature_probs,
                fronts=fronts,
                mutation_rate=mutation_rate,
                weighted_activate_prob=weighted_activate_prob,
            )
            off_obj = np.full((pop_size, N_OBJ), np.inf)
            off_metrics: list[dict] = [{} for _ in range(pop_size)]
            # Batched offspring eval: a single simulate_rule_batch over all
            # pop_size offspring (vs pop_size batch=1 dispatches). Reuses the
            # same helper as the initial-population eval path. NOTE: this path
            # does not compute CV fold returns; with
            # PHASE2_F3_OBJECTIVE="profit_factor" (active config) this matches
            # the existing initial-pop behaviour. cv_fold_min would need
            # separate batched CV handling (pre-existing limitation).
            off_stats = _evaluate_population_indices(
                offspring,
                list(range(pop_size)),
                dont_cares,
                engine,
                pareto_archive,
                off_obj,
                off_metrics,
                val_engine=val_engine,
                global_metrics_cache=global_metrics_cache,
                diversity_reference=diversity_reference,
                diversity_metrics_by_key=diversity_metrics_by_key,
                stage_params=stage_params,
                run_val=run_val_this_gen,
                cv_fold_evaluator=cv_fold_evaluator,
            )

        gen_eval_stats = _merge_gen_eval_stats(
            parent_stats, inject_stats, off_stats,
        )
        history[-1]["eval_cache_hit_rate"] = _eval_cache_hit_rate(
            gen_eval_stats)

        maybe_log_generation(
            logger, tag, gen, n_generations, len(pareto_indices), mean_ret,
            max_return_pct=max_ret,
            median_return_pct=median_ret,
            max_sortino=max_sort,
            mean_robust_return_pct=robust_stats["mean_robust_return_pct"],
            max_robust_return_pct=robust_stats["max_robust_return_pct"],
            max_robust_sortino=robust_stats["max_robust_sortino"],
            max_train_val_gap_pct=gap_stats["max_train_val_gap_pct"],
            max_train_val_gap_ratio=gap_stats["max_train_val_gap_ratio"],
            max_holdout_return_pct=holdout_stats.get("max_holdout_return_pct"),
            max_holdout_sortino=holdout_stats.get("max_holdout_sortino"),
            objective_corr_f1_f3=float(
                pareto_diag.get("objective_corr_f1_f3", 0.0)),
            rank0_fraction=float(rank_diag.get("rank0_fraction", 0.0)),
            valid_count=val_count,
            unique_chromosome_ratio=float(
                pareto_diag.get("unique_chromosome_ratio", 0.0)),
            pop_unique_chromosome_ratio=pop_diversity_ratio,
            cache_hit_rate=_eval_cache_hit_rate(gen_eval_stats),
            deployable_count=deployable_count,
            pop_viable_count=pop_viable_count,
            plateau_streak=plateau_streak,
            restarts="%d/%d" % (restart_count, max_restarts),
            loop_start=gen_loop_start,
        )

        if val_count < 10:
            from gpu_fuzzy_trader.phases.phase2_support import (
                _feasibility_gate_failures,
            )
            breakdown = {k: 0 for k in [
                "train_trade_floor", "train_return_floor", "train_pf_floor",
                "val_required", "val_ret_positive", "val_trade_floor",
                "val_return_floor", "val_pf_floor", "train_val_gap",
                "f4_concentration",
            ]}
            for i in range(len(metrics_cache)):
                train_m = metrics_cache[i]
                val_m = _val_metrics_from_cache(train_m)
                gate_fails = _feasibility_gate_failures(train_m, val_m)
                for k in breakdown:
                    breakdown[k] += gate_fails[k]
            logger.warning(
                "Phase 2 [%s] gen %d/%d: feasibility collapse breakdown "
                "(valid_rules=%d < 10): %s",
                tag, gen + 1, n_generations, val_count, breakdown,
            )

        pareto_ret = _pareto_return_pct_for_early_stop(
            pareto_indices, metrics_cache,
        )
        if _should_early_stop_phase2(
            gen, pareto_ret, val_count, stage_params=stage_params,
            island_profile=island_profile,
        ):
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

        if _should_plateau_early_stop_phase2(
            gen,
            plateau_streak,
            deployable_count=deployable_count,
            unique_chromosome_ratio=unique_ratio,
            population_unique_ratio=pop_diversity_ratio,
            stage_params=stage_params,
            island_profile=island_profile,
        ):
            # --- diversity restart on first plateau, break on second ---
            restart_enabled = bool(getattr(
                _cfg, "PHASE2_PLATEAU_DIVERSITY_RESTART_ENABLED", True,
            ))
            max_restarts = int(getattr(
                _cfg, "PHASE2_PLATEAU_MAX_RESTARTS", 1,
            ))
            if (
                restart_enabled
                and max_restarts > 0
                and restart_count < max_restarts
            ):
                # If restart would exceed remaining generations, just break.
                remaining_gens = n_generations - gen - 1
                if remaining_gens > 0:
                    elite_kept = _plateau_diversity_restart(
                        population, objectives, metrics_cache,
                        feature_infos, rng,
                        pareto_indices=pareto_indices,
                        pop_size=pop_size,
                        init_strategy=init_strategy,
                        stratum_fractions=stratum_fractions,
                    )
                    # Set multi-generation mutation boost instead of one-shot
                    post_restart_gens_remaining = int(getattr(
                        _cfg,
                        "PHASE2_PLATEAU_POST_RESTART_BOOST_GENS",
                        3,
                    ))
                    restart_count += 1
                    plateau_streak = 0
                    post_restart_best_progress = -np.inf
                    post_restart_no_improve_streak = 0
                    just_restarted = True
                    logger.info(
                        "Phase 2 [%s]: plateau restart at gen %d "
                        "(restart %d/%d, reinit %.0f%%, elite_kept=%d, mutation=%.2f)",
                        tag,
                        gen + 1,
                        restart_count,
                        max_restarts,
                        float(_cfg.PHASE2_PLATEAU_DIVERSITY_RESTART_FRACTION) * 100,
                        elite_kept,
                        mutation_rate,
                    )
                    continue  # skip env selection; go to next gen
                # else fall through to break
            patience_used = _resolve_plateau_patience(stage_params, island_profile)
            logger.info(
                "%s: plateau early stop at gen %d (progress=%.2f%% unchanged "
                "for %d gens, patience=%d, min_delta=%.2f%%, "
                "deployable_preview=%d, pop_unique=%.2f)",
                tag,
                gen + 1,
                plateau_metric,
                plateau_streak,
                patience_used,
                float(_cfg.PHASE2_PLATEAU_EARLY_STOP_MIN_DELTA_PCT),
                deployable_count,
                pop_diversity_ratio,
            )
            break

        if (
            restart_count > 0
            and _should_post_restart_early_stop_phase2(
                post_restart_no_improve_streak,
                stage_params=stage_params,
                island_profile=island_profile,
            )
        ):
            logger.info(
                "%s: post-restart early stop at gen %d "
                "(no improvement for %d gens after restart %d/%d, "
                "best_progress=%.2f%%, deployable_preview=%d)",
                tag,
                gen + 1,
                post_restart_no_improve_streak,
                restart_count,
                max_restarts,
                plateau_best_progress,
                deployable_count,
            )
            break

        if is_last_gen:
            break

        merge_pop = np.vstack([population, offspring])
        merge_fit = np.vstack([objectives, off_obj])
        merge_metrics = metrics_cache + off_metrics

        population, objectives, sel_idx = _nsga3_environmental_selection(
            merge_pop, merge_fit, ref_vec, pop_size, feature_infos, dont_cares,
        )

        n_alive = len(population)
        metrics_cache = [merge_metrics[int(i)] for i in sel_idx[:n_alive]]

        # --- elite preservation ---
        _preserve_deployable_elites(
            population, objectives, deployable_archive, metrics_cache,
            _cfg, gen,
        )

    final_state = Phase2EvolutionState(
        population=population,
        objectives=objectives,
        metrics_cache=metrics_cache,
        pareto_archive=pareto_archive,
        hall_of_fame=hall_of_fame,
        deployable_archive=deployable_archive,
        global_metrics_cache=global_metrics_cache,
        history=history,
        generation_offset=generation_offset + (len(history) - hist_before_loop),
        plateau_best_progress=plateau_best_progress,
        plateau_streak=plateau_streak,
        post_restart_no_improve_streak=post_restart_no_improve_streak,
        post_restart_best_progress=post_restart_best_progress,
        post_restart_gens_remaining=post_restart_gens_remaining,
        ref_vec=ref_vec,
        mutation_rate=mutation_rate,
        weighted_activate_prob=weighted_activate_prob,
        stage=stage,
        restart_count=restart_count,
    )
    if not build_pool:
        if return_state:
            return [], history, final_state
        return [], history

    metrics_by_chrom = _metrics_dict_from_population(population, metrics_cache)
    harvest_archive = _harvest_archive_chromosomes(
        deployable_archive, hall_of_fame, pareto_archive,
    )
    pareto_pool = _build_pool_from_archive(
        harvest_archive,
        feature_infos,
        dont_cares,
        engine,
        metrics_by_chrom=metrics_by_chrom,
        val_engine=pool_val_engine if pool_val_engine is not None else val_engine,
        cv_fold_evaluator=cv_fold_evaluator,
        holdout_n_valid_rows=holdout_n_valid_rows,
        train_n_rows=train_n_rows,
        direction=log_tag or "",
    )
    if return_state:
        return pareto_pool, history, final_state
    return pareto_pool, history


def _preserve_deployable_elites(
    population: np.ndarray,
    objectives: np.ndarray,
    deployable_archive: dict[tuple[int, ...], dict],
    metrics_cache: list[dict],
    cfg,
    current_gen: int,
) -> None:
    """Force-preserve top-K deployable-archive elites in the live population.

    Guarantees a non-dominated elite at gen N survives to gen N+k unless
    a genuinely Pareto-dominating rule appears — preventing mid-epoch
    erosion from recomputed dynamic penalties (diversity/support drift).
    """
    if not cfg.PHASE2_ELITE_PRESERVATION_ENABLED:
        return
    if current_gen < cfg.PHASE2_ELITE_PRESERVATION_MIN_GEN:
        return
    if not deployable_archive:
        return

    top_k = min(cfg.PHASE2_ELITE_PRESERVATION_TOP_K, len(deployable_archive))
    if top_k == 0:
        return

    # Rank deployable_archive by rank_score desc, take top-K
    sorted_elites = sorted(
        deployable_archive.values(),
        key=lambda e: float(e.get("rank_score", 0) or 0),
        reverse=True,
    )[:top_k]

    pop_size = population.shape[0]

    # Track slots already overwritten in this call so multiple elites
    # don't compete for the same slot.
    used_slots: set[int] = set()

    for elite_entry in sorted_elites:
        chrom = elite_entry["chromosome"]
        if not isinstance(chrom, np.ndarray):
            chrom = np.array(chrom, dtype=population.dtype)

        # Check if already present (by exact chromosome match)
        already_present = False
        for i in range(pop_size):
            if np.array_equal(population[i], chrom):
                already_present = True
                break
        if already_present:
            continue

        # Evict the most-crowded survivor (highest rank, lowest crowding)
        # among slots not already used by a prior elite in this call.
        fronts = non_dominated_sort(objectives)
        ranks, crowding = _build_rank_and_crowding(objectives, fronts)
        max_rank = int(np.max(ranks))
        # Candidates: worst-rank individuals, excluding already-used slots
        max_rank_indices = [
            i for i in range(pop_size)
            if ranks[i] == max_rank and i not in used_slots
        ]
        if not max_rank_indices:
            # Fallback: any remaining slot
            max_rank_indices = [
                i for i in range(pop_size) if i not in used_slots
            ]
            if not max_rank_indices:
                break  # all slots already used
        worst_idx = int(min(max_rank_indices, key=lambda i: crowding[i]))

        used_slots.add(worst_idx)

        # Overwrite the slot
        population[worst_idx] = chrom.copy()
        objectives[worst_idx] = np.full(objectives.shape[1], np.inf)
        if metrics_cache is not None:
            metrics_cache[worst_idx] = {}


def extract_deployable_migrants(
    state: Phase2EvolutionState,
    *,
    top_k: int = 5,
) -> list[dict]:
    """Return elite deployable-preview entries suitable for guarded migration."""
    if not state.deployable_archive:
        return []
    ranked = sorted(
        state.deployable_archive.values(),
        key=lambda entry: float(entry.get("rank_score", -np.inf)),
        reverse=True,
    )[: max(1, int(top_k))]
    migrants: list[dict] = []
    for entry in ranked:
        chrom = entry.get("chromosome")
        metrics = entry.get("metrics", {})
        if chrom is None:
            continue
        migrants.append({
            "chromosome": np.asarray(chrom, dtype=np.int32).tolist()
            if not isinstance(chrom, list)
            else chrom,
            "objectives": {
                "sortino_ratio": float(metrics.get("sortino_ratio", 0.0)),
                "total_return_pct": float(metrics.get("total_return_pct", 0.0)),
                "max_drawdown_pct": float(metrics.get("max_drawdown_pct", 0.0)),
                "profit_factor": float(metrics.get("profit_factor", 1.0)),
            },
            "executed_trades": int(metrics.get("executed_trades", 0)),
            "val_objectives": {
                "total_return_pct": float(metrics.get("val_total_return_pct", 0.0)),
                "profit_factor": float(metrics.get("val_profit_factor", 1.0)),
                "max_drawdown_pct": float(metrics.get("val_max_drawdown_pct", 0.0)),
            },
            "val_executed_trades": int(metrics.get("val_executed_trades", 0)),
            "tp": float(_cfg.PHASE2_TP),
            "sl": float(_cfg.PHASE2_SL),
            "capital_pct": float(_cfg.PHASE2_CAPITAL_PCT),
            "migrant_rank_score": float(entry.get("rank_score", 0.0)),
        })
    return migrants


def run_phase2_evolution_epoch(
    feature_infos: list[dict],
    engine,
    pop_size: int,
    n_generations: int,
    rng: np.random.Generator,
    state: Phase2EvolutionState | None = None,
    seed_chromosomes: np.ndarray | None = None,
    log_tag: str | None = None,
    val_engine=None,
    feature_probs: np.ndarray | None = None,
    init_strategy: str | None = None,
    stratum_fractions: tuple[float, float] | None = None,
    seed_fraction: float | None = None,
    stage: StageLabel = None,
    apply_seed_chromosomes: bool | None = None,
    reset_plateau: bool = False,
    island_profile: str = "global",
    island_hyperparams: _cfg.IslandHyperparams | None = None,
    refresh_objectives_on_resume: bool = False,
) -> tuple[Phase2EvolutionState, list[dict]]:
    """Evolve one island epoch and return updated resumable state."""
    result = run_phase2_evolution(
        feature_infos=feature_infos,
        engine=engine,
        pop_size=pop_size,
        n_generations=n_generations,
        rng=rng,
        seed_chromosomes=seed_chromosomes,
        log_tag=log_tag,
        val_engine=val_engine,
        feature_probs=feature_probs,
        init_strategy=init_strategy,
        stratum_fractions=stratum_fractions,
        seed_fraction=seed_fraction,
        stage=stage,
        state=state,
        build_pool=False,
        return_state=True,
        apply_seed_chromosomes=apply_seed_chromosomes,
        reset_plateau=reset_plateau,
        island_profile=island_profile,
        island_hyperparams=island_hyperparams,
        refresh_objectives_on_resume=refresh_objectives_on_resume,
    )
    _, history, new_state = result
    return new_state, history


def run_phase2_evolution(
    feature_infos: list[dict],
    engine,
    pop_size: int,
    n_generations: int,
    rng: np.random.Generator,
    seed_chromosomes: np.ndarray | None = None,
    log_tag: str | None = None,
    val_engine=None,
    pool_val_engine=None,
    cv_fold_evaluator=None,
    holdout_n_valid_rows: int | None = None,
    train_n_rows: int | None = None,
    feature_probs: np.ndarray | None = None,
    init_strategy: str | None = None,
    stratum_fractions: tuple[float, float] | None = None,
    seed_fraction: float | None = None,
    reset_plateau: bool = False,
    stage: StageLabel = None,
    state: Phase2EvolutionState | None = None,
    build_pool: bool = True,
    return_state: bool = False,
    apply_seed_chromosomes: bool | None = None,
    island_profile: str = "global",
    island_hyperparams: _cfg.IslandHyperparams | None = None,
    refresh_objectives_on_resume: bool = False,
) -> tuple[list[dict], list[dict]] | tuple[list[dict], list[dict], Phase2EvolutionState]:
    """Run Phase 2 NSGA-III evolution. Returns (pareto_pool, history) or state."""
    evo_kwargs = dict(
        feature_probs=feature_probs,
        init_strategy=init_strategy,
        stratum_fractions=stratum_fractions,
        state=state,
        build_pool=build_pool,
        return_state=return_state,
        apply_seed_chromosomes=apply_seed_chromosomes,
        pool_val_engine=pool_val_engine,
        cv_fold_evaluator=cv_fold_evaluator,
        holdout_n_valid_rows=holdout_n_valid_rows,
        train_n_rows=train_n_rows,
        island_profile=island_profile,
        island_hyperparams=island_hyperparams,
        refresh_objectives_on_resume=refresh_objectives_on_resume,
    )
    if not _EVOX_AVAILABLE:
        logger.warning(
            "EvoX not available (%s); falling back to NumPy NSGA-II for Phase 2.",
            _EVOX_IMPORT_ERROR or "import failed",
        )
        fallback_tag = "NSGA-II (fallback)"
        if log_tag:
            fallback_tag = log_tag.replace("NSGA-III", "NSGA-II (fallback)")
        if state is not None or return_state or not build_pool:
            return _run_nsga3(
                feature_infos, engine, pop_size, n_generations, rng,
                seed_chromosomes=seed_chromosomes,
                log_tag=fallback_tag,
                val_engine=val_engine,
                seed_fraction=seed_fraction,
                reset_plateau=reset_plateau,
                stage=stage,
                **evo_kwargs,
            )
        return _run_nsga2_fallback(
            feature_infos, engine, pop_size, n_generations, rng,
            seed_chromosomes=seed_chromosomes,
            log_tag=fallback_tag,
            val_engine=val_engine,
            seed_fraction=seed_fraction,
            reset_plateau=reset_plateau,
            stage=stage,
            **{k: v for k, v in evo_kwargs.items() if k in (
                "feature_probs", "init_strategy", "stratum_fractions",
                "pool_val_engine", "cv_fold_evaluator",
                "holdout_n_valid_rows", "train_n_rows",
                "island_profile", "island_hyperparams",
            )},
        )

    return _run_nsga3(
        feature_infos, engine, pop_size, n_generations, rng,
        seed_chromosomes=seed_chromosomes, log_tag=log_tag,
        val_engine=val_engine,
        seed_fraction=seed_fraction,
        reset_plateau=reset_plateau,
        stage=stage,
        **evo_kwargs,
    )
