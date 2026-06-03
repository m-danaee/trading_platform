"""
phase3_rule_set.py — Rule_Set_Selector (Phase 3)

Greedy rule-set construction plus short Pareto refinement over ordered
combinations of 2–5 rules from the Phase 2 pool.  Evaluated on the
**validation split** using CPUBacktestEngine.

Search space:
    All ordered combinations of PHASE3_MIN_RULES–PHASE3_MAX_RULES rules from
    the Phase 2 pool, with no duplicate rules (order-independent condition set
    equality).

Fitness function (three objectives, all minimised):
    f1 = -validation_sortino_ratio
    f2 = validation_max_drawdown_pct
    f3 = -validation_win_rate

Penalties (added to all objectives proportionally):
    coverage_penalty      — if symbols_with_trades < PHASE3_MIN_SYMBOL_COVERAGE
    zero_trade_penalty    — if total executed_trades == 0
    overfitting_penalty   — |train_return - val_return| / max(|train_return|, 1.0)
    duplicate_rule_penalty — if any two rules have identical condition sets

Output:
    outputs/long.json  and  outputs/short.json
    (exact evaluator_v3.ipynb compatible format)

Skip logic:
    If both files exist and pass schema validation → skip Phase 3.
    If only one exists → skip and proceed with available file.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from typing import Optional

import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.backtest.cpu_engine import CPUBacktestEngine
from gpu_fuzzy_trader.evolution.numba_ops import (
    crowding_distance as _crowding_distance_numba,
    non_dominated_sort as _non_dominated_sort_numba,
)
from gpu_fuzzy_trader.log_progress import maybe_log_generation
from gpu_fuzzy_trader.phases.phase3_cache import (
    Phase3EvalCache,
    build_phase3_eval_cache,
)
from gpu_fuzzy_trader.phases.phase3_greedy import greedy_rule_set_search
from gpu_fuzzy_trader.phases.phase3_objectives import (
    compute_phase3_objectives,
    conditions_key as _conditions_key,
    count_symbols_with_trades as _count_symbols_with_trades,
    has_duplicate_rules as _has_duplicate_rules,
    per_rule_min_symbol_trades_cached as _per_rule_min_symbol_trades_cached,
    symbol_consistency_penalty as _symbol_consistency_penalty,
    symbols_with_trades as _symbols_with_trades,
    train_val_corr_penalty as _train_val_corr_penalty,
    train_val_gap_penalty as _train_val_gap_penalty,
)
from gpu_fuzzy_trader.reporting.reporter import Reporter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

_OUTPUT_PATHS = {
    "long": os.path.join(_cfg.OUTPUTS_DIR, "long.json"),
    "short": os.path.join(_cfg.OUTPUTS_DIR, "short.json"),
}

# ---------------------------------------------------------------------------
# Output JSON schema validation
# ---------------------------------------------------------------------------


def _validate_rule_set_schema(data: object, path: str) -> None:
    """
    Validate the structure of a loaded rule set JSON.

    Raises ValueError if the schema is invalid.
    """
    if not isinstance(data, dict):
        raise ValueError(
            f"Rule set must be a JSON object, got {type(data).__name__}: {path}"
        )
    required_top = {"direction", "rules_set"}
    missing = required_top - set(data.keys())
    if missing:
        raise ValueError(
            f"Rule set missing top-level keys {missing}: {path}"
        )
    if data["direction"] not in ("long", "short"):
        raise ValueError(
            f"Rule set 'direction' must be 'long' or 'short': {path}"
        )
    rules_set = data["rules_set"]
    if not isinstance(rules_set, list):
        raise ValueError(
            f"Rule set 'rules_set' must be a list: {path}"
        )
    if not (2 <= len(rules_set) <= 5):
        raise ValueError(
            f"Rule set 'rules_set' must have 2–5 rules, got {len(rules_set)}: {path}"
        )
    for i, rule in enumerate(rules_set):
        if not isinstance(rule, dict):
            raise ValueError(
                f"Rule set entry {i} must be a dict: {path}"
            )
        required_rule_keys = {"tp", "sl", "capital_pct", "conditions"}
        missing_rule = required_rule_keys - set(rule.keys())
        if missing_rule:
            raise ValueError(
                f"Rule set entry {i} missing keys {missing_rule}: {path}"
            )
        if not isinstance(rule["conditions"], list) or len(rule["conditions"]) == 0:
            raise ValueError(
                f"Rule set entry {i} 'conditions' must be a non-empty list: {path}"
            )
    if "risk_optimized" in data and not isinstance(data["risk_optimized"], bool):
        raise ValueError(
            f"Rule set 'risk_optimized' must be a bool if present: {path}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _per_rule_min_symbol_trades(
    rule_set: list[dict],
    val_engine,
    cache: Phase3EvalCache | None = None,
) -> int:
    """Validation gate helper; uses cache when provided."""
    if cache is not None:
        return _per_rule_min_symbol_trades_cached(
            rule_set, cache.per_rule_min_val_trades)
    worst = float("inf")
    for rule in rule_set:
        try:
            metrics = val_engine.simulate_rule_set(
                _rule_set_to_engine_format([rule]))
        except Exception:
            continue
        from gpu_fuzzy_trader.phases.phase3_objectives import (
            min_per_symbol_trades_from_metrics,
        )
        tc = min_per_symbol_trades_from_metrics(metrics)
        if tc < worst:
            worst = tc
    return 0 if worst == float("inf") else int(worst)


def _simulate_team(
    rule_set: list[dict],
    train_engine,
    val_engine,
    cache: Phase3EvalCache | None,
) -> tuple[dict, dict]:
    """Run train+val backtests for a team, using mask cache when available."""
    if cache is not None and hasattr(train_engine, "simulate_rule_set_from_cache"):
        try:
            train_metrics = train_engine.simulate_rule_set_from_cache(
                rule_set, cache, "train")
            val_metrics = val_engine.simulate_rule_set_from_cache(
                rule_set, cache, "val")
            return train_metrics, val_metrics
        except Exception as exc:
            logger.debug("cached simulate failed, falling back: %s", exc)

    try:
        train_metrics = train_engine.simulate_rule_set(rule_set)
    except Exception as exc:
        logger.debug("train simulate_rule_set failed: %s", exc)
        train_metrics = _empty_metrics()
    try:
        val_metrics = val_engine.simulate_rule_set(rule_set)
    except Exception as exc:
        logger.debug("val simulate_rule_set failed: %s", exc)
        val_metrics = _empty_metrics()
    return train_metrics, val_metrics


def _empty_metrics() -> dict:
    return {
        "sortino_ratio": 0.0,
        "total_return_pct": 0.0,
        "max_drawdown_pct": 100.0,
        "win_rate": 0.0,
        "executed_trades": 0,
        "per_symbol_metrics": {},
    }


def _build_phase3_engines(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    direction: str,
    pool: list[dict],
) -> tuple[object, object, bool, bool, Phase3EvalCache]:
    """
    Create train/val engines, eval cache, and batch flags.

    Returns (val_engine, train_engine, use_parallel_batch, use_jax_gpu, cache).
    """
    feature_modes: dict[str, str] = {}
    use_jax = bool(_cfg.PHASE3_USE_GPU)
    use_parallel = bool(_cfg.PHASE3_USE_PARALLEL_BATCH)

    val_engine: object
    train_engine: object

    if use_jax:
        from gpu_fuzzy_trader.backtest.jax_compat import get_gpu_backtest_engine_class

        GPUBacktestEngine = get_gpu_backtest_engine_class()
        if GPUBacktestEngine is not None:
            val_engine = GPUBacktestEngine(
                val_df,
                feature_modes,
                direction,
                fee_pct=_cfg.FEE_PCT,
            )
            train_engine = GPUBacktestEngine(
                train_df,
                feature_modes,
                direction,
                fee_pct=_cfg.FEE_PCT,
            )
            logger.info(
                "Phase 3 using GPUBacktestEngine (JAX mask + batch eval)")
        else:
            logger.warning(
                "PHASE3_USE_GPU=True but JAX/GPU backtest unavailable; using CPU.")
            use_jax = False
            val_engine = CPUBacktestEngine(
                val_df,
                feature_modes,
                direction,
                fee_pct=_cfg.FEE_PCT,
            )
            train_engine = CPUBacktestEngine(
                train_df,
                feature_modes,
                direction,
                fee_pct=_cfg.FEE_PCT,
            )
    else:
        val_engine = CPUBacktestEngine(
            val_df,
            feature_modes,
            direction,
            fee_pct=_cfg.FEE_PCT,
        )
        train_engine = CPUBacktestEngine(
            train_df,
            feature_modes,
            direction,
            fee_pct=_cfg.FEE_PCT,
        )

    cache = build_phase3_eval_cache(pool, train_df, val_df, val_engine)
    return val_engine, train_engine, use_parallel, use_jax, cache


def _merge_phase3_metrics_worst(
    current: dict | None,
    new: dict,
) -> dict:
    """Conservative merge for CV fold aggregation (min return/PF/WR, max DD)."""
    if current is None:
        return dict(new)
    out = dict(current)
    out["total_return_pct"] = min(
        float(out.get("total_return_pct", 0.0)),
        float(new.get("total_return_pct", 0.0)),
    )
    out["sortino_ratio"] = min(
        float(out.get("sortino_ratio", out.get("total_return_pct", 0.0))),
        float(new.get("sortino_ratio", new.get("total_return_pct", 0.0))),
    )
    out["max_drawdown_pct"] = max(
        float(out.get("max_drawdown_pct", 0.0)),
        float(new.get("max_drawdown_pct", 0.0)),
    )
    out["win_rate"] = min(
        float(out.get("win_rate", 0.0)),
        float(new.get("win_rate", 0.0)),
    )
    out["profit_factor"] = min(
        float(out.get("profit_factor", 0.0)),
        float(new.get("profit_factor", 0.0)),
    )
    out["executed_trades"] = min(
        int(out.get("executed_trades", 0)),
        int(new.get("executed_trades", 0)),
    )
    return out


def _evaluate_rule_set(
    rule_set: list[dict],
    val_engine,
    train_engine,
    cache: Phase3EvalCache | None = None,
    *,
    cv_fold_contexts: list[tuple] | None = None,
) -> tuple[np.ndarray, dict, dict]:
    """
    Evaluate a candidate rule set and return (objectives, train_metrics, val_metrics).

    When *cv_fold_contexts* is set, objectives use the **worst** fold (maximised
    f1/f2/f3 among folds) and returned metrics are conservative merges.
    """
    if cv_fold_contexts:
        worst_obj: np.ndarray | None = None
        agg_train: dict | None = None
        agg_val: dict | None = None
        for val_eng, train_eng, _, _, fold_cache in cv_fold_contexts:
            train_m, val_m = _simulate_team(
                rule_set, train_eng, val_eng, fold_cache)
            per_rule = (
                fold_cache.per_rule_min_val_trades if fold_cache else None)
            val_masks = fold_cache.val_masks if fold_cache else None
            n_rows = fold_cache.n_rows_val if fold_cache else 0
            obj = compute_phase3_objectives(
                train_m,
                val_m,
                rule_set,
                per_rule_min_val_trades=per_rule,
                val_masks_by_key=val_masks,
                n_rows_val=n_rows,
            )
            if worst_obj is None or float(np.sum(obj)) > float(np.sum(worst_obj)):
                worst_obj = obj
            agg_train = _merge_phase3_metrics_worst(agg_train, train_m)
            agg_val = _merge_phase3_metrics_worst(agg_val, val_m)
        assert worst_obj is not None and agg_train is not None and agg_val is not None
        return worst_obj, agg_train, agg_val

    train_metrics, val_metrics = _simulate_team(
        rule_set, train_engine, val_engine, cache)
    per_rule = cache.per_rule_min_val_trades if cache is not None else None
    val_masks = cache.val_masks if cache is not None else None
    n_rows_val = cache.n_rows_val if cache is not None else 0
    objectives = compute_phase3_objectives(
        train_metrics,
        val_metrics,
        rule_set,
        per_rule_min_val_trades=per_rule,
        val_masks_by_key=val_masks,
        n_rows_val=n_rows_val,
    )
    return objectives, train_metrics, val_metrics


# ---------------------------------------------------------------------------
# NSGA-II helpers (Numba-accelerated via evolution.numba_ops when enabled)
# ---------------------------------------------------------------------------

def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Return True if solution *a* dominates *b* (all ≤, at least one <)."""
    return bool(np.all(a <= b) and np.any(a < b))


def _non_dominated_sort(objectives: np.ndarray) -> list[list[int]]:
    """NSGA-II non-dominated sorting (Numba when PHASE3_NUMBA_ENABLED)."""
    if not _cfg.PHASE3_NUMBA_ENABLED:
        from gpu_fuzzy_trader.evolution.numba_ops import _non_dominated_sort_py
        return _non_dominated_sort_py(np.asarray(objectives, dtype=np.float64))
    return _non_dominated_sort_numba(objectives)


def _crowding_distance(objectives: np.ndarray, front: list[int]) -> np.ndarray:
    """Crowding distance for a front (Numba when PHASE3_NUMBA_ENABLED)."""
    if not _cfg.PHASE3_NUMBA_ENABLED:
        from gpu_fuzzy_trader.evolution.numba_ops import _crowding_distance_py
        return _crowding_distance_py(objectives, front)
    return _crowding_distance_numba(objectives, front)


# ---------------------------------------------------------------------------
# Population: each individual is a list of rule dicts (indices into pool)
# ---------------------------------------------------------------------------

def _random_rule_set(
    pool: list[dict],
    rng: random.Random,
    min_rules: int,
    max_rules: int,
) -> list[dict]:
    """
    Sample a random rule set of 2–5 rules from the pool with no duplicates.

    Returns a list of rule dicts (subset of pool entries, with only the
    fields needed by CPUBacktestEngine: conditions, tp, sl, capital_pct).
    """
    n_rules = rng.randint(min_rules, min(max_rules, len(pool)))
    chosen = rng.sample(pool, n_rules)
    # Deduplicate by condition set (order-independent)
    seen: set[frozenset] = set()
    unique: list[dict] = []
    for rule in chosen:
        key = _conditions_key(rule["conditions"])
        if key not in seen:
            seen.add(key)
            unique.append(rule)
    # If deduplication reduced below min_rules, pad from remaining pool
    remaining = [r for r in pool if _conditions_key(
        r["conditions"]) not in seen]
    while len(unique) < min_rules and remaining:
        r = rng.choice(remaining)
        key = _conditions_key(r["conditions"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
            remaining = [x for x in remaining if _conditions_key(
                x["conditions"]) != key]
    return unique


def _rule_set_to_engine_format(rule_set: list[dict]) -> list[dict]:
    """
    Convert pool entries to the format expected by CPUBacktestEngine.

    Pool entries have: conditions, tp, sl, capital_pct (and possibly chromosome, objectives, etc.)
    Engine expects: conditions, tp, sl, capital_pct
    """
    return [
        {
            "conditions": rule["conditions"],
            "tp": float(rule.get("tp", _cfg.PHASE2_TP)),
            "sl": float(rule.get("sl", _cfg.PHASE2_SL)),
            "capital_pct": float(rule.get("capital_pct", _cfg.PHASE2_CAPITAL_PCT)),
        }
        for rule in rule_set
    ]


def _crossover_rule_sets(
    parent_a: list[dict],
    parent_b: list[dict],
    pool: list[dict],
    rng: random.Random,
    min_rules: int,
    max_rules: int,
) -> list[dict]:
    """
    Produce one child by combining rules from two parents.

    Strategy: take a random subset of rules from both parents, deduplicate,
    then trim or pad to a valid size.
    """
    combined = list(parent_a) + list(parent_b)
    rng.shuffle(combined)

    seen: set[frozenset] = set()
    unique: list[dict] = []
    for rule in combined:
        key = _conditions_key(rule["conditions"])
        if key not in seen:
            seen.add(key)
            unique.append(rule)

    # Trim to max_rules
    if len(unique) > max_rules:
        unique = unique[:max_rules]

    # Pad to min_rules if needed
    remaining = [r for r in pool if _conditions_key(
        r["conditions"]) not in seen]
    while len(unique) < min_rules and remaining:
        r = rng.choice(remaining)
        key = _conditions_key(r["conditions"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
            remaining = [x for x in remaining if _conditions_key(
                x["conditions"]) != key]

    return unique


def _mutate_rule_set(
    rule_set: list[dict],
    pool: list[dict],
    rng: random.Random,
    min_rules: int,
    max_rules: int,
    mutation_rate: float = 0.3,
) -> list[dict]:
    """
    Mutate a rule set by randomly replacing, adding, or removing rules.
    """
    child = list(rule_set)
    seen = {_conditions_key(r["conditions"]) for r in child}
    remaining = [r for r in pool if _conditions_key(
        r["conditions"]) not in seen]

    if rng.random() < mutation_rate:
        # Replace a random rule
        if child and remaining:
            idx = rng.randrange(len(child))
            new_rule = rng.choice(remaining)
            seen.discard(_conditions_key(child[idx]["conditions"]))
            child[idx] = new_rule
            seen.add(_conditions_key(new_rule["conditions"]))
            remaining = [r for r in pool if _conditions_key(
                r["conditions"]) not in seen]

    if rng.random() < mutation_rate and len(child) < max_rules and remaining:
        # Add a rule
        new_rule = rng.choice(remaining)
        child.append(new_rule)
        seen.add(_conditions_key(new_rule["conditions"]))

    if rng.random() < mutation_rate and len(child) > min_rules:
        # Remove a rule
        idx = rng.randrange(len(child))
        child.pop(idx)

    # Ensure valid size
    while len(child) < min_rules and remaining:
        r = rng.choice(remaining)
        key = _conditions_key(r["conditions"])
        if key not in seen:
            seen.add(key)
            child.append(r)
            remaining = [x for x in remaining if _conditions_key(
                x["conditions"]) != key]

    if len(child) > max_rules:
        child = child[:max_rules]

    return child


# ---------------------------------------------------------------------------
# NSGA-II combinatorial search
# ---------------------------------------------------------------------------

def _seed_population_from_greedy(
    greedy_set: list[dict],
    pool: list[dict],
    pop_size: int,
    min_rules: int,
    max_rules: int,
    rng: random.Random,
) -> list[list[dict]]:
    """Build initial population centred on the greedy solution."""
    population: list[list[dict]] = [list(greedy_set)]
    while len(population) < pop_size:
        child = _mutate_rule_set(
            list(greedy_set), pool, rng, min_rules, max_rules)
        population.append(child)
    return population


def _simulate_teams_batch(
    rule_sets: list[list[dict]],
    train_engine,
    val_engine,
    cache: Phase3EvalCache | None,
    use_jax: bool,
) -> tuple[list[dict], list[dict]]:
    """Batch train+val simulation for multiple teams."""
    if use_jax and hasattr(val_engine, "simulate_rule_set_batch_jax"):
        val_list = val_engine.simulate_rule_set_batch_jax(
            rule_sets, cache=cache, split="val")
        train_list = train_engine.simulate_rule_set_batch_jax(
            rule_sets, cache=cache, split="train")
        return train_list, val_list

    batch_kw = {"cache": cache}
    if hasattr(val_engine, "simulate_rule_set_batch"):
        val_list = val_engine.simulate_rule_set_batch(
            rule_sets, split="val", **batch_kw)
        train_list = train_engine.simulate_rule_set_batch(
            rule_sets, split="train", **batch_kw)
        return train_list, val_list

    train_list = []
    val_list = []
    for rs in rule_sets:
        tr, va = _simulate_team(rs, train_engine, val_engine, cache)
        train_list.append(tr)
        val_list.append(va)
    return train_list, val_list


def _run_nsga2_combinatorial(
    pool: list[dict],
    val_engine,
    train_engine,
    pop_size: int,
    n_generations: int,
    min_rules: int,
    max_rules: int,
    seed: int = 42,
    initial_population: list[list[dict]] | None = None,
    use_batch: bool = False,
    use_jax: bool = False,
    cache: Phase3EvalCache | None = None,
    log_tag: str | None = None,
    cv_fold_contexts: list[tuple] | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Run NSGA-II combinatorial search over rule set combinations.

    Parameters
    ----------
    pool : list[dict]
        Phase 2 pool entries (each has "conditions", "tp", "sl", "capital_pct").
    val_engine : CPUBacktestEngine
        Engine initialised on the validation split.
    train_engine : CPUBacktestEngine
        Engine initialised on the training split.
    pop_size : int
    n_generations : int
    min_rules : int
    max_rules : int
    seed : int

    Returns
    -------
    pareto_front_rule_sets : list[list[dict]]
        Rule sets on the Pareto front (each is a list of rule dicts).
    history : list[dict]
        Per-generation metrics.
    """
    rng = random.Random(seed)

    if len(pool) < min_rules:
        raise ValueError(
            f"Pool has only {len(pool)} rules, need at least {min_rules}."
        )

    # Clamp pop_size to a reasonable value given pool size
    effective_pop = min(pop_size, max(4, len(pool) * 2))

    # Initialise population
    if initial_population:
        population = [
            list(rs) for rs in initial_population[:effective_pop]
        ]
        while len(population) < effective_pop:
            population.append(
                _random_rule_set(pool, rng, min_rules, max_rules)
            )
    else:
        population = [
            _random_rule_set(pool, rng, min_rules, max_rules)
            for _ in range(effective_pop)
        ]

    objectives = np.full((effective_pop, 3), np.inf)
    history: list[dict] = []

    tag = log_tag or "Phase 3 NSGA-II"
    use_batch_eff = bool(use_batch) and not cv_fold_contexts
    logger.info(
        "%s: pool=%d, pop=%d, gen=%d",
        tag, len(pool), effective_pop, n_generations,
    )
    gen_loop_start = time.monotonic()

    for gen in range(n_generations):
        # Evaluate unevaluated individuals
        pending = [i for i in range(effective_pop)
                   if np.any(np.isinf(objectives[i]))]
        if pending and use_batch_eff:
            fmts = [_rule_set_to_engine_format(population[i]) for i in pending]
            train_list, val_list = _simulate_teams_batch(
                fmts, train_engine, val_engine, cache, use_jax)
            per_rule = cache.per_rule_min_val_trades if cache else None
            val_masks = cache.val_masks if cache else None
            n_rows_val = cache.n_rows_val if cache else 0
            for j, i in enumerate(pending):
                objectives[i] = compute_phase3_objectives(
                    train_list[j], val_list[j], population[i],
                    per_rule_min_val_trades=per_rule,
                    val_masks_by_key=val_masks,
                    n_rows_val=n_rows_val,
                )
        else:
            for i in pending:
                engine_fmt = _rule_set_to_engine_format(population[i])
                obj, _, _ = _evaluate_rule_set(
                    engine_fmt,
                    val_engine,
                    train_engine,
                    cache=cache,
                    cv_fold_contexts=cv_fold_contexts,
                )
                objectives[i] = obj

        # Non-dominated sort
        fronts = _non_dominated_sort(objectives)
        pareto_indices = fronts[0]

        # Record history
        pareto_obj = objectives[pareto_indices]
        history.append({
            "generation": gen,
            "pareto_size": len(pareto_indices),
            "mean_f1": float(np.mean(pareto_obj[:, 0])),
            "mean_f2": float(np.mean(pareto_obj[:, 1])),
            "mean_f3": float(np.mean(pareto_obj[:, 2])),
        })

        mean_f1 = float(np.mean(pareto_obj[:, 0])) if len(pareto_obj) else 0.0
        maybe_log_generation(
            logger, tag, gen, n_generations, len(pareto_indices), mean_f1,
            loop_start=gen_loop_start,
        )

        if gen == n_generations - 1:
            break

        # Build next generation
        new_population: list[list[dict]] = []
        new_objectives = np.full((effective_pop, 3), np.inf)

        # Elitism: keep Pareto front (up to half population)
        elite_count = min(len(pareto_indices), effective_pop // 2)
        cd = _crowding_distance(objectives, pareto_indices)
        cd_order = np.argsort(-cd)
        elite_indices = [pareto_indices[j] for j in cd_order[:elite_count]]

        for j, idx in enumerate(elite_indices):
            new_population.append(list(population[idx]))
            new_objectives[j] = objectives[idx].copy()

        # Fill rest with offspring
        all_indices = list(range(effective_pop))
        while len(new_population) < effective_pop:
            # Tournament selection (size 2)
            cands_a = rng.sample(all_indices, min(2, len(all_indices)))
            cands_b = rng.sample(all_indices, min(2, len(all_indices)))
            pa = cands_a[0] if objectives[cands_a[0],
                                          0] <= objectives[cands_a[-1], 0] else cands_a[-1]
            pb = cands_b[0] if objectives[cands_b[0],
                                          0] <= objectives[cands_b[-1], 0] else cands_b[-1]

            child = _crossover_rule_sets(
                population[pa], population[pb], pool, rng, min_rules, max_rules
            )
            child = _mutate_rule_set(child, pool, rng, min_rules, max_rules)
            new_population.append(child)

        population = new_population[:effective_pop]
        objectives = new_objectives

    # Return Pareto front rule sets
    fronts = _non_dominated_sort(objectives)
    pareto_indices = fronts[0]
    pareto_rule_sets = [population[i] for i in pareto_indices]

    return pareto_rule_sets, history


def _max_team_jaccard(
    rule_set: list[dict],
    val_masks_by_key: dict[frozenset, np.ndarray] | None,
) -> float:
    """Maximum pairwise Jaccard similarity on validation entry masks."""
    if not val_masks_by_key or len(rule_set) <= 1:
        return 0.0
    masks: list[np.ndarray] = []
    for rule in rule_set:
        key = _conditions_key(rule.get("conditions", []))
        mask = val_masks_by_key.get(key)
        if mask is not None:
            masks.append(mask)
    if len(masks) <= 1:
        return 0.0
    max_j = 0.0
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            union = int(np.count_nonzero(masks[i] | masks[j]))
            if union <= 0:
                continue
            inter = int(np.count_nonzero(masks[i] & masks[j]))
            max_j = max(max_j, inter / union)
    return float(max_j)


def _pareto_selection_tiebreak(
    train_metrics: dict,
    val_metrics: dict,
    rule_set: list[dict],
    cache: Phase3EvalCache | None,
) -> tuple[float, float, float, float]:
    """
  Lower is better (lexicographic): gap penalty, symbol inconsistency,
  max Jaccard overlap, negative symbol overlap ratio.
    """
    gap = _train_val_gap_penalty(train_metrics, val_metrics)
    sym_pen = _symbol_consistency_penalty(train_metrics, val_metrics)
    val_masks = cache.val_masks if cache is not None else None
    max_j = _max_team_jaccard(rule_set, val_masks)
    train_syms = _symbols_with_trades(train_metrics)
    val_syms = _symbols_with_trades(val_metrics)
    if train_syms or val_syms:
        overlap = len(train_syms & val_syms) / len(train_syms | val_syms)
    else:
        overlap = 0.0
    return (gap, sym_pen, max_j, -float(overlap))


def _select_best_from_pareto(
    pareto_rule_sets: list[list[dict]],
    val_engine: CPUBacktestEngine,
    train_engine: CPUBacktestEngine,
    cache: Phase3EvalCache | None = None,
    cv_fold_contexts: list[tuple] | None = None,
) -> list[dict]:
    """
    Select the best rule set from the Pareto front.

    Default: maximin(train_return, val_return) with profitability floors.
    Tie-break: lower train/val gap, symbol consistency, Jaccard overlap.
    """
    if not pareto_rule_sets:
        raise ValueError(
            "Pareto front is empty — cannot select best rule set.")

    best_idx = 0
    best_score = -np.inf
    best_tiebreak: tuple[float, ...] | None = None
    best_f1 = np.inf

    for i, rs in enumerate(pareto_rule_sets):
        engine_fmt = _rule_set_to_engine_format(rs)
        obj, train_metrics, val_metrics = _evaluate_rule_set(
            engine_fmt,
            val_engine,
            train_engine,
            cache=cache,
            cv_fold_contexts=cv_fold_contexts,
        )
        score = _maximin_selection_score(train_metrics, val_metrics)
        f1 = float(obj[0])
        tiebreak = _pareto_selection_tiebreak(
            train_metrics, val_metrics, rs, cache)
        better = score > best_score
        if score == best_score and best_tiebreak is not None:
            better = tiebreak < best_tiebreak
        if score == best_score and tiebreak == best_tiebreak and f1 < best_f1:
            better = True
        if better:
            best_score = score
            best_tiebreak = tiebreak
            best_f1 = f1
            best_idx = i

    return _cap_capital_per_rule(pareto_rule_sets[best_idx])


# ---------------------------------------------------------------------------
# Output serialisation
# ---------------------------------------------------------------------------

def _cap_capital_per_rule(rule_set: list[dict]) -> list[dict]:
    """Limit per-rule capital_pct to reduce concentration risk."""
    cap = float(_cfg.PHASE3_MAX_CAPITAL_PCT_PER_RULE)
    out: list[dict] = []
    for rule in rule_set:
        r = dict(rule)
        r["capital_pct"] = min(
            float(r.get("capital_pct", _cfg.PHASE2_CAPITAL_PCT)), cap)
        out.append(r)
    return out


def _maximin_selection_score(train_metrics: dict, val_metrics: dict) -> float:
    """Higher is better; hard-reject unprofitable splits."""
    train_ret = float(train_metrics.get("total_return_pct", 0.0))
    val_ret = float(val_metrics.get("total_return_pct", 0.0))
    train_pf = float(train_metrics.get("profit_factor", 0.0))
    val_pf = float(val_metrics.get("profit_factor", 0.0))
    if (
        train_ret < _cfg.PHASE3_TRAIN_RETURN_FLOOR_PCT
        or val_ret < _cfg.PHASE3_VAL_RETURN_FLOOR_PCT
        or train_pf < _cfg.PHASE3_TRAIN_PROFIT_FACTOR_FLOOR
        or val_pf < _cfg.PHASE3_VAL_PROFIT_FACTOR_FLOOR
    ):
        return -1e9
    return min(train_ret, val_ret)


def _build_output_dict(rule_set: list[dict], direction: str) -> dict:
    """
    Build the evaluator_v3.ipynb-compatible output dict.

    Uses Phase 2 static TP/SL/capital_pct values (Phase 4 will update them).
    """
    rule_set = _cap_capital_per_rule(rule_set)
    rules_list = []
    for rule in rule_set:
        rules_list.append({
            "tp": float(rule.get("tp", _cfg.PHASE2_TP)),
            "sl": float(rule.get("sl", _cfg.PHASE2_SL)),
            "capital_pct": float(rule.get("capital_pct", _cfg.PHASE2_CAPITAL_PCT)),
            "conditions": list(rule["conditions"]),
        })
    return {
        "direction": direction,
        "risk_optimized": False,
        "rules_set": rules_list,
    }


# ---------------------------------------------------------------------------
# Rule_Set_Selector
# ---------------------------------------------------------------------------

class Rule_Set_Selector:
    """
    Phase 3: greedy construction + Pareto refinement over rule set combinations.

    Selects the best ordered combination of 2–5 rules from the Phase 2 pool,
    evaluated on the validation split using CPUBacktestEngine.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training split DataFrame (already prepared).
    val_df : pd.DataFrame
        Validation split DataFrame (already prepared).
    pool : list[dict]
        Phase 2 pool entries.  Each entry must have at minimum:
        "conditions" (list[str]).  "tp", "sl", "capital_pct" are optional
        (defaults to Phase 2 static values if absent).
    direction : str
        "long" or "short".
    refine_pop_size : int, optional
        Override PHASE3_REFINE_POP_SIZE (useful for testing).
    refine_generations : int, optional
        Override PHASE3_REFINE_GENERATIONS (useful for testing).
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        pool: list[dict],
        direction: str,
        refine_pop_size: int | None = None,
        refine_generations: int | None = None,
        seed: int = 42,
        cv_folds: list | None = None,
    ) -> None:
        if direction not in ("long", "short"):
            raise ValueError(
                f"direction must be 'long' or 'short', got {direction!r}")
        if not pool:
            raise ValueError("pool must not be empty.")
        if len(pool) < _cfg.PHASE3_MIN_RULES:
            raise ValueError(
                f"pool must have at least {_cfg.PHASE3_MIN_RULES} rules, "
                f"got {len(pool)}."
            )

        self.direction = direction
        self.pool = pool
        self.refine_pop_size = (
            refine_pop_size if refine_pop_size is not None else _cfg.PHASE3_REFINE_POP_SIZE
        )
        self.refine_generations = (
            refine_generations
            if refine_generations is not None
            else _cfg.PHASE3_REFINE_GENERATIONS
        )
        self.seed = seed

        self._cv_fold_contexts: list[tuple] | None = None
        use_cv = (
            cv_folds
            and len(cv_folds) > 0
            and str(_cfg.SPLIT_MODE).strip().lower() == "purged_rolling_cv"
        )
        if use_cv:
            self._cv_fold_contexts = [
                _build_phase3_engines(f.train_df, f.val_df, direction, pool)
                for f in cv_folds
            ]
            (
                self._val_engine,
                self._train_engine,
                self._use_parallel_batch,
                self._use_jax_gpu,
                self._eval_cache,
            ) = self._cv_fold_contexts[-1]
            logger.info(
                "Phase 3 [%s]: purged CV evaluation (%d folds)",
                direction,
                len(self._cv_fold_contexts),
            )
        else:
            (
                self._val_engine,
                self._train_engine,
                self._use_parallel_batch,
                self._use_jax_gpu,
                self._eval_cache,
            ) = _build_phase3_engines(train_df, val_df, direction, pool)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """
        Run greedy construction and Pareto refinement.

        Returns the best rule set dict (evaluator_v3.ipynb compatible format).
        Also persists to outputs/{direction}.json.

        Returns
        -------
        dict
            {"direction": ..., "rules_set": [...]}
        """
        logger.info(
            "Phase 3 [%s]: pool=%d, refine_pop=%d, refine_gen=%d, "
            "parallel_batch=%s, jax_gpu=%s",
            self.direction,
            len(self.pool),
            self.refine_pop_size,
            self.refine_generations,
            self._use_parallel_batch,
            self._use_jax_gpu,
        )

        greedy_set, n_greedy_evals = greedy_rule_set_search(
            pool=self.pool,
            val_engine=self._val_engine,
            train_engine=self._train_engine,
            min_rules=_cfg.PHASE3_MIN_RULES,
            max_rules=_cfg.PHASE3_MAX_RULES,
            use_batch=self._use_parallel_batch or self._use_jax_gpu,
            use_jax=self._use_jax_gpu,
            cache=self._eval_cache,
            cv_fold_contexts=self._cv_fold_contexts,
        )
        logger.info(
            "Phase 3 [%s]: greedy done (%d evals), refining...",
            self.direction,
            n_greedy_evals,
        )
        initial_pop = _seed_population_from_greedy(
            greedy_set,
            self.pool,
            self.refine_pop_size,
            _cfg.PHASE3_MIN_RULES,
            _cfg.PHASE3_MAX_RULES,
            random.Random(self.seed),
        )

        refine_tag = "Phase 3 [%s] refine" % self.direction
        pareto_rule_sets, history = _run_nsga2_combinatorial(
            pool=self.pool,
            val_engine=self._val_engine,
            train_engine=self._train_engine,
            pop_size=self.refine_pop_size,
            n_generations=self.refine_generations,
            min_rules=_cfg.PHASE3_MIN_RULES,
            max_rules=_cfg.PHASE3_MAX_RULES,
            seed=self.seed,
            initial_population=initial_pop,
            use_batch=self._use_parallel_batch or self._use_jax_gpu,
            use_jax=self._use_jax_gpu,
            cache=self._eval_cache,
            log_tag=refine_tag,
            cv_fold_contexts=self._cv_fold_contexts,
        )
        logger.info(
            "Phase 3 [%s]: refine complete, pareto_front=%d rule sets",
            self.direction, len(pareto_rule_sets),
        )

        if not pareto_rule_sets:
            # Fallback: use first min_rules rules from pool
            logger.warning(
                "Phase 3 [%s]: Pareto front empty, using first %d pool rules.",
                self.direction, _cfg.PHASE3_MIN_RULES,
            )
            best_rule_set = self.pool[: _cfg.PHASE3_MIN_RULES]
        else:
            best_rule_set = _select_best_from_pareto(
                pareto_rule_sets,
                self._val_engine,
                self._train_engine,
                cache=self._eval_cache,
                cv_fold_contexts=self._cv_fold_contexts,
            )

        output_dict = _build_output_dict(best_rule_set, self.direction)

        # Persist
        os.makedirs(_cfg.OUTPUTS_DIR, exist_ok=True)
        output_path = _OUTPUT_PATHS[self.direction]
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(output_dict, fh, indent=2)

        logger.info(
            "Phase 3 [%s]: best rule set has %d rules, saved to %s",
            self.direction, len(output_dict["rules_set"]), output_path,
        )

        # Reporter: equity curves and per-symbol CSVs for train and validation splits
        try:
            engine_fmt = _rule_set_to_engine_format(best_rule_set)
            train_metrics, train_log = self._train_engine.simulate_rule_set(
                engine_fmt, return_logs=True
            )
            Reporter().plot_equity_curve(train_log, "train", self.direction)
            Reporter().write_per_symbol_csv(
                train_metrics, "train", direction=self.direction)
        except Exception as exc:
            logger.warning(
                "Reporter train equity/csv failed (non-fatal): %s", exc
            )

        try:
            engine_fmt = _rule_set_to_engine_format(best_rule_set)
            val_metrics, val_log = self._val_engine.simulate_rule_set(
                engine_fmt, return_logs=True
            )
            Reporter().plot_equity_curve(val_log, "validation", self.direction)
            Reporter().write_per_symbol_csv(
                val_metrics, "validation", direction=self.direction)
        except Exception as exc:
            logger.warning(
                "Reporter validation equity/csv failed (non-fatal): %s", exc
            )

        return output_dict

    @staticmethod
    def load_rule_set(direction: str) -> Optional[dict]:
        """
        Load existing rule set if valid, return None if missing.

        Parameters
        ----------
        direction : str
            "long" or "short".

        Returns
        -------
        dict | None
            Loaded rule set if file exists and is valid, None if missing.

        Raises
        ------
        ValueError
            If the file exists but is corrupted or has an invalid schema.
        """
        if direction not in _OUTPUT_PATHS:
            raise ValueError(
                f"direction must be 'long' or 'short', got {direction!r}")

        path = _OUTPUT_PATHS[direction]
        if not os.path.exists(path):
            return None

        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(
                f"Rule set file is unreadable or corrupted: {path}"
            ) from exc

        _validate_rule_set_schema(data, path)
        return data

    @staticmethod
    def skip_if_valid() -> Optional[dict[str, dict]]:
        """
        Return loaded rule sets if both valid, None if need to run.

        Per spec:
          - If both long.json and short.json exist and pass validation → return both.
          - If only one exists → return the available one (partial dict).
          - If neither exists → return None.

        Returns
        -------
        dict[str, dict] | None
            {"long": ..., "short": ...} if both valid,
            {"long": ...} or {"short": ...} if only one valid,
            None if neither exists.
        """
        result: dict[str, dict] = {}

        for direction in ("long", "short"):
            try:
                loaded = Rule_Set_Selector.load_rule_set(direction)
                if loaded is not None:
                    result[direction] = loaded
            except ValueError:
                # File exists but is invalid — treat as missing
                pass

        if not result:
            return None
        return result
