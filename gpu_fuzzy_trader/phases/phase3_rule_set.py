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
from gpu_fuzzy_trader.log_progress import maybe_log_generation
from gpu_fuzzy_trader.phases.phase3_greedy import greedy_rule_set_search
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

def _conditions_key(conditions: list[str]) -> frozenset[str]:
    """Order-independent condition set key for duplicate detection."""
    return frozenset(conditions)


def _has_duplicate_rules(rule_set: list[dict]) -> bool:
    """Return True if any two rules have identical condition sets (order-independent)."""
    seen: set[frozenset] = set()
    for rule in rule_set:
        key = _conditions_key(rule["conditions"])
        if key in seen:
            return True
        seen.add(key)
    return False


def _count_symbols_with_trades(metrics: dict) -> int:
    """Count how many symbols have at least one trade."""
    per_sym = metrics.get("per_symbol_metrics", {})
    return sum(1 for v in per_sym.values() if v.get("trade_count", 0) > 0)


def _symbols_with_trades(metrics: dict) -> set:
    """Return symbol IDs with at least one trade."""
    per_sym = metrics.get("per_symbol_metrics", {})
    return {
        s for s, v in per_sym.items()
        if v.get("trade_count", 0) > 0
    }


def _symbol_consistency_penalty(train_metrics: dict, val_metrics: dict) -> float:
    """Penalise rule sets that trade different symbols on train vs validation."""
    train_syms = _symbols_with_trades(train_metrics)
    val_syms = _symbols_with_trades(val_metrics)
    if not train_syms or not val_syms:
        return 0.0
    overlap = len(train_syms & val_syms) / len(train_syms | val_syms)
    return (1.0 - overlap) * _cfg.PHASE3_SYMBOL_CONSISTENCY_WEIGHT


def _build_phase3_engines(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    direction: str,
) -> tuple[CPUBacktestEngine | object, CPUBacktestEngine | object, bool]:
    """Create validation/train engines; GPU batch when configured and available."""
    feature_modes: dict[str, str] = {}
    use_gpu = bool(_cfg.PHASE3_USE_GPU)

    if use_gpu:
        try:
            from gpu_fuzzy_trader.backtest.gpu_engine import GPUBacktestEngine

            val_engine = GPUBacktestEngine(val_df, feature_modes, direction)
            train_engine = GPUBacktestEngine(
                train_df, feature_modes, direction)
            logger.info(
                "Phase 3 using GPUBacktestEngine (batch rule-set eval)")
            return val_engine, train_engine, True
        except ImportError:
            logger.warning(
                "PHASE3_USE_GPU=True but JAX unavailable; using CPU.")

    val_engine = CPUBacktestEngine(val_df, feature_modes, direction)
    train_engine = CPUBacktestEngine(train_df, feature_modes, direction)
    return val_engine, train_engine, False


def _evaluate_rule_set(
    rule_set: list[dict],
    val_engine,
    train_engine,
) -> tuple[np.ndarray, dict]:
    """
    Evaluate a candidate rule set and return (objectives, val_metrics).

    objectives = [f1, f2, f3] (all minimised, with penalties applied).
    """
    # Duplicate rule penalty (fast check before backtest)
    dup_penalty = 0.0
    if _has_duplicate_rules(rule_set):
        dup_penalty = 50.0

    # Evaluate on validation split
    try:
        val_metrics = val_engine.simulate_rule_set(rule_set)
    except Exception as exc:
        logger.debug("val simulate_rule_set failed: %s", exc)
        val_metrics = {
            "sortino_ratio": 0.0,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 100.0,
            "win_rate": 0.0,
            "executed_trades": 0,
            "per_symbol_metrics": {},
        }

    val_sortino = float(val_metrics.get(
        "sortino_ratio", val_metrics.get("total_return_pct", 0.0)))
    val_dd = float(val_metrics.get("max_drawdown_pct", 100.0))
    val_wr = float(val_metrics.get("win_rate", 0.0))
    val_trades = int(val_metrics.get("executed_trades", 0))

    # Zero-trade penalty
    zero_penalty = 0.0
    if val_trades == 0:
        zero_penalty = 100.0

    # Coverage penalty
    coverage_penalty = 0.0
    symbols_with_trades = _count_symbols_with_trades(val_metrics)
    if symbols_with_trades < _cfg.PHASE3_MIN_SYMBOL_COVERAGE:
        coverage_penalty = (
            (_cfg.PHASE3_MIN_SYMBOL_COVERAGE - symbols_with_trades) * 5.0
        )

    overfitting_penalty = 0.0
    train_metrics: dict = {}
    try:
        train_metrics = train_engine.simulate_rule_set(rule_set)
        train_sortino = float(train_metrics.get(
            "sortino_ratio", train_metrics.get("total_return_pct", 0.0)))
        overfitting_penalty = abs(
            train_sortino - val_sortino) / max(abs(train_sortino), 1.0)
    except Exception as exc:
        logger.debug("train simulate_rule_set failed: %s", exc)
        overfitting_penalty = 10.0

    symbol_consistency_penalty = _symbol_consistency_penalty(
        train_metrics, val_metrics)

    total_penalty = (
        zero_penalty + coverage_penalty + overfitting_penalty
        + dup_penalty + symbol_consistency_penalty
    )

    f1 = -val_sortino + total_penalty
    f2 = val_dd + total_penalty
    f3 = -val_wr + total_penalty

    objectives = np.array([f1, f2, f3], dtype=np.float64)
    return objectives, val_metrics


# ---------------------------------------------------------------------------
# NSGA-II helpers (reused from phase2 pattern)
# ---------------------------------------------------------------------------

def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Return True if solution *a* dominates *b* (all ≤, at least one <)."""
    return bool(np.all(a <= b) and np.any(a < b))


def _non_dominated_sort(objectives: np.ndarray) -> list[list[int]]:
    """
    NSGA-II non-dominated sorting.

    Returns fronts in order: fronts[0] is the Pareto front.
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
    """Compute crowding distance for solutions in *front*."""
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
    log_tag: str | None = None,
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
    logger.info(
        "%s: pool=%d, pop=%d, gen=%d",
        tag, len(pool), effective_pop, n_generations,
    )
    gen_loop_start = time.monotonic()

    for gen in range(n_generations):
        # Evaluate unevaluated individuals
        pending = [i for i in range(effective_pop)
                   if np.any(np.isinf(objectives[i]))]
        if pending and use_batch and hasattr(val_engine, "simulate_rule_set_batch"):
            fmts = [_rule_set_to_engine_format(population[i]) for i in pending]
            val_list = val_engine.simulate_rule_set_batch(fmts)
            if hasattr(train_engine, "simulate_rule_set_batch"):
                train_list = train_engine.simulate_rule_set_batch(fmts)
            else:
                train_list = [
                    train_engine.simulate_rule_set(f) for f in fmts
                ]
            from gpu_fuzzy_trader.phases.phase3_greedy import _objectives_from_metrics

            for j, i in enumerate(pending):
                obj = _objectives_from_metrics(
                    val_list[j], train_list[j], population[i]
                )
                objectives[i] = obj
        else:
            for i in pending:
                engine_fmt = _rule_set_to_engine_format(population[i])
                obj, _ = _evaluate_rule_set(
                    engine_fmt, val_engine, train_engine)
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


def _select_best_from_pareto(
    pareto_rule_sets: list[list[dict]],
    val_engine: CPUBacktestEngine,
    train_engine: CPUBacktestEngine,
) -> list[dict]:
    """
    Select the best rule set from the Pareto front.

    Strategy: pick the rule set with the highest validation return (minimum f1).
    If the Pareto front is empty, fall back to the first rule set.
    """
    if not pareto_rule_sets:
        raise ValueError(
            "Pareto front is empty — cannot select best rule set.")

    best_idx = 0
    best_f1 = np.inf

    for i, rs in enumerate(pareto_rule_sets):
        engine_fmt = _rule_set_to_engine_format(rs)
        obj, _ = _evaluate_rule_set(engine_fmt, val_engine, train_engine)
        if obj[0] < best_f1:
            best_f1 = obj[0]
            best_idx = i

    return pareto_rule_sets[best_idx]


# ---------------------------------------------------------------------------
# Cross-fold aggregated selection
# ---------------------------------------------------------------------------

def select_best_cross_fold(
    candidates: list[list[dict]],
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    direction: str,
) -> list[dict]:
    """
    Select the best candidate by aggregating validation performance across
    all purged walk-forward folds.

    For each candidate rule set, evaluate on every fold's validation window,
    compute the mean Sortino ratio across folds, and pick the candidate with
    the highest cross-fold average Sortino.

    Parameters
    ----------
    candidates : list[list[dict]]
        Pareto-front candidates from ``Rule_Set_Selector.build_pareto_candidates()``.
    folds : list[tuple[pd.DataFrame, pd.DataFrame]]
        Purged walk-forward fold pairs (train_df, val_df).
    direction : str
        "long" or "short".

    Returns
    -------
    list[dict]
        The best candidate rule set (pool-format list of rule dicts).
    """
    from gpu_fuzzy_trader.backtest.cpu_engine import CPUBacktestEngine

    best_candidate = candidates[0] if candidates else []
    best_avg_sortino = -np.inf

    for cand_i, cand in enumerate(candidates):
        engine_fmt = _rule_set_to_engine_format(cand)
        fold_sortinos: list[float] = []
        fold_trades: list[int] = []

        for fi, (_train_df, val_df) in enumerate(folds):
            try:
                engine = CPUBacktestEngine(val_df, {}, direction)
                metrics = engine.simulate_rule_set(engine_fmt)
                s = float(metrics.get(
                    "sortino_ratio",
                    metrics.get("total_return_pct", 0.0),
                ))
                t = int(metrics.get("executed_trades", 0))
                fold_sortinos.append(s)
                fold_trades.append(t)
            except Exception as exc:
                logger.debug(
                    "Cross-fold candidate %d fold %d failed: %s", cand_i, fi, exc)
                fold_sortinos.append(0.0)
                fold_trades.append(0)

        if not fold_sortinos:
            continue

        avg_sortino = float(np.mean(fold_sortinos))
        total_trades = sum(fold_trades)

        # Penalise candidates with zero or very few trades
        if total_trades == 0:
            avg_sortino = -np.inf

        if avg_sortino > best_avg_sortino:
            best_avg_sortino = avg_sortino
            best_candidate = cand

    logger.info(
        "Cross-fold selection: %d candidates evaluated on %d folds, "
        "best avg Sortino=%.4f",
        len(candidates), len(folds), best_avg_sortino,
    )
    return best_candidate


# ---------------------------------------------------------------------------
# Output serialisation
# ---------------------------------------------------------------------------

def _build_output_dict(rule_set: list[dict], direction: str) -> dict:
    """
    Build the evaluator_v3.ipynb-compatible output dict.

    Uses Phase 2 static TP/SL/capital_pct values (Phase 4 will update them).
    """
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

        self._val_engine, self._train_engine, self._use_gpu_batch = _build_phase3_engines(
            train_df, val_df, direction
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_pareto_candidates(self) -> list[list[dict]]:
        """
        Run greedy construction and Pareto refinement, returning the raw
        Pareto front of rule-set candidates.  Does NOT call
        ``_select_best_from_pareto`` — the caller is responsible for
        selecting the best candidate (e.g. via cross-fold aggregation).

        Returns
        -------
        list[list[dict]]
            Pareto-front candidates, each a list of rule dicts.
        """
        logger.info(
            "Phase 3 [%s]: pool=%d, refine_pop=%d, refine_gen=%d, gpu_batch=%s",
            self.direction,
            len(self.pool),
            self.refine_pop_size,
            self.refine_generations,
            self._use_gpu_batch,
        )

        greedy_set, n_greedy_evals = greedy_rule_set_search(
            pool=self.pool,
            val_engine=self._val_engine,
            train_engine=self._train_engine,
            min_rules=_cfg.PHASE3_MIN_RULES,
            max_rules=_cfg.PHASE3_MAX_RULES,
            use_batch=self._use_gpu_batch,
        )
        logger.info(
            "Phase 3 [%s]: greedy done (%d evals), refining...",
            self.direction, n_greedy_evals,
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
        pareto_rule_sets, _history = _run_nsga2_combinatorial(
            pool=self.pool,
            val_engine=self._val_engine,
            train_engine=self._train_engine,
            pop_size=self.refine_pop_size,
            n_generations=self.refine_generations,
            min_rules=_cfg.PHASE3_MIN_RULES,
            max_rules=_cfg.PHASE3_MAX_RULES,
            seed=self.seed,
            initial_population=initial_pop,
            use_batch=self._use_gpu_batch,
            log_tag=refine_tag,
        )
        logger.info(
            "Phase 3 [%s]: refine complete, pareto_front=%d candidates",
            self.direction, len(pareto_rule_sets),
        )

        if not pareto_rule_sets:
            pareto_rule_sets = [self.pool[:_cfg.PHASE3_MIN_RULES]]

        return pareto_rule_sets

    def run(self) -> dict:
        """
        Run greedy construction, Pareto refinement, and select best.

        Delegates to ``build_pareto_candidates()`` + single-fold selection.
        For cross-fold aggregation use ``build_pareto_candidates()`` then
        ``select_best_cross_fold()``.

        Returns
        -------
        dict
            {"direction": ..., "rules_set": [...]}
        """
        pareto_rule_sets = self.build_pareto_candidates()

        best_rule_set = _select_best_from_pareto(
            pareto_rule_sets, self._val_engine, self._train_engine
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
            Reporter().write_per_symbol_csv(train_metrics, "train")
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
            Reporter().write_per_symbol_csv(val_metrics, "validation")
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
