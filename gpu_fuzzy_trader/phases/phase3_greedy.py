"""
phase3_greedy.py — Greedy submodular rule-set construction for Phase 3.

Builds an ordered rule set of 2–5 rules by successive extension:
  round 1: best single rule
  round 2..k: best extension of the current set with one new pool rule

Uses a scalar score for tie-breaking; multi-objective quality is restored
by the subsequent short refinement step.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

import numpy as np

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.phases.phase3_cache import Phase3EvalCache
from gpu_fuzzy_trader.phases.phase3_objectives import compute_phase3_objectives

logger = logging.getLogger(__name__)


class BacktestEnginePair(Protocol):
    """Validation + training engines used for fitness."""

    val_engine: Any
    train_engine: Any


def _helpers():
    from gpu_fuzzy_trader.phases import phase3_rule_set as p3

    return p3


def _scalar_score(
    val_metrics: dict,
    objectives: np.ndarray,
    weights: tuple[float, float, float],
    train_metrics: dict | None = None,
) -> float:
    """Higher is better."""
    w1, w2, w3 = weights
    use_train = _cfg.PHASE3_USE_TRAIN_TARGET and train_metrics is not None
    src = train_metrics if use_train else val_metrics

    train_ret = float(train_metrics.get(
        "total_return_pct", 0.0)) if train_metrics else 0.0
    val_ret = float(val_metrics.get("total_return_pct", 0.0))
    if _cfg.PHASE3_USE_MAXIMIN_SCORE and train_metrics is not None:
        primary_sortino = min(
            float(train_metrics.get(
                "sortino_ratio", train_metrics.get("total_return_pct", 0.0))),
            float(val_metrics.get(
                "sortino_ratio", val_metrics.get("total_return_pct", 0.0))),
        )
        primary_ret = min(train_ret, val_ret)
        primary_sortino = min(primary_sortino, primary_ret)
    else:
        primary_sortino = float(src.get(
            "sortino_ratio", src.get("total_return_pct", 0.0)))

    primary_dd = max(
        float(train_metrics.get("max_drawdown_pct", 0.0)
              ) if train_metrics else 0.0,
        float(val_metrics.get("max_drawdown_pct", 0.0)),
    ) if (_cfg.PHASE3_USE_MAXIMIN_SCORE and train_metrics is not None) else float(
        src.get("max_drawdown_pct", 0.0))
    primary_wr = float(src.get("win_rate", 0.0))
    penalty = float(objectives[0] + primary_sortino)
    return w1 * primary_sortino - w2 * primary_dd + w3 * primary_wr - penalty


def _objectives_from_metrics(
    val_metrics: dict,
    train_metrics: dict,
    rule_set_template: list[dict],
    val_engine: Any | None = None,
    cache: Phase3EvalCache | None = None,
    pool_size: int | None = None,
) -> np.ndarray:
    """Compute objectives from precomputed metrics (backward-compatible wrapper)."""
    _ = val_engine  # legacy callers may pass engine; cache replaces gate sims
    per_rule = cache.per_rule_min_val_trades if cache is not None else None
    val_masks = cache.val_masks if cache is not None else None
    n_rows_val = cache.n_rows_val if cache is not None else 0
    return compute_phase3_objectives(
        train_metrics,
        val_metrics,
        rule_set_template,
        per_rule_min_val_trades=per_rule,
        val_masks_by_key=val_masks,
        n_rows_val=n_rows_val,
        pool_size=pool_size,
    )


def _evaluate_candidates_batch(
    candidates: list[list[dict]],
    val_engine: Any,
    train_engine: Any,
    use_batch: bool,
    use_jax: bool = False,
    cache: Phase3EvalCache | None = None,
    cv_fold_contexts: list[tuple] | None = None,
    pool_size: int | None = None,
) -> list[tuple[np.ndarray, dict, dict]]:
    """Evaluate rule-set candidates; parallel or JAX batch when available."""
    p3 = _helpers()
    engine_fmt = [p3._rule_set_to_engine_format(c) for c in candidates]
    results: list[tuple[np.ndarray, dict, dict]] = []
    per_rule = cache.per_rule_min_val_trades if cache is not None else None
    val_masks = cache.val_masks if cache is not None else None
    n_rows_val = cache.n_rows_val if cache is not None else 0

    if cv_fold_contexts:
        for rs in candidates:
            fmt = p3._rule_set_to_engine_format(rs)
            obj, train_m, val_m = p3._evaluate_rule_set(
                fmt,
                val_engine,
                train_engine,
                cache=cache,
                cv_fold_contexts=cv_fold_contexts,
                pool_size=pool_size,
            )
            results.append((obj, val_m, train_m))
        return results

    if use_batch:
        train_list, val_list = p3._simulate_teams_batch(
            engine_fmt, train_engine, val_engine, cache, use_jax)
        for rs, train_m, val_m in zip(candidates, train_list, val_list):
            obj = compute_phase3_objectives(
                train_m,
                val_m,
                rs,
                per_rule_min_val_trades=per_rule,
                val_masks_by_key=val_masks,
                n_rows_val=n_rows_val,
                pool_size=pool_size,
            )
            results.append((obj, val_m, train_m))
        return results

    for rs in candidates:
        fmt = p3._rule_set_to_engine_format(rs)
        train_m, val_m = p3._simulate_team(
            fmt, train_engine, val_engine, cache)
        obj = compute_phase3_objectives(
            train_m,
            val_m,
            rs,
            per_rule_min_val_trades=per_rule,
            val_masks_by_key=val_masks,
            n_rows_val=n_rows_val,
            pool_size=pool_size,
        )
        results.append((obj, val_m, train_m))

    return results


def greedy_rule_set_search(
    pool: list[dict],
    val_engine: Any,
    train_engine: Any,
    min_rules: int,
    max_rules: int,
    weights: tuple[float, float, float] | None = None,
    use_batch: bool = False,
    use_jax: bool = False,
    cache: Phase3EvalCache | None = None,
    cv_fold_contexts: list[tuple] | None = None,
) -> tuple[list[dict], int]:
    """
    Greedy construction of an ordered rule set.

    Returns
    -------
    best_rule_set : list[dict]
        Pool-format rule dicts (conditions, tp, sl, capital_pct).
    n_evaluations : int
        Number of candidate rule sets evaluated.
    """
    weights = weights if weights is not None else _cfg.PHASE3_GREEDY_WEIGHTS
    pool_size = len(pool)
    n_evals = 0

    if len(pool) < min_rules:
        raise ValueError(
            f"pool needs at least {min_rules} rules, got {len(pool)}")

    use_batch_eff = bool(use_batch) and not cv_fold_contexts

    candidates = [[pool[i]] for i in range(len(pool))]
    batch_results = _evaluate_candidates_batch(
        candidates,
        val_engine,
        train_engine,
        use_batch_eff,
        use_jax,
        cache,
        cv_fold_contexts=cv_fold_contexts,
        pool_size=pool_size,
    )
    n_evals += len(candidates)

    best_score = -np.inf
    best_set: list[dict] = [pool[0]]
    for i, (obj, val_m, train_m) in enumerate(batch_results):
        score = _scalar_score(val_m, obj, weights, train_metrics=train_m)
        if score > best_score:
            best_score = score
            best_set = [pool[i]]

    logger.info(
        "Phase 3 greedy round 1/%d: %d candidates, best_score=%.2f (1 rule)",
        max_rules, len(candidates), best_score,
    )

    p3 = _helpers()
    used_keys = {p3._conditions_key(r["conditions"]) for r in best_set}

    for k in range(2, max_rules + 1):
        extensions: list[list[dict]] = []
        for rule in pool:
            key = p3._conditions_key(rule["conditions"])
            if key in used_keys:
                continue
            extensions.append(best_set + [rule])

        if not extensions:
            break

        if len(extensions) + len(best_set) < min_rules and k < min_rules:
            continue

        batch_results = _evaluate_candidates_batch(
            extensions,
            val_engine,
            train_engine,
            use_batch_eff,
            use_jax,
            cache,
            cv_fold_contexts=cv_fold_contexts,
            pool_size=pool_size,
        )
        n_evals += len(extensions)

        round_best_score = -np.inf
        round_best: list[dict] | None = None
        for ext, (obj, val_m, train_m) in zip(extensions, batch_results):
            score = _scalar_score(val_m, obj, weights, train_metrics=train_m)
            if score > round_best_score:
                round_best_score = score
                round_best = ext

        if round_best is not None:
            if (
                bool(_cfg.PHASE3_GREEDY_STOP_ON_WORSEN)
                and round_best_score <= best_score
            ):
                logger.info(
                    "Phase 3 greedy: stop at round %d — marginal score "
                    "%.2f did not improve over %.2f",
                    k,
                    round_best_score,
                    best_score,
                )
                break
            best_score = round_best_score
            best_set = round_best
            used_keys = {p3._conditions_key(r["conditions"]) for r in best_set}
            logger.info(
                "Phase 3 greedy round %d/%d: %d candidates, best_score=%.2f (%d rules)",
                k, max_rules, len(extensions), round_best_score, len(best_set),
            )

    while len(best_set) < min_rules:
        for rule in pool:
            key = p3._conditions_key(rule["conditions"])
            if key not in used_keys:
                best_set = best_set + [rule]
                used_keys.add(key)
                break
        else:
            break

    logger.info(
        "Greedy Phase 3: %d rules selected after %d evaluations",
        len(best_set),
        n_evals,
    )
    return best_set, n_evals
