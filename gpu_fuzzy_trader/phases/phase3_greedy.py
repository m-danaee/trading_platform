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
) -> float:
    """Higher is better."""
    w1, w2, w3 = weights
    val_return = float(val_metrics.get("total_return_pct", 0.0))
    val_dd = float(val_metrics.get("max_drawdown_pct", 0.0))
    val_wr = float(val_metrics.get("win_rate", 0.0))
    penalty = float(objectives[0] + val_return)  # excess penalty vs raw return
    return w1 * val_return - w2 * val_dd + w3 * val_wr - penalty


def _evaluate_candidates_batch(
    candidates: list[list[dict]],
    val_engine: Any,
    train_engine: Any,
    use_batch: bool,
) -> list[tuple[np.ndarray, dict]]:
    """Evaluate rule-set candidates; use GPU batch path when available."""
    p3 = _helpers()
    engine_fmt = [p3._rule_set_to_engine_format(c) for c in candidates]
    results: list[tuple[np.ndarray, dict]] = []

    if use_batch and hasattr(val_engine, "simulate_rule_set_batch"):
        val_metrics_list = val_engine.simulate_rule_set_batch(engine_fmt)
        if hasattr(train_engine, "simulate_rule_set_batch"):
            train_metrics_list = train_engine.simulate_rule_set_batch(
                engine_fmt)
        else:
            train_metrics_list = [
                train_engine.simulate_rule_set(rs) for rs in engine_fmt
            ]

        for rs, val_m, train_m in zip(candidates, val_metrics_list, train_metrics_list):
            obj = _objectives_from_metrics(val_m, train_m, rs)
            results.append((obj, val_m))
        return results

    for rs in candidates:
        fmt = p3._rule_set_to_engine_format(rs)
        obj, val_m = p3._evaluate_rule_set(fmt, val_engine, train_engine)
        results.append((obj, val_m))

    return results


def _objectives_from_metrics(
    val_metrics: dict,
    train_metrics: dict,
    rule_set_template: list[dict],
) -> np.ndarray:
    """Replicate _evaluate_rule_set penalty logic from metrics only."""
    p3 = _helpers()
    dup_penalty = 50.0 if p3._has_duplicate_rules(rule_set_template) else 0.0

    val_return = float(val_metrics.get("total_return_pct", 0.0))
    val_dd = float(val_metrics.get("max_drawdown_pct", 100.0))
    val_wr = float(val_metrics.get("win_rate", 0.0))
    val_trades = int(val_metrics.get("executed_trades", 0))

    zero_penalty = 100.0 if val_trades == 0 else 0.0
    symbols_with_trades = p3._count_symbols_with_trades(val_metrics)
    coverage_penalty = 0.0
    if symbols_with_trades < _cfg.PHASE3_MIN_SYMBOL_COVERAGE:
        coverage_penalty = (
            (_cfg.PHASE3_MIN_SYMBOL_COVERAGE - symbols_with_trades) * 5.0
        )

    train_return = float(train_metrics.get("total_return_pct", 0.0))
    overfitting_penalty = abs(train_return - val_return) / \
        max(abs(train_return), 1.0)

    total_penalty = zero_penalty + coverage_penalty + \
        overfitting_penalty + dup_penalty
    return np.array(
        [-val_return + total_penalty, val_dd +
            total_penalty, -val_wr + total_penalty],
        dtype=np.float64,
    )


def greedy_rule_set_search(
    pool: list[dict],
    val_engine: Any,
    train_engine: Any,
    min_rules: int,
    max_rules: int,
    weights: tuple[float, float, float] | None = None,
    use_batch: bool = False,
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
    n_evals = 0

    if len(pool) < min_rules:
        raise ValueError(
            f"pool needs at least {min_rules} rules, got {len(pool)}")

    # Round 1: single rules
    candidates = [[pool[i]] for i in range(len(pool))]
    batch_results = _evaluate_candidates_batch(
        candidates, val_engine, train_engine, use_batch
    )
    n_evals += len(candidates)

    best_score = -np.inf
    best_set: list[dict] = [pool[0]]
    for i, (obj, val_m) in enumerate(batch_results):
        score = _scalar_score(val_m, obj, weights)
        if score > best_score:
            best_score = score
            best_set = [pool[i]]

    logger.info(
        "Phase 3 greedy round 1/%d: %d candidates, best_val=%.2f%% (1 rule)",
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
            extensions, val_engine, train_engine, use_batch
        )
        n_evals += len(extensions)

        round_best_score = -np.inf
        round_best: list[dict] | None = None
        for ext, (obj, val_m) in zip(extensions, batch_results):
            score = _scalar_score(val_m, obj, weights)
            if score > round_best_score:
                round_best_score = score
                round_best = ext

        if round_best is not None:
            best_set = round_best
            used_keys = {p3._conditions_key(r["conditions"]) for r in best_set}
            logger.info(
                "Phase 3 greedy round %d/%d: %d candidates, best_val=%.2f%% (%d rules)",
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
