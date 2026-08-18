"""Decoupled Direction & Strength Ensembling for Hierarchical MTF.

Calculates continuous (Direction, Strength) scores from active rule masks and
non-negative rule weights w_r = max(0, Edge) * max(0, Stability). Rules with
non-positive skill or MCC <= 0 receive zero weight.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence, Union
import numpy as np


def compute_rule_weights(rules: Sequence[dict[str, Any]]) -> np.ndarray:
    """Compute non-negative rule weights from validation edge and stability scores.

    Formula:
        w_r = max(0, DirectionalEdge_r) * max(0, StabilityScore_r)

    Admission constraints:
        - Rules with non-positive skill (skill <= 0) receive zero weight.
        - Rules with non-positive OOF MCC (mcc <= 0) receive zero weight.
        - Rules with non-positive edge or stability receive zero weight.

    Parameters
    ----------
    rules : Sequence[dict[str, Any]]
        List of rule dictionaries containing metrics.

    Returns
    -------
    np.ndarray
        1D np.ndarray of float64 weights of length len(rules).
    """
    if not rules:
        return np.empty(0, dtype=np.float64)

    n_rules = len(rules)
    weights = np.zeros(n_rules, dtype=np.float64)

    for i, r in enumerate(rules):
        # Extract directional edge with alias fallbacks
        edge = float(r.get("directional_edge", r.get("edge", 0.0)))

        # Extract stability score with alias fallbacks
        stability = float(
            r.get("stability", r.get("stability_score", r.get("temporal_stability", 1.0)))
        )

        # Extract MCC with alias fallbacks (defaults to 1.0 if omitted)
        mcc = float(r.get("mcc", r.get("oof_mcc", 1.0)))

        # Extract skill with alias fallbacks (defaults to 1.0 if omitted)
        skill = float(r.get("skill", 1.0))

        # Check hard admission constraints
        if mcc <= 0.0 or skill <= 0.0 or edge <= 0.0 or stability <= 0.0:
            weights[i] = 0.0
        else:
            weights[i] = max(0.0, edge) * max(0.0, stability)

    return weights


def compute_ensemble_direction_and_strength(
    active_matrix: Union[np.ndarray, Sequence[Sequence[Any]]],
    directions: Sequence[str],
    weights: Union[np.ndarray, Sequence[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute continuous Direction in [-1.0, +1.0] and Evidence Strength in [0.0, 1.0].

    Formulas:
        Direction_t = (W_long_active(t) - W_short_active(t)) / (W_long_active(t) + W_short_active(t))
                      if (W_long_active(t) + W_short_active(t)) > 0 else 0.0

        Strength_t  = (W_long_active(t) + W_short_active(t)) / W_all
                      if W_all > 0 else 0.0

    Parameters
    ----------
    active_matrix : np.ndarray or Sequence
        2D boolean / numeric array of shape (n_samples, n_rules) indicating active rules.
    directions : Sequence[str]
        List or array of length n_rules indicating rule direction ('long' / 'short').
    weights : np.ndarray or Sequence[float]
        1D array of non-negative rule weights of length n_rules.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (direction_scores, strength_scores) as 1D np.ndarray of float64.
    """
    act = np.asarray(active_matrix, dtype=np.float64)
    if act.ndim != 2:
        if act.size == 0:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
        raise ValueError(f"active_matrix must be 2D, got shape {act.shape}")

    n_samples, n_rules = act.shape
    if n_samples == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    if n_rules == 0 or len(directions) == 0 or len(weights) == 0:
        return np.zeros(n_samples, dtype=np.float64), np.zeros(n_samples, dtype=np.float64)

    w = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    if len(w) != n_rules:
        raise ValueError(f"Length of weights ({len(w)}) does not match n_rules ({n_rules})")
    if len(directions) != n_rules:
        raise ValueError(f"Length of directions ({len(directions)}) does not match n_rules ({n_rules})")

    # Direction mask decomposition
    is_long = np.array(
        [str(d).strip().lower() in ("long", "1", "buy", "+1") for d in directions],
        dtype=bool,
    )
    is_short = np.array(
        [str(d).strip().lower() in ("short", "-1", "sell") for d in directions],
        dtype=bool,
    )

    long_weights = np.where(is_long, w, 0.0)
    short_weights = np.where(is_short, w, 0.0)

    # Matrix multiplication: (n_samples, n_rules) @ (n_rules,) -> (n_samples,)
    w_long_active = act @ long_weights
    w_short_active = act @ short_weights
    w_active = w_long_active + w_short_active
    w_all = float(np.sum(w))

    # 1. Continuous Direction score in [-1.0, +1.0]
    direction_score = np.zeros(n_samples, dtype=np.float64)
    pos_active = w_active > 1e-12
    direction_score[pos_active] = (
        (w_long_active[pos_active] - w_short_active[pos_active]) / w_active[pos_active]
    )
    direction_score = np.clip(direction_score, -1.0, 1.0)

    # 2. Continuous Evidence Strength score in [0.0, 1.0]
    strength_score = np.zeros(n_samples, dtype=np.float64)
    if w_all > 1e-12:
        strength_score = np.clip(w_active / w_all, 0.0, 1.0)

    return direction_score, strength_score


def deduplicate_rules(
    rules: Sequence[dict[str, Any]],
    ranking_key: Callable[[dict[str, Any]], Any] | None = None,
) -> list[dict[str, Any]]:
    """Deduplicate rules by condition set and direction prior to ensembling.

    If multiple rules have identical condition sets (order-invariant) and direction,
    the rule with the highest score (directional edge / MCC) is retained.

    Parameters
    ----------
    rules : Sequence[dict[str, Any]]
        List of rule dictionaries.
    ranking_key : Callable or None, optional
        Optional function mapping a rule to a sortable metric. Defaults to
        (directional_edge, mcc).

    Returns
    -------
    list[dict[str, Any]]
        Deduplicated list of rule dictionaries.
    """
    if not rules:
        return []

    if ranking_key is None:
        def default_ranking(r: dict[str, Any]) -> tuple[float, float]:
            edge = float(r.get("directional_edge", r.get("edge", 0.0)))
            mcc = float(r.get("mcc", r.get("oof_mcc", 0.0)))
            return (edge, mcc)
        ranking_key = default_ranking

    grouped: dict[tuple[str, str, tuple[str, ...]], dict[str, Any]] = {}

    for rule in rules:
        tf = str(rule.get("timeframe", rule.get("tf", ""))).strip().lower()
        direction = str(rule.get("direction", "")).strip().lower()
        conds = tuple(sorted(str(c).strip() for c in rule.get("conditions", [])))

        key = (tf, direction, conds)
        if key not in grouped:
            grouped[key] = rule
        else:
            # Compare and retain the superior rule
            if ranking_key(rule) > ranking_key(grouped[key]):
                grouped[key] = rule

    return list(grouped.values())
