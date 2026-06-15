"""
evaluator_health.py — Pure functions for evaluator-failure-mode awareness.

These functions detect and penalise execution issues that the evaluator's
backtest engine (evaluator_v5) would report: high skip ratio (signals below
MIN_POSITION_NOTIONAL = 1.0), low executed/raw ratio, and excessive
simultaneous positions.

No engine calls, no I/O, no side effects — pure metric analysis.
"""

from __future__ import annotations

import logging

from gpu_fuzzy_trader import config as _cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(metrics: dict, key: str, default: float = 0.0) -> float:
    """Read a numeric metric, returning *default* for missing / None / NaN / Inf."""
    val = metrics.get(key)
    if val is None:
        return float(default)
    try:
        f = float(val)
        return f if (f == f) and abs(f) != float("inf") else float(default)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(metrics: dict, key: str, default: int = 0) -> int:
    """Read an integer metric safely."""
    f = _safe_float(metrics, key, default=float(default))
    return int(f) if (f == f) else int(default)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluator_health_penalty(
    metrics: dict,
    *,
    role: str = "valid",
) -> float:
    """Penalty for evaluator_v5 failure modes — higher is worse.

    Parameters
    ----------
    metrics : dict
        Backtest metrics dict from ``CPUBacktestEngine.simulate_rule_set``.
        Must contain keys ``raw_signal_count``, ``executed_trades``,
        ``skipped_min_notional_count``, ``max_simultaneous_positions``.
    role : str
        One of ``"train"``, ``"valid"``, or ``"test"``.
        ``"test"`` applies a 1.5× multiplier to skip/exec penalties
        and a 1.2× multiplier to max-positions penalty.

    Returns
    -------
    float
        Non-negative penalty value.

    Notes
    -----
    Ported from ``friend_project/gpu_fuzzy_trader/rb_governor.py``
    ``_evaluator_health_penalty``.
    """
    raw = max(0, _safe_int(metrics, "raw_signal_count", 0))
    executed = max(0, _safe_int(metrics, "executed_trades", 0))
    skipped = max(0, _safe_int(metrics, "skipped_min_notional_count", 0))
    max_pos = max(0, _safe_int(metrics, "max_simultaneous_positions", 0))

    # --- Config thresholds ---
    max_skip = float(getattr(_cfg, "EVAL_HEALTH_MAX_SKIPPED_RATIO", 0.20))
    min_exec = float(getattr(_cfg, "EVAL_HEALTH_MIN_EXECUTED_RATIO", 0.60))
    skip_weight = float(getattr(_cfg, "EVAL_HEALTH_SKIPPED_WEIGHT", 3500.0))
    exec_weight = float(getattr(_cfg, "EVAL_HEALTH_EXECUTED_WEIGHT", 2500.0))
    pos_limit = int(getattr(_cfg, "EVAL_HEALTH_MAX_SIMULTANEOUS_POSITIONS", 10))
    pos_weight = float(getattr(_cfg, "EVAL_HEALTH_MAX_POSITIONS_WEIGHT", 120.0))

    # --- Role multipliers ---
    role_mult = 1.5 if role == "test" else 1.0
    pos_role_mult = 1.2 if role == "test" else 1.0

    penalty = 0.0

    if raw > 0:
        skip_ratio = skipped / raw
        exec_ratio = executed / raw

        if skip_ratio > max_skip:
            penalty += (skip_ratio - max_skip) * skip_weight * role_mult

        if exec_ratio < min_exec:
            penalty += (min_exec - exec_ratio) * exec_weight * role_mult

    if max_pos > pos_limit:
        penalty += (max_pos - pos_limit) * pos_weight * pos_role_mult

    return float(penalty)


def execution_ok(metrics: dict) -> bool:
    """Return ``True`` iff the evaluator would execute this rule set reasonably.

    A rule set passes if *all* of the following hold:

    * ``raw_signal_count > 0``
    * ``skipped_min_notional_count / raw_signal_count <= EVAL_HEALTH_MAX_SKIPPED_RATIO``
    * ``executed_trades / raw_signal_count >= EVAL_HEALTH_MIN_EXECUTED_RATIO``

    Missing keys or zero ``raw_signal_count`` return ``False``.

    Parameters
    ----------
    metrics : dict
        Backtest metrics dict (same shape as for ``evaluator_health_penalty``).

    Returns
    -------
    bool
    """
    raw = max(0, _safe_int(metrics, "raw_signal_count", 0))
    if raw <= 0:
        return False

    skipped = max(0, _safe_int(metrics, "skipped_min_notional_count", 0))
    executed = max(0, _safe_int(metrics, "executed_trades", 0))

    max_skip = float(getattr(_cfg, "EVAL_HEALTH_MAX_SKIPPED_RATIO", 0.20))
    min_exec = float(getattr(_cfg, "EVAL_HEALTH_MIN_EXECUTED_RATIO", 0.60))

    skip_ratio_ok = (skipped / raw) <= max_skip
    exec_ratio_ok = (executed / raw) >= min_exec

    return bool(skip_ratio_ok and exec_ratio_ok)
