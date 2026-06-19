"""Scoring helpers shared across pipeline phases.

Re-exports :func:`evaluator_health_penalty` and :func:`execution_ok` from
:mod:`gpu_fuzzy_trader.scoring.evaluator_health` and adds small numeric
helpers used by the RB Governor scoring objective.
"""

from __future__ import annotations

import math

from gpu_fuzzy_trader.scoring.evaluator_health import (
    evaluator_health_penalty,
    execution_ok,
)


def return_to_drawdown(
    return_pct: float,
    drawdown_pct: float,
    dd_floor: float = 1.0,
) -> float:
    """Return divided by max drawdown; higher is better.

    A small drawdown floor avoids exploding scores for tiny/no-trade drawdowns.
    Negative returns stay negative, which is what we want.
    """
    try:
        ret = float(return_pct)
        dd = abs(float(drawdown_pct))
    except Exception:
        return -1e6
    if not math.isfinite(ret) or not math.isfinite(dd):
        return -1e6
    return ret / max(dd, float(dd_floor), 1e-9)


def profit_factor_term(pf: float, cap: float = 3.0) -> float:
    """Clamp a profit factor into ``[0, cap]``; non-finite → ``cap``."""
    try:
        pf = float(pf)
    except Exception:
        pf = 0.0
    if not math.isfinite(pf):
        pf = cap
    return max(0.0, min(pf, cap))


__all__ = [
    "evaluator_health_penalty",
    "execution_ok",
    "return_to_drawdown",
    "profit_factor_term",
]
