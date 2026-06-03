"""
Validation-primary objective for config Optuna studies.
"""

from __future__ import annotations

from typing import Any

from gpu_fuzzy_trader import config as _cfg


def extract_test_metrics(
    phase5_result: dict[str, Any],
    direction: str,
) -> dict[str, Any]:
    """Extract test-split metrics for a direction (nested or legacy flat)."""
    entry = phase5_result.get(direction, {})
    if not entry:
        return {}
    if "test" in entry:
        return entry["test"]
    return entry


def extract_validation_metrics(
    phase5_result: dict[str, Any],
    direction: str,
) -> dict[str, Any]:
    """Extract validation-split metrics for a direction."""
    entry = phase5_result.get(direction, {})
    if not entry:
        return {}
    return entry.get("validation", {})


def compute_validation_objective(
    phase5_result: dict[str, Any],
    *,
    drawdown_weight: float = 0.5,
    gate_penalty: float = 20.0,
    val_return_gate_pct: float | None = None,
) -> tuple[float, dict[str, float]]:
    """
    Maximize validation robustness across long and short.

    score = min(val_return_long, val_return_short)
            - drawdown_weight * max(val_dd_long, val_dd_short)
            - gate_penalty if either direction fails the validation return gate

    Returns
    -------
    score, details
        Scalar objective and diagnostic floats for logging / user_attrs.
    """
    gate = (
        _cfg.PHASE5_VALIDATION_RETURN_GATE_PCT
        if val_return_gate_pct is None
        else val_return_gate_pct
    )

    details: dict[str, float] = {}
    val_returns: list[float] = []
    val_dds: list[float] = []
    test_returns: dict[str, float] = {}

    for direction in ("long", "short"):
        val_m = extract_validation_metrics(phase5_result, direction)
        test_m = extract_test_metrics(phase5_result, direction)

        val_ret = float(val_m.get("total_return_pct", 0.0))
        val_dd = float(val_m.get("max_drawdown_pct", 0.0))
        val_returns.append(val_ret)
        val_dds.append(val_dd)

        details[f"val_return_{direction}"] = val_ret
        details[f"val_dd_{direction}"] = val_dd
        details[f"test_return_{direction}"] = float(
            test_m.get("total_return_pct", 0.0)
        )

    if not val_returns:
        return -1e6, details

    min_val_return = min(val_returns)
    max_val_dd = max(val_dds)
    penalty = 0.0
    if any(r < gate for r in val_returns):
        penalty = gate_penalty

    score = min_val_return - drawdown_weight * max_val_dd - penalty
    details["score"] = score
    details["min_val_return"] = min_val_return
    details["max_val_dd"] = max_val_dd
    details["gate_penalty"] = penalty

    return score, details
