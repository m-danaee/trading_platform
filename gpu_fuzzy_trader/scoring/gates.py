"""Shared metric gates used by Phase 2 admission and RB Governor.

The gate is deliberately pure: it does not simulate rules, read files, or
consult a phase-specific module.  Callers provide the thresholds that are
appropriate for their data slice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PositiveGoodThresholds:
    """Minimum evidence required on train and validation splits."""

    min_train_return: float = 0.0
    min_valid_return: float = 0.0
    min_train_profit_factor: float = 1.0
    min_valid_profit_factor: float = 1.0
    min_train_trades: int = 0
    min_valid_trades: int = 0
    require_execution_health: bool = False


def _finite_number(metrics: dict | None, key: str, default: float = 0.0) -> float:
    value = (metrics or {}).get(key)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _metric_int(metrics: dict | None, key: str) -> int:
    return int(_finite_number(metrics, key, 0.0))


def positive_good_reject_reasons(
    train_metrics: dict,
    valid_metrics: dict | None,
    thresholds: PositiveGoodThresholds,
) -> list[str]:
    """Return stable, machine-readable reasons for a gate rejection."""

    reasons: list[str] = []
    train_return = _finite_number(train_metrics, "total_return_pct")
    valid_return = _finite_number(valid_metrics, "total_return_pct")
    train_pf = _finite_number(train_metrics, "profit_factor")
    valid_pf = _finite_number(valid_metrics, "profit_factor")
    train_trades = _metric_int(train_metrics, "executed_trades")
    valid_trades = _metric_int(valid_metrics, "executed_trades")

    if train_return <= float(thresholds.min_train_return):
        reasons.append("train_return_floor")
    if valid_return <= float(thresholds.min_valid_return):
        reasons.append("valid_return_floor")
    if train_pf < float(thresholds.min_train_profit_factor):
        reasons.append("train_profit_factor_floor")
    if valid_pf < float(thresholds.min_valid_profit_factor):
        reasons.append("valid_profit_factor_floor")
    if train_trades < int(thresholds.min_train_trades):
        reasons.append("train_trade_floor")
    if valid_trades < int(thresholds.min_valid_trades):
        reasons.append("valid_trade_floor")

    if thresholds.require_execution_health:
        from gpu_fuzzy_trader.scoring.evaluator_health import execution_ok

        if not execution_ok(train_metrics or {}):
            reasons.append("train_execution_health")
        if not execution_ok(valid_metrics or {}):
            reasons.append("valid_execution_health")

    return reasons


def gate_positive_good(
    train_metrics: dict,
    valid_metrics: dict | None,
    thresholds: PositiveGoodThresholds | None = None,
    **legacy_thresholds: object,
) -> bool:
    """Return whether both splits satisfy one explicit evidence contract.

    ``legacy_thresholds`` accepts the old keyword names temporarily so callers
    can migrate without changing metric semantics in the same commit.
    """

    if thresholds is None:
        thresholds = PositiveGoodThresholds(
            min_train_return=float(legacy_thresholds.pop("min_train_return", 0.0)),
            min_valid_return=float(
                legacy_thresholds.pop(
                    "min_valid_return",
                    legacy_thresholds.pop("min_val_return", 0.0),
                )
            ),
            min_train_profit_factor=float(
                legacy_thresholds.pop("min_train_pf", 1.0)
            ),
            min_valid_profit_factor=float(
                legacy_thresholds.pop(
                    "min_valid_pf",
                    legacy_thresholds.pop("min_val_pf", 1.0),
                )
            ),
            min_train_trades=int(legacy_thresholds.pop("min_train_trades", 0)),
            min_valid_trades=int(
                legacy_thresholds.pop(
                    "min_valid_trades",
                    legacy_thresholds.pop("min_val_trades", 0),
                )
            ),
            require_execution_health=bool(
                legacy_thresholds.pop("require_execution_health", False)
            ),
        )
    if legacy_thresholds:
        unknown = ", ".join(sorted(legacy_thresholds))
        raise TypeError(f"Unknown positive-good threshold(s): {unknown}")
    return not positive_good_reject_reasons(train_metrics, valid_metrics, thresholds)

