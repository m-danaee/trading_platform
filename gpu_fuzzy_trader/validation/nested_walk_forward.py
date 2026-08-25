"""Backward-compatible import shim for the stability diagnostic.

The canonical implementation lives in :mod:`walk_forward_stability_report`.
Keep this small adapter for research notebooks and archived tests that still
import the historical module path; production code must use the stability API.
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from gpu_fuzzy_trader.validation.walk_forward_stability_report import (
    StabilityFold,
    build_stability_folds,
    evaluate_strategy_stability,
    write_strategy_stability_reports,
)


NestedFold = StabilityFold


def build_nested_folds(
    frame: pd.DataFrame,
    *,
    n_outer: int = 3,
    min_train_fraction: float = 0.40,
    purge_candles: int | None = None,
) -> list[StabilityFold]:
    """Compatibility wrapper for the renamed chronological fold builder."""
    return build_stability_folds(
        frame,
        n_windows=n_outer,
        min_train_fraction=min_train_fraction,
        purge_candles=purge_candles,
    )


def evaluate_nested_strategy(
    frame: pd.DataFrame,
    strategy: dict[str, Any],
    *,
    n_outer: int = 3,
    evaluator: Callable[[pd.DataFrame, list[dict]], dict] | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper for :func:`evaluate_strategy_stability`."""
    return evaluate_strategy_stability(
        frame,
        strategy,
        n_windows=n_outer,
        evaluator=evaluator,
    )


def write_nested_reports(
    output_dir: str,
    strategies: dict[str, dict[str, Any]],
    frame: pd.DataFrame,
    *,
    n_outer: int = 3,
) -> dict[str, dict[str, Any]]:
    """Compatibility wrapper for the renamed stability report writer."""
    return write_strategy_stability_reports(
        output_dir,
        strategies,
        frame,
        n_windows=n_outer,
    )


__all__ = [
    "NestedFold",
    "build_nested_folds",
    "evaluate_nested_strategy",
    "write_nested_reports",
]
