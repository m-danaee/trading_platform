"""Chronological strategy-stability diagnostics.

The production pipeline has one adaptive master-fold system.  This module is a
diagnostic only: it evaluates an already-frozen strategy on chronological
stability windows and never performs candidate search or threshold fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.backtest.cpu_engine import CPUBacktestEngine
from gpu_fuzzy_trader.validation.multiplicity import summarize_multiplicity


@dataclass(frozen=True)
class StabilityFold:
    """One chronological stability window and its purged training prefix."""

    fold_id: int
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    purge_candles: int

    @property
    def inner_train_df(self) -> pd.DataFrame:
        """Compatibility view for callers that used the old fold names."""
        return self.train_df

    @property
    def outer_valid_df(self) -> pd.DataFrame:
        """Compatibility view for callers that used the old fold names."""
        return self.test_df


def _sorted_symbol_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "symbol" not in frame.columns:
        return frame.sort_values("datetime").reset_index(drop=True)
    order = ["symbol", "datetime"] if "datetime" in frame.columns else ["symbol"]
    return frame.sort_values(order).reset_index(drop=True)


def _split_bounds(
    frame: pd.DataFrame,
    *,
    n_windows: int,
    min_train_fraction: float,
) -> list[tuple[int, int]]:
    n_rows = len(frame)
    start = max(1, int(n_rows * min_train_fraction))
    usable = max(0, n_rows - start)
    if usable == 0:
        return []
    base, remainder = divmod(usable, max(1, n_windows))
    bounds: list[tuple[int, int]] = []
    cursor = start
    for index in range(max(1, n_windows)):
        width = base + (1 if index < remainder else 0)
        end = min(n_rows, cursor + width)
        if end > cursor:
            bounds.append((cursor, end))
        cursor = end
    return bounds


def build_stability_folds(
    frame: pd.DataFrame,
    *,
    n_windows: int = 3,
    min_train_fraction: float = 0.40,
    purge_candles: int | None = None,
) -> list[StabilityFold]:
    """Build chronological windows for a frozen-strategy stability report."""
    if frame.empty:
        return []
    purge = int(
        purge_candles
        if purge_candles is not None
        else getattr(_cfg, "MAX_HOLD_CANDLES", 0)
    )
    if "symbol" not in frame.columns:
        groups = {"__all__": _sorted_symbol_frame(frame)}
    else:
        groups = {
            str(symbol): _sorted_symbol_frame(group)
            for symbol, group in frame.groupby("symbol", sort=True, observed=False)
        }
    group_bounds = {
        symbol: _split_bounds(
            group,
            n_windows=max(1, int(n_windows)),
            min_train_fraction=float(min_train_fraction),
        )
        for symbol, group in groups.items()
    }
    n_folds = max((len(bounds) for bounds in group_bounds.values()), default=0)
    folds: list[StabilityFold] = []
    for fold_id in range(n_folds):
        train_parts: list[pd.DataFrame] = []
        test_parts: list[pd.DataFrame] = []
        for symbol, group in groups.items():
            bounds = group_bounds[symbol]
            if fold_id >= len(bounds):
                continue
            test_start, test_end = bounds[fold_id]
            train_end = max(0, test_start - purge)
            if train_end > 0:
                train_parts.append(group.iloc[:train_end].copy())
            test_parts.append(group.iloc[test_start:test_end].copy())
        if not test_parts:
            continue
        folds.append(
            StabilityFold(
                fold_id=fold_id,
                train_df=(
                    pd.concat(train_parts, ignore_index=True)
                    if train_parts
                    else pd.DataFrame(columns=frame.columns)
                ),
                test_df=pd.concat(test_parts, ignore_index=True),
                purge_candles=purge,
            )
        )
    return folds


def _metric_summary(metrics: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(metrics)
    if not rows:
        return {
            "folds": 0,
            "median_return_pct": 0.0,
            "worst_return_pct": 0.0,
            "median_profit_factor": 0.0,
            "worst_drawdown_pct": 0.0,
            "min_trades": 0,
            "metrics": [],
        }
    returns = [float(row.get("total_return_pct", 0.0)) for row in rows]
    profit_factors = [float(row.get("profit_factor", 0.0)) for row in rows]
    drawdowns = [float(row.get("max_drawdown_pct", 0.0)) for row in rows]
    trades = [int(row.get("executed_trades", 0)) for row in rows]
    return {
        "folds": len(rows),
        "median_return_pct": float(np.median(returns)),
        "worst_return_pct": float(np.min(returns)),
        "median_profit_factor": float(np.median(profit_factors)),
        "worst_drawdown_pct": float(np.max(drawdowns)),
        "min_trades": int(np.min(trades)),
        "metrics": rows,
    }


def evaluate_strategy_stability(
    frame: pd.DataFrame,
    strategy: dict[str, Any],
    *,
    n_windows: int = 3,
    evaluator: Callable[[pd.DataFrame, list[dict]], dict] | None = None,
) -> dict[str, Any]:
    """Evaluate one immutable strategy across chronological stability windows.

    The strategy is not re-selected and no parameters are fitted in this
    report.  The purged training prefixes are recorded only to make each
    comparison window auditable.
    """
    folds = build_stability_folds(frame, n_windows=n_windows)
    rule_set = list(strategy.get("rules_set", []))
    if evaluator is None:

        def evaluator(stability_frame: pd.DataFrame, rules: list[dict]) -> dict:
            direction = str(strategy.get("direction", "long"))
            return CPUBacktestEngine(
                stability_frame, {}, direction,
            ).simulate_rule_set(rules)

    metrics = [evaluator(fold.test_df, rule_set) for fold in folds]
    summary = _metric_summary(metrics)
    summary.update({
        "direction": strategy.get("direction"),
        "strategy_id": strategy.get("strategy_id"),
        "purge_candles": int(
            folds[0].purge_candles
            if folds
            else getattr(_cfg, "MAX_HOLD_CANDLES", 0)
        ),
        "stability_windows": [
            {
                "fold_id": fold.fold_id,
                "train_rows": int(len(fold.train_df)),
                "test_rows": int(len(fold.test_df)),
            }
            for fold in folds
        ],
        "stability_contract": "frozen_strategy_chronological_comparison",
        "multiplicity": summarize_multiplicity(
            fold_returns=[
                float(row.get("total_return_pct", 0.0))
                for row in metrics
            ],
            n_trials=int(strategy.get("trial_count", max(1, len(rule_set)))),
        ),
    })
    return summary


def write_strategy_stability_reports(
    output_dir: str,
    strategies: dict[str, dict[str, Any]],
    frame: pd.DataFrame,
    *,
    n_windows: int = 3,
) -> dict[str, dict[str, Any]]:
    """Write per-direction frozen-strategy stability reports."""
    import json
    from pathlib import Path

    reports_dir = Path(output_dir) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    for direction, strategy in strategies.items():
        if direction not in {"long", "short"} or not strategy.get("rules_set"):
            continue
        report = evaluate_strategy_stability(
            frame,
            strategy,
            n_windows=n_windows,
        )
        results[direction] = report
        (reports_dir / f"strategy_stability_{direction}.json").write_text(
            json.dumps(report, indent=2, default=str),
            encoding="utf-8",
        )
    return results


__all__ = [
    "StabilityFold",
    "build_stability_folds",
    "evaluate_strategy_stability",
    "write_strategy_stability_reports",
]
