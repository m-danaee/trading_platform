"""Lightweight nested, purged walk-forward evaluation.

The expensive Phase 2 search belongs in inner folds.  This module supplies the
fold contract and an outer comparison report so the canonical pipeline can
evaluate immutable strategy packages without ever consulting ``test_new.csv``.
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
class NestedFold:
    """One outer validation fold and its inner tuning prefix."""

    fold_id: int
    inner_train_df: pd.DataFrame
    outer_valid_df: pd.DataFrame
    purge_candles: int


def _sorted_symbol_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "symbol" not in frame.columns:
        return frame.sort_values("datetime").reset_index(drop=True)
    order = ["symbol", "datetime"] if "datetime" in frame.columns else ["symbol"]
    return frame.sort_values(order).reset_index(drop=True)


def _split_bounds(
    frame: pd.DataFrame,
    *,
    n_outer: int,
    min_train_fraction: float,
) -> list[tuple[int, int]]:
    n = len(frame)
    start = max(1, int(n * min_train_fraction))
    usable = max(0, n - start)
    if usable == 0:
        return []
    base, remainder = divmod(usable, max(1, n_outer))
    bounds: list[tuple[int, int]] = []
    cursor = start
    for idx in range(max(1, n_outer)):
        width = base + (1 if idx < remainder else 0)
        end = min(n, cursor + width)
        if end > cursor:
            bounds.append((cursor, end))
        cursor = end
    return bounds


def build_nested_folds(
    frame: pd.DataFrame,
    *,
    n_outer: int = 3,
    min_train_fraction: float = 0.40,
    purge_candles: int | None = None,
) -> list[NestedFold]:
    """Build chronological outer folds with per-symbol label purging."""
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
            n_outer=max(1, int(n_outer)),
            min_train_fraction=float(min_train_fraction),
        )
        for symbol, group in groups.items()
    }
    n_folds = max((len(bounds) for bounds in group_bounds.values()), default=0)
    folds: list[NestedFold] = []
    for fold_id in range(n_folds):
        train_parts: list[pd.DataFrame] = []
        valid_parts: list[pd.DataFrame] = []
        for symbol, group in groups.items():
            bounds = group_bounds[symbol]
            if fold_id >= len(bounds):
                continue
            valid_start, valid_end = bounds[fold_id]
            train_end = max(0, valid_start - purge)
            if train_end > 0:
                train_parts.append(group.iloc[:train_end].copy())
            valid_parts.append(group.iloc[valid_start:valid_end].copy())
        if not valid_parts:
            continue
        folds.append(NestedFold(
            fold_id=fold_id,
            inner_train_df=pd.concat(train_parts, ignore_index=True)
            if train_parts else pd.DataFrame(columns=frame.columns),
            outer_valid_df=pd.concat(valid_parts, ignore_index=True),
            purge_candles=purge,
        ))
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
    pfs = [float(row.get("profit_factor", 0.0)) for row in rows]
    dds = [float(row.get("max_drawdown_pct", 0.0)) for row in rows]
    trades = [int(row.get("executed_trades", 0)) for row in rows]
    return {
        "folds": len(rows),
        "median_return_pct": float(np.median(returns)),
        "worst_return_pct": float(np.min(returns)),
        "median_profit_factor": float(np.median(pfs)),
        "worst_drawdown_pct": float(np.max(dds)),
        "min_trades": int(np.min(trades)),
        "metrics": rows,
    }


def evaluate_nested_strategy(
    frame: pd.DataFrame,
    strategy: dict[str, Any],
    *,
    n_outer: int = 3,
    evaluator: Callable[[pd.DataFrame, list[dict]], dict] | None = None,
) -> dict[str, Any]:
    """Evaluate one immutable strategy on outer folds and report inner sizes."""
    folds = build_nested_folds(frame, n_outer=n_outer)
    rule_set = list(strategy.get("rules_set", []))
    if evaluator is None:
        def evaluator(inner_or_outer: pd.DataFrame, rules: list[dict]) -> dict:
            direction = str(strategy.get("direction", "long"))
            return CPUBacktestEngine(
                inner_or_outer, {}, direction,
            ).simulate_rule_set(rules)
    outer_metrics = [
        evaluator(fold.outer_valid_df, rule_set)
        for fold in folds
    ]
    summary = _metric_summary(outer_metrics)
    summary.update({
        "direction": strategy.get("direction"),
        "strategy_id": strategy.get("strategy_id"),
        "purge_candles": int(
            folds[0].purge_candles if folds else getattr(
                _cfg, "MAX_HOLD_CANDLES", 0
            )
        ),
        "inner_folds": [
            {
                "fold_id": fold.fold_id,
                "inner_train_rows": int(len(fold.inner_train_df)),
                "outer_valid_rows": int(len(fold.outer_valid_df)),
            }
            for fold in folds
        ],
        "nested_contract": "inner_tuning_outer_comparison",
        "multiplicity": summarize_multiplicity(
            fold_returns=[
                float(row.get("total_return_pct", 0.0))
                for row in outer_metrics
            ],
            n_trials=int(strategy.get("trial_count", max(1, len(rule_set)))),
        ),
    })
    return summary


def nested_select_candidates(
    frame: pd.DataFrame,
    candidates: list[dict[str, Any]],
    *,
    n_outer: int = 3,
    score_key: str = "total_return_pct",
) -> dict[str, Any]:
    """Select candidates on each inner prefix, then score on outer folds.

    This is the reusable nested contract for future full Phase 2 inner-fold
    runs. It is intentionally generic so a caller can pass immutable strategy
    packages or baseline candidates without coupling the validator to RB.
    """
    folds = build_nested_folds(frame, n_outer=n_outer)
    selected_ids: list[Any] = []
    outer_metrics: list[dict] = []
    for fold in folds:
        if not candidates:
            continue
        inner_scores = []
        for candidate in candidates:
            direction = str(candidate.get("direction", "long"))
            rules = list(candidate.get("rules_set", []))
            inner_metrics = CPUBacktestEngine(
                fold.inner_train_df, {}, direction,
            ).simulate_rule_set(rules)
            inner_scores.append(float(inner_metrics.get(score_key, 0.0)))
        winner_index = int(np.argmax(np.asarray(inner_scores, dtype=float)))
        winner = candidates[winner_index]
        direction = str(winner.get("direction", "long"))
        outer_metrics.append(CPUBacktestEngine(
            fold.outer_valid_df, {}, direction,
        ).simulate_rule_set(list(winner.get("rules_set", []))))
        selected_ids.append(winner.get("strategy_id", winner_index))
    report = _metric_summary(outer_metrics)
    report.update({
        "candidate_count": len(candidates),
        "selected_strategy_ids": selected_ids,
        "inner_selection_metric": score_key,
        "nested_contract": "inner_candidate_selection_outer_comparison",
    })
    return report


def write_nested_reports(
    output_dir: str,
    strategies: dict[str, dict[str, Any]],
    frame: pd.DataFrame,
    *,
    n_outer: int = 3,
) -> dict[str, dict[str, Any]]:
    """Write per-direction nested reports for a pipeline run."""
    import json
    from pathlib import Path

    reports_dir = Path(output_dir) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    for direction, strategy in strategies.items():
        if direction not in {"long", "short"} or not strategy.get("rules_set"):
            continue
        report = evaluate_nested_strategy(
            frame,
            strategy,
            n_outer=n_outer,
        )
        results[direction] = report
        (reports_dir / f"nested_validation_{direction}.json").write_text(
            json.dumps(report, indent=2, default=str),
            encoding="utf-8",
        )
    return results
