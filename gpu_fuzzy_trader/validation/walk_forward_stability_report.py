"""Chronological strategy-stability diagnostics.

The production pipeline has one adaptive master-fold system.  This module is a
diagnostic only: it evaluates an already-frozen strategy on chronological
stability windows and never performs candidate search or threshold fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.backtest.cpu_engine import CPUBacktestEngine
from gpu_fuzzy_trader.validation.multiplicity import (
    aggregate_seed_metrics,
    read_candidate_fold_matrix,
    summarize_multiplicity,
)
from gpu_fuzzy_trader.validation.uncertainty import compute_trade_uncertainty


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


def _history_before_stability_window(
    frame: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return raw rows strictly before each test symbol's first timestamp."""
    if frame.empty or test_df.empty:
        return frame.iloc[0:0].copy()
    if "datetime" not in frame.columns or "datetime" not in test_df.columns:
        return frame.iloc[0:0].copy()

    frame_times = pd.to_datetime(frame["datetime"], errors="raise", utc=True)
    test_times = pd.to_datetime(test_df["datetime"], errors="raise", utc=True)
    if "symbol" not in frame.columns or "symbol" not in test_df.columns:
        return _sorted_symbol_frame(
            frame.loc[frame_times < test_times.min()].copy()
        )

    test_with_times = pd.DataFrame({
        "symbol": test_df["symbol"].to_numpy(),
        "_test_start": test_times.to_numpy(),
    })
    starts = test_with_times.groupby(
        "symbol", sort=False, observed=False,
    )["_test_start"].min()
    cutoffs = frame["symbol"].map(starts)
    history_mask = cutoffs.notna().to_numpy() & (
        frame_times.to_numpy() < cutoffs.to_numpy()
    )
    return _sorted_symbol_frame(frame.loc[history_mask].copy())


def _raw_mtf_stability_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove precomputed MTF features before the frozen runtime rebuilds them."""
    derived_prefixes = ("lwc_", "mwc_", "hwc_", "mtf_", "_mtf")
    columns = [
        column
        for column in frame.columns
        if not str(column).startswith(derived_prefixes)
    ]
    return frame.loc[:, columns].copy()


def _evaluate_mtf_strategy_stability(
    frame: pd.DataFrame,
    strategy: dict[str, Any],
    folds: list[StabilityFold],
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    """Evaluate a frozen MTF candidate with causal per-window history."""
    candidate_payload = strategy.get("mtf_candidate")
    if not isinstance(candidate_payload, dict):
        raise ValueError("MTF stability evaluation requires mtf_candidate")

    # Keep these imports local.  The backtest package is imported during Phase
    # 2 collection, and a module-level MTF runtime import recreates that cycle.
    from gpu_fuzzy_trader.mtf.candidate import HierarchicalStrategyCandidate
    from gpu_fuzzy_trader.mtf.runtime import evaluate_candidate_rule_masks

    candidate = HierarchicalStrategyCandidate.from_dict(candidate_payload)
    direction = str(strategy.get("direction", candidate.direction))
    raw_frame = _raw_mtf_stability_frame(frame)
    metrics: list[dict[str, Any]] = []
    trade_logs: list[pd.DataFrame] = []
    for fold in folds:
        raw_test_df = _raw_mtf_stability_frame(fold.test_df)
        history_df = _history_before_stability_window(raw_frame, raw_test_df)
        rule_masks, _stats, _audit = evaluate_candidate_rule_masks(
            candidate,
            raw_test_df,
            history_df=history_df,
        )
        fold_metrics, fold_log = CPUBacktestEngine(
            fold.test_df, {}, direction,
        ).simulate_signal_masks(
            rule_masks,
            candidate.lwc_rules,
            return_logs=True,
        )
        metrics.append(fold_metrics)
        trade_logs.append(fold_log)
    return metrics, trade_logs


def _candidate_id(candidate: dict[str, Any], *, source: str, direction: str) -> str:
    """Return a stable ID for a candidate that has no explicit ID."""
    explicit = candidate.get("candidate_id") or candidate.get("strategy_id")
    if explicit:
        return str(explicit)
    payload = json.dumps(candidate, sort_keys=True, default=str, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{source}:{direction}:{digest}"


def _candidate_rows(candidate: Any) -> list[dict[str, Any]]:
    """Return candidate records from either a mapping or a sequence."""
    if isinstance(candidate, dict):
        return [candidate]
    if isinstance(candidate, (list, tuple)):
        return [value for value in candidate if isinstance(value, dict)]
    return []


def _precomputed_candidate_fold_rows(
    candidate: dict[str, Any],
    *,
    candidate_id: str,
    source: str,
    direction: str,
) -> list[dict[str, Any]]:
    """Read fold scores already attached by a search implementation."""
    values = candidate.get("fold_scores", candidate.get("candidate_fold_scores"))
    if values is None:
        is_values = candidate.get("in_sample_scores", candidate.get("is_scores"))
        oos_values = candidate.get("out_of_sample_scores", candidate.get("oos_scores"))
        if is_values is None or oos_values is None:
            return []
        values = [
            {
                "fold_id": index,
                "is_score": is_value,
                "oos_score": oos_value,
            }
            for index, (is_value, oos_value) in enumerate(
                zip(is_values, oos_values, strict=False)
            )
        ]
    if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
        return []
    rows: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            continue
        is_score = value.get("is_score", value.get("in_sample_score"))
        oos_score = value.get("oos_score", value.get("out_of_sample_score"))
        try:
            is_score = float(is_score)
            oos_score = float(oos_score)
        except (TypeError, ValueError):
            continue
        fold_id = value.get("fold_id", index)
        try:
            fold_id = int(fold_id)
        except (TypeError, ValueError):
            fold_id = str(fold_id)
        rows.append({
            "candidate_id": candidate_id,
            "fold_id": fold_id,
            "is_score": is_score,
            "oos_score": oos_score,
            "source": source,
            "direction": direction,
        })
    return rows


def build_candidate_fold_matrix(
    frame: pd.DataFrame | None,
    candidates: Any,
    *,
    n_windows: int = 3,
) -> list[dict[str, Any]]:
    """Evaluate candidates on chronological IS/OOS stability folds.

    The function records one return score for every candidate and fold.  It
    also accepts precomputed ``fold_scores`` so callers can use the exact
    evaluator scores when a search stage already produced them.  Candidate
    IDs and fold IDs are sorted by the writer, not by evaluation completion.
    """
    entries: list[dict[str, Any]] = []
    if isinstance(candidates, dict):
        for source_key in sorted(candidates):
            source_value = candidates[source_key]
            source = str(source_key)
            if source_value is None:
                continue
            if isinstance(source_value, dict) and (
                "rules_set" in source_value or "conditions" in source_value
            ):
                source_values = [source_value]
            elif isinstance(source_value, dict):
                source_values = [
                    dict(value, direction=direction)
                    for direction, value in sorted(source_value.items())
                    if isinstance(value, dict)
                ]
            else:
                source_values = _candidate_rows(source_value)
            for value in source_values:
                entry = dict(value)
                entry.setdefault("source", source)
                entries.append(entry)
    else:
        entries = [dict(value) for value in _candidate_rows(candidates)]

    rows: list[dict[str, Any]] = []
    fold_list = build_stability_folds(
        frame,
        n_windows=max(1, int(n_windows)),
    ) if isinstance(frame, pd.DataFrame) and not frame.empty else []
    for candidate in entries:
        source = str(candidate.get("source", "phase2"))
        direction = str(candidate.get("direction", "long")).lower()
        candidate_id = _candidate_id(
            candidate,
            source=source,
            direction=direction,
        )
        precomputed = _precomputed_candidate_fold_rows(
            candidate,
            candidate_id=candidate_id,
            source=source,
            direction=direction,
        )
        if precomputed:
            rows.extend(precomputed)
            continue
        if not fold_list:
            continue
        rules = candidate.get("rules_set")
        if not isinstance(rules, list):
            rules = [candidate] if candidate.get("conditions") else []
        if not rules:
            continue
        for fold in fold_list:
            try:
                is_metrics = CPUBacktestEngine(
                    fold.train_df, {}, direction,
                ).simulate_rule_set(rules)
                oos_metrics = CPUBacktestEngine(
                    fold.test_df, {}, direction,
                ).simulate_rule_set(rules)
                is_score = float(is_metrics.get("total_return_pct", 0.0))
                oos_score = float(oos_metrics.get("total_return_pct", 0.0))
            except Exception:
                # A malformed legacy candidate must not stop the report for
                # every other candidate.  The persisted phase metrics remain
                # available through the fallback ledger path.
                continue
            rows.append({
                "candidate_id": candidate_id,
                "fold_id": int(fold.fold_id),
                "is_score": is_score,
                "oos_score": oos_score,
                "source": source,
                "direction": direction,
            })
    return rows


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
    mtf_strategy = isinstance(strategy.get("mtf_candidate"), dict)
    trade_logs: list[pd.DataFrame | None] = []
    if evaluator is None and mtf_strategy:
        metrics, mtf_trade_logs = _evaluate_mtf_strategy_stability(
            frame, strategy, folds,
        )
        trade_logs = list(mtf_trade_logs)
        evaluator_contract = "mtf_candidate_runtime"
        history_contract = "strictly_prior_per_symbol_to_test_window"
    else:
        evaluator_contract = "custom" if evaluator is not None else "cpu_rule_set"
        history_contract = "not_required"

        if evaluator is None:

            def evaluator(stability_frame: pd.DataFrame, rules: list[dict]) -> dict:
                direction = str(strategy.get("direction", "long"))
                return CPUBacktestEngine(
                    stability_frame, {}, direction,
                ).simulate_rule_set(rules, return_logs=True)

        metrics = []
        for fold in folds:
            evaluated = evaluator(fold.test_df, rule_set)
            if (
                isinstance(evaluated, tuple)
                and len(evaluated) == 2
                and isinstance(evaluated[0], dict)
            ):
                metric_row, fold_log = evaluated
                metrics.append(metric_row)
                trade_logs.append(
                    fold_log if isinstance(fold_log, pd.DataFrame) else None
                )
            else:
                metrics.append(evaluated)
                trade_logs.append(None)

    # Add uncertainty after evaluation only.  These rows are descriptive and
    # are not fed back into candidate or risk selection.
    audited_metrics: list[dict[str, Any]] = []
    for fold_index, metric_row in enumerate(metrics):
        audited = dict(metric_row)
        if fold_index < len(trade_logs) and trade_logs[fold_index] is not None:
            audited["trade_uncertainty"] = compute_trade_uncertainty(
                trade_logs[fold_index],
                seed=int(getattr(_cfg, "GLOBAL_SEED", 42) or 42) + fold_index,
            )
        audited_metrics.append(audited)
    metrics = audited_metrics
    summary = _metric_summary(metrics)
    matrix = strategy.get("candidate_fold_matrix")
    if isinstance(matrix, (str, bytes)):
        matrix = read_candidate_fold_matrix(matrix)
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
        "evaluator_contract": evaluator_contract,
        "historical_context": history_contract,
        "uncertainty_contract": (
            "descriptive_moving_block_bootstrap_on_realized_trades; "
            "not_used_for_selection"
        ),
        "multiplicity": summarize_multiplicity(
            fold_returns=[
                float(row.get("total_return_pct", 0.0))
                for row in metrics
            ],
            n_trials=int(strategy.get("trial_count", max(1, len(rule_set)))),
            matrix=matrix,
            trial_count_ledger=strategy.get("trial_count_ledger"),
        ),
    })
    return summary


def write_strategy_stability_reports(
    output_dir: str,
    strategies: dict[str, dict[str, Any]],
    frame: pd.DataFrame,
    *,
    n_windows: int = 3,
    candidate_fold_matrix: Any | None = None,
    trial_count_ledger: int | None = None,
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
        report_strategy = dict(strategy)
        if candidate_fold_matrix is not None:
            report_strategy["candidate_fold_matrix"] = candidate_fold_matrix
        if trial_count_ledger is not None:
            report_strategy["trial_count_ledger"] = int(trial_count_ledger)
        report = evaluate_strategy_stability(
            frame,
            report_strategy,
            n_windows=n_windows,
        )
        results[direction] = report
        (reports_dir / f"strategy_stability_{direction}.json").write_text(
            json.dumps(report, indent=2, default=str),
            encoding="utf-8",
        )
    return results


def write_golden_baseline_report(
    output_dir: str,
    seed_records: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Write a multi-seed golden-baseline aggregation report."""
    from pathlib import Path

    report = aggregate_seed_metrics(seed_records)
    configured_path = getattr(
        _cfg,
        "GOLDEN_BASELINE_REPORT_PATH",
        Path(output_dir) / "reports" / "golden_baseline.json",
    )
    if str(configured_path).endswith(
        "outputs/reports/golden_baseline.json"
    ):
        configured_path = Path(output_dir) / "reports" / "golden_baseline.json"
    path = Path(configured_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


# Keep the baseline helper discoverable from the stability-report module, which
# is the report entry point used by the pipeline.
aggregate_golden_baseline = aggregate_seed_metrics
aggregate_golden_baselines = aggregate_seed_metrics


__all__ = [
    "StabilityFold",
    "aggregate_golden_baseline",
    "aggregate_golden_baselines",
    "aggregate_seed_metrics",
    "build_candidate_fold_matrix",
    "build_stability_folds",
    "evaluate_strategy_stability",
    "write_golden_baseline_report",
    "write_strategy_stability_reports",
]
