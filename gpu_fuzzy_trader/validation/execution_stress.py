"""Diagnostic execution-latency stress for frozen rule portfolios."""

from __future__ import annotations

from collections.abc import Sequence
import logging
import math
from typing import Any

import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.backtest.cpu_engine import (
    CPUBacktestEngine,
    _build_entries_from_rule_set,
    _sort_entries_by_allocation_priority,
)


logger = logging.getLogger(__name__)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _engine_like(value: Any) -> bool:
    return isinstance(value, pd.DataFrame) or callable(
        getattr(value, "simulate_rule_set", None)
    )


def _frame_for(source: Any) -> pd.DataFrame | None:
    if isinstance(source, pd.DataFrame):
        return source
    frame = getattr(source, "df", None)
    return frame if isinstance(frame, pd.DataFrame) else None


def _coerce_engine(source: Any, direction: str | None) -> Any:
    if isinstance(source, pd.DataFrame):
        return CPUBacktestEngine(source, {}, direction or "long")
    return source


def next_row_indices(
    df: pd.DataFrame,
    delay_bars: int = 1,
) -> np.ndarray:
    """Map each row to the delayed row in the same symbol.

    ``-1`` means that the delayed entry is outside the available frame.  The
    mapping follows ``_symbol_bar_index`` when present and otherwise follows
    timestamp/input order.  It never crosses a symbol boundary.
    """
    delay = int(delay_bars)
    if delay < 0:
        raise ValueError("delay_bars must be non-negative")
    n_rows = len(df)
    result = np.full(n_rows, -1, dtype=np.int64)
    if delay == 0 or n_rows == 0:
        return np.arange(n_rows, dtype=np.int64)

    if "symbol" in df.columns:
        symbols = df["symbol"].astype(str).to_numpy()
    else:
        symbols = np.full(n_rows, "__single_symbol__", dtype=object)

    if "_symbol_bar_index" in df.columns:
        bar_index = pd.to_numeric(
            df["_symbol_bar_index"], errors="coerce"
        ).to_numpy(dtype=float)
    else:
        bar_index = np.arange(n_rows, dtype=float)
    if "datetime" in df.columns:
        timestamps = pd.to_datetime(df["datetime"], errors="coerce")
    else:
        timestamps = pd.Series(np.arange(n_rows), index=df.index)

    for symbol in pd.unique(symbols):
        positions = np.flatnonzero(symbols == symbol)
        if len(positions) <= delay:
            continue
        # Stable sorting retains input order for ties and handles source tapes
        # which were concatenated by symbol rather than globally sorted.
        order_frame = pd.DataFrame(
            {
                "position": positions,
                "bar": bar_index[positions],
                "timestamp": timestamps.iloc[positions].to_numpy(),
            }
        )
        order_frame = order_frame.sort_values(
            ["bar", "timestamp", "position"],
            kind="stable",
            na_position="last",
        )
        ordered = order_frame["position"].to_numpy(dtype=np.int64)
        result[ordered[:-delay]] = ordered[delay:]
    return result


def shift_signal_mask(
    df: pd.DataFrame,
    signal_mask: np.ndarray | pd.Series,
    delay_bars: int = 1,
) -> np.ndarray:
    """Shift a signal to a later same-symbol row without look-ahead."""
    mask = np.asarray(signal_mask, dtype=bool)
    if len(mask) != len(df):
        raise ValueError("signal_mask length does not match dataframe")
    shifted = np.zeros(len(df), dtype=bool)
    mapping = next_row_indices(df, delay_bars)
    source_rows = np.flatnonzero(mask)
    target_rows = mapping[source_rows]
    valid = target_rows >= 0
    if np.any(valid):
        shifted[target_rows[valid]] = True
    return shifted


def _delayed_entries(
    engine: Any,
    rules: Sequence[dict],
    delay_bars: int,
) -> list[dict] | None:
    """Build delayed entries while retaining per-rule TP/SL and sizing."""
    frame = _frame_for(engine)
    if frame is None:
        return None
    if not hasattr(engine, "_condition_mask_cache"):
        return None
    entries = _build_entries_from_rule_set(
        frame,
        [dict(rule) for rule in rules],
        getattr(engine, "_condition_mask_cache", None),
        row_priority=getattr(engine, "entry_time_priority", None),
        context_mask=getattr(engine, "_context_mask", None),
    )
    mapping = next_row_indices(frame, delay_bars)
    shifted: list[dict] = []
    for entry in entries:
        source_idx = int(entry["idx"])
        if source_idx < 0 or source_idx >= len(mapping):
            continue
        target_idx = int(mapping[source_idx])
        if target_idx < 0:
            continue
        delayed = dict(entry)
        delayed["idx"] = target_idx
        priorities = getattr(engine, "entry_time_priority", None)
        delayed["entry_priority"] = (
            int(priorities[target_idx])
            if priorities is not None and target_idx < len(priorities)
            else target_idx
        )
        delayed["_source_idx"] = source_idx
        shifted.append(delayed)

    # Two adjacent signals may target the same row.  Keep the first signal in
    # the original timestamp/rule order, matching the frozen allocation rule.
    shifted.sort(
        key=lambda item: (
            int(item.get("entry_priority", item["idx"])),
            int(item.get("_source_idx", item["idx"])),
            int(item.get("rule_index", 0)),
            int(item["idx"]),
        )
    )
    deduplicated: list[dict] = []
    seen_rows: set[int] = set()
    for entry in shifted:
        idx = int(entry["idx"])
        if idx in seen_rows:
            continue
        seen_rows.add(idx)
        entry.pop("_source_idx", None)
        deduplicated.append(entry)
    _sort_entries_by_allocation_priority(deduplicated)
    return deduplicated


def simulate_delayed_entries(
    engine_or_frame: CPUBacktestEngine | pd.DataFrame | Any,
    rules: Sequence[dict],
    *,
    direction: str | None = None,
    delay_bars: int = 1,
    return_logs: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], pd.DataFrame]:
    """Simulate entries one bar later than the original signal.

    The original rule set is never rewritten.  For a real CPU engine only the
    entry row is shifted; the delayed row's own forward labels provide the
    entry price and barrier outcome.  This is diagnostic-only behaviour.
    """
    if int(delay_bars) < 0:
        raise ValueError("delay_bars must be non-negative")
    engine = _coerce_engine(engine_or_frame, direction)
    delayed = _delayed_entries(engine, rules, int(delay_bars))
    if delayed is None:
        simulate = getattr(engine, "simulate_rule_set", None)
        if not callable(simulate):
            empty = {"sortino_ratio": 0.0, "executed_trades": 0}
            return (empty, pd.DataFrame()) if return_logs else empty
        result = simulate([dict(rule) for rule in rules])
        return result
    return engine._simulate_rule_set_entries(  # type: ignore[attr-defined]
        delayed,
        return_logs=return_logs,
        initial_capital=float(getattr(engine, "initial_capital", _cfg.INITIAL_CAPITAL)),
    )


def simulate_delayed_signal(
    engine_or_frame: CPUBacktestEngine | pd.DataFrame | Any,
    signal_mask: np.ndarray | pd.Series,
    *,
    tp: float,
    sl: float,
    capital_pct: float,
    direction: str | None = None,
    delay_bars: int = 1,
    return_logs: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], pd.DataFrame]:
    """Delayed counterpart for a pre-composed signal mask."""
    engine = _coerce_engine(engine_or_frame, direction)
    frame = _frame_for(engine)
    if frame is None or not callable(getattr(engine, "simulate_signal_mask", None)):
        raise TypeError("simulate_delayed_signal requires a CPU-style engine")
    shifted = shift_signal_mask(frame, signal_mask, delay_bars)
    return engine.simulate_signal_mask(
        shifted,
        tp=tp,
        sl=sl,
        capital_pct=capital_pct,
        return_logs=return_logs,
    )


def _snapshot(metrics: Any) -> dict[str, Any]:
    values = metrics if isinstance(metrics, dict) else {}
    sortino = _finite_float(values.get("sortino_ratio", values.get("sortino")))
    total_return = _finite_float(
        values.get("total_return_pct", values.get("return_pct", values.get("Return")))
    )
    profit_factor = _finite_float(
        values.get("profit_factor", values.get("pf", values.get("PF")))
    )
    drawdown = _finite_float(
        values.get("max_drawdown_pct", values.get("mdd_pct", values.get("MDD")))
    )
    return {
        "sortino_ratio": sortino,
        "sortino": sortino,
        "total_return_pct": total_return,
        "return_pct": total_return,
        "profit_factor": profit_factor,
        "max_drawdown_pct": drawdown,
        "executed_trades": int(_finite_float(values.get("executed_trades"))),
        "available": bool(values),
    }


def _stress_one(
    engine: Any,
    rules: Sequence[dict],
    delay_bars: int,
) -> dict[str, Any]:
    simulate = getattr(engine, "simulate_rule_set", None)
    if not callable(simulate) or not rules:
        return {
            "normal": _snapshot({}),
            "delayed": _snapshot({}),
            "sortino_delta": 0.0,
            "available": False,
        }
    try:
        normal = simulate([dict(rule) for rule in rules])
        delayed = simulate_delayed_entries(
            engine,
            rules,
            delay_bars=delay_bars,
        )
    except Exception as exc:  # pragma: no cover - defensive report path
        logger.warning("Execution stress simulation failed: %s", exc)
        return {
            "normal": _snapshot({}),
            "delayed": _snapshot({}),
            "sortino_delta": 0.0,
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    normal_row = _snapshot(normal)
    delayed_row = _snapshot(delayed)
    return {
        "normal": normal_row,
        "delayed": delayed_row,
        "normal_sortino": normal_row["sortino_ratio"],
        "delayed_sortino": delayed_row["sortino_ratio"],
        "sortino_delta": float(
            delayed_row["sortino_ratio"] - normal_row["sortino_ratio"]
        ),
        "available": bool(normal_row["available"] or delayed_row["available"]),
    }


def execution_stress_certificate(
    engine_or_train: CPUBacktestEngine | pd.DataFrame | Any,
    rules_or_validation: Sequence[dict] | CPUBacktestEngine | pd.DataFrame | Any,
    validation_engine: CPUBacktestEngine | pd.DataFrame | Any | Sequence[dict] | None = None,
    *,
    direction: str | None = None,
    delay_bars: int = 1,
    enabled: bool | None = None,
) -> dict[str, Any]:
    """Build the one-bar delayed-entry certificate.

    Supported forms are ``(engine, rules)`` and
    ``(train_engine, validation_engine, rules)``.  In the two-split form the
    validation result is also promoted to the headline fields.
    """
    # Accept the natural three-positional-argument form without making callers
    # know the keyword name for the validation engine.
    rules: Sequence[dict]
    validation: Any | None
    if _engine_like(rules_or_validation) and isinstance(validation_engine, Sequence) and not isinstance(validation_engine, (str, bytes)):
        validation = rules_or_validation
        rules = validation_engine
    else:
        validation = validation_engine
        rules = rules_or_validation  # type: ignore[assignment]
    rules_value = list(rules)
    enabled_value = bool(
        getattr(_cfg, "RB_EXECUTION_STRESS_ENABLED", True)
        if enabled is None
        else enabled
    )
    report_only = bool(
        getattr(_cfg, "RB_ROBUSTNESS_REPORT_ONLY", True)
        or getattr(_cfg, "RB_EXECUTION_STRESS_REPORT_ONLY", True)
    )
    if not enabled_value:
        return {
            "schema_version": "1.0",
            "certificate": "execution_stress",
            "enabled": False,
            "available": False,
            "diagnostic_only": True,
            "report_only": report_only,
            "delay_bars": int(delay_bars),
            "verdict": "disabled",
        }

    train_engine = _coerce_engine(engine_or_train, direction)
    split_results = {
        "train": _stress_one(train_engine, rules_value, int(delay_bars))
    }
    if validation is not None:
        split_results["validation"] = _stress_one(
            _coerce_engine(validation, direction), rules_value, int(delay_bars)
        )
    headline_key = "validation" if "validation" in split_results else "train"
    headline = split_results[headline_key]
    deltas = [
        float(row.get("sortino_delta", 0.0))
        for row in split_results.values()
        if row.get("available")
    ]
    available = any(bool(row.get("available")) for row in split_results.values())
    return {
        "schema_version": "1.0",
        "certificate": "execution_stress",
        "enabled": True,
        "available": available,
        "diagnostic_only": True,
        "report_only": report_only,
        "delay_bars": int(delay_bars),
        "splits": split_results,
        "normal": headline.get("normal", {}),
        "delayed": headline.get("delayed", {}),
        "normal_sortino": float(headline.get("normal_sortino", 0.0)),
        "delayed_sortino": float(headline.get("delayed_sortino", 0.0)),
        "sortino_delta": float(headline.get("sortino_delta", 0.0)),
        "worst_sortino_delta": float(min(deltas)) if deltas else 0.0,
        "verdict": (
            "unavailable"
            if not available
            else "degraded"
            if any(delta < -1e-12 for delta in deltas)
            else "stable"
        ),
    }


run_execution_stress = execution_stress_certificate
build_execution_stress_certificate = execution_stress_certificate
delayed_entry_stress = execution_stress_certificate
evaluate_execution_stress = execution_stress_certificate
simulate_delayed_entry = simulate_delayed_entries


__all__ = [
    "build_execution_stress_certificate",
    "delayed_entry_stress",
    "execution_stress_certificate",
    "evaluate_execution_stress",
    "next_row_indices",
    "run_execution_stress",
    "shift_signal_mask",
    "simulate_delayed_entry",
    "simulate_delayed_entries",
    "simulate_delayed_signal",
]
