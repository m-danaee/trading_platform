"""Regime and rule-dropout robustness certificates.

These checks are deliberately descriptive.  They split an already frozen
portfolio into observable market states and leave one rule out at a time.  No
result from this module changes selection or deployment acceptance.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.backtest.cpu_engine import CPUBacktestEngine


logger = logging.getLogger(__name__)

_REGIMES = ("high_vol", "low_vol", "trend", "range")
_METRIC_KEYS = (
    "total_return_pct",
    "profit_factor",
    "max_drawdown_pct",
    "sortino_ratio",
    "executed_trades",
)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _find_column(df: pd.DataFrame, names: Sequence[str]) -> str | None:
    lowered = {str(column).lower(): column for column in df.columns}
    for name in names:
        if name.lower() in lowered:
            return str(lowered[name.lower()])
    return None


def _price_series(df: pd.DataFrame) -> pd.Series:
    column = _find_column(
        df,
        ("close", "close_price", "adj_close", "price", "label_open_next"),
    )
    if column is None:
        column = _find_column(df, ("open", "open_price"))
    if column is None:
        return pd.Series(np.arange(len(df), dtype=float), index=df.index)
    values = pd.to_numeric(df[column], errors="coerce")
    return values.astype(float)


def _volatility_series(df: pd.DataFrame, window: int) -> pd.Series:
    """Return a causal volatility proxy from ATR or realised returns."""
    explicit = [
        column
        for column in df.columns
        if any(token in str(column).lower() for token in ("atr", "realized_vol", "volatility"))
    ]
    if explicit:
        values = pd.to_numeric(df[explicit[0]], errors="coerce")
        return values.abs().astype(float)

    close = _price_series(df)
    high_column = _find_column(df, ("high", "high_price"))
    low_column = _find_column(df, ("low", "low_price"))
    if high_column is not None and low_column is not None:
        high = pd.to_numeric(df[high_column], errors="coerce").astype(float)
        low = pd.to_numeric(df[low_column], errors="coerce").astype(float)
        previous_close = close.shift(1)
        true_range = pd.concat(
            [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
            axis=1,
        ).max(axis=1)
        denominator = close.abs().replace(0.0, np.nan)
        return (true_range / denominator).rolling(
            max(1, int(window)), min_periods=1
        ).mean()

    returns = close.pct_change()
    return returns.rolling(max(1, int(window)), min_periods=2).std().abs()


def _trend_strength_series(df: pd.DataFrame, window: int) -> pd.Series:
    """Return absolute distance from a causal moving average."""
    price = _price_series(df)
    sma = price.rolling(max(1, int(window)), min_periods=1).mean()
    return (price / sma.replace(0.0, np.nan) - 1.0).abs()


def _threshold(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        return 0.0
    return float(numeric.median())


def _state_mask(values: pd.Series, threshold: float, high: bool) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    numeric = np.nan_to_num(numeric, nan=threshold, posinf=threshold, neginf=threshold)
    mask = numeric >= threshold if high else numeric < threshold
    # A constant proxy should still produce two deterministic halves.  This is
    # a coverage fallback, not a data-derived performance gate.
    if len(numeric) > 1 and (bool(mask.all()) or not bool(mask.any())):
        order = np.argsort(numeric, kind="stable")
        cut = max(1, len(order) // 2)
        if high:
            fallback = np.zeros(len(numeric), dtype=bool)
            fallback[order[-cut:]] = True
            mask = fallback
        else:
            fallback = np.ones(len(numeric), dtype=bool)
            fallback[order[-cut:]] = False
            mask = fallback
    return np.asarray(mask, dtype=bool)


def classify_regimes(
    df: pd.DataFrame,
    *,
    reference_df: pd.DataFrame | None = None,
    volatility_window: int | None = None,
    sma_window: int | None = None,
) -> pd.DataFrame:
    """Classify rows as high/low volatility and trend/range.

    Thresholds are fitted on ``reference_df`` when supplied.  The validation
    frame can therefore be classified without fitting a threshold on its own
    outcomes.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    vol_window = int(
        volatility_window
        if volatility_window is not None
        else getattr(_cfg, "RB_REGIME_VOLATILITY_WINDOW", 20)
    )
    trend_window = int(
        sma_window
        if sma_window is not None
        else getattr(_cfg, "RB_REGIME_SMA_WINDOW", 20)
    )
    reference = df if reference_df is None else reference_df
    reference_vol = _volatility_series(reference, vol_window)
    reference_trend = _trend_strength_series(reference, trend_window)
    vol_threshold = _threshold(reference_vol)
    trend_threshold = _threshold(reference_trend)

    volatility = _volatility_series(df, vol_window)
    trend_strength = _trend_strength_series(df, trend_window)
    high_mask = _state_mask(volatility, vol_threshold, high=True)
    trend_mask = _state_mask(trend_strength, trend_threshold, high=True)
    return pd.DataFrame(
        {
            "volatility": pd.to_numeric(volatility, errors="coerce").to_numpy(dtype=float),
            "trend_strength": pd.to_numeric(trend_strength, errors="coerce").to_numpy(dtype=float),
            "volatility_regime": np.where(high_mask, "high_vol", "low_vol"),
            "trend_regime": np.where(trend_mask, "trend", "range"),
        },
        index=df.index,
    )


def split_regimes(
    df: pd.DataFrame,
    *,
    reference_df: pd.DataFrame | None = None,
    volatility_window: int | None = None,
    sma_window: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Return the four causal regime slices for a DataFrame."""
    labels = classify_regimes(
        df,
        reference_df=reference_df,
        volatility_window=volatility_window,
        sma_window=sma_window,
    )
    return {
        "high_vol": df.loc[labels["volatility_regime"] == "high_vol"].copy(),
        "low_vol": df.loc[labels["volatility_regime"] == "low_vol"].copy(),
        "trend": df.loc[labels["trend_regime"] == "trend"].copy(),
        "range": df.loc[labels["trend_regime"] == "range"].copy(),
    }


def _frame_for(source: Any) -> pd.DataFrame | None:
    if isinstance(source, pd.DataFrame):
        return source
    frame = getattr(source, "df", None)
    return frame if isinstance(frame, pd.DataFrame) else None


def _direction_for(source: Any, direction: str | None) -> str:
    return str(direction or getattr(source, "trade_direction", "long"))


def _engine_for_frame(
    source: Any,
    frame: pd.DataFrame,
    direction: str,
) -> CPUBacktestEngine | Any:
    if frame is _frame_for(source) and callable(getattr(source, "simulate_rule_set", None)):
        return source
    if not len(frame):
        # CPUBacktestEngine needs label columns even for an empty frame.  The
        # caller handles an empty slice before reaching this branch.
        return None
    if callable(getattr(source, "simulate_rule_set", None)):
        kwargs = {
            name: getattr(source, name)
            for name in (
                "initial_capital",
                "leverage",
                "fee_pct",
                "spread_bps",
                "slippage_bps",
                "max_hold_candles",
                "max_total_exposure_pct",
                "min_position_notional",
            )
            if hasattr(source, name)
        }
        engine = CPUBacktestEngine(
            frame,
            getattr(source, "feature_modes", {}),
            direction,
            **kwargs,
        )
        context_mask = getattr(source, "_context_mask", None)
        source_frame = _frame_for(source)
        if context_mask is not None and source_frame is not None:
            # The subset retains the source index, so align by labels where
            # possible.  A positional fallback covers reset-index fixtures.
            try:
                positions = source_frame.index.get_indexer(frame.index)
                if np.all(positions >= 0):
                    engine._context_mask = np.asarray(context_mask)[positions].copy()  # type: ignore[attr-defined]
            except Exception:
                pass
        return engine
    return CPUBacktestEngine(frame, {}, direction)


def _metric_snapshot(metrics: Mapping[str, Any] | None) -> dict[str, Any]:
    values = metrics or {}
    total_return = values.get(
        "total_return_pct", values.get("return_pct", values.get("Return"))
    )
    profit_factor = values.get("profit_factor", values.get("pf", values.get("PF")))
    drawdown = values.get(
        "max_drawdown_pct", values.get("mdd_pct", values.get("MDD"))
    )
    sortino = values.get("sortino_ratio", values.get("sortino"))
    return {
        "total_return_pct": _finite_float(total_return),
        "return_pct": _finite_float(total_return),
        "profit_factor": _finite_float(profit_factor),
        "PF": _finite_float(profit_factor),
        "max_drawdown_pct": _finite_float(drawdown),
        "MDD": _finite_float(drawdown),
        "sortino_ratio": _finite_float(sortino),
        "sortino": _finite_float(sortino),
        "executed_trades": int(_finite_float(values.get("executed_trades"))),
        "Return": _finite_float(total_return),
        "available": bool(values),
    }


def _evaluate_slice(
    source: Any,
    frame: pd.DataFrame,
    rules: Sequence[dict],
    direction: str,
) -> dict[str, Any]:
    if frame.empty or not rules:
        return _metric_snapshot({}) | {"available": False}
    try:
        engine = _engine_for_frame(source, frame, direction)
        if engine is None:
            return _metric_snapshot({}) | {"available": False}
        result = engine.simulate_rule_set([dict(rule) for rule in rules])
        return _metric_snapshot(result)
    except Exception as exc:  # pragma: no cover - diagnostic fail-safe
        logger.warning("Regime slice simulation failed: %s", exc)
        return _metric_snapshot({}) | {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _profit_concentration(
    regimes: Mapping[str, Mapping[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    profits = {
        name: max(0.0, _finite_float(metrics.get("total_return_pct")))
        for name, metrics in regimes.items()
    }
    total_profit = float(sum(profits.values()))
    if total_profit <= 0.0:
        shares = {name: 0.0 for name in profits}
        top_regime = ""
        top_share = 0.0
        hhi = 0.0
    else:
        shares = {name: value / total_profit for name, value in profits.items()}
        top_regime = max(shares, key=shares.get)
        top_share = float(shares[top_regime])
        hhi = float(sum(value * value for value in shares.values()))
    warning = bool(total_profit > 0.0 and top_share >= float(threshold))
    return {
        "positive_profit_pct": profits,
        "total_positive_profit_pct": total_profit,
        "profit_shares": shares,
        "top_regime": top_regime,
        "top_regime_profit_share": top_share,
        "hhi": hhi,
        "hhi_concentration": hhi,
        "concentration_threshold": float(threshold),
        "concentration_warning": warning,
        "warning": warning,
    }


def _regime_split_report(
    source: Any,
    frame: pd.DataFrame,
    reference_df: pd.DataFrame,
    rules: Sequence[dict],
    direction: str,
) -> dict[str, Any]:
    slices = split_regimes(frame, reference_df=reference_df)
    regimes: dict[str, dict[str, Any]] = {}
    for name in _REGIMES:
        slice_df = slices[name]
        metrics = _evaluate_slice(source, slice_df, rules, direction)
        # Keep the common metrics both flat and under ``metrics``.  The flat
        # form is useful for CSV/dashboard consumers and the nested form keeps
        # the certificate self-describing.
        regimes[name] = {
            "row_count": int(len(slice_df)),
            "metrics": metrics,
            **metrics,
            "return": metrics["total_return_pct"],
            "pf": metrics["profit_factor"],
            "mdd": metrics["max_drawdown_pct"],
        }
    concentration = _profit_concentration(
        regimes,
        _finite_float(
            getattr(_cfg, "RB_REGIME_PROFIT_CONCENTRATION_THRESHOLD", 0.70),
            0.70,
        ),
    )
    return {
        "row_count": int(len(frame)),
        "regimes": regimes,
        # Alias retained for report readers that call this section metrics.
        "regime_metrics": regimes,
        "profit_concentration": concentration,
        "hhi": concentration["hhi"],
        "hhi_concentration": concentration["hhi"],
        "top_regime": concentration["top_regime"],
        "top_regime_profit_share": concentration["top_regime_profit_share"],
        "concentration_warning": concentration["concentration_warning"],
        "warning": concentration["warning"],
    }


def regime_robustness_certificate(
    train_source: pd.DataFrame | CPUBacktestEngine | Any,
    validation_source: pd.DataFrame | CPUBacktestEngine | Any | Sequence[dict] | None = None,
    rules: Sequence[dict] | None = None,
    direction: str | None = None,
    *,
    enabled: bool | None = None,
) -> dict[str, Any]:
    """Report train/validation results for four observable regimes."""
    # Accept ``(source, rules)`` for a single frozen split as well as the
    # governor's ``(train_source, validation_source, rules)`` form.
    if rules is None:
        if isinstance(validation_source, Sequence) and not isinstance(
            validation_source, (str, bytes, pd.DataFrame)
        ):
            rules = validation_source
            validation_source = None
        else:
            rules = []
    rules_value = list(rules)
    enabled_value = bool(
        getattr(_cfg, "RB_REGIME_ROBUSTNESS_ENABLED", True)
        if enabled is None
        else enabled
    )
    report_only = bool(
        getattr(_cfg, "RB_ROBUSTNESS_REPORT_ONLY", True)
        or getattr(_cfg, "RB_REGIME_ROBUSTNESS_REPORT_ONLY", True)
    )
    if not enabled_value:
        return {
            "schema_version": "1.0",
            "certificate": "regime_robustness",
            "enabled": False,
            "available": False,
            "diagnostic_only": True,
            "report_only": report_only,
            "splits": {},
            "verdict": "disabled",
        }

    train_frame = _frame_for(train_source)
    validation_frame = _frame_for(validation_source)
    if train_frame is None or train_frame.empty or not rules_value:
        return {
            "schema_version": "1.0",
            "certificate": "regime_robustness",
            "enabled": True,
            "available": False,
            "diagnostic_only": True,
            "report_only": report_only,
            "splits": {},
            "verdict": "unavailable",
        }
    direction_value = _direction_for(train_source, direction)
    splits: dict[str, dict[str, Any]] = {
        "train": _regime_split_report(
            train_source,
            train_frame,
            train_frame,
            rules_value,
            direction_value,
        )
    }
    if validation_frame is not None and not validation_frame.empty:
        splits["validation"] = _regime_split_report(
            validation_source,
            validation_frame,
            train_frame,
            rules_value,
            _direction_for(validation_source, direction_value),
        )
    warnings = [
        name
        for name, split in splits.items()
        if bool(split.get("concentration_warning"))
    ]
    report = {
        "schema_version": "1.0",
        "certificate": "regime_robustness",
        "enabled": True,
        "available": bool(splits) and bool(rules_value),
        "diagnostic_only": True,
        "report_only": report_only,
        "splits": splits,
        "hhi_warning_splits": warnings,
        "concentration_warning": bool(warnings),
        "verdict": "concentrated" if warnings else "stable",
    }
    # Keep the split names at the top level as a convenience for report readers
    # and for the common two-split certificate shape.
    report.update(splits)
    return report


def _simulate_rules(source: Any, rules: Sequence[dict]) -> dict[str, Any]:
    if source is None:
        return {}
    simulate = getattr(source, "simulate_rule_set", None)
    if not callable(simulate):
        return {}
    try:
        return dict(simulate([dict(rule) for rule in rules]))
    except Exception as exc:  # pragma: no cover - diagnostic fail-safe
        logger.warning("Rule dropout simulation failed: %s", exc)
        return {}


def rule_dropout_stress(
    engine_or_frame: CPUBacktestEngine | pd.DataFrame | Any,
    rules: Sequence[dict],
    *,
    direction: str | None = None,
    enabled: bool | None = None,
) -> dict[str, Any]:
    """Compare the frozen portfolio with each rule removed once."""
    enabled_value = bool(
        getattr(_cfg, "RB_RULE_DROPOUT_STRESS_ENABLED", True)
        if enabled is None
        else enabled
    )
    report_only = bool(
        getattr(_cfg, "RB_ROBUSTNESS_REPORT_ONLY", True)
        or getattr(_cfg, "RB_RULE_DROPOUT_STRESS_REPORT_ONLY", True)
    )
    if not enabled_value:
        return {
            "schema_version": "1.0",
            "certificate": "rule_dropout_stress",
            "enabled": False,
            "available": False,
            "diagnostic_only": True,
            "report_only": report_only,
            "per_rule": [],
            "verdict": "disabled",
        }
    source = engine_or_frame
    if isinstance(source, pd.DataFrame):
        source = CPUBacktestEngine(source, {}, direction or "long")
    rules_value = list(rules)
    full_metrics = _simulate_rules(source, rules_value)
    full = _metric_snapshot(full_metrics)
    if not rules_value:
        full["available"] = False
    rows: list[dict[str, Any]] = []
    for index, rule in enumerate(rules_value):
        without_rules = [
            dict(item)
            for item_index, item in enumerate(rules_value)
            if item_index != index
        ]
        without_metrics = _metric_snapshot(_simulate_rules(source, without_rules))
        delta = {
            key: float(full[key] - without_metrics[key])
            for key in _METRIC_KEYS
            if key in full and key in without_metrics
        }
        rows.append(
            {
                "rule_index": int(index + 1),
                "rule_id": str(rule.get("rule_id", rule.get("phase2_rule_id", index + 1))),
                "rule": dict(rule),
                "full": full,
                "without_rule": without_metrics,
                "delta": delta,
                "return_delta_pct": float(delta.get("total_return_pct", 0.0)),
                "sortino_delta": float(delta.get("sortino_ratio", 0.0)),
            }
        )

    return_deltas = [row["return_delta_pct"] for row in rows]
    if return_deltas:
        # ``full - without_rule`` is the return lost when a rule is removed.
        # The worst dropout is therefore the largest positive delta, not the
        # smallest one (a negative delta means that removal improved return).
        worst_row = max(rows, key=lambda row: row["return_delta_pct"])
        median_values = {
            key: float(
                np.median(
                    np.asarray(
                        [row["delta"].get(key, 0.0) for row in rows],
                        dtype=float,
                    )
                )
            )
            for key in _METRIC_KEYS
        }
        median_return_delta = median_values["total_return_pct"]
        worst_return_delta = float(worst_row["return_delta_pct"])
    else:
        worst_row = None
        median_values = {key: 0.0 for key in _METRIC_KEYS}
        median_return_delta = 0.0
        worst_return_delta = 0.0
    median_without_values = {
        key: float(
            np.median(
                np.asarray(
                    [row["without_rule"].get(key, 0.0) for row in rows],
                    dtype=float,
                )
            )
        )
        for key in _METRIC_KEYS
    } if rows else {key: 0.0 for key in _METRIC_KEYS}
    full_return = _finite_float(full.get("total_return_pct"))
    positive_loss = max(0.0, max(return_deltas, default=0.0))
    dependency_share = positive_loss / max(full_return, 1e-12) if full_return > 0 else 0.0
    dependency_threshold = _finite_float(
        getattr(_cfg, "RB_RULE_DROPOUT_DEPENDENCY_THRESHOLD", 0.70),
        0.70,
    )
    dependency = bool(
        full.get("available")
        and rows
        and (len(rows) == 1 or dependency_share >= dependency_threshold)
    )
    return {
        "schema_version": "1.0",
        "certificate": "rule_dropout_stress",
        "enabled": True,
        "available": bool(full.get("available")),
        "diagnostic_only": True,
        "report_only": report_only,
        "rule_count": int(len(rules_value)),
        "full": full,
        "full_metrics": full,
        "per_rule": rows,
        "dropouts": rows,
        "rules": rows,
        "worst_dropout": worst_row,
        "median_dropout": {
            "return_delta_pct": median_return_delta,
            "delta": median_values,
            "metrics": median_values,
            "without_rule": median_without_values,
            "return_pct": median_without_values["total_return_pct"],
        },
        "worst_return_delta_pct": worst_return_delta,
        "worst_dropout_return_delta_pct": worst_return_delta,
        "worst_dropout_return_pct": (
            float(worst_row["without_rule"]["total_return_pct"])
            if worst_row is not None
            else 0.0
        ),
        "median_return_delta_pct": median_return_delta,
        "median_dropout_return_pct": median_without_values["total_return_pct"],
        "dependency_share": float(dependency_share),
        "dependency_threshold": float(dependency_threshold),
        "single_rule_dependency": dependency,
        "verdict": "single_rule_dependency" if dependency else "independent",
    }


run_rule_dropout_stress = rule_dropout_stress
dropout_stress_certificate = rule_dropout_stress
build_rule_dropout_certificate = rule_dropout_stress
evaluate_rule_dropout_stress = rule_dropout_stress
run_regime_robustness = regime_robustness_certificate
build_regime_robustness_certificate = regime_robustness_certificate
regime_stress_certificate = regime_robustness_certificate
evaluate_regime_robustness = regime_robustness_certificate


def _write_json(certificate: Mapping[str, Any], filename: str, output_dir: str | Path | None) -> str:
    target = Path(output_dir or getattr(_cfg, "REPORTS_DIR", "outputs/reports"))
    if target.suffix.lower() != ".json":
        target = target / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(certificate), indent=2, default=str), encoding="utf-8")
    return str(target.resolve())


def write_regime_robustness_report(
    certificate: Mapping[str, Any],
    output_dir: str | Path | None = None,
) -> str:
    """Write ``reports/regime_robustness.json``."""
    return _write_json(certificate, "regime_robustness.json", output_dir)


def write_rule_dropout_report(
    certificate: Mapping[str, Any],
    output_dir: str | Path | None = None,
) -> str:
    """Write ``reports/rule_dropout_stress.json``."""
    return _write_json(certificate, "rule_dropout_stress.json", output_dir)


__all__ = [
    "build_regime_robustness_certificate",
    "build_rule_dropout_certificate",
    "classify_regimes",
    "dropout_stress_certificate",
    "evaluate_regime_robustness",
    "evaluate_rule_dropout_stress",
    "regime_stress_certificate",
    "regime_robustness_certificate",
    "rule_dropout_stress",
    "run_regime_robustness",
    "run_rule_dropout_stress",
    "split_regimes",
    "write_regime_robustness_report",
    "write_rule_dropout_report",
]
