"""Transaction-cost stress diagnostics for the frozen portfolio.

The cost certificate evaluates one already selected rule set at several cost
levels.  It never changes the rule set, exits, or sizing.  The certificate can
be report-only or an explicit acceptance gate, as configured by the caller.
This keeps the result as economic evidence for the frozen portfolio rather
than a second optimisation pass.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import json
import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.backtest.cpu_engine import CPUBacktestEngine


logger = logging.getLogger(__name__)


def _finite_float(value: Any, default: float = 0.0) -> float:
    """Return a finite float without allowing malformed metrics to leak out."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _metric_snapshot(metrics: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep the stable metrics used by all robustness reports."""
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
        # ``return_pct`` is a short alias used by report consumers.
        "return_pct": _finite_float(total_return),
        "profit_factor": _finite_float(profit_factor),
        "max_drawdown_pct": _finite_float(drawdown),
        "sortino_ratio": _finite_float(sortino),
        "executed_trades": int(_finite_float(values.get("executed_trades"))),
        "available": bool(values),
    }


def _normalise_multipliers(
    multipliers: Iterable[float] | None,
) -> tuple[float, ...]:
    """Return sorted, unique cost multipliers, including no invalid values."""
    raw = (
        multipliers
        if multipliers is not None
        else getattr(_cfg, "RB_COST_STRESS_MULTIPLIERS", (1.0, 1.5, 2.0))
    )
    result: set[float] = set()
    for value in raw:
        try:
            factor = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(factor) and factor >= 1.0:
            result.add(factor)
    if not result:
        result = {1.0, 1.5, 2.0}
    return tuple(sorted(result))


def _source_df(source: Any) -> pd.DataFrame | None:
    if isinstance(source, pd.DataFrame):
        return source
    frame = getattr(source, "df", None)
    return frame if isinstance(frame, pd.DataFrame) else None


def _source_direction(source: Any, direction: str | None) -> str:
    if direction:
        return str(direction)
    return str(getattr(source, "trade_direction", "long"))


def _simulate_at_cost(
    source: Any,
    rules: Sequence[dict],
    multiplier: float,
    *,
    direction: str | None,
    signal_masks: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Simulate *rules* with costs multiplied by *multiplier*.

    Lightweight test doubles often expose only ``simulate_rule_set`` and no
    DataFrame.  They remain supported as a report-only fallback; a real
    ``CPUBacktestEngine`` is rebuilt from its frozen input frame so the cost is
    actually applied.
    """
    if source is None or not rules:
        return {}
    frame = _source_df(source)
    if frame is None:
        simulate = getattr(source, "simulate_rule_set", None)
        if not callable(simulate):
            return {}
        try:
            return dict(simulate([dict(rule) for rule in rules]))
        except Exception as exc:  # pragma: no cover - defensive diagnostics
            logger.warning("Cost stress fallback simulation failed: %s", exc)
            return {}

    fee_pct = _finite_float(
        getattr(source, "fee_pct", getattr(_cfg, "FEE_PCT", 0.0))
    ) * multiplier
    spread_bps = _finite_float(
        getattr(source, "spread_bps", getattr(_cfg, "SPREAD_BPS", 0.0))
    ) * multiplier
    slippage_bps = _finite_float(
        getattr(source, "slippage_bps", getattr(_cfg, "SLIPPAGE_BPS", 0.0))
    ) * multiplier
    engine = CPUBacktestEngine(
        frame,
        getattr(source, "feature_modes", {}),
        _source_direction(source, direction),
        initial_capital=_finite_float(
            getattr(source, "initial_capital", getattr(_cfg, "INITIAL_CAPITAL", 1000.0)),
            1000.0,
        ),
        leverage=_finite_float(
            getattr(source, "leverage", getattr(_cfg, "LEVERAGE", 1.0)),
            1.0,
        ),
        fee_pct=fee_pct,
        spread_bps=spread_bps,
        slippage_bps=slippage_bps,
        max_hold_candles=int(
            getattr(source, "max_hold_candles", getattr(_cfg, "MAX_HOLD_CANDLES", 96))
        ),
        max_total_exposure_pct=_finite_float(
            getattr(
                source,
                "max_total_exposure_pct",
                getattr(_cfg, "MAX_TOTAL_EXPOSURE_PCT", 100.0),
            ),
            100.0,
        ),
        min_position_notional=_finite_float(
            getattr(
                source,
                "min_position_notional",
                getattr(_cfg, "MIN_POSITION_NOTIONAL", 1.0),
            ),
            1.0,
        ),
    )
    # Preserve a caller-provided context mask when one exists.  The frame and
    # row order are unchanged, so this is safe and avoids a hidden policy drift
    # during a diagnostic run.
    context_mask = getattr(source, "_context_mask", None)
    if context_mask is not None and len(context_mask) == len(frame):
        engine._context_mask = context_mask.copy()  # type: ignore[attr-defined]
    formatted_rules = [dict(rule) for rule in rules]
    if signal_masks is not None:
        return dict(engine.simulate_signal_masks(signal_masks, formatted_rules))
    return dict(engine.simulate_rule_set(formatted_rules))


def _row_for_multiplier(
    multiplier: float,
    train_metrics: Mapping[str, Any],
    validation_metrics: Mapping[str, Any],
    *,
    min_return_pct: float,
    min_pf: float,
) -> dict[str, Any]:
    """Build one stable row, using validation as the headline result."""
    train = _metric_snapshot(train_metrics)
    validation = _metric_snapshot(validation_metrics)
    headline = validation if validation["available"] else train
    passed = bool(
        headline["available"]
        and headline["total_return_pct"] >= min_return_pct
        and headline["profit_factor"] >= min_pf
    )
    return {
        "multiplier": float(multiplier),
        "fee_multiplier": float(multiplier),
        "cost_multiplier": float(multiplier),
        "train": train,
        "validation": validation,
        # Flat fields make the JSON convenient for simple dashboards and keep
        # compatibility with the original RB gate's list-of-results shape.
        "total_return_pct": float(headline["total_return_pct"]),
        "return_pct": float(headline["return_pct"]),
        "profit_factor": float(headline["profit_factor"]),
        "pf": float(headline["profit_factor"]),
        "PF": float(headline["profit_factor"]),
        "max_drawdown_pct": float(headline["max_drawdown_pct"]),
        "mdd_pct": float(headline["max_drawdown_pct"]),
        "MDD": float(headline["max_drawdown_pct"]),
        "sortino_ratio": float(headline["sortino_ratio"]),
        "sortino": float(headline["sortino_ratio"]),
        "executed_trades": int(headline["executed_trades"]),
        "Return": float(headline["total_return_pct"]),
        "passed": passed,
    }


def summarise_cost_stress_results(
    results: Sequence[Mapping[str, Any]],
    *,
    enabled: bool = True,
    report_only: bool | None = None,
    min_return_pct: float | None = None,
    min_pf: float | None = None,
) -> dict[str, Any]:
    """Summarise rows returned by the RB compatibility gate.

    This helper lets the governor avoid running the expensive stress curve a
    second time merely to construct the aggregate report.
    """
    rows = [dict(row) for row in results]
    report_only_value = bool(
        (
            getattr(_cfg, "RB_COST_STRESS_REPORT_ONLY", True)
        )
        if report_only is None
        else report_only
    )
    return _finalise_certificate(
        rows,
        enabled=enabled,
        report_only=report_only_value,
        min_return_pct=(
            _finite_float(getattr(_cfg, "RB_COST_STRESS_MIN_RETURN_PCT", 0.0))
            if min_return_pct is None
            else float(min_return_pct)
        ),
        min_pf=(
            _finite_float(getattr(_cfg, "RB_COST_STRESS_MIN_PF", 1.0), 1.0)
            if min_pf is None
            else float(min_pf)
        ),
    )


def _finalise_certificate(
    rows: list[dict[str, Any]],
    *,
    enabled: bool,
    report_only: bool,
    min_return_pct: float,
    min_pf: float,
) -> dict[str, Any]:
    available = bool(rows) and any(
        bool(row.get("validation", {}).get("available"))
        or bool(row.get("train", {}).get("available"))
        # Rows from the historical gate have only flat fields.  New rows
        # always carry nested availability markers, so a zeroed unavailable
        # row is not mistaken for usable evidence.
        or (
            "train" not in row
            and "validation" not in row
            and "profit_factor" in row
        )
        for row in rows
    )
    valid_rows = [row for row in rows if "profit_factor" in row]
    pf_values = [
        _finite_float(row.get("profit_factor")) for row in valid_rows
    ]
    last_row = valid_rows[-1] if valid_rows else {}
    fragile_floor = _finite_float(
        getattr(_cfg, "RB_COST_STRESS_FRAGILE_PF_FLOOR", 1.0), 1.0
    )
    fragile = bool(
        available
        and valid_rows
        and (
            any(
                _finite_float(row.get("multiplier"), 1.0) > 1.0
                and _finite_float(row.get("profit_factor")) < fragile_floor
                for row in valid_rows
            )
            or (
                len(pf_values) >= 2
                and pf_values[-1] < pf_values[0]
                and pf_values[-1] < fragile_floor
            )
        )
    )
    verdict = (
        "disabled"
        if not enabled
        else "unavailable"
        if not available
        else "fragile"
        if fragile
        else "robust"
    )
    by_multiplier = {
        str(row.get("multiplier")): {
            "profit_factor": _finite_float(row.get("profit_factor")),
            "max_drawdown_pct": _finite_float(row.get("max_drawdown_pct")),
            "total_return_pct": _finite_float(row.get("total_return_pct")),
        }
        for row in valid_rows
    }
    passed = (
        not enabled
        or bool(valid_rows)
        and all(bool(row.get("passed", False)) for row in valid_rows)
    )
    return {
        "schema_version": "1.0",
        "certificate": "cost_stress",
        "enabled": bool(enabled),
        "available": bool(available),
        "diagnostic_only": True,
        "report_only": bool(report_only),
        "hard_gate": bool(not report_only and getattr(_cfg, "RB_COST_STRESS_HARD_GATE", False)),
        "min_return_pct": float(min_return_pct),
        "min_profit_factor": float(min_pf),
        "fragile_threshold_profit_factor": float(fragile_floor),
        "stress_curve": rows,
        "results": rows,
        "multipliers": [float(row.get("multiplier", 0.0)) for row in valid_rows],
        "profit_factors": [
            _finite_float(row.get("profit_factor")) for row in valid_rows
        ],
        "drawdowns": [
            _finite_float(row.get("max_drawdown_pct")) for row in valid_rows
        ],
        "returns": [
            _finite_float(row.get("total_return_pct")) for row in valid_rows
        ],
        "by_multiplier": by_multiplier,
        "metrics_by_multiplier": by_multiplier,
        "verdict": verdict,
        "fragile": fragile,
        "fragile_verdict": "FRAGILE" if fragile else verdict.upper(),
        "passed": bool(passed),
        "headline": {
            "multiplier": last_row.get("multiplier"),
            "profit_factor": _finite_float(last_row.get("profit_factor")),
            "max_drawdown_pct": _finite_float(last_row.get("max_drawdown_pct")),
            "total_return_pct": _finite_float(last_row.get("total_return_pct")),
        },
    }


def cost_stress_certificate(
    train_engine: CPUBacktestEngine | pd.DataFrame | Any,
    validation_engine: CPUBacktestEngine | pd.DataFrame | Any | Sequence[dict] | None = None,
    rules: Sequence[dict] | None = None,
    *,
    multipliers: Iterable[float] | None = None,
    direction: str | None = None,
    enabled: bool | None = None,
    report_only: bool | None = None,
    min_return_pct: float | None = None,
    min_pf: float | None = None,
    train_signal_masks: Sequence[Any] | None = None,
    validation_signal_masks: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a frozen portfolio at 1.0x, 1.5x, and 2.0x costs.

    The default multipliers come from configuration.  Current configuration
    includes all three requested points, while callers may pass a custom curve
    for a controlled diagnostic.
    """
    # Support both ``(engine, rules)`` and ``(train_engine, valid_engine,
    # rules)``.  The one-engine form is useful for direct diagnostics and
    # keeps the certificate independent from the governor's calling shape.
    if rules is None:
        if isinstance(validation_engine, Sequence) and not isinstance(
            validation_engine, (str, bytes, pd.DataFrame)
        ):
            rules = validation_engine
            validation_engine = train_engine
        else:
            raise TypeError(
                "rules must be supplied as the second or third argument"
            )
    rules_value = list(rules)

    enabled_value = bool(
        getattr(_cfg, "RB_COST_STRESS_ENABLED", True)
        if enabled is None
        else enabled
    )
    report_only_value = bool(
        getattr(_cfg, "RB_COST_STRESS_REPORT_ONLY", True)
        if report_only is None
        else report_only
    )
    min_return_value = (
        _finite_float(getattr(_cfg, "RB_COST_STRESS_MIN_RETURN_PCT", 0.0))
        if min_return_pct is None
        else float(min_return_pct)
    )
    min_pf_value = (
        _finite_float(getattr(_cfg, "RB_COST_STRESS_MIN_PF", 1.0), 1.0)
        if min_pf is None
        else float(min_pf)
    )
    if not enabled_value:
        return _finalise_certificate(
            [],
            enabled=False,
            report_only=report_only_value,
            min_return_pct=min_return_value,
            min_pf=min_pf_value,
        )

    rows: list[dict[str, Any]] = []
    for multiplier in _normalise_multipliers(multipliers):
        try:
            train_metrics = _simulate_at_cost(
                train_engine,
                rules_value,
                multiplier,
                direction=direction,
                signal_masks=train_signal_masks,
            )
        except Exception as exc:  # pragma: no cover - defensive report path
            logger.warning("Cost stress train evaluation failed at %.2fx: %s", multiplier, exc)
            train_metrics = {}
        try:
            validation_metrics = _simulate_at_cost(
                validation_engine,
                rules_value,
                multiplier,
                direction=direction,
                signal_masks=validation_signal_masks,
            )
        except Exception as exc:  # pragma: no cover - defensive report path
            logger.warning(
                "Cost stress validation evaluation failed at %.2fx: %s",
                multiplier,
                exc,
            )
            validation_metrics = {}
        rows.append(
            _row_for_multiplier(
                multiplier,
                train_metrics,
                validation_metrics,
                min_return_pct=min_return_value,
                min_pf=min_pf_value,
            )
        )
    return _finalise_certificate(
        rows,
        enabled=True,
        report_only=report_only_value,
        min_return_pct=min_return_value,
        min_pf=min_pf_value,
    )


# Clear aliases for callers that prefer an action-oriented name.
run_cost_stress = cost_stress_certificate
build_cost_stress_certificate = cost_stress_certificate
summarize_cost_stress_results = summarise_cost_stress_results


def write_cost_stress_report(
    certificate: Mapping[str, Any],
    output_dir: str | Path | None = None,
) -> str:
    """Write one cost certificate to ``reports/cost_stress.json``."""
    target = Path(output_dir or getattr(_cfg, "REPORTS_DIR", "outputs/reports"))
    if target.suffix.lower() != ".json":
        target = target / "cost_stress.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(dict(certificate), indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("Saved cost stress certificate: %s", target)
    return str(target.resolve())


__all__ = [
    "build_cost_stress_certificate",
    "cost_stress",
    "cost_stress_certificate",
    "evaluate_cost_stress",
    "run_cost_stress",
    "summarize_cost_stress_results",
    "summarise_cost_stress_results",
    "write_cost_stress_report",
]


# Short names are kept for callers that expose certificates as diagnostics
# rather than as a validation stage.
cost_stress = cost_stress_certificate
evaluate_cost_stress = cost_stress_certificate
