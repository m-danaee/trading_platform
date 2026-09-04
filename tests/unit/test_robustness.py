"""Focused tests for report-only robustness certificates."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config
from gpu_fuzzy_trader.backtest.cpu_engine import CPUBacktestEngine
from gpu_fuzzy_trader.reporting.reporter import Reporter
from gpu_fuzzy_trader.validation.cost_stress import cost_stress_certificate
from gpu_fuzzy_trader.validation.execution_stress import (
    next_row_indices,
    execution_stress_certificate,
    shift_signal_mask,
    simulate_delayed_entries,
)
from gpu_fuzzy_trader.validation.regime_robustness import (
    regime_robustness_certificate,
    rule_dropout_stress,
)


def _frame(rows: int = 48) -> pd.DataFrame:
    index = np.arange(rows)
    close = 100.0 + index * 0.1
    close[rows // 2 :] += np.sin(index[rows // 2 :] / 2.0) * 2.0
    return pd.DataFrame(
        {
            "symbol": "BTCUSDT",
            "datetime": pd.date_range("2024-01-01", periods=rows, freq="15min"),
            "_symbol_bar_index": index,
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "signal_a": (index % 3 == 0).astype(float),
            "signal_b": (index % 5 == 0).astype(float),
            "label_open_next": close + 0.05,
            "label_max_288": close + 1.0,
            "label_min_288": close - 0.5,
            "label_close_288": close + np.where(index % 4 == 0, 0.8, -0.2),
            "label_max_before_min": 1,
        }
    )


def _engine(df: pd.DataFrame) -> CPUBacktestEngine:
    return CPUBacktestEngine(df, {}, "long", max_hold_candles=2)


def _rules() -> list[dict]:
    return [
        {
            "conditions": ["[signal_a] >= 1"],
            "tp": 2.0,
            "sl": 1.2,
            "capital_pct": 10.0,
        },
        {
            "conditions": ["[signal_b] >= 1"],
            "tp": 2.0,
            "sl": 1.2,
            "capital_pct": 10.0,
        },
    ]


class _DropoutMetricsEngine:
    """Small deterministic engine for testing dropout ranking semantics."""

    def simulate_rule_set(self, rules):
        rule_ids = tuple(rule.get("rule_id") for rule in rules)
        metrics = {
            ("r1", "r2"): (10.0, 1.8),
            ("r2",): (3.0, 1.2),
            ("r1",): (8.0, 1.5),
            (): (0.0, 0.0),
        }
        total_return, profit_factor = metrics[rule_ids]
        return {
            "total_return_pct": total_return,
            "profit_factor": profit_factor,
            "max_drawdown_pct": 2.0,
            "sortino_ratio": 1.0,
            "executed_trades": 10,
        }


def test_cost_certificate_has_three_frozen_cost_points(tmp_path):
    df = _frame()
    certificate = cost_stress_certificate(
        _engine(df), _engine(df), _rules(), multipliers=(1.0, 1.5, 2.0)
    )

    assert [row["multiplier"] for row in certificate["stress_curve"]] == [
        1.0,
        1.5,
        2.0,
    ]
    for row in certificate["stress_curve"]:
        assert {"profit_factor", "max_drawdown_pct", "total_return_pct"} <= set(row)
    assert certificate["report_only"] is False
    output = tmp_path / "cost_stress.json"
    output.write_text(json.dumps(certificate), encoding="utf-8")
    assert json.loads(output.read_text(encoding="utf-8"))["verdict"] in {
        "robust",
        "fragile",
        "unavailable",
    }


def test_cost_certificate_can_stress_frozen_per_rule_masks():
    df = _frame()
    masks = [np.zeros(len(df), dtype=bool) for _ in _rules()]
    certificate = cost_stress_certificate(
        _engine(df),
        _engine(df),
        _rules(),
        train_signal_masks=masks,
        validation_signal_masks=masks,
    )

    assert certificate["available"] is True
    assert all(
        row["train"]["executed_trades"] == 0
        and row["validation"]["executed_trades"] == 0
        for row in certificate["stress_curve"]
    )


def test_execution_certificate_reports_delayed_sortino():
    df = _frame()
    certificate = execution_stress_certificate(_engine(df), _rules())

    assert certificate["delay_bars"] == 1
    assert "normal_sortino" in certificate
    assert "delayed_sortino" in certificate
    assert "sortino_delta" in certificate
    assert certificate["diagnostic_only"] is True


def test_one_bar_delay_stays_within_each_symbol():
    df = _frame(8).copy()
    df["symbol"] = ["A", "B"] * 4
    df["_symbol_bar_index"] = [0, 0, 1, 1, 2, 2, 3, 3]

    assert next_row_indices(df, 1).tolist() == [2, 3, 4, 5, 6, 7, -1, -1]
    assert shift_signal_mask(df, np.array([True] * len(df)), 1).tolist() == [
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
    ]


def test_delayed_entries_move_the_entry_row_by_one_bar():
    df = _frame()
    engine = _engine(df)
    (normal, normal_logs) = engine.simulate_rule_set(_rules()[0:1], return_logs=True)
    (delayed, delayed_logs) = simulate_delayed_entries(
        engine, _rules()[0:1], delay_bars=1, return_logs=True
    )

    assert delayed["executed_trades"] <= normal["executed_trades"]
    if not normal_logs.empty and not delayed_logs.empty:
        assert delayed_logs["Entry_Index"].min() == normal_logs["Entry_Index"].min() + 1


def test_regime_certificate_reports_metrics_and_concentration():
    df = _frame()
    certificate = regime_robustness_certificate(
        df.iloc[:32].copy(), df.iloc[32:].copy(), _rules(), direction="long"
    )

    assert {"train", "validation"} <= set(certificate["splits"])
    for split in certificate["splits"].values():
        assert {"high_vol", "low_vol", "trend", "range"} <= set(split["regimes"])
        assert "hhi" in split
        assert "concentration_warning" in split
    assert certificate["report_only"] is True


def test_rule_dropout_reports_worst_median_and_dependency():
    certificate = rule_dropout_stress(_engine(_frame()), _rules())

    assert len(certificate["per_rule"]) == 2
    assert "worst_dropout" in certificate
    assert "median_dropout" in certificate
    assert "single_rule_dependency" in certificate
    assert certificate["report_only"] is True


def test_rule_dropout_worst_is_largest_performance_loss():
    rules = [{"rule_id": "r1"}, {"rule_id": "r2"}]
    certificate = rule_dropout_stress(_DropoutMetricsEngine(), rules)

    assert certificate["worst_dropout"]["rule_id"] == "r1"
    assert certificate["worst_return_delta_pct"] == 7.0
    assert certificate["median_return_delta_pct"] == 4.5


def test_reporter_surfaces_each_certificate(tmp_path):
    df = _frame()
    certificates = {
        "cost_stress": cost_stress_certificate(_engine(df), _engine(df), _rules()),
        "execution_stress": execution_stress_certificate(_engine(df), _rules()),
        "regime_robustness": regime_robustness_certificate(
            df.iloc[:32].copy(), df.iloc[32:].copy(), _rules()
        ),
        "rule_dropout_stress": rule_dropout_stress(_engine(df), _rules()),
    }
    paths = Reporter().write_robustness_certificates(
        certificates, output_dir=str(tmp_path)
    )

    assert (tmp_path / "cost_stress.json").exists()
    assert (tmp_path / "execution_stress.json").exists()
    assert (tmp_path / "regime_robustness.json").exists()
    assert (tmp_path / "rule_dropout_stress.json").exists()
    assert (tmp_path / "robustness_certificates.json").exists()
    assert set(paths) >= {
        "cost_stress",
        "execution_stress",
        "regime_robustness",
        "rule_dropout_stress",
    }


def test_robustness_defaults_are_report_only():
    assert config.RB_COST_STRESS_ENABLED is True
    assert config.RB_EXECUTION_STRESS_ENABLED is True
    assert config.RB_REGIME_ROBUSTNESS_ENABLED is True
    assert config.RB_RULE_DROPOUT_STRESS_ENABLED is True
    assert config.RB_ROBUSTNESS_REPORT_ONLY is True
    assert config.RB_COST_STRESS_REPORT_ONLY is False
    assert config.RB_COST_STRESS_HARD_GATE is True
    assert config.SPREAD_BPS > 0.0
    assert config.SLIPPAGE_BPS > 0.0
