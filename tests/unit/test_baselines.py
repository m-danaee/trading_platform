"""Tests for train and frozen-validation baseline reporting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gpu_fuzzy_trader.validation.baselines import write_baseline_reports


def _frame(rows: int = 120) -> pd.DataFrame:
    return pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=rows, freq="15min"),
        "symbol": "BTCUSDT",
        "feature": np.ones(rows),
        "label_open_next": np.full(rows, 100.0),
        "label_close_288": np.full(rows, 101.0),
        "label_min_288": np.full(rows, 99.0),
        "label_max_288": np.full(rows, 102.0),
        "label_max_before_min": np.ones(rows),
    })


def test_baseline_report_keeps_train_and_adds_validation_only(tmp_path):
    strategy = {
        "direction": "long",
        "strategy_id": "test-strategy",
        "rules_set": [{
            "conditions": ["[feature] >= 0"],
            "tp": 2.0,
            "sl": 1.0,
            "capital_pct": 10.0,
        }],
    }

    reports = write_baseline_reports(
        str(tmp_path),
        _frame(),
        {"long": strategy},
        validation_frame=_frame(),
    )

    report = reports["long"]
    assert "fixed_phase2_exit" in report
    assert report["validation_contract"].endswith("not_used_for_selection")
    assert report["validation"]["direction"] == "long"
    assert (tmp_path / "reports" / "baseline_long.json").exists()
