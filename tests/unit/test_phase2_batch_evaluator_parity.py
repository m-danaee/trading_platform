"""Regression tests for evaluator-facing Phase 2 chromosome semantics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gpu_fuzzy_trader.backtest.cpu_engine import (
    CPUBacktestEngine,
    _build_rule_signal_mask,
)
from gpu_fuzzy_trader.features.encoder import decode_chromosome


def _backtest_df(feature_values: list[float]) -> pd.DataFrame:
    n_rows = len(feature_values)
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=n_rows, freq="5min"),
            "symbol": ["SYM"] * n_rows,
            "label_open_next": [100.0] * n_rows,
            "label_max_288": [103.0] * n_rows,
            "label_min_288": [98.0] * n_rows,
            "label_close_288": [101.0] * n_rows,
            "label_max_before_min": [1] * n_rows,
            "feature": feature_values,
        }
    )


def test_batch_chromosome_signals_match_decoded_rule_conditions() -> None:
    """Search fitness must use the same fuzzy class as RB/Phase 5 evaluation."""
    df = _backtest_df([0.1, 0.3, 0.5, 0.7, 0.9])
    feature_infos = [{"name": "feature", "mode": "positive"}]
    conditions = decode_chromosome(np.array([3], dtype=np.int32), feature_infos)
    direct_signals = _build_rule_signal_mask(df, conditions)

    engine = CPUBacktestEngine(df, {"feature": "positive"}, "long")
    batch_metrics = engine.simulate_rule_batch(
        np.array([[3]], dtype=np.int32),
        tp=2.0,
        sl=1.2,
        capital_pct=10.0,
    )[0]

    assert conditions == ["[feature] IS High"]
    assert int(batch_metrics["raw_signal_count"]) == int(direct_signals.sum())
