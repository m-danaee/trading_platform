"""Tests for report-only trade uncertainty diagnostics."""

from __future__ import annotations

import pandas as pd

from gpu_fuzzy_trader.validation.uncertainty import compute_trade_uncertainty


def _trade_log() -> pd.DataFrame:
    return pd.DataFrame({
        "Net_PnL": [10.0, -5.0, 8.0, -2.0],
        "Equity_Before_Entry": [1000.0, 1010.0, 1005.0, 1013.0],
        "Realized": [True, True, True, True],
    })


def test_no_trades_are_explicitly_reported():
    result = compute_trade_uncertainty(pd.DataFrame())

    assert result["status"] == "no_realized_trades"
    assert result["trade_count"] == 0
    assert result["sign_test_p_value"] == 1.0


def test_bootstrap_is_deterministic_and_has_audit_fields():
    first = compute_trade_uncertainty(
        _trade_log(), seed=17, samples=80, block_length=2,
    )
    second = compute_trade_uncertainty(
        _trade_log(), seed=17, samples=80, block_length=2,
    )

    assert first == second
    assert first["status"] == "diagnostic"
    assert first["trade_count"] == 4
    assert first["bootstrap_samples"] == 80
    assert first["block_length"] == 2
    assert len(first["mean_trade_return_ci95_pct"]) == 2
    assert len(first["compound_return_ci95_pct"]) == 2


def test_zero_return_trades_are_not_counted_by_sign_test():
    log = pd.DataFrame({
        "Net_PnL": [1.0, 0.0, -1.0],
        "Equity_Before_Entry": [100.0, 100.0, 100.0],
    })

    result = compute_trade_uncertainty(log, samples=0)

    assert result["trade_count"] == 3
    assert result["sign_test_p_value"] == 1.0
