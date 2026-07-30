"""Focused tests for the shared positive-good admission contract."""

from __future__ import annotations

from gpu_fuzzy_trader.scoring.gates import (
    PositiveGoodThresholds,
    gate_positive_good,
    positive_good_reject_reasons,
)


def _metrics(*, ret: float = 2.0, pf: float = 1.2, trades: int = 20) -> dict:
    return {
        "total_return_pct": ret,
        "profit_factor": pf,
        "executed_trades": trades,
        "raw_signal_count": trades,
        "skipped_min_notional_count": 0,
    }


def test_shared_gate_accepts_metrics_that_meet_both_split_floors() -> None:
    thresholds = PositiveGoodThresholds(
        min_train_return=1.0,
        min_valid_return=0.5,
        min_train_profit_factor=1.1,
        min_valid_profit_factor=1.05,
        min_train_trades=15,
        min_valid_trades=10,
    )
    assert gate_positive_good(_metrics(), _metrics(ret=1.0), thresholds) is True


def test_shared_gate_reports_each_failed_threshold() -> None:
    thresholds = PositiveGoodThresholds(
        min_train_return=3.0,
        min_valid_return=3.0,
        min_train_profit_factor=1.5,
        min_valid_profit_factor=1.5,
        min_train_trades=30,
        min_valid_trades=30,
    )
    reasons = positive_good_reject_reasons(
        _metrics(ret=1.0, pf=1.1, trades=5),
        _metrics(ret=-1.0, pf=0.8, trades=4),
        thresholds,
    )
    assert reasons == [
        "train_return_floor",
        "valid_return_floor",
        "train_profit_factor_floor",
        "valid_profit_factor_floor",
        "train_trade_floor",
        "valid_trade_floor",
    ]
    assert gate_positive_good(
        _metrics(ret=1.0, pf=1.1, trades=5),
        _metrics(ret=-1.0, pf=0.8, trades=4),
        thresholds,
    ) is False


def test_missing_validation_metrics_fail_closed() -> None:
    thresholds = PositiveGoodThresholds(min_valid_return=0.1, min_valid_trades=1)
    assert gate_positive_good(_metrics(), None, thresholds) is False
    assert "valid_return_floor" in positive_good_reject_reasons(
        _metrics(), None, thresholds
    )


def test_execution_health_is_an_explicit_optional_gate() -> None:
    thresholds = PositiveGoodThresholds(
        min_train_trades=1,
        min_valid_trades=1,
        require_execution_health=True,
    )
    bad = _metrics(trades=50)
    bad.update(raw_signal_count=100, skipped_min_notional_count=80)
    assert gate_positive_good(bad, bad, thresholds) is False
    reasons = positive_good_reject_reasons(bad, bad, thresholds)
    assert "train_execution_health" in reasons
    assert "valid_execution_health" in reasons


def test_legacy_keyword_aliases_preserve_metric_semantics() -> None:
    assert gate_positive_good(
        _metrics(),
        _metrics(ret=1.0),
        min_train_return=1.0,
        min_val_return=0.5,
        min_train_pf=1.1,
        min_val_pf=1.1,
        min_train_trades=20,
        min_val_trades=20,
    ) is True
