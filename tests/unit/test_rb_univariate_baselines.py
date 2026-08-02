"""Tests for deterministic RB one-condition symbol baselines."""

from __future__ import annotations

import pandas as pd

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.rb_governor import (
    CandidateRecord,
    _diversification_shortlist,
    _is_recency_good,
    _symbol_specialized_variants,
    _univariate_baseline_pool,
)


def test_univariate_baselines_cover_pool_feature_modes_and_symbols(monkeypatch):
    pool = [
        {
            "conditions": [
                "[ff_signed] IS Bullish",
                "[ff_positive] IS High",
            ],
        },
    ]
    frame = pd.DataFrame({"symbol": ["BTCUSDT", "ETHUSDT"]})
    monkeypatch.setattr(_cfg, "RB_UNIVARIATE_BASELINE_ENABLED", True)
    monkeypatch.setattr(_cfg, "RB_UNIVARIATE_BASELINE_MAX_RULES", 100)
    monkeypatch.setattr(_cfg, "RB_UNIVARIATE_GENERALIST_ENABLED", False)

    augmented, added = _univariate_baseline_pool(pool, frame)
    conditions = {
        tuple(row["conditions"])
        for row in augmented[1:]
    }

    assert added == 30  # 10 signed + 5 positive states, for two symbols.
    assert (
        "[ff_signed] IS Extreme Bearish", "symbol is BTCUSDT"
    ) in conditions
    assert (
        "[ff_positive] IS Very High", "symbol is ETHUSDT"
    ) in conditions


def test_univariate_baselines_reserve_generalist_conditions_first(monkeypatch):
    pool = [{"conditions": ["[ff_signed] IS Bullish"]}]
    frame = pd.DataFrame({"symbol": ["BTCUSDT", "ETHUSDT"]})
    monkeypatch.setattr(_cfg, "RB_UNIVARIATE_BASELINE_ENABLED", True)
    monkeypatch.setattr(_cfg, "RB_UNIVARIATE_GENERALIST_ENABLED", True)
    monkeypatch.setattr(_cfg, "RB_UNIVARIATE_BASELINE_MAX_RULES", 9)

    augmented, added = _univariate_baseline_pool(pool, frame)

    assert added == 9
    assert all(
        not any(str(c).startswith("symbol is ") for c in row["conditions"])
        for row in augmented[1:]
    )


def test_recency_certificate_requires_both_validation_halves(monkeypatch):
    monkeypatch.setattr(_cfg, "RB_RECENCY_RESCUE_ENABLED", True)

    def metrics(return_pct: float) -> dict:
        return {
            "total_return_pct": return_pct,
            "profit_factor": 1.2,
            "max_drawdown_pct": 5.0,
            "executed_trades": 30,
            "per_symbol_metrics": {
                "BTCUSDT": {"trade_count": 15, "net_pnl": 2.0},
                "ETHUSDT": {"trade_count": 15, "net_pnl": 3.0},
            },
            "per_symbol_metrics_available": True,
        }

    assert _is_recency_good(metrics(-5.0), metrics(1.0), metrics(0.8))
    assert not _is_recency_good(metrics(-5.0), metrics(-0.1), metrics(0.8))


def test_explicit_symbol_baseline_is_not_broadened_in_generalist_mode(monkeypatch):
    rule = {
        "conditions": ["[ff_signed] IS Bullish", "symbol is BTCUSDT"],
        "tp": 2.0,
        "sl": 1.2,
        "capital_pct": 12.5,
    }
    monkeypatch.setattr(_cfg, "RB_REQUIRE_SYMBOL_FILTERS", False)

    variants = _symbol_specialized_variants(rule, None, None, ["BTCUSDT"])

    assert variants == [rule]


def test_diversification_shortlist_keeps_symbol_return_leader(monkeypatch):
    def record(name: str, score: float, valid_return: float) -> CandidateRecord:
        metrics = {
            "total_return_pct": valid_return,
            "per_symbol_metrics": {
                "BTCUSDT": {"trade_count": 20, "net_pnl": valid_return},
            },
        }
        return CandidateRecord(
            rule={"conditions": [f"[ff_x] IS {name}"]},
            train_metrics={"total_return_pct": valid_return},
            valid_metrics=metrics,
            score=score,
        )

    score_leader = record("Low", 100.0, 1.0)
    return_leader = record("High", -500.0, 5.0)
    monkeypatch.setattr(_cfg, "RB_DIVERSIFICATION_GLOBAL_LEADERS", 1)
    monkeypatch.setattr(_cfg, "RB_DIVERSIFICATION_SYMBOL_LEADERS", 1)
    monkeypatch.setattr(_cfg, "RB_DIVERSIFICATION_RETURN_LEADERS", 1)

    shortlist = _diversification_shortlist([score_leader, return_leader])

    assert score_leader in shortlist
    assert return_leader in shortlist
