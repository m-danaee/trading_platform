"""Focused regressions for certificate-first RB and Phase 2 selection."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.phases.phase2_rule_pool import (
    _build_pool_from_archive,
    _reserve_symbol_pool_candidates,
)
from gpu_fuzzy_trader.rb_governor import (
    CandidateRecord,
    _compose_ruleset,
    _portfolio_selection_certificate,
)


def _metrics(
    per_symbol: dict[str, dict],
    *,
    return_pct: float = 5.0,
) -> dict:
    return {
        "total_return_pct": return_pct,
        "profit_factor": 1.5,
        "max_drawdown_pct": 2.0,
        "executed_trades": 30,
        "per_symbol_metrics": per_symbol,
    }


def _candidate(name: str, score: float, symbol: str, mask: list[bool]) -> CandidateRecord:
    return CandidateRecord(
        rule={
            "conditions": [f"[feat] IS {name}"],
            "tp": 2.0,
            "sl": 1.0,
            "capital_pct": 10.0,
        },
        train_metrics=_metrics({
            symbol: {"trade_count": 10, "net_pnl": 8.0, "win_rate": 55.0},
        }),
        valid_metrics=_metrics({
            symbol: {"trade_count": 10, "net_pnl": 8.0, "win_rate": 55.0},
        }),
        score=score,
        mask=np.asarray(mask, dtype=bool),
    )


def test_certificate_rejects_eth_only_and_accepts_balanced_team():
    eth_only = _metrics({
        "ETHUSDT": {"trade_count": 12, "net_pnl": 10.0, "win_rate": 55.0},
    })
    balanced = _metrics({
        "ETHUSDT": {"trade_count": 12, "net_pnl": 10.0, "win_rate": 55.0},
        "BTCUSDT": {"trade_count": 8, "net_pnl": 9.0, "win_rate": 52.0},
    })

    with patch.object(_cfg, "RB_MIN_DISTINCT_SYMBOLS", 2):
        failed, failure = _portfolio_selection_certificate(eth_only)
        passed, success = _portfolio_selection_certificate(balanced)

    assert not failed
    assert "symbol_contribution" in failure["reasons"]
    assert passed
    assert success["symbol_contribution"]["qualifying_symbols"] == [
        "BTCUSDT",
        "ETHUSDT",
    ]


def test_compose_uses_balanced_beam_seed_instead_of_eth_leader():
    eth = _candidate("ETH", 100.0, "ETHUSDT", [True, False])
    btc = _candidate("BTC", 70.0, "BTCUSDT", [False, True])

    def evaluate(_train_engine, _valid_engine, rules):
        names = {str(rule["conditions"][0]).split()[-1] for rule in rules}
        if names == {"ETH"}:
            per = {
                "ETHUSDT": {
                    "trade_count": 12,
                    "net_pnl": 10.0,
                    "win_rate": 55.0,
                },
            }
            return _metrics(per), _metrics(per), 100.0
        if names == {"BTC"}:
            per = {
                "BTCUSDT": {
                    "trade_count": 12,
                    "net_pnl": 9.0,
                    "win_rate": 52.0,
                },
            }
            return _metrics(per), _metrics(per), 70.0
        per = {
            "ETHUSDT": {
                "trade_count": 12,
                "net_pnl": 10.0,
                "win_rate": 55.0,
            },
            "BTCUSDT": {
                "trade_count": 12,
                "net_pnl": 9.0,
                "win_rate": 52.0,
            },
        }
        return _metrics(per, return_pct=8.0), _metrics(per, return_pct=8.0), 140.0

    with patch.object(_cfg, "RB_MIN_DISTINCT_SYMBOLS", 2), patch.object(
        _cfg, "RB_MAX_RULES", 2
    ), patch.object(_cfg, "RB_DIVERSIFICATION_BEAM_WIDTH", 6), patch.object(
        _cfg, "RB_DIVERSIFICATION_STEPS", 4
    ), patch.object(
        _cfg, "RB_RULESET_MUST_BEAT_SUBSETS", False
    ), patch.object(
        _cfg, "RB_RULE_ADD_IGNORE_OVERLAP", False
    ), patch.object(
        _cfg, "RB_MAX_PAIR_OVERLAP", 0.35
    ), patch(
        "gpu_fuzzy_trader.rb_governor._evaluate_ruleset",
        side_effect=evaluate,
    ), patch(
        "gpu_fuzzy_trader.rb_governor._is_positive_good",
        return_value=True,
    ):
        selected, _, valid, _, history = _compose_ruleset(
            [eth, btc], object(), object(), "long",
        )

    assert {record.rule["conditions"][0] for record in selected} == {
        "[feat] IS ETH",
        "[feat] IS BTC",
    }
    assert _portfolio_selection_certificate(valid)[0]
    assert any(
        item["action"] == "diversification_certificate_passed"
        for item in history
    )


def _pool_entry(
    chromosome: list[int],
    symbol: str,
    net_pnl: float,
    *,
    admission_passed: bool = True,
) -> dict:
    return {
        "chromosome": chromosome,
        "conditions": ["[feat] IS High"],
        "objectives": {
            "sortino_ratio": 2.0,
            "total_return_pct": 5.0,
            "profit_factor": 1.5,
            "max_drawdown_pct": 2.0,
            "win_rate": 55.0,
        },
        "executed_trades": 30,
        "val_objectives": {
            "sortino_ratio": 2.0,
            "total_return_pct": 4.0,
            "profit_factor": 1.5,
            "max_drawdown_pct": 2.0,
            "win_rate": 55.0,
        },
        "val_executed_trades": 12,
        "valid_per_symbol_metrics": {
            symbol: {"trade_count": 8, "net_pnl": net_pnl, "win_rate": 55.0},
        },
        "admission_passed": admission_passed,
    }


def test_pool_reservation_keeps_positive_btc_and_eth_only():
    pool = [
        _pool_entry([1], "ETHUSDT", 10.0),
        _pool_entry([2], "BTCUSDT", 9.0),
        _pool_entry([3], "BTCUSDT", 20.0, admission_passed=False),
    ]
    report: dict = {}

    with patch.object(_cfg, "PHASE2_MAX_RESERVED_RULES_PER_SYMBOL", 1):
        retained = _reserve_symbol_pool_candidates(
            pool,
            keep_top=2,
            coverage_report=report,
        )

    assert {tuple(entry["chromosome"]) for entry in retained} == {(1,), (2,)}
    assert report["reservation_counts"] == {"BTCUSDT": 1, "ETHUSDT": 1}


class _BatchEngine:
    def __init__(self, metrics):
        self.metrics = metrics

    def simulate_rule_batch(self, **_kwargs):
        return [self.metrics]


def test_phase2_cpu_reevaluation_overrides_stale_gpu_metrics():
    train_metrics = _metrics({
        "BTCUSDT": {"trade_count": 10, "net_pnl": 8.0, "win_rate": 55.0},
        "ETHUSDT": {"trade_count": 10, "net_pnl": 8.0, "win_rate": 55.0},
    })
    valid_metrics = _metrics({
        "BTCUSDT": {"trade_count": 8, "net_pnl": 7.0, "win_rate": 55.0},
        "ETHUSDT": {"trade_count": 8, "net_pnl": 7.0, "win_rate": 55.0},
    })
    stale_gpu_metrics = _metrics({
        "ETHUSDT": {"trade_count": 30, "net_pnl": 100.0, "win_rate": 80.0},
    })
    feature_infos = [
        {"name": f"feat_{idx}", "mode": "binary", "score": 0.5}
        for idx in range(4)
    ]
    archive = [np.zeros(4, dtype=np.int32)]
    dont_cares = np.full(4, 2, dtype=np.int32)
    report: dict = {}

    pool = _build_pool_from_archive(
        archive,
        feature_infos,
        dont_cares,
        object(),
        metrics_by_chrom={(0, 0, 0, 0): stale_gpu_metrics},
        val_engine=object(),
        cpu_engine=_BatchEngine(train_metrics),
        cpu_val_engine=_BatchEngine(valid_metrics),
        direction="long",
        coverage_report=report,
    )

    assert len(pool) == 1
    assert pool[0]["val_per_symbol_metrics"] == valid_metrics["per_symbol_metrics"]
    assert pool[0]["val_per_symbol_metrics"] != stale_gpu_metrics["per_symbol_metrics"]
    assert report["cpu_reevaluation"] is True
    assert report["cpu_evaluated"] == 1
