"""Tests for RB concentration / tail-holdout hard fail-closed behaviour.

When symbol-concentration or tail-holdout gates fail, the RB Governor must
write an empty strategy with ``deployment_accepted=false`` and
``fail_closed=true`` (same pattern as insufficient_distinct_symbols).
Return/PF-only soft failures still retain rules with deployment_accepted=False.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.output.writer import Output_Writer
from gpu_fuzzy_trader.rb_governor import CandidateRecord, run_rb_governor_pipeline


def _rule(sym: str = "BTC") -> dict:
    ctx = list(_cfg.mandatory_context_conditions("long"))
    return {
        "conditions": [f"symbol is {sym}", "[feat] IS High", *ctx],
        "tp": 2.0,
        "sl": 1.2,
        "capital_pct": 10.0,
    }


def _dummy_df(size: int = 100) -> pd.DataFrame:
    np.random.seed(42)
    symbols = [f"sym_{i}" for i in range(5)]
    entry = np.random.uniform(1.0, 2.0, size=size)
    return pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=size, freq="5min"),
        "symbol": np.random.choice(symbols, size=size),
        "label_open_next": entry,
        "label_close_288": entry * np.random.uniform(0.9, 1.1, size=size),
        "label_min_288": entry * np.random.uniform(0.8, 0.99, size=size),
        "label_max_288": entry * np.random.uniform(1.01, 1.2, size=size),
        "label_max_before_min": np.random.choice([0, 1], size=size),
        "feat_1": np.random.randn(size),
        "feat_2": np.random.randn(size),
    })


def _mock_metrics(**kwargs) -> dict:
    return {
        "total_return_pct": kwargs.get("return_pct", 5.0),
        "max_drawdown_pct": kwargs.get("dd", 2.0),
        "profit_factor": kwargs.get("pf", 1.5),
        "win_rate": kwargs.get("wr", 55.0),
        "executed_trades": kwargs.get("trades", 30),
        "raw_signal_count": kwargs.get("raw", 40),
        "skipped_min_notional_count": kwargs.get("skipped", 2),
        "max_simultaneous_positions": kwargs.get("max_pos", 2),
        "per_symbol_metrics": kwargs.get("per_symbol", {}),
    }


def _candidates(rules: list[dict]) -> list[CandidateRecord]:
    return [
        CandidateRecord(
            rule=dict(rule),
            train_metrics=_mock_metrics(),
            valid_metrics=_mock_metrics(return_pct=3.0),
            score=50.0,
            mask=np.ones(200, dtype=bool),
        )
        for rule in rules
    ]


def _run_pipeline(
    *,
    tmpdir: str,
    rules: list[dict],
    sym_ok: bool,
    tail_ok: bool,
    valid_return_pct: float = 5.0,
    valid_pf: float = 1.5,
):
    train_df = _dummy_df(100)
    val_df = _dummy_df(100)
    pools = {"long": rules}
    valid_m = _mock_metrics(return_pct=valid_return_pct, pf=valid_pf)
    train_m = _mock_metrics()

    with patch.object(_cfg, "RB_REQUIRE_SYMBOL_FILTERS", False), patch.object(
        _cfg, "RB_COST_STRESS_HARD_GATE", False,
    ), patch.object(
        _cfg, "RB_COST_STRESS_REPORT_ONLY", True,
    ), patch(
        "gpu_fuzzy_trader.rb_governor._filter_good_rules",
        return_value=_candidates(rules),
    ), patch(
        "gpu_fuzzy_trader.rb_governor._compose_ruleset",
        return_value=(
            _candidates(rules),
            train_m,
            valid_m,
            50.0,
            [{"step": 1, "action": "seed", "score": 50.0}],
        ),
    ), patch(
        "gpu_fuzzy_trader.rb_governor._optimize_risk",
        return_value=(rules, train_m, valid_m, 50.0,
                      [{"tail_return_pct": 1.0}]),
    ), patch(
        "gpu_fuzzy_trader.rb_governor._run_profit_amplifier",
        return_value=(rules, train_m, valid_m, 50.0, {"accepted": False}),
    ), patch(
        "gpu_fuzzy_trader.rb_governor.CPUBacktestEngine",
    ), patch(
        "gpu_fuzzy_trader.rb_governor._make_walk_forward_fold_engines",
        return_value=([], None),
    ), patch(
        "gpu_fuzzy_trader.rb_governor._write_clean_evaluator",
    ), patch(
        "gpu_fuzzy_trader.rb_governor._passes_symbol_concentration_gate",
        return_value=(
            sym_ok,
            {
                "passed": sym_ok,
                "top_symbol_share_abs_pnl": 0.9 if not sym_ok else 0.2,
                "hhi_abs_pnl": 0.8 if not sym_ok else 0.3,
                "top_symbol": "BTC",
            },
        ),
    ), patch(
        "gpu_fuzzy_trader.rb_governor._passes_tail_holdout_gate",
        return_value=(
            tail_ok,
            {
                "passed": tail_ok,
                "tail_return_pct": -1.0 if not tail_ok else 1.0,
                "min_return_pct": 0.0,
            },
        ),
    ):
        return run_rb_governor_pipeline(
            train_df,
            val_df,
            pools,
            ("long",),
            output_dir=tmpdir,
        )


class TestConcentrationTailFailClosed:
    def test_concentration_fails_writes_empty_strategy(self):
        rules = [_rule("BTC"), _rule("ETH")]
        with tempfile.TemporaryDirectory() as tmpdir:
            results = _run_pipeline(
                tmpdir=tmpdir, rules=rules, sym_ok=False, tail_ok=True,
            )
            strategy = results["long"]
            assert strategy["rules_set"] == []
            assert strategy["deployment_accepted"] is False
            assert strategy.get("reason") == "symbol_concentration"

            with open(Path(tmpdir) / "long.json") as fh:
                saved = json.load(fh)
            assert saved["rules_set"] == []
            assert saved["deployment_accepted"] is False

            with open(Path(tmpdir) / "reports" / "rb_governor_long_report.json") as fh:
                report = json.load(fh)
            assert report["fail_closed"] is True
            assert report["fail_closed_reason"] == "symbol_concentration"
            assert report["selected_rules"] == 0

    def test_tail_fails_writes_empty_strategy(self):
        rules = [_rule("BTC"), _rule("ETH")]
        with tempfile.TemporaryDirectory() as tmpdir:
            results = _run_pipeline(
                tmpdir=tmpdir, rules=rules, sym_ok=True, tail_ok=False,
            )
            strategy = results["long"]
            assert strategy["rules_set"] == []
            assert strategy["deployment_accepted"] is False
            assert strategy.get("reason") == "tail_holdout"

            with open(Path(tmpdir) / "reports" / "rb_governor_long_report.json") as fh:
                report = json.load(fh)
            assert report["fail_closed"] is True
            assert report["fail_closed_reason"] == "tail_holdout"

    def test_both_fail_combined_reason(self):
        rules = [_rule("BTC")]
        with tempfile.TemporaryDirectory() as tmpdir:
            results = _run_pipeline(
                tmpdir=tmpdir, rules=rules, sym_ok=False, tail_ok=False,
            )
            assert results["long"].get(
                "reason") == "symbol_concentration+tail_holdout"
            with open(Path(tmpdir) / "reports" / "rb_governor_long_report.json") as fh:
                report = json.load(fh)
            assert report["fail_closed_reason"] == "symbol_concentration+tail_holdout"

    def test_gates_pass_keeps_rules_and_accepts(self):
        rules = [_rule("BTC"), _rule("ETH"), _rule("SOL")]
        with tempfile.TemporaryDirectory() as tmpdir:
            results = _run_pipeline(
                tmpdir=tmpdir,
                rules=rules,
                sym_ok=True,
                tail_ok=True,
                valid_return_pct=5.0,
                valid_pf=1.5,
            )
            strategy = results["long"]
            assert len(strategy["rules_set"]) > 0
            assert strategy["deployment_accepted"] is True
            assert strategy.get("reason") is None

            with open(Path(tmpdir) / "reports" / "rb_governor_long_report.json") as fh:
                report = json.load(fh)
            assert report.get("fail_closed") is not True

    def test_return_pf_soft_fail_keeps_rules(self):
        """Return/PF below gate but sym+tail OK → rules retained, not accepted."""
        rules = [_rule("BTC"), _rule("ETH")]
        with tempfile.TemporaryDirectory() as tmpdir:
            results = _run_pipeline(
                tmpdir=tmpdir,
                rules=rules,
                sym_ok=True,
                tail_ok=True,
                valid_return_pct=0.1,  # below PHASE5_VALIDATION_RETURN_GATE_PCT=2.0
                valid_pf=1.5,
            )
            strategy = results["long"]
            assert len(strategy["rules_set"]) > 0
            assert strategy["deployment_accepted"] is False
            assert strategy.get("reason") is None

    def test_fail_closed_strategy_loadable(self):
        rules = [_rule("BTC")]
        with tempfile.TemporaryDirectory() as tmpdir:
            _run_pipeline(tmpdir=tmpdir, rules=rules,
                          sym_ok=False, tail_ok=True)
            strategy_path = Path(tmpdir) / "long.json"
            with open(strategy_path) as fh:
                raw = json.load(fh)
            assert raw["rules_set"] == []
            assert raw["deployment_accepted"] is False
            Output_Writer().load_and_validate(strategy_path)
