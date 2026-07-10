"""Tests for RB Governor min-distinct-symbols hard gate.

After final opt_rules (post risk opt + profit amp), if
``RB_REQUIRE_SYMBOL_FILTERS`` is True and ``RB_MIN_DISTINCT_SYMBOLS > 0``
and ``len(_symbols_in_rules(opt_rules)) < RB_MIN_DISTINCT_SYMBOLS``,
the ruleset must be replaced with an empty strategy and
``deployment_accepted=false`` with reason ``insufficient_distinct_symbols``.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.output.writer import Output_Writer
from gpu_fuzzy_trader.rb_governor import (
    _symbols_in_rules,
    run_rb_governor_pipeline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _single_symbol_rule(sym: str) -> dict:
    return {
        "conditions": [f"symbol is {sym}", "[feat] IS High"],
        "tp": 2.0,
        "sl": 1.2,
        "capital_pct": 10.0,
    }


def _multi_symbol_rules() -> list[dict]:
    return [_single_symbol_rule(f"sym_{i}") for i in range(6)]


def _no_symbol_rule() -> dict:
    return {
        "conditions": ["[feat] IS High"],
        "tp": 2.0,
        "sl": 1.2,
        "capital_pct": 18.0,
    }


def _dummy_df(size: int = 100) -> pd.DataFrame:
    """Return a minimal DataFrame that passes _prepare_scoring_frame."""
    np.random.seed(42)
    symbols = [f"sym_{i}" for i in range(5)]
    entry = np.random.uniform(1.0, 2.0, size=size)
    df = pd.DataFrame({
        "datetime": pd.date_range("2020-01-01", periods=size, freq="5min"),
        "symbol": np.random.choice(symbols, size=size),
        "label_open_next": entry,
        "label_close_288": entry * np.random.uniform(0.9, 1.1, size=size),
        "label_min_288": entry * np.random.uniform(0.8, 0.99, size=size),
        "label_max_288": entry * np.random.uniform(1.01, 1.2, size=size),
        "label_max_before_min": np.random.choice([0, 1], size=size),
        "feat_1": np.random.randn(size),
        "feat_2": np.random.randn(size),
    })
    return df


def _mock_train_metrics(**kwargs) -> dict:
    """Return plausible training metrics for CPUBacktestEngine.simulate_rule_set."""
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


def _make_candidate_records(rules: list[dict]):
    """Build minimal CandidateRecord-like objects from rules.

    We don't actually simulate; we just return objects that behave enough
    for the pipeline to pass through patched paths.
    """
    from gpu_fuzzy_trader.rb_governor import CandidateRecord
    records = []
    for rule in rules:
        record = CandidateRecord(
            rule=dict(rule),
            train_metrics=_mock_train_metrics(),
            valid_metrics=_mock_train_metrics(return_pct=3.0),
            score=50.0,
            mask=np.ones(200, dtype=bool),
        )
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# Tests for _symbols_in_rules helper
# ---------------------------------------------------------------------------


class TestSymbolsInRules:
    """Verify _symbols_in_rules extracts symbols correctly."""

    def test_single_symbol_rule(self):
        rules = [_single_symbol_rule("BTC")]
        assert _symbols_in_rules(rules) == {"btc"}

    def test_multi_symbol_rules(self):
        rules = [_single_symbol_rule("BTC"), _single_symbol_rule("ETH")]
        assert _symbols_in_rules(rules) == {"btc", "eth"}

    def test_no_symbol_rule(self):
        rules = [_no_symbol_rule()]
        assert _symbols_in_rules(rules) == set()

    def test_empty_rules(self):
        assert _symbols_in_rules([]) == set()

    def test_bracket_symbol_condition(self):
        rules = [{"conditions": ["[symbol] is BTC", "[feat] IS High"]}]
        assert _symbols_in_rules(rules) == {"btc"}

    def test_mixed_conditions(self):
        rules = [
            {"conditions": ["symbol is BTC", "[feat] IS High"]},
            {"conditions": ["[symbol] is ETH", "[feat] IS Low"]},
        ]
        assert _symbols_in_rules(rules) == {"btc", "eth"}


# ---------------------------------------------------------------------------
# Tests for the min-distinct-symbols gate (via pipeline)
# ---------------------------------------------------------------------------


class TestMinDistinctSymbolsGate:
    """Tests that the gate triggers correctly in run_rb_governor_pipeline."""

    def test_filters_on_1_symbol_min_5_fails_closed(self):
        """Filters on + 1-symbol team + min=5 → empty rules, reason set.

        Given:
        - RB_REQUIRE_SYMBOL_FILTERS=True
        - RB_MIN_DISTINCT_SYMBOLS=5
        - opt_rules after compose/risk/profit contain only 1 symbol
        Then:
        - strategy has deployment_accepted=False
        - reason is insufficient_distinct_symbols
        - n_symbols and required are in extra/report
        """
        train_df = _dummy_df(100)
        val_df = _dummy_df(100)
        pools = {"long": [_single_symbol_rule("BTC")]}

        # One-symbol rules to feed into compose → risk → profit
        # The final opt_rules will have only 1 symbol
        one_sym_rules = [_single_symbol_rule("BTC")]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(
                _cfg, "RB_REQUIRE_SYMBOL_FILTERS", True,
            ), patch.object(
                _cfg, "RB_MIN_DISTINCT_SYMBOLS", 5,
            ), patch(
                "gpu_fuzzy_trader.rb_governor._filter_good_rules",
                return_value=_make_candidate_records(one_sym_rules),
            ), patch(
                "gpu_fuzzy_trader.rb_governor._compose_ruleset",
                return_value=(
                    _make_candidate_records(one_sym_rules),
                    _mock_train_metrics(),
                    _mock_train_metrics(return_pct=3.0),
                    50.0,
                    [{"step": 1, "action": "seed", "score": 50.0}],
                ),
            ), patch(
                "gpu_fuzzy_trader.rb_governor._optimize_risk",
                return_value=(one_sym_rules, _mock_train_metrics(), _mock_train_metrics(return_pct=3.0), 50.0, []),
            ), patch(
                "gpu_fuzzy_trader.rb_governor._run_profit_amplifier",
                return_value=(one_sym_rules, _mock_train_metrics(), _mock_train_metrics(return_pct=3.0), 50.0, {"accepted": False}),
            ), patch(
                "gpu_fuzzy_trader.rb_governor.CPUBacktestEngine",
            ), patch(
                "gpu_fuzzy_trader.rb_governor._make_walk_forward_fold_engines",
                return_value=([], None),
            ), patch(
                "gpu_fuzzy_trader.rb_governor._write_clean_evaluator",
            ):
                results = run_rb_governor_pipeline(
                    train_df, val_df,
                    pools,
                    ("long",),
                    output_dir=tmpdir,
                )

            assert "long" in results
            strategy = results["long"]
            assert strategy["direction"] == "long"
            assert strategy["rules_set"] == []
            assert strategy["deployment_accepted"] is False
            assert strategy.get("reason") == "insufficient_distinct_symbols"
            assert strategy.get("n_symbols") == 1
            assert strategy.get("required") == 5

            # Verify the JSON file was written
            strategy_path = Path(tmpdir) / "long.json"
            assert strategy_path.exists()
            with open(strategy_path) as f:
                saved = json.load(f)
            assert saved["rules_set"] == []
            assert saved["deployment_accepted"] is False
            assert saved.get("reason") == "insufficient_distinct_symbols"

            # Verify report was written
            report_path = Path(tmpdir) / "reports" / "rb_governor_long_report.json"
            assert report_path.exists()
            with open(report_path) as f:
                report = json.load(f)
            assert report.get("fail_closed") is True
            assert report.get("fail_closed_reason") == "insufficient_distinct_symbols"
            assert report.get("n_symbols") == 1
            assert report.get("required_symbols") == 5
            assert report["selected_rules"] == 0

    def test_filters_on_enough_symbols_passes(self):
        """Filters on + enough symbols → not rejected by this gate alone.

        Given:
        - RB_REQUIRE_SYMBOL_FILTERS=True
        - RB_MIN_DISTINCT_SYMBOLS=5
        - opt_rules span 6 distinct symbols (>= 5)
        Then:
        - strategy is NOT fail-closed by this gate
        - deployment_accepted depends on other gates but not this one
        """
        train_df = _dummy_df(100)
        val_df = _dummy_df(100)
        pools = {"long": _multi_symbol_rules()}

        six_sym_rules = _multi_symbol_rules()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(
                _cfg, "RB_REQUIRE_SYMBOL_FILTERS", True,
            ), patch.object(
                _cfg, "RB_MIN_DISTINCT_SYMBOLS", 5,
            ), patch(
                "gpu_fuzzy_trader.rb_governor._filter_good_rules",
                return_value=_make_candidate_records(six_sym_rules),
            ), patch(
                "gpu_fuzzy_trader.rb_governor._compose_ruleset",
                return_value=(
                    _make_candidate_records(six_sym_rules),
                    _mock_train_metrics(),
                    _mock_train_metrics(return_pct=3.0),
                    50.0,
                    [{"step": 1, "action": "seed", "score": 50.0}],
                ),
            ), patch(
                "gpu_fuzzy_trader.rb_governor._optimize_risk",
                return_value=(six_sym_rules, _mock_train_metrics(), _mock_train_metrics(return_pct=3.0), 50.0, []),
            ), patch(
                "gpu_fuzzy_trader.rb_governor._run_profit_amplifier",
                return_value=(six_sym_rules, _mock_train_metrics(), _mock_train_metrics(return_pct=3.0), 50.0, {"accepted": False}),
            ), patch(
                "gpu_fuzzy_trader.rb_governor.CPUBacktestEngine",
            ), patch(
                "gpu_fuzzy_trader.rb_governor._make_walk_forward_fold_engines",
                return_value=([], None),
            ), patch(
                "gpu_fuzzy_trader.rb_governor._write_clean_evaluator",
            ), patch(
                "gpu_fuzzy_trader.rb_governor._passes_symbol_concentration_gate",
                return_value=(True, {"passed": True}),
            ), patch(
                "gpu_fuzzy_trader.rb_governor._passes_tail_holdout_gate",
                return_value=(True, {"passed": True}),
            ):
                results = run_rb_governor_pipeline(
                    train_df, val_df,
                    pools,
                    ("long",),
                    output_dir=tmpdir,
                )

            assert "long" in results
            strategy = results["long"]
            # The gate should NOT have triggered, so rules should NOT be empty
            # (unless other gates reject it, but we mocked those to pass)
            assert len(strategy["rules_set"]) > 0, (
                "Gate should not reject a ruleset with enough symbols"
            )
            # Verify reason is NOT insufficient_distinct_symbols
            assert strategy.get("reason") != "insufficient_distinct_symbols"

    def test_filters_off_gate_skipped(self):
        """Filters off → gate skipped even with 1-symbol rules.

        Given:
        - RB_REQUIRE_SYMBOL_FILTERS=False
        - RB_MIN_DISTINCT_SYMBOLS=5
        - opt_rules have only 1 symbol
        Then:
        - Gate is skipped entirely
        - Normal pipeline flow continues
        """
        train_df = _dummy_df(100)
        val_df = _dummy_df(100)
        pools = {"long": [_single_symbol_rule("BTC")]}

        one_sym_rules = [_single_symbol_rule("BTC")]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(
                _cfg, "RB_REQUIRE_SYMBOL_FILTERS", False,
            ), patch.object(
                _cfg, "RB_MIN_DISTINCT_SYMBOLS", 5,
            ), patch(
                "gpu_fuzzy_trader.rb_governor._filter_good_rules",
                return_value=_make_candidate_records(one_sym_rules),
            ), patch(
                "gpu_fuzzy_trader.rb_governor._compose_ruleset",
                return_value=(
                    _make_candidate_records(one_sym_rules),
                    _mock_train_metrics(),
                    _mock_train_metrics(return_pct=3.0),
                    50.0,
                    [{"step": 1, "action": "seed", "score": 50.0}],
                ),
            ), patch(
                "gpu_fuzzy_trader.rb_governor._optimize_risk",
                return_value=(one_sym_rules, _mock_train_metrics(), _mock_train_metrics(return_pct=3.0), 50.0, []),
            ), patch(
                "gpu_fuzzy_trader.rb_governor._run_profit_amplifier",
                return_value=(one_sym_rules, _mock_train_metrics(), _mock_train_metrics(return_pct=3.0), 50.0, {"accepted": False}),
            ), patch(
                "gpu_fuzzy_trader.rb_governor.CPUBacktestEngine",
            ), patch(
                "gpu_fuzzy_trader.rb_governor._make_walk_forward_fold_engines",
                return_value=([], None),
            ), patch(
                "gpu_fuzzy_trader.rb_governor._write_clean_evaluator",
            ), patch(
                "gpu_fuzzy_trader.rb_governor._passes_symbol_concentration_gate",
                return_value=(True, {"passed": True}),
            ), patch(
                "gpu_fuzzy_trader.rb_governor._passes_tail_holdout_gate",
                return_value=(True, {"passed": True}),
            ):
                results = run_rb_governor_pipeline(
                    train_df, val_df,
                    pools,
                    ("long",),
                    output_dir=tmpdir,
                )

            assert "long" in results
            strategy = results["long"]
            # Gate should be skipped entirely, rules should still be present
            assert len(strategy["rules_set"]) > 0, (
                "Gate should be skipped when RB_REQUIRE_SYMBOL_FILTERS=False"
            )
            # Reason should NOT be insufficient_distinct_symbols
            assert strategy.get("reason") != "insufficient_distinct_symbols"

    def test_multi_symbol_team_passes_gate(self):
        """Multi-symbol team meeting min not rejected by this gate alone.

        Multiple rules each with different symbols, total distinct >= min.
        """
        train_df = _dummy_df(100)
        val_df = _dummy_df(100)
        # Create 5 rules with 5 different symbols
        rules_5_sym = [_single_symbol_rule(f"sym_{i}") for i in range(5)]
        pools = {"long": rules_5_sym}

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(
                _cfg, "RB_REQUIRE_SYMBOL_FILTERS", True,
            ), patch.object(
                _cfg, "RB_MIN_DISTINCT_SYMBOLS", 5,
            ), patch(
                "gpu_fuzzy_trader.rb_governor._filter_good_rules",
                return_value=_make_candidate_records(rules_5_sym),
            ), patch(
                "gpu_fuzzy_trader.rb_governor._compose_ruleset",
                return_value=(
                    _make_candidate_records(rules_5_sym),
                    _mock_train_metrics(),
                    _mock_train_metrics(return_pct=3.0),
                    50.0,
                    [{"step": 1, "action": "seed", "score": 50.0}],
                ),
            ), patch(
                "gpu_fuzzy_trader.rb_governor._optimize_risk",
                return_value=(rules_5_sym, _mock_train_metrics(), _mock_train_metrics(return_pct=3.0), 50.0, []),
            ), patch(
                "gpu_fuzzy_trader.rb_governor._run_profit_amplifier",
                return_value=(rules_5_sym, _mock_train_metrics(), _mock_train_metrics(return_pct=3.0), 50.0, {"accepted": False}),
            ), patch(
                "gpu_fuzzy_trader.rb_governor.CPUBacktestEngine",
            ), patch(
                "gpu_fuzzy_trader.rb_governor._make_walk_forward_fold_engines",
                return_value=([], None),
            ), patch(
                "gpu_fuzzy_trader.rb_governor._write_clean_evaluator",
            ), patch(
                "gpu_fuzzy_trader.rb_governor._passes_symbol_concentration_gate",
                return_value=(True, {"passed": True}),
            ), patch(
                "gpu_fuzzy_trader.rb_governor._passes_tail_holdout_gate",
                return_value=(True, {"passed": True}),
            ):
                results = run_rb_governor_pipeline(
                    train_df, val_df,
                    pools,
                    ("long",),
                    output_dir=tmpdir,
                )

            assert "long" in results
            strategy = results["long"]
            # Gate should pass since we have exactly 5 distinct symbols
            assert len(strategy["rules_set"]) > 0
            assert strategy.get("reason") != "insufficient_distinct_symbols"

    def test_min_distinct_zero_skips_gate(self):
        """RB_MIN_DISTINCT_SYMBOLS=0 skips gate even with 1 symbol."""
        train_df = _dummy_df(100)
        val_df = _dummy_df(100)
        pools = {"long": [_single_symbol_rule("BTC")]}
        one_sym_rules = [_single_symbol_rule("BTC")]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(
                _cfg, "RB_REQUIRE_SYMBOL_FILTERS", True,
            ), patch.object(
                _cfg, "RB_MIN_DISTINCT_SYMBOLS", 0,
            ), patch(
                "gpu_fuzzy_trader.rb_governor._filter_good_rules",
                return_value=_make_candidate_records(one_sym_rules),
            ), patch(
                "gpu_fuzzy_trader.rb_governor._compose_ruleset",
                return_value=(
                    _make_candidate_records(one_sym_rules),
                    _mock_train_metrics(),
                    _mock_train_metrics(return_pct=3.0),
                    50.0,
                    [{"step": 1, "action": "seed", "score": 50.0}],
                ),
            ), patch(
                "gpu_fuzzy_trader.rb_governor._optimize_risk",
                return_value=(one_sym_rules, _mock_train_metrics(), _mock_train_metrics(return_pct=3.0), 50.0, []),
            ), patch(
                "gpu_fuzzy_trader.rb_governor._run_profit_amplifier",
                return_value=(one_sym_rules, _mock_train_metrics(), _mock_train_metrics(return_pct=3.0), 50.0, {"accepted": False}),
            ), patch(
                "gpu_fuzzy_trader.rb_governor.CPUBacktestEngine",
            ), patch(
                "gpu_fuzzy_trader.rb_governor._make_walk_forward_fold_engines",
                return_value=([], None),
            ), patch(
                "gpu_fuzzy_trader.rb_governor._write_clean_evaluator",
            ), patch(
                "gpu_fuzzy_trader.rb_governor._passes_symbol_concentration_gate",
                return_value=(True, {"passed": True}),
            ), patch(
                "gpu_fuzzy_trader.rb_governor._passes_tail_holdout_gate",
                return_value=(True, {"passed": True}),
            ):
                results = run_rb_governor_pipeline(
                    train_df, val_df,
                    pools,
                    ("long",),
                    output_dir=tmpdir,
                )

            assert "long" in results
            strategy = results["long"]
            # Gate should be skipped since min_distinct=0
            assert len(strategy["rules_set"]) > 0


# ---------------------------------------------------------------------------
# Tests for empty strategy loadability
# ---------------------------------------------------------------------------


class TestLoadAndValidate:
    """Verify fail-closed strategies can be loaded via Output_Writer."""

    def test_fail_closed_strategy_loadable(self):
        """Empty strategy from min-symbols gate passes load_and_validate."""
        train_df = _dummy_df(100)
        val_df = _dummy_df(100)
        pools = {"long": [_single_symbol_rule("BTC")]}
        one_sym_rules = [_single_symbol_rule("BTC")]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(
                _cfg, "RB_REQUIRE_SYMBOL_FILTERS", True,
            ), patch.object(
                _cfg, "RB_MIN_DISTINCT_SYMBOLS", 5,
            ), patch(
                "gpu_fuzzy_trader.rb_governor._filter_good_rules",
                return_value=_make_candidate_records(one_sym_rules),
            ), patch(
                "gpu_fuzzy_trader.rb_governor._compose_ruleset",
                return_value=(
                    _make_candidate_records(one_sym_rules),
                    _mock_train_metrics(),
                    _mock_train_metrics(return_pct=3.0),
                    50.0,
                    [{"step": 1, "action": "seed", "score": 50.0}],
                ),
            ), patch(
                "gpu_fuzzy_trader.rb_governor._optimize_risk",
                return_value=(one_sym_rules, _mock_train_metrics(), _mock_train_metrics(return_pct=3.0), 50.0, []),
            ), patch(
                "gpu_fuzzy_trader.rb_governor._run_profit_amplifier",
                return_value=(one_sym_rules, _mock_train_metrics(), _mock_train_metrics(return_pct=3.0), 50.0, {"accepted": False}),
            ), patch(
                "gpu_fuzzy_trader.rb_governor.CPUBacktestEngine",
            ), patch(
                "gpu_fuzzy_trader.rb_governor._make_walk_forward_fold_engines",
                return_value=([], None),
            ), patch(
                "gpu_fuzzy_trader.rb_governor._write_clean_evaluator",
            ):
                run_rb_governor_pipeline(
                    train_df, val_df,
                    pools,
                    ("long",),
                    output_dir=tmpdir,
                )

            strategy_path = Path(tmpdir) / "long.json"
            assert strategy_path.exists()
            # Raw JSON must contain deployment_accepted=False
            with open(strategy_path) as f:
                raw = json.load(f)
            assert raw["direction"] == "long"
            assert raw["rules_set"] == []
            assert raw["deployment_accepted"] is False
            assert raw.get("reason") == "insufficient_distinct_symbols"
            # load_and_validate must accept the empty strategy (fail-closed path)
            writer = Output_Writer()
            writer.load_and_validate(strategy_path)
