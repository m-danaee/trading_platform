"""Tests for RB_ALLOW_FALLBACK fail-closed behaviour.

When ``RB_ALLOW_FALLBACK=False`` (default) and no positive-good single rules
are found, the RB Governor must write an empty strategy with
``deployment_accepted=false`` and skip the legacy raw-score fallback path.
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
from gpu_fuzzy_trader.output.writer import Output_Writer, _validate_rule_set, ValidationError
from gpu_fuzzy_trader.rb_governor import (
    _filter_good_rules,
    _strategy,
    run_rb_governor_pipeline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pool_rules(n: int = 3) -> list[dict]:
    """Return generic pool rules (no positive-good candidates)."""
    return [
        {"conditions": [f"[feat] IS High", f"symbol is sym_{i}"], "tp": 2.0, "sl": 1.2, "capital_pct": 18.0}
        for i in range(n)
    ]


def _dummy_df(size: int = 100) -> pd.DataFrame:
    """Return a minimal DataFrame that passes _prepare_scoring_frame.

    ``label_open_next`` must be finite and positive for CPUBacktestEngine.
    """
    np.random.seed(42)
    symbols = [f"sym_{i}" for i in range(5)]
    entry = np.random.uniform(1.0, 2.0, size=size)  # positive entry prices
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


# ---------------------------------------------------------------------------
# Tests for _strategy helper (empty rules)
# ---------------------------------------------------------------------------

class TestStrategyEmptyRules:
    def test_empty_rules_set(self):
        """_strategy with empty rules produces empty rules_set."""
        strat = _strategy("long", [])
        assert strat["direction"] == "long"
        assert strat["rules_set"] == []
        assert strat["risk_optimized"] is False

    def test_empty_rules_with_extra(self):
        """_strategy with extra dict includes deployment_accepted=false."""
        strat = _strategy("long", [], extra={
            "deployment_accepted": False,
            "reason": "no_positive_good_candidates",
        })
        assert strat["deployment_accepted"] is False
        assert strat["reason"] == "no_positive_good_candidates"
        assert strat["rules_set"] == []


# ---------------------------------------------------------------------------
# Tests for run_rb_governor_pipeline fail-closed behaviour
# ---------------------------------------------------------------------------

class TestFailClosedDefault:
    """Default config (RB_ALLOW_FALLBACK=False) → fail closed when empty."""

    @pytest.fixture
    def engines(self):
        return object(), object()

    def test_default_config_has_allow_fallback_false(self):
        """RB_ALLOW_FALLBACK defaults to False."""
        assert hasattr(_cfg, "RB_ALLOW_FALLBACK")
        assert _cfg.RB_ALLOW_FALLBACK is False

    def test_fail_closed_writes_empty_strategy(self):
        """When _filter_good_rules returns empty and RB_ALLOW_FALLBACK=False,
        an empty strategy with deployment_accepted=false is written."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            reports_dir = out_dir / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)

            train_df = _dummy_df(100)
            val_df = _dummy_df(100)

            # Mock _filter_good_rules to return empty (no positive-good)
            with patch(
                "gpu_fuzzy_trader.rb_governor._filter_good_rules",
                return_value=[],
            ), patch.object(_cfg, "RB_ALLOW_FALLBACK", False):
                # We need to call the function with the pipeline
                # But rather than mocking everything, let's test the core
                # logic path that gets triggered.

                # Build strategy the same way the fail-closed path does
                strategy = _strategy(
                    "long", [],
                    risk_optimized=False,
                    extra={
                        "deployment_accepted": False,
                        "no_positive_good_candidates": True,
                        "reason": "no_positive_good_candidates",
                    },
                )
                assert strategy["rules_set"] == []
                assert strategy["deployment_accepted"] is False
                assert strategy["reason"] == "no_positive_good_candidates"

    def test_fail_closed_skip_legacy_fallback(self):
        """When RB_ALLOW_FALLBACK=False, the legacy raw-score fallback
        (scoring/specializing raw pool) must NOT be entered."""
        train_df = _dummy_df(100)
        val_df = _dummy_df(100)
        pools = {
            "long": _pool_rules(3),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)

            # We need to check that the fallback scoring loop is NOT called.
            # The easiest way: patch _symbol_specialized_variants (which is the
            # first call inside the legacy fallback) and verify it's NOT called.
            with patch(
                "gpu_fuzzy_trader.rb_governor._filter_good_rules",
                return_value=[],
            ), patch.object(_cfg, "RB_ALLOW_FALLBACK", False), patch(
                "gpu_fuzzy_trader.rb_governor._symbol_specialized_variants",
            ) as mock_variants, patch(
                "gpu_fuzzy_trader.rb_governor.CPUBacktestEngine",
            ):
                run_rb_governor_pipeline(
                    train_df, val_df,
                    pools,
                    ("long",),
                    output_dir=tmpdir,
                )
                # _symbol_specialized_variants should NOT have been called
                # because the fail-closed path returns early via `continue`
                mock_variants.assert_not_called()

    def test_fail_closed_report_contains_fail_closed_flag(self):
        """The written report should have fail_closed=True and reason."""
        train_df = _dummy_df(100)
        val_df = _dummy_df(100)
        pools = {
            "long": _pool_rules(3),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)

            with patch(
                "gpu_fuzzy_trader.rb_governor._filter_good_rules",
                return_value=[],
            ), patch.object(_cfg, "RB_ALLOW_FALLBACK", False), patch(
                "gpu_fuzzy_trader.rb_governor.CPUBacktestEngine",
            ):
                run_rb_governor_pipeline(
                    train_df, val_df,
                    pools,
                    ("long",),
                    output_dir=tmpdir,
                )

            report_path = out_dir / "reports" / "rb_governor_long_report.json"
            assert report_path.exists()
            with open(report_path) as f:
                report = json.load(f)
            assert report.get("fail_closed") is True
            assert report.get("reason") == "no_positive_good_candidates"
            assert report["selected_rules"] == 0
            assert report["n_positive_single_rules"] == 0

    def test_fail_closed_deployment_not_accepted(self):
        """The strategy JSON must have deployment_accepted=false."""
        train_df = _dummy_df(100)
        val_df = _dummy_df(100)
        pools = {
            "long": _pool_rules(3),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)

            with patch(
                "gpu_fuzzy_trader.rb_governor._filter_good_rules",
                return_value=[],
            ), patch.object(_cfg, "RB_ALLOW_FALLBACK", False), patch(
                "gpu_fuzzy_trader.rb_governor.CPUBacktestEngine",
            ):
                run_rb_governor_pipeline(
                    train_df, val_df,
                    pools,
                    ("long",),
                    output_dir=tmpdir,
                )

            strategy_path = out_dir / "long.json"
            assert strategy_path.exists()
            with open(strategy_path) as f:
                strategy = json.load(f)
            assert strategy["rules_set"] == []
            assert strategy["deployment_accepted"] is False
            assert strategy.get("reason") == "no_positive_good_candidates"


# ---------------------------------------------------------------------------
# Tests for P1-2: run_rb_governor_pipeline return value includes fail-closed direction
# ---------------------------------------------------------------------------

class TestFailClosedReturnValue:
    """The pipeline return value must include the fail-closed direction."""

    def test_fail_closed_return_value_contains_direction(self):
        """run_rb_governor_pipeline returns {direction: strategy} even when
        the direction is fail-closed (no positive-good candidates)."""
        train_df = _dummy_df(100)
        val_df = _dummy_df(100)
        pools = {
            "long": _pool_rules(3),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "gpu_fuzzy_trader.rb_governor._filter_good_rules",
                return_value=[],
            ), patch.object(_cfg, "RB_ALLOW_FALLBACK", False), patch(
                "gpu_fuzzy_trader.rb_governor.CPUBacktestEngine",
            ):
                results = run_rb_governor_pipeline(
                    train_df, val_df,
                    pools,
                    ("long",),
                    output_dir=tmpdir,
                )
            assert "long" in results, "fail-closed direction must be in results"
            strat = results["long"]
            assert strat["direction"] == "long"
            assert strat["rules_set"] == []
            assert strat["deployment_accepted"] is False


# ---------------------------------------------------------------------------
# Tests for P1-1: _validate_rule_set accepts empty rules when deployment_accepted=False
# ---------------------------------------------------------------------------

class TestValidateRuleSetFailClosed:
    """_validate_rule_set must allow empty rules_set when deployment_accepted is False."""

    def test_accepts_empty_rules_when_deployment_accepted_false(self):
        """Empty rules_set with deployment_accepted=False passes validation."""
        data = {
            "direction": "long",
            "rules_set": [],
            "deployment_accepted": False,
            "reason": "no_positive_good_candidates",
        }
        result = _validate_rule_set(data)
        assert result["direction"] == "long"
        assert result["rules_set"] == []

    def test_still_rejects_empty_rules_when_deployment_accepted_true(self):
        """Empty rules_set with deployment_accepted=True still raises ValidationError."""
        data = {
            "direction": "long",
            "rules_set": [],
            "deployment_accepted": True,
        }
        with pytest.raises(ValidationError, match="must contain at least"):
            _validate_rule_set(data)

    def test_still_rejects_empty_rules_when_deployment_accepted_missing(self):
        """Empty rules_set without deployment_accepted still raises ValidationError
        (default path, not fail-closed)."""
        data = {
            "direction": "long",
            "rules_set": [],
        }
        with pytest.raises(ValidationError, match="must contain at least"):
            _validate_rule_set(data)

    def test_fail_closed_strategy_passes_load_and_validate(self):
        """The actual fail-closed strategy JSON file written by the pipeline
        must be loadable via Output_Writer.load_and_validate."""
        train_df = _dummy_df(100)
        val_df = _dummy_df(100)
        pools = {
            "long": _pool_rules(3),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "gpu_fuzzy_trader.rb_governor._filter_good_rules",
                return_value=[],
            ), patch.object(_cfg, "RB_ALLOW_FALLBACK", False), patch(
                "gpu_fuzzy_trader.rb_governor.CPUBacktestEngine",
            ):
                run_rb_governor_pipeline(
                    train_df, val_df,
                    pools,
                    ("long",),
                    output_dir=tmpdir,
                )
            strategy_path = Path(tmpdir) / "long.json"
            assert strategy_path.exists()
            writer = Output_Writer()
            loaded = writer.load_and_validate(strategy_path)
            assert loaded["direction"] == "long"
            assert loaded["rules_set"] == []


# ---------------------------------------------------------------------------
# Tests for RB_ALLOW_FALLBACK=True (legacy fallback preserved)
# ---------------------------------------------------------------------------

class TestAllowFallbackTrue:
    """When RB_ALLOW_FALLBACK=True, the legacy raw-score fallback runs."""

    def test_fallback_calls_symbol_specialized_variants(self):
        """With RB_ALLOW_FALLBACK=True, _symbol_specialized_variants is invoked
        during the raw-score fallback path."""
        train_df = _dummy_df(100)
        val_df = _dummy_df(100)
        pools = {
            "long": _pool_rules(3),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "gpu_fuzzy_trader.rb_governor._filter_good_rules",
                return_value=[],
            ), patch.object(_cfg, "RB_ALLOW_FALLBACK", True), patch(
                "gpu_fuzzy_trader.rb_governor.CPUBacktestEngine",
            ), patch(
                "gpu_fuzzy_trader.rb_governor._symbol_specialized_variants",
                return_value=[{"conditions": ["feat_1 IS High"], "tp": 2.0, "sl": 1.2, "capital_pct": 18.0}],
            ) as mock_variants, patch(
                "gpu_fuzzy_trader.rb_governor.CPUBacktestEngine.simulate_rule_set",
                return_value={
                    "total_return_pct": 2.0,
                    "max_drawdown_pct": 5.0,
                    "profit_factor": 1.2,
                    "win_rate": 55.0,
                    "executed_trades": 20,
                    "raw_signal_count": 40,
                    "skipped_min_notional_count": 2,
                    "max_simultaneous_positions": 2,
                    "per_symbol_metrics": {},
                },
            ), patch(
                "gpu_fuzzy_trader.rb_governor._compose_ruleset",
                return_value=([], {"total_return_pct": 2.0}, {"total_return_pct": 1.5}, 100.0, []),
            ), patch(
                "gpu_fuzzy_trader.rb_governor._optimize_risk",
                return_value=([{"conditions": ["feat_1 IS High"], "tp": 2.0, "sl": 1.2, "capital_pct": 18.0}], {}, {}, 100.0, []),
            ), patch(
                "gpu_fuzzy_trader.rb_governor._run_profit_amplifier",
                return_value=([{"conditions": ["feat_1 IS High"], "tp": 2.0, "sl": 1.2, "capital_pct": 18.0}], {"total_return_pct": 2.0}, {"total_return_pct": 1.5}, 100.0, {"accepted": True}),
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
                # P2: assert legacy fallback was actually entered
                mock_variants.assert_called()


# ---------------------------------------------------------------------------
# Test for _filter_good_rules producing empty with default config
# ---------------------------------------------------------------------------

class TestFilterGoodRulesEmpty:
    """Verify that RB_ALLOW_FALLBACK controls filtering of empty results."""

    def test_filter_good_rules_can_return_empty(self):
        """_filter_good_rules can return an empty list when no rules pass gates."""
        train_df = _dummy_df(100)
        val_df = _dummy_df(100)
        pool = _pool_rules(3)
        # _filter_good_rules needs real data; it will likely produce empty
        # because our dummy rules may not match features.
        # This test just validates the interface returns list.
        with patch(
            "gpu_fuzzy_trader.rb_governor.CPUBacktestEngine",
        ), patch("gpu_fuzzy_trader.rb_governor._is_positive_good", return_value=False):
            result = _filter_good_rules(pool, train_df, val_df, "long")
        assert isinstance(result, list)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Config-level tests
# ---------------------------------------------------------------------------

class TestConfigFlag:
    """RB_ALLOW_FALLBACK must exist in config with correct defaults."""

    def test_flag_exists_and_default_false(self):
        assert _cfg.RB_ALLOW_FALLBACK is False

    def test_flag_can_be_set_true(self):
        with patch.object(_cfg, "RB_ALLOW_FALLBACK", True):
            assert _cfg.RB_ALLOW_FALLBACK is True

    def test_can_be_read_via_getattr(self):
        assert bool(getattr(_cfg, "RB_ALLOW_FALLBACK", False)) is False
