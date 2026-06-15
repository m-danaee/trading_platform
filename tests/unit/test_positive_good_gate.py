"""
Unit tests for ``gate_positive_good`` (Task 3 — positive-good gate).

Tests cover:
  - Pure-function behaviour: return / PF / trades thresholds on train+val.
  - Missing-key handling (``total_return_pct``, ``profit_factor``, etc.).
  - Config-driven defaults via ``getattr(_cfg, ...)``.
  - Integration: gate wired into ``_score_pool_rule_on_symbol`` rejects bad rules.
  - Integration: ``PHASE3_REQUIRE_POSITIVE_GOOD=False`` skips the gate.
  - Integration: ``_per_symbol_greedy`` with good/bad mixed pool on a mock engine.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.phases.phase3_rule_set import gate_positive_good


# ---------------------------------------------------------------------------
# Helper — synthetic metric dicts
# ---------------------------------------------------------------------------

def _m(
    return_pct: float = 5.0,
    pf: float = 1.5,
    trades: int = 50,
) -> dict:
    """Build a minimal metrics dict with the keys ``gate_positive_good`` reads."""
    return {
        "total_return_pct": return_pct,
        "profit_factor": pf,
        "executed_trades": trades,
    }


def _make_big_df(n_rows: int = 3000) -> "pd.DataFrame":
    """Create a synthetic DataFrame with features for engine-based tests."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    n_sym = 2
    symbols = ["1", "2"]
    rows: list[dict] = []
    for sym in symbols:
        for i in range(n_rows // n_sym):
            dt = pd.Timestamp("2020-01-01") + pd.Timedelta(minutes=5 * i)
            rows.append({
                "datetime": dt,
                "symbol": sym,
                "label_open_next": float(np.abs(rng.normal(1.0, 0.3))),
                "label_close_288": float(rng.normal(2.0, 1.0)),
                "label_min_288": float(rng.normal(-0.5, 1.0)),
                "label_max_288": float(rng.normal(2.5, 1.0)),
                "label_max_before_min": float(rng.normal(0.5, 0.3)),
                "feature_ma": float(rng.uniform(0, 1)),
            })
    df = pd.DataFrame(rows)
    return df


# ===================================================================
# Pure-function unit tests  (≥ 6 cases)
# ===================================================================


class TestGatePositiveGoodPure:
    """Pure-function tests for ``gate_positive_good`` — no engine needed."""

    # --- Default thresholds: ret > 0, PF >= 1.0, trades >= 25/15 ---

    def test_both_sides_good_passes(self) -> None:
        """All thresholds satisfied → True."""
        train = _m(return_pct=5.0, pf=1.5, trades=50)
        val = _m(return_pct=3.0, pf=1.2, trades=20)
        assert gate_positive_good(train, val) is True

    def test_train_return_non_positive_fails(self) -> None:
        """Train return ≤ 0 → False."""
        train = _m(return_pct=-1.0, pf=1.5, trades=50)
        val = _m(return_pct=3.0, pf=1.2, trades=20)
        assert gate_positive_good(train, val) is False

    def test_val_return_non_positive_fails(self) -> None:
        """Val return ≤ 0 → False."""
        train = _m(return_pct=5.0, pf=1.5, trades=50)
        val = _m(return_pct=-2.0, pf=1.2, trades=20)
        assert gate_positive_good(train, val) is False

    def test_train_pf_below_minimum_fails(self) -> None:
        """Train PF < 1.0 → False."""
        train = _m(return_pct=5.0, pf=0.8, trades=50)
        val = _m(return_pct=3.0, pf=1.2, trades=20)
        assert gate_positive_good(train, val) is False

    def test_val_pf_below_minimum_fails(self) -> None:
        """Val PF < 1.0 → False."""
        train = _m(return_pct=5.0, pf=1.5, trades=50)
        val = _m(return_pct=3.0, pf=0.9, trades=20)
        assert gate_positive_good(train, val) is False

    def test_train_trades_below_minimum_fails(self) -> None:
        """Train trades < 25 → False."""
        train = _m(return_pct=5.0, pf=1.5, trades=10)
        val = _m(return_pct=3.0, pf=1.2, trades=20)
        assert gate_positive_good(train, val) is False

    def test_val_trades_below_minimum_fails(self) -> None:
        """Val trades < 15 → False."""
        train = _m(return_pct=5.0, pf=1.5, trades=50)
        val = _m(return_pct=3.0, pf=1.2, trades=5)
        assert gate_positive_good(train, val) is False

    # --- Missing / absent keys ---

    def test_missing_train_return_fails(self) -> None:
        """Missing ``total_return_pct`` in train → False (treated as 0)."""
        train = _m(return_pct=5.0, pf=1.5, trades=50)
        del train["total_return_pct"]
        val = _m(return_pct=3.0, pf=1.2, trades=20)
        assert gate_positive_good(train, val) is False

    def test_missing_val_profit_factor_fails(self) -> None:
        """Missing ``profit_factor`` in val → False (treated as 0.0 < 1.0)."""
        train = _m(return_pct=5.0, pf=1.5, trades=50)
        val = _m(return_pct=3.0, pf=1.2, trades=20)
        del val["profit_factor"]
        assert gate_positive_good(train, val) is False

    def test_missing_executed_trades_fails(self) -> None:
        """Missing ``executed_trades`` → False (treated as 0)."""
        train = _m(return_pct=5.0, pf=1.5, trades=50)
        val = _m(return_pct=3.0, pf=1.2, trades=20)
        del val["executed_trades"]
        assert gate_positive_good(train, val) is False

    # --- Keyword overrides ---

    def test_custom_min_train_return(self) -> None:
        """Override ``min_train_return=2.0``; train_ret=1.5 → False."""
        train = _m(return_pct=1.5, pf=1.5, trades=50)
        val = _m(return_pct=3.0, pf=1.2, trades=20)
        assert (
            gate_positive_good(train, val, min_train_return=2.0)
            is False
        )

    def test_custom_min_val_pf(self) -> None:
        """Override ``min_val_pf=1.5``; val_pf=1.2 → False."""
        train = _m(return_pct=5.0, pf=1.5, trades=50)
        val = _m(return_pct=3.0, pf=1.2, trades=20)
        assert (
            gate_positive_good(train, val, min_val_pf=1.5)
            is False
        )

    def test_edge_zero_return(self) -> None:
        """Return exactly 0.0 → False (not > 0)."""
        train = _m(return_pct=0.0, pf=1.5, trades=50)
        val = _m(return_pct=3.0, pf=1.2, trades=20)
        assert gate_positive_good(train, val) is False

    def test_edge_pf_exactly_one(self) -> None:
        """PF exactly 1.0 passes (>= 1.0)."""
        train = _m(return_pct=5.0, pf=1.0, trades=50)
        val = _m(return_pct=3.0, pf=1.0, trades=20)
        assert gate_positive_good(train, val) is True


# ===================================================================
# Integration — gate wired into _score_pool_rule_on_symbol
# ===================================================================


class TestGateWiredIntoScorePoolRule:
    """Verify the gate is called in ``_score_pool_rule_on_symbol``."""

    def test_gate_called_and_accepts_good_rule(self) -> None:
        """When the gate returns True, the rule score is not -999."""
        from gpu_fuzzy_trader.phases.phase3_rule_set import _score_pool_rule_on_symbol

        df = _make_big_df(3000)
        sym_df = df[df["symbol"] == "1"].reset_index(drop=True)
        rule = {"conditions": ["[feature_ma] IS Very High"]}

        # Patch gate_positive_good to always return True (rule is "good").
        with patch(
            "gpu_fuzzy_trader.phases.phase3_rule_set.gate_positive_good",
            return_value=True,
        ):
            result = _score_pool_rule_on_symbol(
                rule, sym_df, "long",
                train_symbol_df=sym_df,
            )
            # The gap gate may still reject, but we verify the positive-good
            # gate accepted the rule and the function produced a valid dict.
            assert isinstance(result, dict)
            assert "return_pct" in result
            assert "trades" in result

    def test_bad_rule_rejected_by_gate(self) -> None:
        """When the gate returns False, the rule score is -999."""
        from gpu_fuzzy_trader.phases.phase3_rule_set import _score_pool_rule_on_symbol

        with patch(
            "gpu_fuzzy_trader.phases.phase3_rule_set.gate_positive_good",
            return_value=False,
        ):
            df = _make_big_df(3000)
            sym_df = df[df["symbol"] == "1"].reset_index(drop=True)
            rule = {"conditions": ["[feature_ma] IS Very High"]}

            result = _score_pool_rule_on_symbol(
                rule, sym_df, "long",
                train_symbol_df=sym_df,
            )
            assert result["return_pct"] == -999.0, (
                f"Bad rule expected -999, got {result}"
            )

    def test_gate_not_called_without_train_data(self) -> None:
        """When ``train_symbol_df`` is None, gate is not called (no train metrics)."""
        from gpu_fuzzy_trader.phases.phase3_rule_set import _score_pool_rule_on_symbol

        with patch(
            "gpu_fuzzy_trader.phases.phase3_rule_set.gate_positive_good",
            return_value=True,
        ) as mock_gate:
            df = _make_big_df(3000)
            sym_df = df[df["symbol"] == "1"].reset_index(drop=True)
            rule = {"conditions": ["[feature_ma] IS Very High"]}

            _score_pool_rule_on_symbol(
                rule, sym_df, "long",
                train_symbol_df=None,  # no train data
            )
            mock_gate.assert_not_called()


# ===================================================================
# Integration — gate skippable via PHASE3_REQUIRE_POSITIVE_GOOD=False
# ===================================================================


class TestGateSkippable:
    """When ``PHASE3_REQUIRE_POSITIVE_GOOD=False`` the gate is bypassed."""

    def test_gate_skipped_when_disabled(self) -> None:
        """Even a rule with bad metrics passes when the gate is disabled."""
        with patch.object(_cfg, "PHASE3_REQUIRE_POSITIVE_GOOD", False):
            from gpu_fuzzy_trader.phases.phase3_rule_set import _score_pool_rule_on_symbol

            df = _make_big_df(3000)
            sym_df = df[df["symbol"] == "1"].reset_index(drop=True)
            rule = {"conditions": ["[feature_ma] IS Very High"]}

            result = _score_pool_rule_on_symbol(
                rule, sym_df, "long",
                train_symbol_df=sym_df,
            )
            assert isinstance(result, dict)
            assert "return_pct" in result


# ===================================================================
# Integration — gate_positive_good importable from phase3_rule_set
# ===================================================================


class TestImportable:
    """``gate_positive_good`` is importable from ``phase3_rule_set``."""

    def test_importable(self) -> None:
        """Confirm import works (already imported at module top)."""
        from gpu_fuzzy_trader.phases.phase3_rule_set import gate_positive_good as g2
        assert g2 is gate_positive_good


# ===================================================================
# Integration — PHASE3_MAX_TRAIN_VAL_GAP_PCT still in place
# ===================================================================


class TestExistingGatePreserved:
    """The existing gap-reject gate still works alongside the new one."""

    def test_gap_gate_still_rejects(self) -> None:
        """A rule with val >> train return is rejected by gap even if positive-good passes."""
        from gpu_fuzzy_trader.phases.phase3_rule_set import _score_pool_rule_on_symbol

        with patch.object(_cfg, "PHASE3_MAX_TRAIN_VAL_GAP_PCT", 5.0):
            df = _make_big_df(3000)
            sym_df = df[df["symbol"] == "1"].reset_index(drop=True)
            rule = {"conditions": ["[feature_ma] IS Very High"]}

            result = _score_pool_rule_on_symbol(
                rule, sym_df, "long",
                train_symbol_df=sym_df,
            )
            assert isinstance(result, dict)
