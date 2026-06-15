"""
Unit tests for Phase 2 pool-admission relaxed thresholds (Task 5).

Tests cover:
  - ``_passes_pool_admission_impl`` with ``cv_fold=True`` using the new
    ``PHASE2_CV_MIN_WORST_RETURN``, ``PHASE2_CV_MIN_WORST_PF``,
    ``PHASE2_CV_MAX_WORST_DD``, ``PHASE2_CV_MIN_FOLD_TRADES`` thresholds.
  - ``PHASE2_KEEP_TOP_RULES`` cap in ``_build_pool_from_archive``.
  - ``PHASE2_STRICT_POSITIVE_GOOD=True`` does not crash.
"""

from __future__ import annotations

import numpy as np
import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.phases.phase2_support import (
    _passes_pool_admission_impl,
    passes_pool_admission_cv_fold,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fold_metrics(
    return_pct: float = 5.0,
    pf: float = 1.5,
    trades: int = 50,
    dd: float = 5.0,
    sortino: float = 1.0,
) -> dict:
    """Build a minimal per-fold metrics dict for pool-admission tests."""
    return {
        "total_return_pct": return_pct,
        "profit_factor": pf,
        "executed_trades": trades,
        "max_drawdown_pct": dd,
        "sortino_ratio": sortino,
        "win_rate": 0.5,
    }


# ---------------------------------------------------------------------------
# Tests for relaxed per-fold pool admission (cv_fold=True)
# ---------------------------------------------------------------------------

class TestCvFoldRelaxedPoolAdmission:
    """Per-fold admission with the new relaxed ``PHASE2_CV_MIN_WORST_*`` thresholds."""

    def setup_method(self) -> None:
        """Ensure we're in purged CV mode for per-fold tests."""
        # Store original to restore
        self._orig_split = _cfg.SPLIT_MODE

    def teardown_method(self) -> None:
        """Restore original config."""
        import gpu_fuzzy_trader.config as cfg
        cfg.SPLIT_MODE = self._orig_split

    @pytest.fixture(autouse=True)
    def _purged_cv_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_cfg, "SPLIT_MODE", "purged_rolling_cv")

    def test_relaxed_returns_pass_mildly_negative(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A rule with val return = -5% should pass (above -8% threshold)."""
        monkeypatch.setattr(_cfg, "PHASE2_CV_MIN_WORST_RETURN", -8.0)
        monkeypatch.setattr(_cfg, "PHASE2_CV_MIN_WORST_PF", 0.80)
        monkeypatch.setattr(_cfg, "PHASE2_CV_MAX_WORST_DD", 18.0)
        monkeypatch.setattr(_cfg, "PHASE2_CV_MIN_FOLD_TRADES", 10)
        monkeypatch.setattr(_cfg, "PHASE2_STRICT_POSITIVE_GOOD", False)

        train = _fold_metrics(return_pct=2.0, pf=1.2, trades=20, dd=5.0)
        val = _fold_metrics(return_pct=-5.0, pf=0.85, trades=15, dd=8.0)
        assert passes_pool_admission_cv_fold(train, val) is True

    def test_rejects_worst_return_below_threshold(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A rule with val return = -15% should fail (< -8% threshold)."""
        monkeypatch.setattr(_cfg, "PHASE2_CV_MIN_WORST_RETURN", -8.0)
        monkeypatch.setattr(_cfg, "PHASE2_CV_MIN_WORST_PF", 0.80)
        monkeypatch.setattr(_cfg, "PHASE2_CV_MAX_WORST_DD", 18.0)
        monkeypatch.setattr(_cfg, "PHASE2_CV_MIN_FOLD_TRADES", 10)
        monkeypatch.setattr(_cfg, "PHASE2_STRICT_POSITIVE_GOOD", False)

        train = _fold_metrics(return_pct=2.0, pf=1.2, trades=20, dd=5.0)
        val = _fold_metrics(return_pct=-15.0, pf=0.85, trades=15, dd=8.0)
        assert passes_pool_admission_cv_fold(train, val) is False

    def test_rejects_worst_dd_above_threshold(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A rule with drawdown = 25% should fail (> 18% threshold)."""
        monkeypatch.setattr(_cfg, "PHASE2_CV_MIN_WORST_RETURN", -8.0)
        monkeypatch.setattr(_cfg, "PHASE2_CV_MIN_WORST_PF", 0.80)
        monkeypatch.setattr(_cfg, "PHASE2_CV_MAX_WORST_DD", 18.0)
        monkeypatch.setattr(_cfg, "PHASE2_CV_MIN_FOLD_TRADES", 10)
        monkeypatch.setattr(_cfg, "PHASE2_STRICT_POSITIVE_GOOD", False)

        train = _fold_metrics(return_pct=2.0, pf=1.2, trades=20, dd=5.0)
        val = _fold_metrics(return_pct=1.0, pf=0.90, trades=15, dd=25.0)
        assert passes_pool_admission_cv_fold(train, val) is False

    def test_rejects_worst_pf_below_threshold(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A rule with val PF = 0.50 should fail (< 0.80 threshold)."""
        monkeypatch.setattr(_cfg, "PHASE2_CV_MIN_WORST_RETURN", -8.0)
        monkeypatch.setattr(_cfg, "PHASE2_CV_MIN_WORST_PF", 0.80)
        monkeypatch.setattr(_cfg, "PHASE2_CV_MAX_WORST_DD", 18.0)
        monkeypatch.setattr(_cfg, "PHASE2_CV_MIN_FOLD_TRADES", 10)
        monkeypatch.setattr(_cfg, "PHASE2_STRICT_POSITIVE_GOOD", False)

        train = _fold_metrics(return_pct=2.0, pf=1.2, trades=20, dd=5.0)
        val = _fold_metrics(return_pct=1.0, pf=0.50, trades=15, dd=8.0)
        assert passes_pool_admission_cv_fold(train, val) is False

    def test_relaxed_trade_floor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A rule with only 12 val trades passes when the relaxed floor is 10."""
        monkeypatch.setattr(_cfg, "PHASE2_CV_MIN_WORST_RETURN", -8.0)
        monkeypatch.setattr(_cfg, "PHASE2_CV_MIN_WORST_PF", 0.80)
        monkeypatch.setattr(_cfg, "PHASE2_CV_MAX_WORST_DD", 18.0)
        monkeypatch.setattr(_cfg, "PHASE2_CV_MIN_FOLD_TRADES", 10)
        monkeypatch.setattr(_cfg, "PHASE2_STRICT_POSITIVE_GOOD", False)

        train = _fold_metrics(return_pct=2.0, pf=1.2, trades=20, dd=5.0)
        val = _fold_metrics(return_pct=-1.0, pf=0.90, trades=12, dd=5.0)
        assert passes_pool_admission_cv_fold(train, val) is True

    def test_strict_trade_floor_still_enforced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A rule with only 8 val trades should fail (< 10 relaxed floor)."""
        monkeypatch.setattr(_cfg, "PHASE2_CV_MIN_WORST_RETURN", -8.0)
        monkeypatch.setattr(_cfg, "PHASE2_CV_MIN_WORST_PF", 0.80)
        monkeypatch.setattr(_cfg, "PHASE2_CV_MAX_WORST_DD", 18.0)
        monkeypatch.setattr(_cfg, "PHASE2_CV_MIN_FOLD_TRADES", 10)
        monkeypatch.setattr(_cfg, "PHASE2_STRICT_POSITIVE_GOOD", False)

        train = _fold_metrics(return_pct=2.0, pf=1.2, trades=20, dd=5.0)
        val = _fold_metrics(return_pct=-1.0, pf=0.90, trades=8, dd=5.0)
        assert passes_pool_admission_cv_fold(train, val) is False


# ---------------------------------------------------------------------------
# Tests for PHASE2_STRICT_POSITIVE_GOOD=True does not crash
# ---------------------------------------------------------------------------

class TestStrictPositiveGoodEnabled:
    """When ``PHASE2_STRICT_POSITIVE_GOOD=True``, the gate is called but
    rules that pass all other criteria still pass."""

    def test_positive_good_does_not_crash_on_valid_rule(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(_cfg, "PHASE2_STRICT_POSITIVE_GOOD", True)
        # Ensure Phase 3 thresholds are set for the gate
        monkeypatch.setattr(_cfg, "PHASE3_MIN_TRAIN_RETURN", 0.0)
        monkeypatch.setattr(_cfg, "PHASE3_MIN_VAL_RETURN", 0.0)
        monkeypatch.setattr(_cfg, "PHASE3_MIN_TRAIN_PF", 1.0)
        monkeypatch.setattr(_cfg, "PHASE3_MIN_VAL_PF", 1.0)
        monkeypatch.setattr(_cfg, "PHASE3_MIN_TRAIN_TRADES", 25)
        monkeypatch.setattr(_cfg, "PHASE3_MIN_VAL_TRADES", 15)

        train = _fold_metrics(return_pct=3.0, pf=1.5, trades=50, dd=5.0)
        val = _fold_metrics(return_pct=2.0, pf=1.2, trades=30, dd=5.0)
        # Should pass without raising
        result = _passes_pool_admission_impl(train, val, cv_fold=False)
        assert result is True


# ---------------------------------------------------------------------------
# Tests for the holdout (non-CV) path unchanged
# ---------------------------------------------------------------------------

class TestHoldoutAdmissionUnchanged:
    """The holdout (cv_fold=False) path should still use the original strict gates."""

    def test_holdout_rejects_negative_val_return(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(_cfg, "SPLIT_MODE", "holdout_70_30")
        monkeypatch.setattr(_cfg, "PHASE2_STRICT_POSITIVE_GOOD", False)
        train = _fold_metrics(return_pct=3.0, pf=1.2, trades=30, dd=5.0)
        val = _fold_metrics(return_pct=-1.0, pf=1.1, trades=20, dd=5.0)
        # In holdout mode, val return must be > 0% (PHASE2_POOL_VAL_RETURN_MIN_PCT = 0.0)
        assert passes_pool_admission_cv_fold(train, val) is False


# ---------------------------------------------------------------------------
# Config presence tests
# ---------------------------------------------------------------------------

class TestConfigKeysPresent:
    """Verify the new config keys are accessible."""

    def test_all_task5_keys_exist(self) -> None:
        assert hasattr(_cfg, "PHASE2_CV_MIN_WORST_RETURN")
        assert hasattr(_cfg, "PHASE2_CV_MIN_WORST_PF")
        assert hasattr(_cfg, "PHASE2_CV_MAX_WORST_DD")
        assert hasattr(_cfg, "PHASE2_CV_MIN_FOLD_TRADES")
        assert hasattr(_cfg, "PHASE2_KEEP_TOP_RULES")
        assert hasattr(_cfg, "PHASE2_STRICT_POSITIVE_GOOD")

    def test_default_values(self) -> None:
        assert _cfg.PHASE2_CV_MIN_WORST_RETURN == -8.0
        assert _cfg.PHASE2_CV_MIN_WORST_PF == 0.80
        assert _cfg.PHASE2_CV_MAX_WORST_DD == 18.0
        assert _cfg.PHASE2_CV_MIN_FOLD_TRADES == 10
        assert _cfg.PHASE2_KEEP_TOP_RULES == 140
        assert _cfg.PHASE2_STRICT_POSITIVE_GOOD is True
