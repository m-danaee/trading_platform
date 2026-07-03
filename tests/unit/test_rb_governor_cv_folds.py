"""Unit tests for CV-fold and PF/DD penalty code paths in RB Governor.

Covers:
- ``_score_metrics`` with ``cv_fold_returns`` CV-fold consistency term
- ``_combined_return_score`` with ``prev_pf`` / ``prev_dd`` PF/DD penalty
- Edge cases: ``cv_fold_returns=None``, ``cv_fold_returns=[]``
- ``_eval_cv_fold_returns`` with ``fold_engines=None``
"""

from __future__ import annotations

import numpy as np
import pytest

from gpu_fuzzy_trader.rb_governor import (
    _score_metrics,
    _combined_return_score,
    _eval_cv_fold_returns,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _train_metrics(
    *,
    return_pct: float = 5.0,
    dd: float = 2.0,
    pf: float = 1.5,
    wr: float = 55.0,
    trades: int = 30,
) -> dict:
    raw = max(trades + 5, 40)
    return {
        "total_return_pct": return_pct,
        "max_drawdown_pct": dd,
        "profit_factor": pf,
        "win_rate": wr,
        "executed_trades": trades,
        "raw_signal_count": raw,
        "skipped_min_notional_count": 2,
        "max_simultaneous_positions": 2,
        "eval_v5_zoo_health_pct": 100.0,
        "eval_v5_success_rate_pct": 100.0,
    }


def _valid_metrics(
    *,
    return_pct: float = 4.0,
    dd: float = 2.0,
    pf: float = 1.5,
    wr: float = 55.0,
    trades: int = 20,
) -> dict:
    raw = max(trades + 5, 40)
    return {
        "total_return_pct": return_pct,
        "max_drawdown_pct": dd,
        "profit_factor": pf,
        "win_rate": wr,
        "executed_trades": trades,
        "raw_signal_count": raw,
        "skipped_min_notional_count": 2,
        "max_simultaneous_positions": 2,
        "eval_v5_zoo_health_pct": 100.0,
        "eval_v5_success_rate_pct": 100.0,
    }


# ---------------------------------------------------------------------------
# F2a: _score_metrics CV-fold consistency term
# ---------------------------------------------------------------------------


class TestScoreMetricsCvFoldTerm:
    """Verify the CV-fold consistency penalty in ``_score_metrics``.

    New formula (skipped for len <= 1):
      - If cv_min < 0:  score -= abs(cv_min) * 5.0
      - If abs(cv_mean) > 0.01 and cv_range > abs(cv_mean) * 2.0:
          score -= (cv_range / max(abs(cv_mean), 0.01) - 2.0) * 5.0
    """

    def test_cv_fold_term_applied_penalizes_negative_min_and_high_variance(self):
        """cv_fold_returns=[5.0, -2.0, 3.0] — worst fold negative + high variance."""
        train_m = _train_metrics(return_pct=5.0)
        valid_m = _valid_metrics(return_pct=4.0)

        score_no_cv = _score_metrics(train_m, valid_m, cv_fold_returns=None)
        cv_returns = [5.0, -2.0, 3.0]
        score_with_cv = _score_metrics(train_m, valid_m, cv_fold_returns=cv_returns)

        # Expected penalties:
        # cv_min = -2.0 < 0 -> penalty = 2.0 * 5.0 = 10.0
        # cv_mean = 2.0, cv_range = 7.0
        # abs(2.0) > 0.01 and 7.0 > 2.0 * 2.0=4.0
        # penalty = (7.0 / 2.0 - 2.0) * 5.0 = (3.5 - 2.0) * 5.0 = 7.5
        # Total = 10.0 + 7.5 = 17.5
        expected_penalty = 10.0 + 7.5
        assert score_with_cv == pytest.approx(score_no_cv - expected_penalty, abs=1e-9)

    def test_cv_fold_all_positive_low_variance_no_penalty(self):
        """All positive folds with low variance — no penalty."""
        train_m = _train_metrics(return_pct=5.0)
        valid_m = _valid_metrics(return_pct=4.0)

        score_no_cv = _score_metrics(train_m, valid_m, cv_fold_returns=None)
        cv_returns = [4.0, 5.0, 6.0]
        score_with_cv = _score_metrics(train_m, valid_m, cv_fold_returns=cv_returns)

        # cv_min = 4.0 >= 0 -> no negative-min penalty
        # cv_mean = 5.0, cv_range = 2.0
        # 2.0 > 5.0 * 2.0 = 10.0? No -> no variance penalty
        assert score_with_cv == pytest.approx(score_no_cv, abs=1e-9)

    def test_cv_fold_none_skips_term(self):
        """cv_fold_returns=None skips the CV term."""
        train_m = _train_metrics(return_pct=5.0)
        valid_m = _valid_metrics(return_pct=4.0)

        score_none = _score_metrics(train_m, valid_m, cv_fold_returns=None)
        score_default = _score_metrics(train_m, valid_m)

        assert score_none == pytest.approx(score_default, abs=1e-9)

    def test_cv_fold_empty_skips_term(self):
        """cv_fold_returns=[] skips the CV term (empty list is falsy)."""
        train_m = _train_metrics(return_pct=5.0)
        valid_m = _valid_metrics(return_pct=4.0)

        score_empty = _score_metrics(train_m, valid_m, cv_fold_returns=[])
        score_none = _score_metrics(train_m, valid_m, cv_fold_returns=None)

        assert score_empty == pytest.approx(score_none, abs=1e-9)

    def test_cv_fold_single_element_skipped(self):
        """Single-element cv_fold_returns (len=1) is skipped."""
        train_m = _train_metrics(return_pct=5.0)
        valid_m = _valid_metrics(return_pct=4.0)

        score_no_cv = _score_metrics(train_m, valid_m, cv_fold_returns=None)
        score_single = _score_metrics(train_m, valid_m, cv_fold_returns=[3.0])

        assert score_single == pytest.approx(score_no_cv, abs=1e-9)


# ---------------------------------------------------------------------------
# F2b: _combined_return_score PF/DD penalty
# ---------------------------------------------------------------------------


class TestCombinedReturnScorePfDdPenalty:
    """Verify the PF/DD penalty in ``_combined_return_score``."""

    def test_pf_dd_penalty_applied(self):
        """Call _combined_return_score with prev_pf=2.0, prev_dd=1.0 and new_pf=1.5, new_dd=2.0.

        Expected penalties:
          PF:  -2.0 * max(0, 2.0 - 1.5) = -1.0
          DD:  -3.0 * max(0, 2.0 - 1.0) = -3.0
          Total penalty = -4.0
        """
        train_m = _train_metrics(return_pct=5.0)
        valid_m = _valid_metrics(return_pct=4.0, pf=1.5, dd=2.0)

        score_no_penalty = _combined_return_score(train_m, valid_m, prev_pf=None, prev_dd=None)
        score_with_penalty = _combined_return_score(train_m, valid_m, prev_pf=2.0, prev_dd=1.0)

        expected_penalty = -2.0 * max(0.0, 2.0 - 1.5) + -3.0 * max(0.0, 2.0 - 1.0)
        assert expected_penalty == pytest.approx(-4.0, abs=1e-9)
        assert score_with_penalty == pytest.approx(score_no_penalty + expected_penalty, abs=1e-9)

    def test_no_degradation_no_penalty(self):
        """When new PF >= prev PF and new DD <= prev DD, no penalty."""
        train_m = _train_metrics(return_pct=5.0)
        valid_m = _valid_metrics(return_pct=4.0, pf=3.0, dd=0.5)

        score_no_penalty = _combined_return_score(train_m, valid_m, prev_pf=None, prev_dd=None)
        score_with_penalty = _combined_return_score(train_m, valid_m, prev_pf=2.0, prev_dd=1.0)

        # PF improved (3.0 >= 2.0), DD improved (0.5 <= 1.0) => no penalty
        assert score_with_penalty == pytest.approx(score_no_penalty, abs=1e-9)

    def test_prev_none_skips_penalty(self):
        """When both prev_pf and prev_dd are None, no penalty is applied."""
        train_m = _train_metrics(return_pct=5.0)
        valid_m = _valid_metrics(return_pct=4.0)

        score = _combined_return_score(train_m, valid_m, prev_pf=None, prev_dd=None)
        score_default = _combined_return_score(train_m, valid_m)

        assert score == pytest.approx(score_default, abs=1e-9)


# ---------------------------------------------------------------------------
# F2c: _eval_cv_fold_returns edge cases
# ---------------------------------------------------------------------------


class TestEvalCvFoldReturns:
    """Verify the helper handles None / empty fold_engines safely."""

    def test_fold_engines_none(self):
        """fold_engines=None returns None without crashing."""
        result = _eval_cv_fold_returns({"conditions": []}, None)
        assert result is None

    def test_fold_engines_empty_list(self):
        """fold_engines=[] returns None without crashing."""
        result = _eval_cv_fold_returns({"conditions": []}, [])
        assert result is None
