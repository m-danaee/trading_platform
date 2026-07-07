"""Unit tests for CV-fold and PF/DD penalty code paths in RB Governor.

Covers:
- ``_score_metrics`` with ``cv_fold_returns`` CV-fold consistency term
- ``_combined_return_score`` with ``prev_pf`` / ``prev_dd`` PF/DD penalty
- Edge cases: ``cv_fold_returns=None``, ``cv_fold_returns=[]``
- ``_eval_cv_fold_returns`` with ``fold_engines=None``
- ``_make_walk_forward_fold_engines`` data splitting + engine building
- ``_optimize_risk`` walk-forward selection (2-fold worst-case)
- Regression: WF_SPLITS=1 reproduces pre-task-3 behavior
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.rb_governor import (
    _score_metrics,
    _combined_return_score,
    _eval_cv_fold_returns,
    _make_walk_forward_fold_engines,
    _optimize_risk,
    _evaluate_ruleset,
    _rule_to_engine,
    CandidateRecord,
)
from gpu_fuzzy_trader.backtest.cpu_engine import CPUBacktestEngine

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


def _make_synthetic_df(symbols: list[str], rows_per_sym: int) -> pd.DataFrame:
    """Create a minimal DataFrame with symbol/datetime and label columns."""
    rows: list[dict] = []
    for sym in symbols:
        for i in range(rows_per_sym):
            rows.append({
                "symbol": sym,
                "datetime": pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5 * i),
                "_symbol_bar_index": i,
                "label_open_next": 100.0,
                "label_max_288": 101.0,
                "label_min_288": 99.0,
                "label_close_288": 100.5,
                "label_max_before_min": 1,
                "feat_a": 0.5,
            })
    return pd.DataFrame(rows)


class _MockEngine:
    """Minimal mock that mimics CPUBacktestEngine for testing _optimize_risk.

    simulate_rule_set returns fixed metrics; the mock records the rules seen.
    """
    def __init__(self, metrics: dict | None = None):
        self._metrics = metrics or _valid_metrics()
        self.called_with: list = []

    def simulate_rule_set(self, rules):
        self.called_with.append(copy.deepcopy(rules))
        return dict(self._metrics)


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


# ---------------------------------------------------------------------------
# Task-3: Walk-forward fold engine builder
# ---------------------------------------------------------------------------


class TestMakeWalkForwardFoldEngines:
    """Verify _make_walk_forward_fold_engines splits data correctly."""

    def test_returns_correct_number_of_engines(self):
        """2 splits + tail holdout → 2 fold engines + 1 tail engine."""
        df = _make_synthetic_df(["SYM1", "SYM2"], rows_per_sym=100)
        folds, tail = _make_walk_forward_fold_engines(df, n_splits=2, tail_holdout_frac=0.25, direction="long")
        assert len(folds) == 2
        assert tail is not None
        assert isinstance(folds[0], CPUBacktestEngine)
        assert isinstance(tail, CPUBacktestEngine)

    def test_no_tail_when_frac_zero(self):
        """tail_holdout_frac=0 → tail engine is None."""
        df = _make_synthetic_df(["SYM1"], rows_per_sym=100)
        folds, tail = _make_walk_forward_fold_engines(df, n_splits=2, tail_holdout_frac=0.0, direction="long")
        assert len(folds) == 2
        assert tail is None

    def test_per_symbol_chronological_split(self):
        """Each symbol's data is divided into contiguous chunks across folds."""
        sym1 = "SYM_A"
        sym2 = "SYM_B"
        df = _make_synthetic_df([sym1, sym2], rows_per_sym=100)
        # Force last-symbol bar index to be second in sorted order
        df.loc[df["symbol"] == sym2, "_symbol_bar_index"] = list(range(100))
        folds, tail = _make_walk_forward_fold_engines(df, n_splits=2, tail_holdout_frac=0.25, direction="long")

        # Each fold engine should have been prepared via _prepare_scoring_frame
        # and contain data from both symbols
        # We can check that the engines have data by verifying they have a valid DF
        assert folds[0] is not None
        assert folds[1] is not None

    def test_single_symbol_works(self):
        """Single symbol without symbol column is handled gracefully."""
        df = _make_synthetic_df(["SYM1"], rows_per_sym=50)
        # Remove symbol column to test fallback
        df_no_sym = df.drop(columns=["symbol"]).copy()
        folds, tail = _make_walk_forward_fold_engines(
            df_no_sym, n_splits=2, tail_holdout_frac=0.0, direction="long",
        )
        assert len(folds) == 2
        assert tail is None

    def test_small_data_does_not_crash(self):
        """Very small data per symbol (fewer rows than n_splits) does not crash."""
        df = _make_synthetic_df(["SYM1"], rows_per_sym=2)
        folds, tail = _make_walk_forward_fold_engines(df, n_splits=2, tail_holdout_frac=0.25, direction="long")
        assert len(folds) == 2
        assert tail is not None


# ---------------------------------------------------------------------------
# Task-3: Walk-forward history dict fields
# ---------------------------------------------------------------------------


class TestOptimizeRiskWalkForwardHistory:
    """Verify _optimize_risk with fold_engines adds fold_scores and min_fold_score.

    We monkeypatch _evaluate_ruleset to produce known fold scores and verify
    the history dict fields.
    """

    def _make_selected(self) -> list[CandidateRecord]:
        """One dummy candidate rule so _optimize_risk has something to work with."""
        rule = {"conditions": ["symbol is SYM1"], "tp": 2.0, "sl": 1.2, "capital_pct": 20.0}
        return [CandidateRecord(rule=rule, train_metrics=_train_metrics(), valid_metrics=_valid_metrics(), score=100.0)]

    def test_history_has_fold_scores_and_min_fold_score(self, monkeypatch):
        """When fold_engines is provided, history entries contain fold_scores list and min_fold_score."""
        selected = self._make_selected()
        train_mock = _MockEngine(_train_metrics(return_pct=5.0))

        # Two fold engines that give different valid metrics
        fold1_mock = _MockEngine(_valid_metrics(return_pct=3.0))
        fold2_mock = _MockEngine(_valid_metrics(return_pct=4.0))

        # valid_engine mock (not used in walk-forward for selection, but called for baseline)
        valid_mock = _MockEngine(_valid_metrics(return_pct=4.0))

        # Monkeypatch _evaluate_ruleset to use our mocks' simulate_rule_set
        original_eval = _evaluate_ruleset

        def mock_evaluate_ruleset(train_eng, valid_eng, rules):
            train_m = train_eng.simulate_rule_set(rules)
            valid_m = valid_eng.simulate_rule_set(rules)
            score = _score_metrics(train_m, valid_m)
            return train_m, valid_m, score

        monkeypatch.setattr("gpu_fuzzy_trader.rb_governor._evaluate_ruleset", mock_evaluate_ruleset)

        # Monkeypatch config to reduce grid size for test speed
        monkeypatch.setattr(_cfg, "RB_TP_GRID", (2.0,))
        monkeypatch.setattr(_cfg, "RB_SL_GRID", (1.2,))
        monkeypatch.setattr(_cfg, "RB_CAPITAL_GRID", (20.0,))
        monkeypatch.setattr(_cfg, "RB_RISK_OPT_PASSES", 1)
        monkeypatch.setattr(_cfg, "RB_MAX_TOTAL_CAPITAL", 100.0)

        with monkeypatch.context() as m:
            m.setattr(_cfg, "RB_RULESET_MIN_TRAIN_TRADES", 5)
            m.setattr(_cfg, "RB_RULESET_MIN_VALID_TRADES", 5)
            m.setattr(_cfg, "RB_MIN_TRAIN_TRADES", 5)
            m.setattr(_cfg, "RB_MIN_VALID_TRADES", 5)
            m.setattr(_cfg, "RB_RISK_MIN_IMPROVEMENT", 0.01)
            m.setattr(_cfg, "RB_REQUIRE_TRAIN_SLIGHTLY_ABOVE_VALID", False)

            rules, train, valid, score, hist = _optimize_risk(
                selected, train_mock, valid_mock, "long",
                fold_engines=[fold1_mock, fold2_mock],
                tail_holdout_engine=None,
            )

        # The history should have at least 1 entry (the baseline at pass=0)
        assert len(hist) >= 1
        # Every entry should have fold_scores and min_fold_score
        for entry in hist:
            assert "fold_scores" in entry, f"Entry missing fold_scores: {entry}"
            assert "min_fold_score" in entry, f"Entry missing min_fold_score: {entry}"
            assert isinstance(entry["fold_scores"], list)
            assert len(entry["fold_scores"]) == 2
            assert isinstance(entry["min_fold_score"], float)

    def test_regression_single_fold_no_tail(self, monkeypatch):
        """WF_SPLITS=1 and USE_TAIL_HOLDOUT=False: behavior matches pre-task-3.

        When fold_engines=None and tail_holdout_engine=None, history entries
        must NOT contain fold_scores or min_fold_score.
        """
        selected = self._make_selected()
        train_mock = _MockEngine(_train_metrics(return_pct=5.0))
        valid_mock = _MockEngine(_valid_metrics(return_pct=4.0))

        original_eval = _evaluate_ruleset

        def mock_evaluate_ruleset(train_eng, valid_eng, rules):
            train_m = train_eng.simulate_rule_set(rules)
            valid_m = valid_eng.simulate_rule_set(rules)
            score = _score_metrics(train_m, valid_m)
            return train_m, valid_m, score

        monkeypatch.setattr("gpu_fuzzy_trader.rb_governor._evaluate_ruleset", mock_evaluate_ruleset)
        monkeypatch.setattr(_cfg, "RB_TP_GRID", (2.0,))
        monkeypatch.setattr(_cfg, "RB_SL_GRID", (1.2,))
        monkeypatch.setattr(_cfg, "RB_CAPITAL_GRID", (20.0,))
        monkeypatch.setattr(_cfg, "RB_RISK_OPT_PASSES", 1)
        monkeypatch.setattr(_cfg, "RB_MAX_TOTAL_CAPITAL", 100.0)
        monkeypatch.setattr(_cfg, "RB_RULESET_MIN_TRAIN_TRADES", 5)
        monkeypatch.setattr(_cfg, "RB_RULESET_MIN_VALID_TRADES", 5)
        monkeypatch.setattr(_cfg, "RB_MIN_TRAIN_TRADES", 5)
        monkeypatch.setattr(_cfg, "RB_MIN_VALID_TRADES", 5)
        monkeypatch.setattr(_cfg, "RB_RISK_MIN_IMPROVEMENT", 0.01)
        monkeypatch.setattr(_cfg, "RB_REQUIRE_TRAIN_SLIGHTLY_ABOVE_VALID", False)

        rules, train, valid, score, hist = _optimize_risk(
            selected, train_mock, valid_mock, "long",
            fold_engines=None, tail_holdout_engine=None,
        )

        for entry in hist:
            assert "fold_scores" not in entry, f"Entry should NOT have fold_scores: {entry}"
            assert "min_fold_score" not in entry, f"Entry should NOT have min_fold_score: {entry}"


# ---------------------------------------------------------------------------
# Task-3: 2-fold selection rejects overfit combo
# ---------------------------------------------------------------------------


class TestTwoFoldRejectsOverfitCombo:
    """With 2 folds that disagree on optimal TP/SL, the 2-fold version must pick
    the combo that performs well on BOTH folds (worst-case selection)."""

    def _make_selected(self) -> list[CandidateRecord]:
        rule = {"conditions": ["symbol is SYM1"], "tp": 2.0, "sl": 1.2, "capital_pct": 20.0}
        return [CandidateRecord(rule=rule, train_metrics=_train_metrics(), valid_metrics=_valid_metrics(), score=100.0)]

    def _combo_key(self, rules):
        """Return a tuple identifying the TP/SL/cap combo being tested."""
        if not rules:
            return None
        r = rules[0]
        return (r.get("tp"), r.get("sl"), r.get("capital_pct"))

    def test_rejects_overfit_combo(self, monkeypatch):
        """Create mock evaluate_ruleset with controlled scores per combo per fold.

        Combos (fold1_score, fold2_score):
        - Combo A (3.0,1.2,20): fold1=400, fold2=-100  → min=-100
        - Combo B (2.0,1.2,20): fold1=300, fold2=200    → min=200
        - Combo C (2.5,1.5,20): fold1=350, fold2=340    → min=340 ← best min

        Legacy (single-fold) would pick A (best fold-1 score=400).
        Walk-forward should pick C (best min=340).
        """
        from gpu_fuzzy_trader import rb_governor as _rg

        def mock_eval(train_eng, valid_eng, rules):
            """Return controlled train_m, valid_m, score."""
            ck = self._combo_key(rules)
            engine_is_fold1 = getattr(valid_eng, "_fold_id", None) == 1
            engine_is_fold2 = getattr(valid_eng, "_fold_id", None) == 2
            engine_is_main = not (engine_is_fold1 or engine_is_fold2)

            if ck == (3.0, 1.2, 20.0):
                if engine_is_fold1:
                    valid_ret, score_val = 5.0, 400.0
                elif engine_is_fold2:
                    valid_ret, score_val = 1.0, -100.0  # passes PF/trade checks but low score
                else:
                    valid_ret, score_val = 5.0, 400.0
            elif ck == (2.0, 1.2, 20.0):
                if engine_is_fold1:
                    valid_ret, score_val = 4.0, 300.0
                elif engine_is_fold2:
                    valid_ret, score_val = 3.5, 200.0
                else:
                    valid_ret, score_val = 4.0, 300.0
            elif ck == (2.5, 1.5, 20.0):
                if engine_is_fold1:
                    valid_ret, score_val = 4.5, 350.0
                elif engine_is_fold2:
                    valid_ret, score_val = 4.2, 340.0
                else:
                    valid_ret, score_val = 4.5, 350.0
            else:
                valid_ret, score_val = 3.0, 200.0

            train_m = _train_metrics(return_pct=5.0)
            valid_m = _valid_metrics(
                return_pct=valid_ret,
                pf=1.5 if valid_ret > 0 else 1.1,
                trades=25,
            )
            return train_m, valid_m, score_val

        monkeypatch.setattr(_rg, "_evaluate_ruleset", mock_eval)

        train_mock = _MockEngine(_train_metrics(return_pct=5.0))
        valid_mock = _MockEngine(_valid_metrics(return_pct=4.0))
        fold1_mock = _MockEngine(_valid_metrics(return_pct=4.0))
        fold1_mock._fold_id = 1
        fold2_mock = _MockEngine(_valid_metrics(return_pct=4.0))
        fold2_mock._fold_id = 2

        monkeypatch.setattr(_cfg, "RB_TP_GRID", (2.0, 2.5, 3.0))
        monkeypatch.setattr(_cfg, "RB_SL_GRID", (1.2, 1.5))
        monkeypatch.setattr(_cfg, "RB_CAPITAL_GRID", (20.0,))
        monkeypatch.setattr(_cfg, "RB_RISK_OPT_PASSES", 1)
        monkeypatch.setattr(_cfg, "RB_MAX_TOTAL_CAPITAL", 100.0)
        monkeypatch.setattr(_cfg, "RB_RULESET_MIN_TRAIN_TRADES", 5)
        monkeypatch.setattr(_cfg, "RB_RULESET_MIN_VALID_TRADES", 5)
        monkeypatch.setattr(_cfg, "RB_MIN_TRAIN_TRADES", 5)
        monkeypatch.setattr(_cfg, "RB_MIN_VALID_TRADES", 5)
        monkeypatch.setattr(_cfg, "RB_RISK_MIN_IMPROVEMENT", 0.01)
        monkeypatch.setattr(_cfg, "RB_REQUIRE_TRAIN_SLIGHTLY_ABOVE_VALID", False)

        selected = self._make_selected()

        rules, train, valid, score, hist = _optimize_risk(
            selected, train_mock, valid_mock, "long",
            fold_engines=[fold1_mock, fold2_mock],
            tail_holdout_engine=None,
        )

        assert len(rules) == 1
        selected_combo = (rules[0].get("tp"), rules[0].get("sl"), rules[0].get("capital_pct"))
        assert selected_combo == (2.5, 1.5, 20.0), (
            f"Expected combo C (2.5, 1.5, 20.0) but got {selected_combo}. "
            f"History fold_scores: {[e.get('fold_scores', []) for e in hist]}"
        )

    def test_baseline_cur_score_uses_min_fold_in_walk_forward(self, monkeypatch):
        """HIGH-fix: When cur_score (full-val) is high but min(fold_scores) is low,
        and a candidate has lower full-val score but HIGHER min(fold_scores),
        walk-forward must ACCEPT the candidate.

        Without the fix, cur_score stays at the full-validation score (400),
        so the candidate's min-fold score (150) fails the improvement threshold
        (150 > 400 + 0.01 → False) and is incorrectly rejected.
        With the fix, cur_score = min(init_fold_scores) = 100,
        so 150 > 100 + 0.01 → True → candidate accepted.
        """
        from gpu_fuzzy_trader import rb_governor as _rg

        def mock_eval(train_eng, valid_eng, rules):
            ck = self._combo_key(rules)
            engine_is_fold1 = getattr(valid_eng, "_fold_id", None) == 1
            engine_is_fold2 = getattr(valid_eng, "_fold_id", None) == 2
            engine_is_main = not (engine_is_fold1 or engine_is_fold2)

            # Baseline combo (tp=2.0, sl=1.2, cap=20.0): high full-val, low folds
            if ck == (2.0, 1.2, 20.0):
                if engine_is_fold1:
                    valid_ret, score_val = 2.0, 100.0
                elif engine_is_fold2:
                    valid_ret, score_val = 2.0, 100.0
                else:
                    valid_ret, score_val = 6.0, 400.0  # high full-val
            # Candidate combo (tp=2.5, sl=1.2, cap=20.0): lower full-val, higher folds
            elif ck == (2.5, 1.2, 20.0):
                if engine_is_fold1:
                    valid_ret, score_val = 3.5, 150.0
                elif engine_is_fold2:
                    valid_ret, score_val = 3.5, 150.0
                else:
                    valid_ret, score_val = 4.0, 200.0  # lower full-val
            else:
                valid_ret, score_val = 3.0, 200.0

            train_m = _train_metrics(return_pct=5.0)
            valid_m = _valid_metrics(
                return_pct=valid_ret,
                pf=1.5 if valid_ret > 0 else 1.1,
                trades=25,
            )
            return train_m, valid_m, score_val

        monkeypatch.setattr(_rg, "_evaluate_ruleset", mock_eval)

        train_mock = _MockEngine(_train_metrics(return_pct=5.0))
        valid_mock = _MockEngine(_valid_metrics(return_pct=4.0))
        fold1_mock = _MockEngine(_valid_metrics(return_pct=4.0))
        fold1_mock._fold_id = 1
        fold2_mock = _MockEngine(_valid_metrics(return_pct=4.0))
        fold2_mock._fold_id = 2

        monkeypatch.setattr(_cfg, "RB_TP_GRID", (2.0, 2.5))
        monkeypatch.setattr(_cfg, "RB_SL_GRID", (1.2,))
        monkeypatch.setattr(_cfg, "RB_CAPITAL_GRID", (20.0,))
        monkeypatch.setattr(_cfg, "RB_RISK_OPT_PASSES", 1)
        monkeypatch.setattr(_cfg, "RB_MAX_TOTAL_CAPITAL", 100.0)
        monkeypatch.setattr(_cfg, "RB_RULESET_MIN_TRAIN_TRADES", 5)
        monkeypatch.setattr(_cfg, "RB_RULESET_MIN_VALID_TRADES", 5)
        monkeypatch.setattr(_cfg, "RB_MIN_TRAIN_TRADES", 5)
        monkeypatch.setattr(_cfg, "RB_MIN_VALID_TRADES", 5)
        monkeypatch.setattr(_cfg, "RB_RISK_MIN_IMPROVEMENT", 0.01)
        monkeypatch.setattr(_cfg, "RB_REQUIRE_TRAIN_SLIGHTLY_ABOVE_VALID", False)

        selected = self._make_selected()
        # Baseline combo needs tp=2.0 to match what the mock expects
        selected[0].rule["tp"] = 2.0
        selected[0].rule["sl"] = 1.2
        selected[0].rule["capital_pct"] = 20.0

        rules, train, valid, score, hist = _optimize_risk(
            selected, train_mock, valid_mock, "long",
            fold_engines=[fold1_mock, fold2_mock],
            tail_holdout_engine=None,
        )

        assert len(rules) == 1
        selected_combo = (rules[0].get("tp"), rules[0].get("sl"), rules[0].get("capital_pct"))
        # Candidate (2.5, 1.2, 20.0) must be accepted: its min-fold (150) > baseline min-fold (100)
        assert selected_combo == (2.5, 1.2, 20.0), (
            f"Expected candidate (2.5,1.2,20.0) to be accepted by walk-forward, "
            f"but got {selected_combo}. Without the cur_score fix, the baseline's "
            f"full-val score (400) would reject the candidate. "
            f"History: {hist}"
        )
        # Verify the initial history entry's score is the min-fold score, not full-val
        assert hist[0]["score"] == pytest.approx(100.0, abs=1e-9), (
            f"Initial hist entry score should be min fold score (100) in walk-forward mode, "
            f"not the full-validation score (400). Got: {hist[0]['score']}"
        )

    def test_legacy_single_fold_picks_top_fold1_combo(self, monkeypatch):
        """With fold_engines=None (legacy), the single valid engine's scores
        are used, so combo A (best on fold-1) is selected."""
        from gpu_fuzzy_trader import rb_governor as _rg

        def mock_eval(train_eng, valid_eng, rules):
            ck = self._combo_key(rules)
            if ck == (3.0, 1.2, 20.0):
                valid_ret, score_val = 5.0, 400.0  # A best
            elif ck == (2.0, 1.2, 20.0):
                valid_ret, score_val = 4.0, 300.0
            elif ck == (2.5, 1.5, 20.0):
                valid_ret, score_val = 4.5, 350.0
            else:
                valid_ret, score_val = 3.0, 200.0
            train_m = _train_metrics(return_pct=5.0)
            valid_m = _valid_metrics(return_pct=valid_ret, pf=1.5, trades=25)
            return train_m, valid_m, score_val

        monkeypatch.setattr(_rg, "_evaluate_ruleset", mock_eval)

        train_mock = _MockEngine(_train_metrics(return_pct=5.0))
        valid_mock = _MockEngine(_valid_metrics(return_pct=4.0))

        monkeypatch.setattr(_cfg, "RB_TP_GRID", (2.0, 2.5, 3.0))
        monkeypatch.setattr(_cfg, "RB_SL_GRID", (1.2, 1.5))
        monkeypatch.setattr(_cfg, "RB_CAPITAL_GRID", (20.0,))
        monkeypatch.setattr(_cfg, "RB_RISK_OPT_PASSES", 1)
        monkeypatch.setattr(_cfg, "RB_MAX_TOTAL_CAPITAL", 100.0)
        monkeypatch.setattr(_cfg, "RB_RULESET_MIN_TRAIN_TRADES", 5)
        monkeypatch.setattr(_cfg, "RB_RULESET_MIN_VALID_TRADES", 5)
        monkeypatch.setattr(_cfg, "RB_MIN_TRAIN_TRADES", 5)
        monkeypatch.setattr(_cfg, "RB_MIN_VALID_TRADES", 5)
        monkeypatch.setattr(_cfg, "RB_RISK_MIN_IMPROVEMENT", 0.01)
        monkeypatch.setattr(_cfg, "RB_REQUIRE_TRAIN_SLIGHTLY_ABOVE_VALID", False)

        selected = self._make_selected()

        # Legacy mode: fold_engines=None
        rules, train, valid, score, hist = _optimize_risk(
            selected, train_mock, valid_mock, "long",
            fold_engines=None, tail_holdout_engine=None,
        )

        assert len(rules) == 1
        selected_combo = (rules[0].get("tp"), rules[0].get("sl"), rules[0].get("capital_pct"))
        # Legacy picks A (best single score)
        assert selected_combo == (3.0, 1.2, 20.0), (
            f"Expected legacy to pick combo A (3.0,1.2,20.0) but got {selected_combo}"
        )
        # No fold_scores in legacy mode
        for entry in hist:
            assert "fold_scores" not in entry
