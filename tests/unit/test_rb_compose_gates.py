"""Verify RB Governor compose gates read config and branch correctly."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.rb_governor import CandidateRecord, _compose_ruleset


def _train_metrics(return_pct: float, *, trades: int = 30) -> dict:
    raw = max(trades + 5, 40)
    return {
        "total_return_pct": return_pct,
        "max_drawdown_pct": 2.0,
        "profit_factor": 1.5,
        "win_rate": 55.0,
        "executed_trades": trades,
        "raw_signal_count": raw,
        "skipped_min_notional_count": 2,
        "max_simultaneous_positions": 2,
    }


def _valid_metrics(return_pct: float, *, trades: int = 20) -> dict:
    raw = max(trades + 5, 40)
    return {
        "total_return_pct": return_pct,
        "max_drawdown_pct": 2.0,
        "profit_factor": 1.5,
        "win_rate": 55.0,
        "executed_trades": trades,
        "raw_signal_count": raw,
        "skipped_min_notional_count": 2,
        "max_simultaneous_positions": 2,
    }


def _cand(name: str, train_ret: float, valid_ret: float, mask: np.ndarray) -> CandidateRecord:
    rule = {"conditions": [f"[feat] IS High", f"symbol is {name}"], "tp": 2.0, "sl": 1.0, "capital_pct": 20.0}
    train_m = _train_metrics(train_ret)
    valid_m = _valid_metrics(valid_ret)
    return CandidateRecord(
        rule=rule,
        train_metrics=train_m,
        valid_metrics=valid_m,
        score=100.0,
        mask=mask,
    )


@pytest.fixture
def engines():
  return object(), object()


def test_config_compose_defaults_match_expected_bundle():
    assert _cfg.RB_RULE_ADD_BY_RETURN_ONLY is False
    assert _cfg.RB_RULESET_MUST_BEAT_SUBSETS is False
    assert _cfg.RB_RULE_ADD_IGNORE_OVERLAP is False
    assert _cfg.RB_MAX_PAIR_OVERLAP == 0.25
    assert _cfg.RB_MIN_SCORE_IMPROVEMENT == 0.03


def test_strict_mode_uses_score_improvement_not_return_only(engines):
    train_engine, valid_engine = engines
    mask_a = np.array([True, False, True, False], dtype=bool)
    mask_b = np.array([False, True, False, True], dtype=bool)
    candidates = [
        _cand("A", 5.0, 4.0, mask_a),
        _cand("B", 3.0, 3.0, mask_b),
    ]

    def fake_eval(_train_engine, _valid_engine, rules):
        n = len(rules)
        if n == 1:
            train_m = _train_metrics(5.0)
            valid_m = _valid_metrics(4.0)
            score = 100.0
        else:
            train_m = _train_metrics(6.0)
            valid_m = _valid_metrics(4.2)
            score = 100.05
        return train_m, valid_m, score

    with patch.object(_cfg, "RB_RULE_ADD_BY_RETURN_ONLY", False), patch.object(
        _cfg, "RB_RULESET_MUST_BEAT_SUBSETS", False
    ), patch.object(_cfg, "RB_RULE_ADD_IGNORE_OVERLAP", False), patch.object(
        _cfg, "RB_MAX_PAIR_OVERLAP", 0.25
    ), patch.object(_cfg, "RB_MIN_SCORE_IMPROVEMENT", 0.02), patch(
        "gpu_fuzzy_trader.rb_governor._evaluate_ruleset", side_effect=fake_eval
    ), patch("gpu_fuzzy_trader.rb_governor._is_positive_good", return_value=True):
        selected, _, _, _, _ = _compose_ruleset(candidates, train_engine, valid_engine, "long")

    assert len(selected) == 2


def test_score_improvement_threshold_blocks_marginal_add(engines):
    train_engine, valid_engine = engines
    mask_a = np.array([True, False], dtype=bool)
    mask_b = np.array([False, True], dtype=bool)
    candidates = [_cand("A", 5.0, 4.0, mask_a), _cand("B", 3.0, 3.0, mask_b)]

    def fake_eval(_train_engine, _valid_engine, rules):
        if len(rules) == 1:
            return _train_metrics(5.0), _valid_metrics(4.0), 100.0
        return _train_metrics(6.0), _valid_metrics(4.2), 100.01

    with patch.object(_cfg, "RB_RULE_ADD_BY_RETURN_ONLY", False), patch.object(
        _cfg, "RB_RULESET_MUST_BEAT_SUBSETS", False
    ), patch.object(_cfg, "RB_RULE_ADD_IGNORE_OVERLAP", False), patch.object(
        _cfg, "RB_MAX_PAIR_OVERLAP", 0.25
    ), patch.object(_cfg, "RB_MIN_SCORE_IMPROVEMENT", 0.02), patch(
        "gpu_fuzzy_trader.rb_governor._evaluate_ruleset", side_effect=fake_eval
    ), patch("gpu_fuzzy_trader.rb_governor._is_positive_good", return_value=True):
        selected, _, _, _, _ = _compose_ruleset(candidates, train_engine, valid_engine, "long")

    assert len(selected) == 1


def test_overlap_gate_rejects_high_overlap_when_enabled(engines):
    train_engine, valid_engine = engines
    mask_a = np.array([True, True, True, False], dtype=bool)
    mask_b = np.array([True, True, False, False], dtype=bool)
    candidates = [_cand("A", 5.0, 4.0, mask_a), _cand("B", 3.0, 3.0, mask_b)]

    def fake_eval(_train_engine, _valid_engine, rules):
        if len(rules) == 1:
            return _train_metrics(5.0), _valid_metrics(4.0), 100.0
        return _train_metrics(8.0), _valid_metrics(6.0), 150.0

    with patch.object(_cfg, "RB_RULE_ADD_BY_RETURN_ONLY", False), patch.object(
        _cfg, "RB_RULESET_MUST_BEAT_SUBSETS", False
    ), patch.object(_cfg, "RB_RULE_ADD_IGNORE_OVERLAP", False), patch.object(
        _cfg, "RB_MAX_PAIR_OVERLAP", 0.25
    ), patch.object(_cfg, "RB_MIN_SCORE_IMPROVEMENT", 0.01), patch(
        "gpu_fuzzy_trader.rb_governor._evaluate_ruleset", side_effect=fake_eval
    ), patch("gpu_fuzzy_trader.rb_governor._is_positive_good", return_value=True):
        selected, _, _, _, _ = _compose_ruleset(candidates, train_engine, valid_engine, "long")

    assert len(selected) == 1


def test_ignore_overlap_allows_high_overlap_candidate(engines):
    train_engine, valid_engine = engines
    mask_a = np.array([True, True, True, False], dtype=bool)
    mask_b = np.array([True, True, False, False], dtype=bool)
    candidates = [_cand("A", 5.0, 4.0, mask_a), _cand("B", 3.0, 3.0, mask_b)]

    def fake_eval(_train_engine, _valid_engine, rules):
        if len(rules) == 1:
            return _train_metrics(5.0), _valid_metrics(4.0), 100.0
        return _train_metrics(8.0), _valid_metrics(6.0), 150.0

    with patch.object(_cfg, "RB_RULE_ADD_BY_RETURN_ONLY", False), patch.object(
        _cfg, "RB_RULESET_MUST_BEAT_SUBSETS", False
    ), patch.object(_cfg, "RB_RULE_ADD_IGNORE_OVERLAP", True), patch.object(
        _cfg, "RB_MAX_PAIR_OVERLAP", 0.25
    ), patch.object(_cfg, "RB_MIN_SCORE_IMPROVEMENT", 0.01), patch(
        "gpu_fuzzy_trader.rb_governor._evaluate_ruleset", side_effect=fake_eval
    ), patch("gpu_fuzzy_trader.rb_governor._is_positive_good", return_value=True):
        selected, _, _, _, _ = _compose_ruleset(candidates, train_engine, valid_engine, "long")

    assert len(selected) == 2


def test_subset_beat_blocks_when_enabled(engines):
    train_engine, valid_engine = engines
    mask_a = np.array([True, False], dtype=bool)
    mask_b = np.array([False, True], dtype=bool)
    candidates = [_cand("A", 6.0, 5.0, mask_a), _cand("B", 5.0, 4.5, mask_b)]

    def fake_eval(_train_engine, _valid_engine, rules):
        if len(rules) == 1:
            return _train_metrics(6.0), _valid_metrics(5.0), 120.0
        return _train_metrics(6.5), _valid_metrics(4.0), 130.0

    with patch.object(_cfg, "RB_RULE_ADD_BY_RETURN_ONLY", False), patch.object(
        _cfg, "RB_RULESET_MUST_BEAT_SUBSETS", True
    ), patch.object(_cfg, "RB_MIN_TRAIN_RETURN_IMPROVEMENT", 0.0), patch.object(
        _cfg, "RB_MIN_VALID_RETURN_IMPROVEMENT", 0.0
    ), patch.object(_cfg, "RB_RULE_ADD_IGNORE_OVERLAP", True), patch.object(
        _cfg, "RB_MIN_SCORE_IMPROVEMENT", 0.01
    ), patch(
        "gpu_fuzzy_trader.rb_governor._evaluate_ruleset", side_effect=fake_eval
    ), patch("gpu_fuzzy_trader.rb_governor._is_positive_good", return_value=True):
        selected, _, _, _, _ = _compose_ruleset(candidates, train_engine, valid_engine, "long")

    assert len(selected) == 1


def test_ignore_subset_beat_only_applies_in_return_only_mode(engines):
    train_engine, valid_engine = engines
    mask_a = np.array([True, False], dtype=bool)
    mask_b = np.array([False, True], dtype=bool)
    candidates = [_cand("A", 6.0, 5.0, mask_a), _cand("B", 5.0, 4.5, mask_b)]

    def fake_eval(_train_engine, _valid_engine, rules):
        if len(rules) == 1:
            return _train_metrics(6.0), _valid_metrics(5.0), 0.0
        return _train_metrics(7.0), _valid_metrics(4.0), 0.0

    with patch.object(_cfg, "RB_RULE_ADD_BY_RETURN_ONLY", False), patch.object(
        _cfg, "RB_RULESET_MUST_BEAT_SUBSETS", True
    ), patch.object(_cfg, "RB_RULE_ADD_IGNORE_SUBSET_BEAT", True), patch.object(
        _cfg, "RB_RULE_ADD_IGNORE_OVERLAP", True
    ), patch.object(_cfg, "RB_MIN_TRAIN_RETURN_IMPROVEMENT", 0.0), patch.object(
        _cfg, "RB_MIN_VALID_RETURN_IMPROVEMENT", 0.0
    ), patch.object(_cfg, "RB_MIN_SCORE_IMPROVEMENT", 0.01), patch(
        "gpu_fuzzy_trader.rb_governor._evaluate_ruleset", side_effect=fake_eval
    ), patch("gpu_fuzzy_trader.rb_governor._is_positive_good", return_value=True):
        selected, _, _, _, _ = _compose_ruleset(candidates, train_engine, valid_engine, "long")

    assert len(selected) == 1


def test_return_only_mode_uses_combined_return_threshold(engines):
    train_engine, valid_engine = engines
    mask_a = np.array([True, False], dtype=bool)
    mask_b = np.array([False, True], dtype=bool)
    candidates = [_cand("A", 3.0, 2.0, mask_a), _cand("B", 2.0, 2.0, mask_b)]

    def fake_eval(_train_engine, _valid_engine, rules):
        if len(rules) == 1:
            return _train_metrics(3.0), _valid_metrics(2.0), 50.0
        return _train_metrics(3.02), _valid_metrics(2.02), 55.0

    with patch.object(_cfg, "RB_RULE_ADD_BY_RETURN_ONLY", True), patch.object(
        _cfg, "RB_RULE_ADD_IGNORE_SUBSET_BEAT", True
    ), patch.object(_cfg, "RB_RULE_ADD_IGNORE_OVERLAP", True), patch.object(
        _cfg, "RB_MIN_COMBINED_RETURN_IMPROVEMENT", 0.05
    ), patch(
        "gpu_fuzzy_trader.rb_governor._evaluate_ruleset", side_effect=fake_eval
    ), patch("gpu_fuzzy_trader.rb_governor._positive_returns", return_value=True):
        selected, _, _, _, _ = _compose_ruleset(candidates, train_engine, valid_engine, "long")

    assert len(selected) == 1


def _generalist_cand(
    name: str,
    train_ret: float,
    valid_ret: float,
    mask: np.ndarray,
    traded: list[str],
) -> CandidateRecord:
    """Pool-style rule: fuzzy conditions only, coverage via per_symbol_metrics."""
    per = {s: {"trade_count": 10, "net_pnl": 1.0, "win_rate": 50.0}
           for s in traded}
    train_m = _train_metrics(train_ret)
    valid_m = _valid_metrics(valid_ret)
    train_m["per_symbol_metrics"] = dict(per)
    valid_m["per_symbol_metrics"] = dict(per)
    return CandidateRecord(
        rule={"conditions": [f"[feat] IS {name}"],
              "tp": 2.0, "sl": 1.0, "capital_pct": 18.0},
        train_metrics=train_m,
        valid_metrics=valid_m,
        score=100.0,
        mask=mask,
    )


def test_generalist_mode_compose_uses_traded_symbol_coverage(engines):
    """Mode A: no symbol filters; MIN_DISTINCT must use metrics coverage, not filters."""
    train_engine, valid_engine = engines
    mask_a = np.array([True, False, True, False], dtype=bool)
    mask_b = np.array([False, True, False, True], dtype=bool)
    candidates = [
        _generalist_cand("High", 5.0, 4.0, mask_a, traded=["1", "2"]),
        _generalist_cand("Low", 4.0, 3.5, mask_b, traded=["3", "4"]),
    ]

    def fake_eval(_train_engine, _valid_engine, rules):
        if len(rules) == 1:
            return _train_metrics(5.0), _valid_metrics(4.0), 100.0
        return _train_metrics(6.5), _valid_metrics(5.0), 120.0

    with patch.object(_cfg, "RB_REQUIRE_SYMBOL_FILTERS", False), patch.object(
        _cfg, "RB_MIN_DISTINCT_SYMBOLS", 3
    ), patch.object(_cfg, "RB_RULE_ADD_BY_RETURN_ONLY", False), patch.object(
        _cfg, "RB_RULESET_MUST_BEAT_SUBSETS", False
    ), patch.object(_cfg, "RB_RULE_ADD_IGNORE_OVERLAP", False), patch.object(
        _cfg, "RB_MAX_PAIR_OVERLAP", 0.25
    ), patch.object(_cfg, "RB_MIN_SCORE_IMPROVEMENT", 0.02), patch(
        "gpu_fuzzy_trader.rb_governor._evaluate_ruleset", side_effect=fake_eval
    ), patch("gpu_fuzzy_trader.rb_governor._is_positive_good", return_value=True):
        selected, _, _, _, _ = _compose_ruleset(
            candidates, train_engine, valid_engine, "long")

    assert len(selected) == 2
    for rec in selected:
        assert not any(str(c).lower().startswith("symbol is ")
                       for c in rec.rule["conditions"])


def test_generalist_mode_without_metrics_coverage_used_to_block_all_adds(engines):
    """Regression: empty metrics + MIN_DISTINCT must not freeze compose forever
    when the seed already has enough coverage from its own metrics."""
    train_engine, valid_engine = engines
    mask_a = np.array([True, False], dtype=bool)
    mask_b = np.array([False, True], dtype=bool)
    # Seed already covers 3 symbols → MIN_DISTINCT satisfied → second rule can add on score
    seed = _generalist_cand("High", 5.0, 4.0, mask_a, traded=["1", "2", "3"])
    other = _generalist_cand("Low", 4.0, 3.5, mask_b, traded=["1", "2", "3"])

    def fake_eval(_train_engine, _valid_engine, rules):
        if len(rules) == 1:
            return _train_metrics(5.0), _valid_metrics(4.0), 100.0
        return _train_metrics(6.5), _valid_metrics(5.0), 120.0

    with patch.object(_cfg, "RB_REQUIRE_SYMBOL_FILTERS", False), patch.object(
        _cfg, "RB_MIN_DISTINCT_SYMBOLS", 3
    ), patch.object(_cfg, "RB_RULESET_MUST_BEAT_SUBSETS", False), patch.object(
        _cfg, "RB_MIN_SCORE_IMPROVEMENT", 0.02
    ), patch(
        "gpu_fuzzy_trader.rb_governor._evaluate_ruleset", side_effect=fake_eval
    ), patch("gpu_fuzzy_trader.rb_governor._is_positive_good", return_value=True):
        selected, _, _, _, _ = _compose_ruleset(
            [seed, other], train_engine, valid_engine, "long"
        )

    assert len(selected) == 2
