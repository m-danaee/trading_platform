"""Tests for RB correlation diagnostics and portfolio-aware selection."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.portfolio.clustering import (
    adjusted_quality,
    greedy_adjusted_quality,
    threshold_graph_clusters,
)
from gpu_fuzzy_trader.portfolio.correlation import (
    downside_dependence,
    pnl_correlation,
    signal_overlap,
)
from gpu_fuzzy_trader.portfolio.redundancy import (
    redundancy_matrix,
    stable_corr,
)
from gpu_fuzzy_trader.rb_governor import CandidateRecord, _compose_ruleset


def _metrics(return_pct: float) -> dict:
    return {
        "total_return_pct": return_pct,
        "max_drawdown_pct": 2.0,
        "profit_factor": 1.5,
        "win_rate": 55.0,
        "executed_trades": 20,
        "raw_signal_count": 25,
        "skipped_min_notional_count": 0,
        "max_simultaneous_positions": 1,
    }


def _candidate(name: str, mask: np.ndarray, pnl: list[float], score: float) -> CandidateRecord:
    return CandidateRecord(
        rule={"conditions": [f"[feature] IS {name}"]},
        train_metrics=_metrics(score),
        valid_metrics=_metrics(score),
        score=score,
        mask=mask,
        pnl_series=np.asarray(pnl, dtype=float),
    )


def test_signal_overlap_is_jaccard_and_pnl_is_pearson():
    assert signal_overlap([1, 1, 0, 0], [1, 0, 1, 0]) == 1.0 / 3.0
    assert pnl_correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == 1.0
    assert pnl_correlation([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == -1.0
    assert downside_dependence([1.0, -2.0, 1.0], [2.0, -4.0, 2.0]) == 1.0


def test_redundancy_matrix_uses_signal_and_positive_pnl_correlation():
    signals = [
        np.array([1, 1, 0, 0], dtype=bool),
        np.array([1, 1, 0, 0], dtype=bool),
        np.array([0, 0, 1, 1], dtype=bool),
    ]
    pnl = [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 2.0, 1.0]]
    matrix = redundancy_matrix(signals, pnl)

    assert matrix.shape == (3, 3)
    assert matrix[0, 0] == 1.0
    assert matrix[0, 1] == 1.0
    assert matrix[0, 2] == 0.0
    assert matrix[2, 0] == matrix[0, 2]


def test_stable_corr_is_median_plus_alpha_std():
    values = np.asarray([0.2, 0.8, 0.5], dtype=float)
    expected = float(np.median(values) + 0.5 * np.std(values))
    assert stable_corr(values, alpha=0.5) == expected


def test_threshold_clusters_and_greedy_selection_pick_across_clusters():
    matrix = np.array(
        [
            [1.0, 0.95, 0.10, 0.05, 0.05],
            [0.95, 1.0, 0.10, 0.05, 0.05],
            [0.10, 0.10, 1.0, 0.90, 0.05],
            [0.05, 0.05, 0.90, 1.0, 0.05],
            [0.05, 0.05, 0.05, 0.05, 1.0],
        ]
    )
    clusters = threshold_graph_clusters(matrix, threshold=0.75)
    assert clusters == [[0, 1], [2, 3], [4]]

    selected = greedy_adjusted_quality(
        [5.0, 4.9, 4.0, 3.9, 3.0],
        matrix,
        lambda_=10.0,
        max_items=3,
        clusters=clusters,
        require_cross_cluster=True,
    )
    assert selected == [0, 3, 4]


def test_correlation_flags_are_defined_and_validated():
    assert isinstance(_cfg.RB_CORRELATION_AWARE_SELECTION, bool)
    assert _cfg.RB_SIGNAL_OVERLAP_WEIGHT == 0.5
    assert _cfg.RB_PNL_CORR_WEIGHT == 0.5
    assert _cfg.RB_REDUNDANCY_PENALTY == 0.0
    _cfg.validate_config()


def test_adjusted_quality_supports_zero_low_and_medium_penalties():
    assert adjusted_quality(10.0, 0.4, lambda_=0) == 10.0
    assert adjusted_quality(10.0, 0.4, lambda_="low") == 9.9
    assert adjusted_quality(10.0, 0.4, lambda_="medium") == 9.8


def test_compose_uses_cross_cluster_picks_when_correlation_selection_enabled():
    candidates = [
        _candidate("A", np.array([1, 1, 0, 0], dtype=bool), [1, 2, 3], 5.0),
        _candidate("B", np.array([1, 1, 0, 0], dtype=bool), [2, 4, 6], 4.9),
        _candidate("C", np.array([0, 0, 1, 1], dtype=bool), [3, 1, 2], 4.0),
    ]

    def fake_eval(_train_engine, _valid_engine, rules):
        n = len(rules)
        return _metrics(5.0 + n), _metrics(4.0 + n), 100.0 + n

    with patch.object(_cfg, "RB_CORRELATION_AWARE_SELECTION", True), patch.object(
        _cfg, "RB_CORRELATION_CLUSTER_THRESHOLD", 0.75
    ), patch.object(_cfg, "RB_REDUNDANCY_PENALTY", 10.0), patch.object(
        _cfg, "RB_MAX_RULES", 2
    ), patch.object(_cfg, "RB_MAX_PAIR_OVERLAP", 1.0), patch.object(
        _cfg, "RB_MIN_SCORE_IMPROVEMENT", 0.0
    ), patch.object(_cfg, "RB_RULESET_MUST_BEAT_SUBSETS", False), patch.object(
        _cfg, "RB_MIN_TRAIN_RETURN_IMPROVEMENT", 0.0
    ), patch.object(_cfg, "RB_MIN_VALID_RETURN_IMPROVEMENT", 0.0), patch.object(
        _cfg, "RB_MIN_DISTINCT_SYMBOLS", 0
    ), patch(
        "gpu_fuzzy_trader.rb_governor._evaluate_ruleset", side_effect=fake_eval
    ), patch("gpu_fuzzy_trader.rb_governor._is_positive_good", return_value=True):
        selected, _, _, _, history = _compose_ruleset(
            candidates, object(), object(), "long"
        )

    assert [record.rule["conditions"][0] for record in selected] == [
        "[feature] IS A",
        "[feature] IS C",
    ]
    assert history[0]["correlation_report"]["report_only"] is False
    assert history[0]["correlation_report"]["clusters"] == [[0, 1], [2]]


def test_disabled_correlation_is_report_only():
    candidates = [
        _candidate("A", np.array([1, 1, 0, 0], dtype=bool), [1, 2, 3], 5.0),
        _candidate("B", np.array([1, 1, 0, 0], dtype=bool), [2, 4, 6], 4.9),
    ]

    def fake_eval(_train_engine, _valid_engine, rules):
        n = len(rules)
        return _metrics(5.0 + n), _metrics(4.0 + n), 100.0 + n

    with patch.object(_cfg, "RB_CORRELATION_AWARE_SELECTION", False), patch.object(
        _cfg, "RB_MAX_RULES", 2
    ), patch.object(_cfg, "RB_MAX_PAIR_OVERLAP", 1.0), patch.object(
        _cfg, "RB_MIN_SCORE_IMPROVEMENT", 0.0
    ), patch.object(_cfg, "RB_RULESET_MUST_BEAT_SUBSETS", False), patch.object(
        _cfg, "RB_MIN_TRAIN_RETURN_IMPROVEMENT", 0.0
    ), patch.object(_cfg, "RB_MIN_VALID_RETURN_IMPROVEMENT", 0.0), patch.object(
        _cfg, "RB_MIN_DISTINCT_SYMBOLS", 0
    ), patch(
        "gpu_fuzzy_trader.rb_governor._evaluate_ruleset", side_effect=fake_eval
    ), patch("gpu_fuzzy_trader.rb_governor._is_positive_good", return_value=True):
        selected, _, _, _, history = _compose_ruleset(
            candidates, object(), object(), "long"
        )

    assert len(selected) == 2
    assert history[0]["correlation_report"]["report_only"] is True
