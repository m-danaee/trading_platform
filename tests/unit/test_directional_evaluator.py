"""Unit tests for directional and conditional evaluators and rule search profiles."""

import numpy as np
import pytest

from gpu_fuzzy_trader.evolution.directional_evaluator import (
    classify_directional_labels,
    compute_conditional_mwc_labels,
    compute_forward_movement_labels,
    evaluate_conditional_directional_rule,
    evaluate_directional_rule,
    fit_directional_threshold,
)
from gpu_fuzzy_trader.research_profile import (
    RuleSearchProfile,
    get_rule_search_profile,
)


def test_forward_movement_and_threshold():
    """Verify ATR-normalized forward movement calculation and threshold fitting."""
    close = np.array([100.0, 102.0, 105.0, 101.0, 108.0, 110.0])
    atr = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    # horizon = 2
    # move[0] = (105 - 100)/2 = 2.5
    # move[1] = (101 - 102)/2 = -0.5
    # move[2] = (108 - 105)/2 = 1.5
    # move[3] = (110 - 101)/2 = 4.5
    moves = compute_forward_movement_labels(close, atr, horizon_bars=2)
    assert np.isclose(moves[0], 2.5)
    assert np.isclose(moves[1], -0.5)
    assert np.isclose(moves[2], 1.5)
    assert np.isclose(moves[3], 4.5)
    assert np.isnan(moves[4])  # Warmup/tail
    assert np.isnan(moves[5])

    theta = fit_directional_threshold(moves[:4], quantile=0.50)
    assert theta > 0.0
    assert np.isclose(theta, 2.0)  # abs sorted: 0.5, 1.5, 2.5, 4.5 -> median = 2.0


def test_classify_directional_labels():
    """Verify classification of continuous moves into ternary directional labels (+1, -1, 0)."""
    moves = np.array([2.5, -0.5, 1.5, 4.5, -3.0, 0.0, np.nan])
    theta = 1.0
    labels = classify_directional_labels(moves, theta=theta)
    # moves > 1.0 -> +1 (idx 0, 2, 3)
    # moves < -1.0 -> -1 (idx 4)
    # abs(moves) <= 1.0 or nan -> 0 (idx 1, 5, 6)
    expected = np.array([1, 0, 1, 1, -1, 0, 0], dtype=np.int32)
    np.testing.assert_array_equal(labels, expected)


def test_evaluate_directional_rule_metrics():
    """Verify DirectionalEdge, MCC, and CoveragePenalty calculation for long rules."""
    labels = np.array([1, 1, -1, 0, 1, -1, 1, 0, 1, 1])  # 6 long, 2 short, 2 neutral
    active_mask = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 0], dtype=bool)  # 4 active, all long
    edge, mcc, cov_penalty = evaluate_directional_rule(
        active_mask, labels, direction="long", target_coverage=(0.20, 0.60)
    )
    # Active = 4, all target long -> precision = 1.0
    # Base rate = 6 / 10 = 0.60 -> edge = 0.40
    assert np.isclose(edge, 0.40)
    assert mcc > 0.0
    # MCC: TP=4, FP=0, FN=2, TN=4 -> (16 - 0) / sqrt(4 * 6 * 4 * 6) = 16 / 24 = 2/3
    assert np.isclose(mcc, 2.0 / 3.0)
    # Coverage: 4/10 = 40% which is inside [20%, 60%] -> penalty = 0.0
    assert cov_penalty == 0.0


def test_evaluate_directional_rule_short():
    """Verify evaluation for short direction."""
    labels = np.array([-1, -1, 1, 0, -1, 1, -1, 0, -1, -1])  # 6 short
    active_mask = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 0], dtype=bool)  # 4 active, all short
    edge, mcc, cov_penalty = evaluate_directional_rule(
        active_mask, labels, direction="short", target_coverage=(0.20, 0.60)
    )
    assert np.isclose(edge, 0.40)
    assert np.isclose(mcc, 2.0 / 3.0)
    assert cov_penalty == 0.0


def test_coverage_penalties_continuous():
    """Verify soft continuous coverage penalty outside target coverage range."""
    labels = np.array([1] * 100)
    target_coverage = (0.20, 0.60)

    # 1. Inside target coverage (40%) -> 0.0
    mask_in = np.zeros(100, dtype=bool)
    mask_in[:40] = True
    _, _, penalty_in = evaluate_directional_rule(
        mask_in, labels, direction="long", target_coverage=target_coverage
    )
    assert penalty_in == 0.0

    # 2. Below target coverage (10% vs min 20%) -> penalty = (0.20 - 0.10) / 0.20 = 0.50
    mask_low = np.zeros(100, dtype=bool)
    mask_low[:10] = True
    _, _, penalty_low = evaluate_directional_rule(
        mask_low, labels, direction="long", target_coverage=target_coverage
    )
    assert np.isclose(penalty_low, 0.50)

    # 3. Zero coverage -> penalty = (0.20 - 0.0) / 0.20 = 1.0
    mask_zero = np.zeros(100, dtype=bool)
    _, _, penalty_zero = evaluate_directional_rule(
        mask_zero, labels, direction="long", target_coverage=target_coverage
    )
    assert np.isclose(penalty_zero, 1.0)

    # 4. Above target coverage (80% vs max 60%) -> penalty = (0.80 - 0.60) / (1.0 - 0.60) = 0.20 / 0.40 = 0.50
    mask_high = np.zeros(100, dtype=bool)
    mask_high[:80] = True
    _, _, penalty_high = evaluate_directional_rule(
        mask_high, labels, direction="long", target_coverage=target_coverage
    )
    assert np.isclose(penalty_high, 0.50)

    # 5. 100% coverage -> penalty = 1.0
    mask_all = np.ones(100, dtype=bool)
    _, _, penalty_all = evaluate_directional_rule(
        mask_all, labels, direction="long", target_coverage=target_coverage
    )
    assert np.isclose(penalty_all, 1.0)


def test_directional_rule_edge_cases():
    """Verify robust handling of edge cases (empty masks, zero division, constant labels)."""
    # Empty inputs
    edge, mcc, cov_penalty = evaluate_directional_rule(
        np.array([], dtype=bool), np.array([], dtype=np.int32)
    )
    assert edge == 0.0
    assert mcc == 0.0
    assert cov_penalty == 1.0

    # Constant labels (no positive examples)
    labels = np.zeros(50, dtype=np.int32)
    mask = np.ones(50, dtype=bool)
    edge, mcc, cov_penalty = evaluate_directional_rule(mask, labels, direction="long")
    assert edge == 0.0
    assert mcc == 0.0


def test_conditional_mwc_evaluation():
    """Verify conditional MWC evaluation given upstream HWC score."""
    # 10 bars
    # HWC score: positive for bars 0..4 (long bias), negative for bars 5..9 (short bias)
    hwc_score = np.array([0.6, 0.7, 0.5, 0.8, 0.6, -0.6, -0.7, -0.5, -0.8, -0.6])
    # Forward moves
    moves = np.array([2.0, 1.5, -1.0, 3.0, 0.5, -2.0, -1.5, 1.0, -3.0, -0.5])
    theta_mwc = 1.0

    # Long continuation when HWC is bullish (hwc_score >= 0.20)
    cond_labels_long = compute_conditional_mwc_labels(
        moves, hwc_score, theta_mwc=theta_mwc, direction="long", support_threshold=0.20
    )
    # Bars 0..4 have HWC >= 0.20: moves are 2.0 (target +1), 1.5 (+1), -1.0 (0), 3.0 (+1), 0.5 (0)
    # Bars 5..9 have HWC < 0.20: conditioned out -> 0
    assert cond_labels_long[0] == 1
    assert cond_labels_long[1] == 1
    assert cond_labels_long[2] == 0
    assert cond_labels_long[3] == 1
    assert cond_labels_long[4] == 0
    assert np.all(cond_labels_long[5:] == 0)

    # Active mask on bars 0, 1, 3
    active_mask = np.zeros(10, dtype=bool)
    active_mask[[0, 1, 3]] = True
    edge, mcc, cov_penalty = evaluate_conditional_directional_rule(
        active_mask, moves, hwc_score, theta_mwc=theta_mwc, direction="long", support_threshold=0.20,
        target_coverage=(0.10, 0.40)
    )
    assert edge > 0.0
    assert mcc > 0.0
    # Conditional coverage is measured only over the five HWC-supported
    # observations, so 3/5 is above the soft upper target and is penalized.
    assert np.isclose(cov_penalty, (0.60 - 0.40) / (1.0 - 0.40))


def test_rule_search_profiles():
    """Verify RuleSearchProfile dataclass and factory functions for HWC, MWC, and LWC."""
    hwc = get_rule_search_profile("hwc")
    assert isinstance(hwc, RuleSearchProfile)
    assert hwc.role == "hwc"
    assert hwc.timeframe_minutes == 240
    assert hwc.min_conditions == 1
    assert hwc.max_conditions == 2
    assert hwc.target_coverage == (0.20, 0.60)
    assert hwc.forward_horizon_bars == 6
    assert hwc.quantile == 0.60

    mwc = get_rule_search_profile("mwc")
    assert isinstance(mwc, RuleSearchProfile)
    assert mwc.role == "mwc"
    assert mwc.timeframe_minutes == 60
    assert mwc.min_conditions == 1
    assert mwc.max_conditions == 3
    assert mwc.target_coverage == (0.10, 0.40)
    assert mwc.forward_horizon_bars == 4
    assert mwc.quantile == 0.60

    lwc = get_rule_search_profile("lwc")
    assert isinstance(lwc, RuleSearchProfile)
    assert lwc.role == "lwc"
    assert lwc.timeframe_minutes == 15
    assert lwc.min_conditions == 2
    assert lwc.max_conditions == 4

    # Test helper methods on RuleSearchProfile
    assert RuleSearchProfile.hwc() == hwc
    assert RuleSearchProfile.mwc() == mwc
    assert RuleSearchProfile.lwc() == lwc

    # Test serialization
    data = hwc.as_dict()
    assert data["role"] == "hwc"
    assert data["timeframe_minutes"] == 240
