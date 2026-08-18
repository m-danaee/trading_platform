"""Unit tests for MTF Composer, Asymmetric Soft Veto, and Trade Retention Guard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gpu_fuzzy_trader.mtf.composer import (
    compose_hierarchical_signals,
    compose_bidirectional_signals,
    compute_trade_retention_diagnostics,
)
from gpu_fuzzy_trader.mtf.diagnostics import (
    MIN_RETENTION_SAMPLE,
    compute_granular_retention_diagnostics,
    format_retention_report,
)


def test_asymmetric_soft_veto_and_retention():
    # 4 raw LWC long triggers
    lwc_triggers = np.array([1, 1, 1, 1, 0], dtype=np.int8)

    # t0: HWC supportive (+0.8, 0.4) -> Accepted
    # t1: HWC neutral (0.0, 0.0) -> Accepted
    # t2: HWC opposing (-0.75, 0.3) > V_HWC_LONG (0.65) -> Vetoed by HWC
    # t3: MWC opposing (-0.70, 0.3) > V_MWC_LONG (0.60) -> Vetoed by MWC
    # t4: No trigger -> No Trade
    hwc_dir = np.array([0.8, 0.0, -0.75, 0.2, 0.0])
    hwc_str = np.array([0.4, 0.0, 0.3, 0.2, 0.0])
    mwc_dir = np.array([0.5, 0.1, 0.0, -0.70, 0.0])
    mwc_str = np.array([0.3, 0.1, 0.0, 0.3, 0.0])

    signals, stats = compose_hierarchical_signals(
        lwc_triggers=lwc_triggers,
        direction="long",
        hwc_direction=hwc_dir,
        hwc_strength=hwc_str,
        mwc_direction=mwc_dir,
        mwc_strength=mwc_str,
        v_hwc=0.65,
        v_mwc=0.60,
        min_strength_hwc=0.15,
        min_strength_mwc=0.15,
    )

    assert (signals == np.array([1, 1, 0, 0, 0])).all()
    diag = compute_trade_retention_diagnostics(stats)
    assert diag["raw_triggers"] == 4
    assert diag["hwc_vetoed"] == 1
    assert diag["hwc_survived"] == 3
    assert diag["mwc_vetoed"] == 1
    assert diag["accepted_trades"] == 2
    assert np.isclose(diag["retention_ratio"], 0.50)
    assert np.isclose(diag["hwc_veto_rate"], 1 / 4)
    assert np.isclose(diag["mwc_incremental_veto_rate"], 1 / 3)


def test_short_direction_soft_veto():
    # 4 raw LWC short triggers
    lwc_triggers = np.array([1, 1, 1, 1, 0], dtype=np.int8)

    # For short, opposing is bullish bias (> +threshold)
    # t0: HWC opposing (+0.70, 0.3) > V_HWC_SHORT (0.60) -> Vetoed by HWC
    # t1: HWC bearish (-0.80, 0.4) -> Survived HWC, MWC bearish (-0.5, 0.3) -> Accepted
    # t2: HWC neutral (0.1, 0.1) -> Survived HWC, MWC opposing (+0.65, 0.25) > V_MWC_SHORT (0.55) -> Vetoed by MWC
    # t3: HWC bearish (-0.4, 0.2), MWC neutral (0.0, 0.0) -> Accepted
    hwc_dir = np.array([0.70, -0.80, 0.10, -0.40, 0.0])
    hwc_str = np.array([0.30, 0.40, 0.10, 0.20, 0.0])
    mwc_dir = np.array([0.00, -0.50, 0.65, 0.00, 0.0])
    mwc_str = np.array([0.00, 0.30, 0.25, 0.00, 0.0])

    signals, stats = compose_hierarchical_signals(
        lwc_triggers=lwc_triggers,
        direction="short",
        hwc_direction=hwc_dir,
        hwc_strength=hwc_str,
        mwc_direction=mwc_dir,
        mwc_strength=mwc_str,
        v_hwc=0.60,
        v_mwc=0.55,
        min_strength_hwc=0.15,
        min_strength_mwc=0.15,
    )

    assert (signals == np.array([0, 1, 0, 1, 0])).all()
    diag = compute_trade_retention_diagnostics(stats)
    assert diag["raw_triggers"] == 4
    assert diag["hwc_vetoed"] == 1
    assert diag["hwc_survived"] == 3
    assert diag["mwc_vetoed"] == 1
    assert diag["accepted_trades"] == 2
    assert np.isclose(diag["retention_ratio"], 0.50)


def test_asymmetric_veto_threshold_parameters():
    # Long triggers
    lwc_long = np.array([1, 1], dtype=np.int8)
    hwc_dir = np.array([-0.62, -0.62])
    hwc_str = np.array([0.30, 0.30])
    mwc_dir = np.array([0.0, 0.0])
    mwc_str = np.array([0.0, 0.0])

    # If v_hwc_long = 0.65, -0.62 does not reach -0.65 -> Accepted
    sig_long, stats_long = compose_hierarchical_signals(
        lwc_triggers=lwc_long,
        direction="long",
        hwc_direction=hwc_dir,
        hwc_strength=hwc_str,
        mwc_direction=mwc_dir,
        mwc_strength=mwc_str,
        v_hwc_long=0.65,
        v_hwc_short=0.60,
        v_mwc_long=0.60,
        v_mwc_short=0.55,
    )
    assert (sig_long == np.array([1, 1])).all()
    assert stats_long["hwc_vetoed"] == 0

    # Short triggers: opposing is > +threshold. If hwc_dir is +0.62 and v_hwc_short = 0.60 -> Vetoed
    lwc_short = np.array([1, 1], dtype=np.int8)
    hwc_dir_pos = np.array([0.62, 0.62])
    sig_short, stats_short = compose_hierarchical_signals(
        lwc_triggers=lwc_short,
        direction="short",
        hwc_direction=hwc_dir_pos,
        hwc_strength=hwc_str,
        mwc_direction=mwc_dir,
        mwc_strength=mwc_str,
        v_hwc_long=0.65,
        v_hwc_short=0.60,
        v_mwc_long=0.60,
        v_mwc_short=0.55,
    )
    assert (sig_short == np.array([0, 0])).all()
    assert stats_short["hwc_vetoed"] == 2


def test_evidence_strength_gate():
    # Direction is opposing (-0.90) but strength is weak (0.05 < 0.15) -> NOT Vetoed
    lwc_triggers = np.array([1, 1], dtype=np.int8)
    hwc_dir = np.array([-0.90, -0.90])
    hwc_str = np.array([0.05, 0.20])  # t0 weak strength, t1 strong strength
    mwc_dir = np.array([0.0, 0.0])
    mwc_str = np.array([0.0, 0.0])

    signals, stats = compose_hierarchical_signals(
        lwc_triggers=lwc_triggers,
        direction="long",
        hwc_direction=hwc_dir,
        hwc_strength=hwc_str,
        mwc_direction=mwc_dir,
        mwc_strength=mwc_str,
        v_hwc=0.65,
        v_mwc=0.60,
        min_strength_hwc=0.15,
        min_strength_mwc=0.15,
    )

    assert signals[0] == 1  # Accepted because strength is too low to veto
    assert signals[1] == 0  # Vetoed because strength >= 0.15
    assert stats["hwc_vetoed"] == 1


def test_bidirectional_signals_composition():
    # Array with +1 (long), -1 (short), and 0 (no trade)
    lwc_signals = np.array([1, -1, 1, -1, 0], dtype=np.int8)
    # t0: Long trigger, HWC supportive (+0.8, 0.3) -> +1
    # t1: Short trigger, HWC opposing (+0.7, 0.3) -> 0 (vetoed)
    # t2: Long trigger, MWC opposing (-0.7, 0.3) -> 0 (vetoed)
    # t3: Short trigger, HWC/MWC bearish (-0.8, 0.3 / -0.6, 0.3) -> -1
    # t4: No trigger -> 0
    hwc_dir = np.array([0.8, 0.7, 0.0, -0.8, 0.0])
    hwc_str = np.array([0.3, 0.3, 0.0, 0.3, 0.0])
    mwc_dir = np.array([0.0, 0.0, -0.7, -0.6, 0.0])
    mwc_str = np.array([0.0, 0.0, 0.3, 0.3, 0.0])

    composed, stats = compose_bidirectional_signals(
        lwc_triggers=lwc_signals,
        hwc_direction=hwc_dir,
        hwc_strength=hwc_str,
        mwc_direction=mwc_dir,
        mwc_strength=mwc_str,
        v_hwc_long=0.65,
        v_hwc_short=0.60,
        v_mwc_long=0.60,
        v_mwc_short=0.55,
        min_strength_hwc=0.15,
        min_strength_mwc=0.15,
    )

    assert (composed == np.array([1, 0, 0, -1, 0])).all()
    assert stats["long"]["raw_triggers"] == 2
    assert stats["long"]["accepted_trades"] == 1
    assert stats["short"]["raw_triggers"] == 2
    assert stats["short"]["accepted_trades"] == 1
    assert stats["total"]["raw_triggers"] == 4
    assert stats["total"]["accepted_trades"] == 2


def test_retention_diagnostics_guardrails():
    # 1. Insufficient sample size (< 15 triggers)
    stats_small = {
        "raw_triggers": 10,
        "hwc_vetoed": 6,
        "hwc_survived": 4,
        "mwc_vetoed": 2,
        "accepted_trades": 2,
    }
    diag_small = compute_trade_retention_diagnostics(stats_small, min_retention_sample=15)
    assert diag_small["status"] == "INSUFFICIENT_SUPPORT"
    assert diag_small["has_sufficient_support"] is False
    assert diag_small["passes_floor"] is True  # does not hard-fail low sample

    # 2. Sufficient sample (>= 15), but retention ratio < 50% -> FAIL_RETENTION_FLOOR
    stats_fail = {
        "raw_triggers": 100,
        "hwc_vetoed": 40,
        "hwc_survived": 60,
        "mwc_vetoed": 20,
        "accepted_trades": 40,  # 40% retention < 50%
    }
    diag_fail = compute_trade_retention_diagnostics(stats_fail, retention_floor=0.50)
    assert diag_fail["status"] == "FAIL_RETENTION_FLOOR"
    assert diag_fail["has_sufficient_support"] is True
    assert diag_fail["passes_floor"] is False
    assert np.isclose(diag_fail["retention_ratio"], 0.40)

    # 3. Sufficient sample, retention >= 60% -> TARGET_MET / PASS
    stats_pass = {
        "raw_triggers": 100,
        "hwc_vetoed": 20,
        "hwc_survived": 80,
        "mwc_vetoed": 15,
        "accepted_trades": 65,  # 65% retention >= 60%
    }
    diag_pass = compute_trade_retention_diagnostics(stats_pass, retention_floor=0.50, retention_target=0.60)
    assert diag_pass["status"] in ("PASS", "TARGET_MET")
    assert diag_pass["passes_floor"] is True
    assert np.isclose(diag_pass["retention_ratio"], 0.65)


def test_granular_retention_diagnostics():
    # DataFrame with multiple symbols, directions, folds, and dates
    n = 60
    dates = pd.date_range("2024-01-01", periods=n, freq="1D")
    df = pd.DataFrame({
        "datetime": dates,
        "symbol": ["BTCUSDT"] * 30 + ["ETHUSDT"] * 30,
        "direction": ["long", "short"] * 30,
        "fold_id": [1] * 20 + [2] * 20 + [3] * 20,
        "lwc_trigger": [1] * n,
        "hwc_veto": [0, 1] * 30,  # 30 vetoed by HWC
        "mwc_veto": [0, 0, 1, 0] * 15,  # 15 vetoed by MWC
    })
    # Surviving / accepted:
    # row vetoed if hwc_veto or mwc_veto
    df["accepted"] = ((df["hwc_veto"] == 0) & (df["mwc_veto"] == 0)).astype(int)

    granular = compute_granular_retention_diagnostics(
        df,
        trigger_col="lwc_trigger",
        hwc_veto_col="hwc_veto",
        mwc_veto_col="mwc_veto",
        accepted_col="accepted",
        datetime_col="datetime",
        symbol_col="symbol",
        direction_col="direction",
        fold_col="fold_id",
    )

    assert "overall" in granular
    assert "by_direction" in granular
    assert "by_symbol" in granular
    assert "by_fold" in granular
    assert "by_month" in granular

    assert "long" in granular["by_direction"]
    assert "short" in granular["by_direction"]
    assert "BTCUSDT" in granular["by_symbol"]
    assert "ETHUSDT" in granular["by_symbol"]

    # Report text generator
    report_text = format_retention_report(granular)
    assert isinstance(report_text, str)
    assert "Trade Retention Funnel Diagnostics" in report_text
    assert "BTCUSDT" in report_text


def test_empty_and_zero_trigger_edge_cases():
    # Empty triggers
    empty_trig = np.array([], dtype=np.int8)
    empty_vec = np.array([], dtype=np.float64)

    sigs, stats = compose_hierarchical_signals(
        lwc_triggers=empty_trig,
        direction="long",
        hwc_direction=empty_vec,
        hwc_strength=empty_vec,
        mwc_direction=empty_vec,
        mwc_strength=empty_vec,
    )
    assert len(sigs) == 0
    assert stats["raw_triggers"] == 0
    assert stats["accepted_trades"] == 0

    diag = compute_trade_retention_diagnostics(stats)
    assert diag["raw_triggers"] == 0
    assert diag["status"] == "INSUFFICIENT_SUPPORT"

    # All zero triggers
    zeros = np.zeros(10, dtype=np.int8)
    dir_vec = np.ones(10, dtype=np.float64)
    str_vec = np.ones(10, dtype=np.float64)

    sigs, stats = compose_hierarchical_signals(
        lwc_triggers=zeros,
        direction="long",
        hwc_direction=dir_vec,
        hwc_strength=str_vec,
        mwc_direction=dir_vec,
        mwc_strength=str_vec,
    )
    assert (sigs == 0).all()
    assert stats["raw_triggers"] == 0
    assert stats["accepted_trades"] == 0
