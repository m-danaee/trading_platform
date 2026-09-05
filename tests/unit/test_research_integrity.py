from __future__ import annotations

import json

import pandas as pd
import pytest

from gpu_fuzzy_trader.phases.rule_identity import strategy_id
from gpu_fuzzy_trader.research_integrity import (
    ExperimentLedger,
    reserve_forward_evaluation,
    write_forward_acceptance_record,
)
from gpu_fuzzy_trader.research_profile import ResearchProfile
from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.validation.nested_walk_forward import build_nested_folds


def test_strategy_identity_excludes_capital_but_includes_exit():
    base = [{
        "conditions": ["[feature] IS Bullish", "symbol is BTCUSDT"],
        "eligible_symbols": ["BTCUSDT"],
        "tp": 2.0,
        "sl": 1.2,
        "capital_pct": 5.0,
    }]
    changed_size = [dict(base[0], capital_pct=18.0)]
    changed_exit = [dict(base[0], tp=3.0)]
    assert strategy_id(
        direction="long",
        rules=base,
        horizon_bars=48,
        cost_model_id="test",
    ) == strategy_id(
        direction="long",
        rules=changed_size,
        horizon_bars=48,
        cost_model_id="test",
    )
    assert strategy_id(
        direction="long",
        rules=base,
        horizon_bars=48,
        cost_model_id="test",
    ) != strategy_id(
        direction="long",
        rules=changed_exit,
        horizon_bars=48,
        cost_model_id="test",
    )


def test_forward_acceptance_is_one_shot(tmp_path):
    forward = tmp_path / "forward.csv"
    pd.DataFrame({
        "datetime": ["2026-08-01T00:00:00Z"],
        "symbol": ["BTCUSDT"],
    }).to_csv(forward, index=False)
    metadata = reserve_forward_evaluation(forward, tmp_path)
    write_forward_acceptance_record(
        tmp_path,
        metadata,
        {"status": "accepted"},
    )
    with pytest.raises(RuntimeError, match="already been evaluated"):
        reserve_forward_evaluation(forward, tmp_path)


def test_experiment_ledger_appends_jsonl(tmp_path):
    ledger = ExperimentLedger(tmp_path)
    ledger.append({"record_type": "test", "trial_count": 3})
    ledger.append({"record_type": "test", "trial_count": 4})
    rows = [
        json.loads(line)
        for line in ledger.path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["trial_count"] for row in rows] == [3, 4]


def test_nested_folds_purge_label_horizon():
    frame = pd.DataFrame({
        "datetime": pd.date_range("2025-01-01", periods=30, freq="h"),
        "symbol": ["BTCUSDT"] * 30,
        "_symbol_bar_index": list(range(30)),
    })
    folds = build_nested_folds(
        frame,
        n_outer=3,
        min_train_fraction=0.4,
        purge_candles=3,
    )
    assert len(folds) == 3
    for fold in folds:
        assert fold.inner_train_df["_symbol_bar_index"].max() < (
            fold.outer_valid_df["_symbol_bar_index"].min() - 2
        )


def test_research_profile_is_stable_and_versioned():
    profile = ResearchProfile.from_config(_cfg)
    assert profile.schema_version == 5
    assert len(profile.profile_id) == 20
    assert profile.rb_risk_optimize_exits is False
    assert profile.rule_exclude_raw_ohlcv is True
    assert profile.rule_allowed_ff_features == _cfg.RULE_ALLOWED_FF_FEATURES
