"""Regression tests for the bounded validation-only RB recovery pass."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.rb_governor import (
    CandidateRecord,
    run_rb_governor_pipeline,
)


def _frame(rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=rows, freq="5min"),
            "symbol": ["BTCUSDT", "ETHUSDT"] * (rows // 2),
            "feat": [0.5, -0.5] * (rows // 2),
        }
    )


def _metrics(return_pct: float) -> dict:
    return {
        "total_return_pct": return_pct,
        "profit_factor": 1.5,
        "max_drawdown_pct": 1.0,
        "executed_trades": 30,
        "per_symbol_metrics": {
            "BTCUSDT": {"trade_count": 15, "net_pnl": 8.0, "win_rate": 55.0},
            "ETHUSDT": {"trade_count": 15, "net_pnl": 7.0, "win_rate": 54.0},
        },
    }


def test_recovery_retries_only_rejected_direction_on_full_validation(tmp_path: Path):
    train_df = _frame(8)
    val_df = _frame(8)
    val_selection = val_df.iloc[4:].reset_index(drop=True)
    ctx = list(_cfg.mandatory_context_conditions("long"))
    rule = {
        "conditions": ["[feat] IS High", *ctx],
        "tp": 2.0,
        "sl": 1.2,
        "capital_pct": 5.0,
    }
    candidate = CandidateRecord(
        rule=rule,
        train_metrics=_metrics(3.0),
        valid_metrics=_metrics(3.0),
        score=10.0,
    )

    def filter_by_frame(_pool, _train, valid, _direction, **_kwargs):
        # The half-window is intentionally too sparse for this synthetic
        # direction; the complete validation frame supplies the candidate.
        return [candidate] if len(valid) == len(val_df) else []

    with patch.object(_cfg, "RB_FULL_VALIDATION_RECOVERY_ENABLED", True), patch(
        "gpu_fuzzy_trader.rb_governor.CPUBacktestEngine"
    ), patch(
        "gpu_fuzzy_trader.rb_governor._filter_good_rules",
        side_effect=filter_by_frame,
    ), patch(
        "gpu_fuzzy_trader.rb_governor._make_walk_forward_fold_engines",
        return_value=([], None),
    ), patch(
        "gpu_fuzzy_trader.rb_governor._compose_ruleset",
        return_value=([candidate], _metrics(3.0), _metrics(3.0), 10.0, []),
    ), patch(
        "gpu_fuzzy_trader.rb_governor._optimize_risk",
        return_value=([rule], _metrics(3.0), _metrics(3.0), 10.0, []),
    ), patch(
        "gpu_fuzzy_trader.rb_governor._run_profit_amplifier",
        return_value=([rule], _metrics(3.0), _metrics(3.0), 10.0, {"accepted": False}),
    ), patch(
        "gpu_fuzzy_trader.rb_governor._write_clean_evaluator"
    ):
        result = run_rb_governor_pipeline(
            train_df,
            val_df,
            {"long": [{"conditions": ["[feat] IS High", *_cfg.mandatory_context_conditions("long")]}]},
            ("long",),
            output_dir=tmp_path,
            val_selection_df=val_selection,
        )

    strategy = result["long"]
    assert strategy["deployment_accepted"] is True
    assert strategy["validation_recovery"] == {
        "used": True,
        "selection_frame": "complete_validation_holdout",
    }
    saved = json.loads((tmp_path / "long.json").read_text())
    assert saved["validation_recovery"]["used"] is True

