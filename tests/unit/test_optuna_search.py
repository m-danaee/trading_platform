"""Validation-only Optuna contract tests."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import optuna
import pytest

from gpu_fuzzy_trader import config as cfg
from gpu_fuzzy_trader import optuna_search as search


def test_search_space_contains_only_live_active_config_keys() -> None:
    assert search.SEARCH_SPACE
    assert "PHASE2_MIN_PROFITABLE_SYMBOLS" not in search.SEARCH_SPACE
    for key in search.SEARCH_SPACE:
        assert hasattr(cfg, key), key
        assert not key.startswith(("PHASE3_", "PHASE4_"))
        assert not key.startswith("test_")


def test_disabled_phase1_parameter_is_not_sampled() -> None:
    assert "PHASE1_TOP_K_FEATURES" not in search._active_search_space()


def test_trial_sampling_derives_coherent_stage_budgets() -> None:
    previous = search._fast_mode
    search._fast_mode = False
    try:
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        params = search.sample_trial_params(trial)
        assert params["PHASE2_STAGE_A_GENERATIONS"] + params["PHASE2_STAGE_B_GENERATIONS"] == params["PHASE2_GENERATIONS"]
        assert params["PHASE2_STAGE_A_MUTATION_RATE"] >= params["PHASE2_STAGE_B_MUTATION_RATE"]
        assert "PHASE2_STAGE_B_GENERATIONS" not in search.SEARCH_SPACE
    finally:

        search._fast_mode = previous


def test_every_sampled_trial_is_validated_before_execution() -> None:
    previous = search._fast_mode
    search._fast_mode = False
    try:
        study = optuna.create_study(direction="maximize")
        params = search.sample_trial_params(study.ask())
        patchers = search.apply_trial_config(params)
        try:
            cfg.validate_config()
        finally:
            for patcher in reversed(patchers):
                patcher.stop()
    finally:
        search._fast_mode = previous


def test_nested_trial_config_disables_tail_selection_during_trials() -> None:
    patchers = search.apply_trial_config({}, nested_validation=True)
    try:
        assert cfg.RB_NESTED_VALIDATION_SELECTION_ONLY is True
        assert cfg.RB_RISK_GRID_USE_TAIL_HOLDOUT is False
        assert cfg.RB_TAIL_HOLDOUT_SELECTION_GATE is False
        assert cfg.RB_TAIL_HOLDOUT_HARD_GATE is False
        assert cfg.RB_FULL_VALIDATION_RECOVERY_ENABLED is False
    finally:
        for patcher in reversed(patchers):
            patcher.stop()


def test_best_trial_tail_is_evaluated_once_after_freeze(tmp_path: Path) -> None:
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    study.tell(trial, 3.5)
    trial_dir = tmp_path / "trial_0"
    reports_dir = trial_dir / "reports"
    reports_dir.mkdir(parents=True)
    tail_records = [
        {
            "datetime": "2024-01-01T00:00:00",
            "symbol": "BTCUSDT",
            "feature": 1.0,
        }
    ]
    for direction in ("long", "short"):
        (reports_dir / f"optuna_reserved_tail_{direction}.json").write_text(
            json.dumps(tail_records),
            encoding="utf-8",
        )
        (trial_dir / f"{direction}.json").write_text(
            json.dumps({
                "rules_set": [{"conditions": ["[feature] IS High"]}],
            }),
            encoding="utf-8",
        )

    with patch(
        "gpu_fuzzy_trader.backtest.cpu_engine.CPUBacktestEngine"
    ) as engine_type:
        engine_type.return_value.simulate_rule_set.return_value = {
            "total_return_pct": 1.0,
            "executed_trades": 1,
        }
        result = search.evaluate_best_trial_tail(
            study,
            output_root=str(tmp_path),
        )

        with pytest.raises(RuntimeError, match="already been consumed"):
            search.evaluate_best_trial_tail(
                study,
                output_root=str(tmp_path),
            )

        trial_1 = study.ask()
        study.tell(trial_1, 4.5)
        trial_1_dir = tmp_path / "trial_1"
        trial_1_reports = trial_1_dir / "reports"
        trial_1_reports.mkdir(parents=True)
        changed_tail_records = [dict(tail_records[0], feature=2.0)]
        for direction in ("long", "short"):
            (trial_1_reports / f"optuna_reserved_tail_{direction}.json").write_text(
                json.dumps(changed_tail_records),
                encoding="utf-8",
            )
            (trial_1_dir / f"{direction}.json").write_text(
                json.dumps({
                    "rules_set": [{"conditions": ["[feature] IS High"]}],
                }),
                encoding="utf-8",
            )

        with pytest.raises(RuntimeError, match="sealed"):
            search.evaluate_best_trial_tail(
                study,
                output_root=str(tmp_path),
            )

    assert result["frozen"] is True
    assert result["evaluated_once"] is True
    assert result["best_trial"] == 0
    assert all(
        result["directions"][direction]["available"]
        for direction in ("long", "short")
    )
    assert engine_type.return_value.simulate_rule_set.call_count == 2
    claim = study.user_attrs["optuna_reserved_tail_consumption_v1"]
    assert len(claim["consumptions"]) == 1
    assert claim["consumptions"][0]["study_name"] == study.study_name
    assert claim["consumptions"][0]["best_trial_number"] == 0
    assert claim["consumptions"][0]["tail_hash"] == result["tail_hash"]


def test_main_rejects_a_sealed_study_before_new_trials(monkeypatch) -> None:
    optimize = Mock()
    sealed_study = SimpleNamespace(
        study_name="sealed-study",
        user_attrs={
            search._TAIL_CONSUMPTION_ATTR: {
                "schema_version": "1.0",
                "sealed": True,
                "consumptions": [{
                    "study_name": "sealed-study",
                    "best_trial_number": 0,
                    "tail_hash": "a" * 64,
                }],
            },
        },
        optimize=optimize,
    )
    monkeypatch.setattr(
        search.optuna,
        "create_study",
        lambda **kwargs: sealed_study,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["optuna_search", "--study-name", "sealed-study"],
    )

    with pytest.raises(RuntimeError, match="sealed"):
        search.main()
    optimize.assert_not_called()


def test_validation_metric_collection_has_no_oos_or_test_metrics(tmp_path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    for direction, ret in (("long", 4.0), ("short", 2.0)):
        (reports / f"rb_governor_{direction}_report.json").write_text(
            json.dumps(
                {
                    "valid_metrics": {
                        "total_return_pct": ret,
                        "max_drawdown_pct": 5.0,
                        "profit_factor": 1.3,
                    },
                    "risk_history": [
                        {"risk_tail_holdout_return_pct": ret - 0.5}
                    ],
                    "fail_closed": False,
                }
            )
        )
    metrics = search.collect_validation_metrics({}, str(tmp_path))
    assert metrics["valid_long_return_pct"] == 4.0
    assert metrics["tail_short_return_pct"] == 1.5
    assert not any(key.startswith("test_") for key in metrics)
    assert search.compute_score(metrics) > -1_000_000


def test_fail_closed_score_is_a_hard_penalty() -> None:
    metrics = {
        "fail_closed_long": 1.0,
        "fail_closed_short": 1.0,
    }
    assert search.compute_score(metrics) == -1_000_000.0


def test_tail_holdout_does_not_change_optuna_score() -> None:
    base = {
        "valid_long_return_pct": 2.0,
        "valid_short_return_pct": 1.0,
        "valid_long_dd_pct": 3.0,
        "valid_short_dd_pct": 4.0,
        "valid_long_pf": 1.4,
        "valid_short_pf": 1.2,
        "tail_long_return_pct": -99.0,
        "tail_short_return_pct": 99.0,
    }
    changed = dict(base)
    changed.update({
        "tail_long_return_pct": 99.0,
        "tail_short_return_pct": -99.0,
    })

    assert search.compute_score(base) == search.compute_score(changed)


def test_objective_source_is_validation_and_tail_only() -> None:
    source = inspect.getsource(search.objective)
    assert "test_" not in source
    assert "phase5" not in source.lower()
    assert "rb_validation_only" in source
