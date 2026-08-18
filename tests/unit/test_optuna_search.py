"""Validation-only Optuna contract tests."""

from __future__ import annotations

import inspect
import json

import optuna

from gpu_fuzzy_trader import config as cfg
from gpu_fuzzy_trader import optuna_search as search


def test_search_space_contains_only_live_active_config_keys() -> None:
    assert search.SEARCH_SPACE
    assert "PHASE2_MIN_PROFITABLE_SYMBOLS" not in search.SEARCH_SPACE
    for key in search.SEARCH_SPACE:
        assert hasattr(cfg, key), key
        assert not key.startswith(("PHASE3_", "PHASE4_"))
        assert not key.startswith("test_")


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


def test_objective_source_is_validation_and_tail_only() -> None:
    source = inspect.getsource(search.objective)
    assert "test_" not in source
    assert "phase5" not in source.lower()
    assert "rb_validation_and_tail" in source
