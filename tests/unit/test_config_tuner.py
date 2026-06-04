"""
Unit tests for gpu_fuzzy_trader.tuning (config overlay + objective).
"""

from __future__ import annotations

import json
import os

import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.tuning.config_overlay import _TUNABLE_NAMES, apply_trial_params
from gpu_fuzzy_trader.tuning.config_space import (
    LOW_RAM_PROFILE,
    get_profile_params,
)
from gpu_fuzzy_trader.tuning.objective import (
    compute_validation_objective,
    extract_test_metrics,
    extract_validation_metrics,
)
from gpu_fuzzy_trader.tuning._bootstrap import configure_tuning_cpu_env
from gpu_fuzzy_trader.tuning.study_runner import (
    build_merged_trial_config,
    copy_phase1_artifacts,
    export_best_config,
    validate_baseline_prerequisites,
)


class TestConfigOverlay:
    def test_restores_attributes_after_context(self):
        original_support = _cfg.MIN_TRADE_SUPPORT
        with apply_trial_params({"MIN_TRADE_SUPPORT": 77}):
            assert _cfg.MIN_TRADE_SUPPORT == 77
        assert _cfg.MIN_TRADE_SUPPORT == original_support

    def test_trial_seed_patches_global_and_phase_seeds(self):
        with apply_trial_params({}, trial_seed=12345):
            assert _cfg.GLOBAL_SEED == 12345
            assert _cfg.PHASE2_SEED == 12345
            assert _cfg.PHASE4_SEED == 12345

    def test_rejects_unknown_keys(self):
        with pytest.raises(ValueError, match="Unsupported"):
            with apply_trial_params({"NOT_A_REAL_CONFIG_KEY": 1}):
                pass


class TestObjective:
    def _nested_phase5(self) -> dict:
        return {
            "long": {
                "validation": {
                    "total_return_pct": 5.0,
                    "max_drawdown_pct": 8.0,
                },
                "test": {"total_return_pct": 3.0},
            },
            "short": {
                "validation": {
                    "total_return_pct": 2.0,
                    "max_drawdown_pct": 12.0,
                },
                "test": {"total_return_pct": -10.0},
            },
        }

    def test_nested_validation_extraction(self):
        p5 = self._nested_phase5()
        assert extract_validation_metrics(
            p5, "long")["total_return_pct"] == 5.0
        assert extract_test_metrics(p5, "short")["total_return_pct"] == -10.0

    def test_legacy_flat_test_metrics(self):
        flat = {"long": {"total_return_pct": 1.5}}
        assert extract_test_metrics(flat, "long")["total_return_pct"] == 1.5

    def test_compute_validation_objective(self):
        score, details = compute_validation_objective(
            self._nested_phase5(),
            phase2_pool_long=25,
            phase2_pool_short=25,
            drawdown_weight=0.5,
            gate_penalty=20.0,
            val_return_gate_pct=0.0,
        )
        # min val return 2.0 - 0.5 * 12.0 = -4.0 (pools meet min total 40)
        assert score == pytest.approx(-4.0)
        assert details["pool_penalty"] == 0.0
        assert details["test_return_short"] == -10.0

    def test_pool_shortfall_penalty(self):
        p5 = {
            "long": {
                "validation": {"total_return_pct": 10.0, "max_drawdown_pct": 5.0},
                "test": {},
            },
            "short": {
                "validation": {"total_return_pct": 10.0, "max_drawdown_pct": 5.0},
                "test": {},
            },
        }
        score_full, details_full = compute_validation_objective(
            p5,
            phase2_pool_long=20,
            phase2_pool_short=20,
            pool_min_total=40,
            pool_shortfall_penalty=0.5,
            val_return_gate_pct=0.0,
        )
        score_starved, details_starved = compute_validation_objective(
            p5,
            phase2_pool_long=6,
            phase2_pool_short=7,
            pool_min_total=40,
            pool_shortfall_penalty=0.5,
            val_return_gate_pct=0.0,
        )
        assert details_starved["pool_shortfall"] == 27.0
        assert details_starved["pool_penalty"] == pytest.approx(13.5)
        assert score_starved < score_full

    def test_gate_penalty_when_validation_negative(self):
        p5 = {
            "long": {
                "validation": {"total_return_pct": -1.0, "max_drawdown_pct": 5.0},
                "test": {},
            },
            "short": {
                "validation": {"total_return_pct": 4.0, "max_drawdown_pct": 5.0},
                "test": {},
            },
        }
        score, details = compute_validation_objective(
            p5, val_return_gate_pct=0.0, gate_penalty=20.0,
        )
        assert details["gate_penalty"] == 20.0
        assert score < -15.0


class TestConfigSpace:
    def test_low_ram_profile_keys(self):
        profile = get_profile_params("low_ram")
        assert profile["PHASE2_POPULATION_SIZE"] == 100
        assert profile["PHASE2_GENERATIONS"] == 50
        assert "CV_N_FOLDS" not in profile

    def test_low_ram_profile_forces_phase2_cpu(self):
        profile = get_profile_params("low_ram")
        assert profile["PHASE2_USE_GPU"] is False
        assert profile["PHASE3_USE_GPU"] is False

    def test_low_ram_profile_disables_phase2_early_stop(self):
        profile = get_profile_params("low_ram")
        assert profile["PHASE2_EARLY_STOP_ENABLED"] is False

    def test_tunable_names_exist_on_config_module(self):
        for name in _TUNABLE_NAMES:
            assert hasattr(_cfg, name), name


class TestTuningBootstrap:
    def test_configure_tuning_cpu_env_sets_jax_platforms(self, monkeypatch):
        monkeypatch.delenv("JAX_PLATFORMS", raising=False)
        configure_tuning_cpu_env(force=True)
        assert os.environ["JAX_PLATFORMS"] == "cpu"

    def test_bootstrap_survives_run_pipeline_import(self):
        import subprocess
        import sys

        repo = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
        code = (
            "import os\n"
            "import runpy\n"
            "runpy.run_module('gpu_fuzzy_trader.tuning', run_name='__main__', "
            "alter_sys=True)\n"
        )
        # Only check env after bootstrap: run configure via -c before importing run_pipeline
        probe = (
            "from gpu_fuzzy_trader.tuning._bootstrap import configure_tuning_cpu_env\n"
            "configure_tuning_cpu_env(force=True)\n"
            "import os\n"
            "assert os.environ['JAX_PLATFORMS'] == 'cpu'\n"
            "from gpu_fuzzy_trader.run_pipeline import Pipeline_Orchestrator  # noqa: F401\n"
            "assert os.environ['JAX_PLATFORMS'] == 'cpu'\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=repo,
        )
        assert result.returncode == 0, result.stderr or result.stdout


class TestMergedConfig:
    def test_build_merged_trial_config(self):
        merged = build_merged_trial_config(
            {"MIN_TRADE_SUPPORT": 150}, "low_ram")
        assert merged["MIN_TRADE_SUPPORT"] == 150
        assert merged["PHASE2_USE_GPU"] is False
        assert "CV_N_FOLDS" not in merged or merged.get("CV_N_FOLDS") in (2, 3)


class TestStudyRunnerHelpers:
    def test_validate_baseline_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="selected_features"):
            validate_baseline_prerequisites(str(tmp_path))

    def test_copy_phase1_artifacts(self, tmp_path):
        baseline = tmp_path / "baseline"
        trial = tmp_path / "trial"
        baseline.mkdir()
        for name in ("selected_features_long.json", "selected_features_short.json"):
            (baseline / name).write_text("[]", encoding="utf-8")

        copy_phase1_artifacts(str(baseline), str(trial))
        assert (trial / "selected_features_long.json").is_file()

    def test_export_best_config(self, tmp_path):
        import optuna

        study = optuna.create_study(direction="maximize")
        study.enqueue_trial({"x": 1})
        study.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=1)

        out = tmp_path / "best_config.json"
        export_best_config(study, str(out), profile="low_ram")
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "params" in data
        assert "merged_config" in data
        assert data["merged_config"]["PHASE2_USE_GPU"] is False
        assert data["trial_number"] == 0
