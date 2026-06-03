"""
Unit tests for gpu_fuzzy_trader.tuning (config overlay + objective).
"""

from __future__ import annotations

import json
import os

import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.tuning.config_overlay import apply_trial_params
from gpu_fuzzy_trader.tuning.config_space import (
    LOW_RAM_PROFILE,
    get_profile_params,
)
from gpu_fuzzy_trader.tuning.objective import (
    compute_validation_objective,
    extract_test_metrics,
    extract_validation_metrics,
)
from gpu_fuzzy_trader.tuning.study_runner import (
    copy_phase1_artifacts,
    export_best_config,
    validate_baseline_prerequisites,
)


class TestConfigOverlay:
    def test_restores_attributes_after_context(self):
        original_pop = _cfg.PHASE2_POPULATION_SIZE
        with apply_trial_params({"PHASE2_POPULATION_SIZE": 77}):
            assert _cfg.PHASE2_POPULATION_SIZE == 77
        assert _cfg.PHASE2_POPULATION_SIZE == original_pop

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
            drawdown_weight=0.5,
            gate_penalty=20.0,
            val_return_gate_pct=0.0,
        )
        # min val return 2.0 - 0.5 * 12.0 = -4.0
        assert score == pytest.approx(-4.0)
        assert details["test_return_short"] == -10.0

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
        assert profile["PHASE2_POPULATION_SIZE"] == LOW_RAM_PROFILE[
            "PHASE2_POPULATION_SIZE"
        ]
        assert profile["CV_N_FOLDS"] == 2


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
        export_best_config(study, str(out))
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "params" in data
        assert data["trial_number"] == 0
