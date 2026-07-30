"""RB-only pipeline dispatch and fail-closed orchestration tests."""

from __future__ import annotations

import inspect
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from gpu_fuzzy_trader import run_pipeline as pipeline
from gpu_fuzzy_trader.run_pipeline import Pipeline_Orchestrator, _log_phase_entry


def _frame(rows: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    entry = np.full(rows, 100.0)
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=rows, freq="5min"),
            "symbol": np.where(np.arange(rows) % 2, "A", "B"),
            "label_open_next": entry,
            "label_close_288": entry * 1.01,
            "label_min_288": entry * 0.99,
            "label_max_288": entry * 1.02,
            "label_max_before_min": np.zeros(rows),
            "feature": rng.normal(size=rows),
        }
    )


def _strategies() -> dict[str, dict]:
    return {
        direction: {
            "direction": direction,
            "rules_set": [],
            "deployment_accepted": False,
            "fail_closed": True,
            "reason": "empty_phase2_pool",
        }
        for direction in ("long", "short")
    }


def test_phase_log_entry_is_structured_json(tmp_path) -> None:
    path = tmp_path / "pipeline.log"
    _log_phase_entry(
        str(path), "RB Governor", "start", "end", 1.25, False,
        {"long": {"status": "fail_closed"}},
    )
    row = json.loads(path.read_text().strip())
    assert row["phase"] == "RB Governor"
    assert row["result_summary"]["long"]["status"] == "fail_closed"


def test_phase3_and_phase4_are_rb_compatibility_aliases(tmp_path) -> None:
    orch = Pipeline_Orchestrator(output_dir=str(tmp_path))
    frame = _frame()
    rb_result = _strategies()
    orch._load_and_split_data = MagicMock(return_value=(frame, frame))
    orch._validate_active_configuration = MagicMock()
    orch._validation_scoring_frames = MagicMock(return_value=(frame, frame))
    orch._load_phase2_outputs = MagicMock(return_value={"long": [], "short": []})
    orch._release_between_phases = MagicMock()
    orch._run_rb_governor = MagicMock(return_value=rb_result)

    phase3 = orch.run_phase(3)
    phase4 = orch.run_phase(4)

    assert phase3["rb_governor"] is rb_result
    assert phase3["phase3"] is rb_result
    assert phase3["phase4"] is rb_result
    assert phase4["rb_governor"] is rb_result
    assert phase4["phase3"] is rb_result
    assert phase4["phase4"] is rb_result
    assert orch._run_rb_governor.call_count == 2


def test_missing_phase2_outputs_are_recorded_for_rb_failure_reason(tmp_path) -> None:
    orch = Pipeline_Orchestrator(output_dir=str(tmp_path))
    orch._phase2_status = {
        "short": {
            "status": "error",
            "reason": "missing_phase2_output",
            "pool_size": 0,
        }
    }
    with patch.object(pipeline._phase2_module.Rule_Pool_Generator, "load_pool", return_value=None):
        pools = orch._load_phase2_outputs()

    assert pools["long"] == []
    assert pools["short"] == []
    assert orch._phase2_status["short"]["reason"] == "missing_phase2_output"


def test_rb_runner_passes_non_ok_phase2_status_to_canonical_governor(tmp_path) -> None:
    orch = Pipeline_Orchestrator(output_dir=str(tmp_path))
    frame = _frame()
    orch._phase2_status = {
        "long": {"status": "ok", "reason": "generated", "pool_size": 1},
        "short": {"status": "error", "reason": "phase2_error", "pool_size": 0},
    }
    with pipeline._temporary_output_paths(str(tmp_path)), patch.object(
        pipeline._cfg, "write_config_audit_report"
    ), patch.object(
        pipeline._rb_governor_module,
        "run_rb_governor_pipeline",
        return_value=_strategies(),
    ) as runner:
        result = orch._run_rb_governor(frame, frame, {"long": [], "short": []})

    assert set(result) == {"long", "short"}
    reasons = runner.call_args.kwargs["failure_reasons"]
    assert reasons == {"short": "phase2_error"}


def test_rb_runner_handles_governor_exception_by_writing_both_directions(tmp_path) -> None:
    orch = Pipeline_Orchestrator(output_dir=str(tmp_path))
    frame = _frame()
    with pipeline._temporary_output_paths(str(tmp_path)), patch.object(
        pipeline._cfg, "write_config_audit_report"
    ), patch.object(
        pipeline._rb_governor_module,
        "run_rb_governor_pipeline",
        side_effect=RuntimeError("synthetic RB failure"),
    ):
        result = orch._run_rb_governor(frame, frame, {"long": [], "short": []})

    assert set(result) == {"long", "short"}
    assert all(strategy["rules_set"] == [] for strategy in result.values())
    assert all(strategy["reason"] == "rb_governor_error" for strategy in result.values())


def test_temporary_output_paths_rebind_and_restore(tmp_path) -> None:
    old_output = pipeline._cfg.OUTPUTS_DIR
    old_reports = pipeline._cfg.REPORTS_DIR
    with pipeline._temporary_output_paths(str(tmp_path / "run")):
        assert pipeline._cfg.OUTPUTS_DIR == str(tmp_path / "run")
        assert pipeline._cfg.REPORTS_DIR == str(tmp_path / "run" / "reports")
    assert pipeline._cfg.OUTPUTS_DIR == old_output
    assert pipeline._cfg.REPORTS_DIR == old_reports


def test_pipeline_source_has_no_legacy_selection_or_risk_classes() -> None:
    source = inspect.getsource(Pipeline_Orchestrator)
    assert "Rule_Set_Selector" not in source
    assert "WalkForwardRiskOptimizer" not in source
