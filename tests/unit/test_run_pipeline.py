"""RB-only pipeline dispatch and fail-closed orchestration tests."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from gpu_fuzzy_trader import run_pipeline as pipeline
from gpu_fuzzy_trader.run_pipeline import (
    Pipeline_Orchestrator,
    _context_coverage_preflight,
    _context_coverage_report,
    _log_phase_entry,
    _validate_enriched_context_contract,
)


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


def _context_frame(rows: int = 12) -> pd.DataFrame:
    """Return a small frame with both direction context gates."""
    frame = _frame(rows)
    for column in (
        "tf_permission_long",
        "tf_permission_short",
        "lwc_pullback_reversal_long",
        "lwc_pullback_reversal_short",
    ):
        frame[column] = np.zeros(rows, dtype=np.int8)
    frame.loc[[0, 1], [
        "tf_permission_long",
        "lwc_pullback_reversal_long",
    ]] = 1
    frame.loc[[2, 3], [
        "tf_permission_short",
        "lwc_pullback_reversal_short",
    ]] = 1
    return frame


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


def test_context_coverage_report_includes_split_and_symbol_counts() -> None:
    frame = _context_frame()

    report = _context_coverage_report(
        frame,
        frame.iloc[:6].copy(),
        frame.iloc[6:].copy(),
    )

    assert set(report) == {
        "train", "validation_fitness", "validation_selection",
    }
    assert report["train"]["long"]["eligible_rows"] == 2
    assert report["train"]["long"]["by_symbol"] == {"A": 1, "B": 1}
    assert report["train"]["short"]["eligible_rows"] == 2
    assert report["validation_selection"]["long"]["eligible_rows"] == 0


def test_context_coverage_preflight_logs_all_splits_and_directions(
    caplog: pytest.LogCaptureFixture,
) -> None:
    frame = _context_frame()

    with caplog.at_level(logging.INFO, logger="gpu_fuzzy_trader.run_pipeline"):
        report = _context_coverage_preflight(
            frame,
            frame,
            frame,
            floor_aware=False,
        )

    assert report["validation_fitness"]["short"]["eligible_rows"] == 2
    for split_name in ("train", "validation_fitness", "validation_selection"):
        for direction in ("long", "short"):
            assert f"Context coverage [{split_name}/{direction}]" in caplog.text
    assert "per_symbol={'A': 1, 'B': 1}" in caplog.text


def test_context_coverage_preflight_stops_zero_direction_before_phase2(
    tmp_path,
) -> None:
    frame = _context_frame()
    frame.loc[:, [
        "tf_permission_short",
        "lwc_pullback_reversal_short",
    ]] = 0
    orch = Pipeline_Orchestrator(output_dir=str(tmp_path))
    orch._load_and_split_data = MagicMock(return_value=(frame, frame))
    orch._validate_active_configuration = MagicMock()
    orch._validation_scoring_frames = MagicMock(return_value=(frame, frame))
    orch._run_phase1 = MagicMock()
    orch._run_phase2 = MagicMock()

    with pytest.raises(RuntimeError, match="validation_fitness/short"):
        orch.run(force=True)

    orch._run_phase1.assert_not_called()
    orch._run_phase2.assert_not_called()


def test_context_coverage_preflight_rejects_nonempty_raw_splits() -> None:
    frame = _frame()

    with pytest.raises(RuntimeError, match="no context columns"):
        _context_coverage_preflight(frame, frame, frame)


def test_context_coverage_reports_permission_trigger_breakdown() -> None:
    frame = _context_frame(rows=8)
    frame.loc[4, "tf_permission_long"] = 1
    frame.loc[5, "lwc_pullback_reversal_long"] = 1

    stats = pipeline._context_coverage_for_direction(frame, "long")

    assert stats["permission_rows"] == 3
    assert stats["trigger_rows"] == 3
    assert stats["eligible_rows"] == 2
    assert stats["permission_only_rows"] == 1
    assert stats["trigger_only_rows"] == 1
    assert stats["by_symbol_detail"]["A"]["eligible_rows"] == 1


def test_floor_aware_preflight_rejects_sparse_island_support() -> None:
    frame = _context_frame(rows=20)
    for column in (
        "tf_permission_long",
        "tf_permission_short",
        "lwc_pullback_reversal_long",
        "lwc_pullback_reversal_short",
    ):
        frame[column] = 1

    with pytest.raises(RuntimeError, match="min_trade_support"):
        _context_coverage_preflight(
            frame,
            frame,
            frame,
            floor_aware=True,
            run_id="test-run",
        )


def test_preflight_skips_starved_island_when_sibling_passes(monkeypatch) -> None:
    frame = _context_frame(rows=200)
    for column in (
        "tf_permission_long",
        "tf_permission_short",
        "lwc_pullback_reversal_long",
        "lwc_pullback_reversal_short",
    ):
        frame[column] = 1

    monkeypatch.setattr(
        pipeline._cfg, "phase2_island_mode_enabled", lambda: True,
    )
    monkeypatch.setattr(
        pipeline._cfg, "PHASE2_SKIP_CONTEXT_STARVED_ISLANDS", True,
    )
    monkeypatch.setattr(
        pipeline,
        "_context_island_sample_report",
        lambda *args, **kwargs: {
            "mode": "singleton_proxy",
            "reference_rows": 200,
            "sampling_total": 200,
            "islands": {},
            "failures": [
                "long/1/train_sample: eligible_rows=1<min_trade_support=30",
                "short/1/validation_fitness_sample: "
                "eligible_rows=1<validation_trade_floor=8",
            ],
            "passed_islands_by_direction": {
                "long": [{"island_id": "0", "symbols": ["A"]}],
                "short": [{"island_id": "0", "symbols": ["A"]}],
            },
            "failed_islands_by_direction": {
                "long": [{
                    "island_id": "1",
                    "symbols": ["B"],
                    "failures": [
                        "long/1/train_sample: "
                        "eligible_rows=1<min_trade_support=30",
                    ],
                }],
                "short": [{
                    "island_id": "1",
                    "symbols": ["B"],
                    "failures": [
                        "short/1/validation_fitness_sample: "
                        "eligible_rows=1<validation_trade_floor=8",
                    ],
                }],
            },
        },
    )

    report = _context_coverage_preflight(
        frame, frame, frame, floor_aware=True, run_id="test-run",
    )

    assert report["blocked_directions"] == []
    assert report["skipped_context_starved_islands"]
    assert any("symbols=['B']" in item or "symbols=[\"B\"]" in item
               or "B" in item
               for item in report["skipped_context_starved_islands"])


def test_context_preflight_blocks_only_unsupported_direction(monkeypatch) -> None:
    frame = _context_frame(rows=200)
    frame.loc[:, [
        "tf_permission_long",
        "lwc_pullback_reversal_long",
    ]] = 0
    frame.loc[:, [
        "tf_permission_short",
        "lwc_pullback_reversal_short",
    ]] = 1
    monkeypatch.setattr(
        pipeline._cfg,
        "phase2_island_mode_enabled",
        lambda: False,
    )

    report = _context_coverage_preflight(frame, frame, frame)

    assert report["blocked_directions"] == ["long"]
    assert report["direction_failures"]["long"]
    assert report["direction_failures"]["short"] == []


def test_phase2_skips_context_blocked_direction(tmp_path) -> None:
    frame = _context_frame(rows=200)
    orchestrator = Pipeline_Orchestrator(output_dir=str(tmp_path))

    pools = orchestrator._run_phase2(
        frame,
        {"long": [{"phase2_rule_id": "blocked"}], "short": []},
        force=True,
        val_df=frame,
        blocked_directions=frozenset({"long"}),
    )

    assert pools == {"long": [], "short": []}
    assert orchestrator._phase2_status["long"]["reason"] == (
        "context_support_preflight"
    )


def test_phase2_regenerates_legacy_pool_before_generator_can_merge_it(
    tmp_path, monkeypatch,
) -> None:
    """A schema-valid pool without an input identity is never a resume hit."""
    stale_pool_path = tmp_path / "phase2_long_pool.json"
    stale_history_path = tmp_path / "phase2_long_history.json"
    stale_pool_path.write_text(
        json.dumps([{
            "chromosome": [2],
            "conditions": [],
            "objectives": {
                "sortino_ratio": 1.0,
                "max_drawdown_pct": 1.0,
                "win_rate": 50.0,
            },
            "executed_trades": 1,
        }]),
        encoding="utf-8",
    )
    stale_history_path.write_text("[]", encoding="utf-8")
    frame = _context_frame(rows=200)
    actual_generator = pipeline._phase2_module.Rule_Pool_Generator
    seen = {"run": False}

    class FakeGenerator:
        def __init__(self, **_kwargs) -> None:
            pass

        def run(self) -> list[dict]:
            seen["run"] = True
            assert not stale_pool_path.exists()
            assert not stale_history_path.exists()
            return []

        skip_if_valid = staticmethod(actual_generator.skip_if_valid)
        discard_cached_pool = staticmethod(actual_generator.discard_cached_pool)
        write_pool_resume_identity = staticmethod(
            actual_generator.write_pool_resume_identity
        )

    monkeypatch.setattr(
        pipeline._cfg, "phase2_island_mode_enabled", lambda: False,
    )
    monkeypatch.setattr(pipeline, "Rule_Pool_Generator", FakeGenerator)
    orch = Pipeline_Orchestrator(output_dir=str(tmp_path))

    with pipeline._temporary_output_paths(str(tmp_path)):
        pools = orch._run_phase2(
            frame,
            {"long": [{"name": "feature", "mode": "positive"}], "short": []},
            force=False,
            val_df=frame,
        )

    assert seen["run"]
    assert pools == {"long": [], "short": []}
    assert not stale_pool_path.exists()
    assert not stale_history_path.exists()


def test_full_run_cleanup_removes_phase2_resume_sidecars(tmp_path) -> None:
    orch = Pipeline_Orchestrator(output_dir=str(tmp_path))
    sidecars = (
        tmp_path / "phase2_long_pool.json.identity.json",
        tmp_path / "phase2_short_pool.json.identity.json",
    )
    with pipeline._temporary_output_paths(str(tmp_path)):
        tmp_path.mkdir(parents=True, exist_ok=True)
        for path in sidecars:
            path.write_text("stale", encoding="utf-8")
        orch._begin_run("full", clear_derived=False, clear_phase2=True)

    assert all(not path.exists() for path in sidecars)


def test_standalone_phase1_loader_rejects_unproven_artifacts(tmp_path) -> None:
    """A disk artifact is not a Phase 2 prerequisite without current inputs."""
    stale = _frame()
    stale.loc[0, "feature"] += 1.0
    orch = Pipeline_Orchestrator(output_dir=str(tmp_path))

    with pipeline._temporary_output_paths(str(tmp_path)):
        for direction, path in pipeline._selector_module._DIRECTION_PATHS.items():
            artifact_path = Path(path)
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text(
                json.dumps({
                    "direction": direction,
                    "features": [{
                        "name": "feature",
                        "mode": "positive",
                        "score": 0.5,
                    }],
                    "phase1_disabled": bool(pipeline._cfg.PHASE1_DISABLED),
                    "phase1_input_identity": (
                        pipeline._selector_module._phase1_input_identity(stale)
                    ),
                }),
                encoding="utf-8",
            )

        with pytest.raises(FileNotFoundError, match="matching current Phase 1 input"):
            orch._load_phase1_outputs()
        with pytest.raises(FileNotFoundError, match="matching current Phase 1 input"):
            orch._load_phase1_outputs(_frame())

        current_identity = pipeline._selector_module._phase1_input_identity(
            _frame())
        for direction, path in pipeline._selector_module._DIRECTION_PATHS.items():
            Path(path).write_text(
                json.dumps({
                    "direction": direction,
                    "features": [{
                        "name": "feature",
                        "mode": "positive",
                        "score": 0.5,
                    }],
                    "phase1_disabled": bool(pipeline._cfg.PHASE1_DISABLED),
                    "phase1_input_identity": current_identity,
                }),
                encoding="utf-8",
            )

        assert orch._load_phase1_outputs(_frame()) == {
            "long": [{"name": "feature", "mode": "positive", "score": 0.5}],
            "short": [{"name": "feature", "mode": "positive", "score": 0.5}],
        }


def test_standalone_phase2_loader_rejects_unproven_artifacts(tmp_path) -> None:
    """A schema-valid pool is not an RB prerequisite without its identity."""
    pool = [{
        "chromosome": [2],
        "conditions": [],
        "objectives": {
            "sortino_ratio": 1.0,
            "max_drawdown_pct": 1.0,
            "win_rate": 50.0,
        },
        "executed_trades": 1,
    }]
    orch = Pipeline_Orchestrator(output_dir=str(tmp_path))

    with pipeline._temporary_output_paths(str(tmp_path)):
        for path in pipeline._phase2_module._POOL_PATHS.values():
            artifact_path = Path(path)
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text(json.dumps(pool), encoding="utf-8")

        pools = orch._load_phase2_outputs()
        assert pools == {"long": [], "short": []}
        assert {
            direction: status["reason"]
            for direction, status in orch._phase2_status.items()
        } == {
            "long": "phase2_identity_unavailable",
            "short": "phase2_identity_unavailable",
        }

        pools = orch._load_phase2_outputs(
            _frame(), _frame(), {"long": [], "short": []},
        )
        assert pools == {"long": [], "short": []}
        assert {
            direction: status["reason"]
            for direction, status in orch._phase2_status.items()
        } == {
            "long": "missing_phase2_output",
            "short": "missing_phase2_output",
        }

        current = _frame()
        phase1_result = {"long": [], "short": []}
        for direction in ("long", "short"):
            identity = pipeline._phase2_resume_identity(
                current,
                current,
                phase1_result[direction],
                direction,
                orch._cv_folds,
            )
            pipeline.Rule_Pool_Generator.write_pool_resume_identity(
                direction, identity,
            )
        pools = orch._load_phase2_outputs(current, current, phase1_result)

    assert pools == {"long": pool, "short": pool}
    assert {
        direction: status["reason"]
        for direction, status in orch._phase2_status.items()
    } == {
        "long": "loaded_pool",
        "short": "loaded_pool",
    }


def test_stale_enriched_context_contract_is_rejected(tmp_path, monkeypatch) -> None:
    enriched_path = tmp_path / "train_new_hwc_mwc_lwc.csv"
    enriched_path.write_text("datetime,symbol\n", encoding="utf-8")
    manifest_path = tmp_path / "trend_context_manifest.json"
    manifest_path.write_text(
        json.dumps({
            "context_algorithm_version": "regime_v3_next_open_alignment",
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(pipeline._cfg, "TRAIN_CSV_PATH", str(enriched_path))
    monkeypatch.setattr(
        pipeline._cfg,
        "ENRICHED_MANIFEST_PATH",
        str(manifest_path),
    )

    with pytest.raises(RuntimeError, match="use context contract"):
        _validate_enriched_context_contract()


def test_enriched_context_contract_rejects_tape_hash_mismatch(
    tmp_path, monkeypatch,
) -> None:
    train_path = tmp_path / "train_new_hwc_mwc_lwc.csv"
    test_path = tmp_path / "test_new_hwc_mwc_lwc.csv"
    train_path.write_text("train-current", encoding="utf-8")
    test_path.write_text("test-current", encoding="utf-8")
    manifest_path = tmp_path / "trend_context_manifest.json"
    manifest_path.write_text(
        json.dumps({
            "context_algorithm_version": pipeline._cfg.CONTEXT_ALGORITHM_VERSION,
            "tapes": {
                "train": {"sha256": "stale-train-hash"},
                "test": {"sha256": "stale-test-hash"},
            },
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(pipeline._cfg, "TRAIN_CSV_PATH", str(train_path))
    monkeypatch.setattr(pipeline._cfg, "TEST_CSV_PATH", str(test_path))
    monkeypatch.setattr(
        pipeline._cfg,
        "ENRICHED_MANIFEST_PATH",
        str(manifest_path),
    )

    with pytest.raises(RuntimeError, match="does not match the hash"):
        _validate_enriched_context_contract()


def test_current_run_oos_directions_follow_accepted_phase2_pool(tmp_path) -> None:
    frame = _context_frame(rows=200)
    for column in (
        "tf_permission_long",
        "tf_permission_short",
        "lwc_pullback_reversal_long",
        "lwc_pullback_reversal_short",
    ):
        frame[column] = 1

    orch = Pipeline_Orchestrator(output_dir=str(tmp_path))
    orch._load_and_split_data = MagicMock(return_value=(frame, frame))
    orch._validate_active_configuration = MagicMock()
    orch._validation_scoring_frames = MagicMock(return_value=(frame, frame))
    orch._run_phase1 = MagicMock(return_value={
        "long": [{"name": "feature", "mode": "positive"}],
        "short": [{"name": "feature", "mode": "positive"}],
    })
    orch._run_phase2 = MagicMock(return_value={
        "long": [{"phase2_rule_id": "long-rule"}],
        "short": [],
    })
    orch._run_rb_governor = MagicMock(return_value={
        "long": {
            "rules_set": [{"conditions": []}],
            "deployment_accepted": True,
        },
        "short": {
            "rules_set": [],
            "deployment_accepted": False,
        },
    })
    orch._run_nested_validation = MagicMock(return_value={})
    orch._release_between_phases = MagicMock()
    orch._run_phase5 = MagicMock(return_value={})

    result = orch.run(force=True)

    assert result["run_id"]
    assert orch._run_phase5.call_args.kwargs["allowed_directions"] == frozenset(
        {"long"}
    )
    manifest = json.loads(
        (tmp_path / "reports" / "run_manifest.json").read_text()
    )
    assert manifest["run_id"] == result["run_id"]
    assert manifest["status"] == "completed"


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
    orch._load_phase1_outputs = MagicMock(return_value={"long": [], "short": []})
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
    frame = _frame()
    orch._phase2_status = {
        "short": {
            "status": "error",
            "reason": "missing_phase2_output",
            "pool_size": 0,
        }
    }
    with patch.object(
        pipeline.Rule_Pool_Generator, "skip_if_valid", return_value=None,
    ):
        pools = orch._load_phase2_outputs(
            frame, frame, {"long": [], "short": []},
        )

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
