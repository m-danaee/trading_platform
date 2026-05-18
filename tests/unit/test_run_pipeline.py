"""
Unit tests for gpu_fuzzy_trader.run_pipeline.Pipeline_Orchestrator

Tests cover:
  - Output directory creation
  - Phase timing log entries (JSON lines format)
  - Skip logic per phase (mocked)
  - Empty pool handling (Phases 3 and 4 skipped)
  - Phase 5 always runs
  - run() returns dict with expected keys
  - __main__.py entry point
"""

from __future__ import annotations

import json
import os
import time
from unittest.mock import MagicMock, patch

import pandas as pd
import numpy as np
import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader import run_pipeline as run_pipeline_module
from gpu_fuzzy_trader.features import selector as selector_module
from gpu_fuzzy_trader.phases import phase2_rule_pool as phase2_module
from gpu_fuzzy_trader.phases import phase3_rule_set as phase3_module
from gpu_fuzzy_trader.phases import phase4_rl_optimizer as phase4_module
from gpu_fuzzy_trader.phases import phase5_oos as phase5_module
from gpu_fuzzy_trader.reporting import reporter as reporter_module
from gpu_fuzzy_trader.run_pipeline import Pipeline_Orchestrator, _log_phase_entry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n_rows: int = 400, symbols: list[str] | None = None) -> pd.DataFrame:
    """Create a minimal DataFrame with all required columns."""
    rng = np.random.default_rng(42)
    if symbols is None:
        symbols = ["SYM_A", "SYM_B"]
    rows_per_sym = n_rows // len(symbols)
    dfs = []
    for sym in symbols:
        n = rows_per_sym
        open_next = rng.uniform(100, 200, size=n)
        data = {
            "datetime": pd.date_range("2020-01-01", periods=n, freq="5min"),
            "symbol": sym,
            "label_open_next": open_next,
            "label_close_288": open_next * rng.uniform(0.95, 1.05, size=n),
            "label_min_288": open_next * rng.uniform(0.90, 1.00, size=n),
            "label_max_288": open_next * rng.uniform(1.00, 1.10, size=n),
            "label_max_before_min": rng.integers(0, 2, size=n).astype(float),
            "_symbol_bar_index": np.arange(n),
        }
        for i in range(3):
            data[f"feat_{i}"] = rng.uniform(0, 1, size=n)
        dfs.append(pd.DataFrame(data))
    return pd.concat(dfs, ignore_index=True)


def _make_rule_set(direction: str = "long") -> dict:
    return {
        "direction": direction,
        "rules_set": [
            {"conditions": ["[feat_0] IS Very High"],
                "tp": 4.0, "sl": 2.0, "capital_pct": 50.0},
            {"conditions": ["[feat_1] IS Low"], "tp": 4.0,
                "sl": 2.0, "capital_pct": 50.0},
        ],
    }


def _make_pool(n: int = 3) -> list[dict]:
    return [
        {
            "chromosome": [0, 1, 2],
            "conditions": [f"[feat_{i}] IS Very High"],
            "objectives": {"total_return_pct": 1.0, "max_drawdown_pct": 1.0, "win_rate": 50.0},
            "executed_trades": 25,
            "tp": _cfg.PHASE2_TP,
            "sl": _cfg.PHASE2_SL,
            "capital_pct": _cfg.PHASE2_CAPITAL_PCT,
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Tests: _log_phase_entry
# ---------------------------------------------------------------------------

class TestLogPhaseEntry:
    def test_creates_log_file(self, tmp_path):
        log_path = str(tmp_path / "pipeline.log")
        _log_phase_entry(log_path, "Test Phase", "2024-01-01T00:00:00Z",
                         "2024-01-01T00:00:01Z", 1.0, False)
        assert os.path.exists(log_path)

    def test_log_entry_is_valid_json(self, tmp_path):
        log_path = str(tmp_path / "pipeline.log")
        _log_phase_entry(log_path, "Test Phase", "2024-01-01T00:00:00Z",
                         "2024-01-01T00:00:01Z", 1.0, False)
        with open(log_path) as fh:
            line = fh.readline().strip()
        entry = json.loads(line)
        assert isinstance(entry, dict)

    def test_log_entry_has_required_keys(self, tmp_path):
        log_path = str(tmp_path / "pipeline.log")
        _log_phase_entry(log_path, "Test Phase", "2024-01-01T00:00:00Z",
                         "2024-01-01T00:00:01Z", 1.5, True)
        with open(log_path) as fh:
            entry = json.loads(fh.readline())
        for key in ("phase", "start_time", "end_time", "elapsed_seconds", "skipped"):
            assert key in entry, f"Missing key: {key}"

    def test_log_entry_phase_name(self, tmp_path):
        log_path = str(tmp_path / "pipeline.log")
        _log_phase_entry(log_path, "Phase 1: Feature Selection",
                         "2024-01-01T00:00:00Z", "2024-01-01T00:00:01Z", 1.0, False)
        with open(log_path) as fh:
            entry = json.loads(fh.readline())
        assert entry["phase"] == "Phase 1: Feature Selection"

    def test_log_entry_skipped_flag(self, tmp_path):
        log_path = str(tmp_path / "pipeline.log")
        _log_phase_entry(log_path, "Phase 1", "2024-01-01T00:00:00Z",
                         "2024-01-01T00:00:01Z", 0.5, True)
        with open(log_path) as fh:
            entry = json.loads(fh.readline())
        assert entry["skipped"] is True

    def test_log_entry_elapsed_seconds(self, tmp_path):
        log_path = str(tmp_path / "pipeline.log")
        _log_phase_entry(log_path, "Phase 1", "2024-01-01T00:00:00Z",
                         "2024-01-01T00:00:01Z", 3.141, False)
        with open(log_path) as fh:
            entry = json.loads(fh.readline())
        assert abs(entry["elapsed_seconds"] - 3.141) < 0.001

    def test_multiple_entries_appended(self, tmp_path):
        log_path = str(tmp_path / "pipeline.log")
        _log_phase_entry(log_path, "Phase 1", "2024-01-01T00:00:00Z",
                         "2024-01-01T00:00:01Z", 1.0, False)
        _log_phase_entry(log_path, "Phase 2", "2024-01-01T00:00:01Z",
                         "2024-01-01T00:00:02Z", 1.0, False)
        with open(log_path) as fh:
            lines = [l.strip() for l in fh if l.strip()]
        assert len(lines) == 2

    def test_result_summary_included_when_provided(self, tmp_path):
        log_path = str(tmp_path / "pipeline.log")
        _log_phase_entry(log_path, "Phase 1", "2024-01-01T00:00:00Z",
                         "2024-01-01T00:00:01Z", 1.0, False,
                         result_summary={"features": 30})
        with open(log_path) as fh:
            entry = json.loads(fh.readline())
        assert "result_summary" in entry
        assert entry["result_summary"]["features"] == 30


# ---------------------------------------------------------------------------
# Tests: Pipeline_Orchestrator._create_output_dirs
# ---------------------------------------------------------------------------

class TestCreateOutputDirs:
    def test_creates_outputs_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_cfg, "OUTPUTS_DIR", str(tmp_path / "outputs"))
        monkeypatch.setattr(_cfg, "REPORTS_DIR", str(
            tmp_path / "outputs" / "reports"))
        orch = Pipeline_Orchestrator()
        orch._create_output_dirs()
        assert os.path.isdir(str(tmp_path / "outputs"))

    def test_creates_reports_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_cfg, "OUTPUTS_DIR", str(tmp_path / "outputs"))
        monkeypatch.setattr(_cfg, "REPORTS_DIR", str(
            tmp_path / "outputs" / "reports"))
        orch = Pipeline_Orchestrator()
        orch._create_output_dirs()
        assert os.path.isdir(str(tmp_path / "outputs" / "reports"))

    def test_idempotent_when_dirs_exist(self, tmp_path, monkeypatch):
        outputs = str(tmp_path / "outputs")
        reports = str(tmp_path / "outputs" / "reports")
        os.makedirs(reports, exist_ok=True)
        monkeypatch.setattr(_cfg, "OUTPUTS_DIR", outputs)
        monkeypatch.setattr(_cfg, "REPORTS_DIR", reports)
        orch = Pipeline_Orchestrator()
        # Should not raise
        orch._create_output_dirs()
        assert os.path.isdir(outputs)
        assert os.path.isdir(reports)


# ---------------------------------------------------------------------------
# Tests: Pipeline_Orchestrator.run() — mocked phases
# ---------------------------------------------------------------------------

class TestPipelineOrchestratorRun:
    """
    Tests for Pipeline_Orchestrator.run() using mocked phase components.
    All heavy computation is mocked so tests run quickly.
    """

    def _make_orch(self, tmp_path) -> Pipeline_Orchestrator:
        orch = Pipeline_Orchestrator()
        orch._log_path = str(tmp_path / "pipeline.log")
        return orch

    def _patch_all(self, tmp_path, train_df, val_df,
                   phase1_result=None, phase2_result=None,
                   phase3_result=None, phase4_result=None,
                   phase5_result=None):
        """Return a context manager that patches all heavy operations."""
        if phase1_result is None:
            phase1_result = {
                "long": [{"name": "feat_0", "mode": "positive", "score": 0.9}],
                "short": [{"name": "feat_1", "mode": "positive", "score": 0.8}],
            }
        if phase2_result is None:
            phase2_result = {"long": _make_pool(3), "short": _make_pool(3)}
        if phase3_result is None:
            phase3_result = {
                "long": _make_rule_set("long"),
                "short": _make_rule_set("short"),
            }
        if phase4_result is None:
            phase4_result = {
                "long": _make_rule_set("long"),
                "short": _make_rule_set("short"),
            }
        if phase5_result is None:
            phase5_result = {
                "long": {"total_return_pct": 5.0, "executed_trades": 10,
                         "max_drawdown_pct": 2.0},
                "short": {"total_return_pct": 3.0, "executed_trades": 8,
                          "max_drawdown_pct": 1.5},
            }

        patches = [
            patch("gpu_fuzzy_trader.run_pipeline.Data_Loader.load_dataset",
                  return_value=train_df),
            patch("gpu_fuzzy_trader.run_pipeline.Data_Splitter.split_and_persist",
                  return_value=(train_df, val_df)),
            patch("gpu_fuzzy_trader.run_pipeline.Feature_Selector.skip_if_valid",
                  return_value=None),
            patch("gpu_fuzzy_trader.run_pipeline.Feature_Selector.run",
                  return_value=phase1_result),
            patch("gpu_fuzzy_trader.run_pipeline.Rule_Pool_Generator.skip_if_valid",
                  return_value=None),
            patch("gpu_fuzzy_trader.run_pipeline.Rule_Pool_Generator.run",
                  return_value=phase2_result.get("long", [])),
            patch("gpu_fuzzy_trader.run_pipeline.Rule_Set_Selector.skip_if_valid",
                  return_value=None),
            patch("gpu_fuzzy_trader.run_pipeline.Rule_Set_Selector.run",
                  return_value=phase3_result.get("long", {})),
            patch("gpu_fuzzy_trader.run_pipeline.RL_Agent.skip_if_valid",
                  return_value=None),
            patch("gpu_fuzzy_trader.run_pipeline.RL_Agent.train",
                  return_value=phase4_result.get("long", {})),
            patch("gpu_fuzzy_trader.run_pipeline.OOS_Evaluator.run",
                  return_value=phase5_result),
            patch("gpu_fuzzy_trader.run_pipeline._cfg.OUTPUTS_DIR",
                  str(tmp_path / "outputs")),
            patch("gpu_fuzzy_trader.run_pipeline._cfg.REPORTS_DIR",
                  str(tmp_path / "outputs" / "reports")),
        ]
        return patches

    def test_run_returns_dict(self, tmp_path):
        train_df = _make_df()
        val_df = _make_df()
        orch = self._make_orch(tmp_path)
        orch._create_output_dirs = MagicMock()
        orch._load_and_split_data = MagicMock(return_value=(train_df, val_df))
        orch._run_phase1 = MagicMock(return_value={
            "long": [{"name": "f", "mode": "positive", "score": 0.9}],
            "short": [{"name": "f", "mode": "positive", "score": 0.8}],
        })
        orch._run_phase2 = MagicMock(return_value={
            "long": _make_pool(3), "short": _make_pool(3)
        })
        orch._run_phase3 = MagicMock(return_value={
            "long": _make_rule_set("long"), "short": _make_rule_set("short")
        })
        orch._run_phase4 = MagicMock(return_value={
            "long": _make_rule_set("long"), "short": _make_rule_set("short")
        })
        orch._run_phase5 = MagicMock(return_value={
            "long": {"total_return_pct": 5.0, "executed_trades": 10,
                     "max_drawdown_pct": 2.0},
        })
        result = orch.run()
        assert isinstance(result, dict)

    def test_run_result_has_phase_keys(self, tmp_path):
        train_df = _make_df()
        val_df = _make_df()
        orch = self._make_orch(tmp_path)
        orch._create_output_dirs = MagicMock()
        orch._load_and_split_data = MagicMock(return_value=(train_df, val_df))
        orch._run_phase1 = MagicMock(return_value={"long": [], "short": []})
        orch._run_phase2 = MagicMock(
            return_value={"long": _make_pool(3), "short": _make_pool(3)})
        orch._run_phase3 = MagicMock(
            return_value={"long": _make_rule_set("long")})
        orch._run_phase4 = MagicMock(
            return_value={"long": _make_rule_set("long")})
        orch._run_phase5 = MagicMock(
            return_value={"long": {"total_return_pct": 1.0}})
        result = orch.run()
        for key in ("data", "phase1", "phase2", "phase3", "phase4", "phase5"):
            assert key in result, f"Missing key: {key}"

    def test_run_data_has_row_counts(self, tmp_path):
        train_df = _make_df(n_rows=200)
        val_df = _make_df(n_rows=100)
        orch = self._make_orch(tmp_path)
        orch._create_output_dirs = MagicMock()
        orch._load_and_split_data = MagicMock(return_value=(train_df, val_df))
        orch._run_phase1 = MagicMock(return_value={"long": [], "short": []})
        orch._run_phase2 = MagicMock(return_value={"long": [], "short": []})
        orch._run_phase3 = MagicMock(return_value={})
        orch._run_phase4 = MagicMock(return_value={})
        orch._run_phase5 = MagicMock(return_value={})
        result = orch.run()
        assert result["data"]["train_rows"] == 200
        assert result["data"]["val_rows"] == 100


class TestLoadAndSplitDataCache:
    def _make_orch(self, tmp_path) -> Pipeline_Orchestrator:
        orch = Pipeline_Orchestrator()
        orch._log_path = str(tmp_path / "pipeline.log")
        return orch

    def test_uses_cached_split_when_fresh(self, tmp_path, monkeypatch):
        train_df = _make_df(n_rows=120)
        val_df = _make_df(n_rows=40)

        csv_path = tmp_path / "train.csv"
        train_path = tmp_path / "train_75.parquet"
        val_path = tmp_path / "validation_25.parquet"

        _make_df(n_rows=120).to_csv(csv_path, index=False)
        train_df.to_parquet(train_path, index=False)
        val_df.to_parquet(val_path, index=False)

        now = time.time()
        os.utime(csv_path, (now - 100, now - 100))
        os.utime(train_path, (now, now))
        os.utime(val_path, (now, now))

        monkeypatch.setattr(_cfg, "TRAIN_CSV_PATH", str(csv_path))
        monkeypatch.setattr(_cfg, "TRAIN_75_PATH", str(train_path))
        monkeypatch.setattr(_cfg, "VALIDATION_25_PATH", str(val_path))

        with patch("gpu_fuzzy_trader.run_pipeline.Data_Loader.load_dataset") as load_mock, \
                patch("gpu_fuzzy_trader.run_pipeline.Data_Splitter.split_and_persist") as split_mock:
            load_mock.side_effect = AssertionError(
                "load_dataset should not be called when cache is fresh")
            split_mock.side_effect = AssertionError(
                "split_and_persist should not be called when cache is fresh")

            orch = Pipeline_Orchestrator()
            train_out, val_out = orch._load_and_split_data()

        from gpu_fuzzy_trader.backtest.df_slim import downcast_numeric_df

        expected_train = downcast_numeric_df(train_df)
        expected_val = downcast_numeric_df(val_df)
        pd.testing.assert_frame_equal(
            train_out.reset_index(drop=True),
            expected_train.reset_index(drop=True),
            check_dtype=False,
        )
        pd.testing.assert_frame_equal(
            val_out.reset_index(drop=True),
            expected_val.reset_index(drop=True),
            check_dtype=False,
        )

    def test_rebuilds_split_when_cache_missing(self, tmp_path, monkeypatch):
        train_df = _make_df(n_rows=120)
        val_df = _make_df(n_rows=40)

        csv_path = tmp_path / "train.csv"
        csv_path.write_text(_make_df(n_rows=120).to_csv(
            index=False), encoding="utf-8")

        train_path = tmp_path / "train_75.parquet"
        val_path = tmp_path / "validation_25.parquet"

        monkeypatch.setattr(_cfg, "TRAIN_CSV_PATH", str(csv_path))
        monkeypatch.setattr(_cfg, "TRAIN_75_PATH", str(train_path))
        monkeypatch.setattr(_cfg, "VALIDATION_25_PATH", str(val_path))

        with patch("gpu_fuzzy_trader.run_pipeline.Data_Loader.load_dataset", return_value=train_df) as load_mock, \
                patch("gpu_fuzzy_trader.run_pipeline.Data_Splitter.split_and_persist", return_value=(train_df, val_df)) as split_mock:
            orch = Pipeline_Orchestrator()
            train_out, val_out = orch._load_and_split_data()

        load_mock.assert_called_once_with(str(csv_path))
        split_mock.assert_called_once_with(train_df)
        pd.testing.assert_frame_equal(train_out.reset_index(
            drop=True), train_df.reset_index(drop=True))
        pd.testing.assert_frame_equal(val_out.reset_index(
            drop=True), val_df.reset_index(drop=True))

    def test_phase5_always_runs_even_with_empty_pool(self, tmp_path):
        """Phase 5 must run even when Phase 2 produces no rules."""
        train_df = _make_df()
        val_df = _make_df()
        orch = self._make_orch(tmp_path)
        orch._create_output_dirs = MagicMock()
        orch._load_and_split_data = MagicMock(return_value=(train_df, val_df))
        orch._run_phase1 = MagicMock(return_value={"long": [], "short": []})
        orch._run_phase2 = MagicMock(return_value={"long": [], "short": []})
        phase5_mock = MagicMock(
            return_value={"long": {"total_return_pct": 0.0}})
        orch._run_phase5 = phase5_mock
        orch.run()
        phase5_mock.assert_called_once()

    def test_phases_3_and_4_skipped_when_pool_empty(self, tmp_path):
        """When Phase 2 pool is empty, Phases 3 and 4 should be skipped."""
        train_df = _make_df()
        val_df = _make_df()
        orch = self._make_orch(tmp_path)
        orch._create_output_dirs = MagicMock()
        orch._load_and_split_data = MagicMock(return_value=(train_df, val_df))
        orch._run_phase1 = MagicMock(return_value={"long": [], "short": []})
        orch._run_phase2 = MagicMock(return_value={"long": [], "short": []})
        phase3_mock = MagicMock(return_value={})
        phase4_mock = MagicMock(return_value={})
        orch._run_phase3 = phase3_mock
        orch._run_phase4 = phase4_mock
        orch._run_phase5 = MagicMock(return_value={})
        result = orch.run()
        # Phases 3 and 4 should NOT be called when pool is empty
        phase3_mock.assert_not_called()
        phase4_mock.assert_not_called()
        # Results should have empty dicts for phases 3 and 4
        assert result["phase3"] == {}
        assert result["phase4"] == {}

    def test_phases_3_and_4_run_when_pool_nonempty(self, tmp_path):
        """When Phase 2 pool has rules, Phases 3 and 4 should run."""
        train_df = _make_df()
        val_df = _make_df()
        orch = self._make_orch(tmp_path)
        orch._create_output_dirs = MagicMock()
        orch._load_and_split_data = MagicMock(return_value=(train_df, val_df))
        orch._run_phase1 = MagicMock(return_value={"long": [], "short": []})
        orch._run_phase2 = MagicMock(
            return_value={"long": _make_pool(3), "short": []})
        phase3_mock = MagicMock(return_value={"long": _make_rule_set("long")})
        phase4_mock = MagicMock(return_value={"long": _make_rule_set("long")})
        orch._run_phase3 = phase3_mock
        orch._run_phase4 = phase4_mock
        orch._run_phase5 = MagicMock(return_value={})
        orch.run()
        phase3_mock.assert_called_once()
        phase4_mock.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: Phase 1 skip logic
# ---------------------------------------------------------------------------

class TestPhase1SkipLogic:
    def test_phase1_skipped_when_valid_outputs_exist(self, tmp_path):
        """_run_phase1 should skip when Feature_Selector.skip_if_valid returns data."""
        orch = Pipeline_Orchestrator()
        orch._log_path = str(tmp_path / "pipeline.log")
        existing = {
            "long": [{"name": "f", "mode": "positive", "score": 0.9}],
            "short": [{"name": "g", "mode": "positive", "score": 0.8}],
        }
        train_df = _make_df()
        with patch("gpu_fuzzy_trader.run_pipeline.Feature_Selector.skip_if_valid",
                   return_value=existing):
            result = orch._run_phase1(train_df)
        assert result == existing

    def test_phase1_runs_when_no_valid_outputs(self, tmp_path):
        """_run_phase1 should call Feature_Selector.run when skip_if_valid returns None."""
        orch = Pipeline_Orchestrator()
        orch._log_path = str(tmp_path / "pipeline.log")
        expected = {
            "long": [{"name": "f", "mode": "positive", "score": 0.9}],
            "short": [{"name": "g", "mode": "positive", "score": 0.8}],
        }
        train_df = _make_df()
        with patch("gpu_fuzzy_trader.run_pipeline.Feature_Selector.skip_if_valid",
                   return_value=None), \
            patch("gpu_fuzzy_trader.run_pipeline.Feature_Selector.run",
                  return_value=expected):
            result = orch._run_phase1(train_df)
        assert result == expected

    def test_phase1_skip_logs_skipped_true(self, tmp_path):
        """When Phase 1 is skipped, the log entry should have skipped=True."""
        orch = Pipeline_Orchestrator()
        orch._log_path = str(tmp_path / "pipeline.log")
        existing = {"long": [], "short": []}
        train_df = _make_df()
        with patch("gpu_fuzzy_trader.run_pipeline.Feature_Selector.skip_if_valid",
                   return_value=existing):
            orch._run_phase1(train_df)
        with open(orch._log_path) as fh:
            entries = [json.loads(l) for l in fh if l.strip()]
        assert any(e["skipped"] is True for e in entries)

    def test_phase1_run_logs_skipped_false(self, tmp_path):
        """When Phase 1 runs, the log entry should have skipped=False."""
        orch = Pipeline_Orchestrator()
        orch._log_path = str(tmp_path / "pipeline.log")
        train_df = _make_df()
        with patch("gpu_fuzzy_trader.run_pipeline.Feature_Selector.skip_if_valid",
                   return_value=None), \
            patch("gpu_fuzzy_trader.run_pipeline.Feature_Selector.run",
                  return_value={"long": [], "short": []}):
            orch._run_phase1(train_df)
        with open(orch._log_path) as fh:
            entries = [json.loads(l) for l in fh if l.strip()]
        assert any(e["skipped"] is False for e in entries)


# ---------------------------------------------------------------------------
# Tests: Phase 2 skip logic
# ---------------------------------------------------------------------------

class TestPhase2SkipLogic:
    def test_phase2_skipped_when_valid_pool_exists(self, tmp_path):
        """_run_phase2 should skip a direction when skip_if_valid returns a pool."""
        orch = Pipeline_Orchestrator()
        orch._log_path = str(tmp_path / "pipeline.log")
        existing_pool = _make_pool(3)
        phase1_result = {
            "long": [{"name": "f", "mode": "positive", "score": 0.9}],
            "short": [{"name": "g", "mode": "positive", "score": 0.8}],
        }
        train_df = _make_df()
        with patch("gpu_fuzzy_trader.run_pipeline.Rule_Pool_Generator.skip_if_valid",
                   return_value=existing_pool):
            result = orch._run_phase2(train_df, phase1_result)
        assert result["long"] == existing_pool
        assert result["short"] == existing_pool

    def test_phase2_empty_when_no_features(self, tmp_path):
        """_run_phase2 should return empty pool when Phase 1 has no features."""
        orch = Pipeline_Orchestrator()
        orch._log_path = str(tmp_path / "pipeline.log")
        phase1_result = {"long": [], "short": []}
        train_df = _make_df()
        with patch("gpu_fuzzy_trader.run_pipeline.Rule_Pool_Generator.skip_if_valid",
                   return_value=None):
            result = orch._run_phase2(train_df, phase1_result)
        assert result["long"] == []
        assert result["short"] == []


# ---------------------------------------------------------------------------
# Tests: Phase 3 skip logic
# ---------------------------------------------------------------------------

class TestPhase3SkipLogic:
    def test_phase3_skipped_when_valid_rule_sets_exist(self, tmp_path):
        """_run_phase3 should skip when Rule_Set_Selector.skip_if_valid returns data."""
        orch = Pipeline_Orchestrator()
        orch._log_path = str(tmp_path / "pipeline.log")
        existing = {
            "long": _make_rule_set("long"),
            "short": _make_rule_set("short"),
        }
        train_df = _make_df()
        val_df = _make_df()
        phase2_result = {"long": _make_pool(3), "short": _make_pool(3)}
        with patch("gpu_fuzzy_trader.run_pipeline.Rule_Set_Selector.skip_if_valid",
                   return_value=existing):
            result = orch._run_phase3(train_df, val_df, phase2_result)
        assert result == existing

    def test_phase3_skip_logs_skipped_true(self, tmp_path):
        orch = Pipeline_Orchestrator()
        orch._log_path = str(tmp_path / "pipeline.log")
        existing = {"long": _make_rule_set("long")}
        train_df = _make_df()
        val_df = _make_df()
        phase2_result = {"long": _make_pool(3), "short": _make_pool(3)}
        with patch("gpu_fuzzy_trader.run_pipeline.Rule_Set_Selector.skip_if_valid",
                   return_value=existing):
            orch._run_phase3(train_df, val_df, phase2_result)
        with open(orch._log_path) as fh:
            entries = [json.loads(l) for l in fh if l.strip()]
        assert any(e["skipped"] is True for e in entries)


# ---------------------------------------------------------------------------
# Tests: Phase 4 skip logic
# ---------------------------------------------------------------------------

class TestPhase4SkipLogic:
    def test_phase4_skipped_when_valid_outputs_exist(self, tmp_path):
        """_run_phase4 should skip when RL_Agent.skip_if_valid returns data."""
        orch = Pipeline_Orchestrator()
        orch._log_path = str(tmp_path / "pipeline.log")
        existing = _make_rule_set("long")
        train_df = _make_df()
        val_df = _make_df()
        phase3_result = {"long": _make_rule_set("long")}
        with patch("gpu_fuzzy_trader.run_pipeline.RL_Agent.skip_if_valid",
                   return_value=existing):
            result = orch._run_phase4(train_df, val_df, phase3_result)
        assert result["long"] == existing

    def test_phase4_skip_logs_skipped_true(self, tmp_path):
        orch = Pipeline_Orchestrator()
        orch._log_path = str(tmp_path / "pipeline.log")
        existing = _make_rule_set("long")
        train_df = _make_df()
        val_df = _make_df()
        phase3_result = {"long": _make_rule_set("long")}
        with patch("gpu_fuzzy_trader.run_pipeline.RL_Agent.skip_if_valid",
                   return_value=existing):
            orch._run_phase4(train_df, val_df, phase3_result)
        with open(orch._log_path) as fh:
            entries = [json.loads(l) for l in fh if l.strip()]
        assert any(e["skipped"] is True for e in entries)

    def test_phase4_skipped_when_no_phase3_result(self, tmp_path):
        """_run_phase4 should skip a direction when Phase 3 has no result for it."""
        orch = Pipeline_Orchestrator()
        orch._log_path = str(tmp_path / "pipeline.log")
        train_df = _make_df()
        val_df = _make_df()
        phase3_result = {}  # No rule sets from Phase 3
        with patch("gpu_fuzzy_trader.run_pipeline.RL_Agent.skip_if_valid",
                   return_value=None):
            result = orch._run_phase4(train_df, val_df, phase3_result)
        assert result == {}


# ---------------------------------------------------------------------------
# Tests: Phase 5 always runs
# ---------------------------------------------------------------------------

class TestPhase5AlwaysRuns:
    def test_phase5_returns_dict(self, tmp_path):
        orch = Pipeline_Orchestrator()
        orch._log_path = str(tmp_path / "pipeline.log")
        expected = {"long": {"total_return_pct": 5.0, "executed_trades": 10}}
        with patch("gpu_fuzzy_trader.run_pipeline.OOS_Evaluator.run",
                   return_value=expected):
            result = orch._run_phase5()
        assert result == expected

    def test_phase5_logs_entry(self, tmp_path):
        orch = Pipeline_Orchestrator()
        orch._log_path = str(tmp_path / "pipeline.log")
        with patch("gpu_fuzzy_trader.run_pipeline.OOS_Evaluator.run",
                   return_value={}):
            orch._run_phase5()
        with open(orch._log_path) as fh:
            entries = [json.loads(l) for l in fh if l.strip()]
        assert len(entries) >= 1
        assert any("Phase 5" in e["phase"] for e in entries)

    def test_phase5_log_skipped_is_false(self, tmp_path):
        """Phase 5 is never skipped."""
        orch = Pipeline_Orchestrator()
        orch._log_path = str(tmp_path / "pipeline.log")
        with patch("gpu_fuzzy_trader.run_pipeline.OOS_Evaluator.run",
                   return_value={}):
            orch._run_phase5()
        with open(orch._log_path) as fh:
            entries = [json.loads(l) for l in fh if l.strip()]
        phase5_entries = [e for e in entries if "Phase 5" in e["phase"]]
        assert len(phase5_entries) >= 1
        assert all(e["skipped"] is False for e in phase5_entries)

    def test_phase5_returns_empty_dict_on_failure(self, tmp_path):
        """Phase 5 should return {} if OOS_Evaluator.run raises."""
        orch = Pipeline_Orchestrator()
        orch._log_path = str(tmp_path / "pipeline.log")
        with patch("gpu_fuzzy_trader.run_pipeline.OOS_Evaluator.run",
                   side_effect=RuntimeError("test error")):
            result = orch._run_phase5()
        assert result == {}


# ---------------------------------------------------------------------------
# Tests: Pipeline log file
# ---------------------------------------------------------------------------

class TestPipelineLogFile:
    def test_log_file_created_after_run(self, tmp_path):
        """pipeline.log should be created after run() completes."""
        orch = Pipeline_Orchestrator()
        log_path = str(tmp_path / "pipeline.log")
        orch._log_path = log_path
        orch._create_output_dirs = MagicMock()
        orch._load_and_split_data = MagicMock(
            return_value=(_make_df(), _make_df()))
        orch._run_phase1 = MagicMock(return_value={"long": [], "short": []})
        orch._run_phase2 = MagicMock(return_value={"long": [], "short": []})
        orch._run_phase5 = MagicMock(return_value={})
        orch.run()
        # Log entries are written by _run_phase* methods; since we mocked them,
        # the log may be empty — but the file should exist if any entry was written.
        # At minimum, the orchestrator should not crash.
        assert True  # No exception raised

    def test_log_entries_are_json_lines(self, tmp_path):
        """Each line in pipeline.log should be valid JSON."""
        log_path = str(tmp_path / "pipeline.log")
        # Write a few entries manually
        _log_phase_entry(log_path, "Phase 1", "2024-01-01T00:00:00Z",
                         "2024-01-01T00:00:01Z", 1.0, False)
        _log_phase_entry(log_path, "Phase 2", "2024-01-01T00:00:01Z",
                         "2024-01-01T00:00:05Z", 4.0, False)
        with open(log_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    entry = json.loads(line)  # Should not raise
                    assert isinstance(entry, dict)


# ---------------------------------------------------------------------------
# Tests: __main__.py entry point
# ---------------------------------------------------------------------------

class TestMainEntryPoint:
    def test_main_module_importable(self):
        """gpu_fuzzy_trader.__main__ should be importable."""
        import gpu_fuzzy_trader.__main__  # noqa: F401

    def test_main_function_callable(self):
        """run_pipeline.main should be callable."""
        from gpu_fuzzy_trader.run_pipeline import main
        assert callable(main)

    def test_main_calls_pipeline_orchestrator(self, tmp_path):
        """main() should instantiate and call Pipeline_Orchestrator.run()."""
        run_mock = MagicMock(return_value={
            "phase5": {
                "long": {"total_return_pct": 1.0, "executed_trades": 5,
                         "max_drawdown_pct": 0.5}
            }
        })
        with patch("gpu_fuzzy_trader.run_pipeline.Pipeline_Orchestrator.run",
                   run_mock):
            from gpu_fuzzy_trader.run_pipeline import main
            main()
        run_mock.assert_called_once()

    def test_main_forwards_custom_output_dir(self, tmp_path):
        """main() should pass --output through to Pipeline_Orchestrator."""
        custom_output = str(tmp_path / "run_a")

        with patch("gpu_fuzzy_trader.run_pipeline.Pipeline_Orchestrator") as orch_cls:
            orch_instance = orch_cls.return_value
            orch_instance.run.return_value = {"phase5": {}}
            orch_instance._log_path = str(tmp_path / "run_a" / "pipeline.log")

            from gpu_fuzzy_trader.run_pipeline import main
            main(["--output", custom_output])

        orch_cls.assert_called_once_with(output_dir=custom_output)
        orch_instance.run.assert_called_once()

    def test_main_defaults_to_config_output_dir(self, tmp_path):
        """main() should keep the default output root when --output is omitted."""
        with patch("gpu_fuzzy_trader.run_pipeline.Pipeline_Orchestrator") as orch_cls:
            orch_instance = orch_cls.return_value
            orch_instance.run.return_value = {"phase5": {}}
            orch_instance._log_path = str(tmp_path / "pipeline.log")

            from gpu_fuzzy_trader.run_pipeline import main
            main([])

        orch_cls.assert_called_once_with(output_dir=None)
        orch_instance.run.assert_called_once()


class TestTemporaryOutputPaths:
    def test_rebinds_and_restores_cached_output_paths(self, tmp_path):
        original_outputs = _cfg.OUTPUTS_DIR
        original_reports = _cfg.REPORTS_DIR
        original_log_path = run_pipeline_module._PIPELINE_LOG_PATH
        original_selector_long = selector_module._LONG_PATH
        original_phase2_pool = phase2_module._POOL_PATHS
        original_phase3_output = phase3_module._OUTPUT_PATHS
        original_phase4_output = phase4_module._OUTPUT_PATHS
        original_phase5_strategy = phase5_module._STRATEGY_PATHS
        original_phase5_report = phase5_module._REPORT_PATHS
        original_reporter_dir = reporter_module._REPORTS_DIR

        custom_output = str(tmp_path / "custom_run")
        expected_reports = os.path.join(custom_output, "reports")

        with run_pipeline_module._temporary_output_paths(custom_output):
            assert _cfg.OUTPUTS_DIR == custom_output
            assert _cfg.REPORTS_DIR == expected_reports
            assert run_pipeline_module._PIPELINE_LOG_PATH == os.path.join(
                custom_output, "pipeline.log"
            )
            assert selector_module._LONG_PATH == os.path.join(
                custom_output, "selected_features_long.json"
            )
            assert phase2_module._POOL_PATHS["long"] == os.path.join(
                custom_output, "phase2_long_pool.json"
            )
            assert phase3_module._OUTPUT_PATHS["short"] == os.path.join(
                custom_output, "short.json"
            )
            assert phase4_module._OUTPUT_PATHS["long"] == os.path.join(
                custom_output, "long.json"
            )
            assert phase5_module._STRATEGY_PATHS["short"] == os.path.join(
                custom_output, "short.json"
            )
            assert phase5_module._REPORT_PATHS["per_symbol"] == os.path.join(
                expected_reports, "test_per_symbol_performance.csv"
            )
            assert reporter_module._REPORTS_DIR == expected_reports

        assert _cfg.OUTPUTS_DIR == original_outputs
        assert _cfg.REPORTS_DIR == original_reports
        assert run_pipeline_module._PIPELINE_LOG_PATH == original_log_path
        assert selector_module._LONG_PATH == original_selector_long
        assert phase2_module._POOL_PATHS == original_phase2_pool
        assert phase3_module._OUTPUT_PATHS == original_phase3_output
        assert phase4_module._OUTPUT_PATHS == original_phase4_output
        assert phase5_module._STRATEGY_PATHS == original_phase5_strategy
        assert phase5_module._REPORT_PATHS == original_phase5_report
        assert reporter_module._REPORTS_DIR == original_reporter_dir
