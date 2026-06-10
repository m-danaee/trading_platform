"""Unit tests for crash-fix-and-run-logging spec.

Covers:
  - Task 1.1: Smoke tests for config changes (PHASE1_SAMPLING_TOTAL, RUN_LOG_PATH)
  - Task 2.1: Smoke test for XLA_PYTHON_CLIENT_PREALLOCATE being set by configure_jax_env()
"""

from __future__ import annotations
from unittest.mock import patch
import pandas as pd
import numpy as np
import logging
import contextlib

import os

import pytest

import gpu_fuzzy_trader.config as _cfg
from gpu_fuzzy_trader._jax_env import configure_jax_env


class TestConfigPhase2Seed:
    """Phase 2 reproducibility default seed."""

    def test_phase2_seed_matches_get_seed(self) -> None:
        """PHASE2_SEED is drawn once at import via get_seed(); set GLOBAL_SEED=42 to fix it."""
        assert _cfg.PHASE2_SEED == _cfg.get_seed()

    def test_global_seed_overrides_get_seed(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(_cfg, "GLOBAL_SEED", 42)
        assert _cfg.get_seed() == 42


class TestConfigRunLogPath:
    """Task 1.1 — Requirement 6.1: RUN_LOG_PATH defined and derived from OUTPUTS_DIR."""

    def test_run_log_path_ends_with_run_log(self) -> None:
        assert _cfg.RUN_LOG_PATH.endswith("run.log")

    def test_run_log_path_starts_with_outputs_dir(self) -> None:
        assert _cfg.RUN_LOG_PATH.startswith(_cfg.OUTPUTS_DIR)


class TestConfigureJaxEnvPreallocate:
    """Task 2.1 — smoke test for XLA_PYTHON_CLIENT_PREALLOCATE."""

    def test_sets_preallocate_to_false_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When XLA_PYTHON_CLIENT_PREALLOCATE is not in the environment,
        configure_jax_env() must set it to 'false' to avoid VRAM hoarding."""
        monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)

        configure_jax_env()

        assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"


# ---------------------------------------------------------------------------
# Helpers shared by Task 4.x tests
# ---------------------------------------------------------------------------


def _make_feature_infos_crash(modes: list[str]) -> list[dict]:
    return [{"name": f"feat_{i}", "mode": m, "score": 0.5} for i, m in enumerate(modes)]


def _make_train_df_crash(n_rows: int = 200, n_features: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(0)
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
            "label_min_288": open_next * rng.uniform(0.90, 1.02, size=n),
            "label_max_288": open_next * rng.uniform(0.98, 1.10, size=n),
            "label_max_before_min": rng.integers(0, 2, size=n).astype(float),
            "_symbol_bar_index": np.arange(n),
        }
        for i in range(n_features):
            data[f"feat_{i}"] = rng.integers(0, 5, size=n).astype(float)
        dfs.append(pd.DataFrame(data))
    return pd.concat(dfs, ignore_index=True)


@contextlib.contextmanager
def _capture_logs(logger_name: str, level: int):
    """Capture log records from a named logger at or above *level*."""
    target_logger = logging.getLogger(logger_name)
    records: list[logging.LogRecord] = []

    class _Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Handler(level=level)
    old_level = target_logger.level
    target_logger.setLevel(min(old_level, level) if old_level != 0 else level)
    target_logger.addHandler(handler)
    try:
        yield records
    finally:
        target_logger.removeHandler(handler)
        target_logger.setLevel(old_level)


# ---------------------------------------------------------------------------
# Task 4.1 — Archive save call ordering (Requirement 3.1)
# ---------------------------------------------------------------------------

class TestArchiveSaveCallOrdering:
    """
    Requirement 3.1 — save_archive is called before _release_resources in run().
    """

    def _setup_paths(self, tmp_path, direction: str = "long"):
        import gpu_fuzzy_trader.phases.phase2_rule_pool as m
        original_pool = m._POOL_PATHS.copy()
        original_hist = m._HISTORY_PATHS.copy()
        original_archive = m._ARCHIVE_PATHS.copy()
        m._POOL_PATHS[direction] = str(
            tmp_path / f"phase2_{direction}_pool.json")
        m._HISTORY_PATHS[direction] = str(
            tmp_path / f"phase2_{direction}_history.json")
        m._ARCHIVE_PATHS[direction] = str(
            tmp_path / f"phase2_{direction}_archive.json")
        return original_pool, original_hist, original_archive

    def _restore_paths(self, original_pool, original_hist, original_archive):
        import gpu_fuzzy_trader.phases.phase2_rule_pool as m
        m._POOL_PATHS.update(original_pool)
        m._HISTORY_PATHS.update(original_hist)
        m._ARCHIVE_PATHS.update(original_archive)

    def test_save_archive_called_before_release_resources(self, tmp_path):
        """
        Mock save_archive and _release_resources, run Rule_Pool_Generator.run()
        with a minimal population (pop_size=2, n_generations=1) using CPUBacktestEngine,
        and assert save_archive was called before _release_resources.
        """
        from gpu_fuzzy_trader.phases.phase2_rule_pool import Rule_Pool_Generator

        fi = _make_feature_infos_crash(
            ["positive", "positive", "positive", "positive"])
        df = _make_train_df_crash(n_rows=200, n_features=4)

        original_pool, original_hist, original_archive = self._setup_paths(
            tmp_path, "long")

        call_order: list[str] = []

        def fake_save_archive(direction, feature_infos, pool, **kwargs):
            call_order.append("save_archive")
            return pool

        def fake_release_resources(self_inner):
            call_order.append("_release_resources")

        try:
            gen = Rule_Pool_Generator(
                df, fi, "long", pop_size=2, n_generations=1, seed=0)

            with patch.object(
                Rule_Pool_Generator, "save_archive", side_effect=fake_save_archive
            ), patch.object(
                Rule_Pool_Generator, "_release_resources", fake_release_resources
            ):
                gen.run()

            assert "save_archive" in call_order, "save_archive was never called"
            assert "_release_resources" in call_order, "_release_resources was never called"
            assert call_order.index("save_archive") < call_order.index("_release_resources"), (
                f"Expected save_archive before _release_resources, got order: {call_order}"
            )
        finally:
            self._restore_paths(original_pool, original_hist, original_archive)

    def test_save_archive_receives_correct_direction(self, tmp_path):
        """save_archive is called with self.direction as the first argument."""
        from gpu_fuzzy_trader.phases.phase2_rule_pool import Rule_Pool_Generator

        fi = _make_feature_infos_crash(
            ["positive", "positive", "positive", "positive"])
        df = _make_train_df_crash(n_rows=200, n_features=4)

        original_pool, original_hist, original_archive = self._setup_paths(
            tmp_path, "short")

        captured_direction: list[str] = []

        def fake_save_archive(direction, feature_infos, pool, **kwargs):
            captured_direction.append(direction)
            return pool

        try:
            gen = Rule_Pool_Generator(
                df, fi, "short", pop_size=2, n_generations=1, seed=0)

            with patch.object(Rule_Pool_Generator, "save_archive", side_effect=fake_save_archive):
                gen.run()

            assert captured_direction == ["short"], (
                f"Expected save_archive called with 'short', got {captured_direction}"
            )
        finally:
            self._restore_paths(original_pool, original_hist, original_archive)


# ---------------------------------------------------------------------------
# Task 4.2 — Archive save exception handling (Requirement 3.3)
# ---------------------------------------------------------------------------

class TestArchiveSaveExceptionHandling:
    """
    Requirement 3.3 — If save_archive raises, the exception is caught, a WARNING
    is logged, and execution continues to _release_resources() without re-raising.
    """

    def _setup_paths(self, tmp_path, direction: str = "long"):
        import gpu_fuzzy_trader.phases.phase2_rule_pool as m
        original_pool = m._POOL_PATHS.copy()
        original_hist = m._HISTORY_PATHS.copy()
        original_archive = m._ARCHIVE_PATHS.copy()
        m._POOL_PATHS[direction] = str(
            tmp_path / f"phase2_{direction}_pool.json")
        m._HISTORY_PATHS[direction] = str(
            tmp_path / f"phase2_{direction}_history.json")
        m._ARCHIVE_PATHS[direction] = str(
            tmp_path / f"phase2_{direction}_archive.json")
        return original_pool, original_hist, original_archive

    def _restore_paths(self, original_pool, original_hist, original_archive):
        import gpu_fuzzy_trader.phases.phase2_rule_pool as m
        m._POOL_PATHS.update(original_pool)
        m._HISTORY_PATHS.update(original_hist)
        m._ARCHIVE_PATHS.update(original_archive)

    def test_save_archive_exception_does_not_propagate(self, tmp_path):
        """
        When save_archive raises RuntimeError("disk full"), run() must not
        propagate the exception and must still return the pool list.
        """
        from gpu_fuzzy_trader.phases.phase2_rule_pool import Rule_Pool_Generator

        fi = _make_feature_infos_crash(
            ["positive", "positive", "positive", "positive"])
        df = _make_train_df_crash(n_rows=200, n_features=4)

        original_pool, original_hist, original_archive = self._setup_paths(
            tmp_path, "long")

        try:
            gen = Rule_Pool_Generator(
                df, fi, "long", pop_size=2, n_generations=1, seed=0)

            with patch.object(
                Rule_Pool_Generator,
                "save_archive",
                side_effect=RuntimeError("disk full"),
            ):
                result = gen.run()  # must not raise

            assert isinstance(
                result, list), "run() should still return the pool list"
        finally:
            self._restore_paths(original_pool, original_hist, original_archive)

    def test_save_archive_exception_emits_warning_log(self, tmp_path):
        """
        When save_archive raises, a WARNING log record containing
        'archive save failed' must be emitted.
        """
        from gpu_fuzzy_trader.phases.phase2_rule_pool import Rule_Pool_Generator

        fi = _make_feature_infos_crash(
            ["positive", "positive", "positive", "positive"])
        df = _make_train_df_crash(n_rows=200, n_features=4)

        original_pool, original_hist, original_archive = self._setup_paths(
            tmp_path, "long")

        try:
            gen = Rule_Pool_Generator(
                df, fi, "long", pop_size=2, n_generations=1, seed=0)

            with patch.object(
                Rule_Pool_Generator,
                "save_archive",
                side_effect=RuntimeError("disk full"),
            ), _capture_logs(
                "gpu_fuzzy_trader.phases.phase2_rule_pool", logging.WARNING
            ) as log_records:
                gen.run()

            warning_messages = [
                r.getMessage() for r in log_records
                if r.levelno == logging.WARNING
            ]
            assert any("archive save failed" in msg for msg in warning_messages), (
                f"Expected WARNING containing 'archive save failed', got: {warning_messages}"
            )
        finally:
            self._restore_paths(original_pool, original_hist, original_archive)

    def test_release_resources_still_called_after_save_archive_exception(self, tmp_path):
        """
        When save_archive raises, _release_resources must still be called.
        """
        from gpu_fuzzy_trader.phases.phase2_rule_pool import Rule_Pool_Generator

        fi = _make_feature_infos_crash(
            ["positive", "positive", "positive", "positive"])
        df = _make_train_df_crash(n_rows=200, n_features=4)

        original_pool, original_hist, original_archive = self._setup_paths(
            tmp_path, "long")

        release_called: list[bool] = []

        def fake_release(self_inner):
            release_called.append(True)

        try:
            gen = Rule_Pool_Generator(
                df, fi, "long", pop_size=2, n_generations=1, seed=0)

            with patch.object(
                Rule_Pool_Generator,
                "save_archive",
                side_effect=RuntimeError("disk full"),
            ), patch.object(
                Rule_Pool_Generator, "_release_resources", fake_release
            ):
                gen.run()

            assert release_called, "_release_resources was not called after save_archive exception"
        finally:
            self._restore_paths(original_pool, original_hist, original_archive)


# ---------------------------------------------------------------------------
# Task 5.1 — run.log handler lifecycle (Requirements 1.1, 1.4, 1.5, 1.6, 1.7)
# ---------------------------------------------------------------------------

class TestRunLogHandlerLifecycle:
    """
    Requirements 1.1, 1.4, 1.5, 1.6, 1.7 — run.log FileHandler is attached,
    writes START/END separators, and is detached after run() returns.
    """

    @staticmethod
    def _count_file_handlers(path: str) -> int:
        """Count FileHandlers on the root logger pointing to *path*."""
        root = logging.getLogger()
        return sum(
            1
            for h in root.handlers
            if isinstance(h, logging.FileHandler)
            and os.path.abspath(getattr(h, "baseFilename", "")) == os.path.abspath(path)
        )

    @staticmethod
    def _mock_all_phases(monkeypatch: pytest.MonkeyPatch) -> None:
        """Patch every phase method on Pipeline_Orchestrator to be a no-op."""
        from gpu_fuzzy_trader.run_pipeline import Pipeline_Orchestrator

        monkeypatch.setattr(
            Pipeline_Orchestrator,
            "_load_and_split_data",
            lambda self: (pd.DataFrame(), pd.DataFrame()),
        )
        monkeypatch.setattr(
            Pipeline_Orchestrator,
            "_run_phase1",
            lambda self, train_df, **_kw: {"long": [], "short": []},
        )
        monkeypatch.setattr(
            Pipeline_Orchestrator,
            "_prune_train_df_after_phase1",
            lambda self, train_df, phase1_result: train_df,
        )
        monkeypatch.setattr(
            Pipeline_Orchestrator,
            "_run_phase2",
            lambda self, train_df, phase1_result, **_kw: {
                "long": [], "short": []},
        )
        monkeypatch.setattr(
            Pipeline_Orchestrator,
            "_run_phase3",
            lambda self, train_df, val_df, phase2_result, **_kw: {},
        )
        monkeypatch.setattr(
            Pipeline_Orchestrator,
            "_run_phase4",
            lambda self, train_df, val_df, phase3_result, **_kw: {},
        )
        monkeypatch.setattr(
            Pipeline_Orchestrator,
            "_run_phase5",
            lambda self, **_kw: {},
        )

    def test_run_log_file_created_and_contains_start_end(
        self, tmp_path: "pytest.TempPathFactory", monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run.log must exist after run() and contain both separator lines."""
        import gpu_fuzzy_trader.config as cfg
        from gpu_fuzzy_trader.run_pipeline import Pipeline_Orchestrator

        run_log = tmp_path / "run.log"
        monkeypatch.setattr(cfg, "RUN_LOG_PATH", str(run_log))
        monkeypatch.setattr(cfg, "OUTPUTS_DIR", str(tmp_path))
        monkeypatch.setattr(cfg, "REPORTS_DIR", str(tmp_path / "reports"))

        self._mock_all_phases(monkeypatch)

        orch = Pipeline_Orchestrator(output_dir=str(tmp_path))
        orch.run()

        assert run_log.exists(), "run.log was not created"
        content = run_log.read_text(encoding="utf-8")
        assert "Pipeline run START" in content, "START separator missing from run.log"
        assert "Pipeline run END" in content, "END separator missing from run.log"

    def test_handler_detached_after_run(
        self, tmp_path: "pytest.TempPathFactory", monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Root logger must have no extra FileHandlers pointing to run.log after run()."""
        import gpu_fuzzy_trader.config as cfg
        from gpu_fuzzy_trader.run_pipeline import Pipeline_Orchestrator

        run_log = tmp_path / "run.log"
        monkeypatch.setattr(cfg, "RUN_LOG_PATH", str(run_log))
        monkeypatch.setattr(cfg, "OUTPUTS_DIR", str(tmp_path))
        monkeypatch.setattr(cfg, "REPORTS_DIR", str(tmp_path / "reports"))

        self._mock_all_phases(monkeypatch)

        orch = Pipeline_Orchestrator(output_dir=str(tmp_path))
        orch.run()

        assert self._count_file_handlers(str(run_log)) == 0, (
            "FileHandler for run.log was not removed from root logger after run()"
        )

    def test_handler_detached_after_run_exception(
        self, tmp_path: "pytest.TempPathFactory", monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Handler must be detached even when run() raises an exception."""
        import gpu_fuzzy_trader.config as cfg
        from gpu_fuzzy_trader.run_pipeline import Pipeline_Orchestrator

        run_log = tmp_path / "run.log"
        monkeypatch.setattr(cfg, "RUN_LOG_PATH", str(run_log))
        monkeypatch.setattr(cfg, "OUTPUTS_DIR", str(tmp_path))
        monkeypatch.setattr(cfg, "REPORTS_DIR", str(tmp_path / "reports"))

        # Make _load_and_split_data raise to simulate a mid-run crash
        monkeypatch.setattr(
            Pipeline_Orchestrator,
            "_load_and_split_data",
            lambda self: (_ for _ in ()).throw(
                RuntimeError("simulated crash")),
        )

        orch = Pipeline_Orchestrator(output_dir=str(tmp_path))
        with pytest.raises(RuntimeError, match="simulated crash"):
            orch.run()

        assert self._count_file_handlers(str(run_log)) == 0, (
            "FileHandler for run.log was not removed after an exception in run()"
        )

    def test_run_log_append_mode_across_two_runs(
        self, tmp_path: "pytest.TempPathFactory", monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Second run() must append to run.log; first run's separators must still be present."""
        import gpu_fuzzy_trader.config as cfg
        from gpu_fuzzy_trader.run_pipeline import Pipeline_Orchestrator

        run_log = tmp_path / "run.log"
        monkeypatch.setattr(cfg, "RUN_LOG_PATH", str(run_log))
        monkeypatch.setattr(cfg, "OUTPUTS_DIR", str(tmp_path))
        monkeypatch.setattr(cfg, "REPORTS_DIR", str(tmp_path / "reports"))

        self._mock_all_phases(monkeypatch)

        orch = Pipeline_Orchestrator(output_dir=str(tmp_path))
        orch.run()
        content_after_first = run_log.read_text(encoding="utf-8")

        # Second run — file must be appended, not truncated
        orch2 = Pipeline_Orchestrator(output_dir=str(tmp_path))
        orch2.run()
        content_after_second = run_log.read_text(encoding="utf-8")

        # First run's content must still be present
        assert content_after_first in content_after_second, (
            "run.log was truncated between runs (not opened in append mode)"
        )
        # Both runs must have written START and END
        assert content_after_second.count("Pipeline run START") == 2
        assert content_after_second.count("Pipeline run END") == 2
