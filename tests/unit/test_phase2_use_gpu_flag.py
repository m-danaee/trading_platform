"""Phase 2 respects PHASE2_USE_GPU=False without loading JAX GPU engine."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.phases.phase2_rule_pool import Rule_Pool_Generator


def _minimal_backtest_df(n: int = 50) -> pd.DataFrame:
    return pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=n, freq="h"),
        "symbol": ["A"] * n,
        "label_open_next": [100.0] * n,
        "label_max_288": [105.0] * n,
        "label_min_288": [97.0] * n,
        "label_close_288": [102.0] * n,
        "label_max_before_min": [1] * n,
        "feat_a": [0.5] * n,
    })


class TestPhase2UseGpuFlag:
    def test_phase2_use_gpu_false_skips_jax_lookup(
        self, monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr(_cfg, "PHASE2_USE_GPU", False)
        gen = SimpleNamespace(
            _feature_modes={},
            direction="long",
            _regime_row_fractions=None,
        )
        mock_get_gpu = MagicMock(return_value=MagicMock)
        df = _minimal_backtest_df()
        with patch(
            "gpu_fuzzy_trader.backtest.jax_compat.get_gpu_backtest_engine_class",
            mock_get_gpu,
        ):
            engine = Rule_Pool_Generator._build_engine_for_df(gen, df)

        mock_get_gpu.assert_not_called()
        from gpu_fuzzy_trader.backtest.cpu_engine import CPUBacktestEngine

        assert isinstance(engine, CPUBacktestEngine)

    def test_large_window_cpu_route_skips_jax_engine_construction(
        self, monkeypatch: pytest.MonkeyPatch,
    ):
        """The memory-safe CPU route must happen before JAX allocates arrays."""
        monkeypatch.setattr(_cfg, "PHASE2_USE_GPU", True)
        monkeypatch.setattr(_cfg, "PHASE2_GPU_CPU_ROUTE_LARGE_DATA", True)
        monkeypatch.setattr(_cfg, "PHASE2_GPU_CPU_ROUTE_MIN_BARS", 1)
        monkeypatch.setattr(_cfg, "PHASE2_GPU_CPU_ROUTE_MAX_BATCH", 200)
        gen = SimpleNamespace(
            _feature_modes={},
            direction="long",
            _regime_row_fractions=None,
            pop_size=200,
            _train_df=_minimal_backtest_df(),
        )
        df = _minimal_backtest_df()
        mock_get_gpu = MagicMock(side_effect=AssertionError("JAX lookup is unsafe"))
        with patch(
            "gpu_fuzzy_trader.backtest.jax_compat.get_gpu_backtest_engine_class",
            mock_get_gpu,
        ):
            engine = Rule_Pool_Generator._build_engine_for_df(gen, df)

        mock_get_gpu.assert_not_called()
        from gpu_fuzzy_trader.backtest.cpu_engine import CPUBacktestEngine

        assert isinstance(engine, CPUBacktestEngine)

    def test_cpu_backend_skips_jax_warmup(
        self, monkeypatch: pytest.MonkeyPatch,
    ):
        """A selected CPU backend must not initialize JAX just to warm up."""
        monkeypatch.setattr(_cfg, "PHASE2_USE_GPU", True)
        cpu_engine = SimpleNamespace(simulate_rule_batch=lambda **_: [])
        from gpu_fuzzy_trader import _gpu_runtime

        with patch.object(_gpu_runtime, "log_gpu_runtime_config") as log_runtime, patch.object(
            _gpu_runtime, "warmup_phase2_gpu_kernels"
        ) as warmup:
            _gpu_runtime.configure_phase2_gpu_runtime(cpu_engine)

        log_runtime.assert_not_called()
        warmup.assert_not_called()
