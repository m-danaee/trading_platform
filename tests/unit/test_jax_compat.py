"""Tests for JAX / GPU engine availability detection."""

from __future__ import annotations

import builtins

import pytest


def test_get_gpu_backtest_engine_class_returns_none_on_runtime_error(monkeypatch):
    import gpu_fuzzy_trader.backtest.jax_compat as jc

    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "gpu_fuzzy_trader.backtest.gpu_engine":
            raise RuntimeError(
                "This version of jaxlib was built using AVX instructions"
            )
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert jc.get_gpu_backtest_engine_class() is None
    assert jc.jax_gpu_backtest_available() is False


def test_cpu_engine_import_without_jax(monkeypatch):
    """Package init must not crash when gpu_engine import fails."""
    import gpu_fuzzy_trader.backtest.jax_compat as jc

    monkeypatch.setattr(jc, "get_gpu_backtest_engine_class", lambda: None)

    from gpu_fuzzy_trader.backtest.cpu_engine import CPUBacktestEngine

    assert CPUBacktestEngine is not None


def test_run_pipeline_imports_on_cpu_only_host(monkeypatch):
    import gpu_fuzzy_trader.backtest.jax_compat as jc

    monkeypatch.setattr(jc, "get_gpu_backtest_engine_class", lambda: None)

    from gpu_fuzzy_trader.run_pipeline import Pipeline_Orchestrator

    assert Pipeline_Orchestrator is not None
