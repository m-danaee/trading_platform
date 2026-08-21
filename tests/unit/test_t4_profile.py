"""Unit tests for T4 profile detection and hardware probes."""

from __future__ import annotations

import os
import subprocess
from unittest.mock import patch

import pytest

from gpu_fuzzy_trader import _gpu_runtime
from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.research_profile import HardwareProfile


def test_detect_gpu_name_success():
    _gpu_runtime.detect_gpu_name.cache_clear()
    with patch("subprocess.check_output", return_value="Tesla T4\n"):
        assert _gpu_runtime.detect_gpu_name() == "Tesla T4"


def test_detect_gpu_name_failure():
    _gpu_runtime.detect_gpu_name.cache_clear()
    with patch("subprocess.check_output", side_effect=FileNotFoundError):
        assert _gpu_runtime.detect_gpu_name() is None


def test_is_t4_runtime_by_name():
    _gpu_runtime.is_t4_runtime.cache_clear()
    _gpu_runtime.detect_gpu_name.cache_clear()
    _gpu_runtime.detect_gpu_vram_gb.cache_clear()
    with patch.dict("os.environ", {}, clear=True), \
         patch("gpu_fuzzy_trader._gpu_runtime.detect_gpu_name", return_value="NVIDIA Tesla T4"):
        assert _gpu_runtime.is_t4_runtime() is True


def test_is_t4_runtime_by_vram_heuristic():
    _gpu_runtime.is_t4_runtime.cache_clear()
    _gpu_runtime.detect_gpu_name.cache_clear()
    _gpu_runtime.detect_gpu_vram_gb.cache_clear()
    with patch.dict("os.environ", {}, clear=True), \
         patch("gpu_fuzzy_trader._gpu_runtime.detect_gpu_name", return_value=None), \
         patch("gpu_fuzzy_trader._gpu_runtime.detect_gpu_vram_gb", return_value=15.0):
        assert _gpu_runtime.is_t4_runtime() is True


def test_is_t4_runtime_env_override():
    _gpu_runtime.is_t4_runtime.cache_clear()
    with patch.dict("os.environ", {"GPU_OPT_T4": "1"}):
        assert _gpu_runtime.is_t4_runtime() is True

    _gpu_runtime.is_t4_runtime.cache_clear()
    with patch.dict("os.environ", {"GPU_OPT_T4": "0"}):
        assert _gpu_runtime.is_t4_runtime() is False


def test_is_t4_runtime_non_t4():
    _gpu_runtime.is_t4_runtime.cache_clear()
    _gpu_runtime.detect_gpu_name.cache_clear()
    _gpu_runtime.detect_gpu_vram_gb.cache_clear()
    with patch.dict("os.environ", {}, clear=True), \
         patch("gpu_fuzzy_trader._gpu_runtime.detect_gpu_name", return_value="NVIDIA RTX 4050 Laptop"), \
         patch("gpu_fuzzy_trader._gpu_runtime.detect_gpu_vram_gb", return_value=6.0):
        assert _gpu_runtime.is_t4_runtime() is False


def test_detect_hardware_profile():
    _gpu_runtime.detect_hardware_profile.cache_clear()
    _gpu_runtime.is_t4_runtime.cache_clear()
    _gpu_runtime.detect_gpu_name.cache_clear()
    _gpu_runtime.detect_gpu_vram_gb.cache_clear()
    _gpu_runtime.detect_system_ram_gb.cache_clear()
    with patch.dict("os.environ", {}, clear=True), \
         patch("gpu_fuzzy_trader._gpu_runtime.detect_gpu_name", return_value="Tesla T4"), \
         patch("gpu_fuzzy_trader._gpu_runtime.detect_gpu_vram_gb", return_value=15.0), \
         patch("gpu_fuzzy_trader._gpu_runtime.detect_system_ram_gb", return_value=12.7), \
         patch("os.cpu_count", return_value=2):
        prof = _gpu_runtime.detect_hardware_profile()
        assert prof.gpu_name == "Tesla T4"
        assert prof.vram_gb == 15.0
        assert prof.ram_gb == 12.7
        assert prof.cpu_count == 2
        assert prof.is_t4 is True


def test_backtest_batch_workers_capped():
    with patch("os.cpu_count", return_value=2):
        workers = min(8, os.cpu_count() or 4)
        assert workers <= 2
        assert workers <= (os.cpu_count() or 1)


def test_scripts_benchmark_t4_dry_run():
    from scripts import benchmark_t4
    report = benchmark_t4.collect_profile_report(skip_bench=True)
    assert "hardware" in report
    assert "resolved_config" in report
    assert "cpu_count" in report["hardware"]
    assert "vram_gb" in report["hardware"]
    assert "batch_size" in report["resolved_config"]


def test_research_profile_hardware_snapshot():
    hw = HardwareProfile(
        gpu_name="Tesla T4",
        vram_gb=15.0,
        ram_gb=12.7,
        cpu_count=2,
        jax_backend="gpu",
        devices=["cuda:0"],
        is_t4=True,
    )
    d = hw.as_dict()
    assert d["gpu_name"] == "Tesla T4"
    assert d["is_t4"] is True


def test_vram_batch_caps_all_tiers():
    from gpu_fuzzy_trader._gpu_runtime import _vram_batch_cap

    assert _vram_batch_cap(None, 256) == 64
    assert _vram_batch_cap(6.0, 256) == 64
    assert _vram_batch_cap(8.0, 256) == 64
    assert _vram_batch_cap(12.0, 256) == 256
    assert _vram_batch_cap(15.0, 256) == 256
    assert _vram_batch_cap(16.0, 256) == 256
    assert _vram_batch_cap(24.0, 256) == 256
    assert _vram_batch_cap(48.0, 512) == 512


def test_ram_batch_caps_all_tiers():
    from gpu_fuzzy_trader._gpu_runtime import _ram_batch_cap

    assert _ram_batch_cap(None) is None
    assert _ram_batch_cap(8.0) == 64
    assert _ram_batch_cap(12.0) == 64
    assert _ram_batch_cap(12.8, gpu_route_active=False) == 64
    assert _ram_batch_cap(12.8, gpu_route_active=True) == 128
    assert _ram_batch_cap(15.5) == 256
    assert _ram_batch_cap(32.0) is None


def test_hardware_defaults_applied_for_t4(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_cfg, "PHASE2_GPU_BATCH_SIZE_AUTO", False)
    monkeypatch.setattr(_cfg, "PHASE2_GPU_CPU_ROUTE_LARGE_DATA", True)
    monkeypatch.setattr(_cfg, "PHASE2_SCAN_UNROLL", 32)
    monkeypatch.setattr(_cfg, "is_t4_runtime", lambda: True)
    monkeypatch.setattr(_cfg, "is_colab_runtime", lambda: False)
    monkeypatch.delenv("GPU_OPT_DISABLE", raising=False)

    _cfg._apply_hardware_gpu_defaults()

    assert _cfg.PHASE2_GPU_BATCH_SIZE_AUTO is True
    assert _cfg.PHASE2_GPU_CPU_ROUTE_LARGE_DATA is False
    assert _cfg.PHASE2_SCAN_UNROLL == 16


def test_hardware_defaults_skipped_when_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_cfg, "PHASE2_GPU_BATCH_SIZE_AUTO", False)
    monkeypatch.setattr(_cfg, "PHASE2_GPU_CPU_ROUTE_LARGE_DATA", True)
    monkeypatch.setattr(_cfg, "PHASE2_SCAN_UNROLL", 32)
    monkeypatch.setattr(_cfg, "is_t4_runtime", lambda: True)
    monkeypatch.setenv("GPU_OPT_DISABLE", "1")

    _cfg._apply_hardware_gpu_defaults()

    assert _cfg.PHASE2_GPU_BATCH_SIZE_AUTO is False
    assert _cfg.PHASE2_GPU_CPU_ROUTE_LARGE_DATA is True
    assert _cfg.PHASE2_SCAN_UNROLL == 32


def test_configure_jax_env():
    from gpu_fuzzy_trader import _jax_env
    with patch.dict("os.environ", {}, clear=True):
        _jax_env.configure_jax_env()
        assert os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE") == "false"
        assert os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION") == "0.8"
        assert os.environ.get("JAX_PLATFORMS") == "cuda,cpu"
        assert os.environ.get("JAX_COMPILATION_CACHE_DIR") == "/tmp/jax_cache"
        assert os.path.isdir(os.environ.get("JAX_COMPILATION_CACHE_DIR"))

