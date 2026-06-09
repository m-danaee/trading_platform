"""Unit tests for Phase 2 GPU runtime batch-size heuristics."""

from __future__ import annotations

import gpu_fuzzy_trader.config as _cfg
from gpu_fuzzy_trader._gpu_runtime import resolve_phase2_gpu_batch_size


def test_resolve_batch_size_env_override(monkeypatch) -> None:
    monkeypatch.setenv("PHASE2_GPU_BATCH_SIZE", "88")
    monkeypatch.delenv("PHASE2_GPU_BATCH_SIZE_AUTO", raising=False)
    assert resolve_phase2_gpu_batch_size() == 88


def test_resolve_batch_size_vram_tiers(monkeypatch) -> None:
    monkeypatch.delenv("PHASE2_GPU_BATCH_SIZE", raising=False)
    monkeypatch.setenv("PHASE2_GPU_BATCH_SIZE_AUTO", "true")
    monkeypatch.setattr(_cfg, "PHASE2_GPU_BATCH_SIZE", 64)

    from gpu_fuzzy_trader import _gpu_runtime as gr

    monkeypatch.setattr(gr, "detect_system_ram_gb", lambda: 32.0)

    monkeypatch.setattr(gr, "detect_gpu_vram_gb", lambda: 6.0)
    assert resolve_phase2_gpu_batch_size() == 16

    monkeypatch.setattr(gr, "detect_gpu_vram_gb", lambda: 11.0)
    assert resolve_phase2_gpu_batch_size() == 32

    monkeypatch.setattr(gr, "detect_gpu_vram_gb", lambda: 15.0)
    assert resolve_phase2_gpu_batch_size() == 64

    monkeypatch.setattr(_cfg, "PHASE2_GPU_BATCH_SIZE", 128)
    monkeypatch.setattr(gr, "detect_gpu_vram_gb", lambda: 23.0)
    assert resolve_phase2_gpu_batch_size() == 96


def test_resolve_batch_size_config_auto_disabled(monkeypatch) -> None:
    monkeypatch.delenv("PHASE2_GPU_BATCH_SIZE", raising=False)
    monkeypatch.delenv("PHASE2_GPU_BATCH_SIZE_AUTO", raising=False)
    monkeypatch.setattr(_cfg, "PHASE2_GPU_BATCH_SIZE", 128)
    monkeypatch.setattr(_cfg, "PHASE2_GPU_BATCH_SIZE_AUTO", False)

    from gpu_fuzzy_trader import _gpu_runtime as gr

    monkeypatch.setattr(gr, "detect_gpu_vram_gb", lambda: 15.0)
    monkeypatch.setattr(gr, "detect_system_ram_gb", lambda: 32.0)
    assert resolve_phase2_gpu_batch_size() == 128


def test_resolve_batch_size_respects_config_cap(monkeypatch) -> None:
    monkeypatch.delenv("PHASE2_GPU_BATCH_SIZE", raising=False)
    monkeypatch.setenv("PHASE2_GPU_BATCH_SIZE_AUTO", "true")
    monkeypatch.setattr(_cfg, "PHASE2_GPU_BATCH_SIZE", 48)

    from gpu_fuzzy_trader import _gpu_runtime as gr

    monkeypatch.setattr(gr, "detect_gpu_vram_gb", lambda: 15.0)
    monkeypatch.setattr(gr, "detect_system_ram_gb", lambda: 32.0)
    assert resolve_phase2_gpu_batch_size() == 48


def test_resolve_batch_size_low_ram_colab_cap(monkeypatch) -> None:
    monkeypatch.delenv("PHASE2_GPU_BATCH_SIZE", raising=False)
    monkeypatch.setenv("PHASE2_GPU_BATCH_SIZE_AUTO", "true")
    monkeypatch.setattr(_cfg, "PHASE2_GPU_BATCH_SIZE", 198)

    from gpu_fuzzy_trader import _gpu_runtime as gr

    monkeypatch.setattr(gr, "detect_gpu_vram_gb", lambda: 15.0)
    monkeypatch.setattr(gr, "detect_system_ram_gb", lambda: 12.7)
    assert resolve_phase2_gpu_batch_size() == 32


def test_resolve_batch_size_unknown_vram_uses_conservative_cap(monkeypatch) -> None:
    monkeypatch.delenv("PHASE2_GPU_BATCH_SIZE", raising=False)
    monkeypatch.setenv("PHASE2_GPU_BATCH_SIZE_AUTO", "true")
    monkeypatch.setattr(_cfg, "PHASE2_GPU_BATCH_SIZE", 198)

    from gpu_fuzzy_trader import _gpu_runtime as gr

    monkeypatch.setattr(gr, "detect_gpu_vram_gb", lambda: None)
    monkeypatch.setattr(gr, "detect_system_ram_gb", lambda: 32.0)
    assert resolve_phase2_gpu_batch_size() == 32
