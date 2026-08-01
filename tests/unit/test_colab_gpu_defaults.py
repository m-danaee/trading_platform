"""Colab GPU default overrides and symbol-metric enrichment gating."""

from __future__ import annotations

import pytest

from gpu_fuzzy_trader import config as cfg
from gpu_fuzzy_trader._gpu_runtime import _ram_batch_cap, _vram_batch_cap


def test_phase2_should_enrich_without_symbol_scope() -> None:
    assert cfg.phase2_should_enrich_symbol_metrics(None) is (
        cfg.PHASE2_GPU_ENRICH_SYMBOL_METRICS
    )


def test_colab_defaults_apply_when_content_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cfg, "PHASE2_GPU_BATCH_SIZE_AUTO", False)
    monkeypatch.setattr(cfg.os.path, "isdir", lambda path: path == "/content")
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)

    cfg._apply_colab_gpu_defaults()

    assert cfg.PHASE2_GPU_BATCH_SIZE_AUTO is True


def test_3080ti_batch_tiers_use_full_population_chunk() -> None:
    """A 12-GiB desktop GPU with ~16-GiB RAM should avoid tiny chunks."""
    assert _vram_batch_cap(12.0, 256) == 256
    assert _ram_batch_cap(15.6) == 256
    assert _ram_batch_cap(13.0) == 64
