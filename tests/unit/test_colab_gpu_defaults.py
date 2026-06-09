"""Colab GPU default overrides and symbol-metric enrichment gating."""

from __future__ import annotations

import pytest

from gpu_fuzzy_trader import config as cfg


def test_phase2_should_enrich_without_symbol_scope() -> None:
    assert cfg.phase2_should_enrich_symbol_metrics(None) is (
        cfg.PHASE2_GPU_ENRICH_SYMBOL_METRICS
    )


class _ScopedEngine:
    _symbol_scope = "1"


def test_phase2_should_skip_enrich_for_symbol_scope() -> None:
    assert cfg.phase2_should_enrich_symbol_metrics(_ScopedEngine()) is False


def test_colab_defaults_apply_when_content_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cfg, "PHASE2_CV_FOLD_WORKERS", 1)
    monkeypatch.setattr(cfg, "PHASE3_USE_GPU", False)
    monkeypatch.setattr(cfg, "PHASE2_GPU_BATCH_SIZE_AUTO", False)
    monkeypatch.setattr(cfg.os.path, "isdir", lambda path: path == "/content")
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)

    cfg._apply_colab_gpu_defaults()

    assert cfg.PHASE2_CV_FOLD_WORKERS == 1
    assert cfg.PHASE3_USE_GPU is True
    assert cfg.PHASE2_GPU_BATCH_SIZE_AUTO is True
