"""Colab GPU default overrides and symbol-metric enrichment gating."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpu_fuzzy_trader import config as cfg
from gpu_fuzzy_trader._gpu_runtime import _ram_batch_cap, _vram_batch_cap


def test_main_notebook_pins_triton_for_torch_26() -> None:
    notebook_path = Path(__file__).resolve().parents[2] / "main.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    dependency_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and "COLAB_TORCH_CPU" in "".join(cell.get("source", []))
    )

    assert 'COLAB_TRITON = "triton==3.2.0"' in dependency_source
    assert (
        'subprocess.check_call(PIP + ["-U", COLAB_TRITON])'
        in dependency_source
    )
    assert dependency_source.index("COLAB_TORCH_CPU") < dependency_source.index(
        "COLAB_TRITON"
    )


def test_phase2_should_enrich_without_symbol_scope() -> None:
    assert cfg.phase2_should_enrich_symbol_metrics(None) is (
        cfg.PHASE2_GPU_ENRICH_SYMBOL_METRICS
    )


def test_colab_defaults_apply_when_content_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_route = cfg.PHASE2_GPU_CPU_ROUTE_LARGE_DATA
    original_unroll = cfg.PHASE2_SCAN_UNROLL
    monkeypatch.setattr(cfg, "PHASE2_GPU_BATCH_SIZE_AUTO", False)
    monkeypatch.setattr(cfg, "PHASE2_GPU_CPU_ROUTE_LARGE_DATA", True)
    monkeypatch.setattr(cfg, "PHASE2_SCAN_UNROLL", 32)
    monkeypatch.setattr(cfg.os.path, "isdir", lambda path: path == "/content")
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)

    cfg._apply_colab_gpu_defaults()

    assert cfg.PHASE2_GPU_BATCH_SIZE_AUTO is True
    assert cfg.PHASE2_GPU_CPU_ROUTE_LARGE_DATA is False
    assert cfg.PHASE2_SCAN_UNROLL == 16

    # The helper mutates module defaults by design; restore the local-test
    # process so later tests still exercise the desktop policy.
    monkeypatch.setattr(cfg, "PHASE2_GPU_CPU_ROUTE_LARGE_DATA", original_route)
    monkeypatch.setattr(cfg, "PHASE2_SCAN_UNROLL", original_unroll)


def test_3080ti_batch_tiers_use_full_population_chunk() -> None:
    """A 12-GiB desktop GPU with ~16-GiB RAM should avoid tiny chunks."""
    assert _vram_batch_cap(12.0, 256) == 256
    assert _ram_batch_cap(15.6) == 256
    assert _ram_batch_cap(13.0) == 64
