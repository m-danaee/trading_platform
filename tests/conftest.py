"""Shared pytest configuration: low-memory defaults and JAX allocator settings."""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

# Keep Numba's pool aligned with the later JAX runtime setup.  This must happen
# before test collection imports a Numba-backed module; changing the value after
# its pool starts raises RuntimeError and leaves low-memory runs non-repeatable.
_THREAD_CAP = str(max(1, min(2, os.cpu_count() or 1)))
for _thread_env in (
    "NUMBA_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_env, _THREAD_CAP)

# Set before any test module imports JAX (pytest loads conftest first).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

_LOW_MEMORY = os.environ.get("PYTEST_LOW_MEMORY", "").strip().lower() in (
    "1",
    "true",
    "yes",
)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "benchmark: throughput / warmup benchmarks (skipped unless RUN_BENCHMARKS=1)",
    )
    config.addinivalue_line(
        "markers",
        "uses_jax: test imports JAX; extra cache cleanup when PYTEST_LOW_MEMORY=1",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if os.environ.get("RUN_BENCHMARKS", "").strip().lower() in ("1", "true", "yes"):
        return
    skip_benchmark = pytest.mark.skip(
        reason="benchmark tests skipped (set RUN_BENCHMARKS=1 to run)",
    )
    for item in items:
        if "benchmark" in item.keywords:
            item.add_marker(skip_benchmark)


def _clear_loaded_jax_caches() -> None:
    """Release JAX caches without loading JAX for a CPU-only test."""
    jax = sys.modules.get("jax")
    if jax is None:
        return
    try:
        jax.clear_caches()
    except Exception:
        pass


def _close_loaded_matplotlib_figures() -> None:
    """Close figures without importing Matplotlib for tests that never used it."""
    pyplot = sys.modules.get("matplotlib.pyplot")
    if pyplot is None:
        return
    try:
        pyplot.close("all")
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _isolate_split_cache(tmp_path_factory, monkeypatch):
    """Keep splitter cache writes out of the repository tree."""
    cache_dir = tmp_path_factory.mktemp("split_cache")
    train = str(cache_dir / "development_train.parquet")
    validation = str(cache_dir / "validation.parquet")
    fitness = str(cache_dir / "validation_fitness.parquet")
    selection = str(cache_dir / "validation_selection.parquet")
    manifest = str(cache_dir / "split_manifest.json")
    import gpu_fuzzy_trader.config as config_mod
    import gpu_fuzzy_trader.data.splitter as splitter_mod

    monkeypatch.setattr(config_mod, "DEVELOPMENT_TRAIN_PATH", train)
    monkeypatch.setattr(config_mod, "TRAIN_70_PATH", train)
    monkeypatch.setattr(config_mod, "VALIDATION_PATH", validation)
    monkeypatch.setattr(config_mod, "VALIDATION_30_PATH", validation)
    monkeypatch.setattr(config_mod, "VALIDATION_FITNESS_PATH", fitness)
    monkeypatch.setattr(config_mod, "VALIDATION_SELECTION_PATH", selection)
    monkeypatch.setattr(config_mod, "SPLIT_MANIFEST_PATH", manifest)
    monkeypatch.setattr(splitter_mod, "TRAIN_70_PATH", train)
    monkeypatch.setattr(splitter_mod, "VALIDATION_30_PATH", validation)
    monkeypatch.setattr(splitter_mod, "VALIDATION_FITNESS_PATH", fitness)
    monkeypatch.setattr(splitter_mod, "VALIDATION_SELECTION_PATH", selection)


@pytest.fixture(autouse=True)
def _low_memory_cleanup():
    yield
    if not _LOW_MEMORY:
        return
    gc.collect()
    _clear_loaded_jax_caches()


@pytest.fixture(autouse=True)
def _close_matplotlib_figures():
    """Close all open matplotlib figures after each test.

    Prevents figure objects from accumulating in memory across tests in
    the reporter test suite, which is the dominant source of matplotlib
    overhead under low-memory runs.
    """
    yield
    _close_loaded_matplotlib_figures()
