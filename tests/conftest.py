"""Shared pytest configuration: low-memory defaults and JAX allocator settings."""

from __future__ import annotations

import gc
import os

import pytest

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


@pytest.fixture(autouse=True)
def _low_memory_cleanup(request: pytest.FixtureRequest):
    yield
    if not _LOW_MEMORY:
        return
    gc.collect()
    try:
        import jax

        jax.clear_caches()
    except Exception:
        pass
