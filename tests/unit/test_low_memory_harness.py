"""Regression tests for the shared low-memory Pytest harness."""

from __future__ import annotations

import os
import sys

from tests import conftest as test_config
from tests.property.hypothesis_config import prop_settings


class _FakeJax:
    def __init__(self) -> None:
        self.clear_calls = 0

    def clear_caches(self) -> None:
        self.clear_calls += 1


class _FakePyplot:
    def __init__(self) -> None:
        self.close_calls: list[str] = []

    def close(self, value: str) -> None:
        self.close_calls.append(value)


def test_numba_thread_cap_is_set_before_test_modules_execute() -> None:
    """Late JAX setup must not reconfigure an already-started Numba pool."""
    expected = str(max(1, min(2, os.cpu_count() or 1)))
    assert os.environ["NUMBA_NUM_THREADS"] == expected


def test_low_memory_property_settings_disable_timing_deadlines() -> None:
    """Resource-constrained runs must not fail correct properties on warm-up."""
    assert prop_settings().deadline is None


def test_optional_cleanup_does_not_import_unloaded_modules(monkeypatch) -> None:
    """CPU-only tests must not load JAX or Matplotlib during teardown."""
    monkeypatch.delitem(sys.modules, "jax", raising=False)
    monkeypatch.delitem(sys.modules, "matplotlib.pyplot", raising=False)

    test_config._clear_loaded_jax_caches()
    test_config._close_loaded_matplotlib_figures()

    assert "jax" not in sys.modules
    assert "matplotlib.pyplot" not in sys.modules


def test_optional_cleanup_releases_already_loaded_modules(monkeypatch) -> None:
    """A test that loaded an optional library still receives its cleanup."""
    jax = _FakeJax()
    pyplot = _FakePyplot()
    monkeypatch.setitem(sys.modules, "jax", jax)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)

    test_config._clear_loaded_jax_caches()
    test_config._close_loaded_matplotlib_figures()

    assert jax.clear_calls == 1
    assert pyplot.close_calls == ["all"]
