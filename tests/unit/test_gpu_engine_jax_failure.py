"""
Tests for the lazy-JAX failure path in gpu_engine.py.

These tests verify that importing gpu_engine.py does NOT crash when JAX
fails to import, and that using GPUBacktestEngine raises a clear RuntimeError.

We use a subprocess because JAX is already installed (and cached in sys.modules)
in the parent process, so monkeypatching builtins.__import__ in-process would
not trigger the import path inside gpu_engine.py.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


def _subprocess_script(err_type: str = "RuntimeError", err_msg: str = "Simulated JAX crash") -> str:
    """Build a subprocess script that simulates JAX import failure."""
    return textwrap.dedent(f'''\
    import builtins
    import sys

    _real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "jax" or (isinstance(name, str) and name.startswith("jax.")):
            raise {err_type}("{err_msg}")
        return _real_import(name, *args, **kwargs)

    builtins.__import__ = _fake_import

    # ---- Test (a): importing gpu_engine must not raise ----
    try:
        import gpu_fuzzy_trader.backtest.gpu_engine as ge
    except Exception as exc:
        print(f"FAIL: import raised {{type(exc).__name__}}: {{exc}}")
        sys.exit(1)

    # ---- Verify module-level globals are None ----
    assert ge.jax is None, "jax should be None"
    assert ge.jnp is None, "jnp should be None"
    assert ge.jit is None, "jit should be None"
    assert ge.vmap is None, "vmap should be None"
    assert ge.lax is None, "lax should be None"
    assert ge._JXF is None, "_JXF should be None"
    assert ge._JX_INT is None, "_JX_INT should be None"
    assert ge._jax_import_error is not None, "_jax_import_error should be set"
    assert "{err_msg}" in str(ge._jax_import_error), (
        f"_jax_import_error should contain original error message"
    )

    # ---- Test (b): _require_jax must raise RuntimeError ----
    try:
        ge._require_jax()
        print("FAIL: _require_jax() did not raise")
        sys.exit(1)
    except RuntimeError as e:
        assert "{err_msg}" in str(e), (
            f"RuntimeError message should contain original error: {{e}}"
        )
        # Exception chain must be preserved via raise ... from
        assert e.__cause__ is not None, (
            "RuntimeError should have __cause__ set (raise ... from)"
        )
        assert "{err_msg}" in str(e.__cause__), (
            f"__cause__ should contain original error: {{e.__cause__}}"
        )

    # ---- Test (c): instantiating GPUBacktestEngine must raise RuntimeError ----
    import pandas as pd
    df = pd.DataFrame({{
        "symbol": ["SYM"] * 5,
        "datetime": pd.date_range("2024-01-01", periods=5, freq="5min"),
        "_symbol_bar_index": list(range(5)),
        "label_open_next": [100.0] * 5,
        "label_max_288": [105.0] * 5,
        "label_min_288": [97.0] * 5,
        "label_close_288": [102.0] * 5,
        "label_max_before_min": [1] * 5,
    }})
    try:
        engine = ge.GPUBacktestEngine(df, feature_modes={{}}, direction="long")
        print("FAIL: GPUBacktestEngine instantiation did not raise")
        sys.exit(1)
    except RuntimeError as e:
        assert "{err_msg}" in str(e), (
            f"RuntimeError message should contain original error: {{e}}"
        )
        assert e.__cause__ is not None, (
            "RuntimeError should have __cause__ set (raise ... from)"
        )

    print("ALL CHECKS PASSED")
    ''')


@pytest.mark.parametrize(
    ("err_type", "err_msg"),
    [
        ("RuntimeError", "jaxlib was built with AVX but CPU does not support it"),
        ("AttributeError", "module 'jax' has no attribute 'numpy'"),
        ("ImportError", "No module named 'jax'"),
    ],
)
def test_gpu_engine_import_does_not_crash_on_jax_failure(err_type, err_msg):
    """Verify importing gpu_engine.py handles various JAX failure modes gracefully."""
    code = _subprocess_script(err_type=err_type, err_msg=err_msg)
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=60,
        env={"NO_JAX_RECURSIVE_IMPORT": "1", "PYTHONWARNINGS": "ignore"},
    )
    # Print stdout/stderr for debugging on failure
    if result.returncode != 0:
        print("--- STDOUT ---")
        print(result.stdout)
        print("--- STDERR ---")
        print(result.stderr)

    assert result.returncode == 0, (
        f"Subprocess exited with code {result.returncode}.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "ALL CHECKS PASSED" in result.stdout
