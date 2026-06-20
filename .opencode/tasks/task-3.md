# Task 3: Suppress JAX import crash in gpu_engine.py module level

**File:** `gpu_fuzzy_trader/backtest/gpu_engine.py`

## Change

Replace the module-level `try/except ImportError` block (lines 44-65) to catch `Exception` and set JAX globals to `None` instead of re-raising.

**Current (lines 44-65):**
```python
configure_jax_env()

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap
    import jax.lax as lax

    def _resolve_jax_float_dtype():
        use_fp32 = bool(getattr(_cfg, "PHASE2_GPU_USE_FP32", True))
        if use_fp32:
            jax.config.update("jax_enable_x64", False)
            return jnp.float32
        jax.config.update("jax_enable_x64", True)
        return jnp.float64

    _JXF = _resolve_jax_float_dtype()
    _JX_INT = jnp.int8 if bool(
        getattr(_cfg, "PHASE2_GPU_DATA_INT8", True)) else jnp.int32
except ImportError as _jax_err:
    raise ImportError(
        "JAX is required for GPUBacktestEngine but could not be imported. "
        "Install it with: pip install jax jaxlib\n"
        f"Original error: {_jax_err}"
    ) from _jax_err
```

**Target:**
```python
configure_jax_env()

_jax_import_error = None

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap
    import jax.lax as lax

    def _resolve_jax_float_dtype():
        use_fp32 = bool(getattr(_cfg, "PHASE2_GPU_USE_FP32", True))
        if use_fp32:
            jax.config.update("jax_enable_x64", False)
            return jnp.float32
        jax.config.update("jax_enable_x64", True)
        return jnp.float64

    _JXF = _resolve_jax_float_dtype()
    _JX_INT = jnp.int8 if bool(
        getattr(_cfg, "PHASE2_GPU_DATA_INT8", True)) else jnp.int32
except Exception as _jax_err:
    jax = jnp = jit = vmap = lax = None
    _JXF = _JX_INT = None
    _jax_import_error = str(_jax_err)
```

Then add a guard function:
```python
def _require_jax():
    """Raise RuntimeError if JAX failed to import at module level."""
    if jax is None:
        raise RuntimeError(
            "JAX could not be imported (required for GPUBacktestEngine). "
            f"Original error: {_jax_import_error}"
        )
```

And call `_require_jax()` at the top of `GPUBacktestEngine.__init__()` (or `simulate_rule_batch`, whichever is the primary entry point).

## Why

Currently, `gpu_engine.py` re-raises `ImportError` when JAX fails — which propagates through `jax_compat.py`'s try/except (since `ImportError` IS caught). BUT the problem is that `jax_compat.py` catches `_GPU_ENGINE_ERRORS` which was `(ImportError, RuntimeError, OSError)` BEFORE task-1. The original error (`AttributeError`) was NOT in that tuple.

After task-1, `_GPU_ENGINE_ERRORS` includes `AttributeError` too. But `gpu_engine.py` catches only `ImportError` and re-raises it. If JAX fails with `AttributeError`, it's NOT caught by `gpu_engine.py`'s try/except (which only catches `ImportError`). The `AttributeError` would propagate to `jax_compat.py` which NOW catches it (after task-1).

So actually, after task-1, the pipeline SHOULD work! But task-3 provides additional defense-in-depth by:
1. Catching ALL exception types (not just `ImportError`) at gpu_engine.py's module level
2. Providing a clear error message when GPU functions are actually called (lazy error, not import-time crash)
3. Allowing `gpu_engine.py` to be imported safely even when JAX is completely broken

## Acceptance Criteria

1. Module-level `try/except ImportError` replaced with `try/except Exception`
2. JAX globals set to `None` on failure instead of re-raising
3. `_require_jax()` guard function added, called at `GPUBacktestEngine.__init__()` entry
4. When JAX is available: all existing behavior preserved
5. When JAX is broken: importing gpu_engine.py does not crash; using it raises clear RuntimeError
6. All existing tests pass: `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_gpu_engine.py -v`
