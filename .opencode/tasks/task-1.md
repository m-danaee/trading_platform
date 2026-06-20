# Task 1: Add AttributeError to jax_compat error handling

**File:** `gpu_fuzzy_trader/backtest/jax_compat.py`

## Change

Add `AttributeError` to the `_GPU_ENGINE_ERRORS` tuple (line 16).

**Current:**
```python
_GPU_ENGINE_ERRORS = (ImportError, RuntimeError, OSError)
```

**Target:**
```python
_GPU_ENGINE_ERRORS = (ImportError, RuntimeError, OSError, AttributeError)
```

## Why

When `gpu_engine.py` does `import jax` and jax itself fails with an `AttributeError` during initialization (e.g., `partially initialized module 'jax' has no attribute 'version'`), this exception is not caught because `jax_compat.py` only catches `ImportError, RuntimeError, OSError`. The uncaught exception propagates up and crashes Phase 2 entirely, causing every Optuna trial to return a sentinel score of -1275.

Adding `AttributeError` ensures that ALL jax import failures result in `get_gpu_backtest_engine_class()` returning `None`, allowing the pipeline to gracefully fall back to `CPUBacktestEngine`.

## Acceptance Criteria

1. `_GPU_ENGINE_ERRORS` includes `AttributeError`
2. When jax fails to import for any reason (including `AttributeError`), `get_gpu_backtest_engine_class()` returns `None` instead of crashing
3. No other logic changes in the file
4. Verify: the import chain `from gpu_fuzzy_trader.backtest.jax_compat import jax_gpu_backtest_available` works without jax installed
