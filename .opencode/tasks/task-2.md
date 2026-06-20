# Task 2: Make backtest/__init__.py robust to GPU engine import failure

**File:** `gpu_fuzzy_trader/backtest/__init__.py`

## Change

Wrap the eager `GPUBacktestEngine = get_gpu_backtest_engine_class()` call (line 14) in a try/except block so the subpackage is importable even when jax is broken.

**Current:**
```python
GPUBacktestEngine = get_gpu_backtest_engine_class()
if GPUBacktestEngine is not None:
    __all__.append("GPUBacktestEngine")
```

**Target:**
```python
try:
    GPUBacktestEngine = get_gpu_backtest_engine_class()
except Exception:
    from gpu_fuzzy_trader.backtest.jax_compat import _GPU_ENGINE_ERRORS
    GPUBacktestEngine = None
if GPUBacktestEngine is not None:
    __all__.append("GPUBacktestEngine")
```

Alternative simpler approach: just wrap in try/except:
```python
try:
    GPUBacktestEngine = get_gpu_backtest_engine_class()
except Exception:
    GPUBacktestEngine = None
if GPUBacktestEngine is not None:
    __all__.append("GPUBacktestEngine")
```

## Why

`backtest/__init__.py` calls `get_gpu_backtest_engine_class()` at module level. Since `jax_compat.py`'s `get_gpu_backtest_engine_class()` already catches `_GPU_ENGINE_ERRORS` internally and returns `None`, this call should be safe after task-1. However, as an additional defense-in-depth measure, wrapping this call ensures that even unexpected exceptions during GPU engine detection won't prevent importing the `backtest` subpackage (which is needed for `CPUBacktestEngine`).

## Acceptance Criteria

1. `from gpu_fuzzy_trader.backtest import CPUBacktestEngine` works regardless of jax availability
2. `GPUBacktestEngine` is `None` when jax is unavailable (current behavior, but without crashing the import)
3. `GPUBacktestEngine` is the real class when jax is available (existing behavior preserved)
4. `__all__` still only includes `"GPUBacktestEngine"` when it's available
