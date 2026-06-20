# Plan: Fix JAX import crash blocking Optuna Phase 2

## Goal

Fix the `AttributeError: partially initialized module 'jax' has no attribute 'version'` crash that causes every Optuna trial to fail at Phase 2, wasting all 35 trials (~2.5 hours).

## Root Cause

`jax_compat.py` catches only `ImportError, RuntimeError, OSError` but JAX fails with **`AttributeError`**. When `gpu_engine.py` does `import jax` at module level, the uncaught exception propagates → Phase 2 returns no rules → sentinel score -1275 for every trial.

### Secondary issue
`backtest/__init__.py` eagerly calls `get_gpu_backtest_engine_class()` at import time, making the entire backtest subpackage depend on jax availability.

## Approach

Three targeted changes, all defensive-only (no logic changes):

1. **`jax_compat.py`** — Add `AttributeError` to the caught exception tuple
2. **`backtest/__init__.py`** — Wrap eager engine detection in try/except
3. **`gpu_engine.py`** — Add lazy-import helper so jax isn't imported at module level

## Tasks

### task-1: Add AttributeError to jax_compat error handling

**File:** `gpu_fuzzy_trader/backtest/jax_compat.py`

**Change:** Add `AttributeError` to `_GPU_ENGINE_ERRORS` tuple.

**Acceptance criteria:**
- `_GPU_ENGINE_ERRORS` includes `AttributeError`
- When jax fails to import (any reason), `get_gpu_backtest_engine_class()` returns `None` instead of crashing
- Pipeline falls back to CPUBacktestEngine gracefully

### task-2: Make backtest/__init__.py robust to GPU engine import failure

**File:** `gpu_fuzzy_trader/backtest/__init__.py`

**Change:** Wrap `GPUBacktestEngine = get_gpu_backtest_engine_class()` in try/except so the subpackage is importable even when jax is broken.

**Acceptance criteria:**
- `from gpu_fuzzy_trader.backtest import CPUBacktestEngine` works regardless of jax availability
- `GPUBacktestEngine` is `None` when jax is unavailable (current behavior, but without crashing)

### task-3: Lazy-import jax in gpu_engine.py

**File:** `gpu_fuzzy_trader/backtest/gpu_engine.py`

**Change:** Replace module-level `import jax` + `import jax.numpy as jnp` etc. with a lazy `_ensure_jax()` helper that imports and caches on first use. All functions that use jax/jnp/lax/vmap call `_ensure_jax()` first.

**Acceptance criteria:**
- Module-level `import jax` removed from `gpu_engine.py`
- All existing jax-using functions work identically (first call triggers import)
- If jax is unavailable, a clear `RuntimeError` is raised explaining the fallback
- `gpu_engine.py` can be imported without jax installed

## Verification

```bash
cd /home/danaee/trading_platform && \
  .venv/bin/python -c "
from gpu_fuzzy_trader.backtest.jax_compat import jax_gpu_backtest_available
print('jax available:', jax_gpu_backtest_available())
from gpu_fuzzy_trader.backtest import CPUBacktestEngine
print('CPU engine importable:', CPUBacktestEngine is not None)
"
```

## Non-goals

- Fixing the jax package itself on the remote machine
- Modifying Optuna or pipeline logic
- Adding GPU fallback retries
