# Task 1: GPU Runtime Support

## Goal
Add GPU runtime infrastructure to friend_project so Phase 2 can use JAX GPU backtesting, matching the main project's GPU readiness.

## Target Files
1. **`gpu_fuzzy_trader/_gpu_runtime.py`** (NEW) — Copy+adapt from main project's `/home/danaee/trading_platform/gpu_fuzzy_trader/_gpu_runtime.py`
2. **`gpu_fuzzy_trader/config.py`** — Add GPU config knobs
3. **`gpu_fuzzy_trader/_jax_env.py`** — Add Colab cache path
4. **`gpu_fuzzy_trader/run_pipeline.py`** — Wire GPU runtime into Phase 2
5. **`gpu_fuzzy_trader/phases/phase2_rule_pool.py`** — Use resolved batch size

## Detailed Spec

### 1. `_gpu_runtime.py` (NEW)
Copy from `/home/danaee/trading_platform/gpu_fuzzy_trader/_gpu_runtime.py` and adapt:
- Remove all references to `phase2_sparse_encoding` (does not exist in friend_project)
  - Remove: `from gpu_fuzzy_trader.phases.phase2_sparse_encoding import use_sparse_slots`
  - Remove: `from gpu_fuzzy_trader.phases.phase2_sparse_encoding import empty_slots, use_sparse_slots`
  - In `_warmup_signature()`: remove sparse encoding branch, keep only dense path
  - In `_warmup_engine()`: remove sparse encoding branch (`use_sparse_slots()`, `empty_slots()`), keep only dense chromosome path
- Keep: `detect_gpu_vram_gb()`, `detect_gpu_memory_used_gb()`, `detect_system_ram_gb()`, `resolve_phase2_gpu_batch_size()`, `log_gpu_runtime_config()`, `warmup_phase2_gpu_kernels()`, `configure_phase2_gpu_runtime()`, `_warmup_signature()`, `_warmup_engine()`, `_iter_warmup_targets()`
- The dense path in `_warmup_engine()` should use `int(target._data_matrix_jax.shape[1])` for k, `int(np.asarray(target._dont_cares_jax)[0])` for dc
- Import `os`, `logging`, `subprocess`, `numpy as np` from local scope where needed

### 2. `config.py` additions
Add these GPU knobs (copy values from main project's config.py):
```python
PHASE2_USE_GPU = True
PHASE2_GPU_BATCH_SIZE = 198
PHASE2_GPU_BATCH_SIZE_AUTO = True
PHASE2_SCAN_UNROLL = 32
PHASE2_GPU_USE_FP32 = True
PHASE2_GPU_DATA_INT8 = True
```

Add `is_colab_runtime()` function and `_apply_colab_gpu_defaults()`:
```python
def is_colab_runtime() -> bool:
    """Return True when running inside a Colab environment."""
    try:
        import importlib.util
        if importlib.util.find_spec("google.colab") is not None:
            return True
    except Exception:
        pass
    return os.environ.get("COLAB_RELEASE_TAG") is not None

def _apply_colab_gpu_defaults() -> None:
    """Override GPU knobs when running on Colab."""
    global PHASE3_USE_GPU, PHASE2_GPU_BATCH_SIZE_AUTO
    if not is_colab_runtime():
        return
    PHASE2_GPU_BATCH_SIZE_AUTO = True

_apply_colab_gpu_defaults()
```

### 3. `_jax_env.py` update
Update `configure_jax_env()` to add Colab cache path. Currently the friend_project's version is minimal. Update to:
- Add `JAX_COMPILATION_CACHE_DIR` to `/content/jax_cache` when running on Colab (`/content` exists)
- Set `JAX_ENABLE_X64 = False`
- Set `TF_CPP_MIN_LOG_LEVEL = "3"`, `ABSL_MIN_LOGLEVEL = "3"`
- Keep existing `XLA_PYTHON_CLIENT_PREALLOCATE`, `XLA_PYTHON_CLIENT_MEM_FRACTION`, `JAX_PLATFORMS`

### 4. `run_pipeline.py` — wire GPU runtime
- After Phase 2 engine is created (when `PHASE2_USE_GPU=True`), call `configure_phase2_gpu_runtime(engine)` to warm up JAX kernels
- Add import: `from gpu_fuzzy_trader._gpu_runtime import configure_phase2_gpu_runtime, resolve_phase2_gpu_batch_size`
- Find where Phase 2 engine is initialized in `_run_phase2()`, add warmup call after GPU engine creation

### 5. `phase2_rule_pool.py` — batch size resolution
- Import `resolve_phase2_gpu_batch_size` from `_gpu_runtime`
- Where `PHASE2_GPU_BATCH_SIZE` is used to configure the GPU engine's batch size, replace with `resolve_phase2_gpu_batch_size()`
- If the friend_project's `phase2_rule_pool.py` doesn't have explicit GPU batch size config, ensure the `GPUBacktestEngine` or evolution runner receives the resolved batch size

## Acceptance Criteria
- [ ] `_gpu_runtime.py` exists and imports without errors (no sparse_encoding refs)
- [ ] `config.py` has all 6 GPU knobs with correct defaults
- [ ] `is_colab_runtime()` returns True on Colab, False locally
- [ ] `_apply_colab_gpu_defaults()` autodetects and configures
- [ ] `_jax_env.py` has Colab JAX cache path
- [ ] `run_pipeline.py` calls `configure_phase2_gpu_runtime()` for GPU Phase 2
- [ ] `phase2_rule_pool.py` uses `resolve_phase2_gpu_batch_size()`
- [ ] No import errors from sparse_encoding references
- [ ] Existing tests still pass (or skip if GPU-dependent)

## Dependencies
None — this is the first task.

## Handoff
Write `.opencode/handoffs/task-1-implementer.json` on completion.
