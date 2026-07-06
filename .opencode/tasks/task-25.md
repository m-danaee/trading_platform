# Task 25 — RAM low-cost knobs (A1 + A2 + A3)

## Branch
`fix/ram-knobs-final` (from `main`)

## Problem
The 2026-07-06 re-run shows system RAM climbing to 9.6/12.7 GB at gen 9
(inside cluster_0's first epoch, before any between-cluster cleanup
has fired). The Task 4/5 fixes only address between-cluster steady-state;
they do not help within a cluster's run.

## Required changes

### A1. Halve global metrics cache
File: `gpu_fuzzy_trader/config.py` line 414

Change:
```python
PHASE2_EVAL_GLOBAL_CACHE_MAX_SIZE = 600
```
to:
```python
PHASE2_EVAL_GLOBAL_CACHE_MAX_SIZE = 200
```

Update the docstring above the line so the explanation matches the new
value (the current comment says "600 = 200 (population) × 2 (eval runs)
× 1.5 (cache generations)" and the rationale text mentions being
"Halved from 1200"). Keep the rationale text — just change "1200 to
600" to "1200 → 600 → 200" and update the math comment to reflect the
new ratio.

### A2. Shorten island epoch generations
File: `gpu_fuzzy_trader/config.py` line 1104

Change:
```python
PHASE2_ISLAND_EPOCH_GENERATIONS = 25
```
to:
```python
PHASE2_ISLAND_EPOCH_GENERATIONS = 13
```

Update the docstring above the line. The current comment says "Increased
15→25: fewer epoch rebuilds (~40% overhead reduction); 15-gen epochs
caused 10+ epoch starts with ~15s engine rebuild overhead each." Add a
new line explaining the 2026-07-06 reversal: lowered 25→13 so
`trim_evolution_state_memory` (which fires once per epoch via
`park_engines()`) runs more often within a cluster's 44-gen lifetime,
capping in-cluster RAM growth. Total budget (44 gens × 3 clusters)
unchanged.

### A3. malloc_trim after gc.collect
File: `gpu_fuzzy_trader/evolution/evox_runner.py` line 2793

Current code:
```python
        if gen % 3 == 0 and gen > 0:
            import gc as _gc
            _gc.collect()
```

New code (add `malloc_trim(0)` wrapped in a try/except for portability):
```python
        if gen % 3 == 0 and gen > 0:
            import gc as _gc
            _gc.collect()
            # Return freed glibc arena memory to the OS. Colab (Linux/glibc)
            # is the primary host; the OSError guard makes this a no-op on
            # non-glibc systems (macOS, musl).
            try:
                import ctypes
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except OSError:
                pass
```

## Acceptance criteria
1. `PHASE2_EVAL_GLOBAL_CACHE_MAX_SIZE = 200` at `config.py:414`
2. `PHASE2_ISLAND_EPOCH_GENERATIONS = 13` at `config.py:1104`
3. `malloc_trim(0)` call after `gc.collect()` at `evox_runner.py:2793`
   with `try/except OSError` guard
4. Docstring comments updated to reflect the new values
5. No other config values changed
6. `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_rule_pool.py tests/unit/test_phase2_island_scheduler.py tests/unit/test_evox_runner.py tests/unit/test_config_trade_scaling.py -v` passes
7. No regressions in the previous RAM Task 4/5 fixes (cache cleanup,
   sequential warmup still present and working)

## Out of scope
- Do NOT change `PHASE2_POPULATION_SIZE` (line 1061) — last-resort lever only
- Do NOT change `PHASE2_GPU_BATCH_SIZE` — already capped at 32 on Colab
  via `_ram_batch_cap` in `_gpu_runtime.py`
- Do NOT change the `gc.collect()` cadence from 3 — keep as-is
- Do NOT touch any of the per-cluster teardown code from Task 4/5
