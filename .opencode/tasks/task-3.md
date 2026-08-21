# Task 3: 2-core CPU contention mitigation (worker/thread pinning)
> id: task-3
> slug: t4-cpu-contention
> commit: bae3102 (bae3102470c445d9907f226e080dca72aa0d4500)
> base_branch: main
> effort: S
> confidence: HIGH
> depends_on: task-2
> branch: feature/task-3-t4-cpu-contention

## Evidence
- `gpu_fuzzy_trader/config.py:382-385` – `BACKTEST_BATCH_WORKERS = min(8, os.cpu_count() or 4)` → 2 on 2-core but comment scoped to 8-core host; JAX fork hazard not scoped to 2-core
- `gpu_fuzzy_trader/backtest/cpu_engine.py:30-65` – `_jax_runtime_loaded()` guard chooses ThreadPool after JAX import else ProcessPool; payload `_batch_eval_rule_set_pickled` is fork-sensitive
- `gpu_fuzzy_trader/evolution/numba_ops.py` – Numba `njit` with default `num_threads=cpu_count` (oversubscribes with torch/XLA threads)
- `gpu_fuzzy_trader/rb_governor.py` – parallel RB rule-set simulations via `ProcessPoolExecutor`/`ThreadPoolExecutor`
- `gpu_fuzzy_trader/_jax_env.py` – currently no `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`NUMBA_NUM_THREADS` caps

## Scope
- In: `gpu_fuzzy_trader/config.py` (worker caps + helper), `gpu_fuzzy_trader/backtest/cpu_engine.py` (pool choice + worker count), `gpu_fuzzy_trader/evolution/numba_ops.py` (thread cap), `gpu_fuzzy_trader/rb_governor.py` (worker cap), `gpu_fuzzy_trader/_jax_env.py` (env thread hints)
- Out (do NOT touch): `gpu_engine.py` JAX kernels, `evaluator_v5.ipynb`, `config` validation thresholds, `phase2_rule_pool` genome logic
- Related callers (blast): `run_pipeline.py` (calls RB), `phases/*`, `rb_governor.py`, any use of `BACKTEST_BATCH_WORKERS` – run `nexus impact --json --targets gpu_fuzzy_trader/config.py,gpu_fuzzy_trader/backtest/cpu_engine.py,gpu_fuzzy_trader/evolution/numba_ops.py,gpu_fuzzy_trader/rb_governor.py`

## Acceptance criteria
- [ ] On `cpu_count==2`, `BACKTEST_BATCH_WORKERS` resolves to 2 (or 1 when JAX already loaded to avoid fork) and max workers for RB/Phase2 batch eval is `min(2, BACKTEST_BATCH_WORKERS)`; no code path spawns >2 concurrent CPU-heavy workers
- [ ] Numba ops thread count respects `NUMBA_NUM_THREADS=min(2,cpu_count)` set before numba import and enforced via `numba.set_num_threads()` after import; verified by `numba.get_num_threads()==2`
- [ ] `cpu_engine.py` correctly chooses `ThreadPoolExecutor` when `jax` already in `sys.modules` else `ProcessPoolExecutor`, with worker count derived from `BACKTEST_BATCH_WORKERS` (centralized helper, no ad-hoc `os.cpu_count()` duplication)
- [ ] `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS` default to `2` on 2-core via `os.environ.setdefault` in `_jax_env.py` early import, without overwriting explicit user settings
- [ ] RB Governor + island scheduler cap parallelism to 2 and log once

## Verification gates (exact commands)
1. `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest -q tests/unit/test_cpu_engine.py tests/unit/test_rb_fail_closed.py` – expected: passing
2. `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest -q tests/property/test_cpu_engine_properties.py` – expected: passing
3. `.venv/bin/python -c "import os, unittest.mock; import gpu_fuzzy_trader.config as c; print(c.BACKTEST_BATCH_WORKERS); import numba; print(numba.get_num_threads())"` – mocked cpu_count=2: prints `2` and `2`
4. `git diff main...feature/task-3-t4-cpu-contention --stat` – only `config.py`, `cpu_engine.py`, `numba_ops.py`, `rb_governor.py`, `_jax_env.py` touched

## STOP conditions
- STOP if `BACKTEST_BATCH_WORKERS` becomes 0 or > cpu_count on 2-core mock (oversubscription/undersubscription guard)
- STOP if `numba.get_num_threads() > 2` after import on mocked 2-core (thread leak)
- STOP if `tests/unit/test_cpu_engine.py` newly spawns `ProcessPoolExecutor` when `jax` mocked as already loaded (should be ThreadPool) – fork-after-JAX hazard
- STOP if `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest -q tests/unit/test_cpu_engine.py` fails after change

## Implementation sketch
- In `config.py`, introduce `resolve_backtest_workers()` helper reading `os.cpu_count()` dynamically, capping at 2 for `<=2` cores else 8; set `BACKTEST_BATCH_WORKERS = resolve_backtest_workers()`
- In `_jax_env.py` early (before numba/torch import), `os.environ.setdefault("NUMBA_NUM_THREADS", str(min(2,cpu_count)))` and same for `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`
- In `evolution/numba_ops.py` after import, call `numba.set_num_threads(int(os.environ.get("NUMBA_NUM_THREADS", str(numba.get_num_threads()))))`
- In `cpu_engine.py` and `rb_governor.py`, replace ad-hoc `os.cpu_count()` calls with `config.BACKTEST_BATCH_WORKERS` or `resolve_backtest_workers()`, and ensure pool choice respects `_jax_runtime_loaded()` guard

## Graph context
- Hubs: `config.py` (worker constant), `cpu_engine.py` (RB & Phase2 exact sim), `numba_ops.py` (NSGA sorting)
- Blast: `nexus impact --json --targets gpu_fuzzy_trader/config.py,gpu_fuzzy_trader/backtest/cpu_engine.py,gpu_fuzzy_trader/evolution/numba_ops.py,gpu_fuzzy_trader/rb_governor.py`
