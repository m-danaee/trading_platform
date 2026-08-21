# Task 1: Baseline benchmark & hardware-aware runtime profile
> id: task-1
> slug: t4-baseline-profile
> commit: bae3102 (bae3102470c445d9907f226e080dca72aa0d4500)
> base_branch: main
> effort: S
> confidence: HIGH
> depends_on: none
> branch: feature/task-1-t4-baseline-profile

## Evidence
- `gpu_fuzzy_trader/research_profile.py` – existing phase timers, prior art for profiling
- `gpu_fuzzy_trader/_gpu_runtime.py:42-95` – `detect_gpu_vram_gb()`, `detect_system_ram_gb()`, `_vram_batch_cap`, `_ram_batch_cap`, `resolve_phase2_gpu_batch_size()` (lru_cache)
- `gpu_fuzzy_trader/config.py:2334-2357` – `_apply_colab_gpu_defaults()` Colab-only detection (`is_colab_runtime()` checks `/content` or `COLAB_RELEASE_TAG`)
- `gpu_fuzzy_trader/config.py:735-804` – batch/scan/cpu-route knobs to be profiled
- `tests/benchmark/test_phase2_gpu_throughput.py` – benchmark harness pattern (RUN_BENCHMARKS=1)

## Scope
- In: `gpu_fuzzy_trader/research_profile.py`, `gpu_fuzzy_trader/_gpu_runtime.py`, `gpu_fuzzy_trader/config.py` (profile helper only, no behavior mutation yet), `scripts/benchmark_t4.py` (new), `tests/unit/test_t4_profile.py` (new)
- Out (do NOT touch): `evaluator_v5.ipynb`, `gpu_fuzzy_trader/backtest/gpu_engine.py`, `gpu_fuzzy_trader/backtest/cpu_engine.py`, `outputs/*.json` commits, RB thresholds
- Related callers (blast): `run_pipeline.py:log_gpu_runtime_config` calls hardware probes; `config.py` imported by ~20 modules (blast radius high – run `nexus impact --json --targets gpu_fuzzy_trader/config.py,gpu_fuzzy_trader/_gpu_runtime.py` before implementer)

## Acceptance criteria
- [ ] `scripts/benchmark_t4.py` (or research_profile extension) emits per-phase timings + snapshot `{cpu_count, vram_gb, gpu_name, ram_gb, jax_backend, devices, batch_resolved, scan_unroll, cpu_route, workers}` to `outputs/reports/t4_profile.json` or stdout at `--dry-run`
- [ ] Helper `is_t4_runtime()` / `detect_hardware_profile()` correctly identifies T4 via `nvidia-smi --query-gpu=name` containing "T4" OR vram 15-16 GiB + driver, independent of `/content`; unit test mocks subprocess output (T4 name, non-T4 name, no GPU)
- [ ] `BACKTEST_BATCH_WORKERS` logged and test asserts `<= cpu_count` and `<=2` when `cpu_count==2`
- [ ] Import still works: `.venv/bin/python -c "from gpu_fuzzy_trader.run_pipeline import Pipeline_Orchestrator; print('OK')"` prints `OK`

## Verification gates (exact commands)
1. `.venv/bin/python -c "from gpu_fuzzy_trader.run_pipeline import Pipeline_Orchestrator; print('OK')"` – expected: `OK`
2. `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest -q tests/unit/test_t4_profile.py` – expected: all passing (3+ cases: T4, non-T4, no-GPU)
3. `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest -q tests/unit/test_config_validation.py` – expected: passing (no regression)
4. `.venv/bin/python scripts/benchmark_t4.py --dry-run` – expected: exits 0, writes JSON with keys `cpu_count`, `vram_gb`, `backend`, `batch_size`
5. `git diff main...feature/task-1-t4-baseline-profile --stat` – only `research_profile.py`, `_gpu_runtime.py`, `config.py`, `scripts/benchmark_t4.py`, `tests/unit/test_t4_profile.py` changed

## STOP conditions
- STOP if `git rev-parse --short HEAD` base drift shows file `gpu_fuzzy_trader/config.py:735` no longer contains `PHASE2_GPU_BATCH_SIZE = 256` (plan assumptions broken)
- STOP if `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest -q tests/unit/test_config_validation.py` fails on base branch before edits (baseline broken – fix first)
- STOP if new `nvidia-smi` probe requires network or breaks CPU fallback (`jax_compat.py` path must remain importable without GPU)
- STOP if `gpu_fuzzy_trader/config.py` no longer defines `is_colab_runtime()` (preserve legacy path)

## Implementation sketch
- Extend `_gpu_runtime.py` with `detect_gpu_name()` mirroring `detect_gpu_vram_gb()` pattern, add `detect_hardware_profile()` returning dict, make `log_gpu_runtime_config()` richer and lru_cached
- Add `is_t4_runtime()` + skeleton `_apply_t4_gpu_defaults()` (env `GPU_OPT_T4=0/1` auto) that only logs for now (actual mutation in task-2); keep Colab behavior identical
- Add `scripts/benchmark_t4.py` that probes hardware, optionally runs tiny synthetic `GPUBacktestEngine.simulate_rule_batch` micro-bench (if JAX available) and `Data_Loader` bench, outputs JSON
- Extend `research_profile.py` to include hardware snapshot if reports existed

## Graph context
- Hub candidates: `config.py` (20 importers), `run_pipeline.py` (fan-out), `_gpu_runtime.py` (low fan-in, high leverage)
- Blast radius placeholder: run `nexus impact --json --targets gpu_fuzzy_trader/config.py,gpu_fuzzy_trader/_gpu_runtime.py` before implementer dispatch to fill dependents/related tests
