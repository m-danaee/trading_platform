# Task 2: T4 GPU-first runtime tuning (JAX/XLA, batch/scan, VRAM caps, cache)
> id: task-2
> slug: t4-gpu-tuning
> commit: bae3102 (bae3102470c445d9907f226e080dca72aa0d4500)
> base_branch: main
> effort: M
> confidence: HIGH
> depends_on: task-1
> branch: feature/task-2-t4-gpu-tuning

## Evidence
- `gpu_fuzzy_trader/_gpu_runtime.py:56-95` – `_vram_batch_cap` (24 GiB→96 anomaly), `_ram_batch_cap` (≤13 GiB→64 hides T4), `resolve_phase2_gpu_batch_size()` with `lru_cache`
- `gpu_fuzzy_trader/_jax_env.py:22-80` – `configure_jax_env()` sets `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `MEM_FRACTION=0.8`, `JAX_PLATFORMS=cuda,cpu`, `JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache or /content/jax_cache`, `XLA_FLAGS` with `cuda_data_dir` + `ptxas` fallback
- `gpu_fuzzy_trader/config.py:735-820` – `PHASE2_GPU_BATCH_SIZE=256`, `PHASE2_GPU_BATCH_SIZE_AUTO=True`, `PHASE2_SCAN_UNROLL=32`, `PHASE2_GPU_CPU_ROUTE_LARGE_DATA=True`, `PHASE2_GPU_DATA_INT8=True`, `PHASE2_GPU_USE_FP32=True`, `PHASE2_GPU_EVENT_DRIVEN=True`
- `gpu_fuzzy_trader/backtest/gpu_engine.py:180-260` – O(B×N×K) linear VRAM growth, `lax.scan` unroll
- `gpu_fuzzy_trader/config.py:2334-2357` – Colab caps scan to 16, but generic T4 does not

## Scope
- In: `gpu_fuzzy_trader/config.py` (hardware-aware defaults), `gpu_fuzzy_trader/_gpu_runtime.py` (cap logic), `gpu_fuzzy_trader/_jax_env.py` (XLA flags, thread hints, cache dir creation)
- Out (do NOT touch): `cpu_engine.py` kernel logic, `evaluator_v5.ipynb`, RB gate thresholds, feature modes
- Related callers (blast): `phases/phase2_rule_pool.py` (reads batch/scan), `backtest/gpu_engine.py` (warmup), every file importing `config` (~20 files) – run `nexus impact --json --targets gpu_fuzzy_trader/_gpu_runtime.py,gpu_fuzzy_trader/_jax_env.py,gpu_fuzzy_trader/config.py`

## Acceptance criteria
- [ ] Generic T4 detection (`is_t4_runtime()` or `cpu_count<=2` + `vram≈16 GiB` + `name contains T4`) auto-sets: `PHASE2_GPU_CPU_ROUTE_LARGE_DATA=False`, `PHASE2_SCAN_UNROLL=16` (capped from 32), `PHASE2_GPU_BATCH_SIZE_AUTO` heuristic allows T4 to sustain 256 until host RAM cap; env `PHASE2_GPU_BATCH_SIZE` still overrides
- [ ] Fixed VRAM tier: 24 GiB tier no longer limits to 96 (now `min(config,256)` or documented >24 GiB behavior), and 13 GiB host-RAM cap raised from 64→128 when GPU route active (or conditional); each tier covered by unit test
- [ ] `configure_jax_env()` on 2-core ensures `JAX_COMPILATION_CACHE_DIR` directory exists (`os.makedirs`), does not overwrite user-provided `XLA_FLAGS`/`MEM_FRACTION`/`PREALLOCATE`, logs chosen flags, and respects `JAX_PLATFORMS=cuda,cpu`
- [ ] `log_gpu_runtime_config()` now logs `gpu_name`, `batch_resolved`, `scan_unroll`, `cpu_route`, `cache_dir` once at startup (no duplicate probe per generation due to `lru_cache`)
- [ ] Parity preserved: `tests/property/test_gpu_engine_properties.py::test_property_16_gpu_cpu_all_metrics_parity` passes on CPU fallback and would pass on T4

## Verification gates (exact commands)
1. `.venv/bin/python -c "from gpu_fuzzy_trader.config import PHASE2_GPU_CPU_ROUTE_LARGE_DATA, PHASE2_SCAN_UNROLL; print(PHASE2_GPU_CPU_ROUTE_LARGE_DATA, PHASE2_SCAN_UNROLL)"` – mocked T4 env: prints `False 16`
2. `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest -q tests/unit/test_t4_profile.py tests/unit/test_config_validation.py tests/property/test_gpu_engine_properties.py -k parity` – expected: passing
3. `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest -q tests/unit/test_cpu_engine.py` – expected: passing
4. `git diff main...feature/task-2-t4-gpu-tuning --stat` – only `_gpu_runtime.py`, `_jax_env.py`, `config.py`, and T4 profile tests changed

## STOP conditions
- STOP if `gpu_fuzzy_trader/config.py` no longer defines `_apply_colab_gpu_defaults` (must preserve legacy Colab identity; generalize to `_apply_hardware_gpu_defaults` instead of deleting)
- STOP if new T4 detection triggers on 8-core+RTX4050 host (must be T4-specific, not broad cpu_count<=8)
- STOP if `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest -q tests/property/test_gpu_engine_properties.py` fails (parity broken)
- STOP if `resolve_phase2_gpu_batch_size()` returns <16 or >512 on mocked T4/2-core (sanity)

## Implementation sketch
- Fix `_vram_batch_cap`: 24 GiB tier `min(config,256)` (remove 96 choke); keep >24 GiB as `config` default
- Adjust `_ram_batch_cap`: keep 64 only for ≤12 GiB, 128 for 13-16 GiB when GPU route active, or condition on `PHASE2_GPU_CPU_ROUTE_LARGE_DATA`
- Generalize `_apply_colab_gpu_defaults` → `_apply_hardware_gpu_defaults` handling `is_colab_runtime() or is_t4_runtime()`, T4 path sets `PHASE2_GPU_CPU_ROUTE_LARGE_DATA=False`, caps `PHASE2_SCAN_UNROLL` to 16, keeps `PHASE2_GPU_BATCH_SIZE_AUTO=True`; gate with `GPU_OPT_DISABLE=1` escape hatch
- Enhance `_jax_env.py`: ensure cache dir `os.makedirs(exist_ok=True)`, `os.environ.setdefault` for flags, cap host threads via `XLA_FLAGS` if 2-core, do not overwrite user-provided values; log once

## Graph context
- Hubs: `config.py` (20 importers), `_gpu_runtime.py` (feeds `gpu_engine.py` batch size), `_jax_env.py` (imported before first `import jax`)
- Blast: `nexus impact --json --targets gpu_fuzzy_trader/config.py,gpu_fuzzy_trader/_gpu_runtime.py,gpu_fuzzy_trader/_jax_env.py`
