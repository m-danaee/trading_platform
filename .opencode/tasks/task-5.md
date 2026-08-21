# Task 5: Evolution throughput & final GPU verification
> id: task-5
> slug: t4-evo-throughput
> commit: bae3102 (bae3102470c445d9907f226e080dca72aa0d4500)
> base_branch: main
> effort: M
> confidence: MEDIUM
> depends_on: task-4
> branch: feature/task-5-t4-evo-throughput

## Evidence
- `gpu_fuzzy_trader/config.py:825-860` – `PHASE2_EVAL_GLOBAL_CACHE_MAX_SIZE=600`, `PHASE2_EVAL_BATCH_DEDUP=True`, `PHASE2_VAL_SIM_INTERVAL`, `PHASE2_CPU_BATCH_SIZE=16`, `PHASE2_GPU_EVENT_DRIVEN=True`, `PHASE2_GPU_EVENT_MAX_EVENTS=4096`, `PHASE2_SAMPLING_TOTAL=701000`
- `gpu_fuzzy_trader/phases/phase2_rule_pool.py:150-350` – global cache `chromosome_key→metrics`, hit rate observed 0-4% at 600 (prior Colab log), per-direction trim at 200 via `trim_evolution_state_memory`
- `gpu_fuzzy_trader/_gpu_runtime.py:162-230` – warmup at `batch_size=1` misses real shape, `jax.block_until_ready` barrier, `evict_cluster_signatures` per cluster
- `gpu_fuzzy_trader/evolution/evox_runner.py:80-150` – `_should_run_val_this_gen` cadence (every gen when `JOINT_TRAIN_VAL` or `VAL_IN_FITNESS_PENALTY`, else throttled by `VAL_SIM_INTERVAL`)
- `gpu_fuzzy_trader/config.py:704-730` – `PHASE2_SAMPLE_MAX_BARS_PER_SYMBOL=60000`, `PHASE2_PER_EPOCH_WINDOW_ROTATION=True`, `PHASE1_SAMPLING_TOTAL` linear VRAM lever

## Scope
- In: `gpu_fuzzy_trader/config.py` (cache/sampling/eval intervals), `gpu_fuzzy_trader/phases/phase2_rule_pool.py` (cache eviction/dedup), `gpu_fuzzy_trader/phases/phase2_sparse_encoding.py`, `gpu_fuzzy_trader/_gpu_runtime.py` (warmup), `gpu_fuzzy_trader/evolution/evox_runner.py` (val cadence), `scripts/benchmark_t4.py` (extend), `gpu_fuzzy_trader/_memory.py` (peak RSS logging)
- Out (do NOT touch): `evaluator_v5.ipynb`, RB gate logic, MTF thresholds (`V_HWC`, `MIN_EVIDENCE`), feature modes/detection
- Related callers (blast): `phase2_rule_pool.py` called per direction×island by `run_pipeline.py`, `evox_runner.py` inner loop per generation – run `nexus impact --json --targets gpu_fuzzy_trader/phases/phase2_rule_pool.py,gpu_fuzzy_trader/_gpu_runtime.py,gpu_fuzzy_trader/evolution/evox_runner.py`

## Acceptance criteria
- [ ] Global cache adaptive for T4: `PHASE2_EVAL_GLOBAL_CACHE_MAX_SIZE` raised when `ram>=13 GiB` (e.g., 900) or kept at 600 with LRU trim at 200 already in `trim_evolution_state_memory`; micro-bench shows hit rate not degraded and peak RSS <7.2 GiB on 2-core (`_memory.py:log_memory_rss`)
- [ ] `PHASE2_CPU_BATCH_SIZE` tuned: remains 16 on 2-core (conservative) or raised to 32 only if vectorized CPU memory profile passes `PYTEST_LOW_MEMORY=1` guard; decision justified by benchmark JSON before/after
- [ ] Warmup compiled at representative batch `max(64, resolve_phase2_gpu_batch_size()//4)` and with both dense/k shapes when `PHASE2_ENCODING` sparse, so first generation not paying full XLA compile; `_WARMED_SIGNATURES` populated and log "warmup done" (not silently skipped via `CPURoute`)
- [ ] Sampling/rotation correctness: `PHASE1_SAMPLING_TOTAL` not increased without proportional trade-floor scaling per config comment; `PHASE2_PER_EPOCH_WINDOW_ROTATION` and caps remain correct
- [ ] Final pipeline smoke: `PHASE2_GENERATIONS=2` mini-run or `tests/unit/test_full_pipeline_smoke.py` (if exists) completes in <50% baseline wall-time on synthetic tape, outputs `long.json`/`short.json` still evaluator-compatible (fail-closed empty allowed, non-empty passes `tests/unit/test_rb_fail_closed.py` parity)
- [ ] No duplicated/wasted code (single `_apply_hardware_gpu_defaults` replaces ad-hoc T4 vs Colab helpers)

## Verification gates (exact commands)
1. `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest -q tests/unit/test_config_validation.py tests/unit/test_cpu_engine.py tests/property/test_gpu_engine_properties.py` – expected: passing
2. `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest -q tests/benchmark/test_phase2_gpu_throughput.py` with `RUN_BENCHMARKS=1` if JAX present – expected: throughput (rules/sec) ≥ baseline, no OOM
3. `.venv/bin/python scripts/benchmark_t4.py --component evolution --generations 3 --pop 32` – expected: JSON with `cache_hit_rate`, `warmup_ms`, `gen_avg_ms`, `peak_rss_mib` and `gen_avg_ms` < baseline
4. `.venv/bin/python -c "from gpu_fuzzy_trader.run_pipeline import Pipeline_Orchestrator; import tempfile; print('smoke import ok')"` – expected: `smoke import ok`
5. `git diff main...feature/task-5-t4-evo-throughput --stat` – only evo/config/warmup/profiling files changed; no strategy JSON committed

## STOP conditions
- STOP if `PHASE2_EVAL_GLOBAL_CACHE_MAX_SIZE` raised causes `PYTEST_LOW_MEMORY=1` OOM or `RSS>7.2 GiB` on 2-core micro bench (RUN.md memory guard)
- STOP if warmup now silently skips via `CPURoute` (`_WARMED_SIGNATURES` not populated for expected signature; log must show warmup vs skipped decision)
- STOP if any property test `test_property_16_gpu_cpu_all_metrics_parity` shows >1e-9 drift beyond FP32 tolerance after batch/scan changes
- STOP if `PHASE2_VAL_SIM_INTERVAL` throttling incorrectly applied when `PHASE2_JOINT_TRAIN_VAL=True` (must be every gen in that mode per `evox_runner.py:_should_run_val_this_gen`)

## Implementation sketch
- Keep cache adaptive: `PHASE2_EVAL_GLOBAL_CACHE_MAX_SIZE = min(600 if ram<=13 else 900, PH)` with LRU trim at 200 per direction already in `trim_evolution_state_memory`; benchmark hit rate vs RAM
- Warmup at realistic batch: in `_warmup_engine`, if `resolve_phase2_gpu_batch_size()>32` also pre-warm at `batch=resolve_phase2_gpu_batch_size()//2`; keep `jax.block_until_ready` barrier
- Validate val cadence: ensure `_should_run_val_this_gen` unchanged semantics; document T4 choice (throttled only when joint/val-in-fitness false)
- Centralize hardware helpers: single `detect_hardware_profile()` + `_apply_hardware_gpu_defaults()`; delete duplicated Colab-only branches; ensure `GPU_OPT_DISABLE=1` still escapes

## Graph context
- Hubs: `phase2_rule_pool.py` (Phase2 heart), `_gpu_runtime.py` (warmup), `evox_runner.py` (NSGA loop)
- Blast: `nexus impact --json --targets gpu_fuzzy_trader/phases/phase2_rule_pool.py,gpu_fuzzy_trader/_gpu_runtime.py,gpu_fuzzy_trader/evolution/evox_runner.py`
