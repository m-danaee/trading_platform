# CPU vs GPU Runtime Benchmark

## Goal

Add a reproducible, targeted benchmark that compares the CPU and physical GPU
implementations of Phase 2 `simulate_rule_batch`, then use measured warm and
cold runtimes to determine which backend is faster for the intended workload.

## task-1: Add comparable benchmark

- Target files: `tests/benchmark/test_phase2_cpu_gpu_runtime.py`
- Dependencies: existing synthetic benchmark fixture patterns, CPU and GPU
  backtest engines, JAX synchronization.
- Acceptance criteria:
  - Uses identical data, chromosomes, and simulation constants for both engines.
  - Reports engine construction, first-call, and steady-state runtimes.
  - Verifies that a physical JAX GPU is active and skips otherwise.
  - Avoids full pipeline execution and large project datasets.

## task-2: Verify behavior

- Target files: benchmark plus existing GPU engine tests.
- Dependencies: `.venv`, `PYTEST_LOW_MEMORY=1`, `RUN_BENCHMARKS=1`.
- Acceptance criteria:
  - New benchmark passes on the local RTX 4050 GPU.
  - Related CPU/GPU engine unit tests pass.
  - Output clearly identifies the lower steady-state runtime.

## task-3: Document recommendation

- Target files: `RUN.md` only if benchmark usage is not self-explanatory.
- Acceptance criteria:
  - State that Phase 2 chromosome search can use GPU acceleration.
  - State that exact rule-set/RB/OOS paths remain CPU-backed.
  - Report the measured local result without claiming universal hardware parity.

## Verification

```bash
PYTEST_LOW_MEMORY=1 RUN_BENCHMARKS=1 .venv/bin/python -m pytest -q -s tests/benchmark/test_phase2_cpu_gpu_runtime.py
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest -q tests/unit/test_gpu_engine.py tests/unit/test_phase2_use_gpu_flag.py
```
