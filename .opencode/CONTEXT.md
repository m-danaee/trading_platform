# Nexus Context

workflow: default
execution_mode: delegated
branch_cleanup_policy: always
active_objective: T4 GPU + 2-core CPU performance optimization (GPU-first)
current_phase: PLANNED (awaiting TASK_IMPACT_READY)
base_branch: main
branch_policy: isolated
execution_mode: delegated
verification_baseline:
  build: ".venv/bin/python -c \"from gpu_fuzzy_trader.run_pipeline import Pipeline_Orchestrator; print('OK')\" – expected OK"
  test: "PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest -q tests/unit/test_config_validation.py tests/unit/test_cpu_engine.py tests/property/test_gpu_engine_properties.py – must pass; NEVER run full suite without PYTEST_LOW_MEMORY=1"
  lint: "none detected (ruff cache present)"
  typecheck: "none detected"
  benchmark: "scripts/benchmark_t4.py --dry-run (to be created in task-1); tests/benchmark/test_phase2_gpu_throughput.py with RUN_BENCHMARKS=1"
plan_commit: bae3102 (bae3102470c445d9907f226e080dca72aa0d4500)
generated_at: 2026-08-21T16:05:09Z
graphify:
  graph_json: not present (graphify-out/GRAPH_REPORT.md missing; seeded via graphify update if needed)
  lessons: not present
pending_blockers: none
next_action: nexus run transition --to PLANNED, then nexus impact --json --targets for task-1

hardware_target:
  cpu: 2 cores (oversubscription-sensitive; BACKTEST_BATCH_WORKERS, NUMBA_NUM_THREADS, OMP/MKL must cap at 2)
  gpu: T4 16 GiB (prefer GPU over CPU; PHASE2_GPU_CPU_ROUTE_LARGE_DATA should be False, batch 256, scan_unroll 16)
  ram: ~12-16 GiB host (RAM cap currently 64 for ≤13 GiB hides T4 throughput – fix in task-2)
  current_baseline: RTX 4050 6 GiB + 8-core hybrid policy + Colab-only T4 handling
