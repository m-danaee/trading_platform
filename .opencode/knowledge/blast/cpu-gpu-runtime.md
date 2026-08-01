# Blast Radius — execution unit `cpu-gpu-runtime`

- **Task**: `.opencode/tasks/execution-unit-cpu-gpu-runtime.json`
- **Plan**: `.opencode/plans/PLAN.md` (CPU vs GPU Runtime Benchmark)
- **Branch**: `feature/task-1-cpu-gpu-benchmark` (base: `main`)
- **Profile**: `balanced`
- **Analyzed**: manual blast analysis (scripts/nexus-blast.js and scripts/nexus-graph.sh are **unavailable** in repo; no graph.json cached; degraded to shell/manual analysis with node available but no scripts present — warning)

## Scope In (planned editable files)

| File | Kind | Change type |
|---|---|---|
| `tests/benchmark/test_phase2_cpu_gpu_runtime.py` | NEW | additive test file |
| `RUN.md` | EXISTING | optional docs-only (task-3, only if benchmark usage not self-explanatory) |

## Risk: **LOW** — score 1 / 10

### Rationale

1. **No production code touched.** Scope In contains only a *new* benchmark test file
   (`tests/benchmark/`) and optionally the docs file `RUN.md`. Nothing under
   `gpu_fuzzy_trader/` (config, backtest engines, phases, evolution, RB) is edited.
2. **New file → zero existing callers.** `test_phase2_cpu_gpu_runtime.py` does not
   exist yet; grep confirms the only references are the plan and execution-unit JSON.
   Nothing imports a benchmark test module.
3. **Existing pattern proven.** The new test mirrors `tests/benchmark/test_phase2_gpu_throughput.py`
   (same imports, `_gpu_available()` skip guard, `jax.block_until_ready` timing,
   `@pytest.mark.benchmark` marker). `conftest.py` already skips benchmark-marked tests
   unless `RUN_BENCHMARKS=1`, so default CI/low-memory runs are unaffected.
4. **Imports are read-only consumers.** The new file will import from
   `gpu_fuzzy_trader.backtest.gpu_engine`, `backtest.cpu_engine`, `backtest.jax_compat`,
   `_gpu_runtime`, and `config` — all consumed, none modified.
5. **RUN.md is documentation.** Docs have no code callers; task-3 change is optional
   and non-gating.
6. **GPU-dependent test skips gracefully** via `get_gpu_backtest_engine_class()`
   (already catches ImportError/RuntimeError/OSError/AttributeError) and the
   `_gpu_available()` / `pytest.skip` guard — safe on CPU-only or memory-constrained hosts.
7. **Verification scope is targeted**: new benchmark + `test_gpu_engine.py` +
   `test_phase2_use_gpu_flag.py` under `PYTEST_LOW_MEMORY=1 RUN_BENCHMARKS=1`.

### Residual risks (all LOW, non-blocking)

- New test could fail if the new file has an import-time bug → caught by targeted
  verification before merge; collection failure would be contained to benchmark dir.
- GPU benchmark timing on RTX 4050 may be noisy → benchmark is informational, not a
  gate; no assertions on absolute time.
- README consistency: RUN.md addition is docs-only; no code reads RUN.md.

## Callers / dependents

- Direct callers of the new file: **none** (new file; nothing references it).
- Files depending on `RUN.md`: none (human/CI docs only).
- Production files read (not modified): `gpu_engine.py`, `cpu_engine.py`,
  `jax_compat.py`, `_gpu_runtime.py`, `config.py` — unchanged, so no re-test burden
  beyond the two related GPU tests listed in PLAN verification.

## Impacted paths

- `tests/benchmark/test_phase2_cpu_gpu_runtime.py` (add)
- `RUN.md` (optional doc edit)

## Required review policy

- **LOW blast** → standard `balanced` review applies.
- Recommended: `unified-reviewer` pass over the feature branch diff
  (`tests/benchmark/test_phase2_cpu_gpu_runtime.py`, `RUN.md`).
- No `spec-reviewer` escalation needed (blast is LOW, not HIGH; no production code,
  no shared API surface, no data/security migration).
- Branch cleanup after merge via `scripts/nexus-branch-cleanup.sh` (script policy).
