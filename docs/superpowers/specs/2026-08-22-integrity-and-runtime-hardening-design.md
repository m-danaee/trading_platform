# Integrity and Runtime Hardening Design

## Goal

Restore strict exact-execution and frozen-archive integrity, make runtime and
benchmark behaviour truthful, and remove generated OpenCode artifacts from
the repository history without changing research or OOS admission policy.

## Scope

### Barrier cache contract

Barrier cache identity will include a named cache-format version, the horizon,
and the complete sorted Cartesian TP/SL pair set returned by
`configured_barrier_pairs()`. Cache reads will also require the exact current
set of barrier columns before use. A stale, malformed, or incomplete cache is
treated as a miss and is rebuilt; it must never select the aggregate-label
fallback path.

### Frozen MTF archive contract

For a frozen directional candidate, the LWC rule hashes must equal the
directional archive hashes as a `Counter`, including multiplicity. The
candidate `rules_set` remains required to equal `lwc_rules`. HWC and MWC
equality checks are unchanged.

### Production Phase 2 benchmark contract

The production loader attaches exact barriers, and the current GPU engine
correctly dispatches those batches to the CPU reference evaluator. The T4
benchmark will therefore create exact barrier columns and report that
CPU-exact route. It will not present a legacy aggregate-label JAX microbenchmark
as production Phase 2 performance. A new JAX exact-execution engine is out of
scope because it would additionally need parity for per-symbol position locks
and time-priority release ordering, and cannot be accepted without live GPU
and CPU-reference parity evidence.

The evolution benchmark will use the production don't-care class codes for
binary, positive, and ternary features: `[2, 5, 3]`.

### Runtime initialization and T4 detection

`run_pipeline` will call `configure_jax_env()` after standard-library imports
but before importing NumPy, pandas, or project modules that can import either.
The T4 VRAM fallback applies only when GPU name probing is unavailable; a
known non-T4 16 GiB device remains non-T4.

### Dependency and repository hygiene

The virtual environment will be synchronized to the declared pandas range
`>=2.0,<3` and checked with `pip check`. Trailing whitespace will be removed
from the three reported files.

Generated `.opencode/cache/`, `.opencode/handoffs/`, `.opencode/impact/`,
`.opencode/runs/`, and `.opencode/trajectories/` paths will be ignored and
removed from all reachable local history. Human-authored `.opencode/plans/`
and `.opencode/tasks/` remain tracked. The history rewrite covers `main` and
`chore/low-memory-test-suite`, after which the rewritten `main` is force-pushed
to `origin/main` as explicitly authorized.

## Error Handling

Cache reads fail closed to recomputation. Invalid frozen strategies continue
to be skipped with `ValidationError`. Missing JAX, CUDA, or T4 hardware only
skips hardware execution; no performance result will be inferred from that
skip.

## Testing

Focused low-memory tests will prove cache invalidation and incomplete-cache
rejection, exact LWC multiset equality, non-T4 16 GiB detection, early
OpenBLAS setup in a fresh interpreter, exact benchmark route/sentinels, and
whitespace/compilation. Tests run sequentially through `.venv` with
`PYTEST_LOW_MEMORY=1` and a writable `MPLCONFIGDIR`.
