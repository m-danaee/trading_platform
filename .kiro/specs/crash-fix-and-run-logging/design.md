# Design Document: crash-fix-and-run-logging

## Overview

This document describes the design for six targeted fixes to the GPU-Fuzzy Trading Pipeline that address a WSL terminal crash during Phase 2, missing persistent log capture, and a bug that prevents the rule archive from being saved between runs.

The fixes are surgical — each touches exactly one module and does not alter the pipeline's public API or data contracts. The changes are:

1. **`run_pipeline.py`** — attach a `FileHandler` to the root logger writing to `outputs/run.log`
2. **`_jax_env.py`** — set `XLA_PYTHON_CLIENT_PREALLOCATE=false` before JAX imports
3. **`phases/phase2_rule_pool.py`** — call `save_archive()` before `_release_resources()` in `Rule_Pool_Generator.run()`
4. **`config.py`** — reduce `PHASE1_SAMPLING_TOTAL` from 300,000 to 150,000 and add `RUN_LOG_PATH`
5. **`evolution/evox_runner.py`** — replace the O(N²) metrics cache rebuild loop with an O(N) dict lookup

---

## Architecture

No architectural changes are made. All fixes are in-place modifications to existing modules. The call graph and data flow remain identical.

```
run_pipeline.py
  └─ configure_jax_env()          ← Fix 2: sets XLA_PYTHON_CLIENT_PREALLOCATE=false
  └─ Pipeline_Orchestrator.run()
       ├─ _attach_run_log_handler()   ← Fix 1 (new helper): attaches FileHandler
       ├─ Phase 1 ...
       ├─ Phase 2: Rule_Pool_Generator.run()
       │    ├─ run_phase2_evolution()
       │    │    └─ _run_nsga3()      ← Fix 5: O(N) metrics cache rebuild
       │    ├─ save pool + history
       │    ├─ save_archive()         ← Fix 3 (new call): persists archive
       │    └─ _release_resources()
       ├─ Phase 3 ...
       └─ _detach_run_log_handler()   ← Fix 1 (new helper): removes FileHandler
```

---

## Components and Interfaces

### Fix 1 — run.log FileHandler (`run_pipeline.py`)

Two private helpers are added to `Pipeline_Orchestrator`:

```python
def _attach_run_log_handler(self) -> logging.FileHandler:
    """
    Attach a FileHandler to the root logger writing to RUN_LOG_PATH.
    Returns the handler so it can be detached later.
    """

def _detach_run_log_handler(self, handler: logging.FileHandler) -> None:
    """Remove the handler from the root logger and close it."""
```

`run()` is modified to call `_attach_run_log_handler()` at the very start (after creating output directories) and `_detach_run_log_handler()` in a `finally` block so it always runs even on exception.

The separator lines are written directly to the handler's stream (or via a dedicated logger call) so they appear in `run.log` regardless of the configured log level.

**Handler configuration:**
- Path: `_cfg.RUN_LOG_PATH` (dynamically resolved via `_temporary_output_paths`)
- Mode: `"a"` (append)
- Level: `logging.DEBUG`
- Formatter: `"%(asctime)s [%(levelname)s] %(name)s: %(message)s"` with `datefmt="%Y-%m-%dT%H:%M:%S"`

**Separator format:**
```
================================================================================
[2025-01-15T14:32:01 UTC] Pipeline run START
================================================================================
```

### Fix 2 — JAX memory pre-allocation (`_jax_env.py`)

One line is added to `configure_jax_env()`:

```python
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
```

This must appear before the existing `JAX_PLATFORMS` line to ensure it is set before any JAX import can trigger XLA initialisation.

### Fix 3 — Archive save in `Rule_Pool_Generator.run()` (`phase2_rule_pool.py`)

After the `Reporter` block and before `self._release_resources()`, the following is inserted:

```python
try:
    saved = Rule_Pool_Generator.save_archive(
        self.direction, self.feature_infos, pool
    )
    logger.info(
        "Phase 2 [%s]: archive saved with %d rules to %s",
        self.direction, len(saved), _ARCHIVE_PATHS[self.direction],
    )
except Exception as exc:
    logger.warning(
        "Phase 2 [%s]: archive save failed (non-fatal): %s",
        self.direction, exc,
    )
```

### Fix 4 — Config constants (`config.py`)

Two changes:

```python
# Primary memory control knob for Phase 2. Raising this value increases the
# size of JAX device arrays allocated by GPUBacktestEngine proportionally.
# On WSL with limited GPU memory, keep this at or below 150_000.
PHASE1_SAMPLING_TOTAL = 150_000

RUN_LOG_PATH = os.path.join(OUTPUTS_DIR, "run.log")
```

`RUN_LOG_PATH` is placed immediately after `OUTPUTS_DIR` so the dependency is visually clear.

### Fix 5 — O(N) metrics cache rebuild (`evox_runner.py`)

The existing O(N²) loop in `_run_nsga3()`:

```python
# BEFORE (O(N²))
metrics_cache = [{} for _ in range(n_alive)]
for i in range(n_alive):
    key = tuple(population[i].tolist())
    for j, m in enumerate(merge_metrics):
        if m and tuple(merge_pop[j].tolist()) == key:
            metrics_cache[i] = m
            break
```

Is replaced with:

```python
# AFTER (O(N))
_merge_metrics_by_key: dict[tuple, dict] = {
    tuple(merge_pop[j].tolist()): m
    for j, m in enumerate(merge_metrics)
    if m
}
metrics_cache = [
    _merge_metrics_by_key.get(tuple(population[i].tolist()), {})
    for i in range(n_alive)
]
```

The dict comprehension iterates `merge_metrics` once (O(2N) = O(N)). The list comprehension then does N O(1) dict lookups.

---

## Data Models

No new data models are introduced. All existing JSON schemas (`pool`, `archive`, `history`) are unchanged.

The `run.log` file is plain text (not JSON-lines) to make it human-readable in a crash scenario. Each line is a standard Python logging record or a separator line.

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

Two of the acceptance criteria are suitable for property-based testing:

### Property 1: Log formatter preserves message content

*For any* non-empty log message string, when the run.log FileHandler formats a `logging.LogRecord` containing that message, the formatted output string SHALL contain the original message as a substring.

**Validates: Requirements 1.3**

### Property 2: configure_jax_env does not overwrite pre-existing env vars

*For any* non-empty string value pre-assigned to `XLA_PYTHON_CLIENT_PREALLOCATE` in `os.environ` before `configure_jax_env()` is called, the value of `XLA_PYTHON_CLIENT_PREALLOCATE` after the call SHALL equal the pre-assigned value.

**Validates: Requirements 2.2**

### Property 3: O(N) metrics cache rebuild correctness

*For any* merged population array and corresponding metrics list, after rebuilding the metrics cache using the dict-based O(N) lookup, each entry in the rebuilt cache SHALL equal the metrics dict associated with the chromosome at the same index in the surviving population (or an empty dict if no metrics were recorded for that chromosome).

**Validates: Requirements 5.1, 5.2, 5.3**

---

## Error Handling

| Scenario | Module | Behaviour |
|---|---|---|
| `run.log` directory does not exist | `run_pipeline.py` | `os.makedirs` is called before attaching the handler (already done by `_create_output_dirs`) |
| `run.log` is not writable (permissions) | `run_pipeline.py` | `FileHandler` constructor raises; exception propagates and pipeline does not start |
| `save_archive` raises any exception | `phase2_rule_pool.py` | Caught, logged as WARNING, execution continues to `_release_resources()` |
| `XLA_PYTHON_CLIENT_PREALLOCATE` already set | `_jax_env.py` | `setdefault` is a no-op; existing value preserved |
| Chromosome not found in metrics lookup dict | `evox_runner.py` | `dict.get(..., {})` returns empty dict; no exception |

---

## Testing Strategy

This feature involves configuration side-effects, logging infrastructure, and a pure-function algorithmic fix. The testing approach is:

**Unit tests** (in `tests/unit/test_crash_fix_and_run_logging.py`):
- Verify `run.log` FileHandler is attached and detached correctly
- Verify separator lines appear in `run.log`
- Verify `save_archive` is called before `_release_resources` in `Rule_Pool_Generator.run()`
- Verify `save_archive` exceptions are caught and logged as WARNING
- Verify `PHASE1_SAMPLING_TOTAL == 150_000` in config
- Verify `RUN_LOG_PATH` is derived from `OUTPUTS_DIR`
- Verify `XLA_PYTHON_CLIENT_PREALLOCATE` is set to `"false"` by `configure_jax_env()`

**Property-based tests** (in `tests/property/test_crash_fix_properties.py`) using Hypothesis:
- **Property 1**: For any log message, the formatted output contains the message (validates formatter correctness)
- **Property 2**: For any pre-existing env var value, `configure_jax_env()` does not overwrite it (validates `setdefault` usage)
- **Property 3**: For any merged population and metrics list, the O(N) cache rebuild produces correct per-individual metrics (validates the algorithmic fix)

Each property test runs a minimum of 100 iterations via Hypothesis `@settings(max_examples=100)`.

Property test tag format: `# Feature: crash-fix-and-run-logging, Property {N}: {property_text}`
