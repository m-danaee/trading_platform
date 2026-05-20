# Requirements Document

## Introduction

The GPU-Fuzzy Trading Pipeline crashes the WSL terminal or VSCode at the end of Phase 2 due to out-of-memory (OOM) conditions caused by JAX pre-allocating GPU memory, large JAX device arrays, and an O(N²) metrics cache rebuild in the NSGA-III loop. Additionally, all in-progress log output is lost when the terminal crashes because no persistent log file captures stdout/stderr. A separate bug causes the Phase 2 rule archive to never be saved after a run, so each run starts from scratch instead of benefiting from previously discovered rules.

This spec covers six targeted fixes: adding a persistent `run.log` file, preventing JAX memory over-allocation on WSL, saving the archive after Phase 2 completes, reducing the default sampling footprint, fixing the O(N²) metrics cache rebuild, and adding the `run.log` path to config.

## Glossary

- **Pipeline_Orchestrator**: The top-level class in `run_pipeline.py` that sequences all five pipeline phases.
- **Rule_Pool_Generator**: The Phase 2 class in `phases/phase2_rule_pool.py` that runs NSGA-III evolution and persists the rule pool.
- **run.log**: A plain-text append-mode log file at `outputs/run.log` that captures all Python logging output (stdout/stderr) for every pipeline run.
- **pipeline.log**: The existing JSON-lines structured timing log at `outputs/pipeline.log`.
- **JAX**: The GPU-accelerated numerical computing library used by `GPUBacktestEngine`.
- **XLA**: The compiler backend used by JAX; controls GPU memory allocation behaviour.
- **WSL**: Windows Subsystem for Linux — the execution environment where the crash is observed.
- **PHASE1_SAMPLING_TOTAL**: Config constant controlling how many rows are sampled for the Phase 2 backtest engine.
- **RUN_LOG_PATH**: New config constant for the `outputs/run.log` file path.
- **Archive**: The persistent JSON file (`phase2_rule_archive/phase2_{direction}_archive.json`) that stores the best rules across runs to seed future populations.
- **NSGA-III**: The multi-objective evolutionary algorithm used in Phase 2, implemented in `evolution/evox_runner.py`.
- **metrics_cache**: A per-individual list of backtest metric dicts maintained during NSGA-III evolution.
- **_jax_env.py**: Module that sets XLA/JAX environment variables before any JAX import.

## Requirements

### Requirement 1: Persistent Run Log File

**User Story:** As a developer running the pipeline on WSL, I want all log output written to a persistent file, so that I can diagnose crashes that kill the terminal before logs are flushed.

#### Acceptance Criteria

1. WHEN the Pipeline_Orchestrator starts a run, THE Pipeline_Orchestrator SHALL attach a `logging.FileHandler` to the root Python logger that writes to `outputs/run.log` before any phase executes.
2. THE run.log FileHandler SHALL capture log records at DEBUG level and above, including all records from all named loggers in the package.
3. THE run.log FileHandler SHALL format each record with an ISO 8601 timestamp, log level, logger name, and message.
4. THE run.log FileHandler SHALL open `outputs/run.log` in append mode so that successive runs accumulate in the same file.
5. WHEN a pipeline run starts, THE Pipeline_Orchestrator SHALL write a separator line containing the UTC timestamp and the text "Pipeline run START" to `run.log`.
6. WHEN a pipeline run ends (whether successfully or via an unhandled exception), THE Pipeline_Orchestrator SHALL write a separator line containing the UTC timestamp and the text "Pipeline run END" to `run.log`.
7. THE Pipeline_Orchestrator SHALL remove the run.log FileHandler from the root logger after the run ends, so that subsequent programmatic uses of Pipeline_Orchestrator in the same process do not accumulate duplicate handlers.

### Requirement 2: JAX Memory Pre-allocation Fix

**User Story:** As a developer running the pipeline on WSL with a GPU, I want JAX to not pre-allocate 75% of GPU memory on startup, so that the pipeline does not OOM-kill the WSL session during Phase 2.

#### Acceptance Criteria

1. WHEN `configure_jax_env()` is called in `_jax_env.py`, THE configure_jax_env function SHALL set the environment variable `XLA_PYTHON_CLIENT_PREALLOCATE` to `"false"` before any JAX module is imported.
2. THE configure_jax_env function SHALL use `os.environ.setdefault` so that a value already set in the environment by the user is not overridden.
3. WHEN `XLA_PYTHON_CLIENT_PREALLOCATE` is already set to `"false"` by the user, THE configure_jax_env function SHALL leave the existing value unchanged.

### Requirement 3: Archive Save After Phase 2 Run

**User Story:** As a developer running the pipeline multiple times, I want the Phase 2 rule archive to be saved after each run, so that future runs can seed their populations from previously discovered rules.

#### Acceptance Criteria

1. WHEN `Rule_Pool_Generator.run()` completes evolution and saves the pool and history files, THE Rule_Pool_Generator SHALL call `Rule_Pool_Generator.save_archive(self.direction, self.feature_infos, pool)` before calling `_release_resources()`.
2. WHEN `save_archive` is called, THE Rule_Pool_Generator SHALL log the number of rules written to the archive at INFO level.
3. IF `save_archive` raises an exception, THEN THE Rule_Pool_Generator SHALL log the exception as a WARNING and continue to `_release_resources()` without re-raising.

### Requirement 4: Reduce Default Memory Footprint

**User Story:** As a developer on a memory-constrained WSL machine, I want the default sampling size to be smaller, so that Phase 2 does not allocate excessively large JAX device arrays.

#### Acceptance Criteria

1. THE config.py module SHALL set `PHASE1_SAMPLING_TOTAL` to `150_000` as the default value.
2. THE config.py module SHALL include a comment on the `PHASE1_SAMPLING_TOTAL` line explaining that it is the primary memory control knob for Phase 2 and that increasing it raises JAX device array size proportionally.

### Requirement 5: O(N) Metrics Cache Rebuild in NSGA-III

**User Story:** As a developer running Phase 2 with a population of 200, I want the metrics cache rebuild after NSGA-III selection to be O(N) rather than O(N²), so that each generation completes faster and Python keeps fewer objects alive.

#### Acceptance Criteria

1. WHEN `_run_nsga3()` in `evox_runner.py` rebuilds `metrics_cache` after calling `_nsga3_environmental_selection`, THE _run_nsga3 function SHALL use a dict keyed by chromosome tuple to look up metrics in O(1) per surviving individual.
2. THE rebuilt metrics_cache SHALL contain the correct metrics dict for each surviving individual, matching the chromosome at the same index in the new `population` array.
3. IF a surviving individual's chromosome is not found in the lookup dict, THEN THE _run_nsga3 function SHALL assign an empty dict `{}` for that individual's metrics entry.

### Requirement 6: Add RUN_LOG_PATH to Config

**User Story:** As a developer, I want the run.log path defined in config.py, so that all path constants remain in the single source of truth and other modules can reference it without hardcoding strings.

#### Acceptance Criteria

1. THE config.py module SHALL define `RUN_LOG_PATH = os.path.join(OUTPUTS_DIR, "run.log")` after the `OUTPUTS_DIR` definition.
2. THE Pipeline_Orchestrator in `run_pipeline.py` SHALL reference `_cfg.RUN_LOG_PATH` (or the dynamically rebound equivalent) when constructing the run.log FileHandler path, rather than hardcoding the string.
