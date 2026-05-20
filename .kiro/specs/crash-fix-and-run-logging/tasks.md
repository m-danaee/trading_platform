# Implementation Plan: crash-fix-and-run-logging

## Overview

Six surgical fixes applied across four files. Each task is self-contained and can be verified independently. Tasks are ordered so that the safest, most isolated changes come first.

## Tasks

- [x] 1. Add `RUN_LOG_PATH` to `config.py` and reduce `PHASE1_SAMPLING_TOTAL`
  - In `gpu_fuzzy_trader/config.py`, add `RUN_LOG_PATH = os.path.join(OUTPUTS_DIR, "run.log")` immediately after the `OUTPUTS_DIR` definition.
  - Change `PHASE1_SAMPLING_TOTAL` from `300_000` to `150_000`.
  - Replace the existing comment on `PHASE1_SAMPLING_TOTAL` with: `# Primary memory control knob for Phase 2. Raising this value increases JAX device array size proportionally. On WSL with limited GPU memory, keep at or below 150_000.`
  - _Requirements: 4.1, 4.2, 6.1_

  - [ ]* 1.1 Write smoke tests for config changes
    - Assert `_cfg.PHASE1_SAMPLING_TOTAL == 150_000`
    - Assert `_cfg.RUN_LOG_PATH` ends with `"run.log"` and starts with `_cfg.OUTPUTS_DIR`
    - _Requirements: 4.1, 6.1_

- [x] 2. Fix JAX memory pre-allocation in `_jax_env.py`
  - In `gpu_fuzzy_trader/_jax_env.py`, add `os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")` as the first line inside `configure_jax_env()`, before the existing `JAX_PLATFORMS` line.
  - _Requirements: 2.1, 2.2_

  - [ ]* 2.1 Write smoke test for `XLA_PYTHON_CLIENT_PREALLOCATE`
    - Unset `XLA_PYTHON_CLIENT_PREALLOCATE` from `os.environ`, call `configure_jax_env()`, assert `os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"`.
    - _Requirements: 2.1_

  - [ ]* 2.2 Write property test for `configure_jax_env` not overwriting pre-existing env vars
    - **Property 2: configure_jax_env does not overwrite pre-existing env vars**
    - Use `@given(st.text(min_size=1))` to generate arbitrary pre-existing values for `XLA_PYTHON_CLIENT_PREALLOCATE`.
    - Set the env var to the generated value, call `configure_jax_env()`, assert the value is unchanged.
    - `@settings(max_examples=100)`
    - `# Feature: crash-fix-and-run-logging, Property 2: configure_jax_env does not overwrite pre-existing env vars`
    - _Requirements: 2.2_

- [x] 3. Fix O(N²) metrics cache rebuild in `evox_runner.py`
  - In `gpu_fuzzy_trader/evolution/evox_runner.py`, inside `_run_nsga3()`, locate the metrics cache rebuild block that runs after `_nsga3_environmental_selection`.
  - Replace the nested loop with a dict-based O(N) lookup:
    ```python
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
  - _Requirements: 5.1, 5.2, 5.3_

  - [ ]* 3.1 Write property test for O(N) metrics cache rebuild correctness
    - **Property 3: O(N) metrics cache rebuild correctness**
    - Use `@given(...)` to generate a random `merge_pop` (2D int array), `merge_metrics` (list of dicts), and a `population` (subset of rows from `merge_pop`).
    - Apply the O(N) rebuild logic and assert each `metrics_cache[i]` equals the metrics dict for `population[i]` if it exists in `merge_metrics`, or `{}` otherwise.
    - `@settings(max_examples=100)`
    - `# Feature: crash-fix-and-run-logging, Property 3: O(N) metrics cache rebuild correctness`
    - _Requirements: 5.1, 5.2, 5.3_

- [x] 4. Add archive save to `Rule_Pool_Generator.run()` in `phase2_rule_pool.py`
  - In `gpu_fuzzy_trader/phases/phase2_rule_pool.py`, inside `Rule_Pool_Generator.run()`, add the archive save block after the `Reporter` block and before `self._release_resources()`:
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
  - _Requirements: 3.1, 3.2, 3.3_

  - [ ]* 4.1 Write unit test for archive save call ordering
    - Mock `Rule_Pool_Generator.save_archive` and `Rule_Pool_Generator._release_resources`.
    - Run `Rule_Pool_Generator.run()` with a minimal population (pop_size=2, n_generations=1) using `CPUBacktestEngine`.
    - Assert `save_archive` was called before `_release_resources` using `unittest.mock.call_args_list` or `Mock.assert_called`.
    - _Requirements: 3.1_

  - [ ]* 4.2 Write unit test for archive save exception handling
    - Mock `Rule_Pool_Generator.save_archive` to raise `RuntimeError("disk full")`.
    - Run `Rule_Pool_Generator.run()` and assert no exception propagates.
    - Assert a WARNING log record containing "archive save failed" was emitted.
    - _Requirements: 3.3_

- [x] 5. Add `run.log` FileHandler to `Pipeline_Orchestrator` in `run_pipeline.py`
  - Add `_cfg.RUN_LOG_PATH` to the `_temporary_output_paths` context manager so it is rebound when `output_dir` is overridden (mirror the pattern used for `_PIPELINE_LOG_PATH`).
  - Add two private methods to `Pipeline_Orchestrator`:
    ```python
    def _attach_run_log_handler(self) -> logging.FileHandler:
        os.makedirs(os.path.dirname(_cfg.RUN_LOG_PATH) or ".", exist_ok=True)
        handler = logging.FileHandler(_cfg.RUN_LOG_PATH, mode="a", encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        ))
        logging.getLogger().addHandler(handler)
        return handler

    def _detach_run_log_handler(self, handler: logging.FileHandler) -> None:
        logging.getLogger().removeHandler(handler)
        handler.close()
    ```
  - In `Pipeline_Orchestrator.run()`, call `_attach_run_log_handler()` immediately after `_create_output_dirs()`. Store the returned handler. Wrap the remainder of `run()` in a `try/finally` block that calls `_detach_run_log_handler(handler)`.
  - Write the START separator immediately after attaching the handler and the END separator inside the `finally` block before detaching.
  - Separator format: a line of 80 `=` characters, then `[{UTC ISO timestamp}] Pipeline run START` (or `END`), then another line of 80 `=` characters.
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 6.2_

  - [ ]* 5.1 Write unit tests for run.log handler lifecycle
    - Use a `tmp_path` fixture (pytest) to redirect `_cfg.RUN_LOG_PATH` to a temp file.
    - Run `Pipeline_Orchestrator.run()` with all phases mocked out.
    - Assert `run.log` exists and contains "Pipeline run START" and "Pipeline run END".
    - Assert root logger has no extra FileHandlers after `run()` returns (handler was detached).
    - Run `run()` a second time and assert the file still contains both separator lines from the first run (append mode).
    - _Requirements: 1.1, 1.4, 1.5, 1.6, 1.7_

  - [ ]* 5.2 Write property test for log formatter message preservation
    - **Property 1: Log formatter preserves message content**
    - Use `@given(st.text(min_size=1))` to generate arbitrary log messages.
    - Create a `logging.LogRecord` with the generated message, apply the run.log formatter, assert the formatted string contains the original message.
    - `@settings(max_examples=100)`
    - `# Feature: crash-fix-and-run-logging, Property 1: Log formatter preserves message content`
    - _Requirements: 1.3_

- [x] 6. Final checkpoint — ensure all tests pass
  - Run `PYTHONPATH=. pytest tests/unit/test_crash_fix_and_run_logging.py tests/property/test_crash_fix_properties.py -v` and confirm all tests pass.
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster fix-only pass.
- Tasks 1–4 are fully independent and can be applied in any order.
- Task 5 depends on Task 1 (needs `RUN_LOG_PATH` in config).
- No existing tests should be broken by these changes; `PHASE1_SAMPLING_TOTAL` reduction may make existing property tests faster.
- Property tests use Hypothesis with `@settings(max_examples=100)`.
- All new tests go in `tests/unit/test_crash_fix_and_run_logging.py` (unit) and `tests/property/test_crash_fix_properties.py` (property).

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1", "2", "3", "4"] },
    { "id": 1, "tasks": ["5"] },
    { "id": 2, "tasks": ["6"] }
  ]
}
```
