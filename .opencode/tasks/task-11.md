# Task 11: Raise PHASE2_VAL_SIM_INTERVAL to 3

## Task ID
`task-11` (eleventh of 12 tasks in the 2026-07-07 audit fix plan)

## Title
Raise PHASE2_VAL_SIM_INTERVAL to 3

## Goal
Fix audit finding #10 (SUSPECTED): `_should_run_val_this_gen`
returns True every generation because `PHASE2_VAL_SIM_INTERVAL=1`.
With joint fitness, val feeds objectives so the every-gen run is
"honest", but with task-1's per-epoch rotation (val window is
fixed, train window rotates per epoch), the val metrics for a
given chromosome are deterministic across generations. Skipping
val on gens 2/3 is safe and cuts GPU work ~33% with zero metric
drift. Raise the interval to 3.

## Audit Citation
- Confirmed by static inspection:
  - `config.py:774` — `PHASE2_VAL_SIM_INTERVAL = 1`
  - `evolution/evox_runner.py:114-128` — `_should_run_val_this_gen`
    returns True every gen when interval=1
  - `evolution/evox_runner.py:254-264` — `_val_metrics_from_cache`
    reads val sidecar from the train metrics dict (cached on the
    chromosome, not on the gen)
- Run log evidence (2026-07-07): "joint train+val fitness enabled
  (val_rows=127644)" at every gen boundary; same val window,
  repeated GPU work.

## Target Files
- `gpu_fuzzy_trader/config.py`
  - Change `PHASE2_VAL_SIM_INTERVAL` from 1 to 3.
  - Update the comment to reflect the new default and the
    rationale (val window fixed after task-1, cache safe).
- `evolution/evox_runner.py` (optional, only if needed)
  - Verify the val metrics cache is keyed on chromosome (not on
    gen). The current implementation stores val as a sidecar on
    the train metrics dict, so it's already chromosome-keyed. No
    change expected; just confirm.
- `tests/unit/test_phase2_val_sim_interval.py`
  - Existing tests (6 tests) should pass after the interval
    change. Verify.
  - Add a test asserting that val metrics for chromosome X at
    gen 1 == gen 2 == gen 3 (deterministic, no cache drift).

## Current Behavior
- `config.py:774`: `PHASE2_VAL_SIM_INTERVAL = 1` (val every gen).
- `evolution/evox_runner.py:114-128`: when interval=1,
  `gen % 1 == 0` is always True, so val runs every gen.
- The val metrics are cached as a sidecar on the train metrics
  dict (lines 254-264). When val is skipped, the code reads
  the cached sidecar.
- Cost: 8 TP × 5 SL × 4 capital × 120 pop × 13 gens × 6 clusters
  × 2 directions = ~37M val backtests over the full pipeline.
  With interval=3, this drops to ~12M (~67% reduction).
- The `is_last_gen` check at line 124 ensures val always runs on
  the last gen (pool admission needs fresh metrics).

## Scope
1. **Raise the interval** (config.py:774):
   - Change `PHASE2_VAL_SIM_INTERVAL = 1` → `PHASE2_VAL_SIM_INTERVAL = 3`.
   - Update the comment block at lines 764-774:
     ```python
     # PHASE2_VAL_SIM_INTERVAL — run val backtest every N generations during
     # evolution (default 3, was 1 before task-11). With per-epoch
     # window rotation (PHASE2_PER_EPOCH_WINDOW_ROTATION, post task-1),
     # the val window is fixed across epochs, so val metrics for a
     # given chromosome are deterministic and safe to cache. Skipping
     # val on gens 2/3 cuts GPU work ~33% with zero metric drift.
     # Val ALWAYS runs on the epoch's last gen (pool-admission freshness).
     #   1 → val every gen (legacy, expensive).
     #   3 → val every 3rd gen (default post task-11; ~33% GPU savings).
     # → fixes audit finding #10 (val every gen is wasteful when
     # window is fixed; cache is safe)
     PHASE2_VAL_SIM_INTERVAL = 3
     ```
2. **Verify val cache is chromosome-keyed** (evox_runner.py:254-264):
   - The current implementation stores val as a sidecar on the
     train metrics dict (e.g., `metrics["val_total_return_pct"]`).
   - When val is skipped, `_val_metrics_from_cache(metrics)`
     reconstructs the val dict from the sidecar.
   - This is chromosome-keyed (the train metrics dict is per-chromosome).
   - No code change expected; just verify the cache works
     correctly with interval=3.
3. **Add audit-finding linkage**:
   - Add `# → fixes audit finding #10 (val every gen is wasteful
     when window is fixed; cache is safe)` in the config.py
     comment block.
4. **Add a regression test** in `tests/unit/test_phase2_val_sim_interval.py`:
   - Test that val metrics for chromosome X at gen 1 == gen 2 == gen 3
     (deterministic, no cache drift). The existing tests may
     already cover this; verify and add if missing.
5. **Do NOT change**:
   - The `is_last_gen` check (val always runs on last gen).
   - The val cache implementation (it remains chromosome-keyed).
   - Any other file outside `config.py` and the test file
     (unless verification finds the cache needs to be made
     chromosome-keyed explicitly).

## Acceptance Criteria
1. `PHASE2_VAL_SIM_INTERVAL = 3` (default, was 1).
2. Val is backtested every 3rd gen (gens 1, 4, 7, ...) instead
   of every gen.
3. Val metrics for chromosome X at gen 1 == gen 2 == gen 3
   (deterministic, no cache drift).
4. The last gen of every epoch always runs val (pool admission
   needs fresh metrics).
5. All existing tests pass: `test_phase2_val_sim_interval.py`,
   `test_phase2_rule_pool.py`, `test_phase2_window_rotation.py`,
   `test_phase2_island_scheduler.py`, `test_phase2_monthly_admission.py`,
   `test_evox_runner.py`, etc.

## Verification
Run only related unit tests with `PYTEST_LOW_MEMORY=1` and `.venv`:

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_val_sim_interval.py -q
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_rule_pool.py -q
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_window_rotation.py -q
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_island_scheduler.py -q
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_monthly_admission.py -q
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_evox_runner.py -q
```

## Notes
- Do NOT modify `evaluator_v5.ipynb`.
- Do NOT run the full project or full test suite locally (OOM risk
  per AGENTS.md).
- This is a small surgical fix (~5 lines in config.py, ~30 lines
  in 1 test file).
- The runtime savings will be validated by the user's next Colab
  run (post all tasks). Should reduce Phase 2 wall time by ~25-30%.
- This task is the dependency child of task-1 (per-epoch
  rotation). With task-1 merged, the val window is fixed and the
  cache is safe. Without task-1, the val window might rotate too
  and the cache could go stale — but since task-1 is already
  merged, this is not a concern.
- The 33% GPU work reduction is a heuristic; the OOS improvement
  is expected to be zero (val metrics are deterministic).
- If the existing tests fail after the interval change, they may
  be hardcoded to expect val every gen. Update them to expect
  val every 3rd gen.
