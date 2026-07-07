# Task 10: Gate Cache Refresh on Per-Epoch Window Rotation

## Task ID
`task-10` (tenth of 12 tasks in the 2026-07-07 audit fix plan)

## Title
Gate Cache Refresh on Per-Epoch Window Rotation

## Goal
Fix audit finding #8. The `refresh_objectives_on_resume` flag
(evolution/evox_runner.py:2620) wipes `objectives`,
`metrics_cache`, and `global_metrics_cache` at every epoch
resume. This is CORRECT when the train window changes between
epochs (per-epoch rotation, default after task-1) — the cache
IS stale. But it's WASTEFUL when the window is fixed
(`PHASE2_PER_EPOCH_WINDOW_ROTATION=False`, legacy mode) — the
cache is valid, and the wipe forces re-evaluation of the entire
population, dropping the cache hit rate to 0-7% in the log.

Gate the refresh on `PHASE2_PER_EPOCH_WINDOW_ROTATION`: refresh
when rotation is on, skip when rotation is off.

## Audit Citation
- Confirmed by static inspection:
  - `evolution/evox_runner.py:2620-2628` — the refresh block
  - `phases/phase2_rule_pool.py:3261` — the call site that sets
    `refresh_objectives_on_resume = not first_epoch` unconditionally
  - `config.py:380-390` — `PHASE2_PER_EPOCH_WINDOW_ROTATION = True`
    (default after task-1)
- Run log evidence (2026-07-07): cache_hit_rate=0.00 at most gens,
  0.06 at best. The cache is being cleared before it can do useful
  work.

## Target Files
- `gpu_fuzzy_trader/phases/phase2_rule_pool.py`
  - Line 3261: gate `refresh_objectives_on_resume` on
    `PHASE2_PER_EPOCH_WINDOW_ROTATION`. If rotation is ON (default),
    keep the current behavior (`refresh = not first_epoch`). If
    rotation is OFF (legacy), set `refresh = False` (skip the
    wipe).
  - Update the comment block to reflect the gating.
- `tests/unit/test_phase2_rule_pool.py`
  - Add a test asserting that when
    `PHASE2_PER_EPOCH_WINDOW_ROTATION=True` (default),
    `refresh_objectives_on_resume` is `not first_epoch` (current
    behavior, regression guard).
  - Add a test asserting that when
    `PHASE2_PER_EPOCH_WINDOW_ROTATION=False`, the refresh is
    skipped even on resumed epochs.

## Current Behavior
- `phases/phase2_rule_pool.py:3261`:
  ```python
  refresh_objectives_on_resume = not first_epoch
  ```
  This sets the refresh to True on every resumed epoch regardless
  of whether the window changes. When `PHASE2_PER_EPOCH_WINDOW_ROTATION=True`
  (the default after task-1), the window DOES change (resampled
  per epoch), so refresh IS correct. When `PHASE2_PER_EPOCH_WINDOW_ROTATION=False`
  (legacy fixed-window mode), the window does NOT change, so the
  refresh is wasteful.
- The actual refresh logic is in `evolution/evox_runner.py:2620-2628`:
  ```python
  if refresh_objectives_on_resume and state is not None:
      objectives[:] = np.full((pop_size, N_OBJ), np.inf)
      metrics_cache = [{} for _ in range(pop_size)]
      if _cfg.PHASE2_EVAL_GLOBAL_CACHE and global_metrics_cache is not None:
          global_metrics_cache.clear()
      logger.info(...)
  ```
  This is the engine-side wipe. It runs whenever the caller passes
  `refresh_objectives_on_resume=True`.

## Scope
1. **Gate the refresh on rotation** (phase2_rule_pool.py:3261):
   - Replace:
     ```python
     refresh_objectives_on_resume = not first_epoch
     ```
   - With:
     ```python
     # → fixes audit finding #8: only refresh when the window changes
     # (rotation on, default). If rotation is off, the cache is valid
     # and the wipe is wasteful (cache_hit_rate drops to 0).
     refresh_objectives_on_resume = (
         not first_epoch
         and bool(getattr(_cfg, "PHASE2_PER_EPOCH_WINDOW_ROTATION", True))
     )
     ```
2. **Update the comment** at line 3257-3260:
   - Replace the "Conservative: always refresh when resuming..."
   comment with a more accurate one explaining the gating.
3. **Add audit-finding linkage**:
   - Add `# → fixes audit finding #8 (cache refresh was wasteful
     in fixed-window mode; now gated on rotation flag)` in the
     comment block.
4. **Do NOT change**:
   - The actual refresh logic in `evox_runner.py:2620` (it remains
     a no-op when `refresh_objectives_on_resume=False`).
   - The `PHASE2_PER_EPOCH_WINDOW_ROTATION` config flag (set by
     task-1).
   - Any other file outside `phases/phase2_rule_pool.py` and the
     test file.

## Acceptance Criteria
1. When `PHASE2_PER_EPOCH_WINDOW_ROTATION=True` (default), behavior
   matches pre-task-10 exactly (`refresh = not first_epoch`,
   cache cleared on resumed epochs, log shows "refresh_objectives_on_resume enabled").
2. When `PHASE2_PER_EPOCH_WINDOW_ROTATION=False` (legacy), the
   refresh is SKIPPED on resumed epochs (cache NOT cleared,
   log does NOT show "refresh_objectives_on_resume enabled").
3. The log line `cache_hit_rate` is expected to rise to ≥ 0.4 in
   mid-gen when rotation is OFF (verifiable in the test by
   asserting the refresh flag is False).
4. The default config behavior is bit-identical to pre-task-10
   (since `PHASE2_PER_EPOCH_WINDOW_ROTATION=True` is the default).
5. All existing tests pass: `test_phase2_rule_pool.py`,
   `test_phase2_window_rotation.py`, `test_phase2_island_scheduler.py`,
   `test_phase2_monthly_admission.py`, `test_evox_runner.py`, etc.

## Verification
Run only related unit tests with `PYTEST_LOW_MEMORY=1` and `.venv`:

```bash
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
- This is a small surgical fix (~5 lines of source code, ~30
  lines in 1 test file). Keep the diff minimal.
- The fix is a one-line behavioral change; the default behavior
  is preserved (rotation on → refresh on).
- This task is the dependency child of task-1 (per-epoch
  window rotation). With task-1 merged, the refresh is correct
  by default. With task-1 disabled (legacy mode), the refresh
  is now correctly disabled too.
