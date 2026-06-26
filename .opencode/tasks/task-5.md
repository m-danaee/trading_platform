# Task 5 — `fix/plateau-config-tuning-and-banner` (Fix F + config)

## Branch
`fix/plateau-config-tuning-and-banner` (from latest `main`, after task-4 merge).

## Problem
(a) The pipeline banner advertises `gen=132` but actual per-run gen is 10/4 —
misleading. (b) Plateau config is over-twitchy: patience=5, min_gen=3,
min_delta=0.02 — combined with the (now-fixed) leak this killed runs at gen 3.
Now that the leak and behavior are fixed (tasks 1-4), tune the knobs against
correct behavior.

## Required Changes

### Fix F — Honest banner
**File:** `gpu_fuzzy_trader/run_pipeline.py` (line ~223, the
`"PHASE2 algo=%s pop=%d gen=%d ..."` format string).

Surface the real island budget when island mode is active:
```python
# When island mode:
"PHASE2 algo=%s pop=%d island_total=%d per_cluster=%d epoch=%d joint_train_val=%s | "
# When single-island:
"PHASE2 algo=%s pop=%d gen=%d joint_train_val=%s | "
```
Use `PHASE2_ISLAND_TOTAL_GENERATIONS`, `gens_per_cluster = total // n_clusters`,
and `PHASE2_ISLAND_EPOCH_GENERATIONS`. Compute `gens_per_cluster` the same way
`_run_cluster_islands` does (`max(1, total // max(1, n_clusters))`). Keep the
existing surrounding log format; only the labeled fields change.

### Config tuning
**File:** `gpu_fuzzy_trader/config.py`

| Key | Current | New | Rationale |
|-----|---------|-----|-----------|
| `PHASE2_PLATEAU_EARLY_STOP_PATIENCE` | 5 | 8 | Let epochs explore before stopping |
| `PHASE2_PLATEAU_EARLY_STOP_MIN_GENERATION` | 3 | 6 | Avoid stopping in transient |
| `PHASE2_PLATEAU_EARLY_STOP_MIN_DELTA_PCT` | 0.02 | 0.05 | 0.02% is below noise floor |
| `PHASE2_ISLAND_EPOCH_GENERATIONS` | 10 | 15 | Fewer, longer epochs = less migration overhead, more convergence headroom per epoch |

Note: `PHASE2_ISLAND_TOTAL_GENERATIONS = PHASE2_GENERATIONS` (132) stays, so
`gens_per_cluster = 132 // 3 = 44`, run in epochs of 15 (→ 3 epochs per cluster,
last one short). Verify the `assert` statements at the bottom of `config.py`
(e.g. `PHASE2_PLATEAU_EARLY_STOP_MIN_GENERATION <= PHASE2_STAGE_A_GENERATIONS`)
still hold — `PHASE2_STAGE_A_GENERATIONS=85` so `6 <= 85` ✓. If any assert
would break, adjust the *asserted constant* consistently, not the knob.

### Docs
Update `README.md` config table for all four changed keys + the banner.

## Acceptance Criteria
1. Banner shows `island_total=132 per_cluster=44 epoch=15` (when island mode) —
   not the misleading `gen=132`.
2. The four config keys have the new values.
3. `config.py` asserts at import still pass (run
   `.venv/bin/python -c "import gpu_fuzzy_trader.config"`).
4. `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit -q` passes — fix
   any test that hard-coded the old values (update to new values, do not weaken
   assertions incorrectly).

## Target Files
- `gpu_fuzzy_trader/run_pipeline.py`
- `gpu_fuzzy_trader/config.py`
- `README.md`
- any test hard-coding old values.

## Verification
```
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit -q
.venv/bin/python -c "import gpu_fuzzy_trader.config; print('config OK')"
```
Do NOT run the full pipeline.

## Notes
- This task is intentionally last: tuning must be measured against the corrected
  behavior from tasks 1-4, not the leaky behavior.
- If a test legitimately encodes the old twitchy threshold as a *correctness*
  expectation (not a hard-coded value), discuss before changing — prefer
  updating the value to match the new tuned default.
- Clean up dead code after changes (per AGENTS.md).
