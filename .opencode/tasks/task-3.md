# Task 3 — `fix/diversity-restart-on-plateau` (Fix D)

## Branch
`fix/diversity-restart-on-plateau` (from latest `main`, after task-2 merge).

## Problem
On intra-epoch plateau the evolution loop `break`s immediately
(`evox_runner.py:2138-2155`). Despite `pop_unique=1.00`, `max_return` freezes at
identical values across gens/epochs → frozen-elite attractor that mutation=0.30
+ elitism cannot escape. A `_inject_diversity_recovery` helper exists
(`evox_runner.py:747`) but is only used to reinit on collapse, not to extend the
run past a plateau.

## Required Changes

### Diversity restart on first plateau (instead of break)
**File:** `gpu_fuzzy_trader/evolution/evox_runner.py` — the plateau early-stop
branch around line 2138 (`_should_plateau_early_stop_phase2` true branch).

Replace the immediate `break` with a restart-then-continue policy:
1. On the FIRST plateau in an epoch: call a new helper
   `_plateau_diversity_restart(population, objectives, metrics_cache, rng, ...)`
   that:
   - Preserves the current Pareto elite (top-K deployable, e.g. K=min(5, pareto_size)).
   - Reinitializes a configurable fraction
     (`PHASE2_PLATEAU_DIVERSITY_RESTART_FRACTION`, default 0.40) of the
     remaining population via `_init_population` (existing function).
   - Resets those slots' `objectives` to `inf` and `metrics_cache` to `{}`.
   - Bumps `mutation_rate` by a boost factor for ONE generation
     (`PHASE2_PLATEAU_DIVERSITY_RESTART_MUTATION_BOOST`, default 1.6×, capped at
     e.g. 0.6) to kick the search off the elite — restore original rate next gen.
   - Increments a per-epoch `restart_count`.
   - Resets `plateau_streak = 0` (so the restarted phase can plateau again).
2. On the SECOND plateau (i.e. `restart_count >= PHASE2_PLATEAU_MAX_RESTARTS`,
   default 1) in the same epoch: `break` as before (genuinely converged).
3. If the restart would exceed remaining generations, just `break`.

The existing `_should_plateau_early_stop_phase2` stays as the trigger; the new
behavior is in the *response* branch. Add a log line:
`"%s: plateau restart at gen %d (restart %d/%d, reinit %.0f%%, elite_kept=%d)"`.

### New config keys
**File:** `gpu_fuzzy_trader/config.py`
```python
PHASE2_PLATEAU_DIVERSITY_RESTART_ENABLED = True
PHASE2_PLATEAU_DIVERSITY_RESTART_FRACTION = 0.40   # share of pop reinitialized
PHASE2_PLATEAU_DIVERSITY_RESTART_MUTATION_BOOST = 1.6
PHASE2_PLATEAU_MAX_RESTARTS = 1                    # restarts before final break
```
Add doc comments matching the existing config style. Gate the whole feature
behind `PHASE2_PLATEAU_DIVERSITY_RESTART_ENABLED`.

### Respect `island_profile`
The restart must respect the existing `_cfg.scoped_island_profile(island_profile)`
guarding pattern used in `_should_plateau_early_stop_phase2` (don't restart if
plateau early-stop is disabled for the profile).

## Acceptance Criteria
1. First plateau triggers a diversity restart (not a break) when enabled and
   `restart_count < MAX_RESTARTS`; second plateau breaks.
2. Pareto elite is preserved across the restart (test: elite chromosomes survive).
3. `mutation_rate` is boosted for one gen then restored (test).
4. `plateau_streak` resets to 0 after a restart (test).
5. When `PHASE2_PLATEAU_DIVERSITY_RESTART_ENABLED=False`, behavior is identical
   to current (immediate break) — no regression.
6. `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit -q` passes.

## Target Files
- `gpu_fuzzy_trader/evolution/evox_runner.py` (new helper + branch change)
- `gpu_fuzzy_trader/config.py` (4 new keys)
- `README.md` (config table)
- `tests/unit/test_phase2_plateau_restart.py` (new) or extend `test_island_early_stop.py`.

## Verification
```
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit -q
```
Do NOT run the full pipeline.

## Notes
- Reuse `_init_population` and `_inject_diversity_recovery` patterns; do not
  duplicate logic. If `_inject_diversity_recovery` can be generalized to serve
  this, refactor it — but keep its existing call sites working.
- Clean up dead code after the change (per AGENTS.md).
