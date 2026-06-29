# Plan — Phase 2 Post-Restart Early Stop

## Goal
Cut Phase 2 wall-clock by breaking island epochs when a plateau restart fails
to produce improvement, without reducing search budget or quality.

## Task 1 — Post-restart no-improvement early stop

### Files
1. `gpu_fuzzy_trader/config.py` — 4 new knobs + 1 default change.
2. `gpu_fuzzy_trader/evolution/evox_runner.py` — new helper + logic in BOTH
   generation loops (`_run_nsga2_fallback` ~line 1634, `_run_nsga3` ~line 2094).
3. `tests/unit/test_phase2_post_restart_stop.py` — new unit tests.
4. `tests/unit/test_island_early_stop.py` — update one default assertion.

### config.py additions (insert after `PHASE2_PLATEAU_POST_RESTART_BOOST_GENS`)
```python
PHASE2_PLATEAU_POST_RESTART_STOP_ENABLED = True
PHASE2_PLATEAU_POST_RESTART_STOP_PATIENCE = 3
PHASE2_ISLAND_PLATEAU_POST_RESTART_STOP_ENABLED = True
PHASE2_ISLAND_PLATEAU_POST_RESTART_STOP_PATIENCE = 3
```
Also change `PHASE2_ISLAND_PLATEAU_EARLY_STOP_PATIENCE: int = 8` → `6`
(restarts fire sooner; 6 gens of <0.5% improvement is a clear stall).

### evox_runner.py — dataclass `Phase2EvolutionState`
Add two fields (after `post_restart_gens_remaining`):
```python
post_restart_no_improve_streak: int = 0
post_restart_best_progress: float = -np.inf
```

### New helper (after `_should_plateau_early_stop_phase2`)
```python
def _should_post_restart_early_stop_phase2(
    post_restart_streak: int,
    *,
    island_profile: str = "global",
    stage_params: Phase2StageParams | None = None,
) -> bool:
    """Break the epoch when a plateau restart yields no improvement."""
    if _cfg.scoped_island_profile(island_profile):
        if not bool(getattr(_cfg, "PHASE2_ISLAND_PLATEAU_POST_RESTART_STOP_ENABLED", True)):
            return False
        patience = int(getattr(_cfg, "PHASE2_ISLAND_PLATEAU_POST_RESTART_STOP_PATIENCE",
                               getattr(_cfg, "PHASE2_PLATEAU_POST_RESTART_STOP_PATIENCE", 3)))
    else:
        if not bool(getattr(_cfg, "PHASE2_PLATEAU_POST_RESTART_STOP_ENABLED", True)):
            return False
        patience = int(getattr(_cfg, "PHASE2_PLATEAU_POST_RESTART_STOP_PATIENCE", 3))
    return post_restart_streak >= patience
```

### Both loops — changes (identical pattern)

1. **Init** (next to `post_restart_gens_remaining: int = 0`):
   ```python
   post_restart_no_improve_streak: int = 0
   post_restart_best_progress: float = -np.inf
   ```

2. **Top of each gen** (before the viability-collapse check): `just_restarted = False`.

3. **In BOTH restart blocks** (viability-collapse + plateau), after
   `restart_count += 1`:
   ```python
   post_restart_best_progress = plateau_best_progress
   post_restart_no_improve_streak = 0
   just_restarted = True
   ```
   (Plateau block already `continue`s after, so it skips the steps below — correct.)

4. **After the main `_update_max_return_plateau(...)` call**, add:
   ```python
   if restart_count > 0 and not just_restarted:
       post_restart_best_progress, post_restart_no_improve_streak = (
           _update_max_return_plateau(
               plateau_metric, post_restart_best_progress,
               post_restart_no_improve_streak,
           )
       )
   ```

5. **New stop check** — AFTER the `_should_plateau_early_stop_phase2` block
   (so it only runs when plateau did NOT trigger this gen), BEFORE the
   `if gen == n_generations - 1: break` / offspring generation:
   ```python
   if restart_count > 0 and _should_post_restart_early_stop_phase2(
       post_restart_no_improve_streak,
       stage_params=stage_params,
       island_profile=island_profile,
   ):
       logger.info(
           "%s: post-restart early stop at gen %d "
           "(no improvement for %d gens after restart %d/%d, "
           "best_progress=%.2f%%, deployable_preview=%d)",
           tag, gen + 1, post_restart_no_improve_streak,
           restart_count, max_restarts, plateau_best_progress,
           deployable_count,
       )
       break
   ```

6. **`_run_nsga3` only (resumable)**: on resume read
   `post_restart_no_improve_streak = int(getattr(state, "post_restart_no_improve_streak", 0))`
   and `post_restart_best_progress = float(getattr(state, "post_restart_best_progress", -np.inf))`
   (use getattr — `test_plateau_state_leak._mock_evolution_state` uses
   `__new__` and won't set these).  In `final_state = Phase2EvolutionState(...)`
   pass the two new fields.

### Tests — `tests/unit/test_phase2_post_restart_stop.py`
- `test_config_defaults`: assert the 4 new knobs + island patience == 6.
- `test_post_restart_stop_disabled_island`: when
  `PHASE2_ISLAND_PLATEAU_POST_RESTART_STOP_ENABLED=False` → helper returns False.
- `test_post_restart_stop_patience_island`: streak 2 → False, streak 3 → True
  (island profile).
- `test_post_restart_stop_global_patience`: global profile uses
  `PHASE2_PLATEAU_POST_RESTART_STOP_PATIENCE`.
- `test_post_restart_stop_respects_min_delta`: (optional) confirm reuse of
  `_update_max_return_plateau` delta logic.

### Update existing test
- `tests/unit/test_island_early_stop.py::test_config_defaults`:
  `PHASE2_ISLAND_PLATEAU_EARLY_STOP_PATIENCE == 8` → `== 6`.

### Acceptance criteria
1. `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_post_restart_stop.py tests/unit/test_island_early_stop.py tests/unit/test_phase2_island_early_stop.py tests/unit/test_phase2_plateau_restart.py tests/unit/test_plateau_state_leak.py tests/unit/test_evox_runner.py -q` passes.
2. No other unit test regresses (run the full `tests/unit/` suite with `PYTEST_LOW_MEMORY=1`).
3. New knobs have doc comments; no dead code left behind.
4. `python -c "import gpu_fuzzy_trader.evolution.evox_runner"` imports cleanly.

### Out of scope
- No change to `PHASE2_ISLAND_TOTAL_GENERATIONS`, epoch length, population, or
  val-sim frequency.
- No change to `evaluator_v5.ipynb`.
