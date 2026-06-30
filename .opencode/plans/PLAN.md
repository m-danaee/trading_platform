# Plan — Phase 2 Priority A Runtime Fixes

## Goal
Cut Phase 2 per-generation wall-clock ~10-20x via batched offspring eval,
periodic val simulation, and an island-patience bug fix. No search-budget cut;
minimal quality risk.

## Task 1 (A1) — Batch offspring evaluation

### Problem
In both generation loops the 200 offspring are evaluated one chromosome at a
time via `_evaluate_chromosome(...)`, each call dispatching a batch=1
`simulate_rule_batch` (train) + batch=1 (val) = 400 GPU dispatches/gen. The
initial population is already batched via `_evaluate_population_indices`.

### Fix
Replace the per-chromosome offspring loop with a single call to
`_evaluate_population_indices(offspring, list(range(pop_size)), ...)` — the
SAME helper already used for initial-pop eval. It batches train+val, dedups
identical chromosomes, and uses the global eval cache.

### Files
1. `gpu_fuzzy_trader/evolution/evox_runner.py` — both loops.
2. `tests/unit/test_phase2_offspring_batch.py` (NEW).

### Exact edits — `_run_nsga3` (~line 1996)
Replace:
```python
        off_obj = np.full((pop_size, 3), np.inf)
        off_metrics: list[dict] = [{} for _ in range(pop_size)]
        for i in range(pop_size):
            obj, metrics = _evaluate_chromosome(
                offspring[i], dont_cares, engine, pareto_archive,
                val_engine=val_engine,
                stage_params=stage_params,
                cv_fold_evaluator=cv_fold_evaluator,
            )
            off_obj[i] = obj
            off_metrics[i] = metrics
```
with:
```python
        off_obj = np.full((pop_size, 3), np.inf)
        off_metrics: list[dict] = [{} for _ in range(pop_size)]
        _evaluate_population_indices(
            offspring,
            list(range(pop_size)),
            dont_cares,
            engine,
            pareto_archive,
            off_obj,
            off_metrics,
            val_engine=val_engine,
            global_metrics_cache=global_metrics_cache,
            diversity_reference=diversity_reference,
            diversity_metrics_by_key=diversity_metrics_by_key,
            stage_params=stage_params,
        )
```
(All those vars are in scope in `_run_nsga3`: `global_metrics_cache`,
`diversity_reference`, `diversity_metrics_by_key` are all set up before the loop.)

### Exact edits — `_run_nsga2_fallback` (~line 1678, identical loop)
Replace the same per-chromosome loop with the same `_evaluate_population_indices`
call. The fallback loop may not have `diversity_reference` /
`diversity_metrics_by_key` / `global_metrics_cache` in scope — check; if absent,
omit those kwargs (they default to None). Keep `val_engine`, `stage_params`.

### Note on cv_fold_min
`_evaluate_population_indices` does NOT compute CV fold returns (matches the
existing initial-pop path). With `PHASE2_F3_OBJECTIVE="profit_factor"` (active
config) this is correct. If `cv_fold_min` is ever enabled, BOTH initial-pop and
offspring batched paths would need CV handling — pre-existing limitation, out of
scope. Add a brief comment noting this.

### Test — `tests/unit/test_phase2_offspring_batch.py`
- Mock `engine.simulate_rule_batch` to count calls; run a short evolution
  (pop=10, 2 gens, `_EVOX_AVAILABLE=False` → fallback path, FakeEngine-like
  stub returning a list of N metric dicts).
- Assert `simulate_rule_batch` is called ONCE per generation for train (not
  pop_size times). Use a wrapper that records call count and batch sizes.
- Assert objectives/metrics are populated (not inf) after the loop.

### Acceptance
1. `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_offspring_batch.py tests/unit/test_evox_runner.py tests/unit/test_phase2_plateau_restart.py tests/unit/test_phase2_post_restart_stop.py tests/unit/test_plateau_state_leak.py -q` passes.
2. Full `tests/unit/` passes (no regressions).
3. Import check passes.
4. No dead code; the old per-chromosome offspring loop is fully removed (do
   NOT leave it as a commented fallback).

---

## Task 2 (A2) — Periodic val simulation

### Problem
Val `simulate_rule_batch` runs every gen for every chromosome but
`PHASE2_JOINT_TRAIN_VAL=False` ⇒ val never affects objectives. Only used by
deployable_archive tracking + pool admission.

### Fix
Add `PHASE2_VAL_SIM_INTERVAL` (default 3). Run val sim when
`gen % interval == 0` OR on the last gen of the epoch. When skipped,
val_metrics=None and `_update_deployable_archive` is skipped that gen (existing
archive entries persist; new admits only on val-gens).

### Files
1. `gpu_fuzzy_trader/config.py` — new knob.
2. `gpu_fuzzy_trader/evolution/evox_runner.py` — guard val in
   `_evaluate_population_indices` (add `run_val: bool = True` param) and at the
   2 call sites for offspring; guard the `_update_deployable_archive` call.
3. `tests/unit/test_phase2_val_sim_interval.py` (NEW).

### config.py
```python
# PHASE2_VAL_SIM_INTERVAL — run val backtest every N generations during
# evolution (1 = every gen).  Only matters when PHASE2_JOINT_TRAIN_VAL=False
# (val doesn't affect objectives then).  Val always runs on the epoch's last
# gen for pool-admission freshness.
#   1 → original behaviour (val every gen).
#   3 → 3x fewer val sims; deployable_archive refreshes every 3 gens (default).
PHASE2_VAL_SIM_INTERVAL = 3
```

### evox_runner.py
- Add `run_val: bool = True` param to `_evaluate_population_indices`. Guard
  the `val_metrics_list = val_engine.simulate_rule_batch(...)` call:
  `if val_engine is not None and run_val:`.
- Same `run_val` param on `_evaluate_chromosome` (single path) for consistency.
- At offspring call sites: pass `run_val=(gen % int(_cfg.PHASE2_VAL_SIM_INTERVAL) == 0)`.
  For the initial-pop eval call (top of loop), pass `run_val=True`.
- Guard `_update_deployable_archive(...)` with `if run_val_this_gen:`
  (only admit new deployables when val ran). The final gen always runs val
  (it's `gen == n_generations - 1` ⇒ `gen % interval` may not be 0, so force
  `run_val = is_last_gen or gen % interval == 0`).

### Test
- With `PHASE2_VAL_SIM_INTERVAL=2`, mock val_engine; assert val
  `simulate_rule_batch` called only on even gens (0,2,4) over a 5-gen run, not
  every gen. Assert train sim called every gen.

### Acceptance
Same suite passes; no regressions.

---

## Task 3 (A3) — Fix island patience dead-code bug

### Problem
`_should_plateau_early_stop_phase2` for island profiles reads
`stage_params.plateau_early_stop_patience` (=8, from the None-stage profile)
instead of `PHASE2_ISLAND_PLATEAU_EARLY_STOP_PATIENCE` (=6). The island knob
is unreachable.

### Fix
For island profiles, always use `PHASE2_ISLAND_PLATEAU_EARLY_STOP_PATIENCE`
(ignore stage_params patience, since islands run single-stage with stage=None).

### Files
1. `gpu_fuzzy_trader/evolution/evox_runner.py` — `_should_plateau_early_stop_phase2`.
2. `tests/unit/test_phase2_island_early_stop.py` — add a regression test.

### Exact edit (~line 590)
Replace:
```python
    if _cfg.scoped_island_profile(island_profile):
        patience = (
            int(stage_params.plateau_early_stop_patience)
            if stage_params is not None and getattr(stage_params, "plateau_early_stop_patience", None) is not None
            else int(getattr(_cfg, "PHASE2_ISLAND_PLATEAU_EARLY_STOP_PATIENCE", _cfg.PHASE2_PLATEAU_EARLY_STOP_PATIENCE))
        )
    else:
        patience = (
            int(stage_params.plateau_early_stop_patience)
            if stage_params is not None
            else int(_cfg.PHASE2_PLATEAU_EARLY_STOP_PATIENCE)
        )
```
with:
```python
    if _cfg.scoped_island_profile(island_profile):
        # Islands run single-stage (stage=None); the stage_params patience is
        # the GLOBAL default baked into the None profile, NOT the island knob.
        # Use the island-scoped config directly so PHASE2_ISLAND_PLATEAU_EARLY_STOP_PATIENCE
        # actually takes effect.
        patience = int(getattr(
            _cfg, "PHASE2_ISLAND_PLATEAU_EARLY_STOP_PATIENCE",
            _cfg.PHASE2_PLATEAU_EARLY_STOP_PATIENCE,
        ))
    else:
        patience = (
            int(stage_params.plateau_early_stop_patience)
            if stage_params is not None
            else int(_cfg.PHASE2_PLATEAU_EARLY_STOP_PATIENCE)
        )
```

### Test
- Add `test_island_patience_uses_island_knob`: with
  `PHASE2_ISLAND_PLATEAU_EARLY_STOP_PATIENCE=6` and a stage_params whose
  `plateau_early_stop_patience=8` (the bug condition), assert
  `_should_plateau_early_stop_phase2(9, 6, deployable_count=5, island_profile="cluster_0")`
  returns True (streak 6 >= island patience 6) — proving the island knob wins.

### Acceptance
Same suite passes; no regressions.
