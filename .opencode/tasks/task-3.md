# Task 3: Island Scheduler + Pool Admission Fixes

**ID:** task-3
**Branch:** `feat/phase2-island-fixes`
**Files:** 
- `gpu_fuzzy_trader/phases/phase2_island_scheduler.py`
- `gpu_fuzzy_trader/phases/phase2_rule_pool.py`
- `gpu_fuzzy_trader/evolution/evox_runner.py`
**Risk:** Medium (logic changes in scheduling and pool admission)

## Description

Fix three issues in the island scheduling and pool admission logic:

1. **Minimum epoch size guard** — Skip epochs with remaining < 5 generations (useless 1-gen epochs that waste ~30s on engine rebuild)
2. **Monthly gate island-scoped ratio** — Use island-scoped `monthly_admission_min_ratio` instead of global default
3. **Logging patience fix** — Plateau early-stop log message should use the same patience value as the decision logic

## Changes Required

### Change 1: Minimum Epoch Size Guard

**Location:** `gpu_fuzzy_trader/phases/phase2_island_scheduler.py` → `_run_cluster_islands()`

**Problem:** When an island's generation budget is nearly exhausted, the scheduler launches tiny epochs (1-4 generations) that waste ~30s on engine rebuild but provide negligible evolutionary benefit.

**Fix:** Add a guard to skip epochs with `remaining < MIN_EPOCH_GENS` (5 generations):

```python
while any(g._island_generations_done < gens_per_cluster for g in generators.values()):
    for cid in cluster_ids:
        gen = generators[cid]
        if gen._island_generations_done >= gens_per_cluster:
            continue
        remaining = gens_per_cluster - gen._island_generations_done
        
        # NEW: Skip tiny remaining epochs
        MIN_EPOCH_GENS = 5
        if remaining < MIN_EPOCH_GENS:
            logger.info(
                "Phase 2 [%s]: skipping final epoch for cluster %s "
                "(remaining=%d < MIN_EPOCH_GENS=%d)",
                direction, cid, remaining, MIN_EPOCH_GENS,
            )
            gen._island_generations_done = gens_per_cluster  # exit loop cleanly
            continue
        
        epoch_gens = min(_cfg.PHASE2_ISLAND_EPOCH_GENERATIONS, remaining)
        gen.run_epoch(n_generations=epoch_gens)
```

**Acceptance criteria:**
- Epochs with `remaining < 5` are skipped with a log message
- `_island_generations_done` is set to `gens_per_cluster` to exit the loop
- No functional change for epochs with `remaining >= 5`

### Change 2: Monthly Gate Island-Scoped Ratio

**Location:** `gpu_fuzzy_trader/phases/phase2_rule_pool.py` → `finalize_island()`

**Problem:** The monthly admission gate uses the global `PHASE2_MONTHLY_ADMISSION_MIN_RATIO` (0.5) for all islands, but island-scoped hyperparameters should override this.

**Fix:** Use `self.island_hyperparams.monthly_admission_min_ratio` if available:

```python
def finalize_island(self) -> list[dict]:
    # ... existing code ...
    
    # Monthly admission gate
    if _cfg.PHASE2_MONTHLY_ADMISSION_ENABLED:
        # Use island-scoped ratio if available, else global default
        min_ratio = (
            self.island_hyperparams.monthly_admission_min_ratio
            if self.island_hyperparams is not None
            and hasattr(self.island_hyperparams, 'monthly_admission_min_ratio')
            else _cfg.PHASE2_MONTHLY_ADMISSION_MIN_RATIO
        )
        
        pool = [
            rule for rule in pool
            if self._passes_monthly_gate(rule, min_ratio)
        ]
```

**Acceptance criteria:**
- `finalize_island()` checks for `island_hyperparams.monthly_admission_min_ratio`
- Falls back to `PHASE2_MONTHLY_ADMISSION_MIN_RATIO` if not available
- No change to `_passes_monthly_gate()` logic

### Change 3: Logging Patience Fix

**Location:** `gpu_fuzzy_trader/evolution/evox_runner.py` → `_run_nsga3()` and `_run_nsga2_fallback()`

**Problem:** The plateau early-stop log message uses the global `PHASE2_PLATEAU_EARLY_STOP_PATIENCE` (8) instead of the island-scoped `PHASE2_ISLAND_PLATEAU_EARLY_STOP_PATIENCE` (6), causing confusion in logs.

**Fix:** Use the same patience value in the log message as in the decision logic:

```python
# In _run_nsga3() and _run_nsga2_fallback(), when logging plateau early-stop:

# Determine the patience value used in the decision
if island_profile == "island":
    patience_used = _cfg.PHASE2_ISLAND_PLATEAU_EARLY_STOP_PATIENCE
else:
    patience_used = _cfg.PHASE2_PLATEAU_EARLY_STOP_PATIENCE

logger.info(
    "Phase 2 [%s]: plateau early-stop at gen %d "
    "(streak=%d >= patience=%d, best_return=%.4f)",
    island_profile, gen, plateau_streak, patience_used, best_return,
)
```

**Acceptance criteria:**
- Log message uses the same patience value as the decision logic
- For island runs, uses `PHASE2_ISLAND_PLATEAU_EARLY_STOP_PATIENCE`
- For non-island runs, uses `PHASE2_PLATEAU_EARLY_STOP_PATIENCE`

## Acceptance Criteria

1. Epochs with `remaining < 5` are skipped in `_run_cluster_islands`
2. `finalize_island` monthly gate uses island-scoped ratio when `island_hyperparams` is set
3. Log messages show the actual patience value used in the decision
4. Existing tests pass: `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/ -x -q`
5. New test for min epoch size guard

## Verification Commands

```bash
# Run tests
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/ -x -q

# Verify min epoch guard
grep -B 2 -A 10 "MIN_EPOCH_GENS" gpu_fuzzy_trader/phases/phase2_island_scheduler.py

# Verify island-scoped monthly ratio
grep -B 2 -A 8 "monthly_admission_min_ratio" gpu_fuzzy_trader/phases/phase2_rule_pool.py

# Verify logging patience fix
grep -B 2 -A 5 "patience_used" gpu_fuzzy_trader/evolution/evox_runner.py
```

## Rationale

### Why skip epochs with remaining < 5?

- Engine rebuild takes ~30s (JAX compilation, GPU memory allocation)
- 1-4 generations provide negligible evolutionary benefit
- Wastes ~30s per tiny epoch with no meaningful progress
- Setting `_island_generations_done = gens_per_cluster` exits the loop cleanly

### Why island-scoped monthly ratio?

- Different islands may have different data characteristics
- Island hyperparameters allow fine-tuning per-island behavior
- Global default may be too strict or too lenient for specific islands
- Consistent with other island-scoped parameters (e.g., `plateau_patience`)

### Why fix the logging patience?

- Logs showed `patience=8` but decision used `patience=6` for islands
- Caused confusion when debugging early-stop behavior
- Log message should reflect the actual decision logic

## Notes

- This is a medium-risk task (logic changes in scheduling and pool admission)
- The min epoch guard is low-risk (just skips tiny epochs)
- The monthly gate fix is low-risk (just uses the right parameter)
- The logging fix is low-risk (just corrects the log message)
- All changes should be tested on next Colab run to validate behavior
