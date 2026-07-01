# Task 2: EvoX Runner Code Fixes (Cache + Diversity)

**ID:** task-2
**Branch:** `feat/phase2-evox-fixes`
**Files:** `gpu_fuzzy_trader/evolution/evox_runner.py`
**Risk:** Medium (logic changes in evolution loop)

## Description

Fix two critical issues in the EvoX runner that cause cache thrashing and premature convergence:

1. **LRU Cache Trimming** — Replace random eviction with FIFO (dict insertion order) in `_trim_global_metrics_cache()`. Random eviction destroys useful cached results, causing near-zero cache hit rates (0.00-0.07 in logs).

2. **Phenotype-Collapse Recovery Trigger** — Add a new trigger in `_should_inject_diversity_recovery()` that fires when the Pareto front collapses to ≤3 members despite high genetic uniqueness. This addresses the "pareto=1 for 7+ generations" problem where all 200 chromosomes are genetically unique but phenotypically identical.

## Changes Required

### 1. LRU Cache Trimming

**Location:** `gpu_fuzzy_trader/evolution/evox_runner.py` → `_trim_global_metrics_cache()`

**Current implementation:**
```python
def _trim_global_metrics_cache(global_metrics_cache, max_size):
    """Bound run-wide eval cache size to limit RAM growth across long runs."""
    overflow = len(global_metrics_cache) - int(max_size)
    if overflow <= 0:
        return
    import random as _trim_random
    keys_to_remove = _trim_random.sample(
        list(global_metrics_cache.keys()), k=overflow,
    )
    for key in keys_to_remove:
        global_metrics_cache.pop(key, None)
```

**Problem:** Random eviction destroys useful cached results. With 200 pop × 2 evals/gen = 400 sims/gen and cache size 1200, the cache fills in ~3 generations. Random eviction means parents evaluated in gen 1 are likely evicted before offspring in gen 2 are evaluated, causing cache miss rates >90%.

**Fix:** Use FIFO eviction (remove oldest entries first). Python dicts preserve insertion order (3.7+), so we can just pop from the front:

```python
def _trim_global_metrics_cache(global_metrics_cache, max_size):
    """Bound run-wide eval cache size to limit RAM growth across long runs.
    
    Uses FIFO eviction (oldest entries first) to preserve recent evaluations
    and maximize cache hit rates across generations.
    """
    overflow = len(global_metrics_cache) - int(max_size)
    if overflow <= 0:
        return
    # Remove oldest entries (dict preserves insertion order)
    keys_to_remove = list(global_metrics_cache.keys())[:overflow]
    for key in keys_to_remove:
        global_metrics_cache.pop(key, None)
```

### 2. Phenotype-Collapse Recovery Trigger

**Location:** `gpu_fuzzy_trader/evolution/evox_runner.py` → `_should_inject_diversity_recovery()`

**Current implementation:**
```python
def _should_inject_diversity_recovery(
    population_unique_ratio: float,
    stage_params: Phase2StageParams | None = None,
    *,
    pareto_size: int = 0,
    plateau_streak: int = 0,
    pop_size: int = 0,
    valid_count: int = 0,
) -> bool:
    if not bool(getattr(_cfg, "PHASE2_DIVERSITY_RECOVERY_ENABLED", True)):
        return False
    if (
        population_unique_ratio
        < _diversity_recovery_min_unique_ratio(stage_params)
    ):
        return True
    collapse_threshold = max(2, int(pop_size) // 40)
    if (
        pareto_size > 0
        and pareto_size <= collapse_threshold
        and int(plateau_streak) >= 2
    ):
        return True
    if _should_viability_recovery(
        stage_params,
        valid_count=valid_count,
        plateau_streak=plateau_streak,
    ):
        return True
    return False
```

**Problem:** The function already has a `pareto_size` parameter and checks for small Pareto fronts, but the threshold is `pop_size // 40` (e.g., 200 // 40 = 5). This is too lenient — logs show `pareto=1` for 7+ consecutive generations with `pop_unique=1.00` (all 200 chromosomes genetically unique but phenotypically identical). The diversity recovery never fires because `population_unique_ratio=1.00` passes the first check.

**Fix:** Add an explicit phenotype-collapse trigger that fires when `pareto_size <= 3` and `plateau_streak >= 2`, regardless of genetic uniqueness:

```python
def _should_inject_diversity_recovery(
    population_unique_ratio: float,
    stage_params: Phase2StageParams | None = None,
    *,
    pareto_size: int = 0,
    plateau_streak: int = 0,
    pop_size: int = 0,
    valid_count: int = 0,
) -> bool:
    if not bool(getattr(_cfg, "PHASE2_DIVERSITY_RECOVERY_ENABLED", True)):
        return False
    
    # Check 1: Genetic uniqueness collapse
    if (
        population_unique_ratio
        < _diversity_recovery_min_unique_ratio(stage_params)
    ):
        return True
    
    # Check 2: Pareto front collapse (existing logic)
    collapse_threshold = max(2, int(pop_size) // 40)
    if (
        pareto_size > 0
        and pareto_size <= collapse_threshold
        and int(plateau_streak) >= 2
    ):
        return True
    
    # Check 3: Phenotype collapse (NEW) — Pareto front ≤3 despite high genetic uniqueness
    # This catches the case where all 200 chromosomes are genetically unique
    # but phenotypically identical (same trading behavior, different gene encoding)
    if pareto_size > 0 and pareto_size <= 3 and int(plateau_streak) >= 2:
        return True
    
    # Check 4: Viability recovery
    if _should_viability_recovery(
        stage_params,
        valid_count=valid_count,
        plateau_streak=plateau_streak,
    ):
        return True
    
    return False
```

### 3. Update Call Sites

**Location:** `gpu_fuzzy_trader/evolution/evox_runner.py` → `_run_nsga3()` and `_run_nsga2_fallback()`

Both functions call `_should_inject_diversity_recovery()` and need to pass the current `len(pareto_indices)` as the `pareto_size` parameter.

**Verification:** The call already passes `pareto_size=len(pareto_indices)`, so no change needed here. Just verify both `_run_nsga3` and `_run_nsga2_fallback` pass this parameter.

## Acceptance Criteria

1. `_trim_global_metrics_cache` uses FIFO eviction (first-inserted keys removed first)
2. `_should_inject_diversity_recovery` triggers when `pareto_size <= 3 and plateau_streak >= 2`
3. Both `_run_nsga3` and `_run_nsga2_fallback` pass `pareto_size=len(pareto_indices)` to `_should_inject_diversity_recovery`
4. Existing tests pass: `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/ -x -q`
5. New test for phenotype-collapse trigger (verify it fires when `pareto_size=2, plateau_streak=3, pop_unique_ratio=1.0`)

## Verification Commands

```bash
# Run tests
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/ -x -q

# Verify FIFO eviction (manual check)
grep -A 10 "def _trim_global_metrics_cache" gpu_fuzzy_trader/evolution/evox_runner.py

# Verify phenotype trigger (manual check)
grep -A 5 "Check 3: Phenotype collapse" gpu_fuzzy_trader/evolution/evox_runner.py
```

## Rationale

### Why FIFO instead of random?

With 200 pop × 2 evals/gen = 400 sims/gen and cache size 1200, the cache fills in ~3 generations. Random eviction means:
- Gen 1: Evaluate 200 parents, cache them
- Gen 2: Evaluate 200 offspring, cache them (cache now has 400 entries)
- Gen 3: Evaluate 200 parents, cache them (cache now has 600 entries)
- Gen 4: Evaluate 200 offspring, cache them (cache now has 800 entries)
- Gen 5: Evaluate 200 parents, cache them (cache now has 1000 entries)
- Gen 6: Evaluate 200 offspring, cache them (cache now has 1200 entries, at capacity)
- Gen 7: Evaluate 200 parents → random eviction removes ~200 entries, likely including gen 1-2 parents that are still useful for offspring crossover

FIFO eviction removes the oldest entries (gen 1-2), preserving recent evaluations (gen 5-6) that are more likely to be reused in gen 7 offspring.

### Why phenotype collapse at pareto_size ≤ 3?

Logs show multiple epochs with `pareto=1` for 7+ consecutive generations:
```
gen 4/15: pareto=1 mean_return=15.28% ... pop_viable=199
gen 5/15: pareto=1 mean_return=15.28% ... pop_viable=199
...
gen 10/15: pareto=1 mean_return=15.28% ... pop_viable=200
```

This means 199 out of 200 chromosomes are dominated by one rule. The `pop_unique=1.00` (all genetically unique) is misleading — they're phenotypically identical (same trading behavior, different gene encoding due to sparse slots).

Setting the threshold at `pareto_size <= 3` catches this collapse while allowing normal Pareto front diversity (typically 5-20 members). The `plateau_streak >= 2` requirement prevents false triggers during early transient convergence.

## Notes

- This is a medium-risk task (logic changes in evolution loop)
- The FIFO change is low-risk (just changes eviction order)
- The phenotype trigger is medium-risk (new logic path, but well-bounded)
- Both changes should be tested on next Colab run to validate cache hit rates and Pareto front diversity
