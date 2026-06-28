# Task-19: Cleanup & Observability

**Branch:** `fix/cleanup-observability`
**Priority:** 🟢 Low
**Fixes:** L1, L2, L3, L4, L5, L6
**Depends on:** task-16 merged (shared `evox_runner.py`)

## Problem

Dead code (`_agent_debug_log`), silent EvoX fallback, broken restart logging, unused viability signal, misleading API (`_sample_df` random_state), non-uniform reference vector fallback. Per `AGENTS.md`: "remove additional/wasted parts from old implementation to keep project clean."

## Files to Modify

1. `gpu_fuzzy_trader/evolution/evox_runner.py` — L1, L2, L3, L4, L6
2. `gpu_fuzzy_trader/phases/phase2_rule_pool.py` — L5
3. `gpu_fuzzy_trader/config.py` — L4 config flags

## Detailed Changes

### L1: Remove dead `_agent_debug_log`

**evox_runner.py (lines ~58–66):**
- Delete the entire `_agent_debug_log` function.
- Grep for any call sites and remove them (likely none — it's a no-op `return`).
```bash
grep -rn "_agent_debug_log" gpu_fuzzy_trader/
```

### L2: Warn if EvoX unavailable

**evox_runner.py (lines ~75–85):**
- Add module-level warning flag:
```python
_EVOX_WARNED = False

def _warn_evox_unavailable():
    global _EVOX_WARNED
    if not _EVOX_AVAILABLE and not _EVOX_WARNED:
        logger.warning(
            "EvoX unavailable (%s); falling back to NSGA-II. "
            "Pipeline config may specify NSGA3 — verify evox/torch installation.",
            _EVOX_IMPORT_ERROR,
        )
        _EVOX_WARNED = True
```
- Call `_warn_evox_unavailable()` at the start of `_run_nsga2_fallback` and `_nsga3_environmental_selection` (the two paths that depend on `_EVOX_AVAILABLE`).

### L3: Fix plateau restart log

**evox_runner.py — wherever plateau restart is triggered:**
- BEFORE: `logger.info("Phase 2 [%s]: plateau restart at gen %d (restart 1/1, reinit 40%%, elite_kept=5)", ...)`
- AFTER:
```python
logger.info(
    "Phase 2 [%s]: plateau restart at gen %d (restart %d/%d, reinit %.0f%%, elite_kept=%d, mutation=%.2f)",
    tag, gen + 1, restart_count, max_restarts,
    100.0 * float(_cfg.PHASE2_PLATEAU_DIVERSITY_RESTART_FRACTION),
    n_elite_kept, effective_mutation_rate,
)
```
- Add `restart_count` and `max_restarts` to the generation log line:
```python
logger.info(
    "Phase 2 [%s] gen %d/%d: ... restarts=%d/%d ...",
    tag, gen + 1, n_generations, ..., restart_count, max_restarts,
)
```

### L4: Add viability-collapse trigger

**config.py — add:**
```python
PHASE2_VIABILITY_COLLAPSE_THRESHOLD = 0.5  # pop_viable < 0.5*pop_size = collapse
PHASE2_VIABILITY_COLLAPSE_STREAK = 3       # 3 gens of collapse → forced restart
```

**evox_runner.py — main loop (both NSGA-II fallback and EvoX paths):**
- Add `viability_collapse_streak` counter (local to the loop, reset on restart).
- After computing `pop_viable`:
```python
viability_threshold = int(
    getattr(_cfg, "PHASE2_VIABILITY_COLLAPSE_THRESHOLD", 0.5) * pop_size
)
if pop_viable < viability_threshold:
    viability_collapse_streak += 1
else:
    viability_collapse_streak = 0

viability_collapse_patience = int(
    getattr(_cfg, "PHASE2_VIABILITY_COLLAPSE_STREAK", 3)
)
if (
    viability_collapse_streak >= viability_collapse_patience
    and restart_count < max_restarts
):
    n_elite_kept = _plateau_diversity_restart(...)
    restart_count += 1
    viability_collapse_streak = 0
    post_restart_gens_remaining = int(
        getattr(_cfg, "PHASE2_PLATEAU_POST_RESTART_BOOST_GENS", 3)
    )
    logger.info(
        "Phase 2 [%s]: viability-collapse restart at gen %d (pop_viable=%d < %d, streak=%d)",
        tag, gen + 1, pop_viable, viability_threshold, viability_collapse_streak,
    )
```

### L5: Clean up `_sample_df` random_state docstring

**phase2_rule_pool.py::_sample_df:**
- BEFORE:
```python
def _sample_df(df, total_rows, random_state=None):
    """
    Sample up to *total_rows* rows from *df*, distributed equally across symbols.
    ...
    *random_state* is accepted for API compatibility but ignored.
    """
    del random_state
```
- AFTER: Keep param (callers pass it), improve docstring:
```python
def _sample_df(df, total_rows, random_state=None):
    """Sample up to *total_rows* rows, distributed equally across symbols.

    Rows are taken in chronological order per symbol (deterministic stride
    downsampling). Random sampling is intentionally avoided because backtest
    engines rely on row order and ``_symbol_bar_index`` for exposure release.

    Args:
        df: Input DataFrame.
        total_rows: Target row count.
        random_state: **Intentionally ignored.** Accepted only for API
            compatibility with callers that pass a seed. Sampling is
            chronologically deterministic; do not rely on this for
            reproducibility — set ``PHASE2_SEED`` upstream instead.
    """
    del random_state  # chronology-preserving sampling is deterministic
    ...
```

### L6: Proper Das-Dennis fallback for reference vectors

**evox_runner.py::_get_reference_vectors:**
- Replace the fallback (3 axis vectors + biased `[t, (1-t)/2, (1-t)/2]` fill) with proper Das-Dennis:
```python
def _das_dennis(n_partitions: int, n_objs: int) -> np.ndarray:
    """Das-Dennis reference vectors on the unit simplex."""
    from itertools import product
    points = []
    for combo in product(range(n_partitions + 1), repeat=n_objs):
        if sum(combo) == n_partitions:
            points.append([c / n_partitions for c in combo])
    return np.array(points, dtype=np.float64)

def _get_reference_vectors(pop_size, n_objs=3, rng=None):
    if _EVOX_AVAILABLE and uniform_sampling is not None:
        refs = uniform_sampling(pop_size, n_objs)[0].cpu().numpy()
        while len(refs) < pop_size:
            extra, _ = uniform_sampling(pop_size - len(refs), n_objs)
            refs = np.vstack([refs, extra.cpu().numpy()])
        return refs[:pop_size]

    # Das-Dennis fallback: increase partitions until >= pop_size
    n_partitions = n_objs  # start small
    while True:
        refs = _das_dennis(n_partitions, n_objs)
        if len(refs) >= pop_size:
            return refs[:pop_size]
        n_partitions += 1
        if n_partitions > 100:  # safety
            break
    # Pad with random simplex points if still short
    fallback_rng = rng if rng is not None else np.random.default_rng()
    while len(refs) < pop_size:
        r = fallback_rng.random(n_objs - 1)
        r = np.sort(r)
        last = np.concatenate([[r[0]], np.diff(r), [1 - r[-1]]])
        refs = np.vstack([refs, last])
    return refs[:pop_size]
```

## Acceptance Criteria

- [ ] `_agent_debug_log` deleted; no call sites remain (`grep -rn "_agent_debug_log" gpu_fuzzy_trader/` returns nothing).
- [ ] EvoX unavailability logged at `WARNING` level once per process.
- [ ] Plateau restart logs `restart {n}/{max}` with mutation rate and elite count.
- [ ] Generation log includes `restarts={n}/{max}`.
- [ ] Viability-collapse trigger implemented (3-gen streak → forced restart).
- [ ] `PHASE2_VIABILITY_COLLAPSE_THRESHOLD = 0.5`, `PHASE2_VIABILITY_COLLAPSE_STREAK = 3` config flags.
- [ ] `_sample_df` docstring clarifies `random_state` is intentionally ignored.
- [ ] `_get_reference_vectors` fallback uses Das-Dennis (uniform simplex coverage).
- [ ] No dead code remaining (per `AGENTS.md` cleanup rule).
- [ ] `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_evox_runner.py -x -q` passes.

## Verification

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/ -x -q -k "evox or reference_vector"
grep -rn "_agent_debug_log" gpu_fuzzy_trader/  # should return nothing
```
