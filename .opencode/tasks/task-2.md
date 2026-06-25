# task-2: Elite preservation under (μ+λ) selection

**Branch:** `fix/elite-preservation`
**Depends on:** task-1 (merged to main)

## Goal

Prevent mid-epoch elite erosion: a non-dominated rule present at generation N must survive to generation N+k unless a genuinely Pareto-dominating rule appears. This applies in both island and global mode.

## Root cause

Recomputed dynamic diversity/support penalties drift upward as `hall_of_fame` and `pareto_archive` grow. Under (μ+λ) NSGA-III selection, a non-dominated elite can be evicted purely by penalty growth — not because a better rule was found.

## Changes

### 1. `gpu_fuzzy_trader/config.py` — new elite preservation block

Add after `PHASE2_DEPLOYABLE_ARCHIVE_MAX_SIZE` (~line 868):

```python
# --- elite-preservation guard (prevents mid-epoch erosion) ---
PHASE2_ELITE_PRESERVATION_ENABLED: bool = True
PHASE2_ELITE_PRESERVATION_TOP_K: int = 5
PHASE2_ELITE_PRESERVATION_MIN_GEN: int = 1
```

### 2. `gpu_fuzzy_trader/evolution/evox_runner.py` — `_preserve_deployable_elites` helper

Add a new function (before `run_phase2_evolution`):

```python
def _preserve_deployable_elites(
    state,
    cfg,
    current_gen: int,
):
    """Force-preserve top-K deployable-archive elites in the live population.
    
    Guarantees a non-dominated elite at gen N survives to gen N+k unless
    a genuinely Pareto-dominating rule appears — preventing mid-epoch
    erosion from recomputed dynamic penalties (diversity/support drift).
    """
    if not cfg.PHASE2_ELITE_PRESERVATION_ENABLED:
        return
    if current_gen < cfg.PHASE2_ELITE_PRESERVATION_MIN_GEN:
        return
    archive = getattr(state, "deployable_archive", None)
    if not archive:
        return
    
    top_k = min(cfg.PHASE2_ELITE_PRESERVATION_TOP_K, len(archive))
    if top_k == 0:
        return
    
    # Rank deployable_archive by rank_score desc, take top-K
    sorted_elites = sorted(
        archive.values() if isinstance(archive, dict) else archive,
        key=lambda e: (getattr(e, "rank_score", 0) or 0),
        reverse=True,
    )[:top_k]
    
    pop_size = state.population.shape[0]
    
    for elite_entry in sorted_elites:
        chrom = elite_entry.chromosome if hasattr(elite_entry, "chromosome") else elite_entry
        if not isinstance(chrom, np.ndarray):
            chrom = np.array(chrom, dtype=state.population.dtype)
        
        # Check if already present (by exact chromosome match)
        already_present = False
        for i in range(pop_size):
            if np.array_equal(state.population[i], chrom):
                already_present = True
                break
        if already_present:
            continue
        
        # Evict the most-crowded survivor (least-crowded = last in crowding sort)
        # Use the existing _build_rank_and_crowding helper
        ranks, crowding = _build_rank_and_crowding(state.objectives)
        # Find the worst (highest rank, or lowest crowding within highest rank)
        max_rank = int(np.max(ranks))
        worst_idx = -1
        worst_crowding = float("inf")
        for i in range(pop_size):
            if ranks[i] == max_rank:
                if crowding[i] < worst_crowding:
                    worst_crowding = crowding[i]
                    worst_idx = i
        if worst_idx == -1:
            worst_idx = pop_size - 1  # fallback: replace last
        
        # Overwrite the slot
        state.population[worst_idx] = chrom.copy()
        state.objectives[worst_idx] = np.full(state.objectives.shape[1], np.inf)
        if hasattr(state, "metrics_cache") and state.metrics_cache is not None:
            state.metrics_cache[worst_idx] = {}
```

### 3. Wire into `run_phase2_evolution` (NSGA-III path)

In `run_phase2_evolution`, right after the `_nsga3_environmental_selection` call (~line 2177), before `metrics_cache = [merge_metrics[int(i)] for i in sel_idx[:n_alive]]`, add:

```python
# --- elite preservation ---
_preserve_deployable_elites(state, cfg, gen)
```

### 4. Wire into `_run_nsga2_fallback` (NSGA-II path)

In `_run_nsga2_fallback`, after the environmental selection step (find the corresponding point after `merge_pop` is trimmed to `pop_size`), add the same call.

### 5. `README.md` — §5.2 evolution table

Add rows for the three new config keys, plus a note that elite preservation prevents mid-epoch erosion under growing archives.

## Acceptance criteria

- **AC-T2.1**: Build a `Phase2EvolutionState` with 20 unique chromosomes, place one "champion" in `deployable_archive` with high `rank_score`, run the selection+preservation step for 15 generations with a growing `hall_of_fame` (simulating penalty drift). Assert the champion is present in `state.population` at every generation. Without the fix (disabled), assert the champion is evicted by gen ~8.

- **AC-T2.2**: Preservation never exceeds `PHASE2_ELITE_PRESERVATION_TOP_K` slots and never evicts a chromosome that is itself in the top-K of the live Pareto front (rank 1 members).

- **AC-T2.3**: With `PHASE2_ELITE_PRESERVATION_ENABLED=False`, evolution loop byte-for-byte identical to pre-task behavior (snapshot test: 2-gen run with fixed seed → identical `history`).

- **AC-T2.4**: Preserved elite's `objectives` are reset to `inf` (forces re-eval with current penalties) — assert no stale objectives survive across generations.

## Verification

```bash
cd /home/danaee/trading_platform && source .venv/bin/activate && \
  PYTEST_LOW_MEMORY=1 python -m pytest \
    tests/unit/test_elite_preservation.py \
    tests/unit/test_evox_runner.py -x -q --tb=short
```

## Files to modify

- `gpu_fuzzy_trader/config.py` — 3 new keys
- `gpu_fuzzy_trader/evolution/evox_runner.py` — new helper + 2 call sites
- `README.md` — config table update
- `tests/unit/test_elite_preservation.py` — new test file
