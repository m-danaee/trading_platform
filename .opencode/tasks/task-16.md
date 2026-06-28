# Task-16: Evolution Convergence Tuning

**Branch:** `fix/evolution-convergence`
**Priority:** 🟠 High
**Fixes:** H3, H5, M4, M5
**Depends on:** task-15 merged (shared function `compute_phase2_objectives_from_metrics`)

## Problem

Every island hits the same failure: `plateau restart at gen 9 (restart 1/1, reinit 40%, elite_kept=5)` then the *same* `max_return` reappears within 1–2 gens. 5 elites + 60% preserved pop = "restart in name only". Only 1 restart allowed. Plateau delta 0.05% is noise-level. Deployable archive + hall-of-fame persist across epochs → epoch 2/3 starts pre-converged. Min-max normalization fragile to `trade_penalty=50` outliers.

## Files to Modify

1. `gpu_fuzzy_trader/config.py` — plateau/restart config
2. `gpu_fuzzy_trader/evolution/evox_runner.py` — `_plateau_diversity_restart`, main loop, `_normalize_for_association`, `_inject_diversity_recovery`
3. `gpu_fuzzy_trader/phases/phase2_rule_pool.py` — `run_epoch` (state carry-over)

## Detailed Changes

### H3: Plateau restart — more restarts, more reinit, fewer elites, adaptive mutation

**config.py:**
```python
# Existing — change values:
PHASE2_PLATEAU_EARLY_STOP_MIN_DELTA_PCT = 0.5     # was 0.05 (noise-level)
PHASE2_PLATEAU_DIVERSITY_RESTART_FRACTION = 0.65  # was 0.40

# New flags:
PHASE2_PLATEAU_MAX_RESTARTS = 3
PHASE2_PLATEAU_POST_RESTART_MUTATION_BOOST = 0.45  # temp mutation for post-restart gens
PHASE2_PLATEAU_POST_RESTART_BOOST_GENS = 3         # gens to hold boosted mutation
```

**evox_runner.py::_plateau_diversity_restart:**
```python
# BEFORE: n_elite = min(5, max(1, len(pareto_indices)))
# AFTER:
n_elite = min(2, max(1, len(pareto_indices)))  # keep only top 2 elites
```

**evox_runner.py — main loop (both `_run_nsga2_fallback` and EvoX path):**
- Read `max_restarts = int(getattr(_cfg, "PHASE2_PLATEAU_MAX_RESTARTS", 3))` instead of hardcoded `1`.
- Add `post_restart_gens_remaining` counter to `Phase2EvolutionState`.
- After a restart:
  ```python
  state.post_restart_gens_remaining = int(
      getattr(_cfg, "PHASE2_PLATEAU_POST_RESTART_BOOST_GENS", 3)
  )
  ```
- In the mutation step:
  ```python
  if state.post_restart_gens_remaining > 0:
      effective_mutation_rate = float(
          getattr(_cfg, "PHASE2_PLATEAU_POST_RESTART_MUTATION_BOOST", 0.45)
      )
      state.post_restart_gens_remaining -= 1
  else:
      effective_mutation_rate = _stage_mutation_rate(stage_params)
  ```

**Add `post_restart_gens_remaining: int = 0` to `Phase2EvolutionState` dataclass.**

### H5: Epoch state carry-over — clear cache, cap HoF, diverse seeding

**phase2_rule_pool.py::run_epoch (~line 2645):**
- Before calling `run_phase2_evolution_epoch` on a *non-first* epoch (when seeds are being applied):
  ```python
  if not first_epoch and seed_chromosomes is not None:
      from gpu_fuzzy_trader.phases.phase2_sparse_encoding import chromosome_key
      seeded_keys = {chromosome_key(c) for c in seed_chromosomes}
      for key in list(self._evolution_state.global_metrics_cache.keys()):
          if key in seeded_keys:
              self._evolution_state.global_metrics_cache.pop(key, None)
  ```
- Cap hall-of-fame carry-over:
  ```python
  if not first_epoch and self._evolution_state.hall_of_fame:
      max_carry = int(getattr(_cfg, "PHASE2_HOF_EPOCH_CARRYOVER", 10))
      # Keep top-N by deployability rank (need to track rank — use pareto_archive order)
      # Simplest: keep first N entries (insertion order ≈ discovery order)
      keys = list(self._evolution_state.hall_of_fame.keys())[:max_carry]
      self._evolution_state.hall_of_fame = {
          k: self._evolution_state.hall_of_fame[k] for k in keys
      }
  ```

**config.py:** add `PHASE2_HOF_EPOCH_CARRYOVER = 10`

**evox_runner.py — diverse-subset seeding:**
- Add helper:
  ```python
  def _select_diverse_subset(
      chromosomes: list[np.ndarray], k: int,
  ) -> list[np.ndarray]:
      """Max-min Hamming diversity sampling: greedy pick farthest from chosen."""
      from gpu_fuzzy_trader.phases.phase2_sparse_encoding import chromosome_key
      
      if len(chromosomes) <= k:
          return list(chromosomes)
      chosen = [chromosomes[0]]
      remaining = list(chromosomes[1:])
      while len(chosen) < k and remaining:
          best_idx = max(
              range(len(remaining)),
              key=lambda i: min(_hamming_distance(remaining[i], c) for c in chosen),
          )
          chosen.append(remaining.pop(best_idx))
      return chosen
  ```
- In `_inject_diversity_recovery` and Stage B seeding paths: replace top-K selection with `_select_diverse_subset`.

### M4: Scale diversity Hamming threshold to chromosome length

**config.py:**
```python
PHASE2_DIVERSITY_HAMMING_THRESHOLD = 0   # 0 = auto-scale to max(3, K_active // 5)
PHASE2_DIVERSITY_HAMMING_THRESHOLD_AUTO = True
```

**phase2_rule_pool.py::_diversity_penalty_blended:**
```python
# BEFORE: diversity_hamming_threshold = int(stage_params.diversity_hamming_threshold) or int(_cfg.PHASE2_DIVERSITY_HAMMING_THRESHOLD)
# AFTER:
if bool(getattr(_cfg, "PHASE2_DIVERSITY_HAMMING_THRESHOLD_AUTO", True)):
    k_active = _count_active_conditions(chromosome, dont_cares)
    diversity_hamming_threshold = max(3, k_active // 5)
else:
    diversity_hamming_threshold = (
        int(stage_params.diversity_hamming_threshold)
        if stage_params is not None
        else int(_cfg.PHASE2_DIVERSITY_HAMMING_THRESHOLD)
    )
```

### M5: Rank-based normalization for NSGA-III association

**evox_runner.py::_normalize_for_association:**
```python
def _normalize_for_association(merge_fit, ref):
    """Rank-based normalization (robust to outliers like trade_penalty=50)."""
    fit = np.asarray(merge_fit, dtype=np.float64)
    fit = np.where(np.isfinite(fit), fit, 1e12)
    # Replace values with percentile ranks (0..1)
    n = len(fit)
    rank_fit = np.empty_like(fit)
    for j in range(fit.shape[1]):
        # Average rank for ties
        order = np.argsort(fit[:, j], kind="mergesort")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(1, n + 1, dtype=np.float64)
        # Handle ties: average ranks within equal-value groups
        # (simpler: scipy.stats.rankdata if available)
        try:
            from scipy.stats import rankdata
            rank_fit[:, j] = rankdata(fit[:, j], method="average") / n
        except ImportError:
            rank_fit[:, j] = ranks / n
    ref_n = ref / np.linalg.norm(ref, axis=1, keepdims=True).clip(1e-10)
    fit_n = rank_fit / np.linalg.norm(rank_fit, axis=1, keepdims=True).clip(1e-10)
    return fit_n, ref_n
```

## Acceptance Criteria

- [ ] `PHASE2_PLATEAU_MAX_RESTARTS = 3` in config; evox_runner reads it (no hardcoded `1`).
- [ ] `PHASE2_PLATEAU_EARLY_STOP_MIN_DELTA_PCT = 0.5`.
- [ ] `PHASE2_PLATEAU_DIVERSITY_RESTART_FRACTION = 0.65`.
- [ ] `PHASE2_PLATEAU_POST_RESTART_MUTATION_BOOST = 0.45`, `PHASE2_PLATEAU_POST_RESTART_BOOST_GENS = 3`.
- [ ] `post_restart_gens_remaining` field added to `Phase2EvolutionState`.
- [ ] Post-restart mutation boost applied for 3 gens, then anneals back.
- [ ] `n_elite = min(2, ...)` in `_plateau_diversity_restart`.
- [ ] `global_metrics_cache` cleared for seeded keys at new epoch.
- [ ] `PHASE2_HOF_EPOCH_CARRYOVER = 10`; hall-of-fame trimmed at epoch start.
- [ ] `_select_diverse_subset` (max-min Hamming) implemented and used.
- [ ] Hamming threshold auto-scales (`PHASE2_DIVERSITY_HAMMING_THRESHOLD_AUTO = True`).
- [ ] `_normalize_for_association` uses rank-based normalization.
- [ ] Tests pass: `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/ -x -q -k "plateau or evox or restart or diversity or normalize"`

## Verification

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/ -x -q -k "plateau or evox or restart or diversity or normalize"
```
