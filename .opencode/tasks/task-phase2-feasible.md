# Phase 2 feasible-search fixes (items 1–4)

Implement all four on branch `feature/phase2-feasible-search-1-4`.

## task-1: Evolution PF floors
In `gpu_fuzzy_trader/phases/phase2_support.py`:
- Add `_evolution_feasibility_floors(n_valid_rows)` same shape as `_pool_admission_floors` but `pf_floor = PHASE2_PROFIT_FACTOR_FLOOR_EVOLUTION` (1.05).
- `_raw_feasibility_violation_score` and `_feasibility_gate_failures` must use **evolution** floors.
- `_passes_pool_admission_impl` / `_pool_admission_floors` stay on **ADMISSION** PF 1.15.
- Update docstrings: evolution preview ≠ pool admission.
- Tests: assert evolution violation fails at PF 1.05 boundary vs admission still 1.15.

## task-2: Island min_profitable
In `resolve_island_hyperparams` (config.py), cluster branch:
```python
# was: max(3, (sym_n + 1) // 2)
scaled = max(2, (sym_n + 1) // 2)  # 3-sym → 2
```
Update `tests/unit/test_phase2_island_hyperparams.py` (n=3,4 expect 2 not 3; n=5 still 3; etc.)

## task-3: Corr hybrid clustering
In `features/symbol_cluster.py`:
- When `PHASE2_CLUSTER_USE_RETURN_CORR` True, build return series per symbol (prefer `close`, else pct-change of `label_open_next` or first available price-like col), pairwise Pearson corr, use corr-matrix rows as embedding.
- Blend with feature means: weights from config FEATURE_WEIGHT and CORR_WEIGHT (normalize if sum≠1).
- Keep balanced KMeans path.
- method = `hybrid_corr_v1` when corr on, else `hybrid_v1`.
Config knobs + comments in config.py near island section.

## task-4: VAL_IN_FITNESS_PENALTY=False
config.py default False + comment. Fix tests pinning True.

## Constraints
- .venv, PYTHONPATH=., PYTEST_LOW_MEMORY=1
- No evaluator_v5, no full pipeline, no outputs/
- Do not set PHASE1_DISABLED
- Commit on feature branch; write handoff `.opencode/handoffs/phase2-feasible-1-4-implementer.json`
