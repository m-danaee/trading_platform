# PLAN: Phase 2 Feasible-Search Fixes (suggestions 1–4)

## Goal
Unblock Phase 2 evolution (non-empty feasible set) while keeping hard
anti-overfit gates at pool admission. Implement diagnosis items 1–4:
evolution PF vs admission PF, island min_profitable, corr clustering,
soften val-in-fitness penalties.

Do NOT touch `evaluator_v5.ipynb`. Targeted tests only + `PYTEST_LOW_MEMORY=1`.
Do not run full pipeline (OOM).

## Context
- base_branch: main
- branch_policy: isolated
- execution_mode: continuous (user: implement 1,2,3,4 without asking)

## Tasks

### task-1: Evolution feasibility uses EVOLUTION PF (not admission 1.15)
- Add `_evolution_feasibility_floors()` using
  `PHASE2_PROFIT_FACTOR_FLOOR_EVOLUTION` (1.05).
- Use it in `_raw_feasibility_violation_score` and
  `_feasibility_gate_failures` (evolution path / collapse log).
- Keep `_pool_admission_floors` + `_passes_pool_admission_impl` on
  **ADMISSION** PF 1.15 unchanged.
- Targets: `phases/phase2_support.py`, tests.
- Acceptance: evolution violation score uses 1.05; pool admission still 1.15.

### task-2: Island min_profitable_symbols = ceil-half (3-sym → 2)
- In `resolve_island_hyperparams` cluster branch: require
  `max(2, (n_symbols+1)//2)` instead of `max(3, (n_symbols+1)//2)`.
- Cap still by `PHASE2_MIN_PROFITABLE_SYMBOLS`.
- Update `test_phase2_island_hyperparams.py` expectations (3→2 for n=3,4).
- Targets: `config.py`, tests.
- Acceptance: 3-symbol island min_profitable=2; orphan still 1.

### task-3: Correlation-aware hybrid symbol clustering
- Extend `build_hybrid_symbol_clusters` to blend feature means with
  per-symbol return-correlation embedding (config-weighted).
- Config: `PHASE2_CLUSTER_USE_RETURN_CORR=True`,
  `PHASE2_CLUSTER_FEATURE_WEIGHT`, `PHASE2_CLUSTER_CORR_WEIGHT`.
- Keep balanced assignment (no 1-symbol clusters).
- method tag e.g. `hybrid_corr_v1` when corr enabled.
- Targets: `features/symbol_cluster.py`, `config.py`,
  `tests/unit/test_symbol_cluster.py`.
- Acceptance: still covers all symbols; balance K=3 on 10 syms ≥3 each;
  corr path exercised when flag True.

### task-4: Soften val-in-fitness penalty stack
- Set `PHASE2_VAL_IN_FITNESS_PENALTY = False` (val still used at admission).
- Document: with JOINT=False, val penalties during evolution starved search;
  hard gates remain at pool admission.
- Fix any tests pinning True.
- Targets: `config.py`, tests.

## Out of scope
- Global mode A/B (item 6)
- Return floor retune (item 5) unless needed later
- TP/SL retune, evaluator_v5, full pipeline

## Verification
```bash
source .venv/bin/activate
PYTHONPATH=. PYTEST_LOW_MEMORY=1 python -m pytest \
  tests/unit/test_phase2_island_hyperparams.py \
  tests/unit/test_symbol_cluster.py \
  tests/unit/test_anti_overfit_config.py \
  -q
# + new tests for evolution floors
```
