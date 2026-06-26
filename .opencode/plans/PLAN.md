# Plan — Phase 2 Island Plateau State-Leakage & OOS Fixes

## Goal

Eliminate the premature-convergence cascade in Phase 2 island mode that
produced poor OOS results on `test.csv`. Fixes target six diagnosed issues
(A–F) plus config tuning, across five isolated feature branches executed
sequentially with auto-continue.

## Diagnosis (verified in code)

- **Issue A (critical):** `phase2_rule_pool.py:2672` `reset_plateau = entering_stage_b`,
  and `entering_stage_b` is always `False` when `PHASE2_ISLAND_TWO_STAGE_ENABLED=False`
  (`config.py:881`). Plateau state carries across epochs via `Phase2EvolutionState`
  (`evox_runner.py:1798-1799`). Every epoch after the first starts with
  `plateau_streak=5–17` and dies at `min_gen=3`.
- **Issue B:** `phase2_rule_pool.py:2765` `self._island_generations_done += epoch_gens`
  charges requested gens (10), not actual gens run (3 after early-stop).
- **Issue C:** `config.py:570` `PHASE2_JOINT_TRAIN_VAL=True` folds the holdout
  `val` (209590 rows) into Phase 2 fitness via `robust_return_pct=min(train,val)`
  (`phase2_support.py:314-330`). Val is the model-selection holdout and must not
  drive evolution fitness. `test.csv` is the true OOS (Phase 5).
- **Issue D:** On intra-epoch plateau, the loop `break`s immediately
  (`evox_runner.py:2138-2155`) while `pop_unique=1.00` but `max_return` frozen at
  identical values → frozen-elite attractor. `_inject_diversity_recovery` exists
  but isn't used to extend the run, only reinit on collapse.
- **Issue E:** K-means produced `{'0':[9], '1':[1,2,6], '2':[3,4,5,7,8,10]}`
  (`symbol_cluster.py:105-112`). 1-symbol cluster overfits; 6-symbol cluster
  starved by `min_profitable_symbols=3` (deployable=0 most epochs).
- **Issue F:** Banner advertises `gen=132` but actual per-run gen is 10/4 — misleading.

## Ordered Tasks

### task-1 — `fix/plateau-state-leak` (Fixes A + B) — CRITICAL
Reset plateau per epoch so each epoch evaluates its own improvement; charge actual
gens run to the island budget. Unblocks measuring all other fixes.
**Files:** `phase2_rule_pool.py`, `evox_runner.py` (state invariant), tests.
**Depends on:** none.

### task-2 — `fix/holdout-fitness-leak` (Fix C)
Set `PHASE2_JOINT_TRAIN_VAL=False` so evolution fitness is train-only (robustness
from the already-wired purged 4-fold CV). Verify `val_df` is used in Phase 3
selection only (proper holdout use), NOT in Phase 2 fitness. Optional: skip val
simulation in Phase 2 when joint is off (GPU saving).
**Files:** `config.py`, `phase2_support.py` (optional skip), README, tests.
**Depends on:** none (independent config change, but sequenced after task-1 merge).

### task-3 — `fix/diversity-restart-on-plateau` (Fix D)
On first intra-epoch plateau, inject a diversity restart (reinit 30-50% of pop
keeping Pareto elite, bump mutation one gen) and continue; break only on a second
plateau. New config keys for restart fraction + max restarts.
**Files:** `evox_runner.py`, `config.py`, tests.
**Depends on:** task-1 (per-epoch plateau makes restart meaningful).

### task-4 — `fix/cluster-balancing` (Fix E)
Balanced K-means (min symbols/cluster, or balanced assignment) so 10 symbols split
~3/3/4 not 1/3/6. Scale `min_profitable_symbols` with cluster size; route lone
single-symbol clusters through orphan-boost instead of primary island.
**Files:** `features/symbol_cluster.py`, `config.py`, tests.
**Depends on:** none.

### task-5 — `fix/plateau-config-tuning-and-banner` (Fix F + config)
Banner shows `island_total/per_cluster/epoch`. Config: patience 5→8, min_gen 3→6,
min_delta 0.02→0.05, epoch_gens 10→15. Built on corrected behavior from tasks 1-4.
**Files:** `run_pipeline.py`, `config.py`, README.
**Depends on:** tasks 1-4 (tune against fixed behavior).

## Verification (per task)

- `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit -q` passes.
- Targeted regression tests added per task.
- No revert of prior migration/elite/early-stop work.
- `evaluator_v5.ipynb` untouched.

## Out of Scope (deferred)

- Re-enabling `PHASE2_ISLAND_TWO_STAGE_ENABLED` (separate evaluation round).
- Full pipeline run (Colab only).
