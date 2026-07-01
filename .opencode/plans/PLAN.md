# Plan: Phase 2 Runtime & OOS Quality Overhaul

**Created:** 2026-07-01
**Status:** active
**base_branch:** `main`
**branch_policy:** isolated
**execution_mode:** checkpoint

## Goal

Fix 7 critical issues and 5 moderate issues identified in the Phase 2 long-direction run that took ~3 hours and produced only 1 surviving rule. Target: **~60% runtime reduction** + **improved OOS generalization** through better fitness objectives, diversity mechanisms, and pool admission gates.

## Context

The previous plan (Priority A runtime fixes) is complete and merged. This plan addresses the deeper algorithmic and configuration issues found during the latest Colab run analysis.

## Tasks

### Task 1: Config Parameter Tuning (Runtime + OOS)
**Branch:** `feat/phase2-config-tuning`
**Files:** `gpu_fuzzy_trader/config.py`
**Risk:** Low (parameter-only changes, no logic changes)

Changes:
- **Runtime reduction:**
  - `PHASE2_GENERATIONS = 100` (was 132, diminishing returns past 100)
  - `PHASE2_ISLAND_TOTAL_GENERATIONS = PHASE2_GENERATIONS` (stays linked)
  - `PHASE2_ISLAND_EPOCH_GENERATIONS = 25` (was 15, fewer epoch rebuilds)
  - `PHASE2_PLATEAU_POST_RESTART_STOP_PATIENCE = 5` (was 3, less aggressive early stop)
  - `PHASE2_ISLAND_PLATEAU_POST_RESTART_STOP_PATIENCE = 5` (was 3)
  - `PHASE2_PLATEAU_POST_RESTART_BOOST_GENS = 4` (was 3, more boost time)
  - `PHASE2_PLATEAU_EARLY_STOP_MIN_DELTA_PCT = 1.0` (was 0.5, require meaningful improvement)
  - `PHASE2_MIGRATION_ENABLED = False` (was True, overhead without benefit per config comment)
  - `PHASE2_EVAL_GLOBAL_CACHE_MAX_SIZE = 1200` (was 575, prevent premature eviction)
  - `PHASE2_VAL_SIM_INTERVAL = 2` (was 3, more frequent archive updates)

- **OOS improvement:**
  - `PHASE2_JOINT_TRAIN_VAL = True` (was False, anti-overfit via min(train,val) fitness)
  - `PHASE2_F3_OBJECTIVE = "cv_fold_min"` (was "profit_factor", worst-case CV fold)
  - `PHASE2_DIVERSITY_PENALTY = 2.0` (was 0.5, prevent phenotype collapse)
  - `PHASE2_PHENOTYPE_SORTINO_STEP = 0.15` (was 0.3, finer behavioral buckets)
  - `PHASE2_PHENOTYPE_F3_STEP = 2.0` (was 5.0, finer f3 buckets)
  - `PHASE2_MUTATION_RATE = 0.35` (was 0.3, more exploration)
  - `PHASE2_MIN_PROFITABLE_SYMBOLS = 5` (was 4, broader cross-symbol edge)

- **Pool admission fixes:**
  - `PHASE2_MONTHLY_ADMISSION_MIN_RATIO = 0.5` (was 0.667, island-friendly)
  - `PHASE2_MONTHLY_ADMISSION_MIN_PROFITABLE_RATIO = 0.4` (was 0.5)

- **Orphan fix:**
  - `PHASE2_ORPHAN_ENABLED = False` (was True, consistently fails with viability collapse)

**Acceptance criteria:**
- All parameter values updated in config.py
- All docstrings/comments updated to reflect new values and rationale
- No logic changes in any other file
- Existing tests pass with `PYTEST_LOW_MEMORY=1`

---

### Task 2: EvoX Runner Code Fixes (Cache + Diversity)
**Branch:** `feat/phase2-evox-fixes`
**Files:** `gpu_fuzzy_trader/evolution/evox_runner.py`
**Risk:** Medium (logic changes in evolution loop)

Changes:
1. **LRU cache trimming** — Replace random eviction with FIFO (dict insertion order) in `_trim_global_metrics_cache()`. Random eviction was destroying useful cached results, causing near-zero cache hit rates.

2. **Phenotype-collapse recovery trigger** — Add a new trigger in `_should_inject_diversity_recovery()` that fires when the Pareto front collapses to ≤3 members despite high genetic uniqueness. This addresses the "pareto=1 for 7+ generations" problem where all 200 chromosomes are genetically unique but phenotypically identical.

**Acceptance criteria:**
- `_trim_global_metrics_cache` uses FIFO eviction (first-inserted keys removed first)
- `_should_inject_diversity_recovery` accepts `pareto_size` parameter and triggers when `pareto_size <= 3 and plateau_streak >= 2`
- The new trigger is called from `_run_nsga3` and `_run_nsga2_fallback` with the current `len(pareto_indices)`
- Existing tests pass with `PYTEST_LOW_MEMORY=1`
- New test for phenotype-collapse trigger

---

### Task 3: Island Scheduler + Pool Admission Fixes
**Branch:** `feat/phase2-island-fixes`
**Files:** `gpu_fuzzy_trader/phases/phase2_island_scheduler.py`, `gpu_fuzzy_trader/phases/phase2_rule_pool.py`, `gpu_fuzzy_trader/evolution/evox_runner.py`
**Risk:** Medium (logic changes in scheduling and pool admission)

Changes:
1. **Minimum epoch size guard** in `_run_cluster_islands()` — Skip epochs with remaining < 5 generations (useless 1-gen epochs that waste ~30s on engine rebuild). Set `_island_generations_done = gens_per_cluster` to exit the loop cleanly.

2. **Monthly gate island-scoped ratio** in `finalize_island()` — Use `self.island_hyperparams.monthly_admission_min_ratio` (if available) instead of the global `PHASE2_MONTHLY_ADMISSION_MIN_RATIO` for the monthly admission gate.

3. **Logging patience fix** in `_run_nsga3()` and `_run_nsga2_fallback()` — The plateau early-stop log message should use the same patience value as the decision logic (island-scoped patience, not global default).

**Acceptance criteria:**
- Epochs with remaining < MIN_EPOCH_GENS (5) are skipped in `_run_cluster_islands`
- `finalize_island` monthly gate uses island-scoped ratio when `island_hyperparams` is set
- Log messages show the actual patience value used in the decision
- Existing tests pass with `PYTEST_LOW_MEMORY=1`
- New test for min epoch size guard

---

## Verification

After all tasks are merged:
1. Run `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/ -x -q` to verify no regressions
2. Verify config values with a quick import check
3. Next Colab run should show:
   - Fewer epoch starts (larger epoch size)
   - Higher cache hit rates (LRU + larger cache)
   - Better Pareto front diversity (phenotype trigger + stronger penalty)
   - More rules surviving monthly gate (lower ratio threshold)
   - Shorter total runtime (~60% target)
