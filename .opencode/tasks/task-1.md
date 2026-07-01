# Task 1: Config Parameter Tuning (Runtime + OOS)

**ID:** task-1
**Branch:** `feat/phase2-config-tuning`
**Files:** `gpu_fuzzy_trader/config.py`
**Risk:** Low (parameter-only changes, no logic changes)

## Description

Update Phase 2 configuration parameters to address 7 critical and 5 moderate issues identified during the latest Colab run analysis. The run took ~3 hours and produced only 1 surviving rule. Target: ~60% runtime reduction + improved OOS generalization.

## Changes Required

### Runtime Reduction Parameters
```python
PHASE2_GENERATIONS = 100                    # was 132, diminishing returns past 100
PHASE2_ISLAND_TOTAL_GENERATIONS = PHASE2_GENERATIONS  # stays linked
PHASE2_ISLAND_EPOCH_GENERATIONS = 25        # was 15, fewer epoch rebuilds (~40% overhead reduction)
PHASE2_PLATEAU_POST_RESTART_STOP_PATIENCE = 5  # was 3, less aggressive early stop
PHASE2_ISLAND_PLATEAU_POST_RESTART_STOP_PATIENCE = 5  # was 3
PHASE2_PLATEAU_POST_RESTART_BOOST_GENS = 4  # was 3, more boost time before evaluation
PHASE2_PLATEAU_EARLY_STOP_MIN_DELTA_PCT = 1.0  # was 0.5, require meaningful improvement
PHASE2_MIGRATION_ENABLED = False            # was True, overhead without benefit (per config comment)
PHASE2_EVAL_GLOBAL_CACHE_MAX_SIZE = 1200    # was 575, prevent premature eviction
PHASE2_VAL_SIM_INTERVAL = 2                 # was 3, more frequent archive updates
```

### OOS Improvement Parameters
```python
PHASE2_JOINT_TRAIN_VAL = True               # was False, anti-overfit via min(train,val) fitness
PHASE2_F3_OBJECTIVE = "cv_fold_min"         # was "profit_factor", worst-case CV fold return
PHASE2_DIVERSITY_PENALTY = 2.0              # was 0.5, prevent phenotype collapse
PHASE2_PHENOTYPE_SORTINO_STEP = 0.15        # was 0.3, finer behavioral buckets
PHASE2_PHENOTYPE_F3_STEP = 2.0              # was 5.0, finer f3 buckets
PHASE2_MUTATION_RATE = 0.35                 # was 0.3, more exploration
PHASE2_MIN_PROFITABLE_SYMBOLS = 5           # was 4, broader cross-symbol edge
```

### Pool Admission Fixes
```python
PHASE2_MONTHLY_ADMISSION_MIN_RATIO = 0.5    # was 0.667, island-friendly threshold
PHASE2_MONTHLY_ADMISSION_MIN_PROFITABLE_RATIO = 0.4  # was 0.5
```

### Orphan Boost Disable
```python
PHASE2_ORPHAN_ENABLED = False               # was True, consistently fails with viability collapse
```

## Acceptance Criteria

1. All parameter values updated in `config.py` to the values listed above
2. All inline comments/docstrings updated to reflect new values and rationale
3. No logic changes in any other file (this is parameter-only)
4. Existing tests pass: `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/ -x -q`
5. Config imports cleanly: `.venv/bin/python -c "from gpu_fuzzy_trader import config; print(config.PHASE2_GENERATIONS)"` returns `100`

## Verification Commands

```bash
# Verify config imports
.venv/bin/python -c "from gpu_fuzzy_trader import config; print('GENERATIONS:', config.PHASE2_GENERATIONS); print('JOINT:', config.PHASE2_JOINT_TRAIN_VAL); print('F3:', config.PHASE2_F3_OBJECTIVE)"

# Run tests
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/ -x -q
```

## Rationale

### Why these changes?

**Runtime Issues:**
- **Excessive epoch count**: 15-gen epochs caused 10+ epoch starts with ~15s engine rebuild overhead each
- **Aggressive post-restart stop**: patience=3 killed epochs before restart could recover
- **Migration overhead**: Enabled but config comment says it "degraded locally-adapted elites"
- **Cache too small**: 575 entries with 200 pop × 2 evals/gen = cache fills in 1.5 gens, random eviction destroys useful results
- **Tiny final epochs**: Adaptive budget shrinking created 1-gen epochs (useless but still ~30s overhead)

**OOS Issues:**
- **Overfitting**: `JOINT_TRAIN_VAL=False` means f3 uses train-only profit_factor, no val signal
- **Weak diversity**: `DIVERSITY_PENALTY=0.5` is negligible vs 50+ infeasible penalties, allowing phenotype collapse
- **Coarse buckets**: Sortino step 0.3 and f3 step 5.0 create too few behavioral buckets, rules cluster
- **Low mutation**: 0.3 insufficient to escape local optima in 100 gens
- **Strict monthly gate**: 0.667 ratio rejects island-evolved rules trained on 3-4 symbols (noisier monthly windows)
- **Orphan boost failure**: Symbol '7' orphan run hit 3 viability collapses in 10 gens, wasted ~180s

## Notes

- This is a low-risk task (parameter-only, no logic changes)
- The previous plan (Priority A) is complete and merged
- These changes should be tested on next Colab run to validate runtime reduction
