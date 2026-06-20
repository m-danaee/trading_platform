# Plan: Phase 2 Speed Optimizations

## Goal
Reduce Phase 2 runtime by 55-70% through early stopping, parallel CV folds,
GPU batch increase, and larger epochs. No quality regression expected.

## Tasks

### task-1: Early stopping at plateau=5
**File:** `gpu_fuzzy_trader/evolution/evox_runner.py`
Add plateau-based early termination. If Pareto front unchanged for 5 consecutive
generations, stop island evolution early.

### task-2: Parallel CV fold evaluation
**File:** `gpu_fuzzy_trader/phases/phase2_rule_pool.py`
Use ThreadPoolExecutor in CvFoldValEvaluator.simulate_rule_batch to run
both folds simultaneously.

### task-3: GPU batch size and epoch size
**File:** `gpu_fuzzy_trader/config.py`
Increase PHASE2_GPU_BATCH_SIZE (64→128) and PHASE2_ISLAND_EPOCH_GENERATIONS (10→25).
