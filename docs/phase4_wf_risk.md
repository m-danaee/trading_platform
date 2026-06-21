# Phase 4 — Walk-forward risk optimizer (legacy)

Tunes TP / SL / capital per rule on validation walk-forward folds. **Skipped** when `RB_GOVERNOR_ENABLED=True` (default).

**Source code:**
- WF optimizer: [`gpu_fuzzy_trader/phases/phase4_wf_optimizer.py`](../gpu_fuzzy_trader/phases/phase4_wf_optimizer.py)
- Cache: [`gpu_fuzzy_trader/phases/phase4_cache.py`](../gpu_fuzzy_trader/phases/phase4_cache.py)

**Hyperparameter reference:** [README.md §6.2](../README.md#62-rb_governor_enabledfalse--legacy-phase-3--4)

## Walk-forward geometry

`effective_phase4_wf_splits()` returns:

| `SPLIT_MODE` | Effective `PHASE4_WF_SPLITS` |
|--------------|------------------------------|
| `holdout_70_30` | `PHASE4_WF_SPLITS` (default 2) + optional tail holdout |
| `purged_walk_forward` | **Forced to 1** — avoids triple WF stacking |

## Algorithm

1. Split validation into WF windows (count from above).
2. For each rule, evaluate `PHASE4_GRID_TP_VALUES × PHASE4_GRID_SL_VALUES × PHASE4_GRID_CAPITAL_VALUES` combinations.
3. Reject if worst fold violates `PHASE4_MIN_WORST_*`.
4. Score by `PHASE4_USE_ROBUST_SCORE` (`min(train, val) return`) with worst-fold weights.
5. Capital hard-cap normalize if `PHASE4_HARD_CAP_NORMALIZE`.
6. `PHASE4_GRID_PASSES` round-robin refinements.

## Output

Updated `outputs/{direction}.json` with risk parameters per rule.
