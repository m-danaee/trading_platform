# Phase 2 — Rule evolution

NSGA-III search for tradeable rules over the Phase 1 feature set, gated by train/val admissibility.

**Source code:**
- Rule pool generator: [`gpu_fuzzy_trader/phases/phase2_rule_pool.py`](../gpu_fuzzy_trader/phases/phase2_rule_pool.py)
- EvoX runner: [`gpu_fuzzy_trader/evolution/evox_runner.py`](../gpu_fuzzy_trader/evolution/evox_runner.py)
- Island scheduler: [`gpu_fuzzy_trader/phases/phase2_island_scheduler.py`](../gpu_fuzzy_trader/phases/phase2_island_scheduler.py)
- Stage A/B: [`gpu_fuzzy_trader/phases/phase2_stage.py`](../gpu_fuzzy_trader/phases/phase2_stage.py)
- Support / penalties: [`gpu_fuzzy_trader/phases/phase2_support.py`](../gpu_fuzzy_trader/phases/phase2_support.py)
- Sparse encoding: [`gpu_fuzzy_trader/phases/phase2_sparse_encoding.py`](../gpu_fuzzy_trader/phases/phase2_sparse_encoding.py)

**Hyperparameter reference:** [README.md §5](../README.md#5-phase-2--rule-evolution)

## Modes

| Mode | Trigger | Search layout |
|------|---------|---------------|
| `global` | `PHASE2_ISLAND_MODE="global"` | One NSGA-III on full universe |
| `cluster` | `PHASE2_ISLAND_MODE="cluster"` | K symbol clusters + orphan boost |

`PHASE2_JOINT_TRAIN_VAL=True` adds `min(train, val)` to the fitness; `False` makes fitness train-only and shifts the overfit check to pool admission.

## Stacked pool gates

A rule must pass **all** of these to enter the pool:

1. Evolution feasibility (`PHASE2_RETURN_FLOOR_PCT`, PF, drawdown gate)
2. `passes_pool_trade_floor` — minimum trade support
3. `passes_pool_admission_gate` — train/val returns, gap, PF
4. `PHASE2_STRICT_POSITIVE_GOOD` → `gate_positive_good`
5. `PHASE2_MONTHLY_ADMISSION_ENABLED` → monthly ratio on train
6. `PHASE2_KEEP_TOP_RULES` cap

The pool is then consumed by Phase 3/4 (legacy path) or `rb_governor` (RB Governor path).
