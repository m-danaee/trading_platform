# Task 8 — `fix/trade-floor-island-aware` (make hard reject floor island-aware + use config constant)

## Branch
`fix/trade-floor-island-aware` (from latest `main`).

## Problem
The hard-reject trade floor (`MIN_TRADE_POOL_FLOOR=25`) in
`compute_phase2_objectives_from_metrics` is hardcoded and does NOT respect
island mode. The `IslandHyperparams` object already has a correctly-scaled
`min_trade_pool_floor` (set at island creation via
`scale_trade_floor_by_universe`), but line 640 reads the global default
directly, ignoring it.

This means for smaller islands (cluster_1, cluster_2 with `min_trade_support=22`):
- The graduated penalty target is 22 (island-aware ✅)
- But the hard reject floor is 25 (hardcoded, NOT island-aware ❌)
- So a rule with 22 trades passes the graduated target but gets HARD-KILLED
  by the hardcoded floor (22 < 25)

Additionally, the inline magic number `50.0` at line 644 should use the
existing config constant `PHASE2_INFEASIBLE_OBJECTIVE_PENALTY` (defined at
config.py:699 but never referenced anywhere — dead config).

## Required Changes

### Fix 1 — Make the hard reject floor island-aware
**File:** `gpu_fuzzy_trader/phases/phase2_rule_pool.py` (line ~640)

Current:
```python
trade_floor = _cfg.MIN_TRADE_POOL_FLOOR
```

Change to use the island's scaled pool floor when available, falling back to
the existing scaling function for non-island mode:
```python
if island_hyperparams is not None:
    trade_floor = int(island_hyperparams.min_trade_pool_floor)
else:
    trade_floor = _cfg.effective_min_trade_pool_floor(n_valid_rows)
```

This makes the hard reject coherent with the graduated penalty (which already
uses `floors.min_trade_support` from `resolve_evolution_floors`, which in turn
reads `island_hyperparams.min_trade_support`). Both paths now respect island
scaling.

### Fix 2 — Use config constant instead of inline magic number
**File:** `gpu_fuzzy_trader/phases/phase2_rule_pool.py` (line ~644)

Current:
```python
trade_penalty = 50.0
```

Change to:
```python
trade_penalty = float(_cfg.PHASE2_INFEASIBLE_OBJECTIVE_PENALTY)
```

This wires up the dead `PHASE2_INFEASIBLE_OBJECTIVE_PENALTY` config constant
(defined at config.py:699 but never referenced). Now it's tunable.

### No other changes needed
- The `island_hyperparams` parameter is already in scope (it's a function
  parameter at line 479).
- `n_valid_rows` is also in scope (line 478).
- The `effective_min_trade_pool_floor` function already exists (config.py:1856)
  and calls `scale_trade_floor` correctly.
- `IslandHyperparams.min_trade_pool_floor` is already set correctly at island
  creation time via `resolve_island_hyperparams`.

## Acceptance Criteria
1. When `island_hyperparams is not None`, `trade_floor` equals
   `island_hyperparams.min_trade_pool_floor` (not the global 25).
2. When `island_hyperparams is None`, `trade_floor` equals
   `_cfg.effective_min_trade_pool_floor(n_valid_rows)` (scales by rows).
3. `trade_penalty` uses `_cfg.PHASE2_INFEASIBLE_OBJECTIVE_PENALTY` (not inline 50.0).
4. `config.py` asserts at import still pass.
5. `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit -q` passes (no NEW failures beyond the 2 pre-existing MAX_CONDITIONS ones).
6. Add or update a unit test that verifies the hard reject floor respects
   `island_hyperparams.min_trade_pool_floor` when provided. Use the established
   mocking pattern from existing Phase 2 tests.

## Verification Commands
```
cd /home/danaee/trading_platform
.venv/bin/python -c "import gpu_fuzzy_trader.config; print('asserts OK')"
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_rule_pool.py tests/unit/test_evox_runner.py -q 2>&1 | tail -10
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit -q 2>&1 | tail -5
```
Do NOT run the full pipeline.

## Target Files
- `gpu_fuzzy_trader/phases/phase2_rule_pool.py` (2 lines changed)
- `tests/unit/test_phase2_rule_pool.py` (add 1-2 tests for island-aware floor)
- `README.md` — no config table change needed (PHASE2_INFEASIBLE_OBJECTIVE_PENALTY already documented; MIN_TRADE_POOL_FLOOR already documented)

## Notes
- This is a coherence/correctness fix, not a tuning change.
- The island's `min_trade_pool_floor` is already correctly scaled at creation
  time — this fix just makes the hard reject READ it.
- `PHASE2_INFEASIBLE_OBJECTIVE_PENALTY` is no longer dead config after this.
- Do NOT change the default value of MIN_TRADE_POOL_FLOOR (25) or
  PHASE2_INFEASIBLE_OBJECTIVE_PENALTY (50.0) — just wire them correctly.
