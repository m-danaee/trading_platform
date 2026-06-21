# Task 2: Per-Symbol Phase 2 Training

## Goal
Modify Phase 2 to train evolutionary rule pools per symbol instead of across all symbols simultaneously. Each symbol gets its own pool of rules evolved on that symbol's data. Rules are tagged with `symbol is S` before passing to downstream phases.

## Target Files
1. **`gpu_fuzzy_trader/config.py`** — Add `PER_SYMBOL_PHASE2` flag, per-symbol pool paths
2. **`gpu_fuzzy_trader/run_pipeline.py`** — Per-symbol loop in `_run_phase2()`
3. **`gpu_fuzzy_trader/phases/phase2_rule_pool.py`** — Purged CV skip for per-symbol, symbol tagging helper

## Detailed Spec

### 1. `config.py` additions
```python
# Per-symbol Phase 2 training mode.
# True → Phase 2 runs separately for each symbol; rules tagged with "symbol is S".
# False → original behavior (single combined pool across all symbols).
PER_SYMBOL_PHASE2 = True

# Directory for per-symbol Phase 2 pools.
PHASE2_PER_SYMBOL_POOL_DIR = os.path.join(_PROJECT_ROOT, "pools", "per_symbol")
```
Place these near the existing Phase 2 config section.

### 2. `run_pipeline.py._run_phase2()` — per-symbol loop

The current `_run_phase2()` processes each direction (long/short) with the full train_df. Modify so that when `PER_SYMBOL_PHASE2=True`:

```
For each direction in self._directions:
    feature_infos = phase1_result[direction]
    
    if PER_SYMBOL_PHASE2:
        symbols = sorted(train_df["symbol"].unique())
        per_symbol_pools = []
        for symbol in symbols:
            symbol_train_df = train_df[train_df["symbol"] == symbol].copy()
            if len(symbol_train_df) < MIN_ROWS_THRESHOLD:  # e.g., 1000
                logger.warning("Skipping symbol %s: insufficient rows (%d)", symbol, len(symbol_train_df))
                continue
            
            # Run Phase 2 on this symbol's data
            generator = Rule_Pool_Generator(symbol_train_df, feature_infos, direction)
            pool = generator.run()
            
            # Tag each rule with "symbol is S" condition
            pool = _tag_pool_with_symbol(pool, symbol)
            
            # Save per-symbol pool
            per_symbol_path = f"pools/per_symbol/phase2_{direction}_{symbol}_pool.json"
            save_json(per_symbol_path, pool)
            per_symbol_pools.append(pool)
        
        # Merge all per-symbol pools for backward compat
        merged_pool = _merge_pools(per_symbol_pools)
        save_json(f"pools/phase2_{direction}_pool.json", merged_pool)
        pools[direction] = merged_pool  # Pass merged to RB Governor
    else:
        # Original behavior (unchanged)
        generator = Rule_Pool_Generator(train_df, feature_infos, direction)
        pools[direction] = generator.run()
```

**Key implementation details:**
- Extract unique symbols from `train_df["symbol"]` column
- Sort symbols for deterministic ordering
- Skip symbols with too few rows (configurable threshold, e.g., `PER_SYMBOL_MIN_ROWS = 1000`)
- For each symbol: filter train_df, create `Rule_Pool_Generator`, run evolution
- Tag rules with symbol condition (see below)
- Save per-symbol pool to `pools/per_symbol/` directory
- Merge all per-symbol pools (concatenate lists, no dedup needed — rules are already per-symbol)
- Save merged pool to standard `pools/phase2_{direction}_pool.json` path
- Return merged pool in results dict (RB Governor consumes this)
- Log progress: "Phase 2 [long]: symbol X — generated N rules", "Phase 2 [long]: merged M total rules from K symbols"

### 3. Symbol tagging — `_tag_pool_with_symbol(pool, symbol)`

Add a helper function (in `run_pipeline.py` or `phase2_rule_pool.py`):
```python
def _tag_pool_with_symbol(pool: list[dict], symbol: str) -> list[dict]:
    """Add 'symbol is <symbol>' condition to each rule in the pool."""
    tagged = []
    for rule in pool:
        rule = dict(rule)  # shallow copy
        conditions = list(rule.get("conditions", []))
        # Only add if not already present
        if not any(c.get("feature") == "symbol" and c.get("value") == str(symbol) for c in conditions):
            conditions.append({"feature": "symbol", "operator": "is", "value": str(symbol)})
        rule["conditions"] = conditions
        tagged.append(rule)
    return tagged
```

### 4. `phase2_rule_pool.py` — skip purged CV for per-symbol

When per-symbol mode is active, individual symbols may have insufficient rows for purged CV folds (`PURGED_CV_MIN_VALID_ROWS = 5000`). 

**Approach:** In `_run_phase2()` in `run_pipeline.py`, before calling `Rule_Pool_Generator.run()`, temporarily override `PHASE2_CV_FILTER_ENABLED = False` when `PER_SYMBOL_PHASE2=True`:

```python
if PER_SYMBOL_PHASE2:
    _cfg.PHASE2_CV_FILTER_ENABLED = False
```

Or add a parameter to `Rule_Pool_Generator` to skip purged CV. The simplest approach is the runtime override since the generator reads the config flag at filter time.

### 5. Pool merging — `_merge_pools(per_symbol_pools)`

Simple list concatenation:
```python
def _merge_pools(per_symbol_pools: list[list[dict]]) -> list[dict]:
    merged = []
    for pool in per_symbol_pools:
        merged.extend(pool)
    return merged
```

No deduplication needed — rules from different symbols have different `symbol is S` conditions, so they're inherently distinct.

## Acceptance Criteria
- [ ] `PER_SYMBOL_PHASE2 = True` in config.py (default)
- [ ] `PHASE2_PER_SYMBOL_POOL_DIR` path exists in config
- [ ] `_run_phase2()` loops over symbols when `PER_SYMBOL_PHASE2=True`
- [ ] Per-symbol pools saved to `pools/per_symbol/phase2_{direction}_{symbol}_pool.json`
- [ ] Each rule in per-symbol pool has `{"feature": "symbol", "operator": "is", "value": "S"}` condition
- [ ] Merged pool saved to `pools/phase2_{direction}_pool.json` (backward compat)
- [ ] Purged CV disabled for per-symbol runs (runtime override)
- [ ] Symbols with too few rows are skipped with warning
- [ ] Log messages indicate per-symbol progress
- [ ] `PER_SYMBOL_PHASE2=False` restores original behavior (single combined run)
- [ ] No changes to data splitter, Phase 1, or feature selection
- [ ] Existing tests still pass

## Dependencies
- task-1 (GPU runtime) — this task branches from `feature/task-1-gpu-runtime`

## Handoff
Write `.opencode/handoffs/task-2-implementer.json` on completion.
