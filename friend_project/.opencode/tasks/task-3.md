# Task 3: Per-Symbol RB Governor Validation

## Goal
Run RB Governor per symbol on per-symbol train/val data with per-symbol pools. Merge resulting per-symbol rule sets into the final strategy output. Rules are already tagged with `symbol is S` from Phase 2, so skip `_symbol_specialized_variants` by setting `RB_REQUIRE_SYMBOL_FILTERS=False`.

## Target Files
1. **`gpu_fuzzy_trader/run_pipeline.py`** — Per-symbol RB Governor loop + strategy merge
2. **`gpu_fuzzy_trader/rb_governor.py`** — No code changes expected (handled via config override)
3. **`gpu_fuzzy_trader/config.py`** — Add `PER_SYMBOL_RB_OUTPUT_DIR` if needed

## Detailed Spec

### 1. Current flow (in `run_pipeline.py`, around line 396)

Currently when `RB_ENGINE_ENABLED=True`:
```python
rb_result = run_rb_governor_pipeline(
    train_df, val_df, phase2_result, self._directions, output_dir=_cfg.OUTPUTS_DIR
)
results["phase3"] = rb_result
results["phase4"] = rb_result
```

### 2. New per-symbol flow

When `PER_SYMBOL_PHASE2=True` (and `RB_ENGINE_ENABLED=True`):

```python
if PER_SYMBOL_PHASE2:
    symbols = sorted(train_df["symbol"].unique())
    all_strategies: dict[str, list[dict]] = {"long": [], "short": []}  # direction -> list of per-symbol strategy dicts
    
    # Temporarily disable symbol filter requirement (rules already tagged)
    orig_rb_require = getattr(_cfg, "RB_REQUIRE_SYMBOL_FILTERS", True)
    _cfg.RB_REQUIRE_SYMBOL_FILTERS = False
    
    try:
        for symbol in symbols:
            # Filter data to this symbol
            symbol_train = train_df[train_df["symbol"] == symbol]
            symbol_val = val_df[val_df["symbol"] == symbol]
            
            if len(symbol_train) < PER_SYMBOL_MIN_ROWS or len(symbol_val) < PER_SYMBOL_MIN_ROWS // 2:
                logger.warning("RB Governor: skipping symbol %s (insufficient data)", symbol)
                continue
            
            # Get per-symbol pool (already tagged with "symbol is S")
            symbol_pools = {}
            for direction in self._directions:
                per_symbol_pool_path = os.path.join(
                    _cfg.PHASE2_PER_SYMBOL_POOL_DIR, f"phase2_{direction}_{symbol}_pool.json"
                )
                if os.path.exists(per_symbol_pool_path):
                    with open(per_symbol_pool_path) as f:
                        symbol_pools[direction] = json.load(f)
                else:
                    symbol_pools[direction] = []
            
            if all(len(pool) == 0 for pool in symbol_pools.values()):
                logger.warning("RB Governor: skipping symbol %s (empty pools)", symbol)
                continue
            
            # Run RB Governor on this symbol's data
            per_symbol_output_dir = os.path.join(_cfg.OUTPUTS_DIR, "per_symbol", str(symbol))
            os.makedirs(per_symbol_output_dir, exist_ok=True)
            
            try:
                symbol_result = run_rb_governor_pipeline(
                    symbol_train, symbol_val, symbol_pools, self._directions,
                    output_dir=per_symbol_output_dir
                )
                for direction in self._directions:
                    strategy = symbol_result.get(direction)
                    if strategy:
                        all_strategies[direction].append(strategy)
                logger.info("RB Governor [%s]: completed", symbol)
            except Exception as exc:
                logger.warning("RB Governor [%s]: failed — %s", symbol, exc, exc_info=True)
                continue
        
        # Merge per-symbol strategies into final output
        rb_result = _merge_per_symbol_strategies(all_strategies, self._directions)
    finally:
        _cfg.RB_REQUIRE_SYMBOL_FILTERS = orig_rb_require
    
    results["phase3"] = rb_result
    results["phase4"] = rb_result

else:
    # Original behavior (unchanged)
    rb_result = run_rb_governor_pipeline(
        train_df, val_df, phase2_result, self._directions, output_dir=_cfg.OUTPUTS_DIR
    )
    results["phase3"] = rb_result
    results["phase4"] = rb_result
```

### 3. `_merge_per_symbol_strategies()` function

```python
def _merge_per_symbol_strategies(
    all_strategies: dict[str, list[dict]],
    directions: list[str],
) -> dict[str, dict]:
    """
    Merge per-symbol RB Governor results into final combined strategies.
    
    Combines rules_set from all per-symbol strategies. Distributes capital_pct
    proportionally. Writes final outputs/{direction}.json.
    """
    merged: dict[str, dict] = {}
    
    for direction in directions:
        strategies = all_strategies.get(direction, [])
        if not strategies:
            merged[direction] = {}
            continue
        
        combined_rules = []
        for strategy in strategies:
            combined_rules.extend(strategy.get("rules_set", []))
        
        # Distribute capital evenly across rules
        max_total = float(getattr(_cfg, "PHASE4_MAX_TOTAL_CAPITAL", 35.0))
        n_rules = max(len(combined_rules), 1)
        capital_per_rule = min(max_total / n_rules, 12.5)
        
        for rule in combined_rules:
            if "capital_pct" not in rule or rule.get("capital_pct", 0) > capital_per_rule:
                rule["capital_pct"] = round(capital_per_rule, 2)
        
        merged_strategy = {
            "direction": direction,
            "rules_set": combined_rules,
            "risk_optimized": True,
            "rb_governor": True,
            "per_symbol_merged": True,
            "symbols": len(strategies),
            "total_rules": len(combined_rules),
        }
        
        # Write final output
        output_path = os.path.join(_cfg.OUTPUTS_DIR, f"{direction}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(merged_strategy, f, indent=2)
        
        merged[direction] = merged_strategy
        logger.info(
            "Merged %d strategies for %s: %d rules total",
            len(strategies), direction, len(combined_rules),
        )
    
    return merged
```

### 4. Config additions

```python
# Per-symbol RB Governor: minimum val rows for a symbol to be processed
PER_SYMBOL_RB_MIN_VAL_ROWS = 500
```

## Acceptance Criteria
- [ ] RB Governor runs per symbol when `PER_SYMBOL_PHASE2=True`
- [ ] Per-symbol train/val data filtered correctly
- [ ] `RB_REQUIRE_SYMBOL_FILTERS=False` set temporarily (rules already tagged)
- [ ] Per-symbol RB Governor output saved to `outputs/per_symbol/{symbol}/`
- [ ] `_merge_per_symbol_strategies()` combines all per-symbol rule sets
- [ ] Final `outputs/{direction}.json` contains merged strategy
- [ ] Capital distributed evenly across merged rules (≤ PHASE4_MAX_TOTAL_CAPITAL)
- [ ] Exception isolation: one symbol's RB Governor failure doesn't abort others
- [ ] `PER_SYMBOL_PHASE2=False` restores original single RB Governor call
- [ ] Existing tests still pass
- [ ] `rb_governor.py` itself needs NO code changes (config override only)

## Dependencies
- task-2 (per-symbol Phase 2 pools with `symbol is S` tags)

## Handoff
Write `.opencode/handoffs/task-3-implementer.json` on completion.
