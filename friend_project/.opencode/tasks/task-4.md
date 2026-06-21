# Task 4: Phase 5 Rule Filtering (Remove Negative-PnL Rules on Test)

## Goal
Enhance Phase 5 to filter out rules with non-positive Net_PnL on test data — matching the main project's `_remove_negative_pnl_rules()` behavior. Rules that lose money on out-of-sample data are removed from the strategy. The strategy file is rewritten with the cleaned rule set.

## Target Files
1. **`gpu_fuzzy_trader/phases/phase5_oos.py`** — Add `_remove_negative_pnl_rules()`, wire into `run()`
2. **`gpu_fuzzy_trader/output/writer.py`** — Add `_maybe_write_evaluator_clean()` helper
3. **`gpu_fuzzy_trader/config.py`** — Add `PHASE5_REMOVE_NEGATIVE_PNL_RULES`, `PHASE3_GLOBAL_MIN_RULES`, `WRITE_EVALUATOR_CLEAN`

## Detailed Spec

### 1. Config additions (`config.py`)

```python
# Phase 5: remove rules with non-positive Net_PnL on test data.
PHASE5_REMOVE_NEGATIVE_PNL_RULES = True

# Minimum number of rules to keep after pruning (safeguard against over-pruning).
PHASE3_GLOBAL_MIN_RULES = 2

# When True, write a stripped evaluator-clean JSON (direction + rules_set only).
WRITE_EVALUATOR_CLEAN = True
```

### 2. `_remove_negative_pnl_rules()` method (in `OOS_Evaluator` class, `phase5_oos.py`)

Add this method to the `OOS_Evaluator` class:

```python
def _remove_negative_pnl_rules(
    self,
    strategy: dict,
    trade_log: pd.DataFrame,
    direction: str,
) -> tuple[dict, bool]:
    """
    Remove rules whose Net_PnL on test data is <= 0.
    
    Safeguard: keeps at least PHASE3_GLOBAL_MIN_RULES rules.
    Rewrites the cleaned strategy to disk.
    
    Returns (strategy, cleaned) where cleaned=True if rules were removed.
    """
    rules = strategy.get("rules_set", [])
    if not rules or trade_log is None or trade_log.empty:
        return strategy, False
    
    kept: list[dict] = []
    for rule_idx, rule in enumerate(rules, start=1):
        rule_trades = trade_log[trade_log["Rule_Index"] == rule_idx]
        if rule_trades.empty:
            kept.append(rule)  # Keep rules with no trades (can't judge)
            continue
        total_pnl = float(rule_trades["Net_PnL"].sum())
        if total_pnl > 0:
            kept.append(rule)
    
    global_min = int(getattr(_cfg, "PHASE3_GLOBAL_MIN_RULES", 2))
    if len(kept) < global_min:
        logger.warning(
            "Phase 5 [%s]: cannot remove negative-PnL rules: "
            "would drop below minimum %d rules. Keeping all %d rules.",
            direction, global_min, len(rules),
        )
        return strategy, False
    
    removed = len(rules) - len(kept)
    if removed > 0:
        logger.info(
            "Phase 5 [%s]: removed %d negative-PnL rules, kept %d",
            direction, removed, len(kept),
        )
        strategy = dict(strategy)  # shallow copy
        strategy["rules_set"] = kept
        
        # Rewrite strategy to disk
        output_path = _STRATEGY_PATHS[direction]
        try:
            with open(output_path, "w", encoding="utf-8") as fh:
                json.dump(strategy, fh, indent=2)
            _maybe_write_evaluator_clean(strategy, output_path, direction)
        except OSError as exc:
            logger.warning(
                "Phase 5 [%s]: failed to rewrite cleaned strategy: %s",
                direction, exc,
            )
        return strategy, True
    
    return strategy, False
```

### 3. Wire into `run()` method

In the `run()` method, after the per-direction evaluation loop (after `metrics_by_split` and `trade_logs_by_split` are populated), add:

```python
# Remove negative-PnL rules from the strategy
cleaned = False
if getattr(_cfg, "PHASE5_REMOVE_NEGATIVE_PNL_RULES", False):
    test_trade_log = trade_logs_by_split.get("test")
    if test_trade_log is not None and not test_trade_log.empty:
        strategy, cleaned = self._remove_negative_pnl_rules(
            strategy, test_trade_log, direction
        )

# Re-evaluate test split after cleanup so reports use correct metrics
if cleaned:
    test_metrics2, per_symbol_rows2, trade_log2 = self._evaluate_strategy(
        datasets_by_split["test"], strategy, direction
    )
    metrics_by_split["test"] = test_metrics2
    trade_logs_by_split["test"] = trade_log2
    # Rebuild all_per_symbol for test
    all_per_symbol = [
        r for r in all_per_symbol if r.get("dataset") != "test"
    ] + per_symbol_rows2
```

This should be placed right after the split evaluation loop and before the reports/reporter calls, so that report plots use the cleaned metrics.

### 4. `_maybe_write_evaluator_clean()` (in `output/writer.py`)

Add this function at module level in `output/writer.py`:

```python
def _maybe_write_evaluator_clean(
    strategy: dict, main_path: str | Path, direction: str
) -> None:
    """
    Write a stripped strategy file containing only direction and rules_set.
    
    Placed in evaluator_clean/ subdirectory next to the main strategy file.
    Gated by WRITE_EVALUATOR_CLEAN config flag.
    """
    if not bool(getattr(_cfg, "WRITE_EVALUATOR_CLEAN", True)):
        return
    main_path = Path(main_path)
    clean_dir = main_path.parent / "evaluator_clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    clean_path = clean_dir / f"{direction}_evaluator_clean.json"
    
    clean = {
        "direction": strategy.get("direction", direction),
        "rules_set": strategy.get("rules_set", []),
    }
    try:
        with open(clean_path, "w", encoding="utf-8") as fh:
            json.dump(clean, fh, indent=2)
    except Exception as exc:
        logger.debug("evaluator_clean write failed for %s: %s", direction, exc)
```

### 5. Also update `phase5_oos.py` imports

Add at top of `phase5_oos.py`:
```python
from gpu_fuzzy_trader.output.writer import _maybe_write_evaluator_clean
```

And ensure `json` is imported at the top of the file (check if already imported — it is for loading strategies, but verify for json.dump usage).

## Acceptance Criteria
- [ ] `PHASE5_REMOVE_NEGATIVE_PNL_RULES = True` in config.py
- [ ] `PHASE3_GLOBAL_MIN_RULES = 2` in config.py
- [ ] `WRITE_EVALUATOR_CLEAN = True` in config.py
- [ ] `_remove_negative_pnl_rules()` method in `OOS_Evaluator` removes rules with Net_PnL <= 0 on test
- [ ] Rules with 0 trades on test are kept (can't judge)
- [ ] Minimum rule safeguard: won't drop below `PHASE3_GLOBAL_MIN_RULES`
- [ ] After pruning, test split is re-evaluated with cleaned rules
- [ ] Cleaned strategy overwrites `outputs/{direction}.json`
- [ ] Evaluator-clean JSON written to `outputs/evaluator_clean/{direction}_evaluator_clean.json`
- [ ] `_maybe_write_evaluator_clean()` in `output/writer.py`
- [ ] Gated by `PHASE5_REMOVE_NEGATIVE_PNL_RULES` flag (False = skip)
- [ ] Existing tests still pass
- [ ] No changes to data splitter, Phase 1, Phase 2, or RB Governor

## Dependencies
- task-3 (per-symbol RB Governor produces the strategy JSON that Phase 5 reads)

## Handoff
Write `.opencode/handoffs/task-4-implementer.json` on completion.
