# Task 6 — Add multi-symbol combinations in Phase 3

## Why
My current Phase 3 produces rules with single-symbol filters like
`"symbol is 5"` or `"symbol is 1, symbol is 5"` (the latter only
because the SAME rule was independently selected for two different
symbols in the per-symbol greedy and then merged by
`_merge_per_symbol_rules`). The friend does explicit 2- and
3-symbol combinations: every pool rule is evaluated as
`symbol is X`, `symbol is Y`, `symbol is X AND symbol is Y`,
`symbol is X AND symbol is Y AND symbol is Z`, etc. The best
combination is kept.

This expands the search space significantly. A rule that doesn't
work for `symbol 5` alone might work great for `symbol 1, 5` because
the cross-symbol diversification reduces volatility.

## Required reading
- `.opencode/plans/PLAN.md`
- `.opencode/CONTEXT.md` (JSON output contract)
- The friend's reference: `friend_project/gpu_fuzzy_trader/rb_governor.py` lines 386-465 (`_symbol_specialized_variants`).
- My existing `gpu_fuzzy_trader/phases/phase3_rule_set.py` (`_merge_per_symbol_rules` merges identical rules across symbols; `_score_pool_rule_on_symbol` scores per-symbol).

## Behavior changes

### Step 1 — Add config keys

```python
# Multi-symbol combinations in Phase 3 symbol specialization
SYMBOL_SPECIALIZATION_USE_COMBINATIONS = True
SYMBOL_SPECIALIZATION_MAX_SYMBOLS_PER_RULE = 3
SYMBOL_SPECIALIZATION_TOP_SINGLE_SYMBOLS = 5
SYMBOL_SPECIALIZATION_MAX_VARIANTS_PER_RULE = 10
SYMBOL_SPECIALIZATION_MIN_TRAIN_TRADES = 10
SYMBOL_SPECIALIZATION_MIN_VAL_TRADES = 6
```

These are the friend's defaults. I already have
`SYMBOL_SPECIALIZATION_MAX_SYMBOLS_PER_RULE=3` in my config (from
the previous owner); verify and update if needed.

### Step 2 — Add `_build_symbol_specialized_variants` to `phase3_rule_set.py`

Port the friend's function. Signature:
```python
def _build_symbol_specialized_variants(
    rule: dict,
    train_engine: CPUBacktestEngine,
    val_engine: CPUBacktestEngine,
    symbols: list[str],
) -> list[dict]:
    """Build top-K 1-, 2-, and 3-symbol variants of *rule*.
    
    Returns a list of `dict` rules with `symbol is X`, `symbol is X,Y`,
    or `symbol is X,Y,Z` conditions appended. Each variant is
    evaluated on train + val; only those passing
    `gate_positive_good` (Task 3) with `min_train_trades` and
    `min_val_trades` trades are kept.
    """
```

Algorithm:
1. For each of the top-K best single-symbol variants of `rule`, evaluate and rank by `_score_metrics` or similar.
2. Take the top-`SYMBOL_SPECIALIZATION_TOP_SINGLE_SYMBOLS=5` symbols.
3. Generate all 2- and 3-symbol combinations of those 5 (10 + 10 = 20 candidates).
4. For each combination, build a rule with `symbol is X,Y` and evaluate.
5. Sort by score, take top-`SYMBOL_SPECIALIZATION_MAX_VARIANTS_PER_RULE=10`.
6. Return the variants.

If `SYMBOL_SPECIALIZATION_USE_COMBINATIONS=False`, only return single-symbol variants (the existing behavior).

### Step 3 — Wire into `_merge_per_symbol_rules` (or its successor)

The current `_merge_per_symbol_rules` is called from the per-symbol
greedy path. Replace (or augment) it so:
- For each pool rule, build the multi-symbol variants.
- Pick the best variant.
- Use that variant in the final merged rule set.

This is the only behavior change in Phase 3. The per-symbol greedy
itself (which decides how many rules each symbol gets) stays the same.

## Out of scope
- Do NOT change the JSON output format.
- Do NOT modify `evaluator_v5.ipynb`.
- Do NOT touch the GPU engine or EvoX runner.
- Do NOT change the per-symbol greedy logic for rule selection (the *number* of rules per symbol).
- Do NOT add Tasks 7-9 features.

## Acceptance criteria
1. All 6 new config keys are present and accessible (some may already exist; verify).
2. `_build_symbol_specialized_variants` is importable from `phase3_rule_set.py`.
3. The function returns a list of dicts, each with `conditions` ending in `symbol is X[, Y, Z]` conditions.
4. When `SYMBOL_SPECIALIZATION_USE_COMBINATIONS=False`, the function returns at most single-symbol variants.
5. When `SYMBOL_SPECIALIZATION_USE_COMBINATIONS=True`, the function returns a mix of 1-, 2-, and 3-symbol variants sorted by score.
6. The variants all pass `gate_positive_good` (Task 3) on train+val (or are filtered out).
7. The new function is wired into the per-symbol greedy result-merging step.
8. New unit test `tests/unit/test_multi_symbol_combinations.py` with ≥ 4 cases:
   - Single-symbol variant is kept when it's the only one passing the gate.
   - 2-symbol variant is preferred over single-symbol when both pass the gate.
   - 3-symbol variants are generated when `MAX_SYMBOLS_PER_RULE=3`.
   - The `USE_COMBINATIONS=False` flag disables 2/3-symbol generation.
9. All existing tests pass.
10. No changes to `evaluator_v5.ipynb` or the GPU engine.

## Constraints
- Stay on `feature/task-6-multi-symbol-combinations` (off `main` after task-5 is merged).
- 12.7 GiB RAM total.
- PEP 8, type hints, module logger.
- Use only existing third-party deps.

## Files I will touch
- `gpu_fuzzy_trader/config.py` — verify/add 6 `SYMBOL_SPECIALIZATION_*` keys
- `gpu_fuzzy_trader/phases/phase3_rule_set.py` — add `_build_symbol_specialized_variants`; wire it into the per-symbol greedy result path
- `tests/unit/test_multi_symbol_combinations.py` (new) — ≥ 4 cases
