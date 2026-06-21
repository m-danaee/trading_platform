# Phase 5 — Out-of-sample evaluation

True OOS evaluation on `test.csv`. The only phase that ever touches test data (per AGENTS.md).

**Source code:**
- OOS evaluator: [`gpu_fuzzy_trader/phases/phase5_oos.py`](../gpu_fuzzy_trader/phases/phase5_oos.py)

**Hyperparameter reference:** [README.md §8](../README.md#8-phase-5--out-of-sample)

## Behavior

| Parameter | Effect |
|-----------|--------|
| `PHASE5_VALIDATION_RETURN_GATE_PCT` | Min val return % for deployable flag |
| `PHASE5_VALIDATION_PROFIT_FACTOR_GATE` | Min val PF for deployable flag |
| `PHASE5_REMOVE_NEGATIVE_PNL_RULES` | Strip rules with negative test PnL from final strategy |

Always runs last, regardless of which path Phase 3/4 used (legacy or RB Governor).

## Output

`outputs/{direction}.json` (final, with risk baked in) plus OOS report under `outputs/reports/`.
