# Phase 3 — Rule-set selection (legacy)

Greedy team composition from the Phase 2 pool. **Skipped** when `RB_GOVERNOR_ENABLED=True` (default).

**Source code:**
- Rule set selector: [`gpu_fuzzy_trader/phases/phase3_rule_set.py`](../gpu_fuzzy_trader/phases/phase3_rule_set.py)
- Cache: [`gpu_fuzzy_trader/phases/phase3_cache.py`](../gpu_fuzzy_trader/phases/phase3_cache.py)
- Objectives: [`gpu_fuzzy_trader/phases/phase3_objectives.py`](../gpu_fuzzy_trader/phases/phase3_objectives.py)

**Hyperparameter reference:** [README.md §6.2](../README.md#62-rb_governor_enabledfalse--legacy-phase-3--4)

## Algorithm

1. Per-symbol positive-good filter (`PHASE3_REQUIRE_POSITIVE_GOOD`).
2. Optional symbol-specialization variants (`SYMBOL_SPECIALIZATION_*`).
3. Greedy per-symbol rounds (`PHASE3_PER_SYMBOL_MAX_RULES`, `PHASE3_PER_SYMBOL_GREEDY_TOP_K`).
4. Global combine + evaluator health penalty (`PHASE3_EVAL_HEALTH_WEIGHT`).
5. Monthly penalty if `MONTHLY_VALIDATION_ENABLED`.

## Output

`outputs/{direction}.json` plus `outputs/reports/phase3_*.csv` if `PHASE3_DIAGNOSTIC_REPORT_ENABLED=True`.
