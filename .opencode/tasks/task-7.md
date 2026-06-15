# Task 7 — Add `_optimize_risk` deterministic grid search

## Why
My current Phase 4 uses Optuna with a narrow grid: TP up to 3.0%,
SL up to 2.0%, capital up to 12.5%. This means the final risk
parameters are very conservative and don't have room to find a
profitable TP on the test set. The friend uses a much more
aggressive grid: TP up to 10%, SL up to 3%, capital up to 50%.

The friend ALSO uses a deterministic grid search (not Optuna) for
risk optimization. The search runs per-rule, evaluating every
(TP, SL, capital_pct) combination, picking the one that passes
`gate_positive_good` (Task 3) and maximizes `_score_metrics`
(composite of return / DD / PF). 2 passes of round-robin per-rule
tuning are run.

## Required reading
- `.opencode/plans/PLAN.md`
- `.opencode/CONTEXT.md` (JSON output contract)
- The friend's reference: `friend_project/gpu_fuzzy_trader/rb_governor.py` lines 612-680 (`_optimize_risk`).
- My existing `gpu_fuzzy_trader/phases/phase4_wf_optimizer.py` (the Optuna walk-forward).
- My existing TP/SL/capital grids in `gpu_fuzzy_trader/config.py`.

## Behavior changes

### Step 1 — Add new config keys for the grid search

```python
# Phase 4 risk-optimization grid (replaces/suppplements Optuna)
PHASE4_GRID_ENABLED = True
PHASE4_GRID_TP_VALUES = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0)
PHASE4_GRID_SL_VALUES = (1.0, 1.2, 1.5, 2.0, 2.5, 3.0)
PHASE4_GRID_CAPITAL_VALUES = (5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 25.0, 35.0, 50.0)
PHASE4_GRID_MAX_TOTAL_CAPITAL = 95.0
PHASE4_GRID_PASSES = 2
PHASE4_GRID_MIN_IMPROVEMENT = 0.02
```

These are the friend's defaults.

### Step 2 — Add `_optimize_risk_grid` to `phase4_wf_optimizer.py`

Port the friend's function. Signature:
```python
def _optimize_risk_grid(
    rules: list[dict],
    train_engine: CPUBacktestEngine,
    val_engine: CPUBacktestEngine,
    *,
    min_improvement: float = 0.02,
) -> tuple[list[dict], dict, dict, float, list[dict]]:
    """Per-rule round-robin grid search over TP, SL, capital_pct.
    
    Returns (optimized_rules, train_metrics, val_metrics, score, history).
    """
```

Algorithm:
1. Start with the input `rules`. Compute initial train+val metrics and score.
2. For `passes` rounds (default 2):
   - For each rule index:
     - For each (TP, SL, capital) combination in the grid:
       - Apply to the rule.
       - Re-evaluate the full rule set on train+val.
       - Check `sum(capital_pct) <= max_total_capital`.
       - Check `gate_positive_good` (Task 3).
       - If score improves by `>= min_improvement`, keep it.
3. Return the optimized rules + metrics.

### Step 3 — Add a new entry point `WalkForwardRiskOptimizer.optimize_risk_grid`

A new public method on the existing `WalkForwardRiskOptimizer` class
that calls `_optimize_risk_grid` and writes the result to
`outputs/{direction}.json`. This is a NEW path; the existing
`optimize` (Optuna) method stays.

### Step 4 — Wire into `run_pipeline.py`

In `run_pipeline.py` (the orchestrator's phase4 runner), check
`PHASE4_GRID_ENABLED`. If True, call `optimize_risk_grid` INSTEAD of
the Optuna `optimize`. If False, fall back to Optuna (legacy).

The friend uses ONLY the grid. I will do the same by default (grid
ON, Optuna OFF) but keep the Optuna path behind the flag for
debugging.

## Out of scope
- Do NOT change the JSON output format.
- Do NOT modify `evaluator_v5.ipynb`.
- Do NOT touch the GPU engine or EvoX runner.
- Do NOT change the per-symbol greedy logic in Phase 3.
- Do NOT add Tasks 8-9 features.

## Acceptance criteria
1. All 7 new config keys are present and accessible.
2. `_optimize_risk_grid` is importable from `phase4_wf_optimizer.py`.
3. The function returns `(optimized_rules, train_metrics, val_metrics, score, history)` with the right shape.
4. For an input rule with default TP=2.0, SL=1.0, capital_pct=12.5, the function tries the full grid and picks the best.
5. The function respects `PHASE4_GRID_MAX_TOTAL_CAPITAL=95%`.
6. Each trial passes `gate_positive_good` (Task 3).
7. The function performs `PHASE4_GRID_PASSES=2` rounds of round-robin.
8. The new `optimize_risk_grid` method is wired into the pipeline.
9. New unit test `tests/unit/test_risk_grid_search.py` with ≥ 4 cases:
   - A rule improves after the grid.
   - A rule stays unchanged when the grid doesn't improve.
   - The `max_total_capital` constraint is respected.
   - The `gate_positive_good` check filters out bad combinations.
10. All existing tests pass.
11. No changes to `evaluator_v5.ipynb` or the GPU engine.

## Constraints
- Stay on `feature/task-7-risk-grid-search` (off `main` after task-6 is merged).
- 12.7 GiB RAM total.
- PEP 8, type hints, module logger.
- Use only existing third-party deps.

## Files I will touch
- `gpu_fuzzy_trader/config.py` — 7 new `PHASE4_GRID_*` keys
- `gpu_fuzzy_trader/phases/phase4_wf_optimizer.py` — `_optimize_risk_grid` + new public method
- `gpu_fuzzy_trader/run_pipeline.py` — wire the new path
- `tests/unit/test_risk_grid_search.py` (new) — ≥ 4 cases
