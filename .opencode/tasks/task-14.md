# Task-14: RB Governor Rebalance (val overfit fix)

**Branch:** `fix/rb-governor-rebalance`
**Priority:** 🔴 Critical
**Fixes:** C1, C2, C3, C4, M7
**Depends on:** Pre-flight (commit config.py tuning to main first)

## Problem

RB Governor weights validation 2.7–3.8× over train (`120*valid + 45*train` ratios, `4.5*valid + 1.2*train` returns). `RB_REQUIRE_TRAIN_SLIGHTLY_ABOVE_VALID=True` forces val-beating. Gap penalty 0.55 is 12× too weak. CV folds computed but unused in governor. Rule addition ignores PF/DD degradation.

Baseline: val Sortino 1.20 > train 0.65 (val overfit), val→test Δ = −17.6%.

## Files to Modify

1. `gpu_fuzzy_trader/rb_governor.py` — `_score_metrics`, `_combined_return_score`
2. `gpu_fuzzy_trader/config.py` — RB_* flags

## Detailed Changes

### C1: Rebalance weights in `_score_metrics` (~lines 188–210)

```python
# BEFORE:
score = (
    120.0 * valid_ratio
    + 45.0 * train_ratio
    + 4.5 * valid_ret
    + 1.2 * train_ret
    ...
)

# AFTER:
score = (
    60.0 * valid_ratio
    + 60.0 * train_ratio
    + 3.0 * valid_ret
    + 3.0 * train_ret
    ...
)
```

### C2: config.py line 1536
```python
RB_REQUIRE_TRAIN_SLIGHTLY_ABOVE_VALID: bool = False
```

### C3: config.py
```python
RB_TRAIN_VALID_RETURN_GAP_WEIGHT: float = 4.0   # was 0.55
RB_TRAIN_VALID_RATIO_GAP_WEIGHT: float = 30.0   # was 12.0
```

### C4: Add CV-fold consistency term to `_score_metrics`

- Import `CvFoldValEvaluator` (already exists in phase2_rule_pool.py).
- In the RB governor pipeline, when evaluating a candidate rule, run it through the 3 purged CV folds.
- Add to `_score_metrics`:
  ```python
  # CV-fold consistency (injected via new params or metrics sidecar)
  cv_fold_returns = metrics.get("_cv_fold_returns", [])  # list of per-fold return_pct
  if cv_fold_returns:
      cv_min = min(cv_fold_returns)
      cv_std = float(np.std(cv_fold_returns))
      score += 8.0 * cv_min    # reward cross-fold minimum
      score -= 3.0 * cv_std    # penalize cross-fold variance
  ```
- Wire `_cv_fold_returns` into the candidate evaluation path in `run_rb_governor_pipeline` (call `CvFoldValEvaluator` per candidate rule).

### M7: Tighten `_combined_return_score` (line 232)

```python
def _combined_return_score(train_m, valid_m, prev_pf=None, prev_dd=None):
    train_ret = _f(train_m, "total_return_pct")
    valid_ret = _f(valid_m, "total_return_pct")
    score = train_ret + valid_ret
    # Penalize edge-quality erosion
    if prev_pf is not None:
        new_pf = _f(valid_m, "profit_factor", 0.0)
        score -= 2.0 * max(0.0, prev_pf - new_pf)
    if prev_dd is not None:
        new_dd = _f(valid_m, "max_drawdown_pct", 0.0)
        score -= 3.0 * max(0.0, new_dd - prev_dd)
    score -= _evaluator_health_penalty(train_m, role="train") / 35.0
    score -= _evaluator_health_penalty(valid_m, role="valid") / 35.0
    return float(score)
```
- Update all callers in `_compose_ruleset` / `_compose_ruleset_return_only` to pass `prev_pf` and `prev_dd`.

## Acceptance Criteria

- [ ] `_score_metrics` weights are 60/60 (ratio) and 3/3 (return).
- [ ] `RB_REQUIRE_TRAIN_SLIGHTLY_ABOVE_VALID = False`.
- [ ] `RB_TRAIN_VALID_RETURN_GAP_WEIGHT = 4.0`, `RB_TRAIN_VALID_RATIO_GAP_WEIGHT = 30.0`.
- [ ] `_score_metrics` includes CV-fold-min term (coefficient 8.0).
- [ ] `_combined_return_score` penalizes PF degradation (k_pf=2.0) and DD increase (k_dd=3.0).
- [ ] All callers of `_combined_return_score` pass prev_pf/prev_dd.
- [ ] Unit tests pass: `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/ -x -q -k "rb_governor or scoring"`

## Verification

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/ -x -q -k "rb_governor or scoring"
```
