# Task-15: Fitness & Objective Redesign

**Branch:** `fix/fitness-objective-redesign`
**Priority:** 🔴 Critical
**Fixes:** C5, C6, C7, H1, H2, M3
**Depends on:** task-14 merged

## Problem

f1 (Sortino) has 0.0 support penalty → lucky 5-trade rules dominate. f3 (win-rate) is degenerate. Sortino tanh-saturated (flat >3.0). Val leaks into fitness despite `JOINT_TRAIN_VAL=False`. No symbol-spread penalty → all rules symbol-locked. Infeasible rules survive on f1 (no trade_penalty).

## Files to Modify

1. `gpu_fuzzy_trader/config.py` — SORTINO_*, PHASE2_SUPPORT_PENALTY_WEIGHT_F1, new flags
2. `gpu_fuzzy_trader/phases/phase2_rule_pool.py` — `compute_phase2_objectives_from_metrics`, `_mutate`
3. `gpu_fuzzy_trader/evolution/evox_runner.py` — minor (f3 plumbing if needed)

## Detailed Changes

### C7: config.py line 437
```python
PHASE2_SUPPORT_PENALTY_WEIGHT_F1 = 0.4   # was 0.0
```

### H1: config.py lines 560, 565
```python
SORTINO_CAP = 20.0    # was 5.0
SORTINO_SCALE = 10.0  # was 5.0
```

### H2: Replace f3 with CV-fold minimum

config.py — add new flag:
```python
# PHASE2_F3_OBJECTIVE — third objective: "cv_fold_min" (default),
# "profit_factor" (fallback), "win_rate" (legacy, not recommended).
PHASE2_F3_OBJECTIVE = "cv_fold_min"
```

phase2_rule_pool.py::compute_phase2_objectives_from_metrics — replace f3 computation:
```python
# BEFORE: f3_val = win_rate (or robust_return if PHASE2_USE_TOTAL_RETURN_OBJ)

# AFTER:
f3_objective = str(getattr(_cfg, "PHASE2_F3_OBJECTIVE", "cv_fold_min"))
if f3_objective == "cv_fold_min":
    cv_fold_returns = metrics.get("_cv_fold_returns", [])
    if cv_fold_returns:
        f3_val = min(cv_fold_returns)
    else:
        # Fallback when CV not available: use profit_factor
        f3_val = profit_factor
elif f3_objective == "profit_factor":
    f3_val = profit_factor
else:  # "win_rate" legacy
    f3_val = win_rate
    if val_metrics is not None and _cfg.PHASE2_JOINT_TRAIN_VAL:
        ...  # existing logic
```

- Wire `_cv_fold_returns` into `_evaluate_chromosome` and the batch eval path.
- The `CvFoldValEvaluator` already caches fold engines. Call it per unique chromosome.
- Set `PHASE2_USE_TOTAL_RETURN_OBJ = False` (win-rate path disabled).

### C5: Symbol-spread penalty + mutation bias

phase2_rule_pool.py::compute_phase2_objectives_from_metrics — add after per_symbol_metrics:
```python
per_sym = metrics.get("per_symbol_metrics", {}) or {}
n_profitable_symbols = sum(
    1 for v in per_sym.values()
    if isinstance(v, dict) and float(v.get("net_pnl", 0.0)) > 0.0
)
min_symbols = int(getattr(_cfg, "PHASE2_MIN_PROFITABLE_SYMBOLS_PENALTY", 3))
if n_profitable_symbols < min_symbols:
    support_penalty += float(min_symbols - n_profitable_symbols) * 2.0
```

phase2_rule_pool.py::_mutate — symbol gene bias:
- Identify the symbol gene (by feature_info where name=="symbol" or mode indicates symbol).
- With probability `PHASE2_SYMBOL_GENE_DONT_CARE_PROB = 0.4`, force it to `dont_care`.
```python
symbol_gene_prob = float(getattr(_cfg, "PHASE2_SYMBOL_GENE_DONT_CARE_PROB", 0.4))
for k, fi in enumerate(feature_infos):
    if str(fi.get("name", "")).lower() == "symbol":
        if rng.random() < symbol_gene_prob:
            out[k] = int(dont_cares[k])
            continue
    # ... normal mutation
```

config.py — add:
```python
PHASE2_MIN_PROFITABLE_SYMBOLS_PENALTY = 3
PHASE2_SYMBOL_GENE_DONT_CARE_PROB = 0.4
```

### C6: Gate val penalties behind JOINT_TRAIN_VAL

phase2_rule_pool.py::compute_phase2_objectives_from_metrics (~lines 528–560):
- Wrap these in `if _cfg.PHASE2_JOINT_TRAIN_VAL:`:
  - `val_floor_penalty` computation (val return < floor, val PF < floor)
  - `_symbol_robustness_penalty(val_metrics)` addition
  - `support_penalty = max(support_penalty, SUPPORT_PENALTY_MAX)` for val trades < floor
- Add config flag:
```python
PHASE2_VAL_IN_FITNESS_PENALTY = False  # val only for pool admission, not fitness
```
- When both `JOINT_TRAIN_VAL=False` AND `VAL_IN_FITNESS_PENALTY=False`: val_metrics stored on metrics dict for reporting/admission only, NOT entering support_penalty.

### M3: Add trade_penalty to f1

phase2_rule_pool.py::compute_phase2_objectives_from_metrics:
```python
# BEFORE: f1 = -sortino_for_obj + 0.0*support_penalty + diversity_penalty
# AFTER:
f1 = (
    -sortino_for_obj
    + (_cfg.PHASE2_SUPPORT_PENALTY_WEIGHT_F1 * support_penalty)
    + diversity_penalty
    + trade_penalty   # NEW
)
```

## Acceptance Criteria

- [ ] `PHASE2_SUPPORT_PENALTY_WEIGHT_F1 = 0.4`, `SORTINO_CAP = 20.0`, `SORTINO_SCALE = 10.0`.
- [ ] `PHASE2_F3_OBJECTIVE` config flag exists; default `"cv_fold_min"`.
- [ ] f3 uses `min(cv_fold_returns)` — NOT `win_rate`.
- [ ] `PHASE2_USE_TOTAL_RETURN_OBJ = False`.
- [ ] Symbol-spread penalty active when `n_profitable_symbols < 3`.
- [ ] `_mutate` forces symbol gene to `dont_care` with prob 0.4.
- [ ] Val penalties gated behind `JOINT_TRAIN_VAL` or `VAL_IN_FITNESS_PENALTY`.
- [ ] f1 includes `trade_penalty`.
- [ ] New config flags: `PHASE2_MIN_PROFITABLE_SYMBOLS_PENALTY`, `PHASE2_SYMBOL_GENE_DONT_CARE_PROB`, `PHASE2_F3_OBJECTIVE`, `PHASE2_VAL_IN_FITNESS_PENALTY`.
- [ ] `evaluator_v5.ipynb` NOT modified.
- [ ] Tests pass: `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/ -x -q -k "phase2 or objective or rule_pool"`

## Verification

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/ -x -q -k "phase2 or objective or rule_pool or mutate"
```
