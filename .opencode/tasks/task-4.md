# Task 4 — Evaluator-failure-mode awareness

## Why
The evaluator reports metrics that the user's existing
`CPUBacktestEngine.simulate_rule_set` does NOT track:
- `raw_signal_count` (total matched entries before min-notional filter)
- `executed_trades` (kept the same name; the friend uses this)
- `skipped_min_notional_count`
- `max_simultaneous_positions`
- `max_total_open_exposure`
- `loss_count`, `time_closed_count`
- `account_ruined` (the friend calls this; mine uses a similar boolean)

The friend uses these to penalize a candidate whose raw signals are
mostly skipped by the evaluator's `MIN_POSITION_NOTIONAL = 1.0`
threshold. Without this awareness, my pipeline happily selects rules
that internally look profitable (return +5%, PF 1.3) but generate 0
PnL on test because 90% of their signals get skipped.

## Required reading
- `.opencode/plans/PLAN.md`
- `.opencode/CONTEXT.md` (JSON output contract)
- The friend's `EvaluatorV5BacktestEngine` in `friend_project/gpu_fuzzy_trader/rb_evaluator_v5.py` (the `_build_trade_outcome_single` and `simulate_rule_set` methods).
- The friend's `_evaluator_health_penalty` in `friend_project/gpu_fuzzy_trader/rb_governor.py` lines 60-83.
- My existing `gpu_fuzzy_trader/backtest/cpu_engine.py` (look for the `simulate_rule_set` method and the metrics it returns).

## Behavior changes

### Step 1 — Expose the new fields from `CPUBacktestEngine.simulate_rule_set`

The current return dict (in `cpu_engine.py`) is missing these. The
metrics ARE computed internally — the engine tracks `loss_count`,
`time_closed_count`, `account_ruined`, and the position-tracking
variables — but it does not put them in the return dict. Add them.

Required new fields in the return dict (all default to 0 / False / 0.0 if absent):
- `raw_signal_count: int` — total entries that matched (before min-notional filter).
- `executed_trades: int` — number of trades that actually opened (already present; verify).
- `skipped_min_notional_count: int` — number of entries skipped because `position_notional < MIN_POSITION_NOTIONAL`.
- `max_simultaneous_positions: int` — peak number of open positions at any point.
- `max_total_open_exposure: float` — peak total exposure (in dollars) at any point.
- `loss_count: int` — number of closed positions with negative net PnL.
- `time_closed_count: int` — number of positions closed at the 288-bar time limit.
- `account_ruined: bool` — equity reached 0 at any point.

The friend's engine additionally tracks `final_equity` and `profit_factor`; mine has these. Do not break existing fields.

### Step 2 — Add `_evaluator_health_penalty` helper

Port the friend's helper (in `rb_governor.py`) into a new
`gpu_fuzzy_trader/scoring/evaluator_health.py` module:

```python
def evaluator_health_penalty(
    metrics: dict,
    *,
    role: str = "valid",
) -> float:
    """Penalty for evaluator_v5 failure modes. Higher = worse.
    
    role: "train" or "valid" or "test". "test" gets a 1.5x multiplier.
    """
    raw = max(0, int(metrics.get("raw_signal_count", 0)))
    executed = max(0, int(metrics.get("executed_trades", 0)))
    skipped = max(0, int(metrics.get("skipped_min_notional_count", 0)))
    max_pos = max(0, int(metrics.get("max_simultaneous_positions", 0)))
    
    cfg = _cfg  # or use getattr with defaults
    
    if raw > 0:
        skip_ratio = skipped / raw
        exec_ratio = executed / raw
        ...
    if max_pos > cfg.RB_MAX_SIMULTANEOUS_POSITIONS:
        penalty += ...
    
    role_mult = 1.5 if role == "test" else 1.0
    return penalty * role_mult
```

Use the friend's `RB_*` config keys for thresholds and weights. I will
add the same keys (with `RB_` prefix) to my config:
```python
EVAL_HEALTH_MAX_SKIPPED_RATIO = 0.20
EVAL_HEALTH_MIN_EXECUTED_RATIO = 0.60
EVAL_HEALTH_SKIPPED_WEIGHT = 3500.0
EVAL_HEALTH_EXECUTED_WEIGHT = 2500.0
EVAL_HEALTH_MAX_SIMULTANEOUS_POSITIONS = 10
EVAL_HEALTH_MAX_POSITIONS_WEIGHT = 120.0
```

(I use the `EVAL_HEALTH_` prefix to avoid colliding with friend
project's `RB_` prefix when both are imported; my code shouldn't
import from the friend project.)

### Step 3 — Wire into Phase 3 and Phase 4 scoring

In `phase3_rule_set.py`: subtract
`evaluator_health_penalty(train_metrics, role="train") + evaluator_health_penalty(val_metrics, role="valid")`
from the per-rule-set score. Add a config flag `PHASE3_EVAL_HEALTH_WEIGHT = 1.0` (multiplier).

In `phase4_wf_optimizer.py`: same thing for the Optuna trial.

### Step 4 — Wire into `gate_positive_good` (Task 3's helper)

The friend has a `execution_ok(metrics)` check inside `_is_positive_good`:
```python
def execution_ok(m: dict) -> bool:
    raw = max(0, _i(m, "raw_signal_count", 0))
    if raw <= 0: return False
    skipped = max(0, _i(m, "skipped_min_notional_count", 0))
    executed = max(0, _i(m, "executed_trades", 0))
    max_skip = float(getattr(_cfg, "RB_MAX_SKIPPED_SIGNAL_RATIO", 0.20))
    min_exec = float(getattr(_cfg, "RB_MIN_EXECUTED_RAW_RATIO", 0.60))
    return (skipped / raw) <= max_skip and (executed / raw) >= min_exec
```

I will add an `execution_ok(metrics)` helper in the same
`evaluator_health.py` module, and add an OPTIONAL (default OFF)
execution-health check to `gate_positive_good`. Behind a config flag
`PHASE3_GATE_EXECUTION_HEALTH = True` (default ON). When ON, the
gate requires both train and val to pass `execution_ok`.

This is a small extension to Task 3's gate and ties in naturally with Task 4.

## Out of scope
- Do NOT change the JSON output format.
- Do NOT modify `evaluator_v5.ipynb`.
- Do NOT modify the GPU engine.
- Do NOT touch the existing gap-reject gate (`PHASE3_MAX_TRAIN_VAL_GAP_PCT`).
- Do NOT add Tasks 5-9 features.

## Acceptance criteria
1. `CPUBacktestEngine(...).simulate_rule_set(rule_set)` returns a dict
   with all 7 new fields populated (not just 0). Run a simple test
   that creates an engine, runs a rule set, and prints the new fields.
2. `from gpu_fuzzy_trader.scoring.evaluator_health import evaluator_health_penalty, execution_ok` works.
3. `evaluator_health_penalty({"raw_signal_count": 100, "executed_trades": 50, "skipped_min_notional_count": 50})` returns a positive penalty (because 50% skip ratio > 20% threshold).
4. `evaluator_health_penalty({"raw_signal_count": 100, "executed_trades": 80, "skipped_min_notional_count": 20})` returns 0 (skip ratio 20% = threshold).
5. `execution_ok({...})` returns `True` when skip ratio ≤ 0.20 and exec ratio ≥ 0.60.
6. `execution_ok({...})` returns `False` when skip ratio > 0.20.
7. Phase 3 per-rule-set score subtracts `(eval_health_penalty(train) + eval_health_penalty(val)) * PHASE3_EVAL_HEALTH_WEIGHT` from the base score.
8. Phase 4 Optuna trial does the same.
9. `gate_positive_good(train, val)` calls `execution_ok(train) and execution_ok(val)` when `PHASE3_GATE_EXECUTION_HEALTH=True`. Test this is wired up.
10. New unit test `tests/unit/test_evaluator_health.py` with ≥ 8 cases (health penalty under various metrics, execution_ok boundary cases, role multiplier, wire-in to phase3/4).
11. All existing tests still pass.
12. No changes to `evaluator_v5.ipynb` or the GPU engine.

## Constraints
- Stay on `feature/task-4-evaluator-health-penalty` (off `main` after task-3 is merged).
- 12.7 GiB RAM total.
- PEP 8, type hints, module logger.
- Use only existing third-party deps.
- The new module `gpu_fuzzy_trader/scoring/evaluator_health.py` is NEW — confirm the `scoring/` dir doesn't exist yet, and create it (with `__init__.py`).

## Files I will touch
- `gpu_fuzzy_trader/backtest/cpu_engine.py` — add 7 new fields to the metrics dict
- `gpu_fuzzy_trader/scoring/__init__.py` (new, empty)
- `gpu_fuzzy_trader/scoring/evaluator_health.py` (new) — `evaluator_health_penalty` + `execution_ok`
- `gpu_fuzzy_trader/config.py` — 6 new `EVAL_HEALTH_*` config keys + 2 new phase keys
- `gpu_fuzzy_trader/phases/phase3_rule_set.py` — wire penalty into per-rule-set score; add `execution_ok` to gate
- `gpu_fuzzy_trader/phases/phase4_wf_optimizer.py` — wire penalty into Optuna trial
- `tests/unit/test_evaluator_health.py` (new) — ≥ 8 cases
