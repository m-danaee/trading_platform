# Task 28 — Feasibility collapse observability (D)

## Branch
`feat/feasibility-observability` (from `main`)

## Problem
The run log shows `valid_rules=2-4` out of `pop=200` — 95-99% of the
population fails feasibility. The number is reported but the
**breakdown** (which of the 9 feasibility gates is killing the most
individuals) is not visible. Without this breakdown, we don't know
whether to:
- Relax a specific floor (e.g., `PHASE2_PROFIT_FACTOR_FLOOR` 1.15 → 1.05)
- Investigate the `_require_last_fold_positive` gate (kills any rule with
  val_ret ≤ 0, common in random initial populations)
- Look at the train/val gap gate (a different problem entirely)

The overfit fixes from Tasks 1-3 and the f3 blend from Task 26 will
help, but floor relaxation is gated on having actual data first.

## Files to Edit
- `gpu_fuzzy_trader/phases/phase2_support.py` — add a new function
  `_feasibility_gate_failures` that returns per-gate failure flags
- `gpu_fuzzy_trader/evolution/evox_runner.py` — call the new function
  and log the breakdown when `val_count < 10`

## Required Behavior

### Source change 1: `phase2_support.py`
Add a new function (place it after `_passes_pool_admission_impl`):

```python
def _feasibility_gate_failures(
    train_metrics: dict,
    val_metrics: dict | None,
    *,
    n_valid_rows: int | None = None,
) -> dict[str, int]:
    """Return per-gate failure flags for the 9 pool-admission gates.

    Each value is 1 if the rule FAILS that gate, 0 if it passes. A rule
    that passes all gates returns all zeros. A rule that fails multiple
    gates has multiple 1s (gates are AND-combined, but a rule can fail
    several simultaneously).

    The 9 gates mirror `_passes_pool_admission_impl` exactly:
    - train_trade_floor   : train_trades < train_floor
    - train_return_floor  : train_ret <= train_ret_min
    - train_pf_floor      : train_pf < pf_floor
    - val_required        : val_metrics is None (cannot evaluate val gates)
    - val_ret_positive    : PHASE2_REQUIRE_LAST_FOLD_POSITIVE and val_ret <= 0
    - val_trade_floor     : val_trades < min_val_trades
    - val_return_floor    : val_ret <= val_ret_min
    - val_pf_floor        : val_pf < pf_floor
    - train_val_gap       : train_ret - val_ret > max_gap

    Args:
        train_metrics: Train dict with executed_trades, total_return_pct,
            profit_factor, etc.
        val_metrics: Validation dict (same shape) or None.
        n_valid_rows: Optional row count for island-floor scaling.

    Returns:
        Dict mapping gate name to 0 (passed) or 1 (failed).
    """
```

Implementation: call `_pool_admission_floors(n_valid_rows)` to get
the scaled floors, then mirror the gate logic of
`_passes_pool_admission_impl` exactly, returning 1 for each failed gate
and 0 for each passed gate. Use the SAME config lookups
(`PHASE2_REQUIRE_LAST_FOLD_POSITIVE`, etc.) as the original.

### Source change 2: `evox_runner.py`
Add a new log line after the existing per-generation
`maybe_log_generation` call (around line 2207, after the call's
`logger.info("... valid_rules=%d ...", val_count, ...)` log if any, or
after the `maybe_log_generation` block).

The new log line should:
1. Be triggered ONLY when `val_count < 10` (no log spam in healthy runs)
2. Iterate over ALL 200 individuals in `metrics_cache` (not just Pareto)
3. For each, call `_feasibility_gate_failures(metrics_cache[i], val_m)`
   where `val_m = _val_metrics_from_cache(metrics_cache[i])`
4. Sum the per-gate failures across all individuals
5. Log a single line with the breakdown, e.g.:
   ```
   Phase 2 [long] gen 5: feasibility collapse breakdown (valid_rules=4 < 10):
     train_trade_floor=12 train_return_floor=8 train_pf_floor=45
     val_required=0 val_ret_positive=178 val_trade_floor=23
     val_return_floor=15 val_pf_floor=42 train_val_gap=33
   ```

Use a WARNING level (not INFO) so it stands out in logs.

### Threshold rationale
- `val_count < 10` is arbitrary but matches the existing "valid_rules=2-4"
  observation in the run log. Could be made configurable via
  `PHASE2_FEASIBILITY_LOG_THRESHOLD` but that's optional.
- Hardcoded to 10 for simplicity. If a future use case needs a different
  threshold, easy to change in one place.

## Tests
Add a new test in `tests/unit/test_phase2_support.py` (or wherever
feasibility tests live — verify the test file exists first):

```python
def test_feasibility_gate_failures_all_pass(self):
    """A rule passing all gates returns all-zero dict."""
    train = {"executed_trades": 100, "total_return_pct": 5.0, "profit_factor": 2.0}
    val = {"executed_trades": 50, "total_return_pct": 3.0, "profit_factor": 1.5}
    result = _feasibility_gate_failures(train, val)
    assert all(v == 0 for v in result.values())
    assert len(result) == 9  # all 9 gates

def test_feasibility_gate_failures_train_floor(self):
    """A rule with too few train trades fails train_trade_floor."""
    train = {"executed_trades": 5, "total_return_pct": 5.0, "profit_factor": 2.0}
    val = {"executed_trades": 50, "total_return_pct": 3.0, "profit_factor": 1.5}
    result = _feasibility_gate_failures(train, val)
    assert result["train_trade_floor"] == 1
    assert result["val_ret_positive"] == 0
    # ... other gates 0

def test_feasibility_gate_failures_val_negative(self):
    """A rule with val_ret <= 0 fails val_ret_positive gate."""
    train = {"executed_trades": 100, "total_return_pct": 5.0, "profit_factor": 2.0}
    val = {"executed_trades": 50, "total_return_pct": -1.0, "profit_factor": 1.5}
    result = _feasibility_gate_failures(train, val)
    assert result["val_ret_positive"] == 1
```

The exact test count can be 3-5, covering the most common failure modes
(train_floor, val_floor, val_ret_positive, multi-gate failure, None
val_metrics). Cover what you can verify cleanly.

## Acceptance criteria
1. New function `_feasibility_gate_failures` exists in
   `phase2_support.py` with the signature and behavior above
2. Function returns a dict of 9 keys, all values 0 or 1
3. New log line in `evox_runner.py` after `maybe_log_generation`,
   triggered only when `val_count < 10`
4. Log level is WARNING
5. Log line shows the 9-gate failure counts aggregated across all
   `pop_size` individuals
6. 3+ new tests in `test_phase2_support.py` (or appropriate file) covering
   the function's behavior
7. All existing tests pass with `PYTEST_LOW_MEMORY=1`
8. `git diff` is limited to `phase2_support.py` (new function + tests)
   and `evox_runner.py` (new log block) and the test file

## Out of scope
- Do NOT change any floor values in config.py
- Do NOT change `_passes_pool_admission_impl` behavior
- Do NOT change `_raw_feasibility_violation_score` behavior
- Do NOT make the threshold (10) configurable (keep hardcoded for now)
- Do NOT add a per-symbol breakdown (just the 9 pool gates)
- Do NOT add an aggregated end-of-run summary (per-generation is enough)
