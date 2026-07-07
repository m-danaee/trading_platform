# Task 2: Return-Concentration 4th NSGA Objective

## Task ID
`task-2` (second of 12 tasks in the 2026-07-07 audit fix plan)

## Title
Return-Concentration 4th NSGA Objective

## Goal
Fix audit finding #2 (uncapped time-exit returns drive f3 toward
outlier rules; max_robust_return_pct=107.52% in 2026-07-07 log).
The cleanest fix without breaking evaluator_v5.ipynb parity (per
AGENTS.md) is to add a 4th NSGA-III objective: return concentration
ratio, defined as `max_single_trade_pnl / max(sum_positive_trade_pnl, ε)`.
This kills rules whose edge is one outlier trade without changing
the time-exit return branch.

## Audit Citation
- Confirmed by static inspection: `backtest/gpu_engine.py:233` and
  `backtest/cpu_engine.py:573` return raw `close_ret` (intentional,
  per the comments; cannot cap without breaking evaluator parity).
- Run log evidence: `max_robust_return_pct=107.52%` in long cluster
  (one outlier trade drove the entire front).

## Target Files
- `gpu_fuzzy_trader/backtest/gpu_engine.py`
  (track per-trade PnL, expose `max_single_trade_pnl`,
   `sum_positive_trade_pnl`, `sum_negative_trade_pnl` in metrics)
- `gpu_fuzzy_trader/backtest/cpu_engine.py` (same additions;
   the existing `stats["gross_profit_sum"]` / `gross_loss_sum` are
   the seeds; add per-trade max tracking)
- `gpu_fuzzy_trader/phases/phase2_rule_pool.py` (add f4 calc in
  `_evaluate_chromosome` and `_finalize_objectives`)
- `gpu_fuzzy_trader/evolution/evox_runner.py` (bump
  `np.full((pop_size, 3), inf)` and `_get_reference_vectors(pop_size, 3, ...)`
  calls to 4 objectives)
- `gpu_fuzzy_trader/config.py` (add `PHASE2_F4_ENABLED` flag, default
  True; `PHASE2_F4_CONCENTRATION_FLOOR`, default 0.5 — reject if
  f4 > 0.5 at pool admission; `PHASE2_N_OBJECTIVES=4`)
- `tests/unit/test_phase2_rule_pool.py` (f4 tests)
- `tests/property/test_cpu_engine_properties.py` (f4 monotonicity
  property test)

## Current Behavior
- `objectives.shape == (pop_size, 3)` in 5+ places in evox_runner.py
- `_get_reference_vectors(pop_size, 3, rng)` hardcodes 3 objectives
- `f1 = -Sortino + support + diversity` (no overfit_gap, no trade_penalty)
- `f2 = DD + support + trade_penalty + DD_gate` (no overfit_gap)
- `f3 = -robust_return + support + cond + overfit_gap` (no trade_penalty)
- Engines return aggregate `gross_profit_sum` and `gross_loss_sum`
  but NOT per-trade PnL list or max single-trade

## Scope
1. **Engine metrics additions** (gpu_engine.py + cpu_engine.py):
   - Track `max_single_trade_pnl` per rule evaluation.
   - Track `sum_positive_trade_pnl` and `sum_negative_trade_pnl`.
   - Add all three to the returned metrics dict.
   - Do NOT change the time-exit return branch (parity requirement).
2. **f4 calculation** (phase2_rule_pool.py):
   - `f4_val = max_single_trade_pnl / max(sum_positive_trade_pnl, ε)`
   - Default ε = 1e-6 (sum_positive=0 → f4=0)
   - All aggregates come from the BOTH train and val engines when
     JOINT_TRAIN_VAL is enabled; use min(train_f4, val_f4) like f3.
   - When `executed_trades < MIN_TRADE_POOL_FLOOR`, return f4 = 0
     (no concentration issue when too few trades).
3. **4-objective NSGA-III** (evox_runner.py):
   - Bump all `np.full((pop_size, 3), inf)` and `np.full(3, inf)` to 4.
   - Bump `_get_reference_vectors(pop_size, 3, rng)` to 4.
   - Bump NSGA-III reference vector count: with 4 objectives,
     pop=120 needs more reference vectors; check the existing
     `PHASE2_NSGA3_REF_PARTITIONS` (or similar) and adjust.
   - Verify the EvoX `non_dominate_rank` call handles 4 objectives
     (it should — the function is generic).
4. **Pool admission** (phase2_rule_pool.py + phase2_support.py):
   - Add `f4` to `_feasibility_gate_failures` as a 10th gate:
     reject if `f4 > PHASE2_F4_CONCENTRATION_FLOOR=0.5`.
   - Update the gate dictionary keys list.
5. **Config flags** (config.py):
   - `PHASE2_F4_ENABLED = True`
   - `PHASE2_F4_CONCENTRATION_FLOOR = 0.5`
   - `PHASE2_N_OBJECTIVES = 4` (centralize so all `np.full(..., 4)`
     and reference vector calls use this)
6. **Regression guard**:
   - `PHASE2_F4_ENABLED = False` must preserve current 3-objective
     behavior exactly. The implementer should gate the f4 computation,
     NOT hardcode 4 objectives in numpy/EvoX calls.

## Acceptance Criteria
1. A rule with 80 trades where 1 trade = +60% and 79 avg -0.5%
   receives `f4 ≈ 0.85` (60 / 60 = 1.0 if the one trade is the only
   positive; if some of the 79 are positive at -0.5 they're actually
   negative — example: 1 trade +60% and 79 trades avg -0.5% means
   sum_positive=60, f4=1.0).
2. A rule with uniform +1% across 80 trades receives `f4 ≈ 0.0125`
   (1.0 / 80).
3. When `PHASE2_F4_ENABLED = False`, the behavior is bit-identical
   to the prior 3-objective flow (regression test required).
4. The `_feasibility_gate_failures` dict gains an `f4_concentration`
   key when f4 > 0.5.
5. `_get_reference_vectors` correctly returns 4D reference vectors
   when called with `n_objs=4`; the resulting NSGA-III ranking still
   has 4 axes.
6. All existing tests pass; no regression in the prior 76-test
   window_rotation suite OR the existing 130K-line test_phase2_rule_pool.py.

## Verification
Run only related unit tests with `PYTEST_LOW_MEMORY=1` and `.venv`:

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_rule_pool.py -q
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_window_rotation.py -q
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_island_scheduler.py -q
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/property/test_cpu_engine_properties.py -q
```

If property tests are too slow, mark them `uses_jax` so the
low-memory fixture clears caches between runs.

## Notes
- Do NOT modify `evaluator_v5.ipynb`.
- Do NOT modify the time-exit return branch in either engine.
- Do NOT touch `rb_governor.py`, `phase2_island_scheduler.py`
  (except `_get_reference_vectors` calls in `evox_runner.py` if
  they live there), or other phase files.
- This task changes objective geometry — the prior 3-objective
  Pareto fronts will look different. That's expected. Task-3
  (RB Governor walk-forward) and downstream will be the proof
  that f4 actually improves OOS.
- If `PHASE2_NSGA3_REF_PARTITIONS` needs to change, the implementer
  must NOT silently break the 3-objective case (gate on
  `PHASE2_N_OBJECTIVES`).
- If implementation reveals the 4-objective geometry is degenerate
  (e.g., EvoX non_dominate_rank fails), return BLOCKED with the
  exact error and a minimal repro rather than shipping a partial fix.
