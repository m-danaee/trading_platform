# Task 2: Fix GPU Engine PnL Timing — Defer PnL to Release Step

**Branch:** `fix/gpu-pnl-defer-to-release`
**Base:** `main`

## Goal

Fix the GPU backtest engine (`gpu_engine.py`) to credit trade PnL at position **release time** (close) instead of **entry time** (open), matching the CPU engine's correct semantics.

## Root Cause

In `_jax_simulate_equity_batch()`, the scan `step` function computes PnL from the entry bar's forward-looking labels and **immediately adds it to equity**:

```python
# CURRENT (WRONG): equity grows at ENTRY
gross_pnl = position_notional * (price_return_pct / 100.0)
net_pnl = gross_pnl - fee
new_equity = equity + net_pnl   # ← future return credited NOW
```

The `_jax_release_open_slots` function only frees exposure slots — it does NOT update equity. By contrast, the CPU engine (`cpu_engine.py:670`) correctly defers PnL: `equity += pos["net_pnl"]` happens only in `_release_due_positions`.

**Impact:** Equity grows on unrealized gains → next bar's position size is inflated → exponential compounding → 953% max_return with sequential sampling.

## Changes

### File: `gpu_fuzzy_trader/backtest/gpu_engine.py`

#### 1. Add `slot_pnl` to scan carry

Currently the carry has `slot_release` (int32) and `slot_notional` (float). Add a third parallel array:

```python
init_slot_pnl = jnp.zeros(max_slots, dtype=_JXF)  # NEW

init_carry = (
    init_cap, init_cap, _JXF(0.0), _JXF(0.0),  # equity, peak, dd, exposure
    jnp.int32(0), jnp.int32(0), _JXF(0.0), _JXF(0.0),  # wins, losses, gp, gl
    jnp.int32(0), jnp.int32(0),  # executed, skipped
    jnp.bool_(False),  # ruined
    _JXF(0.0), jnp.int32(0), _JXF(0.0),  # trade_return_sum, n_neg, neg_sq_sum
    init_slot_release, init_slot_notional, init_slot_pnl,  # ← ADD pnl
)
```

Also update the carry unpacking in `step()` and `final_carry`.

#### 2. Modify `_jax_release_open_slots` to return PnL to credit

Change signature to also accept `slot_pnl` and return `equity_delta` + updated stats:

```python
def _jax_release_open_slots(
    slot_release, slot_notional, slot_pnl,  # ← added slot_pnl
    open_exposure, current_row,
) -> tuple[slot_release, slot_notional, slot_pnl, open_exposure, equity_delta,
           wins_delta, losses_delta, gp_delta, gl_delta, trade_ret_sum_delta,
           n_neg_delta, neg_sq_sum_delta]:
```

Logic: for each releasing slot, sum its `slot_pnl` into `equity_delta`, count wins/losses, update Sortino running stats.

#### 3. Modify `_jax_open_slot` to store PnL

Add `net_pnl` parameter. Store it in `slot_pnl` at the free slot index (currently stores 0).

```python
def _jax_open_slot(
    slot_release, slot_notional, slot_pnl,  # ← added slot_pnl
    open_exposure, release_idx,
    position_notional, net_pnl,  # ← added net_pnl
    can_trade,
) -> tuple[slot_release, slot_notional, slot_pnl, open_exposure]:
```

#### 4. Remove entry-time equity/Sortino updates from scan `step`

Currently at entry time:
- `new_equity = equity + net_pnl` ← REMOVE (move to release)
- `new_wins`, `new_losses`, `new_gross_profit`, `new_gross_loss` ← REMOVE (move to release)
- `new_trade_return_sum`, `new_n_neg`, `new_neg_sq_sum` ← REMOVE (move to release)

After the fix:
- Entry only increments `executed` and updates `slot_*` arrays
- Release updates equity + all stats from `equity_delta` returned by `_jax_release_open_slots`

#### 5. Handle final unreleased positions after scan

After `lax.scan` completes, positions opened in the last 288 bars are still open. Credit their PnL to equity (they were held to data end). Use a simple loop over remaining active slots adding their PnL to final equity.

#### 6. Keep Sortino stats consistent

The Sortino ratio at the end uses `trade_return_sum`, `n_neg`, `neg_sq_sum` which must now be accumulated at release time (not entry time). Update these in `_jax_release_open_slots`.

## Key Constraints

- The scan step MUST release before trading (current order is correct: release, then check signals, then open)
- `max_drawdown` calculation depends on equity updates at release time — peak equity tracking must update when positions close
- `new_peak = max(peak_equity, equity + equity_delta)` after release
- `dd = (peak - new_equity) / peak * 100` after release + entry combined
- Position sizing for new entries must use equity AFTER released PnL is added (current order achieves this)

## Acceptance Criteria

1. GPU engine `total_return_pct` matches CPU engine within ±1% for the SAME rule on the SAME data slice
2. `win_rate`, `profit_factor`, `executed_trades` match CPU engine exactly
3. `max_drawdown_pct` matches CPU engine within ±5%
4. `sortino_ratio` matches CPU engine within ±5%
5. Existing Phase 2 tests pass: `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/ -x -q`
6. New test: `test_gpu_cpu_return_parity` — compare GPU and CPU engine for 10 random chromosomes

## Files Changed
- `gpu_fuzzy_trader/backtest/gpu_engine.py` (primary change)
- `tests/test_gpu_engine.py` (new parity test)
