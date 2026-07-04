# Nexus Context

**Updated:** 2026-07-04
**base_branch:** `main`
**branch_policy:** isolated
**execution_mode:** direct
**status:** ACTIVE — Task 24 in progress

## Active Task
- **Task 24** 🔄 in progress — Fix `_sample_df` stride sampling bug (break temporal causality)

## Branch
- `feature/fix-contiguous-sampling` (off `main`)

## Problem
`_downsample_chronological` in `gpu_fuzzy_trader/phases/phase2_rule_pool.py:300` uses
`np.linspace(0, total-1, num=n_rows)` which selects **strided** rows (every Nth bar).
For trading backtests this is a **critical bug**: bars are temporal events and the
backtest engine iterates them sequentially for position management, exposure release,
and intraday-pattern recognition. Skipping bars causes the engine to miss intermediate
candles → positions opened/closed on skipped bars are silently lost.

## Required Behavior
Per symbol, pick a **random start index** bounded so the requested `n_rows` fit
forward (`start ∈ [0, len(sym_df) - n_rows]`), then take a **contiguous slice**
`sym_df.iloc[start:start+n_rows]`. Use the caller's `random_state` for reproducibility
(do not ignore it as the current code does).

## Acceptance Criteria
1. Rows taken per symbol are contiguous (no stride gaps).
2. Random start index is bounded so `n_rows` always fit forward.
3. Same `random_state` → same output (deterministic).
4. Different `random_state` → different output.
5. `random_state=None` defaults to a stable seed (e.g. 0) for reproducibility.
6. Unit tests updated/added to cover new contract.
