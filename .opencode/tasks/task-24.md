# Task 24 — Fix `_sample_df` stride sampling bug

## Problem
`gpu_fuzzy_trader/phases/phase2_rule_pool.py:300` (`_downsample_chronological`)
implements **stride downsampling** via `np.linspace(0, total-1, num=n_rows)`. This
violates temporal causality required by the backtest engine:

- Bars are sequential temporal events. Position state, exposure release, and
  intraday patterns depend on every consecutive bar being processed.
- Stride sampling skips intermediate bars → mid-stride positions/exits are lost.
- Intraday time-of-day patterns are destroyed.

The function's docstring even says "chronological order" but the implementation
is "chronological stride" — not contiguous.

## Required Behavior

For each symbol's chronological bars (length `N_sym`):

1. Compute `n = min(rows_per_sym, N_sym)` (cap when symbol has fewer rows).
2. If `n == N_sym` → return the symbol's full chronological bars (no random pick).
3. Else pick `start = rng.integers(0, N_sym - n + 1)` (uniform, inclusive bound).
4. Return contiguous slice `sym_df.iloc[start : start + n]`.

The `random_state` parameter on `_sample_df` (int, Generator, or None) must be
honored:
- `None` → use `np.random.default_rng(seed=0)` for reproducibility.
- `int` → `np.random.default_rng(seed=random_state)`.
- `Generator` → use it directly.

## Files to Edit

- `gpu_fuzzy_trader/phases/phase2_rule_pool.py` (function + docstring)
- `tests/unit/test_phase2_rule_pool.py` (update `test_sampling_is_deterministic`,
  add `test_sampling_is_contiguous`)

## Acceptance Criteria

1. ✅ Rows per symbol are contiguous (no gaps in `_symbol_bar_index` or `datetime`).
2. ✅ Random start bounded so `n_rows` always fits forward (no IndexError).
3. ✅ Same `random_state` int → same rows.
4. ✅ Different `random_state` → different rows (with high probability).
5. ✅ `random_state=None` → deterministic (uses default seed).
6. ✅ All existing `_sample_df` tests still pass (after updates).
7. ✅ Targeted test run (`tests/unit/test_phase2_rule_pool.py::TestSampleDf`)
   passes with `PYTEST_LOW_MEMORY=1`.

## Out of Scope

- Do NOT change `PHASE1_SAMPLING_TOTAL` or other config values.
- Do NOT change the evaluator (`evaluator_v5.ipynb`).
- Do NOT change call sites in `phase2_island_scheduler.py` or
  `phase2_rule_pool.py` (their `random_state=seed` calls just start working).
