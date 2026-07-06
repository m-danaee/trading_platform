# Task 27 — Document intentional evaluator_v5 parity (C)

## Branch
`docs/time-exit-evaluator-parity` (from `main`)

## Problem
The backtest engines (`cpu_engine.py` and `gpu_engine.py`) return raw,
uncapped `close_ret` (or `-close_ret`) on the time-exit branch — when
neither TP nor SL triggers and the trade holds to MAX_HOLD_CANDLES.

This is INTENTIONALLY uncapped, NOT an oversight. The project's ground
truth for rule testing is `evaluator_v5.ipynb` (per AGENTS.md:
"Your evaluation must be based on evaluator_v5.ipynb because testing my
rule sets is based on this file!"). The evaluator at `evaluator_v5.ipynb:958, 971`
also does not cap. Adding a cap to the engines would create a
divergence that invalidates the user's standalone OOS check.

A previous attempt to add a cap (commit `46cb88a`, `MAX_TIME_EXIT_RETURN_PCT=50.0`)
was reverted the same day (commit `072c527`) — likely for this parity
reason, though the revert commit message is silent.

Without an in-code comment explaining this design choice, future
contributors (including the same person a year from now) will likely
re-encounter the same concern and re-implement the cap, hitting the
same reversion cycle.

## Files to Edit (comment-only — NO behavior change)
- `gpu_fuzzy_trader/backtest/cpu_engine.py` — add a comment block above
  lines 572 (long time-exit) and 584 (short time-exit)
- `gpu_fuzzy_trader/backtest/gpu_engine.py` — add a comment block above
  lines 235 (long time-exit) and 244 (short time-exit)

## Required Behavior

### Comment block template (use this exact text for both files)

```python
            # INTENTIONALLY UNCAPPED time-exit return.
            # This branch returns the raw close_ret/-close_ret when a trade
            # holds to MAX_HOLD_CANDLES without hitting TP or SL. A previous
            # attempt to cap this with MAX_TIME_EXIT_RETURN_PCT (commit
            # 46cb88a, default 50.0) was reverted (commit 072c527) to preserve
            # parity with evaluator_v5.ipynb (lines 958, 971), which is the
            # user's ground truth for rule testing per AGENTS.md. Re-applying
            # any cap here would create a divergence that invalidates the
            # user's standalone OOS check. If you want to re-introduce a cap,
            # update evaluator_v5.ipynb FIRST and document the change.
```

### Placement
- **cpu_engine.py line 572** (long): add the comment block immediately
  above `return float(s_close), "Time_288"`. Indent the comment block to
  match the surrounding code (8 spaces inside the `if self.trade_direction == "long":` block).
- **cpu_engine.py line 584** (short): add the SAME comment block above
  `return float(-s_close), "Time_288"`. Indent to match (8 spaces).
- **gpu_engine.py line 235** (long, in `jnp.where` chain): add the
  comment block above the `jnp.where(long_hit_sl, -sl_f, close_ret)`
  sub-expression. Indent to match (4 spaces inside the function).
- **gpu_engine.py line 244** (short, in `jnp.where` chain): add the SAME
  comment block above the `-close_ret` sub-expression. Indent to match.

## Why comment-only
- This is a documentation change. No code logic changes.
- No tests need to be added or modified.
- The existing tests must continue to pass unchanged.

## Acceptance criteria
1. 4 comment blocks added (2 per file), all using the template text above
2. The comments are placed correctly (immediately above the time-exit return statements)
3. The comments are correctly indented to match the surrounding code
4. NO code changes — only comments
5. `git diff` shows comment-only changes (the implementer can verify with `git diff --stat` — only comment lines added, no logic changes)
6. All existing tests pass with `PYTEST_LOW_MEMORY=1`
7. `git diff main...docs/time-exit-evaluator-parity` is comment-only

## Out of scope
- Do NOT change any code logic
- Do NOT add a cap to close_ret (this is the intentional behavior)
- Do NOT add a config flag for the cap
- Do NOT change evaluator_v5.ipynb
- Do NOT touch the test files
