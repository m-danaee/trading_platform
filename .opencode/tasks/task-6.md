# Task 6 — `fix/oos-honesty-and-amplifier-tune` (OOS honesty + conservative amplifier de-tune)

## Branch
`fix/oos-honesty-and-amplifier-tune` (from latest `main`).

## Problem
Two issues found in the 2026-06-27 completed run:

1. **Phase 5 test-set leakage (CRITICAL):** `PHASE5_REMOVE_NEGATIVE_PNL_RULES=True`
   removes rules with negative PnL **on the test trade log**, then re-evaluates on
   test. This is post-hoc selection on the OOS set — the reported 6.21%/6.53% are
   inflated. Log evidence: "removed 2 negative-PnL rules, kept 6" (long) and
   "removed 6 negative-PnL rules, kept 2" (short) — both decided using test PnL.

2. **RB Governor amplifier over-fits LONG to val (HIGH):** The profit amplifier
   weights val 1.6× over train (`RB_PROFIT_AMP_VALID_WEIGHT=1.60`) and runs 2
   capital-reallocation passes (`RB_PROFIT_AMP_CAPITAL_PASSES=2`), pushing
   capital_pct to 25-35%. Result: LONG val=14.9% → test=6.2% (8.7% gap) while
   SHORT val=6.5% → test=6.5% (0% gap). The asymmetry shows LONG rules were
   val-fitted by the amplifier.

## Required Changes (all config-only, no code changes)

### Change 1 — Stop Phase 5 test leakage (CRITICAL)
**File:** `gpu_fuzzy_trader/config.py` (line ~1419)
```python
# BEFORE
PHASE5_REMOVE_NEGATIVE_PNL_RULES = True
# AFTER
PHASE5_REMOVE_NEGATIVE_PNL_RULES = False
```
Rule removal already happens upstream (RB Governor selects on train+val). Phase 5
must ONLY evaluate — never select. With this off, the reported test metrics are
honest OOS (no post-hoc cleanup).

NOTE: Verify that when `False`, Phase 5 still reports all rules and does NOT
crash on the `cleaned` flag path. Read `phase5_oos.py:167-180` — if `cleaned`
is False, the re-evaluation block is skipped (correct). Confirm no test fails.

### Change 2 — Stop over-weighting val in amplifier (HIGH)
**File:** `gpu_fuzzy_trader/config.py` (line ~1619)
```python
# BEFORE
RB_PROFIT_AMP_VALID_WEIGHT: float = 1.60
# AFTER
RB_PROFIT_AMP_VALID_WEIGHT: float = 1.00
```
Equal-weight train and val in the amplifier objective. This stops the amplifier
from greedily fitting val at the expense of train robustness. Train and val are
both holdouts from Phase 2 (after the JOINT_TRAIN_VAL=False fix), so equal
weighting is the honest choice.

### Change 3 — Reduce capital amplification passes (HIGH)
**File:** `gpu_fuzzy_trader/config.py` (line ~1635)
```python
# BEFORE
RB_PROFIT_AMP_CAPITAL_PASSES: int = 2
# AFTER
RB_PROFIT_AMP_CAPITAL_PASSES: int = 1
```
One capital-reallocation pass instead of two. This caps how aggressively
capital_pct gets pushed up based on val. The amplifier still selects rules and
does one reallocation pass — just less aggressive.

### Docs
Update `README.md` config table for all three changed keys with new values +
rationale.

## Acceptance Criteria
1. `PHASE5_REMOVE_NEGATIVE_PNL_RULES == False`.
2. `RB_PROFIT_AMP_VALID_WEIGHT == 1.00`.
3. `RB_PROFIT_AMP_CAPITAL_PASSES == 1`.
4. `config.py` asserts at import still pass (`.venv/bin/python -c "import gpu_fuzzy_trader.config"`).
5. `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit -q` passes (no NEW failures beyond the 2 pre-existing MAX_CONDITIONS ones).
6. If any test hard-codes the old values (1.60, 2, True), update them to the new values — do NOT weaken correctness assertions.
7. README config table updated for all 3 keys.

## Verification Commands
```
cd /home/danaee/trading_platform
.venv/bin/python -c "from gpu_fuzzy_trader import config as c; print('P5_REMOVE=',c.PHASE5_REMOVE_NEGATIVE_PNL_RULES); print('AMP_VAL_W=',c.RB_PROFIT_AMP_VALID_WEIGHT); print('AMP_CAP_PASSES=',c.RB_PROFIT_AMP_CAPITAL_PASSES)"
.venv/bin/python -c "import gpu_fuzzy_trader.config; print('config asserts OK')"
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit -q 2>&1 | tail -5
```
Do NOT run the full pipeline.

## Target Files
- `gpu_fuzzy_trader/config.py` (3 values)
- `README.md` (config table)
- any test hard-coding old values.

## Notes
- All three changes are config-only — no production code changes.
- This is intentionally conservative: the amplifier stays enabled (it does
  valuable rule selection + monthly certificate), just less aggressive on val.
- After this, the user should re-run on Colab and compare:
  (a) test returns are now HONEST (no post-hoc rule removal)
  (b) LONG val→test gap should shrink (less val-fitting)
- Symbol-pinning (rules have "symbol is X" conditions) is deferred to a future
  round — wait to see if these 3 changes are enough first.
