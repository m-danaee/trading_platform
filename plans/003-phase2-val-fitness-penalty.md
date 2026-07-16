# Plan 003: Align Phase2 fitness with deployable via val penalties

> **Executor instructions**: Follow step by step. Run every verification command before proceeding. On STOP conditions, stop and report. Do **not** update `plans/README.md` (reviewer maintains index).
>
> **Drift check (run first)**: `git diff --stat 425f469..HEAD -- gpu_fuzzy_trader/config.py gpu_fuzzy_trader/phases/phase2_rule_pool.py tests/unit/test_anti_overfit_config.py tests/unit/test_phase2_rule_pool.py`
> Mismatch with "Current state" excerpts → STOP.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: MED
- **Depends on**: none (prefer after 001 to avoid comment churn; independent of 002)
- **Category**: perf (search quality / anti-overfit)
- **Planned at**: commit `425f469`, 2026-07-16

## Why this matters

Colab Phase2 ran with train-only fitness (`PHASE2_JOINT_TRAIN_VAL=False`, `PHASE2_VAL_IN_FITNESS_PENALTY=False`). Val metrics were computed for reporting/admission but **did not steer** NSGA-III. Result: huge train≫val gaps, near-zero deployable counts while search ranked train-feasible junk, and `objective_corr_f1_f3`≈0.97–1.0 when return-based objectives collapse.

Flipping `PHASE2_JOINT_TRAIN_VAL=True` alone is **rejected**: config documents double-counting the same holdout in fitness and pool gates (leak). The better lever already in code is `PHASE2_VAL_IN_FITNESS_PENALTY=True`, which enables overfit-gap and val floor penalties in fitness while keeping train-primary Sortino/PF objectives.

## Current state

[`gpu_fuzzy_trader/config.py`](gpu_fuzzy_trader/config.py) (~790–810):

```python
PHASE2_JOINT_TRAIN_VAL = False
PHASE2_VAL_IN_FITNESS_PENALTY = False
```

Overfit gap weight already set (~683–695):

- `PHASE2_OVERFIT_GAP_PENALTY_WEIGHT = 15.0`
- `PHASE2_OVERFIT_GAP_PCT_THRESHOLD = 8.0`

But gap only applies when joint **or** val-in-fitness is True — see [`phase2_rule_pool.py`](gpu_fuzzy_trader/phases/phase2_rule_pool.py) (~968–984):

```python
and (_cfg.PHASE2_JOINT_TRAIN_VAL or getattr(_cfg, "PHASE2_VAL_IN_FITNESS_PENALTY", False))
```

Same gate for val floor / support / symbol_robustness penalties (~791–841).

f3 already prefers profit_factor path:

- `PHASE2_USE_TOTAL_RETURN_OBJ = False` (~552) — keep this (avoids f1≈f3 when joint is off).

Bundle lock test [`tests/unit/test_anti_overfit_config.py`](tests/unit/test_anti_overfit_config.py) asserts:

- `PHASE2_VAL_IN_FITNESS_PENALTY is False` (line 11)
- `PHASE2_JOINT_TRAIN_VAL is False` (line 9)
- `PHASE2_USE_TOTAL_RETURN_OBJ is False` (line 35)

Existing coverage for gap-with-penalty-on: `tests/unit/test_phase2_rule_pool.py` (~1910+) monkeypatches `PHASE2_VAL_IN_FITNESS_PENALTY=True`.

**AGENTS.md**: `.venv`; `PYTEST_LOW_MEMORY=1`; related tests only; no full pipeline on WSL.

## Commands you will need

| Purpose         | Command                                                                                                                                                                                                   | Expected on success         |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| Bundle test     | `PYTEST_LOW_MEMORY=1 .venv/bin/pytest tests/unit/test_anti_overfit_config.py -q`                                                                                                                          | pass after assertion update |
| Fitness tests   | `PYTEST_LOW_MEMORY=1 .venv/bin/pytest tests/unit/test_phase2_rule_pool.py tests/unit/test_phase2_stage.py tests/unit/test_conflict_table_validation.py -q`                                                | all pass                    |
| Import defaults | `.venv/bin/python -c "from gpu_fuzzy_trader import config as c; assert c.PHASE2_JOINT_TRAIN_VAL is False; assert c.PHASE2_VAL_IN_FITNESS_PENALTY is True; assert c.PHASE2_USE_TOTAL_RETURN_OBJ is False"` | exit 0                      |

## Suggested executor toolkit

- Invoke **codelookup** on `PHASE2_VAL_IN_FITNESS_PENALTY`, `_compute_objectives` / fitness builders in `phase2_rule_pool.py`, and callers that assume penalty-off defaults.
- Update every assertion that hard-codes the old default.

## Scope

**In scope**:

- `gpu_fuzzy_trader/config.py`: set `PHASE2_VAL_IN_FITNESS_PENALTY = True`; refresh comments to document the anti-leak rationale (joint stays False).
- `tests/unit/test_anti_overfit_config.py`: flip assertion for val-in-fitness.
- Any unit test that asserts the **default** is False (search before editing).
- Optional comment-only clarity in `phase2_rule_pool.py` module docstring if it still says penalties are off by default — only if inaccurate after change.

**Out of scope**:

- Setting `PHASE2_JOINT_TRAIN_VAL=True`
- Changing pop/gens, island mode, Phase1 (plan 004)
- RB governor (plan 002)
- Dataset rename (001)
- Large objective redesign (moving support penalty off f1) — deferred unless STOP shows `objective_corr_f1_f3` still ~1.0 with PF f3 + gap on f3 only after this change lands in a Colab run
- `evaluator_v5.ipynb`

## Git workflow

- Branch: `advisor/003-phase2-val-fitness-penalty`
- Commit: `fix: enable Phase2 val-in-fitness penalties without joint train/val`
- Do NOT push or open PR unless instructed

## Steps

### Step 1: Codelookup + inventory default assertions

Search:

```bash
rg -n 'PHASE2_VAL_IN_FITNESS_PENALTY' gpu_fuzzy_trader tests
```

List every place that assumes default False.

**Verify**: inventory complete before edits.

### Step 2: Flip config default + comments

In `config.py`:

1. Set `PHASE2_VAL_IN_FITNESS_PENALTY = True`.
2. Keep `PHASE2_JOINT_TRAIN_VAL = False`.
3. Keep `PHASE2_USE_TOTAL_RETURN_OBJ = False`.
4. Rewrite the comment block (~801–809) to state:
   - True = val overfit-gap / val floors / val support / val symbol_robustness enter fitness.
   - Joint remains False to avoid double-counting holdout in joint Sortino/return **and** admission gates.
   - If feasible set starves, retune floors/weights — do not silently flip back without measuring deployable counts.

Do **not** change gap weight/threshold in this plan (already 15.0 / 8.0).

**Verify**:

```bash
.venv/bin/python -c "from gpu_fuzzy_trader import config as c; assert c.PHASE2_JOINT_TRAIN_VAL is False; assert c.PHASE2_VAL_IN_FITNESS_PENALTY is True; assert c.PHASE2_USE_TOTAL_RETURN_OBJ is False"
```

### Step 3: Update anti-overfit bundle test

In `test_anti_overfit_config.py` line ~11:

- `assert cfg.PHASE2_VAL_IN_FITNESS_PENALTY is True`

Leave other assertions unchanged unless they conflict.

**Verify**:

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/pytest tests/unit/test_anti_overfit_config.py -q
```

### Step 4: Fix any other default-dependent tests

Run related Phase2 tests; update only tests that assert the old **default**, not tests that monkeypatch True/False for behavior.

**Verify**:

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/pytest tests/unit/test_phase2_rule_pool.py tests/unit/test_phase2_stage.py tests/unit/test_conflict_table_validation.py tests/unit/test_phase2_val_sim_interval.py -q
```

### Step 5: Confirm overfit gap path is live

Read `phase2_rule_pool.py` overfit block — with new default, gap applies whenever `val_metrics` present and weight > 0. No code change required if getattr default is already False-only as fallback; optional: change getattr default to `True` for consistency with config — **only if** getattr fallback would disagree with config import in tests that omit the attribute (prefer leave getattr as False for safety on old checkpoints).

**Verify**: existing test that monkeypatches penalty True still passes; add a small unit test only if none assert gap > 0 when config default True and joint False (prefer extend `test_phase2_rule_pool.py` if coverage gap is clear).

## Test plan

- Bundle + phase2_rule_pool / stage / conflict / val_sim_interval as above.
- Colab (operator, not executor on WSL): after merge, one short Phase2 smoke should show non-zero overfit_gap contributions in logs and fewer extreme train/val ratio warnings vs 2026-07-13 run. Do not require Colab for plan DONE.

## Done criteria

- [ ] `PHASE2_VAL_IN_FITNESS_PENALTY` defaults True; joint stays False; USE_TOTAL_RETURN_OBJ stays False
- [ ] Comments document anti-leak rationale
- [ ] `test_anti_overfit_config.py` and related Phase2 unit tests pass with `PYTEST_LOW_MEMORY=1`
- [ ] Codelookup dependents updated
- [ ] No JOINT=True; no island/Phase1 changes

## STOP conditions

- Enabling val penalties causes mass infeasible population in **unit** tests with no clear floor tweak — stop and report; do not silently set JOINT=True.
- Live code already has VAL_IN_FITNESS_PENALTY=True — mark plan redundant after drift check.
- Fix requires changing pool admission gates / monthly restore — out of scope.

## Maintenance notes

- Plan 004 (Phase1 + multi-symbol islands) should land after this so search space and fitness both push multi-symbol generalization.
- If Colab still shows `objective_corr_f1_f3`≈1.0, follow-up: reduce shared `support_penalty` on f1 (`PHASE2_SUPPORT_PENALTY_WEIGHT_F1`) or apply overfit_gap only to f3 (already mostly on f3 in formula — verify f1 does not also get gap).
