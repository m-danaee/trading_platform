# Plan 002: RB fail-closed on concentration and tail-holdout gates

> **Executor instructions**: Follow step by step. Run every verification command before proceeding. On STOP conditions, stop and report. Do **not** update `plans/README.md` (reviewer maintains index).
>
> **Drift check (run first)**: `git diff --stat 425f469..HEAD -- gpu_fuzzy_trader/rb_governor.py gpu_fuzzy_trader/config.py tests/unit/test_rb_fail_closed.py tests/unit/test_rb_min_symbols.py tests/unit/test_rb_governor_tail_holdout.py`
> Mismatch with "Current state" excerpts → STOP.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: MED
- **Depends on**: none (can run parallel with 001)
- **Category**: bug
- **Planned at**: commit `425f469`, 2026-07-16

## Why this matters

Symbol-concentration and tail-holdout gates set `deployment_accepted=False` but still **persist the full ruleset** to `{direction}.json`. Phase 5 always loads those strategies and OOS-evals them, so “rejected” teams still trade. Other RB hard failures already write **empty** strategies (`no_positive_good_candidates`, `insufficient_distinct_symbols`). Concentration/tail must match that fail-closed pattern so rejected deployments cannot pollute OOS or look deployable by accident.

Evidence from Colab 2026-07-13: long top_share≈0.63 and tail_ret=-0.40% failed gates yet strategies were saved and Phase 5 ran near-flat OOS.

## Current state

In [`gpu_fuzzy_trader/rb_governor.py`](gpu_fuzzy_trader/rb_governor.py) after risk/profit amplify (~1772–1828):

```python
sym_ok, sym_gate = _passes_symbol_concentration_gate(opt_test)
tail_ok, tail_gate = _passes_tail_holdout_gate(risk_history)
deployable = (
    val_ret >= (ret_gate - 1e-9)
    and val_pf >= (pf_gate - 1e-9)
    and sym_ok
    and tail_ok
)
# ... warnings ...
strategy = _strategy(
    direction,
    opt_rules,  # <-- FULL rules even when not deployable
    risk_optimized=bool(deployable),
    extra={
        "deployment_accepted": bool(deployable),
        ...
    },
)
# writes strategy_path with those rules
```

Contrast empty fail-closed for insufficient symbols (~1696–1765): clears `opt_rules`, writes empty `rules_set`, `deployment_accepted=False`, `fail_closed=True`, `continue`.

Gate helpers (~76–101) read:

- `RB_MAX_SYMBOL_SHARE_ABS_PNL` (0.50), `RB_MAX_SYMBOL_HHI` (0.55)
- `RB_TAIL_HOLDOUT_HARD_GATE` (True), `RB_TAIL_HOLDOUT_MIN_RETURN_PCT` (0.0)

[`gpu_fuzzy_trader/run_pipeline.py`](gpu_fuzzy_trader/run_pipeline.py): Phase 5 **always runs** on directions returned by RB — do **not** skip Phase 5; empty strategies neutralize trading.

Existing tests to mirror:

- [`tests/unit/test_rb_fail_closed.py`](tests/unit/test_rb_fail_closed.py)
- [`tests/unit/test_rb_min_symbols.py`](tests/unit/test_rb_min_symbols.py) (empty + `fail_closed_reason`)
- [`tests/unit/test_rb_governor_tail_holdout.py`](tests/unit/test_rb_governor_tail_holdout.py)

**AGENTS.md**: `.venv`; `PYTEST_LOW_MEMORY=1`; related tests only; no full pipeline; no `evaluator_v5.ipynb` edits.

## Commands you will need

| Purpose       | Command                                                                                                                                                                                       | Expected on success |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- |
| Drift         | `git diff --stat 425f469..HEAD -- gpu_fuzzy_trader/rb_governor.py tests/unit/test_rb_*.py`                                                                                                    | review before edit  |
| Related tests | `PYTEST_LOW_MEMORY=1 .venv/bin/pytest tests/unit/test_rb_fail_closed.py tests/unit/test_rb_min_symbols.py tests/unit/test_rb_governor_tail_holdout.py tests/unit/test_rb_compose_gates.py -q` | all pass            |
| New tests     | include new file(s) in same pytest invocation                                                                                                                                                 | all pass            |

## Suggested executor toolkit

- Invoke **codelookup** on `run_rb_governor_pipeline`, `_passes_symbol_concentration_gate`, `_passes_tail_holdout_gate`, `_strategy`, and writers of `{direction}.json`.
- Cascade-update any Phase4 twin if it shares the same soft-reject pattern (`phase4_wf_optimizer.py` has `deployment_accepted` — **out of scope** unless identical bug is confirmed; do not expand without STOP + plan amend).

## Scope

**In scope**:

- `gpu_fuzzy_trader/rb_governor.py` — fail-closed write path when `not sym_ok` or `not tail_ok` (and optionally when return/PF gates fail — see Step 2 committed choice)
- New/extended unit tests under `tests/unit/` mirroring `test_rb_min_symbols.py`
- Brief comment in `gpu_fuzzy_trader/config.py` next to `RB_TAIL_HOLDOUT_HARD_GATE` / concentration knobs clarifying fail-closed empties rules (no new feature flags required)

**Out of scope**:

- Skipping Phase 5 in `run_pipeline.py`
- Loosening `RB_MAX_SYMBOL_SHARE_ABS_PNL` / HHI / tail min return
- Phase 2 fitness / islands (plans 003–004)
- `evaluator_v5.ipynb`
- Dataset rename (001)

## Git workflow

- Branch: `advisor/002-rb-fail-closed-concentration-tail`
- Commit style: `fix: fail-closed empty RB strategies on concentration/tail gates`
- Do NOT push or open PR unless instructed

## Steps

### Step 1: Codelookup + locate write sites

Confirm all paths that write `{direction}.json` under `run_rb_governor_pipeline`. Note the exact empty-strategy block for `insufficient_distinct_symbols` as the copy template.

**Verify**: codelookup list includes `rb_governor.py` write sites; no surprise callers.

### Step 2: Implement fail-closed for concentration and tail (committed behavior)

**Committed choice**: When `not sym_ok` **or** `not tail_ok`, **do not** save `opt_rules`. Instead:

1. Log warning (existing messages OK).
2. Write empty strategy via `_strategy(direction, [], risk_optimized=False, extra={...})` with:
   - `deployment_accepted=False`
   - `reason` / `fail_closed_reason`: `"symbol_concentration"` and/or `"tail_holdout"` (if both fail, prefer a combined reason string `"symbol_concentration+tail_holdout"` or list both flags)
   - Include `symbol_concentration_gate` / `tail_holdout_gate` payloads for diagnostics
3. Write report JSON with `fail_closed=True` and gate details (mirror min-symbols report).
4. `continue` to next direction (same as other fail-closed paths).

**Committed choice for return/PF-only failure** (`val_ret` / `val_pf` below gate but sym+tail OK): keep current behavior (save rules with `deployment_accepted=False`). Those gates are soft deployability for operator review; concentration/tail are the hard structural rejects that matched the Colab failure mode. Do not change return/PF path unless a test already requires it.

Place the new block **after** gates are computed and **before** the successful full-rules write (~1772–1828). Avoid duplicating file I/O — factor a small local helper `_write_fail_closed_strategy(...)` **only if** it reduces duplication with the min-symbols block without a large refactor. Prefer minimal diff: inline copy of the existing pattern is acceptable.

**Verify**: unit tests in Step 3; manual read of saved JSON shape matches empty fail-closed.

### Step 3: Tests

Add tests (new file `tests/unit/test_rb_concentration_tail_fail_closed.py` or extend existing) that mock gates / pipeline far enough to assert:

- When concentration fails → saved strategy `rules_set == []`, `deployment_accepted is False`, report `fail_closed is True`
- When tail fails → same
- When both gates pass and return/PF pass → non-empty rules still saved with `deployment_accepted True` (smoke; mock as needed)
- Empty fail-closed strategy still loadable via `Output_Writer` / `_validate_rule_set` (reuse pattern from `test_rb_min_symbols.py`)

Follow existing mock style in `test_rb_min_symbols.py` (patch engines/gates, tempfile out_dir).

**Verify**:

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/pytest tests/unit/test_rb_fail_closed.py tests/unit/test_rb_min_symbols.py tests/unit/test_rb_governor_tail_holdout.py tests/unit/test_rb_concentration_tail_fail_closed.py -q
```

(If you extended an existing file instead of creating the new one, substitute that path.)

### Step 4: Config comment only

Update comments at `RB_TAIL_HOLDOUT_HARD_GATE` / `RB_MAX_SYMBOL_SHARE_ABS_PNL` in `config.py` to state: failure clears ruleset (fail-closed empty strategy), not merely `deployment_accepted=False` with rules retained.

**Verify**: comment accurate; no knob value changes unless required by tests (defaults stay).

## Test plan

- New assertions above.
- Regression: existing fail-closed and min-symbols tests still pass.
- Do not run full RB pipeline on WSL.

## Done criteria

- [ ] Concentration or tail failure writes empty `rules_set` and `deployment_accepted=False`
- [ ] Report includes `fail_closed=True` and actionable reason
- [ ] Return/PF-only soft path unchanged
- [ ] Related pytest passes with `PYTEST_LOW_MEMORY=1`
- [ ] Codelookup blast radius addressed; Phase5 still always runs
- [ ] No out-of-scope files modified

## STOP conditions

- Current state excerpts don't match (gates already fail-closed).
- Making concentration fail-closed breaks a documented intentional “save for analysis” path that product owners require — stop and report (do not invent a feature flag unless user amends plan).
- Fix appears to require skipping Phase 5 — stop; that is out of scope.
- Verification fails twice.

## Maintenance notes

- After this lands, Colab OOS for rejected directions should show empty/no-trade strategies rather than near-flat traded OOS from concentrated teams.
- Plans 003–004 reduce how often these gates fire by improving multi-symbol generalization upstream.
