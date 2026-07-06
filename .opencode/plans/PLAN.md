# Plan: Post-RAM-fix run analysis — 4 verified improvements

**Created:** 2026-07-06
**Status:** active
**base_branch:** `main`
**branch_policy:** isolated
**execution_mode:** checkpoint
**Source plan:** `~/.claude/plans/i-m-partially-run-my-temporal-treehouse.md`

## Goal

Address the four issues identified from the 2026-07-06 re-run analysis
(post-Task-4/5 merge) that the previous session left as recommendations:

1. **A** — RAM still climbing in cluster_0 first epoch (allocator fragmentation,
   not a code bug)
2. **B** — `f3` (profit_factor) objective is still train-only, allowing rules
   with great train / bad val to dominate Pareto front (Item 10, deferred
   twice; this is the third and final attempt at this)
3. **C** — Reverted time-exit return cap (`MAX_TIME_EXIT_RETURN_PCT=50.0`):
   identified as **NOT SAFE TO RE-IMPLEMENT** because it would diverge
   from evaluator_v5.ipynb
4. **D** — Feasibility collapse (valid_rules=2-4 out of 200): identified as
   the floors being too tight in two places; but relaxing them would *worsen*
   the overfit problem B is trying to fix. Plan is to add observability
   first, defer floor relaxation until we have evidence it's needed.

## Background

The previous Claude Code session at
`~/.claude/plans/i-m-partially-run-my-temporal-treehouse.md` did thorough
analysis and produced 4 recommendations (A, B, C, D). The user has now
authorized implementation of all four *if* they improve the project.
Each option has been re-verified against current source — see "Verification
results" below for findings.

## Verification results (re-verified against source 2026-07-06)

### A. RAM knobs — IMPLEMENT (all 3 sub-items)
- `PHASE2_EVAL_GLOBAL_CACHE_MAX_SIZE: 600 → 200` (line 414) — safe; cache
  hit rate is 0-4% in current logs
- `PHASE2_ISLAND_EPOCH_GENERATIONS: 25 → 13` (line 1104) — safe; total
  budget (44 gens/cluster × 3 clusters) unchanged, makes the existing
  `trim_evolution_state_memory` fire ~2x more often. Respects
  `PHASE2_ISLAND_MIN_EPOCH_GENERATIONS=5` floor
- `malloc_trim(0)` after `gc.collect()` in `evox_runner.py:2793` — safe
  (Linux glibc only; Colab is Linux). Returns freed arena memory to OS

### B. f3 train+val blend — IMPLEMENT
- `phase2_rule_pool.py:720-737` shows the pattern in the `win_rate` branch:
  `f3_val = min(win_rate, val_wr)` when `JOINT_TRAIN_VAL`
- The `profit_factor` branch just sets `f3_val = profit_factor` (train-only)
- Need to add the `val_metrics.get("val_profit_factor", 1.0)` guard the
  same way `win_rate` does
- Real change to NSGA-III selection pressure: rules with great train / bad
  val profit_factor will no longer dominate. This is the indicated next
  fix per the prior plan's own decision criteria (gap still present after
  Stages 1-3)

### C. Reverted time-exit cap — **DO NOT IMPLEMENT**
- Commit 46cb88a was clean: 4 new tests, jnp.clip / np.clip on
  `close_ret` and `-close_ret` only when neither TP nor SL hit, ±50% cap
- Revert 072c527 has no recorded reason in commit message
- **The likely reason for the revert is evaluator_v5.ipynb parity.**
  `evaluator_v5.ipynb:958, 971` does `return float(s_close), "Time_288"`
  and `return float(-s_close), "Time_288"` with NO cap. AGENTS.md says
  "Your evaluation must be based on evaluator_v5.ipynb because testing my
  rule sets is based on this file!" — re-implementing the cap would
  create a divergence that invalidates the user's standalone evaluation
- **Action: add a comment in `cpu_engine.py` and `gpu_engine.py` near the
  time-exit return paths explaining the intentional parity with
  evaluator_v5.ipynb, and link to AGENTS.md.** This is a small,
  safe documentation change — no behavior change

### D. Feasibility collapse — IMPLEMENT OBSERVABILITY ONLY
- Floors at `phase2_support.py:336-365` and `phase2_rule_pool.py:794-812`
  are not unreasonable for 10-symbol data with 1.2M train rows
- The dominant failure mode is `PHASE2_REQUIRE_LAST_FOLD_POSITIVE=True`
  (val_ret > 0) — most random rules start with negative val
- Relaxing floors would worsen overfit; the right fix is B (f3 blend) +
  better visibility into the failure mode
- **Action: add a per-generation log line in `evox_runner.py` when
  `valid_rules < 10`, showing the breakdown of which floor is failing
  most often. Safe, no behavior change.** Floor relaxation deferred
  until we have evidence from a re-run

## Tasks (ordered, independently reviewable)

### Task 25: RAM low-cost knobs (A1 + A2 + A3)
**Branch:** `fix/ram-knobs-final` (from `main`)
**Risk:** Low (config + 1-line malloc_trim + cache adjustment)
**Est. savings:** ~1-2 GB on Colab

**Changes:**
1. `config.py:414` — `PHASE2_EVAL_GLOBAL_CACHE_MAX_SIZE = 200` (was 600)
2. `config.py:1104` — `PHASE2_ISLAND_EPOCH_GENERATIONS = 13` (was 25)
3. `evox_runner.py:2793` — add `ctypes.CDLL("libc.so.6").malloc_trim(0)`
   after the existing `gc.collect()` call (Linux only; wrap in
   `try/except OSError` to be safe on non-glibc systems)

**Acceptance criteria:**
- All 3 changes are present at the indicated line numbers
- No other config values are changed
- The `malloc_trim(0)` is wrapped in a `try/except OSError` so it fails
  silently on non-Linux (defensive; Colab is Linux so it always runs)
- All touched test suites pass with `PYTEST_LOW_MEMORY=1`
- No regressions in existing behavior

---

### Task 26: f3 train+val blend (B / Item 10)
**Branch:** `fix/f3-train-val-blend` (from `main`)
**Risk:** Medium (changes NSGA-III selection pressure on the f3 objective)
**Est. impact:** Should reduce `corr_f1_f3=1.00` warnings and
`max_train_val_gap_ratio` further below 2.7x-4.5x

**Files to touch:**
- `gpu_fuzzy_trader/phases/phase2_rule_pool.py:720-737` — mirror the
  `win_rate` branch's blend pattern in the `profit_factor` branch
- `tests/unit/test_phase2_rule_pool.py` — add a test that confirms
  `f3_val = min(profit_factor_train, profit_factor_val)` when
  `JOINT_TRAIN_VAL=True` and `val_metrics` is provided

**Pattern to mirror (from current `win_rate` branch):**
```python
elif f3_objective == "win_rate":
    f3_val = win_rate
    if val_metrics is not None and _cfg.PHASE2_JOINT_TRAIN_VAL:
        val_wr = float(val_metrics.get("win_rate", 0.0))
        if int(val_metrics.get("executed_trades", 0)) < val_trade_floor:
            f3_val = min(win_rate, 0.0)
        else:
            f3_val = min(win_rate, val_wr)
```

**New pattern for `profit_factor` branch:**
```python
elif f3_objective == "profit_factor":
    f3_val = profit_factor
    if val_metrics is not None and _cfg.PHASE2_JOINT_TRAIN_VAL:
        val_pf = float(val_metrics.get("val_profit_factor", profit_factor))
        if int(val_metrics.get("executed_trades", 0)) < val_trade_floor:
            f3_val = min(profit_factor, 0.0)
        else:
            f3_val = min(profit_factor, val_pf)
```

**Acceptance criteria:**
- The `profit_factor` branch in the `f3_objective` selection at
  `phase2_rule_pool.py:720-737` blends train and val PF the same way
  the `win_rate` branch does
- The `val_profit_factor` key on val_metrics is the one already written
  at `phase2_rule_pool.py:655-661` (verify this is the right key)
- A new test in `test_phase2_rule_pool.py` covers:
  - `f3_val = min(train_pf, val_pf)` when JOINT_TRAIN_VAL=True
  - `f3_val = min(train_pf, 0.0)` when val_trades < val_trade_floor
  - `f3_val = train_pf` (unchanged) when JOINT_TRAIN_VAL=False
- All touched test suites pass with `PYTEST_LOW_MEMORY=1`
- The `cv_fold_min` and `PHASE2_USE_TOTAL_RETURN_OBJ` paths are
  unchanged

---

### Task 27: Document intentional evaluator_v5 parity (C)
**Branch:** `docs/time-exit-evaluator-parity` (from `main`)
**Risk:** None (comments only, no behavior change)

**Changes:**
- `gpu_fuzzy_trader/backtest/cpu_engine.py` — add a 3-4 line comment
  above lines 572 and 584 (`return float(s_close), "Time_288"`)
  explaining:
  - This branch intentionally does NOT cap `close_ret` to match
    `evaluator_v5.ipynb` (the user's ground truth per AGENTS.md)
  - A previous attempt to cap (commit 46cb88a, MAX_TIME_EXIT_RETURN_PCT=50.0)
    was reverted (commit 072c527) to preserve parity
- `gpu_fuzzy_trader/backtest/gpu_engine.py` — add a 3-4 line comment
  above the equivalent `close_ret` / `-close_ret` returns (lines 235, 245)
  with the same content

**Acceptance criteria:**
- 2 comment blocks added (one per engine)
- No code changes
- All touched test suites pass with `PYTEST_LOW_MEMORY=1`
- `git diff` is comment-only

---

### Task 28: Feasibility collapse observability (D)
**Branch:** `feat/feasibility-observability` (from `main`)
**Risk:** Low (adds a log line; no behavior change)

**Changes:**
- `gpu_fuzzy_trader/evolution/evox_runner.py` — find the per-generation
  log line (the one that currently logs `valid_rules=N`) and add a
  new log line **only when `valid_rules < 10`** that shows the
  breakdown: how many of the 200 individuals failed each of:
  - train trade floor
  - train return floor
  - train profit factor floor
  - val trade floor
  - val return floor
  - val profit factor floor
  - train/val gap
- The breakdown is computed in `phase2_rule_pool.py:_evaluate_chromosome`
  (or wherever the per-individual feasibility check happens) — pass
  the counts back via the metrics dict or a new `feasibility_breakdown`
  field, then aggregate in the per-gen log

**Acceptance criteria:**
- New log line appears when `valid_rules < 10` with the 7-component
  breakdown
- No log spam when `valid_rules >= 10` (current behavior)
- All touched test suites pass with `PYTEST_LOW_MEMORY=1`
- No regressions

---

## Verification (after all tasks merged)

Re-run the pipeline on Colab. Expected:
- RAM peaks lower than 9.6 GB at gen 9 (Task 25)
- `corr_f1_f3=1.00` warnings reduced or eliminated (Task 26)
- `max_train_val_gap_ratio` lower than 2.7x-4.5x (Task 26)
- New feasibility-breakdown log line shows up at gen 6+ in any
  generation where `valid_rules < 10` (Task 28)
- No divergence from evaluator_v5.ipynb behavior (Task 27)

## Out of scope (deferred)

- Floor relaxation for feasibility collapse — needs evidence from
  Task 28's observability first
- Re-investigation of `MAX_TIME_EXIT_RETURN_PCT` cap — decision is
  to keep evaluator_v5 parity (Task 27)
- Per-island train/val window resampling — was Stage 4 in prior plan;
  not in scope this pass
- Migration-cadence side effect from Task 5 (sequential chain replacing
  round-robin) — separate investigation
