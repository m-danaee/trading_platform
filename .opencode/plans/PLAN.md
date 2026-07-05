# Plan: Fix Phase 2 overfit-gap blind spots + confirmed bugs (Stages 1-3)

**Created:** 2026-07-05
**Status:** active
**base_branch:** `main`
**branch_policy:** isolated
**execution_mode:** checkpoint

## Goal

Close the loop so an overfit rule (high train return, weak/negative validation return) can no longer be silently marked "deployable," dominate the Pareto front, and get pinned in place by elite preservation for many generations — while fixing the independent bugs found along the way.

**Background:** Investigation (3 parallel code audits) traced the runaway `max_return` behavior (up to 203% by gen 9, 7.77x gap vs `max_robust_return`) to the fitness function, not to the recent sampling rewrite. The sampling change is mechanically correct; it just stopped incidentally masking a pre-existing gap: validation/robustness performance has almost no real weight in NSGA-III selection.

**Scope:** Stages 1-3 only (mechanical fixes, seed/migration bugs, fitness-function gap fixes). Stage 4 (resampling train/val per island-epoch) is deferred to a follow-up plan.

## Source plan

`/home/danaee/.claude/plans/you-are-a-senior-pure-cupcake.md` — verified against current source on 2026-07-05, all line numbers and claims confirmed accurate.

## Tasks

### Task 1: Stage 1 — Mechanical fixes (zero behavior risk)
**Branch:** `fix/phase2-stage1-mechanical`
**Files:** `gpu_fuzzy_trader/config.py`, `gpu_fuzzy_trader/data/splitter.py`, `gpu_fuzzy_trader/phases/phase2_rule_pool.py`, `gpu_fuzzy_trader/evolution/evox_runner.py`, `gpu_fuzzy_trader/run_pipeline.py`, `gpu_fuzzy_trader/phases/phase2_island_scheduler.py`
**Risk:** Low (label/comment/structural changes, no behavior change)

Changes:
1. **SPLIT_MODE stale label** — Rename `"holdout_70_30"` → `"holdout"` (atomic across all comparison sites). Compute percentages from `HOLDOUT_TRAIN_FRACTION` for logging instead of baking them in.
2. **Stale docstring** in `phases/phase2_rule_pool.py:14-17` — Update `f3 = -win_rate` to reflect `PHASE2_F3_OBJECTIVE` default (`"profit_factor"`).
3. **`corr_f1_f3` Pareto-collapse warning** — Bump from DEBUG to INFO (or WARNING) so it's visible without debug logging.
4. **Duplicated generation-budget arithmetic** — Extract `compute_cluster_generation_budgets(total_gens, n_clusters) -> dict[int, int]` helper in `phase2_island_scheduler.py`; use from both `_run_cluster_islands` and `_log_pipeline_config`. Rename `per_cluster=` → `per_cluster_gens=` in the log line. Repoint `test_gens_per_cluster_split` / `test_epoch_rounds_cover_budget` to the new helper.
5. **`PHASE2_MIGRATION_ENABLED` stale comment** — Rewrite to describe re-enable after guarded re-evaluation, not the old failure mode.

**Acceptance criteria:**
- All SPLIT_MODE comparison sites use `"holdout"` (not `"holdout_70_30"`)
- All log lines computing percentages derive from `HOLDOUT_TRAIN_FRACTION`
- Phase2 rule pool docstring reflects `PHASE2_F3_OBJECTIVE` semantics
- `corr_f1_f3` warning logs at INFO or higher
- `compute_cluster_generation_budgets` exists in `phase2_island_scheduler.py` and is used by both call sites
- Log line uses `per_cluster_gens=` (not `per_cluster=`)
- PHASE2_MIGRATION_ENABLED comment narrates current state, not historical failure
- Tests `test_gens_per_cluster_split` / `test_epoch_rounds_cover_budget` point at new helper
- All touched test suites pass with `PYTEST_LOW_MEMORY=1`

---

### Task 2: Stage 2 — Seed collision + migration cadence bugs
**Branch:** `fix/phase2-stage2-seed-migration`
**Files:** `gpu_fuzzy_trader/phases/phase2_island_scheduler.py`, `tests/unit/test_phase2_island_scheduler.py`, `tests/unit/test_migration_safety.py`
**Risk:** Medium (changes RNG-derived seed values, must be tested)

Changes:
1. **Long/short seed collision** — At the two call sites where `_derive_island_seed` is invoked in `phase2_island_scheduler.py`, include `direction` in the `island_id` string:
   - `_run_cluster_islands` (~line 380): `_derive_island_seed(seed, cid)` → `_derive_island_seed(seed, f"{direction}_{cid}")`
   - orphan-boost path (~line 296): `_derive_island_seed(seed, f"orphan_{sym}")` → `_derive_island_seed(seed, f"{direction}_orphan_{sym}")`
   - New test: assert `_derive_island_seed(seed, "long_0") != _derive_island_seed(seed, "short_0")`.

2. **Migration cadence bug** — `epoch_counter` currently increments once per cluster inside the `for cid` loop, but the modulo gate is checked once per outer `while` round (currently masked by `interval=1`). Extract `_should_migrate_this_round(round_index: int, interval: int) -> bool` and increment a separate round counter outside the `for cid` loop. Existing tests reimplement the formula in isolation; they don't catch the per-cluster increment site.
   - New test: drive `_run_cluster_islands` (or just `_should_migrate_this_round` directly) with `n_clusters=3`, `interval=2` across several rounds, and assert migration fires on round cadence (every 2 rounds), not every 2 cluster-epoch calls.

**Acceptance criteria:**
- `_derive_island_seed` is called with `direction` in the `island_id` at both call sites
- New test in `test_phase2_island_scheduler.py` asserts seed uniqueness across directions
- `_should_migrate_this_round` helper exists in `phase2_island_scheduler.py`
- Round counter increments once per outer `while` round (not per cluster)
- New test asserts round-cadence firing
- All touched test suites pass with `PYTEST_LOW_MEMORY=1`

---

### Task 3: Stage 3 — Fitness-function gap fixes (the core fix)
**Branch:** `fix/phase2-stage3-fitness-gap`
**Files:** `gpu_fuzzy_trader/phases/phase2_rule_pool.py`, `gpu_fuzzy_trader/phases/phase2_support.py`, `gpu_fuzzy_trader/config.py`, `tests/unit/test_phase2_rule_pool.py`, `tests/unit/test_phase2_support.py`
**Risk:** Medium-High (changes fitness-function behavior; closes the actual blind spot)

Changes:
1. **Fix `overfit_gap_penalty`'s blind spot** — Switch from ratio-based (gated on `val_ret > 0`) to subtraction-based (`train_ret - val_ret`), matching the final pool-admission gate. Unified with the existing subtraction-based check.
   - Rename `PHASE2_OVERFIT_GAP_RATIO_THRESHOLD` → `PHASE2_OVERFIT_GAP_PCT_THRESHOLD` (units change from ratio to pct-points).
   - Default new threshold to ~8-10pp (below the hard gate `PHASE2_MAX_TRAIN_VAL_GAP_PCT=16.0`) — tunable.
   - New test: assert penalty for train=99%/val=-10% > penalty for train=99%/val=1% (monotonicity / no-blind-spot).

2. **Add same gap check to `_raw_feasibility_violation_score`** — This is the highest-leverage single change. The choke point feeds `passes_evolution_deployability_preview` (→ `deployable_archive` membership → `_preserve_deployable_elites` force-pinning) AND the real objectives via `support_penalty` (weighted 0.4/0.6/0.6 into f1/f2/f3). It currently never checks train-vs-val gap, unlike the final pool-admission gate.
   - Mirror the gate's check (phase2_support.py ~179-181) as an additive violation term.
   - New test: train_ret=90%/val_ret=10% (gap=80pp, over threshold) but all individual floors otherwise passing → assert `_raw_feasibility_violation_score > 0.0` and `passes_evolution_deployability_preview is False`.

**Explicitly deferred:** Item 10 (f1/f3 asymmetry — making `f3` worst-of-train/val like `f1`) is deferred per the plan. Land #8/#9 first, re-run, and only pursue if the gap is still insufficiently controlled.

**Acceptance criteria:**
- `overfit_gap_penalty` uses subtraction (`train_ret - val_ret`), not ratio
- `overfit_gap_penalty` is well-defined for `val_ret <= 0` (no `if val_ret > 0` gate)
- `PHASE2_OVERFIT_GAP_PCT_THRESHOLD` (new name) exists in config; old `PHASE2_OVERFIT_GAP_RATIO_THRESHOLD` removed/replaced
- New monotonicity test for `overfit_gap_penalty` passes
- `_raw_feasibility_violation_score` includes train-vs-val gap check using `PHASE2_MAX_TRAIN_VAL_GAP_PCT`
- New test: train_ret=90%/val_ret=10% with otherwise-passing floors → `_raw_feasibility_violation_score > 0` AND `passes_evolution_deployability_preview is False`
- All touched test suites pass with `PYTEST_LOW_MEMORY=1`

## Verification (after all tasks merged)

- Run touched suites: `pytest tests/unit/test_phase2_support.py tests/unit/test_phase2_rule_pool.py tests/unit/test_phase2_island_scheduler.py tests/unit/test_island_scheduler_migration.py tests/unit/test_migration_safety.py tests/unit/test_elite_preservation.py tests/unit/test_evox_runner.py -v` with `PYTEST_LOW_MEMORY=1`
- Add new regression tests *before* the corresponding fix where practical; confirm they fail on old code, then pass after — this is the only way to know items 7-9 actually would have caught the original bugs.
- No `purged_config_fingerprint()` bump needed — fingerprint (`data/splitter.py:182`) is scoped to the persisted train/val split cache only; none of Stages 1-3 touch the split itself.
- End-to-end signal: re-run Phase 2 on the same seed/data and compare `max_train_val_gap_ratio` trajectory (should stay bounded, not climb to 7-8x) and whether the final pool shrinks materially after admission gating.
