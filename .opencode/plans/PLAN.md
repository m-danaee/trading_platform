# PLAN: Quant Pipeline Audit Fixes

## Goal
Fix the 13 CONFIRMED/SUSPECTED findings from the gpu_fuzzy_trader OOS audit
(see `outputs/run.log` 2026-07-07 run; train→val→test collapse:
long 58.21%→49.92%→22.22%, short 60.57%→50.47%→15.11%).
Each task is small, independently reviewable, and ordered by OOS impact.
Tasks map 1:1 (or 2:1) to audit findings, marked inline as `→ fixes #N` /
`→ implements Nx`. Do NOT touch `evaluator_v5.ipynb` (per AGENTS.md).

## Context
- Active objective: improve OOS generalization by removing the three
  overfit amplifiers (dead resampling, outlier-driven f3, risk-grid
  tuning on val_selection) and the four dead/weak gates that were
  supposed to catch them.
- base_branch: main
- branch_policy: isolated (each task on its own feature branch)
- execution_mode: checkpoint (Nexus will stop after each task)

## Execution order (high → low OOS impact)

### task-1: Per-epoch train-window rotation
- **Why first**: unblocks `task-10` (cache refresh), makes the
  `max_train_val_gap` signal meaningful, single highest-leverage fix.
- → fixes #1 (dead resampling), implements N2.
- Scope: in `phase2_island_scheduler.py:_run_cluster_islands` (and/or
  `phase2_rule_pool.py:__init__`), rotate the train-window start per
  epoch using a deterministic seed `hash(island_id, epoch_idx)` instead
  of sampling once at cluster init. Cap the per-sym request so the
  RNG branch at `phase2_rule_pool.py:471` actually fires
  (`total_rows = min(PHASE1_SAMPLING_TOTAL, 4 × safe_len_per_sym)`).
  Keep alignment: shared start bar across all syms in a cluster.
- Targets: `phases/phase2_island_scheduler.py`, `phases/phase2_rule_pool.py`,
  `phases/phase2_support.py` (helper for epoch seed), `config.py`
  (new `PHASE2_PER_EPOCH_WINDOW_ROTATION` flag, default True).
- Acceptance:
  - The "requested N bars/sym exceeds largest safe range" warning does
    NOT fire on the first cluster epoch when `n_per_sym <= safe_len`.
  - The train DataFrame for cluster 0 at gen-1 differs from gen-7
    (deterministic; verifiable via `_symbol_bar_index` min/max).
  - Engine warmup and re-park path still work; no VRAM regression
    (keep sequential cluster warmup, single cluster alive at a time).
- Tests: extend `tests/unit/test_phase2_island_scheduler.py` with a
  rotation test (assert df hash differs across epochs); add
  `tests/unit/test_phase2_window_rotation.py` for the helper.

### task-2: Return-concentration objective (4th NSGA axis)
- → fixes #2 (outlier-driven f3), implements N1.
- Scope: in `phase2_rule_pool.py:_evaluate_chromosome` and
  `phase2_rule_pool.py:_finalize_objectives`, add a 4th objective
  `f4 = max_single_trade_pnl / max(sum_positive_pnl, 1e-6)` computed
  from per-trade PnL already returned by the engine. Bump
  NSGA-III population partitions from 3 → 4 axes. Update
  `_evaluate_chromosome` return shape to `(4,)` and refresh every
  downstream consumer (Pareto rank, diversity penalty, archive).
- Targets: `phases/phase2_rule_pool.py`, `evolution/evox_runner.py`
  (pareto ranker, archive), `evolution/numba_ops.py` if shape
  changes break it.
- Acceptance:
  - A rule with 1 trade = +60% and 79 trades avg -0.5% receives
    f4 ≈ 0.85; a rule with uniform +1% across 80 trades gets f4 ≈ 0.012.
  - Pareto fronts on log include f4 column; `pareto_unique` still
    returns valid count.
  - `outputs/run.log` would no longer show `max_robust_return=107.52%`
    dominating the front.
- Tests: `tests/unit/test_phase2_rule_pool.py` add a f4
  concentration test; `tests/property/test_cpu_engine_properties.py`
  property test for f4 monotonicity w.r.t. concentration.

### task-3: RB Governor 2-fold walk-forward risk grid
- → fixes #3 (risk-grid overfit on val_selection) and #12
  (PHASE4_TAIL_HOLDOUT_FRACTION orphan), implements N3 + N4.
- Scope: in `rb_governor.py`, split `val_selection_df` into 2
  chronological folds. In `_optimize_risk` (rb_governor.py:703),
  score every TP/SL/capital combo on BOTH folds, pick the combo
  with the best `min(fold1_score, fold2_score)` (worst-fold
  selection). Wire `PHASE4_TAIL_HOLDOUT_FRACTION=0.25` as a
  third option: reserve the final 25% of val_selection as an
  untouched risk-grid holdout used only for the final tie-break.
- Targets: `rb_governor.py`, `config.py` (add
  `RB_RISK_GRID_WF_SPLITS=2`, `RB_RISK_GRID_USE_TAIL_HOLDOUT=True`).
- Acceptance:
  - The two best train-scoring combos (log shows
    "score=4620.73 train=58.21% valid=49.92%") would NOT both be
    chosen if they disagree on the two folds.
  - Selecting the same TP/SL/capital combo on the tail-holdout is
    reported as `risk_tail_holdout_return_pct` in the RB JSON
    output.
- Tests: `tests/unit/test_rb_governor_cv_folds.py` add a 2-fold
  WF test with synthetic folds; add `tests/unit/test_rb_governor_tail_holdout.py`.

### task-4: Monthly admission gate runs on val, not train
- → fixes #4 (dead-near-dead OOS guardrail).
- Scope: in `phase2_rule_pool.py` lines 2683 and 3163, change
  `build_monthly_windows(self._train_df)` →
  `build_monthly_windows(self._cached_slim_val)`. Update
  docstring + log message.
- Targets: `phases/phase2_rule_pool.py` (two call sites),
  `phases/phase2_support.py` if a helper wraps the call.
- Acceptance:
  - Log line "monthly-admission gate N → N rules" filters a
    non-trivial number of rules (currently 0-8 of 75 on train;
    target >10 of 75 on val for at least one direction).
  - The 75→75/75/67 collapse on long becomes 75→X/X/Y with
    monotonically decreasing X across the 3 finalizes.
- Tests: `tests/unit/test_phase2_monthly_admission.py` add a
  "gate uses val, not train" test that asserts the val and train
  DataFrames passed to `build_monthly_windows` differ.

### task-5: Delete dead f3 profit_factor branch
- → fixes #5 (dead code misleading future tuners).
- Scope: in `phase2_rule_pool.py:744-770`, wrap the
  `f3_objective` switch in `if not _cfg.PHASE2_USE_TOTAL_RETURN_OBJ:`
  (or delete the branch entirely with a comment explaining
  USE_TOTAL_RETURN_OBJ takes precedence). Add a unit test that
  asserts the path is unreachable.
- Targets: `phases/phase2_rule_pool.py`.
- Acceptance:
  - Lines 744-770 are only executed when
    `PHASE2_USE_TOTAL_RETURN_OBJ=False`; a test flips the flag and
    verifies the branch fires (covers the regression case).
  - No behavior change in current default config
    (`USE_TOTAL_RETURN_OBJ=True`).
- Tests: `tests/unit/test_phase2_rule_pool.py` add a parametrized
  test over `PHASE2_USE_TOTAL_RETURN_OBJ` × `PHASE2_F3_OBJECTIVE`.

### task-6: Hard overfit ratio gate at pool admission
- → fixes #7 (overfit-gap penalty too weak vs return signal).
- Scope: in `phase2_rule_pool.py:_passes_pool_admission_impl` and
  `phase2_support.py:_feasibility_gate_failures`, add a 10th
  gate: reject if `train_return / max(val_return, 0.1) > 3.0`.
  Also raise `PHASE2_OVERFIT_GAP_PENALTY_WEIGHT` 5.0 → 15.0
  in `config.py` (the soft penalty should be 3× stronger).
- Targets: `phases/phase2_rule_pool.py`, `phases/phase2_support.py`,
  `config.py`.
- Acceptance:
  - A rule with train=30%, val=5% (6× ratio) is now rejected at
    pool admission (was admitted previously).
  - `feasibility_collapse breakdown` logs include the new
    `overfit_ratio` key when the rule fails the ratio gate.
- Tests: `tests/unit/test_phase2_rule_pool.py` add a ratio-gate
  test; extend `tests/unit/test_phase2_support.py` with the
  new gate in `_feasibility_gate_failures`.

### task-7: Lower val PF floor during evolution only
- → fixes #9 (PF=1.15 too high given thin val trade counts).
- Scope: in `config.py`, split the single `PHASE2_PROFIT_FACTOR_FLOOR`
  into two: `PHASE2_PROFIT_FACTOR_FLOOR_EVOLUTION=1.05` and
  `PHASE2_PROFIT_FACTOR_FLOOR_ADMISSION=1.15`. Update the use
  sites in `phase2_rule_pool.py` and `evox_runner.py`.
- Targets: `config.py`, `phases/phase2_rule_pool.py`,
  `evolution/evox_runner.py`.
- Acceptance:
  - Log shows `train_pf_floor` rejects 0-10/120 mid-run (was
    40-70/120 with floor=1.15); `valid_rules` Pareto front size
    grows from 2-10 to 8-20 by gen 13.
  - Pool admission still rejects PF<1.15 (the hard gate).
- Tests: `tests/unit/test_phase2_rule_pool.py` add a
  `PHASE2_PROFIT_FACTOR_FLOOR_EVOLUTION` test.

### task-8: Use or remove `val_df` in Phase 1 sign consistency
- → fixes #11 (dead parameter).
- Scope: in `features/selector.py:_check_spearman_sign_consistency`
  (line 219), the `val_df` parameter is never used. Two options:
  (a) actually use it — append a "val fold" to `folds` and require
  the val sign to match the train majority sign, OR
  (b) remove the parameter and update the docstring.
  Pick (a) (the more useful behavior) unless it's expensive.
- Targets: `features/selector.py`, `tests/unit/test_feature_selector.py`.
- Acceptance:
  - A feature with consistent train signs but flipped val sign is
    now blacklisted (was admitted).
  - The function's signature changes are backwards-compatible
    (default `val_df=None` preserves old behavior).
- Tests: `tests/unit/test_feature_selector.py` add a
  "val sign mismatch blacklists feature" test.

### task-9: Migration log/rename to "sequential chain"
- → fixes #6 (interval gate is dead; logs oversell round-robin).
- Scope: in `phases/phase2_island_scheduler.py:493-540`, the
  migration log line at the top of the cluster section
  ("island mode migration=enabled sequential post-cluster
  chain") is already correct, but the
  `_should_migrate_this_round` helper at :370 is dead.
  Delete it (and its docstring) and grep for any remaining
  callers. Also update the `PHASE2_MIGRATION_EPOCH_INTERVAL`
  config docstring to "deprecated; no-op in current scheduler."
- Targets: `phases/phase2_island_scheduler.py`, `config.py`.
- Acceptance:
  - `_should_migrate_this_round` removed; no imports/uses remain
    (grep returns 0).
  - `PHASE2_MIGRATION_EPOCH_INTERVAL` config still exists for
    backwards compat but is marked deprecated.
  - All existing `test_phase2_migration_gate.py` and
    `test_phase2_island_scheduler.py` tests still pass.
- Tests: `tests/unit/test_phase2_island_scheduler.py` add a
  "no interval gate is consulted" test.

### task-10: Disable cache refresh now that windows rotate (depends on task-1)
- → fixes #8 (refresh wipes cache every epoch when windows were fixed).
- Scope: in `evolution/evox_runner.py:2606`, the
  `refresh_objectives_on_resume` block clears `metrics_cache` and
  `global_metrics_cache`. With per-epoch window rotation (task-1),
  this is correct, so KEEP it. If task-1 is rejected, the
  alternative is to gate the refresh on
  `PHASE2_PER_EPOCH_WINDOW_ROTATION` and disable it in fixed-window
  mode. This task is therefore *conditional on task-1's outcome*.
- Targets: `evolution/evox_runner.py`, `config.py`.
- Acceptance:
  - With `PHASE2_PER_EPOCH_WINDOW_ROTATION=True`, cache is refreshed
    per epoch (current behavior).
  - With `PHASE2_PER_EPOCH_WINDOW_ROTATION=False`, cache is NOT
    refreshed; log shows `cache_hit_rate` ≥ 0.4 in mid-gen.
- Tests: `tests/unit/test_phase2_rule_pool.py` add a
  "cache hit rate depends on rotation flag" test.

### task-11: PHASE2_VAL_SIM_INTERVAL=3 (depends on task-1 keeping window-fixed property for cached metrics)
- → fixes #10 (val every gen is wasteful when metrics are deterministic per chromosome).
- Scope: in `config.py:716`, raise `PHASE2_VAL_SIM_INTERVAL` from 1
  to 3. Verify the val metrics cache is keyed on chromosome
  (not gen) so cached values are reused. If the cache is keyed
  on `(chromosome, gen)`, fix it in `evox_runner.py`.
- Targets: `config.py`, `evolution/evox_runner.py`.
- Acceptance:
  - Val is backtested every 3rd gen instead of every gen.
  - Val metrics for chromosome X at gen-1 == gen-2 == gen-3
    (deterministic, no cache drift).
  - Last-gen val always runs (pool admission needs fresh metrics).
- Tests: `tests/unit/test_phase2_val_sim_interval.py` update
  existing tests for the new interval; add a "val metrics
  deterministic across cached gens" test.

### task-12: Gate Pareto-collapse warning to pareto_size ≥ 5
- → fixes #13 (noisy 2-point correlation warning).
- Scope: in `evolution/evox_runner.py:2988` and the warning site
  at :2228, gate the `objective_corr_f1_f2/f1_f3/f2_f3`
  warnings on `len(pareto_indices) >= 5`. Below that, the
  Pearson correlation is degenerate.
- Targets: `evolution/evox_runner.py`.
- Acceptance:
  - No `Pareto collapse risk` warnings fire when pareto < 5
    (current log shows them at gen 1 with pareto=1-2).
  - Warnings still fire when pareto ≥ 5 with |corr| > 0.9.
- Tests: `tests/unit/test_phase2_rule_pool.py` add a
  "warning gated on pareto size" test using caplog.

## Verification (per task)
- Run ONLY the related unit tests with
  `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/<file> -x`
  (per AGENTS.md).
- No full pipeline run (OOM risk on local/WSL; user runs on Colab).
- For tasks that change objective geometry (task-2, task-7, task-11),
  re-check the `max_robust_return` and `valid_rules` columns in
  the next Colab run log against the pre-fix baseline.

## Post-plan
- After all tasks merge, the user runs the full pipeline on Colab
  (per their normal workflow) and we compare:
  - `long/short` train→val→test return gaps
  - `valid_rules` Pareto front size at gen 13
  - `cache_hit_rate` (should rise with task-10/11)
  - `feasibility_collapse` `overfit_ratio` rejections
- Expected: train→val→test gap shrinks; test return closer to
  validation return; OOS return improves by single-digit % points.
