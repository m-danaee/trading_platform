# PLAN — OOS Generalization Fixes (27 items)

**Created:** 2026-06-28
**Diagnosis source:** 2026-06-28 pipeline run log + `outputs/reports/*.json`
**Baseline OOS:** LONG test=−4.45% (PF 0.76, WR 38%), SHORT test=−0.00% (PF 1.00, WR 44%)
**Goal:** Recover OOS test returns to ≥0% (breakeven) and improve equity flatness via 27 targeted fixes across 6 tasks.

---

## Root Causes (ranked by OOS impact)

1. **Validation overfitting** — RB Governor weights val 2.7–3.8× over train; `RB_REQUIRE_TRAIN_SLIGHTLY_ABOVE_VALID=True` forces val-beating; gap penalty 0.55 is 12× too weak. Val regime ≠ test (Δ=−17.6%).
2. **Symbol-locked rules** — all 18 candidate rules contain `symbol is X`; test fires on 5/10 symbols.
3. **Fitness mis-specification** — f1 has 0.0 support penalty (lucky micro-rules dominate); f3 is noisy win-rate; Sortino tanh-saturated (flat gradient >3.0); val leaks into fitness despite `JOINT_TRAIN_VAL=False`.
4. **Premature convergence** — 1 restart max, 40% reinit, 5 elites; plateau delta 0.05% is noise; migration disabled (fake island model).
5. **State carry-over** — deployable archive + hall-of-fame persist across epochs → epoch 2/3 starts pre-converged.

---

## Task-14: RB Governor Rebalance (val overfit fix)

**Branch:** `fix/rb-governor-rebalance`
**Files:** `gpu_fuzzy_trader/rb_governor.py`, `gpu_fuzzy_trader/config.py`
**Priority:** 🔴 Critical
**Fixes:** C1, C2, C3, C4, M7

### Changes

**C1 — Rebalance valid/train weights** (`rb_governor.py::_score_metrics`, ~lines 188–210)
- Current: `120.0 * valid_ratio + 45.0 * train_ratio` and `4.5 * valid_ret + 1.2 * train_ret`
- New: `60.0 * valid_ratio + 60.0 * train_ratio` and `3.0 * valid_ret + 3.0 * train_ret`
- Rationale: validation should *gate* (reject bad rules), not *drive* selection. Equal weight prevents val-regime overfitting.

**C2 — Disable `RB_REQUIRE_TRAIN_SLIGHTLY_ABOVE_VALID`** (`config.py`, line 1536)
- Current: `RB_REQUIRE_TRAIN_SLIGHTLY_ABOVE_VALID: bool = True`
- New: `RB_REQUIRE_TRAIN_SLIGHTLY_ABOVE_VALID: bool = False`
- Rationale: forcing train > val explicitly prefers val-overfitting rules. A robust rule should generalize train→val with a *small* natural drop.

**C3 — Strengthen generalization-gap penalty** (`config.py`)
- `RB_TRAIN_VALID_RETURN_GAP_WEIGHT`: `0.55` → `4.0`
- `RB_TRAIN_VALID_RATIO_GAP_WEIGHT`: `12.0` → `30.0`
- Rationale: gap penalty must be comparable to return coefficients to actually bite.

**C4 — Add CV-fold consistency term** (`rb_governor.py::_score_metrics`)
- Add a `+ k_cv * min(fold_returns) − k_std * std(fold_returns)` term.
- Re-run the 3 purged CV folds (via `CvFoldValEvaluator`) for each candidate rule during RB composition.
- `k_cv = 8.0`, `k_std = 3.0` (tunable).
- The CvFoldValEvaluator already exists and caches fold engines. Call its evaluate method per candidate.
- Rationale: cross-fold minimum is the single best OOS generalization proxy; currently unused.

**M7 — Tighten `_combined_return_score` (rule addition)** (`rb_governor.py::_combined_return_score`, line 232)
- Current: `score = train_ret + valid_ret − health_penalty/35`
- New: add marginal-PF and marginal-DD terms:
  ```python
  score = train_ret + valid_ret
  score -= k_pf * max(0, prev_pf - new_pf)   # k_pf = 2.0
  score -= k_dd * max(0, new_dd - prev_dd)   # k_dd = 3.0
  score -= _evaluator_health_penalty(train_m, role="train") / 35.0
  score -= _evaluator_health_penalty(valid_m, role="valid") / 35.0
  ```
- Rationale: reject rules that raise return by eroding edge quality (PF dropped 5.97→2.22 over 6 rules).

### Acceptance Criteria
- [ ] `_score_metrics` has equal train/valid weights (60/60, 3/3).
- [ ] `RB_REQUIRE_TRAIN_SLIGHTLY_ABOVE_VALID = False` in config.
- [ ] `RB_TRAIN_VALID_RETURN_GAP_WEIGHT ≥ 4.0`, `RB_TRAIN_VALID_RATIO_GAP_WEIGHT ≥ 30.0`.
- [ ] `_score_metrics` includes a CV-fold-min term (non-zero coefficient).
- [ ] `_combined_return_score` penalizes PF degradation and DD increase.
- [ ] `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_rb_governor*.py -x -q` passes.
- [ ] No new imports beyond what's already available (`CvFoldValEvaluator`).

### Verification
```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/ -x -q -k "rb_governor or scoring"
```

---

## Task-15: Fitness & Objective Redesign

**Branch:** `fix/fitness-objective-redesign`
**Files:** `gpu_fuzzy_trader/phases/phase2_rule_pool.py`, `gpu_fuzzy_trader/config.py`, `gpu_fuzzy_trader/evolution/evox_runner.py`
**Priority:** 🔴 Critical
**Depends on:** task-14 merged (avoids config conflicts)
**Fixes:** C5, C6, C7, H1, H2, M3

### Changes

**C7 — Enable f1 support penalty** (`config.py`, line 437)
- `PHASE2_SUPPORT_PENALTY_WEIGHT_F1`: `0.0` → `0.4`
- Rationale: f1 (Sortino) currently immune to feasibility → 5-trade lucky rules dominate.

**H1 — Relax Sortino saturation** (`config.py`, lines 560, 565)
- `SORTINO_CAP`: `5.0` → `20.0`
- `SORTINO_SCALE`: `5.0` → `10.0`
- Rationale: tanh(raw/5)*5 is flat above Sortino ~3.0; NSGA-III cannot distinguish good from great.

**H2 — Replace f3 (win-rate) with cross-fold robustness** (`phase2_rule_pool.py::compute_phase2_objectives_from_metrics`, f3 definition)
- Current: `f3 = -win_rate + 0.6*support_penalty + diversity_penalty + cond_penalty + trade_penalty`
- New: `f3 = -min_cv_fold_return + 0.6*support_penalty + diversity_penalty + cond_penalty + trade_penalty`
  where `min_cv_fold_return` is computed via `CvFoldValEvaluator` (already injected into `Rule_Pool_Generator`).
- If CV evaluation per-individual is too expensive on GPU, fallback: `f3 = -profit_factor + ...` (set `PHASE2_F3_OBJECTIVE = "profit_factor"`).
- Add config flag `PHASE2_F3_OBJECTIVE = "cv_fold_min"` (default) | `"profit_factor"` | `"win_rate"` (legacy).
- Also set `PHASE2_USE_TOTAL_RETURN_OBJ = False` (win-rate path disabled).
- Rationale: win-rate is degenerate (ignores payoff, high-variance, anti-correlates with OOS). CV-fold-min directly targets generalization.

**C5 — Symbol-spread penalty + symbol gene mutation bias** (`phase2_rule_pool.py`)
- In `compute_phase2_objectives_from_metrics`, after computing `per_symbol_metrics`:
  ```python
  n_profitable_symbols = sum(
      1 for v in per_sym.values()
      if isinstance(v, dict) and float(v.get("net_pnl", 0)) > 0
  )
  min_symbols = int(getattr(_cfg, "PHASE2_MIN_PROFITABLE_SYMBOLS_PENALTY", 3))
  if n_profitable_symbols < min_symbols:
      symbol_lock_penalty = float(min_symbols - n_profitable_symbols) * 2.0
      support_penalty += symbol_lock_penalty
  ```
- In `_mutate` (phase2_rule_pool.py): if the chromosome has a symbol gene, with probability `PHASE2_SYMBOL_GENE_DONT_CARE_PROB = 0.4`, force it to `dont_care`. Identify the symbol gene by feature_info `mode` or name `"symbol"`.
- Rationale: all 18 candidate rules are symbol-locked → test fires on 5/10 symbols.

**C6 — Gate val penalties behind `JOINT_TRAIN_VAL`** (`phase2_rule_pool.py::compute_phase2_objectives_from_metrics`, ~lines 528–560)
- Wrap these blocks in `if _cfg.PHASE2_JOINT_TRAIN_VAL:`:
  - `val_floor_penalty` computation (val return < floor, val PF < floor)
  - `_symbol_robustness_penalty(val_metrics)` addition to `support_penalty`
  - The `support_penalty = max(support_penalty, SUPPORT_PENALTY_MAX)` when val trades < floor
- Add new flag `PHASE2_VAL_IN_FITNESS_PENALTY = False` (explicit, defaults off).
- When `JOINT_TRAIN_VAL=False` AND `VAL_IN_FITNESS_PENALTY=False`, val_metrics should only be stored on the metrics dict (for reporting/pool admission), NOT enter support_penalty.
- Rationale: config says "val for admission only" but code leaks val into fitness — triple-counting val.

**M3 — Add trade_penalty to f1** (`phase2_rule_pool.py::compute_phase2_objectives_from_metrics`)
- Current: `f1 = -sortino_for_obj + 0.0*support_penalty + diversity_penalty` (no trade_penalty)
- New: `f1 = -sortino_for_obj + 0.4*support_penalty + diversity_penalty + trade_penalty`
- Rationale: infeasible rules survive on f1 because trade_penalty only hits f2/f3.

### Acceptance Criteria
- [ ] `PHASE2_SUPPORT_PENALTY_WEIGHT_F1 = 0.4`, `SORTINO_CAP = 20.0`, `SORTINO_SCALE = 10.0`.
- [ ] `PHASE2_F3_OBJECTIVE` config flag exists; default `"cv_fold_min"`.
- [ ] f3 uses `min_cv_fold_return` (or `profit_factor` fallback) — NOT `win_rate`.
- [ ] `compute_phase2_objectives_from_metrics` includes symbol-spread penalty.
- [ ] `_mutate` forces symbol gene to `dont_care` with prob ≥ 0.4.
- [ ] Val-derived penalties gated behind `PHASE2_JOINT_TRAIN_VAL` or `PHASE2_VAL_IN_FITNESS_PENALTY`.
- [ ] f1 includes `trade_penalty`.
- [ ] `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2*.py -x -q` passes.
- [ ] No behavioral change to `evaluator_v5.ipynb` (do not modify it).

### Verification
```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/ -x -q -k "phase2 or objective or rule_pool"
```

---

## Task-16: Evolution Convergence Tuning

**Branch:** `fix/evolution-convergence`
**Files:** `gpu_fuzzy_trader/evolution/evox_runner.py`, `gpu_fuzzy_trader/phases/phase2_rule_pool.py`, `gpu_fuzzy_trader/config.py`
**Priority:** 🟠 High
**Depends on:** task-15 merged (shared function `compute_phase2_objectives_from_metrics`)
**Fixes:** H3, H5, M4, M5

### Changes

**H3 — Plateau restart: more restarts, more reinit, fewer elites, adaptive mutation** (`evox_runner.py`, `config.py`)
- `config.py`:
  - Add `PHASE2_PLATEAU_MAX_RESTARTS = 3` (currently hardcoded `1` in evox_runner).
  - `PHASE2_PLATEAU_EARLY_STOP_MIN_DELTA_PCT`: `0.05` → `0.5` (0.05% is noise on 5–20% returns).
  - `PHASE2_PLATEAU_DIVERSITY_RESTART_FRACTION`: `0.40` → `0.65`.
  - Add `PHASE2_PLATEAU_POST_RESTART_MUTATION_BOOST = 0.45` (temp mutation for 3 gens post-restart).
  - Add `PHASE2_PLATEAU_POST_RESTART_BOOST_GENS = 3`.
- `evox_runner.py::_plateau_diversity_restart`:
  - `n_elite = min(2, max(1, len(pareto_indices)))` (was `min(5, ...)`).
  - Use `PHASE2_PLATEAU_DIVERSITY_RESTART_FRACTION` for reinit fraction (already does, but verify after config bump).
- `evox_runner.py::_run_nsga2_fallback` / main loop:
  - Read `PHASE2_PLATEAU_MAX_RESTARTS` instead of hardcoded `1`.
  - After restart, set `mutation_rate = PHASE2_PLATEAU_POST_RESTART_MUTATION_BOOST` for `PHASE2_PLATEAU_POST_RESTART_BOOST_GENS` generations, then anneal back to base.
  - Track `post_restart_gens_remaining` counter on state.

**H5 — Epoch state carry-over: clear cache, cap hall-of-fame, diverse seeding** (`phase2_rule_pool.py::run_epoch`, `evox_runner.py`)
- `phase2_rule_pool.py::run_epoch`:
  - Before calling `run_phase2_evolution_epoch` on a new epoch (when `first_epoch=False` and seeds are being applied), clear `global_metrics_cache` entries whose keys match the seeded deployable chromosomes. This forces re-evaluation.
  - Cap `hall_of_fame` carry-over: add `PHASE2_HOF_EPOCH_CARRYOVER = 10` config; at epoch start, trim `hall_of_fame` to the top-10 by deployability rank.
- `evox_runner.py::_inject_diversity_recovery` / seed selection:
  - When seeding from deployable archive, select a **diverse subset** (k-means on Hamming distance or max-min diversity sampling), not top-K by `rank_score`.
  - Implement a `_select_diverse_subset(chromosomes, k)` helper using max-min Hamming distance greedy.

**M4 — Scale diversity Hamming threshold to chromosome length** (`config.py`, `phase2_rule_pool.py::_diversity_penalty_blended`)
- Current: `PHASE2_DIVERSITY_HAMMING_THRESHOLD = 7` (fixed).
- New: `PHASE2_DIVERSITY_HAMMING_THRESHOLD = 0` means "auto = max(3, K_active // 5)".
  - Compute `K_active = _count_active_conditions(chromosome, dont_cares)`.
  - `threshold = max(3, K_active // 5)`.
- Or add `PHASE2_DIVERSITY_HAMMING_THRESHOLD_AUTO = True` flag.

**M5 — Rank-based normalization for NSGA-III association** (`evox_runner.py::_normalize_for_association`)
- Current: min-max normalization (fragile to outliers from `trade_penalty=50`).
- New: replace objective values with their **percentile rank** (0..1) before unit-normalizing.
  ```python
  from scipy.stats import rankdata  # or implement manually
  rank_fit = np.column_stack([
      rankdata(merge_fit[:, j], method="average") / len(merge_fit)
      for j in range(merge_fit.shape[1])
  ])
  # then unit-normalize rank_fit as before
  ```
- Handle ties by averaging ranks (method="average").

### Acceptance Criteria
- [ ] `PHASE2_PLATEAU_MAX_RESTARTS = 3` in config; evox_runner reads it (not hardcoded `1`).
- [ ] `PHASE2_PLATEAU_EARLY_STOP_MIN_DELTA_PCT = 0.5`.
- [ ] `PHASE2_PLATEAU_DIVERSITY_RESTART_FRACTION = 0.65`.
- [ ] Post-restart mutation boost (0.45 for 3 gens) implemented and tested.
- [ ] `n_elite = min(2, ...)` in `_plateau_diversity_restart`.
- [ ] `global_metrics_cache` cleared for seeded keys at new epoch.
- [ ] `hall_of_fame` trimmed to ≤10 carry-over entries per epoch.
- [ ] Diverse-subset seeding (max-min Hamming) implemented.
- [ ] Hamming threshold auto-scales with active gene count.
- [ ] `_normalize_for_association` uses rank-based normalization.
- [ ] `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_plateau*.py tests/unit/test_evox_runner.py -x -q` passes.

### Verification
```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/ -x -q -k "plateau or evox or restart or diversity"
```

---

## Task-17: Island Migration & Rule Structure

**Branch:** `fix/island-migration-rule-structure`
**Files:** `gpu_fuzzy_trader/config.py`, `gpu_fuzzy_trader/phases/phase2_island_scheduler.py`
**Priority:** 🟠 High
**Depends on:** task-15 merged (config ordering)
**Fixes:** H4, H6

### Changes

**H4 — Enable migration (or document why island model is dropped)** (`config.py`, `phase2_island_scheduler.py`)
- Decision required from user (implementer should check `.opencode/tasks/task-17.md` for the chosen option):
  - **Option A (preferred):** `PHASE2_MIGRATION_ENABLED = True`. Verify `PHASE2_MIGRATION_EPOCH_INTERVAL`, `PHASE2_MIGRATION_TOP_K`, `PHASE2_MIGRATION_MIN_VAL_RETURN_PCT`, `PHASE2_MIGRATION_MIN_VAL_TRADES` are set to reasonable defaults (document them in config comments).
  - **Option B:** If migration's GPU re-evaluation cost is prohibitive, drop the island model: set `PHASE2_N_CLUSTERS = 1` and run a single NSGA-III on the full symbol set for `PHASE2_GENERATIONS` (132) gens. Remove or short-circuit `phase2_island_scheduler._run_cluster_islands`.
- **Default: Option A** unless the implementer finds GPU re-eval cost is >30% of epoch time.
- Add a startup log line: `Phase 2 island mode: migration={'enabled' if PHASE2_MIGRATION_ENABLED else 'disabled (independent islands)'}`.

**H6 — Raise MIN/MAX_CONDITIONS** (`config.py`, lines 405–406)
- `MIN_CONDITIONS`: `3` → `4`
- `MAX_CONDITIONS`: `4` → `5`
- Verify `assert MIN_CONDITIONS <= MAX_CONDITIONS` still passes (line 2078).
- Rationale: with 3–4 conditions, symbol gene is 25–33% of the rule. 4–5 dilutes it and forces richer market-structure combinations.

### Acceptance Criteria
- [ ] `PHASE2_MIGRATION_ENABLED = True` OR `PHASE2_N_CLUSTERS = 1` (documented choice).
- [ ] If migration enabled: migration config params documented and logged at startup.
- [ ] `MIN_CONDITIONS = 4`, `MAX_CONDITIONS = 5`.
- [ ] Config assertion passes.
- [ ] `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_island*.py -x -q` passes.
- [ ] No regressions in `test_phase2_island_hyperparams.py` or `test_island_scheduler_migration.py`.

### Verification
```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/ -x -q -k "island or migration or cluster"
```

---

## Task-18: Admission Gates & Robustness

**Branch:** `fix/admission-gates-robustness`
**Files:** `gpu_fuzzy_trader/phases/phase2_rule_pool.py`, `gpu_fuzzy_trader/backtest/cpu_engine.py`, `gpu_fuzzy_trader/config.py`, `gpu_fuzzy_trader/data/splitter.py` (investigation only)
**Priority:** 🟡 Medium
**Depends on:** task-15 merged
**Fixes:** M1, M2, M6, M8

### Changes

**M1 — Fix per-symbol `win_rate = 0.0` bug** (`backtest/cpu_engine.py`)
- Grep for `per_symbol_metrics` in `cpu_engine.py::simulate_rule_set`.
- In the per-symbol branch, compute `win_rate = (wins / trades) * 100.0` like the aggregate.
- The bug: `rb_governor_long_report.json → train_metrics.per_symbol_metrics` shows `"win_rate": 0.0` for every symbol, while `train_long_per_symbol_performance.csv` shows real values (37–55%).
- Verify the rule-set path (multi-rule) populates per-symbol `win_rate` correctly.

**M2 — Investigate val/test regime mismatch** (`data/splitter.py`, `config.py`)
- This is an **investigation + recommendation**, NOT a code fix in this task.
- From `generalization_diagnostics_long.json`: train & test have `amihud_illiquidity_20` "Very Low/Bullish" dominant; val has "Very High". Same for `roc_10`, `rsi_centered_14`.
- Inspect `data/splitter.py` purged walk-forward split logic. Document the train/val/test time boundaries.
- Write a short recommendation to `.opencode/tasks/task-18.md`:
  - Whether a 3-way walk-forward (trainA → val → trainB → test) is feasible.
  - Whether nested CV with multiple val folds would be better.
  - Whether the current split is acceptable if task-14/C4 (CV-fold consistency) compensates.
- Do NOT change the split logic in this task (high risk, needs pipeline re-run to validate).

**M6 — Tune global metrics cache** (`config.py`, `evox_runner.py::_evaluate_population_indices`)
- Current: `PHASE2_EVAL_GLOBAL_CACHE = True`, cache trimmed to `max(200, pop_size)` = 200. Log shows `cache_hit_rate=0.00–0.04`.
- Either:
  - **Option A:** Increase cache to `max(375, pop_size + deployable_archive_max + hof_max)` so seeded elites actually hit.
  - **Option B:** Disable `PHASE2_EVAL_GLOBAL_CACHE` for the live population (keep only for archive seeding).
- **Default: Option A** — set cache size to `pop_size + PHASE2_DEPLOYABLE_ARCHIVE_MAX_SIZE + 300` (200 + 75 + 300 = 575).
- Add a log line at gen 0: `cache_size={len(global_metrics_cache)} capacity={max_cache}`.

**M8 — Raise monthly-admission min_ratio** (`phase2_rule_pool.py::_apply_monthly_admission_gate`, `config.py`)
- Current: log shows `min_ratio=0.500` (rules profitable in only 50% of months admitted).
- Add config flag `PHASE2_MONTHLY_ADMISSION_MIN_RATIO = 0.667` (4 of 6 months).
- Use it in `_apply_monthly_admission_gate` instead of the hardcoded `0.500`.
- Rationale: your stated goal is "flat or improving equity" — 50% losing months is too lenient.

### Acceptance Criteria
- [ ] Per-symbol `win_rate` is non-zero in `rb_governor_*_report.json` after a test run.
- [ ] `data/splitter.py` regime investigation written to `.opencode/tasks/task-18.md` (no code change).
- [ ] `PHASE2_EVAL_GLOBAL_CACHE` size increased to ≥575 OR disabled for live pop (documented choice).
- [ ] `PHASE2_MONTHLY_ADMISSION_MIN_RATIO = 0.667` config flag exists and is used.
- [ ] `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_monthly_admission.py tests/unit/test_phase2_rule_pool.py -x -q` passes.

### Verification
```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/ -x -q -k "monthly or rule_pool or cpu_engine"
```

---

## Task-19: Cleanup & Observability

**Branch:** `fix/cleanup-observability`
**Files:** `gpu_fuzzy_trader/evolution/evox_runner.py`, `gpu_fuzzy_trader/phases/phase2_rule_pool.py`, misc
**Priority:** 🟢 Low
**Depends on:** task-16 merged (shared files)
**Fixes:** L1, L2, L3, L4, L5, L6

### Changes

**L1 — Remove dead `_agent_debug_log`** (`evox_runner.py`, lines ~58–66)
- Delete the function and any call sites.
- Per `AGENTS.md`: "remove additional/wasted parts from old implementation."

**L2 — Warn if EvoX unavailable** (`evox_runner.py`, lines ~75–85)
- Current: `_EVOX_IMPORT_ERROR` stored but only `logger.debug`.
- New: at first use (or module import), `logger.warning("EvoX unavailable (%s); falling back to NSGA-II. Pipeline config says NSGA3.", _EVOX_IMPORT_ERROR)`.
- Log once per process (use a module-level `_warned` flag).

**L3 — Fix plateau restart log** (`evox_runner.py`)
- Log `restart {n}/{max}` at the moment of each restart (not just `1/1` always).
- Include the new mutation rate and elite count in the log line.
- Add `restart_count` and `max_restarts` to the generation log line.

**L4 — Add viability-collapse trigger** (`evox_runner.py`)
- In the main generation loop, after computing `pop_viable`:
  ```python
  if pop_viable < 0.5 * pop_size:
      viability_collapse_streak += 1
  else:
      viability_collapse_streak = 0
  if viability_collapse_streak >= 3 and restart_count < max_restarts:
      # force diversity restart independent of plateau_streak
      _plateau_diversity_restart(...)
  ```
- Add `PHASE2_VIABILITY_COLLAPSE_THRESHOLD = 0.5` and `PHASE2_VIABILITY_COLLAPSE_STREAK = 3` config flags.

**L5 — Clean up `_sample_df` random_state param** (`phase2_rule_pool.py::_sample_df`)
- The `random_state` param is accepted but `del random_state` (intentionally ignored — chronology-preserving).
- Either: remove the param from the signature (and update callers), OR add a clear docstring: "Accepted for API compatibility but intentionally ignored — sampling is chronologically deterministic."
- **Default:** keep param, improve docstring (callers in `phase2_island_scheduler.py` pass it).

**L6 — Proper Das-Dennis fallback for reference vectors** (`evox_runner.py::_get_reference_vectors`)
- Current fallback (when EvoX unavailable): 3 axis vectors + `[t, (1-t)/2, (1-t)/2]` — biases the simplex center, never hits f2/f3 axes.
- Implement proper Das-Dennis in NumPy (~15 lines):
  ```python
  def _das_dennis(n_partitions, n_objs):
      # generates all points on the (n_objs-1)-simplex with n_partitions divisions
      from itertools import product
      points = []
      for combo in product(range(n_partitions + 1), repeat=n_objs):
          if sum(combo) == n_partitions:
              points.append([c / n_partitions for c in combo])
      return np.array(points)
  ```
- Choose `n_partitions` such that `len(refs) >= pop_size` (e.g. p=12 for 3 objs → 91 points; p=14 → 120; increase until ≥ pop_size, then truncate).

### Acceptance Criteria
- [ ] `_agent_debug_log` deleted; no call sites remain.
- [ ] EvoX unavailability logged at `WARNING` level once per process.
- [ ] Plateau restart logs `restart {n}/{max}` with mutation rate and elite count.
- [ ] Viability-collapse trigger implemented (3-gen streak → forced restart).
- [ ] `_sample_df` docstring clarifies `random_state` is intentionally ignored.
- [ ] `_get_reference_vectors` fallback uses Das-Dennis (uniform simplex coverage).
- [ ] `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_evox_runner.py -x -q` passes.
- [ ] No dead code remaining (per `AGENTS.md` cleanup rule).

### Verification
```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/ -x -q -k "evox or reference_vector or cleanup"
```

---

## Cross-Task Dependencies

```
task-14 (RB Governor)  ──► task-15 (Fitness) ──► task-16 (Evolution)
                                                  │
task-17 (Migration)  ◄─────────────────────────────┤  (parallel after 15)
                                                  │
task-18 (Admission) ◄─────────────────────────────┤  (parallel after 15)
                                                  ▼
task-19 (Cleanup)   ◄─────────────────────────────┘  (after 16)
```

- task-14 → task-15: both touch scoring; 14 is config+governor, 15 is fitness. Avoid config conflicts.
- task-15 → task-16: both touch `compute_phase2_objectives_from_metrics` / evox_runner. 15 must land first.
- task-17/18 can start after 15 (different files mostly; 18 touches cpu_engine + splitter).
- task-19 after 16 (shared evox_runner.py).

## Final OOS Validation (after all 6 tasks merged)

Run on Colab GPU:
```bash
python -m gpu_fuzzy_trader.run_pipeline --output /content/trading_platform_outputs
```
Compare `outputs/reports/test_long_report.json` and `test_short_report.json`:
- **Success:** test return ≥ 0% (breakeven) for both directions.
- **Stretch:** test return ≥ +3%, PF ≥ 1.2, win-rate ≥ 45%.
- **Equity flatness:** max drawdown ≤ 5%, no single-month drawdown > 3%.

Do NOT run pipeline locally (OOM per `AGENTS.md`).
