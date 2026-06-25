# Plan — Fix Phase 2 island-mode elite erosion & migration degradation

**Created:** 2026-06-25
**base_branch:** `main`
**branch_policy:** isolated
**execution_mode:** checkpoint
**Do not run:** the full pipeline locally / on WSL (OOM — Colab GPU only, per AGENTS.md).
**Untouchable:** `evaluator_v5.ipynb` (behavioral contract).

## Goal

Two regressions observed in the 2026-06-25 cluster-mode run log must be fixed:

1. **Migration degrades locally-adapted elites.** After the epoch-2 migration
   boundary, the two strong islands regressed (cluster_1: 21.74%→11.70% max
   return; cluster_0: 8.15%→7.59%). Foreign rules re-evaluated on the
   receiver's symbol slice rarely match its feature distribution, but the
   loose migration gates (`MIN_VAL_RETURN_PCT=0.0`, `MIN_VAL_TRADES=5`)
   admit them, and they displace up to `PHASE2_ARCHIVE_SEED_FRACTION=0.25`
   (50 of 200) of the converged local elites.

2. **Mid-epoch elite erosion.** On cluster_2 a 3.14%/2.89%-robust rule at
   gen 5 disappeared by gen 10 (→0.87%), with `deployable=0` the whole epoch.
   Root cause: recomputed dynamic diversity/support penalties drift upward
   as `hall_of_fame`/`pareto_archive` grow, so a non-dominated elite can be
   evicted under (μ+λ) selection purely by penalty growth — not by a
   genuinely dominating rule. Island mode disables both early stops
   (`PHASE2_ISLAND_EARLY_STOP_ENABLED=False`,
   `PHASE2_ISLAND_PLATEAU_EARLY_STOP_ENABLED=False`) and
   `PHASE2_PLATEAU_BLOCK_WHEN_DEPLOYABLE_ZERO=True` keeps dead islands
   churning, accelerating the erosion.

## Acceptance criteria (whole plan)

- AC1: A non-dominated elite present at generation N survives to generation
  N+k unless a genuinely Pareto-dominating rule appears — in **both** island
  and global profiles. (Verified by a unit test that constructs a population
  whose best rule is non-dominated, runs N+k gens with a growing archive,
  and asserts the rule is still present.)
- AC2: Migration is off by default; when re-enabled, migrants must clear
  `PHASE2_MIGRATION_MIN_VAL_RETURN_PCT ≥ 2.0` and displace ≤
  `PHASE2_MIGRATION_SEED_FRACTION` of the live population (independent of
  `PHASE2_ARCHIVE_SEED_FRACTION`).
- AC3: A dead island (deployable=0) that has plateaued for `patience` gens
  can early-stop instead of running its full gen budget.
- AC4: No regression in global (`PHASE2_ISLAND_MODE="global"`) mode —
  identical behavior when `PHASE2_MIGRATION_ENABLED=False` and elite
  preservation defaults preserve current global dynamics.
- AC5: All existing tests pass under `PYTEST_LOW_MEMORY=1`; `evaluator_v5.ipynb`
  unchanged; no new OOM risk (changes are logic/scoring, not VRAM).

## Non-goals

- Redesign `symbol_cluster.py` cluster assignment.
- Change Phase 3 / Phase 4 / RB Governor.
- Change evaluator parity constants (`FEE_PCT`, `MAX_HOLD_CANDLES`, etc.).
- Change the (μ+λ) selection algorithm itself — only add an elite-preservation
  guarantee on top of it.

---

## task-1 — Migration safety (config + small code)

**Branch:** `fix/migration-safety`

### Changes

**`gpu_fuzzy_trader/config.py`** (near the `PHASE2_MIGRATION_*` block, ~line 879):
- Add `PHASE2_MIGRATION_ENABLED: bool = False` — master switch. When False,
  the entire migration block in `_run_cluster_islands` is skipped (islands
  still run independently; final pool merge is unchanged).
- Raise defaults for the strict-gating path (used only when a user opts in):
  - `PHASE2_MIGRATION_MIN_VAL_RETURN_PCT: 0.0 → 2.0`
  - `PHASE2_MIGRATION_MIN_VAL_TRADES: 5 → 15`
  - `PHASE2_MIGRATION_TOP_K: 5 → 2` (shrink displacement footprint)
- Add `PHASE2_MIGRATION_SEED_FRACTION: float = 0.05` — separate, smaller
  seed fraction used *only* for migrant injection, decoupled from
  `PHASE2_ARCHIVE_SEED_FRACTION` (which stays 0.25 for cross-run warm-start).

**`gpu_fuzzy_trader/phases/phase2_island_scheduler.py`** (`_run_cluster_islands`):
- Guard the migration block:
  ```python
  if (
      _cfg.PHASE2_MIGRATION_ENABLED
      and epoch_counter % int(_cfg.PHASE2_MIGRATION_EPOCH_INTERVAL) == 0
      and n_clusters > 1
  ):
      ...  # existing migration body unchanged
  ```
- Remove now-dead `from gpu_fuzzy_trader.evolution.evox_runner import
  extract_deployable_migrants` from the hot path; keep it imported at module
  top so it stays available when migration is re-enabled (no dead-code
  removal inside the function body beyond the guard).

**`gpu_fuzzy_trader/phases/phase2_rule_pool.py`** (`run_epoch`, ~line 2693):
- When `self._pending_migrant_seeds` is non-empty, use
  `PHASE2_MIGRATION_SEED_FRACTION` for `seed_fraction` instead of the
  `PHASE2_STAGE_A_ARCHIVE_SEED_FRACTION` / `PHASE2_ARCHIVE_SEED_FRACTION`
  branch. This caps migrant displacement at 5% (10 of 200) rather than 25%.

**`README.md`** (migration row in §5.1 island table): note
`PHASE2_MIGRATION_ENABLED` default flipped to False and the new seed-fraction
knob. (README already has a pending user edit — only touch the migration
table row, nothing else.)

### Acceptance criteria (task-1)

- AC-T1.1: With `PHASE2_MIGRATION_ENABLED=False` (default), no
  `migration_rejected` / `Phase 2 migration: cluster … accepted` log lines
  appear and `set_pending_migrant_seeds` is never called. (Unit test
  monkeypatches `Rule_Pool_Generator.set_pending_migrant_seeds` to assert
  not called, runs a 2-cluster scheduler with tiny fake data + a mock
  engine returning constant metrics.)
- AC-T1.2: With `PHASE2_MIGRATION_ENABLED=True`, a migrant with val_return
  0.5% is rejected (log `migration_rejected reason=val_return`); a migrant
  with val_return 2.5% and ≥15 val trades is accepted. (Unit test on
  `filter_migrants_for_cluster` with synthetic metrics.)
- AC-T1.3: Migrant injection uses ≤ `PHASE2_MIGRATION_SEED_FRACTION ×
  pop_size` slots (assert in the existing `_pending_migrant_seeds` path
  test; verify the `local_cap` bound uses the new fraction).
- AC-T1.4: Global mode (`PHASE2_ISLAND_MODE="global"`) run path is byte-for-byte
  unaffected — migration code is only reached in cluster mode.

### Verification

```bash
# local, low-memory safe (no pipeline run)
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_migration*.py \
  tests/unit/test_island_scheduler*.py -x -q
.venv/bin/python -c "from gpu_fuzzy_trader import config as c; \
  assert c.PHASE2_MIGRATION_ENABLED is False; \
  assert c.PHASE2_MIGRATION_MIN_VAL_RETURN_PCT == 2.0; \
  assert c.PHASE2_MIGRATION_SEED_FRACTION == 0.05"
```

### Dependencies
- None (first task).

---

## task-2 — Elite preservation under (μ+λ) selection (code + config)

**Branch:** `fix/elite-preservation`
**Depends on:** task-1 (lands first so review diffs are clean).

### Changes

**`gpu_fuzzy_trader/config.py`** (new block after `PHASE2_DEPLOYABLE_ARCHIVE_MAX_SIZE`):
- `PHASE2_ELITE_PRESERVATION_ENABLED: bool = True` — master switch.
- `PHASE2_ELITE_PRESERVATION_TOP_K: int = 5` — number of deployable-archive
  elites force-preserved in the live population each generation.
- `PHASE2_ELITE_PRESERVATION_MIN_GEN: int = 1` — only kicks in after gen 0
  (let the first generation initialize normally).

**`gpu_fuzzy_trader/evolution/evox_runner.py`** (`run_phase2_evolution`, right
after the `_nsga3_environmental_selection` call at ~line 2177, before
`metrics_cache = [merge_metrics[int(i)] for i in sel_idx[:n_alive]]`):
- Add a helper `_preserve_deployable_elites(population, objectives,
  metrics_cache, deployable_archive, dont_cares, feature_infos, top_k,
  min_gen, current_gen)`:
  1. If disabled or `current_gen < min_gen` or `deployable_archive` empty → return.
  2. Rank `deployable_archive` by `rank_score` desc, take top-K chromosomes.
  3. Compute current live-pop keys. For each elite not already present,
     overwrite the **lowest-crowding** survivor slot (use the existing
     `_build_rank_and_crowding` on `objectives` to pick the least-crowded
     index, so we evict the most redundant survivor — not a random elite).
  4. Reset that slot's `objectives` to `inf` and `metrics_cache` to `{}` so
     it's re-evaluated next gen with consistent (current-archive) penalties.
- Wire the helper into the post-selection block in **both** the NSGA-III
  path (`run_phase2_evolution`) and the NSGA-II fallback
  (`_run_nsga2_fallback`) so global mode benefits too.

**Cleanup:** after wiring, remove any now-redundant re-evaluation of elites
that were already guaranteed preserved (none expected — this is additive).

### Acceptance criteria (task-2)

- AC-T2.1: **Elite-preservation unit test.** Build a `Phase2EvolutionState`
  with a population of 20 unique chromosomes, place one "champion"
  chromosome in `deployable_archive` with a high `rank_score`, then run the
  selection+preservation step for 15 generations with a `hall_of_fame` that
  grows each gen (simulating penalty drift). Assert the champion is present
  in `state.population` at every generation. Without the fix, the champion
  is evicted by gen ~8 (reproduce the bug first to prove the test is valid).
- AC-T2.2: Preservation never exceeds `top_k` slots and never evicts a
  chromosome that is itself in the top-K of the live Pareto front (assert
  no Pareto-front member is overwritten).
- AC-T2.3: With `PHASE2_ELITE_PRESERVATION_ENABLED=False`, the evolution loop
  is byte-for-byte identical to pre-task behavior (snapshot test on a 2-gen
  run with fixed seed → identical `history`).
- AC-T2.4: The preserved elite's `objectives` are reset to `inf` (forces
  re-eval) — assert no stale objectives survive across generations.

### Verification

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest \
  tests/unit/test_elite_preservation.py \
  tests/unit/test_evox_runner.py -x -q
# regression: global-mode snapshot
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest \
  tests/unit/test_evox_runner_global_snapshot.py -x -q
```

### Dependencies
- task-1 (so the elite-preservation branch is reviewed against a clean
  migration-gaurd baseline).

---

## task-3 — Island early-stop safety net (config + small code)

**Branch:** `fix/island-early-stop`
**Depends on:** task-2 (elite preservation makes early-stop safe — a dead
island that stops won't lose its deployable archive, which task-2 guarantees
is re-injected on resume / final pool merge).

### Changes

**`gpu_fuzzy_trader/config.py`** (island block, ~line 874):
- `PHASE2_ISLAND_PLATEAU_EARLY_STOP_ENABLED: False → True` — let plateaued
  islands stop.
- Add `PHASE2_ISLAND_PLATEAU_BLOCK_WHEN_DEPLOYABLE_ZERO: bool = False` —
  *island-scoped* override. In cluster mode a thin island that has produced
  0 deployable rules for `patience` generations is exactly the one we want
  to stop (it's just churning and eroding). The global
  `PHASE2_PLATEAU_BLOCK_WHEN_DEPLOYABLE_ZERO=True` stays as-is for global mode.
- Add `PHASE2_ISLAND_PLATEAU_EARLY_STOP_PATIENCE: int = 8` — island patience
  (longer than global's 5, since islands have less data and need more slack).

**`gpu_fuzzy_trader/evolution/evox_runner.py`** (`_should_plateau_early_stop_phase2`):
- When `scoped_island_profile(island_profile)` is True, read
  `PHASE2_ISLAND_PLATEAU_BLOCK_WHEN_DEPLOYABLE_ZERO` instead of the global
  `PHASE2_PLATEAU_BLOCK_WHEN_DEPLOYABLE_ZERO`.
- When `scoped_island_profile`, use `PHASE2_ISLAND_PLATEAU_EARLY_STOP_PATIENCE`
  if set (fall back to `PHASE2_PLATEAU_EARLY_STOP_PATIENCE` for global).
- The existing `island_plateau_early_stop_enabled()` helper already gates on
  `PHASE2_ISLAND_PLATEAU_EARLY_STOP_ENABLED` — no change needed there.

**`gpu_fuzzy_trader/config.py`** (`island_plateau_early_stop_enabled`):
- No change (already correct). Document the new patience knob in the
  helper's docstring region.

**`README.md`** (§5.1 island early-stop row + §9 interaction matrix):
- Note island plateau early-stop now defaults on; add a conflict-table row:
  "Island plateau stop on + thin island → stops at plateau instead of
  churning (was: churned full budget, eroding elites)."

### Acceptance criteria (task-3)

- AC-T3.1: A synthetic island with `deployable=0` for `patience=8` gens and
  no robust-return improvement stops at gen 8 (assert `history` length == 8
  and a `plateau early stop` log line). Pre-fix: runs full `n_generations`.
- AC-T3.2: A healthy island (`deployable>0`, still improving) does **not**
  stop early (assert `history` length == `n_generations`).
- AC-T3.3: Global mode (`island_profile="global"`) early-stop behavior is
  unchanged — `PHASE2_PLATEAU_BLOCK_WHEN_DEPLOYABLE_ZERO=True` still blocks
  plateau stop when deployable=0 (snapshot test).
- AC-T3.4: The new patience knob is respected (island stops at gen 8, not
  gen 5, when `PHASE2_ISLAND_PLATEAU_EARLY_STOP_PATIENCE=8`).

### Verification

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest \
  tests/unit/test_island_early_stop.py \
  tests/unit/test_evox_runner.py -x -q
.venv/bin/python -c "from gpu_fuzzy_trader import config as c; \
  assert c.PHASE2_ISLAND_PLATEAU_EARLY_STOP_ENABLED is True; \
  assert c.PHASE2_ISLAND_PLATEAU_BLOCK_WHEN_DEPLOYABLE_ZERO is False; \
  assert c.PHASE2_ISLAND_PLATEAU_EARLY_STOP_PATIENCE == 8"
```

### Dependencies
- task-2 (elite preservation ensures a stopped island's good rules survive
  in `deployable_archive` and are re-injected on resume / final pool merge).

---

## Execution order

1. task-1 (migration safety) → review → merge to `main`.
2. task-2 (elite preservation) → review → merge to `main`.
3. task-3 (island early-stop) → review → merge to `main`.

`execution_mode: checkpoint` — orchestrator pauses after each task for the
user's "continue task N" signal. `branch_policy: isolated` — each task gets
its own feature branch off `main`; pre-dispatch isolation validation runs
before the implementer is dispatched.

## Final acceptance (end of plan)

All three tasks merged to `main`, feature branches deleted by the
implementer via `branch-cleanup-prompt.md`, `CONTEXT.md` updated to
"all tasks merged", and the user re-runs the pipeline on Colab GPU to
confirm: (a) no `Phase 2 migration: … accepted` lines, (b) cluster_1-style
islands hold their peak across epochs, (c) cluster_2-style dead islands
early-stop instead of churning to 0.87%.