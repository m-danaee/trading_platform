# Task 1: Per-Epoch Train-Window Rotation

## Task ID
`task-1` (new series; this is the first task in the 2026-07-07 audit fix plan)

## Title
Per-Epoch Train-Window Rotation

## Goal
Fix audit finding #1 (per-island/per-epoch window resampling is dead).
Every epoch currently backtests the SAME fixed full-train slice
because `_sample_df` is called once at cluster init with
`PHASE1_SAMPLING_TOTAL=701_000` rows / 4 syms = 175k/sym, which
exceeds the per-symbol safe range (119059 in holdout mode), forcing
`start = safe_start = 0` for the whole cluster lifetime. Rotate the
train-window start per epoch with a deterministic seed so each epoch
sees a different contiguous sub-window. This unblocks task-10
(disable cache refresh when window is fixed) and task-11 (raise
`PHASE2_VAL_SIM_INTERVAL` to 3).

## Audit Citation
- Confirmed by static inspection: `phases/phase2_rule_pool.py:391`
  (fallback at `:479`) and `phases/phase2_island_scheduler.py:421`
  (single `_sample_df` call per cluster).
- Run log evidence (2026-07-07): the "requested 175000 bars/sym
  exceeds largest safe range 119059" warning fires on EVERY cluster,
  EVERY epoch.

## Target Files
- `gpu_fuzzy_trader/phases/phase2_island_scheduler.py`
  (`_run_cluster_islands` at :414-440; per-epoch sample rebuild)
- `gpu_fuzzy_trader/phases/phase2_rule_pool.py`
  (`_sample_df` cap helper or caller-side cap; `_rebuild_train_df`
  at :2375)
- `gpu_fuzzy_trader/phases/phase2_support.py`
  (new helper: `_sample_epoch_window(df, total_rows, seed, syms, ...)`)
- `gpu_fuzzy_trader/config.py` (new `PHASE2_PER_EPOCH_WINDOW_ROTATION`
  flag, default True; new `PHASE2_PER_EPOCH_WINDOW_SEED_MODE`
  enum, default `"hash_island_epoch"`)
- `tests/unit/test_phase2_island_scheduler.py`
- New `tests/unit/test_phase2_window_rotation.py`

## Current Behavior
1. `phase2_island_scheduler.py:_run_cluster_islands` calls
   `_sample_df(scoped_train, PHASE1_SAMPLING_TOTAL, random_state=seed)`
   ONCE per cluster at line 421, computes `sampled_rows` for island
   hyperparam resolution, then constructs `Rule_Pool_Generator` with
   `defer_warmup=True`.
2. `Rule_Pool_Generator.__init__` re-samples (independently) at
   `phase2_rule_pool.py:2223-2231` and caches the slim result in
   `self._cached_slim_train`.
3. `run_epoch()` calls `_ensure_engines()` (phase2_rule_pool.py:2967),
   which calls `_rebuild_train_df` (`:2375`) — restores the SAME
   cached slice every epoch.
4. The `_sample_df` RNG branch at :471 (`start = int(rng.integers(...))`)
   is unreachable because `n_per_sym > safe_len` triggers the
   fallback at :479.

## Scope
- Add a per-epoch sample rotation in
  `phase2_island_scheduler.py:_run_cluster_islands` (or
  equivalently in `Rule_Pool_Generator.run_epoch`) that:
  - Uses seed `hash(island_id, epoch_idx)` (or `_derive_island_seed`
    composed with `epoch_idx`) — deterministic, no RNG state leak.
  - Caps `total_rows` to `min(PHASE1_SAMPLING_TOTAL, 4 × safe_len_per_sym)`,
    so the RNG branch at `phase2_rule_pool.py:471` actually fires.
  - Keeps temporal alignment: a single shared start bar across all
    syms in the cluster (matches current contract; required for
    cross-symbol relative-strength signals).
- Gate the behavior behind a new config flag
  `PHASE2_PER_EPOCH_WINDOW_ROTATION` (default True). When False,
  preserve current single-sample behavior (useful for A/B tests).
- Update `Rule_Pool_Generator._rebuild_train_df` to optionally
  re-sample with the per-epoch seed, or add a new
  `_rebuild_train_df_for_epoch(epoch_idx)` method.
- Do NOT change the engine warmup/re-park lifecycle (keep sequential
  cluster warmup, single cluster alive at a time).
- Do NOT touch `evaluator_v5.ipynb`, `rb_governor.py`, or any other
  phase files.

## Acceptance Criteria
1. The "requested N bars/sym exceeds largest safe range" warning
   does NOT fire on the first cluster epoch when
   `n_per_sym <= safe_len_per_sym`.
2. The train DataFrame for cluster 0 at gen 1 differs from gen 7
   (verifiable by checking `_symbol_bar_index.min()` and
   `datetime.min()` per symbol).
3. With `PHASE2_PER_EPOCH_WINDOW_ROTATION=False`, behavior matches
   pre-task exactly (regression guard).
4. With `PHASE2_PER_EPOCH_WINDOW_ROTATION=True`, the
   `refresh_objectives_on_resume` flag in `evox_runner.py:2606`
   is still consulted (do NOT disable it here — that's task-10).
5. Sequential cluster warmup still works; no VRAM regression
   (peak signatures stay at 2 per cluster as in the current
   `evict_cluster_signatures` flow).
6. All existing `test_phase2_island_scheduler.py`,
   `test_phase2_migration_gate.py`, `test_phase2_island_early_stop.py`,
   `test_phase2_island_hyperparams.py` tests still pass.

## Verification
Run only the related unit tests with `PYTEST_LOW_MEMORY=1` and
`.venv`, for example:

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_window_rotation.py -q
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_island_scheduler.py -q
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_island_early_stop.py -q
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_migration_gate.py -q
```

If any test requires JAX warmup that is too expensive, mark it
`@pytest.mark.uses_jax` so the low-memory fixture can clear caches.

## Notes
- Do not modify `evaluator_v5.ipynb`.
- Do not run the full project or full test suite locally
  (per AGENTS.md; user runs on Colab GPU).
- Keep the diff small and focused on this task. Tasks 10 and 11
  in the plan depend on this work — they will be dispatched in
  subsequent rounds.
- If implementation reveals the fix is unsafe (e.g., causes VRAM
  regression or breaks cluster eviction), return BLOCKED with
  evidence rather than ship a partial fix.
