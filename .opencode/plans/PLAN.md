# Plan: Fix Empty Phase 2 Pool (Missing Val Engine Rebuild)

## Goal
Fix the bug where Phase 2 produces zero rules despite islands finding deployable rules
during evolution. The root cause is that `_scoped_val_df` is discarded after the first
`_build_engines()` call, preventing the val backtest engine from being rebuilt after
`park_engines()` — which causes ALL pool entries to fail the admission gate.

## Background

The pipeline log shows:
- Islands produce deployable rules during evolution (e.g., `deployable=6`)
- Pool builder accepts ~100 eligible chromosomes per island
- But final pools are EMPTY (pool_size=0 in pipeline log)

Root cause chain:
1. `__init__` saves `val_df` → `_scoped_val_df`, calls `_build_engines()`, then sets `_scoped_val_df = None`
2. `_build_engines()` also sets `_scoped_val_df = None` at its end
3. `park_engines()` clears `_val_engine = None`
4. `finalize_island()` → `_ensure_engines()` → `_build_engines()` can't rebuild val engine because `_scoped_val_df` is None
5. `_build_pool_from_archive()` runs with `val_engine=None` → all entries fail `passes_pool_admission_gate()`

Train data works because it has `_cached_slim_train` but val data has no equivalent cache.

## Task

### task-1: Cache slim val DataFrame for engine rebuilds

**Target files:**
- `gpu_fuzzy_trader/phases/phase2_rule_pool.py`

**Changes:**

1. In `__init__` (after `_build_engines()` but before discarding `_scoped_val_df`):
   - Save the already-built val engine's slim DataFrame (`self._val_engine._df`) as `self._cached_slim_val`
   - Save regime context: `self._cached_val_regime_ids` and `self._cached_val_regime_row_counts`

2. In `_build_engines()`:
   - Replace `if self._scoped_val_df is not None:` with `if self._cached_slim_val is not None:`
   - Rebuild val engine directly from cached slim data (no re-sampling needed)

3. Remove redundant `self._scoped_val_df = None` in `_build_engines()` (keep the one in `__init__`)

**Acceptance criteria:**
- `finalize_island()` produces non-empty pools when islands have deployable rules
- The fix mirrors the existing `_cached_slim_train` / `_rebuild_train_df` pattern
- No new dependencies or config changes
