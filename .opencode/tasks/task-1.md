# Task 1: Cache slim val DataFrame for engine rebuilds

## Goal
Fix the bug where `_scoped_val_df` is discarded after the first `_build_engines()` call,
preventing the val backtest engine from being rebuilt after `park_engines()`. This causes
ALL pool entries to fail the admission gate (val_engine=None → val_metrics=None → rejected).

## Root Cause
1. `__init__` sets `_scoped_val_df = val_df`, calls `_build_engines()`, then discards it: `_scoped_val_df = None`
2. `_build_engines()` also sets `_scoped_val_df = None` at its end
3. `park_engines()` clears `_val_engine = None`
4. `finalize_island()` → `_ensure_engines()` → `_build_engines()` can't rebuild val engine
5. `_build_pool_from_archive()` runs with `val_engine=None` → all entries fail admission gate

Train data works because it has `_cached_slim_train` / `_rebuild_train_df()`. Val data needs the same pattern.

## Target File
- `gpu_fuzzy_trader/phases/phase2_rule_pool.py`

## Required Changes

### 1. Add cached val attributes in `__init__` (around line 2030-2042)
After `_build_engines()` but before `_scoped_val_df = None`, save the val engine's internal data:
```python
# Cache val data for rebuilds (mirrors _cached_slim_train pattern)
if self._val_engine is not None:
    self._cached_slim_val = getattr(self._val_engine, "_df", None)
    self._cached_val_regime_ids = getattr(self._val_engine, "_regime_ids", None)
    self._cached_val_regime_row_counts = self._val_regime_row_counts
else:
    self._cached_slim_val = None
    self._cached_val_regime_ids = None
    self._cached_val_regime_row_counts = None
```

### 2. Add initialization of cached val attrs early in `__init__`
Before `_build_engines()` call, add:
```python
self._cached_slim_val = None
self._cached_val_regime_ids = None
self._cached_val_regime_row_counts = None
```

### 3. Modify `_build_engines()` val engine rebuild (around line 2072-2122)
Replace the check `if self._scoped_val_df is not None:` with `if self._cached_slim_val is not None:`
and rebuild the val engine directly from the cached slim DataFrame:
```python
if self._cached_slim_val is not None:
    self._val_engine = self._build_engine_for_df(
        self._cached_slim_val,
        regime_ids=self._cached_val_regime_ids,
        n_regimes=len(self._cached_val_regime_row_counts) if self._cached_val_regime_row_counts is not None else 0,
    )
    self._holdout_n_valid_rows = len(self._cached_slim_val)
    self._val_engine.n_valid_rows = len(self._cached_slim_val)
    if self._cached_val_regime_row_counts is not None:
        self._val_engine._regime_row_counts = self._cached_val_regime_row_counts
    # Log message (keep existing joint_train_val log)
```

### 4. Remove `self._scoped_val_df = None` from `_build_engines()` (line ~2122)
Let the `__init__` cleanup handle it instead.

### 5. Do NOT clear cached val data in `park_engines()`
Keep `_cached_slim_val`, `_cached_val_regime_ids`, `_cached_val_regime_row_counts` intact.

## Acceptance Criteria
- `finalize_island()` produces non-empty pools when islands have deployable rules
- After `park_engines()` + `_ensure_engines()`, `self._val_engine` is NOT None
- The fix mirrors the existing `_cached_slim_train` / `_rebuild_train_df` pattern
- No new dependencies or config changes

## Verification
- Run: `python -c "from gpu_fuzzy_trader.phases.phase2_rule_pool import Rule_Pool_Generator; print('Import OK')"`
- Code review: val engine rebuild path works after parking
