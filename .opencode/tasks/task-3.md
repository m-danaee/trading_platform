# Task 3: Strip regime from Phase 2 (support, pool, init, scheduler)

**Goal:** Remove all regime-related functions, parameters, and branching from the 4 Phase 2 modules. This is the largest change — ~500 lines removed.

## Target files

1. `gpu_fuzzy_trader/phases/phase2_support.py`
2. `gpu_fuzzy_trader/phases/phase2_rule_pool.py`
3. `gpu_fuzzy_trader/phases/phase2_init.py`
4. `gpu_fuzzy_trader/phases/phase2_island_scheduler.py`

---

## Changes in phase2_support.py

### 3a. Remove dead functions (delete entirely)
- `_compact_regime_labels()` (lines ~74-109)
- `regime_row_fractions()` (lines ~129-137)
- `per_regime_trade_thresholds()` (lines ~141-163)
- `_is_regime_specialist()` (lines ~166-210)
- `val_regime_confirmation()` (lines ~482-506)
- `metrics_regime_arrays()` (lines ~550-556)

### 3b. Simplify `trade_support_penalty()` (lines ~229-278)
Remove all regime parameters:
```python
def trade_support_penalty(
    executed: int,
    *,
    min_trade_support: int | None = None,
) -> tuple[float, bool, int]:
```
Body: always call `_static_support_penalty(executed, min_trade_support=support_target)`, return `(pen, False, -1)`. Remove all `use_regime` / `to_host_numpy` / `_is_regime_specialist` logic.

### 3c. Simplify `compute_support_penalty_and_specialist()` (lines ~561-597)
Remove `regime_row_fractions_arr`, `val_regime_row_counts`, `val_metrics` parameters. Body: just call `trade_support_penalty(executed, min_trade_support=min_trade_support)` and return `(penalty, False, -1)`.

### 3d. Simplify `_passes_pool_admission_impl()` (lines ~300-380)
- Remove the "Regime Profitability Gate" block (lines ~324-336) that checks `PHASE2_REGIME_PROFITABILITY_GATE`
- Remove the `PHASE2_REGIME_REQUIRE_VAL_CONFIRMATION` / `val_regime_confirmation()` block (lines ~367-373)

### 3e. Simplify `passes_pool_trade_floor()` (lines ~443-480)
- Remove `regime_row_fractions_arr` parameter
- Remove the entire `PHASE2_REGIME_SUPPORT_ENABLED` / regime specialist waiver block after `return False`
- After the `executed >= trade_floor` check returns True, just `return False` on failure

### 3f. Clean `passes_pool_admission_gate()` (line ~431-432)
Remove `regime_specialist` and `dominant_regime` from `train_metrics` dict construction (they'll always be False/-1).

### 3g. Update imports
Remove any remaining regime-related imports from the top of the file. If `to_host_numpy` is still used elsewhere in the file, keep it.

---

## Changes in phase2_rule_pool.py

### 3h. Remove `_prepare_regime_context()` function (lines ~75-131)
Delete the entire function and all its imports (`assign_regime_labels`, `load_regime_model`, `_compact_regime_labels`).

### 3i. Remove `_regime_feature_indices()` function
If present in this file (check), delete it. It should only exist in phase2_init.py.

### 3j. Remove regime init from `_init_population()` (lines ~1095-1128)
Remove the 3-stratum (`elite/explorer/regime`) init path. Always use the 2-stratum (`elite/explorer`) path.
- Remove `_regime_enabled` variable and the `if _regime_enabled:` branch
- Remove `sample_regime_stratum_chromosome` import (line ~970)
- Remove `regime_frac` / `three_fractions` / `regime` stratum sampling

### 3k. Clean function signatures (multiple locations)
Remove from all functions:
- `regime_row_fractions_arr` parameter
- `val_regime_row_counts` parameter
- `regime_row_fractions_arr=getattr(engine, "_regime_row_fractions", None)` calls
- `val_regime_row_counts = getattr(val_engine, "_regime_row_counts", None)` calls

### 3l. Clean `compute_phase2_objectives_from_metrics()` (line ~536+)
- Remove `regime_row_fractions_arr` and `val_regime_row_counts` parameters
- Simplify the `compute_support_penalty_and_specialist()` call — remove regime params, just pass `metrics` and `min_trade_support`
- Remove `is_specialist`/`dominant_regime` variable usage and `metrics["regime_specialist"]` / `metrics["dominant_regime"]` assignments

### 3m. Remove regime pool-entry fields (lines ~1437-1441)
Remove lines that set `pool_entry["regime_specialist"]`, `pool_entry["dominant_regime"]`, `pool_entry["regime_trade_counts"]`.

### 3n. Clean engine attributes (lines ~1999-2130)
In the engine's init and `_init_engines()`, remove:
- `self._regime_row_fractions`, `self._n_regimes`, `self._val_regime_row_counts`
- `self._train_regime_ids`, `self._cached_val_regime_ids`, `self._cached_val_regime_row_counts`
- `self._cached_slim_train_regime` and related caching
- All `_prepare_regime_context()` calls and `regime_ids=` engine kwargs
- `val_regime_ids` / `val_regime_row_counts` computation
- Engine `_regime_row_counts` attribute setting

---

## Changes in phase2_init.py

### 3o. Revert `Stratum` literal (line ~13)
```python
Stratum = Literal["elite", "explorer"]  # remove "regime"
```

### 3p. Remove dead functions
- `_regime_feature_indices()` (lines ~16-28)
- `assign_three_strata_to_indices()` (lines ~284-330)
- `sample_regime_stratum_chromosome()` (lines ~335-370)

### 3q. Simplify `_init_population()`
- Remove `_regime_enabled` / 3-stratum branching — always use 2-stratum
- Remove `regime_frac` related code
- The `stratum_fractions` parameter should be `tuple[float, float]` for 2-stratum
- Remove code that builds 3-tuple fractions from 2-tuple

---

## Changes in phase2_island_scheduler.py

### 3r. Remove regime parameter (lines ~128-129)
Remove `regime_row_fractions_arr=getattr(train_engine, "_regime_row_fractions", None)` from function calls.

---

## Acceptance criteria

1. **Zero regime references in all 4 files:**
   ```bash
   rg -i regime gpu_fuzzy_trader/phases/phase2_support.py gpu_fuzzy_trader/phases/phase2_rule_pool.py gpu_fuzzy_trader/phases/phase2_init.py gpu_fuzzy_trader/phases/phase2_island_scheduler.py
   ```
   Returns empty.

2. **All imports work:**
   ```bash
   python -c "from gpu_fuzzy_trader.phases.phase2_support import trade_support_penalty, _passes_pool_admission_impl; print('support OK')"
   python -c "from gpu_fuzzy_trader.phases.phase2_rule_pool import Rule_Pool_Generator; print('pool OK')"
   python -c "from gpu_fuzzy_trader.phases.phase2_init import _init_population, Stratum; print('init OK')"
   python -c "from gpu_fuzzy_trader.phases.phase2_island_scheduler import IslandScheduler; print('scheduler OK')"
   ```

3. **No `regime_cluster` imports in any file:**
   ```bash
   rg "regime_cluster" gpu_fuzzy_trader/phases/phase2_*.py
   ```
   Returns empty.

4. **`Stratum` literal is `Literal["elite", "explorer"]` (no "regime"):**
   ```bash
   python -c "from gpu_fuzzy_trader.phases.phase2_init import Stratum; import typing; assert typing.get_args(Stratum) == ('elite', 'explorer'), f'Got {typing.get_args(Stratum)}'"
   ```

5. **Run existing tests (excluding regime-specific ones that will fail):**
   ```bash
   PYTEST_LOW_MEMORY=1 python -m pytest tests/unit/ -x --timeout=120 --ignore=tests/unit/test_regime_cluster.py --ignore=tests/unit/test_regime_keyword_stratum.py --ignore=tests/unit/test_regime_profitability_gate.py --ignore=tests/unit/test_gpu_engine_regime.py 2>&1 | tail -30
   ```

## Verification commands

```bash
cd /home/danaee/trading_platform && source .venv/bin/activate

# Zero regime references
rg -i regime gpu_fuzzy_trader/phases/phase2_support.py gpu_fuzzy_trader/phases/phase2_rule_pool.py gpu_fuzzy_trader/phases/phase2_init.py gpu_fuzzy_trader/phases/phase2_island_scheduler.py && echo "FAIL" || echo "PASS"

# Zero regime_cluster imports  
rg "regime_cluster" gpu_fuzzy_trader/phases/phase2_*.py && echo "FAIL" || echo "PASS"

# Import checks
python -c "from gpu_fuzzy_trader.phases.phase2_support import trade_support_penalty; print('support OK')"
python -c "from gpu_fuzzy_trader.phases.phase2_rule_pool import Rule_Pool_Generator; print('pool OK')"
python -c "from gpu_fuzzy_trader.phases.phase2_init import _init_population; print('init OK')"
python -c "from gpu_fuzzy_trader.phases.phase2_island_scheduler import IslandScheduler; print('scheduler OK')"

# Stratum check
python -c "from gpu_fuzzy_trader.phases.phase2_init import Stratum; import typing; assert typing.get_args(Stratum) == ('elite', 'explorer'), f'Got {typing.get_args(Stratum)}'; print('stratum OK')"

# Tests (excluding regime-specific ones)
PYTEST_LOW_MEMORY=1 python -m pytest tests/unit/ -x --timeout=120 --ignore=tests/unit/test_regime_cluster.py --ignore=tests/unit/test_regime_keyword_stratum.py --ignore=tests/unit/test_regime_profitability_gate.py --ignore=tests/unit/test_gpu_engine_regime.py 2>&1 | tail -35
```
