# task-3: Island early-stop safety net

**Branch:** `fix/island-early-stop`
**Depends on:** task-2 (merged to main — elite preservation ensures stopped island's good rules survive in `deployable_archive`)

## Goal

Let dead islands (deployable=0, plateaued) early-stop instead of churning for the full generation budget. Currently `PHASE2_ISLAND_PLATEAU_EARLY_STOP_ENABLED=False` and `PHASE2_PLATEAU_BLOCK_WHEN_DEPLOYABLE_ZERO=True` traps thin islands that have produced zero deployable rules — they keep evolving, eroding their few viable elites across the entire epoch (as seen on cluster_2: 6.64% → 0.87%).

## Changes

### 1. `gpu_fuzzy_trader/config.py` — flip island early-stop defaults

In the island block (~line 883):

```python
# BEFORE:
PHASE2_ISLAND_PLATEAU_EARLY_STOP_ENABLED = False

# AFTER — add these lines:
PHASE2_ISLAND_PLATEAU_EARLY_STOP_ENABLED = True
PHASE2_ISLAND_PLATEAU_BLOCK_WHEN_DEPLOYABLE_ZERO: bool = False
PHASE2_ISLAND_PLATEAU_EARLY_STOP_PATIENCE: int = 8
```

Notes:
- `PHASE2_ISLAND_PLATEAU_EARLY_STOP_ENABLED`: flip from `False` → `True`
- `PHASE2_ISLAND_PLATEAU_BLOCK_WHEN_DEPLOYABLE_ZERO`: new — island-scoped override. `False` means a dead island (0 deployable) CAN still early-stop on plateau. The global `PHASE2_PLATEAU_BLOCK_WHEN_DEPLOYABLE_ZERO=True` stays as-is for global mode.
- `PHASE2_ISLAND_PLATEAU_EARLY_STOP_PATIENCE`: new — island patience of 8 gens (longer than global's 5, since islands have less data and need more slack to recover).

### 2. `gpu_fuzzy_trader/evolution/evox_runner.py` — `_should_plateau_early_stop_phase2`

In `_should_plateau_early_stop_phase2` (~line 559), modify two blocks:

**Block A: deployable-zero guard (~line 581)**

```python
# BEFORE:
if bool(getattr(_cfg, "PHASE2_PLATEAU_BLOCK_WHEN_DEPLOYABLE_ZERO", True)):
    if deployable_count <= 0:
        return False

# AFTER:
if scoped_island_profile(island_profile):
    if bool(getattr(_cfg, "PHASE2_ISLAND_PLATEAU_BLOCK_WHEN_DEPLOYABLE_ZERO", False)):
        if deployable_count <= 0:
            return False
elif bool(getattr(_cfg, "PHASE2_PLATEAU_BLOCK_WHEN_DEPLOYABLE_ZERO", True)):
    if deployable_count <= 0:
        return False
```

**Block B: patience (~line 590)**

```python
# BEFORE:
patience = (
    int(stage_params.plateau_early_stop_patience)
    if stage_params is not None
    else int(_cfg.PHASE2_PLATEAU_EARLY_STOP_PATIENCE)
)

# AFTER:
if scoped_island_profile(island_profile):
    patience = (
        int(stage_params.plateau_early_stop_patience)
        if stage_params is not None and getattr(stage_params, "plateau_early_stop_patience", None) is not None
        else int(getattr(_cfg, "PHASE2_ISLAND_PLATEAU_EARLY_STOP_PATIENCE", _cfg.PHASE2_PLATEAU_EARLY_STOP_PATIENCE))
    )
else:
    patience = (
        int(stage_params.plateau_early_stop_patience)
        if stage_params is not None
        else int(_cfg.PHASE2_PLATEAU_EARLY_STOP_PATIENCE)
    )
```

### 3. `tests/unit/test_island_early_stop.py` — new test file

Write tests that:

- **AC-T3.1**: A synthetic island with `deployable=0` for `patience=8` gens and no robust-return improvement stops at gen 8 (assert `history` length == 8 and log contains "plateau early stop"). Pre-fix behavior: runs full `n_generations` (mock `PHASE2_ISLAND_PLATEAU_EARLY_STOP_ENABLED=False` and assert history length == n_generations).

- **AC-T3.2**: A healthy island (`deployable>0`, still improving robust-return) does NOT stop early (assert history length == n_generations).

- **AC-T3.3**: Global mode (`island_profile="global"`) early-stop behavior is unchanged — `PHASE2_PLATEAU_BLOCK_WHEN_DEPLOYABLE_ZERO=True` still blocks plateau stop when deployable=0.

- **AC-T3.4**: The island patience knob is respected (stops at gen 8 when `PHASE2_ISLAND_PLATEAU_EARLY_STOP_PATIENCE=8`, not gen 5).

### 4. `README.md` — update §5.1 island table

Update the island early-stop row to reflect `PHASE2_ISLAND_PLATEAU_EARLY_STOP_ENABLED=True`. Add rows for the two new knobs.

## Acceptance criteria

- AC-T3.1: Dead island (deployable=0, plateaued) stops at gen 8 instead of running full budget.
- AC-T3.2: Healthy island (deployable>0, improving) does NOT stop early.
- AC-T3.3: Global mode unchanged — `PHASE2_PLATEAU_BLOCK_WHEN_DEPLOYABLE_ZERO=True` still blocks.
- AC-T3.4: `PHASE2_ISLAND_PLATEAU_EARLY_STOP_PATIENCE=8` is respected (not global's 5).

## Verification

```bash
cd /home/danaee/trading_platform && source .venv/bin/activate && \
  PYTEST_LOW_MEMORY=1 python -m pytest \
    tests/unit/test_island_early_stop.py \
    -x -v --tb=short
```

Also verify config:
```bash
python -c "
from gpu_fuzzy_trader import config as c
assert c.PHASE2_ISLAND_PLATEAU_EARLY_STOP_ENABLED is True
assert c.PHASE2_ISLAND_PLATEAU_BLOCK_WHEN_DEPLOYABLE_ZERO is False
assert c.PHASE2_ISLAND_PLATEAU_EARLY_STOP_PATIENCE == 8
print('config OK')
"
```

## Files to modify

- `gpu_fuzzy_trader/config.py` — 1 flip + 2 new keys
- `gpu_fuzzy_trader/evolution/evox_runner.py` — 2 blocks in `_should_plateau_early_stop_phase2`
- `tests/unit/test_island_early_stop.py` — new test file
- `README.md` — island early-stop table update
