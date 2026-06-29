# Task 1 — Post-Restart No-Improvement Early Stop

**Branch:** `feature/task-1-post-restart-early-stop` (from `main`)
**Skill:** implementer

## Goal
Cut Phase 2 wall-clock by breaking island epochs when a plateau restart fails
to yield improvement.  Surgical: only cuts generations that produce zero
improvement after a stall was already declared.

## Files to edit
1. `gpu_fuzzy_trader/config.py`
2. `gpu_fuzzy_trader/evolution/evox_runner.py`
3. `tests/unit/test_phase2_post_restart_stop.py` (NEW)
4. `tests/unit/test_island_early_stop.py` (1 assertion update)

## Hard constraints (AGENTS.md)
- Use `.venv` for all commands.
- Run tests ONLY with `PYTEST_LOW_MEMORY=1` env var set (OOM risk otherwise).
- Do NOT run the full pipeline. Do NOT touch `evaluator_v5.ipynb`.
- Remove dead/obsolete code after edits.

---

## EDIT 1 — config.py

### 1a. Change island plateau patience 8 → 6
Find:
```python
PHASE2_ISLAND_PLATEAU_EARLY_STOP_PATIENCE: int = 8
```
Change to:
```python
PHASE2_ISLAND_PLATEAU_EARLY_STOP_PATIENCE: int = 6
```
(Rationale: 6 gens of <0.5% improvement is a clear stall; restarts fire sooner
so the new post-restart stop has more gens to evaluate.)

### 1b. Add 4 new knobs
Insert immediately AFTER the `PHASE2_PLATEAU_POST_RESTART_BOOST_GENS = 3`
block (which sits before `PHASE2_PLATEAU_MAX_RESTARTS`).  Use this exact text:

```python
# PHASE2_PLATEAU_POST_RESTART_STOP_ENABLED — break the epoch when the best
#   metric fails to improve for PHASE2_PLATEAU_POST_RESTART_STOP_PATIENCE gens
#   AFTER a plateau restart.  A restart already signals a stall; if fresh blood
#   + boosted mutation yields no progress within the boost window, further gens
#   are very unlikely to help.  Cuts only provably-unproductive generations.
#   True  → stop after a failed restart (default; safe runtime win).
#   False → always run the full epoch budget after a restart (original behaviour).
PHASE2_PLATEAU_POST_RESTART_STOP_ENABLED = True

# PHASE2_PLATEAU_POST_RESTART_STOP_PATIENCE — gens of no improvement after a
#   restart before breaking.  Should be >= PHASE2_PLATEAU_POST_RESTART_BOOST_GENS
#   so the boosted-mutation window gets a full chance to recover.
#   Higher → more conservative (give restart more time); slower.
#   Lower  → stop sooner after a failed restart; faster, tiny risk of cutting a
#            late breakthrough (an improvement resets the streak, so no false stop).
PHASE2_PLATEAU_POST_RESTART_STOP_PATIENCE = 3

# Island-scoped variants (mirror PHASE2_ISLAND_PLATEAU_EARLY_STOP_* scoping).
PHASE2_ISLAND_PLATEAU_POST_RESTART_STOP_ENABLED = True
PHASE2_ISLAND_PLATEAU_POST_RESTART_STOP_PATIENCE = 3
```

---

## EDIT 2 — evox_runner.py

### 2a. Phase2EvolutionState dataclass
Add two fields immediately AFTER the existing `post_restart_gens_remaining: int = 0`
field (around line 58):
```python
    post_restart_no_improve_streak: int = 0
    post_restart_best_progress: float = -np.inf
```

### 2b. New helper
Add this function immediately AFTER `_should_plateau_early_stop_phase2`
(its `return streak >= patience` is the last line of that function, ~line 600):
```python
def _should_post_restart_early_stop_phase2(
    post_restart_streak: int,
    *,
    island_profile: str = "global",
    stage_params: Phase2StageParams | None = None,
) -> bool:
    """Break the epoch when a plateau restart yields no improvement.

    Independent of the main plateau streak/patience so it never interferes
    with the restart decision itself — it only fires on generations AFTER a
    restart, measuring whether the restart produced any progress.
    """
    if _cfg.scoped_island_profile(island_profile):
        if not bool(getattr(
            _cfg, "PHASE2_ISLAND_PLATEAU_POST_RESTART_STOP_ENABLED", True,
        )):
            return False
        patience = int(getattr(
            _cfg, "PHASE2_ISLAND_PLATEAU_POST_RESTART_STOP_PATIENCE",
            getattr(_cfg, "PHASE2_PLATEAU_POST_RESTART_STOP_PATIENCE", 3),
        ))
    else:
        if not bool(getattr(
            _cfg, "PHASE2_PLATEAU_POST_RESTART_STOP_ENABLED", True,
        )):
            return False
        patience = int(getattr(
            _cfg, "PHASE2_PLATEAU_POST_RESTART_STOP_PATIENCE", 3,
        ))
    return post_restart_streak >= patience
```

### 2c. BOTH generation loops — identical pattern

There are TWO loops with the same restart/stop structure:
- `_run_nsga2_fallback` — loop starts ~line 1634 (`for gen in range(n_generations):`)
- `_run_nsga3` — loop starts ~line 2094 (`for gen in range(n_generations):`)

Apply ALL of the following to BOTH loops:

**(i) Init** — next to the existing `post_restart_gens_remaining: int = 0`
declaration (which sits just before the `for gen in range(n_generations):` line):
```python
    post_restart_no_improve_streak: int = 0
    post_restart_best_progress: float = -np.inf
```

**(ii) `just_restarted` flag** — at the very top of the loop body (the first
statement inside `for gen in range(n_generations):`), add:
```python
        just_restarted = False
```

**(iii) Viability-collapse restart block** — find the block:
```python
            n_elite_kept = _plateau_diversity_restart(
                population, objectives, metrics_cache,
                feature_infos, rng,
                pareto_indices=pareto_indices,
                pop_size=pop_size,
                init_strategy=init_strategy,
                stratum_fractions=stratum_fractions,
            )
            restart_count += 1
            pre_reset_streak = viability_collapse_streak
            viability_collapse_streak = 0
            post_restart_gens_remaining = int(getattr(
                _cfg,
                "PHASE2_PLATEAU_POST_RESTART_BOOST_GENS",
                3,
            ))
```
After the `post_restart_gens_remaining = int(...)` assignment inside this
viability block, add:
```python
            post_restart_best_progress = plateau_best_progress
            post_restart_no_improve_streak = 0
            just_restarted = True
```
(This block does NOT `continue`, so the `just_restarted` guard below prevents a
spurious +1 on the restart generation itself.)

**(iv) After the main `_update_max_return_plateau` call** — find:
```python
        plateau_best_progress, plateau_streak = _update_max_return_plateau(
            plateau_metric, plateau_best_progress, plateau_streak,
        )
```
Immediately AFTER it, add:
```python
        if restart_count > 0 and not just_restarted:
            post_restart_best_progress, post_restart_no_improve_streak = (
                _update_max_return_plateau(
                    plateau_metric,
                    post_restart_best_progress,
                    post_restart_no_improve_streak,
                )
            )
```

**(v) Plateau restart block (the one with `continue`)** — find the block that
logs `"Phase 2 [%s]: plateau restart at gen %d"`.  After the line
`plateau_streak = 0` (which sits before the `logger.info(...)` for the restart),
add:
```python
                    post_restart_best_progress = plateau_best_progress
                    post_restart_no_improve_streak = 0
                    just_restarted = True
```
This block ends with `continue  # skip env selection; go to next gen`, so the
new stop check below is correctly skipped on the restart generation.

**(vi) New post-restart stop check** — locate the `_should_plateau_early_stop_phase2`
block (the `if _should_plateau_early_stop_phase2(...):` ... `break` block).  Place
the new check IMMEDIATELY AFTER that entire block closes (after its `break`),
and BEFORE the `if gen == n_generations - 1: break` line (in `_run_nsga2_fallback`)
or the `if is_last_gen: break` line (in `_run_nsga3`):
```python
        if (
            restart_count > 0
            and _should_post_restart_early_stop_phase2(
                post_restart_no_improve_streak,
                stage_params=stage_params,
                island_profile=island_profile,
            )
        ):
            logger.info(
                "%s: post-restart early stop at gen %d "
                "(no improvement for %d gens after restart %d/%d, "
                "best_progress=%.2f%%, deployable_preview=%d)",
                tag,
                gen + 1,
                post_restart_no_improve_streak,
                restart_count,
                max_restarts,
                plateau_best_progress,
                deployable_count,
            )
            break
```

### 2d. `_run_nsga3` ONLY — resumable state read/write
- On resume (where it reads `plateau_streak = int(state.plateau_streak)` etc.),
  add right after reading `post_restart_gens_remaining`:
  ```python
        post_restart_no_improve_streak = int(
            getattr(state, "post_restart_no_improve_streak", 0)
        )
        post_restart_best_progress = float(
            getattr(state, "post_restart_best_progress", -np.inf)
        )
  ```
  (Use `getattr` — `test_plateau_state_leak._mock_evolution_state` constructs the
  state via `__new__` without these fields.)
- In the `final_state = Phase2EvolutionState(...)` constructor call, add:
  ```python
        post_restart_no_improve_streak=post_restart_no_improve_streak,
        post_restart_best_progress=post_restart_best_progress,
  ```

---

## EDIT 3 — tests/unit/test_phase2_post_restart_stop.py (NEW)
Create with these tests:
```python
"""Unit tests for post-restart no-improvement early stop (Phase 2 runtime)."""

from __future__ import annotations

from gpu_fuzzy_trader import config as cfg
from gpu_fuzzy_trader.evolution.evox_runner import (
    _should_post_restart_early_stop_phase2,
)


def test_config_defaults():
    assert cfg.PHASE2_PLATEAU_POST_RESTART_STOP_ENABLED is True
    assert cfg.PHASE2_PLATEAU_POST_RESTART_STOP_PATIENCE == 3
    assert cfg.PHASE2_ISLAND_PLATEAU_POST_RESTART_STOP_ENABLED is True
    assert cfg.PHASE2_ISLAND_PLATEAU_POST_RESTART_STOP_PATIENCE == 3
    assert cfg.PHASE2_ISLAND_PLATEAU_EARLY_STOP_PATIENCE == 6


def test_island_streak_below_patience_no_stop():
    assert not _should_post_restart_early_stop_phase2(
        2, island_profile="cluster_0",
    )


def test_island_streak_at_patience_stops():
    assert _should_post_restart_early_stop_phase2(
        3, island_profile="cluster_0",
    )


def test_island_disabled_no_stop(monkeypatch):
    monkeypatch.setattr(
        cfg, "PHASE2_ISLAND_PLATEAU_POST_RESTART_STOP_ENABLED", False,
    )
    assert not _should_post_restart_early_stop_phase2(
        10, island_profile="cluster_0",
    )


def test_global_uses_global_knobs(monkeypatch):
    monkeypatch.setattr(cfg, "PHASE2_PLATEAU_POST_RESTART_STOP_ENABLED", True)
    monkeypatch.setattr(cfg, "PHASE2_PLATEAU_POST_RESTART_STOP_PATIENCE", 2)
    assert not _should_post_restart_early_stop_phase2(1, island_profile="global")
    assert _should_post_restart_early_stop_phase2(2, island_profile="global")


def test_global_disabled_no_stop(monkeypatch):
    monkeypatch.setattr(cfg, "PHASE2_PLATEAU_POST_RESTART_STOP_ENABLED", False)
    assert not _should_post_restart_early_stop_phase2(
        10, island_profile="global",
    )


def test_orphan_uses_island_knobs():
    assert not _should_post_restart_early_stop_phase2(
        2, island_profile="orphan",
    )
    assert _should_post_restart_early_stop_phase2(
        3, island_profile="orphan",
    )
```

## EDIT 4 — tests/unit/test_island_early_stop.py
In `test_config_defaults`, change:
```python
    assert cfg.PHASE2_ISLAND_PLATEAU_EARLY_STOP_PATIENCE == 8
```
to:
```python
    assert cfg.PHASE2_ISLAND_PLATEAU_EARLY_STOP_PATIENCE == 6
```

---

## Acceptance criteria
1. This exact command passes (all green):
   ```
   PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest \
     tests/unit/test_phase2_post_restart_stop.py \
     tests/unit/test_island_early_stop.py \
     tests/unit/test_phase2_island_early_stop.py \
     tests/unit/test_phase2_plateau_restart.py \
     tests/unit/test_plateau_state_leak.py \
     tests/unit/test_evox_runner.py -q
   ```
2. Full unit suite passes (no regressions):
   ```
   PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/ -q
   ```
3. `cd /home/danaee/trading_platform && .venv/bin/python -c "import gpu_fuzzy_trader.evolution.evox_runner; import gpu_fuzzy_trader.config"` exits 0.
4. New config knobs have doc comments; no dead/obsolete code left behind.
5. Commit on `feature/task-1-post-restart-early-stop` with a clear message.

## Verification commands (run from repo root)
```
git checkout -b feature/task-1-post-restart-early-stop
# ...edits...
.venv/bin/python -c "import gpu_fuzzy_trader.evolution.evox_runner"
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_post_restart_stop.py tests/unit/test_island_early_stop.py tests/unit/test_phase2_island_early_stop.py tests/unit/test_phase2_plateau_restart.py tests/unit/test_plateau_state_leak.py tests/unit/test_evox_runner.py -q
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/ -q
git add -A && git commit -m "feat(phase2): post-restart no-improvement early stop"
```
