# Task 2 (A2) — Periodic Val Simulation

**Branch:** `feature/task-2-val-sim-interval` (from `main`)
**Skill:** implementer

## Goal
Val `simulate_rule_batch` runs every generation for every chromosome, but
`PHASE2_JOINT_TRAIN_VAL=False` + `PHASE2_VAL_IN_FITNESS_PENALTY=False` means
val metrics NEVER affect NSGA-III objectives. Val is only used for
deployable_archive tracking + pool admission. Run val every Nth gen (default 3)
instead of every gen → ~25-30% per-epoch wall-clock reduction. Val always runs
on the epoch's last gen for pool-admission freshness.

## Hard constraints (AGENTS.md)
- Use `.venv` for all commands; tests with `PYTEST_LOW_MEMORY=1` only.
- Do NOT run the pipeline. Do NOT touch `evaluator_v5.ipynb`.
- Remove dead/obsolete code after edits.

## Files
1. `gpu_fuzzy_trader/config.py` — 1 new knob.
2. `gpu_fuzzy_trader/evolution/evox_runner.py` — `_evaluate_population_indices`
   signature + both loops (nsga3 + nsga2_fallback).
3. `gpu_fuzzy_trader/phases/phase2_rule_pool.py` — `_evaluate_chromosome`
   signature (consistency; used by the single-chromosome fallback path).
4. `tests/unit/test_phase2_val_sim_interval.py` (NEW).

## EDIT 1 — config.py
Insert this knob near the other PHASE2_JOINT_TRAIN_VAL / PHASE2_VAL_* knobs
(search for `PHASE2_VAL_IN_FITNESS_PENALTY` and place it after that block):
```python
# PHASE2_VAL_SIM_INTERVAL — run val backtest every N generations during
# evolution (1 = every gen, original behaviour).  Only matters when
# PHASE2_JOINT_TRAIN_VAL=False (val doesn't affect objectives then); val
# metrics feed deployable_archive tracking + pool admission.  Val ALWAYS runs
# on the epoch's last gen regardless of this setting (pool-admission freshness).
#   1 → val every gen (original, 2x GPU work for no objective benefit).
#   3 → val every 3rd gen; deployable_archive refreshes every 3 gens (default).
PHASE2_VAL_SIM_INTERVAL = 3
```

## EDIT 2 — evox_runner.py `_evaluate_population_indices` (~line 1218)
Add a `run_val: bool = True` parameter to the signature (after `stage_params`):
```python
def _evaluate_population_indices(
    population: np.ndarray,
    indices: list[int],
    dont_cares: np.ndarray,
    engine,
    pareto_archive: list[np.ndarray],
    objectives: np.ndarray,
    metrics_cache: list[dict],
    val_engine=None,
    global_metrics_cache: dict[tuple[int, ...], dict] | None = None,
    diversity_reference: list[np.ndarray] | None = None,
    diversity_metrics_by_key: dict[tuple[int, ...], dict] | None = None,
    stage_params: Phase2StageParams | None = None,
    run_val: bool = True,
) -> dict[str, int]:
```
Change the val-sim guard (search for `val_metrics_list = None` / `if val_engine is not None:`):
```python
        val_metrics_list = None
        if val_engine is not None and run_val:
```
(Just add `and run_val` to the existing `if val_engine is not None:` condition.)

## EDIT 3 — phase2_rule_pool.py `_evaluate_chromosome` (~line 700)
Add the same `run_val: bool = True` parameter to `_evaluate_chromosome`'s
signature (after `cv_fold_evaluator`), and guard its val block identically:
```python
    val_metrics: dict | None = None
    if val_engine is not None and run_val:
```
(This is the single-chromosome fallback path used by `_reevaluate_infinite_objectives`;
keeping the param consistent avoids a signature mismatch. The TODO comment above
this block can stay or be updated to note the periodic-val implementation.)

## EDIT 4 — evox_runner.py BOTH loops: compute run_val_this_gen + pass to call sites

### `_run_nsga3` (~line 2046+)
At the TOP of the loop body, right after `just_restarted = False` (which was
added in the prior task), add:
```python
        is_last_gen = gen == n_generations - 1
        run_val_this_gen = is_last_gen or (
            gen % int(_cfg.PHASE2_VAL_SIM_INTERVAL) == 0
        )
```
NOTE: There is already an `is_last_gen = gen == n_generations - 1` line later in
the loop (~line 2331). Leave that later assignment in place (it just reassigns
the same value; removing it risks breaking the `if not is_last_gen:` /
`if is_last_gen: break` logic). Computing it early at the top is safe and lets
us use it for the eval calls.

Then pass `run_val=run_val_this_gen` to BOTH `_evaluate_population_indices`
call sites in nsga3:
- The **parent eval** call (~line 2190, `parent_stats = _evaluate_population_indices(...)`):
  add `run_val=run_val_this_gen,` to the kwargs.
- The **offspring eval** call (~line 2396, `off_stats = _evaluate_population_indices(...)`):
  add `run_val=run_val_this_gen,` to the kwargs.

Then GUARD the `_update_deployable_archive` call (~line 2212) so new deployables
are only admitted when val ran this gen (existing archive entries persist on
non-val gens; they just don't grow). Wrap it:
```python
        if run_val_this_gen:
            _update_deployable_archive(
                deployable_archive,
                population,
                list(range(pop_size)),
                metrics_cache,
            )
```
(Keep `_update_hall_of_fame(...)` and `_count_deployable_preview(...)` UN-guarded —
those don't need val.)

### `_run_nsga2_fallback` (~line 1590+)
Apply the SAME pattern. At the top of the loop body (after `just_restarted = False`),
add:
```python
        is_last_gen = gen == n_generations - 1
        run_val_this_gen = is_last_gen or (
            gen % int(_cfg.PHASE2_VAL_SIM_INTERVAL) == 0
        )
```
Pass `run_val=run_val_this_gen` to:
- The **initial-pop eval** loop (the `for i in range(pop_size): _evaluate_chromosome(...)`
  loop at ~line 1672): add `run_val=run_val_this_gen,` to the `_evaluate_chromosome` call.
- The **offspring eval** `_evaluate_population_indices` call (~line 1998): add
  `run_val=run_val_this_gen,` to kwargs.
Guard the `_update_deployable_archive` call (~line 1687) the same way as nsga3.

## EDIT 5 — Tests `tests/unit/test_phase2_val_sim_interval.py` (NEW)
Prove val is skipped on non-interval gens. Use a stub engine pair (train+val)
that records calls; run a 5-gen evolution with `PHASE2_VAL_SIM_INTERVAL=2`;
assert val called only on gens 0,2,4 (and last gen), train called every gen.
```python
"""Unit tests for periodic val simulation (Phase 2 runtime A2)."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from gpu_fuzzy_trader import config as cfg
from gpu_fuzzy_trader.evolution.evox_runner import run_phase2_evolution


class CountingEngine:
    def __init__(self):
        self.calls: list[int] = []

    def simulate_rule_batch(self, chromosomes, tp=None, sl=None, capital_pct=None):
        B = int(chromosomes.shape[0])
        self.calls.append(B)
        return [
            {
                "sortino_ratio": 1.0, "total_return_pct": 1.0,
                "max_drawdown_pct": 2.0, "win_rate": 50.0, "executed_trades": 50,
            }
            for _ in range(B)
        ]


def test_val_skipped_on_non_interval_gens(monkeypatch):
    """With PHASE2_VAL_SIM_INTERVAL=2, val sim runs only on even gens (0,2,4)
    + last gen; train sim runs every gen."""
    train_engine = CountingEngine()
    val_engine = CountingEngine()

    monkeypatch.setattr(cfg, "PHASE2_VAL_SIM_INTERVAL", 2)
    monkeypatch.setattr(cfg, "PHASE2_EARLY_STOP_ENABLED", False)
    monkeypatch.setattr(cfg, "PHASE2_EARLY_STOP_MIN_GENERATION", 999)
    monkeypatch.setattr(cfg, "PHASE2_PLATEAU_EARLY_STOP_ENABLED", False)
    monkeypatch.setattr(cfg, "PHASE2_PLATEAU_POST_RESTART_STOP_ENABLED", False)
    monkeypatch.setattr(cfg, "PHASE2_DIVERSITY_RECOVERY_ENABLED", False)

    with mock.patch("gpu_fuzzy_trader.evolution.evox_runner._EVOX_AVAILABLE", False):
        rng = np.random.default_rng(0)
        feature_infos = [
            {"name": "f0", "mode": "binary", "score": 0.5},
            {"name": "f1", "mode": "binary", "score": 0.5},
        ]
        run_phase2_evolution(
            feature_infos=feature_infos,
            engine=train_engine,
            val_engine=val_engine,
            pop_size=10,
            n_generations=5,
            rng=rng,
        )

    # Train should run every gen (>=5 calls, one per gen minimum).
    assert len(train_engine.calls) >= 5, (
        f"Train should run every gen; got {len(train_engine.calls)} calls"
    )
    # Val should run LESS often than train (skipped on odd gens).
    assert len(val_engine.calls) < len(train_engine.calls), (
        f"Val should be skipped on non-interval gens; "
        f"val={len(val_engine.calls)} train={len(train_engine.calls)}"
    )
    # Val must run at least on gen 0 (interval) and last gen (forced).
    assert len(val_engine.calls) >= 2, (
        f"Val should run on gen 0 and last gen at minimum; got {len(val_engine.calls)}"
    )


def test_val_runs_every_gen_when_interval_1(monkeypatch):
    """PHASE2_VAL_SIM_INTERVAL=1 preserves original behaviour (val every gen)."""
    train_engine = CountingEngine()
    val_engine = CountingEngine()

    monkeypatch.setattr(cfg, "PHASE2_VAL_SIM_INTERVAL", 1)
    monkeypatch.setattr(cfg, "PHASE2_EARLY_STOP_ENABLED", False)
    monkeypatch.setattr(cfg, "PHASE2_EARLY_STOP_MIN_GENERATION", 999)
    monkeypatch.setattr(cfg, "PHASE2_PLATEAU_EARLY_STOP_ENABLED", False)
    monkeypatch.setattr(cfg, "PHASE2_PLATEAU_POST_RESTART_STOP_ENABLED", False)
    monkeypatch.setattr(cfg, "PHASE2_DIVERSITY_RECOVERY_ENABLED", False)

    with mock.patch("gpu_fuzzy_trader.evolution.evox_runner._EVOX_AVAILABLE", False):
        rng = np.random.default_rng(0)
        feature_infos = [
            {"name": "f0", "mode": "binary", "score": 0.5},
        ]
        run_phase2_evolution(
            feature_infos=feature_infos,
            engine=train_engine,
            val_engine=val_engine,
            pop_size=10,
            n_generations=3,
            rng=rng,
        )

    assert len(val_engine.calls) > 0, "Val should run when interval=1"
    # With interval=1, val count should be close to train count (every gen).
    assert len(val_engine.calls) >= len(train_engine.calls) - 1, (
        f"interval=1 should run val every gen; val={len(val_engine.calls)} "
        f"train={len(train_engine.calls)}"
    )
```
(Adjust assertions if dedup/last-gen-skip changes exact counts — the key
invariant is val_count < train_count when interval>1, and val runs on gen 0 + last gen.)

## Verification (run from repo root)
1. `.venv/bin/python -c "import gpu_fuzzy_trader.evolution.evox_runner; import gpu_fuzzy_trader.phases.phase2_rule_pool"`
2. `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_val_sim_interval.py tests/unit/test_phase2_offspring_batch.py tests/unit/test_evox_runner.py tests/unit/test_phase2_plateau_restart.py tests/unit/test_phase2_post_restart_stop.py tests/unit/test_plateau_state_leak.py tests/unit/test_island_early_stop.py tests/unit/test_phase2_island_early_stop.py -q`
3. `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/ -q`

## Acceptance criteria
1. New tests pass; val sim count < train sim count when interval>1.
2. All targeted + full unit tests pass (no regressions).
3. Import check exits 0.
4. No dead code; old `if val_engine is not None:` guards updated to include `and run_val`.
5. Single commit on `feature/task-2-val-sim-interval`: `perf(phase2): periodic val simulation`.
