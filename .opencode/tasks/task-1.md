# Task 1 (A1) — Batch Offspring Evaluation

**Branch:** `feature/task-1-offspring-batch` (from `main`)
**Skill:** implementer

## Goal
Replace the per-chromosome offspring evaluation loop (200 individual GPU
dispatches/gen) with a single batched call, reusing the existing
`_evaluate_population_indices` helper. Expected 5-10x per-gen speedup.

## Hard constraints (AGENTS.md)
- Use `.venv` for all commands (`.venv/bin/python`).
- Run tests ONLY with `PYTEST_LOW_MEMORY=1` (OOM risk).
- Do NOT run the pipeline. Do NOT touch `evaluator_v5.ipynb`.
- Remove dead/obsolete code after edits.

## Files
1. `gpu_fuzzy_trader/evolution/evox_runner.py` — BOTH generation loops.
2. `tests/unit/test_phase2_offspring_batch.py` (NEW).

## EDIT 1 — `_run_nsga3` offspring eval (~line 1996)

Find this exact block inside `_run_nsga3` (after `_make_offspring_population(...)`):
```python
        off_obj = np.full((pop_size, 3), np.inf)
        off_metrics: list[dict] = [{} for _ in range(pop_size)]
        for i in range(pop_size):
            obj, metrics = _evaluate_chromosome(
                offspring[i], dont_cares, engine, pareto_archive,
                val_engine=val_engine,
                stage_params=stage_params,
                cv_fold_evaluator=cv_fold_evaluator,
            )
            off_obj[i] = obj
            off_metrics[i] = metrics
```

Replace with:
```python
        off_obj = np.full((pop_size, 3), np.inf)
        off_metrics: list[dict] = [{} for _ in range(pop_size)]
        # Batched offspring eval: a single simulate_rule_batch over all 200
        # offspring (vs 200 batch=1 dispatches). Reuses the same helper as the
        # initial-population eval path. NOTE: this path does not compute CV
        # fold returns; with PHASE2_F3_OBJECTIVE="profit_factor" (active
        # config) this matches the existing initial-pop behaviour. cv_fold_min
        # would need separate batched CV handling (pre-existing limitation).
        _evaluate_population_indices(
            offspring,
            list(range(pop_size)),
            dont_cares,
            engine,
            pareto_archive,
            off_obj,
            off_metrics,
            val_engine=val_engine,
            global_metrics_cache=global_metrics_cache,
            diversity_reference=diversity_reference,
            diversity_metrics_by_key=diversity_metrics_by_key,
            stage_params=stage_params,
        )
```

All referenced vars (`global_metrics_cache`, `diversity_reference`,
`diversity_metrics_by_key`) are in scope in `_run_nsga3` — they are set up
before the `for gen in range(n_generations):` loop begins. Verify by grep if
unsure; do NOT add new variable declarations.

## EDIT 2 — `_run_nsga2_fallback` offspring eval (~line 1678)

Find the SAME per-chromosome loop pattern inside `_run_nsga2_fallback`
(after `_make_offspring_population(...)`):
```python
        off_obj = np.full((pop_size, 3), np.inf)
        off_metrics: list[dict] = [{} for _ in range(pop_size)]
        for i in range(pop_size):
            obj, metrics = _evaluate_chromosome(
                offspring[i], dont_cares, engine, pareto_archive,
                val_engine=val_engine,
                stage_params=stage_params,
                cv_fold_evaluator=cv_fold_evaluator,
            )
            off_obj[i] = obj
            off_metrics[i] = metrics
```

Replace with the same `_evaluate_population_indices` call. IMPORTANT: the
fallback loop may NOT have `global_metrics_cache`, `diversity_reference`, or
`diversity_metrics_by_key` in scope. CHECK first (grep the function body). If
any are absent, OMIT those kwargs (they default to `None` in
`_evaluate_population_indices`). Keep `val_engine=val_engine` and
`stage_params=stage_params` (these are in scope). For example, if
`global_metrics_cache` is not in scope, the call becomes:
```python
        _evaluate_population_indices(
            offspring,
            list(range(pop_size)),
            dont_cares,
            engine,
            pareto_archive,
            off_obj,
            off_metrics,
            val_engine=val_engine,
            stage_params=stage_params,
        )
```
(Only include the kwargs for vars that ARE in scope in that function.)

## EDIT 3 — Tests `tests/unit/test_phase2_offspring_batch.py` (NEW)

Create a test that proves offspring are evaluated in a SINGLE batch call per
generation (train), not pop_size individual calls. Use the fallback path
(`_EVOX_AVAILABLE=False`) with a stub engine that records call count + batch
sizes.

```python
"""Unit tests for batched offspring evaluation (Phase 2 runtime A1)."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from gpu_fuzzy_trader import config as cfg
from gpu_fuzzy_trader.evolution.evox_runner import run_phase2_evolution
from gpu_fuzzy_trader.phases.phase2_rule_pool import _init_population


class CountingEngine:
    """Stub engine recording every simulate_rule_batch call's batch size."""

    def __init__(self):
        self.train_calls: list[int] = []

    def simulate_rule_batch(self, chromosomes, tp=None, sl=None, capital_pct=None):
        B = int(chromosomes.shape[0])
        self.train_calls.append(B)
        return [
            {
                "sortino_ratio": 1.0,
                "total_return_pct": 1.0,
                "max_drawdown_pct": 2.0,
                "win_rate": 50.0,
                "executed_trades": 50,
            }
            for _ in range(B)
        ]


def test_offspring_evaluated_in_single_batch_per_gen():
    """Offspring should be evaluated via ONE simulate_rule_batch call per gen,
    not pop_size individual calls."""
    engine = CountingEngine()

    with mock.patch(
        "gpu_fuzzy_trader.evolution.evox_runner._EVOX_AVAILABLE", False,
    ), mock.patch.object(cfg, "PHASE2_EARLY_STOP_ENABLED", False), \
         mock.patch.object(cfg, "PHASE2_EARLY_STOP_MIN_GENERATION", 999), \
         mock.patch.object(cfg, "PHASE2_PLATEAU_EARLY_STOP_ENABLED", False), \
         mock.patch.object(cfg, "PHASE2_PLATEAU_POST_RESTART_STOP_ENABLED", False), \
         mock.patch.object(cfg, "PHASE2_DIVERSITY_RECOVERY_ENABLED", False):
        rng = np.random.default_rng(0)
        feature_infos = [
            {"name": "f0", "mode": "binary", "score": 0.5},
            {"name": "f1", "mode": "binary", "score": 0.5},
        ]
        run_phase2_evolution(
            feature_infos=feature_infos,
            engine=engine,
            pop_size=10,
            n_generations=3,
            rng=rng,
        )

    # Each generation should produce exactly ONE train batch call for offspring.
    # (The initial population also uses one batch call at gen 0.)
    # Assert that no batch of size 1 was used for offspring — all offspring
    # batches should be size pop_size (10).
    batch_sizes = engine.train_calls
    # There must be at least one batch of size == pop_size (the offspring batch).
    assert pop_size_batch := any(
        b == 10 for b in batch_sizes
    ), f"Expected a batch of size 10 (offspring), got sizes {batch_sizes}"
    # And there must NOT be 10 separate calls of size 1 (the old behaviour).
    size_1_calls = sum(1 for b in batch_sizes if b == 1)
    # Allow at most a couple of size-1 fallback calls (error paths), but not 10/gen.
    assert size_1_calls < 10, (
        f"Too many size-1 batch calls ({size_1_calls}); offspring not batched. "
        f"Sizes: {batch_sizes}"
    )
```

(Adjust the assertion logic if the stub needs `val_engine` handling — pass
`val_engine=None` to `run_phase2_evolution` so no val calls occur, keeping
the test focused on train batches.)

## Verification (run from repo root)
1. `.venv/bin/python -c "import gpu_fuzzy_trader.evolution.evox_runner"`
2. `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_offspring_batch.py tests/unit/test_evox_runner.py tests/unit/test_phase2_plateau_restart.py tests/unit/test_phase2_post_restart_stop.py tests/unit/test_plateau_state_leak.py tests/unit/test_island_early_stop.py tests/unit/test_phase2_island_early_stop.py -q`
3. `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/ -q` (full regression)

## Acceptance criteria
1. New test passes; offspring are evaluated in batch (no 200 size-1 calls).
2. All targeted + full unit tests pass (no regressions).
3. Import check exits 0.
4. Old per-chromosome offspring loop fully REMOVED from both functions (not
   commented out, not left as fallback).
5. Single commit on `feature/task-1-offspring-batch`:
   `perf(phase2): batch offspring evaluation`.
