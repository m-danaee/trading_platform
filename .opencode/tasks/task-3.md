# Task 3 (A3) — Fix Island Patience Dead-Code Bug

**Branch:** `feature/task-3-island-patience-fix` (from `main`)
**Skill:** implementer

## Goal
`_should_plateau_early_stop_phase2` for island profiles reads
`stage_params.plateau_early_stop_patience` (=8, baked into the None-stage
profile) instead of `PHASE2_ISLAND_PLATEAU_EARLY_STOP_PATIENCE` (=6). The
island knob is unreachable — confirmed by the run log showing restarts at
streak 8, not 6. Fix: for island profiles, always use the island-scoped config
knob directly.

## Hard constraints (AGENTS.md)
- Use `.venv` for all commands; tests with `PYTEST_LOW_MEMORY=1` only.
- Do NOT run the pipeline. Do NOT touch `evaluator_v5.ipynb`.
- Remove dead/obsolete code after edits.

## Files
1. `gpu_fuzzy_trader/evolution/evox_runner.py` — `_should_plateau_early_stop_phase2`.
2. `tests/unit/test_phase2_island_early_stop.py` — add regression test.

## EDIT 1 — evox_runner.py `_should_plateau_early_stop_phase2` (~line 588)

Find this block (the patience resolution at the end of the function):
```python
    if _cfg.scoped_island_profile(island_profile):
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

Replace with:
```python
    if _cfg.scoped_island_profile(island_profile):
        # Islands run single-stage (stage=None); the stage_params patience is
        # the GLOBAL default baked into the None profile, NOT the island knob.
        # Use the island-scoped config directly so
        # PHASE2_ISLAND_PLATEAU_EARLY_STOP_PATIENCE actually takes effect.
        patience = int(getattr(
            _cfg, "PHASE2_ISLAND_PLATEAU_EARLY_STOP_PATIENCE",
            _cfg.PHASE2_PLATEAU_EARLY_STOP_PATIENCE,
        ))
    else:
        patience = (
            int(stage_params.plateau_early_stop_patience)
            if stage_params is not None
            else int(_cfg.PHASE2_PLATEAU_EARLY_STOP_PATIENCE)
        )
```

## EDIT 2 — test_phase2_island_early_stop.py — add regression test

Add this test to `tests/unit/test_phase2_island_early_stop.py`:
```python
def test_island_patience_uses_island_knob_not_stage_params(monkeypatch):
    """Regression: island patience must come from
    PHASE2_ISLAND_PLATEAU_EARLY_STOP_PATIENCE, NOT stage_params.

    Islands run single-stage (stage=None) → resolve_phase2_stage_params(None)
    bakes in PHASE2_PLATEAU_EARLY_STOP_PATIENCE=8 into stage_params. Before the
    fix, stage_params.plateau_early_stop_patience (=8) was read instead of the
    island knob (=6), making the island knob dead code.
    """
    from gpu_fuzzy_trader.phases.phase2_stage import resolve_phase2_stage_params

    monkeypatch.setattr(cfg, "PHASE2_ISLAND_PLATEAU_EARLY_STOP_ENABLED", True)
    monkeypatch.setattr(cfg, "PHASE2_ISLAND_PLATEAU_BLOCK_WHEN_DEPLOYABLE_ZERO", False)
    monkeypatch.setattr(cfg, "PHASE2_ISLAND_PLATEAU_EARLY_STOP_PATIENCE", 6)
    monkeypatch.setattr(cfg, "PHASE2_PLATEAU_EARLY_STOP_PATIENCE", 8)

    # stage_params has patience=8 (the GLOBAL default baked into None profile).
    stage_params = resolve_phase2_stage_params(None)
    assert stage_params.plateau_early_stop_patience == 8, (
        "Test precondition: stage_params patience should be the global 8"
    )

    # Island profile: streak=6 must trigger stop (island patience=6 wins).
    assert _should_plateau_early_stop_phase2(
        9, 6, deployable_count=5, island_profile="cluster_0",
        stage_params=stage_params,
    ), "streak=6 should stop with island patience=6 (not wait for 8)"

    # And streak=5 must NOT trigger (below island patience=6).
    assert not _should_plateau_early_stop_phase2(
        9, 5, deployable_count=5, island_profile="cluster_0",
        stage_params=stage_params,
    ), "streak=5 should not stop with island patience=6"
```

## Verification (run from repo root)
1. `.venv/bin/python -c "import gpu_fuzzy_trader.evolution.evox_runner"`
2. `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_island_early_stop.py tests/unit/test_island_early_stop.py tests/unit/test_phase2_plateau_restart.py tests/unit/test_phase2_post_restart_stop.py tests/unit/test_evox_runner.py tests/unit/test_phase2_val_sim_interval.py tests/unit/test_phase2_offspring_batch.py tests/unit/test_plateau_state_leak.py -q`
3. `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/ -q`

## Acceptance criteria
1. New regression test passes (island patience=6 wins over stage_params=8).
2. All existing tests pass (no regressions — the existing `test_island_patience_respected_at_8` and `test_global_patience_still_5` should still pass since they monkeypatch the knobs directly).
3. Import check exits 0.
4. Single commit on `feature/task-3-island-patience-fix`: `fix(phase2): use island-scoped plateau patience instead of stage_params`.
