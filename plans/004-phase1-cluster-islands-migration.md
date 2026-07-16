# Plan 004: Re-enable Phase1 top-K and multi-symbol islands with migration

> **Executor instructions**: Follow step by step. Run every verification command before proceeding. On STOP conditions, stop and report. Do **not** update `plans/README.md` (reviewer maintains index).
>
> **Drift check (run first)**: `git diff --stat 425f469..HEAD -- gpu_fuzzy_trader/config.py tests/unit/test_anti_overfit_config.py tests/unit/test_feature_selector.py tests/unit/test_phase2_island_scheduler.py tests/unit/test_migration_safety.py tests/unit/test_island_scheduler_migration.py`
> Mismatch with "Current state" excerpts → STOP.
>
> **Prerequisite**: Plan 003 should be DONE (or landed on the same branch before this). Fitness without val steering + multi-symbol islands still overfits train windows.

## Status

- **Priority**: P2
- **Effort**: M
- **Risk**: MED
- **Depends on**: plans/003-phase2-val-fitness-penalty.md
- **Category**: direction (search architecture)
- **Planned at**: commit `425f469`, 2026-07-16

## Why this matters

The 2026-07-13 Colab run used `PHASE1_DISABLED=True` (all ~27 features), `PHASE2_ONE_SYMBOL_ISLANDS=True` (K≈10 specialist islands), and `PHASE2_MIGRATION_ENABLED=False`, with a short budget (pop=60, gens=20). That produces symbol-specialist rules that then fail RB concentration (`RB_MAX_SYMBOL_SHARE_ABS_PNL=0.50`) and yield thin multi-symbol teams.

Re-enable MI top-K Phase1, switch to **hybrid multi-symbol clusters** (`PHASE2_N_CLUSTERS=3`, corr blend already configured), and turn **migration on** so elites transfer across islands. Keep pop/gens unchanged in this plan (memory-safe); widen budget only after Colab proves need.

## Current state

[`gpu_fuzzy_trader/config.py`](gpu_fuzzy_trader/config.py):

```python
PHASE1_TOP_K_FEATURES = 20
PHASE1_DISABLED: bool = True          # ~323
PHASE2_ISLAND_MODE = "cluster"        # ~1224
PHASE2_ONE_SYMBOL_ISLANDS = True      # ~1229
PHASE2_N_CLUSTERS = 3                 # ~1232
PHASE2_CLUSTER_USE_RETURN_CORR = True
PHASE2_CLUSTER_FEATURE_WEIGHT = 0.3
PHASE2_CLUSTER_CORR_WEIGHT = 0.7
PHASE2_MIGRATION_ENABLED: bool = False  # ~1270
```

Bundle lock [`tests/unit/test_anti_overfit_config.py`](tests/unit/test_anti_overfit_config.py):

- lines 33, 51–52 assert Phase1 disabled, one-symbol True, migration False.

Related tests (monkeypatch both modes — update only **default** assertions):

- `tests/unit/test_feature_selector.py`
- `tests/unit/test_phase2_island_scheduler.py`
- `tests/unit/test_migration_safety.py`
- `tests/unit/test_island_scheduler_migration.py`

`resolve_island_hyperparams`: one-symbol path forces `min_profitable_symbols == 1`; cluster multi-symbol keeps robustness (`test_cluster_island_symbol_robustness_enabled`).

**AGENTS.md**: `.venv`; `PYTEST_LOW_MEMORY=1`; related tests only; no full pipeline on WSL; do not modify `evaluator_v5.ipynb`.

## Commands you will need

| Purpose                      | Command                                                                                                                                                                                                                                                                                          | Expected on success |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------- |
| Defaults                     | `.venv/bin/python -c "from gpu_fuzzy_trader import config as c; assert c.PHASE1_DISABLED is False; assert c.PHASE2_ONE_SYMBOL_ISLANDS is False; assert c.PHASE2_MIGRATION_ENABLED is True; assert c.PHASE2_N_CLUSTERS == 3; assert c.PHASE2_ISLAND_MODE == 'cluster'"`                           | exit 0              |
| Bundle                       | `PYTEST_LOW_MEMORY=1 .venv/bin/pytest tests/unit/test_anti_overfit_config.py -q`                                                                                                                                                                                                                 | pass                |
| Phase1 / islands / migration | `PYTEST_LOW_MEMORY=1 .venv/bin/pytest tests/unit/test_feature_selector.py tests/unit/test_phase2_island_scheduler.py tests/unit/test_phase2_island_hyperparams.py tests/unit/test_migration_safety.py tests/unit/test_island_scheduler_migration.py tests/unit/test_phase2_migration_gate.py -q` | all pass            |

## Suggested executor toolkit

- Invoke **codelookup** on `PHASE1_DISABLED`, `PHASE2_ONE_SYMBOL_ISLANDS`, `PHASE2_MIGRATION_ENABLED`, `resolve_island_hyperparams`, Feature_Selector.run.
- Cascade-update default assertions; leave monkeypatched scenario tests intact.

## Scope

**In scope**:

- `gpu_fuzzy_trader/config.py` defaults + comments for Phase1 disabled flag, one-symbol islands, migration
- `tests/unit/test_anti_overfit_config.py` and any other **default** assertions
- Comment updates explaining alignment with RB concentration gates

**Out of scope**:

- Changing `PHASE2_POPULATION_SIZE` / `PHASE2_GENERATIONS`
- Changing `PHASE2_N_CLUSTERS` away from 3
- Re-tuning RB concentration thresholds
- JOINT_TRAIN_VAL
- Dataset rename / RB fail-closed (001/002)
- Implementing new clustering algorithms
- `evaluator_v5.ipynb`

## Git workflow

- Branch: `advisor/004-phase1-cluster-islands-migration`
- Commit: `feat: enable Phase1 top-K and multi-symbol island migration`
- Do NOT push or open PR unless instructed

## Steps

### Step 1: Confirm 003 landed

```bash
.venv/bin/python -c "from gpu_fuzzy_trader import config as c; assert c.PHASE2_VAL_IN_FITNESS_PENALTY is True"
```

If False → STOP (dependency). If working on a combined branch, land 003 commits first.

### Step 2: Flip architecture defaults

In `config.py` set exactly:

| Knob                        | New value |
| --------------------------- | --------- |
| `PHASE1_DISABLED`           | `False`   |
| `PHASE2_ONE_SYMBOL_ISLANDS` | `False`   |
| `PHASE2_MIGRATION_ENABLED`  | `True`    |

Leave unchanged:

- `PHASE1_TOP_K_FEATURES = 20`
- `PHASE2_ISLAND_MODE = "cluster"`
- `PHASE2_N_CLUSTERS = 3`
- corr blend weights
- pop/gens / island epoch sizes

Update comments:

- Remove or rewrite “2026-07-11b: True — user found multi-symbol clustering harmful” to note: one-symbol islands conflict with RB concentration; multi-symbol clusters + migration restored 2026-07-16 (plan 004).
- Phase1: disabled bypass enlarges gene space; top-K=20 is the intended Colab budget.

**Verify**: defaults assert command in table above.

### Step 3: Update anti-overfit bundle + default tests

Update `test_anti_overfit_config.py`:

- `PHASE1_DISABLED is False`
- `PHASE2_ONE_SYMBOL_ISLANDS is False`
- `PHASE2_MIGRATION_ENABLED is True`

Fix any other test that asserts these three defaults (e.g. `test_migration_safety.py` AC that documents defaults — update expectations to new defaults while keeping behavior tests that monkeypatch False).

**Verify**: pytest command in table above.

### Step 4: Sanity-check hyperparams path

```bash
.venv/bin/python - <<'PY'
from gpu_fuzzy_trader import config as c
hp = c.resolve_island_hyperparams("cluster", n_rows=175_000, reference_rows=700_000, n_symbols=4)
assert hp.skip_symbol_robustness_penalty is False
assert hp.min_profitable_symbols >= 2
print("ok", hp.min_profitable_symbols)
PY
```

**Verify**: prints `ok` with min_profitable_symbols >= 2.

## Test plan

- Unit tests listed above only.
- Operator Colab check (not required for DONE): Phase1 should log top-K≈20 not “all features”; Phase2 should show fewer than ~10 one-symbol islands (expect 3 clusters); migration log lines when enabled; RB concentration failures should decline vs 2026-07-13.

## Done criteria

- [ ] Defaults: Phase1 enabled, one-symbol False, migration True, N_CLUSTERS=3, island mode cluster
- [ ] Comments updated; pop/gens unchanged
- [ ] Bundle + island/migration/feature_selector related tests pass with `PYTEST_LOW_MEMORY=1`
- [ ] Prerequisite 003 val-in-fitness still True
- [ ] Codelookup blast radius addressed
- [ ] No out-of-scope files modified

## STOP conditions

- Plan 003 not applied.
- Enabling migration fails unit tests due to broken migrant seeding — fix within migration modules only if required for defaults; if large redesign needed, STOP and report.
- Drift shows one-symbol already False — reconcile rather than double-apply.
- Pressure to raise pop/gens in this PR — defer; do not expand scope.

## Maintenance notes

- Historical note: one-symbol islands were chosen 2026-07-11b after clustering looked harmful; that interacted badly with RB concentration. If Colab regresses with clusters, capture metrics before reverting — prefer tightening migration admission gates over returning to one-symbol.
- After 002+003+004, empty fail-closed should fire less often; when it fires, OOS stays clean.
