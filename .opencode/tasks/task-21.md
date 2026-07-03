# Task-21: Island RNG State Leakage & Generation-Budget Realization

**Branch:** `fix/island-rng-and-budget`
**Priority:** 🟠 High
**Depends on:** none (safe to run in parallel with task-20)

## Problem

Two related issues in how `phase2_island_scheduler.py` and
`Rule_Pool_Generator` (`phase2_rule_pool.py`) manage per-island state across
repeated `run_epoch()` calls.

### S1: RNG never advances across resumed epochs (confirmed)

`Rule_Pool_Generator.seed` is assigned once in `__init__` and never mutated
(`phase2_rule_pool.py:1972`: `self.seed = seed  # preserved as-is`). Every
call to `run_epoch()` (`phase2_rule_pool.py:2846`), `run()` (`:2239`), and
`finalize_island()` (`:2888`) does:

```python
rng = np.random.default_rng(self.seed)
```

i.e. **a fresh RNG seeded identically every time**, regardless of how many
epochs have already run for this island. `self._evolution_state` (population,
archives) *does* persist correctly across epochs (confirmed — this part is
fine), but the pseudo-random stream driving mutation-site selection,
crossover points, and tournament draws inside that epoch **restarts from
scratch** each call instead of continuing. Any island that needs more than
one epoch to exhaust its `gens_per_cluster` budget (the common case — see S3
below) will replay the identical draw sequence in epoch 2 that it already
consumed in epoch 1, coupling supposedly-independent generations.

### S2: All islands share one identical seed (confirmed)

`_run_cluster_islands` (`phase2_island_scheduler.py:317-359`) resolves a
single `seed` once in `run_cluster_phase2` and passes the *same* value into
**every** cluster's `Rule_Pool_Generator(..., seed=seed, ...)`
(`phase2_island_scheduler.py:352`). Combined with S1, cluster_0, cluster_1,
and cluster_2 all draw from identically-seeded RNGs — their stochastic
choices (which gene mutates, which parents get picked) are correlated across
islands instead of being independent stochastic restarts, partially
undermining the diversification rationale for running islands at all.

### S3: Per-cluster generation budget is rarely realized in one epoch (confirmed, not itself a bug but interacts with S1)

`_run_cluster_islands` computes `gens_per_cluster = total_gens // n_clusters`
(e.g. 100 // 3 = 33) and calls `gen.run_epoch(n_generations=min(epoch_gens,
remaining))` in a round-robin `while` loop. `run_epoch` charges
`self._island_generations_done += len(epoch_history)` — the *actual* gens run,
which is almost always cut short by the plateau/post-restart early-stop
cascade (island-scoped patience = 6 to trigger restart, 5 more to
post-restart-stop, per `config.py:1023,772`) well before the requested 25.
In the sampled log, cluster islands stopped at gen 12 and gen 16 of a 25-gen
epoch request — meaning **every cluster needs a second (and possibly third)
`run_epoch()` call** to consume its 33-gen budget, so S1's RNG-reset bug is
not a rare edge case, it fires on effectively every island, every run.

## Files to Modify

1. `gpu_fuzzy_trader/phases/phase2_rule_pool.py` — persist an advancing RNG on `Rule_Pool_Generator` instead of recreating from `self.seed` every call; derive a per-island seed.
2. `gpu_fuzzy_trader/phases/phase2_island_scheduler.py` — pass a per-cluster derived seed instead of the shared raw seed.

## Detailed Changes

### Fix S1: persist one RNG instance per generator across epochs

```python
# __init__:
self.seed = seed
self._rng: np.random.Generator | None = None

def _get_rng(self) -> np.random.Generator:
    if self._rng is None:
        self._rng = np.random.default_rng(self.seed)
    return self._rng
```

Replace `rng = np.random.default_rng(self.seed)` at `run_epoch` (`:2846`),
`run()` (`:2239`), and `finalize_island()` (`:2888`) with `rng =
self._get_rng()`. This keeps determinism (same `self.seed` still fully
determines the whole island's trajectory across epochs given a fixed call
sequence) while making the "random" stream actually advance instead of
resetting.

### Fix S2: derive an independent seed per cluster

In `_run_cluster_islands` (`phase2_island_scheduler.py:335-359`), replace the
shared `seed` with a per-cluster derivation, e.g.:

```python
for cid in cluster_ids:
    ...
    cluster_seed = None if seed is None else int(seed) + hash(cid) % 100_000
    # or, simpler and still reproducible:
    cluster_seed = None if seed is None else int(seed) * 1000 + int(cid)
    generators[cid] = Rule_Pool_Generator(
        ...,
        seed=cluster_seed,
        ...
    )
```

Keep `_sample_df`'s deterministic chronological sampling untouched (it
ignores `random_state` by design per `task-19`/L5) — this only affects the
evolutionary RNG stream, not data sampling.

### Also verify: no cross-island cache sharing

Confirmed clean during this audit: `global_metrics_cache`,
`hall_of_fame`, `pareto_archive`, and `deployable_archive` all live inside
`Phase2EvolutionState`, which is a **per-`Rule_Pool_Generator`-instance**
attribute (`self._evolution_state`), and each cluster gets its own generator
instance in `_run_cluster_islands`'s `generators: dict[str,
Rule_Pool_Generator]`. No code change needed here — this task should add a
regression test asserting two generators never share the same
`Phase2EvolutionState`/cache object identity, to guard against future
refactors accidentally hoisting this to module/class level.

## Acceptance Criteria

- [ ] `Rule_Pool_Generator` keeps one `np.random.Generator` instance alive across `run_epoch`/`run`/`finalize_island` calls (no `np.random.default_rng(self.seed)` re-created mid-island-lifetime).
- [ ] Each cluster island in `_run_cluster_islands` receives a distinct derived seed, not the raw shared `seed`.
- [ ] New unit test: instantiate two `Rule_Pool_Generator`s with the same base seed via the cluster-derivation helper, assert their first-epoch chromosome draws differ.
- [ ] New unit test: call `run_epoch()` twice on the same generator (small pop/gens fixture) and assert the second call's RNG state is NOT a reset replay of the first (e.g. compare consumed random draws or resulting mutation patterns are not identical when population differs).
- [ ] New regression test: assert `id(gen_a._evolution_state) != id(gen_b._evolution_state)` and cache dict identities differ across two cluster generators.
- [ ] Existing tests pass: `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_island_scheduler.py tests/unit/test_phase2_island_hyperparams.py tests/unit/test_island_scheduler_migration.py -x -q`

## Verification

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/ -x -q -k "island or rule_pool_generator or seed"
```
