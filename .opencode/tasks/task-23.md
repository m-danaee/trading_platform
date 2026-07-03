# Task-23: Config/Logging Anomaly Cleanup

**Branch:** `fix/config-logging-anomalies`
**Priority:** 🟢 Low (cosmetic/consistency, but cheap and removes confusion during future debugging)
**Depends on:** none

## Problem

Smaller inconsistencies found while cross-referencing `config.py` against the
actual runtime parameters used by the evolutionary loop and pipeline banner.

### A1: Startup banner always claims legacy Phase 3/4, even when RB Governor runs it

`run_pipeline.py::_log_pipeline_config` (`:246-253`) hardcodes:
```python
logger.info(
    "Pipeline config: PHASE1 top_k=%d | " + phase2_fmt +
    " | PHASE3 per-symbol greedy | PHASE4 grid_search=True | %s",
    ...
)
```
regardless of `RB_GOVERNOR_ENABLED` (`config.py:1542`, default `True`, and the
active path for this project). Every run log — including the one just
analyzed — prints a startup banner describing a code path (legacy per-symbol
greedy Phase 3 + grid-search Phase 4) that isn't actually what runs. This is
the exact kind of "configuration mismatch between the global pipeline config
and actual parameters" that's easy to mis-diagnose from a log alone.

### A2: `IslandHyperparams.skip_symbol_robustness_penalty` is hardcoded `True` unconditionally

`resolve_island_hyperparams` (`config.py:2052-2110`) returns
`skip_symbol_robustness_penalty=True` for **both** `"cluster"` and
`"orphan"` profiles — there is no branch that ever produces `False`. Since
`PHASE2_ISLAND_MODE = "cluster"` is the only active mode, this permanently
disables `_symbol_robustness_penalty()` (`phase2_rule_pool.py:584-589`)
project-wide, leaving only the separate C5 inline penalty
(`phase2_rule_pool.py:574-582`) active. This may be intentional (avoiding a
penalty tuned for a 10-symbol universe being applied to 3-4-symbol islands),
but as written it reads like a "resolved" per-profile parameter when it is
actually a dead constant — confusing for future maintenance.

### A3: `gens_per_cluster` silently drops remainder generations

`_run_cluster_islands` / `_log_pipeline_config` both compute
`gens_per_cluster = total_gens // n_clusters` (integer floor). With
`PHASE2_ISLAND_TOTAL_GENERATIONS=100` and `PHASE2_N_CLUSTERS=3`, this yields
33/cluster x 3 = 99, silently discarding 1 generation of the declared 100-gen
budget. Trivial in isolation, but worth a one-line fix for precision (e.g.
distribute the remainder to the first `total_gens % n_clusters` clusters).

### A4: `min_profitable_symbols` island formula is proportionally stricter for odd/small clusters

`resolve_island_hyperparams` (`config.py:2086-2089`):
```python
min_profitable = min(
    int(PHASE2_MIN_PROFITABLE_SYMBOLS),   # 5 (of 10 symbols = 50%)
    max(1, (sym_n + 1) // 2),             # ceil(sym_n / 2)
)
```
For a 4-symbol cluster this is `2/4 = 50%` (matches the global ratio), but for
a 3-symbol cluster it's `2/3 = 67%` — proportionally stricter than the
intended 50% cross-symbol bar, purely as a side effect of integer ceiling on
small `sym_n`. Confirmed from the log: cluster_1/cluster_2 (3 symbols each)
both show `min_profitable_symbols=2`. Consider using a floor (`sym_n // 2`,
minimum 1) instead of a ceiling if the intent is "no stricter than the global
50% ratio."

## Files to Modify

1. `gpu_fuzzy_trader/run_pipeline.py` — banner text.
2. `gpu_fuzzy_trader/config.py` — `resolve_island_hyperparams` clarity; `min_profitable_symbols` rounding; remainder distribution helper (or leave A3 as a documented rounding choice if the team decides it's not worth the complexity).

## Detailed Changes

### Fix A1

```python
governor_suffix = (
    "RB_GOVERNOR unified selection+risk"
    if _cfg.RB_GOVERNOR_ENABLED
    else "PHASE3 per-symbol greedy | PHASE4 grid_search=True"
)
logger.info(
    "Pipeline config: PHASE1 top_k=%d | " + phase2_fmt + " | %s | %s",
    _cfg.PHASE1_TOP_K_FEATURES, *phase2_args, governor_suffix, debug_suffix,
)
```

### Fix A2

Either make `skip_symbol_robustness_penalty` genuinely conditional (e.g. only
skip when `n_symbols < some_threshold`), or — if permanently skipping for all
island profiles is the deliberate final answer — replace the dataclass field
with a plain module-level comment at the call site in
`compute_phase2_objectives_from_metrics` explaining why island mode never
runs `_symbol_robustness_penalty`, and remove the now-pointless per-profile
plumbing so a future reader doesn't assume it varies.

### Fix A3 (optional, low priority)

```python
def _split_generation_budget(total_gens: int, n_clusters: int) -> list[int]:
    base, remainder = divmod(total_gens, n_clusters)
    return [base + (1 if i < remainder else 0) for i in range(n_clusters)]
```
Use per-cluster budgets from this helper instead of a single
`gens_per_cluster` scalar in both `_run_cluster_islands` and
`_log_pipeline_config`.

### Fix A4 (needs a product decision, not just a code change)

Change `(sym_n + 1) // 2` to `max(1, sym_n // 2)` if the team wants clusters
never stricter than 50%; otherwise, document the intentional ceiling
behaviour in a comment so it isn't mistaken for a bug again.

## Acceptance Criteria

- [ ] Startup banner reflects `RB_GOVERNOR_ENABLED` truthfully.
- [ ] `skip_symbol_robustness_penalty`'s intent is either made genuinely conditional or explicitly documented as a permanent island-mode skip (team's choice, but no longer silently "always True regardless of input").
- [ ] (If implemented) generation-budget remainder is distributed rather than dropped, and the pipeline banner's `per_cluster` figure matches what the scheduler actually runs.
- [ ] (If implemented) `min_profitable_symbols` rounding matches the documented intent (50% floor vs ceiling) — decide and comment either way.
- [ ] Existing tests pass: `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_island_hyperparams.py tests/unit/ -x -q -k "island or pipeline_config"`

## Verification

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/ -x -q -k "island or config or pipeline"
```
