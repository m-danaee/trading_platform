# Task-22: Restore Phase2→RB-Governor→OOS Objective Continuity

**Branch:** `fix/objective-continuity`
**Priority:** 🔴 Critical (this is the primary suspect for "good Phase 2 metrics, bad OOS")
**Depends on:** none, but do AFTER task-20 lands (cheaper to add CV-fold compute once the duplicate-CPU-sim cost is gone)

## Problem

Two compounding issues mean the rule that Phase 2's NSGA-III search actually
optimizes for is only loosely related to what ends up deployed and evaluated
out-of-sample.

### O1: `PHASE2_F3_OBJECTIVE = "cv_fold_min"` never fires in the evolutionary hot path (confirmed)

`config.py:459` sets the active f3 objective to `"cv_fold_min"` (the
task-15/H2 "OOS-focused redesign"). The dispatch in
`compute_phase2_objectives_from_metrics` (`phase2_rule_pool.py:600-606`):

```python
if f3_objective == "cv_fold_min":
    cv_fold_returns = metrics.get("_cv_fold_returns", [])
    if cv_fold_returns:
        f3_val = min(cv_fold_returns)
    else:
        f3_val = profit_factor   # <-- silent fallback, this is what actually runs
```

`metrics["_cv_fold_returns"]` is only populated inside `_evaluate_chromosome`
when a `cv_fold_evaluator` is passed in. Tracing the live path: `_run_nsga3`
(the active algorithm — EvoX is available) evaluates **every** parent and
offspring individual via `_evaluate_population_indices`
(`evox_runner.py:1239-1421`, calls at `:2225` and `:2443`), and that function
**has no `cv_fold_evaluator` parameter at all**. `_run_nsga3` itself only
forwards `cv_fold_evaluator` to the one-time final
`_build_pool_from_archive(...)` call (`evox_runner.py:2651-2662`), and even
there it's gated behind `PURGED_WF_REQUIRE_ALL_CV_FOLDS`, which defaults
`False` (`config.py:195`). **Net effect: f3 is `-profit_factor` for the
entire evolutionary search, every generation, for 100% of individuals** — the
config says `cv_fold_min`, the code delivers `profit_factor`. `profit_factor`
is a pure train-side, single-split statistic with no cross-period robustness
signal — exactly the kind of thing that overfits and produces "good Phase 2
metrics, bad OOS."

The same gap exists in RB Governor: CV-fold consistency is wired into
`_filter_good_rules` (`rb_governor.py:553-555`, a one-time screen) but **not**
into `_compose_ruleset`'s team-growth loop, `_optimize_risk`'s grid search
(`rb_governor.py:679-751`), or the profit-amplifier override
(`rb_governor.py:1057-1123`) — all of which call `_evaluate_ruleset` /
`_score_metrics` without ever passing CV-fold returns.

### O2: RB Governor re-parametrizes every rule's risk before it even screens them (confirmed)

Phase 2 evolves and evaluates every chromosome at fixed
`PHASE2_TP=2.0 / PHASE2_SL=1.0 / PHASE2_CAPITAL_PCT=30.0`
(`config.py:389,394,399`). The Phase 2 pool JSON schema
(`_build_pool_from_archive`, `phase2_rule_pool.py:1382-1392`, enforced by
`_validate_pool_schema`, `phase2_rule_pool.py:3005-3008`) **only** contains
`chromosome`, `conditions`, `objectives`, `executed_trades` — it never
persists `tp`/`sl`/`capital_pct`.

RB Governor's `_rule_to_engine` (`rb_governor.py:136-147`):

```python
tp = float(rule.get("tp", getattr(_cfg, "RB_DEFAULT_TP", 2.0)))
sl = float(rule.get("sl", getattr(_cfg, "RB_DEFAULT_SL", 1.2)))
...
"capital_pct": float(rule.get("capital_pct", getattr(_cfg, "RB_DEFAULT_CAPITAL_PCT", 12.5))),
```

Since pool entries never have these keys, `.get()` **always** falls through
to `RB_DEFAULT_SL=1.2` (`config.py:1654`, vs Phase 2's `1.0` — a 20% wider
stop) and `RB_DEFAULT_CAPITAL_PCT=20.0` (`config.py:1655`, vs Phase 2's
`30.0` — a 33% smaller position). This reparametrization happens at the very
first RB Governor screening step (`_filter_good_rules`,
`rb_governor.py:541`), before RB Governor's own scoring
(`_score_metrics`, `rb_governor.py:162-227` — a hand-tuned linear blend
structurally unrelated to Phase 2's Sortino/drawdown/Pareto ranking), its
TP/SL/capital grid search (`_optimize_risk`, 160 combinations x
`RB_TP_GRID`/`RB_SL_GRID`/`RB_CAPITAL_GRID`, greedily maximizing
`_score_metrics` on the **same** train+valid split already used for
selection — a data-snooping risk with no CV term), and a fourth objective
(`_profit_amp_objective`, `rb_governor.py:754-766`) that can override the
composed team again with different drawdown weighting. **Nothing in this
chain re-validates against Phase 2's own fitness criteria** — Phase 2's
multi-objective search is effectively reduced to a candidate-generator, and
the actual "what gets deployed" decision is made by 3 sequential,
independently-tuned objective functions operating on different risk
parameters than what was searched over. Phase 5 OOS itself is faithful to
whatever RB Governor writes out (`phase5_oos.py:448-517` reads `tp`/`sl`
already baked into the strategy JSON) — the divergence is introduced
strictly between Phase 2 and RB Governor.

## Files to Modify

1. `gpu_fuzzy_trader/evolution/evox_runner.py` — thread `cv_fold_evaluator` into `_evaluate_population_indices` and both generation loops.
2. `gpu_fuzzy_trader/phases/phase2_rule_pool.py` — persist `tp`/`sl`/`capital_pct` on pool entries; update `_validate_pool_schema`; delete/update stale comments referencing `PHASE2_F3_OBJECTIVE="profit_factor"` as "active config".
3. `gpu_fuzzy_trader/rb_governor.py` — use Phase-2-provided risk params as the grid-search starting point instead of unconditional `RB_DEFAULT_*`; thread CV-fold consistency into `_compose_ruleset`/`_optimize_risk`.
4. `gpu_fuzzy_trader/config.py` — no value changes required, but add a short comment clarifying the intended Phase2→RB continuity contract.

## Detailed Changes

### F1: Wire real CV-fold returns into the hot loop (cheaply)

Add an optional `cv_fold_evaluator` parameter to `_evaluate_population_indices`
and thread it from `_run_nsga3`'s parent (`:2225`) and offspring (`:2443`)
calls. To avoid re-introducing task-20's runtime problem, do NOT recompute
full CV-fold backtests for every individual every generation. Options, in
order of preference:

- **(a) Admission-gate approach (cheapest, recommended):** Leave the live f3
  as `profit_factor` (rename `PHASE2_F3_OBJECTIVE` default back to
  `"profit_factor"` to make the config honest), but flip
  `PURGED_WF_REQUIRE_ALL_CV_FOLDS = True` so `_build_pool_from_archive`'s
  one-time, end-of-run CV check (already implemented, just disabled) becomes
  a hard admission gate on the harvested archive. This is a single CV
  evaluation per surviving Pareto member (small set), not per-generation.
- **(b) Throttled live objective:** Compute `_cv_fold_returns` only every
  `PHASE2_VAL_SIM_INTERVAL` generations (post task-20 fix) and only for
  unique chromosomes already deduped by `PHASE2_EVAL_BATCH_DEDUP`, caching
  the result in `global_metrics_cache` so it persists between the throttled
  refreshes.

Whichever is chosen, delete the stale comments at `evox_runner.py` (near the
offspring-eval calls, ~`:2436-2442` and ~`:2019-2025`) that assert
`PHASE2_F3_OBJECTIVE="profit_factor"` is "active config" — it no longer
matches `config.py`, and it's actively misleading.

### F2: Persist Phase 2's risk parameters onto pool entries

`_build_pool_from_archive` (`phase2_rule_pool.py:1382-1392`): add
`"tp": _cfg.PHASE2_TP, "sl": _cfg.PHASE2_SL, "capital_pct":
_cfg.PHASE2_CAPITAL_PCT` to `pool_entry`. Update `_validate_pool_schema`
(`phase2_rule_pool.py:2998-3015`) to tolerate (not require, for backward
compatibility with archived pools) these new optional keys.

### F3: Make RB Governor start from Phase 2's operating point

`rb_governor.py::_rule_to_engine` (`:136-147`): the `.get("tp", ...)` /
`.get("sl", ...)` / `.get("capital_pct", ...)` fallbacks already read from the
rule dict first — once F2 lands, this automatically starts from Phase 2's
actual values instead of `RB_DEFAULT_*`. Additionally, seed
`_optimize_risk`'s grid search (`rb_governor.py:679-751`) to search *around*
the incoming `tp`/`sl`/`capital_pct` (e.g. include those exact values as the
first grid points so "no change" is always a candidate) rather than a fixed
grid oblivious to where the rule came from.

### F4: Thread CV-fold consistency through RB Governor's scoring

Pass `cv_fold_returns` (via the existing `_eval_cv_fold_returns` used in
`_filter_good_rules`) into `_score_metrics` calls from `_compose_ruleset`
(`rb_governor.py:614,648-653`) and `_optimize_risk` (`rb_governor.py:718`),
not only the initial filter, so team composition and risk tuning cannot pick
a configuration that only `_filter_good_rules` considers CV-robust.

## Acceptance Criteria

- [ ] `PHASE2_F3_OBJECTIVE`'s config value and the code path that consumes it are consistent (either genuinely computing `cv_fold_min` per the chosen option, or the config default is changed back to `"profit_factor"` with an honest comment — no silent mismatch).
- [ ] If option (a): `PURGED_WF_REQUIRE_ALL_CV_FOLDS = True` and `_build_pool_from_archive` demonstrably rejects a synthetic rule that is profitable overall but has a negative worst-CV-fold return.
- [ ] If option (b): `_evaluate_population_indices` accepts `cv_fold_evaluator`; a unit test confirms `metrics["_cv_fold_returns"]` is populated on throttled generations.
- [ ] Stale comments referencing `PHASE2_F3_OBJECTIVE="profit_factor"` as "active config" are removed/updated.
- [ ] Phase 2 pool entries carry `tp`/`sl`/`capital_pct`; `_validate_pool_schema` accepts them as optional.
- [ ] `rb_governor._rule_to_engine` uses Phase-2-provided risk params when present (verified via a unit test with a pool entry that sets non-default `sl`/`capital_pct`).
- [ ] `_optimize_risk`'s grid search includes the incoming rule's risk params as an explicit candidate.
- [ ] `_compose_ruleset`/`_optimize_risk` scoring incorporates CV-fold consistency, not just `_filter_good_rules`.
- [ ] Existing tests pass: `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_rb_governor_cv_folds.py tests/unit/ -x -q -k "rb_governor or rule_pool or objective"`
- [ ] `evaluator_v5.ipynb` NOT modified.

## Verification

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/ -x -q -k "rb_governor or rule_pool or objective or cv_fold"
```
