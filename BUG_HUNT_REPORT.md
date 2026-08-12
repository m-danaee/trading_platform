# Executive Summary

Audit scope: the documented train-to-forward research path, causal context
enrichment, split/cache boundaries, Phase 1, Phase 2, RB Governor, strategy
output, Phase 5, the canonical evaluator notebook contract, and CPU/JAX
execution paths. This was a correctness audit, not an architectural refactor.

Architecture verified:

```text
raw train/test/forward tapes
  -> causal trend-context enrichment (thresholds fitted on train only)
  -> enriched train split into train / validation fitness / validation selection
  -> Phase 1 direction-specific feature selection
  -> Phase 2 rule search and CPU archive admission
  -> RB Governor selection and sizing
  -> immutable strategy JSON
  -> Phase 5 diagnostic test evaluation
  -> optional strictly newer forward acceptance
```

Research boundaries are train -> validation -> consumed test -> strictly newer
forward. Test and forward data are not intended to be inputs to Phase 1, Phase
2, RB, or tuning; Phase 5 is diagnostic on the consumed test tape.

- Total confirmed bugs: 8
- CRITICAL: 0
- HIGH: 6
- MEDIUM: 2
- LOW: 0

Overall assessment: the audited CPU/enriched-data research path is materially
safer after the fixes. No confirmed train/validation/test leakage or
multi-timeframe look-ahead was found in the inspected implementation and
synthetic causality tests. Before this audit, stale artifact reuse, rejected
strategy evaluation, invalid numeric strategy parameters, and two CPU
reporting/parity defects could undermine research conclusions. A real CUDA
device and a genuine untouched forward acceptance run remain unverified.

The final low-memory suite passed:

```text
PYTEST_LOW_MEMORY=1 MPLCONFIGDIR=/tmp .venv/bin/python -m pytest -q
1473 passed, 2 skipped, 18 warnings in 734.58s
```

The warnings are `ConstantInputWarning` from SciPy Spearman calculations in
reporter property tests. Constant vectors make correlation undefined; the
property test exercises the same expected condition. They were inspected and
are not a confirmed strategy-selection defect.

# Confirmed Bugs

## BUG-001 — Split cache trusted mtime instead of source contents

Severity: HIGH  
Status: Fixed and regression-tested  
Affected files/functions: `gpu_fuzzy_trader/data/splitter.py`,
`load_cached_split_if_fresh`, `Data_Splitter.persist_splits`,
`validation/rolling_cv.py::write_cv_folds_manifest`

### Root cause

The split cache validated the source path, configuration fingerprint, and
mtimes, but not the source CSV bytes. Editing a CSV while restoring its mtime
let an old train/validation split be reused for different data.

### Reproduction

`tests/unit/test_data_splitter.py::TestLoadCachedSplitIfFresh::test_cache_rejected_when_csv_content_changes_without_mtime_change`

Before the fix, the test changed source content, restored its mtime, and the
loader returned the stale cached split.

### Expected behavior

Any source-content change must invalidate the split cache.

### Actual behavior

The stale split was accepted when its mtime still appeared fresh.

### Research/trading impact

Phase 1, Phase 2, and RB could receive splits from another tape while reports
appeared to describe the current source data.

### Fix

Persist a streaming SHA-256 of `TRAIN_CSV_PATH` in the split manifest and
require an exact current digest match. Legacy manifests without a hash fail
closed. A source that cannot be hashed is persisted as non-reusable.

### Regression test

The test above failed before the fix and passes after it. Related split/CV
tests: `42 passed`.

### Validation performed

Focused split/CV tests, the reduced pipeline, and the final full suite.

---

## BUG-002 — Strategy writer allowed non-finite and non-positive risk fields

Severity: HIGH  
Status: Fixed and regression-tested  
Affected files/functions: `gpu_fuzzy_trader/output/writer.py::_validate_rule`

### Root cause

The writer converted `tp`, `sl`, and `capital_pct` to floats but rejected only
the case where all three were zero. It accepted `NaN`, positive/negative
infinity, zero in an individual field, and negative values.

### Reproduction

`tests/unit/test_output_writer.py::TestWriteRiskParameterValidation::test_nonfinite_or_nonpositive_value_raises_validation_error`

Before the fix, parameterized values including `NaN`, `inf`, `-inf`, zero, and
negative values could pass writer validation.

### Expected behavior

Evaluator-facing risk fields must be finite and strictly positive.

### Actual behavior

Malformed strategy JSON could be written and later cause undefined sizing,
invalid barrier behavior, or evaluator inconsistency.

### Research/trading impact

Invalid numbers can silently poison metrics or make an artifact look usable
until a later consumer rejects it.

### Fix

Require every `tp`, `sl`, and `capital_pct` value to be finite and greater
than zero. The obsolete all-three-zero special case was removed.

### Regression test

The new parameterized unit regression was failing before the fix and passes
after it. The output-writer property generator was also corrected to generate
strictly positive fields when it labels a rule as valid; this aligns the test
data with the evaluator contract rather than relaxing validation.

### Validation performed

`tests/unit/test_output_writer.py` plus
`tests/property/test_output_writer_properties.py`: `59 passed`; final full
suite passed.

---

## BUG-003 — CPU time-exit metric depended on log generation

Severity: MEDIUM  
Status: Fixed and regression-tested  
Affected files/functions: `gpu_fuzzy_trader/backtest/cpu_engine.py`

### Root cause

`time_closed_count` was inferred only while `return_logs=True`. The no-log
evaluation path left the exit reason unset, so valid time exits were counted as
zero even though PnL was calculated.

### Reproduction

`tests/unit/test_cpu_engine.py::TestTradeOutcomeLong::test_time_closed_count_does_not_depend_on_return_logs`

Before the fix, the same scenario returned a different time-close count based
only on whether logs were requested.

### Expected behavior

Metrics must be identical regardless of whether diagnostic trade logs are
requested.

### Actual behavior

Time-exit reporting was understated in the no-log path.

### Research/trading impact

Holding-period diagnostics and any downstream report using that count could be
incorrect.

### Fix

Derive an exit reason in both paths: directly from exact barrier values when
available and from the single-trade fallback otherwise.

### Regression test

The focused no-log regression failed before the fix and passes after it.

### Validation performed

CPU engine and CPU property tests: `103 passed`; final full suite passed.

---

## BUG-004 — Phase 5 evaluated explicitly RB-rejected strategies

Severity: HIGH  
Status: Fixed and regression-tested  
Affected files/functions: `gpu_fuzzy_trader/phases/phase5_oos.py::OOS_Evaluator.load_strategies`

### Root cause

The output normalizer strips metadata not needed by the evaluator. Phase 5
validated the normalized result without first checking raw
`deployment_accepted`, so a syntactically valid strategy marked
`deployment_accepted: false` could still be evaluated.

### Reproduction

`tests/unit/test_phase5_oos.py::TestLoadStrategies::test_skips_explicitly_rejected_strategy_even_when_rules_are_valid`

Before the fix, a valid rule set with explicit RB rejection was returned by
`load_strategies`.

### Expected behavior

An explicitly rejected strategy must never enter standalone or pipeline Phase
5 evaluation.

### Actual behavior

Phase 5 could produce test metrics for a rejected strategy.

### Research/trading impact

Rejected candidates could appear in diagnostic results and be mistaken for
current eligible output.

### Fix

Read raw JSON first and skip a direction when `deployment_accepted is False`.
Unreadable JSON and filesystem errors now also fail closed.

### Regression test

The focused rejected-strategy regression failed before the fix and passes
after it.

### Validation performed

Phase 5 unit/property tests: `64 passed`; reduced pipeline produced fresh
fail-closed strategy files; final full suite passed.

---

## BUG-005 — CPU reference arithmetic diverged from canonical evaluator precision

Severity: MEDIUM  
Status: Fixed and regression-tested  
Affected files/functions: `gpu_fuzzy_trader/backtest/cpu_engine.py`

### Root cause

The CPU engine down-cast entry prices, label returns, and fallback outcomes to
`float32`. The canonical notebook performs this accounting with Python/NumPy
float precision. Cumulative equity and derived metrics therefore differed on a
deterministic exact-barrier tape.

### Reproduction

`tests/unit/test_cpu_engine.py::TestCanonicalNotebookPrecision::test_exact_barrier_metrics_match_canonical_notebook`

Before the fix, the CPU total return was `0.059473514556884766`, while the
canonical notebook result was `0.05946889285850521` for the same tape.

### Expected behavior

CPU reference metrics must exactly match the canonical evaluator for the same
trade tape and rule parameters.

### Actual behavior

Small float32 differences accumulated into measurable metric differences.

### Research/trading impact

Although small in this probe, a boundary metric can change candidate ranking
or a gate decision.

### Fix

Keep reference price/return arrays and fallback outcomes at float64 precision;
preserve the barrier and fee semantics.

### Regression test

The canonical notebook precision regression failed before the fix and passes
after it.

### Validation performed

The exact synthetic long and short tapes now match the notebook on trade logs,
PnL, equity, return, MDD, profit factor, average notional, and time-close
count. CPU engine/property tests: `103 passed`; final full suite passed.

---

## BUG-006 — Phase 1 resume reused schema-valid selections without input identity

Severity: HIGH  
Status: Fixed and regression-tested  
Affected files/functions: `gpu_fuzzy_trader/features/selector.py`,
`Pipeline_Orchestrator._run_phase1`

### Root cause

`Feature_Selector.skip_if_valid()` previously checked only file schema and the
disabled flag. The `--resume` path supplied no current data, so an old selected
feature file could be reused after source, feature, target, context, or
relevant configuration changes.

### Reproduction

`tests/unit/test_feature_selector.py::TestSkipIfValid::test_legacy_schema_valid_files_are_not_reused_without_identity`

Before the fix, schema-valid legacy JSON was returned instead of forcing Phase
1 to rerun.

### Expected behavior

Phase 1 artifacts may be reused only when they prove they were produced from
the current masked training input and selection contract.

### Actual behavior

Stale features could seed a new Phase 2 search.

### Research/trading impact

Feature selection results could be attributed to the wrong dataset or
configuration.

### Fix

Persist a Phase 1 identity built from canonical train/validation frame hashes,
selection-relevant configuration, seed, label/meta/internal contracts, context
digest, and selector/detector code hashes. Reuse fails closed without current
input or an exact identity match.

### Regression test

Legacy-artifact, changed-frame, corrupt-artifact, and disabled-toggle cases
were exercised. The legacy regression failed before the fix and passes after
it.

### Validation performed

`tests/unit/test_feature_selector.py`: `76 passed`; reduced pipeline showed
identical-input Phase 1 reuse; final full suite passed.

---

## BUG-007 — Phase 2 resume reused pools without a semantic identity

Severity: HIGH  
Status: Fixed and regression-tested  
Affected files/functions: `gpu_fuzzy_trader/phases/phase2_rule_pool.py`,
`gpu_fuzzy_trader/run_pipeline.py::_run_phase2`

### Root cause

`Rule_Pool_Generator.skip_if_valid(direction)` accepted any schema-valid pool.
It did not bind the pool to train/validation contents, Phase 1 features, CV
boundaries, context, configuration, or relevant search/evaluator code. A stale
pool and history could also be merged into a new generation.

### Reproduction

`tests/unit/test_phase2_rule_pool.py::TestLoadPool::test_skip_if_valid_rejects_legacy_pool_without_identity` and
`tests/unit/test_run_pipeline.py::test_phase2_regenerates_legacy_pool_before_generator_can_merge_it`

Before the fix, a bare schema-valid pool was a resume hit; the unit regression
observed a list where it expected `None`.

### Expected behavior

Resume must accept only a pool proven to match the exact current Phase 2
inputs, and stale cache files must not reach the generator.

### Actual behavior

Old candidate search results could be silently reused or mixed into a new
population.

### Research/trading impact

Search, validation, RB selection, and reported results could be based on a
different rule universe than the current run declares.

### Fix

Add an atomic `*.identity.json` sidecar containing version, direction, full
input identity, and a SHA-256 of the pool bytes. The identity covers canonical
train/validation hashes, selected features, CV boundaries, context digest, all
uppercase configuration values, and relevant code hashes. Missing, changed,
or malformed sidecars fail closed; stale pool, history, and sidecar files are
removed before fresh generation.

### Regression test

Added legacy, mismatch, post-sidecar-byte-change, stale-merge-prevention, and
sidecar-cleanup regressions. The legacy regression failed before the fix and
passes after it.

### Validation performed

`tests/unit/test_phase2_rule_pool.py`: `171 passed`; pipeline orchestration
tests passed; an identical diagnostic resume reused the bound pool while a
`FEE_PCT` change discarded and regenerated it; final full suite passed.

---

## BUG-008 — Standalone phase commands bypassed prerequisite provenance checks

Severity: HIGH  
Status: Fixed and regression-tested  
Affected files/functions: `Pipeline_Orchestrator.run_from_phase2`,
`run_phase(2/3/4)`, `_load_phase1_outputs`, `_load_phase2_outputs`

### Root cause

Even after the normal resume fixes, standalone loaders called raw schema
loaders: Phase 1 used `load_and_validate` and Phase 2 used `load_pool`. Thus
`--from-phase 2`, `--phase 2`, and RB-only phase aliases could bypass the
identity checks and trust copied or stale artifacts.

### Reproduction

`tests/unit/test_run_pipeline.py::test_standalone_phase1_loader_rejects_unproven_artifacts` and
`tests/unit/test_run_pipeline.py::test_standalone_phase2_loader_rejects_unproven_artifacts`

Both regressions failed before the fix: a stale Phase 1 JSON was accepted and
a bare Phase 2 pool was loaded.

### Expected behavior

Standalone phases must have the same provenance guarantees as a normal resume.

### Actual behavior

Disk-backed prerequisites could cross a research boundary without a current
input check.

### Research/trading impact

An RB-only or Phase-2-only command could produce a current-looking strategy
from stale feature selection or stale candidate pools.

### Fix

Standalone Phase 2 now masks current train data and verifies Phase 1 identity.
Standalone RB aliases additionally rebuild the same pruned train/validation/CV
view, derive the expected Phase 2 identity, and load only a matching sidecar.
Missing identity fails closed with an explicit status.

### Regression test

The two failing pre-fix regressions now prove rejection of stale artifacts and
acceptance of correctly bound artifacts. Phase 3/4 compatibility tests were
updated for the explicit prerequisite validation.

### Validation performed

`tests/unit/test_run_pipeline.py`: `22 passed`; final full suite passed.

---

# Suspected Issues

These are not confirmed bugs because the required evidence was unavailable or
would require unsafe concurrent/CUDA execution in this environment.

1. **GPU candidate-ranking coverage difference (potentially HIGH).** Static
   inspection shows GPU/JAX can use a simplified/vectorized path for compatible
   label-only data, while exact-barrier production tapes and final archive/RB/
   Phase 5 evaluation use CPU. Final accepted artifacts therefore receive CPU
   admission, but candidate discovery/ranking could differ in a custom path
   without exact barriers. CPU/JAX reference parity tests passed, but no CUDA
   device was available to test the actual kernel.

2. **Shared output directories are not a concurrent-run contract (potentially
   MEDIUM).** The new Phase 2 identity sidecar is atomically replaced, but
   Phase 1 JSON, base Phase 2 pool/history, strategy JSON, and Phase 5 reports
   still have direct writes. There is no inter-process output-directory lock.
   A normal single run is covered; two simultaneous runs targeting the same
   `--output` directory could race. Use separate output directories until a
   separately scoped concurrency hardening effort is approved.

# Evaluator Parity Results

The canonical notebook cells relevant to loading, rule evaluation, costs, and
metrics were executed against deterministic synthetic tapes. The same rules
were also evaluated through the CPU engine and covered by Phase 2/Phase 5
parity tests.

| Scenario | Result |
| --- | --- |
| Exact-barrier, 8-row long tape | Exact entry/exit logs and PnL matched. Return `0.05946889285850521`, MDD `0.2099279568000094`, PF `1.1649607310981513`, final equity `1000.5946889285851`, average notional `100.06833996932981`, time closes `2`. |
| Exact-barrier, symmetric short tape | Same metric values and matching logs under the short rule. |
| Fallback barrier/time tape | Exact checked metrics: final equity `1001.3004197839999`, MDD `0.1197844096475211`, PF `1.5416519497355823`, return `0.1300419783999862`. |
| Same-bar TP/SL ambiguity | The shared barrier policy is conservative stop-first; CPU tests cover both directions. |
| Holding horizon | 96 x 15-minute bars is consistently used by labels, CPU execution, validation geometry, Phase 5, and the notebook contract. |

The float32/float64 discrepancy in BUG-005 was the only observed CPU/notebook
divergence. It is fixed. Phase 2 batch evaluator, RB, and Phase 5 tests passed
on the current CPU contract; no unexplained parity difference remains in the
tested paths.

# Data Leakage Audit

Reviewed boundary flow:

- Trend thresholds are fitted from the training prefix only. Test and forward
  enrichment reuse frozen train-derived thresholds.
- The pipeline loads enriched training data for splitting, Phase 1, Phase 2,
  and RB. Validation is divided into Phase 2 fitness and RB selection windows.
- Phase 5 reads the enriched consumed test tape only after RB has written its
  strategy. It is diagnostic and does not tune or rewrite that strategy.
- Forward data is optional, must be strictly newer, and is the only acceptance
  tape. It is not fed into earlier phases.
- Dataset/context manifests bind source and enrichment contracts; the cache
  fixes above remove the verified stale-split/Phase-1/Phase-2 paths.

No confirmed train/validation/test/forward contamination was found in the
inspected code or deterministic tests. This is a scoped conclusion, not a
claim that an untested external caller cannot misuse output files. The
standalone artifact bypass was found and fixed as BUG-008.

# Multi-Timeframe Causality Audit

Reviewed `trend_context.py` timestamp validation, per-symbol resampling,
complete-bucket filtering, next-open publication, frozen thresholds, and
history handling.

- A completed 1h bar opened at 10:00 may first influence the 10:45 signal row,
  whose entry is at 11:00.
- A completed 4h bar opened at 08:00 may first influence the 11:45 signal row,
  whose entry is at 12:00.
- Incomplete leading/trailing higher-timeframe buckets are dropped.
- Input requires timezone-naive, unique `(datetime, symbol)` rows on the
  strict 15-minute grid; off-grid timestamps, duplicates, and gaps are
  rejected.
- Warm-up history is used causally but not emitted/scored as target rows.
- `test_appending_shocked_future_rows_does_not_change_prior_context` freezes
  thresholds, appends 128 shocked rows, and proves all context columns for the
  original 640 rows are unchanged.
- `test_higher_timeframe_state_isolated_by_symbol` proves no HWC/MWC state
  crosses a symbol boundary. Existing trigger tests also cover interleaving and
  pullback-window symbol isolation.

No confirmed one-bar alignment error or cross-symbol MTF leakage was found.

# Phase 1, Phase 2, and RB Audit

- Phase 1 excludes labels, metadata, internal columns, and fixed context
  fields from evolved feature selection. Long and short selections remain
  direction-specific.
- The mandatory `tf_permission_*` and `lwc_pullback_reversal_*` conditions are
  fixed execution masks, not chromosome genes. The reviewed Phase 2 and RB
  tests cover their preservation through mutation/crossover/admission/export.
- Phase 2 uses CPU final archive admission on the production exact-barrier
  path. Rule pools are now identity-bound before resume.
- RB tests cover fail-closed empty/missing pools, capital feasibility, tail and
  concentration gates, distinct-symbol rules, composition, and immutable
  strategy identity. A rejected direction writes an empty strategy with
  `deployment_accepted: false`.
- BUG-004 closes the final Phase 5 load gap for a raw explicitly rejected
  strategy.

No separate confirmed mutation, mandatory-context, objective-orientation, or
RB exposure-normalization defect was found in this audit.

# Cache/Resume Audit

Dynamic semantic changes exercised during the audit:

- Changing CSV contents while preserving mtime now rejects the split cache.
- Changing the Phase 1 frame rejects its persisted selection.
- Legacy Phase 1 and Phase 2 artifacts without provenance fail closed.
- Changing `FEE_PCT` in the reduced run discarded Phase 2 pools before
  regeneration; an identical rerun reused the bound Phase 1 and Phase 2
  artifacts.

Static identity coverage after the fixes:

- Split: source bytes plus existing split configuration fingerprint.
- Phase 1: canonical frame content, relevant selection config, seed,
  label/meta/internal contracts, context digest, and selector/detector code.
- Phase 2: train/validation content, selected features, CV boundaries,
  context digest, all uppercase configuration, pool-byte hash, and relevant
  evaluator/search code.
- Standalone Phase 2/RB commands derive and require those current identities.

The audit did not treat a cache hit as evidence of correctness; stale cache
files are now rejected rather than silently trusted.

# CPU/JAX Parity Results

- `tests/unit/test_gpu_engine.py::TestGPUCPUNumericalParity::test_parity_total_return`: passed.
- `tests/unit/test_gpu_engine.py::TestGPUCPUNumericalParity::test_simulate_rule_set_exact_parity`: passed.
- `tests/unit/test_gpu_rule_set_batch.py::TestGPUCPUReturnParity::test_gpu_cpu_return_parity`: `3 passed`.
- CPU engine/property suite: `103 passed`.

CUDA was unavailable in this environment: JAX initialization reported
`cuInit(0)` error 304 and exposed only `cpu:0`. Therefore these results verify
the CPU/JAX reference path, not a real CUDA execution. See the GPU suspected
issue above.

# Numerical Edge Cases

Checked zero/empty trades, zero winners/losers, zero downside, NaN/inf,
flat-equity behavior, fee monotonicity, barriers, and metric aggregation via
existing and added unit/property tests.

- BUG-002 prevents non-finite/non-positive risk fields from becoming strategy
  artifacts.
- BUG-003 removes a no-log metric dependency.
- BUG-005 removes CPU float32 reference drift.
- Existing property tests cover empty trades, fees, drawdown, profit factor,
  and per-symbol metric consistency.
- The 18 full-suite Spearman warnings occur only for constant test vectors;
  they do not represent an accepted `NaN` fitness/risk value.

No additional confirmed silent numerical-selection bug was found.

# Reduced Pipeline Run

A small isolated diagnostic run was executed sequentially to respect local RAM
limits:

```text
raw train/test -> outputs/bug_hunt_reduced/enriched
-> Phase 1 -> tiny CPU Phase 2 -> RB Governor -> Phase 5
```

The run used one BTC symbol, CPU route, population 4, two generations, and a
small sample. It completed and wrote a completed manifest at
`outputs/bug_hunt_reduced/pipeline/reports/run_manifest.json`. The intentionally
tiny search produced empty pools, so both RB outputs correctly failed closed:

```json
{"deployment_accepted": false, "fail_closed": true, "reason": "empty_pool"}
```

This proves error handling and artifact flow, not strategy quality. The
feasibility-collapse warnings are expected for the deliberately undersized
diagnostic search.

# Tests Added

- Split content mutation with preserved mtime.
- Non-finite, zero, and negative strategy risk fields.
- No-log time-close metric consistency.
- Exact canonical-notebook CPU precision parity.
- Explicit RB-rejected strategy rejection in Phase 5.
- Phase 1 legacy, changed-input, corrupt-artifact, and disabled-toggle cache
  behavior.
- Phase 2 legacy/mismatched/mutated-sidecar cache behavior, stale-pool removal
  before generation, and sidecar cleanup.
- Standalone Phase 1 and Phase 2/RB artifact provenance checks.
- Future-shock MTF invariance and higher-timeframe symbol isolation.

Focused suites run during the audit included split/CV (`42 passed`), feature
selection (`76 passed`), Phase 2 (`171 passed`), output writer/property (`59
passed`), trend context (`31 passed`), CPU/property (`103 passed`), Phase 5/
property (`64 passed`), and pipeline orchestration (`22 passed`).

# Files Changed

Production fixes:

- `gpu_fuzzy_trader/data/splitter.py`
- `gpu_fuzzy_trader/validation/rolling_cv.py`
- `gpu_fuzzy_trader/features/selector.py`
- `gpu_fuzzy_trader/phases/phase2_rule_pool.py`
- `gpu_fuzzy_trader/run_pipeline.py`
- `gpu_fuzzy_trader/output/writer.py`
- `gpu_fuzzy_trader/backtest/cpu_engine.py`
- `gpu_fuzzy_trader/phases/phase5_oos.py`

Regression/property coverage:

- `tests/unit/test_data_splitter.py`
- `tests/unit/test_feature_selector.py`
- `tests/unit/test_phase2_rule_pool.py`
- `tests/unit/test_run_pipeline.py`
- `tests/unit/test_output_writer.py`
- `tests/unit/test_cpu_engine.py`
- `tests/unit/test_phase5_oos.py`
- `tests/unit/test_trend_context.py`
- `tests/property/test_output_writer_properties.py`
- `tests/property/test_cpu_engine_properties.py`

Audit documentation and graph:

- `BUG_HUNT_REPORT.md`
- `graphify-out/` (AST-only graph refresh after the source changes)

Unrelated pre-existing worktree changes were preserved. Generated reduced-run
artifacts remain under `outputs/bug_hunt_reduced/` and are separate from source
and test changes.

# Remaining Risks

1. No CUDA hardware execution was possible; real-device GPU kernel behavior,
   performance, and candidate-ranking equivalence remain to be tested on a
   CUDA host.
2. The reduced run is a correctness smoke test only. It is not a full
   production Phase 2 search and must not be interpreted as performance or
   forward/OOS acceptance evidence.
3. No untouched strictly newer forward tape was available for a real acceptance
   run. Current results remain historical train/validation plus consumed-test
   diagnostics.
4. Do not run two pipelines in the same output directory concurrently; see the
   suspected concurrency issue.
5. The full suite is green under low-memory mode, but future changes to
   evaluator semantics, context contract, source data, or hardware should
   trigger fresh enrichment and identity-bound reruns.
