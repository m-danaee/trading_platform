# Executive Summary

- Total confirmed bugs: 1
- CRITICAL: 1
- HIGH: 0
- MEDIUM: 0
- LOW: 0
- Overall assessment: the audited enrichment path was not research-isolated.
  Regime thresholds included rows beyond the effective Phase-0 training split.
  The confirmed leakage is fixed and covered by a regression test. The full
  production search and CUDA execution were not run, so this report does not
  certify the complete system as correct.

# Confirmed Bugs

## BUG-001 — Trend thresholds crossed the effective training boundary

Severity: CRITICAL  
Status: Fixed  
Affected files/functions: `gpu_fuzzy_trader/data/trend_context.py::build_train_prefix`

### Root cause

`build_train_prefix()` applied the 65% split directly to every raw symbol
tape. The runtime loader first removes the final `TAIL_DROP_ROWS` (96) rows
per symbol and the splitter applies 65% to that shorter tape. For 1,000 raw
rows, threshold fitting therefore used 650 rows while Phase-0 training used
only `floor((1000 - 96) * 0.65) = 587` rows. Rows 587–649 belonged after the
effective training boundary but influenced all fitted LWC/MWC/HWC regime
thresholds.

### Reproduction

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest -q \
  tests/unit/test_trend_context.py::TestTrainPrefixOnlyFitting::test_prefix_matches_the_tail_trimmed_phase0_split
```

Before the fix, the assertion reported `{'AA': 650, 'BB': 650}` instead of
`{'AA': 587, 'BB': 587}`.

### Expected behavior

Threshold fitting must use exactly the leading rows later assigned to the
Phase-0 training split after the label-horizon tail trim.

### Actual behavior

Threshold fitting consumed 63 post-training rows per symbol in the minimal
reproduction.

### Research/trading impact

Validation-period price distributions affected the classifier thresholds
that generate mandatory HWC/MWC permission and LWC reversal context. This can
alter candidate signals, Phase 1/2 outcomes, RB selection, and reported
validation performance, invalidating research isolation.

### Fix

Compute the eligible per-symbol row count after subtracting
`TAIL_DROP_ROWS`, then apply the shared training fraction. No strategy or
evaluator policy was otherwise changed.

### Regression test

Added a deterministic two-symbol test asserting exact agreement between the
enrichment prefix and the loader/splitter boundary. The existing future-tail
metamorphic test now perturbs everything after this corrected boundary.

### Validation performed

The new test failed before the fix and passed after it. Trend-context,
splitter, and research-integrity tests were run together after the change.

---

# Suspected Issues

- The canonical notebook intentionally evaluates the reference/train tape
  with `drop_tail=False`, while production training splits use
  `drop_tail=True`. The notebook documents this train-view asymmetry. It is
  not classified as a bug because test/OOS evaluation still uses the same
  first-touch and execution contract, but notebook train metrics are not
  directly comparable to pipeline train ledgers.
- GPU execution was not available in this environment. Static and CPU-backed
  tests cannot certify CUDA/JIT ranking behavior.

# Evaluator Parity Results

The notebook and CPU implementation were inspected for entry-at-next-open,
exact first-touch outcomes, conservative same-bar TP/SL handling, 96-bar time
exit, fee calculation, exposure release, and rule-order allocation. Existing
targeted parity tests were included in the feasible suite. No new divergence
was confirmed. A full notebook execution across Phase 2, RB, and Phase 5 was
not performed.

# Data Leakage Audit

Architecture and research boundaries:

```text
raw train -> train-prefix threshold fit -> separate causal enrichment
          -> loader/label tail trim -> per-symbol train | embargo | validation
          -> Phase 1 direction features -> Phase 2 specialist pools
          -> RB validation selection/sizing -> locked strategy JSON
raw test  -> frozen-threshold enrichment -> Phase 5 diagnostic only
forward   -> frozen-threshold enrichment -> one-time acceptance only
strategy JSON + enriched tape -> evaluator_v5.ipynb
```

BUG-001 was a validation-to-threshold leak. Test and forward paths reuse
frozen thresholds and were not observed as direct tuning inputs. Artifact
identity, allowed-direction filtering, and fail-closed behavior have existing
focused coverage, but no full production artifact set was generated here.

# Multi-Timeframe Causality Audit

The enrichment implementation aggregates independently per symbol, keeps
only complete HTF buckets, publishes at HTF close, and aligns against the
next-open execution timestamp with backward search. Existing deterministic
boundary, warm-up, append-future, timezone, and symbol-isolation tests were
reviewed/run through the trend-context suite. No one-bar defect was confirmed.

# Cache/Resume Audit

Split, feature, rule-pool, archive, dataset, and strategy identities were
inspected through their focused test coverage. No additional reproducible
stale-artifact reuse was confirmed. A reduced real search was not completed,
so concurrent crash/resume of production artifacts remains unverified.

# CPU/JAX Parity Results

CPU/reference and installed JAX-path tests were part of the feasible suite;
benchmark tests remained skipped by project policy. CUDA hardware execution
was not tested and no claim of GPU parity is made.

# Numerical Edge Cases

The evaluator uses finite-entry validation and bounded Profit Factor behavior;
the CPU engine maps empty trades, no-loss trades, zero downside deviation,
expectancy, expected shortfall, and drawdown to finite metrics. Existing unit
and property tests exercise these paths. No new numerical bug was confirmed.

# Tests Added

- Exact enrichment-prefix versus tail-trimmed Phase-0 split boundary.
- Strengthened the future-tail threshold invariance test to begin at the
  actual effective training boundary.

# Files Changed

- `gpu_fuzzy_trader/data/trend_context.py`
- `tests/unit/test_trend_context.py`
- `BUG_HUNT_REPORT.md`

# Remaining Risks

- No CUDA device was available, so real GPU execution remains unverified.
- The full expensive Phase 2 search was intentionally not launched.
- No untouched forward tape is checked in, so one-time forward acceptance was
  not runtime-tested end to end.
- The dependency index was unavailable; the environment used a `.venv` backed
  by already-installed system packages rather than a fresh lockfile install.
- Any enriched datasets and downstream artifacts generated before BUG-001 was
  fixed must be regenerated; their thresholds were fitted with the leaked
  boundary.
