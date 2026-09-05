# GPU Fuzzy Trader

This project discovers fuzzy trading rules in Phase 2, composes them through a
causal hierarchical MTF runtime (4h HWC, 1h MWC, 15m LWC), selects and
risk-tunes candidates with the RB Governor, and evaluates evaluator-compatible
strategy JSON with the `evaluator_v5.ipynb` contract. HWC and MWC provide
continuous direction/evidence scores and contradiction vetoes; LWC alone emits
entry triggers.

Long Phase 2 and RB runs are intended for a CUDA host. The default runtime is
hybrid-tuned for an 8-core CPU plus a 6-GiB RTX 4050: large-window Phase 2
ranking uses the optimized CPU batch path, while JAX/GPU remains available for
smaller or high-throughput windows. Exact rule-set/RB/OOS evaluation stays on
CPU. Do not run the full pipeline on a memory-constrained WSL machine. See
[RUN.md](RUN.md) for the execution policy and command reference.

For the alternate Colab T4 path, run [main.ipynb](main.ipynb). Colab is
detected automatically and keeps Phase 2 on the JAX GPU with memory-safe
batching; the notebook stages data and outputs on `/content` before syncing
results to Google Drive.

## Repository layout

```text
gpu_fuzzy_trader/   Core pipeline package
tests/              Unit, property, and benchmark tests
RUN.md              Detailed runbook (setup, CLI, tests, troubleshooting)
evaluator_v5.ipynb  Canonical evaluator notebook for rule-set evaluation
requirements.txt    Shared dependency contract; main.ipynb selects the CUDA extra
```

## Pipeline

1. Raw 15m OHLCV is normalized to UTC; incomplete or gapped 1h/4h buckets are
   discarded and features are computed independently per timeframe.
2. One adaptive expanding master-fold system discovers HWC directional rules,
   then MWC conditional continuation rules using only OOF HWC scores. Each role
   applies its own derived forward-label purge at row retrieval time.
3. LWC discovery receives only OOF HWC and MWC scores; validation and OOS use
   frozen full-train ensembles.
4. The MTF composer applies asymmetric HWC/MWC contradiction vetoes to LWC
   triggers and records retention globally, by direction, symbol, fold, and
   month.
5. RB Governor evaluates the composed candidate and freezes its archives,
   manifest, and execution parameters. The active MTF path reserves the final
   25% of validation-selection data as a chronological tail check by default;
   it is not used to construct the candidate.
6. Phase 5 is diagnostic on the consumed test tape; only a strictly newer,
   untouched forward tape can provide release acceptance.

The RB Governor treats entry conditions, symbol scope, TP/SL, horizon, and cost
model as one immutable strategy identity. Production RB may size capital, but
it does not rescue rejected Phase 2 rules by silently searching a new TP/SL
envelope.

The canonical MTF path has no mandatory legacy context columns. HWC/MWC are
soft veto layers: neutral or weak evidence is permissive, and only strong
opposing direction with sufficient evidence can block an LWC trigger.

Existing callers using `--phase 3` or `--phase 4` remain compatible: both
normalize to the RB Governor and return the historical result keys alongside
`rb_governor`. They are compatibility inputs, not production pipeline stages.
There is no fallback strategy: a missing or rejected direction writes an
explicit empty, `deployment_accepted: false` strategy with a reason.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -c "from gpu_fuzzy_trader.run_pipeline import Pipeline_Orchestrator; print('OK')"
```

`main.ipynb` installs the hardware-specific JAX extra after the shared base:

```bash
# Colab T4 / CUDA 12
.venv/bin/pip install -U "jax[cuda12]==0.10.1"

# Local WSL/Linux with CUDA 13
.venv/bin/pip install -U "jax[cuda13]==0.10.1"
```

## Run

Use the repository virtual environment for every command.

The canonical pipeline consumes raw tapes directly and builds causal MTF data
in memory. Dataset and strategy identity includes the raw hashes, fold
boundaries, feature schemas, fitted thresholds, archive hashes, and frozen
composer parameters.

The normal pipeline uses the raw train/test tapes by default. Override paths
only for a controlled experiment; Phase 5 computes the frozen MTF runtime from
the raw tape and does not refit on test or forward data:

```bash
TRAIN_CSV_PATH=data/train_new.csv \
TEST_CSV_PATH=data/test_new.csv \
FORWARD_CSV_PATH=data/forward.csv \
.venv/bin/python -m gpu_fuzzy_trader.run_pipeline
```

Leave `FORWARD_CSV_PATH` unset when no untouched forward tape exists.

```bash
.venv/bin/python -m gpu_fuzzy_trader.run_pipeline
.venv/bin/python -m gpu_fuzzy_trader.run_pipeline --output outputs/run_a
.venv/bin/python -m gpu_fuzzy_trader.run_pipeline --resume
.venv/bin/python -m gpu_fuzzy_trader.run_pipeline --phase 2
.venv/bin/python -m gpu_fuzzy_trader.run_pipeline --from-phase 2
.venv/bin/python -m gpu_fuzzy_trader.run_pipeline --phase 5
```

By default the orchestrator rebuilds the deterministic train-only rule-feature
catalog, Phase 2, and RB. Pass `--resume` to reuse valid on-disk Phase 2
artifacts. `--from-phase 2` also rebuilds the rule-feature catalog.

Before expensive work, the orchestrator writes
`outputs/reports/config_audit.json`. It contains the evaluator contract,
resolved Phase 2 budgets and floors, RB capital feasibility, and active gate
thresholds.

## Data and evaluator

- `data/train_new.csv` is the raw 15m training tape and the only source used
  for rule discovery, OOF cross-fitting, and train-only threshold fitting.
- Validation fitness and selection windows feed Phase 2 and RB only.
- `data/test_new.csv` is the consumed raw diagnostic holdout. Phase 5 computes
  causal indicators from it but never refits rules, weights, or thresholds.
- When a CSV contains both OHLCV and labels, the loader recomputes every
  forward label from OHLCV and rejects any mismatch. Label-only legacy fixtures
  remain supported.
- Raw OHLCV tapes must contain finite numeric bars with valid high/low geometry
  and non-negative volume.
- The supplied `ff_*` indicators are not regenerated by this repository. Their
  source code and causal, as-of-bar construction must be independently audited
  before any release claim.
- The active supplied-feature contract is `config.RULE_ALLOWED_FF_FEATURES`:
  the raw train/test tapes contain 18 allowed `ff_*` columns. The rule-feature
  catalog excludes every other `ff_*` name.
- `FORWARD_CSV_PATH` must point to a strictly newer untouched raw tape.
  It must use the same symbols and contiguous 15m coverage, contain more than
  the label horizon for every symbol, and pass per-symbol ordering checks. A
  run is accepted only when long, short, and the joint portfolio pass the
  minimum-trade and positive bootstrap-CI gates on that forward period.
- `evaluator_v5.ipynb` is the canonical evaluator contract. Its constants and
  dynamic time-exit behavior match the 96-bar CPU pipeline contract.

CSV timestamps are 15m bar-open times. The MTF layer normalizes them to
timezone-naive UTC, rejects duplicate keys and missing constituents, and only
publishes an HTF feature after its candle close is at or before the LWC next
open. Each timeframe has an independent warm-up; unfinished HTF candles never
influence an earlier execution row.

Rule-facing MTF technical features are emitted in the evaluator's bounded
``[-1, 1]`` representation. Positive features use a causal trailing-magnitude
scale, and signed features use the matching signed transform. Absolute
``ATR(14)`` and ``KAMA(10)`` values are retained only for internal label
construction and are excluded from rule discovery. A changed representation
invalidates dependent Phase 2/MTF archives and requires regeneration.

The checked-in `train_new.csv` and `test_new.csv` profile contains the balanced
`BTCUSDT`/`ETHUSDT` universe. Production Phase 2 therefore uses independent
one-symbol BTC/ETH specialist islands with guarded round-based migration; the
RB Governor composes the specialists at portfolio level. A configured
multi-symbol release fails closed if one symbol has no qualifying specialist;
it is never silently converted into an ETH-only product. The legacy global and
clustered modes remain available for controlled experiments.

Every run writes `reports/dataset_manifest.json`,
`reports/experiment_ledger.jsonl`, frozen-strategy stability diagnostics, and
train plus validation baseline reports. Phase 5 reports transaction costs,
turnover, drawdown, tail loss, and descriptive moving-block bootstrap
uncertainty. `test_new.csv` is frozen and never feeds the rule-feature catalog, RB, or
Optuna. A forward tape is accepted at most once per output directory.

Override paths with `DATA_ROOT`, `TRAIN_CSV_PATH`, or `TEST_CSV_PATH` when
needed. The evaluator-facing strategy files are `outputs/long.json` and
`outputs/short.json`; clean copies are written under
`outputs/evaluator_clean/`.

## Configuration contract

`gpu_fuzzy_trader.config.validate_config()` checks the relationships that
matter for results: evaluator fees/capital/leverage/exposure/notional, label
horizon and embargo geometry, stage budgets, population/archive limits,
mutation and support floors, monthly windows, RB risk grids, rule-capital
  feasibility, threshold ordering, risk-tail geometry, and the raw MTF/runtime
  contract.

The maximum holding period is 96 15m bars, or 24 hours. Label generation,
barrier outcomes, force exits, tail drops, the holdout embargo, and validation
purge use this horizon. MTF HWC, MWC, and LWC purges are derived from their
configured horizons and are recorded per fold. Legacy label column names
ending in `_288` remain temporarily for schema compatibility; the values and
runtime behavior use 96 bars.

Strategy, pool, archive, split, rule-feature catalog, and dataset identities
include raw dataset hashes, fitted thresholds, OOF fold boundaries, timeframes,
feature schemas, timestamp semantics, archive hashes, composer parameters, and
holding horizon. Resume and cache loading reject artifacts when that identity
changes.

The canonical exposure contract is 100%. RB permits up to 20 rules and starts
the capital grid at 5%, so the maximum rule count remains feasible before
normalization. Phase 2 uses `PHASE2_USE_TOTAL_RETURN_OBJ=False`; its comments
and descriptions document that active behavior.

The runtime also enforces a few data-dependent safety contracts before the
search starts: a full run must satisfy the configured profitable-symbol floor,
cluster count, and RB distinct-symbol requirement. An explicit debug scope may
cap those effective values to its smaller diagnostic universe. Validation is
throttled by `PHASE2_VAL_SIM_INTERVAL` only when validation is report-only; if
validation contributes to fitness, it is evaluated every generation. The
monthly admission gate fails closed when no rule passes, and Phase 5 reports
the locked strategy without pruning or rewriting it from test-set PnL.

## Outputs

- `phase2_{long,short}_pool.json`: Phase 2 pools.
- `{long,short}.json`: evaluator-facing RB strategies.
- `evaluator_clean/{direction}_evaluator_clean.json`: minimal evaluator JSON.
- `reports/rb_governor_{direction}_report.json`: gate, risk, tail, and
  fail-closed diagnostics.
- `reports/config_audit.json`: effective configuration snapshot.
- `mtf_manifest.json`: raw dataset hashes, feature schemas, adaptive OOF fold
  boundaries, per-fold rows and symbol coverage, role-specific purges,
  base/scaled count gates, eligibility reasons, label thresholds, archive
  hashes, frozen composer parameters, and release policy.
- `reports/test_*`: consumed-test diagnostics, marked
  `acceptance_status=diagnostic_only`.
- `reports/forward_*`: optional forward-candidate reports when
  `FORWARD_CSV_PATH` is configured.
- `reports/strategy_stability_*`: frozen-strategy chronological windows with
  purge and report-only uncertainty diagnostics.

Legacy `selected_features_{long,short}.json` files are ignored and never
modified by the pipeline.

Rejected directions overwrite any stale strategy file with an empty
fail-closed result, so old artifacts cannot be reused accidentally.

## Dashboard

The dependency-free dashboard reads existing artifacts only; it does not load
market data or execute the pipeline. Point it at the same explicit output
directory used for a run:

```bash
.venv/bin/python -m gpu_fuzzy_trader.dashboard --output outputs/run_a
.venv/bin/python -m gpu_fuzzy_trader.dashboard --output outputs/run_a --serve
```

Open `outputs/run_a/dashboard.html`, or use the local URL printed by
`--serve`. It shows direction status, train/validation/test metrics, RB
fail-closed reasons, Phase 2 history when available, and generated report
images. Missing artifacts are shown as unavailable rather than treated as
successful results.

## Hyperparameter search

```bash
.venv/bin/python -m gpu_fuzzy_trader.optuna_search --n-trials 50 --fast
```

Optuna tunes only active Phase 2/RB settings; the data-dependent symbol floor
is kept as a validated dataset contract rather than sampled per trial. It
derives the stage budgets,
validates every patched trial before execution, scores validation and RB tail
metrics only, and records effective values and rejection details. It never
optimizes against Phase 5 or test metrics. Completed trials can be inspected
through `outputs/reports/hyperparameter_correlation.json`.

## Tests

On local or WSL hosts, always set `PYTEST_LOW_MEMORY=1`. It scales down
Hypothesis example counts and releases JAX caches or matplotlib figures only
when a test has actually loaded those optional libraries. Benchmark tests stay
skipped unless you also set `RUN_BENCHMARKS=1`.

Do not run the full GPU project locally without low-memory mode. Prefer
targeted runs while iterating on a change:

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest -q tests/unit/test_config_validation.py
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest -q tests/unit/test_multi_timeframe.py
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest -q tests/unit/test_cpu_engine.py
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest -q tests/unit/test_rb_fail_closed.py
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest -q tests/unit/test_optuna_search.py
```

For complete local validation, use the sequential suite runner:

```bash
npm run test:all
```

It runs CPU/property tests first and direct-JAX tests second, in separate
processes so JAX memory is released between groups.
