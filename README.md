# GPU Fuzzy Trader

This project evolves fuzzy trading rules in Phase 2, selects and risk-tunes
teams with the RB Governor, and evaluates only evaluator-compatible strategy
JSON with the `evaluator_v5.ipynb` contract. Entries use a causal
multi-timeframe trend hierarchy: 4h and 1h regimes grant directional
permission, while a 15m pullback reversal triggers the entry search.

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

1. Causal enrichment fits regime thresholds from the training tape only and
   produces separate enriched train, test, and optional forward CSVs.
2. Data preparation loads the enriched training tape and builds cached
   train/validation splits.
3. Phase 1 selects direction-specific 15m `ff_*` confirmations. Fixed context
   columns are excluded from feature selection and chromosomes.
4. Phase 2 generates and admits rule pools using island-resolved floors while
   applying the mandatory context mask to every candidate.
5. RB Governor is the single production selection and risk path. It preserves
   the fixed context policy and keeps existing capital sizing behavior.
6. Phase 5 records diagnostics on the consumed enriched test tape; an optional
   strictly newer enriched forward tape is the only acceptance tape.

The RB Governor treats entry conditions, symbol scope, TP/SL, horizon, and cost
model as one immutable strategy identity. Production RB may size capital, but
it does not rescue rejected Phase 2 rules by silently searching a new TP/SL
envelope.

The first release is trend-following only. Long entries require Bullish 4h and
1h states; short entries require Bearish 4h and 1h states. Range, Noisy,
opposite, and unavailable states block entries. The current 15m state must also
reverse in the permitted direction after an opposite state occurred within the
previous 24 completed 15m states.

These two direction-specific conditions are mandatory execution policy in
every exported rule:

```text
Long:  [tf_permission_long] IS Active (1)
       [lwc_pullback_reversal_long] IS Active (1)
Short: [tf_permission_short] IS Active (1)
       [lwc_pullback_reversal_short] IS Active (1)
```

They are not NSGA-III genes. Mutation, crossover, feature selection, and
condition-count limits cannot remove them. `MIN_CONDITIONS` and
`MAX_CONDITIONS` count only evolved 15m confirmations.

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

The active context contract uses a 24-bar LWC pullback lookback and frozen
train-only classifier quantiles of 55th/55th/45th percentile. Changing this
contract changes dataset and strategy identity: re-enrich every train, test, and
forward tape, then rebuild splits and Phase 1/Phase 2 artifacts. Do not reuse
enriched CSVs or derived artifacts from the previous contract.

Generate enriched tapes before running the multi-timeframe pipeline. Raw tapes
are never overwritten:

```bash
.venv/bin/python -m gpu_fuzzy_trader.data.trend_context \
  --train data/train_new.csv \
  --test data/test_new.csv \
  --forward data/forward.csv
```

Omit `--forward` until a strictly newer untouched tape is available. The
default outputs are:

```text
data/enriched/train_new_hwc_mwc_lwc.csv
data/enriched/test_new_hwc_mwc_lwc.csv
data/enriched/forward_hwc_mwc_lwc.csv
data/enriched/trend_context_manifest.json
```

The normal pipeline uses those enriched train/test tapes by default. Override
paths only for a controlled experiment; Phase 5 explicitly rejects a raw,
non-enriched test or forward tape:

```bash
TRAIN_CSV_PATH=data/enriched/train_new_hwc_mwc_lwc.csv \
TEST_CSV_PATH=data/enriched/test_new_hwc_mwc_lwc.csv \
FORWARD_CSV_PATH=data/enriched/forward_hwc_mwc_lwc.csv \
.venv/bin/python -m gpu_fuzzy_trader.run_pipeline
```

Leave `FORWARD_CSV_PATH` unset when no untouched forward tape exists.

```bash
.venv/bin/python -m gpu_fuzzy_trader.run_pipeline
.venv/bin/python -m gpu_fuzzy_trader.run_pipeline --output outputs/run_a
.venv/bin/python -m gpu_fuzzy_trader.run_pipeline --resume
.venv/bin/python -m gpu_fuzzy_trader.run_pipeline --phase 1
.venv/bin/python -m gpu_fuzzy_trader.run_pipeline --phase 2
.venv/bin/python -m gpu_fuzzy_trader.run_pipeline --from-phase 2
.venv/bin/python -m gpu_fuzzy_trader.run_pipeline --phase 5
```

By default the orchestrator re-runs Phase 1, Phase 2, and RB. Pass
`--resume` to reuse valid on-disk Phase 1/2 artifacts. Use `--from-phase 2`
when Phase 1 outputs already exist under `--output`.

Before expensive work, the orchestrator writes
`outputs/reports/config_audit.json`. It contains the evaluator contract,
resolved Phase 2 budgets and floors, RB capital feasibility, and active gate
thresholds.

## Data and evaluator

- `data/train_new.csv` is the raw training tape and the only source used to fit
  pooled regime thresholds. Thresholds are frozen for every later tape.
- `data/enriched/train_new_hwc_mwc_lwc.csv` feeds Phase 1 and Phase 2 after
  enrichment. Its OHLCV columns are used to derive forward labels separately
  within the training research boundary.
- Validation fitness and selection windows feed Phase 2 and RB only.
- `data/test_new.csv` is the raw consumed diagnostic holdout. Phase 5 loads its
  enriched counterpart and never fits thresholds, selects rules, or rewrites
  strategies from test results.
- `FORWARD_CSV_PATH` must point to an enriched, strictly newer untouched tape.
  A run is accepted only when long, short, and the joint portfolio are all
  profitable on that forward period.
- `evaluator_v5.ipynb` is the canonical evaluator contract. Its constants and
  dynamic time-exit behavior match the 96-bar CPU pipeline contract.

CSV timestamps are timezone-naive 15m bar-open times aligned to
`:00/:15/:30/:45`. Enrichment rejects duplicate `(datetime, symbol)` keys,
gaps, off-grid timestamps, and ambiguous history/target overlap. A signal row
uses the HWC/MWC state available at its next-open execution time: the 10:45
signal can use a 1h bar closing at 11:00 for an 11:00 entry, and the 11:45
signal can use the 4h bar closing at 12:00 for a 12:00 entry. Rolling warm-up
may use preceding research history, but history rows are not emitted or scored.

The checked-in `train_new.csv` and `test_new.csv` profile contains the balanced
`BTCUSDT`/`ETHUSDT` universe. Production Phase 2 therefore uses independent
one-symbol BTC/ETH specialist islands with guarded round-based migration; the
RB Governor composes the specialists at portfolio level. A configured
multi-symbol release fails closed if one symbol has no qualifying specialist;
it is never silently converted into an ETH-only product. The legacy global and
clustered modes remain available for controlled experiments.

Every run writes `reports/dataset_manifest.json`,
`reports/experiment_ledger.jsonl`, nested outer-fold diagnostics, and baseline
reports. `test_new.csv` is frozen and never feeds feature selection, RB, or
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
feasibility, threshold ordering, risk-tail geometry, and the complete trend
context contract.

The maximum holding period is 96 15m bars, or 24 hours. Label generation,
barrier outcomes, force exits, tail drops, holdout embargo, purged-walk-forward
embargo, and validation-half purge use the same 96-bar horizon. Legacy label
column names ending in `_288` remain temporarily for schema compatibility; the
values and runtime behavior use 96 bars.

Strategy, pool, archive, split, feature-selection, and dataset identities
include the context algorithm, fitted thresholds, threshold-fitting source and
interval, raw/history/enriched hashes, timeframes, timestamp semantics, state
codes, pullback lookback, permission policy, and holding horizon. Resume and
cache loading reject artifacts when that identity changes.

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

- `selected_features_{long,short}.json`: Phase 1 features.
- `phase2_{long,short}_pool.json`: Phase 2 pools.
- `{long,short}.json`: evaluator-facing RB strategies.
- `evaluator_clean/{direction}_evaluator_clean.json`: minimal evaluator JSON.
- `reports/rb_governor_{direction}_report.json`: gate, risk, tail, and
  fail-closed diagnostics.
- `reports/config_audit.json`: effective configuration snapshot.
- `data/enriched/trend_context_manifest.json`: context thresholds, source and
  history lineage, enriched hashes, timeframes, and timestamp policy.
- `reports/test_*`: consumed-test diagnostics, marked
  `acceptance_status=diagnostic_only`.
- `reports/forward_*`: optional forward-candidate reports when
  `FORWARD_CSV_PATH` is configured.

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
Hypothesis example counts, clears JAX caches between tests, and closes
matplotlib figures to reduce peak memory. Benchmark tests stay skipped unless
you also set `RUN_BENCHMARKS=1`.

Do not run the full GPU project locally without low-memory mode. Prefer
targeted runs while iterating on a change:

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest -q tests/unit/test_config_validation.py
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest -q tests/unit/test_multi_timeframe.py
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest -q tests/unit/test_cpu_engine.py
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest -q tests/unit/test_rb_fail_closed.py
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest -q tests/unit/test_optuna_search.py
```

The full suite is available with `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest -q`
but can still take a long time on memory-constrained machines.
