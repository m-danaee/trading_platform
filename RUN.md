# Running the project

Use the repository virtual environment for every command. Long Phase 2/RB
runs are intended for a CUDA host; do not run the full project on a
memory-constrained WSL machine.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -c "from gpu_fuzzy_trader.run_pipeline import Pipeline_Orchestrator; print('OK')"
```

`main.ipynb` installs the appropriate JAX CUDA extra after the shared
requirements: `jax[cuda12]==0.10.1` on Colab T4 and
`jax[cuda13]==0.10.1` on local CUDA 13 hosts.

## RTX 4050 + 8-core CPU execution policy

The runtime uses a hybrid policy tuned for a 6-GiB RTX 4050 and an 8-core
host:

- Phase 1 feature transforms, dataframe preparation, exact rule-set scoring,
  RB risk tuning, and Phase 5 evaluation stay on CPU.
- Phase 2 uses the optimized CPU batch evaluator for the default large
  (~90k-bar) train window. This avoids a slower full-window GPU scan on this
  hardware and also supplies reference-quality ranking metrics.
- JAX/GPU is retained for smaller windows and larger dense batches, where
  vectorized matching and the event-driven sparse kernel can amortize launch
  overhead. FP32, int8 feature storage, bounded batches, and lazy compilation
  keep VRAM use safe.

The large-window route is enabled by default. To force JAX for an A/B test,
set `gpu_fuzzy_trader.config.PHASE2_GPU_CPU_ROUTE_LARGE_DATA = False`; adjust
`PHASE2_GPU_CPU_ROUTE_MIN_BARS` and `PHASE2_GPU_CPU_ROUTE_MAX_BATCH` only when
benchmarking a different CPU/GPU combination.

## Colab T4 notebook execution

Run `main.ipynb` for the Colab T4 path. When `/content` is detected, the
worker subprocess keeps the Phase 2 ranking on JAX/GPU, caps the default batch
to 64 through the notebook environment, and uses scan unroll 16 to reduce
host-RAM compile pressure. CSVs and outputs are staged on local `/content`
storage; the notebook syncs results and the Phase 2 archive back to Drive.

## Pipeline commands

```bash
.venv/bin/python -m gpu_fuzzy_trader.run_pipeline
.venv/bin/python -m gpu_fuzzy_trader.run_pipeline --output outputs/
.venv/bin/python -m gpu_fuzzy_trader.run_pipeline --resume
.venv/bin/python -m gpu_fuzzy_trader.run_pipeline --phase 1
.venv/bin/python -m gpu_fuzzy_trader.run_pipeline --phase 2
.venv/bin/python -m gpu_fuzzy_trader.run_pipeline --phase 5
```

The production order is data preparation, Phase 1 feature selection,
independent symbol-specialist Phase 2 rule-pool evolution, RB Governor
selection/capital sizing, and Phase 5 evaluation. TP/SL and horizon are part of
the immutable strategy identity; production RB does not rescue a candidate by
changing its exit geometry. Phase 5 reports the consumed
test tape diagnostically; only a strictly newer `FORWARD_CSV_PATH` can produce
an acceptance decision. Existing integrations that use `--phase 3` or `--phase 4` remain
compatible: both normalize to the RB Governor and return the historical result
keys. They are intentionally omitted from the production command list above.

The orchestrator validates configuration before expensive execution and writes
`outputs/reports/config_audit.json`. The report includes evaluator constants,
split geometry, effective stage budgets, sample budgets, island floors, RB
capital feasibility, and all active gate thresholds.

Missing or invalid Phase 2 output is a direction-specific failure. RB writes a
fresh empty strategy with `deployment_accepted: false` and a machine-readable
reason. Phase 5 receives only non-empty strategies from the current run, so
stale files are not evaluated.

Configuration preflight rejects a full-run symbol universe that cannot satisfy
the configured Phase 2 profitable-symbol floor, cluster count, or RB distinct-
symbol requirement. An explicit debug scope may cap those effective values to
its smaller diagnostic universe. Validation cadence is throttled only when
validation does not affect fitness; otherwise it runs every generation. The
monthly admission gate is fail-closed when every rule fails, and Phase 5 is
report-only: it never prunes or rewrites strategies using held-out test
performance.

## Data and evaluator

- `data/train_new.csv` feeds Phase 1 and Phase 2. Its OHLCV columns are used
  to derive the forward labels required by the backtest.
- Validation fitness and selection windows feed Phase 2 and RB only.
- `data/test_new.csv` is a consumed Phase 5 diagnostic holdout and uses the
  same OHLCV/features schema. Set `FORWARD_CSV_PATH` to an untouched tape
  whose first timestamp is strictly after the complete test tape for an
  acceptance check.
- `evaluator_v5.ipynb` is read-only and is the final evaluation authority.

The checked-in `train_new.csv` and `test_new.csv` profile contains the balanced
`BTCUSDT`/`ETHUSDT` universe. The production Phase 2 path runs independent
one-symbol BTC/ETH islands with recipient-validated round-based migration;
portfolio-level RB composition is responsible for joint coverage. A
multi-symbol direction fails closed when one symbol has no qualifying
specialist. Global and clustered modes remain available for controlled
experiments.

Each run writes a dataset manifest, append-only experiment ledger, nested
outer-fold report, and baseline/ablation reports. The consumed test file is
never a tuning input. A forward acceptance tape is locked after its first
evaluation in an output directory.

The evaluator-facing strategy files are `outputs/long.json` and
`outputs/short.json`; clean copies are written below
`outputs/evaluator_clean/`.

## Dashboard

Render a read-only dashboard from an existing run directory:

```bash
.venv/bin/python -m gpu_fuzzy_trader.dashboard --output outputs/
.venv/bin/python -m gpu_fuzzy_trader.dashboard --output outputs/ --serve
```

The first command writes `dashboard.html`; the second also serves the output
directory locally. The dashboard consumes only JSON/CSV/PNG artifacts and
gracefully displays missing or fail-closed directions.

## Validation-only Optuna

```bash
.venv/bin/python optuna_search.py --n-trials 50 --fast
```

Optuna tunes active Phase 2/RB keys; the data-dependent symbol floor remains a
validated dataset contract rather than a trial parameter. It derives dependent
stage budgets, validates each trial, and scores RB validation plus tail metrics
only. It never uses Phase 5/test metrics. Trial parameters, derived effective
configuration, and rejection/error details are stored as trial attributes; the
correlation report is `outputs/reports/hyperparameter_correlation.json`.

## Tests

On local or WSL hosts, always set `PYTEST_LOW_MEMORY=1`. It scales down
Hypothesis example counts, clears JAX caches between tests, and closes
matplotlib figures to reduce peak memory. Benchmark tests stay skipped unless
you also set `RUN_BENCHMARKS=1`.

### All tests (low memory)

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest -q
```

The full suite can still take a long time on a memory-constrained machine.
Prefer targeted runs while iterating on a change.

### Targeted tests

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest -q tests/unit/test_config_validation.py
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest -q tests/unit/test_rb_fail_closed.py
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest -q tests/unit/test_optuna_search.py
```
