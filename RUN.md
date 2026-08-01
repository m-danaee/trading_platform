# Running the project

Use the repository virtual environment for every command. Long Phase 2/RB
runs are intended for Colab or another GPU host; do not run the full project
on a memory-constrained WSL machine.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -c "from gpu_fuzzy_trader.run_pipeline import Pipeline_Orchestrator; print('OK')"
```

Install the appropriate JAX GPU package separately when using a CUDA host.

## Pipeline commands

```bash
.venv/bin/python -m gpu_fuzzy_trader.run_pipeline
.venv/bin/python -m gpu_fuzzy_trader.run_pipeline --output outputs/run_a
.venv/bin/python -m gpu_fuzzy_trader.run_pipeline --resume
.venv/bin/python -m gpu_fuzzy_trader.run_pipeline --phase 1
.venv/bin/python -m gpu_fuzzy_trader.run_pipeline --phase 2
.venv/bin/python -m gpu_fuzzy_trader.run_pipeline --phase 5
```

The production order is data preparation, Phase 1 feature selection, Phase 2
rule-pool evolution, RB Governor selection/risk tuning, and Phase 5 OOS
evaluation. Existing integrations that use `--phase 3` or `--phase 4` remain
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
- `data/test_new.csv` is reserved for Phase 5 and uses the same OHLCV/features
  schema.
- `evaluator_v5.ipynb` is read-only and is the final evaluation authority.

The checked-in `train_new.csv` and `test_new.csv` profile contains the balanced
`BTCUSDT`/`ETHUSDT` universe. The default Phase 2 path is one global search with
a two-symbol robustness target. Cluster mode remains available for larger
universes; its active cluster count must fit the data preflight.

The evaluator-facing strategy files are `outputs/long.json` and
`outputs/short.json`; clean copies are written below
`outputs/evaluator_clean/`.

## Dashboard

Render a read-only dashboard from an existing run directory:

```bash
.venv/bin/python -m gpu_fuzzy_trader.dashboard --output outputs/run_a
.venv/bin/python -m gpu_fuzzy_trader.dashboard --output outputs/run_a --serve
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
