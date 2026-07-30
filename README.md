# GPU Fuzzy Trader

This project evolves fuzzy trading rules in Phase 2, selects and risk-tunes
teams with the RB Governor, and evaluates only evaluator-compatible strategy
JSON with the read-only `evaluator_v5.ipynb` contract.

Long Phase 2 and RB runs are intended for Colab or another GPU host. Do not
run the full pipeline on a memory-constrained WSL machine. See [RUN.md](RUN.md)
for extended setup and command reference.

## Repository layout

```text
gpu_fuzzy_trader/   Core pipeline package
tests/              Unit, property, and benchmark tests
RUN.md              Detailed runbook (setup, CLI, tests, troubleshooting)
evaluator_v5.ipynb  Canonical evaluator notebook for rule-set evaluation
requirements.txt    Base dependencies
requirements-gpu.txt GPU JAX plugin dependencies
```

## Pipeline

1. Data preparation loads `train_new.csv` and builds cached train/validation splits.
2. Phase 1 selects direction-specific features.
3. Phase 2 generates and admits rule pools using island-resolved floors.
4. RB Governor is the single production selection and risk path.
5. Phase 5 evaluates the current RB outputs out of sample on `test_new.csv`.

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

Install the matching JAX GPU stack separately on CUDA hosts:

```bash
# Colab T4 / CUDA 12
.venv/bin/pip install -U "jax[cuda12]==0.10.1"

# Local WSL/Linux with CUDA 13 (e.g. RTX 4050)
.venv/bin/pip install -r requirements-gpu.txt
```

## Run

Use the repository virtual environment for every command.

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

- `data/train_new.csv` feeds Phase 1 and Phase 2. Its OHLCV columns are used
  to derive the forward labels required by the backtest.
- Validation fitness and selection windows feed Phase 2 and RB only.
- `data/test_new.csv` is reserved for Phase 5 and uses the same OHLCV/features
  schema.
- `evaluator_v5.ipynb` is read-only and is the final evaluation authority.

Override paths with `DATA_ROOT`, `TRAIN_CSV_PATH`, or `TEST_CSV_PATH` when
needed. The evaluator-facing strategy files are `outputs/long.json` and
`outputs/short.json`; clean copies are written under
`outputs/evaluator_clean/`.

## Configuration contract

`gpu_fuzzy_trader.config.validate_config()` checks the relationships that
matter for results: evaluator fees/capital/leverage/exposure/notional, label
horizon and embargo geometry, stage budgets, population/archive limits,
mutation and support floors, monthly windows, RB risk grids, rule-capital
feasibility, threshold ordering, and risk-tail geometry.

The canonical exposure contract is 100%. RB permits up to 20 rules and starts
the capital grid at 5%, so the maximum rule count remains feasible before
normalization. Phase 2 uses `PHASE2_USE_TOTAL_RETURN_OBJ=False`; its comments
and descriptions document that active behavior.

The runtime also enforces a few data-dependent safety contracts before the
search starts: the active universe must satisfy the configured profitable-
symbol floor, cluster count, and RB distinct-symbol requirement. Validation is
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

Optuna tunes only active Phase 2/RB settings. It derives the stage budgets,
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
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest -q tests/unit/test_rb_fail_closed.py
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest -q tests/unit/test_optuna_search.py
```

The full suite is available with `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest -q`
but can still take a long time on memory-constrained machines.
