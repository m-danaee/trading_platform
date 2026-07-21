# Trading Platform (GPU-Fuzzy Trading Pipeline)

A research-focused algorithmic trading pipeline that builds and evaluates long/short rule sets using feature selection, evolutionary search, and out-of-sample validation.

## What this project does

The pipeline runs in five stages:

1. **Data prep** — load and split market data.
2. **Phase 1** — feature selection.
3. **Phase 2** — rule pool generation (NSGA search, GPU-accelerated when available).
4. **Phase 3–4** — rule-set composition and risk tuning (or RB Governor path).
5. **Phase 5** — out-of-sample evaluation on held-out data.

Primary implementation lives in `gpu_fuzzy_trader/`, orchestrated by `gpu_fuzzy_trader/run_pipeline.py`.

## Repository layout

```text
gpu_fuzzy_trader/   Core pipeline package
tests/              Unit, property, and benchmark tests
RUN.md              Detailed runbook (setup, CLI, tests, troubleshooting)
evaluator_v5.ipynb  Canonical evaluator notebook for rule-set evaluation
requirements.txt    Base dependencies
requirements-gpu.txt GPU JAX plugin dependencies
```

## Requirements

- Python **3.10+** (3.12 tested)
- A virtual environment (`.venv`) is recommended
- Input data CSVs:
  - `data/train.csv` (training/validation source)
  - `data/test.csv` (held-out OOS evaluation for Phase 5)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional NVIDIA GPU setup (after `requirements.txt`):

```bash
pip install -r requirements-gpu.txt
```

For Colab CUDA 12:

```bash
pip install -U "jax[cuda12]==0.10.1"
```

## Run the pipeline

From repository root:

```bash
source .venv/bin/activate
python -m gpu_fuzzy_trader.run_pipeline
```

Useful options:

- `--output <dir>`: write outputs to a custom directory
- `--resume`: skip completed valid phases when possible
- `--phase {1..5}`: run only one phase (requires prior artifacts)
- `--from-phase 2`: start from phase 2 (requires phase 1 outputs)

You can also use:

```bash
python -m gpu_fuzzy_trader
```

## Running tests

Use low-memory mode locally/WSL to reduce OOM risk:

```bash
source .venv/bin/activate
PYTEST_LOW_MEMORY=1 PYTHONPATH=. pytest tests/unit/ --hypothesis-seed=42
```

Run all tests (high-memory environments):

```bash
PYTEST_LOW_MEMORY=1 PYTHONPATH=. pytest tests/ --hypothesis-seed=42
```

## Outputs

By default, results are written to `outputs/`, including:

- phase artifacts (selected features, evolved pools, strategies)
- reports in `outputs/reports/`
- structured timing logs in `outputs/pipeline.log`

## Important notes

- Full runs can be memory intensive; prefer Colab GPU or high-memory machines.
- Keep evaluation aligned with `evaluator_v5.ipynb`.
- If you need full operational details and troubleshooting, use `RUN.md`.
