# Running the Project

Quick reference for setup, running the GPU-Fuzzy trading pipeline, tests, and evaluation. For architecture and design details, see [README.md](README.md).

---

## Prerequisites

- **Python 3.10+** (3.12 tested)
- **Repository root as working directory** — all paths in `gpu_fuzzy_trader/config.py` are relative to the project root
- **Data files** (included in this repo):
  - `data/train.csv` — training data (Phases 1–4)
  - `data/test.csv` — held-out test data (Phase 5 only)

---

## Setup

### 1. Create a virtual environment (recommended)

```bash
cd /path/to/trading_platform
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**CPU-only minimum** (omit GPU/RL packages; Phase 2/4 use built-in fallbacks):

```bash
pip install pandas numpy scikit-learn matplotlib pyarrow pytest hypothesis
```

| Package                                   | Used by                                                                                        |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `pandas`, `numpy`, `pyarrow`              | Data loading, splits, backtests                                                                |
| `scikit-learn`                            | Phase 1 feature selection (mutual information)                                                 |
| `matplotlib`                              | Reports and equity-curve plots                                                                 |
| `jax`, `jaxlib`                           | Phase 2 GPU backtest engine (falls back to CPU if missing)                                     |
| `evox`, `torch`                           | Phase 2 NSGA-III (reference vectors + niche selection; falls back to NumPy NSGA-II if missing) |
| `stable-baselines3`, `gymnasium`, `torch` | Phase 4 RL (falls back to random search if missing)                                            |
| `pytest`, `hypothesis`                    | Test suite                                                                                     |

### 3. Verify import

```bash
python -c "from gpu_fuzzy_trader.run_pipeline import Pipeline_Orchestrator; print('OK')"
```

---

## Run the full pipeline

For hyperparameter tuning guidance (defaults, performance effects, failure modes), see **[docs/hyperparameters/](docs/hyperparameters/README.md)**.

From the project root:

```bash
python -m gpu_fuzzy_trader.run_pipeline
```

Write outputs to another directory:

```bash
python -m gpu_fuzzy_trader.run_pipeline --output outputs
```

Equivalent entry point (same behavior):

```bash
python -m gpu_fuzzy_trader
```

Run one phase only after its prerequisite files already exist:

```bash
python -m gpu_fuzzy_trader.run_pipeline --phase 1
python -m gpu_fuzzy_trader.run_pipeline --phase 2
python -m gpu_fuzzy_trader.run_pipeline --phase 3
python -m gpu_fuzzy_trader.run_pipeline --phase 4
python -m gpu_fuzzy_trader.run_pipeline --phase 5
```

### What it does

Runs all five phases in order:

1. **Data prep** — load `data/train.csv`, per-symbol 75/25 chronological split → `data/train_75.parquet`, `data/validation_25.parquet`
2. **Phase 1** — direction-specific feature selection
3. **Phase 2** — rule pool generation via **NSGA-III** (`evolution/evox_runner.py` + EvoX; long + short)
4. **Phase 3** — **greedy** rule-set construction + short Pareto refinement → `outputs/long.json`, `outputs/short.json`
5. **Phase 4** — RL risk optimization (TP / SL / capital per rule)
6. **Phase 5** — out-of-sample evaluation on `data/test.csv` (always runs)

On success, a short summary is printed to stdout. Structured timing is appended to `outputs/pipeline.log`.

### Resume / skip behavior

The default `python -m gpu_fuzzy_trader.run_pipeline` command **always reruns Phases 1–4** (and Phase 5) into the default `outputs/` tree. Pass `--output DIR` if you want to write somewhere else.

To skip phases whose outputs are already valid on disk, pass **`--resume`**:

```bash
python -m gpu_fuzzy_trader.run_pipeline --resume
```

To run a single phase, pass `--phase 1` through `--phase 5`. The selected phase will rerun, but it will not auto-run earlier phases, so its prerequisite artifacts must already exist on disk.

### Phase 2 pools (always runs on default CLI)

Phase 2 **always** executes the full NSGA-III search on a default run. It does **not** skip because `pools/phase2_*_pool.json` already exists.

- **Before evolution:** about `PHASE2_ARCHIVE_SEED_FRACTION` (default **35%**) of the population is seeded from chromosomes in the existing `pools/phase2_{long|short}_pool.json` file (if present).
- **After evolution:** the saved pool merges the **best** rules from the previous pool and the new Pareto front (non-dominated ranking, capped by `PHASE2_ARCHIVE_MAX_SIZE`).

`phase2_rule_archive/` is still updated after each Phase 2 run for long-term storage, but initial seeding uses **`pools/` only**, not the archive.

To rebuild data or inspect a specific phase, delete the files for that phase manually or use the matching phase command:

```bash
# Force fresh train/val split from CSV
rm -f data/train_75.parquet data/validation_25.parquet

# Re-run only Phase 2 (long) via single-phase mode
python -m gpu_fuzzy_trader.run_pipeline --phase 2
```

Tune Phase 2 hyperparameters in `gpu_fuzzy_trader/config.py` (`PHASE2_POPULATION_SIZE`, `PHASE2_GENERATIONS`, `PHASE2_ARCHIVE_SEED_FRACTION`, etc.).

---

## Configuration

Edit `gpu_fuzzy_trader/config.py` before running. Common settings:

| Setting                        | Default                | Purpose                                                                        |
| ------------------------------ | ---------------------- | ------------------------------------------------------------------------------ |
| `TRAIN_CSV_PATH`               | `data/train.csv`       | Training CSV                                                                   |
| `TEST_CSV_PATH`                | `data/test.csv`        | Test CSV                                                                       |
| `PHASE1_TOP_K_FEATURES`        | `30`                   | Features per direction                                                         |
| `PHASE2_ALGORITHM`             | `"NSGA3"`              | Fixed; NSGA-III when EvoX is installed                                         |
| `PHASE2_POPULATION_SIZE`       | `200`                  | Phase 2 population                                                             |
| `PHASE2_GENERATIONS`           | `100`                  | Phase 2 generations                                                            |
| `PHASE2_POOL_DIR`              | `pools/`               | Persistent Phase 2 pool/history files in the project root                      |
| `PHASE2_ARCHIVE_DIR`           | `phase2_rule_archive/` | Persistent Phase 2 archive in the project root                                 |
| `PHASE2_ARCHIVE_MAX_SIZE`      | `500`                  | Max archive size per direction                                                 |
| `PHASE2_ARCHIVE_SEED_FRACTION` | `0.35`                 | Fraction of Phase 2 population seeded from archive                             |
| `PHASE3_REFINE_GENERATIONS`    | `15`                   | Refinement generations after greedy                                            |
| `PHASE3_REFINE_POP_SIZE`       | `40`                   | Refinement population size                                                     |
| `PHASE3_USE_GPU`               | `False`                | Batched rule-set eval via `GPUBacktestEngine` (enable after parity tests pass) |
| `PHASE4_TOTAL_TIMESTEPS`       | `500000`               | Phase 4 RL steps                                                               |

---

## Outputs

After a full run, expect artifacts under `outputs/`:

| Path                                                   | Phase                           |
| ------------------------------------------------------ | ------------------------------- |
| `outputs/selected_features_{long,short}.json`          | 1                               |
| `pools/phase2_{long,short}_pool.json`                  | 2                               |
| `pools/phase2_{long,short}_history.json`               | 2                               |
| `phase2_rule_archive/phase2_{long,short}_archive.json` | 2 persistent archive            |
| `outputs/long.json`, `outputs/short.json`              | 3–4                             |
| `outputs/reports/test_*`                               | 5 (metrics, equity plots, CSVs) |
| `outputs/pipeline.log`                                 | All phases (JSON lines)         |

Strategy files `outputs/long.json` and `outputs/short.json` are compatible with **`evaluator_v3.ipynb`** at the repo root.

---

## Evaluate with the notebook

1. Run the pipeline (or copy existing `outputs/long.json` and `outputs/short.json`).
2. Open `evaluator_v3.ipynb` in Jupyter.
3. Point the notebook at the generated strategy JSON and `data/test.csv` (or `data/train.csv` for in-sample checks).

```bash
jupyter notebook evaluator_v3.ipynb
```

---

## Run tests

From the project root (requires `pytest` and `hypothesis`):

```bash
# All tests (GPU/JAX tests skipped automatically without JAX)
PYTHONPATH=. pytest tests/ --hypothesis-seed=42

# Unit tests only
pytest tests/unit/

# Property-based tests only
pytest tests/property/ --hypothesis-seed=42

# Verbose / single file
pytest tests/ --hypothesis-seed=42 -v
pytest tests/unit/test_cpu_engine.py -v
```

GPU/JAX tests are skipped automatically when JAX is not installed.

---

## Programmatic usage

Run the orchestrator from Python (same as the CLI):

```python
from gpu_fuzzy_trader.run_pipeline import Pipeline_Orchestrator

orchestrator = Pipeline_Orchestrator()
results = orchestrator.run()
print(results["phase5"])  # OOS metrics
```

Run individual phases — see [README.md § Running Individual Phases](README.md#running-individual-phases).

---

## Troubleshooting

| Issue                                                       | What to do                                                                                                                     |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `ModuleNotFoundError: No module named 'pandas'` (or others) | Activate venv and install dependencies (see Setup)                                                                             |
| `FileNotFoundError` for `data/train.csv`                    | Run from project root; ensure data files exist                                                                                 |
| Phase 2 very slow on CPU                                    | Install `jax` + `jaxlib` and `evox` + `torch`; optional GPU/CUDA build for faster backtests                                    |
| Phase 2 uses NSGA-II instead of NSGA-III                    | Install `evox` and `torch`; check `phase2_*_history.json` — `"NSGA-II (fallback)"` means EvoX was unavailable                  |
| Stale Phase 2 pools after code changes                      | `rm pools/phase2_{long,short}_{pool,history}.json` and re-run                                                                  |
| Phase 3 still slow                                          | Lower `PHASE3_REFINE_GENERATIONS` / `PHASE3_REFINE_POP_SIZE`, or set `PHASE3_USE_GPU=True` after parity tests for batched eval |
| Phase 4 uses random search                                  | Install `stable-baselines3`, `gymnasium`, and `torch` for DDPG/PPO                                                             |
| Want fresh OOS only                                         | Keep `outputs/long.json` / `short.json`; re-run pipeline (Phase 5 always runs)                                                 |
