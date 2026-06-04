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
pip install "numpy>=1.26,<2.4" pandas scikit-learn matplotlib pyarrow pytest hypothesis
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

### Older x86 CPUs (NumPy `X86_V2` error)

If import fails with:

```text
RuntimeError: NumPy was built with baseline optimizations:
(X86_V2) but your machine doesn't support:
(X86_V2).
```

your host lacks **x86-64-v2** (typical on pre-2009 hardware or some cheap VPS). NumPy **2.4+** wheels target that baseline.

**Fix** (inside `.venv` on the server):

```bash
pip install --upgrade "numpy>=1.26,<2.4"
python -c "import numpy; print(numpy.__version__)"
python -c "from gpu_fuzzy_trader.run_pipeline import Pipeline_Orchestrator; print('OK')"
```

If you still need the newest NumPy on that machine, build from source for your CPU:

```bash
pip install --no-binary numpy "numpy>=1.26,<2.4" -Csetup-args=-Dcpu-baseline=none
```

Re-run `pip install -r requirements.txt` afterward so other packages stay aligned.

### JAX / AVX error on old CPUs

If you see:

```text
RuntimeError: This version of jaxlib was built using AVX instructions,
which your CPU and/or operating system do not support.
```

the pipeline **automatically falls back to `CPUBacktestEngine`** (Phase 2/3/5). You do not need JAX on that host. After updating the repo, verify:

```bash
python -c "from gpu_fuzzy_trader.run_pipeline import Pipeline_Orchestrator; print('OK')"
```

Optional: remove JAX entirely on CPU-only servers (faster imports, no failed probe):

```bash
pip uninstall jax jaxlib -y
```

Phase 2 evolution still runs via **Numba CPU**; only GPU-accelerated backtests are disabled.

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

### Optuna config tuner (low-RAM profile)

Automated search over high-impact `config.py` knobs using **validation** metrics (test is logged only, not optimized). Each trial runs **Phases 2–5** with Phase 1 features copied from a baseline run.

The tuner CLI pins **CPU-only** execution (`JAX_PLATFORMS=cpu`, `PHASE2_USE_GPU=False`, `PHASE3_USE_GPU=False` via the `low_ram` profile). Full pipeline runs on **Colab GPU** are unchanged — use `main.ipynb` or `run_pipeline` there with default `PHASE2_USE_GPU=True`.

#### Local CPU-only (WSL / no GPU)

**Prerequisites** (once):

```bash
source .venv/bin/activate
python -m gpu_fuzzy_trader.run_pipeline --phase 1
# or full pipeline into outputs/
```

**Run tuning** (2-core / 4GB friendly defaults: fixed pop=100/gen=50, searches CV folds {2,3}, Phase 2–4 gates, pool-size objective penalty, CPU engines):

```bash
python -m gpu_fuzzy_trader.tuning \
  --baseline-output outputs \
  --study-dir tuning_studies/low_ram \
  --n-trials 2 \
  --profile low_ram \
  --seed 42
```

`--force-cpu` is on by default. Use `--no-force-cpu` only if you intentionally want JAX GPU on the tuning host.

Confirm CPU mode in the log line `Tuning runtime: JAX_PLATFORMS=cpu ... trial PHASE2_USE_GPU=False`.

#### Colab GPU verification

1. Copy **generalization knobs** from `tuning_studies/low_ram/best_config.json` → `params` or `merged_config` into `gpu_fuzzy_trader/config.py` (gates, CV floors, pop/gen, etc.).
2. On Colab, **keep** `PHASE2_USE_GPU=True` (default) and do **not** copy tuning-only CPU caps (`PHASE2_USE_GPU=False`, small pop caps) unless you want a slow run.
3. Run `main.ipynb` or `python -m gpu_fuzzy_trader.run_pipeline` and confirm: `Phase 2 using GPUBacktestEngine (backend: gpu)`.
4. Final acceptance: **`evaluator_v3.ipynb`** (same backtest contract as Phase 5). CPU trial scores may differ slightly from GPU Phase 2.

Outputs under `tuning_studies/low_ram/`:

| File | Purpose |
|------|---------|
| `optuna.db` | SQLite study storage |
| `best_config.json` | `params` (Optuna knobs), `merged_config` (profile + knobs for handoff) |
| `trials_summary.csv` | Val/test metrics per trial |
| `trial_N/` | Isolated pipeline outputs per trial |

Re-run a single trial path manually (uses current `config.py`, not trial overlay):

```bash
python -m gpu_fuzzy_trader.run_pipeline --output tuning_studies/low_ram/trial_0 --from-phase 2
```

---

## Configuration

Edit `gpu_fuzzy_trader/config.py` before running. Common settings:

| Setting                        | Default                | Purpose                                                   |
| ------------------------------ | ---------------------- | --------------------------------------------------------- |
| `TRAIN_CSV_PATH`               | `data/train.csv`       | Training CSV                                              |
| `TEST_CSV_PATH`                | `data/test.csv`        | Test CSV                                                  |
| `PHASE1_TOP_K_FEATURES`        | `30`                   | Features per direction                                    |
| `PHASE2_ALGORITHM`             | `"NSGA3"`              | Fixed; NSGA-III when EvoX is installed                    |
| `PHASE2_POPULATION_SIZE`       | `200`                  | Phase 2 population                                        |
| `PHASE2_GENERATIONS`           | `200`                  | Phase 2 generations                                       |
| `PHASE2_POOL_DIR`              | `pools/`               | Persistent Phase 2 pool/history files in the project root |
| `PHASE2_ARCHIVE_DIR`           | `phase2_rule_archive/` | Persistent Phase 2 archive in the project root            |
| `PHASE2_ARCHIVE_MAX_SIZE`      | `500`                  | Max archive size per direction                            |
| `PHASE2_ARCHIVE_SEED_FRACTION` | `0.35`                 | Fraction of Phase 2 population seeded from archive        |
| `PHASE3_REFINE_GENERATIONS`    | `80`                   | Refinement generations after greedy                       |
| `PHASE3_REFINE_POP_SIZE`       | `100`                  | Refinement population size                                |
| `PHASE3_USE_PARALLEL_BATCH`    | `True`                 | Parallel CPU batch eval for greedy + NSGA-II              |
| `PHASE3_USE_GPU`               | `False`                | JAX cached-mask batch path (enable after parity tests)    |
| `PHASE4_N_TRIALS`              | `200`                  | Phase 4 Optuna trials                                     |
| `PHASE4_WF_SPLITS`             | `4`                    | Phase 4 walk-forward validation windows                   |
| `PHASE4_N_JOBS`                | `1`                    | Phase 4 parallel Optuna workers (see phase4_wf_risk.md)   |

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

| Issue                                                       | What to do                                                                                                                                             |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `ModuleNotFoundError: No module named 'pandas'` (or others) | Activate venv and install dependencies (see Setup)                                                                                                     |
| `FileNotFoundError` for `data/train.csv`                    | Run from project root; ensure data files exist                                                                                                         |
| Phase 2 very slow on CPU                                    | Install `jax` + `jaxlib` and `evox` + `torch`; optional GPU/CUDA build for faster backtests                                                            |
| Phase 2 uses NSGA-II instead of NSGA-III                    | Install `evox` and `torch`; check `phase2_*_history.json` — `"NSGA-II (fallback)"` means EvoX was unavailable                                          |
| Stale Phase 2 pools after code changes                      | `rm pools/phase2_{long,short}_{pool,history}.json` and re-run                                                                                          |
| Phase 3 still slow                                          | Ensure `PHASE3_USE_PARALLEL_BATCH=True`; raise `PHASE3_BATCH_WORKERS`; then increase `PHASE3_REFINE_*` after profiling; optional `PHASE3_USE_GPU=True` |
| Phase 4 uses random search                                  | Install `stable-baselines3`, `gymnasium`, and `torch` for DDPG/PPO                                                                                     |
| Want fresh OOS only                                         | Keep `outputs/long.json` / `short.json`; re-run pipeline (Phase 5 always runs)                                                                         |
