# Tech Stack

## Language & Runtime

- **Python 3.10+** (3.12 tested)
- Project root is the working directory — all paths in `config.py` are relative to root

## Core Dependencies

| Package | Purpose |
|---------|---------|
| `pandas`, `numpy`, `pyarrow` | Data loading, splits, backtests |
| `scikit-learn` | Phase 1 feature selection (mutual information) |
| `matplotlib` | Reports and equity-curve plots (Agg backend) |
| `jax`, `jaxlib` | Phase 2 GPU backtest engine (optional, falls back to CPU) |
| `evox`, `torch` | Phase 2 NSGA-III (optional; falls back to NumPy NSGA-II without EvoX) |
| `stable-baselines3`, `gymnasium`, `torch` | Phase 4 RL (optional, falls back to random search) |
| `pytest`, `hypothesis` | Unit and property-based testing |

## Build & Test Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python -m gpu_fuzzy_trader.run_pipeline

# Run all tests
pytest tests/ --hypothesis-seed=42

# Unit tests only
pytest tests/unit/

# Property-based tests only
pytest tests/property/ --hypothesis-seed=42

# Single test file
pytest tests/unit/test_cpu_engine.py -v
```

## Configuration

All hyperparameters live in `gpu_fuzzy_trader/config.py`. No runtime flags. Edit this file to tune pipeline behavior (population sizes, generations, feature counts, RL steps, etc.).

## Skip Logic

Phases 1-4 skip automatically when their output files exist and pass validation. Phase 5 (OOS evaluation) always runs. Delete output files under `outputs/` to force re-run.

## Optional Dependencies

The pipeline gracefully degrades when GPU/RL packages are missing:
- Without JAX: Phase 2 uses CPU backtest engine
- Without evox/torch: Phase 2 uses built-in NumPy NSGA-II (history: `"NSGA-II (fallback)"`)
- Without stable-baselines3: Phase 4 uses random search with Elbow Method stopping
