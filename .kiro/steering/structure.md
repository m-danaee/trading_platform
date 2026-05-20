# GPU-Fuzzy Trading Pipeline - Project Structure

## Directory Organization

```
trading_platform/
├── .kiro/
│   ├── specs/
│   │   └── gpu-fuzzy-trading-pipeline/     # Existing spec documentation
│   │       ├── .config.kiro
│   │       ├── requirements.md
│   │       ├── design.md
│   │       └── tasks.md
│   └── steering/                           # Steering documents (this folder)
│       ├── product.md
│       ├── tech.md
│       └── structure.md
├── data/                                   # Input data
│   ├── train.csv                          # Training data (Phases 1-4)
│   ├── test.csv                           # Test data (Phase 5 only)
│   ├── train_75.parquet                   # Auto-generated 75% split
│   └── validation_25.parquet              # Auto-generated 25% split
├── gpu_fuzzy_trader/                      # Main Python package
│   ├── __init__.py
│   ├── __main__.py                        # Entry point: python -m gpu_fuzzy_trader
│   ├── config.py                          # Single source of truth for all hyperparameters
│   ├── run_pipeline.py                    # Top-level orchestrator
│   ├── _jax_env.py                        # JAX environment setup
│   ├── _memory.py                         # Memory management utilities
│   ├── log_progress.py                    # Progress logging utilities
│   │
│   ├── data/                              # Data loading and preparation
│   │   ├── __init__.py
│   │   ├── loader.py                      # Data_Loader: CSV loading, datetime parsing
│   │   └── splitter.py                    # Data_Splitter: per-symbol chronological split
│   │
│   ├── features/                          # Feature processing
│   │   ├── __init__.py
│   │   ├── detector.py                    # Feature_Detector: mode classification
│   │   ├── encoder.py                     # Encoder: gene → fuzzy value mapping
│   │   └── selector.py                    # Feature_Selector: direction-specific scoring
│   │
│   ├── backtest/                          # Backtest engines
│   │   ├── __init__.py
│   │   ├── cpu_engine.py                  # CPUBacktestEngine: reference implementation
│   │   └── gpu_engine.py                  # GPUBacktestEngine: JAX-accelerated
│   │
│   ├── evolution/                         # Evolutionary algorithms
│   │   ├── __init__.py
│   │   └── evox_runner.py                 # Phase 2 NSGA-III loop
│   │
│   ├── phases/                            # Pipeline phases
│   │   ├── __init__.py
│   │   ├── phase2_rule_pool.py            # Rule_Pool_Generator: Phase 2 orchestration
│   │   ├── phase3_rule_set.py             # Rule_Set_Selector: Phase 3 greedy + refinement
│   │   ├── phase4_rl_optimizer.py         # RL_Agent: Phase 4 risk optimization
│   │   └── phase5_oos.py                  # OOS_Evaluator: Phase 5 out-of-sample evaluation
│   │
│   ├── output/                            # Output serialization
│   │   ├── __init__.py
│   │   └── writer.py                      # Output_Writer: JSON serialization & validation
│   │
│   └── reporting/                         # Reporting and visualization
│       ├── __init__.py
│       └── reporter.py                    # Reporter: equity curves, metrics, plots
│
├── outputs/                               # Generated outputs (per run)
│   ├── pipeline.log                       # JSON-lines phase timing log
│   ├── selected_features_long.json        # Phase 1: long features
│   ├── selected_features_short.json       # Phase 1: short features
│   ├── long.json                          # Phase 3/4: final long strategy
│   ├── short.json                         # Phase 3/4: final short strategy
│   └── reports/                           # Visual reports
│       ├── phase2_long_metrics.png        # Phase 2 evolution metrics
│       ├── phase2_short_metrics.png
│       ├── train_long_equity.png          # Phase 3 training equity curves
│       ├── validation_long_equity.png     # Phase 3 validation equity curves
│       ├── phase4_long_rl_curve.png       # Phase 4 RL training curves
│       ├── test_long_equity.png           # Phase 5 test equity curves
│       ├── train_per_symbol_performance.csv  # Per-symbol metrics
│       ├── validation_per_symbol_performance.csv
│       └── test_per_symbol_performance.csv
│
├── pools/                                 # Persistent Phase 2 outputs (project root)
│   ├── phase2_long_pool.json              # Pareto-front rule pool (long)
│   ├── phase2_short_pool.json             # Pareto-front rule pool (short)
│   ├── phase2_long_history.json           # Per-generation metrics (long)
│   └── phase2_short_history.json          # Per-generation metrics (short)
│
├── phase2_rule_archive/                   # Persistent Phase 2 archive (project root)
│   ├── phase2_long_archive.json           # Best-rule archive (long)
│   └── phase2_short_archive.json          # Best-rule archive (short)
│
├── tests/                                 # Test suite
│   ├── unit/                              # Unit tests
│   └── property/                          # Property-based tests
│
├── evaluator_v3.ipynb                     # Evaluation notebook (compatible with outputs)
├── README.md                              # Comprehensive documentation
├── RUN.md                                 # Quick reference for running the project
├── requirements.txt                       # Python dependencies
└── .gitignore                             # Git ignore rules
```

## Module Responsibilities

### Core Orchestration
- **`run_pipeline.py`** - `Pipeline_Orchestrator`: Top-level coordinator, phase sequencing, skip logic
- **`config.py`** - Central configuration, constants, paths, hyperparameters

### Data Layer
- **`data/loader.py`** - `Data_Loader`: CSV loading, datetime parsing, NaN handling, tail drop
- **`data/splitter.py`** - `Data_Splitter`: Per-symbol chronological 75/25 split

### Feature Processing
- **`features/detector.py`** - `Feature_Detector`: Classify feature modes (binary/ternary/etc.)
- **`features/selector.py`** - `Feature_Selector`: Direction-specific feature scoring and selection
- **`features/encoder.py`** - `Encoder`: Map integer genes to fuzzy value names, format condition strings

### Backtest Engines
- **`backtest/cpu_engine.py`** - `CPUBacktestEngine`: Reference implementation matching `evaluator_v3.ipynb`
- **`backtest/gpu_engine.py`** - `GPUBacktestEngine`: JAX-accelerated, numerically equivalent to CPU engine

### Evolutionary Algorithms
- **`evolution/evox_runner.py`** - `run_phase2_evolution`: NSGA-III multi-objective search with EvoX

### Pipeline Phases
- **`phases/phase2_rule_pool.py`** - `Rule_Pool_Generator`: Phase 2 orchestration, persistence, reporting
- **`phases/phase3_rule_set.py`** - `Rule_Set_Selector`: Greedy construction + Pareto refinement
- **`phases/phase4_rl_optimizer.py`** - `RL_Agent`: DDPG/PPO risk parameter optimization
- **`phases/phase5_oos.py`** - `OOS_Evaluator`: Out-of-sample evaluation on test data

### Output & Reporting
- **`output/writer.py`** - `Output_Writer`: JSON serialization, schema validation, file writing
- **`reporting/reporter.py`** - `Reporter`: Equity curves, metrics, plots, CSV generation

## File Naming Conventions

### Data Files
- `{direction}.json` - Final strategy files (long.json, short.json)
- `selected_features_{direction}.json` - Phase 1 feature selections
- `phase2_{direction}_{type}.json` - Phase 2 outputs (pool, history, archive)
- `train_75.parquet`, `validation_25.parquet` - Split data files

### Report Files
- `{phase}_{direction}_{metric}.png` - Visualization plots
- `{split}_per_symbol_performance.csv` - Per-symbol metrics
- `{split}_{direction}_equity.png` - Equity curve plots

### Test Files
- `test_{module_name}.py` - Unit tests
- `property_test_{domain}.py` - Property-based tests

## Path Management Rules

1. **Project root as working directory** - All paths in config.py are relative to project root
2. **Absolute paths for persistence** - Phase 2 pools/archives use `_PROJECT_ROOT` for absolute paths
3. **Output directory separation** - `outputs/` is per-run, `pools/` and `phase2_rule_archive/` are persistent
4. **Data directory isolation** - Raw data in `data/`, processed splits also in `data/`

## Import Patterns

```python
# Standard library imports first
import os
import sys
from datetime import datetime

# Third-party imports
import pandas as pd
import numpy as np

# Local imports (relative to gpu_fuzzy_trader package)
from . import config as _cfg
from .data.loader import Data_Loader
from .features.selector import Feature_Selector

# Avoid circular imports
# Use `from __future__ import annotations` for forward references
```

## Testing Structure

- **Unit tests** in `tests/unit/` - Test individual components in isolation
- **Property-based tests** in `tests/property/` - Test invariants and properties
- **Test data** - Use fixtures and Hypothesis for data generation
- **GPU test skipping** - Tests requiring JAX skip automatically when not available