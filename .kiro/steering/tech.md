# GPU-Fuzzy Trading Pipeline - Tech Stack

## Technology Stack

### Core Dependencies (Required)
- **Python 3.10+** (3.12 tested)
- **pandas>=2.0** - Data manipulation and analysis
- **numpy>=1.24** - Numerical computing
- **scikit-learn>=1.3** - Feature selection (mutual information)
- **matplotlib>=3.7** - Visualization and reporting
- **pyarrow>=14.0** - Parquet file format support

### GPU Acceleration (Optional but Recommended)
- **jax>=0.4.20** - GPU-accelerated numerical computing
- **jaxlib>=0.4.20** - JAX backend
- **evox>=0.9.0** - Evolutionary algorithms library (NSGA-III)
- **torch>=2.0.0** - Required for EvoX reference vectors

### Reinforcement Learning (Optional)
- **stable-baselines3>=2.2.0** - RL algorithms (DDPG/PPO)
- **gymnasium>=0.29.0** - RL environment interface

### Development & Testing
- **pytest>=7.4.0** - Test framework
- **hypothesis>=6.90.0** - Property-based testing

## Build System & Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# CPU-only minimum (skip GPU/RL packages)
pip install pandas numpy scikit-learn matplotlib pyarrow pytest hypothesis
```

### Running the Pipeline
```bash
# Full pipeline (from project root)
python -m gpu_fuzzy_trader.run_pipeline

# Alternative entry point
python -m gpu_fuzzy_trader

# Programmatic usage
from gpu_fuzzy_trader.run_pipeline import Pipeline_Orchestrator
orchestrator = Pipeline_Orchestrator()
results = orchestrator.run()
```

### Testing
```bash
# All tests (GPU/JAX tests skipped automatically without JAX)
PYTHONPATH=. pytest tests/ --hypothesis-seed=42

# Unit tests only
pytest tests/unit/

# Property-based tests only
pytest tests/property/ --hypothesis-seed=42

# Verbose output
pytest tests/ --hypothesis-seed=42 -v
```

### Data Management
```bash
# Force re-run from Phase 1 (delete cached outputs)
rm -rf outputs/ data/train_75.parquet data/validation_25.parquet

# Re-run only Phase 2 (long direction)
rm pools/phase2_long_pool.json pools/phase2_long_history.json

# Re-run only Phase 2 (short direction)
rm pools/phase2_short_pool.json pools/phase2_short_history.json
```

### Evaluation
```bash
# Run the evaluation notebook
jupyter notebook evaluator_v3.ipynb
```

## Configuration

**Single source of truth**: `gpu_fuzzy_trader/config.py`

### Key Configuration Patterns
1. **No module-level defaults** - All constants must come from config.py
2. **Path management** - Use `os.path.join` with `_PROJECT_ROOT` for absolute paths
3. **Skip logic** - Each phase validates cached outputs before skipping
4. **Fallback mechanisms** - GPU/RL phases have CPU/random-search fallbacks

### Common Configuration Settings
- `PHASE1_TOP_K_FEATURES` - Features selected per direction (default: 25)
- `PHASE2_POPULATION_SIZE` - Evolution population size (default: 200)
- `PHASE2_GENERATIONS` - Number of generations (default: 100)
- `PHASE3_MIN_RULES` / `PHASE3_MAX_RULES` - Rule set size bounds (default: 2-3)
- `PHASE4_TOTAL_TIMESTEPS` - RL training steps (default: 500,000)

## Code Style & Conventions

### Import Organization
```python
from __future__ import annotations

import os
import sys
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Local imports
from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.data.loader import Data_Loader
```

### Error Handling
- Use custom exception classes for domain-specific errors
- Log warnings for fallback scenarios (missing GPU/RL packages)
- Validate inputs and outputs with schema checking

### Logging
- Structured JSON-lines logging to `outputs/pipeline.log`
- Phase timing with start/end timestamps
- Progress reporting for long-running operations

### Testing Patterns
- Unit tests for individual components
- Property-based tests for backtest engine correctness
- GPU tests skipped automatically when JAX not available
- Hypothesis for generating test data