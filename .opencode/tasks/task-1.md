# task-1: Create optuna_search.py

## Goal
Create `gpu_fuzzy_trader/optuna_search.py` — an Optuna-based hyperparameter optimization script that searches for the best combination of config hyperparameters from `gpu_fuzzy_trader/config.py`.

## Files to create
- `gpu_fuzzy_trader/optuna_search.py` (main script)

## Detailed specification

### 1. Imports and setup
```python
#!/usr/bin/env python3
"""Optuna hyperparameter search for GPU Fuzzy Trader config.

Usage:
    python -m gpu_fuzzy_trader.optuna_search --n-trials 50
    python -m gpu_fuzzy_trader.optuna_search --fast --n-trials 30
    python -m gpu_fuzzy_trader.optuna_search --debug --n-trials 10
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
```

### 2. Search space definition

Define a dictionary `SEARCH_SPACE` mapping parameter names to their candidate values (categorical):

| Parameter | Values |
|-----------|--------|
| `PHASE1_TOP_K_FEATURES` | [10, 15, 20, 25, 30] |
| `MIN_TRADE_SUPPORT` | [30, 60, 90, 120, 150] |
| `MIN_TRADE_POOL_FLOOR` | [8, 12, 17, 25, 35] |
| `PHASE2_MAX_DRAWDOWN_GATE` | [12.0, 18.0, 25.0, 30.0, 35.0] |
| `PHASE2_MAX_TRAIN_VAL_GAP_PCT` | [4.0, 6.0, 8.0, 10.0, 15.0] |
| `PHASE2_KEEP_TOP_RULES` | [40, 60, 80, 100, 120] |
| `PHASE2_POPULATION_SIZE` | [100, 150, 200, 250] |
| `PHASE2_MUTATION_RATE` | [0.12, 0.17, 0.22, 0.27, 0.32] |
| `PHASE2_STAGE_A_GENERATIONS` | [50, 70, 85, 100, 120] |
| `PHASE3_VAL_RETURN_FLOOR_PCT` | [2.0, 4.0, 5.0, 7.0, 10.0] |
| `RB_MIN_TRAIN_RETURN` | [1.0, 2.0, 3.0, 5.0] |
| `RB_MIN_VALID_RETURN` | [1.0, 2.0, 3.0, 5.0] |

### 3. Config patching

Use `unittest.mock.patch` to temporarily set config values:

```python
from unittest.mock import patch

def apply_trial_config(trial_params: dict[str, Any]):
    """Patch config module globals with trial hyperparameters."""
    patchers = []
    for key, value in trial_params.items():
        patchers.append(patch(f"gpu_fuzzy_trader.config.{key}", value))
    for p in patchers:
        p.start()
    return patchers
```

After each trial, stop all patchers to reset.

### 4. Pipeline execution

```python
def run_pipeline_for_trial(trial_params: dict, fast_mode: bool, debug_mode: bool) -> dict:
    """Run the pipeline with trial hyperparameters. Returns metrics dict."""
    # Set debug scope if debug mode
    if debug_mode:
        import gpu_fuzzy_trader.config as cfg
        cfg.DEBUG_SYMBOL_SCOPE_ENABLED = True
        cfg.DEBUG_SYMBOL_COUNT = 4
    
    # Run pipeline
    from gpu_fuzzy_trader.run_pipeline import run_pipeline
    run_pipeline(resume=fast_mode)
    
    # Collect metrics from Phase 5 output
    metrics = collect_phase5_metrics()
    return metrics
```

### 5. Metrics collection

Parse the Phase 5 output (from `outputs/` directory or pipeline log) to get:
- `test_long_return_pct`: total_return_pct for long direction on test
- `test_short_return_pct`: total_return_pct for short direction on test
- `test_long_dd_pct`: max_drawdown_pct for long
- `test_short_dd_pct`: max_drawdown_pct for short
- `test_long_pf`: profit_factor for long
- `test_short_pf`: profit_factor for short

Look at how `run_pipeline.py` Phase 5 writes its output — likely JSON files in `outputs/` or `outputs/reports/`.

Alternative: Import evaluator_v5 functions and run evaluation directly on the output strategy JSONs.

### 6. Objective function

```python
def objective(trial: optuna.Trial) -> float:
    """Optuna objective: maximize test return with drawdown penalty."""
    
    # Sample hyperparameters
    trial_params = {}
    for param_name, values in SEARCH_SPACE.items():
        trial_params[param_name] = trial.suggest_categorical(param_name, values)
    
    # Apply config patches
    patchers = apply_trial_config(trial_params)
    
    try:
        # Run pipeline
        metrics = run_pipeline_for_trial(trial_params, fast_mode, debug_mode)
        
        # Compute composite score
        long_return = metrics.get("test_long_return_pct", -999)
        short_return = metrics.get("test_short_return_pct", -999)
        long_dd = metrics.get("test_long_dd_pct", 100)
        short_dd = metrics.get("test_short_dd_pct", 100)
        
        combined_return = (long_return + short_return) / 2.0
        max_dd = max(long_dd, short_dd)
        
        # Score: reward return, penalize drawdown above threshold
        DD_THRESHOLD = 8.0
        DD_PENALTY_WEIGHT = 3.0
        
        dd_penalty = DD_PENALTY_WEIGHT * max(0, max_dd - DD_THRESHOLD)
        score = combined_return - dd_penalty
        
        # Store metrics in trial user attrs
        trial.set_user_attr("long_return", long_return)
        trial.set_user_attr("short_return", short_return)
        trial.set_user_attr("long_dd", long_dd)
        trial.set_user_attr("short_dd", short_dd)
        trial.set_user_attr("combined_return", combined_return)
        trial.set_user_attr("max_dd", max_dd)
        
        return score
        
    except Exception as e:
        # Handle failures gracefully
        trial.set_user_attr("error", str(e))
        traceback.print_exc()
        return -999.0  # Heavy penalty for failed trials
        
    finally:
        # Always restore config
        for p in patchers:
            p.stop()
```

### 7. Main function with CLI

```python
def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--fast", action="store_true", help="Fast mode: skip Phases 1-2")
    parser.add_argument("--debug", action="store_true", help="Debug mode: use 4-symbol scope")
    parser.add_argument("--study-name", type=str, default="gpu_fuzzy_optuna", help="Study name")
    parser.add_argument("--storage", type=str, default="sqlite:///outputs/optuna_study.db")
    args = parser.parse_args()
    
    # Ensure outputs dir
    os.makedirs("outputs", exist_ok=True)
    
    # Create study
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    
    # Store flags for use in objective
    global _fast_mode, _debug_mode
    _fast_mode = args.fast
    _debug_mode = args.debug
    
    study.optimize(
        lambda trial: objective(trial),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )
    
    # Print results
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best score: {study.best_value:.4f}")
    print(f"Best params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save best params to JSON
    best_path = "outputs/optuna_best_params.json"
    with open(best_path, "w") as f:
        json.dump({
            "best_trial": study.best_trial.number,
            "best_score": study.best_value,
            "best_params": study.best_params,
            "user_attrs": study.best_trial.user_attrs,
        }, f, indent=2)
    print(f"\nBest params saved to {best_path}")
    
    # Print top 5 trials
    print("\nTop 5 trials:")
    for trial in study.trials[:5]:
        if trial.value is not None:
            print(f"  Trial {trial.number}: score={trial.value:.4f}")

if __name__ == "__main__":
    main()
```

### 8. Pipeline runner integration

The `run_pipeline_for_trial` function needs to actually call the pipeline. Look at `gpu_fuzzy_trader/run_pipeline.py` to find the main entry function. It might be `run_pipeline()` or `main()`.

If the pipeline runs as `python -m gpu_fuzzy_trader.run_pipeline`, we can:
- Option A: Import and call the pipeline function directly
- Option B: Use subprocess to run it (cleaner isolation)

For simplicity and avoiding import issues, use **subprocess**:

```python
import subprocess

def run_pipeline_via_subprocess(fast_mode: bool, debug_mode: bool) -> int:
    """Run pipeline as subprocess. Returns exit code."""
    cmd = [sys.executable, "-m", "gpu_fuzzy_trader.run_pipeline"]
    if fast_mode:
        cmd.append("--resume")
    
    env = os.environ.copy()
    if debug_mode:
        # Enable debug scope via env vars (check if config picks these up)
        # Actually debug scope is set via config globals, so we need to
        # set env vars that config.py checks, OR pre-patch config before subprocess
        pass
    
    result = subprocess.run(cmd, capture_output=False, env=env)
    return result.returncode
```

However, using subprocess means config patches won't work. Better approach: **import and call directly**, with careful module management:

```python
def run_pipeline_directly():
    """Import and run pipeline directly."""
    # Force reimport of critical modules to pick up patched config
    import importlib
    import gpu_fuzzy_trader.run_pipeline as rp
    importlib.reload(rp)
    
    # Call the pipeline main function
    # Check what function signature run_pipeline.py exposes
    rp.run_pipeline(resume=_fast_mode)
```

Actually, looking at `run_pipeline.py`, the cleanest approach is:
1. Patch config globals
2. Import and call `run_pipeline()` from `gpu_fuzzy_trader.run_pipeline`

But since config values are read at import time for some modules, we may need to reload modules after patching. Simplest approach: use subprocess with environment variables.

**Final decision: Use subprocess with environment variable overrides.** Many config values can be overridden via environment variables per the config.py comments (e.g., `DATA_ROOT`, `TRAIN_CSV_PATH`). But for most hyperparameters, we need direct patching.

**Better approach**: Write trial params to a temporary JSON file, then have a wrapper that reads them. But that's complex.

**Simplest working approach**: 
1. In the Optuna script, use `unittest.mock.patch` on config
2. Import pipeline modules INSIDE the patched context (so they pick up patched values)
3. Call pipeline.run_pipeline()

OR even simpler: reload config and all dependent modules after patching.

Let me go with the approach where we:
1. Patch config values directly on the module object
2. Use importlib.reload on key modules that cache config values
3. Then import and call run_pipeline

Actually, the simplest robust approach: modify config.py globals, then use `subprocess.run()` with `PYTHONPATH` set correctly and pass the patched config as a temporary config override file. But that's too complex.

**Let's do the import-based approach** with careful module reloading:

```python
def _patch_and_run():
    import gpu_fuzzy_trader.config as cfg
    # Set values directly
    for key, value in _current_trial_params.items():
        setattr(cfg, key, value)
    
    # Reload modules that cache config values at import time
    import importlib
    import gpu_fuzzy_trader.run_pipeline as rp
    importlib.reload(rp)
    
    # Run
    rp.run_pipeline()
```

This should work because most modules read config at call time, not import time. The `run_pipeline.py` reads config values as `_cfg.PHASE1_TOP_K_FEATURES` etc. at function execution time, not at module import time. Let me verify...

Looking at run_pipeline.py from the task output: "All hyperparameters are accessed as _cfg.PHASE1_TOP_K_FEATURES, _cfg.OUTPUTS_DIR, _cfg.FEE_PCT, etc." — these are accessed at call time.

So just patching the config module attributes should work without reloading!

## Acceptance criteria
1. `gpu_fuzzy_trader/optuna_search.py` file exists and is syntactically valid
2. Script has CLI with --n-trials, --fast, --debug, --study-name, --storage
3. Search space covers all 12 hyperparameters from the plan
4. Objective function computes composite score from test return and drawdown
5. Config values are patched per-trial and restored after
6. Study saved to SQLite DB, best params exported to JSON
7. Failed trials return penalty score (-999) rather than crashing
8. Script uses `from __future__ import annotations` and matches project code style
