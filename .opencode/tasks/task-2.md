# task-2: Integration test — verify optuna_search works end-to-end

## Goal
Verify that `gpu_fuzzy_trader/optuna_search.py` runs correctly end-to-end with `--debug --n-trials 3` and `--fast --n-trials 2`.

## Verification steps

### 1. Import test
```bash
cd /home/danaee/trading_platform
PYTEST_LOW_MEMORY=1 .venv/bin/python -c "import gpu_fuzzy_trader.optuna_search; print('Import OK')"
```

### 2. Debug mode run (3 trials with 4-symbol scope)
```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m gpu_fuzzy_trader.optuna_search --debug --n-trials 3 --study-name test_debug_run
```

Expected:
- 3 trials complete (or fail gracefully)
- `outputs/optuna_study.db` exists
- `outputs/optuna_best_params.json` exists with valid JSON
- Different trials have different param values

### 3. Fast mode run (2 trials, resume from existing outputs)
```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m gpu_fuzzy_trader.optuna_search --fast --n-trials 2 --study-name test_fast_run
```

Expected:
- 2 trials complete using --resume pipeline
- Study is appended to (or new study created)

### 4. Artifact check
- Confirm `outputs/optuna_study.db` file size > 0
- Confirm `outputs/optuna_best_params.json` has keys: best_trial, best_score, best_params, user_attrs
- Confirm best_params dict contains all 12 hyperparameter names

### 5. Help text
```bash
.venv/bin/python -m gpu_fuzzy_trader.optuna_search --help
```
Expected: Shows all CLI options with descriptions.

## Acceptance criteria
- Import succeeds without errors
- `--debug --n-trials 3` completes (all trials score, none crash)
- `--fast --n-trials 2` completes
- Output artifacts exist and are valid
- CLI help shows all expected arguments
