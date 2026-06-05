# Design: Train/Validation Split Ratio Update (70/30)

Change the chronological train/validation split ratio from 75/25 to 70/30 to increase the size of the validation set (since the validation set currently has too few rows).

## Proposed Changes

### 1. Configuration (`gpu_fuzzy_trader/config.py`)
- Rename `TRAIN_75_PATH` to `TRAIN_70_PATH = "data/train_70.parquet"`.
- Rename `VALIDATION_25_PATH` to `VALIDATION_30_PATH = "data/validation_30.parquet"`.
- Update comments and documentation referencing `75/25` or `75_25` splits to `70/30` or `70/30`.

### 2. Splitting Logic (`gpu_fuzzy_trader/data/cv_folds.py`)
- Rename the function `holdout_75_25_split(df)` to `holdout_70_30_split(df)`.
- Change split point formula:
  - From: `split_point = math.floor(n * 0.75)`
  - To: `split_point = math.floor(n * 0.70)`
- Update docstring references from 75/25 to 70/30.

### 3. Splitter Logic (`gpu_fuzzy_trader/data/splitter.py`)
- Import `TRAIN_70_PATH`, `VALIDATION_30_PATH`, and `holdout_70_30_split`.
- Update the legacy split mode string check from `"holdout_75_25"` to `"holdout_70_30"`.
- Update persisted file paths to `TRAIN_70_PATH` and `VALIDATION_30_PATH`.
- Remove references/imports of the old 75/25 paths and function.

### 4. Pipeline & Evaluation Components
- In [phase5_oos.py](file:///home/danaee/trading_platform/gpu_fuzzy_trader/phases/phase5_oos.py), load datasets using `TRAIN_70_PATH` and `VALIDATION_30_PATH`.
- In [run_pipeline.py](file:///home/danaee/trading_platform/gpu_fuzzy_trader/run_pipeline.py), update references from `TRAIN_75_PATH`/`VALIDATION_25_PATH` to `TRAIN_70_PATH`/`VALIDATION_30_PATH`.

### 5. Tests
- Update unit tests in [test_data_splitter.py](file:///home/danaee/trading_platform/tests/unit/test_data_splitter.py):
  - Rename helper references.
  - Change expected train count to `math.floor(n * 0.70)` and validation count to `n - math.floor(n * 0.70)`.
  - Update ratio tolerance checks to expect `0.70` instead of `0.75`.
- Update property tests in [test_data_splitter_properties.py](file:///home/danaee/trading_platform/tests/property/test_data_splitter_properties.py):
  - Change property assertions from `math.floor(n * 0.75)` to `math.floor(n * 0.70)`.
- Update unit tests in [test_purged_cv_folds.py](file:///home/danaee/trading_platform/tests/unit/test_purged_cv_folds.py) and [test_run_pipeline.py](file:///home/danaee/trading_platform/tests/unit/test_run_pipeline.py) to reference new path configurations.

## Verification Plan

### Automated Tests
Run the following commands using the `.venv` virtual environment:
```bash
PYTHONPATH=. .venv/bin/pytest tests/unit/test_data_splitter.py
PYTHONPATH=. .venv/bin/pytest tests/property/test_data_splitter_properties.py
PYTHONPATH=. .venv/bin/pytest tests/unit/test_purged_cv_folds.py
PYTHONPATH=. .venv/bin/pytest tests/unit/test_run_pipeline.py
```
Check that all tests pass.
