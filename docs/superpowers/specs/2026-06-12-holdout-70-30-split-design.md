# Design: Configure Split Mode to Holdout 70/30

Configure the pipeline data split mode from `purged_rolling_cv` to `holdout_70_30` as requested. Ensure the pipeline and tests continue to pass and stale cache files are cleaned up.

## Proposed Changes

### 1. Configuration (`gpu_fuzzy_trader/config.py`)
- Update `SPLIT_MODE = "purged_rolling_cv"` to `SPLIT_MODE = "holdout_70_30"`.

### 2. Cache Cleanup
- Delete stale cached data split parquet files and manifest:
  - `data/train_70.parquet`
  - `data/validation_30.parquet`
  - `data/cv_folds_manifest.json`

## Verification Plan

### Automated Tests
Run the unit test suite inside `.venv` to verify code correctness under `holdout_70_30` split mode:
```bash
PYTHONPATH=. .venv/bin/pytest tests/unit/
```

### Pipeline Run Verification
Trigger a dry-run or validation phase of the pipeline to verify it correctly splits the raw data into 70/30 partitions and persists them.
