# Set Split Mode to Holdout 70/30 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Configure the pipeline data split mode from `purged_rolling_cv` to `holdout_70_30` in `config.py`, delete stale cached data split/manifest files to force regeneration, and verify the whole project runs and passes all tests successfully.

**Architecture:** Update `SPLIT_MODE` in `gpu_fuzzy_trader/config.py`, delete cached parquet and json manifest files in the `data/` directory, and run `pytest` verification commands to ensure no regression.

**Tech Stack:** Python, Bash, Pytest, Pandas

---

### Task 1: Update Configuration to Holdout 70/30

**Files:**
- Modify: `gpu_fuzzy_trader/config.py`

- [ ] **Step 1: Edit SPLIT_MODE in config.py**

  Update line 278:
  ```python
  SPLIT_MODE = "holdout_70_30"
  ```

- [ ] **Step 2: Commit Task 1**
  ```bash
  git add gpu_fuzzy_trader/config.py
  git commit -m "config: set SPLIT_MODE to holdout_70_30"
  ```

---

### Task 2: Clean Stale Cached Files

**Files:**
- Delete: `data/train_70.parquet`
- Delete: `data/validation_30.parquet`
- Delete: `data/cv_folds_manifest.json`

- [ ] **Step 1: Delete stale files**
  Delete `data/train_70.parquet`, `data/validation_30.parquet`, and `data/cv_folds_manifest.json` if they exist.

  ```bash
  rm -f data/train_70.parquet data/validation_30.parquet data/cv_folds_manifest.json
  ```

- [ ] **Step 2: Commit Task 2**
  ```bash
  git commit -am "data: remove stale cached data split and manifest files"
  ```

---

### Task 3: Verify Unit Tests

**Files:**
- None

- [ ] **Step 1: Run the unit tests**
  ```bash
  PYTHONPATH=. .venv/bin/pytest tests/unit/
  ```
  Verify that all unit tests pass successfully.
