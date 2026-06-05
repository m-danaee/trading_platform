# Train/Validation Split Ratio Update (70/30) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Change the chronological train/validation split ratio from 75/25 to 70/30 across the trading platform codebase, update all configurations, function names, path keys, split routing, and associated tests to verify correctness.

**Architecture:** Update paths and keys in `gpu_fuzzy_trader/config.py`, change split calculations in `cv_folds.py`, adjust routing in `splitter.py`, update file imports in evaluation/pipeline files, and update test assertions to check for the 70/30 split logic.

**Tech Stack:** Python, Pandas, NumPy, Pytest, Hypothesis

---

### Task 1: Configuration Updates

**Files:**
- Modify: `gpu_fuzzy_trader/config.py`

- [ ] **Step 1: Edit config paths and split mode documentation**
  Replace `TRAIN_75_PATH` with `TRAIN_70_PATH` and `VALIDATION_25_PATH` with `VALIDATION_30_PATH`. Update comments.
  
  ```python
  # Cached splits from train.csv (Phases 2–5). Rebuilt when train.csv is newer.
  TRAIN_70_PATH = "data/train_70.parquet"
  VALIDATION_30_PATH = "data/validation_30.parquet"
  ```
  
  Update comments on lines 126–135:
  
  ```python
  # Phases 4–5 always use persisted train_70 + validation_30 (see splitter.py).
  #
  # SPLIT_MODE options:
  #   "purged_rolling_cv" — K expanding-window folds, 288-bar embargo, ≥2 months
  #                         train per fold; Phase 2/3 score worst fold. Default.
  #   "holdout_70_30"     — legacy single 70/30 per symbol; faster, easier to
  #                         overfit one validation season (risky for short).
  ```

- [ ] **Step 2: Commit Task 1**
  ```bash
  git add gpu_fuzzy_trader/config.py
  git commit -m "config: update train/validation paths and modes to 70/30"
  ```

---

### Task 2: Split Logic Updates in `cv_folds.py`

**Files:**
- Modify: `gpu_fuzzy_trader/data/cv_folds.py`

- [ ] **Step 1: Rename holdout split function and update math to 70/30**
  Modify `holdout_75_25_split` to `holdout_70_30_split` and use `math.floor(n * 0.70)`.
  
  ```python
  def holdout_70_30_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
      """Legacy per-symbol 70/30 chronological split."""
      import math
  
      train_parts: list[pd.DataFrame] = []
      validation_parts: list[pd.DataFrame] = []
  
      for _, group in df.groupby("symbol", sort=True):
          n = len(group)
          split_point = math.floor(n * 0.70)
          train_parts.append(group.iloc[:split_point])
          validation_parts.append(group.iloc[split_point:])
  
      train_df = (
          pd.concat(train_parts, ignore_index=True)
          if train_parts
          else pd.DataFrame(columns=df.columns)
      )
      validation_df = (
          pd.concat(validation_parts, ignore_index=True)
          if validation_parts
          else pd.DataFrame(columns=df.columns)
      )
      return train_df, validation_df
  ```
  
  Also update references in file docstring (line 5):
  
  ```python
  # use the last fold's train/val blocks persisted as train_70 / validation_30.
  ```

- [ ] **Step 2: Commit Task 2**
  ```bash
  git add gpu_fuzzy_trader/data/cv_folds.py
  git commit -m "data: update split function to holdout_70_30_split"
  ```

---

### Task 3: Data Splitter Implementation Updates

**Files:**
- Modify: `gpu_fuzzy_trader/data/splitter.py`

- [ ] **Step 1: Update imports, routing, and persistence paths**
  Update import statement from `config.py` and `cv_folds.py`.
  
  ```python
  from gpu_fuzzy_trader.config import (
      CV_FOLDS_MANIFEST_PATH,
      TRAIN_70_PATH,
      VALIDATION_30_PATH,
  )
  from gpu_fuzzy_trader.data.cv_folds import (
      PurgedFold,
      build_purged_rolling_folds,
      holdout_70_30_split,
      primary_holdout_from_folds,
  )
  ```
  
  Update `split_and_persist`:
  
  ```python
  class Data_Splitter:
      """Chronological train/validation splitter with optional purged rolling CV."""
  
      def split_and_persist(
          self,
          df: pd.DataFrame,
      ) -> tuple[pd.DataFrame, pd.DataFrame, list[PurgedFold]]:
          """
          Build train/validation DataFrames and persist to Parquet.
  
          When ``SPLIT_MODE == "purged_rolling_cv"``, also writes a fold manifest.
          Persisted train_70 / validation_30 always match the **last** CV fold
          (or the 70/30 holdout when legacy mode is selected).
  
          Returns
          -------
          tuple[pd.DataFrame, pd.DataFrame, list[PurgedFold]]
              ``(train_df, validation_df, cv_folds)``. *cv_folds* is non-empty only
              in ``purged_rolling_cv`` mode.
          """
          mode = str(_cfg.SPLIT_MODE).strip().lower()
          folds: list[PurgedFold] = []
  
          if mode == "purged_rolling_cv":
              folds = build_purged_rolling_folds(df)
              if not folds:
                  logger.warning(
                      "Purged rolling CV produced no folds (min_train=%d bars, "
                      "K=%d); falling back to holdout_70_30",
                      int(_cfg.CV_MIN_TRAIN_MONTHS * 30 * _cfg.CV_BARS_PER_DAY),
                      _cfg.CV_N_FOLDS,
                  )
                  train_df, validation_df = holdout_70_30_split(df)
              else:
                  train_df, validation_df = primary_holdout_from_folds(folds)
                  self._persist_cv_manifest(folds, train_df, validation_df)
          elif mode == "holdout_70_30":
              train_df, validation_df = holdout_70_30_split(df)
          else:
              raise ValueError(
                  f"Unknown SPLIT_MODE={_cfg.SPLIT_MODE!r}; "
                  "use 'holdout_70_30' or 'purged_rolling_cv'"
              )
  
          train_df = downcast_numeric_df(train_df)
          validation_df = downcast_numeric_df(validation_df)
  
          train_df.to_parquet(TRAIN_70_PATH, index=False)
          validation_df.to_parquet(VALIDATION_30_PATH, index=False)
  
          return train_df, validation_df, folds
  ```
  
  In `_persist_cv_manifest` (line 138):
  
  ```python
  "note": "train_70/validation_30 parquet = last fold",
  ```

- [ ] **Step 2: Commit Task 3**
  ```bash
  git add gpu_fuzzy_trader/data/splitter.py
  git commit -m "data: update Data_Splitter implementation to use holdout_70_30"
  ```

---

### Task 4: Pipeline and Phase 5 OOS Updates

**Files:**
- Modify: `gpu_fuzzy_trader/phases/phase5_oos.py`
- Modify: `gpu_fuzzy_trader/run_pipeline.py`

- [ ] **Step 1: Update path references in Phase 5 OOS**
  Modify line 358–363 in `phase5_oos.py` to reference `TRAIN_70_PATH` and `VALIDATION_30_PATH`:
  
  ```python
          if os.path.exists(_cfg.TRAIN_70_PATH) and os.path.exists(_cfg.VALIDATION_30_PATH):
              try:
                  train_df_ref = downcast_numeric_df(
                      pd.read_parquet(_cfg.TRAIN_70_PATH))
                  val_df_ref = downcast_numeric_df(
                      pd.read_parquet(_cfg.VALIDATION_30_PATH))
  ```

- [ ] **Step 2: Update path references in `run_pipeline.py`**
  Modify lines 823–824 in `run_pipeline.py`:
  
  ```python
                      _cfg.TRAIN_70_PATH,
                      _cfg.VALIDATION_30_PATH,
  ```
  
  And lines 865–866:
  
  ```python
          train_path = _cfg.TRAIN_70_PATH
          val_path = _cfg.VALIDATION_30_PATH
  ```

- [ ] **Step 3: Commit Task 4**
  ```bash
  git add gpu_fuzzy_trader/phases/phase5_oos.py gpu_fuzzy_trader/run_pipeline.py
  git commit -m "pipeline: update Phase 5 and pipeline scripts to use 70/30 split paths"
  ```

---

### Task 5: Unit Tests Updates for Splitter

**Files:**
- Modify: `tests/unit/test_data_splitter.py`

- [ ] **Step 1: Write the failing tests (by updating existing 75/25 split assertions to expect 70/30 split behavior)**
  Modify `tests/unit/test_data_splitter.py`:
  - Update imports on lines 27-28 to `TRAIN_70_PATH` and `VALIDATION_30_PATH`.
  - Update `TestSplitRatio` methods:
    - Expected train size `math.floor(n * 0.70)`
    - Expected val size `n - math.floor(n * 0.70)`
  - Update `test_single_row_symbol_goes_to_train`:
    - `floor(1 * 0.70) = 0`, so it goes to validation.
  - Update `test_four_row_symbol_split`:
    - `floor(4 * 0.70) = 2`, so 2 train, 2 validation.
  - Update `test_empty_dataframe_returns_empty_dfs`:
    - Patch `TRAIN_70_PATH` and `VALIDATION_30_PATH`.
    - Set `config_mod.SPLIT_MODE = "holdout_70_30"`.
  - Update `test_large_symbol_split_ratio_close_to_075`:
    - Rename to `test_large_symbol_split_ratio_close_to_070`
    - Assert `abs(ratio - 0.70) < 0.001`.
  - Update other test cases referencing `0.75` / `75_25` / `TRAIN_75_PATH`.

- [ ] **Step 2: Run test to verify it fails**
  Run: `PYTHONPATH=. .venv/bin/pytest tests/unit/test_data_splitter.py -v`
  Expected: FAIL (since code still returns 75/25 splits or tests have not been fully aligned)

- [ ] **Step 3: Verify tests pass after implementation**
  Run: `PYTHONPATH=. .venv/bin/pytest tests/unit/test_data_splitter.py -v`
  Expected: PASS

- [ ] **Step 4: Commit Task 5**
  ```bash
  git add tests/unit/test_data_splitter.py
  git commit -m "test: update unit tests in test_data_splitter.py to 70/30 split"
  ```

---

### Task 6: Property Tests Updates

**Files:**
- Modify: `tests/property/test_data_splitter_properties.py`

- [ ] **Step 1: Update property assertions for 70/30 split**
  Modify `tests/property/test_data_splitter_properties.py`:
  - Replace `TRAIN_75_PATH` / `VALIDATION_25_PATH` in `_run_split` with `TRAIN_70_PATH` / `VALIDATION_30_PATH`.
  - Change `config_mod.SPLIT_MODE = "holdout_70_30"`.
  - Change all `0.75` math splits to `0.70`:
    - `expected_train = math.floor(n * 0.70)`
    - `expected_val = n - expected_train`
    - `total_expected_train = sum(math.floor(n * 0.70) for n in symbol_counts.values())`
    - `total_expected_val = sum(n - math.floor(n * 0.70) for n in symbol_counts.values())`

- [ ] **Step 2: Run test to verify it passes**
  Run: `PYTHONPATH=. .venv/bin/pytest tests/property/test_data_splitter_properties.py -v`
  Expected: PASS

- [ ] **Step 3: Commit Task 6**
  ```bash
  git add tests/property/test_data_splitter_properties.py
  git commit -m "test: update data splitter property tests to 70/30 split"
  ```

---

### Task 7: Remaining Test Adjustments and Cleanup

**Files:**
- Modify: `tests/unit/test_purged_cv_folds.py`
- Modify: `tests/unit/test_run_pipeline.py`

- [ ] **Step 1: Update path/mode references in other tests**
  In `tests/unit/test_purged_cv_folds.py` (line 114–115):
  
  ```python
          splitter_mod.TRAIN_70_PATH = str(tmp_path / "train_70.parquet")
          splitter_mod.VALIDATION_30_PATH = str(tmp_path / "val_30.parquet")
  ```
  
  In `tests/unit/test_run_pipeline.py` (lines 403, 416–417, 452, 456–457):
  
  ```python
          train_path = tmp_path / "train_70.parquet"
          val_path = tmp_path / "validation_30.parquet"
          monkeypatch.setattr(_cfg, "TRAIN_70_PATH", str(train_path))
          monkeypatch.setattr(_cfg, "VALIDATION_30_PATH", str(val_path))
  ```

- [ ] **Step 2: Run all updated tests**
  Run: `PYTHONPATH=. .venv/bin/pytest tests/unit/test_data_splitter.py tests/property/test_data_splitter_properties.py tests/unit/test_purged_cv_folds.py tests/unit/test_run_pipeline.py`
  Expected: PASS

- [ ] **Step 3: Commit Task 7**
  ```bash
  git add tests/unit/test_purged_cv_folds.py tests/unit/test_run_pipeline.py
  git commit -m "test: align remaining tests with 70/30 paths"
  ```
