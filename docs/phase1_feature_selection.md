# Phase 1 — Feature selection

Reduces the ~100+ raw columns from `train.csv` to a stable long/short feature list that becomes Phase 2's gene space.

**Source code:**
- Selector: [`gpu_fuzzy_trader/features/feature_selector.py`](../gpu_fuzzy_trader/features/feature_selector.py)
- MI / sign consistency / stationarity: [`gpu_fuzzy_trader/features/`](../gpu_fuzzy_trader/features/)

**Hyperparameter reference:** [README.md §4](../README.md#4-phase-1--feature-selection)

## Algorithm

1. Dispersion filter — drop near-constant columns (`PHASE1_DISPERSION_THRESHOLD`).
2. Mutual information against forward returns — top-K candidates per direction.
3. Asymmetric target — when `PHASE1_ASYMMETRIC_TARGET=True`, MI targets long and short return separately.
4. Sign-consistency filter — drop features whose correlation sign flips across folds.
5. Stationarity filter — drop features whose MI rank jumps across chronological/regime folds.
6. Long/short Jaccard overlap cap (`PHASE1_MAX_FEATURE_OVERLAP`).

Runs on **train split only** — no validation labels are used in ranking, so the holdout remains unseen until Phase 2 admission.

## Output

Writes the selected feature names to the Phase 1 result; the splitter then prunes `train_70.parquet`, `validation_30.parquet`, and CV folds down to these columns before Phase 2 begins.
