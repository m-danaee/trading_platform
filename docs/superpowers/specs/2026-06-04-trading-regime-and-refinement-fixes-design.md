# Design Specification: Trading Regime and Refinement Fixes

## 1. Executive Summary

This design addresses four major algorithmic ideas and three key structural problems identified in recent runs:
1. **Idea 1**: A new Regime-Profitability Gate in Phase 2 to filter out rules that fail to generalize across market regimes.
2. **Idea 2**: A Spearman sign consistency filter in Phase 1 to blacklist non-stationary features before evolution.
3. **Idea 3**: A Recency-Weighted Objective in Phase 2 to emphasize recent training data (last 25% of training bars).
4. **Idea 4**: A Minimum Recent-Period Gate (Last-Fold Return Gate) in Phase 2 Archive/Pool admission.
5. **Problem 2**: Eliminating the Phase 3 population clamping lock when `pool < 50`.
6. **Problem 3**: Fixing Phase 4 Empty WF splits (already set to 2 in config).

---

## 2. Detailed Technical Design

### Idea 1: Regime-Profitability Gate in Phase 2
* **Config parameters to add (in `config.py`):**
  ```python
  PHASE2_REGIME_PROFITABILITY_GATE: bool = True     # require profit > 0 in >=2 of 3 regimes
  PHASE2_REGIME_MIN_RETURN_PER_REGIME: float = 0.0  # per-regime return floor
  ```
* **Implementation changes:**
  * Update `_merge_metrics_worst_case` in `gpu_fuzzy_trader/phases/phase2_cv.py` to merge `regime_net_pnl`, `regime_trade_counts`, and `regime_win_counts` conservatively (element-wise minimum across folds).
  * Update `_passes_pool_admission_impl` in `gpu_fuzzy_trader/phases/phase2_support.py` to check that the rule has a net PnL greater than `PHASE2_REGIME_MIN_RETURN_PER_REGIME` in at least 2 out of the 3 regimes (or all if regimes < 2).

### Idea 2: Stationarity Blacklist on Features
* **Config parameters to add (in `config.py`):**
  ```python
  PHASE1_REQUIRE_SIGN_CONSISTENCY: bool = True     # drop features with Spearman sign flip across folds
  PHASE1_SIGN_CONSISTENCY_MIN_FOLDS: int = 2       # must have same sign in >= N folds
  ```
* **Implementation changes:**
  * Implement Spearman correlation scoring across chronological folds in `gpu_fuzzy_trader/features/selector.py`.
  * Define `_get_spearman_folds(df, n_folds)` to partition the data chronologically per symbol and then combine them to form `n_folds` folds.
  * Define `_check_spearman_sign_consistency(df, feature_cols, n_folds, min_folds)` to calculate Spearman correlation of each feature against `label_close_288` on each fold and blacklist any feature whose sign flips between positive and negative.
  * Apply this filter at the start of `Feature_Selector.select_features`.

### Idea 3: Recency-Weighted Objective in Phase 2
* **Config parameters to add (in `config.py`):**
  ```python
  PHASE2_RECENCY_WEIGHT_ENABLED: bool = True
  PHASE2_RECENCY_WEIGHT_FRACTION: float = 0.25
  PHASE2_RECENCY_WEIGHT_MULTIPLIER: float = 2.0
  ```
* **Implementation changes:**
  * Precompute `recency_weights` in `_jax_simulate_equity_batch` and `_jax_simulate_equity_batch_regime` inside `gpu_fuzzy_trader/backtest/gpu_engine.py`.
  * Pass `recency_weights` in the scan loop input.
  * Multiply `net_pnl` by the recency weight in `step` before updating the cumulative equity and trade returns.

### Idea 4: Last-Fold Return Gate in Phase 2
* **Config parameters to add (in `config.py`):**
  ```python
  PHASE2_REQUIRE_LAST_FOLD_POSITIVE: bool = True   # rule must be profitable on validation split of most recent fold
  ```
* **Implementation changes:**
  * Update `_passes_pool_admission_impl` in `gpu_fuzzy_trader/phases/phase2_support.py` to enforce that the validation return is positive if `PHASE2_REQUIRE_LAST_FOLD_POSITIVE` is True (for holdout mode).
  * Update `evaluate_purged_cv_pool_admission_batch` in `gpu_fuzzy_trader/phases/phase2_cv.py` to filter out rules that do not achieve a positive validation return on the last chronological fold (fold index `len(folds) - 1`).

### Problem 2: Phase 3 Population Clamping Fix
* **Implementation changes:**
  * In `gpu_fuzzy_trader/phases/phase3_rule_set.py` under the function `_refine_nsga2`, replace:
    ```python
    effective_pop = min(pop_size, max(4, len(pool) * 2))
    ```
    with:
    ```python
    effective_pop = pop_size
    ```
  * This guarantees that diversity is maintained during the combinatorial optimization phase.

---

## 3. Testing Plan

We will run the unit tests under `tests/unit/` using pytest to verify that nothing is broken, and add regression tests for:
1. Feature stationarity (sign consistency filter) on dummy datasets.
2. Clamping removal in Phase 3 population initialization.
3. Last-fold positive gate check.
