# Phase 1 — Direction-Specific Feature Selection

**Module:** `gpu_fuzzy_trader/features/selector.py` → `Feature_Selector`
**Supporting modules:** `features/detector.py`, `features/encoder.py`, `features/regime_cluster.py`

Phase 1 produces two independent ranked feature lists — one for long trades, one for short trades — from the training split only. These lists drive the chromosome structure in Phase 2: each selected feature becomes one gene slot in the evolutionary search.

---

## 1. Why Direction-Specific Selection?

Long and short trades have fundamentally different success conditions. A feature that predicts upward price movement is not necessarily useful for predicting downward movement. Using a single shared feature list would force the evolutionary search to use the same gene space for both directions, diluting the signal.

The `PHASE1_ASYMMETRIC_TARGET` flag (default: `True`) enforces this separation by building genuinely different target signals for each direction (see Section 4).

---

## 2. Feature Mode Detection — `Feature_Detector` (`features/detector.py`)

Before scoring, every feature column is classified into one of six discrete modes. This classification is done on the **training split only** to prevent test-set leakage.

### Detection algorithm (exact match to `evaluator_v5.ipynb`)

```python
unique_vals = series.dropna().unique()
n_unique = len(unique_vals)

if n_unique <= 2 and set(unique_vals).issubset({0, 1}):
    return "binary"
if n_unique <= 3 and set(unique_vals).issubset({-1, 0, 1}):
    return "ternary"

zero_ratio = (series == 0).mean()   # computed on FULL series including zeros

if series.min() < 0:
    return "sparse_signed" if zero_ratio > 0.3 else "signed"
return "sparse_positive" if zero_ratio > 0.3 else "positive"
```

**Critical detail:** `zero_ratio` is computed on the full series including zeros, not just non-NaN values. This matches `evaluator_v5.ipynb` exactly and affects which features are classified as `sparse_*`.

### Mode → gene space mapping

| Mode | num_classes | dont_care sentinel | Fuzzy value names |
|---|---|---|---|
| `binary` | 2 | 2 | Inactive (0), Active (1) |
| `ternary` | 3 | 3 | Negative (-1), Neutral (0), Positive (1) |
| `positive` | 5 | 5 | Very Low, Low, Medium, High, Very High |
| `sparse_positive` | 5 | 5 | Very Low, Low, Medium, High, Very High |
| `sparse_signed` | 5 | 5 | Strong Negative, Weak Negative, Exactly Zero, Weak Positive, Strong Positive |
| `signed` | 10 | 10 | Extreme Bearish … Extreme Bullish |

The mode determines how many gene values are valid for that feature's chromosome slot. A `signed` feature has 10 possible active values plus a dont_care sentinel (11 total), while a `binary` feature has only 2 active values plus a dont_care (3 total). This directly affects the search space size in Phase 2.

---

## 3. The Full Selection Pipeline — `Feature_Selector.select_features`

### Step 1 — Identify candidate columns

Excludes `LABEL_COLUMNS`, `META_COLUMNS`, `INTERNAL_COLUMNS`, and any column starting with `_`. Everything else is a candidate feature.

### Step 2 — Detect feature modes

Calls `Feature_Detector.detect_all_modes` on the training split. Modes are stored and used later for the MI discrete mask and for the chromosome gene space.

### Step 3 — Remove near-zero dispersion features

```python
top_freq = series.value_counts(normalize=True, dropna=False).iloc[0]
if top_freq > PHASE1_DISPERSION_THRESHOLD:
    drop it
```

`PHASE1_DISPERSION_THRESHOLD = 0.95` (config). A feature where 95%+ of values are identical carries almost no information. Dropping it reduces the gene space and speeds up Phase 2.

**Effect of changing this parameter:**
- Increasing toward 1.0: keeps more near-constant features, adding noise to the gene space.
- Decreasing toward 0.5: aggressively removes features, potentially discarding useful ones with moderate concentration.

### Step 4 — Build direction-specific target signal — `_build_target`

This is the most important step for ensuring long and short feature lists diverge.

**When `PHASE1_ASYMMETRIC_TARGET = True` (default):**

A 3-class signed PnL surrogate is built:
- Class 2 (win): TP was hit and came before SL
- Class 0 (loss): SL was hit and came before TP
- Class 1 (neutral): neither side cleanly resolved

For **long**:
```
tp_level = open_next × (1 + PHASE2_TP/100)
sl_level = open_next × (1 − PHASE2_SL/100)
win  = (max_288 >= tp_level AND max came before min) OR (max_288 >= tp_level AND min_288 > sl_level)
loss = (min_288 <= sl_level AND min came before max) OR (min_288 <= sl_level AND max_288 < tp_level)
```

For **short**, the logic is mirrored: TP requires the minimum to be reached first.

This 3-class target gives mutual information a richer signal than binary success/failure. More importantly, the long and short targets are genuinely different because the win/loss conditions are asymmetric.

**When `PHASE1_ASYMMETRIC_TARGET = False` (legacy):**
Binary target: 1 if the trade would have been a win, 0 otherwise. Long and short targets are nearly identical, causing the feature lists to overlap heavily.

**Note:** The target uses `PHASE2_TP` and `PHASE2_SL` as the TP/SL levels. This means Phase 1 feature selection is implicitly calibrated to the Phase 2 risk parameters. If you change `PHASE2_TP` or `PHASE2_SL`, Phase 1 should be re-run.

### Step 5 — Score each feature per symbol using Mutual Information

For each symbol independently:
```python
scores = mutual_info_classif(X, y, discrete_features=discrete_mask, random_state=42)
```

`discrete_features` is set to `True` for `binary` and `ternary` modes (which use the categorical MI estimator) and `False` for all other modes (which use the k-NN MI estimator). This distinction matters: using the continuous estimator on binary features would give incorrect scores.

Symbols where the target has only one class (e.g., all wins or all losses) are skipped for that symbol's scoring.

### Step 6 — Compute cross-symbol stability

```python
stability = 1.0 − (std(per_symbol_scores) / mean(per_symbol_scores))
```

Clipped to `[0, 1]`. A feature that scores high on some symbols but near-zero on others has high variance and low stability. A feature that scores consistently across all symbols has stability close to 1.

**Why this matters:** A rule that only works on 2 of 10 symbols will fail the Phase 3 symbol-coverage gate. Selecting features with high cross-symbol stability biases the search toward rules that generalize across the portfolio.

### Step 7 — Final score = relevance × stability

```python
final_score = mean(per_symbol_scores) × stability
```

This multiplicative combination means a feature must be both relevant (high MI) and stable (consistent across symbols) to rank highly. A feature with high MI on one symbol but zero on others will have low stability and thus a low final score.

### Step 7b — Stationarity filter

After scoring, features are filtered by temporal stability across folds. This is controlled by `PHASE1_STATIONARITY_STRATIFY`:

**Chronological folds (`PHASE1_STATIONARITY_STRATIFY = "chronological"`):**
The training data is split into `PHASE1_STATIONARITY_FOLDS` equal time windows. MI is computed in each window. A feature passes if:
1. Its coefficient of variation (std/mean of fold scores) ≤ `PHASE1_STATIONARITY_CV_MAX`
2. Its rank (by MI score) does not shift by more than `PHASE1_STATIONARITY_RANK_DRIFT_MAX` positions across folds

**Regime folds (`PHASE1_STATIONARITY_STRATIFY = "regime"`):**
Uses the GMM/KMeans regime labels from `regime_cluster.py` to define folds. Each regime is one fold. This tests whether a feature is informative across different market regimes (trending, ranging, volatile), not just different time periods. Rank drift uses `PHASE1_STATIONARITY_RANK_DRIFT_MAX` as-is (same as chronological); if fewer than two regime folds have enough rows, the filter falls back to chronological folds.

**Effect of stationarity parameters:**
- `PHASE1_STATIONARITY_CV_MAX = 1.0`: A CV of 1.0 means std = mean, which is quite permissive. Decreasing this (e.g., to 0.5) enforces stricter temporal consistency but may remove too many features.
- `PHASE1_STATIONARITY_RANK_DRIFT_MAX = 10`: A feature can shift up to 10 rank positions across folds. Decreasing this enforces stricter rank stability.
- `PHASE1_STATIONARITY_FOLDS = 3`: Number of folds. More folds = more granular stability check but requires more data per fold.

### Step 8 — Within-mode redundancy removal

Pairwise Pearson correlation is computed within each mode group. If two features have `|corr| > 0.95`, the lower-scored one is dropped. This prevents the evolutionary search from wasting gene slots on near-identical features.

### Step 9 — Select top K features

Features are sorted by `final_score` descending. The top `PHASE1_TOP_K_FEATURES × 2` candidates are kept (the overlap reduction step in `run()` then trims to `PHASE1_TOP_K_FEATURES`).

---

## 4. Overlap Reduction — `_reduce_overlap`

After both directions are scored independently, the overlap between long and short feature lists is capped at `PHASE1_MAX_FEATURE_OVERLAP × PHASE1_TOP_K_FEATURES`.

For shared features that exceed the cap, the one with the smaller score difference between directions is removed from the direction where it scores lower. The removed slots are backfilled from the next-ranked features in that direction's pool.

**Effect of `PHASE1_MAX_FEATURE_OVERLAP = 0.50`:** At most 50% of the top-K features can be shared between long and short. Decreasing this forces more divergence between the two gene spaces, which can help if long and short rules are currently too similar. Increasing it allows more overlap, which may be appropriate if the market is symmetric.

---

## 5. Regime Detection — `regime_cluster.py`

Used when `PHASE1_STATIONARITY_STRATIFY = "regime"`. Computes macro market regimes using a Dual-Window Rolling Linear Regression on daily prices, then applies median pre-filtering and a minimum duration constraint to merge short-term noise.

### Algorithm Details
For each symbol:
1. Daily close price (`label_open_next`) is resampled and smoothed using a 3-day rolling mean.
2. Fast (10-day) and Slow (24-day) linear regressions are run daily.
3. Market regimes are assigned based on slope and R2:
   - **Bullish (Regime 2)**: Fast trend is strong and positive, and slow trend is not strongly bearish.
   - **Bearish (Regime 1)**: Fast trend is strong and negative, and slow trend is not strongly bullish.
   - **Sideways (Regime 0)**: Otherwise.
4. Short-term noise is removed using a 9-day median filter.
5. Blocks shorter than 14 days are merged iteratively with their longer neighbor.

### Configuration Reference

| Parameter | Default | Effect |
|---|---|---|
| `PHASE1_REGIME_FAST_WINDOW` | `10` | Fast regression window size (days). |
| `PHASE1_REGIME_SLOW_WINDOW` | `24` | Slow regression window size (days). |
| `PHASE1_REGIME_FAST_R2_THRESHOLD` | `0.20` | R2 threshold for the fast regression to trigger a trend. |
| `PHASE1_REGIME_SLOW_R2_THRESHOLD` | `0.25` | R2 threshold for the slow trend filter. |
| `PHASE1_REGIME_FAST_SLOPE_THRESHOLD` | `0.0016` | Price slope threshold for fast regression trend detection. |
| `PHASE1_REGIME_SLOW_SLOPE_THRESHOLD` | `0.0010` | Price slope threshold for slow regression trend detection. |
| `PHASE1_REGIME_MED_WINDOW` | `9` | Median filter window size (days) to clean high-frequency noise. |
| `PHASE1_REGIME_MIN_DAYS` | `14` | Minimum duration (days) for any detected regime block. |

---

## 6. Configuration Reference

| Parameter | Default | Technical effect |
|---|---|---|
| `PHASE1_DISPERSION_THRESHOLD` | `0.95` | Drop features where the most common value appears in >95% of rows. Increase → keep more near-constant features (more noise). Decrease → more aggressive pruning. |
| `PHASE1_TOP_K_FEATURES` | `20` | Final number of features per direction. Increasing expands the Phase 2 chromosome length K, growing the search space exponentially. Decreasing reduces search space but may miss useful features. |
| `PHASE1_MAX_FEATURE_OVERLAP` | `0.50` | Maximum fraction of features shared between long and short. Decrease to force more direction-specific gene spaces. |
| `PHASE1_ASYMMETRIC_TARGET` | `True` | Use 3-class signed PnL target. Setting to `False` uses binary target, causing long/short lists to converge. |
| `PHASE1_STATIONARITY_FOLDS` | `3` | Number of folds for stationarity check. Set to `< 2` to disable the stationarity filter entirely. |
| `PHASE1_STATIONARITY_CV_MAX` | `1.0` | Maximum coefficient of variation across folds. Decrease to enforce stricter temporal consistency. |
| `PHASE1_STATIONARITY_RANK_DRIFT_MAX` | `10` | Maximum rank shift across folds. Decrease to enforce stricter rank stability. |
| `PHASE1_STATIONARITY_STRATIFY` | `"chronological"` | `"regime"` uses rolling regression regimes as folds; `"chronological"` uses time windows. |
| `PHASE1_SAMPLING_TOTAL` | `600_000` | Total rows sampled for Phase 2 GPU evaluation (equal per symbol). This is the primary GPU memory knob. Decrease if you get OOM errors in Phase 2. |

---

## 7. Outputs

- `outputs/selected_features_long.json` — JSON array of `{"name": str, "mode": str, "score": float}` dicts, sorted by score descending.
- `outputs/selected_features_short.json` — Same for short direction.
- `outputs/phase1_regime_cluster.joblib` — Saved regime configuration bundle (only when `STRATIFY = "regime"`).

The `score` field in the output is the `relevance × stability` product. It is used only for ranking; the absolute value has no direct interpretation.
