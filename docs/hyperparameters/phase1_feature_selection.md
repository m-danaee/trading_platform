# Phase 1 — Feature Selection

**Module:** [`gpu_fuzzy_trader/features/selector.py`](../../gpu_fuzzy_trader/features/selector.py)  
**Config prefix:** `PHASE1_*`  
**Data split:** Train only (75% chronological per symbol)

[← Back to index](README.md) | [Phase 0 — Shared](phase0_shared.md) | [Phase 2 →](phase2_rule_pool.md)

---

## Purpose and statistical framing

Phase 1 reduces the high-dimensional discretized feature space to a **direction-specific shortlist** (long vs short) before combinatorial rule search in Phase 2.

This is not a predictive model in the usual sense — it is **filter-based feature ranking**:

1. Score each feature with **mutual information (MI)** against a direction-specific target.
2. Penalize features whose MI **varies across symbols** (instability).
3. Drop **non-stationary** features (MI drift across regime or chronological folds).
4. Remove **redundant** features within the same fuzzy mode (correlation > 0.95).
5. Cap **long/short overlap** so the two directions can discover different alpha.

**Leakage guard:** scoring uses train split only. Labels enter only as the MI target, never as input features.

---

## Algorithm summary

For each direction (`long`, `short`):

```
1. Exclude LABEL_COLUMNS, META_COLUMNS, INTERNAL_COLUMNS, _-prefixed cols
2. Detect feature modes (binary, ternary, positive, signed, sparse_*)
3. Drop low-dispersion features (> PHASE1_DISPERSION_THRESHOLD identical values)
4. Build direction-specific target (asymmetric 3-class or legacy binary)
5. Per symbol: MI(feature, target) with discrete_features mask for binary/ternary
6. relevance = mean(per_symbol_MI)
7. stability = 1 - std(per_symbol_MI) / mean(per_symbol_MI)   [when mean > 0]
8. score = relevance × stability
9. Stationarity filter across N folds (default: regime-stratified MI; optional chronological)
10. Redundancy removal within mode (pairwise |corr| > 0.95, keep higher score)
11. Rank and take top 2×K candidates, then overlap reduction → K features per direction
```

Outputs: `outputs/selected_features_long.json`, `outputs/selected_features_short.json`.

---

## Hyperparameters

| Parameter                            | Default   | Valid range (typical) | ↑ increase                                                      | ↓ decrease                                                                |
| ------------------------------------ | --------- | --------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `PHASE1_DISPERSION_THRESHOLD`        | `0.95`    | 0.90–0.99             | Keeps more near-constant features; more noise genes in Phase 2  | Drops more features early; risk losing weak signals                       |
| `PHASE1_TOP_K_FEATURES`              | `15`      | 10–25                 | Wider Phase 2 search space; more overfit risk; slower evolution | Narrower search; may miss complementary features                          |
| `PHASE1_MAX_FEATURE_OVERLAP`         | `0.50`    | 0.0–1.0               | More shared long/short features; similar strategies             | Forces directional specialization; may drop strong features from one side |
| `PHASE1_STATIONARITY_FOLDS`          | `3`       | 2–5                   | More regimes / time slices for drift detection                  | Coarser splits; fewer features rejected for drift                         |
| `PHASE1_STATIONARITY_STRATIFY`       | `"regime"` | `regime`, `chronological` | Regime-robust MI stability (default)                        | Time-based folds (ablation)                                               |
| `PHASE1_STATIONARITY_CV_MAX`         | `1.0`     | 0.5–2.0               | Stricter; drops features with volatile MI across folds          | Permits regime-sensitive features; overfit risk                           |
| `PHASE1_STATIONARITY_RANK_DRIFT_MAX` | `30`      | 10–50                 | Stricter rank stability (chronological); caps regime mode       | Allows large rank swings between folds                                    |
| `PHASE1_REGIME_FEATURES`             | 8 cols    | —                     | Vol/trend/liquidity indicators for clustering                   | —                                                                         |
| `PHASE1_REGIME_MIN_SAMPLES`          | `100`     | 50–500                | Stricter per-regime MI estimates                                | Noisier regime MI; more regimes skipped                                   |
| `PHASE1_REGIME_CLUSTERER`            | `"gmm"`   | `gmm`, `kmeans`       | Elliptical clusters; `k-means++` init + `reg_covar`             | Faster, simpler KMeans                                                    |
| `PHASE1_REGIME_GMM_REG_COVAR`        | `1e-6`    | 1e-8–1e-4             | More stable covariance estimates                                | Less regularization                                                       |
| `PHASE1_ASYMMETRIC_TARGET`           | `True`    | bool                  | 3-class signed PnL surrogate; long ≠ short targets              | Legacy binary win flag; lists often nearly identical                      |
| `PHASE1_SAMPLING_TOTAL`              | `150_000` | 50k–300k              | **Phase 2 only:** less fitness noise; more GPU VRAM/RAM         | Faster Phase 2; noisier Sortino estimates                                 |

---

## Parameter details

### `PHASE1_DISPERSION_THRESHOLD`

Features where more than 95% of values are identical are removed before MI. These columns carry almost no information for classification and inflate multiple-testing burden.

- **Performance:** Lower threshold (e.g. 0.90) → cleaner Phase 2 gene pool, but may drop rare-event indicators.
- **Compute:** Fewer features → slightly faster Phase 1 and Phase 2.

### `PHASE1_TOP_K_FEATURES`

Each selected feature becomes one gene in the Phase 2 chromosome. With ternary/binary modes, alphabet size per gene is small, but **combinations explode** with K.

Default was lowered from 25 → 15 because MI scores below ~0.01 were floor noise; extra genes let the GA fit noise.

- **OOS effect:** K=15 balances expressiveness vs overfit. K>20 often widens pool without improving test Sortino.
- **Compute:** Phase 2 cost scales with population × generations × backtest rows; K affects decode/evaluate slightly.

### `PHASE1_MAX_FEATURE_OVERLAP`

After independent long/short ranking, shared features are trimmed until at most `floor(K × overlap)` names appear in both lists. When removing, the feature is dropped from the direction where it has the **lower relative score**.

- **Performance:** Critical for long/short diversification. Overlap=1.0 would make both directions search the same subspace.
- **Suggested range:** 0.3–0.5 for meaningfully different portfolios.

### Stationarity (`PHASE1_STATIONARITY_*` + `PHASE1_REGIME_*`)

**Default (`PHASE1_STATIONARITY_STRATIFY="regime"`):**

1. Z-score `PHASE1_REGIME_FEATURES` **within each symbol** (zero-variance columns → 0.0, no NaNs).
2. Fit a pooled **GMM** (`init_params="k-means++"`, `reg_covar=PHASE1_REGIME_GMM_REG_COVAR`) on train only → `N = PHASE1_STATIONARITY_FOLDS` regime labels.
3. Per feature, recompute MI **within each regime** (skip regimes with &lt; `PHASE1_REGIME_MIN_SAMPLES` rows).
4. Drop features failing CV and rank-drift checks (same as before).

**Regime rank drift:** effective limit = `min(PHASE1_STATIONARITY_RANK_DRIFT_MAX, n_valid_regimes - 1)`. With 3 regimes, max rank swing is **2** — a feature ranked #1 in high-vol but last in sideways is removed.

**Chronological ablation:** set `PHASE1_STATIONARITY_STRATIFY="chronological"` — time-ordered folds, `RANK_DRIFT_MAX=30` unchanged.

**Fallback:** missing regime columns, GMM failure, or &lt;2 valid regime folds → chronological stationarity.

**Artifacts:** `outputs/phase1_regime_cluster.joblib` (fitted scalers + clusterer for later phases).

A feature survives when:

- Coefficient of variation of per-fold MI scores ≤ `PHASE1_STATIONARITY_CV_MAX`
- Maximum rank change across folds ≤ effective rank drift limit

- **Generalization:** Stricter filters improve val/test stability at the cost of fewer features.
- **Failure mode:** If Phase 1 logs "stationarity filter removed everything", relax `CV_MAX` or `RANK_DRIFT_MAX` (chronological) or switch stratify mode.

### `PHASE1_ASYMMETRIC_TARGET`

When `True` (default), target is a 3-class signed outcome derived from TP/SL path logic using **`PHASE2_TP` and `PHASE2_SL`** (see `_build_target()`):

| Class                   | Long meaning              | Short meaning             |
| ----------------------- | ------------------------- | ------------------------- |
| Win (+1 → encoded 2)    | TP hit first / clear win  | TP hit first / clear win  |
| Loss (−1 → encoded 0)   | SL hit first / clear loss | SL hit first / clear loss |
| Neutral (0 → encoded 1) | Neither resolved cleanly  | Neither resolved cleanly  |

Long uses `label_max_288`; short uses `label_min_288`. This makes MI targets genuinely asymmetric.

When `False`, binary win flag only — useful for ablation, not recommended for production.

### `PHASE1_SAMPLING_TOTAL` (Phase 2 consumption)

Despite the name, this constant is read by **Phase 2** (`Rule_Pool_Generator`) to subsample train (and validation) rows before GPU backtests. Sampling is **stratified by symbol** (~equal rows per symbol).

- **Performance:** Higher N → fitness estimates closer to full-data backtest; rules generalize better to unsampled bars.
- **Memory:** Primary JAX memory knob. On WSL with limited GPU RAM, keep ≤ 150,000 (per config comment).
- **Trade-off:** 150k ≈ 15k rows/symbol for 10 symbols vs full train may be 500k+ rows.

---

## Redundancy filter (fixed at 0.95)

Not configurable in `config.py`. Within each feature mode group, if `|correlation| > 0.95`, the lower-scoring feature is dropped. Prevents Phase 2 from treating duplicate information as independent conditions.

---

## Interactions and tuning order

1. Set `PHASE1_ASYMMETRIC_TARGET=True` before tuning overlap.
2. Adjust `TOP_K` only after dispersion/stationarity are reasonable.
3. If Phase 2 OOM → reduce `PHASE1_SAMPLING_TOTAL` before reducing `PHASE2_POPULATION_SIZE`.
4. Phase 1 target uses `PHASE2_TP/SL` — if you change static Phase 2 risk, re-run Phase 1 for consistent MI targets.

---

## Diagnostics

| Artifact                           | What to check                                                           |
| ---------------------------------- | ----------------------------------------------------------------------- |
| `outputs/selected_features_*.json` | Feature count = K; scores decrease monotonically; modes look sensible   |
| Pipeline log                       | Stationarity/redundancy drop counts; shared feature count after overlap |
| Compare long vs short lists        | Jaccard similarity; should be ≤ overlap fraction                        |

---

## Code references

Target construction (asymmetric):

```494:563:gpu_fuzzy_trader/features/selector.py
def _build_target(df: pd.DataFrame, direction: str) -> pd.Series:
    ...
    tp = config.PHASE2_TP
    sl = config.PHASE2_SL
    ...
    if config.PHASE1_ASYMMETRIC_TARGET:
        target[win] = 2
        target[loss] = 0
        target[~win & ~loss] = 1
```

Overlap reduction:

```61:105:gpu_fuzzy_trader/features/selector.py
def _reduce_overlap(
    ranked: dict[str, list[dict]],
    max_overlap_pct: float,
    top_k: int,
) -> dict[str, list[dict]]:
```
