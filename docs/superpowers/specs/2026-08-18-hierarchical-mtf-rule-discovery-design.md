# Hierarchical Multi-Timeframe Rule Discovery System Specification

**Design Document & Architecture Contract**  
**Date:** 2026-08-18  
**Status:** Approved for Implementation Planning  
**Target Subsystem:** Multi-Timeframe (MTF) Data, Discovery, Ensembling, Composition, and OOS Validation  

---

## 1. System Overview & Core Objectives

This specification defines the complete architectural refactor replacing the legacy handcrafted 4-state regime/trend classifier (`Bullish`, `Bearish`, `Range`, `Noisy` in `gpu_fuzzy_trader/data/trend_context.py`) with an evolutionary **Hierarchical Multi-Timeframe Rule Discovery** system.

### 1.1 Root Problem Addressed
The legacy architecture relied on handcrafted quantile thresholds and mandatory intersection filters:
$$\text{Trade Signal} = \text{HWC Bullish} \land (\text{MWC Bullish} \lor \text{Range}) \land \text{LWC Pullback Reversal} \land \text{LWC Rule}$$
This created extreme context sparsity, frequently resulting in empty feasible rule pools, starved monthly validation windows, and failed out-of-sample (OOS) execution.

### 1.2 Target Architecture Principles
1. **Separation of Concerns Across Time Scales:**
   - **HWC (4H / 240m):** Broad directional evidence (macro trend bias).
   - **MWC (1H / 60m):** Conditional setup and continuation confirmation given macro bias.
   - **LWC (15m):** Sole execution trigger and entry timing with full barrier/risk simulation.
2. **Asymmetric Soft Veto Composition:**
   - LWC generates trade triggers.
   - HWC and MWC act strictly as soft veto / contradiction filters. Neutral or weakly positive signals never block an entry.
3. **Out-of-Fold (OOF) Hierarchical Training:**
   - Downstream models (MWC and LWC) train strictly on OOF scores produced by upstream models to eliminate target leakage and stacking overfit.
4. **Decoupled Ensemble Score (Direction vs. Strength):**
   - Direction bias and active evidence availability are calculated independently.
5. **Trade Retention Safeguard:**
   - Enforces minimum retention ratio ($\ge 50\%$) with staged diagnostics across the funnel.

---

## 2. Multi-Timeframe Causal Data Layer

**Module:** `gpu_fuzzy_trader/data/multi_timeframe.py` (replaces `trend_context.py`)

```text
Raw 15m OHLCV Tape (UTC-Aligned)
       │
       ├─► LWC (15m):  1 constituent 15m bar
       ├─► MWC (60m):  exactly 4 consecutive 15m bars
       └─► HWC (240m): exactly 16 consecutive 15m bars
```

### 2.1 Resampling & Continuity Contract
- **Fixed UTC Anchoring:** Candle boundaries are globally aligned to standard exchange intervals:
  - 1H: `00:00-01:00`, `01:00-02:00`, ...
  - 4H: `00:00-04:00`, `04:00-08:00`, `08:00-12:00`, `12:00-16:00`, `16:00-20:00`, `20:00-00:00`.
- **Strict Completeness Requirement:**
  - An MWC bar must contain exactly $4 \times 15\text{m}$ bars.
  - An HWC bar must contain exactly $16 \times 15\text{m}$ bars.
  - If any 15m bar is missing within a bucket, the higher timeframe bar is marked invalid and excluded.
- **Start/End & Live Incomplete Candle Policy:**
  - Partial boundary candles at the beginning/end of historical datasets are excluded.
  - Live/incoming incomplete HTF candles are never published until the candle has officially closed.

### 2.2 Independent Feature Computation
- Features (RSI, ATR, KAMA, Bollinger Bands, Realized Volatility, Momentum, Volume Spikes) are computed independently on the completed bars of each timeframe.
- Higher timeframe features maintain independent warmup periods.

### 2.3 Point-in-Time Causal Alignment
For any HTF candle spanning interval $[T, T + \Delta)$:
- $\text{Published Timestamp} = T + \Delta$.
- A 15m bar with open time $D$ executes at $D + 15\text{m}$.
- The aligned HTF state for execution at $D + 15\text{m}$ is the latest completed HTF candle whose close time satisfies:
  $$\text{HTF Close} \le D + 15\text{m}$$
- **Invariance Test:** Modifying data in any unfinished HTF candle must produce zero change in previously aligned features.

---

## 3. Rule Search Profiles & Directional Evaluators

**Modules:** `gpu_fuzzy_trader/research_profile.py`, `gpu_fuzzy_trader/evolution/directional_evaluator.py`

### 3.1 Profile Configurations

| Parameter / Dimension | HWC Profile | MWC Profile | LWC Profile |
| :--- | :--- | :--- | :--- |
| **Timeframe** | 240m (4H) | 60m (1H) | 15m |
| **Role** | Macro Directional Bias | Conditional Setup / Continuation | Entry Execution Trigger |
| **Max Conditions** | 1–2 | 1–3 | 2–4 |
| **Target Coverage Zone**| 20% – 60% | 10% – 40% | Direct Trade Count Targets |
| **Forward Horizon ($K$)**| $4\text{--}6$ bars ($16\text{--}24\text{h}$) | $3\text{--}6$ bars ($3\text{--}6\text{h}$) | N/A (Full Trade Simulation) |
| **Primary Evaluator** | Directional CPU Evaluator | Conditional Directional Evaluator | GPU/CPU Barrier Engine |

### 3.2 Labeling & Objectives

#### 3.2.1 ATR-Normalized Forward Return Labeling (HWC & MWC)
$$\text{move}_{t, K} = \frac{\text{Close}_{t+K} - \text{Close}_t}{\text{ATR}_t}$$
where $\text{ATR}_t$ is computed strictly from closed candles up to $t$.

Classification threshold $\theta$ is estimated via quantile on the **Train fold only** and frozen:
- $\text{move}_{t, K} > +\theta \implies \text{LONG\_FAVORABLE}$
- $\text{move}_{t, K} < -\theta \implies \text{SHORT\_FAVORABLE}$
- Otherwise $\implies \text{NEUTRAL}$

#### 3.2.2 Conditional MWC Labeling
MWC evaluates continuation given the upstream OOF HWC score:
- If $\text{HWC Score}_t \ge +\text{threshold}_{\text{support}}$: target is positive forward move ($> +\theta_{\text{MWC}}$).
- If $\text{HWC Score}_t \le -\text{threshold}_{\text{support}}$: target is negative forward move ($< -\theta_{\text{MWC}}$).

#### 3.2.3 NSGA-III Objectives (Max 4 per layer)
1. **$f_1 = -\text{Directional Edge}$**: where $\text{Directional Edge} = \text{Precision} - \text{Base Rate}$.
2. **$f_2 = -\text{MCC}$**: Matthews Correlation Coefficient on the active mask.
3. **$f_3 = \text{Coverage Penalty}$**: Soft continuous penalty outside the target zone $[C_{\min}, C_{\max}]$.
4. **$f_4 = \text{Temporal Instability Penalty}$**: Variance of MCC across validation splits.

#### 3.2.4 Hard Admission Constraints
- Active condition count $\le \text{Max Conditions}$.
- Minimum cross-symbol skill ($\text{MCC} > 0$ across all required symbols).
- Minimum fold support ($N_{\text{active}} \ge N_{\min}$).

---

## 4. Hierarchical Out-Of-Fold (OOF) Cross-Fitting & Ensembling

**Module:** `gpu_fuzzy_trader/mtf/cross_fitting.py`, `gpu_fuzzy_trader/mtf/ensembler.py`

### 4.1 Master Temporal Fold Scheme & Purged Embargo
All three timeframes share identical timestamp boundaries defined in the Master Fold Manifest:

```text
Fold 1: Train [T0, T1] ──(Purge)──► Predict OOF [T1, T2]
Fold 2: Train [T0, T2] ──(Purge)──► Predict OOF [T2, T3]
Fold 3: Train [T0, T3] ──(Purge)──► Predict OOF [T3, T4]
```

- **Purge Rule:** For prediction interval $[T_{\text{start}}, T_{\text{end}}]$, purge all training samples whose forward label/trade outcome horizon extends into or past $T_{\text{start}}$.
  - HWC Purge: $K_{\text{HWC}} \times 240\text{ min}$.
  - MWC Purge: $K_{\text{MWC}} \times 60\text{ min}$.
  - LWC Purge: $\text{Max Trade Duration} \times 15\text{ min}$.
- **Seed Period:** Fold 1 (which has no prior OOF history) is excluded from downstream training.

### 4.2 Decoupled Ensemble Score Architecture

For any timeframe at bar $t$:

#### Direction Score ($\in [-1.0, +1.0]$):
$$\text{Direction}_t = \begin{cases} 
\dfrac{W_{\text{long}}^{\text{active}}(t) - W_{\text{short}}^{\text{active}}(t)}{W_{\text{long}}^{\text{active}}(t) + W_{\text{short}}^{\text{active}}(t)}, & \text{if } W_{\text{long}}^{\text{active}}(t) + W_{\text{short}}^{\text{active}}(t) > 0 \\ 
0.0, & \text{otherwise (Neutral)} 
\end{cases}$$

#### Evidence Strength ($\in [0.0, 1.0]$):
$$\text{Strength}_t = \frac{W_{\text{long}}^{\text{active}}(t) + W_{\text{short}}^{\text{active}}(t)}{W_{\text{all}}}$$

#### Non-Negative Rule Weight Calculation:
$$w_r = \max(0, \text{DirectionalEdge}_r) \times \max(0, \text{StabilityScore}_r)$$
- Rules with $\text{OOF MCC} \le 0$ or non-positive skill are dropped during admission.
- Rules are deduplicated prior to ensembling to prevent identical genome clustering.

---

## 5. MTF Composer & Trade Retention Guard

**Module:** `gpu_fuzzy_trader/mtf/composer.py`

### 5.1 Asymmetric Contradiction Veto Logic

For an LWC Long entry trigger at bar $t$:
```python
veto_hwc = (
    hwc_direction[t] < -V_HWC_LONG 
    and hwc_strength[t] >= MIN_EVIDENCE_STRENGTH_HWC
)
veto_mwc = (
    mwc_direction[t] < -V_MWC_LONG 
    and mwc_strength[t] >= MIN_EVIDENCE_STRENGTH_MWC
)

if not lwc_trigger_long[t]:
    signal = NO_TRADE
elif veto_hwc or veto_mwc:
    signal = VETOED (NO_TRADE)
else:
    signal = ACCEPTED_LONG
```
Analogous logic applies to Short entries with $V_{\text{HWC\_SHORT}}$ and $V_{\text{MWC\_SHORT}}$.

- **Veto Calibration:** Asymmetric thresholds $(V_{\text{HWC\_LONG}}, V_{\text{HWC\_SHORT}}, V_{\text{MWC\_LONG}}, V_{\text{MWC\_SHORT}})$ and minimum evidence strength parameters are calibrated strictly on Train OOF predictions, then frozen.

### 5.2 Funnel Diagnostics & Retention Guard

$$\text{HWC Veto Rate} = \frac{N_{\text{LWC triggers vetoed by HWC}}}{N_{\text{Raw LWC triggers}}}$$
$$\text{MWC Incremental Veto Rate} = \frac{N_{\text{Candidates vetoed by MWC}}}{N_{\text{Candidates surviving HWC}}}$$
$$\text{Final Trade Retention Ratio} = \frac{N_{\text{Accepted MTF Trades}}}{N_{\text{Raw LWC triggers}}}$$

- **Retention Floor:** Hard floor $\ge 50\%$, preferred target $\ge 60\%$.
- **Statistical Support Guard:** If $N_{\text{Raw LWC triggers}} < \text{MIN\_RETENTION\_SAMPLE}$ (e.g. $< 15$), report `INSUFFICIENT_SUPPORT` rather than hard crashing.
- **Reporting Granularity:** Retention metrics logged globally, per direction, per symbol, per fold, and per monthly window.

---

## 6. Manifest, Archives & Frozen OOS Contract

### 6.1 Manifest Schema (`mtf_manifest.json`)
```yaml
schema_version: "2.0.0"
dataset_hashes:
  train: "sha256..."
  validation: "sha256..."
  oos: "sha256..."

timeframes:
  lwc_minutes: 15
  mwc_minutes: 60
  hwc_minutes: 240
  timezone: "UTC"
  candle_boundary: "fixed_utc"

labels:
  hwc_horizon_bars: 6
  mwc_horizon_bars: 4
  theta_per_oof_fold:
    fold_1: 0.85
    fold_2: 0.82
    fold_3: 0.84
  theta_final_train: 0.83

cross_fitting:
  fold_boundaries: [...]
  purge_durations_minutes:
    hwc: 1440
    mwc: 240
    lwc: 720

archives:
  hwc_archive_hash: "sha256..."
  mwc_archive_hash: "sha256..."
  lwc_archive_hash: "sha256..."

composer_parameters:
  v_hwc_long: 0.65
  v_hwc_short: 0.60
  v_mwc_long: 0.60
  v_mwc_short: 0.55
  min_evidence_strength_hwc: 0.15
  min_evidence_strength_mwc: 0.15
  retention_floor: 0.50

reproducibility:
  git_commit: "..."
  config_hash: "..."
```

### 6.2 Three-Stage Lifecycle & OOS Invariance
1. **Train:** Rule discovery, OOF cross-fitting, $\theta$ calibration per fold, ensemble weighting, initial veto calibration.
2. **Validation:** Final strategy admission, composer policy validation, RB Governor risk/portfolio tuning.
3. **True OOS (Strictly One-Shot):**
   - Evaluates incoming OOS candles with live indicator updates.
   - Strictly zero refitting, zero feature re-selection, zero weight re-estimation, and zero threshold recalibration.

---

## 7. Migration & Legacy Cleanup Plan

The following legacy components will be removed following successful integration of the new MTF engine:

### 7.1 Deprecated Files to Delete
- `gpu_fuzzy_trader/data/trend_context.py`
- `gpu_fuzzy_trader/context_diagnostics.py`
- `scripts/diagnose_context_mask.py`
- `tests/unit/test_trend_context.py`
- `tests/unit/test_sparse_context_search_guards.py`
- `data/enriched/train_new_hwc_mwc_lwc.csv`
- `data/enriched/test_new_hwc_mwc_lwc.csv`
- `data/enriched/trend_context_manifest.json`

### 7.2 Deprecated Columns & Configs
- Columns: `hwc_state`, `mwc_state`, `lwc_state`, `tf_permission_long`, `tf_permission_short`, `lwc_pullback_reversal_long`, `lwc_pullback_reversal_short`.
- Configs: `CONTEXT_*`, `REQUIRE_CONTEXT_IN_STRATEGY`, `LWC_PULLBACK_LOOKBACK`.
- Mandatory condition injections in CPU Engine, Phase 2, and RB Governor replaced with MTF Composer container.
