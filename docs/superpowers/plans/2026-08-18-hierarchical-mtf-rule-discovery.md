# Hierarchical Multi-Timeframe Rule Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Completely replace the handcrafted 4-state trend/regime classifier with an evolutionary Hierarchical Multi-Timeframe (HWC $\rightarrow$ MWC $\rightarrow$ LWC) Rule Discovery system with OOF cross-fitting, decoupled direction/strength ensembling, asymmetric soft-veto composition, and trade retention safeguards.

**Architecture:** 
1. `multi_timeframe.py` constructs complete, UTC-aligned HWC (4H) and MWC (1H) bars from 15m raw tape and computes independent indicators causally aligned to 15m execution rows.
2. Directional Evaluators evaluate ATR-normalized forward return labels using Directional Edge, MCC, soft coverage penalties, and temporal stability over shared purged temporal folds.
3. Decoupled Ensembler generates continuous `(Direction, Strength)` scores for HWC and MWC.
4. MTF Composer applies asymmetric soft-veto filters on raw LWC entry triggers and verifies trade retention floors ($\ge 50\%$).
5. Pipeline, RB Governor, and Phase 5 OOS evaluate hierarchical strategy candidates using frozen `mtf_manifest.json`.

**Tech Stack:** Python 3.10+, NumPy, Pandas, Numba, Pytest.
**Spec:** `docs/superpowers/specs/2026-08-18-hierarchical-mtf-rule-discovery-design.md`

## Global Constraints
- Always use `.venv` for running commands.
- Do not run all tests together; run targeted test files with `PYTEST_LOW_MEMORY=1`.
- Clean up deprecated code and artifacts after migration.
- After code modifications, run `graphify update .`.

---

### Task 1: Causal Multi-Timeframe Data Engine

**Files:**
- Create: `gpu_fuzzy_trader/data/multi_timeframe.py`
- Test: `tests/unit/test_multi_timeframe.py`

**Interfaces:**
- Consumes: Raw 15m DataFrame with `datetime`, `symbol`, `open`, `high`, `low`, `close`, `volume`.
- Produces:
  - `build_complete_higher_bars(df: pd.DataFrame, timeframe_minutes: int) -> pd.DataFrame`
  - `compute_timeframe_features(df_bars: pd.DataFrame, timeframe_minutes: int) -> pd.DataFrame`
  - `align_htf_features_causal(lwc_df: pd.DataFrame, htf_features: pd.DataFrame, timeframe_minutes: int) -> pd.DataFrame`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_multi_timeframe.py
import numpy as np
import pandas as pd
import pytest
from gpu_fuzzy_trader.data.multi_timeframe import (
    build_complete_higher_bars,
    align_htf_features_causal,
)

def test_build_complete_higher_bars_utc_and_completeness():
    # 20 15m bars starting at 00:00 UTC
    dt = pd.date_range("2024-01-01 00:00", periods=20, freq="15min")
    df = pd.DataFrame({
        "datetime": dt,
        "symbol": "BTCUSDT",
        "open": np.linspace(100, 120, 20),
        "high": np.linspace(101, 121, 20),
        "low": np.linspace(99, 119, 20),
        "close": np.linspace(100.5, 120.5, 20),
        "volume": np.ones(20) * 10.0,
    })
    # 1H bars (4 x 15m) -> 5 complete bars
    mwc = build_complete_higher_bars(df, 60)
    assert len(mwc) == 5
    assert mwc["datetime"].iloc[0] == pd.Timestamp("2024-01-01 00:00")
    assert mwc["close"].iloc[0] == df["close"].iloc[3]
    assert mwc["volume"].iloc[0] == 40.0

    # 4H bars (16 x 15m) -> exactly 1 complete 4H bar (first 16 rows), 4 rows dropped as incomplete
    hwc = build_complete_higher_bars(df, 240)
    assert len(hwc) == 1
    assert hwc["datetime"].iloc[0] == pd.Timestamp("2024-01-01 00:00")
    assert hwc["volume"].iloc[0] == 160.0

def test_causal_alignment_no_lookahead():
    # 16 15m bars for 00:00 -> 04:00 4H candle
    dt = pd.date_range("2024-01-01 00:00", periods=18, freq="15min")
    df = pd.DataFrame({
        "datetime": dt,
        "symbol": "BTCUSDT",
        "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 10.0,
    })
    hwc = build_complete_higher_bars(df, 240)
    hwc["hwc_feature"] = 42.0
    aligned = align_htf_features_causal(df, hwc, 240)
    
    # 00:00 to 03:45 15m rows execute at 00:15 to 04:00.
    # The 4H bar closes at 04:00, so row at 03:45 (executing at 04:00) sees it. Earlier rows get NaN.
    assert np.isnan(aligned.loc[df["datetime"] == "2024-01-01 03:30", "hwc_feature"].iloc[0])
    assert aligned.loc[df["datetime"] == "2024-01-01 03:45", "hwc_feature"].iloc[0] == 42.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_LOW_MEMORY=1 .venv/bin/pytest tests/unit/test_multi_timeframe.py -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError`.

- [ ] **Step 3: Implement `gpu_fuzzy_trader/data/multi_timeframe.py`**

Implement complete UTC bar construction, missing constituent validation, independent feature generator, and causal forward-alignment (`align_htf_features_causal`).

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTEST_LOW_MEMORY=1 .venv/bin/pytest tests/unit/test_multi_timeframe.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gpu_fuzzy_trader/data/multi_timeframe.py tests/unit/test_multi_timeframe.py
git commit -m "feat(mtf): implement causal multi-timeframe data layer"
```

---

### Task 2: Directional & Conditional Evaluators & Rule Search Profiles

**Files:**
- Create: `gpu_fuzzy_trader/evolution/directional_evaluator.py`
- Modify: `gpu_fuzzy_trader/research_profile.py`
- Test: `tests/unit/test_directional_evaluator.py`

**Interfaces:**
- Consumes: Aligned feature matrix, price series, ATR series.
- Produces:
  - `compute_forward_movement_labels(close: np.ndarray, atr: np.ndarray, horizon_bars: int) -> np.ndarray`
  - `fit_directional_threshold(move: np.ndarray, quantile: float = 0.60) -> float`
  - `evaluate_directional_rule(active_mask: np.ndarray, labels: np.ndarray, base_rate: float, target_coverage: tuple[float, float]) -> tuple[float, float, float]` (returns DirectionalEdge, MCC, CoveragePenalty)
  - `RuleSearchProfile` dataclasses for HWC, MWC, and LWC.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_directional_evaluator.py
import numpy as np
import pytest
from gpu_fuzzy_trader.evolution.directional_evaluator import (
    compute_forward_movement_labels,
    fit_directional_threshold,
    evaluate_directional_rule,
)

def test_forward_movement_and_threshold():
    close = np.array([100.0, 102.0, 105.0, 101.0, 108.0, 110.0])
    atr = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    # horizon = 2
    # move[0] = (105 - 100)/2 = 2.5
    # move[1] = (101 - 102)/2 = -0.5
    # move[2] = (108 - 105)/2 = 1.5
    # move[3] = (110 - 101)/2 = 4.5
    moves = compute_forward_movement_labels(close, atr, horizon_bars=2)
    assert np.isclose(moves[0], 2.5)
    assert np.isclose(moves[1], -0.5)
    assert np.isnan(moves[4])  # Warmup/tail

    theta = fit_directional_threshold(moves[:4], quantile=0.50)
    assert theta > 0.0

def test_evaluate_directional_rule_metrics():
    labels = np.array([1, 1, -1, 0, 1, -1, 1, 0, 1, 1])  # 6 long, 2 short, 2 neutral
    active_mask = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 0], dtype=bool) # 4 active, all long
    edge, mcc, cov_penalty = evaluate_directional_rule(
        active_mask, labels, direction="long", target_coverage=(0.20, 0.60)
    )
    assert edge > 0.0
    assert mcc > 0.0
    assert cov_penalty == 0.0  # 4/10 = 40% in [20%, 60%]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_LOW_MEMORY=1 .venv/bin/pytest tests/unit/test_directional_evaluator.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `directional_evaluator.py` & update `research_profile.py`**

Implement vectorized directional metrics (Edge, MCC, soft continuous coverage penalties) and configure profiles for HWC (1-2 conditions, 20-60% coverage), MWC (1-3 conditions, 10-40% coverage), and LWC (2-4 conditions).

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTEST_LOW_MEMORY=1 .venv/bin/pytest tests/unit/test_directional_evaluator.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gpu_fuzzy_trader/evolution/directional_evaluator.py gpu_fuzzy_trader/research_profile.py tests/unit/test_directional_evaluator.py
git commit -m "feat(evolution): add directional evaluator and MTF rule profiles"
```

---

### Task 3: Master Temporal Folds, Purged Embargo & OOF Cross-Fitting

**Files:**
- Create: `gpu_fuzzy_trader/mtf/cross_fitting.py`
- Test: `tests/unit/test_mtf_cross_fitting.py`

**Interfaces:**
- Consumes: Temporal timestamps, horizon parameters ($K_{\text{HWC}}, K_{\text{MWC}}$), raw DataFrames.
- Produces:
  - `build_master_temporal_folds(df: pd.DataFrame, n_folds: int = 4, embargo_minutes: int = 1440) -> list[TemporalFold]`
  - `apply_purge_embargo(train_df: pd.DataFrame, pred_start_dt: pd.Timestamp, purge_minutes: int) -> pd.DataFrame`
  - `generate_oof_scores(...) -> pd.DataFrame`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_mtf_cross_fitting.py
import pandas as pd
import numpy as np
import pytest
from gpu_fuzzy_trader.mtf.cross_fitting import (
    build_master_temporal_folds,
    apply_purge_embargo,
)

def test_master_temporal_folds_purging():
    dt = pd.date_range("2024-01-01", "2024-04-30 23:45", freq="15min")
    df = pd.DataFrame({"datetime": dt, "symbol": "BTCUSDT", "close": 100.0})
    folds = build_master_temporal_folds(df, n_folds=3)
    assert len(folds) == 3
    
    # Check that training set for Fold 2 strictly purges samples whose forward label extends into test start
    test_start = folds[1].test_start
    train_subset = df[df["datetime"] < test_start]
    purged_train = apply_purge_embargo(train_subset, test_start, purge_minutes=1440) # 24h purge
    assert purged_train["datetime"].max() <= test_start - pd.Timedelta(minutes=1440)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_LOW_MEMORY=1 .venv/bin/pytest tests/unit/test_mtf_cross_fitting.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `cross_fitting.py`**

Implement `TemporalFold`, `build_master_temporal_folds`, `apply_purge_embargo`, and the OOF runner across folds.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTEST_LOW_MEMORY=1 .venv/bin/pytest tests/unit/test_mtf_cross_fitting.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gpu_fuzzy_trader/mtf/cross_fitting.py tests/unit/test_mtf_cross_fitting.py
git commit -m "feat(mtf): implement master temporal folds and purged cross-fitting"
```

---

### Task 4: Decoupled Ensemble Score (Direction & Strength) & Rule Archives

**Files:**
- Create: `gpu_fuzzy_trader/mtf/ensembler.py`
- Create: `gpu_fuzzy_trader/mtf/archives.py`
- Test: `tests/unit/test_mtf_ensembler.py`

**Interfaces:**
- Consumes: Rule population / archive, active condition masks, validation metrics.
- Produces:
  - `compute_rule_weights(rules: list[dict]) -> np.ndarray` (non-negative weights, $w_r = \max(0, \text{Edge}) \times \max(0, \text{Stability})$)
  - `compute_ensemble_direction_and_strength(active_matrix: np.ndarray, directions: list[str], weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]`
  - `save_mtf_rule_archive(timeframe: str, rules: list[dict], path: str) -> None`
  - `load_mtf_rule_archive(path: str) -> list[dict]`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_mtf_ensembler.py
import numpy as np
import pytest
from gpu_fuzzy_trader.mtf.ensembler import (
    compute_rule_weights,
    compute_ensemble_direction_and_strength,
)

def test_decoupled_direction_and_strength():
    # 2 Long rules, 1 Short rule
    rules = [
        {"direction": "long", "directional_edge": 0.15, "stability": 0.8},  # w = 0.12
        {"direction": "long", "directional_edge": 0.10, "stability": 0.6},  # w = 0.06
        {"direction": "short", "directional_edge": 0.20, "stability": 0.9}, # w = 0.18
    ]
    weights = compute_rule_weights(rules)
    assert (weights >= 0).all()
    assert np.isclose(weights[0], 0.12)
    
    # 3 timestamps:
    # t0: only short rule active -> Direction = -1.0, Strength = 0.18 / 0.36 = 0.50
    # t1: 1 long and 1 short active -> Direction = (0.12 - 0.18)/(0.12 + 0.18) = -0.20, Strength = 0.30/0.36 = 0.833
    # t2: no rules active -> Direction = 0.0, Strength = 0.0
    active_matrix = np.array([
        [False, False, True],
        [True, False, True],
        [False, False, False],
    ])
    directions = ["long", "long", "short"]
    direction_score, strength_score = compute_ensemble_direction_and_strength(
        active_matrix, directions, weights
    )
    assert np.isclose(direction_score[0], -1.0)
    assert np.isclose(strength_score[0], 0.5)
    assert np.isclose(direction_score[1], -0.2)
    assert np.isclose(direction_score[2], 0.0)
    assert np.isclose(strength_score[2], 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_LOW_MEMORY=1 .venv/bin/pytest tests/unit/test_mtf_ensembler.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `ensembler.py` and `archives.py`**

Implement decoupled scoring formulas, deduplication, and structured JSON archive persistence under `rule_archives/{hwc,mwc,lwc}/`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTEST_LOW_MEMORY=1 .venv/bin/pytest tests/unit/test_mtf_ensembler.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gpu_fuzzy_trader/mtf/ensembler.py gpu_fuzzy_trader/mtf/archives.py tests/unit/test_mtf_ensembler.py
git commit -m "feat(mtf): implement decoupled direction/strength ensembling and archives"
```

---

### Task 5: MTF Composer, Asymmetric Soft Veto, and Trade Retention Guard

**Files:**
- Create: `gpu_fuzzy_trader/mtf/composer.py`
- Create: `gpu_fuzzy_trader/mtf/diagnostics.py`
- Test: `tests/unit/test_mtf_composer.py`

**Interfaces:**
- Consumes: Raw LWC entry triggers, HWC/MWC direction and strength scores, veto thresholds.
- Produces:
  - `compose_hierarchical_signals(...) -> tuple[np.ndarray, dict]`
  - `compute_trade_retention_diagnostics(...) -> dict`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_mtf_composer.py
import numpy as np
import pytest
from gpu_fuzzy_trader.mtf.composer import (
    compose_hierarchical_signals,
    compute_trade_retention_diagnostics,
)

def test_asymmetric_soft_veto_and_retention():
    # 4 raw LWC long triggers
    lwc_triggers = np.array([1, 1, 1, 1, 0], dtype=np.int8)
    
    # t0: HWC supportive (+0.8, 0.4) -> Accepted
    # t1: HWC neutral (0.0, 0.0) -> Accepted
    # t2: HWC opposing (-0.75, 0.3) > V_HWC_LONG (0.65) -> Vetoed by HWC
    # t3: MWC opposing (-0.70, 0.3) > V_MWC_LONG (0.60) -> Vetoed by MWC
    # t4: No trigger -> No Trade
    hwc_dir = np.array([0.8, 0.0, -0.75, 0.2, 0.0])
    hwc_str = np.array([0.4, 0.0, 0.3, 0.2, 0.0])
    mwc_dir = np.array([0.5, 0.1, 0.0, -0.70, 0.0])
    mwc_str = np.array([0.3, 0.1, 0.0, 0.3, 0.0])
    
    signals, stats = compose_hierarchical_signals(
        lwc_triggers=lwc_triggers,
        direction="long",
        hwc_direction=hwc_dir,
        hwc_strength=hwc_str,
        mwc_direction=mwc_dir,
        mwc_strength=mwc_str,
        v_hwc=0.65,
        v_mwc=0.60,
        min_strength_hwc=0.15,
        min_strength_mwc=0.15,
    )
    
    assert (signals == np.array([1, 1, 0, 0, 0])).all()
    diag = compute_trade_retention_diagnostics(stats)
    assert diag["raw_triggers"] == 4
    assert diag["hwc_vetoed"] == 1
    assert diag["mwc_vetoed"] == 1
    assert diag["accepted_trades"] == 2
    assert np.isclose(diag["retention_ratio"], 0.50)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_LOW_MEMORY=1 .venv/bin/pytest tests/unit/test_mtf_composer.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `composer.py` and `diagnostics.py`**

Implement asymmetric soft-veto logic, staged veto rates (`hwc_veto_rate`, `mwc_incremental_veto_rate`), retention floors ($\ge 50\%$), and low-sample guardrails (`MIN_RETENTION_SAMPLE`).

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTEST_LOW_MEMORY=1 .venv/bin/pytest tests/unit/test_mtf_composer.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gpu_fuzzy_trader/mtf/composer.py gpu_fuzzy_trader/mtf/diagnostics.py tests/unit/test_mtf_composer.py
git commit -m "feat(mtf): implement composer, asymmetric soft veto, and retention guard"
```

---

### Task 6: Pipeline Integration (`run_pipeline.py`, `config.py`, `loader.py`, `cpu_engine.py`, `rb_governor.py`, `phase5_oos.py`)

**Files:**
- Modify: `gpu_fuzzy_trader/config.py`
- Modify: `gpu_fuzzy_trader/data/loader.py`
- Modify: `gpu_fuzzy_trader/backtest/cpu_engine.py`
- Modify: `gpu_fuzzy_trader/phases/phase2_rule_pool.py`
- Modify: `gpu_fuzzy_trader/rb_governor.py`
- Modify: `gpu_fuzzy_trader/phases/phase5_oos.py`
- Modify: `gpu_fuzzy_trader/run_pipeline.py`
- Test: `tests/unit/test_mtf_pipeline_integration.py`

**Interfaces:**
- Updates `run_pipeline.py` to run Phase 1H (HWC), Phase 1M (MWC), Phase 2 (LWC), MTF Composer, RB Governor, and Phase 5 OOS with `mtf_manifest.json`.
- Removes legacy mandatory context checks from `cpu_engine.py`, `phase2_rule_pool.py`, `rb_governor.py`, and `phase5_oos.py`.

- [ ] **Step 1: Write integration tests**

```python
# tests/unit/test_mtf_pipeline_integration.py
import pytest
from gpu_fuzzy_trader.run_pipeline import Pipeline_Runner

def test_pipeline_mtf_phases_contract():
    # Verify pipeline runner exposes new hierarchical phases and mtf_manifest contract
    runner = Pipeline_Runner()
    assert hasattr(runner, "run_phase1_hwc")
    assert hasattr(runner, "run_phase1_mwc")
    assert hasattr(runner, "run_phase2")
    assert hasattr(runner, "run_mtf_composition")
```

- [ ] **Step 2: Run test to verify failure**

Run: `PYTEST_LOW_MEMORY=1 .venv/bin/pytest tests/unit/test_mtf_pipeline_integration.py -v`
Expected: FAIL.

- [ ] **Step 3: Update `config.py`, `loader.py`, `cpu_engine.py`, `phase2_rule_pool.py`, `rb_governor.py`, `phase5_oos.py`, and `run_pipeline.py`**

- Connect `multi_timeframe.py` and `mtf/composer.py`.
- Remove legacy `tf_permission_*` and `lwc_pullback_reversal_*` assertions.
- Wrap strategies as `HierarchicalStrategyCandidate` in RB Governor and Phase 5 OOS.
- Generate and verify `mtf_manifest.json`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTEST_LOW_MEMORY=1 .venv/bin/pytest tests/unit/test_mtf_pipeline_integration.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gpu_fuzzy_trader/ tests/unit/test_mtf_pipeline_integration.py
git commit -m "feat(pipeline): integrate hierarchical MTF rule discovery across pipeline"
```

---

### Task 7: Legacy Cleanup & Final Verification

**Files:**
- Delete: `gpu_fuzzy_trader/data/trend_context.py`
- Delete: `gpu_fuzzy_trader/context_diagnostics.py`
- Delete: `scripts/diagnose_context_mask.py`
- Delete: `tests/unit/test_trend_context.py`
- Delete: `tests/unit/test_sparse_context_search_guards.py`
- Delete: `data/enriched/trend_context_manifest.json`

- [ ] **Step 1: Remove deprecated files and legacy test fixtures**

```bash
rm -f gpu_fuzzy_trader/data/trend_context.py
rm -f gpu_fuzzy_trader/context_diagnostics.py
rm -f scripts/diagnose_context_mask.py
rm -f tests/unit/test_trend_context.py
rm -f tests/unit/test_sparse_context_search_guards.py
rm -f data/enriched/trend_context_manifest.json
```

- [ ] **Step 2: Run focused unit test suite with `PYTEST_LOW_MEMORY=1`**

Run:
```bash
PYTEST_LOW_MEMORY=1 .venv/bin/pytest tests/unit/test_multi_timeframe.py tests/unit/test_directional_evaluator.py tests/unit/test_mtf_cross_fitting.py tests/unit/test_mtf_ensembler.py tests/unit/test_mtf_composer.py tests/unit/test_mtf_pipeline_integration.py -v
```
Expected: All tests PASS.

- [ ] **Step 3: Update knowledge graph**

Run: `graphify update .`

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(cleanup): remove legacy trend_context and finalize hierarchical MTF discovery"
```
