# راهنمای Hyperparameters — GPU-Fuzzy Trading Pipeline

> **مخاطب:** Data Scientist  
> **هدف:** درک کامل هر فاز، function های مهم، و تأثیر دقیق هر hyperparameter

---

## نقشه کلی Pipeline

```
data/train.csv
      │
      ▼
[Phase 0] Data_Loader + Data_Splitter
      │
      ├──► train_75.parquet (75%)
      └──► validation_25.parquet (25%)
                    │
                    ▼
[Phase 1] Feature_Selector
      │
      ├──► selected_features_long.json
      └──► selected_features_short.json
                    │
                    ▼
[Phase 2] Rule_Pool_Generator (NSGA-III)
      │
      ├──► phase2_long_pool.json
      └──► phase2_short_pool.json
                    │
                    ▼
[Phase 3] Rule_Set_Selector (Greedy + NSGA-II)
      │
      ├──► long.json (Phase 2 TP/SL)
      └──► short.json (Phase 2 TP/SL)
                    │
                    ▼
[Phase 4] WalkForwardRiskOptimizer (Optuna)
      │
      ├──► long.json (optimized TP/SL/capital)
      └──► short.json (optimized TP/SL/capital)
                    │
                    ▼
[Phase 5] OOS_Evaluator (data/test.csv)
      │
      └──► outputs/reports/
```

---

## فهرست داکیومنت‌ها

| فایل | محتوا |
|------|-------|
| [phase0_shared.md](phase0_shared.md) | Data loading، splitting، backtest engine، پارامترهای مشترک |
| [phase1_feature_selection.md](phase1_feature_selection.md) | MI scoring، stationarity filter، regime clustering |
| [phase2_rule_pool.md](phase2_rule_pool.md) | NSGA-III، chromosome encoding، penalties، archive |
| [phase3_rule_set.md](phase3_rule_set.md) | Greedy search، NSGA-II refinement، overfitting gates |
| [phase4_wf_risk.md](phase4_wf_risk.md) | Walk-forward، Optuna، TP/SL/capital optimization |
| [phase5_oos.md](phase5_oos.md) | OOS evaluation، metrics، تفسیر نتایج |

---

## جدول سریع همه Hyperparameters

### Phase 0 — Shared

| پارامتر | پیش‌فرض | فایل |
|---------|---------|------|
| `INITIAL_CAPITAL` | `1000.0` | config.py |
| `LEVERAGE` | `1.0` | config.py |
| `FEE_PCT` | `0.20` | config.py |
| `MAX_HOLD_CANDLES` | `288` | config.py |
| `MAX_TOTAL_EXPOSURE_PCT` | `100.0` | config.py |
| `MIN_POSITION_NOTIONAL` | `1.0` | config.py |
| `TAIL_DROP_ROWS` | `288` | config.py |
| `SORTINO_CAP` | `5.0` | config.py |
| `SORTINO_SCALE` | `3.0` | config.py |

### Phase 1 — Feature Selection

| پارامتر | پیش‌فرض | فایل |
|---------|---------|------|
| `PHASE1_DISPERSION_THRESHOLD` | `0.95` | config.py |
| `PHASE1_TOP_K_FEATURES` | `20` | config.py |
| `PHASE1_MAX_FEATURE_OVERLAP` | `0.50` | config.py |
| `PHASE1_ASYMMETRIC_TARGET` | `True` | config.py |
| `PHASE1_STATIONARITY_FOLDS` | `3` | config.py |
| `PHASE1_STATIONARITY_CV_MAX` | `1.0` | config.py |
| `PHASE1_STATIONARITY_RANK_DRIFT_MAX` | `10` | config.py |
| `PHASE1_STATIONARITY_STRATIFY` | `"chronological"` | config.py |
| `PHASE1_REGIME_N_CLUSTERS` | `3` | config.py |
| `PHASE1_REGIME_MIN_SAMPLES` | `100` | config.py |
| `PHASE1_REGIME_CLUSTERER` | `"gmm"` | config.py |
| `PHASE1_REGIME_GMM_REG_COVAR` | `1e-6` | config.py |
| `PHASE1_SAMPLING_TOTAL` | `600_000` | config.py |

### Phase 2 — Rule Pool

| پارامتر | پیش‌فرض | فایل |
|---------|---------|------|
| `PHASE2_TP` | `3.0` | config.py |
| `PHASE2_SL` | `1.5` | config.py |
| `PHASE2_CAPITAL_PCT` | `48.0` | config.py |
| `MIN_CONDITIONS` | `3` | config.py |
| `MAX_CONDITIONS` | `4` | config.py |
| `MIN_TRADE_SUPPORT` | `300` | config.py |
| `SUPPORT_PENALTY_MAX` | `50.0` | config.py |
| `MIN_TRADE_POOL_FLOOR` | `75` | config.py |
| `PHASE2_DIVERSITY_HAMMING_THRESHOLD` | `2` | config.py |
| `PHASE2_DIVERSITY_PENALTY` | `5.0` | config.py |
| `PHASE2_POPULATION_SIZE` | `200` | config.py |
| `PHASE2_GENERATIONS` | `200` | config.py |
| `PHASE2_ARCHIVE_MAX_SIZE` | `500` | config.py |
| `PHASE2_ARCHIVE_SEED_FRACTION` | `0.35` | config.py |
| `PHASE2_JOINT_TRAIN_VAL` | `True` | config.py |
| `PHASE2_REGIME_SUPPORT_ENABLED` | `True` | config.py |
| `PHASE2_REGIME_CONCENTRATION_MIN` | `0.90` | config.py |
| `PHASE2_REGIME_MIN_WIN_RATE` | `0.40` | config.py |

### Phase 3 — Rule Set

| پارامتر | پیش‌فرض | فایل |
|---------|---------|------|
| `PHASE3_MIN_RULES` | `2` | config.py |
| `PHASE3_MAX_RULES` | `3` | config.py |
| `PHASE3_MIN_SYMBOL_COVERAGE` | `7` | config.py |
| `PHASE3_USE_GPU` | `False` | config.py |
| `PHASE3_USE_PARALLEL_BATCH` | `True` | config.py |
| `PHASE3_BATCH_WORKERS` | `min(32, cpu_count)` | config.py |
| `PHASE3_NUMBA_ENABLED` | `True` | config.py |
| `PHASE3_REFINE_GENERATIONS` | `80` | config.py |
| `PHASE3_REFINE_POP_SIZE` | `100` | config.py |
| `PHASE3_GREEDY_WEIGHTS` | `(1.0, 0.7, 0.5)` | config.py |
| `PHASE3_SYMBOL_CONSISTENCY_WEIGHT` | `10.0` | config.py |
| `PHASE3_USE_TRAIN_TARGET` | `True` | config.py |
| `PHASE3_VAL_SORTINO_RATIO_GATE` | `0.5` | config.py |
| `PHASE3_VAL_DRAWDOWN_RATIO_GATE` | `1.5` | config.py |
| `PHASE3_PER_RULE_MIN_VAL_TRADES_PER_SYMBOL` | `5` | config.py |
| `PHASE3_TRAIN_VAL_CORR_WEIGHT` | `5.0` | config.py |
| `PHASE3_VAL_GATE_PENALTY` | `75.0` | config.py |

### Phase 4 — Risk Optimization

| پارامتر | پیش‌فرض | فایل |
|---------|---------|------|
| `PHASE4_TP_MIN` | `2.0` | config.py |
| `PHASE4_TP_MAX` | `4.0` | config.py |
| `PHASE4_SL_MIN` | `1.0` | config.py |
| `PHASE4_SL_MAX` | `2.0` | config.py |
| `PHASE4_CAPITAL_PCT_MIN` | `10.0` | config.py |
| `PHASE4_CAPITAL_PCT_MAX` | `50.0` | config.py |
| `PHASE4_TP_STEP` | `0.2` | config.py |
| `PHASE4_SL_STEP` | `0.2` | config.py |
| `PHASE4_CAPITAL_STEP` | `5.0` | config.py |
| `PHASE4_TOTAL_CAP_PENALTY` | `2.0` | config.py |
| `PHASE4_N_TRIALS` | `1000` | config.py |
| `PHASE4_WF_SPLITS` | `2` | config.py |
| `PHASE4_MAX_WORST_DRAWDOWN_PCT` | `15.0` | config.py |
| `PHASE4_SAMPLER` | `"nsga2"` | config.py |
| `PHASE4_SEED` | `42` | config.py |
| `PHASE4_N_JOBS` | `1` | config.py |
| `PHASE4_HARD_CAP_NORMALIZE` | `True` | config.py |

---

## راهنمای سریع عیب‌یابی

### مشکل: Pool خالی است
→ کاهش `MIN_TRADE_SUPPORT` و `MIN_TRADE_POOL_FLOOR`  
→ افزایش `PHASE2_GENERATIONS`

### مشکل: Overfitting (train خوب، test بد)
→ افزایش `PHASE3_VAL_SORTINO_RATIO_GATE`  
→ افزایش `PHASE3_VAL_GATE_PENALTY`  
→ کاهش `PHASE1_STATIONARITY_CV_MAX`

### مشکل: Pipeline خیلی کند است
→ کاهش `PHASE1_SAMPLING_TOTAL`  
→ کاهش `PHASE2_POPULATION_SIZE` و `PHASE2_GENERATIONS`  
→ افزایش `PHASE3_BATCH_WORKERS`

### مشکل: Long و Short feature list مشابه هستند
→ کاهش `PHASE1_MAX_FEATURE_OVERLAP`  
→ تأیید `PHASE1_ASYMMETRIC_TARGET = True`

### مشکل: Drawdown خیلی بالا است
→ کاهش `PHASE4_MAX_WORST_DRAWDOWN_PCT`  
→ کاهش `PHASE4_CAPITAL_PCT_MAX`  
→ کاهش `MAX_TOTAL_EXPOSURE_PCT`
