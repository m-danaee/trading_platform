# Plan: Optuna Hyperparameter Optimization for GPU Fuzzy Trader

## Goal

Create an Optuna-based hyperparameter search script that finds the best combination of the most impactful hyperparameters in `gpu_fuzzy_trader/config.py`. The objective function evaluates the pipeline output using `evaluator_v5.ipynb` metrics (test total return, max drawdown, profit factor).

## Approach

- **Script location**: `gpu_fuzzy_trader/optuna_search.py`
- **Method**: Each Optuna trial modifies config values, runs the pipeline, then scores the OOS result
- **Search space**: 26 hyperparameters with discrete/categorical choices around current values
- **Pruning**: Use MedianPruner to kill bad trials early
- **Storage**: Optuna SQLite DB at `outputs/optuna_study.db`

## Selected Hyperparameters to Search

### Phase 1 — Feature Selection (1 param)
| # | Parameter | Values |
|---|-----------|--------|
| 1 | `PHASE1_TOP_K_FEATURES` | [10, 15, 20, 25, 30] |

### Phase 2 — Trade Support & Pool Admission (3 params)
| # | Parameter | Values |
|---|-----------|--------|
| 2 | `MIN_TRADE_SUPPORT` | [30, 60, 90, 120, 150] |
| 3 | `MIN_TRADE_POOL_FLOOR` | [8, 12, 17, 25, 35] |
| 4 | `PHASE2_MONTHLY_ADMISSION_MIN_PROFITABLE_RATIO` | [0.3, 0.4, 0.5, 0.6, 0.7] |

### Phase 2 — Risk Params During Evolution (3 params)
| # | Parameter | Values |
|---|-----------|--------|
| 5 | `PHASE2_TP` | [1.0, 1.5, 2.0, 3.0, 4.0] |
| 6 | `PHASE2_SL` | [0.5, 1.0, 1.5, 2.0] |
| 7 | `PHASE2_CAPITAL_PCT` | [15, 20, 25, 30, 40] |

### Phase 2 — Quality Floors (5 params)
| # | Parameter | Values |
|---|-----------|--------|
| 8 | `PHASE2_RETURN_FLOOR_PCT` | [-2, 0, 2, 5] |
| 9 | `PHASE2_VAL_RETURN_FLOOR_PCT` | [0, 0.5, 2, 5] |
| 10 | `PHASE2_PROFIT_FACTOR_FLOOR` | [1.0, 1.05, 1.10, 1.20] |
| 11 | `PHASE2_MAX_DRAWDOWN_GATE` | [12.0, 18.0, 25.0, 30.0, 35.0] |
| 12 | `PHASE2_MAX_TRAIN_VAL_GAP_PCT` | [4.0, 6.0, 8.0, 10.0, 15.0] |

### Phase 2 — Search Budget & Diversity (6 params)
| # | Parameter | Values |
|---|-----------|--------|
| 13 | `PHASE2_POPULATION_SIZE` | [100, 150, 200, 250] |
| 14 | `PHASE2_MUTATION_RATE` | [0.12, 0.17, 0.22, 0.27, 0.32] |
| 15 | `PHASE2_STAGE_A_GENERATIONS` | [50, 70, 85, 100, 120] |
| 16 | `PHASE2_STAGE_B_GENERATIONS` | [25, 35, 45, 60, 80] |
| 17 | `PHASE2_DIVERSITY_PENALTY` | [3, 5, 8, 12, 16] |
| 18 | `PHASE2_FEASIBILITY_VIOLATION_WEIGHT` | [10, 15, 25, 35, 50] |

### Phase 2 — Pool & Cross-Symbol (2 params)
| # | Parameter | Values |
|---|-----------|--------|
| 19 | `PHASE2_KEEP_TOP_RULES` | [40, 60, 80, 100, 120] |
| 20 | `PHASE2_MIN_PROFITABLE_SYMBOLS` | [2, 3, 4, 5, 6] |

### Phase 3 — Rule Set Selection (2 params)
| # | Parameter | Values |
|---|-----------|--------|
| 21 | `PHASE3_VAL_RETURN_FLOOR_PCT` | [2.0, 4.0, 5.0, 7.0, 10.0] |
| 22 | `PHASE3_PER_SYMBOL_MIN_TRADES` | [4, 6, 8, 12, 16] |

### RB Governor (4 params)
| # | Parameter | Values |
|---|-----------|--------|
| 23 | `RB_MIN_TRAIN_RETURN` | [1.0, 2.0, 3.0, 5.0] |
| 24 | `RB_MIN_VALID_RETURN` | [1.0, 2.0, 3.0, 5.0] |
| 25 | `RB_KEEP_TOP_RULES` | [40, 60, 80, 100, 120] |
| 26 | `RB_MAX_PAIR_OVERLAP` | [0.15, 0.20, 0.25, 0.30, 0.40] |

**Total: 26 hyperparameters**

## Tasks

### task-1 ✅ DONE: Create `optuna_search.py` (12 params)
### task-2 ✅ DONE: Integration test verification

### task-3: Extend search space from 12 → 26 hyperparameters

**Files to modify:**
- `gpu_fuzzy_trader/optuna_search.py` — update `SEARCH_SPACE` dict

**What it does:**
1. Adds 14 new hyperparameters to `SEARCH_SPACE`
2. Each new param uses `suggest_categorical` with values from the table above
3. No other logic changes needed (objective, patching, CLI remain identical)

**Acceptance criteria:**
- `SEARCH_SPACE` contains exactly 26 entries
- All 14 new parameters have correct values
- Script imports and runs without errors
- `--help` still works
- Existing 12 params unchanged

## Dependencies

- `optuna` package
- Existing pipeline modules: `run_pipeline`, `config`
- `PYTEST_LOW_MEMORY=1` for test runs

## Non-goals

- Not tuning ALL 200+ config values
- Not modifying evaluator_v5.ipynb
- Not changing the pipeline architecture
- Not implementing distributed Optuna (single-machine only)
