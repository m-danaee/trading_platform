# task-3: Extend search space from 12 → 26 hyperparameters

## Goal
Add 14 additional impactful hyperparameters to the `SEARCH_SPACE` dict in `gpu_fuzzy_trader/optuna_search.py`. No other logic changes needed.

## Files to modify
- `gpu_fuzzy_trader/optuna_search.py` — update `SEARCH_SPACE` dict only

## New parameters to add (exact values)

```python
# Phase 2 — Risk params during rule evolution
"PHASE2_TP": [1.0, 1.5, 2.0, 3.0, 4.0],
"PHASE2_SL": [0.5, 1.0, 1.5, 2.0],
"PHASE2_CAPITAL_PCT": [15, 20, 25, 30, 40],

# Phase 2 — Quality floors
"PHASE2_RETURN_FLOOR_PCT": [-2, 0, 2, 5],
"PHASE2_VAL_RETURN_FLOOR_PCT": [0, 0.5, 2, 5],
"PHASE2_PROFIT_FACTOR_FLOOR": [1.0, 1.05, 1.10, 1.20],

# Phase 2 — Search budget & diversity
"PHASE2_STAGE_B_GENERATIONS": [25, 35, 45, 60, 80],
"PHASE2_DIVERSITY_PENALTY": [3, 5, 8, 12, 16],
"PHASE2_FEASIBILITY_VIOLATION_WEIGHT": [10, 15, 25, 35, 50],

# Phase 2 — Admission gates
"PHASE2_MONTHLY_ADMISSION_MIN_PROFITABLE_RATIO": [0.3, 0.4, 0.5, 0.6, 0.7],
"PHASE2_MIN_PROFITABLE_SYMBOLS": [2, 3, 4, 5, 6],

# Phase 3 — Per-symbol gates
"PHASE3_PER_SYMBOL_MIN_TRADES": [4, 6, 8, 12, 16],

# RB Governor
"RB_KEEP_TOP_RULES": [40, 60, 80, 100, 120],
"RB_MAX_PAIR_OVERLAP": [0.15, 0.20, 0.25, 0.30, 0.40],
```

## What NOT to change
- Do NOT change any existing 12 parameter values
- Do NOT change the `compute_score`, `collect_phase5_metrics`, `objective`, `main`, or any other function
- Do NOT change imports, CLI arguments, docstrings
- Only add lines to the `SEARCH_SPACE` dict

## Verification
```bash
# Import test
.venv/bin/python -c "import gpu_fuzzy_trader.optuna_search; s=gpu_fuzzy_trader.optuna_search.SEARCH_SPACE; print(f'{len(s)} params'); assert len(s)==26"

# Help still works
.venv/bin/python -m gpu_fuzzy_trader.optuna_search --help
```

## Acceptance criteria
1. `SEARCH_SPACE` contains exactly 26 entries (12 original + 14 new)
2. All 14 new parameters have exact values specified above
3. All 12 original parameters unchanged
4. Script imports without errors
5. `--help` still shows all CLI arguments
