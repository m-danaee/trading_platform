# Phase 3 — Rule Set Selection

**Modules:**
- `gpu_fuzzy_trader/phases/phase3_rule_set.py` → `Rule_Set_Selector` (orchestration)
- `gpu_fuzzy_trader/phases/phase3_greedy.py` → `greedy_rule_set_search` (greedy construction)
- `gpu_fuzzy_trader/phases/phase3_objectives.py` → `compute_phase3_objectives` (fitness function)
- `gpu_fuzzy_trader/phases/phase3_cache.py` → `Phase3EvalCache` (signal mask cache)

Phase 3 takes the pool of individual rules from Phase 2 and selects the best **combination** of 2–3 rules to form a complete trading strategy. The key insight is that a team of rules can cover more market conditions than any single rule, and the interaction between rules (priority ordering, symbol coverage) matters.

---

## 1. The Search Problem

**Input:** A pool of N rules (typically 50–500 from Phase 2).
**Output:** An ordered list of 2–3 rules (the strategy).
**Evaluation:** The combined strategy is backtested with `CPUBacktestEngine` on train and validation data. When `SPLIT_MODE == "purged_rolling_cv"`, each candidate is scored on **all CV folds** and the **worst fold** drives the objectives (see Section 5).

The search space is all ordered combinations of `PHASE3_MIN_RULES` to `PHASE3_MAX_RULES` rules from the pool, with no duplicate rules. For a pool of 100 rules and 2–3 rule sets, this is approximately:

```
C(100, 2) + C(100, 3) = 4950 + 161700 ≈ 166,650 combinations
```

This is too large for exhaustive search, so Phase 3 uses a two-stage approach: greedy construction followed by NSGA-II refinement.

---

## 2. Stage 1 — Greedy Construction (`phase3_greedy.py`)

### Algorithm

The greedy search builds the rule set incrementally:

**Round 1:** Evaluate every single rule from the pool as a 1-rule strategy. Select the rule with the highest scalar score.

**Round 2:** For each remaining pool rule, evaluate the current best set extended by that rule. Select the extension with the highest scalar score.

**Round 3 (if `MAX_RULES ≥ 3`):** Repeat for a third rule.

This is a greedy submodular maximization approach. It is not guaranteed to find the global optimum, but it provides a strong starting point for the NSGA-II refinement.

### Scalar score — `_scalar_score`

The greedy search uses a scalar score for tie-breaking (multi-objective quality is restored by the refinement step):

```python
score = w1 × primary_sortino − w2 × primary_dd + w3 × primary_wr − penalty
```

`PHASE3_GREEDY_WEIGHTS = (1.0, 0.7, 0.5)` (config): weights for Sortino, drawdown, and win rate.

**When `PHASE3_USE_TRAIN_TARGET = True` (default):** `primary_*` metrics come from the **training split**. Validation is used for gate penalties (Section 4). Recommended with purged CV so objectives are not tuned directly on a single lucky val quarter.

**When `PHASE3_USE_TRAIN_TARGET = False`:** `primary_*` metrics come from validation. Risks overfitting to the persisted `validation_25` block — avoid for short unless experimenting.

**`PHASE3_USE_MAXIMIN_SCORE`:** Greedy tie-breaking uses `min(train_return, val_return)` and similar minimax Sortino logic when both splits are available.

**Effect of `PHASE3_GREEDY_WEIGHTS`:** The first weight (Sortino) dominates by default. Increasing the second weight (drawdown) will favor lower-drawdown strategies. Increasing the third weight (win rate) will favor higher win-rate strategies.

---

## 3. Stage 2 — NSGA-II Refinement (`_run_nsga2_combinatorial`)

After greedy construction, the best rule set seeds a short NSGA-II search over rule set combinations.

### Population initialization — `_seed_population_from_greedy`

The initial population is built by mutating the greedy solution:
- 1 individual = the greedy solution itself
- `PHASE3_REFINE_POP_SIZE − 1` individuals = mutations of the greedy solution

This warm-starts the refinement from a good region of the search space.

### Crossover — `_crossover_rule_sets`

Two parent rule sets are combined by:
1. Concatenating all rules from both parents.
2. Shuffling and deduplicating (by condition set equality).
3. Trimming to `MAX_RULES` or padding to `MIN_RULES` from the pool.

### Mutation — `_mutate_rule_set`

With probability `mutation_rate = 0.3`:
- Replace a random rule with a new pool rule.
- Add a new pool rule (if below `MAX_RULES`).
- Remove a random rule (if above `MIN_RULES`).

### NSGA-II selection

Standard NSGA-II: non-dominated sort + crowding distance truncation. The Numba-accelerated implementations from `evolution/numba_ops.py` are used when `PHASE3_NUMBA_ENABLED = True`.

**Elitism:** The top `min(|Pareto front|, pop_size // 2)` individuals (by crowding distance) are carried over to the next generation unchanged.

### Budget

| Parameter | Default | Effect |
|---|---|---|
| `PHASE3_REFINE_POP_SIZE` | `100` | Population size for refinement. Increase for better Pareto coverage (linear compute cost). |
| `PHASE3_REFINE_GENERATIONS` | `80` | Number of refinement generations. Increase for more evolution. |

Total refinement evaluations ≈ `PHASE3_REFINE_POP_SIZE × PHASE3_REFINE_GENERATIONS = 8,000` per direction.

---

## 4. Fitness Function — `compute_phase3_objectives`

Three objectives (all minimized):

```
f1 = −primary_sortino + total_penalty
f2 = primary_dd + total_penalty
f3 = −primary_wr + total_penalty
```

where `primary_*` is from training (when `PHASE3_USE_TRAIN_TARGET = True`) or validation.

### Penalties

**Zero-trade penalty (100.0):** Applied if `train_trades == 0` OR `val_trades == 0`. A strategy that generates no trades on either split is useless.

**Coverage penalty:** Applied if `val_symbols_with_trades < PHASE3_MIN_SYMBOL_COVERAGE`:
```python
coverage_penalty = (PHASE3_MIN_SYMBOL_COVERAGE − val_symbols_with_trades) × 5.0
```

`PHASE3_MIN_SYMBOL_COVERAGE = 7` (config). The strategy must generate at least one trade on at least 7 of the 10 symbols in the validation split. This prevents strategies that only work on 1–2 symbols.

**Effect of `PHASE3_MIN_SYMBOL_COVERAGE`:** Increasing this (e.g., to 9) requires broader coverage but may be hard to satisfy with a 2-rule strategy. Decreasing it (e.g., to 5) allows more concentrated strategies.

**Duplicate rule penalty (50.0):** Applied if any two rules in the set have identical condition sets (order-independent). This prevents the search from selecting the same rule twice.

**Symbol consistency penalty — `symbol_consistency_penalty`:**
```python
overlap = |train_symbols ∩ val_symbols| / |train_symbols ∪ val_symbols|
penalty = (1.0 − overlap) × PHASE3_SYMBOL_CONSISTENCY_WEIGHT
```

`PHASE3_SYMBOL_CONSISTENCY_WEIGHT = 10.0` (config). If the strategy trades on different symbols in training vs. validation, it receives a penalty. This discourages strategies that happen to work on different symbols in different time periods.

**Train-val correlation penalty — `train_val_corr_penalty`:**
```python
corr = Pearson correlation of per-symbol net PnL (train vs. val)
penalty = (1.0 − corr) × 0.5 × PHASE3_TRAIN_VAL_CORR_WEIGHT
```

`PHASE3_TRAIN_VAL_CORR_WEIGHT = 5.0` (config). If the per-symbol PnL pattern in training is uncorrelated with validation (e.g., profitable on different symbols), the strategy receives a penalty. This is a soft overfitting detector.

**Effect of `PHASE3_TRAIN_VAL_CORR_WEIGHT`:** Increasing this more aggressively penalizes strategies where the per-symbol PnL pattern changes between train and val.

### Incremental orthogonality penalties (new in Phase 3)

To reduce over-trading and suppress “clone rules” that enter on the same
validation candles, Phase 3 adds two soft penalties via `compute_phase3_objectives`
(included in `total_penalty` and applied to all three objectives):

1. **Incremental trade gate — `PHASE3_MIN_INCREMENTAL_TRADES` / `PHASE3_INCREMENTAL_GATE_PENALTY`**
   - For an ordered rule set `[R1, R2, ...]`, define the incremental coverage of
     rule `Ri` as the number of validation rows where `Ri` signals **and**
     no earlier rule already signaled:
     `incremental = count(M(Ri) AND NOT(M(R1) OR ... OR M(R{i-1})))`.
   - If `i > 1` and `incremental < PHASE3_MIN_INCREMENTAL_TRADES`,
     a penalty is added proportional to the shortfall.

2. **Trade Jaccard similarity penalty — `PHASE3_JACCARD_SIMILARITY_GATE` / `PHASE3_JACCARD_PENALTY_WEIGHT`**
   - For every pair of rules `(Ri, Rj)` compute
     `jaccard = |M(Ri) ∩ M(Rj)| / |M(Ri) ∪ M(Rj)|` over validation entry-mask signals.
   - If `jaccard` exceeds `PHASE3_JACCARD_SIMILARITY_GATE`, a penalty is added.

These penalties are computed from cached validation masks in `Phase3EvalCache`.

### Validation gate penalties — `PHASE3_USE_TRAIN_TARGET = True`

When using training as the primary objective, three additional gate penalties are applied based on validation performance:

**Sortino ratio gate:**
```python
if val_sortino < PHASE3_VAL_SORTINO_RATIO_GATE × train_sortino:
    gate_penalty += PHASE3_VAL_GATE_PENALTY
```

`PHASE3_VAL_SORTINO_RATIO_GATE = 0.7`, `PHASE3_VAL_GATE_PENALTY = 100.0` (config). If validation Sortino is below 70% of training Sortino, a large penalty is applied.

**Drawdown gate:**
```python
if val_dd > PHASE3_VAL_DRAWDOWN_RATIO_GATE × train_dd:
    gate_penalty += PHASE3_VAL_GATE_PENALTY
```

`PHASE3_VAL_DRAWDOWN_RATIO_GATE = 1.15` (config). If validation drawdown exceeds 115% of training drawdown, penalized.

**Per-rule minimum validation trades gate:**
```python
if min_per_rule_val_trades < PHASE3_PER_RULE_MIN_VAL_TRADES_PER_SYMBOL:
    gate_penalty += PHASE3_VAL_GATE_PENALTY
```

`PHASE3_PER_RULE_MIN_VAL_TRADES_PER_SYMBOL = 15` (config). Each rule must have sufficient per-symbol trades on validation.

**Train/val gap penalties:** `PHASE3_TRAIN_VAL_GAP_MAX_PCT` and `PHASE3_VAL_TRAIN_GAP_MAX_PCT` (default 10%) penalize strategies where one split is much better than the other — catches short val-lucky / train-lucky failure modes.

**Effect of `PHASE3_VAL_GATE_PENALTY`:** At 100.0, a single gate violation dominates the objective vector. Increase for stricter gates; decrease only if the search finds no feasible teams.

---

## 5. Purged CV evaluation (`SPLIT_MODE == "purged_rolling_cv"`)

`Rule_Set_Selector` receives `cv_folds` from the pipeline and builds one `Phase3EvalCache` + engine pair per fold.

For each candidate rule set, `_evaluate_rule_set(..., cv_fold_contexts=...)`:

1. Simulates the team on every fold's train and val engines.
2. Computes `compute_phase3_objectives` per fold.
3. Keeps the fold with the **worst** total objective (highest `f1 + f2 + f3` among folds, since objectives are minimized).
4. Returns **conservative merged** train/val metrics (min return/PF/WR, max DD) for reporting and maximin selection.

Greedy and NSGA-II paths disable JAX/parallel batch when CV is active (per-fold evaluation loop).

Mask-based penalties (Jaccard, incremental trades) use the **last fold's** cache — the same block persisted as `validation_25`.

---

## 6. Phase3EvalCache — Signal Mask Precomputation

`build_phase3_eval_cache` precomputes boolean signal masks for every rule in the pool on both the training and validation DataFrames. This avoids recomputing `_apply_dynamic_rule` for every rule in every candidate evaluation.

For a pool of 100 rules and 600,000 training rows, this precomputation takes a few seconds but saves significant time during the 8,000+ evaluations in the refinement phase.

The cache also stores `per_rule_min_val_trades`: the minimum per-symbol trade count for each rule on the validation split. This is used by the per-rule validation trades gate without requiring additional simulations.

---

## 7. Parallel Batch Evaluation

When `PHASE3_USE_PARALLEL_BATCH = True` (default) and **not** using purged CV, multiple rule set candidates are evaluated in parallel using `CPUBacktestEngine.simulate_rule_set_batch`:

- If the cache is available: uses `ThreadPoolExecutor` (thread-safe, no pickling overhead).
- Without cache: uses `ProcessPoolExecutor` (separate processes to bypass the GIL).

`PHASE3_BATCH_WORKERS = min(32, os.cpu_count())` (config). On a 16-core machine, this evaluates up to 16 rule sets simultaneously.

**Effect of `PHASE3_BATCH_WORKERS`:** Increasing this uses more CPU cores, reducing wall-clock time for the refinement phase. The optimal value is typically `os.cpu_count()`.

---

## 8. Best Selection from Pareto Front — `_select_best_from_pareto`

After refinement, the Pareto front contains multiple non-dominated rule sets. Selection uses `_maximin_selection_score` (min of train/val returns with profitability floors), with tie-breakers on train/val gap, symbol consistency, and Jaccard overlap. Under purged CV, each candidate is re-evaluated across all folds before scoring.

---

## 9. Configuration Reference

See **`gpu_fuzzy_trader/config.py`** (Phase 3 section) for current defaults and tuning notes. Key knobs:

| Parameter | Role |
|---|---|
| `PHASE3_USE_TRAIN_TARGET` | Train metrics for objectives; val for gates |
| `PHASE3_USE_MAXIMIN_SCORE` | Greedy minimax train/val returns |
| `PHASE3_VAL_*_GATE_*` | Anti-overfit penalties |
| `PHASE3_*_GAP_*` | Penalise val >> train or train >> val |
| `PHASE3_MIN_INCREMENTAL_TRADES` / `PHASE3_JACCARD_*` | Rule-team orthogonality |
| `SPLIT_MODE` | Enables multi-fold evaluation when `purged_rolling_cv` |

---

## 10. Outputs

- `outputs/long.json` / `outputs/short.json` — Strategy files in `evaluator_v3.ipynb` format. At this stage, TP/SL/capital_pct are still the Phase 2 static values. Phase 4 will update them.
- `outputs/reports/train_long_equity.png` / `outputs/reports/validation_long_equity.png` — Equity curves on training and validation splits.
- `outputs/reports/train_per_symbol_performance.csv` / `outputs/reports/validation_per_symbol_performance.csv` — Per-symbol metrics.
