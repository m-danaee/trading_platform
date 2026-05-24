# Phase 2 — Rule Pool Generation

**Module:** [`gpu_fuzzy_trader/phases/phase2_rule_pool.py`](../../gpu_fuzzy_trader/phases/phase2_rule_pool.py), [`gpu_fuzzy_trader/evolution/evox_runner.py`](../../gpu_fuzzy_trader/evolution/evox_runner.py)  
**Config prefix:** `PHASE2_*`, `MIN_*`, `SUPPORT_*`, `SORTINO_*`  
**Data split:** Sampled train (+ sampled validation when joint objective enabled)

[← Phase 1](phase1_feature_selection.md) | [Index](README.md) | [Phase 3 →](phase3_rule_set.md)

---

## Purpose and statistical framing

Phase 2 discovers a **Pareto-optimal pool of single fuzzy rules** via multi-objective evolutionary search (NSGA-III when EvoX is installed; NumPy NSGA-II fallback).

Each **chromosome** is a vector of integer genes — one per Phase 1 feature — where each gene selects a fuzzy term or a **don't-care** sentinel (inactive condition). Decoded rules are backtested with **fixed** TP/SL/capital so fitness reflects **signal quality**, not risk tuning (Phase 4 handles risk).

**Leakage guard:** When `PHASE2_JOINT_TRAIN_VAL=True`, f1 uses `min(train_sortino, val_sortino)` after saturation — rules must work on both splits.

---

## Chromosome and objectives

**Encoding:** `gene_i ∈ {0, …, num_classes_i - 1, dont_care_i}` where `dont_care_i = num_classes_i`.

**Three minimized objectives** (penalties added to all three):

| Objective | Formula                          | Interpretation                                   |
| --------- | -------------------------------- | ------------------------------------------------ |
| f1        | `-saturated_sortino + penalties` | Risk-adjusted return (joint: worst of train/val) |
| f2        | `max_drawdown_pct + penalties`   | Tail risk                                        |
| f3        | `-win_rate + penalties`          | Hit rate                                         |

**Saturating Sortino** (prevents flat fitness plateaus):

```
saturated_sortino = tanh(raw_sortino / SORTINO_SCALE) × SORTINO_CAP
```

Default: `SORTINO_SCALE=3.0`, `SORTINO_CAP=5.0`.

**Penalties:**

| Penalty         | Trigger                                                                                     |
| --------------- | ------------------------------------------------------------------------------------------- |
| Condition count | Active conditions ∉ [`MIN_CONDITIONS`, `MAX_CONDITIONS`]                                    |
| Support         | `executed_trades < MIN_TRADE_SUPPORT` (graduated); hard reject below `MIN_TRADE_POOL_FLOOR` |
| Diversity       | Hamming distance to nearest Pareto member ≤ `PHASE2_DIVERSITY_HAMMING_THRESHOLD`            |

---

## Static risk (Phase 2 only)

| Parameter            | Default | Role                             |
| -------------------- | ------- | -------------------------------- |
| `PHASE2_TP`          | `2.0`   | Take-profit % during rule mining |
| `PHASE2_SL`          | `1.0`   | Stop-loss % during rule mining   |
| `PHASE2_CAPITAL_PCT` | `48.0`  | Capital per rule during mining   |

**Why fixed:** If TP/SL co-evolved with conditions, the GA could overfit risk parameters to each rule. Phase 4 optimizes these per rule on the selected team.

**Performance effects:**

| Change        | Effect                                                                           |
| ------------- | -------------------------------------------------------------------------------- |
| ↑ TP          | Fewer TP hits, larger wins when hit; changes Phase 1 asymmetric target if re-run |
| ↓ SL          | Tighter stops; fewer trades survive; higher win rate possible but lower avg win  |
| ↑ CAPITAL_PCT | Larger PnL swings; drawdown objective more sensitive                             |

Keep constant across Phase 2 runs when comparing pools.

---

## Rule constraint hyperparameters

| Parameter              | Default | ↑ increase                                      | ↓ decrease                                                   |
| ---------------------- | ------- | ----------------------------------------------- | ------------------------------------------------------------ |
| `MIN_CONDITIONS`       | `3`     | Simpler rules penalized; forces richer logic    | Allows very sparse rules; overfit risk                       |
| `MAX_CONDITIONS`       | `4`     | Allows more complex rules; harder search        | Forces simpler, more interpretable rules                     |
| `MIN_TRADE_SUPPORT`    | `300`   | Fewer rules pass; higher statistical confidence | Larger pool; noisier Sortino (~3 trades/symbol/month at 300) |
| `MIN_TRADE_POOL_FLOOR` | `75`    | Hard reject fewer rules                         | More junk in archive                                         |
| `SUPPORT_PENALTY_MAX`  | `50.0`  | Stronger push toward high-trade rules           | Sparse "lucky" rules survive longer                          |

Support penalty shape (between floor and support):

```
shortfall = (MIN_TRADE_SUPPORT - executed) / MIN_TRADE_SUPPORT
penalty = min(shortfall² × SUPPORT_PENALTY_MAX, SUPPORT_PENALTY_MAX)
```

Below `MIN_TRADE_POOL_FLOOR`: penalty = `2 × SUPPORT_PENALTY_MAX` (effective hard reject).

---

## Sortino saturation

| Parameter       | Default | Effect                                                     |
| --------------- | ------- | ---------------------------------------------------------- |
| `SORTINO_SCALE` | `3.0`   | Larger → flatter tanh; less gradient for improving Sortino |
| `SORTINO_CAP`   | `5.0`   | Upper bound on saturated Sortino contribution to f1        |

**Historical note:** A flat cap previously pinned `best_sortino` at a sentinel from generation 0. Saturation lets the Pareto front continue moving while keeping f1 on comparable scale to drawdown penalties.

---

## Joint train+validation objective

| Parameter                | Default | Effect                                        |
| ------------------------ | ------- | --------------------------------------------- |
| `PHASE2_JOINT_TRAIN_VAL` | `True`  | f1 uses `min(saturated_train, saturated_val)` |

When validation trade count is very low (`< max(MIN_TRADE_POOL_FLOOR // 4, 10)`), support penalty is maxed and Sortino objective collapses.

- **Generalization:** Strongly recommended ON. Reduces rules that spike on train only.
- **Compute:** ~2× backtest evaluations per chromosome per generation.

---

## Diversity penalty

| Parameter                            | Default | Effect                                                           |
| ------------------------------------ | ------- | ---------------------------------------------------------------- |
| `PHASE2_DIVERSITY_HAMMING_THRESHOLD` | `2`     | Apply penalty if within this Hamming distance of a Pareto member |
| `PHASE2_DIVERSITY_PENALTY`           | `5.0`   | Magnitude added to all objectives                                |

Encourages spread on the Pareto front. Too high → front members look artificially worse; too low → duplicate chromosomes cluster.

---

## Evolution budget

| Parameter                | Default   | Compute              | Performance                                      |
| ------------------------ | --------- | -------------------- | ------------------------------------------------ |
| `PHASE2_POPULATION_SIZE` | `200`     | Linear in evals/gen  | More exploration; slower                         |
| `PHASE2_GENERATIONS`     | `200`     | Linear in wall-clock | History showed front stalling ~gen 50 at pop=100 |
| `PHASE2_ALGORITHM`       | `"NSGA3"` | Identifier only      | EvoX NSGA-III vs NSGA-II fallback                |

**Rough cost:** ~`POPULATION × GENERATIONS × 2` backtests per direction when joint val enabled (e.g. 200×200×2 = 80k evals/direction).

Emergency RAM knobs (commented in config): halve population and generations.

---

## Archive and warm-start

| Parameter                      | Default | Effect                                                      |
| ------------------------------ | ------- | ----------------------------------------------------------- |
| `PHASE2_ARCHIVE_MAX_SIZE`      | `500`   | Max persisted rules per direction in `phase2_rule_archive/` |
| `PHASE2_ARCHIVE_SEED_FRACTION` | `0.35`  | Fraction of initial population from archive                 |

Archive persists **across output directories**. Feature signature must match current Phase 1 features or archive is rejected.

- **Performance:** Faster convergence when re-running; may bias toward past local optima if data regime changed.
- **Action:** Delete archive when feature set or data distribution shifts materially.

---

## Sampling (from Phase 1 config)

Phase 2 uses `PHASE1_SAMPLING_TOTAL` — see [Phase 1 doc](phase1_feature_selection.md#phase1_sampling_total-phase-2-consumption).

---

## Interactions and tuning order

1. Fix `PHASE2_TP/SL/CAPITAL_PCT` before long evolutionary runs.
2. Set `MIN_TRADE_SUPPORT` based on desired min trades per symbol.
3. Scale `POPULATION × GENERATIONS` for compute budget.
4. Enable `PHASE2_JOINT_TRAIN_VAL` before tightening support.
5. Tune `SORTINO_SCALE` only if f1 gradient looks flat in history JSON.

---

## Diagnostics

| Artifact                               | What to check                                                |
| -------------------------------------- | ------------------------------------------------------------ |
| `outputs/phase2_*_history.json`        | `best_sortino_ratio`, Pareto front progression by generation |
| `outputs/phase2_*_pool.json`           | Rule count, trade counts, condition diversity                |
| `outputs/reports/phase2_*_metrics.png` | Visual Pareto movement                                       |
| `phase2_rule_archive/*.json`           | Archive size; stale entries after feature change             |

---

## Code references

Fitness evaluation core:

```168:263:gpu_fuzzy_trader/phases/phase2_rule_pool.py
def _evaluate_chromosome(...):
    ...
    sortino_for_obj = sortino_train
    if _cfg.PHASE2_JOINT_TRAIN_VAL and val_engine is not None:
        ...
        sortino_for_obj = min(sortino_train, sortino_val)
    ...
    f1 = -sortino_for_obj + support_penalty + diversity_penalty + cond_penalty
    f2 = max_dd + support_penalty + diversity_penalty + cond_penalty
    f3 = -win_rate + support_penalty + diversity_penalty + cond_penalty
```

Support penalty:

```47:59:gpu_fuzzy_trader/phases/phase2_rule_pool.py
def trade_support_penalty(executed: int) -> float:
    ...
```

---

## Deferred: Phase 2 MOME

Config comments describe a future **4×10 descriptor grid** (active conditions × symbol coverage) with per-cell Pareto archives. **Not implemented** in the current release — no tunable parameters.
