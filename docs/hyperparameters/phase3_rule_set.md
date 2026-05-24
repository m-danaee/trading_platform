# Phase 3 — Rule Set Selection

**Modules:** [`gpu_fuzzy_trader/phases/phase3_rule_set.py`](../../gpu_fuzzy_trader/phases/phase3_rule_set.py), [`gpu_fuzzy_trader/phases/phase3_greedy.py`](../../gpu_fuzzy_trader/phases/phase3_greedy.py)  
**Config prefix:** `PHASE3_*`  
**Data split:** Train objectives + validation gates (anti-leakage design)

[← Phase 2](phase2_rule_pool.md) | [Index](README.md) | [Phase 4 →](phase4_rl_risk.md)

---

## Purpose and statistical framing

Phase 3 selects an **ordered team of 2–3 rules** from the Phase 2 pool. Individual rules may be strong alone but interact when run together (exposure limits, priority ordering, overlapping entries).

**Two-stage search:**

1. **Greedy construction** — iteratively add the rule that maximizes a scalar score from weighted objectives.
2. **NSGA-II refinement** — combinatorial multi-objective search seeded from greedy result.

### Train-as-target / validation-as-gate (`PHASE3_USE_TRAIN_TARGET=True`)

When enabled (default), optimization objectives are computed on **train**:

```
f1 = -train_sortino + total_penalty
f2 = train_max_drawdown + total_penalty
f3 = -train_win_rate + total_penalty
```

Validation is **not** used as the primary fitness signal. Instead, **hard gates** add `PHASE3_VAL_GATE_PENALTY` when validation degrades disproportionately. This reduces validation-set overfitting compared to optimizing directly on val Sortino.

When `PHASE3_USE_TRAIN_TARGET=False` (legacy ablation path), objectives use validation metrics directly.

---

## Penalty structure

All penalties sum into `total_penalty` and are added to f1, f2, f3.

| Penalty                   | Trigger                                                | Default weight                                     |
| ------------------------- | ------------------------------------------------------ | -------------------------------------------------- |
| Zero-trade                | `train_trades == 0` or `val_trades == 0`               | 100 (fixed)                                        |
| Duplicate rules           | Identical condition sets in team                       | 50 (fixed)                                         |
| Coverage                  | Val symbols with trades < `PHASE3_MIN_SYMBOL_COVERAGE` | `(7 - count) × 5`                                  |
| Symbol consistency        | Low Jaccard(train symbols, val symbols)                | `(1 - overlap) × PHASE3_SYMBOL_CONSISTENCY_WEIGHT` |
| Train/val PnL correlation | Low corr(per-symbol PnL train, val)                    | `(1 - corr) × 0.5 × PHASE3_TRAIN_VAL_CORR_WEIGHT`  |
| Validation gates          | See below                                              | `+PHASE3_VAL_GATE_PENALTY` each                    |

### Validation gates (when `PHASE3_USE_TRAIN_TARGET=True`)

Each failed gate adds `PHASE3_VAL_GATE_PENALTY` (default 75):

| Gate                   | Condition                                                                                   |
| ---------------------- | ------------------------------------------------------------------------------------------- |
| Sortino ratio          | `val_sortino < PHASE3_VAL_SORTINO_RATIO_GATE × train_sortino` (when train_sortino > 0)      |
| Drawdown               | `val_drawdown > PHASE3_VAL_DRAWDOWN_RATIO_GATE × max(train_drawdown, 1.0)`                  |
| Per-rule symbol trades | Any rule has `< PHASE3_PER_RULE_MIN_VAL_TRADES_PER_SYMBOL` trades on some validation symbol |

---

## Hyperparameters

| Parameter                                   | Default           | ↑ increase                                     | ↓ decrease                                 |
| ------------------------------------------- | ----------------- | ---------------------------------------------- | ------------------------------------------ |
| `PHASE3_MIN_RULES`                          | `2`               | Forces larger teams                            | Allows single-rule sets (if pool small)    |
| `PHASE3_MAX_RULES`                          | `3`               | More diversification; dilution risk            | Simpler teams; less combinatorial search   |
| `PHASE3_MIN_SYMBOL_COVERAGE`                | `7` (of 10)       | Rejects symbol-concentrated sets               | Allows niche symbol strategies             |
| `PHASE3_USE_PARALLEL_BATCH`                 | `True`            | ProcessPool/thread batch team eval on CPU      | Sequential eval only                       |
| `PHASE3_BATCH_WORKERS`                      | `min(32, CPUs)`   | Parallelism for batch eval                     | Fewer workers                              |
| `PHASE3_NUMBA_ENABLED`                      | `True`            | Numba NSGA-II sort/crowding in refinement      | Pure Python NSGA helpers                   |
| `PHASE3_USE_GPU`                            | `False`           | JAX path + cached masks (parity-tested)        | CPU cache + parallel batch only            |
| `PHASE3_REFINE_POP_SIZE`                    | `100`             | More refinement exploration                    | Faster Phase 3                             |
| `PHASE3_REFINE_GENERATIONS`                 | `80`              | Better Pareto refinement                       | Faster; may stop at greedy solution        |
| `PHASE3_GREEDY_WEIGHTS`                     | `(1.0, 0.7, 0.5)` | See below                                      | See below                                  |
| `PHASE3_SYMBOL_CONSISTENCY_WEIGHT`          | `10.0`            | Stronger penalty for train/val symbol mismatch | Allows different symbol sets across splits |
| `PHASE3_USE_TRAIN_TARGET`                   | `True`            | Train-opt + val gates                          | Legacy val-direct optimization             |
| `PHASE3_VAL_SORTINO_RATIO_GATE`             | `0.5`             | Val must be ≥50% of train Sortino              | Stricter → fewer candidates pass           |
| `PHASE3_VAL_DRAWDOWN_RATIO_GATE`            | `1.5`             | Val DD must be ≤1.5× train DD                  | Stricter → reject volatile val profiles    |
| `PHASE3_PER_RULE_MIN_VAL_TRADES_PER_SYMBOL` | `5`               | Each rule must fire more on every symbol       | Allows sparse per-rule symbol activity     |
| `PHASE3_TRAIN_VAL_CORR_WEIGHT`              | `5.0`             | Penalizes decorrelated per-symbol PnL          | Less emphasis on cross-split PnL alignment |
| `PHASE3_VAL_GATE_PENALTY`                   | `75.0`            | Harder rejection of gate failures              | Gate failures less dominant in objectives  |

---

## Parameter details

### `PHASE3_GREEDY_WEIGHTS` — `(sortino, drawdown, win_rate)`

Greedy scalarization for ranking candidate teams (not the NSGA objectives directly):

```
score = w0 × sortino - w1 × drawdown + w2 × win_rate - gate_penalties
```

(signs handled in `_scalar_score`; drawdown subtracted)

| Weight   | Default | ↑ effect                        |
| -------- | ------- | ------------------------------- |
| Sortino  | 1.0     | Prioritize risk-adjusted return |
| Drawdown | 0.7     | Favor lower drawdown teams      |
| Win rate | 0.5     | Favor higher hit rate           |

Imbalanced weights can seed refinement toward a narrow region of the Pareto front.

### Team size (`MIN_RULES`, `MAX_RULES`)

Default 2–3 rules balances:

- **Diversification** — multiple entry conditions, priority-based assignment.
- **Signal dilution** — more rules → more exposure competition and overlapping filters.

Output schema supports up to 5 rules; search is capped at `MAX_RULES`.

### Symbol coverage and consistency

- **Coverage** ensures at least 7/10 symbols see trades on validation — catches rules that only work on BTC/ETH-style concentration.
- **Consistency** uses Jaccard overlap of symbol sets with trades between train and val.
- **PnL correlation** compares per-symbol net PnL vectors — catches "works on different symbols" failure mode.

### Validation gate tuning

| Symptom                             | Knob                                                                            |
| ----------------------------------- | ------------------------------------------------------------------------------- |
| Good train, poor val teams selected | ↓ `PHASE3_VAL_SORTINO_RATIO_GATE` (e.g. 0.6→0.4) or ↑ `PHASE3_VAL_GATE_PENALTY` |
| No teams pass gates                 | ↑ ratio gates slightly or ↓ `PHASE3_VAL_GATE_PENALTY`                           |
| Rules inactive on alt symbols       | ↑ `PHASE3_PER_RULE_MIN_VAL_TRADES_PER_SYMBOL`                                   |

### Refinement budget

`PHASE3_REFINE_POP_SIZE × PHASE3_REFINE_GENERATIONS` ≈ 8,000 team evaluations after greedy. Each evaluation runs train + validation backtests; per-rule gate stats are **precomputed once** at selector init (no extra val sims per team).

**Performance knobs (see `phase3_cache.py`, `phase3_objectives.py`):**

| Bottleneck (old)                         | Mitigation                                      |
| ---------------------------------------- | ----------------------------------------------- |
| Repeated condition parsing               | `Phase3EvalCache` signal masks (train/val)      |
| Up to 3 extra val sims / team (gate)     | `per_rule_min_val_trades` cache                 |
| Sequential NSGA evals                    | `PHASE3_USE_PARALLEL_BATCH` + `simulate_rule_set_batch` |
| NSGA sort O(n²)                          | `PHASE3_NUMBA_ENABLED` (shared with Phase 2)  |

After profiling on your hardware, you can raise pop×gen (e.g. toward `500×200` ≈ 100k evals). Defaults stay `100×80` until a full pipeline benchmark confirms wall-clock budget.

---

## Static risk at Phase 3 output

Rule sets written to `outputs/long.json` and `outputs/short.json` still carry **Phase 2 static** TP/SL/capital_pct. Phase 4 replaces these with optimized values.

---

## Interactions and tuning order

1. Keep `PHASE3_USE_TRAIN_TARGET=True` unless ablating.
2. Set `MIN_SYMBOL_COVERAGE` and `MIN_TRADE_SUPPORT` (Phase 2) coherently — coverage is about teams, support about individual rules.
3. Adjust gate ratios before refinement budget.
4. Keep `PHASE3_USE_PARALLEL_BATCH=True` on multi-core hosts.
5. Enable `PHASE3_USE_GPU` only after parity tests pass (cached-mask path; speed only).

---

## Diagnostics

| Artifact                                       | What to check                        |
| ---------------------------------------------- | ------------------------------------ |
| `outputs/long.json`, `outputs/short.json`      | Rule count, conditions, static TP/SL |
| `outputs/reports/train_*_equity.png`           | Train curve shape                    |
| `outputs/reports/validation_*_equity.png`      | Val vs train degradation             |
| `outputs/reports/*_per_symbol_performance.csv` | Symbol coverage                      |

---

## Code references

Train-target evaluation:

```249:370:gpu_fuzzy_trader/phases/phase3_rule_set.py
def _evaluate_rule_set(...):
    ...
    if _cfg.PHASE3_USE_TRAIN_TARGET:
        f1 = -train_sortino + total_penalty
        f2 = train_dd + total_penalty
        f3 = -train_wr + total_penalty
```

Symbol consistency penalty:

```152:159:gpu_fuzzy_trader/phases/phase3_rule_set.py
def _symbol_consistency_penalty(train_metrics: dict, val_metrics: dict) -> float:
    ...
    overlap = len(train_syms & val_syms) / len(train_syms | val_syms)
    return (1.0 - overlap) * _cfg.PHASE3_SYMBOL_CONSISTENCY_WEIGHT
```

Greedy weights:

```219:219:gpu_fuzzy_trader/phases/phase3_greedy.py
    weights = weights if weights is not None else _cfg.PHASE3_GREEDY_WEIGHTS
```
