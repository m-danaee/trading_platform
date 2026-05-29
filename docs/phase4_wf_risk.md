# Phase 4 — Walk-Forward Risk Optimization

**Module:** `gpu_fuzzy_trader/phases/phase4_wf_optimizer.py` → `WalkForwardRiskOptimizer`

Phase 4 fine-tunes the risk parameters (TP, SL, and capital allocation) for each rule in the strategy selected by Phase 3. The rule conditions are frozen — Phase 4 only adjusts the numbers, not the logic.

Defaults (see `config.py`): `PHASE4_N_TRIALS = 200`, `PHASE4_WF_SPLITS = 2`, `PHASE4_SAMPLER = "tpe"`, quantized TP/SL steps of `0.5`.

**Important:** Phase 4 walk-forward runs on **`validation_25.parquet` only** — not on all purged CV folds from Phases 2–3. When `SPLIT_MODE == "purged_rolling_cv"`, that file is the **last CV fold's** validation block (most recent in-sample period before `test.csv`).

---

## 1. What Phase 4 Optimizes

For each rule in the strategy, Phase 4 searches for the best values of:

- `tp`: Take-profit percentage (how far price must move in your favor to close the trade as a win)
- `sl`: Stop-loss percentage (how far price must move against you before the trade is closed as a loss)
- `capital_pct`: Percentage of current equity to allocate to this rule's trades

These three parameters per rule define the risk profile of the strategy. Phase 2 used fixed values (`PHASE2_TP`, `PHASE2_SL`, `PHASE2_CAPITAL_PCT`) to isolate rule quality from risk tuning. Phase 4 optimizes them jointly on the persisted validation split.

---

## 2. Walk-Forward Validation — `split_validation_walk_forward`

The **`validation_25`** DataFrame (not full `train.csv`) is divided into K chronological windows (`PHASE4_WF_SPLITS = 2` by default), plus an optional tail holdout window.

### How the split works

For each symbol independently:

1. Rows are sorted by datetime.
2. `np.array_split(np.arange(N), K)` divides the rows into K equal-sized chunks.
3. Window i = concatenation of all symbols' chunk i.

For example, with K=2 and a symbol with 1000 validation rows:

- Window 1: rows 0–499
- Window 2: rows 500–999

With `PHASE4_INCLUDE_TAIL_HOLDOUT = True`, an extra window uses the last `PHASE4_TAIL_HOLDOUT_FRACTION` (25%) of each symbol's validation rows.

### Relationship to Phase 2/3 purged CV

| Layer | What is split | Purpose |
|---|---|---|
| Phases 2–3 (`purged_rolling_cv`) | Full `train.csv` into K folds | Rule discovery must work in every season |
| Phase 4 (`PHASE4_WF_SPLITS`) | `validation_25` only | Risk tuning on recent in-sample OOS |

**Recommendation:** If short still fails on `test.csv` after purged CV in P2/P3, increase `PHASE4_WF_SPLITS` (e.g. 4–6) so risk params are stressed on more validation sub-windows.

### Why walk-forward?

A single validation evaluation can be lucky or unlucky. Phase 4 uses the **worst-case** return / drawdown across windows (plus feasibility floors `PHASE4_MIN_WORST_FOLD_*`).

**Effect of `PHASE4_WF_SPLITS`:** More windows → stricter estimate, but fewer rows per window. If any symbol has fewer than K validation rows, Phase 4 raises `ValueError`.

---

## 3. Optuna Multi-Objective Search

Phase 4 uses [Optuna](https://optuna.org/) for hyperparameter optimization.

### Objectives (two, both optimized simultaneously)

```
maximize: worst_sortino = min(sortino across K windows) − overalloc_penalty
minimize: worst_drawdown = max(drawdown across K windows) + overalloc_penalty
```

The worst-case Sortino and worst-case drawdown across all K windows are the two objectives. This is a bi-objective problem, and Optuna's multi-objective sampler finds the Pareto front.

### Overallocation penalty — `_overalloc_penalty`

```python
total_cap = sum(capital_pct for each rule)
penalty = max(0, total_cap − 100.0) / 100.0 × PHASE4_TOTAL_CAP_PENALTY
```

`PHASE4_TOTAL_CAP_PENALTY = 2.0` (config). If the total capital allocation across all rules exceeds 100%, a penalty is applied to both objectives. This discourages over-leveraged strategies.

**Effect of `PHASE4_TOTAL_CAP_PENALTY`:** Increasing this more aggressively penalizes over-allocation. At 2.0, allocating 150% total capital adds a penalty of `(150-100)/100 × 2.0 = 1.0` to the worst Sortino (subtracted) and worst drawdown (added).

### Search space (quantized)

Each rule's parameters are sampled from quantized ranges:

| Parameter     | Range          | Step | Config keys                       |
| ------------- | -------------- | ---- | --------------------------------- |
| `tp`          | [4.0%, 6.0%]   | 0.5% | `PHASE4_TP_MIN/MAX/STEP`          |
| `sl`          | [2.0%, 3.0%]   | 0.5% | `PHASE4_SL_MIN/MAX/STEP`          |
| `capital_pct` | [10.0%, 50.0%] | 5.0% | `PHASE4_CAPITAL_PCT_MIN/MAX/STEP` |

The quantization (step sizes) reduces the search space and prevents the optimizer from finding spurious precision. For example, TP has `(6.0 − 4.0) / 0.5 + 1 = 5` possible values.

**Effect of search space bounds:**

- `PHASE4_TP_MIN/MAX`: Narrowing the TP range focuses the search. If you know from Phase 2 that rules work best with TP around 3%, you could narrow to [2.5%, 3.5%].
- `PHASE4_SL_MIN/MAX`: A tighter SL range (e.g., [1.0%, 1.5%]) forces more conservative stops.
- `PHASE4_CAPITAL_PCT_MIN/MAX`: Lowering the maximum (e.g., to 30%) limits position sizing, reducing both upside and downside.

### Sampler — `PHASE4_SAMPLER`

| Value               | Algorithm                        | Behavior                                                                                                                                |
| ------------------- | -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `"nsga2"` (default) | NSGA-II                          | Multi-objective evolutionary search. Maintains a Pareto front of trials. Good for exploring the trade-off between Sortino and drawdown. |
| `"tpe"`             | Tree-structured Parzen Estimator | Bayesian optimization. Faster convergence but less diversity on the Pareto front.                                                       |

**Effect of `PHASE4_N_TRIALS`:** The total number of Optuna trials. At 1000 trials with 2 rules, each trial evaluates 2 walk-forward windows = 2000 backtest simulations. Increasing this improves the quality of the Pareto front but increases compute time linearly.

---

## 4. Trial Selection — `_select_pareto_trial`

After optimization, the best trial is selected from the Pareto front:

1. Filter Pareto front trials by `worst_drawdown ≤ PHASE4_MAX_WORST_DRAWDOWN_PCT` (default: 15.0%).
2. Among filtered trials, pick the one with the highest `worst_sortino`.
3. If no trials pass the drawdown filter, fall back to: pick the trial with the minimum drawdown, then highest Sortino among ties.

**Effect of `PHASE4_MAX_WORST_DRAWDOWN_PCT`:** This is the primary risk constraint. Increasing it (e.g., to 25%) allows more aggressive strategies. Decreasing it (e.g., to 10%) enforces stricter drawdown control but may result in very conservative capital allocation.

---

## 5. Capital Normalization — `_normalize_capital_pct`

When `PHASE4_HARD_CAP_NORMALIZE = True` (default), after selecting the best trial, the capital allocations are scaled so their sum does not exceed `MAX_TOTAL_EXPOSURE_PCT` (100%):

```python
if total_cap > MAX_TOTAL_EXPOSURE_PCT:
    scale = MAX_TOTAL_EXPOSURE_PCT / total_cap
    for rule in rules_set:
        rule["capital_pct"] *= scale
```

This is a post-processing step that ensures the strategy never over-allocates, regardless of what the optimizer found.

**Effect of `PHASE4_HARD_CAP_NORMALIZE`:** Setting to `False` allows the optimizer's raw capital allocations to be used, which may exceed 100% total. This is only appropriate if you intend to use leverage.

---

## 6. Parallel Optimization — `PHASE4_N_JOBS`

`PHASE4_N_JOBS = 1` (default). Setting this to a higher value runs multiple Optuna trials in parallel. Each trial creates its own `CPUBacktestEngine` instances (one per walk-forward window), so there are no shared state issues.

**Warning:** With `PHASE4_N_JOBS > 1` and `PHASE4_SAMPLER = "nsga2"`, Optuna's NSGA-II sampler may not benefit from parallelism as much as TPE, because NSGA-II needs to see the results of previous trials to guide the next generation. TPE is generally more parallelism-friendly.

---

## 7. Skip Logic

Phase 4 is skipped if `outputs/{direction}.json` already exists, has `risk_optimized: true`, and all TP/SL/capital_pct values are within the configured bounds. This allows re-running Phase 5 without re-running Phase 4.

---

## 8. Configuration Reference

| Parameter                       | Default   | Technical effect                                                                             |
| ------------------------------- | --------- | -------------------------------------------------------------------------------------------- |
| `PHASE4_TP_MIN`                 | `2.0`     | Minimum TP % in search space.                                                                |
| `PHASE4_TP_MAX`                 | `4.0`     | Maximum TP % in search space.                                                                |
| `PHASE4_SL_MIN`                 | `1.0`     | Minimum SL % in search space.                                                                |
| `PHASE4_SL_MAX`                 | `2.0`     | Maximum SL % in search space.                                                                |
| `PHASE4_CAPITAL_PCT_MIN`        | `10.0`    | Minimum capital % per rule.                                                                  |
| `PHASE4_CAPITAL_PCT_MAX`        | `50.0`    | Maximum capital % per rule.                                                                  |
| `PHASE4_TP_STEP`                | `0.2`     | TP quantization step. Decrease for finer search (larger search space).                       |
| `PHASE4_SL_STEP`                | `0.2`     | SL quantization step.                                                                        |
| `PHASE4_CAPITAL_STEP`           | `5.0`     | Capital % quantization step.                                                                 |
| `PHASE4_TOTAL_CAP_PENALTY`      | `2.0`     | Penalty per unit of over-allocation. Increase to more aggressively penalize over-allocation. |
| `PHASE4_N_TRIALS`               | `200`     | Total Optuna trials. Increase for better optimization (linear compute cost).                 |
| `PHASE4_WF_SPLITS`              | `4`       | Number of walk-forward windows. Increase for more robust worst-case estimates.               |
| `PHASE4_MAX_WORST_DRAWDOWN_PCT` | `15.0`    | Maximum acceptable worst-case drawdown. Decrease for more conservative strategies.           |
| `PHASE4_SAMPLER`                | `"nsga2"` | Optuna sampler. `"tpe"` for faster convergence, `"nsga2"` for better Pareto diversity.       |
| `PHASE4_SEED`                   | `42`      | Random seed for reproducibility.                                                             |
| `PHASE4_N_JOBS`                 | `1`       | Parallel Optuna workers. Increase to use more CPU cores.                                     |
| `PHASE4_HARD_CAP_NORMALIZE`     | `True`    | Scale capital_pct so sum ≤ MAX_TOTAL_EXPOSURE_PCT. Disable only if using leverage.           |

---

## 9. Outputs

- `outputs/long.json` / `outputs/short.json` — Updated strategy files with optimized TP/SL/capital_pct and `"risk_optimized": true`.
- `outputs/reports/phase4_long_pareto.png` / `outputs/reports/phase4_short_pareto.png` — Pareto frontier plot (worst Sortino vs. worst drawdown across all trials).
