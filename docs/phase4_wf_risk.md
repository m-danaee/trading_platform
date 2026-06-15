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

## 3. Deterministic Risk Grid Search

Phase 4 uses a deterministic round-robin grid search for risk optimization. This replaces the legacy Optuna evolutionary search to ensure stability and exhaustively explore a wider aggressive parameter space.

### Search space (quantized)

Each rule's parameters are enumerated against the configured grid:

| Parameter     | Range         | Config keys                       |
| ------------- | ------------- | --------------------------------- |
| `tp`          | Up to 10.0%   | `PHASE4_GRID_TP_VALUES`           |
| `sl`          | Up to 3.0%    | `PHASE4_GRID_SL_VALUES`           |
| `capital_pct` | Up to 50.0%   | `PHASE4_GRID_CAPITAL_VALUES`      |

By default, the grid covers a very aggressive space compared to earlier versions. 

### Algorithm

1. **Initial Score:** The strategy is evaluated with default values to establish a baseline score using `_score_metrics` (a composite of return, drawdown, and profit factor).
2. **Round-Robin Passes:** For `PHASE4_GRID_PASSES` rounds (default 2), the search iterates through each rule:
   - Every combination of `(tp, sl, capital_pct)` is temporarily applied.
   - The strategy is re-evaluated across the walk-forward windows.
   - **Constraints:** The combination is skipped if `sum(capital_pct) > PHASE4_GRID_MAX_TOTAL_CAPITAL` (default 95.0%) or if the rule fails the `gate_positive_good` test (which demands PF ≥ 1.0 and positive returns on both train and val).
   - **Improvement:** If the new combination improves the overall score by at least `PHASE4_GRID_MIN_IMPROVEMENT` (default 0.02), it is accepted.
3. **Completion:** Returns the rules with optimized parameters and updated metrics.

**Why deterministic grid search?** Optuna's stochastic nature sometimes gets stuck in local minima and can be inconsistent across runs, especially with a narrow parameter bounds. The grid search evaluates every point of the parameter space aggressively but prunes infeasible paths early.

---

## 4. Constraint Gates

Phase 4 heavily relies on two constraints to ensure out-of-sample health:

- `gate_positive_good`: Enforces that each rule maintains a Profit Factor ≥ 1.0 and strictly positive returns across the in-sample period.
- `PHASE4_GRID_MAX_TOTAL_CAPITAL`: An explicit ceiling on total leverage during the search (unlike the legacy mode which required a post-optimization normalization step).

---

## 5. Optuna Fallback (Legacy)

The Optuna NSGA-II/TPE search remains available for debugging or specific experiments by setting `PHASE4_GRID_ENABLED = False`. In legacy mode:
- Parameters are sampled continuously within `PHASE4_TP_MIN/MAX`, etc.
- A Pareto front is generated for `worst_sortino` vs `worst_drawdown`.
- The best trial is selected via `PHASE4_MAX_WORST_DRAWDOWN_PCT`.
- Total capital is scaled down via `PHASE4_HARD_CAP_NORMALIZE` post-optimization.

---

## 6. Skip Logic

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
