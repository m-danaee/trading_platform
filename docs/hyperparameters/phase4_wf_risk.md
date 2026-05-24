# Phase 4 — Walk-Forward Risk Optimization

**Module:** [`gpu_fuzzy_trader/phases/phase4_wf_optimizer.py`](../../gpu_fuzzy_trader/phases/phase4_wf_optimizer.py)  
**Config prefix:** `PHASE4_*`  
**Data split:** Validation only (walk-forward windows)

[← Phase 3](phase3_rule_set.md) | [Index](README.md) | [Phase 5 →](phase5_oos.md)

---

## Purpose

Phase 4 **fine-tunes risk parameters** for each rule in the team selected in Phase 3:

- Take-profit (`tp`)
- Stop-loss (`sl`)
- Capital allocation (`capital_pct`)

**Rule conditions are frozen** — only continuous risk knobs change.

Optimization uses **Optuna multi-objective search** (NSGA-II by default) over **quantized** parameter grids, evaluated on **K walk-forward validation windows** with a **worst-case** objective across windows.

---

## Walk-forward splits

Validation data is split into `PHASE4_WF_SPLITS` (default **2**) windows:

- Per symbol: rows sorted by `datetime`, split into K equal contiguous chunks.
- Window _i_ = concatenation of all symbols' chunk _i_.
- No shuffling — candle order is preserved.

Each symbol must have at least K validation rows.

---

## Search space (quantized)

Per rule _i_:

| Parameter     | Default range | Step |
| ------------- | ------------- | ---- |
| `tp`          | 2.0 – 4.0 %   | 0.2  |
| `sl`          | 1.0 – 2.0 %   | 0.2  |
| `capital_pct` | 10.0 – 50.0 % | 5.0  |

Configured via `PHASE4_TP_MIN/MAX`, `PHASE4_SL_MIN/MAX`, `PHASE4_CAPITAL_PCT_MIN/MAX`, and `PHASE4_*_STEP`.

---

## Objective (worst-case across windows)

For each trial, run `CPUBacktestEngine.simulate_rule_set` on every walk-forward window.  
Metrics: `sortino_ratio`, `max_drawdown_pct`.

```
overalloc_penalty = max(0, sum(capital_pct) - 100) / 100 × PHASE4_TOTAL_CAP_PENALTY

worst_sortino   = min(split_sortinos)   - overalloc_penalty   → maximize
worst_drawdown  = max(split_drawdowns) + overalloc_penalty   → minimize
```

---

## Pareto selection

After `PHASE4_N_TRIALS` (default 1000):

1. Take Optuna Pareto front (`study.best_trials`).
2. Discard trials with `worst_drawdown > PHASE4_MAX_WORST_DRAWDOWN_PCT` (default 15%).
3. Among survivors, pick **highest worst-case Sortino**.
4. Fallback if none pass filter: minimum worst drawdown, then highest Sortino.
5. Apply `PHASE4_HARD_CAP_NORMALIZE` if sum of `capital_pct` > 100%.
6. Write `outputs/long.json` and `outputs/short.json` with `risk_optimized: true`.

---

## Parallel trials (`PHASE4_N_JOBS`)

Default **`PHASE4_N_JOBS = 1`**.

When `n_jobs > 1`, Optuna runs trials concurrently in-process (threads). Each trial must:

- Create a **new** `CPUBacktestEngine(split_df, {}, direction)` per walk-forward window.
- Treat `val_splits` and rule `conditions` as **read-only**.
- **Not** use `simulate_rule_set_batch` or `Phase3EvalCache`.

Backtests are CPU-bound; thread speedup may be sub-linear due to the GIL.

---

## Config reference

| Key                             | Default   | Role                                   |
| ------------------------------- | --------- | -------------------------------------- |
| `PHASE4_N_TRIALS`               | `1000`    | Optuna trials                          |
| `PHASE4_WF_SPLITS`              | `2`       | Walk-forward windows                   |
| `PHASE4_SAMPLER`                | `"nsga2"` | `"nsga2"` or `"tpe"`                   |
| `PHASE4_SEED`                   | `42`      | Sampler seed                           |
| `PHASE4_N_JOBS`                 | `1`       | Parallel trial workers                 |
| `PHASE4_MAX_WORST_DRAWDOWN_PCT` | `15.0`    | Pareto safety filter                   |
| `PHASE4_TOTAL_CAP_PENALTY`      | `2.0`     | Soft overallocation penalty            |
| `PHASE4_HARD_CAP_NORMALIZE`     | `True`    | Scale capital to ≤ 100% post-selection |

---

## Outputs

| Path                                                                | Description                      |
| ------------------------------------------------------------------- | -------------------------------- |
| `outputs/long.json`, `outputs/short.json`                           | Final strategies for Phase 5     |
| `outputs/reports/phase4_long_pareto.png`, `phase4_short_pareto.png` | Pareto frontier + selected trial |

---

## Dependencies

- **Required:** `optuna>=3.5.0`
- Phase 4 no longer uses stable-baselines3, gymnasium, or PyTorch.
