# Phase 4 — RL Risk Optimization

**Module:** [`gpu_fuzzy_trader/phases/phase4_rl_optimizer.py`](../../gpu_fuzzy_trader/phases/phase4_rl_optimizer.py)  
**Config prefix:** `PHASE4_*`  
**Data split:** Train environment / validation scoring for checkpoint selection

[← Phase 3](phase3_rule_set.md) | [Index](README.md) | [Phase 5 →](phase5_oos.md)

---

## Purpose and statistical framing

Phase 4 **fine-tunes risk parameters** for each rule in the selected team:

- Take-profit (`tp`)
- Stop-loss (`sl`)
- Capital allocation (`capital_pct`)

**Rule conditions are frozen** — only continuous risk knobs change. This separates **alpha** (Phase 2–3) from **position sizing and exit levels** (Phase 4).

### Implementation paths

| Path                              | When used                                           |
| --------------------------------- | --------------------------------------------------- |
| **stable-baselines3** DDPG or PPO | `stable-baselines3`, `gymnasium`, `torch` available |
| **Optuna TPE**                    | SB3/gym missing                                     |
| **Random search + elbow**         | Optuna also missing                                 |

All paths use validation metrics for checkpoint selection via the **elbow method** on the validation returns curve.

---

## Action space

Per rule `i` (R rules in team):

```
action = [tp_i, sl_i, capital_pct_i]
```

Bounds from config:

| Parameter                | Default | Meaning                  |
| ------------------------ | ------- | ------------------------ |
| `PHASE4_TP_MIN`          | `2.0`   | Minimum TP %             |
| `PHASE4_TP_MAX`          | `4.0`   | Maximum TP %             |
| `PHASE4_SL_MIN`          | `1.0`   | Minimum SL %             |
| `PHASE4_SL_MAX`          | `2.0`   | Maximum SL %             |
| `PHASE4_CAPITAL_PCT_MIN` | `10.0`  | Minimum capital per rule |
| `PHASE4_CAPITAL_PCT_MAX` | `50.0`  | Maximum capital per rule |

**Performance effects of widening bounds:**

- Wider TP/SL → more extreme risk profiles; higher overfit risk on validation elbow.
- Narrow bounds → safer search; may underfit if Phase 2 static risk was far from optimal.

Default bounds are tighter than historical README values (aligned with Phase 2 static TP=2, SL=1).

---

## Reward vision and reward (RL environment)

**State vector:**

```
[K market features, R rule activation strengths, equity_normalized, open_exposure_normalized]
```

**Step reward (training env):**

```
reward = net_pnl_normalized - drawdown_penalty - overalloc_penalty
```

**Evaluation window:** `PHASE4_RL_EVAL_WINDOW` bars (default 288 — one full label horizon).

---

## Validation scoring (checkpoint selection)

Primary validation objective for search fallbacks and elbow selection:

```
score = val_return + sortino × PHASE4_VAL_SORTINO_WEIGHT - overalloc_penalty
```

Where:

```
sortino = clip(raw_val_sortino, -PHASE4_VAL_SORTINO_BONUS_CAP, +PHASE4_VAL_SORTINO_BONUS_CAP)
overalloc_penalty = max(0, sum(capital_pct) - 100) / 100 × PHASE4_TOTAL_CAP_PENALTY
```

| Parameter                      | Default | Effect                                                                      |
| ------------------------------ | ------- | --------------------------------------------------------------------------- |
| `PHASE4_VAL_SORTINO_WEIGHT`    | `1.0`   | Weight on validation Sortino (raised from 0.2 — now co-primary with return) |
| `PHASE4_VAL_SORTINO_BONUS_CAP` | `5.0`   | Caps Sortino influence; prevents outlier Sortino from dominating            |
| `PHASE4_TOTAL_CAP_PENALTY`     | `2.0`   | Penalizes sum of capital_pct > 100%                                         |

---

## Capital normalization

| Parameter                   | Default | Effect                                                    |
| --------------------------- | ------- | --------------------------------------------------------- |
| `PHASE4_HARD_CAP_NORMALIZE` | `True`  | After optimization, scale all `capital_pct` so sum ≤ 100% |

Historical issue: Phase 4 outputs summed to 75% (long) and 108% (short) without normalization — short violated `MAX_TOTAL_EXPOSURE_PCT`.

- **Keep enabled** for production runs.
- **Disable only** for ablation of raw RL output behavior.

---

## Training budget and early stopping

| Parameter                | Default   | Effect                                                      |
| ------------------------ | --------- | ----------------------------------------------------------- |
| `PHASE4_RL_ALGORITHM`    | `"DDPG"`  | `"PPO"` alternative when SB3 available                      |
| `PHASE4_TOTAL_TIMESTEPS` | `500_000` | RL training length (SB3 path)                               |
| `PHASE4_ELBOW_WINDOW`    | `15`      | Checkpoint every N episodes/samples; elbow picks best index |

**Fallback trial count:**

```
n_trials ≈ max(PHASE4_ELBOW_WINDOW, PHASE4_TOTAL_TIMESTEPS // 100)  → 5000 default
```

### Elbow method

Validation returns recorded at each checkpoint. The index with **maximum perpendicular distance** from the line connecting first and last checkpoint is selected — reduces overfitting to late training noise.

| Symptom                                    | Knob                                    |
| ------------------------------------------ | --------------------------------------- |
| Elbow at first checkpoint (under-training) | ↑ timesteps or ↓ elbow_window           |
| Elbow at last checkpoint (over-training)   | ↓ timesteps or ↑ elbow_window           |
| Noisy validation curve                     | ↑ elbow_window for smoother checkpoints |

---

## Hyperparameter summary table

| Parameter                      | Default   | ↑ increase                           | ↓ decrease                        |
| ------------------------------ | --------- | ------------------------------------ | --------------------------------- |
| `PHASE4_TP_MIN/MAX`            | 2–4%      | Wider TP search                      | Narrower exits                    |
| `PHASE4_SL_MIN/MAX`            | 1–2%      | Wider SL search                      | Tighter stops                     |
| `PHASE4_CAPITAL_PCT_MIN/MAX`   | 10–50%    | Larger position sizes allowed        | More conservative sizing          |
| `PHASE4_TOTAL_CAP_PENALTY`     | `2.0`     | Stronger penalty for >100% total cap | Allows overallocation in search   |
| `PHASE4_RL_EVAL_WINDOW`        | `288`     | Longer rolling eval horizon          | Faster env steps; noisier reward  |
| `PHASE4_VAL_SORTINO_WEIGHT`    | `1.0`     | Sortino drives selection more        | Return-only selection             |
| `PHASE4_VAL_SORTINO_BONUS_CAP` | `5.0`     | Allows higher Sortino bonus          | Dampens Sortino outliers          |
| `PHASE4_TOTAL_TIMESTEPS`       | `500_000` | Longer RL training                   | Faster; may underfit risk surface |
| `PHASE4_ELBOW_WINDOW`          | `15`      | Fewer checkpoints; smoother curve    | More granular checkpoint grid     |
| `PHASE4_HARD_CAP_NORMALIZE`    | `True`    | (bool) enforce exposure cap          | Raw RL sums may exceed 100%       |

---

## Interactions

- **Phase 2 static risk** seeds rule JSON; Phase 4 searches around a different box (TP 2–4 vs Phase 2 TP=2).
- **`MAX_TOTAL_EXPOSURE_PCT`** (Phase 0) enforced at simulation; normalization ensures JSON outputs respect cap.
- **`FEE_PCT`** affects reward magnitude in env — not tunable in Phase 4 but affects optimal trade frequency.

---

## Diagnostics

| Artifact                                  | What to check                                                |
| ----------------------------------------- | ------------------------------------------------------------ |
| `outputs/long.json`, `outputs/short.json` | `risk_optimized: true`; per-rule tp/sl/capital_pct in bounds |
| `outputs/reports/phase4_*_rl_curve.png`   | Validation curve shape; elbow index                          |
| Sum of `capital_pct`                      | Should be ≤ 100 after normalization                          |

---

## Code references

Validation score:

```171:189:gpu_fuzzy_trader/phases/phase4_rl_optimizer.py
def _phase4_val_score(val_metrics: dict, candidate_params: list[dict]) -> float:
    ...
    return (
        val_return
        + sortino * _cfg.PHASE4_VAL_SORTINO_WEIGHT
        - overalloc
    )
```

Fallback sample count:

```166:168:gpu_fuzzy_trader/phases/phase4_rl_optimizer.py
def _phase4_sample_count(total_timesteps: int, elbow_window: int) -> int:
    return max(elbow_window, total_timesteps // 100)
```

Elbow selection:

```85:163:gpu_fuzzy_trader/phases/phase4_rl_optimizer.py
def find_elbow_point(validation_returns: list[float]) -> int:
    ...
```
