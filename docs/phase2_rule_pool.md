# Phase 2 — GPU-Accelerated Rule Pool Generation

**Modules:**

- `gpu_fuzzy_trader/phases/phase2_rule_pool.py` → `Rule_Pool_Generator` (orchestration, persistence, archive)
- `gpu_fuzzy_trader/phases/phase2_cv.py` → purged CV engine facades (`PurgedCVTrainEngine`, `PurgedCVValEngine`)
- `gpu_fuzzy_trader/evolution/evox_runner.py` → `run_phase2_evolution` (NSGA-III/II loop)
- `gpu_fuzzy_trader/phases/phase2_support.py` → regime-aware support penalties
- `gpu_fuzzy_trader/evolution/numba_ops.py` → Numba-accelerated NSGA helpers

Phase 2 is the core search phase. It evolves a large, diverse pool of candidate fuzzy trading rules using multi-objective evolutionary optimization. The output is a Pareto-front pool of rules that trade off Sortino ratio, drawdown, and win rate — not a single "best" rule.

---

## 1. Chromosome Encoding

Each candidate rule is represented as an integer chromosome:

```
chromosome = [gene_0, gene_1, ..., gene_{K-1}]
```

where K = number of selected features from Phase 1 (e.g., 20 for long, 20 for short).

Each gene `gene_i` can take values in `{0, 1, ..., num_classes_i − 1, dont_care_i}`:

- Values `0` to `num_classes_i − 1` are **active** conditions: the rule requires that feature to be in that fuzzy state.
- Value `dont_care_i = num_classes_i` means the condition is **inactive**: the feature is ignored by this rule.

For example, a chromosome for a `signed` feature (10 classes) can have gene values 0–9 (active) or 10 (dont_care). A `binary` feature can have 0, 1 (active) or 2 (dont_care).

### Active condition count

The number of active conditions in a chromosome is `sum(gene_i != dont_care_i)`. This is constrained by `MIN_CONDITIONS` and `MAX_CONDITIONS` (default: 3–4). Rules with fewer or more active conditions receive a condition count penalty.

**Effect of `MIN_CONDITIONS` / `MAX_CONDITIONS`:**

- Increasing `MIN_CONDITIONS`: forces more specific rules (more conditions must match simultaneously), reducing trade frequency but potentially improving precision.
- Decreasing `MAX_CONDITIONS`: forces simpler rules, increasing trade frequency but potentially reducing precision.
- The range 3–4 is a balance between specificity and trade support.

---

## 2. Three Objectives (all minimized)

```
f1 = −sortino_ratio   (maximize Sortino)
f2 = max_drawdown_pct (minimize drawdown)
f3 = −win_rate        (maximize win rate)
```

All three objectives have penalties added before the evolutionary algorithm sees them. The penalties are described in Section 4.

### Sortino saturation — `_saturating_sortino`

Raw Sortino values are passed through a tanh transform before being used as objectives:

```python
saturated = tanh(raw_sortino / SORTINO_SCALE) × SORTINO_CAP
```

`SORTINO_SCALE = 5.0`, `SORTINO_CAP = 15.0` (config).

**Why saturation?** Without it, a rule with Sortino = 100 would dominate the Pareto front and crowd out rules with Sortino = 5 that might have better drawdown or win rate. The tanh transform compresses extreme values, allowing the Pareto front to maintain diversity across the objective space.

**Effect of `SORTINO_SCALE`:** Increasing this value makes the saturation less aggressive (the tanh curve is flatter), allowing higher raw Sortino values to differentiate. Decreasing it compresses more aggressively.

**Effect of `SORTINO_CAP`:** The maximum possible saturated Sortino. Increasing this allows more differentiation at the top end.

### Joint train/val objective — `PHASE2_JOINT_TRAIN_VAL`

When `PHASE2_JOINT_TRAIN_VAL = True` (default), f1 uses:

```python
sortino_for_obj = min(saturated_train_sortino, saturated_val_sortino)
```

This means a rule must perform well on **both** the training and validation sides to achieve a good f1 score. Rules that overfit to training (high train Sortino, low val Sortino) are penalized.

If the validation engine is unavailable or val trades are below `MIN_TRADE_POOL_FLOOR // 4`, the val Sortino is treated as 0 and `sortino_for_obj = min(train_sortino, 0.0)`.

### Purged rolling CV (`SPLIT_MODE == "purged_rolling_cv"`)

When the pipeline passes `cv_folds` into `Rule_Pool_Generator`:

1. `build_cv_fold_engines()` builds one train + one val engine per fold (each subsampled to `PHASE1_SAMPLING_TOTAL`).
2. `PurgedCVTrainEngine` / `PurgedCVValEngine` wrap the fold engines and expose `simulate_rule_batch()` that merges metrics with a **worst-case** rule across folds (min return/Sortino/PF, max drawdown).
3. Joint fitness uses `min(train_sortino_fold, val_sortino_fold)` **per fold**, then the facade's worst-case merge across folds.

**Effect:** A rule that shines in only one season but fails in another gets a poor f1.

**Pool admission (CV):** `evaluate_purged_cv_pool_admission()` checks each fold with `passes_pool_admission_cv_fold` (lower trade/return floors). A rule enters the pool when at least `PHASE2_CV_POOL_MIN_FOLDS_PASS` folds pass (default 2 of 3). Stored objectives still use worst-case merged metrics from the CV facades; pool JSON also stores `cv_folds_passing` / `cv_folds_total` so the post-merge filter uses the same CV criterion (merged metrics alone can be negative while 2/3 folds pass). Early stop is disabled in CV (`PHASE2_EARLY_STOP_DISABLED_IN_CV`).

When `SPLIT_MODE == "holdout_75_25"`, behaviour is unchanged: one train engine (sampled full `train_75`) and one optional val engine (`validation_25`).

---

## 3. Static Risk Parameters During Phase 2

During Phase 2, all rules are evaluated with fixed risk parameters:

- `PHASE2_TP = 2.0` (%)
- `PHASE2_SL = 1.5` (%)
- `PHASE2_CAPITAL_PCT = 32.0` (%)

**Why fixed?** Phase 2 is searching for rules with predictive alpha — the ability to identify market conditions that precede favorable price moves. By fixing TP/SL/capital, the search isolates rule quality from risk parameter tuning. Phase 4 handles risk optimization separately.

**Effect of `PHASE2_TP` / `PHASE2_SL`:** These define what counts as a "win" during Phase 2 evaluation. A higher TP requires larger price moves to trigger a win, reducing trade frequency but increasing per-trade return. A lower SL is more conservative but may cut winners early. These values also feed back into Phase 1's target construction (see Phase 1 docs).

---

## 4. Penalties

All penalties are added to all three Phase 2 objectives simultaneously, but
some penalties are scaled per-objective (notably the support penalty via
`PHASE2_SUPPORT_PENALTY_WEIGHT_F1/F2/F3`).

### Support penalty — `trade_support_penalty`

If `executed_trades < MIN_TRADE_SUPPORT` (default: 200), a graduated penalty is applied:

```python
if executed < MIN_TRADE_POOL_FLOOR:
    penalty = 2.0 × SUPPORT_PENALTY_MAX   # hard reject
else:
    shortfall = (MIN_TRADE_SUPPORT − executed) / MIN_TRADE_SUPPORT
    penalty = min(shortfall² × SUPPORT_PENALTY_MAX, SUPPORT_PENALTY_MAX)
```

`SUPPORT_PENALTY_MAX = 12.0`, `MIN_TRADE_POOL_FLOOR = 50` (config).

In Phase 2 fitness evaluation, this support penalty is additionally
multiplied by per-objective weights `PHASE2_SUPPORT_PENALTY_WEIGHT_F1`,
`PHASE2_SUPPORT_PENALTY_WEIGHT_F2`, and `PHASE2_SUPPORT_PENALTY_WEIGHT_F3`.

**Why?** A rule with 10 trades has a noisy Sortino estimate. The support penalty discourages the search from converging on low-frequency rules whose apparent performance is statistical noise.

**Effect of `MIN_TRADE_SUPPORT`:** Increasing this requires more trades before a rule is considered reliable, reducing the risk of overfitting to rare patterns but also potentially discarding valid low-frequency rules. Decreasing it allows noisier rules into the pool.

**Effect of `SUPPORT_PENALTY_MAX`:** Increasing this makes the penalty steeper, more aggressively pushing low-support rules off the Pareto front.

### Regime-aware support penalty — `phase2_support.py`

When `PHASE2_REGIME_SUPPORT_ENABLED = False`, regime-specialist waivers are disabled and all rules follow the global support penalty. If you re-enable the flag for an experiment, a rule that concentrates its trades in one market regime can bypass the global support penalty if it meets the specialist criteria:

1. **Concentration:** `trades_in_dominant_regime / total_trades ≥ PHASE2_REGIME_CONCENTRATION_MIN` (default: 0.90)
2. **Threshold:** `trades_in_dominant_regime ≥ per_regime_trade_thresholds[dominant_regime]`
3. **Quality:** win rate in dominant regime ≥ `PHASE2_REGIME_MIN_WIN_RATE` (0.40) OR net PnL in dominant regime > 0

The per-regime threshold scales with both the regime's row fraction and the rule's total trade count, so a regime that represents 30% of the data requires proportionally fewer trades than a 90% regime.

**Effect of `PHASE2_REGIME_CONCENTRATION_MIN`:** Increasing this (e.g., to 0.95) requires rules to be even more concentrated in one regime to qualify as specialists. Decreasing it allows more diffuse rules to bypass the penalty.

**Effect of `PHASE2_REGIME_MIN_WIN_RATE`:** The minimum win rate a specialist must achieve in its dominant regime. Increasing this raises the quality bar for specialists.

### Diversity penalty — `PHASE2_DIVERSITY_HAMMING_THRESHOLD` / `PHASE2_DIVERSITY_PENALTY`

If a chromosome's Hamming distance to the nearest Pareto-front member is ≤ `PHASE2_DIVERSITY_HAMMING_THRESHOLD` (default: 2), it receives a `PHASE2_DIVERSITY_PENALTY` (default: 5.0) on all objectives.

**Why?** Without this, the Pareto front tends to cluster around a few high-performing chromosomes with minor variations. The diversity penalty encourages the search to explore different regions of the chromosome space.

**Effect of `PHASE2_DIVERSITY_HAMMING_THRESHOLD`:** Increasing this enforces more diversity (chromosomes must differ in more gene positions to avoid the penalty). Decreasing it allows more similar chromosomes.

### Condition count penalty

If `active_conditions < MIN_CONDITIONS`: penalty = `(MIN_CONDITIONS − active) × 10.0`
If `active_conditions > MAX_CONDITIONS`: penalty = `(active − MAX_CONDITIONS) × 10.0`

This steers the search toward rules with 3–4 active conditions.

### Fee model

Phase 2 and Phase 3 use the shared fee setting `FEE_PCT` for optimization
backtests.

---

## 5. NSGA-III Evolutionary Loop — `evox_runner.py`

### Population initialization — `_init_population` / `phase2_init.py`

Default strategy (`PHASE2_INIT_STRATEGY = "stratified_sparse"`):

1. `PHASE2_ARCHIVE_SEED_FRACTION = 0.35` of slots are filled from the cross-run archive (unchanged).
2. Each remaining individual picks `k` uniformly in `[MIN_CONDITIONS, MAX_CONDITIONS]`, starts with all genes at `dont_care`, then activates exactly `k` genes via one of three strata (fractions from `PHASE2_INIT_STRATUM_FRACTIONS`, default 50% / 30% / 20%):
   - **Elite:** feature indices sampled without replacement using softmax Phase 1 scores (`PHASE2_INIT_SOFTMAX_TEMP`, with `PHASE2_INIT_UNIFORM_MIX` floor).
   - **Explorer:** uniform feature sampling.
   - **Regime specialist (disabled in regression mode):** this stratum is deactivated because regime features are not directly selected. The regime share automatically falls back to elites.

Legacy mode (`init_strategy="legacy"`) keeps independent per-gene `dont_care_prob` sampling for tests.

Train/val splits do not need to preserve GMM features since daily regimes are computed directly from label/metadata columns.

**Follow-up (not yet implemented):** union of Pareto fronts across generations is required to reach 40–60 rules in the exported pool; initialization alone only fixes early-generation fitness artifacts.

### Offspring generation — `_make_offspring_population`

1. **Binary tournament selection:** Two random individuals are compared by Pareto rank (lower is better), then by crowding distance (higher is better). The winner becomes a parent.
2. **Uniform crossover:** Each gene is independently chosen from either parent with probability 0.5.
3. **Mutation:** Each gene is mutated with probability `mutation_rate = 0.1`. Activating a `dont_care` gene uses `PHASE2_MUTATION_WEIGHTED_ACTIVATE_PROB` (default 0.7) to pick an inactive feature from the Phase 1 softmax distribution, else uniform. Active count is repaired to `[MIN_CONDITIONS, MAX_CONDITIONS]` after mutation.

### NSGA-III environmental selection — `_nsga3_environmental_selection`

After merging parents and offspring (2N individuals), the next generation of N individuals is selected:

1. **Non-dominated ranking** (via EvoX's `non_dominate_rank`): Individuals are assigned Pareto ranks 0, 1, 2, …
2. **Critical front identification:** The worst rank `r*` such that the first `r*` fronts contain ≤ N individuals is found.
3. **Niche-based selection on the critical front:** Reference vectors (Das-Dennis simplex sampling) are used to associate each individual with the nearest reference vector. Individuals are selected to fill niches with the fewest members, promoting diversity across the objective space.

**When EvoX is not available:** Falls back to NSGA-II (crowding distance truncation on the critical front instead of niche-based selection). The history records `"NSGA-II (fallback)"`.

### Reference vectors — `_get_reference_vectors`

Das-Dennis style reference vectors are generated using EvoX's `uniform_sampling`. These vectors define directions in the 3-objective space. The niche-based selection ensures that the final population covers all directions, not just the extremes.

---

## 6. Population and Generation Budget

| Parameter                | Default | Effect                                                                                                                                                       |
| ------------------------ | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `PHASE2_POPULATION_SIZE` | `200`   | Number of chromosomes per generation. Increasing improves Pareto front coverage but linearly increases compute time per generation.                          |
| `PHASE2_GENERATIONS`     | `200`   | Number of generations. Increasing allows more evolution but with diminishing returns after convergence. Total evaluations ≈ `POPULATION_SIZE × GENERATIONS`. |

At default settings, Phase 2 performs approximately 200 × 200 = 40,000 chromosome evaluations per direction. Each evaluation is a full backtest simulation.

**Compute cost:** With the GPU engine (JAX), evaluations are batched and run in parallel on the GPU. With the CPU engine, evaluations are sequential. The GPU engine is roughly 10–100× faster depending on hardware.

---

## 7. Data Sampling — `_sample_df`

Phase 2 does not use the full training dataset. Instead, it draws a random sample of up to `PHASE1_SAMPLING_TOTAL = 600_000` rows, distributed equally across symbols:

```python
rows_per_sym = max(1, total_rows // n_sym)
```

**Why sampling?** The full training dataset may have millions of rows. Loading all of them into GPU memory for JAX evaluation would exceed VRAM on most hardware. Sampling reduces memory usage while preserving the per-symbol distribution.

The subset is re-drawn on each run, so repeated runs can explore different slices of the data while keeping the per-symbol balance.

**Effect of `PHASE1_SAMPLING_TOTAL`:** This is the primary GPU memory knob. Increasing it improves the statistical reliability of fitness evaluations but increases VRAM usage roughly linearly. On a 16GB GPU, 600,000 rows with ~100 features uses approximately 4–6GB. Decrease to 150,000–300,000 if you encounter OOM errors.

---

## 8. Archive System

### Per-run pool (overwritten each run)

After evolution, the Pareto-front chromosomes are converted to pool entries and saved to `outputs/phase2_long_pool.json` / `outputs/phase2_short_pool.json`. Each entry contains:

- `chromosome`: integer array
- `conditions`: decoded condition strings
- `objectives`: `{sortino_ratio, max_drawdown_pct, win_rate}`
- `executed_trades`: integer

### Cross-run archive (persistent)

The best rules from all runs are accumulated in `phase2_rule_archive/phase2_long_archive.json` / `phase2_rule_archive/phase2_short_archive.json`. This archive is never cleared by `--output` flags.

Archive merging uses `_merge_archive_entries`:

1. Deduplicate by chromosome (keep the entry with better objectives for duplicate chromosomes).
2. Non-dominated sort all unique entries.
3. Keep the best `PHASE2_ARCHIVE_MAX_SIZE` (default: 500) entries using Pareto rank + crowding distance truncation.

**Effect of `PHASE2_ARCHIVE_MAX_SIZE`:** Increasing this preserves more historical rules for warm-starting future runs. Decreasing it keeps only the most elite rules.

### Pool trade floor gate — `passes_pool_trade_floor`

A chromosome is only included in the pool if `executed_trades ≥ MIN_TRADE_POOL_FLOOR` (default: 50), unless it qualifies as a regime specialist. This is a hard gate that prevents very low-frequency rules from entering the pool regardless of their apparent Sortino.

---

## 9. Configuration Reference

| Parameter                                | Default               | Technical effect                                                                                                                |
| ---------------------------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `PHASE2_TP`                              | `3.0`                 | TP % used for all Phase 2 evaluations. Increasing requires larger price moves to win, reducing trade frequency.                 |
| `PHASE2_SL`                              | `1.5`                 | SL % used for all Phase 2 evaluations. Increasing allows more drawdown before stopping out.                                     |
| `PHASE2_CAPITAL_PCT`                     | `32.0`                | Capital % per rule during Phase 2. Affects position sizing and thus absolute PnL, but not Sortino (which is return-normalized). |
| `MIN_CONDITIONS`                         | `3`                   | Minimum active conditions per rule. Increase for more specific rules.                                                           |
| `MAX_CONDITIONS`                         | `4`                   | Maximum active conditions per rule. Decrease for simpler rules.                                                                 |
| `MIN_TRADE_SUPPORT`                      | `200`                 | Minimum trades for zero support penalty. Increase to require more statistical evidence.                                         |
| `SUPPORT_PENALTY_MAX`                    | `12.0`                | Maximum support penalty magnitude. Increase to more aggressively penalize low-frequency rules.                                  |
| `MIN_TRADE_POOL_FLOOR`                   | `50`                  | Hard minimum trades for pool inclusion. Rules below this are excluded regardless of Sortino.                                    |
| `PHASE2_SUPPORT_PENALTY_WEIGHT_F1`       | `1.0`                 | Multiplicative weight applied to support penalty in Phase 2 objective `f1`.                                                     |
| `PHASE2_SUPPORT_PENALTY_WEIGHT_F2`       | `0.35`                | Multiplicative weight applied to support penalty in Phase 2 objective `f2`.                                                     |
| `PHASE2_SUPPORT_PENALTY_WEIGHT_F3`       | `0.35`                | Multiplicative weight applied to support penalty in Phase 2 objective `f3`.                                                     |
| `SORTINO_CAP`                            | `5.0`                 | Maximum saturated Sortino value. Increase to allow more differentiation at the top end.                                         |
| `SORTINO_SCALE`                          | `3.0`                 | tanh saturation scale. Increase for less aggressive compression.                                                                |
| `PHASE2_JOINT_TRAIN_VAL`                 | `True`                | Use min(train, val) Sortino as objective. Disable to optimize on training only (higher overfitting risk).                       |
| `PHASE2_DIVERSITY_HAMMING_THRESHOLD`     | `2`                   | Hamming distance threshold for diversity penalty. Increase to enforce more diversity.                                           |
| `PHASE2_DIVERSITY_PENALTY`               | `5.0`                 | Penalty magnitude for near-duplicate chromosomes. Increase to more aggressively enforce diversity.                              |
| `PHASE2_POPULATION_SIZE`                 | `200`                 | Population size. Increase for better Pareto coverage (linear compute cost).                                                     |
| `PHASE2_GENERATIONS`                     | `200`                 | Number of generations. Increase for more evolution (linear compute cost).                                                       |
| `PHASE2_ALGORITHM`                       | `"NSGA3"`             | Fixed. NSGA-III when EvoX is installed, NSGA-II fallback otherwise.                                                             |
| `PHASE2_ARCHIVE_MAX_SIZE`                | `500`                 | Maximum archive size per direction.                                                                                             |
| `PHASE2_ARCHIVE_SEED_FRACTION`           | `0.35`                | Fraction of population seeded from previous pool. Increase for faster convergence, decrease for more exploration.               |
| `PHASE2_INIT_STRATEGY`                   | `"stratified_sparse"` | `stratified_sparse` enforces 3–4 active genes; `legacy` uses per-gene dont_care probability.                                    |
| `PHASE2_INIT_STRATUM_FRACTIONS`          | `(0.5, 0.3, 0.2)`     | Elite / explorer / regime shares of non-seeded population.                                                                      |
| `PHASE2_INIT_SOFTMAX_TEMP`               | `0.5`                 | Temperature for Phase 1 score softmax in elite stratum.                                                                         |
| `PHASE2_INIT_UNIFORM_MIX`                | `0.05`                | Uniform floor mixed into feature sampling probabilities.                                                                        |
| `PHASE2_MUTATION_WEIGHTED_ACTIVATE_PROB` | `0.70`                | Probability of softmax-weighted vs uniform choice when mutation activates a gene.                                               |
| `PHASE2_REGIME_SUPPORT_ENABLED`          | `True`                | Enable regime-aware specialist bypass. Disable to use only global support penalty.                                              |
| `PHASE2_REGIME_CONCENTRATION_MIN`        | `0.90`                | Minimum trade concentration in one regime for specialist status.                                                                |
| `PHASE2_REGIME_MIN_WIN_RATE`             | `0.40`                | Minimum win rate in dominant regime for specialist quality gate.                                                                |
| `PHASE2_REGIME_USE_PNL_GATE`             | `True`                | Allow positive PnL (instead of win rate) to satisfy the specialist quality gate.                                                |
| `PHASE2_NUMBA_ENABLED`                   | `True`                | Use Numba-JIT NSGA helpers. Disable only for debugging.                                                                         |

---

## 10. Outputs

- `outputs/phase2_long_pool.json` / `outputs/phase2_short_pool.json` — Per-run Pareto-front pool.
- `outputs/phase2_long_history.json` / `outputs/phase2_short_history.json` — Per-generation metrics (pareto_size, mean_f1/f2/f3, best_sortino_ratio, algorithm name).
- `phase2_rule_archive/phase2_long_archive.json` / `phase2_rule_archive/phase2_short_archive.json` — Persistent cross-run archive.
- `outputs/reports/phase2_long_metrics.png` / `outputs/reports/phase2_short_metrics.png` — Objective evolution plots.
