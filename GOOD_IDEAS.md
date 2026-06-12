# Good Ideas: Two-Stage Phase 2 and Per-Symbol Phase 3 Integration

This document outlines the design, motivation, and technical implementation details of two critical updates to the fuzzy rule trading pipeline:
1. **Phase 2: Two-Stage Evolution (Stage A & Stage B)**
2. **Phase 3: Per-Symbol Greedy Rule Selection & Merging**

---

## 1. Phase 2: Two-Stage Evolution

### Motivation
Fuzzy rule evolution is a computationally expensive process. The search space is extremely large, and standard multi-objective evolutionary algorithms can get stuck in local optima or overfit early on. 
The **Two-Stage Evolution** design divides the generation budget into two distinct evolutionary stages:
* **Stage A (Exploration)**: Focused on exploring the wide search space to discover diverse and profitable candidate rules.
* **Stage B (Refinement)**: Focused on polishing and validating the most robust candidates discovered during Stage A.

### High-Level Flow
```
+--------------------------------------------------------+
|                      Start Phase 2                     |
+--------------------------------------------------------+
                            |
                            v
+--------------------------------------------------------+
|          Stage A: Exploratory Evolution (Gen A)        |
+--------------------------------------------------------+
                            |
                            v
+--------------------------------------------------------+
|   Seed Stage B with:                                   |
|   - Top PHASE2_STAGE_B_SEED_TOP_K Stage A Rules        |
|   - Historic Archive Seeds                             |
+--------------------------------------------------------+
                            |
                            v
+--------------------------------------------------------+
|   Host memory/VRAM cleanup:                            |
|   - gc.collect() & jax.clear_caches()                  |
|   - Prevents OOM during XLA compile in Stage B         |
+--------------------------------------------------------+
                            |
                            v
+--------------------------------------------------------+
|           Stage B: Refinement Evolution (Gen B)        |
+--------------------------------------------------------+
                            |
                            v
+--------------------------------------------------------+
|   Merge Pool, Apply Admission Filters & Save Archive    |
+--------------------------------------------------------+
```

### Stage A: Exploration
* **Generations**: Controlled by `PHASE2_STAGE_A_GENERATIONS` (e.g., 85 generations).
* **Behavior**: Evolves a randomly initialized population biased by features selected in Phase 1. 
* **Focus**: Finding a diverse set of rules across all three objectives (Sortino, return, drawdown).

### Stage B: Refinement
* **Generations**: Controlled by `PHASE2_STAGE_B_GENERATIONS` (e.g., 45 generations).
* **Behavior**: Warm-started by seeding a fraction (`PHASE2_STAGE_B_SEED_FRACTION`) of the population with:
  1. The top `PHASE2_STAGE_B_SEED_TOP_K` rules from Stage A.
  2. The existing historical pool/archive seeds.
* **Focus**: Polishing and refining the best rules to satisfy strict validation gates and trade support constraints.

### Memory & Performance Optimization
Because Stage B introduces new seed chromosomes and a different population structure, JAX triggers a new XLA kernel compilation. This compilation step causes temporary spikes in host RAM. 
To prevent out-of-memory (OOM) errors and environment SIGKILLs:
1. **Cache Clearing**: The pipeline explicitly runs garbage collection and clears JAX caches before spawning Stage B:
   ```python
   import gc
   import jax
   gc.collect()
   jax.clear_caches()
   ```
2. **Data Caching**: During park/unpark cycles (where GPU engines are temporarily released to conserve VRAM), the training DataFrame is cached (`_cached_slim_train`) instead of being re-sampled on the fly. This avoids redundant CPU computations and stabilizes the dataset across unpark actions.

---

## 2. Phase 3: Per-Symbol Greedy Selection

### Motivation
Previously, Phase 3 constructed a global "team" of 2–3 rules evaluated across the entire portfolio of symbols. This often resulted in "average" rules that performed moderately well across the portfolio but missed symbol-specific micro-patterns.
The new design shifts to a **Per-Symbol Greedy Selection**. It optimizes rule combinations independently for each symbol and then merges the results using symbol-conditional constraints.

### High-Level Flow
```
+--------------------------------------------------------+
|                      Start Phase 3                     |
+--------------------------------------------------------+
                            |
                            v
+--------------------------------------------------------+
|   For each Symbol in Universe:                         |
|   1. Filter symbol-specific Train & Val DataFrames     |
|   2. Run 3-Round Greedy Selection using                |
|      Robust Return: min(train_return, val_return)      |
|   3. Apply Gap Rejection:                              |
|      val_return - train_return <= MAX_GAP              |
+--------------------------------------------------------+
                            |
                            v
+--------------------------------------------------------+
|   Merge Selected Rules:                                |
|   - Add "symbol is X" conditions for each symbol       |
+--------------------------------------------------------+
                            |
        +-------------------+-------------------+
        |                                       |
        v (Fewer than min rules)                v (Sufficient rules)
+-------------------------------+       +-------------------------------+
| Global Pool Fallback          |       | Cap at Global Max Rules       |
| - Rank pool by robust return  |       +-------------------------------+
| - Extract top N unique rules  |                       |
+-------------------------------+                       |
        |                                               |
        +-------------------+---------------------------+
                            |
                            v
+--------------------------------------------------------+
|             Validate Schema & Save JSON Outputs        |
+--------------------------------------------------------+
```

### Per-Symbol Greedy Search Algorithm
For each symbol in the universe, the selector builds a localized rule set of up to `PHASE3_PER_SYMBOL_MAX_RULES` (default 2 or 3) rules:
* **Round 1**: Evaluates every rule in the Phase 2 pool on the symbol's training and validation datasets. Selects the rule with the highest **Robust Return**:
  $$\text{Robust Return} = \min(\text{train\_return\_pct}, \text{val\_return\_pct})$$
* **Round 2 & 3**: Evaluates combinations of the selected rule(s) extended by other pool rules. Candidates are chosen using the top-K pool rules (`PHASE3_PER_SYMBOL_GREEDY_TOP_K`), optimizing the joint robust return.

### Key Constraints & Penalties

#### 1. Robust Return & Overfit Gate
To prevent rules that are lucky only on the validation split from being selected:
* The primary score is $\min(\text{train\_return}, \text{val\_return})$.
* **Gap Rejection**: If a rule's validation return exceeds its training return by more than `PHASE3_MAX_TRAIN_VAL_GAP_PCT` (default 40.0%), it is hard-rejected:
  $$\text{val\_return} - \text{train\_return} > 40\% \implies \text{Score} = -999.0$$

#### 2. Redundancy & Orthogonality Penalties
When evaluating multi-rule combinations, soft penalties are added to prevent overlapping entries:
* **Trade Jaccard Similarity**: Penalizes combinations where rules fire on the same timestamps:
  $$\text{Jaccard} = \frac{|M(R_i) \cap M(R_j)|}{|M(R_i) \cup M(R_j)|}$$
  If Jaccard exceeds `PHASE3_JACCARD_SIMILARITY_GATE`, a penalty is applied.
* **Incremental Trade Gate**: Ensures subsequent rules add a minimum number of new trades (`PHASE3_MIN_INCREMENTAL_TRADES`) not already triggered by preceding rules.

### Rule Merging & Condition Insertion
Once rules are selected per symbol, they are merged. Since rules are represented globally in the final strategy, symbols that share the same underlying logic are consolidated:
1. If Rule $R$ is selected for symbol `BTCUSDT` and `ETHUSDT`, the parser appends `symbol is BTCUSDT` and `symbol is ETHUSDT` as OR-like constraints.
2. The final condition set looks like:
   `["condition_1", "condition_2", "symbol is BTCUSDT", "symbol is ETHUSDT"]`
3. This format is fully compatible with `evaluator_v4.ipynb`.

### Fallback Mechanism
If the per-symbol greedy selection yields no rules across any symbol, or if the merged set is smaller than `PHASE3_GLOBAL_MIN_RULES` (default 2), the selector triggers a **Global Pool Fallback**:
* It ranks the entire Phase 2 pool by global robust return.
* It selects the top unique rules up to the minimum global rule requirement.
* It validates the fallback set against validation return floors before saving.
