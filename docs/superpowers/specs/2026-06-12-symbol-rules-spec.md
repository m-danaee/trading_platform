# Spec: Phase 2 and Phase 3 Symbol Rule Integration Documentation

* **Date**: 2026-06-12
* **Topic**: Documenting Two-Stage Phase 2 Evolution and Per-Symbol Phase 3 Greedy Selection in `GOOD_IDEAS.md`.

## 1. Context & Scope
The user requested a document in `GOOD_IDEAS.md` detailing the changes made in Phase 2 and Phase 3 regarding symbol rules.
This design spec covers the addition of `GOOD_IDEAS.md` in the root of the workspace to act as a centralized technical guide for:
- Two-Stage Phase 2 evolution (Stage A exploration vs Stage B refinement, memory management).
- Per-symbol greedy selection and merging using `symbol is X` conditions in Phase 3.

## 2. Location
- Documentation file: [GOOD_IDEAS.md](file:///home/danaee/trading_platform/GOOD_IDEAS.md)

## 3. Key Concepts Covered in Spec
- **Phase 2 Stage A vs Stage B**: Budget split, warm start, seed top-K, and historical seeds.
- **Phase 2 Resource Management**: Host RAM/VRAM optimization (garbage collection, JAX cache clearing) and training dataframe caching.
- **Phase 3 Per-Symbol Greedy Search**: Round-by-round selection, robust scoring using `min(train_return, val_return)`.
- **Phase 3 Redundancy Gates**: Jaccard similarity and incremental trade thresholds.
- **Phase 3 Merging & Fallback**: Appending `symbol is X` constraints to rule conditions and falling back to a global pool ranking if rules are insufficient.

## 4. Verification
- Validate that the file [GOOD_IDEAS.md](file:///home/danaee/trading_platform/GOOD_IDEAS.md) exists and has correct formatting.
- Ensure all config keys and JAX optimizations match actual implementations in [gpu_fuzzy_trader/config.py](file:///home/danaee/trading_platform/gpu_fuzzy_trader/config.py), [gpu_fuzzy_trader/phases/phase2_rule_pool.py](file:///home/danaee/trading_platform/gpu_fuzzy_trader/phases/phase2_rule_pool.py), and [gpu_fuzzy_trader/phases/phase3_rule_set.py](file:///home/danaee/trading_platform/gpu_fuzzy_trader/phases/phase3_rule_set.py).
