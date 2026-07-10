# Nexus Context

Active objective: Phase 2 feasible-search fixes (items 1–4) — COMPLETE.

## Status: MERGED to main

### Changes (branch feature/phase2-feasible-search-1-4)
| Item | Change |
|------|--------|
| 1 | Evolution feasibility PF = 1.05; admission stays 1.15 |
| 2 | Island min_profitable = max(2, ceil-half) → 3-sym gets 2 |
| 3 | Corr hybrid clustering ON (0.3 feat / 0.7 corr), weights preserved |
| 4 | PHASE2_VAL_IN_FITNESS_PENALTY = False |

### Commits
- 9b35e41 feat: Phase 2 feasible-search fixes 1–4
- 375b00c fix: enable corr clustering by default
- 8c7a26e test: expect hybrid_corr_v1
- 72d043a fix: preserve blend weights

### Next
User re-runs Phase 2 on Colab. Expect: higher valid_rules, lower train_pf_floor counts in collapse log, method=hybrid_corr_v1 in symbol_clusters.json.
