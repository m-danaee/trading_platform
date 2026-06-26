# Task 4 — `fix/cluster-balancing` (Fix E)

## Branch
`fix/cluster-balancing` (from latest `main`, after task-3 merge).

## Problem
K-means over feature profiles produced a badly imbalanced split:
`{'0':['9'], '1':['1','2','6'], '2':['10','3','4','5','7','8']}`. The lone
1-symbol cluster overfits (hits 11.79% on train, unlikely to generalize), while
the 6-symbol cluster is starved by `min_profitable_symbols=3` → `deployable=0`
most epochs, so its (more generalizable) rules get filtered out.

## Required Changes

### Balanced clustering
**File:** `gpu_fuzzy_trader/features/symbol_cluster.py`
(`build_hybrid_symbol_clusters`, lines 58-122)

Current: plain `KMeans(n_clusters=k, random_state=seed).fit(profiles)` then
assign each symbol to its label — no size constraint.

Change to a balanced assignment so clusters are roughly equal in symbol count:
- With 10 symbols and K=3, target ~3/3/4 (never a 1-symbol primary cluster).
- Approach options (pick the simplest robust one):
  (a) Sort symbols by their predicted cluster, then round-robin assign into K
      bins ordered by centroid distance (greedy balanced assignment).
  (b) Use a constrained assignment: for each symbol (in random order), assign to
      the nearest cluster that hasn't reached `ceil(n_symbols / k)` capacity.
- Keep the existing fallback for `n_symbols <= k` (one symbol per cluster) and
  `n_clusters == 1` paths unchanged.
- Add a `min_symbols_per_cluster` guard (default 2): if any cluster would have
  < 2 symbols and `n_symbols >= 2*k`, rebalance. (A genuine 1-symbol cluster is
  handled by the orphan-boost path, not a primary island.)
- Log the balanced assignment the same way (`symbol_cluster: K=%d assignment=%s`).

### Relax large-cluster admission gates
**File:** `gpu_fuzzy_trader/config.py` — island hyperparams
(`resolve_island_hyperparams`, search for `min_profitable_symbols`).

Current: `min_profitable_symbols=3` for the large cluster. With a balanced ~3-4
symbol cluster this is too strict (requires 75-100% of symbols profitable).

Change: scale `min_profitable_symbols` with cluster size, e.g.
`max(1, round(n_symbols * 0.5))` so a 3-symbol cluster needs 2, a 4-symbol
cluster needs 2. Apply this in `resolve_island_hyperparams` for the `cluster`
profile (do not change `orphan` profile — it's already 1 symbol).

### Route lone clusters to orphan-boost
If after balancing a cluster still ends up with exactly 1 symbol (only possible
when `n_symbols < 2*k`), route it through the orphan-boost path instead of a
primary island. This is a small change in `_run_cluster_islands`
(`phase2_island_scheduler.py`) — but ONLY implement if balancing still leaves a
1-symbol cluster; otherwise skip (balancing should prevent it).

## Acceptance Criteria
1. With 10 symbols and K=3, `build_hybrid_symbol_clusters` returns clusters with
   no cluster smaller than 3 symbols (i.e. 3/3/4 or similar). Regression test
   with deterministic seed.
2. `min_profitable_symbols` scales with cluster size (test: a 3-symbol cluster
   resolves to 2, a 6-symbol cluster resolves to 3).
3. Existing single-symbol fallback (`n_symbols <= k`) still works.
4. `SYMBOL_CLUSTERS_PATH` payload shape unchanged (only assignment differs).
5. `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit -q` passes.

## Target Files
- `gpu_fuzzy_trader/features/symbol_cluster.py`
- `gpu_fuzzy_trader/config.py` (`resolve_island_hyperparams`)
- `gpu_fuzzy_trader/phases/phase2_island_scheduler.py` (only if lone-cluster
  routing is needed)
- `README.md` (config table)
- `tests/unit/test_phase2_island_hyperparams.py` (extend) and/or
  `tests/unit/test_phase2_island_scheduler.py` (extend); add a cluster-balance
  test if none exists.

## Verification
```
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit -q
```
Do NOT run the full pipeline.

## Notes
- Keep the `build_hybrid_symbol_clusters` signature stable (add optional
  `balance: bool = True` kwarg if cleaner, defaulting True).
- Do not change `PHASE2_N_CLUSTERS` (stays 3).
- Clean up dead code after changes (per AGENTS.md).
