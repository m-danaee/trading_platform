# Task 5 — Expand Phase 2 pool admission

## Why
My current Phase 2 produces only 5 long + 8 short rules in the
final pool (run log shows "pool=5, saved to ...phase2_long_pool.json"
and "pool=8, saved to ...phase2_short_pool.json"). This is the
*post-archive* size after `_build_pool_from_archive` filters by the
20% gap-reject threshold. The friend keeps ~140 candidate rules
after filtering, which gives Phase 3 a much larger and more diverse
pool to choose from. With only 5-8 rules, Phase 3's per-symbol
greedy has nothing to combine, which is why my final output is just
2 rules per direction with thin coverage.

The friend does this through:
1. Relaxed pool-admission thresholds (`PHASE2_CV_MIN_WORST_RETURN=-8%`, `PHASE2_CV_MAX_WORST_DD=18%`, `PHASE2_CV_MIN_WORST_PF=0.80`).
2. Turning on `PHASE2_STRICT_POSITIVE_GOOD` (the optional flag added in Task 3) as a *secondary* check, not a hard reject.
3. `PHASE2_KEEP_TOP_RULES=140` cap to bound the pool.
4. `PHASE2_ARCHIVE_MAX_SIZE=500` (mine is already 500; no change needed).

## Required reading
- `.opencode/plans/PLAN.md`
- `.opencode/CONTEXT.md` (JSON output contract)
- The friend's reference: `friend_project/gpu_fuzzy_trader/config.py` lines 145-152 (`PHASE2_CV_*` keys) and `PHASE2_KEEP_TOP_RULES=140` line 269. Also `_filter_good_rules` in `friend_project/gpu_fuzzy_trader/rb_governor.py` lines 535-578.
- My existing `gpu_fuzzy_trader/phases/phase2_support.py` (`_passes_pool_admission_impl`, `_raw_feasibility_violation_score`, `deployability_rank_score`).
- My existing `gpu_fuzzy_trader/phases/phase2_rule_pool.py` (`_build_pool_from_archive` — this is where the final pool size is set).

## Behavior changes

### Step 1 — Add new pool-admission config keys

```python
# Phase 2 pool admission thresholds (relaxed from the 20% gap-reject)
PHASE2_CV_MIN_WORST_RETURN = -8.0
PHASE2_CV_MIN_WORST_PF = 0.80
PHASE2_CV_MAX_WORST_DD = 18.0
PHASE2_CV_MIN_FOLD_TRADES = 10

# Pool size cap (friend's value)
PHASE2_KEEP_TOP_RULES = 140

# Optional: enable the Task 3 positive-good gate for Phase 2 pool admission
# Default OFF; will turn on in this task to demonstrate the larger pool size.
PHASE2_STRICT_POSITIVE_GOOD = True
```

### Step 2 — Use the new thresholds in `_passes_pool_admission_impl`

In `phases/phase2_support.py`, modify the per-fold pool admission to
use the new `PHASE2_CV_MIN_WORST_RETURN`, `PHASE2_CV_MIN_WORST_PF`,
`PHASE2_CV_MAX_WORST_DD` thresholds INSTEAD of the strict
`PHASE2_MAX_TRAIN_VAL_GAP_PCT` 20% gap-reject.

The friend does this by computing the *worst* metrics across CV folds
and applying these floors:
- worst_return >= PHASE2_CV_MIN_WORST_RETURN (-8%)
- worst_pf >= PHASE2_CV_MIN_WORST_PF (0.80)
- worst_dd <= PHASE2_CV_MAX_WORST_DD (18%)
- min_fold_trades >= PHASE2_CV_MIN_FOLD_TRADES (10)

Note: this is the *per-fold* admission. The aggregated `passes_pool_admission`
function (used for the final `deployable` flag) keeps the stricter criteria.

### Step 3 — Apply `PHASE2_KEEP_TOP_RULES` cap in `_build_pool_from_archive`

After the per-rule admission filter, sort rules by `deployability_rank_score`
(or equivalent) and keep the top `PHASE2_KEEP_TOP_RULES` (140). Currently
mine keeps all admitted rules.

### Step 4 — Turn on `PHASE2_STRICT_POSITIVE_GOOD`

In `config.py`, set `PHASE2_STRICT_POSITIVE_GOOD = True` (the Task 3
flag). The Task 3 implementation in `phases/phase2_support.py` already
handles the gate. Confirm it works and does not crash.

### Step 5 — Verify the pool size on a small synthetic run

Run a small Phase 2 dry-run (e.g., 50 generations, 50 population) on
a 10k-row synthetic DataFrame and confirm the final pool has ≥ 30
rules. This is an integration test, not a unit test.

## Out of scope
- Do NOT change the JSON output format.
- Do NOT modify `evaluator_v5.ipynb`.
- Do NOT touch the GPU engine or EvoX runner.
- Do NOT change the per-symbol greedy logic in Phase 3.
- Do NOT change the risk optimization in Phase 4.
- Do NOT add Tasks 6-9 features.

## Acceptance criteria
1. All 6 new config keys (`PHASE2_CV_MIN_WORST_RETURN`, `PHASE2_CV_MIN_WORST_PF`, `PHASE2_CV_MAX_WORST_DD`, `PHASE2_CV_MIN_FOLD_TRADES`, `PHASE2_KEEP_TOP_RULES`, `PHASE2_STRICT_POSITIVE_GOOD=True`) are present and accessible.
2. `_passes_pool_admission_impl` uses the new thresholds; the old `PHASE2_MAX_TRAIN_VAL_GAP_PCT=40.0` is no longer the only path to rejection.
3. `_build_pool_from_archive` caps the pool at `PHASE2_KEEP_TOP_RULES` (140 by default).
4. A new unit test `tests/unit/test_phase2_pool_admission.py` exercises the new thresholds:
   - A rule with `worst_return=-5%`, `worst_pf=0.85`, `worst_dd=15%` passes.
   - A rule with `worst_return=-15%` fails.
   - A rule with `worst_dd=25%` fails.
   - A rule with `worst_pf=0.50` fails.
5. **(OPTIONAL — skipped per RAM constraint)** A small integration test runs a 50-gen Phase 2 on a 10k-row synthetic DataFrame and confirms the final pool has ≥ 30 rules. The user can verify this on a real run later; this is a smoke test, not a hard acceptance criterion.
6. All existing tests pass.
7. No changes to `evaluator_v5.ipynb`, the GPU engine, or the JSON output contract.

## Constraints
- Stay on `feature/task-5-expand-phase2-pool` (off `main` after task-4 is merged).
- 12.7 GiB RAM total.
- PEP 8, type hints, module logger.
- Use only existing third-party deps.

## Files I will touch
- `gpu_fuzzy_trader/config.py` — 5 new `PHASE2_CV_*` keys + `PHASE2_KEEP_TOP_RULES` + flip `PHASE2_STRICT_POSITIVE_GOOD` to True
- `gpu_fuzzy_trader/phases/phase2_support.py` — modify `_passes_pool_admission_impl` to use the new thresholds
- `gpu_fuzzy_trader/phases/phase2_rule_pool.py` — modify `_build_pool_from_archive` to cap at `PHASE2_KEEP_TOP_RULES`
- `tests/unit/test_phase2_pool_admission.py` (new) — ≥ 4 cases
