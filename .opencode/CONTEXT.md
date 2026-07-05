# Nexus Context

**Updated:** 2026-07-05
**base_branch:** `main`
**branch_policy:** isolated
**execution_mode:** checkpoint
**status:** active — new plan: Phase 2 RAM optimization (avoid Colab OOM)

## Active Plan
**Plan: Phase 2 RAM Optimization (avoid Colab OOM)**
**Source:** Pre-fix re-run log showed 9.9 / 12.7 GB RAM at gen 6; OOM risk imminent
**PLAN:** `.opencode/plans/PLAN.md`

| Task | Title | Branch | Status |
|------|-------|--------|--------|
| 4 | RAM quick wins (cache + teardown + gc) | `fix/ram-quick-wins` | ⏳ Pending |
| 5 | Sequential cluster warmup (drop 3/4 signatures) | `fix/ram-sequential-clusters` | ⏳ Pending |

## Previous plan (completed)
**Plan: Fix Phase 2 overfit-gap blind spots + confirmed bugs (Stages 1-3)**
- ✅ Stages 1-3 merged to main (8 commits, ahead of `origin/main` by 8)
- ✅ All 3 feature branches deleted
- ✅ Post-fix re-run (2026-07-05 22:09) confirms Stages 1-3 working:
  - SPLIT_MODE: "holdout 65/35" ✓
  - corr_f1_f3: WARNING level ✓
  - max_train_val_gap bounded at 12.70% (was 86.46%) ✓
  - max_train_val_gap_ratio bounded at 1.57-5.50x (was 7.77x frozen) ✓

## RAM Analysis (from 2026-07-05 22:09 log)
- Peak RAM: 9.9 / 12.7 GB at gen 6
- Top consumers: 4× JAX compiled programs (~3 GB), 3× cluster DataFrames (~1.5 GB), global metrics cache (~0.8 GB)
- Plan targets ~6-7 GB peak (1.5-3.5 GB savings)

## Hard rules
- `PYTEST_LOW_MEMORY=1` for any test run
- Only run touched test suites (OOM risk on full suite)
- Use `.venv/bin/python`
- Per AGENTS.md: do not run the project (OOM risk) — Colab GPU only
- `evaluator_v5.ipynb` is the source of truth for rule testing (do not change)

## Branch cleanup
3 feature branches from previous plan are deleted.
