# Nexus Context — OOS Generalization Fixes (27-item plan)

**Updated:** 2026-06-28
**base_branch:** `main`
**branch_policy:** isolated
**execution_mode:** checkpoint
**plan:** `.opencode/plans/PLAN.md` (tasks 14–19)

## Active Objective

Fix the out-of-sample equity collapse diagnosed from the 2026-06-28 run log.
Pipeline summary: LONG test=−4.45%, SHORT test=−0.00%. Validation overfitting
(val Sortino 1.20 > train 0.65; val→test Δ=−17.6%) compounded by symbol-locked
rules, degenerate win-rate objective, disabled migration, and premature
convergence. 27 fixes across 6 tasks.

## Pre-flight Blocker

Uncommitted changes on `main`:
- `gpu_fuzzy_trader/config.py` — 3 tuning lines (MIN_TRADE_SUPPORT 75→120,
  MIN_TRADE_POOL_FLOOR 25→40, PHASE2_ORPHAN_MIN_TRADE_SUPPORT 15→20).
  These are legitimate task-8 tuning remnants. **Commit to `main` before
  branching task-14.**
- `outputs/**` — pipeline output artifacts (reports, JSONs, PNGs). Safe to
  commit or `.gitignore` — they are run outputs, not source.

## Task Summary (6 tasks, sequential, isolated branches)

| # | Branch | Fixes | Priority | Status |
|---|--------|-------|----------|--------|
| 14 | `fix/rb-governor-rebalance` | C1,C2,C3,C4,M7 — val overfit, CV folds, gap penalty | 🔴 Critical | pending |
| 15 | `fix/fitness-objective-redesign` | C5,C6,C7,H1,H2,M3 — symbol-lock, f1/f3, Sortino sat, val leak | 🔴 Critical | pending |
| 16 | `fix/evolution-convergence` | H3,H5,M4,M5 — restart, state carry-over, normalization | 🟠 High | pending |
| 17 | `fix/island-migration-rule-structure` | H4,H6 — migration enable, MIN/MAX_CONDITIONS | 🟠 High | pending |
| 18 | `fix/admission-gates-robustness` | M1,M2,M6,M8 — per-symbol WR bug, cache, monthly gate | 🟡 Medium | pending |
| 19 | `fix/cleanup-observability` | L1–L6 — dead code, EvoX warn, Das-Dennis, viability trigger | 🟢 Low | pending |

## Execution Order

```
task-14 (RB Governor)  ──► task-15 (Fitness) ──► task-16 (Evolution)
                                                  │
task-17 (Migration)  ◄─────────────────────────────┤
                                                  │
task-18 (Admission) ◄─────────────────────────────┤
                                                  ▼
task-19 (Cleanup)  ◄──────────────────────────────┘
```

- task-14 first: highest leverage, config+scoring only, no evolution code.
- task-15 second: touches `compute_phase2_objectives_from_metrics` — must land
  before task-16 to avoid merge conflicts in the same function.
- task-16/17/18 can run in parallel after 15 (different files mostly).
- task-19 last (pure cleanup, no behavioral risk).

## Hard Rules (from AGENTS.md)

- Always use `.venv` for running commands.
- Run tests with `PYTEST_LOW_MEMORY=1` (OOM risk on local/WSSL).
- Do NOT run the full pipeline locally (runs on Colab GPU).
- Do NOT modify `evaluator_v5.ipynb`.
- After changing code, remove wasted/old implementation to keep project clean.

## Verification

- Unit tests: `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/ -x -q`
- OOS validation: run pipeline on Colab GPU, compare test_long/short_report.json
  return vs baseline (−4.45% / −0.00%).
- Do NOT run `run_pipeline.py` locally (OOM).
