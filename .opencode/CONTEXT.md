# Nexus Context

**Updated:** 2026-07-04
**base_branch:** `main`
**branch_policy:** isolated
**execution_mode:** checkpoint
**status:** COMPLETED — Both tasks merged to main

## Completed Plan
**Plan: Switch to Holdout 70/30 + Embargo, Remove Purged CV** (`.opencode/plans/PLAN.md`)

| Task | Branch | Commit | Status |
|------|--------|--------|--------|
| Task 1: Holdout+Embargo split | `fix/holdout-embargo-split` | `bdfbeec` | ✅ Merged |
| Task 2: GPU PnL defer to release | `fix/gpu-pnl-defer-to-release` | `b3966fd` | ✅ Merged |

### Merge flow
```
main: 77ac162 → df905e8 (Task 1 fast-forward) → 2d4c002 (Task 2 merge)
```

### What changed
- **Holdout 65/35 + 288-bar embargo** — replaces purged CV. Safe range: ~119k bars/sym (was 72k). Random start enabled.
- **GPU engine PnL timing** — defers PnL from entry to release. Matches CPU engine. 953% max_return eliminated.
- **GPU/CPU parity** — returns within ±1%, win/profit/trades exact, Sortino ±5%, DD ±5%
- **Tests**: 1312 passed, 0 regressions

### Feature Branches
- `fix/holdout-embargo-split` — can be deleted
- `fix/gpu-pnl-defer-to-release` — can be deleted

## Next Action
- Run Colab pipeline to verify the fixes in production
