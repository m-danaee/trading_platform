# Design: Short Top-100 TP/SL Optimize on train_2_full

**Date:** 2026-07-16  
**Status:** Approved  
**Input strategy:** `optimized_short/short_top100.json`  
**Dataset:** `data/train_2_full.csv`  
**Evaluator semantics:** `evaluator_v5.ipynb` (read-only)

## Goals

1. Optimize **TP/SL only** for the existing 100 short rules.
2. Keep **rule order**, **symbol pins**, **feature conditions**, and **capital_pct** fixed.
3. Two-stage search (approved **B**):
   - Stage 1: solo TP/SL grid per rule
   - Stage 2: portfolio accept — keep a solo best only if the full 100-rule book improves
4. Write new files under `optimized_short/`; do not edit `short.json` or `short_top100.json`.

## Non-goals

- No reordering, symbol re-pinning, capital_pct changes, or rule drops
- No edits to `evaluator_v5.ipynb` or the GPU pipeline

## Grids and constraints

- TP: `1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0`
- SL: `1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0`
- Require `TP / SL >= 1.0`
- Always evaluate the rule’s current (seed) TP/SL as a candidate

## Scoring

**Solo (Stage 1):** short label outcomes (evaluator_v5 short TP/SL/time logic) minus `FEE_PCT` (0.20), ranked with the same standalone score helper used by `optimize_long_rules.score_standalone`.

**Portfolio (Stage 2):** fast capital-managed first-match simulator (`optimize_long_rules.simulate_portfolio`) with **short** price returns. Accept a per-rule TP/SL change iff portfolio lexicographic key improves:

`(score, total_return_pct, profit_factor, -max_drawdown_pct, executed_trades)`

One forward pass over rules in current JSON order.

## Outputs

- `optimized_short/short_top100_tpsl.json`
- `optimized_short/tpsl_optimize_report.json`
- Verification: evaluator_v5 on `train_2_full` → metrics + equity under `optimized_short/`

## Failure modes

| Case                       | Behavior                                      |
| -------------------------- | --------------------------------------------- |
| Rule has zero matches      | Keep seed TP/SL                               |
| Portfolio never improves   | Output equals input TP/SL (still write files) |
| Missing dataset / strategy | Abort with clear error                        |
