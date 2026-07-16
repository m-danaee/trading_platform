# Design: Short Top-100 Reorder + Symbol Pins

**Date:** 2026-07-16  
**Status:** Approved (design dialogue); awaiting spec review before implementation  
**Inputs:** `short.json` (242 SHORT rules), `data/train_2.csv`  
**Evaluator:** `evaluator_v5.ipynb` (do not modify)  
**Outputs:** `optimized_short/` (leave `short.json` untouched)

## Problem

In `evaluator_v5`, when multiple rules fire on the same bar, the **first matching rule in `rules_set` wins**. Simultaneous entries also allocate capital by JSON rule order. The current `short.json` is unordered relative to solo quality, and many rules fire on symbols where they are weak.

## Goals

1. Keep exactly **100** rules (or fewer if gates leave fewer than 100 survivors).
2. **Reorder** so highest-quality rules sit at the top of `rules_set`.
3. **Pin** each kept rule to symbols where it performs well on `train_2.csv`.
4. Score with solo backtests (not the combined 242 portfolio), using  
   **score = Net_PnL × profit_factor**.
5. Write results under `optimized_short/`; never edit `short.json`.

## Non-goals

- No changes to TP / SL / `capital_pct` / feature conditions (except symbol pin).
- No cluster pinning, test-set gating, or train/test gap optimization in this pass.
- No edits to `evaluator_v5.ipynb` or the full trading pipeline.

## Ranking and gates (locked)

| Item                 | Choice                                                     |
| -------------------- | ---------------------------------------------------------- |
| Solo metric          | `score = net_pnl * profit_factor`                          |
| Symbol keep gate     | `net_pnl > 0` AND `profit_factor >= 1.2` AND `trades >= 5` |
| Rule keep            | At least one symbol passes the gate after pinning          |
| Rule aggregate score | Sum of per-symbol scores over kept symbols                 |
| Selection            | Sort rules by aggregate score descending; take top 100     |
| Output order         | Same sort order (best first)                               |

**Profit-factor edge cases**

- If a symbol has wins and zero losses, treat `profit_factor` as a finite cap of **10.0** for scoring (still require `net_pnl > 0` and `trades >= 5` for the gate).
- If a symbol has zero trades, it fails the gate.

## Symbol pinning

- Evaluator format: `"symbol is 1,2,3"` (comma-separated values are OR-ed).
- Strip any existing `symbol is ...` / `[symbol] IS ...` conditions from the rule, then append one pin for the kept symbol set (sorted ascending as integers for stability).
- Feature conditions stay unchanged.

## Pipeline

```text
short.json (242)
    │
    ▼
Load evaluator_v5 helpers + train_2.csv once
    │
    ▼
For each rule i = 1..242:
  solo backtest on train_2
  group trades by Symbol
  keep symbols that pass gate B
  if none → drop rule
  else pin + compute aggregate score
    │
    ▼
Sort survivors by aggregate score DESC
Keep top 100
    │
    ▼
Write optimized_short/short_top100.json
     + optimized_short/optimize_report.json
```

### Efficiency requirement

Do **not** reload CSV / rebuild the fuzzy engine 242 times.

1. Load `train_2.csv` and build schema/modes once (safe27 schema from `train_2` itself).
2. Build `AdaptiveFuzzyEngine` once.
3. For each rule, convert to a one-rule strategy and run `CapitalManagedTradeSimulator` (or `evaluate_student_strategy_on_dataset` with a one-rule dict) against the shared engine/data.
4. Free per-rule trade logs after aggregating metrics; keep only summary rows.

If memory pressure appears on WSL, process rules in batches but still reuse the loaded frame and fuzzy engine.

## Output files

### `optimized_short/short_top100.json`

Deployable strategy only:

```json
{
  "direction": "short",
  "rules_set": [
    /* up to 100 pinned rules, best first */
  ]
}
```

No large optimization metadata inside this file (keep it evaluator-compatible).

### `optimized_short/optimize_report.json`

Machine-readable audit trail:

- gates and scoring formula
- dataset path and row counts
- per-rule: original 1-based index, kept symbols, per-symbol metrics, aggregate score, kept_rank (1..100 or null if dropped)
- counts: evaluated / dropped_no_symbol / kept_after_rank
- optional: wall time

### Optional

- `optimized_short/README.md` — one short recipe paragraph pointing at this spec.

## Verification

After writing outputs, run `evaluator_v5` once on `train_2.csv` with `short_top100.json` and record:

- total return %, max drawdown %, profit factor, win rate, executed trades, status

Compare briefly to the prior full-242 `train_2` baseline if available (from earlier student evaluations); do not require re-running the 242-rule baseline in this task unless cheap.

Do **not** run the full GPU pipeline (OOM risk per `AGENTS.md`). Use `.venv`.

## Failure modes

| Case                                   | Behavior                                      |
| -------------------------------------- | --------------------------------------------- |
| Fewer than 100 rules pass symbol gates | Keep all survivors, still sorted by score     |
| Rule already has symbol filter         | Replace with new pin from gate B              |
| Solo eval throws for one rule          | Log error in report, skip that rule, continue |
| `train_2.csv` missing                  | Abort with clear error                        |

## Implementation touchpoints

- New script: `scripts/optimize_short_top100.py` (or equivalent under `scripts/`)
- Read-only: `short.json`, `evaluator_v5.ipynb`, `data/train_2.csv`
- Write-only: `optimized_short/*`
