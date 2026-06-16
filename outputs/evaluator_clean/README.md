# `outputs/evaluator_clean/` — Stripped strategy JSONs for strict evaluators

This folder holds **defensive stripped copies** of the strategy JSON files
(`long.json` / `short.json`).  Each clean file contains **only** the two
keys that `evaluator_v5.ipynb` requires: `direction` and `rules_set`.  All
metadata (`risk_optimized`, `selection_accepted`,
`selection_rejection_reason`, `validation_gate`, etc.) is stripped.

## What is here

| File | Content |
|---|---|
| `long_evaluator_clean.json` | Clean copy of `outputs/long.json` |
| `short_evaluator_clean.json` | Clean copy of `outputs/short.json` |

Each is a JSON object with exactly two top-level keys:

```json
{
  "direction": "long",
  "rules_set": [
    {
      "conditions": ["[feature] IS Value", "symbol is X"],
      "tp": 4.0,
      "sl": 2.0,
      "capital_pct": 50.0
    }
  ]
}
```

If no rules were selected, `rules_set` is an empty array `[]`.

## When is it written

The clean file is written every time a strategy JSON is persisted to
`outputs/{direction}.json`.  This happens at the end of:

- **Phase 3** (`Rule_Set_Selector._persist_output`)
- **Phase 4** (`phase4_wf_optimizer` — risk-optimized version)
- **Phase 5** (`phase5_oos` — final accepted version)

The writer is `_maybe_write_evaluator_clean()` in
`gpu_fuzzy_trader/output/writer.py`.  It is guarded by the config flag
`WRITE_EVALUATOR_CLEAN` (default `True`).

## Why does it exist

A future version of `evaluator_v5.ipynb` (or a hidden-test pipeline) might
reject strategy JSONs that contain unexpected top-level keys.  The metadata
keys the pipeline adds (`risk_optimized`, `selection_accepted`, etc.) are
helpful for debugging but are not part of the evaluator's expected schema.

By writing a stripped copy to `evaluator_clean/` we ensure that a stricter
evaluator always has a **schema-compliant** file to read, even if the main
`outputs/long.json` has grown extra keys over time.

## How to read it

- **`rules_set: []`** (empty array): Phase 3 rejected all rules.  Check the
  corresponding **metadata JSON** (`outputs/long.json` or
  `outputs/short.json`) for the `selection_rejection_reason` field to
  understand why.

- **`rules_set` has ≥ 2 rules**: The pipeline produced a valid strategy.
  The clean file is equivalent to the main file's `direction` and
  `rules_set` — no information is lost.

- **File missing**: The pipeline has not been run yet, or
  `WRITE_EVALUATOR_CLEAN` is `False` in the config.
