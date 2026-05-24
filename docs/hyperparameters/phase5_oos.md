# Phase 5 — Out-of-Sample Evaluation

**Module:** [`gpu_fuzzy_trader/phases/phase5_oos.py`](../../gpu_fuzzy_trader/phases/phase5_oos.py)  
**Config:** No phase-specific tunables — inherits [Phase 0 shared constants](phase0_shared.md)

[← Phase 4](phase4_rl_risk.md) | [Index](README.md)

---

## Purpose and statistical framing

Phase 5 is the **only phase that touches `data/test.csv`**. It loads the final strategies from Phase 4 (`outputs/long.json`, `outputs/short.json`) and reports performance on truly held-out data.

There is nothing to tune in Phase 5 itself. This document explains **inherited constants**, **reported metrics**, and **how to interpret** train → validation → test degradation as a data scientist.

---

## Workflow

1. Load `outputs/long.json` and/or `outputs/short.json` via `Output_Writer.load_and_validate()`.
2. Prepare test data with the **same pipeline** as training:
   - Sort by `(symbol, datetime)`
   - Drop last `TAIL_DROP_ROWS` (288) per symbol
   - Drop NaN label rows
   - Fill feature NaN with 0
   - Compute `_symbol_bar_index`
3. Evaluate on **train**, **validation**, and **test** splits using `CPUBacktestEngine` (canonical semantics).
4. Write reports to `outputs/reports/`.

Phase 5 always runs — even when Phases 1–4 are skipped via `--resume`.

---

## Inherited hyperparameters (from config)

All simulation uses [Phase 0 backtest constants](phase0_shared.md):

| Constant | Default | OOS impact |
|----------|---------|------------|
| `INITIAL_CAPITAL` | `1000.0` | Scales reported PnL; not Sortino |
| `LEVERAGE` | `1.0` | Position sizing multiplier |
| `FEE_PCT` | `0.20` | Direct drag on test return |
| `MAX_HOLD_CANDLES` | `288` | Exit horizon |
| `MAX_TOTAL_EXPOSURE_PCT` | `100.0` | Portfolio cap |
| `MIN_POSITION_NOTIONAL` | `1.0` | Dust filter |

Per-rule **TP, SL, capital_pct** come from Phase 4 outputs — not from `PHASE2_*` static values.

Data paths:

| Constant | Role in Phase 5 |
|----------|-----------------|
| `TEST_CSV_PATH` | Primary OOS dataset |
| `TRAIN_CSV_PATH` | Re-split for train/val comparison reports |
| `REPORTS_DIR` | Output location |

---

## Reported metrics

| Metric | Interpretation | Red flags |
|--------|----------------|-----------|
| **Total return %** | Net PnL / initial capital | High train, negative test → overfit |
| **Sortino ratio** | Downside-risk-adjusted return | Unstable when trade count < ~30 |
| **Max drawdown %** | Peak-to-trough equity | Test DD ≫ val DD → tail risk not captured |
| **Win rate %** | Fraction winning trades | High WR + low return → small wins, large losses |
| **Profit factor** | Gross profit / gross loss | < 1.0 → losing strategy |
| **Executed trades** | Sample size for inference | Zero trades → metrics default to 0% return |
| **Account status** | survived / ruined | Ruin only if equity hit zero |

### Per-symbol breakdown

`outputs/reports/test_per_symbol_performance.csv` includes trade count, win rate, and net PnL per symbol. Use this to detect **concentration** — strong aggregate test Sortino driven by one symbol is fragile.

---

## Zero-trade and ruin semantics

- **Zero trades:** Reports **0% total return**. Does **not** mark account as ruined unless equity actually reached zero.
- **Ruin:** Only when simulation equity hits zero under backtest rules.

This matches `evaluator_v3.ipynb` semantics and avoids false "ruined" flags on inactive strategies.

---

## Interpreting train → val → test curves

| Pattern | Likely diagnosis | Upstream knobs |
|---------|------------------|----------------|
| Train ≫ val ≫ test | Progressive overfit | Phase 3 gates, Phase 2 joint objective, reduce search budget exploitation |
| Val ≈ test ≪ train | Validation representative but overfit to train | Phase 3 train-target design; tighten gates |
| Test > val | Possible favorable test regime or small sample luck | Check test period length and symbol mix |
| Similar Sortino, lower test return | Fee/hold effects or fewer trades | Check trade counts per split |

**Do not tune Phase 5 or test split** to improve numbers — adjust upstream phases and re-run pipeline.

---

## Diagnostics

| Artifact | Content |
|----------|---------|
| `outputs/reports/test_long_report.json` | Aggregate test metrics (long) |
| `outputs/reports/test_short_report.json` | Aggregate test metrics (short) |
| `outputs/reports/test_per_symbol_performance.csv` | Per-symbol test breakdown |
| `outputs/reports/test_*_equity.png` | Test equity curves |
| Cross-split reports | Train/val/test comparison in same run |

Compare with Phase 3 validation equity plots to see whether test degradation was predictable from val gates.

---

## Code reference

Phase 5 uses the same loader/splitter preparation as training and evaluates via CPU engine only (GPU not required for final OOS truth).

```1:24:gpu_fuzzy_trader/phases/phase5_oos.py
"""
phase5_oos.py — OOS_Evaluator (Phase 5)

Final out-of-sample evaluation on the held-out test.csv.
...
  6. Save outputs in outputs/reports/ including the existing test JSON/CSV files
      plus the new cross-split reporting artifacts
"""
```

---

## Related documentation

- [Index — failure mode quick reference](README.md#failure-mode--knob-quick-reference)
- [Phase 3 — validation gates](phase3_rule_set.md#validation-gates-when-phase3_use_train_targettrue)
- [Phase 4 — final risk parameters](phase4_rl_risk.md)
