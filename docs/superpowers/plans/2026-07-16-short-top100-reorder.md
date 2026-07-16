# Short Top-100 Reorder + Symbol Pins Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `optimized_short/short_top100.json` by solo-scoring `short.json` rules on `train_2.csv`, pinning each survivor to strong symbols, and keeping the top 100 ordered best-first.

**Architecture:** A script loads `evaluator_v5.ipynb` helpers once, loads `train_2.csv` once, solo-backtests each rule (feature conditions only for discovery), applies gate B per symbol, pins survivors, ranks by `net_pnl * profit_factor` sum, writes JSON outputs, then verifies with one portfolio eval.

**Tech Stack:** Python 3, pandas/numpy, `evaluator_v5.ipynb` (read-only), pytest, `.venv`

## Global Constraints

- Do not modify `short.json` or `evaluator_v5.ipynb`
- Use `.venv` for all commands
- Score only on `data/train_2.csv`
- Symbol gate: `net_pnl > 0` AND `PF >= 1.2` AND `trades >= 5`
- Score: `net_pnl * profit_factor` (PF capped at 10.0 when no losses)
- Keep top 100 (or all survivors if fewer)
- Write only under `optimized_short/`
- Do not run the full GPU pipeline

---

### Task 1: Pure helpers + unit tests

**Files:**

- Create: `scripts/optimize_short_top100.py` (helper functions only first)
- Create: `tests/unit/test_optimize_short_top100.py`

**Interfaces:**

- Produces:
  - `compute_profit_factor(net_pnls: list[float] | np.ndarray) -> float`
  - `symbol_passes_gate(net_pnl: float, profit_factor: float, trades: int, *, min_pf: float = 1.2, min_trades: int = 5) -> bool`
  - `rule_score(net_pnl: float, profit_factor: float, *, pf_cap: float = 10.0) -> float`
  - `strip_symbol_conditions(conditions: list[str]) -> list[str]`
  - `pin_rule(rule: dict, symbols: list[str | int]) -> dict` (copy; feature conditions preserved; one `symbol is a,b,...` appended; symbols sorted as ints)

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_optimize_short_top100.py
import math
import pytest

from scripts.optimize_short_top100 import (
    compute_profit_factor,
    pin_rule,
    rule_score,
    strip_symbol_conditions,
    symbol_passes_gate,
)


def test_compute_profit_factor_basic():
    assert compute_profit_factor([10.0, -5.0]) == pytest.approx(2.0)


def test_compute_profit_factor_no_losses_returns_inf_sentinel_for_gate_then_capped_in_score():
    pf = compute_profit_factor([3.0, 2.0])
    assert math.isinf(pf)
    assert rule_score(5.0, pf) == pytest.approx(50.0)  # 5 * 10 cap


def test_symbol_passes_gate_b():
    assert symbol_passes_gate(10.0, 1.2, 5) is True
    assert symbol_passes_gate(10.0, 1.19, 5) is False
    assert symbol_passes_gate(10.0, 1.2, 4) is False
    assert symbol_passes_gate(0.0, 2.0, 10) is False
    assert symbol_passes_gate(-1.0, 2.0, 10) is False


def test_strip_and_pin_rule():
    rule = {
        "conditions": ["[rsi_centered_14] IS Bearish", "symbol is 1,2"],
        "tp": 2.0,
        "sl": 1.0,
        "capital_pct": 10.0,
    }
    assert strip_symbol_conditions(rule["conditions"]) == ["[rsi_centered_14] IS Bearish"]
    pinned = pin_rule(rule, [3, 1, 10])
    assert pinned["conditions"][-1] == "symbol is 1,3,10"
    assert pinned["conditions"][0] == "[rsi_centered_14] IS Bearish"
    assert rule["conditions"][-1] == "symbol is 1,2"  # original untouched
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTEST_LOW_MEMORY=1 .venv/bin/pytest tests/unit/test_optimize_short_top100.py -q`  
Expected: FAIL (import / missing functions)

- [ ] **Step 3: Implement helpers in `scripts/optimize_short_top100.py`**

Implement the five functions above. `strip_symbol_conditions` must drop any condition whose text matches `(?i)symbol\s+is\s+` or `[symbol]` case-insensitive. `pin_rule` deep-copies the rule dict.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTEST_LOW_MEMORY=1 .venv/bin/pytest tests/unit/test_optimize_short_top100.py -q`  
Expected: PASS

---

### Task 2: Solo scoring pipeline + write outputs

**Files:**

- Modify: `scripts/optimize_short_top100.py`
- Create: `optimized_short/README.md`

**Interfaces:**

- Consumes: helpers from Task 1; `evaluator_v5` via notebook exec (same skip markers as `scripts/evaluator_parity_check.py`)
- Produces:
  - `main() -> int`
  - `load_evaluator_namespace() -> dict`
  - `solo_eval_rule(ns, df, feature_cols, feature_modes, direction, rule) -> pandas.DataFrame` (trade logs; may be empty)
  - `score_rule_by_symbol(trade_logs) -> dict[str, dict]` mapping symbol -> `{net_pnl, profit_factor, trades, score}`
  - writes `optimized_short/short_top100.json` and `optimized_short/optimize_report.json`

- [ ] **Step 1: Implement discovery + rank + write**

Logic:

1. Load `short.json`; abort if missing `data/train_2.csv`.
2. Load evaluator namespace; `load_reference_dataset_schema("data/train_2.csv")` once; keep `df`, `feature_cols`, `feature_modes`.
3. Build shared `AdaptiveFuzzyEngine` + reuse one `CapitalManagedTradeSimulator` constructed once (direction=`short`).
4. For each rule (1-based original index):
   - Discovery copy: `pin_rule`-style strip only (feature conditions, no symbol pin yet).
   - Build one-rule strategy via `strategy_dict_to_rule_set`, `simulate_rule_set(..., return_logs=True)`.
   - Suppress verbose prints during the 242 loop (redirect stdout or pass a quiet path).
   - Per symbol: compute net/PF/trades/score; keep symbols passing gate B.
   - If none kept: mark dropped; continue.
   - Else: `aggregate_score = sum(scores)`; store pinned rule via `pin_rule(original, kept_symbols)`.
5. Sort survivors by `aggregate_score` DESC; take top 100.
6. Write deployable JSON + report JSON.
7. Write short README pointing at the design spec.

Quiet solo eval pattern (reuse engine):

```python
student = ns["strategy_dict_to_rule_set"](
    {"direction": "short", "rules_set": [discovery_rule]},
    df_eval=df,
)
metrics, logs = simulator.simulate_rule_set(student["rule_set"], return_logs=True)
```

- [ ] **Step 2: Run the optimizer**

Run: `PYTEST_LOW_MEMORY=1 MPLBACKEND=Agg .venv/bin/python scripts/optimize_short_top100.py`  
Expected: prints kept count, writes both JSON files; `short.json` unchanged.

- [ ] **Step 3: Sanity-check outputs**

Run:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
opt=json.loads(Path('optimized_short/short_top100.json').read_text())
rep=json.loads(Path('optimized_short/optimize_report.json').read_text())
assert opt['direction']=='short'
assert 1 <= len(opt['rules_set']) <= 100
assert all(any('symbol is' in c.lower() for c in r['conditions']) for r in opt['rules_set'])
assert Path('short.json').read_text()  # still exists
print('n_rules', len(opt['rules_set']), 'kept_in_report', rep['counts']['kept_after_rank'])
PY
```

Expected: assertions pass.

---

### Task 3: Verify optimized strategy on train_2

**Files:**

- Create: `optimized_short/train_2_verification.json` (metrics only)

- [ ] **Step 1: Evaluate `short_top100.json` on `train_2.csv` with evaluator_v5**

Reuse the evaluation pattern from prior sessions (load ns, schema from train_2, `evaluate_student_strategy_on_dataset`, save equity under `optimized_short/equity_train_2.png`).

- [ ] **Step 2: Record metrics**

Write `optimized_short/train_2_verification.json` with return/DD/PF/win_rate/trades/status.

- [ ] **Step 3: Confirm `short.json` untouched**

Run: `git status -- short.json` or checksum compare if not a git repo; ensure no modifications to `short.json` / `evaluator_v5.ipynb`.

---

## Spec coverage checklist

| Spec requirement                  | Task     |
| --------------------------------- | -------- |
| Solo score Net_PnL × PF           | Task 2   |
| Symbol gate B                     | Task 1–2 |
| PF cap 10 for scoring             | Task 1   |
| Pin `symbol is a,b,...`           | Task 1–2 |
| Top 100 best-first                | Task 2   |
| Outputs under `optimized_short/`  | Task 2–3 |
| Leave `short.json` untouched      | Task 2–3 |
| Verify on train_2                 | Task 3   |
| Efficient single data/engine load | Task 2   |
