# Task 3 — Add `_is_positive_good`-style gate

## Why
The friend requires every rule that survives selection to be:
- positive on BOTH train and validation (`return > 0` on both)
- PF ≥ 1.0 on BOTH train and validation
- ≥ MIN_TRAIN_TRADES trades on train, ≥ MIN_VALID_TRADES on val
- (optionally) execution-healthy: `(skipped/raw) ≤ 0.20` and `(executed/raw) ≥ 0.60` — that part is Task 4

My current Phase 3 only checks `min(train, val)` return with a 20-40% gap-reject. A rule with `train_return=15%, val_return=2%, train_pf=0.95, val_pf=1.05` would pass my current gate (return positive on both, gap < 20%) but is not actually profitable on the train side. The friend's `_is_positive_good` rejects this.

## Required reading
- `.opencode/plans/PLAN.md` (overall plan)
- `.opencode/CONTEXT.md` (JSON output contract, evaluator_v5 contract)
- The friend's reference: `friend_project/gpu_fuzzy_trader/rb_governor.py` lines 264-298 (`_is_positive_good`).
- Existing `gpu_fuzzy_trader/phases/phase3_rule_set.py` (per-symbol greedy + global pool fallback) and `phases/phase2_rule_pool.py` (pool admission).

## Behavior changes

### Add a `gate_positive_good` helper in `phases/phase3_rule_set.py`
```python
def gate_positive_good(
    train_metrics: dict,
    val_metrics: dict,
    *,
    min_train_return: float = 0.0,
    min_val_return: float = 0.0,
    min_train_pf: float = 1.0,
    min_val_pf: float = 1.0,
    min_train_trades: int = 25,
    min_val_trades: int = 15,
) -> bool:
    """Return True iff the rule is positive on both train and val with PF >= 1.0."""
```
This is a pure function: take two metric dicts, return a bool. No side effects. Place it near the top of `phase3_rule_set.py` so other modules can import it.

### Wire the gate into Phase 3 selection
In the per-symbol greedy (`_per_symbol_greedy`) and the global pool fallback (`_try_global_pool_fallback`), filter candidates so a rule that fails `gate_positive_good` on the train+val engines is rejected. The friend uses this gate in `RB_GLOBAL_REQUIRE_POSITIVE_TRAIN_VALID = True` mode. I will default it to ON behind a config flag.

In `_score_pool_rule_on_symbol` and `_robust_combo_return` (the inner per-symbol functions), the gate should be applied as a hard reject (return -999 / empty list) before the score is computed.

### Wire the gate into Phase 2 pool admission (light-touch)
In `phases/phase2_support.py` (or wherever `_passes_pool_admission_impl` lives), add an optional stricter pool admission that requires `gate_positive_good`. Default OFF to avoid breaking the existing pool. The friend does this in `PHASE2_CV_MIN_WORST_RETURN`. I will add a new config flag `PHASE2_STRICT_POSITIVE_GOOD = False` (default OFF; turn ON in Task 5 when expanding the pool).

### Config keys to add to `config.py`
```python
PHASE3_REQUIRE_POSITIVE_GOOD = True
PHASE3_MIN_TRAIN_RETURN = 0.0
PHASE3_MIN_VAL_RETURN = 0.0
PHASE3_MIN_TRAIN_PF = 1.0
PHASE3_MIN_VAL_PF = 1.0
PHASE3_MIN_TRAIN_TRADES = 25
PHASE3_MIN_VAL_TRADES = 15
PHASE2_STRICT_POSITIVE_GOOD = False  # Task 5 will turn this on
```

## Out of scope
- Do NOT add evaluator-failure-mode awareness (Task 4).
- Do NOT change the JSON output format.
- Do NOT modify `evaluator_v5.ipynb`.
- Do NOT change the per-symbol greedy max-rules cap.
- Do NOT change the GPU engine or EvoX runner.

## Acceptance criteria
1. `from gpu_fuzzy_trader.phases.phase3_rule_set import gate_positive_good` works.
2. `gate_positive_good(train_m, val_m)` returns `True` when both `total_return_pct > 0`, both `profit_factor >= 1.0`, and trades meet the floors.
3. `gate_positive_good(train_m, val_m)` returns `False` if either return is ≤ 0, either PF is < 1.0, or trades are below the floors.
4. `gate_positive_good` handles missing keys gracefully (returns `False` for missing `total_return_pct` or `profit_factor`).
5. Phase 3 per-symbol greedy, when `PHASE3_REQUIRE_POSITIVE_GOOD=True`, never returns a rule whose `gate_positive_good(train, val)` is `False`. (Test this by mocking the engines with a controlled pool of mixed good/bad rules.)
6. The existing `PHASE3_MAX_TRAIN_VAL_GAP_PCT=40.0` constant stays in place (we add the new gate alongside, not replacing).
7. New unit test `tests/unit/test_positive_good_gate.py` exists with at least 6 cases (all combinations of train/val return / PF / trades).
8. All 11 existing tests still pass; new tests pass.
9. No changes to `evaluator_v5.ipynb` or the GPU engine.

## Constraints
- Stay on `feature/task-3-positive-good-gate` (off `main` after task-2 is merged).
- 12.7 GiB RAM total — use synthetic small DataFrames in tests.
- PEP 8, type hints, module logger.
- Use only existing third-party deps.

## Files I will touch
- `gpu_fuzzy_trader/config.py` — 8 new config keys
- `gpu_fuzzy_trader/phases/phase3_rule_set.py` — `gate_positive_good` helper + wire it into per-symbol greedy
- `gpu_fuzzy_trader/phases/phase2_support.py` — optional stricter pool admission (default OFF)
- `tests/unit/test_positive_good_gate.py` — new test file (≥ 6 cases)
