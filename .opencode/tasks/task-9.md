# Task 9 — Add `evaluator_clean` writer

## Why
The friend writes two output files for each direction:
1. `outputs/long.json` and `outputs/short.json` — the full strategy
   with metadata (e.g., `rb_score`, `rb_train_return_pct`, etc.)
2. `outputs/evaluator_clean/long_evaluator_clean.json` and
   `outputs/evaluator_clean/short_evaluator_clean.json` — a stripped
   version with ONLY `direction` and `rules_set`.

The current `evaluator_v5.ipynb` reads only `direction` and
`rules_set` (per my analysis), so the extra metadata is tolerated.
But a future stricter version of the evaluator might reject unknown
top-level keys, so the clean file is a safety net.

This is a defensive change. The current evaluator works with my
extra metadata, but the clean writer is a low-cost insurance.

## Required reading
- `.opencode/plans/PLAN.md`
- `.opencode/CONTEXT.md` (JSON output contract)
- The friend's reference: `friend_project/gpu_fuzzy_trader/rb_governor.py` lines 707-732 (`_write_clean_evaluator`).
- My existing `gpu_fuzzy_trader/output/writer.py` (the existing `Output_Writer` that writes the main `outputs/long.json` / `outputs/short.json`).
- The current shape of `outputs/long.json` and `outputs/short.json`.

## Behavior changes

### Step 1 — Add a `write_evaluator_clean` function

In `gpu_fuzzy_trader/output/writer.py` (or a new file
`output/evaluator_clean.py`), add:
```python
def write_evaluator_clean(strategy: dict, output_path: str | Path) -> None:
    """Write a stripped strategy file with only direction and rules_set.
    
    Useful for evaluators that reject unknown top-level keys.
    """
    clean = {
        "direction": strategy["direction"],
        "rules_set": strategy["rules_set"],
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(clean, fh, indent=2)
```

The function takes a strategy dict (the same one passed to the main
writer) and writes a stripped version.

### Step 2 — Wire into the existing `Output_Writer`

When `Output_Writer.write(strategy, direction)` is called, also call
`write_evaluator_clean(strategy, output_dir / "evaluator_clean" / f"{direction}_evaluator_clean.json")`.

The clean write happens AFTER the main write (so a main-write failure
doesn't leave a stale clean file).

### Step 3 — Add a config flag for opt-out

```python
# In config.py
WRITE_EVALUATOR_CLEAN = True
```

When `False`, skip the clean write. Default `True` (the friend always
writes the clean file).

## Out of scope
- Do NOT change the JSON shape of the main `outputs/long.json` / `outputs/short.json`.
- Do NOT modify `evaluator_v5.ipynb`.
- Do NOT touch the GPU engine or EvoX runner.
- Do NOT add Task 10 features.

## Acceptance criteria
1. `WRITE_EVALUATOR_CLEAN` config key is present and accessible (default `True`).
2. `from gpu_fuzzy_trader.output.writer import write_evaluator_clean` works.
3. `write_evaluator_clean({"direction": "long", "rules_set": [...], "extra_key": "extra_value"}, "/tmp/test.json")` writes a file with ONLY `direction` and `rules_set` (the `extra_key` is stripped).
4. The function creates parent directories if they don't exist.
5. The function is wired into the main `Output_Writer.write` path.
6. New unit test `tests/unit/test_evaluator_clean_writer.py` with ≥ 3 cases:
   - A strategy with extra keys produces a clean file with only `direction` and `rules_set`.
   - Parent directory is created if missing.
   - The function returns `None` and does not raise when the strategy has no extra keys.
7. All existing tests pass.
8. No changes to `evaluator_v5.ipynb` or the GPU engine.

## Constraints
- Stay on `feature/task-9-evaluator-clean-writer` (off `main` after task-8 is merged).
- 12.7 GiB RAM total.
- PEP 8, type hints, module logger.
- Use only existing third-party deps.

## Files I will touch
- `gpu_fuzzy_trader/config.py` — 1 new `WRITE_EVALUATOR_CLEAN` key
- `gpu_fuzzy_trader/output/writer.py` — add `write_evaluator_clean` function; wire into the main `write` method
- `tests/unit/test_evaluator_clean_writer.py` (new) — ≥ 3 cases
