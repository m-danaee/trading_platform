# Plan 001: Rename train_2/test_2 datasets to train/test

> **Executor instructions**: Follow step by step. Run every verification command before proceeding. On STOP conditions, stop and report. Do **not** update `plans/README.md` (reviewer maintains index).
>
> **Drift check (run first)**: `git diff --stat 425f469..HEAD -- gpu_fuzzy_trader/config.py main.ipynb README.md RUN.md build_train2_test2.py build_train2_extended.py download_ohlcv_2024.py gpu_fuzzy_trader/run_pipeline.py tests/unit/test_data_splitter.py tests/unit/test_phase5_oos.py tests/unit/test_rb_governor_data_load.py`
> Mismatch with "Current state" excerpts → STOP.

## Status

- **Priority**: P1
- **Effort**: S
- **Risk**: LOW
- **Depends on**: none
- **Category**: migration
- **Planned at**: commit `425f469`, 2026-07-16

## Why this matters

Defaults and docs still point at `train_2.csv` / `test_2.csv`. The user wants canonical names `train.csv` / `test.csv` everywhere in Python and `main.ipynb`. Mismatched names cause Colab/Drive bootstrap failures and confuse operators. This is a pure rename of path strings and docs — no algorithm change.

## Current state

Defaults in [`gpu_fuzzy_trader/config.py`](gpu_fuzzy_trader/config.py):

```python
# ~108–115
TRAIN_CSV_PATH = _env_str(
    "TRAIN_CSV_PATH",
    os.path.join(DATA_ROOT, "train_2.csv") if DATA_ROOT else "data/train_2.csv",
)
TEST_CSV_PATH = _env_str(
    "TEST_CSV_PATH",
    os.path.join(DATA_ROOT, "test_2.csv") if DATA_ROOT else "data/test_2.csv",
)
```

Comments nearby still say `train_2.csv` (cache rebuild, SPLIT_MODE).

[`main.ipynb`](main.ipynb) bootstrap sets Drive env vars to `train_2.csv` / `test_2.csv`.

Builder scripts write `data/train_2.csv` / `data/test_2.csv`:

- [`build_train2_test2.py`](build_train2_test2.py)
- [`build_train2_extended.py`](build_train2_extended.py)
- [`download_ohlcv_2024.py`](download_ohlcv_2024.py) (docstring only)

Docs: [`README.md`](README.md), [`RUN.md`](RUN.md). Docstring in [`gpu_fuzzy_trader/run_pipeline.py`](gpu_fuzzy_trader/run_pipeline.py) (~973).

Tests use **fixture filenames** under `tmp_path` (not product defaults):

- [`tests/unit/test_data_splitter.py`](tests/unit/test_data_splitter.py) — `tmp_path / "train_2.csv"`
- [`tests/unit/test_phase5_oos.py`](tests/unit/test_phase5_oos.py)
- [`tests/unit/test_rb_governor_data_load.py`](tests/unit/test_rb_governor_data_load.py)

**AGENTS.md constraints**: use `.venv`; `PYTEST_LOW_MEMORY=1`; related tests only; do not run full pipeline; do not modify `evaluator_v5.ipynb`.

**Convention**: env overrides `TRAIN_CSV_PATH` / `TEST_CSV_PATH` remain the escape hatch; only change default string values and documentation.

## Commands you will need

| Purpose         | Command                                                                                                                                           | Expected on success                                                                                           |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Drift           | `git diff --stat 425f469..HEAD -- <in-scope paths>`                                                                                               | empty or only expected prior work                                                                             |
| Codelookup      | `.venv/bin/python .cursor/skills/codelookup/scripts/pre_check.py` (or follow codelookup skill) before edits                                       | blast-radius noted                                                                                            |
| Grep leftovers  | `rg -n 'train_2\.csv\|test_2\.csv' --glob '*.py' --glob '*.md' --glob '*.ipynb' --glob '*.sh'`                                                    | no hits in in-scope product/docs/tests (skill docs under `.cursor/` may still mention old names — leave them) |
| Config defaults | `.venv/bin/python -c "from gpu_fuzzy_trader import config as c; print(c.TRAIN_CSV_PATH); print(c.TEST_CSV_PATH)"`                                 | ends with `train.csv` and `test.csv` (path prefix may be DATA_ROOT)                                           |
| Related tests   | `PYTEST_LOW_MEMORY=1 .venv/bin/pytest tests/unit/test_data_splitter.py tests/unit/test_phase5_oos.py tests/unit/test_rb_governor_data_load.py -q` | all pass                                                                                                      |

## Suggested executor toolkit

- Invoke **codelookup** skill before any source edit.
- Run blast-radius check for `TRAIN_CSV_PATH`, `TEST_CSV_PATH`, and string `train_2.csv`.
- Cascade-update all dependents in the same change.
- For [`main.ipynb`](main.ipynb): prefer Cursor `EditNotebook` / notebook MCP — **do not** raw-edit notebook JSON with sed/Write (breaks notebook format).

## Scope

**In scope**:

- `gpu_fuzzy_trader/config.py` (defaults + comments that name the CSV)
- `gpu_fuzzy_trader/run_pipeline.py` (docstrings mentioning train_2/test_2)
- `main.ipynb` (markdown + bootstrap env defaults)
- `README.md`, `RUN.md`
- `build_train2_test2.py`, `build_train2_extended.py`, `download_ohlcv_2024.py` (output paths + comments; **do not** rename the script files themselves unless trivial and already in scope — keep script filenames, change CSV output names)
- Fixture paths in the three test files listed above

**Out of scope**:

- Renaming physical files on disk / Drive (operator action; document in Maintenance notes)
- `evaluator_v5.ipynb`
- `.cursor/skills/**` historical references (optional cleanup only if grepped as required — prefer leave)
- Algorithm / config numeric knobs
- Plans 002–004

## Git workflow

- Branch: `advisor/001-rename-train-test-csv`
- Conventional commits matching repo style (e.g. `chore: rename train_2/test_2 csv defaults to train/test`)
- Do NOT push or open PR unless instructed

## Steps

### Step 1: Codelookup + drift check

Run drift check. Invoke codelookup on `TRAIN_CSV_PATH` / `TEST_CSV_PATH` / `train_2.csv`. List every call site that will change.

**Verify**: drift clean relative to plan excerpts; blast radius includes config, notebook, builders, docs, tests above.

### Step 2: Update config defaults and comments

In `gpu_fuzzy_trader/config.py`:

- Change default filenames `train_2.csv` → `train.csv`, `test_2.csv` → `test.csv`.
- Update comments that say splits/cache come from `train_2.csv` to `train.csv`.
- Keep env var names `TRAIN_CSV_PATH` / `TEST_CSV_PATH` unchanged.

**Verify**:

```bash
.venv/bin/python -c "from gpu_fuzzy_trader import config as c; assert c.TRAIN_CSV_PATH.endswith('train.csv'), c.TRAIN_CSV_PATH; assert c.TEST_CSV_PATH.endswith('test.csv'), c.TEST_CSV_PATH"
```

### Step 3: Update builders, pipeline docstring, docs

- Builders: write/read `data/train.csv` and `data/test.csv`.
- `run_pipeline.py`: docstring text only.
- `README.md` / `RUN.md`: all user-facing `train_2` / `test_2` CSV references → `train` / `test`. Preserve the hard invariant wording: test CSV is Phase 5 only.

**Verify**: `rg -n 'train_2\.csv|test_2\.csv' README.md RUN.md build_train2_test2.py build_train2_extended.py download_ohlcv_2024.py gpu_fuzzy_trader/run_pipeline.py gpu_fuzzy_trader/config.py` → no matches.

### Step 4: Update main.ipynb

Via EditNotebook (or notebook MCP if available):

- Markdown: Drive / data file names → `train.csv` / `test.csv`.
- Bootstrap: `TRAIN_CSV_PATH` / `TEST_CSV_PATH` env defaults → `.../train.csv` and `.../test.csv`.
- Any copy/link logic that hardcodes `train_2.csv` / `test_2.csv`.

**Verify**: notebook still valid JSON; `rg -n 'train_2\.csv|test_2\.csv' main.ipynb` → no matches.

### Step 5: Update test fixture filenames

Rename `tmp_path / "train_2.csv"` (and any `test_2.csv` fixtures) to `train.csv` / `test.csv` in the three unit test files. Do not change test logic.

**Verify**:

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/pytest tests/unit/test_data_splitter.py tests/unit/test_phase5_oos.py tests/unit/test_rb_governor_data_load.py -q
```

### Step 6: Final leftover grep

```bash
rg -n 'train_2\.csv|test_2\.csv' --glob '*.py' --glob '*.md' --glob '*.ipynb' --glob '*.sh'
```

Any remaining hit under in-scope product/docs/tests → fix. Hits only under `.cursor/` or historical plans → leave (or note in report).

## Test plan

- No new test file required.
- Existing loader/splitter/OOS/RB data-load tests must pass with new fixture names.
- Manual operator note: rename or symlink Drive files `train_2.csv`→`train.csv`, `test_2.csv`→`test.csv` before Colab run.

## Done criteria

- [ ] Config defaults end with `train.csv` / `test.csv`
- [ ] No `train_2.csv` / `test_2.csv` in in-scope product, docs, notebook, builders, listed tests
- [ ] Related pytest command passes with `PYTEST_LOW_MEMORY=1`
- [ ] Codelookup blast radius addressed
- [ ] No out-of-scope files modified
- [ ] `evaluator_v5.ipynb` untouched

## STOP conditions

- Current state excerpts don't match live code.
- Verification fails twice after reasonable fix.
- Fix requires algorithm/config knob changes (wrong plan — use 003/004).
- Physical data files missing and tests need real CSVs from disk (should not; tests use tmp_path).
- Notebook edit corrupts JSON and cannot be repaired safely.

## Maintenance notes

- Operators must rename Drive/local CSVs once; env overrides can temporarily point at old names during migration.
- Parquet cache keys are independent of CSV filename but rebuild when train CSV mtime/rows change — after rename, expect one cache rebuild.
- Follow-ups 002–004 do not depend on this rename functionally, but do 001 first to avoid string churn.
