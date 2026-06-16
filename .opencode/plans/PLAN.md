# Plan — Task 11: Remove purged CV feature completely

## Goal

Completely remove the "purged CV" (`purged_rolling_cv` / `PurgedFold` /
`PHASE2_CV_*` / `cv_folds`) feature from the project — **zero leftovers**
in source, tests, docs, or comments — and keep the `holdout_70_30`
path (current default) working unchanged.

## Background

The purged rolling cross-validation feature was a parallel evaluation
path for Phases 2–4 that was added in earlier tasks. The project's
default `SPLIT_MODE` has been `"holdout_70_30"` for some time, so the
entire purged CV code path is **dead code in production** (gated by
`SPLIT_MODE == "purged_rolling_cv"` and only exercised by dedicated
unit tests). The user wants it gone to simplify the codebase and
remove the maintenance burden of two parallel evaluation paths.

Reference baseline (from `gpu_fuzzy_trader/config.py` line 284):

```python
SPLIT_MODE = "holdout_70_30"
```

## Current scope (pre-removal inventory)

| File | Purged CV references | Action |
|---|---|---|
| `gpu_fuzzy_trader/data/cv_folds.py` | Whole file (PurgedFold, build_purged_rolling_folds, primary_holdout_from_folds) | **Delete file**; keep `holdout_70_30_split` (move to `splitter.py`) |
| `gpu_fuzzy_trader/validation/rolling_cv.py` | Whole file (duplicate of cv_folds.py) | **Delete file** |
| `gpu_fuzzy_trader/data/splitter.py` | PurgedFold imports, `cv_folds` param, purged branch, `_persist_cv_manifest`, `load_cv_folds_from_manifest`, `build_cv_folds` | Strip; keep only the `holdout_70_30` branch |
| `gpu_fuzzy_trader/config.py` | CV_N_FOLDS, CV_EMBARGO_BARS, CV_BARS_PER_DAY, CV_MIN_TRAIN_MONTHS, CV_FOLDS_MANIFEST_PATH, PHASE2_CV_* (15+), PHASE2_CV_FOLD_WORKERS, PHASE2_EARLY_STOP_DISABLED_IN_CV, PHASE2_PLATEAU_EARLY_STOP_DISABLED_IN_CV, `_cv_pool_min_folds_pass()`, `_CV_POOL_MIN_FOLDS_PASS`, `_CV_RANK_MIN_FOLDS_PASS`; SPLIT_MODE comment block, assert statements | Strip all |
| `gpu_fuzzy_trader/phases/phase2_rule_pool.py` | `_val_trade_floor_for_objectives`, `cv_fold` local, `_uses_cv_engines()`, `_build_engines()` cv branch, `use_cv_admission` block, `evaluate_purged_cv_pool_admission_batch` reference | Strip; fix signatures |
| `gpu_fuzzy_trader/phases/phase2_support.py` | `_split_mode_is_purged_cv()`, `_pool_admission_floors` cv_fold branch, `_passes_pool_admission_impl` cv_fold param, `passes_pool_admission_cv_fold()` (whole fn), `passes_pool_entry_admission` cv_fold branch, `passes_pool_trade_floor` cv_fold branch | Strip |
| `gpu_fuzzy_trader/phases/phase3_rule_set.py` | `cv_folds` parameter on `Rule_Set_Selector.__init__` | Remove param; keep signature clean |
| `gpu_fuzzy_trader/phases/phase4_wf_optimizer.py` | `cv_folds` param, `build_phase4_multi_cv_fold_splits` (whole fn), `_walk_forward_splits_for_val_block`, `Phase4WalkForwardEvaluator` cv_folds branch | Strip |
| `gpu_fuzzy_trader/phases/phase5_oos.py` | `_cv_folds = splitter.split_and_persist(train_full)` (3-tuple destructure) | Use 2-tuple |
| `gpu_fuzzy_trader/evolution/evox_runner.py` | `_split_mode_is_purged_cv()`, all `if SPLIT_MODE == "purged_rolling_cv"` branches | Strip |
| `gpu_fuzzy_trader/run_pipeline.py` | `self._cv_folds: list`, `_prune_cv_folds_after_phase1`, `_apply_debug_symbol_scope` cv-folds block, `_load_and_split_data` cv-folds set | Strip; `_apply_debug_symbol_scope` only filters train/val |
| `gpu_fuzzy_trader/backtest/gpu_engine.py` | `_phase2_trade_floor()` SPLIT_MODE branch | Use `MIN_TRADE_POOL_FLOOR` only |
| `tests/unit/test_phase2_pool_admission.py` | Tests for `passes_pool_admission_cv_fold` and per-fold relaxed thresholds | **Delete file**; functionality subsumed by the holdout path |
| `tests/unit/test_run_pipeline.py` | `PurgedFold` imports, `TestPruneCvFoldsAfterPhase1`, `TestDebugSymbolScope` cv-folds checks | Strip purged tests; keep `TestDebugSymbolScope` (it still works without cv-folds plumbing) |
| `tests/unit/test_evox_runner.py` | One test sets `SPLIT_MODE = "purged_rolling_cv"` | Strip that test |
| `docs/phase0_shared.md`, `docs/phase2_rule_pool.md`, `docs/phase3_rule_set.md`, `docs/phase4_wf_risk.md`, `docs/phase5_oos.md`, `docs/README.md` | ~30 mentions of purged CV | Strip from docs |
| `.opencode/CONTEXT.md` | Notes about purged CV mode | Update task ledger to add task 11 |

## Acceptance criteria

1. **No `purged` references in code/tests/docs (except the friend
   project)** — `grep -rn "purged\|PurgedFold" gpu_fuzzy_trader tests docs
   --include="*.py" --include="*.ipynb" --include="*.md"` returns ZERO
   matches.
2. **No `cv_folds` instance state or imports remain** —
   `grep -rn "_cv_folds\|cv_folds:" gpu_fuzzy_trader tests` returns
   ZERO matches.
3. **No `PHASE2_CV_*` config keys remain** —
   `grep -n "PHASE2_CV\|PURGED_CV" gpu_fuzzy_trader/config.py` returns
   ZERO matches.
4. **Files removed**:
   - `gpu_fuzzy_trader/data/cv_folds.py` (deleted)
   - `gpu_fuzzy_trader/validation/rolling_cv.py` (deleted)
   - `tests/unit/test_phase2_pool_admission.py` (deleted)
5. **The `holdout_70_30` path still works**:
   - `python -c "from gpu_fuzzy_trader.data.splitter import Data_Splitter; print(Data_Splitter)"` works
   - `python -c "import gpu_fuzzy_trader.run_pipeline as m; print(m.Pipeline_Orchestrator)"` works
   - `python -c "import gpu_fuzzy_trader.phases.phase2_support; print('ok')"` works
   - The 4 small unit tests we can safely run pass (one each from
     `test_data_splitter`, `test_run_pipeline`,
     `test_phase2_support`, `test_reporter`).
6. **`SPLIT_MODE` constant is removed** (no longer a configurable
   option — the project only has one path). All `str(_cfg.SPLIT_MODE)`
   references throughout the code are removed.
7. **No dead imports, no dead functions, no dead branches** left
   behind (per AGENTS.md: "remove additional wasted parts").
8. **Friend project is untouched** — `friend_project/` is committed
   code and is the reference project; the user did not ask to touch
   it.

## Tasks

### Task 11 — Remove purged CV (one branch, one implementer)

**Priority: high**

**Branch**: `feature/task-11-remove-purged-cv`

**Steps** for the implementer (in order):

1. **Create the feature branch from `main`**:
   ```bash
   git checkout main && git pull
   git checkout -b feature/task-11-remove-purged-cv
   ```

2. **Delete the standalone purged CV modules**:
   - `git rm gpu_fuzzy_trader/data/cv_folds.py`
   - `git rm gpu_fuzzy_trader/validation/rolling_cv.py`
   - `git rm tests/unit/test_phase2_pool_admission.py`

3. **Strip `gpu_fuzzy_trader/config.py`**:
   - Remove `CV_FOLDS_MANIFEST_PATH`.
   - Remove `CV_N_FOLDS`, `CV_EMBARGO_BARS`, `CV_BARS_PER_DAY`,
     `CV_MIN_TRAIN_MONTHS` and their comment blocks.
   - Remove `_cv_pool_min_folds_pass()`, `_CV_POOL_MIN_FOLDS_PASS`,
     `_CV_RANK_MIN_FOLDS_PASS`.
   - Remove the entire "Purged CV pool admission" section
     (PHASE2_CV_POOL_MIN_FOLDS_PASS, PHASE2_CV_MERGED_GATE_HARD,
     PHASE2_CV_MIN_TRADE_POOL_FLOOR, PHASE2_CV_POOL_TRAIN_RETURN_MIN_PCT,
     PHASE2_CV_POOL_VAL_RETURN_MIN_PCT, PHASE2_CV_PROFIT_FACTOR_FLOOR,
     PHASE2_CV_MIN_VAL_TRADES, PHASE2_CV_POOL_TARGET_MIN,
     PHASE2_CV_POOL_RANK_ADMIT_TOP_K, PHASE2_CV_RANK_MIN_FOLDS_PASS).
   - Remove the "Phase 2 relaxed pool-admission thresholds (Task 5)"
     PHASE2_CV_MIN_WORST_RETURN, PHASE2_CV_MIN_WORST_PF,
     PHASE2_CV_MAX_WORST_DD, PHASE2_CV_MIN_FOLD_TRADES block (or the
     whole section if it only exists for purged CV).
   - Remove `PHASE2_CV_FOLD_WORKERS`.
   - Remove `PHASE2_EARLY_STOP_DISABLED_IN_CV`.
   - Remove `PHASE2_PLATEAU_EARLY_STOP_DISABLED_IN_CV`.
   - Remove the SPLIT_MODE constant and the comment block above it.
   - Remove the tuning cheat-sheet rows that mention CV_N_FOLDS /
     PHASE2_CV_* / SPLIT_MODE.
   - Remove the `assert CV_N_FOLDS ...`, `assert 1 <= PHASE2_CV_*`
     asserts at the bottom.
   - Update the docstring at the top if it mentions SPLIT_MODE / CV.

4. **Strip `gpu_fuzzy_trader/data/splitter.py`**:
   - Move `holdout_70_30_split` (from `cv_folds.py`) into this file
     (it was already imported from cv_folds; inline it).
   - Remove the `PurgedFold` import.
   - Remove the `build_purged_rolling_folds` import.
   - Remove the `primary_holdout_from_folds` import.
   - Change `split_and_persist` to return `tuple[pd.DataFrame, pd.DataFrame]`
     (drop the third element).
   - Remove the `purged_rolling_cv` mode branch from `split_and_persist`.
   - Remove `build_cv_folds` method.
   - Remove `load_cv_folds_from_manifest` static method.
   - Remove `_persist_cv_manifest` static method.
   - Remove module-level `build_cv_folds` function.
   - Update the class docstring to drop the purged CV mention.
   - Update the split_and_persist docstring to drop the cv_folds
     note.

5. **Strip `gpu_fuzzy_trader/phases/phase2_rule_pool.py`**:
   - Remove the `from gpu_fuzzy_trader.data.cv_folds import
     PurgedFold` style imports (if any survive).
   - Replace `_val_trade_floor_for_objectives()` with a plain
     `min(int(_cfg.MIN_TRADE_POOL_FLOOR) // 4, 10)` call.
   - Remove the `cv_fold = str(_cfg.SPLIT_MODE).strip().lower() ==
     "purged_rolling_cv"` lines and always pass `cv_fold=False`.
   - Remove `_uses_cv_engines()` (or inline it as `False`).
   - In `_build_engines()`, remove the cv branch and the
     `from gpu_fuzzy_trader.phases.phase2_cv import build_cv_fold_engines`
     import (phase2_cv.py never existed — this was a latent bug).
   - Remove the `use_cv_admission = False` block (and the entire
     `if use_cv_admission:` body).
   - Remove the `evaluate_purged_cv_pool_admission_batch` reference
     and the related docstring paragraph.
   - Update Rule_Pool_Generator.__init__ to drop the `cv_folds`
     parameter (signature must remain backward compatible — i.e.
     accepting the param but ignoring it is OK, or just dropping it).
   - Search for any other `cv_fold` / `cv_folds` / `purged` reference
     in this file and remove it.

6. **Strip `gpu_fuzzy_trader/phases/phase2_support.py`**:
   - Remove `_split_mode_is_purged_cv()`.
   - In `_pool_admission_floors`, drop the `cv_fold` parameter and
     always return the holdout tuple.
   - In `_passes_pool_admission_impl`, drop the `cv_fold` parameter;
     remove the `cv_fold and _split_mode_is_purged_cv()` drawdown
     checks; keep the rest.
   - Remove the entire `passes_pool_admission_cv_fold()` function.
   - In `passes_pool_entry_admission`, remove the `_split_mode_is_purged_cv()`
     branch and the `cv_folds_passing` / `cv_folds_total` check; keep
     the holdout logic.
   - In `passes_pool_trade_floor`, remove the `if
     _split_mode_is_purged_cv():` trade-floor switch.
   - Search for any other `cv_fold` / `purged` reference and remove it.

7. **Strip `gpu_fuzzy_trader/phases/phase3_rule_set.py`**:
   - Remove the `cv_folds` parameter from `Rule_Set_Selector.__init__`
     (and any internal storage of it). Drop the corresponding
     docstring paragraph.

8. **Strip `gpu_fuzzy_trader/phases/phase4_wf_optimizer.py`**:
   - Remove the `cv_folds` parameter from `WalkForwardRiskOptimizer`
     and from `Phase4WalkForwardEvaluator`.
   - Delete the `build_phase4_multi_cv_fold_splits` function and
     `_walk_forward_splits_for_val_block` helper.
   - In `Phase4WalkForwardEvaluator.__init__`, drop the `cv_folds`
     branch.
   - Update the docstring to drop the cv_folds paragraph.

9. **Strip `gpu_fuzzy_trader/phases/phase5_oos.py`**:
   - Change `train_df, val_df, _cv_folds = splitter.split_and_persist(train_full)`
     to `train_df, val_df = splitter.split_and_persist(train_full)`.

10. **Strip `gpu_fuzzy_trader/evolution/evox_runner.py`**:
    - Remove `_split_mode_is_purged_cv()`.
    - Remove `if str(_cfg.SPLIT_MODE).strip().lower() == "purged_rolling_cv"`
      blocks (keep the else-branch logic).
    - Remove `if _split_mode_is_purged_cv()` blocks.
    - Search for any other `purged` reference and remove it.

11. **Strip `gpu_fuzzy_trader/run_pipeline.py`**:
    - Remove `self._cv_folds: list = []` from `__init__`.
    - Remove `_prune_cv_folds_after_phase1()` method entirely.
    - Remove the three calls to `self._prune_cv_folds_after_phase1(...)`.
    - In `_load_and_split_data`, drop the `_cv_folds = []` assignment
      and update the return to drop the cv_folds plumbing.
    - In `_apply_debug_symbol_scope`, remove the
      `from gpu_fuzzy_trader.data.cv_folds import PurgedFold` import
      and the `if self._cv_folds:` block that rebuilds cv folds.
    - Update the docstring on `_load_and_split_data`.

12. **Strip `gpu_fuzzy_trader/backtest/gpu_engine.py`**:
    - In `_phase2_trade_floor()`, remove the SPLIT_MODE branch and
      always return `int(_cfg.MIN_TRADE_POOL_FLOOR)`.

13. **Strip `tests/unit/test_run_pipeline.py`**:
    - Remove the `PurgedFold` import.
    - Remove the `TestPruneCvFoldsAfterPhase1` class entirely.
    - In `TestDebugSymbolScope`, remove the `fold = PurgedFold(...)` /
      `orch._cv_folds = [fold]` setup and the assertions that read
      `orch._cv_folds[0]`. The remaining train/val filter assertions
      stay.

14. **Strip `tests/unit/test_evox_runner.py`**:
    - Remove the test that sets `SPLIT_MODE = "purged_rolling_cv"`.
      Identify the exact test by reading the test file.

15. **Strip `docs/`**:
    - `docs/phase0_shared.md`: remove the "A. Purged rolling CV"
      section, the "SPLIT_MODE" row, the "CV_FOLDS_MANIFEST_PATH"
      row, and the "SPLIT_MODE | `purged_rolling_cv`" recommendation
      paragraphs. Keep the "holdout_75_25" / 70-30 narrative.
    - `docs/phase2_rule_pool.md`: remove the "Purged CV" section and
      any `PHASE2_CV_*` mentions.
    - `docs/phase3_rule_set.md`: remove the "Purged CV evaluation"
      section and any `PHASE2_CV_*` / `SPLIT_MODE` mentions.
    - `docs/phase4_wf_risk.md`: remove the "Relationship to Phase 2/3
      purged CV" section and the "last CV fold" mentions.
    - `docs/phase5_oos.md`: remove the "purged rolling CV" mentions.
    - `docs/README.md`: remove the `(purged CV vs 75/25)` /
      `purged_rolling_cv` entries.

16. **Update `.opencode/CONTEXT.md`**:
    - Add task 11 row to the task ledger: "Remove purged CV feature
      completely" / **DONE / APPROVED** / `feature/task-11-remove-purged-cv`
      / `<commit>` / **YES** (`<merge-sha>` on `main`).
    - Update the "Active orchestration state" to point at task 11.

17. **Commit the changes**:
    ```bash
    git add -A
    git commit -m "task-11: remove purged CV feature completely

    - Delete gpu_fuzzy_trader/data/cv_folds.py
    - Delete gpu_fuzzy_trader/validation/rolling_cv.py
    - Delete tests/unit/test_phase2_pool_admission.py
    - Strip purged_rolling_cv branch from data/splitter.py
    - Strip PHASE2_CV_*/CV_N_FOLDS/etc from config.py
    - Strip cv_fold branching from phase2_rule_pool.py and
      phase2_support.py (passes_pool_admission_cv_fold removed)
    - Strip cv_folds params from phase3_rule_set.py,
      phase4_wf_optimizer.py, phase5_oos.py
    - Strip _split_mode_is_purged_cv from evolution/evox_runner.py
    - Strip _cv_folds plumbing from run_pipeline.py
    - Strip SPLIT_MODE branch from backtest/gpu_engine.py
    - Strip purged CV tests from test_run_pipeline.py and
      test_evox_runner.py
    - Update docs/ to drop purged CV references
    - Update .opencode/CONTEXT.md task ledger
    "
    ```

18. **Write handoff JSON** to
    `.opencode/handoffs/task-11-implementer.json` with:
    - `branch`: `feature/task-11-remove-purged-cv`
    - `commit`: the implementer commit SHA
    - `summary`: file list removed + key sections stripped
    - `verification`: results of the imports / grep checks
    - `base_branch`: `main`

## Verification (run by the implementer before commit)

1. `git grep -n "purged\|PurgedFold" -- 'gpu_fuzzy_trader/' 'tests/' 'docs/'`
   returns **zero matches**.
2. `git grep -n "cv_folds\|cv_fold" -- 'gpu_fuzzy_trader/' 'tests/'`
   returns **zero matches** (after the cv_folds internal var and
   imports are gone).
3. `git grep -n "PHASE2_CV_\|PURGED_CV" -- 'gpu_fuzzy_trader/config.py'`
   returns **zero matches**.
4. `.venv/bin/python -c "import ast; ast.parse(open('gpu_fuzzy_trader/data/splitter.py').read())"`
   exits 0.
5. `.venv/bin/python -c "import ast; ast.parse(open('gpu_fuzzy_trader/config.py').read())"`
   exits 0.
6. `.venv/bin/python -c "import ast; ast.parse(open('gpu_fuzzy_trader/run_pipeline.py').read())"`
   exits 0.
7. `.venv/bin/python -c "import ast; ast.parse(open('gpu_fuzzy_trader/phases/phase2_support.py').read())"`
   exits 0.
8. `.venv/bin/python -c "import ast; ast.parse(open('gpu_fuzzy_trader/phases/phase2_rule_pool.py').read())"`
   exits 0.
9. `.venv/bin/python -c "from gpu_fuzzy_trader.data.splitter import Data_Splitter, split_and_persist; print('ok')"`
   exits 0.
10. `.venv/bin/python -c "from gpu_fuzzy_trader.run_pipeline import Pipeline_Orchestrator; print('ok')"`
    exits 0.
11. `.venv/bin/python -c "from gpu_fuzzy_trader.phases.phase2_support import _passes_pool_admission_impl, passes_pool_admission_gate, passes_pool_entry_admission, passes_pool_trade_floor; print('ok')"`
    exits 0.
12. Run the 4 small unit tests we can safely run with the user's
    RAM budget (one each from `test_data_splitter`,
    `test_run_pipeline`, `test_phase2_support`, `test_reporter`).
    Do **not** run the full suite (per AGENTS.md and the user's
    RAM limit).

## Risks & mitigations

- **Risk**: Hidden references in friend_project (committed code).
  **Mitigation**: the user explicitly said "my project"; we
  restrict `git grep` to `gpu_fuzzy_trader/`, `tests/`, `docs/`.
- **Risk**: Removing a function still called by an unrelated path.
  **Mitigation**: implementer runs the verification checks (#9–#11)
  + small unit tests.
- **Risk**: Removing a docstring paragraph that explained a real
  behaviour. **Mitigation**: keep all `holdout_70_30` paragraphs;
  only strip the `purged_rolling_cv` paragraphs.
- **Risk**: Changing `split_and_persist` return type breaks Phase 5
  (and any other call site). **Mitigation**: implementer does a
  grep for `split_and_persist` and updates every call site.
- **Risk**: Removing `Rule_Pool_Generator(cv_folds=...)` breaks
  tests. **Mitigation**: if any test passes `cv_folds=`, drop the
  kwarg (default is `None` / nothing).
- **Risk**: Removing `WalkForwardRiskOptimizer(cv_folds=...)` breaks
  tests. Same fix.

## Out of scope (deferred)

- Refactoring the `holdout_70_30` path itself.
- Improving the Phase 2 pool admission (the `PHASE2_CV_MIN_WORST_*`
  thresholds in the deleted section are gone — the holdout uses
  `PHASE2_POOL_TRAIN_RETURN_MIN_PCT`, `PHASE2_POOL_VAL_RETURN_MIN_PCT`,
  `PHASE2_PROFIT_FACTOR_FLOOR`).
- Re-running the pipeline (the user will re-run on their own).

## Target files

| File | Change |
|---|---|
| `gpu_fuzzy_trader/data/cv_folds.py` | **deleted** |
| `gpu_fuzzy_trader/validation/rolling_cv.py` | **deleted** |
| `tests/unit/test_phase2_pool_admission.py` | **deleted** |
| `gpu_fuzzy_trader/data/splitter.py` | strip purged branch, return 2-tuple, inline `holdout_70_30_split` |
| `gpu_fuzzy_trader/config.py` | strip `SPLIT_MODE`, `CV_*`, `PHASE2_CV_*`, `_cv_pool_*` |
| `gpu_fuzzy_trader/phases/phase2_rule_pool.py` | strip cv_fold branching, dead cv path |
| `gpu_fuzzy_trader/phases/phase2_support.py` | strip `passes_pool_admission_cv_fold`, cv_fold branches |
| `gpu_fuzzy_trader/phases/phase3_rule_set.py` | drop `cv_folds` param |
| `gpu_fuzzy_trader/phases/phase4_wf_optimizer.py` | drop `cv_folds` param, `build_phase4_multi_cv_fold_splits` |
| `gpu_fuzzy_trader/phases/phase5_oos.py` | 2-tuple destructure |
| `gpu_fuzzy_trader/evolution/evox_runner.py` | drop `_split_mode_is_purged_cv` |
| `gpu_fuzzy_trader/run_pipeline.py` | drop `_cv_folds` plumbing |
| `gpu_fuzzy_trader/backtest/gpu_engine.py` | drop `_phase2_trade_floor` SPLIT_MODE branch |
| `tests/unit/test_run_pipeline.py` | drop `PurgedFold` import + `TestPruneCvFoldsAfterPhase1` |
| `tests/unit/test_evox_runner.py` | drop `SPLIT_MODE = "purged_rolling_cv"` test |
| `docs/*.md` | strip purged CV mentions |
| `.opencode/CONTEXT.md` | add task 11 to ledger |
| `.opencode/handoffs/task-11-implementer.json` | **new** |
| `.opencode/tasks/task-11.md` | **new** |

## Workflow

- base_branch: `main`
- branch_policy: **isolated** (one branch per task)
- execution_mode: **checkpoint** (stop after implementer for user
  review)
- review flow: implementer → spec-reviewer → code-reviewer →
  user-confirmed merge
