# Task 11 — Remove purged CV feature completely

## Goal

Completely remove the "purged CV" (`purged_rolling_cv` / `PurgedFold` /
`PHASE2_CV_*` / `cv_folds`) feature from the project — **zero leftovers**
in source, tests, docs, or comments — and keep the `holdout_70_30`
path (current default) working unchanged.

The full plan is in `.opencode/plans/PLAN.md` (task 11 section).
The CONTEXT.md is in `.opencode/CONTEXT.md`.

## Background

The purged rolling cross-validation feature was a parallel evaluation
path for Phases 2–4 that was added in earlier tasks. The project's
default `SPLIT_MODE` has been `"holdout_70_30"` for some time, so the
entire purged CV code path is **dead code in production** (gated by
`SPLIT_MODE == "purged_rolling_cv"` and only exercised by dedicated
unit tests). The user wants it gone to simplify the codebase.

## Branch

- base_branch: `main`
- feature branch: `feature/task-11-remove-purged-cv`
- branch_policy: isolated (one branch per task)
- execution_mode: checkpoint (stop after implementer for user review)

## Workflow

1. Implementer creates the branch and does the removal (one commit).
2. Spec-reviewer verifies the removal is complete (zero leftovers).
3. Code-reviewer checks for broken imports / dead code.
4. User is asked for confirmation before merging to `main`.

## Acceptance criteria

1. **No `purged` references in code/tests/docs (except the friend
   project)** — `git grep -n "purged\|PurgedFold" -- 'gpu_fuzzy_trader/' 'tests/' 'docs/'`
   returns **zero matches**.
2. **No `cv_folds` instance state or imports remain** — `git grep -n
   "_cv_folds\|cv_folds" -- 'gpu_fuzzy_trader/' 'tests/'` returns
   **zero matches**.
3. **No `PHASE2_CV_*` / `PURGED_CV` config keys remain** — `git
   grep -n "PHASE2_CV_\|PURGED_CV" -- 'gpu_fuzzy_trader/config.py'`
   returns **zero matches**.
4. **Files removed**:
   - `gpu_fuzzy_trader/data/cv_folds.py` (deleted)
   - `gpu_fuzzy_trader/validation/rolling_cv.py` (deleted)
   - `tests/unit/test_phase2_pool_admission.py` (deleted)
5. **The `holdout_70_30` path still works**:
   - `python -c "from gpu_fuzzy_trader.data.splitter import Data_Splitter; print('ok')"` works
   - `python -c "from gpu_fuzzy_trader.run_pipeline import Pipeline_Orchestrator; print('ok')"` works
   - `python -c "from gpu_fuzzy_trader.phases.phase2_support import _passes_pool_admission_impl, passes_pool_admission_gate, passes_pool_entry_admission, passes_pool_trade_floor; print('ok')"` works
   - The 4 small unit tests we can safely run pass.
6. **`SPLIT_MODE` constant is removed** (no longer a configurable
   option — the project only has one path).
7. **No dead imports, no dead functions, no dead branches** left
   behind.
8. **Friend project is untouched**.

## Steps for the implementer

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
   - Remove `SPLIT_MODE` constant and its comment block.
   - Remove `CV_FOLDS_MANIFEST_PATH`, `CV_N_FOLDS`, `CV_EMBARGO_BARS`,
     `CV_BARS_PER_DAY`, `CV_MIN_TRAIN_MONTHS` and their comment
     blocks.
   - Remove `_cv_pool_min_folds_pass()`, `_CV_POOL_MIN_FOLDS_PASS`,
     `_CV_RANK_MIN_FOLDS_PASS`.
   - Remove the entire "Purged CV pool admission" section.
   - Remove the "Phase 2 relaxed pool-admission thresholds (Task 5)"
     block (PHASE2_CV_MIN_WORST_RETURN, PHASE2_CV_MIN_WORST_PF,
     PHASE2_CV_MAX_WORST_DD, PHASE2_CV_MIN_FOLD_TRADES).
   - Remove `PHASE2_CV_FOLD_WORKERS`,
     `PHASE2_EARLY_STOP_DISABLED_IN_CV`,
     `PHASE2_PLATEAU_EARLY_STOP_DISABLED_IN_CV`.
   - Remove the tuning cheat-sheet rows that mention
     CV_N_FOLDS / PHASE2_CV_* / SPLIT_MODE.
   - Remove the `assert CV_N_FOLDS ...`, `assert 1 <= PHASE2_CV_*`
     asserts at the bottom.
   - Update the module docstring if it mentions SPLIT_MODE / CV.

4. **Strip `gpu_fuzzy_trader/data/splitter.py`**:
   - Inline `holdout_70_30_split` (from `cv_folds.py`) into this
     file (it was already imported from cv_folds).
   - Remove the `PurgedFold`, `build_purged_rolling_folds`,
     `primary_holdout_from_folds` imports.
   - Change `split_and_persist` to return
     `tuple[pd.DataFrame, pd.DataFrame]` (drop the third element).
   - Remove the `purged_rolling_cv` mode branch from
     `split_and_persist`.
   - Remove `build_cv_folds` method.
   - Remove `load_cv_folds_from_manifest` static method.
   - Remove `_persist_cv_manifest` static method.
   - Remove module-level `build_cv_folds` function.
   - Update the class and method docstrings to drop the purged CV
     mention.
   - **Update every caller of `split_and_persist`** (Phase 5 / run_pipeline)
     to use the 2-tuple return.

5. **Strip `gpu_fuzzy_trader/phases/phase2_rule_pool.py`**:
   - Replace `_val_trade_floor_for_objectives()` with an inline call
     `max(int(_cfg.MIN_TRADE_POOL_FLOOR) // 4, 10)`.
   - Remove the `cv_fold = str(_cfg.SPLIT_MODE).strip().lower() ==
     "purged_rolling_cv"` lines; always pass `cv_fold=False`.
   - Remove `_uses_cv_engines()` (or inline it as `False`).
   - In `_build_engines()`, remove the cv branch and the
     `from gpu_fuzzy_trader.phases.phase2_cv import
     build_cv_fold_engines` import (phase2_cv.py never existed).
   - Remove the `use_cv_admission = False` block (and the entire
     `if use_cv_admission:` body).
   - Remove the `evaluate_purged_cv_pool_admission_batch` reference.
   - Update `Rule_Pool_Generator.__init__` to drop the `cv_folds`
     parameter (keep kwargs backward compatible — accept but ignore
     is OK; or drop outright).
   - Search for any other `cv_fold` / `cv_folds` / `purged` reference
     in this file and remove it.

6. **Strip `gpu_fuzzy_trader/phases/phase2_support.py`**:
   - Remove `_split_mode_is_purged_cv()`.
   - In `_pool_admission_floors`, drop the `cv_fold` parameter and
     always return the holdout tuple.
   - In `_passes_pool_admission_impl`, drop the `cv_fold` parameter;
     remove the `cv_fold and _split_mode_is_purged_cv()` drawdown
     checks.
   - Remove the entire `passes_pool_admission_cv_fold()` function.
   - In `passes_pool_entry_admission`, remove the
     `_split_mode_is_purged_cv()` branch and the
     `cv_folds_passing` / `cv_folds_total` check.
   - In `passes_pool_trade_floor`, remove the `if
     _split_mode_is_purged_cv():` trade-floor switch.
   - Search for any other `cv_fold` / `purged` reference and remove
     it.

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
      `orch._cv_folds[0]`.

14. **Strip `tests/unit/test_evox_runner.py`**:
    - Remove the test that sets `SPLIT_MODE = "purged_rolling_cv"`.

15. **Strip `docs/`**:
    - `docs/phase0_shared.md`: remove the "A. Purged rolling CV"
      section, the "SPLIT_MODE" row, the "CV_FOLDS_MANIFEST_PATH"
      row, and the "SPLIT_MODE | `purged_rolling_cv`" recommendation
      paragraphs.
    - `docs/phase2_rule_pool.md`: remove the "Purged CV" section and
      any `PHASE2_CV_*` mentions.
    - `docs/phase3_rule_set.md`: remove the "Purged CV evaluation"
      section.
    - `docs/phase4_wf_risk.md`: remove the "Relationship to Phase 2/3
      purged CV" section.
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
    Do **not** run the full suite.

## Out of scope

- Refactoring the `holdout_70_30` path itself.
- Re-running the pipeline.
- Touching `friend_project/` (separate committed reference project).
