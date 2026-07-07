# Task 8: Use val_df in Phase 1 Sign Consistency Filter

## Task ID
`task-8` (eighth of 12 tasks in the 2026-07-07 audit fix plan)

## Title
Use val_df in Phase 1 Sign Consistency Filter

## Goal
Fix audit finding #11: `_check_spearman_sign_consistency(df, ...,
val_df=None, ...)` in `features/selector.py:219` accepts a
`val_df` parameter but never reads it. The function body only uses
`df` (the train df) for fold-based Spearman correlation. A
feature whose train sign is stable but whose val sign is flipped
silently survives the Phase 1 sign-consistency gate. Either
actually use the val_df to add a val-side sign check, or remove
the dead parameter.

This task implements Option (a): actually use the val_df to
add a val sign check. The val sign must match the train majority
sign; if not, the feature is blacklisted.

## Audit Citation
- Confirmed by static inspection:
  - `features/selector.py:219` — function signature accepts `val_df`
  - `features/selector.py:228-258` — function body only uses `df`
    (the train df), never reads `val_df`
  - `features/selector.py:363` — call site passes `val_df=val_df`
    but the value is discarded
- Run log evidence: no direct log evidence (this is a silent
  pass-through bug — the function works correctly on train, but
  the val-side check is missing).

## Target Files
- `gpu_fuzzy_trader/features/selector.py`
  - In `_check_spearman_sign_consistency` (line 219-258):
    - After computing the train fold sign, compute the val
      Spearman correlation if `val_df is not None`.
    - If the val sign disagrees with the train majority sign
      (i.e., train is all positive, val is negative, or vice
      versa), blacklist the feature.
    - Update the docstring to describe the new behavior.
  - In the call site at line 363: no change needed (already
    passes `val_df=val_df`).
- `tests/unit/test_feature_selector.py`
  - Add a test asserting that a feature with consistent train
    signs but flipped val sign is NOW blacklisted.
  - Add a test asserting that a feature with consistent train
    signs and matching val sign is still kept.
  - Add a test asserting that when `val_df=None` (the default),
    behavior matches the pre-task-8 behavior (regression guard).

## Current Behavior
- `features/selector.py:228-258`: the function loops over
  `_get_spearman_folds(df, n_folds)` (train folds only). It
  computes Spearman correlation for each fold, then checks if
  significant correlations are all positive or all negative
  (the "stable sign" check).
- The `val_df` parameter is declared in the signature (line 224)
  but never read in the body. The function returns the same
  `stable_features` set whether or not `val_df` is provided.
- At the call site (line 363): `stable_cols = _check_spearman_sign_consistency(
  train_df, feature_cols, n_folds, min_folds, val_df=val_df)`. The
  `val_df` value is passed but discarded.

## Scope
1. **Use val_df in the sign check** (features/selector.py:228-258):
   - After the train-folds loop (after computing `corrs`,
     `significant`, `has_pos`, `has_neg`), add a val-side check:
     ```python
     if val_df is not None and has_pos and not has_neg:
         # Train is all positive; check val
         val_corr = _spearman(val_df[col], val_df[label_col])
         if not np.isnan(val_corr) and val_corr < -min_abs_corr:
             # Val sign disagrees; blacklist
             logger.info("Blacklisting non-stationary feature %s: train sign positive, val sign negative (val_rho=%.3f)", col, val_corr)
             continue  # don't add to stable_features
     elif val_df is not None and has_neg and not has_pos:
         # Train is all negative; check val
         val_corr = _spearman(val_df[col], val_df[label_col])
         if not np.isnan(val_corr) and val_corr > min_abs_corr:
             # Val sign disagrees; blacklist
             logger.info("Blacklisting non-stationary feature %s: train sign negative, val sign positive (val_rho=%.3f)", col, val_corr)
             continue
     ```
   - The `min_abs_corr` threshold ensures we only check against
     significant val correlations (consistent with the train-side
     logic).
   - When `val_df is None` (e.g., no val provided at init), the
     val check is skipped — preserves the pre-task-8 behavior.
2. **Update the docstring** (features/selector.py:228-232):
   - Add a new paragraph: "When `val_df` is provided, the val
     Spearman correlation is also computed. If the val sign
     disagrees with the train majority sign (above
     `min_abs_corr` threshold), the feature is blacklisted.
     This catches features that have a stable train sign but
     fail on val (silent OOS leak)."
3. **Add audit-finding linkage**:
   - Add `# → fixes audit finding #11 (val_df was dead parameter;
     now actually checks val sign consistency)` in the function
     docstring.
4. **Do NOT change**:
   - The `_get_spearman_folds` function (correct as-is).
   - The `_spearman` function (correct as-is).
   - Any other file outside `features/selector.py` and the test.

## Acceptance Criteria
1. A feature with consistent train signs (all positive) but
   val sign negative (val_rho < -min_abs_corr) is NOW blacklisted
   (was admitted previously).
2. A feature with consistent train signs (all positive) and
   matching val sign positive is still kept.
3. A feature with consistent train signs and tiny val
   correlation (|val_rho| < min_abs_corr) is still kept
   (the val check requires the val corr to be significant
   to override the train sign).
4. When `val_df=None` (the default), behavior matches
   pre-task-8 exactly (regression guard).
5. When `val_df` is provided but doesn't have `label_close_288`
   column (degenerate case), the val check is skipped
   (no crash).
6. The function's signature is unchanged (backward compat).
7. All existing `test_feature_selector.py` tests pass.

## Verification
Run only related unit tests with `PYTEST_LOW_MEMORY=1` and `.venv`:

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_feature_selector.py -q
```

## Notes
- Do NOT modify `evaluator_v5.ipynb`.
- Do NOT run the full project or full test suite locally (OOM risk
  per AGENTS.md).
- This is a small surgical fix (~15 lines of source code, ~50
  lines in 1 test file). Keep the diff minimal.
- The val-side check uses the same `_spearman` helper as the
  train-side; no new dependencies.
- The val_df may be much smaller than the train_df; the
  Spearman correlation is still well-defined as long as
  both have at least ~30 rows.
- This task is the cleanup twin of task-4 (monthly gate on val)
  and task-5 (delete dead f3 branch). All three are small
  fixes with no behavior change in default config (when
  val_df is provided, which is the standard case).
