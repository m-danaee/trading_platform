# Task 1: Stage 1 — Mechanical fixes (5 items)

## Source plan
`/home/danaee/.claude/plans/you-are-a-senior-pure-cupcake.md` — Stages 1, items 1-5

## Branch
`fix/phase2-stage1-mechanical` (from `main`)

## Files to touch
- `gpu_fuzzy_trader/config.py`
- `gpu_fuzzy_trader/data/splitter.py`
- `gpu_fuzzy_trader/validation/rolling_cv.py`
- `gpu_fuzzy_trader/phases/phase2_rule_pool.py`
- `gpu_fuzzy_trader/evolution/evox_runner.py`
- `gpu_fuzzy_trader/run_pipeline.py`
- `gpu_fuzzy_trader/phases/phase2_island_scheduler.py`
- `tests/unit/test_phase2_island_scheduler.py`

## Changes

### Item 1: SPLIT_MODE stale label rename
- `config.py:180`: rename `SPLIT_MODE = "holdout_70_30"` → `SPLIT_MODE = "holdout"`.
- `config.py:176-180`: update comment block to describe new mode name + the percentage-agnostic selector.
- `config.py:193`: update "SPLIT_MODE is now `holdout_70_30`" → "SPLIT_MODE is now `holdout`".
- `data/splitter.py:6`: update module docstring `holdout_70_30` → `holdout`.
- `validation/rolling_cv.py:485`: change `"holdout_70_30"` default → `"holdout"`.
- **Grep the entire repo** for `"holdout_70_30"` literal and update every comparison site atomically. The string is a functional selector (compared via `==`), so do not leave any sites stale.
- Anywhere the actual percentages are logged/printed, compute them from `HOLDOUT_TRAIN_FRACTION` (e.g. `f"{int(frac*100)}/{int((1-frac)*100)}"`) instead of baking them into a literal string.
- Add a tiny helper in `config.py` (or a sensible location) like `def holdout_train_val_label(frac: float) -> str:` so both call sites compute it the same way.

### Item 2: Stale docstring
- `phases/phase2_rule_pool.py:14-17`: Update `f3 = -win_rate` to reflect that `f3` is driven by `PHASE2_F3_OBJECTIVE` (default `"profit_factor"`), matching the accurate comment at `config.py:495-505`.

### Item 3: corr_f1_f3 warning level
- `evolution/evox_runner.py:2660-2673`: Change `logger.debug(...)` for the correlation warning to `logger.warning(...)` (recommended per plan) so it's visible by default. The correlation value itself is already in every INFO-level progress line; this just makes the interpretive warning match.

### Item 4: Shared helper for generation budgets
- `phases/phase2_island_scheduler.py`: Extract a module-level helper:
  ```python
  def compute_cluster_generation_budgets(
      total_gens: int, n_clusters: int,
  ) -> dict[int, int]:
      """Split total generation budget across clusters (base + remainder)."""
      k = max(1, n_clusters)
      base = max(1, total_gens // k)
      remainder = total_gens % k
      return {
          idx: base + (1 if idx < remainder else 0)
          for idx in range(k)
      }
  ```
  (use cluster IDs as keys — pass them in or use a separate helper that takes cluster_ids + total_gens).
- Update `_run_cluster_islands` to use the new helper instead of inlining the arithmetic.
- Update `run_pipeline.py:_log_pipeline_config` (lines 227-242) to use the same helper.
- While touching `_log_pipeline_config`, rename log key `per_cluster=` → `per_cluster_gens=` so it doesn't read as a population split.
- **Update tests** `tests/unit/test_phase2_island_scheduler.py::test_gens_per_cluster_split` and `test_epoch_rounds_cover_budget` to exercise the new helper directly (assert on its output, not reimplement the formula).

### Item 5: PHASE2_MIGRATION_ENABLED stale comment
- `config.py:1098-1108`: Rewrite the comment block to describe that migration is **enabled** with guarded re-evaluation (mention the receiver-side re-evaluation + admission gates in `filter_migrants_for_cluster`). Do not narrate the old failure mode as if still current. Keep `PHASE2_MIGRATION_ENABLED: bool = True`.
- Flag in the implementer handoff that you considered the "revert to False" alternative and chose the rewrite-comments path per the plan's recommendation.

## Acceptance criteria
- [ ] All SPLIT_MODE comparison sites use `"holdout"` (not `"holdout_70_30"`)
- [ ] No literal "70/30" or "65/35" in log strings; percentages derive from `HOLDOUT_TRAIN_FRACTION`
- [ ] `phases/phase2_rule_pool.py:14-17` docstring reflects `PHASE2_F3_OBJECTIVE` semantics
- [ ] `corr_f1_f3` warning logs at WARNING (or INFO at minimum)
- [ ] `compute_cluster_generation_budgets` exists in `phase2_island_scheduler.py` and is used by both `_run_cluster_islands` and `_log_pipeline_config`
- [ ] Log line uses `per_cluster_gens=` (not `per_cluster=`)
- [ ] `PHASE2_MIGRATION_ENABLED` comment narrates current state (re-enabled with guards)
- [ ] `test_gens_per_cluster_split` and `test_epoch_rounds_cover_budget` exercise the new helper directly
- [ ] Touched test suites pass with `PYTEST_LOW_MEMORY=1`:
  ```
  PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_island_scheduler.py tests/unit/test_phase2_rule_pool.py tests/unit/test_phase2_support.py tests/unit/test_migration_safety.py tests/unit/test_island_scheduler_migration.py tests/unit/test_data_splitter.py -v
  ```

## Hard rules
- Do not change behavior. All items here are label/comment/structural only.
- Do not touch any other files outside the listed ones.
- Do not run the full test suite. Only the suites listed above.
- Use `PYTEST_LOW_MEMORY=1` for any test run.
- Use `.venv/bin/python` for any test command.
- One atomic commit per item, OR one consolidated commit for the whole task if the changes are tightly coupled.
- Commit message prefix: `fix(task-1): <item summary>`.
- Do NOT push to remote, do NOT merge to main. The orchestrator will handle merging.
