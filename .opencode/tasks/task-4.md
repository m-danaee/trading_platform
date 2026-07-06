# Task 4: Make Sampling And Migration Semantics Honest

## Task ID
`task-4`

## Title
Make Sampling And Migration Semantics Honest

## Goal
Align logs/config comments with actual island migration behavior and reduce train/validation sampling coupling. The run log currently says migration interval/epoch semantics, but the sequential scheduler performs one-way post-cluster migration. Also, train and validation sampling currently use the same deterministic seed, which can select matching relative regimes in train and validation windows.

## Target Files
- `gpu_fuzzy_trader/phases/phase2_rule_pool.py`
- `gpu_fuzzy_trader/phases/phase2_island_scheduler.py`
- `gpu_fuzzy_trader/config.py`
- Related tests under `tests/unit/`, especially `tests/unit/test_phase2_rule_pool.py` and `tests/unit/test_phase2_island_scheduler.py` if present.

## Evidence From Run Log / Analysis
- Log reports `island mode migration=enabled (interval=1, top_k=5, seed_frac=0.10)`.
- Active scheduler in `phase2_island_scheduler.py` processes clusters sequentially and migrates only after a cluster finishes, forwarding to the next cluster. The existing `_should_migrate_this_round()` helper is not used in this active path.
- `Rule_Pool_Generator._build_engines()` samples validation with `random_state=self._sample_seed`, same as train sampling, which risks selecting matching relative windows across train/validation splits.
- `_sample_df()` contiguous per-symbol sampling is directionally correct, but warnings should make no-op/full-safe-range sampling clear.

## Scope
- Update logging and config comments so current sequential post-cluster migration is not described as epoch-round/interval migration unless actually implemented.
- If small and safe, wire interval semantics into the sequential scheduler; otherwise prefer honest logging/comments over broad scheduler rewrite.
- Make train and validation sampling seeds distinct but deterministic by default.
- Keep existing contiguous per-symbol sampling deterministic.
- Add timestamp-alignment safeguards only if small and well-tested; otherwise document current bar-index alignment limitations in comments/logging without broad engine changes.
- Ensure `_sample_df()` warnings remain useful and make it clear when the requested sample exceeds the safe range and effectively uses all available safe bars.

## Non-Goals
- Do not implement a large round-robin island scheduler rewrite unless trivial and fully tested.
- Do not change objective construction; Task 3 handled that.
- Do not relax feasibility gates.
- Do not modify `evaluator_v5.ipynb`.
- Do not run the full project or full test suite locally.

## Acceptance Criteria
- Logs accurately describe whether migration is post-cluster chain migration or epoch-round migration.
- Train and validation sampling windows are not selected with the identical RNG seed by default.
- Existing contiguous per-symbol sampling behavior remains deterministic.
- Related sampling/scheduler tests pass.
- No evaluator notebook changes.

## Verification
Run only related tests, for example:

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_rule_pool.py tests/unit/test_phase2_island_scheduler.py -q
```

If a test file does not exist or a narrower target is appropriate, use available related tests and report exactly what ran.
