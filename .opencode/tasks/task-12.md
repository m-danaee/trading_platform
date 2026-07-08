# Task 12: Gate Pareto-Collapse Warning to pareto_size >= 5

## Task ID
`task-12` (twelfth and FINAL task in the 2026-07-07 audit fix plan)

## Title
Gate Pareto-Collapse Warning to pareto_size >= 5

## Goal
Fix audit finding #13: the `Pareto collapse risk` warning at
`evolution/evox_runner.py:2740-2744` fires on degenerate 2-point
correlations. When the Pareto front has only 2 rules with
different objective values, the Pearson correlation is ±1.0 by
construction — a degenerate artifact, not a real collapse signal.
By gen 13, the correlation settles to -0.3 to -0.6 (healthy
sortino↔return tension), but the noisy gen-1 warnings are
misleading. Gate the warning to fire only when
`len(pareto_indices) >= 5` (a configurable threshold).

## Audit Citation
- Confirmed by static inspection:
  - `evolution/evox_runner.py:2736-2745` — the warning site (no
    pareto_size guard)
  - `config.py:676-678` — `PHASE2_OBJECTIVE_CORR_WARN_THRESHOLD = 0.9`
- Run log evidence (2026-07-07): "Pareto collapse risk" warnings
  fire at gen 1-2 when pareto size is 1-3 rules. By gen 13,
  the warning settles to no-fire (corr_f1_f3 = -0.3 to -0.6).

## Target Files
- `gpu_fuzzy_trader/evolution/evox_runner.py`
  - Lines 2736-2745: gate the warning on `len(pareto_indices)`.
    Skip the loop when pareto size is below the threshold.
  - Add a code comment explaining the gate (2-point correlation
    is degenerate).
- `gpu_fuzzy_trader/config.py`
  - Add `PHASE2_OBJECTIVE_CORR_MIN_PARETO_SIZE = 5` (default; the
    minimum pareto size before the warning fires).
  - Add `# → fixes audit finding #13` comment.
- `tests/unit/test_phase2_rule_pool.py` (or new
  `tests/unit/test_phase2_corr_warn_gate.py`)
  - Add a test asserting that the warning does NOT fire when
    pareto size < 5 (degenerate case).
  - Add a test asserting that the warning DOES fire when
    pareto size >= 5 with high correlation.
  - Add a test for the configurable threshold (e.g.,
    `PHASE2_OBJECTIVE_CORR_MIN_PARETO_SIZE=3` allows the
    warning to fire at pareto=3).

## Current Behavior
- `evolution/evox_runner.py:2736-2745`:
  ```python
  for corr_key in ("objective_corr_f1_f2", "objective_corr_f1_f3", "objective_corr_f2_f3"):
      corr_val = float(pareto_diag.get(corr_key, 0.0))
      if abs(corr_val) >= corr_threshold:
          logger.warning(
              "Phase 2 [%s] gen %d: %s=%.2f (Pareto collapse risk)",
              tag, gen + 1, corr_key, corr_val,
          )
  ```
  No pareto_size guard. With 2 rules in the Pareto front, the
  Pearson correlation is degenerate (trivially ±1.0 if the
  objectives differ on those 2 points).
- `config.py:676-678`:
  ```python
  PHASE2_OBJECTIVE_CORR_WARN_THRESHOLD = 0.9
  ```
  Only the correlation threshold is configurable; the minimum
  pareto size is not.

## Scope
1. **Gate the warning** (`evolution/evox_runner.py:2736-2745`):
   - Add a `min_pareto_size` check at the top of the loop:
     ```python
     # → fixes audit finding #13: 2-point Pearson correlations are
     # degenerate (trivially ±1.0 by construction when the 2
     # objectives differ). Only flag real collapse risk when the
     # Pareto front has enough rules for the correlation to be
     # statistically meaningful.
     min_pareto_size = int(getattr(
         _cfg, "PHASE2_OBJECTIVE_CORR_MIN_PARETO_SIZE", 5,
     ))
     if len(pareto_indices) < min_pareto_size:
         continue  # skip the warning loop
     for corr_key in (
         "objective_corr_f1_f2",
         "objective_corr_f1_f3",
         "objective_corr_f2_f3",
     ):
         corr_val = float(pareto_diag.get(corr_key, 0.0))
         if abs(corr_val) >= corr_threshold:
             logger.warning(
                 "Phase 2 [%s] gen %d: %s=%.2f (Pareto collapse risk, "
                 "pareto_size=%d)",
                 tag, gen + 1, corr_key, corr_val, len(pareto_indices),
             )
     ```
2. **Add new config flag** (`config.py`):
   - Add `PHASE2_OBJECTIVE_CORR_MIN_PARETO_SIZE = 5` near the
     existing `PHASE2_OBJECTIVE_CORR_WARN_THRESHOLD` (line 676).
   - Update the comment block to reference audit finding #13.
3. **Update log message** to include the pareto_size (already
   covered by the change above).
4. **Add audit-finding linkage**:
   - Add `# → fixes audit finding #13` comment at both sites.
5. **Do NOT change**:
   - The correlation computation itself (`pareto_diag` building
     and the Pearson calc).
   - The `PHASE2_OBJECTIVE_CORR_WARN_THRESHOLD` value (still 0.9).
   - Any other file outside `evolution/evox_runner.py`,
     `config.py`, and the test file.

## Acceptance Criteria
1. The warning at `evox_runner.py:2736-2745` is gated on
   `len(pareto_indices) >= PHASE2_OBJECTIVE_CORR_MIN_PARETO_SIZE`.
2. When pareto size < 5 (default), no `Pareto collapse risk`
   warning fires (the loop is skipped).
3. When pareto size >= 5 with |corr| >= 0.9, the warning fires
   (with the new `pareto_size=N` suffix in the message).
4. `PHASE2_OBJECTIVE_CORR_MIN_PARETO_SIZE = 5` is the new config
   default.
5. With `PHASE2_OBJECTIVE_CORR_MIN_PARETO_SIZE = 0` (or 1), the
   warning fires regardless of pareto size (regression guard for
   pre-task-12 behavior).
6. All existing tests pass: `test_phase2_rule_pool.py`,
   `test_phase2_window_rotation.py`, `test_phase2_island_scheduler.py`,
   `test_evox_runner.py`, etc.

## Verification
Run only related unit tests with `PYTEST_LOW_MEMORY=1` and `.venv`:

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_rule_pool.py -q
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_window_rotation.py -q
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_island_scheduler.py -q
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_evox_runner.py -q
```

If a new test file `test_phase2_corr_warn_gate.py` is created:
```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_corr_warn_gate.py -q
```

## Notes
- Do NOT modify `evaluator_v5.ipynb`.
- Do NOT run the full project or full test suite locally (OOM risk
  per AGENTS.md).
- This is a small surgical fix (~5 lines in source, ~5 lines in
  config, ~30 lines in 1 test file). Keep the diff minimal.
- This is the FINAL task of the 12-task audit fix plan. After
  this merges, the audit is fully addressed (all CONFIRMED and
  SUSPECTED findings except SUSPECTED S1 — the profit amplifier,
  which was deferred from the plan).
- The pareto_size threshold of 5 is a heuristic; a lower value
  (3) would also work. The implementer may use 5 as the spec
  suggests or 3 if there's a clear rationale.
- This task is purely cosmetic (no behavior change, only log
  noise reduction).
