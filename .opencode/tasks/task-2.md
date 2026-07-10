# task-2: Hard-require min distinct symbols on final output (P0)

## Goal
Reject final RB teams that fail `RB_MIN_DISTINCT_SYMBOLS` (e.g. 1-rule
HHI=1.0 teams) instead of only blocking *adds* during compose.

## Context
`_compose_ruleset` uses `RB_MIN_DISTINCT_SYMBOLS` only when considering
whether a candidate adds a new symbol; a single-rule seed is still
accepted. Diagnosis: long/short shipped 1-symbol teams with
`deployment_accepted=false` but Phase 5 still ran them.

## Scope

### rb_governor.py
After compose + risk optimization (when final `opt_rules` is known), if:
- `RB_REQUIRE_SYMBOL_FILTERS` is True, AND
- `RB_MIN_DISTINCT_SYMBOLS > 0`, AND
- `len(_symbols_in_rules(opt_rules)) < RB_MIN_DISTINCT_SYMBOLS`

Then fail closed:
- Replace with empty `rules_set` strategy via `_strategy`
- `deployment_accepted=false`
- reason `insufficient_distinct_symbols`
- Include `n_symbols` and `required` in extra/report
- Populate `results[direction] = strategy` (same as task-1 fail-closed)
- Write strategy JSON + report
- Skip further processing for that direction if needed, or write final empty

When `RB_REQUIRE_SYMBOL_FILTERS` is False, skip this gate entirely
(cross-symbol rules have no `symbol is X` conditions).

Prefer applying the gate on the **final** selected rules (after risk opt
and profit amp) so the output is what would have been shipped.

### Tests
- New tests in `tests/unit/test_rb_min_symbols.py` or extend existing.
- Cases:
  1. Filters on + 1-symbol team + min=5 → empty rules, reason set
  2. Filters on + enough symbols → not rejected by this gate alone
  3. Filters off → gate skipped even with 1 symbol-less or 1-symbol rules
- Use mocks; no full pipeline.
- `.venv` + `PYTEST_LOW_MEMORY=1` + `PYTHONPATH=.`

## Out of scope
- Phase 5 equity (task-3)
- Config anti-leak / pool knobs (task-4/5)
- evaluator_v5, outputs/

## Acceptance
- [ ] Final 1-rule `symbol is X` team rejected when min=5 and filters on
- [ ] Multi-symbol team meeting min not rejected by this gate alone
- [ ] Gate skipped when `RB_REQUIRE_SYMBOL_FILTERS=False`
- [ ] Empty fail-closed strategy still loadable (deployment_accepted=False)
- [ ] Tests pass; commit on feature branch only

## Branch
- base: main
- feature: feature/task-2-min-distinct-symbols
