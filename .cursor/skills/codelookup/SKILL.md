---
name: codelookup
description: Scans Python import dependencies, traces callers and dependents, and maps blast radius before code changes. Use before editing, creating, or deleting functions, classes, files, variables, config, or tests — or when the user asks about impact analysis, blast radius, cascade updates, or what might break from a change.
---

# CodeLookup

Always check impact before changing the codebase. Code changes can break dependent systems. Follow this checklist.

## Action Protocol

BEFORE editing, creating, or deleting code (functions, classes, files, variables, tables, config):

1. **Run Dependency Pre-Check**:
   - If `.codelookup/graph.json` is missing or stale (new/moved modules), run:
     ```bash
     .venv/bin/python .cursor/skills/codelookup/scripts/generate_graph.py
     ```
   - Then run:
     ```bash
     .venv/bin/python .cursor/skills/codelookup/scripts/pre_check.py
     ```
   - Always output the generated Mermaid blast-radius flowchart in your response analysis.

2. **Verify Callers**:
   - Inspect files identified by the pre-check tool.
   - Trace contract, return type, schema, or signature changes.
   - For `gpu_fuzzy_trader/config.py` changes, check phase modules and tests that read those knobs.
   - For phase module changes, check `run_pipeline.py`, downstream phases, and `tests/unit/test_*` mirrors.

3. **Plan Cascade Updates**:
   - Ensure all affected callers and linked systems are modified in the same change step. No partial commits.
   - Remove wasted parts from old implementations after cascade updates (per `AGENTS.md`).

4. **Run Integration Tests**:
   - Run tests covering both the modified file and dependent files:
     ```bash
     PYTEST_LOW_MEMORY=1 .venv/bin/pytest <related test paths> -q
     ```
   - Do not run the full test suite or full pipeline on WSL (OOM risk).

## Manual fallback

If scripts fail (no git repo, parse error), manually:

1. `rg` for imports of the changed module/symbol.
2. Grep tests referencing the symbol or config knob.
3. Draw a Mermaid `flowchart TD` from dependents → changed module.

## Trading-platform hotspots

| Change area                  | Likely dependents                                                    |
| ---------------------------- | -------------------------------------------------------------------- |
| `gpu_fuzzy_trader/config.py` | All `phases/*`, `run_pipeline.py`, many `tests/unit/test_config*.py` |
| `phases/phase2_*`            | `phase3_*`, `phase4_*`, evolution/numba_ops, benchmark tests         |
| `scoring/*`                  | Phase 3/4/5, reporter, validation                                    |
| `features/*`                 | Phase 1, encoder, selector tests                                     |

## Composes with improve

When executing an **improve** plan, run codelookup before every source edit and include blast-radius analysis in the implementation report.
