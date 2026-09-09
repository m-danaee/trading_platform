---
name: gpu-fuzzy-trader
description: Develop, test, audit, or run the GPU Fuzzy Trader quantitative-research pipeline. Use for work on data, causal MTF features, Phase 2 rule discovery, RB governance, OOS evaluation, experiment artifacts, and dashboards in this repository; not for generic market advice or live trading.
---

# GPU Fuzzy Trader

Use this skill for repository work that can affect research evidence, strategy
identity, evaluator compatibility, or machine resources. Preserve the user's
scope. Treat every result as research evidence, not as trading advice or a
live-trading claim.

## Begin with the right boundary

- Read the relevant parts of [README.md](README.md) and [RUN.md](RUN.md).
  They define the current pipeline contract and supported commands.
- Inspect the worktree before editing. Preserve unrelated user changes and
  generated artifacts.
- For structural code work, follow the graph workflow in `AGENTS.md`: find the
  symbols and callers first, check index coverage, then read the exact source.
  Use `rg` for literals, configuration, notebooks, and other non-code files.
- Before changing a result-affecting path, map the raw data, labels, feature
  fitting, splits, gates, evaluator, output directory, and resume identity.
  Do this before proposing a performance fix.
- Make the smallest change that satisfies the request. Remove an old path only
  when it is proven unused or replaced, and update its focused tests.

## Project map

| Area | Main locations | Start with these tests |
| --- | --- | --- |
| Pipeline, run identity, artifacts | `gpu_fuzzy_trader/run_pipeline.py`, `gpu_fuzzy_trader/research_integrity.py`, `gpu_fuzzy_trader/config.py` | `test_run_pipeline.py`, `test_research_integrity.py`, `test_config_validation.py` |
| Data, labels, splits, caches | `gpu_fuzzy_trader/data/loader.py`, `gpu_fuzzy_trader/data/labels.py`, `gpu_fuzzy_trader/data/splitter.py` | `test_data_loader.py`, `test_data_splitter.py`, `test_purge_leakage.py` |
| Causal MTF | `gpu_fuzzy_trader/mtf/runtime.py`, `gpu_fuzzy_trader/mtf/cross_fitting.py`, `gpu_fuzzy_trader/mtf/discovery.py`, `gpu_fuzzy_trader/mtf/composer.py` | `test_multi_timeframe.py`, `test_mtf_*` |
| Rule discovery and simulation | `gpu_fuzzy_trader/features/`, `gpu_fuzzy_trader/phases/phase2_*`, `gpu_fuzzy_trader/evolution/`, `gpu_fuzzy_trader/backtest/` | `test_phase2_*`, `test_cpu_engine.py`, GPU tests only when relevant |
| Selection, release gates, OOS | `gpu_fuzzy_trader/rb_governor.py`, `gpu_fuzzy_trader/phases/phase5_oos.py`, `gpu_fuzzy_trader/validation/` | `test_rb_*`, `test_phase5_oos.py`, `test_optuna_search.py` |
| Reports and review UI | `gpu_fuzzy_trader/output/`, `gpu_fuzzy_trader/reporting/`, `gpu_fuzzy_trader/dashboard.py` | `test_output_writer.py`, `test_reporter.py`, `test_dashboard.py` |

`Pipeline_Orchestrator.run_from_phase2()` is the normal integrated path. It
loads and splits the train tape, builds the train-only catalog, runs Phase 2,
passes current-run candidates to the RB Governor, then runs Phase 5 only for
directions accepted in that run. Do not use a stale strategy file as a shortcut.

## Research-integrity rules

These rules are release-critical. Do not weaken them to obtain a better result.

1. Keep the data roles separate.
   - `train_new.csv` supports feature fitting and discovery.
   - The validation windows support selection and RB checks.
   - `test_new.csv` is a consumed Phase 5 diagnostic tape. It is never a
     tuning objective.
   - Only a strictly newer, untouched forward tape can support acceptance.

2. Keep time causality intact.
   - Derive labels and purges from the configured holding horizon.
   - Fit thresholds, scaling, and feature catalogs on the permitted train rows
     only, then freeze them for validation, test, and forward evaluation.
   - For MTF work, materialize causal higher-timeframe features first. A higher
     timeframe value may appear only after its candle has closed before the
     next low-timeframe execution point. Keep per-timeframe warm-up separate.
   - Treat supplied `ff_*` columns as external inputs. Audit their as-of-bar
     construction before making a release claim.

3. Preserve strategy and artifact identity.
   - Entry conditions, symbols, TP, SL, holding horizon, and the cost model
     form one strategy identity. RB may size capital; it must not rescue a
     rejected candidate by changing exit geometry.
   - A changed source tape, split, feature representation, schema, fold,
     threshold, archive, composer parameter, or horizon invalidates dependent
     caches and resume artifacts. Regenerate them rather than relabeling them.
   - Missing or rejected directions must remain explicit fail-closed outputs
     with `deployment_accepted: false`. Never substitute a fallback strategy
     or leave a stale accepted file in place.

4. Keep held-out resources single-use.
   - An Optuna study is sealed after it consumes its reserved tail. Use a new
     study and a new untouched tail before more optimization.
   - A forward tape is reserved once per output directory. Use a new, strictly
     later tape or a new output directory for another release candidate.

5. Preserve gate meaning.
   - Do not tune against final OOS or forward outcomes.
   - Do not relax concentration, trade-count, tail, cost, or uncertainty gates
     only because a candidate fails. State the failed gate and test a
     pre-specified research change instead.
   - Do not claim that a Phase 2, validation, RB, or consumed-test result is
     live readiness.

## Safe execution and experiments

- Use the repository virtual environment for every Python command:
  `.venv/bin/python ...`.
- On local or WSL hosts, run one memory-heavy process at a time. Do not use a
  full pipeline run as a routine smoke test. Long Phase 2 and RB work belongs
  on an appropriate CUDA host or the supported Colab path.
- Start result-affecting runs with a configuration preflight:

  ```bash
  .venv/bin/python -c "import gpu_fuzzy_trader.config as c; c.validate_config(); print('config OK')"
  ```

- Give each material experiment an explicit output directory. Use `--resume`
  only after checking that its identity matches the current data and settings.
- Keep CPU/GPU comparisons controlled. Large local windows normally use the
  CPU batch path; JAX/GPU is not a reason to change evaluator semantics.
- Use `main.ipynb` only for the documented Colab T4 workflow. Treat
  `evaluator_v5.ipynb` as the evaluator authority; do not change it unless the
  user explicitly asks to change the evaluator contract.
- The dashboard reads existing artifacts. It does not validate data or run the
  pipeline, so do not use a rendered dashboard as execution evidence.

## Change and validation workflow

1. Reproduce the reported behavior with the smallest safe fixture or command.
   For a confirmed defect, add a focused regression test that fails before the
   fix when practical.
2. Change the production path and its direct contract together. For example,
   a split or feature-identity change needs cache/resume coverage; an MTF
   change needs causal-alignment coverage; an RB or OOS change needs
   fail-closed coverage.
3. Run the directly affected test file immediately after each edit. On this
   host, use low-memory mode and an isolated Matplotlib cache:

   ```bash
   PYTEST_LOW_MEMORY=1 MPLCONFIGDIR=/tmp/trading-platform-mpl \
     .venv/bin/python -m pytest -q tests/unit/test_name.py
   ```

4. Run GPU-specific tests only when the edited path uses them, and keep them in
   a separate process. Do not run broad raw `pytest` commands on local WSL.
   If broad validation is requested, use `npm run test:all`; it runs the CPU
   and direct-JAX groups serially with the project virtual environment.
5. Run `git diff --check` after edits. Report the exact test command, terminal
   result, generated artifacts, and untested paths such as CUDA, full pipeline,
   untouched forward data, or live execution.

## How to report evidence

Use the strongest accurate label. Do not collapse these levels.

| Evidence | Valid conclusion | Not valid conclusion |
| --- | --- | --- |
| Focused tests and static checks | The changed contract passed its mechanical checks. | The strategy generalizes. |
| Train, validation, Phase 2, or RB | The candidate met development-stage criteria. | OOS, deployment, or live success. |
| Phase 5 on `test_new.csv` | Diagnostic evidence on a consumed holdout. | Forward acceptance or live readiness. |
| Untouched forward tape passes all gates | Forward acceptance for that candidate and tape. | A live-trading forecast. |
| Live or paper execution with realistic controls | Operational evidence within that exact setup. | General performance outside that setup. |

For an experiment or audit, report the data role, dates or hashes, split and
purge geometry, cost model, direction and symbol coverage, gate outcomes,
artifact paths, and limitations. Report long and short results separately when
both directions are in scope.

## Completion standard

Before handoff, confirm that the requested change is present, the directly
affected tests passed, and no old implementation remains without a purpose.
Separate confirmed fixes from risks and untested acceptance layers. Do not
place orders, connect an exchange, or describe research output as financial
advice without explicit user instruction and a separate safety review.
