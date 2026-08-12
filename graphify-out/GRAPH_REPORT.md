# Graph Report - trading_platform  (2026-08-12)

## Corpus Check
- 188 files · ~264,375 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 4974 nodes · 11123 edges · 199 communities (190 shown, 9 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 563 edges (avg confidence: 0.57)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `2bac6a5a`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- evox_runner.py
- Rule_Pool_Generator
- CPUBacktestEngine
- _make_engine
- .run
- trend_context.py
- test_evox_runner.py
- _score_metrics
- _get_dont_cares
- _make_df
- phase2_sparse_encoding.py
- run_phase2_evolution
- test_phase2_support.py
- test_phase5_oos.py
- Pipeline_Orchestrator
- _apply_monthly_admission_gate
- prop_settings
- _make_train_df
- _split
- test_cpu_engine_properties.py
- detect_feature_mode
- selector.py
- Feature_Selector
- rolling_cv.py
- TestWritePerSymbolCsv
- test_encoder_properties.py
- write_evaluator_clean
- CandidateRecord
- nested_walk_forward.py
- _preserve_deployable_elites
- maybe_log_generation
- Hybrid CPU and GPU execution policy
- TestSelectDiverseSubset
- test_rb_min_symbols.py
- Graphify Pipeline
- Output_Writer
- config.py
- TestEquityCurveDateAxis
- test_phase2_window_rotation.py
- research_integrity.py
- _loader_from_rows
- _symbol_specialized_variants
- gpu_engine.py
- _gpu_runtime.py
- ._ensure_dir
- _derive_island_seed
- test_certificate_first_selection.py
- non_dominated_sort
- optuna_search.py
- phase5_oos.py
- test_phase2_rule_pool_properties.py
- _compute_rule_signal_mask
- validate_config
- build_hybrid_symbol_clusters
- test_phase2_init.py
- TestInitPopulation
- Feature_Detector
- test_reporter.py
- GPUBacktestEngine
- TestGPUCPUNumericalParity
- ValueError
- ._build_engine_for_df
- ._build_per_symbol_rows
- TestRulePoolGeneratorRun
- test_plateau_state_leak.py
- test_phase5_oos_properties.py
- dashboard.py
- test_data_splitter_properties.py
- test_data_loader_properties.py
- barrier_column_names
- _m
- _apply_dynamic_rule
- test_cpu_engine.py
- test_rb_fail_closed.py
- execution_ok
- TestWriteStrategyEvaluationTable
- test_gpu_engine.py
- test_crash_fix_and_run_logging.py
- TestPrecomputeReleaseIndices
- test_feature_selector_properties.py
- loader.py
- compute_labels
- TestPlotPerRuleBreakdown
- resolve_island_hyperparams
- BUG_HUNT_REPORT.md
- _build_target
- filter_migrants_for_cluster
- test_gpu_engine_properties.py
- writer.py
- _make_rule
- Data_Splitter
- set_purged_wf_reference_rows
- downcast_numeric_df
- .encode_condition
- test_output_writer_properties.py
- Data_Loader
- get_gpu_backtest_engine_class
- .decode_chromosome
- .load_strategies
- gate_positive_good
- phase2_island_scheduler.py
- .save_archive
- TestRemoveRedundantFeatures
- TestEquityCurvePlots
- test_crash_fix_properties.py
- Reporter
- baselines.py
- phase2_rule_pool.py
- _jax_compute_trade_outcomes
- _should_post_restart_early_stop_phase2
- _write_and_reload
- resolve_phase2_stage_params
- _make_df
- .run
- TestHammingThresholdAutoScale
- TestValLeakGate
- ._simulate_rule_set_entries
- TestComputeRuleSignals
- TestParetoCollapseWarningGate
- TestMakeWalkForwardFoldEngines
- _compute_stability
- ._prune_splits_after_phase1
- MonthlyWindowSummary
- ._engine
- test_rb_concentration_tail_fail_closed.py
- constrained_non_dominated_sort
- test_feature_selector.py
- ._load_datasets_by_split
- stratification_scenario_strategy
- TestSpearmanSignConsistency
- run_pipeline.py
- log_memory_rss
- TestPlotPhase2Metrics
- _apply_colab_gpu_defaults
- splitter.py
- phase2_support.py
- OOS_Evaluator
- compute_phase2_objectives_from_metrics
- BUG-003 — CPU time-exit metric depended on log generation
- test_evaluator_clean_writer.py
- TestMigrationSeedFraction
- _run_cluster_islands
- _pool_admission_floors
- TestPlotPhase2Pnl
- TestRefreshObjectivesOnResumeGate
- conftest.py
- context_contract_digest
- .skip_if_valid
- _validate_schema
- ._record_research_integrity
- TestDeferredWarmup
- .skip_if_valid
- BFS and DFS Graph Traversal
- TestModePriority
- BUG-001 — Split cache trusted mtime instead of source contents
- TestSparsePositiveMode
- TestZeroRatioBoundary
- TestCausalPublicationTiming
- TestGlobalMetricsCacheClearing
- BUG-002 — Strategy writer allowed non-finite and non-positive risk fields
- BUG-004 — Phase 5 evaluated explicitly RB-rejected strategies
- _resolve_plateau_patience
- TestNaNHandling
- TestSparseSignedMode
- BUG-005 — CPU reference arithmetic diverged from canonical evaluator precision
- test_evaluator_health.py
- test_phase2_rule_pool.py
- BUG-006 — Phase 1 resume reused schema-valid selections without input identity
- BUG-001 — Trend thresholds crossed the effective training boundary
- BUG-007 — Phase 2 resume reused pools without a semantic identity
- BUG-008 — Standalone phase commands bypassed prerequisite provenance checks
- TestNormalizeForAssociation
- TestHallOfFameTrim
- passes_pool_admission_gate
- ConfigError
- evaluator_health.py
- test_gpu_engine_import_does_not_crash_on_jax_failure
- test_jax_compat.py
- TestExecutionHealthInGate
- _series
- _pareto_sortino_stats
- TestPerSymbolMetrics
- TestEdgeCases
- TestSignedMode
- TestEvalCvFoldReturns
- TestPerSymbolIndependence
- .detect_feature_mode
- _legacy_writer_contract
- opencode.json
- test_rb_governor_tail_holdout.py
- graphify.js
- Obsolete implementation cleanup policy
- data/__init__.py
- features/__init__.py
- gpu_fuzzy_trader/__init__.py
- output/__init__.py
- phases/__init__.py

## God Nodes (most connected - your core abstractions)
1. `CPUBacktestEngine` - 159 edges
2. `Reporter` - 159 edges
3. `Rule_Pool_Generator` - 137 edges
4. `Output_Writer` - 86 edges
5. `prop_settings()` - 79 edges
6. `Pipeline_Orchestrator` - 72 edges
7. `Data_Loader` - 62 edges
8. `_run_nsga3()` - 59 edges
9. `compute_phase2_objectives_from_metrics()` - 59 edges
10. `_run_nsga2_fallback()` - 58 edges

## Surprising Connections (you probably didn't know these)
- `Graphify Pipeline` --semantically_similar_to--> `Graphify Pipeline`  [INFERRED] [semantically similar]
  .codex/skills/graphify/SKILL.md → .opencode/skills/graphify/SKILL.md
- `Parallel Semantic Extraction` --semantically_similar_to--> `Parallel Semantic Extraction`  [INFERRED] [semantically similar]
  .codex/skills/graphify/SKILL.md → .opencode/skills/graphify/SKILL.md
- `Extraction Confidence Rubric` --semantically_similar_to--> `Extraction Confidence Rubric`  [INFERRED] [semantically similar]
  .codex/skills/graphify/references/extraction-spec.md → .opencode/skills/graphify/references/extraction-spec.md
- `Graph Work Memory` --semantically_similar_to--> `Graph Work Memory`  [INFERRED] [semantically similar]
  .codex/skills/graphify/references/query.md → .opencode/skills/graphify/references/query.md
- `Replace-on-Re-Extract Merge` --semantically_similar_to--> `Replace-on-Re-Extract Merge`  [INFERRED] [semantically similar]
  .codex/skills/graphify/references/update.md → .opencode/skills/graphify/references/update.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Codex Structural and Semantic Extraction Flow** — _codex_skills_graphify_skill_graphify_pipeline, _codex_skills_graphify_skill_semantic_extraction, _codex_skills_graphify_references_extraction_spec_confidence_rubric, _codex_skills_graphify_skill_graph_health_gate [EXTRACTED 1.00]
- **OpenCode Structural and Semantic Extraction Flow** — _opencode_skills_graphify_skill_graphify_pipeline, _opencode_skills_graphify_skill_semantic_extraction, _opencode_skills_graphify_references_extraction_spec_confidence_rubric, _opencode_skills_graphify_skill_graph_health_gate [EXTRACTED 1.00]
- **Per-symbol evidence to certified deployment flow** — plan_cpu_candidate_reevaluation, plan_per_symbol_pool_evidence, plan_rb_symbol_certificate, plan_bounded_diversification_search, plan_certificate_preserving_risk_search [EXTRACTED 1.00]
- **Fail-closed research boundary** — readme_configuration_contract, readme_symbol_specialist_islands, readme_rb_governor, readme_holdout_acceptance_contract, readme_fail_closed_deployment [INFERRED 0.95]
- **Hardware-aware execution flow** — run_rtx4050_execution_policy, run_colab_t4_path, run_pipeline_orchestrator, readme_hybrid_execution_policy [INFERRED 0.95]

## Communities (199 total, 9 thin omitted)

### Community 0 - "evox_runner.py"
Cohesion: 0.03
Nodes (128): _assign_eval_result(), _behavior_descriptor(), _binary_tournament_pick(), _build_diversity_reference(), _build_rank_and_crowding(), _constraint_violations(), _count_deployable_preview(), _count_pop_viable() (+120 more)

### Community 1 - "Rule_Pool_Generator"
Cohesion: 0.07
Nodes (23): _derive_val_sample_seed(), Derive a deterministic validation sample seed from the training seed. This…, Phase 2: GPU-accelerated multi-objective evolutionary rule pool generation.…, Inject guarded migration seeds for the next epoch., Attach optional island metadata; safe when *owner* is a partial mock., Rule_Pool_Generator, _make_feature_infos(), In holdout mode, val engine must be built for pool admission even when… (+15 more)

### Community 2 - "CPUBacktestEngine"
Cohesion: 0.07
Nodes (69): CPUBacktestEngine, CPU backtest engine that exactly mirrors evaluator_v5.ipynb's…, expectancy_lcb_pct(), Return a conservative lower bound for per-trade net expectancy. Exact CPU…, _available_symbols(), _balanced_phase2_shortlist(), _cost_stress_gate(), _eval_cv_fold_returns() (+61 more)

### Community 3 - "_make_engine"
Cohesion: 0.06
Nodes (31): _build_entries_from_rule_set(), Priority-based rule assignment: first matching rule wins per row. Mirrors…, _make_df(), _make_engine(), DataFrame, Simulate catastrophic losses to trigger account ruin. With min_288=0…, Trades with tiny equity should be skipped., All TP trades → win_rate = 100%. (+23 more)

### Community 4 - ".run"
Cohesion: 0.06
Nodes (38): FileHandler, _log_phase_entry(), _log_pipeline_config(), _now_iso(), _phase2_frame_identity(), _phase2_resume_identity(), DataFrame, Create a run identity and remove artifacts that cannot be trusted. (+30 more)

### Community 5 - "trend_context.py"
Cohesion: 0.08
Nodes (49): Number of leading per-symbol rows belonging to the training prefix. Shared by…, train_prefix_row_count(), align_completed_states_to_rows(), average_true_range(), build_higher_bars(), build_manifest(), build_train_prefix(), _classify_hf_bars() (+41 more)

### Community 6 - "test_evox_runner.py"
Cohesion: 0.04
Nodes (53): _diversity_recovery_min_unique_ratio(), _inherit_val_metrics_from_global_cache(), Phase2EvolutionState, Copy val_* from global cache for identical chromosomes when val is skipped.…, Evolve one island epoch and return updated resumable state., Resumable NSGA-III state for symbol-island epoch scheduling., Return survivors that do not already carry a validation snapshot. Validation…, Earliest gen for plateau stop. Island epochs (~20 gens) and scaled Stage A… (+45 more)

### Community 7 - "_score_metrics"
Cohesion: 0.07
Nodes (40): _combined_return_score(), _evaluate_ruleset(), _optimize_risk(), Return (ok, bonus, penalty) for the desired train-valid balance shape. In…, Dominant objective: return/DD with train-valid balance, plus CV-fold…, Profit objective for lenient rule addition, but now evaluator_v5 aware. A new…, _score_metrics(), _train_valid_shape() (+32 more)

### Community 8 - "_get_dont_cares"
Cohesion: 0.10
Nodes (17): _get_dont_cares(), _mutate(), Mutate a chromosome (returns a copy). When activating a dont_care gene, feature…, Return array of dont_care sentinels for each feature., sparse_to_dense(), C5 mutation bias: force symbol-gene to dont_care / inactive with probability…, Create feature_infos with a feature whose name contains 'symbol'., PHASE2_SYMBOL_GENE_DONT_CARE_PROB=1.0: symbol gene always forced to dont_care. (+9 more)

### Community 9 - "_make_df"
Cohesion: 0.09
Nodes (18): _make_df(), _make_engine(), MonkeyPatch, Chromosome that matches nothing returns 0 executed trades., PHASE2_SKIP_ZERO_SIGNAL_SCAN must match always-scan metrics., Chromosome matching all rows should execute trades., Batch of B chromosomes returns B results., Event packing must preserve full-scan equity metrics. (+10 more)

### Community 10 - "phase2_sparse_encoding.py"
Cohesion: 0.13
Nodes (37): _random_active_class(), _count_active_conditions(), Count active rule conditions (sparse slots or dense dont_care encoding)., canonicalize_slots(), _clamp_slot_gene(), count_active_slots(), crossover_sparse(), dense_to_sparse() (+29 more)

### Community 11 - "run_phase2_evolution"
Cohesion: 0.03
Nodes (67): Run Phase 2 NSGA-III evolution. Returns (pareto_pool, history) or state., run_phase2_evolution(), _should_plateau_early_stop_phase2(), Evolutionary algorithm drivers for Phase 2., The fallback must not switch f3 from CV return to PF after gen 0., Verify decision logic uses correct patience values (regression: logs showed…, Island profile: streak=6 triggers when island_patience=6 even when…, Global profile: streak=6 does NOT trigger when global_patience=8. (+59 more)

### Community 12 - "test_phase2_support.py"
Cohesion: 0.08
Nodes (26): deployability_rank_score(), feasibility_violation_score(), _joint_primary_metric(), passes_evolution_deployability_preview(), passes_pool_trade_floor(), Return stage-aware fitness floors; defaults to global strict knobs. When both…, Pool/archive inclusion gate., Train-only or conservative min(train, val) for ranking / objectives. (+18 more)

### Community 13 - "test_phase5_oos.py"
Cohesion: 0.10
Nodes (16): _isolate_phase5_reporter_outputs(), fixture, Unit tests for gpu_fuzzy_trader.phases.phase5_oos.OOS_Evaluator Tests cover: -…, Override module-level path dicts and return originals., Keep Reporter plots/CSVs out of the checked-in outputs directory., Write a valid selected-features JSON to path., Write a synthetic test CSV with all required columns (including feat_0..4) to a…, Give integration tests an isolated, valid enriched train/split pair. (+8 more)

### Community 14 - "Pipeline_Orchestrator"
Cohesion: 0.10
Nodes (42): _context_coverage_preflight(), _context_coverage_report(), Pipeline_Orchestrator, Return the active output root for this run., Top-level orchestrator for the GPU-Fuzzy Trading Pipeline. Runs all five phases…, Temporarily rebind all cached output paths for one pipeline run., Return split-aware context coverage for both trading directions., Reject a mixed, stale, or altered enriched train/test input pair. (+34 more)

### Community 15 - "_apply_monthly_admission_gate"
Cohesion: 0.06
Nodes (29): _apply_monthly_admission_gate(), _evaluate_rule_on_window(), _monthly_window_metrics(), Evaluate a single pool rule on a single monthly window. Returns the full window…, Normalize a window evaluator result for the monthly gate. The float fallback…, Apply the monthly-window shadow-test gate to a pool of rules. Each rule is…, _DeterministicEvaluator, _DeterministicMetricsEvaluator (+21 more)

### Community 16 - "prop_settings"
Cohesion: 0.07
Nodes (62): HealthCheck, prop_settings(), Shared Hypothesis settings for property tests. Set PYTEST_LOW_MEMORY=1 to scale…, Hypothesis settings with optional low-memory example scaling., all_nan_series(), arbitrary_numeric_series(), binary_series(), large_value_series() (+54 more)

### Community 17 - "_make_train_df"
Cohesion: 0.07
Nodes (30): _downsample_chronological(), _monthly_admission_source_df(), DataFrame, Prefer unsampled monthly val; fall back to sampled slim val., Preserve per-symbol time order required by exposure/release simulation., Pick a contiguous chronological slice of *n_rows* from *df*. Critical for…, Sample up to *total_rows* rows, distributed equally across symbols. A single…, _sample_df() (+22 more)

### Community 18 - "_split"
Cohesion: 0.08
Nodes (17): Helper: build df, patch paths, run split, return (train, val)., Compute expected train/val per symbol under holdout+embargo split., floor(N * HOLDOUT_TRAIN_FRACTION) rows go to train., Remaining rows after embargo gap go to validation., For N=101: floor(101 * HOLDOUT_TRAIN_FRACTION) used, not round., train + validation + embargo dropped == total rows., train + val + embargo_dropped == total for each symbol., All train datetimes for a symbol must be < validation datetimes (embargo gap). (+9 more)

### Community 19 - "test_cpu_engine_properties.py"
Cohesion: 0.08
Nodes (43): equity_tracking_scenario(), _expected_outcome(), fee_deduction_scenario(), _make_engine(), _make_engine_custom(), _make_minimal_df(), _make_simple_df(), multi_symbol_scenario() (+35 more)

### Community 20 - "detect_feature_mode"
Cohesion: 0.18
Nodes (7): detect_feature_mode(), Module-level convenience wrapper around Feature_Detector.detect_feature_mode., Two unique values ⊆ {-1, 0, 1} but not ⊆ {0, 1} → ternary., All non-negative, zero_ratio ≤ 0.3 → positive., zero_ratio == 0.3 is NOT > 0.3, so mode is positive., TestPositiveMode, TestTernaryMode

### Community 21 - "selector.py"
Cohesion: 0.08
Nodes (39): get_seed(), Return a per-process random seed (stable within a run, different across runs).…, build_phase1_shared_context(), _build_symbol_masks(), _candidate_feature_columns(), _check_spearman_sign_consistency(), _compute_chronological_stationarity_scores(), _frame_identity() (+31 more)

### Community 22 - "Feature_Selector"
Cohesion: 0.10
Nodes (19): Feature_Selector, Score and rank features separately for long and short directions., _make_train_df(), MonkeyPatch, Create a minimal training DataFrame with label columns and feature columns., Label columns must not appear in selected features., Meta columns must not appear in selected features., Loader internal columns and ``_``-prefixed names are not candidates. (+11 more)

### Community 23 - "rolling_cv.py"
Cohesion: 0.10
Nodes (35): aggregate_fold_metrics(), _bar_index_col(), _build_fold_from_ranges(), build_forbidden_ranges(), build_purged_walk_forward_folds(), cv_folds_only(), derive_primary_holdout(), FoldMetricsSummary (+27 more)

### Community 24 - "TestWritePerSymbolCsv"
Cohesion: 0.18
Nodes (4): _make_per_symbol_metrics(), Symbols with missing sub-keys should default to 0., Create a metrics dict with per_symbol_metrics., TestWritePerSymbolCsv

### Community 25 - "test_encoder_properties.py"
Cohesion: 0.07
Nodes (43): See module-level :func:`get_dont_care`., all_active_chromosome_strategy(), chromosome_with_dont_cares_strategy(), feature_name_strategy(), composite, DrawFn, given, ndarray (+35 more)

### Community 26 - "write_evaluator_clean"
Cohesion: 0.11
Nodes (20): _maybe_write_evaluator_clean(), Path, Write a stripped strategy file containing only ``direction`` and ``rules_set``.…, Write a stripped strategy file alongside the main strategy JSON. This is a…, Validate rule_set and write to JSON at path. After the main write, also writes…, write_evaluator_clean(), Path, Parent directory is created when it does not exist. (+12 more)

### Community 27 - "CandidateRecord"
Cohesion: 0.09
Nodes (44): effective_rb_min_distinct_symbols(), Return the RB coverage target for the active debug universe. Full runs keep…, _candidate_coverage_symbols(), _candidate_positive_symbols(), CandidateRecord, _compose_ruleset(), _diversification_beam(), _diversification_shortlist() (+36 more)

### Community 28 - "nested_walk_forward.py"
Cohesion: 0.13
Nodes (25): Validation helpers for monthly, nested, and multiplicity-safe research., deflated_sharpe_ratio(), estimate_pbo(), Selection-multiplicity diagnostics for strategy research artifacts., Estimate the fraction of folds where the IS winner misses OOS median. Inputs…, Return a normal-approximation deflated Sharpe probability. This is a…, Create a compact ledger-ready multiplicity report., summarize_multiplicity() (+17 more)

### Community 29 - "_preserve_deployable_elites"
Cohesion: 0.10
Nodes (23): environmental_selection_nsga2(), _preserve_deployable_elites(), Canonical NSGA-II truncation on a 2N merged population., Force-preserve top-K deployable-archive elites in the live population.…, _make_chromosome(), _make_deployable_entry(), ndarray, Unit tests for elite preservation under (μ+λ) selection. Verifies that top-K… (+15 more)

### Community 30 - "maybe_log_generation"
Cohesion: 0.09
Nodes (18): generation_log_interval(), iteration_log_interval(), log_generation(), maybe_log_generation(), log_progress.py — Throttled progress logging for long pipeline loops., Return how often to log generation progress. Uses LOG_GENERATION_INTERVAL from…, Log generation progress when the step matches the throttle interval., Log first step, last step, and every *interval* steps in between. (+10 more)

### Community 31 - "Hybrid CPU and GPU execution policy"
Cohesion: 0.07
Nodes (37): Local low-memory test policy, Virtual environment command policy, Package compile gate, Locked research dependencies, Focused low-memory test gate, Python 3.11 CI runtime, Research CI workflow, Balanced Mode-A assumption (+29 more)

### Community 32 - "TestSelectDiverseSubset"
Cohesion: 0.12
Nodes (12): ndarray, Verify _select_diverse_subset correctness for edge cases., Create n distinct dense chromosomes., k=0 should return [] even with non-empty chromosomes., k<0 should return []., k=1 returns one chromosome., k > len(chromosomes) returns all chromosomes (shallow copy)., k == len(chromosomes) returns all chromosomes. (+4 more)

### Community 33 - "test_rb_min_symbols.py"
Cohesion: 0.10
Nodes (26): mandatory_context_conditions(), Return the fixed, mandatory context conditions for *direction*., _symbols_in_rules(), _dummy_df(), _make_candidate_records(), _mock_train_metrics(), _multi_symbol_rules(), _no_symbol_rule() (+18 more)

### Community 34 - "Graphify Pipeline"
Cohesion: 0.06
Nodes (36): Folder Watcher, URL Ingestion, Conditional Graph Exports, Graphify MCP Server, Extraction Confidence Rubric, Deterministic Full-Path Node IDs, Semantic Hyperedges, Cross-Repository Graph Merge (+28 more)

### Community 35 - "Output_Writer"
Cohesion: 0.10
Nodes (12): Output_Writer, Serializes RuleSet dicts to JSON and loads/validates existing JSON files.…, parametrize, Unit tests for gpu_fuzzy_trader.output.writer — Output_Writer Tests cover: -…, Verify the exact example from the spec works end-to-end., TestLoadAndValidateErrors, TestLoadAndValidateHappyPath, TestSpecExample (+4 more)

### Community 36 - "config.py"
Cohesion: 0.06
Nodes (39): context_contract(), _debug_symbol_universe_size(), effective_config_snapshot(), effective_min_profitable_symbols(), effective_min_trade_support(), effective_monthly_min_trades(), effective_pool_min_val_trades(), effective_val_trade_floor_for_objectives() (+31 more)

### Community 37 - "TestEquityCurveDateAxis"
Cohesion: 0.09
Nodes (21): _make_trade_log(), DataFrame, fixture, Unit tests for date-based x-axis in Reporter.plot_equity_curve (Task 10.3).…, When Entry_Time is absent, x-axis label is 'Trade #'., When Entry_Time is all-NaN, x-axis label is 'Trade #'., axhline is called (initial-capital line) in date mode., axhline is called (initial-capital line) in Trade # mode. (+13 more)

### Community 38 - "test_phase2_window_rotation.py"
Cohesion: 0.05
Nodes (40): _derive_epoch_seed(), Return a deterministic per-epoch seed derived from *base_seed* + epoch. Used by…, _largest_safe_range(), Return (start, end) of the largest contiguous bar range not in *forbidden*. The…, Cap *total_rows* so the per-symbol request fits within the safe range. When…, _resolve_sample_total_rows(), _make_multi_sym_df(), DataFrame (+32 more)

### Community 39 - "research_integrity.py"
Cohesion: 0.16
Nodes (24): _canonical_json(), count_trials(), dataset_manifest(), ExperimentLedger, forward_acceptance_lock_path(), Any, Path, PathLike (+16 more)

### Community 40 - "_loader_from_rows"
Cohesion: 0.09
Nodes (24): _base_row(), _loader_from_rows(), _make_ohlcv_rows(), _make_rows(), _make_timestamps(), DataFrame, Unit tests for gpu_fuzzy_trader.data.loader.Data_Loader Tests cover: - CSV…, The first N-288 rows (chronologically) should be kept. (+16 more)

### Community 41 - "_symbol_specialized_variants"
Cohesion: 0.11
Nodes (27): _attach_source_symbol_filters(), _ensure_symbol_filtered_rule(), _has_symbol_condition(), _is_recency_good(), _is_symbol_condition(), Add deterministic single-condition RB candidates. Evolution is deliberately…, Island/cluster symbols carried on Phase 2 pool entries., Feature ANDs + island symbol ORs (engine ORs multiple ``symbol is X``). (+19 more)

### Community 42 - "gpu_engine.py"
Cohesion: 0.06
Nodes (48): _batch_metrics_from_array(), _build_event_batch(), _exposure_slot_capacity(), _fast_reject_result_rows(), _jax_compute_rule_signals(), _jax_compute_rule_signals_batch(), _jax_compute_rule_signals_sparse_batch(), _jax_open_slot() (+40 more)

### Community 43 - "_gpu_runtime.py"
Cohesion: 0.12
Nodes (29): configure_phase2_gpu_runtime(), detect_gpu_memory_used_gb(), detect_gpu_vram_gb(), detect_system_ram_gb(), _iter_warmup_targets(), log_gpu_runtime_config(), _ram_batch_cap(), Phase 2 GPU runtime helpers: VRAM-aware batch size and JAX warmup. (+21 more)

### Community 44 - "._ensure_dir"
Cohesion: 0.07
Nodes (17): _bucket_series_by_mode(), DataFrame, Series, Write feature-stratified performance metrics to CSV files (one per split). For…, Create parent directories for *path* if they do not exist., Compute maximum drawdown percentage from an equity series. Parameters…, Write compact train/validation/test generalization diagnostics to JSON. The…, Plot a per-rule performance breakdown across train/validation/test splits.… (+9 more)

### Community 45 - "_derive_island_seed"
Cohesion: 0.08
Nodes (32): context_coverage_for_direction(), context_coverage_report(), context_floor_failures(), Any, DataFrame, Shared diagnostics for the mandatory direction-specific context contract., Return coverage diagnostics for every named frame and direction., Return mathematically impossible trade-floor failures for coverage. (+24 more)

### Community 46 - "test_certificate_first_selection.py"
Cohesion: 0.13
Nodes (23): _passes_symbol_concentration_gate(), _passes_symbol_contribution_certificate(), _portfolio_selection_certificate(), Any, Require positive, supported validation PnL from multiple symbols. Symbol…, Build the per-direction symbol certificate policy. Specialist islands are…, Return the certificate used by compose, risk, and profit selection., Return (hhi_abs_pnl, top_symbol_share, top_symbol) from per_symbol_metrics. (+15 more)

### Community 47 - "non_dominated_sort"
Cohesion: 0.13
Nodes (26): batch_hamming_min(), crowding_distance(), _crowding_distance_numba(), _crowding_distance_py(), _disable_numba_sort(), _dominates_numba(), _dominates_py(), non_dominated_sort() (+18 more)

### Community 48 - "optuna_search.py"
Cohesion: 0.10
Nodes (29): _active_search_space(), apply_trial_config(), collect_validation_metrics(), compute_score(), _copy_phase1_2_outputs(), main(), objective(), _pearson() (+21 more)

### Community 49 - "phase5_oos.py"
Cohesion: 0.14
Nodes (14): apply_fuzzy_feature_scaling(), fit_fuzzy_feature_scaling(), Any, DataFrame, Train-fitted normalization for ordinal ``ff_*`` fuzzy feature codes. The…, Build a train-only scaling contract for ordinal ``ff_*`` columns., Apply an existing train-fitted scaling contract in place and return *df*., _Phase5JSONEncoder (+6 more)

### Community 50 - "test_phase2_rule_pool_properties.py"
Cohesion: 0.11
Nodes (21): feature_infos_and_train_df(), _isolate_phase2_archive_paths(), _make_feature_infos(), _make_train_df(), composite, DataFrame, DrawFn, fixture (+13 more)

### Community 51 - "_compute_rule_signal_mask"
Cohesion: 0.15
Nodes (13): conditions_cache_key(), get_or_build_rule_mask(), ndarray, condition_cache.py — Cached boolean masks for textual rule conditions. Avoids…, Stable hashable key for a list of condition strings., Return a boolean row mask for *conditions*, using *cache* when provided. The…, _compute_rule_signal_mask(), Build one boolean signal mask (evaluator_v5 parity). Feature conditions are… (+5 more)

### Community 52 - "validate_config"
Cohesion: 0.18
Nodes (19): Validate all high-impact hyperparameter relationships. The function is…, Write the effective configuration snapshot and return its path., validate_config(), write_config_audit_report(), MonkeyPatch, Path, Cross-parameter configuration and evaluator-parity tests., test_audit_report_writes_the_effective_snapshot() (+11 more)

### Community 53 - "build_hybrid_symbol_clusters"
Cohesion: 0.12
Nodes (28): build_hybrid_symbol_clusters(), _corr_embedding_block(), _feature_names_union(), _feature_profile_block(), Any, DataFrame, ndarray, symbol_cluster.py — Per-symbol clustering for Phase 2 island scheduling using… (+20 more)

### Community 54 - "test_phase2_init.py"
Cohesion: 0.14
Nodes (20): assign_strata_to_indices(), build_feature_sampling_probs(), pick_active_count(), _pick_active_index(), _pick_inactive_index(), ndarray, phase2_init.py — Sparsity-guided stratified population initialization for Phase…, Assign elite / explorer labels to non-seeded population rows. (+12 more)

### Community 55 - "TestInitPopulation"
Cohesion: 0.11
Nodes (13): _chromosome_with_min_active(), _is_all_inactive_sparse(), _pop_contains_dense_seed(), ndarray, f3 profit_factor branch mirrors win_rate train+val blend. Covers three…, All active sparse genes must be in [0, dont_care] inclusive., With dont_care_prob=0.5, inactive sparse rows should appear., With dont_care_prob=0, no fully inactive sparse rows should appear. (+5 more)

### Community 56 - "Feature_Detector"
Cohesion: 0.16
Nodes (10): detect_all_modes(), Feature_Detector, DataFrame, detector.py — Feature_Detector Classifies each feature column into exactly one…, Classify feature columns by their discretization type., Classify every column in *feature_cols* and return a mapping. Parameters…, Module-level convenience wrapper around Feature_Detector.detect_all_modes., Unit tests for gpu_fuzzy_trader.features.detector.Feature_Detector Tests cover:… (+2 more)

### Community 57 - "test_reporter.py"
Cohesion: 0.07
Nodes (29): _make_dataset_with_label(), _make_datasets_by_split(), _make_selected_features(), _make_stratified_dataset(), _make_stratified_datasets_by_split(), _make_stratified_logs_by_split(), _make_stratified_trade_log(), DataFrame (+21 more)

### Community 58 - "GPUBacktestEngine"
Cohesion: 0.07
Nodes (19): GPUBacktestEngine, CPU engine for rule-set simulation used by Phase 2/RB., Return whether this host's CPU path is the faster large-window path., Rule-set sim using a mask cache (delegates to the CPU engine)., Evaluate multiple rule sets (parallel CPU; optional mask cache)., RB JAX path: cached mask entry build + parallel CPU equity. Equity results…, Same interface as CPUBacktestEngine for compatibility., JAX-accelerated backtest engine for Phase 2 rule pool generation.… (+11 more)

### Community 59 - "TestGPUCPUNumericalParity"
Cohesion: 0.10
Nodes (16): DataFrame, ndarray, simulate_rule_set (delegated to CPU) is exactly identical to CPU engine., GPU backtest engine must handle padded chunking correctly., CPU backtest engine must support simulate_rule_batch with correct metrics., Compare GPU and CPU engine for 10 random chromosomes (Task 2 acceptance…, Build a multi-symbol DataFrame with mixed TP/SL/time-exit outcomes., Generate random chromosomes for binary + signed features. (+8 more)

### Community 60 - "ValueError"
Cohesion: 0.06
Nodes (40): _append_allocated_entries(), _build_rule_signal_mask(), compute_entry_time_priority(), _expectancy_lcb_pct(), _expected_shortfall_pct(), _parse_condition(), precompute_release_indices(), precompute_release_indices_from_offsets() (+32 more)

### Community 61 - "._build_engine_for_df"
Cohesion: 0.11
Nodes (13): Build train/val backtest engines., Restore slimmed training data from cache (no re-sampling needed)., Rebuild engines after ``park_engines`` dropped GPU state., Build the selected Phase 2 backend for the sampled train frame., Return whether this generator should avoid allocating a JAX engine.…, Build an engine on *df* using the same backend selection logic., _minimal_backtest_df(), DataFrame (+5 more)

### Community 62 - "._build_per_symbol_rows"
Cohesion: 0.22
Nodes (4): Evaluate a single strategy on the test DataFrame. Returns ------- metrics :…, Return an explicit, non-success result for a failed split., Build a flat list of per-symbol metric dicts for CSV output. Uses the…, TestBuildPerSymbolRows

### Community 63 - "TestRulePoolGeneratorRun"
Cohesion: 0.14
Nodes (8): Integration tests using tiny population and generation counts., Verify that Rule_Pool_Generator's persistent self._rng advances across multiple…, After two run_epoch() calls, the internal RNG state must differ from the…, The RNG should produce *different* sequences in two consecutive run_epoch()…, Two generators with different seeds must have different RNG state., Rule_Pool_Generator must initialize self._rng as a Generator., TestRulePoolGeneratorRng, TestRulePoolGeneratorRun

### Community 64 - "test_plateau_state_leak.py"
Cohesion: 0.11
Nodes (21): _make_minimal_gen(), _mock_evolution_state(), _mock_stage_plan(), Regression tests for plateau-state leak fixes (Fixes A + B). Fix A:…, AC-1, AC-3: plateau_streak resets per epoch; reset_plateau=True always., reset_plateau=True is passed on the very first epoch call., reset_plateau=True is also passed on epoch 2 (regression for leak)., Even when PHASE2_ISLAND_TWO_STAGE_ENABLED=False, reset_plateau=True. (+13 more)

### Community 65 - "test_phase5_oos_properties.py"
Cohesion: 0.18
Nodes (12): _make_timestamps(), composite, DataFrame, DrawFn, given, Timestamp, Property-based tests for gpu_fuzzy_trader.phases.phase5_oos.OOS_Evaluator…, **Property 27: Test Data Preparation Consistency** **Validates: Requirements… (+4 more)

### Community 66 - "dashboard.py"
Cohesion: 0.15
Nodes (25): build_dashboard_data(), _direction_data(), _history_rows(), _load_json(), main(), Any, Path, Read-only HTML dashboard for existing pipeline artifacts. The dashboard… (+17 more)

### Community 67 - "test_data_splitter_properties.py"
Cohesion: 0.17
Nodes (14): _make_timestamps(), composite, DataFrame, DrawFn, given, Timestamp, Property-based tests for gpu_fuzzy_trader.data.splitter.Data_Splitter…, Patch TRAIN_70_PATH / VALIDATION_30_PATH to tmp_path and run split. (+6 more)

### Community 68 - "test_data_loader_properties.py"
Cohesion: 0.14
Nodes (26): dataframe_with_nan_features(), dataframe_with_nan_labels(), _load_from_df(), _make_datetime_series(), composite, DataFrame, DrawFn, given (+18 more)

### Community 69 - "barrier_column_names"
Cohesion: 0.15
Nodes (17): attach_barrier_outcomes(), barrier_column_names(), configured_barrier_pairs(), _first_touch_for_symbol(), _first_touch_for_symbol_numba(), _first_touch_for_symbol_python(), _number_token(), DataFrame (+9 more)

### Community 70 - "_m"
Cohesion: 0.13
Nodes (16): evaluator_health_penalty(), Penalty for evaluator_v5 failure modes — higher is worse. Parameters ----------…, _m(), Max positions 15 > 10 → positive penalty., Max positions 8 <= 10 → no additional penalty from positions., Missing ``raw_signal_count`` → treated as 0 → no skip/exec penalty., ``raw_signal_count=0`` → no skip/exec ratio computed → no penalty., Role 'train' same as 'valid' (1.0x). (+8 more)

### Community 71 - "_apply_dynamic_rule"
Cohesion: 0.21
Nodes (4): _apply_dynamic_rule(), Apply one exported text condition using the original threshold logic. Exactly…, Test all threshold branches of _apply_dynamic_rule., TestApplyDynamicRule

### Community 72 - "test_cpu_engine.py"
Cohesion: 0.07
Nodes (26): _normalize_direction(), Compute a non-annualized Sortino Ratio from per-trade returns., _rule_symbols_for_allocation(), _safe_profit_factor(), _sortino_ratio_from_returns(), jax.lax.scan-based sequential equity simulation (legacy compat). Parameters…, JointPortfolioEngine, DataFrame (+18 more)

### Community 73 - "test_rb_fail_closed.py"
Cohesion: 0.13
Nodes (24): _assert_capital_budget(), _assert_mandatory_context(), _enforce_capital_budget(), Path, Persist an explicit empty strategy and diagnostic report., Normalize rule capital_pct so sum <= RB_MAX_TOTAL_CAPITAL., Fail closed if the fixed trend-context conditions were lost. The mandatory…, _strategy() (+16 more)

### Community 74 - "execution_ok"
Cohesion: 0.15
Nodes (11): execution_ok(), Return ``True`` iff the evaluator would execute this rule set reasonably. A…, Tests for ``execution_ok``., Skip ratio 0.10 <= 0.20 and exec ratio 0.90 >= 0.60 → True., Skip ratio 0.30 > 0.20 → False., Exec ratio 0.50 < 0.60 → False., Missing ``raw_signal_count`` → treated as 0 → False., ``raw_signal_count=0`` → False. (+3 more)

### Community 75 - "TestWriteStrategyEvaluationTable"
Cohesion: 0.21
Nodes (9): _make_eval_rule_set(), _make_metrics_by_split(), _make_trade_logs_by_split(), Create a minimal rule set for evaluation table tests., Create a metrics_by_split dict with all three splits populated., Create a trade_logs_by_split dict with all three splits populated., Sharpe ratio should equal mean(r)/std(r, ddof=1) for a known log., win_rate, mdd_pct etc. should come from metrics_by_split. (+1 more)

### Community 76 - "test_gpu_engine.py"
Cohesion: 0.15
Nodes (9): _discretize_series(), Series, Discretize a feature series using evaluator_v5's fixed fuzzy bins. Exported…, disable_skip_optimization(), fixture, Unit tests for GPUBacktestEngine. Tests verify: - JAX availability detection…, Disable signal skip optimization for all GPU engine tests. The skip…, TestBuildDataMatrix (+1 more)

### Community 77 - "test_crash_fix_and_run_logging.py"
Cohesion: 0.06
Nodes (32): _capture_logs(), _make_feature_infos_crash(), _make_train_df_crash(), DataFrame, MonkeyPatch, Unit tests for crash-fix-and-run-logging spec. Covers: - Task 1.1: Smoke tests…, Requirement 3.1 — save_archive is called before _release_resources in run()., Mock save_archive and _release_resources, run Rule_Pool_Generator.run() with a… (+24 more)

### Community 78 - "TestPrecomputeReleaseIndices"
Cohesion: 0.29
Nodes (4): Release index should point to the row where bar_index + 288 is reached., Rows near the end should get release_index = len(df)., Each symbol's release indices are computed independently., TestPrecomputeReleaseIndices

### Community 79 - "test_feature_selector_properties.py"
Cohesion: 0.15
Nodes (23): dataset_with_high_dispersion_feature(), dataset_with_label_and_meta_columns(), dataset_with_sufficient_dispersion_feature(), _make_label_block(), composite, DataFrame, DrawFn, given (+15 more)

### Community 80 - "loader.py"
Cohesion: 0.13
Nodes (16): _ensure_labels(), load_dataset(), DataFrame, data/loader.py — Data_Loader Stateless CSV loading with full preparation…, Recompute the LWC pullback-reversal triggers and compare row-by-row. A stale or…, Keep supplied labels or derive all labels from raw OHLCV columns. The original…, Load a CSV dataset with full preparation pipeline: 1. Read CSV with comma…, Module-level wrapper around ``Data_Loader.load_dataset``. (+8 more)

### Community 81 - "compute_labels"
Cohesion: 0.20
Nodes (10): compute_labels(), DataFrame, Forward-window label computation for OHLCV bars. Horizon is ``TAIL_DROP_ROWS``…, Compute the 5 label columns per symbol. - label_open_next: open[t+1] -…, DataFrame, ndarray, Unit tests for forward-window label semantics…, Regression: old backward rolling must not match corrected forward labels. (+2 more)

### Community 82 - "TestPlotPerRuleBreakdown"
Cohesion: 0.16
Nodes (11): _make_full_trade_log(), _make_rule_set(), _make_split_logs(), Create a trade log with all columns needed for sharpe computation., Create a minimal rule set with *n* rules., Create a trade log with Rule_Index, Net_PnL, Equity_After columns., Create trade_logs_by_split dict with all three splits populated., One split None, others valid — should not raise. (+3 more)

### Community 83 - "resolve_island_hyperparams"
Cohesion: 0.11
Nodes (21): effective_phase2_val_return_floor_pct(), Direction-aware Phase 2 validation return floor for fitness penalties., Scale integer trade floors by slice size vs full-universe reference., Resolve scaled trade floors and relaxed cross-symbol gates., resolve_island_hyperparams(), scale_trade_floor_by_universe(), Regression tests for anti-overfit / exploration-retune config bundle., test_anti_overfit_config_bundle() (+13 more)

### Community 84 - "BUG_HUNT_REPORT.md"
Cohesion: 0.09
Nodes (22): Cache/Resume Audit, Cache/Resume Audit, CPU/JAX Parity Results, CPU/JAX Parity Results, Data Leakage Audit, Data Leakage Audit, Evaluator Parity Results, Evaluator Parity Results (+14 more)

### Community 85 - "_build_target"
Cohesion: 0.17
Nodes (12): _build_target(), Build a direction-specific target signal. Default…, Encoding-aware win class: 2 in asymmetric mode, 1 in legacy mode., Encoding-aware loss class: 0 in either mode., Long: max >= entry*(1+TP/100), min > entry*(1-SL/100) → success., Long: both hit but max_before_min==0 → SL first → failure., Long: both hit but max_before_min==1 → TP first → success., Long: neither TP nor SL hit → failure (or neutral in asymmetric mode). (+4 more)

### Community 86 - "filter_migrants_for_cluster"
Cohesion: 0.10
Nodes (21): _exchange_migrants_between_islands(), filter_migrants_for_cluster(), _migrant_to_metrics(), Backtest one migrant chromosome on receiver cluster engines., Accept only migrants that pass deployability on the receiver cluster slice., Perform a guarded, order-independent migration exchange. Islands are processed…, _make_migrant_dict(), _make_mock_receiver() (+13 more)

### Community 87 - "test_gpu_engine_properties.py"
Cohesion: 0.16
Nodes (17): _assert_parity(), _make_engines(), _make_parity_df(), parity_scenario_strategy(), composite, DataFrame, DrawFn, given (+9 more)

### Community 88 - "writer.py"
Cohesion: 0.10
Nodes (21): context_permission_column(), context_trigger_column(), Return the direction-specific permission column name., Return the direction-specific LWC pullback-reversal trigger column., _condition_feature(), _context_feature_direction(), writer.py — Output_Writer Serializes RuleSet dicts to JSON with exact schema…, Validate an optional symbol filter (symbol is X / [symbol] IS X). (+13 more)

### Community 89 - "_make_rule"
Cohesion: 0.16
Nodes (5): _make_rule(), Spot-check a variety of valid fuzzy value names., TestWriteConditionValidation, TestWriteFieldCoercion, TestWriteSymbolConditionValidation

### Community 90 - "Data_Splitter"
Cohesion: 0.13
Nodes (19): Data_Splitter, load_cached_split_if_fresh(), Load cached split parquets when they are newer than the source CSV. Validates…, Chronological train/validation splitter., _make_df(), _make_timestamps(), _patch_split_paths(), DataFrame (+11 more)

### Community 91 - "set_purged_wf_reference_rows"
Cohesion: 0.18
Nodes (7): Store full train_new.csv row count after loader prep (split time)., set_purged_wf_reference_rows(), holdout_mode(), purged_mode(), fixture, Tests for purged-WF trade-floor scaling helpers., TestScaleTradeFloor

### Community 92 - "downcast_numeric_df"
Cohesion: 0.20
Nodes (15): downcast_numeric_df(), prune_train_columns(), DataFrame, df_slim.py — Reduce DataFrame memory for backtest engines. Keeps only columns…, Downcast label and feature columns to float32; symbol to category (inplace)., Return a copy with only meta, label, and requested feature columns. Parameters…, Drop unused feature columns from the full training split after Phase 1., slim_backtest_df() (+7 more)

### Community 93 - ".encode_condition"
Cohesion: 0.07
Nodes (25): ConfigurationError, decode_chromosome(), encode_condition(), Encoder, get_dont_care(), Exception, ndarray, encoder.py — Encoder Maps gene integer values to fuzzy value names, formats… (+17 more)

### Community 94 - "test_output_writer_properties.py"
Cohesion: 0.16
Nodes (23): all_zero_rule_st(), _is_valid_exported_condition(), oversized_rule_set_st(), composite, DrawFn, given, Property-based tests for gpu_fuzzy_trader.output.writer.Output_Writer Property…, Append the direction's mandatory context conditions to every rule. (+15 more)

### Community 95 - "Data_Loader"
Cohesion: 0.07
Nodes (22): Data_Loader, Stateless data loader for the GPU-Fuzzy Trading Pipeline., Exception, Raised when a rule set fails schema validation., ValidationError, _enriched(), Regression: thresholds must never see validation-period rows., Default: opposite LWC print counts even if permission was off. (+14 more)

### Community 96 - "get_gpu_backtest_engine_class"
Cohesion: 0.31
Nodes (7): CPU and GPU backtest engine sub-package., get_gpu_backtest_engine_class(), jax_gpu_backtest_available(), Any, Detect whether JAX / GPUBacktestEngine can be loaded on this host. JAX can fail…, Return ``GPUBacktestEngine`` if ``gpu_engine`` imports cleanly; else ``None``., True when ``get_gpu_backtest_engine_class()`` would succeed.

### Community 97 - ".decode_chromosome"
Cohesion: 0.26
Nodes (4): See module-level :func:`decode_chromosome`., All 10 signed values decode correctly via decode_chromosome., Chromosome with numpy int32/int64 values should work., TestDecodeChromosome

### Community 98 - ".load_strategies"
Cohesion: 0.18
Nodes (6): Load long.json and short.json via Output_Writer.load_and_validate(). Parameters…, Return only non-empty RB strategies accepted for deployment., Write a valid rule set JSON to path., Override module-level path dicts and return originals (for standalone tests)., TestLoadStrategies, _write_rule_set()

### Community 99 - "gate_positive_good"
Cohesion: 0.26
Nodes (17): _passes_pool_admission_impl(), _finite_number(), gate_positive_good(), _metric_int(), positive_good_reject_reasons(), PositiveGoodThresholds, Shared metric gates used by Phase 2 admission and RB Governor. The gate is…, Minimum evidence required on train and validation splits. (+9 more)

### Community 100 - "phase2_island_scheduler.py"
Cohesion: 0.16
Nodes (22): filter_df_to_symbols(), DataFrame, Return rows for the given symbols; raises if column missing or no rows., persist_symbol_clusters(), _context_support_preflight(), _load_phase1_feature_lists(), Any, DataFrame (+14 more)

### Community 101 - ".save_archive"
Cohesion: 0.20
Nodes (9): _archive_feature_signature(), Return the ordered feature signature used to validate archive reuse., Read JSON from *path* and return None when the file cannot be loaded., Validate the archive JSON structure and feature compatibility., Load a compatible persistent archive if it exists, otherwise return None.…, Merge the latest pool into a persistent archive and write atomically., _read_json_payload(), _validate_archive_payload() (+1 more)

### Community 102 - "TestRemoveRedundantFeatures"
Cohesion: 0.25
Nodes (4): Two features with corr > 0.95 → keep higher-scored one., Two uncorrelated features → both kept., Features in different modes are not compared for redundancy., TestRemoveRedundantFeatures

### Community 103 - "TestEquityCurvePlots"
Cohesion: 0.16
Nodes (10): Verify plot_equity_curve is called for all three splits and handles empty logs., Helper: run OOS_Evaluator.run() with mocked internals and capture plot calls., plot_equity_curve must be called with 'train', 'validation', and 'test'., plot_equity_curve must be called exactly three times (one per split)., Empty train trade log must not raise an exception., Empty validation trade log must not raise an exception., Empty test trade log must not raise an exception., All three empty trade logs must not raise an exception. (+2 more)

### Community 104 - "test_crash_fix_properties.py"
Cohesion: 0.10
Nodes (26): _append_xla_flag(), configure_jax_env(), _cuda_package_root(), Path, JAX/XLA runtime configuration — call before the first ``import jax``., Find a pip-installed CUDA toolkit root, if one is available., Append an XLA flag once, preserving explicit user configuration., Configure JAX/XLA runtime for predictable desktop-friendly GPU usage. -… (+18 more)

### Community 105 - "Reporter"
Cohesion: 0.13
Nodes (11): Generates visual and tabular reports for each pipeline phase. All output files…, Reporter, _make_dist_logs_by_split(), _make_dist_trade_log(), _make_trade_log(), Create a trade log with all columns needed for distribution/equity plots., Create trade_logs_by_split dict with all three splits populated., A trade log with a single trade should not raise. (+3 more)

### Community 106 - "baselines.py"
Cohesion: 0.28
Nodes (15): _compact(), _drop_one_condition_variants(), _equal_weight(), _evaluate(), evaluate_baselines(), _feature_shuffle(), _fixed_exit(), Any (+7 more)

### Community 107 - "phase2_rule_pool.py"
Cohesion: 0.04
Nodes (81): IslandHyperparams, Resolved Phase 2 knobs for cluster or orphan slices., extract_deployable_migrants(), Return elite deployable-preview entries suitable for guarded migration., _archive_direction(), _archive_objective_vector(), attach_cv_fold_returns_batch(), _build_cpu_archive_engine() (+73 more)

### Community 108 - "_jax_compute_trade_outcomes"
Cohesion: 0.24
Nodes (5): _jax_compute_trade_outcomes(), JAX-jitted vectorized trade outcome computation., Vectorized trade outcome computation for all rows. Mirrors…, Multiple rows processed simultaneously., TestComputeTradeOutcomes

### Community 109 - "_should_post_restart_early_stop_phase2"
Cohesion: 0.13
Nodes (18): True for cluster/orphan scoped runs rather than the global path., scoped_island_profile(), Break the epoch when a plateau restart yields no improvement. Independent of…, _should_early_stop_phase2(), _should_post_restart_early_stop_phase2(), Unit tests for island early-stop bypass., Regression: island patience must come from…, test_cluster_profile_disables_early_stop() (+10 more)

### Community 110 - "_write_and_reload"
Cohesion: 0.23
Nodes (5): _make_rule_set(), Write rule_set to a temp file and reload the raw JSON., TestWriteHappyPath, TestWriteTruncation, _write_and_reload()

### Community 111 - "resolve_phase2_stage_params"
Cohesion: 0.08
Nodes (24): Stop burning gens when the feasible set is empty and restarts are spent. Sparse…, True when Stage A viability is critically low and search has plateaued., _should_abort_zero_deployable_collapse(), _should_viability_recovery(), island_stage_budgets(), IslandStagePlan, StageLabel, Phase 2 two-stage search hyperparameter profiles (exploration vs refinement). (+16 more)

### Community 112 - "_make_df"
Cohesion: 0.26
Nodes (8): _make_df(), _make_rule_set(), DataFrame, When no trades are executed, account_ruined must be False., When no trades are executed, total_return_pct must be 0.0., Create a minimal valid rule set dict., Create a minimal DataFrame with all required columns., TestEvaluateStrategy

### Community 113 - ".run"
Cohesion: 0.16
Nodes (6): Run out-of-sample evaluation. Parameters ---------- allowed_directions :…, Load selected features for a direction when available., Remove only known Phase 5 artifacts from the active report root., Save a split report, marking consumed test data as diagnostic-only., Save the combined per-symbol performance CSV., TestSavePerSymbolCsv

### Community 114 - "TestHammingThresholdAutoScale"
Cohesion: 0.15
Nodes (10): Verify the max(3, k_active // 5) formula. The formula is applied in…, Replicate the auto-scaling formula., k_active=0 → threshold = max(3, 0//5) = 3., k_active=5 → threshold = max(3, 5//5=1) = 3., k_active=15 → threshold = max(3, 15//5=3) = 3., k_active=20 → threshold = max(3, 20//5=4) = 4., k_active=50 → threshold = max(3, 50//5=10) = 10., k_active=100 → threshold = max(3, 100//5=20) = 20. (+2 more)

### Community 115 - "TestValLeakGate"
Cohesion: 0.20
Nodes (10): C6: Val-derived penalties must be gated behind JOINT_TRAIN_VAL or…, Return standard monkeypatching for clean baseline metrics., Apply base settings with optional overrides., Metrics that trigger no train-side penalties., Val metrics that WOULD trigger penalties if the gate were open., When both JOINT_TRAIN_VAL and VAL_IN_FITNESS_PENALTY are False, val-derived…, Bad val must not set feasibility_violation when gate is closed., When VAL_IN_FITNESS_PENALTY=True, val-derived penalties DO enter objectives. (+2 more)

### Community 116 - "._simulate_rule_set_entries"
Cohesion: 0.09
Nodes (17): _batch_eval_rule_set_pickled(), _jax_runtime_loaded(), Simulate a rule set on rows [row_start, row_end) without copying the df. Used…, Simulate a rule set and return performance metrics. Parameters ----------…, Simulate using a precomputed rule-evaluation mask cache. The mandatory…, Drop cached entries on rows the fixed context mask forbids. ``idx`` validity is…, Evaluate multiple rule sets without forking an active JAX runtime., Top-level worker for ProcessPoolExecutor (must be picklable). (+9 more)

### Community 117 - "TestComputeRuleSignals"
Cohesion: 0.18
Nodes (6): All rows match when chromosome equals data_matrix values., No rows match when chromosome differs from data_matrix., Only rows where all active conditions match., Columns where chromosome == dont_care are ignored., All dont_care chromosome matches every row., TestComputeRuleSignals

### Community 118 - "TestParetoCollapseWarningGate"
Cohesion: 0.15
Nodes (10): _FakeEngine, AC 4: The default value of the config flag is 5., AC 5: min_pareto_size=0 → warning fires even with small pareto (size=4)., The log message includes 'pareto_size=N' suffix., Fake engine that returns metrics producing a tradeoff between f1 (-sortino) and…, AC 1–5: warning gated on len(pareto_indices) >= config threshold., Run 2-gen evolution and return count of 'Pareto collapse risk' warnings., AC 2: pareto_size=4 < min_pareto_size=5 → no warning fires. (+2 more)

### Community 119 - "TestMakeWalkForwardFoldEngines"
Cohesion: 0.17
Nodes (10): _make_synthetic_df(), DataFrame, Verify _make_walk_forward_fold_engines splits data correctly., 2 splits + tail holdout → 2 fold engines + 1 tail engine., tail_holdout_frac=0 → tail engine is None., Each symbol's data is divided into contiguous chunks across folds., Single symbol without symbol column is handled gracefully., Very small data per symbol (fewer rows than n_splits) does not crash. (+2 more)

### Community 120 - "_compute_stability"
Cohesion: 0.17
Nodes (9): _compute_stability(), Compute cross-symbol stability score. stability = 1 - (std / mean) If mean is 0…, All symbols have same score → std=0 → stability=1., Mean is 0 → stability=0., Single symbol with positive score → stability=1., Single symbol with zero score → stability=0., Very high variance → stability could be negative → clipped to 0., Stability should always be in [0, 1]. (+1 more)

### Community 121 - "._prune_splits_after_phase1"
Cohesion: 0.25
Nodes (4): Selected fuzzy features for Phase 2., Drop unused feature columns from train/val splits to reduce RAM., Drop unused feature columns from train split (legacy single-split API)., TestPruneSplitsAfterPhase1

### Community 122 - "MonthlyWindowSummary"
Cohesion: 0.07
Nodes (38): build_monthly_windows(), _datetime_series(), evaluate_rule_set_monthly(), monthly_penalty(), monthly_return_counts_as_good(), MonthlyWindowSummary, DataFrame, Series (+30 more)

### Community 123 - "._engine"
Cohesion: 0.18
Nodes (4): Test _build_trade_outcome_single for long direction., Test _build_trade_outcome_single for short direction., TestTradeOutcomeLong, TestTradeOutcomeShort

### Community 124 - "test_rb_concentration_tail_fail_closed.py"
Cohesion: 0.26
Nodes (9): _candidates(), _dummy_df(), _mock_metrics(), DataFrame, Tests for RB concentration / tail-holdout hard fail-closed behaviour. When…, Return/PF below gate but sym+tail OK → rules retained, not accepted., _rule(), _run_pipeline() (+1 more)

### Community 125 - "constrained_non_dominated_sort"
Cohesion: 0.23
Nodes (13): _clean_violation(), constrained_dominates(), constrained_non_dominated_sort(), _pareto_dominates(), ndarray, Constraint-aware Pareto ordering for Phase 2 evolution. Objectives alone are…, Return whether *left* Pareto-dominates *right* (minimisation)., Return whether the left candidate dominates the right candidate. (+5 more)

### Community 126 - "test_feature_selector.py"
Cohesion: 0.09
Nodes (18): _align_feature_array(), _mutual_info_discrete_mask(), Return the set of feature names that PASS both stationarity checks. A feature…, Cap long/short feature overlap and backfill each direction to top_k features., Per-column ``discrete_features`` flags for ``mutual_info_classif``. Only…, Slice precomputed matrix columns when *selected_cols* is a subset., Remove features where more than `threshold` fraction of values are identical.…, _reduce_overlap() (+10 more)

### Community 127 - "._load_datasets_by_split"
Cohesion: 0.23
Nodes (6): DataFrame, Prepare test data using Data_Loader.load_dataset(). Applies the same…, Load prepared train, validation, and test datasets., Load and validate a strictly newer, untouched forward period., prepare_test_data should return a DataFrame., TestPrepareTestData

### Community 128 - "stratification_scenario_strategy"
Cohesion: 0.24
Nodes (15): dataset_with_features_strategy(), metrics_strategy(), composite, DataFrame, DrawFn, Generate a trade log DataFrame with 2–50 rows. Parameters ---------- n_rules:…, Generate a metrics dict with reasonable float values. Keys: win_rate,…, Generate a dataset DataFrame with 10–100 rows. Parameters ----------… (+7 more)

### Community 129 - "TestSpearmanSignConsistency"
Cohesion: 0.17
Nodes (10): DataFrame, ndarray, AC3: Train all positive, tiny |val_rho| < min_abs_corr → feature still kept., AC4: val_df=None → pre-task-8 behavior (mixed train signs still blacklisted)., AC5: val_df without label_close_288 column → val check skipped (no crash)., Tests for _check_spearman_sign_consistency, including the val_df check., Build a minimal DataFrame without 'symbol' to avoid symbol-based folding., AC1: Train all positive, val negative → feature blacklisted. (+2 more)

### Community 130 - "run_pipeline.py"
Cohesion: 0.11
Nodes (22): __main__.py — Entry point for `python -m gpu_fuzzy_trader.run_pipeline` Allows…, _context_coverage_for_direction(), _context_island_sample_report(), _identity_value(), main(), _NumpyJSONEncoder, _phase2_cv_structure(), _phase5_test_metrics() (+14 more)

### Community 131 - "log_memory_rss"
Cohesion: 0.19
Nodes (11): log_memory_rss(), Optional process memory logging (set LOG_MEMORY=1)., Log current process RSS in MiB when LOG_MEMORY=1., Force GC and clear JAX compilation caches between Phase 2 directions. Releases:…, release_phase2_resources(), Release GPU engines and slim in-memory data between island epochs., Drop engine and sampled data to free RAM before the next direction., Tests for optional memory logging helpers. (+3 more)

### Community 132 - "TestPlotPhase2Metrics"
Cohesion: 0.23
Nodes (4): _make_history(), History entries with missing keys should not raise., Create a minimal Phase 2 history list., TestPlotPhase2Metrics

### Community 133 - "_apply_colab_gpu_defaults"
Cohesion: 0.33
Nodes (6): _apply_colab_gpu_defaults(), is_colab_runtime(), True when running on Google Colab (/content runtime)., Colab T4 optimization for Phase 2 runs., MonkeyPatch, test_colab_defaults_apply_when_content_exists()

### Community 134 - "splitter.py"
Cohesion: 0.12
Nodes (23): _chronological_half_split(), _file_sha256(), _holdout_embargo_split(), _purged_walk_forward_split(), DataFrame, data/splitter.py — Data_Splitter Per-symbol chronological split for Phase 2,…, Per-symbol chronological first or second half of *df*. ``purge_rows`` is…, Split validation into purged fitness and selection halves per symbol. The gap… (+15 more)

### Community 135 - "phase2_support.py"
Cohesion: 0.19
Nodes (10): compute_support_penalty_and_specialist(), EvolutionFloors, phase2_support.py — Trade support penalties for Phase 2., Support penalty. Returns ------- penalty : float is_specialist : bool (always…, Resolved evolution-time floors (pool admission gates remain strict)., Support penalty from train metrics. Returns (penalty, False, -1)., Legacy graduated penalty., _static_support_penalty() (+2 more)

### Community 136 - "OOS_Evaluator"
Cohesion: 0.27
Nodes (4): OOS_Evaluator, Out-of-sample evaluator for Phase 5. Loads the final long/short strategies,…, TestOOSEvaluatorInit, TestSaveReport

### Community 137 - "compute_phase2_objectives_from_metrics"
Cohesion: 0.04
Nodes (33): compute_phase2_objectives_from_metrics(), _diversity_penalty_blended(), _hamming_distance(), _phenotype_bucket_key(), Hamming distance between two chromosomes (active pairs when sparse)., Discretise objective-relevant metrics for behavioral diversity., Hamming OR phenotype-bucket crowding penalty (same weight on both)., Penalty for weak cross-symbol robustness on one split. (+25 more)

### Community 138 - "BUG-003 — CPU time-exit metric depended on log generation"
Cohesion: 0.20
Nodes (10): Actual behavior, BUG-003 — CPU time-exit metric depended on log generation, Confirmed Bugs, Expected behavior, Fix, Regression test, Reproduction, Research/trading impact (+2 more)

### Community 139 - "test_evaluator_clean_writer.py"
Cohesion: 0.18
Nodes (11): _make_rule(), minimal_strategy(), fixture, Unit tests for ``gpu_fuzzy_trader.output.writer.write_evaluator_clean``. Tests…, Tests that ``Output_Writer.write`` also produces the evaluator-clean file., Output_Writer.write produces both the main file and the evaluator-clean file., Long direction also produces the correct clean file., Strategy with only the two required keys (no extras). (+3 more)

### Community 140 - "TestMigrationSeedFraction"
Cohesion: 0.20
Nodes (6): Verify that migrant injection uses PHASE2_MIGRATION_SEED_FRACTION, not ARCHIVE., PHASE2_MIGRATION_SEED_FRACTION defaults to 0.10., Ensure the migration fraction is not the same as archive fraction., Simulate the run_epoch migrant path and assert local_cap uses migration…, Simulate the new run_epoch logic: migrant entries are capped by migration…, TestMigrationSeedFraction

### Community 141 - "_run_cluster_islands"
Cohesion: 0.07
Nodes (29): clear_global_metrics_cache(), Clear the global eval cache and force GC. Used to free RAM between cluster runs…, evict_cluster_signatures(), Evict JAX compiled signatures for a completed cluster. Removes entries from…, compute_cluster_generation_budgets(), Resolve per-island generation budgets. By default each island receives the full…, Check if an epoch should be skipped due to small remaining budget. Engine…, _run_cluster_islands() (+21 more)

### Community 142 - "_pool_admission_floors"
Cohesion: 0.20
Nodes (8): effective_min_trade_pool_floor(), _evolution_feasibility_floors(), _pool_admission_floors(), Return (train_trade_floor, train_ret_min, val_ret_min, pf_floor,…, Return (train_trade_floor, train_ret_min, val_ret_min, pf_floor,…, _pool_admission_floors returns the ADMISSION floor (1.15), not the EVOLUTION…, _evolution_feasibility_floors returns PF=1.0, not ADMISSION 1.15., TestPoolAdmissionScaledFloors

### Community 143 - "TestPlotPhase2Pnl"
Cohesion: 0.23
Nodes (4): _make_pnl_history(), History entries with missing keys should not raise., Create a minimal Phase 2 history list with PnL fields., TestPlotPhase2Pnl

### Community 144 - "TestRefreshObjectivesOnResumeGate"
Cohesion: 0.23
Nodes (7): Task-10: gate cache refresh on PHASE2_PER_EPOCH_WINDOW_ROTATION. Verifies the…, Evaluate the gate expression as it appears in the source., PHASE2_PER_EPOCH_WINDOW_ROTATION=True (default): refresh = not first_epoch →…, PHASE2_PER_EPOCH_WINDOW_ROTATION=True (default): refresh = not first_epoch →…, PHASE2_PER_EPOCH_WINDOW_ROTATION=False (legacy): refresh is False even on…, PHASE2_PER_EPOCH_WINDOW_ROTATION=False (legacy): refresh is False on first…, TestRefreshObjectivesOnResumeGate

### Community 145 - "conftest.py"
Cohesion: 0.22
Nodes (10): Config, FixtureRequest, Item, _close_matplotlib_figures(), _low_memory_cleanup(), fixture, pytest_collection_modifyitems(), pytest_configure() (+2 more)

### Community 146 - "context_contract_digest"
Cohesion: 0.24
Nodes (10): context_contract_digest(), Return a stable hash of the static contract and fitted enrichment., feature_conditions_only(), phase2_rule_id(), Any, Stable provenance identifiers for Phase 2 feature rules. The RB Governor may…, Return normalized non-symbol, non-context conditions in order. Symbol scope and…, Hash the immutable Phase 2 feature logic and its discovery scope. (+2 more)

### Community 147 - ".skip_if_valid"
Cohesion: 0.31
Nodes (3): Check if output files exist and are valid. Returns ------- dict[str,…, fixture, TestSkipIfValid

### Community 148 - "_validate_schema"
Cohesion: 0.17
Nodes (5): Validate the structure of a loaded feature selection JSON. Raises ValueError if…, Load and validate a feature selection JSON file. Parameters ---------- path :…, _validate_schema(), TestLoadAndValidate, TestValidateSchema

### Community 149 - "._record_research_integrity"
Cohesion: 0.22
Nodes (6): Any, Typed, versioned profile for the active research contract., Small stable surface for comparing experiments. The legacy config module…, ResearchProfile, Append one auditable record after a pipeline evaluation completes., test_research_profile_is_stable_and_versioned()

### Community 150 - "TestDeferredWarmup"
Cohesion: 0.20
Nodes (6): Unit tests for the ``defer_warmup`` flag on ``Rule_Pool_Generator``. When…, Existing callers without defer_warmup still warm at init., The configure_phase2_gpu_runtime call is inside 'if not self._defer_warmup:'…, _run_cluster_islands passes defer_warmup=True to all generators., _run_cluster_islands calls warmup_phase2_gpu_kernels per cluster., TestDeferredWarmup

### Community 151 - ".skip_if_valid"
Cohesion: 0.12
Nodes (9): Hash an artifact without loading it all into RAM., Load existing pool if valid, return None if missing., Atomically bind the current pool bytes to a Phase 2 input identity., Return a schema-valid pool proven to match this run's inputs. Bare historical…, Validate the structure of a loaded pool JSON. Raises ValueError if the schema…, _sha256_path(), _validate_pool_schema(), TestLoadPool (+1 more)

### Community 152 - "BFS and DFS Graph Traversal"
Cohesion: 0.20
Nodes (10): CLAUDE.md Graphify Integration, Constrained Query Expansion, BFS and DFS Graph Traversal, Graph Work Memory, Existing Graph Fast Path, CLAUDE.md Graphify Integration, Constrained Query Expansion, BFS and DFS Graph Traversal (+2 more)

### Community 153 - "TestModePriority"
Cohesion: 0.18
Nodes (6): Values {0, 1} match both binary and ternary criteria; binary wins., Values {-1, 0, 1} match ternary; should NOT fall through to signed., Adding value 2 to {0, 1} breaks binary → falls through to positive., Adding value 2 to {-1, 0, 1} breaks ternary → falls through to…, Adding value 2 to {-1, 0, 1} with low zero_ratio → signed., TestModePriority

### Community 154 - "BUG-001 — Split cache trusted mtime instead of source contents"
Cohesion: 0.22
Nodes (9): Actual behavior, BUG-001 — Split cache trusted mtime instead of source contents, Expected behavior, Fix, Regression test, Reproduction, Research/trading impact, Root cause (+1 more)

### Community 155 - "TestSparsePositiveMode"
Cohesion: 0.22
Nodes (5): All non-negative, zero_ratio > 0.3 → sparse_positive., All zeros: zero_ratio = 1.0 > 0.3, min = 0 → sparse_positive., zero_ratio just above 0.3 → sparse_positive., zero_ratio must be computed on the full series, not just non-NaN., TestSparsePositiveMode

### Community 156 - "TestZeroRatioBoundary"
Cohesion: 0.22
Nodes (5): Exactly 30% zeros with non-negative values → positive (not sparse_positive)., 31% zeros with non-negative values → sparse_positive., Exactly 30% zeros with negative values → signed (not sparse_signed)., Just above 30% zeros with negative values → sparse_signed., TestZeroRatioBoundary

### Community 158 - "TestGlobalMetricsCacheClearing"
Cohesion: 0.27
Nodes (6): When seeded_keys is empty, nothing is removed., When all keys are seeded, cache becomes empty., Verify that only seeded keys are removed from the global cache., Create a deterministic chromosome key., Cache entries matching seeded_keys are removed; non-matching survive., TestGlobalMetricsCacheClearing

### Community 159 - "BUG-002 — Strategy writer allowed non-finite and non-positive risk fields"
Cohesion: 0.22
Nodes (9): Actual behavior, BUG-002 — Strategy writer allowed non-finite and non-positive risk fields, Expected behavior, Fix, Regression test, Reproduction, Research/trading impact, Root cause (+1 more)

### Community 160 - "BUG-004 — Phase 5 evaluated explicitly RB-rejected strategies"
Cohesion: 0.22
Nodes (9): Actual behavior, BUG-004 — Phase 5 evaluated explicitly RB-rejected strategies, Expected behavior, Fix, Regression test, Reproduction, Research/trading impact, Root cause (+1 more)

### Community 161 - "_resolve_plateau_patience"
Cohesion: 0.24
Nodes (6): Resolve the plateau patience value based on profile and stage. Cluster/orphan…, _resolve_plateau_patience(), Tests for _resolve_plateau_patience helper., Helper to create a Phase2StageParams with controlled patience., Regression: Stage A min_gen=30 used to disable island plateau., TestResolvePlateauPatience

### Community 162 - "TestNaNHandling"
Cohesion: 0.29
Nodes (4): All-NaN series: unique_vals = [], n_unique = 0 ≤ 2, set() ⊆ {0,1} → binary., NaN is not == 0, so it does not inflate zero_ratio., NaN should not push zero_ratio above threshold., TestNaNHandling

### Community 163 - "TestSparseSignedMode"
Cohesion: 0.33
Nodes (3): Has negative values, zero_ratio > 0.3 → sparse_signed., NaN does not count as zero; zero_ratio on full series., TestSparseSignedMode

### Community 164 - "BUG-005 — CPU reference arithmetic diverged from canonical evaluator precision"
Cohesion: 0.22
Nodes (9): Actual behavior, BUG-005 — CPU reference arithmetic diverged from canonical evaluator precision, Expected behavior, Fix, Regression test, Reproduction, Research/trading impact, Root cause (+1 more)

### Community 165 - "test_evaluator_health.py"
Cohesion: 0.17
Nodes (7): Unit tests for ``evaluator_health_penalty`` and ``execution_ok`` (Task 4).…, Verify that evaluator health penalty is subtracted from combo score., When evaluator_health_penalty returns > 0, the combo score is lower., When evaluator health is fine, no penalty is applied., Both public functions are importable from the module., TestHealthPenaltyWiredIntoRB, TestModuleImportable

### Community 166 - "test_phase2_rule_pool.py"
Cohesion: 0.06
Nodes (32): _crowding_distance(), _dominates(), _non_dominated_sort(), Return True if solution *a* dominates *b* (all objectives ≤, at least one <)., NSGA-II non-dominated sorting. Parameters ---------- objectives : np.ndarray…, Compute crowding distance for solutions in *front*. Parameters ----------…, _isolate_phase2_archive_paths(), fixture (+24 more)

### Community 167 - "BUG-006 — Phase 1 resume reused schema-valid selections without input identity"
Cohesion: 0.22
Nodes (9): Actual behavior, BUG-006 — Phase 1 resume reused schema-valid selections without input identity, Expected behavior, Fix, Regression test, Reproduction, Research/trading impact, Root cause (+1 more)

### Community 168 - "BUG-001 — Trend thresholds crossed the effective training boundary"
Cohesion: 0.20
Nodes (10): Actual behavior, BUG-001 — Trend thresholds crossed the effective training boundary, Confirmed Bugs, Expected behavior, Fix, Regression test, Reproduction, Research/trading impact (+2 more)

### Community 169 - "BUG-007 — Phase 2 resume reused pools without a semantic identity"
Cohesion: 0.22
Nodes (9): Actual behavior, BUG-007 — Phase 2 resume reused pools without a semantic identity, Expected behavior, Fix, Regression test, Reproduction, Research/trading impact, Root cause (+1 more)

### Community 170 - "BUG-008 — Standalone phase commands bypassed prerequisite provenance checks"
Cohesion: 0.22
Nodes (9): Actual behavior, BUG-008 — Standalone phase commands bypassed prerequisite provenance checks, Expected behavior, Fix, Regression test, Reproduction, Research/trading impact, Root cause (+1 more)

### Community 171 - "TestNormalizeForAssociation"
Cohesion: 0.20
Nodes (6): Verify rank normalization does not crash on degenerate inputs., All-equal objective values should produce valid normalised output with no NaN,…, Single-row input should not crash., Two objectives, all-equal values, should produce valid output. After rank…, Mixed values (not all equal) should still work., TestNormalizeForAssociation

### Community 172 - "TestHallOfFameTrim"
Cohesion: 0.25
Nodes (5): Verify that hall_of_fame is capped to PHASE2_HOF_EPOCH_CARRYOVER=10., HoF with >10 entries is trimmed to exactly 10 oldest entries., HoF with <=10 entries is left unchanged., Empty HoF does not crash during trim logic., TestHallOfFameTrim

### Community 173 - "passes_pool_admission_gate"
Cohesion: 0.05
Nodes (27): passes_pool_admission_gate(), Hard gate for Phase 2 pool/archive on merged holdout metrics. When…, MonkeyPatch, train=15%/val=4% (3.75× ratio, gap=11pp < 16pp) is REJECTED by the ratio gate…, train=15%/val=10% (1.5× ratio) is ADMITTED., train/val ≈2.5× is ADMITTED when under OVERFIT_RATIO_FLOOR and gap gate., With PHASE2_OVERFIT_RATIO_FLOOR=0.0, the ratio gate is disabled and the high-…, fixture (+19 more)

### Community 174 - "ConfigError"
Cohesion: 0.60
Nodes (5): _config_check(), ConfigError, _finite_config_number(), Raised when a configuration violates a cross-parameter contract., _validate_config_grid()

### Community 175 - "evaluator_health.py"
Cohesion: 0.29
Nodes (6): evaluator_health.py — Pure functions for evaluator-failure-mode awareness.…, Read a numeric metric, returning *default* for missing / None / NaN / Inf., Read an integer metric safely., _safe_float(), _safe_int(), Scoring helpers shared across pipeline phases. Re-exports…

### Community 176 - "test_gpu_engine_import_does_not_crash_on_jax_failure"
Cohesion: 0.33
Nodes (6): parametrize, Tests for the lazy-JAX failure path in gpu_engine.py. These tests verify that…, Verify importing gpu_engine.py handles various JAX failure modes gracefully., Build a subprocess script that simulates JAX import failure., _subprocess_script(), test_gpu_engine_import_does_not_crash_on_jax_failure()

### Community 177 - "test_jax_compat.py"
Cohesion: 0.29
Nodes (5): parametrize, Tests for JAX / GPU engine availability detection., Package init must not crash when gpu_engine import fails., test_cpu_engine_import_without_jax(), test_get_gpu_backtest_engine_class_returns_none_on_import_error()

### Community 178 - "TestExecutionHealthInGate"
Cohesion: 0.25
Nodes (5): Verify that ``gate_positive_good`` calls ``execution_ok`` when flagged., When require_execution_health=True and execution_ok fails, gate returns False., When require_execution_health=True and execution_ok passes, gate still returns…, When require_execution_health=False, gate ignores raw_signal_count., TestExecutionHealthInGate

### Community 179 - "_series"
Cohesion: 0.36
Nodes (3): NaN values should not affect binary classification., _series(), TestBinaryMode

### Community 180 - "_pareto_sortino_stats"
Cohesion: 0.47
Nodes (3): _pareto_sortino_stats(), Aggregate raw Sortino and return health over the current Pareto front., TestParetoSortinoStats

### Community 181 - "TestPerSymbolMetrics"
Cohesion: 0.40
Nodes (3): Per-symbol metrics should reflect actual trade distribution., Winning trades should produce positive net_pnl per symbol., TestPerSymbolMetrics

### Community 182 - "TestEdgeCases"
Cohesion: 0.29
Nodes (4): Single row: floor(1 * train_frac) = 0, and 288-bar embargo consumes it., Small symbol where 288-bar embargo leaves no validation rows., For large N, train/total should be very close to HOLDOUT_TRAIN_FRACTION., TestEdgeCases

### Community 183 - "TestSignedMode"
Cohesion: 0.33
Nodes (3): Has negative values, zero_ratio ≤ 0.3 → signed., zero_ratio == 0.3 is NOT > 0.3, so mode is signed (not sparse_signed)., TestSignedMode

### Community 184 - "TestEvalCvFoldReturns"
Cohesion: 0.33
Nodes (4): Verify the helper handles None / empty fold_engines safely., fold_engines=None returns None without crashing., fold_engines=[] returns None without crashing., TestEvalCvFoldReturns

### Community 185 - "TestPerSymbolIndependence"
Cohesion: 0.40
Nodes (3): Each symbol's split point is computed from its own row count., Symbols with different sizes each get the correct floor(N * train_frac) split., TestPerSymbolIndependence

### Community 187 - "_legacy_writer_contract"
Cohesion: 0.67
Nodes (3): _legacy_writer_contract(), fixture, These schema tests predate mandatory trend context.

### Community 189 - "opencode.json"
Cohesion: 0.50
Nodes (3): plugin, $schema, .opencode/plugins/graphify.js

### Community 190 - "test_rb_governor_tail_holdout.py"
Cohesion: 0.12
Nodes (17): _MockEngine, DataFrame, Unit tests for RB Governor tail-holdout path in risk grid. Covers: -…, With tail_holdout_frac=0.25, a tail engine is returned with ~25% of data., With tail_holdout_frac=0.0, no tail engine., Single symbol with tail holdout still works., Verify _optimize_risk with tail_holdout_engine adds tail fields to final…, When tail_holdout_engine is provided, the final history entry contains… (+9 more)

## Knowledge Gaps
- **128 isolated node(s):** `$schema`, `.opencode/plugins/graphify.js`, `Executive Summary`, `Root cause`, `Reproduction` (+123 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **9 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `CPUBacktestEngine` connect `CPUBacktestEngine` to `Rule_Pool_Generator`, `_make_engine`, `_score_metrics`, `OOS_Evaluator`, `_make_df`, `_apply_monthly_admission_gate`, `test_cpu_engine_properties.py`, `CandidateRecord`, `nested_walk_forward.py`, `TestCausalPublicationTiming`, `_symbol_specialized_variants`, `gpu_engine.py`, `test_certificate_first_selection.py`, `phase5_oos.py`, `test_jax_compat.py`, `_compute_rule_signal_mask`, `TestPerSymbolMetrics`, `TestEvalCvFoldReturns`, `GPUBacktestEngine`, `TestGPUCPUNumericalParity`, `ValueError`, `._build_engine_for_df`, `._build_per_symbol_rows`, `test_rb_governor_tail_holdout.py`, `barrier_column_names`, `_apply_dynamic_rule`, `test_cpu_engine.py`, `test_gpu_engine.py`, `TestPrecomputeReleaseIndices`, `filter_migrants_for_cluster`, `test_gpu_engine_properties.py`, `writer.py`, `Data_Loader`, `get_gpu_backtest_engine_class`, `phase2_island_scheduler.py`, `baselines.py`, `phase2_rule_pool.py`, `_jax_compute_trade_outcomes`, `._simulate_rule_set_entries`, `TestComputeRuleSignals`, `TestMakeWalkForwardFoldEngines`, `MonthlyWindowSummary`, `._engine`?**
  _High betweenness centrality (0.136) - this node is a cross-community bridge._
- **Why does `Reporter` connect `Reporter` to `Rule_Pool_Generator`, `TestPlotPhase2Metrics`, `TestEquityCurveDateAxis`, `OOS_Evaluator`, `phase2_rule_pool.py`, `._ensure_dir`, `TestWriteStrategyEvaluationTable`, `TestPlotPhase2Pnl`, `prop_settings`, `.run`, `phase5_oos.py`, `TestPlotPerRuleBreakdown`, `TestWritePerSymbolCsv`, `test_reporter.py`?**
  _High betweenness centrality (0.102) - this node is a cross-community bridge._
- **Why does `Rule_Pool_Generator` connect `Rule_Pool_Generator` to `CPUBacktestEngine`, `log_memory_rss`, `run_pipeline.py`, `.run`, `_get_dont_cares`, `compute_phase2_objectives_from_metrics`, `_run_cluster_islands`, `Pipeline_Orchestrator`, `_apply_monthly_admission_gate`, `TestRefreshObjectivesOnResumeGate`, `_make_train_df`, `TestDeferredWarmup`, `.skip_if_valid`, `test_phase2_rule_pool.py`, `test_phase2_window_rotation.py`, `_derive_island_seed`, `test_phase2_rule_pool_properties.py`, `_pareto_sortino_stats`, `TestInitPopulation`, `._build_engine_for_df`, `TestRulePoolGeneratorRun`, `test_plateau_state_leak.py`, `test_crash_fix_and_run_logging.py`, `filter_migrants_for_cluster`, `downcast_numeric_df`, `phase2_island_scheduler.py`, `.save_archive`, `Reporter`, `phase2_rule_pool.py`, `TestValLeakGate`?**
  _High betweenness centrality (0.085) - this node is a cross-community bridge._
- **Are the 65 inferred relationships involving `CPUBacktestEngine` (e.g. with `GPUBacktestEngine` and `JointPortfolioEngine`) actually correct?**
  _`CPUBacktestEngine` has 65 INFERRED edges - model-reasoned connections that need verification._
- **Are the 15 inferred relationships involving `Reporter` (e.g. with `CvFoldValEvaluator` and `Rule_Pool_Generator`) actually correct?**
  _`Reporter` has 15 INFERRED edges - model-reasoned connections that need verification._
- **Are the 62 inferred relationships involving `Rule_Pool_Generator` (e.g. with `CPUBacktestEngine` and `Reporter`) actually correct?**
  _`Rule_Pool_Generator` has 62 INFERRED edges - model-reasoned connections that need verification._
- **Are the 31 inferred relationships involving `Output_Writer` (e.g. with `OOS_Evaluator` and `_Phase5JSONEncoder`) actually correct?**
  _`Output_Writer` has 31 INFERRED edges - model-reasoned connections that need verification._