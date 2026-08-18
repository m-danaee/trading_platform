# Graph Report - trading_platform  (2026-08-19)

## Corpus Check
- 192 files · ~282,273 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 4892 nodes · 11232 edges · 195 communities (185 shown, 10 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 513 edges (avg confidence: 0.56)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `0d30b98e`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- evox_runner.py
- test_evox_runner.py
- barrier_column_names
- test_cpu_engine.py
- Pipeline_Orchestrator
- TestValLeakGate
- TestRulePoolGeneratorRun
- _score_metrics
- _write_and_reload
- _make_df
- Data_Splitter
- run_phase2_evolution
- splitter.py
- Data_Loader
- test_run_pipeline.py
- _apply_monthly_admission_gate
- test_feature_detector_properties.py
- _make_train_df
- _split
- test_cpu_engine_properties.py
- detect_feature_mode
- selector.py
- Feature_Selector
- build_master_temporal_folds
- Reporter
- test_encoder_properties.py
- write_evaluator_clean
- CandidateRecord
- Rule_Pool_Generator
- _preserve_deployable_elites
- maybe_log_generation
- RB Governor production path
- _select_diverse_subset
- test_rb_min_symbols.py
- Graphify Pipeline
- Output_Writer
- config.py
- TestEquityCurveDateAxis
- _build_pool_from_archive
- resolve_phase2_stage_params
- _loader_from_rows
- phase2_rule_pool.py
- gpu_engine.py
- _gpu_runtime.py
- ._ensure_dir
- compose_hierarchical_signals
- test_certificate_first_selection.py
- test_mtf_pipeline_integration.py
- optuna_search.py
- Hierarchical Multi-Timeframe Rule Discovery System Specification
- test_phase2_rule_pool_properties.py
- _init_population
- validate_config
- test_crash_fix_and_run_logging.py
- test_reporter.py
- _dominates
- _get_dont_cares
- _make_rule
- non_dominated_sort
- ndarray
- nested_walk_forward.py
- test_phase2_use_gpu_flag.py
- PurgedFold
- apply_fuzzy_feature_scaling
- test_plateau_state_leak.py
- _make_df
- dashboard.py
- GPUBacktestEngine
- test_data_loader_properties.py
- ValueError
- _m
- _apply_dynamic_rule
- cpu_engine.py
- .encode_condition
- execution_ok
- TestWriteStrategyEvaluationTable
- test_gpu_engine.py
- TestRunLogHandlerLifecycle
- test_output_writer_properties.py
- test_feature_selector_properties.py
- .run
- _symbol_specialized_variants
- TestPlotPerRuleBreakdown
- _build_entries_from_rule_set
- Feature_Detector
- _build_target
- TestSavePerSymbolCsv
- test_gpu_engine_properties.py
- TestGPUCPUNumericalParity
- build_complete_higher_bars
- run_phase2_evolution_epoch
- set_purged_wf_reference_rows
- test_phase2_window_rotation.py
- Encoder
- compute_phase2_objectives_from_metrics
- test_crash_fix_properties.py
- mtf/__init__.py
- downcast_numeric_df
- .load_strategies
- gate_positive_good
- .get_dont_care
- phase5_oos.py
- TestRemoveRedundantFeatures
- ._build_per_symbol_rows
- hypothesis_config.py
- TestPlotDistributionAndEquity
- baselines.py
- prop_settings
- _jax_compute_trade_outcomes
- _derive_epoch_seed
- OOS_Evaluator
- research_integrity.py
- _validate_pool_schema
- _derive_val_sample_seed
- TestHammingThresholdAutoScale
- .decode_chromosome
- ValidationError
- TestComputeRuleSignals
- TestParetoCollapseWarningGate
- test_rb_governor_tail_holdout.py
- _compute_stability
- passes_pool_admission_gate
- MonthlyWindowSummary
- ._engine
- test_rb_concentration_tail_fail_closed.py
- constrained_non_dominated_sort
- cv_folds_only
- resolve_evolution_floors
- stratification_scenario_strategy
- TestSpearmanSignConsistency
- compute_labels
- log_memory_rss
- ._load_datasets_by_split
- rolling_cv.py
- _should_post_restart_early_stop_phase2
- context_contract_digest
- _apply_colab_gpu_defaults
- _pareto_sortino_stats
- get_gpu_backtest_engine_class
- CPUBacktestEngine
- composite
- test_feature_selector.py
- phase2_support.py
- run_pipeline.py
- TestRefreshObjectivesOnResumeGate
- conftest.py
- test_config_additions.py
- .skip_if_valid
- _validate_schema
- test_directional_evaluator.py
- Global Constraints
- test_phase2_rule_pool.py
- BFS and DFS Graph Traversal
- TestPlotPhase2Metrics
- TestPoolAdmissionOverfitRatioGate
- test_rb_fail_closed.py
- .skip_if_valid
- .simulate_rule_set_batch
- TestGlobalMetricsCacheClearing
- test_evaluator_health.py
- _nsga3_environmental_selection
- _metrics_snapshot
- test_phase5_oos_properties.py
- trade_support_penalty
- TestSparsePositiveMode
- evaluator_health.py
- mandatory_context_conditions
- TestZeroRatioBoundary
- _build_rule_signal_mask
- main
- trade_support_penalty
- TestPrecomputeReleaseIndices
- TestHallOfFameTrim
- _compute_rule_signal_mask
- TestNaNHandling
- scoring/__init__.py
- test_gpu_engine_import_does_not_crash_on_jax_failure
- test_jax_compat.py
- TestExecutionHealthInGate
- TestSparseSignedMode
- effective_phase2_val_return_floor_pct
- TestF3PathResolution
- _identity_value
- load_cv_folds_manifest
- TestEvalCvFoldReturns
- opencode.json
- graphify.js
- Obsolete implementation cleanup policy
- data/__init__.py
- features/__init__.py
- gpu_fuzzy_trader/__init__.py
- output/__init__.py
- phases/__init__.py

## God Nodes (most connected - your core abstractions)
1. `Reporter` - 159 edges
2. `CPUBacktestEngine` - 150 edges
3. `Rule_Pool_Generator` - 126 edges
4. `Pipeline_Orchestrator` - 88 edges
5. `Output_Writer` - 86 edges
6. `prop_settings()` - 79 edges
7. `OOS_Evaluator` - 61 edges
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
- **Fail-closed research boundary** — readme_configuration_contract, readme_symbol_specialist_islands, readme_rb_governor, readme_holdout_acceptance_contract, readme_fail_closed_deployment [INFERRED 0.95]
- **Hardware-aware execution flow** — run_rtx4050_execution_policy, run_colab_t4_path, run_pipeline_orchestrator, readme_hybrid_execution_policy [INFERRED 0.95]

## Communities (195 total, 10 thin omitted)

### Community 0 - "evox_runner.py"
Cohesion: 0.05
Nodes (89): _assign_eval_result(), _behavior_descriptor(), _binary_tournament_pick(), _build_diversity_reference(), clear_global_metrics_cache(), _constraint_violations(), _count_deployable_preview(), _count_pop_viable() (+81 more)

### Community 1 - "test_evox_runner.py"
Cohesion: 0.05
Nodes (44): _diversity_recovery_min_unique_ratio(), _evaluate_population_indices(), extract_deployable_migrants(), _inherit_val_metrics_from_global_cache(), Phase2EvolutionState, Evaluate unevaluated individuals, preferring batch simulate_rule_batch., Copy val_* from global cache for identical chromosomes when val is skipped.…, Return elite deployable-preview entries suitable for guarded migration. (+36 more)

### Community 2 - "barrier_column_names"
Cohesion: 0.12
Nodes (20): attach_barrier_outcomes(), barrier_column_names(), configured_barrier_pairs(), _first_touch_for_symbol(), _first_touch_for_symbol_numba(), _first_touch_for_symbol_python(), _number_token(), DataFrame (+12 more)

### Community 3 - "test_cpu_engine.py"
Cohesion: 0.07
Nodes (30): _make_df(), _make_engine(), DataFrame, Unit tests for CPUBacktestEngine. Tests verify exact evaluator_v5.ipynb…, Simulate catastrophic losses to trigger account ruin. With min_288=0…, Trades with tiny equity should be skipped., Per-symbol metrics should reflect actual trade distribution., Winning trades should produce positive net_pnl per symbol. (+22 more)

### Community 4 - "Pipeline_Orchestrator"
Cohesion: 0.05
Nodes (68): FileHandler, count_trials(), Estimate the number of adaptive evaluations represented by artifacts., _dataframe_schema_sha256(), _dataframe_sha256(), _git_commit_id(), _log_pipeline_config(), _merge_mtf_lwc_runtime_columns() (+60 more)

### Community 5 - "TestValLeakGate"
Cohesion: 0.20
Nodes (10): C6: Val-derived penalties must be gated behind JOINT_TRAIN_VAL or…, Return standard monkeypatching for clean baseline metrics., Apply base settings with optional overrides., Metrics that trigger no train-side penalties., Val metrics that WOULD trigger penalties if the gate were open., When both JOINT_TRAIN_VAL and VAL_IN_FITNESS_PENALTY are False, val-derived…, Bad val must not set feasibility_violation when gate is closed., When VAL_IN_FITNESS_PENALTY=True, val-derived penalties DO enter objectives. (+2 more)

### Community 6 - "TestRulePoolGeneratorRun"
Cohesion: 0.09
Nodes (12): Integration tests using tiny population and generation counts., In holdout mode, val engine must be built for pool admission even when…, All pool entries must have active conditions within [MIN_CONDITIONS,…, Phase 2 must use static TP=PHASE2_TP, SL=PHASE2_SL,…, Pool entries must have executed_trades >= MIN_TRADE_POOL_FLOOR., Verify that Rule_Pool_Generator's persistent self._rng advances across multiple…, After two run_epoch() calls, the internal RNG state must differ from the…, The RNG should produce *different* sequences in two consecutive run_epoch()… (+4 more)

### Community 7 - "_score_metrics"
Cohesion: 0.08
Nodes (36): _combined_return_score(), _evaluate_ruleset(), _optimize_risk(), Return (ok, bonus, penalty) for the desired train-valid balance shape. In…, Dominant objective: return/DD with train-valid balance, plus CV-fold…, Profit objective for lenient rule addition, but now evaluator_v5 aware. A new…, _score_metrics(), _train_valid_shape() (+28 more)

### Community 8 - "_write_and_reload"
Cohesion: 0.23
Nodes (5): _make_rule_set(), Write rule_set to a temp file and reload the raw JSON., TestWriteHappyPath, TestWriteTruncation, _write_and_reload()

### Community 9 - "_make_df"
Cohesion: 0.08
Nodes (22): _make_df(), _make_engine(), DataFrame, MonkeyPatch, Chromosome positions must follow feature_modes insertion order., Exact CPU re-evaluation must interpret genes like the GPU path., Chromosome that matches nothing returns 0 executed trades., Zero-signal chunks should use reject metrics without scanning. (+14 more)

### Community 10 - "Data_Splitter"
Cohesion: 0.12
Nodes (22): Data_Splitter, load_cached_split_if_fresh(), Load cached split parquets when they are newer than the source CSV. Validates…, Chronological train/validation splitter., Module-level wrapper around ``Data_Splitter.split_and_persist``., split_and_persist(), _make_df(), _make_timestamps() (+14 more)

### Community 11 - "run_phase2_evolution"
Cohesion: 0.05
Nodes (42): Shared val-cadence check for both NSGA-II fallback and NSGA-III loops. Val…, Run Phase 2 NSGA-III evolution. Returns (pareto_pool, history) or state., run_phase2_evolution(), _should_plateau_early_stop_phase2(), _should_run_val_this_gen(), _stage_mutation_rate(), The fallback must not switch f3 from CV return to PF after gen 0., TestRunPhase2EvolutionFallback (+34 more)

### Community 12 - "splitter.py"
Cohesion: 0.14
Nodes (20): Number of leading per-symbol rows belonging to the training prefix. Shared by…, train_prefix_row_count(), _chronological_half_split(), _file_sha256(), _holdout_embargo_split(), _purged_walk_forward_split(), DataFrame, data/splitter.py — Data_Splitter Per-symbol chronological split for Phase 2,… (+12 more)

### Community 13 - "Data_Loader"
Cohesion: 0.06
Nodes (28): Data_Loader, Stateless data loader for the GPU-Fuzzy Trading Pipeline., _isolate_phase5_reporter_outputs(), fixture, Unit tests for gpu_fuzzy_trader.phases.phase5_oos.OOS_Evaluator Tests cover: -…, Verify plot_equity_curve is called for all three splits and handles empty logs., Helper: run OOS_Evaluator.run() with mocked internals and capture plot calls., Override module-level path dicts and return originals. (+20 more)

### Community 14 - "test_run_pipeline.py"
Cohesion: 0.07
Nodes (47): _context_coverage_for_direction(), _context_coverage_preflight(), _context_coverage_report(), context_floor_failures(), _log_phase_entry(), Append a structured JSON line to the pipeline log file. Parameters ----------…, Return the active output root for this run., Temporarily rebind all cached output paths for one pipeline run. (+39 more)

### Community 15 - "_apply_monthly_admission_gate"
Cohesion: 0.06
Nodes (29): _apply_monthly_admission_gate(), _evaluate_rule_on_window(), _monthly_window_metrics(), Evaluate a single pool rule on a single monthly window. Returns the full window…, Normalize a window evaluator result for the monthly gate. The float fallback…, Apply the monthly-window shadow-test gate to a pool of rules. Each rule is…, _DeterministicEvaluator, _DeterministicMetricsEvaluator (+21 more)

### Community 16 - "test_feature_detector_properties.py"
Cohesion: 0.09
Nodes (45): all_nan_series(), arbitrary_numeric_series(), binary_series(), large_value_series(), positive_series(), composite, DrawFn, given (+37 more)

### Community 17 - "_make_train_df"
Cohesion: 0.08
Nodes (22): Sample up to *total_rows* rows, distributed equally across symbols. A single…, _sample_df(), _make_train_df(), DataFrame, Critical: bars must be contiguous so the backtest engine preserves temporal…, Random start must be bounded so the slice always fits forward., divmod distribution gives exactly total_rows (no rounding loss). 701_000 % 14…, total_rows < n_symbols must NOT force 1 row per symbol. (+14 more)

### Community 18 - "_split"
Cohesion: 0.06
Nodes (22): DataFrame, Helper: build df, patch paths, run split, return (train, val)., Compute expected train/val per symbol under holdout+embargo split., floor(N * HOLDOUT_TRAIN_FRACTION) rows go to train., Remaining rows after embargo gap go to validation., For N=101: floor(101 * HOLDOUT_TRAIN_FRACTION) used, not round., train + validation + embargo dropped == total rows., train + val + embargo_dropped == total for each symbol. (+14 more)

### Community 19 - "test_cpu_engine_properties.py"
Cohesion: 0.10
Nodes (31): _expected_outcome(), _make_engine(), _make_engine_custom(), _make_minimal_df(), _make_simple_df(), DataFrame, given, Property-based tests for gpu_fuzzy_trader.backtest.cpu_engine.CPUBacktestEngine… (+23 more)

### Community 20 - "detect_feature_mode"
Cohesion: 0.09
Nodes (19): detect_feature_mode(), Module-level convenience wrapper around Feature_Detector.detect_feature_mode., Has negative values, zero_ratio ≤ 0.3 → signed., zero_ratio == 0.3 is NOT > 0.3, so mode is signed (not sparse_signed)., Values {0, 1} match both binary and ternary criteria; binary wins., Values {-1, 0, 1} match ternary; should NOT fall through to signed., Adding value 2 to {0, 1} breaks binary → falls through to positive., Adding value 2 to {-1, 0, 1} breaks ternary → falls through to… (+11 more)

### Community 21 - "selector.py"
Cohesion: 0.08
Nodes (39): get_seed(), Return a per-process random seed (stable within a run, different across runs).…, build_phase1_shared_context(), _build_symbol_masks(), _candidate_feature_columns(), _check_spearman_sign_consistency(), _compute_chronological_stationarity_scores(), _frame_identity() (+31 more)

### Community 22 - "Feature_Selector"
Cohesion: 0.10
Nodes (19): Feature_Selector, Score and rank features separately for long and short directions., _make_train_df(), MonkeyPatch, Create a minimal training DataFrame with label columns and feature columns., Label columns must not appear in selected features., Meta columns must not appear in selected features., Loader internal columns and ``_``-prefixed names are not candidates. (+11 more)

### Community 23 - "build_master_temporal_folds"
Cohesion: 0.11
Nodes (35): apply_purge_embargo(), build_master_temporal_folds(), export_fold_boundaries(), _format_fold_predictions(), generate_oof_scores(), _get_datetime_series(), Any, DataFrame (+27 more)

### Community 24 - "Reporter"
Cohesion: 0.10
Nodes (13): Generates visual and tabular reports for each pipeline phase. All output files…, Reporter, _make_per_symbol_metrics(), _make_pnl_history(), _make_trade_log(), History entries with missing keys should not raise., Symbols with missing sub-keys should default to 0., Create a minimal Phase 2 history list with PnL fields. (+5 more)

### Community 25 - "test_encoder_properties.py"
Cohesion: 0.09
Nodes (37): all_active_chromosome_strategy(), chromosome_with_dont_cares_strategy(), feature_name_strategy(), composite, DrawFn, given, ndarray, Property-based tests for gpu_fuzzy_trader.features.encoder.Encoder **Validates:… (+29 more)

### Community 26 - "write_evaluator_clean"
Cohesion: 0.08
Nodes (29): _maybe_write_evaluator_clean(), Write a stripped strategy file containing only ``direction`` and ``rules_set``.…, Write a stripped strategy file alongside the main strategy JSON. This is a…, write_evaluator_clean(), _make_rule(), minimal_strategy(), fixture, Path (+21 more)

### Community 27 - "CandidateRecord"
Cohesion: 0.10
Nodes (40): effective_rb_min_distinct_symbols(), Return the RB coverage target for the active debug universe. Full runs keep…, _candidate_coverage_symbols(), _candidate_positive_symbols(), CandidateRecord, _compose_ruleset(), _diversification_beam(), _diversification_shortlist() (+32 more)

### Community 28 - "Rule_Pool_Generator"
Cohesion: 0.09
Nodes (18): _downsample_chronological(), DataFrame, Phase 2: GPU-accelerated multi-objective evolutionary rule pool generation.…, Build train/val backtest engines., Restore slimmed training data from cache (no re-sampling needed)., Rebuild engines after ``park_engines`` dropped GPU state., Build the selected Phase 2 backend for the sampled train frame., Return whether this generator should avoid allocating a JAX engine.… (+10 more)

### Community 29 - "_preserve_deployable_elites"
Cohesion: 0.09
Nodes (25): _build_rank_and_crowding(), environmental_selection_nsga2(), _preserve_deployable_elites(), Per-individual Pareto rank (lower is better) and crowding distance., Canonical NSGA-II truncation on a 2N merged population., Force-preserve top-K deployable-archive elites in the live population.…, _make_chromosome(), _make_deployable_entry() (+17 more)

### Community 30 - "maybe_log_generation"
Cohesion: 0.09
Nodes (18): generation_log_interval(), iteration_log_interval(), log_generation(), maybe_log_generation(), log_progress.py — Throttled progress logging for long pipeline loops., Return how often to log generation progress. Uses LOG_GENERATION_INTERVAL from…, Log generation progress when the step matches the throttle interval., Log first step, last step, and every *interval* steps in between. (+10 more)

### Community 31 - "RB Governor production path"
Cohesion: 0.09
Nodes (27): Local low-memory test policy, Virtual environment command policy, Package compile gate, Locked research dependencies, Focused low-memory test gate, Python 3.11 CI runtime, Research CI workflow, Research artifact audit trail (+19 more)

### Community 32 - "_select_diverse_subset"
Cohesion: 0.08
Nodes (23): _normalize_for_association(), Max-min Hamming diversity sampling: greedy pick farthest from chosen. Returns…, Rank-based normalization (robust to outliers like trade_penalty=50)., _select_diverse_subset(), ndarray, Tests for H5/M4/M5 evolution convergence behaviors. Covers: - HoF trimming at…, Verify _select_diverse_subset correctness for edge cases., Create n distinct dense chromosomes. (+15 more)

### Community 33 - "test_rb_min_symbols.py"
Cohesion: 0.11
Nodes (24): _symbols_in_rules(), _dummy_df(), _make_candidate_records(), _mock_train_metrics(), _multi_symbol_rules(), _no_symbol_rule(), DataFrame, Tests for RB Governor min-distinct-symbols hard gate. After final opt_rules… (+16 more)

### Community 34 - "Graphify Pipeline"
Cohesion: 0.06
Nodes (36): Folder Watcher, URL Ingestion, Conditional Graph Exports, Graphify MCP Server, Extraction Confidence Rubric, Deterministic Full-Path Node IDs, Semantic Hyperedges, Cross-Repository Graph Merge (+28 more)

### Community 35 - "Output_Writer"
Cohesion: 0.11
Nodes (9): Output_Writer, Serializes RuleSet dicts to JSON and loads/validates existing JSON files.…, Verify the exact example from the spec works end-to-end., TestLoadAndValidateErrors, TestLoadAndValidateHappyPath, TestSpecExample, TestValidationErrorType, TestWriteDirectionValidation (+1 more)

### Community 36 - "config.py"
Cohesion: 0.07
Nodes (35): _config_check(), ConfigError, context_contract(), _debug_symbol_universe_size(), effective_min_trade_support(), effective_monthly_min_trades(), effective_val_trade_floor_for_objectives(), filter_df_to_symbols() (+27 more)

### Community 37 - "TestEquityCurveDateAxis"
Cohesion: 0.09
Nodes (21): _make_trade_log(), DataFrame, fixture, Unit tests for date-based x-axis in Reporter.plot_equity_curve (Task 10.3).…, When Entry_Time is absent, x-axis label is 'Trade #'., When Entry_Time is all-NaN, x-axis label is 'Trade #'., axhline is called (initial-capital line) in date mode., axhline is called (initial-capital line) in Trade # mode. (+13 more)

### Community 38 - "_build_pool_from_archive"
Cohesion: 0.11
Nodes (22): _archive_direction(), attach_cv_fold_returns_batch(), _build_cpu_archive_engine(), _build_pool_from_archive(), _chromosome_batch(), _chromosome_for_pool_export(), CvFoldValEvaluator, _positive_contributor_symbols() (+14 more)

### Community 39 - "resolve_phase2_stage_params"
Cohesion: 0.10
Nodes (17): True when Stage A viability is critically low and search has plateaued., _should_inject_diversity_recovery(), _should_viability_recovery(), StageLabel, Phase 2 two-stage search hyperparameter profiles (exploration vs refinement)., Return stage-tuned hyperparameters. When *stage* is None (single-stage Phase…, resolve_phase2_stage_params(), Tiny deployable archive must not IndexError on viability recovery. (+9 more)

### Community 40 - "_loader_from_rows"
Cohesion: 0.07
Nodes (32): load_dataset(), Module-level wrapper around ``Data_Loader.load_dataset``., _base_row(), _loader_from_rows(), _make_csv(), _make_ohlcv_rows(), _make_rows(), _make_timestamps() (+24 more)

### Community 41 - "phase2_rule_pool.py"
Cohesion: 0.12
Nodes (47): _random_active_class(), _crossover(), _deployable_archive_pool_entries(), _pool_seed_chromosomes(), phase2_rule_pool.py — Rule_Pool_Generator (Phase 2) GPU-accelerated multi-…, Uniform crossover (dense per-gene or sparse per-slot)., Stack chromosome rows into a batch (dense 2D or sparse 3D)., Extract deduplicated chromosomes from a Phase 2 pool for population seeding. (+39 more)

### Community 42 - "gpu_engine.py"
Cohesion: 0.06
Nodes (47): _batch_metrics_from_array(), _build_event_batch(), _exposure_slot_capacity(), _fast_reject_result_rows(), _jax_compute_rule_signals(), _jax_compute_rule_signals_batch(), _jax_compute_rule_signals_sparse_batch(), _jax_open_slot() (+39 more)

### Community 43 - "_gpu_runtime.py"
Cohesion: 0.09
Nodes (34): phase2_should_enrich_symbol_metrics(), Return True when GPU batch eval should run a follow-up CPU enrichment pass.…, configure_phase2_gpu_runtime(), detect_gpu_memory_used_gb(), detect_gpu_vram_gb(), detect_system_ram_gb(), evict_cluster_signatures(), _iter_warmup_targets() (+26 more)

### Community 44 - "._ensure_dir"
Cohesion: 0.06
Nodes (19): Reporting and visualization sub-package., _bucket_series_by_mode(), DataFrame, Series, reporter.py — Reporting and visualization for the GPU-Fuzzy Trading Pipeline.…, Write feature-stratified performance metrics to CSV files (one per split). For…, Create parent directories for *path* if they do not exist., Compute maximum drawdown percentage from an equity series. Parameters… (+11 more)

### Community 45 - "compose_hierarchical_signals"
Cohesion: 0.10
Nodes (34): Hierarchical MTF Strategy Candidate Container. Encapsulates LWC execution…, Evaluate hierarchical soft-veto composition on input arrays., compose_bidirectional_signals(), compose_hierarchical_signals(), normalize_direction(), Any, ndarray, Series (+26 more)

### Community 46 - "test_certificate_first_selection.py"
Cohesion: 0.15
Nodes (17): _entry_validation_per_symbol_metrics(), _pool_entry_rank(), Read validation per-symbol metrics across pool schema revisions., Compute the existing deployability rank from a pool entry., Cap a pool while reserving admitted candidates for positive symbols. The…, _reserve_symbol_pool_candidates(), _BatchEngine, _candidate() (+9 more)

### Community 47 - "test_mtf_pipeline_integration.py"
Cohesion: 0.07
Nodes (32): HierarchicalStrategyCandidate, Any, ndarray, Evaluate this frozen candidate on raw OHLCV rows. No thresholds, weights,…, Convert candidate to a JSON-serializable dictionary., Construct HierarchicalStrategyCandidate from dictionary., Encapsulates a hierarchical multi-timeframe strategy with LWC/MWC/HWC rules., Compute deterministic strategy SHA-256 identifier. (+24 more)

### Community 48 - "optuna_search.py"
Cohesion: 0.10
Nodes (29): _active_search_space(), apply_trial_config(), collect_validation_metrics(), compute_score(), _copy_phase1_2_outputs(), main(), objective(), _pearson() (+21 more)

### Community 49 - "Hierarchical Multi-Timeframe Rule Discovery System Specification"
Cohesion: 0.06
Nodes (30): 1.1 Root Problem Addressed, 1.2 Target Architecture Principles, 1. System Overview & Core Objectives, 2.1 Resampling & Continuity Contract, 2.2 Independent Feature Computation, 2.3 Point-in-Time Causal Alignment, 2. Multi-Timeframe Causal Data Layer, 3.1 Profile Configurations (+22 more)

### Community 50 - "test_phase2_rule_pool_properties.py"
Cohesion: 0.11
Nodes (21): feature_infos_and_train_df(), _isolate_phase2_archive_paths(), _make_feature_infos(), _make_train_df(), composite, DataFrame, DrawFn, fixture (+13 more)

### Community 51 - "_init_population"
Cohesion: 0.10
Nodes (27): assign_strata_to_indices(), build_feature_sampling_probs(), pick_active_count(), _pick_active_index(), _pick_inactive_index(), ndarray, phase2_init.py — Sparsity-guided stratified population initialization for Phase…, Assign elite / explorer labels to non-seeded population rows. (+19 more)

### Community 52 - "validate_config"
Cohesion: 0.16
Nodes (23): effective_config_snapshot(), effective_min_profitable_symbols(), Cap cross-symbol profitability gate to the active universe size. With…, Validate all high-impact hyperparameter relationships. The function is…, Return resolved values and derived constraints for audit/reporting., Write the effective configuration snapshot and return its path., validate_config(), write_config_audit_report() (+15 more)

### Community 53 - "test_crash_fix_and_run_logging.py"
Cohesion: 0.12
Nodes (17): _capture_logs(), _make_feature_infos_crash(), _make_train_df_crash(), DataFrame, Unit tests for crash-fix-and-run-logging spec. Covers: - Task 1.1: Smoke tests…, Requirement 3.1 — save_archive is called before _release_resources in run()., Mock save_archive and _release_resources, run Rule_Pool_Generator.run() with a…, save_archive is called with self.direction as the first argument. (+9 more)

### Community 54 - "test_reporter.py"
Cohesion: 0.07
Nodes (29): _make_dataset_with_label(), _make_datasets_by_split(), _make_selected_features(), _make_stratified_dataset(), _make_stratified_datasets_by_split(), _make_stratified_logs_by_split(), _make_stratified_trade_log(), DataFrame (+21 more)

### Community 55 - "_dominates"
Cohesion: 0.14
Nodes (11): _archive_objective_vector(), _dominates(), _is_better_archive_entry(), _non_dominated_sort(), Return True if solution *a* dominates *b* (all objectives ≤, at least one <)., NSGA-II non-dominated sorting. Parameters ---------- objectives : np.ndarray…, Convert an archive entry into minimisation objectives for ranking., Return True when *candidate* should replace *incumbent* for the same chromosome. (+3 more)

### Community 56 - "_get_dont_cares"
Cohesion: 0.07
Nodes (28): _count_active_conditions(), _get_dont_cares(), _mutate(), Mutate a chromosome (returns a copy). When activating a dont_care gene, feature…, Return array of dont_care sentinels for each feature., Count active rule conditions (sparse slots or dense dont_care encoding)., sparse_to_dense(), _is_all_inactive_sparse() (+20 more)

### Community 57 - "_make_rule"
Cohesion: 0.11
Nodes (11): _legacy_writer_contract(), _make_rule(), fixture, parametrize, Unit tests for gpu_fuzzy_trader.output.writer — Output_Writer Tests cover: -…, These schema tests predate mandatory trend context., Spot-check a variety of valid fuzzy value names., TestWriteConditionValidation (+3 more)

### Community 58 - "non_dominated_sort"
Cohesion: 0.13
Nodes (26): batch_hamming_min(), crowding_distance(), _crowding_distance_numba(), _crowding_distance_py(), _disable_numba_sort(), _dominates_numba(), _dominates_py(), non_dominated_sort() (+18 more)

### Community 59 - "ndarray"
Cohesion: 0.07
Nodes (27): _append_allocated_entries(), _expectancy_lcb_pct(), _expected_shortfall_pct(), precompute_release_indices(), precompute_release_indices_from_offsets(), ndarray, Series, Simulate a rule set on rows [row_start, row_end) without copying the df. Used… (+19 more)

### Community 60 - "nested_walk_forward.py"
Cohesion: 0.13
Nodes (25): Validation helpers for monthly, nested, and multiplicity-safe research., deflated_sharpe_ratio(), estimate_pbo(), Selection-multiplicity diagnostics for strategy research artifacts., Estimate the fraction of folds where the IS winner misses OOS median. Inputs…, Return a normal-approximation deflated Sharpe probability. This is a…, Create a compact ledger-ready multiplicity report., summarize_multiplicity() (+17 more)

### Community 61 - "test_phase2_use_gpu_flag.py"
Cohesion: 0.25
Nodes (7): _minimal_backtest_df(), DataFrame, MonkeyPatch, Phase 2 respects PHASE2_USE_GPU=False without loading JAX GPU engine., The memory-safe CPU route must happen before JAX allocates arrays., A selected CPU backend must not initialize JAX just to warm up., TestPhase2UseGpuFlag

### Community 62 - "PurgedFold"
Cohesion: 0.16
Nodes (14): build_forbidden_ranges(), derive_primary_holdout(), mask_df_to_safe_region(), PurgedFold, Build per-symbol forbidden ``(start_bar, end_bar)`` ranges from folds. Each…, Drop rows whose per-symbol bar index falls inside any forbidden range., One expanding train + validation slice (per-symbol boundaries merged)., Return (train_df, val_df) from the primary holdout fold. (+6 more)

### Community 63 - "apply_fuzzy_feature_scaling"
Cohesion: 0.29
Nodes (9): apply_fuzzy_feature_scaling(), fit_fuzzy_feature_scaling(), Any, DataFrame, Train-fitted normalization for ordinal ``ff_*`` fuzzy feature codes. The…, Build a train-only scaling contract for ordinal ``ff_*`` columns., Apply an existing train-fitted scaling contract in place and return *df*., Tests for train-fitted ordinal fuzzy feature scaling. (+1 more)

### Community 64 - "test_plateau_state_leak.py"
Cohesion: 0.11
Nodes (17): _make_minimal_gen(), _mock_evolution_state(), Regression tests for plateau-state leak fixes (Fixes A + B). Fix A:…, reset_plateau=True is also passed on epoch 2 (regression for leak)., Even when two-stage is disabled, reset_plateau=True., AC-2: _island_generations_done increments by len(epoch_history)., When all requested gens execute, len(epoch_history) == epoch_gens., Early-stop: 3 actual gens run out of 10 requested, budget += 3. (+9 more)

### Community 65 - "_make_df"
Cohesion: 0.23
Nodes (9): _make_df(), _make_rule_set(), DataFrame, When no trades are executed, account_ruined must be False., When no trades are executed, total_return_pct must be 0.0., Create a minimal valid rule set dict., Create a minimal DataFrame with all required columns., Returned and saved OOS metrics come from the locked strategy. (+1 more)

### Community 66 - "dashboard.py"
Cohesion: 0.15
Nodes (25): build_dashboard_data(), _direction_data(), _history_rows(), _load_json(), main(), Any, Path, Read-only HTML dashboard for existing pipeline artifacts. The dashboard… (+17 more)

### Community 67 - "GPUBacktestEngine"
Cohesion: 0.08
Nodes (18): GPUBacktestEngine, CPU engine for rule-set simulation used by Phase 2/RB., Return whether this host's CPU path is the faster large-window path., Rule-set sim using a mask cache (delegates to the CPU engine)., Evaluate multiple rule sets (parallel CPU; optional mask cache)., RB JAX path: cached mask entry build + parallel CPU equity. Equity results…, Same interface as CPUBacktestEngine for compatibility., jax.lax.scan-based sequential equity simulation (legacy compat). Parameters… (+10 more)

### Community 68 - "test_data_loader_properties.py"
Cohesion: 0.14
Nodes (26): dataframe_with_nan_features(), dataframe_with_nan_labels(), _load_from_df(), _make_datetime_series(), composite, DataFrame, DrawFn, given (+18 more)

### Community 69 - "ValueError"
Cohesion: 0.16
Nodes (29): attach_frozen_layer_scores(), _causal_score_columns(), condition_mask(), ensemble_layer_scores(), evaluate_candidate_frame(), _or_rule_masks(), _prefix_features(), prepare_causal_mtf_frame() (+21 more)

### Community 70 - "_m"
Cohesion: 0.13
Nodes (16): evaluator_health_penalty(), Penalty for evaluator_v5 failure modes — higher is worse. Parameters ----------…, _m(), Max positions 15 > 10 → positive penalty., Max positions 8 <= 10 → no additional penalty from positions., Missing ``raw_signal_count`` → treated as 0 → no skip/exec penalty., ``raw_signal_count=0`` → no skip/exec ratio computed → no penalty., Role 'train' same as 'valid' (1.0x). (+8 more)

### Community 71 - "_apply_dynamic_rule"
Cohesion: 0.21
Nodes (4): _apply_dynamic_rule(), Apply one exported text condition using the original threshold logic. Exactly…, Test all threshold branches of _apply_dynamic_rule., TestApplyDynamicRule

### Community 72 - "cpu_engine.py"
Cohesion: 0.07
Nodes (26): compute_entry_time_priority(), _parse_condition(), cpu_engine.py — CPUBacktestEngine Exact Python/NumPy replication of…, Compute a non-annualized Sortino Ratio from per-trade returns., Parse '[feature_name] IS Fuzzy Value Name' → (feature_name, value_name)., Map each row to a timestamp priority code (evaluator_v5 parity)., _rule_symbols_for_allocation(), _safe_profit_factor() (+18 more)

### Community 73 - ".encode_condition"
Cohesion: 0.18
Nodes (4): See module-level :func:`encode_condition`., TestEncodeConditionBinary, TestEncodeConditionErrors, TestEncodeConditionTernary

### Community 74 - "execution_ok"
Cohesion: 0.15
Nodes (11): execution_ok(), Return ``True`` iff the evaluator would execute this rule set reasonably. A…, Tests for ``execution_ok``., Skip ratio 0.10 <= 0.20 and exec ratio 0.90 >= 0.60 → True., Skip ratio 0.30 > 0.20 → False., Exec ratio 0.50 < 0.60 → False., Missing ``raw_signal_count`` → treated as 0 → False., ``raw_signal_count=0`` → False. (+3 more)

### Community 75 - "TestWriteStrategyEvaluationTable"
Cohesion: 0.21
Nodes (9): _make_eval_rule_set(), _make_metrics_by_split(), _make_trade_logs_by_split(), Create a minimal rule set for evaluation table tests., Create a metrics_by_split dict with all three splits populated., Create a trade_logs_by_split dict with all three splits populated., Sharpe ratio should equal mean(r)/std(r, ddof=1) for a known log., win_rate, mdd_pct etc. should come from metrics_by_split. (+1 more)

### Community 76 - "test_gpu_engine.py"
Cohesion: 0.12
Nodes (14): _build_data_matrix(), _discretize_series(), DataFrame, Series, Discretize a feature series using evaluator_v5's fixed fuzzy bins. Exported…, Build an (N, K) integer matrix of discretized feature values., Raise RuntimeError if JAX failed to import at module level., _require_jax() (+6 more)

### Community 77 - "TestRunLogHandlerLifecycle"
Cohesion: 0.12
Nodes (15): MonkeyPatch, Phase 2 reproducibility default seed., PHASE2_SEED is drawn once at import via get_seed(); set GLOBAL_SEED=42 to fix…, Requirements 1.1, 1.4, 1.5, 1.6, 1.7 — run.log FileHandler is attached, writes…, Count FileHandlers on the root logger pointing to *path*., Patch every phase method on Pipeline_Orchestrator to be a no-op., run.log must exist after run() and contain both separator lines., Root logger must have no extra FileHandlers pointing to run.log after run(). (+7 more)

### Community 78 - "test_output_writer_properties.py"
Cohesion: 0.15
Nodes (25): parse_symbol_condition(), Parse optional symbol filters. Supported formats: "symbol is 1" "symbol IS 1"…, all_zero_rule_st(), _is_valid_exported_condition(), oversized_rule_set_st(), composite, DrawFn, given (+17 more)

### Community 79 - "test_feature_selector_properties.py"
Cohesion: 0.15
Nodes (23): dataset_with_high_dispersion_feature(), dataset_with_label_and_meta_columns(), dataset_with_sufficient_dispersion_feature(), _make_label_block(), composite, DataFrame, DrawFn, given (+15 more)

### Community 80 - ".run"
Cohesion: 0.07
Nodes (31): _archive_feature_signature(), _condition_feature_names(), _filter_compatible_previous_pool(), _filter_pool_by_admission(), _merge_archive_entries(), _monthly_admission_source_df(), Any, Return the ordered feature signature used to validate archive reuse. (+23 more)

### Community 81 - "_symbol_specialized_variants"
Cohesion: 0.11
Nodes (27): _attach_source_symbol_filters(), _ensure_symbol_filtered_rule(), _has_symbol_condition(), _is_recency_good(), _is_symbol_condition(), Add deterministic single-condition RB candidates. Evolution is deliberately…, Island/cluster symbols carried on Phase 2 pool entries., Feature ANDs + island symbol ORs (engine ORs multiple ``symbol is X``). (+19 more)

### Community 82 - "TestPlotPerRuleBreakdown"
Cohesion: 0.16
Nodes (11): _make_full_trade_log(), _make_rule_set(), _make_split_logs(), Create a trade log with all columns needed for sharpe computation., Create a minimal rule set with *n* rules., Create a trade log with Rule_Index, Net_PnL, Equity_After columns., Create trade_logs_by_split dict with all three splits populated., One split None, others valid — should not raise. (+3 more)

### Community 83 - "_build_entries_from_rule_set"
Cohesion: 0.19
Nodes (6): _build_entries_from_rule_set(), Priority-based rule assignment: first matching rule wins per row. Mirrors…, Row matching both rules should be assigned to rule 1 only., Rows not matching rule 1 should be assigned to rule 2., v5: earlier JSON rule wins over lower dataset row index at same time., TestBuildEntriesFromRuleSet

### Community 84 - "Feature_Detector"
Cohesion: 0.13
Nodes (12): detect_all_modes(), Feature_Detector, DataFrame, Series, detector.py — Feature_Detector Classifies each feature column into exactly one…, Classify feature columns by their discretization type., Classify a single feature series into one of six modes. Detection order…, Classify every column in *feature_cols* and return a mapping. Parameters… (+4 more)

### Community 85 - "_build_target"
Cohesion: 0.17
Nodes (12): _build_target(), Build a direction-specific target signal. Default…, Encoding-aware win class: 2 in asymmetric mode, 1 in legacy mode., Encoding-aware loss class: 0 in either mode., Long: max >= entry*(1+TP/100), min > entry*(1-SL/100) → success., Long: both hit but max_before_min==0 → SL first → failure., Long: both hit but max_before_min==1 → TP first → success., Long: neither TP nor SL hit → failure (or neutral in asymmetric mode). (+4 more)

### Community 87 - "test_gpu_engine_properties.py"
Cohesion: 0.16
Nodes (17): _assert_parity(), _make_engines(), _make_parity_df(), parity_scenario_strategy(), composite, DataFrame, DrawFn, given (+9 more)

### Community 88 - "TestGPUCPUNumericalParity"
Cohesion: 0.10
Nodes (15): ndarray, simulate_rule_set (delegated to CPU) is exactly identical to CPU engine., GPU backtest engine must handle padded chunking correctly., CPU backtest engine must support simulate_rule_batch with correct metrics., Compare GPU and CPU engine for 10 random chromosomes (Task 2 acceptance…, Build a multi-symbol DataFrame with mixed TP/SL/time-exit outcomes., Generate random chromosomes for binary + signed features., GPU engine results must match CPU engine within specified tolerances for 10… (+7 more)

### Community 89 - "build_complete_higher_bars"
Cohesion: 0.13
Nodes (29): DatetimeIndex, align_htf_features_causal(), _as_utc_datetime(), build_complete_higher_bars(), _compute_atr(), _compute_kama(), _compute_rsi(), compute_timeframe_features() (+21 more)

### Community 90 - "run_phase2_evolution_epoch"
Cohesion: 0.14
Nodes (15): StageLabel, Evolve one island epoch and return updated resumable state., run_phase2_evolution_epoch(), _FakeEngine, AC: resumed island epoch with reset_plateau=True clears restart counters., Global/non-island mode: reset_plateau=False preserves counters. Uses…, Task 2: Verify refresh_objectives_on_resume resets stale objectives on resumed…, Create a state with non-inf objectives and non-empty metrics_cache. (+7 more)

### Community 91 - "set_purged_wf_reference_rows"
Cohesion: 0.18
Nodes (7): Store full train_new.csv row count after loader prep (split time)., set_purged_wf_reference_rows(), holdout_mode(), purged_mode(), fixture, Tests for purged-WF trade-floor scaling helpers., TestScaleTradeFloor

### Community 92 - "test_phase2_window_rotation.py"
Cohesion: 0.07
Nodes (31): _largest_safe_range(), Return (start, end) of the largest contiguous bar range not in *forbidden*. The…, Cap *total_rows* so the per-symbol request fits within the safe range. When…, _resolve_sample_total_rows(), _make_multi_sym_df(), DataFrame, fixture, Tests for per-epoch train-window rotation (task-1). (+23 more)

### Community 93 - "Encoder"
Cohesion: 0.10
Nodes (20): ConfigurationError, decode_chromosome(), encode_condition(), Encoder, get_dont_care(), Exception, ndarray, encoder.py — Encoder Maps gene integer values to fuzzy value names, formats… (+12 more)

### Community 94 - "compute_phase2_objectives_from_metrics"
Cohesion: 0.04
Nodes (35): compute_phase2_objectives_from_metrics(), Penalty for weak cross-symbol robustness on one split., Build Phase 2 minimisation objectives from precomputed train/val metrics.…, _symbol_robustness_penalty(), True when val-derived feasibility penalties belong in NSGA-III fitness., _val_terms_in_fitness(), f3 uses robust return = min(train_return, val_return) when…, Blind-spot regression: overfit_gap_penalty must fire when val_ret <= 0. (+27 more)

### Community 95 - "test_crash_fix_properties.py"
Cohesion: 0.10
Nodes (26): _append_xla_flag(), configure_jax_env(), _cuda_package_root(), Path, JAX/XLA runtime configuration — call before the first ``import jax``., Find a pip-installed CUDA toolkit root, if one is available., Append an XLA flag once, preserving explicit user configuration., Configure JAX/XLA runtime for predictable desktop-friendly GPU usage. -… (+18 more)

### Community 96 - "mtf/__init__.py"
Cohesion: 0.11
Nodes (42): compute_archive_hash(), compute_rule_hash(), get_default_archive_path(), load_mtf_archive_payload(), load_mtf_rule_archive(), normalize_timeframe(), Any, Path (+34 more)

### Community 97 - "downcast_numeric_df"
Cohesion: 0.23
Nodes (14): downcast_numeric_df(), prune_train_columns(), DataFrame, df_slim.py — Reduce DataFrame memory for backtest engines. Keeps only columns…, Downcast label and feature columns to float32; symbol to category (inplace)., Return a copy with only meta, label, and requested feature columns. Parameters…, Drop unused feature columns from the full training split after Phase 1., slim_backtest_df() (+6 more)

### Community 98 - ".load_strategies"
Cohesion: 0.18
Nodes (6): Load long.json and short.json via Output_Writer.load_and_validate(). Parameters…, Return only non-empty RB strategies accepted for deployment., Write a valid rule set JSON to path., Override module-level path dicts and return originals (for standalone tests)., TestLoadStrategies, _write_rule_set()

### Community 99 - "gate_positive_good"
Cohesion: 0.28
Nodes (16): _finite_number(), gate_positive_good(), _metric_int(), positive_good_reject_reasons(), PositiveGoodThresholds, Shared metric gates used by Phase 2 admission and RB Governor. The gate is…, Minimum evidence required on train and validation splits., Return stable, machine-readable reasons for a gate rejection. (+8 more)

### Community 100 - ".get_dont_care"
Cohesion: 0.17
Nodes (7): See module-level :func:`get_dont_care`., **Property 9: Don't-Care Sentinel Correctness — encode_condition raises**…, **Property 9: Don't-Care Sentinel Correctness — all-dont_care → empty output**…, test_property_9b_encode_condition_raises_for_dont_care(), test_property_9f_all_dont_care_chromosome_returns_empty(), Static methods should be callable on the class itself., TestGetDontCare

### Community 101 - "phase5_oos.py"
Cohesion: 0.10
Nodes (19): _ensure_labels(), DataFrame, data/loader.py — Data_Loader Stateless CSV loading with full preparation…, Keep supplied labels or derive all labels from raw OHLCV columns. The original…, Load a CSV dataset with full preparation pipeline: 1. Read CSV with comma…, Validate the trend-context contract on an enriched frame if present. Fails…, validate_context_columns(), phase5_oos.py — OOS_Evaluator (Phase 5) Final out-of-sample diagnostics on the… (+11 more)

### Community 102 - "TestRemoveRedundantFeatures"
Cohesion: 0.25
Nodes (4): Two features with corr > 0.95 → keep higher-scored one., Two uncorrelated features → both kept., Features in different modes are not compared for redundancy., TestRemoveRedundantFeatures

### Community 104 - "hypothesis_config.py"
Cohesion: 0.15
Nodes (15): Shared Hypothesis settings for property tests. Set PYTEST_LOW_MEMORY=1 to scale…, _make_timestamps(), composite, DataFrame, DrawFn, given, Timestamp, Property-based tests for gpu_fuzzy_trader.data.splitter.Data_Splitter… (+7 more)

### Community 105 - "TestPlotDistributionAndEquity"
Cohesion: 0.16
Nodes (6): _make_dist_logs_by_split(), _make_dist_trade_log(), Create a trade log with all columns needed for distribution/equity plots., Create trade_logs_by_split dict with all three splits populated., A trade log with a single trade should not raise., TestPlotDistributionAndEquity

### Community 106 - "baselines.py"
Cohesion: 0.28
Nodes (15): _compact(), _drop_one_condition_variants(), _equal_weight(), _evaluate(), evaluate_baselines(), _feature_shuffle(), _fixed_exit(), Any (+7 more)

### Community 107 - "prop_settings"
Cohesion: 0.24
Nodes (16): HealthCheck, prop_settings(), Hypothesis settings with optional low-memory example scaling., given, Property-based tests for gpu_fuzzy_trader.reporting.reporter.Reporter This file…, **Validates: Requirements 6.4, 6.5, 6.6** For any valid inputs and any…, **Validates: Requirements 2.2, 2.3, 2.4** For any rule_set of length N with…, test_property_1_file_creation_round_trip() (+8 more)

### Community 108 - "_jax_compute_trade_outcomes"
Cohesion: 0.24
Nodes (5): _jax_compute_trade_outcomes(), JAX-jitted vectorized trade outcome computation., Vectorized trade outcome computation for all rows. Mirrors…, Multiple rows processed simultaneously., TestComputeTradeOutcomes

### Community 109 - "_derive_epoch_seed"
Cohesion: 0.16
Nodes (10): _derive_epoch_seed(), Return a deterministic per-epoch seed derived from base_seed + epoch., Re-sample training data with a per-epoch rotated window. Each epoch gets a…, An unknown PHASE2_PER_EPOCH_WINDOW_SEED_MODE raises ValueError., Deterministic seed derivation for per-epoch windows., Same (base_seed, epoch_idx) produces the same seed., Different epoch indices produce different seeds., None base seed returns None. (+2 more)

### Community 110 - "OOS_Evaluator"
Cohesion: 0.12
Nodes (11): OOS_Evaluator, Return an explicit, non-success result for a failed split., Save a split report, marking consumed test data as diagnostic-only., Out-of-sample evaluator for Phase 5. Loads the final long/short strategies,…, Run out-of-sample evaluation. Parameters ---------- allowed_directions :…, Load selected features for a direction when available., Remove only known Phase 5 artifacts from the active report root., _NumpyJSONEncoder (+3 more)

### Community 111 - "research_integrity.py"
Cohesion: 0.12
Nodes (26): _canonical_json(), dataset_manifest(), ExperimentLedger, forward_acceptance_lock_path(), Any, Path, PathLike, Research-integrity utilities shared by the pipeline and Phase 5. The consumed… (+18 more)

### Community 112 - "_validate_pool_schema"
Cohesion: 0.36
Nodes (3): Validate the structure of a loaded pool JSON. Raises ValueError if the schema…, _validate_pool_schema(), TestValidatePoolSchema

### Community 113 - "_derive_val_sample_seed"
Cohesion: 0.17
Nodes (10): _derive_val_sample_seed(), Derive a deterministic validation sample seed from the training seed. This…, AC: Train and validation sampling use distinct RNG seeds by default., _derive_val_sample_seed returns a value different from the input., Same train seed always produces same val seed., Result is in [0, 2**31) so it is a valid random seed., Rule_Pool_Generator stores distinct _sample_seed and _val_sample_seed., When seed=None, val seed is derived from PHASE2_SEED. (+2 more)

### Community 114 - "TestHammingThresholdAutoScale"
Cohesion: 0.15
Nodes (10): Verify the max(3, k_active // 5) formula. The formula is applied in…, Replicate the auto-scaling formula., k_active=0 → threshold = max(3, 0//5) = 3., k_active=5 → threshold = max(3, 5//5=1) = 3., k_active=15 → threshold = max(3, 15//5=3) = 3., k_active=20 → threshold = max(3, 20//5=4) = 4., k_active=50 → threshold = max(3, 50//5=10) = 10., k_active=100 → threshold = max(3, 100//5=20) = 20. (+2 more)

### Community 115 - ".decode_chromosome"
Cohesion: 0.26
Nodes (4): See module-level :func:`decode_chromosome`., All 10 signed values decode correctly via decode_chromosome., Chromosome with numpy int32/int64 values should work., TestDecodeChromosome

### Community 116 - "ValidationError"
Cohesion: 0.09
Nodes (27): context_permission_column(), context_trigger_column(), Return the direction-specific permission column name., Return the direction-specific LWC pullback-reversal trigger column., _condition_feature(), _context_feature_direction(), Exception, Path (+19 more)

### Community 117 - "TestComputeRuleSignals"
Cohesion: 0.18
Nodes (6): All rows match when chromosome equals data_matrix values., No rows match when chromosome differs from data_matrix., Only rows where all active conditions match., Columns where chromosome == dont_care are ignored., All dont_care chromosome matches every row., TestComputeRuleSignals

### Community 118 - "TestParetoCollapseWarningGate"
Cohesion: 0.13
Nodes (11): _FakeEngine, Unit tests for Pareto-collapse warning gate (audit finding #13). AC: The…, AC 4: The default value of the config flag is 5., AC 5: min_pareto_size=0 → warning fires even with small pareto (size=4)., The log message includes 'pareto_size=N' suffix., Fake engine that returns metrics producing a tradeoff between f1 (-sortino) and…, AC 1–5: warning gated on len(pareto_indices) >= config threshold., Run 2-gen evolution and return count of 'Pareto collapse risk' warnings. (+3 more)

### Community 119 - "test_rb_governor_tail_holdout.py"
Cohesion: 0.07
Nodes (29): _make_walk_forward_fold_engines(), Split val_selection into n_splits chronological folds + optional tail holdout.…, _make_synthetic_df(), DataFrame, Verify _make_walk_forward_fold_engines splits data correctly., 2 splits + tail holdout → 2 fold engines + 1 tail engine., tail_holdout_frac=0 → tail engine is None., Each symbol's data is divided into contiguous chunks across folds. (+21 more)

### Community 120 - "_compute_stability"
Cohesion: 0.17
Nodes (9): _compute_stability(), Compute cross-symbol stability score. stability = 1 - (std / mean) If mean is 0…, All symbols have same score → std=0 → stability=1., Mean is 0 → stability=0., Single symbol with positive score → stability=1., Single symbol with zero score → stability=0., Very high variance → stability could be negative → clipped to 0., Stability should always be in [0, 1]. (+1 more)

### Community 121 - "passes_pool_admission_gate"
Cohesion: 0.08
Nodes (21): _feasibility_gate_failures(), passes_pool_admission_gate(), Return per-gate failure flags for evolution-time feasibility diagnostics. Uses…, Hard gate for Phase 2 pool/archive on merged holdout metrics. When…, fixture, MonkeyPatch, Tests for _feasibility_gate_failures — per-gate breakdown., A rule that should pass all 9 gates. (+13 more)

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

### Community 126 - "cv_folds_only"
Cohesion: 0.17
Nodes (9): aggregate_fold_metrics(), cv_folds_only(), FoldMetricsSummary, CV folds excluding the primary holdout., Summarize metrics across folds (worst-case emphasis)., Collapse per-fold metrics into one dict for objectives / gates. ``worst``: min…, summarize_fold_metrics(), CV valid blocks differ by at most 1 bar when remaining % n != 0. (+1 more)

### Community 127 - "resolve_evolution_floors"
Cohesion: 0.18
Nodes (8): EvolutionFloors, Resolved evolution-time floors (pool admission gates remain strict)., Return stage-aware fitness floors; defaults to global strict knobs. When both…, resolve_evolution_floors(), Stage A soft floors must survive optional floor overrides., _pool_admission_floors returns the ADMISSION floor (1.15), not the EVOLUTION…, TestPoolAdmissionScaledFloors, TestResolveEvolutionFloorsWithOverrides

### Community 128 - "stratification_scenario_strategy"
Cohesion: 0.24
Nodes (15): dataset_with_features_strategy(), metrics_strategy(), composite, DataFrame, DrawFn, Generate a trade log DataFrame with 2–50 rows. Parameters ---------- n_rules:…, Generate a metrics dict with reasonable float values. Keys: win_rate,…, Generate a dataset DataFrame with 10–100 rows. Parameters ----------… (+7 more)

### Community 129 - "TestSpearmanSignConsistency"
Cohesion: 0.17
Nodes (10): DataFrame, ndarray, AC3: Train all positive, tiny |val_rho| < min_abs_corr → feature still kept., AC4: val_df=None → pre-task-8 behavior (mixed train signs still blacklisted)., AC5: val_df without label_close_288 column → val check skipped (no crash)., Tests for _check_spearman_sign_consistency, including the val_df check., Build a minimal DataFrame without 'symbol' to avoid symbol-based folding., AC1: Train all positive, val negative → feature blacklisted. (+2 more)

### Community 130 - "compute_labels"
Cohesion: 0.20
Nodes (10): compute_labels(), DataFrame, Forward-window label computation for OHLCV bars. Horizon is ``TAIL_DROP_ROWS``…, Compute the 5 label columns per symbol. - label_open_next: open[t+1] -…, DataFrame, ndarray, Unit tests for forward-window label semantics…, Regression: old backward rolling must not match corrected forward labels. (+2 more)

### Community 131 - "log_memory_rss"
Cohesion: 0.22
Nodes (10): log_memory_rss(), Optional process memory logging (set LOG_MEMORY=1)., Log current process RSS in MiB when LOG_MEMORY=1., Force GC and clear JAX compilation caches between Phase 2 directions. Releases:…, release_phase2_resources(), Release GPU engines and slim in-memory data between epochs., Tests for optional memory logging helpers., test_log_memory_rss_noop_without_env() (+2 more)

### Community 132 - "._load_datasets_by_split"
Cohesion: 0.23
Nodes (6): DataFrame, Prepare test data using Data_Loader.load_dataset(). Applies the same…, Load prepared train, validation, and test datasets., Load and validate a strictly newer, untouched forward period., prepare_test_data should return a DataFrame., TestPrepareTestData

### Community 133 - "rolling_cv.py"
Cohesion: 0.33
Nodes (12): _bar_index_col(), _build_fold_from_ranges(), build_purged_walk_forward_folds(), _purge_train(), DataFrame, Purged expanding walk-forward folds for train_new.csv. Per-symbol chronological…, Drop train rows whose label horizon could overlap the valid block., Build purged expanding walk-forward folds on ``train_new.csv``. Returns… (+4 more)

### Community 134 - "_should_post_restart_early_stop_phase2"
Cohesion: 0.38
Nodes (5): Break the run when a plateau restart yields no improvement., _should_post_restart_early_stop_phase2(), Unit tests for post-restart no-improvement early stop (Phase 2 runtime)., test_global_disabled_no_stop(), test_global_uses_global_knobs()

### Community 135 - "context_contract_digest"
Cohesion: 0.23
Nodes (10): context_contract_digest(), Return a stable hash of the static contract and fitted enrichment., feature_conditions_only(), phase2_rule_id(), Any, Stable provenance identifiers for Phase 2 feature rules. The RB Governor may…, Return normalized non-symbol, non-context conditions in order. Symbol scope and…, Hash the immutable Phase 2 feature logic and its discovery scope. (+2 more)

### Community 136 - "_apply_colab_gpu_defaults"
Cohesion: 0.33
Nodes (6): _apply_colab_gpu_defaults(), is_colab_runtime(), True when running on Google Colab (/content runtime)., Colab T4 optimization for Phase 2 runs., MonkeyPatch, test_colab_defaults_apply_when_content_exists()

### Community 137 - "_pareto_sortino_stats"
Cohesion: 0.47
Nodes (3): _pareto_sortino_stats(), Aggregate raw Sortino and return health over the current Pareto front., TestParetoSortinoStats

### Community 138 - "get_gpu_backtest_engine_class"
Cohesion: 0.33
Nodes (6): get_gpu_backtest_engine_class(), jax_gpu_backtest_available(), Any, Detect whether JAX / GPUBacktestEngine can be loaded on this host. JAX can fail…, Return ``GPUBacktestEngine`` if ``gpu_engine`` imports cleanly; else ``None``., True when ``get_gpu_backtest_engine_class()`` would succeed.

### Community 139 - "CPUBacktestEngine"
Cohesion: 0.07
Nodes (76): CPUBacktestEngine, CPU backtest engine that exactly mirrors evaluator_v5.ipynb's…, expectancy_lcb_pct(), Return a conservative lower bound for per-trade net expectancy. Exact CPU…, _available_symbols(), _balanced_phase2_shortlist(), _cost_stress_gate(), _eval_cv_fold_returns() (+68 more)

### Community 140 - "composite"
Cohesion: 0.23
Nodes (12): equity_tracking_scenario(), fee_deduction_scenario(), multi_symbol_scenario(), overlapping_rule_set_strategy(), composite, DrawFn, Generate a rule set of 1–4 rules that may overlap in their conditions, plus a…, Generate a random price scenario for a single trade. Returns a dict with:… (+4 more)

### Community 141 - "test_feature_selector.py"
Cohesion: 0.09
Nodes (18): _align_feature_array(), _mutual_info_discrete_mask(), Return the set of feature names that PASS both stationarity checks. A feature…, Cap long/short feature overlap and backfill each direction to top_k features., Per-column ``discrete_features`` flags for ``mutual_info_classif``. Only…, Slice precomputed matrix columns when *selected_cols* is a subset., Remove features where more than `threshold` fraction of values are identical.…, _reduce_overlap() (+10 more)

### Community 142 - "phase2_support.py"
Cohesion: 0.06
Nodes (42): effective_min_trade_pool_floor(), effective_pool_min_val_trades(), IslandHyperparams, Optional Phase 2 floor overrides (tests and diagnostics)., _pool_entry_passes_admission(), Check stored train/val metrics on a pool JSON entry., compute_robust_score(), deployability_rank_score() (+34 more)

### Community 143 - "run_pipeline.py"
Cohesion: 0.08
Nodes (46): _align_upstream_scores(), _build_layer_frame(), _candidate_key(), canonicalize_oof_scores(), _directional_pareto_front(), discover_directional_layer(), _eligible_numeric_features(), _fit_fold_candidates() (+38 more)

### Community 144 - "TestRefreshObjectivesOnResumeGate"
Cohesion: 0.23
Nodes (7): Task-10: gate cache refresh on PHASE2_PER_EPOCH_WINDOW_ROTATION. Verifies the…, Evaluate the gate expression as it appears in the source., PHASE2_PER_EPOCH_WINDOW_ROTATION=True (default): refresh = not first_epoch →…, PHASE2_PER_EPOCH_WINDOW_ROTATION=True (default): refresh = not first_epoch →…, PHASE2_PER_EPOCH_WINDOW_ROTATION=False (legacy): refresh is False even on…, PHASE2_PER_EPOCH_WINDOW_ROTATION=False (legacy): refresh is False on first…, TestRefreshObjectivesOnResumeGate

### Community 145 - "conftest.py"
Cohesion: 0.22
Nodes (10): Config, FixtureRequest, Item, _close_matplotlib_figures(), _low_memory_cleanup(), fixture, pytest_collection_modifyitems(), pytest_configure() (+2 more)

### Community 147 - ".skip_if_valid"
Cohesion: 0.31
Nodes (3): Check if output files exist and are valid. Returns ------- dict[str,…, fixture, TestSkipIfValid

### Community 148 - "_validate_schema"
Cohesion: 0.17
Nodes (5): Validate the structure of a loaded feature selection JSON. Raises ValueError if…, Load and validate a feature selection JSON file. Parameters ---------- path :…, _validate_schema(), TestLoadAndValidate, TestValidateSchema

### Community 149 - "test_directional_evaluator.py"
Cohesion: 0.12
Nodes (31): classify_directional_labels(), compute_conditional_mwc_labels(), compute_forward_movement_labels(), evaluate_conditional_directional_rule(), evaluate_directional_rule(), fit_directional_threshold(), ndarray, Series (+23 more)

### Community 150 - "Global Constraints"
Cohesion: 0.20
Nodes (9): Global Constraints, Hierarchical Multi-Timeframe Rule Discovery Implementation Plan, Task 1: Causal Multi-Timeframe Data Engine, Task 2: Directional & Conditional Evaluators & Rule Search Profiles, Task 3: Master Temporal Folds, Purged Embargo & OOF Cross-Fitting, Task 4: Decoupled Ensemble Score (Direction & Strength) & Rule Archives, Task 5: MTF Composer, Asymmetric Soft Veto, and Trade Retention Guard, Task 6: Pipeline Integration (`run_pipeline.py`, `config.py`, `loader.py`, `cpu_engine.py`, `rb_governor.py`, `phase5_oos.py`) (+1 more)

### Community 151 - "test_phase2_rule_pool.py"
Cohesion: 0.06
Nodes (29): _crowding_distance(), _diversity_penalty_blended(), _evaluate_chromosome(), _hamming_distance(), _phenotype_bucket_key(), Evaluate a single chromosome and return (objectives, metrics). objectives =…, Compute crowding distance for solutions in *front*. Parameters ----------…, Hamming distance between two chromosomes (active pairs when sparse). (+21 more)

### Community 152 - "BFS and DFS Graph Traversal"
Cohesion: 0.20
Nodes (10): CLAUDE.md Graphify Integration, Constrained Query Expansion, BFS and DFS Graph Traversal, Graph Work Memory, Existing Graph Fast Path, CLAUDE.md Graphify Integration, Constrained Query Expansion, BFS and DFS Graph Traversal (+2 more)

### Community 153 - "TestPlotPhase2Metrics"
Cohesion: 0.23
Nodes (4): _make_history(), History entries with missing keys should not raise., Create a minimal Phase 2 history list., TestPlotPhase2Metrics

### Community 154 - "TestPoolAdmissionOverfitRatioGate"
Cohesion: 0.24
Nodes (7): MonkeyPatch, Tests for the hard overfit ratio gate in pool admission. The ratio gate rejects…, train=15%/val=4% (3.75× ratio, gap=11pp < 16pp) is REJECTED by the ratio gate…, train=15%/val=10% (1.5× ratio) is ADMITTED., train/val ≈2.5× is ADMITTED when under OVERFIT_RATIO_FLOOR and gap gate., With PHASE2_OVERFIT_RATIO_FLOOR=0.0, the ratio gate is disabled and the high-…, TestPoolAdmissionOverfitRatioGate

### Community 155 - "test_rb_fail_closed.py"
Cohesion: 0.33
Nodes (10): _dummy_df(), _pool_rules(), DataFrame, Path, RB Governor fail-closed and stale-output regression tests., test_empty_phase2_pool_writes_fail_closed_output_with_reason(), test_fail_closed_output_overwrites_stale_strategy(), test_no_positive_good_candidates_fail_closed_and_do_not_call_fallback() (+2 more)

### Community 156 - ".skip_if_valid"
Cohesion: 0.14
Nodes (12): _pool_path_key(), Return the sidecar that binds a reusable pool to its inputs., Hash an artifact without loading it all into RAM., Load existing pool if valid, return None if missing., Atomically bind the current pool bytes to a Phase 2 input identity., Remove a stale direction cache before a fresh Phase 2 run., Return a schema-valid pool proven to match this run's inputs. Bare historical…, _resolve_history_path() (+4 more)

### Community 157 - ".simulate_rule_set_batch"
Cohesion: 0.20
Nodes (7): _batch_eval_rule_set_pickled(), _jax_runtime_loaded(), Simulate using a precomputed rule-evaluation mask cache. The mandatory…, Drop cached entries on rows the fixed context mask forbids. ``idx`` validity is…, Evaluate multiple rule sets without forking an active JAX runtime., Top-level worker for ProcessPoolExecutor (must be picklable)., Return whether forking would inherit an already multithreaded JAX runtime.

### Community 158 - "TestGlobalMetricsCacheClearing"
Cohesion: 0.27
Nodes (6): When seeded_keys is empty, nothing is removed., When all keys are seeded, cache becomes empty., Verify that only seeded keys are removed from the global cache., Create a deterministic chromosome key., Cache entries matching seeded_keys are removed; non-matching survive., TestGlobalMetricsCacheClearing

### Community 159 - "test_evaluator_health.py"
Cohesion: 0.17
Nodes (7): Unit tests for ``evaluator_health_penalty`` and ``execution_ok`` (Task 4).…, Verify that evaluator health penalty is subtracted from combo score., When evaluator_health_penalty returns > 0, the combo score is lower., When evaluator health is fine, no penalty is applied., Both public functions are importable from the module., TestHealthPenaltyWiredIntoRB, TestModuleImportable

### Community 160 - "_nsga3_environmental_selection"
Cohesion: 0.22
Nodes (8): _deduplicate_selection_indices(), _nsga3_environmental_selection(), NSGA-III environmental selection (rank + niche on last front)., Replace duplicate genotypes in survivors with unused unique merge rows., _warn_evox_unavailable(), skipif, Unit tests for NSGA-III environmental selection., TestNsga3Selection

### Community 161 - "_metrics_snapshot"
Cohesion: 0.20
Nodes (10): _load_global_metrics_cache(), _metrics_snapshot(), Shallow copy for evolution caches (metrics are flat numeric dicts)., Store processed metrics (and optional val sidecar) in the run-wide cache., Bound run-wide eval cache size to limit RAM growth across long runs. Uses FIFO…, Restore metrics and optional val_metrics from a cache entry., Drop bulky resumable state that is already persisted elsewhere., _store_global_metrics_cache() (+2 more)

### Community 162 - "test_phase5_oos_properties.py"
Cohesion: 0.18
Nodes (12): _make_timestamps(), composite, DataFrame, DrawFn, given, Timestamp, Property-based tests for gpu_fuzzy_trader.phases.phase5_oos.OOS_Evaluator…, **Property 27: Test Data Preparation Consistency** **Validates: Requirements… (+4 more)

### Community 163 - "trade_support_penalty"
Cohesion: 0.24
Nodes (7): compute_support_penalty_and_specialist(), Support penalty. Returns ------- penalty : float is_specialist : bool (always…, Support penalty from train metrics. Returns (penalty, False, -1)., Legacy graduated penalty., _static_support_penalty(), trade_support_penalty(), TestTradeSupportPenaltyStatic

### Community 164 - "TestSparsePositiveMode"
Cohesion: 0.22
Nodes (5): All non-negative, zero_ratio > 0.3 → sparse_positive., All zeros: zero_ratio = 1.0 > 0.3, min = 0 → sparse_positive., zero_ratio just above 0.3 → sparse_positive., zero_ratio must be computed on the full series, not just non-NaN., TestSparsePositiveMode

### Community 165 - "evaluator_health.py"
Cohesion: 0.40
Nodes (5): evaluator_health.py — Pure functions for evaluator-failure-mode awareness.…, Read a numeric metric, returning *default* for missing / None / NaN / Inf., Read an integer metric safely., _safe_float(), _safe_int()

### Community 166 - "mandatory_context_conditions"
Cohesion: 0.12
Nodes (22): mandatory_context_conditions(), Return the fixed, mandatory context conditions for *direction*., _assert_capital_budget(), _assert_mandatory_context(), _enforce_capital_budget(), Path, Persist an explicit empty strategy and diagnostic report., Normalize rule capital_pct so sum <= RB_MAX_TOTAL_CAPITAL. (+14 more)

### Community 167 - "TestZeroRatioBoundary"
Cohesion: 0.22
Nodes (5): Exactly 30% zeros with non-negative values → positive (not sparse_positive)., 31% zeros with non-negative values → sparse_positive., Exactly 30% zeros with negative values → signed (not sparse_signed)., Just above 30% zeros with negative values → sparse_signed., TestZeroRatioBoundary

### Community 168 - "_build_rule_signal_mask"
Cohesion: 0.32
Nodes (7): _build_rule_signal_mask(), Cached wrapper around :func:`_compute_rule_signal_mask`., _backtest_df(), DataFrame, Regression tests for evaluator-facing Phase 2 chromosome semantics., Search fitness must use the same fuzzy class as RB/Phase 5 evaluation., test_batch_chromosome_signals_match_decoded_rule_conditions()

### Community 169 - "main"
Cohesion: 0.25
Nodes (7): __main__.py — Entry point for `python -m gpu_fuzzy_trader.run_pipeline` Allows…, main(), _phase5_test_metrics(), _print_run_summary(), Return test-split metrics (supports nested Phase 5 result shape)., Print a concise CLI summary for a full or single-phase run., Run the full pipeline from the command line.

### Community 170 - "trade_support_penalty"
Cohesion: 0.38
Nodes (4): Backward-compatible wrapper returning penalty only., trade_support_penalty(), Between the hard-reject floor and the soft threshold the penalty is graduated., TestTradeSupportPenalty

### Community 171 - "TestPrecomputeReleaseIndices"
Cohesion: 0.29
Nodes (4): Release index should point to the row where bar_index + 288 is reached., Rows near the end should get release_index = len(df)., Each symbol's release indices are computed independently., TestPrecomputeReleaseIndices

### Community 172 - "TestHallOfFameTrim"
Cohesion: 0.25
Nodes (5): Verify that hall_of_fame is capped to PHASE2_HOF_EPOCH_CARRYOVER=10., HoF with >10 entries is trimmed to exactly 10 oldest entries., HoF with <=10 entries is left unchanged., Empty HoF does not crash during trim logic., TestHallOfFameTrim

### Community 173 - "_compute_rule_signal_mask"
Cohesion: 0.15
Nodes (13): conditions_cache_key(), get_or_build_rule_mask(), ndarray, condition_cache.py — Cached boolean masks for textual rule conditions. Avoids…, Stable hashable key for a list of condition strings., Return a boolean row mask for *conditions*, using *cache* when provided. The…, _compute_rule_signal_mask(), Build one boolean signal mask (evaluator_v5 parity). Feature conditions are… (+5 more)

### Community 174 - "TestNaNHandling"
Cohesion: 0.29
Nodes (4): All-NaN series: unique_vals = [], n_unique = 0 ≤ 2, set() ⊆ {0,1} → binary., NaN is not == 0, so it does not inflate zero_ratio., NaN should not push zero_ratio above threshold., TestNaNHandling

### Community 175 - "scoring/__init__.py"
Cohesion: 0.33
Nodes (5): profit_factor_term(), Scoring helpers shared across pipeline phases. Re-exports…, Return divided by max drawdown; higher is better. A small drawdown floor avoids…, Clamp a profit factor into ``[0, cap]``; non-finite → ``cap``., return_to_drawdown()

### Community 176 - "test_gpu_engine_import_does_not_crash_on_jax_failure"
Cohesion: 0.33
Nodes (6): parametrize, Tests for the lazy-JAX failure path in gpu_engine.py. These tests verify that…, Verify importing gpu_engine.py handles various JAX failure modes gracefully., Build a subprocess script that simulates JAX import failure., _subprocess_script(), test_gpu_engine_import_does_not_crash_on_jax_failure()

### Community 177 - "test_jax_compat.py"
Cohesion: 0.29
Nodes (5): parametrize, Tests for JAX / GPU engine availability detection., Package init must not crash when gpu_engine import fails., test_cpu_engine_import_without_jax(), test_get_gpu_backtest_engine_class_returns_none_on_import_error()

### Community 178 - "TestExecutionHealthInGate"
Cohesion: 0.25
Nodes (5): Verify that ``gate_positive_good`` calls ``execution_ok`` when flagged., When require_execution_health=True and execution_ok fails, gate returns False., When require_execution_health=True and execution_ok passes, gate still returns…, When require_execution_health=False, gate ignores raw_signal_count., TestExecutionHealthInGate

### Community 179 - "TestSparseSignedMode"
Cohesion: 0.33
Nodes (3): Has negative values, zero_ratio > 0.3 → sparse_signed., NaN does not count as zero; zero_ratio on full series., TestSparseSignedMode

### Community 180 - "effective_phase2_val_return_floor_pct"
Cohesion: 0.40
Nodes (4): effective_phase2_val_return_floor_pct(), Direction-aware Phase 2 validation return floor for fitness penalties., Regression tests for anti-overfit / exploration-retune config bundle., test_anti_overfit_config_bundle()

### Community 181 - "TestF3PathResolution"
Cohesion: 0.40
Nodes (4): parametrize, Parametrized tests for f3 path resolution (Task 5: audit finding #5). Verifies…, Verify the correct f3 formula runs for each (USE_TOTAL_RETURN_OBJ,…, TestF3PathResolution

### Community 182 - "_identity_value"
Cohesion: 0.50
Nodes (4): _identity_value(), _phase2_cv_structure(), Convert configuration values to a stable, JSON-safe identity form., Capture CV boundaries without duplicating full frame contents in RAM.

### Community 183 - "load_cv_folds_manifest"
Cohesion: 0.67
Nodes (3): load_cv_folds_manifest(), Any, Load manifest if present.

### Community 184 - "TestEvalCvFoldReturns"
Cohesion: 0.33
Nodes (4): Verify the helper handles None / empty fold_engines safely., fold_engines=None returns None without crashing., fold_engines=[] returns None without crashing., TestEvalCvFoldReturns

### Community 189 - "opencode.json"
Cohesion: 0.50
Nodes (3): plugin, $schema, .opencode/plugins/graphify.js

## Knowledge Gaps
- **58 isolated node(s):** `$schema`, `.opencode/plugins/graphify.js`, `Task 1: Causal Multi-Timeframe Data Engine`, `Task 2: Directional & Conditional Evaluators & Rule Search Profiles`, `Task 3: Master Temporal Folds, Purged Embargo & OOF Cross-Fitting` (+53 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `CPUBacktestEngine` connect `CPUBacktestEngine` to `barrier_column_names`, `test_cpu_engine.py`, `Pipeline_Orchestrator`, `_score_metrics`, `_make_df`, `_apply_monthly_admission_gate`, `run_pipeline.py`, `test_cpu_engine_properties.py`, `CandidateRecord`, `Rule_Pool_Generator`, `.simulate_rule_set_batch`, `_build_pool_from_archive`, `_build_rule_signal_mask`, `phase2_rule_pool.py`, `gpu_engine.py`, `TestPrecomputeReleaseIndices`, `trade_support_penalty`, `_compute_rule_signal_mask`, `test_mtf_pipeline_integration.py`, `test_jax_compat.py`, `TestEvalCvFoldReturns`, `ndarray`, `nested_walk_forward.py`, `test_phase2_use_gpu_flag.py`, `GPUBacktestEngine`, `_apply_dynamic_rule`, `cpu_engine.py`, `test_gpu_engine.py`, `_symbol_specialized_variants`, `_build_entries_from_rule_set`, `test_gpu_engine_properties.py`, `TestGPUCPUNumericalParity`, `phase5_oos.py`, `baselines.py`, `_jax_compute_trade_outcomes`, `OOS_Evaluator`, `ValidationError`, `TestComputeRuleSignals`, `test_rb_governor_tail_holdout.py`, `MonthlyWindowSummary`, `._engine`?**
  _High betweenness centrality (0.162) - this node is a cross-community bridge._
- **Why does `Reporter` connect `Reporter` to `phase5_oos.py`, `_build_pool_from_archive`, `TestEquityCurveDateAxis`, `phase2_rule_pool.py`, `TestPlotDistributionAndEquity`, `prop_settings`, `._ensure_dir`, `TestWriteStrategyEvaluationTable`, `OOS_Evaluator`, `.run`, `TestPlotPerRuleBreakdown`, `ValidationError`, `test_reporter.py`, `TestPlotPhase2Metrics`, `Rule_Pool_Generator`?**
  _High betweenness centrality (0.085) - this node is a cross-community bridge._
- **Why does `Rule_Pool_Generator` connect `Rule_Pool_Generator` to `log_memory_rss`, `Pipeline_Orchestrator`, `TestValLeakGate`, `TestRulePoolGeneratorRun`, `context_contract_digest`, `_pareto_sortino_stats`, `CPUBacktestEngine`, `run_pipeline.py`, `_apply_monthly_admission_gate`, `TestRefreshObjectivesOnResumeGate`, `_make_train_df`, `test_phase2_rule_pool.py`, `Reporter`, `TestPoolAdmissionOverfitRatioGate`, `.skip_if_valid`, `phase2_rule_pool.py`, `test_phase2_rule_pool_properties.py`, `test_crash_fix_and_run_logging.py`, `TestF3PathResolution`, `_dominates`, `_get_dont_cares`, `test_phase2_use_gpu_flag.py`, `test_plateau_state_leak.py`, `TestRunLogHandlerLifecycle`, `.run`, `test_phase2_window_rotation.py`, `compute_phase2_objectives_from_metrics`, `_derive_epoch_seed`, `OOS_Evaluator`, `_validate_pool_schema`, `_derive_val_sample_seed`?**
  _High betweenness centrality (0.077) - this node is a cross-community bridge._
- **Are the 15 inferred relationships involving `Reporter` (e.g. with `CvFoldValEvaluator` and `Rule_Pool_Generator`) actually correct?**
  _`Reporter` has 15 INFERRED edges - model-reasoned connections that need verification._
- **Are the 54 inferred relationships involving `CPUBacktestEngine` (e.g. with `GPUBacktestEngine` and `JointPortfolioEngine`) actually correct?**
  _`CPUBacktestEngine` has 54 INFERRED edges - model-reasoned connections that need verification._
- **Are the 58 inferred relationships involving `Rule_Pool_Generator` (e.g. with `CPUBacktestEngine` and `Reporter`) actually correct?**
  _`Rule_Pool_Generator` has 58 INFERRED edges - model-reasoned connections that need verification._
- **Are the 19 inferred relationships involving `Pipeline_Orchestrator` (e.g. with `CPUBacktestEngine` and `Data_Loader`) actually correct?**
  _`Pipeline_Orchestrator` has 19 INFERRED edges - model-reasoned connections that need verification._