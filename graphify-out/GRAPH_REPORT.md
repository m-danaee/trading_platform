# Graph Report - trading_platform  (2026-08-12)

## Corpus Check
- 186 files · ~257,821 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 4858 nodes · 11004 edges · 178 communities (170 shown, 8 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 557 edges (avg confidence: 0.56)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `c32b462c`
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
- phase2_rule_pool.py
- _should_plateau_early_stop_phase2
- TestArchiveSaveExceptionHandling
- TestOOSEvaluatorRun
- Pipeline_Orchestrator
- _apply_monthly_admission_gate
- Feature_Detector
- _make_train_df
- _split
- test_cpu_engine_properties.py
- detect_feature_mode
- selector.py
- Feature_Selector
- rolling_cv.py
- Reporter
- test_encoder_properties.py
- write_evaluator_clean
- CandidateRecord
- nested_walk_forward.py
- _preserve_deployable_elites
- maybe_log_generation
- RB Governor production path
- _select_diverse_subset
- test_rb_min_symbols.py
- Graphify Pipeline
- Output_Writer
- config.py
- TestEquityCurveDateAxis
- sample_df_for_phase2
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
- ._load_and_split_data
- prop_settings
- _compute_rule_signal_mask
- validate_config
- build_hybrid_symbol_clusters
- _init_population
- test_phase2_rule_pool.py
- detect_all_modes
- test_reporter.py
- GPUBacktestEngine
- TestGPUCPUNumericalParity
- cpu_engine.py
- test_phase2_use_gpu_flag.py
- _derive_val_sample_seed
- TestRulePoolGeneratorRun
- test_plateau_state_leak.py
- .prepare_test_data
- dashboard.py
- test_phase2_val_sim_interval.py
- test_data_loader_properties.py
- barrier_column_names
- _m
- _apply_dynamic_rule
- test_cpu_engine.py
- mandatory_context_conditions
- execution_ok
- TestWriteStrategyEvaluationTable
- test_gpu_engine.py
- TestRunLogHandlerLifecycle
- TestPrecomputeReleaseIndices
- test_feature_selector_properties.py
- .get_dont_care
- compute_labels
- TestPlotPerRuleBreakdown
- resolve_island_hyperparams
- test_phase2_island_scheduler.py
- _build_target
- filter_migrants_for_cluster
- test_gpu_engine_properties.py
- test_rb_fail_closed.py
- _derive_epoch_seed
- Data_Loader
- set_purged_wf_reference_rows
- downcast_numeric_df
- .encode_condition
- test_output_writer_properties.py
- ValidationError
- test_phase2_gpu_throughput.py
- .decode_chromosome
- .load_strategies
- gate_positive_good
- phase2_island_scheduler.py
- .save_archive
- TestRemoveRedundantFeatures
- _make_df
- test_crash_fix_and_run_logging.py
- TestPlotDistributionAndEquity
- baselines.py
- .run
- _jax_compute_trade_outcomes
- _should_post_restart_early_stop_phase2
- FakeEngine
- resolve_phase2_stage_params
- _validate_pool_schema
- TestSavePerSymbolCsv
- TestHammingThresholdAutoScale
- TestValLeakGate
- _build_entries_from_rule_set
- _jax_compute_rule_signals
- TestParetoCollapseWarningGate
- TestMakeWalkForwardFoldEngines
- _compute_stability
- TestPoolAdmissionOverfitRatioGate
- MonthlyWindowSummary
- ._engine
- test_rb_concentration_tail_fail_closed.py
- constrained_non_dominated_sort
- test_feature_selector.py
- TestEndToEndRotation
- stratification_scenario_strategy
- TestSpearmanSignConsistency
- run_pipeline.py
- log_memory_rss
- TestPlotPhase2Metrics
- _apply_colab_gpu_defaults
- splitter.py
- ConfigurationError
- OOS_Evaluator
- compute_phase2_objectives_from_metrics
- TestMakeFoldEnginesTailHoldout
- effective_min_profitable_symbols
- TestMigrationSeedFraction
- TestEvictClusterSignatures
- test_phase2_offspring_batch.py
- trade_support_penalty
- TestRefreshObjectivesOnResumeGate
- conftest.py
- test_phase2_batch_evaluator_parity.py
- .skip_if_valid
- _validate_schema
- ResearchProfile
- TestDeferredWarmup
- .skip_if_valid
- BFS and DFS Graph Traversal
- TestPlateauDiversityRestart
- test_nsga3_selection.py
- TestCausalPublicationTiming
- TestGlobalMetricsCacheClearing
- test_evaluator_health.py
- ndarray
- TestHallOfFameTrim
- phase2_support.py
- evaluator_health.py
- test_gpu_engine_import_does_not_crash_on_jax_failure
- test_jax_compat.py
- TestExecutionHealthInGate
- TestPerSymbolMetrics
- TestEvalCvFoldReturns
- opencode.json
- _MockEngine
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
- **Fail-closed research boundary** — readme_configuration_contract, readme_symbol_specialist_islands, readme_rb_governor, readme_holdout_acceptance_contract, readme_fail_closed_deployment [INFERRED 0.95]
- **Hardware-aware execution flow** — run_rtx4050_execution_policy, run_colab_t4_path, run_pipeline_orchestrator, readme_hybrid_execution_policy [INFERRED 0.95]

## Communities (178 total, 8 thin omitted)

### Community 0 - "evox_runner.py"
Cohesion: 0.04
Nodes (124): IslandHyperparams, Resolved Phase 2 knobs for cluster or orphan slices., _assign_eval_result(), _behavior_descriptor(), _binary_tournament_pick(), _build_diversity_reference(), _constraint_violations(), _count_deployable_preview() (+116 more)

### Community 1 - "Rule_Pool_Generator"
Cohesion: 0.12
Nodes (9): Phase 2: GPU-accelerated multi-objective evolutionary rule pool generation.…, Inject guarded migration seeds for the next epoch., Restore slimmed training data from cache (no re-sampling needed)., Rebuild engines after ``park_engines`` dropped GPU state., Return whether this generator should avoid allocating a JAX engine.…, Attach optional island metadata; safe when *owner* is a partial mock., Rule_Pool_Generator, TestRulePoolGeneratorInit (+1 more)

### Community 2 - "CPUBacktestEngine"
Cohesion: 0.07
Nodes (69): _build_rule_signal_mask(), CPUBacktestEngine, Cached wrapper around :func:`_compute_rule_signal_mask`., CPU backtest engine that exactly mirrors evaluator_v5.ipynb's…, expectancy_lcb_pct(), Return a conservative lower bound for per-trade net expectancy. Exact CPU…, _available_symbols(), _balanced_phase2_shortlist() (+61 more)

### Community 3 - "_make_engine"
Cohesion: 0.09
Nodes (19): _make_df(), _make_engine(), DataFrame, All TP trades → win_rate = 100%., No losing trades → profit_factor = 99.0., Build a minimal DataFrame for testing., position_notional = equity * capital_pct/100 * leverage when exposure is free., position_notional is capped when exposure is nearly full. (+11 more)

### Community 4 - ".run"
Cohesion: 0.07
Nodes (29): FileHandler, _log_phase_entry(), _log_pipeline_config(), _now_iso(), Create a run identity and remove artifacts that cannot be trusted., Persist the final status for the current run identity., Mark standalone phase runs failed when an exception escapes., Run all pipeline phases in order. Returns ------- dict Results from each phase,… (+21 more)

### Community 5 - "trend_context.py"
Cohesion: 0.08
Nodes (49): Number of leading per-symbol rows belonging to the training prefix. Shared by…, train_prefix_row_count(), align_completed_states_to_rows(), average_true_range(), build_higher_bars(), build_manifest(), build_train_prefix(), _classify_hf_bars() (+41 more)

### Community 6 - "test_evox_runner.py"
Cohesion: 0.05
Nodes (45): extract_deployable_migrants(), _inherit_val_metrics_from_global_cache(), _inject_diversity_recovery(), Phase2EvolutionState, Replace a fraction of the population with fresh or archive-mutated seeds., Copy val_* from global cache for identical chromosomes when val is skipped.…, Return elite deployable-preview entries suitable for guarded migration., Resumable NSGA-III state for symbol-island epoch scheduling. (+37 more)

### Community 7 - "_score_metrics"
Cohesion: 0.06
Nodes (43): _combined_return_score(), _evaluate_ruleset(), _optimize_risk(), Return (ok, bonus, penalty) for the desired train-valid balance shape. In…, Dominant objective: return/DD with train-valid balance, plus CV-fold…, Profit objective for lenient rule addition, but now evaluator_v5 aware. A new…, _score_metrics(), _train_valid_shape() (+35 more)

### Community 8 - "_get_dont_cares"
Cohesion: 0.07
Nodes (27): _count_active_conditions(), _get_dont_cares(), _mutate(), Mutate a chromosome (returns a copy). When activating a dont_care gene, feature…, Return array of dont_care sentinels for each feature., Count active rule conditions (sparse slots or dense dont_care encoding)., _is_all_inactive_sparse(), _make_feature_infos() (+19 more)

### Community 9 - "_make_df"
Cohesion: 0.08
Nodes (22): _make_df(), _make_engine(), DataFrame, MonkeyPatch, Chromosome positions must follow feature_modes insertion order., Exact CPU re-evaluation must interpret genes like the GPU path., Chromosome that matches nothing returns 0 executed trades., Zero-signal chunks should use reject metrics without scanning. (+14 more)

### Community 10 - "phase2_rule_pool.py"
Cohesion: 0.10
Nodes (50): encoder.py — Encoder Maps gene integer values to fuzzy value names, formats…, _random_active_class(), _archive_direction(), _build_cpu_archive_engine(), _chromosome_for_pool_export(), _crossover(), phase2_rule_pool.py — Rule_Pool_Generator (Phase 2) GPU-accelerated multi-…, Uniform crossover (dense per-gene or sparse per-slot). (+42 more)

### Community 11 - "_should_plateau_early_stop_phase2"
Cohesion: 0.05
Nodes (49): True for cluster/orphan scoped runs rather than the global path., scoped_island_profile(), Resolve the plateau patience value based on profile and stage. Cluster/orphan…, Earliest gen for plateau stop. Island epochs (~20 gens) and scaled Stage A…, _resolve_plateau_min_generation(), _resolve_plateau_patience(), _should_early_stop_phase2(), _should_plateau_early_stop_phase2() (+41 more)

### Community 12 - "TestArchiveSaveExceptionHandling"
Cohesion: 0.15
Nodes (14): _capture_logs(), _make_feature_infos_crash(), _make_train_df_crash(), DataFrame, Requirement 3.1 — save_archive is called before _release_resources in run()., Mock save_archive and _release_resources, run Rule_Pool_Generator.run() with a…, save_archive is called with self.direction as the first argument., Requirement 3.3 — If save_archive raises, the exception is caught, a WARNING is… (+6 more)

### Community 13 - "TestOOSEvaluatorRun"
Cohesion: 0.14
Nodes (5): Write a valid selected-features JSON to path., Integration tests using tmp_path overrides for all output paths. The run()…, Override module-level path dicts and return originals (for standalone tests)., TestOOSEvaluatorRun, _write_selected_features()

### Community 14 - "Pipeline_Orchestrator"
Cohesion: 0.09
Nodes (40): Pipeline_Orchestrator, Return the active output root for this run., Top-level orchestrator for the GPU-Fuzzy Trading Pipeline. Runs all five phases…, Temporarily rebind all cached output paths for one pipeline run., Return only non-empty RB strategies accepted for deployment., Reject a mixed, stale, or altered enriched train/test input pair., _resolve_output_root(), _temporary_output_paths() (+32 more)

### Community 15 - "_apply_monthly_admission_gate"
Cohesion: 0.06
Nodes (29): _apply_monthly_admission_gate(), _evaluate_rule_on_window(), _monthly_window_metrics(), Evaluate a single pool rule on a single monthly window. Returns the full window…, Normalize a window evaluator result for the monthly gate. The float fallback…, Apply the monthly-window shadow-test gate to a pool of rules. Each rule is…, _DeterministicEvaluator, _DeterministicMetricsEvaluator (+21 more)

### Community 16 - "Feature_Detector"
Cohesion: 0.07
Nodes (51): Feature_Detector, Series, detector.py — Feature_Detector Classifies each feature column into exactly one…, Classify feature columns by their discretization type., Classify a single feature series into one of six modes. Detection order…, all_nan_series(), arbitrary_numeric_series(), binary_series() (+43 more)

### Community 17 - "_make_train_df"
Cohesion: 0.09
Nodes (21): Sample up to *total_rows* rows, distributed equally across symbols. A single…, _sample_df(), _make_train_df(), DataFrame, Critical: bars must be contiguous so the backtest engine preserves temporal…, Random start must be bounded so the slice always fits forward., divmod distribution gives exactly total_rows (no rounding loss). 701_000 % 14…, total_rows < n_symbols must NOT force 1 row per symbol. (+13 more)

### Community 18 - "_split"
Cohesion: 0.06
Nodes (26): DataFrame, Unit tests for gpu_fuzzy_trader.data.splitter.Data_Splitter Tests cover: - Per-…, Helper: build df, patch paths, run split, return (train, val)., Compute expected train/val per symbol under holdout+embargo split., floor(N * HOLDOUT_TRAIN_FRACTION) rows go to train., Remaining rows after embargo gap go to validation., For N=101: floor(101 * HOLDOUT_TRAIN_FRACTION) used, not round., train + validation + embargo dropped == total rows. (+18 more)

### Community 19 - "test_cpu_engine_properties.py"
Cohesion: 0.08
Nodes (43): equity_tracking_scenario(), _expected_outcome(), fee_deduction_scenario(), _make_engine(), _make_engine_custom(), _make_minimal_df(), _make_simple_df(), multi_symbol_scenario() (+35 more)

### Community 20 - "detect_feature_mode"
Cohesion: 0.05
Nodes (38): detect_feature_mode(), Module-level convenience wrapper around Feature_Detector.detect_feature_mode., Unit tests for gpu_fuzzy_trader.features.detector.Feature_Detector Tests cover:…, All non-negative, zero_ratio > 0.3 → sparse_positive., All zeros: zero_ratio = 1.0 > 0.3, min = 0 → sparse_positive., zero_ratio just above 0.3 → sparse_positive., zero_ratio must be computed on the full series, not just non-NaN., Has negative values, zero_ratio > 0.3 → sparse_signed. (+30 more)

### Community 21 - "selector.py"
Cohesion: 0.08
Nodes (39): get_seed(), Return a per-process random seed (stable within a run, different across runs).…, build_phase1_shared_context(), _build_symbol_masks(), _candidate_feature_columns(), _check_spearman_sign_consistency(), _compute_chronological_stationarity_scores(), _frame_identity() (+31 more)

### Community 22 - "Feature_Selector"
Cohesion: 0.10
Nodes (19): Feature_Selector, Score and rank features separately for long and short directions., _make_train_df(), MonkeyPatch, Create a minimal training DataFrame with label columns and feature columns., Label columns must not appear in selected features., Meta columns must not appear in selected features., Loader internal columns and ``_``-prefixed names are not candidates. (+11 more)

### Community 23 - "rolling_cv.py"
Cohesion: 0.08
Nodes (42): CvFoldValEvaluator, Evaluate chromosomes on purged CV validation folds (excluding holdout).…, Build and cache one CPUBacktestEngine per CV fold (one-time cost)., aggregate_fold_metrics(), _bar_index_col(), _build_fold_from_ranges(), build_forbidden_ranges(), build_purged_walk_forward_folds() (+34 more)

### Community 24 - "Reporter"
Cohesion: 0.10
Nodes (13): Generates visual and tabular reports for each pipeline phase. All output files…, Reporter, _make_per_symbol_metrics(), _make_pnl_history(), _make_trade_log(), History entries with missing keys should not raise., Symbols with missing sub-keys should default to 0., Create a minimal Phase 2 history list with PnL fields. (+5 more)

### Community 25 - "test_encoder_properties.py"
Cohesion: 0.09
Nodes (37): all_active_chromosome_strategy(), chromosome_with_dont_cares_strategy(), feature_name_strategy(), composite, DrawFn, given, ndarray, Property-based tests for gpu_fuzzy_trader.features.encoder.Encoder **Validates:… (+29 more)

### Community 26 - "write_evaluator_clean"
Cohesion: 0.07
Nodes (32): _maybe_write_evaluator_clean(), Path, Write a stripped strategy file containing only ``direction`` and ``rules_set``.…, Write a stripped strategy file alongside the main strategy JSON. This is a…, Validate rule_set and write to JSON at path. After the main write, also writes…, Load JSON from path and run full schema validation. Parameters ---------- path…, write_evaluator_clean(), _make_rule() (+24 more)

### Community 27 - "CandidateRecord"
Cohesion: 0.09
Nodes (44): effective_rb_min_distinct_symbols(), Return the RB coverage target for the active debug universe. Full runs keep…, _candidate_coverage_symbols(), _candidate_positive_symbols(), CandidateRecord, _compose_ruleset(), _diversification_beam(), _diversification_shortlist() (+36 more)

### Community 28 - "nested_walk_forward.py"
Cohesion: 0.12
Nodes (26): Append one auditable record after a pipeline evaluation completes., Validation helpers for monthly, nested, and multiplicity-safe research., deflated_sharpe_ratio(), estimate_pbo(), Selection-multiplicity diagnostics for strategy research artifacts., Estimate the fraction of folds where the IS winner misses OOS median. Inputs…, Return a normal-approximation deflated Sharpe probability. This is a…, Create a compact ledger-ready multiplicity report. (+18 more)

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
Cohesion: 0.06
Nodes (25): Output_Writer, Serializes RuleSet dicts to JSON and loads/validates existing JSON files.…, _legacy_writer_contract(), _make_rule(), _make_rule_set(), fixture, parametrize, Unit tests for gpu_fuzzy_trader.output.writer — Output_Writer Tests cover: -… (+17 more)

### Community 36 - "config.py"
Cohesion: 0.09
Nodes (31): _config_check(), ConfigError, context_contract(), effective_config_snapshot(), effective_min_trade_support(), effective_monthly_min_trades(), effective_val_trade_floor_for_objectives(), _finite_config_number() (+23 more)

### Community 37 - "TestEquityCurveDateAxis"
Cohesion: 0.09
Nodes (21): _make_trade_log(), DataFrame, fixture, Unit tests for date-based x-axis in Reporter.plot_equity_curve (Task 10.3).…, When Entry_Time is absent, x-axis label is 'Trade #'., When Entry_Time is all-NaN, x-axis label is 'Trade #'., axhline is called (initial-capital line) in date mode., axhline is called (initial-capital line) in Trade # mode. (+13 more)

### Community 38 - "sample_df_for_phase2"
Cohesion: 0.07
Nodes (34): _downsample_chronological(), _largest_safe_range(), DataFrame, Build train/val backtest engines., Preserve per-symbol time order required by exposure/release simulation., Build the selected Phase 2 backend for the sampled train frame., Build an engine on *df* using the same backend selection logic., Pick a contiguous chronological slice of *n_rows* from *df*. Critical for… (+26 more)

### Community 39 - "research_integrity.py"
Cohesion: 0.16
Nodes (24): _canonical_json(), count_trials(), dataset_manifest(), ExperimentLedger, forward_acceptance_lock_path(), Any, Path, PathLike (+16 more)

### Community 40 - "_loader_from_rows"
Cohesion: 0.08
Nodes (28): _base_row(), _loader_from_rows(), _make_csv(), _make_ohlcv_rows(), _make_rows(), _make_timestamps(), DataFrame, Unit tests for gpu_fuzzy_trader.data.loader.Data_Loader Tests cover: - CSV… (+20 more)

### Community 41 - "_symbol_specialized_variants"
Cohesion: 0.13
Nodes (23): _attach_source_symbol_filters(), _ensure_symbol_filtered_rule(), _has_symbol_condition(), _is_symbol_condition(), Island/cluster symbols carried on Phase 2 pool entries., Feature ANDs + island symbol ORs (engine ORs multiple ``symbol is X``)., Return rule with an explicit symbol filter when required. This is a safety net…, Build symbol-filtered variants and rank them using evaluator_v5 scoring.… (+15 more)

### Community 42 - "gpu_engine.py"
Cohesion: 0.07
Nodes (36): _batch_metrics_from_array(), _build_event_batch(), _exposure_slot_capacity(), _fast_reject_result_rows(), _jax_compute_rule_signals_sparse_batch(), _jax_open_slot(), _jax_release_open_slots(), _jax_simulate_equity_batch() (+28 more)

### Community 43 - "_gpu_runtime.py"
Cohesion: 0.10
Nodes (32): phase2_should_enrich_symbol_metrics(), Return True when GPU batch eval should run a follow-up CPU enrichment pass.…, configure_phase2_gpu_runtime(), detect_gpu_memory_used_gb(), detect_gpu_vram_gb(), detect_system_ram_gb(), _iter_warmup_targets(), log_gpu_runtime_config() (+24 more)

### Community 44 - "._ensure_dir"
Cohesion: 0.06
Nodes (19): Reporting and visualization sub-package., _bucket_series_by_mode(), DataFrame, Series, reporter.py — Reporting and visualization for the GPU-Fuzzy Trading Pipeline.…, Write feature-stratified performance metrics to CSV files (one per split). For…, Create parent directories for *path* if they do not exist., Compute maximum drawdown percentage from an equity series. Parameters… (+11 more)

### Community 45 - "_derive_island_seed"
Cohesion: 0.09
Nodes (23): _derive_island_seed(), Return a deterministic but distinct seed per island from base seed + id.…, _coverage(), _print_island_sample_diagnostics(), _print_split_coverage(), DataFrame, Print coverage on deterministic singleton-island train/val windows., Return eligible-row counts for one direction and split frame. (+15 more)

### Community 46 - "test_certificate_first_selection.py"
Cohesion: 0.11
Nodes (27): _passes_symbol_concentration_gate(), _passes_symbol_contribution_certificate(), _passes_tail_selection_gate(), _portfolio_selection_certificate(), Any, Require positive, supported validation PnL from multiple symbols. Symbol…, Build the per-direction symbol certificate policy. Specialist islands are…, Return the certificate used by compose, risk, and profit selection. (+19 more)

### Community 47 - "non_dominated_sort"
Cohesion: 0.13
Nodes (26): batch_hamming_min(), crowding_distance(), _crowding_distance_numba(), _crowding_distance_py(), _disable_numba_sort(), _dominates_numba(), _dominates_py(), non_dominated_sort() (+18 more)

### Community 48 - "optuna_search.py"
Cohesion: 0.10
Nodes (29): _active_search_space(), apply_trial_config(), collect_validation_metrics(), compute_score(), _copy_phase1_2_outputs(), main(), objective(), _pearson() (+21 more)

### Community 49 - "._load_and_split_data"
Cohesion: 0.20
Nodes (11): apply_fuzzy_feature_scaling(), fit_fuzzy_feature_scaling(), Any, DataFrame, Train-fitted normalization for ordinal ``ff_*`` fuzzy feature codes. The…, Build a train-only scaling contract for ordinal ``ff_*`` columns., Apply an existing train-fitted scaling contract in place and return *df*., Load train_new.csv and split into train/validation DataFrames. When… (+3 more)

### Community 50 - "prop_settings"
Cohesion: 0.05
Nodes (52): HealthCheck, prop_settings(), Shared Hypothesis settings for property tests. Set PYTEST_LOW_MEMORY=1 to scale…, Hypothesis settings with optional low-memory example scaling., _make_timestamps(), composite, DataFrame, DrawFn (+44 more)

### Community 51 - "_compute_rule_signal_mask"
Cohesion: 0.15
Nodes (13): conditions_cache_key(), get_or_build_rule_mask(), ndarray, condition_cache.py — Cached boolean masks for textual rule conditions. Avoids…, Stable hashable key for a list of condition strings., Return a boolean row mask for *conditions*, using *cache* when provided. The…, _compute_rule_signal_mask(), Build one boolean signal mask (evaluator_v5 parity). Feature conditions are… (+5 more)

### Community 52 - "validate_config"
Cohesion: 0.18
Nodes (19): Validate all high-impact hyperparameter relationships. The function is…, Write the effective configuration snapshot and return its path., validate_config(), write_config_audit_report(), MonkeyPatch, Path, Cross-parameter configuration and evaluator-parity tests., test_audit_report_writes_the_effective_snapshot() (+11 more)

### Community 53 - "build_hybrid_symbol_clusters"
Cohesion: 0.12
Nodes (28): build_hybrid_symbol_clusters(), _corr_embedding_block(), _feature_names_union(), _feature_profile_block(), Any, DataFrame, ndarray, symbol_cluster.py — Per-symbol clustering for Phase 2 island scheduling using… (+20 more)

### Community 54 - "_init_population"
Cohesion: 0.14
Nodes (22): assign_strata_to_indices(), build_feature_sampling_probs(), pick_active_count(), _pick_active_index(), _pick_inactive_index(), ndarray, phase2_init.py — Sparsity-guided stratified population initialization for Phase…, Assign elite / explorer labels to non-seeded population rows. (+14 more)

### Community 55 - "test_phase2_rule_pool.py"
Cohesion: 0.06
Nodes (33): _crowding_distance(), _deployable_archive_pool_entries(), _merge_archive_entries(), _pareto_sortino_stats(), _pool_seed_chromosomes(), Compute crowding distance for solutions in *front*. Parameters ----------…, Deduplicate and rank archive entries, keeping the best *max_size* rules., Stack chromosome rows into a batch (dense 2D or sparse 3D). (+25 more)

### Community 56 - "detect_all_modes"
Cohesion: 0.33
Nodes (4): detect_all_modes(), DataFrame, Classify every column in *feature_cols* and return a mapping. Parameters…, Module-level convenience wrapper around Feature_Detector.detect_all_modes.

### Community 57 - "test_reporter.py"
Cohesion: 0.07
Nodes (29): _make_dataset_with_label(), _make_datasets_by_split(), _make_selected_features(), _make_stratified_dataset(), _make_stratified_datasets_by_split(), _make_stratified_logs_by_split(), _make_stratified_trade_log(), DataFrame (+21 more)

### Community 58 - "GPUBacktestEngine"
Cohesion: 0.10
Nodes (16): GPUBacktestEngine, CPU engine for rule-set simulation used by Phase 2/RB., Rule-set sim using a mask cache (delegates to the CPU engine)., Evaluate multiple rule sets (parallel CPU; optional mask cache)., RB JAX path: cached mask entry build + parallel CPU equity. Equity results…, Same interface as CPUBacktestEngine for compatibility., JAX-accelerated backtest engine for Phase 2 rule pool generation.…, Return the JAX backend in use ('gpu', 'cpu', or 'tpu'). (+8 more)

### Community 59 - "TestGPUCPUNumericalParity"
Cohesion: 0.10
Nodes (15): ndarray, simulate_rule_set (delegated to CPU) is exactly identical to CPU engine., GPU backtest engine must handle padded chunking correctly., CPU backtest engine must support simulate_rule_batch with correct metrics., Compare GPU and CPU engine for 10 random chromosomes (Task 2 acceptance…, Build a multi-symbol DataFrame with mixed TP/SL/time-exit outcomes., Generate random chromosomes for binary + signed features., GPU engine results must match CPU engine within specified tolerances for 10… (+7 more)

### Community 60 - "cpu_engine.py"
Cohesion: 0.06
Nodes (32): _append_allocated_entries(), _batch_eval_rule_set_pickled(), _expectancy_lcb_pct(), _expected_shortfall_pct(), _jax_runtime_loaded(), _parse_condition(), precompute_release_indices(), precompute_release_indices_from_offsets() (+24 more)

### Community 61 - "test_phase2_use_gpu_flag.py"
Cohesion: 0.25
Nodes (7): _minimal_backtest_df(), DataFrame, MonkeyPatch, Phase 2 respects PHASE2_USE_GPU=False without loading JAX GPU engine., The memory-safe CPU route must happen before JAX allocates arrays., A selected CPU backend must not initialize JAX just to warm up., TestPhase2UseGpuFlag

### Community 62 - "_derive_val_sample_seed"
Cohesion: 0.15
Nodes (11): _derive_val_sample_seed(), Derive a deterministic validation sample seed from the training seed. This…, AC: Train and validation sampling use distinct RNG seeds by default., _derive_val_sample_seed returns a value different from the input., Same train seed always produces same val seed., Result is in [0, 2**31) so it is a valid random seed., Rule_Pool_Generator stores distinct _sample_seed and _val_sample_seed., When seed=None, val seed is derived from PHASE2_SEED. (+3 more)

### Community 63 - "TestRulePoolGeneratorRun"
Cohesion: 0.09
Nodes (12): Integration tests using tiny population and generation counts., In holdout mode, val engine must be built for pool admission even when…, All pool entries must have active conditions within [MIN_CONDITIONS,…, Phase 2 must use static TP=PHASE2_TP, SL=PHASE2_SL,…, Pool entries must have executed_trades >= MIN_TRADE_POOL_FLOOR., Verify that Rule_Pool_Generator's persistent self._rng advances across multiple…, After two run_epoch() calls, the internal RNG state must differ from the…, The RNG should produce *different* sequences in two consecutive run_epoch()… (+4 more)

### Community 64 - "test_plateau_state_leak.py"
Cohesion: 0.11
Nodes (21): _make_minimal_gen(), _mock_evolution_state(), _mock_stage_plan(), Regression tests for plateau-state leak fixes (Fixes A + B). Fix A:…, AC-1, AC-3: plateau_streak resets per epoch; reset_plateau=True always., reset_plateau=True is passed on the very first epoch call., reset_plateau=True is also passed on epoch 2 (regression for leak)., Even when PHASE2_ISLAND_TWO_STAGE_ENABLED=False, reset_plateau=True. (+13 more)

### Community 65 - ".prepare_test_data"
Cohesion: 0.11
Nodes (16): DataFrame, Prepare test data using Data_Loader.load_dataset(). Applies the same…, Load and validate a strictly newer, untouched forward period., _make_timestamps(), composite, DataFrame, DrawFn, given (+8 more)

### Community 66 - "dashboard.py"
Cohesion: 0.15
Nodes (25): build_dashboard_data(), _direction_data(), _history_rows(), _load_json(), main(), Any, Path, Read-only HTML dashboard for existing pipeline artifacts. The dashboard… (+17 more)

### Community 67 - "test_phase2_val_sim_interval.py"
Cohesion: 0.15
Nodes (14): CountingEngine, Unit tests for periodic val simulation (Phase 2 runtime A2)., PHASE2_VAL_SIM_INTERVAL=1 preserves original behaviour (val every gen)., Direct unit test of _should_run_val_this_gen with interval=3. With interval=3,…, Integration test: with interval=3 and 13-gen epoch, val runs on gens 0, 3, 6,…, Val metrics for a chromosome are deterministic when val is skipped. With…, With PHASE2_VAL_SIM_INTERVAL=2, val sim runs on interval gens (0,2,4) + last…, Validation-dependent fitness is evaluated consistently every gen. (+6 more)

### Community 68 - "test_data_loader_properties.py"
Cohesion: 0.14
Nodes (26): dataframe_with_nan_features(), dataframe_with_nan_labels(), _load_from_df(), _make_datetime_series(), composite, DataFrame, DrawFn, given (+18 more)

### Community 69 - "barrier_column_names"
Cohesion: 0.15
Nodes (18): attach_barrier_outcomes(), barrier_column_names(), configured_barrier_pairs(), _first_touch_for_symbol(), _first_touch_for_symbol_numba(), _first_touch_for_symbol_python(), _number_token(), DataFrame (+10 more)

### Community 70 - "_m"
Cohesion: 0.13
Nodes (16): evaluator_health_penalty(), Penalty for evaluator_v5 failure modes — higher is worse. Parameters ----------…, _m(), Max positions 15 > 10 → positive penalty., Max positions 8 <= 10 → no additional penalty from positions., Missing ``raw_signal_count`` → treated as 0 → no skip/exec penalty., ``raw_signal_count=0`` → no skip/exec ratio computed → no penalty., Role 'train' same as 'valid' (1.0x). (+8 more)

### Community 71 - "_apply_dynamic_rule"
Cohesion: 0.21
Nodes (4): _apply_dynamic_rule(), Apply one exported text condition using the original threshold logic. Exactly…, Test all threshold branches of _apply_dynamic_rule., TestApplyDynamicRule

### Community 72 - "test_cpu_engine.py"
Cohesion: 0.08
Nodes (20): _normalize_direction(), Compute a non-annualized Sortino Ratio from per-trade returns., _safe_profit_factor(), _sortino_ratio_from_returns(), CPU and GPU backtest engine sub-package., JointPortfolioEngine, DataFrame, Joint long/short portfolio simulation. Phase 2 and RB score each direction… (+12 more)

### Community 73 - "mandatory_context_conditions"
Cohesion: 0.10
Nodes (26): context_contract_digest(), mandatory_context_conditions(), Return the fixed, mandatory context conditions for *direction*., Return a stable hash of the static contract and fitted enrichment., feature_conditions_only(), phase2_rule_id(), Any, Stable provenance identifiers for Phase 2 feature rules. The RB Governor may… (+18 more)

### Community 74 - "execution_ok"
Cohesion: 0.15
Nodes (11): execution_ok(), Return ``True`` iff the evaluator would execute this rule set reasonably. A…, Tests for ``execution_ok``., Skip ratio 0.10 <= 0.20 and exec ratio 0.90 >= 0.60 → True., Skip ratio 0.30 > 0.20 → False., Exec ratio 0.50 < 0.60 → False., Missing ``raw_signal_count`` → treated as 0 → False., ``raw_signal_count=0`` → False. (+3 more)

### Community 75 - "TestWriteStrategyEvaluationTable"
Cohesion: 0.21
Nodes (9): _make_eval_rule_set(), _make_metrics_by_split(), _make_trade_logs_by_split(), Create a minimal rule set for evaluation table tests., Create a metrics_by_split dict with all three splits populated., Create a trade_logs_by_split dict with all three splits populated., Sharpe ratio should equal mean(r)/std(r, ddof=1) for a known log., win_rate, mdd_pct etc. should come from metrics_by_split. (+1 more)

### Community 76 - "test_gpu_engine.py"
Cohesion: 0.14
Nodes (12): _build_data_matrix(), _discretize_series(), DataFrame, Series, Discretize a feature series using evaluator_v5's fixed fuzzy bins. Exported…, Build an (N, K) integer matrix of discretized feature values., disable_skip_optimization(), fixture (+4 more)

### Community 77 - "TestRunLogHandlerLifecycle"
Cohesion: 0.12
Nodes (15): MonkeyPatch, Phase 2 reproducibility default seed., PHASE2_SEED is drawn once at import via get_seed(); set GLOBAL_SEED=42 to fix…, Requirements 1.1, 1.4, 1.5, 1.6, 1.7 — run.log FileHandler is attached, writes…, Count FileHandlers on the root logger pointing to *path*., Patch every phase method on Pipeline_Orchestrator to be a no-op., run.log must exist after run() and contain both separator lines., Root logger must have no extra FileHandlers pointing to run.log after run(). (+7 more)

### Community 78 - "TestPrecomputeReleaseIndices"
Cohesion: 0.29
Nodes (4): Release index should point to the row where bar_index + 288 is reached., Rows near the end should get release_index = len(df)., Each symbol's release indices are computed independently., TestPrecomputeReleaseIndices

### Community 79 - "test_feature_selector_properties.py"
Cohesion: 0.15
Nodes (23): dataset_with_high_dispersion_feature(), dataset_with_label_and_meta_columns(), dataset_with_sufficient_dispersion_feature(), _make_label_block(), composite, DataFrame, DrawFn, given (+15 more)

### Community 80 - ".get_dont_care"
Cohesion: 0.17
Nodes (7): See module-level :func:`get_dont_care`., **Property 9: Don't-Care Sentinel Correctness — encode_condition raises**…, **Property 9: Don't-Care Sentinel Correctness — all-dont_care → empty output**…, test_property_9b_encode_condition_raises_for_dont_care(), test_property_9f_all_dont_care_chromosome_returns_empty(), Static methods should be callable on the class itself., TestGetDontCare

### Community 81 - "compute_labels"
Cohesion: 0.20
Nodes (10): compute_labels(), DataFrame, Forward-window label computation for OHLCV bars. Horizon is ``TAIL_DROP_ROWS``…, Compute the 5 label columns per symbol. - label_open_next: open[t+1] -…, DataFrame, ndarray, Unit tests for forward-window label semantics…, Regression: old backward rolling must not match corrected forward labels. (+2 more)

### Community 82 - "TestPlotPerRuleBreakdown"
Cohesion: 0.16
Nodes (11): _make_full_trade_log(), _make_rule_set(), _make_split_logs(), Create a trade log with all columns needed for sharpe computation., Create a minimal rule set with *n* rules., Create a trade log with Rule_Index, Net_PnL, Equity_After columns., Create trade_logs_by_split dict with all three splits populated., One split None, others valid — should not raise. (+3 more)

### Community 83 - "resolve_island_hyperparams"
Cohesion: 0.11
Nodes (21): effective_phase2_val_return_floor_pct(), Direction-aware Phase 2 validation return floor for fitness penalties., Scale integer trade floors by slice size vs full-universe reference., Resolve scaled trade floors and relaxed cross-symbol gates., resolve_island_hyperparams(), scale_trade_floor_by_universe(), Regression tests for anti-overfit / exploration-retune config bundle., test_anti_overfit_config_bundle() (+13 more)

### Community 84 - "test_phase2_island_scheduler.py"
Cohesion: 0.16
Nodes (14): compute_cluster_generation_budgets(), Resolve per-island generation budgets. By default each island receives the full…, parametrize, Unit tests for cluster island scheduler budget math and epoch guard., Multi-symbol clusters each get the full island budget (not split)., Behavioral tests for _should_skip_epoch helper used in _run_cluster_islands., Verify _should_skip_epoch returns correct value., Only the evolution runner may abort after its guarded collapse logic. (+6 more)

### Community 85 - "_build_target"
Cohesion: 0.17
Nodes (12): _build_target(), Build a direction-specific target signal. Default…, Encoding-aware win class: 2 in asymmetric mode, 1 in legacy mode., Encoding-aware loss class: 0 in either mode., Long: max >= entry*(1+TP/100), min > entry*(1-SL/100) → success., Long: both hit but max_before_min==0 → SL first → failure., Long: both hit but max_before_min==1 → TP first → success., Long: neither TP nor SL hit → failure (or neutral in asymmetric mode). (+4 more)

### Community 86 - "filter_migrants_for_cluster"
Cohesion: 0.11
Nodes (19): filter_migrants_for_cluster(), _migrant_to_metrics(), Backtest one migrant chromosome on receiver cluster engines., Accept only migrants that pass deployability on the receiver cluster slice., _make_migrant_dict(), _make_mock_receiver(), Unit tests for migration safety — migrant gate and seed fraction. Acceptance…, Migrant with val_return 2.5% and >=15 val trades should be accepted. (+11 more)

### Community 87 - "test_gpu_engine_properties.py"
Cohesion: 0.16
Nodes (17): _assert_parity(), _make_engines(), _make_parity_df(), parity_scenario_strategy(), composite, DataFrame, DrawFn, given (+9 more)

### Community 88 - "test_rb_fail_closed.py"
Cohesion: 0.29
Nodes (11): _dummy_df(), _pool_rules(), DataFrame, Path, RB Governor fail-closed and stale-output regression tests., test_empty_phase2_pool_writes_fail_closed_output_with_reason(), test_fail_closed_output_overwrites_stale_strategy(), test_no_positive_good_candidates_fail_closed_and_do_not_call_fallback() (+3 more)

### Community 89 - "_derive_epoch_seed"
Cohesion: 0.16
Nodes (10): _derive_epoch_seed(), Return a deterministic per-epoch seed derived from *base_seed* + epoch. Used by…, Re-sample training data with a per-epoch rotated window. Each epoch gets a…, An unknown PHASE2_PER_EPOCH_WINDOW_SEED_MODE raises ValueError., Deterministic seed derivation for per-epoch windows., Same (base_seed, epoch_idx) produces the same seed., Different epoch indices produce different seeds., None base seed returns None. (+2 more)

### Community 90 - "Data_Loader"
Cohesion: 0.07
Nodes (35): Data_Loader, _ensure_labels(), load_dataset(), DataFrame, data/loader.py — Data_Loader Stateless CSV loading with full preparation…, Recompute the LWC pullback-reversal triggers and compare row-by-row. A stale or…, Keep supplied labels or derive all labels from raw OHLCV columns. The original…, Stateless data loader for the GPU-Fuzzy Trading Pipeline. (+27 more)

### Community 91 - "set_purged_wf_reference_rows"
Cohesion: 0.18
Nodes (7): Store full train_new.csv row count after loader prep (split time)., set_purged_wf_reference_rows(), holdout_mode(), purged_mode(), fixture, Tests for purged-WF trade-floor scaling helpers., TestScaleTradeFloor

### Community 92 - "downcast_numeric_df"
Cohesion: 0.23
Nodes (14): downcast_numeric_df(), prune_train_columns(), DataFrame, df_slim.py — Reduce DataFrame memory for backtest engines. Keeps only columns…, Downcast label and feature columns to float32; symbol to category (inplace)., Return a copy with only meta, label, and requested feature columns. Parameters…, Drop unused feature columns from the full training split after Phase 1., slim_backtest_df() (+6 more)

### Community 93 - ".encode_condition"
Cohesion: 0.10
Nodes (13): Encoder, Stateless encoder that maps gene values to fuzzy condition strings., See module-level :func:`encode_condition`., parametrize, Unit tests for gpu_fuzzy_trader/features/encoder.py Tests cover: -…, TestEncodeConditionBinary, TestEncodeConditionErrors, TestEncodeConditionPositive (+5 more)

### Community 94 - "test_output_writer_properties.py"
Cohesion: 0.15
Nodes (25): parse_symbol_condition(), Parse optional symbol filters. Supported formats: "symbol is 1" "symbol IS 1"…, all_zero_rule_st(), _is_valid_exported_condition(), oversized_rule_set_st(), composite, DrawFn, given (+17 more)

### Community 95 - "ValidationError"
Cohesion: 0.06
Nodes (32): context_permission_column(), context_trigger_column(), Return the direction-specific permission column name., Return the direction-specific LWC pullback-reversal trigger column., _condition_feature(), _context_feature_direction(), Exception, writer.py — Output_Writer Serializes RuleSet dicts to JSON with exact schema… (+24 more)

### Community 96 - "test_phase2_gpu_throughput.py"
Cohesion: 0.16
Nodes (16): _jax_compute_rule_signals_batch(), Batch rule matching for B chromosomes simultaneously. Returns (B, N) boolean…, get_gpu_backtest_engine_class(), jax_gpu_backtest_available(), Any, Detect whether JAX / GPUBacktestEngine can be loaded on this host. JAX can fail…, Return ``GPUBacktestEngine`` if ``gpu_engine`` imports cleanly; else ``None``., True when ``get_gpu_backtest_engine_class()`` would succeed. (+8 more)

### Community 97 - ".decode_chromosome"
Cohesion: 0.26
Nodes (4): See module-level :func:`decode_chromosome`., All 10 signed values decode correctly via decode_chromosome., Chromosome with numpy int32/int64 values should work., TestDecodeChromosome

### Community 98 - ".load_strategies"
Cohesion: 0.17
Nodes (8): Load long.json and short.json via Output_Writer.load_and_validate(). Parameters…, _isolate_phase5_reporter_outputs(), fixture, Keep Reporter plots/CSVs out of the checked-in outputs directory., Write a valid rule set JSON to path., Run OOS_Evaluator.run() once for the whole class (long direction only)., TestLoadStrategies, _write_rule_set()

### Community 99 - "gate_positive_good"
Cohesion: 0.28
Nodes (16): _finite_number(), gate_positive_good(), _metric_int(), positive_good_reject_reasons(), PositiveGoodThresholds, Shared metric gates used by Phase 2 admission and RB Governor. The gate is…, Minimum evidence required on train and validation splits., Return stable, machine-readable reasons for a gate rejection. (+8 more)

### Community 100 - "phase2_island_scheduler.py"
Cohesion: 0.10
Nodes (31): filter_df_to_symbols(), phase2_history_path(), phase2_pool_path(), DataFrame, Return rows for the given symbols; raises if column missing or no rows., Resolve Phase 2 pool path., Resolve Phase 2 history path., clear_global_metrics_cache() (+23 more)

### Community 101 - ".save_archive"
Cohesion: 0.22
Nodes (9): _archive_feature_signature(), Return the ordered feature signature used to validate archive reuse., Read JSON from *path* and return None when the file cannot be loaded., Validate the archive JSON structure and feature compatibility., Load a compatible persistent archive if it exists, otherwise return None.…, Merge the latest pool into a persistent archive and write atomically., _read_json_payload(), _validate_archive_payload() (+1 more)

### Community 102 - "TestRemoveRedundantFeatures"
Cohesion: 0.25
Nodes (4): Two features with corr > 0.95 → keep higher-scored one., Two uncorrelated features → both kept., Features in different modes are not compared for redundancy., TestRemoveRedundantFeatures

### Community 103 - "_make_df"
Cohesion: 0.09
Nodes (21): _make_df(), _make_rule_set(), DataFrame, Verify plot_equity_curve is called for all three splits and handles empty logs., Helper: run OOS_Evaluator.run() with mocked internals and capture plot calls., Override module-level path dicts and return originals., plot_equity_curve must be called with 'train', 'validation', and 'test'., plot_equity_curve must be called exactly three times (one per split). (+13 more)

### Community 104 - "test_crash_fix_and_run_logging.py"
Cohesion: 0.08
Nodes (29): _append_xla_flag(), configure_jax_env(), _cuda_package_root(), Path, JAX/XLA runtime configuration — call before the first ``import jax``., Find a pip-installed CUDA toolkit root, if one is available., Append an XLA flag once, preserving explicit user configuration., Configure JAX/XLA runtime for predictable desktop-friendly GPU usage. -… (+21 more)

### Community 105 - "TestPlotDistributionAndEquity"
Cohesion: 0.16
Nodes (6): _make_dist_logs_by_split(), _make_dist_trade_log(), Create a trade log with all columns needed for distribution/equity plots., Create trade_logs_by_split dict with all three splits populated., A trade log with a single trade should not raise., TestPlotDistributionAndEquity

### Community 106 - "baselines.py"
Cohesion: 0.24
Nodes (17): compute_entry_time_priority(), Map each row to a timestamp priority code (evaluator_v5 parity)., _compact(), _drop_one_condition_variants(), _equal_weight(), _evaluate(), evaluate_baselines(), _feature_shuffle() (+9 more)

### Community 107 - ".run"
Cohesion: 0.07
Nodes (31): _condition_feature_names(), _entry_validation_per_symbol_metrics(), _filter_compatible_previous_pool(), _filter_pool_by_admission(), _monthly_admission_source_df(), _pool_entry_passes_admission(), _pool_entry_rank(), _pool_path_key() (+23 more)

### Community 108 - "_jax_compute_trade_outcomes"
Cohesion: 0.24
Nodes (5): _jax_compute_trade_outcomes(), JAX-jitted vectorized trade outcome computation., Vectorized trade outcome computation for all rows. Mirrors…, Multiple rows processed simultaneously., TestComputeTradeOutcomes

### Community 109 - "_should_post_restart_early_stop_phase2"
Cohesion: 0.29
Nodes (9): Break the epoch when a plateau restart yields no improvement. Independent of…, _should_post_restart_early_stop_phase2(), Unit tests for post-restart no-improvement early stop (Phase 2 runtime)., test_global_disabled_no_stop(), test_global_uses_global_knobs(), test_island_disabled_no_stop(), test_island_streak_at_patience_stops(), test_island_streak_below_patience_no_stop() (+1 more)

### Community 110 - "FakeEngine"
Cohesion: 0.16
Nodes (9): The fallback must not switch f3 from CV return to PF after gen 0., TestRunPhase2EvolutionFallback, FakeEngine, Minimal engine stub that returns fixed metrics., Verify the plateau branch decision logic in _run_nsga3. We monkeypatch…, First plateau triggers _plateau_diversity_restart., When disabled, _plateau_diversity_restart is NOT called., With max_restarts=2, restart is called at least twice. (+1 more)

### Community 111 - "resolve_phase2_stage_params"
Cohesion: 0.06
Nodes (33): _diversity_recovery_min_unique_ratio(), Stop burning gens when the feasible set is empty and restarts are spent. Sparse…, True when Stage A viability is critically low and search has plateaued., _should_abort_zero_deployable_collapse(), _should_inject_diversity_recovery(), _should_viability_recovery(), island_stage_budgets(), IslandStagePlan (+25 more)

### Community 112 - "_validate_pool_schema"
Cohesion: 0.36
Nodes (3): Validate the structure of a loaded pool JSON. Raises ValueError if the schema…, _validate_pool_schema(), TestValidatePoolSchema

### Community 114 - "TestHammingThresholdAutoScale"
Cohesion: 0.15
Nodes (10): Verify the max(3, k_active // 5) formula. The formula is applied in…, Replicate the auto-scaling formula., k_active=0 → threshold = max(3, 0//5) = 3., k_active=5 → threshold = max(3, 5//5=1) = 3., k_active=15 → threshold = max(3, 15//5=3) = 3., k_active=20 → threshold = max(3, 20//5=4) = 4., k_active=50 → threshold = max(3, 50//5=10) = 10., k_active=100 → threshold = max(3, 100//5=20) = 20. (+2 more)

### Community 115 - "TestValLeakGate"
Cohesion: 0.20
Nodes (10): C6: Val-derived penalties must be gated behind JOINT_TRAIN_VAL or…, Return standard monkeypatching for clean baseline metrics., Apply base settings with optional overrides., Metrics that trigger no train-side penalties., Val metrics that WOULD trigger penalties if the gate were open., When both JOINT_TRAIN_VAL and VAL_IN_FITNESS_PENALTY are False, val-derived…, Bad val must not set feasibility_violation when gate is closed., When VAL_IN_FITNESS_PENALTY=True, val-derived penalties DO enter objectives. (+2 more)

### Community 116 - "_build_entries_from_rule_set"
Cohesion: 0.08
Nodes (20): _build_entries_from_rule_set(), Simulate a rule set on rows [row_start, row_end) without copying the df. Used…, Simulate a rule set and return performance metrics. Parameters ----------…, Priority-based rule assignment: first matching rule wins per row. Mirrors…, _rule_symbols_for_allocation(), _rules_need_normalized_symbols(), get_normalized_symbol_array(), normalize_symbol_value() (+12 more)

### Community 117 - "_jax_compute_rule_signals"
Cohesion: 0.15
Nodes (11): _jax_compute_rule_signals(), _maybe_jit(), JAX-jitted vectorized rule matching (single chromosome)., Apply ``jit`` if JAX is available, otherwise return *fn* unchanged., Vectorized rule matching: returns (N,) boolean mask of matching rows., All rows match when chromosome equals data_matrix values., No rows match when chromosome differs from data_matrix., Only rows where all active conditions match. (+3 more)

### Community 118 - "TestParetoCollapseWarningGate"
Cohesion: 0.15
Nodes (10): _FakeEngine, AC 4: The default value of the config flag is 5., AC 5: min_pareto_size=0 → warning fires even with small pareto (size=4)., The log message includes 'pareto_size=N' suffix., Fake engine that returns metrics producing a tradeoff between f1 (-sortino) and…, AC 1–5: warning gated on len(pareto_indices) >= config threshold., Run 2-gen evolution and return count of 'Pareto collapse risk' warnings., AC 2: pareto_size=4 < min_pareto_size=5 → no warning fires. (+2 more)

### Community 119 - "TestMakeWalkForwardFoldEngines"
Cohesion: 0.17
Nodes (10): _make_synthetic_df(), DataFrame, Verify _make_walk_forward_fold_engines splits data correctly., 2 splits + tail holdout → 2 fold engines + 1 tail engine., tail_holdout_frac=0 → tail engine is None., Each symbol's data is divided into contiguous chunks across folds., Single symbol without symbol column is handled gracefully., Very small data per symbol (fewer rows than n_splits) does not crash. (+2 more)

### Community 120 - "_compute_stability"
Cohesion: 0.17
Nodes (9): _compute_stability(), Compute cross-symbol stability score. stability = 1 - (std / mean) If mean is 0…, All symbols have same score → std=0 → stability=1., Mean is 0 → stability=0., Single symbol with positive score → stability=1., Single symbol with zero score → stability=0., Very high variance → stability could be negative → clipped to 0., Stability should always be in [0, 1]. (+1 more)

### Community 121 - "TestPoolAdmissionOverfitRatioGate"
Cohesion: 0.24
Nodes (7): MonkeyPatch, Tests for the hard overfit ratio gate in pool admission. The ratio gate rejects…, train=15%/val=4% (3.75× ratio, gap=11pp < 16pp) is REJECTED by the ratio gate…, train=15%/val=10% (1.5× ratio) is ADMITTED., train/val ≈2.5× is ADMITTED when under OVERFIT_RATIO_FLOOR and gap gate., With PHASE2_OVERFIT_RATIO_FLOOR=0.0, the ratio gate is disabled and the high-…, TestPoolAdmissionOverfitRatioGate

### Community 122 - "MonthlyWindowSummary"
Cohesion: 0.06
Nodes (40): _monthly_selection_certificate(), Require a composed team to be mostly non-loss across calendar windows., build_monthly_windows(), _datetime_series(), evaluate_rule_set_monthly(), monthly_penalty(), monthly_return_counts_as_good(), MonthlyWindowSummary (+32 more)

### Community 123 - "._engine"
Cohesion: 0.20
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

### Community 127 - "TestEndToEndRotation"
Cohesion: 0.18
Nodes (7): fixture, Integration-style tests with a mocked Rule_Pool_Generator., Patch config for rotation and create a generator with minimal setup., When rotation is enabled, _cached_scoped_train_df is stored., After resample_train_for_epoch, the cached slim train changes., Same epoch_idx produces identical cached slim train., TestEndToEndRotation

### Community 128 - "stratification_scenario_strategy"
Cohesion: 0.24
Nodes (15): dataset_with_features_strategy(), metrics_strategy(), composite, DataFrame, DrawFn, Generate a trade log DataFrame with 2–50 rows. Parameters ---------- n_rules:…, Generate a metrics dict with reasonable float values. Keys: win_rate,…, Generate a dataset DataFrame with 10–100 rows. Parameters ----------… (+7 more)

### Community 129 - "TestSpearmanSignConsistency"
Cohesion: 0.17
Nodes (10): DataFrame, ndarray, AC3: Train all positive, tiny |val_rho| < min_abs_corr → feature still kept., AC4: val_df=None → pre-task-8 behavior (mixed train signs still blacklisted)., AC5: val_df without label_close_288 column → val check skipped (no crash)., Tests for _check_spearman_sign_consistency, including the val_df check., Build a minimal DataFrame without 'symbol' to avoid symbol-based folding., AC1: Train all positive, val negative → feature blacklisted. (+2 more)

### Community 130 - "run_pipeline.py"
Cohesion: 0.09
Nodes (38): context_coverage_for_direction(), context_coverage_report(), context_floor_failures(), Any, DataFrame, Shared diagnostics for the mandatory direction-specific context contract., Return coverage diagnostics for every named frame and direction., Return mathematically impossible trade-floor failures for coverage. (+30 more)

### Community 131 - "log_memory_rss"
Cohesion: 0.11
Nodes (15): log_memory_rss(), Optional process memory logging (set LOG_MEMORY=1)., Log current process RSS in MiB when LOG_MEMORY=1., Force GC and clear JAX compilation caches between Phase 2 directions. Releases:…, release_phase2_resources(), Release GPU engines and slim in-memory data between island epochs., Selected fuzzy features for Phase 2., Drop unused feature columns from train/val splits to reduce RAM. (+7 more)

### Community 132 - "TestPlotPhase2Metrics"
Cohesion: 0.23
Nodes (4): _make_history(), History entries with missing keys should not raise., Create a minimal Phase 2 history list., TestPlotPhase2Metrics

### Community 133 - "_apply_colab_gpu_defaults"
Cohesion: 0.33
Nodes (6): _apply_colab_gpu_defaults(), is_colab_runtime(), True when running on Google Colab (/content runtime)., Colab T4 optimization for Phase 2 runs., MonkeyPatch, test_colab_defaults_apply_when_content_exists()

### Community 134 - "splitter.py"
Cohesion: 0.09
Nodes (27): _chronological_half_split(), _file_sha256(), _holdout_embargo_split(), load_cached_split_if_fresh(), _purged_walk_forward_split(), DataFrame, data/splitter.py — Data_Splitter Per-symbol chronological split for Phase 2,…, Per-symbol chronological first or second half of *df*. ``purge_rows`` is… (+19 more)

### Community 135 - "ConfigurationError"
Cohesion: 0.20
Nodes (10): ConfigurationError, decode_chromosome(), encode_condition(), get_dont_care(), Exception, ndarray, Return '[feature_name] IS Fuzzy Value Name' for a valid gene. Parameters…, Convert a chromosome array to a list of condition strings. Genes equal to the… (+2 more)

### Community 136 - "OOS_Evaluator"
Cohesion: 0.10
Nodes (12): OOS_Evaluator, Out-of-sample evaluator for Phase 5. Loads the final long/short strategies,…, Run out-of-sample evaluation. Parameters ---------- allowed_directions :…, Load selected features for a direction when available., Remove only known Phase 5 artifacts from the active report root., Evaluate a single strategy on the test DataFrame. Returns ------- metrics :…, Return an explicit, non-success result for a failed split., Build a flat list of per-symbol metric dicts for CSV output. Uses the… (+4 more)

### Community 137 - "compute_phase2_objectives_from_metrics"
Cohesion: 0.04
Nodes (47): compute_phase2_objectives_from_metrics(), _diversity_penalty_blended(), _hamming_distance(), _phenotype_bucket_key(), Hamming distance between two chromosomes (active pairs when sparse)., Discretise objective-relevant metrics for behavioral diversity., Hamming OR phenotype-bucket crowding penalty (same weight on both)., Penalty for weak cross-symbol robustness on one split. (+39 more)

### Community 138 - "TestMakeFoldEnginesTailHoldout"
Cohesion: 0.27
Nodes (6): DataFrame, With tail_holdout_frac=0.25, a tail engine is returned with ~25% of data., With tail_holdout_frac=0.0, no tail engine., Single symbol with tail holdout still works., Verify tail holdout engine is created when fraction > 0., TestMakeFoldEnginesTailHoldout

### Community 139 - "effective_min_profitable_symbols"
Cohesion: 0.25
Nodes (5): _debug_symbol_universe_size(), effective_min_profitable_symbols(), Active symbol count when debug scope is on; None for full-universe runs., Cap cross-symbol profitability gate to the active universe size. With…, test_effective_min_profitable_symbols_caps_debug_universe()

### Community 140 - "TestMigrationSeedFraction"
Cohesion: 0.20
Nodes (6): Verify that migrant injection uses PHASE2_MIGRATION_SEED_FRACTION, not ARCHIVE., PHASE2_MIGRATION_SEED_FRACTION defaults to 0.10., Ensure the migration fraction is not the same as archive fraction., Simulate the run_epoch migrant path and assert local_cap uses migration…, Simulate the new run_epoch logic: migrant entries are capped by migration…, TestMigrationSeedFraction

### Community 141 - "TestEvictClusterSignatures"
Cohesion: 0.14
Nodes (10): evict_cluster_signatures(), Evict JAX compiled signatures for a completed cluster. Removes entries from…, Unit tests for ``evict_cluster_signatures`` in ``_gpu_runtime.py``. These tests…, _warmup_signature() appends cluster_id to the returned tuple., evict_cluster_signatures(cluster_id=cid) removes only signatures tagged with…, evict_cluster_signatures(cluster_id=None) evicts ALL signatures., evict_cluster_signatures with a cluster_id that has no signatures returns 0 and…, Structural test: _run_cluster_islands must contain the evict_cluster_signatures… (+2 more)

### Community 142 - "test_phase2_offspring_batch.py"
Cohesion: 0.29
Nodes (5): CountingEngine, Unit tests for batched offspring evaluation (Phase 2 runtime A1)., Stub engine recording every simulate_rule_batch call's batch size., Offspring should be evaluated via ONE simulate_rule_batch call per gen, not…, test_offspring_evaluated_in_single_batch_per_gen()

### Community 143 - "trade_support_penalty"
Cohesion: 0.38
Nodes (4): Backward-compatible wrapper returning penalty only., trade_support_penalty(), Between the hard-reject floor and the soft threshold the penalty is graduated., TestTradeSupportPenalty

### Community 144 - "TestRefreshObjectivesOnResumeGate"
Cohesion: 0.23
Nodes (7): Task-10: gate cache refresh on PHASE2_PER_EPOCH_WINDOW_ROTATION. Verifies the…, Evaluate the gate expression as it appears in the source., PHASE2_PER_EPOCH_WINDOW_ROTATION=True (default): refresh = not first_epoch →…, PHASE2_PER_EPOCH_WINDOW_ROTATION=True (default): refresh = not first_epoch →…, PHASE2_PER_EPOCH_WINDOW_ROTATION=False (legacy): refresh is False even on…, PHASE2_PER_EPOCH_WINDOW_ROTATION=False (legacy): refresh is False on first…, TestRefreshObjectivesOnResumeGate

### Community 145 - "conftest.py"
Cohesion: 0.22
Nodes (10): Config, FixtureRequest, Item, _close_matplotlib_figures(), _low_memory_cleanup(), fixture, pytest_collection_modifyitems(), pytest_configure() (+2 more)

### Community 146 - "test_phase2_batch_evaluator_parity.py"
Cohesion: 0.40
Nodes (5): _backtest_df(), DataFrame, Regression tests for evaluator-facing Phase 2 chromosome semantics., Search fitness must use the same fuzzy class as RB/Phase 5 evaluation., test_batch_chromosome_signals_match_decoded_rule_conditions()

### Community 147 - ".skip_if_valid"
Cohesion: 0.31
Nodes (3): Check if output files exist and are valid. Returns ------- dict[str,…, fixture, TestSkipIfValid

### Community 148 - "_validate_schema"
Cohesion: 0.17
Nodes (5): Validate the structure of a loaded feature selection JSON. Raises ValueError if…, Load and validate a feature selection JSON file. Parameters ---------- path :…, _validate_schema(), TestLoadAndValidate, TestValidateSchema

### Community 149 - "ResearchProfile"
Cohesion: 0.28
Nodes (5): Any, Typed, versioned profile for the active research contract., Small stable surface for comparing experiments. The legacy config module…, ResearchProfile, test_research_profile_is_stable_and_versioned()

### Community 150 - "TestDeferredWarmup"
Cohesion: 0.20
Nodes (6): Unit tests for the ``defer_warmup`` flag on ``Rule_Pool_Generator``. When…, Existing callers without defer_warmup still warm at init., The configure_phase2_gpu_runtime call is inside 'if not self._defer_warmup:'…, _run_cluster_islands passes defer_warmup=True to all generators., _run_cluster_islands calls warmup_phase2_gpu_kernels per cluster., TestDeferredWarmup

### Community 151 - ".skip_if_valid"
Cohesion: 0.15
Nodes (10): Return the sidecar that binds a reusable pool to its inputs., Hash an artifact without loading it all into RAM., Load existing pool if valid, return None if missing., Atomically bind the current pool bytes to a Phase 2 input identity., Remove a stale direction cache before a fresh Phase 2 run., Return a schema-valid pool proven to match this run's inputs. Bare historical…, _resolve_pool_identity_path(), _resolve_pool_path() (+2 more)

### Community 152 - "BFS and DFS Graph Traversal"
Cohesion: 0.20
Nodes (10): CLAUDE.md Graphify Integration, Constrained Query Expansion, BFS and DFS Graph Traversal, Graph Work Memory, Existing Graph Fast Path, CLAUDE.md Graphify Integration, Constrained Query Expansion, BFS and DFS Graph Traversal (+2 more)

### Community 153 - "TestPlateauDiversityRestart"
Cohesion: 0.33
Nodes (4): Reinitialised slots have objectives=np.inf and metrics_cache={}., Even with large Pareto front, at most 10 elite are preserved., Direct unit tests for the _plateau_diversity_restart helper., TestPlateauDiversityRestart

### Community 154 - "test_nsga3_selection.py"
Cohesion: 0.40
Nodes (3): skipif, Unit tests for NSGA-III environmental selection., TestNsga3Selection

### Community 157 - "TestCausalPublicationTiming"
Cohesion: 0.09
Nodes (10): _enriched(), Regression: thresholds must never see validation-period rows., Regression: the loader's truth table must match the actual policy…, Higher-timeframe state is only published after that bar completes and aligned…, _raw_tape(), TestCausalPublicationTiming, TestInputAndBoundaryContracts, TestLoaderContext (+2 more)

### Community 158 - "TestGlobalMetricsCacheClearing"
Cohesion: 0.27
Nodes (6): When seeded_keys is empty, nothing is removed., When all keys are seeded, cache becomes empty., Verify that only seeded keys are removed from the global cache., Create a deterministic chromosome key., Cache entries matching seeded_keys are removed; non-matching survive., TestGlobalMetricsCacheClearing

### Community 165 - "test_evaluator_health.py"
Cohesion: 0.17
Nodes (7): Unit tests for ``evaluator_health_penalty`` and ``execution_ok`` (Task 4).…, Verify that evaluator health penalty is subtracted from combo score., When evaluator_health_penalty returns > 0, the combo score is lower., When evaluator health is fine, no penalty is applied., Both public functions are importable from the module., TestHealthPenaltyWiredIntoRB, TestModuleImportable

### Community 166 - "ndarray"
Cohesion: 0.11
Nodes (18): _archive_objective_vector(), attach_cv_fold_returns_batch(), _chromosome_batch(), _dominates(), _is_better_archive_entry(), _non_dominated_sort(), ndarray, Attach per-fold returns for cv_fold_min f3 on batched metrics. (+10 more)

### Community 172 - "TestHallOfFameTrim"
Cohesion: 0.25
Nodes (5): Verify that hall_of_fame is capped to PHASE2_HOF_EPOCH_CARRYOVER=10., HoF with >10 entries is trimmed to exactly 10 oldest entries., HoF with <=10 entries is left unchanged., Empty HoF does not crash during trim logic., TestHallOfFameTrim

### Community 173 - "phase2_support.py"
Cohesion: 0.03
Nodes (70): effective_min_trade_pool_floor(), effective_pool_min_val_trades(), compute_robust_score(), compute_support_penalty_and_specialist(), deployability_rank_score(), _evolution_feasibility_floors(), EvolutionFloors, _feasibility_gate_failures() (+62 more)

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

### Community 181 - "TestPerSymbolMetrics"
Cohesion: 0.40
Nodes (3): Per-symbol metrics should reflect actual trade distribution., Winning trades should produce positive net_pnl per symbol., TestPerSymbolMetrics

### Community 184 - "TestEvalCvFoldReturns"
Cohesion: 0.33
Nodes (4): Verify the helper handles None / empty fold_engines safely., fold_engines=None returns None without crashing., fold_engines=[] returns None without crashing., TestEvalCvFoldReturns

### Community 189 - "opencode.json"
Cohesion: 0.50
Nodes (3): plugin, $schema, .opencode/plugins/graphify.js

### Community 190 - "_MockEngine"
Cohesion: 0.26
Nodes (8): _MockEngine, Verify _optimize_risk with tail_holdout_engine adds tail fields to final…, When tail_holdout_engine is provided, the final history entry contains…, When tail_holdout_engine=None, NO tail fields in history., Minimal mock that mimics CPUBacktestEngine for testing _optimize_risk., TestOptimizeRiskTailHoldoutFields, _train_metrics(), _valid_metrics()

## Knowledge Gaps
- **31 isolated node(s):** `$schema`, `.opencode/plugins/graphify.js`, `URL Ingestion`, `Folder Watcher`, `Graphify MCP Server` (+26 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **8 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `CPUBacktestEngine` connect `CPUBacktestEngine` to `Rule_Pool_Generator`, `_make_engine`, `_score_metrics`, `OOS_Evaluator`, `_make_df`, `phase2_rule_pool.py`, `TestMakeFoldEnginesTailHoldout`, `_apply_monthly_admission_gate`, `trade_support_penalty`, `test_phase2_batch_evaluator_parity.py`, `test_cpu_engine_properties.py`, `rolling_cv.py`, `CandidateRecord`, `nested_walk_forward.py`, `TestCausalPublicationTiming`, `sample_df_for_phase2`, `_symbol_specialized_variants`, `gpu_engine.py`, `test_certificate_first_selection.py`, `test_jax_compat.py`, `_compute_rule_signal_mask`, `TestPerSymbolMetrics`, `TestEvalCvFoldReturns`, `GPUBacktestEngine`, `TestGPUCPUNumericalParity`, `cpu_engine.py`, `test_phase2_use_gpu_flag.py`, `_MockEngine`, `barrier_column_names`, `_apply_dynamic_rule`, `test_cpu_engine.py`, `test_gpu_engine.py`, `TestPrecomputeReleaseIndices`, `filter_migrants_for_cluster`, `test_gpu_engine_properties.py`, `Data_Loader`, `ValidationError`, `phase2_island_scheduler.py`, `baselines.py`, `_jax_compute_trade_outcomes`, `_build_entries_from_rule_set`, `_jax_compute_rule_signals`, `TestMakeWalkForwardFoldEngines`, `MonthlyWindowSummary`, `._engine`?**
  _High betweenness centrality (0.145) - this node is a cross-community bridge._
- **Why does `Reporter` connect `Reporter` to `Rule_Pool_Generator`, `TestPlotPhase2Metrics`, `TestEquityCurveDateAxis`, `OOS_Evaluator`, `TestPlotDistributionAndEquity`, `phase2_rule_pool.py`, `.run`, `._ensure_dir`, `TestWriteStrategyEvaluationTable`, `prop_settings`, `TestPlotPerRuleBreakdown`, `rolling_cv.py`, `test_reporter.py`, `Data_Loader`?**
  _High betweenness centrality (0.105) - this node is a cross-community bridge._
- **Why does `Rule_Pool_Generator` connect `Rule_Pool_Generator` to `CPUBacktestEngine`, `log_memory_rss`, `run_pipeline.py`, `.run`, `_get_dont_cares`, `compute_phase2_objectives_from_metrics`, `phase2_rule_pool.py`, `TestArchiveSaveExceptionHandling`, `TestEvictClusterSignatures`, `Pipeline_Orchestrator`, `_apply_monthly_admission_gate`, `TestRefreshObjectivesOnResumeGate`, `_make_train_df`, `TestDeferredWarmup`, `.skip_if_valid`, `Reporter`, `sample_df_for_phase2`, `ndarray`, `_derive_island_seed`, `prop_settings`, `test_phase2_rule_pool.py`, `test_phase2_use_gpu_flag.py`, `_derive_val_sample_seed`, `TestRulePoolGeneratorRun`, `test_plateau_state_leak.py`, `TestRunLogHandlerLifecycle`, `test_phase2_island_scheduler.py`, `filter_migrants_for_cluster`, `_derive_epoch_seed`, `Data_Loader`, `phase2_island_scheduler.py`, `.save_archive`, `test_crash_fix_and_run_logging.py`, `.run`, `_validate_pool_schema`, `TestValLeakGate`, `TestPoolAdmissionOverfitRatioGate`, `TestEndToEndRotation`?**
  _High betweenness centrality (0.088) - this node is a cross-community bridge._
- **Are the 65 inferred relationships involving `CPUBacktestEngine` (e.g. with `GPUBacktestEngine` and `JointPortfolioEngine`) actually correct?**
  _`CPUBacktestEngine` has 65 INFERRED edges - model-reasoned connections that need verification._
- **Are the 15 inferred relationships involving `Reporter` (e.g. with `CvFoldValEvaluator` and `Rule_Pool_Generator`) actually correct?**
  _`Reporter` has 15 INFERRED edges - model-reasoned connections that need verification._
- **Are the 62 inferred relationships involving `Rule_Pool_Generator` (e.g. with `CPUBacktestEngine` and `Reporter`) actually correct?**
  _`Rule_Pool_Generator` has 62 INFERRED edges - model-reasoned connections that need verification._
- **Are the 31 inferred relationships involving `Output_Writer` (e.g. with `OOS_Evaluator` and `_Phase5JSONEncoder`) actually correct?**
  _`Output_Writer` has 31 INFERRED edges - model-reasoned connections that need verification._