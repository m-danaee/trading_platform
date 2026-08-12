# Graph Report - trading_platform  (2026-08-10)

## Corpus Check
- 185 files · ~256,939 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 4828 nodes · 10991 edges · 192 communities (175 shown, 17 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 558 edges (avg confidence: 0.57)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `f443ff63`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- evox_runner.py
- phase2_rule_pool.py
- CPUBacktestEngine
- _make_engine
- .run
- _symbol_specialized_variants
- test_evox_runner.py
- _score_metrics
- _get_dont_cares
- _make_df
- _init_population
- run_phase2_evolution
- Output_Writer
- Data_Splitter
- Pipeline_Orchestrator
- _apply_monthly_admission_gate
- test_feature_detector_properties.py
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
- environmental_selection_nsga2
- maybe_log_generation
- Hybrid CPU and GPU execution policy
- _select_diverse_subset
- test_rb_min_symbols.py
- Graphify Pipeline
- _make_rule
- config.py
- TestEquityCurveDateAxis
- test_phase2_window_rotation.py
- research_integrity.py
- _loader_from_rows
- test_phase5_oos.py
- gpu_engine.py
- _gpu_runtime.py
- ._ensure_dir
- _derive_island_seed
- test_certificate_first_selection.py
- non_dominated_sort
- optuna_search.py
- test_phase2_init.py
- test_phase2_rule_pool_properties.py
- _compute_rule_signal_mask
- validate_config
- build_hybrid_symbol_clusters
- _build_entries_from_rule_set
- trend_context.py
- Feature_Detector
- test_reporter.py
- GPUBacktestEngine
- _build_data_matrix
- ValueError
- sample_df_for_phase2
- TestEquityCurvePlots
- TestRulePoolGeneratorRun
- test_plateau_state_leak.py
- passes_pool_admission_gate
- dashboard.py
- TestWriteSpearmanCorrelationReport
- test_data_loader_properties.py
- barrier.py
- evaluator_health_penalty
- _apply_dynamic_rule
- test_cpu_engine.py
- mandatory_context_conditions
- _m
- TestWriteStrategyEvaluationTable
- _discretize_series
- test_crash_fix_and_run_logging.py
- DataFrame
- test_feature_selector_properties.py
- test_data_loader.py
- compute_labels
- TestPlotPerRuleBreakdown
- resolve_island_hyperparams
- test_anti_overfit_config.py
- _build_target
- filter_migrants_for_cluster
- test_gpu_engine_properties.py
- ValidationError
- resolve_evolution_floors
- load_cached_split_if_fresh
- set_purged_wf_reference_rows
- run_pipeline.py
- .encode_condition
- test_output_writer_properties.py
- Data_Loader
- test_phase2_gpu_throughput.py
- .decode_chromosome
- .load_strategies
- gate_positive_good
- phase2_island_scheduler.py
- _should_inject_diversity_recovery
- _remove_low_dispersion
- prop_settings
- test_crash_fix_properties.py
- TestPlotDistributionAndEquity
- baselines.py
- Rule_Pool_Generator
- _jax_compute_trade_outcomes
- _should_post_restart_early_stop_phase2
- _build_pool_from_archive
- resolve_phase2_stage_params
- _make_df
- _validate_pool_schema
- TestHammingThresholdAutoScale
- TestValLeakGate
- DataFrame
- _jax_compute_rule_signals
- TestParetoCollapseWarningGate
- TestMakeWalkForwardFoldEngines
- _compute_stability
- ._prune_splits_after_phase1
- MonthlyWindowSummary
- ._engine
- test_rb_concentration_tail_fail_closed.py
- constrained_non_dominated_sort
- test_feature_selector.py
- .run
- stratification_scenario_strategy
- TestSpearmanSignConsistency
- _NumpyJSONEncoder
- log_memory_rss
- robust_return_pct
- _apply_colab_gpu_defaults
- splitter.py
- .get_dont_care
- OOS_Evaluator
- compute_phase2_objectives_from_metrics
- test_rb_fail_closed.py
- strategy_id
- TestMigrationSeedFraction
- _run_cluster_islands
- _sortino_ratio_from_returns
- trade_support_penalty
- TestRefreshObjectivesOnResumeGate
- conftest.py
- _derive_epoch_seed
- .skip_if_valid
- _validate_schema
- TestPermissionGatedPullback
- TestSplitRatio
- TestLoadPool
- BFS and DFS Graph Traversal
- TestPoolAdmissionOverfitRatioGate
- _base_row
- TestSparsePositiveMode
- TestZeroRatioBoundary
- TestCausalPublicationTiming
- TestGlobalMetricsCacheClearing
- TestSavePerSymbolCsv
- .load_and_validate
- _should_plateau_early_stop_phase2
- TestNaNHandling
- TestSparseSignedMode
- trim_evolution_state_memory
- TestModuleImportable
- test_phase2_rule_pool.py
- TestTriggerPerSymbol
- TestPlateauDiversityRestart
- TestContextContract
- disable_skip_optimization
- TestHallOfFameTrim
- phase2_support.py
- ConfigError
- TestContextEntryPaths
- test_gpu_engine_import_does_not_crash_on_jax_failure
- test_jax_compat.py
- _legacy_writer_contract
- _nsga3_environmental_selection
- test_config_additions.py
- TestEvalCvFoldReturns
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
1. `Reporter` - 159 edges
2. `CPUBacktestEngine` - 158 edges
3. `Rule_Pool_Generator` - 135 edges
4. `Output_Writer` - 85 edges
5. `prop_settings()` - 79 edges
6. `Pipeline_Orchestrator` - 68 edges
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

## Communities (192 total, 17 thin omitted)

### Community 0 - "evox_runner.py"
Cohesion: 0.05
Nodes (103): IslandHyperparams, Resolved Phase 2 knobs for cluster or orphan slices., _assign_eval_result(), _behavior_descriptor(), _binary_tournament_pick(), _build_diversity_reference(), _build_rank_and_crowding(), _constraint_violations() (+95 more)

### Community 1 - "phase2_rule_pool.py"
Cohesion: 0.05
Nodes (60): _archive_objective_vector(), _condition_feature_names(), _deployable_archive_pool_entries(), _entry_validation_per_symbol_metrics(), _filter_compatible_previous_pool(), _filter_pool_by_admission(), _is_better_archive_entry(), _merge_archive_entries() (+52 more)

### Community 2 - "CPUBacktestEngine"
Cohesion: 0.06
Nodes (83): CPUBacktestEngine, CPU backtest engine that exactly mirrors evaluator_v5.ipynb's…, expectancy_lcb_pct(), Return a conservative lower bound for per-trade net expectancy. Exact CPU…, _available_symbols(), _balanced_phase2_shortlist(), _cost_stress_gate(), _diversification_beam() (+75 more)

### Community 3 - "_make_engine"
Cohesion: 0.11
Nodes (17): _make_df(), _make_engine(), All TP trades → win_rate = 100%., No losing trades → profit_factor = 99.0., Build a minimal DataFrame for testing., position_notional = equity * capital_pct/100 * leverage when exposure is free., position_notional is capped when exposure is nearly full., When exposure is full, position_notional = 0. (+9 more)

### Community 4 - ".run"
Cohesion: 0.06
Nodes (35): FileHandler, _log_phase_entry(), _log_pipeline_config(), _now_iso(), _phase5_test_metrics(), DataFrame, Create a run identity and remove artifacts that cannot be trusted., Persist the final status for the current run identity. (+27 more)

### Community 5 - "_symbol_specialized_variants"
Cohesion: 0.12
Nodes (25): _attach_source_symbol_filters(), _ensure_symbol_filtered_rule(), _has_symbol_condition(), _is_symbol_condition(), Add deterministic single-condition RB candidates. Evolution is deliberately…, Island/cluster symbols carried on Phase 2 pool entries., Feature ANDs + island symbol ORs (engine ORs multiple ``symbol is X``)., Return rule with an explicit symbol filter when required. This is a safety net… (+17 more)

### Community 6 - "test_evox_runner.py"
Cohesion: 0.05
Nodes (50): _evaluate_population_indices(), _inherit_val_metrics_from_global_cache(), Phase2EvolutionState, Evaluate unevaluated individuals, preferring batch simulate_rule_batch., Copy val_* from global cache for identical chromosomes when val is skipped.…, Evolve one island epoch and return updated resumable state., Resumable NSGA-III state for symbol-island epoch scheduling., Return survivors that do not already carry a validation snapshot. Validation… (+42 more)

### Community 7 - "_score_metrics"
Cohesion: 0.07
Nodes (40): _combined_return_score(), _evaluate_ruleset(), _optimize_risk(), Return (ok, bonus, penalty) for the desired train-valid balance shape. In…, Dominant objective: return/DD with train-valid balance, plus CV-fold…, Profit objective for lenient rule addition, but now evaluator_v5 aware. A new…, _score_metrics(), _train_valid_shape() (+32 more)

### Community 8 - "_get_dont_cares"
Cohesion: 0.07
Nodes (29): _count_active_conditions(), _get_dont_cares(), _mutate(), Mutate a chromosome (returns a copy). When activating a dont_care gene, feature…, Return array of dont_care sentinels for each feature., Count active rule conditions (sparse slots or dense dont_care encoding)., sparse_to_dense(), Tests for H5/M4/M5 evolution convergence behaviors. Covers: - HoF trimming at… (+21 more)

### Community 9 - "_make_df"
Cohesion: 0.08
Nodes (22): _make_df(), _make_engine(), MonkeyPatch, Unit tests for GPUBacktestEngine. Tests verify: - JAX availability detection…, Chromosome positions must follow feature_modes insertion order., Exact CPU re-evaluation must interpret genes like the GPU path., Chromosome that matches nothing returns 0 executed trades., Zero-signal chunks should use reject metrics without scanning. (+14 more)

### Community 10 - "_init_population"
Cohesion: 0.12
Nodes (42): _random_active_class(), _crossover(), _init_population(), Initialise a population of chromosomes. *init_strategy* ``"stratified_sparse"``…, Uniform crossover (dense per-gene or sparse per-slot)., canonicalize_slots(), _clamp_slot_gene(), count_active_slots() (+34 more)

### Community 11 - "run_phase2_evolution"
Cohesion: 0.05
Nodes (43): StageLabel, Run Phase 2 NSGA-III evolution. Returns (pareto_pool, history) or state., run_phase2_evolution(), Evolutionary algorithm drivers for Phase 2., The fallback must not switch f3 from CV return to PF after gen 0., TestRunPhase2EvolutionFallback, Unit tests for Pareto-collapse warning gate (audit finding #13). AC: The…, CountingEngine (+35 more)

### Community 12 - "Output_Writer"
Cohesion: 0.11
Nodes (9): Output_Writer, Serializes RuleSet dicts to JSON and loads/validates existing JSON files.…, Verify the exact example from the spec works end-to-end., TestLoadAndValidateErrors, TestLoadAndValidateHappyPath, TestSpecExample, TestValidationErrorType, TestWriteDirectionValidation (+1 more)

### Community 13 - "Data_Splitter"
Cohesion: 0.13
Nodes (20): Data_Splitter, Chronological train/validation splitter., Module-level wrapper around ``Data_Splitter.split_and_persist``., split_and_persist(), _make_df(), _make_timestamps(), _patch_split_paths(), DataFrame (+12 more)

### Community 14 - "Pipeline_Orchestrator"
Cohesion: 0.10
Nodes (36): _context_coverage_preflight(), _context_coverage_report(), Pipeline_Orchestrator, Top-level orchestrator for the GPU-Fuzzy Trading Pipeline. Runs all five phases…, Return the active output root for this run., Temporarily rebind all cached output paths for one pipeline run., Return split-aware context coverage for both trading directions., Reject a mixed, stale, or altered enriched train/test input pair. (+28 more)

### Community 15 - "_apply_monthly_admission_gate"
Cohesion: 0.06
Nodes (29): _apply_monthly_admission_gate(), _evaluate_rule_on_window(), _monthly_window_metrics(), Evaluate a single pool rule on a single monthly window. Returns the full window…, Normalize a window evaluator result for the monthly gate. The float fallback…, Apply the monthly-window shadow-test gate to a pool of rules. Each rule is…, _DeterministicEvaluator, _DeterministicMetricsEvaluator (+21 more)

### Community 16 - "test_feature_detector_properties.py"
Cohesion: 0.09
Nodes (45): all_nan_series(), arbitrary_numeric_series(), binary_series(), large_value_series(), positive_series(), composite, DrawFn, given (+37 more)

### Community 17 - "_make_train_df"
Cohesion: 0.07
Nodes (29): _downsample_chronological(), _largest_safe_range(), Preserve per-symbol time order required by exposure/release simulation., Pick a contiguous chronological slice of *n_rows* from *df*. Critical for…, Return (start, end) of the largest contiguous bar range not in *forbidden*. The…, Sample up to *total_rows* rows, distributed equally across symbols. A single…, _sample_df(), _sort_chronological() (+21 more)

### Community 18 - "_split"
Cohesion: 0.08
Nodes (15): Helper: build df, patch paths, run split, return (train, val)., Each symbol's split point is computed from its own row count., Symbols with different sizes each get the correct floor(N * train_frac) split., All train datetimes for a symbol must be < validation datetimes (embargo gap)., Chronological ordering holds independently for each symbol., Train rows should be the first floor(N * train_frac) rows by feature_a index., No row should appear in both train and validation sets., Single row: floor(1 * train_frac) = 0, and 288-bar embargo consumes it. (+7 more)

### Community 19 - "test_cpu_engine_properties.py"
Cohesion: 0.08
Nodes (43): equity_tracking_scenario(), _expected_outcome(), fee_deduction_scenario(), _make_engine(), _make_engine_custom(), _make_minimal_df(), _make_simple_df(), multi_symbol_scenario() (+35 more)

### Community 20 - "detect_feature_mode"
Cohesion: 0.09
Nodes (19): detect_feature_mode(), Module-level convenience wrapper around Feature_Detector.detect_feature_mode., Has negative values, zero_ratio ≤ 0.3 → signed., zero_ratio == 0.3 is NOT > 0.3, so mode is signed (not sparse_signed)., Values {0, 1} match both binary and ternary criteria; binary wins., Values {-1, 0, 1} match ternary; should NOT fall through to signed., Adding value 2 to {0, 1} breaks binary → falls through to positive., Adding value 2 to {-1, 0, 1} breaks ternary → falls through to… (+11 more)

### Community 21 - "selector.py"
Cohesion: 0.08
Nodes (37): _align_feature_array(), build_phase1_shared_context(), _build_symbol_masks(), _candidate_feature_columns(), _check_spearman_sign_consistency(), _compute_chronological_stationarity_scores(), _get_spearman_folds(), _mi_scores_for_mask() (+29 more)

### Community 22 - "Feature_Selector"
Cohesion: 0.10
Nodes (20): Feature_Selector, Score and rank features separately for long and short directions., _make_train_df(), MonkeyPatch, Create a minimal training DataFrame with label columns and feature columns., Label columns must not appear in selected features., Meta columns must not appear in selected features., Loader internal columns and ``_``-prefixed names are not candidates. (+12 more)

### Community 23 - "rolling_cv.py"
Cohesion: 0.09
Nodes (39): aggregate_fold_metrics(), _bar_index_col(), _build_fold_from_ranges(), build_forbidden_ranges(), build_purged_walk_forward_folds(), cv_folds_only(), derive_primary_holdout(), FoldMetricsSummary (+31 more)

### Community 24 - "Reporter"
Cohesion: 0.10
Nodes (14): Generates visual and tabular reports for each pipeline phase. All output files…, Reporter, _make_history(), _make_per_symbol_metrics(), _make_pnl_history(), History entries with missing keys should not raise., History entries with missing keys should not raise., Create a minimal Phase 2 history list. (+6 more)

### Community 25 - "test_encoder_properties.py"
Cohesion: 0.08
Nodes (41): all_active_chromosome_strategy(), chromosome_with_dont_cares_strategy(), feature_name_strategy(), composite, DrawFn, given, ndarray, Property-based tests for gpu_fuzzy_trader.features.encoder.Encoder **Validates:… (+33 more)

### Community 26 - "write_evaluator_clean"
Cohesion: 0.08
Nodes (29): _maybe_write_evaluator_clean(), Write a stripped strategy file containing only ``direction`` and ``rules_set``.…, Write a stripped strategy file alongside the main strategy JSON. This is a…, write_evaluator_clean(), _make_rule(), minimal_strategy(), fixture, Path (+21 more)

### Community 27 - "CandidateRecord"
Cohesion: 0.14
Nodes (33): _candidate_coverage_symbols(), _candidate_positive_symbols(), CandidateRecord, _compose_ruleset(), _diversification_shortlist(), Return supported positive validation symbols for one candidate., Keep global plus score and return leaders for each positive symbol., Symbol coverage for compose diversity. Prefer **traded** symbols from backtest… (+25 more)

### Community 28 - "nested_walk_forward.py"
Cohesion: 0.13
Nodes (25): Validation helpers for monthly, nested, and multiplicity-safe research., deflated_sharpe_ratio(), estimate_pbo(), Selection-multiplicity diagnostics for strategy research artifacts., Estimate the fraction of folds where the IS winner misses OOS median. Inputs…, Return a normal-approximation deflated Sharpe probability. This is a…, Create a compact ledger-ready multiplicity report., summarize_multiplicity() (+17 more)

### Community 29 - "environmental_selection_nsga2"
Cohesion: 0.09
Nodes (21): environmental_selection_nsga2(), Canonical NSGA-II truncation on a 2N merged population., _make_chromosome(), _make_deployable_entry(), ndarray, Unit tests for elite preservation under (μ+λ) selection. Verifies that top-K…, Without preservation, champion is evicted by gen ~8 under drift., At most TOP_K slots are overwritten by elite preservation. (+13 more)

### Community 30 - "maybe_log_generation"
Cohesion: 0.09
Nodes (18): generation_log_interval(), iteration_log_interval(), log_generation(), maybe_log_generation(), log_progress.py — Throttled progress logging for long pipeline loops., Return how often to log generation progress. Uses LOG_GENERATION_INTERVAL from…, Log generation progress when the step matches the throttle interval., Log first step, last step, and every *interval* steps in between. (+10 more)

### Community 31 - "Hybrid CPU and GPU execution policy"
Cohesion: 0.07
Nodes (37): Local low-memory test policy, Virtual environment command policy, Package compile gate, Locked research dependencies, Focused low-memory test gate, Python 3.11 CI runtime, Research CI workflow, Balanced Mode-A assumption (+29 more)

### Community 32 - "_select_diverse_subset"
Cohesion: 0.13
Nodes (14): Max-min Hamming diversity sampling: greedy pick farthest from chosen. Returns…, _select_diverse_subset(), ndarray, Verify _select_diverse_subset correctness for edge cases., Create n distinct dense chromosomes., k=0 should return [] even with non-empty chromosomes., k<0 should return []., k=1 returns one chromosome. (+6 more)

### Community 33 - "test_rb_min_symbols.py"
Cohesion: 0.11
Nodes (24): _symbols_in_rules(), _dummy_df(), _make_candidate_records(), _mock_train_metrics(), _multi_symbol_rules(), _no_symbol_rule(), DataFrame, Tests for RB Governor min-distinct-symbols hard gate. After final opt_rules… (+16 more)

### Community 34 - "Graphify Pipeline"
Cohesion: 0.06
Nodes (36): Folder Watcher, URL Ingestion, Conditional Graph Exports, Graphify MCP Server, Extraction Confidence Rubric, Deterministic Full-Path Node IDs, Semantic Hyperedges, Cross-Repository Graph Merge (+28 more)

### Community 35 - "_make_rule"
Cohesion: 0.09
Nodes (13): _make_rule(), _make_rule_set(), Unit tests for gpu_fuzzy_trader.output.writer — Output_Writer Tests cover: -…, A rule with only tp non-zero should be accepted., Spot-check a variety of valid fuzzy value names., Write rule_set to a temp file and reload the raw JSON., TestWriteAllZeroRejection, TestWriteConditionValidation (+5 more)

### Community 36 - "config.py"
Cohesion: 0.07
Nodes (41): context_contract(), context_contract_digest(), _debug_symbol_universe_size(), effective_config_snapshot(), effective_min_profitable_symbols(), effective_min_trade_support(), effective_monthly_min_trades(), effective_pool_min_val_trades() (+33 more)

### Community 37 - "TestEquityCurveDateAxis"
Cohesion: 0.09
Nodes (21): _make_trade_log(), DataFrame, fixture, Unit tests for date-based x-axis in Reporter.plot_equity_curve (Task 10.3).…, When Entry_Time is absent, x-axis label is 'Trade #'., When Entry_Time is all-NaN, x-axis label is 'Trade #'., axhline is called (initial-capital line) in date mode., axhline is called (initial-capital line) in Trade # mode. (+13 more)

### Community 38 - "test_phase2_window_rotation.py"
Cohesion: 0.07
Nodes (29): Cap *total_rows* so the per-symbol request fits within the safe range. When…, _resolve_sample_total_rows(), _make_multi_sym_df(), DataFrame, fixture, Tests for per-epoch train-window rotation (task-1)., Capping logic for per-epoch window rotation., With PHASE2_PER_EPOCH_WINDOW_ROTATION=False, total_rows is unchanged. (+21 more)

### Community 39 - "research_integrity.py"
Cohesion: 0.17
Nodes (23): _canonical_json(), count_trials(), dataset_manifest(), ExperimentLedger, forward_acceptance_lock_path(), Any, Path, PathLike (+15 more)

### Community 40 - "_loader_from_rows"
Cohesion: 0.11
Nodes (16): _loader_from_rows(), _make_rows(), _make_timestamps(), The first N-288 rows (chronologically) should be kept., A row with NaN in any label column should be dropped., NaN label rows should be dropped, not filled with 0., Only specified feature_cols should be filled., Generate n evenly-spaced ISO datetime strings. (+8 more)

### Community 41 - "test_phase5_oos.py"
Cohesion: 0.11
Nodes (15): _isolate_phase5_reporter_outputs(), fixture, Unit tests for gpu_fuzzy_trader.phases.phase5_oos.OOS_Evaluator Tests cover: -…, Override module-level path dicts and return originals., Keep Reporter plots/CSVs out of the checked-in outputs directory., Write a valid selected-features JSON to path., Write a synthetic test CSV with all required columns (including feat_0..4) to a…, Give integration tests an isolated, valid enriched train/split pair. (+7 more)

### Community 42 - "gpu_engine.py"
Cohesion: 0.08
Nodes (33): _batch_metrics_from_array(), _build_event_batch(), _exposure_slot_capacity(), _fast_reject_result_rows(), _jax_compute_rule_signals_sparse_batch(), _jax_open_slot(), _jax_release_open_slots(), _jax_simulate_equity_batch() (+25 more)

### Community 43 - "_gpu_runtime.py"
Cohesion: 0.10
Nodes (32): phase2_should_enrich_symbol_metrics(), Return True when GPU batch eval should run a follow-up CPU enrichment pass.…, configure_phase2_gpu_runtime(), detect_gpu_memory_used_gb(), detect_gpu_vram_gb(), detect_system_ram_gb(), _iter_warmup_targets(), log_gpu_runtime_config() (+24 more)

### Community 44 - "._ensure_dir"
Cohesion: 0.07
Nodes (19): Reporting and visualization sub-package., _bucket_series_by_mode(), DataFrame, Series, reporter.py — Reporting and visualization for the GPU-Fuzzy Trading Pipeline.…, Write feature-stratified performance metrics to CSV files (one per split). For…, Create parent directories for *path* if they do not exist., Compute maximum drawdown percentage from an equity series. Parameters… (+11 more)

### Community 45 - "_derive_island_seed"
Cohesion: 0.06
Nodes (41): context_coverage_for_direction(), context_coverage_report(), context_floor_failures(), Any, DataFrame, Shared diagnostics for the mandatory direction-specific context contract., Return coverage diagnostics for every named frame and direction., Return mathematically impossible trade-floor failures for coverage. (+33 more)

### Community 46 - "test_certificate_first_selection.py"
Cohesion: 0.20
Nodes (14): _portfolio_selection_certificate(), Return the certificate used by compose, risk, and profit selection., _BatchEngine, _candidate(), _metrics(), _pool_entry(), Focused regressions for certificate-first RB and Phase 2 selection., test_certificate_rejects_eth_only_and_accepts_balanced_team() (+6 more)

### Community 47 - "non_dominated_sort"
Cohesion: 0.13
Nodes (26): batch_hamming_min(), crowding_distance(), _crowding_distance_numba(), _crowding_distance_py(), _disable_numba_sort(), _dominates_numba(), _dominates_py(), non_dominated_sort() (+18 more)

### Community 48 - "optuna_search.py"
Cohesion: 0.10
Nodes (29): _active_search_space(), apply_trial_config(), collect_validation_metrics(), compute_score(), _copy_phase1_2_outputs(), main(), objective(), _pearson() (+21 more)

### Community 49 - "test_phase2_init.py"
Cohesion: 0.14
Nodes (20): assign_strata_to_indices(), build_feature_sampling_probs(), pick_active_count(), _pick_active_index(), _pick_inactive_index(), ndarray, phase2_init.py — Sparsity-guided stratified population initialization for Phase…, Assign elite / explorer labels to non-seeded population rows. (+12 more)

### Community 50 - "test_phase2_rule_pool_properties.py"
Cohesion: 0.11
Nodes (21): feature_infos_and_train_df(), _isolate_phase2_archive_paths(), _make_feature_infos(), _make_train_df(), composite, DataFrame, DrawFn, fixture (+13 more)

### Community 51 - "_compute_rule_signal_mask"
Cohesion: 0.42
Nodes (3): _compute_rule_signal_mask(), Build one boolean signal mask (evaluator_v5 parity). Feature conditions are…, TestSymbolFilterSignalMask

### Community 52 - "validate_config"
Cohesion: 0.18
Nodes (19): Validate all high-impact hyperparameter relationships. The function is…, Write the effective configuration snapshot and return its path., validate_config(), write_config_audit_report(), MonkeyPatch, Path, Cross-parameter configuration and evaluator-parity tests., test_audit_report_writes_the_effective_snapshot() (+11 more)

### Community 53 - "build_hybrid_symbol_clusters"
Cohesion: 0.12
Nodes (28): build_hybrid_symbol_clusters(), _corr_embedding_block(), _feature_names_union(), _feature_profile_block(), Any, DataFrame, ndarray, symbol_cluster.py — Per-symbol clustering for Phase 2 island scheduling using… (+20 more)

### Community 54 - "_build_entries_from_rule_set"
Cohesion: 0.18
Nodes (8): _build_entries_from_rule_set(), Sort entries for v5 capital allocation (timestamp, rule, symbol, row)., Priority-based rule assignment: first matching rule wins per row. Mirrors…, _sort_entries_by_allocation_priority(), Row matching both rules should be assigned to rule 1 only., Rows not matching rule 1 should be assigned to rule 2., v5: earlier JSON rule wins over lower dataset row index at same time., TestBuildEntriesFromRuleSet

### Community 55 - "trend_context.py"
Cohesion: 0.08
Nodes (49): Number of leading per-symbol rows belonging to the training prefix. Shared by…, train_prefix_row_count(), align_completed_states_to_rows(), average_true_range(), build_higher_bars(), build_manifest(), build_train_prefix(), _classify_hf_bars() (+41 more)

### Community 56 - "Feature_Detector"
Cohesion: 0.13
Nodes (12): detect_all_modes(), Feature_Detector, DataFrame, Series, detector.py — Feature_Detector Classifies each feature column into exactly one…, Classify feature columns by their discretization type., Classify a single feature series into one of six modes. Detection order…, Classify every column in *feature_cols* and return a mapping. Parameters… (+4 more)

### Community 57 - "test_reporter.py"
Cohesion: 0.15
Nodes (16): _make_selected_features(), _make_stratified_dataset(), _make_stratified_datasets_by_split(), _make_stratified_logs_by_split(), _make_stratified_trade_log(), Unit tests for gpu_fuzzy_trader.reporting.reporter.Reporter Tests cover: -…, Create a dataset with fuzzy-valued feature columns., Create a trade log with Entry_Index values within dataset bounds. (+8 more)

### Community 58 - "GPUBacktestEngine"
Cohesion: 0.10
Nodes (16): GPUBacktestEngine, CPU engine for rule-set simulation used by Phase 2/RB., Rule-set sim using a mask cache (delegates to the CPU engine)., Evaluate multiple rule sets (parallel CPU; optional mask cache)., RB JAX path: cached mask entry build + parallel CPU equity. Equity results…, Same interface as CPUBacktestEngine for compatibility., JAX-accelerated backtest engine for Phase 2 rule pool generation.…, Return the JAX backend in use ('gpu', 'cpu', or 'tpu'). (+8 more)

### Community 59 - "_build_data_matrix"
Cohesion: 0.08
Nodes (22): _build_data_matrix(), DataFrame, Build an (N, K) integer matrix of discretized feature values., Raise RuntimeError if JAX failed to import at module level., _require_jax(), DataFrame, ndarray, simulate_rule_set (delegated to CPU) is exactly identical to CPU engine. (+14 more)

### Community 60 - "ValueError"
Cohesion: 0.04
Nodes (58): conditions_cache_key(), get_or_build_rule_mask(), ndarray, condition_cache.py — Cached boolean masks for textual rule conditions. Avoids…, Stable hashable key for a list of condition strings., Return a boolean row mask for *conditions*, using *cache* when provided. The…, _append_allocated_entries(), _batch_eval_rule_set_pickled() (+50 more)

### Community 61 - "sample_df_for_phase2"
Cohesion: 0.12
Nodes (14): DataFrame, Build train/val backtest engines., Build the selected Phase 2 backend for the sampled train frame., Return whether this generator should avoid allocating a JAX engine.…, Build an engine on *df* using the same backend selection logic., Resolve Phase 2 row budget then sample with aligned symbol windows., sample_df_for_phase2(), _minimal_backtest_df() (+6 more)

### Community 62 - "TestEquityCurvePlots"
Cohesion: 0.13
Nodes (13): Build a flat list of per-symbol metric dicts for CSV output. Uses the…, DataFrame, Verify plot_equity_curve is called for all three splits and handles empty logs., Helper: run OOS_Evaluator.run() with mocked internals and capture plot calls., plot_equity_curve must be called with 'train', 'validation', and 'test'., plot_equity_curve must be called exactly three times (one per split)., Empty train trade log must not raise an exception., Empty validation trade log must not raise an exception. (+5 more)

### Community 63 - "TestRulePoolGeneratorRun"
Cohesion: 0.09
Nodes (12): Integration tests using tiny population and generation counts., In holdout mode, val engine must be built for pool admission even when…, All pool entries must have active conditions within [MIN_CONDITIONS,…, Phase 2 must use static TP=PHASE2_TP, SL=PHASE2_SL,…, Pool entries must have executed_trades >= MIN_TRADE_POOL_FLOOR., Verify that Rule_Pool_Generator's persistent self._rng advances across multiple…, After two run_epoch() calls, the internal RNG state must differ from the…, The RNG should produce *different* sequences in two consecutive run_epoch()… (+4 more)

### Community 64 - "test_plateau_state_leak.py"
Cohesion: 0.11
Nodes (21): _make_minimal_gen(), _mock_evolution_state(), _mock_stage_plan(), Regression tests for plateau-state leak fixes (Fixes A + B). Fix A:…, AC-1, AC-3: plateau_streak resets per epoch; reset_plateau=True always., reset_plateau=True is passed on the very first epoch call., reset_plateau=True is also passed on epoch 2 (regression for leak)., Even when PHASE2_ISLAND_TWO_STAGE_ENABLED=False, reset_plateau=True. (+13 more)

### Community 65 - "passes_pool_admission_gate"
Cohesion: 0.08
Nodes (19): passes_pool_admission_gate(), Hard gate for Phase 2 pool/archive on merged holdout metrics. When…, fixture, MonkeyPatch, Tests for _feasibility_gate_failures — per-gate breakdown., A rule that should pass all 9 gates., A rule with too few train trades., A rule passing all gates returns all-zero dict. (+11 more)

### Community 66 - "dashboard.py"
Cohesion: 0.15
Nodes (25): build_dashboard_data(), _direction_data(), _history_rows(), _load_json(), main(), Any, Path, Read-only HTML dashboard for existing pipeline artifacts. The dashboard… (+17 more)

### Community 67 - "TestWriteSpearmanCorrelationReport"
Cohesion: 0.13
Nodes (10): _make_dataset_with_label(), _make_datasets_by_split(), Feature not in dataset → NaN for that split., Dataset without label_close_288 → NaN for all features on that split., None dataset for a split → NaN for all features on that split., Empty selected_features → CSV with header only., All non-NaN Spearman values must be in [-1.0, 1.0]., Create a dataset DataFrame with feature columns and label_close_288. (+2 more)

### Community 68 - "test_data_loader_properties.py"
Cohesion: 0.15
Nodes (26): dataframe_with_nan_features(), dataframe_with_nan_labels(), _load_from_df(), _make_datetime_series(), composite, DataFrame, DrawFn, given (+18 more)

### Community 69 - "barrier.py"
Cohesion: 0.19
Nodes (16): attach_barrier_outcomes(), barrier_column_names(), configured_barrier_pairs(), _first_touch_for_symbol(), _first_touch_for_symbol_numba(), _first_touch_for_symbol_python(), _number_token(), DataFrame (+8 more)

### Community 70 - "evaluator_health_penalty"
Cohesion: 0.09
Nodes (20): evaluator_health_penalty(), evaluator_health.py — Pure functions for evaluator-failure-mode awareness.…, Read a numeric metric, returning *default* for missing / None / NaN / Inf., Read an integer metric safely., Penalty for evaluator_v5 failure modes — higher is worse. Parameters ----------…, _safe_float(), _safe_int(), Scoring helpers shared across pipeline phases. Re-exports… (+12 more)

### Community 71 - "_apply_dynamic_rule"
Cohesion: 0.21
Nodes (4): _apply_dynamic_rule(), Apply one exported text condition using the original threshold logic. Exactly…, Test all threshold branches of _apply_dynamic_rule., TestApplyDynamicRule

### Community 72 - "test_cpu_engine.py"
Cohesion: 0.11
Nodes (15): _safe_profit_factor(), JointPortfolioEngine, DataFrame, Joint long/short portfolio simulation. Phase 2 and RB score each direction…, Evaluate long and short rule books in one net-position account., get_normalized_symbol_array(), DataFrame, ndarray (+7 more)

### Community 73 - "mandatory_context_conditions"
Cohesion: 0.12
Nodes (22): mandatory_context_conditions(), Return the fixed, mandatory context conditions for *direction*., _assert_capital_budget(), _assert_mandatory_context(), _enforce_capital_budget(), Path, Persist an explicit empty strategy and diagnostic report., Normalize rule capital_pct so sum <= RB_MAX_TOTAL_CAPITAL. (+14 more)

### Community 74 - "_m"
Cohesion: 0.09
Nodes (21): execution_ok(), Return ``True`` iff the evaluator would execute this rule set reasonably. A…, _finite_number(), _metric_int(), Shared metric gates used by Phase 2 admission and RB Governor. The gate is…, _m(), Unit tests for ``evaluator_health_penalty`` and ``execution_ok`` (Task 4).…, Tests for ``execution_ok``. (+13 more)

### Community 75 - "TestWriteStrategyEvaluationTable"
Cohesion: 0.21
Nodes (9): _make_eval_rule_set(), _make_metrics_by_split(), _make_trade_logs_by_split(), Create a minimal rule set for evaluation table tests., Create a metrics_by_split dict with all three splits populated., Create a trade_logs_by_split dict with all three splits populated., Sharpe ratio should equal mean(r)/std(r, ddof=1) for a known log., win_rate, mdd_pct etc. should come from metrics_by_split. (+1 more)

### Community 76 - "_discretize_series"
Cohesion: 0.33
Nodes (4): _discretize_series(), Series, Discretize a feature series using evaluator_v5's fixed fuzzy bins. Exported…, TestDiscretizeSeries

### Community 77 - "test_crash_fix_and_run_logging.py"
Cohesion: 0.06
Nodes (32): _capture_logs(), _make_feature_infos_crash(), _make_train_df_crash(), DataFrame, MonkeyPatch, Unit tests for crash-fix-and-run-logging spec. Covers: - Task 1.1: Smoke tests…, Requirement 3.1 — save_archive is called before _release_resources in run()., Mock save_archive and _release_resources, run Rule_Pool_Generator.run() with a… (+24 more)

### Community 78 - "DataFrame"
Cohesion: 0.13
Nodes (10): DataFrame, Winning trades should produce positive net_pnl per symbol., Release index should point to the row where bar_index + 288 is reached., Rows near the end should get release_index = len(df)., Each symbol's release indices are computed independently., Simulate catastrophic losses to trigger account ruin. With min_288=0…, Per-symbol metrics should reflect actual trade distribution., TestAccountRuin (+2 more)

### Community 79 - "test_feature_selector_properties.py"
Cohesion: 0.15
Nodes (23): dataset_with_high_dispersion_feature(), dataset_with_label_and_meta_columns(), dataset_with_sufficient_dispersion_feature(), _make_label_block(), composite, DataFrame, DrawFn, given (+15 more)

### Community 80 - "test_data_loader.py"
Cohesion: 0.17
Nodes (11): load_dataset(), Module-level wrapper around ``Data_Loader.load_dataset``., _make_csv(), _make_ohlcv_rows(), DataFrame, Unit tests for gpu_fuzzy_trader.data.loader.Data_Loader Tests cover: - CSV…, Module-level function should produce same result as class method., Generate raw OHLCV rows without precomputed forward labels. (+3 more)

### Community 81 - "compute_labels"
Cohesion: 0.20
Nodes (10): compute_labels(), DataFrame, Forward-window label computation for OHLCV bars. Horizon is ``TAIL_DROP_ROWS``…, Compute the 5 label columns per symbol. - label_open_next: open[t+1] -…, DataFrame, ndarray, Unit tests for forward-window label semantics…, Regression: old backward rolling must not match corrected forward labels. (+2 more)

### Community 82 - "TestPlotPerRuleBreakdown"
Cohesion: 0.16
Nodes (11): _make_full_trade_log(), _make_rule_set(), _make_split_logs(), Create a trade log with all columns needed for sharpe computation., Create a minimal rule set with *n* rules., Create a trade log with Rule_Index, Net_PnL, Equity_After columns., Create trade_logs_by_split dict with all three splits populated., One split None, others valid — should not raise. (+3 more)

### Community 83 - "resolve_island_hyperparams"
Cohesion: 0.17
Nodes (15): Scale integer trade floors by slice size vs full-universe reference., Resolve scaled trade floors and relaxed cross-symbol gates., resolve_island_hyperparams(), scale_trade_floor_by_universe(), parametrize, Unit tests for island hyperparameter scaling., Verify cluster islands scale min_profitable = max(2, ceil-half)., The config constant PHASE2_MIN_PROFITABLE_SYMBOLS is the upper bound. (+7 more)

### Community 84 - "test_anti_overfit_config.py"
Cohesion: 0.29
Nodes (6): effective_phase2_val_return_floor_pct(), Direction-aware Phase 2 validation return floor for fitness penalties., Regression tests for anti-overfit / exploration-retune config bundle., test_anti_overfit_config_bundle(), test_cluster_island_symbol_robustness_enabled(), test_one_symbol_island_hyperparams_target_one_profitable()

### Community 85 - "_build_target"
Cohesion: 0.17
Nodes (12): _build_target(), Build a direction-specific target signal. Default…, Encoding-aware win class: 2 in asymmetric mode, 1 in legacy mode., Encoding-aware loss class: 0 in either mode., Long: max >= entry*(1+TP/100), min > entry*(1-SL/100) → success., Long: both hit but max_before_min==0 → SL first → failure., Long: both hit but max_before_min==1 → TP first → success., Long: neither TP nor SL hit → failure (or neutral in asymmetric mode). (+4 more)

### Community 86 - "filter_migrants_for_cluster"
Cohesion: 0.10
Nodes (21): extract_deployable_migrants(), Return elite deployable-preview entries suitable for guarded migration., _exchange_migrants_between_islands(), filter_migrants_for_cluster(), Accept only migrants that pass deployability on the receiver cluster slice., Perform a guarded, order-independent migration exchange. Islands are processed…, _make_migrant_dict(), _make_mock_receiver() (+13 more)

### Community 87 - "test_gpu_engine_properties.py"
Cohesion: 0.16
Nodes (17): _assert_parity(), _make_engines(), _make_parity_df(), parity_scenario_strategy(), composite, DataFrame, DrawFn, given (+9 more)

### Community 88 - "ValidationError"
Cohesion: 0.09
Nodes (27): context_permission_column(), context_trigger_column(), Return the direction-specific permission column name., Return the direction-specific LWC pullback-reversal trigger column., _condition_feature(), _context_feature_direction(), Exception, Path (+19 more)

### Community 89 - "resolve_evolution_floors"
Cohesion: 0.31
Nodes (6): EvolutionFloors, Resolved evolution-time floors (pool admission gates remain strict)., Return stage-aware fitness floors; defaults to global strict knobs. When both…, resolve_evolution_floors(), Stage A soft floors must survive island_hyperparams (cluster two-stage)., TestResolveEvolutionFloorsIslandTwoStage

### Community 90 - "load_cached_split_if_fresh"
Cohesion: 0.31
Nodes (6): load_cached_split_if_fresh(), Load cached split parquets when they are newer than the source CSV. Validates…, load_cv_folds_manifest(), Any, Load manifest if present., TestLoadCachedSplitIfFresh

### Community 91 - "set_purged_wf_reference_rows"
Cohesion: 0.18
Nodes (7): Store full train_new.csv row count after loader prep (split time)., set_purged_wf_reference_rows(), holdout_mode(), purged_mode(), fixture, Tests for purged-WF trade-floor scaling helpers., TestScaleTradeFloor

### Community 92 - "run_pipeline.py"
Cohesion: 0.09
Nodes (34): downcast_numeric_df(), prune_train_columns(), DataFrame, df_slim.py — Reduce DataFrame memory for backtest engines. Keeps only columns…, Downcast label and feature columns to float32; symbol to category (inplace)., Return a copy with only meta, label, and requested feature columns. Parameters…, Drop unused feature columns from the full training split after Phase 1., slim_backtest_df() (+26 more)

### Community 93 - ".encode_condition"
Cohesion: 0.07
Nodes (25): ConfigurationError, decode_chromosome(), encode_condition(), Encoder, get_dont_care(), Exception, ndarray, encoder.py — Encoder Maps gene integer values to fuzzy value names, formats… (+17 more)

### Community 94 - "test_output_writer_properties.py"
Cohesion: 0.11
Nodes (29): normalize_symbol_value(), parse_symbol_condition(), symbol_conditions.py — Symbol filter parsing (evaluator_v5 parity). Feature…, Normalize symbol values so strategy conditions such as: "symbol is 1" "symbol…, Parse optional symbol filters. Supported formats: "symbol is 1" "symbol IS 1"…, all_zero_rule_st(), _is_valid_exported_condition(), oversized_rule_set_st() (+21 more)

### Community 95 - "Data_Loader"
Cohesion: 0.11
Nodes (14): Data_Loader, Stateless data loader for the GPU-Fuzzy Trading Pipeline., _enriched(), Regression: thresholds must never see validation-period rows., Regression: the loader's truth table must match the actual policy…, _raw_tape(), TestContextIdentity, TestIncompleteBoundaryBars (+6 more)

### Community 96 - "test_phase2_gpu_throughput.py"
Cohesion: 0.15
Nodes (17): _jax_compute_rule_signals_batch(), Batch rule matching for B chromosomes simultaneously. Returns (B, N) boolean…, CPU and GPU backtest engine sub-package., get_gpu_backtest_engine_class(), jax_gpu_backtest_available(), Any, Detect whether JAX / GPUBacktestEngine can be loaded on this host. JAX can fail…, Return ``GPUBacktestEngine`` if ``gpu_engine`` imports cleanly; else ``None``. (+9 more)

### Community 97 - ".decode_chromosome"
Cohesion: 0.26
Nodes (4): See module-level :func:`decode_chromosome`., All 10 signed values decode correctly via decode_chromosome., Chromosome with numpy int32/int64 values should work., TestDecodeChromosome

### Community 98 - ".load_strategies"
Cohesion: 0.19
Nodes (6): Load long.json and short.json via Output_Writer.load_and_validate(). Parameters…, Return only non-empty RB strategies accepted for deployment., Write a valid rule set JSON to path., Override module-level path dicts and return originals (for standalone tests)., TestLoadStrategies, _write_rule_set()

### Community 99 - "gate_positive_good"
Cohesion: 0.19
Nodes (18): gate_positive_good(), positive_good_reject_reasons(), PositiveGoodThresholds, Minimum evidence required on train and validation splits., Return stable, machine-readable reasons for a gate rejection., Return whether both splits satisfy one explicit evidence contract.…, Verify that ``gate_positive_good`` calls ``execution_ok`` when flagged., When require_execution_health=True and execution_ok fails, gate returns False. (+10 more)

### Community 100 - "phase2_island_scheduler.py"
Cohesion: 0.16
Nodes (20): filter_df_to_symbols(), DataFrame, Return rows for the given symbols; raises if column missing or no rows., persist_symbol_clusters(), _context_support_preflight(), _load_phase1_feature_lists(), Any, DataFrame (+12 more)

### Community 101 - "_should_inject_diversity_recovery"
Cohesion: 0.14
Nodes (9): _diversity_recovery_min_unique_ratio(), _should_inject_diversity_recovery(), Tiny deployable archive must not IndexError on viability recovery., Test that Check 3 fires when pareto_size=3, plateau_streak=2, pop_size=100. For…, Check 3 requires plateau_streak >= 2 (isolated with pop_size=100)., For pop_size=100, Check 2 threshold=2, so pareto_size=4 should NOT trigger., Check 3 respects PHASE2_DIVERSITY_RECOVERY_ENABLED., TestPopulationDiversityMetrics (+1 more)

### Community 102 - "_remove_low_dispersion"
Cohesion: 0.17
Nodes (10): Remove features where more than `threshold` fraction of values are identical.…, _remove_low_dispersion(), DataFrame, Exactly 95% identical → NOT > 0.95 → keep., 96% identical → > 0.95 → remove., Two features with corr > 0.95 → keep higher-scored one., Two uncorrelated features → both kept., Features in different modes are not compared for redundancy. (+2 more)

### Community 103 - "prop_settings"
Cohesion: 0.07
Nodes (43): HealthCheck, prop_settings(), Shared Hypothesis settings for property tests. Set PYTEST_LOW_MEMORY=1 to scale…, Hypothesis settings with optional low-memory example scaling., _make_timestamps(), composite, DataFrame, DrawFn (+35 more)

### Community 104 - "test_crash_fix_properties.py"
Cohesion: 0.10
Nodes (26): _append_xla_flag(), configure_jax_env(), _cuda_package_root(), Path, JAX/XLA runtime configuration — call before the first ``import jax``., Find a pip-installed CUDA toolkit root, if one is available., Append an XLA flag once, preserving explicit user configuration., Configure JAX/XLA runtime for predictable desktop-friendly GPU usage. -… (+18 more)

### Community 105 - "TestPlotDistributionAndEquity"
Cohesion: 0.16
Nodes (6): _make_dist_logs_by_split(), _make_dist_trade_log(), Create a trade log with all columns needed for distribution/equity plots., Create trade_logs_by_split dict with all three splits populated., A trade log with a single trade should not raise., TestPlotDistributionAndEquity

### Community 106 - "baselines.py"
Cohesion: 0.24
Nodes (17): compute_entry_time_priority(), Map each row to a timestamp priority code (evaluator_v5 parity)., _compact(), _drop_one_condition_variants(), _equal_weight(), _evaluate(), evaluate_baselines(), _feature_shuffle() (+9 more)

### Community 107 - "Rule_Pool_Generator"
Cohesion: 0.08
Nodes (20): _archive_feature_signature(), Return the ordered feature signature used to validate archive reuse., Read JSON from *path* and return None when the file cannot be loaded., Validate the archive JSON structure and feature compatibility., Phase 2: GPU-accelerated multi-objective evolutionary rule pool generation.…, Inject guarded migration seeds for the next epoch., Restore slimmed training data from cache (no re-sampling needed)., Attach optional island metadata; safe when *owner* is a partial mock. (+12 more)

### Community 108 - "_jax_compute_trade_outcomes"
Cohesion: 0.24
Nodes (5): _jax_compute_trade_outcomes(), JAX-jitted vectorized trade outcome computation., Vectorized trade outcome computation for all rows. Mirrors…, Multiple rows processed simultaneously., TestComputeTradeOutcomes

### Community 109 - "_should_post_restart_early_stop_phase2"
Cohesion: 0.29
Nodes (9): Break the epoch when a plateau restart yields no improvement. Independent of…, _should_post_restart_early_stop_phase2(), Unit tests for post-restart no-improvement early stop (Phase 2 runtime)., test_global_disabled_no_stop(), test_global_uses_global_knobs(), test_island_disabled_no_stop(), test_island_streak_at_patience_stops(), test_island_streak_below_patience_no_stop() (+1 more)

### Community 110 - "_build_pool_from_archive"
Cohesion: 0.10
Nodes (27): _migrant_to_metrics(), Backtest one migrant chromosome on receiver cluster engines., _archive_direction(), attach_cv_fold_returns_batch(), _build_cpu_archive_engine(), _build_pool_from_archive(), _chromosome_batch(), _chromosome_for_pool_export() (+19 more)

### Community 111 - "resolve_phase2_stage_params"
Cohesion: 0.08
Nodes (24): Stop burning gens when the feasible set is empty and restarts are spent. Sparse…, True when Stage A viability is critically low and search has plateaued., _should_abort_zero_deployable_collapse(), _should_viability_recovery(), island_stage_budgets(), IslandStagePlan, StageLabel, Phase 2 two-stage search hyperparameter profiles (exploration vs refinement). (+16 more)

### Community 112 - "_make_df"
Cohesion: 0.25
Nodes (8): _make_df(), _make_rule_set(), When no trades are executed, account_ruined must be False., When no trades are executed, total_return_pct must be 0.0., Create a minimal valid rule set dict., Create a minimal DataFrame with all required columns., Returned and saved OOS metrics come from the locked strategy., TestEvaluateStrategy

### Community 113 - "_validate_pool_schema"
Cohesion: 0.36
Nodes (3): Validate the structure of a loaded pool JSON. Raises ValueError if the schema…, _validate_pool_schema(), TestValidatePoolSchema

### Community 114 - "TestHammingThresholdAutoScale"
Cohesion: 0.15
Nodes (10): Verify the max(3, k_active // 5) formula. The formula is applied in…, Replicate the auto-scaling formula., k_active=0 → threshold = max(3, 0//5) = 3., k_active=5 → threshold = max(3, 5//5=1) = 3., k_active=15 → threshold = max(3, 15//5=3) = 3., k_active=20 → threshold = max(3, 20//5=4) = 4., k_active=50 → threshold = max(3, 50//5=10) = 10., k_active=100 → threshold = max(3, 100//5=20) = 20. (+2 more)

### Community 115 - "TestValLeakGate"
Cohesion: 0.20
Nodes (10): C6: Val-derived penalties must be gated behind JOINT_TRAIN_VAL or…, Return standard monkeypatching for clean baseline metrics., Apply base settings with optional overrides., Metrics that trigger no train-side penalties., Val metrics that WOULD trigger penalties if the gate were open., When both JOINT_TRAIN_VAL and VAL_IN_FITNESS_PENALTY are False, val-derived…, Bad val must not set feasibility_violation when gate is closed., When VAL_IN_FITNESS_PENALTY=True, val-derived penalties DO enter objectives. (+2 more)

### Community 116 - "DataFrame"
Cohesion: 0.16
Nodes (6): _make_trade_log(), DataFrame, Dataset with only 1 non-NaN paired row → NaN., Rows must be sorted by abs(train_spearman) descending., Create a minimal trade log DataFrame with Equity_After column., TestPlotEquityCurve

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

### Community 121 - "._prune_splits_after_phase1"
Cohesion: 0.20
Nodes (5): Selected fuzzy features for Phase 2., Drop unused feature columns from train/val splits to reduce RAM., Drop unused feature columns from CV fold DataFrames after Phase 1., Drop unused feature columns from train split (legacy single-split API)., TestPruneSplitsAfterPhase1

### Community 122 - "MonthlyWindowSummary"
Cohesion: 0.07
Nodes (38): build_monthly_windows(), _datetime_series(), evaluate_rule_set_monthly(), monthly_penalty(), monthly_return_counts_as_good(), MonthlyWindowSummary, DataFrame, Series (+30 more)

### Community 123 - "._engine"
Cohesion: 0.22
Nodes (4): Test _build_trade_outcome_single for long direction., Test _build_trade_outcome_single for short direction., TestTradeOutcomeLong, TestTradeOutcomeShort

### Community 124 - "test_rb_concentration_tail_fail_closed.py"
Cohesion: 0.26
Nodes (9): _candidates(), _dummy_df(), _mock_metrics(), DataFrame, Tests for RB concentration / tail-holdout hard fail-closed behaviour. When…, Return/PF below gate but sym+tail OK → rules retained, not accepted., _rule(), _run_pipeline() (+1 more)

### Community 125 - "constrained_non_dominated_sort"
Cohesion: 0.23
Nodes (13): _clean_violation(), constrained_dominates(), constrained_non_dominated_sort(), _pareto_dominates(), ndarray, Constraint-aware Pareto ordering for Phase 2 evolution. Objectives alone are…, Return whether *left* Pareto-dominates *right* (minimisation)., Return whether the left candidate dominates the right candidate. (+5 more)

### Community 126 - "test_feature_selector.py"
Cohesion: 0.18
Nodes (8): _mutual_info_discrete_mask(), Return the set of feature names that PASS both stationarity checks. A feature…, Per-column ``discrete_features`` flags for ``mutual_info_classif``. Only…, _stationarity_filter(), Unit tests for gpu_fuzzy_trader.features.selector.Feature_Selector Tests cover:…, TestMutualInfoDiscreteMask, TestReduceOverlap, TestStationarityFilter

### Community 127 - ".run"
Cohesion: 0.11
Nodes (12): DataFrame, Run out-of-sample evaluation. Parameters ---------- allowed_directions :…, Prepare test data using Data_Loader.load_dataset(). Applies the same…, Load prepared train, validation, and test datasets., Load and validate a strictly newer, untouched forward period., Load selected features for a direction when available., Remove only known Phase 5 artifacts from the active report root., Evaluate a single strategy on the test DataFrame. Returns ------- metrics :… (+4 more)

### Community 128 - "stratification_scenario_strategy"
Cohesion: 0.24
Nodes (15): dataset_with_features_strategy(), metrics_strategy(), composite, DataFrame, DrawFn, Generate a trade log DataFrame with 2–50 rows. Parameters ---------- n_rules:…, Generate a metrics dict with reasonable float values. Keys: win_rate,…, Generate a dataset DataFrame with 10–100 rows. Parameters ----------… (+7 more)

### Community 129 - "TestSpearmanSignConsistency"
Cohesion: 0.18
Nodes (9): ndarray, Tests for _check_spearman_sign_consistency, including the val_df check., Build a minimal DataFrame without 'symbol' to avoid symbol-based folding., AC1: Train all positive, val negative → feature blacklisted., AC2: Train all positive, val positive → feature still kept., AC3: Train all positive, tiny |val_rho| < min_abs_corr → feature still kept., AC4: val_df=None → pre-task-8 behavior (mixed train signs still blacklisted)., AC5: val_df without label_close_288 column → val check skipped (no crash). (+1 more)

### Community 130 - "_NumpyJSONEncoder"
Cohesion: 0.10
Nodes (16): __main__.py — Entry point for `python -m gpu_fuzzy_trader.run_pipeline` Allows…, Any, Typed, versioned profile for the active research contract., Small stable surface for comparing experiments. The legacy config module…, ResearchProfile, main(), _NumpyJSONEncoder, _print_run_summary() (+8 more)

### Community 131 - "log_memory_rss"
Cohesion: 0.19
Nodes (11): log_memory_rss(), Optional process memory logging (set LOG_MEMORY=1)., Log current process RSS in MiB when LOG_MEMORY=1., Force GC and clear JAX compilation caches between Phase 2 directions. Releases:…, release_phase2_resources(), Release GPU engines and slim in-memory data between island epochs., Drop engine and sampled data to free RAM before the next direction., Tests for optional memory logging helpers. (+3 more)

### Community 132 - "robust_return_pct"
Cohesion: 0.27
Nodes (5): feasibility_violation_score(), Conservative return used for objectives, plateau, and archive ranking., Non-negative violation score; 0 means the rule meets deployability floors. Used…, robust_return_pct(), TestDeployabilityHelpers

### Community 133 - "_apply_colab_gpu_defaults"
Cohesion: 0.33
Nodes (6): _apply_colab_gpu_defaults(), is_colab_runtime(), True when running on Google Colab (/content runtime)., Colab T4 optimization for Phase 2 runs., MonkeyPatch, test_colab_defaults_apply_when_content_exists()

### Community 134 - "splitter.py"
Cohesion: 0.24
Nodes (12): _chronological_half_split(), _holdout_embargo_split(), _purged_walk_forward_split(), DataFrame, data/splitter.py — Data_Splitter Per-symbol chronological split for Phase 2,…, Per-symbol chronological first or second half of *df*. ``purge_rows`` is…, Split validation into purged fitness and selection halves per symbol. The gap…, Return whether cached internal halves match the purged geometry. (+4 more)

### Community 136 - "OOS_Evaluator"
Cohesion: 0.27
Nodes (4): OOS_Evaluator, Out-of-sample evaluator for Phase 5. Loads the final long/short strategies,…, TestOOSEvaluatorInit, TestSaveReport

### Community 137 - "compute_phase2_objectives_from_metrics"
Cohesion: 0.04
Nodes (37): compute_phase2_objectives_from_metrics(), Penalty for weak cross-symbol robustness on one split., Build Phase 2 minimisation objectives from precomputed train/val metrics.…, _symbol_robustness_penalty(), True when val-derived feasibility penalties belong in NSGA-III fitness., _val_terms_in_fitness(), parametrize, f3 uses robust return = min(train_return, val_return) when… (+29 more)

### Community 138 - "test_rb_fail_closed.py"
Cohesion: 0.29
Nodes (11): _dummy_df(), _pool_rules(), DataFrame, Path, RB Governor fail-closed and stale-output regression tests., test_empty_phase2_pool_writes_fail_closed_output_with_reason(), test_fail_closed_output_overwrites_stale_strategy(), test_no_positive_good_candidates_fail_closed_and_do_not_call_fallback() (+3 more)

### Community 139 - "strategy_id"
Cohesion: 0.24
Nodes (9): feature_conditions_only(), phase2_rule_id(), Any, Stable provenance identifiers for Phase 2 feature rules. The RB Governor may…, Return normalized non-symbol, non-context conditions in order. Symbol scope and…, Hash the immutable Phase 2 feature logic and its discovery scope., Hash the complete economic strategy, including exit policy. ``phase2_rule_id``…, strategy_id() (+1 more)

### Community 140 - "TestMigrationSeedFraction"
Cohesion: 0.20
Nodes (6): Verify that migrant injection uses PHASE2_MIGRATION_SEED_FRACTION, not ARCHIVE., PHASE2_MIGRATION_SEED_FRACTION defaults to 0.10., Ensure the migration fraction is not the same as archive fraction., Simulate the run_epoch migrant path and assert local_cap uses migration…, Simulate the new run_epoch logic: migrant entries are capped by migration…, TestMigrationSeedFraction

### Community 141 - "_run_cluster_islands"
Cohesion: 0.06
Nodes (35): clear_global_metrics_cache(), Clear the global eval cache and force GC. Used to free RAM between cluster runs…, evict_cluster_signatures(), Evict JAX compiled signatures for a completed cluster. Removes entries from…, compute_cluster_generation_budgets(), Resolve per-island generation budgets. By default each island receives the full…, Check if an epoch should be skipped due to small remaining budget. Engine…, _run_cluster_islands() (+27 more)

### Community 142 - "_sortino_ratio_from_returns"
Cohesion: 0.32
Nodes (4): Compute a non-annualized Sortino Ratio from per-trade returns., _sortino_ratio_from_returns(), jax.lax.scan-based sequential equity simulation (legacy compat). Parameters…, TestSortinoRatioCap

### Community 143 - "trade_support_penalty"
Cohesion: 0.38
Nodes (4): Backward-compatible wrapper returning penalty only., trade_support_penalty(), Between the hard-reject floor and the soft threshold the penalty is graduated., TestTradeSupportPenalty

### Community 144 - "TestRefreshObjectivesOnResumeGate"
Cohesion: 0.23
Nodes (7): Task-10: gate cache refresh on PHASE2_PER_EPOCH_WINDOW_ROTATION. Verifies the…, Evaluate the gate expression as it appears in the source., PHASE2_PER_EPOCH_WINDOW_ROTATION=True (default): refresh = not first_epoch →…, PHASE2_PER_EPOCH_WINDOW_ROTATION=True (default): refresh = not first_epoch →…, PHASE2_PER_EPOCH_WINDOW_ROTATION=False (legacy): refresh is False even on…, PHASE2_PER_EPOCH_WINDOW_ROTATION=False (legacy): refresh is False on first…, TestRefreshObjectivesOnResumeGate

### Community 145 - "conftest.py"
Cohesion: 0.22
Nodes (10): Config, FixtureRequest, Item, _close_matplotlib_figures(), _low_memory_cleanup(), fixture, pytest_collection_modifyitems(), pytest_configure() (+2 more)

### Community 146 - "_derive_epoch_seed"
Cohesion: 0.16
Nodes (10): _derive_epoch_seed(), Return a deterministic per-epoch seed derived from *base_seed* + epoch. Used by…, Re-sample training data with a per-epoch rotated window. Each epoch gets a…, An unknown PHASE2_PER_EPOCH_WINDOW_SEED_MODE raises ValueError., Deterministic seed derivation for per-epoch windows., Same (base_seed, epoch_idx) produces the same seed., Different epoch indices produce different seeds., None base seed returns None. (+2 more)

### Community 147 - ".skip_if_valid"
Cohesion: 0.33
Nodes (3): Check if output files exist and are valid. Returns ------- dict[str,…, fixture, TestSkipIfValid

### Community 148 - "_validate_schema"
Cohesion: 0.31
Nodes (3): Validate the structure of a loaded feature selection JSON. Raises ValueError if…, _validate_schema(), TestValidateSchema

### Community 149 - "TestPermissionGatedPullback"
Cohesion: 0.33
Nodes (3): Default: opposite LWC print counts even if permission was off., v7: Range consolidation in lookback is enough to arm the trigger., TestPermissionGatedPullback

### Community 150 - "TestSplitRatio"
Cohesion: 0.21
Nodes (7): Compute expected train/val per symbol under holdout+embargo split., floor(N * HOLDOUT_TRAIN_FRACTION) rows go to train., Remaining rows after embargo gap go to validation., For N=101: floor(101 * HOLDOUT_TRAIN_FRACTION) used, not round., train + validation + embargo dropped == total rows., train + val + embargo_dropped == total for each symbol., TestSplitRatio

### Community 152 - "BFS and DFS Graph Traversal"
Cohesion: 0.20
Nodes (10): CLAUDE.md Graphify Integration, Constrained Query Expansion, BFS and DFS Graph Traversal, Graph Work Memory, Existing Graph Fast Path, CLAUDE.md Graphify Integration, Constrained Query Expansion, BFS and DFS Graph Traversal (+2 more)

### Community 153 - "TestPoolAdmissionOverfitRatioGate"
Cohesion: 0.24
Nodes (7): MonkeyPatch, Tests for the hard overfit ratio gate in pool admission. The ratio gate rejects…, train=15%/val=4% (3.75× ratio, gap=11pp < 16pp) is REJECTED by the ratio gate…, train=15%/val=10% (1.5× ratio) is ADMITTED., train/val ≈2.5× is ADMITTED when under OVERFIT_RATIO_FLOOR and gap gate., With PHASE2_OVERFIT_RATIO_FLOOR=0.0, the ratio gate is disabled and the high-…, TestPoolAdmissionOverfitRatioGate

### Community 154 - "_base_row"
Cohesion: 0.40
Nodes (3): _base_row(), Return a minimal row dict with all required columns., TestSort

### Community 155 - "TestSparsePositiveMode"
Cohesion: 0.22
Nodes (5): All non-negative, zero_ratio > 0.3 → sparse_positive., All zeros: zero_ratio = 1.0 > 0.3, min = 0 → sparse_positive., zero_ratio just above 0.3 → sparse_positive., zero_ratio must be computed on the full series, not just non-NaN., TestSparsePositiveMode

### Community 156 - "TestZeroRatioBoundary"
Cohesion: 0.22
Nodes (5): Exactly 30% zeros with non-negative values → positive (not sparse_positive)., 31% zeros with non-negative values → sparse_positive., Exactly 30% zeros with negative values → signed (not sparse_signed)., Just above 30% zeros with negative values → sparse_signed., TestZeroRatioBoundary

### Community 158 - "TestGlobalMetricsCacheClearing"
Cohesion: 0.27
Nodes (6): When seeded_keys is empty, nothing is removed., When all keys are seeded, cache becomes empty., Verify that only seeded keys are removed from the global cache., Create a deterministic chromosome key., Cache entries matching seeded_keys are removed; non-matching survive., TestGlobalMetricsCacheClearing

### Community 161 - "_should_plateau_early_stop_phase2"
Cohesion: 0.07
Nodes (39): True for cluster/orphan scoped runs rather than the global path., scoped_island_profile(), Earliest gen for plateau stop. Island epochs (~20 gens) and scaled Stage A…, _resolve_plateau_min_generation(), _should_early_stop_phase2(), _should_plateau_early_stop_phase2(), Verify decision logic uses correct patience values (regression: logs showed…, Island profile: streak=6 triggers when island_patience=6 even when… (+31 more)

### Community 162 - "TestNaNHandling"
Cohesion: 0.29
Nodes (4): All-NaN series: unique_vals = [], n_unique = 0 ≤ 2, set() ⊆ {0,1} → binary., NaN is not == 0, so it does not inflate zero_ratio., NaN should not push zero_ratio above threshold., TestNaNHandling

### Community 163 - "TestSparseSignedMode"
Cohesion: 0.33
Nodes (3): Has negative values, zero_ratio > 0.3 → sparse_signed., NaN does not count as zero; zero_ratio on full series., TestSparseSignedMode

### Community 164 - "trim_evolution_state_memory"
Cohesion: 0.50
Nodes (4): Bound run-wide eval cache size to limit RAM growth across long runs. Uses FIFO…, Drop bulky resumable state that is already persisted elsewhere., trim_evolution_state_memory(), _trim_global_metrics_cache()

### Community 166 - "test_phase2_rule_pool.py"
Cohesion: 0.05
Nodes (34): _crowding_distance(), _diversity_penalty_blended(), _dominates(), _hamming_distance(), _non_dominated_sort(), _pareto_sortino_stats(), _phenotype_bucket_key(), Return True if solution *a* dominates *b* (all objectives ≤, at least one <). (+26 more)

### Community 168 - "TestPlateauDiversityRestart"
Cohesion: 0.25
Nodes (5): Reinitialised slots have objectives=np.inf and metrics_cache={}., Even with large Pareto front, at most 10 elite are preserved., Direct unit tests for the _plateau_diversity_restart helper., Pareto elite chromosomes survive the restart., TestPlateauDiversityRestart

### Community 170 - "disable_skip_optimization"
Cohesion: 0.67
Nodes (3): disable_skip_optimization(), fixture, Disable signal skip optimization for all GPU engine tests. The skip…

### Community 172 - "TestHallOfFameTrim"
Cohesion: 0.25
Nodes (5): Verify that hall_of_fame is capped to PHASE2_HOF_EPOCH_CARRYOVER=10., HoF with >10 entries is trimmed to exactly 10 oldest entries., HoF with <=10 entries is left unchanged., Empty HoF does not crash during trim logic., TestHallOfFameTrim

### Community 173 - "phase2_support.py"
Cohesion: 0.07
Nodes (27): effective_min_trade_pool_floor(), compute_support_penalty_and_specialist(), _evolution_feasibility_floors(), _joint_primary_metric(), _passes_pool_admission_impl(), _pool_admission_floors(), phase2_support.py — Trade support penalties for Phase 2., Support penalty. Returns ------- penalty : float is_specialist : bool (always… (+19 more)

### Community 174 - "ConfigError"
Cohesion: 0.60
Nodes (5): _config_check(), ConfigError, _finite_config_number(), Raised when a configuration violates a cross-parameter contract., _validate_config_grid()

### Community 176 - "test_gpu_engine_import_does_not_crash_on_jax_failure"
Cohesion: 0.33
Nodes (6): parametrize, Tests for the lazy-JAX failure path in gpu_engine.py. These tests verify that…, Verify importing gpu_engine.py handles various JAX failure modes gracefully., Build a subprocess script that simulates JAX import failure., _subprocess_script(), test_gpu_engine_import_does_not_crash_on_jax_failure()

### Community 177 - "test_jax_compat.py"
Cohesion: 0.29
Nodes (5): parametrize, Tests for JAX / GPU engine availability detection., Package init must not crash when gpu_engine import fails., test_cpu_engine_import_without_jax(), test_get_gpu_backtest_engine_class_returns_none_on_import_error()

### Community 178 - "_legacy_writer_contract"
Cohesion: 0.67
Nodes (3): _legacy_writer_contract(), fixture, These schema tests predate mandatory trend context.

### Community 180 - "_nsga3_environmental_selection"
Cohesion: 0.10
Nodes (17): _normalize_for_association(), _nsga3_environmental_selection(), Rank-based normalization (robust to outliers like trade_penalty=50)., NSGA-III environmental selection (rank + niche on last front)., Clamp each gene to a valid class index or dont_care sentinel., Repair every row in a population matrix., _repair_chromosome(), _repair_population() (+9 more)

### Community 184 - "TestEvalCvFoldReturns"
Cohesion: 0.33
Nodes (4): Verify the helper handles None / empty fold_engines safely., fold_engines=None returns None without crashing., fold_engines=[] returns None without crashing., TestEvalCvFoldReturns

### Community 189 - "opencode.json"
Cohesion: 0.50
Nodes (3): plugin, $schema, .opencode/plugins/graphify.js

### Community 190 - "test_rb_governor_tail_holdout.py"
Cohesion: 0.12
Nodes (17): _MockEngine, DataFrame, Unit tests for RB Governor tail-holdout path in risk grid. Covers: -…, With tail_holdout_frac=0.25, a tail engine is returned with ~25% of data., With tail_holdout_frac=0.0, no tail engine., Single symbol with tail holdout still works., Verify _optimize_risk with tail_holdout_engine adds tail fields to final…, When tail_holdout_engine is provided, the final history entry contains… (+9 more)

## Knowledge Gaps
- **34 isolated node(s):** `$schema`, `.opencode/plugins/graphify.js`, `URL Ingestion`, `Folder Watcher`, `Graphify MCP Server` (+29 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **17 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `CPUBacktestEngine` connect `CPUBacktestEngine` to `phase2_rule_pool.py`, `_make_engine`, `_symbol_specialized_variants`, `_score_metrics`, `OOS_Evaluator`, `_make_df`, `_sortino_ratio_from_returns`, `_apply_monthly_admission_gate`, `trade_support_penalty`, `test_cpu_engine_properties.py`, `TestPermissionGatedPullback`, `CandidateRecord`, `nested_walk_forward.py`, `TestCausalPublicationTiming`, `TestTriggerPerSymbol`, `TestContextContract`, `gpu_engine.py`, `TestContextEntryPaths`, `test_jax_compat.py`, `_compute_rule_signal_mask`, `_build_entries_from_rule_set`, `TestEvalCvFoldReturns`, `GPUBacktestEngine`, `_build_data_matrix`, `ValueError`, `sample_df_for_phase2`, `test_rb_governor_tail_holdout.py`, `_apply_dynamic_rule`, `test_cpu_engine.py`, `_discretize_series`, `DataFrame`, `test_gpu_engine_properties.py`, `ValidationError`, `run_pipeline.py`, `test_output_writer_properties.py`, `Data_Loader`, `test_phase2_gpu_throughput.py`, `phase2_island_scheduler.py`, `baselines.py`, `Rule_Pool_Generator`, `_jax_compute_trade_outcomes`, `_build_pool_from_archive`, `_jax_compute_rule_signals`, `TestMakeWalkForwardFoldEngines`, `MonthlyWindowSummary`, `._engine`, `.run`?**
  _High betweenness centrality (0.138) - this node is a cross-community bridge._
- **Why does `Reporter` connect `Reporter` to `phase2_rule_pool.py`, `TestWriteSpearmanCorrelationReport`, `TestEquityCurveDateAxis`, `prop_settings`, `OOS_Evaluator`, `TestPlotDistributionAndEquity`, `Rule_Pool_Generator`, `._ensure_dir`, `TestWriteStrategyEvaluationTable`, `_build_pool_from_archive`, `TestPlotPerRuleBreakdown`, `DataFrame`, `ValidationError`, `test_reporter.py`, `run_pipeline.py`, `.run`?**
  _High betweenness centrality (0.118) - this node is a cross-community bridge._
- **Why does `Rule_Pool_Generator` connect `Rule_Pool_Generator` to `phase2_rule_pool.py`, `CPUBacktestEngine`, `log_memory_rss`, `_NumpyJSONEncoder`, `.run`, `_get_dont_cares`, `compute_phase2_objectives_from_metrics`, `_run_cluster_islands`, `Pipeline_Orchestrator`, `_apply_monthly_admission_gate`, `TestRefreshObjectivesOnResumeGate`, `_make_train_df`, `_derive_epoch_seed`, `TestLoadPool`, `Reporter`, `TestPoolAdmissionOverfitRatioGate`, `test_phase2_rule_pool.py`, `test_phase2_window_rotation.py`, `_derive_island_seed`, `test_phase2_rule_pool_properties.py`, `sample_df_for_phase2`, `TestRulePoolGeneratorRun`, `test_plateau_state_leak.py`, `test_crash_fix_and_run_logging.py`, `filter_migrants_for_cluster`, `run_pipeline.py`, `phase2_island_scheduler.py`, `_build_pool_from_archive`, `_validate_pool_schema`, `TestValLeakGate`?**
  _High betweenness centrality (0.118) - this node is a cross-community bridge._
- **Are the 15 inferred relationships involving `Reporter` (e.g. with `CvFoldValEvaluator` and `Rule_Pool_Generator`) actually correct?**
  _`Reporter` has 15 INFERRED edges - model-reasoned connections that need verification._
- **Are the 64 inferred relationships involving `CPUBacktestEngine` (e.g. with `GPUBacktestEngine` and `JointPortfolioEngine`) actually correct?**
  _`CPUBacktestEngine` has 64 INFERRED edges - model-reasoned connections that need verification._
- **Are the 62 inferred relationships involving `Rule_Pool_Generator` (e.g. with `CPUBacktestEngine` and `Reporter`) actually correct?**
  _`Rule_Pool_Generator` has 62 INFERRED edges - model-reasoned connections that need verification._
- **Are the 31 inferred relationships involving `Output_Writer` (e.g. with `OOS_Evaluator` and `_Phase5JSONEncoder`) actually correct?**
  _`Output_Writer` has 31 INFERRED edges - model-reasoned connections that need verification._