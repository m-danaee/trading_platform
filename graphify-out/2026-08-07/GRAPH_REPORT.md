# Graph Report - trading_platform  (2026-08-07)

## Corpus Check
- 188 files · ~271,110 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 4856 nodes · 11008 edges · 193 communities (182 shown, 11 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 563 edges (avg confidence: 0.57)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `1865873e`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- evox_runner.py
- .run
- rb_governor.py
- _make_engine
- Pipeline_Orchestrator
- run_phase2_evolution_epoch
- test_evox_runner.py
- _valid_metrics
- Rule_Pool_Generator
- _make_df
- phase2_rule_pool.py
- run_phase2_evolution
- Output_Writer
- Data_Splitter
- test_run_pipeline.py
- _apply_monthly_admission_gate
- test_feature_detector_properties.py
- _make_train_df
- _split
- test_cpu_engine_properties.py
- detect_feature_mode
- selector.py
- Feature_Selector
- rolling_cv.py
- TestPlotPhase2Metrics
- test_encoder_properties.py
- write_evaluator_clean
- _compose_ruleset
- nested_walk_forward.py
- _preserve_deployable_elites
- maybe_log_generation
- Hybrid CPU and GPU execution policy
- _select_diverse_subset
- test_rb_min_symbols.py
- Graphify Pipeline
- writer.py
- config.py
- TestEquityCurveDateAxis
- test_phase2_window_rotation.py
- research_integrity.py
- _loader_from_rows
- test_phase5_oos.py
- gpu_engine.py
- _gpu_runtime.py
- ._ensure_dir
- run_pipeline.py
- test_certificate_first_selection.py
- non_dominated_sort
- optuna_search.py
- _init_population
- test_phase2_rule_pool_properties.py
- _compute_rule_signal_mask
- validate_config
- build_hybrid_symbol_clusters
- _legacy_writer_contract
- trend_context.py
- Feature_Detector
- test_reporter.py
- GPUBacktestEngine
- get_normalized_symbol_array
- ValueError
- sample_df_for_phase2
- TestEquityCurvePlots
- TestRulePoolGeneratorRun
- test_plateau_state_leak.py
- TestFeasibilityGateFailures
- dashboard.py
- TestWriteSpearmanCorrelationReport
- test_data_loader_properties.py
- barrier.py
- _m
- _apply_dynamic_rule
- test_cpu_engine.py
- run_rb_governor_pipeline
- execution_ok
- TestWriteStrategyEvaluationTable
- test_gpu_engine.py
- test_crash_fix_and_run_logging.py
- TestDeferredWarmup
- test_feature_selector_properties.py
- phase2_island_scheduler.py
- compute_labels
- TestPlotPerRuleBreakdown
- resolve_island_hyperparams
- evaluator_health.py
- _build_target
- .test_accepts_high_val_return_and_trades
- test_gpu_engine_properties.py
- TestRunLogHandlerLifecycle
- resolve_evolution_floors
- phase5_oos.py
- set_purged_wf_reference_rows
- test_phase2_support.py
- Encoder
- test_output_writer_properties.py
- Data_Loader
- test_phase2_gpu_throughput.py
- .decode_chromosome
- .load_strategies
- gate_positive_good
- _derive_island_seed
- resolve_phase2_stage_params
- _remove_low_dispersion
- prop_settings
- test_crash_fix_properties.py
- Reporter
- baselines.py
- _normalize_for_association
- _jax_compute_trade_outcomes
- _should_post_restart_early_stop_phase2
- _build_pool_from_archive
- TestExecutionHealthInGate
- _make_df
- TestDeriveValSampleSeed
- TestHammingThresholdAutoScale
- TestValLeakGate
- DataFrame
- _jax_compute_rule_signals
- TestParetoCollapseWarningGate
- CPUBacktestEngine
- _compute_stability
- phase2_support.py
- MonthlyWindowSummary
- ._engine
- test_rb_concentration_tail_fail_closed.py
- constrained_non_dominated_sort
- test_feature_selector.py
- .run
- stratification_scenario_strategy
- TestSpearmanSignConsistency
- .encode_condition
- log_memory_rss
- _derive_epoch_seed
- _apply_colab_gpu_defaults
- loader.py
- test_evaluator_health.py
- OOS_Evaluator
- compute_phase2_objectives_from_metrics
- _raw_feasibility_violation_score
- .load_pool
- test_phase5_oos_properties.py
- test_phase2_island_scheduler.py
- TestIslandSchedulerGlobalMode
- TestEndToEndRotation
- TestRefreshObjectivesOnResumeGate
- conftest.py
- .get_dont_care
- .skip_if_valid
- _validate_schema
- splitter.py
- TestSplitRatio
- passes_pool_admission_gate
- BFS and DFS Graph Traversal
- TestGPUCPUNumericalParity
- load_cached_split_if_fresh
- TestSparsePositiveMode
- TestZeroRatioBoundary
- _validate_pool_schema
- TestGlobalMetricsCacheClearing
- cpu_engine.py
- .load_and_validate
- Phase2StageParams
- TestNaNHandling
- TestSparseSignedMode
- TestPlotPhase2Pnl
- TestMigrationSeedFraction
- test_phase2_rule_pool.py
- strategy_id
- TestPoolAdmissionOverfitRatioGate
- test_phase2_batch_evaluator_parity.py
- _plateau_diversity_restart
- TestSavePerSymbolCsv
- TestHallOfFameTrim
- trade_support_penalty
- TestNClustersDefined
- test_nsga3_selection.py
- test_gpu_engine_import_does_not_crash_on_jax_failure
- test_jax_compat.py
- _downsample_chronological
- TestMigrationEnabledByDefault
- TestEvolutionFeasibilityFloors
- _largest_safe_range
- opencode.json
- graphify.js
- Obsolete implementation cleanup policy
- data/__init__.py
- features/__init__.py
- gpu_fuzzy_trader/__init__.py
- output/__init__.py
- phases/__init__.py
- gpu-fuzzy-trader

## God Nodes (most connected - your core abstractions)
1. `Reporter` - 159 edges
2. `CPUBacktestEngine` - 158 edges
3. `Rule_Pool_Generator` - 138 edges
4. `Output_Writer` - 85 edges
5. `prop_settings()` - 79 edges
6. `Pipeline_Orchestrator` - 68 edges
7. `Data_Loader` - 62 edges
8. `_run_nsga3()` - 58 edges
9. `Feature_Selector` - 58 edges
10. `OOS_Evaluator` - 58 edges

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

## Communities (193 total, 11 thin omitted)

### Community 0 - "evox_runner.py"
Cohesion: 0.05
Nodes (93): IslandHyperparams, Resolved Phase 2 knobs for cluster or orphan slices., _assign_eval_result(), _behavior_descriptor(), _binary_tournament_pick(), _build_diversity_reference(), _constraint_violations(), _count_deployable_preview() (+85 more)

### Community 1 - ".run"
Cohesion: 0.05
Nodes (41): _archive_feature_signature(), _deployable_archive_pool_entries(), _entry_validation_per_symbol_metrics(), _filter_pool_by_admission(), _merge_archive_entries(), _monthly_admission_source_df(), _pool_entry_rank(), _pool_path_key() (+33 more)

### Community 2 - "rb_governor.py"
Cohesion: 0.05
Nodes (83): expectancy_lcb_pct(), Return a conservative lower bound for per-trade net expectancy. Exact CPU…, _attach_source_symbol_filters(), _available_symbols(), _balanced_phase2_shortlist(), _cost_stress_gate(), _diversification_shortlist(), _ensure_symbol_filtered_rule() (+75 more)

### Community 3 - "_make_engine"
Cohesion: 0.05
Nodes (33): _build_entries_from_rule_set(), Priority-based rule assignment: first matching rule wins per row. Mirrors…, _make_df(), _make_engine(), DataFrame, All TP trades → win_rate = 100%., No losing trades → profit_factor = 99.0., Row matching both rules should be assigned to rule 1 only. (+25 more)

### Community 4 - "Pipeline_Orchestrator"
Cohesion: 0.06
Nodes (43): FileHandler, _context_coverage_for_direction(), _log_phase_entry(), _log_pipeline_config(), _now_iso(), Pipeline_Orchestrator, DataFrame, Persist the final status for the current run identity. (+35 more)

### Community 5 - "run_phase2_evolution_epoch"
Cohesion: 0.14
Nodes (15): StageLabel, Evolve one island epoch and return updated resumable state., run_phase2_evolution_epoch(), _FakeEngine, Verify reset_plateau clears restart counters on resumed state., AC: resumed island epoch with reset_plateau=True clears restart counters., Global/non-island mode: reset_plateau=False preserves counters. Uses…, Task 2: Verify refresh_objectives_on_resume resets stale objectives on resumed… (+7 more)

### Community 6 - "test_evox_runner.py"
Cohesion: 0.06
Nodes (35): _evaluate_population_indices(), _inherit_val_metrics_from_global_cache(), _load_global_metrics_cache(), _metrics_snapshot(), Phase2EvolutionState, Shallow copy for evolution caches (metrics are flat numeric dicts)., Store processed metrics (and optional val sidecar) in the run-wide cache., Bound run-wide eval cache size to limit RAM growth across long runs. Uses FIFO… (+27 more)

### Community 7 - "_valid_metrics"
Cohesion: 0.08
Nodes (27): _combined_return_score(), Profit objective for lenient rule addition, but now evaluator_v5 aware. A new…, _MockEngine, Minimal mock that mimics CPUBacktestEngine for testing _optimize_risk.…, Verify the CV-fold consistency penalty in ``_score_metrics``. New formula…, cv_fold_returns=[5.0, -2.0, 3.0] — worst fold negative + high variance., All positive folds with low variance — no penalty., cv_fold_returns=None skips the CV term. (+19 more)

### Community 8 - "Rule_Pool_Generator"
Cohesion: 0.05
Nodes (33): _count_active_conditions(), _get_dont_cares(), _mutate(), Mutate a chromosome (returns a copy). When activating a dont_care gene, feature…, Return array of dont_care sentinels for each feature., Count active rule conditions (sparse slots or dense dont_care encoding)., Phase 2: GPU-accelerated multi-objective evolutionary rule pool generation.…, Inject guarded migration seeds for the next epoch. (+25 more)

### Community 9 - "_make_df"
Cohesion: 0.08
Nodes (21): _make_df(), _make_engine(), MonkeyPatch, Chromosome positions must follow feature_modes insertion order., Exact CPU re-evaluation must interpret genes like the GPU path., Chromosome that matches nothing returns 0 executed trades., Zero-signal chunks should use reject metrics without scanning., PHASE2_SKIP_ZERO_SIGNAL_SCAN must match always-scan metrics. (+13 more)

### Community 10 - "phase2_rule_pool.py"
Cohesion: 0.08
Nodes (65): _hamming_distance(), Hamming distance between two chromosomes (sparse-aware)., _random_active_class(), _archive_objective_vector(), _chromosome_for_pool_export(), _crossover(), _diversity_penalty_blended(), _hamming_distance() (+57 more)

### Community 11 - "run_phase2_evolution"
Cohesion: 0.04
Nodes (45): Shared val-cadence check for both NSGA-II fallback and NSGA-III loops. Val…, Run Phase 2 NSGA-III evolution. Returns (pareto_pool, history) or state., run_phase2_evolution(), _should_run_val_this_gen(), Evolutionary algorithm drivers for Phase 2., skipif, TestRunPhase2EvolutionFallback, TestRunPhase2EvolutionSmoke (+37 more)

### Community 12 - "Output_Writer"
Cohesion: 0.06
Nodes (22): Output_Writer, Serializes RuleSet dicts to JSON and loads/validates existing JSON files.…, _make_rule(), _make_rule_set(), Unit tests for gpu_fuzzy_trader.output.writer — Output_Writer Tests cover: -…, A rule with only tp non-zero should be accepted., Spot-check a variety of valid fuzzy value names., Write rule_set to a temp file and reload the raw JSON. (+14 more)

### Community 13 - "Data_Splitter"
Cohesion: 0.13
Nodes (20): Data_Splitter, Chronological train/validation splitter., Module-level wrapper around ``Data_Splitter.split_and_persist``., split_and_persist(), _make_df(), _make_timestamps(), _patch_split_paths(), DataFrame (+12 more)

### Community 14 - "test_run_pipeline.py"
Cohesion: 0.12
Nodes (30): _context_coverage_preflight(), _context_coverage_report(), Temporarily rebind all cached output paths for one pipeline run., Return split-aware context coverage for both trading directions., Reject enriched tapes generated under a different context contract., Log coverage and block only directions that cannot meet their floors., _temporary_output_paths(), _validate_enriched_context_contract() (+22 more)

### Community 15 - "_apply_monthly_admission_gate"
Cohesion: 0.06
Nodes (29): _apply_monthly_admission_gate(), _evaluate_rule_on_window(), _monthly_window_metrics(), Evaluate a single pool rule on a single monthly window. Returns the full window…, Normalize a window evaluator result for the monthly gate. The float fallback…, Apply the monthly-window shadow-test gate to a pool of rules. Each rule is…, _DeterministicEvaluator, _DeterministicMetricsEvaluator (+21 more)

### Community 16 - "test_feature_detector_properties.py"
Cohesion: 0.09
Nodes (45): all_nan_series(), arbitrary_numeric_series(), binary_series(), large_value_series(), positive_series(), composite, DrawFn, given (+37 more)

### Community 17 - "_make_train_df"
Cohesion: 0.09
Nodes (21): Sample up to *total_rows* rows, distributed equally across symbols. A single…, _sample_df(), _make_train_df(), DataFrame, Critical: bars must be contiguous so the backtest engine preserves temporal…, Random start must be bounded so the slice always fits forward., divmod distribution gives exactly total_rows (no rounding loss). 701_000 % 14…, total_rows < n_symbols must NOT force 1 row per symbol. (+13 more)

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
Cohesion: 0.09
Nodes (37): _align_feature_array(), build_phase1_shared_context(), _build_symbol_masks(), _candidate_feature_columns(), _check_spearman_sign_consistency(), _compute_chronological_stationarity_scores(), _get_spearman_folds(), _mi_scores_for_mask() (+29 more)

### Community 22 - "Feature_Selector"
Cohesion: 0.10
Nodes (19): Feature_Selector, Score and rank features separately for long and short directions., _make_train_df(), MonkeyPatch, Create a minimal training DataFrame with label columns and feature columns., Label columns must not appear in selected features., Meta columns must not appear in selected features., Loader internal columns and ``_``-prefixed names are not candidates. (+11 more)

### Community 23 - "rolling_cv.py"
Cohesion: 0.09
Nodes (39): aggregate_fold_metrics(), _bar_index_col(), _build_fold_from_ranges(), build_forbidden_ranges(), build_purged_walk_forward_folds(), cv_folds_only(), derive_primary_holdout(), FoldMetricsSummary (+31 more)

### Community 24 - "TestPlotPhase2Metrics"
Cohesion: 0.23
Nodes (4): _make_history(), History entries with missing keys should not raise., Create a minimal Phase 2 history list., TestPlotPhase2Metrics

### Community 25 - "test_encoder_properties.py"
Cohesion: 0.09
Nodes (37): all_active_chromosome_strategy(), chromosome_with_dont_cares_strategy(), feature_name_strategy(), composite, DrawFn, given, ndarray, Property-based tests for gpu_fuzzy_trader.features.encoder.Encoder **Validates:… (+29 more)

### Community 26 - "write_evaluator_clean"
Cohesion: 0.08
Nodes (29): _maybe_write_evaluator_clean(), Write a stripped strategy file containing only ``direction`` and ``rules_set``.…, Write a stripped strategy file alongside the main strategy JSON. This is a…, write_evaluator_clean(), _make_rule(), minimal_strategy(), fixture, Path (+21 more)

### Community 27 - "_compose_ruleset"
Cohesion: 0.16
Nodes (27): _candidate_coverage_symbols(), _compose_ruleset(), Symbol coverage for compose diversity. Prefer **traded** symbols from backtest…, patch, _cand(), engines(), _generalist_cand(), fixture (+19 more)

### Community 28 - "nested_walk_forward.py"
Cohesion: 0.13
Nodes (25): Validation helpers for monthly, nested, and multiplicity-safe research., deflated_sharpe_ratio(), estimate_pbo(), Selection-multiplicity diagnostics for strategy research artifacts., Estimate the fraction of folds where the IS winner misses OOS median. Inputs…, Return a normal-approximation deflated Sharpe probability. This is a…, Create a compact ledger-ready multiplicity report., summarize_multiplicity() (+17 more)

### Community 29 - "_preserve_deployable_elites"
Cohesion: 0.09
Nodes (25): _build_rank_and_crowding(), environmental_selection_nsga2(), _preserve_deployable_elites(), Per-individual Pareto rank (lower is better) and crowding distance., Canonical NSGA-II truncation on a 2N merged population., Force-preserve top-K deployable-archive elites in the live population.…, _make_chromosome(), _make_deployable_entry() (+17 more)

### Community 30 - "maybe_log_generation"
Cohesion: 0.09
Nodes (18): generation_log_interval(), iteration_log_interval(), log_generation(), maybe_log_generation(), log_progress.py — Throttled progress logging for long pipeline loops., Return how often to log generation progress. Uses LOG_GENERATION_INTERVAL from…, Log generation progress when the step matches the throttle interval., Log first step, last step, and every *interval* steps in between. (+10 more)

### Community 31 - "Hybrid CPU and GPU execution policy"
Cohesion: 0.07
Nodes (37): Local low-memory test policy, Virtual environment command policy, Package compile gate, Locked research dependencies, Focused low-memory test gate, Python 3.11 CI runtime, Research CI workflow, Balanced Mode-A assumption (+29 more)

### Community 32 - "_select_diverse_subset"
Cohesion: 0.12
Nodes (15): Max-min Hamming diversity sampling: greedy pick farthest from chosen. Returns…, _select_diverse_subset(), ndarray, Tests for H5/M4/M5 evolution convergence behaviors. Covers: - HoF trimming at…, Verify _select_diverse_subset correctness for edge cases., Create n distinct dense chromosomes., k=0 should return [] even with non-empty chromosomes., k<0 should return []. (+7 more)

### Community 33 - "test_rb_min_symbols.py"
Cohesion: 0.10
Nodes (26): mandatory_context_conditions(), Return the fixed, mandatory context conditions for *direction*., _symbols_in_rules(), _dummy_df(), _make_candidate_records(), _mock_train_metrics(), _multi_symbol_rules(), _no_symbol_rule() (+18 more)

### Community 34 - "Graphify Pipeline"
Cohesion: 0.06
Nodes (36): Folder Watcher, URL Ingestion, Conditional Graph Exports, Graphify MCP Server, Extraction Confidence Rubric, Deterministic Full-Path Node IDs, Semantic Hyperedges, Cross-Repository Graph Merge (+28 more)

### Community 35 - "writer.py"
Cohesion: 0.09
Nodes (23): context_permission_column(), context_trigger_column(), Return the direction-specific permission column name., Return the direction-specific LWC pullback-reversal trigger column., _condition_feature(), _context_feature_direction(), Path, writer.py — Output_Writer Serializes RuleSet dicts to JSON with exact schema… (+15 more)

### Community 36 - "config.py"
Cohesion: 0.07
Nodes (35): _config_check(), ConfigError, effective_min_trade_support(), effective_monthly_min_trades(), effective_sortino_min_trade_threshold(), effective_val_trade_floor_for_objectives(), _finite_config_number(), get_purged_wf_reference_rows() (+27 more)

### Community 37 - "TestEquityCurveDateAxis"
Cohesion: 0.09
Nodes (21): _make_trade_log(), DataFrame, fixture, Unit tests for date-based x-axis in Reporter.plot_equity_curve (Task 10.3).…, When Entry_Time is absent, x-axis label is 'Trade #'., When Entry_Time is all-NaN, x-axis label is 'Trade #'., axhline is called (initial-capital line) in date mode., axhline is called (initial-capital line) in Trade # mode. (+13 more)

### Community 38 - "test_phase2_window_rotation.py"
Cohesion: 0.10
Nodes (22): Cap *total_rows* so the per-symbol request fits within the safe range. When…, _resolve_sample_total_rows(), _make_multi_sym_df(), DataFrame, Tests for per-epoch train-window rotation (task-1)., Capping logic for per-epoch window rotation., With PHASE2_PER_EPOCH_WINDOW_ROTATION=False, total_rows is unchanged., With rotation enabled, total_rows is capped by rotation fraction. (+14 more)

### Community 39 - "research_integrity.py"
Cohesion: 0.16
Nodes (24): _canonical_json(), count_trials(), dataset_manifest(), ExperimentLedger, forward_acceptance_lock_path(), Any, Path, PathLike (+16 more)

### Community 40 - "_loader_from_rows"
Cohesion: 0.07
Nodes (30): load_dataset(), Module-level wrapper around ``Data_Loader.load_dataset``., _base_row(), _loader_from_rows(), _make_csv(), _make_ohlcv_rows(), _make_rows(), _make_timestamps() (+22 more)

### Community 41 - "test_phase5_oos.py"
Cohesion: 0.10
Nodes (13): _isolate_phase5_reporter_outputs(), fixture, Unit tests for gpu_fuzzy_trader.phases.phase5_oos.OOS_Evaluator Tests cover: -…, Override module-level path dicts and return originals., Keep Reporter plots/CSVs out of the checked-in outputs directory., Write a valid selected-features JSON to path., Write a synthetic test CSV with all required columns (including feat_0..4) to a…, Integration tests using tmp_path overrides for all output paths. The run()… (+5 more)

### Community 42 - "gpu_engine.py"
Cohesion: 0.07
Nodes (34): _batch_metrics_from_array(), _build_event_batch(), _exposure_slot_capacity(), _fast_reject_result_rows(), _jax_compute_rule_signals_sparse_batch(), _jax_open_slot(), _jax_release_open_slots(), _jax_simulate_equity_event_batch() (+26 more)

### Community 43 - "_gpu_runtime.py"
Cohesion: 0.06
Nodes (42): phase2_should_enrich_symbol_metrics(), Return True when GPU batch eval should run a follow-up CPU enrichment pass.…, configure_phase2_gpu_runtime(), detect_gpu_memory_used_gb(), detect_gpu_vram_gb(), detect_system_ram_gb(), evict_cluster_signatures(), _iter_warmup_targets() (+34 more)

### Community 44 - "._ensure_dir"
Cohesion: 0.09
Nodes (17): _bucket_series_by_mode(), DataFrame, Series, Write feature-stratified performance metrics to CSV files (one per split). For…, Create parent directories for *path* if they do not exist., Compute maximum drawdown percentage from an equity series. Parameters…, Write compact train/validation/test generalization diagnostics to JSON. The…, Plot a per-rule performance breakdown across train/validation/test splits.… (+9 more)

### Community 45 - "run_pipeline.py"
Cohesion: 0.09
Nodes (21): __main__.py — Entry point for `python -m gpu_fuzzy_trader.run_pipeline` Allows…, Any, Typed, versioned profile for the active research contract., Small stable surface for comparing experiments. The legacy config module…, ResearchProfile, main(), _NumpyJSONEncoder, _phase5_test_metrics() (+13 more)

### Community 46 - "test_certificate_first_selection.py"
Cohesion: 0.10
Nodes (29): _candidate_positive_symbols(), _passes_symbol_concentration_gate(), _passes_symbol_contribution_certificate(), _passes_tail_holdout_gate(), _portfolio_selection_certificate(), Any, Require positive, supported validation PnL from multiple symbols. Symbol…, Return supported positive validation symbols for one candidate. (+21 more)

### Community 47 - "non_dominated_sort"
Cohesion: 0.12
Nodes (30): batch_hamming_min(), _batch_static_penalties_numba(), batch_static_support_penalties(), crowding_distance(), _crowding_distance_numba(), _crowding_distance_py(), _disable_numba_sort(), _dominates_numba() (+22 more)

### Community 48 - "optuna_search.py"
Cohesion: 0.10
Nodes (29): _active_search_space(), apply_trial_config(), collect_validation_metrics(), compute_score(), _copy_phase1_2_outputs(), main(), objective(), _pearson() (+21 more)

### Community 49 - "_init_population"
Cohesion: 0.09
Nodes (31): assign_strata_to_indices(), build_feature_sampling_probs(), pick_active_count(), _pick_active_index(), _pick_inactive_index(), ndarray, phase2_init.py — Sparsity-guided stratified population initialization for Phase…, Assign elite / explorer labels to non-seeded population rows. (+23 more)

### Community 50 - "test_phase2_rule_pool_properties.py"
Cohesion: 0.11
Nodes (21): feature_infos_and_train_df(), _isolate_phase2_archive_paths(), _make_feature_infos(), _make_train_df(), composite, DataFrame, DrawFn, fixture (+13 more)

### Community 51 - "_compute_rule_signal_mask"
Cohesion: 0.15
Nodes (13): conditions_cache_key(), get_or_build_rule_mask(), ndarray, condition_cache.py — Cached boolean masks for textual rule conditions. Avoids…, Stable hashable key for a list of condition strings., Return a boolean row mask for *conditions*, using *cache* when provided. The…, _compute_rule_signal_mask(), Build one boolean signal mask (evaluator_v5 parity). Feature conditions are… (+5 more)

### Community 52 - "validate_config"
Cohesion: 0.11
Nodes (28): context_contract(), _debug_symbol_universe_size(), effective_config_snapshot(), effective_min_profitable_symbols(), Active symbol count when debug scope is on; None for full-universe runs., Cap cross-symbol profitability gate to the active universe size. With…, Validate all high-impact hyperparameter relationships. The function is…, Return resolved values and derived constraints for audit/reporting. (+20 more)

### Community 53 - "build_hybrid_symbol_clusters"
Cohesion: 0.11
Nodes (30): build_hybrid_symbol_clusters(), _corr_embedding_block(), _feature_names_union(), _feature_profile_block(), load_symbol_clusters(), persist_symbol_clusters(), Any, DataFrame (+22 more)

### Community 54 - "_legacy_writer_contract"
Cohesion: 0.67
Nodes (3): _legacy_writer_contract(), fixture, These schema tests predate mandatory trend context.

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

### Community 59 - "get_normalized_symbol_array"
Cohesion: 0.10
Nodes (16): _batch_eval_rule_set_pickled(), Simulate a rule set on rows [row_start, row_end) without copying the df. Used…, Simulate a rule set and return performance metrics. Parameters ----------…, Evaluate multiple rule sets in parallel (ProcessPool, thread fallback)., Top-level worker for ProcessPoolExecutor (must be picklable)., _rule_symbols_for_allocation(), _rules_need_normalized_symbols(), get_normalized_symbol_array() (+8 more)

### Community 60 - "ValueError"
Cohesion: 0.06
Nodes (35): barrier_column_names(), Return stable internal return/offset column names for one risk pair., _append_allocated_entries(), _build_rule_signal_mask(), compute_entry_time_priority(), _expectancy_lcb_pct(), _expected_shortfall_pct(), _parse_condition() (+27 more)

### Community 61 - "sample_df_for_phase2"
Cohesion: 0.12
Nodes (14): DataFrame, Build train/val backtest engines., Build the selected Phase 2 backend for the sampled train frame., Return whether this generator should avoid allocating a JAX engine.…, Build an engine on *df* using the same backend selection logic., Resolve Phase 2 row budget then sample with aligned symbol windows., sample_df_for_phase2(), _minimal_backtest_df() (+6 more)

### Community 62 - "TestEquityCurvePlots"
Cohesion: 0.13
Nodes (13): Build a flat list of per-symbol metric dicts for CSV output. Uses the…, DataFrame, Verify plot_equity_curve is called for all three splits and handles empty logs., Helper: run OOS_Evaluator.run() with mocked internals and capture plot calls., plot_equity_curve must be called with 'train', 'validation', and 'test'., plot_equity_curve must be called exactly three times (one per split)., Empty train trade log must not raise an exception., Empty validation trade log must not raise an exception. (+5 more)

### Community 63 - "TestRulePoolGeneratorRun"
Cohesion: 0.15
Nodes (8): Integration tests using tiny population and generation counts., Verify that Rule_Pool_Generator's persistent self._rng advances across multiple…, After two run_epoch() calls, the internal RNG state must differ from the…, The RNG should produce *different* sequences in two consecutive run_epoch()…, Two generators with different seeds must have different RNG state., Rule_Pool_Generator must initialize self._rng as a Generator., TestRulePoolGeneratorRng, TestRulePoolGeneratorRun

### Community 64 - "test_plateau_state_leak.py"
Cohesion: 0.11
Nodes (21): _make_minimal_gen(), _mock_evolution_state(), _mock_stage_plan(), Regression tests for plateau-state leak fixes (Fixes A + B). Fix A:…, AC-1, AC-3: plateau_streak resets per epoch; reset_plateau=True always., reset_plateau=True is passed on the very first epoch call., reset_plateau=True is also passed on epoch 2 (regression for leak)., Even when PHASE2_ISLAND_TWO_STAGE_ENABLED=False, reset_plateau=True. (+13 more)

### Community 65 - "TestFeasibilityGateFailures"
Cohesion: 0.09
Nodes (13): fixture, Tests for _feasibility_gate_failures — per-gate breakdown., A rule that should pass all 9 gates., A rule with too few train trades., A rule passing all gates returns all-zero dict., When val_metrics is None, only val_required=1, others=0., A rule with too few train trades fails train_trade_floor., When PHASE2_REQUIRE_LAST_FOLD_POSITIVE=True and val_ret <= 0, val_ret_positive… (+5 more)

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
Cohesion: 0.20
Nodes (14): attach_barrier_outcomes(), configured_barrier_pairs(), _first_touch_for_symbol(), _first_touch_for_symbol_numba(), _first_touch_for_symbol_python(), _number_token(), DataFrame, ndarray (+6 more)

### Community 70 - "_m"
Cohesion: 0.13
Nodes (16): evaluator_health_penalty(), Penalty for evaluator_v5 failure modes — higher is worse. Parameters ----------…, _m(), Max positions 15 > 10 → positive penalty., Max positions 8 <= 10 → no additional penalty from positions., Missing ``raw_signal_count`` → treated as 0 → no skip/exec penalty., ``raw_signal_count=0`` → no skip/exec ratio computed → no penalty., Role 'train' same as 'valid' (1.0x). (+8 more)

### Community 71 - "_apply_dynamic_rule"
Cohesion: 0.21
Nodes (4): _apply_dynamic_rule(), Apply one exported text condition using the original threshold logic. Exactly…, Test all threshold branches of _apply_dynamic_rule., TestApplyDynamicRule

### Community 72 - "test_cpu_engine.py"
Cohesion: 0.08
Nodes (19): _normalize_direction(), Compute a non-annualized Sortino Ratio from per-trade returns., _safe_profit_factor(), _sortino_ratio_from_returns(), JointPortfolioEngine, DataFrame, Evaluate long and short rule books in one net-position account., Backward-compatible wrapper returning penalty only. (+11 more)

### Community 73 - "run_rb_governor_pipeline"
Cohesion: 0.09
Nodes (37): effective_rb_min_distinct_symbols(), Return the RB coverage target for the active debug universe. Full runs keep…, _assert_capital_budget(), _assert_mandatory_context(), _enforce_capital_budget(), evaluate_strategy_file_governor(), Path, PathLike (+29 more)

### Community 74 - "execution_ok"
Cohesion: 0.15
Nodes (11): execution_ok(), Return ``True`` iff the evaluator would execute this rule set reasonably. A…, Tests for ``execution_ok``., Skip ratio 0.10 <= 0.20 and exec ratio 0.90 >= 0.60 → True., Skip ratio 0.30 > 0.20 → False., Exec ratio 0.50 < 0.60 → False., Missing ``raw_signal_count`` → treated as 0 → False., ``raw_signal_count=0`` → False. (+3 more)

### Community 75 - "TestWriteStrategyEvaluationTable"
Cohesion: 0.22
Nodes (9): _make_eval_rule_set(), _make_metrics_by_split(), _make_trade_logs_by_split(), Create a minimal rule set for evaluation table tests., Create a metrics_by_split dict with all three splits populated., Create a trade_logs_by_split dict with all three splits populated., Sharpe ratio should equal mean(r)/std(r, ddof=1) for a known log., win_rate, mdd_pct etc. should come from metrics_by_split. (+1 more)

### Community 76 - "test_gpu_engine.py"
Cohesion: 0.13
Nodes (13): _build_data_matrix(), _discretize_series(), DataFrame, Series, Discretize a feature series using evaluator_v5's fixed fuzzy bins. Exported…, Build an (N, K) integer matrix of discretized feature values., disable_skip_optimization(), DataFrame (+5 more)

### Community 77 - "test_crash_fix_and_run_logging.py"
Cohesion: 0.12
Nodes (17): _capture_logs(), _make_feature_infos_crash(), _make_train_df_crash(), DataFrame, Unit tests for crash-fix-and-run-logging spec. Covers: - Task 1.1: Smoke tests…, Requirement 3.1 — save_archive is called before _release_resources in run()., Mock save_archive and _release_resources, run Rule_Pool_Generator.run() with a…, save_archive is called with self.direction as the first argument. (+9 more)

### Community 78 - "TestDeferredWarmup"
Cohesion: 0.20
Nodes (6): Unit tests for the ``defer_warmup`` flag on ``Rule_Pool_Generator``. When…, Existing callers without defer_warmup still warm at init., The configure_phase2_gpu_runtime call is inside 'if not self._defer_warmup:'…, _run_cluster_islands passes defer_warmup=True to all generators., _run_cluster_islands calls warmup_phase2_gpu_kernels per cluster., TestDeferredWarmup

### Community 79 - "test_feature_selector_properties.py"
Cohesion: 0.15
Nodes (23): dataset_with_high_dispersion_feature(), dataset_with_label_and_meta_columns(), dataset_with_sufficient_dispersion_feature(), _make_label_block(), composite, DataFrame, DrawFn, given (+15 more)

### Community 80 - "phase2_island_scheduler.py"
Cohesion: 0.09
Nodes (34): filter_df_to_symbols(), phase2_history_path(), phase2_pool_path(), DataFrame, Return rows for the given symbols; raises if column missing or no rows., Resolve Phase 2 pool path., Resolve Phase 2 history path., clear_global_metrics_cache() (+26 more)

### Community 81 - "compute_labels"
Cohesion: 0.20
Nodes (10): compute_labels(), DataFrame, Forward-window label computation for OHLCV bars. Horizon is ``TAIL_DROP_ROWS``…, Compute the 5 label columns per symbol. - label_open_next: open[t+1] -…, DataFrame, ndarray, Unit tests for forward-window label semantics…, Regression: old backward rolling must not match corrected forward labels. (+2 more)

### Community 82 - "TestPlotPerRuleBreakdown"
Cohesion: 0.16
Nodes (11): _make_full_trade_log(), _make_rule_set(), _make_split_logs(), Create a trade log with all columns needed for sharpe computation., Create a minimal rule set with *n* rules., Create a trade log with Rule_Index, Net_PnL, Equity_After columns., Create trade_logs_by_split dict with all three splits populated., One split None, others valid — should not raise. (+3 more)

### Community 83 - "resolve_island_hyperparams"
Cohesion: 0.11
Nodes (21): effective_phase2_val_return_floor_pct(), Direction-aware Phase 2 validation return floor for fitness penalties., Scale integer trade floors by slice size vs full-universe reference., Resolve scaled trade floors and relaxed cross-symbol gates., resolve_island_hyperparams(), scale_trade_floor_by_universe(), Regression tests for anti-overfit / exploration-retune config bundle., test_anti_overfit_config_bundle() (+13 more)

### Community 84 - "evaluator_health.py"
Cohesion: 0.29
Nodes (6): evaluator_health.py — Pure functions for evaluator-failure-mode awareness.…, Read a numeric metric, returning *default* for missing / None / NaN / Inf., Read an integer metric safely., _safe_float(), _safe_int(), Scoring helpers shared across pipeline phases. Re-exports…

### Community 85 - "_build_target"
Cohesion: 0.17
Nodes (12): _build_target(), Build a direction-specific target signal. Default…, Encoding-aware win class: 2 in asymmetric mode, 1 in legacy mode., Encoding-aware loss class: 0 in either mode., Long: max >= entry*(1+TP/100), min > entry*(1-SL/100) → success., Long: both hit but max_before_min==0 → SL first → failure., Long: both hit but max_before_min==1 → TP first → success., Long: neither TP nor SL hit → failure (or neutral in asymmetric mode). (+4 more)

### Community 86 - ".test_accepts_high_val_return_and_trades"
Cohesion: 0.23
Nodes (9): _make_migrant_dict(), _make_mock_receiver(), Migrant with val_return 2.5% and >=15 val trades should be accepted., Migrant with enough val_return but too few val trades should be rejected., Build a minimal migrant dict matching extract_deployable_migrants output., Build a minimal mock Rule_Pool_Generator for filter_migrants_for_cluster tests., Test the migration gate logic with mocked engine evaluations., Migrant with val_return 0.5% should be rejected with reason=val_return. (+1 more)

### Community 87 - "test_gpu_engine_properties.py"
Cohesion: 0.16
Nodes (17): _assert_parity(), _make_engines(), _make_parity_df(), parity_scenario_strategy(), composite, DataFrame, DrawFn, given (+9 more)

### Community 88 - "TestRunLogHandlerLifecycle"
Cohesion: 0.12
Nodes (15): MonkeyPatch, Phase 2 reproducibility default seed., PHASE2_SEED is drawn once at import via get_seed(); set GLOBAL_SEED=42 to fix…, Requirements 1.1, 1.4, 1.5, 1.6, 1.7 — run.log FileHandler is attached, writes…, Count FileHandlers on the root logger pointing to *path*., Patch every phase method on Pipeline_Orchestrator to be a no-op., run.log must exist after run() and contain both separator lines., Root logger must have no extra FileHandlers pointing to run.log after run(). (+7 more)

### Community 89 - "resolve_evolution_floors"
Cohesion: 0.31
Nodes (6): EvolutionFloors, Resolved evolution-time floors (pool admission gates remain strict)., Return stage-aware fitness floors; defaults to global strict knobs. When both…, resolve_evolution_floors(), Stage A soft floors must survive island_hyperparams (cluster two-stage)., TestResolveEvolutionFloorsIslandTwoStage

### Community 90 - "phase5_oos.py"
Cohesion: 0.15
Nodes (13): Joint long/short portfolio simulation. Phase 2 and RB score each direction…, apply_fuzzy_feature_scaling(), fit_fuzzy_feature_scaling(), Any, DataFrame, Train-fitted normalization for ordinal ``ff_*`` fuzzy feature codes. The…, Build a train-only scaling contract for ordinal ``ff_*`` columns., Apply an existing train-fitted scaling contract in place and return *df*. (+5 more)

### Community 91 - "set_purged_wf_reference_rows"
Cohesion: 0.18
Nodes (7): Store full train_new.csv row count after loader prep (split time)., set_purged_wf_reference_rows(), holdout_mode(), purged_mode(), fixture, Tests for purged-WF trade-floor scaling helpers., TestScaleTradeFloor

### Community 92 - "test_phase2_support.py"
Cohesion: 0.17
Nodes (10): deployability_rank_score(), _joint_primary_metric(), Train-only or conservative min(train, val) for ranking / objectives., Conservative return used for objectives, plateau, and archive ranking., Conservative win rate for ranking when f3 uses win rate., Higher is better. Used to rank deployable archive entries and Stage B seeds.…, robust_return_pct(), robust_win_rate_pct() (+2 more)

### Community 93 - "Encoder"
Cohesion: 0.10
Nodes (20): ConfigurationError, decode_chromosome(), encode_condition(), Encoder, get_dont_care(), Exception, ndarray, encoder.py — Encoder Maps gene integer values to fuzzy value names, formats… (+12 more)

### Community 94 - "test_output_writer_properties.py"
Cohesion: 0.13
Nodes (26): parse_symbol_condition(), symbol_conditions.py — Symbol filter parsing (evaluator_v5 parity). Feature…, Parse optional symbol filters. Supported formats: "symbol is 1" "symbol IS 1"…, all_zero_rule_st(), _is_valid_exported_condition(), oversized_rule_set_st(), composite, DrawFn (+18 more)

### Community 95 - "Data_Loader"
Cohesion: 0.06
Nodes (26): Data_Loader, Stateless data loader for the GPU-Fuzzy Trading Pipeline., Exception, Raised when a rule set fails schema validation., ValidationError, _Phase5JSONEncoder, Keep numeric report values numeric instead of stringifying NumPy scalars., _enriched() (+18 more)

### Community 96 - "test_phase2_gpu_throughput.py"
Cohesion: 0.13
Nodes (19): _jax_compute_rule_signals_batch(), _jax_simulate_equity_batch(), Batch rule matching for B chromosomes simultaneously. Returns (B, N) boolean…, Compatibility wrapper for a regular full-row scan., CPU and GPU backtest engine sub-package., get_gpu_backtest_engine_class(), jax_gpu_backtest_available(), Any (+11 more)

### Community 97 - ".decode_chromosome"
Cohesion: 0.26
Nodes (4): See module-level :func:`decode_chromosome`., All 10 signed values decode correctly via decode_chromosome., Chromosome with numpy int32/int64 values should work., TestDecodeChromosome

### Community 98 - ".load_strategies"
Cohesion: 0.19
Nodes (6): Load long.json and short.json via Output_Writer.load_and_validate(). Parameters…, Return only non-empty RB strategies accepted for deployment., Write a valid rule set JSON to path., Override module-level path dicts and return originals (for standalone tests)., TestLoadStrategies, _write_rule_set()

### Community 99 - "gate_positive_good"
Cohesion: 0.28
Nodes (16): _finite_number(), gate_positive_good(), _metric_int(), positive_good_reject_reasons(), PositiveGoodThresholds, Shared metric gates used by Phase 2 admission and RB Governor. The gate is…, Minimum evidence required on train and validation splits., Return stable, machine-readable reasons for a gate rejection. (+8 more)

### Community 100 - "_derive_island_seed"
Cohesion: 0.07
Nodes (39): context_coverage_for_direction(), context_coverage_report(), context_floor_failures(), Any, DataFrame, Shared diagnostics for the mandatory direction-specific context contract., Return coverage diagnostics for every named frame and direction., Return mathematically impossible trade-floor failures for coverage. (+31 more)

### Community 101 - "resolve_phase2_stage_params"
Cohesion: 0.07
Nodes (25): True when Stage A viability is critically low and search has plateaued., _should_inject_diversity_recovery(), _should_viability_recovery(), island_stage_budgets(), IslandStagePlan, StageLabel, Phase 2 two-stage search hyperparameter profiles (exploration vs refinement)., Resolved stage and remaining generation budget for one symbol island. (+17 more)

### Community 102 - "_remove_low_dispersion"
Cohesion: 0.17
Nodes (10): Remove features where more than `threshold` fraction of values are identical.…, _remove_low_dispersion(), DataFrame, Exactly 95% identical → NOT > 0.95 → keep., 96% identical → > 0.95 → remove., Two features with corr > 0.95 → keep higher-scored one., Two uncorrelated features → both kept., Features in different modes are not compared for redundancy. (+2 more)

### Community 103 - "prop_settings"
Cohesion: 0.10
Nodes (31): HealthCheck, prop_settings(), Shared Hypothesis settings for property tests. Set PYTEST_LOW_MEMORY=1 to scale…, Hypothesis settings with optional low-memory example scaling., _make_timestamps(), composite, DataFrame, DrawFn (+23 more)

### Community 104 - "test_crash_fix_properties.py"
Cohesion: 0.10
Nodes (26): _append_xla_flag(), configure_jax_env(), _cuda_package_root(), Path, JAX/XLA runtime configuration — call before the first ``import jax``., Find a pip-installed CUDA toolkit root, if one is available., Append an XLA flag once, preserving explicit user configuration., Configure JAX/XLA runtime for predictable desktop-friendly GPU usage. -… (+18 more)

### Community 105 - "Reporter"
Cohesion: 0.12
Nodes (12): Generates visual and tabular reports for each pipeline phase. All output files…, Reporter, _make_dist_logs_by_split(), _make_dist_trade_log(), _make_per_symbol_metrics(), Create a trade log with all columns needed for distribution/equity plots., Create trade_logs_by_split dict with all three splits populated., A trade log with a single trade should not raise. (+4 more)

### Community 106 - "baselines.py"
Cohesion: 0.28
Nodes (15): _compact(), _drop_one_condition_variants(), _equal_weight(), _evaluate(), evaluate_baselines(), _feature_shuffle(), _fixed_exit(), Any (+7 more)

### Community 107 - "_normalize_for_association"
Cohesion: 0.21
Nodes (8): _normalize_for_association(), Rank-based normalization (robust to outliers like trade_penalty=50)., Verify rank normalization does not crash on degenerate inputs., All-equal objective values should produce valid normalised output with no NaN,…, Single-row input should not crash., Two objectives, all-equal values, should produce valid output. After rank…, Mixed values (not all equal) should still work., TestNormalizeForAssociation

### Community 108 - "_jax_compute_trade_outcomes"
Cohesion: 0.24
Nodes (5): _jax_compute_trade_outcomes(), JAX-jitted vectorized trade outcome computation., Vectorized trade outcome computation for all rows. Mirrors…, Multiple rows processed simultaneously., TestComputeTradeOutcomes

### Community 109 - "_should_post_restart_early_stop_phase2"
Cohesion: 0.29
Nodes (9): Break the epoch when a plateau restart yields no improvement. Independent of…, _should_post_restart_early_stop_phase2(), Unit tests for post-restart no-improvement early stop (Phase 2 runtime)., test_global_disabled_no_stop(), test_global_uses_global_knobs(), test_island_disabled_no_stop(), test_island_streak_at_patience_stops(), test_island_streak_below_patience_no_stop() (+1 more)

### Community 110 - "_build_pool_from_archive"
Cohesion: 0.16
Nodes (17): _archive_direction(), attach_cv_fold_returns_batch(), _build_cpu_archive_engine(), _build_pool_from_archive(), _chromosome_batch(), CvFoldValEvaluator, _evaluate_chromosome(), Evaluate a single chromosome and return (objectives, metrics). objectives =… (+9 more)

### Community 111 - "TestExecutionHealthInGate"
Cohesion: 0.25
Nodes (5): Verify that ``gate_positive_good`` calls ``execution_ok`` when flagged., When require_execution_health=True and execution_ok fails, gate returns False., When require_execution_health=True and execution_ok passes, gate still returns…, When require_execution_health=False, gate ignores raw_signal_count., TestExecutionHealthInGate

### Community 112 - "_make_df"
Cohesion: 0.17
Nodes (10): _make_df(), _make_rule_set(), prepare_test_data should return a DataFrame., When no trades are executed, account_ruined must be False., When no trades are executed, total_return_pct must be 0.0., Create a minimal valid rule set dict., Returned and saved OOS metrics come from the locked strategy., Create a minimal DataFrame with all required columns. (+2 more)

### Community 113 - "TestDeriveValSampleSeed"
Cohesion: 0.12
Nodes (9): AC: Train and validation sampling use distinct RNG seeds by default., _derive_val_sample_seed returns a value different from the input., Same train seed always produces same val seed., Result is in [0, 2**31) so it is a valid random seed., Rule_Pool_Generator stores distinct _sample_seed and _val_sample_seed., When seed=None, val seed is derived from PHASE2_SEED., Using val seed for validation sampling produces a different slice than using…, divmod distribution must still work correctly with val seed. (+1 more)

### Community 114 - "TestHammingThresholdAutoScale"
Cohesion: 0.15
Nodes (10): Verify the max(3, k_active // 5) formula. The formula is applied in…, Replicate the auto-scaling formula., k_active=0 → threshold = max(3, 0//5) = 3., k_active=5 → threshold = max(3, 5//5=1) = 3., k_active=15 → threshold = max(3, 15//5=3) = 3., k_active=20 → threshold = max(3, 20//5=4) = 4., k_active=50 → threshold = max(3, 50//5=10) = 10., k_active=100 → threshold = max(3, 100//5=20) = 20. (+2 more)

### Community 115 - "TestValLeakGate"
Cohesion: 0.20
Nodes (10): C6: Val-derived penalties must be gated behind JOINT_TRAIN_VAL or…, Return standard monkeypatching for clean baseline metrics., Apply base settings with optional overrides., Metrics that trigger no train-side penalties., Val metrics that WOULD trigger penalties if the gate were open., When both JOINT_TRAIN_VAL and VAL_IN_FITNESS_PENALTY are False, val-derived…, Bad val must not set feasibility_violation when gate is closed., When VAL_IN_FITNESS_PENALTY=True, val-derived penalties DO enter objectives. (+2 more)

### Community 116 - "DataFrame"
Cohesion: 0.15
Nodes (6): _make_trade_log(), DataFrame, Dataset with only 1 non-NaN paired row → NaN., Rows must be sorted by abs(train_spearman) descending., Create a minimal trade log DataFrame with Equity_After column., TestPlotEquityCurve

### Community 117 - "_jax_compute_rule_signals"
Cohesion: 0.15
Nodes (11): _jax_compute_rule_signals(), _maybe_jit(), JAX-jitted vectorized rule matching (single chromosome)., Apply ``jit`` if JAX is available, otherwise return *fn* unchanged., Vectorized rule matching: returns (N,) boolean mask of matching rows., All rows match when chromosome equals data_matrix values., No rows match when chromosome differs from data_matrix., Only rows where all active conditions match. (+3 more)

### Community 118 - "TestParetoCollapseWarningGate"
Cohesion: 0.15
Nodes (10): _FakeEngine, AC 4: The default value of the config flag is 5., AC 5: min_pareto_size=0 → warning fires even with small pareto (size=4)., The log message includes 'pareto_size=N' suffix., Fake engine that returns metrics producing a tradeoff between f1 (-sortino) and…, AC 1–5: warning gated on len(pareto_indices) >= config threshold., Run 2-gen evolution and return count of 'Pareto collapse risk' warnings., AC 2: pareto_size=4 < min_pareto_size=5 → no warning fires. (+2 more)

### Community 119 - "CPUBacktestEngine"
Cohesion: 0.06
Nodes (55): CPUBacktestEngine, CPU backtest engine that exactly mirrors evaluator_v5.ipynb's…, CandidateRecord, _diversification_beam(), _eval_cv_fold_returns(), _evaluate_ruleset(), _make_walk_forward_fold_engines(), _monthly_selection_certificate() (+47 more)

### Community 120 - "_compute_stability"
Cohesion: 0.17
Nodes (9): _compute_stability(), Compute cross-symbol stability score. stability = 1 - (std / mean) If mean is 0…, All symbols have same score → std=0 → stability=1., Mean is 0 → stability=0., Single symbol with positive score → stability=1., Single symbol with zero score → stability=0., Very high variance → stability could be negative → clipped to 0., Stability should always be in [0, 1]. (+1 more)

### Community 121 - "phase2_support.py"
Cohesion: 0.14
Nodes (13): effective_min_trade_pool_floor(), effective_pool_min_val_trades(), compute_robust_score(), _evolution_feasibility_floors(), _passes_pool_admission_impl(), _pool_admission_floors(), phase2_support.py — Trade support penalties for Phase 2., Return (train_trade_floor, train_ret_min, val_ret_min, pf_floor,… (+5 more)

### Community 122 - "MonthlyWindowSummary"
Cohesion: 0.06
Nodes (40): _profit_amp_monthly_summary(), Evaluate a ruleset on validation-only chronological windows for the amplifier…, build_monthly_windows(), _datetime_series(), evaluate_rule_set_monthly(), monthly_penalty(), monthly_return_counts_as_good(), MonthlyWindowSummary (+32 more)

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
Cohesion: 0.15
Nodes (9): _mutual_info_discrete_mask(), Return the set of feature names that PASS both stationarity checks. A feature…, Per-column ``discrete_features`` flags for ``mutual_info_classif``. Only…, _stationarity_filter(), Unit tests for gpu_fuzzy_trader.features.selector.Feature_Selector Tests cover:…, TestAlignFeatureArray, TestMutualInfoDiscreteMask, TestReduceOverlap (+1 more)

### Community 127 - ".run"
Cohesion: 0.13
Nodes (10): DataFrame, Run out-of-sample evaluation. Parameters ---------- allowed_directions :…, Prepare test data using Data_Loader.load_dataset(). Applies the same…, Load prepared train, validation, and test datasets., Load and validate a strictly newer, untouched forward period., Load selected features for a direction when available., Remove only known Phase 5 artifacts from the active report root., Evaluate a single strategy on the test DataFrame. Returns ------- metrics :… (+2 more)

### Community 128 - "stratification_scenario_strategy"
Cohesion: 0.24
Nodes (15): dataset_with_features_strategy(), metrics_strategy(), composite, DataFrame, DrawFn, Generate a trade log DataFrame with 2–50 rows. Parameters ---------- n_rules:…, Generate a metrics dict with reasonable float values. Keys: win_rate,…, Generate a dataset DataFrame with 10–100 rows. Parameters ----------… (+7 more)

### Community 129 - "TestSpearmanSignConsistency"
Cohesion: 0.18
Nodes (9): ndarray, Tests for _check_spearman_sign_consistency, including the val_df check., Build a minimal DataFrame without 'symbol' to avoid symbol-based folding., AC1: Train all positive, val negative → feature blacklisted., AC2: Train all positive, val positive → feature still kept., AC3: Train all positive, tiny |val_rho| < min_abs_corr → feature still kept., AC4: val_df=None → pre-task-8 behavior (mixed train signs still blacklisted)., AC5: val_df without label_close_288 column → val check skipped (no crash). (+1 more)

### Community 130 - ".encode_condition"
Cohesion: 0.18
Nodes (4): See module-level :func:`encode_condition`., TestEncodeConditionBinary, TestEncodeConditionErrors, TestEncodeConditionTernary

### Community 131 - "log_memory_rss"
Cohesion: 0.19
Nodes (11): log_memory_rss(), Optional process memory logging (set LOG_MEMORY=1)., Log current process RSS in MiB when LOG_MEMORY=1., Force GC and clear JAX compilation caches between Phase 2 directions. Releases:…, release_phase2_resources(), Release GPU engines and slim in-memory data between island epochs., Drop engine and sampled data to free RAM before the next direction., Tests for optional memory logging helpers. (+3 more)

### Community 132 - "_derive_epoch_seed"
Cohesion: 0.16
Nodes (10): _derive_epoch_seed(), Return a deterministic per-epoch seed derived from *base_seed* + epoch. Used by…, Re-sample training data with a per-epoch rotated window. Each epoch gets a…, An unknown PHASE2_PER_EPOCH_WINDOW_SEED_MODE raises ValueError., Deterministic seed derivation for per-epoch windows., Same (base_seed, epoch_idx) produces the same seed., Different epoch indices produce different seeds., None base seed returns None. (+2 more)

### Community 133 - "_apply_colab_gpu_defaults"
Cohesion: 0.33
Nodes (6): _apply_colab_gpu_defaults(), is_colab_runtime(), True when running on Google Colab (/content runtime)., Colab T4 optimization for Phase 2 runs., MonkeyPatch, test_colab_defaults_apply_when_content_exists()

### Community 134 - "loader.py"
Cohesion: 0.20
Nodes (15): downcast_numeric_df(), prune_train_columns(), DataFrame, df_slim.py — Reduce DataFrame memory for backtest engines. Keeps only columns…, Downcast label and feature columns to float32; symbol to category (inplace)., Return a copy with only meta, label, and requested feature columns. Parameters…, Drop unused feature columns from the full training split after Phase 1., slim_backtest_df() (+7 more)

### Community 135 - "test_evaluator_health.py"
Cohesion: 0.17
Nodes (7): Unit tests for ``evaluator_health_penalty`` and ``execution_ok`` (Task 4).…, Verify that evaluator health penalty is subtracted from combo score., When evaluator_health_penalty returns > 0, the combo score is lower., When evaluator health is fine, no penalty is applied., Both public functions are importable from the module., TestHealthPenaltyWiredIntoRB, TestModuleImportable

### Community 136 - "OOS_Evaluator"
Cohesion: 0.27
Nodes (4): OOS_Evaluator, Out-of-sample evaluator for Phase 5. Loads the final long/short strategies,…, TestOOSEvaluatorInit, TestSaveReport

### Community 137 - "compute_phase2_objectives_from_metrics"
Cohesion: 0.04
Nodes (39): compute_phase2_objectives_from_metrics(), Penalty for weak cross-symbol robustness on one split., Build Phase 2 minimisation objectives from precomputed train/val metrics.…, _symbol_robustness_penalty(), compute_support_penalty_and_specialist(), Support penalty from train metrics. Returns (penalty, False, -1)., True when val-derived feasibility penalties belong in NSGA-III fitness., _val_terms_in_fitness() (+31 more)

### Community 138 - "_raw_feasibility_violation_score"
Cohesion: 0.18
Nodes (8): feasibility_violation_score(), Compute violation score using evolution PF floors (1.05) during NSGA-III…, Non-negative violation score; 0 means the rule meets deployability floors. Used…, _raw_feasibility_violation_score(), Regression: train=90%/val=10% (gap=80pp >> 16pp threshold) must produce a non-…, Val terms in _raw_feasibility_violation_score respect include_val flag., TestDeployabilityHelpers, TestTrainOnlyFitnessValGating

### Community 139 - ".load_pool"
Cohesion: 0.12
Nodes (12): _condition_feature_names(), _filter_compatible_previous_pool(), _pool_entry_passes_admission(), Check stored train/val metrics on a pool JSON entry., Extract feature names from textual conditions like: "[feat] IS Value". Returns…, Keep only previous pool entries compatible with current feature selection.…, Load existing pool if valid, return None if missing., Merge local pool, symbol archive, and shared archive (dominant seeds). (+4 more)

### Community 140 - "test_phase5_oos_properties.py"
Cohesion: 0.19
Nodes (12): _make_timestamps(), composite, DataFrame, DrawFn, given, Timestamp, Property-based tests for gpu_fuzzy_trader.phases.phase5_oos.OOS_Evaluator…, **Property 27: Test Data Preparation Consistency** **Validates: Requirements… (+4 more)

### Community 141 - "test_phase2_island_scheduler.py"
Cohesion: 0.09
Nodes (23): compute_cluster_generation_budgets(), Resolve per-island generation budgets. By default each island receives the full…, Check if an epoch should be skipped due to small remaining budget. Engine…, _should_skip_epoch(), _MockGenerator, parametrize, Unit tests for cluster island scheduler budget math and epoch guard., Minimal mock for Rule_Pool_Generator used in epoch guard tests. (+15 more)

### Community 142 - "TestIslandSchedulerGlobalMode"
Cohesion: 0.17
Nodes (7): Unit tests for island scheduler global-mode safety. Acceptance criteria…, AC-T1.4: Global mode must never reach migration code., When PHASE2_ISLAND_MODE='global', _run_cluster_islands is not called., Verify the migration guard would not be reached in global mode., The top-level dispatch should only call run_cluster_phase2 in cluster mode., In global mode, the lazy import of extract_deployable_migrants never fires., TestIslandSchedulerGlobalMode

### Community 143 - "TestEndToEndRotation"
Cohesion: 0.18
Nodes (7): fixture, Integration-style tests with a mocked Rule_Pool_Generator., Patch config for rotation and create a generator with minimal setup., When rotation is enabled, _cached_scoped_train_df is stored., After resample_train_for_epoch, the cached slim train changes., Same epoch_idx produces identical cached slim train., TestEndToEndRotation

### Community 144 - "TestRefreshObjectivesOnResumeGate"
Cohesion: 0.23
Nodes (7): Task-10: gate cache refresh on PHASE2_PER_EPOCH_WINDOW_ROTATION. Verifies the…, Evaluate the gate expression as it appears in the source., PHASE2_PER_EPOCH_WINDOW_ROTATION=True (default): refresh = not first_epoch →…, PHASE2_PER_EPOCH_WINDOW_ROTATION=True (default): refresh = not first_epoch →…, PHASE2_PER_EPOCH_WINDOW_ROTATION=False (legacy): refresh is False even on…, PHASE2_PER_EPOCH_WINDOW_ROTATION=False (legacy): refresh is False on first…, TestRefreshObjectivesOnResumeGate

### Community 145 - "conftest.py"
Cohesion: 0.22
Nodes (10): Config, FixtureRequest, Item, _close_matplotlib_figures(), _low_memory_cleanup(), fixture, pytest_collection_modifyitems(), pytest_configure() (+2 more)

### Community 146 - ".get_dont_care"
Cohesion: 0.17
Nodes (7): See module-level :func:`get_dont_care`., **Property 9: Don't-Care Sentinel Correctness — encode_condition raises**…, **Property 9: Don't-Care Sentinel Correctness — all-dont_care → empty output**…, test_property_9b_encode_condition_raises_for_dont_care(), test_property_9f_all_dont_care_chromosome_returns_empty(), Static methods should be callable on the class itself., TestGetDontCare

### Community 147 - ".skip_if_valid"
Cohesion: 0.33
Nodes (3): Check if output files exist and are valid. Returns ------- dict[str,…, fixture, TestSkipIfValid

### Community 148 - "_validate_schema"
Cohesion: 0.31
Nodes (3): Validate the structure of a loaded feature selection JSON. Raises ValueError if…, _validate_schema(), TestValidateSchema

### Community 149 - "splitter.py"
Cohesion: 0.24
Nodes (12): _chronological_half_split(), _holdout_embargo_split(), _purged_walk_forward_split(), DataFrame, data/splitter.py — Data_Splitter Per-symbol chronological split for Phase 2,…, Per-symbol chronological first or second half of *df*. ``purge_rows`` is…, Split validation into purged fitness and selection halves per symbol. The gap…, Return whether cached internal halves match the purged geometry. (+4 more)

### Community 150 - "TestSplitRatio"
Cohesion: 0.21
Nodes (7): Compute expected train/val per symbol under holdout+embargo split., floor(N * HOLDOUT_TRAIN_FRACTION) rows go to train., Remaining rows after embargo gap go to validation., For N=101: floor(101 * HOLDOUT_TRAIN_FRACTION) used, not round., train + validation + embargo dropped == total rows., train + val + embargo_dropped == total for each symbol., TestSplitRatio

### Community 151 - "passes_pool_admission_gate"
Cohesion: 0.23
Nodes (6): passes_pool_admission_gate(), Hard gate for Phase 2 pool/archive on merged holdout metrics. When…, MonkeyPatch, Multi-gate failure: low trades, low return, low pf, and val_ret <= 0., When PHASE2_JOINT_TRAIN_VAL=True, the f4 gate must use min(train_f4, val_f4) —…, TestPoolAdmissionGate

### Community 152 - "BFS and DFS Graph Traversal"
Cohesion: 0.20
Nodes (10): CLAUDE.md Graphify Integration, Constrained Query Expansion, BFS and DFS Graph Traversal, Graph Work Memory, Existing Graph Fast Path, CLAUDE.md Graphify Integration, Constrained Query Expansion, BFS and DFS Graph Traversal (+2 more)

### Community 153 - "TestGPUCPUNumericalParity"
Cohesion: 0.10
Nodes (15): ndarray, simulate_rule_set (delegated to CPU) is exactly identical to CPU engine., GPU backtest engine must handle padded chunking correctly., CPU backtest engine must support simulate_rule_batch with correct metrics., Compare GPU and CPU engine for 10 random chromosomes (Task 2 acceptance…, Build a multi-symbol DataFrame with mixed TP/SL/time-exit outcomes., Generate random chromosomes for binary + signed features., GPU engine results must match CPU engine within specified tolerances for 10… (+7 more)

### Community 154 - "load_cached_split_if_fresh"
Cohesion: 0.31
Nodes (6): load_cached_split_if_fresh(), Load cached split parquets when they are newer than the source CSV. Validates…, load_cv_folds_manifest(), Any, Load manifest if present., TestLoadCachedSplitIfFresh

### Community 155 - "TestSparsePositiveMode"
Cohesion: 0.22
Nodes (5): All non-negative, zero_ratio > 0.3 → sparse_positive., All zeros: zero_ratio = 1.0 > 0.3, min = 0 → sparse_positive., zero_ratio just above 0.3 → sparse_positive., zero_ratio must be computed on the full series, not just non-NaN., TestSparsePositiveMode

### Community 156 - "TestZeroRatioBoundary"
Cohesion: 0.22
Nodes (5): Exactly 30% zeros with non-negative values → positive (not sparse_positive)., 31% zeros with non-negative values → sparse_positive., Exactly 30% zeros with negative values → signed (not sparse_signed)., Just above 30% zeros with negative values → sparse_signed., TestZeroRatioBoundary

### Community 157 - "_validate_pool_schema"
Cohesion: 0.36
Nodes (3): Validate the structure of a loaded pool JSON. Raises ValueError if the schema…, _validate_pool_schema(), TestValidatePoolSchema

### Community 158 - "TestGlobalMetricsCacheClearing"
Cohesion: 0.27
Nodes (6): When seeded_keys is empty, nothing is removed., When all keys are seeded, cache becomes empty., Verify that only seeded keys are removed from the global cache., Create a deterministic chromosome key., Cache entries matching seeded_keys are removed; non-matching survive., TestGlobalMetricsCacheClearing

### Community 159 - "cpu_engine.py"
Cohesion: 0.21
Nodes (9): precompute_release_indices(), cpu_engine.py — CPUBacktestEngine Exact Python/NumPy replication of…, Sort entries for v5 capital allocation (timestamp, rule, symbol, row)., For each row, find the row index where symbol_bar_index + max_hold_candles is…, _sort_entries_by_allocation_priority(), _mini_df(), DataFrame, Parity between standalone release_index precompute and CPUBacktestEngine. (+1 more)

### Community 161 - "Phase2StageParams"
Cohesion: 0.05
Nodes (49): True for cluster/orphan scoped runs rather than the global path., scoped_island_profile(), _diversity_recovery_min_unique_ratio(), Resolve the plateau patience value based on profile and stage. Cluster/orphan…, Earliest gen for plateau stop. Island epochs (~20 gens) and scaled Stage A…, _resolve_plateau_min_generation(), _resolve_plateau_patience(), _should_early_stop_phase2() (+41 more)

### Community 162 - "TestNaNHandling"
Cohesion: 0.29
Nodes (4): All-NaN series: unique_vals = [], n_unique = 0 ≤ 2, set() ⊆ {0,1} → binary., NaN is not == 0, so it does not inflate zero_ratio., NaN should not push zero_ratio above threshold., TestNaNHandling

### Community 163 - "TestSparseSignedMode"
Cohesion: 0.33
Nodes (3): Has negative values, zero_ratio > 0.3 → sparse_signed., NaN does not count as zero; zero_ratio on full series., TestSparseSignedMode

### Community 164 - "TestPlotPhase2Pnl"
Cohesion: 0.23
Nodes (4): _make_pnl_history(), History entries with missing keys should not raise., Create a minimal Phase 2 history list with PnL fields., TestPlotPhase2Pnl

### Community 165 - "TestMigrationSeedFraction"
Cohesion: 0.20
Nodes (6): Verify that migrant injection uses PHASE2_MIGRATION_SEED_FRACTION, not ARCHIVE., PHASE2_MIGRATION_SEED_FRACTION defaults to 0.10., Ensure the migration fraction is not the same as archive fraction., Simulate the run_epoch migrant path and assert local_cap uses migration…, Simulate the new run_epoch logic: migrant entries are capped by migration…, TestMigrationSeedFraction

### Community 166 - "test_phase2_rule_pool.py"
Cohesion: 0.06
Nodes (28): _crowding_distance(), _dominates(), _non_dominated_sort(), _pareto_sortino_stats(), _phenotype_bucket_key(), Return True if solution *a* dominates *b* (all objectives ≤, at least one <)., NSGA-II non-dominated sorting. Parameters ---------- objectives : np.ndarray…, Compute crowding distance for solutions in *front*. Parameters ----------… (+20 more)

### Community 167 - "strategy_id"
Cohesion: 0.24
Nodes (10): context_contract_digest(), Return a stable hash of the static contract and fitted enrichment., feature_conditions_only(), phase2_rule_id(), Any, Stable provenance identifiers for Phase 2 feature rules. The RB Governor may…, Return normalized non-symbol, non-context conditions in order. Symbol scope and…, Hash the immutable Phase 2 feature logic and its discovery scope. (+2 more)

### Community 168 - "TestPoolAdmissionOverfitRatioGate"
Cohesion: 0.24
Nodes (7): MonkeyPatch, Tests for the hard overfit ratio gate in pool admission. The ratio gate rejects…, train=15%/val=4% (3.75× ratio, gap=11pp < 16pp) is REJECTED by the ratio gate…, train=15%/val=10% (1.5× ratio) is ADMITTED., train/val ≈2.5× is ADMITTED when under OVERFIT_RATIO_FLOOR and gap gate., With PHASE2_OVERFIT_RATIO_FLOOR=0.0, the ratio gate is disabled and the high-…, TestPoolAdmissionOverfitRatioGate

### Community 169 - "test_phase2_batch_evaluator_parity.py"
Cohesion: 0.40
Nodes (5): _backtest_df(), DataFrame, Regression tests for evaluator-facing Phase 2 chromosome semantics., Search fitness must use the same fuzzy class as RB/Phase 5 evaluation., test_batch_chromosome_signals_match_decoded_rule_conditions()

### Community 170 - "_plateau_diversity_restart"
Cohesion: 0.24
Nodes (7): _plateau_diversity_restart(), Reinit a fraction of the population on plateau, preserving Pareto elite. Keeps…, Reinitialised slots have objectives=np.inf and metrics_cache={}., Even with large Pareto front, at most 10 elite are preserved., Direct unit tests for the _plateau_diversity_restart helper., Pareto elite chromosomes survive the restart., TestPlateauDiversityRestart

### Community 172 - "TestHallOfFameTrim"
Cohesion: 0.25
Nodes (5): Verify that hall_of_fame is capped to PHASE2_HOF_EPOCH_CARRYOVER=10., HoF with >10 entries is trimmed to exactly 10 oldest entries., HoF with <=10 entries is left unchanged., Empty HoF does not crash during trim logic., TestHallOfFameTrim

### Community 173 - "trade_support_penalty"
Cohesion: 0.32
Nodes (5): Support penalty. Returns ------- penalty : float is_specialist : bool (always…, Legacy graduated penalty., _static_support_penalty(), trade_support_penalty(), TestTradeSupportPenaltyStatic

### Community 174 - "TestNClustersDefined"
Cohesion: 0.33
Nodes (4): AC: n_clusters is assigned inside _run_cluster_islands so the migration guard…, n_clusters must be assigned in _run_cluster_islands for the migration guard at…, The migration guard must reference n_clusters., TestNClustersDefined

### Community 175 - "test_nsga3_selection.py"
Cohesion: 0.40
Nodes (3): skipif, Unit tests for NSGA-III environmental selection., TestNsga3Selection

### Community 176 - "test_gpu_engine_import_does_not_crash_on_jax_failure"
Cohesion: 0.33
Nodes (6): parametrize, Tests for the lazy-JAX failure path in gpu_engine.py. These tests verify that…, Verify importing gpu_engine.py handles various JAX failure modes gracefully., Build a subprocess script that simulates JAX import failure., _subprocess_script(), test_gpu_engine_import_does_not_crash_on_jax_failure()

### Community 177 - "test_jax_compat.py"
Cohesion: 0.29
Nodes (5): parametrize, Tests for JAX / GPU engine availability detection., Package init must not crash when gpu_engine import fails., test_cpu_engine_import_without_jax(), test_get_gpu_backtest_engine_class_returns_none_on_import_error()

### Community 178 - "_downsample_chronological"
Cohesion: 0.50
Nodes (4): _downsample_chronological(), Preserve per-symbol time order required by exposure/release simulation., Pick a contiguous chronological slice of *n_rows* from *df*. Critical for…, _sort_chronological()

### Community 179 - "TestMigrationEnabledByDefault"
Cohesion: 0.50
Nodes (3): Verify migration default for multi-symbol cluster islands (enabled)., Multi-symbol clusters ship with migration enabled., TestMigrationEnabledByDefault

### Community 180 - "TestEvolutionFeasibilityFloors"
Cohesion: 0.50
Nodes (3): Tests for _evolution_feasibility_floors — EVOLUTION PF 1.0 vs ADMISSION PF 1.15., _raw_feasibility_violation_score uses EVOLUTION PF (1.0) so a rule with PF…, TestEvolutionFeasibilityFloors

### Community 189 - "opencode.json"
Cohesion: 0.50
Nodes (3): plugin, $schema, .opencode/plugins/graphify.js

## Knowledge Gaps
- **35 isolated node(s):** `$schema`, `.opencode/plugins/graphify.js`, `gpu-fuzzy-trader`, `URL Ingestion`, `Folder Watcher` (+30 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **11 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `CPUBacktestEngine` connect `CPUBacktestEngine` to `rb_governor.py`, `_make_engine`, `_valid_metrics`, `Rule_Pool_Generator`, `OOS_Evaluator`, `phase2_rule_pool.py`, `_make_df`, `_apply_monthly_admission_gate`, `test_cpu_engine_properties.py`, `TestGPUCPUNumericalParity`, `_compose_ruleset`, `nested_walk_forward.py`, `cpu_engine.py`, `writer.py`, `test_phase2_batch_evaluator_parity.py`, `gpu_engine.py`, `test_certificate_first_selection.py`, `test_jax_compat.py`, `_compute_rule_signal_mask`, `GPUBacktestEngine`, `get_normalized_symbol_array`, `ValueError`, `sample_df_for_phase2`, `_apply_dynamic_rule`, `test_cpu_engine.py`, `run_rb_governor_pipeline`, `test_gpu_engine.py`, `phase2_island_scheduler.py`, `test_gpu_engine_properties.py`, `phase5_oos.py`, `Data_Loader`, `test_phase2_gpu_throughput.py`, `baselines.py`, `_jax_compute_trade_outcomes`, `_build_pool_from_archive`, `_jax_compute_rule_signals`, `MonthlyWindowSummary`, `._engine`, `.run`?**
  _High betweenness centrality (0.131) - this node is a cross-community bridge._
- **Why does `Reporter` connect `Reporter` to `.run`, `TestWriteSpearmanCorrelationReport`, `TestPlotPhase2Pnl`, `TestEquityCurveDateAxis`, `prop_settings`, `Rule_Pool_Generator`, `OOS_Evaluator`, `phase2_rule_pool.py`, `TestWriteStrategyEvaluationTable`, `._ensure_dir`, `_build_pool_from_archive`, `TestPlotPerRuleBreakdown`, `DataFrame`, `TestPlotPhase2Metrics`, `test_reporter.py`, `phase5_oos.py`, `Data_Loader`, `.run`?**
  _High betweenness centrality (0.110) - this node is a cross-community bridge._
- **Why does `Rule_Pool_Generator` connect `Rule_Pool_Generator` to `.run`, `log_memory_rss`, `_derive_epoch_seed`, `Pipeline_Orchestrator`, `compute_phase2_objectives_from_metrics`, `phase2_rule_pool.py`, `.load_pool`, `test_phase2_island_scheduler.py`, `_apply_monthly_admission_gate`, `TestRefreshObjectivesOnResumeGate`, `_make_train_df`, `TestEndToEndRotation`, `_validate_pool_schema`, `test_phase2_rule_pool.py`, `test_phase2_window_rotation.py`, `TestPoolAdmissionOverfitRatioGate`, `_gpu_runtime.py`, `run_pipeline.py`, `TestNClustersDefined`, `_init_population`, `test_phase2_rule_pool_properties.py`, `sample_df_for_phase2`, `TestRulePoolGeneratorRun`, `test_plateau_state_leak.py`, `test_crash_fix_and_run_logging.py`, `TestDeferredWarmup`, `phase2_island_scheduler.py`, `TestRunLogHandlerLifecycle`, `_derive_island_seed`, `Reporter`, `TestDeriveValSampleSeed`, `TestValLeakGate`, `CPUBacktestEngine`?**
  _High betweenness centrality (0.108) - this node is a cross-community bridge._
- **Are the 15 inferred relationships involving `Reporter` (e.g. with `CvFoldValEvaluator` and `Rule_Pool_Generator`) actually correct?**
  _`Reporter` has 15 INFERRED edges - model-reasoned connections that need verification._
- **Are the 64 inferred relationships involving `CPUBacktestEngine` (e.g. with `GPUBacktestEngine` and `JointPortfolioEngine`) actually correct?**
  _`CPUBacktestEngine` has 64 INFERRED edges - model-reasoned connections that need verification._
- **Are the 65 inferred relationships involving `Rule_Pool_Generator` (e.g. with `CPUBacktestEngine` and `Reporter`) actually correct?**
  _`Rule_Pool_Generator` has 65 INFERRED edges - model-reasoned connections that need verification._
- **Are the 31 inferred relationships involving `Output_Writer` (e.g. with `OOS_Evaluator` and `_Phase5JSONEncoder`) actually correct?**
  _`Output_Writer` has 31 INFERRED edges - model-reasoned connections that need verification._