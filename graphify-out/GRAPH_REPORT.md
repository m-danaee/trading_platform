# Graph Report - trading_platform  (2026-08-18)

## Corpus Check
- 201 files · ~292,105 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 5104 nodes · 11728 edges · 186 communities (175 shown, 11 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 548 edges (avg confidence: 0.58)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `02ab0b20`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- evox_runner.py
- test_evox_runner.py
- barrier_column_names
- _make_engine
- Pipeline_Orchestrator
- compute_phase2_objectives_from_metrics
- TestRulePoolGeneratorRun
- _score_metrics
- _get_dont_cares
- _make_df
- _should_plateau_early_stop_phase2
- test_phase2_val_sim_interval.py
- splitter.py
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
- test_phase2_window_rotation.py
- research_integrity.py
- _loader_from_rows
- TestEndToEndRotation
- ndarray
- _gpu_runtime.py
- ._ensure_dir
- compose_hierarchical_signals
- test_certificate_first_selection.py
- non_dominated_sort
- optuna_search.py
- Hierarchical Multi-Timeframe Rule Discovery System Specification
- test_phase2_rule_pool_properties.py
- _compute_rule_signal_mask
- validate_config
- build_hybrid_symbol_clusters
- test_reporter.py
- test_phase2_rule_pool.py
- phase2_rule_pool.py
- _write_and_reload
- GPUBacktestEngine
- ._simulate_rule_set_entries
- trade_support_penalty
- test_phase2_use_gpu_flag.py
- rolling_cv.py
- .compose
- test_plateau_state_leak.py
- TestSeedDirectionUniqueness
- dashboard.py
- DataFrame
- test_data_loader_properties.py
- ValueError
- _m
- _apply_dynamic_rule
- test_cpu_engine.py
- _build_entries_from_rule_set
- execution_ok
- TestWriteStrategyEvaluationTable
- test_gpu_engine.py
- TestRunLogHandlerLifecycle
- test_output_writer_properties.py
- test_feature_selector_properties.py
- .save_archive
- _symbol_specialized_variants
- TestPlotPerRuleBreakdown
- resolve_island_hyperparams
- test_phase2_island_scheduler.py
- _build_target
- filter_migrants_for_cluster
- test_gpu_engine_properties.py
- TestGPUCPUNumericalParity
- build_complete_higher_bars
- discovery.py
- set_purged_wf_reference_rows
- TestMigrationSeedFraction
- .encode_condition
- Feature_Detector
- gpu_engine.py
- mtf/__init__.py
- test_mtf_pipeline_integration.py
- .load_strategies
- gate_positive_good
- phase2_island_scheduler.py
- loader.py
- _remove_low_dispersion
- _make_df
- prop_settings
- TestPlotDistributionAndEquity
- baselines.py
- TestWriteSpearmanCorrelationReport
- _jax_compute_trade_outcomes
- _should_post_restart_early_stop_phase2
- test_data_splitter_properties.py
- nested_walk_forward.py
- _validate_pool_schema
- OOS_Evaluator
- TestHammingThresholdAutoScale
- .decode_chromosome
- ValidationError
- _jax_compute_rule_signals
- TestParetoCollapseWarningGate
- test_rb_governor_tail_holdout.py
- _compute_stability
- passes_pool_admission_gate
- MonthlyWindowSummary
- ._engine
- test_rb_concentration_tail_fail_closed.py
- constrained_non_dominated_sort
- test_feature_selector.py
- _derive_epoch_seed
- stratification_scenario_strategy
- TestSpearmanSignConsistency
- compute_labels
- log_memory_rss
- _evaluate_chromosome
- _apply_colab_gpu_defaults
- test_data_loader.py
- _build_rule_signal_mask
- .get_dont_care
- resolve_phase2_stage_params
- test_phase2_gpu_throughput.py
- CPUBacktestEngine
- ._prune_splits_after_phase1
- TestSparsePositiveMode
- phase2_support.py
- TestZeroRatioBoundary
- TestRefreshObjectivesOnResumeGate
- conftest.py
- .load_and_validate
- .skip_if_valid
- _validate_schema
- test_directional_evaluator.py
- Global Constraints
- test_phase2_offspring_batch.py
- BFS and DFS Graph Traversal
- ConfigError
- _make_csv
- TestFeatureNaNFill
- .run
- TestNaNHandling
- TestGlobalMetricsCacheClearing
- TestHealthPenaltyWiredIntoRB
- TestSparseSignedMode
- TestContextEntryPaths
- _build_cpu_archive_engine
- TestEvalCvFoldReturns
- test_property_27_test_data_preparation_consistency
- test_evaluator_health.py
- run_rb_governor_pipeline
- test_config_additions.py
- run_pipeline.py
- TestHallOfFameTrim
- cpu_engine.py
- valid_test_csv_dataframe
- test_gpu_engine_import_does_not_crash_on_jax_failure
- test_jax_compat.py
- TestExecutionHealthInGate
- test_phase2_plateau_restart.py
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
2. `CPUBacktestEngine` - 151 edges
3. `Rule_Pool_Generator` - 137 edges
4. `Pipeline_Orchestrator` - 87 edges
5. `Output_Writer` - 86 edges
6. `prop_settings()` - 79 edges
7. `_run_nsga3()` - 59 edges
8. `compute_phase2_objectives_from_metrics()` - 59 edges
9. `OOS_Evaluator` - 59 edges
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

## Communities (186 total, 11 thin omitted)

### Community 0 - "evox_runner.py"
Cohesion: 0.04
Nodes (112): _assign_eval_result(), _behavior_descriptor(), _binary_tournament_pick(), _build_diversity_reference(), _build_rank_and_crowding(), _constraint_violations(), _count_deployable_preview(), _count_pop_viable() (+104 more)

### Community 1 - "test_evox_runner.py"
Cohesion: 0.04
Nodes (58): _inherit_val_metrics_from_global_cache(), _inject_diversity_recovery(), Phase2EvolutionState, StageLabel, Replace a fraction of the population with fresh or archive-mutated seeds., Copy val_* from global cache for identical chromosomes when val is skipped.…, Evolve one island epoch and return updated resumable state., Run Phase 2 NSGA-III evolution. Returns (pareto_pool, history) or state. (+50 more)

### Community 2 - "barrier_column_names"
Cohesion: 0.19
Nodes (16): attach_barrier_outcomes(), barrier_column_names(), configured_barrier_pairs(), _first_touch_for_symbol(), _first_touch_for_symbol_numba(), _first_touch_for_symbol_python(), _number_token(), DataFrame (+8 more)

### Community 3 - "_make_engine"
Cohesion: 0.09
Nodes (19): _make_df(), _make_engine(), Simulate catastrophic losses to trigger account ruin. With min_288=0…, All TP trades → win_rate = 100%., No losing trades → profit_factor = 99.0., Build a minimal DataFrame for testing., position_notional = equity * capital_pct/100 * leverage when exposure is free., position_notional is capped when exposure is nearly full. (+11 more)

### Community 4 - "Pipeline_Orchestrator"
Cohesion: 0.06
Nodes (55): FileHandler, _git_commit_id(), _log_phase_entry(), _log_pipeline_config(), main(), _merge_mtf_lwc_runtime_columns(), _now_iso(), Pipeline_Orchestrator (+47 more)

### Community 5 - "compute_phase2_objectives_from_metrics"
Cohesion: 0.04
Nodes (45): compute_phase2_objectives_from_metrics(), Penalty for weak cross-symbol robustness on one split., Build Phase 2 minimisation objectives from precomputed train/val metrics.…, _symbol_robustness_penalty(), True when val-derived feasibility penalties belong in NSGA-III fitness., _val_terms_in_fitness(), f3 uses robust return = min(train_return, val_return) when…, Blind-spot regression: overfit_gap_penalty must fire when val_ret <= 0. (+37 more)

### Community 6 - "TestRulePoolGeneratorRun"
Cohesion: 0.14
Nodes (8): Integration tests using tiny population and generation counts., Verify that Rule_Pool_Generator's persistent self._rng advances across multiple…, After two run_epoch() calls, the internal RNG state must differ from the…, The RNG should produce *different* sequences in two consecutive run_epoch()…, Two generators with different seeds must have different RNG state., Rule_Pool_Generator must initialize self._rng as a Generator., TestRulePoolGeneratorRng, TestRulePoolGeneratorRun

### Community 7 - "_score_metrics"
Cohesion: 0.06
Nodes (43): _combined_return_score(), _evaluate_ruleset(), _evaluator_health_penalty(), _i(), _optimize_risk(), Return (ok, bonus, penalty) for the desired train-valid balance shape. In…, Penalize evaluator_v5 execution problems: too many skipped signals, low…, Dominant objective: return/DD with train-valid balance, plus CV-fold… (+35 more)

### Community 8 - "_get_dont_cares"
Cohesion: 0.06
Nodes (30): _count_active_conditions(), _get_dont_cares(), Return array of dont_care sentinels for each feature., Count active rule conditions (sparse slots or dense dont_care encoding)., _feature_infos_with_scores(), TestStratifiedInitPopulation, _is_all_inactive_sparse(), _make_feature_infos() (+22 more)

### Community 9 - "_make_df"
Cohesion: 0.08
Nodes (21): _make_df(), _make_engine(), MonkeyPatch, Chromosome positions must follow feature_modes insertion order., Exact CPU re-evaluation must interpret genes like the GPU path., Chromosome that matches nothing returns 0 executed trades., Zero-signal chunks should use reject metrics without scanning., PHASE2_SKIP_ZERO_SIGNAL_SCAN must match always-scan metrics. (+13 more)

### Community 10 - "_should_plateau_early_stop_phase2"
Cohesion: 0.05
Nodes (45): True for cluster/orphan scoped runs rather than the global path., scoped_island_profile(), Resolve the plateau patience value based on profile and stage. Cluster/orphan…, Earliest gen for plateau stop. Island epochs (~20 gens) and scaled Stage A…, _resolve_plateau_min_generation(), _resolve_plateau_patience(), _should_early_stop_phase2(), _should_plateau_early_stop_phase2() (+37 more)

### Community 11 - "test_phase2_val_sim_interval.py"
Cohesion: 0.15
Nodes (14): CountingEngine, Unit tests for periodic val simulation (Phase 2 runtime A2)., PHASE2_VAL_SIM_INTERVAL=1 preserves original behaviour (val every gen)., Direct unit test of _should_run_val_this_gen with interval=3. With interval=3,…, Integration test: with interval=3 and 13-gen epoch, val runs on gens 0, 3, 6,…, Val metrics for a chromosome are deterministic when val is skipped. With…, With PHASE2_VAL_SIM_INTERVAL=2, val sim runs on interval gens (0,2,4) + last…, Validation-dependent fitness is evaluated consistently every gen. (+6 more)

### Community 12 - "splitter.py"
Cohesion: 0.08
Nodes (33): Number of leading per-symbol rows belonging to the training prefix. Shared by…, train_prefix_row_count(), _chronological_half_split(), _file_sha256(), _holdout_embargo_split(), load_cached_split_if_fresh(), _purged_walk_forward_split(), DataFrame (+25 more)

### Community 13 - "Data_Splitter"
Cohesion: 0.07
Nodes (29): Data_Loader, Stateless data loader for the GPU-Fuzzy Trading Pipeline., Data_Splitter, Chronological train/validation splitter., _NumpyJSONEncoder, JSON encoder that converts NumPy/JAX scalar and array types to native Python…, _patch_split_paths(), Module-level function should produce same result as class method. (+21 more)

### Community 14 - "test_run_pipeline.py"
Cohesion: 0.08
Nodes (46): _context_coverage_for_direction(), _context_coverage_preflight(), _context_coverage_report(), context_floor_failures(), _context_island_sample_report(), Temporarily rebind all cached output paths for one pipeline run., Return shared permission/trigger/conjunction coverage diagnostics., Return split-aware context coverage for both trading directions. (+38 more)

### Community 15 - "_apply_monthly_admission_gate"
Cohesion: 0.06
Nodes (29): _apply_monthly_admission_gate(), _evaluate_rule_on_window(), _monthly_window_metrics(), Evaluate a single pool rule on a single monthly window. Returns the full window…, Normalize a window evaluator result for the monthly gate. The float fallback…, Apply the monthly-window shadow-test gate to a pool of rules. Each rule is…, _DeterministicEvaluator, _DeterministicMetricsEvaluator (+21 more)

### Community 16 - "test_feature_detector_properties.py"
Cohesion: 0.09
Nodes (45): all_nan_series(), arbitrary_numeric_series(), binary_series(), large_value_series(), positive_series(), composite, DrawFn, given (+37 more)

### Community 17 - "_make_train_df"
Cohesion: 0.08
Nodes (25): _downsample_chronological(), Preserve per-symbol time order required by exposure/release simulation., Pick a contiguous chronological slice of *n_rows* from *df*. Critical for…, Sample up to *total_rows* rows, distributed equally across symbols. A single…, _sample_df(), _sort_chronological(), _make_train_df(), DataFrame (+17 more)

### Community 18 - "_split"
Cohesion: 0.06
Nodes (27): DataFrame, Unit tests for gpu_fuzzy_trader.data.splitter.Data_Splitter Tests cover: - Per-…, Helper: build df, patch paths, run split, return (train, val)., Compute expected train/val per symbol under holdout+embargo split., floor(N * HOLDOUT_TRAIN_FRACTION) rows go to train., Remaining rows after embargo gap go to validation., For N=101: floor(101 * HOLDOUT_TRAIN_FRACTION) used, not round., train + validation + embargo dropped == total rows. (+19 more)

### Community 19 - "test_cpu_engine_properties.py"
Cohesion: 0.08
Nodes (43): equity_tracking_scenario(), _expected_outcome(), fee_deduction_scenario(), _make_engine(), _make_engine_custom(), _make_minimal_df(), _make_simple_df(), multi_symbol_scenario() (+35 more)

### Community 20 - "detect_feature_mode"
Cohesion: 0.09
Nodes (19): detect_feature_mode(), Module-level convenience wrapper around Feature_Detector.detect_feature_mode., Has negative values, zero_ratio ≤ 0.3 → signed., zero_ratio == 0.3 is NOT > 0.3, so mode is signed (not sparse_signed)., Values {0, 1} match both binary and ternary criteria; binary wins., Values {-1, 0, 1} match ternary; should NOT fall through to signed., Adding value 2 to {0, 1} breaks binary → falls through to positive., Adding value 2 to {-1, 0, 1} breaks ternary → falls through to… (+11 more)

### Community 21 - "selector.py"
Cohesion: 0.07
Nodes (43): get_seed(), Return a per-process random seed (stable within a run, different across runs).…, _align_feature_array(), build_phase1_shared_context(), _build_symbol_masks(), _candidate_feature_columns(), _check_spearman_sign_consistency(), _compute_chronological_stationarity_scores() (+35 more)

### Community 22 - "Feature_Selector"
Cohesion: 0.10
Nodes (20): Feature_Selector, Score and rank features separately for long and short directions., _make_train_df(), MonkeyPatch, Create a minimal training DataFrame with label columns and feature columns., Label columns must not appear in selected features., Meta columns must not appear in selected features., Loader internal columns and ``_``-prefixed names are not candidates. (+12 more)

### Community 23 - "build_master_temporal_folds"
Cohesion: 0.11
Nodes (35): apply_purge_embargo(), build_master_temporal_folds(), export_fold_boundaries(), _format_fold_predictions(), generate_oof_scores(), _get_datetime_series(), Any, DataFrame (+27 more)

### Community 24 - "Reporter"
Cohesion: 0.10
Nodes (14): Generates visual and tabular reports for each pipeline phase. All output files…, Reporter, _make_history(), _make_per_symbol_metrics(), _make_pnl_history(), History entries with missing keys should not raise., History entries with missing keys should not raise., Create a minimal Phase 2 history list. (+6 more)

### Community 25 - "test_encoder_properties.py"
Cohesion: 0.08
Nodes (41): all_active_chromosome_strategy(), chromosome_with_dont_cares_strategy(), feature_name_strategy(), composite, DrawFn, given, ndarray, Property-based tests for gpu_fuzzy_trader.features.encoder.Encoder **Validates:… (+33 more)

### Community 26 - "write_evaluator_clean"
Cohesion: 0.07
Nodes (32): _maybe_write_evaluator_clean(), Path, Write a stripped strategy file containing only ``direction`` and ``rules_set``.…, Write a stripped strategy file alongside the main strategy JSON. This is a…, Validate rule_set and write to JSON at path. After the main write, also writes…, Load JSON from path and run full schema validation. Parameters ---------- path…, write_evaluator_clean(), _make_rule() (+24 more)

### Community 27 - "CandidateRecord"
Cohesion: 0.12
Nodes (38): _candidate_coverage_symbols(), _candidate_positive_symbols(), CandidateRecord, _compose_ruleset(), _diversification_beam(), _diversification_shortlist(), _max_overlap(), _passes_symbol_contribution_certificate() (+30 more)

### Community 28 - "Rule_Pool_Generator"
Cohesion: 0.05
Nodes (32): _deployable_archive_pool_entries(), _derive_val_sample_seed(), DataFrame, Derive a deterministic validation sample seed from the training seed. This…, Convert deployable archive rows into pool-shaped entries for Stage B seeding., Phase 2: GPU-accelerated multi-objective evolutionary rule pool generation.…, Inject guarded migration seeds for the next epoch., Build train/val backtest engines. (+24 more)

### Community 29 - "_preserve_deployable_elites"
Cohesion: 0.10
Nodes (23): environmental_selection_nsga2(), _preserve_deployable_elites(), Canonical NSGA-II truncation on a 2N merged population., Force-preserve top-K deployable-archive elites in the live population.…, _make_chromosome(), _make_deployable_entry(), ndarray, Unit tests for elite preservation under (μ+λ) selection. Verifies that top-K… (+15 more)

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
Cohesion: 0.07
Nodes (20): Output_Writer, Serializes RuleSet dicts to JSON and loads/validates existing JSON files.…, _legacy_writer_contract(), _make_rule(), fixture, parametrize, Unit tests for gpu_fuzzy_trader.output.writer — Output_Writer Tests cover: -…, These schema tests predate mandatory trend context. (+12 more)

### Community 36 - "config.py"
Cohesion: 0.08
Nodes (35): context_contract(), context_contract_digest(), _debug_symbol_universe_size(), effective_config_snapshot(), effective_min_profitable_symbols(), effective_min_trade_support(), effective_monthly_min_trades(), effective_pool_min_val_trades() (+27 more)

### Community 37 - "TestEquityCurveDateAxis"
Cohesion: 0.09
Nodes (21): _make_trade_log(), DataFrame, fixture, Unit tests for date-based x-axis in Reporter.plot_equity_curve (Task 10.3).…, When Entry_Time is absent, x-axis label is 'Trade #'., When Entry_Time is all-NaN, x-axis label is 'Trade #'., axhline is called (initial-capital line) in date mode., axhline is called (initial-capital line) in Trade # mode. (+13 more)

### Community 38 - "test_phase2_window_rotation.py"
Cohesion: 0.09
Nodes (24): _largest_safe_range(), Return (start, end) of the largest contiguous bar range not in *forbidden*. The…, Cap *total_rows* so the per-symbol request fits within the safe range. When…, _resolve_sample_total_rows(), _make_multi_sym_df(), DataFrame, Tests for per-epoch train-window rotation (task-1)., Capping logic for per-epoch window rotation. (+16 more)

### Community 39 - "research_integrity.py"
Cohesion: 0.12
Nodes (27): _canonical_json(), count_trials(), dataset_manifest(), ExperimentLedger, forward_acceptance_lock_path(), Any, Path, PathLike (+19 more)

### Community 40 - "_loader_from_rows"
Cohesion: 0.14
Nodes (13): _loader_from_rows(), _make_rows(), _make_timestamps(), The first N-288 rows (chronologically) should be kept., A row with NaN in any label column should be dropped., Generate n evenly-spaced ISO datetime strings., Bar index should reflect post-drop row count, not original row count., Generate n rows for a given symbol with valid timestamps. (+5 more)

### Community 41 - "TestEndToEndRotation"
Cohesion: 0.18
Nodes (7): fixture, Integration-style tests with a mocked Rule_Pool_Generator., Patch config for rotation and create a generator with minimal setup., When rotation is enabled, _cached_scoped_train_df is stored., After resample_train_for_epoch, the cached slim train changes., Same epoch_idx produces identical cached slim train., TestEndToEndRotation

### Community 42 - "ndarray"
Cohesion: 0.08
Nodes (28): _batch_metrics_from_array(), _build_event_batch(), _exposure_slot_capacity(), _fast_reject_result_rows(), _jax_compute_rule_signals_sparse_batch(), _jax_open_slot(), _jax_release_open_slots(), _jax_simulate_equity_batch() (+20 more)

### Community 43 - "_gpu_runtime.py"
Cohesion: 0.06
Nodes (42): phase2_should_enrich_symbol_metrics(), Return True when GPU batch eval should run a follow-up CPU enrichment pass.…, configure_phase2_gpu_runtime(), detect_gpu_memory_used_gb(), detect_gpu_vram_gb(), detect_system_ram_gb(), evict_cluster_signatures(), _iter_warmup_targets() (+34 more)

### Community 44 - "._ensure_dir"
Cohesion: 0.07
Nodes (19): Reporting and visualization sub-package., _bucket_series_by_mode(), DataFrame, Series, reporter.py — Reporting and visualization for the GPU-Fuzzy Trading Pipeline.…, Write feature-stratified performance metrics to CSV files (one per split). For…, Create parent directories for *path* if they do not exist., Compute maximum drawdown percentage from an equity series. Parameters… (+11 more)

### Community 45 - "compose_hierarchical_signals"
Cohesion: 0.11
Nodes (32): compose_bidirectional_signals(), compose_hierarchical_signals(), normalize_direction(), Any, ndarray, Series, Hierarchical MTF Signal Composer and Asymmetric Soft-Veto Engine. Combines hard…, Compose bidirectional signals from signed LWC triggers (+1 = long, -1 = short,… (+24 more)

### Community 46 - "test_certificate_first_selection.py"
Cohesion: 0.12
Nodes (25): _passes_symbol_concentration_gate(), _passes_tail_holdout_gate(), _portfolio_selection_certificate(), Any, Build the per-direction symbol certificate policy. Specialist islands are…, Return the certificate used by compose, risk, and profit selection., Hard gate on risk-grid tail holdout return (when enabled and present)., Choose a bounded recency candidate using validation-only evidence. The baseline… (+17 more)

### Community 47 - "non_dominated_sort"
Cohesion: 0.13
Nodes (26): batch_hamming_min(), crowding_distance(), _crowding_distance_numba(), _crowding_distance_py(), _disable_numba_sort(), _dominates_numba(), _dominates_py(), non_dominated_sort() (+18 more)

### Community 48 - "optuna_search.py"
Cohesion: 0.10
Nodes (29): _active_search_space(), apply_trial_config(), collect_validation_metrics(), compute_score(), _copy_phase1_2_outputs(), main(), objective(), _pearson() (+21 more)

### Community 49 - "Hierarchical Multi-Timeframe Rule Discovery System Specification"
Cohesion: 0.06
Nodes (30): 1.1 Root Problem Addressed, 1.2 Target Architecture Principles, 1. System Overview & Core Objectives, 2.1 Resampling & Continuity Contract, 2.2 Independent Feature Computation, 2.3 Point-in-Time Causal Alignment, 2. Multi-Timeframe Causal Data Layer, 3.1 Profile Configurations (+22 more)

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

### Community 54 - "test_reporter.py"
Cohesion: 0.15
Nodes (16): _make_selected_features(), _make_stratified_dataset(), _make_stratified_datasets_by_split(), _make_stratified_logs_by_split(), _make_stratified_trade_log(), Unit tests for gpu_fuzzy_trader.reporting.reporter.Reporter Tests cover: -…, Create a dataset with fuzzy-valued feature columns., Create a trade log with Entry_Index values within dataset bounds. (+8 more)

### Community 55 - "test_phase2_rule_pool.py"
Cohesion: 0.05
Nodes (36): _crowding_distance(), _diversity_penalty_blended(), _dominates(), _hamming_distance(), _non_dominated_sort(), _pareto_sortino_stats(), _phenotype_bucket_key(), Return True if solution *a* dominates *b* (all objectives ≤, at least one <). (+28 more)

### Community 56 - "phase2_rule_pool.py"
Cohesion: 0.06
Nodes (82): assign_strata_to_indices(), build_feature_sampling_probs(), pick_active_count(), _pick_active_index(), _pick_inactive_index(), ndarray, _random_active_class(), phase2_init.py — Sparsity-guided stratified population initialization for Phase… (+74 more)

### Community 57 - "_write_and_reload"
Cohesion: 0.23
Nodes (5): _make_rule_set(), Write rule_set to a temp file and reload the raw JSON., TestWriteHappyPath, TestWriteTruncation, _write_and_reload()

### Community 58 - "GPUBacktestEngine"
Cohesion: 0.09
Nodes (17): GPUBacktestEngine, CPU engine for rule-set simulation used by Phase 2/RB., Rule-set sim using a mask cache (delegates to the CPU engine)., Evaluate multiple rule sets (parallel CPU; optional mask cache)., RB JAX path: cached mask entry build + parallel CPU equity. Equity results…, Same interface as CPUBacktestEngine for compatibility., jax.lax.scan-based sequential equity simulation (legacy compat). Parameters…, JAX-accelerated backtest engine for Phase 2 rule pool generation.… (+9 more)

### Community 59 - "._simulate_rule_set_entries"
Cohesion: 0.08
Nodes (26): _append_allocated_entries(), compute_entry_time_priority(), _expectancy_lcb_pct(), _expected_shortfall_pct(), precompute_release_indices(), precompute_release_indices_from_offsets(), DataFrame, ndarray (+18 more)

### Community 60 - "trade_support_penalty"
Cohesion: 0.38
Nodes (4): Backward-compatible wrapper returning penalty only., trade_support_penalty(), Between the hard-reject floor and the soft threshold the penalty is graduated., TestTradeSupportPenalty

### Community 61 - "test_phase2_use_gpu_flag.py"
Cohesion: 0.25
Nodes (7): _minimal_backtest_df(), DataFrame, MonkeyPatch, Phase 2 respects PHASE2_USE_GPU=False without loading JAX GPU engine., The memory-safe CPU route must happen before JAX allocates arrays., A selected CPU backend must not initialize JAX just to warm up., TestPhase2UseGpuFlag

### Community 62 - "rolling_cv.py"
Cohesion: 0.10
Nodes (35): aggregate_fold_metrics(), _bar_index_col(), _build_fold_from_ranges(), build_forbidden_ranges(), build_purged_walk_forward_folds(), cv_folds_only(), derive_primary_holdout(), FoldMetricsSummary (+27 more)

### Community 63 - ".compose"
Cohesion: 0.14
Nodes (9): Any, ndarray, Evaluate hierarchical soft-veto composition on input arrays., Evaluate this frozen candidate on raw OHLCV rows. No thresholds, weights,…, Convert candidate to a JSON-serializable dictionary., Construct HierarchicalStrategyCandidate from dictionary., Compute deterministic strategy SHA-256 identifier., Verify HierarchicalStrategyCandidate encapsulates multi-timeframe rules and… (+1 more)

### Community 64 - "test_plateau_state_leak.py"
Cohesion: 0.11
Nodes (21): _make_minimal_gen(), _mock_evolution_state(), _mock_stage_plan(), Regression tests for plateau-state leak fixes (Fixes A + B). Fix A:…, AC-1, AC-3: plateau_streak resets per epoch; reset_plateau=True always., reset_plateau=True is passed on the very first epoch call., reset_plateau=True is also passed on epoch 2 (regression for leak)., Even when PHASE2_ISLAND_TWO_STAGE_ENABLED=False, reset_plateau=True. (+13 more)

### Community 65 - "TestSeedDirectionUniqueness"
Cohesion: 0.20
Nodes (6): AC: _derive_island_seed produces different seeds for long vs short., Same cluster ID but different direction ⇒ different seed., Same orphan symbol but different direction ⇒ different seed., _derive_island_seed signature must remain (base_seed, island_id) — no direction…, base_seed=None should return None regardless of island_id., TestSeedDirectionUniqueness

### Community 66 - "dashboard.py"
Cohesion: 0.15
Nodes (25): build_dashboard_data(), _direction_data(), _history_rows(), _load_json(), main(), Any, Path, Read-only HTML dashboard for existing pipeline artifacts. The dashboard… (+17 more)

### Community 67 - "DataFrame"
Cohesion: 0.16
Nodes (6): _make_trade_log(), DataFrame, Dataset with only 1 non-NaN paired row → NaN., Rows must be sorted by abs(train_spearman) descending., Create a minimal trade log DataFrame with Equity_After column., TestPlotEquityCurve

### Community 68 - "test_data_loader_properties.py"
Cohesion: 0.15
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

### Community 72 - "test_cpu_engine.py"
Cohesion: 0.09
Nodes (17): _normalize_direction(), Compute a non-annualized Sortino Ratio from per-trade returns., _safe_profit_factor(), _sortino_ratio_from_returns(), JointPortfolioEngine, DataFrame, Joint long/short portfolio simulation. Phase 2 and RB score each direction…, Evaluate long and short rule books in one net-position account. (+9 more)

### Community 73 - "_build_entries_from_rule_set"
Cohesion: 0.10
Nodes (14): _build_entries_from_rule_set(), Priority-based rule assignment: first matching rule wins per row. Mirrors…, DataFrame, Per-symbol metrics should reflect actual trade distribution., Winning trades should produce positive net_pnl per symbol., Row matching both rules should be assigned to rule 1 only., Rows not matching rule 1 should be assigned to rule 2., v5: earlier JSON rule wins over lower dataset row index at same time. (+6 more)

### Community 74 - "execution_ok"
Cohesion: 0.15
Nodes (11): execution_ok(), Return ``True`` iff the evaluator would execute this rule set reasonably. A…, Tests for ``execution_ok``., Skip ratio 0.10 <= 0.20 and exec ratio 0.90 >= 0.60 → True., Skip ratio 0.30 > 0.20 → False., Exec ratio 0.50 < 0.60 → False., Missing ``raw_signal_count`` → treated as 0 → False., ``raw_signal_count=0`` → False. (+3 more)

### Community 75 - "TestWriteStrategyEvaluationTable"
Cohesion: 0.21
Nodes (9): _make_eval_rule_set(), _make_metrics_by_split(), _make_trade_logs_by_split(), Create a minimal rule set for evaluation table tests., Create a metrics_by_split dict with all three splits populated., Create a trade_logs_by_split dict with all three splits populated., Sharpe ratio should equal mean(r)/std(r, ddof=1) for a known log., win_rate, mdd_pct etc. should come from metrics_by_split. (+1 more)

### Community 76 - "test_gpu_engine.py"
Cohesion: 0.12
Nodes (15): _build_data_matrix(), _discretize_series(), DataFrame, Series, Discretize a feature series using evaluator_v5's fixed fuzzy bins. Exported…, Build an (N, K) integer matrix of discretized feature values., Raise RuntimeError if JAX failed to import at module level., _require_jax() (+7 more)

### Community 77 - "TestRunLogHandlerLifecycle"
Cohesion: 0.07
Nodes (29): _capture_logs(), _make_feature_infos_crash(), _make_train_df_crash(), DataFrame, MonkeyPatch, Requirement 3.1 — save_archive is called before _release_resources in run()., Mock save_archive and _release_resources, run Rule_Pool_Generator.run() with a…, save_archive is called with self.direction as the first argument. (+21 more)

### Community 78 - "test_output_writer_properties.py"
Cohesion: 0.13
Nodes (26): parse_symbol_condition(), symbol_conditions.py — Symbol filter parsing (evaluator_v5 parity). Feature…, Parse optional symbol filters. Supported formats: "symbol is 1" "symbol IS 1"…, all_zero_rule_st(), _is_valid_exported_condition(), oversized_rule_set_st(), composite, DrawFn (+18 more)

### Community 79 - "test_feature_selector_properties.py"
Cohesion: 0.15
Nodes (23): dataset_with_high_dispersion_feature(), dataset_with_label_and_meta_columns(), dataset_with_sufficient_dispersion_feature(), _make_label_block(), composite, DataFrame, DrawFn, given (+15 more)

### Community 80 - ".save_archive"
Cohesion: 0.20
Nodes (9): _archive_feature_signature(), Return the ordered feature signature used to validate archive reuse., Read JSON from *path* and return None when the file cannot be loaded., Validate the archive JSON structure and feature compatibility., Load a compatible persistent archive if it exists, otherwise return None.…, Merge the latest pool into a persistent archive and write atomically., _read_json_payload(), _validate_archive_payload() (+1 more)

### Community 81 - "_symbol_specialized_variants"
Cohesion: 0.11
Nodes (27): _attach_source_symbol_filters(), _ensure_symbol_filtered_rule(), _has_symbol_condition(), _is_recency_good(), _is_symbol_condition(), Add deterministic single-condition RB candidates. Evolution is deliberately…, Island/cluster symbols carried on Phase 2 pool entries., Feature ANDs + island symbol ORs (engine ORs multiple ``symbol is X``). (+19 more)

### Community 82 - "TestPlotPerRuleBreakdown"
Cohesion: 0.16
Nodes (11): _make_full_trade_log(), _make_rule_set(), _make_split_logs(), Create a trade log with all columns needed for sharpe computation., Create a minimal rule set with *n* rules., Create a trade log with Rule_Index, Net_PnL, Equity_After columns., Create trade_logs_by_split dict with all three splits populated., One split None, others valid — should not raise. (+3 more)

### Community 83 - "resolve_island_hyperparams"
Cohesion: 0.11
Nodes (21): effective_phase2_val_return_floor_pct(), Direction-aware Phase 2 validation return floor for fitness penalties., Scale integer trade floors by slice size vs full-universe reference., Resolve scaled trade floors and relaxed cross-symbol gates., resolve_island_hyperparams(), scale_trade_floor_by_universe(), Regression tests for anti-overfit / exploration-retune config bundle., test_anti_overfit_config_bundle() (+13 more)

### Community 84 - "test_phase2_island_scheduler.py"
Cohesion: 0.09
Nodes (20): compute_cluster_generation_budgets(), Resolve per-island generation budgets. By default each island receives the full…, parametrize, Unit tests for cluster island scheduler budget math and epoch guard., Multi-symbol clusters each get the full island budget (not split)., Unit tests for the ``defer_warmup`` flag on ``Rule_Pool_Generator``. When…, Existing callers without defer_warmup still warm at init., The configure_phase2_gpu_runtime call is inside 'if not self._defer_warmup:'… (+12 more)

### Community 85 - "_build_target"
Cohesion: 0.17
Nodes (12): _build_target(), Build a direction-specific target signal. Default…, Encoding-aware win class: 2 in asymmetric mode, 1 in legacy mode., Encoding-aware loss class: 0 in either mode., Long: max >= entry*(1+TP/100), min > entry*(1-SL/100) → success., Long: both hit but max_before_min==0 → SL first → failure., Long: both hit but max_before_min==1 → TP first → success., Long: neither TP nor SL hit → failure (or neutral in asymmetric mode). (+4 more)

### Community 86 - "filter_migrants_for_cluster"
Cohesion: 0.11
Nodes (19): filter_migrants_for_cluster(), _migrant_to_metrics(), Backtest one migrant chromosome on receiver cluster engines., Accept only migrants that pass deployability on the receiver cluster slice., _make_migrant_dict(), _make_mock_receiver(), Unit tests for migration safety — migrant gate and seed fraction. Acceptance…, Migrant with val_return 2.5% and >=15 val trades should be accepted. (+11 more)

### Community 87 - "test_gpu_engine_properties.py"
Cohesion: 0.16
Nodes (17): _assert_parity(), _make_engines(), _make_parity_df(), parity_scenario_strategy(), composite, DataFrame, DrawFn, given (+9 more)

### Community 88 - "TestGPUCPUNumericalParity"
Cohesion: 0.10
Nodes (15): ndarray, simulate_rule_set (delegated to CPU) is exactly identical to CPU engine., GPU backtest engine must handle padded chunking correctly., CPU backtest engine must support simulate_rule_batch with correct metrics., Compare GPU and CPU engine for 10 random chromosomes (Task 2 acceptance…, Build a multi-symbol DataFrame with mixed TP/SL/time-exit outcomes., Generate random chromosomes for binary + signed features., GPU engine results must match CPU engine within specified tolerances for 10… (+7 more)

### Community 89 - "build_complete_higher_bars"
Cohesion: 0.13
Nodes (29): DatetimeIndex, align_htf_features_causal(), _as_utc_datetime(), build_complete_higher_bars(), _compute_atr(), _compute_kama(), _compute_rsi(), compute_timeframe_features() (+21 more)

### Community 90 - "discovery.py"
Cohesion: 0.11
Nodes (28): _align_upstream_scores(), _build_layer_frame(), _candidate_key(), _directional_pareto_front(), discover_directional_layer(), _eligible_numeric_features(), _fit_fold_candidates(), _frame_hash() (+20 more)

### Community 91 - "set_purged_wf_reference_rows"
Cohesion: 0.18
Nodes (7): Store full train_new.csv row count after loader prep (split time)., set_purged_wf_reference_rows(), holdout_mode(), purged_mode(), fixture, Tests for purged-WF trade-floor scaling helpers., TestScaleTradeFloor

### Community 92 - "TestMigrationSeedFraction"
Cohesion: 0.20
Nodes (6): Verify that migrant injection uses PHASE2_MIGRATION_SEED_FRACTION, not ARCHIVE., PHASE2_MIGRATION_SEED_FRACTION defaults to 0.10., Ensure the migration fraction is not the same as archive fraction., Simulate the run_epoch migrant path and assert local_cap uses migration…, Simulate the new run_epoch logic: migrant entries are capped by migration…, TestMigrationSeedFraction

### Community 93 - ".encode_condition"
Cohesion: 0.07
Nodes (25): ConfigurationError, decode_chromosome(), encode_condition(), Encoder, get_dont_care(), Exception, ndarray, encoder.py — Encoder Maps gene integer values to fuzzy value names, formats… (+17 more)

### Community 94 - "Feature_Detector"
Cohesion: 0.13
Nodes (12): detect_all_modes(), Feature_Detector, DataFrame, Series, detector.py — Feature_Detector Classifies each feature column into exactly one…, Classify feature columns by their discretization type., Classify a single feature series into one of six modes. Detection order…, Classify every column in *feature_cols* and return a mapping. Parameters… (+4 more)

### Community 95 - "gpu_engine.py"
Cohesion: 0.11
Nodes (16): _min_raw_signals_for_full_scan(), _phase2_trade_floor(), gpu_engine.py — GPUBacktestEngine JAX-accelerated backtest engine for Phase 2…, Minimum executed trades required for a rule to avoid hard trade penalty., Raw match count below this cannot reach trade-floor support., _append_xla_flag(), configure_jax_env(), _cuda_package_root() (+8 more)

### Community 96 - "mtf/__init__.py"
Cohesion: 0.10
Nodes (43): compute_archive_hash(), compute_rule_hash(), get_default_archive_path(), load_mtf_archive_payload(), load_mtf_rule_archive(), normalize_timeframe(), Any, Path (+35 more)

### Community 97 - "test_mtf_pipeline_integration.py"
Cohesion: 0.10
Nodes (21): attach_oof_layer_scores(), Build a training frame carrying only causally available HWC/MWC OOF scores., Unit & Integration tests for Hierarchical MTF Pipeline Integration (Task 6).…, Verify build_mtf_manifest generates valid manifest matching design schema., Verify run_mtf_composition produces valid candidates and executes composition., Verify run_phase1_hwc and run_phase1_mwc discover and persist rules., Verify bidirectional HierarchicalStrategyCandidate composition., Verify pipeline runner exposes new hierarchical phases and contract methods. (+13 more)

### Community 98 - ".load_strategies"
Cohesion: 0.18
Nodes (6): Load long.json and short.json via Output_Writer.load_and_validate(). Parameters…, Return only non-empty RB strategies accepted for deployment., Write a valid rule set JSON to path., Override module-level path dicts and return originals (for standalone tests)., TestLoadStrategies, _write_rule_set()

### Community 99 - "gate_positive_good"
Cohesion: 0.24
Nodes (18): _positive_good_reject_reasons(), Human-readable reasons why ``_is_positive_good`` failed (diagnostics)., _finite_number(), gate_positive_good(), _metric_int(), positive_good_reject_reasons(), PositiveGoodThresholds, Shared metric gates used by Phase 2 admission and RB Governor. The gate is… (+10 more)

### Community 100 - "phase2_island_scheduler.py"
Cohesion: 0.07
Nodes (47): filter_df_to_symbols(), phase2_history_path(), phase2_pool_path(), DataFrame, Return rows for the given symbols; raises if column missing or no rows., Resolve Phase 2 pool path., Resolve Phase 2 history path., clear_global_metrics_cache() (+39 more)

### Community 101 - "loader.py"
Cohesion: 0.21
Nodes (11): _ensure_labels(), load_dataset(), DataFrame, data/loader.py — Data_Loader Stateless CSV loading with full preparation…, Keep supplied labels or derive all labels from raw OHLCV columns. The original…, Load a CSV dataset with full preparation pipeline: 1. Read CSV with comma…, Module-level wrapper around ``Data_Loader.load_dataset``., Validate the trend-context contract on an enriched frame if present. Fails… (+3 more)

### Community 102 - "_remove_low_dispersion"
Cohesion: 0.17
Nodes (10): Remove features where more than `threshold` fraction of values are identical.…, _remove_low_dispersion(), DataFrame, Exactly 95% identical → NOT > 0.95 → keep., 96% identical → > 0.95 → remove., Two features with corr > 0.95 → keep higher-scored one., Two uncorrelated features → both kept., Features in different modes are not compared for redundancy. (+2 more)

### Community 103 - "_make_df"
Cohesion: 0.09
Nodes (19): Build a flat list of per-symbol metric dicts for CSV output. Uses the…, _make_df(), _make_rule_set(), DataFrame, Helper: run OOS_Evaluator.run() with mocked internals and capture plot calls., plot_equity_curve must be called with 'train', 'validation', and 'test'., plot_equity_curve must be called exactly three times (one per split)., Empty train trade log must not raise an exception. (+11 more)

### Community 104 - "prop_settings"
Cohesion: 0.09
Nodes (36): HealthCheck, prop_settings(), Shared Hypothesis settings for property tests. Set PYTEST_LOW_MEMORY=1 to scale…, Hypothesis settings with optional low-memory example scaling., _merge_pop_and_metrics_and_population(), composite, given, ndarray (+28 more)

### Community 105 - "TestPlotDistributionAndEquity"
Cohesion: 0.16
Nodes (6): _make_dist_logs_by_split(), _make_dist_trade_log(), Create a trade log with all columns needed for distribution/equity plots., Create trade_logs_by_split dict with all three splits populated., A trade log with a single trade should not raise., TestPlotDistributionAndEquity

### Community 106 - "baselines.py"
Cohesion: 0.28
Nodes (15): _compact(), _drop_one_condition_variants(), _equal_weight(), _evaluate(), evaluate_baselines(), _feature_shuffle(), _fixed_exit(), Any (+7 more)

### Community 107 - "TestWriteSpearmanCorrelationReport"
Cohesion: 0.13
Nodes (10): _make_dataset_with_label(), _make_datasets_by_split(), Feature not in dataset → NaN for that split., Dataset without label_close_288 → NaN for all features on that split., None dataset for a split → NaN for all features on that split., Empty selected_features → CSV with header only., All non-NaN Spearman values must be in [-1.0, 1.0]., Create a dataset DataFrame with feature columns and label_close_288. (+2 more)

### Community 108 - "_jax_compute_trade_outcomes"
Cohesion: 0.24
Nodes (5): _jax_compute_trade_outcomes(), JAX-jitted vectorized trade outcome computation., Vectorized trade outcome computation for all rows. Mirrors…, Multiple rows processed simultaneously., TestComputeTradeOutcomes

### Community 109 - "_should_post_restart_early_stop_phase2"
Cohesion: 0.29
Nodes (9): Break the epoch when a plateau restart yields no improvement. Independent of…, _should_post_restart_early_stop_phase2(), Unit tests for post-restart no-improvement early stop (Phase 2 runtime)., test_global_disabled_no_stop(), test_global_uses_global_knobs(), test_island_disabled_no_stop(), test_island_streak_at_patience_stops(), test_island_streak_below_patience_no_stop() (+1 more)

### Community 110 - "test_data_splitter_properties.py"
Cohesion: 0.18
Nodes (14): _make_timestamps(), composite, DataFrame, DrawFn, given, Timestamp, Property-based tests for gpu_fuzzy_trader.data.splitter.Data_Splitter…, Patch TRAIN_70_PATH / VALIDATION_30_PATH to tmp_path and run split. (+6 more)

### Community 111 - "nested_walk_forward.py"
Cohesion: 0.13
Nodes (25): Validation helpers for monthly, nested, and multiplicity-safe research., deflated_sharpe_ratio(), estimate_pbo(), Selection-multiplicity diagnostics for strategy research artifacts., Estimate the fraction of folds where the IS winner misses OOS median. Inputs…, Return a normal-approximation deflated Sharpe probability. This is a…, Create a compact ledger-ready multiplicity report., summarize_multiplicity() (+17 more)

### Community 112 - "_validate_pool_schema"
Cohesion: 0.36
Nodes (3): Validate the structure of a loaded pool JSON. Raises ValueError if the schema…, _validate_pool_schema(), TestValidatePoolSchema

### Community 113 - "OOS_Evaluator"
Cohesion: 0.08
Nodes (18): OOS_Evaluator, DataFrame, Save a split report, marking consumed test data as diagnostic-only., Save the combined per-symbol performance CSV., Out-of-sample evaluator for Phase 5. Loads the final long/short strategies,…, Run out-of-sample evaluation. Parameters ---------- allowed_directions :…, Prepare test data using Data_Loader.load_dataset(). Applies the same…, Load prepared train, validation, and test datasets. (+10 more)

### Community 114 - "TestHammingThresholdAutoScale"
Cohesion: 0.15
Nodes (10): Verify the max(3, k_active // 5) formula. The formula is applied in…, Replicate the auto-scaling formula., k_active=0 → threshold = max(3, 0//5) = 3., k_active=5 → threshold = max(3, 5//5=1) = 3., k_active=15 → threshold = max(3, 15//5=3) = 3., k_active=20 → threshold = max(3, 20//5=4) = 4., k_active=50 → threshold = max(3, 50//5=10) = 10., k_active=100 → threshold = max(3, 100//5=20) = 20. (+2 more)

### Community 115 - ".decode_chromosome"
Cohesion: 0.26
Nodes (4): See module-level :func:`decode_chromosome`., All 10 signed values decode correctly via decode_chromosome., Chromosome with numpy int32/int64 values should work., TestDecodeChromosome

### Community 116 - "ValidationError"
Cohesion: 0.11
Nodes (24): context_permission_column(), context_trigger_column(), Return the direction-specific permission column name., Return the direction-specific LWC pullback-reversal trigger column., _condition_feature(), _context_feature_direction(), Exception, writer.py — Output_Writer Serializes RuleSet dicts to JSON with exact schema… (+16 more)

### Community 117 - "_jax_compute_rule_signals"
Cohesion: 0.15
Nodes (11): _jax_compute_rule_signals(), _maybe_jit(), JAX-jitted vectorized rule matching (single chromosome)., Apply ``jit`` if JAX is available, otherwise return *fn* unchanged., Vectorized rule matching: returns (N,) boolean mask of matching rows., All rows match when chromosome equals data_matrix values., No rows match when chromosome differs from data_matrix., Only rows where all active conditions match. (+3 more)

### Community 118 - "TestParetoCollapseWarningGate"
Cohesion: 0.15
Nodes (10): _FakeEngine, AC 4: The default value of the config flag is 5., AC 5: min_pareto_size=0 → warning fires even with small pareto (size=4)., The log message includes 'pareto_size=N' suffix., Fake engine that returns metrics producing a tradeoff between f1 (-sortino) and…, AC 1–5: warning gated on len(pareto_indices) >= config threshold., Run 2-gen evolution and return count of 'Pareto collapse risk' warnings., AC 2: pareto_size=4 < min_pareto_size=5 → no warning fires. (+2 more)

### Community 119 - "test_rb_governor_tail_holdout.py"
Cohesion: 0.07
Nodes (31): _make_walk_forward_fold_engines(), _passes_tail_selection_gate(), Split val_selection into n_splits chronological folds + optional tail holdout.…, Validate a trial ruleset on the reserved chronological validation tail. The…, _make_synthetic_df(), DataFrame, Verify _make_walk_forward_fold_engines splits data correctly., 2 splits + tail holdout → 2 fold engines + 1 tail engine. (+23 more)

### Community 120 - "_compute_stability"
Cohesion: 0.17
Nodes (9): _compute_stability(), Compute cross-symbol stability score. stability = 1 - (std / mean) If mean is 0…, All symbols have same score → std=0 → stability=1., Mean is 0 → stability=0., Single symbol with positive score → stability=1., Single symbol with zero score → stability=0., Very high variance → stability could be negative → clipped to 0., Stability should always be in [0, 1]. (+1 more)

### Community 121 - "passes_pool_admission_gate"
Cohesion: 0.06
Nodes (24): passes_pool_admission_gate(), Hard gate for Phase 2 pool/archive on merged holdout metrics. When…, MonkeyPatch, train=15%/val=4% (3.75× ratio, gap=11pp < 16pp) is REJECTED by the ratio gate…, train=15%/val=10% (1.5× ratio) is ADMITTED., train/val ≈2.5× is ADMITTED when under OVERFIT_RATIO_FLOOR and gap gate., With PHASE2_OVERFIT_RATIO_FLOOR=0.0, the ratio gate is disabled and the high-…, fixture (+16 more)

### Community 122 - "MonthlyWindowSummary"
Cohesion: 0.05
Nodes (52): downcast_numeric_df(), prune_train_columns(), DataFrame, df_slim.py — Reduce DataFrame memory for backtest engines. Keeps only columns…, Downcast label and feature columns to float32; symbol to category (inplace)., Return a copy with only meta, label, and requested feature columns. Parameters…, Drop unused feature columns from the full training split after Phase 1., slim_backtest_df() (+44 more)

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
Cohesion: 0.18
Nodes (8): _mutual_info_discrete_mask(), Return the set of feature names that PASS both stationarity checks. A feature…, Per-column ``discrete_features`` flags for ``mutual_info_classif``. Only…, _stationarity_filter(), Unit tests for gpu_fuzzy_trader.features.selector.Feature_Selector Tests cover:…, TestMutualInfoDiscreteMask, TestReduceOverlap, TestStationarityFilter

### Community 127 - "_derive_epoch_seed"
Cohesion: 0.16
Nodes (10): _derive_epoch_seed(), Return a deterministic per-epoch seed derived from *base_seed* + epoch. Used by…, Re-sample training data with a per-epoch rotated window. Each epoch gets a…, An unknown PHASE2_PER_EPOCH_WINDOW_SEED_MODE raises ValueError., Deterministic seed derivation for per-epoch windows., Same (base_seed, epoch_idx) produces the same seed., Different epoch indices produce different seeds., None base seed returns None. (+2 more)

### Community 128 - "stratification_scenario_strategy"
Cohesion: 0.24
Nodes (15): dataset_with_features_strategy(), metrics_strategy(), composite, DataFrame, DrawFn, Generate a trade log DataFrame with 2–50 rows. Parameters ---------- n_rules:…, Generate a metrics dict with reasonable float values. Keys: win_rate,…, Generate a dataset DataFrame with 10–100 rows. Parameters ----------… (+7 more)

### Community 129 - "TestSpearmanSignConsistency"
Cohesion: 0.18
Nodes (9): ndarray, AC3: Train all positive, tiny |val_rho| < min_abs_corr → feature still kept., AC4: val_df=None → pre-task-8 behavior (mixed train signs still blacklisted)., AC5: val_df without label_close_288 column → val check skipped (no crash)., Tests for _check_spearman_sign_consistency, including the val_df check., Build a minimal DataFrame without 'symbol' to avoid symbol-based folding., AC1: Train all positive, val negative → feature blacklisted., AC2: Train all positive, val positive → feature still kept. (+1 more)

### Community 130 - "compute_labels"
Cohesion: 0.20
Nodes (10): compute_labels(), DataFrame, Forward-window label computation for OHLCV bars. Horizon is ``TAIL_DROP_ROWS``…, Compute the 5 label columns per symbol. - label_open_next: open[t+1] -…, DataFrame, ndarray, Unit tests for forward-window label semantics…, Regression: old backward rolling must not match corrected forward labels. (+2 more)

### Community 131 - "log_memory_rss"
Cohesion: 0.19
Nodes (11): log_memory_rss(), Optional process memory logging (set LOG_MEMORY=1)., Log current process RSS in MiB when LOG_MEMORY=1., Force GC and clear JAX compilation caches between Phase 2 directions. Releases:…, release_phase2_resources(), Release GPU engines and slim in-memory data between island epochs., Drop engine and sampled data to free RAM before the next direction., Tests for optional memory logging helpers. (+3 more)

### Community 132 - "_evaluate_chromosome"
Cohesion: 0.18
Nodes (11): attach_cv_fold_returns_batch(), CvFoldValEvaluator, _evaluate_chromosome(), Evaluate a single chromosome and return (objectives, metrics). objectives =…, Attach per-fold returns for cv_fold_min f3 on batched metrics., Evaluate chromosomes on purged CV validation folds (excluding holdout).…, Build and cache one CPUBacktestEngine per CV fold (one-time cost)., _chromosome_with_min_active() (+3 more)

### Community 133 - "_apply_colab_gpu_defaults"
Cohesion: 0.33
Nodes (6): _apply_colab_gpu_defaults(), is_colab_runtime(), True when running on Google Colab (/content runtime)., Colab T4 optimization for Phase 2 runs., MonkeyPatch, test_colab_defaults_apply_when_content_exists()

### Community 134 - "test_data_loader.py"
Cohesion: 0.20
Nodes (7): _base_row(), _make_ohlcv_rows(), Unit tests for gpu_fuzzy_trader.data.loader.Data_Loader Tests cover: - CSV…, Return a minimal row dict with all required columns., Generate raw OHLCV rows without precomputed forward labels., TestOHLCVLabelGeneration, TestSort

### Community 135 - "_build_rule_signal_mask"
Cohesion: 0.32
Nodes (7): _build_rule_signal_mask(), Cached wrapper around :func:`_compute_rule_signal_mask`., _backtest_df(), DataFrame, Regression tests for evaluator-facing Phase 2 chromosome semantics., Search fitness must use the same fuzzy class as RB/Phase 5 evaluation., test_batch_chromosome_signals_match_decoded_rule_conditions()

### Community 137 - "resolve_phase2_stage_params"
Cohesion: 0.07
Nodes (25): _diversity_recovery_min_unique_ratio(), True when Stage A viability is critically low and search has plateaued., _should_inject_diversity_recovery(), _should_viability_recovery(), island_stage_budgets(), IslandStagePlan, StageLabel, Phase 2 two-stage search hyperparameter profiles (exploration vs refinement). (+17 more)

### Community 138 - "test_phase2_gpu_throughput.py"
Cohesion: 0.15
Nodes (17): _jax_compute_rule_signals_batch(), Batch rule matching for B chromosomes simultaneously. Returns (B, N) boolean…, CPU and GPU backtest engine sub-package., get_gpu_backtest_engine_class(), jax_gpu_backtest_available(), Any, Detect whether JAX / GPUBacktestEngine can be loaded on this host. JAX can fail…, Return ``GPUBacktestEngine`` if ``gpu_engine`` imports cleanly; else ``None``. (+9 more)

### Community 139 - "CPUBacktestEngine"
Cohesion: 0.06
Nodes (74): CPUBacktestEngine, CPU backtest engine that exactly mirrors evaluator_v5.ipynb's…, mandatory_context_conditions(), Return the fixed, mandatory context conditions for *direction*., expectancy_lcb_pct(), Return a conservative lower bound for per-trade net expectancy. Exact CPU…, feature_conditions_only(), phase2_rule_id() (+66 more)

### Community 140 - "._prune_splits_after_phase1"
Cohesion: 0.25
Nodes (4): Selected fuzzy features for Phase 2., Drop unused feature columns from train/val splits to reduce RAM., Drop unused feature columns from train split (legacy single-split API)., TestPruneSplitsAfterPhase1

### Community 141 - "TestSparsePositiveMode"
Cohesion: 0.22
Nodes (5): All non-negative, zero_ratio > 0.3 → sparse_positive., All zeros: zero_ratio = 1.0 > 0.3, min = 0 → sparse_positive., zero_ratio just above 0.3 → sparse_positive., zero_ratio must be computed on the full series, not just non-NaN., TestSparsePositiveMode

### Community 142 - "phase2_support.py"
Cohesion: 0.05
Nodes (56): effective_min_trade_pool_floor(), IslandHyperparams, Resolved Phase 2 knobs for cluster or orphan slices., compute_robust_score(), compute_support_penalty_and_specialist(), deployability_rank_score(), _evolution_feasibility_floors(), EvolutionFloors (+48 more)

### Community 143 - "TestZeroRatioBoundary"
Cohesion: 0.22
Nodes (5): Exactly 30% zeros with non-negative values → positive (not sparse_positive)., 31% zeros with non-negative values → sparse_positive., Exactly 30% zeros with negative values → signed (not sparse_signed)., Just above 30% zeros with negative values → sparse_signed., TestZeroRatioBoundary

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
Cohesion: 0.31
Nodes (3): Validate the structure of a loaded feature selection JSON. Raises ValueError if…, _validate_schema(), TestValidateSchema

### Community 149 - "test_directional_evaluator.py"
Cohesion: 0.12
Nodes (31): classify_directional_labels(), compute_conditional_mwc_labels(), compute_forward_movement_labels(), evaluate_conditional_directional_rule(), evaluate_directional_rule(), fit_directional_threshold(), ndarray, Series (+23 more)

### Community 150 - "Global Constraints"
Cohesion: 0.20
Nodes (9): Global Constraints, Hierarchical Multi-Timeframe Rule Discovery Implementation Plan, Task 1: Causal Multi-Timeframe Data Engine, Task 2: Directional & Conditional Evaluators & Rule Search Profiles, Task 3: Master Temporal Folds, Purged Embargo & OOF Cross-Fitting, Task 4: Decoupled Ensemble Score (Direction & Strength) & Rule Archives, Task 5: MTF Composer, Asymmetric Soft Veto, and Trade Retention Guard, Task 6: Pipeline Integration (`run_pipeline.py`, `config.py`, `loader.py`, `cpu_engine.py`, `rb_governor.py`, `phase5_oos.py`) (+1 more)

### Community 151 - "test_phase2_offspring_batch.py"
Cohesion: 0.29
Nodes (5): CountingEngine, Unit tests for batched offspring evaluation (Phase 2 runtime A1)., Stub engine recording every simulate_rule_batch call's batch size., Offspring should be evaluated via ONE simulate_rule_batch call per gen, not…, test_offspring_evaluated_in_single_batch_per_gen()

### Community 152 - "BFS and DFS Graph Traversal"
Cohesion: 0.20
Nodes (10): CLAUDE.md Graphify Integration, Constrained Query Expansion, BFS and DFS Graph Traversal, Graph Work Memory, Existing Graph Fast Path, CLAUDE.md Graphify Integration, Constrained Query Expansion, BFS and DFS Graph Traversal (+2 more)

### Community 153 - "ConfigError"
Cohesion: 0.60
Nodes (5): _config_check(), ConfigError, _finite_config_number(), Raised when a configuration violates a cross-parameter contract., _validate_config_grid()

### Community 154 - "_make_csv"
Cohesion: 0.33
Nodes (5): _make_csv(), DataFrame, Module-level function should produce same result as class method., Build a CSV string from a list of row dicts., TestModuleLevelFunction

### Community 155 - "TestFeatureNaNFill"
Cohesion: 0.29
Nodes (3): NaN label rows should be dropped, not filled with 0., Only specified feature_cols should be filled., TestFeatureNaNFill

### Community 156 - ".run"
Cohesion: 0.06
Nodes (39): _condition_feature_names(), _entry_validation_per_symbol_metrics(), _filter_compatible_previous_pool(), _filter_pool_by_admission(), _monthly_admission_source_df(), _pool_entry_passes_admission(), _pool_entry_rank(), _pool_path_key() (+31 more)

### Community 157 - "TestNaNHandling"
Cohesion: 0.29
Nodes (4): All-NaN series: unique_vals = [], n_unique = 0 ≤ 2, set() ⊆ {0,1} → binary., NaN is not == 0, so it does not inflate zero_ratio., NaN should not push zero_ratio above threshold., TestNaNHandling

### Community 158 - "TestGlobalMetricsCacheClearing"
Cohesion: 0.27
Nodes (6): When seeded_keys is empty, nothing is removed., When all keys are seeded, cache becomes empty., Verify that only seeded keys are removed from the global cache., Create a deterministic chromosome key., Cache entries matching seeded_keys are removed; non-matching survive., TestGlobalMetricsCacheClearing

### Community 159 - "TestHealthPenaltyWiredIntoRB"
Cohesion: 0.33
Nodes (4): Verify that evaluator health penalty is subtracted from combo score., When evaluator_health_penalty returns > 0, the combo score is lower., When evaluator health is fine, no penalty is applied., TestHealthPenaltyWiredIntoRB

### Community 160 - "TestSparseSignedMode"
Cohesion: 0.33
Nodes (3): Has negative values, zero_ratio > 0.3 → sparse_signed., NaN does not count as zero; zero_ratio on full series., TestSparseSignedMode

### Community 162 - "_build_cpu_archive_engine"
Cohesion: 0.50
Nodes (4): _archive_direction(), _build_cpu_archive_engine(), Resolve a plain long/short direction from an evolution log tag., Build the mandatory CPU evaluator from a Phase 2 engine. GPU batch metrics are…

### Community 163 - "TestEvalCvFoldReturns"
Cohesion: 0.33
Nodes (4): Verify the helper handles None / empty fold_engines safely., fold_engines=None returns None without crashing., fold_engines=[] returns None without crashing., TestEvalCvFoldReturns

### Community 164 - "test_property_27_test_data_preparation_consistency"
Cohesion: 0.50
Nodes (4): DataFrame, given, **Property 27: Test Data Preparation Consistency** **Validates: Requirements…, test_property_27_test_data_preparation_consistency()

### Community 165 - "test_evaluator_health.py"
Cohesion: 0.15
Nodes (9): evaluator_health.py — Pure functions for evaluator-failure-mode awareness.…, Read a numeric metric, returning *default* for missing / None / NaN / Inf., Read an integer metric safely., _safe_float(), _safe_int(), Scoring helpers shared across pipeline phases. Re-exports…, Unit tests for ``evaluator_health_penalty`` and ``execution_ok`` (Task 4).…, Both public functions are importable from the module. (+1 more)

### Community 166 - "run_rb_governor_pipeline"
Cohesion: 0.14
Nodes (23): Path, PathLike, Persist an explicit empty strategy and diagnostic report., Build and optimize rb strategies for each direction and write outputs., run_rb_governor_pipeline(), _write_clean_evaluator(), _write_fail_closed_strategy(), _dummy_df() (+15 more)

### Community 171 - "run_pipeline.py"
Cohesion: 0.06
Nodes (39): apply_fuzzy_feature_scaling(), fit_fuzzy_feature_scaling(), Any, DataFrame, Train-fitted normalization for ordinal ``ff_*`` fuzzy feature codes. The…, Build a train-only scaling contract for ordinal ``ff_*`` columns., Apply an existing train-fitted scaling contract in place and return *df*., __main__.py — Entry point for `python -m gpu_fuzzy_trader.run_pipeline` Allows… (+31 more)

### Community 172 - "TestHallOfFameTrim"
Cohesion: 0.25
Nodes (5): Verify that hall_of_fame is capped to PHASE2_HOF_EPOCH_CARRYOVER=10., HoF with >10 entries is trimmed to exactly 10 oldest entries., HoF with <=10 entries is left unchanged., Empty HoF does not crash during trim logic., TestHallOfFameTrim

### Community 173 - "cpu_engine.py"
Cohesion: 0.07
Nodes (25): _batch_eval_rule_set_pickled(), _jax_runtime_loaded(), _parse_condition(), cpu_engine.py — CPUBacktestEngine Exact Python/NumPy replication of…, Simulate a rule set on rows [row_start, row_end) without copying the df. Used…, Simulate a rule set and return performance metrics. Parameters ----------…, Simulate using a precomputed rule-evaluation mask cache. The mandatory…, Drop cached entries on rows the fixed context mask forbids. ``idx`` validity is… (+17 more)

### Community 174 - "valid_test_csv_dataframe"
Cohesion: 0.33
Nodes (7): _make_timestamps(), composite, DrawFn, Timestamp, Return n strictly increasing timestamp strings starting from *base*., Hypothesis composite strategy that generates a shuffled DataFrame suitable for…, valid_test_csv_dataframe()

### Community 176 - "test_gpu_engine_import_does_not_crash_on_jax_failure"
Cohesion: 0.33
Nodes (6): parametrize, Tests for the lazy-JAX failure path in gpu_engine.py. These tests verify that…, Verify importing gpu_engine.py handles various JAX failure modes gracefully., Build a subprocess script that simulates JAX import failure., _subprocess_script(), test_gpu_engine_import_does_not_crash_on_jax_failure()

### Community 177 - "test_jax_compat.py"
Cohesion: 0.29
Nodes (5): parametrize, Tests for JAX / GPU engine availability detection., Package init must not crash when gpu_engine import fails., test_cpu_engine_import_without_jax(), test_get_gpu_backtest_engine_class_returns_none_on_import_error()

### Community 178 - "TestExecutionHealthInGate"
Cohesion: 0.25
Nodes (5): Verify that ``gate_positive_good`` calls ``execution_ok`` when flagged., When require_execution_health=True and execution_ok fails, gate returns False., When require_execution_health=True and execution_ok passes, gate still returns…, When require_execution_health=False, gate ignores raw_signal_count., TestExecutionHealthInGate

### Community 187 - "test_phase2_plateau_restart.py"
Cohesion: 0.11
Nodes (14): Unit tests for diversity restart on first plateau (Fix D). Verifies that: -…, Reinitialised slots have objectives=np.inf and metrics_cache={}., Even with large Pareto front, at most 10 elite are preserved., Verify AC3 (mutation boost applied to offspring) and AC4 (streak reset). Runs…, Plateau trigger works normally (unchanged)., New config keys have correct defaults., Plateau restart respects island_profile scoping (early stop disabled)., Direct unit tests for the _plateau_diversity_restart helper. (+6 more)

### Community 189 - "opencode.json"
Cohesion: 0.50
Nodes (3): plugin, $schema, .opencode/plugins/graphify.js

## Knowledge Gaps
- **58 isolated node(s):** `$schema`, `.opencode/plugins/graphify.js`, `Task 1: Causal Multi-Timeframe Data Engine`, `Task 2: Directional & Conditional Evaluators & Rule Search Profiles`, `Task 3: Master Temporal Folds, Purged Embargo & OOF Cross-Fitting` (+53 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **11 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `CPUBacktestEngine` connect `CPUBacktestEngine` to `_make_engine`, `_evaluate_chromosome`, `Pipeline_Orchestrator`, `_score_metrics`, `_build_rule_signal_mask`, `_make_df`, `test_phase2_gpu_throughput.py`, `Data_Splitter`, `_apply_monthly_admission_gate`, `test_cpu_engine_properties.py`, `CandidateRecord`, `Rule_Pool_Generator`, `TestContextEntryPaths`, `_build_cpu_archive_engine`, `TestEvalCvFoldReturns`, `run_rb_governor_pipeline`, `run_pipeline.py`, `cpu_engine.py`, `test_certificate_first_selection.py`, `test_jax_compat.py`, `_compute_rule_signal_mask`, `phase2_rule_pool.py`, `GPUBacktestEngine`, `._simulate_rule_set_entries`, `trade_support_penalty`, `test_phase2_use_gpu_flag.py`, `_apply_dynamic_rule`, `test_cpu_engine.py`, `_build_entries_from_rule_set`, `test_gpu_engine.py`, `_symbol_specialized_variants`, `filter_migrants_for_cluster`, `test_gpu_engine_properties.py`, `TestGPUCPUNumericalParity`, `gpu_engine.py`, `test_mtf_pipeline_integration.py`, `phase2_island_scheduler.py`, `baselines.py`, `_jax_compute_trade_outcomes`, `nested_walk_forward.py`, `OOS_Evaluator`, `ValidationError`, `_jax_compute_rule_signals`, `test_rb_governor_tail_holdout.py`, `MonthlyWindowSummary`, `._engine`?**
  _High betweenness centrality (0.146) - this node is a cross-community bridge._
- **Why does `Rule_Pool_Generator` connect `Rule_Pool_Generator` to `log_memory_rss`, `Pipeline_Orchestrator`, `compute_phase2_objectives_from_metrics`, `_evaluate_chromosome`, `TestRulePoolGeneratorRun`, `_get_dont_cares`, `CPUBacktestEngine`, `Data_Splitter`, `_apply_monthly_admission_gate`, `TestRefreshObjectivesOnResumeGate`, `_make_train_df`, `Reporter`, `.run`, `test_phase2_window_rotation.py`, `TestEndToEndRotation`, `run_pipeline.py`, `_gpu_runtime.py`, `test_phase2_rule_pool_properties.py`, `test_phase2_rule_pool.py`, `phase2_rule_pool.py`, `test_phase2_use_gpu_flag.py`, `test_plateau_state_leak.py`, `TestSeedDirectionUniqueness`, `TestRunLogHandlerLifecycle`, `.save_archive`, `test_phase2_island_scheduler.py`, `filter_migrants_for_cluster`, `gpu_engine.py`, `phase2_island_scheduler.py`, `_validate_pool_schema`, `_derive_epoch_seed`?**
  _High betweenness centrality (0.082) - this node is a cross-community bridge._
- **Why does `Reporter` connect `Reporter` to `DataFrame`, `_evaluate_chromosome`, `TestEquityCurveDateAxis`, `prop_settings`, `TestPlotDistributionAndEquity`, `run_pipeline.py`, `._ensure_dir`, `Rule_Pool_Generator`, `TestWriteSpearmanCorrelationReport`, `TestWriteStrategyEvaluationTable`, `OOS_Evaluator`, `TestPlotPerRuleBreakdown`, `ValidationError`, `test_reporter.py`, `phase2_rule_pool.py`, `.run`?**
  _High betweenness centrality (0.066) - this node is a cross-community bridge._
- **Are the 15 inferred relationships involving `Reporter` (e.g. with `CvFoldValEvaluator` and `Rule_Pool_Generator`) actually correct?**
  _`Reporter` has 15 INFERRED edges - model-reasoned connections that need verification._
- **Are the 54 inferred relationships involving `CPUBacktestEngine` (e.g. with `GPUBacktestEngine` and `JointPortfolioEngine`) actually correct?**
  _`CPUBacktestEngine` has 54 INFERRED edges - model-reasoned connections that need verification._
- **Are the 62 inferred relationships involving `Rule_Pool_Generator` (e.g. with `CPUBacktestEngine` and `Reporter`) actually correct?**
  _`Rule_Pool_Generator` has 62 INFERRED edges - model-reasoned connections that need verification._
- **Are the 19 inferred relationships involving `Pipeline_Orchestrator` (e.g. with `CPUBacktestEngine` and `Data_Loader`) actually correct?**
  _`Pipeline_Orchestrator` has 19 INFERRED edges - model-reasoned connections that need verification._