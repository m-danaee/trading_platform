# Graph Report - trading_platform  (2026-08-06)

## Corpus Check
- 185 files · ~258,045 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 4766 nodes · 10754 edges · 189 communities (179 shown, 10 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 547 edges (avg confidence: 0.57)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `6a724adb`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- evox_runner.py
- .save_archive
- CPUBacktestEngine
- _make_engine
- Phase2EvolutionState
- test_evox_runner.py
- TestRulePoolGeneratorRun
- _score_metrics
- _should_plateau_early_stop_phase2
- _make_df
- phase2_sparse_encoding.py
- _dominates
- .run
- Rule_Pool_Generator
- Pipeline_Orchestrator
- _apply_monthly_admission_gate
- test_feature_detector_properties.py
- _make_train_df
- _make_rule
- test_cpu_engine_properties.py
- detect_feature_mode
- selector.py
- Feature_Selector
- run_pipeline.py
- Reporter
- test_encoder_properties.py
- write_evaluator_clean
- CandidateRecord
- phase2_island_scheduler.py
- _preserve_deployable_elites
- maybe_log_generation
- Hybrid CPU and GPU execution policy
- _select_diverse_subset
- test_rb_min_symbols.py
- Graphify Pipeline
- ValidationError
- config.py
- TestEquityCurveDateAxis
- test_phase2_window_rotation.py
- research_integrity.py
- _loader_from_rows
- _split
- gpu_engine.py
- _gpu_runtime.py
- ._ensure_dir
- test_phase2_init.py
- passes_pool_admission_gate
- non_dominated_sort
- optuna_search.py
- _mutate
- test_phase2_rule_pool_properties.py
- _compute_rule_signal_mask
- .__init__
- build_hybrid_symbol_clusters
- test_phase2_island_scheduler.py
- trend_context.py
- Feature_Detector
- test_reporter.py
- GPUBacktestEngine
- Output_Writer
- ValueError
- resolve_phase2_stage_params
- _make_df
- nested_walk_forward.py
- test_plateau_state_leak.py
- validate_config
- dashboard.py
- _symbol_specialized_variants
- test_data_loader_properties.py
- barrier.py
- _m
- _apply_dynamic_rule
- test_cpu_engine.py
- test_rb_fail_closed.py
- TestGPUCPUNumericalParity
- TestWriteStrategyEvaluationTable
- test_gpu_engine.py
- Encoder
- ._record_research_integrity
- test_feature_selector_properties.py
- Data_Loader
- test_phase2_val_sim_interval.py
- TestPlotPerRuleBreakdown
- resolve_island_hyperparams
- test_evaluator_health.py
- _build_target
- filter_migrants_for_cluster
- prop_settings
- test_crash_fix_and_run_logging.py
- TestRunLogHandlerLifecycle
- test_nsga3_selection.py
- TestEndToEndRotation
- phase2_support.py
- .encode_condition
- test_output_writer_properties.py
- test_trend_context.py
- test_phase2_gpu_throughput.py
- .decode_chromosome
- TestOOSEvaluatorRun
- gate_positive_good
- _derive_island_seed
- compute_labels
- _remove_low_dispersion
- resolve_evolution_floors
- test_crash_fix_properties.py
- TestPlotDistributionAndEquity
- baselines.py
- _resolve_plateau_patience
- _jax_compute_trade_outcomes
- _should_post_restart_early_stop_phase2
- TestExecutionHealthInGate
- _pool_seed_chromosomes
- execution_ok
- parity_scenario_strategy
- TestHammingThresholdAutoScale
- TestValLeakGate
- DataFrame
- _jax_compute_rule_signals
- TestParetoCollapseWarningGate
- test_rb_governor_tail_holdout.py
- _compute_stability
- test_certificate_first_selection.py
- MonthlyWindowSummary
- ._engine
- test_rb_concentration_tail_fail_closed.py
- constrained_non_dominated_sort
- test_feature_selector.py
- OOS_Evaluator
- stratification_scenario_strategy
- TestSpearmanSignConsistency
- TestMakeWalkForwardFoldEngines
- log_memory_rss
- _derive_epoch_seed
- _get_dont_cares
- test_property_27_test_data_preparation_consistency
- TestEvictClusterSignatures
- TestHealthPenaltyWiredIntoRB
- compute_phase2_objectives_from_metrics
- set_purged_wf_reference_rows
- .load_pool
- TestNClustersDefined
- TestF3PathResolution
- TestIslandSchedulerGlobalMode
- .load_dataset
- TestRefreshObjectivesOnResumeGate
- conftest.py
- _derive_val_sample_seed
- .skip_if_valid
- _validate_schema
- strategy_id
- test_phase2_use_gpu_flag.py
- test_config_additions.py
- BFS and DFS Graph Traversal
- .get_dont_care
- test_phase2_rule_pool.py
- TestMonthlyGateDataSource
- apply_fuzzy_feature_scaling
- phase2_rule_pool.py
- TestGlobalMetricsCacheClearing
- TestMigrationSeedFraction
- ._prune_splits_after_phase1
- _legacy_writer_contract
- TestDeferredWarmup
- TestMakeFoldEnginesTailHoldout
- _pareto_sortino_stats
- test_rb_full_validation_recovery.py
- test_phase2_batch_evaluator_parity.py
- ._attach_run_log_handler
- TestConfigPhase2Seed
- .load_and_validate
- context_contract_digest
- TestHallOfFameTrim
- trim_evolution_state_memory
- test_phase2_offspring_batch.py
- test_gpu_engine_import_does_not_crash_on_jax_failure
- test_jax_compat.py
- _apply_colab_gpu_defaults
- TestEvalCvFoldReturns
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
2. `CPUBacktestEngine` - 153 edges
3. `Rule_Pool_Generator` - 135 edges
4. `Output_Writer` - 85 edges
5. `prop_settings()` - 79 edges
6. `Pipeline_Orchestrator` - 62 edges
7. `_run_nsga3()` - 58 edges
8. `Feature_Selector` - 58 edges
9. `_run_nsga2_fallback()` - 57 edges
10. `compute_phase2_objectives_from_metrics()` - 57 edges

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

## Communities (189 total, 10 thin omitted)

### Community 0 - "evox_runner.py"
Cohesion: 0.04
Nodes (109): _assign_eval_result(), _behavior_descriptor(), _binary_tournament_pick(), _build_diversity_reference(), _constraint_violations(), _count_deployable_preview(), _count_pop_viable(), _das_dennis() (+101 more)

### Community 1 - ".save_archive"
Cohesion: 0.20
Nodes (9): _archive_feature_signature(), Return the ordered feature signature used to validate archive reuse., Read JSON from *path* and return None when the file cannot be loaded., Validate the archive JSON structure and feature compatibility., Load a compatible persistent archive if it exists, otherwise return None.…, Merge the latest pool into a persistent archive and write atomically., _read_json_payload(), _validate_archive_payload() (+1 more)

### Community 2 - "CPUBacktestEngine"
Cohesion: 0.06
Nodes (83): CPUBacktestEngine, CPU backtest engine that exactly mirrors evaluator_v5.ipynb's…, _available_symbols(), _balanced_phase2_shortlist(), _cost_stress_gate(), _diversification_beam(), _eval_cv_fold_returns(), evaluate_strategy_file_governor() (+75 more)

### Community 3 - "_make_engine"
Cohesion: 0.05
Nodes (33): _build_entries_from_rule_set(), Priority-based rule assignment: first matching rule wins per row. Mirrors…, _make_df(), _make_engine(), DataFrame, All TP trades → win_rate = 100%., No losing trades → profit_factor = 99.0., Row matching both rules should be assigned to rule 1 only. (+25 more)

### Community 4 - "Phase2EvolutionState"
Cohesion: 0.05
Nodes (39): extract_deployable_migrants(), Phase2EvolutionState, StageLabel, Return elite deployable-preview entries suitable for guarded migration., Evolve one island epoch and return updated resumable state., Run Phase 2 NSGA-III evolution. Returns (pareto_pool, history) or state., Resumable NSGA-III state for symbol-island epoch scheduling., run_phase2_evolution() (+31 more)

### Community 5 - "test_evox_runner.py"
Cohesion: 0.05
Nodes (36): _diversity_recovery_min_unique_ratio(), _evaluate_population_indices(), _inherit_val_metrics_from_global_cache(), _load_global_metrics_cache(), True when Stage A viability is critically low and search has plateaued., Restore metrics and optional val_metrics from a cache entry., Evaluate unevaluated individuals, preferring batch simulate_rule_batch., Copy val_* from global cache for identical chromosomes when val is skipped.… (+28 more)

### Community 6 - "TestRulePoolGeneratorRun"
Cohesion: 0.09
Nodes (12): Integration tests using tiny population and generation counts., In holdout mode, val engine must be built for pool admission even when…, All pool entries must have active conditions within [MIN_CONDITIONS,…, Phase 2 must use static TP=PHASE2_TP, SL=PHASE2_SL,…, Pool entries must have executed_trades >= MIN_TRADE_POOL_FLOOR., Verify that Rule_Pool_Generator's persistent self._rng advances across multiple…, After two run_epoch() calls, the internal RNG state must differ from the…, The RNG should produce *different* sequences in two consecutive run_epoch()… (+4 more)

### Community 7 - "_score_metrics"
Cohesion: 0.07
Nodes (40): _combined_return_score(), _evaluate_ruleset(), _optimize_risk(), Return (ok, bonus, penalty) for the desired train-valid balance shape. In…, Dominant objective: return/DD with train-valid balance, plus CV-fold…, Profit objective for lenient rule addition, but now evaluator_v5 aware. A new…, _score_metrics(), _train_valid_shape() (+32 more)

### Community 8 - "_should_plateau_early_stop_phase2"
Cohesion: 0.06
Nodes (43): True for cluster/orphan scoped runs rather than the global path., scoped_island_profile(), Earliest gen for plateau stop. Island epochs (~20 gens) and scaled Stage A…, _resolve_plateau_min_generation(), _should_early_stop_phase2(), _should_plateau_early_stop_phase2(), Verify decision logic uses correct patience values (regression: logs showed…, Island profile: streak=6 triggers when island_patience=6 even when… (+35 more)

### Community 9 - "_make_df"
Cohesion: 0.08
Nodes (21): _make_df(), _make_engine(), MonkeyPatch, Chromosome positions must follow feature_modes insertion order., Exact CPU re-evaluation must interpret genes like the GPU path., Chromosome that matches nothing returns 0 executed trades., Zero-signal chunks should use reject metrics without scanning., PHASE2_SKIP_ZERO_SIGNAL_SCAN must match always-scan metrics. (+13 more)

### Community 10 - "phase2_sparse_encoding.py"
Cohesion: 0.13
Nodes (39): _random_active_class(), canonicalize_slots(), _clamp_slot_gene(), compute_rule_signals_numpy(), count_active_slots(), crossover_sparse(), dense_to_sparse(), empty_slots() (+31 more)

### Community 11 - "_dominates"
Cohesion: 0.18
Nodes (7): _dominates(), _non_dominated_sort(), Return True if solution *a* dominates *b* (all objectives ≤, at least one <)., NSGA-II non-dominated sorting. Parameters ---------- objectives : np.ndarray…, Each solution dominates the next., TestDominates, TestNonDominatedSort

### Community 12 - ".run"
Cohesion: 0.08
Nodes (28): __main__.py — Entry point for `python -m gpu_fuzzy_trader.run_pipeline` Allows…, _log_phase_entry(), _log_pipeline_config(), main(), _now_iso(), DataFrame, Return explicit per-direction RB deployment status and reason., Load Phase 1 feature selection outputs for both directions. (+20 more)

### Community 13 - "Rule_Pool_Generator"
Cohesion: 0.12
Nodes (11): Phase 2: GPU-accelerated multi-objective evolutionary rule pool generation.…, Inject guarded migration seeds for the next epoch., Build train/val backtest engines., Restore slimmed training data from cache (no re-sampling needed)., Rebuild engines after ``park_engines`` dropped GPU state., Build the selected Phase 2 backend for the sampled train frame., Return whether this generator should avoid allocating a JAX engine.…, Attach optional island metadata; safe when *owner* is a partial mock. (+3 more)

### Community 14 - "Pipeline_Orchestrator"
Cohesion: 0.14
Nodes (17): Pipeline_Orchestrator, Load Phase 2 pools for both directions from the persistent cache., Temporarily rebind all cached output paths for one pipeline run., Ensure at least one valid strategy exists before standalone Phase 5., Top-level orchestrator for the GPU-Fuzzy Trading Pipeline. Runs all five phases…, Return the active output root for this run., _resolve_output_root(), _temporary_output_paths() (+9 more)

### Community 15 - "_apply_monthly_admission_gate"
Cohesion: 0.08
Nodes (21): _apply_monthly_admission_gate(), _monthly_window_metrics(), Normalize a window evaluator result for the monthly gate. The float fallback…, Apply the monthly-window shadow-test gate to a pool of rules. Each rule is…, _DeterministicEvaluator, _DeterministicMetricsEvaluator, fixture, Unit tests for the Phase 2 monthly-window shadow-test gate (Task 13). Tests… (+13 more)

### Community 16 - "test_feature_detector_properties.py"
Cohesion: 0.09
Nodes (45): all_nan_series(), arbitrary_numeric_series(), binary_series(), large_value_series(), positive_series(), composite, DrawFn, given (+37 more)

### Community 17 - "_make_train_df"
Cohesion: 0.08
Nodes (23): Sample up to *total_rows* rows, distributed equally across symbols. A single…, _sample_df(), _make_train_df(), DataFrame, Critical: bars must be contiguous so the backtest engine preserves temporal…, Random start must be bounded so the slice always fits forward., divmod distribution gives exactly total_rows (no rounding loss). 701_000 % 14…, total_rows < n_symbols must NOT force 1 row per symbol. (+15 more)

### Community 18 - "_make_rule"
Cohesion: 0.09
Nodes (13): _make_rule(), _make_rule_set(), Unit tests for gpu_fuzzy_trader.output.writer — Output_Writer Tests cover: -…, A rule with only tp non-zero should be accepted., Spot-check a variety of valid fuzzy value names., Write rule_set to a temp file and reload the raw JSON., TestWriteAllZeroRejection, TestWriteConditionValidation (+5 more)

### Community 19 - "test_cpu_engine_properties.py"
Cohesion: 0.08
Nodes (43): equity_tracking_scenario(), _expected_outcome(), fee_deduction_scenario(), _make_engine(), _make_engine_custom(), _make_minimal_df(), _make_simple_df(), multi_symbol_scenario() (+35 more)

### Community 20 - "detect_feature_mode"
Cohesion: 0.05
Nodes (38): detect_feature_mode(), Module-level convenience wrapper around Feature_Detector.detect_feature_mode., Unit tests for gpu_fuzzy_trader.features.detector.Feature_Detector Tests cover:…, All non-negative, zero_ratio > 0.3 → sparse_positive., All zeros: zero_ratio = 1.0 > 0.3, min = 0 → sparse_positive., zero_ratio just above 0.3 → sparse_positive., zero_ratio must be computed on the full series, not just non-NaN., Has negative values, zero_ratio > 0.3 → sparse_signed. (+30 more)

### Community 21 - "selector.py"
Cohesion: 0.09
Nodes (37): _align_feature_array(), build_phase1_shared_context(), _build_symbol_masks(), _candidate_feature_columns(), _check_spearman_sign_consistency(), _compute_chronological_stationarity_scores(), _get_spearman_folds(), _mi_scores_for_mask() (+29 more)

### Community 22 - "Feature_Selector"
Cohesion: 0.10
Nodes (19): Feature_Selector, Score and rank features separately for long and short directions., _make_train_df(), MonkeyPatch, Create a minimal training DataFrame with label columns and feature columns., Label columns must not appear in selected features., Meta columns must not appear in selected features., Loader internal columns and ``_``-prefixed names are not candidates. (+11 more)

### Community 23 - "run_pipeline.py"
Cohesion: 0.06
Nodes (60): downcast_numeric_df(), df_slim.py — Reduce DataFrame memory for backtest engines. Keeps only columns…, Downcast label and feature columns to float32; symbol to category (inplace)., _chronological_half_split(), _holdout_embargo_split(), _purged_walk_forward_split(), DataFrame, data/splitter.py — Data_Splitter Per-symbol chronological split for Phase 2,… (+52 more)

### Community 24 - "Reporter"
Cohesion: 0.09
Nodes (15): Reporting and visualization sub-package., Generates visual and tabular reports for each pipeline phase. All output files…, Reporter, _make_history(), _make_per_symbol_metrics(), _make_pnl_history(), History entries with missing keys should not raise., History entries with missing keys should not raise. (+7 more)

### Community 25 - "test_encoder_properties.py"
Cohesion: 0.09
Nodes (37): all_active_chromosome_strategy(), chromosome_with_dont_cares_strategy(), feature_name_strategy(), composite, DrawFn, given, ndarray, Property-based tests for gpu_fuzzy_trader.features.encoder.Encoder **Validates:… (+29 more)

### Community 26 - "write_evaluator_clean"
Cohesion: 0.07
Nodes (32): _maybe_write_evaluator_clean(), Path, Write a stripped strategy file containing only ``direction`` and ``rules_set``.…, Write a stripped strategy file alongside the main strategy JSON. This is a…, Validate rule_set and write to JSON at path. After the main write, also writes…, Load JSON from path and run full schema validation. Parameters ---------- path…, write_evaluator_clean(), _make_rule() (+24 more)

### Community 27 - "CandidateRecord"
Cohesion: 0.12
Nodes (36): effective_rb_min_distinct_symbols(), Return the RB coverage target for the active debug universe. Full runs keep…, _candidate_coverage_symbols(), _candidate_positive_symbols(), CandidateRecord, _compose_ruleset(), _diversification_shortlist(), _max_overlap() (+28 more)

### Community 28 - "phase2_island_scheduler.py"
Cohesion: 0.11
Nodes (30): filter_df_to_symbols(), phase2_history_path(), phase2_pool_path(), DataFrame, Return rows for the given symbols; raises if column missing or no rows., Resolve Phase 2 pool path., Resolve Phase 2 history path., clear_global_metrics_cache() (+22 more)

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
Cohesion: 0.07
Nodes (24): _normalize_for_association(), Max-min Hamming diversity sampling: greedy pick farthest from chosen. Returns…, Rank-based normalization (robust to outliers like trade_penalty=50)., _select_diverse_subset(), ndarray, Tests for H5/M4/M5 evolution convergence behaviors. Covers: - HoF trimming at…, Verify _select_diverse_subset correctness for edge cases., Create n distinct dense chromosomes. (+16 more)

### Community 33 - "test_rb_min_symbols.py"
Cohesion: 0.10
Nodes (26): mandatory_context_conditions(), Return the fixed, mandatory context conditions for *direction*., _symbols_in_rules(), _dummy_df(), _make_candidate_records(), _mock_train_metrics(), _multi_symbol_rules(), _no_symbol_rule() (+18 more)

### Community 34 - "Graphify Pipeline"
Cohesion: 0.06
Nodes (36): Folder Watcher, URL Ingestion, Conditional Graph Exports, Graphify MCP Server, Extraction Confidence Rubric, Deterministic Full-Path Node IDs, Semantic Hyperedges, Cross-Repository Graph Merge (+28 more)

### Community 35 - "ValidationError"
Cohesion: 0.12
Nodes (22): context_permission_column(), context_trigger_column(), Return the direction-specific permission column name., Return the direction-specific LWC pullback-reversal trigger column., _condition_feature(), _context_feature_direction(), Exception, writer.py — Output_Writer Serializes RuleSet dicts to JSON with exact schema… (+14 more)

### Community 36 - "config.py"
Cohesion: 0.06
Nodes (38): _config_check(), ConfigError, effective_min_trade_support(), effective_monthly_min_trades(), effective_sortino_min_trade_threshold(), effective_val_trade_floor_for_objectives(), _finite_config_number(), get_purged_wf_reference_rows() (+30 more)

### Community 37 - "TestEquityCurveDateAxis"
Cohesion: 0.09
Nodes (21): _make_trade_log(), DataFrame, fixture, Unit tests for date-based x-axis in Reporter.plot_equity_curve (Task 10.3).…, When Entry_Time is absent, x-axis label is 'Trade #'., When Entry_Time is all-NaN, x-axis label is 'Trade #'., axhline is called (initial-capital line) in date mode., axhline is called (initial-capital line) in Trade # mode. (+13 more)

### Community 38 - "test_phase2_window_rotation.py"
Cohesion: 0.09
Nodes (24): _largest_safe_range(), Return (start, end) of the largest contiguous bar range not in *forbidden*. The…, Cap *total_rows* so the per-symbol request fits within the safe range. When…, _resolve_sample_total_rows(), _make_multi_sym_df(), DataFrame, Tests for per-epoch train-window rotation (task-1)., Capping logic for per-epoch window rotation. (+16 more)

### Community 39 - "research_integrity.py"
Cohesion: 0.16
Nodes (24): _canonical_json(), count_trials(), dataset_manifest(), ExperimentLedger, forward_acceptance_lock_path(), Any, Path, PathLike (+16 more)

### Community 40 - "_loader_from_rows"
Cohesion: 0.08
Nodes (26): _base_row(), _loader_from_rows(), _make_csv(), _make_ohlcv_rows(), _make_rows(), _make_timestamps(), DataFrame, Unit tests for gpu_fuzzy_trader.data.loader.Data_Loader Tests cover: - CSV… (+18 more)

### Community 41 - "_split"
Cohesion: 0.06
Nodes (27): _make_timestamps(), Unit tests for gpu_fuzzy_trader.data.splitter.Data_Splitter Tests cover: - Per-…, Helper: build df, patch paths, run split, return (train, val)., Compute expected train/val per symbol under holdout+embargo split., floor(N * HOLDOUT_TRAIN_FRACTION) rows go to train., Remaining rows after embargo gap go to validation., For N=101: floor(101 * HOLDOUT_TRAIN_FRACTION) used, not round., train + validation + embargo dropped == total rows. (+19 more)

### Community 42 - "gpu_engine.py"
Cohesion: 0.07
Nodes (38): _batch_metrics_from_array(), _build_event_batch(), _exposure_slot_capacity(), _fast_reject_result_rows(), _jax_compute_rule_signals_sparse_batch(), _jax_open_slot(), _jax_release_open_slots(), _jax_simulate_equity_batch() (+30 more)

### Community 43 - "_gpu_runtime.py"
Cohesion: 0.11
Nodes (30): configure_phase2_gpu_runtime(), detect_gpu_memory_used_gb(), detect_gpu_vram_gb(), detect_system_ram_gb(), _iter_warmup_targets(), log_gpu_runtime_config(), _ram_batch_cap(), Phase 2 GPU runtime helpers: VRAM-aware batch size and JAX warmup. (+22 more)

### Community 44 - "._ensure_dir"
Cohesion: 0.09
Nodes (17): _bucket_series_by_mode(), DataFrame, Series, Write feature-stratified performance metrics to CSV files (one per split). For…, Create parent directories for *path* if they do not exist., Compute maximum drawdown percentage from an equity series. Parameters…, Write compact train/validation/test generalization diagnostics to JSON. The…, Plot a per-rule performance breakdown across train/validation/test splits.… (+9 more)

### Community 45 - "test_phase2_init.py"
Cohesion: 0.14
Nodes (20): assign_strata_to_indices(), build_feature_sampling_probs(), pick_active_count(), _pick_active_index(), _pick_inactive_index(), ndarray, phase2_init.py — Sparsity-guided stratified population initialization for Phase…, Assign elite / explorer labels to non-seeded population rows. (+12 more)

### Community 46 - "passes_pool_admission_gate"
Cohesion: 0.06
Nodes (28): _feasibility_gate_failures(), passes_pool_admission_gate(), Return per-gate failure flags for evolution-time feasibility diagnostics. Uses…, Hard gate for Phase 2 pool/archive on merged holdout metrics. When…, MonkeyPatch, Tests for the hard overfit ratio gate in pool admission. The ratio gate rejects…, train=15%/val=4% (3.75× ratio, gap=11pp < 16pp) is REJECTED by the ratio gate…, train=15%/val=10% (1.5× ratio) is ADMITTED. (+20 more)

### Community 47 - "non_dominated_sort"
Cohesion: 0.12
Nodes (30): batch_hamming_min(), _batch_static_penalties_numba(), batch_static_support_penalties(), crowding_distance(), _crowding_distance_numba(), _crowding_distance_py(), _disable_numba_sort(), _dominates_numba() (+22 more)

### Community 48 - "optuna_search.py"
Cohesion: 0.10
Nodes (29): _active_search_space(), apply_trial_config(), collect_validation_metrics(), compute_score(), _copy_phase1_2_outputs(), main(), objective(), _pearson() (+21 more)

### Community 49 - "_mutate"
Cohesion: 0.14
Nodes (12): _mutate(), Mutate a chromosome (returns a copy). When activating a dont_care gene, feature…, sparse_to_dense(), C5 mutation bias: force symbol-gene to dont_care / inactive with probability…, Create feature_infos with a feature whose name contains 'symbol'., PHASE2_SYMBOL_GENE_DONT_CARE_PROB=1.0: symbol gene always forced to dont_care., PHASE2_SYMBOL_GENE_DONT_CARE_PROB=0.0: symbol gene never force-set., With probability ~0.5, about half of calls force symbol to dont_care. (+4 more)

### Community 50 - "test_phase2_rule_pool_properties.py"
Cohesion: 0.11
Nodes (21): feature_infos_and_train_df(), _isolate_phase2_archive_paths(), _make_feature_infos(), _make_train_df(), composite, DataFrame, DrawFn, fixture (+13 more)

### Community 51 - "_compute_rule_signal_mask"
Cohesion: 0.15
Nodes (13): conditions_cache_key(), get_or_build_rule_mask(), ndarray, condition_cache.py — Cached boolean masks for textual rule conditions. Avoids…, Stable hashable key for a list of condition strings., Return a boolean row mask for *conditions*, using *cache* when provided. The…, _compute_rule_signal_mask(), Build one boolean signal mask (evaluator_v5 parity). Feature conditions are… (+5 more)

### Community 52 - ".__init__"
Cohesion: 0.29
Nodes (7): _downsample_chronological(), _evaluate_rule_on_window(), DataFrame, Evaluate a single pool rule on a single monthly window. Returns the full window…, Preserve per-symbol time order required by exposure/release simulation., Pick a contiguous chronological slice of *n_rows* from *df*. Critical for…, _sort_chronological()

### Community 53 - "build_hybrid_symbol_clusters"
Cohesion: 0.12
Nodes (29): build_hybrid_symbol_clusters(), _corr_embedding_block(), _feature_names_union(), _feature_profile_block(), load_symbol_clusters(), Any, DataFrame, ndarray (+21 more)

### Community 54 - "test_phase2_island_scheduler.py"
Cohesion: 0.09
Nodes (23): compute_cluster_generation_budgets(), Check if an epoch should be skipped due to small remaining budget. Engine…, Resolve per-island generation budgets. By default each island receives the full…, _should_skip_epoch(), _MockGenerator, parametrize, Unit tests for cluster island scheduler budget math and epoch guard., Minimal mock for Rule_Pool_Generator used in epoch guard tests. (+15 more)

### Community 55 - "trend_context.py"
Cohesion: 0.10
Nodes (41): align_completed_states_to_rows(), average_true_range(), build_higher_bars(), build_manifest(), _classify_hf_bars(), classify_regime(), compute_permissions_and_triggers(), enrich_tape() (+33 more)

### Community 56 - "Feature_Detector"
Cohesion: 0.15
Nodes (10): detect_all_modes(), Feature_Detector, DataFrame, Series, detector.py — Feature_Detector Classifies each feature column into exactly one…, Classify feature columns by their discretization type., Classify a single feature series into one of six modes. Detection order…, Classify every column in *feature_cols* and return a mapping. Parameters… (+2 more)

### Community 57 - "test_reporter.py"
Cohesion: 0.08
Nodes (26): _make_dataset_with_label(), _make_datasets_by_split(), _make_selected_features(), _make_stratified_dataset(), _make_stratified_datasets_by_split(), _make_stratified_logs_by_split(), _make_stratified_trade_log(), Unit tests for gpu_fuzzy_trader.reporting.reporter.Reporter Tests cover: -… (+18 more)

### Community 58 - "GPUBacktestEngine"
Cohesion: 0.10
Nodes (16): GPUBacktestEngine, CPU engine for rule-set simulation used by Phase 2/RB., Rule-set sim using a mask cache (delegates to the CPU engine)., Evaluate multiple rule sets (parallel CPU; optional mask cache)., RB JAX path: cached mask entry build + parallel CPU equity. Equity results…, Same interface as CPUBacktestEngine for compatibility., JAX-accelerated backtest engine for Phase 2 rule pool generation.…, Return the JAX backend in use ('gpu', 'cpu', or 'tpu'). (+8 more)

### Community 59 - "Output_Writer"
Cohesion: 0.11
Nodes (9): Output_Writer, Serializes RuleSet dicts to JSON and loads/validates existing JSON files.…, Verify the exact example from the spec works end-to-end., TestLoadAndValidateErrors, TestLoadAndValidateHappyPath, TestSpecExample, TestValidationErrorType, TestWriteDirectionValidation (+1 more)

### Community 60 - "ValueError"
Cohesion: 0.05
Nodes (47): _append_allocated_entries(), _batch_eval_rule_set_pickled(), _build_rule_signal_mask(), _expectancy_lcb_pct(), _expected_shortfall_pct(), _normalize_direction(), _parse_condition(), precompute_release_indices() (+39 more)

### Community 61 - "resolve_phase2_stage_params"
Cohesion: 0.14
Nodes (14): island_stage_budgets(), IslandStagePlan, StageLabel, Phase 2 two-stage search hyperparameter profiles (exploration vs refinement)., Resolved stage and remaining generation budget for one symbol island., Split an island's total generation budget into Stage A / Stage B portions. Uses…, Map completed island generations to the active two-stage profile., Return stage-tuned hyperparameters. When *stage* is None (single-stage Phase… (+6 more)

### Community 62 - "_make_df"
Cohesion: 0.07
Nodes (27): Build a flat list of per-symbol metric dicts for CSV output. Uses the…, _make_df(), _make_rule_set(), DataFrame, Verify plot_equity_curve is called for all three splits and handles empty logs., Helper: run OOS_Evaluator.run() with mocked internals and capture plot calls., Override module-level path dicts and return originals., plot_equity_curve must be called with 'train', 'validation', and 'test'. (+19 more)

### Community 63 - "nested_walk_forward.py"
Cohesion: 0.14
Nodes (24): Validation helpers for monthly, nested, and multiplicity-safe research., deflated_sharpe_ratio(), estimate_pbo(), Selection-multiplicity diagnostics for strategy research artifacts., Estimate the fraction of folds where the IS winner misses OOS median. Inputs…, Return a normal-approximation deflated Sharpe probability. This is a…, Create a compact ledger-ready multiplicity report., summarize_multiplicity() (+16 more)

### Community 64 - "test_plateau_state_leak.py"
Cohesion: 0.11
Nodes (21): _make_minimal_gen(), _mock_evolution_state(), _mock_stage_plan(), Regression tests for plateau-state leak fixes (Fixes A + B). Fix A:…, AC-1, AC-3: plateau_streak resets per epoch; reset_plateau=True always., reset_plateau=True is passed on the very first epoch call., reset_plateau=True is also passed on epoch 2 (regression for leak)., Even when PHASE2_ISLAND_TWO_STAGE_ENABLED=False, reset_plateau=True. (+13 more)

### Community 65 - "validate_config"
Cohesion: 0.14
Nodes (25): _debug_symbol_universe_size(), effective_config_snapshot(), effective_min_profitable_symbols(), Active symbol count when debug scope is on; None for full-universe runs., Cap cross-symbol profitability gate to the active universe size. With…, Validate all high-impact hyperparameter relationships. The function is…, Return resolved values and derived constraints for audit/reporting., Write the effective configuration snapshot and return its path. (+17 more)

### Community 66 - "dashboard.py"
Cohesion: 0.15
Nodes (25): build_dashboard_data(), _direction_data(), _history_rows(), _load_json(), main(), Any, Path, Read-only HTML dashboard for existing pipeline artifacts. The dashboard… (+17 more)

### Community 67 - "_symbol_specialized_variants"
Cohesion: 0.13
Nodes (24): _attach_source_symbol_filters(), _ensure_symbol_filtered_rule(), _has_symbol_condition(), _is_symbol_condition(), Add deterministic single-condition RB candidates. Evolution is deliberately…, Island/cluster symbols carried on Phase 2 pool entries., Feature ANDs + island symbol ORs (engine ORs multiple ``symbol is X``)., Return rule with an explicit symbol filter when required. This is a safety net… (+16 more)

### Community 68 - "test_data_loader_properties.py"
Cohesion: 0.15
Nodes (26): dataframe_with_nan_features(), dataframe_with_nan_labels(), _load_from_df(), _make_datetime_series(), composite, DataFrame, DrawFn, given (+18 more)

### Community 69 - "barrier.py"
Cohesion: 0.18
Nodes (15): attach_barrier_outcomes(), barrier_column_names(), configured_barrier_pairs(), _first_touch_for_symbol(), _first_touch_for_symbol_numba(), _first_touch_for_symbol_python(), _number_token(), DataFrame (+7 more)

### Community 70 - "_m"
Cohesion: 0.13
Nodes (16): evaluator_health_penalty(), Penalty for evaluator_v5 failure modes — higher is worse. Parameters ----------…, _m(), Max positions 15 > 10 → positive penalty., Max positions 8 <= 10 → no additional penalty from positions., Missing ``raw_signal_count`` → treated as 0 → no skip/exec penalty., ``raw_signal_count=0`` → no skip/exec ratio computed → no penalty., Role 'train' same as 'valid' (1.0x). (+8 more)

### Community 71 - "_apply_dynamic_rule"
Cohesion: 0.21
Nodes (4): _apply_dynamic_rule(), Apply one exported text condition using the original threshold logic. Exactly…, Test all threshold branches of _apply_dynamic_rule., TestApplyDynamicRule

### Community 72 - "test_cpu_engine.py"
Cohesion: 0.07
Nodes (22): Compute a non-annualized Sortino Ratio from per-trade returns., _safe_profit_factor(), _sortino_ratio_from_returns(), CPU and GPU backtest engine sub-package., JointPortfolioEngine, DataFrame, Joint long/short portfolio simulation. Phase 2 and RB score each direction…, Evaluate long and short rule books in one net-position account. (+14 more)

### Community 73 - "test_rb_fail_closed.py"
Cohesion: 0.12
Nodes (25): _assert_capital_budget(), _assert_mandatory_context(), _enforce_capital_budget(), Path, Persist an explicit empty strategy and diagnostic report., Normalize rule capital_pct so sum <= RB_MAX_TOTAL_CAPITAL., Fail closed if the fixed trend-context conditions were lost. The mandatory…, _strategy() (+17 more)

### Community 74 - "TestGPUCPUNumericalParity"
Cohesion: 0.10
Nodes (15): ndarray, simulate_rule_set (delegated to CPU) is exactly identical to CPU engine., GPU backtest engine must handle padded chunking correctly., CPU backtest engine must support simulate_rule_batch with correct metrics., Compare GPU and CPU engine for 10 random chromosomes (Task 2 acceptance…, Build a multi-symbol DataFrame with mixed TP/SL/time-exit outcomes., Generate random chromosomes for binary + signed features., GPU engine results must match CPU engine within specified tolerances for 10… (+7 more)

### Community 75 - "TestWriteStrategyEvaluationTable"
Cohesion: 0.21
Nodes (9): _make_eval_rule_set(), _make_metrics_by_split(), _make_trade_logs_by_split(), Create a minimal rule set for evaluation table tests., Create a metrics_by_split dict with all three splits populated., Create a trade_logs_by_split dict with all three splits populated., Sharpe ratio should equal mean(r)/std(r, ddof=1) for a known log., win_rate, mdd_pct etc. should come from metrics_by_split. (+1 more)

### Community 76 - "test_gpu_engine.py"
Cohesion: 0.13
Nodes (13): _build_data_matrix(), _discretize_series(), DataFrame, Series, Discretize a feature series using evaluator_v5's fixed fuzzy bins. Exported…, Build an (N, K) integer matrix of discretized feature values., disable_skip_optimization(), DataFrame (+5 more)

### Community 77 - "Encoder"
Cohesion: 0.15
Nodes (15): ConfigurationError, decode_chromosome(), encode_condition(), Encoder, get_dont_care(), Exception, ndarray, encoder.py — Encoder Maps gene integer values to fuzzy value names, formats… (+7 more)

### Community 78 - "._record_research_integrity"
Cohesion: 0.13
Nodes (11): Any, Typed, versioned profile for the active research contract., Small stable surface for comparing experiments. The legacy config module…, ResearchProfile, _phase5_test_metrics(), _print_run_summary(), Any, Return test-split metrics (supports nested Phase 5 result shape). (+3 more)

### Community 79 - "test_feature_selector_properties.py"
Cohesion: 0.15
Nodes (23): dataset_with_high_dispersion_feature(), dataset_with_label_and_meta_columns(), dataset_with_sufficient_dispersion_feature(), _make_label_block(), composite, DataFrame, DrawFn, given (+15 more)

### Community 80 - "Data_Loader"
Cohesion: 0.07
Nodes (33): required_barrier_columns(), Data_Loader, load_dataset(), data/loader.py — Data_Loader Stateless CSV loading with full preparation…, Stateless data loader for the GPU-Fuzzy Trading Pipeline., Module-level wrapper around ``Data_Loader.load_dataset``., Data_Splitter, load_cached_split_if_fresh() (+25 more)

### Community 81 - "test_phase2_val_sim_interval.py"
Cohesion: 0.15
Nodes (14): CountingEngine, Unit tests for periodic val simulation (Phase 2 runtime A2)., PHASE2_VAL_SIM_INTERVAL=1 preserves original behaviour (val every gen)., Direct unit test of _should_run_val_this_gen with interval=3. With interval=3,…, Integration test: with interval=3 and 13-gen epoch, val runs on gens 0, 3, 6,…, Val metrics for a chromosome are deterministic when val is skipped. With…, With PHASE2_VAL_SIM_INTERVAL=2, val sim runs on interval gens (0,2,4) + last…, Validation-dependent fitness is evaluated consistently every gen. (+6 more)

### Community 82 - "TestPlotPerRuleBreakdown"
Cohesion: 0.16
Nodes (11): _make_full_trade_log(), _make_rule_set(), _make_split_logs(), Create a trade log with all columns needed for sharpe computation., Create a minimal rule set with *n* rules., Create a trade log with Rule_Index, Net_PnL, Equity_After columns., Create trade_logs_by_split dict with all three splits populated., One split None, others valid — should not raise. (+3 more)

### Community 83 - "resolve_island_hyperparams"
Cohesion: 0.11
Nodes (21): effective_phase2_val_return_floor_pct(), Direction-aware Phase 2 validation return floor for fitness penalties., Scale integer trade floors by slice size vs full-universe reference., Resolve scaled trade floors and relaxed cross-symbol gates., resolve_island_hyperparams(), scale_trade_floor_by_universe(), Regression tests for anti-overfit / exploration-retune config bundle., test_anti_overfit_config_bundle() (+13 more)

### Community 84 - "test_evaluator_health.py"
Cohesion: 0.15
Nodes (9): evaluator_health.py — Pure functions for evaluator-failure-mode awareness.…, Read a numeric metric, returning *default* for missing / None / NaN / Inf., Read an integer metric safely., _safe_float(), _safe_int(), Scoring helpers shared across pipeline phases. Re-exports…, Unit tests for ``evaluator_health_penalty`` and ``execution_ok`` (Task 4).…, Both public functions are importable from the module. (+1 more)

### Community 85 - "_build_target"
Cohesion: 0.17
Nodes (12): _build_target(), Build a direction-specific target signal. Default…, Encoding-aware win class: 2 in asymmetric mode, 1 in legacy mode., Encoding-aware loss class: 0 in either mode., Long: max >= entry*(1+TP/100), min > entry*(1-SL/100) → success., Long: both hit but max_before_min==0 → SL first → failure., Long: both hit but max_before_min==1 → TP first → success., Long: neither TP nor SL hit → failure (or neutral in asymmetric mode). (+4 more)

### Community 86 - "filter_migrants_for_cluster"
Cohesion: 0.11
Nodes (19): filter_migrants_for_cluster(), _migrant_to_metrics(), Backtest one migrant chromosome on receiver cluster engines., Accept only migrants that pass deployability on the receiver cluster slice., _make_migrant_dict(), _make_mock_receiver(), Unit tests for migration safety — migrant gate and seed fraction. Acceptance…, Migrant with val_return 2.5% and >=15 val trades should be accepted. (+11 more)

### Community 87 - "prop_settings"
Cohesion: 0.07
Nodes (44): HealthCheck, prop_settings(), Shared Hypothesis settings for property tests. Set PYTEST_LOW_MEMORY=1 to scale…, Hypothesis settings with optional low-memory example scaling., _make_timestamps(), composite, DataFrame, DrawFn (+36 more)

### Community 88 - "test_crash_fix_and_run_logging.py"
Cohesion: 0.11
Nodes (18): _capture_logs(), _make_feature_infos_crash(), _make_train_df_crash(), Unit tests for crash-fix-and-run-logging spec. Covers: - Task 1.1: Smoke tests…, Requirement 3.1 — save_archive is called before _release_resources in run()., Mock save_archive and _release_resources, run Rule_Pool_Generator.run() with a…, save_archive is called with self.direction as the first argument., Requirement 3.3 — If save_archive raises, the exception is caught, a WARNING is… (+10 more)

### Community 89 - "TestRunLogHandlerLifecycle"
Cohesion: 0.17
Nodes (11): DataFrame, MonkeyPatch, Requirements 1.1, 1.4, 1.5, 1.6, 1.7 — run.log FileHandler is attached, writes…, Count FileHandlers on the root logger pointing to *path*., Patch every phase method on Pipeline_Orchestrator to be a no-op., run.log must exist after run() and contain both separator lines., Root logger must have no extra FileHandlers pointing to run.log after run()., Handler must be detached even when run() raises an exception. (+3 more)

### Community 90 - "test_nsga3_selection.py"
Cohesion: 0.40
Nodes (3): skipif, Unit tests for NSGA-III environmental selection., TestNsga3Selection

### Community 91 - "TestEndToEndRotation"
Cohesion: 0.18
Nodes (7): fixture, Integration-style tests with a mocked Rule_Pool_Generator., Patch config for rotation and create a generator with minimal setup., When rotation is enabled, _cached_scoped_train_df is stored., After resample_train_for_epoch, the cached slim train changes., Same epoch_idx produces identical cached slim train., TestEndToEndRotation

### Community 92 - "phase2_support.py"
Cohesion: 0.05
Nodes (49): effective_min_trade_pool_floor(), effective_pool_min_val_trades(), IslandHyperparams, Resolved Phase 2 knobs for cluster or orphan slices., _pool_entry_passes_admission(), Check stored train/val metrics on a pool JSON entry., compute_support_penalty_and_specialist(), deployability_rank_score() (+41 more)

### Community 93 - ".encode_condition"
Cohesion: 0.10
Nodes (9): See module-level :func:`encode_condition`., parametrize, TestEncodeConditionBinary, TestEncodeConditionErrors, TestEncodeConditionPositive, TestEncodeConditionSigned, TestEncodeConditionSparsePositive, TestEncodeConditionSparseSigned (+1 more)

### Community 94 - "test_output_writer_properties.py"
Cohesion: 0.13
Nodes (26): parse_symbol_condition(), symbol_conditions.py — Symbol filter parsing (evaluator_v5 parity). Feature…, Parse optional symbol filters. Supported formats: "symbol is 1" "symbol IS 1"…, all_zero_rule_st(), _is_valid_exported_condition(), oversized_rule_set_st(), composite, DrawFn (+18 more)

### Community 95 - "test_trend_context.py"
Cohesion: 0.10
Nodes (11): _enriched(), Regression: symbols arrive interleaved (sorted by datetime then symbol), so…, Higher-timeframe state is only published after that bar completes and aligned…, _raw_tape(), TestCausalPublicationTiming, TestContextIdentity, TestInputAndBoundaryContracts, TestLoaderContext (+3 more)

### Community 96 - "test_phase2_gpu_throughput.py"
Cohesion: 0.16
Nodes (16): _jax_compute_rule_signals_batch(), Batch rule matching for B chromosomes simultaneously. Returns (B, N) boolean…, get_gpu_backtest_engine_class(), jax_gpu_backtest_available(), Any, Detect whether JAX / GPUBacktestEngine can be loaded on this host. JAX can fail…, Return ``GPUBacktestEngine`` if ``gpu_engine`` imports cleanly; else ``None``., True when ``get_gpu_backtest_engine_class()`` would succeed. (+8 more)

### Community 97 - ".decode_chromosome"
Cohesion: 0.26
Nodes (4): See module-level :func:`decode_chromosome`., All 10 signed values decode correctly via decode_chromosome., Chromosome with numpy int32/int64 values should work., TestDecodeChromosome

### Community 98 - "TestOOSEvaluatorRun"
Cohesion: 0.09
Nodes (12): Load long.json and short.json via Output_Writer.load_and_validate(). Parameters…, Return only non-empty RB strategies accepted for deployment., _isolate_phase5_reporter_outputs(), fixture, Keep Reporter plots/CSVs out of the checked-in outputs directory., Write a valid rule set JSON to path., Integration tests using tmp_path overrides for all output paths. The run()…, Run OOS_Evaluator.run() once for the whole class (long direction only). (+4 more)

### Community 99 - "gate_positive_good"
Cohesion: 0.24
Nodes (18): _positive_good_reject_reasons(), Human-readable reasons why ``_is_positive_good`` failed (diagnostics)., _finite_number(), gate_positive_good(), _metric_int(), positive_good_reject_reasons(), PositiveGoodThresholds, Shared metric gates used by Phase 2 admission and RB Governor. The gate is… (+10 more)

### Community 100 - "_derive_island_seed"
Cohesion: 0.12
Nodes (16): _derive_island_seed(), Return a deterministic but distinct seed per island from base seed + id.…, AC: _derive_island_seed produces different seeds for long vs short., Same cluster ID but different direction ⇒ different seed., Same orphan symbol but different direction ⇒ different seed., _derive_island_seed signature must remain (base_seed, island_id) — no direction…, base_seed=None should return None regardless of island_id., TestSeedDirectionUniqueness (+8 more)

### Community 101 - "compute_labels"
Cohesion: 0.20
Nodes (10): compute_labels(), DataFrame, Forward-window label computation for OHLCV bars. Horizon is ``TAIL_DROP_ROWS``…, Compute the 5 label columns per symbol. - label_open_next: open[t+1] -…, DataFrame, ndarray, Unit tests for forward-window label semantics…, Regression: old backward rolling must not match corrected forward labels. (+2 more)

### Community 102 - "_remove_low_dispersion"
Cohesion: 0.17
Nodes (10): Remove features where more than `threshold` fraction of values are identical.…, _remove_low_dispersion(), DataFrame, Exactly 95% identical → NOT > 0.95 → keep., 96% identical → > 0.95 → remove., Two features with corr > 0.95 → keep higher-scored one., Two uncorrelated features → both kept., Features in different modes are not compared for redundancy. (+2 more)

### Community 103 - "resolve_evolution_floors"
Cohesion: 0.18
Nodes (8): EvolutionFloors, Resolved evolution-time floors (pool admission gates remain strict)., Return stage-aware fitness floors; defaults to global strict knobs. When both…, resolve_evolution_floors(), Stage A soft floors must survive island_hyperparams (cluster two-stage)., _pool_admission_floors returns the ADMISSION floor (1.15), not the EVOLUTION…, TestPoolAdmissionScaledFloors, TestResolveEvolutionFloorsIslandTwoStage

### Community 104 - "test_crash_fix_properties.py"
Cohesion: 0.10
Nodes (26): _append_xla_flag(), configure_jax_env(), _cuda_package_root(), Path, JAX/XLA runtime configuration — call before the first ``import jax``., Find a pip-installed CUDA toolkit root, if one is available., Append an XLA flag once, preserving explicit user configuration., Configure JAX/XLA runtime for predictable desktop-friendly GPU usage. -… (+18 more)

### Community 105 - "TestPlotDistributionAndEquity"
Cohesion: 0.16
Nodes (6): _make_dist_logs_by_split(), _make_dist_trade_log(), Create a trade log with all columns needed for distribution/equity plots., Create trade_logs_by_split dict with all three splits populated., A trade log with a single trade should not raise., TestPlotDistributionAndEquity

### Community 106 - "baselines.py"
Cohesion: 0.24
Nodes (17): compute_entry_time_priority(), Map each row to a timestamp priority code (evaluator_v5 parity)., _compact(), _drop_one_condition_variants(), _equal_weight(), _evaluate(), evaluate_baselines(), _feature_shuffle() (+9 more)

### Community 107 - "_resolve_plateau_patience"
Cohesion: 0.24
Nodes (6): Resolve the plateau patience value based on profile and stage. Cluster/orphan…, _resolve_plateau_patience(), Tests for _resolve_plateau_patience helper., Helper to create a Phase2StageParams with controlled patience., Regression: Stage A min_gen=30 used to disable island plateau., TestResolvePlateauPatience

### Community 108 - "_jax_compute_trade_outcomes"
Cohesion: 0.24
Nodes (5): _jax_compute_trade_outcomes(), JAX-jitted vectorized trade outcome computation., Vectorized trade outcome computation for all rows. Mirrors…, Multiple rows processed simultaneously., TestComputeTradeOutcomes

### Community 109 - "_should_post_restart_early_stop_phase2"
Cohesion: 0.29
Nodes (9): Break the epoch when a plateau restart yields no improvement. Independent of…, _should_post_restart_early_stop_phase2(), Unit tests for post-restart no-improvement early stop (Phase 2 runtime)., test_global_disabled_no_stop(), test_global_uses_global_knobs(), test_island_disabled_no_stop(), test_island_streak_at_patience_stops(), test_island_streak_below_patience_no_stop() (+1 more)

### Community 110 - "TestExecutionHealthInGate"
Cohesion: 0.25
Nodes (5): Verify that ``gate_positive_good`` calls ``execution_ok`` when flagged., When require_execution_health=True and execution_ok fails, gate returns False., When require_execution_health=True and execution_ok passes, gate still returns…, When require_execution_health=False, gate ignores raw_signal_count., TestExecutionHealthInGate

### Community 111 - "_pool_seed_chromosomes"
Cohesion: 0.20
Nodes (8): _pool_seed_chromosomes(), Stack chromosome rows into a batch (dense 2D or sparse 3D)., Extract deduplicated chromosomes from a Phase 2 pool for population seeding., Pick top Stage A deployable rules and merge with optional base seeds., _stack_chromosome_rows(), _stage_b_seed_chromosomes(), TestPoolSeedChromosomes, TestTwoStageOrchestration

### Community 112 - "execution_ok"
Cohesion: 0.15
Nodes (11): execution_ok(), Return ``True`` iff the evaluator would execute this rule set reasonably. A…, Tests for ``execution_ok``., Skip ratio 0.10 <= 0.20 and exec ratio 0.90 >= 0.60 → True., Skip ratio 0.30 > 0.20 → False., Exec ratio 0.50 < 0.60 → False., Missing ``raw_signal_count`` → treated as 0 → False., ``raw_signal_count=0`` → False. (+3 more)

### Community 113 - "parity_scenario_strategy"
Cohesion: 0.50
Nodes (4): parity_scenario_strategy(), composite, DrawFn, Generate a random dataset and trade parameters for GPU-CPU parity testing.…

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
Cohesion: 0.17
Nodes (9): _jax_compute_rule_signals(), JAX-jitted vectorized rule matching (single chromosome)., Vectorized rule matching: returns (N,) boolean mask of matching rows., All rows match when chromosome equals data_matrix values., No rows match when chromosome differs from data_matrix., Only rows where all active conditions match., Columns where chromosome == dont_care are ignored., All dont_care chromosome matches every row. (+1 more)

### Community 118 - "TestParetoCollapseWarningGate"
Cohesion: 0.15
Nodes (10): _FakeEngine, AC 4: The default value of the config flag is 5., AC 5: min_pareto_size=0 → warning fires even with small pareto (size=4)., The log message includes 'pareto_size=N' suffix., Fake engine that returns metrics producing a tradeoff between f1 (-sortino) and…, AC 1–5: warning gated on len(pareto_indices) >= config threshold., Run 2-gen evolution and return count of 'Pareto collapse risk' warnings., AC 2: pareto_size=4 < min_pareto_size=5 → no warning fires. (+2 more)

### Community 119 - "test_rb_governor_tail_holdout.py"
Cohesion: 0.21
Nodes (11): _MockEngine, Unit tests for RB Governor tail-holdout path in risk grid. Covers: -…, Verify _optimize_risk with tail_holdout_engine adds tail fields to final…, When tail_holdout_engine is provided, the final history entry contains…, When tail_holdout_engine=None, NO tail fields in history., Composition may use the reserved validation tail, never Phase 5 data., Minimal mock that mimics CPUBacktestEngine for testing _optimize_risk., test_tail_selection_gate_requires_positive_return_and_support() (+3 more)

### Community 120 - "_compute_stability"
Cohesion: 0.17
Nodes (9): _compute_stability(), Compute cross-symbol stability score. stability = 1 - (std / mean) If mean is 0…, All symbols have same score → std=0 → stability=1., Mean is 0 → stability=0., Single symbol with positive score → stability=1., Single symbol with zero score → stability=0., Very high variance → stability could be negative → clipped to 0., Stability should always be in [0, 1]. (+1 more)

### Community 121 - "test_certificate_first_selection.py"
Cohesion: 0.23
Nodes (11): _BatchEngine, _candidate(), _metrics(), _pool_entry(), Focused regressions for certificate-first RB and Phase 2 selection., test_certificate_rejects_eth_only_and_accepts_balanced_team(), test_compose_uses_balanced_beam_seed_instead_of_eth_leader(), test_partial_specialist_policy_keeps_full_floor_when_both_symbols_exist() (+3 more)

### Community 122 - "MonthlyWindowSummary"
Cohesion: 0.05
Nodes (49): prune_train_columns(), DataFrame, Return a copy with only meta, label, and requested feature columns. Parameters…, Drop unused feature columns from the full training split after Phase 1., slim_backtest_df(), build_monthly_windows(), _datetime_series(), evaluate_rule_set_monthly() (+41 more)

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

### Community 127 - "OOS_Evaluator"
Cohesion: 0.08
Nodes (18): OOS_Evaluator, DataFrame, Out-of-sample evaluator for Phase 5. Loads the final long/short strategies,…, Run out-of-sample evaluation. Parameters ---------- allowed_directions :…, Prepare test data using Data_Loader.load_dataset(). Applies the same…, Load prepared train, validation, and test datasets., Load and validate a strictly newer, untouched forward period., Load selected features for a direction when available. (+10 more)

### Community 128 - "stratification_scenario_strategy"
Cohesion: 0.24
Nodes (15): dataset_with_features_strategy(), metrics_strategy(), composite, DataFrame, DrawFn, Generate a trade log DataFrame with 2–50 rows. Parameters ---------- n_rules:…, Generate a metrics dict with reasonable float values. Keys: win_rate,…, Generate a dataset DataFrame with 10–100 rows. Parameters ----------… (+7 more)

### Community 129 - "TestSpearmanSignConsistency"
Cohesion: 0.18
Nodes (9): ndarray, Tests for _check_spearman_sign_consistency, including the val_df check., Build a minimal DataFrame without 'symbol' to avoid symbol-based folding., AC1: Train all positive, val negative → feature blacklisted., AC2: Train all positive, val positive → feature still kept., AC3: Train all positive, tiny |val_rho| < min_abs_corr → feature still kept., AC4: val_df=None → pre-task-8 behavior (mixed train signs still blacklisted)., AC5: val_df without label_close_288 column → val check skipped (no crash). (+1 more)

### Community 130 - "TestMakeWalkForwardFoldEngines"
Cohesion: 0.17
Nodes (10): _make_synthetic_df(), DataFrame, Verify _make_walk_forward_fold_engines splits data correctly., 2 splits + tail holdout → 2 fold engines + 1 tail engine., tail_holdout_frac=0 → tail engine is None., Each symbol's data is divided into contiguous chunks across folds., Single symbol without symbol column is handled gracefully., Very small data per symbol (fewer rows than n_splits) does not crash. (+2 more)

### Community 131 - "log_memory_rss"
Cohesion: 0.19
Nodes (11): log_memory_rss(), Optional process memory logging (set LOG_MEMORY=1)., Log current process RSS in MiB when LOG_MEMORY=1., Force GC and clear JAX compilation caches between Phase 2 directions. Releases:…, release_phase2_resources(), Release GPU engines and slim in-memory data between island epochs., Drop engine and sampled data to free RAM before the next direction., Tests for optional memory logging helpers. (+3 more)

### Community 132 - "_derive_epoch_seed"
Cohesion: 0.16
Nodes (10): _derive_epoch_seed(), Return a deterministic per-epoch seed derived from *base_seed* + epoch. Used by…, Re-sample training data with a per-epoch rotated window. Each epoch gets a…, An unknown PHASE2_PER_EPOCH_WINDOW_SEED_MODE raises ValueError., Deterministic seed derivation for per-epoch windows., Same (base_seed, epoch_idx) produces the same seed., Different epoch indices produce different seeds., None base seed returns None. (+2 more)

### Community 133 - "_get_dont_cares"
Cohesion: 0.12
Nodes (16): _count_active_conditions(), _get_dont_cares(), Return array of dont_care sentinels for each feature., Count active rule conditions (sparse slots or dense dont_care encoding)., _is_all_inactive_sparse(), _make_feature_infos(), No symbol feature in feature_infos: bias silently does nothing (no crash)., All active sparse genes must be in [0, dont_care] inclusive. (+8 more)

### Community 134 - "test_property_27_test_data_preparation_consistency"
Cohesion: 0.20
Nodes (11): _make_timestamps(), composite, DataFrame, DrawFn, given, Timestamp, **Property 27: Test Data Preparation Consistency** **Validates: Requirements…, Return n strictly increasing timestamp strings starting from *base*. (+3 more)

### Community 135 - "TestEvictClusterSignatures"
Cohesion: 0.14
Nodes (10): evict_cluster_signatures(), Evict JAX compiled signatures for a completed cluster. Removes entries from…, Unit tests for ``evict_cluster_signatures`` in ``_gpu_runtime.py``. These tests…, _warmup_signature() appends cluster_id to the returned tuple., evict_cluster_signatures(cluster_id=cid) removes only signatures tagged with…, evict_cluster_signatures(cluster_id=None) evicts ALL signatures., evict_cluster_signatures with a cluster_id that has no signatures returns 0 and…, Structural test: _run_cluster_islands must contain the evict_cluster_signatures… (+2 more)

### Community 136 - "TestHealthPenaltyWiredIntoRB"
Cohesion: 0.33
Nodes (4): Verify that evaluator health penalty is subtracted from combo score., When evaluator_health_penalty returns > 0, the combo score is lower., When evaluator health is fine, no penalty is applied., TestHealthPenaltyWiredIntoRB

### Community 137 - "compute_phase2_objectives_from_metrics"
Cohesion: 0.04
Nodes (37): compute_phase2_objectives_from_metrics(), _diversity_penalty_blended(), _phenotype_bucket_key(), Discretise objective-relevant metrics for behavioral diversity., Hamming OR phenotype-bucket crowding penalty (same weight on both)., Penalty for weak cross-symbol robustness on one split., Build Phase 2 minimisation objectives from precomputed train/val metrics.…, _symbol_robustness_penalty() (+29 more)

### Community 138 - "set_purged_wf_reference_rows"
Cohesion: 0.18
Nodes (7): Store full train_new.csv row count after loader prep (split time)., set_purged_wf_reference_rows(), holdout_mode(), purged_mode(), fixture, Tests for purged-WF trade-floor scaling helpers., TestScaleTradeFloor

### Community 139 - ".load_pool"
Cohesion: 0.13
Nodes (7): Load existing pool if valid, return None if missing., Merge local pool, symbol archive, and shared archive (dominant seeds)., Return loaded pool if valid, None if need to run., Validate the structure of a loaded pool JSON. Raises ValueError if the schema…, _validate_pool_schema(), TestLoadPool, TestValidatePoolSchema

### Community 140 - "TestNClustersDefined"
Cohesion: 0.33
Nodes (4): AC: n_clusters is assigned inside _run_cluster_islands so the migration guard…, n_clusters must be assigned in _run_cluster_islands for the migration guard at…, The migration guard must reference n_clusters., TestNClustersDefined

### Community 141 - "TestF3PathResolution"
Cohesion: 0.40
Nodes (4): parametrize, Parametrized tests for f3 path resolution (Task 5: audit finding #5). Verifies…, Verify the correct f3 formula runs for each (USE_TOTAL_RETURN_OBJ,…, TestF3PathResolution

### Community 142 - "TestIslandSchedulerGlobalMode"
Cohesion: 0.17
Nodes (7): Unit tests for island scheduler global-mode safety. Acceptance criteria…, AC-T1.4: Global mode must never reach migration code., When PHASE2_ISLAND_MODE='global', _run_cluster_islands is not called., Verify the migration guard would not be reached in global mode., The top-level dispatch should only call run_cluster_phase2 in cluster mode., In global mode, the lazy import of extract_deployable_migrants never fires., TestIslandSchedulerGlobalMode

### Community 143 - ".load_dataset"
Cohesion: 0.38
Nodes (6): _ensure_labels(), DataFrame, Keep supplied labels or derive all labels from raw OHLCV columns. The original…, Load a CSV dataset with full preparation pipeline: 1. Read CSV with comma…, Validate the mandatory trend-context contract on an enriched frame. Fails…, validate_context_columns()

### Community 144 - "TestRefreshObjectivesOnResumeGate"
Cohesion: 0.23
Nodes (7): Task-10: gate cache refresh on PHASE2_PER_EPOCH_WINDOW_ROTATION. Verifies the…, Evaluate the gate expression as it appears in the source., PHASE2_PER_EPOCH_WINDOW_ROTATION=True (default): refresh = not first_epoch →…, PHASE2_PER_EPOCH_WINDOW_ROTATION=True (default): refresh = not first_epoch →…, PHASE2_PER_EPOCH_WINDOW_ROTATION=False (legacy): refresh is False even on…, PHASE2_PER_EPOCH_WINDOW_ROTATION=False (legacy): refresh is False on first…, TestRefreshObjectivesOnResumeGate

### Community 145 - "conftest.py"
Cohesion: 0.22
Nodes (10): Config, FixtureRequest, Item, _close_matplotlib_figures(), _low_memory_cleanup(), fixture, pytest_collection_modifyitems(), pytest_configure() (+2 more)

### Community 146 - "_derive_val_sample_seed"
Cohesion: 0.19
Nodes (9): _derive_val_sample_seed(), Derive a deterministic validation sample seed from the training seed. This…, AC: Train and validation sampling use distinct RNG seeds by default., _derive_val_sample_seed returns a value different from the input., Same train seed always produces same val seed., Result is in [0, 2**31) so it is a valid random seed., Rule_Pool_Generator stores distinct _sample_seed and _val_sample_seed., When seed=None, val seed is derived from PHASE2_SEED. (+1 more)

### Community 147 - ".skip_if_valid"
Cohesion: 0.33
Nodes (3): Check if output files exist and are valid. Returns ------- dict[str,…, fixture, TestSkipIfValid

### Community 148 - "_validate_schema"
Cohesion: 0.31
Nodes (3): Validate the structure of a loaded feature selection JSON. Raises ValueError if…, _validate_schema(), TestValidateSchema

### Community 149 - "strategy_id"
Cohesion: 0.24
Nodes (9): feature_conditions_only(), phase2_rule_id(), Any, Stable provenance identifiers for Phase 2 feature rules. The RB Governor may…, Return normalized non-symbol, non-context conditions in order. Symbol scope and…, Hash the immutable Phase 2 feature logic and its discovery scope., Hash the complete economic strategy, including exit policy. ``phase2_rule_id``…, strategy_id() (+1 more)

### Community 150 - "test_phase2_use_gpu_flag.py"
Cohesion: 0.25
Nodes (7): _minimal_backtest_df(), DataFrame, MonkeyPatch, Phase 2 respects PHASE2_USE_GPU=False without loading JAX GPU engine., The memory-safe CPU route must happen before JAX allocates arrays., A selected CPU backend must not initialize JAX just to warm up., TestPhase2UseGpuFlag

### Community 152 - "BFS and DFS Graph Traversal"
Cohesion: 0.20
Nodes (10): CLAUDE.md Graphify Integration, Constrained Query Expansion, BFS and DFS Graph Traversal, Graph Work Memory, Existing Graph Fast Path, CLAUDE.md Graphify Integration, Constrained Query Expansion, BFS and DFS Graph Traversal (+2 more)

### Community 153 - ".get_dont_care"
Cohesion: 0.17
Nodes (7): See module-level :func:`get_dont_care`., **Property 9: Don't-Care Sentinel Correctness — encode_condition raises**…, **Property 9: Don't-Care Sentinel Correctness — all-dont_care → empty output**…, test_property_9b_encode_condition_raises_for_dont_care(), test_property_9f_all_dont_care_chromosome_returns_empty(), Static methods should be callable on the class itself., TestGetDontCare

### Community 154 - "test_phase2_rule_pool.py"
Cohesion: 0.07
Nodes (21): _crowding_distance(), _hamming_distance(), Compute crowding distance for solutions in *front*. Parameters ----------…, Hamming distance between two chromosomes (active pairs when sparse)., _chromosome_with_min_active(), _isolate_phase2_archive_paths(), _pop_contains_dense_seed(), fixture (+13 more)

### Community 155 - "TestMonthlyGateDataSource"
Cohesion: 0.29
Nodes (6): DataFrame, Verify the gate sources its DataFrame from ``_cached_slim_val`` (val, not…, Create a minimal training DataFrame similar to test_phase2_rule_pool., ``build_monthly_windows`` receives unsampled monthly val…, When no val_df was provided, ``_cached_slim_val`` is None and the gate is…, TestMonthlyGateDataSource

### Community 156 - "apply_fuzzy_feature_scaling"
Cohesion: 0.29
Nodes (9): apply_fuzzy_feature_scaling(), fit_fuzzy_feature_scaling(), Any, DataFrame, Train-fitted normalization for ordinal ``ff_*`` fuzzy feature codes. The…, Build a train-only scaling contract for ordinal ``ff_*`` columns., Apply an existing train-fitted scaling contract in place and return *df*., Tests for train-fitted ordinal fuzzy feature scaling. (+1 more)

### Community 157 - "phase2_rule_pool.py"
Cohesion: 0.06
Nodes (55): _archive_direction(), _archive_objective_vector(), attach_cv_fold_returns_batch(), _build_cpu_archive_engine(), _build_pool_from_archive(), _chromosome_batch(), _chromosome_for_pool_export(), _condition_feature_names() (+47 more)

### Community 158 - "TestGlobalMetricsCacheClearing"
Cohesion: 0.27
Nodes (6): When seeded_keys is empty, nothing is removed., When all keys are seeded, cache becomes empty., Verify that only seeded keys are removed from the global cache., Create a deterministic chromosome key., Cache entries matching seeded_keys are removed; non-matching survive., TestGlobalMetricsCacheClearing

### Community 159 - "TestMigrationSeedFraction"
Cohesion: 0.20
Nodes (6): Verify that migrant injection uses PHASE2_MIGRATION_SEED_FRACTION, not ARCHIVE., PHASE2_MIGRATION_SEED_FRACTION defaults to 0.10., Ensure the migration fraction is not the same as archive fraction., Simulate the run_epoch migrant path and assert local_cap uses migration…, Simulate the new run_epoch logic: migrant entries are capped by migration…, TestMigrationSeedFraction

### Community 160 - "._prune_splits_after_phase1"
Cohesion: 0.25
Nodes (4): Selected fuzzy features for Phase 2., Drop unused feature columns from train/val splits to reduce RAM., Drop unused feature columns from train split (legacy single-split API)., TestPruneSplitsAfterPhase1

### Community 161 - "_legacy_writer_contract"
Cohesion: 0.67
Nodes (3): _legacy_writer_contract(), fixture, These schema tests predate mandatory trend context.

### Community 162 - "TestDeferredWarmup"
Cohesion: 0.20
Nodes (6): Unit tests for the ``defer_warmup`` flag on ``Rule_Pool_Generator``. When…, Existing callers without defer_warmup still warm at init., The configure_phase2_gpu_runtime call is inside 'if not self._defer_warmup:'…, _run_cluster_islands passes defer_warmup=True to all generators., _run_cluster_islands calls warmup_phase2_gpu_kernels per cluster., TestDeferredWarmup

### Community 164 - "TestMakeFoldEnginesTailHoldout"
Cohesion: 0.27
Nodes (6): DataFrame, With tail_holdout_frac=0.25, a tail engine is returned with ~25% of data., With tail_holdout_frac=0.0, no tail engine., Single symbol with tail holdout still works., Verify tail holdout engine is created when fraction > 0., TestMakeFoldEnginesTailHoldout

### Community 165 - "_pareto_sortino_stats"
Cohesion: 0.47
Nodes (3): _pareto_sortino_stats(), Aggregate raw Sortino and return health over the current Pareto front., TestParetoSortinoStats

### Community 166 - "test_rb_full_validation_recovery.py"
Cohesion: 0.38
Nodes (6): _frame(), _metrics(), DataFrame, Path, Regression tests for the bounded validation-only RB recovery pass., test_recovery_retries_only_rejected_direction_on_full_validation()

### Community 167 - "test_phase2_batch_evaluator_parity.py"
Cohesion: 0.40
Nodes (5): _backtest_df(), DataFrame, Regression tests for evaluator-facing Phase 2 chromosome semantics., Search fitness must use the same fuzzy class as RB/Phase 5 evaluation., test_batch_chromosome_signals_match_decoded_rule_conditions()

### Community 168 - "._attach_run_log_handler"
Cohesion: 0.40
Nodes (3): FileHandler, Attach a FileHandler to the root logger writing to RUN_LOG_PATH. Returns the…, Remove *handler* from the root logger and close it.

### Community 169 - "TestConfigPhase2Seed"
Cohesion: 0.40
Nodes (3): Phase 2 reproducibility default seed., PHASE2_SEED is drawn once at import via get_seed(); set GLOBAL_SEED=42 to fix…, TestConfigPhase2Seed

### Community 171 - "context_contract_digest"
Cohesion: 0.50
Nodes (4): context_contract(), context_contract_digest(), Return the full context contract for strategy/dataset identity., Return a stable hash of the static contract and fitted enrichment.

### Community 172 - "TestHallOfFameTrim"
Cohesion: 0.25
Nodes (5): Verify that hall_of_fame is capped to PHASE2_HOF_EPOCH_CARRYOVER=10., HoF with >10 entries is trimmed to exactly 10 oldest entries., HoF with <=10 entries is left unchanged., Empty HoF does not crash during trim logic., TestHallOfFameTrim

### Community 173 - "trim_evolution_state_memory"
Cohesion: 0.50
Nodes (4): Bound run-wide eval cache size to limit RAM growth across long runs. Uses FIFO…, Drop bulky resumable state that is already persisted elsewhere., trim_evolution_state_memory(), _trim_global_metrics_cache()

### Community 174 - "test_phase2_offspring_batch.py"
Cohesion: 0.29
Nodes (5): CountingEngine, Unit tests for batched offspring evaluation (Phase 2 runtime A1)., Stub engine recording every simulate_rule_batch call's batch size., Offspring should be evaluated via ONE simulate_rule_batch call per gen, not…, test_offspring_evaluated_in_single_batch_per_gen()

### Community 176 - "test_gpu_engine_import_does_not_crash_on_jax_failure"
Cohesion: 0.33
Nodes (6): parametrize, Tests for the lazy-JAX failure path in gpu_engine.py. These tests verify that…, Verify importing gpu_engine.py handles various JAX failure modes gracefully., Build a subprocess script that simulates JAX import failure., _subprocess_script(), test_gpu_engine_import_does_not_crash_on_jax_failure()

### Community 177 - "test_jax_compat.py"
Cohesion: 0.29
Nodes (5): parametrize, Tests for JAX / GPU engine availability detection., Package init must not crash when gpu_engine import fails., test_cpu_engine_import_without_jax(), test_get_gpu_backtest_engine_class_returns_none_on_import_error()

### Community 180 - "_apply_colab_gpu_defaults"
Cohesion: 0.33
Nodes (6): _apply_colab_gpu_defaults(), is_colab_runtime(), True when running on Google Colab (/content runtime)., Colab T4 optimization for Phase 2 runs., MonkeyPatch, test_colab_defaults_apply_when_content_exists()

### Community 184 - "TestEvalCvFoldReturns"
Cohesion: 0.33
Nodes (4): Verify the helper handles None / empty fold_engines safely., fold_engines=None returns None without crashing., fold_engines=[] returns None without crashing., TestEvalCvFoldReturns

### Community 189 - "opencode.json"
Cohesion: 0.50
Nodes (3): plugin, $schema, .opencode/plugins/graphify.js

## Knowledge Gaps
- **35 isolated node(s):** `$schema`, `.opencode/plugins/graphify.js`, `gpu-fuzzy-trader`, `URL Ingestion`, `Folder Watcher` (+30 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `CPUBacktestEngine` connect `CPUBacktestEngine` to `TestMakeWalkForwardFoldEngines`, `_make_engine`, `_score_metrics`, `_make_df`, `Rule_Pool_Generator`, `test_cpu_engine_properties.py`, `test_phase2_use_gpu_flag.py`, `CandidateRecord`, `phase2_island_scheduler.py`, `phase2_rule_pool.py`, `ValidationError`, `TestMakeFoldEnginesTailHoldout`, `test_phase2_batch_evaluator_parity.py`, `gpu_engine.py`, `test_jax_compat.py`, `_compute_rule_signal_mask`, `.__init__`, `TestEvalCvFoldReturns`, `GPUBacktestEngine`, `ValueError`, `nested_walk_forward.py`, `_symbol_specialized_variants`, `_apply_dynamic_rule`, `test_cpu_engine.py`, `TestGPUCPUNumericalParity`, `test_gpu_engine.py`, `Data_Loader`, `filter_migrants_for_cluster`, `prop_settings`, `test_trend_context.py`, `baselines.py`, `_jax_compute_trade_outcomes`, `_jax_compute_rule_signals`, `test_rb_governor_tail_holdout.py`, `MonthlyWindowSummary`, `._engine`, `OOS_Evaluator`?**
  _High betweenness centrality (0.140) - this node is a cross-community bridge._
- **Why does `Rule_Pool_Generator` connect `Rule_Pool_Generator` to `.save_archive`, `CPUBacktestEngine`, `log_memory_rss`, `_derive_epoch_seed`, `Phase2EvolutionState`, `_get_dont_cares`, `TestEvictClusterSignatures`, `TestRulePoolGeneratorRun`, `compute_phase2_objectives_from_metrics`, `.load_pool`, `.run`, `TestNClustersDefined`, `Pipeline_Orchestrator`, `_apply_monthly_admission_gate`, `_dominates`, `TestF3PathResolution`, `_derive_val_sample_seed`, `TestRefreshObjectivesOnResumeGate`, `_make_train_df`, `test_phase2_use_gpu_flag.py`, `run_pipeline.py`, `Reporter`, `test_phase2_rule_pool.py`, `TestMonthlyGateDataSource`, `phase2_island_scheduler.py`, `phase2_rule_pool.py`, `TestDeferredWarmup`, `_pareto_sortino_stats`, `test_phase2_window_rotation.py`, `TestConfigPhase2Seed`, `passes_pool_admission_gate`, `_mutate`, `test_phase2_rule_pool_properties.py`, `.__init__`, `test_phase2_island_scheduler.py`, `test_plateau_state_leak.py`, `filter_migrants_for_cluster`, `test_crash_fix_and_run_logging.py`, `TestRunLogHandlerLifecycle`, `TestEndToEndRotation`, `_derive_island_seed`, `_pool_seed_chromosomes`, `TestValLeakGate`?**
  _High betweenness centrality (0.124) - this node is a cross-community bridge._
- **Why does `Reporter` connect `Reporter` to `TestEquityCurveDateAxis`, `TestPlotDistributionAndEquity`, `TestWriteStrategyEvaluationTable`, `._ensure_dir`, `Rule_Pool_Generator`, `Data_Loader`, `TestPlotPerRuleBreakdown`, `DataFrame`, `prop_settings`, `test_reporter.py`, `phase2_rule_pool.py`, `OOS_Evaluator`?**
  _High betweenness centrality (0.106) - this node is a cross-community bridge._
- **Are the 15 inferred relationships involving `Reporter` (e.g. with `CvFoldValEvaluator` and `Rule_Pool_Generator`) actually correct?**
  _`Reporter` has 15 INFERRED edges - model-reasoned connections that need verification._
- **Are the 59 inferred relationships involving `CPUBacktestEngine` (e.g. with `GPUBacktestEngine` and `JointPortfolioEngine`) actually correct?**
  _`CPUBacktestEngine` has 59 INFERRED edges - model-reasoned connections that need verification._
- **Are the 65 inferred relationships involving `Rule_Pool_Generator` (e.g. with `CPUBacktestEngine` and `Reporter`) actually correct?**
  _`Rule_Pool_Generator` has 65 INFERRED edges - model-reasoned connections that need verification._
- **Are the 31 inferred relationships involving `Output_Writer` (e.g. with `OOS_Evaluator` and `_Phase5JSONEncoder`) actually correct?**
  _`Output_Writer` has 31 INFERRED edges - model-reasoned connections that need verification._