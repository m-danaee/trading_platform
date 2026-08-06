# Graph Report - trading_platform  (2026-08-07)

## Corpus Check
- 210 files · ~307,361 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 4824 nodes · 10934 edges · 172 communities (161 shown, 11 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 550 edges (avg confidence: 0.57)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `a6b5c0ba`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- evox_runner.py
- .run
- rb_governor.py
- _make_engine
- _get_dont_cares
- test_evox_runner.py
- TestRulePoolGeneratorRun
- test_rb_governor_cv_folds.py
- Rule_Pool_Generator
- _make_df
- phase2_rule_pool.py
- run_phase2_evolution
- _make_rule
- Data_Splitter
- Pipeline_Orchestrator
- _apply_monthly_admission_gate
- Feature_Detector
- _make_train_df
- Output_Writer
- test_cpu_engine_properties.py
- detect_feature_mode
- selector.py
- Feature_Selector
- cv_folds_only
- Reporter
- test_encoder_properties.py
- write_evaluator_clean
- CandidateRecord
- _derive_val_sample_seed
- environmental_selection_nsga2
- maybe_log_generation
- Hybrid CPU and GPU execution policy
- _select_diverse_subset
- test_rb_min_symbols.py
- Graphify Pipeline
- ValidationError
- config.py
- TestEquityCurveDateAxis
- _make_multi_sym_df
- research_integrity.py
- _loader_from_rows
- _split
- gpu_engine.py
- _gpu_runtime.py
- ._ensure_dir
- _build_entries_from_rule_set
- _dominates
- non_dominated_sort
- optuna_search.py
- _mutate
- test_phase2_rule_pool_properties.py
- _compute_rule_signal_mask
- nested_walk_forward.py
- build_hybrid_symbol_clusters
- _run_cluster_islands
- trend_context.py
- detector.py
- test_reporter.py
- GPUBacktestEngine
- cpu_engine.py
- ValueError
- resolve_phase2_stage_params
- TestEquityCurvePlots
- TestOOSEvaluatorRun
- test_plateau_state_leak.py
- passes_pool_admission_gate
- dashboard.py
- splitter.py
- test_data_loader_properties.py
- barrier.py
- _m
- _apply_dynamic_rule
- test_cpu_engine.py
- run_rb_governor_pipeline
- split_validation_fitness_selection
- TestWriteStrategyEvaluationTable
- test_gpu_engine.py
- rolling_cv.py
- test_rolling_cv.py
- test_feature_selector_properties.py
- _NumpyJSONEncoder
- test_phase5_oos.py
- TestPlotPerRuleBreakdown
- resolve_island_hyperparams
- evaluator_health.py
- _build_target
- filter_migrants_for_cluster
- parity_scenario_strategy
- test_crash_fix_and_run_logging.py
- test_property_27_test_data_preparation_consistency
- run_pipeline.py
- TestEndToEndRotation
- phase2_support.py
- .encode_condition
- test_output_writer_properties.py
- Data_Loader
- test_phase2_gpu_throughput.py
- .decode_chromosome
- .load_strategies
- gate_positive_good
- phase2_island_scheduler.py
- compute_labels
- _remove_low_dispersion
- prop_settings
- test_crash_fix_properties.py
- TestPlotDistributionAndEquity
- baselines.py
- TestScaleTradeFloor
- _jax_compute_trade_outcomes
- _should_post_restart_early_stop_phase2
- test_evaluator_health.py
- ConfigurationError
- execution_ok
- _validate_pool_schema
- TestHammingThresholdAutoScale
- TestValLeakGate
- DataFrame
- _jax_compute_rule_signals
- TestParetoCollapseWarningGate
- CPUBacktestEngine
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
- _make_walk_forward_fold_engines
- log_memory_rss
- DataFrame
- test_phase2_init.py
- _apply_colab_gpu_defaults
- TestPlateauDiversityRestart
- TestContextEntryPaths
- compute_phase2_objectives_from_metrics
- test_nsga3_selection.py
- .load_pool
- .detect_feature_mode
- TestSavePerSymbolCsv
- TestIslandSchedulerGlobalMode
- .get_dont_care
- TestRefreshObjectivesOnResumeGate
- conftest.py
- .skip_if_valid
- _validate_schema
- test_phase2_use_gpu_flag.py
- BFS and DFS Graph Traversal
- TestGPUCPUNumericalParity
- test_phase2_rule_pool.py
- _build_pool_from_archive
- TestGlobalMetricsCacheClearing
- TestMigrationSeedFraction
- _should_plateau_early_stop_phase2
- test_phase2_support.py
- TestHallOfFameTrim
- test_gpu_engine_import_does_not_crash_on_jax_failure
- test_jax_compat.py
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
2. `CPUBacktestEngine` - 154 edges
3. `Rule_Pool_Generator` - 138 edges
4. `Output_Writer` - 85 edges
5. `prop_settings()` - 79 edges
6. `Pipeline_Orchestrator` - 67 edges
7. `_run_nsga3()` - 58 edges
8. `Feature_Selector` - 58 edges
9. `OOS_Evaluator` - 58 edges
10. `Data_Loader` - 57 edges

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

## Communities (172 total, 11 thin omitted)

### Community 0 - "evox_runner.py"
Cohesion: 0.04
Nodes (104): _assign_eval_result(), _behavior_descriptor(), _binary_tournament_pick(), _build_diversity_reference(), _build_rank_and_crowding(), _constraint_violations(), _count_deployable_preview(), _count_pop_viable() (+96 more)

### Community 1 - ".run"
Cohesion: 0.08
Nodes (29): _archive_feature_signature(), _condition_feature_names(), _filter_compatible_previous_pool(), _filter_pool_by_admission(), _merge_archive_entries(), _pool_path_key(), Any, Return the ordered feature signature used to validate archive reuse. (+21 more)

### Community 2 - "rb_governor.py"
Cohesion: 0.05
Nodes (85): expectancy_lcb_pct(), Return a conservative lower bound for per-trade net expectancy. Exact CPU…, _attach_source_symbol_filters(), _available_symbols(), _balanced_phase2_shortlist(), _cost_stress_gate(), _ensure_symbol_filtered_rule(), _eval_cv_fold_returns() (+77 more)

### Community 3 - "_make_engine"
Cohesion: 0.08
Nodes (22): _make_df(), _make_engine(), All TP trades → win_rate = 100%., No losing trades → profit_factor = 99.0., Build a minimal DataFrame for testing., position_notional = equity * capital_pct/100 * leverage when exposure is free., position_notional is capped when exposure is nearly full., When exposure is full, position_notional = 0. (+14 more)

### Community 4 - "_get_dont_cares"
Cohesion: 0.10
Nodes (23): _count_active_conditions(), _get_dont_cares(), _init_population(), Initialise a population of chromosomes. *init_strategy* ``"stratified_sparse"``…, Return array of dont_care sentinels for each feature., Count active rule conditions (sparse slots or dense dont_care encoding)., _feature_infos_with_scores(), TestStratifiedInitPopulation (+15 more)

### Community 5 - "test_evox_runner.py"
Cohesion: 0.05
Nodes (44): _inherit_val_metrics_from_global_cache(), _inject_diversity_recovery(), Phase2EvolutionState, Replace a fraction of the population with fresh or archive-mutated seeds., Copy val_* from global cache for identical chromosomes when val is skipped.…, Evolve one island epoch and return updated resumable state., Resumable NSGA-III state for symbol-island epoch scheduling., Return survivors that do not already carry a validation snapshot. Validation… (+36 more)

### Community 6 - "TestRulePoolGeneratorRun"
Cohesion: 0.13
Nodes (8): Integration tests using tiny population and generation counts., Verify that Rule_Pool_Generator's persistent self._rng advances across multiple…, After two run_epoch() calls, the internal RNG state must differ from the…, The RNG should produce *different* sequences in two consecutive run_epoch()…, Two generators with different seeds must have different RNG state., Rule_Pool_Generator must initialize self._rng as a Generator., TestRulePoolGeneratorRng, TestRulePoolGeneratorRun

### Community 7 - "test_rb_governor_cv_folds.py"
Cohesion: 0.08
Nodes (30): _combined_return_score(), Profit objective for lenient rule addition, but now evaluator_v5 aware. A new…, _MockEngine, Unit tests for CV-fold and PF/DD penalty code paths in RB Governor. Covers: -…, Minimal mock that mimics CPUBacktestEngine for testing _optimize_risk.…, Verify the CV-fold consistency penalty in ``_score_metrics``. New formula…, cv_fold_returns=[5.0, -2.0, 3.0] — worst fold negative + high variance., All positive folds with low variance — no penalty. (+22 more)

### Community 8 - "Rule_Pool_Generator"
Cohesion: 0.06
Nodes (23): extract_deployable_migrants(), Return elite deployable-preview entries suitable for guarded migration., _exchange_migrants_between_islands(), Perform a guarded, order-independent migration exchange. Islands are processed…, _deployable_archive_pool_entries(), Convert deployable archive rows into pool-shaped entries for Stage B seeding., Phase 2: GPU-accelerated multi-objective evolutionary rule pool generation.…, Inject guarded migration seeds for the next epoch. (+15 more)

### Community 9 - "_make_df"
Cohesion: 0.08
Nodes (21): _make_df(), _make_engine(), MonkeyPatch, Chromosome positions must follow feature_modes insertion order., Exact CPU re-evaluation must interpret genes like the GPU path., Chromosome that matches nothing returns 0 executed trades., Zero-signal chunks should use reject metrics without scanning., PHASE2_SKIP_ZERO_SIGNAL_SCAN must match always-scan metrics. (+13 more)

### Community 10 - "phase2_rule_pool.py"
Cohesion: 0.10
Nodes (51): encoder.py — Encoder Maps gene integer values to fuzzy value names, formats…, _random_active_class(), _crossover(), _diversity_penalty_blended(), _pool_seed_chromosomes(), phase2_rule_pool.py — Rule_Pool_Generator (Phase 2) GPU-accelerated multi-…, Uniform crossover (dense per-gene or sparse per-slot)., Hamming OR phenotype-bucket crowding penalty (same weight on both). (+43 more)

### Community 11 - "run_phase2_evolution"
Cohesion: 0.04
Nodes (44): StageLabel, Run Phase 2 NSGA-III evolution. Returns (pareto_pool, history) or state., run_phase2_evolution(), Evolutionary algorithm drivers for Phase 2., skipif, TestRunPhase2EvolutionFallback, TestRunPhase2EvolutionSmoke, Unit tests for Pareto-collapse warning gate (audit finding #13). AC: The… (+36 more)

### Community 12 - "_make_rule"
Cohesion: 0.11
Nodes (10): _make_rule(), _make_rule_set(), A rule with only tp non-zero should be accepted., Spot-check a variety of valid fuzzy value names., Write rule_set to a temp file and reload the raw JSON., TestWriteAllZeroRejection, TestWriteConditionValidation, TestWriteHappyPath (+2 more)

### Community 13 - "Data_Splitter"
Cohesion: 0.12
Nodes (23): Data_Splitter, load_cached_split_if_fresh(), Load cached split parquets when they are newer than the source CSV. Validates…, Chronological train/validation splitter., Module-level wrapper around ``Data_Splitter.split_and_persist``., split_and_persist(), _make_df(), _make_timestamps() (+15 more)

### Community 14 - "Pipeline_Orchestrator"
Cohesion: 0.04
Nodes (70): FileHandler, _context_coverage_for_direction(), _context_coverage_preflight(), _context_coverage_report(), _log_phase_entry(), _log_pipeline_config(), _now_iso(), _phase5_test_metrics() (+62 more)

### Community 15 - "_apply_monthly_admission_gate"
Cohesion: 0.06
Nodes (29): _apply_monthly_admission_gate(), _evaluate_rule_on_window(), _monthly_window_metrics(), Evaluate a single pool rule on a single monthly window. Returns the full window…, Normalize a window evaluator result for the monthly gate. The float fallback…, Apply the monthly-window shadow-test gate to a pool of rules. Each rule is…, _DeterministicEvaluator, _DeterministicMetricsEvaluator (+21 more)

### Community 16 - "Feature_Detector"
Cohesion: 0.08
Nodes (48): Feature_Detector, Classify feature columns by their discretization type., all_nan_series(), arbitrary_numeric_series(), binary_series(), large_value_series(), positive_series(), composite (+40 more)

### Community 17 - "_make_train_df"
Cohesion: 0.08
Nodes (23): Sample up to *total_rows* rows, distributed equally across symbols. A single…, _sample_df(), _make_train_df(), DataFrame, Critical: bars must be contiguous so the backtest engine preserves temporal…, Random start must be bounded so the slice always fits forward., divmod distribution gives exactly total_rows (no rounding loss). 701_000 % 14…, total_rows < n_symbols must NOT force 1 row per symbol. (+15 more)

### Community 18 - "Output_Writer"
Cohesion: 0.09
Nodes (15): Output_Writer, Serializes RuleSet dicts to JSON and loads/validates existing JSON files.…, _legacy_writer_contract(), fixture, Unit tests for gpu_fuzzy_trader.output.writer — Output_Writer Tests cover: -…, These schema tests predate mandatory trend context., Verify the exact example from the spec works end-to-end., TestLoadAndValidateErrors (+7 more)

### Community 19 - "test_cpu_engine_properties.py"
Cohesion: 0.08
Nodes (43): equity_tracking_scenario(), _expected_outcome(), fee_deduction_scenario(), _make_engine(), _make_engine_custom(), _make_minimal_df(), _make_simple_df(), multi_symbol_scenario() (+35 more)

### Community 20 - "detect_feature_mode"
Cohesion: 0.05
Nodes (38): detect_feature_mode(), Module-level convenience wrapper around Feature_Detector.detect_feature_mode., Unit tests for gpu_fuzzy_trader.features.detector.Feature_Detector Tests cover:…, All non-negative, zero_ratio > 0.3 → sparse_positive., All zeros: zero_ratio = 1.0 > 0.3, min = 0 → sparse_positive., zero_ratio just above 0.3 → sparse_positive., zero_ratio must be computed on the full series, not just non-NaN., Has negative values, zero_ratio > 0.3 → sparse_signed. (+30 more)

### Community 21 - "selector.py"
Cohesion: 0.08
Nodes (39): get_seed(), Return a per-process random seed (stable within a run, different across runs).…, _align_feature_array(), build_phase1_shared_context(), _build_symbol_masks(), _candidate_feature_columns(), _check_spearman_sign_consistency(), _compute_chronological_stationarity_scores() (+31 more)

### Community 22 - "Feature_Selector"
Cohesion: 0.10
Nodes (20): Feature_Selector, Score and rank features separately for long and short directions., _make_train_df(), MonkeyPatch, Create a minimal training DataFrame with label columns and feature columns., Label columns must not appear in selected features., Meta columns must not appear in selected features., Loader internal columns and ``_``-prefixed names are not candidates. (+12 more)

### Community 23 - "cv_folds_only"
Cohesion: 0.17
Nodes (9): aggregate_fold_metrics(), cv_folds_only(), FoldMetricsSummary, CV folds excluding the primary holdout., Summarize metrics across folds (worst-case emphasis)., Collapse per-fold metrics into one dict for objectives / gates. ``worst``: min…, summarize_fold_metrics(), CV valid blocks differ by at most 1 bar when remaining % n != 0. (+1 more)

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
Cohesion: 0.13
Nodes (36): _candidate_coverage_symbols(), _candidate_positive_symbols(), CandidateRecord, _compose_ruleset(), _diversification_beam(), _diversification_shortlist(), _max_overlap(), Return supported positive validation symbols for one candidate. (+28 more)

### Community 28 - "_derive_val_sample_seed"
Cohesion: 0.10
Nodes (23): context_coverage_for_direction(), context_coverage_report(), Any, DataFrame, Shared diagnostics for the mandatory direction-specific context contract., Return coverage diagnostics for every named frame and direction., Return permission, trigger, and conjunction coverage for *direction*., _derive_val_sample_seed() (+15 more)

### Community 29 - "environmental_selection_nsga2"
Cohesion: 0.09
Nodes (20): environmental_selection_nsga2(), Canonical NSGA-II truncation on a 2N merged population., _make_chromosome(), _make_deployable_entry(), ndarray, Without preservation, champion is evicted by gen ~8 under drift., At most TOP_K slots are overwritten by elite preservation., Preserved elite's objectives are reset to inf (forces re-eval). (+12 more)

### Community 30 - "maybe_log_generation"
Cohesion: 0.09
Nodes (18): generation_log_interval(), iteration_log_interval(), log_generation(), maybe_log_generation(), log_progress.py — Throttled progress logging for long pipeline loops., Return how often to log generation progress. Uses LOG_GENERATION_INTERVAL from…, Log generation progress when the step matches the throttle interval., Log first step, last step, and every *interval* steps in between. (+10 more)

### Community 31 - "Hybrid CPU and GPU execution policy"
Cohesion: 0.07
Nodes (37): Local low-memory test policy, Virtual environment command policy, Package compile gate, Locked research dependencies, Focused low-memory test gate, Python 3.11 CI runtime, Research CI workflow, Balanced Mode-A assumption (+29 more)

### Community 32 - "_select_diverse_subset"
Cohesion: 0.08
Nodes (23): _normalize_for_association(), Max-min Hamming diversity sampling: greedy pick farthest from chosen. Returns…, Rank-based normalization (robust to outliers like trade_penalty=50)., _select_diverse_subset(), ndarray, Tests for H5/M4/M5 evolution convergence behaviors. Covers: - HoF trimming at…, Verify _select_diverse_subset correctness for edge cases., Create n distinct dense chromosomes. (+15 more)

### Community 33 - "test_rb_min_symbols.py"
Cohesion: 0.11
Nodes (24): _symbols_in_rules(), _dummy_df(), _make_candidate_records(), _mock_train_metrics(), _multi_symbol_rules(), _no_symbol_rule(), DataFrame, Tests for RB Governor min-distinct-symbols hard gate. After final opt_rules… (+16 more)

### Community 34 - "Graphify Pipeline"
Cohesion: 0.06
Nodes (36): Folder Watcher, URL Ingestion, Conditional Graph Exports, Graphify MCP Server, Extraction Confidence Rubric, Deterministic Full-Path Node IDs, Semantic Hyperedges, Cross-Repository Graph Merge (+28 more)

### Community 35 - "ValidationError"
Cohesion: 0.12
Nodes (23): context_permission_column(), context_trigger_column(), Return the direction-specific permission column name., Return the direction-specific LWC pullback-reversal trigger column., _condition_feature(), _context_feature_direction(), Exception, writer.py — Output_Writer Serializes RuleSet dicts to JSON with exact schema… (+15 more)

### Community 36 - "config.py"
Cohesion: 0.06
Nodes (59): _config_check(), ConfigError, _debug_symbol_universe_size(), effective_config_snapshot(), effective_min_profitable_symbols(), effective_min_trade_pool_floor(), effective_min_trade_support(), effective_monthly_min_trades() (+51 more)

### Community 37 - "TestEquityCurveDateAxis"
Cohesion: 0.09
Nodes (21): _make_trade_log(), DataFrame, fixture, Unit tests for date-based x-axis in Reporter.plot_equity_curve (Task 10.3).…, When Entry_Time is absent, x-axis label is 'Trade #'., When Entry_Time is all-NaN, x-axis label is 'Trade #'., axhline is called (initial-capital line) in date mode., axhline is called (initial-capital line) in Trade # mode. (+13 more)

### Community 38 - "_make_multi_sym_df"
Cohesion: 0.09
Nodes (23): _largest_safe_range(), Return (start, end) of the largest contiguous bar range not in *forbidden*. The…, Cap *total_rows* so the per-symbol request fits within the safe range. When…, _resolve_sample_total_rows(), _make_multi_sym_df(), DataFrame, Capping logic for per-epoch window rotation., With PHASE2_PER_EPOCH_WINDOW_ROTATION=False, total_rows is unchanged. (+15 more)

### Community 39 - "research_integrity.py"
Cohesion: 0.07
Nodes (42): context_contract(), context_contract_digest(), Return the full context contract for strategy/dataset identity., Return a stable hash of the static contract and fitted enrichment., feature_conditions_only(), phase2_rule_id(), Any, Stable provenance identifiers for Phase 2 feature rules. The RB Governor may… (+34 more)

### Community 40 - "_loader_from_rows"
Cohesion: 0.07
Nodes (30): load_dataset(), Module-level wrapper around ``Data_Loader.load_dataset``., _base_row(), _loader_from_rows(), _make_csv(), _make_ohlcv_rows(), _make_rows(), _make_timestamps() (+22 more)

### Community 41 - "_split"
Cohesion: 0.06
Nodes (21): Helper: build df, patch paths, run split, return (train, val)., Compute expected train/val per symbol under holdout+embargo split., floor(N * HOLDOUT_TRAIN_FRACTION) rows go to train., Remaining rows after embargo gap go to validation., For N=101: floor(101 * HOLDOUT_TRAIN_FRACTION) used, not round., train + validation + embargo dropped == total rows., train + val + embargo_dropped == total for each symbol., Each symbol's split point is computed from its own row count. (+13 more)

### Community 42 - "gpu_engine.py"
Cohesion: 0.07
Nodes (34): _batch_metrics_from_array(), _build_event_batch(), _exposure_slot_capacity(), _fast_reject_result_rows(), _jax_compute_rule_signals_sparse_batch(), _jax_open_slot(), _jax_release_open_slots(), _jax_simulate_equity_event_batch() (+26 more)

### Community 43 - "_gpu_runtime.py"
Cohesion: 0.07
Nodes (40): phase2_should_enrich_symbol_metrics(), Return True when GPU batch eval should run a follow-up CPU enrichment pass.…, configure_phase2_gpu_runtime(), detect_gpu_memory_used_gb(), detect_gpu_vram_gb(), detect_system_ram_gb(), _iter_warmup_targets(), log_gpu_runtime_config() (+32 more)

### Community 44 - "._ensure_dir"
Cohesion: 0.09
Nodes (17): _bucket_series_by_mode(), DataFrame, Series, Write feature-stratified performance metrics to CSV files (one per split). For…, Create parent directories for *path* if they do not exist., Compute maximum drawdown percentage from an equity series. Parameters…, Write compact train/validation/test generalization diagnostics to JSON. The…, Plot a per-rule performance breakdown across train/validation/test splits.… (+9 more)

### Community 45 - "_build_entries_from_rule_set"
Cohesion: 0.10
Nodes (14): _build_entries_from_rule_set(), Priority-based rule assignment: first matching rule wins per row. Mirrors…, DataFrame, Row matching both rules should be assigned to rule 1 only., Rows not matching rule 1 should be assigned to rule 2., v5: earlier JSON rule wins over lower dataset row index at same time., Release index should point to the row where bar_index + 288 is reached., Rows near the end should get release_index = len(df). (+6 more)

### Community 46 - "_dominates"
Cohesion: 0.14
Nodes (11): _archive_objective_vector(), _dominates(), _is_better_archive_entry(), _non_dominated_sort(), Return True if solution *a* dominates *b* (all objectives ≤, at least one <)., NSGA-II non-dominated sorting. Parameters ---------- objectives : np.ndarray…, Convert an archive entry into minimisation objectives for ranking., Return True when *candidate* should replace *incumbent* for the same chromosome. (+3 more)

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

### Community 52 - "nested_walk_forward.py"
Cohesion: 0.14
Nodes (23): Validation helpers for monthly, nested, and multiplicity-safe research., deflated_sharpe_ratio(), Selection-multiplicity diagnostics for strategy research artifacts., Return a normal-approximation deflated Sharpe probability. This is a…, Create a compact ledger-ready multiplicity report., summarize_multiplicity(), build_nested_folds(), evaluate_nested_strategy() (+15 more)

### Community 53 - "build_hybrid_symbol_clusters"
Cohesion: 0.12
Nodes (29): build_hybrid_symbol_clusters(), _corr_embedding_block(), _feature_names_union(), _feature_profile_block(), load_symbol_clusters(), Any, DataFrame, ndarray (+21 more)

### Community 54 - "_run_cluster_islands"
Cohesion: 0.05
Nodes (38): clear_global_metrics_cache(), Clear the global eval cache and force GC. Used to free RAM between cluster runs…, evict_cluster_signatures(), Evict JAX compiled signatures for a completed cluster. Removes entries from…, compute_cluster_generation_budgets(), Resolve per-island generation budgets. By default each island receives the full…, Check if an epoch should be skipped due to small remaining budget. Engine…, _run_cluster_islands() (+30 more)

### Community 55 - "trend_context.py"
Cohesion: 0.10
Nodes (41): align_completed_states_to_rows(), average_true_range(), build_higher_bars(), build_manifest(), _classify_hf_bars(), classify_regime(), compute_permissions_and_triggers(), enrich_tape() (+33 more)

### Community 56 - "detector.py"
Cohesion: 0.25
Nodes (5): detect_all_modes(), DataFrame, detector.py — Feature_Detector Classifies each feature column into exactly one…, Classify every column in *feature_cols* and return a mapping. Parameters…, Module-level convenience wrapper around Feature_Detector.detect_all_modes.

### Community 57 - "test_reporter.py"
Cohesion: 0.08
Nodes (26): _make_dataset_with_label(), _make_datasets_by_split(), _make_selected_features(), _make_stratified_dataset(), _make_stratified_datasets_by_split(), _make_stratified_logs_by_split(), _make_stratified_trade_log(), Unit tests for gpu_fuzzy_trader.reporting.reporter.Reporter Tests cover: -… (+18 more)

### Community 58 - "GPUBacktestEngine"
Cohesion: 0.10
Nodes (16): GPUBacktestEngine, CPU engine for rule-set simulation used by Phase 2/RB., Rule-set sim using a mask cache (delegates to the CPU engine)., Evaluate multiple rule sets (parallel CPU; optional mask cache)., RB JAX path: cached mask entry build + parallel CPU equity. Equity results…, Same interface as CPUBacktestEngine for compatibility., JAX-accelerated backtest engine for Phase 2 rule pool generation.…, Return the JAX backend in use ('gpu', 'cpu', or 'tpu'). (+8 more)

### Community 59 - "cpu_engine.py"
Cohesion: 0.08
Nodes (25): _append_allocated_entries(), _batch_eval_rule_set_pickled(), _parse_condition(), precompute_release_indices(), cpu_engine.py — CPUBacktestEngine Exact Python/NumPy replication of…, Simulate a rule set on rows [row_start, row_end) without copying the df. Used…, Simulate a rule set and return performance metrics. Parameters ----------…, Evaluate multiple rule sets in parallel (ProcessPool, thread fallback). (+17 more)

### Community 60 - "ValueError"
Cohesion: 0.07
Nodes (28): _build_rule_signal_mask(), _expectancy_lcb_pct(), _expected_shortfall_pct(), precompute_release_indices_from_offsets(), DataFrame, ndarray, Simulate using a precomputed rule-evaluation mask cache. The mandatory…, Drop cached entries on rows the fixed context mask forbids. ``idx`` validity is… (+20 more)

### Community 61 - "resolve_phase2_stage_params"
Cohesion: 0.07
Nodes (26): _diversity_recovery_min_unique_ratio(), True when Stage A viability is critically low and search has plateaued., _should_inject_diversity_recovery(), _should_viability_recovery(), island_stage_budgets(), IslandStagePlan, StageLabel, Phase 2 two-stage search hyperparameter profiles (exploration vs refinement). (+18 more)

### Community 62 - "TestEquityCurvePlots"
Cohesion: 0.11
Nodes (15): Build a flat list of per-symbol metric dicts for CSV output. Uses the…, DataFrame, Verify plot_equity_curve is called for all three splits and handles empty logs., Helper: run OOS_Evaluator.run() with mocked internals and capture plot calls., Override module-level path dicts and return originals., plot_equity_curve must be called with 'train', 'validation', and 'test'., plot_equity_curve must be called exactly three times (one per split)., Empty train trade log must not raise an exception. (+7 more)

### Community 63 - "TestOOSEvaluatorRun"
Cohesion: 0.11
Nodes (9): _isolate_phase5_reporter_outputs(), fixture, Keep Reporter plots/CSVs out of the checked-in outputs directory., Write a valid selected-features JSON to path., Integration tests using tmp_path overrides for all output paths. The run()…, Run OOS_Evaluator.run() once for the whole class (long direction only)., Override module-level path dicts and return originals (for standalone tests)., TestOOSEvaluatorRun (+1 more)

### Community 64 - "test_plateau_state_leak.py"
Cohesion: 0.11
Nodes (21): _make_minimal_gen(), _mock_evolution_state(), _mock_stage_plan(), Regression tests for plateau-state leak fixes (Fixes A + B). Fix A:…, AC-1, AC-3: plateau_streak resets per epoch; reset_plateau=True always., reset_plateau=True is passed on the very first epoch call., reset_plateau=True is also passed on epoch 2 (regression for leak)., Even when PHASE2_ISLAND_TWO_STAGE_ENABLED=False, reset_plateau=True. (+13 more)

### Community 65 - "passes_pool_admission_gate"
Cohesion: 0.06
Nodes (26): passes_pool_admission_gate(), Hard gate for Phase 2 pool/archive on merged holdout metrics. When…, MonkeyPatch, Tests for the hard overfit ratio gate in pool admission. The ratio gate rejects…, train=15%/val=4% (3.75× ratio, gap=11pp < 16pp) is REJECTED by the ratio gate…, train=15%/val=10% (1.5× ratio) is ADMITTED., train/val ≈2.5× is ADMITTED when under OVERFIT_RATIO_FLOOR and gap gate., With PHASE2_OVERFIT_RATIO_FLOOR=0.0, the ratio gate is disabled and the high-… (+18 more)

### Community 66 - "dashboard.py"
Cohesion: 0.15
Nodes (25): build_dashboard_data(), _direction_data(), _history_rows(), _load_json(), main(), Any, Path, Read-only HTML dashboard for existing pipeline artifacts. The dashboard… (+17 more)

### Community 67 - "splitter.py"
Cohesion: 0.18
Nodes (12): data/splitter.py — Data_Splitter Per-symbol chronological split for Phase 2,…, derive_primary_holdout(), load_cv_folds_manifest(), purged_config_fingerprint(), PurgedFold, Any, One expanding train + validation slice (per-symbol boundaries merged)., Return (train_df, val_df) from the primary holdout fold. (+4 more)

### Community 68 - "test_data_loader_properties.py"
Cohesion: 0.15
Nodes (26): dataframe_with_nan_features(), dataframe_with_nan_labels(), _load_from_df(), _make_datetime_series(), composite, DataFrame, DrawFn, given (+18 more)

### Community 69 - "barrier.py"
Cohesion: 0.15
Nodes (17): attach_barrier_outcomes(), barrier_column_names(), configured_barrier_pairs(), _first_touch_for_symbol(), _first_touch_for_symbol_numba(), _first_touch_for_symbol_python(), _number_token(), DataFrame (+9 more)

### Community 70 - "_m"
Cohesion: 0.10
Nodes (20): evaluator_health_penalty(), Penalty for evaluator_v5 failure modes — higher is worse. Parameters ----------…, _m(), Max positions 15 > 10 → positive penalty., Max positions 8 <= 10 → no additional penalty from positions., Missing ``raw_signal_count`` → treated as 0 → no skip/exec penalty., ``raw_signal_count=0`` → no skip/exec ratio computed → no penalty., Role 'train' same as 'valid' (1.0x). (+12 more)

### Community 71 - "_apply_dynamic_rule"
Cohesion: 0.21
Nodes (4): _apply_dynamic_rule(), Apply one exported text condition using the original threshold logic. Exactly…, Test all threshold branches of _apply_dynamic_rule., TestApplyDynamicRule

### Community 72 - "test_cpu_engine.py"
Cohesion: 0.08
Nodes (18): _normalize_direction(), Compute a non-annualized Sortino Ratio from per-trade returns., _safe_profit_factor(), _sortino_ratio_from_returns(), JointPortfolioEngine, DataFrame, Evaluate long and short rule books in one net-position account., normalize_symbol_value() (+10 more)

### Community 73 - "run_rb_governor_pipeline"
Cohesion: 0.09
Nodes (37): mandatory_context_conditions(), Return the fixed, mandatory context conditions for *direction*., _assert_capital_budget(), _assert_mandatory_context(), _enforce_capital_budget(), evaluate_strategy_file_governor(), Path, PathLike (+29 more)

### Community 74 - "split_validation_fitness_selection"
Cohesion: 0.18
Nodes (13): Store full train_new.csv row count after loader prep (split time)., set_purged_wf_reference_rows(), _chronological_half_split(), _holdout_embargo_split(), _purged_walk_forward_split(), DataFrame, Per-symbol chronological first or second half of *df*. ``purge_rows`` is…, Split validation into purged fitness and selection halves per symbol. The gap… (+5 more)

### Community 75 - "TestWriteStrategyEvaluationTable"
Cohesion: 0.21
Nodes (9): _make_eval_rule_set(), _make_metrics_by_split(), _make_trade_logs_by_split(), Create a minimal rule set for evaluation table tests., Create a metrics_by_split dict with all three splits populated., Create a trade_logs_by_split dict with all three splits populated., Sharpe ratio should equal mean(r)/std(r, ddof=1) for a known log., win_rate, mdd_pct etc. should come from metrics_by_split. (+1 more)

### Community 76 - "test_gpu_engine.py"
Cohesion: 0.13
Nodes (13): _build_data_matrix(), _discretize_series(), DataFrame, Series, Discretize a feature series using evaluator_v5's fixed fuzzy bins. Exported…, Build an (N, K) integer matrix of discretized feature values., disable_skip_optimization(), DataFrame (+5 more)

### Community 77 - "rolling_cv.py"
Cohesion: 0.33
Nodes (12): _bar_index_col(), _build_fold_from_ranges(), build_purged_walk_forward_folds(), _purge_train(), DataFrame, Purged expanding walk-forward folds for train_new.csv. Per-symbol chronological…, Drop train rows whose label horizon could overlap the valid block., Build purged expanding walk-forward folds on ``train_new.csv``. Returns… (+4 more)

### Community 78 - "test_rolling_cv.py"
Cohesion: 0.22
Nodes (10): build_forbidden_ranges(), mask_df_to_safe_region(), Build per-symbol forbidden ``(start_bar, end_bar)`` ranges from folds. Each…, Drop rows whose per-symbol bar index falls inside any forbidden range., _make_multi_symbol_df(), _make_symbol_df(), purged_cv_config(), DataFrame (+2 more)

### Community 79 - "test_feature_selector_properties.py"
Cohesion: 0.15
Nodes (23): dataset_with_high_dispersion_feature(), dataset_with_label_and_meta_columns(), dataset_with_sufficient_dispersion_feature(), _make_label_block(), composite, DataFrame, DrawFn, given (+15 more)

### Community 80 - "_NumpyJSONEncoder"
Cohesion: 0.17
Nodes (10): __main__.py — Entry point for `python -m gpu_fuzzy_trader.run_pipeline` Allows…, main(), _NumpyJSONEncoder, _print_run_summary(), Any, Run the full pipeline from the command line., Persist the diagnostic even when preflight blocks the pipeline., JSON encoder that converts NumPy/JAX scalar and array types to native Python… (+2 more)

### Community 81 - "test_phase5_oos.py"
Cohesion: 0.22
Nodes (9): _make_df(), _make_rule_set(), Unit tests for gpu_fuzzy_trader.phases.phase5_oos.OOS_Evaluator Tests cover: -…, When no trades are executed, account_ruined must be False., When no trades are executed, total_return_pct must be 0.0., Create a minimal valid rule set dict., Returned and saved OOS metrics come from the locked strategy., Create a minimal DataFrame with all required columns. (+1 more)

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

### Community 86 - "filter_migrants_for_cluster"
Cohesion: 0.11
Nodes (19): filter_migrants_for_cluster(), _migrant_to_metrics(), Backtest one migrant chromosome on receiver cluster engines., Accept only migrants that pass deployability on the receiver cluster slice., _make_migrant_dict(), _make_mock_receiver(), Unit tests for migration safety — migrant gate and seed fraction. Acceptance…, Migrant with val_return 2.5% and >=15 val trades should be accepted. (+11 more)

### Community 87 - "parity_scenario_strategy"
Cohesion: 0.50
Nodes (4): parity_scenario_strategy(), composite, DrawFn, Generate a random dataset and trade parameters for GPU-CPU parity testing.…

### Community 88 - "test_crash_fix_and_run_logging.py"
Cohesion: 0.06
Nodes (32): _capture_logs(), _make_feature_infos_crash(), _make_train_df_crash(), DataFrame, MonkeyPatch, Unit tests for crash-fix-and-run-logging spec. Covers: - Task 1.1: Smoke tests…, Requirement 3.1 — save_archive is called before _release_resources in run()., Mock save_archive and _release_resources, run Rule_Pool_Generator.run() with a… (+24 more)

### Community 89 - "test_property_27_test_data_preparation_consistency"
Cohesion: 0.20
Nodes (11): _make_timestamps(), composite, DataFrame, DrawFn, given, Timestamp, **Property 27: Test Data Preparation Consistency** **Validates: Requirements…, Return n strictly increasing timestamp strings starting from *base*. (+3 more)

### Community 90 - "run_pipeline.py"
Cohesion: 0.08
Nodes (34): downcast_numeric_df(), prune_train_columns(), DataFrame, df_slim.py — Reduce DataFrame memory for backtest engines. Keeps only columns…, Downcast label and feature columns to float32; symbol to category (inplace)., Return a copy with only meta, label, and requested feature columns. Parameters…, Drop unused feature columns from the full training split after Phase 1., slim_backtest_df() (+26 more)

### Community 91 - "TestEndToEndRotation"
Cohesion: 0.18
Nodes (7): fixture, Integration-style tests with a mocked Rule_Pool_Generator., Patch config for rotation and create a generator with minimal setup., When rotation is enabled, _cached_scoped_train_df is stored., After resample_train_for_epoch, the cached slim train changes., Same epoch_idx produces identical cached slim train., TestEndToEndRotation

### Community 92 - "phase2_support.py"
Cohesion: 0.11
Nodes (15): compute_robust_score(), EvolutionFloors, _passes_pool_admission_impl(), _pool_admission_floors(), phase2_support.py — Trade support penalties for Phase 2., Support penalty. Returns ------- penalty : float is_specialist : bool (always…, Return (train_trade_floor, train_ret_min, val_ret_min, pf_floor,…, Resolved evolution-time floors (pool admission gates remain strict). (+7 more)

### Community 93 - ".encode_condition"
Cohesion: 0.10
Nodes (13): Encoder, Stateless encoder that maps gene values to fuzzy condition strings., See module-level :func:`encode_condition`., parametrize, Unit tests for gpu_fuzzy_trader/features/encoder.py Tests cover: -…, TestEncodeConditionBinary, TestEncodeConditionErrors, TestEncodeConditionPositive (+5 more)

### Community 94 - "test_output_writer_properties.py"
Cohesion: 0.13
Nodes (26): parse_symbol_condition(), symbol_conditions.py — Symbol filter parsing (evaluator_v5 parity). Feature…, Parse optional symbol filters. Supported formats: "symbol is 1" "symbol IS 1"…, all_zero_rule_st(), _is_valid_exported_condition(), oversized_rule_set_st(), composite, DrawFn (+18 more)

### Community 95 - "Data_Loader"
Cohesion: 0.08
Nodes (19): Data_Loader, Stateless data loader for the GPU-Fuzzy Trading Pipeline., _Phase5JSONEncoder, Keep numeric report values numeric instead of stringifying NumPy scalars., Write a synthetic test CSV with all required columns (including feat_0..4) to a…, TestPhase5CachedSplitFreshness, _write_synthetic_test_csv(), _enriched() (+11 more)

### Community 96 - "test_phase2_gpu_throughput.py"
Cohesion: 0.13
Nodes (19): _jax_compute_rule_signals_batch(), _jax_simulate_equity_batch(), Batch rule matching for B chromosomes simultaneously. Returns (B, N) boolean…, Compatibility wrapper for a regular full-row scan., CPU and GPU backtest engine sub-package., get_gpu_backtest_engine_class(), jax_gpu_backtest_available(), Any (+11 more)

### Community 97 - ".decode_chromosome"
Cohesion: 0.26
Nodes (4): See module-level :func:`decode_chromosome`., All 10 signed values decode correctly via decode_chromosome., Chromosome with numpy int32/int64 values should work., TestDecodeChromosome

### Community 98 - ".load_strategies"
Cohesion: 0.22
Nodes (5): Load long.json and short.json via Output_Writer.load_and_validate(). Parameters…, Return only non-empty RB strategies accepted for deployment., Write a valid rule set JSON to path., TestLoadStrategies, _write_rule_set()

### Community 99 - "gate_positive_good"
Cohesion: 0.31
Nodes (13): _positive_good_reject_reasons(), Human-readable reasons why ``_is_positive_good`` failed (diagnostics)., gate_positive_good(), PositiveGoodThresholds, Minimum evidence required on train and validation splits., Return whether both splits satisfy one explicit evidence contract.…, _metrics(), Focused tests for the shared positive-good admission contract. (+5 more)

### Community 100 - "phase2_island_scheduler.py"
Cohesion: 0.04
Nodes (57): filter_df_to_symbols(), phase2_history_path(), phase2_pool_path(), DataFrame, Return rows for the given symbols; raises if column missing or no rows., Resolve Phase 2 pool path., Resolve Phase 2 history path., context_floor_failures() (+49 more)

### Community 101 - "compute_labels"
Cohesion: 0.20
Nodes (10): compute_labels(), DataFrame, Forward-window label computation for OHLCV bars. Horizon is ``TAIL_DROP_ROWS``…, Compute the 5 label columns per symbol. - label_open_next: open[t+1] -…, DataFrame, ndarray, Unit tests for forward-window label semantics…, Regression: old backward rolling must not match corrected forward labels. (+2 more)

### Community 102 - "_remove_low_dispersion"
Cohesion: 0.17
Nodes (10): Remove features where more than `threshold` fraction of values are identical.…, _remove_low_dispersion(), DataFrame, Exactly 95% identical → NOT > 0.95 → keep., 96% identical → > 0.95 → remove., Two features with corr > 0.95 → keep higher-scored one., Two uncorrelated features → both kept., Features in different modes are not compared for redundancy. (+2 more)

### Community 103 - "prop_settings"
Cohesion: 0.07
Nodes (45): HealthCheck, prop_settings(), Shared Hypothesis settings for property tests. Set PYTEST_LOW_MEMORY=1 to scale…, Hypothesis settings with optional low-memory example scaling., _make_timestamps(), composite, DataFrame, DrawFn (+37 more)

### Community 104 - "test_crash_fix_properties.py"
Cohesion: 0.10
Nodes (26): _append_xla_flag(), configure_jax_env(), _cuda_package_root(), Path, JAX/XLA runtime configuration — call before the first ``import jax``., Find a pip-installed CUDA toolkit root, if one is available., Append an XLA flag once, preserving explicit user configuration., Configure JAX/XLA runtime for predictable desktop-friendly GPU usage. -… (+18 more)

### Community 105 - "TestPlotDistributionAndEquity"
Cohesion: 0.16
Nodes (6): _make_dist_logs_by_split(), _make_dist_trade_log(), Create a trade log with all columns needed for distribution/equity plots., Create trade_logs_by_split dict with all three splits populated., A trade log with a single trade should not raise., TestPlotDistributionAndEquity

### Community 106 - "baselines.py"
Cohesion: 0.24
Nodes (17): compute_entry_time_priority(), Map each row to a timestamp priority code (evaluator_v5 parity)., _compact(), _drop_one_condition_variants(), _equal_weight(), _evaluate(), evaluate_baselines(), _feature_shuffle() (+9 more)

### Community 107 - "TestScaleTradeFloor"
Cohesion: 0.20
Nodes (5): holdout_mode(), purged_mode(), fixture, Tests for purged-WF trade-floor scaling helpers., TestScaleTradeFloor

### Community 108 - "_jax_compute_trade_outcomes"
Cohesion: 0.24
Nodes (5): _jax_compute_trade_outcomes(), JAX-jitted vectorized trade outcome computation., Vectorized trade outcome computation for all rows. Mirrors…, Multiple rows processed simultaneously., TestComputeTradeOutcomes

### Community 109 - "_should_post_restart_early_stop_phase2"
Cohesion: 0.29
Nodes (9): Break the epoch when a plateau restart yields no improvement. Independent of…, _should_post_restart_early_stop_phase2(), Unit tests for post-restart no-improvement early stop (Phase 2 runtime)., test_global_disabled_no_stop(), test_global_uses_global_knobs(), test_island_disabled_no_stop(), test_island_streak_at_patience_stops(), test_island_streak_below_patience_no_stop() (+1 more)

### Community 110 - "test_evaluator_health.py"
Cohesion: 0.14
Nodes (8): Unit tests for ``evaluator_health_penalty`` and ``execution_ok`` (Task 4).…, Verify that ``gate_positive_good`` calls ``execution_ok`` when flagged., When require_execution_health=True and execution_ok fails, gate returns False., When require_execution_health=True and execution_ok passes, gate still returns…, When require_execution_health=False, gate ignores raw_signal_count., Both public functions are importable from the module., TestExecutionHealthInGate, TestModuleImportable

### Community 111 - "ConfigurationError"
Cohesion: 0.20
Nodes (10): ConfigurationError, decode_chromosome(), encode_condition(), get_dont_care(), Exception, ndarray, Return '[feature_name] IS Fuzzy Value Name' for a valid gene. Parameters…, Convert a chromosome array to a list of condition strings. Genes equal to the… (+2 more)

### Community 112 - "execution_ok"
Cohesion: 0.12
Nodes (16): execution_ok(), Return ``True`` iff the evaluator would execute this rule set reasonably. A…, _finite_number(), _metric_int(), positive_good_reject_reasons(), Shared metric gates used by Phase 2 admission and RB Governor. The gate is…, Return stable, machine-readable reasons for a gate rejection., Tests for ``execution_ok``. (+8 more)

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

### Community 119 - "CPUBacktestEngine"
Cohesion: 0.11
Nodes (23): CPUBacktestEngine, CPU backtest engine that exactly mirrors evaluator_v5.ipynb's…, _evaluate_ruleset(), _optimize_risk(), _passes_tail_selection_gate(), Validate a trial ruleset on the reserved chronological validation tail. The…, _MockEngine, DataFrame (+15 more)

### Community 120 - "_compute_stability"
Cohesion: 0.17
Nodes (9): _compute_stability(), Compute cross-symbol stability score. stability = 1 - (std / mean) If mean is 0…, All symbols have same score → std=0 → stability=1., Mean is 0 → stability=0., Single symbol with positive score → stability=1., Single symbol with zero score → stability=0., Very high variance → stability could be negative → clipped to 0., Stability should always be in [0, 1]. (+1 more)

### Community 121 - "test_certificate_first_selection.py"
Cohesion: 0.11
Nodes (26): _passes_symbol_concentration_gate(), _passes_symbol_contribution_certificate(), _passes_tail_holdout_gate(), _portfolio_selection_certificate(), Any, Require positive, supported validation PnL from multiple symbols. Symbol…, Build the per-direction symbol certificate policy. Specialist islands are…, Return the certificate used by compose, risk, and profit selection. (+18 more)

### Community 122 - "MonthlyWindowSummary"
Cohesion: 0.06
Nodes (42): _monthly_selection_certificate(), _profit_amp_monthly_summary(), Require a composed team to be mostly non-loss across calendar windows., Evaluate a ruleset on validation-only chronological windows for the amplifier…, build_monthly_windows(), _datetime_series(), evaluate_rule_set_monthly(), monthly_penalty() (+34 more)

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
Cohesion: 0.18
Nodes (8): _mutual_info_discrete_mask(), Return the set of feature names that PASS both stationarity checks. A feature…, Per-column ``discrete_features`` flags for ``mutual_info_classif``. Only…, _stationarity_filter(), Unit tests for gpu_fuzzy_trader.features.selector.Feature_Selector Tests cover:…, TestMutualInfoDiscreteMask, TestReduceOverlap, TestStationarityFilter

### Community 127 - "OOS_Evaluator"
Cohesion: 0.09
Nodes (16): OOS_Evaluator, DataFrame, Out-of-sample evaluator for Phase 5. Loads the final long/short strategies,…, Run out-of-sample evaluation. Parameters ---------- allowed_directions :…, Prepare test data using Data_Loader.load_dataset(). Applies the same…, Load prepared train, validation, and test datasets., Load and validate a strictly newer, untouched forward period., Load selected features for a direction when available. (+8 more)

### Community 128 - "stratification_scenario_strategy"
Cohesion: 0.24
Nodes (15): dataset_with_features_strategy(), metrics_strategy(), composite, DataFrame, DrawFn, Generate a trade log DataFrame with 2–50 rows. Parameters ---------- n_rules:…, Generate a metrics dict with reasonable float values. Keys: win_rate,…, Generate a dataset DataFrame with 10–100 rows. Parameters ----------… (+7 more)

### Community 129 - "TestSpearmanSignConsistency"
Cohesion: 0.18
Nodes (9): ndarray, Tests for _check_spearman_sign_consistency, including the val_df check., Build a minimal DataFrame without 'symbol' to avoid symbol-based folding., AC1: Train all positive, val negative → feature blacklisted., AC2: Train all positive, val positive → feature still kept., AC3: Train all positive, tiny |val_rho| < min_abs_corr → feature still kept., AC4: val_df=None → pre-task-8 behavior (mixed train signs still blacklisted)., AC5: val_df without label_close_288 column → val check skipped (no crash). (+1 more)

### Community 130 - "_make_walk_forward_fold_engines"
Cohesion: 0.18
Nodes (12): _make_walk_forward_fold_engines(), Split val_selection into n_splits chronological folds + optional tail holdout.…, _make_synthetic_df(), DataFrame, Verify _make_walk_forward_fold_engines splits data correctly., 2 splits + tail holdout → 2 fold engines + 1 tail engine., tail_holdout_frac=0 → tail engine is None., Each symbol's data is divided into contiguous chunks across folds. (+4 more)

### Community 131 - "log_memory_rss"
Cohesion: 0.18
Nodes (11): log_memory_rss(), Optional process memory logging (set LOG_MEMORY=1)., Log current process RSS in MiB when LOG_MEMORY=1., Force GC and clear JAX compilation caches between Phase 2 directions. Releases:…, release_phase2_resources(), Release GPU engines and slim in-memory data between island epochs., Release GPU/host resources between heavy phases (best-effort)., Tests for optional memory logging helpers. (+3 more)

### Community 132 - "DataFrame"
Cohesion: 0.33
Nodes (7): _downsample_chronological(), _monthly_admission_source_df(), DataFrame, Prefer unsampled monthly val; fall back to sampled slim val., Preserve per-symbol time order required by exposure/release simulation., Pick a contiguous chronological slice of *n_rows* from *df*. Critical for…, _sort_chronological()

### Community 133 - "test_phase2_init.py"
Cohesion: 0.15
Nodes (18): assign_strata_to_indices(), build_feature_sampling_probs(), pick_active_count(), _pick_active_index(), _pick_inactive_index(), ndarray, phase2_init.py — Sparsity-guided stratified population initialization for Phase…, Assign elite / explorer labels to non-seeded population rows. (+10 more)

### Community 134 - "_apply_colab_gpu_defaults"
Cohesion: 0.33
Nodes (6): _apply_colab_gpu_defaults(), is_colab_runtime(), True when running on Google Colab (/content runtime)., Colab T4 optimization for Phase 2 runs., MonkeyPatch, test_colab_defaults_apply_when_content_exists()

### Community 135 - "TestPlateauDiversityRestart"
Cohesion: 0.25
Nodes (5): Reinitialised slots have objectives=np.inf and metrics_cache={}., Even with large Pareto front, at most 10 elite are preserved., Direct unit tests for the _plateau_diversity_restart helper., Pareto elite chromosomes survive the restart., TestPlateauDiversityRestart

### Community 137 - "compute_phase2_objectives_from_metrics"
Cohesion: 0.04
Nodes (42): compute_phase2_objectives_from_metrics(), _phenotype_bucket_key(), Discretise objective-relevant metrics for behavioral diversity., Penalty for weak cross-symbol robustness on one split., Minimum validation trades before joint Sortino is trusted., Build Phase 2 minimisation objectives from precomputed train/val metrics.…, tanh-saturated Sortino so the best-front member moves with progress. The…, _saturating_sortino() (+34 more)

### Community 138 - "test_nsga3_selection.py"
Cohesion: 0.40
Nodes (3): skipif, Unit tests for NSGA-III environmental selection., TestNsga3Selection

### Community 139 - ".load_pool"
Cohesion: 0.26
Nodes (3): Load existing pool if valid, return None if missing., Return loaded pool if valid, None if need to run., TestLoadPool

### Community 142 - "TestIslandSchedulerGlobalMode"
Cohesion: 0.17
Nodes (7): Unit tests for island scheduler global-mode safety. Acceptance criteria…, AC-T1.4: Global mode must never reach migration code., When PHASE2_ISLAND_MODE='global', _run_cluster_islands is not called., Verify the migration guard would not be reached in global mode., The top-level dispatch should only call run_cluster_phase2 in cluster mode., In global mode, the lazy import of extract_deployable_migrants never fires., TestIslandSchedulerGlobalMode

### Community 143 - ".get_dont_care"
Cohesion: 0.17
Nodes (7): See module-level :func:`get_dont_care`., **Property 9: Don't-Care Sentinel Correctness — encode_condition raises**…, **Property 9: Don't-Care Sentinel Correctness — all-dont_care → empty output**…, test_property_9b_encode_condition_raises_for_dont_care(), test_property_9f_all_dont_care_chromosome_returns_empty(), Static methods should be callable on the class itself., TestGetDontCare

### Community 144 - "TestRefreshObjectivesOnResumeGate"
Cohesion: 0.23
Nodes (7): Task-10: gate cache refresh on PHASE2_PER_EPOCH_WINDOW_ROTATION. Verifies the…, Evaluate the gate expression as it appears in the source., PHASE2_PER_EPOCH_WINDOW_ROTATION=True (default): refresh = not first_epoch →…, PHASE2_PER_EPOCH_WINDOW_ROTATION=True (default): refresh = not first_epoch →…, PHASE2_PER_EPOCH_WINDOW_ROTATION=False (legacy): refresh is False even on…, PHASE2_PER_EPOCH_WINDOW_ROTATION=False (legacy): refresh is False on first…, TestRefreshObjectivesOnResumeGate

### Community 145 - "conftest.py"
Cohesion: 0.22
Nodes (10): Config, FixtureRequest, Item, _close_matplotlib_figures(), _low_memory_cleanup(), fixture, pytest_collection_modifyitems(), pytest_configure() (+2 more)

### Community 147 - ".skip_if_valid"
Cohesion: 0.33
Nodes (3): Check if output files exist and are valid. Returns ------- dict[str,…, fixture, TestSkipIfValid

### Community 148 - "_validate_schema"
Cohesion: 0.17
Nodes (5): Validate the structure of a loaded feature selection JSON. Raises ValueError if…, Load and validate a feature selection JSON file. Parameters ---------- path :…, _validate_schema(), TestLoadAndValidate, TestValidateSchema

### Community 150 - "test_phase2_use_gpu_flag.py"
Cohesion: 0.25
Nodes (7): _minimal_backtest_df(), DataFrame, MonkeyPatch, Phase 2 respects PHASE2_USE_GPU=False without loading JAX GPU engine., The memory-safe CPU route must happen before JAX allocates arrays., A selected CPU backend must not initialize JAX just to warm up., TestPhase2UseGpuFlag

### Community 152 - "BFS and DFS Graph Traversal"
Cohesion: 0.20
Nodes (10): CLAUDE.md Graphify Integration, Constrained Query Expansion, BFS and DFS Graph Traversal, Graph Work Memory, Existing Graph Fast Path, CLAUDE.md Graphify Integration, Constrained Query Expansion, BFS and DFS Graph Traversal (+2 more)

### Community 153 - "TestGPUCPUNumericalParity"
Cohesion: 0.10
Nodes (15): ndarray, simulate_rule_set (delegated to CPU) is exactly identical to CPU engine., GPU backtest engine must handle padded chunking correctly., CPU backtest engine must support simulate_rule_batch with correct metrics., Compare GPU and CPU engine for 10 random chromosomes (Task 2 acceptance…, Build a multi-symbol DataFrame with mixed TP/SL/time-exit outcomes., Generate random chromosomes for binary + signed features., GPU engine results must match CPU engine within specified tolerances for 10… (+7 more)

### Community 154 - "test_phase2_rule_pool.py"
Cohesion: 0.06
Nodes (28): _crowding_distance(), _hamming_distance(), _pareto_sortino_stats(), Compute crowding distance for solutions in *front*. Parameters ----------…, Hamming distance between two chromosomes (active pairs when sparse)., Aggregate raw Sortino and return health over the current Pareto front., _chromosome_with_min_active(), _isolate_phase2_archive_paths() (+20 more)

### Community 157 - "_build_pool_from_archive"
Cohesion: 0.08
Nodes (32): _archive_direction(), attach_cv_fold_returns_batch(), _build_cpu_archive_engine(), _build_pool_from_archive(), _chromosome_batch(), _chromosome_for_pool_export(), CvFoldValEvaluator, _entry_validation_per_symbol_metrics() (+24 more)

### Community 158 - "TestGlobalMetricsCacheClearing"
Cohesion: 0.27
Nodes (6): When seeded_keys is empty, nothing is removed., When all keys are seeded, cache becomes empty., Verify that only seeded keys are removed from the global cache., Create a deterministic chromosome key., Cache entries matching seeded_keys are removed; non-matching survive., TestGlobalMetricsCacheClearing

### Community 159 - "TestMigrationSeedFraction"
Cohesion: 0.20
Nodes (6): Verify that migrant injection uses PHASE2_MIGRATION_SEED_FRACTION, not ARCHIVE., PHASE2_MIGRATION_SEED_FRACTION defaults to 0.10., Ensure the migration fraction is not the same as archive fraction., Simulate the run_epoch migrant path and assert local_cap uses migration…, Simulate the new run_epoch logic: migrant entries are capped by migration…, TestMigrationSeedFraction

### Community 161 - "_should_plateau_early_stop_phase2"
Cohesion: 0.05
Nodes (45): True for cluster/orphan scoped runs rather than the global path., scoped_island_profile(), Resolve the plateau patience value based on profile and stage. Cluster/orphan…, Earliest gen for plateau stop. Island epochs (~20 gens) and scaled Stage A…, _resolve_plateau_min_generation(), _resolve_plateau_patience(), _should_early_stop_phase2(), _should_plateau_early_stop_phase2() (+37 more)

### Community 164 - "test_phase2_support.py"
Cohesion: 0.06
Nodes (38): IslandHyperparams, Resolved Phase 2 knobs for cluster or orphan slices., _pool_entry_passes_admission(), Check stored train/val metrics on a pool JSON entry., deployability_rank_score(), _evolution_feasibility_floors(), _feasibility_gate_failures(), feasibility_violation_score() (+30 more)

### Community 172 - "TestHallOfFameTrim"
Cohesion: 0.25
Nodes (5): Verify that hall_of_fame is capped to PHASE2_HOF_EPOCH_CARRYOVER=10., HoF with >10 entries is trimmed to exactly 10 oldest entries., HoF with <=10 entries is left unchanged., Empty HoF does not crash during trim logic., TestHallOfFameTrim

### Community 176 - "test_gpu_engine_import_does_not_crash_on_jax_failure"
Cohesion: 0.33
Nodes (6): parametrize, Tests for the lazy-JAX failure path in gpu_engine.py. These tests verify that…, Verify importing gpu_engine.py handles various JAX failure modes gracefully., Build a subprocess script that simulates JAX import failure., _subprocess_script(), test_gpu_engine_import_does_not_crash_on_jax_failure()

### Community 177 - "test_jax_compat.py"
Cohesion: 0.29
Nodes (5): parametrize, Tests for JAX / GPU engine availability detection., Package init must not crash when gpu_engine import fails., test_cpu_engine_import_without_jax(), test_get_gpu_backtest_engine_class_returns_none_on_import_error()

### Community 181 - "TestEvalCvFoldReturns"
Cohesion: 0.33
Nodes (4): Verify the helper handles None / empty fold_engines safely., fold_engines=None returns None without crashing., fold_engines=[] returns None without crashing., TestEvalCvFoldReturns

### Community 189 - "opencode.json"
Cohesion: 0.50
Nodes (3): plugin, $schema, .opencode/plugins/graphify.js

## Knowledge Gaps
- **35 isolated node(s):** `$schema`, `.opencode/plugins/graphify.js`, `gpu-fuzzy-trader`, `URL Ingestion`, `Folder Watcher` (+30 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **11 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `CPUBacktestEngine` connect `CPUBacktestEngine` to `rb_governor.py`, `_make_walk_forward_fold_engines`, `_make_engine`, `test_rb_governor_cv_folds.py`, `Rule_Pool_Generator`, `TestContextEntryPaths`, `phase2_rule_pool.py`, `_make_df`, `_apply_monthly_admission_gate`, `test_cpu_engine_properties.py`, `test_phase2_use_gpu_flag.py`, `TestGPUCPUNumericalParity`, `CandidateRecord`, `_build_pool_from_archive`, `ValidationError`, `gpu_engine.py`, `_build_entries_from_rule_set`, `test_jax_compat.py`, `_compute_rule_signal_mask`, `nested_walk_forward.py`, `TestEvalCvFoldReturns`, `GPUBacktestEngine`, `cpu_engine.py`, `ValueError`, `barrier.py`, `_apply_dynamic_rule`, `test_cpu_engine.py`, `run_rb_governor_pipeline`, `test_gpu_engine.py`, `filter_migrants_for_cluster`, `run_pipeline.py`, `Data_Loader`, `test_phase2_gpu_throughput.py`, `phase2_island_scheduler.py`, `prop_settings`, `baselines.py`, `_jax_compute_trade_outcomes`, `_jax_compute_rule_signals`, `test_certificate_first_selection.py`, `MonthlyWindowSummary`, `._engine`, `OOS_Evaluator`?**
  _High betweenness centrality (0.124) - this node is a cross-community bridge._
- **Why does `Rule_Pool_Generator` connect `Rule_Pool_Generator` to `.run`, `log_memory_rss`, `_get_dont_cares`, `TestRulePoolGeneratorRun`, `compute_phase2_objectives_from_metrics`, `phase2_rule_pool.py`, `.load_pool`, `Pipeline_Orchestrator`, `_apply_monthly_admission_gate`, `TestRefreshObjectivesOnResumeGate`, `_make_train_df`, `test_phase2_use_gpu_flag.py`, `Reporter`, `test_phase2_rule_pool.py`, `_derive_val_sample_seed`, `_build_pool_from_archive`, `_make_multi_sym_df`, `_gpu_runtime.py`, `_dominates`, `_mutate`, `test_phase2_rule_pool_properties.py`, `_run_cluster_islands`, `test_plateau_state_leak.py`, `passes_pool_admission_gate`, `_NumpyJSONEncoder`, `filter_migrants_for_cluster`, `test_crash_fix_and_run_logging.py`, `run_pipeline.py`, `TestEndToEndRotation`, `phase2_island_scheduler.py`, `_validate_pool_schema`, `TestValLeakGate`, `CPUBacktestEngine`?**
  _High betweenness centrality (0.120) - this node is a cross-community bridge._
- **Why does `Reporter` connect `Reporter` to `.run`, `TestEquityCurveDateAxis`, `prop_settings`, `Rule_Pool_Generator`, `TestPlotDistributionAndEquity`, `phase2_rule_pool.py`, `TestWriteStrategyEvaluationTable`, `._ensure_dir`, `TestPlotPerRuleBreakdown`, `DataFrame`, `test_reporter.py`, `run_pipeline.py`, `_build_pool_from_archive`, `Data_Loader`, `OOS_Evaluator`?**
  _High betweenness centrality (0.104) - this node is a cross-community bridge._
- **Are the 15 inferred relationships involving `Reporter` (e.g. with `CvFoldValEvaluator` and `Rule_Pool_Generator`) actually correct?**
  _`Reporter` has 15 INFERRED edges - model-reasoned connections that need verification._
- **Are the 60 inferred relationships involving `CPUBacktestEngine` (e.g. with `GPUBacktestEngine` and `JointPortfolioEngine`) actually correct?**
  _`CPUBacktestEngine` has 60 INFERRED edges - model-reasoned connections that need verification._
- **Are the 65 inferred relationships involving `Rule_Pool_Generator` (e.g. with `CPUBacktestEngine` and `Reporter`) actually correct?**
  _`Rule_Pool_Generator` has 65 INFERRED edges - model-reasoned connections that need verification._
- **Are the 31 inferred relationships involving `Output_Writer` (e.g. with `OOS_Evaluator` and `_Phase5JSONEncoder`) actually correct?**
  _`Output_Writer` has 31 INFERRED edges - model-reasoned connections that need verification._