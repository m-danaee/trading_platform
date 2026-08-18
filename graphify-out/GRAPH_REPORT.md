# Graph Report - trading_platform  (2026-08-18)

## Corpus Check
- 191 files · ~279,254 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 4866 nodes · 11194 edges · 197 communities (186 shown, 11 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 511 edges (avg confidence: 0.56)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `eb65424f`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- evox_runner.py
- test_evox_runner.py
- gpu_engine.py
- _make_engine
- Pipeline_Orchestrator
- TestValLeakGate
- _make_feature_infos
- _score_metrics
- _mutate
- _make_df
- Data_Splitter
- run_phase2_evolution
- splitter.py
- test_phase5_oos.py
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
- research_integrity.py
- _loader_from_rows
- TestEndToEndRotation
- ndarray
- _gpu_runtime.py
- ._ensure_dir
- mtf/__init__.py
- test_certificate_first_selection.py
- non_dominated_sort
- optuna_search.py
- Hierarchical Multi-Timeframe Rule Discovery System Specification
- _get_dont_cares
- _compute_rule_signal_mask
- validate_config
- test_crash_fix_and_run_logging.py
- test_reporter.py
- test_phase2_rule_pool.py
- phase2_rule_pool.py
- _make_rule
- GPUBacktestEngine
- ._simulate_rule_set_entries
- Phase2EvolutionState
- test_phase2_use_gpu_flag.py
- rolling_cv.py
- test_mtf_pipeline_integration.py
- test_plateau_state_leak.py
- _make_df
- dashboard.py
- DataFrame
- test_data_loader_properties.py
- ValueError
- _m
- _apply_dynamic_rule
- JointPortfolioEngine
- .encode_condition
- execution_ok
- TestWriteStrategyEvaluationTable
- test_gpu_engine.py
- TestRunLogHandlerLifecycle
- test_output_writer_properties.py
- test_feature_selector_properties.py
- .save_archive
- _symbol_specialized_variants
- TestPlotPerRuleBreakdown
- effective_phase2_val_return_floor_pct
- Feature_Detector
- _build_target
- OOS_Evaluator
- test_gpu_engine_properties.py
- TestGPUCPUNumericalParity
- build_complete_higher_bars
- discovery.py
- set_purged_wf_reference_rows
- test_phase2_window_rotation.py
- Encoder
- compute_phase2_objectives_from_metrics
- test_crash_fix_properties.py
- test_mtf_ensembler.py
- _dominates
- .load_strategies
- gate_positive_good
- _pool_admission_floors
- Data_Loader
- _remove_low_dispersion
- TestEquityCurvePlots
- prop_settings
- TestPlotDistributionAndEquity
- baselines.py
- TestWriteSpearmanCorrelationReport
- _jax_compute_trade_outcomes
- _should_post_restart_early_stop_phase2
- hypothesis_config.py
- nested_walk_forward.py
- _validate_pool_schema
- .run
- TestHammingThresholdAutoScale
- .decode_chromosome
- ValidationError
- _jax_compute_rule_signals
- TestParetoCollapseWarningGate
- test_rb_governor_tail_holdout.py
- _compute_stability
- _feasibility_gate_failures
- MonthlyWindowSummary
- ._engine
- test_rb_concentration_tail_fail_closed.py
- constrained_non_dominated_sort
- _make_walk_forward_fold_engines
- _derive_epoch_seed
- stratification_scenario_strategy
- TestSpearmanSignConsistency
- compute_labels
- log_memory_rss
- _init_population
- _apply_colab_gpu_defaults
- test_data_loader.py
- _build_rule_signal_mask
- .get_dont_care
- resolve_phase2_stage_params
- test_phase2_gpu_throughput.py
- CPUBacktestEngine
- ._phase1_keep_feature_names
- test_feature_selector.py
- phase2_support.py
- _NumpyJSONEncoder
- TestRefreshObjectivesOnResumeGate
- conftest.py
- _derive_val_sample_seed
- .skip_if_valid
- _validate_schema
- test_directional_evaluator.py
- Global Constraints
- test_phase2_support.py
- BFS and DFS Graph Traversal
- .simulate_rule_set
- load_cached_split_if_fresh
- TestSavePerSymbolCsv
- .run
- passes_pool_admission_gate
- TestGlobalMetricsCacheClearing
- test_evaluator_health.py
- _nsga3_environmental_selection
- _legacy_writer_contract
- test_property_27_test_data_preparation_consistency
- TestPoolAdmissionOverfitRatioGate
- resolve_evolution_floors
- evaluator_health.py
- run_rb_governor_pipeline
- effective_min_profitable_symbols
- composite
- TestSparsePositiveMode
- TestZeroRatioBoundary
- run_pipeline.py
- TestHallOfFameTrim
- cpu_engine.py
- .load_and_validate
- test_phase2_offspring_batch.py
- test_gpu_engine_import_does_not_crash_on_jax_failure
- test_jax_compat.py
- TestExecutionHealthInGate
- trade_support_penalty
- TestChronologicalOrdering
- TestNaNHandling
- scoring/__init__.py
- TestSparseSignedMode
- TestEvalCvFoldReturns
- DataFrame
- TestTrainOnlyFitnessValGating
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
2. `CPUBacktestEngine` - 148 edges
3. `Rule_Pool_Generator` - 126 edges
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

## Communities (197 total, 11 thin omitted)

### Community 0 - "evox_runner.py"
Cohesion: 0.05
Nodes (93): IslandHyperparams, Optional Phase 2 floor overrides (tests and diagnostics)., _behavior_descriptor(), _binary_tournament_pick(), _build_diversity_reference(), clear_global_metrics_cache(), _constraint_violations(), _count_deployable_preview() (+85 more)

### Community 1 - "test_evox_runner.py"
Cohesion: 0.05
Nodes (39): _assign_eval_result(), _evaluate_population_indices(), _inherit_val_metrics_from_global_cache(), Compute penalties/objectives from metrics; write objectives[i] and…, Evaluate unevaluated individuals, preferring batch simulate_rule_batch., Copy val_* from global cache for identical chromosomes when val is skipped.…, Return survivors that do not already carry a validation snapshot. Validation…, Return updated (best_max_return, gens_without_improvement). (+31 more)

### Community 2 - "gpu_engine.py"
Cohesion: 0.09
Nodes (25): attach_barrier_outcomes(), barrier_column_names(), configured_barrier_pairs(), _first_touch_for_symbol(), _first_touch_for_symbol_numba(), _first_touch_for_symbol_python(), _number_token(), DataFrame (+17 more)

### Community 3 - "_make_engine"
Cohesion: 0.05
Nodes (35): _build_entries_from_rule_set(), Priority-based rule assignment: first matching rule wins per row. Mirrors…, _make_df(), _make_engine(), DataFrame, Simulate catastrophic losses to trigger account ruin. With min_288=0…, Per-symbol metrics should reflect actual trade distribution., Winning trades should produce positive net_pnl per symbol. (+27 more)

### Community 4 - "Pipeline_Orchestrator"
Cohesion: 0.05
Nodes (61): FileHandler, Persist discovered MTF rules into a structured, hashed JSON archive atomically.…, save_mtf_rule_archive(), _dataframe_schema_sha256(), _dataframe_sha256(), _git_commit_id(), _log_phase_entry(), _log_pipeline_config() (+53 more)

### Community 5 - "TestValLeakGate"
Cohesion: 0.20
Nodes (10): C6: Val-derived penalties must be gated behind JOINT_TRAIN_VAL or…, Return standard monkeypatching for clean baseline metrics., Apply base settings with optional overrides., Metrics that trigger no train-side penalties., Val metrics that WOULD trigger penalties if the gate were open., When both JOINT_TRAIN_VAL and VAL_IN_FITNESS_PENALTY are False, val-derived…, Bad val must not set feasibility_violation when gate is closed., When VAL_IN_FITNESS_PENALTY=True, val-derived penalties DO enter objectives. (+2 more)

### Community 6 - "_make_feature_infos"
Cohesion: 0.08
Nodes (17): _make_feature_infos(), Integration tests using tiny population and generation counts., In holdout mode, val engine must be built for pool admission even when…, All pool entries must have active conditions within [MIN_CONDITIONS,…, Phase 2 must use static TP=PHASE2_TP, SL=PHASE2_SL,…, Pool entries must have executed_trades >= MIN_TRADE_POOL_FLOOR., Verify that Rule_Pool_Generator's persistent self._rng advances across multiple…, After two run_epoch() calls, the internal RNG state must differ from the… (+9 more)

### Community 7 - "_score_metrics"
Cohesion: 0.07
Nodes (38): _combined_return_score(), _evaluate_ruleset(), _evaluator_health_penalty(), _optimize_risk(), Return (ok, bonus, penalty) for the desired train-valid balance shape. In…, Penalize evaluator_v5 execution problems: too many skipped signals, low…, Dominant objective: return/DD with train-valid balance, plus CV-fold…, Profit objective for lenient rule addition, but now evaluator_v5 aware. A new… (+30 more)

### Community 8 - "_mutate"
Cohesion: 0.13
Nodes (12): _mutate(), Mutate a chromosome (returns a copy). When activating a dont_care gene, feature…, C5 mutation bias: force symbol-gene to dont_care / inactive with probability…, Create feature_infos with a feature whose name contains 'symbol'., PHASE2_SYMBOL_GENE_DONT_CARE_PROB=1.0: symbol gene always forced to dont_care., PHASE2_SYMBOL_GENE_DONT_CARE_PROB=0.0: symbol gene never force-set., With probability ~0.5, about half of calls force symbol to dont_care., No symbol feature in feature_infos: bias silently does nothing (no crash). (+4 more)

### Community 9 - "_make_df"
Cohesion: 0.08
Nodes (21): _make_df(), _make_engine(), MonkeyPatch, Chromosome positions must follow feature_modes insertion order., Exact CPU re-evaluation must interpret genes like the GPU path., Chromosome that matches nothing returns 0 executed trades., Zero-signal chunks should use reject metrics without scanning., PHASE2_SKIP_ZERO_SIGNAL_SCAN must match always-scan metrics. (+13 more)

### Community 10 - "Data_Splitter"
Cohesion: 0.14
Nodes (19): Data_Splitter, Chronological train/validation splitter., Module-level wrapper around ``Data_Splitter.split_and_persist``., split_and_persist(), _make_df(), _make_timestamps(), _patch_split_paths(), DataFrame (+11 more)

### Community 11 - "run_phase2_evolution"
Cohesion: 0.06
Nodes (38): _make_offspring_population(), Shared val-cadence check for both NSGA-II fallback and NSGA-III loops. Val…, Generate pop_size offspring via binary tournament, crossover, mutation.…, Run Phase 2 NSGA-III evolution. Returns (pareto_pool, history) or state., run_phase2_evolution(), _should_plateau_early_stop_phase2(), _should_run_val_this_gen(), The fallback must not switch f3 from CV return to PF after gen 0. (+30 more)

### Community 12 - "splitter.py"
Cohesion: 0.16
Nodes (18): _chronological_half_split(), _file_sha256(), _holdout_embargo_split(), _purged_walk_forward_split(), DataFrame, data/splitter.py — Data_Splitter Per-symbol chronological split for Phase 2,…, Per-symbol chronological first or second half of *df*. ``purge_rows`` is…, Split validation into purged fitness and selection halves per symbol. The gap… (+10 more)

### Community 13 - "test_phase5_oos.py"
Cohesion: 0.11
Nodes (15): _isolate_phase5_reporter_outputs(), fixture, Unit tests for gpu_fuzzy_trader.phases.phase5_oos.OOS_Evaluator Tests cover: -…, Override module-level path dicts and return originals., Keep Reporter plots/CSVs out of the checked-in outputs directory., Write a valid selected-features JSON to path., Write a synthetic test CSV with all required columns (including feat_0..4) to a…, Give integration tests an isolated, valid enriched train/split pair. (+7 more)

### Community 14 - "test_run_pipeline.py"
Cohesion: 0.09
Nodes (41): _context_coverage_for_direction(), _context_coverage_preflight(), _context_coverage_report(), Temporarily rebind all cached output paths for one pipeline run., Return shared permission/trigger/conjunction coverage diagnostics., Return split-aware context coverage for both trading directions., Reject a mixed, stale, or altered enriched train/test input pair., Log coverage and block only directions that cannot meet their floors. (+33 more)

### Community 15 - "_apply_monthly_admission_gate"
Cohesion: 0.06
Nodes (29): _apply_monthly_admission_gate(), _evaluate_rule_on_window(), _monthly_window_metrics(), Evaluate a single pool rule on a single monthly window. Returns the full window…, Normalize a window evaluator result for the monthly gate. The float fallback…, Apply the monthly-window shadow-test gate to a pool of rules. Each rule is…, _DeterministicEvaluator, _DeterministicMetricsEvaluator (+21 more)

### Community 16 - "test_feature_detector_properties.py"
Cohesion: 0.09
Nodes (45): all_nan_series(), arbitrary_numeric_series(), binary_series(), large_value_series(), positive_series(), composite, DrawFn, given (+37 more)

### Community 17 - "_make_train_df"
Cohesion: 0.08
Nodes (23): Sample up to *total_rows* rows, distributed equally across symbols. A single…, _sample_df(), _make_train_df(), DataFrame, Critical: bars must be contiguous so the backtest engine preserves temporal…, Random start must be bounded so the slice always fits forward., divmod distribution gives exactly total_rows (no rounding loss). 701_000 % 14…, total_rows < n_symbols must NOT force 1 row per symbol. (+15 more)

### Community 18 - "_split"
Cohesion: 0.07
Nodes (18): Helper: build df, patch paths, run split, return (train, val)., Compute expected train/val per symbol under holdout+embargo split., floor(N * HOLDOUT_TRAIN_FRACTION) rows go to train., Remaining rows after embargo gap go to validation., For N=101: floor(101 * HOLDOUT_TRAIN_FRACTION) used, not round., train + validation + embargo dropped == total rows., train + val + embargo_dropped == total for each symbol., Each symbol's split point is computed from its own row count. (+10 more)

### Community 19 - "test_cpu_engine_properties.py"
Cohesion: 0.10
Nodes (33): _expected_outcome(), _make_engine(), _make_engine_custom(), _make_minimal_df(), _make_simple_df(), multi_symbol_scenario(), DataFrame, given (+25 more)

### Community 20 - "detect_feature_mode"
Cohesion: 0.09
Nodes (19): detect_feature_mode(), Module-level convenience wrapper around Feature_Detector.detect_feature_mode., Has negative values, zero_ratio ≤ 0.3 → signed., zero_ratio == 0.3 is NOT > 0.3, so mode is signed (not sparse_signed)., Values {0, 1} match both binary and ternary criteria; binary wins., Values {-1, 0, 1} match ternary; should NOT fall through to signed., Adding value 2 to {0, 1} breaks binary → falls through to positive., Adding value 2 to {-1, 0, 1} breaks ternary → falls through to… (+11 more)

### Community 21 - "selector.py"
Cohesion: 0.08
Nodes (41): get_seed(), Return a per-process random seed (stable within a run, different across runs).…, _align_feature_array(), build_phase1_shared_context(), _build_symbol_masks(), _candidate_feature_columns(), _check_spearman_sign_consistency(), _compute_chronological_stationarity_scores() (+33 more)

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
Cohesion: 0.09
Nodes (37): all_active_chromosome_strategy(), chromosome_with_dont_cares_strategy(), feature_name_strategy(), composite, DrawFn, given, ndarray, Property-based tests for gpu_fuzzy_trader.features.encoder.Encoder **Validates:… (+29 more)

### Community 26 - "write_evaluator_clean"
Cohesion: 0.08
Nodes (29): _maybe_write_evaluator_clean(), Write a stripped strategy file containing only ``direction`` and ``rules_set``.…, Write a stripped strategy file alongside the main strategy JSON. This is a…, write_evaluator_clean(), _make_rule(), minimal_strategy(), fixture, Path (+21 more)

### Community 27 - "CandidateRecord"
Cohesion: 0.12
Nodes (35): _candidate_coverage_symbols(), _candidate_positive_symbols(), CandidateRecord, _compose_ruleset(), _diversification_beam(), _diversification_shortlist(), _max_overlap(), Return supported positive validation symbols for one candidate. (+27 more)

### Community 28 - "Rule_Pool_Generator"
Cohesion: 0.11
Nodes (11): Phase 2: GPU-accelerated multi-objective evolutionary rule pool generation.…, Build train/val backtest engines., Restore slimmed training data from cache (no re-sampling needed)., Rebuild engines after ``park_engines`` dropped GPU state., Build the selected Phase 2 backend for the sampled train frame., Return whether this generator should avoid allocating a JAX engine.…, Attach optional floor overrides; safe when *owner* is a partial mock., Build an engine on *df* using the same backend selection logic. (+3 more)

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
Cohesion: 0.10
Nodes (11): Output_Writer, Serializes RuleSet dicts to JSON and loads/validates existing JSON files.…, parametrize, Verify the exact example from the spec works end-to-end., TestLoadAndValidateErrors, TestLoadAndValidateHappyPath, TestSpecExample, TestValidationErrorType (+3 more)

### Community 36 - "config.py"
Cohesion: 0.07
Nodes (33): _config_check(), ConfigError, effective_min_trade_support(), effective_monthly_min_trades(), effective_val_trade_floor_for_objectives(), filter_df_to_symbols(), _finite_config_number(), holdout_train_val_label() (+25 more)

### Community 37 - "TestEquityCurveDateAxis"
Cohesion: 0.09
Nodes (21): _make_trade_log(), DataFrame, fixture, Unit tests for date-based x-axis in Reporter.plot_equity_curve (Task 10.3).…, When Entry_Time is absent, x-axis label is 'Trade #'., When Entry_Time is all-NaN, x-axis label is 'Trade #'., axhline is called (initial-capital line) in date mode., axhline is called (initial-capital line) in Trade # mode. (+13 more)

### Community 38 - "_build_pool_from_archive"
Cohesion: 0.09
Nodes (28): _archive_direction(), attach_cv_fold_returns_batch(), _build_cpu_archive_engine(), _build_pool_from_archive(), _chromosome_batch(), CvFoldValEvaluator, _evaluate_chromosome(), _metrics_dict_from_population() (+20 more)

### Community 39 - "research_integrity.py"
Cohesion: 0.11
Nodes (31): context_contract_digest(), Return a stable hash of the static contract and fitted enrichment., Any, Stable provenance identifiers for Phase 2 feature rules. The RB Governor may…, Hash the complete economic strategy, including exit policy. ``phase2_rule_id``…, strategy_id(), _canonical_json(), count_trials() (+23 more)

### Community 40 - "_loader_from_rows"
Cohesion: 0.12
Nodes (14): _loader_from_rows(), _make_rows(), The first N-288 rows (chronologically) should be kept., A row with NaN in any label column should be dropped., NaN label rows should be dropped, not filled with 0., Only specified feature_cols should be filled., Bar index should reflect post-drop row count, not original row count., Generate n rows for a given symbol with valid timestamps. (+6 more)

### Community 41 - "TestEndToEndRotation"
Cohesion: 0.18
Nodes (7): fixture, Integration-style tests with a mocked Rule_Pool_Generator., Patch config for rotation and create a generator with minimal setup., When rotation is enabled, _cached_scoped_train_df is stored., After resample_train_for_epoch, the cached slim train changes., Same epoch_idx produces identical cached slim train., TestEndToEndRotation

### Community 42 - "ndarray"
Cohesion: 0.07
Nodes (29): _batch_metrics_from_array(), _build_event_batch(), _exposure_slot_capacity(), _fast_reject_result_rows(), _jax_compute_rule_signals_sparse_batch(), _jax_open_slot(), _jax_release_open_slots(), _jax_simulate_equity_batch() (+21 more)

### Community 43 - "_gpu_runtime.py"
Cohesion: 0.09
Nodes (34): phase2_should_enrich_symbol_metrics(), Return True when GPU batch eval should run a follow-up CPU enrichment pass.…, configure_phase2_gpu_runtime(), detect_gpu_memory_used_gb(), detect_gpu_vram_gb(), detect_system_ram_gb(), evict_cluster_signatures(), _iter_warmup_targets() (+26 more)

### Community 44 - "._ensure_dir"
Cohesion: 0.07
Nodes (19): Reporting and visualization sub-package., _bucket_series_by_mode(), DataFrame, Series, reporter.py — Reporting and visualization for the GPU-Fuzzy Trading Pipeline.…, Write feature-stratified performance metrics to CSV files (one per split). For…, Create parent directories for *path* if they do not exist., Compute maximum drawdown percentage from an equity series. Parameters… (+11 more)

### Community 45 - "mtf/__init__.py"
Cohesion: 0.10
Nodes (35): Hierarchical MTF Strategy Candidate Container. Encapsulates LWC execution…, Evaluate hierarchical soft-veto composition on input arrays., compose_bidirectional_signals(), compose_hierarchical_signals(), normalize_direction(), Any, ndarray, Series (+27 more)

### Community 46 - "test_certificate_first_selection.py"
Cohesion: 0.11
Nodes (27): _passes_symbol_concentration_gate(), _passes_symbol_contribution_certificate(), _passes_tail_holdout_gate(), _portfolio_selection_certificate(), Any, Require positive, supported validation PnL from multiple symbols. Symbol…, Build the per-direction symbol certificate policy. Specialist islands are…, Return the certificate used by compose, risk, and profit selection. (+19 more)

### Community 47 - "non_dominated_sort"
Cohesion: 0.13
Nodes (26): batch_hamming_min(), crowding_distance(), _crowding_distance_numba(), _crowding_distance_py(), _disable_numba_sort(), _dominates_numba(), _dominates_py(), non_dominated_sort() (+18 more)

### Community 48 - "optuna_search.py"
Cohesion: 0.10
Nodes (29): _active_search_space(), apply_trial_config(), collect_validation_metrics(), compute_score(), _copy_phase1_2_outputs(), main(), objective(), _pearson() (+21 more)

### Community 49 - "Hierarchical Multi-Timeframe Rule Discovery System Specification"
Cohesion: 0.06
Nodes (30): 1.1 Root Problem Addressed, 1.2 Target Architecture Principles, 1. System Overview & Core Objectives, 2.1 Resampling & Continuity Contract, 2.2 Independent Feature Computation, 2.3 Point-in-Time Causal Alignment, 2. Multi-Timeframe Causal Data Layer, 3.1 Profile Configurations (+22 more)

### Community 50 - "_get_dont_cares"
Cohesion: 0.06
Nodes (41): extract_deployable_migrants(), Return elite deployable-preview entries suitable for guarded migration., build_feature_sampling_probs(), pick_active_count(), Uniform random k in [MIN_CONDITIONS, MAX_CONDITIONS]., Softmax Phase 1 scores with optional uniform floor (length K)., _count_active_conditions(), _get_dont_cares() (+33 more)

### Community 51 - "_compute_rule_signal_mask"
Cohesion: 0.15
Nodes (13): conditions_cache_key(), get_or_build_rule_mask(), ndarray, condition_cache.py — Cached boolean masks for textual rule conditions. Avoids…, Stable hashable key for a list of condition strings., Return a boolean row mask for *conditions*, using *cache* when provided. The…, _compute_rule_signal_mask(), Build one boolean signal mask (evaluator_v5 parity). Feature conditions are… (+5 more)

### Community 52 - "validate_config"
Cohesion: 0.15
Nodes (23): context_contract(), effective_config_snapshot(), Validate all high-impact hyperparameter relationships. The function is…, Return resolved values and derived constraints for audit/reporting., Write the effective configuration snapshot and return its path., Return the full context contract for strategy/dataset identity., validate_config(), write_config_audit_report() (+15 more)

### Community 53 - "test_crash_fix_and_run_logging.py"
Cohesion: 0.13
Nodes (16): _capture_logs(), _make_feature_infos_crash(), _make_train_df_crash(), Unit tests for crash-fix-and-run-logging spec. Covers: - Task 1.1: Smoke tests…, Requirement 3.1 — save_archive is called before _release_resources in run()., Mock save_archive and _release_resources, run Rule_Pool_Generator.run() with a…, save_archive is called with self.direction as the first argument., Requirement 3.3 — If save_archive raises, the exception is caught, a WARNING is… (+8 more)

### Community 54 - "test_reporter.py"
Cohesion: 0.15
Nodes (16): _make_selected_features(), _make_stratified_dataset(), _make_stratified_datasets_by_split(), _make_stratified_logs_by_split(), _make_stratified_trade_log(), Unit tests for gpu_fuzzy_trader.reporting.reporter.Reporter Tests cover: -…, Create a dataset with fuzzy-valued feature columns., Create a trade log with Entry_Index values within dataset bounds. (+8 more)

### Community 55 - "test_phase2_rule_pool.py"
Cohesion: 0.07
Nodes (25): _crowding_distance(), _diversity_penalty_blended(), _hamming_distance(), _pareto_sortino_stats(), _phenotype_bucket_key(), Compute crowding distance for solutions in *front*. Parameters ----------…, Hamming distance between two chromosomes (active pairs when sparse)., Discretise objective-relevant metrics for behavioral diversity. (+17 more)

### Community 56 - "phase2_rule_pool.py"
Cohesion: 0.08
Nodes (59): assign_strata_to_indices(), _pick_active_index(), _pick_inactive_index(), ndarray, _random_active_class(), phase2_init.py — Sparsity-guided stratified population initialization for Phase…, Assign elite / explorer labels to non-seeded population rows., Enforce MIN_CONDITIONS <= active <= MAX_CONDITIONS on a copy. (+51 more)

### Community 57 - "_make_rule"
Cohesion: 0.11
Nodes (11): _make_rule(), _make_rule_set(), Unit tests for gpu_fuzzy_trader.output.writer — Output_Writer Tests cover: -…, Spot-check a variety of valid fuzzy value names., Write rule_set to a temp file and reload the raw JSON., TestWriteConditionValidation, TestWriteFieldCoercion, TestWriteHappyPath (+3 more)

### Community 58 - "GPUBacktestEngine"
Cohesion: 0.10
Nodes (16): GPUBacktestEngine, CPU engine for rule-set simulation used by Phase 2/RB., Rule-set sim using a mask cache (delegates to the CPU engine)., Evaluate multiple rule sets (parallel CPU; optional mask cache)., RB JAX path: cached mask entry build + parallel CPU equity. Equity results…, Same interface as CPUBacktestEngine for compatibility., JAX-accelerated backtest engine for Phase 2 rule pool generation.…, Return the JAX backend in use ('gpu', 'cpu', or 'tpu'). (+8 more)

### Community 59 - "._simulate_rule_set_entries"
Cohesion: 0.09
Nodes (22): _append_allocated_entries(), _expectancy_lcb_pct(), _expected_shortfall_pct(), precompute_release_indices(), precompute_release_indices_from_offsets(), ndarray, Series, Simulate a pre-composed causal signal mask. Hierarchical candidates do not… (+14 more)

### Community 60 - "Phase2EvolutionState"
Cohesion: 0.14
Nodes (17): Phase2EvolutionState, Evolve one island epoch and return updated resumable state., Resumable NSGA-III state for symbol-island epoch scheduling., run_phase2_evolution_epoch(), _FakeEngine, AC: resumed island epoch with reset_plateau=True clears restart counters., Global/non-island mode: reset_plateau=False preserves counters. Uses…, Task 2: Verify refresh_objectives_on_resume resets stale objectives on resumed… (+9 more)

### Community 61 - "test_phase2_use_gpu_flag.py"
Cohesion: 0.25
Nodes (7): _minimal_backtest_df(), DataFrame, MonkeyPatch, Phase 2 respects PHASE2_USE_GPU=False without loading JAX GPU engine., The memory-safe CPU route must happen before JAX allocates arrays., A selected CPU backend must not initialize JAX just to warm up., TestPhase2UseGpuFlag

### Community 62 - "rolling_cv.py"
Cohesion: 0.10
Nodes (35): aggregate_fold_metrics(), _bar_index_col(), _build_fold_from_ranges(), build_forbidden_ranges(), build_purged_walk_forward_folds(), cv_folds_only(), derive_primary_holdout(), FoldMetricsSummary (+27 more)

### Community 63 - "test_mtf_pipeline_integration.py"
Cohesion: 0.07
Nodes (31): HierarchicalStrategyCandidate, Any, ndarray, Evaluate this frozen candidate on raw OHLCV rows. No thresholds, weights,…, Convert candidate to a JSON-serializable dictionary., Construct HierarchicalStrategyCandidate from dictionary., Encapsulates a hierarchical multi-timeframe strategy with LWC/MWC/HWC rules., Compute deterministic strategy SHA-256 identifier. (+23 more)

### Community 64 - "test_plateau_state_leak.py"
Cohesion: 0.11
Nodes (17): _make_minimal_gen(), _mock_evolution_state(), Regression tests for plateau-state leak fixes (Fixes A + B). Fix A:…, reset_plateau=True is also passed on epoch 2 (regression for leak)., Even when two-stage is disabled, reset_plateau=True., AC-2: _island_generations_done increments by len(epoch_history)., When all requested gens execute, len(epoch_history) == epoch_gens., Early-stop: 3 actual gens run out of 10 requested, budget += 3. (+9 more)

### Community 65 - "_make_df"
Cohesion: 0.25
Nodes (8): _make_df(), _make_rule_set(), When no trades are executed, account_ruined must be False., When no trades are executed, total_return_pct must be 0.0., Create a minimal valid rule set dict., Create a minimal DataFrame with all required columns., Returned and saved OOS metrics come from the locked strategy., TestEvaluateStrategy

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
Cohesion: 0.19
Nodes (6): _apply_dynamic_rule(), _parse_condition(), Parse '[feature_name] IS Fuzzy Value Name' → (feature_name, value_name)., Apply one exported text condition using the original threshold logic. Exactly…, Test all threshold branches of _apply_dynamic_rule., TestApplyDynamicRule

### Community 72 - "JointPortfolioEngine"
Cohesion: 0.13
Nodes (10): Compute a non-annualized Sortino Ratio from per-trade returns., _sortino_ratio_from_returns(), JointPortfolioEngine, DataFrame, Evaluate long and short rule books in one net-position account., Trades with tiny equity should be skipped., net_pnl = gross_pnl - fee; fee = position_notional * fee_rate., TestFeeDeduction (+2 more)

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
Nodes (15): _build_data_matrix(), _discretize_series(), DataFrame, Series, Discretize a feature series using evaluator_v5's fixed fuzzy bins. Exported…, Build an (N, K) integer matrix of discretized feature values., Raise RuntimeError if JAX failed to import at module level., _require_jax() (+7 more)

### Community 77 - "TestRunLogHandlerLifecycle"
Cohesion: 0.11
Nodes (16): DataFrame, MonkeyPatch, Phase 2 reproducibility default seed., PHASE2_SEED is drawn once at import via get_seed(); set GLOBAL_SEED=42 to fix…, Requirements 1.1, 1.4, 1.5, 1.6, 1.7 — run.log FileHandler is attached, writes…, Count FileHandlers on the root logger pointing to *path*., Patch every phase method on Pipeline_Orchestrator to be a no-op., run.log must exist after run() and contain both separator lines. (+8 more)

### Community 78 - "test_output_writer_properties.py"
Cohesion: 0.16
Nodes (23): all_zero_rule_st(), _is_valid_exported_condition(), oversized_rule_set_st(), composite, DrawFn, given, Property-based tests for gpu_fuzzy_trader.output.writer.Output_Writer Property…, Append the direction's mandatory context conditions to every rule. (+15 more)

### Community 79 - "test_feature_selector_properties.py"
Cohesion: 0.15
Nodes (23): dataset_with_high_dispersion_feature(), dataset_with_label_and_meta_columns(), dataset_with_sufficient_dispersion_feature(), _make_label_block(), composite, DataFrame, DrawFn, given (+15 more)

### Community 80 - ".save_archive"
Cohesion: 0.20
Nodes (9): _archive_feature_signature(), Return the ordered feature signature used to validate archive reuse., Read JSON from *path* and return None when the file cannot be loaded., Validate the archive JSON structure and feature compatibility., Load a compatible persistent archive if it exists, otherwise return None.…, Merge the latest pool into a persistent archive and write atomically., _read_json_payload(), _validate_archive_payload() (+1 more)

### Community 81 - "_symbol_specialized_variants"
Cohesion: 0.10
Nodes (31): expectancy_lcb_pct(), Return a conservative lower bound for per-trade net expectancy. Exact CPU…, _attach_source_symbol_filters(), _ensure_symbol_filtered_rule(), _has_symbol_condition(), _i(), _is_positive_good(), _is_recency_good() (+23 more)

### Community 82 - "TestPlotPerRuleBreakdown"
Cohesion: 0.16
Nodes (11): _make_full_trade_log(), _make_rule_set(), _make_split_logs(), Create a trade log with all columns needed for sharpe computation., Create a minimal rule set with *n* rules., Create a trade log with Rule_Index, Net_PnL, Equity_After columns., Create trade_logs_by_split dict with all three splits populated., One split None, others valid — should not raise. (+3 more)

### Community 83 - "effective_phase2_val_return_floor_pct"
Cohesion: 0.40
Nodes (4): effective_phase2_val_return_floor_pct(), Direction-aware Phase 2 validation return floor for fitness penalties., Regression tests for anti-overfit / exploration-retune config bundle., test_anti_overfit_config_bundle()

### Community 84 - "Feature_Detector"
Cohesion: 0.13
Nodes (12): detect_all_modes(), Feature_Detector, DataFrame, Series, detector.py — Feature_Detector Classifies each feature column into exactly one…, Classify feature columns by their discretization type., Classify a single feature series into one of six modes. Detection order…, Classify every column in *feature_cols* and return a mapping. Parameters… (+4 more)

### Community 85 - "_build_target"
Cohesion: 0.17
Nodes (12): _build_target(), Build a direction-specific target signal. Default…, Encoding-aware win class: 2 in asymmetric mode, 1 in legacy mode., Encoding-aware loss class: 0 in either mode., Long: max >= entry*(1+TP/100), min > entry*(1-SL/100) → success., Long: both hit but max_before_min==0 → SL first → failure., Long: both hit but max_before_min==1 → TP first → success., Long: neither TP nor SL hit → failure (or neutral in asymmetric mode). (+4 more)

### Community 86 - "OOS_Evaluator"
Cohesion: 0.23
Nodes (5): OOS_Evaluator, Out-of-sample evaluator for Phase 5. Loads the final long/short strategies,…, TestOOSEvaluatorInit, TestPhase5CachedSplitFreshness, TestSaveReport

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
Nodes (27): _align_upstream_scores(), _build_layer_frame(), _candidate_key(), _directional_pareto_front(), discover_directional_layer(), _eligible_numeric_features(), _fit_fold_candidates(), _frame_hash() (+19 more)

### Community 91 - "set_purged_wf_reference_rows"
Cohesion: 0.18
Nodes (7): Store full train_new.csv row count after loader prep (split time)., set_purged_wf_reference_rows(), holdout_mode(), purged_mode(), fixture, Tests for purged-WF trade-floor scaling helpers., TestScaleTradeFloor

### Community 92 - "test_phase2_window_rotation.py"
Cohesion: 0.09
Nodes (26): _largest_safe_range(), Return (start, end) of the largest contiguous bar range not in *forbidden*. The…, Cap *total_rows* so the per-symbol request fits within the safe range. When…, Resolve Phase 2 row budget then sample with aligned symbol windows., _resolve_sample_total_rows(), sample_df_for_phase2(), _make_multi_sym_df(), DataFrame (+18 more)

### Community 93 - "Encoder"
Cohesion: 0.10
Nodes (20): ConfigurationError, decode_chromosome(), encode_condition(), Encoder, get_dont_care(), Exception, ndarray, encoder.py — Encoder Maps gene integer values to fuzzy value names, formats… (+12 more)

### Community 94 - "compute_phase2_objectives_from_metrics"
Cohesion: 0.04
Nodes (37): compute_phase2_objectives_from_metrics(), Penalty for weak cross-symbol robustness on one split., Build Phase 2 minimisation objectives from precomputed train/val metrics.…, _symbol_robustness_penalty(), True when val-derived feasibility penalties belong in NSGA-III fitness., _val_terms_in_fitness(), parametrize, f3 uses robust return = min(train_return, val_return) when… (+29 more)

### Community 95 - "test_crash_fix_properties.py"
Cohesion: 0.10
Nodes (26): _append_xla_flag(), configure_jax_env(), _cuda_package_root(), Path, JAX/XLA runtime configuration — call before the first ``import jax``., Find a pip-installed CUDA toolkit root, if one is available., Append an XLA flag once, preserving explicit user configuration., Configure JAX/XLA runtime for predictable desktop-friendly GPU usage. -… (+18 more)

### Community 96 - "test_mtf_ensembler.py"
Cohesion: 0.10
Nodes (39): compute_archive_hash(), compute_rule_hash(), get_default_archive_path(), load_mtf_archive_payload(), load_mtf_rule_archive(), normalize_timeframe(), Any, Path (+31 more)

### Community 97 - "_dominates"
Cohesion: 0.14
Nodes (11): _archive_objective_vector(), _dominates(), _is_better_archive_entry(), _non_dominated_sort(), Return True if solution *a* dominates *b* (all objectives ≤, at least one <)., NSGA-II non-dominated sorting. Parameters ---------- objectives : np.ndarray…, Convert an archive entry into minimisation objectives for ranking., Return True when *candidate* should replace *incumbent* for the same chromosome. (+3 more)

### Community 98 - ".load_strategies"
Cohesion: 0.18
Nodes (6): Load long.json and short.json via Output_Writer.load_and_validate(). Parameters…, Return only non-empty RB strategies accepted for deployment., Write a valid rule set JSON to path., Override module-level path dicts and return originals (for standalone tests)., TestLoadStrategies, _write_rule_set()

### Community 99 - "gate_positive_good"
Cohesion: 0.26
Nodes (17): _passes_pool_admission_impl(), _finite_number(), gate_positive_good(), _metric_int(), positive_good_reject_reasons(), PositiveGoodThresholds, Shared metric gates used by Phase 2 admission and RB Governor. The gate is…, Minimum evidence required on train and validation splits. (+9 more)

### Community 100 - "_pool_admission_floors"
Cohesion: 0.14
Nodes (12): effective_min_trade_pool_floor(), effective_pool_min_val_trades(), _evolution_feasibility_floors(), _pool_admission_floors(), Return (train_trade_floor, train_ret_min, val_ret_min, pf_floor,…, Return (train_trade_floor, train_ret_min, val_ret_min, pf_floor,…, _pool_admission_floors returns the ADMISSION floor (1.15), not the EVOLUTION…, Tests for _evolution_feasibility_floors — EVOLUTION PF 1.0 vs ADMISSION PF 1.15. (+4 more)

### Community 101 - "Data_Loader"
Cohesion: 0.13
Nodes (19): Data_Loader, _ensure_labels(), load_dataset(), DataFrame, data/loader.py — Data_Loader Stateless CSV loading with full preparation…, Keep supplied labels or derive all labels from raw OHLCV columns. The original…, Stateless data loader for the GPU-Fuzzy Trading Pipeline., Load a CSV dataset with full preparation pipeline: 1. Read CSV with comma… (+11 more)

### Community 102 - "_remove_low_dispersion"
Cohesion: 0.17
Nodes (10): Remove features where more than `threshold` fraction of values are identical.…, _remove_low_dispersion(), DataFrame, Exactly 95% identical → NOT > 0.95 → keep., 96% identical → > 0.95 → remove., Two features with corr > 0.95 → keep higher-scored one., Two uncorrelated features → both kept., Features in different modes are not compared for redundancy. (+2 more)

### Community 103 - "TestEquityCurvePlots"
Cohesion: 0.13
Nodes (13): Build a flat list of per-symbol metric dicts for CSV output. Uses the…, DataFrame, Verify plot_equity_curve is called for all three splits and handles empty logs., Helper: run OOS_Evaluator.run() with mocked internals and capture plot calls., plot_equity_curve must be called with 'train', 'validation', and 'test'., plot_equity_curve must be called exactly three times (one per split)., Empty train trade log must not raise an exception., Empty validation trade log must not raise an exception. (+5 more)

### Community 104 - "prop_settings"
Cohesion: 0.24
Nodes (16): HealthCheck, prop_settings(), Hypothesis settings with optional low-memory example scaling., given, Property-based tests for gpu_fuzzy_trader.reporting.reporter.Reporter This file…, **Validates: Requirements 6.4, 6.5, 6.6** For any valid inputs and any…, **Validates: Requirements 2.2, 2.3, 2.4** For any rule_set of length N with…, test_property_1_file_creation_round_trip() (+8 more)

### Community 105 - "TestPlotDistributionAndEquity"
Cohesion: 0.16
Nodes (6): _make_dist_logs_by_split(), _make_dist_trade_log(), Create a trade log with all columns needed for distribution/equity plots., Create trade_logs_by_split dict with all three splits populated., A trade log with a single trade should not raise., TestPlotDistributionAndEquity

### Community 106 - "baselines.py"
Cohesion: 0.24
Nodes (17): compute_entry_time_priority(), Map each row to a timestamp priority code (evaluator_v5 parity)., _compact(), _drop_one_condition_variants(), _equal_weight(), _evaluate(), evaluate_baselines(), _feature_shuffle() (+9 more)

### Community 107 - "TestWriteSpearmanCorrelationReport"
Cohesion: 0.13
Nodes (10): _make_dataset_with_label(), _make_datasets_by_split(), Feature not in dataset → NaN for that split., Dataset without label_close_288 → NaN for all features on that split., None dataset for a split → NaN for all features on that split., Empty selected_features → CSV with header only., All non-NaN Spearman values must be in [-1.0, 1.0]., Create a dataset DataFrame with feature columns and label_close_288. (+2 more)

### Community 108 - "_jax_compute_trade_outcomes"
Cohesion: 0.24
Nodes (5): _jax_compute_trade_outcomes(), JAX-jitted vectorized trade outcome computation., Vectorized trade outcome computation for all rows. Mirrors…, Multiple rows processed simultaneously., TestComputeTradeOutcomes

### Community 109 - "_should_post_restart_early_stop_phase2"
Cohesion: 0.38
Nodes (5): Break the run when a plateau restart yields no improvement., _should_post_restart_early_stop_phase2(), Unit tests for post-restart no-improvement early stop (Phase 2 runtime)., test_global_disabled_no_stop(), test_global_uses_global_knobs()

### Community 110 - "hypothesis_config.py"
Cohesion: 0.15
Nodes (15): Shared Hypothesis settings for property tests. Set PYTEST_LOW_MEMORY=1 to scale…, _make_timestamps(), composite, DataFrame, DrawFn, given, Timestamp, Property-based tests for gpu_fuzzy_trader.data.splitter.Data_Splitter… (+7 more)

### Community 111 - "nested_walk_forward.py"
Cohesion: 0.14
Nodes (24): Validation helpers for monthly, nested, and multiplicity-safe research., deflated_sharpe_ratio(), estimate_pbo(), Selection-multiplicity diagnostics for strategy research artifacts., Estimate the fraction of folds where the IS winner misses OOS median. Inputs…, Return a normal-approximation deflated Sharpe probability. This is a…, Create a compact ledger-ready multiplicity report., summarize_multiplicity() (+16 more)

### Community 112 - "_validate_pool_schema"
Cohesion: 0.36
Nodes (3): Validate the structure of a loaded pool JSON. Raises ValueError if the schema…, _validate_pool_schema(), TestValidatePoolSchema

### Community 113 - ".run"
Cohesion: 0.11
Nodes (12): DataFrame, Save a split report, marking consumed test data as diagnostic-only., Run out-of-sample evaluation. Parameters ---------- allowed_directions :…, Prepare test data using Data_Loader.load_dataset(). Applies the same…, Load prepared train, validation, and test datasets., Load and validate a strictly newer, untouched forward period., Load selected features for a direction when available., Remove only known Phase 5 artifacts from the active report root. (+4 more)

### Community 114 - "TestHammingThresholdAutoScale"
Cohesion: 0.15
Nodes (10): Verify the max(3, k_active // 5) formula. The formula is applied in…, Replicate the auto-scaling formula., k_active=0 → threshold = max(3, 0//5) = 3., k_active=5 → threshold = max(3, 5//5=1) = 3., k_active=15 → threshold = max(3, 15//5=3) = 3., k_active=20 → threshold = max(3, 20//5=4) = 4., k_active=50 → threshold = max(3, 50//5=10) = 10., k_active=100 → threshold = max(3, 100//5=20) = 20. (+2 more)

### Community 115 - ".decode_chromosome"
Cohesion: 0.26
Nodes (4): See module-level :func:`decode_chromosome`., All 10 signed values decode correctly via decode_chromosome., Chromosome with numpy int32/int64 values should work., TestDecodeChromosome

### Community 116 - "ValidationError"
Cohesion: 0.09
Nodes (27): context_permission_column(), context_trigger_column(), Return the direction-specific permission column name., Return the direction-specific LWC pullback-reversal trigger column., _condition_feature(), _context_feature_direction(), Exception, Path (+19 more)

### Community 117 - "_jax_compute_rule_signals"
Cohesion: 0.15
Nodes (11): _jax_compute_rule_signals(), _maybe_jit(), JAX-jitted vectorized rule matching (single chromosome)., Apply ``jit`` if JAX is available, otherwise return *fn* unchanged., Vectorized rule matching: returns (N,) boolean mask of matching rows., All rows match when chromosome equals data_matrix values., No rows match when chromosome differs from data_matrix., Only rows where all active conditions match. (+3 more)

### Community 118 - "TestParetoCollapseWarningGate"
Cohesion: 0.13
Nodes (11): _FakeEngine, Unit tests for Pareto-collapse warning gate (audit finding #13). AC: The…, AC 4: The default value of the config flag is 5., AC 5: min_pareto_size=0 → warning fires even with small pareto (size=4)., The log message includes 'pareto_size=N' suffix., Fake engine that returns metrics producing a tradeoff between f1 (-sortino) and…, AC 1–5: warning gated on len(pareto_indices) >= config threshold., Run 2-gen evolution and return count of 'Pareto collapse risk' warnings. (+3 more)

### Community 119 - "test_rb_governor_tail_holdout.py"
Cohesion: 0.11
Nodes (19): _passes_tail_selection_gate(), Validate a trial ruleset on the reserved chronological validation tail. The…, _MockEngine, DataFrame, Unit tests for RB Governor tail-holdout path in risk grid. Covers: -…, With tail_holdout_frac=0.25, a tail engine is returned with ~25% of data., With tail_holdout_frac=0.0, no tail engine., Single symbol with tail holdout still works. (+11 more)

### Community 120 - "_compute_stability"
Cohesion: 0.17
Nodes (9): _compute_stability(), Compute cross-symbol stability score. stability = 1 - (std / mean) If mean is 0…, All symbols have same score → std=0 → stability=1., Mean is 0 → stability=0., Single symbol with positive score → stability=1., Single symbol with zero score → stability=0., Very high variance → stability could be negative → clipped to 0., Stability should always be in [0, 1]. (+1 more)

### Community 121 - "_feasibility_gate_failures"
Cohesion: 0.10
Nodes (16): _feasibility_gate_failures(), Return per-gate failure flags for evolution-time feasibility diagnostics. Uses…, fixture, Tests for _feasibility_gate_failures — per-gate breakdown., A rule that should pass all 9 gates., A rule with too few train trades., A rule passing all gates returns all-zero dict., When val_metrics is None, only val_required=1, others=0. (+8 more)

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

### Community 126 - "_make_walk_forward_fold_engines"
Cohesion: 0.18
Nodes (12): _make_walk_forward_fold_engines(), Split val_selection into n_splits chronological folds + optional tail holdout.…, _make_synthetic_df(), DataFrame, Verify _make_walk_forward_fold_engines splits data correctly., 2 splits + tail holdout → 2 fold engines + 1 tail engine., tail_holdout_frac=0 → tail engine is None., Each symbol's data is divided into contiguous chunks across folds. (+4 more)

### Community 127 - "_derive_epoch_seed"
Cohesion: 0.16
Nodes (10): _derive_epoch_seed(), Return a deterministic per-epoch seed derived from base_seed + epoch., Re-sample training data with a per-epoch rotated window. Each epoch gets a…, An unknown PHASE2_PER_EPOCH_WINDOW_SEED_MODE raises ValueError., Deterministic seed derivation for per-epoch windows., Same (base_seed, epoch_idx) produces the same seed., Different epoch indices produce different seeds., None base seed returns None. (+2 more)

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
Nodes (11): log_memory_rss(), Optional process memory logging (set LOG_MEMORY=1)., Log current process RSS in MiB when LOG_MEMORY=1., Force GC and clear JAX compilation caches between Phase 2 directions. Releases:…, release_phase2_resources(), Release GPU engines and slim in-memory data between epochs., Drop engine and sampled data to free RAM before the next direction., Tests for optional memory logging helpers. (+3 more)

### Community 132 - "_init_population"
Cohesion: 0.11
Nodes (18): _plateau_diversity_restart(), Reinit a fraction of the population on plateau, preserving Pareto elite. Keeps…, _init_population(), Initialise a population of chromosomes. *init_strategy* ``"stratified_sparse"``…, Reinitialised slots have objectives=np.inf and metrics_cache={}., Even with large Pareto front, at most 10 elite are preserved., Direct unit tests for the _plateau_diversity_restart helper., Pareto elite chromosomes survive the restart. (+10 more)

### Community 133 - "_apply_colab_gpu_defaults"
Cohesion: 0.33
Nodes (6): _apply_colab_gpu_defaults(), is_colab_runtime(), True when running on Google Colab (/content runtime)., Colab T4 optimization for Phase 2 runs., MonkeyPatch, test_colab_defaults_apply_when_content_exists()

### Community 134 - "test_data_loader.py"
Cohesion: 0.18
Nodes (9): _base_row(), _make_ohlcv_rows(), _make_timestamps(), Unit tests for gpu_fuzzy_trader.data.loader.Data_Loader Tests cover: - CSV…, Generate n evenly-spaced ISO datetime strings., Return a minimal row dict with all required columns., Generate raw OHLCV rows without precomputed forward labels., TestOHLCVLabelGeneration (+1 more)

### Community 135 - "_build_rule_signal_mask"
Cohesion: 0.32
Nodes (7): _build_rule_signal_mask(), Cached wrapper around :func:`_compute_rule_signal_mask`., _backtest_df(), DataFrame, Regression tests for evaluator-facing Phase 2 chromosome semantics., Search fitness must use the same fuzzy class as RB/Phase 5 evaluation., test_batch_chromosome_signals_match_decoded_rule_conditions()

### Community 136 - ".get_dont_care"
Cohesion: 0.17
Nodes (7): See module-level :func:`get_dont_care`., **Property 9: Don't-Care Sentinel Correctness — encode_condition raises**…, **Property 9: Don't-Care Sentinel Correctness — all-dont_care → empty output**…, test_property_9b_encode_condition_raises_for_dont_care(), test_property_9f_all_dont_care_chromosome_returns_empty(), Static methods should be callable on the class itself., TestGetDontCare

### Community 137 - "resolve_phase2_stage_params"
Cohesion: 0.08
Nodes (20): _diversity_recovery_min_unique_ratio(), True when Stage A viability is critically low and search has plateaued., _should_inject_diversity_recovery(), _should_viability_recovery(), StageLabel, Phase 2 two-stage search hyperparameter profiles (exploration vs refinement)., Return stage-tuned hyperparameters. When *stage* is None (single-stage Phase…, resolve_phase2_stage_params() (+12 more)

### Community 138 - "test_phase2_gpu_throughput.py"
Cohesion: 0.15
Nodes (17): _jax_compute_rule_signals_batch(), Batch rule matching for B chromosomes simultaneously. Returns (B, N) boolean…, CPU and GPU backtest engine sub-package., get_gpu_backtest_engine_class(), jax_gpu_backtest_available(), Any, Detect whether JAX / GPUBacktestEngine can be loaded on this host. JAX can fail…, Return ``GPUBacktestEngine`` if ``gpu_engine`` imports cleanly; else ``None``. (+9 more)

### Community 139 - "CPUBacktestEngine"
Cohesion: 0.09
Nodes (52): CPUBacktestEngine, CPU backtest engine that exactly mirrors evaluator_v5.ipynb's…, _available_symbols(), _balanced_phase2_shortlist(), _cost_stress_gate(), _eval_cv_fold_returns(), evaluate_strategy_governor(), _f() (+44 more)

### Community 141 - "test_feature_selector.py"
Cohesion: 0.16
Nodes (10): _mutual_info_discrete_mask(), Return the set of feature names that PASS both stationarity checks. A feature…, Cap long/short feature overlap and backfill each direction to top_k features., Per-column ``discrete_features`` flags for ``mutual_info_classif``. Only…, _reduce_overlap(), _stationarity_filter(), Unit tests for gpu_fuzzy_trader.features.selector.Feature_Selector Tests cover:…, TestMutualInfoDiscreteMask (+2 more)

### Community 142 - "phase2_support.py"
Cohesion: 0.11
Nodes (18): compute_robust_score(), deployability_rank_score(), feasibility_violation_score(), _joint_primary_metric(), phase2_support.py — Trade support penalties for Phase 2., Train-only or conservative min(train, val) for ranking / objectives., Conservative return used for objectives, plateau, and archive ranking., Conservative win rate for ranking when f3 uses win rate. (+10 more)

### Community 143 - "_NumpyJSONEncoder"
Cohesion: 0.15
Nodes (10): LayerDiscoveryResult, Any, Small stable surface for comparing experiments. The legacy config module…, ResearchProfile, _merge_mtf_lwc_runtime_columns(), _NumpyJSONEncoder, Attach only causal HWC/MWC OOF scores to the LWC train tape., Merge causal LWC features and MTF scores into a raw research frame. The runtime… (+2 more)

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

### Community 151 - "test_phase2_support.py"
Cohesion: 0.19
Nodes (10): compute_support_penalty_and_specialist(), passes_pool_entry_admission(), Support penalty. Returns ------- penalty : float is_specialist : bool (always…, Post-merge filter for persisted Phase 2 pool JSON entries., Support penalty from train metrics. Returns (penalty, False, -1)., Legacy graduated penalty., _static_support_penalty(), trade_support_penalty() (+2 more)

### Community 152 - "BFS and DFS Graph Traversal"
Cohesion: 0.20
Nodes (10): CLAUDE.md Graphify Integration, Constrained Query Expansion, BFS and DFS Graph Traversal, Graph Work Memory, Existing Graph Fast Path, CLAUDE.md Graphify Integration, Constrained Query Expansion, BFS and DFS Graph Traversal (+2 more)

### Community 153 - ".simulate_rule_set"
Cohesion: 0.18
Nodes (8): _batch_eval_rule_set_pickled(), _jax_runtime_loaded(), Simulate a rule set and return performance metrics. Parameters ----------…, Simulate using a precomputed rule-evaluation mask cache. The mandatory…, Drop cached entries on rows the fixed context mask forbids. ``idx`` validity is…, Evaluate multiple rule sets without forking an active JAX runtime., Top-level worker for ProcessPoolExecutor (must be picklable)., Return whether forking would inherit an already multithreaded JAX runtime.

### Community 154 - "load_cached_split_if_fresh"
Cohesion: 0.30
Nodes (6): load_cached_split_if_fresh(), Load cached split parquets when they are newer than the source CSV. Validates…, load_cv_folds_manifest(), Any, Load manifest if present., TestLoadCachedSplitIfFresh

### Community 156 - ".run"
Cohesion: 0.06
Nodes (40): _condition_feature_names(), _entry_validation_per_symbol_metrics(), _filter_compatible_previous_pool(), _filter_pool_by_admission(), _merge_archive_entries(), _monthly_admission_source_df(), _pool_entry_passes_admission(), _pool_entry_rank() (+32 more)

### Community 157 - "passes_pool_admission_gate"
Cohesion: 0.30
Nodes (5): passes_pool_admission_gate(), Hard gate for Phase 2 pool/archive on merged holdout metrics. When…, MonkeyPatch, When PHASE2_JOINT_TRAIN_VAL=True, the f4 gate must use max(train_f4, val_f4),…, TestPoolAdmissionGate

### Community 158 - "TestGlobalMetricsCacheClearing"
Cohesion: 0.27
Nodes (6): When seeded_keys is empty, nothing is removed., When all keys are seeded, cache becomes empty., Verify that only seeded keys are removed from the global cache., Create a deterministic chromosome key., Cache entries matching seeded_keys are removed; non-matching survive., TestGlobalMetricsCacheClearing

### Community 159 - "test_evaluator_health.py"
Cohesion: 0.17
Nodes (7): Unit tests for ``evaluator_health_penalty`` and ``execution_ok`` (Task 4).…, Verify that evaluator health penalty is subtracted from combo score., When evaluator_health_penalty returns > 0, the combo score is lower., When evaluator health is fine, no penalty is applied., Both public functions are importable from the module., TestHealthPenaltyWiredIntoRB, TestModuleImportable

### Community 160 - "_nsga3_environmental_selection"
Cohesion: 0.15
Nodes (11): _deduplicate_selection_indices(), _nsga3_environmental_selection(), NSGA-III environmental selection (rank + niche on last front)., Replace duplicate genotypes in survivors with unused unique merge rows., _warn_evox_unavailable(), skipif, TestDuplicateSuppression, TestRunPhase2EvolutionSmoke (+3 more)

### Community 161 - "_legacy_writer_contract"
Cohesion: 0.67
Nodes (3): _legacy_writer_contract(), fixture, These schema tests predate mandatory trend context.

### Community 162 - "test_property_27_test_data_preparation_consistency"
Cohesion: 0.20
Nodes (11): _make_timestamps(), composite, DataFrame, DrawFn, given, Timestamp, **Property 27: Test Data Preparation Consistency** **Validates: Requirements…, Return n strictly increasing timestamp strings starting from *base*. (+3 more)

### Community 163 - "TestPoolAdmissionOverfitRatioGate"
Cohesion: 0.24
Nodes (7): MonkeyPatch, Tests for the hard overfit ratio gate in pool admission. The ratio gate rejects…, train=15%/val=4% (3.75× ratio, gap=11pp < 16pp) is REJECTED by the ratio gate…, train=15%/val=10% (1.5× ratio) is ADMITTED., train/val ≈2.5× is ADMITTED when under OVERFIT_RATIO_FLOOR and gap gate., With PHASE2_OVERFIT_RATIO_FLOOR=0.0, the ratio gate is disabled and the high-…, TestPoolAdmissionOverfitRatioGate

### Community 164 - "resolve_evolution_floors"
Cohesion: 0.31
Nodes (6): EvolutionFloors, Resolved evolution-time floors (pool admission gates remain strict)., Return stage-aware fitness floors; defaults to global strict knobs. When both…, resolve_evolution_floors(), Stage A soft floors must survive optional floor overrides., TestResolveEvolutionFloorsWithOverrides

### Community 165 - "evaluator_health.py"
Cohesion: 0.40
Nodes (5): evaluator_health.py — Pure functions for evaluator-failure-mode awareness.…, Read a numeric metric, returning *default* for missing / None / NaN / Inf., Read an integer metric safely., _safe_float(), _safe_int()

### Community 166 - "run_rb_governor_pipeline"
Cohesion: 0.08
Nodes (41): effective_rb_min_distinct_symbols(), mandatory_context_conditions(), Return the RB coverage target for the active debug universe. Full runs keep…, Return the fixed, mandatory context conditions for *direction*., feature_conditions_only(), phase2_rule_id(), Return normalized non-symbol, non-context conditions in order. Symbol scope and…, Hash the immutable Phase 2 feature logic and its discovery scope. (+33 more)

### Community 167 - "effective_min_profitable_symbols"
Cohesion: 0.25
Nodes (5): _debug_symbol_universe_size(), effective_min_profitable_symbols(), Active symbol count when debug scope is on; None for full-universe runs., Cap cross-symbol profitability gate to the active universe size. With…, test_effective_min_profitable_symbols_caps_debug_universe()

### Community 168 - "composite"
Cohesion: 0.27
Nodes (10): equity_tracking_scenario(), fee_deduction_scenario(), overlapping_rule_set_strategy(), composite, DrawFn, Generate a rule set of 1–4 rules that may overlap in their conditions, plus a…, Generate a random price scenario for a single trade. Returns a dict with:…, Generate a scenario for Property 14. Produces fee_pct, capital_pct,… (+2 more)

### Community 169 - "TestSparsePositiveMode"
Cohesion: 0.22
Nodes (5): All non-negative, zero_ratio > 0.3 → sparse_positive., All zeros: zero_ratio = 1.0 > 0.3, min = 0 → sparse_positive., zero_ratio just above 0.3 → sparse_positive., zero_ratio must be computed on the full series, not just non-NaN., TestSparsePositiveMode

### Community 170 - "TestZeroRatioBoundary"
Cohesion: 0.22
Nodes (5): Exactly 30% zeros with non-negative values → positive (not sparse_positive)., 31% zeros with non-negative values → sparse_positive., Exactly 30% zeros with negative values → signed (not sparse_signed)., Just above 30% zeros with negative values → sparse_signed., TestZeroRatioBoundary

### Community 171 - "run_pipeline.py"
Cohesion: 0.07
Nodes (34): apply_fuzzy_feature_scaling(), fit_fuzzy_feature_scaling(), Any, DataFrame, Train-fitted normalization for ordinal ``ff_*`` fuzzy feature codes. The…, Build a train-only scaling contract for ordinal ``ff_*`` columns., Apply an existing train-fitted scaling contract in place and return *df*., __main__.py — Entry point for `python -m gpu_fuzzy_trader.run_pipeline` Allows… (+26 more)

### Community 172 - "TestHallOfFameTrim"
Cohesion: 0.25
Nodes (5): Verify that hall_of_fame is capped to PHASE2_HOF_EPOCH_CARRYOVER=10., HoF with >10 entries is trimmed to exactly 10 oldest entries., HoF with <=10 entries is left unchanged., Empty HoF does not crash during trim logic., TestHallOfFameTrim

### Community 173 - "cpu_engine.py"
Cohesion: 0.10
Nodes (22): cpu_engine.py — CPUBacktestEngine Exact Python/NumPy replication of…, Simulate a rule set on rows [row_start, row_end) without copying the df. Used…, Sort entries for v5 capital allocation (timestamp, rule, symbol, row)., _rule_symbols_for_allocation(), _rules_need_normalized_symbols(), _safe_profit_factor(), _sort_entries_by_allocation_priority(), Joint long/short portfolio simulation. Phase 2 and RB score each direction… (+14 more)

### Community 175 - "test_phase2_offspring_batch.py"
Cohesion: 0.29
Nodes (5): CountingEngine, Unit tests for batched offspring evaluation (Phase 2 runtime A1)., Stub engine recording every simulate_rule_batch call's batch size., Offspring should be evaluated via ONE simulate_rule_batch call per gen, not…, test_offspring_evaluated_in_single_batch_per_gen()

### Community 176 - "test_gpu_engine_import_does_not_crash_on_jax_failure"
Cohesion: 0.33
Nodes (6): parametrize, Tests for the lazy-JAX failure path in gpu_engine.py. These tests verify that…, Verify importing gpu_engine.py handles various JAX failure modes gracefully., Build a subprocess script that simulates JAX import failure., _subprocess_script(), test_gpu_engine_import_does_not_crash_on_jax_failure()

### Community 177 - "test_jax_compat.py"
Cohesion: 0.29
Nodes (5): parametrize, Tests for JAX / GPU engine availability detection., Package init must not crash when gpu_engine import fails., test_cpu_engine_import_without_jax(), test_get_gpu_backtest_engine_class_returns_none_on_import_error()

### Community 178 - "TestExecutionHealthInGate"
Cohesion: 0.25
Nodes (5): Verify that ``gate_positive_good`` calls ``execution_ok`` when flagged., When require_execution_health=True and execution_ok fails, gate returns False., When require_execution_health=True and execution_ok passes, gate still returns…, When require_execution_health=False, gate ignores raw_signal_count., TestExecutionHealthInGate

### Community 179 - "trade_support_penalty"
Cohesion: 0.38
Nodes (4): Backward-compatible wrapper returning penalty only., trade_support_penalty(), Between the hard-reject floor and the soft threshold the penalty is graduated., TestTradeSupportPenalty

### Community 180 - "TestChronologicalOrdering"
Cohesion: 0.29
Nodes (4): All train datetimes for a symbol must be < validation datetimes (embargo gap)., Chronological ordering holds independently for each symbol., Train rows should be the first floor(N * train_frac) rows by feature_a index., TestChronologicalOrdering

### Community 181 - "TestNaNHandling"
Cohesion: 0.29
Nodes (4): All-NaN series: unique_vals = [], n_unique = 0 ≤ 2, set() ⊆ {0,1} → binary., NaN is not == 0, so it does not inflate zero_ratio., NaN should not push zero_ratio above threshold., TestNaNHandling

### Community 182 - "scoring/__init__.py"
Cohesion: 0.33
Nodes (5): profit_factor_term(), Scoring helpers shared across pipeline phases. Re-exports…, Return divided by max drawdown; higher is better. A small drawdown floor avoids…, Clamp a profit factor into ``[0, cap]``; non-finite → ``cap``., return_to_drawdown()

### Community 183 - "TestSparseSignedMode"
Cohesion: 0.33
Nodes (3): Has negative values, zero_ratio > 0.3 → sparse_signed., NaN does not count as zero; zero_ratio on full series., TestSparseSignedMode

### Community 184 - "TestEvalCvFoldReturns"
Cohesion: 0.33
Nodes (4): Verify the helper handles None / empty fold_engines safely., fold_engines=None returns None without crashing., fold_engines=[] returns None without crashing., TestEvalCvFoldReturns

### Community 185 - "DataFrame"
Cohesion: 0.50
Nodes (5): _downsample_chronological(), DataFrame, Preserve per-symbol time order required by exposure/release simulation., Pick a contiguous chronological slice of *n_rows* from *df*. Critical for…, _sort_chronological()

### Community 189 - "opencode.json"
Cohesion: 0.50
Nodes (3): plugin, $schema, .opencode/plugins/graphify.js

## Knowledge Gaps
- **58 isolated node(s):** `$schema`, `.opencode/plugins/graphify.js`, `Task 1: Causal Multi-Timeframe Data Engine`, `Task 2: Directional & Conditional Evaluators & Rule Search Profiles`, `Task 3: Master Temporal Folds, Purged Embargo & OOF Cross-Fitting` (+53 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **11 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `CPUBacktestEngine` connect `CPUBacktestEngine` to `gpu_engine.py`, `_make_engine`, `Pipeline_Orchestrator`, `_score_metrics`, `_build_rule_signal_mask`, `_make_df`, `test_phase2_gpu_throughput.py`, `_apply_monthly_admission_gate`, `_NumpyJSONEncoder`, `test_cpu_engine_properties.py`, `.simulate_rule_set`, `CandidateRecord`, `Rule_Pool_Generator`, `_build_pool_from_archive`, `run_rb_governor_pipeline`, `run_pipeline.py`, `cpu_engine.py`, `test_certificate_first_selection.py`, `test_jax_compat.py`, `_compute_rule_signal_mask`, `trade_support_penalty`, `phase2_rule_pool.py`, `TestEvalCvFoldReturns`, `GPUBacktestEngine`, `._simulate_rule_set_entries`, `test_phase2_use_gpu_flag.py`, `test_mtf_pipeline_integration.py`, `_apply_dynamic_rule`, `JointPortfolioEngine`, `test_gpu_engine.py`, `_symbol_specialized_variants`, `OOS_Evaluator`, `test_gpu_engine_properties.py`, `TestGPUCPUNumericalParity`, `baselines.py`, `_jax_compute_trade_outcomes`, `nested_walk_forward.py`, `.run`, `ValidationError`, `_jax_compute_rule_signals`, `test_rb_governor_tail_holdout.py`, `MonthlyWindowSummary`, `._engine`, `_make_walk_forward_fold_engines`?**
  _High betweenness centrality (0.164) - this node is a cross-community bridge._
- **Why does `Reporter` connect `Reporter` to `DataFrame`, `TestEquityCurveDateAxis`, `_build_pool_from_archive`, `prop_settings`, `TestPlotDistributionAndEquity`, `run_pipeline.py`, `._ensure_dir`, `Rule_Pool_Generator`, `TestWriteSpearmanCorrelationReport`, `TestWriteStrategyEvaluationTable`, `.run`, `TestPlotPerRuleBreakdown`, `ValidationError`, `OOS_Evaluator`, `test_reporter.py`, `phase2_rule_pool.py`, `.run`?**
  _High betweenness centrality (0.091) - this node is a cross-community bridge._
- **Why does `Rule_Pool_Generator` connect `Rule_Pool_Generator` to `log_memory_rss`, `Pipeline_Orchestrator`, `_init_population`, `_make_feature_infos`, `TestValLeakGate`, `_mutate`, `CPUBacktestEngine`, `_NumpyJSONEncoder`, `_apply_monthly_admission_gate`, `TestRefreshObjectivesOnResumeGate`, `_derive_val_sample_seed`, `_make_train_df`, `Reporter`, `.run`, `TestPoolAdmissionOverfitRatioGate`, `run_rb_governor_pipeline`, `_build_pool_from_archive`, `TestEndToEndRotation`, `run_pipeline.py`, `_get_dont_cares`, `test_crash_fix_and_run_logging.py`, `test_phase2_rule_pool.py`, `phase2_rule_pool.py`, `test_phase2_use_gpu_flag.py`, `test_plateau_state_leak.py`, `TestRunLogHandlerLifecycle`, `.save_archive`, `test_phase2_window_rotation.py`, `compute_phase2_objectives_from_metrics`, `_dominates`, `_validate_pool_schema`, `_derive_epoch_seed`?**
  _High betweenness centrality (0.072) - this node is a cross-community bridge._
- **Are the 15 inferred relationships involving `Reporter` (e.g. with `CvFoldValEvaluator` and `Rule_Pool_Generator`) actually correct?**
  _`Reporter` has 15 INFERRED edges - model-reasoned connections that need verification._
- **Are the 54 inferred relationships involving `CPUBacktestEngine` (e.g. with `GPUBacktestEngine` and `JointPortfolioEngine`) actually correct?**
  _`CPUBacktestEngine` has 54 INFERRED edges - model-reasoned connections that need verification._
- **Are the 58 inferred relationships involving `Rule_Pool_Generator` (e.g. with `CPUBacktestEngine` and `Reporter`) actually correct?**
  _`Rule_Pool_Generator` has 58 INFERRED edges - model-reasoned connections that need verification._
- **Are the 19 inferred relationships involving `Pipeline_Orchestrator` (e.g. with `CPUBacktestEngine` and `Data_Loader`) actually correct?**
  _`Pipeline_Orchestrator` has 19 INFERRED edges - model-reasoned connections that need verification._