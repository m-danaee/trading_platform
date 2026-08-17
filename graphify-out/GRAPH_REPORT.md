# Graph Report - trading_platform  (2026-08-17)

## Corpus Check
- 186 files · ~257,879 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 4859 nodes · 11099 edges · 183 communities (174 shown, 9 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 557 edges (avg confidence: 0.56)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `81e7aa89`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- evox_runner.py
- Rule_Pool_Generator
- ._simulate_rule_set_entries
- test_cpu_engine.py
- .run
- trend_context.py
- test_evox_runner.py
- _score_metrics
- TestSymbolGeneBias
- _make_df
- _get_dont_cares
- run_phase2_evolution
- splitter.py
- test_phase5_oos.py
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
- _make_deployable_entry
- maybe_log_generation
- RB Governor production path
- TestSelectDiverseSubset
- test_rb_min_symbols.py
- Graphify Pipeline
- Output_Writer
- config.py
- TestEquityCurveDateAxis
- test_phase2_window_rotation.py
- research_integrity.py
- _loader_from_rows
- rb_governor.py
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
- TestWriteSpearmanCorrelationReport
- test_phase2_rule_pool.py
- _make_rule
- test_reporter.py
- GPUBacktestEngine
- _make_df
- ValueError
- ._build_engine_for_df
- _derive_val_sample_seed
- TestRulePoolGeneratorRun
- test_plateau_state_leak.py
- .prepare_test_data
- dashboard.py
- DataFrame
- test_data_loader_properties.py
- .encode_condition
- _m
- _apply_dynamic_rule
- JointPortfolioEngine
- run_rb_governor_pipeline
- execution_ok
- TestWriteStrategyEvaluationTable
- test_gpu_engine.py
- test_crash_fix_and_run_logging.py
- test_property_19_phase2_static_risk_parameters
- test_feature_selector_properties.py
- .get_dont_care
- compute_labels
- TestPlotPerRuleBreakdown
- resolve_island_hyperparams
- _run_cluster_islands
- _build_target
- filter_migrants_for_cluster
- test_gpu_engine_properties.py
- _write_and_reload
- TestInitPopulation
- loader.py
- set_purged_wf_reference_rows
- downcast_numeric_df
- Encoder
- ValidationError
- TestPermissionGatedPullback
- test_phase2_gpu_throughput.py
- .decode_chromosome
- .load_strategies
- gate_positive_good
- phase2_island_scheduler.py
- .save_archive
- _remove_low_dispersion
- TestEquityCurvePlots
- test_crash_fix_properties.py
- TestPlotDistributionAndEquity
- baselines.py
- phase2_rule_pool.py
- _jax_compute_trade_outcomes
- _should_post_restart_early_stop_phase2
- test_data_splitter_properties.py
- compute_phase2_objectives_from_metrics
- _validate_pool_schema
- .run
- TestHammingThresholdAutoScale
- TestValLeakGate
- _build_entries_from_rule_set
- _jax_compute_rule_signals
- TestParetoCollapseWarningGate
- _make_walk_forward_fold_engines
- _compute_stability
- passes_pool_admission_gate
- CPUBacktestEngine
- ._engine
- test_rb_concentration_tail_fail_closed.py
- constrained_non_dominated_sort
- test_feature_selector.py
- TestEndToEndRotation
- stratification_scenario_strategy
- TestSpearmanSignConsistency
- run_pipeline.py
- log_memory_rss
- test_evolution_convergence.py
- _apply_colab_gpu_defaults
- Data_Splitter
- test_evaluator_clean_writer.py
- OOS_Evaluator
- TestF4ReturnConcentration
- TestDecoupledObjectives
- context_contract_digest
- TestMigrationSeedFraction
- TestEdgeCases
- ._prune_splits_after_phase1
- trade_support_penalty
- TestRefreshObjectivesOnResumeGate
- conftest.py
- TestIslandAwareTradeFloor
- .skip_if_valid
- _validate_schema
- ResearchProfile
- TestDeferredWarmup
- .skip_if_valid
- BFS and DFS Graph Traversal
- TestPlateauDiversityRestart
- TestCausalPublicationTiming
- environmental_selection_nsga2
- _make_csv
- Data_Loader
- TestGlobalMetricsCacheClearing
- TestRobustReturnObjective
- ConfigError
- TestF3PathResolution
- TestTriggerPerSymbol
- phase2_should_enrich_symbol_metrics
- TestHealthPenaltyWiredIntoRB
- TestHallOfFameTrim
- phase2_support.py
- test_evaluator_health.py
- test_gpu_engine_import_does_not_crash_on_jax_failure
- test_jax_compat.py
- TestExecutionHealthInGate
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

## Communities (183 total, 9 thin omitted)

### Community 0 - "evox_runner.py"
Cohesion: 0.04
Nodes (123): _assign_eval_result(), _behavior_descriptor(), _binary_tournament_pick(), _build_diversity_reference(), _build_rank_and_crowding(), _constraint_violations(), _count_deployable_preview(), _count_pop_viable() (+115 more)

### Community 1 - "Rule_Pool_Generator"
Cohesion: 0.08
Nodes (15): Phase 2: GPU-accelerated multi-objective evolutionary rule pool generation.…, Inject guarded migration seeds for the next epoch., Restore slimmed training data from cache (no re-sampling needed)., Attach optional island metadata; safe when *owner* is a partial mock., Rule_Pool_Generator, _make_feature_infos(), In holdout mode, val engine must be built for pool admission even when…, All pool entries must have active conditions within [MIN_CONDITIONS,… (+7 more)

### Community 2 - "._simulate_rule_set_entries"
Cohesion: 0.09
Nodes (17): _batch_eval_rule_set_pickled(), _jax_runtime_loaded(), Simulate a rule set on rows [row_start, row_end) without copying the df. Used…, Simulate a rule set and return performance metrics. Parameters ----------…, Simulate using a precomputed rule-evaluation mask cache. The mandatory…, Drop cached entries on rows the fixed context mask forbids. ``idx`` validity is…, Evaluate multiple rule sets without forking an active JAX runtime., Top-level worker for ProcessPoolExecutor (must be picklable). (+9 more)

### Community 3 - "test_cpu_engine.py"
Cohesion: 0.07
Nodes (27): _normalize_direction(), _make_df(), _make_engine(), Unit tests for CPUBacktestEngine. Tests verify exact evaluator_v5.ipynb…, Simulate catastrophic losses to trigger account ruin. With min_288=0…, Trades with tiny equity should be skipped., All TP trades → win_rate = 100%., No losing trades → profit_factor = 99.0. (+19 more)

### Community 4 - ".run"
Cohesion: 0.06
Nodes (39): FileHandler, count_trials(), Estimate the number of adaptive evaluations represented by artifacts., _log_phase_entry(), _log_pipeline_config(), _now_iso(), _phase2_frame_identity(), _phase2_resume_identity() (+31 more)

### Community 5 - "trend_context.py"
Cohesion: 0.08
Nodes (49): Number of leading per-symbol rows belonging to the training prefix. Shared by…, train_prefix_row_count(), align_completed_states_to_rows(), average_true_range(), build_higher_bars(), build_manifest(), build_train_prefix(), _classify_hf_bars() (+41 more)

### Community 6 - "test_evox_runner.py"
Cohesion: 0.04
Nodes (58): _diversity_recovery_min_unique_ratio(), _inherit_val_metrics_from_global_cache(), Phase2EvolutionState, True when Stage A viability is critically low and search has plateaued., Copy val_* from global cache for identical chromosomes when val is skipped.…, Evolve one island epoch and return updated resumable state., Resumable NSGA-III state for symbol-island epoch scheduling., Resolve the plateau patience value based on profile and stage. Cluster/orphan… (+50 more)

### Community 7 - "_score_metrics"
Cohesion: 0.07
Nodes (37): _combined_return_score(), _evaluate_ruleset(), _evaluator_health_penalty(), _i(), _optimize_risk(), Penalize evaluator_v5 execution problems: too many skipped signals, low…, Dominant objective: return/DD with train-valid balance, plus CV-fold…, Profit objective for lenient rule addition, but now evaluator_v5 aware. A new… (+29 more)

### Community 8 - "TestSymbolGeneBias"
Cohesion: 0.17
Nodes (9): C5 mutation bias: force symbol-gene to dont_care / inactive with probability…, Create feature_infos with a feature whose name contains 'symbol'., PHASE2_SYMBOL_GENE_DONT_CARE_PROB=1.0: symbol gene always forced to dont_care., PHASE2_SYMBOL_GENE_DONT_CARE_PROB=0.0: symbol gene never force-set., With probability ~0.5, about half of calls force symbol to dont_care., No symbol feature in feature_infos: bias silently does nothing (no crash)., Sparse path: PHASE2_SYMBOL_GENE_DONT_CARE_PROB=1.0 forces symbol slot to…, Sparse path: PHASE2_SYMBOL_GENE_DONT_CARE_PROB=0.0, symbol slot stays active. (+1 more)

### Community 9 - "_make_df"
Cohesion: 0.08
Nodes (21): _make_df(), _make_engine(), MonkeyPatch, Chromosome positions must follow feature_modes insertion order., Exact CPU re-evaluation must interpret genes like the GPU path., Chromosome that matches nothing returns 0 executed trades., Zero-signal chunks should use reject metrics without scanning., PHASE2_SKIP_ZERO_SIGNAL_SCAN must match always-scan metrics. (+13 more)

### Community 10 - "_get_dont_cares"
Cohesion: 0.06
Nodes (75): assign_strata_to_indices(), pick_active_count(), _pick_active_index(), _pick_inactive_index(), ndarray, _random_active_class(), phase2_init.py — Sparsity-guided stratified population initialization for Phase…, Assign elite / explorer labels to non-seeded population rows. (+67 more)

### Community 11 - "run_phase2_evolution"
Cohesion: 0.03
Nodes (79): True for cluster/orphan scoped runs rather than the global path., scoped_island_profile(), Run Phase 2 NSGA-III evolution. Returns (pareto_pool, history) or state., run_phase2_evolution(), _should_early_stop_phase2(), _should_plateau_early_stop_phase2(), Evolutionary algorithm drivers for Phase 2., The fallback must not switch f3 from CV return to PF after gen 0. (+71 more)

### Community 12 - "splitter.py"
Cohesion: 0.13
Nodes (21): _chronological_half_split(), _file_sha256(), _holdout_embargo_split(), _purged_walk_forward_split(), DataFrame, data/splitter.py — Data_Splitter Per-symbol chronological split for Phase 2,…, Per-symbol chronological first or second half of *df*. ``purge_rows`` is…, Split validation into purged fitness and selection halves per symbol. The gap… (+13 more)

### Community 13 - "test_phase5_oos.py"
Cohesion: 0.11
Nodes (15): _isolate_phase5_reporter_outputs(), fixture, Unit tests for gpu_fuzzy_trader.phases.phase5_oos.OOS_Evaluator Tests cover: -…, Override module-level path dicts and return originals., Keep Reporter plots/CSVs out of the checked-in outputs directory., Write a valid selected-features JSON to path., Write a synthetic test CSV with all required columns (including feat_0..4) to a…, Give integration tests an isolated, valid enriched train/split pair. (+7 more)

### Community 14 - "Pipeline_Orchestrator"
Cohesion: 0.10
Nodes (39): _context_coverage_preflight(), _context_coverage_report(), Pipeline_Orchestrator, Return the active output root for this run., Top-level orchestrator for the GPU-Fuzzy Trading Pipeline. Runs all five phases…, Temporarily rebind all cached output paths for one pipeline run., Return split-aware context coverage for both trading directions., Log coverage and block only directions that cannot meet their floors. (+31 more)

### Community 15 - "_apply_monthly_admission_gate"
Cohesion: 0.06
Nodes (27): _apply_monthly_admission_gate(), _monthly_window_metrics(), Normalize a window evaluator result for the monthly gate. The float fallback…, Apply the monthly-window shadow-test gate to a pool of rules. Each rule is…, _DeterministicEvaluator, _DeterministicMetricsEvaluator, DataFrame, fixture (+19 more)

### Community 16 - "test_feature_detector_properties.py"
Cohesion: 0.09
Nodes (45): all_nan_series(), arbitrary_numeric_series(), binary_series(), large_value_series(), positive_series(), composite, DrawFn, given (+37 more)

### Community 17 - "_make_train_df"
Cohesion: 0.08
Nodes (23): Sample up to *total_rows* rows, distributed equally across symbols. A single…, _sample_df(), _make_train_df(), DataFrame, Critical: bars must be contiguous so the backtest engine preserves temporal…, Random start must be bounded so the slice always fits forward., divmod distribution gives exactly total_rows (no rounding loss). 701_000 % 14…, total_rows < n_symbols must NOT force 1 row per symbol. (+15 more)

### Community 18 - "_split"
Cohesion: 0.07
Nodes (20): Helper: build df, patch paths, run split, return (train, val)., Compute expected train/val per symbol under holdout+embargo split., floor(N * HOLDOUT_TRAIN_FRACTION) rows go to train., Remaining rows after embargo gap go to validation., For N=101: floor(101 * HOLDOUT_TRAIN_FRACTION) used, not round., train + validation + embargo dropped == total rows., train + val + embargo_dropped == total for each symbol., Each symbol's split point is computed from its own row count. (+12 more)

### Community 19 - "test_cpu_engine_properties.py"
Cohesion: 0.08
Nodes (43): equity_tracking_scenario(), _expected_outcome(), fee_deduction_scenario(), _make_engine(), _make_engine_custom(), _make_minimal_df(), _make_simple_df(), multi_symbol_scenario() (+35 more)

### Community 20 - "detect_feature_mode"
Cohesion: 0.04
Nodes (48): detect_all_modes(), detect_feature_mode(), Feature_Detector, DataFrame, Series, detector.py — Feature_Detector Classifies each feature column into exactly one…, Classify feature columns by their discretization type., Classify a single feature series into one of six modes. Detection order… (+40 more)

### Community 21 - "selector.py"
Cohesion: 0.08
Nodes (41): get_seed(), Return a per-process random seed (stable within a run, different across runs).…, _align_feature_array(), build_phase1_shared_context(), _build_symbol_masks(), _candidate_feature_columns(), _check_spearman_sign_consistency(), _compute_chronological_stationarity_scores() (+33 more)

### Community 22 - "Feature_Selector"
Cohesion: 0.10
Nodes (20): Feature_Selector, Score and rank features separately for long and short directions., _make_train_df(), MonkeyPatch, Create a minimal training DataFrame with label columns and feature columns., Label columns must not appear in selected features., Meta columns must not appear in selected features., Loader internal columns and ``_``-prefixed names are not candidates. (+12 more)

### Community 23 - "rolling_cv.py"
Cohesion: 0.10
Nodes (35): aggregate_fold_metrics(), _bar_index_col(), _build_fold_from_ranges(), build_forbidden_ranges(), build_purged_walk_forward_folds(), cv_folds_only(), derive_primary_holdout(), FoldMetricsSummary (+27 more)

### Community 24 - "Reporter"
Cohesion: 0.10
Nodes (14): Generates visual and tabular reports for each pipeline phase. All output files…, Reporter, _make_history(), _make_per_symbol_metrics(), _make_pnl_history(), History entries with missing keys should not raise., History entries with missing keys should not raise., Create a minimal Phase 2 history list. (+6 more)

### Community 25 - "test_encoder_properties.py"
Cohesion: 0.09
Nodes (37): all_active_chromosome_strategy(), chromosome_with_dont_cares_strategy(), feature_name_strategy(), composite, DrawFn, given, ndarray, Property-based tests for gpu_fuzzy_trader.features.encoder.Encoder **Validates:… (+29 more)

### Community 26 - "write_evaluator_clean"
Cohesion: 0.12
Nodes (18): _maybe_write_evaluator_clean(), Write a stripped strategy file containing only ``direction`` and ``rules_set``.…, Write a stripped strategy file alongside the main strategy JSON. This is a…, write_evaluator_clean(), Path, Parent directory is created when it does not exist., Strategy with no extra keys works and produces correct output., Missing 'direction' raises KeyError. (+10 more)

### Community 27 - "CandidateRecord"
Cohesion: 0.12
Nodes (38): _candidate_coverage_symbols(), _candidate_positive_symbols(), CandidateRecord, _compose_ruleset(), _diversification_beam(), _diversification_shortlist(), _max_overlap(), _pair_overlap() (+30 more)

### Community 28 - "nested_walk_forward.py"
Cohesion: 0.14
Nodes (24): Validation helpers for monthly, nested, and multiplicity-safe research., deflated_sharpe_ratio(), estimate_pbo(), Selection-multiplicity diagnostics for strategy research artifacts., Estimate the fraction of folds where the IS winner misses OOS median. Inputs…, Return a normal-approximation deflated Sharpe probability. This is a…, Create a compact ledger-ready multiplicity report., summarize_multiplicity() (+16 more)

### Community 29 - "_make_deployable_entry"
Cohesion: 0.13
Nodes (15): _make_chromosome(), _make_deployable_entry(), ndarray, Without preservation, champion is evicted by gen ~8 under drift., At most TOP_K slots are overwritten by elite preservation., Preserved elite's objectives are reset to inf (forces re-eval)., Build a dense chromosome with some active genes., Snapshot: with ENABLED=False, evolution is unchanged (2-gen). (+7 more)

### Community 30 - "maybe_log_generation"
Cohesion: 0.09
Nodes (18): generation_log_interval(), iteration_log_interval(), log_generation(), maybe_log_generation(), log_progress.py — Throttled progress logging for long pipeline loops., Return how often to log generation progress. Uses LOG_GENERATION_INTERVAL from…, Log generation progress when the step matches the throttle interval., Log first step, last step, and every *interval* steps in between. (+10 more)

### Community 31 - "RB Governor production path"
Cohesion: 0.09
Nodes (27): Local low-memory test policy, Virtual environment command policy, Package compile gate, Locked research dependencies, Focused low-memory test gate, Python 3.11 CI runtime, Research CI workflow, Research artifact audit trail (+19 more)

### Community 32 - "TestSelectDiverseSubset"
Cohesion: 0.12
Nodes (12): ndarray, Verify _select_diverse_subset correctness for edge cases., Create n distinct dense chromosomes., k=0 should return [] even with non-empty chromosomes., k<0 should return []., k=1 returns one chromosome., k > len(chromosomes) returns all chromosomes (shallow copy)., k == len(chromosomes) returns all chromosomes. (+4 more)

### Community 33 - "test_rb_min_symbols.py"
Cohesion: 0.11
Nodes (24): _symbols_in_rules(), _dummy_df(), _make_candidate_records(), _mock_train_metrics(), _multi_symbol_rules(), _no_symbol_rule(), DataFrame, Tests for RB Governor min-distinct-symbols hard gate. After final opt_rules… (+16 more)

### Community 34 - "Graphify Pipeline"
Cohesion: 0.06
Nodes (36): Folder Watcher, URL Ingestion, Conditional Graph Exports, Graphify MCP Server, Extraction Confidence Rubric, Deterministic Full-Path Node IDs, Semantic Hyperedges, Cross-Repository Graph Merge (+28 more)

### Community 35 - "Output_Writer"
Cohesion: 0.09
Nodes (14): Output_Writer, Serializes RuleSet dicts to JSON and loads/validates existing JSON files.…, _legacy_writer_contract(), fixture, Unit tests for gpu_fuzzy_trader.output.writer — Output_Writer Tests cover: -…, These schema tests predate mandatory trend context., Verify the exact example from the spec works end-to-end., TestLoadAndValidateErrors (+6 more)

### Community 36 - "config.py"
Cohesion: 0.07
Nodes (33): context_contract(), _debug_symbol_universe_size(), effective_config_snapshot(), effective_min_profitable_symbols(), effective_min_trade_support(), effective_monthly_min_trades(), effective_rb_min_distinct_symbols(), effective_val_trade_floor_for_objectives() (+25 more)

### Community 37 - "TestEquityCurveDateAxis"
Cohesion: 0.09
Nodes (21): _make_trade_log(), DataFrame, fixture, Unit tests for date-based x-axis in Reporter.plot_equity_curve (Task 10.3).…, When Entry_Time is absent, x-axis label is 'Trade #'., When Entry_Time is all-NaN, x-axis label is 'Trade #'., axhline is called (initial-capital line) in date mode., axhline is called (initial-capital line) in Trade # mode. (+13 more)

### Community 38 - "test_phase2_window_rotation.py"
Cohesion: 0.06
Nodes (33): _derive_epoch_seed(), Return a deterministic per-epoch seed derived from *base_seed* + epoch. Used by…, _largest_safe_range(), Return (start, end) of the largest contiguous bar range not in *forbidden*. The…, Cap *total_rows* so the per-symbol request fits within the safe range. When…, _resolve_sample_total_rows(), _make_multi_sym_df(), DataFrame (+25 more)

### Community 39 - "research_integrity.py"
Cohesion: 0.16
Nodes (23): _canonical_json(), dataset_manifest(), ExperimentLedger, forward_acceptance_lock_path(), Any, Path, PathLike, Research-integrity utilities shared by the pipeline and Phase 5. The consumed… (+15 more)

### Community 40 - "_loader_from_rows"
Cohesion: 0.09
Nodes (23): _base_row(), _loader_from_rows(), _make_ohlcv_rows(), _make_rows(), _make_timestamps(), Unit tests for gpu_fuzzy_trader.data.loader.Data_Loader Tests cover: - CSV…, The first N-288 rows (chronologically) should be kept., A row with NaN in any label column should be dropped. (+15 more)

### Community 41 - "rb_governor.py"
Cohesion: 0.07
Nodes (52): expectancy_lcb_pct(), Return a conservative lower bound for per-trade net expectancy. Exact CPU…, _attach_source_symbol_filters(), _available_symbols(), _balanced_phase2_shortlist(), _cost_stress_gate(), _ensure_symbol_filtered_rule(), _eval_cv_fold_returns() (+44 more)

### Community 42 - "gpu_engine.py"
Cohesion: 0.06
Nodes (40): _batch_metrics_from_array(), _build_event_batch(), _exposure_slot_capacity(), _fast_reject_result_rows(), _jax_compute_rule_signals_batch(), _jax_compute_rule_signals_sparse_batch(), _jax_open_slot(), _jax_release_open_slots() (+32 more)

### Community 43 - "_gpu_runtime.py"
Cohesion: 0.11
Nodes (29): configure_phase2_gpu_runtime(), detect_gpu_memory_used_gb(), detect_gpu_vram_gb(), detect_system_ram_gb(), _iter_warmup_targets(), log_gpu_runtime_config(), _ram_batch_cap(), Phase 2 GPU runtime helpers: VRAM-aware batch size and JAX warmup. (+21 more)

### Community 44 - "._ensure_dir"
Cohesion: 0.07
Nodes (19): Reporting and visualization sub-package., _bucket_series_by_mode(), DataFrame, Series, reporter.py — Reporting and visualization for the GPU-Fuzzy Trading Pipeline.…, Write feature-stratified performance metrics to CSV files (one per split). For…, Create parent directories for *path* if they do not exist., Compute maximum drawdown percentage from an equity series. Parameters… (+11 more)

### Community 45 - "_derive_island_seed"
Cohesion: 0.09
Nodes (23): _derive_island_seed(), Return a deterministic but distinct seed per island from base seed + id.…, _coverage(), _print_island_sample_diagnostics(), _print_split_coverage(), DataFrame, Print coverage on deterministic singleton-island train/val windows., Return eligible-row counts for one direction and split frame. (+15 more)

### Community 46 - "test_certificate_first_selection.py"
Cohesion: 0.11
Nodes (27): _passes_symbol_concentration_gate(), _passes_symbol_contribution_certificate(), _passes_tail_holdout_gate(), _portfolio_selection_certificate(), Any, Require positive, supported validation PnL from multiple symbols. Symbol…, Build the per-direction symbol certificate policy. Specialist islands are…, Return the certificate used by compose, risk, and profit selection. (+19 more)

### Community 47 - "non_dominated_sort"
Cohesion: 0.13
Nodes (26): batch_hamming_min(), crowding_distance(), _crowding_distance_numba(), _crowding_distance_py(), _disable_numba_sort(), _dominates_numba(), _dominates_py(), non_dominated_sort() (+18 more)

### Community 48 - "optuna_search.py"
Cohesion: 0.10
Nodes (29): _active_search_space(), apply_trial_config(), collect_validation_metrics(), compute_score(), _copy_phase1_2_outputs(), main(), objective(), _pearson() (+21 more)

### Community 49 - "._load_and_split_data"
Cohesion: 0.15
Nodes (15): apply_fuzzy_feature_scaling(), fit_fuzzy_feature_scaling(), Any, DataFrame, Train-fitted normalization for ordinal ``ff_*`` fuzzy feature codes. The…, Build a train-only scaling contract for ordinal ``ff_*`` columns., Apply an existing train-fitted scaling contract in place and return *df*., Load train_new.csv and split into train/validation DataFrames. When… (+7 more)

### Community 50 - "prop_settings"
Cohesion: 0.11
Nodes (29): HealthCheck, prop_settings(), Shared Hypothesis settings for property tests. Set PYTEST_LOW_MEMORY=1 to scale…, Hypothesis settings with optional low-memory example scaling., _make_timestamps(), composite, DataFrame, DrawFn (+21 more)

### Community 51 - "_compute_rule_signal_mask"
Cohesion: 0.15
Nodes (13): conditions_cache_key(), get_or_build_rule_mask(), ndarray, condition_cache.py — Cached boolean masks for textual rule conditions. Avoids…, Stable hashable key for a list of condition strings., Return a boolean row mask for *conditions*, using *cache* when provided. The…, _compute_rule_signal_mask(), Build one boolean signal mask (evaluator_v5 parity). Feature conditions are… (+5 more)

### Community 52 - "validate_config"
Cohesion: 0.18
Nodes (19): Validate all high-impact hyperparameter relationships. The function is…, Write the effective configuration snapshot and return its path., validate_config(), write_config_audit_report(), MonkeyPatch, Path, Cross-parameter configuration and evaluator-parity tests., test_audit_report_writes_the_effective_snapshot() (+11 more)

### Community 53 - "build_hybrid_symbol_clusters"
Cohesion: 0.12
Nodes (28): build_hybrid_symbol_clusters(), _corr_embedding_block(), _feature_names_union(), _feature_profile_block(), Any, DataFrame, ndarray, symbol_cluster.py — Per-symbol clustering for Phase 2 island scheduling using… (+20 more)

### Community 54 - "TestWriteSpearmanCorrelationReport"
Cohesion: 0.13
Nodes (10): _make_dataset_with_label(), _make_datasets_by_split(), Feature not in dataset → NaN for that split., Dataset without label_close_288 → NaN for all features on that split., None dataset for a split → NaN for all features on that split., Empty selected_features → CSV with header only., All non-NaN Spearman values must be in [-1.0, 1.0]., Create a dataset DataFrame with feature columns and label_close_288. (+2 more)

### Community 55 - "test_phase2_rule_pool.py"
Cohesion: 0.05
Nodes (30): _crowding_distance(), _dominates(), _non_dominated_sort(), _pareto_sortino_stats(), Return True if solution *a* dominates *b* (all objectives ≤, at least one <)., NSGA-II non-dominated sorting. Parameters ---------- objectives : np.ndarray…, Compute crowding distance for solutions in *front*. Parameters ----------…, Aggregate raw Sortino and return health over the current Pareto front. (+22 more)

### Community 56 - "_make_rule"
Cohesion: 0.16
Nodes (6): _make_rule(), parametrize, Spot-check a variety of valid fuzzy value names., TestWriteConditionValidation, TestWriteRiskParameterValidation, TestWriteSymbolConditionValidation

### Community 57 - "test_reporter.py"
Cohesion: 0.15
Nodes (16): _make_selected_features(), _make_stratified_dataset(), _make_stratified_datasets_by_split(), _make_stratified_logs_by_split(), _make_stratified_trade_log(), Unit tests for gpu_fuzzy_trader.reporting.reporter.Reporter Tests cover: -…, Create a dataset with fuzzy-valued feature columns., Create a trade log with Entry_Index values within dataset bounds. (+8 more)

### Community 58 - "GPUBacktestEngine"
Cohesion: 0.06
Nodes (31): GPUBacktestEngine, CPU engine for rule-set simulation used by Phase 2/RB., Rule-set sim using a mask cache (delegates to the CPU engine)., Evaluate multiple rule sets (parallel CPU; optional mask cache)., RB JAX path: cached mask entry build + parallel CPU equity. Equity results…, Same interface as CPUBacktestEngine for compatibility., JAX-accelerated backtest engine for Phase 2 rule pool generation.…, Return the JAX backend in use ('gpu', 'cpu', or 'tpu'). (+23 more)

### Community 59 - "_make_df"
Cohesion: 0.25
Nodes (8): _make_df(), _make_rule_set(), When no trades are executed, account_ruined must be False., When no trades are executed, total_return_pct must be 0.0., Create a minimal valid rule set dict., Create a minimal DataFrame with all required columns., Returned and saved OOS metrics come from the locked strategy., TestEvaluateStrategy

### Community 60 - "ValueError"
Cohesion: 0.06
Nodes (48): attach_barrier_outcomes(), barrier_column_names(), configured_barrier_pairs(), _first_touch_for_symbol(), _first_touch_for_symbol_numba(), _first_touch_for_symbol_python(), _number_token(), DataFrame (+40 more)

### Community 61 - "._build_engine_for_df"
Cohesion: 0.14
Nodes (11): Build train/val backtest engines., Build the selected Phase 2 backend for the sampled train frame., Return whether this generator should avoid allocating a JAX engine.…, Build an engine on *df* using the same backend selection logic., _minimal_backtest_df(), DataFrame, MonkeyPatch, Phase 2 respects PHASE2_USE_GPU=False without loading JAX GPU engine. (+3 more)

### Community 62 - "_derive_val_sample_seed"
Cohesion: 0.19
Nodes (9): _derive_val_sample_seed(), Derive a deterministic validation sample seed from the training seed. This…, AC: Train and validation sampling use distinct RNG seeds by default., _derive_val_sample_seed returns a value different from the input., Same train seed always produces same val seed., Result is in [0, 2**31) so it is a valid random seed., Rule_Pool_Generator stores distinct _sample_seed and _val_sample_seed., When seed=None, val seed is derived from PHASE2_SEED. (+1 more)

### Community 63 - "TestRulePoolGeneratorRun"
Cohesion: 0.14
Nodes (8): Integration tests using tiny population and generation counts., Verify that Rule_Pool_Generator's persistent self._rng advances across multiple…, After two run_epoch() calls, the internal RNG state must differ from the…, The RNG should produce *different* sequences in two consecutive run_epoch()…, Two generators with different seeds must have different RNG state., Rule_Pool_Generator must initialize self._rng as a Generator., TestRulePoolGeneratorRng, TestRulePoolGeneratorRun

### Community 64 - "test_plateau_state_leak.py"
Cohesion: 0.11
Nodes (21): _make_minimal_gen(), _mock_evolution_state(), _mock_stage_plan(), Regression tests for plateau-state leak fixes (Fixes A + B). Fix A:…, AC-1, AC-3: plateau_streak resets per epoch; reset_plateau=True always., reset_plateau=True is passed on the very first epoch call., reset_plateau=True is also passed on epoch 2 (regression for leak)., Even when PHASE2_ISLAND_TWO_STAGE_ENABLED=False, reset_plateau=True. (+13 more)

### Community 65 - ".prepare_test_data"
Cohesion: 0.36
Nodes (3): Prepare test data using Data_Loader.load_dataset(). Applies the same…, prepare_test_data should return a DataFrame., TestPrepareTestData

### Community 66 - "dashboard.py"
Cohesion: 0.15
Nodes (25): build_dashboard_data(), _direction_data(), _history_rows(), _load_json(), main(), Any, Path, Read-only HTML dashboard for existing pipeline artifacts. The dashboard… (+17 more)

### Community 67 - "DataFrame"
Cohesion: 0.16
Nodes (6): _make_trade_log(), DataFrame, Dataset with only 1 non-NaN paired row → NaN., Rows must be sorted by abs(train_spearman) descending., Create a minimal trade log DataFrame with Equity_After column., TestPlotEquityCurve

### Community 68 - "test_data_loader_properties.py"
Cohesion: 0.15
Nodes (26): dataframe_with_nan_features(), dataframe_with_nan_labels(), _load_from_df(), _make_datetime_series(), composite, DataFrame, DrawFn, given (+18 more)

### Community 69 - ".encode_condition"
Cohesion: 0.18
Nodes (4): See module-level :func:`encode_condition`., TestEncodeConditionBinary, TestEncodeConditionErrors, TestEncodeConditionTernary

### Community 70 - "_m"
Cohesion: 0.13
Nodes (16): evaluator_health_penalty(), Penalty for evaluator_v5 failure modes — higher is worse. Parameters ----------…, _m(), Max positions 15 > 10 → positive penalty., Max positions 8 <= 10 → no additional penalty from positions., Missing ``raw_signal_count`` → treated as 0 → no skip/exec penalty., ``raw_signal_count=0`` → no skip/exec ratio computed → no penalty., Role 'train' same as 'valid' (1.0x). (+8 more)

### Community 71 - "_apply_dynamic_rule"
Cohesion: 0.21
Nodes (4): _apply_dynamic_rule(), Apply one exported text condition using the original threshold logic. Exactly…, Test all threshold branches of _apply_dynamic_rule., TestApplyDynamicRule

### Community 72 - "JointPortfolioEngine"
Cohesion: 0.12
Nodes (12): Compute a non-annualized Sortino Ratio from per-trade returns., _safe_profit_factor(), _sortino_ratio_from_returns(), JointPortfolioEngine, DataFrame, Joint long/short portfolio simulation. Phase 2 and RB score each direction…, Evaluate long and short rule books in one net-position account., _Phase5JSONEncoder (+4 more)

### Community 73 - "run_rb_governor_pipeline"
Cohesion: 0.10
Nodes (33): _assert_capital_budget(), _assert_mandatory_context(), _enforce_capital_budget(), Path, PathLike, Persist an explicit empty strategy and diagnostic report., Build and optimize rb strategies for each direction and write outputs., Normalize rule capital_pct so sum <= RB_MAX_TOTAL_CAPITAL. (+25 more)

### Community 74 - "execution_ok"
Cohesion: 0.15
Nodes (11): execution_ok(), Return ``True`` iff the evaluator would execute this rule set reasonably. A…, Tests for ``execution_ok``., Skip ratio 0.10 <= 0.20 and exec ratio 0.90 >= 0.60 → True., Skip ratio 0.30 > 0.20 → False., Exec ratio 0.50 < 0.60 → False., Missing ``raw_signal_count`` → treated as 0 → False., ``raw_signal_count=0`` → False. (+3 more)

### Community 75 - "TestWriteStrategyEvaluationTable"
Cohesion: 0.21
Nodes (9): _make_eval_rule_set(), _make_metrics_by_split(), _make_trade_logs_by_split(), Create a minimal rule set for evaluation table tests., Create a metrics_by_split dict with all three splits populated., Create a trade_logs_by_split dict with all three splits populated., Sharpe ratio should equal mean(r)/std(r, ddof=1) for a known log., win_rate, mdd_pct etc. should come from metrics_by_split. (+1 more)

### Community 76 - "test_gpu_engine.py"
Cohesion: 0.13
Nodes (13): _build_data_matrix(), _discretize_series(), DataFrame, Series, Discretize a feature series using evaluator_v5's fixed fuzzy bins. Exported…, Build an (N, K) integer matrix of discretized feature values., disable_skip_optimization(), DataFrame (+5 more)

### Community 77 - "test_crash_fix_and_run_logging.py"
Cohesion: 0.06
Nodes (32): _capture_logs(), _make_feature_infos_crash(), _make_train_df_crash(), DataFrame, MonkeyPatch, Unit tests for crash-fix-and-run-logging spec. Covers: - Task 1.1: Smoke tests…, Requirement 3.1 — save_archive is called before _release_resources in run()., Mock save_archive and _release_resources, run Rule_Pool_Generator.run() with a… (+24 more)

### Community 78 - "test_property_19_phase2_static_risk_parameters"
Cohesion: 0.12
Nodes (15): feature_infos_and_train_df(), _make_feature_infos(), _make_train_df(), composite, DataFrame, DrawFn, given, ndarray (+7 more)

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

### Community 84 - "_run_cluster_islands"
Cohesion: 0.07
Nodes (29): clear_global_metrics_cache(), Clear the global eval cache and force GC. Used to free RAM between cluster runs…, evict_cluster_signatures(), Evict JAX compiled signatures for a completed cluster. Removes entries from…, compute_cluster_generation_budgets(), Resolve per-island generation budgets. By default each island receives the full…, Check if an epoch should be skipped due to small remaining budget. Engine…, _run_cluster_islands() (+21 more)

### Community 85 - "_build_target"
Cohesion: 0.17
Nodes (12): _build_target(), Build a direction-specific target signal. Default…, Encoding-aware win class: 2 in asymmetric mode, 1 in legacy mode., Encoding-aware loss class: 0 in either mode., Long: max >= entry*(1+TP/100), min > entry*(1-SL/100) → success., Long: both hit but max_before_min==0 → SL first → failure., Long: both hit but max_before_min==1 → TP first → success., Long: neither TP nor SL hit → failure (or neutral in asymmetric mode). (+4 more)

### Community 86 - "filter_migrants_for_cluster"
Cohesion: 0.13
Nodes (17): filter_migrants_for_cluster(), Accept only migrants that pass deployability on the receiver cluster slice., _make_migrant_dict(), _make_mock_receiver(), Unit tests for migration safety — migrant gate and seed fraction. Acceptance…, Migrant with val_return 2.5% and >=15 val trades should be accepted., Migrant with enough val_return but too few val trades should be rejected., Verify migration default for multi-symbol cluster islands (enabled). (+9 more)

### Community 87 - "test_gpu_engine_properties.py"
Cohesion: 0.16
Nodes (17): _assert_parity(), _make_engines(), _make_parity_df(), parity_scenario_strategy(), composite, DataFrame, DrawFn, given (+9 more)

### Community 88 - "_write_and_reload"
Cohesion: 0.23
Nodes (5): _make_rule_set(), Write rule_set to a temp file and reload the raw JSON., TestWriteHappyPath, TestWriteTruncation, _write_and_reload()

### Community 89 - "TestInitPopulation"
Cohesion: 0.15
Nodes (11): _chromosome_with_min_active(), _is_all_inactive_sparse(), _pop_contains_dense_seed(), ndarray, f3 profit_factor branch mirrors win_rate train+val blend. Covers three…, With dont_care_prob=0.5, inactive sparse rows should appear., With dont_care_prob=0, no fully inactive sparse rows should appear., True when every sparse slot is inactive (feat_idx == -1). (+3 more)

### Community 90 - "loader.py"
Cohesion: 0.19
Nodes (12): _ensure_labels(), load_dataset(), DataFrame, data/loader.py — Data_Loader Stateless CSV loading with full preparation…, Recompute the LWC pullback-reversal triggers and compare row-by-row. A stale or…, Keep supplied labels or derive all labels from raw OHLCV columns. The original…, Load a CSV dataset with full preparation pipeline: 1. Read CSV with comma…, Module-level wrapper around ``Data_Loader.load_dataset``. (+4 more)

### Community 91 - "set_purged_wf_reference_rows"
Cohesion: 0.18
Nodes (7): Store full train_new.csv row count after loader prep (split time)., set_purged_wf_reference_rows(), holdout_mode(), purged_mode(), fixture, Tests for purged-WF trade-floor scaling helpers., TestScaleTradeFloor

### Community 92 - "downcast_numeric_df"
Cohesion: 0.20
Nodes (15): downcast_numeric_df(), prune_train_columns(), DataFrame, df_slim.py — Reduce DataFrame memory for backtest engines. Keeps only columns…, Downcast label and feature columns to float32; symbol to category (inplace)., Return a copy with only meta, label, and requested feature columns. Parameters…, Drop unused feature columns from the full training split after Phase 1., slim_backtest_df() (+7 more)

### Community 93 - "Encoder"
Cohesion: 0.10
Nodes (20): ConfigurationError, decode_chromosome(), encode_condition(), Encoder, get_dont_care(), Exception, ndarray, encoder.py — Encoder Maps gene integer values to fuzzy value names, formats… (+12 more)

### Community 94 - "ValidationError"
Cohesion: 0.05
Nodes (60): _rule_symbols_for_allocation(), normalize_symbol_value(), parse_symbol_condition(), symbol_conditions.py — Symbol filter parsing (evaluator_v5 parity). Feature…, Normalize symbol values so strategy conditions such as: "symbol is 1" "symbol…, Parse optional symbol filters. Supported formats: "symbol is 1" "symbol IS 1"…, Split normal feature conditions from optional symbol filters. Feature…, split_feature_and_symbol_conditions() (+52 more)

### Community 95 - "TestPermissionGatedPullback"
Cohesion: 0.33
Nodes (3): Default: opposite LWC print counts even if permission was off., v7: Range consolidation in lookback is enough to arm the trigger., TestPermissionGatedPullback

### Community 96 - "test_phase2_gpu_throughput.py"
Cohesion: 0.16
Nodes (15): CPU and GPU backtest engine sub-package., get_gpu_backtest_engine_class(), jax_gpu_backtest_available(), Any, Detect whether JAX / GPUBacktestEngine can be loaded on this host. JAX can fail…, Return ``GPUBacktestEngine`` if ``gpu_engine`` imports cleanly; else ``None``., True when ``get_gpu_backtest_engine_class()`` would succeed., _gpu_available() (+7 more)

### Community 97 - ".decode_chromosome"
Cohesion: 0.26
Nodes (4): See module-level :func:`decode_chromosome`., All 10 signed values decode correctly via decode_chromosome., Chromosome with numpy int32/int64 values should work., TestDecodeChromosome

### Community 98 - ".load_strategies"
Cohesion: 0.18
Nodes (6): Load long.json and short.json via Output_Writer.load_and_validate(). Parameters…, Return only non-empty RB strategies accepted for deployment., Write a valid rule set JSON to path., Override module-level path dicts and return originals (for standalone tests)., TestLoadStrategies, _write_rule_set()

### Community 99 - "gate_positive_good"
Cohesion: 0.24
Nodes (18): _positive_good_reject_reasons(), Human-readable reasons why ``_is_positive_good`` failed (diagnostics)., _finite_number(), gate_positive_good(), _metric_int(), positive_good_reject_reasons(), PositiveGoodThresholds, Shared metric gates used by Phase 2 admission and RB Governor. The gate is… (+10 more)

### Community 100 - "phase2_island_scheduler.py"
Cohesion: 0.12
Nodes (26): filter_df_to_symbols(), phase2_history_path(), phase2_pool_path(), DataFrame, Return rows for the given symbols; raises if column missing or no rows., Resolve Phase 2 pool path., Resolve Phase 2 history path., persist_symbol_clusters() (+18 more)

### Community 101 - ".save_archive"
Cohesion: 0.11
Nodes (18): _archive_feature_signature(), _downsample_chronological(), _evaluate_rule_on_window(), _monthly_admission_source_df(), DataFrame, Return the ordered feature signature used to validate archive reuse., Read JSON from *path* and return None when the file cannot be loaded., Validate the archive JSON structure and feature compatibility. (+10 more)

### Community 102 - "_remove_low_dispersion"
Cohesion: 0.17
Nodes (10): Remove features where more than `threshold` fraction of values are identical.…, _remove_low_dispersion(), DataFrame, Exactly 95% identical → NOT > 0.95 → keep., 96% identical → > 0.95 → remove., Two features with corr > 0.95 → keep higher-scored one., Two uncorrelated features → both kept., Features in different modes are not compared for redundancy. (+2 more)

### Community 103 - "TestEquityCurvePlots"
Cohesion: 0.13
Nodes (13): Build a flat list of per-symbol metric dicts for CSV output. Uses the…, DataFrame, Verify plot_equity_curve is called for all three splits and handles empty logs., Helper: run OOS_Evaluator.run() with mocked internals and capture plot calls., plot_equity_curve must be called with 'train', 'validation', and 'test'., plot_equity_curve must be called exactly three times (one per split)., Empty train trade log must not raise an exception., Empty validation trade log must not raise an exception. (+5 more)

### Community 104 - "test_crash_fix_properties.py"
Cohesion: 0.10
Nodes (26): _append_xla_flag(), configure_jax_env(), _cuda_package_root(), Path, JAX/XLA runtime configuration — call before the first ``import jax``., Find a pip-installed CUDA toolkit root, if one is available., Append an XLA flag once, preserving explicit user configuration., Configure JAX/XLA runtime for predictable desktop-friendly GPU usage. -… (+18 more)

### Community 105 - "TestPlotDistributionAndEquity"
Cohesion: 0.16
Nodes (6): _make_dist_logs_by_split(), _make_dist_trade_log(), Create a trade log with all columns needed for distribution/equity plots., Create trade_logs_by_split dict with all three splits populated., A trade log with a single trade should not raise., TestPlotDistributionAndEquity

### Community 106 - "baselines.py"
Cohesion: 0.28
Nodes (15): _compact(), _drop_one_condition_variants(), _equal_weight(), _evaluate(), evaluate_baselines(), _feature_shuffle(), _fixed_exit(), Any (+7 more)

### Community 107 - "phase2_rule_pool.py"
Cohesion: 0.04
Nodes (76): extract_deployable_migrants(), Return elite deployable-preview entries suitable for guarded migration., build_feature_sampling_probs(), Softmax Phase 1 scores with optional uniform floor (length K)., _exchange_migrants_between_islands(), _migrant_to_metrics(), Backtest one migrant chromosome on receiver cluster engines., Perform a guarded, order-independent migration exchange. Islands are processed… (+68 more)

### Community 108 - "_jax_compute_trade_outcomes"
Cohesion: 0.24
Nodes (5): _jax_compute_trade_outcomes(), JAX-jitted vectorized trade outcome computation., Vectorized trade outcome computation for all rows. Mirrors…, Multiple rows processed simultaneously., TestComputeTradeOutcomes

### Community 109 - "_should_post_restart_early_stop_phase2"
Cohesion: 0.29
Nodes (9): Break the epoch when a plateau restart yields no improvement. Independent of…, _should_post_restart_early_stop_phase2(), Unit tests for post-restart no-improvement early stop (Phase 2 runtime)., test_global_disabled_no_stop(), test_global_uses_global_knobs(), test_island_disabled_no_stop(), test_island_streak_at_patience_stops(), test_island_streak_below_patience_no_stop() (+1 more)

### Community 110 - "test_data_splitter_properties.py"
Cohesion: 0.18
Nodes (14): _make_timestamps(), composite, DataFrame, DrawFn, given, Timestamp, Property-based tests for gpu_fuzzy_trader.data.splitter.Data_Splitter…, Patch TRAIN_70_PATH / VALIDATION_30_PATH to tmp_path and run split. (+6 more)

### Community 111 - "compute_phase2_objectives_from_metrics"
Cohesion: 0.06
Nodes (37): Stop burning gens when the feasible set is empty and restarts are spent. Sparse…, _should_abort_zero_deployable_collapse(), compute_phase2_objectives_from_metrics(), _diversity_penalty_blended(), _hamming_distance(), _phenotype_bucket_key(), Hamming distance between two chromosomes (active pairs when sparse)., Discretise objective-relevant metrics for behavioral diversity. (+29 more)

### Community 112 - "_validate_pool_schema"
Cohesion: 0.36
Nodes (3): Validate the structure of a loaded pool JSON. Raises ValueError if the schema…, _validate_pool_schema(), TestValidatePoolSchema

### Community 113 - ".run"
Cohesion: 0.11
Nodes (11): DataFrame, Run out-of-sample evaluation. Parameters ---------- allowed_directions :…, Load prepared train, validation, and test datasets., Load and validate a strictly newer, untouched forward period., Load selected features for a direction when available., Remove only known Phase 5 artifacts from the active report root., Evaluate a single strategy on the test DataFrame. Returns ------- metrics :…, Return an explicit, non-success result for a failed split. (+3 more)

### Community 114 - "TestHammingThresholdAutoScale"
Cohesion: 0.15
Nodes (10): Verify the max(3, k_active // 5) formula. The formula is applied in…, Replicate the auto-scaling formula., k_active=0 → threshold = max(3, 0//5) = 3., k_active=5 → threshold = max(3, 5//5=1) = 3., k_active=15 → threshold = max(3, 15//5=3) = 3., k_active=20 → threshold = max(3, 20//5=4) = 4., k_active=50 → threshold = max(3, 50//5=10) = 10., k_active=100 → threshold = max(3, 100//5=20) = 20. (+2 more)

### Community 115 - "TestValLeakGate"
Cohesion: 0.20
Nodes (10): C6: Val-derived penalties must be gated behind JOINT_TRAIN_VAL or…, Return standard monkeypatching for clean baseline metrics., Apply base settings with optional overrides., Metrics that trigger no train-side penalties., Val metrics that WOULD trigger penalties if the gate were open., When both JOINT_TRAIN_VAL and VAL_IN_FITNESS_PENALTY are False, val-derived…, Bad val must not set feasibility_violation when gate is closed., When VAL_IN_FITNESS_PENALTY=True, val-derived penalties DO enter objectives. (+2 more)

### Community 116 - "_build_entries_from_rule_set"
Cohesion: 0.09
Nodes (17): _build_entries_from_rule_set(), Sort entries for v5 capital allocation (timestamp, rule, symbol, row)., Priority-based rule assignment: first matching rule wins per row. Mirrors…, _sort_entries_by_allocation_priority(), DataFrame, Per-symbol metrics should reflect actual trade distribution., Winning trades should produce positive net_pnl per symbol., Row matching both rules should be assigned to rule 1 only. (+9 more)

### Community 117 - "_jax_compute_rule_signals"
Cohesion: 0.17
Nodes (9): _jax_compute_rule_signals(), JAX-jitted vectorized rule matching (single chromosome)., Vectorized rule matching: returns (N,) boolean mask of matching rows., All rows match when chromosome equals data_matrix values., No rows match when chromosome differs from data_matrix., Only rows where all active conditions match., Columns where chromosome == dont_care are ignored., All dont_care chromosome matches every row. (+1 more)

### Community 118 - "TestParetoCollapseWarningGate"
Cohesion: 0.15
Nodes (10): _FakeEngine, AC 4: The default value of the config flag is 5., AC 5: min_pareto_size=0 → warning fires even with small pareto (size=4)., The log message includes 'pareto_size=N' suffix., Fake engine that returns metrics producing a tradeoff between f1 (-sortino) and…, AC 1–5: warning gated on len(pareto_indices) >= config threshold., Run 2-gen evolution and return count of 'Pareto collapse risk' warnings., AC 2: pareto_size=4 < min_pareto_size=5 → no warning fires. (+2 more)

### Community 119 - "_make_walk_forward_fold_engines"
Cohesion: 0.09
Nodes (24): evaluate_strategy_governor(), _load_scoring_frames(), _make_walk_forward_fold_engines(), _prepare_scoring_frame(), Prepare a scoring frame with evaluator ordering., Return prepared training and validation frames., Split val_selection into n_splits chronological folds + optional tail holdout.…, Evaluate a strategy on training and validation frames. (+16 more)

### Community 120 - "_compute_stability"
Cohesion: 0.17
Nodes (9): _compute_stability(), Compute cross-symbol stability score. stability = 1 - (std / mean) If mean is 0…, All symbols have same score → std=0 → stability=1., Mean is 0 → stability=0., Single symbol with positive score → stability=1., Single symbol with zero score → stability=0., Very high variance → stability could be negative → clipped to 0., Stability should always be in [0, 1]. (+1 more)

### Community 121 - "passes_pool_admission_gate"
Cohesion: 0.06
Nodes (24): passes_pool_admission_gate(), Hard gate for Phase 2 pool/archive on merged holdout metrics. When…, MonkeyPatch, train=15%/val=4% (3.75× ratio, gap=11pp < 16pp) is REJECTED by the ratio gate…, train=15%/val=10% (1.5× ratio) is ADMITTED., train/val ≈2.5× is ADMITTED when under OVERFIT_RATIO_FLOOR and gap gate., With PHASE2_OVERFIT_RATIO_FLOOR=0.0, the ratio gate is disabled and the high-…, fixture (+16 more)

### Community 122 - "CPUBacktestEngine"
Cohesion: 0.05
Nodes (64): CPUBacktestEngine, CPU backtest engine that exactly mirrors evaluator_v5.ipynb's…, _mask_for(), _monthly_selection_certificate(), _profit_amp_certificate(), _profit_amp_evaluate_candidate(), _profit_amp_monthly_summary(), _profit_amp_objective() (+56 more)

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
Cohesion: 0.16
Nodes (10): _mutual_info_discrete_mask(), Return the set of feature names that PASS both stationarity checks. A feature…, Cap long/short feature overlap and backfill each direction to top_k features., Per-column ``discrete_features`` flags for ``mutual_info_classif``. Only…, _reduce_overlap(), _stationarity_filter(), Unit tests for gpu_fuzzy_trader.features.selector.Feature_Selector Tests cover:…, TestMutualInfoDiscreteMask (+2 more)

### Community 127 - "TestEndToEndRotation"
Cohesion: 0.18
Nodes (7): fixture, Integration-style tests with a mocked Rule_Pool_Generator., Patch config for rotation and create a generator with minimal setup., When rotation is enabled, _cached_scoped_train_df is stored., After resample_train_for_epoch, the cached slim train changes., Same epoch_idx produces identical cached slim train., TestEndToEndRotation

### Community 128 - "stratification_scenario_strategy"
Cohesion: 0.24
Nodes (15): dataset_with_features_strategy(), metrics_strategy(), composite, DataFrame, DrawFn, Generate a trade log DataFrame with 2–50 rows. Parameters ---------- n_rules:…, Generate a metrics dict with reasonable float values. Keys: win_rate,…, Generate a dataset DataFrame with 10–100 rows. Parameters ----------… (+7 more)

### Community 129 - "TestSpearmanSignConsistency"
Cohesion: 0.18
Nodes (9): ndarray, AC3: Train all positive, tiny |val_rho| < min_abs_corr → feature still kept., AC4: val_df=None → pre-task-8 behavior (mixed train signs still blacklisted)., AC5: val_df without label_close_288 column → val check skipped (no crash)., Tests for _check_spearman_sign_consistency, including the val_df check., Build a minimal DataFrame without 'symbol' to avoid symbol-based folding., AC1: Train all positive, val negative → feature blacklisted., AC2: Train all positive, val positive → feature still kept. (+1 more)

### Community 130 - "run_pipeline.py"
Cohesion: 0.09
Nodes (30): context_coverage_for_direction(), context_coverage_report(), context_floor_failures(), Any, DataFrame, Shared diagnostics for the mandatory direction-specific context contract., Return coverage diagnostics for every named frame and direction., Return mathematically impossible trade-floor failures for coverage. (+22 more)

### Community 131 - "log_memory_rss"
Cohesion: 0.22
Nodes (10): log_memory_rss(), Optional process memory logging (set LOG_MEMORY=1)., Log current process RSS in MiB when LOG_MEMORY=1., Force GC and clear JAX compilation caches between Phase 2 directions. Releases:…, release_phase2_resources(), Release GPU engines and slim in-memory data between island epochs., Tests for optional memory logging helpers., test_log_memory_rss_noop_without_env() (+2 more)

### Community 132 - "test_evolution_convergence.py"
Cohesion: 0.19
Nodes (9): _normalize_for_association(), Rank-based normalization (robust to outliers like trade_penalty=50)., Tests for H5/M4/M5 evolution convergence behaviors. Covers: - HoF trimming at…, Verify rank normalization does not crash on degenerate inputs., All-equal objective values should produce valid normalised output with no NaN,…, Single-row input should not crash., Two objectives, all-equal values, should produce valid output. After rank…, Mixed values (not all equal) should still work. (+1 more)

### Community 133 - "_apply_colab_gpu_defaults"
Cohesion: 0.33
Nodes (6): _apply_colab_gpu_defaults(), is_colab_runtime(), True when running on Google Colab (/content runtime)., Colab T4 optimization for Phase 2 runs., MonkeyPatch, test_colab_defaults_apply_when_content_exists()

### Community 134 - "Data_Splitter"
Cohesion: 0.14
Nodes (19): Data_Splitter, load_cached_split_if_fresh(), Load cached split parquets when they are newer than the source CSV. Validates…, Chronological train/validation splitter., Module-level wrapper around ``Data_Splitter.split_and_persist``., split_and_persist(), _make_df(), _make_timestamps() (+11 more)

### Community 135 - "test_evaluator_clean_writer.py"
Cohesion: 0.18
Nodes (11): _make_rule(), minimal_strategy(), fixture, Unit tests for ``gpu_fuzzy_trader.output.writer.write_evaluator_clean``. Tests…, Tests that ``Output_Writer.write`` also produces the evaluator-clean file., Output_Writer.write produces both the main file and the evaluator-clean file., Long direction also produces the correct clean file., Strategy with only the two required keys (no extras). (+3 more)

### Community 136 - "OOS_Evaluator"
Cohesion: 0.27
Nodes (4): OOS_Evaluator, Out-of-sample evaluator for Phase 5. Loads the final long/short strategies,…, TestOOSEvaluatorInit, TestSaveReport

### Community 137 - "TestF4ReturnConcentration"
Cohesion: 0.14
Nodes (8): Tests for the f4 return-concentration objective (Task 2)., A rule with 1 trade = +60 and 79 trades avg -0.5% receives f4 ≈ 1.0., A rule with uniform +1% across 80 trades receives f4 ≈ 0.0125., Joint concentration must retain the outlier-heavy split., When PHASE2_F4_ENABLED = False, objectives.shape == (3,) and no f4 in metrics., Low-support rules receive worst-case concentration evidence., When PHASE2_F4_ENABLED is deleted from config (missing attr), both the f4…, TestF4ReturnConcentration

### Community 138 - "TestDecoupledObjectives"
Cohesion: 0.17
Nodes (7): Task 3 tests: penalties are not identically added to all objectives., The three objectives should respond differently to the same metrics, proving…, When the trade floor is triggered, only f2 gets the trade_penalty. f1 and f3…, With USE_TOTAL_RETURN_OBJ + JOINT_TRAIN_VAL, f3 uses robust_return_pct., PF floor adds to support_penalty even after decoupling. This ensures the…, Spec scenario: pf=1.10, EVOLUTION=1.05 (new) vs 1.15 (old); the new floor…, TestDecoupledObjectives

### Community 139 - "context_contract_digest"
Cohesion: 0.24
Nodes (10): context_contract_digest(), Return a stable hash of the static contract and fitted enrichment., feature_conditions_only(), phase2_rule_id(), Any, Stable provenance identifiers for Phase 2 feature rules. The RB Governor may…, Return normalized non-symbol, non-context conditions in order. Symbol scope and…, Hash the immutable Phase 2 feature logic and its discovery scope. (+2 more)

### Community 140 - "TestMigrationSeedFraction"
Cohesion: 0.20
Nodes (6): Verify that migrant injection uses PHASE2_MIGRATION_SEED_FRACTION, not ARCHIVE., PHASE2_MIGRATION_SEED_FRACTION defaults to 0.10., Ensure the migration fraction is not the same as archive fraction., Simulate the run_epoch migrant path and assert local_cap uses migration…, Simulate the new run_epoch logic: migrant entries are capped by migration…, TestMigrationSeedFraction

### Community 141 - "TestEdgeCases"
Cohesion: 0.20
Nodes (6): DataFrame, Single row: floor(1 * train_frac) = 0, and 288-bar embargo consumes it., Small symbol where 288-bar embargo leaves no validation rows., An empty input DataFrame should produce empty train and validation., For large N, train/total should be very close to HOLDOUT_TRAIN_FRACTION., TestEdgeCases

### Community 142 - "._prune_splits_after_phase1"
Cohesion: 0.25
Nodes (4): Selected fuzzy features for Phase 2., Drop unused feature columns from train/val splits to reduce RAM., Drop unused feature columns from train split (legacy single-split API)., TestPruneSplitsAfterPhase1

### Community 143 - "trade_support_penalty"
Cohesion: 0.38
Nodes (4): Backward-compatible wrapper returning penalty only., trade_support_penalty(), Between the hard-reject floor and the soft threshold the penalty is graduated., TestTradeSupportPenalty

### Community 144 - "TestRefreshObjectivesOnResumeGate"
Cohesion: 0.23
Nodes (7): Task-10: gate cache refresh on PHASE2_PER_EPOCH_WINDOW_ROTATION. Verifies the…, Evaluate the gate expression as it appears in the source., PHASE2_PER_EPOCH_WINDOW_ROTATION=True (default): refresh = not first_epoch →…, PHASE2_PER_EPOCH_WINDOW_ROTATION=True (default): refresh = not first_epoch →…, PHASE2_PER_EPOCH_WINDOW_ROTATION=False (legacy): refresh is False even on…, PHASE2_PER_EPOCH_WINDOW_ROTATION=False (legacy): refresh is False on first…, TestRefreshObjectivesOnResumeGate

### Community 145 - "conftest.py"
Cohesion: 0.22
Nodes (10): Config, FixtureRequest, Item, _close_matplotlib_figures(), _low_memory_cleanup(), fixture, pytest_collection_modifyitems(), pytest_configure() (+2 more)

### Community 146 - "TestIslandAwareTradeFloor"
Cohesion: 0.25
Nodes (5): Tests for island-aware hard reject floor and config constant usage., When island_hyperparams.min_trade_pool_floor=15 and executed=20, no hard-reject…, When the trade floor is triggered, trade_penalty should equal…, When island_hyperparams is None, trade_floor falls back to…, TestIslandAwareTradeFloor

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
Nodes (9): Return the sidecar that binds a reusable pool to its inputs., Hash an artifact without loading it all into RAM., Load existing pool if valid, return None if missing., Atomically bind the current pool bytes to a Phase 2 input identity., Remove a stale direction cache before a fresh Phase 2 run., Return a schema-valid pool proven to match this run's inputs. Bare historical…, _resolve_pool_identity_path(), _sha256_path() (+1 more)

### Community 152 - "BFS and DFS Graph Traversal"
Cohesion: 0.20
Nodes (10): CLAUDE.md Graphify Integration, Constrained Query Expansion, BFS and DFS Graph Traversal, Graph Work Memory, Existing Graph Fast Path, CLAUDE.md Graphify Integration, Constrained Query Expansion, BFS and DFS Graph Traversal (+2 more)

### Community 153 - "TestPlateauDiversityRestart"
Cohesion: 0.33
Nodes (4): Reinitialised slots have objectives=np.inf and metrics_cache={}., Even with large Pareto front, at most 10 elite are preserved., Direct unit tests for the _plateau_diversity_restart helper., TestPlateauDiversityRestart

### Community 155 - "environmental_selection_nsga2"
Cohesion: 0.38
Nodes (4): environmental_selection_nsga2(), Canonical NSGA-II truncation on a 2N merged population., Unit tests for canonical NSGA-II environmental selection., TestEnvironmentalSelectionNsga2

### Community 156 - "_make_csv"
Cohesion: 0.33
Nodes (5): _make_csv(), DataFrame, Module-level function should produce same result as class method., Build a CSV string from a list of row dicts., TestModuleLevelFunction

### Community 157 - "Data_Loader"
Cohesion: 0.10
Nodes (14): Data_Loader, Stateless data loader for the GPU-Fuzzy Trading Pipeline., _enriched(), Regression: thresholds must never see validation-period rows., _raw_tape(), TestContextContract, TestContextIdentity, TestIncompleteBoundaryBars (+6 more)

### Community 158 - "TestGlobalMetricsCacheClearing"
Cohesion: 0.27
Nodes (6): When seeded_keys is empty, nothing is removed., When all keys are seeded, cache becomes empty., Verify that only seeded keys are removed from the global cache., Create a deterministic chromosome key., Cache entries matching seeded_keys are removed; non-matching survive., TestGlobalMetricsCacheClearing

### Community 159 - "TestRobustReturnObjective"
Cohesion: 0.29
Nodes (4): f3 uses robust return = min(train_return, val_return) when…, Blind-spot regression: overfit_gap_penalty must fire when val_ret <= 0., Direct assertion that penalty is strictly positive when val_ret <= 0., TestRobustReturnObjective

### Community 160 - "ConfigError"
Cohesion: 0.60
Nodes (5): _config_check(), ConfigError, _finite_config_number(), Raised when a configuration violates a cross-parameter contract., _validate_config_grid()

### Community 161 - "TestF3PathResolution"
Cohesion: 0.40
Nodes (4): parametrize, Parametrized tests for f3 path resolution (Task 5: audit finding #5). Verifies…, Verify the correct f3 formula runs for each (USE_TOTAL_RETURN_OBJ,…, TestF3PathResolution

### Community 163 - "phase2_should_enrich_symbol_metrics"
Cohesion: 0.67
Nodes (3): phase2_should_enrich_symbol_metrics(), Return True when GPU batch eval should run a follow-up CPU enrichment pass.…, test_phase2_should_enrich_without_symbol_scope()

### Community 165 - "TestHealthPenaltyWiredIntoRB"
Cohesion: 0.33
Nodes (4): Verify that evaluator health penalty is subtracted from combo score., When evaluator_health_penalty returns > 0, the combo score is lower., When evaluator health is fine, no penalty is applied., TestHealthPenaltyWiredIntoRB

### Community 172 - "TestHallOfFameTrim"
Cohesion: 0.25
Nodes (5): Verify that hall_of_fame is capped to PHASE2_HOF_EPOCH_CARRYOVER=10., HoF with >10 entries is trimmed to exactly 10 oldest entries., HoF with <=10 entries is left unchanged., Empty HoF does not crash during trim logic., TestHallOfFameTrim

### Community 173 - "phase2_support.py"
Cohesion: 0.05
Nodes (53): effective_min_trade_pool_floor(), effective_pool_min_val_trades(), IslandHyperparams, Resolved Phase 2 knobs for cluster or orphan slices., _pool_entry_passes_admission(), Check stored train/val metrics on a pool JSON entry., compute_support_penalty_and_specialist(), deployability_rank_score() (+45 more)

### Community 175 - "test_evaluator_health.py"
Cohesion: 0.15
Nodes (9): evaluator_health.py — Pure functions for evaluator-failure-mode awareness.…, Read a numeric metric, returning *default* for missing / None / NaN / Inf., Read an integer metric safely., _safe_float(), _safe_int(), Scoring helpers shared across pipeline phases. Re-exports…, Unit tests for ``evaluator_health_penalty`` and ``execution_ok`` (Task 4).…, Both public functions are importable from the module. (+1 more)

### Community 176 - "test_gpu_engine_import_does_not_crash_on_jax_failure"
Cohesion: 0.33
Nodes (6): parametrize, Tests for the lazy-JAX failure path in gpu_engine.py. These tests verify that…, Verify importing gpu_engine.py handles various JAX failure modes gracefully., Build a subprocess script that simulates JAX import failure., _subprocess_script(), test_gpu_engine_import_does_not_crash_on_jax_failure()

### Community 177 - "test_jax_compat.py"
Cohesion: 0.29
Nodes (5): parametrize, Tests for JAX / GPU engine availability detection., Package init must not crash when gpu_engine import fails., test_cpu_engine_import_without_jax(), test_get_gpu_backtest_engine_class_returns_none_on_import_error()

### Community 178 - "TestExecutionHealthInGate"
Cohesion: 0.25
Nodes (5): Verify that ``gate_positive_good`` calls ``execution_ok`` when flagged., When require_execution_health=True and execution_ok fails, gate returns False., When require_execution_health=True and execution_ok passes, gate still returns…, When require_execution_health=False, gate ignores raw_signal_count., TestExecutionHealthInGate

### Community 184 - "TestEvalCvFoldReturns"
Cohesion: 0.33
Nodes (4): Verify the helper handles None / empty fold_engines safely., fold_engines=None returns None without crashing., fold_engines=[] returns None without crashing., TestEvalCvFoldReturns

### Community 189 - "opencode.json"
Cohesion: 0.50
Nodes (3): plugin, $schema, .opencode/plugins/graphify.js

### Community 190 - "test_rb_governor_tail_holdout.py"
Cohesion: 0.19
Nodes (13): _passes_tail_selection_gate(), Validate a trial ruleset on the reserved chronological validation tail. The…, _MockEngine, Unit tests for RB Governor tail-holdout path in risk grid. Covers: -…, Verify _optimize_risk with tail_holdout_engine adds tail fields to final…, When tail_holdout_engine is provided, the final history entry contains…, When tail_holdout_engine=None, NO tail fields in history., Composition may use the reserved validation tail, never Phase 5 data. (+5 more)

## Knowledge Gaps
- **31 isolated node(s):** `$schema`, `.opencode/plugins/graphify.js`, `URL Ingestion`, `Folder Watcher`, `Graphify MCP Server` (+26 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **9 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `CPUBacktestEngine` connect `CPUBacktestEngine` to `Rule_Pool_Generator`, `._simulate_rule_set_entries`, `test_cpu_engine.py`, `_score_metrics`, `OOS_Evaluator`, `_make_df`, `trade_support_penalty`, `test_cpu_engine_properties.py`, `TestCausalPublicationTiming`, `CandidateRecord`, `nested_walk_forward.py`, `Data_Loader`, `TestTriggerPerSymbol`, `rb_governor.py`, `gpu_engine.py`, `test_certificate_first_selection.py`, `test_jax_compat.py`, `_compute_rule_signal_mask`, `TestEvalCvFoldReturns`, `GPUBacktestEngine`, `ValueError`, `._build_engine_for_df`, `test_rb_governor_tail_holdout.py`, `_apply_dynamic_rule`, `JointPortfolioEngine`, `run_rb_governor_pipeline`, `test_gpu_engine.py`, `test_gpu_engine_properties.py`, `ValidationError`, `TestPermissionGatedPullback`, `test_phase2_gpu_throughput.py`, `phase2_island_scheduler.py`, `.save_archive`, `baselines.py`, `phase2_rule_pool.py`, `_jax_compute_trade_outcomes`, `.run`, `_build_entries_from_rule_set`, `_jax_compute_rule_signals`, `_make_walk_forward_fold_engines`, `._engine`?**
  _High betweenness centrality (0.147) - this node is a cross-community bridge._
- **Why does `Reporter` connect `Reporter` to `Rule_Pool_Generator`, `DataFrame`, `TestEquityCurveDateAxis`, `JointPortfolioEngine`, `OOS_Evaluator`, `TestPlotDistributionAndEquity`, `phase2_rule_pool.py`, `._ensure_dir`, `TestWriteStrategyEvaluationTable`, `.run`, `prop_settings`, `TestPlotPerRuleBreakdown`, `TestWriteSpearmanCorrelationReport`, `test_reporter.py`?**
  _High betweenness centrality (0.098) - this node is a cross-community bridge._
- **Why does `Rule_Pool_Generator` connect `Rule_Pool_Generator` to `run_pipeline.py`, `log_memory_rss`, `.run`, `TestSymbolGeneBias`, `TestF4ReturnConcentration`, `_get_dont_cares`, `TestDecoupledObjectives`, `Pipeline_Orchestrator`, `_apply_monthly_admission_gate`, `TestRefreshObjectivesOnResumeGate`, `_make_train_df`, `TestIslandAwareTradeFloor`, `TestDeferredWarmup`, `.skip_if_valid`, `Reporter`, `TestRobustReturnObjective`, `TestF3PathResolution`, `test_phase2_window_rotation.py`, `_derive_island_seed`, `test_phase2_rule_pool.py`, `._build_engine_for_df`, `_derive_val_sample_seed`, `TestRulePoolGeneratorRun`, `test_plateau_state_leak.py`, `test_crash_fix_and_run_logging.py`, `test_property_19_phase2_static_risk_parameters`, `_run_cluster_islands`, `filter_migrants_for_cluster`, `TestInitPopulation`, `downcast_numeric_df`, `phase2_island_scheduler.py`, `.save_archive`, `phase2_rule_pool.py`, `compute_phase2_objectives_from_metrics`, `_validate_pool_schema`, `TestValLeakGate`, `CPUBacktestEngine`, `TestEndToEndRotation`?**
  _High betweenness centrality (0.086) - this node is a cross-community bridge._
- **Are the 65 inferred relationships involving `CPUBacktestEngine` (e.g. with `GPUBacktestEngine` and `JointPortfolioEngine`) actually correct?**
  _`CPUBacktestEngine` has 65 INFERRED edges - model-reasoned connections that need verification._
- **Are the 15 inferred relationships involving `Reporter` (e.g. with `CvFoldValEvaluator` and `Rule_Pool_Generator`) actually correct?**
  _`Reporter` has 15 INFERRED edges - model-reasoned connections that need verification._
- **Are the 62 inferred relationships involving `Rule_Pool_Generator` (e.g. with `CPUBacktestEngine` and `Reporter`) actually correct?**
  _`Rule_Pool_Generator` has 62 INFERRED edges - model-reasoned connections that need verification._
- **Are the 31 inferred relationships involving `Output_Writer` (e.g. with `OOS_Evaluator` and `_Phase5JSONEncoder`) actually correct?**
  _`Output_Writer` has 31 INFERRED edges - model-reasoned connections that need verification._