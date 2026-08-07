# Graph Report - trading_platform  (2026-08-07)

## Corpus Check
- 188 files · ~270,326 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 4852 nodes · 11000 edges · 187 communities (176 shown, 11 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 563 edges (avg confidence: 0.57)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `610883f0`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- evox_runner.py
- .run
- rb_governor.py
- _make_engine
- .run
- test_evox_runner.py
- _symbol_specialized_variants
- _score_metrics
- Rule_Pool_Generator
- _make_df
- phase2_rule_pool.py
- run_phase2_evolution
- Output_Writer
- Data_Loader
- Pipeline_Orchestrator
- _apply_monthly_admission_gate
- test_feature_detector_properties.py
- _make_train_df
- _split
- test_cpu_engine_properties.py
- detect_feature_mode
- selector.py
- Feature_Selector
- splitter.py
- Reporter
- test_encoder_properties.py
- write_evaluator_clean
- CandidateRecord
- nested_walk_forward.py
- environmental_selection_nsga2
- maybe_log_generation
- Hybrid CPU and GPU execution policy
- TestSelectDiverseSubset
- test_rb_min_symbols.py
- Graphify Pipeline
- context_permission_column
- config.py
- TestEquityCurveDateAxis
- test_phase2_window_rotation.py
- research_integrity.py
- _loader_from_rows
- _build_entries_from_rule_set
- gpu_engine.py
- _gpu_runtime.py
- ._ensure_dir
- run_pipeline.py
- test_certificate_first_selection.py
- non_dominated_sort
- optuna_search.py
- test_phase2_rule_pool.py
- test_property_19_phase2_static_risk_parameters
- _compute_rule_signal_mask
- validate_config
- build_hybrid_symbol_clusters
- _make_rule
- trend_context.py
- Feature_Detector
- test_reporter.py
- GPUBacktestEngine
- .simulate_rule_set
- ndarray
- ._build_engine_for_df
- _make_df
- TestRulePoolGeneratorRun
- test_plateau_state_leak.py
- passes_pool_admission_gate
- dashboard.py
- TestWriteSpearmanCorrelationReport
- test_data_loader_properties.py
- ValueError
- _m
- _apply_dynamic_rule
- cpu_engine.py
- test_rb_fail_closed.py
- execution_ok
- TestWriteStrategyEvaluationTable
- test_gpu_engine.py
- test_crash_fix_and_run_logging.py
- TestDeferredWarmup
- test_feature_selector_properties.py
- _resolve_plateau_patience
- compute_labels
- TestPlotPerRuleBreakdown
- resolve_island_hyperparams
- evaluator_health.py
- _build_target
- test_migration_safety.py
- parity_scenario_strategy
- TestRunLogHandlerLifecycle
- resolve_phase2_stage_params
- apply_fuzzy_feature_scaling
- set_purged_wf_reference_rows
- phase2_support.py
- .encode_condition
- test_output_writer_properties.py
- CPUBacktestEngine
- test_phase2_gpu_throughput.py
- .decode_chromosome
- TestOOSEvaluatorRun
- gate_positive_good
- phase2_island_scheduler.py
- _should_inject_diversity_recovery
- _remove_low_dispersion
- prop_settings
- test_crash_fix_properties.py
- TestPlotDistributionAndEquity
- baselines.py
- test_evolution_convergence.py
- _jax_compute_trade_outcomes
- _should_post_restart_early_stop_phase2
- test_phase2_stage.py
- TestExecutionHealthInGate
- _make_rule_set
- trade_support_penalty
- TestHammingThresholdAutoScale
- TestValLeakGate
- DataFrame
- _jax_compute_rule_signals
- TestParetoCollapseWarningGate
- test_rb_governor_tail_holdout.py
- _compute_stability
- test_evaluator_clean_writer.py
- MonthlyWindowSummary
- ._engine
- test_rb_concentration_tail_fail_closed.py
- constrained_non_dominated_sort
- test_feature_selector.py
- OOS_Evaluator
- stratification_scenario_strategy
- TestSpearmanSignConsistency
- TestDecoupledObjectives
- log_memory_rss
- ._record_research_integrity
- _apply_colab_gpu_defaults
- .save_archive
- test_evaluator_health.py
- TestDeriveEpochSeed
- compute_phase2_objectives_from_metrics
- test_phase2_island_scheduler.py
- .load_pool
- test_property_27_test_data_preparation_consistency
- _MockGenerator
- TestIslandSchedulerGlobalMode
- TestEndToEndRotation
- TestRefreshObjectivesOnResumeGate
- conftest.py
- .get_dont_care
- .skip_if_valid
- _validate_schema
- resolve_evolution_floors
- TestSeedDirectionUniqueness
- test_phase2_island_early_stop.py
- BFS and DFS Graph Traversal
- TestGPUCPUNumericalParity
- DataFrame
- TestSparsePositiveMode
- TestZeroRatioBoundary
- ._build_per_symbol_rows
- TestGlobalMetricsCacheClearing
- ResearchProfile
- .load_and_validate
- _should_plateau_early_stop_phase2
- TestNaNHandling
- TestSparseSignedMode
- ._prune_splits_after_phase1
- TestPlateauBranchDecision
- _dominates
- TestPlateauEarlyStopBehavior
- TestMinEpochGuard
- TestNClustersDefined
- TestF3PathResolution
- context_contract_digest
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

## Communities (187 total, 11 thin omitted)

### Community 0 - "evox_runner.py"
Cohesion: 0.04
Nodes (120): _assign_eval_result(), _behavior_descriptor(), _binary_tournament_pick(), _build_diversity_reference(), _build_rank_and_crowding(), _constraint_violations(), _count_deployable_preview(), _count_pop_viable() (+112 more)

### Community 1 - ".run"
Cohesion: 0.09
Nodes (26): _condition_feature_names(), _entry_validation_per_symbol_metrics(), _filter_compatible_previous_pool(), _pool_entry_rank(), _pool_path_key(), _positive_contributor_symbols(), Any, Return a JSON-safe, stable snapshot of CPU per-symbol metrics. (+18 more)

### Community 2 - "rb_governor.py"
Cohesion: 0.06
Nodes (79): expectancy_lcb_pct(), Return a conservative lower bound for per-trade net expectancy. Exact CPU…, feature_conditions_only(), phase2_rule_id(), Stable provenance identifiers for Phase 2 feature rules. The RB Governor may…, Return normalized non-symbol, non-context conditions in order. Symbol scope and…, Hash the immutable Phase 2 feature logic and its discovery scope., _available_symbols() (+71 more)

### Community 3 - "_make_engine"
Cohesion: 0.08
Nodes (22): _make_df(), _make_engine(), All TP trades → win_rate = 100%., No losing trades → profit_factor = 99.0., Build a minimal DataFrame for testing., position_notional = equity * capital_pct/100 * leverage when exposure is free., position_notional is capped when exposure is nearly full., When exposure is full, position_notional = 0. (+14 more)

### Community 4 - ".run"
Cohesion: 0.07
Nodes (32): FileHandler, _log_phase_entry(), _log_pipeline_config(), _now_iso(), DataFrame, Persist the final status for the current run identity., Mark standalone phase runs failed when an exception escapes., Run all pipeline phases in order. Returns ------- dict Results from each phase,… (+24 more)

### Community 5 - "test_evox_runner.py"
Cohesion: 0.05
Nodes (45): _inherit_val_metrics_from_global_cache(), Phase2EvolutionState, Copy val_* from global cache for identical chromosomes when val is skipped.…, Evolve one island epoch and return updated resumable state., Resumable NSGA-III state for symbol-island epoch scheduling., Return survivors that do not already carry a validation snapshot. Validation…, Return updated (best_max_return, gens_without_improvement)., run_phase2_evolution_epoch() (+37 more)

### Community 6 - "_symbol_specialized_variants"
Cohesion: 0.13
Nodes (22): _attach_source_symbol_filters(), _ensure_symbol_filtered_rule(), Add deterministic single-condition RB candidates. Evolution is deliberately…, Island/cluster symbols carried on Phase 2 pool entries., Feature ANDs + island symbol ORs (engine ORs multiple ``symbol is X``)., Return rule with an explicit symbol filter when required. This is a safety net…, Build symbol-filtered variants and rank them using evaluator_v5 scoring.…, _source_symbols_from_rule() (+14 more)

### Community 7 - "_score_metrics"
Cohesion: 0.07
Nodes (40): _combined_return_score(), _evaluate_ruleset(), _optimize_risk(), Return (ok, bonus, penalty) for the desired train-valid balance shape. In…, Dominant objective: return/DD with train-valid balance, plus CV-fold…, Profit objective for lenient rule addition, but now evaluator_v5 aware. A new…, _score_metrics(), _train_valid_shape() (+32 more)

### Community 8 - "Rule_Pool_Generator"
Cohesion: 0.08
Nodes (19): Phase 2: GPU-accelerated multi-objective evolutionary rule pool generation.…, Inject guarded migration seeds for the next epoch., Restore slimmed training data from cache (no re-sampling needed)., Rule_Pool_Generator, _make_feature_infos(), In holdout mode, val engine must be built for pool admission even when…, All pool entries must have active conditions within [MIN_CONDITIONS,…, Phase 2 must use static TP=PHASE2_TP, SL=PHASE2_SL,… (+11 more)

### Community 9 - "_make_df"
Cohesion: 0.08
Nodes (21): _make_df(), _make_engine(), MonkeyPatch, Chromosome positions must follow feature_modes insertion order., Exact CPU re-evaluation must interpret genes like the GPU path., Chromosome that matches nothing returns 0 executed trades., Zero-signal chunks should use reject metrics without scanning., PHASE2_SKIP_ZERO_SIGNAL_SCAN must match always-scan metrics. (+13 more)

### Community 10 - "phase2_rule_pool.py"
Cohesion: 0.06
Nodes (75): encoder.py — Encoder Maps gene integer values to fuzzy value names, formats…, assign_strata_to_indices(), build_feature_sampling_probs(), pick_active_count(), _pick_active_index(), _pick_inactive_index(), ndarray, _random_active_class() (+67 more)

### Community 11 - "run_phase2_evolution"
Cohesion: 0.05
Nodes (32): Run Phase 2 NSGA-III evolution. Returns (pareto_pool, history) or state., run_phase2_evolution(), Evolutionary algorithm drivers for Phase 2., Unit tests for Pareto-collapse warning gate (audit finding #13). AC: The…, CountingEngine, Unit tests for batched offspring evaluation (Phase 2 runtime A1)., Stub engine recording every simulate_rule_batch call's batch size., Offspring should be evaluated via ONE simulate_rule_batch call per gen, not… (+24 more)

### Community 12 - "Output_Writer"
Cohesion: 0.11
Nodes (9): Output_Writer, Serializes RuleSet dicts to JSON and loads/validates existing JSON files.…, Verify the exact example from the spec works end-to-end., TestLoadAndValidateErrors, TestLoadAndValidateHappyPath, TestSpecExample, TestValidationErrorType, TestWriteConditionValidation (+1 more)

### Community 13 - "Data_Loader"
Cohesion: 0.07
Nodes (32): Data_Loader, load_dataset(), data/loader.py — Data_Loader Stateless CSV loading with full preparation…, Stateless data loader for the GPU-Fuzzy Trading Pipeline., Module-level wrapper around ``Data_Loader.load_dataset``., Data_Splitter, load_cached_split_if_fresh(), Load cached split parquets when they are newer than the source CSV. Validates… (+24 more)

### Community 14 - "Pipeline_Orchestrator"
Cohesion: 0.09
Nodes (36): _context_coverage_preflight(), _context_coverage_report(), Pipeline_Orchestrator, Return the active output root for this run., Temporarily rebind all cached output paths for one pipeline run., Load Phase 2 pools for both directions from the persistent cache., Ensure at least one valid strategy exists before standalone Phase 5., Return split-aware context coverage for both trading directions. (+28 more)

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
Cohesion: 0.06
Nodes (27): _make_timestamps(), Unit tests for gpu_fuzzy_trader.data.splitter.Data_Splitter Tests cover: - Per-…, Helper: build df, patch paths, run split, return (train, val)., Compute expected train/val per symbol under holdout+embargo split., floor(N * HOLDOUT_TRAIN_FRACTION) rows go to train., Remaining rows after embargo gap go to validation., For N=101: floor(101 * HOLDOUT_TRAIN_FRACTION) used, not round., train + validation + embargo dropped == total rows. (+19 more)

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

### Community 23 - "splitter.py"
Cohesion: 0.05
Nodes (68): downcast_numeric_df(), prune_train_columns(), DataFrame, df_slim.py — Reduce DataFrame memory for backtest engines. Keeps only columns…, Downcast label and feature columns to float32; symbol to category (inplace)., Return a copy with only meta, label, and requested feature columns. Parameters…, Drop unused feature columns from the full training split after Phase 1., slim_backtest_df() (+60 more)

### Community 24 - "Reporter"
Cohesion: 0.10
Nodes (14): Generates visual and tabular reports for each pipeline phase. All output files…, Reporter, _make_history(), _make_per_symbol_metrics(), _make_pnl_history(), History entries with missing keys should not raise., History entries with missing keys should not raise., Create a minimal Phase 2 history list. (+6 more)

### Community 25 - "test_encoder_properties.py"
Cohesion: 0.08
Nodes (41): all_active_chromosome_strategy(), chromosome_with_dont_cares_strategy(), feature_name_strategy(), composite, DrawFn, given, ndarray, Property-based tests for gpu_fuzzy_trader.features.encoder.Encoder **Validates:… (+33 more)

### Community 26 - "write_evaluator_clean"
Cohesion: 0.10
Nodes (21): _maybe_write_evaluator_clean(), Path, Write a stripped strategy file containing only ``direction`` and ``rules_set``.…, Write a stripped strategy file alongside the main strategy JSON. This is a…, Validate rule_set and write to JSON at path. After the main write, also writes…, Load JSON from path and run full schema validation. Parameters ---------- path…, write_evaluator_clean(), Path (+13 more)

### Community 27 - "CandidateRecord"
Cohesion: 0.10
Nodes (42): _candidate_coverage_symbols(), _candidate_positive_symbols(), CandidateRecord, _compose_ruleset(), _diversification_beam(), _diversification_shortlist(), _max_overlap(), Return supported positive validation symbols for one candidate. (+34 more)

### Community 28 - "nested_walk_forward.py"
Cohesion: 0.14
Nodes (24): Validation helpers for monthly, nested, and multiplicity-safe research., deflated_sharpe_ratio(), estimate_pbo(), Selection-multiplicity diagnostics for strategy research artifacts., Estimate the fraction of folds where the IS winner misses OOS median. Inputs…, Return a normal-approximation deflated Sharpe probability. This is a…, Create a compact ledger-ready multiplicity report., summarize_multiplicity() (+16 more)

### Community 29 - "environmental_selection_nsga2"
Cohesion: 0.10
Nodes (19): environmental_selection_nsga2(), Canonical NSGA-II truncation on a 2N merged population., _make_chromosome(), _make_deployable_entry(), ndarray, Without preservation, champion is evicted by gen ~8 under drift., At most TOP_K slots are overwritten by elite preservation., Preserved elite's objectives are reset to inf (forces re-eval). (+11 more)

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

### Community 35 - "context_permission_column"
Cohesion: 0.33
Nodes (6): context_permission_column(), context_trigger_column(), Return the direction-specific permission column name., Return the direction-specific LWC pullback-reversal trigger column., _context_feature_direction(), Return the strategy direction a context column belongs to (if any).

### Community 36 - "config.py"
Cohesion: 0.06
Nodes (39): _config_check(), ConfigError, effective_min_trade_support(), effective_monthly_min_trades(), effective_pool_min_val_trades(), effective_sortino_min_trade_threshold(), effective_val_trade_floor_for_objectives(), _finite_config_number() (+31 more)

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

### Community 41 - "_build_entries_from_rule_set"
Cohesion: 0.09
Nodes (17): _build_entries_from_rule_set(), Sort entries for v5 capital allocation (timestamp, rule, symbol, row)., Priority-based rule assignment: first matching rule wins per row. Mirrors…, _sort_entries_by_allocation_priority(), DataFrame, Row matching both rules should be assigned to rule 1 only., Rows not matching rule 1 should be assigned to rule 2., v5: earlier JSON rule wins over lower dataset row index at same time. (+9 more)

### Community 42 - "gpu_engine.py"
Cohesion: 0.07
Nodes (38): _batch_metrics_from_array(), _build_event_batch(), _exposure_slot_capacity(), _fast_reject_result_rows(), _jax_compute_rule_signals_sparse_batch(), _jax_open_slot(), _jax_release_open_slots(), _jax_simulate_equity_batch() (+30 more)

### Community 43 - "_gpu_runtime.py"
Cohesion: 0.06
Nodes (42): phase2_should_enrich_symbol_metrics(), Return True when GPU batch eval should run a follow-up CPU enrichment pass.…, configure_phase2_gpu_runtime(), detect_gpu_memory_used_gb(), detect_gpu_vram_gb(), detect_system_ram_gb(), evict_cluster_signatures(), _iter_warmup_targets() (+34 more)

### Community 44 - "._ensure_dir"
Cohesion: 0.07
Nodes (19): Reporting and visualization sub-package., _bucket_series_by_mode(), DataFrame, Series, reporter.py — Reporting and visualization for the GPU-Fuzzy Trading Pipeline.…, Write feature-stratified performance metrics to CSV files (one per split). For…, Create parent directories for *path* if they do not exist., Compute maximum drawdown percentage from an equity series. Parameters… (+11 more)

### Community 45 - "run_pipeline.py"
Cohesion: 0.10
Nodes (26): context_coverage_for_direction(), context_coverage_report(), context_floor_failures(), Any, DataFrame, Shared diagnostics for the mandatory direction-specific context contract., Return coverage diagnostics for every named frame and direction., Return mathematically impossible trade-floor failures for coverage. (+18 more)

### Community 46 - "test_certificate_first_selection.py"
Cohesion: 0.13
Nodes (22): _passes_symbol_concentration_gate(), _passes_symbol_contribution_certificate(), _portfolio_selection_certificate(), Any, Require positive, supported validation PnL from multiple symbols. Symbol…, Build the per-direction symbol certificate policy. Specialist islands are…, Return the certificate used by compose, risk, and profit selection., Return (hhi_abs_pnl, top_symbol_share, top_symbol) from per_symbol_metrics. (+14 more)

### Community 47 - "non_dominated_sort"
Cohesion: 0.12
Nodes (30): batch_hamming_min(), _batch_static_penalties_numba(), batch_static_support_penalties(), crowding_distance(), _crowding_distance_numba(), _crowding_distance_py(), _disable_numba_sort(), _dominates_numba() (+22 more)

### Community 48 - "optuna_search.py"
Cohesion: 0.10
Nodes (29): _active_search_space(), apply_trial_config(), collect_validation_metrics(), compute_score(), _copy_phase1_2_outputs(), main(), objective(), _pearson() (+21 more)

### Community 49 - "test_phase2_rule_pool.py"
Cohesion: 0.04
Nodes (48): _count_active_conditions(), _crowding_distance(), _get_dont_cares(), _mutate(), _pareto_sortino_stats(), Compute crowding distance for solutions in *front*. Parameters ----------…, Mutate a chromosome (returns a copy). When activating a dont_care gene, feature…, Return array of dont_care sentinels for each feature. (+40 more)

### Community 50 - "test_property_19_phase2_static_risk_parameters"
Cohesion: 0.12
Nodes (15): feature_infos_and_train_df(), _make_feature_infos(), _make_train_df(), composite, DataFrame, DrawFn, given, ndarray (+7 more)

### Community 51 - "_compute_rule_signal_mask"
Cohesion: 0.15
Nodes (13): conditions_cache_key(), get_or_build_rule_mask(), ndarray, condition_cache.py — Cached boolean masks for textual rule conditions. Avoids…, Stable hashable key for a list of condition strings., Return a boolean row mask for *conditions*, using *cache* when provided. The…, _compute_rule_signal_mask(), Build one boolean signal mask (evaluator_v5 parity). Feature conditions are… (+5 more)

### Community 52 - "validate_config"
Cohesion: 0.11
Nodes (28): _debug_symbol_universe_size(), effective_config_snapshot(), effective_min_profitable_symbols(), effective_rb_min_distinct_symbols(), Active symbol count when debug scope is on; None for full-universe runs., Cap cross-symbol profitability gate to the active universe size. With…, Return the RB coverage target for the active debug universe. Full runs keep…, Validate all high-impact hyperparameter relationships. The function is… (+20 more)

### Community 53 - "build_hybrid_symbol_clusters"
Cohesion: 0.12
Nodes (29): build_hybrid_symbol_clusters(), _corr_embedding_block(), _feature_names_union(), _feature_profile_block(), load_symbol_clusters(), Any, DataFrame, ndarray (+21 more)

### Community 54 - "_make_rule"
Cohesion: 0.12
Nodes (13): _legacy_writer_contract(), _make_rule(), fixture, Unit tests for gpu_fuzzy_trader.output.writer — Output_Writer Tests cover: -…, A rule with only tp non-zero should be accepted., These schema tests predate mandatory trend context., Spot-check a variety of valid fuzzy value names., Write rule_set to a temp file and reload the raw JSON. (+5 more)

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

### Community 59 - ".simulate_rule_set"
Cohesion: 0.17
Nodes (8): _batch_eval_rule_set_pickled(), Simulate a rule set on rows [row_start, row_end) without copying the df. Used…, Simulate a rule set and return performance metrics. Parameters ----------…, Simulate using a precomputed rule-evaluation mask cache. The mandatory…, Drop cached entries on rows the fixed context mask forbids. ``idx`` validity is…, Evaluate multiple rule sets in parallel (ProcessPool, thread fallback)., Top-level worker for ProcessPoolExecutor (must be picklable)., _rules_need_normalized_symbols()

### Community 60 - "ndarray"
Cohesion: 0.08
Nodes (27): _build_rule_signal_mask(), _expectancy_lcb_pct(), _expected_shortfall_pct(), precompute_release_indices(), precompute_release_indices_from_offsets(), DataFrame, ndarray, Return mean trade return, sample std, and a normal lower bound in %. (+19 more)

### Community 61 - "._build_engine_for_df"
Cohesion: 0.13
Nodes (11): Build the selected Phase 2 backend for the sampled train frame., Return whether this generator should avoid allocating a JAX engine.…, Attach optional island metadata; safe when *owner* is a partial mock., Build an engine on *df* using the same backend selection logic., _minimal_backtest_df(), DataFrame, MonkeyPatch, Phase 2 respects PHASE2_USE_GPU=False without loading JAX GPU engine. (+3 more)

### Community 62 - "_make_df"
Cohesion: 0.08
Nodes (25): _make_df(), _make_rule_set(), DataFrame, Verify plot_equity_curve is called for all three splits and handles empty logs., Helper: run OOS_Evaluator.run() with mocked internals and capture plot calls., Override module-level path dicts and return originals., plot_equity_curve must be called with 'train', 'validation', and 'test'., plot_equity_curve must be called exactly three times (one per split). (+17 more)

### Community 63 - "TestRulePoolGeneratorRun"
Cohesion: 0.15
Nodes (8): Integration tests using tiny population and generation counts., Verify that Rule_Pool_Generator's persistent self._rng advances across multiple…, After two run_epoch() calls, the internal RNG state must differ from the…, The RNG should produce *different* sequences in two consecutive run_epoch()…, Two generators with different seeds must have different RNG state., Rule_Pool_Generator must initialize self._rng as a Generator., TestRulePoolGeneratorRng, TestRulePoolGeneratorRun

### Community 64 - "test_plateau_state_leak.py"
Cohesion: 0.11
Nodes (21): _make_minimal_gen(), _mock_evolution_state(), _mock_stage_plan(), Regression tests for plateau-state leak fixes (Fixes A + B). Fix A:…, AC-1, AC-3: plateau_streak resets per epoch; reset_plateau=True always., reset_plateau=True is passed on the very first epoch call., reset_plateau=True is also passed on epoch 2 (regression for leak)., Even when PHASE2_ISLAND_TWO_STAGE_ENABLED=False, reset_plateau=True. (+13 more)

### Community 65 - "passes_pool_admission_gate"
Cohesion: 0.05
Nodes (28): passes_pool_admission_gate(), Hard gate for Phase 2 pool/archive on merged holdout metrics. When…, MonkeyPatch, Tests for the hard overfit ratio gate in pool admission. The ratio gate rejects…, train=15%/val=4% (3.75× ratio, gap=11pp < 16pp) is REJECTED by the ratio gate…, train=15%/val=10% (1.5× ratio) is ADMITTED., train/val ≈2.5× is ADMITTED when under OVERFIT_RATIO_FLOOR and gap gate., With PHASE2_OVERFIT_RATIO_FLOOR=0.0, the ratio gate is disabled and the high-… (+20 more)

### Community 66 - "dashboard.py"
Cohesion: 0.15
Nodes (25): build_dashboard_data(), _direction_data(), _history_rows(), _load_json(), main(), Any, Path, Read-only HTML dashboard for existing pipeline artifacts. The dashboard… (+17 more)

### Community 67 - "TestWriteSpearmanCorrelationReport"
Cohesion: 0.13
Nodes (10): _make_dataset_with_label(), _make_datasets_by_split(), Feature not in dataset → NaN for that split., Dataset without label_close_288 → NaN for all features on that split., None dataset for a split → NaN for all features on that split., Empty selected_features → CSV with header only., All non-NaN Spearman values must be in [-1.0, 1.0]., Create a dataset DataFrame with feature columns and label_close_288. (+2 more)

### Community 68 - "test_data_loader_properties.py"
Cohesion: 0.15
Nodes (26): dataframe_with_nan_features(), dataframe_with_nan_labels(), _load_from_df(), _make_datetime_series(), composite, DataFrame, DrawFn, given (+18 more)

### Community 69 - "ValueError"
Cohesion: 0.09
Nodes (29): attach_barrier_outcomes(), barrier_column_names(), configured_barrier_pairs(), _first_touch_for_symbol(), _first_touch_for_symbol_numba(), _first_touch_for_symbol_python(), _number_token(), DataFrame (+21 more)

### Community 70 - "_m"
Cohesion: 0.13
Nodes (16): evaluator_health_penalty(), Penalty for evaluator_v5 failure modes — higher is worse. Parameters ----------…, _m(), Max positions 15 > 10 → positive penalty., Max positions 8 <= 10 → no additional penalty from positions., Missing ``raw_signal_count`` → treated as 0 → no skip/exec penalty., ``raw_signal_count=0`` → no skip/exec ratio computed → no penalty., Role 'train' same as 'valid' (1.0x). (+8 more)

### Community 71 - "_apply_dynamic_rule"
Cohesion: 0.21
Nodes (4): _apply_dynamic_rule(), Apply one exported text condition using the original threshold logic. Exactly…, Test all threshold branches of _apply_dynamic_rule., TestApplyDynamicRule

### Community 72 - "cpu_engine.py"
Cohesion: 0.07
Nodes (26): _append_allocated_entries(), _normalize_direction(), cpu_engine.py — CPUBacktestEngine Exact Python/NumPy replication of…, Compute a non-annualized Sortino Ratio from per-trade returns., _rule_symbols_for_allocation(), _safe_profit_factor(), _sortino_ratio_from_returns(), CPU and GPU backtest engine sub-package. (+18 more)

### Community 73 - "test_rb_fail_closed.py"
Cohesion: 0.12
Nodes (24): Any, Hash the complete economic strategy, including exit policy. ``phase2_rule_id``…, strategy_id(), _assert_capital_budget(), _assert_mandatory_context(), _enforce_capital_budget(), Normalize rule capital_pct so sum <= RB_MAX_TOTAL_CAPITAL., Fail closed if the fixed trend-context conditions were lost. The mandatory… (+16 more)

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
Cohesion: 0.13
Nodes (16): _capture_logs(), _make_feature_infos_crash(), _make_train_df_crash(), Unit tests for crash-fix-and-run-logging spec. Covers: - Task 1.1: Smoke tests…, Requirement 3.1 — save_archive is called before _release_resources in run()., Mock save_archive and _release_resources, run Rule_Pool_Generator.run() with a…, save_archive is called with self.direction as the first argument., Requirement 3.3 — If save_archive raises, the exception is caught, a WARNING is… (+8 more)

### Community 78 - "TestDeferredWarmup"
Cohesion: 0.20
Nodes (6): Unit tests for the ``defer_warmup`` flag on ``Rule_Pool_Generator``. When…, Existing callers without defer_warmup still warm at init., The configure_phase2_gpu_runtime call is inside 'if not self._defer_warmup:'…, _run_cluster_islands passes defer_warmup=True to all generators., _run_cluster_islands calls warmup_phase2_gpu_kernels per cluster., TestDeferredWarmup

### Community 79 - "test_feature_selector_properties.py"
Cohesion: 0.15
Nodes (23): dataset_with_high_dispersion_feature(), dataset_with_label_and_meta_columns(), dataset_with_sufficient_dispersion_feature(), _make_label_block(), composite, DataFrame, DrawFn, given (+15 more)

### Community 80 - "_resolve_plateau_patience"
Cohesion: 0.18
Nodes (10): True for cluster/orphan scoped runs rather than the global path., scoped_island_profile(), Resolve the plateau patience value based on profile and stage. Cluster/orphan…, Earliest gen for plateau stop. Island epochs (~20 gens) and scaled Stage A…, _resolve_plateau_min_generation(), _resolve_plateau_patience(), Tests for _resolve_plateau_patience helper., Helper to create a Phase2StageParams with controlled patience. (+2 more)

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

### Community 86 - "test_migration_safety.py"
Cohesion: 0.09
Nodes (19): _make_migrant_dict(), _make_mock_receiver(), Unit tests for migration safety — migrant gate and seed fraction. Acceptance…, Migrant with val_return 2.5% and >=15 val trades should be accepted., Migrant with enough val_return but too few val trades should be rejected., Verify that migrant injection uses PHASE2_MIGRATION_SEED_FRACTION, not ARCHIVE., PHASE2_MIGRATION_SEED_FRACTION defaults to 0.10., Ensure the migration fraction is not the same as archive fraction. (+11 more)

### Community 87 - "parity_scenario_strategy"
Cohesion: 0.50
Nodes (4): parity_scenario_strategy(), composite, DrawFn, Generate a random dataset and trade parameters for GPU-CPU parity testing.…

### Community 88 - "TestRunLogHandlerLifecycle"
Cohesion: 0.11
Nodes (16): DataFrame, MonkeyPatch, Phase 2 reproducibility default seed., PHASE2_SEED is drawn once at import via get_seed(); set GLOBAL_SEED=42 to fix…, Requirements 1.1, 1.4, 1.5, 1.6, 1.7 — run.log FileHandler is attached, writes…, Count FileHandlers on the root logger pointing to *path*., Patch every phase method on Pipeline_Orchestrator to be a no-op., run.log must exist after run() and contain both separator lines. (+8 more)

### Community 89 - "resolve_phase2_stage_params"
Cohesion: 0.18
Nodes (8): True when Stage A viability is critically low and search has plateaued., _should_viability_recovery(), StageLabel, Return stage-tuned hyperparameters. When *stage* is None (single-stage Phase…, resolve_phase2_stage_params(), TestResolvePhase2StageParams, TestStageObjectivePenalties, TestViabilityRecovery

### Community 90 - "apply_fuzzy_feature_scaling"
Cohesion: 0.29
Nodes (9): apply_fuzzy_feature_scaling(), fit_fuzzy_feature_scaling(), Any, DataFrame, Train-fitted normalization for ordinal ``ff_*`` fuzzy feature codes. The…, Build a train-only scaling contract for ordinal ``ff_*`` columns., Apply an existing train-fitted scaling contract in place and return *df*., Tests for train-fitted ordinal fuzzy feature scaling. (+1 more)

### Community 91 - "set_purged_wf_reference_rows"
Cohesion: 0.18
Nodes (7): Store full train_new.csv row count after loader prep (split time)., set_purged_wf_reference_rows(), holdout_mode(), purged_mode(), fixture, Tests for purged-WF trade-floor scaling helpers., TestScaleTradeFloor

### Community 92 - "phase2_support.py"
Cohesion: 0.05
Nodes (47): effective_min_trade_pool_floor(), IslandHyperparams, Resolved Phase 2 knobs for cluster or orphan slices., _filter_pool_by_admission(), _pool_entry_passes_admission(), Check stored train/val metrics on a pool JSON entry., compute_robust_score(), compute_support_penalty_and_specialist() (+39 more)

### Community 93 - ".encode_condition"
Cohesion: 0.08
Nodes (21): ConfigurationError, encode_condition(), Encoder, get_dont_care(), Exception, Return '[feature_name] IS Fuzzy Value Name' for a valid gene. Parameters…, Raised when a chromosome contains an invalid gene value (e.g. dont_care)., Stateless encoder that maps gene values to fuzzy condition strings. (+13 more)

### Community 94 - "test_output_writer_properties.py"
Cohesion: 0.15
Nodes (25): parse_symbol_condition(), Parse optional symbol filters. Supported formats: "symbol is 1" "symbol IS 1"…, all_zero_rule_st(), _is_valid_exported_condition(), oversized_rule_set_st(), composite, DrawFn, given (+17 more)

### Community 95 - "CPUBacktestEngine"
Cohesion: 0.06
Nodes (36): CPUBacktestEngine, CPU backtest engine that exactly mirrors evaluator_v5.ipynb's…, _condition_feature(), Exception, writer.py — Output_Writer Serializes RuleSet dicts to JSON with exact schema…, Validate an optional symbol filter (symbol is X / [symbol] IS X)., Validate a single condition string. Accepts either: - [feature_name] IS Fuzzy…, Validate a single rule object. Returns the validated rule dict (with numeric… (+28 more)

### Community 96 - "test_phase2_gpu_throughput.py"
Cohesion: 0.16
Nodes (16): _jax_compute_rule_signals_batch(), Batch rule matching for B chromosomes simultaneously. Returns (B, N) boolean…, get_gpu_backtest_engine_class(), jax_gpu_backtest_available(), Any, Detect whether JAX / GPUBacktestEngine can be loaded on this host. JAX can fail…, Return ``GPUBacktestEngine`` if ``gpu_engine`` imports cleanly; else ``None``., True when ``get_gpu_backtest_engine_class()`` would succeed. (+8 more)

### Community 97 - ".decode_chromosome"
Cohesion: 0.20
Nodes (7): decode_chromosome(), ndarray, Convert a chromosome array to a list of condition strings. Genes equal to the…, See module-level :func:`decode_chromosome`., All 10 signed values decode correctly via decode_chromosome., Chromosome with numpy int32/int64 values should work., TestDecodeChromosome

### Community 98 - "TestOOSEvaluatorRun"
Cohesion: 0.09
Nodes (12): Load long.json and short.json via Output_Writer.load_and_validate(). Parameters…, Return only non-empty RB strategies accepted for deployment., _isolate_phase5_reporter_outputs(), fixture, Keep Reporter plots/CSVs out of the checked-in outputs directory., Write a valid rule set JSON to path., Integration tests using tmp_path overrides for all output paths. The run()…, Run OOS_Evaluator.run() once for the whole class (long direction only). (+4 more)

### Community 99 - "gate_positive_good"
Cohesion: 0.28
Nodes (16): _finite_number(), gate_positive_good(), _metric_int(), positive_good_reject_reasons(), PositiveGoodThresholds, Shared metric gates used by Phase 2 admission and RB Governor. The gate is…, Minimum evidence required on train and validation splits., Return stable, machine-readable reasons for a gate rejection. (+8 more)

### Community 100 - "phase2_island_scheduler.py"
Cohesion: 0.04
Nodes (66): filter_df_to_symbols(), DataFrame, Return rows for the given symbols; raises if column missing or no rows., clear_global_metrics_cache(), extract_deployable_migrants(), Clear the global eval cache and force GC. Used to free RAM between cluster runs…, Return elite deployable-preview entries suitable for guarded migration., persist_symbol_clusters() (+58 more)

### Community 101 - "_should_inject_diversity_recovery"
Cohesion: 0.17
Nodes (8): _diversity_recovery_min_unique_ratio(), _should_inject_diversity_recovery(), Test that Check 3 fires when pareto_size=3, plateau_streak=2, pop_size=100. For…, Check 3 requires plateau_streak >= 2 (isolated with pop_size=100)., For pop_size=100, Check 2 threshold=2, so pareto_size=4 should NOT trigger., Check 3 respects PHASE2_DIVERSITY_RECOVERY_ENABLED., TestPopulationDiversityMetrics, TestParetoCollapseDiversityRecovery

### Community 102 - "_remove_low_dispersion"
Cohesion: 0.17
Nodes (10): Remove features where more than `threshold` fraction of values are identical.…, _remove_low_dispersion(), DataFrame, Exactly 95% identical → NOT > 0.95 → keep., 96% identical → > 0.95 → remove., Two features with corr > 0.95 → keep higher-scored one., Two uncorrelated features → both kept., Features in different modes are not compared for redundancy. (+2 more)

### Community 103 - "prop_settings"
Cohesion: 0.07
Nodes (44): HealthCheck, prop_settings(), Shared Hypothesis settings for property tests. Set PYTEST_LOW_MEMORY=1 to scale…, Hypothesis settings with optional low-memory example scaling., _make_timestamps(), composite, DataFrame, DrawFn (+36 more)

### Community 104 - "test_crash_fix_properties.py"
Cohesion: 0.10
Nodes (26): _append_xla_flag(), configure_jax_env(), _cuda_package_root(), Path, JAX/XLA runtime configuration — call before the first ``import jax``., Find a pip-installed CUDA toolkit root, if one is available., Append an XLA flag once, preserving explicit user configuration., Configure JAX/XLA runtime for predictable desktop-friendly GPU usage. -… (+18 more)

### Community 105 - "TestPlotDistributionAndEquity"
Cohesion: 0.16
Nodes (6): _make_dist_logs_by_split(), _make_dist_trade_log(), Create a trade log with all columns needed for distribution/equity plots., Create trade_logs_by_split dict with all three splits populated., A trade log with a single trade should not raise., TestPlotDistributionAndEquity

### Community 106 - "baselines.py"
Cohesion: 0.24
Nodes (17): compute_entry_time_priority(), Map each row to a timestamp priority code (evaluator_v5 parity)., _compact(), _drop_one_condition_variants(), _equal_weight(), _evaluate(), evaluate_baselines(), _feature_shuffle() (+9 more)

### Community 107 - "test_evolution_convergence.py"
Cohesion: 0.17
Nodes (9): _normalize_for_association(), Rank-based normalization (robust to outliers like trade_penalty=50)., Tests for H5/M4/M5 evolution convergence behaviors. Covers: - HoF trimming at…, Verify rank normalization does not crash on degenerate inputs., All-equal objective values should produce valid normalised output with no NaN,…, Single-row input should not crash., Two objectives, all-equal values, should produce valid output. After rank…, Mixed values (not all equal) should still work. (+1 more)

### Community 108 - "_jax_compute_trade_outcomes"
Cohesion: 0.24
Nodes (5): _jax_compute_trade_outcomes(), JAX-jitted vectorized trade outcome computation., Vectorized trade outcome computation for all rows. Mirrors…, Multiple rows processed simultaneously., TestComputeTradeOutcomes

### Community 109 - "_should_post_restart_early_stop_phase2"
Cohesion: 0.29
Nodes (9): Break the epoch when a plateau restart yields no improvement. Independent of…, _should_post_restart_early_stop_phase2(), Unit tests for post-restart no-improvement early stop (Phase 2 runtime)., test_global_disabled_no_stop(), test_global_uses_global_knobs(), test_island_disabled_no_stop(), test_island_streak_at_patience_stops(), test_island_streak_below_patience_no_stop() (+1 more)

### Community 110 - "test_phase2_stage.py"
Cohesion: 0.24
Nodes (9): island_stage_budgets(), IslandStagePlan, Phase 2 two-stage search hyperparameter profiles (exploration vs refinement)., Resolved stage and remaining generation budget for one symbol island., Split an island's total generation budget into Stage A / Stage B portions. Uses…, Map completed island generations to the active two-stage profile., resolve_island_stage(), Unit tests for Phase 2 stage-specific hyperparameter profiles. (+1 more)

### Community 111 - "TestExecutionHealthInGate"
Cohesion: 0.25
Nodes (5): Verify that ``gate_positive_good`` calls ``execution_ok`` when flagged., When require_execution_health=True and execution_ok fails, gate returns False., When require_execution_health=True and execution_ok passes, gate still returns…, When require_execution_health=False, gate ignores raw_signal_count., TestExecutionHealthInGate

### Community 112 - "_make_rule_set"
Cohesion: 0.22
Nodes (3): _make_rule_set(), TestWriteDirectionValidation, TestWriteHappyPath

### Community 113 - "trade_support_penalty"
Cohesion: 0.38
Nodes (4): Backward-compatible wrapper returning penalty only., trade_support_penalty(), Between the hard-reject floor and the soft threshold the penalty is graduated., TestTradeSupportPenalty

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
Cohesion: 0.07
Nodes (31): _make_walk_forward_fold_engines(), _passes_tail_selection_gate(), Split val_selection into n_splits chronological folds + optional tail holdout.…, Validate a trial ruleset on the reserved chronological validation tail. The…, _make_synthetic_df(), DataFrame, Verify _make_walk_forward_fold_engines splits data correctly., 2 splits + tail holdout → 2 fold engines + 1 tail engine. (+23 more)

### Community 120 - "_compute_stability"
Cohesion: 0.17
Nodes (9): _compute_stability(), Compute cross-symbol stability score. stability = 1 - (std / mean) If mean is 0…, All symbols have same score → std=0 → stability=1., Mean is 0 → stability=0., Single symbol with positive score → stability=1., Single symbol with zero score → stability=0., Very high variance → stability could be negative → clipped to 0., Stability should always be in [0, 1]. (+1 more)

### Community 121 - "test_evaluator_clean_writer.py"
Cohesion: 0.18
Nodes (11): _make_rule(), minimal_strategy(), fixture, Unit tests for ``gpu_fuzzy_trader.output.writer.write_evaluator_clean``. Tests…, Tests that ``Output_Writer.write`` also produces the evaluator-clean file., Output_Writer.write produces both the main file and the evaluator-clean file., Long direction also produces the correct clean file., Strategy with only the two required keys (no extras). (+3 more)

### Community 122 - "MonthlyWindowSummary"
Cohesion: 0.07
Nodes (38): build_monthly_windows(), _datetime_series(), evaluate_rule_set_monthly(), monthly_penalty(), monthly_return_counts_as_good(), MonthlyWindowSummary, DataFrame, Series (+30 more)

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
Cohesion: 0.08
Nodes (18): OOS_Evaluator, DataFrame, Out-of-sample evaluator for Phase 5. Loads the final long/short strategies,…, Run out-of-sample evaluation. Parameters ---------- allowed_directions :…, Prepare test data using Data_Loader.load_dataset(). Applies the same…, Load prepared train, validation, and test datasets., Load and validate a strictly newer, untouched forward period., Load selected features for a direction when available. (+10 more)

### Community 128 - "stratification_scenario_strategy"
Cohesion: 0.24
Nodes (15): dataset_with_features_strategy(), metrics_strategy(), composite, DataFrame, DrawFn, Generate a trade log DataFrame with 2–50 rows. Parameters ---------- n_rules:…, Generate a metrics dict with reasonable float values. Keys: win_rate,…, Generate a dataset DataFrame with 10–100 rows. Parameters ----------… (+7 more)

### Community 129 - "TestSpearmanSignConsistency"
Cohesion: 0.18
Nodes (9): ndarray, Tests for _check_spearman_sign_consistency, including the val_df check., Build a minimal DataFrame without 'symbol' to avoid symbol-based folding., AC1: Train all positive, val negative → feature blacklisted., AC2: Train all positive, val positive → feature still kept., AC3: Train all positive, tiny |val_rho| < min_abs_corr → feature still kept., AC4: val_df=None → pre-task-8 behavior (mixed train signs still blacklisted)., AC5: val_df without label_close_288 column → val check skipped (no crash). (+1 more)

### Community 130 - "TestDecoupledObjectives"
Cohesion: 0.14
Nodes (8): Task 3 tests: penalties are not identically added to all objectives., The three objectives should respond differently to the same metrics, proving…, When the trade floor is triggered, only f2 gets the trade_penalty. f1 and f3…, With USE_TOTAL_RETURN_OBJ + JOINT_TRAIN_VAL, f3 uses robust_return_pct., PF floor adds to support_penalty even after decoupling. This ensures the…, Spec scenario: pf=1.10, EVOLUTION=1.05 (new) vs 1.15 (old); the new floor…, PHASE2_PROFIT_FACTOR_FLOOR (deprecated alias) still returns 1.15 by default for…, TestDecoupledObjectives

### Community 131 - "log_memory_rss"
Cohesion: 0.19
Nodes (11): log_memory_rss(), Optional process memory logging (set LOG_MEMORY=1)., Log current process RSS in MiB when LOG_MEMORY=1., Force GC and clear JAX compilation caches between Phase 2 directions. Releases:…, release_phase2_resources(), Release GPU engines and slim in-memory data between island epochs., Drop engine and sampled data to free RAM before the next direction., Tests for optional memory logging helpers. (+3 more)

### Community 132 - "._record_research_integrity"
Cohesion: 0.15
Nodes (11): __main__.py — Entry point for `python -m gpu_fuzzy_trader.run_pipeline` Allows…, main(), _phase5_test_metrics(), _print_run_summary(), Any, Append one auditable record after a pipeline evaluation completes., Run the full pipeline from the command line., Persist the diagnostic even when preflight blocks the pipeline. (+3 more)

### Community 133 - "_apply_colab_gpu_defaults"
Cohesion: 0.33
Nodes (6): _apply_colab_gpu_defaults(), is_colab_runtime(), True when running on Google Colab (/content runtime)., Colab T4 optimization for Phase 2 runs., MonkeyPatch, test_colab_defaults_apply_when_content_exists()

### Community 134 - ".save_archive"
Cohesion: 0.27
Nodes (5): Read JSON from *path* and return None when the file cannot be loaded., Load a compatible persistent archive if it exists, otherwise return None.…, Merge the latest pool into a persistent archive and write atomically., _read_json_payload(), TestArchivePersistence

### Community 135 - "test_evaluator_health.py"
Cohesion: 0.17
Nodes (7): Unit tests for ``evaluator_health_penalty`` and ``execution_ok`` (Task 4).…, Verify that evaluator health penalty is subtracted from combo score., When evaluator_health_penalty returns > 0, the combo score is lower., When evaluator health is fine, no penalty is applied., Both public functions are importable from the module., TestHealthPenaltyWiredIntoRB, TestModuleImportable

### Community 136 - "TestDeriveEpochSeed"
Cohesion: 0.17
Nodes (7): An unknown PHASE2_PER_EPOCH_WINDOW_SEED_MODE raises ValueError., Deterministic seed derivation for per-epoch windows., Same (base_seed, epoch_idx) produces the same seed., Different epoch indices produce different seeds., None base seed returns None., PHASE2_PER_EPOCH_WINDOW_SEED_MODE='hash_island_epoch' produces deterministic…, TestDeriveEpochSeed

### Community 137 - "compute_phase2_objectives_from_metrics"
Cohesion: 0.04
Nodes (57): _archive_direction(), attach_cv_fold_returns_batch(), _build_cpu_archive_engine(), _build_pool_from_archive(), _chromosome_batch(), _chromosome_for_pool_export(), compute_phase2_objectives_from_metrics(), CvFoldValEvaluator (+49 more)

### Community 138 - "test_phase2_island_scheduler.py"
Cohesion: 0.25
Nodes (10): compute_cluster_generation_budgets(), Resolve per-island generation budgets. By default each island receives the full…, Unit tests for cluster island scheduler budget math and epoch guard., Multi-symbol clusters each get the full island budget (not split)., The dead migration helper _should_migrate_this_round has been deleted. Verify…, test_epoch_rounds_cover_budget(), test_gens_per_cluster_full_budget_each(), test_one_symbol_islands_get_full_budget_each() (+2 more)

### Community 139 - ".load_pool"
Cohesion: 0.15
Nodes (6): Load existing pool if valid, return None if missing., Return loaded pool if valid, None if need to run., Validate the structure of a loaded pool JSON. Raises ValueError if the schema…, _validate_pool_schema(), TestLoadPool, TestValidatePoolSchema

### Community 140 - "test_property_27_test_data_preparation_consistency"
Cohesion: 0.20
Nodes (11): _make_timestamps(), composite, DataFrame, DrawFn, given, Timestamp, **Property 27: Test Data Preparation Consistency** **Validates: Requirements…, Return n strictly increasing timestamp strings starting from *base*. (+3 more)

### Community 141 - "_MockGenerator"
Cohesion: 0.22
Nodes (6): _MockGenerator, Minimal mock for Rule_Pool_Generator used in epoch guard tests., Test the epoch guard loop logic using mocked generators., The guard fires when remaining < PHASE2_ISLAND_MIN_EPOCH_GENERATIONS. The…, The guard does NOT fire when remaining >= PHASE2_ISLAND_MIN_EPOCH_GENERATIONS., TestMinEpochGuardWithMocks

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

### Community 147 - ".skip_if_valid"
Cohesion: 0.33
Nodes (3): Check if output files exist and are valid. Returns ------- dict[str,…, fixture, TestSkipIfValid

### Community 148 - "_validate_schema"
Cohesion: 0.31
Nodes (3): Validate the structure of a loaded feature selection JSON. Raises ValueError if…, _validate_schema(), TestValidateSchema

### Community 149 - "resolve_evolution_floors"
Cohesion: 0.31
Nodes (6): EvolutionFloors, Resolved evolution-time floors (pool admission gates remain strict)., Return stage-aware fitness floors; defaults to global strict knobs. When both…, resolve_evolution_floors(), Stage A soft floors must survive island_hyperparams (cluster two-stage)., TestResolveEvolutionFloorsIslandTwoStage

### Community 150 - "TestSeedDirectionUniqueness"
Cohesion: 0.20
Nodes (6): AC: _derive_island_seed produces different seeds for long vs short., Same cluster ID but different direction ⇒ different seed., Same orphan symbol but different direction ⇒ different seed., _derive_island_seed signature must remain (base_seed, island_id) — no direction…, base_seed=None should return None regardless of island_id., TestSeedDirectionUniqueness

### Community 151 - "test_phase2_island_early_stop.py"
Cohesion: 0.28
Nodes (8): island_early_stop_enabled(), _should_early_stop_phase2(), Unit tests for island early-stop bypass., Regression: island patience must come from…, test_cluster_profile_disables_early_stop(), test_cluster_profile_disables_plateau(), test_island_patience_uses_island_knob_not_stage_params(), test_orphan_profile_disables_early_stop()

### Community 152 - "BFS and DFS Graph Traversal"
Cohesion: 0.20
Nodes (10): CLAUDE.md Graphify Integration, Constrained Query Expansion, BFS and DFS Graph Traversal, Graph Work Memory, Existing Graph Fast Path, CLAUDE.md Graphify Integration, Constrained Query Expansion, BFS and DFS Graph Traversal (+2 more)

### Community 153 - "TestGPUCPUNumericalParity"
Cohesion: 0.10
Nodes (15): ndarray, simulate_rule_set (delegated to CPU) is exactly identical to CPU engine., GPU backtest engine must handle padded chunking correctly., CPU backtest engine must support simulate_rule_batch with correct metrics., Compare GPU and CPU engine for 10 random chromosomes (Task 2 acceptance…, Build a multi-symbol DataFrame with mixed TP/SL/time-exit outcomes., Generate random chromosomes for binary + signed features., GPU engine results must match CPU engine within specified tolerances for 10… (+7 more)

### Community 154 - "DataFrame"
Cohesion: 0.25
Nodes (9): _downsample_chronological(), _evaluate_rule_on_window(), _monthly_admission_source_df(), DataFrame, Evaluate a single pool rule on a single monthly window. Returns the full window…, Prefer unsampled monthly val; fall back to sampled slim val., Preserve per-symbol time order required by exposure/release simulation., Pick a contiguous chronological slice of *n_rows* from *df*. Critical for… (+1 more)

### Community 155 - "TestSparsePositiveMode"
Cohesion: 0.22
Nodes (5): All non-negative, zero_ratio > 0.3 → sparse_positive., All zeros: zero_ratio = 1.0 > 0.3, min = 0 → sparse_positive., zero_ratio just above 0.3 → sparse_positive., zero_ratio must be computed on the full series, not just non-NaN., TestSparsePositiveMode

### Community 156 - "TestZeroRatioBoundary"
Cohesion: 0.22
Nodes (5): Exactly 30% zeros with non-negative values → positive (not sparse_positive)., 31% zeros with non-negative values → sparse_positive., Exactly 30% zeros with negative values → signed (not sparse_signed)., Just above 30% zeros with negative values → sparse_signed., TestZeroRatioBoundary

### Community 158 - "TestGlobalMetricsCacheClearing"
Cohesion: 0.27
Nodes (6): When seeded_keys is empty, nothing is removed., When all keys are seeded, cache becomes empty., Verify that only seeded keys are removed from the global cache., Create a deterministic chromosome key., Cache entries matching seeded_keys are removed; non-matching survive., TestGlobalMetricsCacheClearing

### Community 159 - "ResearchProfile"
Cohesion: 0.28
Nodes (5): Any, Typed, versioned profile for the active research contract., Small stable surface for comparing experiments. The legacy config module…, ResearchProfile, test_research_profile_is_stable_and_versioned()

### Community 161 - "_should_plateau_early_stop_phase2"
Cohesion: 0.09
Nodes (28): _should_plateau_early_stop_phase2(), Unit tests for island plateau early-stop safety net (task-3). Verifies that…, Island patience=8: streak=7 → False, streak=8 → True., Global patience remains 5., When island early-stop is explicitly disabled, dead islands still run., Orphan profile follows island-scoped settings., Island plateau early-stop is off for short full-budget one-symbol runs., Dead island (deployable=0, plateau_streak >= patience) → stops. (+20 more)

### Community 162 - "TestNaNHandling"
Cohesion: 0.29
Nodes (4): All-NaN series: unique_vals = [], n_unique = 0 ≤ 2, set() ⊆ {0,1} → binary., NaN is not == 0, so it does not inflate zero_ratio., NaN should not push zero_ratio above threshold., TestNaNHandling

### Community 163 - "TestSparseSignedMode"
Cohesion: 0.33
Nodes (3): Has negative values, zero_ratio > 0.3 → sparse_signed., NaN does not count as zero; zero_ratio on full series., TestSparseSignedMode

### Community 164 - "._prune_splits_after_phase1"
Cohesion: 0.25
Nodes (4): Selected fuzzy features for Phase 2., Drop unused feature columns from train/val splits to reduce RAM., Drop unused feature columns from train split (legacy single-split API)., TestPruneSplitsAfterPhase1

### Community 165 - "TestPlateauBranchDecision"
Cohesion: 0.25
Nodes (5): Verify the plateau branch decision logic in _run_nsga3. We monkeypatch…, First plateau triggers _plateau_diversity_restart., When disabled, _plateau_diversity_restart is NOT called., With max_restarts=2, restart is called at least twice., TestPlateauBranchDecision

### Community 166 - "_dominates"
Cohesion: 0.18
Nodes (7): _dominates(), _non_dominated_sort(), Return True if solution *a* dominates *b* (all objectives ≤, at least one <)., NSGA-II non-dominated sorting. Parameters ---------- objectives : np.ndarray…, Each solution dominates the next., TestDominates, TestNonDominatedSort

### Community 167 - "TestPlateauEarlyStopBehavior"
Cohesion: 0.38
Nodes (4): Verify decision logic uses correct patience values (regression: logs showed…, Island profile: streak=6 triggers when island_patience=6 even when…, Global profile: streak=6 does NOT trigger when global_patience=8., TestPlateauEarlyStopBehavior

### Community 168 - "TestMinEpochGuard"
Cohesion: 0.29
Nodes (5): parametrize, Behavioral tests for _should_skip_epoch helper used in _run_cluster_islands., Verify _should_skip_epoch returns correct value., Integration test: when _should_skip_epoch returns True, the…, TestMinEpochGuard

### Community 169 - "TestNClustersDefined"
Cohesion: 0.33
Nodes (4): AC: n_clusters is assigned inside _run_cluster_islands so the migration guard…, n_clusters must be assigned in _run_cluster_islands for the migration guard at…, The migration guard must reference n_clusters., TestNClustersDefined

### Community 170 - "TestF3PathResolution"
Cohesion: 0.40
Nodes (4): parametrize, Parametrized tests for f3 path resolution (Task 5: audit finding #5). Verifies…, Verify the correct f3 formula runs for each (USE_TOTAL_RETURN_OBJ,…, TestF3PathResolution

### Community 171 - "context_contract_digest"
Cohesion: 0.50
Nodes (4): context_contract(), context_contract_digest(), Return the full context contract for strategy/dataset identity., Return a stable hash of the static contract and fitted enrichment.

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

- **Why does `CPUBacktestEngine` connect `CPUBacktestEngine` to `rb_governor.py`, `_make_engine`, `_symbol_specialized_variants`, `_score_metrics`, `Rule_Pool_Generator`, `compute_phase2_objectives_from_metrics`, `phase2_rule_pool.py`, `_make_df`, `Data_Loader`, `test_cpu_engine_properties.py`, `TestGPUCPUNumericalParity`, `DataFrame`, `CandidateRecord`, `nested_walk_forward.py`, `_build_entries_from_rule_set`, `gpu_engine.py`, `test_certificate_first_selection.py`, `test_jax_compat.py`, `_compute_rule_signal_mask`, `TestEvalCvFoldReturns`, `GPUBacktestEngine`, `.simulate_rule_set`, `ndarray`, `._build_engine_for_df`, `ValueError`, `_apply_dynamic_rule`, `cpu_engine.py`, `test_gpu_engine.py`, `phase2_island_scheduler.py`, `prop_settings`, `baselines.py`, `_jax_compute_trade_outcomes`, `trade_support_penalty`, `_jax_compute_rule_signals`, `test_rb_governor_tail_holdout.py`, `MonthlyWindowSummary`, `._engine`, `OOS_Evaluator`?**
  _High betweenness centrality (0.123) - this node is a cross-community bridge._
- **Why does `Rule_Pool_Generator` connect `Rule_Pool_Generator` to `.run`, `TestDecoupledObjectives`, `log_memory_rss`, `.run`, `.save_archive`, `TestDeriveEpochSeed`, `compute_phase2_objectives_from_metrics`, `phase2_rule_pool.py`, `.load_pool`, `test_phase2_island_scheduler.py`, `Data_Loader`, `Pipeline_Orchestrator`, `_MockGenerator`, `_apply_monthly_admission_gate`, `TestRefreshObjectivesOnResumeGate`, `_make_train_df`, `TestEndToEndRotation`, `TestSeedDirectionUniqueness`, `Reporter`, `_dominates`, `test_phase2_window_rotation.py`, `TestMinEpochGuard`, `TestNClustersDefined`, `TestF3PathResolution`, `_gpu_runtime.py`, `run_pipeline.py`, `test_phase2_rule_pool.py`, `test_property_19_phase2_static_risk_parameters`, `._build_engine_for_df`, `TestRulePoolGeneratorRun`, `test_plateau_state_leak.py`, `passes_pool_admission_gate`, `test_crash_fix_and_run_logging.py`, `TestDeferredWarmup`, `TestRunLogHandlerLifecycle`, `CPUBacktestEngine`, `phase2_island_scheduler.py`, `TestValLeakGate`?**
  _High betweenness centrality (0.110) - this node is a cross-community bridge._
- **Why does `Reporter` connect `Reporter` to `.run`, `TestWriteSpearmanCorrelationReport`, `TestEquityCurveDateAxis`, `prop_settings`, `Rule_Pool_Generator`, `compute_phase2_objectives_from_metrics`, `phase2_rule_pool.py`, `TestPlotDistributionAndEquity`, `._ensure_dir`, `Data_Loader`, `TestWriteStrategyEvaluationTable`, `TestPlotPerRuleBreakdown`, `DataFrame`, `test_reporter.py`, `OOS_Evaluator`?**
  _High betweenness centrality (0.100) - this node is a cross-community bridge._
- **Are the 15 inferred relationships involving `Reporter` (e.g. with `CvFoldValEvaluator` and `Rule_Pool_Generator`) actually correct?**
  _`Reporter` has 15 INFERRED edges - model-reasoned connections that need verification._
- **Are the 64 inferred relationships involving `CPUBacktestEngine` (e.g. with `GPUBacktestEngine` and `JointPortfolioEngine`) actually correct?**
  _`CPUBacktestEngine` has 64 INFERRED edges - model-reasoned connections that need verification._
- **Are the 65 inferred relationships involving `Rule_Pool_Generator` (e.g. with `CPUBacktestEngine` and `Reporter`) actually correct?**
  _`Rule_Pool_Generator` has 65 INFERRED edges - model-reasoned connections that need verification._
- **Are the 31 inferred relationships involving `Output_Writer` (e.g. with `OOS_Evaluator` and `_Phase5JSONEncoder`) actually correct?**
  _`Output_Writer` has 31 INFERRED edges - model-reasoned connections that need verification._