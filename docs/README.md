# Documentation Index

Technical documentation for the GPU-Fuzzy Trading Pipeline. Each document covers one phase in depth: algorithm details, function-by-function explanations, and the effect of every hyperparameter in `config.py`.

| Document | Phase | Key topics |
|---|---|---|
| [phase0_shared.md](phase0_shared.md) | Phase 0 — Shared | Data loading, 75/25 split, CPU backtest engine, all shared config constants |
| [phase1_feature_selection.md](phase1_feature_selection.md) | Phase 1 | Feature mode detection, mutual information scoring, stationarity filter, regime clustering, overlap reduction |
| [phase2_rule_pool.md](phase2_rule_pool.md) | Phase 2 | Chromosome encoding, NSGA-III/II evolution, support penalties, regime specialists, archive system |
| [phase3_rule_set.md](phase3_rule_set.md) | Phase 3 | Greedy construction, NSGA-II refinement, validation gates, symbol coverage, train-val correlation penalty |
| [phase4_wf_risk.md](phase4_wf_risk.md) | Phase 4 | Walk-forward splits, Optuna multi-objective search, TP/SL/capital search space, trial selection |
| [phase5_oos.md](phase5_oos.md) | Phase 5 | OOS evaluation, cross-split reporting, metric interpretation, failure mode diagnosis |

## Quick hyperparameter index

If you want to change a specific behavior, find the relevant parameter here:

| Goal | Parameter | Phase doc |
|---|---|---|
| Reduce GPU memory usage | `PHASE1_SAMPLING_TOTAL` | Phase 2 |
| Require more trades per rule | `MIN_TRADE_SUPPORT` | Phase 2 |
| Allow more/fewer conditions per rule | `MIN_CONDITIONS`, `MAX_CONDITIONS` | Phase 2 |
| More evolutionary search budget | `PHASE2_POPULATION_SIZE`, `PHASE2_GENERATIONS` | Phase 2 |
| Warm-start from previous runs | `PHASE2_ARCHIVE_SEED_FRACTION` | Phase 2 |
| Require broader symbol coverage | `PHASE3_MIN_SYMBOL_COVERAGE` | Phase 3 |
| Stricter anti-overfitting gate | `PHASE3_VAL_SORTINO_RATIO_GATE`, `PHASE3_VAL_GATE_PENALTY` | Phase 3 |
| More refinement budget | `PHASE3_REFINE_POP_SIZE`, `PHASE3_REFINE_GENERATIONS` | Phase 3 |
| Tighter drawdown constraint | `PHASE4_MAX_WORST_DRAWDOWN_PCT` | Phase 4 |
| More risk optimization budget | `PHASE4_N_TRIALS` | Phase 4 |
| More walk-forward windows | `PHASE4_WF_SPLITS` | Phase 4 |
| Change TP/SL search range | `PHASE4_TP_MIN/MAX`, `PHASE4_SL_MIN/MAX` | Phase 4 |
| More/fewer features per direction | `PHASE1_TOP_K_FEATURES` | Phase 1 |
| Force long/short feature divergence | `PHASE1_MAX_FEATURE_OVERLAP` | Phase 1 |
| Change fee rate | `FEE_PCT` | Phase 0 |
| Change label horizon | `TAIL_DROP_ROWS`, `MAX_HOLD_CANDLES` | Phase 0 |
