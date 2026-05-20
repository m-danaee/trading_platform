# GPU-Fuzzy Trading Pipeline

## Product Overview

The GPU-Fuzzy Trading Pipeline is a ground-up, GPU-accelerated fuzzy rule mining system that discovers, optimizes, and evaluates trading strategies across 10 financial instruments. It's not a conventional predictive model but a rule-mining system that:

- Starts with pre-engineered, discretized feature columns
- Uses future labels only for scoring and backtesting (never as model inputs)
- Selects stable, direction-specific features per output type
- Evolves candidate fuzzy rules using GPU-accelerated NSGA-III multi-objective search
- Assembles compact rule teams (2-3 rules) via greedy construction and Pareto refinement
- Fine-tunes risk parameters (TP, SL, capital allocation) using deep reinforcement learning
- Evaluates final strategies on held-out test data

## Core Design Principles

1. **Symbol-aware chronological split** - Data is split per symbol chronologically (75/25)
2. **Mode-aware feature selection** - Features are classified by their fuzzy modes (binary/ternary/etc.)
3. **Explicit fuzzy rule encoding** - Rules use human-interpretable condition strings
4. **Backtest engine compatibility** - Exactly mirrors `evaluator_v3.ipynb` semantics
5. **Skip logic with validation** - Phases skip automatically when valid cached outputs exist
6. **Persistent archives** - Best rules are archived across runs for seeding future populations

## Key Artifacts

- **Strategy files**: `long.json` and `short.json` - Fully compatible with `evaluator_v3.ipynb`
- **Feature selections**: `selected_features_long.json` and `selected_features_short.json`
- **Rule pools**: `phase2_{long,short}_pool.json` - Pareto-front candidate rules
- **Reports**: Comprehensive metrics, equity curves, and per-symbol performance CSVs

## Five-Phase Pipeline

1. **Phase 1**: Direction-specific feature selection
2. **Phase 2**: GPU-accelerated rule pool generation (NSGA-III evolution)
3. **Phase 3**: Rule set selection (greedy + Pareto refinement)
4. **Phase 4**: RL-based risk optimization (TP/SL/capital allocation)
5. **Phase 5**: Out-of-sample evaluation on held-out test data

## Success Criteria

- Strategies must be interpretable (human-readable fuzzy rules)
- Must achieve positive risk-adjusted returns on out-of-sample data
- Must maintain symbol coverage (minimum 7 of 10 symbols with trades)
- Must avoid overfitting (validation vs. training performance monitoring)
- Must be computationally efficient (GPU acceleration where possible)