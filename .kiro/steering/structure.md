# Project Structure

```
trading_platform/
├── data/                          # CSV datasets (not in git)
│   ├── train.csv                  # Training data
│   ├── test.csv                   # Held-out test data
│   ├── train_75.parquet           # Auto-generated 75% split
│   └── validation_25.parquet      # Auto-generated 25% split
│
├── gpu_fuzzy_trader/              # Main package
│   ├── config.py                  # Single source of truth for hyperparameters
│   ├── run_pipeline.py            # Pipeline_Orchestrator (top-level entry)
│   ├── __main__.py                # Entry point for python -m
│   │
│   ├── data/                      # Data loading and splitting
│   │   ├── loader.py              # Data_Loader: CSV loading, NaN handling
│   │   └── splitter.py            # Data_Splitter: per-symbol chronological split
│   │
│   ├── features/                  # Feature detection, selection, encoding
│   │   ├── detector.py            # Feature_Detector: mode classification
│   │   ├── selector.py            # Feature_Selector: direction-specific scoring
│   │   └── encoder.py             # Encoder: gene → fuzzy value name
│   │
│   ├── backtest/                  # Trade simulation engines
│   │   ├── cpu_engine.py          # CPUBacktestEngine (canonical reference)
│   │   └── gpu_engine.py          # GPUBacktestEngine (JAX-accelerated)
│   │
│   ├── phases/                    # Five-phase pipeline modules
│   │   ├── phase2_rule_pool.py    # Rule_Pool_Generator: NSGA-II/MOEAD search
│   │   ├── phase3_rule_set.py     # Rule_Set_Selector: combinatorial optimization
│   │   ├── phase4_rl_optimizer.py # RL_Agent: DDPG/PPO risk tuning
│   │   └── phase5_oos.py          # OOS_Evaluator: final test
│   │
│   ├── output/                    # JSON serialization
│   │   └── writer.py              # Output_Writer: schema validation
│   │
│   └── reporting/                 # Reports and visualizations
│       └── reporter.py            # Reporter: equity curves, CSVs
│
├── outputs/                       # Generated artifacts (not in git)
│   ├── long.json                  # Final long strategy
│   ├── short.json                 # Final short strategy
│   ├── phase2_*.json              # Rule pools and history
│   ├── selected_features_*.json   # Phase 1 outputs
│   ├── pipeline.log               # JSON-lines timing log
│   └── reports/                   # Equity curves, per-symbol CSVs
│
├── tests/
│   ├── unit/                      # Unit tests per module
│   └── property/                  # Hypothesis property-based tests
│
├── evaluator_v3.ipynb             # Reference notebook for backtest semantics
├── requirements.txt               # Python dependencies
├── README.md                      # Full architecture documentation
└── RUN.md                         # Quick start guide
```

## Key Conventions

- **config.py is the single source of truth** — no module defines its own defaults
- **Backtest engines must be numerically equivalent** — GPU engine within 1e-4 tolerance of CPU engine
- **Labels are look-ahead values** — never used as model inputs, only for trade simulation
- **Last 288 rows per symbol are dropped** — no valid labels in that window
- **Phase outputs are JSON** — validated against schema, compatible with evaluator_v3.ipynb
