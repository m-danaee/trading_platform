# Product Overview

GPU-Fuzzy Trading Pipeline is a rule-mining trading system that discovers, optimizes, and evaluates trading strategies across multiple symbols. It mines interpretable fuzzy rules from discretized feature data, refines risk parameters via deep reinforcement learning, and produces JSON strategy files compatible with `evaluator_v3.ipynb`.

## Key Capabilities

- **Direction-specific feature selection** — mode-aware scoring and ranking for long/short strategies
- **GPU-accelerated rule pool generation** — NSGA-II/MOEAD evolutionary search with JAX
- **Rule set assembly** — combinatorial optimization for 2-5 rule teams
- **RL-based risk optimization** — DDPG/PPO tuning of TP/SL/capital allocation with Elbow Method stopping
- **Out-of-sample evaluation** — final test on held-out data

## Core Philosophy

This is a rule-mining system, not a predictive model. Labels are used only for scoring and backtesting — never as model inputs. The backtest engine exactly mirrors `evaluator_v3.ipynb`'s `CapitalManagedTradeSimulator` so optimization scores match final evaluation scores.

## Strategy Output

Final outputs are `long.json` and `short.json` containing 2-5 fuzzy rules each, with conditions like `[feature_name] IS Fuzzy Value Name` and risk parameters (TP, SL, capital_pct).
