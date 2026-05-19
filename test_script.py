import sys
import numpy as np
from gpu_fuzzy_trader.phases.phase2_rule_pool import _pareto_sortino_stats
from gpu_fuzzy_trader.phases.phase3_rule_set import _evaluate_rule_set
from gpu_fuzzy_trader.phases.phase3_greedy import _scalar_score
from gpu_fuzzy_trader import config as _cfg

# 1) Assert _pareto_sortino_stats([], [{}]) returns {'mean_sortino_ratio': 0.0, 'best_sortino_ratio': 0.0}
res1 = _pareto_sortino_stats([], [{}])
assert res1 == {'mean_sortino_ratio': 0.0, 'best_sortino_ratio': 0.0}, f"Failed test 1: {res1}"

# 2) Assert _pareto_sortino_stats([0, 1], [{'sortino_ratio': 2.0}, {'sortino_ratio': 4.0}]) returns mean 3.0 and best 4.0
res2 = _pareto_sortino_stats([0, 1], [{'sortino_ratio': 2.0}, {'sortino_ratio': 4.0}])
assert res2['mean_sortino_ratio'] == 3.0, f"Failed local test 2 mean: {res2['mean_sortino_ratio']}"
assert res2['best_sortino_ratio'] == 4.0, f"Failed local test 2 best: {res2['best_sortino_ratio']}"

# 3) Build mock val/train engine pair
class MockEngine:
    def simulate_rule_set(self, rule_set):
        return {
            'sortino_ratio': 5.0,
            'total_return_pct': 10.0,
            'max_drawdown_pct': 1.0,
            'win_rate': 50.0,
            'executed_trades': 10,
            'per_symbol_metrics': {f'S{i}': {'trade_count': 1} for i in range(10)}
        }

# Setup config to ensure zero coverage penalty
_cfg.PHASE3_MIN_SYMBOL_COVERAGE = 7

val_engine = MockEngine()
train_engine = MockEngine()

rule_set = [{'tp': 0.02, 'sl': 0.01, 'capital_pct': 0.1, 'conditions': []}]
objectives, val_metrics = _evaluate_rule_set(rule_set, val_engine, train_engine)

# In _evaluate_rule_set:
# val_sortino = 5.0
# val_dd = 1.0
# val_wr = 50.0
# val_trades = 10
# symbols_with_trades = 10 (>= 7, so coverage_penalty = 0)
# overfitting_penalty = abs(5.0 - 5.0) / 5.0 = 0.0
# dup_penalty = 0.0
# total_penalty = 0.0 + 0.0 + 0.0 + 0.0 = 0.0
# f1 = -5.0 + 0.0 = -5.0
# f2 = 1.0 + 0.0 = 1.0
# f3 = -50.0 + 0.0 = -50.0

assert np.isclose(objectives[0], -5.0), f"f1 mismatch: {objectives[0]}"
assert np.isclose(objectives[1], 1.0), f"f2 mismatch: {objectives[1]}"
assert np.isclose(objectives[2], -50.0), f"f3 mismatch: {objectives[2]}"

# 4) Feed into _scalar_score with weights (1.0, 0.7, 0.5) and assert finite
weights = (1.0, 0.7, 0.5)
score = _scalar_score(val_metrics, objectives, weights)
assert np.isfinite(score), f"Score is not finite: {score}"

print("All assertions passed successfully.")
