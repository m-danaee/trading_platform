import numpy as np
import pytest
from gpu_fuzzy_trader.phases.phase3_rule_set import _run_nsga2_combinatorial

def test_population_clamping_removed():
    # Tiny pool of size 5
    pool = [{"conditions": [f"[feat_{i}] IS Very High"]} for i in range(5)]
    
    # We want pop_size = 20 (which is > len(pool) * 2 = 10)
    # If clamping is active, population would clip to 10.
    # If clamping is removed, population should be 20.
    
    evaluated_sets = []
    
    class MockEngine:
        def simulate_rule_set(self, rule_set, **kwargs):
            evaluated_sets.append(rule_set)
            return {
                "sortino_ratio": 1.2,
                "total_return_pct": 5.0,
                "max_drawdown_pct": 2.0,
                "win_rate": 0.5,
                "executed_trades": 100,
                "per_symbol_metrics": {"SYM_A": {"trade_count": 50}},
            }
            
    engine = MockEngine()
    
    _run_nsga2_combinatorial(
        pool=pool,
        val_engine=engine,
        train_engine=engine,
        pop_size=20,
        n_generations=1,
        min_rules=2,
        max_rules=3,
        seed=42,
        use_batch=False,
    )
    
    # Each individual's rule set is simulated on train_engine and val_engine,
    # so we expect 2 * effective_pop evaluations.
    # If clamping is active, effective_pop is 10, meaning 20 evaluations.
    # If clamping is removed, effective_pop is 20, meaning 40 evaluations.
    assert len(evaluated_sets) == 40
