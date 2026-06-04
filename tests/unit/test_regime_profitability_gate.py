from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.phases.phase2_support import passes_pool_admission_gate

def test_regime_profitability_gate():
    orig_gate = _cfg.PHASE2_REGIME_PROFITABILITY_GATE
    orig_splits = _cfg.PHASE2_POOL_REQUIRE_POSITIVE_SPLITS
    try:
        train_metrics = {
            "executed_trades": 60,
            "total_return_pct": 5.0,
            "profit_factor": 1.2,
            "regime_net_pnl": [10.0, -2.0, -1.0] # Only 1 regime is positive!
        }
        val_metrics = {
            "executed_trades": 20,
            "total_return_pct": 2.0,
            "profit_factor": 1.1
        }
        
        # Enabled -> Should fail (1 < 2 positive regimes)
        _cfg.PHASE2_POOL_REQUIRE_POSITIVE_SPLITS = True
        _cfg.PHASE2_REGIME_PROFITABILITY_GATE = True
        assert not passes_pool_admission_gate(train_metrics, val_metrics)
        
        # 2 positive regimes -> should pass
        train_metrics["regime_net_pnl"] = [10.0, 1.0, -1.0]
        assert passes_pool_admission_gate(train_metrics, val_metrics)
    finally:
        _cfg.PHASE2_REGIME_PROFITABILITY_GATE = orig_gate
        _cfg.PHASE2_POOL_REQUIRE_POSITIVE_SPLITS = orig_splits
