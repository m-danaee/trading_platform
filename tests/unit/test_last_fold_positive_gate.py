import numpy as np
from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.phases.phase2_cv import evaluate_purged_cv_pool_admission_batch

class DummyFoldEngine:
    def __init__(self, val_ret):
        self.val_ret = val_ret
    def simulate_rule_batch(self, **kwargs):
        return [{"executed_trades": 60, "total_return_pct": 5.0, "profit_factor": 1.2}]
        
class DummyValFoldEngine:
    def __init__(self, val_ret):
        self.val_ret = val_ret
    def simulate_rule_batch(self, **kwargs):
        return [{"executed_trades": 20, "total_return_pct": self.val_ret, "profit_factor": 1.1}]

class DummyCVEngine:
    def __init__(self, val_rets):
        self._fold_engines = [DummyFoldEngine(r) for r in val_rets]
    def simulate_rule_batch(self, **kwargs):
        return [{"executed_trades": 60, "total_return_pct": 5.0, "profit_factor": 1.2}]

class DummyCVValEngine:
    def __init__(self, val_rets):
        self._fold_engines = [DummyValFoldEngine(r) for r in val_rets]
    def simulate_rule_batch(self, **kwargs):
        return [{"executed_trades": 20, "total_return_pct": min(val_rets), "profit_factor": 1.1}]

def test_last_fold_positive_gate():
    orig_gate = _cfg.PHASE2_REQUIRE_LAST_FOLD_POSITIVE
    orig_min_pass = _cfg.PHASE2_CV_POOL_MIN_FOLDS_PASS
    try:
        # last fold validation return is -1.0 (non-positive)
        train_cv = DummyCVEngine([5.0, 5.0, 5.0])
        val_cv = DummyCVValEngine([2.0, 3.0, -1.0])
        
        chroms = np.zeros((1, 10), dtype=np.int32)
        
        # Gate enabled -> Should fail
        _cfg.PHASE2_REQUIRE_LAST_FOLD_POSITIVE = True
        _cfg.PHASE2_CV_POOL_MIN_FOLDS_PASS = 2
        results = evaluate_purged_cv_pool_admission_batch(train_cv, val_cv, chroms)
        assert not results[0][0]
        
        # Gate disabled -> Should pass
        _cfg.PHASE2_REQUIRE_LAST_FOLD_POSITIVE = False
        results = evaluate_purged_cv_pool_admission_batch(train_cv, val_cv, chroms)
        assert results[0][0]
    finally:
        _cfg.PHASE2_REQUIRE_LAST_FOLD_POSITIVE = orig_gate
        _cfg.PHASE2_CV_POOL_MIN_FOLDS_PASS = orig_min_pass
