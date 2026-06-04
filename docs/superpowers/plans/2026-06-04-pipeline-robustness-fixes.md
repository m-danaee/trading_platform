# Pipeline Robustness & Generalization Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate non-stationary features flipping sign between train and validation, prevent search collapse on zero-trade rules in Phase 2, and increase the diversity of the Phase 3 ruleset pool by properly handling regime specialists.

**Architecture:** 
1. Append `val_df` as an additional fold in the Phase 1 Spearman sign consistency check to catch and filter out features flipping sign before they reach evolution.
2. In Phase 2 fitness evaluation, assign a maximum drawdown value of `100.0` to any candidate rule with trades below the minimum pool floor to prevent the optimizer from favoring inactive rules.
3. In Phase 2 pool admission, bypass the multi-regime profitability requirement for rules flagged as **regime specialists**, requiring them to be profitable only within their dominant regime.

**Tech Stack:** Python, NumPy, JAX, EvoX

---

### Task 1: Check Spearman Sign Consistency with Validation Folds
**Files:**
- Modify: `gpu_fuzzy_trader/features/selector.py`
- Modify: `gpu_fuzzy_trader/run_pipeline.py`
- Test: `tests/unit/test_feature_sign_consistency.py`

- [ ] **Step 1: Update the sign consistency unit test to cover validation data**
Modify `tests/unit/test_feature_sign_consistency.py` to test that passing a validation DataFrame containing a sign flip successfully blacklists the feature:

```python
import numpy as np
import pandas as pd
from gpu_fuzzy_trader.features.selector import _check_spearman_sign_consistency

def test_check_spearman_sign_consistency():
    n = 300
    df = pd.DataFrame({
        "symbol": ["A"] * n,
        "feature_a": np.linspace(1, 10, n),
        "feature_b": np.hstack([np.linspace(1, 10, 100), np.linspace(10, 1, 100), np.linspace(1, 10, 100)]),
        "label_close_288": np.linspace(1, 10, n)
    })
    
    stable = _check_spearman_sign_consistency(df, ["feature_a", "feature_b"], n_folds=3, min_folds=2)
    assert "feature_a" in stable
    assert "feature_b" not in stable

def test_check_spearman_sign_consistency_with_validation():
    n = 100
    train_df = pd.DataFrame({
        "symbol": ["A"] * n,
        "feature_c": np.linspace(1, 10, n),
        "label_close_288": np.linspace(1, 10, n)
    })
    val_df = pd.DataFrame({
        "symbol": ["A"] * n,
        "feature_c": np.linspace(10, 1, n),
        "label_close_288": np.linspace(1, 10, n)
    })
    
    # feature_c is consistent in train, but flips in val
    stable = _check_spearman_sign_consistency(train_df, ["feature_c"], n_folds=2, min_folds=1, val_df=val_df)
    assert "feature_c" not in stable
```

- [ ] **Step 2: Run pytest to verify the new test fails (TDD RED step)**
Run: `PYTHONPATH=. .venv/bin/pytest tests/unit/test_feature_sign_consistency.py -v`
Expected: Fail with `TypeError` due to unexpected argument `val_df`

- [ ] **Step 3: Modify `_check_spearman_sign_consistency` in `gpu_fuzzy_trader/features/selector.py`**
Update the function definition to accept `val_df: pd.DataFrame | None = None` and append it to `folds` list if present:

```python
def _check_spearman_sign_consistency(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_folds: int,
    min_folds: int,
    val_df: pd.DataFrame | None = None,
) -> set[str]:
    folds = _get_spearman_folds(df, n_folds)
    if val_df is not None:
        folds.append(val_df)
    label_col = "label_close_288"
    if label_col not in df.columns:
        return set(feature_cols)
        
    stable_features = set()
    for col in feature_cols:
        corrs = []
        for fold in folds:
            if col not in fold.columns:
                continue
            corr = _spearman(fold[col], fold[label_col])
            if not np.isnan(corr):
                corrs.append(corr)
        if len(corrs) < min_folds:
            continue
        has_pos = any(c > 0 for c in corrs)
        has_neg = any(c < 0 for c in corrs)
        if has_pos and has_neg:
            logger.info(
                "Blacklisting non-stationary feature %s: Spearman signs across folds: %s",
                col, corrs
            )
        else:
            stable_features.add(col)
    return stable_features
```

Also, update `select_features` and `run` to accept `val_df` and pass it down:

```python
    def select_features(
        self,
        train_df: pd.DataFrame,
        direction: str,
        regime_labels: Optional[pd.Series] = None,
        shared: Phase1SharedContext | None = None,
        val_df: pd.DataFrame | None = None,
    ) -> list[dict]:
```

In `select_features` line 346:
```python
        if config.PHASE1_REQUIRE_SIGN_CONSISTENCY:
            n_folds = config.PHASE1_STATIONARITY_FOLDS
            min_folds = config.PHASE1_SIGN_CONSISTENCY_MIN_FOLDS
            stable_cols = _check_spearman_sign_consistency(
                train_df, feature_cols, n_folds, min_folds, val_df=val_df
            )
            logger.info(
                "Phase 1 [%s]: sign consistency filter kept %d/%d candidate features",
                direction, len(stable_cols), len(feature_cols)
            )
            feature_cols = [c for c in feature_cols if c in stable_cols]
```

And in `run`:
```python
    def run(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> dict[str, list[dict]]:
```

```python
        ranked: dict[str, list[dict]] = {}
        for direction in ("long", "short"):
            ranked[direction] = self.select_features(
                train_df,
                direction,
                regime_labels=self._regime_labels,
                shared=shared,
                val_df=val_df,
            )
```

- [ ] **Step 4: Update the caller in `gpu_fuzzy_trader/run_pipeline.py`**
Pass `val_df` into `Feature_Selector.run` in `_run_phase1` of `run_pipeline.py`:

```python
        # Run Phase 1
        logger.info("Running %s …", phase_name)
        try:
            selector = Feature_Selector()
            result = selector.run(train_df, val_df=val_df)
```

- [ ] **Step 5: Run pytest to verify all feature sign consistency tests pass**
Run: `PYTHONPATH=. .venv/bin/pytest tests/unit/test_feature_sign_consistency.py -v`
Expected: PASS

- [ ] **Step 6: Commit**
```bash
git add tests/unit/test_feature_sign_consistency.py gpu_fuzzy_trader/features/selector.py gpu_fuzzy_trader/run_pipeline.py
git commit -m "feat: check feature Spearman sign consistency against validation data"
```

---

### Task 2: Prevent Search Collapse on 0-Trade Rules in Phase 2
**Files:**
- Modify: `gpu_fuzzy_trader/evolution/evox_runner.py`
- Test: `tests/unit/test_evox_runner.py`

- [ ] **Step 1: Write test case for 0-trade drawdown penalty**
Add a test in `tests/unit/test_evox_runner.py` to verify that when a chromosome has fewer than the minimum trades, its drawdown objective is set to `100.0 + pen` to prevent it from dominating:

```python
import numpy as np
from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.evolution.evox_runner import _evaluate_population_indices

def test_low_trade_drawdown_penalty():
    # Set CV fold mode trade floor to 25
    _cfg.SPLIT_MODE = "purged_rolling_cv"
    _cfg.PHASE2_CV_MIN_TRADE_POOL_FLOOR = 25
    
    # population of 1 candidate
    pop = np.zeros((1, 10), dtype=np.int32)
    dont_cares = np.ones(10, dtype=np.int32) * 5
    objectives = np.full((1, 3), np.inf)
    metrics_cache = [{}]
    
    class MockEngine:
        def simulate_rule_batch(self, chromosomes, **kwargs):
            # return metrics with 5 executed trades and 0.0 drawdown
            return [{"executed_trades": 5, "total_return_pct": 1.0, "sortino_ratio": 0.5, "max_drawdown_pct": 0.0, "win_rate": 0.5}]
            
    engine = MockEngine()
    _evaluate_population_indices(
        pop, [0], dont_cares, engine, [], objectives, metrics_cache
    )
    
    # Drawdown objective (index 1) should be penalized to 100.0 + support_penalty
    # support_penalty for 5 trades will be positive
    assert objectives[0, 1] >= 100.0
```

- [ ] **Step 2: Run pytest to verify the new test fails (TDD RED step)**
Run: `PYTHONPATH=. .venv/bin/pytest tests/unit/test_evox_runner.py -k test_low_trade_drawdown_penalty -v`
Expected: Fail because the drawdown objective is close to 0.0 + penalty.

- [ ] **Step 3: Modify `_evaluate_population_indices` in `gpu_fuzzy_trader/evolution/evox_runner.py`**
Change lines 479-482 to apply the `100.0` drawdown penalty for candidates with trade counts below the pool floor:

```python
            # Vectorized min-Hamming against pareto archive
            diversity_penalty = 0.0
            if pareto_archive:
                min_hamming = batch_hamming_min(chromosome, pareto_archive)
                if min_hamming <= _cfg.PHASE2_DIVERSITY_HAMMING_THRESHOLD:
                    diversity_penalty = _cfg.PHASE2_DIVERSITY_PENALTY

            pen = support_penalty + diversity_penalty + cond_penalty
            
            dd_val = max_dd
            trade_floor = _cfg.PHASE2_CV_MIN_TRADE_POOL_FLOOR if str(_cfg.SPLIT_MODE).strip().lower() == "purged_rolling_cv" else _cfg.MIN_TRADE_POOL_FLOOR
            if executed < trade_floor:
                dd_val = 100.0
                
            objectives[i] = np.array(
                [-sortino_for_obj + pen, dd_val + pen, -win_rate + pen],
                dtype=np.float64,
            )
```

- [ ] **Step 4: Run pytest to verify it passes**
Run: `PYTHONPATH=. .venv/bin/pytest tests/unit/test_evox_runner.py -k test_low_trade_drawdown_penalty -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add gpu_fuzzy_trader/evolution/evox_runner.py tests/unit/test_evox_runner.py
git commit -m "fix: apply drawdown penalty to low-trade rules in Phase 2 fitness"
```

---

### Task 3: Bypass Multi-Regime profitability check for Regime Specialists
**Files:**
- Modify: `gpu_fuzzy_trader/phases/phase2_support.py`
- Test: `tests/unit/test_regime_profitability_gate.py`

- [ ] **Step 1: Write test case to verify specialist gate bypass**
Add a test in `tests/unit/test_regime_profitability_gate.py` to ensure that regime specialists are only required to be profitable in their dominant regime:

```python
from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.phases.phase2_support import passes_pool_admission_gate

def test_regime_specialist_profitability_gate():
    orig_gate = _cfg.PHASE2_REGIME_PROFITABILITY_GATE
    try:
        _cfg.PHASE2_REGIME_PROFITABILITY_GATE = True
        
        # Candidate only profitable in regime 0
        train_metrics = {
            "executed_trades": 60,
            "total_return_pct": 5.0,
            "profit_factor": 1.2,
            "regime_net_pnl": [10.0, -2.0, -1.0],
            "regime_specialist": True,
            "dominant_regime": 0
        }
        val_metrics = {
            "executed_trades": 20,
            "total_return_pct": 2.0,
            "profit_factor": 1.1
        }
        
        # Should pass because it is a specialist in regime 0, which has positive PnL
        assert passes_pool_admission_gate(train_metrics, val_metrics)
        
        # Should fail if dominant regime itself has negative return
        train_metrics["dominant_regime"] = 1
        assert not passes_pool_admission_gate(train_metrics, val_metrics)
    finally:
        _cfg.PHASE2_REGIME_PROFITABILITY_GATE = orig_gate
```

- [ ] **Step 2: Run pytest to verify the new test fails (TDD RED step)**
Run: `PYTHONPATH=. .venv/bin/pytest tests/unit/test_regime_profitability_gate.py -k test_regime_specialist_profitability_gate -v`
Expected: Fail (specialist is evaluated with the multi-regime gate and rejected)

- [ ] **Step 3: Modify `_passes_pool_admission_impl` in `gpu_fuzzy_trader/phases/phase2_support.py`**
Update the Regime Profitability Gate check to bypass multi-regime checks if `regime_specialist` is true:

```python
    # Regime Profitability Gate
    if _cfg.PHASE2_REGIME_PROFITABILITY_GATE:
        regime_pnl = train_metrics.get("regime_net_pnl")
        if regime_pnl is not None:
            if train_metrics.get("regime_specialist", False):
                dom = int(train_metrics.get("dominant_regime", -1))
                if dom >= 0 and dom < len(regime_pnl):
                    if regime_pnl[dom] <= _cfg.PHASE2_REGIME_MIN_RETURN_PER_REGIME:
                        return False
            else:
                n_passing = sum(1 for p in regime_pnl if p > _cfg.PHASE2_REGIME_MIN_RETURN_PER_REGIME)
                min_pass = 2 if len(regime_pnl) >= 2 else len(regime_pnl)
                if n_passing < min_pass:
                    return False
```

- [ ] **Step 4: Run pytest to verify the test passes**
Run: `PYTHONPATH=. .venv/bin/pytest tests/unit/test_regime_profitability_gate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add gpu_fuzzy_trader/phases/phase2_support.py tests/unit/test_regime_profitability_gate.py
git commit -m "fix: bypass multi-regime profitability check for regime specialists"
```
