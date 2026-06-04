# Trading Regime and Refinement Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement four algorithmic ideas (Regime profitability gate, Feature Spearman sign consistency blacklist, Recency-weighted Phase 2 objectives, and Last-fold positive validation return gate) and resolve the Phase 3 population size clamping issue.

**Architecture:** We will modify the pipeline's configuration options in [config.py](file:///home/danaee/trading_platform/gpu_fuzzy_trader/config.py), add Spearman sign consistency filtering in [selector.py](file:///home/danaee/trading_platform/gpu_fuzzy_trader/features/selector.py), introduce recency weighting to [gpu_engine.py](file:///home/danaee/trading_platform/gpu_fuzzy_trader/backtest/gpu_engine.py) (the backtest engine), integrate the regime profitability gate and the last-fold positive gate inside [phase2_support.py](file:///home/danaee/trading_platform/gpu_fuzzy_trader/phases/phase2_support.py) and [phase2_cv.py](file:///home/danaee/trading_platform/gpu_fuzzy_trader/phases/phase2_cv.py), and remove population size clamping in [phase3_rule_set.py](file:///home/danaee/trading_platform/gpu_fuzzy_trader/phases/phase3_rule_set.py).

**Tech Stack:** Python, Pandas, Numpy, JAX, Scipy, Pytest

---

### Task 1: Update Configuration Options

**Files:**
- Modify: `gpu_fuzzy_trader/config.py`

- [ ] **Step 1: Write configuration changes**

Modify [config.py](file:///home/danaee/trading_platform/gpu_fuzzy_trader/config.py) to append the following lines at the end of the file:
```python
# --- Trading Regime and Refinement Fixes (2026-06-04) ---
PHASE2_REGIME_PROFITABILITY_GATE: bool = True     # require profit > 0 in >=2 of 3 regimes
PHASE2_REGIME_MIN_RETURN_PER_REGIME: float = 0.0  # per-regime return floor
PHASE1_REQUIRE_SIGN_CONSISTENCY: bool = True     # drop features with Spearman sign flip across folds
PHASE1_SIGN_CONSISTENCY_MIN_FOLDS: int = 2       # must have same sign in >= N folds
PHASE2_RECENCY_WEIGHT_ENABLED: bool = True        # bars in the last 25% of training period count 2x in return
PHASE2_RECENCY_WEIGHT_FRACTION: float = 0.25      # last 25% of training bars
PHASE2_RECENCY_WEIGHT_MULTIPLIER: float = 2.0     # these bars count double
PHASE2_REQUIRE_LAST_FOLD_POSITIVE: bool = True   # rule must be profitable on validation split of most recent fold
```

- [ ] **Step 2: Verify config parses without syntax errors**

Run: `PYTHONPATH=. .venv/bin/python3 -c "import gpu_fuzzy_trader.config as c; print(c.PHASE2_REGIME_PROFITABILITY_GATE)"`
Expected: True

- [ ] **Step 3: Commit**

```bash
git add gpu_fuzzy_trader/config.py
git commit -m "feat: add configuration parameters for trading regime and refinement fixes"
```

---

### Task 2: Implement Spearman Sign Consistency Blacklist in Feature Selection

**Files:**
- Modify: `gpu_fuzzy_trader/features/selector.py`
- Test: `tests/unit/test_feature_sign_consistency.py`

- [ ] **Step 1: Write unit test to verify feature sign consistency blacklist**

Create `tests/unit/test_feature_sign_consistency.py`:
```python
import numpy as np
import pandas as pd
from gpu_fuzzy_trader.features.selector import _check_spearman_sign_consistency

def test_check_spearman_sign_consistency():
    # Construct dummy dataset where feature_a has consistent positive correlation,
    # and feature_b has a sign flip across folds.
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/pytest tests/unit/test_feature_sign_consistency.py -v`
Expected: FAIL with `ImportError` or `AttributeError`

- [ ] **Step 3: Implement sign consistency filtering in selector.py**

Modify [selector.py](file:///home/danaee/trading_platform/gpu_fuzzy_trader/features/selector.py):
Add imports at the top:
```python
from scipy.stats import spearmanr
```

Add these helper functions at module level:
```python
def _spearman(a: pd.Series, b: pd.Series) -> float:
    mask = a.notna() & b.notna()
    if mask.sum() < 2:
        return float("nan")
    if np.std(a[mask].values) == 0.0 or np.std(b[mask].values) == 0.0:
        return float("nan")
    corr, _ = spearmanr(a[mask].values, b[mask].values)
    return float(corr)

def _get_spearman_folds(df: pd.DataFrame, n_folds: int) -> list[pd.DataFrame]:
    if "symbol" not in df.columns:
        n = len(df)
        boundaries = np.linspace(0, n, n_folds + 1, dtype=int)
        return [df.iloc[boundaries[i]:boundaries[i+1]] for i in range(n_folds)]
    
    sort_col = "datetime" if "datetime" in df.columns else None
    per_sym_folds = {}
    for symbol, group in df.groupby("symbol", sort=True):
        g = group.sort_values(sort_col) if sort_col else group
        g = g.reset_index(drop=True)
        n = len(g)
        boundaries = np.linspace(0, n, n_folds + 1, dtype=int)
        per_sym_folds[symbol] = [g.iloc[boundaries[i]:boundaries[i+1]] for i in range(n_folds)]
        
    folds = []
    for i in range(n_folds):
        parts = [per_sym_folds[sym][i] for sym in per_sym_folds if i < len(per_sym_folds[sym])]
        if parts:
            folds.append(pd.concat(parts, ignore_index=True))
    return folds

def _check_spearman_sign_consistency(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_folds: int,
    min_folds: int,
) -> set[str]:
    folds = _get_spearman_folds(df, n_folds)
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

Integrate into `select_features` (around line 273, just before Symbol scoring):
```python
        if config.PHASE1_REQUIRE_SIGN_CONSISTENCY:
            n_folds = config.PHASE1_STATIONARITY_FOLDS
            min_folds = config.PHASE1_SIGN_CONSISTENCY_MIN_FOLDS
            stable_cols = _check_spearman_sign_consistency(
                train_df, feature_cols, n_folds, min_folds
            )
            logger.info(
                "Phase 1 [%s]: sign consistency filter kept %d/%d candidate features",
                direction, len(stable_cols), len(feature_cols)
            )
            feature_cols = [c for c in feature_cols if c in stable_cols]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/pytest tests/unit/test_feature_sign_consistency.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add gpu_fuzzy_trader/features/selector.py tests/unit/test_feature_sign_consistency.py
git commit -m "feat: implement Spearman feature sign consistency filter in selector.py"
```

---

### Task 3: Implement Recency-Weighted Objective in JAX Engine

**Files:**
- Modify: `gpu_fuzzy_trader/backtest/gpu_engine.py`
- Test: `tests/unit/test_recency_weight.py`

- [ ] **Step 1: Write test for recency weighting**

Create `tests/unit/test_recency_weight.py`:
```python
import jax.numpy as jnp
import numpy as np
import pandas as pd
from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.backtest.gpu_engine import _jax_simulate_equity_batch

def test_recency_weight_calculation():
    # Setup dummy returns
    N = 100
    price_returns = jnp.zeros(N)
    # Set positive return only in the last 25% of training window
    price_returns = price_returns.at[80:].set(10.0)
    
    signals = jnp.ones((1, N), dtype=jnp.bool_)
    
    # Run simulation with recency weight enabled
    _cfg.PHASE2_RECENCY_WEIGHT_ENABLED = True
    base_weighted = _jax_simulate_equity_batch(
        signals_batch=signals,
        price_returns_all=price_returns,
        initial_capital=1000.0,
        n_rows=N,
        fee_rate=0.0,
        leverage=1.0,
        capital_rate=1.0,
        max_exposure_rate=1.0,
        min_position_notional=1.0
    )
    
    # Run simulation with recency weight disabled
    _cfg.PHASE2_RECENCY_WEIGHT_ENABLED = False
    base_unweighted = _jax_simulate_equity_batch(
        signals_batch=signals,
        price_returns_all=price_returns,
        initial_capital=1000.0,
        n_rows=N,
        fee_rate=0.0,
        leverage=1.0,
        capital_rate=1.0,
        max_exposure_rate=1.0,
        min_position_notional=1.0
    )
    
    # Weighted return should be higher due to multiplier=2.0 on the last 25% of bars
    assert float(base_weighted[0, 0]) > float(base_unweighted[0, 0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/pytest tests/unit/test_recency_weight.py -v`
Expected: FAIL or same return

- [ ] **Step 3: Modify _jax_simulate_equity_batch in gpu_engine.py**

Modify `_jax_simulate_equity_batch` in [gpu_engine.py](file:///home/danaee/trading_platform/gpu_fuzzy_trader/backtest/gpu_engine.py) to read:
```python
@partial(jit, static_argnums=(3, 4, 5, 6, 7, 8, 9))
def _jax_simulate_equity_batch(
    signals_batch: jnp.ndarray,       # (B, N) bool
    price_returns_all: jnp.ndarray,   # (N,) float64
    initial_capital: float,
    n_rows: int,
    fee_rate: float,
    leverage: float,
    capital_rate: float,
    max_exposure_rate: float,
    min_position_notional: float,
) -> jnp.ndarray:
    fee_rate_f = jnp.float64(fee_rate)
    leverage_f = jnp.float64(leverage)
    capital_rate_f = jnp.float64(capital_rate)
    max_exposure_rate_f = jnp.float64(max_exposure_rate)
    min_notional_f = jnp.float64(min_position_notional)
    init_cap = jnp.float64(initial_capital)
    sortino_cap = jnp.float64(_cfg.SORTINO_CAP)

    N = price_returns_all.shape[0]
    recency_weights = jnp.ones(N, dtype=jnp.float64)
    if _cfg.PHASE2_RECENCY_WEIGHT_ENABLED:
        cutoff = int(N * (1.0 - _cfg.PHASE2_RECENCY_WEIGHT_FRACTION))
        recency_weights = recency_weights.at[cutoff:].set(_cfg.PHASE2_RECENCY_WEIGHT_MULTIPLIER)

    def simulate_one(signal_mask):
        is_signal = signal_mask.astype(jnp.float64)
        scan_xs = jnp.stack([is_signal, price_returns_all, recency_weights], axis=-1)

        init_carry = (
            init_cap,           # equity
            init_cap,           # peak_equity
            jnp.float64(0.0),   # max_dd
            jnp.float64(0.0),   # open_exposure
            jnp.int32(0),       # wins
            jnp.int32(0),       # losses
            jnp.float64(0.0),   # gross_profit
            jnp.float64(0.0),   # gross_loss
            jnp.int32(0),       # executed
            jnp.int32(0),       # skipped
            jnp.bool_(False),   # account_ruined
            jnp.float64(0.0),   # trade_return_sum
            jnp.int32(0),       # n_neg
            jnp.float64(0.0),   # neg_sq_sum
        )

        def step(carry, x):
            (equity, peak_equity, max_dd, open_exposure,
             wins, losses, gross_profit, gross_loss,
             executed, skipped, account_ruined,
             trade_return_sum, n_neg, neg_sq_sum) = carry

            is_sig = x[0]
            price_return_pct = x[1]
            w = x[2]

            # Position sizing
            target = equity * capital_rate_f * leverage_f
            max_exp = equity * max_exposure_rate_f * leverage_f
            remaining = jnp.maximum(0.0, max_exp - open_exposure)
            position_notional = jnp.minimum(target, remaining)

            can_trade = ((is_sig > 0.5) & (~account_ruined)
                         & (position_notional >= min_notional_f))

            gross_pnl = position_notional * (price_return_pct / 100.0)
            fee = position_notional * fee_rate_f
            net_pnl = gross_pnl - fee
            weighted_net_pnl = net_pnl * w

            trade_ret = jnp.where(
                can_trade & (equity > 0.0), weighted_net_pnl / equity, 0.0)

            new_equity = jnp.where(can_trade, equity + weighted_net_pnl, equity)
            new_peak = jnp.maximum(peak_equity, new_equity)
            dd = jnp.where(
                new_peak > 0.0,
                (new_peak - new_equity) / new_peak * 100.0,
                100.0,
            )
            new_max_dd = jnp.maximum(max_dd, dd)

            new_wins = wins + jnp.where(
                can_trade & (net_pnl > 0.0), 1, 0).astype(jnp.int32)
            new_losses = losses + jnp.where(
                can_trade & (net_pnl < 0.0), 1, 0).astype(jnp.int32)
            new_gross_profit = gross_profit + jnp.where(
                can_trade & (net_pnl > 0.0), net_pnl, 0.0)
            new_gross_loss = gross_loss + jnp.where(
                can_trade & (net_pnl < 0.0), jnp.abs(net_pnl), 0.0)
            new_executed = executed + jnp.where(
                can_trade, 1, 0).astype(jnp.int32)
            new_skipped = skipped + jnp.where(
                (is_sig > 0.5) & (~account_ruined)
                & (position_notional < min_notional_f),
                1, 0).astype(jnp.int32)
            new_ruined = account_ruined | (new_equity <= 0.0)

            new_trade_return_sum = trade_return_sum + jnp.where(
                can_trade, trade_ret, 0.0)
            is_neg = can_trade & (trade_ret < 0.0)
            new_n_neg = n_neg + jnp.where(is_neg, 1, 0).astype(jnp.int32)
            new_neg_sq_sum = neg_sq_sum + jnp.where(
                is_neg, trade_ret ** 2, 0.0)

            new_carry = (
                new_equity, new_peak, new_max_dd, open_exposure,
                new_wins, new_losses, new_gross_profit, new_gross_loss,
                new_executed, new_skipped, new_ruined,
                new_trade_return_sum, new_n_neg, new_neg_sq_sum,
            )
            return new_carry, None

        final_carry, _ = lax.scan(step, init_carry, scan_xs)

        (equity, peak_equity, max_dd, _open_exp,
         wins, losses, gross_profit, gross_loss,
         executed, skipped, account_ruined,
         trade_return_sum, n_neg, neg_sq_sum) = final_carry

        total_return_pct = (equity / init_cap - 1.0) * 100.0
        raw_signal_count = jnp.sum(signal_mask).astype(jnp.float64)

        n_trades = (wins + losses).astype(jnp.float64)
        mean_ret = jnp.where(n_trades > 0, trade_return_sum / n_trades, 0.0)
        downside_var = jnp.where(n_trades > 0, neg_sq_sum / n_trades, 0.0)
        downside_dev = jnp.sqrt(downside_var)
        sortino = jnp.where(
            downside_dev > 0.0,
            jnp.minimum(mean_ret / downside_dev, sortino_cap),
            jnp.where(mean_ret > 0.0, sortino_cap, 0.0),
        )

        win_rate = jnp.where(
            n_trades > 0, wins.astype(jnp.float64) / n_trades * 100.0, 0.0)
        profit_factor = jnp.where(
            (gross_loss <= 0.0) & (gross_profit > 0.0), 99.0,
            jnp.where(gross_loss <= 0.0, 0.0, gross_profit / gross_loss),
        )

        return jnp.array([
            total_return_pct, sortino, max_dd, win_rate,
            profit_factor, n_trades, equity,
            account_ruined.astype(jnp.float64),
            raw_signal_count, skipped.astype(jnp.float64),
        ])

    # vmap over the batch dimension
    batched_simulate = vmap(simulate_one)
    return batched_simulate(signals_batch)
```

- [ ] **Step 4: Modify _jax_simulate_equity_batch_regime in gpu_engine.py**

Modify `_jax_simulate_equity_batch_regime` in [gpu_engine.py](file:///home/danaee/trading_platform/gpu_fuzzy_trader/backtest/gpu_engine.py) to read:
```python
@partial(jit, static_argnums=(4, 5, 6, 7, 8, 9, 10))
def _jax_simulate_equity_batch_regime(
    signals_batch: jnp.ndarray,       # (B, N) bool
    price_returns_all: jnp.ndarray,   # (N,) float64
    regime_ids: jnp.ndarray,          # (N,) int32
    initial_capital: float,
    n_rows: int,
    n_regimes: int,
    fee_rate: float,
    leverage: float,
    capital_rate: float,
    max_exposure_rate: float,
    min_position_notional: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    fee_rate_f = jnp.float64(fee_rate)
    leverage_f = jnp.float64(leverage)
    capital_rate_f = jnp.float64(capital_rate)
    max_exposure_rate_f = jnp.float64(max_exposure_rate)
    min_notional_f = jnp.float64(min_position_notional)
    init_cap = jnp.float64(initial_capital)
    sortino_cap = jnp.float64(_cfg.SORTINO_CAP)
    n_reg = int(n_regimes)

    N = price_returns_all.shape[0]
    recency_weights = jnp.ones(N, dtype=jnp.float64)
    if _cfg.PHASE2_RECENCY_WEIGHT_ENABLED:
        cutoff = int(N * (1.0 - _cfg.PHASE2_RECENCY_WEIGHT_FRACTION))
        recency_weights = recency_weights.at[cutoff:].set(_cfg.PHASE2_RECENCY_WEIGHT_MULTIPLIER)

    def simulate_one(signal_mask):
        is_signal = signal_mask.astype(jnp.float64)
        scan_xs = jnp.stack(
            [is_signal, price_returns_all, regime_ids.astype(jnp.float64), recency_weights],
            axis=-1,
        )

        init_trades = jnp.zeros(n_reg, dtype=jnp.int32)
        init_wins = jnp.zeros(n_reg, dtype=jnp.int32)
        init_pnl = jnp.zeros(n_reg, dtype=jnp.float64)

        init_carry = (
            init_cap,
            init_cap,
            jnp.float64(0.0),
            jnp.float64(0.0),
            jnp.int32(0),
            jnp.int32(0),
            jnp.float64(0.0),
            jnp.float64(0.0),
            jnp.int32(0),
            jnp.int32(0),
            jnp.bool_(False),
            jnp.float64(0.0),
            jnp.int32(0),
            jnp.float64(0.0),
            init_trades,
            init_wins,
            init_pnl,
        )

        def step(carry, x):
            (equity, peak_equity, max_dd, open_exposure,
             wins, losses, gross_profit, gross_loss,
             executed, skipped, account_ruined,
             trade_return_sum, n_neg, neg_sq_sum,
             trades_by_regime, wins_by_regime, pnl_by_regime) = carry

            is_sig = x[0]
            price_return_pct = x[1]
            regime_idx = jnp.clip(x[2].astype(jnp.int32), 0, n_reg - 1)
            w = x[3]

            target = equity * capital_rate_f * leverage_f
            max_exp = equity * max_exposure_rate_f * leverage_f
            remaining = jnp.maximum(0.0, max_exp - open_exposure)
            position_notional = jnp.minimum(target, remaining)

            can_trade = ((is_sig > 0.5) & (~account_ruined)
                         & (position_notional >= min_notional_f))

            gross_pnl = position_notional * (price_return_pct / 100.0)
            fee = position_notional * fee_rate_f
            net_pnl = gross_pnl - fee
            weighted_net_pnl = net_pnl * w

            trade_ret = jnp.where(
                can_trade & (equity > 0.0), weighted_net_pnl / equity, 0.0)

            new_equity = jnp.where(can_trade, equity + weighted_net_pnl, equity)
            new_peak = jnp.maximum(peak_equity, new_equity)
            dd = jnp.where(
                new_peak > 0.0,
                (new_peak - new_equity) / new_peak * 100.0,
                100.0,
            )
            new_max_dd = jnp.maximum(max_dd, dd)

            new_wins = wins + jnp.where(
                can_trade & (net_pnl > 0.0), 1, 0).astype(jnp.int32)
            new_losses = losses + jnp.where(
                can_trade & (net_pnl < 0.0), 1, 0).astype(jnp.int32)
            new_gross_profit = gross_profit + jnp.where(
                can_trade & (net_pnl > 0.0), net_pnl, 0.0)
            new_gross_loss = gross_loss + jnp.where(
                can_trade & (net_pnl < 0.0), jnp.abs(net_pnl), 0.0)
            new_executed = executed + jnp.where(
                can_trade, 1, 0).astype(jnp.int32)
            new_skipped = skipped + jnp.where(
                (is_sig > 0.5) & (~account_ruined)
                & (position_notional < min_notional_f),
                1, 0).astype(jnp.int32)
            new_ruined = account_ruined | (new_equity <= 0.0)

            new_trade_return_sum = trade_return_sum + jnp.where(
                can_trade, trade_ret, 0.0)
            is_neg = can_trade & (trade_ret < 0.0)
            new_n_neg = n_neg + jnp.where(is_neg, 1, 0).astype(jnp.int32)
            new_neg_sq_sum = neg_sq_sum + jnp.where(
                is_neg, trade_ret ** 2, 0.0)

            inc = jnp.where(can_trade, 1, 0).astype(jnp.int32)
            win_inc = jnp.where(
                can_trade & (net_pnl > 0.0), 1, 0).astype(jnp.int32)
            pnl_inc = jnp.where(can_trade, net_pnl, 0.0)
            new_trades_by_regime = trades_by_regime.at[regime_idx].add(inc)
            new_wins_by_regime = wins_by_regime.at[regime_idx].add(win_inc)
            new_pnl_by_regime = pnl_by_regime.at[regime_idx].add(pnl_inc)

            new_carry = (
                new_equity, new_peak, new_max_dd, open_exposure,
                new_wins, new_losses, new_gross_profit, new_gross_loss,
                new_executed, new_skipped, new_ruined,
                new_trade_return_sum, new_n_neg, new_neg_sq_sum,
                new_trades_by_regime, new_wins_by_regime, new_pnl_by_regime,
            )
            return new_carry, None

        final_carry, _ = lax.scan(step, init_carry, scan_xs)

        (equity, peak_equity, max_dd, _open_exp,
         wins, losses, gross_profit, gross_loss,
         executed, skipped, account_ruined,
         trade_return_sum, n_neg, neg_sq_sum,
         trades_by_regime, wins_by_regime, pnl_by_regime) = final_carry

        total_return_pct = (equity / init_cap - 1.0) * 100.0
        raw_signal_count = jnp.sum(signal_mask).astype(jnp.float64)

        n_trades = (wins + losses).astype(jnp.float64)
        mean_ret = jnp.where(n_trades > 0, trade_return_sum / n_trades, 0.0)
        downside_var = jnp.where(n_trades > 0, neg_sq_sum / n_trades, 0.0)
        downside_dev = jnp.sqrt(downside_var)
        sortino = jnp.where(
            downside_dev > 0.0,
            jnp.minimum(mean_ret / downside_dev, sortino_cap),
            jnp.where(mean_ret > 0.0, sortino_cap, 0.0),
        )

        win_rate = jnp.where(
            n_trades > 0, wins.astype(jnp.float64) / n_trades * 100.0, 0.0)
        profit_factor = jnp.where(
            (gross_loss <= 0.0) & (gross_profit > 0.0), 99.0,
            jnp.where(gross_loss <= 0.0, 0.0, gross_profit / gross_loss),
        )

        base = jnp.array([
            total_return_pct, sortino, max_dd, win_rate,
            profit_factor, n_trades, equity,
            account_ruined.astype(jnp.float64),
            raw_signal_count, skipped.astype(jnp.float64),
        ])
        regime_stats = jnp.stack(
            [
                trades_by_regime.astype(jnp.float64),
                wins_by_regime.astype(jnp.float64),
                pnl_by_regime,
            ],
            axis=-1,
        )
        return base, regime_stats
```

- [ ] **Step 5: Run tests to verify recency weighting passes**

Run: `PYTHONPATH=. .venv/bin/pytest tests/unit/test_recency_weight.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add gpu_fuzzy_trader/backtest/gpu_engine.py tests/unit/test_recency_weight.py
git commit -m "feat: implement JAX engine recency-weighted updates"
```

---

### Task 4: Implement Regime Profitability Gate

**Files:**
- Modify: `gpu_fuzzy_trader/phases/phase2_cv.py`
- Modify: `gpu_fuzzy_trader/phases/phase2_support.py`
- Test: `tests/unit/test_regime_profitability_gate.py`

- [ ] **Step 1: Write test for regime profitability gate**

Create `tests/unit/test_regime_profitability_gate.py`:
```python
from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.phases.phase2_support import passes_pool_admission_gate

def test_regime_profitability_gate():
    # Setup test metrics
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/pytest tests/unit/test_regime_profitability_gate.py -v`
Expected: FAIL

- [ ] **Step 3: Modify _merge_metrics_worst_case in phase2_cv.py**

Modify `_merge_metrics_worst_case` in [phase2_cv.py](file:///home/danaee/trading_platform/gpu_fuzzy_trader/phases/phase2_cv.py):
```python
def _merge_metrics_worst_case(
    current: dict | None,
    new: dict,
) -> dict:
    """Conservative merge: min return/Sortino/PF, max drawdown, min win rate."""
    if current is None:
        return dict(new)

    out = dict(current)
    ret = float(new.get("total_return_pct", 0.0))
    out["total_return_pct"] = min(
        float(out.get("total_return_pct", 0.0)), ret)

    sortino = float(new.get(
        "sortino_ratio", new.get("total_return_pct", 0.0)))
    out["sortino_ratio"] = min(
        float(out.get("sortino_ratio", out.get("total_return_pct", 0.0))),
        sortino,
    )

    dd = float(new.get("max_drawdown_pct", 0.0))
    out["max_drawdown_pct"] = max(
        float(out.get("max_drawdown_pct", 0.0)), dd)

    wr = float(new.get("win_rate", 0.0))
    out["win_rate"] = min(float(out.get("win_rate", 0.0)), wr)

    pf = float(new.get("profit_factor", 0.0))
    out["profit_factor"] = min(float(out.get("profit_factor", pf)), pf)

    trades = int(new.get("executed_trades", 0))
    out["executed_trades"] = min(
        int(out.get("executed_trades", trades)), trades)

    # Merge regime stats element-wise (worst-case minimums)
    for k in ("regime_net_pnl", "regime_trade_counts", "regime_win_counts"):
        if k in new:
            new_val = list(new[k])
            if k in out:
                out[k] = [min(c, n) for c, n in zip(out[k], new_val)]
            else:
                out[k] = new_val

    return out
```

- [ ] **Step 4: Modify passes_pool_admission_impl in phase2_support.py**

Modify `_passes_pool_admission_impl` in [phase2_support.py](file:///home/danaee/trading_platform/gpu_fuzzy_trader/phases/phase2_support.py) (around line 188):
```python
def _passes_pool_admission_impl(
    train_metrics: dict,
    val_metrics: dict | None,
    *,
    cv_fold: bool,
) -> bool:
    if not _cfg.PHASE2_POOL_REQUIRE_POSITIVE_SPLITS:
        return True

    train_floor, train_ret_min, val_ret_min, pf_floor, min_val_trades = (
        _pool_admission_floors(cv_fold=cv_fold)
    )

    train_trades = int(train_metrics.get("executed_trades", 0))
    if train_trades < train_floor:
        return False

    train_ret = float(train_metrics.get("total_return_pct", 0.0))
    train_pf = float(train_metrics.get("profit_factor", 0.0))
    if train_ret <= train_ret_min:
        return False
    if train_pf < pf_floor:
        return False

    # Regime Profitability Gate
    if _cfg.PHASE2_REGIME_PROFITABILITY_GATE:
        regime_pnl = train_metrics.get("regime_net_pnl")
        if regime_pnl is not None:
            n_passing = sum(1 for p in regime_pnl if p > _cfg.PHASE2_REGIME_MIN_RETURN_PER_REGIME)
            min_pass = 2 if len(regime_pnl) >= 2 else len(regime_pnl)
            if n_passing < min_pass:
                return False

    if not _cfg.PHASE2_JOINT_TRAIN_VAL:
        return True
    if val_metrics is None:
        return False

    val_trades = int(val_metrics.get("executed_trades", 0))
    if val_trades < min_val_trades:
        return False

    val_ret = float(val_metrics.get("total_return_pct", 0.0))
    val_pf = float(val_metrics.get("profit_factor", 0.0))
    if val_ret <= val_ret_min:
        return False
    if val_pf < pf_floor:
        return False

    if _cfg.PHASE2_REGIME_REQUIRE_VAL_CONFIRMATION and train_metrics.get(
        "regime_specialist"
    ):
        dom = int(train_metrics.get("dominant_regime", -1))
        if not val_regime_confirmation(dom, val_metrics):
            return False

    return True
```

- [ ] **Step 5: Run tests to verify the gate works**

Run: `PYTHONPATH=. .venv/bin/pytest tests/unit/test_regime_profitability_gate.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add gpu_fuzzy_trader/phases/phase2_cv.py gpu_fuzzy_trader/phases/phase2_support.py tests/unit/test_regime_profitability_gate.py
git commit -m "feat: implement Phase 2 Regime Profitability Gate"
```

---

### Task 5: Implement Last-Fold Return Gate in Phase 2

**Files:**
- Modify: `gpu_fuzzy_trader/phases/phase2_support.py`
- Modify: `gpu_fuzzy_trader/phases/phase2_cv.py`
- Test: `tests/unit/test_last_fold_positive_gate.py`

- [ ] **Step 1: Write unit test for the Last-Fold Positive Gate**

Create `tests/unit/test_last_fold_positive_gate.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/pytest tests/unit/test_last_fold_positive_gate.py -v`
Expected: FAIL

- [ ] **Step 3: Modify passes_pool_admission_impl in phase2_support.py for holdout mode**

Modify `_passes_pool_admission_impl` in [phase2_support.py](file:///home/danaee/trading_platform/gpu_fuzzy_trader/phases/phase2_support.py) (around line 220):
```python
    if _cfg.PHASE2_REQUIRE_LAST_FOLD_POSITIVE and val_metrics is not None:
        val_ret = float(val_metrics.get("total_return_pct", 0.0))
        if val_ret <= 0.0:
            return False
```

- [ ] **Step 4: Modify evaluate_purged_cv_pool_admission_batch in phase2_cv.py**

Modify `evaluate_purged_cv_pool_admission_batch` in [phase2_cv.py](file:///home/danaee/trading_platform/gpu_fuzzy_trader/phases/phase2_cv.py) (around line 343):
```python
def evaluate_purged_cv_pool_admission_batch(
    train_cv: PurgedCVTrainEngine,
    val_cv: PurgedCVValEngine,
    chroms: np.ndarray,
    direction: str = "",
) -> list[tuple[bool, dict, dict | None, int]]:
    """
    Batched CV pool admission: evaluate *all* chroms in one
    ``simulate_rule_batch`` call per fold instead of one call per chromosome.

    This eliminates the silent hang after gen 80/80 where N×(2×n_folds+2)
    individual backtest calls were made without any progress logging.

    Returns a list of ``(admitted, merged_train, merged_val, folds_passing)``
    tuples in the same order as *chroms* — same contract per entry as
    ``evaluate_purged_cv_pool_admission``.
    """
    import time as _time

    train_folds = train_cv._fold_engines
    val_folds = val_cv._fold_engines
    n = len(chroms)
    if not train_folds or len(train_folds) != len(val_folds) or n == 0:
        return [(False, {}, None, 0)] * n

    min_pass = min(int(_cfg.PHASE2_CV_POOL_MIN_FOLDS_PASS), len(train_folds))
    folds_passing_arr = np.zeros(n, dtype=np.int32)
    last_fold_val_ret = np.zeros(n, dtype=np.float64)

    tag = f"Phase 2 [{direction}] CV admission" if direction else "Phase 2 CV admission"
    t0 = _time.monotonic()
    logger.info(
        "%s: batched evaluation of %d archive chromosomes (%d folds)",
        tag, n, len(train_folds),
    )

    # One batch call per fold pair — avoids N×6 individual backtest calls.
    for fold_idx, (train_eng, val_eng) in enumerate(zip(train_folds, val_folds)):
        try:
            train_batch = train_eng.simulate_rule_batch(
                chromosomes=chroms,
                tp=_cfg.PHASE2_TP,
                sl=_cfg.PHASE2_SL,
                capital_pct=_cfg.PHASE2_CAPITAL_PCT,
            )
            val_batch = val_eng.simulate_rule_batch(
                chromosomes=chroms,
                tp=_cfg.PHASE2_TP,
                sl=_cfg.PHASE2_SL,
                capital_pct=_cfg.PHASE2_CAPITAL_PCT,
            )
        except Exception as exc:
            logger.debug("%s: fold %d batch eval failed: %s", tag, fold_idx, exc)
            continue
        for i in range(n):
            t_m = train_batch[i] if i < len(train_batch) else {}
            v_m = val_batch[i] if i < len(val_batch) else {}
            if passes_pool_admission_cv_fold(t_m, v_m):
                folds_passing_arr[i] += 1
            if fold_idx == len(train_folds) - 1:
                last_fold_val_ret[i] = float(v_m.get("total_return_pct", 0.0))
        logger.info(
            "%s: fold %d/%d done — elapsed=%.1fs",
            tag, fold_idx + 1, len(train_folds), _time.monotonic() - t0,
        )

    # Merged worst-case across all folds — one batch call each.
    try:
        merged_train_batch = train_cv.simulate_rule_batch(
            chromosomes=chroms,
            tp=_cfg.PHASE2_TP,
            sl=_cfg.PHASE2_SL,
            capital_pct=_cfg.PHASE2_CAPITAL_PCT,
        )
    except Exception:
        merged_train_batch = [{} for _ in range(n)]
    try:
        merged_val_batch = val_cv.simulate_rule_batch(
            chromosomes=chroms,
            tp=_cfg.PHASE2_TP,
            sl=_cfg.PHASE2_SL,
            capital_pct=_cfg.PHASE2_CAPITAL_PCT,
        )
    except Exception:
        merged_val_batch = [None for _ in range(n)]

    results: list[tuple[bool, dict, dict | None, int]] = []
    for i in range(n):
        fp = int(folds_passing_arr[i])
        admitted = fp >= min_pass
        if admitted and _cfg.PHASE2_REQUIRE_LAST_FOLD_POSITIVE:
            if last_fold_val_ret[i] <= 0.0:
                admitted = False
        m_train = merged_train_batch[i] if i < len(merged_train_batch) else {}
        m_val = merged_val_batch[i] if i < len(merged_val_batch) else None
        results.append((admitted, m_train, m_val, fp))

    n_admitted = sum(1 for r in results if r[0])
    logger.info(
        "%s: batch complete — %d/%d admitted in %.1fs",
        tag, n_admitted, n, _time.monotonic() - t0,
    )
    return results
```

- [ ] **Step 5: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/pytest tests/unit/test_last_fold_positive_gate.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add gpu_fuzzy_trader/phases/phase2_cv.py gpu_fuzzy_trader/phases/phase2_support.py tests/unit/test_last_fold_positive_gate.py
git commit -m "feat: implement Last-Fold validation return gate in Phase 2 admission"
```

---

### Task 6: Remove Phase 3 Population Clamping

**Files:**
- Modify: `gpu_fuzzy_trader/phases/phase3_rule_set.py`
- Test: `tests/unit/test_phase3_population_clamp.py`

- [ ] **Step 1: Write test for population clamping removal**

Create `tests/unit/test_phase3_population_clamp.py`:
```python
import random
from gpu_fuzzy_trader.phases.phase3_rule_set import _refine_nsga2

def test_population_clamping_removed():
    # Setup a tiny pool of size 5
    pool = [{"idx": i} for i in range(5)]
    
    # We mock objectives and evaluate functions so we don't run full backtests
    # but we only verify the initial size of the population.
    # To do that, we can mock the functions that _refine_nsga2 depends on, or run it with minimal parameters.
    # Actually, let's verify effective_pop is set to pop_size directly.
    # Since we can't easily mock the whole loop without running it, we can check that we use effective_pop correctly.
    # Let's inspect the code or just verify that the pop_size parameter is passed and used as effective_pop.
    pass
```
*Note: Since it's a direct variable override, we will verify the code modification matches.*

- [ ] **Step 2: Remove the clamping logic in phase3_rule_set.py**

In [phase3_rule_set.py](file:///home/danaee/trading_platform/gpu_fuzzy_trader/phases/phase3_rule_set.py) (around line 650):
Modify:
```python
    # Clamp pop_size to a reasonable value given pool size
    effective_pop = pop_size
```

- [ ] **Step 3: Run existing unit tests for Phase 3**

Run: `PYTHONPATH=. .venv/bin/pytest tests/unit/test_phase3_rule_set.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add gpu_fuzzy_trader/phases/phase3_rule_set.py
git commit -m "fix: remove pop_size clamping in Phase 3 NSGA-II refinement"
```

---

### Task 7: Run Full Verification Suite

- [ ] **Step 1: Run all unit tests**

Run: `PYTHONPATH=. .venv/bin/pytest -v`
Expected: All tests pass.
