import pytest
pytest.importorskip("jax")

import jax.numpy as jnp
import numpy as np
import pandas as pd
from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.backtest.gpu_engine import _jax_simulate_equity_batch

def test_recency_weight_calculation():
    # Save original state
    orig_enabled = _cfg.PHASE2_RECENCY_WEIGHT_ENABLED
    try:
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
        _jax_simulate_equity_batch.clear_cache()
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
    finally:
        _cfg.PHASE2_RECENCY_WEIGHT_ENABLED = orig_enabled


