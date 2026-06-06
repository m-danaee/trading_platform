"""Phase 2 GPU throughput benchmark (non-gating).

Times rule matching, equity scan, and full simulate_rule_batch with
``jax.block_until_ready`` for honest GPU measurements.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.backtest.gpu_engine import (
    GPUBacktestEngine,
    _jax_compute_rule_signals_batch,
    _jax_simulate_equity_batch,
)
from gpu_fuzzy_trader.backtest.jax_compat import get_gpu_backtest_engine_class
from gpu_fuzzy_trader._gpu_runtime import resolve_phase2_gpu_batch_size


def _make_benchmark_df(n: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "symbol": ["SYM"] * n,
        "datetime": pd.date_range("2024-01-01", periods=n, freq="5min"),
        "_symbol_bar_index": np.arange(n),
        "label_open_next": rng.uniform(90, 110, n),
        "label_max_288": rng.uniform(92, 115, n),
        "label_min_288": rng.uniform(85, 98, n),
        "label_close_288": rng.uniform(90, 110, n),
        "label_max_before_min": rng.integers(0, 2, n),
        "feat_a": rng.integers(0, 2, n),
        "feat_b": rng.integers(0, 5, n),
        "feat_c": rng.integers(0, 3, n),
    })


def _gpu_available() -> bool:
    GPUBacktestEngineCls = get_gpu_backtest_engine_class()
    if GPUBacktestEngineCls is None:
        return False
    try:
        import jax
        return jax.default_backend() == "gpu"
    except Exception:
        return False


def _timed_gpu(fn, *args, warmup: int = 2, repeats: int = 3) -> float:
    import jax

    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    start = time.perf_counter()
    for _ in range(repeats):
        jax.block_until_ready(fn(*args))
    return (time.perf_counter() - start) / repeats


@pytest.mark.benchmark
def test_phase2_gpu_kernel_throughput(capsys) -> None:
    """Profile rule match vs equity scan vs full batch on GPU."""
    if not _gpu_available():
        pytest.skip("JAX GPU backend unavailable")

    df = _make_benchmark_df(n=4000)
    feature_modes = {"feat_a": "binary", "feat_b": "positive", "feat_c": "ternary"}
    engine = GPUBacktestEngine(df, feature_modes, "long")

    import jax
    import jax.numpy as jnp

    batch_size = resolve_phase2_gpu_batch_size()
    k = len(feature_modes)
    dont_cares = engine._dont_cares_jax
    chroms = jnp.full((batch_size, k), 0, dtype=jnp.int32)
    data = engine._data_matrix_jax
    price_returns = engine._get_trade_outcomes(_cfg.PHASE2_TP, _cfg.PHASE2_SL)

    signals = _jax_compute_rule_signals_batch(data, chroms, dont_cares)
    t_signals = _timed_gpu(
        lambda: _jax_compute_rule_signals_batch(data, chroms, dont_cares),
    )

    capital_rate = _cfg.PHASE2_CAPITAL_PCT / 100.0
    max_exp = _cfg.MAX_TOTAL_EXPOSURE_PCT / 100.0
    n_rows = len(df)
    max_slots = max(1, int(max_exp / capital_rate) + 1)

    t_equity = _timed_gpu(
        lambda: _jax_simulate_equity_batch(
            signals,
            price_returns,
            engine._release_indices_jax,
            float(_cfg.INITIAL_CAPITAL),
            n_rows,
            max_slots,
            engine.fee_rate,
            engine.leverage,
            capital_rate,
            max_exp,
            engine.min_position_notional,
        ),
    )

    chrom_np = np.asarray(chroms)
    t_full = _timed_gpu(
        lambda: engine.simulate_rule_batch(
            chrom_np,
            tp=_cfg.PHASE2_TP,
            sl=_cfg.PHASE2_SL,
            capital_pct=_cfg.PHASE2_CAPITAL_PCT,
        ),
        warmup=1,
    )

    jax.block_until_ready(signals)
    print(
        f"\nPhase2 GPU throughput (N={n_rows}, B={batch_size}, "
        f"unroll={_cfg.PHASE2_SCAN_UNROLL}): "
        f"signals={t_signals*1000:.1f}ms "
        f"equity={t_equity*1000:.1f}ms "
        f"full_batch={t_full*1000:.1f}ms"
    )
