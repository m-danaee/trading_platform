"""GPU engine per-regime trade metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gpu_fuzzy_trader.backtest.gpu_engine import GPUBacktestEngine


def _mini_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "symbol": ["AAA"] * n,
        "_symbol_bar_index": np.arange(n, dtype=np.int32),
        "label_open_next": rng.uniform(99, 101, n),
        "label_max_288": rng.uniform(100, 105, n),
        "label_min_288": rng.uniform(95, 100, n),
        "label_close_288": rng.uniform(99, 102, n),
        "label_max_before_min": rng.integers(0, 2, n),
        "feat_a": rng.uniform(0, 1, n).astype(np.float32),
    })


@pytest.mark.skipif(
    not pytest.importorskip("jax", reason="JAX required"),
    reason="JAX not installed",
)
class TestGpuEngineRegimeMetrics:
    def test_regime_counts_sum_to_executed(self) -> None:
        df = _mini_df(300)
        n_regimes = 3
        regime_ids = np.random.default_rng(1).integers(0, n_regimes, len(df))
        eng = GPUBacktestEngine(
            df,
            {"feat_a": "positive"},
            "long",
            regime_ids=regime_ids,
            n_regimes=n_regimes,
        )
        chrom = np.array([[0]], dtype=np.int32)
        metrics = eng.simulate_rule_batch(
            chrom, tp=2.0, sl=1.0, capital_pct=48.0)[0]
        assert "regime_trade_counts" in metrics
        assert sum(metrics["regime_trade_counts"]
                   ) == metrics["executed_trades"]
        wins = sum(metrics["regime_win_counts"])
        assert wins <= metrics["executed_trades"]
