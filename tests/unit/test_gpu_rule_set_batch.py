"""Strict parity tests for GPUBacktestEngine.simulate_rule_set_batch."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

jax_available = True
try:
    import jax  # noqa: F401
except ImportError:
    jax_available = False

pytestmark = pytest.mark.skipif(
    not jax_available,
    reason="JAX not installed",
)

if jax_available:
    from gpu_fuzzy_trader.backtest.cpu_engine import CPUBacktestEngine
    from gpu_fuzzy_trader.backtest.gpu_engine import GPUBacktestEngine


def _make_df(n_rows: int = 120, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    symbols = ["SYM_A", "SYM_B"]
    parts = []
    for sym in symbols:
        n = n_rows // 2
        open_next = rng.uniform(100, 200, size=n)
        parts.append(
            pd.DataFrame(
                {
                    "symbol": sym,
                    "datetime": pd.date_range("2020-01-01", periods=n, freq="5min"),
                    "_symbol_bar_index": np.arange(n),
                    "label_open_next": open_next,
                    "label_close_288": open_next * 1.02,
                    "label_min_288": open_next * 0.98,
                    "label_max_288": open_next * 1.05,
                    "label_max_before_min": rng.integers(0, 2, size=n).astype(float),
                    "feat_0": rng.uniform(0, 1, size=n),
                    "feat_1": rng.uniform(0, 1, size=n),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def _rule_sets() -> list[list[dict]]:
    return [
        [
            {
                "conditions": ["[feat_0] IS Very High"],
                "tp": 4.0,
                "sl": 2.0,
                "capital_pct": 50.0,
            },
            {
                "conditions": ["[feat_1] IS Very High"],
                "tp": 4.0,
                "sl": 2.0,
                "capital_pct": 50.0,
            },
        ],
        [
            {
                "conditions": ["[feat_0] IS High"],
                "tp": 3.0,
                "sl": 2.0,
                "capital_pct": 40.0,
            },
        ],
    ]


class TestSimulateRuleSetBatchParity:
    def test_batch_matches_sequential_cpu(self):
        df = _make_df()
        feature_modes: dict[str, str] = {}
        cpu = CPUBacktestEngine(df, feature_modes, "long")
        gpu = GPUBacktestEngine(df, feature_modes, "long")
        rule_sets = _rule_sets()

        batch = gpu.simulate_rule_set_batch(rule_sets)
        assert len(batch) == len(rule_sets)

        for rs, metrics in zip(rule_sets, batch):
            ref = cpu.simulate_rule_set(rs)
            assert metrics["executed_trades"] == ref["executed_trades"]
            assert metrics["sortino_ratio"] == pytest.approx(
                ref["sortino_ratio"], rel=1e-4, abs=1e-4
            )
            assert metrics["total_return_pct"] == pytest.approx(
                ref["total_return_pct"], rel=1e-4, abs=1e-4
            )
            assert metrics["max_drawdown_pct"] == pytest.approx(
                ref["max_drawdown_pct"], rel=1e-4, abs=1e-4
            )
            assert metrics["win_rate"] == pytest.approx(
                ref["win_rate"], rel=1e-4, abs=1e-4
            )
            assert metrics["per_symbol_metrics"].keys(
            ) == ref["per_symbol_metrics"].keys()
            for sym in ref["per_symbol_metrics"]:
                assert metrics["per_symbol_metrics"][sym]["trade_count"] == (
                    ref["per_symbol_metrics"][sym]["trade_count"]
                )

    def test_single_element_batch(self):
        df = _make_df()
        gpu = GPUBacktestEngine(df, {}, "short")
        rs = _rule_sets()[:1]
        one = gpu.simulate_rule_set_batch(rs)
        direct = gpu.simulate_rule_set(rs[0])
        assert one[0]["executed_trades"] == direct["executed_trades"]

    def test_jax_batch_with_cache_matches_cpu(self):
        from gpu_fuzzy_trader.phases.phase3_cache import build_phase3_eval_cache

        df = _make_df()
        cpu = CPUBacktestEngine(df, {}, "long")
        gpu = GPUBacktestEngine(df, {}, "long")
        rule_sets = _rule_sets()
        pool = [
            {"conditions": r["conditions"], "tp": r["tp"],
             "sl": r["sl"], "capital_pct": r["capital_pct"]}
            for rs in rule_sets for r in rs
        ]
        seen = []
        unique_pool = []
        for p in pool:
            key = frozenset(p["conditions"])
            if key not in seen:
                seen.append(key)
                unique_pool.append(p)
        cache = build_phase3_eval_cache(unique_pool, df, df, cpu)
        jax_batch = gpu.simulate_rule_set_batch_jax(
            rule_sets, cache=cache, split="val")
        for rs, metrics in zip(rule_sets, jax_batch):
            ref = cpu.simulate_rule_set_from_cache(rs, cache, "val")
            assert metrics["executed_trades"] == ref["executed_trades"]
            assert metrics["sortino_ratio"] == pytest.approx(
                ref["sortino_ratio"], rel=1e-4, abs=1e-4)
