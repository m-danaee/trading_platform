"""Tests for Phase 3 eval cache and unified objectives."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.backtest.cpu_engine import CPUBacktestEngine
from gpu_fuzzy_trader.phases.phase3_cache import (
    build_phase3_eval_cache,
    verify_mask_cache_parity,
)
from gpu_fuzzy_trader.phases.phase3_objectives import (
    compute_phase3_objectives,
    per_rule_min_symbol_trades_cached,
)
from gpu_fuzzy_trader.phases.phase3_rule_set import (
    _per_rule_min_symbol_trades,
    _rule_set_to_engine_format,
)


def _make_df(n_rows: int = 120, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sym = "SYM_A"
    n = n_rows
    open_next = rng.uniform(100, 200, size=n)
    return pd.DataFrame({
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
    })


def _pool() -> list[dict]:
    return [
        {
            "conditions": ["[feat_0] IS Very High"],
            "tp": 4.0,
            "sl": 2.0,
            "capital_pct": 50.0,
        },
        {
            "conditions": ["[feat_1] IS High"],
            "tp": 3.0,
            "sl": 2.0,
            "capital_pct": 40.0,
        },
    ]


class TestPhase3EvalCache:
    def test_mask_cache_parity(self):
        df = _make_df()
        pool = _pool()
        engine = CPUBacktestEngine(df, {}, "long")
        cache = build_phase3_eval_cache(pool, df, df, engine)
        team = _rule_set_to_engine_format(pool)
        assert verify_mask_cache_parity(df, team, cache.train_masks)

    def test_cached_metrics_match_direct(self):
        df = _make_df()
        pool = _pool()
        engine = CPUBacktestEngine(df, {}, "long")
        cache = build_phase3_eval_cache(pool, df, df, engine)
        team = _rule_set_to_engine_format(pool)
        direct = engine.simulate_rule_set(team)
        cached = engine.simulate_rule_set_from_cache(team, cache, "train")
        assert cached["executed_trades"] == direct["executed_trades"]
        assert cached["sortino_ratio"] == pytest.approx(
            direct["sortino_ratio"], rel=1e-4, abs=1e-4)

    def test_per_rule_gate_cache_matches_brute_force(self):
        df = _make_df()
        pool = _pool()
        engine = CPUBacktestEngine(df, {}, "long")
        cache = build_phase3_eval_cache(pool, df, df, engine)
        team = _rule_set_to_engine_format(pool)
        brute = _per_rule_min_symbol_trades(team, engine, cache=None)
        cached = per_rule_min_symbol_trades_cached(
            team, cache.per_rule_min_val_trades)
        assert cached == brute

    def test_objectives_use_cache_gate(self):
        df = _make_df()
        pool = _pool()
        engine = CPUBacktestEngine(df, {}, "long")
        cache = build_phase3_eval_cache(pool, df, df, engine)
        team = _rule_set_to_engine_format(pool)
        train_m = engine.simulate_rule_set_from_cache(team, cache, "train")
        val_m = engine.simulate_rule_set_from_cache(team, cache, "val")
        obj = compute_phase3_objectives(
            train_m, val_m, team,
            per_rule_min_val_trades=cache.per_rule_min_val_trades,
        )
        assert obj.shape == (3,)
        assert np.all(np.isfinite(obj))
