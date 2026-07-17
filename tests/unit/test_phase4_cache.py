"""Tests for Phase 4 grid signal-mask caching."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gpu_fuzzy_trader.backtest.cpu_engine import CPUBacktestEngine
from gpu_fuzzy_trader.phases.phase3_cache import (
    build_rules_signal_cache,
    verify_mask_cache_parity,
)
from gpu_fuzzy_trader.phases.phase4_wf_optimizer import _evaluate_ruleset


def _make_df(n_rows: int = 120, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    open_next = rng.uniform(100, 200, size=n_rows)
    return pd.DataFrame({
        "symbol": "SYM_A",
        "datetime": pd.date_range("2024-01-01", periods=n_rows, freq="5min"),
        "_symbol_bar_index": np.arange(n_rows),
        "label_open_next": open_next,
        "label_close_288": open_next * 1.02,
        "label_min_288": open_next * 0.98,
        "label_max_288": open_next * 1.05,
        "label_max_before_min": rng.integers(0, 2, size=n_rows).astype(float),
        "feat_0": rng.uniform(0, 1, size=n_rows),
        "feat_1": rng.uniform(0, 1, size=n_rows),
    })


def _rules() -> list[dict]:
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
            "sl": 1.5,
            "capital_pct": 25.0,
        },
    ]


class TestPhase4SignalCache:
    def test_cached_grid_eval_matches_direct(self) -> None:
        train_df = _make_df()
        val_df = _make_df(seed=11)
        rules = _rules()
        train_engine = CPUBacktestEngine(train_df, {}, "long")
        val_engine = CPUBacktestEngine(val_df, {}, "long")
        cache = build_rules_signal_cache(rules, train_df, val_df)

        assert verify_mask_cache_parity(train_df, rules, cache.train_masks)
        assert verify_mask_cache_parity(val_df, rules, cache.val_masks)

        direct_train, direct_val, direct_score = _evaluate_ruleset(
            train_engine, val_engine, rules)
        cached_train, cached_val, cached_score = _evaluate_ruleset(
            train_engine, val_engine, rules, eval_cache=cache)

        assert cached_train["executed_trades"] == direct_train["executed_trades"]
        assert cached_val["executed_trades"] == direct_val["executed_trades"]
        assert cached_score == pytest.approx(direct_score, rel=1e-9, abs=1e-6)

    def test_tp_sl_change_reuses_masks(self) -> None:
        train_df = _make_df()
        val_df = _make_df(seed=11)
        base_rules = _rules()
        tweaked = [dict(r) for r in base_rules]
        tweaked[0]["tp"] = 6.0
        tweaked[0]["sl"] = 1.0
        tweaked[0]["capital_pct"] = 15.0

        train_engine = CPUBacktestEngine(train_df, {}, "long")
        val_engine = CPUBacktestEngine(val_df, {}, "long")
        cache = build_rules_signal_cache(base_rules, train_df, val_df)

        base_m = train_engine.simulate_rule_set_from_cache(
            base_rules, cache, "train")
        tweaked_m = train_engine.simulate_rule_set_from_cache(
            tweaked, cache, "train")

        assert base_m["raw_signal_count"] == tweaked_m["raw_signal_count"]
