"""Phase 4 per-rule symbol-scoped risk optimization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gpu_fuzzy_trader.backtest.cpu_engine import CPUBacktestEngine
from gpu_fuzzy_trader.phases.phase4_wf_optimizer import (
    _build_rule_opt_contexts,
    _evaluate_ruleset,
    _evaluate_single_rule,
    _filter_df_by_symbols,
    _optimize_risk_grid,
    _symbols_for_rule,
)


def _make_df(
    symbols: list[str],
    n_rows_per_symbol: int = 80,
    seed: int = 3,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    for sym in symbols:
        n_rows = n_rows_per_symbol
        open_next = rng.uniform(100, 200, size=n_rows)
        parts.append(pd.DataFrame({
            "symbol": sym,
            "datetime": pd.date_range(
                "2020-01-01", periods=n_rows, freq="5min"),
            "_symbol_bar_index": np.arange(n_rows),
            "label_open_next": open_next,
            "label_close_288": open_next * 1.02,
            "label_min_288": open_next * 0.98,
            "label_max_288": open_next * 1.05,
            "label_max_before_min": rng.integers(0, 2, size=n_rows).astype(float),
            "feat_0": rng.uniform(0, 1, size=n_rows),
        }))
    return pd.concat(parts, ignore_index=True)


def _rule(sym: str, feat_val: str = "Very High") -> dict:
    return {
        "conditions": [
            f"[feat_0] IS {feat_val}",
            f"symbol is {sym}",
        ],
        "tp": 2.0,
        "sl": 1.0,
        "capital_pct": 20.0,
    }


class TestSymbolHelpers:
    def test_symbols_for_rule_extracts_filters(self) -> None:
        rule = _rule("9")
        assert _symbols_for_rule(rule, 0) == ["9"]

    def test_filter_df_by_symbols_keeps_only_assigned_rows(self) -> None:
        df = _make_df(["1", "5", "9"])
        filtered = _filter_df_by_symbols(df, ["9"])
        assert set(filtered["symbol"].astype(str).unique()) == {"9"}
        assert len(filtered) == 80


class TestRuleOptContexts:
    def test_context_engines_use_symbol_slice_only(self) -> None:
        train_df = _make_df(["1", "9"])
        val_df = _make_df(["1", "9"], seed=11)
        rules = [_rule("9")]
        contexts = _build_rule_opt_contexts(
            rules, train_df, val_df, "long")

        assert len(contexts) == 1
        assert contexts[0].symbols == ["9"]
        assert len(contexts[0].train_engine.df) == 80
        assert len(contexts[0].val_engine.df) == 80

    def test_single_rule_eval_matches_filtered_full_eval(self, monkeypatch) -> None:
        from gpu_fuzzy_trader import config as _cfg

        monkeypatch.setattr(_cfg, "MONTHLY_VALIDATION_ENABLED", False)

        train_df = _make_df(["1", "9"])
        val_df = _make_df(["1", "9"], seed=11)
        rules = [_rule("9")]
        contexts = _build_rule_opt_contexts(
            rules, train_df, val_df, "long")

        single_train, single_val, single_score = _evaluate_single_rule(
            rules[0], contexts[0])

        filtered_train = _filter_df_by_symbols(train_df, ["9"])
        filtered_val = _filter_df_by_symbols(val_df, ["9"])
        full_train_engine = CPUBacktestEngine(filtered_train, {}, "long")
        full_val_engine = CPUBacktestEngine(filtered_val, {}, "long")
        full_train, full_val, full_score = _evaluate_ruleset(
            full_train_engine, full_val_engine, rules)

        assert single_train["executed_trades"] == full_train["executed_trades"]
        assert single_val["executed_trades"] == full_val["executed_trades"]
        assert single_score == pytest.approx(full_score, rel=1e-9, abs=1e-6)


class TestPerSymbolGridOptimization:
    def test_grid_uses_per_rule_symbol_contexts(self, monkeypatch) -> None:
        from gpu_fuzzy_trader import config as _cfg

        monkeypatch.setattr(_cfg, "PHASE4_GRID_TP_VALUES", (2.0, 3.0))
        monkeypatch.setattr(_cfg, "PHASE4_GRID_SL_VALUES", (1.0,))
        monkeypatch.setattr(_cfg, "PHASE4_GRID_CAPITAL_VALUES", (20.0,))
        monkeypatch.setattr(_cfg, "PHASE4_GRID_PASSES", 1)
        monkeypatch.setattr(_cfg, "PHASE4_GRID_MIN_IMPROVEMENT", 0.0)
        monkeypatch.setattr(_cfg, "MONTHLY_VALIDATION_ENABLED", False)

        train_df = _make_df(["1", "9"])
        val_df = _make_df(["1", "9"], seed=11)
        rules = [_rule("1"), _rule("9", feat_val="High")]
        rule_contexts = _build_rule_opt_contexts(
            rules, train_df, val_df, "long")

        train_engine = CPUBacktestEngine(train_df, {}, "long")
        val_engine = CPUBacktestEngine(val_df, {}, "long")

        optimized, _, _, _, history = _optimize_risk_grid(
            rules,
            train_engine,
            val_engine,
            rule_contexts=rule_contexts,
            min_improvement=0.0,
        )

        assert len(optimized) == 2
        assert _symbols_for_rule(optimized[0], 0) == ["1"]
        assert _symbols_for_rule(optimized[1], 1) == ["9"]
        assert rule_contexts[0].symbols == ["1"]
        assert rule_contexts[1].symbols == ["9"]
        assert len(history) >= 1
