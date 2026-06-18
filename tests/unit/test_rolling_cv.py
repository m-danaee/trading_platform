"""Unit tests for purged walk-forward fold construction."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from gpu_fuzzy_trader.validation.rolling_cv import (
    PurgedFold,
    aggregate_fold_metrics,
    build_purged_walk_forward_folds,
    cv_folds_only,
    derive_primary_holdout,
    summarize_fold_metrics,
)


def _make_symbol_df(symbol: str, n: int, *, start_index: int = 0) -> pd.DataFrame:
    base = pd.Timestamp("2020-01-01")
    rows = []
    for i in range(n):
        rows.append(
            {
                "datetime": base + pd.Timedelta(minutes=5 * (start_index + i)),
                "symbol": symbol,
                "label_open_next": 1.0,
                "label_close_288": 1.0,
                "label_min_288": 0.99,
                "label_max_288": 1.01,
                "label_max_before_min": 1.0,
                "_symbol_bar_index": start_index + i,
                "feature_a": float(i),
            }
        )
    return pd.DataFrame(rows)


def _make_multi_symbol_df(per_symbol: int, symbols: tuple[str, ...] = ("A", "B")) -> pd.DataFrame:
    parts = [_make_symbol_df(sym, per_symbol) for sym in symbols]
    return pd.concat(parts, ignore_index=True).sort_values(
        ["symbol", "datetime"]
    ).reset_index(drop=True)


@pytest.fixture
def purged_cv_config(monkeypatch):
    monkeypatch.setattr(
        "gpu_fuzzy_trader.validation.rolling_cv._cfg.PURGED_WF_N_SPLITS",
        4,
    )
    monkeypatch.setattr(
        "gpu_fuzzy_trader.validation.rolling_cv._cfg.PURGED_WF_HOLDOUT_FRACTION",
        0.25,
    )
    monkeypatch.setattr(
        "gpu_fuzzy_trader.validation.rolling_cv._cfg.PURGED_WF_EMBARGO_CANDLES",
        288,
    )
    monkeypatch.setattr(
        "gpu_fuzzy_trader.validation.rolling_cv._cfg.PURGED_WF_MIN_TRAIN_FRACTION",
        0.20,
    )
    monkeypatch.setattr(
        "gpu_fuzzy_trader.validation.rolling_cv._cfg.PURGED_WF_MIN_VALID_ROWS",
        200,
    )


class TestPurgedWalkForward:
    def test_build_folds_purge_gap(self, purged_cv_config):
        df = _make_multi_symbol_df(8000)
        folds = build_purged_walk_forward_folds(df)
        assert len(folds) >= 2

        cv = cv_folds_only(folds)
        assert cv, "expected at least one CV fold"

        for fold in cv:
            for sym, group in fold.train_df.groupby("symbol"):
                valid_sym = fold.valid_df[fold.valid_df["symbol"] == sym]
                if valid_sym.empty or group.empty:
                    continue
                last_train_idx = int(group["_symbol_bar_index"].max())
                first_valid_idx = int(valid_sym["_symbol_bar_index"].min())
                gap = first_valid_idx - last_train_idx
                assert gap >= 288, (
                    f"symbol {sym} fold {fold.fold_id}: purge gap {gap} < 288"
                )

    def test_expanding_train_grows(self, purged_cv_config):
        df = _make_multi_symbol_df(8000)
        folds = cv_folds_only(build_purged_walk_forward_folds(df))
        if len(folds) < 2:
            pytest.skip("not enough CV folds in synthetic data")
        assert folds[0].n_train_rows <= folds[-1].n_train_rows

    def test_derive_primary_holdout_matches_tail(self, purged_cv_config):
        df = _make_multi_symbol_df(6000)
        folds = build_purged_walk_forward_folds(df)
        train_df, val_df = derive_primary_holdout(folds)
        holdout = next(f for f in folds if f.is_holdout)
        assert len(train_df) == holdout.n_train_rows
        assert len(val_df) == holdout.n_valid_rows

    def test_aggregate_fold_metrics_worst(self):
        metrics = [
            {"total_return_pct": 5.0, "profit_factor": 1.5, "sortino_ratio": 1.0,
             "max_drawdown_pct": 3.0, "executed_trades": 40, "win_rate": 55.0},
            {"total_return_pct": -2.0, "profit_factor": 1.1, "sortino_ratio": 0.5,
             "max_drawdown_pct": 8.0, "executed_trades": 20, "win_rate": 48.0},
        ]
        agg = aggregate_fold_metrics(metrics, mode="worst")
        assert agg["total_return_pct"] == pytest.approx(-2.0)
        assert agg["executed_trades"] == 20
        assert agg["max_drawdown_pct"] == pytest.approx(8.0)

    def test_summarize_fold_metrics_empty(self):
        summary = summarize_fold_metrics([])
        assert summary.folds == 0
        assert summary.min_trades == 0
