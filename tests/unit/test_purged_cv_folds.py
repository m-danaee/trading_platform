"""Unit tests for purged rolling CV fold construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import gpu_fuzzy_trader.config as config_mod
from gpu_fuzzy_trader.data.cv_folds import (
    PurgedFold,
    _symbol_fold_slices,
    build_purged_rolling_folds,
    cv_embargo_bars,
    cv_min_train_bars,
)
from gpu_fuzzy_trader.data.splitter import Data_Splitter


def _make_symbol_df(n: int, symbol: int = 1) -> pd.DataFrame:
    base = pd.Timestamp("2020-01-01")
    rows = []
    for i in range(n):
        rows.append(
            {
                "datetime": base + pd.Timedelta(minutes=5 * i),
                "symbol": symbol,
                "label_open_next": 1.0,
                "label_close_288": 1.0,
                "label_min_288": 0.99,
                "label_max_288": 1.01,
                "label_max_before_min": 1.0,
                "feature_a": float(i),
                "_symbol_bar_index": i,
            }
        )
    return pd.DataFrame(rows)


class TestSymbolFoldSlices:
    def test_embargo_excludes_rows_before_validation(self, monkeypatch):
        monkeypatch.setattr(config_mod, "CV_N_FOLDS", 3)
        monkeypatch.setattr(config_mod, "CV_EMBARGO_BARS", 10)
        monkeypatch.setattr(config_mod, "CV_MIN_TRAIN_MONTHS", 0.01)
        monkeypatch.setattr(config_mod, "CV_BARS_PER_DAY", 288)

        min_train = cv_min_train_bars()
        n = min_train + 500
        slices = _symbol_fold_slices(
            n,
            n_folds=3,
            embargo=cv_embargo_bars(),
            min_train=min_train,
        )
        assert len(slices) == 3
        for train_idx, val_idx in slices:
            assert train_idx.max() < val_idx.min() - cv_embargo_bars()
            assert len(train_idx) >= min_train

    def test_short_series_returns_no_folds(self, monkeypatch):
        monkeypatch.setattr(config_mod, "CV_MIN_TRAIN_MONTHS", 2.0)
        monkeypatch.setattr(config_mod, "CV_BARS_PER_DAY", 288)
        slices = _symbol_fold_slices(
            1000,
            n_folds=3,
            embargo=288,
            min_train=cv_min_train_bars(),
        )
        assert slices == []


class TestBuildPurgedRollingFolds:
    def test_multi_symbol_folds_no_overlap(self, monkeypatch):
        monkeypatch.setattr(config_mod, "CV_N_FOLDS", 3)
        monkeypatch.setattr(config_mod, "CV_EMBARGO_BARS", 5)
        monkeypatch.setattr(config_mod, "CV_MIN_TRAIN_MONTHS", 0.02)
        monkeypatch.setattr(config_mod, "CV_BARS_PER_DAY", 288)

        df = pd.concat(
            [_make_symbol_df(4000, 1), _make_symbol_df(4000, 2)],
            ignore_index=True,
        )
        folds = build_purged_rolling_folds(df)
        assert len(folds) == 3
        for fold in folds:
            assert isinstance(fold, PurgedFold)
            train_keys = set(
                zip(fold.train_df["symbol"], fold.train_df["datetime"]))
            val_keys = set(zip(fold.val_df["symbol"], fold.val_df["datetime"]))
            assert train_keys.isdisjoint(val_keys)

    def test_last_fold_is_latest_validation_block(self, monkeypatch):
        monkeypatch.setattr(config_mod, "CV_N_FOLDS", 2)
        monkeypatch.setattr(config_mod, "CV_EMBARGO_BARS", 2)
        monkeypatch.setattr(config_mod, "CV_MIN_TRAIN_MONTHS", 0.01)
        monkeypatch.setattr(config_mod, "CV_BARS_PER_DAY", 288)

        df = _make_symbol_df(3000)
        folds = build_purged_rolling_folds(df)
        assert folds
        assert folds[-1].val_df["datetime"].max() == df["datetime"].max()


class TestSplitterPurgedMode:
    def test_split_and_persist_returns_folds(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config_mod, "SPLIT_MODE", "purged_rolling_cv")
        monkeypatch.setattr(config_mod, "CV_N_FOLDS", 2)
        monkeypatch.setattr(config_mod, "CV_EMBARGO_BARS", 3)
        monkeypatch.setattr(config_mod, "CV_MIN_TRAIN_MONTHS", 0.01)
        monkeypatch.setattr(config_mod, "CV_BARS_PER_DAY", 288)

        import gpu_fuzzy_trader.data.splitter as splitter_mod

        splitter_mod.TRAIN_70_PATH = str(tmp_path / "train_70.parquet")
        splitter_mod.VALIDATION_30_PATH = str(tmp_path / "val_30.parquet")
        splitter_mod.CV_FOLDS_MANIFEST_PATH = str(tmp_path / "manifest.json")

        df = _make_symbol_df(2500)
        train_df, val_df, folds = Data_Splitter().split_and_persist(df)
        assert len(folds) >= 1
        assert len(train_df) > 0
        assert len(val_df) > 0
        assert folds[-1].val_df["datetime"].max() == df["datetime"].max()
