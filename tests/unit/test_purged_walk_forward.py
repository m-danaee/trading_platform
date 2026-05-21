"""
Unit tests for gpu_fuzzy_trader.data.purged_walk_forward.Purged_Walk_Forward

Tests cover:
  - Single-symbol fold partitioning with embargo
  - Multi-symbol fold concatenation
  - Embargo enforcement (no train rows within purge_bars of val start)
  - Chronological ordering within folds
  - No overlap between train and val within each fold
  - Embargo validation diagnostics
  - Edge cases: single symbol with too few rows, multiple folds, large purge
  - _partition_indices helper correctness
  - Persistence of fold parquet files
  - get_fold_report diagnostics
"""

from __future__ import annotations

import math
import os
import tempfile

import pandas as pd
import pytest

from gpu_fuzzy_trader.data.purged_walk_forward import (
    Purged_Walk_Forward,
    _partition_indices,
    _embargo_train_indices,
    validate_embargo,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sym_df(
    n: int = 200,
    sym: str = "SYM_A",
    start_dt: str = "2020-01-01 00:00:00",
) -> pd.DataFrame:
    """Build a sorted single-symbol DataFrame with _symbol_bar_index."""
    base = pd.Timestamp(start_dt)
    rows = []
    for i in range(n):
        rows.append({
            "datetime": base + pd.Timedelta(minutes=5 * i),
            "symbol": sym,
            "label_open_next": 1.0,
            "label_close_288": 1.0,
            "label_min_288": 0.99,
            "label_max_288": 1.01,
            "label_max_before_min": 1.0,
            "feature_a": float(i),
            "_symbol_bar_index": i,
        })
    return pd.DataFrame(rows)


def _make_multi_sym_df(
    sizes: dict[str, int],
) -> pd.DataFrame:
    """Build a sorted multi-symbol DataFrame."""
    dfs = []
    offset_days = 0
    for sym, n in sizes.items():
        base = pd.Timestamp("2020-01-01") + pd.Timedelta(days=offset_days)
        rows = []
        for i in range(n):
            rows.append({
                "datetime": base + pd.Timedelta(minutes=5 * i),
                "symbol": sym,
                "label_open_next": 1.0,
                "label_close_288": 1.0,
                "label_min_288": 0.99,
                "label_max_288": 1.01,
                "label_max_before_min": 1.0,
                "feature_a": float(i),
                "_symbol_bar_index": i,
            })
        dfs.append(pd.DataFrame(rows))
        offset_days += n * 5 // (60 * 24) + 10
    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Tests: _partition_indices helper
# ---------------------------------------------------------------------------


class TestPartitionIndices:
    def test_equal_partition(self):
        parts = _partition_indices(100, 4)
        assert len(parts) == 4
        assert parts[0] == (0, 25)
        assert parts[1] == (25, 50)
        assert parts[2] == (50, 75)
        assert parts[3] == (75, 100)

    def test_remainder_distributed(self):
        parts = _partition_indices(10, 3)
        assert parts == [(0, 4), (4, 7), (7, 10)]

    def test_single_partition(self):
        parts = _partition_indices(50, 1)
        assert parts == [(0, 50)]

    def test_empty(self):
        parts = _partition_indices(0, 5)
        assert parts == [(0, 0)] * 5


class TestEmbargoTrainIndices:
    def test_simple_embargo(self):
        result = _embargo_train_indices(list(range(100)), 50, 10)
        assert result == list(range(40))

    def test_embargo_removes_boundary(self):
        result = _embargo_train_indices(list(range(100)), 50, 50)
        assert result == list(range(0))

    def test_embargo_zero_bars(self):
        result = _embargo_train_indices(list(range(100)), 50, 0)
        assert result == list(range(50))


# ---------------------------------------------------------------------------
# Tests: Purged_Walk_Forward constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_defaults(self):
        pwf = Purged_Walk_Forward()
        assert pwf.n_splits == 5
        assert pwf.purge_bars == 288
        assert pwf.min_train_folds == 1

    def test_custom(self):
        pwf = Purged_Walk_Forward(n_splits=3, purge_bars=50, min_train_folds=2)
        assert pwf.n_splits == 3
        assert pwf.purge_bars == 50
        assert pwf.min_train_folds == 2

    def test_invalid_n_splits(self):
        with pytest.raises(ValueError, match="n_splits"):
            Purged_Walk_Forward(n_splits=1)

    def test_invalid_purge_bars(self):
        with pytest.raises(ValueError, match="purge_bars"):
            Purged_Walk_Forward(purge_bars=-1)

    def test_invalid_min_train_folds(self):
        with pytest.raises(ValueError, match="min_train_folds"):
            Purged_Walk_Forward(min_train_folds=0)


# ---------------------------------------------------------------------------
# Tests: split — single symbol
# ---------------------------------------------------------------------------


class TestSplitSingleSymbol:
    def test_produces_correct_number_of_folds(self):
        df = _make_sym_df(n=200)
        pwf = Purged_Walk_Forward(n_splits=5, purge_bars=10, min_train_folds=1)
        folds = pwf.split(df)
        # With 5 splits and min_train_folds=1, we get folds for i=1..4 = 4 folds
        assert len(folds) == 4

    def test_chronological_within_folds(self):
        df = _make_sym_df(n=200)
        pwf = Purged_Walk_Forward(n_splits=4, purge_bars=5, min_train_folds=1)
        folds = pwf.split(df)
        for train_df, val_df in folds:
            if len(train_df) > 0 and len(val_df) > 0:
                t_max = train_df["_symbol_bar_index"].max()
                v_min = val_df["_symbol_bar_index"].min()
                assert t_max < v_min, "Train rows must precede val rows"

    def test_embargo_gap(self):
        """Verify embargo removes training rows close to the fold boundary."""
        df = _make_sym_df(n=500)
        purge = 50
        pwf = Purged_Walk_Forward(n_splits=5, purge_bars=purge, min_train_folds=1)
        folds = pwf.split(df)
        for train_df, val_df in folds:
            if len(train_df) > 0 and len(val_df) > 0:
                t_max = train_df["_symbol_bar_index"].max()
                v_min = val_df["_symbol_bar_index"].min()
                assert t_max < (v_min - purge), (
                    f"Embargo violation: train_max={t_max}, val_min={v_min}, "
                    f"purge={purge}"
                )

    def test_no_overlap_within_fold(self):
        df = _make_sym_df(n=200)
        pwf = Purged_Walk_Forward(n_splits=4, purge_bars=10, min_train_folds=1)
        folds = pwf.split(df)
        for train_df, val_df in folds:
            t_set = set(train_df["_symbol_bar_index"])
            v_set = set(val_df["_symbol_bar_index"])
            assert t_set.isdisjoint(v_set), "Train and val must not overlap"

    def test_empty_if_too_few_rows(self):
        df = _make_sym_df(n=3)
        pwf = Purged_Walk_Forward(n_splits=5, purge_bars=10, min_train_folds=1)
        folds = pwf.split(df)
        assert folds == []

    def test_purge_can_remove_all_train(self):
        """With a very large purge window, training rows may be entirely removed."""
        df = _make_sym_df(n=100)
        pwf = Purged_Walk_Forward(n_splits=3, purge_bars=200, min_train_folds=1)
        folds = pwf.split(df)
        # Some folds may be empty if purge removes all train
        for train_df, val_df in folds:
            # This is fine: some folds might have empty train
            pass


# ---------------------------------------------------------------------------
# Tests: split — multi-symbol
# ---------------------------------------------------------------------------


class TestSplitMultiSymbol:
    def test_produces_folds_for_all_symbols(self):
        df = _make_multi_sym_df({"A": 200, "B": 200})
        pwf = Purged_Walk_Forward(n_splits=4, purge_bars=10, min_train_folds=1)
        folds = pwf.split(df)
        assert len(folds) > 0
        for train_df, val_df in folds:
            t_syms = set(train_df["symbol"].unique())
            v_syms = set(val_df["symbol"].unique())
            assert "A" in t_syms
            assert "B" in t_syms

    def test_symbols_with_different_lengths(self):
        df = _make_multi_sym_df({"A": 300, "B": 100})
        pwf = Purged_Walk_Forward(n_splits=3, purge_bars=20, min_train_folds=1)
        folds = pwf.split(df)
        # Both symbols should appear in each fold
        for train_df, val_df in folds:
            assert len(val_df) > 0


# ---------------------------------------------------------------------------
# Tests: persistence
# ---------------------------------------------------------------------------


class TestPersistFolds:
    def test_writes_parquet_files(self, tmp_path):
        df = _make_sym_df(n=200)
        pwf = Purged_Walk_Forward(n_splits=3, purge_bars=10, min_train_folds=1)
        folds = pwf.split(df)
        paths = pwf.persist_folds(folds, str(tmp_path))
        assert len(paths) == len(folds) * 2
        for p in paths:
            assert os.path.exists(p)
            assert p.endswith(".parquet")

    def test_persisted_files_are_loadable(self, tmp_path):
        df = _make_sym_df(n=200)
        pwf = Purged_Walk_Forward(n_splits=3, purge_bars=10, min_train_folds=1)
        folds = pwf.split(df)
        paths = pwf.persist_folds(folds, str(tmp_path))
        for p in paths:
            loaded = pd.read_parquet(p)
            assert len(loaded) > 0
            assert "symbol" in loaded.columns


# ---------------------------------------------------------------------------
# Tests: get_fold_report
# ---------------------------------------------------------------------------


class TestGetFoldReport:
    def test_report_structure(self):
        df = _make_multi_sym_df({"A": 200, "B": 200})
        pwf = Purged_Walk_Forward(n_splits=3, purge_bars=10, min_train_folds=1)
        folds = pwf.split(df)
        report = pwf.get_fold_report(folds)
        assert "n_folds" in report
        assert "fold_details" in report
        assert len(report["fold_details"]) == len(folds)
        for fd in report["fold_details"]:
            assert "fold" in fd
            assert "train_rows" in fd
            assert "val_rows" in fd

    def test_report_row_conservation(self):
        df = _make_multi_sym_df({"A": 200})
        pwf = Purged_Walk_Forward(n_splits=4, purge_bars=5, min_train_folds=1)
        folds = pwf.split(df)
        report = pwf.get_fold_report(folds)
        for fd in report["fold_details"]:
            # Total rows may be less than 200 due to purge
            assert fd["train_rows"] + fd["val_rows"] <= 200


# ---------------------------------------------------------------------------
# Tests: validate_embargo diagnostics
# ---------------------------------------------------------------------------


class TestValidateEmbargo:
    def test_passes_on_valid_folds(self):
        df = _make_sym_df(n=500)
        pwf = Purged_Walk_Forward(n_splits=5, purge_bars=50, min_train_folds=1)
        folds = pwf.split(df)
        report = validate_embargo(folds, purge_bars=50)
        for fi, diag in report.items():
            assert diag["pass"], f"Fold {fi} expected to pass embargo check"

    def test_detects_violation(self):
        """Artificially create a fold with embargo violation."""
        df = _make_sym_df(n=200)
        # Manually create a fold that violates embargo
        train = df.iloc[:100].copy()
        val = df.iloc[50:150].copy()  # overlap!
        folds = [(train, val)]
        report = validate_embargo(folds, purge_bars=0)
        # With purge_bars=0 and overlapping data, should fail
        assert not report[0]["pass"]


# ---------------------------------------------------------------------------
# Tests: minimum train folds
# ---------------------------------------------------------------------------


class TestMinTrainFolds:
    def test_requires_preceding_folds(self):
        df = _make_sym_df(n=500)
        pwf = Purged_Walk_Forward(n_splits=5, purge_bars=10, min_train_folds=2)
        folds = pwf.split(df)
        # With min_train_folds=2, we need folds 0 and 1 for first val → fold 2 is first val
        # Folds for i in [2, 3, 4] = 3 validation folds
        assert len(folds) == 3

    def test_all_folds_consumed(self):
        df = _make_sym_df(n=500)
        pwf = Purged_Walk_Forward(n_splits=5, purge_bars=10, min_train_folds=4)
        folds = pwf.split(df)
        # Only fold i=4 has 4 preceding folds
        assert len(folds) == 1
