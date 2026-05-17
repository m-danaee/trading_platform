"""
Unit tests for gpu_fuzzy_trader.data.splitter.Data_Splitter

Tests cover:
  - Per-symbol 75/25 split using floor(N * 0.75)
  - Chronological ordering preserved (train rows precede validation rows)
  - No row overlap between train and validation sets
  - Row count conservation (train + validation == total)
  - Parquet persistence to TRAIN_75_PATH and VALIDATION_25_PATH
  - Multi-symbol independence (each symbol split independently)
  - Edge cases: single-row symbol, odd/even row counts
  - Module-level convenience function
"""

from __future__ import annotations

import math
import os
import tempfile

import pandas as pd
import pytest

from gpu_fuzzy_trader.config import (
    LABEL_COLUMNS,
    TAIL_DROP_ROWS,
    TRAIN_75_PATH,
    VALIDATION_25_PATH,
)
from gpu_fuzzy_trader.data.splitter import Data_Splitter, split_and_persist


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_timestamps(n: int, start: str = "2020-01-01 00:00:00", freq_minutes: int = 5) -> list[str]:
    """Generate n evenly-spaced ISO datetime strings."""
    base = pd.Timestamp(start)
    return [
        (base + pd.Timedelta(minutes=freq_minutes * i)).strftime("%Y-%m-%d %H:%M:%S")
        for i in range(n)
    ]


def _make_df(symbol_sizes: dict[int | str, int]) -> pd.DataFrame:
    """
    Build a minimal DataFrame with the given per-symbol row counts.

    Parameters
    ----------
    symbol_sizes:
        Mapping of symbol → number of rows.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: datetime, symbol, label_open_next,
        label_close_288, label_min_288, label_max_288,
        label_max_before_min, feature_a.
        Rows are sorted by (symbol, datetime) as Data_Loader would produce.
    """
    rows: list[dict] = []
    for symbol, n in symbol_sizes.items():
        timestamps = _make_timestamps(n)
        for i, ts in enumerate(timestamps):
            rows.append(
                {
                    "datetime": pd.Timestamp(ts),
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
    df = pd.DataFrame(rows).sort_values(["symbol", "datetime"]).reset_index(drop=True)
    return df


def _split(symbol_sizes: dict, tmp_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Helper: build df, patch paths, run split, return (train, val)."""
    import gpu_fuzzy_trader.data.splitter as splitter_mod
    import gpu_fuzzy_trader.config as config_mod

    original_train = config_mod.TRAIN_75_PATH
    original_val = config_mod.VALIDATION_25_PATH

    train_path = os.path.join(tmp_dir, "train_75.parquet")
    val_path = os.path.join(tmp_dir, "validation_25.parquet")

    # Patch module-level constants used inside splitter
    splitter_mod.TRAIN_75_PATH = train_path
    splitter_mod.VALIDATION_25_PATH = val_path

    try:
        df = _make_df(symbol_sizes)
        train_df, val_df = Data_Splitter().split_and_persist(df)
    finally:
        splitter_mod.TRAIN_75_PATH = original_train
        splitter_mod.VALIDATION_25_PATH = original_val

    return train_df, val_df, train_path, val_path


# ---------------------------------------------------------------------------
# Tests: split ratio
# ---------------------------------------------------------------------------

class TestSplitRatio:
    def test_single_symbol_train_size_uses_floor(self, tmp_path):
        """floor(N * 0.75) rows go to train."""
        n = 100
        train_df, val_df, _, _ = _split({1: n}, str(tmp_path))
        expected_train = math.floor(n * 0.75)  # 75
        assert len(train_df) == expected_train

    def test_single_symbol_validation_size_is_remainder(self, tmp_path):
        """Remaining rows after floor(N * 0.75) go to validation."""
        n = 100
        train_df, val_df, _, _ = _split({1: n}, str(tmp_path))
        expected_val = n - math.floor(n * 0.75)  # 25
        assert len(val_df) == expected_val

    def test_floor_not_round_for_odd_sizes(self, tmp_path):
        """For N=101: floor(101 * 0.75) = 75, not 76."""
        n = 101
        train_df, val_df, _, _ = _split({1: n}, str(tmp_path))
        assert len(train_df) == math.floor(n * 0.75)
        assert len(val_df) == n - math.floor(n * 0.75)

    def test_row_count_conservation(self, tmp_path):
        """train + validation == total rows."""
        n = 200
        train_df, val_df, _, _ = _split({1: n}, str(tmp_path))
        assert len(train_df) + len(val_df) == n

    def test_row_count_conservation_multi_symbol(self, tmp_path):
        """train + validation == total rows across all symbols."""
        sizes = {1: 100, 2: 150, 3: 80}
        total = sum(sizes.values())
        train_df, val_df, _, _ = _split(sizes, str(tmp_path))
        assert len(train_df) + len(val_df) == total


# ---------------------------------------------------------------------------
# Tests: per-symbol independence
# ---------------------------------------------------------------------------

class TestPerSymbolIndependence:
    def test_each_symbol_split_independently(self, tmp_path):
        """Each symbol's split point is computed from its own row count."""
        sizes = {1: 100, 2: 200}
        train_df, val_df, _, _ = _split(sizes, str(tmp_path))

        for sym, n in sizes.items():
            expected_train = math.floor(n * 0.75)
            expected_val = n - expected_train
            assert len(train_df[train_df["symbol"] == sym]) == expected_train
            assert len(val_df[val_df["symbol"] == sym]) == expected_val

    def test_symbols_with_different_sizes_split_correctly(self, tmp_path):
        """Symbols with different sizes each get the correct floor(N*0.75) split."""
        sizes = {1: 7, 2: 13, 3: 40}
        train_df, val_df, _, _ = _split(sizes, str(tmp_path))

        for sym, n in sizes.items():
            expected_train = math.floor(n * 0.75)
            assert len(train_df[train_df["symbol"] == sym]) == expected_train


# ---------------------------------------------------------------------------
# Tests: chronological ordering
# ---------------------------------------------------------------------------

class TestChronologicalOrdering:
    def test_train_rows_precede_validation_rows_per_symbol(self, tmp_path):
        """All train datetimes for a symbol must be <= all validation datetimes."""
        n = 100
        train_df, val_df, _, _ = _split({1: n}, str(tmp_path))

        train_max_dt = train_df[train_df["symbol"] == 1]["datetime"].max()
        val_min_dt = val_df[val_df["symbol"] == 1]["datetime"].min()
        assert train_max_dt < val_min_dt

    def test_train_rows_precede_validation_rows_multi_symbol(self, tmp_path):
        """Chronological ordering holds independently for each symbol."""
        sizes = {1: 100, 2: 80}
        train_df, val_df, _, _ = _split(sizes, str(tmp_path))

        for sym in sizes:
            train_max = train_df[train_df["symbol"] == sym]["datetime"].max()
            val_min = val_df[val_df["symbol"] == sym]["datetime"].min()
            assert train_max < val_min, f"Symbol {sym}: train/val overlap"

    def test_train_rows_are_first_floor_n_075_rows(self, tmp_path):
        """Train rows should be the first floor(N*0.75) rows by feature_a index."""
        n = 20
        train_df, val_df, _, _ = _split({1: n}, str(tmp_path))
        split_point = math.floor(n * 0.75)

        # feature_a encodes the original row index (0..n-1)
        train_indices = sorted(train_df[train_df["symbol"] == 1]["feature_a"].tolist())
        val_indices = sorted(val_df[val_df["symbol"] == 1]["feature_a"].tolist())

        assert train_indices == list(range(split_point))
        assert val_indices == list(range(split_point, n))


# ---------------------------------------------------------------------------
# Tests: no overlap
# ---------------------------------------------------------------------------

class TestNoOverlap:
    def test_no_row_overlap_between_train_and_validation(self, tmp_path):
        """No row should appear in both train and validation sets."""
        n = 100
        train_df, val_df, _, _ = _split({1: n}, str(tmp_path))

        # Use (symbol, datetime) as a unique row identifier
        train_keys = set(zip(train_df["symbol"], train_df["datetime"]))
        val_keys = set(zip(val_df["symbol"], val_df["datetime"]))
        assert train_keys.isdisjoint(val_keys)

    def test_no_overlap_multi_symbol(self, tmp_path):
        sizes = {1: 100, 2: 80}
        train_df, val_df, _, _ = _split(sizes, str(tmp_path))

        train_keys = set(zip(train_df["symbol"], train_df["datetime"]))
        val_keys = set(zip(val_df["symbol"], val_df["datetime"]))
        assert train_keys.isdisjoint(val_keys)


# ---------------------------------------------------------------------------
# Tests: Parquet persistence
# ---------------------------------------------------------------------------

class TestParquetPersistence:
    def test_train_parquet_file_created(self, tmp_path):
        _, _, train_path, _ = _split({1: 100}, str(tmp_path))
        assert os.path.exists(train_path)

    def test_validation_parquet_file_created(self, tmp_path):
        _, _, _, val_path = _split({1: 100}, str(tmp_path))
        assert os.path.exists(val_path)

    def test_train_parquet_content_matches_returned_df(self, tmp_path):
        train_df, _, train_path, _ = _split({1: 100}, str(tmp_path))
        loaded = pd.read_parquet(train_path)
        pd.testing.assert_frame_equal(
            train_df.reset_index(drop=True),
            loaded.reset_index(drop=True),
        )

    def test_validation_parquet_content_matches_returned_df(self, tmp_path):
        _, val_df, _, val_path = _split({1: 100}, str(tmp_path))
        loaded = pd.read_parquet(val_path)
        pd.testing.assert_frame_equal(
            val_df.reset_index(drop=True),
            loaded.reset_index(drop=True),
        )


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_row_symbol_goes_to_train(self, tmp_path):
        """floor(1 * 0.75) = 0, so the single row goes to validation."""
        train_df, val_df, _, _ = _split({1: 1}, str(tmp_path))
        # floor(1 * 0.75) = 0 → 0 train rows, 1 validation row
        assert len(train_df) == 0
        assert len(val_df) == 1

    def test_four_row_symbol_split(self, tmp_path):
        """floor(4 * 0.75) = 3 → 3 train, 1 validation."""
        train_df, val_df, _, _ = _split({1: 4}, str(tmp_path))
        assert len(train_df) == 3
        assert len(val_df) == 1

    def test_empty_dataframe_returns_empty_dfs(self, tmp_path):
        """An empty input DataFrame should produce empty train and validation."""
        import gpu_fuzzy_trader.data.splitter as splitter_mod
        import gpu_fuzzy_trader.config as config_mod

        original_train = config_mod.TRAIN_75_PATH
        original_val = config_mod.VALIDATION_25_PATH
        splitter_mod.TRAIN_75_PATH = str(tmp_path / "train_75.parquet")
        splitter_mod.VALIDATION_25_PATH = str(tmp_path / "validation_25.parquet")

        try:
            empty_df = pd.DataFrame(
                columns=["datetime", "symbol", "feature_a", "_symbol_bar_index"]
            )
            train_df, val_df = Data_Splitter().split_and_persist(empty_df)
            assert len(train_df) == 0
            assert len(val_df) == 0
        finally:
            splitter_mod.TRAIN_75_PATH = original_train
            splitter_mod.VALIDATION_25_PATH = original_val

    def test_large_symbol_split_ratio_close_to_075(self, tmp_path):
        """For large N, train/total should be very close to 0.75."""
        n = 10_000
        train_df, val_df, _, _ = _split({1: n}, str(tmp_path))
        ratio = len(train_df) / n
        assert abs(ratio - 0.75) < 0.001


# ---------------------------------------------------------------------------
# Tests: return value
# ---------------------------------------------------------------------------

class TestReturnValue:
    def test_returns_tuple_of_two_dataframes(self, tmp_path):
        result = _split({1: 100}, str(tmp_path))
        train_df, val_df = result[0], result[1]
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(val_df, pd.DataFrame)

    def test_returned_train_df_contains_all_symbols(self, tmp_path):
        sizes = {1: 100, 2: 80}
        train_df, _, _, _ = _split(sizes, str(tmp_path))
        assert set(train_df["symbol"].unique()) == set(sizes.keys())

    def test_returned_val_df_contains_all_symbols(self, tmp_path):
        sizes = {1: 100, 2: 80}
        _, val_df, _, _ = _split(sizes, str(tmp_path))
        assert set(val_df["symbol"].unique()) == set(sizes.keys())


# ---------------------------------------------------------------------------
# Tests: module-level convenience function
# ---------------------------------------------------------------------------

class TestModuleLevelFunction:
    def test_split_and_persist_function_returns_tuple(self, tmp_path):
        import gpu_fuzzy_trader.data.splitter as splitter_mod
        import gpu_fuzzy_trader.config as config_mod

        original_train = config_mod.TRAIN_75_PATH
        original_val = config_mod.VALIDATION_25_PATH
        splitter_mod.TRAIN_75_PATH = str(tmp_path / "train_75.parquet")
        splitter_mod.VALIDATION_25_PATH = str(tmp_path / "validation_25.parquet")

        try:
            df = _make_df({1: 100})
            result = split_and_persist(df)
            assert isinstance(result, tuple)
            assert len(result) == 2
        finally:
            splitter_mod.TRAIN_75_PATH = original_train
            splitter_mod.VALIDATION_25_PATH = original_val

    def test_split_and_persist_function_matches_class(self, tmp_path):
        """Module-level function should produce same result as class method."""
        import gpu_fuzzy_trader.data.splitter as splitter_mod
        import gpu_fuzzy_trader.config as config_mod

        original_train = config_mod.TRAIN_75_PATH
        original_val = config_mod.VALIDATION_25_PATH
        splitter_mod.TRAIN_75_PATH = str(tmp_path / "train_75.parquet")
        splitter_mod.VALIDATION_25_PATH = str(tmp_path / "validation_25.parquet")

        try:
            df = _make_df({1: 100})
            train_func, val_func = split_and_persist(df)
            train_class, val_class = Data_Splitter().split_and_persist(df)
            pd.testing.assert_frame_equal(
                train_func.reset_index(drop=True),
                train_class.reset_index(drop=True),
            )
            pd.testing.assert_frame_equal(
                val_func.reset_index(drop=True),
                val_class.reset_index(drop=True),
            )
        finally:
            splitter_mod.TRAIN_75_PATH = original_train
            splitter_mod.VALIDATION_25_PATH = original_val
