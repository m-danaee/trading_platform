"""
Unit tests for gpu_fuzzy_trader.data.loader.Data_Loader

Tests cover:
  - CSV loading and datetime parsing
  - Sort by (datetime, symbol)
  - Tail drop of last TAIL_DROP_ROWS rows per symbol
  - NaN label row removal
  - Feature NaN fill with 0
  - _symbol_bar_index computation
"""

import os
import tempfile

import pandas as pd
import pytest

from gpu_fuzzy_trader.config import LABEL_COLUMNS, META_COLUMNS, TAIL_DROP_ROWS
from gpu_fuzzy_trader.data.loader import Data_Loader, load_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_timestamps(n: int, start: str = "2024-01-01 00:00:00", freq_minutes: int = 5) -> list[str]:
    """Generate n evenly-spaced ISO datetime strings."""
    base = pd.Timestamp(start)
    return [
        (base + pd.Timedelta(minutes=freq_minutes * i)).strftime("%Y-%m-%d %H:%M:%S")
        for i in range(n)
    ]


def _base_row(symbol: int, dt: str, label_val: float = 1.0, feat_val: float = 1.0) -> dict:
    """Return a minimal row dict with all required columns."""
    return {
        "datetime": dt,
        "symbol": symbol,
        "label_open_next": label_val,
        "label_close_288": label_val,
        "label_min_288": label_val,
        "label_max_288": label_val,
        "label_max_before_min": label_val,
        "feature_a": feat_val,
        "feature_b": feat_val,
    }


def _make_rows(
    symbol: int,
    n: int,
    label_val: float = 1.0,
    feat_val: float = 1.0,
    start: str = "2024-01-01 00:00:00",
) -> list[dict]:
    """Generate n rows for a given symbol with valid timestamps."""
    timestamps = _make_timestamps(n, start=start)
    return [_base_row(symbol, ts, label_val=label_val, feat_val=feat_val) for ts in timestamps]


def _make_ohlcv_rows(
    symbol: str,
    n: int,
    start: str = "2024-01-01 00:00:00",
) -> list[dict]:
    """Generate raw OHLCV rows without precomputed forward labels."""
    timestamps = _make_timestamps(n, start=start)
    return [
        {
            "datetime": ts,
            "symbol": symbol,
            "open": 100.0 + i,
            "high": 101.0 + i,
            "low": 99.0 + i,
            "close": 100.5 + i,
            "volume": 10.0 + i,
            "feature_a": float(i),
        }
        for i, ts in enumerate(timestamps)
    ]


def _make_csv(rows: list[dict]) -> str:
    """Build a CSV string from a list of row dicts."""
    return pd.DataFrame(rows).to_csv(index=False)


def _loader_from_rows(rows: list[dict], **kwargs) -> pd.DataFrame:
    """Write rows to a temp CSV and load via Data_Loader."""
    csv_text = _make_csv(rows)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        f.write(csv_text)
        tmp_path = f.name
    try:
        return Data_Loader().load_dataset(tmp_path, **kwargs)
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Tests: datetime parsing
# ---------------------------------------------------------------------------

class TestDatetimeParsing:
    def test_datetime_column_is_datetime_dtype(self):
        rows = _make_rows(1, TAIL_DROP_ROWS + 1)
        df = _loader_from_rows(rows)
        assert pd.api.types.is_datetime64_any_dtype(df["datetime"])

    def test_datetime_values_are_correct(self):
        rows = _make_rows(1, TAIL_DROP_ROWS + 2)
        df = _loader_from_rows(rows)
        # First row should be 2024-01-01 00:00:00
        assert df["datetime"].iloc[0] == pd.Timestamp("2024-01-01 00:00:00")


# ---------------------------------------------------------------------------
# Tests: sort by (datetime, symbol)
# ---------------------------------------------------------------------------

class TestSort:
    def test_rows_sorted_by_datetime_then_symbol(self):
        # Same timestamp, symbols out of order
        ts = "2024-01-01 00:00:00"
        rows = [
            _base_row(2, ts),
            _base_row(1, ts),
            *_make_rows(1, TAIL_DROP_ROWS, start="2024-01-01 00:05:00"),
        ]
        df = _loader_from_rows(rows)
        same_ts = df[df["datetime"] == pd.Timestamp(ts)]
        assert same_ts["symbol"].tolist() == sorted(same_ts["symbol"].tolist())
        assert df["datetime"].is_monotonic_increasing

    def test_datetime_monotonic_per_symbol(self):
        # Provide rows in reverse datetime order for symbol 1
        rows = list(reversed(_make_rows(1, TAIL_DROP_ROWS + 3)))
        df = _loader_from_rows(rows)
        for sym, grp in df.groupby("symbol"):
            assert grp["datetime"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# Tests: raw OHLCV label generation
# ---------------------------------------------------------------------------

class TestOHLCVLabelGeneration:
    def test_labels_are_derived_when_csv_has_raw_ohlcv(self):
        rows = _make_ohlcv_rows("BTCUSDT", TAIL_DROP_ROWS + 5)
        df = _loader_from_rows(rows)

        assert len(df) == 5
        assert df["open"].iloc[0] == 100.0
        assert df["label_open_next"].iloc[0] == 101.0
        assert df["label_close_288"].iloc[0] == 196.5
        assert df["label_min_288"].iloc[0] == 100.0
        assert df["label_max_288"].iloc[0] == 197.0
        assert df["label_max_before_min"].iloc[0] == 0.0

    def test_partial_labels_are_rejected(self):
        rows = _make_rows(1, TAIL_DROP_ROWS + 1)
        for row in rows:
            row.pop("label_close_288")

        with pytest.raises(ValueError, match="only some required label columns"):
            _loader_from_rows(rows)


# ---------------------------------------------------------------------------
# Tests: tail drop
# ---------------------------------------------------------------------------

class TestTailDrop:
    def test_last_288_rows_dropped_per_symbol(self):
        n = TAIL_DROP_ROWS + 50
        rows = _make_rows(1, n)
        df = _loader_from_rows(rows)
        assert len(df) == 50

    def test_tail_drop_applied_independently_per_symbol(self):
        n1 = TAIL_DROP_ROWS + 10
        n2 = TAIL_DROP_ROWS + 20
        rows = _make_rows(1, n1) + _make_rows(2, n2)
        df = _loader_from_rows(rows)
        assert len(df[df["symbol"] == 1]) == 10
        assert len(df[df["symbol"] == 2]) == 20

    def test_symbol_with_fewer_rows_than_tail_drop_yields_empty(self):
        rows = _make_rows(1, TAIL_DROP_ROWS - 1)
        df = _loader_from_rows(rows)
        assert len(df) == 0

    def test_symbol_with_exactly_tail_drop_rows_yields_empty(self):
        rows = _make_rows(1, TAIL_DROP_ROWS)
        df = _loader_from_rows(rows)
        assert len(df) == 0

    def test_kept_rows_are_the_earliest_rows(self):
        """The first N-288 rows (chronologically) should be kept."""
        n = TAIL_DROP_ROWS + 5
        rows = _make_rows(1, n)
        df = _loader_from_rows(rows)
        # The last kept datetime should be the (n - TAIL_DROP_ROWS - 1)-th timestamp
        expected_last_ts = pd.Timestamp(_make_timestamps(n)[n - TAIL_DROP_ROWS - 1])
        assert df["datetime"].max() == expected_last_ts


# ---------------------------------------------------------------------------
# Tests: NaN label row removal
# ---------------------------------------------------------------------------

class TestNaNLabelDrop:
    def test_rows_with_nan_labels_are_dropped(self):
        rows = _make_rows(1, TAIL_DROP_ROWS + 5)
        # Inject NaN into one label column for the first row
        rows[0]["label_open_next"] = float("nan")
        df = _loader_from_rows(rows)
        # 5 rows survive tail drop; 1 has NaN label → 4 remain
        assert len(df) == 4

    def test_no_nan_in_label_columns_after_load(self):
        rows = _make_rows(1, TAIL_DROP_ROWS + 5)
        df = _loader_from_rows(rows)
        label_cols_present = [c for c in LABEL_COLUMNS if c in df.columns]
        assert not df[label_cols_present].isna().any().any()

    def test_all_label_columns_checked_for_nan(self):
        """A row with NaN in any label column should be dropped."""
        rows = _make_rows(1, TAIL_DROP_ROWS + 5)
        # Set NaN in a different label column
        rows[0]["label_max_288"] = float("nan")
        df = _loader_from_rows(rows)
        assert len(df) == 4


# ---------------------------------------------------------------------------
# Tests: feature NaN fill
# ---------------------------------------------------------------------------

class TestFeatureNaNFill:
    def test_feature_nan_filled_with_zero(self):
        rows = _make_rows(1, TAIL_DROP_ROWS + 3)
        rows[0]["feature_a"] = float("nan")
        df = _loader_from_rows(rows)
        assert df["feature_a"].iloc[0] == 0.0

    def test_no_nan_in_feature_columns_after_load(self):
        rows = _make_rows(1, TAIL_DROP_ROWS + 3)
        rows[1]["feature_b"] = float("nan")
        df = _loader_from_rows(rows)
        assert not df["feature_b"].isna().any()

    def test_label_nan_rows_dropped_not_filled(self):
        """NaN label rows should be dropped, not filled with 0."""
        rows = _make_rows(1, TAIL_DROP_ROWS + 3)
        rows[0]["label_open_next"] = float("nan")
        df = _loader_from_rows(rows)
        # 3 rows survive tail drop; 1 has NaN label → 2 remain
        assert len(df) == 2

    def test_explicit_feature_cols_respected(self):
        """Only specified feature_cols should be filled."""
        rows = _make_rows(1, TAIL_DROP_ROWS + 3)
        rows[0]["feature_a"] = float("nan")
        rows[0]["feature_b"] = float("nan")
        # Only fill feature_a; feature_b NaN should also be filled (it's a feature)
        df = _loader_from_rows(rows, feature_cols=["feature_a"])
        assert df["feature_a"].iloc[0] == 0.0
        # feature_b was not in feature_cols, so it should still be NaN
        assert pd.isna(df["feature_b"].iloc[0])


# ---------------------------------------------------------------------------
# Tests: _symbol_bar_index
# ---------------------------------------------------------------------------

class TestSymbolBarIndex:
    def test_symbol_bar_index_starts_at_zero(self):
        rows = _make_rows(1, TAIL_DROP_ROWS + 5)
        df = _loader_from_rows(rows)
        assert df["_symbol_bar_index"].iloc[0] == 0

    def test_symbol_bar_index_is_sequential(self):
        rows = _make_rows(1, TAIL_DROP_ROWS + 5)
        df = _loader_from_rows(rows)
        for sym, grp in df.groupby("symbol"):
            assert list(grp["_symbol_bar_index"]) == list(range(len(grp)))

    def test_symbol_bar_index_resets_per_symbol(self):
        rows = _make_rows(1, TAIL_DROP_ROWS + 3) + _make_rows(2, TAIL_DROP_ROWS + 3)
        df = _loader_from_rows(rows)
        for sym, grp in df.groupby("symbol"):
            assert grp["_symbol_bar_index"].iloc[0] == 0

    def test_symbol_bar_index_computed_after_all_drops(self):
        """Bar index should reflect post-drop row count, not original row count."""
        n = TAIL_DROP_ROWS + 10
        rows = _make_rows(1, n)
        df = _loader_from_rows(rows)
        # After tail drop, 10 rows remain; bar index should go 0..9
        assert df["_symbol_bar_index"].max() == 9

    def test_symbol_bar_index_column_present(self):
        rows = _make_rows(1, TAIL_DROP_ROWS + 1)
        df = _loader_from_rows(rows)
        assert "_symbol_bar_index" in df.columns


# ---------------------------------------------------------------------------
# Tests: module-level convenience function
# ---------------------------------------------------------------------------

class TestModuleLevelFunction:
    def test_load_dataset_function_returns_dataframe(self):
        rows = _make_rows(1, TAIL_DROP_ROWS + 1)
        csv_text = _make_csv(rows)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            f.write(csv_text)
            tmp_path = f.name
        try:
            df = load_dataset(tmp_path)
            assert isinstance(df, pd.DataFrame)
        finally:
            os.unlink(tmp_path)

    def test_load_dataset_function_matches_class(self):
        """Module-level function should produce same result as class method."""
        rows = _make_rows(1, TAIL_DROP_ROWS + 5)
        csv_text = _make_csv(rows)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            f.write(csv_text)
            tmp_path = f.name
        try:
            df_func = load_dataset(tmp_path)
            df_class = Data_Loader().load_dataset(tmp_path)
            pd.testing.assert_frame_equal(df_func, df_class)
        finally:
            os.unlink(tmp_path)
