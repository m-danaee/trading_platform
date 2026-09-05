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

import numpy as np
import pandas as pd
import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.backtest.barrier import (
    attach_barrier_outcomes,
    barrier_column_names,
    required_barrier_columns,
)
from gpu_fuzzy_trader.config import LABEL_COLUMNS, TAIL_DROP_ROWS
from gpu_fuzzy_trader.data.labels import compute_labels
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


def _write_raw_ohlcv_csv(tmp_path, *, rows: int):
    """Write a raw tape whose cache can be reused across loader invocations."""
    path = tmp_path / "raw_ohlcv.csv"
    pd.DataFrame(_make_ohlcv_rows("BTCUSDT", rows)).to_csv(path, index=False)
    return path


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

    def test_rejects_duplicate_timestamp_for_precomputed_labels(self):
        """A supplied-label tape must not contain two bars for one symbol/time."""
        rows = _make_rows(1, TAIL_DROP_ROWS + 3)
        rows.insert(1, dict(rows[0]))

        with pytest.raises(ValueError, match="duplicate.*datetime.*symbol"):
            _loader_from_rows(rows)


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

    def test_supplied_labels_are_checked_against_ohlcv(self):
        """Raw prices must be the source of truth for forward targets."""
        raw = pd.DataFrame(
            _make_ohlcv_rows("BTCUSDT", TAIL_DROP_ROWS + 5)
        )
        labels = compute_labels(raw)
        labelled = raw.merge(labels, on=["datetime", "symbol"], validate="one_to_one")
        path = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline="",
        )
        path.close()
        try:
            labelled.to_csv(path.name, index=False)
            loaded = Data_Loader().load_dataset(path.name)
            assert loaded["label_close_288"].iloc[0] == pytest.approx(
                labels["label_close_288"].iloc[0]
            )
        finally:
            os.unlink(path.name)

    def test_mismatched_ohlcv_labels_are_rejected(self):
        """A label override must not create a hidden look-ahead target."""
        raw = pd.DataFrame(
            _make_ohlcv_rows("BTCUSDT", TAIL_DROP_ROWS + 5)
        )
        labelled = raw.merge(
            compute_labels(raw),
            on=["datetime", "symbol"],
            validate="one_to_one",
        )
        labelled.loc[0, "label_close_288"] += 1.0
        path = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline="",
        )
        path.close()
        try:
            labelled.to_csv(path.name, index=False)
            with pytest.raises(ValueError, match="Supplied labels do not match"):
                Data_Loader().load_dataset(path.name)
        finally:
            os.unlink(path.name)

    def test_partial_ohlcv_with_supplied_labels_is_rejected(self):
        raw = pd.DataFrame(
            _make_ohlcv_rows("BTCUSDT", TAIL_DROP_ROWS + 5)
        )
        labelled = raw.merge(
            compute_labels(raw),
            on=["datetime", "symbol"],
            validate="one_to_one",
        ).drop(columns=["volume"])

        path = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline="",
        )
        path.close()
        try:
            labelled.to_csv(path.name, index=False)
            with pytest.raises(ValueError, match="only some OHLCV columns"):
                Data_Loader().load_dataset(path.name)
        finally:
            os.unlink(path.name)

    def test_nonfinite_ohlcv_is_rejected(self):
        raw = pd.DataFrame(
            _make_ohlcv_rows("BTCUSDT", TAIL_DROP_ROWS + 5)
        )
        raw.loc[0, "high"] = np.nan

        path = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline="",
        )
        path.close()
        try:
            raw.to_csv(path.name, index=False)
            with pytest.raises(ValueError, match="non-numeric or non-finite"):
                Data_Loader().load_dataset(path.name)
        finally:
            os.unlink(path.name)


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
# Tests: feature NaN preservation / fill
# ---------------------------------------------------------------------------

class TestFeatureNaNFill:
    def test_feature_nan_preserved_by_default(self):
        """Feature NaNs must be preserved by default (not artificially filled with 0)."""
        rows = _make_rows(1, TAIL_DROP_ROWS + 3)
        rows[0]["feature_a"] = float("nan")
        df = _loader_from_rows(rows)
        assert pd.isna(df["feature_a"].iloc[0])

    def test_feature_nan_filled_with_zero_when_configured(self, monkeypatch):
        """When FILL_NA_WITH_ZERO is enabled in config, feature NaNs are filled with 0."""
        from gpu_fuzzy_trader import config as _cfg
        monkeypatch.setattr(_cfg, "FILL_NA_WITH_ZERO", True)
        rows = _make_rows(1, TAIL_DROP_ROWS + 3)
        rows[0]["feature_a"] = float("nan")
        df = _loader_from_rows(rows)
        assert df["feature_a"].iloc[0] == 0.0

    def test_label_nan_rows_dropped_not_filled(self):
        """NaN label rows should be dropped, not filled with 0."""
        rows = _make_rows(1, TAIL_DROP_ROWS + 3)
        rows[0]["label_open_next"] = float("nan")
        df = _loader_from_rows(rows)
        # 3 rows survive tail drop; 1 has NaN label → 2 remain
        assert len(df) == 2

    def test_explicit_feature_cols_respected_when_configured(self, monkeypatch):
        """Only specified feature_cols should be filled when FILL_NA_WITH_ZERO is enabled."""
        from gpu_fuzzy_trader import config as _cfg
        monkeypatch.setattr(_cfg, "FILL_NA_WITH_ZERO", True)
        rows = _make_rows(1, TAIL_DROP_ROWS + 3)
        rows[0]["feature_a"] = float("nan")
        rows[0]["feature_b"] = float("nan")
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
# Tests: exact barrier cache contract
# ---------------------------------------------------------------------------

class TestBarrierCache:
    def test_empty_barrier_pair_override_preserves_configured_default(
        self,
        monkeypatch,
    ):
        monkeypatch.setattr(_cfg, "RB_TP_GRID", ())
        monkeypatch.setattr(_cfg, "RB_SL_GRID", ())
        raw = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01", periods=3, freq="15min"),
            "symbol": ["BTCUSDT"] * 3,
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
        })

        attached = attach_barrier_outcomes(raw, horizon=1, pairs=[])

        assert required_barrier_columns().issubset(attached.columns)

    def test_required_barrier_columns_consumes_single_pass_iterable_once(self):
        pair = (2.0, 1.2)
        columns = required_barrier_columns(
            (candidate for candidate in [pair]),
        )

        expected = set()
        for direction in ("long", "short"):
            expected.update(barrier_column_names(direction, *pair))
        assert columns == expected

    def test_barrier_cache_rebuilds_when_risk_grid_changes(
        self,
        tmp_path,
        monkeypatch,
    ):
        monkeypatch.setattr(_cfg, "OUTPUTS_DIR", str(tmp_path / "outputs"))
        monkeypatch.setattr(_cfg, "RB_TP_GRID", ())
        monkeypatch.setattr(_cfg, "RB_SL_GRID", ())
        csv_path = _write_raw_ohlcv_csv(
            tmp_path,
            rows=TAIL_DROP_ROWS + 4,
        )

        first = Data_Loader().load_dataset(
            str(csv_path),
            drop_tail=False,
            include_barrier_outcomes=True,
        )
        first_columns = {
            column for column in first if column.startswith("_barrier_")
        }

        monkeypatch.setattr(_cfg, "RB_TP_GRID", (3.0,))
        monkeypatch.setattr(_cfg, "RB_SL_GRID", (1.2,))
        refreshed = Data_Loader().load_dataset(
            str(csv_path),
            drop_tail=False,
            include_barrier_outcomes=True,
        )
        refreshed_columns = {
            column for column in refreshed if column.startswith("_barrier_")
        }

        assert first_columns < refreshed_columns
        assert refreshed_columns == required_barrier_columns()

    def test_barrier_cache_rebuilds_when_cached_columns_are_incomplete(
        self,
        tmp_path,
        monkeypatch,
    ):
        monkeypatch.setattr(_cfg, "OUTPUTS_DIR", str(tmp_path / "outputs"))
        csv_path = _write_raw_ohlcv_csv(
            tmp_path,
            rows=TAIL_DROP_ROWS + 4,
        )

        Data_Loader().load_dataset(
            str(csv_path),
            drop_tail=False,
            include_barrier_outcomes=True,
        )
        cache_dir = tmp_path / "outputs" / ".cache" / "barriers"
        cache_file = next(cache_dir.glob("*.parquet"))
        cached = pd.read_parquet(cache_file)
        cached.drop(columns=[cached.columns[0]]).to_parquet(cache_file)

        rebuilt = Data_Loader().load_dataset(
            str(csv_path),
            drop_tail=False,
            include_barrier_outcomes=True,
        )
        rebuilt_columns = {
            column for column in rebuilt if column.startswith("_barrier_")
        }
        assert rebuilt_columns == required_barrier_columns()

    def test_barrier_cache_rebuilds_when_values_are_corrupted(
        self,
        tmp_path,
        monkeypatch,
    ):
        """A structurally valid but modified cache must not change outcomes."""
        monkeypatch.setattr(_cfg, "OUTPUTS_DIR", str(tmp_path / "outputs"))
        csv_path = _write_raw_ohlcv_csv(
            tmp_path,
            rows=TAIL_DROP_ROWS + 4,
        )

        first = Data_Loader().load_dataset(
            str(csv_path),
            drop_tail=False,
            include_barrier_outcomes=True,
        )
        cache_file = next(
            (tmp_path / "outputs" / ".cache" / "barriers").glob("*.parquet")
        )
        cached = pd.read_parquet(cache_file)
        column = cached.columns[0]
        expected = float(first.loc[0, column])
        cached.loc[0, column] = 777.0
        cached.to_parquet(cache_file, index=False)

        rebuilt = Data_Loader().load_dataset(
            str(csv_path),
            drop_tail=False,
            include_barrier_outcomes=True,
        )

        assert float(rebuilt.loc[0, column]) == pytest.approx(expected)


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
