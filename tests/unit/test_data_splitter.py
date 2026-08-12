"""
Unit tests for gpu_fuzzy_trader.data.splitter.Data_Splitter

Tests cover:
  - Per-symbol holdout+embargo split (HOLDOUT_TRAIN_FRACTION + embargo gap)
  - Chronological ordering preserved (train rows precede validation rows)
  - No row overlap between train and validation sets
  - Row count conservation (train + validation + embargo == total)
  - Parquet persistence to TRAIN_70_PATH and VALIDATION_30_PATH
  - Multi-symbol independence (each symbol split independently)
  - Edge cases: single-row symbol, small row counts (embargo consumes all)
  - Module-level convenience function
"""

from __future__ import annotations

import math
import os
import tempfile

import pandas as pd
import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.config import (
    LABEL_COLUMNS,
    TAIL_DROP_ROWS,
    TRAIN_70_PATH,
    VALIDATION_30_PATH,
    HOLDOUT_TRAIN_FRACTION,
    HOLDOUT_EMBARGO_CANDLES,
)
from gpu_fuzzy_trader.data.splitter import (
    Data_Splitter,
    load_cached_split_if_fresh,
    split_validation_fitness_selection,
    split_and_persist,
)


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


def _patch_split_paths(tmp_dir: str):
    """Return context manager patches for split parquet + manifest paths."""
    import contextlib

    import gpu_fuzzy_trader.config as config_mod
    import gpu_fuzzy_trader.data.splitter as splitter_mod

    train_path = os.path.join(tmp_dir, "train_70.parquet")
    val_path = os.path.join(tmp_dir, "validation_30.parquet")
    fitness_path = os.path.join(tmp_dir, "validation_fitness.parquet")
    selection_path = os.path.join(tmp_dir, "validation_selection.parquet")
    manifest_path = os.path.join(tmp_dir, "cv_folds_manifest.json")

    originals = {
        "train": (config_mod.TRAIN_70_PATH, splitter_mod.TRAIN_70_PATH),
        "val": (config_mod.VALIDATION_30_PATH, splitter_mod.VALIDATION_30_PATH),
        "fitness": (config_mod.VALIDATION_FITNESS_PATH, splitter_mod.VALIDATION_FITNESS_PATH),
        "selection": (
            config_mod.VALIDATION_SELECTION_PATH,
            splitter_mod.VALIDATION_SELECTION_PATH,
        ),
        "manifest": (config_mod.CV_FOLDS_MANIFEST_PATH,),
        "mode": (config_mod.SPLIT_MODE,),
    }

    @contextlib.contextmanager
    def _cm(split_mode: str = "holdout"):
        config_mod.TRAIN_70_PATH = train_path
        config_mod.VALIDATION_30_PATH = val_path
        config_mod.VALIDATION_FITNESS_PATH = fitness_path
        config_mod.VALIDATION_SELECTION_PATH = selection_path
        config_mod.CV_FOLDS_MANIFEST_PATH = manifest_path
        config_mod.SPLIT_MODE = split_mode
        splitter_mod.TRAIN_70_PATH = train_path
        splitter_mod.VALIDATION_30_PATH = val_path
        splitter_mod.VALIDATION_FITNESS_PATH = fitness_path
        splitter_mod.VALIDATION_SELECTION_PATH = selection_path
        try:
            yield {
                "train": train_path,
                "val": val_path,
                "fitness": fitness_path,
                "selection": selection_path,
                "manifest": manifest_path,
            }
        finally:
            config_mod.TRAIN_70_PATH = originals["train"][0]
            config_mod.VALIDATION_30_PATH = originals["val"][0]
            config_mod.VALIDATION_FITNESS_PATH = originals["fitness"][0]
            config_mod.VALIDATION_SELECTION_PATH = originals["selection"][0]
            config_mod.CV_FOLDS_MANIFEST_PATH = originals["manifest"][0]
            config_mod.SPLIT_MODE = originals["mode"][0]
            splitter_mod.TRAIN_70_PATH = originals["train"][1]
            splitter_mod.VALIDATION_30_PATH = originals["val"][1]
            splitter_mod.VALIDATION_FITNESS_PATH = originals["fitness"][1]
            splitter_mod.VALIDATION_SELECTION_PATH = originals["selection"][1]

    return _cm


def _split(symbol_sizes: dict, tmp_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Helper: build df, patch paths, run split, return (train, val)."""
    with _patch_split_paths(tmp_dir)() as paths:
        df = _make_df(symbol_sizes)
        train_df, val_df, cv_folds = Data_Splitter().split_and_persist(df)
        assert cv_folds is None
        assert os.path.exists(paths["manifest"])
        return train_df, val_df, paths["train"], paths["val"]


# ---------------------------------------------------------------------------
# Tests: split ratio
# ---------------------------------------------------------------------------

class TestSplitRatio:
    def _expected_train_val(self, n):
        """Compute expected train/val per symbol under holdout+embargo split."""
        train_frac = float(_cfg.HOLDOUT_TRAIN_FRACTION)
        embargo = int(_cfg.HOLDOUT_EMBARGO_CANDLES)
        exp_train = math.floor(n * train_frac)
        embargo_end = min(exp_train + embargo, n)
        exp_val = n - embargo_end
        return exp_train, exp_val

    def test_single_symbol_train_size_uses_floor(self, tmp_path):
        """floor(N * HOLDOUT_TRAIN_FRACTION) rows go to train."""
        n = 100
        train_df, val_df, _, _ = _split({1: n}, str(tmp_path))
        exp_train, _ = self._expected_train_val(n)
        assert len(train_df) == exp_train

    def test_single_symbol_validation_size_after_embargo(self, tmp_path):
        """Remaining rows after embargo gap go to validation."""
        n = 2000  # large enough to have val after 288-bar embargo
        train_df, val_df, _, _ = _split({1: n}, str(tmp_path))
        _, exp_val = self._expected_train_val(n)
        assert len(val_df) == exp_val

    def test_floor_not_round_for_odd_sizes(self, tmp_path):
        """For N=101: floor(101 * HOLDOUT_TRAIN_FRACTION) used, not round."""
        n = 101
        train_df, val_df, _, _ = _split({1: n}, str(tmp_path))
        exp_train, exp_val = self._expected_train_val(n)
        assert len(train_df) == exp_train
        assert len(val_df) == exp_val

    def test_row_count_conservation(self, tmp_path):
        """train + validation + embargo dropped == total rows."""
        n = 2000
        train_df, val_df, _, _ = _split({1: n}, str(tmp_path))
        embargo = int(_cfg.HOLDOUT_EMBARGO_CANDLES)
        exp_train, _ = self._expected_train_val(n)
        embargo_dropped = min(exp_train + embargo, n) - exp_train
        assert len(train_df) + embargo_dropped + len(val_df) == n

    def test_row_count_conservation_multi_symbol(self, tmp_path):
        """train + val + embargo_dropped == total for each symbol."""
        sizes = {1: 2000, 2: 1500, 3: 3000}
        total = sum(sizes.values())
        train_df, val_df, _, _ = _split(sizes, str(tmp_path))
        embargo = int(_cfg.HOLDOUT_EMBARGO_CANDLES)
        total_embargo_dropped = 0
        for n in sizes.values():
            exp_train = math.floor(n * float(_cfg.HOLDOUT_TRAIN_FRACTION))
            embargo_dropped = min(exp_train + embargo, n) - exp_train
            total_embargo_dropped += embargo_dropped
        assert len(train_df) + total_embargo_dropped + len(val_df) == total


# ---------------------------------------------------------------------------
# Tests: per-symbol independence
# ---------------------------------------------------------------------------

class TestPerSymbolIndependence:
    def test_each_symbol_split_independently(self, tmp_path):
        """Each symbol's split point is computed from its own row count."""
        sizes = {1: 2000, 2: 1500}
        train_df, val_df, _, _ = _split(sizes, str(tmp_path))
        train_frac = float(_cfg.HOLDOUT_TRAIN_FRACTION)
        embargo = int(_cfg.HOLDOUT_EMBARGO_CANDLES)

        for sym, n in sizes.items():
            exp_train = math.floor(n * train_frac)
            embargo_end = min(exp_train + embargo, n)
            exp_val = n - embargo_end
            assert len(train_df[train_df["symbol"] == sym]) == exp_train
            assert len(val_df[val_df["symbol"] == sym]) == exp_val

    def test_symbols_with_different_sizes_split_correctly(self, tmp_path):
        """Symbols with different sizes each get the correct floor(N * train_frac) split."""
        sizes = {1: 2000, 2: 1500, 3: 3000}
        train_df, val_df, _, _ = _split(sizes, str(tmp_path))
        train_frac = float(_cfg.HOLDOUT_TRAIN_FRACTION)

        for sym, n in sizes.items():
            exp_train = math.floor(n * train_frac)
            assert len(train_df[train_df["symbol"] == sym]) == exp_train


# ---------------------------------------------------------------------------
# Tests: chronological ordering
# ---------------------------------------------------------------------------

class TestChronologicalOrdering:
    def test_train_rows_precede_validation_rows_per_symbol(self, tmp_path):
        """All train datetimes for a symbol must be < validation datetimes (embargo gap)."""
        n = 2000
        train_df, val_df, _, _ = _split({1: n}, str(tmp_path))

        # Only check when validation is non-empty
        if len(val_df) > 0:
            train_max_dt = train_df[train_df["symbol"] == 1]["datetime"].max()
            val_min_dt = val_df[val_df["symbol"] == 1]["datetime"].min()
            assert train_max_dt < val_min_dt

    def test_train_rows_precede_validation_rows_multi_symbol(self, tmp_path):
        """Chronological ordering holds independently for each symbol."""
        sizes = {1: 2000, 2: 1800}
        train_df, val_df, _, _ = _split(sizes, str(tmp_path))

        for sym in sizes:
            sym_val = val_df[val_df["symbol"] == sym]
            if len(sym_val) == 0:
                continue
            train_max = train_df[train_df["symbol"] == sym]["datetime"].max()
            val_min = sym_val["datetime"].min()
            assert train_max < val_min, f"Symbol {sym}: train/val overlap"

    def test_train_rows_are_first_floor_n_times_train_frac_rows(self, tmp_path):
        """Train rows should be the first floor(N * train_frac) rows by feature_a index."""
        n = 2000
        train_df, val_df, _, _ = _split({1: n}, str(tmp_path))
        train_frac = float(_cfg.HOLDOUT_TRAIN_FRACTION)
        embargo = int(_cfg.HOLDOUT_EMBARGO_CANDLES)
        split_point = math.floor(n * train_frac)

        # feature_a encodes the original row index (0..n-1)
        train_indices = sorted(train_df[train_df["symbol"] == 1]["feature_a"].tolist())
        val_indices = sorted(val_df[val_df["symbol"] == 1]["feature_a"].tolist())

        assert train_indices == list(range(split_point)), (
            f"Expected train indices 0..{split_point - 1}, got {train_indices[:5]}..."
        )
        # Validation starts after embargo gap
        embargo_start = split_point + embargo
        if embargo_start < n:
            assert val_indices == list(range(embargo_start, n)), (
                f"Expected val indices {embargo_start}..{n - 1}, got {val_indices[:5]}..."
            )


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
        _, _, train_path, _ = _split({1: 2000}, str(tmp_path))
        assert os.path.exists(train_path)

    def test_validation_parquet_file_created(self, tmp_path):
        _, _, _, val_path = _split({1: 2000}, str(tmp_path))
        assert os.path.exists(val_path)

    def test_train_parquet_content_matches_returned_df(self, tmp_path):
        train_df, _, train_path, _ = _split({1: 2000}, str(tmp_path))
        loaded = pd.read_parquet(train_path)
        pd.testing.assert_frame_equal(
            train_df.reset_index(drop=True),
            loaded.reset_index(drop=True),
        )

    def test_validation_parquet_content_matches_returned_df(self, tmp_path):
        _, val_df, _, val_path = _split({1: 2000}, str(tmp_path))
        loaded = pd.read_parquet(val_path)
        pd.testing.assert_frame_equal(
            val_df.reset_index(drop=True),
            loaded.reset_index(drop=True),
        )


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_row_symbol_all_embargo(self, tmp_path):
        """Single row: floor(1 * train_frac) = 0, and 288-bar embargo consumes it."""
        train_df, val_df, _, _ = _split({1: 1}, str(tmp_path))
        # train = floor(1 * 0.65) = 0
        # embargo_end = min(0 + 288, 1) = 1 → val = 0
        assert len(train_df) == 0
        assert len(val_df) == 0

    def test_small_symbol_embargo_consumes_all_validation(self, tmp_path):
        """Small symbol where 288-bar embargo leaves no validation rows."""
        train_df, val_df, _, _ = _split({1: 4}, str(tmp_path))
        train_frac = float(_cfg.HOLDOUT_TRAIN_FRACTION)
        exp_train = math.floor(4 * train_frac)  # floor(4 * 0.65) = 2
        assert len(train_df) == exp_train
        # embargo_end = min(2 + 288, 4) = 4 → val = 0
        assert len(val_df) == 0

    def test_empty_dataframe_returns_empty_dfs(self, tmp_path):
        """An empty input DataFrame should produce empty train and validation."""
        with _patch_split_paths(str(tmp_path))():
            empty_df = pd.DataFrame(
                columns=["datetime", "symbol", "feature_a", "_symbol_bar_index"]
            )
            train_df, val_df, _ = Data_Splitter().split_and_persist(empty_df)
            assert len(train_df) == 0
            assert len(val_df) == 0

    def test_large_symbol_split_ratio(self, tmp_path):
        """For large N, train/total should be very close to HOLDOUT_TRAIN_FRACTION."""
        n = 10_000
        train_df, val_df, _, _ = _split({1: n}, str(tmp_path))
        train_frac = float(_cfg.HOLDOUT_TRAIN_FRACTION)
        ratio = len(train_df) / n
        assert abs(ratio - train_frac) < 0.001


# ---------------------------------------------------------------------------
# Tests: return value
# ---------------------------------------------------------------------------

class TestReturnValue:
    def test_returns_tuple_of_two_dataframes(self, tmp_path):
        result = _split({1: 2000}, str(tmp_path))
        train_df, val_df = result[0], result[1]
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(val_df, pd.DataFrame)

    def test_returned_train_df_contains_all_symbols(self, tmp_path):
        sizes = {1: 2000, 2: 1500}
        train_df, _, _, _ = _split(sizes, str(tmp_path))
        assert set(train_df["symbol"].unique()) == set(sizes.keys())

    def test_returned_val_df_contains_all_symbols(self, tmp_path):
        sizes = {1: 2000, 2: 1500}
        _, val_df, _, _ = _split(sizes, str(tmp_path))
        assert set(val_df["symbol"].unique()) == set(sizes.keys())


# ---------------------------------------------------------------------------
# Tests: module-level convenience function
# ---------------------------------------------------------------------------

class TestModuleLevelFunction:
    def test_split_and_persist_function_returns_tuple(self, tmp_path):
        with _patch_split_paths(str(tmp_path))():
            df = _make_df({1: 2000})
            result = split_and_persist(df)
            assert isinstance(result, tuple)
            assert len(result) == 3
            _train_df, _val_df, cv_folds = result
            assert cv_folds is None

    def test_split_and_persist_function_matches_class(self, tmp_path):
        """Module-level function should produce same result as class method."""
        with _patch_split_paths(str(tmp_path))():
            df = _make_df({1: 2000})
            train_func, val_func, cv_func = split_and_persist(df)
            train_class, val_class, cv_class = Data_Splitter().split_and_persist(df)
            pd.testing.assert_frame_equal(
                train_func.reset_index(drop=True),
                train_class.reset_index(drop=True),
            )
            pd.testing.assert_frame_equal(
                val_func.reset_index(drop=True),
                val_class.reset_index(drop=True),
            )
            assert cv_func == cv_class


class TestValidationHalfPurge:
    def test_fitness_and_selection_are_separated_by_label_horizon(self):
        validation = _make_df({"BTCUSDT": 1600, "ETHUSDT": 1701})
        fitness, selection = split_validation_fitness_selection(validation)
        purge = int(_cfg.VALIDATION_HALF_PURGE_CANDLES)

        for symbol, group in validation.groupby("symbol", observed=False):
            group = group.sort_values("datetime").reset_index(drop=True)
            boundary = len(group) // 2
            fitness_bars = fitness.loc[
                fitness["symbol"] == symbol, "_symbol_bar_index"
            ].tolist()
            selection_bars = selection.loc[
                selection["symbol"] == symbol, "_symbol_bar_index"
            ].tolist()
            assert fitness_bars == list(range(max(0, boundary - purge)))
            assert selection_bars == list(
                range(min(len(group), boundary + purge), len(group))
            )
            if fitness_bars and selection_bars:
                assert fitness_bars[-1] + purge < selection_bars[0]


class TestPurgedWalkForwardSplit:
    def test_purged_mode_returns_cv_folds(self, tmp_path, monkeypatch):
        import gpu_fuzzy_trader.config as config_mod

        monkeypatch.setattr(config_mod, "PURGED_WF_MIN_VALID_ROWS", 200)
        monkeypatch.setattr(config_mod, "PURGED_WF_MIN_TRAIN_FRACTION", 0.20)

        with _patch_split_paths(str(tmp_path))("purged_walk_forward") as paths:
            df = _make_df({1: 6000, 2: 6000})
            train_df, val_df, cv_folds = Data_Splitter().split_and_persist(df)
            assert cv_folds is not None
            assert len(cv_folds) >= 2
            assert len(train_df) + len(val_df) <= len(df)
            assert os.path.exists(paths["manifest"])


class TestLoadCachedSplitIfFresh:
    def _write_holdout_cache(self, tmp_path, monkeypatch, *, split_mode="holdout"):
        import gpu_fuzzy_trader.config as config_mod

        csv_path = tmp_path / "train.csv"
        csv_path.write_text("x\n1\n", encoding="utf-8")
        monkeypatch.setattr(config_mod, "TRAIN_CSV_PATH", str(csv_path))

        n = 4000  # large enough for both purged validation halves
        with _patch_split_paths(str(tmp_path))(split_mode) as paths:
            df = _make_df({1: n})
            Data_Splitter().split_and_persist(df)

            os.utime(csv_path, (1, 1))
            for path in paths.values():
                os.utime(path, (2, 2))

        monkeypatch.setattr(config_mod, "TRAIN_CSV_PATH", str(csv_path))
        monkeypatch.setattr(config_mod, "TRAIN_70_PATH", paths["train"])
        monkeypatch.setattr(config_mod, "VALIDATION_30_PATH", paths["val"])
        monkeypatch.setattr(
            config_mod, "VALIDATION_FITNESS_PATH", paths["fitness"])
        monkeypatch.setattr(
            config_mod, "VALIDATION_SELECTION_PATH", paths["selection"],
        )
        monkeypatch.setattr(
            config_mod, "CV_FOLDS_MANIFEST_PATH", paths["manifest"])
        monkeypatch.setattr(config_mod, "SPLIT_MODE", split_mode)
        return paths

    def test_holdout_cache_hit_returns_all_frames(self, tmp_path, monkeypatch):
        self._write_holdout_cache(tmp_path, monkeypatch)
        result = load_cached_split_if_fresh()
        assert result is not None
        train_df, val_df, val_fitness, val_selection, cv_folds = result
        assert len(train_df) > 0
        assert len(val_df) > 0
        assert len(val_fitness) > 0
        assert len(val_selection) > 0
        assert cv_folds is None

    def test_holdout_cache_rejected_when_manifest_split_mode_differs(
        self, tmp_path, monkeypatch,
    ):
        paths = self._write_holdout_cache(tmp_path, monkeypatch)
        import json

        manifest = json.loads(open(paths["manifest"], encoding="utf-8").read())
        manifest["split_mode"] = "purged_walk_forward"
        open(paths["manifest"], "w",
             encoding="utf-8").write(json.dumps(manifest))
        assert load_cached_split_if_fresh() is None

    def test_cache_rejected_when_csv_newer_than_parquets(self, tmp_path, monkeypatch):
        paths = self._write_holdout_cache(tmp_path, monkeypatch)
        import gpu_fuzzy_trader.config as config_mod

        os.utime(config_mod.TRAIN_CSV_PATH, (3, 3))
        assert load_cached_split_if_fresh() is None

    def test_cache_rejected_when_csv_content_changes_without_mtime_change(
        self, tmp_path, monkeypatch,
    ):
        self._write_holdout_cache(tmp_path, monkeypatch)
        import gpu_fuzzy_trader.config as config_mod

        before = os.stat(config_mod.TRAIN_CSV_PATH)
        with open(config_mod.TRAIN_CSV_PATH, "w", encoding="utf-8") as fh:
            fh.write("x\n2\n")
        os.utime(
            config_mod.TRAIN_CSV_PATH,
            ns=(before.st_atime_ns, before.st_mtime_ns),
        )

        assert load_cached_split_if_fresh() is None

    def test_cache_rejected_when_fitness_parquet_missing(self, tmp_path, monkeypatch):
        paths = self._write_holdout_cache(tmp_path, monkeypatch)
        os.remove(paths["fitness"])
        assert load_cached_split_if_fresh() is None

    def test_purged_reference_rows_mismatch_rejects_cache(
        self, tmp_path, monkeypatch,
    ):
        import json

        import gpu_fuzzy_trader.config as config_mod

        csv_path = tmp_path / "train.csv"
        csv_path.write_text("x\n1\n", encoding="utf-8")
        monkeypatch.setattr(config_mod, "TRAIN_CSV_PATH", str(csv_path))
        df = _make_df({1: 6000, 2: 6000})

        with _patch_split_paths(str(tmp_path))("purged_walk_forward") as paths:
            Data_Splitter().split_and_persist(df)

            manifest = json.loads(
                open(paths["manifest"], encoding="utf-8").read())
            manifest["reference_rows"] = int(manifest["reference_rows"]) + 999
            open(paths["manifest"], "w", encoding="utf-8").write(
                json.dumps(manifest),
            )

            os.utime(csv_path, (1, 1))
            for path in paths.values():
                os.utime(path, (2, 2))

        monkeypatch.setattr(config_mod, "TRAIN_CSV_PATH", str(csv_path))
        monkeypatch.setattr(config_mod, "TRAIN_70_PATH", paths["train"])
        monkeypatch.setattr(config_mod, "VALIDATION_30_PATH", paths["val"])
        monkeypatch.setattr(
            config_mod, "VALIDATION_FITNESS_PATH", paths["fitness"])
        monkeypatch.setattr(
            config_mod, "VALIDATION_SELECTION_PATH", paths["selection"],
        )
        monkeypatch.setattr(
            config_mod, "CV_FOLDS_MANIFEST_PATH", paths["manifest"])
        monkeypatch.setattr(config_mod, "SPLIT_MODE", "purged_walk_forward")

        monkeypatch.setattr(
            "gpu_fuzzy_trader.data.loader.Data_Loader.load_dataset",
            lambda self, path: df.copy(),
        )
        assert load_cached_split_if_fresh() is None

    def test_purged_cache_sets_reference_rows_from_fresh_load(
        self, tmp_path, monkeypatch,
    ):
        import gpu_fuzzy_trader.config as config_mod

        csv_path = tmp_path / "train.csv"
        csv_path.write_text("x\n1\n", encoding="utf-8")
        monkeypatch.setattr(config_mod, "TRAIN_CSV_PATH", str(csv_path))
        df = _make_df({1: 6000, 2: 6000})

        with _patch_split_paths(str(tmp_path))("purged_walk_forward") as paths:
            Data_Splitter().split_and_persist(df)
            expected_rows = len(df)

            os.utime(csv_path, (1, 1))
            for path in paths.values():
                os.utime(path, (2, 2))

        monkeypatch.setattr(config_mod, "TRAIN_CSV_PATH", str(csv_path))
        monkeypatch.setattr(config_mod, "TRAIN_70_PATH", paths["train"])
        monkeypatch.setattr(config_mod, "VALIDATION_30_PATH", paths["val"])
        monkeypatch.setattr(
            config_mod, "VALIDATION_FITNESS_PATH", paths["fitness"])
        monkeypatch.setattr(
            config_mod, "VALIDATION_SELECTION_PATH", paths["selection"],
        )
        monkeypatch.setattr(
            config_mod, "CV_FOLDS_MANIFEST_PATH", paths["manifest"])
        monkeypatch.setattr(config_mod, "SPLIT_MODE", "purged_walk_forward")
        config_mod.set_purged_wf_reference_rows(0)

        monkeypatch.setattr(
            "gpu_fuzzy_trader.data.loader.Data_Loader.load_dataset",
            lambda self, path: df.copy(),
        )
        result = load_cached_split_if_fresh()
        assert result is not None
        _, _, _, _, cv_folds = result
        assert cv_folds is not None
        assert config_mod._PURGED_WF_REFERENCE_ROWS == expected_rows
