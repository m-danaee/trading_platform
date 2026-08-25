"""Regression tests for the unified master temporal fold contract."""

from __future__ import annotations

import pandas as pd
import pytest

from gpu_fuzzy_trader.mtf.cross_fitting import (
    build_master_temporal_folds,
    validate_master_temporal_folds,
)


def _make_synthetic_df(
    symbol_periods: dict[str, tuple[str, int]] | None = None,
    *,
    freq: str = "1h",
) -> pd.DataFrame:
    """Build deterministic UTC-normalized bars for fold tests."""
    specs = symbol_periods or {"BTCUSDT": ("2024-01-01", 241)}
    frames = []
    for symbol, (start, periods) in specs.items():
        dates = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
        frames.append(
            pd.DataFrame(
                {
                    "datetime": dates.tz_localize(None),
                    "symbol": symbol,
                    "close": range(periods),
                }
            )
        )
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["datetime", "symbol"])
        .reset_index(drop=True)
    )


def _datetime_series(df: pd.DataFrame) -> pd.Series:
    """Return timestamps in the same UTC-naive form as fold boundaries."""
    return pd.to_datetime(df["datetime"], utc=True).dt.tz_localize(None)


def _interval_rows(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    timestamps = _datetime_series(df)
    return df.loc[(timestamps >= start) & (timestamps < end)]


def _fold_has_symbol_coverage(
    df: pd.DataFrame,
    fold,
    *,
    min_rows_per_symbol: int,
) -> bool:
    """Inline eligibility rule used before the production eligibility API exists."""
    symbols = set(df["symbol"])
    train = _interval_rows(df, fold.train_start, fold.train_end)
    test = _interval_rows(df, fold.test_start, fold.test_end)
    for symbol in symbols:
        if (train["symbol"] == symbol).sum() < min_rows_per_symbol:
            return False
        if (test["symbol"] == symbol).sum() < min_rows_per_symbol:
            return False
    return True


def _build_adaptive_folds_inline(
    df: pd.DataFrame,
    *,
    max_folds: int = 4,
    min_folds: int = 2,
    min_rows_per_symbol: int = 5,
) -> list:
    """Apply the planned K=4,3,2 fail-closed rule on current fold geometry."""
    for n_folds in range(max_folds, min_folds - 1, -1):
        folds = build_master_temporal_folds(
            df,
            n_folds=n_folds,
            embargo_minutes=0,
        )
        if all(
            _fold_has_symbol_coverage(
                df,
                fold,
                min_rows_per_symbol=min_rows_per_symbol,
            )
            for fold in folds
        ):
            return folds
    raise ValueError("no eligible adaptive fold count")


def test_folds_are_chronological_non_overlapping_and_expanding():
    df = _make_synthetic_df()
    folds = build_master_temporal_folds(df, n_folds=4, embargo_minutes=0)

    assert validate_master_temporal_folds(folds) is True
    assert len(folds) == 4

    for index, fold in enumerate(folds):
        assert fold.train_start < fold.train_end
        assert fold.train_end == fold.test_start
        assert fold.test_start < fold.test_end

        train = _interval_rows(df, fold.train_start, fold.test_start)
        test = _interval_rows(df, fold.test_start, fold.test_end)
        assert not set(train.index).intersection(test.index)
        assert train["datetime"].max() < test["datetime"].min()

        if index:
            previous = folds[index - 1]
            assert fold.train_start == previous.train_start
            assert fold.train_end > previous.train_end
            assert fold.test_start == previous.test_end


def test_test_intervals_are_contiguous_and_equal_by_time():
    df = _make_synthetic_df()
    folds = build_master_temporal_folds(df, n_folds=4, embargo_minutes=0)
    durations = [fold.test_end - fold.test_start for fold in folds]

    assert all(
        previous.test_end == current.test_start
        for previous, current in zip(folds, folds[1:])
    )
    assert max(durations) - min(durations) <= pd.Timedelta(nanoseconds=1)


def test_adaptive_k_selects_larger_eligible_count_and_rejects_tiny_folds():
    enough_rows = _make_synthetic_df(
        {"BTCUSDT": ("2024-01-01", 21)},
    )
    folds = _build_adaptive_folds_inline(
        enough_rows,
        min_rows_per_symbol=5,
    )
    assert len(folds) == 3
    assert validate_master_temporal_folds(folds) is True

    tiny = _make_synthetic_df({"BTCUSDT": ("2024-01-01", 9)})
    with pytest.raises(ValueError, match="no eligible"):
        _build_adaptive_folds_inline(tiny, min_rows_per_symbol=5)


def test_newcoin_half_history_fails_per_symbol_coverage():
    df = _make_synthetic_df(
        {
            "BTCUSDT": ("2024-01-01", 241),
            "NEWCOIN": ("2024-01-06", 121),
        }
    )
    folds = build_master_temporal_folds(df, n_folds=4, embargo_minutes=0)

    assert validate_master_temporal_folds(folds) is True
    assert any(
        not _fold_has_symbol_coverage(
            df,
            fold,
            min_rows_per_symbol=5,
        )
        for fold in folds
    )
    with pytest.raises(ValueError, match="no eligible"):
        _build_adaptive_folds_inline(df, min_rows_per_symbol=5)
