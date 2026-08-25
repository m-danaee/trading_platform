"""Unit tests for Master Temporal Folds, Purged Embargo, and OOF Cross-Fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gpu_fuzzy_trader.mtf.cross_fitting import (
    DEFAULT_HWC_PURGE_MINUTES,
    DEFAULT_LWC_PURGE_MINUTES,
    DEFAULT_MWC_PURGE_MINUTES,
    TemporalFold,
    apply_purge_embargo,
    build_master_temporal_folds,
    export_fold_boundaries,
    generate_oof_predictions,
    generate_oof_scores,
    validate_master_temporal_folds,
)


def test_master_temporal_folds_purging():
    dt = pd.date_range("2024-01-01", "2024-04-30 23:45", freq="15min")
    df = pd.DataFrame({"datetime": dt, "symbol": "BTCUSDT", "close": 100.0})
    folds = build_master_temporal_folds(df, n_folds=3)
    assert len(folds) == 3

    # Check that training set for Fold 2 strictly purges samples whose forward label extends into test start
    test_start = folds[1].test_start
    train_subset = df[df["datetime"] < test_start]
    purged_train = apply_purge_embargo(train_subset, test_start, purge_minutes=1440)  # 24h purge
    assert purged_train["datetime"].max() <= test_start - pd.Timedelta(minutes=1440)


def test_master_temporal_folds_structure():
    dt = pd.date_range("2024-01-01", "2024-05-31 23:45", freq="15min")
    df = pd.DataFrame({"datetime": dt, "symbol": "BTCUSDT", "close": 100.0})
    folds = build_master_temporal_folds(df, n_folds=4, embargo_minutes=1440)

    assert len(folds) == 4
    assert validate_master_temporal_folds(folds) is True

    # Fold IDs describe geometry; role eligibility is checked by the caller.
    assert folds[0].fold_id == 1
    assert not hasattr(folds[0], "is_seed")
    assert folds[0].train_start == dt[0]

    # Expanding training window
    for i in range(1, len(folds)):
        assert folds[i].fold_id == i + 1
        assert not hasattr(folds[i], "is_seed")
        assert folds[i].train_start == folds[0].train_start
        assert folds[i].train_end == folds[i].test_start
        assert folds[i].test_start == folds[i - 1].test_end

    # Serialized export
    exported = export_fold_boundaries(folds)
    assert len(exported) == 4
    assert exported[0]["fold_id"] == 1
    assert "is_seed" not in exported[0]
    assert "train_start" in exported[0]


def test_master_temporal_folds_validation_failures():
    dt = pd.date_range("2024-01-01", "2024-03-31", freq="1h")
    df = pd.DataFrame({"datetime": dt})

    # Empty df raises ValueError
    with pytest.raises(ValueError, match="empty DataFrame"):
        build_master_temporal_folds(pd.DataFrame(), n_folds=3)

    # Invalid n_folds
    with pytest.raises(ValueError, match="n_folds must be >= 1"):
        build_master_temporal_folds(df, n_folds=0)

    # Single timestamp
    single_df = pd.DataFrame({"datetime": [pd.Timestamp("2024-01-01")]})
    with pytest.raises(ValueError, match="at least 2 distinct timestamps"):
        build_master_temporal_folds(single_df, n_folds=3)

    # Corrupted folds fail validation
    folds = build_master_temporal_folds(df, n_folds=3)
    corrupted_folds = list(folds)
    # Corrupt fold geometry
    corrupted_folds[0] = TemporalFold(
        fold_id=2,
        train_start=folds[0].train_start,
        train_end=folds[0].train_end,
        test_start=folds[0].test_start,
        test_end=folds[0].test_end,
    )
    assert validate_master_temporal_folds(corrupted_folds) is False

    # Empty list
    assert validate_master_temporal_folds([]) is False


def test_apply_purge_embargo_horizons():
    dt = pd.date_range("2024-01-01 00:00", "2024-01-10 00:00", freq="15min")
    df = pd.DataFrame({"datetime": dt, "val": range(len(dt))})
    test_start = pd.Timestamp("2024-01-08 00:00")

    train_subset = df[df["datetime"] < test_start]

    # HWC Purge: 1440 min (24h) -> max train datetime <= 2024-01-07 00:00
    hwc_purged = apply_purge_embargo(
        train_subset, pred_start_dt=test_start, purge_minutes=DEFAULT_HWC_PURGE_MINUTES
    )
    assert hwc_purged["datetime"].max() <= pd.Timestamp("2024-01-07 00:00")

    # MWC Purge: 240 min (4h) -> max train datetime <= 2024-01-07 20:00
    mwc_purged = apply_purge_embargo(
        train_subset, test_start=test_start, purge_minutes=DEFAULT_MWC_PURGE_MINUTES
    )
    assert mwc_purged["datetime"].max() <= pd.Timestamp("2024-01-07 20:00")

    # LWC Purge: 720 min (12h) -> max train datetime <= 2024-01-07 12:00
    lwc_purged = apply_purge_embargo(
        train_subset, test_start=test_start, purge_minutes=DEFAULT_LWC_PURGE_MINUTES
    )
    assert lwc_purged["datetime"].max() <= pd.Timestamp("2024-01-07 12:00")

    # Zero purge returns full train subset
    no_purge = apply_purge_embargo(train_subset, test_start=test_start, purge_minutes=0)
    assert len(no_purge) == len(train_subset)

    # DatetimeIndex support
    df_idx = df.set_index("datetime")
    train_idx = df_idx[df_idx.index < test_start]
    purged_idx = apply_purge_embargo(
        train_idx, pred_start_dt=test_start, purge_minutes=DEFAULT_MWC_PURGE_MINUTES
    )
    assert purged_idx.index.max() <= pd.Timestamp("2024-01-07 20:00")


def test_fold_get_train_test_slice():
    dt = pd.date_range("2024-01-01", "2024-04-30 23:45", freq="15min")
    df = pd.DataFrame({"datetime": dt, "symbol": "BTCUSDT", "close": 100.0})
    folds = build_master_temporal_folds(df, n_folds=3, embargo_minutes=DEFAULT_HWC_PURGE_MINUTES)

    fold2 = folds[1]
    train_slice = fold2.get_train_slice(df)
    test_slice = fold2.get_test_slice(df)

    assert train_slice["datetime"].max() <= fold2.test_start - pd.Timedelta(minutes=DEFAULT_HWC_PURGE_MINUTES)
    assert test_slice["datetime"].min() >= fold2.test_start
    assert test_slice["datetime"].max() < fold2.test_end


def test_generate_oof_scores_callback():
    dt = pd.date_range("2024-01-01", "2024-04-30 23:45", freq="15min")
    df = pd.DataFrame({
        "datetime": dt,
        "symbol": "BTCUSDT",
        "close": np.linspace(100, 200, len(dt)),
    })
    folds = build_master_temporal_folds(df, n_folds=3, embargo_minutes=1440)

    # Mock fit & predict callback that computes mean price on train and outputs diff on test
    def mock_estimator(train_df: pd.DataFrame, test_df: pd.DataFrame, fold: TemporalFold) -> pd.DataFrame:
        mean_val = train_df["close"].mean()
        return pd.DataFrame({
            "direction_score": np.where(test_df["close"] > mean_val, 1.0, -1.0),
            "strength_score": np.full(len(test_df), 0.5),
        }, index=test_df.index)

    # HWC OOF includes Fold 1.
    oof_df = generate_oof_scores(
        df=df,
        folds=folds,
        fit_predict_fn=mock_estimator,
        purge_minutes=1440,
        role="hwc",
    )

    assert not oof_df.empty
    assert "fold_id" in oof_df.columns
    assert "is_seed" not in oof_df.columns
    assert "direction_score" in oof_df.columns
    assert "strength_score" in oof_df.columns
    assert set(oof_df["fold_id"].unique()) == {1, 2, 3}

    # MWC OOF excludes Fold 1 because upstream OOF evidence is unavailable.
    oof_no_seed = generate_oof_predictions(
        df=df,
        folds=folds,
        fit_predict_fn=mock_estimator,
        purge_minutes=1440,
        role="mwc",
    )
    assert set(oof_no_seed["fold_id"].unique()) == {2, 3}
    assert "is_seed" not in oof_no_seed.columns


def test_generate_oof_scores_excludes_seed_by_default_and_purges_exact_boundary():
    dt = pd.date_range("2024-01-01", periods=24, freq="1h")
    df = pd.DataFrame({"datetime": dt, "symbol": "BTCUSDT", "x": 1.0})
    folds = build_master_temporal_folds(df, n_folds=2, embargo_minutes=2)

    seen_train_sizes = []

    def callback(train_df, test_df, fold):
        seen_train_sizes.append(len(train_df))
        return np.zeros(len(test_df))

    oof = generate_oof_scores(df, folds, callback, purge_minutes=2)
    assert not oof.empty
    assert "is_seed" not in oof.columns
    assert len(seen_train_sizes) == 1

    cutoff = folds[1].test_start - pd.Timedelta(minutes=2)
    boundary = df[df["datetime"] == cutoff]
    if not boundary.empty:
        purged = apply_purge_embargo(
            df[df["datetime"] < folds[1].test_start],
            pred_start_dt=folds[1].test_start,
            purge_minutes=2,
        )
        assert cutoff not in set(purged["datetime"])


def test_generate_oof_scores_various_return_types():
    dt = pd.date_range("2024-01-01", "2024-02-28 23:45", freq="1h")
    df = pd.DataFrame({"datetime": dt, "symbol": "ETHUSDT", "x": 1.0})
    folds = build_master_temporal_folds(df, n_folds=2, embargo_minutes=240)

    # 1. Callback returns 1D ndarray
    oof_arr = generate_oof_scores(
        df=df,
        folds=folds,
        fit_predict_fn=lambda tr, te, f: np.ones(len(te)),
    )
    assert "prediction" in oof_arr.columns

    # 2. Callback returns Series
    oof_ser = generate_oof_scores(
        df=df,
        folds=folds,
        fit_predict_fn=lambda tr, te, f: pd.Series(np.zeros(len(te)), name="mcc_score"),
    )
    assert "mcc_score" in oof_ser.columns

    # 3. Callback returns Dict
    oof_dict = generate_oof_scores(
        df=df,
        folds=folds,
        fit_predict_fn=lambda tr, te, f: {"custom_signal": np.ones(len(te))},
    )
    assert "custom_signal" in oof_dict.columns


def test_multi_timeframe_shared_boundaries():
    """Verify that HWC (4H), MWC (1H), and LWC (15m) share exact identical fold boundaries."""
    # LWC: 15m
    dt_lwc = pd.date_range("2024-01-01 00:00", "2024-04-30 23:45", freq="15min")
    df_lwc = pd.DataFrame({"datetime": dt_lwc, "symbol": "BTCUSDT"})

    # MWC: 1H
    dt_mwc = pd.date_range("2024-01-01 00:00", "2024-04-30 23:00", freq="1h")
    df_mwc = pd.DataFrame({"datetime": dt_mwc, "symbol": "BTCUSDT"})

    # HWC: 4H
    dt_hwc = pd.date_range("2024-01-01 00:00", "2024-04-30 20:00", freq="4h")
    df_hwc = pd.DataFrame({"datetime": dt_hwc, "symbol": "BTCUSDT"})

    # Master folds built on master range (LWC)
    master_folds = build_master_temporal_folds(df_lwc, n_folds=3)

    # Check slicing across all three timeframes
    for fold in master_folds:
        # LWC
        lwc_train = fold.get_train_slice(df_lwc, purge_minutes=DEFAULT_LWC_PURGE_MINUTES)
        lwc_test = fold.get_test_slice(df_lwc)
        assert lwc_train["datetime"].max() <= fold.test_start - pd.Timedelta(minutes=DEFAULT_LWC_PURGE_MINUTES)

        # MWC
        mwc_train = fold.get_train_slice(df_mwc, purge_minutes=DEFAULT_MWC_PURGE_MINUTES)
        mwc_test = fold.get_test_slice(df_mwc)
        assert mwc_train["datetime"].max() <= fold.test_start - pd.Timedelta(minutes=DEFAULT_MWC_PURGE_MINUTES)

        # HWC
        hwc_train = fold.get_train_slice(df_hwc, purge_minutes=DEFAULT_HWC_PURGE_MINUTES)
        hwc_test = fold.get_test_slice(df_hwc)
        assert hwc_train["datetime"].max() <= fold.test_start - pd.Timedelta(minutes=DEFAULT_HWC_PURGE_MINUTES)

        if not lwc_test.empty and not mwc_test.empty and not hwc_test.empty:
            assert lwc_test["datetime"].min() >= fold.test_start
            assert mwc_test["datetime"].min() >= fold.test_start
            assert hwc_test["datetime"].min() >= fold.test_start
