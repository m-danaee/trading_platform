# tests/unit/test_multi_timeframe.py
import numpy as np
import pandas as pd
import pytest
from gpu_fuzzy_trader.data.multi_timeframe import (
    build_complete_higher_bars,
    compute_timeframe_features,
    align_htf_features_causal,
)


def test_build_complete_higher_bars_utc_and_completeness():
    # 20 15m bars starting at 00:00 UTC
    dt = pd.date_range("2024-01-01 00:00", periods=20, freq="15min")
    df = pd.DataFrame({
        "datetime": dt,
        "symbol": "BTCUSDT",
        "open": np.linspace(100, 120, 20),
        "high": np.linspace(101, 121, 20),
        "low": np.linspace(99, 119, 20),
        "close": np.linspace(100.5, 120.5, 20),
        "volume": np.ones(20) * 10.0,
    })
    # 1H bars (4 x 15m) -> 5 complete bars
    mwc = build_complete_higher_bars(df, 60)
    assert len(mwc) == 5
    assert mwc["datetime"].iloc[0] == pd.Timestamp("2024-01-01 00:00")
    assert mwc["close"].iloc[0] == df["close"].iloc[3]
    assert mwc["volume"].iloc[0] == 40.0

    # 4H bars (16 x 15m) -> exactly 1 complete 4H bar (first 16 rows), 4 rows dropped as incomplete
    hwc = build_complete_higher_bars(df, 240)
    assert len(hwc) == 1
    assert hwc["datetime"].iloc[0] == pd.Timestamp("2024-01-01 00:00")
    assert hwc["volume"].iloc[0] == 160.0


def test_causal_alignment_no_lookahead():
    # 16 15m bars for 00:00 -> 04:00 4H candle
    dt = pd.date_range("2024-01-01 00:00", periods=18, freq="15min")
    df = pd.DataFrame({
        "datetime": dt,
        "symbol": "BTCUSDT",
        "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 10.0,
    })
    hwc = build_complete_higher_bars(df, 240)
    hwc["hwc_feature"] = 42.0
    aligned = align_htf_features_causal(df, hwc, 240)

    # 00:00 to 03:45 15m rows execute at 00:15 to 04:00.
    # The 4H bar closes at 04:00, so row at 03:45 (executing at 04:00) sees it. Earlier rows get NaN.
    assert np.isnan(aligned.loc[df["datetime"] == "2024-01-01 03:30", "hwc_feature"].iloc[0])
    assert aligned.loc[df["datetime"] == "2024-01-01 03:45", "hwc_feature"].iloc[0] == 42.0


def test_missing_constituent_drops_incomplete_bucket():
    # 16 15m bars, but missing bar at 01:30
    dt = pd.date_range("2024-01-01 00:00", periods=16, freq="15min")
    df = pd.DataFrame({
        "datetime": dt,
        "symbol": "BTCUSDT",
        "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 10.0,
    })
    df_missing = df[df["datetime"] != "2024-01-01 01:30"].reset_index(drop=True)
    # For 1H bars: 00:00-01:00 complete (4 bars), 01:00-02:00 has only 3 bars (dropped), 02:00-03:00 complete (4 bars), 03:00-04:00 complete (4 bars)
    mwc = build_complete_higher_bars(df_missing, 60)
    assert len(mwc) == 3
    assert pd.Timestamp("2024-01-01 01:00") not in mwc["datetime"].values

    # For 4H bars: 00:00-04:00 has only 15 bars instead of 16 (dropped)
    hwc = build_complete_higher_bars(df_missing, 240)
    assert len(hwc) == 0


def test_duplicate_constituent_is_rejected_instead_of_counted_as_complete():
    dt = pd.date_range("2024-01-01 00:00", periods=4, freq="15min")
    df = pd.DataFrame({
        "datetime": dt,
        "symbol": "BTCUSDT",
        "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 10.0,
    })
    duplicate = df.copy()
    duplicate.loc[3, "datetime"] = duplicate.loc[2, "datetime"]
    with pytest.raises(ValueError, match="duplicate constituent"):
        build_complete_higher_bars(duplicate, 60)


def test_non_utc_offsets_are_normalized_before_bucket_alignment():
    # These timestamps represent 00:00, 00:15, 00:30, 00:45 UTC despite the
    # source carrying a +05:30 offset.
    dt = pd.date_range("2024-01-01 05:30", periods=4, freq="15min", tz="Asia/Kolkata")
    df = pd.DataFrame({
        "datetime": dt,
        "symbol": "BTCUSDT",
        "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 10.0,
    })
    bars = build_complete_higher_bars(df, 60)
    assert bars["datetime"].iloc[0] == pd.Timestamp("2024-01-01 00:00")
    assert bars["datetime"].dt.tz is None


def test_htf_ohlcv_cannot_overwrite_lwc_execution_data():
    dt = pd.date_range("2024-01-01 00:00", periods=8, freq="15min")
    lwc = pd.DataFrame({
        "datetime": dt,
        "symbol": "BTCUSDT",
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10.0,
    })
    htf = pd.DataFrame({
        "datetime": [pd.Timestamp("2024-01-01 00:00")],
        "symbol": ["BTCUSDT"],
        "open": [200.0], "high": [205.0], "low": [195.0], "close": [202.0],
        "volume": [40.0], "derived_feature": [7.0],
    })
    aligned = align_htf_features_causal(lwc, htf, 60)
    assert np.allclose(aligned["open"], lwc["open"])
    assert np.allclose(aligned["close"], lwc["close"])
    assert aligned.loc[aligned["datetime"] == "2024-01-01 00:45", "derived_feature"].iloc[0] == 7.0


def test_compute_timeframe_features():
    # Generate 50 1H bars
    dt = pd.date_range("2024-01-01 00:00", periods=50, freq="1h")
    np.random.seed(42)
    close = 100.0 + np.cumsum(np.random.randn(50))
    high = close + np.abs(np.random.randn(50)) + 0.5
    low = close - np.abs(np.random.randn(50)) - 0.5
    open_ = close + np.random.randn(50) * 0.2
    volume = np.random.uniform(100, 500, size=50)

    df_bars = pd.DataFrame({
        "datetime": dt,
        "symbol": "BTCUSDT",
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })

    feats = compute_timeframe_features(df_bars, 60)
    assert len(feats) == 50
    # Check expected feature presence
    expected_cols = [
        "rsi_14",
        "atr_14",
        "kama_10",
        "kama_slope_10",
        "bollinger_pct_b",
        "bollinger_bandwidth",
        "realized_volatility",
        "momentum_roc",
        "relative_volume_20",
    ]
    for col in expected_cols:
        assert col in feats.columns
    # Warmup produces NaNs initially
    assert feats["rsi_14"].isna().iloc[0]

    rule_feats = compute_timeframe_features(
        df_bars, 60, include_raw_features=False,
    )
    assert "atr_14" not in rule_feats.columns
    assert "kama_10" not in rule_feats.columns
    derived = rule_feats.drop(
        columns=["datetime", "symbol", "open", "high", "low", "close", "volume"],
    )
    finite = derived.to_numpy(dtype=float)
    assert np.nanmin(finite) >= -1.0
    assert np.nanmax(finite) <= 1.0


def test_feature_representation_is_causal_under_future_mutation():
    dt = pd.date_range("2024-01-01", periods=80, freq="15min")
    close = 100.0 + np.cumsum(np.sin(np.arange(80) / 4.0))
    base = pd.DataFrame({
        "datetime": dt,
        "symbol": "BTCUSDT",
        "open": close,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": np.linspace(10.0, 20.0, 80),
    })
    mutated = base.copy()
    mutated.loc[50:, "close"] = 500.0
    mutated.loc[50:, "open"] = 500.0
    mutated.loc[50:, "high"] = 501.0
    mutated.loc[50:, "low"] = 499.0
    mutated.loc[50:, "volume"] = 1000.0

    original_features = compute_timeframe_features(
        base, 15, include_raw_features=False,
    )
    mutated_features = compute_timeframe_features(
        mutated, 15, include_raw_features=False,
    )
    derived = [
        column for column in original_features.columns
        if column not in {"datetime", "symbol", "open", "high", "low", "close", "volume"}
    ]
    pd.testing.assert_frame_equal(
        original_features.loc[:49, derived].reset_index(drop=True),
        mutated_features.loc[:49, derived].reset_index(drop=True),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_causal_invariance_future_mutation():
    # Modify data in unfinished HTF bar and verify zero change in previously aligned features
    dt = pd.date_range("2024-01-01 00:00", periods=32, freq="15min") # 2 4H bars
    df = pd.DataFrame({
        "datetime": dt,
        "symbol": "BTCUSDT",
        "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 10.0,
    })
    hwc1 = build_complete_higher_bars(df, 240)
    hwc1["feature"] = [10.0, 20.0]
    aligned1 = align_htf_features_causal(df, hwc1, 240)

    # Mutate 2nd 4H candle's future data (say rows 20-31)
    df_mutated = df.copy()
    df_mutated.loc[20:31, "close"] = 999.0
    # Rebuild only the 2nd candle with a mutated feature
    hwc2 = hwc1.copy()
    hwc2.loc[1, "feature"] = 999.0
    aligned2 = align_htf_features_causal(df_mutated, hwc2, 240)

    # Rows prior to 04:00 (first 16 rows) must be identical
    pd.testing.assert_series_equal(
        aligned1.loc[:15, "feature"],
        aligned2.loc[:15, "feature"],
    )


def test_multi_symbol_isolation():
    # Verify independent aggregation and causal alignment per symbol
    dt = pd.date_range("2024-01-01 00:00", periods=20, freq="15min")
    df_btc = pd.DataFrame({
        "datetime": dt,
        "symbol": "BTCUSDT",
        "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 10.0,
    })
    df_eth = pd.DataFrame({
        "datetime": dt,
        "symbol": "ETHUSDT",
        "open": 200.0, "high": 205.0, "low": 195.0, "close": 202.0, "volume": 20.0,
    })
    df = pd.concat([df_btc, df_eth], ignore_index=True)

    mwc = build_complete_higher_bars(df, 60)
    assert len(mwc) == 10  # 5 bars for BTC + 5 bars for ETH
    assert set(mwc["symbol"].unique()) == {"BTCUSDT", "ETHUSDT"}

    mwc.loc[mwc["symbol"] == "BTCUSDT", "sym_feat"] = 1.0
    mwc.loc[mwc["symbol"] == "ETHUSDT", "sym_feat"] = 2.0

    aligned = align_htf_features_causal(df, mwc, 60)
    # Check BTC at 01:00 vs ETH at 01:00
    btc_row = aligned[(aligned["symbol"] == "BTCUSDT") & (aligned["datetime"] == "2024-01-01 00:45")]
    eth_row = aligned[(aligned["symbol"] == "ETHUSDT") & (aligned["datetime"] == "2024-01-01 00:45")]
    assert btc_row["sym_feat"].iloc[0] == 1.0
    assert eth_row["sym_feat"].iloc[0] == 2.0


def test_empty_and_invalid_inputs():
    empty_df = pd.DataFrame(columns=["datetime", "symbol", "open", "high", "low", "close", "volume"])
    bars = build_complete_higher_bars(empty_df, 60)
    assert bars.empty

    feats = compute_timeframe_features(empty_df, 60)
    assert feats.empty

    aligned = align_htf_features_causal(empty_df, empty_df, 60)
    assert aligned.empty

    with pytest.raises(ValueError):
        build_complete_higher_bars(pd.DataFrame({"a": [1]}), 60)
