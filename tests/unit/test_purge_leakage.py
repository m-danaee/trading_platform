"""Regression tests for role-specific purge and seed-fold eligibility."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.mtf import cross_fitting as _cross_fitting


def _make_synthetic_df(
    periods: int = 1_000,
    *,
    freq: str = "15min",
) -> pd.DataFrame:
    """Create bars with a clear future spike for forward-label checks."""
    dates = pd.date_range("2024-01-01", periods=periods, freq=freq, tz="UTC")
    spike = np.zeros(periods, dtype=np.int8)
    spike[periods // 2 :] = 1
    return pd.DataFrame(
        {
            "datetime": dates.tz_localize(None),
            "symbol": "BTCUSDT",
            "close": np.arange(periods, dtype=float),
            "future_spike": spike,
        }
    )


def _purge_for_role(role: str) -> int:
    role = role.lower()
    helper = getattr(_cfg, "purge_for_role", None)
    if callable(helper):
        return int(helper(role))

    config_name = f"MTF_{role.upper()}_PURGE_MINUTES"
    if hasattr(_cfg, config_name):
        return int(getattr(_cfg, config_name))
    return int(getattr(_cross_fitting, f"DEFAULT_{role.upper()}_PURGE_MINUTES"))


def _utc_naive_timestamps(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["datetime"], utc=True).dt.tz_localize(None)


def _with_forward_label(
    df: pd.DataFrame,
    test_start: pd.Timestamp,
    role: str,
    purge_minutes: int,
) -> pd.DataFrame:
    labeled = df.copy()
    timestamps = _utc_naive_timestamps(labeled)
    future_spike = (timestamps >= test_start).astype(np.int8)
    labeled["future_spike"] = future_spike
    horizon_bars = max(1, purge_minutes // 15)
    labeled[f"{role}_label"] = future_spike.shift(-horizon_bars, fill_value=0)
    return labeled


def test_future_spike_is_purged_for_hwc_mwc_and_lwc():
    df = _make_synthetic_df()
    folds = _cross_fitting.build_master_temporal_folds(
        df,
        n_folds=3,
        embargo_minutes=0,
    )
    fold = folds[1]
    timestamps = _utc_naive_timestamps(df)

    for role in ("hwc", "mwc", "lwc"):
        purge_minutes = _purge_for_role(role)
        labeled = _with_forward_label(
            df,
            fold.test_start,
            role,
            purge_minutes,
        )
        raw_train = labeled.loc[timestamps < fold.test_start]
        purged = _cross_fitting.apply_purge_embargo(
            raw_train,
            pred_start_dt=fold.test_start,
            purge_minutes=purge_minutes,
        )

        assert not purged.empty
        last_train_time = _utc_naive_timestamps(purged).max()
        assert last_train_time + pd.Timedelta(minutes=purge_minutes) < fold.test_start
        assert int(purged[f"{role}_label"].max()) == 0


def _generate_role_oof(df, folds, role: str, purge_minutes: int) -> pd.DataFrame:
    """Generate OOF scores through the role-aware production API."""
    callback = lambda train, test, fold: np.zeros(len(test))
    return _cross_fitting.generate_oof_scores(
        df,
        folds,
        callback,
        purge_minutes=purge_minutes,
        role=role,
    )


def test_hwc_seed_fold_is_usable_but_mwc_seed_fold_is_unavailable():
    """HWC may use Fold 1; MWC requires upstream OOF history."""
    df = _make_synthetic_df(periods=400)
    folds = _cross_fitting.build_master_temporal_folds(
        df,
        n_folds=3,
        embargo_minutes=0,
    )
    hwc_oof = _generate_role_oof(df, folds, "hwc", _purge_for_role("hwc"))
    mwc_oof = _generate_role_oof(df, folds, "mwc", _purge_for_role("mwc"))

    assert 1 in set(hwc_oof["fold_id"])
    assert 1 not in set(mwc_oof["fold_id"])
