"""Master Temporal Folds, Purged Embargo, and Out-of-Fold (OOF) Cross-Fitting.

Coordinates expanding temporal train/test splits shared identically across all
timeframes (HWC, MWC, LWC), strictly enforces forward-lookahead purge embargoes,
and aggregates out-of-fold validation scores.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Callable, Sequence, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Standard default purge durations in minutes
DEFAULT_HWC_PURGE_MINUTES: int = 1440  # K_HWC (6) * 240m = 24h
DEFAULT_MWC_PURGE_MINUTES: int = 240   # K_MWC (4) * 60m = 4h
DEFAULT_LWC_PURGE_MINUTES: int = 720   # Max trade duration (48) * 15m = 12h


@dataclass(frozen=True)
class TemporalFold:
    """Dataclass defining a single master temporal cross-fitting fold.

    Attributes:
        fold_id: 1-based integer index of the fold.
        train_start: Starting timestamp of the training interval.
        train_end: Nominal ending timestamp of the training interval (before purging).
        test_start: Starting timestamp of the out-of-fold test interval.
        test_end: Ending timestamp of the out-of-fold test interval.
        is_seed: True if this is the initial seed fold (Fold 1) lacking prior OOF history.
        purge_minutes: Optional default purge duration in minutes for this fold.
    """

    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    is_seed: bool = False
    purge_minutes: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert fold metadata to JSON-serializable dictionary."""
        return {
            "fold_id": int(self.fold_id),
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
            "is_seed": bool(self.is_seed),
            "purge_minutes": int(self.purge_minutes),
        }

    def get_train_slice(
        self,
        df: pd.DataFrame,
        purge_minutes: int | None = None,
        datetime_col: str = "datetime",
    ) -> pd.DataFrame:
        """Extract purged training data for this fold from a DataFrame."""
        p_min = self.purge_minutes if purge_minutes is None else purge_minutes
        dt_series = _get_datetime_series(df, datetime_col)
        raw_train = df.loc[(dt_series >= self.train_start) & (dt_series < self.test_start)].copy()
        return apply_purge_embargo(
            train_df=raw_train,
            pred_start_dt=self.test_start,
            purge_minutes=p_min,
            datetime_col=datetime_col,
        )

    def get_test_slice(
        self,
        df: pd.DataFrame,
        datetime_col: str = "datetime",
        inclusive_end: bool = False,
    ) -> pd.DataFrame:
        """Extract out-of-fold test data for this fold from a DataFrame."""
        dt_series = _get_datetime_series(df, datetime_col)
        if inclusive_end:
            mask = (dt_series >= self.test_start) & (dt_series <= self.test_end)
        else:
            mask = (dt_series >= self.test_start) & (dt_series < self.test_end)
        return df.loc[mask].copy()


def _get_datetime_series(df: pd.DataFrame, datetime_col: str = "datetime") -> pd.Series:
    """Extract and validate datetime Series from column or DatetimeIndex."""
    if df.empty:
        return pd.Series(dtype="datetime64[ns, UTC]")

    if datetime_col in df.columns:
        dt = df[datetime_col]
        parsed = pd.to_datetime(dt, errors="raise", utc=True)
        return parsed.dt.tz_localize(None)
    elif isinstance(df.index, pd.DatetimeIndex):
        parsed_index = pd.to_datetime(df.index, errors="raise", utc=True).tz_localize(None)
        return pd.Series(parsed_index, index=df.index)
    else:
        raise ValueError(
            f"Cannot find datetime column '{datetime_col}' or DatetimeIndex in DataFrame."
        )


def build_master_temporal_folds(
    df: pd.DataFrame,
    n_folds: int = 4,
    embargo_minutes: int = 1440,
    datetime_col: str = "datetime",
) -> list[TemporalFold]:
    """Construct expanding master temporal fold boundaries across all timeframes.

    Divides the timeline [T0, T_max] into (n_folds + 1) contiguous segments
    [B0, B1, ..., B_{n+1}] such that:
      - Fold 1 (Seed):  Train [B0, B1], Test [B1, B2], is_seed = True
      - Fold 2:         Train [B0, B2], Test [B2, B3], is_seed = False
      - ...
      - Fold N:         Train [B0, BN], Test [BN, B_{n+1}], is_seed = False

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing datetime series/index.
    n_folds : int, default 4
        Number of out-of-fold evaluation folds (n_folds >= 1).
    embargo_minutes : int, default 1440
        Default purge embargo gap duration in minutes.
    datetime_col : str, default "datetime"
        Name of the datetime column.

    Returns
    -------
    list[TemporalFold]
        List of TemporalFold instances defining the shared cross-fitting splits.
    """
    if df.empty:
        raise ValueError("Cannot build temporal folds from an empty DataFrame.")
    if n_folds < 1:
        raise ValueError(f"n_folds must be >= 1, got {n_folds}")

    dt_series = _get_datetime_series(df, datetime_col)
    t_min = dt_series.min()
    t_max = dt_series.max()

    if pd.isna(t_min) or pd.isna(t_max) or t_min >= t_max:
        raise ValueError("DataFrame datetime span must have at least 2 distinct timestamps.")

    n_blocks = n_folds + 1
    total_delta = t_max - t_min
    step_delta = total_delta / n_blocks

    boundaries: list[pd.Timestamp] = [t_min + i * step_delta for i in range(n_blocks)]
    boundaries.append(t_max)

    folds: list[TemporalFold] = []
    for k in range(1, n_folds + 1):
        fold = TemporalFold(
            fold_id=k,
            train_start=boundaries[0],
            train_end=boundaries[k],
            test_start=boundaries[k],
            test_end=boundaries[k + 1],
            is_seed=(k == 1),
            purge_minutes=int(embargo_minutes),
        )
        folds.append(fold)

    return folds


def apply_purge_embargo(
    train_df: pd.DataFrame,
    pred_start_dt: Union[pd.Timestamp, str, None] = None,
    purge_minutes: int = 0,
    datetime_col: str = "datetime",
    *,
    test_start: Union[pd.Timestamp, str, None] = None,
) -> pd.DataFrame:
    """Purge training rows whose forward label/trade outcome horizon extends into test_start.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training set DataFrame.
    pred_start_dt : pd.Timestamp, str, or None
        Start timestamp of the prediction / out-of-fold evaluation window.
    purge_minutes : int, default 0
        Embargo / forward horizon gap in minutes to strictly purge.
    datetime_col : str, default "datetime"
        Name of datetime column.
    test_start : pd.Timestamp, str, or None, optional
        Keyword alias for pred_start_dt.

    Returns
    -------
    pd.DataFrame
        Filtered training DataFrame guaranteed to contain only bars with
        timestamp < (test_start - purge_minutes).
    """
    if train_df.empty or purge_minutes <= 0:
        return train_df.copy()

    start_dt = pred_start_dt if pred_start_dt is not None else test_start
    if start_dt is None:
        raise ValueError("Either pred_start_dt or test_start must be provided.")

    target_dt = pd.to_datetime(start_dt, errors="raise", utc=True).tz_localize(None)
    cutoff_dt = target_dt - pd.Timedelta(minutes=int(purge_minutes))

    if datetime_col in train_df.columns:
        dt_series = train_df[datetime_col]
        dt_series = pd.to_datetime(dt_series, errors="raise", utc=True).dt.tz_localize(None)
        # A label whose forward horizon closes exactly at the prediction start
        # still touches the prediction interval and must be purged.
        mask = dt_series < cutoff_dt
        return train_df.loc[mask].copy()
    elif isinstance(train_df.index, pd.DatetimeIndex):
        index = pd.to_datetime(train_df.index, errors="raise", utc=True).tz_localize(None)
        mask = index < cutoff_dt
        return train_df.loc[mask].copy()
    else:
        raise ValueError(
            f"Cannot find datetime column '{datetime_col}' or DatetimeIndex in train_df."
        )


def generate_oof_scores(
    df: pd.DataFrame,
    folds: Sequence[TemporalFold],
    fit_predict_fn: Callable[[pd.DataFrame, pd.DataFrame, TemporalFold], Any],
    purge_minutes: int | None = None,
    datetime_col: str = "datetime",
    exclude_seed: bool = True,
) -> pd.DataFrame:
    """Coordinate fitting on purged training folds and generating out-of-fold predictions.

    Parameters
    ----------
    df : pd.DataFrame
        Full master dataset DataFrame containing features and timestamps.
    folds : Sequence[TemporalFold]
        Temporal folds built by ``build_master_temporal_folds``.
    fit_predict_fn : Callable[[pd.DataFrame, pd.DataFrame, TemporalFold], Any]
        Callback receiving (purged_train_df, test_df, fold) and returning
        predictions (as DataFrame, Series, ndarray, or dict) aligned to test_df.
    purge_minutes : int or None, default None
        Override purge minutes for training set. If None, uses fold.purge_minutes.
    datetime_col : str, default "datetime"
        Name of the datetime column.
    exclude_seed : bool, default True
        Excludes Fold 1 (the seed period without prior OOF history) by default.
        Set to False only for diagnostics that explicitly need seed predictions.

    Returns
    -------
    pd.DataFrame
        Combined out-of-fold prediction DataFrame with columns:
        ``fold_id``, ``is_seed``, and prediction fields.
    """
    if df.empty or not folds:
        return pd.DataFrame()

    dt_series = _get_datetime_series(df, datetime_col)
    n_folds = len(folds)
    results: list[pd.DataFrame] = []

    for i, fold in enumerate(folds):
        if exclude_seed and fold.is_seed:
            continue

        p_min = fold.purge_minutes if purge_minutes is None else purge_minutes

        # 1. Train slice: [train_start, test_start)
        train_mask = (dt_series >= fold.train_start) & (dt_series < fold.test_start)
        raw_train = df.loc[train_mask].copy()
        purged_train = apply_purge_embargo(
            train_df=raw_train,
            pred_start_dt=fold.test_start,
            purge_minutes=p_min,
            datetime_col=datetime_col,
        )

        # 2. Test slice: [test_start, test_end) or [test_start, test_end] for the last fold
        is_last = (i == n_folds - 1)
        if is_last:
            test_mask = (dt_series >= fold.test_start) & (dt_series <= fold.test_end)
        else:
            test_mask = (dt_series >= fold.test_start) & (dt_series < fold.test_end)

        test_slice = df.loc[test_mask].copy()
        if test_slice.empty:
            continue

        # 3. Fit & Predict callback
        preds = fit_predict_fn(purged_train, test_slice, fold)
        pred_df = _format_fold_predictions(preds, test_slice, fold, datetime_col)
        results.append(pred_df)

    if not results:
        return pd.DataFrame()

    combined = pd.concat(results, axis=0, ignore_index=True)
    if datetime_col in combined.columns:
        combined = combined.sort_values(datetime_col).reset_index(drop=True)

    return combined


# Alias for generate_oof_scores
generate_oof_predictions = generate_oof_scores


def _format_fold_predictions(
    preds: Any,
    test_slice: pd.DataFrame,
    fold: TemporalFold,
    datetime_col: str,
) -> pd.DataFrame:
    """Format callback prediction output into a standardized DataFrame."""
    if isinstance(preds, pd.DataFrame):
        out = preds.copy()
    elif isinstance(preds, pd.Series):
        col_name = preds.name if preds.name else "prediction"
        out = pd.DataFrame({col_name: preds.values}, index=test_slice.index)
    elif isinstance(preds, np.ndarray):
        if preds.ndim == 1:
            out = pd.DataFrame({"prediction": preds}, index=test_slice.index)
        else:
            cols = [f"pred_{j}" for j in range(preds.shape[1])]
            out = pd.DataFrame(preds, columns=cols, index=test_slice.index)
    elif isinstance(preds, dict):
        out = pd.DataFrame(preds, index=test_slice.index)
    else:
        out = pd.DataFrame({"prediction": preds}, index=test_slice.index)

    # Attach datetime and symbol if present in test_slice and not in out
    if datetime_col in test_slice.columns and datetime_col not in out.columns:
        out.insert(0, datetime_col, test_slice[datetime_col].values)
    if "symbol" in test_slice.columns and "symbol" not in out.columns:
        out.insert(1, "symbol", test_slice["symbol"].values)

    out["fold_id"] = fold.fold_id
    out["is_seed"] = fold.is_seed
    return out


def export_fold_boundaries(folds: Sequence[TemporalFold]) -> list[dict[str, Any]]:
    """Export list of fold boundaries to dictionary list for manifest serialization."""
    return [fold.to_dict() for fold in folds]


def validate_master_temporal_folds(folds: Sequence[TemporalFold]) -> bool:
    """Validate that temporal folds are strictly monotonic and expanding.

    Checks:
    1. Fold IDs are strictly sequential starting at 1.
    2. Fold 1 has is_seed = True; subsequent folds have is_seed = False.
    3. Train start is identical across all folds.
    4. Train end == test start for every fold.
    5. Test intervals are contiguous (test_end_{k} == test_start_{k+1}).
    """
    if not folds:
        return False

    first = folds[0]
    if first.fold_id != 1 or not first.is_seed:
        return False

    base_train_start = first.train_start

    for i, fold in enumerate(folds):
        expected_id = i + 1
        if fold.fold_id != expected_id:
            return False
        if fold.train_start != base_train_start:
            return False
        if fold.train_end != fold.test_start:
            return False
        if fold.test_start >= fold.test_end:
            return False
        if i == 0 and not fold.is_seed:
            return False
        if i > 0 and fold.is_seed:
            return False
        if i > 0:
            prev_fold = folds[i - 1]
            if prev_fold.test_end != fold.test_start:
                return False

    return True
