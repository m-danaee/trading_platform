"""
data/splitter.py — Data_Splitter

Per-symbol chronological splits for Phases 2–3:

- ``holdout_70_30`` (legacy): single 70/30 train/validation split.
- ``purged_rolling_cv``: K purged expanding-window folds with embargo; the last
  fold's train/val blocks are persisted as train_70 / validation_30 for Phases 4–5.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import pandas as pd

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.backtest.df_slim import downcast_numeric_df
from gpu_fuzzy_trader.config import (
    CV_FOLDS_MANIFEST_PATH,
    TRAIN_70_PATH,
    VALIDATION_30_PATH,
)
from gpu_fuzzy_trader.data.cv_folds import (
    PurgedFold,
    build_purged_rolling_folds,
    holdout_70_30_split,
    primary_holdout_from_folds,
)

logger = logging.getLogger(__name__)


class Data_Splitter:
    """Chronological train/validation splitter with optional purged rolling CV."""

    def split_and_persist(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[PurgedFold]]:
        """
        Build train/validation DataFrames and persist to Parquet.

        When ``SPLIT_MODE == "purged_rolling_cv"``, also writes a fold manifest.
        Persisted train_70 / validation_30 always match the **last** CV fold
        (or the 70/30 holdout when legacy mode is selected).

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, list[PurgedFold]]
            ``(train_df, validation_df, cv_folds)``. *cv_folds* is non-empty only
            in ``purged_rolling_cv`` mode.
        """
        mode = str(_cfg.SPLIT_MODE).strip().lower()
        folds: list[PurgedFold] = []

        if mode == "purged_rolling_cv":
            folds = build_purged_rolling_folds(df)
            if not folds:
                logger.warning(
                    "Purged rolling CV produced no folds (min_train=%d bars, "
                    "K=%d); falling back to holdout_70_30",
                    int(_cfg.CV_MIN_TRAIN_MONTHS * 30 * _cfg.CV_BARS_PER_DAY),
                    _cfg.CV_N_FOLDS,
                )
                train_df, validation_df = holdout_70_30_split(df)
            else:
                train_df, validation_df = primary_holdout_from_folds(folds)
                self._persist_cv_manifest(folds, train_df, validation_df)
        elif mode == "holdout_70_30":
            train_df, validation_df = holdout_70_30_split(df)
        else:
            raise ValueError(
                f"Unknown SPLIT_MODE={_cfg.SPLIT_MODE!r}; "
                "use 'holdout_70_30' or 'purged_rolling_cv'"
            )

        train_df = downcast_numeric_df(train_df)
        validation_df = downcast_numeric_df(validation_df)

        train_df.to_parquet(TRAIN_70_PATH, index=False)
        validation_df.to_parquet(VALIDATION_30_PATH, index=False)

        return train_df, validation_df, folds

    def build_cv_folds(self, df: pd.DataFrame) -> list[PurgedFold]:
        """Return purged rolling folds (empty when legacy split mode is active)."""
        mode = str(_cfg.SPLIT_MODE).strip().lower()
        if mode != "purged_rolling_cv":
            return []
        return build_purged_rolling_folds(df)

    @staticmethod
    def load_cv_folds_from_manifest() -> list[dict[str, Any]] | None:
        """Load fold metadata written by ``split_and_persist`` (not full DataFrames)."""
        if not os.path.exists(CV_FOLDS_MANIFEST_PATH):
            return None
        with open(CV_FOLDS_MANIFEST_PATH, encoding="utf-8") as fh:
            payload = json.load(fh)
        return payload.get("folds")

    @staticmethod
    def _persist_cv_manifest(
        folds: list[PurgedFold],
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> None:
        manifest_dir = os.path.dirname(CV_FOLDS_MANIFEST_PATH)
        if manifest_dir:
            os.makedirs(manifest_dir, exist_ok=True)

        fold_summaries: list[dict[str, Any]] = []
        for fold in folds:
            fold_summaries.append(
                {
                    "fold_index": fold.fold_index,
                    "train_rows": len(fold.train_df),
                    "val_rows": len(fold.val_df),
                    "train_datetime_min": _dt_min(fold.train_df),
                    "train_datetime_max": _dt_max(fold.train_df),
                    "val_datetime_min": _dt_min(fold.val_df),
                    "val_datetime_max": _dt_max(fold.val_df),
                }
            )

        payload = {
            "split_mode": "purged_rolling_cv",
            "n_folds": len(folds),
            "embargo_bars": int(_cfg.CV_EMBARGO_BARS),
            "min_train_months": float(_cfg.CV_MIN_TRAIN_MONTHS),
            "folds": fold_summaries,
            "persisted_holdout": {
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "note": "train_70/validation_30 parquet = last fold",
            },
        }
        with open(CV_FOLDS_MANIFEST_PATH, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)


def _dt_min(frame: pd.DataFrame) -> str | None:
    if "datetime" not in frame.columns or frame.empty:
        return None
    return str(frame["datetime"].min())


def _dt_max(frame: pd.DataFrame) -> str | None:
    if "datetime" not in frame.columns or frame.empty:
        return None
    return str(frame["datetime"].max())


def split_and_persist(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[PurgedFold]]:
    """Module-level wrapper around ``Data_Splitter.split_and_persist``."""
    return Data_Splitter().split_and_persist(df)


def build_cv_folds(df: pd.DataFrame) -> list[PurgedFold]:
    """Module-level wrapper for purged rolling folds."""
    return Data_Splitter().build_cv_folds(df)
