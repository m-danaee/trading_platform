"""
Purged expanding walk-forward folds for train_new.csv.

Per-symbol chronological folds with an embargo (purge) gap between train and
validation blocks so label look-ahead (MAX_HOLD_CANDLES) does not leak.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config as _cfg

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PurgedFold:
    """One expanding train + validation slice (per-symbol boundaries merged)."""

    fold_id: int
    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    train_end_bar: int
    valid_start_bar: int
    valid_end_bar: int
    n_train_rows: int
    n_valid_rows: int
    is_holdout: bool = False


@dataclass
class FoldMetricsSummary:
    folds: int
    worst_return_pct: float
    worst_profit_factor: float
    worst_sortino_ratio: float
    worst_drawdown_pct: float
    min_trades: int
    mean_return_pct: float
    mean_profit_factor: float
    metrics: list[dict] = field(default_factory=list)


def _bar_index_col(df: pd.DataFrame) -> str:
    if "_symbol_bar_index" in df.columns:
        return "_symbol_bar_index"
    return "_symbol_bar_index"


def _sort_group(group: pd.DataFrame) -> pd.DataFrame:
    if "datetime" in group.columns:
        return group.sort_values("datetime")
    return group


def _split_symbol_segments(
    group: pd.DataFrame,
    *,
    holdout_fraction: float,
    n_cv_folds: int,
    min_train_fraction: float = 0.25,
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    """
    Return CV valid (start, end) bar ranges and holdout (start, end) for one symbol.

    The first CV valid block starts at ``floor(n * min_train_fraction)`` so every
    fold has at least that many training bars before it.  The remaining prefix
    (after min_train) is divided into ``n_cv_folds`` equal contiguous segments.
    """
    g = _sort_group(group).reset_index(drop=True)
    n = len(g)
    if n < 2:
        return [], (0, max(0, n - 1))

    holdout_n = max(1, int(math.floor(n * holdout_fraction)))
    prefix_n = n - holdout_n
    holdout_start = prefix_n
    holdout_end = n - 1

    min_train_start = max(1, int(math.floor(n * min_train_fraction)))

    cv_valid_ranges: list[tuple[int, int]] = []
    if n_cv_folds > 0 and prefix_n > min_train_start:
        remaining = prefix_n - min_train_start
        base, rem = divmod(remaining, n_cv_folds)
        start = min_train_start
        for i in range(n_cv_folds):
            size = base + 1 if i < rem else base
            if size <= 0:
                continue
            end = min(prefix_n - 1, start + size - 1)
            if start <= end and end < holdout_start:
                cv_valid_ranges.append((start, end))
            start = end + 1

    return cv_valid_ranges, (holdout_start, holdout_end)


def _slice_symbol_rows(
    group: pd.DataFrame,
    start_bar: int,
    end_bar: int,
) -> pd.DataFrame:
    g = _sort_group(group)
    idx = g[_bar_index_col(g)].values
    mask = (idx >= start_bar) & (idx <= end_bar)
    return g.loc[mask].copy()


def build_forbidden_ranges(
    cv_folds: list[PurgedFold],
    embargo: int | None = None,
) -> list[tuple[int, int]]:
    """Build per-symbol forbidden ``(start_bar, end_bar)`` ranges from folds.

    Each fold's valid region is forbidden. The embargo is pulled back from
    each valid start so train labels (``MAX_HOLD_CANDLES`` lookahead) cannot
    peek into the valid block.
    """
    if not cv_folds:
        return []
    if embargo is None:
        embargo = int(
            getattr(_cfg, "PURGED_WF_EMBARGO_CANDLES", _cfg.MAX_HOLD_CANDLES)
        )
    ranges = [(f.valid_start_bar, f.valid_end_bar) for f in cv_folds]
    return [(max(0, int(s) - int(embargo)), int(e)) for s, e in ranges]


def mask_df_to_safe_region(
    df: pd.DataFrame,
    forbidden_ranges: list[tuple[int, int]],
    bar_col: str = "_symbol_bar_index",
) -> pd.DataFrame:
    """Drop rows whose per-symbol bar index falls inside any forbidden range."""
    if not forbidden_ranges or df.empty or bar_col not in df.columns:
        return df
    mask = np.ones(len(df), dtype=bool)
    idx = df[bar_col].to_numpy()
    for s, e in forbidden_ranges:
        mask &= ~((idx >= int(s)) & (idx <= int(e)))
    return df.loc[mask].reset_index(drop=True)


def _purge_train(
    train: pd.DataFrame,
    valid_start_bar: int,
    embargo: int,
) -> pd.DataFrame:
    """Drop train rows whose label horizon could overlap the valid block."""
    if train.empty:
        return train
    col = _bar_index_col(train)
    cutoff = int(valid_start_bar) - int(embargo)
    return train.loc[train[col] < cutoff].copy()


def _build_fold_from_ranges(
    df: pd.DataFrame,
    fold_id: int,
    valid_ranges: dict[str, tuple[int, int]],
    train_end_bar: int,
    *,
    is_holdout: bool,
) -> PurgedFold | None:
    embargo = int(
        getattr(_cfg, "PURGED_WF_EMBARGO_CANDLES", _cfg.MAX_HOLD_CANDLES))
    min_valid = int(getattr(_cfg, "PURGED_WF_MIN_VALID_ROWS", 3000))

    train_parts: list[pd.DataFrame] = []
    valid_parts: list[pd.DataFrame] = []
    valid_start = min(v[0] for v in valid_ranges.values())
    valid_end = max(v[1] for v in valid_ranges.values())

    for symbol, group in df.groupby("symbol", sort=True, observed=False):
        sym = str(symbol)
        if sym not in valid_ranges:
            continue
        v_start, v_end = valid_ranges[sym]
        train_raw = _slice_symbol_rows(group, 0, train_end_bar)
        valid_slice = _slice_symbol_rows(group, v_start, v_end)
        if len(valid_slice) == 0:
            continue
        train_purged = _purge_train(train_raw, v_start, embargo)
        if train_purged.empty and not is_holdout:
            continue
        train_parts.append(train_purged)
        valid_parts.append(valid_slice)

    if not valid_parts:
        return None

    train_df = (
        pd.concat(train_parts, ignore_index=True)
        if train_parts
        else pd.DataFrame(columns=df.columns)
    )
    valid_df = pd.concat(valid_parts, ignore_index=True)

    if len(valid_df) < min_valid and not is_holdout:
        logger.warning(
            "Purged fold %d: valid rows %d < min %d; skipping",
            fold_id,
            len(valid_df),
            min_valid,
        )
        return None

    return PurgedFold(
        fold_id=fold_id,
        train_df=train_df,
        valid_df=valid_df,
        train_end_bar=int(train_end_bar),
        valid_start_bar=int(valid_start),
        valid_end_bar=int(valid_end),
        n_train_rows=int(len(train_df)),
        n_valid_rows=int(len(valid_df)),
        is_holdout=is_holdout,
    )


def build_purged_walk_forward_folds(
    df: pd.DataFrame,
    *,
    n_splits: int | None = None,
    holdout_fraction: float | None = None,
    embargo_candles: int | None = None,
    min_train_fraction: float | None = None,
    min_valid_rows: int | None = None,
) -> list[PurgedFold]:
    """
    Build purged expanding walk-forward folds on ``train_new.csv``.

    Returns ``n_splits`` CV folds plus one primary holdout fold (last).
    """
    if df.empty:
        return []

    n_splits = int(
        n_splits if n_splits is not None else getattr(
            _cfg, "PURGED_WF_N_SPLITS", 4)
    )
    holdout_fraction = float(
        holdout_fraction
        if holdout_fraction is not None
        else getattr(_cfg, "PURGED_WF_HOLDOUT_FRACTION", 0.25)
    )
    if embargo_candles is not None:
        _ = embargo_candles  # applied in _purge_train via config
    min_train_fraction = float(
        min_train_fraction
        if min_train_fraction is not None
        else getattr(_cfg, "PURGED_WF_MIN_TRAIN_FRACTION", 0.40)
    )
    if min_valid_rows is not None:
        _ = min_valid_rows

    n_cv_folds = max(0, n_splits)
    per_sym: dict[str, tuple[list[tuple[int, int]], tuple[int, int]]] = {}
    for symbol, group in df.groupby("symbol", sort=True, observed=False):
        per_sym[str(symbol)] = _split_symbol_segments(
            group,
            holdout_fraction=holdout_fraction,
            n_cv_folds=n_cv_folds,
            min_train_fraction=min_train_fraction,
        )

    folds: list[PurgedFold] = []
    fold_id = 0

    for cv_idx in range(n_cv_folds):
        valid_ranges: dict[str, tuple[int, int]] = {}
        train_end = -1
        for sym, (cv_ranges, holdout_range) in per_sym.items():
            if cv_idx >= len(cv_ranges):
                continue
            v_start, v_end = cv_ranges[cv_idx]
            valid_ranges[sym] = (v_start, v_end)
            train_end = max(train_end, v_start - 1)
        if not valid_ranges or train_end < 0:
            continue

        min_train_bars = 0
        for sym, group in df.groupby("symbol", sort=True, observed=False):
            n_sym = len(group)
            min_train_bars = max(min_train_bars, int(
                math.floor(n_sym * min_train_fraction)))

        if train_end + 1 < min_train_bars:
            logger.warning(
                "CV fold %d: train bars %d < min_train_fraction requirement %d; skip",
                cv_idx,
                train_end + 1,
                min_train_bars,
            )
            continue

        fold = _build_fold_from_ranges(
            df,
            fold_id,
            valid_ranges,
            train_end,
            is_holdout=False,
        )
        if fold is not None:
            folds.append(fold)
            fold_id += 1

    holdout_ranges: dict[str, tuple[int, int]] = {}
    holdout_train_end = -1
    for sym, (_, holdout_range) in per_sym.items():
        h_start, h_end = holdout_range
        if h_start <= h_end:
            holdout_ranges[sym] = (h_start, h_end)
            holdout_train_end = max(holdout_train_end, h_start - 1)

    if holdout_ranges and holdout_train_end >= 0:
        holdout_fold = _build_fold_from_ranges(
            df,
            fold_id,
            holdout_ranges,
            holdout_train_end,
            is_holdout=True,
        )
        if holdout_fold is not None:
            folds.append(holdout_fold)

    return folds


def derive_primary_holdout(
    folds: list[PurgedFold],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, val_df) from the primary holdout fold."""
    if not folds:
        return pd.DataFrame(), pd.DataFrame()

    holdout = next((f for f in reversed(folds) if f.is_holdout), folds[-1])
    train_df = holdout.train_df
    val_df = holdout.valid_df
    return train_df, val_df


def cv_folds_only(folds: list[PurgedFold]) -> list[PurgedFold]:
    """CV folds excluding the primary holdout."""
    return [f for f in folds if not f.is_holdout]


def summarize_fold_metrics(metrics: Iterable[dict]) -> FoldMetricsSummary:
    """Summarize metrics across folds (worst-case emphasis)."""
    items = list(metrics)
    if not items:
        return FoldMetricsSummary(
            folds=0,
            worst_return_pct=-100.0,
            worst_profit_factor=0.0,
            worst_sortino_ratio=0.0,
            worst_drawdown_pct=100.0,
            min_trades=0,
            mean_return_pct=-100.0,
            mean_profit_factor=0.0,
            metrics=[],
        )

    returns = [float(m.get("total_return_pct", 0.0)) for m in items]
    pfs = [float(m.get("profit_factor", 0.0)) for m in items]
    sortinos = [float(m.get("sortino_ratio", 0.0)) for m in items]
    dds = [float(m.get("max_drawdown_pct", 0.0)) for m in items]
    trades = [int(m.get("executed_trades", 0)) for m in items]

    return FoldMetricsSummary(
        folds=len(items),
        worst_return_pct=float(min(returns)),
        worst_profit_factor=float(min(pfs)),
        worst_sortino_ratio=float(min(sortinos)),
        worst_drawdown_pct=float(max(dds)),
        min_trades=int(min(trades)) if trades else 0,
        mean_return_pct=float(np.mean(returns)),
        mean_profit_factor=float(np.mean(pfs)),
        metrics=items,
    )


def aggregate_fold_metrics(
    fold_metrics: list[dict],
    mode: str | None = None,
) -> dict:
    """
    Collapse per-fold metrics into one dict for objectives / gates.

    ``worst``: min return/PF/trades, max drawdown (conservative).
    ``mean``: average return/PF, max drawdown, min trades.
    """
    if not fold_metrics:
        return {
            "total_return_pct": -100.0,
            "profit_factor": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown_pct": 100.0,
            "executed_trades": 0,
            "win_rate": 0.0,
        }

    mode = (
        str(mode or getattr(_cfg, "PURGED_WF_AGGREGATION", "worst")).strip().lower()
    )
    summary = summarize_fold_metrics(fold_metrics)

    if mode == "mean":
        return {
            "total_return_pct": summary.mean_return_pct,
            "profit_factor": summary.mean_profit_factor,
            "sortino_ratio": float(
                np.mean([float(m.get("sortino_ratio", 0.0))
                        for m in fold_metrics])
            ),
            "max_drawdown_pct": summary.worst_drawdown_pct,
            "executed_trades": summary.min_trades,
            "win_rate": float(
                np.mean([float(m.get("win_rate", 0.0)) for m in fold_metrics])
            ),
        }

    worst_return_fold = min(
        fold_metrics,
        key=lambda m: float(m.get("total_return_pct", -1e9)),
    )
    return {
        "total_return_pct": summary.worst_return_pct,
        "profit_factor": summary.worst_profit_factor,
        "sortino_ratio": summary.worst_sortino_ratio,
        "max_drawdown_pct": summary.worst_drawdown_pct,
        "executed_trades": summary.min_trades,
        "win_rate": float(worst_return_fold.get("win_rate", 0.0)),
    }


_SPLIT_CONFIG_KEYS = (
    # Purged walk-forward keys (deprecated mode)
    "PURGED_WF_N_SPLITS",
    "PURGED_WF_HOLDOUT_FRACTION",
    "PURGED_WF_EMBARGO_CANDLES",
    "PURGED_WF_MIN_TRAIN_FRACTION",
    "PURGED_WF_MIN_VALID_ROWS",
    # Holdout split keys (active mode)
    "HOLDOUT_TRAIN_FRACTION",
    "HOLDOUT_EMBARGO_CANDLES",
    "VALIDATION_HALF_PURGE_CANDLES",
)


def purged_config_fingerprint() -> str:
    """Return a short hash of all split-config knobs for cache invalidation."""
    import hashlib

    parts: list[str] = []
    for key in _SPLIT_CONFIG_KEYS:
        val = getattr(_cfg, key, None)
        parts.append(f"{key}={val!r}")
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def write_cv_folds_manifest(
    folds: list[PurgedFold] | None,
    *,
    reference_rows: int,
    path: str | None = None,
) -> str:
    """Persist split/fold metadata JSON; returns the path written."""
    fold_list = folds or []
    out_path = path or getattr(
        _cfg, "CV_FOLDS_MANIFEST_PATH", "data/cv_folds_manifest.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    payload: dict[str, Any] = {
        "split_mode": getattr(_cfg, "SPLIT_MODE", "holdout"),
        "config_fingerprint": purged_config_fingerprint(),
        "reference_rows": int(reference_rows),
        "n_folds": len(fold_list),
        "config": {
            "PURGED_WF_N_SPLITS": _cfg.PURGED_WF_N_SPLITS,
            "PURGED_WF_HOLDOUT_FRACTION": _cfg.PURGED_WF_HOLDOUT_FRACTION,
            "PURGED_WF_EMBARGO_CANDLES": _cfg.PURGED_WF_EMBARGO_CANDLES,
            "VALIDATION_HALF_PURGE_CANDLES": getattr(
                _cfg, "VALIDATION_HALF_PURGE_CANDLES", 0,
            ),
        },
        "folds": [
            {
                "fold_id": f.fold_id,
                "is_holdout": f.is_holdout,
                "n_train_rows": f.n_train_rows,
                "n_valid_rows": f.n_valid_rows,
                "train_end_bar": f.train_end_bar,
                "valid_start_bar": f.valid_start_bar,
                "valid_end_bar": f.valid_end_bar,
            }
            for f in fold_list
        ],
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return out_path


def load_cv_folds_manifest(path: str | None = None) -> dict[str, Any] | None:
    """Load manifest if present."""
    out_path = path or getattr(
        _cfg, "CV_FOLDS_MANIFEST_PATH", "data/cv_folds_manifest.json")
    if not os.path.exists(out_path):
        return None
    with open(out_path, encoding="utf-8") as fh:
        return json.load(fh)
