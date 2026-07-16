
from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.backtest.cpu_engine import CPUBacktestEngine
from gpu_fuzzy_trader.backtest.df_slim import slim_backtest_df

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PurgedFold:
    """One expanding-window validation fold."""

    fold_id: int
    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    train_end_bar: int | None
    valid_start_bar: int | None
    valid_end_bar: int | None


@dataclass(frozen=True)
class FoldMetricsSummary:
    """Aggregated worst-fold metrics for one rule or rule set."""

    folds: int
    worst_return_pct: float
    worst_profit_factor: float
    worst_sortino_ratio: float
    worst_drawdown_pct: float
    min_trades: int
    mean_return_pct: float
    mean_profit_factor: float
    metrics: list[dict]


def _prepare_ordered_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a stable chronological copy with per-symbol bar indices."""
    if df.empty:
        return df.copy()

    out = df.copy()
    if "datetime" in out.columns:
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
        if "symbol" in out.columns:
            out = out.sort_values(["symbol", "datetime"], kind="mergesort")
        else:
            out = out.sort_values(["datetime"], kind="mergesort")
    elif "symbol" in out.columns and "_symbol_bar_index" in out.columns:
        out = out.sort_values(["symbol", "_symbol_bar_index"], kind="mergesort")
    elif "symbol" in out.columns:
        out = out.sort_values(["symbol"], kind="mergesort")
    out = out.reset_index(drop=True)

    if "symbol" in out.columns:
        out["_symbol_bar_index"] = out.groupby("symbol", sort=False).cumcount()
    else:
        out["_symbol_bar_index"] = np.arange(len(out), dtype=np.int64)
    return out


def build_purged_rolling_folds(
    df: pd.DataFrame,
    n_splits: int | None = None,
    embargo_candles: int | None = None,
    min_train_frac: float | None = None,
    min_valid_rows: int | None = None,
) -> list[PurgedFold]:
    """Build expanding rolling folds with per-symbol embargo.

    Validation blocks are defined by per-symbol bar index ranges.  For each
    fold, train rows are strictly before ``valid_start - embargo`` and validation
    rows lie in the next block.  This prevents future rows and label-overlap rows
    from entering the training side of the same fold.
    """
    if df is None or df.empty:
        return []

    n_splits = int(n_splits if n_splits is not None else _cfg.PURGED_CV_N_SPLITS)
    embargo_candles = int(
        embargo_candles if embargo_candles is not None else _cfg.PURGED_CV_EMBARGO_CANDLES
    )
    min_train_frac = float(
        min_train_frac if min_train_frac is not None else _cfg.PURGED_CV_MIN_TRAIN_FRACTION
    )
    min_valid_rows = int(
        min_valid_rows if min_valid_rows is not None else _cfg.PURGED_CV_MIN_VALID_ROWS
    )

    if n_splits <= 0:
        return []

    ordered = _prepare_ordered_df(df)
    if "_symbol_bar_index" not in ordered.columns:
        return []

    max_bar = int(ordered["_symbol_bar_index"].max())
    if max_bar <= 0:
        return []

    first_valid_bar = max(1, int(round(max_bar * min_train_frac)))
    remaining = max_bar - first_valid_bar + 1
    if remaining <= n_splits:
        return []

    edges = np.linspace(first_valid_bar, max_bar + 1, n_splits + 1).round().astype(int)
    folds: list[PurgedFold] = []

    for fold_id in range(n_splits):
        valid_start = int(edges[fold_id])
        valid_end = int(edges[fold_id + 1])
        if valid_end <= valid_start:
            continue

        train_end = valid_start - embargo_candles
        if train_end <= 0:
            continue

        train_mask = ordered["_symbol_bar_index"].to_numpy() < train_end
        valid_bars = ordered["_symbol_bar_index"].to_numpy()
        valid_mask = (valid_bars >= valid_start) & (valid_bars < valid_end)

        train_part = ordered.loc[train_mask].reset_index(drop=True)
        valid_part = ordered.loc[valid_mask].reset_index(drop=True)
        if len(train_part) == 0 or len(valid_part) < min_valid_rows:
            continue

        folds.append(
            PurgedFold(
                fold_id=len(folds),
                train_df=train_part,
                valid_df=valid_part,
                train_end_bar=int(train_end),
                valid_start_bar=valid_start,
                valid_end_bar=valid_end - 1,
            )
        )

    logger.info(
        "Built %d purged rolling folds (n_splits=%d, embargo=%d, min_train_frac=%.2f)",
        len(folds), n_splits, embargo_candles, min_train_frac,
    )
    return folds


def build_fold_engines(
    df: pd.DataFrame,
    direction: str,
    feature_names: list[str] | None = None,
    n_splits: int | None = None,
    embargo_candles: int | None = None,
) -> list[CPUBacktestEngine]:
    """Build validation engines for purged rolling folds."""
    folds = build_purged_rolling_folds(
        df,
        n_splits=n_splits,
        embargo_candles=embargo_candles,
    )
    engines: list[CPUBacktestEngine] = []
    for fold in folds:
        part = fold.valid_df
        if feature_names:
            try:
                part = slim_backtest_df(part, feature_names)
            except Exception:
                pass
        engine = CPUBacktestEngine(part, {}, direction)
        setattr(engine, "_fold_id", fold.fold_id)
        setattr(engine, "_valid_start_bar", fold.valid_start_bar)
        setattr(engine, "_valid_end_bar", fold.valid_end_bar)
        engines.append(engine)
    return engines


def summarize_fold_metrics(metrics: list[dict]) -> FoldMetricsSummary:
    """Aggregate a list of fold metrics into worst/mean values."""
    if not metrics:
        return FoldMetricsSummary(
            folds=0,
            worst_return_pct=-100.0,
            worst_profit_factor=0.0,
            worst_sortino_ratio=-10.0,
            worst_drawdown_pct=100.0,
            min_trades=0,
            mean_return_pct=-100.0,
            mean_profit_factor=0.0,
            metrics=[],
        )

    returns = np.asarray([float(m.get("total_return_pct", 0.0)) for m in metrics], dtype=float)
    pfs = np.asarray([float(m.get("profit_factor", 0.0)) for m in metrics], dtype=float)
    sortinos = np.asarray([float(m.get("sortino_ratio", m.get("total_return_pct", 0.0))) for m in metrics], dtype=float)
    dds = np.asarray([float(m.get("max_drawdown_pct", 0.0)) for m in metrics], dtype=float)
    trades = np.asarray([int(m.get("executed_trades", 0)) for m in metrics], dtype=int)
    return FoldMetricsSummary(
        folds=len(metrics),
        worst_return_pct=float(np.min(returns)),
        worst_profit_factor=float(np.min(pfs)),
        worst_sortino_ratio=float(np.min(sortinos)),
        worst_drawdown_pct=float(np.max(dds)),
        min_trades=int(np.min(trades)),
        mean_return_pct=float(np.mean(returns)),
        mean_profit_factor=float(np.mean(pfs)),
        metrics=metrics,
    )


def evaluate_rule_set_on_fold_engines(
    rule_set: list[dict],
    fold_engines: list[CPUBacktestEngine],
) -> FoldMetricsSummary:
    """Evaluate one rule set on validation engines and return worst-fold summary."""
    metrics: list[dict] = []
    for engine in fold_engines:
        try:
            m = engine.simulate_rule_set(rule_set)
        except Exception as exc:
            logger.debug("fold rule-set simulation failed: %s", exc)
            m = {
                "total_return_pct": -100.0,
                "profit_factor": 0.0,
                "sortino_ratio": -10.0,
                "max_drawdown_pct": 100.0,
                "executed_trades": 0,
            }
        m = dict(m)
        m["fold_id"] = int(getattr(engine, "_fold_id", -1))
        metrics.append(m)
    return summarize_fold_metrics(metrics)
