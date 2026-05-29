"""
Phase 2 — purged CV fold engine construction and conservative metric aggregation.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.data.cv_folds import PurgedFold
from gpu_fuzzy_trader.phases.phase2_rule_pool import (
    Rule_Pool_Generator,
    _sample_df,
)

logger = logging.getLogger(__name__)


def _merge_metrics_worst_case(
    current: dict | None,
    new: dict,
) -> dict:
    """Conservative merge: min return/Sortino/PF, max drawdown, min win rate."""
    if current is None:
        return dict(new)

    out = dict(current)
    ret = float(new.get("total_return_pct", 0.0))
    out["total_return_pct"] = min(
        float(out.get("total_return_pct", 0.0)), ret)

    sortino = float(new.get(
        "sortino_ratio", new.get("total_return_pct", 0.0)))
    out["sortino_ratio"] = min(
        float(out.get("sortino_ratio", out.get("total_return_pct", 0.0))),
        sortino,
    )

    dd = float(new.get("max_drawdown_pct", 0.0))
    out["max_drawdown_pct"] = max(
        float(out.get("max_drawdown_pct", 0.0)), dd)

    wr = float(new.get("win_rate", 0.0))
    out["win_rate"] = min(float(out.get("win_rate", 0.0)), wr)

    pf = float(new.get("profit_factor", 0.0))
    out["profit_factor"] = min(float(out.get("profit_factor", pf)), pf)

    trades = int(new.get("executed_trades", 0))
    out["executed_trades"] = min(
        int(out.get("executed_trades", trades)), trades)
    return out


class _FoldBacktestEngine:
    """Delegates simulate_rule_batch to an inner engine (GPU or CPU)."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        for attr in ("_regime_row_fractions", "_regime_row_counts", "backend"):
            if hasattr(inner, attr):
                setattr(self, attr, getattr(inner, attr))

    def simulate_rule_batch(self, **kwargs: Any) -> list[dict]:
        return self._inner.simulate_rule_batch(**kwargs)


class PurgedCVTrainEngine:
    """
    Train-side facade: batch metrics are worst-case across CV train folds.

    Used as ``engine`` in Phase 2 evolution when CV is enabled.
    """

    def __init__(self, fold_engines: list[Any]) -> None:
        self._fold_engines = fold_engines
        if fold_engines:
            fr = getattr(fold_engines[0], "_regime_row_fractions", None)
            self._regime_row_fractions = fr

    def simulate_rule_batch(self, **kwargs: Any) -> list[dict]:
        if not self._fold_engines:
            raise RuntimeError("PurgedCVTrainEngine has no fold engines")

        merged: list[dict | None] | None = None
        for eng in self._fold_engines:
            batch = eng.simulate_rule_batch(**kwargs)
            if merged is None:
                merged = [None] * len(batch)
            for j, met in enumerate(batch):
                merged[j] = _merge_metrics_worst_case(merged[j], met)
        return [m if m is not None else {} for m in merged]


class PurgedCVValEngine:
    """Val-side facade: worst-case metrics across CV validation folds."""

    def __init__(self, fold_engines: list[Any]) -> None:
        self._fold_engines = fold_engines
        if fold_engines:
            rc = getattr(fold_engines[0], "_regime_row_counts", None)
            self._regime_row_counts = rc

    def simulate_rule_batch(self, **kwargs: Any) -> list[dict]:
        if not self._fold_engines:
            raise RuntimeError("PurgedCVValEngine has no fold engines")

        merged: list[dict | None] | None = None
        for eng in self._fold_engines:
            batch = eng.simulate_rule_batch(**kwargs)
            if merged is None:
                merged = [None] * len(batch)
            for j, met in enumerate(batch):
                merged[j] = _merge_metrics_worst_case(merged[j], met)
        return [m if m is not None else {} for m in merged]


def build_cv_fold_engines(
    folds: list[PurgedFold],
    feature_infos: list[dict],
    direction: str,
    *,
    seed: int | None,
    builder: Rule_Pool_Generator,
) -> tuple[PurgedCVTrainEngine | None, PurgedCVValEngine | None]:
    """
    Build purged CV train/val engine facades for Phase 2.

    Returns (None, None) when *folds* is empty.
    """
    if not folds:
        return None, None

    feature_names = [fi["name"] for fi in feature_infos]
    from gpu_fuzzy_trader.backtest.df_slim import slim_backtest_df
    from gpu_fuzzy_trader.phases.phase2_rule_pool import _prepare_regime_context

    train_engines: list[Any] = []
    val_engines: list[Any] = []

    sample_seed = seed if seed is not None else _cfg.PHASE2_SEED

    for fold in folds:
        train_sampled = _sample_df(
            fold.train_df,
            _cfg.PHASE1_SAMPLING_TOTAL,
            random_state=sample_seed,
        )
        val_sampled = _sample_df(
            fold.val_df,
            _cfg.PHASE1_SAMPLING_TOTAL,
            random_state=sample_seed,
        )
        train_regime_ids, _, n_regimes = _prepare_regime_context(train_sampled)
        val_regime_ids, _, val_n_regimes = _prepare_regime_context(val_sampled)

        slim_train = slim_backtest_df(train_sampled, feature_names)
        slim_val = slim_backtest_df(val_sampled, feature_names)

        train_eng = builder._build_engine_for_df(
            slim_train,
            regime_ids=train_regime_ids,
            n_regimes=n_regimes,
        )
        val_eng = builder._build_engine_for_df(
            slim_val,
            regime_ids=val_regime_ids,
            n_regimes=val_n_regimes,
        )
        if val_regime_ids is not None:
            val_eng._regime_row_counts = np.bincount(
                val_regime_ids.astype(np.int64),
                minlength=val_n_regimes,
            ).astype(np.int64)

        train_engines.append(_FoldBacktestEngine(train_eng))
        val_engines.append(_FoldBacktestEngine(val_eng))

    logger.info(
        "Phase 2 [%s]: purged CV engines built (%d folds, embargo=%d bars)",
        direction,
        len(folds),
        _cfg.CV_EMBARGO_BARS,
    )
    return PurgedCVTrainEngine(train_engines), PurgedCVValEngine(val_engines)
