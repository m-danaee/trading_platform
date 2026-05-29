"""
phase2_support.py — Regime-aware trade support penalties for Phase 2.

Specialist rules concentrated in one market regime with sufficient per-regime
trades and quality (win rate or net PnL) bypass global support penalties and
the pool trade floor.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from gpu_fuzzy_trader import config as _cfg

logger = logging.getLogger(__name__)


def to_host_numpy(
    x: Any,
    *,
    dtype: np.dtype | type | None = None,
) -> np.ndarray | None:
    """Coerce JAX/DeviceArray or sequences to a host NumPy array."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        arr = x
    else:
        arr = np.asarray(x)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return np.asarray(arr)


def regime_row_fractions(regime_ids: np.ndarray, n_regimes: int) -> np.ndarray:
    """Fraction of rows per regime (length n_regimes)."""
    counts = np.bincount(
        regime_ids.astype(np.int64, copy=False),
        minlength=n_regimes,
    ).astype(np.float64)
    total = float(counts.sum())
    if total <= 0:
        return np.ones(n_regimes, dtype=np.float64) / max(n_regimes, 1)
    return counts / total


def per_regime_trade_thresholds(
    regime_row_fractions_arr: np.ndarray,
    executed: int,
) -> np.ndarray:
    """
    Minimum executed trades required in each regime for specialist status.

    Scales with both the regime's row share and the rule's total trade count so
    a 60-trade specialist in a 90%% row-fraction regime is not asked for 270 trades.
    """
    frac = np.asarray(regime_row_fractions_arr, dtype=np.float64)
    scale = max(executed, 0) / max(_cfg.MIN_TRADE_SUPPORT, 1)
    raw = _cfg.MIN_TRADE_SUPPORT * frac * \
        _cfg.PHASE2_REGIME_MIN_TRADE_FRACTION * scale
    thresholds = np.maximum(_cfg.MIN_TRADE_POOL_FLOOR,
                            np.round(raw)).astype(np.int64)
    return thresholds


def _is_regime_specialist(
    executed: int,
    regime_trade_counts: np.ndarray,
    regime_win_counts: np.ndarray,
    regime_net_pnl: np.ndarray,
    regime_row_fractions_arr: np.ndarray,
) -> tuple[bool, int]:
    """Return (is_specialist, dominant_regime_index)."""
    counts = np.asarray(regime_trade_counts, dtype=np.int64)
    if executed <= 0 or counts.size == 0:
        return False, -1

    d = int(np.argmax(counts))
    trades_d = int(counts[d])
    if trades_d <= 0:
        return False, d

    concentration = trades_d / executed
    if concentration < _cfg.PHASE2_REGIME_CONCENTRATION_MIN:
        return False, d

    thresholds = per_regime_trade_thresholds(
        regime_row_fractions_arr, executed)
    if trades_d < thresholds[d]:
        return False, d

    wins = np.asarray(regime_win_counts, dtype=np.int64)
    pnl = np.asarray(regime_net_pnl, dtype=np.float64)
    win_rate_ok = (
        trades_d > 0
        and float(wins[d]) / float(trades_d) >= _cfg.PHASE2_REGIME_MIN_WIN_RATE
    )
    pnl_ok = _cfg.PHASE2_REGIME_USE_PNL_GATE and float(pnl[d]) > 0.0
    if not (win_rate_ok or pnl_ok):
        return False, d

    return True, d


def _static_support_penalty(executed: int) -> float:
    """Legacy graduated penalty (no regime context)."""
    if executed >= _cfg.MIN_TRADE_SUPPORT:
        return 0.0
    if executed < _cfg.MIN_TRADE_POOL_FLOOR:
        return 2.0 * _cfg.SUPPORT_PENALTY_MAX
    shortfall = (_cfg.MIN_TRADE_SUPPORT - executed) / _cfg.MIN_TRADE_SUPPORT
    return min(shortfall ** 2 * _cfg.SUPPORT_PENALTY_MAX, _cfg.SUPPORT_PENALTY_MAX)


def trade_support_penalty(
    executed: int,
    *,
    regime_trade_counts: np.ndarray | None = None,
    regime_win_counts: np.ndarray | None = None,
    regime_net_pnl: np.ndarray | None = None,
    regime_row_fractions_arr: np.ndarray | None = None,
) -> tuple[float, bool, int]:
    """
    Regime-aware support penalty.

    Returns
    -------
    penalty : float
    is_regime_specialist : bool
    dominant_regime : int (-1 if N/A)
    """
    if executed >= _cfg.MIN_TRADE_SUPPORT:
        return 0.0, False, -1

    use_regime = (
        _cfg.PHASE2_REGIME_SUPPORT_ENABLED
        and regime_trade_counts is not None
        and regime_row_fractions_arr is not None
        and regime_win_counts is not None
        and regime_net_pnl is not None
    )

    if use_regime:
        counts = to_host_numpy(regime_trade_counts, dtype=np.int64)
        wins = to_host_numpy(regime_win_counts, dtype=np.int64)
        pnl = to_host_numpy(regime_net_pnl, dtype=np.float64)
        fracs = to_host_numpy(regime_row_fractions_arr, dtype=np.float64)
        if counts is not None and fracs is not None and wins is not None and pnl is not None:
            is_spec, dom = _is_regime_specialist(
                executed, counts, wins, pnl, fracs,
            )
            if is_spec:
                return 0.0, True, dom

    pen = _static_support_penalty(executed)
    return pen, False, -1


def _split_mode_is_purged_cv() -> bool:
    return str(_cfg.SPLIT_MODE).strip().lower() == "purged_rolling_cv"


def _pool_admission_floors(
    *,
    cv_fold: bool,
) -> tuple[int, float, float, float, int]:
    """Return (train_trade_floor, train_ret_min, val_ret_min, pf_floor, min_val_trades)."""
    if cv_fold and _split_mode_is_purged_cv():
        return (
            int(_cfg.PHASE2_CV_MIN_TRADE_POOL_FLOOR),
            float(_cfg.PHASE2_CV_POOL_TRAIN_RETURN_MIN_PCT),
            float(_cfg.PHASE2_CV_POOL_VAL_RETURN_MIN_PCT),
            float(_cfg.PHASE2_CV_PROFIT_FACTOR_FLOOR),
            int(_cfg.PHASE2_CV_MIN_VAL_TRADES),
        )
    return (
        int(_cfg.MIN_TRADE_POOL_FLOOR),
        float(_cfg.PHASE2_POOL_TRAIN_RETURN_MIN_PCT),
        float(_cfg.PHASE2_POOL_VAL_RETURN_MIN_PCT),
        float(_cfg.PHASE2_PROFIT_FACTOR_FLOOR),
        max(int(_cfg.MIN_TRADE_POOL_FLOOR) // 4, 10),
    )


def _passes_pool_admission_impl(
    train_metrics: dict,
    val_metrics: dict | None,
    *,
    cv_fold: bool,
) -> bool:
    if not _cfg.PHASE2_POOL_REQUIRE_POSITIVE_SPLITS:
        return True

    train_floor, train_ret_min, val_ret_min, pf_floor, min_val_trades = (
        _pool_admission_floors(cv_fold=cv_fold)
    )

    train_trades = int(train_metrics.get("executed_trades", 0))
    if train_trades < train_floor:
        return False

    train_ret = float(train_metrics.get("total_return_pct", 0.0))
    train_pf = float(train_metrics.get("profit_factor", 0.0))
    if train_ret <= train_ret_min:
        return False
    if train_pf < pf_floor:
        return False

    if not _cfg.PHASE2_JOINT_TRAIN_VAL:
        return True
    if val_metrics is None:
        return False

    val_trades = int(val_metrics.get("executed_trades", 0))
    if val_trades < min_val_trades:
        return False

    val_ret = float(val_metrics.get("total_return_pct", 0.0))
    val_pf = float(val_metrics.get("profit_factor", 0.0))
    if val_ret <= val_ret_min:
        return False
    if val_pf < pf_floor:
        return False

    if _cfg.PHASE2_REGIME_REQUIRE_VAL_CONFIRMATION and train_metrics.get(
        "regime_specialist"
    ):
        dom = int(train_metrics.get("dominant_regime", -1))
        if not val_regime_confirmation(dom, val_metrics):
            return False

    return True


def passes_pool_admission_gate(
    train_metrics: dict,
    val_metrics: dict | None = None,
) -> bool:
    """
    Hard gate for Phase 2 pool/archive on merged holdout metrics.

    When ``PHASE2_POOL_REQUIRE_POSITIVE_SPLITS`` is False, always returns True.
    """
    return _passes_pool_admission_impl(train_metrics, val_metrics, cv_fold=False)


def passes_pool_admission_cv_fold(
    train_metrics: dict,
    val_metrics: dict | None,
) -> bool:
    """Per-fold admission gate when ``SPLIT_MODE`` is ``purged_rolling_cv``."""
    return _passes_pool_admission_impl(train_metrics, val_metrics, cv_fold=True)


def passes_pool_trade_floor(
    executed: int,
    metrics: dict,
    *,
    regime_row_fractions_arr: np.ndarray | None = None,
) -> bool:
    """Pool/archive inclusion gate (may waive floor for regime specialists)."""
    trade_floor = int(_cfg.MIN_TRADE_POOL_FLOOR)
    if _split_mode_is_purged_cv():
        trade_floor = int(_cfg.PHASE2_CV_MIN_TRADE_POOL_FLOOR)
    if executed >= trade_floor:
        return True

    if not _cfg.PHASE2_REGIME_SUPPORT_ENABLED:
        return False

    counts = metrics.get("regime_trade_counts")
    if counts is None:
        return False

    _, is_spec, _ = trade_support_penalty(
        executed,
        regime_trade_counts=to_host_numpy(counts, dtype=np.int64),
        regime_win_counts=to_host_numpy(
            metrics.get("regime_win_counts"), dtype=np.int64,
        ),
        regime_net_pnl=to_host_numpy(
            metrics.get("regime_net_pnl"), dtype=np.float64,
        ),
        regime_row_fractions_arr=regime_row_fractions_arr,
    )
    return is_spec


def val_regime_confirmation(
    train_dominant_regime: int,
    val_metrics: dict,
    *,
    val_regime_row_counts: np.ndarray | None = None,
) -> bool:
    """
    Optional validation confirmation for train specialists.

  When PHASE2_REGIME_REQUIRE_VAL_CONFIRMATION is False, always True.
  Missing val regime rows/trades → inconclusive pass (True).
    """
    if not _cfg.PHASE2_REGIME_REQUIRE_VAL_CONFIRMATION:
        return True
    if train_dominant_regime < 0:
        return True

    min_val_trades = max(_cfg.MIN_TRADE_POOL_FLOOR // 4, 10)

    if val_regime_row_counts is not None:
        row_counts = to_host_numpy(val_regime_row_counts, dtype=np.int64)
        if row_counts is not None and train_dominant_regime < len(row_counts):
            if int(row_counts[train_dominant_regime]) <= 0:
                logger.debug(
                    "val_regime_absent: no validation rows in train-dominant regime %d",
                    train_dominant_regime,
                )
                return True

    val_counts = to_host_numpy(
        val_metrics.get("regime_trade_counts"), dtype=np.int64,
    )
    if val_counts is None or train_dominant_regime >= len(val_counts):
        return True

    val_trades_d = int(val_counts[train_dominant_regime])
    val_executed = int(val_metrics.get("executed_trades", 0))
    if val_trades_d == 0 or val_executed == 0:
        logger.debug(
            "val_regime_absent: zero val trades in regime %d",
            train_dominant_regime,
        )
        return True

    if val_trades_d < min_val_trades:
        return True

    concentration = val_trades_d / val_executed
    if concentration < _cfg.PHASE2_REGIME_CONCENTRATION_MIN:
        return False

    wins = to_host_numpy(val_metrics.get("regime_win_counts"), dtype=np.int64)
    pnl = to_host_numpy(val_metrics.get("regime_net_pnl"), dtype=np.float64)
    if wins is None or pnl is None:
        return True

    win_rate_ok = (
        val_trades_d > 0
        and float(wins[train_dominant_regime]) / float(val_trades_d)
        >= _cfg.PHASE2_REGIME_MIN_WIN_RATE
    )
    pnl_ok = (
        _cfg.PHASE2_REGIME_USE_PNL_GATE
        and float(pnl[train_dominant_regime]) > 0.0
    )
    return win_rate_ok or pnl_ok


def metrics_regime_arrays(
    metrics: dict,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Extract host NumPy regime arrays from an engine metrics dict."""
    return (
        to_host_numpy(metrics.get("regime_trade_counts"), dtype=np.int64),
        to_host_numpy(metrics.get("regime_win_counts"), dtype=np.int64),
        to_host_numpy(metrics.get("regime_net_pnl"), dtype=np.float64),
    )


def compute_support_penalty_and_specialist(
    metrics: dict,
    regime_row_fractions_arr: np.ndarray | None,
    *,
    val_metrics: dict | None = None,
    val_regime_row_counts: np.ndarray | None = None,
) -> tuple[float, bool, int]:
    """
    Support penalty from train metrics, with optional val confirmation.

    Returns (penalty, is_regime_specialist, dominant_regime).
    """
    executed = int(metrics.get("executed_trades", 0))
    counts, wins, pnl = metrics_regime_arrays(metrics)
    penalty, is_spec, dom = trade_support_penalty(
        executed,
        regime_trade_counts=counts,
        regime_win_counts=wins,
        regime_net_pnl=pnl,
        regime_row_fractions_arr=regime_row_fractions_arr,
    )

    if is_spec and val_metrics is not None:
        if not val_regime_confirmation(
            dom,
            val_metrics,
            val_regime_row_counts=val_regime_row_counts,
        ):
            penalty = max(penalty, _cfg.SUPPORT_PENALTY_MAX)
            is_spec = False

    return penalty, is_spec, dom
