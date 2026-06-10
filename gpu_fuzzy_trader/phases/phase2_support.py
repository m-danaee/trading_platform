"""
phase2_support.py — Regime-aware trade support penalties for Phase 2.

Specialist rules concentrated in one market regime with sufficient per-regime
trades and quality (win rate or net PnL) bypass global support penalties and
the pool trade floor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from gpu_fuzzy_trader import config as _cfg

if TYPE_CHECKING:
    from gpu_fuzzy_trader.phases.phase2_stage import Phase2StageParams

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvolutionFloors:
    """Resolved evolution-time floors (pool admission gates remain strict)."""

    return_floor_pct: float
    min_trade_support: int
    use_robust_return_obj: bool
    soft_feasibility: bool
    pool_require_positive_splits: bool


def resolve_evolution_floors(
    stage_params: Phase2StageParams | None = None,
) -> EvolutionFloors:
    """Return stage-aware fitness floors; defaults to global strict knobs."""
    if stage_params is None:
        return EvolutionFloors(
            return_floor_pct=float(_cfg.PHASE2_RETURN_FLOOR_PCT),
            min_trade_support=int(_cfg.MIN_TRADE_SUPPORT),
            use_robust_return_obj=bool(_cfg.PHASE2_USE_ROBUST_RETURN_OBJ),
            soft_feasibility=False,
            pool_require_positive_splits=bool(
                _cfg.PHASE2_POOL_REQUIRE_POSITIVE_SPLITS
            ),
        )
    return EvolutionFloors(
        return_floor_pct=float(stage_params.return_floor_pct),
        min_trade_support=int(stage_params.min_trade_support),
        use_robust_return_obj=bool(stage_params.use_robust_return_obj),
        soft_feasibility=bool(stage_params.soft_feasibility),
        pool_require_positive_splits=bool(
            stage_params.pool_require_positive_splits
        ),
    )


def _compact_regime_labels(
    regime_ids: np.ndarray,
    n_regimes: int,
) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    """
    Drop regimes with zero row count and remap surviving labels to 0..k-1.

    When only one regime remains, regime support is disabled for this slice.
    """
    ids = np.asarray(regime_ids, dtype=np.int32)
    counts = np.bincount(ids.astype(np.int64), minlength=n_regimes)
    active = [i for i in range(n_regimes) if counts[i] > 0]
    if len(active) == n_regimes:
        return ids, regime_row_fractions(ids, n_regimes), n_regimes
    if len(active) <= 1:
        logger.warning(
            "Phase 2 regime compaction: %d → %d active regimes; "
            "disabling regime support for this slice",
            n_regimes,
            len(active),
        )
        return None, None, 0
    dropped = [i for i in range(n_regimes) if counts[i] == 0]
    remap = {old: new for new, old in enumerate(active)}
    compacted = np.array(
        [remap[int(r)] for r in ids],
        dtype=np.int32,
    )
    k = len(active)
    logger.warning(
        "Phase 2 regime compaction: %d → %d active regimes (dropped empty: %s)",
        n_regimes,
        k,
        dropped,
    )
    return compacted, regime_row_fractions(compacted, k), k


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
    *,
    min_trade_support: int | None = None,
) -> np.ndarray:
    """
    Minimum executed trades required in each regime for specialist status.

    Scales with both the regime's row share and the rule's total trade count so
    a 60-trade specialist in a 90%% row-fraction regime is not asked for 270 trades.
    """
    support_target = (
        int(min_trade_support)
        if min_trade_support is not None
        else int(_cfg.MIN_TRADE_SUPPORT)
    )
    frac = np.asarray(regime_row_fractions_arr, dtype=np.float64)
    scale = max(executed, 0) / max(support_target, 1)
    raw = support_target * frac * _cfg.PHASE2_REGIME_MIN_TRADE_FRACTION * scale
    thresholds = np.maximum(_cfg.MIN_TRADE_POOL_FLOOR,
                            np.round(raw)).astype(np.int64)
    return thresholds


def _is_regime_specialist(
    executed: int,
    regime_trade_counts: np.ndarray,
    regime_win_counts: np.ndarray,
    regime_net_pnl: np.ndarray,
    regime_row_fractions_arr: np.ndarray,
    *,
    min_trade_support: int | None = None,
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
        regime_row_fractions_arr,
        executed,
        min_trade_support=min_trade_support,
    )
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


def _static_support_penalty(
    executed: int,
    *,
    min_trade_support: int | None = None,
) -> float:
    """Legacy graduated penalty (no regime context)."""
    support_target = (
        int(min_trade_support)
        if min_trade_support is not None
        else int(_cfg.MIN_TRADE_SUPPORT)
    )
    if executed >= support_target:
        return 0.0
    if executed < _cfg.MIN_TRADE_POOL_FLOOR:
        return 2.0 * _cfg.SUPPORT_PENALTY_MAX
    shortfall = (support_target - executed) / max(support_target, 1)
    return min(shortfall ** 2 * _cfg.SUPPORT_PENALTY_MAX, _cfg.SUPPORT_PENALTY_MAX)


def trade_support_penalty(
    executed: int,
    *,
    min_trade_support: int | None = None,
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
    support_target = (
        int(min_trade_support)
        if min_trade_support is not None
        else int(_cfg.MIN_TRADE_SUPPORT)
    )
    if executed >= support_target:
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
                executed,
                counts,
                wins,
                pnl,
                fracs,
                min_trade_support=support_target,
            )
            if is_spec:
                return 0.0, True, dom

    pen = _static_support_penalty(executed, min_trade_support=support_target)
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

    # Regime Profitability Gate
    if _cfg.PHASE2_REGIME_PROFITABILITY_GATE:
        regime_pnl = train_metrics.get("regime_net_pnl")
        if regime_pnl is not None:
            if train_metrics.get("regime_specialist", False):
                dom = int(train_metrics.get("dominant_regime", -1))
                if dom >= 0 and dom < len(regime_pnl):
                    if regime_pnl[dom] <= _cfg.PHASE2_REGIME_MIN_RETURN_PER_REGIME:
                        return False
            else:
                n_passing = sum(1 for p in regime_pnl if p > _cfg.PHASE2_REGIME_MIN_RETURN_PER_REGIME)
                min_pass = 2 if len(regime_pnl) >= 2 else len(regime_pnl)
                if n_passing < min_pass:
                    return False

    # Pool admission always requires validation metrics when positive splits
    # are enforced — independent of PHASE2_JOINT_TRAIN_VAL (evolution-only flag).
    if val_metrics is None:
        return False

    if not cv_fold and _cfg.PHASE2_REQUIRE_LAST_FOLD_POSITIVE:
        val_ret = float(val_metrics.get("total_return_pct", 0.0))
        if val_ret <= 0.0:
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


def passes_pool_entry_admission(entry: dict) -> bool:
    """
    Post-merge filter for persisted Phase 2 pool JSON entries.

    In purged CV mode, entries built via ``evaluate_purged_cv_pool_admission``
    store ``cv_folds_passing`` / ``cv_folds_total``; use those instead of
    merged worst-case objectives (which can be negative while 2/3 folds pass).
    Legacy entries without CV metadata fall back to holdout gates on objectives.
    """
    if not _cfg.PHASE2_POOL_REQUIRE_POSITIVE_SPLITS:
        return True

    val_obj = entry.get("val_objectives")
    if _split_mode_is_purged_cv():
        cv_pass = entry.get("cv_folds_passing")
        if cv_pass is not None:
            total = int(entry.get("cv_folds_total", _cfg.CV_N_FOLDS))
            if not is_cv_deployable(int(cv_pass), total):
                return False
            # Fold majority is the hard gate; merged metrics are for ranking only.
            return val_obj is not None

    objectives = entry.get("objectives", {}) or {}
    train_metrics = {
        "total_return_pct": float(objectives.get("total_return_pct", 0.0)),
        "profit_factor": float(objectives.get("profit_factor", 1.0)),
        "executed_trades": int(entry.get("executed_trades", 0)),
        "regime_specialist": entry.get("regime_specialist", False),
        "dominant_regime": entry.get("dominant_regime", -1),
    }
    val_obj = entry.get("val_objectives")
    if val_obj is None:
        return False
    val_metrics = {
        "total_return_pct": float(val_obj.get("total_return_pct", 0.0)),
        "profit_factor": float(val_obj.get("profit_factor", 1.0)),
        "executed_trades": int(entry.get("val_executed_trades", 0)),
    }
    return passes_pool_admission_gate(train_metrics, val_metrics)


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
    min_trade_support: int | None = None,
) -> tuple[float, bool, int]:
    """
    Support penalty from train metrics, with optional val confirmation.

    Returns (penalty, is_regime_specialist, dominant_regime).
    """
    executed = int(metrics.get("executed_trades", 0))
    counts, wins, pnl = metrics_regime_arrays(metrics)
    penalty, is_spec, dom = trade_support_penalty(
        executed,
        min_trade_support=min_trade_support,
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


def cv_min_folds_to_pass(total_folds: int | None = None) -> int:
    """Minimum CV folds required for deployability."""
    total = int(total_folds if total_folds is not None else _cfg.CV_N_FOLDS)
    return min(int(_cfg.PHASE2_CV_POOL_MIN_FOLDS_PASS), max(total, 1))


def is_cv_deployable(folds_passing: int, total_folds: int | None = None) -> bool:
    """True when enough CV folds pass per-fold admission."""
    total = int(total_folds if total_folds is not None else _cfg.CV_N_FOLDS)
    return int(folds_passing) >= cv_min_folds_to_pass(total)


def robust_return_pct(
    train_metrics: dict,
    val_metrics: dict | None,
) -> float:
    """Conservative return used for objectives, plateau, and archive ranking."""
    train_ret = float(train_metrics.get("total_return_pct", 0.0))
    if val_metrics is None:
        return train_ret
    val_ret = float(val_metrics.get("total_return_pct", 0.0))
    return min(train_ret, val_ret)


def _raw_feasibility_violation_score(
    train_metrics: dict,
    val_metrics: dict | None,
    *,
    cv_fold: bool = False,
) -> float:
    """Compute violation score against pool admission floors (ignores stage soft mode)."""
    if not _cfg.PHASE2_POOL_REQUIRE_POSITIVE_SPLITS:
        return 0.0

    train_floor, train_ret_min, val_ret_min, pf_floor, min_val_trades = (
        _pool_admission_floors(cv_fold=cv_fold)
    )
    score = 0.0

    train_trades = int(train_metrics.get("executed_trades", 0))
    if train_trades < train_floor:
        score += float(train_floor - train_trades)

    train_ret = float(train_metrics.get("total_return_pct", 0.0))
    train_pf = float(train_metrics.get("profit_factor", 0.0))
    if train_ret <= train_ret_min:
        score += abs(train_ret_min - train_ret) + 1.0
    if train_pf < pf_floor:
        score += (pf_floor - train_pf) * 5.0

    if val_metrics is None:
        return score + 5.0

    val_trades = int(val_metrics.get("executed_trades", 0))
    if val_trades < min_val_trades:
        score += float(min_val_trades - val_trades)

    val_ret = float(val_metrics.get("total_return_pct", 0.0))
    val_pf = float(val_metrics.get("profit_factor", 0.0))
    if val_ret <= val_ret_min:
        score += abs(val_ret_min - val_ret) + 1.0
    if val_pf < pf_floor:
        score += (pf_floor - val_pf) * 5.0

    return score


def feasibility_violation_score(
    train_metrics: dict,
    val_metrics: dict | None,
    *,
    cv_fold: bool = False,
    stage_params: Phase2StageParams | None = None,
) -> float:
    """
    Non-negative violation score; 0 means the rule meets deployability floors.

    Used to penalize infeasible chromosomes during evolution.
    When Stage A soft feasibility is active, returns 0 (penalties applied elsewhere).
    """
    floors = resolve_evolution_floors(stage_params)
    if floors.soft_feasibility:
        return 0.0
    if not floors.pool_require_positive_splits:
        return 0.0
    return _raw_feasibility_violation_score(
        train_metrics, val_metrics, cv_fold=cv_fold,
    )


def passes_evolution_deployability_preview(
    train_metrics: dict,
    val_metrics: dict | None,
    *,
    regime_row_fractions_arr: np.ndarray | None = None,
) -> bool:
    """
    Lightweight deployability check used during evolution (no per-fold CV).

    Mirrors holdout admission floors on merged train/val metrics from the
    evolution engines.
    """
    if not passes_pool_trade_floor(
        int(train_metrics.get("executed_trades", 0)),
        train_metrics,
        regime_row_fractions_arr=regime_row_fractions_arr,
    ):
        return False
    return feasibility_violation_score(
        train_metrics, val_metrics, cv_fold=_split_mode_is_purged_cv(),
    ) <= 0.0


def deployability_rank_score(
    train_metrics: dict,
    val_metrics: dict | None,
    *,
    folds_passing: int = 0,
) -> float:
    """
    Higher is better. Used to rank deployable archive entries and Stage B seeds.
    """
    robust = robust_return_pct(train_metrics, val_metrics)
    fold_bonus = float(folds_passing) * 0.5
    sortino = float(
        train_metrics.get("sortino_ratio", train_metrics.get(
            "total_return_pct", 0.0))
    )
    if val_metrics is not None:
        val_sortino = float(
            val_metrics.get(
                "sortino_ratio", val_metrics.get("total_return_pct", 0.0),
            )
        )
        sortino = min(sortino, val_sortino)
    dd = float(train_metrics.get("max_drawdown_pct", 100.0))
    if val_metrics is not None:
        dd = max(dd, float(val_metrics.get("max_drawdown_pct", dd)))
    return robust + fold_bonus + 0.1 * sortino - 0.05 * dd


def compute_robust_score(
    train_metrics: dict,
    val_metrics: dict | None,
    *,
    source_symbols: list[str] | None = None,
) -> float:
    """Archive robustness score used for shared-archive promotion."""
    base = deployability_rank_score(train_metrics, val_metrics)
    if source_symbols:
        base += 0.25 * len(set(source_symbols))
    return float(base)
