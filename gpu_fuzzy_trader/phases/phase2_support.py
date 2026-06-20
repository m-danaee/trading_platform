"""
phase2_support.py — Trade support penalties for Phase 2.
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
    *,
    n_rows: int | None = None,
    island_hyperparams: _cfg.IslandHyperparams | None = None,
) -> EvolutionFloors:
    """Return stage-aware fitness floors; defaults to global strict knobs."""
    if island_hyperparams is not None:
        return EvolutionFloors(
            return_floor_pct=float(_cfg.PHASE2_RETURN_FLOOR_PCT),
            min_trade_support=int(island_hyperparams.min_trade_support),
            use_robust_return_obj=bool(_cfg.PHASE2_USE_ROBUST_RETURN_OBJ),
            soft_feasibility=False,
            pool_require_positive_splits=bool(
                _cfg.PHASE2_POOL_REQUIRE_POSITIVE_SPLITS
            ),
        )
    if stage_params is None:
        return EvolutionFloors(
            return_floor_pct=float(_cfg.PHASE2_RETURN_FLOOR_PCT),
            min_trade_support=int(_cfg.effective_min_trade_support(n_rows)),
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


def _static_support_penalty(
    executed: int,
    *,
    min_trade_support: int | None = None,
) -> float:
    """Legacy graduated penalty."""
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
) -> tuple[float, bool, int]:
    """
    Support penalty.

    Returns
    -------
    penalty : float
    is_specialist : bool (always False)
    dominant_label : int (always -1)
    """
    support_target = (
        int(min_trade_support)
        if min_trade_support is not None
        else int(_cfg.MIN_TRADE_SUPPORT)
    )
    if executed >= support_target:
        return 0.0, False, -1

    pen = _static_support_penalty(executed, min_trade_support=support_target)
    return pen, False, -1


def _pool_admission_floors(
    n_valid_rows: int | None = None,
) -> tuple[int, float, float, float, int]:
    """Return (train_trade_floor, train_ret_min, val_ret_min, pf_floor, min_val_trades)."""
    min_val = _cfg.effective_pool_min_val_trades(n_valid_rows)
    train_floor = _cfg.effective_min_trade_pool_floor(n_valid_rows)
    return (
        int(train_floor),
        float(_cfg.PHASE2_POOL_TRAIN_RETURN_MIN_PCT),
        float(_cfg.PHASE2_POOL_VAL_RETURN_MIN_PCT),
        float(_cfg.PHASE2_PROFIT_FACTOR_FLOOR),
        int(min_val),
    )


def _passes_pool_admission_impl(
    train_metrics: dict,
    val_metrics: dict | None,
    *,
    n_valid_rows: int | None = None,
) -> bool:
    if not _cfg.PHASE2_POOL_REQUIRE_POSITIVE_SPLITS:
        return True

    train_floor, train_ret_min, val_ret_min, pf_floor, min_val_trades = (
        _pool_admission_floors(n_valid_rows)
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

    # Pool admission always requires validation metrics when positive splits
    # are enforced — independent of PHASE2_JOINT_TRAIN_VAL (evolution-only flag).
    if val_metrics is None:
        return False

    # Holdout-path gate: in single-fold (holdout) mode, optionally require the
    # validation fold to have positive total return. Originally introduced for
    # the multi-fold CV era (named "last fold positive") but the check itself
    # runs in the holdout path; it remains a useful deployability filter even
    # without purged rolling CV.
    if _cfg.PHASE2_REQUIRE_LAST_FOLD_POSITIVE:
        if float(val_metrics.get("total_return_pct", 0.0)) <= 0.0:
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

    max_gap = float(getattr(_cfg, "PHASE2_MAX_TRAIN_VAL_GAP_PCT", 20.0))
    if train_ret - val_ret > max_gap:
        return False

    # Optional stricter positive-good gate (default OFF; enabled in Task 5).
    if bool(getattr(_cfg, "PHASE2_STRICT_POSITIVE_GOOD", False)):
        # Lazy import to avoid circular dependency (phase3_rule_set is a sibling).
        from gpu_fuzzy_trader.phases.phase3_rule_set import (
            gate_positive_good as _gate_positive_good,
        )
        if not _gate_positive_good(
            train_metrics,
            val_metrics,
            min_train_return=float(
                getattr(_cfg, "PHASE3_MIN_TRAIN_RETURN", 0.0)),
            min_val_return=float(
                getattr(_cfg, "PHASE3_MIN_VAL_RETURN", 0.0)),
            min_train_pf=float(
                getattr(_cfg, "PHASE3_MIN_TRAIN_PF", 1.0)),
            min_val_pf=float(
                getattr(_cfg, "PHASE3_MIN_VAL_PF", 1.0)),
            min_train_trades=int(
                getattr(_cfg, "PHASE3_MIN_TRAIN_TRADES", 25)),
            min_val_trades=int(
                getattr(_cfg, "PHASE3_MIN_VAL_TRADES", 15)),
        ):
            return False

    return True


def passes_pool_admission_gate(
    train_metrics: dict,
    val_metrics: dict | None = None,
    *,
    n_valid_rows: int | None = None,
) -> bool:
    """
    Hard gate for Phase 2 pool/archive on merged holdout metrics.

    When ``PHASE2_POOL_REQUIRE_POSITIVE_SPLITS`` is False, always returns True.
    """
    return _passes_pool_admission_impl(
        train_metrics, val_metrics, n_valid_rows=n_valid_rows,
    )


def passes_pool_entry_admission(entry: dict) -> bool:
    """
    Post-merge filter for persisted Phase 2 pool JSON entries.
    """
    if not _cfg.PHASE2_POOL_REQUIRE_POSITIVE_SPLITS:
        return True

    objectives = entry.get("objectives", {}) or {}
    train_metrics = {
        "total_return_pct": float(objectives.get("total_return_pct", 0.0)),
        "profit_factor": float(objectives.get("profit_factor", 1.0)),
        "executed_trades": int(entry.get("executed_trades", 0)),
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


def passes_pool_trade_floor(
    executed: int,
    metrics: dict,
    *,
    n_rows: int | None = None,
    island_hyperparams: _cfg.IslandHyperparams | None = None,
) -> bool:
    """Pool/archive inclusion gate."""
    if island_hyperparams is not None:
        trade_floor = int(island_hyperparams.min_trade_pool_floor)
    else:
        trade_floor = _cfg.effective_min_trade_pool_floor(n_rows)
    if executed >= trade_floor:
        return True
    return False


def compute_support_penalty_and_specialist(
    metrics: dict,
    *,
    min_trade_support: int | None = None,
) -> tuple[float, bool, int]:
    """
    Support penalty from train metrics.

    Returns (penalty, False, -1).
    """
    executed = int(metrics.get("executed_trades", 0))
    penalty, _, _ = trade_support_penalty(
        executed,
        min_trade_support=min_trade_support,
    )
    return penalty, False, -1


def _joint_primary_metric(
    train_val: float,
    val_val: float | None,
    *,
    joint: bool,
) -> float:
    """Train-only or conservative min(train, val) for ranking / objectives."""
    if not joint or val_val is None:
        return float(train_val)
    return float(min(train_val, val_val))


def robust_return_pct(
    train_metrics: dict,
    val_metrics: dict | None,
    *,
    joint: bool | None = None,
) -> float:
    """Conservative return used for objectives, plateau, and archive ranking."""
    train_ret = float(train_metrics.get("total_return_pct", 0.0))
    if val_metrics is None:
        return train_ret
    val_ret = float(val_metrics.get("total_return_pct", 0.0))
    if joint is None:
        joint = bool(_cfg.PHASE2_JOINT_TRAIN_VAL)
    return _joint_primary_metric(train_ret, val_ret, joint=joint)


def robust_win_rate_pct(
    train_metrics: dict,
    val_metrics: dict | None,
    *,
    joint: bool | None = None,
) -> float:
    """Conservative win rate for ranking when f3 uses win rate."""
    train_wr = float(train_metrics.get("win_rate", 0.0))
    if val_metrics is None:
        return train_wr
    val_wr = float(val_metrics.get("win_rate", 0.0))
    if joint is None:
        joint = bool(_cfg.PHASE2_JOINT_TRAIN_VAL)
    return _joint_primary_metric(train_wr, val_wr, joint=joint)


def _raw_feasibility_violation_score(
    train_metrics: dict,
    val_metrics: dict | None,
    *,
    n_valid_rows: int | None = None,
) -> float:
    """Compute violation score against pool admission floors (ignores stage soft mode)."""
    if not _cfg.PHASE2_POOL_REQUIRE_POSITIVE_SPLITS:
        return 0.0

    train_floor, train_ret_min, val_ret_min, pf_floor, min_val_trades = (
        _pool_admission_floors(n_valid_rows)
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
    stage_params: Phase2StageParams | None = None,
    n_valid_rows: int | None = None,
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
        train_metrics, val_metrics, n_valid_rows=n_valid_rows,
    )


def passes_evolution_deployability_preview(
    train_metrics: dict,
    val_metrics: dict | None,
    *,
    island_hyperparams: _cfg.IslandHyperparams | None = None,
) -> bool:
    """
    Lightweight deployability check used during evolution (no per-fold CV).

    Mirrors holdout admission floors on merged train/val metrics from the
    evolution engines.
    """
    if not passes_pool_trade_floor(
        int(train_metrics.get("executed_trades", 0)),
        train_metrics,
        island_hyperparams=island_hyperparams,
    ):
        return False
    return feasibility_violation_score(
        train_metrics, val_metrics,
    ) <= 0.0


def deployability_rank_score(
    train_metrics: dict,
    val_metrics: dict | None,
) -> float:
    """
    Higher is better. Used to rank deployable archive entries and Stage B seeds.

    Primary term mirrors f3: win rate when ``PHASE2_USE_TOTAL_RETURN_OBJ`` is
    False, else robust return. Sortino / drawdown respect ``PHASE2_JOINT_TRAIN_VAL``.
    """
    joint = bool(_cfg.PHASE2_JOINT_TRAIN_VAL)

    train_sortino = float(
        train_metrics.get(
            "sortino_ratio", train_metrics.get("total_return_pct", 0.0),
        )
    )
    val_sortino = None
    if val_metrics is not None:
        val_sortino = float(
            val_metrics.get(
                "sortino_ratio", val_metrics.get("total_return_pct", 0.0),
            )
        )
    sortino = _joint_primary_metric(train_sortino, val_sortino, joint=joint)

    train_dd = float(train_metrics.get("max_drawdown_pct", 100.0))
    val_dd = None
    if val_metrics is not None:
        val_dd = float(val_metrics.get("max_drawdown_pct", train_dd))
    if joint and val_dd is not None:
        dd = max(train_dd, val_dd)
    else:
        dd = train_dd

    if bool(_cfg.PHASE2_USE_TOTAL_RETURN_OBJ):
        use_robust = bool(_cfg.PHASE2_USE_ROBUST_RETURN_OBJ)
        primary = robust_return_pct(
            train_metrics, val_metrics, joint=joint and use_robust,
        )
    else:
        primary = robust_win_rate_pct(train_metrics, val_metrics, joint=joint)

    return primary + 0.1 * sortino - 0.05 * dd


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
