"""
phase2_support.py — Trade support penalties for Phase 2.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.scoring.gates import (
    PositiveGoodThresholds,
    gate_positive_good,
)

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
    """Return stage-aware fitness floors; defaults to global strict knobs.

    When both *stage_params* and *island_hyperparams* are set (cluster
    two-stage), Stage A keeps soft exploration floors; Stage B uses
    island-scaled support with strict feasibility.
    """
    if stage_params is not None:
        min_support = int(stage_params.min_trade_support)
        if island_hyperparams is not None:
            island_support = int(island_hyperparams.min_trade_support)
            if bool(stage_params.soft_feasibility) or stage_params.stage == "A":
                # Soft Stage A: never stricter than the island-scaled target.
                min_support = min(min_support, island_support)
            else:
                # Stage B refinement: island-scaled support floors.
                min_support = island_support
        return EvolutionFloors(
            return_floor_pct=float(stage_params.return_floor_pct),
            min_trade_support=min_support,
            use_robust_return_obj=bool(stage_params.use_robust_return_obj),
            soft_feasibility=bool(stage_params.soft_feasibility),
            pool_require_positive_splits=bool(
                stage_params.pool_require_positive_splits
            ),
        )
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
    return EvolutionFloors(
        return_floor_pct=float(_cfg.PHASE2_RETURN_FLOOR_PCT),
        min_trade_support=int(_cfg.effective_min_trade_support(n_rows)),
        use_robust_return_obj=bool(_cfg.PHASE2_USE_ROBUST_RETURN_OBJ),
        soft_feasibility=False,
        pool_require_positive_splits=bool(
            _cfg.PHASE2_POOL_REQUIRE_POSITIVE_SPLITS
        ),
    )


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
    return min(shortfall ** float(_cfg.TRADE_SUPPORT_PENALTY_EXPONENT) * _cfg.SUPPORT_PENALTY_MAX, _cfg.SUPPORT_PENALTY_MAX)


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
    island_hyperparams: _cfg.IslandHyperparams | None = None,
) -> tuple[int, float, float, float, int]:
    """Return (train_trade_floor, train_ret_min, val_ret_min, pf_floor, min_val_trades).

    Uses ADMISSION PF 1.15 — this is the hard gate at pool entry.
    """
    min_val = (
        int(island_hyperparams.val_trade_floor)
        if island_hyperparams is not None
        else _cfg.effective_pool_min_val_trades(n_valid_rows)
    )
    train_floor = (
        int(island_hyperparams.min_trade_pool_floor)
        if island_hyperparams is not None
        else _cfg.effective_min_trade_pool_floor(n_valid_rows)
    )
    return (
        int(train_floor),
        float(_cfg.PHASE2_POOL_TRAIN_RETURN_MIN_PCT),
        float(_cfg.PHASE2_POOL_VAL_RETURN_MIN_PCT),
        float(_cfg.PHASE2_PROFIT_FACTOR_FLOOR_ADMISSION),  # hard gate: 1.15
        int(min_val),
    )


def _evolution_feasibility_floors(
    n_valid_rows: int | None = None,
    island_hyperparams: _cfg.IslandHyperparams | None = None,
) -> tuple[int, float, float, float, int]:
    """Return (train_trade_floor, train_ret_min, val_ret_min, pf_floor, min_val_trades).

    Uses EVOLUTION PF 1.0 — softer penalty threshold during NSGA-III fitness
    so the feasible set isn't artificially collapsed when val trade counts are thin.
    Pool admission (hard gate) still uses ADMISSION PF 1.15.
    """
    min_val = (
        int(island_hyperparams.val_trade_floor)
        if island_hyperparams is not None
        else _cfg.effective_pool_min_val_trades(n_valid_rows)
    )
    train_floor = (
        int(island_hyperparams.min_trade_pool_floor)
        if island_hyperparams is not None
        else _cfg.effective_min_trade_pool_floor(n_valid_rows)
    )
    return (
        int(train_floor),
        float(_cfg.PHASE2_POOL_TRAIN_RETURN_MIN_PCT),
        float(_cfg.PHASE2_POOL_VAL_RETURN_MIN_PCT),
        float(_cfg.PHASE2_PROFIT_FACTOR_FLOOR_EVOLUTION),  # soft penalty: 1.05
        int(min_val),
    )


def _passes_pool_admission_impl(
    train_metrics: dict,
    val_metrics: dict | None,
    *,
    n_valid_rows: int | None = None,
    island_hyperparams: _cfg.IslandHyperparams | None = None,
) -> bool:
    if not _cfg.PHASE2_POOL_REQUIRE_POSITIVE_SPLITS:
        return True

    train_floor, train_ret_min, val_ret_min, pf_floor, min_val_trades = (
        _pool_admission_floors(n_valid_rows, island_hyperparams)
    )

    # Pool admission always requires validation metrics when positive splits
    # are enforced — independent of PHASE2_JOINT_TRAIN_VAL (evolution-only flag).
    if val_metrics is None:
        return False

    # This is the single Phase 2 positive-good gate.  The floors come from the
    # active island/data slice, so a small cluster is not silently evaluated
    # against unrelated downstream RB thresholds.
    if not gate_positive_good(
        train_metrics,
        val_metrics,
        PositiveGoodThresholds(
            min_train_return=train_ret_min,
            min_valid_return=val_ret_min,
            min_train_profit_factor=pf_floor,
            min_valid_profit_factor=pf_floor,
            min_train_trades=train_floor,
            min_valid_trades=min_val_trades,
        ),
    ):
        return False

    # Holdout-path gate: in single-fold (holdout) mode, optionally require the
    # validation fold to have positive total return. Originally introduced for
    # the multi-fold CV era (named "last fold positive") but the check itself
    # runs in the holdout path; it remains a useful deployability filter even
    # without purged rolling CV.
    if _cfg.PHASE2_REQUIRE_LAST_FOLD_POSITIVE:
        if float(val_metrics.get("total_return_pct", 0.0)) <= 0.0:
            return False

    train_ret = float(train_metrics.get("total_return_pct", 0.0))
    val_ret = float(val_metrics.get("total_return_pct", 0.0))

    max_gap = float(getattr(_cfg, "PHASE2_MAX_TRAIN_VAL_GAP_PCT", 20.0))
    if train_ret - val_ret > max_gap:
        return False

    # f4 return-concentration gate (Task 2): reject if f4 > floor
    if getattr(_cfg, "PHASE2_F4_ENABLED", False):
        max_tr = float(train_metrics.get("max_single_trade_pnl", 0.0))
        sum_pos = float(train_metrics.get("sum_positive_trade_pnl", 0.0))
        eps = float(getattr(_cfg, "PHASE2_F4_EPSILON", 1e-6))
        f4_val = max_tr / max(sum_pos, eps) if sum_pos > 0 else 0.0
        # Concentration is a loss metric, so joint evidence uses the worse
        # split.  A single outlier-driven split must not pass the hard gate.
        if (val_metrics is not None
                and bool(getattr(_cfg, "PHASE2_JOINT_TRAIN_VAL", False))):
            max_tr_v = float(val_metrics.get("max_single_trade_pnl", 0.0))
            sum_pos_v = float(val_metrics.get("sum_positive_trade_pnl", 0.0))
            f4_val_v = max_tr_v / max(sum_pos_v, eps) if sum_pos_v > 0 else 0.0
            f4_val = max(f4_val, f4_val_v)
        f4_floor = float(getattr(_cfg, "PHASE2_F4_CONCENTRATION_FLOOR", 0.5))
        if f4_val > f4_floor:
            return False

    # Overfit ratio gate (Task 6): hard-reject when train_return / val_return
    # exceeds PHASE2_OVERFIT_RATIO_FLOOR. Complements the absolute-pp gate above:
    # catches high-ratio but low-absolute-gap cases (e.g., train=15%/val=4%).
    # → fixes audit finding #7 (absolute-pp gate missed high-ratio cases)
    max_ratio = float(getattr(_cfg, "PHASE2_OVERFIT_RATIO_FLOOR", 3.0))
    if max_ratio > 0.0:
        val_ret_safe = max(val_ret, 0.1)  # avoid div-by-near-zero
        if train_ret / val_ret_safe > max_ratio:
            return False

    return True


def _feasibility_gate_failures(
    train_metrics: dict,
    val_metrics: dict | None,
    *,
    n_valid_rows: int | None = None,
    island_hyperparams: _cfg.IslandHyperparams | None = None,
) -> dict[str, int]:
    """Return per-gate failure flags for evolution-time feasibility diagnostics.

    Uses EVOLUTION PF 1.05 (softer threshold) so the collapse breakdown
    reflects evolution feasibility, not the hard admission gate.

    The gates mirror ``_passes_pool_admission_impl`` exactly except the PF
    threshold comes from ``_evolution_feasibility_floors`` (1.05) instead of
    ``_pool_admission_floors`` (1.15):

    - train_trade_floor    : train_trades < train_floor
    - train_return_floor   : train_ret <= train_ret_min
    - train_pf_floor       : train_pf < pf_floor
    - val_required         : val_metrics is None (cannot evaluate val gates)
    - val_ret_positive     : PHASE2_REQUIRE_LAST_FOLD_POSITIVE and val_ret <= 0
    - val_trade_floor      : val_trades < min_val_trades
    - val_return_floor     : val_ret <= val_ret_min
    - val_pf_floor         : val_pf < pf_floor
    - train_val_gap        : train_ret - val_ret > max_gap
    - f4_concentration     : PHASE2_F4_ENABLED and f4 > PHASE2_F4_CONCENTRATION_FLOOR
    - overfit_ratio        : PHASE2_OVERFIT_RATIO_FLOOR > 0 and train_ret / max(val_ret, 0.1) > floor

    Args:
        train_metrics: Train dict with executed_trades, total_return_pct,
            profit_factor, etc.
        val_metrics: Validation dict (same shape) or None.
        n_valid_rows: Optional row count for island-floor scaling.

    Returns:
        Dict mapping gate name to 0 (passed) or 1 (failed).
    """
    train_floor, train_ret_min, val_ret_min, pf_floor, min_val_trades = (
        _evolution_feasibility_floors(n_valid_rows, island_hyperparams)
    )
    failures: dict[str, int] = {
        "train_trade_floor": 0,
        "train_return_floor": 0,
        "train_pf_floor": 0,
        "val_required": 0,
        "val_ret_positive": 0,
        "val_trade_floor": 0,
        "val_return_floor": 0,
        "val_pf_floor": 0,
        "train_val_gap": 0,
        "f4_concentration": 0,
        "overfit_ratio": 0,
    }

    train_trades = int(train_metrics.get("executed_trades", 0))
    if train_trades < train_floor:
        failures["train_trade_floor"] = 1

    train_ret = float(train_metrics.get("total_return_pct", 0.0))
    if train_ret <= train_ret_min:
        failures["train_return_floor"] = 1

    train_pf = float(train_metrics.get("profit_factor", 0.0))
    if train_pf < pf_floor:
        failures["train_pf_floor"] = 1

    if val_metrics is None:
        failures["val_required"] = 1
        return failures

    if _cfg.PHASE2_REQUIRE_LAST_FOLD_POSITIVE and float(val_metrics.get("total_return_pct", 0.0)) <= 0.0:
        failures["val_ret_positive"] = 1

    val_trades = int(val_metrics.get("executed_trades", 0))
    if val_trades < min_val_trades:
        failures["val_trade_floor"] = 1

    val_ret = float(val_metrics.get("total_return_pct", 0.0))
    if val_ret <= val_ret_min:
        failures["val_return_floor"] = 1

    val_pf = float(val_metrics.get("profit_factor", 0.0))
    if val_pf < pf_floor:
        failures["val_pf_floor"] = 1

    max_gap = float(getattr(_cfg, "PHASE2_MAX_TRAIN_VAL_GAP_PCT", 20.0))
    if train_ret - val_ret > max_gap:
        failures["train_val_gap"] = 1

    # f4 return-concentration gate (Task 2): reject if f4 > floor
    if getattr(_cfg, "PHASE2_F4_ENABLED", False):
        max_tr = float(train_metrics.get("max_single_trade_pnl", 0.0))
        sum_pos = float(train_metrics.get("sum_positive_trade_pnl", 0.0))
        eps = float(getattr(_cfg, "PHASE2_F4_EPSILON", 1e-6))
        f4_val = max_tr / max(sum_pos, eps) if sum_pos > 0 else 0.0
        # Concentration is a loss metric, so joint evidence uses the worse
        # split.  A single outlier-driven split must not pass the hard gate.
        if (val_metrics is not None
                and bool(getattr(_cfg, "PHASE2_JOINT_TRAIN_VAL", False))):
            max_tr_v = float(val_metrics.get("max_single_trade_pnl", 0.0))
            sum_pos_v = float(val_metrics.get("sum_positive_trade_pnl", 0.0))
            f4_val_v = max_tr_v / max(sum_pos_v, eps) if sum_pos_v > 0 else 0.0
            f4_val = max(f4_val, f4_val_v)
        f4_floor = float(getattr(_cfg, "PHASE2_F4_CONCENTRATION_FLOOR", 0.5))
        if f4_val > f4_floor:
            failures["f4_concentration"] = 1

    # Overfit ratio gate (Task 6): hard-reject when train_return / val_return
    # exceeds PHASE2_OVERFIT_RATIO_FLOOR.
    # → fixes audit finding #7 (absolute-pp gate missed high-ratio cases)
    max_ratio = float(getattr(_cfg, "PHASE2_OVERFIT_RATIO_FLOOR", 3.0))
    if max_ratio > 0.0:
        val_ret_safe = max(val_ret, 0.1)  # avoid div-by-near-zero
        if train_ret / val_ret_safe > max_ratio:
            failures["overfit_ratio"] = 1

    return failures


def passes_pool_admission_gate(
    train_metrics: dict,
    val_metrics: dict | None = None,
    *,
    n_valid_rows: int | None = None,
    island_hyperparams: _cfg.IslandHyperparams | None = None,
) -> bool:
    """
    Hard gate for Phase 2 pool/archive on merged holdout metrics.

    When ``PHASE2_POOL_REQUIRE_POSITIVE_SPLITS`` is False, always returns True.
    """
    return _passes_pool_admission_impl(
        train_metrics,
        val_metrics,
        n_valid_rows=n_valid_rows,
        island_hyperparams=island_hyperparams,
    )


def passes_pool_entry_admission(
    entry: dict,
    *,
    island_hyperparams: _cfg.IslandHyperparams | None = None,
) -> bool:
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
    return passes_pool_admission_gate(
        train_metrics,
        val_metrics,
        island_hyperparams=island_hyperparams,
    )


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


def expectancy_lcb_pct(
    metrics: dict | None,
    *,
    z_score: float | None = None,
) -> float:
    """Return a conservative lower bound for per-trade net expectancy.

    Exact CPU reports carry trade-level dispersion. GPU reports use the
    aggregate fallback, which is intentionally conservative and is replaced
    by CPU values during final admission.
    """
    values = metrics or {}
    try:
        direct = float(values.get("expectancy_lcb_pct_per_trade"))
    except (TypeError, ValueError):
        direct = float("nan")
    if np.isfinite(direct):
        return direct
    trades = max(1, int(values.get("executed_trades", 0) or 0))
    expectancy = float(values.get("total_return_pct", 0.0) or 0.0) / trades
    try:
        std = float(values.get("trade_return_std_pct", 0.0) or 0.0)
    except (TypeError, ValueError):
        std = 0.0
    z = float(
        z_score
        if z_score is not None
        else getattr(_cfg, "PHASE2_EXPECTANCY_LCB_Z", 1.645)
    )
    return float(expectancy - z * abs(std) / np.sqrt(trades))


def _val_terms_in_fitness() -> bool:
    """True when val-derived feasibility penalties belong in NSGA-III fitness."""
    return bool(_cfg.PHASE2_JOINT_TRAIN_VAL) or bool(
        getattr(_cfg, "PHASE2_VAL_IN_FITNESS_PENALTY", False),
    )


def _raw_feasibility_violation_score(
    train_metrics: dict,
    val_metrics: dict | None,
    *,
    n_valid_rows: int | None = None,
    include_val: bool = True,
    island_hyperparams: _cfg.IslandHyperparams | None = None,
) -> float:
    """Compute violation score using evolution PF floors (1.05) during NSGA-III fitness.

    Pool admission (hard gate) still uses ADMISSION PF 1.15 via
    ``_pool_admission_floors`` / ``_passes_pool_admission_impl``.

    When ``include_val`` is False (train-only fitness with
    ``PHASE2_VAL_IN_FITNESS_PENALTY=False``), only train-side floors contribute.
    Deployability preview passes ``include_val=True`` to keep full val checks.
    """
    if not _cfg.PHASE2_POOL_REQUIRE_POSITIVE_SPLITS:
        return 0.0

    train_floor, train_ret_min, val_ret_min, pf_floor, min_val_trades = (
        _evolution_feasibility_floors(n_valid_rows, island_hyperparams)
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

    if not include_val:
        return score

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

    # train-vs-val gap check: mirror the final pool-admission gate (line ~179).
    # A large train>>val gap is a classic overfit signal; penalise it here so
    # it flows into both deployability preview and the real objectives.
    gap = train_ret - val_ret
    max_gap = float(getattr(_cfg, "PHASE2_MAX_TRAIN_VAL_GAP_PCT", 16.0))
    if gap > max_gap:
        score += (gap - max_gap) * 1.0

    return score


def feasibility_violation_score(
    train_metrics: dict,
    val_metrics: dict | None,
    *,
    stage_params: Phase2StageParams | None = None,
    n_valid_rows: int | None = None,
    island_hyperparams: _cfg.IslandHyperparams | None = None,
) -> float:
    """
    Non-negative violation score; 0 means the rule meets deployability floors.

    Used to penalize infeasible chromosomes during evolution.
    When Stage A soft feasibility is active, returns 0 (penalties applied elsewhere).
    """
    floors = resolve_evolution_floors(
        stage_params,
        n_rows=n_valid_rows,
        island_hyperparams=island_hyperparams,
    )
    if floors.soft_feasibility:
        return 0.0
    if not floors.pool_require_positive_splits:
        return 0.0
    return _raw_feasibility_violation_score(
        train_metrics,
        val_metrics,
        n_valid_rows=n_valid_rows,
        island_hyperparams=island_hyperparams,
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
        train_metrics,
        val_metrics,
        island_hyperparams=island_hyperparams,
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

    if bool(getattr(_cfg, "PHASE2_RANK_USE_LCB_EXPECTANCY", True)):
        lcb_train = expectancy_lcb_pct(train_metrics)
        lcb_val = expectancy_lcb_pct(val_metrics) if val_metrics is not None else lcb_train
        # Prefer the weaker split while retaining a small PF tie-breaker.
        primary = min(lcb_train, lcb_val) + 0.25 * min(
            float(train_metrics.get("profit_factor", 0.0)),
            float((val_metrics or {}).get("profit_factor", 0.0)),
        )
    elif bool(_cfg.PHASE2_USE_TOTAL_RETURN_OBJ):
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
