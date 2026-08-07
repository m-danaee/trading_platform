"""Guards that keep Phase 2 searchable under sparse context coverage."""

from __future__ import annotations

from gpu_fuzzy_trader import config as cfg
from gpu_fuzzy_trader.evolution.evox_runner import (
    _should_abort_zero_deployable_collapse,
)
from gpu_fuzzy_trader.phases.phase2_stage import resolve_phase2_stage_params


def test_singleton_island_floors_are_capped_for_sparse_context() -> None:
    hp = cfg.resolve_island_hyperparams(
        "cluster", n_rows=29600, reference_rows=59200, n_symbols=1,
    )
    assert hp.min_trade_support <= cfg.PHASE2_SPARSE_ISLAND_MAX_TRADE_SUPPORT
    assert hp.min_trade_pool_floor <= cfg.PHASE2_ISLAND_TRADE_FLOOR_ABSOLUTE_MIN
    assert hp.val_trade_floor <= cfg.PHASE2_SPARSE_ISLAND_MAX_VAL_TRADE_FLOOR


def test_stage_a_support_is_reachable_under_sparse_caps() -> None:
    stage_a = resolve_phase2_stage_params("A")
    assert stage_a.min_trade_support <= cfg.PHASE2_SPARSE_ISLAND_MAX_TRADE_SUPPORT
    assert stage_a.soft_feasibility is True


def test_abort_zero_deployable_after_exhausted_restarts() -> None:
    assert _should_abort_zero_deployable_collapse(
        deployable_count=0,
        restart_count=1,
        max_restarts=1,
        viability_collapse_streak=3,
        island_profile="cluster",
        stage_params=resolve_phase2_stage_params("A"),
    )
    assert not _should_abort_zero_deployable_collapse(
        deployable_count=1,
        restart_count=1,
        max_restarts=1,
        viability_collapse_streak=3,
        island_profile="cluster",
        stage_params=resolve_phase2_stage_params("A"),
    )
    assert not _should_abort_zero_deployable_collapse(
        deployable_count=0,
        restart_count=0,
        max_restarts=1,
        viability_collapse_streak=3,
        island_profile="cluster",
        stage_params=resolve_phase2_stage_params("A"),
    )
