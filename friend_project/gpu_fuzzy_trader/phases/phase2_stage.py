"""Phase 2 two-stage search hyperparameter profiles (exploration vs refinement)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from gpu_fuzzy_trader import config as _cfg

StageLabel = Literal["A", "B"] | None


@dataclass(frozen=True)
class Phase2StageParams:
    """Resolved evolution knobs for a Phase 2 stage (or single-stage fallback)."""

    stage: StageLabel
    mutation_rate: float
    mutation_weighted_activate_prob: float
    diversity_penalty: float
    diversity_hamming_threshold: int
    diversity_recovery_min_unique_ratio: float
    diversity_recovery_inject_fraction: float
    diversity_recovery_mutation_boost: float
    plateau_early_stop_patience: int
    plateau_early_stop_min_generation: int
    early_stop_min_generation: int
    seed_fraction: float
    return_floor_pct: float
    min_trade_support: int
    use_robust_return_obj: bool
    soft_feasibility: bool
    pool_require_positive_splits: bool


def resolve_phase2_stage_params(stage: StageLabel) -> Phase2StageParams:
    """
    Return stage-tuned hyperparameters.

    When *stage* is None (single-stage Phase 2), global ``PHASE2_*`` defaults apply.
    """
    if stage == "A":
        return Phase2StageParams(
            stage="A",
            mutation_rate=float(_cfg.PHASE2_STAGE_A_MUTATION_RATE),
            mutation_weighted_activate_prob=float(
                _cfg.PHASE2_STAGE_A_MUTATION_WEIGHTED_ACTIVATE_PROB
            ),
            diversity_penalty=float(_cfg.PHASE2_STAGE_A_DIVERSITY_PENALTY),
            diversity_hamming_threshold=int(
                _cfg.PHASE2_STAGE_A_DIVERSITY_HAMMING_THRESHOLD
            ),
            diversity_recovery_min_unique_ratio=float(
                _cfg.PHASE2_STAGE_A_DIVERSITY_RECOVERY_MIN_UNIQUE_RATIO
            ),
            diversity_recovery_inject_fraction=float(
                _cfg.PHASE2_STAGE_A_DIVERSITY_RECOVERY_INJECT_FRACTION
            ),
            diversity_recovery_mutation_boost=float(
                _cfg.PHASE2_STAGE_A_DIVERSITY_RECOVERY_MUTATION_BOOST
            ),
            plateau_early_stop_patience=int(
                _cfg.PHASE2_STAGE_A_PLATEAU_EARLY_STOP_PATIENCE
            ),
            plateau_early_stop_min_generation=int(
                _cfg.PHASE2_STAGE_A_PLATEAU_EARLY_STOP_MIN_GENERATION
            ),
            early_stop_min_generation=int(
                _cfg.PHASE2_STAGE_A_EARLY_STOP_MIN_GENERATION
            ),
            seed_fraction=float(_cfg.PHASE2_STAGE_A_ARCHIVE_SEED_FRACTION),
            return_floor_pct=float(_cfg.PHASE2_STAGE_A_RETURN_FLOOR_PCT),
            min_trade_support=int(_cfg.PHASE2_STAGE_A_MIN_TRADE_SUPPORT),
            use_robust_return_obj=bool(
                _cfg.PHASE2_STAGE_A_USE_ROBUST_RETURN_OBJ),
            soft_feasibility=bool(_cfg.PHASE2_STAGE_A_SOFT_FEASIBILITY),
            pool_require_positive_splits=False,
        )
    if stage == "B":
        return Phase2StageParams(
            stage="B",
            mutation_rate=float(_cfg.PHASE2_STAGE_B_MUTATION_RATE),
            mutation_weighted_activate_prob=float(
                _cfg.PHASE2_STAGE_B_MUTATION_WEIGHTED_ACTIVATE_PROB
            ),
            diversity_penalty=float(_cfg.PHASE2_STAGE_B_DIVERSITY_PENALTY),
            diversity_hamming_threshold=int(
                _cfg.PHASE2_STAGE_B_DIVERSITY_HAMMING_THRESHOLD
            ),
            diversity_recovery_min_unique_ratio=float(
                _cfg.PHASE2_STAGE_B_DIVERSITY_RECOVERY_MIN_UNIQUE_RATIO
            ),
            diversity_recovery_inject_fraction=float(
                _cfg.PHASE2_STAGE_B_DIVERSITY_RECOVERY_INJECT_FRACTION
            ),
            diversity_recovery_mutation_boost=float(
                _cfg.PHASE2_STAGE_B_DIVERSITY_RECOVERY_MUTATION_BOOST
            ),
            plateau_early_stop_patience=int(
                _cfg.PHASE2_STAGE_B_PLATEAU_EARLY_STOP_PATIENCE
            ),
            plateau_early_stop_min_generation=int(
                _cfg.PHASE2_STAGE_B_PLATEAU_EARLY_STOP_MIN_GENERATION
            ),
            early_stop_min_generation=int(
                _cfg.PHASE2_STAGE_B_EARLY_STOP_MIN_GENERATION
            ),
            seed_fraction=float(_cfg.PHASE2_STAGE_B_SEED_FRACTION),
            return_floor_pct=float(_cfg.PHASE2_RETURN_FLOOR_PCT),
            min_trade_support=int(_cfg.MIN_TRADE_SUPPORT),
            use_robust_return_obj=bool(_cfg.PHASE2_USE_ROBUST_RETURN_OBJ),
            soft_feasibility=False,
            pool_require_positive_splits=bool(
                _cfg.PHASE2_POOL_REQUIRE_POSITIVE_SPLITS
            ),
        )
    return Phase2StageParams(
        stage=None,
        mutation_rate=float(_cfg.PHASE2_MUTATION_RATE),
        mutation_weighted_activate_prob=float(
            _cfg.PHASE2_MUTATION_WEIGHTED_ACTIVATE_PROB
        ),
        diversity_penalty=float(_cfg.PHASE2_DIVERSITY_PENALTY),
        diversity_hamming_threshold=int(
            _cfg.PHASE2_DIVERSITY_HAMMING_THRESHOLD),
        diversity_recovery_min_unique_ratio=float(
            _cfg.PHASE2_DIVERSITY_RECOVERY_MIN_UNIQUE_RATIO
        ),
        diversity_recovery_inject_fraction=float(
            _cfg.PHASE2_DIVERSITY_RECOVERY_INJECT_FRACTION
        ),
        diversity_recovery_mutation_boost=float(
            _cfg.PHASE2_DIVERSITY_RECOVERY_MUTATION_BOOST
        ),
        plateau_early_stop_patience=int(
            _cfg.PHASE2_PLATEAU_EARLY_STOP_PATIENCE),
        plateau_early_stop_min_generation=int(
            _cfg.PHASE2_PLATEAU_EARLY_STOP_MIN_GENERATION
        ),
        early_stop_min_generation=int(_cfg.PHASE2_EARLY_STOP_MIN_GENERATION),
        seed_fraction=float(_cfg.PHASE2_ARCHIVE_SEED_FRACTION),
        return_floor_pct=float(_cfg.PHASE2_RETURN_FLOOR_PCT),
        min_trade_support=int(_cfg.MIN_TRADE_SUPPORT),
        use_robust_return_obj=bool(_cfg.PHASE2_USE_ROBUST_RETURN_OBJ),
        soft_feasibility=False,
        pool_require_positive_splits=bool(
            _cfg.PHASE2_POOL_REQUIRE_POSITIVE_SPLITS
        ),
    )
