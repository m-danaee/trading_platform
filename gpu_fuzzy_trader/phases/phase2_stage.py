"""Phase 2 two-stage search hyperparameter profiles (exploration vs refinement)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from gpu_fuzzy_trader import config as _cfg

StageLabel = Literal["A", "B"] | None


@dataclass(frozen=True)
class IslandStagePlan:
    """Resolved stage and remaining generation budget for one symbol island."""

    stage: StageLabel
    remaining_in_stage: int
    entering_stage_b: bool
    two_stage_active: bool


@dataclass(frozen=True)
class Phase2StageParams:
    """Resolved evolution knobs for a Phase 2 stage (or single-stage fallback)."""

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


def resolve_phase2_stage_params(stage: StageLabel) -> Phase2StageParams:
    """
    Return stage-tuned hyperparameters.

    When *stage* is None (single-stage Phase 2), global ``PHASE2_*`` defaults apply.
    """
    if stage == "A":
        return Phase2StageParams(
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
        )
    if stage == "B":
        return Phase2StageParams(
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
        )
    return Phase2StageParams(
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
    )


def island_stage_budgets(
    total_generations: int | None = None,
) -> tuple[int, int]:
    """
    Split an island's total generation budget into Stage A / Stage B portions.

    Uses the same A:B ratio as ``PHASE2_STAGE_A_GENERATIONS`` /
    ``PHASE2_STAGE_B_GENERATIONS`` but scaled to *total_generations*.
    """
    total = int(
        total_generations if total_generations is not None else _cfg.PHASE2_GENERATIONS)
    if not bool(getattr(_cfg, "PHASE2_TWO_STAGE_ENABLED", False)):
        return total, 0
    a_full = int(_cfg.PHASE2_STAGE_A_GENERATIONS)
    b_full = int(_cfg.PHASE2_STAGE_B_GENERATIONS)
    if a_full <= 0 or b_full <= 0:
        return total, 0
    stage_a = max(1, int(round(total * a_full / (a_full + b_full))))
    stage_b = max(0, total - stage_a)
    return stage_a, stage_b


def resolve_island_stage(
    generations_done: int,
    total_generations: int | None = None,
) -> IslandStagePlan:
    """Map completed island generations to the active two-stage profile."""
    total = int(
        total_generations if total_generations is not None else _cfg.PHASE2_GENERATIONS)
    stage_a, stage_b = island_stage_budgets(total)
    if stage_b <= 0:
        active = bool(getattr(_cfg, "PHASE2_TWO_STAGE_ENABLED", False))
        return IslandStagePlan(
            "A" if active else None,
            max(0, total - int(generations_done)),
            False,
            active,
        )
    done = int(generations_done)
    if done < stage_a:
        return IslandStagePlan("A", stage_a - done, False, True)
    entering_b = done == stage_a
    if done < stage_a + stage_b:
        return IslandStagePlan("B", stage_a + stage_b - done, entering_b, True)
    return IslandStagePlan("B", 0, False, True)
