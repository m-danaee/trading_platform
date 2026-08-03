"""Typed, versioned profile for the active research contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ResearchProfile:
    """Small stable surface for comparing experiments.

    The legacy config module remains the source of defaults. This profile is
    the serialized contract recorded with a run, so unrelated compatibility
    knobs do not obscure the actual experiment delta.
    """

    schema_version: int
    phase2_tp: float
    phase2_sl: float
    phase2_min_conditions: int
    phase2_max_conditions: int
    phase2_population_size: int
    phase2_generations: int
    phase2_symbol_specialists: bool
    phase2_shared_generation_budget: bool
    phase2_migration_enabled: bool
    phase2_migration_topology: str
    phase2_val_in_fitness_penalty: bool
    rb_risk_optimize_exits: bool
    rb_candidate_risk_admission: bool
    rb_provenance_only: bool
    rb_allow_partial_coverage: bool
    rb_monthly_certificate: bool
    rb_monthly_min_profitable_ratio: float
    rb_monthly_max_bearish_ratio: float
    rb_cost_stress: bool
    phase1_stationarity_folds: int
    phase1_min_abs_corr: float
    phase1_symbol_union: bool

    @classmethod
    def from_config(cls, config: Any) -> "ResearchProfile":
        return cls(
            schema_version=1,
            phase2_tp=float(config.PHASE2_TP),
            phase2_sl=float(config.PHASE2_SL),
            phase2_min_conditions=int(config.MIN_CONDITIONS),
            phase2_max_conditions=int(config.MAX_CONDITIONS),
            phase2_population_size=int(config.PHASE2_POPULATION_SIZE),
            phase2_generations=int(config.PHASE2_GENERATIONS),
            phase2_symbol_specialists=bool(
                config.PHASE2_SYMBOL_SPECIALISTS_ENABLED
            ),
            phase2_shared_generation_budget=bool(
                config.PHASE2_SHARED_ISLAND_GENERATION_BUDGET
            ),
            phase2_migration_enabled=bool(config.PHASE2_MIGRATION_ENABLED),
            phase2_migration_topology=str(config.PHASE2_MIGRATION_TOPOLOGY),
            phase2_val_in_fitness_penalty=bool(
                config.PHASE2_VAL_IN_FITNESS_PENALTY
            ),
            rb_risk_optimize_exits=bool(config.RB_RISK_OPTIMIZE_EXITS),
            rb_candidate_risk_admission=bool(
                config.RB_CANDIDATE_RISK_ADMISSION_ENABLED
            ),
            rb_provenance_only=bool(config.RB_PHASE2_PROVENANCE_ONLY),
            rb_allow_partial_coverage=bool(
                config.RB_ALLOW_PARTIAL_SPECIALIST_COVERAGE
            ),
            rb_monthly_certificate=bool(config.RB_MONTHLY_CERTIFICATE_ENABLED),
            rb_monthly_min_profitable_ratio=float(
                config.RB_MONTHLY_MIN_PROFITABLE_RATIO
            ),
            rb_monthly_max_bearish_ratio=float(
                config.RB_MONTHLY_MAX_BEARISH_RATIO
            ),
            rb_cost_stress=bool(config.RB_COST_STRESS_ENABLED),
            phase1_stationarity_folds=int(config.PHASE1_STATIONARITY_FOLDS),
            phase1_min_abs_corr=float(
                config.PHASE1_SIGN_CONSISTENCY_MIN_ABS_CORR
            ),
            phase1_symbol_union=bool(config.PHASE1_SYMBOL_UNION_ENABLED),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def profile_id(self) -> str:
        encoded = json.dumps(
            self.as_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:20]
