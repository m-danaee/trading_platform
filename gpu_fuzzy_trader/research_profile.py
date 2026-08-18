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
    phase1_disabled: bool
    phase1_dispersion_threshold: float
    phase1_top_k_features: int
    phase1_stationarity_folds: int
    phase1_min_abs_corr: float
    phase1_symbol_union: bool
    effective_phase1_symbol_union: bool

    @classmethod
    def from_config(cls, config: Any) -> "ResearchProfile":
        return cls(
            schema_version=2,
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
            phase1_disabled=bool(config.PHASE1_DISABLED),
            phase1_dispersion_threshold=float(config.PHASE1_DISPERSION_THRESHOLD),
            phase1_top_k_features=int(config.PHASE1_TOP_K_FEATURES),
            phase1_stationarity_folds=int(config.PHASE1_STATIONARITY_FOLDS),
            phase1_min_abs_corr=float(
                config.PHASE1_SIGN_CONSISTENCY_MIN_ABS_CORR
            ),
            phase1_symbol_union=bool(config.PHASE1_SYMBOL_UNION_ENABLED),
            effective_phase1_symbol_union=bool(
                not config.PHASE1_DISABLED
                and config.PHASE1_SYMBOL_UNION_ENABLED
                and config.PHASE2_SYMBOL_SPECIALISTS_ENABLED
            ),
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


@dataclass(frozen=True)
class RuleSearchProfile:
    """Configuration contract for evolutionary rule search at a specific timeframe layer.

    Defines search constraints and evaluation targets for HWC (4H), MWC (1H), and LWC (15m).
    """

    role: str
    timeframe_minutes: int
    min_conditions: int
    max_conditions: int
    target_coverage: tuple[float, float]
    forward_horizon_bars: int
    quantile: float = 0.60
    support_threshold: float = 0.20

    @classmethod
    def hwc(cls) -> "RuleSearchProfile":
        """Standard profile for 4H Macro Directional Bias rules."""
        return cls(
            role="hwc",
            timeframe_minutes=240,
            min_conditions=1,
            max_conditions=2,
            target_coverage=(0.20, 0.60),
            forward_horizon_bars=6,
            quantile=0.60,
            support_threshold=0.20,
        )

    @classmethod
    def mwc(cls) -> "RuleSearchProfile":
        """Standard profile for 1H Conditional Setup / Continuation rules."""
        return cls(
            role="mwc",
            timeframe_minutes=60,
            min_conditions=1,
            max_conditions=3,
            target_coverage=(0.10, 0.40),
            forward_horizon_bars=4,
            quantile=0.60,
            support_threshold=0.20,
        )

    @classmethod
    def lwc(cls) -> "RuleSearchProfile":
        """Standard profile for 15m Execution Trigger rules."""
        return cls(
            role="lwc",
            timeframe_minutes=15,
            min_conditions=2,
            max_conditions=4,
            target_coverage=(0.01, 0.20),
            forward_horizon_bars=0,
            quantile=0.0,
            support_threshold=0.0,
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


def get_rule_search_profile(role: str) -> RuleSearchProfile:
    """Retrieve standard RuleSearchProfile by layer role or timeframe name.

    Args:
        role: "hwc" / "4h", "mwc" / "1h", or "lwc" / "15m".

    Returns:
        RuleSearchProfile instance.
    """
    role_lower = role.strip().lower()
    if role_lower in ("hwc", "4h", "240m", "macro"):
        return RuleSearchProfile.hwc()
    elif role_lower in ("mwc", "1h", "60m", "setup"):
        return RuleSearchProfile.mwc()
    elif role_lower in ("lwc", "15m", "trigger", "execution"):
        return RuleSearchProfile.lwc()
    else:
        raise ValueError(
            f"Unknown rule search profile role: {role!r}. Expected 'hwc', 'mwc', or 'lwc'."
        )

