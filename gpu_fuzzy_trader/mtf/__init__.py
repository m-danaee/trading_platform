"""gpu_fuzzy_trader.mtf — Hierarchical Multi-Timeframe Discovery & Ensembling."""

from __future__ import annotations

from gpu_fuzzy_trader.mtf.cross_fitting import (
    DEFAULT_HWC_PURGE_MINUTES,
    DEFAULT_LWC_PURGE_MINUTES,
    DEFAULT_MWC_PURGE_MINUTES,
    TemporalFold,
    apply_purge_embargo,
    build_master_temporal_folds,
    export_fold_boundaries,
    generate_oof_predictions,
    generate_oof_scores,
    validate_master_temporal_folds,
)

__all__ = [
    "DEFAULT_HWC_PURGE_MINUTES",
    "DEFAULT_LWC_PURGE_MINUTES",
    "DEFAULT_MWC_PURGE_MINUTES",
    "TemporalFold",
    "apply_purge_embargo",
    "build_master_temporal_folds",
    "export_fold_boundaries",
    "generate_oof_predictions",
    "generate_oof_scores",
    "validate_master_temporal_folds",
]
