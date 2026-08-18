"""gpu_fuzzy_trader.mtf — Hierarchical Multi-Timeframe Discovery & Ensembling."""

from __future__ import annotations

from gpu_fuzzy_trader.mtf.archives import (
    ARCHIVE_SCHEMA_VERSION,
    compute_archive_hash,
    compute_rule_hash,
    get_default_archive_path,
    load_mtf_archive_payload,
    load_mtf_rule_archive,
    normalize_timeframe,
    save_mtf_rule_archive,
    validate_archive_schema,
    validate_rule_schema,
)
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
from gpu_fuzzy_trader.mtf.ensembler import (
    compute_ensemble_direction_and_strength,
    compute_rule_weights,
    deduplicate_rules,
)

__all__ = [
    "ARCHIVE_SCHEMA_VERSION",
    "DEFAULT_HWC_PURGE_MINUTES",
    "DEFAULT_LWC_PURGE_MINUTES",
    "DEFAULT_MWC_PURGE_MINUTES",
    "TemporalFold",
    "apply_purge_embargo",
    "build_master_temporal_folds",
    "compute_archive_hash",
    "compute_ensemble_direction_and_strength",
    "compute_rule_hash",
    "compute_rule_weights",
    "deduplicate_rules",
    "export_fold_boundaries",
    "generate_oof_predictions",
    "generate_oof_scores",
    "get_default_archive_path",
    "load_mtf_archive_payload",
    "load_mtf_rule_archive",
    "normalize_timeframe",
    "save_mtf_rule_archive",
    "validate_archive_schema",
    "validate_master_temporal_folds",
    "validate_rule_schema",
]

