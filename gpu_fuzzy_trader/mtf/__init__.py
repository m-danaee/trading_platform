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
from gpu_fuzzy_trader.mtf.candidate import HierarchicalStrategyCandidate
from gpu_fuzzy_trader.mtf.composer import (
    DEFAULT_MIN_EVIDENCE_STRENGTH,
    DEFAULT_V_HWC_LONG,
    DEFAULT_V_HWC_SHORT,
    DEFAULT_V_MWC_LONG,
    DEFAULT_V_MWC_SHORT,
    compose_bidirectional_signals,
    compose_hierarchical_signals,
    normalize_direction,
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
from gpu_fuzzy_trader.mtf.diagnostics import (
    DEFAULT_RETENTION_FLOOR,
    DEFAULT_RETENTION_TARGET,
    MIN_RETENTION_SAMPLE,
    TradeRetentionDiagnostics,
    compute_granular_retention_diagnostics,
    compute_trade_retention_diagnostics,
    format_retention_report,
)
from gpu_fuzzy_trader.mtf.ensembler import (
    compute_ensemble_direction_and_strength,
    compute_rule_weights,
    deduplicate_rules,
)
from gpu_fuzzy_trader.mtf.discovery import (
    LayerDiscoveryResult,
    discover_directional_layer,
)
from gpu_fuzzy_trader.mtf.runtime import (
    attach_frozen_layer_scores,
    attach_oof_layer_scores,
    evaluate_candidate_frame,
    prepare_causal_mtf_frame,
)

__all__ = [
    "ARCHIVE_SCHEMA_VERSION",
    "DEFAULT_HWC_PURGE_MINUTES",
    "DEFAULT_LWC_PURGE_MINUTES",
    "DEFAULT_MIN_EVIDENCE_STRENGTH",
    "DEFAULT_MWC_PURGE_MINUTES",
    "DEFAULT_RETENTION_FLOOR",
    "DEFAULT_RETENTION_TARGET",
    "DEFAULT_V_HWC_LONG",
    "DEFAULT_V_HWC_SHORT",
    "DEFAULT_V_MWC_LONG",
    "DEFAULT_V_MWC_SHORT",
    "HierarchicalStrategyCandidate",
    "LayerDiscoveryResult",
    "MIN_RETENTION_SAMPLE",
    "TemporalFold",
    "TradeRetentionDiagnostics",
    "apply_purge_embargo",
    "attach_frozen_layer_scores",
    "attach_oof_layer_scores",
    "build_master_temporal_folds",
    "compose_bidirectional_signals",
    "compose_hierarchical_signals",
    "compute_archive_hash",
    "compute_ensemble_direction_and_strength",
    "compute_granular_retention_diagnostics",
    "compute_rule_hash",
    "compute_rule_weights",
    "compute_trade_retention_diagnostics",
    "deduplicate_rules",
    "discover_directional_layer",
    "evaluate_candidate_frame",
    "export_fold_boundaries",
    "format_retention_report",
    "generate_oof_predictions",
    "generate_oof_scores",
    "get_default_archive_path",
    "load_mtf_archive_payload",
    "load_mtf_rule_archive",
    "normalize_direction",
    "normalize_timeframe",
    "prepare_causal_mtf_frame",
    "save_mtf_rule_archive",
    "validate_archive_schema",
    "validate_master_temporal_folds",
    "validate_rule_schema",
]
