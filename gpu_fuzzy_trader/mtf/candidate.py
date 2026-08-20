"""Hierarchical MTF Strategy Candidate Container.

Encapsulates LWC execution rules, HWC macro-trend rules, MWC intermediate setup rules,
and asymmetric soft-veto composer parameters into a unified, serializable strategy object.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from gpu_fuzzy_trader.mtf.archives import (
    compute_archive_hash,
)
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
from gpu_fuzzy_trader.mtf.diagnostics import DEFAULT_RETENTION_FLOOR

logger = logging.getLogger(__name__)


class HierarchicalStrategyCandidate:
    """Encapsulates a hierarchical multi-timeframe strategy with LWC/MWC/HWC rules."""

    def __init__(
        self,
        direction: str,
        lwc_rules: Sequence[dict[str, Any]],
        hwc_rules: Sequence[dict[str, Any]] | None = None,
        mwc_rules: Sequence[dict[str, Any]] | None = None,
        composer_params: Mapping[str, Any] | None = None,
        strategy_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        mtf_manifest: Mapping[str, Any] | None = None,
    ) -> None:
        raw_dir = str(direction).strip().lower()
        if raw_dir in ("bidirectional", "both", "joint"):
            self.direction = "bidirectional"
        else:
            self.direction = normalize_direction(raw_dir)

        self.lwc_rules = [dict(r) for r in lwc_rules]
        self.hwc_rules = [dict(r) for r in (hwc_rules or [])]
        self.mwc_rules = [dict(r) for r in (mwc_rules or [])]

        default_params = {
            "v_hwc_long": DEFAULT_V_HWC_LONG,
            "v_hwc_short": DEFAULT_V_HWC_SHORT,
            "v_mwc_long": DEFAULT_V_MWC_LONG,
            "v_mwc_short": DEFAULT_V_MWC_SHORT,
            "min_evidence_strength_hwc": DEFAULT_MIN_EVIDENCE_STRENGTH,
            "min_evidence_strength_mwc": DEFAULT_MIN_EVIDENCE_STRENGTH,
            "retention_floor": DEFAULT_RETENTION_FLOOR,
        }
        if composer_params:
            default_params.update(dict(composer_params))
        self.composer_params = default_params

        self.metadata = dict(metadata or {})
        self.mtf_manifest = dict(mtf_manifest) if mtf_manifest is not None else None

        if strategy_id is not None:
            self.strategy_id = str(strategy_id)
        else:
            self.strategy_id = self._generate_strategy_id()

    def _generate_strategy_id(self) -> str:
        """Compute deterministic strategy SHA-256 identifier."""
        lwc_hash = compute_archive_hash(self.lwc_rules)
        hwc_hash = compute_archive_hash(self.hwc_rules)
        mwc_hash = compute_archive_hash(self.mwc_rules)
        payload = {
            "direction": self.direction,
            "lwc_hash": lwc_hash,
            "hwc_hash": hwc_hash,
            "mwc_hash": mwc_hash,
            "composer_params": self.composer_params,
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return f"mtf_{self.direction}_{hashlib.sha256(encoded).hexdigest()[:16]}"

    def compose(
        self,
        lwc_triggers: np.ndarray,
        hwc_direction: np.ndarray,
        hwc_strength: np.ndarray,
        mwc_direction: np.ndarray,
        mwc_strength: np.ndarray,
        output_dtype: Any = np.int8,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Evaluate hierarchical soft-veto composition on input arrays."""
        if self.direction == "bidirectional":
            return compose_bidirectional_signals(
                lwc_triggers=lwc_triggers,
                hwc_direction=hwc_direction,
                hwc_strength=hwc_strength,
                mwc_direction=mwc_direction,
                mwc_strength=mwc_strength,
                v_hwc_long=self.composer_params.get("v_hwc_long", DEFAULT_V_HWC_LONG),
                v_hwc_short=self.composer_params.get("v_hwc_short", DEFAULT_V_HWC_SHORT),
                v_mwc_long=self.composer_params.get("v_mwc_long", DEFAULT_V_MWC_LONG),
                v_mwc_short=self.composer_params.get("v_mwc_short", DEFAULT_V_MWC_SHORT),
                min_strength_hwc=self.composer_params.get("min_evidence_strength_hwc", DEFAULT_MIN_EVIDENCE_STRENGTH),
                min_strength_mwc=self.composer_params.get("min_evidence_strength_mwc", DEFAULT_MIN_EVIDENCE_STRENGTH),
                output_dtype=output_dtype,
            )
        else:
            v_hwc = (
                self.composer_params.get("v_hwc_long", DEFAULT_V_HWC_LONG)
                if self.direction == "long"
                else self.composer_params.get("v_hwc_short", DEFAULT_V_HWC_SHORT)
            )
            v_mwc = (
                self.composer_params.get("v_mwc_long", DEFAULT_V_MWC_LONG)
                if self.direction == "long"
                else self.composer_params.get("v_mwc_short", DEFAULT_V_MWC_SHORT)
            )
            min_str_hwc = self.composer_params.get("min_evidence_strength_hwc", DEFAULT_MIN_EVIDENCE_STRENGTH)
            min_str_mwc = self.composer_params.get("min_evidence_strength_mwc", DEFAULT_MIN_EVIDENCE_STRENGTH)

            return compose_hierarchical_signals(
                lwc_triggers=lwc_triggers,
                direction=self.direction,
                hwc_direction=hwc_direction,
                hwc_strength=hwc_strength,
                mwc_direction=mwc_direction,
                mwc_strength=mwc_strength,
                v_hwc=v_hwc,
                v_mwc=v_mwc,
                min_strength_hwc=min_str_hwc,
                min_strength_mwc=min_str_mwc,
                output_dtype=output_dtype,
            )

    def evaluate_frame(
        self,
        raw_df,
        history_df=None,
    ) -> tuple[np.ndarray, dict[str, Any], "pd.DataFrame"]:
        """Evaluate this frozen candidate on raw OHLCV rows.

        No thresholds, weights, features, or composer parameters are fitted by
        this method.  It is the runtime entry point used by Phase 5 OOS.
        """
        from gpu_fuzzy_trader.mtf.runtime import evaluate_candidate_frame

        return evaluate_candidate_frame(self, raw_df, history_df=history_df)

    def to_dict(self) -> dict[str, Any]:
        """Convert candidate to a JSON-serializable dictionary."""
        return {
            "strategy_id": self.strategy_id,
            "direction": self.direction,
            "lwc_rules": self.lwc_rules,
            "hwc_rules": self.hwc_rules,
            "mwc_rules": self.mwc_rules,
            "composer_params": self.composer_params,
            "metadata": self.metadata,
            "mtf_manifest": self.mtf_manifest,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HierarchicalStrategyCandidate:
        """Construct HierarchicalStrategyCandidate from dictionary."""
        return cls(
            direction=data.get("direction", "long"),
            lwc_rules=data.get("lwc_rules", data.get("rules_set", [])),
            hwc_rules=data.get("hwc_rules", []),
            mwc_rules=data.get("mwc_rules", []),
            composer_params=data.get("composer_params", {}),
            strategy_id=data.get("strategy_id"),
            metadata=data.get("metadata", {}),
            mtf_manifest=data.get("mtf_manifest"),
        )
