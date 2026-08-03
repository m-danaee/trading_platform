"""Stable provenance identifiers for Phase 2 feature rules.

The RB Governor may change execution scope and risk parameters, but a feature
rule discovered by Phase 2 should remain auditable.  These helpers deliberately
exclude symbol conditions from the feature identity: symbol scope is execution
metadata, while the ordered feature-condition list is the discovered logic.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any


def feature_conditions_only(conditions: Iterable[object]) -> list[str]:
    """Return normalized non-symbol conditions without changing their order."""
    result: list[str] = []
    for condition in conditions:
        text = str(condition).strip()
        lowered = text.lower()
        if lowered.startswith("symbol is ") or lowered.startswith("[symbol] is "):
            continue
        result.append(text)
    return result


def phase2_rule_id(
    conditions: Iterable[object],
    *,
    direction: str | None = None,
    source_symbols: Iterable[object] | None = None,
) -> str:
    """Hash the immutable Phase 2 feature logic and its discovery scope."""
    payload = {
        "direction": str(direction).strip().lower() if direction else None,
        "feature_conditions": feature_conditions_only(conditions),
        "source_symbols": sorted(
            {str(symbol).strip() for symbol in (source_symbols or []) if str(symbol).strip()}
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:20]


def strategy_id(
    *,
    direction: str,
    rules: Iterable[dict[str, Any]],
    horizon_bars: int,
    cost_model_id: str,
) -> str:
    """Hash the complete economic strategy, including exit policy.

    ``phase2_rule_id`` intentionally identifies only discovered feature logic.
    This identity is stricter: changing TP, SL, horizon, symbol scope, or rule
    order creates a new strategy package and therefore a new research object.
    """
    normalized_rules: list[dict[str, Any]] = []
    for rule in rules:
        conditions = list(rule.get("conditions", []))
        feature_conditions = feature_conditions_only(conditions)
        source_symbols = rule.get(
            "eligible_symbols",
            rule.get("source_symbols", []),
        )
        normalized_rules.append({
            "feature_conditions": feature_conditions,
            "eligible_symbols": sorted(
                {str(symbol).strip() for symbol in source_symbols if str(symbol).strip()}
            ),
            "tp": float(rule.get("tp", 0.0)),
            "sl": float(rule.get("sl", 0.0)),
        })
    payload = {
        "direction": str(direction).strip().lower(),
        "horizon_bars": int(horizon_bars),
        "cost_model_id": str(cost_model_id),
        "rules": normalized_rules,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()[:24]

