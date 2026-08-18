"""Structured JSON Rule Archives for Hierarchical MTF.

Provides schema validation, SHA-256 hash generation, and atomic persistence
for discovered rule pools across HWC, MWC, and LWC timeframes.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import tempfile
from typing import Any, Sequence, Union

logger = logging.getLogger(__name__)

ARCHIVE_SCHEMA_VERSION: str = "2.0.0"
VALID_TIMEFRAMES: set[str] = {"hwc", "mwc", "lwc", "240m", "60m", "15m"}
TIMEFRAME_CANONICAL_MAP: dict[str, str] = {
    "hwc": "hwc",
    "240m": "hwc",
    "4h": "hwc",
    "mwc": "mwc",
    "60m": "mwc",
    "1h": "mwc",
    "lwc": "lwc",
    "15m": "lwc",
}


def normalize_timeframe(timeframe: str) -> str:
    """Normalize timeframe string to canonical identifier ('hwc', 'mwc', 'lwc')."""
    tf_clean = str(timeframe).strip().lower()
    if tf_clean not in TIMEFRAME_CANONICAL_MAP:
        raise ValueError(
            f"Invalid timeframe '{timeframe}'. Must be one of {sorted(TIMEFRAME_CANONICAL_MAP.keys())}"
        )
    return TIMEFRAME_CANONICAL_MAP[tf_clean]


def compute_rule_hash(rule: dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hash of the complete rule identity.

    Condition list order is normalized (sorted) to ensure permutation invariance,
    while validation metrics and provenance fields remain part of the identity.
    This prevents a stale archive from retaining the same hash after its OOF
    evidence or data lineage changes.
    """
    canonical_obj = dict(rule)
    canonical_obj.pop("rule_hash", None)
    tf = canonical_obj.get("timeframe", canonical_obj.get("tf", ""))
    canonical_obj["timeframe"] = normalize_timeframe(tf) if tf else ""
    canonical_obj.pop("tf", None)
    canonical_obj["direction"] = str(canonical_obj.get("direction", "")).strip().lower()
    canonical_obj["conditions"] = sorted(
        str(c).strip() for c in canonical_obj.get("conditions", [])
    )
    encoded = json.dumps(
        canonical_obj, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compute_archive_hash(rules_or_payload: Union[Sequence[dict[str, Any]], dict[str, Any]]) -> str:
    """Compute deterministic SHA-256 hash of rules plus stable archive metadata."""
    if isinstance(rules_or_payload, dict) and "rules" in rules_or_payload:
        rules_seq = rules_or_payload["rules"]
        canonical_obj: Any = {
            "timeframe": (
                normalize_timeframe(rules_or_payload["timeframe"])
                if rules_or_payload.get("timeframe")
                else ""
            ),
            "metadata": rules_or_payload.get("metadata", {}),
            "rules": sorted(compute_rule_hash(r) for r in rules_seq),
        }
    else:
        rules_seq = rules_or_payload  # type: ignore
        canonical_obj = {"rules": sorted(compute_rule_hash(r) for r in rules_seq)}

    encoded = json.dumps(
        canonical_obj, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_rule_schema(
    rule: dict[str, Any],
    raise_error: bool = True,
    *,
    require_provenance: bool = False,
) -> bool:
    """Validate that a rule dictionary conforms to the required MTF rule schema.

    Required fields:
        - timeframe: valid timeframe ('hwc', 'mwc', 'lwc', etc.)
        - direction: 'long' or 'short'
        - conditions: non-empty list of condition strings
        - coverage: float or float-compatible in [0, 1]
    """
    # 1. Required fields check
    for field in ("timeframe", "direction", "conditions", "coverage"):
        if field not in rule and (field != "timeframe" or "tf" not in rule):
            if raise_error:
                raise ValueError(f"Missing required rule field '{field}' in rule: {rule}")
            return False

    # 2. Timeframe validation
    raw_tf = str(rule.get("timeframe", rule.get("tf", ""))).strip().lower()
    if raw_tf not in TIMEFRAME_CANONICAL_MAP:
        if raise_error:
            raise ValueError(f"Invalid timeframe '{raw_tf}' in rule: {rule}")
        return False

    # 3. Direction validation
    direction = str(rule.get("direction", "")).strip().lower()
    if direction not in ("long", "short", "buy", "sell", "1", "-1"):
        if raise_error:
            raise ValueError(f"Invalid direction '{direction}' in rule: {rule}")
        return False

    # 4. Conditions validation
    conditions = rule.get("conditions")
    if not isinstance(conditions, (list, tuple)) or len(conditions) == 0:
        if raise_error:
            raise ValueError(f"Rule conditions must be a non-empty list: {rule}")
        return False
    if any(not isinstance(condition, str) or not condition.strip() for condition in conditions):
        if raise_error:
            raise ValueError("Rule conditions must contain non-empty strings")
        return False

    # 5. Coverage validation
    try:
        cov = float(rule["coverage"])
        if not (0.0 <= cov <= 1.0):
            if raise_error:
                raise ValueError(f"Coverage {cov} must be in [0, 1]")
            return False
    except (ValueError, TypeError):
        if raise_error:
            raise ValueError(f"Invalid coverage value '{rule.get('coverage')}'")
        return False

    if require_provenance:
        oof_metrics = rule.get("oof_metrics")
        if not isinstance(oof_metrics, dict):
            if raise_error:
                raise ValueError("Production MTF rules require an 'oof_metrics' mapping")
            return False
        for metric in ("directional_edge", "mcc", "coverage", "stability"):
            if metric not in oof_metrics:
                if raise_error:
                    raise ValueError(
                        f"Production MTF rule oof_metrics is missing {metric!r}"
                    )
                return False
        for field in ("data_hash", "feature_schema_hash"):
            if not rule.get(field):
                if raise_error:
                    raise ValueError(f"Production MTF rule is missing {field!r}")
                return False

    return True


def validate_archive_schema(payload: dict[str, Any], raise_error: bool = True) -> bool:
    """Validate that an archive dictionary conforms to the MTF Archive schema."""
    if not isinstance(payload, dict):
        if raise_error:
            raise ValueError(f"Archive payload must be a dict, got {type(payload)}")
        return False

    # Check schema version
    version = str(payload.get("schema_version", ""))
    if version != ARCHIVE_SCHEMA_VERSION:
        if raise_error:
            raise ValueError(
                f"Unsupported schema_version '{version}'. Expected '{ARCHIVE_SCHEMA_VERSION}'."
            )
        return False

    # Check timeframe
    raw_tf = str(payload.get("timeframe", "")).strip().lower()
    if raw_tf not in TIMEFRAME_CANONICAL_MAP:
        if raise_error:
            raise ValueError(f"Invalid archive timeframe '{raw_tf}'.")
        return False

    # Check rules list
    rules = payload.get("rules")
    if not isinstance(rules, list):
        if raise_error:
            raise ValueError(f"Archive rules must be a list, got {type(rules)}")
        return False

    require_provenance = bool(
        isinstance(payload.get("metadata"), dict)
        and payload["metadata"].get("provenance_required", False)
    )
    for rule in rules:
        if not isinstance(rule, dict):
            if raise_error:
                raise ValueError("Archive rules must contain mapping objects")
            return False
        if not validate_rule_schema(
            rule, raise_error=raise_error, require_provenance=require_provenance
        ):
            return False
        declared_rule_hash = rule.get("rule_hash")
        if declared_rule_hash and str(declared_rule_hash) != compute_rule_hash(rule):
            if raise_error:
                raise ValueError("Rule hash does not match the stored rule contents")
            return False

    try:
        declared_rule_count = int(payload.get("rule_count", len(rules)))
    except (TypeError, ValueError):
        if raise_error:
            raise ValueError("Archive rule_count must be an integer")
        return False
    if declared_rule_count != len(rules):
        if raise_error:
            raise ValueError("Archive rule_count does not match the stored rules")
        return False
    declared_hash = payload.get("archive_hash")
    if declared_hash:
        expected_hash = compute_archive_hash(payload)
        if str(declared_hash) != expected_hash:
            if raise_error:
                raise ValueError("Archive hash does not match its rules and metadata")
            return False

    return True


def get_default_archive_path(timeframe: str, base_dir: Union[str, Path] = "rule_archives") -> Path:
    """Return canonical path for a timeframe rule archive."""
    canonical_tf = normalize_timeframe(timeframe)
    return Path(base_dir) / canonical_tf / f"{canonical_tf}_rules.json"


def save_mtf_rule_archive(
    timeframe: str,
    rules: Sequence[dict[str, Any]],
    path: Union[str, Path, None] = None,
    metadata: dict[str, Any] | None = None,
    base_dir: Union[str, Path] = "rule_archives",
    require_provenance: bool = False,
) -> str:
    """Persist discovered MTF rules into a structured, hashed JSON archive atomically.

    Parameters
    ----------
    timeframe : str
        Timeframe identifier ('hwc', 'mwc', 'lwc', etc.).
    rules : Sequence[dict[str, Any]]
        List of rule dictionaries to save.
    path : str, Path, or None, optional
        Target file path. If None, resolves to default archive path.
    metadata : dict or None, optional
        Additional metadata (dataset hashes, fit timestamps, fold counts, etc.).
    base_dir : str or Path, default "rule_archives"
        Base directory if path is not specified.

    Returns
    -------
    str
        Deterministic SHA-256 hash of the archive contents.
    """
    canonical_tf = normalize_timeframe(timeframe)
    target_path = Path(path) if path is not None else get_default_archive_path(canonical_tf, base_dir)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    enriched_rules: list[dict[str, Any]] = []
    for r in rules:
        rule_copy = dict(r)
        rule_copy["timeframe"] = canonical_tf
        rule_copy["direction"] = str(rule_copy.get("direction", "")).strip().lower()
        rule_copy["complexity"] = int(rule_copy.get("complexity", len(rule_copy.get("conditions", []))))
        rule_copy["rule_hash"] = compute_rule_hash(rule_copy)
        validate_rule_schema(
            rule_copy,
            raise_error=True,
            require_provenance=require_provenance,
        )
        enriched_rules.append(rule_copy)

    archive_metadata = dict(metadata or {})
    if require_provenance:
        archive_metadata["provenance_required"] = True

    archive_hash = compute_archive_hash({
        "timeframe": canonical_tf,
        "metadata": archive_metadata,
        "rules": enriched_rules,
    })

    payload: dict[str, Any] = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "timeframe": canonical_tf,
        "archive_hash": archive_hash,
        "rule_count": len(enriched_rules),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": archive_metadata,
        "rules": enriched_rules,
    }

    # Atomic file write via tempfile in same directory
    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(target_path.parent),
        delete=False,
        suffix=".tmp",
    )
    try:
        json.dump(payload, temp_file, indent=2, ensure_ascii=False)
        temp_file.flush()
        temp_file.close()
        Path(temp_file.name).replace(target_path)
    except Exception:
        if Path(temp_file.name).exists():
            Path(temp_file.name).unlink()
        raise

    logger.info(
        "Saved %d MTF rules for %s to %s (archive_hash=%s)",
        len(enriched_rules),
        canonical_tf,
        target_path,
        archive_hash,
    )
    return archive_hash


def load_mtf_rule_archive(
    path: Union[str, Path],
    validate_schema: bool = True,
) -> list[dict[str, Any]]:
    """Load list of rules from an MTF rule archive JSON file.

    Parameters
    ----------
    path : str or Path
        Path to the archive JSON file.
    validate_schema : bool, default True
        Whether to enforce schema validation on loaded rules.

    Returns
    -------
    list[dict[str, Any]]
        List of rule dictionaries.
    """
    payload = load_mtf_archive_payload(path, validate_schema=validate_schema)
    if "rules" in payload and isinstance(payload["rules"], list):
        return payload["rules"]
    elif isinstance(payload, list):
        return payload
    return []


def load_mtf_archive_payload(
    path: Union[str, Path],
    validate_schema: bool = True,
) -> dict[str, Any]:
    """Load full payload dictionary from an MTF rule archive JSON file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"MTF rule archive file not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if validate_schema and isinstance(payload, dict):
        validate_archive_schema(payload, raise_error=True)

    return payload
