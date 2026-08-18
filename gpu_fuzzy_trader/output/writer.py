"""
writer.py — Output_Writer

Serializes RuleSet dicts to JSON with exact schema validation and
loads/validates existing JSON files.

Schema (must match evaluator_v5.ipynb exactly):
{
  "direction": "long" | "short",
  "rules_set": [
    {
      "tp": <float>,
      "sl": <float>,
      "capital_pct": <float>,
      "conditions": [
        "[feature_name] IS Fuzzy Value Name",
        "symbol is 1",
        "[symbol] IS 1",
        ...
      ]
    }
  ]
}

Constraints (Requirements 12.1–12.9):
  - direction must be "long" or "short"
  - rules_set must contain RB_MIN_RULES–RB_MAX_RULES rules
      - Truncates rules_set to RB_MAX_RULES if needed (log WARNING).
      - Rules must already be score-ranked (RB Governor sorts before write).
  - Each rule must have exactly: tp, sl, capital_pct, conditions
  - tp, sl, capital_pct must be finite positive floats
  - conditions must be a non-empty list of strings: feature conditions match
    [feature_name] IS Fuzzy Value Name; optional symbol filters match
    symbol is X or [symbol] IS X (evaluator_v5 parity)
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.backtest.symbol_conditions import (
    MAX_SYMBOL_FILTERS_PER_RULE,
    parse_symbol_condition,
    split_feature_and_symbol_conditions,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class ValidationError(Exception):
    """Raised when a rule set fails schema validation."""


# ---------------------------------------------------------------------------
# Condition string validation
# ---------------------------------------------------------------------------

# Exported fuzzy conditions plus frozen numeric thresholds emitted by the MTF
# directional search.
_CONDITION_RE = re.compile(r"^\s*\[(.+?)\]\s+IS\s+(.+?)\s*$")
_NUMERIC_CONDITION_RE = re.compile(
    r"^\s*\[(.+?)\]\s*(>=|<=|==|>|<)\s*"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$"
)


def _condition_feature(condition: str) -> str | None:
    m = _CONDITION_RE.match(condition)
    if not m:
        return None
    return m.group(1).strip()


def _context_feature_direction(feature: str) -> str | None:
    """Return the strategy direction a context column belongs to (if any)."""
    if feature in _cfg.CONTEXT_PERMISSION_COLUMNS or feature in _cfg.CONTEXT_TRIGGER_COLUMNS:
        for direction in ("long", "short"):
            if feature == _cfg.context_permission_column(direction) or \
               feature == _cfg.context_trigger_column(direction):
                return direction
    return None


def _validate_context_contract(rule_set: dict, validated_rules: list[dict]) -> None:
    """Enforce the mandatory trend-context contract on a strategy.

    Rejects:
      - rules containing opposite-direction context conditions,
      - duplicate mandatory context conditions,
      - (when the strategy carries any context condition, or the strict flag
        is set) rules missing the direction's two mandatory conditions.

    MIN_CONDITIONS / MAX_CONDITIONS count evolved conditions; the fixed
    context conditions are policy, not ordinary genes.
    """
    direction = rule_set.get("direction")
    if direction not in ("long", "short"):
        return
    mandatory = _cfg.mandatory_context_conditions(direction)

    for idx, rule in enumerate(validated_rules, start=1):
        conditions = rule["conditions"]
        present = [c for c in conditions if isinstance(c, str)]

        opposite_cols = set()
        for direction_name in ("long", "short"):
            if direction_name != direction:
                opposite_cols.add(_cfg.context_permission_column(direction_name))
                opposite_cols.add(_cfg.context_trigger_column(direction_name))
        bad_opposite = [c for c in present if _condition_feature(c) in opposite_cols]
        if bad_opposite:
            raise ValidationError(
                f"Rule {idx}: contains opposite-direction context condition(s) "
                f"{bad_opposite} for direction {direction!r}. Direction-specific "
                "context cannot cross sides."
            )

        # Duplicate mandatory conditions are rejected.
        seen: set[str] = set()
        for c in present:
            feat = _condition_feature(c)
            if feat and feat in _cfg.CONTEXT_COLUMNS:
                if c in seen:
                    raise ValidationError(
                        f"Rule {idx}: duplicate mandatory context condition "
                        f"{c!r}."
                    )
                seen.add(c)

        if _cfg.REQUIRE_CONTEXT_IN_STRATEGY:
            for ctx_condition in mandatory:
                if ctx_condition not in present:
                    raise ValidationError(
                        f"Rule {idx}: missing mandatory context condition "
                        f"{ctx_condition!r} for direction {direction!r}."
                    )


def _validate_symbol_condition(condition: str) -> None:
    """Validate an optional symbol filter (symbol is X / [symbol] IS X)."""
    try:
        parsed_symbols = parse_symbol_condition(condition)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc

    if parsed_symbols is None:
        raise ValidationError(
            f"Condition {condition!r} is not a valid symbol filter."
        )
    if not parsed_symbols:
        raise ValidationError(
            f"Symbol filter {condition!r} must include at least one symbol value."
        )
    if len(parsed_symbols) > MAX_SYMBOL_FILTERS_PER_RULE:
        raise ValidationError(
            f"Symbol filter {condition!r} includes more than "
            f"{MAX_SYMBOL_FILTERS_PER_RULE} symbol values."
        )


def _validate_condition(condition: str) -> None:
    """
    Validate a single condition string.

    Accepts either:
      - [feature_name] IS Fuzzy Value Name
      - symbol is X / [symbol] IS X (including comma-separated lists)

    Raises ValidationError if invalid.
    """
    if not isinstance(condition, str):
        raise ValidationError(
            f"Condition must be a string, got {type(condition).__name__!r}: {condition!r}"
        )

    if parse_symbol_condition(condition) is not None:
        _validate_symbol_condition(condition)
        return

    m = _CONDITION_RE.match(condition)
    numeric = _NUMERIC_CONDITION_RE.match(condition)
    if not m and not numeric:
        raise ValidationError(
            f"Condition {condition!r} does not match the required pattern "
            "'[feature_name] IS Fuzzy Value Name' or a numeric comparison."
        )
    if numeric:
        feature_name = numeric.group(1).strip()
        value_name = numeric.group(3).strip()
    else:
        assert m is not None
        feature_name = m.group(1).strip()
        value_name = m.group(2).strip()
    if not feature_name:
        raise ValidationError(
            f"Condition {condition!r} has an empty feature name."
        )
    if not value_name:
        raise ValidationError(
            f"Condition {condition!r} has an empty fuzzy value name."
        )


# ---------------------------------------------------------------------------
# Rule object validation
# ---------------------------------------------------------------------------

_REQUIRED_RULE_KEYS = {"tp", "sl", "capital_pct", "conditions"}


def _validate_rule(rule: object, rule_index: int) -> dict:
    """
    Validate a single rule object.

    Returns the validated rule dict (with numeric fields cast to float).
    Raises ValidationError on any violation.
    """
    if not isinstance(rule, dict):
        raise ValidationError(
            f"Rule {rule_index}: expected a dict, got {type(rule).__name__!r}."
        )

    missing = _REQUIRED_RULE_KEYS - rule.keys()
    if missing:
        raise ValidationError(
            f"Rule {rule_index}: missing required keys: {sorted(missing)}."
        )

    # Cast numeric fields
    try:
        tp = float(rule["tp"])
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"Rule {rule_index}: 'tp' must be a number, got {rule['tp']!r}."
        ) from exc

    try:
        sl = float(rule["sl"])
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"Rule {rule_index}: 'sl' must be a number, got {rule['sl']!r}."
        ) from exc

    try:
        capital_pct = float(rule["capital_pct"])
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"Rule {rule_index}: 'capital_pct' must be a number, got {rule['capital_pct']!r}."
        ) from exc

    for field_name, value in (
        ("tp", tp),
        ("sl", sl),
        ("capital_pct", capital_pct),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValidationError(
                f"Rule {rule_index}: '{field_name}' must be finite and positive."
            )

    # Validate conditions
    conditions = rule.get("conditions")
    if not isinstance(conditions, list) or len(conditions) == 0:
        raise ValidationError(
            f"Rule {rule_index}: 'conditions' must be a non-empty list of strings."
        )

    for i, cond in enumerate(conditions):
        try:
            _validate_condition(cond)
        except ValidationError as exc:
            raise ValidationError(
                f"Rule {rule_index}, condition {i}: {exc}"
            ) from exc

    try:
        split_feature_and_symbol_conditions(list(conditions), rule_number=rule_index)
    except ValueError as exc:
        raise ValidationError(f"Rule {rule_index}: {exc}") from exc

    return {
        "tp": tp,
        "sl": sl,
        "capital_pct": capital_pct,
        "conditions": list(conditions),
    }


# ---------------------------------------------------------------------------
# Top-level rule set validation
# ---------------------------------------------------------------------------

def _validate_rule_set(rule_set: object) -> dict:
    """
    Validate and normalise a rule_set dict.

    Applies all schema constraints (Requirements 12.1–12.9):
      - Checks top-level keys
      - Validates direction
      - Truncates rules_set to RB_MAX_RULES if needed (log WARNING)
      - Validates RB_MIN_RULES–RB_MAX_RULES after truncation
        (Exception: empty rules_set is allowed when ``deployment_accepted`` is
        explicitly ``False`` — the fail-closed RB fallback path.)
      - Validates each rule object

    Returns a normalised dict ready for JSON serialisation.
    Raises ValidationError on any violation.
    """
    if not isinstance(rule_set, dict):
        raise ValidationError(
            f"rule_set must be a dict, got {type(rule_set).__name__!r}."
        )

    # Requirement 12.1: top-level keys
    missing_top = {"direction", "rules_set"} - rule_set.keys()
    if missing_top:
        raise ValidationError(
            f"rule_set is missing required top-level keys: {sorted(missing_top)}."
        )

    # Requirement 12.2: direction
    direction = rule_set.get("direction")
    if direction not in ("long", "short"):
        raise ValidationError(
            f"'direction' must be 'long' or 'short', got {direction!r}."
        )

    # Requirement 12.3: rules_set is a list
    rules_list = rule_set.get("rules_set")
    if not isinstance(rules_list, list):
        raise ValidationError(
            f"'rules_set' must be a list, got {type(rules_list).__name__!r}."
        )

    # Output bounds belong to the active RB Governor path.
    schema_max = int(_cfg.RB_MAX_RULES)
    schema_min = int(_cfg.RB_MIN_RULES)
    if len(rules_list) > schema_max:
        logger.warning(
            "rules_set contains %d rules (max %d); truncating to top %d "
            "(score-ranked list order).",
            len(rules_list), schema_max, schema_max,
        )
        rules_list = rules_list[:schema_max]

    # Requirement 12.8: must have at least min rules
    # Exception: empty rules_set is allowed when deployment_accepted is False
    # (fail-closed path: no positive-good candidates found, intentionally empty).
    if len(rules_list) < schema_min:
        is_fail_closed = len(rules_list) == 0 and rule_set.get("deployment_accepted") is False
        if not is_fail_closed:
            raise ValidationError(
                f"'rules_set' must contain at least {schema_min} rules, got {len(rules_list)}."
            )

    # Validate each rule
    validated_rules = []
    for i, rule in enumerate(rules_list, start=1):
        validated_rules.append(_validate_rule(rule, i))

    _validate_context_contract(rule_set, validated_rules)

    normalized = {
        "direction": direction,
        "rules_set": validated_rules,
    }
    # Preserve audit metadata in the full strategy artifact.  The evaluator
    # clean writer below still strips this to the strict notebook schema.
    for key in (
        "strategy_id",
        "strategy_contract",
        "provenance",
        "deployment_accepted",
        "deployment_reason",
        "fail_closed",
        "reason",
        "mtf_candidate",
        "mtf_manifest",
        "mtf_runtime",
    ):
        if key in rule_set:
            normalized[key] = rule_set[key]
    return normalized


# ---------------------------------------------------------------------------
# Evaluator-clean writer
# ---------------------------------------------------------------------------


def write_evaluator_clean(strategy: dict, output_path: str | Path) -> None:
    """
    Write a stripped strategy file containing only ``direction`` and
    ``rules_set``.

    Extra top-level keys (e.g. ``risk_optimized``, ``deployment_accepted``,
    ``validation_gate``) are stripped. This is a safety net for evaluators
    that may reject unknown top-level keys in future versions.

    Parameters
    ----------
    strategy : dict
        The full strategy dict (must contain ``direction`` and ``rules_set``).
    output_path : str or Path
        Destination path for the clean JSON file. Parent directories are
        created automatically if they do not exist.

    Raises
    ------
    KeyError
        If ``strategy`` is missing ``direction`` or ``rules_set``.
    """
    clean = {
        "direction": strategy["direction"],
        "rules_set": strategy["rules_set"],
    }
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        json.dump(clean, fh, indent=2)
    logger.info(
        "Wrote evaluator-clean file (%s, %d rules) to %s",
        clean["direction"],
        len(clean["rules_set"]),
        dest,
    )


def _maybe_write_evaluator_clean(
    strategy: dict, main_path: str | Path, direction: str
) -> None:
    """
    Write a stripped strategy file alongside the main strategy JSON.

    This is a convenience helper for production pipeline code that writes
    strategy files via direct ``json.dump`` (Phases 3/4/5) rather than
    through ``Output_Writer.write``.

    Parameters
    ----------
    strategy : dict
        The full strategy dict (must contain ``direction`` and ``rules_set``).
    main_path : str or Path
        The path of the main strategy file that was just written. The clean
        file is placed next to it in an ``evaluator_clean/`` subdirectory.
    direction : str
        ``"long"`` or ``"short"`` — used to name the clean file.
    """
    main_path = Path(main_path)
    clean_path = main_path.parent / "evaluator_clean" / f"{direction}_evaluator_clean.json"
    try:
        write_evaluator_clean(strategy, clean_path)
    except Exception as exc:
        logger.debug("evaluator_clean write failed for %s: %s", direction, exc)


# ---------------------------------------------------------------------------
# Output_Writer
# ---------------------------------------------------------------------------

class Output_Writer:
    """
    Serializes RuleSet dicts to JSON and loads/validates existing JSON files.

    Methods
    -------
    write(rule_set, path)
        Validate rule_set and write to JSON at path. Also writes an
        evaluator-clean variant with only ``direction`` and ``rules_set``.
    load_and_validate(path)
        Load JSON from path and run full schema validation.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(self, rule_set: dict, path: str | Path) -> None:
        """
        Validate rule_set and write to JSON at path.

        After the main write, also writes an evaluator-clean file
        (``<parent>/evaluator_clean/<stem>_evaluator_clean.json``).

        Parameters
        ----------
        rule_set : dict
            Must conform to the output schema:
            {
              "direction": "long" | "short",
              "rules_set": [
                {"tp": float, "sl": float, "capital_pct": float,
                 "conditions": ["[feature_name] IS Fuzzy Value Name", ...]}
              ]
            }
        path : str or Path
            Destination file path. Parent directories are created if needed.

        Raises
        ------
        ValidationError
            If rule_set fails schema validation.
        """
        validated = _validate_rule_set(rule_set)

        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        with dest.open("w", encoding="utf-8") as fh:
            json.dump(validated, fh, indent=2)

        logger.info(
            "Wrote %d rules (%s) to %s",
            len(validated["rules_set"]),
            validated["direction"],
            dest,
        )

        # Write evaluator-clean variant (defensive — strip extra metadata).
        direction = validated["direction"]
        clean_dir = dest.parent / "evaluator_clean"
        clean_path = clean_dir / f"{direction}_evaluator_clean.json"
        write_evaluator_clean(validated, clean_path)

    def load_and_validate(self, path: str | Path) -> dict:
        """
        Load JSON from path and run full schema validation.

        Parameters
        ----------
        path : str or Path
            Path to the JSON file to load.

        Returns
        -------
        dict
            Validated and normalised rule set dict.

        Raises
        ------
        ValidationError
            If the file cannot be read, is not valid JSON, or fails schema
            validation.
        """
        src = Path(path)

        if not src.exists():
            raise ValidationError(f"File not found: {src}")

        try:
            with src.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValidationError(
                f"File {src} is not valid JSON: {exc}"
            ) from exc
        except OSError as exc:
            raise ValidationError(
                f"Cannot read file {src}: {exc}"
            ) from exc

        return _validate_rule_set(data)
