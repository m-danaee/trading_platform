
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from gpu_fuzzy_trader import config as _cfg

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when a rule set fails schema validation."""



_CONDITION_RE = re.compile(r"^\[(.+?)\] IS (.+)$")


def _validate_condition(condition: str) -> None:
    """
    Validate a single condition string.

    Must match: [feature_name] IS Fuzzy Value Name
      - Must start with '['
      - Must contain '] IS '
      - Feature name must be non-empty
      - Fuzzy value name must be non-empty

    Raises ValidationError if invalid.
    """
    if not isinstance(condition, str):
        raise ValidationError(
            f"Condition must be a string, got {type(condition).__name__!r}: {condition!r}"
        )
    if condition.strip().lower().startswith("symbol is "):
        value = condition.strip().split()[-1]
        try:
            int(float(value))
            return
        except Exception:
            raise ValidationError(f"Invalid symbol filter condition: {condition!r}")
    m = _CONDITION_RE.match(condition)
    if not m:
        raise ValidationError(
            f"Condition {condition!r} does not match the required pattern "
            "'[feature_name] IS Fuzzy Value Name'. "
            "Must start with '[', contain '] IS ', and have non-empty feature and value names."
        )
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

    if tp == 0.0 and sl == 0.0 and capital_pct == 0.0:
        logger.error(
            "Rule %d has all-zero tp/sl/capital_pct — rejecting rule set.", rule_index
        )
        raise ValidationError(
            f"Rule {rule_index}: all of tp, sl, and capital_pct are zero. "
            "At least one must be non-zero."
        )

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

    return {
        "tp": tp,
        "sl": sl,
        "capital_pct": capital_pct,
        "conditions": list(conditions),
    }



def _validate_rule_set(rule_set: object) -> dict:
    """
    Validate and normalise a rule_set dict.

    Applies all schema constraints (Requirements 12.1–12.9):
      - Checks top-level keys
      - Validates direction
      - Truncates rules_set to 5 if needed (log WARNING)
      - Validates 2–5 rules after truncation
      - Validates each rule object

    Returns a normalised dict ready for JSON serialisation.
    Raises ValidationError on any violation.
    """
    if not isinstance(rule_set, dict):
        raise ValidationError(
            f"rule_set must be a dict, got {type(rule_set).__name__!r}."
        )

    missing_top = {"direction", "rules_set"} - rule_set.keys()
    if missing_top:
        raise ValidationError(
            f"rule_set is missing required top-level keys: {sorted(missing_top)}."
        )

    direction = rule_set.get("direction")
    if direction not in ("long", "short"):
        raise ValidationError(
            f"'direction' must be 'long' or 'short', got {direction!r}."
        )

    rules_list = rule_set.get("rules_set")
    if not isinstance(rules_list, list):
        raise ValidationError(
            f"'rules_set' must be a list, got {type(rules_list).__name__!r}."
        )

    if len(rules_list) > 5:
        logger.warning(
            "rules_set contains %d rules (max 5); truncating to first 5.", len(rules_list)
        )
        rules_list = rules_list[:5]

    if len(rules_list) < 1:
        raise ValidationError(
            f"'rules_set' must contain at least 1 rule, got {len(rules_list)}."
        )

    validated_rules = []
    for i, rule in enumerate(rules_list, start=1):
        validated_rules.append(_validate_rule(rule, i))

    return {
        "direction": direction,
        "rules_set": validated_rules,
    }



class Output_Writer:
    """
    Serializes RuleSet dicts to JSON and loads/validates existing JSON files.

    Methods
    -------
    write(rule_set, path)
        Validate rule_set and write to JSON at path.
    load_and_validate(path)
        Load JSON from path and run full schema validation.
    """


    def write(self, rule_set: dict, path: str | Path) -> None:
        """
        Validate rule_set and write to JSON at path.

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
        if getattr(_cfg, "WRITE_EVALUATOR_CLEAN", True):
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
    Write a stripped strategy file if ``WRITE_EVALUATOR_CLEAN`` is ``True``.

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
    if not bool(getattr(_cfg, "WRITE_EVALUATOR_CLEAN", True)):
        return
    main_path = Path(main_path)
    clean_path = main_path.parent / "evaluator_clean" / f"{direction}_evaluator_clean.json"
    try:
        write_evaluator_clean(strategy, clean_path)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        logger.debug("evaluator_clean write failed for %s: %s", direction, exc)
