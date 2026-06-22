"""
Property-based tests for gpu_fuzzy_trader.output.writer.Output_Writer

Property 23: JSON Output Schema Validity
  **Validates: Requirements 12.1–12.9**

  For any valid rule set (direction, 2-5 rules, each with valid tp/sl/capital_pct
  and conditions), Output_Writer.write() must produce a JSON file that:
    1. Has exactly the keys "direction" and "rules_set"
    2. "direction" is "long" or "short"
    3. "rules_set" has 2-5 rules
    4. Each rule has exactly "tp", "sl", "capital_pct", "conditions"
    5. Each condition matches `[feature_name] IS Fuzzy Value Name` pattern
    6. The file can be loaded back with load_and_validate() without errors

  For any rule set with > PHASE3_GLOBAL_MAX_RULES rules, the output must be
  truncated to PHASE3_GLOBAL_MAX_RULES rules.

  For any rule set with all-zero tp/sl/capital_pct in any rule, write() must raise
  ValidationError.
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, HealthCheck
from hypothesis import strategies as st

from tests.property.hypothesis_config import prop_settings

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.backtest.symbol_conditions import parse_symbol_condition
from gpu_fuzzy_trader.output.writer import Output_Writer, ValidationError


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONDITION_RE = re.compile(r"^\[(.+?)\] IS (.+)$")


def _is_valid_exported_condition(condition: str) -> bool:
    if parse_symbol_condition(condition) is not None:
        return True
    match = _CONDITION_RE.match(condition)
    if match is None:
        return False
    return bool(match.group(1).strip()) and bool(match.group(2).strip())

# Valid fuzzy value names from the design document (all modes)
_FUZZY_VALUE_NAMES = [
    # binary
    "Inactive (0)", "Active (1)",
    # ternary
    "Negative (-1)", "Neutral (0)", "Positive (1)",
    # positive / sparse_positive
    "Very Low", "Low", "Medium", "High", "Very High",
    # sparse_signed
    "Strong Negative", "Weak Negative", "Exactly Zero", "Weak Positive", "Strong Positive",
    # signed
    "Extreme Bearish", "Strong Bearish", "Bearish", "Weak Bearish",
    "Neutral Negative", "Neutral Positive", "Weak Bullish", "Bullish",
    "Strong Bullish", "Extreme Bullish",
]

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Feature names: alphanumeric + underscores, non-empty
_feature_name_st = st.from_regex(r"[a-z][a-z0-9_]{0,19}", fullmatch=True)

# Fuzzy value name: pick from the known valid names
_fuzzy_value_name_st = st.sampled_from(_FUZZY_VALUE_NAMES)


@st.composite
def valid_condition_st(draw: st.DrawFn) -> str:
    """Generate a valid condition string: '[feature_name] IS Fuzzy Value Name'."""
    feature = draw(_feature_name_st)
    value = draw(_fuzzy_value_name_st)
    return f"[{feature}] IS {value}"


@st.composite
def valid_rule_st(draw: st.DrawFn) -> dict:
    """
    Generate a valid rule dict with:
      - tp, sl, capital_pct: floats where at least one is non-zero
      - conditions: 1-5 valid condition strings
    """
    # Generate tp/sl/capital_pct ensuring not all zero
    tp = draw(st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False))
    sl = draw(st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False))
    capital_pct = draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False))

    # Ensure at least one is non-zero (avoid the all-zero rejection case)
    if tp == 0.0 and sl == 0.0 and capital_pct == 0.0:
        tp = draw(st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False))

    n_conditions = draw(st.integers(min_value=1, max_value=5))
    conditions = draw(
        st.lists(valid_condition_st(), min_size=n_conditions, max_size=n_conditions)
    )

    return {
        "tp": tp,
        "sl": sl,
        "capital_pct": capital_pct,
        "conditions": conditions,
    }


@st.composite
def valid_rule_set_st(draw: st.DrawFn, min_rules: int = 2, max_rules: int = 5) -> dict:
    """
    Generate a valid rule_set dict with:
      - direction: "long" or "short"
      - rules_set: list of 2-5 valid rules
    """
    direction = draw(st.sampled_from(["long", "short"]))
    n_rules = draw(st.integers(min_value=min_rules, max_value=max_rules))
    rules = draw(st.lists(valid_rule_st(), min_size=n_rules, max_size=n_rules))
    return {"direction": direction, "rules_set": rules}


@st.composite
def oversized_rule_set_st(draw: st.DrawFn) -> dict:
    """
    Generate a rule_set with more than PHASE3_GLOBAL_MAX_RULES rules.
    Used to test truncation behaviour.
    """
    schema_max = int(_cfg.PHASE3_GLOBAL_MAX_RULES)
    direction = draw(st.sampled_from(["long", "short"]))
    n_rules = draw(st.integers(min_value=schema_max +
                   1, max_value=schema_max + 5))
    rules = draw(st.lists(valid_rule_st(), min_size=n_rules, max_size=n_rules))
    return {"direction": direction, "rules_set": rules}


@st.composite
def all_zero_rule_st(draw: st.DrawFn) -> dict:
    """Generate a rule where tp, sl, and capital_pct are all exactly zero."""
    n_conditions = draw(st.integers(min_value=1, max_value=3))
    conditions = draw(
        st.lists(valid_condition_st(), min_size=n_conditions, max_size=n_conditions)
    )
    return {"tp": 0.0, "sl": 0.0, "capital_pct": 0.0, "conditions": conditions}


@st.composite
def rule_set_with_all_zero_rule_st(draw: st.DrawFn) -> dict:
    """
    Generate a rule_set that contains at least one all-zero rule.
    The all-zero rule is inserted at a random position among otherwise valid rules.
    """
    direction = draw(st.sampled_from(["long", "short"]))
    # Total rules: 2-5, at least one is all-zero
    n_total = draw(st.integers(min_value=2, max_value=5))
    # Position of the all-zero rule
    zero_idx = draw(st.integers(min_value=0, max_value=n_total - 1))

    rules = []
    for i in range(n_total):
        if i == zero_idx:
            rules.append(draw(all_zero_rule_st()))
        else:
            rules.append(draw(valid_rule_st()))

    return {"direction": direction, "rules_set": rules}


# ---------------------------------------------------------------------------
# Property 23a: Valid rule sets produce correct JSON schema
# Validates: Requirements 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7
# ---------------------------------------------------------------------------

@given(rule_set=valid_rule_set_st())
@prop_settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_property_23a_valid_rule_set_schema(rule_set: dict) -> None:
    """
    **Property 23: JSON Output Schema Validity**
    **Validates: Requirements 12.1–12.9**

    For any valid rule set (direction ∈ {"long","short"}, 2-5 rules, each with
    valid tp/sl/capital_pct and conditions), Output_Writer.write() must produce
    a JSON file that:
      1. Has exactly the keys "direction" and "rules_set"
      2. "direction" is "long" or "short"
      3. "rules_set" has 2-5 rules
      4. Each rule has exactly "tp", "sl", "capital_pct", "conditions"
      5. Each condition matches [feature_name] IS Fuzzy Value Name pattern
      6. The file can be loaded back with load_and_validate() without errors
    """
    writer = Output_Writer()

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / f"{rule_set['direction']}.json"

        # write() must not raise for a valid rule set
        writer.write(rule_set, out_path)

        # The file must exist and be valid JSON
        assert out_path.exists(), "Output file was not created."
        with out_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        # Requirement 12.1: exactly "direction" and "rules_set" at top level
        assert set(data.keys()) == {"direction", "rules_set"}, (
            f"Top-level keys must be exactly {{'direction', 'rules_set'}}, "
            f"got {set(data.keys())}"
        )

        # Requirement 12.2: direction is "long" or "short"
        assert data["direction"] in ("long", "short"), (
            f"'direction' must be 'long' or 'short', got {data['direction']!r}"
        )

        # Requirement 12.3 / 12.8: rules_set has 2-5 rules
        rules = data["rules_set"]
        assert isinstance(rules, list), "'rules_set' must be a list."
        assert 2 <= len(rules) <= 5, (
            f"'rules_set' must have 2-5 rules, got {len(rules)}"
        )

        for i, rule in enumerate(rules):
            # Requirement 12.3: each rule has exactly the four required keys
            assert set(rule.keys()) == {"tp", "sl", "capital_pct", "conditions"}, (
                f"Rule {i}: keys must be exactly {{'tp','sl','capital_pct','conditions'}}, "
                f"got {set(rule.keys())}"
            )

            # Requirements 12.4, 12.5: tp, sl, capital_pct are floats
            assert isinstance(rule["tp"], (int, float)), (
                f"Rule {i}: 'tp' must be a number, got {type(rule['tp']).__name__}"
            )
            assert isinstance(rule["sl"], (int, float)), (
                f"Rule {i}: 'sl' must be a number, got {type(rule['sl']).__name__}"
            )
            assert isinstance(rule["capital_pct"], (int, float)), (
                f"Rule {i}: 'capital_pct' must be a number, got {type(rule['capital_pct']).__name__}"
            )

            # Requirement 12.9: not all three zero
            assert not (rule["tp"] == 0.0 and rule["sl"] == 0.0 and rule["capital_pct"] == 0.0), (
                f"Rule {i}: all of tp/sl/capital_pct are zero — should have been rejected."
            )

            # Requirement 12.6: conditions is a non-empty list of strings
            conditions = rule["conditions"]
            assert isinstance(conditions, list) and len(conditions) > 0, (
                f"Rule {i}: 'conditions' must be a non-empty list."
            )

            # Requirement 12.6 / 12.7: feature or symbol filter pattern
            for j, cond in enumerate(conditions):
                assert isinstance(cond, str), (
                    f"Rule {i}, condition {j}: must be a string, got {type(cond).__name__}"
                )
                assert _is_valid_exported_condition(cond), (
                    f"Rule {i}, condition {j}: {cond!r} is not a valid feature or "
                    "symbol filter condition."
                )
                if parse_symbol_condition(cond) is not None:
                    continue
                m = _CONDITION_RE.match(cond)
                feature_name = m.group(1).strip()
                value_name = m.group(2).strip()
                assert feature_name, (
                    f"Rule {i}, condition {j}: feature name is empty in {cond!r}"
                )
                assert value_name, (
                    f"Rule {i}, condition {j}: fuzzy value name is empty in {cond!r}"
                )

        # Requirement 12.6 (round-trip): load_and_validate() must succeed
        loaded = writer.load_and_validate(out_path)
        assert loaded["direction"] == data["direction"], (
            "Round-trip direction mismatch."
        )
        assert len(loaded["rules_set"]) == len(rules), (
            "Round-trip rules_set length mismatch."
        )


# ---------------------------------------------------------------------------
# Property 23b: Oversized rule sets are truncated to PHASE3_GLOBAL_MAX_RULES
# Validates: Requirement 12.8
# ---------------------------------------------------------------------------

@given(rule_set=oversized_rule_set_st())
@prop_settings(
    max_examples=30,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_property_23b_oversized_rule_set_truncated_to_global_max(rule_set: dict) -> None:
    """
    **Property 23: JSON Output Schema Validity**
    **Validates: Requirements 12.8**

    For any rule set with > PHASE3_GLOBAL_MAX_RULES rules, Output_Writer.write()
    must truncate the output to exactly PHASE3_GLOBAL_MAX_RULES rules (first in order).
    """
    schema_max = int(_cfg.PHASE3_GLOBAL_MAX_RULES)
    writer = Output_Writer()
    original_rules = rule_set["rules_set"]
    assert len(original_rules) > schema_max, (
        f"Precondition: input must have > {schema_max} rules."
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / f"{rule_set['direction']}.json"

        # write() must not raise — truncation is a warning, not an error
        writer.write(rule_set, out_path)

        with out_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        assert len(data["rules_set"]) == schema_max, (
            f"Expected exactly {schema_max} rules after truncation, "
            f"got {len(data['rules_set'])}"
        )

        for i in range(schema_max):
            assert float(data["rules_set"][i]["tp"]) == float(original_rules[i]["tp"]), (
                f"Truncated rule {i}: tp mismatch."
            )
            assert float(data["rules_set"][i]["sl"]) == float(original_rules[i]["sl"]), (
                f"Truncated rule {i}: sl mismatch."
            )
            assert float(data["rules_set"][i]["capital_pct"]) == float(
                original_rules[i]["capital_pct"]
            ), (
                f"Truncated rule {i}: capital_pct mismatch."
            )

        loaded = writer.load_and_validate(out_path)
        assert len(loaded["rules_set"]) == schema_max


# ---------------------------------------------------------------------------
# Property 23c: All-zero tp/sl/capital_pct raises ValidationError
# Validates: Requirement 12.9
# ---------------------------------------------------------------------------

@given(rule_set=rule_set_with_all_zero_rule_st())
@prop_settings(
    max_examples=30,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_property_23c_all_zero_risk_raises_validation_error(rule_set: dict) -> None:
    """
    **Property 23: JSON Output Schema Validity**
    **Validates: Requirements 12.9**

    For any rule set that contains a rule where tp, sl, and capital_pct are
    all zero, Output_Writer.write() must raise ValidationError.
    """
    writer = Output_Writer()

    # Verify the precondition: at least one rule has all-zero risk params
    has_all_zero = any(
        r["tp"] == 0.0 and r["sl"] == 0.0 and r["capital_pct"] == 0.0
        for r in rule_set["rules_set"]
    )
    assert has_all_zero, "Precondition: rule_set must contain at least one all-zero rule."

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / f"{rule_set['direction']}.json"

        with pytest.raises(ValidationError):
            writer.write(rule_set, out_path)
