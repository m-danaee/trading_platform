"""
Unit tests for gpu_fuzzy_trader.output.writer — Output_Writer

Tests cover:
  - write(): valid rule sets are serialised to JSON correctly
  - write(): schema enforcement (direction, rules_set size, rule fields,
             condition format, all-zero tp/sl/capital_pct)
  - load_and_validate(): round-trip correctness
  - load_and_validate(): error cases (missing file, bad JSON, schema violations)
  - ValidationError is raised (not a generic exception) on all violations
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.output.writer import Output_Writer, ValidationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rule(
    tp: float = 4.0,
    sl: float = 2.0,
    capital_pct: float = 50.0,
    conditions: list[str] | None = None,
) -> dict:
    if conditions is None:
        conditions = ["[feature_a] IS Bearish", "[feature_b] IS Very High"]
    return {"tp": tp, "sl": sl, "capital_pct": capital_pct, "conditions": conditions}


def _make_rule_set(
    direction: str = "long",
    n_rules: int = 2,
    rule_override: dict | None = None,
) -> dict:
    rule = rule_override if rule_override is not None else _make_rule()
    return {"direction": direction, "rules_set": [rule] * n_rules}


def _write_and_reload(rule_set: dict) -> dict:
    """Write rule_set to a temp file and reload the raw JSON."""
    writer = Output_Writer()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = f.name
    try:
        writer.write(rule_set, tmp_path)
        with open(tmp_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Tests: write() — happy path
# ---------------------------------------------------------------------------

class TestWriteHappyPath:
    def test_write_creates_file(self):
        writer = Output_Writer()
        rule_set = _make_rule_set()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name
        try:
            writer.write(rule_set, tmp_path)
            assert Path(tmp_path).exists()
        finally:
            os.unlink(tmp_path)

    def test_write_produces_valid_json(self):
        data = _write_and_reload(_make_rule_set())
        assert isinstance(data, dict)

    def test_write_direction_preserved(self):
        for direction in ("long", "short"):
            data = _write_and_reload(_make_rule_set(direction=direction))
            assert data["direction"] == direction

    def test_write_rules_set_key_present(self):
        data = _write_and_reload(_make_rule_set())
        assert "rules_set" in data

    def test_write_rule_fields_preserved(self):
        rule = _make_rule(tp=3.5, sl=1.5, capital_pct=25.0,
                          conditions=["[feat_x] IS Bullish"])
        data = _write_and_reload({"direction": "long", "rules_set": [rule, rule]})
        r = data["rules_set"][0]
        assert r["tp"] == 3.5
        assert r["sl"] == 1.5
        assert r["capital_pct"] == 25.0
        assert r["conditions"] == ["[feat_x] IS Bullish"]

    def test_write_two_rules(self):
        data = _write_and_reload(_make_rule_set(n_rules=2))
        assert len(data["rules_set"]) == 2

    def test_write_five_rules(self):
        data = _write_and_reload(_make_rule_set(n_rules=5))
        assert len(data["rules_set"]) == 5

    def test_write_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "a" / "b" / "out.json"
            writer = Output_Writer()
            writer.write(_make_rule_set(), nested)
            assert nested.exists()

    def test_write_accepts_path_object(self):
        writer = Output_Writer()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = Path(f.name)
        try:
            writer.write(_make_rule_set(), tmp_path)
            assert tmp_path.exists()
        finally:
            tmp_path.unlink()

    def test_write_accepts_string_path(self):
        writer = Output_Writer()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name
        try:
            writer.write(_make_rule_set(), tmp_path)
            assert Path(tmp_path).exists()
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Tests: write() — truncation of > global_max rules
# ---------------------------------------------------------------------------

class TestWriteTruncation:
    def test_exceeds_max_truncated(self):
        max_rules = 5
        rule_set = _make_rule_set(n_rules=max_rules + 1)
        data = _write_and_reload(rule_set)
        assert len(data["rules_set"]) == max_rules

    def test_truncation_keeps_first_max_rules(self):
        max_rules = 5
        n_rules = max_rules + 3
        rules = [
            _make_rule(tp=float(i), sl=1.0, capital_pct=10.0,
                       conditions=["[f] IS Bullish", "[g] IS Very High"])
            for i in range(1, n_rules + 1)
        ]
        rule_set = {"direction": "long", "rules_set": rules}
        data = _write_and_reload(rule_set)
        assert len(data["rules_set"]) == max_rules
        for i, r in enumerate(data["rules_set"], start=1):
            assert r["tp"] == float(i)


# ---------------------------------------------------------------------------
# Tests: write() — direction validation
# ---------------------------------------------------------------------------

class TestWriteDirectionValidation:
    def test_invalid_direction_raises_validation_error(self):
        rule_set = _make_rule_set()
        rule_set["direction"] = "neutral"
        writer = Output_Writer()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name
        try:
            with pytest.raises(ValidationError):
                writer.write(rule_set, tmp_path)
        finally:
            if Path(tmp_path).exists():
                os.unlink(tmp_path)

    def test_uppercase_direction_raises_validation_error(self):
        rule_set = _make_rule_set()
        rule_set["direction"] = "Long"
        writer = Output_Writer()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name
        try:
            with pytest.raises(ValidationError):
                writer.write(rule_set, tmp_path)
        finally:
            if Path(tmp_path).exists():
                os.unlink(tmp_path)

    def test_missing_direction_raises_validation_error(self):
        rule_set = {"rules_set": [_make_rule(), _make_rule()]}
        writer = Output_Writer()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name
        try:
            with pytest.raises(ValidationError):
                writer.write(rule_set, tmp_path)
        finally:
            if Path(tmp_path).exists():
                os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Tests: write() — rules_set size validation
# ---------------------------------------------------------------------------

class TestWriteRulesSetSize:
    def test_zero_rules_raises_validation_error(self):
        rule_set = {"direction": "long", "rules_set": []}
        writer = Output_Writer()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name
        try:
            with pytest.raises(ValidationError):
                writer.write(rule_set, tmp_path)
        finally:
            if Path(tmp_path).exists():
                os.unlink(tmp_path)

    def test_one_rule_raises_validation_error(self):
        rule_set = {"direction": "long", "rules_set": [_make_rule()]}
        writer = Output_Writer()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name
        try:
            with pytest.raises(ValidationError):
                writer.write(rule_set, tmp_path)
        finally:
            if Path(tmp_path).exists():
                os.unlink(tmp_path)

    def test_missing_rules_set_raises_validation_error(self):
        rule_set = {"direction": "long"}
        writer = Output_Writer()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name
        try:
            with pytest.raises(ValidationError):
                writer.write(rule_set, tmp_path)
        finally:
            if Path(tmp_path).exists():
                os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Tests: write() — all-zero tp/sl/capital_pct rejection
# ---------------------------------------------------------------------------

class TestWriteAllZeroRejection:
    def test_all_zero_rule_raises_validation_error(self):
        bad_rule = _make_rule(tp=0.0, sl=0.0, capital_pct=0.0)
        rule_set = {"direction": "long", "rules_set": [bad_rule, _make_rule()]}
        writer = Output_Writer()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name
        try:
            with pytest.raises(ValidationError):
                writer.write(rule_set, tmp_path)
        finally:
            if Path(tmp_path).exists():
                os.unlink(tmp_path)

    def test_nonzero_tp_only_is_accepted(self):
        """A rule with only tp non-zero should be accepted."""
        rule = _make_rule(tp=1.0, sl=0.0, capital_pct=0.0)
        rule_set = {"direction": "long", "rules_set": [rule, rule]}
        data = _write_and_reload(rule_set)
        assert len(data["rules_set"]) == 2

    def test_nonzero_sl_only_is_accepted(self):
        rule = _make_rule(tp=0.0, sl=1.0, capital_pct=0.0)
        rule_set = {"direction": "long", "rules_set": [rule, rule]}
        data = _write_and_reload(rule_set)
        assert len(data["rules_set"]) == 2

    def test_nonzero_capital_pct_only_is_accepted(self):
        rule = _make_rule(tp=0.0, sl=0.0, capital_pct=50.0)
        rule_set = {"direction": "long", "rules_set": [rule, rule]}
        data = _write_and_reload(rule_set)
        assert len(data["rules_set"]) == 2


# ---------------------------------------------------------------------------
# Tests: write() — condition string validation
# ---------------------------------------------------------------------------

class TestWriteSymbolConditionValidation:
    def test_symbol_is_format_accepted(self):
        rule = _make_rule(
            conditions=["symbol is 1", "[feat_x] IS Bullish", "[feat_y] IS High"]
        )
        rule_set = {"direction": "long", "rules_set": [rule, rule]}
        data = _write_and_reload(rule_set)
        assert data["rules_set"][0]["conditions"][0] == "symbol is 1"

    def test_bracket_symbol_format_accepted(self):
        rule = _make_rule(
            conditions=["[symbol] IS 2", "[feat_x] IS Bullish", "[feat_y] IS High"]
        )
        rule_set = {"direction": "long", "rules_set": [rule, rule]}
        data = _write_and_reload(rule_set)
        assert data["rules_set"][0]["conditions"][0] == "[symbol] IS 2"

    def test_comma_separated_symbol_list_accepted(self):
        rule = _make_rule(
            conditions=["symbol is 1,2", "[feat_x] IS Bullish", "[feat_y] IS High"]
        )
        rule_set = {"direction": "long", "rules_set": [rule, rule]}
        data = _write_and_reload(rule_set)
        assert data["rules_set"][0]["conditions"][0] == "symbol is 1,2"

    def test_empty_symbol_value_raises_validation_error(self):
        bad_rule = _make_rule(conditions=["symbol is ", "[f] IS High", "[g] IS Low"])
        rule_set = {"direction": "long", "rules_set": [bad_rule, _make_rule()]}
        writer = Output_Writer()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name
        try:
            with pytest.raises(ValidationError):
                writer.write(rule_set, tmp_path)
        finally:
            if Path(tmp_path).exists():
                os.unlink(tmp_path)


class TestWriteConditionValidation:
    def test_valid_condition_accepted(self):
        rule = _make_rule(conditions=["[feature_a] IS Bearish"])
        rule_set = {"direction": "long", "rules_set": [rule, rule]}
        data = _write_and_reload(rule_set)
        assert data["rules_set"][0]["conditions"] == ["[feature_a] IS Bearish"]

    def test_condition_without_brackets_raises_validation_error(self):
        bad_rule = _make_rule(conditions=["feature_a IS Bearish", "[f] IS High"])
        rule_set = {"direction": "long", "rules_set": [bad_rule, _make_rule()]}
        writer = Output_Writer()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name
        try:
            with pytest.raises(ValidationError):
                writer.write(rule_set, tmp_path)
        finally:
            if Path(tmp_path).exists():
                os.unlink(tmp_path)

    def test_condition_without_is_raises_validation_error(self):
        bad_rule = _make_rule(conditions=["[feature_a] Bearish", "[f] IS High"])
        rule_set = {"direction": "long", "rules_set": [bad_rule, _make_rule()]}
        writer = Output_Writer()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name
        try:
            with pytest.raises(ValidationError):
                writer.write(rule_set, tmp_path)
        finally:
            if Path(tmp_path).exists():
                os.unlink(tmp_path)

    def test_empty_conditions_list_raises_validation_error(self):
        bad_rule = _make_rule(conditions=[])
        rule_set = {"direction": "long", "rules_set": [bad_rule, _make_rule()]}
        writer = Output_Writer()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name
        try:
            with pytest.raises(ValidationError):
                writer.write(rule_set, tmp_path)
        finally:
            if Path(tmp_path).exists():
                os.unlink(tmp_path)

    def test_condition_with_empty_feature_name_raises_validation_error(self):
        bad_rule = _make_rule(conditions=["[] IS Bearish", "[f] IS High"])
        rule_set = {"direction": "long", "rules_set": [bad_rule, _make_rule()]}
        writer = Output_Writer()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name
        try:
            with pytest.raises(ValidationError):
                writer.write(rule_set, tmp_path)
        finally:
            if Path(tmp_path).exists():
                os.unlink(tmp_path)

    def test_condition_with_empty_value_name_raises_validation_error(self):
        bad_rule = _make_rule(conditions=["[feature_a] IS ", "[f] IS High"])
        rule_set = {"direction": "long", "rules_set": [bad_rule, _make_rule()]}
        writer = Output_Writer()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name
        try:
            with pytest.raises(ValidationError):
                writer.write(rule_set, tmp_path)
        finally:
            if Path(tmp_path).exists():
                os.unlink(tmp_path)

    def test_multiple_conditions_per_rule_accepted(self):
        rule = _make_rule(conditions=[
            "[feature_a] IS Bearish",
            "[feature_b] IS Very High",
            "[feature_c] IS Neutral (0)",
        ])
        rule_set = {"direction": "short", "rules_set": [rule, rule]}
        data = _write_and_reload(rule_set)
        assert len(data["rules_set"][0]["conditions"]) == 3

    def test_all_fuzzy_value_names_accepted(self):
        """Spot-check a variety of valid fuzzy value names."""
        value_names = [
            "Inactive (0)", "Active (1)",
            "Negative (-1)", "Neutral (0)", "Positive (1)",
            "Very Low", "Low", "Medium", "High", "Very High",
            "Strong Negative", "Weak Negative", "Exactly Zero",
            "Weak Positive", "Strong Positive",
            "Extreme Bearish", "Strong Bearish", "Bearish",
            "Weak Bearish", "Neutral Negative", "Neutral Positive",
            "Weak Bullish", "Bullish", "Strong Bullish", "Extreme Bullish",
        ]
        for vn in value_names:
            rule = _make_rule(conditions=[f"[feat] IS {vn}", "[g] IS High"])
            rule_set = {"direction": "long", "rules_set": [rule, rule]}
            # Should not raise
            data = _write_and_reload(rule_set)
            assert data["rules_set"][0]["conditions"][0] == f"[feat] IS {vn}"


# ---------------------------------------------------------------------------
# Tests: write() — rule field type coercion
# ---------------------------------------------------------------------------

class TestWriteFieldCoercion:
    def test_integer_tp_sl_capital_pct_coerced_to_float(self):
        rule = {"tp": 4, "sl": 2, "capital_pct": 50,
                "conditions": ["[f] IS Bullish", "[g] IS High"]}
        rule_set = {"direction": "long", "rules_set": [rule, rule]}
        data = _write_and_reload(rule_set)
        r = data["rules_set"][0]
        assert isinstance(r["tp"], float)
        assert isinstance(r["sl"], float)
        assert isinstance(r["capital_pct"], float)

    def test_missing_rule_key_raises_validation_error(self):
        bad_rule = {"tp": 4.0, "sl": 2.0, "conditions": ["[f] IS High", "[g] IS Low"]}
        # missing capital_pct
        rule_set = {"direction": "long", "rules_set": [bad_rule, _make_rule()]}
        writer = Output_Writer()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name
        try:
            with pytest.raises(ValidationError):
                writer.write(rule_set, tmp_path)
        finally:
            if Path(tmp_path).exists():
                os.unlink(tmp_path)

    def test_non_dict_rule_raises_validation_error(self):
        rule_set = {"direction": "long", "rules_set": ["not_a_dict", _make_rule()]}
        writer = Output_Writer()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name
        try:
            with pytest.raises(ValidationError):
                writer.write(rule_set, tmp_path)
        finally:
            if Path(tmp_path).exists():
                os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Tests: load_and_validate() — happy path
# ---------------------------------------------------------------------------

class TestLoadAndValidateHappyPath:
    def _write_json(self, data: dict) -> str:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f)
            return f.name

    def test_round_trip_preserves_direction(self):
        rule_set = _make_rule_set(direction="short")
        writer = Output_Writer()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name
        try:
            writer.write(rule_set, tmp_path)
            loaded = writer.load_and_validate(tmp_path)
            assert loaded["direction"] == "short"
        finally:
            os.unlink(tmp_path)

    def test_round_trip_preserves_rules(self):
        rule = _make_rule(tp=2.5, sl=1.2, capital_pct=15.0,
                          conditions=["[dmi] IS Bearish", "[vol] IS Very Low"])
        rule_set = {"direction": "long", "rules_set": [rule, rule]}
        writer = Output_Writer()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name
        try:
            writer.write(rule_set, tmp_path)
            loaded = writer.load_and_validate(tmp_path)
            r = loaded["rules_set"][0]
            assert r["tp"] == 2.5
            assert r["sl"] == 1.2
            assert r["capital_pct"] == 15.0
            assert r["conditions"] == ["[dmi] IS Bearish", "[vol] IS Very Low"]
        finally:
            os.unlink(tmp_path)

    def test_load_returns_dict(self):
        rule_set = _make_rule_set()
        writer = Output_Writer()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name
        try:
            writer.write(rule_set, tmp_path)
            loaded = writer.load_and_validate(tmp_path)
            assert isinstance(loaded, dict)
        finally:
            os.unlink(tmp_path)

    def test_load_accepts_path_object(self):
        rule_set = _make_rule_set()
        writer = Output_Writer()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = Path(f.name)
        try:
            writer.write(rule_set, tmp_path)
            loaded = writer.load_and_validate(tmp_path)
            assert "direction" in loaded
        finally:
            tmp_path.unlink()


# ---------------------------------------------------------------------------
# Tests: load_and_validate() — error cases
# ---------------------------------------------------------------------------

class TestLoadAndValidateErrors:
    def test_missing_file_raises_validation_error(self):
        writer = Output_Writer()
        with pytest.raises(ValidationError, match="not found"):
            writer.load_and_validate("/nonexistent/path/file.json")

    def test_invalid_json_raises_validation_error(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            f.write("{ this is not valid json }")
            tmp_path = f.name
        try:
            writer = Output_Writer()
            with pytest.raises(ValidationError):
                writer.load_and_validate(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_schema_violation_in_loaded_file_raises_validation_error(self):
        bad_data = {
            "direction": "long",
            "rules_set": [
                {"tp": 0.0, "sl": 0.0, "capital_pct": 0.0,
                 "conditions": ["[f] IS High", "[g] IS Low"]}
            ]
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(bad_data, f)
            tmp_path = f.name
        try:
            writer = Output_Writer()
            with pytest.raises(ValidationError):
                writer.load_and_validate(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_wrong_direction_in_file_raises_validation_error(self):
        bad_data = {
            "direction": "sideways",
            "rules_set": [_make_rule(), _make_rule()]
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(bad_data, f)
            tmp_path = f.name
        try:
            writer = Output_Writer()
            with pytest.raises(ValidationError):
                writer.load_and_validate(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_too_few_rules_in_file_raises_validation_error(self):
        bad_data = {"direction": "long", "rules_set": [_make_rule()]}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(bad_data, f)
            tmp_path = f.name
        try:
            writer = Output_Writer()
            with pytest.raises(ValidationError):
                writer.load_and_validate(tmp_path)
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Tests: ValidationError is the correct exception type
# ---------------------------------------------------------------------------

class TestValidationErrorType:
    def test_validation_error_is_exception_subclass(self):
        assert issubclass(ValidationError, Exception)

    def test_write_raises_validation_error_not_generic(self):
        rule_set = {"direction": "bad", "rules_set": [_make_rule(), _make_rule()]}
        writer = Output_Writer()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name
        try:
            with pytest.raises(ValidationError):
                writer.write(rule_set, tmp_path)
        finally:
            if Path(tmp_path).exists():
                os.unlink(tmp_path)

    def test_load_raises_validation_error_not_generic(self):
        writer = Output_Writer()
        with pytest.raises(ValidationError):
            writer.load_and_validate("/no/such/file.json")


# ---------------------------------------------------------------------------
# Tests: exact schema from spec example
# ---------------------------------------------------------------------------

class TestSpecExample:
    """Verify the exact example from the spec works end-to-end."""

    def test_spec_example_round_trip(self):
        rule_set = {
            "direction": "long",
            "rules_set": [
                {
                    "tp": 4.0,
                    "sl": 2.0,
                    "capital_pct": 50.0,
                    "conditions": [
                        "[feature_a] IS Bearish",
                        "[feature_b] IS Very High",
                    ],
                },
                {
                    "tp": 3.1,
                    "sl": 1.8,
                    "capital_pct": 12.0,
                    "conditions": [
                        "[amihud_illiquidity_20] IS Very High",
                        "[macd_hist_atr] IS Extreme Bearish",
                        "[return_skew_30] IS Strong Bearish",
                    ],
                },
            ],
        }
        writer = Output_Writer()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name
        try:
            writer.write(rule_set, tmp_path)
            loaded = writer.load_and_validate(tmp_path)
            assert loaded["direction"] == "long"
            assert len(loaded["rules_set"]) == 2
            assert loaded["rules_set"][0]["tp"] == 4.0
            assert loaded["rules_set"][1]["conditions"][0] == "[amihud_illiquidity_20] IS Very High"
        finally:
            os.unlink(tmp_path)
