"""
Unit tests for ``gpu_fuzzy_trader.output.writer.write_evaluator_clean``.

Tests cover:
  - Extra top-level keys are stripped; only ``direction`` and ``rules_set`` remain.
  - Parent directory is created when it does not exist.
  - Strategy with no extra keys (only ``direction`` + ``rules_set``) works.
  - ``KeyError`` is raised when ``direction`` or ``rules_set`` is missing.
  - Wiring into ``Output_Writer.write`` also produces the clean file.
  - The clean file is NOT written when ``WRITE_EVALUATOR_CLEAN`` is ``False``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.output.writer import (
    Output_Writer,
    ValidationError,
    write_evaluator_clean,
)

# ---------------------------------------------------------------------------
# Fixtures
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


@pytest.fixture
def minimal_strategy() -> dict:
    """Strategy with only the two required keys (no extras)."""
    return {
        "direction": "long",
        "rules_set": [
            _make_rule(tp=3.0, sl=1.5, capital_pct=30.0),
            _make_rule(tp=5.0, sl=2.0, capital_pct=20.0),
        ],
    }


@pytest.fixture
def strategy_with_extras() -> dict:
    """Strategy with extra metadata keys that should be stripped."""
    return {
        "direction": "short",
        "rules_set": [
            _make_rule(
                tp=4.0,
                sl=2.0,
                capital_pct=50.0,
                conditions=["[vol] IS High", "[atr] IS Rising"],
            ),
            _make_rule(
                tp=3.0,
                sl=1.5,
                capital_pct=25.0,
                conditions=["[trend] IS Falling", "[rsi] IS Oversold"],
            ),
        ],
        "risk_optimized": True,
        "deployment_accepted": True,
        "validation_gate": {
            "return_pct": 1.23,
            "profit_factor": 2.5,
        },
        "extra_key_that_should_not_appear": "some_value",
    }


# ---------------------------------------------------------------------------
# Test: standalone write_evaluator_clean
# ---------------------------------------------------------------------------


class TestWriteEvaluatorCleanStandalone:
    """Tests for the standalone ``write_evaluator_clean`` function."""

    def test_strips_extra_keys(self, tmp_path: Path, strategy_with_extras: dict) -> None:
        """Extra top-level keys are stripped; only direction and rules_set remain."""
        output_path = tmp_path / "short_evaluator_clean.json"
        write_evaluator_clean(strategy_with_extras, output_path)

        assert output_path.exists()
        with output_path.open("r") as fh:
            data = json.load(fh)

        assert set(data.keys()) == {"direction", "rules_set"}
        assert data["direction"] == "short"
        assert len(data["rules_set"]) == 2
        assert "risk_optimized" not in data
        assert "deployment_accepted" not in data
        assert "validation_gate" not in data
        assert "extra_key_that_should_not_appear" not in data

    def test_creates_parent_directory(self, tmp_path: Path, minimal_strategy: dict) -> None:
        """Parent directory is created when it does not exist."""
        deep_path = tmp_path / "a" / "b" / "c" / "long_evaluator_clean.json"
        assert not deep_path.parent.exists()

        write_evaluator_clean(minimal_strategy, deep_path)

        assert deep_path.parent.exists()
        assert deep_path.exists()
        with deep_path.open("r") as fh:
            data = json.load(fh)
        assert data["direction"] == "long"
        assert len(data["rules_set"]) == 2

    def test_no_extra_keys_ok(self, tmp_path: Path, minimal_strategy: dict) -> None:
        """Strategy with no extra keys works and produces correct output."""
        output_path = tmp_path / "long_evaluator_clean.json"
        write_evaluator_clean(minimal_strategy, output_path)

        assert output_path.exists()
        with output_path.open("r") as fh:
            data = json.load(fh)

        assert set(data.keys()) == {"direction", "rules_set"}
        assert data["direction"] == "long"
        assert len(data["rules_set"]) == 2
        # Spot-check one rule
        assert data["rules_set"][0]["tp"] == 3.0
        assert data["rules_set"][0]["sl"] == 1.5

    def test_missing_direction_raises(self, tmp_path: Path) -> None:
        """Missing 'direction' raises KeyError."""
        bad = {"rules_set": []}
        output_path = tmp_path / "bad.json"
        with pytest.raises(KeyError, match="direction"):
            write_evaluator_clean(bad, output_path)

    def test_missing_rules_set_raises(self, tmp_path: Path) -> None:
        """Missing 'rules_set' raises KeyError."""
        bad = {"direction": "long"}
        output_path = tmp_path / "bad.json"
        with pytest.raises(KeyError, match="rules_set"):
            write_evaluator_clean(bad, output_path)

    def test_returns_none(self, tmp_path: Path, minimal_strategy: dict) -> None:
        """Function returns None (no return value)."""
        output_path = tmp_path / "none_test.json"
        result = write_evaluator_clean(minimal_strategy, output_path)
        assert result is None


# ---------------------------------------------------------------------------
# Test: wiring into Output_Writer.write
# ---------------------------------------------------------------------------


class TestWriteEvaluatorCleanWired:
    """Tests that ``Output_Writer.write`` also produces the evaluator-clean file."""

    def test_clean_file_written_after_main(self, tmp_path: Path, strategy_with_extras: dict) -> None:
        """
        Output_Writer.write produces both the main file and the
        evaluator-clean file.
        """
        original_value = getattr(_cfg, "WRITE_EVALUATOR_CLEAN", True)
        try:
            # Ensure the flag is True for this test
            _cfg.WRITE_EVALUATOR_CLEAN = True

            main_path = tmp_path / "short.json"
            writer = Output_Writer()
            writer.write(strategy_with_extras, main_path)

            # Main file exists
            assert main_path.exists()

            # Clean file exists in evaluator_clean subdirectory
            clean_path = tmp_path / "evaluator_clean" / "short_evaluator_clean.json"
            assert clean_path.exists(), f"Expected clean file at {clean_path}"

            # Clean file has only direction and rules_set
            with clean_path.open("r") as fh:
                clean_data = json.load(fh)
            assert set(clean_data.keys()) == {"direction", "rules_set"}
            assert clean_data["direction"] == "short"

            # Main file retains extra keys (since Output_Writer.write returns
            # the validated dict which only has direction + rules_set; extra
            # keys in input are stripped by _validate_rule_set).
            with main_path.open("r") as fh:
                main_data = json.load(fh)
            assert set(main_data.keys()) == {"direction", "rules_set"}
        finally:
            _cfg.WRITE_EVALUATOR_CLEAN = original_value

    def test_clean_file_not_written_when_flag_false(
        self, tmp_path: Path, strategy_with_extras: dict
    ) -> None:
        """
        When ``WRITE_EVALUATOR_CLEAN`` is ``False``, the clean file is NOT
        written.
        """
        original_value = getattr(_cfg, "WRITE_EVALUATOR_CLEAN", True)
        try:
            _cfg.WRITE_EVALUATOR_CLEAN = False

            main_path = tmp_path / "short.json"
            writer = Output_Writer()
            writer.write(strategy_with_extras, main_path)

            clean_path = tmp_path / "evaluator_clean" / "short_evaluator_clean.json"
            assert not clean_path.exists(), (
                f"Clean file should NOT exist when WRITE_EVALUATOR_CLEAN=False: "
                f"{clean_path}"
            )
        finally:
            _cfg.WRITE_EVALUATOR_CLEAN = original_value

    def test_clean_file_written_for_long(
        self, tmp_path: Path, minimal_strategy: dict
    ) -> None:
        """Long direction also produces the correct clean file."""
        original_value = getattr(_cfg, "WRITE_EVALUATOR_CLEAN", True)
        try:
            _cfg.WRITE_EVALUATOR_CLEAN = True

            main_path = tmp_path / "long.json"
            writer = Output_Writer()
            writer.write(minimal_strategy, main_path)

            clean_path = tmp_path / "evaluator_clean" / "long_evaluator_clean.json"
            assert clean_path.exists()
            with clean_path.open("r") as fh:
                data = json.load(fh)
            assert data["direction"] == "long"
            assert len(data["rules_set"]) == 2
        finally:
            _cfg.WRITE_EVALUATOR_CLEAN = original_value
