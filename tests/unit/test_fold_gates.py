"""Regression tests for fold-aware count gates and fixed quality gates."""

from __future__ import annotations

import pytest

from gpu_fuzzy_trader.validation.fold_gates import (
    FoldExposure,
    required_folds,
    scale_count_gate,
)


COUNT_GATES = {
    "MinTrades",
    "MinSignals",
    "MinSupport",
    "MinCandidateSupport",
    "min_trades_per_symbol",
}


def _exposure(rows: int) -> FoldExposure:
    return FoldExposure(rows=rows, duration_bars=0, per_symbol_rows={})


def _scale_gates(
    base_gates: dict[str, float | int],
    effective_rows: int,
    reference_rows: int = 100_000,
    absolute_min: int = 5,
) -> dict[str, float | int]:
    return {
        name: (
            scale_count_gate(
                int(value),
                _exposure(effective_rows),
                _exposure(reference_rows),
                absolute_min,
            )
            if name in COUNT_GATES
            else value
        )
        for name, value in base_gates.items()
    }


@pytest.mark.parametrize(
    ("effective_rows", "expected"),
    [
        (100_000, 40),
        (50_000, 20),
        (25_000, 10),
        (5_000, 5),
    ],
)
def test_count_gate_scales_by_effective_exposure(
    effective_rows: int,
    expected: int,
):
    assert scale_count_gate(
        40,
        _exposure(effective_rows),
        _exposure(100_000),
        5,
    ) == expected


def test_quality_gates_do_not_change_when_fold_size_shrinks():
    base_gates = {
        "MinTrades": 40,
        "MinSignals": 40,
        "MinSupport": 20,
        "MinCandidateSupport": 10,
        "min_trades_per_symbol": 5,
        "PF": 1.20,
        "MCC": 0.10,
        "MDD": 0.25,
    }

    large = _scale_gates(base_gates, 100_000)
    small = _scale_gates(base_gates, 25_000)

    assert {name: large[name] for name in ("PF", "MCC", "MDD")} == {
        "PF": 1.20,
        "MCC": 0.10,
        "MDD": 0.25,
    }
    assert {name: small[name] for name in ("PF", "MCC", "MDD")} == {
        "PF": 1.20,
        "MCC": 0.10,
        "MDD": 0.25,
    }
    assert small["MinTrades"] == 10
    assert large["MinTrades"] == 40


def _resolve_stage_gate(
    base: int,
    stage: str,
    exposure: dict[str, int],
    reference_rows: int = 100_000,
) -> int:
    """Use train exposure for train gates and OOF exposure for OOF gates."""
    rows = exposure["train_rows"] if stage == "train" else exposure["oof_rows"]
    return scale_count_gate(base, _exposure(rows), _exposure(reference_rows))


@pytest.mark.parametrize(
    ("stage", "expected"),
    [("train", 40), ("oof", 10)],
)
def test_gate_stage_selects_its_own_exposure(stage: str, expected: int):
    exposure = {"train_rows": 100_000, "oof_rows": 25_000}

    assert _resolve_stage_gate(40, stage, exposure) == expected


def test_required_folds_uses_the_production_helper():
    assert required_folds(3, 0.67) == 3
