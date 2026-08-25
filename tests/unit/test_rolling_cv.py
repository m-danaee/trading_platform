"""Unit tests for the current fold geometry and count-gate helpers."""

from __future__ import annotations

import pandas as pd

from gpu_fuzzy_trader.mtf.cross_fitting import TemporalFold, eligible_for_role
from gpu_fuzzy_trader.validation.fold_gates import (
    FoldExposure,
    required_folds,
    scale_count_gate,
)


def test_required_folds_uses_configured_support_ratio() -> None:
    assert required_folds(2, 0.67) == 2
    assert required_folds(3, 0.67) == 3
    assert required_folds(4, 0.67) == 3


def test_scale_count_gate_uses_ceiling_and_absolute_floor() -> None:
    reference = FoldExposure(rows=100_000, duration_bars=0, per_symbol_rows={})
    quarter = FoldExposure(rows=25_000, duration_bars=0, per_symbol_rows={})

    assert scale_count_gate(40, quarter, reference, absolute_min=5) == 10
    assert scale_count_gate(45, quarter, reference, absolute_min=5) == 12
    assert scale_count_gate(1, quarter, reference, absolute_min=5) == 5


def test_role_eligibility_uses_fold_geometry() -> None:
    fold_one = TemporalFold(
        fold_id=1,
        train_start=pd.Timestamp("2024-01-01"),
        train_end=pd.Timestamp("2024-01-02"),
        test_start=pd.Timestamp("2024-01-03"),
        test_end=pd.Timestamp("2024-01-04"),
    )
    fold_two = TemporalFold(
        fold_id=2,
        train_start=pd.Timestamp("2024-01-01"),
        train_end=pd.Timestamp("2024-01-03"),
        test_start=pd.Timestamp("2024-01-04"),
        test_end=pd.Timestamp("2024-01-05"),
    )

    assert eligible_for_role(fold_one, "hwc") is True
    assert eligible_for_role(fold_one, "mwc") is False
    assert eligible_for_role(fold_two, "mwc") is True
