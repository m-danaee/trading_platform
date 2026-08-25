"""Tests for fold-aware trade-count scaling helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_fuzzy_trader import config as cfg
from gpu_fuzzy_trader.validation.fold_gates import FoldExposure, scale_count_gate


class TestScaleCountGate:
    @staticmethod
    def exposure(rows: int) -> FoldExposure:
        return FoldExposure(rows=rows, duration_bars=0, per_symbol_rows={})

    def test_scales_count_gates_by_fold_exposure(self):
        reference = self.exposure(100_000)
        assert scale_count_gate(40, self.exposure(100_000), reference, 5) == 40
        assert scale_count_gate(40, self.exposure(50_000), reference, 5) == 20
        assert scale_count_gate(40, self.exposure(25_000), reference, 5) == 10
        assert scale_count_gate(40, self.exposure(5_000), reference, 5) == 5

    def test_scaling_is_monotonic_with_rows(self):
        reference = self.exposure(700_000)
        small = scale_count_gate(45, self.exposure(50_000), reference, 5)
        large = scale_count_gate(45, self.exposure(150_000), reference, 5)
        assert large >= small

    def test_deprecated_config_wrapper_delegates_to_fold_gates(self):
        expected = scale_count_gate(
            45,
            self.exposure(175_000),
            self.exposure(700_000),
            cfg.FOLD_ABSOLUTE_MIN_TRADES,
            rounding="legacy",
        )
        assert cfg.scale_trade_floor(45, 175_000, 700_000) == expected

    def test_effective_wrappers_without_exposure_keep_base_values(self):
        assert cfg.effective_min_trade_support(None) == cfg.MIN_TRADE_SUPPORT
        assert cfg.effective_pool_min_val_trades(None) == max(
            cfg.MIN_TRADE_POOL_FLOOR // 4, 10
        )


def test_deprecated_purged_configuration_is_removed() -> None:
    source = Path(cfg.__file__).read_text(encoding="utf-8")
    for name in (
        "PURGED_WF_",
        "CV_FOLDS_MANIFEST_PATH",
        "_PURGED_WF_REFERENCE_ROWS",
        "split_mode_is_purged_walk_forward",
        "set_purged_wf_reference_rows",
    ):
        assert name not in source
