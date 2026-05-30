"""Unit tests for purged-CV Phase 2 pool admission."""

from __future__ import annotations

import numpy as np
import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.phases.phase2_cv import (
    PurgedCVTrainEngine,
    PurgedCVValEngine,
    evaluate_purged_cv_pool_admission,
)
from gpu_fuzzy_trader.phases.phase2_support import (
    passes_pool_admission_cv_fold,
    passes_pool_entry_admission,
)


def _metrics(
    ret: float,
    pf: float,
    trades: int,
) -> dict:
    return {
        "total_return_pct": ret,
        "profit_factor": pf,
        "executed_trades": trades,
        "sortino_ratio": ret,
        "max_drawdown_pct": 5.0,
        "win_rate": 0.5,
    }


class _MockFoldEngine:
    def __init__(self, metrics: dict) -> None:
        self._metrics = metrics

    def simulate_rule_batch(self, **kwargs: object) -> list[dict]:
        return [self._metrics]


class TestCvFoldAdmission:
    def test_cv_fold_uses_lower_trade_floor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_cfg, "SPLIT_MODE", "purged_rolling_cv")
        train = _metrics(1.0, 1.1, _cfg.PHASE2_CV_MIN_TRADE_POOL_FLOOR)
        val = _metrics(0.5, 1.05, _cfg.PHASE2_CV_MIN_VAL_TRADES)
        assert passes_pool_admission_cv_fold(train, val) is True

    def test_cv_fold_rejects_negative_return(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_cfg, "SPLIT_MODE", "purged_rolling_cv")
        train = _metrics(-0.1, 1.2, 30)
        val = _metrics(1.0, 1.1, 20)
        assert passes_pool_admission_cv_fold(train, val) is False


class TestEvaluatePurgedCvPoolAdmission:
    def test_passes_when_two_of_three_folds_ok(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(_cfg, "SPLIT_MODE", "purged_rolling_cv")
        monkeypatch.setattr(_cfg, "PHASE2_CV_POOL_MIN_FOLDS_PASS", 2)
        good = _metrics(2.0, 1.2, 30)
        bad = _metrics(-5.0, 0.8, 30)
        val_good = _metrics(1.0, 1.1, 15)

        train_cv = PurgedCVTrainEngine([
            _MockFoldEngine(good),
            _MockFoldEngine(good),
            _MockFoldEngine(bad),
        ])
        val_cv = PurgedCVValEngine([
            _MockFoldEngine(val_good),
            _MockFoldEngine(val_good),
            _MockFoldEngine(val_good),
        ])
        chrom = np.array([0, 1, 2], dtype=np.int64)
        ok, train_m, val_m, folds_passing = evaluate_purged_cv_pool_admission(
            train_cv, val_cv, chrom,
        )
        assert ok is True
        assert folds_passing == 2
        assert float(train_m["total_return_pct"]) == pytest.approx(-5.0)
        assert val_m is not None

    def test_fails_when_only_one_fold_ok(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(_cfg, "SPLIT_MODE", "purged_rolling_cv")
        monkeypatch.setattr(_cfg, "PHASE2_CV_POOL_MIN_FOLDS_PASS", 2)
        good = _metrics(2.0, 1.2, 30)
        bad = _metrics(-5.0, 0.8, 30)
        val_good = _metrics(1.0, 1.1, 15)

        train_cv = PurgedCVTrainEngine([
            _MockFoldEngine(good),
            _MockFoldEngine(bad),
            _MockFoldEngine(bad),
        ])
        val_cv = PurgedCVValEngine([
            _MockFoldEngine(val_good),
            _MockFoldEngine(val_good),
            _MockFoldEngine(val_good),
        ])
        chrom = np.array([0, 1], dtype=np.int64)
        ok, _, _, folds_passing = evaluate_purged_cv_pool_admission(
            train_cv, val_cv, chrom,
        )
        assert ok is False
        assert folds_passing == 1


class TestPoolEntryAdmission:
    def test_cv_metadata_passes_despite_negative_merged_return(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(_cfg, "SPLIT_MODE", "purged_rolling_cv")
        monkeypatch.setattr(_cfg, "PHASE2_CV_POOL_MIN_FOLDS_PASS", 2)
        entry = {
            "cv_folds_passing": 2,
            "cv_folds_total": 3,
            "objectives": {
                "total_return_pct": -5.0,
                "profit_factor": 0.8,
            },
            "val_objectives": {
                "total_return_pct": 1.0,
                "profit_factor": 1.1,
            },
            "executed_trades": 30,
            "val_executed_trades": 15,
        }
        assert passes_pool_entry_admission(entry) is True

    def test_legacy_entry_uses_merged_objectives(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(_cfg, "SPLIT_MODE", "purged_rolling_cv")
        entry = {
            "objectives": {
                "total_return_pct": -5.0,
                "profit_factor": 0.8,
            },
            "val_objectives": {
                "total_return_pct": 1.0,
                "profit_factor": 1.1,
            },
            "executed_trades": 30,
            "val_executed_trades": 15,
        }
        assert passes_pool_entry_admission(entry) is False
