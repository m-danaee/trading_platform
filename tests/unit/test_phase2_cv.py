"""Unit tests for Phase 2 CV configuration and admission strictness."""

from __future__ import annotations

import numpy as np
import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.config import _cv_pool_min_folds_pass
from gpu_fuzzy_trader.phases.phase2_cv import evaluate_purged_cv_pool_admission
from gpu_fuzzy_trader.phases.phase2_cv import (
    PurgedCVTrainEngine,
    PurgedCVValEngine,
)


def _metrics(ret: float, pf: float, trades: int) -> dict:
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


class TestCvPoolMinFoldsPass:
    def test_even_k_requires_true_majority(self) -> None:
        assert _cv_pool_min_folds_pass(2) == 2
        assert _cv_pool_min_folds_pass(4) == 3

    def test_odd_k_requires_ceil_half(self) -> None:
        assert _cv_pool_min_folds_pass(3) == 2
        assert _cv_pool_min_folds_pass(5) == 3

    def test_config_default_for_two_folds(self) -> None:
        if _cfg.CV_N_FOLDS == 2:
            assert _cfg.PHASE2_CV_POOL_MIN_FOLDS_PASS == 2


class TestTwoFoldCvAdmission:
    def test_rejects_when_only_one_of_two_folds_passes(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(_cfg, "SPLIT_MODE", "purged_rolling_cv")
        monkeypatch.setattr(_cfg, "PHASE2_CV_POOL_MIN_FOLDS_PASS", 2)
        monkeypatch.setattr(_cfg, "PHASE2_CV_MERGED_GATE_HARD", True)
        good = _metrics(2.0, 1.2, 30)
        bad = _metrics(-0.5, 0.8, 30)
        val_good = _metrics(1.0, 1.1, 15)
        val_bad = _metrics(-1.0, 0.8, 15)

        train_cv = PurgedCVTrainEngine([
            _MockFoldEngine(good),
            _MockFoldEngine(bad),
        ])
        val_cv = PurgedCVValEngine([
            _MockFoldEngine(val_good),
            _MockFoldEngine(val_bad),
        ])
        chrom = np.array([0, 1], dtype=np.int64)
        ok, _, _, folds_passing = evaluate_purged_cv_pool_admission(
            train_cv, val_cv, chrom,
        )
        assert folds_passing == 1
        assert ok is False

    def test_admits_when_both_folds_pass(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(_cfg, "SPLIT_MODE", "purged_rolling_cv")
        monkeypatch.setattr(_cfg, "PHASE2_CV_POOL_MIN_FOLDS_PASS", 2)
        monkeypatch.setattr(_cfg, "PHASE2_CV_MERGED_GATE_HARD", True)
        good = _metrics(2.0, 1.2, 30)
        val_good = _metrics(1.0, 1.1, 15)

        train_cv = PurgedCVTrainEngine([
            _MockFoldEngine(good),
            _MockFoldEngine(good),
        ])
        val_cv = PurgedCVValEngine([
            _MockFoldEngine(val_good),
            _MockFoldEngine(val_good),
        ])
        chrom = np.array([0, 1], dtype=np.int64)
        ok, _, _, folds_passing = evaluate_purged_cv_pool_admission(
            train_cv, val_cv, chrom,
        )
        assert folds_passing == 2
        assert ok is True
