"""Phase 4 robust scoring and val-train gap gate."""

from __future__ import annotations

from gpu_fuzzy_trader.phases.phase4_wf_optimizer import _score_metrics


def _metrics(ret: float, dd: float = 3.0, pf: float = 1.3) -> dict:
    return {
        "total_return_pct": ret,
        "max_drawdown_pct": dd,
        "profit_factor": pf,
        "win_rate": 50.0,
    }


def test_robust_score_penalizes_validation_spike(monkeypatch) -> None:
    from gpu_fuzzy_trader import config as _cfg

    monkeypatch.setattr(_cfg, "PHASE4_MAX_VAL_TRAIN_GAP_PCT", 5.0)
    train = _metrics(8.0)
    val_ok = _metrics(9.0)
    val_spike = _metrics(25.0, dd=2.0, pf=1.8)

    assert _score_metrics(train, val_ok) > _score_metrics(train, val_spike)
