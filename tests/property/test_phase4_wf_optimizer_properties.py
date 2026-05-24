"""
Property-based tests for gpu_fuzzy_trader.phases.phase4_wf_optimizer

Property: Quantized parameter grid
  For any trial-suggested TP/SL/capital values within config bounds,
  values lie on the configured step grid.
"""

from __future__ import annotations
from gpu_fuzzy_trader.phases.phase4_wf_optimizer import _create_sampler

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from gpu_fuzzy_trader import config as _cfg

pytest.importorskip("optuna")


def _on_grid(value: float, low: float, step: float) -> bool:
    if step <= 0:
        return False
    n = round((value - low) / step)
    expected = low + n * step
    return math.isclose(value, expected, rel_tol=0, abs_tol=1e-9)


@settings(max_examples=30, deadline=None)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_suggested_floats_on_quantization_grid(seed: int):
    import optuna

    sampler = _create_sampler(seed)

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        tp = trial.suggest_float(
            "tp",
            _cfg.PHASE4_TP_MIN,
            _cfg.PHASE4_TP_MAX,
            step=_cfg.PHASE4_TP_STEP,
        )
        sl = trial.suggest_float(
            "sl",
            _cfg.PHASE4_SL_MIN,
            _cfg.PHASE4_SL_MAX,
            step=_cfg.PHASE4_SL_STEP,
        )
        cap = trial.suggest_float(
            "cap",
            _cfg.PHASE4_CAPITAL_PCT_MIN,
            _cfg.PHASE4_CAPITAL_PCT_MAX,
            step=_cfg.PHASE4_CAPITAL_STEP,
        )
        assert _on_grid(tp, _cfg.PHASE4_TP_MIN, _cfg.PHASE4_TP_STEP)
        assert _on_grid(sl, _cfg.PHASE4_SL_MIN, _cfg.PHASE4_SL_STEP)
        assert _on_grid(cap, _cfg.PHASE4_CAPITAL_PCT_MIN,
                        _cfg.PHASE4_CAPITAL_STEP)
        return tp, -sl

    study = optuna.create_study(
        directions=["maximize", "minimize"],
        sampler=sampler,
    )
    study.optimize(objective, n_trials=5, show_progress_bar=False)
