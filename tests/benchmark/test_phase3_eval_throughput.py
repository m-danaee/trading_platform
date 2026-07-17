"""Phase 3 evaluation throughput benchmark (non-gating)."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.backtest.cpu_engine import CPUBacktestEngine
from gpu_fuzzy_trader.phases.phase3_cache import build_phase3_eval_cache
from gpu_fuzzy_trader.phases.phase3_rule_set import (
    _rule_set_to_engine_format,
    _simulate_team,
)


def _make_df(n_rows: int = 400, n_feats: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    sym = "SYM_A"
    open_next = rng.uniform(100, 200, size=n_rows)
    data = {
        "symbol": sym,
        "datetime": pd.date_range("2024-01-01", periods=n_rows, freq="5min"),
        "_symbol_bar_index": np.arange(n_rows),
        "label_open_next": open_next,
        "label_close_288": open_next * 1.02,
        "label_min_288": open_next * 0.98,
        "label_max_288": open_next * 1.05,
        "label_max_before_min": rng.integers(0, 2, size=n_rows).astype(float),
    }
    for i in range(n_feats):
        data[f"feat_{i}"] = rng.uniform(0, 1, size=n_rows)
    return pd.DataFrame(data)


def _make_rule_pool(n: int, n_feats: int) -> list[dict]:
    return [
        {
            "conditions": [f"[feat_{i % n_feats}] IS Very High"],
            "tp": _cfg.PHASE2_TP,
            "sl": _cfg.PHASE2_SL,
            "capital_pct": _cfg.PHASE2_CAPITAL_PCT,
        }
        for i in range(n)
    ]


@pytest.mark.benchmark
def test_phase3_cached_eval_throughput(capsys) -> None:
    """Log evals/sec for cached vs uncached team evaluation."""
    df = _make_df()
    rule_pool = _make_rule_pool(12, 6)
    engine = CPUBacktestEngine(df, {}, "long")
    cache = build_phase3_eval_cache(rule_pool, df, df, engine)
    teams = [
        _rule_set_to_engine_format([
            rule_pool[i], rule_pool[(i + 1) % len(rule_pool)]])
        for i in range(min(20, len(rule_pool)))
    ]

    t0 = time.perf_counter()
    for team in teams:
        engine.simulate_rule_set(team)
    baseline = len(teams) / (time.perf_counter() - t0)

    t0 = time.perf_counter()
    for team in teams:
        _simulate_team(team, engine, engine, cache)
    cached = len(teams) / (time.perf_counter() - t0)

    print(
        f"\nPhase3 throughput: baseline={baseline:.1f} evals/s, "
        f"cached={cached:.1f} evals/s, speedup={cached/max(baseline, 1e-9):.2f}x"
    )
    assert cached > 0
