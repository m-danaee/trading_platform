"""Phase 2 CPU vs GPU runtime benchmark (non-gating, RUN_BENCHMARKS=1).

Times ``simulate_rule_batch`` on the CPU engine and the physical JAX GPU
engine using the *same* deterministic dataframe, feature modes, int32
chromosomes, and simulation constants. Reports:

  - engine construction (separately for each backend),
  - cold first-call time (the GPU call includes JAX/XLA compilation),
  - synchronized steady-state call time (JAX async execution is accounted
    for via ``jax.block_until_ready``; ``GPUBacktestEngine.simulate_rule_batch``
    also blocks internally before returning host arrays).

The benchmark only measures the Phase 2 *batch-ranking* path. Exact rule-set /
RB / OOS evaluation remains CPU-backed (``CPUBacktestEngine`` /
``evaluator_v5.ipynb``); GPU results are an approximate ranking model used
during evolution. No assertion requires the GPU to win — the outcome is
informational and host/hardware dependent.

Skips unless a physical JAX GPU backend is active.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.backtest.cpu_engine import CPUBacktestEngine
from gpu_fuzzy_trader.backtest.jax_compat import get_gpu_backtest_engine_class
from gpu_fuzzy_trader._gpu_runtime import resolve_phase2_gpu_batch_size

# ---------------------------------------------------------------------------
# Benchmark workload (kept small so it runs under PYTEST_LOW_MEMORY=1)
# ---------------------------------------------------------------------------
_N_ROWS = 2000          # bars in the shared deterministic dataframe
_BATCH = 64             # chromosomes evaluated per simulate_rule_batch call
_WARMUP = 2             # untimed warm-up calls before steady-state timing
_REPEATS = 5            # timed calls averaged for steady-state


def _make_deterministic_df(n: int = _N_ROWS) -> pd.DataFrame:
    """Deterministic synthetic Phase 2 dataframe shared by both engines."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "symbol": ["SYM"] * n,
        "datetime": pd.date_range("2024-01-01", periods=n, freq="5min"),
        "_symbol_bar_index": np.arange(n),
        "label_open_next": rng.uniform(90, 110, n),
        "label_max_288": rng.uniform(92, 115, n),
        "label_min_288": rng.uniform(85, 98, n),
        "label_close_288": rng.uniform(90, 110, n),
        "label_max_before_min": rng.integers(0, 2, n),
        "feat_a": rng.integers(0, 2, n),   # binary  -> 2 classes (dont-care=2)
        "feat_b": rng.integers(0, 5, n),   # positive -> 5 classes (dont-care=5)
        "feat_c": rng.integers(0, 3, n),   # ternary -> 3 classes (dont-care=3)
    })


def _make_chromosomes(batch: int = _BATCH) -> np.ndarray:
    """Deterministic int32 dense chromosomes: (B, K), genes in [0, dont_care]."""
    num_classes = {"feat_a": 2, "feat_b": 5, "feat_c": 3}
    rng = np.random.default_rng(1)
    chroms = np.empty((batch, len(num_classes)), dtype=np.int32)
    for j, n_classes in enumerate(num_classes.values()):
        # Value == n_classes acts as the don't-care gene (matches any value).
        chroms[:, j] = rng.integers(0, n_classes + 1, size=batch)
    return chroms


def _physical_gpu_available() -> bool:
    """True when a physical JAX GPU backend is active."""
    if get_gpu_backtest_engine_class() is None:
        return False
    try:
        import jax

        if jax.default_backend() != "gpu":
            return False
        devices = jax.devices()
        return bool(devices) and any(
            getattr(d, "platform", "") == "gpu" for d in devices
        )
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Timing helpers (honest JAX async handling)
# ---------------------------------------------------------------------------

def _block_ready(result):
    """Force any pending JAX async dispatch to finish (no-op on host values)."""
    try:
        import jax

        return jax.block_until_ready(result)
    except Exception:
        return result


def _time_once(fn) -> tuple[float, object]:
    """Time one call (cold first-call), synchronizing before returning."""
    start = time.perf_counter()
    result = _block_ready(fn())
    return time.perf_counter() - start, result


def _time_steady(fn, warmup: int = _WARMUP, repeats: int = _REPEATS) -> float:
    """Time synchronized steady-state calls; returns per-call average seconds."""
    for _ in range(warmup):
        _block_ready(fn())
    start = time.perf_counter()
    for _ in range(repeats):
        _block_ready(fn())
    return (time.perf_counter() - start) / repeats


# ---------------------------------------------------------------------------
# Benchmark test
# ---------------------------------------------------------------------------

@pytest.mark.benchmark
@pytest.mark.uses_jax
def test_phase2_cpu_gpu_runtime_benchmark(
    capsys, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Time identical Phase 2 chromosome batches on CPU vs physical GPU."""
    if not _physical_gpu_available():
        pytest.skip("JAX physical GPU backend unavailable")

    # Honest GPU-only timing: disable the optional CPU per-symbol enrichment
    # pass so it cannot add CPU work to every GPU call we are timing.
    monkeypatch.setattr(_cfg, "PHASE2_GPU_ENRICH_SYMBOL_METRICS", False)

    # Pin the GPU vmap chunk to exactly our batch so the GPU call processes
    # the same B chromosomes as the CPU call (no pad-to-chunk skew). The batch
    # size resolver is lru-cached per process, so clear it before/after use.
    monkeypatch.setenv("PHASE2_GPU_BATCH_SIZE", str(_BATCH))
    resolve_phase2_gpu_batch_size.cache_clear()
    try:
        resolved_chunk = resolve_phase2_gpu_batch_size()  # untimed probe
    finally:
        resolve_phase2_gpu_batch_size.cache_clear()
    assert resolved_chunk == _BATCH, (
        f"GPU chunk {resolved_chunk} != benchmark batch {_BATCH}; "
        "benchmark comparison would be skewed by padding."
    )

    df = _make_deterministic_df()
    feature_modes = {"feat_a": "binary", "feat_b": "positive", "feat_c": "ternary"}
    chroms = _make_chromosomes()
    # Same simulation constants for both engines (config defaults).
    tp = float(_cfg.PHASE2_TP)
    sl = float(_cfg.PHASE2_SL)
    capital_pct = float(_cfg.PHASE2_CAPITAL_PCT)

    GPUBacktestEngineCls = get_gpu_backtest_engine_class()

    # ---- Construction (measured separately per backend) ----
    t_cpu_construct, cpu_engine = _time_once(
        lambda: CPUBacktestEngine(df, feature_modes, "long")
    )
    t_gpu_construct, gpu_engine = _time_once(
        lambda: GPUBacktestEngineCls(df, feature_modes, "long")
    )

    def cpu_call():
        return cpu_engine.simulate_rule_batch(
            chroms, tp=tp, sl=sl, capital_pct=capital_pct
        )

    def gpu_call():
        return gpu_engine.simulate_rule_batch(
            chroms, tp=tp, sl=sl, capital_pct=capital_pct
        )

    # ---- Cold first-call (GPU includes JAX/XLA compilation) ----
    t_gpu_cold, gpu_results = _time_once(gpu_call)
    t_cpu_cold, cpu_results = _time_once(cpu_call)

    # ---- Synchronized steady-state calls ----
    t_cpu_steady = _time_steady(cpu_call)
    t_gpu_steady = _time_steady(gpu_call)

    # ---- Basic output shape / finite-metric sanity (no win assertions) ----
    for label, results in (("GPU", gpu_results), ("CPU", cpu_results)):
        assert isinstance(results, list), f"{label} results not a list"
        assert len(results) == len(chroms), (
            f"{label} returned {len(results)} rows for {len(chroms)} chromosomes"
        )
        numeric_keys = (
            "total_return_pct", "sortino_ratio", "max_drawdown_pct",
            "win_rate", "profit_factor", "executed_trades",
            "final_equity", "raw_signal_count",
        )
        for r in results:
            assert isinstance(r, dict), f"{label} result row not a dict"
            for key in numeric_keys:
                assert key in r, f"{label} result missing {key!r}"
            for key in (
                "total_return_pct", "sortino_ratio", "max_drawdown_pct",
                "win_rate", "profit_factor", "final_equity",
            ):
                assert np.isfinite(float(r[key])), (
                    f"{label} {key} is not finite: {r[key]!r}"
                )
            assert float(r["executed_trades"]) >= 0.0
            assert float(r["raw_signal_count"]) >= 0.0

    # ---- Clearly labeled results ----
    cpu_signals = sum(float(r["raw_signal_count"]) for r in cpu_results)
    gpu_signals = sum(float(r["raw_signal_count"]) for r in gpu_results)
    faster = "CPU" if t_cpu_steady < t_gpu_steady else "GPU"
    slower_time, faster_time = (
        (t_gpu_steady, t_cpu_steady) if faster == "CPU"
        else (t_cpu_steady, t_gpu_steady)
    )
    speedup = slower_time / faster_time if faster_time > 0.0 else float("nan")

    print(
        f"\nPhase 2 CPU vs GPU batch-ranking runtime "
        f"(N={len(df)}, B={_BATCH}, K={len(feature_modes)})"
    )
    print(
        f"  construction  : CPU={t_cpu_construct * 1000:8.1f} ms   "
        f"GPU={t_gpu_construct * 1000:8.1f} ms"
    )
    print(
        f"  cold first-call: CPU={t_cpu_cold * 1000:8.1f} ms   "
        f"GPU={t_gpu_cold * 1000:8.1f} ms   (GPU includes XLA compile)"
    )
    print(
        f"  steady-state  : CPU={t_cpu_steady * 1000:8.1f} ms   "
        f"GPU={t_gpu_steady * 1000:8.1f} ms   "
        f"(sync, {_REPEATS} reps, gpu_chunk={resolved_chunk})"
    )
    print(
        f"  raw signals   : CPU={cpu_signals:.0f}   GPU={gpu_signals:.0f}"
    )
    print(
        f"  -> lower steady-state runtime backend: {faster} "
        f"({faster_time * 1000:.1f} ms vs {slower_time * 1000:.1f} ms, "
        f"{speedup:.2f}x)"
    )
    print(
        "  Note: this measures Phase 2 batch ranking only; exact rule-set / "
        "RB / OOS evaluation remains CPU-backed."
    )
