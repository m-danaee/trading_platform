#!/usr/bin/env python3
"""Benchmark and hardware baseline profiler for T4 / local environments.

Probes host and GPU hardware capabilities, runs synthetic micro-benchmarks
(GPU backtest engine and Data Loader if available), and outputs a structured
JSON report to stdout (with --dry-run) or to outputs/reports/t4_profile.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader._gpu_runtime import (
    detect_hardware_profile,
    resolve_phase2_gpu_batch_size,
)

logger = logging.getLogger("benchmark_t4")


def run_data_loader_benchmark(n_samples: int = 50000) -> dict[str, Any]:
    """Run a quick micro-benchmark on synthetic feature framing and Data_Loader."""
    import tempfile

    import numpy as np
    import pandas as pd

    from gpu_fuzzy_trader.data.loader import Data_Loader

    rng = np.random.default_rng(42)
    symbols = ["SYM_A", "SYM_B", "SYM_C", "SYM_D"]
    rows_per_sym = n_samples // len(symbols)
    dfs = []
    for sym in symbols:
        dt = pd.date_range("2024-01-01", periods=rows_per_sym, freq="5min")
        opens = rng.uniform(100, 110, rows_per_sym)
        spreads = rng.uniform(0.002, 0.02, rows_per_sym)
        lows = opens * (1.0 - spreads)
        highs = opens * (1.0 + spreads)
        dfs.append(pd.DataFrame({
            "datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": sym,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": lows + (highs - lows) * rng.random(rows_per_sym),
            "volume": rng.uniform(1000, 5000, rows_per_sym),
            "feature_1": rng.uniform(-1.0, 1.0, rows_per_sym),
            "feature_2": rng.uniform(-1.0, 1.0, rows_per_sym),
            "feature_3": rng.uniform(-1.0, 1.0, rows_per_sym),
        }))
    synth_df = pd.concat(dfs, ignore_index=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        synth_df.to_csv(f.name, index=False)
        tmp_path = f.name

    try:
        t0 = time.perf_counter()
        loaded = Data_Loader().load_dataset(
            tmp_path,
            require_context=False,
            include_barrier_outcomes=True,
        )
        elapsed = time.perf_counter() - t0
        loader_ms = elapsed * 1000.0
        return {
            "status": "success",
            "rows": len(loaded),
            "loader_ms": round(loader_ms, 2),
            "elapsed_sec": round(elapsed, 4),
            "symbol_is_category": str(loaded["symbol"].dtype) == "category",
            "datetime_is_datetime64": pd.api.types.is_datetime64_any_dtype(loaded["datetime"]),
            "barrier_column_count": sum(
                column.startswith("_barrier_") for column in loaded.columns
            ),
        }
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def run_exact_engine_benchmark(
    batch_size: int | None = None,
    n_samples: int = 1000,
) -> dict[str, Any]:
    """Benchmark the production exact-barrier route through the engine facade."""
    t0 = time.perf_counter()
    try:
        import numpy as np
        import pandas as pd

        from gpu_fuzzy_trader.backtest.barrier import attach_barrier_outcomes
        from gpu_fuzzy_trader.backtest.gpu_engine import GPUBacktestEngine
        from gpu_fuzzy_trader.config import LABEL_COLUMNS
        from gpu_fuzzy_trader.data.labels import compute_labels

        resolved_batch = batch_size or resolve_phase2_gpu_batch_size()
        n = int(n_samples)
        if n <= int(_cfg.TAIL_DROP_ROWS):
            raise ValueError(
                "n_samples must exceed TAIL_DROP_ROWS for exact benchmark data"
            )
        rng = np.random.default_rng(42)
        opens = 100.0 + np.cumsum(rng.normal(0.0, 0.2, n))
        opens = np.maximum(opens, 1.0)
        spreads = rng.uniform(0.002, 0.02, n)
        lows = opens * (1.0 - spreads)
        highs = opens * (1.0 + spreads)
        closes = lows + (highs - lows) * rng.random(n)
        raw = pd.DataFrame({
            "symbol": ["BENCH"] * n,
            "datetime": pd.date_range("2024-01-01", periods=n, freq="15min"),
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": rng.uniform(1000, 5000, n),
            "feat_a": rng.integers(0, 2, n),
            "feat_b": rng.integers(0, 5, n),
            "feat_c": rng.integers(0, 3, n),
        })
        labels = compute_labels(
            raw[["datetime", "symbol", "open", "high", "low", "close", "volume"]]
        )
        df = raw.merge(
            labels,
            on=["datetime", "symbol"],
            how="left",
            sort=False,
            validate="one_to_one",
        )
        exact_pair = (float(_cfg.PHASE2_TP), float(_cfg.PHASE2_SL))
        df = attach_barrier_outcomes(
            df,
            horizon=_cfg.TAIL_DROP_ROWS,
            pairs=[exact_pair],
        )
        df = df.dropna(subset=LABEL_COLUMNS).reset_index(drop=True)
        df["_symbol_bar_index"] = np.arange(len(df), dtype=np.int32)
        feature_modes = {"feat_a": "binary", "feat_b": "positive", "feat_c": "ternary"}
        engine = GPUBacktestEngine(df, feature_modes, "long")
        if not engine._has_exact_barrier_pair(*exact_pair):
            raise RuntimeError("Exact benchmark barrier pair was not attached")

        k = len(feature_modes)
        chrom = np.zeros((resolved_batch, k), dtype=np.int32)

        # Warmup
        engine.simulate_rule_batch(
            chrom,
            tp=_cfg.PHASE2_TP,
            sl=_cfg.PHASE2_SL,
            capital_pct=_cfg.PHASE2_CAPITAL_PCT,
            generation=1,
            is_last_gen=False,
        )

        t_start = time.perf_counter()
        engine.simulate_rule_batch(
            chrom,
            tp=_cfg.PHASE2_TP,
            sl=_cfg.PHASE2_SL,
            capital_pct=_cfg.PHASE2_CAPITAL_PCT,
            generation=1,
            is_last_gen=False,
        )
        t_exec = time.perf_counter() - t_start

        return {
            "status": "success",
            "batch_size": resolved_batch,
            "rows": len(df),
            "execution_route": "cpu_exact",
            "exact_barrier_pair": list(exact_pair),
            "exec_time_sec": round(t_exec, 4),
            "total_time_sec": round(time.perf_counter() - t0, 4),
        }
    except Exception as exc:
        return {
            "status": "skipped",
            "reason": str(exc),
            "total_time_sec": round(time.perf_counter() - t0, 4),
        }


def run_jax_engine_benchmark(
    batch_size: int | None = None,
    n_samples: int = 1000,
) -> dict[str, Any]:
    """Backward-compatible alias for the exact production-route benchmark."""
    return run_exact_engine_benchmark(
        batch_size=batch_size,
        n_samples=n_samples,
    )


def run_evolution_benchmark(
    generations: int = 3,
    pop_size: int = 32,
    n_samples: int = 2000,
) -> dict[str, Any]:
    """Run synthetic NSGA-III / NSGA-II evolution micro-benchmark."""
    import numpy as np
    import pandas as pd
    import psutil

    from gpu_fuzzy_trader.backtest.cpu_engine import CPUBacktestEngine
    from gpu_fuzzy_trader.evolution.evox_runner import (
        N_OBJ,
        Phase2EvolutionState,
        _evaluate_population_indices,
        trim_evolution_state_memory,
    )
    from gpu_fuzzy_trader.phases.phase2_rule_pool import (
        _get_dont_cares,
        _init_population,
    )

    t0 = time.perf_counter()
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "symbol": ["BENCH"] * n_samples,
        "datetime": pd.date_range("2024-01-01", periods=n_samples, freq="5min"),
        "_symbol_bar_index": np.arange(n_samples),
        "label_open_next": rng.uniform(90, 110, n_samples),
        "label_max_288": rng.uniform(92, 115, n_samples),
        "label_min_288": rng.uniform(85, 98, n_samples),
        "label_close_288": rng.uniform(90, 110, n_samples),
        "label_max_before_min": rng.integers(0, 2, n_samples),
        "feat_a": rng.integers(0, 2, n_samples),
        "feat_b": rng.integers(0, 5, n_samples),
        "feat_c": rng.integers(0, 3, n_samples),
    })
    feature_modes = {"feat_a": "binary", "feat_b": "positive", "feat_c": "ternary"}
    feature_infos = [{"name": k, "mode": v} for k, v in feature_modes.items()]
    dont_cares = _get_dont_cares(feature_infos)
    engine = CPUBacktestEngine(df, feature_modes, "long")

    t_warm_start = time.perf_counter()
    # Warmup evaluation
    test_pop = _init_population(pop_size, feature_infos, rng=rng)
    test_obj = np.full((pop_size, N_OBJ), np.inf)
    test_metrics = [{} for _ in range(pop_size)]
    global_cache: dict[tuple[int, ...], dict] = {}
    _evaluate_population_indices(
        test_pop,
        list(range(pop_size)),
        dont_cares,
        engine,
        pareto_archive=[],
        objectives=test_obj,
        metrics_cache=test_metrics,
        global_metrics_cache=global_cache,
        generation=1,
    )
    warmup_ms = (time.perf_counter() - t_warm_start) * 1000.0

    gen_times = []
    total_cache_hits = 0
    total_evals = 0

    state = Phase2EvolutionState(
        population=test_pop,
        objectives=test_obj,
        metrics_cache=test_metrics,
        pareto_archive=[],
        hall_of_fame={},
        deployable_archive={},
        global_metrics_cache=global_cache,
    )

    for g in range(1, generations + 1):
        tg0 = time.perf_counter()
        # Mutate half pop to simulate generation churn with new and recurring individuals
        for idx in range(pop_size // 4, pop_size):
            state.objectives[idx] = np.inf
            if idx >= pop_size // 2:
                # Randomize second half so they are fresh cache misses
                state.population[idx] = _init_population(1, feature_infos, rng=rng)[0]
        eval_stats = _evaluate_population_indices(
            state.population,
            list(range(pop_size)),
            dont_cares,
            engine,
            pareto_archive=state.pareto_archive,
            objectives=state.objectives,
            metrics_cache=state.metrics_cache,
            global_metrics_cache=state.global_metrics_cache,
            generation=g,
        )
        total_cache_hits += eval_stats.get("cache_hits", 0)
        total_evals += eval_stats.get("pending", 0)
        gen_times.append((time.perf_counter() - tg0) * 1000.0)

    trim_evolution_state_memory(state, pop_size=pop_size)
    rss_mib = psutil.Process().memory_info().rss / (1024 * 1024)

    cache_hit_rate = (total_cache_hits / total_evals) if total_evals > 0 else 0.0
    gen_avg_ms = float(np.mean(gen_times)) if gen_times else 0.0

    return {
        "status": "success",
        "generations": generations,
        "pop_size": pop_size,
        "dont_care_codes": dont_cares.tolist(),
        "cache_hit_rate": round(cache_hit_rate, 4),
        "warmup_ms": round(warmup_ms, 2),
        "gen_avg_ms": round(gen_avg_ms, 2),
        "peak_rss_mib": round(rss_mib, 2),
        "total_time_sec": round(time.perf_counter() - t0, 4),
    }


def collect_profile_report(
    skip_bench: bool = False,
    component: str = "all",
    rows: int = 50000,
    generations: int = 3,
    pop: int = 32,
) -> dict[str, Any]:
    """Collect complete hardware snapshot and benchmark metrics."""
    hw = detect_hardware_profile()
    batch_resolved = resolve_phase2_gpu_batch_size()

    report: dict[str, Any] = {
        "timestamp": time.time(),
        "hardware": {
            "gpu_name": hw.gpu_name,
            "vram_gb": hw.vram_gb,
            "ram_gb": hw.ram_gb,
            "cpu_count": hw.cpu_count,
            "jax_backend": hw.jax_backend,
            "devices": hw.devices,
            "is_t4": hw.is_t4,
        },
        "resolved_config": {
            "batch_size": batch_resolved,
            "scan_unroll": _cfg.PHASE2_SCAN_UNROLL,
            "cpu_route_large_data": getattr(_cfg, "PHASE2_GPU_CPU_ROUTE_LARGE_DATA", True),
            "workers": getattr(_cfg, "BACKTEST_BATCH_WORKERS", 1),
            "global_cache_max_size": getattr(_cfg, "PHASE2_EVAL_GLOBAL_CACHE_MAX_SIZE", 600),
            "cpu_batch_size": getattr(_cfg, "PHASE2_CPU_BATCH_SIZE", 16),
        },
        "benchmarks": {},
    }

    if not skip_bench:
        if component in ("all", "loader"):
            report["benchmarks"]["data_loader"] = run_data_loader_benchmark(n_samples=rows)
        if component in ("all", "gpu"):
            report["benchmarks"]["gpu_engine"] = run_jax_engine_benchmark(batch_resolved)
        if component in ("all", "evolution"):
            report["benchmarks"]["evolution"] = run_evolution_benchmark(
                generations=generations,
                pop_size=pop,
            )

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="T4 Hardware Profiler & Benchmark")
    parser.add_argument("--dry-run", action="store_true", help="Print JSON to stdout without writing file")
    parser.add_argument("--output", type=str, default="", help="Path to write JSON profile")
    parser.add_argument("--skip-micro-bench", action="store_true", help="Skip running synthetic JAX/Loader micro-bench")
    parser.add_argument(
        "--component",
        type=str,
        default="all",
        choices=["all", "loader", "gpu", "evolution"],
        help="Micro-bench component to run",
    )
    parser.add_argument("--rows", type=int, default=50000, help="Number of rows for data loader benchmark")
    parser.add_argument("--generations", type=int, default=3, help="Generations for evolution benchmark")
    parser.add_argument("--pop", type=int, default=32, help="Population size for evolution benchmark")
    args = parser.parse_args()

    report = collect_profile_report(
        skip_bench=args.skip_micro_bench,
        component=args.component,
        rows=args.rows,
        generations=args.generations,
        pop=args.pop,
    )
    rendered = json.dumps(report, indent=2)

    if args.dry_run or not args.output:
        print(rendered)

    if not args.dry_run and args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered, encoding="utf-8")
        print(f"Profile saved to: {out_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
